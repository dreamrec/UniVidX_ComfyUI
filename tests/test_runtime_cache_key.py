"""Tests for vram_buffer wiring + cache-key correctness in src.runtime.load_model.

Covers the v0.3 Tier-A correctness fix: prior versions wired
`vram_buffer_gb` into `model.pipe.enable_vram_management(...)`, but the
real entrypoint is `model.enable_vram_management(...)` on UniVidX's
pipeline class (vendor/UniVidX/src/pipelines/univid_*.py). The
mis-targeted call silently no-op'd because `hasattr(model.pipe, "...")`
returned False.

Asserts:
- load_model() calls model.enable_vram_management(vram_buffer=...) with
  the value the caller passed, and on `model` itself (not model.pipe).
- A class lacking the method gets a logged warning so the no-op is
  visible, not silent.
- Two loads with the same vram_buffer share a cache slot.
- Two loads with different vram_buffer get distinct cache slots — the
  knob actually controls what was passed to UniVidX.
"""
from __future__ import annotations

import logging
import os
import sys
import types
from contextlib import nullcontext

import pytest

HERE = __file__
_REPO = os.path.abspath(os.path.join(os.path.dirname(HERE), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class _FakeUniVidModel:
    """Mimics the surface area load_model() touches: a constructor that
    accepts the kwargs UniVidX uses, a `train()` method, and the
    enable_vram_management() entrypoint we're trying to reach."""

    instance_count = 0

    def __init__(self, **kwargs):
        type(self).instance_count += 1
        self.init_kwargs = kwargs
        self.evm_calls: list[float] = []

    def train(self, mode):
        return self

    def enable_vram_management(self, vram_buffer=0.5, **_kw):
        self.evm_calls.append(float(vram_buffer))


class _FakeNoEVMModel:
    """Mimics a hypothetical upstream rev where enable_vram_management
    was renamed/removed — the runtime should not crash, but should log."""

    instance_count = 0

    def __init__(self, **kwargs):
        type(self).instance_count += 1

    def train(self, mode):
        return self
    # deliberately no enable_vram_management


@pytest.fixture
def stub_runtime(monkeypatch):
    """Stub every external dependency load_model touches: filesystem
    (resolve_paths), symlink setup (initialize), the chdir context
    (unividx_cwd), the mmgp safetensors patch, the SDPA un-polluter, and
    UniVidX's MODEL_REGISTRY. Yields the fake model class so tests can
    inspect it."""
    from src import runtime

    runtime._MODEL_CACHE.clear()
    _FakeUniVidModel.instance_count = 0
    _FakeNoEVMModel.instance_count = 0

    fake_paths = {
        "univid_intrinsic_ckpt": "/fake/intrinsic.safetensors",
        "univid_alpha_ckpt": "/fake/alpha.safetensors",
        "wan_t5": "/fake/t5",
        "wan_vae": "/fake/vae",
    }
    monkeypatch.setattr(runtime, "resolve_paths", lambda root: fake_paths)
    monkeypatch.setattr(runtime, "initialize", lambda: None)
    monkeypatch.setattr(runtime, "unividx_cwd", lambda: nullcontext())
    monkeypatch.setattr(runtime, "_patch_unividx_load_file_to_readonly",
                        lambda: None)
    monkeypatch.setattr(runtime, "_restore_native_sdpa_if_polluted",
                        lambda: False)

    fake_registry = types.ModuleType("scripts.registry")
    fake_registry.MODEL_REGISTRY = {
        "UniVidIntrinsic": _FakeUniVidModel,
        "UniVidAlpha": _FakeUniVidModel,
    }
    fake_scripts = types.ModuleType("scripts")
    fake_scripts.registry = fake_registry
    monkeypatch.setitem(sys.modules, "scripts", fake_scripts)
    monkeypatch.setitem(sys.modules, "scripts.registry", fake_registry)

    yield fake_registry
    runtime._MODEL_CACHE.clear()


def test_load_model_invokes_enable_vram_management_on_model(stub_runtime):
    """The runtime must call enable_vram_management on `model` itself
    (not model.pipe — that's DiffSynth's WanVideoPipeline which has no
    such method) and pass the caller-supplied buffer value through."""
    from src.runtime import load_model

    model = load_model("intrinsic", vram_buffer=7.5)
    assert model.evm_calls == [7.5], (
        f"Expected enable_vram_management called once with 7.5, "
        f"got {model.evm_calls!r}"
    )


def test_load_model_warns_when_model_lacks_enable_vram_management(
        stub_runtime, caplog):
    """If a future upstream rev removes/renames enable_vram_management,
    skipping silently would re-create the same bug we just fixed. The
    runtime should log a warning so the no-op is visible in the user's
    ComfyUI console."""
    from src.runtime import load_model

    stub_runtime.MODEL_REGISTRY["UniVidIntrinsic"] = _FakeNoEVMModel

    with caplog.at_level(logging.WARNING, logger="unividx"):
        load_model("intrinsic", vram_buffer=4.0)

    matching = [r for r in caplog.records
                if "enable_vram_management" in r.message]
    assert matching, (
        "Expected a WARNING mentioning enable_vram_management when the "
        "model class lacks the method; got log records: "
        f"{[r.message for r in caplog.records]}"
    )


def test_load_model_cache_hit_same_vram_buffer(stub_runtime):
    """Identical vram_buffer values must share a cache slot — otherwise
    every multi-node graph would reload the 28 GB DiT."""
    from src.runtime import load_model

    a = load_model("intrinsic", vram_buffer=4.0)
    b = load_model("intrinsic", vram_buffer=4.0)
    assert a is b
    assert _FakeUniVidModel.instance_count == 1


def test_load_model_cache_miss_different_vram_buffer(stub_runtime):
    """Distinct vram_buffer values must produce distinct cache entries.
    Otherwise two loader nodes with different vram_buffer_gb settings
    would silently share whichever model loaded first, contradicting
    the UI."""
    from src.runtime import load_model

    a = load_model("intrinsic", vram_buffer=4.0)
    b = load_model("intrinsic", vram_buffer=12.0)
    assert a is not b
    assert _FakeUniVidModel.instance_count == 2
    assert a.evm_calls == [4.0]
    assert b.evm_calls == [12.0]
