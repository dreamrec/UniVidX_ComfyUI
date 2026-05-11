"""Tests for vram_buffer wiring + cache-key correctness in src.runtime.load_model.

Covers the v0.3 Tier-A correctness fixes:
- The runtime invokes `model.pipe.enable_vram_management(vram_buffer=...)`.
  `model.pipe` is an instance of UniVidX's OWN WanVideoPipeline subclass
  (vendor/UniVidX/src/pipelines/univid_intrinsic.py:24, method at line
  210), NOT UniVidIntrinsic itself and NOT DiffSynth's stock pipeline.
- `vram_buffer` is part of the model cache key — distinct buffer values
  produce distinct cache entries (was a real bug in 0.1.0–0.2.1 where
  the key omitted it).
- A missing method on `.pipe` (or a missing `.pipe`) gets a WARNING log
  so a future upstream rename cannot silently no-op the way the prior
  "deprecated, no-op" framing in 0.2.1 was misdiagnosed.

Asserts:
- load_model() calls model.pipe.enable_vram_management(vram_buffer=...)
  with the value the caller passed.
- A pipe lacking the method gets a logged warning, not silent skip.
- Two loads with the same vram_buffer share a cache slot.
- Two loads with different vram_buffer get distinct cache slots.
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


class _FakePipe:
    """Stand-in for UniVidX's WanVideoPipeline subclass — the object
    that lives at `model.pipe` and is the actual receiver of
    enable_vram_management()."""

    def __init__(self):
        self.evm_calls: list[float] = []

    def enable_vram_management(self, vram_buffer=0.5, **_kw):
        self.evm_calls.append(float(vram_buffer))


class _FakeUniVidModel:
    """Mimics the UniVidIntrinsic / UniVidAlpha surface area load_model
    touches: constructor + train() + a `.pipe` attribute that carries
    the real enable_vram_management entrypoint."""

    instance_count = 0

    def __init__(self, **kwargs):
        type(self).instance_count += 1
        self.init_kwargs = kwargs
        self.pipe = _FakePipe()

    def train(self, mode):
        return self


class _FakeNoEVMModel:
    """Mimics a hypothetical upstream rev where enable_vram_management
    was renamed/removed on the pipe. The runtime should not crash but
    should log a WARNING surfacing the no-op."""

    instance_count = 0

    def __init__(self, **kwargs):
        type(self).instance_count += 1
        self.pipe = types.SimpleNamespace()  # no enable_vram_management

    def train(self, mode):
        return self


@pytest.fixture
def stub_runtime(monkeypatch):
    """Stub every external dependency load_model touches: filesystem
    (resolve_paths), symlink setup (initialize), the chdir context
    (unividx_cwd), the mmgp safetensors patch, the SDPA un-polluter, and
    UniVidX's MODEL_REGISTRY. Yields the fake registry so tests can
    swap the registered class on demand."""
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
    monkeypatch.setattr(runtime, "resolve_paths",
                        lambda root, variant=None: fake_paths)
    monkeypatch.setattr(runtime, "initialize", lambda variant=None: None)
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


def test_load_model_invokes_enable_vram_management_on_pipe(stub_runtime):
    """The runtime must reach enable_vram_management THROUGH model.pipe
    (UniVidX's own WanVideoPipeline subclass) and pass the caller-
    supplied buffer value through."""
    from src.runtime import load_model

    model = load_model("intrinsic", vram_buffer=7.5)
    assert model.pipe.evm_calls == [7.5], (
        f"Expected model.pipe.enable_vram_management called once with 7.5, "
        f"got {model.pipe.evm_calls!r}"
    )


def test_load_model_warns_when_pipe_lacks_enable_vram_management(
        stub_runtime, caplog):
    """If a future upstream rev removes/renames enable_vram_management
    on the .pipe object, skipping silently would re-create the same
    misdiagnosis that gave us the bogus 'deprecated, no-op' framing in
    0.2.1. The runtime should log a WARNING."""
    from src.runtime import load_model

    stub_runtime.MODEL_REGISTRY["UniVidIntrinsic"] = _FakeNoEVMModel

    with caplog.at_level(logging.WARNING, logger="unividx"):
        load_model("intrinsic", vram_buffer=4.0)

    matching = [r for r in caplog.records
                if "enable_vram_management" in r.message]
    assert matching, (
        "Expected a WARNING mentioning enable_vram_management when the "
        "pipe lacks the method; got log records: "
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


def test_cache_evicts_oldest_when_capacity_exceeded(stub_runtime):
    """LRU eviction guard: the model cache must not grow unbounded.
    Loading more distinct cache keys than the capacity should evict
    the oldest entries so subsequent loads cold-load fresh — keeps a
    multi-condition bench (or a graph that toggles dit_weight_mode
    repeatedly) from ballooning into N×28 GB host RAM."""
    from src import runtime
    from src.runtime import load_model

    # Force capacity to 2 for the test (matches the production default).
    runtime._MODEL_CACHE_MAX_SIZE = 2

    a = load_model("intrinsic", vram_buffer=4.0)
    b = load_model("intrinsic", vram_buffer=8.0)
    c = load_model("intrinsic", vram_buffer=12.0)   # evicts `a`
    # Cache should now hold {b's key, c's key}; `a`'s key is gone.
    assert len(runtime._MODEL_CACHE) == 2
    # Loading `a` again should produce a NEW instance (cache miss).
    a2 = load_model("intrinsic", vram_buffer=4.0)
    assert a is not a2, "a should have been evicted; loading it again must cold-load"
    # Three distinct models cached so far... wait, after a2 cache holds
    # {c, a2}, having evicted b. So load_model("intrinsic", 8.0) is
    # another miss.
    b2 = load_model("intrinsic", vram_buffer=8.0)
    assert b is not b2


def test_clear_cache_drops_all_entries(stub_runtime):
    """The explicit clear_cache() helper must drop every entry so a
    bench driver can force a clean state between conditions."""
    from src import runtime
    from src.runtime import load_model, clear_cache

    load_model("intrinsic", vram_buffer=4.0)
    load_model("intrinsic", vram_buffer=8.0)
    assert len(runtime._MODEL_CACHE) == 2
    clear_cache()
    assert len(runtime._MODEL_CACHE) == 0


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
    assert a.pipe.evm_calls == [4.0]
    assert b.pipe.evm_calls == [12.0]
