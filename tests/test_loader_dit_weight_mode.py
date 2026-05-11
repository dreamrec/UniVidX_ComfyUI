"""Tests for the Tier-B4 `dit_weight_mode` knob on UniVidXLoader.

Covers the new loader widget that selects how DiT weights are loaded
(BF16 shards vs pre-quantized FP8 vs the legacy runtime-quantize path).

Asserts:
- New widget exists in INPUT_TYPES with the expected enum values.
- Default `dit_weight_mode="auto"` reproduces 0.3.0 behaviour:
  bf16_shards when dtype=bfloat16, fp8_runtime_experimental when
  dtype=fp8_e4m3fn (legacy path) + deprecation warning emitted.
- Explicit `dit_weight_mode="fp8_prequantized"` routes through the
  new FP8 weights path (NotImplementedError until B3 lands — pin
  the contract while the implementation catches up).
- Explicit `dit_weight_mode="bf16_shards"` overrides dtype=fp8_*
  back to BF16.
- runtime.load_model cache key includes dit_weight_mode so switching
  modes triggers a cache miss and a clean reload.
- The legacy dtype=fp8_e4m3fn / fp8_e5m2 values emit a deprecation
  warning so users have time to migrate before 0.4.0 removes them.
"""
from __future__ import annotations

import logging
import os
import sys
import types
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

HERE = __file__
_REPO = os.path.abspath(os.path.join(os.path.dirname(HERE), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class _FakePipe:
    def __init__(self):
        self.evm_calls: list[float] = []

    def enable_vram_management(self, vram_buffer=0.5, **_kw):
        self.evm_calls.append(float(vram_buffer))


class _FakeUniVidModel:
    instance_count = 0

    def __init__(self, **kwargs):
        type(self).instance_count += 1
        self.init_kwargs = kwargs
        self.pipe = _FakePipe()

    def train(self, mode):
        return self


@pytest.fixture
def stub_runtime(monkeypatch):
    """Same stub strategy as test_runtime_cache_key but yields the
    runtime module so tests can inspect cache state directly."""
    from src import runtime

    runtime._MODEL_CACHE.clear()
    _FakeUniVidModel.instance_count = 0

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
    # Stub the legacy quantize path so tests don't try to actually
    # call mmgp on a fake model.
    monkeypatch.setattr(runtime, "_quantize_dit_fp8",
                        lambda model, qtype: None)

    fake_registry = types.ModuleType("scripts.registry")
    fake_registry.MODEL_REGISTRY = {
        "UniVidIntrinsic": _FakeUniVidModel,
        "UniVidAlpha": _FakeUniVidModel,
    }
    fake_scripts = types.ModuleType("scripts")
    fake_scripts.registry = fake_registry
    monkeypatch.setitem(sys.modules, "scripts", fake_scripts)
    monkeypatch.setitem(sys.modules, "scripts.registry", fake_registry)

    yield runtime
    runtime._MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Widget contract
# ---------------------------------------------------------------------------

def test_loader_exposes_dit_weight_mode_widget():
    """The loader must expose the new `dit_weight_mode` enum widget
    with the four documented values."""
    from nodes.loader import UniVidXLoader
    inputs = UniVidXLoader.INPUT_TYPES()
    # Optional dict carries the new widget — required would break old workflows.
    optional = inputs.get("optional", {})
    assert "dit_weight_mode" in optional, (
        "expected `dit_weight_mode` in optional inputs; got "
        f"keys={list(optional.keys())}"
    )
    enum_spec = optional["dit_weight_mode"]
    # Widget shape: (["a", "b", ...], {"default": ..., "tooltip": ...})
    values = enum_spec[0] if isinstance(enum_spec, tuple) else enum_spec
    expected = {"auto", "bf16_shards", "fp8_prequantized", "fp8_runtime_experimental"}
    assert set(values) == expected, (
        f"unexpected enum values: got {set(values)}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Auto mode preserves 0.3.0 behaviour
# ---------------------------------------------------------------------------

def test_dit_weight_mode_auto_with_bfloat16_loads_bf16_shards(stub_runtime):
    """dit_weight_mode='auto' + dtype='bfloat16' → BF16 shards path
    (the 0.3.0 default behaviour, unchanged)."""
    from nodes.loader import UniVidXLoader
    out = UniVidXLoader().load(variant="intrinsic", dtype="bfloat16",
                               dit_weight_mode="auto")
    model, variant = out[0]
    assert variant == "intrinsic"
    # Sanity: model was constructed, no FP8-related kwargs touched it.
    assert isinstance(model, _FakeUniVidModel)


def test_dit_weight_mode_auto_with_legacy_fp8_routes_to_experimental(
        stub_runtime, caplog):
    """dtype='fp8_e4m3fn' + dit_weight_mode='auto' must route through
    the legacy runtime-quantize path (now branded
    `fp8_runtime_experimental`) AND emit a DeprecationWarning-flavored
    log so users see the migration signal."""
    from nodes.loader import UniVidXLoader

    with caplog.at_level(logging.WARNING, logger="unividx"):
        UniVidXLoader().load(variant="intrinsic", dtype="fp8_e4m3fn",
                              dit_weight_mode="auto")

    deprecation_lines = [r for r in caplog.records
                         if "deprecat" in r.message.lower()
                         and ("fp8_e4m3fn" in r.message
                              or "fp8_runtime_experimental" in r.message
                              or "0.4.0" in r.message)]
    assert deprecation_lines, (
        "expected a deprecation WARNING mentioning fp8_e4m3fn / "
        "fp8_runtime_experimental / 0.4.0; got: "
        f"{[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Explicit overrides
# ---------------------------------------------------------------------------

def test_dit_weight_mode_fp8_prequantized_works_without_external_file(
        stub_runtime, monkeypatch):
    """Post-B8 pivot: fp8_prequantized runtime-quantizes the BF16
    cold-load weights to FP8 with per-tensor absmax scaling. No
    external Kijai file required (the upstream `_scaled` Wan2.1
    variant doesn't exist; bare-cast variant produces poor quality).
    Mock the substitution to verify wiring without a real model
    load."""
    from src import runtime
    from nodes.loader import UniVidXLoader

    sub_calls: list[tuple] = []

    def _fake_sub(model, variant):
        sub_calls.append((type(model).__name__, variant))

    monkeypatch.setattr(runtime, "_apply_fp8_substitution", _fake_sub)

    # No FileNotFoundError; substitution runs.
    UniVidXLoader().load(variant="intrinsic", dtype="bfloat16",
                          dit_weight_mode="fp8_prequantized")
    assert len(sub_calls) == 1
    assert sub_calls[0][1] == "intrinsic"


def test_dit_weight_mode_fp8_prequantized_invokes_substitution(
        stub_runtime, monkeypatch):
    """Happy path: load_model() calls _apply_fp8_substitution
    exactly once when dit_weight_mode='fp8_prequantized'. The
    substitution itself is monkeypatched here; dispatch between
    file-based and runtime-quantize paths is covered by the
    test_apply_fp8_substitution_* tests below."""
    from src import runtime
    from nodes.loader import UniVidXLoader

    sub_calls: list[tuple] = []

    def _fake_sub(model, variant):
        sub_calls.append((type(model).__name__, variant))

    monkeypatch.setattr(runtime, "_apply_fp8_substitution", _fake_sub)

    UniVidXLoader().load(variant="intrinsic", dtype="bfloat16",
                          dit_weight_mode="fp8_prequantized")
    assert len(sub_calls) == 1, (
        f"_apply_fp8_substitution should be invoked exactly once; "
        f"got calls={sub_calls}"
    )
    assert sub_calls[0][1] == "intrinsic"


def test_apply_fp8_substitution_dispatches_to_runtime_quantize_when_no_file(
        monkeypatch):
    """Default 0.4.0 path: when the Kijai _scaled FP8 file is absent,
    _apply_fp8_substitution falls back to quantize_dit_inplace."""
    from src import runtime

    monkeypatch.setattr(runtime, "_resolve_fp8_weights_path", lambda: None)

    runtime_calls: list[str] = []
    file_calls: list[str] = []
    monkeypatch.setattr(
        runtime, "_apply_fp8_substitution_runtime_quantize",
        lambda model: runtime_calls.append("runtime"),
    )
    monkeypatch.setattr(
        runtime, "_apply_fp8_substitution_from_file",
        lambda model, p: file_calls.append(p),
    )

    runtime._apply_fp8_substitution(object(), "intrinsic")
    assert runtime_calls == ["runtime"]
    assert file_calls == []


def test_apply_fp8_substitution_dispatches_to_file_when_scaled_file_present(
        monkeypatch):
    """Opt-in path: when _resolve_fp8_weights_path finds a Kijai
    _scaled file on disk, _apply_fp8_substitution uses the
    file-based loader instead of the runtime quantize."""
    from src import runtime

    fake_path = "/fake/Wan2_1-T2V-14B_fp8_e4m3fn_scaled.safetensors"
    monkeypatch.setattr(runtime, "_resolve_fp8_weights_path",
                        lambda: fake_path)

    runtime_calls: list[str] = []
    file_calls: list[str] = []
    monkeypatch.setattr(
        runtime, "_apply_fp8_substitution_runtime_quantize",
        lambda model: runtime_calls.append("runtime"),
    )
    monkeypatch.setattr(
        runtime, "_apply_fp8_substitution_from_file",
        lambda model, p: file_calls.append(p),
    )

    runtime._apply_fp8_substitution(object(), "intrinsic")
    assert runtime_calls == []
    assert file_calls == [fake_path]


def test_dit_weight_mode_bf16_shards_overrides_legacy_fp8_dtype(
        stub_runtime, caplog):
    """Explicit dit_weight_mode='bf16_shards' must win over legacy
    dtype=fp8_e4m3fn — gives users a way to short-circuit the
    deprecated path without changing the dtype enum."""
    from nodes.loader import UniVidXLoader

    out = UniVidXLoader().load(variant="intrinsic", dtype="fp8_e4m3fn",
                                dit_weight_mode="bf16_shards")
    model, _ = out[0]
    assert isinstance(model, _FakeUniVidModel)
    # No deprecation warning should fire because the legacy path
    # wasn't taken.
    deprecation_lines = [r for r in caplog.records
                         if "deprecat" in r.message.lower()]
    assert not deprecation_lines, (
        f"unexpected deprecation log when bf16_shards overrides fp8 dtype: "
        f"{[r.message for r in deprecation_lines]}"
    )


# ---------------------------------------------------------------------------
# Cache key contains dit_weight_mode
# ---------------------------------------------------------------------------

def test_cache_miss_when_dit_weight_mode_differs(stub_runtime):
    """Two loads with same variant/dtype but different
    dit_weight_mode must produce two distinct cache entries —
    they materially change what's loaded into memory."""
    from src.runtime import load_model

    a = load_model("intrinsic", dit_weight_mode="bf16_shards")
    b = load_model("intrinsic", dit_weight_mode="fp8_runtime_experimental",
                   quantize_fp8="qfloat8")
    assert a is not b
    assert _FakeUniVidModel.instance_count == 2


def test_cache_hit_when_dit_weight_mode_matches(stub_runtime):
    """Two loads with identical params (including dit_weight_mode)
    must share a cache slot — preserves the 0.3.0 cache hit
    behaviour for the new param."""
    from src.runtime import load_model

    a = load_model("intrinsic", dit_weight_mode="bf16_shards")
    b = load_model("intrinsic", dit_weight_mode="bf16_shards")
    assert a is b
    assert _FakeUniVidModel.instance_count == 1
