"""Tests for Tier B2 (state-dict triage) and B6 (FP8Linear forward).

Validates the FP8 weight handling against synthetic data so the
correctness contract is pinned before the 14 GB Kijai download finishes
and integration testing becomes possible.

Covers:
- FP8Linear: stores weight as float8_e4m3fn buffer; forward path
  dequantizes (weight * scale_weight) to input.dtype before
  F.linear; bias added correctly; batch dim handled.
- Numerical correctness: an FP8Linear with weight=identity_quantized
  and scale=1.0 produces output ≈ input (modulo FP8 quantization
  noise — tolerance set generously).
- triage_kijai_state_dict: splits a synthetic Kijai-style state_dict
  into the four buckets (fp8_weight, scale_weight, scale_input,
  fp_aux) and produces (key -> parent_module_name) mapping that
  load_fp8_dit can use to find the right targets in the model.
- load_fp8_state_dict_into: given a small nn.Module containing two
  nn.Linear layers, loads a synthetic Kijai-shaped state_dict;
  asserts the Linears are now FP8Linear instances with the correct
  weight/scale tensors attached.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = __file__
_REPO = os.path.abspath(os.path.join(os.path.dirname(HERE), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# FP8Linear forward semantics
# ---------------------------------------------------------------------------

def test_fp8_linear_forward_with_zero_input_returns_bias():
    """forward(zeros) must equal bias — proves the dequant path doesn't
    NaN on the zero edge case and that bias is added, not dropped."""
    from src.fp8_loader import FP8Linear

    layer = FP8Linear(in_features=4, out_features=3, bias=True)
    layer.bias.data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)

    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    y = layer(x)
    assert y.shape == (2, 3)
    assert torch.allclose(y[0], layer.bias.to(y.dtype))


def test_fp8_linear_forward_identity_recovers_input_approximately():
    """With weight = quantize(identity) and scale = 1.0, F.linear should
    pass input through approximately unchanged. Tolerance set wide
    enough to absorb FP8 e4m3 quantization noise (~1-3%)."""
    from src.fp8_loader import FP8Linear

    n = 8
    layer = FP8Linear(in_features=n, out_features=n, bias=False)
    # Quantize identity by casting BF16 -> FP8 -> store.
    eye = torch.eye(n, dtype=torch.bfloat16)
    layer.weight.data = eye.to(torch.float8_e4m3fn)
    layer.scale_weight.data = torch.tensor([1.0], dtype=torch.float32)

    x = torch.randn(4, n, dtype=torch.bfloat16) * 0.5  # keep in FP8 range
    y = layer(x)
    # FP8 e4m3 has ~3 bits mantissa → noise floor of a few percent on
    # individual element values. Use generous atol for the smoke test.
    assert torch.allclose(y, x, atol=0.1, rtol=0.1), (
        f"identity reconstruction failed: max diff "
        f"{(y - x).abs().max().item():.3f}"
    )


def test_fp8_linear_forward_dtype_matches_input():
    """The forward dequantizes the weight to input.dtype so downstream
    BF16 / FP16 layers don't see a dtype mismatch."""
    from src.fp8_loader import FP8Linear

    layer = FP8Linear(in_features=4, out_features=4, bias=True)
    x_bf16 = torch.zeros(1, 4, dtype=torch.bfloat16)
    x_fp16 = torch.zeros(1, 4, dtype=torch.float16)
    y_bf16 = layer(x_bf16)
    y_fp16 = layer(x_fp16)
    assert y_bf16.dtype == torch.bfloat16
    assert y_fp16.dtype == torch.float16


def test_fp8_linear_weight_persists_as_float8_e4m3fn():
    """After construction, layer.weight must be FP8 dtype (not silently
    promoted to BF16). Otherwise we lose the memory win."""
    from src.fp8_loader import FP8Linear

    layer = FP8Linear(in_features=4, out_features=4, bias=False)
    assert layer.weight.dtype == torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# State-dict triage
# ---------------------------------------------------------------------------

def _make_synthetic_kijai_sd():
    """Build a state_dict that mirrors Kijai's Wan FP8 layout:
    - 2 FP8 Linear modules (q + k), each with weight + scale_weight +
      scale_input + bias.
    - 1 BF16 norm (time_embedding.0.weight) — full-precision aux.
    """
    return {
        "blocks.0.cross_attn.q.weight":
            torch.zeros(8, 8, dtype=torch.float8_e4m3fn),
        "blocks.0.cross_attn.q.scale_weight":
            torch.tensor(0.123, dtype=torch.float32),  # scalar
        "blocks.0.cross_attn.q.scale_input":
            torch.tensor(0.456, dtype=torch.float32),
        "blocks.0.cross_attn.q.bias":
            torch.zeros(8, dtype=torch.bfloat16),
        "blocks.0.cross_attn.k.weight":
            torch.zeros(8, 8, dtype=torch.float8_e4m3fn),
        "blocks.0.cross_attn.k.scale_weight":
            torch.tensor(0.234, dtype=torch.float32),
        "blocks.0.cross_attn.k.scale_input":
            torch.tensor(0.567, dtype=torch.float32),
        "blocks.0.cross_attn.k.bias":
            torch.zeros(8, dtype=torch.bfloat16),
        "time_embedding.0.weight":
            torch.zeros(16, 8, dtype=torch.bfloat16),
    }


def test_triage_splits_kijai_state_dict_into_four_buckets():
    """triage_kijai_state_dict returns (fp8_weights, scale_weights,
    scale_inputs, fp_aux) with the expected key memberships."""
    from src.fp8_loader import triage_kijai_state_dict

    sd = _make_synthetic_kijai_sd()
    fp8, sw, si, aux = triage_kijai_state_dict(sd)

    assert set(fp8) == {
        "blocks.0.cross_attn.q.weight",
        "blocks.0.cross_attn.k.weight",
    }
    assert set(sw) == {
        "blocks.0.cross_attn.q.scale_weight",
        "blocks.0.cross_attn.k.scale_weight",
    }
    assert set(si) == {
        "blocks.0.cross_attn.q.scale_input",
        "blocks.0.cross_attn.k.scale_input",
    }
    # fp_aux: anything not in the FP8 trio. Includes biases and norms.
    assert set(aux) == {
        "blocks.0.cross_attn.q.bias",
        "blocks.0.cross_attn.k.bias",
        "time_embedding.0.weight",
    }


def test_triage_pairs_scales_to_parent_fp8_weight():
    """The triage function returns (or makes accessible) a mapping
    from each FP8 weight key to its (scale_weight_key, scale_input_key)
    pair so the loader can attach them to the same module."""
    from src.fp8_loader import triage_kijai_state_dict

    sd = _make_synthetic_kijai_sd()
    fp8, sw, si, aux = triage_kijai_state_dict(sd)

    # Convention: <key>.scale_weight pairs with <key>.weight via the
    # shared "blocks.0.cross_attn.q" prefix. Verify by constructing
    # expected pairings.
    for fp8_key in fp8:
        prefix = fp8_key.rsplit(".weight", 1)[0]
        assert f"{prefix}.scale_weight" in sw
        assert f"{prefix}.scale_input" in si


# ---------------------------------------------------------------------------
# Integration: load_fp8_state_dict_into mutates a small target model
# ---------------------------------------------------------------------------

class _TinyTarget(nn.Module):
    """Small stand-in for the parts of UniVidX's WanModel that B3
    will operate on. Has the same structure: bare top-level keys
    (no diffusion_model. / dit. prefix), nested blocks with
    cross_attn.q / cross_attn.k Linear modules."""

    def __init__(self):
        super().__init__()
        # Use a single block to keep the test compact.
        self.blocks = nn.ModuleList([_TinyBlock()])
        # A non-Linear top-level tensor that should stay BF16.
        self.time_embedding = nn.Sequential(nn.Linear(8, 16, bias=False))
        # Make time_embedding's weight BF16 so the loader's bf16 path
        # has a target to write into.
        self.time_embedding[0].weight.data = self.time_embedding[0].weight.data.to(torch.bfloat16)


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = _TinyCrossAttn()


class _TinyCrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 8)
        self.k = nn.Linear(8, 8)


def test_load_fp8_state_dict_into_replaces_linears_with_fp8linear():
    """load_fp8_state_dict_into(model, kijai_sd) must walk the model,
    find the Linear modules whose names match the FP8 keys in the
    state_dict, replace them with FP8Linear instances, and load the
    quantized weight + scales onto each."""
    from src.fp8_loader import FP8Linear, load_fp8_state_dict_into

    model = _TinyTarget()
    sd = _make_synthetic_kijai_sd()

    # Pre-condition: both attn projections are stock nn.Linear.
    assert isinstance(model.blocks[0].cross_attn.q, nn.Linear)
    assert not isinstance(model.blocks[0].cross_attn.q, FP8Linear)

    report = load_fp8_state_dict_into(model, sd)

    # Post-condition: both attn projections are now FP8Linear; carry
    # the scale_weight values from the synthetic state_dict.
    q = model.blocks[0].cross_attn.q
    k = model.blocks[0].cross_attn.k
    assert isinstance(q, FP8Linear)
    assert isinstance(k, FP8Linear)
    assert q.weight.dtype == torch.float8_e4m3fn
    assert torch.allclose(q.scale_weight, torch.tensor([0.123]), atol=1e-3)
    assert torch.allclose(k.scale_weight, torch.tensor([0.234]), atol=1e-3)

    # Aux BF16 tensor loaded into time_embedding.0.weight (untouched
    # in dtype).
    assert model.time_embedding[0].weight.dtype == torch.bfloat16

    # Report (returned by load_fp8_state_dict_into) tells caller what
    # happened. Useful for diagnostic logs in B3.
    assert report["fp8_linears_replaced"] == 2
    assert report["aux_keys_loaded"] >= 1  # at least time_embedding


def test_load_fp8_state_dict_into_descends_into_peft_wrappers():
    """After UniVidX's add_multiple_loras_to_model, every target
    Linear is wrapped in peft.tuners.lora.Linear with the original
    nn.Linear hanging off `.base_layer`. The FP8 loader must descend
    into the wrapper so it can replace the BASE layer (leaving the
    LoRA adapters above it untouched and in BF16, per B5)."""
    from src.fp8_loader import FP8Linear, load_fp8_state_dict_into

    # Build a tiny model and PEFT-wrap one of its Linears.
    model = _TinyTarget()
    inner_q = model.blocks[0].cross_attn.q
    assert isinstance(inner_q, nn.Linear)

    # Stand-in PEFT-style wrapper: same surface area as
    # peft.tuners.lora.Linear (has .base_layer, isn't a subclass of
    # nn.Linear itself).
    class _PEFTWrapper(nn.Module):
        def __init__(self, base_layer):
            super().__init__()
            self.base_layer = base_layer
            # LoRA-style adapters live alongside base_layer and must
            # survive the FP8 substitution untouched.
            self.lora_A = nn.Linear(base_layer.in_features, 4, bias=False)
            self.lora_B = nn.Linear(4, base_layer.out_features, bias=False)

        def forward(self, x):
            return self.base_layer(x) + self.lora_B(self.lora_A(x))

    wrapper = _PEFTWrapper(inner_q)
    model.blocks[0].cross_attn.q = wrapper

    # Snapshot LoRA weights so we can confirm they're untouched.
    lora_A_before = wrapper.lora_A.weight.data.clone()

    sd = _make_synthetic_kijai_sd()
    report = load_fp8_state_dict_into(model, sd)

    # The wrapper itself remains in place — only its base_layer
    # changed identity (now FP8Linear, was nn.Linear).
    new_wrapper = model.blocks[0].cross_attn.q
    assert isinstance(new_wrapper, _PEFTWrapper), (
        "PEFT wrapper must survive FP8 substitution"
    )
    assert isinstance(new_wrapper.base_layer, FP8Linear), (
        "base_layer must have been replaced with FP8Linear"
    )
    assert new_wrapper.base_layer.weight.dtype == torch.float8_e4m3fn

    # LoRA adapters preserved — this is the B5 contract.
    assert torch.allclose(new_wrapper.lora_A.weight.data, lora_A_before), (
        "lora_A weights must not be touched by FP8 substitution"
    )

    # Report still shows we replaced 2 Linears (the q via PEFT
    # descent + the k via direct match).
    assert report["fp8_linears_replaced"] == 2


def test_load_fp8_state_dict_into_warns_on_unmatched_keys(caplog):
    """If the Kijai state_dict has keys that don't match any module
    in the target model, the loader should log them — same as
    nn.Module.load_state_dict(strict=False) gives missing/unexpected
    lists, but here we're doing it manually."""
    import logging
    from src.fp8_loader import load_fp8_state_dict_into

    model = _TinyTarget()
    sd = _make_synthetic_kijai_sd()
    sd["blocks.99.unknown.weight"] = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    sd["blocks.99.unknown.scale_weight"] = torch.tensor(1.0, dtype=torch.float32)
    sd["blocks.99.unknown.scale_input"] = torch.tensor(1.0, dtype=torch.float32)

    with caplog.at_level(logging.WARNING, logger="unividx"):
        report = load_fp8_state_dict_into(model, sd)

    # Either a warning fired OR the report itemizes the unmatched keys.
    assert (
        report.get("unmatched_keys")
        or any("unmatched" in r.message.lower() or "99" in r.message
               for r in caplog.records)
    ), f"expected unmatched-key signal, got report={report}, logs={[r.message for r in caplog.records]}"
