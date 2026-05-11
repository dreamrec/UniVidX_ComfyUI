"""Tests for src.lora_merge.merge_lora_into_base — used by Tier C
(LightX2V step-distill stacking).

The merge math is pure linear algebra: for each base Linear matching
a key in the LoRA state-dict, replace W with W + scale * (B @ A) where
scale = strength * (alpha / rank). Tests pin the math against
synthetic LoRAs so the integration with a real LightX2V file is
purely "did we get the keys right" not "did we get the math right."
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


class _TinyTarget(nn.Module):
    """Mirror the parts of UniVidX's WanModel the merge walks:
    bare top-level keys, nested blocks with attention Linears."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock()])


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TinyAttn()


class _TinyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 8, bias=False)
        self.k = nn.Linear(8, 8, bias=False)


def _identity_lora(in_features: int, out_features: int, rank: int):
    """Build a LoRA pair whose B @ A is the identity matrix scaled by
    `magnitude`. Useful for verifying that strength scales linearly."""
    # B @ A = I_n requires rank >= n. Pick A = first `n` rows of an
    # identity-ish matrix, B similar. For rank == out_features == in_features,
    # A = I, B = I gives B @ A = I.
    assert rank >= out_features and rank >= in_features
    A = torch.zeros(rank, in_features)
    A[:in_features, :in_features] = torch.eye(in_features)
    B = torch.zeros(out_features, rank)
    B[:out_features, :out_features] = torch.eye(out_features)
    return A, B


def test_merge_lora_into_base_strength_zero_is_identity():
    """strength=0 must leave base weights bit-identical."""
    from src.lora_merge import merge_lora_into_base

    model = _TinyTarget()
    snapshot = {n: p.data.clone() for n, p in model.named_parameters()}
    n = 8
    A, B = _identity_lora(n, n, rank=n)
    lora_sd = {
        "blocks.0.self_attn.q.lora_A.weight": A,
        "blocks.0.self_attn.q.lora_B.weight": B,
    }

    merge_lora_into_base(model, lora_sd, strength=0.0, alpha=n, rank=n)

    for name, before in snapshot.items():
        assert torch.equal(dict(model.named_parameters())[name].data, before), (
            f"{name} changed when strength=0"
        )


def test_merge_lora_into_base_unit_strength_adds_BA():
    """At strength=1, alpha=rank (scale=1), the base weight should be
    base + B@A."""
    from src.lora_merge import merge_lora_into_base

    model = _TinyTarget()
    base_before = model.blocks[0].self_attn.q.weight.data.clone()
    n = 8
    A, B = _identity_lora(n, n, rank=n)
    lora_sd = {
        "blocks.0.self_attn.q.lora_A.weight": A,
        "blocks.0.self_attn.q.lora_B.weight": B,
    }

    merge_lora_into_base(model, lora_sd, strength=1.0, alpha=n, rank=n)

    base_after = model.blocks[0].self_attn.q.weight.data
    expected = base_before + (B @ A)
    assert torch.allclose(base_after, expected, atol=1e-5), (
        f"merge math wrong: max diff "
        f"{(base_after - expected).abs().max().item():.5f}"
    )


def test_merge_lora_into_base_strength_scales_linearly():
    """At strength=2, the residual should be twice the size."""
    from src.lora_merge import merge_lora_into_base

    model = _TinyTarget()
    base_before = model.blocks[0].self_attn.q.weight.data.clone()
    n = 8
    A, B = _identity_lora(n, n, rank=n)
    lora_sd = {
        "blocks.0.self_attn.q.lora_A.weight": A,
        "blocks.0.self_attn.q.lora_B.weight": B,
    }

    merge_lora_into_base(model, lora_sd, strength=2.0, alpha=n, rank=n)

    base_after = model.blocks[0].self_attn.q.weight.data
    expected = base_before + 2.0 * (B @ A)
    assert torch.allclose(base_after, expected, atol=1e-5)


def test_merge_lora_into_base_alpha_over_rank_scales_residual():
    """PEFT convention: effective scale = strength * (alpha / rank).
    With alpha=64, rank=32, strength=1 → multiply residual by 2."""
    from src.lora_merge import merge_lora_into_base

    model = _TinyTarget()
    base_before = model.blocks[0].self_attn.q.weight.data.clone()
    n = 8
    A, B = _identity_lora(n, n, rank=n)
    lora_sd = {
        "blocks.0.self_attn.q.lora_A.weight": A,
        "blocks.0.self_attn.q.lora_B.weight": B,
    }

    merge_lora_into_base(model, lora_sd, strength=1.0, alpha=16, rank=8)

    expected = base_before + (16 / 8) * (B @ A)
    actual = model.blocks[0].self_attn.q.weight.data
    assert torch.allclose(actual, expected, atol=1e-5)


def test_merge_lora_into_base_ignores_unmatched_keys(caplog):
    """LoRA keys not matching any Linear in the model should be reported
    but not crash. Keys matching an existing Linear must be applied."""
    import logging
    from src.lora_merge import merge_lora_into_base

    model = _TinyTarget()
    n = 8
    A, B = _identity_lora(n, n, rank=n)
    lora_sd = {
        "blocks.0.self_attn.q.lora_A.weight": A,
        "blocks.0.self_attn.q.lora_B.weight": B,
        "blocks.99.unknown.lora_A.weight": A,
        "blocks.99.unknown.lora_B.weight": B,
    }

    with caplog.at_level(logging.WARNING, logger="unividx"):
        report = merge_lora_into_base(model, lora_sd, strength=1.0,
                                       alpha=n, rank=n)
    assert report["merged"] == 1
    assert report["unmatched"] >= 1


def test_merge_lora_into_base_descends_into_peft_wrappers():
    """When the target Linear is wrapped by a PEFT-style module
    (`.base_layer` attribute holding the real nn.Linear), the merge
    must apply to base_layer.weight, not to the wrapper. The wrapper's
    sibling LoRA adapters stay untouched."""
    from src.lora_merge import merge_lora_into_base

    model = _TinyTarget()
    inner_q = model.blocks[0].self_attn.q
    base_before = inner_q.weight.data.clone()
    n = 8

    class _PEFTWrapper(nn.Module):
        def __init__(self, base_layer):
            super().__init__()
            self.base_layer = base_layer
            # UniVidX-style sibling adapters — must NOT be touched by merge.
            self.lora_A = nn.Linear(n, 4, bias=False)
            self.lora_B = nn.Linear(4, n, bias=False)

        def forward(self, x):
            return self.base_layer(x) + self.lora_B(self.lora_A(x))

    wrapper = _PEFTWrapper(inner_q)
    model.blocks[0].self_attn.q = wrapper
    lora_A_before = wrapper.lora_A.weight.data.clone()
    lora_B_before = wrapper.lora_B.weight.data.clone()

    A, B = _identity_lora(n, n, rank=n)
    lora_sd = {
        "blocks.0.self_attn.q.lora_A.weight": A,
        "blocks.0.self_attn.q.lora_B.weight": B,
    }
    merge_lora_into_base(model, lora_sd, strength=1.0, alpha=n, rank=n)

    # Inner base_layer.weight was merged.
    expected = base_before + (B @ A)
    assert torch.allclose(wrapper.base_layer.weight.data, expected, atol=1e-5)
    # Sibling LoRA adapters survived untouched.
    assert torch.equal(wrapper.lora_A.weight.data, lora_A_before)
    assert torch.equal(wrapper.lora_B.weight.data, lora_B_before)


def test_merge_lora_into_base_handles_prefix_stripping():
    """Some safetensors LoRA files ship keys with a `diffusion_model.`
    or `model.` prefix; the merge must accept a key_prefix to strip.
    Without it, keys won't match the bare model paths."""
    from src.lora_merge import merge_lora_into_base

    model = _TinyTarget()
    n = 8
    A, B = _identity_lora(n, n, rank=n)
    prefixed_sd = {
        "diffusion_model.blocks.0.self_attn.q.lora_A.weight": A,
        "diffusion_model.blocks.0.self_attn.q.lora_B.weight": B,
    }
    base_before = model.blocks[0].self_attn.q.weight.data.clone()

    report = merge_lora_into_base(model, prefixed_sd, strength=1.0,
                                   alpha=n, rank=n,
                                   strip_prefix="diffusion_model.")
    assert report["merged"] == 1
    expected = base_before + (B @ A)
    assert torch.allclose(model.blocks[0].self_attn.q.weight.data, expected,
                          atol=1e-5)


def test_merge_lora_into_base_kohya_lora_down_up_naming():
    """LightX2V (and many Kohya/Civitai LoRAs) use `lora_down`/`lora_up`
    instead of `lora_A`/`lora_B`. The merge must handle both.

    Convention: lora_down has shape (rank, in_features), lora_up has
    shape (out_features, rank). delta = lora_up @ lora_down."""
    from src.lora_merge import merge_lora_into_base

    model = _TinyTarget()
    base_before = model.blocks[0].self_attn.q.weight.data.clone()
    n = 8
    # Kohya naming: down then up.
    lora_down, lora_up = _identity_lora(n, n, rank=n)
    lora_sd = {
        "blocks.0.self_attn.q.lora_down.weight": lora_down,
        "blocks.0.self_attn.q.lora_up.weight": lora_up,
    }

    report = merge_lora_into_base(model, lora_sd, strength=1.0, alpha=n, rank=n)

    expected = base_before + (lora_up @ lora_down)
    assert torch.allclose(model.blocks[0].self_attn.q.weight.data, expected,
                          atol=1e-5)
    assert report["merged"] >= 1


def test_merge_lora_into_base_applies_bias_diff():
    """LightX2V ships `.diff_b` keys carrying additive bias deltas.
    For a Linear whose base has bias=None, the merge should CREATE
    the bias parameter and set it to strength * diff_b."""
    from src.lora_merge import merge_lora_into_base

    model = _TinyTarget()
    assert model.blocks[0].self_attn.q.bias is None, (
        "test fixture expected bias=False on q"
    )
    n = 8
    diff_b = torch.linspace(-1.0, 1.0, n)
    lora_sd = {
        "blocks.0.self_attn.q.diff_b": diff_b,
    }
    report = merge_lora_into_base(model, lora_sd, strength=1.0, rank=n)
    new_bias = model.blocks[0].self_attn.q.bias
    assert new_bias is not None, "bias should have been created from diff_b"
    assert torch.allclose(new_bias.data, diff_b, atol=1e-5)
    assert report.get("biases_patched", 0) >= 1


def test_merge_lora_into_base_applies_weight_diff_for_norms():
    """LightX2V's `.diff` keys carry direct weight deltas for non-Linear
    modules like RMSNorm and patch_embedding. The merge must apply them
    to the target module's `.weight` parameter."""
    from src.lora_merge import merge_lora_into_base

    # Wire a norm parameter onto our tiny target.
    class _WithNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_TinyBlockWithNorm()])

    class _TinyBlockWithNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _TinyAttnWithNorm()

    class _TinyAttnWithNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_q = nn.RMSNorm(8)  # PyTorch 2.4+ has RMSNorm built in
            self.q = nn.Linear(8, 8, bias=False)

    model = _WithNorm()
    n = 8
    diff = torch.linspace(-0.1, 0.1, n)
    norm_before = model.blocks[0].self_attn.norm_q.weight.data.clone()
    lora_sd = {
        "blocks.0.self_attn.norm_q.diff": diff,
    }
    report = merge_lora_into_base(model, lora_sd, strength=1.0, rank=n)
    expected = norm_before + diff
    assert torch.allclose(model.blocks[0].self_attn.norm_q.weight.data,
                          expected, atol=1e-5)
    assert report.get("weights_patched", 0) >= 1


def test_merge_lora_into_base_descends_into_autowrapped_module():
    """Regression for P1 (2026-05-11 external audit): UniVidX's
    `enable_vram_management()` wraps non-Linear modules (RMSNorms,
    Conv3d patch_embedding, head) in `AutoWrappedModule`, which stores
    the original module at `.module`. The state-dict path therefore
    becomes `<base>.module.weight` — but LightX2V's `.diff` patches
    address the bare `<base>.weight` path. Without descent through
    `.module`, `_apply_weight_diff` finds the wrapper, sees
    `hasattr(target, 'weight') is False`, and silently skips the
    patch (incremented `skipped`, no per-key warning).

    The fix: re-order step_distill before enable_vram_management so
    AutoWrappedModule wrapping happens AFTER the merge — and
    additionally make the resolver descend through `.module` so the
    merge still works if someone wires the order differently in a
    future code path. Both belt and braces.
    """
    from src.lora_merge import merge_lora_into_base

    class _AutoWrappedLike(nn.Module):
        """Stand-in for UniVidX's AutoWrappedModule. Holds the real
        module at .module — that's the indirection that bit us."""

        def __init__(self, inner):
            super().__init__()
            self.module = inner

        def forward(self, x):
            return self.module(x)

    class _WithNorm(nn.Module):
        def __init__(self):
            super().__init__()
            inner_norm = nn.RMSNorm(8)
            # Wrap it the way enable_vram_management would.
            self.norm_q = _AutoWrappedLike(inner_norm)

    model = _WithNorm()
    inner = model.norm_q.module
    n = 8
    diff = torch.linspace(-0.1, 0.1, n)
    weight_before = inner.weight.data.clone()
    lora_sd = {"norm_q.diff": diff}

    report = merge_lora_into_base(model, lora_sd, strength=1.0, rank=n)

    expected = weight_before + diff
    assert torch.allclose(inner.weight.data, expected, atol=1e-5), (
        f"merge did not descend through .module wrapper; "
        f"inner.weight unchanged "
        f"(max diff {(inner.weight.data - expected).abs().max().item():.5f})"
    )
    assert report["weights_patched"] >= 1
    assert report["unmatched"] == 0


def test_merge_lora_into_base_alpha_inferred_from_lora_state_dict_alpha_key():
    """Some LoRA files include an `<key>.alpha` scalar tensor encoding
    the alpha-per-target_module value. When present, the merge should
    use it instead of the global alpha override."""
    from src.lora_merge import merge_lora_into_base

    model = _TinyTarget()
    n = 8
    A, B = _identity_lora(n, n, rank=n)
    lora_sd = {
        "blocks.0.self_attn.q.lora_A.weight": A,
        "blocks.0.self_attn.q.lora_B.weight": B,
        "blocks.0.self_attn.q.alpha": torch.tensor(16.0),  # alpha = 16
    }
    base_before = model.blocks[0].self_attn.q.weight.data.clone()

    # Default rank=8 (from A.shape[0]); alpha=16 → scale = 16/8 = 2.0.
    # If our merge ignores the alpha key, it would use alpha=rank → scale=1.0.
    merge_lora_into_base(model, lora_sd, strength=1.0, rank=n)

    expected = base_before + 2.0 * (B @ A)
    actual = model.blocks[0].self_attn.q.weight.data
    assert torch.allclose(actual, expected, atol=1e-5), (
        f"Expected per-key alpha=16 to override default; "
        f"max diff {(actual - expected).abs().max().item():.5f}"
    )
