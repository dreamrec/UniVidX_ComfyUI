"""Regression tests for the `drop_out` kwarg + the variant-DiT patch
selection.

Background: UniVidX's vendored `wan_video_dit_intrinsic.py` and
`wan_video_dit_alpha.py` both define
    flash_attention(q, k, v, num_heads, compatibility_mode=False, drop_out=None)
and call it as `flash_attention(..., drop_out=drop_out)` from the
attention modules. The `drop_out` value gates a CMSA cross-batch K/V
reshape (it's a routing threshold, not a probability).

Two correctness invariants:
1. Our wrapper's signature MUST accept `drop_out` so that any future
   patch of the variant-specific modules doesn't crash with TypeError
   on the kwarg.
2. The wrapper installation MUST NOT patch the variant-specific modules
   today — until we have a numerical-equivalence test for sage vs
   SDPA on the CMSA reshape pattern, those CMSA paths need to keep
   the original SDPA implementation. Only the base `wan_video_dit`
   (no drop_out semantics) and DiffSynth's `wan_video_dit` get patched.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


HERE = __file__
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(HERE), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch


@pytest.fixture(autouse=True)
def _reset_warn_dedup():
    from src import runtime
    runtime._attention_fallback_warned.clear()
    yield
    runtime._attention_fallback_warned.clear()


def _install_fake_modules(monkeypatch):
    """Install fake sageattention + a stack of fake wan_video_dit
    modules: the base + both CMSA variants."""
    fake_sage_mod = types.ModuleType("sageattention")
    fake_sage_mod.sageattn = lambda q, k, v, **kwargs: q
    monkeypatch.setitem(sys.modules, "sageattention", fake_sage_mod)

    parent = types.ModuleType("diffsynth")
    models = types.ModuleType("diffsynth.models")

    base = types.ModuleType("diffsynth.models.wan_video_dit")
    base.SAGE_ATTN_AVAILABLE = True
    base.FLASH_ATTN_2_AVAILABLE = False
    base.flash_attention = MagicMock(name="diffsynth_flash_attention")

    models.wan_video_dit = base
    parent.models = models

    monkeypatch.setitem(sys.modules, "diffsynth", parent)
    monkeypatch.setitem(sys.modules, "diffsynth.models", models)
    monkeypatch.setitem(sys.modules, "diffsynth.models.wan_video_dit", base)

    # UniVidX vendored modules: a base (no drop_out) and two CMSA variants
    # (with drop_out semantics) that the wrapper must intentionally leave
    # alone.
    sentinel_intrinsic = MagicMock(name="vendor_intrinsic_flash_attention")
    sentinel_alpha = MagicMock(name="vendor_alpha_flash_attention")
    sentinel_base = MagicMock(name="vendor_base_flash_attention")

    vendor_base = types.ModuleType("src.models.wan_video_dit")
    vendor_base.flash_attention = sentinel_base

    vendor_intrinsic = types.ModuleType("src.models.wan_video_dit_intrinsic")
    vendor_intrinsic.flash_attention = sentinel_intrinsic

    vendor_alpha = types.ModuleType("src.models.wan_video_dit_alpha")
    vendor_alpha.flash_attention = sentinel_alpha

    monkeypatch.setitem(sys.modules, "src.models.wan_video_dit", vendor_base)
    monkeypatch.setitem(sys.modules, "src.models.wan_video_dit_intrinsic", vendor_intrinsic)
    monkeypatch.setitem(sys.modules, "src.models.wan_video_dit_alpha", vendor_alpha)

    return {
        "diffsynth_base": base,
        "vendor_base": vendor_base,
        "vendor_intrinsic": vendor_intrinsic,
        "vendor_alpha": vendor_alpha,
        "sentinel_intrinsic": sentinel_intrinsic,
        "sentinel_alpha": sentinel_alpha,
    }


def test_force_sage_does_not_patch_cmsa_variant_modules(monkeypatch):
    """The variant-specific CMSA modules (intrinsic / alpha) MUST NOT
    be touched by the sage patch — they have drop_out cross-batch
    semantics that haven't been validated against sage numerically."""
    fakes = _install_fake_modules(monkeypatch)

    from src.runtime import _force_sage_over_fa2
    assert _force_sage_over_fa2() is True

    # Base DiffSynth module: patched (sentinel was a MagicMock instance,
    # the wrapper is a function).
    assert not isinstance(fakes["diffsynth_base"].flash_attention, MagicMock)

    # Vendored base module: also patched.
    assert not isinstance(fakes["vendor_base"].flash_attention, MagicMock)

    # CMSA variant modules: MUST still be the original sentinel.
    assert fakes["vendor_intrinsic"].flash_attention is fakes["sentinel_intrinsic"]
    assert fakes["vendor_alpha"].flash_attention is fakes["sentinel_alpha"]


def test_wrapper_signature_accepts_drop_out(monkeypatch):
    """If a future upstream rev does route a drop_out=... call through
    our wrapper (e.g. via a base-class call that inherits from the CMSA
    variants), the wrapper must accept the kwarg without raising
    TypeError."""
    fakes = _install_fake_modules(monkeypatch)

    from src.runtime import _force_sage_over_fa2
    _force_sage_over_fa2()
    wrapper = fakes["diffsynth_base"].flash_attention

    num_heads = 4
    head_dim = 128
    q = torch.randn(1, 8, num_heads * head_dim)
    k = torch.randn(1, 8, num_heads * head_dim)
    v = torch.randn(1, 8, num_heads * head_dim)

    # No drop_out — current expected call shape.
    out = wrapper(q, k, v, num_heads=num_heads)
    assert out.shape == q.shape

    # drop_out=None (the upstream default).
    out = wrapper(q, k, v, num_heads=num_heads, drop_out=None)
    assert out.shape == q.shape

    # drop_out=0 (typical inference value).
    out = wrapper(q, k, v, num_heads=num_heads, drop_out=0)
    assert out.shape == q.shape

    # Unrecognised kwargs from a future upstream rev: also accepted
    # (the wrapper has **_kwargs).
    out = wrapper(q, k, v, num_heads=num_heads, drop_out=0, future_arg="foo")
    assert out.shape == q.shape
