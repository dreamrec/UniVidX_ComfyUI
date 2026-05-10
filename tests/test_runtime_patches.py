"""Mock-based unit tests for the monkey-patch helpers in src.runtime.

No GPU or model load required — we substitute fakes for sageattention,
torch.nn.functional, and the diffsynth/UniVidX wan_video_dit modules.

What's covered:
- _restore_native_sdpa_if_polluted: detects pollution, restores native,
  is idempotent, returns True/False correctly.
- _force_sage_over_fa2: returns False when sage missing, returns False
  when wan_video_dit missing, installs the wrapper when both present,
  the wrapper falls back FA2->SDPA on sage exceptions, the wrapper
  pre-skips sage/FA2 for head_dim>256.
- _warn_attention_fallback: dedupes by (backend, head_dim, exc-type).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


# Make `from src.runtime import ...` resolve when tests are run from repo root.
HERE = __file__
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(HERE), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # required by runtime.py at import time


@pytest.fixture(autouse=True)
def _reset_warn_dedup():
    """Clear the per-process dedup map between tests so warning-emit
    behavior is deterministic."""
    from src import runtime
    runtime._attention_fallback_warned.clear()
    yield
    runtime._attention_fallback_warned.clear()


# ---------------------------------------------------------------------------
# _restore_native_sdpa_if_polluted
# ---------------------------------------------------------------------------

def test_restore_native_sdpa_no_op_when_unpolluted():
    """If F.SDPA is already torch._C._nn.SDPA, the call returns False."""
    import torch.nn.functional as F
    from src.runtime import _restore_native_sdpa_if_polluted

    # Make sure we're starting clean.
    F.scaled_dot_product_attention = torch._C._nn.scaled_dot_product_attention

    assert _restore_native_sdpa_if_polluted() is False
    assert F.scaled_dot_product_attention is torch._C._nn.scaled_dot_product_attention


def test_restore_native_sdpa_undoes_pollution():
    """If F.SDPA has been swapped to a different callable (simulating
    Stable3DGen's `F.scaled_dot_product_attention = sageattn`), the
    restore swaps it back to the C++ implementation."""
    import torch.nn.functional as F
    from src.runtime import _restore_native_sdpa_if_polluted

    fake_sage = lambda q, k, v, **_: None  # noqa: E731
    F.scaled_dot_product_attention = fake_sage
    assert F.scaled_dot_product_attention is fake_sage  # pollution confirmed

    assert _restore_native_sdpa_if_polluted() is True
    assert F.scaled_dot_product_attention is torch._C._nn.scaled_dot_product_attention


def test_restore_native_sdpa_idempotent():
    """Calling the restore twice in a row returns False on the second call."""
    import torch.nn.functional as F
    from src.runtime import _restore_native_sdpa_if_polluted

    fake_sage = lambda q, k, v, **_: None  # noqa: E731
    F.scaled_dot_product_attention = fake_sage

    assert _restore_native_sdpa_if_polluted() is True
    assert _restore_native_sdpa_if_polluted() is False


# ---------------------------------------------------------------------------
# _force_sage_over_fa2
# ---------------------------------------------------------------------------

def test_force_sage_returns_false_when_sageattention_missing(monkeypatch):
    """When sageattention can't be imported, the function returns False
    and does NOT install any wrapper."""
    monkeypatch.setitem(sys.modules, "sageattention", None)
    from src.runtime import _force_sage_over_fa2
    assert _force_sage_over_fa2() is False


def test_force_sage_returns_false_when_wan_dit_module_missing(monkeypatch):
    """When diffsynth's wan_video_dit isn't importable, returns False."""
    # Provide a fake sageattention that imports cleanly.
    fake_sage_mod = types.ModuleType("sageattention")
    fake_sage_mod.sageattn = lambda q, k, v, **_: q  # identity
    monkeypatch.setitem(sys.modules, "sageattention", fake_sage_mod)
    # And ensure the diffsynth path raises ImportError.
    monkeypatch.setitem(sys.modules, "diffsynth.models.wan_video_dit", None)

    from src.runtime import _force_sage_over_fa2
    # We patched the import to None which makes `from diffsynth.models import
    # wan_video_dit` raise ImportError when accessing the attribute. Either
    # way the function returns False.
    result = _force_sage_over_fa2()
    assert result is False


def test_force_sage_installs_wrapper_when_both_present(monkeypatch):
    """Happy path: both sage and wan_video_dit available, the function
    swaps in our wrapper."""
    # Stub sageattention with a callable identity-ish sageattn.
    fake_sage_mod = types.ModuleType("sageattention")

    def fake_sageattn(q, k, v, **kwargs):
        return q  # identity for shape check
    fake_sage_mod.sageattn = fake_sageattn
    monkeypatch.setitem(sys.modules, "sageattention", fake_sage_mod)

    # Stub diffsynth.models.wan_video_dit with the SAGE_ATTN_AVAILABLE flag.
    parent = types.ModuleType("diffsynth")
    models = types.ModuleType("diffsynth.models")
    wan_dit = types.ModuleType("diffsynth.models.wan_video_dit")
    wan_dit.SAGE_ATTN_AVAILABLE = True
    wan_dit.FLASH_ATTN_2_AVAILABLE = False  # avoid FA2 import path
    original_flash_attention = MagicMock(name="original_flash_attention")
    wan_dit.flash_attention = original_flash_attention
    models.wan_video_dit = wan_dit
    parent.models = models

    monkeypatch.setitem(sys.modules, "diffsynth", parent)
    monkeypatch.setitem(sys.modules, "diffsynth.models", models)
    monkeypatch.setitem(sys.modules, "diffsynth.models.wan_video_dit", wan_dit)

    from src.runtime import _force_sage_over_fa2
    assert _force_sage_over_fa2() is True
    # Wrapper installed in the diffsynth module.
    assert wan_dit.flash_attention is not original_flash_attention


def test_force_sage_wrapper_pre_skips_large_head_dims(monkeypatch):
    """The wrapper should pre-skip sage/FA2 for head_dim > 256 and
    route directly to SDPA — sage and FA2 don't support those sizes."""
    sage_called = MagicMock()

    fake_sage_mod = types.ModuleType("sageattention")

    def fake_sageattn(q, k, v, **kwargs):
        sage_called()
        return q
    fake_sage_mod.sageattn = fake_sageattn
    monkeypatch.setitem(sys.modules, "sageattention", fake_sage_mod)

    parent = types.ModuleType("diffsynth")
    models = types.ModuleType("diffsynth.models")
    wan_dit = types.ModuleType("diffsynth.models.wan_video_dit")
    wan_dit.SAGE_ATTN_AVAILABLE = True
    wan_dit.FLASH_ATTN_2_AVAILABLE = False
    wan_dit.flash_attention = lambda *a, **k: None
    models.wan_video_dit = wan_dit
    parent.models = models
    monkeypatch.setitem(sys.modules, "diffsynth", parent)
    monkeypatch.setitem(sys.modules, "diffsynth.models", models)
    monkeypatch.setitem(sys.modules, "diffsynth.models.wan_video_dit", wan_dit)

    from src.runtime import _force_sage_over_fa2
    _force_sage_over_fa2()

    # Build a Q tensor with head_dim=384 (UniVidX VAE pattern).
    num_heads = 1
    head_dim = 384
    q = torch.randn(1, 16, num_heads * head_dim)
    k = torch.randn(1, 16, num_heads * head_dim)
    v = torch.randn(1, 16, num_heads * head_dim)

    # Call the wrapper — should skip sage entirely since head_dim>256.
    result = wan_dit.flash_attention(q, k, v, num_heads=num_heads)
    assert sage_called.call_count == 0  # sage never invoked
    assert result.shape == q.shape


def test_force_sage_wrapper_falls_back_to_sdpa_on_sage_exception(monkeypatch):
    """When sage raises (e.g. unsupported head_dim), the wrapper should
    swallow the exception, log once, and fall through to SDPA."""
    fake_sage_mod = types.ModuleType("sageattention")

    def raising_sageattn(q, k, v, **kwargs):
        raise ValueError("Unsupported head_dim: 384")
    fake_sage_mod.sageattn = raising_sageattn
    monkeypatch.setitem(sys.modules, "sageattention", fake_sage_mod)

    parent = types.ModuleType("diffsynth")
    models = types.ModuleType("diffsynth.models")
    wan_dit = types.ModuleType("diffsynth.models.wan_video_dit")
    wan_dit.SAGE_ATTN_AVAILABLE = True
    wan_dit.FLASH_ATTN_2_AVAILABLE = False
    wan_dit.flash_attention = lambda *a, **k: None
    models.wan_video_dit = wan_dit
    parent.models = models
    monkeypatch.setitem(sys.modules, "diffsynth", parent)
    monkeypatch.setitem(sys.modules, "diffsynth.models", models)
    monkeypatch.setitem(sys.modules, "diffsynth.models.wan_video_dit", wan_dit)

    from src.runtime import _force_sage_over_fa2, _attention_fallback_warned
    _force_sage_over_fa2()

    # Use head_dim=128 (sage's range, so the pre-skip doesn't catch it).
    num_heads = 4
    head_dim = 128
    q = torch.randn(1, 8, num_heads * head_dim)
    k = torch.randn(1, 8, num_heads * head_dim)
    v = torch.randn(1, 8, num_heads * head_dim)

    result = wan_dit.flash_attention(q, k, v, num_heads=num_heads)
    assert result.shape == q.shape
    # Warning recorded once.
    assert ("sage", head_dim, "ValueError") in _attention_fallback_warned


# ---------------------------------------------------------------------------
# _warn_attention_fallback
# ---------------------------------------------------------------------------

def test_warn_attention_fallback_dedupes_by_tuple():
    """Repeated identical (backend, head_dim, exc-type) calls log once."""
    from src import runtime
    exc = ValueError("test")
    runtime._warn_attention_fallback("sage", 128, exc)
    runtime._warn_attention_fallback("sage", 128, exc)
    runtime._warn_attention_fallback("sage", 128, exc)
    # Different exc-type triggers a new entry.
    runtime._warn_attention_fallback("sage", 128, RuntimeError("other"))
    assert ("sage", 128, "ValueError") in runtime._attention_fallback_warned
    assert ("sage", 128, "RuntimeError") in runtime._attention_fallback_warned
    # 2 entries (one per exc-type), not 4.
    assert len(runtime._attention_fallback_warned) == 2
