# src/fp8_loader.py
"""Tier B2 + B6: Pre-quantized FP8 weight loading + dequantize-on-forward.

Two pieces here:

1. ``FP8Linear`` — drop-in replacement for ``nn.Linear`` that stores
   its weight as ``torch.float8_e4m3fn`` and dequantizes on forward
   via Kijai-style scalar ``scale_weight``. The forward path computes
   ``F.linear(x, weight.to(x.dtype) * scale_weight.to(x.dtype), bias)``
   — Phase 1 design (B6). Phase 2 (B7) will replace this with
   ``torch._scaled_mm`` for the same numerics with ~2-3x throughput.

2. ``triage_kijai_state_dict`` + ``load_fp8_state_dict_into`` — split a
   Kijai-style state_dict into FP8 weights / scale_weight / scale_input /
   full-precision aux keys (B2), then walk a target ``nn.Module``
   (UniVidX's ``WanModel``), replace matching ``nn.Linear`` modules with
   ``FP8Linear`` instances carrying the FP8 weight + scales, and load
   remaining aux tensors via standard state-dict semantics.

This module is the foundation B3 ("alternate DiT loader") builds on:
B3 constructs an empty ``WanModel``, calls ``load_fp8_state_dict_into``
to FP8-ify the Linear modules, then hands the model to UniVidX's
``add_multiple_loras_to_model()`` which wraps each ``FP8Linear`` with
PEFT adapter layers (LoRA stays BF16; FP8 base + BF16 LoRA is the
correctness recipe).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger("unividx")


class FP8Linear(nn.Module):
    """``nn.Linear`` lookalike with FP8 weight storage + dequantize-on-
    forward.

    Storage:
        weight        : Parameter, dtype=float8_e4m3fn, shape [out, in]
                         (requires_grad=False; we never train this)
        scale_weight  : Buffer, dtype=float32, shape [1]
        scale_input   : Buffer, dtype=float32, shape [1]  (Phase 1 unused)
        bias          : Parameter, dtype=bfloat16, shape [out]  (optional)

    Forward:
        Phase 1 — dequant-then-matmul:
            w_dq = weight.to(x.dtype) * scale_weight.to(x.dtype)
            return F.linear(x, w_dq, bias)

        Phase 2 (B7) will replace this body with torch._scaled_mm.

    Memory: ~50% of an equivalent BF16 ``nn.Linear`` (weight 1 byte/elem
    vs 2). Per-step throughput in Phase 1 matches BF16 baseline (the
    dequant cost is hidden behind the matmul). The headline win is host
    RAM + GPU residency, not per-step speed; Phase 2 unlocks the per-step
    win.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features,
                        dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        # Scales held as [1] tensors so the forward path doesn't have to
        # case 0-dim vs shape-[1] (Kijai files ship 0-dim scalars).
        self.register_buffer("scale_weight",
                             torch.ones(1, dtype=torch.float32))
        self.register_buffer("scale_input",
                             torch.ones(1, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dq = self.weight.to(x.dtype) * self.scale_weight.to(x.dtype)
        # F.linear requires bias to share the matmul output dtype. The
        # FP8 weight is stored once and reused across BF16/FP16
        # compute calls, but bias is typically stored in BF16 — cast
        # it on the hot path so a downstream FP16 compute doesn't
        # error with "self and mat2 must have the same dtype."
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_dq, b)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"weight_dtype={self.weight.dtype}")


# ---------------------------------------------------------------------------
# State-dict triage
# ---------------------------------------------------------------------------

def triage_kijai_state_dict(
    state_dict: dict,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split a Kijai-style state_dict into four buckets:

    - fp8_weight_keys : keys whose tensor dtype is float8_e4m3fn AND
                       whose name ends in ``.weight`` (the quantized
                       Linear weights — 407 of these in Wan 14B FP8).
    - scale_weight_keys : keys ending in ``.scale_weight`` (the F32
                          scalar per-tensor weight scales).
    - scale_input_keys  : keys ending in ``.scale_input`` (Kijai's
                          pre-baked activation scales — Phase 2 will
                          use these for ``torch._scaled_mm``; Phase 1
                          ignores them).
    - fp_aux_keys     : everything else (biases, norms, embeddings,
                       head — kept in BF16/F32 per Kijai's quantization
                       policy).
    """
    fp8_weight_keys: list[str] = []
    scale_weight_keys: list[str] = []
    scale_input_keys: list[str] = []
    fp_aux_keys: list[str] = []

    for key, tensor in state_dict.items():
        if key.endswith(".scale_weight"):
            scale_weight_keys.append(key)
        elif key.endswith(".scale_input"):
            scale_input_keys.append(key)
        elif (key.endswith(".weight")
              and tensor.dtype == torch.float8_e4m3fn):
            fp8_weight_keys.append(key)
        else:
            fp_aux_keys.append(key)

    return fp8_weight_keys, scale_weight_keys, scale_input_keys, fp_aux_keys


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _resolve_parent(model: nn.Module,
                    dotted_path: str) -> tuple[Optional[nn.Module], Optional[str]]:
    """Walk a dotted attribute path on ``model`` and return
    ``(parent_module, leaf_name)`` so the caller can ``setattr``.

    Handles integer indices for ``nn.ModuleList`` (e.g.
    ``blocks.0.cross_attn.q`` → ``model.blocks[0].cross_attn``, ``"q"``).
    Returns ``(None, None)`` if the path doesn't resolve."""
    parts = dotted_path.split(".")
    obj: nn.Module = model
    for part in parts[:-1]:
        if part.isdigit():
            try:
                obj = obj[int(part)]  # type: ignore[index]
            except (IndexError, TypeError, KeyError):
                return None, None
        else:
            if not hasattr(obj, part):
                return None, None
            obj = getattr(obj, part)
    return obj, parts[-1]


def _set_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    """Setattr that works for both nn.Module attributes and
    nn.ModuleList indices."""
    if name.isdigit():
        parent[int(name)] = child  # type: ignore[index]
    else:
        setattr(parent, name, child)


def load_fp8_state_dict_into(model: nn.Module, state_dict: dict) -> dict:
    """Mutate ``model`` so its ``nn.Linear`` modules (whose names match
    FP8 weight keys in ``state_dict``) are replaced with ``FP8Linear``
    instances carrying the FP8 weight + scale_weight + scale_input + bias.
    Remaining aux keys are loaded via ``model.load_state_dict(strict=False)``.

    Returns a diagnostic report dict:
        {
            "fp8_linears_replaced": int,
            "aux_keys_loaded":      int,
            "unmatched_keys":       list[str],
        }
    """
    fp8_keys, sw_keys, si_keys, aux_keys = triage_kijai_state_dict(state_dict)

    # Index scales by their parent prefix so we can pair each FP8 weight
    # to its (scale_weight, scale_input) tensors in O(1).
    sw_by_prefix: dict[str, str] = {
        k.rsplit(".scale_weight", 1)[0]: k for k in sw_keys
    }
    si_by_prefix: dict[str, str] = {
        k.rsplit(".scale_input", 1)[0]: k for k in si_keys
    }

    unmatched: list[str] = []
    fp8_replaced = 0
    fp8_bias_keys: set[str] = set()  # biases we'll handle via FP8Linear

    for fp8_key in fp8_keys:
        prefix = fp8_key.rsplit(".weight", 1)[0]
        parent, leaf = _resolve_parent(model, prefix)
        if parent is None or leaf is None:
            unmatched.append(fp8_key)
            continue

        old = getattr(parent, leaf, None)
        if old is None and leaf.isdigit():
            try:
                old = parent[int(leaf)]  # type: ignore[index]
            except (IndexError, TypeError):
                pass

        # PEFT descent: after UniVidX wires per-modality LoRAs via
        # add_multiple_loras_to_model(), each target Linear is wrapped
        # in peft.tuners.lora.Linear (or similar) holding the real
        # nn.Linear at `.base_layer`. We descend into the wrapper and
        # replace the BASE layer in-place; the LoRA adapters above
        # stay untouched (B5: LoRA stays BF16).
        peft_wrapper: Optional[nn.Module] = None
        if (not isinstance(old, nn.Linear)
                and not isinstance(old, FP8Linear)
                and hasattr(old, "base_layer")
                and isinstance(getattr(old, "base_layer"),
                               (nn.Linear, FP8Linear))):
            peft_wrapper = old
            old = old.base_layer

        if not isinstance(old, nn.Linear):
            # Already replaced? Idempotent: load into existing FP8Linear.
            if isinstance(old, FP8Linear):
                new = old
            else:
                unmatched.append(fp8_key)
                continue
        else:
            device = old.weight.device
            new = FP8Linear(
                in_features=old.in_features,
                out_features=old.out_features,
                bias=(old.bias is not None),
            ).to(device)
            if peft_wrapper is not None:
                peft_wrapper.base_layer = new
            else:
                _set_child(parent, leaf, new)

        device = new.weight.device
        new.weight.data = state_dict[fp8_key].to(device)
        if prefix in sw_by_prefix:
            sv = state_dict[sw_by_prefix[prefix]]
            if sv.dim() == 0:
                sv = sv.unsqueeze(0)
            new.scale_weight.data = sv.to(device).to(torch.float32)
        if prefix in si_by_prefix:
            sv = state_dict[si_by_prefix[prefix]]
            if sv.dim() == 0:
                sv = sv.unsqueeze(0)
            new.scale_input.data = sv.to(device).to(torch.float32)

        bias_key = f"{prefix}.bias"
        if bias_key in state_dict:
            fp8_bias_keys.add(bias_key)
            if new.bias is not None:
                new.bias.data = state_dict[bias_key].to(device).to(
                    new.bias.dtype
                )

        fp8_replaced += 1

    # Remaining aux tensors (norms, embeddings, head, and any biases
    # not consumed by an FP8Linear) go through the standard path.
    aux_sd = {k: state_dict[k] for k in aux_keys if k not in fp8_bias_keys}
    aux_keys_loaded = 0
    if aux_sd:
        result = model.load_state_dict(aux_sd, strict=False)
        aux_keys_loaded = len(aux_sd) - len(result.unexpected_keys)
        unmatched.extend(result.unexpected_keys)

    if unmatched:
        _log.warning(
            "FP8 loader: %d unmatched key(s); first 5: %s",
            len(unmatched), unmatched[:5],
        )

    return {
        "fp8_linears_replaced": fp8_replaced,
        "aux_keys_loaded": aux_keys_loaded,
        "unmatched_keys": unmatched,
    }
