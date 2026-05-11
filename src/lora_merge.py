# src/lora_merge.py
"""Tier C: merge a LoRA / distill-patch state-dict into a target model's
base weights.

Supports the union of patterns we've found across the relevant
upstream files:

  PEFT convention:
    <base>.lora_A.weight   shape [rank, in_features]
    <base>.lora_B.weight   shape [out_features, rank]
    delta = strength * (alpha/rank) * (B @ A)

  Kohya / LightX2V convention:
    <base>.lora_down.weight  shape [rank, in_features]   (== A)
    <base>.lora_up.weight    shape [out_features, rank]  (== B)
    delta = strength * (alpha/rank) * (up @ down)

  Bias delta (LightX2V):
    <base>.diff_b            shape [out_features]
    new_bias = old_bias (or 0) + strength * diff_b

  Direct weight delta (LightX2V, for non-Linear modules):
    <base>.diff              shape [...]
    new_weight = old_weight + strength * diff

  Per-key alpha override (some LoRAs):
    <base>.alpha             scalar tensor

Resolution: dotted-path walk on the target model, with PEFT-aware
descent (if a path resolves to a wrapper with a `.base_layer` attr
holding the real nn.Linear, the merge applies to the inner base_layer
so PEFT adapter siblings stay untouched — same B5 contract as the
FP8 loader).

Idempotency: calling merge_lora_into_base with the SAME state_dict
twice doubles the effective strength. Call with strength=0 to verify
no-op. There is no "unmerge" — to roll back, reload the base model.

Used by Tier C to merge LightX2V's step-distill / cfg-distill LoRA
into UniVidX's BF16 base weights BEFORE the FP8 substitution runs.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn

_log = logging.getLogger("unividx")


# ---------------------------------------------------------------------------
# Path resolver — same dotted-path scheme as fp8_loader._resolve_parent
# ---------------------------------------------------------------------------

def _resolve_target(model: nn.Module, dotted: str
                     ) -> Optional[nn.Module]:
    """Walk `model.<a>.<b>.<c>` and return the leaf module, or None."""
    obj: nn.Module = model
    for part in dotted.split("."):
        if part.isdigit():
            try:
                obj = obj[int(part)]  # type: ignore[index]
            except (IndexError, TypeError, KeyError):
                return None
        else:
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
    return obj


def _descend_peft(module: nn.Module) -> nn.Module:
    """If module is a PEFT-style wrapper with .base_layer holding a
    Linear, return the base_layer. Otherwise return module unchanged."""
    if (not isinstance(module, nn.Linear)
            and hasattr(module, "base_layer")
            and isinstance(getattr(module, "base_layer"), nn.Module)):
        return module.base_layer
    return module


# ---------------------------------------------------------------------------
# Key grouping
# ---------------------------------------------------------------------------

# Suffixes that identify the four delta patterns we know how to merge.
_LORA_DOWN_SUFFIXES = (".lora_down.weight",)
_LORA_UP_SUFFIXES = (".lora_up.weight",)
_LORA_A_SUFFIXES = (".lora_A.weight",)
_LORA_B_SUFFIXES = (".lora_B.weight",)
_DIFF_B_SUFFIXES = (".diff_b",)
_DIFF_SUFFIXES = (".diff",)
_ALPHA_SUFFIXES = (".alpha",)


def _strip_suffix(key: str, suffixes: tuple[str, ...]) -> Optional[str]:
    for sfx in suffixes:
        if key.endswith(sfx):
            return key[: -len(sfx)]
    return None


def _group_by_base(state_dict: dict, strip_prefix: str = "") -> dict:
    """Group state_dict keys by their base path (the part before the
    LoRA-suffix). Returns a dict mapping base -> {kind: tensor}."""
    groups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for raw_key, tensor in state_dict.items():
        key = raw_key
        if strip_prefix and key.startswith(strip_prefix):
            key = key[len(strip_prefix):]
        for kind, suffixes in [
            ("lora_down", _LORA_DOWN_SUFFIXES),
            ("lora_up", _LORA_UP_SUFFIXES),
            ("lora_A", _LORA_A_SUFFIXES),
            ("lora_B", _LORA_B_SUFFIXES),
            ("diff_b", _DIFF_B_SUFFIXES),
            ("diff", _DIFF_SUFFIXES),
            ("alpha", _ALPHA_SUFFIXES),
        ]:
            base = _strip_suffix(key, suffixes)
            if base is not None:
                groups[base][kind] = tensor
                break
    return dict(groups)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _apply_rank_decomp(target: nn.Module, A: torch.Tensor, B: torch.Tensor,
                       strength: float, alpha: Optional[float],
                       rank: Optional[int]) -> bool:
    """Apply W += scale * (B @ A) to target.weight. Returns True on
    success."""
    if not hasattr(target, "weight") or target.weight is None:
        return False
    W = target.weight
    # Resolve scale = strength * (alpha / rank).
    eff_rank = rank if rank is not None else A.shape[0]
    eff_alpha = alpha if alpha is not None else float(eff_rank)
    scale = strength * (eff_alpha / max(1, eff_rank))
    # Compute in float32 to avoid BF16 rounding eating the delta on
    # small-magnitude updates. Cast back to W's dtype on assignment.
    delta = (B.float() @ A.float()) * scale
    W.data.add_(delta.to(W.dtype).to(W.device))
    return True


def _apply_bias_diff(target: nn.Module, diff_b: torch.Tensor,
                     strength: float) -> bool:
    """Apply b += strength * diff_b. Creates the bias Parameter if
    target.bias is currently None (LightX2V adds biases to some
    Wan2.1 Linears that were trained without them)."""
    if not isinstance(target, nn.Module):
        return False
    delta = diff_b * strength
    if getattr(target, "bias", None) is None:
        # Create a new Parameter on the same device/dtype as the weight
        # if one exists, else CPU bf16.
        if hasattr(target, "weight") and target.weight is not None:
            dtype = target.weight.dtype
            device = target.weight.device
        else:
            dtype = torch.bfloat16
            device = torch.device("cpu")
        target.bias = nn.Parameter(
            delta.to(dtype).to(device),
            requires_grad=False,
        )
    else:
        target.bias.data.add_(delta.to(target.bias.dtype).to(target.bias.device))
    return True


def _apply_weight_diff(target: nn.Module, diff: torch.Tensor,
                       strength: float) -> bool:
    """Apply w += strength * diff for non-Linear modules with .weight
    (RMSNorm, patch_embedding Conv3d, head Linear)."""
    if not hasattr(target, "weight") or target.weight is None:
        return False
    delta = diff * strength
    target.weight.data.add_(delta.to(target.weight.dtype).to(target.weight.device))
    return True


def merge_lora_into_base(
    model: nn.Module,
    state_dict: dict,
    *,
    strength: float = 1.0,
    alpha: Optional[float] = None,
    rank: Optional[int] = None,
    strip_prefix: str = "",
) -> dict:
    """Merge a LoRA / distill-patch state-dict into the base weights of
    ``model`` in place.

    See the module docstring for the four key-pattern conventions
    supported. Returns a diagnostic report:

        {
            "merged":          int,   # rank-decomp deltas applied
            "biases_patched":  int,   # diff_b applications
            "weights_patched": int,   # diff applications (non-Linear)
            "unmatched":       int,   # bases with no resolvable target
            "skipped":         int,   # bases with mismatched key sets
        }

    Use strength=0.0 to perform a structural dry-run (no-op).
    """
    groups = _group_by_base(state_dict, strip_prefix=strip_prefix)

    merged = 0
    biases_patched = 0
    weights_patched = 0
    unmatched = 0
    skipped = 0

    for base, kinds in groups.items():
        target = _resolve_target(model, base)
        if target is None:
            unmatched += 1
            _log.debug("lora_merge: no target for %s", base)
            continue
        target = _descend_peft(target)

        # Per-key alpha override
        per_key_alpha = None
        if "alpha" in kinds:
            a_t = kinds["alpha"]
            try:
                per_key_alpha = float(a_t.item() if a_t.numel() == 1 else a_t.flatten()[0].item())
            except Exception:
                per_key_alpha = None
        effective_alpha = (per_key_alpha if per_key_alpha is not None
                            else alpha)

        # Rank-decomposed delta. Prefer PEFT naming if present, else
        # Kohya. Skip if only one half of the pair is present.
        applied_rank_decomp = False
        if "lora_A" in kinds and "lora_B" in kinds:
            A = kinds["lora_A"]
            B = kinds["lora_B"]
            eff_rank = rank if rank is not None else A.shape[0]
            if _apply_rank_decomp(target, A, B, strength,
                                   effective_alpha, eff_rank):
                merged += 1
                applied_rank_decomp = True
            else:
                skipped += 1
        elif "lora_down" in kinds and "lora_up" in kinds:
            A = kinds["lora_down"]
            B = kinds["lora_up"]
            eff_rank = rank if rank is not None else A.shape[0]
            if _apply_rank_decomp(target, A, B, strength,
                                   effective_alpha, eff_rank):
                merged += 1
                applied_rank_decomp = True
            else:
                skipped += 1
        elif ("lora_A" in kinds) ^ ("lora_B" in kinds):
            skipped += 1
            _log.warning("lora_merge: %s has lora_A xor lora_B; skipping",
                         base)
        elif ("lora_down" in kinds) ^ ("lora_up" in kinds):
            skipped += 1
            _log.warning("lora_merge: %s has lora_down xor lora_up; skipping",
                         base)

        # Bias delta (orthogonal to rank-decomp, both can apply).
        if "diff_b" in kinds:
            if _apply_bias_diff(target, kinds["diff_b"], strength):
                biases_patched += 1
            else:
                skipped += 1

        # Direct weight delta — only if we DIDN'T already apply a
        # rank-decomp delta to the same weight (otherwise we'd
        # double-count).
        if "diff" in kinds and not applied_rank_decomp:
            if _apply_weight_diff(target, kinds["diff"], strength):
                weights_patched += 1
            else:
                skipped += 1

    if unmatched > 0:
        _log.warning(
            "lora_merge: %d base path(s) did not resolve to a module "
            "in the target model (skipped). Examples: %s",
            unmatched,
            ", ".join(list(groups.keys())[:3]),
        )

    return {
        "merged": merged,
        "biases_patched": biases_patched,
        "weights_patched": weights_patched,
        "unmatched": unmatched,
        "skipped": skipped,
    }
