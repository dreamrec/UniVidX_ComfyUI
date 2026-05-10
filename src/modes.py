# src/modes.py
"""
Static mode metadata for UniVidX.

A mode name encodes which modalities are conditions (left of '2') and
which are targets (right of '2'). e.g. RA2IN: RGB+Albedo are conditions,
Irradiance+Normal are targets.

Letter codes:
    Intrinsic family: R=rgb, A=albedo, I=irradiance, N=normal
    Alpha family:     R=composite (com), P=pha, F=fgr, B=bgr
"""
from typing import Dict, List, Set


class InvalidModeError(ValueError):
    pass


# Mapping from intrinsic letter codes to the keys returned by UniVidX's pipe()
_INTRINSIC_KEY = {"R": "rgb", "A": "albedo", "I": "irradiance", "N": "normal_unit"}
# Mapping from intrinsic letter codes to the kwargs accepted by pipe()
_INTRINSIC_INPUT_KEY = {"R": "rgb", "A": "albedo", "I": "irradiance", "N": "normal"}

# Alpha letter codes — composite RGB renamed to lowercase keys returned by pipe()
_ALPHA_KEY = {"R": "rgb", "P": "pha", "F": "fgr", "B": "bgr"}
_ALPHA_INPUT_KEY = {"R": "rgb", "P": "pha", "F": "fgr", "B": "bgr"}


def _parse(mode: str):
    """Split 'RA2IN' into (['R','A'], ['I','N']) or 't2RAIN' into ([], ['R','A','I','N'])."""
    if "2" not in mode:
        raise InvalidModeError(f"Invalid mode: {mode}")
    left, right = mode.split("2", 1)
    if left.lower() == "t":
        # text-to-everything: no conditions
        return [], list(right)
    return list(left), list(right)


def _enumerate_modes(letters: str):
    """All non-empty strict subsets of `letters` for conditions, plus the t2 mode."""
    from itertools import combinations
    modes = [f"t2{letters}"]
    n = len(letters)
    # Conditions can be 1..n-1 letters; targets are the complement
    for k in range(1, n):
        for combo in combinations(letters, k):
            cond = "".join(combo)
            tgt = "".join(c for c in letters if c not in combo)
            modes.append(f"{cond}2{tgt}")
    return modes


INTRINSIC_MODES: List[str] = _enumerate_modes("RAIN")
ALPHA_MODES: List[str] = _enumerate_modes("RPFB")


def _family(mode: str) -> str:
    if mode in INTRINSIC_MODES:
        return "intrinsic"
    if mode in ALPHA_MODES:
        return "alpha"
    raise InvalidModeError(f"Unknown mode: {mode}")


def required_inputs(mode: str) -> Set[str]:
    """Return the set of input modality names this mode requires (pipe kwargs)."""
    family = _family(mode)
    cond_letters, _ = _parse(mode)
    if family == "intrinsic":
        return {_INTRINSIC_INPUT_KEY[c] for c in cond_letters}
    return {_ALPHA_INPUT_KEY[c] for c in cond_letters}


def output_keys(mode: str) -> List[str]:
    """Return the list of dict keys pipe() will return for this mode, in canonical order."""
    family = _family(mode)
    _, tgt_letters = _parse(mode)
    if family == "intrinsic":
        return [_INTRINSIC_KEY[c] for c in tgt_letters]
    return [_ALPHA_KEY[c] for c in tgt_letters]


def family_of(mode: str) -> str:
    """Return 'intrinsic' or 'alpha'."""
    return _family(mode)


def validate_mode(mode: str, supplied_inputs: Set[str]) -> None:
    """Raise if any required input is missing."""
    needed = required_inputs(mode)
    missing = needed - supplied_inputs
    if missing:
        raise ValueError(
            f"Mode {mode} requires inputs {sorted(needed)}, "
            f"missing: {sorted(missing)}"
        )
