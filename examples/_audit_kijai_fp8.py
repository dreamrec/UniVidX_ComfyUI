"""Tier B1 audit: inspect Kijai's FP8 Wan safetensors to learn the
key-prefix convention, dtype layout, and scale-tensor presence.

Why this script: ROADMAP_v0.3.md's Tier B1 calls for an audit of
`Wan2_1-T2V-14B_fp8_e4m3fn.safetensors`. That file isn't on disk
yet (~14 GB download pending), but we have several Wan2.x Kijai FP8
files locally — they follow the same naming convention and the same
structural template, so we can learn the format here and confirm
key-by-key once the Wan2.1 file lands.

Outputs:
- Total tensor count + cumulative size.
- Dtype histogram (float8_e4m3fn / float8_e5m2 / bfloat16 / float32 /
  uint8 etc.) with cumulative bytes per dtype — tells us how the file
  splits between FP8 weights vs full-precision norms/embeds.
- Sample of weight-key suffixes and prefixes (head + tail of sorted
  keys) — tells us whether keys are bare `blocks.0.self_attn.q.weight`
  or prefixed `diffusion_model.blocks.0...` or `dit.blocks.0...`.
- Detect scale tensors: keys matching `*scale_weight*` or
  `*weight_scale*` or `*.scale*` — tells us whether the file ships
  per-tensor or per-channel scales (and whether they're stored as
  separate keys or fused into a wrapper tensor).
- Sample 5 random FP8 weight tensors: shape + dtype + (if scale
  exists) corresponding scale shape — gives us the per-tensor vs
  per-channel scale shape distinction.

Usage:
    python examples/_audit_kijai_fp8.py <path-to-kijai-fp8.safetensors>

Or with no arg, picks the first local Wan2.x FP8 file under
ComfyUI/models/diffusion_models/ as a stand-in.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import safetensors.torch as st
from safetensors import safe_open

DEFAULT_SEARCH_ROOT = Path("C:/Users/dr5090/Documents/ComfyUI/models/diffusion_models")


def _pick_stand_in() -> Path | None:
    if not DEFAULT_SEARCH_ROOT.is_dir():
        return None
    for p in sorted(DEFAULT_SEARCH_ROOT.glob("wan2*fp8*.safetensors")):
        if p.is_file() and p.stat().st_size > 1_000_000_000:  # > 1 GB
            return p
    return None


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def audit(path: Path) -> dict:
    print(f"\n=== auditing {path.name} ===")
    print(f"file size: {_human_bytes(path.stat().st_size)}")

    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}

        # Dtype histogram + cumulative bytes per dtype.
        dtype_count: Counter = Counter()
        dtype_bytes: defaultdict[str, int] = defaultdict(int)
        scale_keys: list[str] = []
        fp8_weight_keys: list[str] = []
        sample_fp8: list[tuple[str, list[int], str]] = []

        for k in keys:
            slice_ = f.get_slice(k)
            dtype_str = str(slice_.get_dtype())
            shape = slice_.get_shape()
            # bytes per element: try to infer from dtype string. Safetensors
            # exposes get_dtype() as a string like "F8_E4M3" or "BF16".
            # rough byte-width map for the dtypes we expect.
            elem_bytes = {
                "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1,
                "F16": 2, "BF16": 2, "I16": 2,
                "F32": 4, "I32": 4,
                "F64": 8, "I64": 8,
            }.get(dtype_str, 0)
            n_elem = 1
            for d in shape:
                n_elem *= d
            dtype_count[dtype_str] += 1
            dtype_bytes[dtype_str] += elem_bytes * n_elem

            if any(s in k.lower() for s in ("scale_weight", "weight_scale",
                                            ".scale", "scale_input")):
                scale_keys.append(k)
            if dtype_str.startswith("F8"):
                fp8_weight_keys.append(k)

        if fp8_weight_keys:
            for k in fp8_weight_keys[:5]:
                slice_ = f.get_slice(k)
                sample_fp8.append((k, list(slice_.get_shape()),
                                   str(slice_.get_dtype())))

    print(f"\ntotal tensors: {len(keys)}")
    print(f"metadata: {json.dumps(metadata, indent=2) if metadata else '(none)'}")

    print("\ndtype histogram:")
    for dt, n in dtype_count.most_common():
        sz = _human_bytes(dtype_bytes.get(dt, 0))
        print(f"  {dt:>10s}: {n:>5d} tensors, {sz:>10s}")

    print(f"\nFP8 weight tensors: {len(fp8_weight_keys)}")
    print(f"scale-like keys:    {len(scale_keys)}")

    print("\nkey prefixes (first 5 sorted):")
    for k in sorted(keys)[:5]:
        print(f"  {k}")
    print("key prefixes (last 5 sorted):")
    for k in sorted(keys)[-5:]:
        print(f"  {k}")

    if scale_keys:
        print("\nfirst 10 scale-like keys:")
        for k in scale_keys[:10]:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                slice_ = f.get_slice(k)
                shape = list(slice_.get_shape())
                dtype = str(slice_.get_dtype())
            print(f"  {k}  shape={shape}  dtype={dtype}")

    if sample_fp8:
        print("\nsample FP8 weight tensors:")
        for k, shape, dtype in sample_fp8:
            # Try to find a matching scale by common naming patterns.
            scale_candidate = None
            for pattern in (f"{k}.scale", f"{k}_scale",
                            k.replace(".weight", ".scale_weight"),
                            k.replace(".weight", ".weight_scale")):
                if pattern in keys:
                    scale_candidate = pattern
                    break
            print(f"  {k}  shape={shape}  dtype={dtype}"
                  f"  scale={scale_candidate or '(none found)'}")

    # Block-prefix detection: are keys flat (blocks.0...) or prefixed
    # (diffusion_model.blocks.0... / dit.blocks.0...)?
    print("\nblock-prefix detection:")
    has_bare = any(k.startswith("blocks.") for k in keys)
    has_diffmodel = any(k.startswith("diffusion_model.") for k in keys)
    has_dit = any(k.startswith("dit.") for k in keys)
    has_model = any(k.startswith("model.") for k in keys)
    print(f"  starts with 'blocks.':           {has_bare}")
    print(f"  starts with 'diffusion_model.':  {has_diffmodel}")
    print(f"  starts with 'dit.':              {has_dit}")
    print(f"  starts with 'model.':            {has_model}")

    return {
        "path": str(path),
        "total_tensors": len(keys),
        "fp8_count": len(fp8_weight_keys),
        "scale_count": len(scale_keys),
        "dtype_count": dict(dtype_count),
        "dtype_bytes": {k: int(v) for k, v in dtype_bytes.items()},
        "has_bare": has_bare,
        "has_diffmodel": has_diffmodel,
        "has_dit": has_dit,
        "has_model": has_model,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    help="Path to a Kijai FP8 .safetensors file. If omitted, "
                         "picks a Wan2.x stand-in from models/diffusion_models/.")
    args = ap.parse_args()

    if args.path:
        path = Path(args.path)
    else:
        path = _pick_stand_in()
        if path is None:
            print("No stand-in found and no path given.", file=sys.stderr)
            return 2
        print(f"(no path arg given, using stand-in: {path.name})")

    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1
    audit(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
