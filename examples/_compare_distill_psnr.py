"""Tier C4: PSNR / max-diff / mean-diff comparison of step-distill
outputs against the BF16 reference, per modality.

Loads two sets of per-modality PNGs from ComfyUI/output and reports
PSNR (against the BF16 baseline) per modality.

Reference set: unividx_LTX_R2AIN_FP8VAL_BF16_*  (0.4.0-rc1's clean
BF16 baseline — no sage, 20 steps, cfg=5).

Distill set: unividx_smoke_distill_*  (current run — BF16 + distill,
4 steps, cfg=1).

Useful followup: drop in different distill_strength runs (0.5, 0.75,
1.0, 1.25, 1.5) and see how PSNR scales with strength to find the
sweet spot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT = Path("C:/Users/dr5090/Documents/ComfyUI/output")
MODALITIES = ("placeholder", "albedo", "irradiance", "normal")


def load_tensor(prefix: str, modality: str) -> np.ndarray | None:
    paths = sorted(OUTPUT.glob(f"{prefix}*{modality}*.png"))
    if not paths:
        return None
    arrs = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB"),
                       dtype=np.float32) / 255.0
        arrs.append(a)
    return np.stack(arrs, axis=0)  # [T, H, W, 3]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    mse = ((a[:n] - b[:n]) ** 2).mean()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compare(ref_prefix: str, test_prefix: str, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"  ref: {ref_prefix}*")
    print(f"  test: {test_prefix}*")
    print(f"{'modality':<14s}  {'n':>3s}  {'PSNR (dB)':>10s}  "
          f"{'max-diff':>9s}  {'mean-diff':>9s}")
    for m in MODALITIES:
        a = load_tensor(ref_prefix, m)
        b = load_tensor(test_prefix, m)
        if a is None or b is None:
            n_a = 0 if a is None else a.shape[0]
            n_b = 0 if b is None else b.shape[0]
            print(f"{m:<14s}  {n_a:>3d}/{n_b:<3d}  {'n/a':>10s}  "
                  f"{'n/a':>9s}  {'n/a':>9s}")
            continue
        p = psnr(a, b)
        n = min(a.shape[0], b.shape[0])
        max_d = float(np.abs(a[:n] - b[:n]).max())
        mean_d = float(np.abs(a[:n] - b[:n]).mean())
        p_str = f"{p:.2f}" if p != float("inf") else "inf"
        print(f"{m:<14s}  {n:>3d}  {p_str:>10s}  "
              f"{max_d:>9.4f}  {mean_d:>9.4f}")


def main() -> int:
    print("Tier C4 / C5 PSNR comparison against BF16 baseline")
    print("(BF16 reference = 0.4.0-rc1 unividx_LTX_R2AIN_FP8VAL_BF16_*)")

    compare(
        ref_prefix="unividx_LTX_R2AIN_FP8VAL_BF16_",
        test_prefix="unividx_smoke_distill_",
        label="C4: BF16 + distill (4 steps cfg=1) vs BF16 baseline (20 steps cfg=5)",
    )

    # Also surface FP8 baseline for comparison (already shipped in 0.4.0).
    compare(
        ref_prefix="unividx_LTX_R2AIN_FP8VAL_BF16_",
        test_prefix="unividx_LTX_R2AIN_FP8VAL_FP8PRE_",
        label="0.4.0-rc1 reference: FP8 (20 steps cfg=5) vs BF16 baseline",
    )

    # If FP8 + distill outputs exist, compare them too.
    fp8_distill = list(OUTPUT.glob("unividx_fp8_distill_*albedo*.png"))
    if fp8_distill:
        compare(
            ref_prefix="unividx_LTX_R2AIN_FP8VAL_BF16_",
            test_prefix="unividx_fp8_distill_",
            label="C5: FP8 + distill (4 steps cfg=1) vs BF16 baseline",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
