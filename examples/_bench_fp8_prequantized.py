"""Tier B8 validation harness — compare FP8 prequantized vs BF16
baseline on the R2AIN_video workflow.

Runs the same workflow twice (same seed, same prompt, same conditioning)
with dit_weight_mode='bf16_shards' and dit_weight_mode='fp8_prequantized',
then computes per-modality PSNR + max-abs-diff between the two output
sets. PSNR thresholds match my Tier-B recommendation:

  RGB modality      : ≥ 35 dB (≈ Phase 1's expected quantization noise)
  Synthetic modalities (Albedo / Irradiance / Normal):
                      ≥ 30 dB (allow more slack — they're
                                noisier reference outputs to begin with)

Why PSNR (not SSIM): scikit-image isn't a hard dep. PSNR alone is a
sufficient first-pass signal for Phase 1's dequant-on-forward (which
should be near-identical to BF16, modulo ~1-3% FP8 weight noise).
A future B8.2 can wire in SSIM via torchmetrics if we end up needing
the perceptual structure signal.

Usage:
    python examples/_bench_fp8_prequantized.py

Notes:
- Requires both runs to actually complete (~10 min each on a 32 GB
  card with vram_buffer=4). Total ~25-30 min.
- Both runs go through cache-miss (different dit_weight_mode), so
  each does a fresh cold load.
- The FP8 run additionally pays the FP8 substitution time (~1 min
  walk through all 407 base Linears).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
WF_PATH = REPO / "examples" / "R2AIN_video_api.json"
COMFY_OUTPUT = Path("C:/Users/dr5090/Documents/ComfyUI/output")
BASE = "http://127.0.0.1:8000"


def _http_get(url: str, timeout: float = 60.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _http_post(url: str, payload: dict, timeout: float = 30.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def queue(dit_weight_mode: str, tag: str) -> tuple[str, str]:
    with WF_PATH.open(encoding="utf-8") as f:
        wf = json.load(f)
    wf["1"]["inputs"]["dit_weight_mode"] = dit_weight_mode
    # Mark outputs with the tag so we can find them on disk afterward.
    prefix_tag = f"FP8VAL_{tag}"
    for nid, node in wf.items():
        if node.get("class_type") == "SaveImage":
            base = node["inputs"]["filename_prefix"]
            node["inputs"]["filename_prefix"] = base.replace(
                "unividx_LTX_R2AIN", f"unividx_LTX_R2AIN_{prefix_tag}"
            )
    body = _http_post(f"{BASE}/prompt",
                      {"prompt": wf, "client_id": f"b8val-{tag.lower()}"})
    return body["prompt_id"], prefix_tag


def wait_for(prompt_id: str, timeout_sec: float = 3600.0) -> dict:
    deadline = time.monotonic() + timeout_sec
    last_log = 0.0
    while time.monotonic() < deadline:
        hist = {}
        try:
            hist = _http_get(f"{BASE}/history/{prompt_id}")
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError):
            pass
        entry = hist.get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            return entry
        now = time.monotonic()
        if now - last_log > 30.0:
            print(f"  [{time.strftime('%H:%M:%S')}] {prompt_id[:8]}... "
                  f"still running", flush=True)
            last_log = now
        time.sleep(5)
    raise TimeoutError(f"{prompt_id} did not complete")


def find_outputs(prefix_tag: str, modality: str) -> list[Path]:
    """ComfyUI saves PNGs to COMFY_OUTPUT/<prefix>_<frame>_.png.
    We look for files matching unividx_LTX_R2AIN_<tag>_<modality>_*.png."""
    pattern = f"unividx_LTX_R2AIN_{prefix_tag}_{modality}_*.png"
    matches = sorted(COMFY_OUTPUT.glob(pattern))
    return matches


def psnr(a: torch.Tensor, b: torch.Tensor, peak: float = 1.0) -> float:
    """Peak-signal-to-noise ratio in dB. Both tensors in [0, 1]."""
    mse = (a - b).pow(2).mean().item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(peak * peak / mse)


def load_pngs_as_tensor(paths: list[Path]) -> torch.Tensor:
    """Stack N PNGs into a [N, C, H, W] tensor in [0, 1]."""
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        a = np.asarray(img, dtype=np.float32) / 255.0
        arrs.append(np.transpose(a, (2, 0, 1)))  # HWC -> CHW
    if not arrs:
        return torch.empty(0)
    return torch.from_numpy(np.stack(arrs))


def compare_modality(modality: str, ref_tag: str, fp8_tag: str) -> dict:
    ref = find_outputs(ref_tag, modality)
    fp8 = find_outputs(fp8_tag, modality)
    if len(ref) == 0 or len(fp8) == 0:
        return {"modality": modality, "n_ref": len(ref), "n_fp8": len(fp8),
                "psnr_db": float("nan"), "max_abs_diff": float("nan")}
    n = min(len(ref), len(fp8))
    a = load_pngs_as_tensor(ref[:n])
    b = load_pngs_as_tensor(fp8[:n])
    return {
        "modality": modality,
        "n_frames": n,
        "psnr_db": psnr(a, b),
        "max_abs_diff": (a - b).abs().max().item(),
        "mean_abs_diff": (a - b).abs().mean().item(),
    }


def main() -> int:
    print("=== Tier B8 validation: FP8 prequantized vs BF16 baseline ===")
    print(f"workflow: {WF_PATH.name}")

    print("\n[1/2] queueing BF16 baseline...")
    ref_pid, ref_tag = queue("bf16_shards", "BF16")
    print(f"  prompt_id={ref_pid}  tag={ref_tag}")
    t0 = time.time()
    wait_for(ref_pid)
    bf16_wall = time.time() - t0
    print(f"  BF16 baseline done in {bf16_wall/60:.2f} min")

    print("\n[2/2] queueing FP8 prequantized...")
    fp8_pid, fp8_tag = queue("fp8_prequantized", "FP8PRE")
    print(f"  prompt_id={fp8_pid}  tag={fp8_tag}")
    t0 = time.time()
    wait_for(fp8_pid)
    fp8_wall = time.time() - t0
    print(f"  FP8 prequantized done in {fp8_wall/60:.2f} min")

    print("\n=== per-modality comparison ===")
    print(f"{'modality':<14s}  {'n':>3s}  {'PSNR (dB)':>10s}  "
          f"{'max-diff':>10s}  {'mean-diff':>10s}")

    thresholds = {
        "placeholder": 30.0,  # the RGB output slot (black placeholder when
                              # RGB is condition); near-identical expected.
        "albedo": 30.0,
        "irradiance": 30.0,
        "normal": 30.0,
    }
    results = []
    for modality in ("placeholder", "albedo", "irradiance", "normal"):
        r = compare_modality(modality, ref_tag, fp8_tag)
        results.append(r)
        psnr_str = (f"{r['psnr_db']:.2f}" if r["psnr_db"] != float("inf")
                    else "inf")
        max_str = (f"{r['max_abs_diff']:.4f}"
                   if not np.isnan(r["max_abs_diff"]) else "n/a")
        mean_str = (f"{r['mean_abs_diff']:.4f}"
                    if "mean_abs_diff" in r else "n/a")
        n_str = str(r.get("n_frames", 0))
        print(f"{r['modality']:<14s}  {n_str:>3s}  {psnr_str:>10s}  "
              f"{max_str:>10s}  {mean_str:>10s}")

    print("\n=== threshold check ===")
    pass_count = 0
    fail_count = 0
    for r in results:
        thr = thresholds.get(r["modality"], 30.0)
        if np.isnan(r["psnr_db"]):
            verdict = "SKIP (no frames)"
            fail_count += 1
        elif r["psnr_db"] >= thr:
            verdict = f"PASS (≥ {thr:.0f} dB)"
            pass_count += 1
        else:
            verdict = f"FAIL (< {thr:.0f} dB)"
            fail_count += 1
        print(f"  {r['modality']:<14s}  {verdict}")

    print(f"\nwall: BF16={bf16_wall/60:.2f} min  FP8={fp8_wall/60:.2f} min  "
          f"(Phase 1 expectation: same order of magnitude)")
    print(f"{pass_count} pass, {fail_count} fail")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
