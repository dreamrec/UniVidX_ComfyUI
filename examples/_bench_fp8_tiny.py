"""Tiny-scale Phase 1 quality check.

Queues t2RAIN_tiny_api.json with dit_weight_mode='bf16_shards', then
compares per-modality PSNR vs the FP8 outputs already on disk from
the earlier sanity run (unividx_sanity_fp8_*).

~3 min total (one tiny ~2 min cold-load run + 30 sec analysis).
The full B8 harness (_bench_fp8_prequantized.py) does the same
comparison at production scale (~25 min) once we trust this.
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
WF_PATH = REPO / "examples" / "t2RAIN_tiny_api.json"
COMFY_OUTPUT = Path("C:/Users/dr5090/Documents/ComfyUI/output")
BASE = "http://127.0.0.1:8000"


def queue_bf16_baseline() -> str:
    with WF_PATH.open(encoding="utf-8") as f:
        wf = json.load(f)
    wf["1"]["inputs"]["dit_weight_mode"] = "bf16_shards"
    for nid, node in wf.items():
        if node.get("class_type") == "SaveImage":
            node["inputs"]["filename_prefix"] = (
                node["inputs"]["filename_prefix"].replace(
                    "unividx_tiny", "unividx_tiny_bf16ref"
                )
            )
    payload = json.dumps({"prompt": wf,
                          "client_id": "tiny-bf16ref"}).encode("utf-8")
    req = urllib.request.Request(f"{BASE}/prompt", data=payload,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    return body["prompt_id"]


def wait_for(prompt_id: str, timeout_sec: float = 900.0) -> None:
    deadline = time.monotonic() + timeout_sec
    last_log = 0.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/history/{prompt_id}",
                                         timeout=60) as resp:
                hist = json.load(resp)
            entry = hist.get(prompt_id)
            if entry and entry.get("status", {}).get("completed"):
                return
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError):
            pass
        now = time.monotonic()
        if now - last_log > 20.0:
            print(f"  [{time.strftime('%H:%M:%S')}] still running...",
                  flush=True)
            last_log = now
        time.sleep(5)
    raise TimeoutError(f"{prompt_id} timed out")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def load_tensor(paths: list[Path]) -> torch.Tensor:
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        a = np.asarray(img, dtype=np.float32) / 255.0
        arrs.append(np.transpose(a, (2, 0, 1)))
    return torch.from_numpy(np.stack(arrs))


def main() -> int:
    print("=== queueing BF16 baseline (tiny t2RAIN, seed=42) ===")
    pid = queue_bf16_baseline()
    print(f"  prompt_id={pid}")
    t0 = time.time()
    wait_for(pid)
    print(f"  done in {(time.time() - t0)/60:.2f} min")

    print("\n=== per-modality PSNR (FP8 vs BF16 reference) ===")
    print(f"{'modality':<12s}  {'n':>3s}  {'PSNR (dB)':>10s}  "
          f"{'max-diff':>9s}  {'mean-diff':>9s}")
    results = {}
    for modality in ("rgb", "albedo", "irradiance", "normal"):
        ref = sorted(COMFY_OUTPUT.glob(
            f"unividx_tiny_bf16ref_{modality}_*.png"))
        fp8 = sorted(COMFY_OUTPUT.glob(
            f"unividx_sanity_fp8_{modality}_*.png"))
        n = min(len(ref), len(fp8))
        if n == 0:
            print(f"{modality:<12s}  {'0':>3s}  {'n/a':>10s}  "
                  f"{'n/a':>9s}  {'n/a':>9s}")
            continue
        a = load_tensor(ref[:n])
        b = load_tensor(fp8[:n])
        p = psnr(a, b)
        md = (a - b).abs().max().item()
        ad = (a - b).abs().mean().item()
        results[modality] = p
        p_str = f"{p:.2f}" if p != float("inf") else "inf"
        print(f"{modality:<12s}  {n:>3d}  {p_str:>10s}  "
              f"{md:>9.4f}  {ad:>9.4f}")

    # Threshold check.
    print("\n=== threshold check ===")
    fail = 0
    for mod, p in results.items():
        thr = 35.0 if mod == "rgb" else 30.0
        verdict = (f"PASS (>= {thr:.0f} dB)" if (p == float("inf") or p >= thr)
                   else f"FAIL (< {thr:.0f} dB)")
        if "FAIL" in verdict:
            fail += 1
        print(f"  {mod:<12s}  {verdict}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
