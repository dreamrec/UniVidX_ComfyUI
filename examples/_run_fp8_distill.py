"""Tier C5: queue a single FP8 + step-distill compound run for PSNR
comparison.

Same workflow as the BF16 baseline (R2AIN_video_api.json, 480×640×21,
cfg=1, 4 steps) but with BOTH dit_weight_mode='fp8_prequantized' and
step_distill_lora='lightx2v'. Tests whether the two optimizations
compound cleanly or whether FP8 quantization noise + distill quality
loss destructively interact.

Outputs go to ComfyUI/output/unividx_fp8_distill_* for the PSNR
comparator to consume.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WF_PATH = REPO / "examples" / "R2AIN_video_api.json"
BASE = "http://127.0.0.1:8000"


def main() -> int:
    with WF_PATH.open(encoding="utf-8") as f:
        wf = json.load(f)
    for nid, node in wf.items():
        ct = node.get("class_type")
        if ct == "UniVidXLoader":
            node["inputs"]["dit_weight_mode"] = "fp8_prequantized"
            node["inputs"]["step_distill_lora"] = "lightx2v"
            node["inputs"]["step_distill_strength"] = 1.0
            node["inputs"]["prefer_sage_attn"] = False
        elif ct == "UniVidXSampler":
            node["inputs"]["num_inference_steps"] = 4
            node["inputs"]["cfg_scale"] = 1.0
        elif ct == "SaveImage":
            base = node["inputs"]["filename_prefix"]
            node["inputs"]["filename_prefix"] = (
                base.replace("unividx_LTX_R2AIN", "unividx_fp8_distill")
            )

    payload = json.dumps({"prompt": wf,
                          "client_id": "c5-fp8-distill"}).encode("utf-8")
    req = urllib.request.Request(f"{BASE}/prompt", data=payload,
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    prompt_id = body["prompt_id"]
    print(f"queued prompt_id = {prompt_id} at {time.strftime('%H:%M:%S')}")

    deadline = time.monotonic() + 1800
    last = 0.0
    entry = None
    while time.monotonic() < deadline:
        hist = {}
        try:
            with urllib.request.urlopen(f"{BASE}/history/{prompt_id}",
                                         timeout=60) as resp:
                hist = json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError):
            pass
        entry = hist.get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            break
        now = time.monotonic()
        if now - last > 30.0:
            print(f"  [{time.strftime('%H:%M:%S')}] running...",
                  flush=True)
            last = now
        time.sleep(5)
    else:
        print("ERROR: timed out", file=sys.stderr)
        return 2

    wall = time.time() - t0
    print(f"completed in {wall:.1f} sec ({wall/60:.2f} min)")
    status = entry.get("status", {}).get("status_str")
    if status != "success":
        print(f"ERROR: run did not succeed: {status}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
