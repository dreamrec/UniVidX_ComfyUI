"""Tier C smoke test: confirm step_distill_lora='lightx2v' is wired
end-to-end in the running ComfyUI process.

Queues an R2AIN_video_api.json workflow with step_distill_lora='lightx2v',
step_distill_strength=1.0, and reduced sampler settings
(num_inference_steps=4, cfg_scale=1.0) matching what the distillation
is trained for. Watches the live ComfyUI log for the merge marker
line: "Step-distill merge complete (lightx2v, strength=1.00):
N Linears merged, M biases patched, K weights patched, ...".

Exits non-zero on missing marker or failed sampling. Wall ~3-5 min
(cold-load + merge walk + 4 sample steps + VAE).

Outputs land under ComfyUI/output/unividx_smoke_distill_* so you
can visually inspect after.
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
LOG_PATH = Path("C:/Users/dr5090/Documents/ComfyUI/user/comfyui_8000.log")
MARKER = "Step-distill merge complete"
BASE = "http://127.0.0.1:8000"


def main() -> int:
    with WF_PATH.open(encoding="utf-8") as f:
        wf = json.load(f)
    # Loader: enable lightx2v step-distill at full strength; keep FP8 off
    # so this run is JUST distill (we'll test the compound separately).
    for nid, node in wf.items():
        if node.get("class_type") == "UniVidXLoader":
            node["inputs"]["step_distill_lora"] = "lightx2v"
            node["inputs"]["step_distill_strength"] = 1.0
            node["inputs"]["dit_weight_mode"] = "bf16_shards"
            node["inputs"]["prefer_sage_attn"] = False
        elif node.get("class_type") == "UniVidXSampler":
            # Distillation targets cfg=1 + low step count.
            node["inputs"]["num_inference_steps"] = 4
            node["inputs"]["cfg_scale"] = 1.0
        elif node.get("class_type") == "SaveImage":
            base = node["inputs"]["filename_prefix"]
            node["inputs"]["filename_prefix"] = (
                base.replace("unividx_LTX_R2AIN", "unividx_smoke_distill")
            )

    log_start = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
    print(f"log baseline byte offset = {log_start}")

    payload = json.dumps({"prompt": wf,
                          "client_id": "smoke-distill"}).encode("utf-8")
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
                TimeoutError, OSError) as exc:
            print(f"  [{time.strftime('%H:%M:%S')}] poll {type(exc).__name__}",
                  flush=True)
        entry = hist.get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            break
        now = time.monotonic()
        if now - last > 20.0:
            print(f"  [{time.strftime('%H:%M:%S')}] running...",
                  flush=True)
            last = now
        time.sleep(5)
    else:
        print("ERROR: timed out", file=sys.stderr)
        return 2

    wall = time.time() - t0
    print(f"completed in {wall:.1f} sec ({wall/60:.2f} min)")

    status = entry.get("status", {})
    if status.get("status_str") != "success":
        print(f"ERROR: run did not succeed: status={status}", file=sys.stderr)
        return 3

    with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(log_start)
        tail = f.read()
    matches = [line for line in tail.splitlines() if MARKER in line]
    if not matches:
        print(f"ERROR: marker '{MARKER}' not found in log tail "
              f"({len(tail)} bytes scanned)", file=sys.stderr)
        for line in tail.splitlines()[-40:]:
            print(line, file=sys.stderr)
        return 5

    print("\n=== DISTILL SMOKE PASSED ===")
    for line in matches:
        print(line.strip())
    # Quick output stats so we can tell if the model produced
    # meaningful content vs all-noise / all-saturated.
    try:
        import numpy as np
        from PIL import Image
        out_dir = Path("C:/Users/dr5090/Documents/ComfyUI/output")
        for modality in ("placeholder", "albedo", "irradiance", "normal"):
            paths = sorted(out_dir.glob(f"unividx_smoke_distill_*{modality}*.png"))
            if not paths:
                continue
            stats = []
            for p in paths[:3]:
                a = np.asarray(Image.open(p).convert("RGB"),
                               dtype=np.float32) / 255.0
                stats.append((a.mean(), a.std()))
            print(f"  {modality:<12s} ({len(paths)} frames): "
                  + ", ".join(f"mean={m:.3f} std={s:.3f}"
                              for m, s in stats[:3]))
    except Exception as exc:
        print(f"  (image stat probe failed: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
