"""Tier B sanity check: confirm dit_weight_mode='fp8_prequantized' loads
the FP8 substitution path live in the running ComfyUI process.

What this proves:
- The new code (fp8_loader.py, runtime branch, loader widget) is in
  sys.modules.
- The Kijai FP8 file resolves on disk.
- load_fp8_state_dict_into walks model.pipe.dit, descends through
  PEFT wrappers, replaces ~400 base nn.Linear with FP8Linear, loads
  the F32 aux tensors.
- The expected log line "FP8 substitution complete: N Linears ->
  FP8Linear, M aux loaded, K unmatched" fires.
- A tiny sampling run produces output without erroring.

Greps the live ComfyUI log file for the FP8 marker. Exits nonzero
if the marker is missing or the run fails.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WF_PATH = REPO / "examples" / "t2RAIN_tiny_api.json"
LOG_PATH = Path("C:/Users/dr5090/Documents/ComfyUI/user/comfyui_8000.log")
MARKER = "FP8 substitution complete"
BASE = "http://127.0.0.1:8000"


def main() -> int:
    with WF_PATH.open(encoding="utf-8") as f:
        wf = json.load(f)
    wf["1"]["inputs"]["dit_weight_mode"] = "fp8_prequantized"
    for nid, node in wf.items():
        if node.get("class_type") == "SaveImage":
            node["inputs"]["filename_prefix"] = (
                node["inputs"]["filename_prefix"].replace(
                    "unividx_tiny", "unividx_sanity_fp8"
                )
            )

    log_start_size = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
    print(f"log baseline byte offset = {log_start_size}")

    payload = json.dumps({"prompt": wf,
                          "client_id": "sanity-fp8"}).encode("utf-8")
    req = urllib.request.Request(f"{BASE}/prompt", data=payload,
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    prompt_id = body["prompt_id"]
    print(f"queued prompt_id = {prompt_id} at {time.strftime('%H:%M:%S')}")

    deadline = time.monotonic() + 1800  # 30 min cap — FP8 load is slower
    last_log = 0.0
    entry = None
    while time.monotonic() < deadline:
        hist = {}
        try:
            with urllib.request.urlopen(f"{BASE}/history/{prompt_id}",
                                         timeout=60) as resp:
                hist = json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError) as exc:
            print(f"  [{time.strftime('%H:%M:%S')}] poll error "
                  f"({type(exc).__name__}), retrying", flush=True)
        entry = hist.get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            break
        now = time.monotonic()
        if now - last_log > 20.0:
            print(f"  [{time.strftime('%H:%M:%S')}] still running...",
                  flush=True)
            last_log = now
        time.sleep(5)
    else:
        print("ERROR: run timed out", file=sys.stderr)
        return 2

    t1 = time.time()
    wall = t1 - t0
    print(f"completed in {wall:.1f} sec ({wall/60:.2f} min)")

    status = entry.get("status", {})
    if status.get("status_str") != "success":
        print(f"ERROR: run did not succeed: status={status}", file=sys.stderr)
        return 3

    if not LOG_PATH.exists():
        print(f"ERROR: log file not found at {LOG_PATH}", file=sys.stderr)
        return 4
    with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(log_start_size)
        tail = f.read()
    matches = [line for line in tail.splitlines() if MARKER in line]
    if not matches:
        print(f"ERROR: marker '{MARKER}' not found in log tail "
              f"({len(tail)} bytes scanned)", file=sys.stderr)
        print("--- last 40 lines of log tail ---", file=sys.stderr)
        for line in tail.splitlines()[-40:]:
            print(line, file=sys.stderr)
        return 5
    print("\n=== FP8 SANITY PASSED ===")
    for line in matches:
        print(line.strip())
    # Also surface any "unmatched" warnings.
    unmatched_lines = [line for line in tail.splitlines()
                       if "unmatched" in line.lower()
                       or "warning" in line.lower() and "FP8" in line]
    if unmatched_lines:
        print("\n--- WARNINGS during FP8 load ---")
        for line in unmatched_lines[:10]:
            print(line.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
