"""Queue a perf benchmark of the R2AIN_video workflow with chosen
loader settings. Used to populate the README's performance table.

Examples:
    python examples/_bench_perf.py --vram-buffer 12 --tag VRAM12
    python examples/_bench_perf.py --dtype fp8_e4m3fn --vram-buffer 0.5 --tag FP8
    python examples/_bench_perf.py --dtype fp8_e5m2  --vram-buffer 0.5 --tag FP8E5
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WF_PATH = REPO / "examples" / "R2AIN_video_api.json"


def queue(dtype: str, vram_buffer_gb: float, tag: str) -> str:
    with WF_PATH.open(encoding="utf-8") as f:
        wf = json.load(f)
    wf["1"]["inputs"]["dtype"] = dtype
    wf["1"]["inputs"]["vram_buffer_gb"] = float(vram_buffer_gb)
    for nid, node in wf.items():
        if node.get("class_type") == "SaveImage":
            node["inputs"]["filename_prefix"] = (
                node["inputs"]["filename_prefix"].replace("R2AIN", f"R2AIN_BENCH_{tag}")
            )
    payload = json.dumps({"prompt": wf, "client_id": f"bench-{tag.lower()}"}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/prompt", data=payload,
        headers={"Content-Type": "application/json"},
    )
    print(f"queue {tag} dtype={dtype} vram_buffer={vram_buffer_gb} at {time.strftime('%H:%M:%S')}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.load(resp)
    print(f"  prompt_id = {body['prompt_id']}  number = {body['number']}")
    return body["prompt_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "fp8_e4m3fn", "fp8_e5m2"])
    ap.add_argument("--vram-buffer", type=float, default=4.0)
    ap.add_argument("--tag", required=True, help="Unique tag for SaveImage prefix")
    args = ap.parse_args()
    queue(args.dtype, args.vram_buffer, args.tag)


if __name__ == "__main__":
    main()
