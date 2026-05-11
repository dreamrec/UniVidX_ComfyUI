"""vram_buffer_gb perf-delta bench on a live ComfyUI server.

Queues R2AIN_video_api.json twice with `vram_buffer_gb=4.0` and
`vram_buffer_gb=12.0` (default sweep) and reports wall + per-step
time for each. Produced the +65% delta number that landed in 0.3.0's
perf-table row.

Both runs go through the cache-miss path (distinct cache keys after
the 0.3.0 fix), so each does a fresh cold load + 20 sample steps.
The wall delta isolates the effect of vram_buffer on the sample
phase: lower buffer = more layers resident = faster sampling; higher
buffer = more layers offloaded = slower sampling but more activation
headroom.

Usage:
    python examples/_bench_vram_buffer.py
    python examples/_bench_vram_buffer.py --buffers 4.0 8.0 12.0
    python examples/_bench_vram_buffer.py --host 127.0.0.1 --port 8000

Notes:
    - Expect ~10-15 min per condition on RTX 5090 (480x640 x21 frames x20 steps,
      no sage/compile/FP8).
    - Two conditions = ~25 min total. Three = ~40 min.
    - With dit_weight_mode='fp8_prequantized' (0.4.0+), the vram_buffer
      knob becomes effectively a no-op because the FP8 DiT fits fully
      resident — bench should show ~9-10 min wall regardless of value.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WF_PATH = REPO / "examples" / "R2AIN_video_api.json"


def _http_get(url: str, timeout: float = 60.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _http_post(url: str, payload: dict, timeout: float = 10.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def queue_run(base_url: str, vram_buffer_gb: float, tag: str) -> str:
    with WF_PATH.open(encoding="utf-8") as f:
        wf = json.load(f)
    wf["1"]["inputs"]["vram_buffer_gb"] = float(vram_buffer_gb)
    for nid, node in wf.items():
        if node.get("class_type") == "SaveImage":
            node["inputs"]["filename_prefix"] = (
                node["inputs"]["filename_prefix"].replace("R2AIN", f"R2AIN_BENCH_{tag}")
            )
    body = _http_post(f"{base_url}/prompt",
                      {"prompt": wf, "client_id": f"bench-{tag.lower()}"})
    return body["prompt_id"]


def wait_for_completion(base_url: str, prompt_id: str,
                        poll_interval: float = 5.0,
                        timeout_sec: float = 3600.0) -> dict:
    """Poll /history/<prompt_id> until status.completed is True.

    Returns the full history entry (with messages timeline) once done.
    Raises TimeoutError if it doesn't complete in `timeout_sec`.
    """
    deadline = time.monotonic() + timeout_sec
    last_log = 0.0
    while time.monotonic() < deadline:
        hist = {}
        try:
            hist = _http_get(f"{base_url}/history/{prompt_id}")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  [{time.strftime('%H:%M:%S')}] poll error ({type(exc).__name__}), retrying",
                  flush=True)
        entry = hist.get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
        now = time.monotonic()
        if now - last_log > 30.0:
            print(f"  [{time.strftime('%H:%M:%S')}] still running ({prompt_id[:8]}...)",
                  flush=True)
            last_log = now
        time.sleep(poll_interval)
    raise TimeoutError(f"prompt {prompt_id} did not complete in {timeout_sec}s")


def extract_timing(entry: dict) -> dict:
    """Pull start/end timestamps and step timings out of the history entry.

    ComfyUI's history entry has `status.messages` — a list of
    `[event_name, payload]` pairs. We look for execution_start /
    execution_success and any `progress` messages to compute per-step.
    """
    messages = entry.get("status", {}).get("messages", [])
    start_ts = end_ts = None
    step_timestamps: list[float] = []
    for msg in messages:
        if not isinstance(msg, (list, tuple)) or len(msg) < 2:
            continue
        event, payload = msg[0], msg[1]
        ts = payload.get("timestamp") if isinstance(payload, dict) else None
        if event == "execution_start" and ts is not None:
            start_ts = float(ts) / 1000.0  # ms -> sec
        if event == "execution_success" and ts is not None:
            end_ts = float(ts) / 1000.0
        if event == "progress" and ts is not None:
            step_timestamps.append(float(ts) / 1000.0)
    wall = (end_ts - start_ts) if (start_ts and end_ts) else None
    per_step = None
    if len(step_timestamps) >= 2:
        deltas = [b - a for a, b in zip(step_timestamps, step_timestamps[1:])]
        per_step = sum(deltas) / len(deltas)
    return {"wall_sec": wall, "per_step_sec": per_step,
            "num_progress_events": len(step_timestamps)}


def bench_one(base_url: str, buffer_gb: float) -> dict:
    tag = f"VRAM{int(buffer_gb) if buffer_gb.is_integer() else buffer_gb}"
    print(f"\n=== vram_buffer_gb = {buffer_gb} ({tag}) ===")
    t0 = time.time()
    prompt_id = queue_run(base_url, buffer_gb, tag)
    print(f"  queued prompt_id = {prompt_id}")
    entry = wait_for_completion(base_url, prompt_id)
    t1 = time.time()
    timing = extract_timing(entry)
    timing["client_wall_sec"] = t1 - t0
    print(f"  client wall:      {timing['client_wall_sec']:.1f} sec "
          f"= {timing['client_wall_sec']/60:.2f} min")
    if timing["wall_sec"] is not None:
        print(f"  server wall:      {timing['wall_sec']:.1f} sec "
              f"= {timing['wall_sec']/60:.2f} min")
    if timing["per_step_sec"] is not None:
        print(f"  avg per-step:     {timing['per_step_sec']:.2f} sec "
              f"({timing['num_progress_events']} progress events)")
    return {"buffer_gb": buffer_gb, **timing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--buffers", nargs="+", type=float, default=[4.0, 12.0],
                    help="vram_buffer_gb values to bench (default: 4.0 12.0)")
    args = ap.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    print(f"benching {len(args.buffers)} condition(s) at {base_url}")
    print(f"workflow: {WF_PATH.name}")

    results: list[dict] = []
    for b in args.buffers:
        results.append(bench_one(base_url, b))

    print("\n=== summary ===")
    print(f"{'vram_buffer':>12s}  {'wall (min)':>12s}  {'per-step (s)':>14s}")
    for r in results:
        wall_min = (r["wall_sec"] or r["client_wall_sec"]) / 60.0
        ps = r["per_step_sec"]
        print(f"{r['buffer_gb']:>12.1f}  {wall_min:>12.2f}  "
              f"{ps if ps is not None else float('nan'):>14.2f}")

    if len(results) >= 2:
        a, b = results[0], results[-1]
        wall_a = (a["wall_sec"] or a["client_wall_sec"]) / 60.0
        wall_b = (b["wall_sec"] or b["client_wall_sec"]) / 60.0
        delta_pct = (wall_b - wall_a) / wall_a * 100.0
        print(f"\ndelta (vram_buffer {a['buffer_gb']:.1f} -> {b['buffer_gb']:.1f}): "
              f"{wall_b - wall_a:+.2f} min ({delta_pct:+.1f}%)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
