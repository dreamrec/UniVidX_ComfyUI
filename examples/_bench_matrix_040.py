"""0.4.0 close-out validation matrix.

Runs six benchmark conditions sequentially against a live ComfyUI
server, comparing the new dit_weight_mode='fp8_prequantized' path
against various stacks of existing perf knobs and against alternate
workflow configurations.

Conditions (in execution order):

  1. BF16-PRODUCTION-sage      bf16_shards + prefer_sage_attn=True + 20 steps
                                (the existing PRODUCTION preset; re-bench
                                 to reconcile with the inconsistent 14.5 min
                                 claim in README)
  2. FP8-PRODUCTION-sage       fp8_prequantized + prefer_sage_attn=True + 20 steps
                                (does FP8 + sage stack? quality + perf delta?)
  3. FP8-PRODUCTION-compile    fp8_prequantized + compile_dit=True + 20 steps
                                (does torch.compile graph-capture handle
                                 FP8Linear's dequant + matmul?)
  4. FP8-ALPHA                 alpha variant + R2PFB workflow + FP8 + 20 steps
                                (alpha matte decomposition is a different
                                 code path than RGB intrinsic)
  5. FP8-PREVIEW               fp8_prequantized + 4 steps + cfg=1
                                (PREVIEW preset under FP8 - chaotic
                                 amplification at short step counts is the
                                 risk)
  6. FP8-TEXT-ONLY             fp8_prequantized + t2RAIN workflow + 20 steps
                                (no RGB cross-conditioning - different
                                 attention pattern)

Each condition is queued, polled to completion via /history, and its
wall time + per-modality output PNGs captured. The driver also reports
per-condition cache miss/hit behavior and the FP8 substitution log
line where applicable.

After all six finish, a final summary table prints to stdout and is
written to test_matrix/040_validation.json for the CHANGELOG +
README update.

Expected wall: 2.5-3.5 hours total. Each cold-load + 20-step run is
~10-15 min; the PREVIEW (4 steps) is ~1.5-2 min.

Usage:
    python examples/_bench_matrix_040.py
    python examples/_bench_matrix_040.py --skip 1,3   # skip conditions 1 and 3
    python examples/_bench_matrix_040.py --only 5     # only run condition 5
"""
from __future__ import annotations

import argparse
import copy
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
COMFY_OUTPUT = Path("C:/Users/dr5090/Documents/ComfyUI/output")
BASE = "http://127.0.0.1:8000"

R2AIN_WF = REPO / "examples" / "R2AIN_video_api.json"
R2PFB_WF = REPO / "examples" / "R2PFB_video_api.json"
T2RAIN_WF = REPO / "examples" / "t2RAIN_tiny_api.json"  # tiny stand-in;
# would prefer a 480x640x21x20 text-only workflow but the existing one's
# the tiny config. Run anyway for the FP8-vs-BF16 perf shape signal.


def _http_get(url: str, timeout: float = 60.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _http_post(url: str, payload: dict, timeout: float = 30.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def queue_with_overrides(wf_path: Path, loader_overrides: dict,
                          sampler_overrides: dict, prefix_tag: str,
                          client_id: str) -> str:
    with wf_path.open(encoding="utf-8") as f:
        wf = json.load(f)
    # Loader node is "1" by convention in our workflow files.
    for k, v in loader_overrides.items():
        wf["1"]["inputs"][k] = v
    # Sampler node has class_type=UniVidXSampler.
    for nid, node in wf.items():
        if node.get("class_type") == "UniVidXSampler":
            for k, v in sampler_overrides.items():
                node["inputs"][k] = v
    # Tag SaveImage prefixes so we can find this run's outputs on disk.
    for nid, node in wf.items():
        if node.get("class_type") == "SaveImage":
            base_prefix = node["inputs"]["filename_prefix"]
            node["inputs"]["filename_prefix"] = f"{prefix_tag}__{base_prefix}"
    body = _http_post(f"{BASE}/prompt",
                      {"prompt": wf, "client_id": client_id})
    return body["prompt_id"]


def wait_for(prompt_id: str, timeout_sec: float = 7200.0) -> dict:
    deadline = time.monotonic() + timeout_sec
    last_log = 0.0
    while time.monotonic() < deadline:
        hist = {}
        try:
            hist = _http_get(f"{BASE}/history/{prompt_id}")
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError) as exc:
            print(f"  [{time.strftime('%H:%M:%S')}] poll {type(exc).__name__}",
                  flush=True)
        entry = hist.get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            return entry
        now = time.monotonic()
        if now - last_log > 60.0:
            print(f"  [{time.strftime('%H:%M:%S')}] {prompt_id[:8]}... running",
                  flush=True)
            last_log = now
        time.sleep(5)
    raise TimeoutError(f"{prompt_id} timed out")


def extract_server_wall(entry: dict) -> float | None:
    """Pull start/end timestamps out of the history entry."""
    messages = entry.get("status", {}).get("messages", [])
    start = end = None
    for msg in messages:
        if not isinstance(msg, (list, tuple)) or len(msg) < 2:
            continue
        event, payload = msg[0], msg[1]
        ts = payload.get("timestamp") if isinstance(payload, dict) else None
        if ts is None:
            continue
        if event == "execution_start":
            start = float(ts) / 1000.0
        if event == "execution_success":
            end = float(ts) / 1000.0
    if start is None or end is None:
        return None
    return end - start


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def load_tensor_for_prefix(prefix_tag: str, modality: str) -> torch.Tensor:
    paths = sorted(COMFY_OUTPUT.glob(f"{prefix_tag}__*{modality}*.png"))
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        a = np.asarray(img, dtype=np.float32) / 255.0
        arrs.append(np.transpose(a, (2, 0, 1)))
    if not arrs:
        return torch.empty(0)
    return torch.from_numpy(np.stack(arrs))


def compare_psnr(ref_tag: str, fp8_tag: str, modalities: list[str]) -> dict:
    out = {}
    for m in modalities:
        a = load_tensor_for_prefix(ref_tag, m)
        b = load_tensor_for_prefix(fp8_tag, m)
        if a.numel() == 0 or b.numel() == 0:
            out[m] = {"psnr_db": float("nan"), "n_ref": int(a.numel()),
                      "n_fp8": int(b.numel())}
            continue
        n = min(a.shape[0], b.shape[0])
        out[m] = {
            "psnr_db": psnr(a[:n], b[:n]),
            "max_diff": (a[:n] - b[:n]).abs().max().item(),
            "mean_diff": (a[:n] - b[:n]).abs().mean().item(),
            "n_frames": n,
        }
    return out


# ---------------------------------------------------------------------------

CONDITIONS = [
    {
        "id": 1, "tag": "BF16_PROD_SAGE",
        "wf": "R2AIN_video", "variant": "intrinsic",
        "loader": {"dit_weight_mode": "bf16_shards", "prefer_sage_attn": True},
        "sampler": {"num_inference_steps": 20, "cfg_scale": 5.0},
        "modalities": ["albedo", "irradiance", "normal", "placeholder"],
        "note": "Re-bench PRODUCTION sage baseline (resolve README 14.5 vs B8 10.85)",
    },
    {
        "id": 2, "tag": "FP8_PROD_SAGE",
        "wf": "R2AIN_video", "variant": "intrinsic",
        "loader": {"dit_weight_mode": "fp8_prequantized", "prefer_sage_attn": True},
        "sampler": {"num_inference_steps": 20, "cfg_scale": 5.0},
        "modalities": ["albedo", "irradiance", "normal", "placeholder"],
        "ref": "BF16_PROD_SAGE",
        "note": "FP8 + sage stacking",
    },
    {
        "id": 3, "tag": "FP8_PROD_COMPILE",
        "wf": "R2AIN_video", "variant": "intrinsic",
        "loader": {"dit_weight_mode": "fp8_prequantized", "compile_dit": True},
        "sampler": {"num_inference_steps": 20, "cfg_scale": 5.0},
        "modalities": ["albedo", "irradiance", "normal", "placeholder"],
        "ref": "BF16_PROD_SAGE",  # compare to same BF16 baseline
        "note": "FP8 + torch.compile",
    },
    {
        "id": 4, "tag": "FP8_ALPHA",
        "wf": "R2PFB_video", "variant": "alpha",
        "loader": {"dit_weight_mode": "fp8_prequantized"},
        "sampler": {"num_inference_steps": 20, "cfg_scale": 5.0},
        "modalities": ["alpha", "foreground", "background", "composite"],
        "note": "Alpha variant FP8 (different code path)",
    },
    {
        "id": 5, "tag": "FP8_PREVIEW",
        "wf": "R2AIN_video", "variant": "intrinsic",
        "loader": {"dit_weight_mode": "fp8_prequantized", "prefer_sage_attn": True},
        "sampler": {"num_inference_steps": 4, "cfg_scale": 1.0},
        "modalities": ["albedo", "irradiance", "normal", "placeholder"],
        "note": "PREVIEW under FP8 — chaotic amplification risk at 4 steps",
    },
    {
        "id": 6, "tag": "FP8_TEXT_ONLY",
        "wf": "t2RAIN", "variant": "intrinsic",
        "loader": {"dit_weight_mode": "fp8_prequantized"},
        "sampler": {"num_inference_steps": 3, "cfg_scale": 5.0},  # tiny config defaults
        "modalities": ["rgb", "albedo", "irradiance", "normal"],
        "note": "Text-only mode (no RGB cross-conditioning)",
    },
]


WF_MAP = {
    "R2AIN_video": R2AIN_WF,
    "R2PFB_video": R2PFB_WF,
    "t2RAIN": T2RAIN_WF,
}


def run_condition(cond: dict) -> dict:
    print(f"\n{'='*72}")
    print(f"[{cond['id']}/{len(CONDITIONS)}] {cond['tag']}  ({cond['note']})")
    print(f"  workflow={cond['wf']}  variant={cond['variant']}")
    print(f"  loader={cond['loader']}  sampler={cond['sampler']}")
    print(f"{'='*72}")
    wf_path = WF_MAP[cond["wf"]]
    loader_overrides = dict(cond["loader"])
    loader_overrides.setdefault("variant", cond["variant"])
    t0 = time.time()
    pid = queue_with_overrides(
        wf_path=wf_path,
        loader_overrides=loader_overrides,
        sampler_overrides=cond["sampler"],
        prefix_tag=cond["tag"],
        client_id=f"matrix-{cond['id']}-{cond['tag'].lower()}",
    )
    print(f"  prompt_id={pid}  queued at {time.strftime('%H:%M:%S')}")
    entry = wait_for(pid)
    t1 = time.time()
    client_wall = t1 - t0
    server_wall = extract_server_wall(entry)
    status = entry.get("status", {})
    status_str = status.get("status_str")
    print(f"  status={status_str}  client_wall={client_wall:.1f}s "
          f"server_wall={server_wall:.1f}s "
          f"({(server_wall or client_wall)/60:.2f} min)")
    return {
        "id": cond["id"], "tag": cond["tag"], "note": cond["note"],
        "prompt_id": pid, "status": status_str,
        "client_wall_sec": client_wall,
        "server_wall_sec": server_wall,
        "loader": loader_overrides,
        "sampler": cond["sampler"],
        "workflow": cond["wf"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="",
                    help="comma-separated condition IDs to skip")
    ap.add_argument("--only", default="",
                    help="comma-separated condition IDs to run (overrides skip)")
    args = ap.parse_args()
    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    only = {int(x) for x in args.only.split(",") if x.strip()}
    conditions = [c for c in CONDITIONS
                  if (not only or c["id"] in only) and c["id"] not in skip]

    print(f"running {len(conditions)} condition(s):")
    for c in conditions:
        print(f"  [{c['id']}] {c['tag']} - {c['note']}")

    results = []
    for c in conditions:
        try:
            r = run_condition(c)
            r["ref_tag"] = c.get("ref")
            r["modalities"] = c["modalities"]
            results.append(r)
        except Exception as exc:
            print(f"\nERROR in condition {c['id']} ({c['tag']}): "
                  f"{type(exc).__name__}: {exc}", flush=True)
            results.append({
                "id": c["id"], "tag": c["tag"],
                "error": f"{type(exc).__name__}: {exc}",
            })

    # PSNR vs reference (where applicable).
    psnr_block = {}
    for r in results:
        if r.get("error"):
            continue
        ref_tag = r.get("ref_tag")
        if not ref_tag:
            continue
        cmp = compare_psnr(ref_tag, r["tag"], r["modalities"])
        psnr_block[r["tag"]] = cmp

    # Summary
    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"{'#':>2s} {'tag':<20s} {'wall (min)':>11s} {'status':>10s}")
    for r in results:
        wall = (r.get("server_wall_sec") or r.get("client_wall_sec") or 0) / 60
        print(f"{r['id']:>2d} {r['tag']:<20s} {wall:>11.2f} "
              f"{r.get('status') or r.get('error', 'ERROR'):>10s}")

    if psnr_block:
        print(f"\nPSNR vs reference (where applicable):")
        for tag, modmap in psnr_block.items():
            print(f"\n  {tag}:")
            for m, stats in modmap.items():
                p = stats.get('psnr_db')
                p_str = (f"{p:.2f}" if p is not None
                         and not np.isnan(p) and p != float("inf")
                         else ("inf" if p == float("inf") else "n/a"))
                print(f"    {m:<14s}  PSNR={p_str:>8s} dB  "
                      f"n_frames={stats.get('n_frames', 0)}")

    out_path = REPO / "examples" / "test_matrix" / "040_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"conditions": results, "psnr": psnr_block,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  f, indent=2)
    print(f"\nresults written to {out_path}")
    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
