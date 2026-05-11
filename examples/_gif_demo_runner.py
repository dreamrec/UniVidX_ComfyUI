"""One-shot demo runner: queues the R2AIN GIF-output workflow against the
local ComfyUI server, polls /history until completion, then copies the
four output GIFs (rgb input + albedo + irradiance + normal) into
assets/results/ for README embedding.

Run from repo root with ComfyUI running on port 8000:
    python examples/_gif_demo_runner.py

The workflow uses 21 frames at 640x480, R2AIN mode, FP8 prequantized DiT,
production-quality preset (steps=20, cfg=5.0), matching CHANGELOG bench
parameters so the published GIFs reflect real production output.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Force UTF-8 stdout on Windows so progress prints with arrows / labels don't
# explode under cp1252 when this script is run with redirected stdout (the
# default Windows console codepage can't encode "→", and a single bad print
# in the post-success collect() step would otherwise crash the runner after
# 11+ minutes of useful work.)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, io.UnsupportedOperation):
        pass

BASE = "http://127.0.0.1:8000"
REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets" / "results"
OUTPUT_DIR = Path("C:/Users/dr5090/Documents/ComfyUI/output")

# Workflow nodes: standard R2AIN pipeline + ImageScale (480x360) +
# VHS_VideoCombine per modality, format=image/gif, frame_rate=12.
WORKFLOW = {
    "1": {
        "class_type": "UniVidXLoader",
        "inputs": {
            "variant": "intrinsic",
            "dtype": "bfloat16",
            "dit_weight_mode": "fp8_prequantized",
        },
    },
    "2": {
        "class_type": "UniVidXTaskMode",
        "inputs": {"mode": "R2AIN"},
    },
    "3": {
        "class_type": "VHS_LoadVideoPath",
        "inputs": {
            "video": "C:/Users/dr5090/Documents/ComfyUI/input/LTX_2.3_t2v_00239_.mp4",
            "force_rate": 0,
            "custom_width": 640,
            "custom_height": 480,
            "frame_load_cap": 21,
            "skip_first_frames": 0,
            "select_every_nth": 1,   # contiguous frames — natural motion, no jump-cuts
            "format": "Wan",
        },
    },
    "5": {
        "class_type": "UniVidXSampler",
        "inputs": {
            "model": ["1", 0],
            "task": ["2", 0],
            "rgb": ["3", 0],
            "prompt": (
                "a cinematic portrait of a young man with long brown hair, "
                "white shirt, candlelit room, ornate wallpaper, gothic "
                "atmosphere, soft warm lighting"
            ),
            "negative_prompt": (
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"
                "画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
                "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
                "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
                "三条腿，背景人很多，倒着走"
            ),
            "num_inference_steps": 20,
            "cfg_scale": 5.0,
            "denoising_strength": 1.0,
            "num_frames": 21,
            "height": 480,
            "width": 640,
            "seed": 42,
            "tiled": True,
        },
    },
    "6": {
        "class_type": "UniVidXDecodeIntrinsic",
        "inputs": {"result": ["5", 0]},
    },
    # Downscale to 480x360 (kept aspect ratio, halves GIF byte count)
    "11": {"class_type": "ImageScale", "inputs": {
        "image": ["3", 0], "upscale_method": "lanczos",
        "width": 480, "height": 360, "crop": "disabled"}},
    "12": {"class_type": "ImageScale", "inputs": {
        "image": ["6", 1], "upscale_method": "lanczos",
        "width": 480, "height": 360, "crop": "disabled"}},
    "13": {"class_type": "ImageScale", "inputs": {
        "image": ["6", 2], "upscale_method": "lanczos",
        "width": 480, "height": 360, "crop": "disabled"}},
    "14": {"class_type": "ImageScale", "inputs": {
        "image": ["6", 3], "upscale_method": "lanczos",
        "width": 480, "height": 360, "crop": "disabled"}},
    # GIF outputs - 12 fps × 21 frames = 1.75 s loop, infinite loop
    "21": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["11", 0], "frame_rate": 24, "loop_count": 0,
        "filename_prefix": "unividx_demo_rgb", "format": "image/gif",
        "pingpong": False, "save_output": True}},
    "22": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["12", 0], "frame_rate": 24, "loop_count": 0,
        "filename_prefix": "unividx_demo_albedo", "format": "image/gif",
        "pingpong": False, "save_output": True}},
    "23": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["13", 0], "frame_rate": 24, "loop_count": 0,
        "filename_prefix": "unividx_demo_irradiance", "format": "image/gif",
        "pingpong": False, "save_output": True}},
    "24": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["14", 0], "frame_rate": 24, "loop_count": 0,
        "filename_prefix": "unividx_demo_normal", "format": "image/gif",
        "pingpong": False, "save_output": True}},
}


def queue() -> str:
    data = json.dumps({"prompt": WORKFLOW, "client_id": "unividx-gif-demo"}).encode()
    req = urllib.request.Request(
        f"{BASE}/prompt", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    pid = body["prompt_id"]
    print(f"queued: prompt_id={pid}")
    return pid


def wait(pid: str, timeout_sec: float = 1500.0) -> dict:
    """Poll /history/<pid> until the entry reports completed=True."""
    deadline = time.monotonic() + timeout_sec
    last_log = 0.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/history/{pid}", timeout=15) as r:
                hist = json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError):
            hist = {}
        entry = hist.get(pid)
        if entry and entry.get("status", {}).get("completed"):
            print(f"completed: status={entry['status'].get('status_str')}")
            return entry
        now = time.monotonic()
        if now - last_log > 30.0:
            elapsed = int(now - (deadline - timeout_sec))
            print(f"  [{time.strftime('%H:%M:%S')}] {elapsed}s elapsed, "
                  f"still running...")
            last_log = now
        time.sleep(5)
    raise TimeoutError(f"prompt {pid} did not complete in {timeout_sec}s")


def collect(entry: dict) -> list[Path]:
    """Walk the history entry's per-node outputs and copy each saved GIF
    into assets/results/, renamed to a clean target name."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    target_map = {
        "21": "demo_rgb.gif",
        "22": "demo_albedo.gif",
        "23": "demo_irradiance.gif",
        "24": "demo_normal.gif",
    }
    copied: list[Path] = []
    outputs = entry.get("outputs", {})
    for node_id, target_name in target_map.items():
        gifs = outputs.get(node_id, {}).get("gifs", [])
        if not gifs:
            print(f"  WARN node {node_id}: no gif output recorded "
                  f"(keys: {list(outputs.get(node_id, {}).keys())})")
            continue
        meta = gifs[0]
        src = OUTPUT_DIR / meta.get("subfolder", "") / meta["filename"]
        if not src.is_file():
            print(f"  WARN node {node_id}: source missing at {src}")
            continue
        dst = ASSETS / target_name
        shutil.copy2(src, dst)
        size_kb = dst.stat().st_size // 1024
        print(f"  copied {src.name} → {dst.relative_to(REPO)} ({size_kb} KB)")
        copied.append(dst)
    return copied


def main() -> int:
    try:
        with urllib.request.urlopen(f"{BASE}/system_stats", timeout=5):
            pass
    except Exception as exc:
        print(f"ComfyUI not reachable on {BASE}: {exc}")
        return 2

    t0 = time.time()
    pid = queue()
    entry = wait(pid)
    print(f"total wall: {(time.time() - t0)/60:.2f} min")
    gifs = collect(entry)
    if len(gifs) != 4:
        print(f"WARN: only {len(gifs)}/4 GIFs were collected")
        return 1
    print("all 4 GIFs in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
