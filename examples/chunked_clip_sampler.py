"""Process a long clip through UniVidX by chunking + overlap-blend stitching.

UniVidX's training distribution is fixed at 21 frames per sample. For clips
longer than ~1 second of source video, this driver chunks the source into
overlapping 21-frame windows, runs UniVidX on each, then stitches the
per-modality outputs with a linear crossfade across the overlap region.

EXPERIMENTAL: the chunk-planning math is unit-tested (deterministic), but
the end-to-end pipeline against a real source clip hasn't been validated
yet at the time of commit. Treat this as a starting point — you'll likely
want to add anchor-frame conditioning, per-chunk retry policy refinement,
and possibly a custom workflow template for your specific task mode.

Usage:
    python examples/chunked_clip_sampler.py \\
        --input  C:/path/to/source.mp4 \\
        --mode   R2AIN \\
        --output-dir C:/path/to/output \\
        --preset FP8_PREVIEW

Presets (measured 0.5.0):
    FP8                  FP8 + 20 steps cfg=5            9.43 min/chunk   (production quality)
    FP8_DISTILL_PREVIEW  FP8 + lightx2v + 4 steps cfg=1  4.59 min/chunk   (3.15x faster, ~22-26 dB PSNR vs BF16)
    PRODUCTION           BF16 + sage + 20 steps         14.48 min/chunk  (legacy, slower)
    FP8_SAGE             FP8 + sage + 20 steps          11.75 min/chunk  (legacy)
    PREVIEW              BF16 + sage + 4 steps cfg=1     6.20 min/chunk  (legacy)
    FP8_PREVIEW          FP8 + 4 steps cfg=1             6.20 min/chunk  (legacy; pre-distill)

Wall-time estimate: ~total_chunks × per-chunk wall (after first chunk's
cold-load, subsequent chunks share cache key for instant reload).

For 1 min @ 24 fps (1440 frames, ~90 chunks):
    FP8                 ~14 hours    (production quality finals)
    FP8_DISTILL_PREVIEW  ~4.4 hours  (fast iteration; 3.15x faster)

LRU model cache (introduced in commit 47162ab) ensures the cold-load
happens ONCE; subsequent chunks share the same cache key and are pure
cache hits.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"


def _detect_comfy_output_dir(cli_arg: str | None = None) -> Path:
    """Locate ComfyUI's output directory so we can find the per-chunk
    PNGs written by SaveImage nodes.

    Resolution order:
      1. Explicit CLI arg if provided (`--comfy-output-dir <path>`)
      2. `UNIVIDX_COMFY_OUTPUT` env var
      3. Live query: GET /system_stats and infer from --output-directory
         in the running ComfyUI process's argv
      4. Conventional fallback: `<repo_root>/output` (works for portable
         ComfyUI installs where the repo IS the comfy root)

    Raises FileNotFoundError if none of these resolves to an existing
    directory.
    """
    candidates: list[Path] = []
    if cli_arg:
        candidates.append(Path(cli_arg))
    env_val = os.environ.get("UNIVIDX_COMFY_OUTPUT")
    if env_val:
        candidates.append(Path(env_val))
    # Try the live ComfyUI server: --output-directory is in /system_stats argv.
    try:
        with urllib.request.urlopen(f"{BASE}/system_stats", timeout=5) as resp:
            stats = json.load(resp)
        argv = stats.get("system", {}).get("argv", []) or []
        for i, a in enumerate(argv):
            if a == "--output-directory" and i + 1 < len(argv):
                candidates.append(Path(argv[i + 1]))
                break
    except Exception:
        pass
    # Conventional fallback for portable ComfyUI installs that look like
    # <ComfyUI>/custom_nodes/UniVidX_ComfyUI/, so output is at
    # <ComfyUI>/output/ — three parents up from this file.
    candidates.append(REPO.parent.parent / "output")

    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "Could not locate ComfyUI's output directory. Pass "
        "--comfy-output-dir <path>, set UNIVIDX_COMFY_OUTPUT in the "
        "environment, or run this script from a system where the "
        f"running ComfyUI server reports it via /system_stats.\n"
        f"Tried: {[str(c) for c in candidates]}"
    )


# Resolved lazily in main() after CLI parsing. Module-level callers
# should use _detect_comfy_output_dir() directly.
COMFY_OUTPUT: Path = REPO.parent.parent / "output"  # placeholder for type hints

# Map our preset names onto the loader/sampler settings the workflow
# templates expect.
PRESETS: dict[str, dict] = {
    # Production (0.4.0 default) — best quality, slowest. Use for final
    # deliverables.
    "FP8": {
        "loader": {"prefer_sage_attn": False,
                   "dit_weight_mode": "fp8_prequantized",
                   "step_distill_lora": "none"},
        "sampler": {"num_inference_steps": 20, "cfg_scale": 5.0},
    },
    # Fast preview (0.5.0 NEW) — FP8 + LightX2V step distill at 4 steps
    # cfg=1. 3.15x faster than old PRODUCTION; ~22-26 dB PSNR vs BF16
    # reference (visibly different but plausible decompositions). Use
    # for iteration loops and long-clip processing.
    "FP8_DISTILL_PREVIEW": {
        "loader": {"prefer_sage_attn": False,
                   "dit_weight_mode": "fp8_prequantized",
                   "step_distill_lora": "lightx2v",
                   "step_distill_strength": 1.0},
        "sampler": {"num_inference_steps": 4, "cfg_scale": 1.0},
    },
    # Legacy presets — kept for back-compat. The 0.5.0 measurements
    # show sage and compile_dit are regressions on top of FP8 on this
    # workload, but they're left here in case someone has a different
    # config where they help.
    "PRODUCTION": {
        "loader": {"prefer_sage_attn": True,
                   "dit_weight_mode": "bf16_shards",
                   "step_distill_lora": "none"},
        "sampler": {"num_inference_steps": 20, "cfg_scale": 5.0},
    },
    "FP8_SAGE": {
        "loader": {"prefer_sage_attn": True,
                   "dit_weight_mode": "fp8_prequantized",
                   "step_distill_lora": "none"},
        "sampler": {"num_inference_steps": 20, "cfg_scale": 5.0},
    },
    "PREVIEW": {
        "loader": {"prefer_sage_attn": True,
                   "dit_weight_mode": "bf16_shards",
                   "step_distill_lora": "none"},
        "sampler": {"num_inference_steps": 4, "cfg_scale": 1.0},
    },
    "FP8_PREVIEW": {
        "loader": {"prefer_sage_attn": False,
                   "dit_weight_mode": "fp8_prequantized",
                   "step_distill_lora": "none"},
        "sampler": {"num_inference_steps": 4, "cfg_scale": 1.0},
    },
}

# Map a UniVidX task mode to (workflow template, output modality names).
MODE_WORKFLOWS: dict[str, dict] = {
    "R2AIN":  {"template": "R2AIN_video_api.json",
               "modalities": ["placeholder", "albedo", "irradiance", "normal"]},
    "R2PFB":  {"template": "R2PFB_video_api.json",
               "modalities": ["composite", "alpha", "foreground", "background"]},
    # Add more as your workflow library grows.
}


# ---------------------------------------------------------------------------
# Chunk planning (deterministic, unit-tested)
# ---------------------------------------------------------------------------

def plan_chunks(total_frames: int, chunk_size: int = 21,
                overlap: int = 5) -> list[tuple[int, int]]:
    """Return a list of (start, end_exclusive) frame indices covering
    [0, total_frames) with chunks of size `chunk_size` overlapping by
    `overlap` frames. The final chunk is anchored to total_frames so no
    frames are dropped at the tail.

    Invariants:
      - Every chunk is exactly chunk_size frames long.
      - Consecutive chunks overlap by exactly `overlap` frames (except
        possibly the last, which anchors to the tail).
      - All total_frames frames are covered.
      - Returns at least one chunk for any total_frames >= chunk_size.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"invalid chunking: chunk_size={chunk_size}, overlap={overlap}; "
            "require chunk_size > 0 and 0 <= overlap < chunk_size"
        )
    if total_frames < chunk_size:
        raise ValueError(
            f"source has only {total_frames} frames; need at least "
            f"{chunk_size} (chunk_size). Use a shorter chunk_size or a "
            f"longer source clip."
        )
    stride = chunk_size - overlap
    chunks: list[tuple[int, int]] = []
    i = 0
    while i + chunk_size <= total_frames:
        chunks.append((i, i + chunk_size))
        i += stride
    # If the last chunk's end is < total_frames, add a final chunk
    # anchored to total_frames. This may overlap the previous chunk by
    # more than `overlap`, but is necessary to cover the tail.
    if chunks and chunks[-1][1] < total_frames:
        chunks.append((total_frames - chunk_size, total_frames))
    elif not chunks:
        # total_frames == chunk_size exactly
        chunks.append((0, chunk_size))
    return chunks


# ---------------------------------------------------------------------------
# Source-video probing
# ---------------------------------------------------------------------------

def probe_video(path: Path) -> dict:
    """Return {n_frames, fps, width, height} for a source video via ffprobe."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not on PATH; install ffmpeg")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets,avg_frame_rate,width,height",
        "-of", "json", str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    info = json.loads(out)["streams"][0]
    num, den = info["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) > 0 else 0.0
    return {
        "n_frames": int(info["nb_read_packets"]),
        "fps": fps,
        "width": int(info["width"]),
        "height": int(info["height"]),
    }


# ---------------------------------------------------------------------------
# Workflow queueing
# ---------------------------------------------------------------------------

def build_chunk_workflow(template_path: Path, source: Path, chunk_idx: int,
                          start_frame: int, frame_count: int,
                          width: int, height: int,
                          preset_loader: dict, preset_sampler: dict,
                          seed: int, prefix_tag: str) -> dict:
    """Load a workflow template and customize for one chunk."""
    with template_path.open(encoding="utf-8") as f:
        wf = json.load(f)
    # Find VHS_LoadVideoPath, UniVidXLoader, UniVidXSampler, SaveImage nodes.
    for nid, node in wf.items():
        ct = node.get("class_type")
        if ct == "VHS_LoadVideoPath":
            node["inputs"]["video"] = str(source)
            node["inputs"]["skip_first_frames"] = start_frame
            node["inputs"]["frame_load_cap"] = frame_count
            node["inputs"]["select_every_nth"] = 1
            node["inputs"]["custom_width"] = width
            node["inputs"]["custom_height"] = height
        elif ct == "UniVidXLoader":
            for k, v in preset_loader.items():
                node["inputs"][k] = v
        elif ct == "UniVidXSampler":
            for k, v in preset_sampler.items():
                node["inputs"][k] = v
            node["inputs"]["seed"] = seed
            node["inputs"]["num_frames"] = frame_count
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
        elif ct == "SaveImage":
            base = node["inputs"]["filename_prefix"]
            node["inputs"]["filename_prefix"] = (
                f"{prefix_tag}_chunk{chunk_idx:03d}_{base.split('_')[-1]}"
            )
    return wf


def _http_post(url: str, payload: dict, timeout: float = 30.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _http_get(url: str, timeout: float = 60.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def queue_and_wait(wf: dict, chunk_idx: int,
                    timeout_sec: float = 3600.0) -> dict:
    body = _http_post(f"{BASE}/prompt",
                      {"prompt": wf, "client_id": f"chunked-{chunk_idx}"})
    prompt_id = body["prompt_id"]
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
        if now - last_log > 60.0:
            print(f"    [{time.strftime('%H:%M:%S')}] chunk {chunk_idx:03d} running",
                  flush=True)
            last_log = now
        time.sleep(5)
    raise TimeoutError(f"chunk {chunk_idx:03d} ({prompt_id}) timed out")


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def load_chunk_frames(prefix_tag: str, chunk_idx: int,
                       modality: str) -> np.ndarray:
    """Load all PNGs for a (chunk, modality) pair into [T, H, W, C] uint8."""
    pattern = f"{prefix_tag}_chunk{chunk_idx:03d}_*{modality}*.png"
    paths = sorted(COMFY_OUTPUT.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no outputs for chunk {chunk_idx} {modality}")
    arrs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    return np.stack(arrs, axis=0)  # [T, H, W, 3]


def build_crossfade_weights(chunk_plan: list[tuple[int, int]],
                             chunk_idx: int) -> np.ndarray:
    """Per-frame crossfade weight vector for chunk ``chunk_idx``.

    Length matches that chunk's frame count (``end - start``). The
    weight ramps linearly up across the overlap with the previous
    chunk, plateaus at 1.0, then ramps linearly down across the
    overlap with the next chunk. The first chunk has no head ramp;
    the last has no tail ramp.

    Extracted from ``stitch_modality`` so the math can be unit-tested
    independently of disk I/O and PIL.
    """
    start, end = chunk_plan[chunk_idx]
    T = end - start
    w = np.ones(T, dtype=np.float32)
    if chunk_idx > 0:
        prev_end = chunk_plan[chunk_idx - 1][1]
        overlap = max(0, prev_end - start)
        for j in range(overlap):
            w[j] = (j + 1) / (overlap + 1)
    if chunk_idx < len(chunk_plan) - 1:
        next_start = chunk_plan[chunk_idx + 1][0]
        overlap = max(0, end - next_start)
        for j in range(overlap):
            w[T - 1 - j] = (j + 1) / (overlap + 1)
    return w


def stitch_modality(chunk_plan: list[tuple[int, int]], prefix_tag: str,
                     modality: str, total_frames: int) -> np.ndarray:
    """Stitch per-chunk frames into a single [total_frames, H, W, 3] array
    with linear crossfade across overlap regions."""
    chunk_arrs = [load_chunk_frames(prefix_tag, i, modality)
                  for i in range(len(chunk_plan))]
    H, W = chunk_arrs[0].shape[1], chunk_arrs[0].shape[2]
    output = np.zeros((total_frames, H, W, 3), dtype=np.float32)
    weight = np.zeros((total_frames,), dtype=np.float32)
    for ci, ((start, end), arr) in enumerate(zip(chunk_plan, chunk_arrs)):
        w = build_crossfade_weights(chunk_plan, ci)
        T = end - start
        for f_local in range(T):
            f_global = start + f_local
            if 0 <= f_global < total_frames:
                output[f_global] += arr[f_local].astype(np.float32) * w[f_local]
                weight[f_global] += w[f_local]
    weight = np.clip(weight, 1e-6, None)
    output = (output / weight[:, None, None, None]).clip(0, 255).astype(np.uint8)
    return output


def encode_to_mp4(frames: np.ndarray, out_path: Path, fps: float) -> None:
    """Encode an [T, H, W, 3] uint8 array to mp4 via imageio."""
    import imageio.v3 as iio
    iio.imwrite(out_path, frames, fps=fps, codec="libx264",
                quality=8, pixelformat="yuv420p")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Source video file (mp4, mov, etc.)")
    ap.add_argument("--mode", default="R2AIN",
                    choices=list(MODE_WORKFLOWS.keys()))
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Directory for stitched MP4s")
    ap.add_argument("--preset", default="FP8_DISTILL_PREVIEW",
                    choices=list(PRESETS.keys()))
    ap.add_argument("--chunk-size", type=int, default=21)
    ap.add_argument("--overlap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--comfy-output-dir", type=str, default=None,
                    help="ComfyUI's output directory (where SaveImage "
                         "writes per-chunk PNGs we then stitch). If "
                         "omitted, resolved via the UNIVIDX_COMFY_OUTPUT "
                         "env var, the live /system_stats argv, or a "
                         "portable-install fallback. See "
                         "_detect_comfy_output_dir() for the full order.")
    args = ap.parse_args()
    global COMFY_OUTPUT
    COMFY_OUTPUT = _detect_comfy_output_dir(args.comfy_output_dir)
    print(f"ComfyUI output dir: {COMFY_OUTPUT}")

    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(args.input)
    print(f"source: {info['n_frames']} frames @ {info['fps']:.2f} fps, "
          f"{info['width']}x{info['height']}")
    print(f"target render: {args.width}x{args.height} per chunk "
          f"(may differ from source)")
    print(f"preset: {args.preset}  mode: {args.mode}  seed: {args.seed}")

    chunks = plan_chunks(info["n_frames"], args.chunk_size, args.overlap)
    print(f"plan: {len(chunks)} chunks (chunk_size={args.chunk_size}, "
          f"overlap={args.overlap})")

    template = REPO / "examples" / MODE_WORKFLOWS[args.mode]["template"]
    if not template.is_file():
        print(f"workflow template missing: {template}", file=sys.stderr)
        return 3
    preset = PRESETS[args.preset]
    run_id = time.strftime("run%Y%m%d-%H%M%S")
    prefix_tag = f"chunked_{args.mode}_{run_id}"

    failures: list[int] = []
    t_start = time.time()
    for ci, (start, end) in enumerate(chunks):
        print(f"\n[chunk {ci+1}/{len(chunks)}] frames [{start}..{end})")
        wf = build_chunk_workflow(
            template_path=template, source=args.input.resolve(),
            chunk_idx=ci, start_frame=start, frame_count=end - start,
            width=args.width, height=args.height,
            preset_loader=preset["loader"], preset_sampler=preset["sampler"],
            seed=args.seed, prefix_tag=prefix_tag,
        )
        try:
            t0 = time.time()
            entry = queue_and_wait(wf, ci)
            t1 = time.time()
            status = entry.get("status", {}).get("status_str")
            if status != "success":
                print(f"  chunk {ci} status={status}; will skip", flush=True)
                failures.append(ci)
            else:
                print(f"  chunk {ci} done in {(t1 - t0)/60:.2f} min", flush=True)
        except Exception as exc:
            print(f"  chunk {ci} EXCEPTION: {type(exc).__name__}: {exc}",
                  flush=True)
            failures.append(ci)

    t_all = time.time() - t_start
    print(f"\nall chunks done in {t_all/60:.1f} min "
          f"({len(failures)} failures: {failures})")

    print("\nstitching modalities...")
    modalities = MODE_WORKFLOWS[args.mode]["modalities"]
    for m in modalities:
        try:
            print(f"  stitching {m}...")
            stitched = stitch_modality(chunks, prefix_tag, m, info["n_frames"])
            out_path = args.output_dir / f"{run_id}_{args.mode}_{m}.mp4"
            encode_to_mp4(stitched, out_path, info["fps"])
            print(f"    wrote {out_path}")
        except FileNotFoundError as exc:
            print(f"    SKIPPED {m}: {exc}")

    print(f"\nresults in {args.output_dir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
