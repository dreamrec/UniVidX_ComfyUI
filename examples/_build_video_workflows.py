"""Generate R2AIN_video_api.json and R2PFB_video_api.json — the API-format
video-conditioned workflows that load 21 evenly-spaced frames from an MP4
via VHS_LoadVideoPath and feed them as RGB conditioning into UniVidX.

Self-contained: builds the node graph in-memory rather than reading from a
template JSON, so deleting other example workflows does not break this.

Run from repo root:  python examples/_build_video_workflows.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"

# VHS_LoadVideoPath validates `video` against ComfyUI's resolved input dir
# AND against absolute paths. The plain filename form fails validation in
# this build ("Invalid file path"), so we ship an absolute path that
# matches the demo MP4 we used for the README results — users edit it.
DEMO_VIDEO_PATH = "C:/Users/dr5090/Documents/ComfyUI/input/LTX_2.3_t2v_00239_.mp4"

DEFAULT_PROMPT = (
    "a cinematic portrait of a young man with long brown hair, white shirt, "
    "candlelit room, ornate wallpaper, gothic atmosphere, soft warm lighting"
)

# UniVidX's standard Chinese negative prompt — same as upstream inference.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)

VARIANTS = [
    dict(
        out="R2AIN_video_api.json",
        variant="intrinsic", mode="R2AIN",
        height=480, width=640,
        prefix="unividx_LTX_R2AIN",
        decode_class="UniVidXDecodeIntrinsic",
        save_names=["placeholder", "albedo", "irradiance", "normal"],
    ),
    dict(
        out="R2PFB_video_api.json",
        variant="alpha", mode="R2PFB",
        height=432, width=768,
        prefix="unividx_LTX_R2PFB",
        decode_class="UniVidXDecodeAlpha",
        save_names=["placeholder", "matte", "foreground", "background"],
    ),
]


def build(cfg: dict) -> dict:
    return {
        "1": {"class_type": "UniVidXLoader",
              "inputs": {
                  "variant": cfg["variant"],
                  "dtype": "bfloat16",
                  # 0.5.0 default: FP8 prequantized. ~13% faster than BF16
                  # on RTX 5090 (9.43 min vs 10.85 min on this workflow),
                  # 50% lower DiT VRAM, PSNR >= 30 dB per modality vs BF16
                  # reference. Set to "bf16_shards" if you want the 0.3.x
                  # baseline behavior. See README for the full perf matrix.
                  "dit_weight_mode": "fp8_prequantized",
              }},
        "2": {"class_type": "UniVidXTaskMode",
              "inputs": {"mode": cfg["mode"]}},
        "3": {"class_type": "VHS_LoadVideoPath",
              "inputs": {
                  "video": DEMO_VIDEO_PATH,
                  "force_rate": 0,
                  "custom_width": cfg["width"],
                  "custom_height": cfg["height"],
                  "frame_load_cap": 21,
                  "skip_first_frames": 0,
                  "select_every_nth": 23,
                  "format": "Wan",
              }},
        "5": {"class_type": "UniVidXSampler",
              "inputs": {
                  "model": ["1", 0],
                  "task": ["2", 0],
                  "rgb": ["3", 0],
                  "prompt": DEFAULT_PROMPT,
                  "negative_prompt": NEGATIVE_PROMPT,
                  "num_inference_steps": 20,
                  "cfg_scale": 5.0,
                  "denoising_strength": 1.0,
                  "num_frames": 21,
                  "height": cfg["height"],
                  "width": cfg["width"],
                  "seed": 42,
                  "tiled": True,
              }},
        "6": {"class_type": cfg["decode_class"],
              "inputs": {"result": ["5", 0]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0],
                         "filename_prefix": f"{cfg['prefix']}_{cfg['save_names'][0]}"}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 1],
                         "filename_prefix": f"{cfg['prefix']}_{cfg['save_names'][1]}"}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 2],
                         "filename_prefix": f"{cfg['prefix']}_{cfg['save_names'][2]}"}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["6", 3],
                          "filename_prefix": f"{cfg['prefix']}_{cfg['save_names'][3]}"}},
    }


def main() -> None:
    for cfg in VARIANTS:
        wf = build(cfg)
        out_path = EXAMPLES / cfg["out"]
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(wf, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path.relative_to(REPO)} ({len(wf)} nodes)")


if __name__ == "__main__":
    main()
