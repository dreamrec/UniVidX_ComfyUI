# Example Workflows

Drag-and-drop ready ComfyUI workflows for UniVidX. Each `*.json` is a UI workflow you can drop onto the canvas; each `*_api.json` is the matching programmatic-queue payload (regenerate with `_build_ui_workflows.py` after editing).

> **As of 0.5.0**, all shipped workflows default to `dit_weight_mode=fp8_prequantized` — the FP8 path that gives a 13% wall-time speedup and 50% DiT VRAM cut on RTX 5090. Set the widget to `bf16_shards` if you want the 0.3.x baseline behavior. See the [main README](../README.md) for the full perf matrix.

> **Before you queue anything:** make sure both model packs are downloaded — see [Model setup](#model-setup) below. Without them, the loader node raises `MissingModelFile` at startup.

## Quick start

```bash
# 1. download models (once, ~83 GB total)
hf download Wan-AI/Wan2.1-T2V-14B  --local-dir ComfyUI/models/wan21_t2v_14b
hf download houyuanchen/UniVidX    --local-dir ComfyUI/models/unividx

# 2. (optional, 0.5.0+) download the LightX2V distill LoRA for the
#    fast-preview mode (4-step + cfg=1, ~5 min/chunk):
hf download lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v \
    loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors \
    --local-dir ComfyUI/models/loras/lightx2v

# 3. open ComfyUI in your browser, drag any *.json from this folder onto the canvas
# 4. queue it — the first run takes ~2-3 min cold-load + sampling time
```

## Workflow catalogue

Wall times measured on RTX 5090 (32 GB), R2AIN_video @ 480×640×21 frames, 20 steps, with the 0.5.0 default loader settings (`dit_weight_mode=fp8_prequantized`).

| File | Mode | Variant | Inputs | Targets | Wall time¹ |
|---|---|---|---|---|---|
| [`t2RAIN_basic.json`](t2RAIN_basic.json) | `t2RAIN` | intrinsic | text | RGB + Albedo + Irradiance + Normal | ~9-10 min |
| [`t2RAIN_tiny_api.json`](t2RAIN_tiny_api.json) | `t2RAIN` | intrinsic | text | RGB + Albedo + Irradiance + Normal | ~2 min |
| [`R2AIN_video_api.json`](R2AIN_video_api.json) | `R2AIN` | intrinsic | RGB **video clip** via `VHS_LoadVideoPath` | Albedo + Irradiance + Normal | **9.43 min** *(measured)* |
| [`t2RPFB_basic.json`](t2RPFB_basic.json) | `t2RPFB` | alpha | text | Composite + Pha + Fgr + Bgr | ~10-12 min |
| [`R2PFB_video_api.json`](R2PFB_video_api.json) | `R2PFB` | alpha | RGB **video clip** via `VHS_LoadVideoPath` | Pha + Fgr + Bgr | **12.36 min** *(measured)* |
| [`I_video_output.json`](I_video_output.json) | `t2RAIN` | intrinsic | text | 4× MP4 via VHS_VideoCombine | ~9-10 min |
| [`J_alpha_compositing.json`](J_alpha_compositing.json) | `R2PFB` | alpha | RGB still | Composited PNG | ~10-12 min |

> **The `_video_` workflows** are the canonical RGB-conditioned demos. They load 21 evenly-spaced frames from an MP4 via `VHS_LoadVideoPath` and feed them as conditioning into UniVidX. **Edit node 3's `video` field to point at your own MP4** — VHS requires an absolute path on this build (`Invalid file path` error otherwise). Both workflows require [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite).

¹ First run in a session adds ~2-3 min for cold-load + FP8 quantization walk. Subsequent runs hit the model cache and skip both. The `_tiny_` variant uses 256×256 × 5 frames × 3 steps for fast smoke tests.

## Processing clips longer than 21 frames

UniVidX is trained at 21 frames per inference. For source clips longer than ~1 second, use the **chunked sampler driver**:

```bash
# Fast preview / iteration (FP8 + LightX2V step-distill, 4 steps cfg=1)
# 1 min @ 24 fps clip → ~4.4 hours
python examples/chunked_clip_sampler.py \
    --input  C:/path/to/your_clip.mp4 \
    --mode   R2AIN \
    --output-dir  C:/path/to/output \
    --preset FP8_DISTILL_PREVIEW    # default in 0.5.0

# Production finals (FP8, 20 steps cfg=5)
# 1 min @ 24 fps clip → ~14 hours
python examples/chunked_clip_sampler.py \
    --input  C:/path/to/your_clip.mp4 \
    --mode   R2AIN \
    --output-dir  C:/path/to/output \
    --preset FP8
```

The driver slices your source into overlapping 21-frame windows, runs UniVidX on each, and stitches the per-modality outputs into 4 MP4s with a linear crossfade across the overlap. See the [main README's "Processing longer clips"](../README.md#processing-longer-clips-chunked-sampler) section for the full preset table + quality caveats.

### `t2RAIN_basic.json` — Text → all four intrinsic modalities

The flagship demo. Type a prompt, get RGB + Albedo + Irradiance + Normal as four 21-frame PNG sequences. Useful as a sanity check after install and as a baseline for the other modes.

```text
[Loader (intrinsic)]      [Task Mode (t2RAIN)]
        \                        /
         \                      /
          ▾                    ▾
          [Sampler] ──── result ──── ▸ [Decode Intrinsic] ──┬─▸ SaveImage (rgb)
              ▴                                              ├─▸ SaveImage (albedo)
              │   prompt:                                    ├─▸ SaveImage (irradiance)
              │   "a parrot flapping its wings..."           └─▸ SaveImage (normal)
```

**Tip:** the bundled negative prompt is the upstream Chinese standard. Keep it. English-only negatives noticeably weaken the result because Wan2.1's text encoder was trained on Chinese.

### `R2AIN_video_api.json` — RGB-conditioned re-decomposition (video input)

Feed any MP4. UniVidX produces matched Albedo / Irradiance / Normal for the same 21 evenly-spaced frames. The decoder's `rgb` slot is a black placeholder (RGB was the input, not regenerated).

Edit node 3's `video` field to point at your own clip (absolute path required). Tweak `select_every_nth` if your source clip is much shorter or longer than ~480 frames — the goal is 21 frames spaced across whatever you want UniVidX to see.

### `t2RPFB_basic.json` — Text → composite/matte/fg/bg

The alpha family from text alone. Be aware: text-only alpha decomposition tends to produce a near-uniform white matte (`mean ≈ 254.9`, `std ≈ 0.7` in our test runs) because the model can't decide what's foreground without an RGB reference. Use `R2PFB_video_api.json` instead for production-quality mattes.

### `R2PFB_video_api.json` — Sharp alpha matte from a video clip

The most useful alpha workflow. Feed an MP4, get a clean alpha matte + isolated foreground + clean background. The matte is a binary-quality mask (in our LTX test run, `mean ≈ 32`, `std ≈ 80` — sharp figure-ground separation). The `composite_rgb` decoder slot is black (RGB was the input). Same `VHS_LoadVideoPath` setup as `R2AIN_video_api.json` — edit node 3's path.

### `I_video_output.json` — Direct MP4 export

Same graph as `t2RAIN_basic.json` but the four `SaveImage` nodes are replaced with [VHS_VideoCombine](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) — you get four MP4 files per run instead of 84 PNGs. Requires ComfyUI-VideoHelperSuite installed.

### `J_alpha_compositing.json` — End-to-end VFX composite

Demonstrates that the alpha matte is a real binary-quality mask. Pipeline:

```text
[R2PFB sampler] ─▸ [Decode Alpha] ─┬─▸ alpha ──▸ ImageToMask ──┐
                                    └─▸ foreground ─────────────┴─▸ ImageCompositeMasked ─▸ SaveImage
                                              [EmptyImage cyan bg] ────────▴
```

Drop your own `LoadImage` in place of `EmptyImage` to composite the foreground onto a real backdrop.

## Test matrix (`test_matrix/`)

A separate set of tighter workflows built for CI / regression testing — all use the **tiny** config (256×256, 5 frames, 3 steps, seed 42) so each test runs in 25-30 sec.

| File | Mode | Variant | What it validates |
|---|---|---|---|
| [`C_RA2IN.json`](test_matrix/C_RA2IN.json) | `RA2IN` | intrinsic | Multi-input intrinsic (2 IMAGE inputs) |
| [`D_RAI2N.json`](test_matrix/D_RAI2N.json) | `RAI2N` | intrinsic | Maximum-conditioning intrinsic (3 inputs → 1 output) |
| [`E_t2RPFB.json`](test_matrix/E_t2RPFB.json) | `t2RPFB` | alpha | Alpha variant + DecodeAlpha node |
| [`F_R2PFB.json`](test_matrix/F_R2PFB.json) | `R2PFB` | alpha | Alpha conditioning, sharp matte |
| [`G_error_family_mismatch.json`](test_matrix/G_error_family_mismatch.json) | `t2RPFB` | intrinsic | Sampler validation rejects family/variant mismatch |
| [`H_error_missing_input.json`](test_matrix/H_error_missing_input.json) | `R2AIN` | intrinsic | `validate_mode()` rejects missing required input |
| [`I_video_output.json`](test_matrix/I_video_output.json) | `t2RAIN` | intrinsic | UniVidX outputs flow into `VHS_VideoCombine` |
| [`J_alpha_compositing.json`](test_matrix/J_alpha_compositing.json) | `R2PFB` | alpha | UniVidX alpha → `ImageCompositeMasked` |

Reproduce the whole matrix:

```bash
python examples/test_matrix/_build.py     # regenerate JSONs from the templates
python examples/test_matrix/_run.py       # run all + assert
python examples/test_matrix/_run.py --filter alpha   # run subset
```

See [`test_matrix/REPORT.md`](test_matrix/REPORT.md) for the full pass report including per-modality pixel statistics from the most recent run (10/10 pass as of 2026-05-10).

## Model setup

Both model packs are downloaded manually because of size. Auto-download isn't practical at 83 GB total.

| Pack | Where it goes | Size | Files |
|---|---|---|---|
| [Wan-AI / Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | `ComfyUI/models/wan21_t2v_14b/` | ~69 GB | 6 DiT shards (~9.84 GB each), T5 encoder (~11 GB), VAE, tokenizer, configs |
| [houyuanchen / UniVidX](https://huggingface.co/houyuanchen/UniVidX) | `ComfyUI/models/unividx/` | ~1.6 GB | `univid_intrinsic.safetensors` (~800 MB), `univid_alpha.safetensors` (~800 MB) |

```bash
# install the Hugging Face CLI if you don't have it
pip install -U "huggingface_hub[cli]"

# download
hf download Wan-AI/Wan2.1-T2V-14B --local-dir ComfyUI/models/wan21_t2v_14b
hf download houyuanchen/UniVidX   --local-dir ComfyUI/models/unividx
```

Re-run the `hf download` commands to repair partial downloads — the CLI skips already-complete files.

The bundled `python install.py` creates a Windows directory junction (or POSIX symlink) from `vendor/UniVidX/models/` to `ComfyUI/models/wan21_t2v_14b/`, plus hardlinks for the two LoRA adapters. This bridges UniVidX's hardcoded relative paths to ComfyUI's `models/` tree without forking upstream.

### Companion node packs

| Workflow | Requires |
|---|---|
| `I_video_output.json`, `test_matrix/I_video_output.json` | [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) (provides `VHS_VideoCombine`) |
| All others | None — only ComfyUI core nodes + UniVidX |

## Common gotchas

- **`MissingModelFile` at startup** — re-run the `hf download` commands. The path resolver lists the exact missing file.
- **`R2AIN` `rgb` output is black** — that's correct. `R2AIN` uses RGB as input, so the decoder's `rgb` slot returns a black placeholder of the right shape (so downstream nodes don't break on a missing key).
- **Text-only alpha matte (`t2RPFB`) comes out white** — known model limitation, not a bug. Use `R2PFB_video_api.json` instead and feed a video clip.
- **OOM on a 24 GB GPU** — make sure `dit_weight_mode=fp8_prequantized` on the loader (drops DiT from ~28 GB to ~14 GB). If you're already on FP8 and still OOM, raise `vram_buffer_gb` to 8 in the loader widget, or lower `num_frames` / `height` / `width` in the sampler.
- **Per-step time > 1 min on a ≥32 GB GPU** — VRAM management didn't activate (or you're on the BF16 path without it). Switch the loader to `dit_weight_mode=fp8_prequantized` — the FP8 DiT fits fully resident on a 32 GB card so layer streaming isn't needed. See the [main README perf matrix](../README.md#full-performance-matrix-050-rtx-5090-r2ain_video--480640214-frames).
- **First run takes 3-5 min before sampling starts** — that's the cold model load (28 GB DiT + LoRA attachment). Subsequent runs in the same ComfyUI session hit the cache and skip this.
- **Switching between `intrinsic` and `alpha` reloads the model** — the cache is keyed per-variant. Group your runs by variant if you have many to do.
- **Resolution / frame count is rigid** — Wan2.1 was trained at `480×640` (intrinsic) and `432×768` (alpha) with 21 frames. Other sizes work but quality drifts. The sampler caps at 81 frames in steps of 4.
- **English prompts work but are noticeably weaker** — Wan2.1's text encoder was trained heavily on Chinese. Translate your prompt or use ChatGPT-style translation; the bundled negative prompt is already in the upstream Chinese form.

## File reference

```text
examples/
├── README.md                     ← you are here
├── _build_ui_workflows.py        ← regenerates all *.json from templates after node-schema edits
├── _smoke_runner.py              ← submits the t2RAIN UI workflow for end-to-end smoke
├── _smoke_runner_tiny.py         ← variant for the tiny config (256×256 × 5f × 3 steps)
├── _build_ui_workflows.py        ← regenerates the UI-format JSONs (text-only + I + J)
├── _build_video_workflows.py     ← regenerates R2AIN_video_api / R2PFB_video_api
├── t2RAIN_basic.json             ← UI workflow (drag onto canvas)
├── t2RAIN_basic_api.json         ← API payload (POST to /prompt)
├── t2RAIN_tiny_api.json          ← API payload, tiny config (CI)
├── t2RPFB_basic.json + _api.json
├── R2AIN_video_api.json          ← video-conditioned intrinsic (edit `video` path)
├── R2PFB_video_api.json          ← video-conditioned alpha (edit `video` path)
├── I_video_output.json + _api.json
├── J_alpha_compositing.json + _api.json
└── test_matrix/
    ├── REPORT.md                 ← latest pass report with pixel statistics
    ├── _build.py                 ← regenerate matrix JSONs
    ├── _run.py                   ← run + assert
    └── C_RA2IN.json … J_alpha_compositing.json   ← the 8 matrix tests
```

## Going further

- **Custom downstream graphs** — UniVidX outputs are standard ComfyUI `IMAGE` batches. Anything that consumes `IMAGE` (latent decode, upscalers, ControlNet conditioners, video combine, image compositing, mask ops) just works. See `J_alpha_compositing.json` for a worked example with `ImageToMask` + `ImageCompositeMasked`.
- **Programmatic queueing** — use the `*_api.json` payloads with the ComfyUI `/prompt` endpoint. `_smoke_runner.py` is a working reference.
- **Other modes** — the 30 task modes are all valid; just pick a different one in `UniVidXTaskMode`. Mode names encode `<conditions>2<targets>` — see the [README mode reference](../README.md#mode-reference) for the full list.
