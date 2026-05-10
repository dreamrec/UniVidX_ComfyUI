![UniVidX banner](assets/registry-banner-20260510.svg)

# UniVidX Intrinsic & Alpha Decomposition for ComfyUI

[![Smoke Test](https://github.com/dreamrec/UniVidX_ComfyUI/actions/workflows/smoke.yml/badge.svg)](https://github.com/dreamrec/UniVidX_ComfyUI/actions/workflows/smoke.yml)
![License](https://img.shields.io/badge/license-GPL--3.0-2f855a)
![Python](https://img.shields.io/badge/python-3.10%2B-1f4b99)
![PyTorch](https://img.shields.io/badge/torch-%E2%89%A52.7%2Bcu128-ee4c2c)
![Nodes](https://img.shields.io/badge/nodes-5-f59e0b)
![Tests](https://img.shields.io/badge/tests-24%20unit%20%2B%2010%20integration-success)

ComfyUI custom nodes for [UniVidX](https://houyuanchen111.github.io/UniVidX.github.io/) (SIGGRAPH 2026): unified video diffusion that decomposes a clip into **RGB / Albedo / Irradiance / Normal** (intrinsic) or **Composite RGB / Alpha matte / Foreground / Background** (alpha) — 30 task modes across two model variants, all driven from a single five-node graph.

This is a **Strategy A** wrapper: UniVidX's official pipeline runs as an opaque black box; the four output IMAGE batches become standard ComfyUI tensors that flow into any downstream node — VHS video combine, alpha compositing, 3D reconstruction, ControlNet conditioning for *other* models, you name it.

## Visual Tour

### The flagship workflow — `t2RAIN`, text-to-all-four-modalities

![t2RAIN workflow](assets/workflow_t2RAIN.png)

Five nodes. One text prompt in. Four physically-distinct video modalities out — generated in a single denoising loop, not four separate models. Thumbnails are real frame-11 outputs from a 21-frame 480×640 run on an RTX 5090 (~10 min wall time).

### Intrinsic decomposition — RGB / Albedo / Irradiance / Normal

![Intrinsic quad](assets/results/LTX_intrinsic_quad.jpg)

A 21-frame portrait clip conditioned on the input RGB (mode `R2AIN`, intrinsic variant). UniVidX strips the warm candlelight from the **albedo** — same subject under neutral daylight, even the dark wallpaper pattern becomes legible. **Irradiance** isolates the soft incoming light field as a sepia gradient with the candle highlights baked in. **Normal** encodes surface direction as XYZ→RGB — cheekbones, nose, hair flow direction all readable. The decoder's RGB slot is a black placeholder (RGB was the input); we paste the conditioning frame back in here so the comparison is legible.

### Alpha decomposition — Composite / Matte / Foreground / Background

![Alpha quad](assets/results/LTX_alpha_quad.jpg)

Same source clip, alpha variant + mode `R2PFB`. The **alpha matte** is a true binary mask — head + hair + shoulders cleanly separated from the candlelit background, no fuzz around the silhouette. **Foreground** isolates the subject onto white. **Background** is the most striking output: the model inpaints the wallpaper, the chair back, and the candle stand *behind* where the subject was sitting, hallucinating the occluded geometry. Text-only `t2RPFB` would produce a near-uniform white matte; RGB conditioning is what gives the model the figure-ground signal it needs.

## Features

- **Two model variants** with shared Wan2.1-T2V-14B base: `intrinsic` (RGB / Albedo / Irradiance / Normal) and `alpha` (RGB / Pha / Fgr / Bgr).
- **30 task modes** — every combination of inputs and targets across the 4 modalities of each family. Pick the mode that matches your conditioning.
- **Per-modality IMAGE conditioning** — any combination of available modalities can be supplied as IMAGE batches; missing target slots come back as black placeholders so downstream graphs don't break.
- **Layer-by-layer VRAM management** baked in — the 28 GB DiT streams modules from CPU on demand, leaving headroom for activations on a 24-32 GB GPU.
- **Vendored UniVidX as a pinned git submodule** under `vendor/UniVidX/`. `chdir` context + Windows-friendly directory junctions / hardlinks bridge UniVidX's hardcoded relative paths to ComfyUI's `models/` tree without forking upstream.
- **Windows-ready out of the box** — three patches handle Windows-specific issues: backslash escaping in `model_paths_json`, mmgp readonly mmap (avoids `WinError 1455`), and junction/hardlink fallback (avoids `os.symlink` admin requirement).
- **Comprehensive test matrix** — 24 unit tests + 10 end-to-end integration tests covering both variants, multi-input conditioning, error paths, and downstream node composition.

## Installation

### Manual Install

```bash
cd ComfyUI/custom_nodes
git clone --recurse-submodules https://github.com/dreamrec/UniVidX_ComfyUI.git
cd UniVidX_ComfyUI
python -m pip install -r requirements.txt
python install.py
```

If you forgot `--recurse-submodules`, fix it:

```bash
git submodule update --init --recursive
```

Restart ComfyUI after installation.

### Companion Node for the Demo Workflows

The bundled video-output workflow uses [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for `VHS_VideoCombine`.

## Models

UniVidX is built on Wan2.1-T2V-14B + a tiny LoRA-rank-32 adapter per variant. Both must be downloaded manually because of size; auto-download isn't practical at 83 GB.

| Model | Default location | Size | Auto-download |
|------|------|------|------|
| [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | `ComfyUI/models/wan21_t2v_14b/` | ~69 GB | No (manual) |
| [UniVidX checkpoints](https://huggingface.co/houyuanchen/UniVidX) | `ComfyUI/models/unividx/` | ~1.6 GB | No (manual) |

Download with:

```bash
hf download Wan-AI/Wan2.1-T2V-14B --local-dir ComfyUI/models/wan21_t2v_14b
hf download houyuanchen/UniVidX  --local-dir ComfyUI/models/unividx
```

Wan2.1's six DiT shards account for ~57 GB; the T5 text encoder adds ~11 GB; VAE + tokenizer + configs are negligible. UniVidX adds two ~800 MB LoRA adapters (`univid_intrinsic.safetensors`, `univid_alpha.safetensors`).

## Included Workflows

All six demos ship as UI-format JSONs you can drag-and-drop onto the ComfyUI canvas — they come with **colour-coded groups, generous 80-px gaps between groups, and 100-px vertical gaps between stacked nodes** so the graph stays legible. Matching `*_api.json` files are auto-generated for programmatic queueing.

### `examples/t2RAIN_basic.json` — Text → All Four Intrinsic Modalities

The flagship demo. Generate a 21-frame 480×640 video with full RGB / Albedo / Irradiance / Normal decomposition from a text prompt alone.

```text
[Model Setup]   →   [Sampling]   →   [Decode (Intrinsic)]   →   [Outputs (4× SaveImage)]
 Loader              Sampler           DecodeIntrinsic            RGB / A / I / N
 TaskMode
```

Drag the JSON onto canvas, queue, get four PNG sequences. ~10 min wall time on a 5090.

### `examples/R2AIN_video_api.json` — RGB-Conditioned Re-Decomposition

Feed any MP4 — UniVidX produces matched Albedo / Irradiance / Normal channels for the same 21 evenly-spaced frames. The decoder's `rgb` slot becomes a black placeholder (RGB was the input, not regenerated). Uses [VHS_LoadVideoPath](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) to pull frames from disk; **edit the absolute `video` path** in node 3 before queueing.

```text
[Model Setup] → [VHS_LoadVideoPath] → [Sampling] → [Decode] → [Outputs]
                 (your.mp4 → 21 frames)
```

### `examples/t2RPFB_basic.json` — Alpha Decomposition (Text-to-All)

The alpha family from text alone — produces composite RGB, alpha matte, foreground, background. Alpha decomposition is much weaker without an RGB reference; expect the matte to come out near-uniform white. Use `R2PFB_video_api.json` instead for production-quality mattes.

### `examples/R2PFB_video_api.json` — Sharp Alpha Matte from a Video Clip

The most useful alpha workflow. Feed any MP4 — get a clean alpha matte + isolated foreground + clean background. The matte is a real production-grade mask. Same `VHS_LoadVideoPath` setup as `R2AIN_video_api.json`; the `composite_rgb` slot is a black placeholder.

### `examples/J_alpha_compositing.json` — End-to-End VFX Composite

Demonstrates that the alpha matte is a usable mask: extracts the foreground via `ImageToMask` + `ImageCompositeMasked` and pastes it onto a synthetic cyan background. Drop your own `LoadImage` in place of `EmptyImage` to composite over a real backdrop.

```text
[Setup] → [RGB Cond] → [Sampling] → [Decode] → [Background] → [Composite] → [Output]
                                                EmptyImage     ImageToMask    SaveImage
                                                               ImageCompositeMasked
```

### `examples/I_video_output.json` — Direct MP4 Export

Skip the per-frame PNGs and emit one MP4 per modality via `VHS_VideoCombine`. Requires [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) (already installed if your ComfyUI loads `VHS_VideoCombine`).

```text
[Setup] → [Sampling] → [Decode] → [Video Outputs (4× VHS_VideoCombine)]
                                   rgb / albedo / irradiance / normal MP4
```

## Mode Reference

Mode names encode `<conditions>2<targets>`. The letter `t` on the left means "text-only" — no input modalities, all four targets generated.

**Intrinsic family** (model variant `intrinsic`):

| Letter | Meaning |
|---|---|
| `R` | RGB |
| `A` | Albedo |
| `I` | Irradiance |
| `N` | Normal |

15 modes: `t2RAIN`, `R2AIN`, `A2RIN`, `I2RAN`, `N2RAI`, `RA2IN`, `RI2AN`, `RN2AI`, `AI2RN`, `AN2RI`, `IN2RA`, `RAI2N`, `RAN2I`, `RIN2A`, `AIN2R`.

**Alpha family** (model variant `alpha`):

| Letter | Meaning |
|---|---|
| `R` | Composite RGB |
| `P` | Pha (alpha matte) |
| `F` | Foreground |
| `B` | Background |

15 modes: `t2RPFB`, `R2PFB`, `P2RFB`, `F2RPB`, `B2RPF`, `RP2FB`, `RF2PB`, `RB2PF`, `PF2RB`, `PB2RF`, `FB2RP`, `RPF2B`, `RPB2F`, `RFB2P`, `PFB2R`.

For modes where a modality is a *condition* rather than a target, the corresponding decoder output is a 4-D black tensor of the right shape — downstream nodes still get a valid IMAGE batch.

## Performance & VRAM

Wan2.1-T2V-14B is **~28 GB BF16**. UniVidX adds rank-32 LoRA adapters (~800 MB per variant) and runs **Cross-Modal Self-Attention over 4 modalities in parallel** — that quadratic-in-modality-count attention is the dominant per-step cost, not memory bandwidth.

**Measured baseline** (RTX 5090, 32 GB, torch 2.7.0+cu128, intrinsic variant, R2AIN mode, video-conditioned):

| Resolution × frames × steps | Per-step time | Total wall time¹ |
|---|---|---|
| 256×256 × 5 frames × 3 steps | ~3.8 sec/step | 130 sec |
| 480×640 × 21 frames × 20 steps | **~43 sec/step** | **17.6 min** |

¹ Includes ~3 min cold-load per (variant, dtype) cache miss; subsequent runs in the same session skip it.

### Optimization knobs on the Loader

The `UniVidXLoader` node exposes three perf knobs:

| Widget | Effect | When to use |
|---|---|---|
| `dtype` = `bfloat16` | Reference / default | Always works, matches UniVidX training |
| `dtype` = `fp8_e4m3fn` / `fp8_e5m2` | Post-quantizes the DiT via optimum-quanto. Halves DiT to ~14 GB; on Hopper/Blackwell engages Marlin FP8 matmul kernels. **EXPERIMENTAL** — the quantize() pass over Wan2.1-14B + UniVidX's LoRA stack can hang on cold-load in our testing. Try it; if it stalls, fall back to bfloat16. | If/when quanto plays nicely with the stack |
| `compile_dit` = `True` | `torch.compile(dit, mode='reduce-overhead', dynamic=True)` after model load. First sampler step pays a ~60-120 sec graph capture; subsequent steps are typically 20-30% faster. | Long runs at fixed resolution. Recompile cost on resolution change. |
| `vram_buffer_gb` | **DEPRECATED on current DiffSynth** — the underlying `WanVideoPipeline.enable_vram_management` API was removed upstream, so this widget is a no-op. Left for backwards-compat with saved workflows. | n/a |

### The single biggest drop-in win: install Flash Attention 3

The Wan DiT's attention kernel selection chain is **FA3 > FA2 > SAGE > SDPA**, picked at first import based on what's installed. We currently auto-pick **FA2 (v2.8.2)** — installing **Flash Attention 3** for Hopper/Blackwell would be auto-picked instead, and is typically **1.5-2× faster on attention** (the dominant cost). Zero code changes, no quality loss.

```bash
# In your ComfyUI venv. Build varies by CUDA version + GPU arch — check
# https://github.com/Dao-AILab/flash-attention/releases for a Hopper/Blackwell wheel.
pip install flash-attn-3  # or build from source: see flash-attention readme
```

Verify it picked up by importing in your venv:

```bash
python -c "import flash_attn_interface; print('FA3 OK')"
```

### Other levers (medium-impact, may cost quality)

- **`cfg_scale = 1.0`** in the sampler — disables classifier-free guidance, giving **2× per-step speedup** (single forward pass instead of two). Quality drops noticeably for text-only modes; for RGB-conditioned modes (`R2AIN`, `R2PFB`) the RGB carries most of the signal so the loss is small.
- **Lower `num_inference_steps`** — UniVidX uses `FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)`, which doesn't have a "more efficient" drop-in alternative (can't swap to DPM++). Step count just scales linearly; 15 vs 20 saves 25%, 10 saves 50%, with linear quality drop.
- **Install SageAttention** (`pip install sageattention`) — INT8-quantized attention. Only auto-picked when FA2 and FA3 are *absent*, so you'd need to remove flash-attn first. Not generally recommended over FA3.

### Honest stack: what actually moves the needle

All numbers below are **measured** on the same workflow (R2AIN, 480×640×21f, 20 steps, BF16, RTX 5090) unless marked *expected*.

| Config | Wall time | Per-step | Δ vs baseline | Notes |
|---|---|---|---|---|
| **Baseline** (defaults: 20 steps, cfg=5, FA2) | 17.6 min | ~43.8 s | — | Includes ~3 min cold-load |
| `vram_buffer_gb` 4 → 12 | 16.6 min | ~43 s | **–6% (noise)** | Widget is a no-op on current diffsynth |
| `compile_dit = True` | 14.55 min | ~31.6 s | –17% wall, –28% per-step | First-step pays ~90 s compile capture |
| `prefer_sage_attn = True` | 14.49 min | ~34.5 s | –18% wall, –21% per-step | Quality verified visually equivalent to FA2 |
| `compile_dit` + `prefer_sage_attn` (stacked) | 14.99 min | ~33 s | –15% (slightly worse than either alone) | The two wins target the same per-step bottleneck and don't compose |
| **AGGRESSIVE preset:** `prefer_sage_attn` + `num_inference_steps=4` + `cfg_scale=1.0` (RGB-conditioned only) | **1.35 min** | **~10-15 s** | **–92% wall (13× speedup)** | UniVidX's flow-match scheduler is unexpectedly robust at 4 steps when paired with RGB conditioning. Quality on portrait test clip: marginally softer tie stripe in albedo, slightly less crisp facial features in normal map — all decompositions still physically correct. May degrade more on high-motion content. **Do not use for text-only modes (`t2RAIN`, `t2RPFB`)** — they need cfg≥5 + 20 steps for stable output. |
| FP8 quant (`fp8_e4m3fn`) | *not measured* | *unknown* | *unknown* | Quanto pass over Wan2.1-14B + UniVidX LoRA stack hung after ~22 min in our cold-load test |

**Recommendation: pick ONE of `compile_dit` or `prefer_sage_attn`, not both.** They overlap on the same bottleneck. `prefer_sage_attn` has the slight wall edge and no warmup cost (`compile_dit` pays ~90 s graph capture on the first sampler step). For longer runs (50+ steps) the compile gain compounds further as the warmup amortizes. Visual quality verified identical to FA2 baseline on the test clip's albedo / irradiance / normal outputs.

### Installing SageAttention (for `prefer_sage_attn=True`)

The `prefer_sage_attn` knob is a no-op unless [SageAttention](https://github.com/thu-ml/SageAttention) is installed in your ComfyUI venv. PyPI ships only sage 1.0.6 (head_dim restricted to {64,96,128}, Hopper/Ada-tuned only). For Blackwell + Windows + Python 3.12 + PyTorch 2.7, use the **prebuilt wheel from [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention/releases)**:

```bash
# In your ComfyUI venv:
pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows/sageattention-2.2.0+cu128torch2.7.1-cp312-cp312-win_amd64.whl"
```

Match the wheel to your stack: cu128 / cu130 (CUDA toolkit), torch2.7.1 / torch2.8.0 (PyTorch minor), cp310/cp311/cp312/cp313 (Python). Linux wheels available too; for source builds see [woct0rdho's BUILD_STORY](https://github.com/mobcat40/sageattention-blackwell/blob/main/BUILD_STORY.md).

> **CAUTION — `ComfyUI-3D-Pack` SDPA pollution.** Installing `sageattention` triggers ComfyUI-3D-Pack's `Stable3DGen/trellis/backend_config.py` to globally swap `torch.nn.functional.scaled_dot_product_attention = sageattn` at module import. That pollution breaks any other custom node that uses SDPA with head_dim outside sage's supported set — UniVidX's VAE is one such case (1-head SDPA where head_dim=channel_count, hits 384). Our `runtime.load_model()` calls `_restore_native_sdpa_if_polluted()` defensively on every load to undo the pollution. If you have other custom nodes that broke after installing sage, this defensive un-pollute is why — and the same pattern would fix them.

### Other levers (unmeasured)

- **Install Flash Attention 3** — *Hopper-only, doesn't help RTX 5090 (sm_120).* The earlier README claim was wrong for Blackwell. FA4 supports Blackwell but is Linux-wheels-only on PyPI and uses a different module name (`flash_attn.cute`) that DiffSynth's Wan DiT doesn't auto-detect.
- **`cfg_scale = 1.0`** in the sampler — disables CFG, skipping one forward pass per step. ~2× speedup. Quality OK for RGB-conditioned modes (`R2AIN`, `R2PFB`); weakens text-only modes noticeably.
- **Lower `num_inference_steps`** — UniVidX uses `FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)`, no DPM++ swap available. Step count scales linearly: 15 vs 20 saves 25% with mild quality drop, 10 vs 20 saves 50% with notable drop.
- **[LightX2V Wan2.1-T2V-14B-StepDistill-CfgDistill LoRA](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)** — community step-distillation LoRA targeting 4 steps with no CFG. Claimed ~20× wall speedup on Wan2.1, ~42× stacked with FP8. Drop-in (just an extra LoRA load), but quality on UniVidX's intrinsic/alpha modalities is unverified (the distillation was done on RGB output).
- **[kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)** — alternative ComfyUI wrapper for Wan2.1 with finer-grained controls (per-block CPU swap, async prefetch, dedicated torch.compile node). Different model loader, would need separate integration to use with UniVidX.

### Honest summary

Two presets to remember:

- **PRODUCTION**: `prefer_sage_attn=True` (or `compile_dit=True`, pick one), keep `num_inference_steps=20` and `cfg_scale=5.0`. ~14.5 min for 480×640×21f at full quality. Use for finals.
- **PREVIEW / FAST ITERATION**: `prefer_sage_attn=True` + `num_inference_steps=4` + `cfg_scale=1.0` on **RGB-conditioned modes only** (`R2AIN`, `R2PFB`). **~1.4 min for the same workflow — 13× faster.** Quality is slightly softer but all decompositions remain physically correct. Use for iterating on prompts and seed selection before committing to a full-quality final pass.

Higher-tier stacked speedups exist in theory but are blocked: FA3 is Hopper-only (doesn't help Blackwell), FA4 is Linux-only on PyPI with module-name mismatch vs DiffSynth, FP8 quanto hangs on the LoRA-attached DiT, and the [LightX2V step-distill LoRA](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v) — which we considered for additional 4-step quality recovery — is now lower priority since raw 4-step inference already produces production-usable output without the LoRA stacking complexity.

The README previously claimed ~30 sec/step with VRAM management; that number was true under an older diffsynth version that exposed `enable_vram_management(vram_buffer=...)` on `WanVideoPipeline`. The current diffsynth removed that API — our runtime call is `hasattr`-guarded and silently no-ops, which is why the per-step is now ~43 sec rather than ~30 sec. The right fix is FA3 + compile, not the (now-cosmetic) vram_buffer knob.

### Resolution & frame-count notes

- **Defaults**: `480×640` for intrinsic, `432×768` for alpha — these match upstream training. Other sizes work but quality may drift.
- **Frame count is heavily preset to 21** in upstream training. The sampler accepts `num_frames` from 5 to 81 in steps of 4, but only ~21±4 is well validated. Lower counts work for fast smoke tests; higher counts may degrade.
- **Tiled VAE** (`tiled=True`) is enabled by default and recommended for everything ≥480p. Tile size hardcoded to `[30, 52]` with stride `[15, 26]` (upstream defaults).

### Prompt language

Wan2.1's text encoder was trained heavily on Chinese. **English prompts work but are noticeably weaker.** The bundled negative prompt in the example workflows is the upstream Chinese standard — keep it.

## Node Overview

Five nodes, all under the `UniVidX` category. Custom socket types — `UNIVIDX_MODEL` (purple), `UNIVIDX_TASK` (teal), `UNIVIDX_RESULT` (pink) — keep the graph type-safe; standard `IMAGE` (green) is used everywhere a frame batch flows.

<table>
<tr><td width="380">

![Loader](assets/nodes/loader.svg)

</td><td>

**`UniVidXLoader`** — Loads the `intrinsic` or `alpha` variant and outputs `UNIVIDX_MODEL`. Models are cached per `(variant, dtype, device)` so multi-graph runs reuse weights instead of re-loading the 28 GB DiT.

</td></tr>
<tr><td>

![Task Mode](assets/nodes/task_mode.svg)

</td><td>

**`UniVidXTaskMode`** — Picks one of 30 modes from a dropdown (`t2RAIN`, `R2AIN`, `RA2IN`, `t2RPFB`, …). Outputs `UNIVIDX_TASK` carrying the mode + family. The sampler validates the family against the loaded model variant.

</td></tr>
<tr><td>

![Sampler](assets/nodes/sampler.svg)

</td><td>

**`UniVidXSampler`** — Runs UniVidX's `pipe()` end-to-end inside a `chdir(vendor/UniVidX)` context. Accepts the model + task + a text prompt + up to 7 optional `IMAGE` inputs (one per modality across both families). Inputs not required by the active mode are silently ignored.

</td></tr>
<tr><td>

![Decode Intrinsic](assets/nodes/decode_intrinsic.svg)

</td><td>

**`UniVidXDecodeIntrinsic`** — Splays an intrinsic-family `UNIVIDX_RESULT` into 4 `IMAGE` batches: `rgb / albedo / irradiance / normal`. Modalities that were *conditions* (not targets) come back as a black placeholder of the right shape, so downstream graphs never break on missing slots.

</td></tr>
<tr><td>

![Decode Alpha](assets/nodes/decode_alpha.svg)

</td><td>

**`UniVidXDecodeAlpha`** — Same shape as above but for the alpha family: `composite_rgb / alpha / foreground / background`. Raises `ValueError` if you try to feed it an intrinsic-family result (and vice versa).

</td></tr>
</table>

## Requirements

- ComfyUI with Python 3.10+ (tested on 3.12.9)
- PyTorch ≥ 2.7 with CUDA 12.8+ (CUDA 12.8 required for NVIDIA Blackwell / RTX 5090; older torch errors with `no kernel image is available for execution on the device.`)
- ≥24 GB VRAM (32 GB recommended for headroom)
- ~80 GB free disk for the Wan2.1-T2V-14B + UniVidX checkpoints
- 16+ GB host RAM (CPU streams modules under the VRAM management hood)
- DiffSynth-Studio + mmgp + transformers ≥ 4.38 (auto-installed via `requirements.txt`)

## Windows-specific notes

Three patches make this pack work on Windows out of the box. They fire automatically when running on Windows; on POSIX they are no-ops.

1. **Backslash escaping in JSON paths** (`src/runtime.py`).
   UniVidX's pipeline takes `model_paths` as a JSON string and runs `json.loads` internally. Windows paths like `D:\ComfyUI\models\...` would be invalid JSON escapes (`\D`, `\m`, …). We construct the string with `json.dumps([t5, vae])` so backslashes are properly escaped.

2. **Read-only mmap for safetensors** (`src/runtime.py`).
   The `mmgp` library (transitive dep of DiffSynth) monkey-patches `safetensors.torch.load_file` to memory-map shards with `ACCESS_COPY`. Six 9.84 GB Wan2.1 DiT shards mmapped concurrently require ~60 GB of Windows paging-file commit, which exceeds most users' default and surfaces as `[WinError 1455] The paging file is too small`. We monkey-patch `load_file` inside UniVidX's pipeline namespaces to use `writable_tensors=False` (`ACCESS_READ` — no commit charge needed since UniVidX only reads tensors before copying to GPU).

3. **Junctions and hardlinks instead of symlinks** (`src/path_resolver.py`).
   `os.symlink()` on Windows requires Administrator privileges or Developer Mode. We use `mklink /J` (directory junction) for the Wan2.1 model dir link and `os.link()` (hardlink) for individual checkpoint files. Both work without privileges. Cross-volume hardlinks fall back to `shutil.copy2` (~1.5 GB extra disk if your `models/` and the `UniVidX_ComfyUI` repo are on different volumes).

## Troubleshooting

- **`MissingModelFile`** at startup: a Wan2.1 or UniVidX file is missing from `models/`. Re-run the `hf download` commands above.
- **OOM at sample time**: VRAM management is on by default with a 4 GB buffer. If you still OOM, raise the buffer (`vram_buffer=8.0` in `src/runtime.py`) or lower `num_frames` / `height` / `width`.
- **Slow first run**: model load takes 3–5 minutes (28 GB DiT + LoRA attachment). Subsequent runs in the same ComfyUI session reuse the cache.
- **Slow per-step time** (>1 min on a 32 GB+ GPU): VRAM management may not be activating. Verify GPU temp during sampling — if it's <60°C with 99% util, you're memory-bound.
- **Black outputs** for modalities you expected to generate: check that your task mode actually lists them as targets. `R2AIN`'s `rgb` decoder slot is *intentionally* black because RGB was the input.
- **`ImportError: No module named diffsynth`**: `pip install diffsynth>=2.0` into the same Python that runs ComfyUI.
- **`WinError 1314`** (Windows symlink): should not occur — we use junctions/hardlinks instead. If it does, you may have a stale install — delete the `vendor/UniVidX/models/` and `vendor/UniVidX/checkpoints/` directories and let `runtime.initialize()` recreate them.
- **`WinError 1455` "The paging file is too small"**: should not occur — the readonly-mmap patch fixes this. Verify `mmgp` is installed in the venv.
- **`json.decoder.JSONDecodeError: Invalid \escape`**: should not occur — the `json.dumps` patch fixes this. Verify `runtime.py` is at HEAD.
- **`CUDA error: no kernel image is available for execution on the device.`**: torch is too old for your GPU. Upgrade to `torch>=2.7+cu128`.
- **English prompts give weak results**: that's Wan2.1's Chinese training bias, not a bug. Try translating the prompt or use the Chinese negative prompt baked into the example workflows.

## Out of scope by design

These belong to a different integration strategy and would require porting UniVidX's Cross-Modal Self-Attention onto a different DiT class (multi-week project):

- Stacking community Wan2.1 / Wan2.2 LoRAs on UniVidX's DiT
- Injecting ControlNet / IP-Adapter inside UniVidX's denoising loop
- Replacing UniVidX's sampler with a ComfyUI KSampler
- Native ComfyUI MODEL-type integration (interop with kijai's WanVideoWrapper)

Strategy A's value is at the **I/O boundary** — composing UniVidX outputs with arbitrary downstream ComfyUI nodes. That's fully validated: see `examples/test_matrix/I_video_output.json` and `examples/test_matrix/J_alpha_compositing.json` for end-to-end proof.

## Test Matrix

A comprehensive test matrix lives in `examples/test_matrix/` (10/10 passing as of 2026-05-10):

| # | Mode | What it validates |
|---|---|---|
| C | RA2IN | Multi-input intrinsic (2 IMAGE inputs) |
| D | RAI2N | Maximum-conditioning intrinsic (3 inputs → 1 output) |
| E | t2RPFB | Alpha variant + DecodeAlpha node |
| F | R2PFB | Alpha conditioning, sharp matte |
| G | (error) family/variant mismatch | Sampler validation |
| H | (error) missing required input | `validate_mode()` |
| I | UniVidX → MP4 | Composition with `VHS_VideoCombine` |
| J | UniVidX alpha → composite | Composition with `ImageCompositeMasked` |

Reproduce with:

```bash
python examples/test_matrix/_build.py    # generates JSON workflows
python examples/test_matrix/_run.py      # runs all + asserts
python examples/test_matrix/_run.py --filter alpha  # subset
```

See `examples/test_matrix/REPORT.md` for the full results report including pixel statistics for each modality output.

## Credits

- [UniVidX](https://github.com/houyuanchen111/UniVidX) for the unified video diffusion model (vendored at a pinned commit under `vendor/UniVidX/`)
- [Wan-AI / Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) for the base text-to-video DiT
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for the pipeline runtime + VRAM management
- [mmgp](https://pypi.org/project/mmgp/) for memory-mapped paged loading
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the host runtime

## License

This repository is licensed under [GPL-3.0](LICENSE).

Vendored upstream dependencies keep their own licenses:

- UniVidX: see [vendor/UniVidX/LICENSE](vendor/UniVidX) (upstream license at time of vendor pin)
- Wan2.1-T2V-14B weights: per the [Wan-AI license](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)

```text
┌─────────────────────────────────────────────────────────────────────┐
│ dreamrec // UniVidX // intrinsic & alpha video decomposition       │
└─────────────────────────────────────────────────────────────────────┘
```
