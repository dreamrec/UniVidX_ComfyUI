![UniVidX banner](assets/registry-banner-20260510.svg)

# UniVidX Intrinsic & Alpha Decomposition for ComfyUI

[![Smoke Test](https://github.com/dreamrec/UniVidX_ComfyUI/actions/workflows/smoke.yml/badge.svg)](https://github.com/dreamrec/UniVidX_ComfyUI/actions/workflows/smoke.yml)
![License](https://img.shields.io/badge/license-GPL--3.0-2f855a)
![PyTorch](https://img.shields.io/badge/torch-%E2%89%A52.7%2Bcu128-ee4c2c)
![Nodes](https://img.shields.io/badge/nodes-5-f59e0b)
![Tests](https://img.shields.io/badge/tests-85%20unit%20%2B%2010%20integration-success)
![GPU](https://img.shields.io/badge/GPU-%E2%89%A532%20GB%20VRAM%20%7C%20RTX%205090%20validated-blueviolet)

ComfyUI custom nodes for [UniVidX](https://houyuanchen111.github.io/UniVidX.github.io/) (SIGGRAPH 2026): unified video diffusion that decomposes a clip into **RGB / Albedo / Irradiance / Normal** (intrinsic) or **Composite RGB / Alpha matte / Foreground / Background** (alpha). 30 task modes across two model variants, all driven from a single five-node graph.

> ## ⚠️ Hardware requirements (read this first)
>
> This is a **14-billion-parameter video diffusion model** running locally. It is **not lightweight**. Verify your system can handle it before installing.
>
> **Minimum to run at all:**
>
> | Resource | Requirement |
> |---|---|
> | **GPU VRAM** | **≥ 24 GB** with the 0.5.0 FP8 path; **≥ 32 GB** for the BF16 baseline |
> | **GPU architecture** | CUDA compute capability **8.0+** (Ampere / RTX 3000+, Ada / RTX 4000+, Hopper / H100, Blackwell / RTX 5000+) |
> | **System RAM** | **≥ 32 GB** (peak ~28 GB during cold-load of the BF16 DiT shards); 64+ GB comfortable |
> | **Disk** | **~85 GB free** (Wan2.1-T2V-14B 69 GB + UniVidX 1.6 GB + optional LightX2V 0.6 GB + working space) |
> | **PyTorch** | **≥ 2.7 with CUDA 12.8** (older torch errors on Blackwell GPUs with `no kernel image is available`) |
> | **Python** | **3.10+** (tested on 3.12.9) |
> | **ComfyUI** | **0.20+** |
>
> **Validated configuration (all benchmarks in this README):** RTX 5090 (32 GB Blackwell sm_120), Windows 11, Python 3.12.9, PyTorch 2.7.0+cu128, ComfyUI Desktop 0.20.1.
>
> **Card-by-card honest read:**
>
> | GPU | Status | Notes |
> |---|---|---|
> | **RTX 5090** (32 GB Blackwell) | ✅ **Validated** | The benchmark target. FP8 path: 9.43 min/chunk; preview: 4.59 min/chunk. |
> | **RTX 4090** (24 GB Ada, FP8 native) | 🟢 **Should work; not validated by us** | FP8 path fits cleanly (~14 GB DiT + activations). BF16 path tight but possible with `vram_buffer_gb=8+`. Expect ~10-15% slower than 5090 due to memory bandwidth. |
> | **RTX 6000 Ada / A6000** (48 GB Ada) | 🟢 **Should work** | Plenty of headroom for both paths. Could batch 2-3 clips in parallel with custom orchestration. |
> | **RTX 6000 Pro Blackwell** (96 GB) | 🟢 **Should work, ideal for batch** | Sweet spot if you process many clips per day. Same per-clip speed as 5090; massive parallelism headroom. |
> | **H100 / H200 / B200** (80-192 GB datacenter) | 🟢 **Should work, overkill for inference** | Same per-clip speed as Blackwell consumer. Worth the cost only if you're also fine-tuning. |
> | **RTX 3090 / 3090 Ti** (24 GB Ampere, no native FP8) | 🟡 **Should work, slower** | FP8 path runs via software cast (no Blackwell tensor-core FP8). Memory fits; per-step ~20-30% slower than 4090. |
> | **RTX 4080 / 5080** (16 GB) | 🔴 **Will OOM** | FP8 DiT alone is ~14 GB. Add activations + VAE + text encoder → exceeds 16 GB at production resolution. |
> | **RTX 3080 / 4070 / 4060 Ti** (12 GB) | ❌ **Cannot run** | Insufficient VRAM even with aggressive layer streaming. |
> | **Pre-Ampere** (RTX 20-series, V100, P100, etc.) | ❌ **Cannot run** | CUDA compute capability < 8.0. Wan2.1's Flash-Attention-2 path needs sm_80+. |
>
> **Cloud option:** if you don't have local hardware, **rent an L40 (48 GB) or RTX 6000 (48 GB) instance** from RunPod / Vast.ai / Lambda Labs — they're typically $0.50-1.00/hour and a single 1-min clip via the chunked sampler (FP8 preview, ~4.4 hr) runs for ~$3-5 of compute.
>
> **Per-clip wall times** (RTX 5090 reference; all measured):
> - One 21-frame chunk @ 480×640: **~9.43 min** (production FP8) or **~4.59 min** (fast preview FP8+distill)
> - 1-minute @ 24 fps clip via chunked sampler: **~14 hours** (production) or **~4.4 hours** (preview)
> - This is **not real-time**. Plan workflows around overnight / multi-hour processing.

**What you'd use it for:** relighting (swap the irradiance channel, recombine), VFX alpha pulls without a green screen (a clean matte from any clip), 3D reconstruction pipelines that need normals + albedo as conditioning, ControlNet-style guidance for *other* video models that consume normal maps.

**Strategy A wrapper** — UniVidX's official pipeline runs as an opaque black box. The four output IMAGE batches become standard ComfyUI tensors that flow into any downstream node (VHS video combine, alpha compositing, 3D reconstruction, ControlNet for *other* models, etc.).

### Use this when

- You specifically need **intrinsic** (RGB / Albedo / Irradiance / Normal) or **alpha** (matte / fg / bg) decomposition of a video clip — that's UniVidX's whole reason to exist. No other Wan2.1 wrapper does this.
- You want clean drag-and-drop ComfyUI workflows for the 30 task modes without writing pipeline code.

### Use [`kijai/ComfyUI-WanVideoWrapper`](https://github.com/kijai/ComfyUI-WanVideoWrapper) when

- You want generic Wan2.1/2.2 T2V or I2V (just RGB out, not decomposition).
- You need finer-grained per-block CPU swap, async prefetch, or kijai-curated FP8/GGUF Wan checkpoints. Their wrapper has more model-management surface; ours has the UniVidX-specific decomposition head.

## Visual tour

### Flagship workflow — `t2RAIN`, text → all four intrinsic modalities

![t2RAIN workflow](assets/workflow_t2RAIN.png)

### Intrinsic decomposition

![Intrinsic quad](assets/results/LTX_intrinsic_quad.jpg)

A 21-frame portrait clip conditioned on RGB (mode `R2AIN`). UniVidX strips the candlelight from the **albedo**, isolates the soft incoming **irradiance** field, and emits a clean **normal map** of the face. The decoder's RGB slot is a black placeholder (RGB was the input); we paste the conditioning frame back in here for legibility.

### Alpha decomposition

![Alpha quad](assets/results/LTX_alpha_quad.jpg)

Same source clip, alpha variant + mode `R2PFB`. The **alpha matte** is a true binary-quality mask. **Background** is the most striking output: the model inpaints the wallpaper, chair, and candle stand *behind* where the subject was sitting.

## Quick start

```bash
# 1. Install. The --recurse-submodules flag pulls in the UniVidX vendor
#    repo (~500 MB of upstream Python + small assets — no Git LFS needed).
cd ComfyUI/custom_nodes
git clone --recurse-submodules https://github.com/dreamrec/UniVidX_ComfyUI.git
cd UniVidX_ComfyUI
python -m pip install -r requirements.txt
python install.py        # creates Win junction / POSIX symlink. No admin needed.

# 2. Install the Hugging Face CLI if you don't have it, then download models.
#    ~83 GB total — UniVidX is built on Wan2.1-T2V-14B which is the bulk.
pip install -U "huggingface_hub[cli]"
hf download Wan-AI/Wan2.1-T2V-14B  --local-dir ComfyUI/models/wan21_t2v_14b
hf download houyuanchen/UniVidX    --local-dir ComfyUI/models/unividx

# 3. Restart ComfyUI, drag examples/t2RAIN_basic.json onto the canvas, queue.
```

**ComfyUI Desktop / portable / manual** — paths above assume a layout where `ComfyUI/models/` is a sibling of `ComfyUI/custom_nodes/`. ComfyUI Desktop installs put `models/` under `Documents/ComfyUI/`; the `python install.py` step auto-resolves either layout.

For real video-clip conditioning (your own MP4), use [`examples/R2AIN_video_api.json`](examples/R2AIN_video_api.json) (intrinsic) or [`examples/R2PFB_video_api.json`](examples/R2PFB_video_api.json) (alpha). They load 21 evenly-spaced frames from disk via `VHS_LoadVideoPath`, which means you'll also need [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) installed.

## Two recommendations (0.5.0)

**For production finals (best quality):**

```
UniVidXLoader.dit_weight_mode = fp8_prequantized
```

Everything else default. Wall: **9.43 min** on R2AIN_video. Quality verified PSNR ≥ 30 dB per modality against BF16. Use this for anything you'd ship.

**For iteration / long-clip processing (fast):**

```
UniVidXLoader.dit_weight_mode    = fp8_prequantized
UniVidXLoader.step_distill_lora  = lightx2v
UniVidXLoader.step_distill_strength = 1.0
UniVidXSampler.num_inference_steps = 4
UniVidXSampler.cfg_scale           = 1.0
```

Wall: **4.59 min** on R2AIN_video — **3.15× faster than 0.3.x PRODUCTION**, 2× faster than 0.4.0 FP8 alone. Quality is ~22-26 dB PSNR vs BF16 — visibly different decompositions but plausible content. **Use this for iteration loops, long-clip processing (with `chunked_clip_sampler.py`), or anywhere "fast and pretty good" beats "slow and pristine."** Don't ship this output as a final deliverable without an eye check.

## Full performance matrix (0.5.0, RTX 5090, R2AIN_video @ 480×640×21 frames)

| Configuration | Steps | cfg | Wall (min) | Δ vs BF16 baseline | Notes |
|---|---|---|---|---|---|
| **BF16 baseline** (no extras) | 20 | 5.0 | 10.85 | 0% | The reference point. ~28 GB DiT, vram_buffer streaming. |
| **🏆 FP8 baseline** (`dit_weight_mode=fp8_prequantized`) | 20 | 5.0 | **9.43** | **−13.1%** | **Production default.** ~14 GB DiT fully resident. |
| **🚀 FP8 + distill (lightx2v)** | 4 | 1.0 | **4.59** | **−57.7%** | **Fast preview / iteration.** 3.15× vs old PRODUCTION. Quality ~22-26 dB PSNR. |
| BF16 + distill (lightx2v) | 4 | 1.0 | 5.77 | −46.8% | FP8 strictly better than BF16 under distill too. |
| BF16 + sage | 20 | 5.0 | 14.48 | +33.5% | sage_attn is +33% wall on this workload; *not* the −18% advertised by 0.2.0. |
| FP8 + sage | 20 | 5.0 | 11.75 | +8.3% | sage compounds with FP8 negatively. |
| FP8 + compile_dit | 20 | 5.0 | 11.65 | +7.4% | Graph-captures cleanly on FP8Linear but no per-step speedup on top of FP8's residency. |
| FP8 alpha (R2PFB) | 20 | 5.0 | 12.36 | +14% | Alpha variant works; slightly slower than intrinsic. |
| FP8 PREVIEW + sage | 4 | 1.0 | 6.20 | (different config) | Cold-load dominates short runs. |
| FP8 text-only tiny (t2RAIN 256×256×5×3) | 3 | 5.0 | 4.66 | (different config) | Tiny smoke baseline. |

### Quality (FP8 vs BF16, R2AIN_video, same seed, 21 frames)

| Modality | PSNR (dB) | Threshold | Verdict |
|---|---|---|---|
| placeholder (RGB output slot when RGB is condition) | inf (exact) | ≥30 | PASS |
| albedo | 30.89 | ≥30 | PASS |
| irradiance | 39.17 | ≥30 | PASS, comfortable margin |
| normal | 36.28 | ≥30 | PASS, comfortable margin |

## Using FP8 (new in 0.4.0)

Set this on `UniVidXLoader`:

```
variant            = intrinsic     (or alpha)
dtype              = bfloat16
dit_weight_mode    = fp8_prequantized
vram_buffer_gb     = 4.0
prefer_sage_attn   = False
compile_dit        = False
```

How it works under the hood: after UniVidX's standard BF16 cold-load, the loader walks the DiT, computes per-tensor absmax scales for each Linear layer, casts the weights to `torch.float8_e4m3fn`, and replaces each Linear with an `FP8Linear` that dequantizes on forward. UniVidX's per-modality LoRA adapters (the four `lora_A/B_<mod>` pairs at each attention block) are preserved at BF16 by walking through PEFT wrappers and replacing only the inner base layer. No external file needed — when a Kijai `Wan2_1-T2V-14B_fp8_e4m3fn_scaled.safetensors` lands upstream and is dropped into `models/diffusion_models/`, the loader will use it directly instead of runtime-quantizing.

## Processing longer clips (chunked sampler)

UniVidX is trained at 21 frames per inference. For source clips longer than ~1 second, use `examples/chunked_clip_sampler.py` — it slices your source into overlapping 21-frame windows, runs UniVidX on each, and stitches the per-modality outputs into 4 MP4s with a linear crossfade across the overlap.

```bash
# Fast iteration / preview: ~4.4 hours per 1 min @ 24 fps clip
python examples/chunked_clip_sampler.py \
    --input  C:/path/to/your_clip.mp4 \
    --mode   R2AIN \
    --output-dir  C:/path/to/output \
    --preset FP8_DISTILL_PREVIEW    # default in 0.5.0

# Production finals: ~14 hours per 1 min @ 24 fps clip
python examples/chunked_clip_sampler.py \
    --input  C:/path/to/your_clip.mp4 \
    --mode   R2AIN \
    --output-dir  C:/path/to/output \
    --preset FP8
```

Wall-time guide for a 1-minute @ 24 fps clip (1440 frames → 90 chunks):

| Preset | Per-chunk | Full clip | Quality |
|---|---|---|---|
| **FP8_DISTILL_PREVIEW** *(0.5.0 default)* | **4.59 min** | **~4.4 hours** | ~22-26 dB PSNR vs BF16; iteration / preview |
| FP8 | 9.43 min | ~14 hours | Production-quality finals (PSNR ≥ 30 dB) |
| PRODUCTION (legacy: BF16+sage) | 14.48 min | ~22 hours | Same quality as FP8 baseline, slower |

Caveat: each chunk samples from its own noise seed (same seed across chunks, but the trajectory diverges anyway from per-chunk numerical drift), so global identity drift between chunks is possible on lighting-varying content. The overlap crossfade hides per-pixel seams but not global drift. For clips with consistent lighting throughout the minute, drift is minor; for clips with cuts or lighting changes, expect visible breath in the per-modality channels at chunk boundaries.

The `FP8_DISTILL_PREVIEW` preset requires `lightx2v` LoRA at `models/loras/lightx2v/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors`. Download:

```bash
hf download lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v \
    loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors \
    --local-dir ComfyUI/models/loras/lightx2v
```

## What stopped helping on Blackwell

These knobs were useful in 0.2.0 / 0.3.0 but FP8 (0.4.0) is strictly better — leave them off unless you have a specific reason:

- **`prefer_sage_attn=True`** — measured +33% wall on BF16 baseline, +25% on FP8 baseline. SageAttention's INT8 quantized kernels apparently don't win on Wan2.1-14B's attention shapes at this resolution. Earlier docs (0.2.0) advertised "−18% wall" — that measurement was on a different config; not reproducible on R2AIN_video as of 0.4.0.
- **`compile_dit=True`** — measured +24% wall on FP8 baseline. `torch.compile` graph-captures cleanly on `FP8Linear` (good news, no crash) but the speedup it was designed to deliver doesn't materialize on top of FP8's already-resident state. Graph capture itself adds ~90 s overhead on first step.
- **`dtype=fp8_e4m3fn` / `fp8_e5m2`** — **DEPRECATED, removed in 0.5.0.** The legacy `mmgp.offload.quantize` path hangs during cold-load. Use `dit_weight_mode=fp8_prequantized` instead.
- **Flash Attention 3** — Hopper-only (H100/H800). Doesn't apply to RTX 5090.
- **Flash Attention 4** — Linux-only on PyPI; module name (`flash_attn.cute`) doesn't match DiffSynth's auto-detect.

## Other tuning knobs (Loader)

These have non-trivial effects and are worth understanding:

| Knob | Effect | Notes |
|---|---|---|
| `vram_buffer_gb` | GB kept free for activations; passed to `model.pipe.enable_vram_management()`. Controls layer-streaming aggressiveness on the BF16 path. | **+65% wall going 4.0 → 12.0** measured at BF16. Lower = more residency = faster; raise only if you hit OOM. 4.0 GB default is near-optimal on 32 GB cards. **Effectively no-op when `dit_weight_mode=fp8_prequantized`** because the FP8 DiT fits fully resident. |
| `dit_weight_mode` | `auto / bf16_shards / fp8_prequantized / fp8_runtime_experimental`. `auto` (default) preserves 0.3.x behaviour based on the legacy `dtype` widget. | See "The 0.4.0 recommendation" above. |

### SageAttention install (for `prefer_sage_attn=True`)

PyPI ships only sage 1.0.6 (head_dim restricted to {64,96,128}, Hopper/Ada-tuned only). For Blackwell + Windows + cp312 + Torch 2.7, use the prebuilt wheel from [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention/releases):

```bash
pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows/sageattention-2.2.0+cu128torch2.7.1-cp312-cp312-win_amd64.whl"
```

Match the wheel to your stack: `cu128`/`cu130` (CUDA), `torch2.7.1`/`2.8.0` (PyTorch), `cp310`/`cp311`/`cp312`/`cp313` (Python).

> **Cross-plugin gotcha — Stable3DGen SDPA pollution.** ComfyUI-3D-Pack's `Stable3DGen/trellis/backend_config.py` does `F.scaled_dot_product_attention = sageattn` *globally* at module import when sageattention is importable. That hostile swap breaks any other custom node using SDPA with head_dim outside sage's set (UniVidX's VAE has 1-head SDPA where head_dim = channel_count, hits 384). Our `runtime.load_model()` defensively restores `F.scaled_dot_product_attention` from `torch._C._nn.scaled_dot_product_attention` (the C++ impl, immune to Python alias rebinding). If other custom nodes broke after you installed sage, this is why.

### What does NOT help on Blackwell

- **Flash Attention 3** — Hopper-only (H100/H800). Doesn't apply to RTX 5090.
- **Flash Attention 4** — Linux-only on PyPI; module name (`flash_attn.cute`) doesn't match DiffSynth's auto-detect.

### FP8 status

We wired `dtype=fp8_e4m3fn` / `fp8_e5m2` via `mmgp.offload.quantize(model.pipe.dit, weights="qfloat8", exclude=["*lora_*"])`. **The quantize() pass hung in our cold-load test** (no completion after 22 min, required killing ComfyUI). Likely cause: quanto walks all ~720 Linear layers in Wan2.1-14B + UniVidX's PEFT-attached LoRA pairs, computing per-tensor scales over the 28 GB BF16 DiT through mmgp's read-only mmap — possibly genuinely slow, possibly genuinely deadlocked.

The widget is shipped but **flagged EXPERIMENTAL in tooltip + this README**. Use at your own risk on this stack today. See [Roadmap](#roadmap) for the planned proper fix.

## Node overview

Five nodes, all under the `UniVidX` category. Custom socket types — `UNIVIDX_MODEL` (purple), `UNIVIDX_TASK` (teal), `UNIVIDX_RESULT` (pink) — keep the graph type-safe; standard `IMAGE` (green) is used everywhere a frame batch flows.

<table>
<tr><td width="380">

![Loader](assets/nodes/loader.svg)

</td><td>

**`UniVidXLoader`** — Loads `intrinsic` or `alpha` variant, exposes the perf knobs (`compile_dit`, `prefer_sage_attn`, `dtype`, `vram_buffer_gb`, `dit_weight_mode`). Models are cached per `(variant, ckpt, device, dtype, vram_buffer, fp8_qtype, compile_dit, prefer_sage_attn, dit_weight_mode)` so toggling any of them triggers a clean re-load. `vram_buffer_gb` is in the key as of 0.3.0 because it now actually controls VRAM management (was a no-op in 0.1.0–0.2.1). `dit_weight_mode` is in the key as of 0.4.0 — picking `fp8_prequantized` drops DiT steady-state VRAM ~50% (with -13% wall as a bonus); see the perf table below.

</td></tr>
<tr><td>

![Task Mode](assets/nodes/task_mode.svg)

</td><td>

**`UniVidXTaskMode`** — Picks one of 30 modes from a dropdown. Outputs `UNIVIDX_TASK` carrying the mode + family. The sampler validates the family against the loaded model variant.

</td></tr>
<tr><td>

![Sampler](assets/nodes/sampler.svg)

</td><td>

**`UniVidXSampler`** — Runs UniVidX's `pipe()` end-to-end inside a `chdir(vendor/UniVidX)` context. Accepts the model + task + a text prompt + up to 7 optional `IMAGE` inputs (one per modality across both families). Inputs not required by the active mode are silently ignored.

</td></tr>
<tr><td>

![Decode Intrinsic](assets/nodes/decode_intrinsic.svg)

</td><td>

**`UniVidXDecodeIntrinsic`** — Splays an intrinsic-family `UNIVIDX_RESULT` into 4 `IMAGE` batches: `rgb / albedo / irradiance / normal`. Modalities that were *conditions* come back as a black placeholder of the right shape, so downstream graphs never break on missing slots.

</td></tr>
<tr><td>

![Decode Alpha](assets/nodes/decode_alpha.svg)

</td><td>

**`UniVidXDecodeAlpha`** — Same shape as above but for the alpha family: `composite_rgb / alpha / foreground / background`. Raises `ValueError` if you try to feed it an intrinsic-family result (and vice versa).

</td></tr>
</table>

## Models

| Pack | Where | Size |
|---|---|---|
| [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | `ComfyUI/models/wan21_t2v_14b/` | ~69 GB |
| [houyuanchen/UniVidX](https://huggingface.co/houyuanchen/UniVidX) | `ComfyUI/models/unividx/` | ~1.6 GB |

`install.py` verifies the vendored UniVidX submodule is at the pinned commit, copies bundled demo workflows into ComfyUI's user workflow directory, and prints a hint about the model files you still need to download. The actual **path bridging** — Windows directory junction (or POSIX symlink) from `vendor/UniVidX/models/` to `ComfyUI/models/wan21_t2v_14b/`, plus hardlinks for the two LoRA adapters — happens **at runtime** on first model load via `src/path_resolver.ensure_symlinks()` (called from `runtime.initialize()`). This lazy approach lets the install step stay quick and avoids touching the filesystem if the user never queues a UniVidX workflow.

## Mode reference

Mode names encode `<conditions>2<targets>`. `t` on the left = "text-only".

**Intrinsic** (variant `intrinsic`): R=RGB, A=Albedo, I=Irradiance, N=Normal — 15 modes total: `t2RAIN`, `R2AIN`, `A2RIN`, `I2RAN`, `N2RAI`, `RA2IN`, `RI2AN`, `RN2AI`, `AI2RN`, `AN2RI`, `IN2RA`, `RAI2N`, `RAN2I`, `RIN2A`, `AIN2R`.

**Alpha** (variant `alpha`): R=Composite RGB, P=Pha (matte), F=Fgr, B=Bgr — 15 modes total: `t2RPFB`, `R2PFB`, `P2RFB`, `F2RPB`, `B2RPF`, `RP2FB`, `RF2PB`, `RB2PF`, `PF2RB`, `PB2RF`, `FB2RP`, `RPF2B`, `RPB2F`, `RFB2P`, `PFB2R`.

For modes where a modality is a *condition*, the corresponding decoder output is a black tensor of the right shape — downstream nodes still get a valid `IMAGE`.

## Roadmap

Full v0.3 execution plan in [`ROADMAP_v0.3.md`](ROADMAP_v0.3.md). Summary of priority order (corrected after second-pass review):

1. ~~**`vram_buffer_gb` correctness fix**~~ — **shipped in 0.3.0.** Cache key now includes `vram_buffer` so distinct values get distinct cache entries (was the actual bug). The wiring itself (`model.pipe.enable_vram_management(...)`) was already correct in 0.2.1 — the roadmap's "wrong method target" diagnosis was disproven by a live runtime probe; see CHANGELOG 0.3.0 "Diagnosis correction." Wrapped in `if/else` with explicit INFO-on-success + WARNING-on-missing logging so a future regression can't recur silently. Tier-A5 bench measured **+65% wall going `vram_buffer` 4.0 → 12.0** — this is the biggest single perf lever in the system, not the "deprecated" knob it was labelled.
2. **FP8 via pre-quantized Kijai weights** (replaces the hung runtime-quantize path). The current `dtype=fp8_*` knob calls `mmgp.offload.quantize()` AFTER constructing a full BF16 DiT — which both hangs and destroys the main benefit (the BF16 cold load). Right design is a deeper refactor:
  1. **Split the public knob** into `compute_dtype = {bf16, fp16}` and `dit_weight_mode = {bf16_shards, fp8_prequantized, fp8_runtime_experimental}` so users get clear choices and runtime quantization is hidden behind an explicit experimental flag.
  2. **Add path resolution** for pre-quantized weights at `ComfyUI/models/diffusion_models/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors` (Kijai/ComfyUI convention).
  3. **Implement an alternate DiT loader** that bypasses upstream's hardcoded six-shard BF16 loop in [`vendor/UniVidX/src/pipelines/univid_intrinsic.py:447`](vendor/UniVidX/src/pipelines/univid_intrinsic.py) and `univid_alpha.py:425`. Instantiate `WanModel`, normalize key prefixes, stream-load FP8 safetensors, keep norms/bias/time/text/patch/head in BF16/FP32, and preserve scale tensors when present.
  4. **Keep UniVidX LoRA adapters in BF16.** PEFT's dynamic `adapter_names=[...]` switching is central to UniVidX's per-modality routing — a generic FP8 `nn.Linear` substitution would break it.
  5. **Phase 1 ("memory-safe FP8"):** FP8 base weights + BF16 LoRA + dequantize-or-scaled-linear base path. Correctness first.
  6. **Phase 2 ("fast FP8"):** adapter-aware `_scaled_mm` for the base projection plus BF16 LoRA residuals. Needs a custom adapter-aware Linear or a real PEFT integration — not just Kijai's no-LoRA fast path.
  7. **Validate** with the tiny R2AIN/R2PFB workflows first, then benchmark BF16 vs FP8 on cold-load time, peak VRAM, per-step time, and output sanity (per-modality SSIM/PSNR vs the BF16 reference).
- **Step-distill LoRA stacking** — try [LightX2V's `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors`](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v) on top of UniVidX's per-modality LoRA. Could close the small quality gap of the PREVIEW 4-step preset. Needs PEFT compatibility verification with UniVidX's `add_multiple_loras_to_model` machinery.

## Requirements (detailed)

The summary table at the top of this README covers the "can I run this?" question. The detail below is for users who want to know exactly why each requirement matters.

### Software

| Dependency | Version | Why |
|---|---|---|
| **Python** | 3.10+ (tested 3.12.9) | UniVidX uses Python 3.10's `str \| None` PEP-604 union syntax. The runtime explicitly fails fast on older Pythons. |
| **PyTorch** | ≥ 2.7 with CUDA 12.8 | Blackwell sm_120 (RTX 5090) requires `cu128`. Older PyTorch builds error with `no kernel image is available for execution on the device.` `torch.float8_e4m3fn` (used by the 0.4.0+ FP8 path) needs PyTorch 2.1+. |
| **ComfyUI** | 0.20+ | The node-registration API and the `INPUT_TYPES` schema we ship target this version's frontend. Older versions may render widgets incorrectly. |
| **DiffSynth-Studio** | ≥ 2.0 | UniVidX wraps DiffSynth's `WanVideoPipeline`. The pipeline-level VRAM management and tokenizer config use DiffSynth 2.x APIs. Auto-installed via `requirements.txt`. |
| **mmgp** | latest | Provides the read-only memory-mapped safetensors loader that prevents the Windows paging-file commit blowup when six 9.84 GB DiT shards are mapped concurrently. |
| **PEFT** | ≥ 0.10 | UniVidX uses `peft.inject_adapter_in_model` for the four per-modality LoRA adapters. The FP8 substitution and step-distill merge both descend through PEFT wrappers. |
| **safetensors** | ≥ 0.4 | All model weights ship as safetensors; the FP8 and step-distill loaders read them via `safetensors.safe_open`. |

### GPU

The binding constraint is **VRAM**. The Wan2.1-T2V-14B base model is ~28 GB in BF16 and ~14 GB in FP8. On top of that you need ~4-6 GB for activations / KV cache / VAE decode / text encoder during sampling.

| Path | DiT footprint | Total VRAM during sampling | Min card |
|---|---|---|---|
| BF16 baseline | ~28 GB | ~32-34 GB | 32 GB+ |
| **FP8 prequantized (0.5.0 default)** | **~14 GB** | **~18-20 GB** | **24 GB+** |
| FP8 + step-distill (fast preview) | ~14 GB | ~18-20 GB | 24 GB+ |

**CUDA compute capability**: 8.0 or higher (Ampere generation onward). UniVidX's attention path uses Flash-Attention-2's pattern which requires sm_80+. SageAttention 1.x (optional) supports head_dim ∈ {64, 96, 128} and is Hopper/Ada tuned; SageAttention 2.x from [woct0rdho's prebuilt wheels](https://github.com/woct0rdho/SageAttention/releases) covers Blackwell. As of 0.5.0, neither sage nor compile_dit help on the FP8 path — leave them off (see the full perf matrix above).

### System RAM

Peak host RAM during a cold-load:
- BF16 path: ~28 GB peak (loading the six DiT shards before VRAM management kicks in)
- FP8 path: ~28 GB peak (same cold-load; FP8 quantization runs after on the loaded BF16 weights)

64 GB system RAM gives comfortable headroom for ComfyUI + browser + other applications during the load. 32 GB is the practical minimum; the OS swaps to page file under stress on smaller systems.

### Disk

| Pack | Size | Required? |
|---|---|---|
| [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | ~69 GB | Yes — the base text-to-video DiT |
| [houyuanchen/UniVidX](https://huggingface.co/houyuanchen/UniVidX) | ~1.6 GB | Yes — the per-modality LoRA adapters (intrinsic + alpha) |
| [LightX2V step-distill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v) (rank-64 LoRA only) | ~600 MB | Optional — enables the 0.5.0 fast-preview mode |
| [Kijai `_scaled` Wan2.1 FP8](https://huggingface.co/Kijai/WanVideo_comfy) (if/when it lands) | ~14 GB | Optional — auto-engages the file-based FP8 loader instead of runtime quantize |

Plan for **85 GB** of disk including the optional packs + working space for output PNGs and MP4s (a 1-min clip at 480×640×24fps × 4 modalities is ~500 MB of intermediate PNGs).

### Operating system

| OS | Status |
|---|---|
| Windows 11 (validated) | ✅ All three Windows-specific patches (JSON path escaping, mmgp readonly mmap, junctions instead of symlinks) ship enabled |
| Windows 10 | 🟢 Should work, untested |
| Linux | 🟢 Should work; the Windows-specific patches no-op gracefully |
| macOS | 🔴 Not viable; no CUDA support for the Wan2.1 + FP8 path |

## Windows-specific patches

Three patches applied automatically on Windows; no-op on POSIX. Fully documented inline in `src/runtime.py` and `src/path_resolver.py`:

1. **JSON path escaping** — `json.dumps([t5, vae])` so Windows backslashes don't break UniVidX's `json.loads`.
2. **Read-only mmap for safetensors** — patch `mmgp.safetensors2.torch_load_file` to use `writable_tensors=False` (avoids `[WinError 1455] paging file too small` on six 9.84 GB DiT shards).
3. **Junctions + hardlinks instead of symlinks** — `mklink /J` + `os.link()` (no Admin / Developer Mode required).

## Troubleshooting

- **`MissingModelFile`** — re-run the `hf download` commands.
- **`R2AIN` rgb output is black** — correct, RGB was the input; decoder emits a black placeholder of the right shape.
- **Text-only alpha matte (`t2RPFB`) is white** — known model limit, not a bug. Use `R2PFB_video_api.json` instead.
- **Per-step time > 1 min on 32 GB+ GPU** — VRAM management didn't activate. Verify GPU temp <60°C with 99% util = memory-bound.
- **`CUDA error: no kernel image is available`** — torch too old for Blackwell; upgrade to `torch>=2.7+cu128`.
- **Custom nodes broke after installing sageattention** — see the Stable3DGen pollution note above. Our defensive un-pollute fixes UniVidX; other affected nodes need the same fix or `pip uninstall sageattention`.

For workflow-specific gotchas see [`examples/README.md`](examples/README.md) and [`examples/test_matrix/README.md`](examples/test_matrix/README.md).

## Out of scope (Strategy A boundary)

These would require porting UniVidX's Cross-Modal Self-Attention onto a different DiT class (multi-week project):

- Stacking community Wan2.1/2.2 LoRAs on UniVidX's DiT
- Injecting ControlNet / IP-Adapter inside UniVidX's denoising loop
- Replacing UniVidX's sampler with a ComfyUI KSampler
- Native `MODEL`-type integration (interop with kijai's WanVideoWrapper)

Strategy A's value is at the **I/O boundary** — composing UniVidX outputs with arbitrary downstream ComfyUI nodes. Validated end-to-end in `examples/test_matrix/` (10/10 passing).

## Credits

- [UniVidX](https://github.com/houyuanchen111/UniVidX) — vendored at a pinned commit
- [Wan-AI / Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) — base text-to-video DiT
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — pipeline runtime
- [mmgp](https://pypi.org/project/mmgp/) — paged memory loading
- [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention) — Blackwell sage 2.x wheels
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — host runtime

## License

[GPL-3.0](LICENSE). Vendored upstream deps keep their own licenses (UniVidX, Wan-AI/Wan2.1-T2V-14B).
