# UniVidX Intrinsic & Alpha Decomposition for ComfyUI

![License](https://img.shields.io/badge/license-GPL--3.0-2f855a)
![Python](https://img.shields.io/badge/python-3.10%2B-1f4b99)
![PyTorch](https://img.shields.io/badge/torch-%E2%89%A52.7%2Bcu128-ee4c2c)
![Nodes](https://img.shields.io/badge/nodes-5-f59e0b)
![Tests](https://img.shields.io/badge/tests-24%20unit%20%2B%2010%20integration-success)

ComfyUI custom nodes for [UniVidX](https://houyuanchen111.github.io/UniVidX.github.io/) (SIGGRAPH 2026): unified video diffusion that decomposes a clip into **RGB / Albedo / Irradiance / Normal** (intrinsic) or **Composite RGB / Alpha matte / Foreground / Background** (alpha) — 30 task modes across two model variants, all driven from a single five-node graph.

This is a **Strategy A** wrapper: UniVidX's official pipeline runs as an opaque black box; the four output IMAGE batches become standard ComfyUI tensors that flow into any downstream node — VHS video combine, alpha compositing, 3D reconstruction, ControlNet conditioning for *other* models, you name it.

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

### `examples/t2RAIN_basic.json` — Text → All Four Intrinsic Modalities

The flagship demo. Generate a 21-frame 480×640 video with full RGB / Albedo / Irradiance / Normal decomposition from a text prompt alone.

```text
LoadModel(intrinsic) ──┐
                       ├─→ Sample(t2RAIN) ─→ Decode (Intrinsic) ─→ 4× SaveImage
TaskMode(t2RAIN) ──────┘
```

Drag the JSON onto canvas, queue, get four PNG sequences. ~10 min wall time on a 5090.

### `examples/R2AIN_basic_api.json` — RGB-Conditioned Re-Decomposition

Provide an existing RGB video; UniVidX produces matched Albedo / Irradiance / Normal channels. The decoder's `rgb` slot becomes a black placeholder (RGB was the input, not regenerated).

```text
LoadImage → RepeatImageBatch(21) ─→ Sample(R2AIN, rgb=...) ─→ Decode → 4× SaveImage
```

### `examples/test_matrix/E_t2RPFB.json` — Alpha Decomposition (Text-to-All)

Same idea but for the **alpha** family — pulls clean foreground / background pairs and an alpha matte. Note: alpha decomposition works much better when an RGB reference is provided (see R2PFB below).

### `examples/test_matrix/F_R2PFB.json` — Sharp Alpha Matte from RGB

The most useful alpha workflow: feed an RGB clip, get a clean alpha matte + isolated foreground + clean background. The matte is a real production-grade mask, not just a visualization.

### `examples/test_matrix/J_alpha_compositing.json` — End-to-End VFX Composite

Demonstrates that the alpha matte is a usable mask: extract the foreground via `ImageToMask` + `ImageCompositeMasked` and paste it onto any new background.

```text
Sample(R2PFB) → Decode → ImageToMask(alpha) ──┐
                       └─→ foreground ────────┤
                                              ├─→ ImageCompositeMasked → SaveImage
                                EmptyImage ───┘
```

### `examples/test_matrix/I_video_output.json` — Direct MP4 Export

Skip the per-frame PNGs and emit one MP4 per modality via `VHS_VideoCombine`.

```text
Sample(t2RAIN) → Decode → 4× VHS_VideoCombine
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

Wan2.1-T2V-14B is **~28 GB FP16**. On a 32 GB GPU it would otherwise pin VRAM at the ceiling and leave no headroom for activations / KV cache / VAE decode, making each inference step memory-bound (GPU at 99% util but only ~50°C — classic memory-bound signature) at 2-3+ minutes per step.

`runtime.load_model()` calls `pipe.enable_vram_management(vram_buffer=4.0)` after model construction. This wraps the DiT's `Linear` / `Conv3d` / `LayerNorm` / `RMSNorm` modules with auto-offload wrappers — modules live on CPU and stream to GPU only during their forward pass. With 16+ GB host RAM this leaves comfortable working memory on GPU.

**Validated benchmarks** (RTX 5090, 32 GB, torch 2.7.0+cu128, intrinsic variant, t2RAIN mode):

| Resolution × frames × steps | Per-step time | Total wall time |
|---|---|---|
| 256×256 × 5 frames × 3 steps | ~3.8 sec/step | 130 sec (incl. cold load) |
| 480×640 × 21 frames × 20 steps | ~30.2 sec/step | 628 sec (incl. cold load) |

Cache hits on subsequent same-variant runs skip the ~3 min cold load. Switching variants (intrinsic ↔ alpha) forces a reload.

### Tuning for your hardware

- **Less VRAM**: lower the `vram_buffer` floor in `src/runtime.py` to e.g. `vram_buffer=8.0`. More aggressive offload, slower per step but leaves more headroom for higher-res runs.
- **More headroom desired**: pass a smaller buffer.
- **Pin a specific param count resident**: pass `num_persistent_param_in_dit=N` instead of `vram_buffer`.

### Resolution & frame-count notes

- **Defaults**: `480×640` for intrinsic, `432×768` for alpha — these match upstream training. Other sizes work but quality may drift.
- **Frame count is heavily preset to 21** in upstream training. The sampler accepts `num_frames` from 5 to 81 in steps of 4, but only ~21±4 is well validated. Lower counts work for fast smoke tests; higher counts may degrade.
- **Tiled VAE** (`tiled=True`) is enabled by default and recommended for everything ≥480p. Tile size hardcoded to `[30, 52]` with stride `[15, 26]` (upstream defaults).

### Prompt language

Wan2.1's text encoder was trained heavily on Chinese. **English prompts work but are noticeably weaker.** The bundled negative prompt in the example workflows is the upstream Chinese standard — keep it.

## Node Overview

Five nodes, all under the `UniVidX` category:

| Node | What it does |
|------|------|
| **UniVidX • Load Model** | Loads `intrinsic` or `alpha` variant; outputs `UNIVIDX_MODEL`. Cached per `(variant, dtype, device)` so multi-graph runs reuse weights. |
| **UniVidX • Task Mode** | Picks one of the 30 modes from a dropdown; outputs `UNIVIDX_TASK` and validates against the loaded model's family. |
| **UniVidX • Sample** | Runs UniVidX's `pipe()`. Accepts up to 7 optional IMAGE inputs (one per modality across both families); ignores the ones not required by the active mode. |
| **UniVidX • Decode (Intrinsic)** | Splays the result into RGB / Albedo / Irradiance / Normal IMAGE batches. Black-fills any modality that was a condition rather than a target. |
| **UniVidX • Decode (Alpha)** | Same but for Composite / Alpha / Foreground / Background. |

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
   `os.symlink()` on Windows requires Administrator privileges or Developer Mode. We use `mklink /J` (directory junction) for the Wan2.1 model dir link and `os.link()` (hardlink) for individual checkpoint files. Both work without privileges. Cross-volume hardlinks fall back to `shutil.copy2` (~1.5 GB extra disk if your `models/` and the comfyui-unividx repo are on different volumes).

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
