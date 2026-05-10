# ComfyUI-UniVidX

ComfyUI custom nodes for UniVidX (SIGGRAPH 2026): unified video diffusion across
RGB / Albedo / Irradiance / Normal (intrinsic decomposition) and
Composite / Alpha / Foreground / Background (alpha decomposition).

This is a Strategy A wrapper: UniVidX's official pipeline is run as an opaque
black box. The four output IMAGE batches become standard ComfyUI tensors that
flow into any downstream node (3D reconstruction, ControlNet, compositing,
upscaling, etc.). UniVidX's interior (its forked Wan2.1 DiT, CMSA attention,
DGL LoRAs) is not exposed — combining UniVidX with community Wan LoRAs or
ControlNet on its denoising loop requires a different integration strategy
(see project notes).

## Requirements

- ComfyUI (recent build)
- Python 3.10+ (tested on 3.12.9)
- ≥24 GB VRAM (32 GB recommended for headroom — Wan2.1-T2V-14B is ~28 GB FP16)
- ~80 GB free disk
- CUDA 12.x (CUDA 12.8+ required for NVIDIA Blackwell / RTX 5090)

## Install

1. Clone into `${COMFY_ROOT}/custom_nodes/`:

   ```bash
   cd ${COMFY_ROOT}/custom_nodes
   git clone --recurse-submodules https://github.com/<your-org>/comfyui-unividx
   ```

2. Install Python deps:

   ```bash
   cd comfyui-unividx
   pip install -r requirements.txt
   pip install -r vendor/UniVidX/requirement.txt
   ```

   If on Blackwell (RTX 5090), ensure `torch>=2.7+cu128` is installed; older
   builds error at first kernel launch with
   `no kernel image is available for execution on the device.`

3. Download Wan2.1-T2V-14B (~69 GB):

   ```bash
   huggingface-cli download Wan-AI/Wan2.1-T2V-14B \
     --local-dir ${COMFY_ROOT}/models/wan21_t2v_14b
   ```

4. Download UniVidX checkpoints:

   ```bash
   huggingface-cli download houyuanchen/UniVidX \
     --local-dir ${COMFY_ROOT}/models/unividx
   ```

5. Restart ComfyUI.

### Windows-specific notes

This pack uses directory junctions (`mklink /J`) and hardlinks (`os.link`) to bridge
ComfyUI's `models/` paths to UniVidX's hardcoded layout. Neither requires
Administrator privileges or Developer Mode. If you see `WinError 1314: A required
privilege is not held by the client`, you've hit the symlink fallback — please
open an issue with your Windows version.

## Nodes

- **UniVidX • Load Model** — pick `intrinsic` or `alpha` variant; outputs `UNIVIDX_MODEL`.
  Takes a variant string (and optional precision/device flags); returns a loaded pipeline handle that downstream nodes consume.
- **UniVidX • Task Mode** — pick a mode (e.g. `t2RAIN`, `R2AIN`, `RA2IN`, `t2RPFB`); outputs `UNIVIDX_TASK`.
  Takes a mode code string; returns a task descriptor that tells `Sample` which modalities are conditions vs. targets.
- **UniVidX • Sample** — runs the pipeline; takes optional IMAGE inputs per modality; outputs `UNIVIDX_RESULT`.
  Inputs: `UNIVIDX_MODEL`, `UNIVIDX_TASK`, prompt text, seed, num_frames, height, width, and any conditioning IMAGE batches required by the mode. Output: an opaque `UNIVIDX_RESULT` tensor bundle.
- **UniVidX • Decode (Intrinsic)** — splay result into RGB / Albedo / Irradiance / Normal IMAGE batches.
  Input: `UNIVIDX_RESULT` from an intrinsic-variant model. Outputs: four IMAGE batches (one per modality), with non-target modalities filled as black placeholders.
- **UniVidX • Decode (Alpha)** — splay result into Composite / Alpha / Foreground / Background IMAGE batches.
  Input: `UNIVIDX_RESULT` from an alpha-variant model. Outputs: four IMAGE batches (one per modality), with non-target modalities filled as black placeholders.

## Mode reference

Letter codes (intrinsic): `R`=RGB, `A`=Albedo, `I`=Irradiance, `N`=Normal.
Letter codes (alpha): `R`=Composite RGB, `P`=alPha, `F`=Foreground, `B`=Background.

A mode like `RA2IN` means: RGB+Albedo are inputs (conditions), Irradiance+Normal are generated. `t2RAIN` means: text-only input, all four modalities generated.

For modes where a modality is a condition rather than a target, the corresponding decoder output is a black placeholder.

## Known limits

- Frame count is heavily preset to 21 in upstream training. Other frame counts may degrade quality.
- Default resolutions: 480×640 (intrinsic), 432×768 (alpha). Other sizes have not been validated.
- Prompts work better in Chinese — Wan2.1 was trained on a heavily Chinese-language corpus.
- No support for stacking community Wan LoRAs / ControlNet inside the UniVidX pipeline. That requires a deeper native integration (Strategy B/C).
- Outputs for non-target modalities are black placeholder IMAGE batches (e.g. for `R2AIN`, the decoder's `rgb` output is black because RGB was the input, not regenerated).

## Performance & VRAM management

Wan2.1-T2V-14B is **~28 GB FP16**. On a 32 GB GPU it would otherwise pin VRAM
at the ceiling and leave no headroom for activations / KV cache / VAE decode,
making each inference step memory-bound (GPU at 99 % util but only 50 °C —
classic memory-bound signature) at 2-3+ minutes per step.

`runtime.load_model()` calls `pipe.enable_vram_management(vram_buffer=4.0)`
after model construction. This wraps the DiT's `Linear` / `Conv3d` /
`LayerNorm` / `RMSNorm` modules with auto-offload wrappers — modules live on
CPU and stream to GPU only during their forward pass. With 16+ GB host RAM
this leaves comfortable working memory on GPU.

**Validated benchmarks** (RTX 5090, 32 GB, torch 2.7.0+cu128, t2RAIN mode):

| Resolution × frames × steps | Per-step time | Total wall time |
|---|---|---|
| 256×256 × 5 frames × 3 steps  | ~7.9 sec/step | 130 sec  (incl. cold load) |
| 480×640 × 21 frames × 20 steps | ~30.2 sec/step | 628 sec (incl. cold load) |

If you need to tune for your hardware:
- **Less VRAM**: lower `vram_buffer` floor (more aggressive offload, slower per
  step). Edit `src/runtime.py` and pass e.g. `vram_buffer=8.0`.
- **More headroom**: increase `vram_buffer` for headroom during high-res runs.
- **Pin specific param count**: pass `num_persistent_param_in_dit=N` to keep
  exactly N DiT params resident (e.g. for streaming setups).

## Windows-specific implementation notes

Three Windows-specific compatibility patches are baked into this pack. They
fire automatically on Windows; on POSIX systems they are no-ops.

1. **Backslash escaping in JSON paths** (`src/runtime.py`).
   UniVidX's `WanVideoPipeline` constructor takes a `model_paths` argument as
   a JSON string and runs `json.loads(model_paths)` internally. Windows paths
   like `D:\ComfyUI\models\...` contain `\D` and `\m` which are invalid JSON
   escapes. We construct the string with `json.dumps([t5, vae])` so backslashes
   are properly escaped. Without this, the loader fails with
   `json.decoder.JSONDecodeError: Invalid \escape`.

2. **Read-only mmap for safetensors** (`src/runtime.py`).
   The mmgp library (transitive dep of DiffSynth) monkey-patches
   `safetensors.torch.load_file` with a memory-mapped reader using
   `mmap.ACCESS_COPY`. Six 9.84 GB Wan2.1 DiT shards mmapped concurrently
   require ~60 GB of Windows paging-file commit, which exceeds most users'
   default and surfaces as `[WinError 1455] The paging file is too small`.
   We monkey-patch `load_file` inside `vendor/UniVidX/src/pipelines/...`
   namespaces to use `writable_tensors=False` (`ACCESS_READ`) instead — no
   commit charge needed since UniVidX only reads the tensors before copying
   to GPU.

3. **Junctions and hardlinks instead of symlinks** (`src/path_resolver.py`).
   `os.symlink()` on Windows requires Administrator privileges or Developer
   Mode. We use `mklink /J` (directory junction) for the Wan2.1 model dir
   link and `os.link()` (hardlink) for individual checkpoint files. Both
   work without privileges. If source and destination are on different
   volumes, hardlinks fall back to `shutil.copy2` (which costs ~1.5 GB extra
   disk for the UniVidX checkpoints if they're cross-volume).

## Troubleshooting

- **`MissingModelFile`** at startup: a Wan2.1 or UniVidX file is missing from `models/`. Re-run the download commands above.
- **OOM at sample time**: VRAM management is enabled by default with a 4 GB buffer. If you still OOM, raise the buffer (edit `src/runtime.py`'s `enable_vram_management(vram_buffer=...)`) or lower `num_frames` / `height` / `width`.
- **Black outputs**: pipe likely failed silently. Check ComfyUI's terminal for tracebacks.
- **Slow first run**: model load takes 5+ minutes (28 GB of weights). Subsequent runs reuse the cache.
- **Slow per-step time** (>1 min on a 32 GB+ GPU): VRAM management may not be activating. Verify `pipe.enable_vram_management` was called by checking the GPU temp during sampling — if it's <60 °C with 99 % util, you're memory-bound.
- **`ImportError: No module named diffsynth`**: `pip install diffsynth>=2.0` into the same Python that runs ComfyUI.
- **`WinError 1314`** (Windows symlink): see Windows-specific notes above. Should not occur — we use junctions/hardlinks instead.
- **`WinError 1455` The paging file is too small**: should not occur — the readonly mmap patch fixes this. If it does, verify `mmgp` is installed in the venv.
- **`json.decoder.JSONDecodeError: Invalid \escape`**: should not occur — the `json.dumps` patch fixes this. If it does, verify `runtime.py` is the patched version (commit `e0b17b7` or later).
- **`CUDA error: no kernel image is available for execution on the device.`**: torch is too old for your GPU. Upgrade to `torch>=2.7+cu128`.
