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

## Troubleshooting

- **`MissingModelFile`** at startup: a Wan2.1 or UniVidX file is missing from `models/`. Re-run the download commands above.
- **OOM at sample time**: lower `num_frames`, `height`, or `width`; UniVidX has DiffSynth-Studio's VRAM management built in but 14B is tight on 24 GB.
- **Black outputs**: pipe likely failed silently. Check ComfyUI's terminal for tracebacks.
- **Slow first run**: model load takes 5+ minutes (28 GB of weights). Subsequent runs reuse the cache.
- **`ImportError: No module named diffsynth`**: `pip install diffsynth>=2.0` into the same Python that runs ComfyUI.
- **`WinError 1314`** (Windows symlink): see Windows-specific notes above.
- **`CUDA error: no kernel image is available for execution on the device.`**: torch is too old for your GPU. Upgrade to `torch>=2.7+cu128`.
