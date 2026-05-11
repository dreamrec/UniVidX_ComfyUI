# nodes/loader.py
"""
UniVidXLoader: load univid_intrinsic.safetensors or univid_alpha.safetensors.

Outputs UNIVIDX_MODEL — an opaque tuple (model_instance, variant_name) that
flows into the sampler.
"""
import logging

import torch

try:
    from ..src.runtime import load_model  # when imported as the UniVidX_ComfyUI package (ComfyUI runtime)
except ImportError:
    from src.runtime import load_model    # when imported flat (smoke test)

_log = logging.getLogger("unividx")


class UniVidXLoader:
    """Load the intrinsic or alpha UniVidX variant.

    Outputs the opaque ``UNIVIDX_MODEL`` tuple ``(model_instance, variant_name)``
    that flows into ``UniVidXSampler``. Models are cached per
    ``(variant, ckpt, device, dtype)`` in ``src.runtime`` so multi-graph
    sessions reuse weights instead of reloading the 28 GB DiT.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": (["intrinsic", "alpha"], {"default": "intrinsic"}),
                "dtype": (["bfloat16", "float16", "fp8_e4m3fn", "fp8_e5m2"], {
                    "default": "bfloat16",
                    "tooltip": (
                        "Compute dtype. bfloat16 (default) matches UniVidX's "
                        "training. float16 is functionally equivalent on most "
                        "Blackwell/Ada cards. fp8_e4m3fn / fp8_e5m2 post-"
                        "quantize the DiT via optimum-quanto's qfloat8 path: "
                        "halves DiT memory (~28 GB → ~14 GB), enables Marlin "
                        "FP8 matmul on Hopper/Blackwell. EXPERIMENTAL — the "
                        "quantize() pass over Wan2.1-14B + UniVidX's LoRA "
                        "stack is slow (10+ min) and may hang on some "
                        "configurations; if your run stalls on cold-load, "
                        "fall back to bfloat16."
                    ),
                }),
            },
            # Optional so that old saved workflows still validate.
            "optional": {
                "compile_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Run torch.compile(dit, mode='reduce-overhead', "
                        "dynamic=True) after model load. First sampler step "
                        "is 60-120 sec slower (graph capture), subsequent "
                        "steps are typically 20-30% faster. Best for "
                        "long runs at fixed resolution; loses its compile "
                        "cache when you change resolution/frame-count. "
                        "Cached separately per (variant, dtype, compile) "
                        "tuple so toggling triggers a re-load + re-compile."
                    ),
                }),
                "prefer_sage_attn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "If True (and `sageattention` is installed in the "
                        "ComfyUI venv), monkey-patch DiffSynth's Wan DiT "
                        "attention chain so SageAttention wins over Flash "
                        "Attention 2. SageAttention's INT8-quantized "
                        "attention is typically faster on Hopper/Blackwell "
                        "(claimed 2-5x by upstream, our measurements vary). "
                        "On Blackwell (sm_120) FA3 isn't available — FA2 "
                        "is the default winner — so this flag is the "
                        "main attention-backend lever. No-op if "
                        "sageattention isn't installed."
                    ),
                }),
                "vram_buffer_gb": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 96.0, "step": 0.5,
                    "tooltip": (
                        "GB of VRAM kept free for activations / KV cache / "
                        "VAE decode. Passed to UniVidX's pipeline-level "
                        "enable_vram_management(), which wraps text encoder "
                        "+ DiT + VAE through DiffSynth's offload helper so "
                        "layers live on CPU and stream to GPU on demand. "
                        "Higher = more headroom but more streaming (slower). "
                        "Lower = more residency. 4.0 GB is a sane default "
                        "for BF16 Wan2.1-14B + UniVidX LoRAs on 32 GB cards. "
                        "Cached per (variant, dtype, vram_buffer, ...) so "
                        "two loader nodes with different values get distinct "
                        "model instances."
                    ),
                }),
                "step_distill_lora": (
                    ["none", "lightx2v"],
                    {
                        "default": "none",
                        "tooltip": (
                            "Merge a step-distillation LoRA into the DiT "
                            "base weights at load time, enabling near-"
                            "production-quality decompositions at 4-6 "
                            "sample steps + cfg_scale=1. Cuts wall-time "
                            "per chunk by ~3-5x. Currently supports "
                            "'lightx2v' (Wan2.1-T2V-14B-StepDistill-"
                            "CfgDistill-Lightx2v, rank-64). EXPERIMENTAL: "
                            "step-distill quality on UniVidX's per-"
                            "modality decompositions (Albedo / Irradiance "
                            "/ Normal / Alpha) is unverified - LightX2V "
                            "was trained on natural-image content, not on "
                            "synthetic decomposition targets. RGB-style "
                            "outputs are most likely to retain quality; "
                            "Normal and Alpha mattes are highest risk. "
                            "Pairs with `step_distill_strength`. Requires "
                            "the file under models/loras/lightx2v/ (see "
                            "FileNotFoundError message for download cmd)."
                        ),
                    },
                ),
                "step_distill_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": (
                        "Effective merge strength for the step-distill "
                        "LoRA. 1.0 = standard merge (recommended for the "
                        "distillation effect); 0.0 = no merge (equivalent "
                        "to step_distill_lora='none'); >1.0 = overdrive "
                        "(may produce artifacts but worth testing if "
                        "1.0 gives weak step-reduction effect on your "
                        "content). Cached as part of the model key, so "
                        "changing this triggers a full reload."
                    ),
                }),
                "dit_weight_mode": (
                    ["fp8_prequantized", "bf16_shards", "auto",
                     "fp8_runtime_experimental"],
                    {
                        "default": "fp8_prequantized",
                        "tooltip": (
                            "How the DiT weights are stored after load. "
                            "0.5.0 default: `fp8_prequantized`. "
                            "\n\n"
                            "`fp8_prequantized` (recommended): the DiT's "
                            "~400 Linears + biases + norms are converted "
                            "to FP8 e4m3fn after the standard BF16 "
                            "cold-load. Two implementation paths share "
                            "this label — (a) FILE-BASED if a Kijai "
                            "`Wan2_1-T2V-14B_fp8_e4m3fn_scaled.safetensors` "
                            "is present under models/diffusion_models/ "
                            "(no such file exists upstream for Wan2.1 "
                            "as of 0.5.0; reserved forward-compat slot), "
                            "(b) RUNTIME-QUANTIZE otherwise (the actual "
                            "0.5.0 path on every Wan2.1 install today). "
                            "Quality: PSNR ≥ 30 dB per modality vs BF16 "
                            "reference. Wall: 9.43 min on R2AIN_video, "
                            "13% faster than bf16_shards, ~14 GB DiT VRAM. "
                            "\n\n"
                            "`bf16_shards`: standard BF16 path — ~28 GB "
                            "DiT in VRAM (with vram_buffer streaming "
                            "layers as needed). The 0.3.x baseline. "
                            "Pixel-for-pixel identical to UniVidX's "
                            "vanilla output but slower. "
                            "\n\n"
                            "`auto` (legacy): preserved for back-compat "
                            "with old saved workflows. In 0.5.0+ this "
                            "resolves to `fp8_prequantized` (was "
                            "`bf16_shards` in 0.4.0). "
                            "\n\n"
                            "`fp8_runtime_experimental`: legacy "
                            "mmgp.offload.quantize() pass after BF16 cold "
                            "load — known to hang on this stack. Kept "
                            "only as an escape hatch for users who need "
                            "to replicate pre-0.4.0 quirks; will be "
                            "removed in 0.6.0."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("UNIVIDX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "UniVidX"

    def load(self, variant: str, dtype: str,
             compile_dit: bool = False, prefer_sage_attn: bool = False,
             vram_buffer_gb: float = 4.0,
             dit_weight_mode: str = "fp8_prequantized",
             step_distill_lora: str = "none",
             step_distill_strength: float = 1.0):
        # Legacy dtype=fp8_e4m3fn / fp8_e5m2: the README has documented
        # these as DEPRECATED since 0.4.0 and "removed in 0.5.0" — but
        # the enum values are still in the dtype widget for backward-
        # compat with old saved workflows. Per the 2026-05-11 external
        # audit, the current behavior of silently routing to the
        # known-hanging `fp8_runtime_experimental` path is the worst
        # of both worlds. Migrate them to the supported fp8_prequantized
        # path with a deprecation warning so the user lands somewhere
        # working, and surface the migration so they know to update
        # their saved workflow.
        if dtype in ("fp8_e4m3fn", "fp8_e5m2"):
            _log.warning(
                "dtype=%s is DEPRECATED (since 0.4.0). The legacy "
                "mmgp.offload.quantize() path was removed; auto-"
                "migrating this load to dtype=bfloat16 + "
                "dit_weight_mode=fp8_prequantized (the supported FP8 "
                "path). Update your workflow's loader widgets to "
                "silence this warning: set dtype=bfloat16 and "
                "dit_weight_mode=fp8_prequantized. The dtype=fp8_* "
                "values will be removed entirely in 0.6.0.",
                dtype,
            )
            dtype = "bfloat16"
            dit_weight_mode = "fp8_prequantized"

        effective_mode = dit_weight_mode
        if effective_mode == "auto":
            # `auto` is retained for back-compat with old saved
            # workflows; it now resolves to fp8_prequantized (the
            # 0.5.0-recommended default), not bf16_shards as in 0.4.0.
            # Set dit_weight_mode='bf16_shards' explicitly if you want
            # the 0.3.x baseline behavior.
            effective_mode = "fp8_prequantized"

        # Compute dtype is BF16 for every FP8-related mode (UniVidX
        # constructs the pipeline at BF16 regardless; quantization /
        # FP8 dequant happen on top). Only the BF16/FP16 shard path
        # honours dtype=float16.
        if effective_mode == "bf16_shards" and dtype == "float16":
            compute_dtype = torch.float16
        else:
            compute_dtype = torch.bfloat16

        # Wire the runtime kwargs by mode.
        runtime_kwargs = dict(
            device="cuda",
            dtype=compute_dtype,
            vram_buffer=float(vram_buffer_gb),
            compile_dit=bool(compile_dit),
            prefer_sage_attn=bool(prefer_sage_attn),
            dit_weight_mode=effective_mode,
            quantize_fp8=None,
            step_distill_lora=str(step_distill_lora),
            step_distill_strength=float(step_distill_strength),
        )

        if effective_mode == "fp8_runtime_experimental":
            # Legacy mmgp.offload.quantize path. Known to hang in cold
            # load on Wan2.1-14B + UniVidX LoRA stack (see CHANGELOG).
            # Keep it functional through 0.3.x with a deprecation
            # warning; will be removed in 0.4.0 in favour of the
            # pre-quantized path.
            qtype = legacy_fp8_qtype or "qfloat8"  # safe default for explicit pick
            runtime_kwargs["quantize_fp8"] = qtype
            if legacy_fp8_qtype is not None:
                _log.warning(
                    "dtype=%s is DEPRECATED — routes through "
                    "fp8_runtime_experimental (mmgp.offload.quantize "
                    "post-pass), which is known to hang on this stack. "
                    "Migrate to dit_weight_mode='fp8_prequantized' "
                    "(Tier B, lands in 0.3.1+) when it's available; both "
                    "fp8_e4m3fn and fp8_e5m2 dtype values will be removed "
                    "in 0.4.0.",
                    dtype,
                )

        model = load_model(variant, **runtime_kwargs)
        return ((model, variant),)
