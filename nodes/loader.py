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
                "dit_weight_mode": (
                    ["auto", "bf16_shards", "fp8_prequantized",
                     "fp8_runtime_experimental"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "How the DiT weights are stored after load. "
                            "`auto` (default): pick from the dtype widget — "
                            "bfloat16/float16 → bf16_shards, "
                            "fp8_e4m3fn/fp8_e5m2 → fp8_runtime_experimental "
                            "(preserves 0.3.x behaviour, DEPRECATED). "
                            "`bf16_shards`: standard BF16 path, ~28 GB DiT in "
                            "VRAM (with vram_buffer streaming layers as needed). "
                            "`fp8_prequantized`: cold-load BF16 shards as "
                            "usual, then runtime-quantize all DiT Linears to "
                            "FP8 e4m3fn with per-tensor absmax scaling. "
                            "Steady-state VRAM drops to ~14 GB (the FP8 "
                            "matmul still dequantizes to BF16 on forward in "
                            "Phase 1; Phase 2 will use torch._scaled_mm). "
                            "LoRA adapters stay BF16. PEFT-aware. "
                            "`fp8_runtime_experimental`: legacy "
                            "mmgp.offload.quantize() pass after BF16 cold "
                            "load — known to hang on this stack. The "
                            "dtype=fp8_* values route here for now; both will "
                            "be removed in 0.4.0."
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
             dit_weight_mode: str = "auto"):
        # Resolve the effective weight-load mode. `auto` collapses to the
        # value implied by the legacy `dtype` widget so old saved
        # workflows preserve their 0.3.x behaviour without action.
        legacy_fp8_qtype = {"fp8_e4m3fn": "qfloat8",
                            "fp8_e5m2": "qfloat8_e5m2"}.get(dtype)
        effective_mode = dit_weight_mode
        if effective_mode == "auto":
            effective_mode = ("fp8_runtime_experimental"
                              if legacy_fp8_qtype is not None
                              else "bf16_shards")

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
