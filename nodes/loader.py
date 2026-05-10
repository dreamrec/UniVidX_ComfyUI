# nodes/loader.py
"""
UniVidXLoader: load univid_intrinsic.safetensors or univid_alpha.safetensors.

Outputs UNIVIDX_MODEL — an opaque tuple (model_instance, variant_name) that
flows into the sampler.
"""
import torch

try:
    from ..src.runtime import load_model  # when imported as the UniVidX_ComfyUI package (ComfyUI runtime)
except ImportError:
    from src.runtime import load_model    # when imported flat (smoke test)


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
                        "DEPRECATED on current diffsynth — the underlying "
                        "WanVideoPipeline.enable_vram_management API was "
                        "removed. The runtime call is a no-op on this build. "
                        "Left here for backwards-compat with saved workflows; "
                        "value has no effect on speed. (Will be re-wired if a "
                        "future diffsynth version reintroduces a comparable "
                        "memory-management entrypoint.)"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("UNIVIDX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "UniVidX"

    def load(self, variant: str, dtype: str,
             compile_dit: bool = False, prefer_sage_attn: bool = False,
             vram_buffer_gb: float = 4.0):
        # FP8 path: load as bfloat16 (UniVidX construction is hardcoded BF16),
        # then post-quantize the DiT via mmgp/optimum-quanto.
        # `fp8_variant` is the optimum-quanto qtype name (or None for non-FP8
        # dtypes); load_model uses None as the "skip quantization" sentinel.
        fp8_variant = {"fp8_e4m3fn": "qfloat8", "fp8_e5m2": "qfloat8_e5m2"}.get(dtype)
        compute_dtype = torch.bfloat16 if fp8_variant is not None \
            else {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        model = load_model(variant, device="cuda", dtype=compute_dtype,
                           vram_buffer=float(vram_buffer_gb),
                           quantize_fp8=fp8_variant,
                           compile_dit=bool(compile_dit),
                           prefer_sage_attn=bool(prefer_sage_attn))
        return ((model, variant),)
