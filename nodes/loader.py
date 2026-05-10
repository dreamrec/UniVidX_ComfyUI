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
                        "quantize the DiT via optimum-quanto: ~14 GB instead "
                        "of ~28 GB, fits fully on a 32 GB GPU (no streaming), "
                        "Blackwell native FP8 matmul. e4m3fn has larger "
                        "mantissa (better precision); e5m2 has larger "
                        "exponent (better dynamic range). Experimental — "
                        "LoRA layers are excluded from quantization to "
                        "preserve adapter precision."
                    ),
                }),
            },
            # Optional so that old saved workflows lacking this widget
            # still validate (default 4.0 is applied transparently).
            "optional": {
                "vram_buffer_gb": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 96.0, "step": 0.5,
                    "tooltip": (
                        "GB of GPU VRAM to keep free for activations. "
                        "Lower = more model resident on GPU = faster per-step. "
                        "Range 0-96 covers 24 GB GPUs (RTX 4090/5090 → 4.0 default) "
                        "through 32 GB Blackwell consumer (5090 → 12.0) up to "
                        "96 GB workstation cards (RTX 6000 Pro Blackwell, "
                        "RTX 5000 Ada → 4-8 if dedicated, more if you're "
                        "running other models alongside). Models are cached "
                        "separately per buffer value, so changing this "
                        "triggers a one-time reload."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("UNIVIDX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "UniVidX"

    def load(self, variant: str, dtype: str, vram_buffer_gb: float = 4.0):
        # FP8 path: load as bfloat16 (UniVidX construction is hardcoded BF16),
        # then post-quantize the DiT via mmgp/optimum-quanto.
        fp8_variant = {"fp8_e4m3fn": "qfloat8", "fp8_e5m2": "qfloat8_e5m2"}.get(dtype)
        quantize_fp8 = fp8_variant is not None
        compute_dtype = torch.bfloat16 if quantize_fp8 \
            else {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        model = load_model(variant, device="cuda", dtype=compute_dtype,
                           vram_buffer=float(vram_buffer_gb),
                           quantize_fp8=fp8_variant)
        return ((model, variant),)
