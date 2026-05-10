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
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("UNIVIDX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "UniVidX"

    def load(self, variant: str, dtype: str):
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        model = load_model(variant, device="cuda", dtype=torch_dtype)
        return ((model, variant),)
