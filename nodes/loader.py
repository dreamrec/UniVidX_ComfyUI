# nodes/loader.py
"""
UniVidXLoader: load univid_intrinsic.safetensors or univid_alpha.safetensors.

Outputs UNIVIDX_MODEL — an opaque tuple (model_instance, variant_name) that
flows into the sampler.
"""
import torch

try:
    from ..src.runtime import load_model  # when imported as part of comfyui-unividx package (ComfyUI)
except ImportError:
    from src.runtime import load_model    # when imported flat (smoke test)


class UniVidXLoader:
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
