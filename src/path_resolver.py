"""
Resolves Wan2.1-T2V-14B and UniVidX file paths from ComfyUI's models/ directory.

Returns absolute paths; raises MissingModelFile with a clear message if any
required file is absent so the user can fix the install before the model load
attempts and crashes deep inside DiffSynth.
"""
from pathlib import Path
from typing import Dict, List, Union


class MissingModelFile(FileNotFoundError):
    """Raised when an expected Wan2.1 or UniVidX file is missing from ComfyUI/models."""


_REQUIRED_DIT_SHARDS = [
    f"diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)
]


def resolve_paths(comfy_root: str) -> Dict[str, Union[str, List[str]]]:
    """Locate Wan2.1-T2V-14B and UniVidX checkpoints under ${comfy_root}/models/."""
    root = Path(comfy_root)
    wan_dir = root / "models" / "wan21_t2v_14b"
    univ_dir = root / "models" / "unividx"

    def need(p: Path) -> str:
        if not p.exists():
            raise MissingModelFile(
                f"Expected {p} but it doesn't exist. "
                f"See README install instructions."
            )
        return str(p.resolve())

    shards = [need(wan_dir / name) for name in _REQUIRED_DIT_SHARDS]

    # Order: check core required files first (T5, VAE, tokenizer, UniVidX ckpts)
    # before the index file, so missing-T5 errors surface clearly.
    wan_t5 = need(wan_dir / "models_t5_umt5-xxl-enc-bf16.pth")
    wan_vae = need(wan_dir / "Wan2.1_VAE.pth")
    wan_tokenizer_dir = need(wan_dir / "google" / "umt5-xxl")
    univid_intrinsic_ckpt = need(univ_dir / "univid_intrinsic.safetensors")
    univid_alpha_ckpt = need(univ_dir / "univid_alpha.safetensors")
    wan_dit_index = need(wan_dir / "diffusion_pytorch_model.safetensors.index.json")

    return {
        "wan_dit_shards": shards,
        "wan_dit_index": wan_dit_index,
        "wan_t5": wan_t5,
        "wan_vae": wan_vae,
        "wan_tokenizer_dir": wan_tokenizer_dir,
        "univid_intrinsic_ckpt": univid_intrinsic_ckpt,
        "univid_alpha_ckpt": univid_alpha_ckpt,
    }
