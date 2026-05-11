"""
Resolves Wan2.1-T2V-14B and UniVidX file paths from ComfyUI's models/ directory.

Returns absolute paths; raises MissingModelFile with a clear message if any
required file is absent so the user can fix the install before the model load
attempts and crashes deep inside DiffSynth.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union


class MissingModelFile(FileNotFoundError):
    """Raised when an expected Wan2.1 or UniVidX file is missing from ComfyUI/models."""


_REQUIRED_DIT_SHARDS = [
    f"diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)
]

_VARIANT_CKPT_NAMES = {
    "intrinsic": "univid_intrinsic.safetensors",
    "alpha": "univid_alpha.safetensors",
}


def resolve_paths(comfy_root: str,
                  variant: Optional[str] = None
                  ) -> Dict[str, Union[str, List[str]]]:
    """Locate Wan2.1-T2V-14B and UniVidX checkpoints under ``${comfy_root}/models/``.

    ``variant`` (``"intrinsic"`` / ``"alpha"`` / ``None``) selects which
    UniVidX checkpoint must be present. When ``None`` (back-compat
    default, used by ``ensure_symlinks`` without a specific variant)
    BOTH ``univid_intrinsic.safetensors`` and ``univid_alpha.safetensors``
    are required. When set, only the matching checkpoint is required;
    the other key in the returned dict is ``None``. This lets users
    download a single variant's UniVidX weights (~0.8 GB) without
    needing both.
    """
    if variant is not None and variant not in _VARIANT_CKPT_NAMES:
        raise ValueError(
            f"variant must be 'intrinsic', 'alpha', or None; got {variant!r}"
        )

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

    def _resolve_ckpt(key: str) -> Optional[str]:
        path = univ_dir / _VARIANT_CKPT_NAMES[key]
        if variant is None or variant == key:
            return need(path)
        return str(path.resolve()) if path.exists() else None

    univid_intrinsic_ckpt = _resolve_ckpt("intrinsic")
    univid_alpha_ckpt = _resolve_ckpt("alpha")
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


def _link_dir(src: Path, dst: Path) -> None:
    """Create a directory link at dst pointing to src. Uses junction on Windows, symlink on POSIX."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        # Windows junction — does not require admin or developer mode.
        # subprocess avoids needing _winapi imports and works on all Python 3.10+ builds.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise OSError(f"mklink /J failed: {result.stderr or result.stdout}")
    else:
        os.symlink(src, dst, target_is_directory=True)


def _link_file(src: Path, dst: Path) -> None:
    """Create a file link at dst pointing to src. Uses hardlink on Windows, symlink on POSIX."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        # Hardlink — works without admin if both paths are on the same volume.
        # Falls back to copy if cross-volume (common in CI).
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def ensure_symlinks(comfy_root: str, unividx_root: str,
                    variant: Optional[str] = None) -> None:
    """
    Create directory junction + hardlinks inside ${unividx_root}/{models,checkpoints}/
    that point to real files under ${comfy_root}/models/. UniVidX's pipeline code
    uses hardcoded relative paths assuming this layout.

    ``variant`` is forwarded to :func:`resolve_paths`. When set, the
    other variant's checkpoint may be absent — link only what's
    present. When ``None`` (back-compat), both variants must exist.
    """
    paths = resolve_paths(comfy_root, variant=variant)
    unividx = Path(unividx_root)

    wan_target = Path(paths["wan_t5"]).parent  # the wan21_t2v_14b dir
    wan_link = unividx / "models" / "Wan-AI" / "Wan2.1-T2V-14B"
    _link_dir(wan_target, wan_link)

    for key, name in [
        ("univid_intrinsic_ckpt", "univid_intrinsic.safetensors"),
        ("univid_alpha_ckpt", "univid_alpha.safetensors"),
    ]:
        ckpt = paths[key]
        if ckpt is None:
            continue
        link = unividx / "checkpoints" / name
        _link_file(Path(ckpt), link)
