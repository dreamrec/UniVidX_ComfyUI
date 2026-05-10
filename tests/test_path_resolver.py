import os
import tempfile
from pathlib import Path

import pytest

from src.path_resolver import resolve_paths, MissingModelFile


def _touch(p: Path, size: int = 0):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        if size:
            f.write(b"\0" * size)
        else:
            f.write(b"")


def test_resolve_paths_returns_all_expected_keys(tmp_path):
    comfy_root = tmp_path / "ComfyUI"
    wan_dir = comfy_root / "models" / "wan21_t2v_14b"
    univ_dir = comfy_root / "models" / "unividx"

    # Create stub files
    for i in range(1, 7):
        _touch(wan_dir / f"diffusion_pytorch_model-0000{i}-of-00006.safetensors")
    _touch(wan_dir / "diffusion_pytorch_model.safetensors.index.json")
    _touch(wan_dir / "models_t5_umt5-xxl-enc-bf16.pth")
    _touch(wan_dir / "Wan2.1_VAE.pth")
    _touch(wan_dir / "google" / "umt5-xxl" / "spiece.model")
    _touch(univ_dir / "univid_intrinsic.safetensors")
    _touch(univ_dir / "univid_alpha.safetensors")

    paths = resolve_paths(str(comfy_root))

    assert set(paths.keys()) >= {
        "wan_dit_shards", "wan_t5", "wan_vae", "wan_tokenizer_dir",
        "univid_intrinsic_ckpt", "univid_alpha_ckpt",
    }
    assert len(paths["wan_dit_shards"]) == 6
    assert paths["wan_t5"].endswith("models_t5_umt5-xxl-enc-bf16.pth")
    assert paths["wan_vae"].endswith("Wan2.1_VAE.pth")


def test_resolve_paths_raises_when_t5_missing(tmp_path):
    comfy_root = tmp_path / "ComfyUI"
    wan_dir = comfy_root / "models" / "wan21_t2v_14b"
    # Intentionally do NOT create T5
    for i in range(1, 7):
        _touch(wan_dir / f"diffusion_pytorch_model-0000{i}-of-00006.safetensors")
    _touch(wan_dir / "Wan2.1_VAE.pth")

    with pytest.raises(MissingModelFile, match="models_t5_umt5-xxl-enc-bf16.pth"):
        resolve_paths(str(comfy_root))


def test_ensure_symlinks_creates_expected_links(tmp_path):
    from src.path_resolver import ensure_symlinks

    comfy_root = tmp_path / "ComfyUI"
    wan_dir = comfy_root / "models" / "wan21_t2v_14b"
    univ_dir = comfy_root / "models" / "unividx"
    for i in range(1, 7):
        _touch(wan_dir / f"diffusion_pytorch_model-0000{i}-of-00006.safetensors")
    _touch(wan_dir / "diffusion_pytorch_model.safetensors.index.json")
    _touch(wan_dir / "models_t5_umt5-xxl-enc-bf16.pth")
    _touch(wan_dir / "Wan2.1_VAE.pth")
    _touch(wan_dir / "google" / "umt5-xxl" / "spiece.model")
    _touch(univ_dir / "univid_intrinsic.safetensors")
    _touch(univ_dir / "univid_alpha.safetensors")

    unividx_root = tmp_path / "vendor" / "UniVidX"
    unividx_root.mkdir(parents=True)

    ensure_symlinks(str(comfy_root), str(unividx_root))

    # Wan dir was linked (junction on Windows, symlink on POSIX) — either way it should exist as a dir
    assert (unividx_root / "models" / "Wan-AI" / "Wan2.1-T2V-14B").is_dir()
    # Checkpoint files exist (hardlink on Windows, symlink on POSIX)
    assert (unividx_root / "checkpoints" / "univid_intrinsic.safetensors").exists()
    assert (unividx_root / "checkpoints" / "univid_alpha.safetensors").exists()
