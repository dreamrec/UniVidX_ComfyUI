# src/runtime.py
"""
Bridges ComfyUI custom-node runtime to UniVidX's pipelines.

Responsibilities:
- Add vendor/UniVidX to sys.path on first import.
- Resolve all model paths and create symlinks under vendor/UniVidX/.
- Provide a context manager that temporarily chdirs into vendor/UniVidX/
  so UniVidX's hardcoded relative paths resolve.
- Cache loaded model instances per (variant, checkpoint_path) tuple so
  multi-node graphs don't reload weights.
"""
import contextlib
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import torch

from .path_resolver import ensure_symlinks, resolve_paths


_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _THIS_DIR.parent  # custom_nodes/comfyui-unividx/
_UNIVIDX_ROOT = _PLUGIN_ROOT / "vendor" / "UniVidX"

_LOAD_LOCK = threading.Lock()
_MODEL_CACHE: dict = {}


def _comfy_root() -> str:
    """Walk upward from this file until we find ComfyUI's root (contains models/ dir)."""
    p = _PLUGIN_ROOT
    for _ in range(5):
        if (p / "models").is_dir():
            return str(p)
        p = p.parent
    raise RuntimeError(
        f"Could not locate ComfyUI root from {_PLUGIN_ROOT}. "
        f"Expected to find a 'models/' directory in an ancestor."
    )


def initialize() -> None:
    """One-time setup: add UniVidX to sys.path, create symlinks. Idempotent."""
    if str(_UNIVIDX_ROOT) not in sys.path:
        sys.path.insert(0, str(_UNIVIDX_ROOT))
    ensure_symlinks(_comfy_root(), str(_UNIVIDX_ROOT))


@contextlib.contextmanager
def unividx_cwd():
    """Temporarily chdir into vendor/UniVidX/ so its relative paths resolve."""
    prev = os.getcwd()
    os.chdir(_UNIVIDX_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def load_model(variant: str, *, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """
    Load (or return from cache) UniVidIntrinsic or UniVidAlpha.

    variant: 'intrinsic' or 'alpha'.
    """
    if variant not in ("intrinsic", "alpha"):
        raise ValueError(f"variant must be 'intrinsic' or 'alpha', got {variant!r}")

    paths = resolve_paths(_comfy_root())
    ckpt = paths[f"univid_{variant}_ckpt"]
    cache_key = (variant, ckpt, device, dtype)

    with _LOAD_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        initialize()
        with unividx_cwd():
            from scripts.registry import MODEL_REGISTRY  # type: ignore

            cls_name = "UniVidIntrinsic" if variant == "intrinsic" else "UniVidAlpha"
            ModelCls = MODEL_REGISTRY[cls_name]

            # Construct the params exactly as UniVidX's YAML does. The model_paths
            # value is a JSON-string-of-a-list — that's how UniVidX parses it.
            t5 = paths["wan_t5"]
            vae = paths["wan_vae"]
            model_paths_json = f'["{t5}","{vae}"]'

            modalities = ["rgb", "albedo", "irradiance", "normal"] if variant == "intrinsic" \
                         else ["com", "pha", "fgr", "bgr"]

            model = ModelCls(
                model_paths=model_paths_json,
                lora_base_model="dit",
                lora_target_modules="self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2",
                lora_rank=32,
                lora_modalities=modalities,
                resume_from_checkpoint=ckpt,
            )
            # Switch to inference mode (equivalent to model.eval()).
            model.train(False)

        _MODEL_CACHE[cache_key] = model
        return model


def clear_cache() -> None:
    """Drop cached models. Call before unloading the plugin."""
    with _LOAD_LOCK:
        _MODEL_CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
