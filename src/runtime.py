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
import json
import os
import sys
import threading
from pathlib import Path

import torch

from .path_resolver import ensure_symlinks, resolve_paths


_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _THIS_DIR.parent  # custom_nodes/UniVidX_ComfyUI/
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

            # mmgp monkey-patches safetensors.torch.load_file with a writable_tensors=True
            # default. On Windows that uses ACCESS_COPY mmap, which forces a virtual-memory
            # commit equal to the file size. Six 9.84 GB DiT shards mmapped concurrently
            # blow past most users' paging file -> [WinError 1455]. UniVidX only ever
            # reads the loaded tensors (state_dict.update + load_state_dict), so a read-
            # only mmap is sufficient and skips the commit charge.
            _patch_unividx_load_file_to_readonly()

            cls_name = "UniVidIntrinsic" if variant == "intrinsic" else "UniVidAlpha"
            ModelCls = MODEL_REGISTRY[cls_name]

            # Construct the params exactly as UniVidX's YAML does. The model_paths
            # value is a JSON-string-of-a-list — that's how UniVidX parses it.
            t5 = paths["wan_t5"]
            vae = paths["wan_vae"]
            # json.dumps so Windows backslashes are escaped — UniVidX json.loads's this string.
            model_paths_json = json.dumps([t5, vae])

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

            # Enable layer-by-layer VRAM management. The Wan2.1-T2V-14B DiT is
            # ~28 GB FP16; on a 32 GB GPU the model alone would saturate VRAM
            # and leave no headroom for activations / KV cache / VAE decode.
            # DiffSynth-Studio's enable_vram_management wraps Linear/Conv/Norm
            # modules so they live on CPU and stream to GPU only during forward.
            # vram_buffer=4.0 keeps ~4 GB free for activations.
            #
            # Tunable: pass a smaller buffer for more headroom (slower per step
            # but leaves room for a higher resolution). Pass num_persistent_param_in_dit
            # to keep a specific number of DiT params resident.
            if hasattr(model, "pipe") and hasattr(model.pipe, "enable_vram_management"):
                model.pipe.enable_vram_management(vram_buffer=4.0)

        _MODEL_CACHE[cache_key] = model
        return model


def _patch_unividx_load_file_to_readonly() -> None:
    """
    Replace UniVidX's load_file binding so it uses mmgp.torch_load_file with
    writable_tensors=False. Idempotent. Called inside the unividx_cwd context
    after registry import (which imports the pipeline modules).
    """
    import sys as _sys
    try:
        import mmgp.safetensors2 as _mmgp_st
    except ImportError:
        return  # mmgp not installed — nothing to patch

    def _readonly_load_file(filename, device="cpu"):
        return _mmgp_st.torch_load_file(filename, device=device, writable_tensors=False)
    _readonly_load_file._unividx_readonly_patch = True  # type: ignore[attr-defined]

    for mod_name in ("src.pipelines.univid_intrinsic", "src.pipelines.univid_alpha"):
        mod = _sys.modules.get(mod_name)
        if mod is None:
            continue
        existing = getattr(mod, "load_file", None)
        if getattr(existing, "_unividx_readonly_patch", False):
            continue
        mod.load_file = _readonly_load_file


def clear_cache() -> None:
    """Drop cached models. Call before unloading the plugin."""
    with _LOAD_LOCK:
        _MODEL_CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
