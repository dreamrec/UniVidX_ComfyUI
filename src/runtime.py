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


def load_model(variant: str, *, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
               vram_buffer: float = 4.0, quantize_fp8: str | None = None,
               compile_dit: bool = False):
    """
    Load (or return from cache) UniVidIntrinsic or UniVidAlpha.

    variant: 'intrinsic' or 'alpha'.
    vram_buffer: DEPRECATED on current DiffSynth (the
        WanVideoPipeline.enable_vram_management entrypoint was removed).
        Cached separately per value purely so old workflows that wired
        this in still get a unique cache slot; the value otherwise has
        no effect on speed.
    quantize_fp8: None (default), or one of optimum-quanto's qtype names —
        "qfloat8" (e4m3, larger mantissa) or "qfloat8_e5m2" (larger
        exponent). Post-quantizes the DiT via mmgp.offload.quantize.
        Halves DiT memory (~28 GB BF16 → ~14 GB FP8), enabling full
        residency on 32 GB GPUs and Blackwell-native FP8 matmul.
        EXPERIMENTAL: the quantize() pass over the LoRA-attached DiT can
        take 10+ min and has hung on Wan2.1-14B + UniVidX's adapter
        stack in our testing. LoRA layers are excluded from quantization
        to preserve adapter precision.
    compile_dit: torch.compile(dit, mode='reduce-overhead', dynamic=True)
        after model assembly. First sampler step pays a 60-120 sec graph
        capture cost; subsequent steps are typically 20-30% faster on
        Blackwell/Ada. The compile cache is keyed by tensor shapes — if
        you change resolution or frame count between runs, the next run
        re-captures.
    """
    if variant not in ("intrinsic", "alpha"):
        raise ValueError(f"variant must be 'intrinsic' or 'alpha', got {variant!r}")

    paths = resolve_paths(_comfy_root())
    ckpt = paths[f"univid_{variant}_ckpt"]
    cache_key = (variant, ckpt, device, dtype, float(vram_buffer),
                 quantize_fp8, bool(compile_dit))

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
            # Optional post-load FP8 quantization. UniVidX hardcodes
            # WanVideoPipeline construction at bfloat16; we can't change
            # that without patching upstream, but we *can* post-quantize
            # the assembled DiT (base + LoRA-attached) to qfloat8 via
            # optimum-quanto. We exclude lora_A/lora_B/lora_embedding
            # layers from quantization so the LoRA adapter stays full
            # precision. On a 32 GB GPU the FP8 DiT (~14 GB) fits fully
            # resident, so we also drop the VRAM buffer to 0.0 (no need
            # to stream when everything's already on-chip).
            if quantize_fp8:
                _quantize_dit_fp8(model, qtype=quantize_fp8)

            # Backwards-compat with old saved workflows — the underlying
            # diffsynth API was removed, so this is a no-op on the
            # current build. Kept so the hasattr-guard fires cleanly if
            # a future diffsynth version reintroduces it.
            if hasattr(model, "pipe") and hasattr(model.pipe, "enable_vram_management"):
                model.pipe.enable_vram_management(vram_buffer=float(vram_buffer))

            if compile_dit:
                _compile_dit(model)

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


def _compile_dit(model) -> None:
    """torch.compile the assembled DiT for ~20-30% per-step speedup on
    Blackwell/Ada. Mode 'reduce-overhead' uses CUDA Graphs where safe;
    dynamic=True keeps a single compile valid across resolution changes
    (with a small per-shape recompile cost on first hit).
    """
    dit = getattr(getattr(model, "pipe", None), "dit", None)
    if dit is None:
        raise RuntimeError(
            "Cannot find model.pipe.dit to compile — UniVidX pipeline "
            "structure may have changed."
        )
    model.pipe.dit = torch.compile(dit, mode="reduce-overhead", dynamic=True, fullgraph=False)


def _quantize_dit_fp8(model, qtype: str = "qfloat8") -> None:
    """Post-quantize model.pipe.dit to FP8 via mmgp/optimum-quanto.

    qtype is an optimum-quanto qtype name — "qfloat8" (e4m3) or
    "qfloat8_e5m2". LoRA adapter layers (lora_A/B/_embedding_A/B) are
    excluded so the rank-32 adapters stay in BF16 — quantizing rank-32
    Linear layers gives no memory win and tends to degrade adapter
    quality.
    """
    try:
        from mmgp import offload as _mmgp_offload
    except ImportError as e:
        raise RuntimeError(
            "FP8 quantization requested but mmgp is not installed. "
            "Run `pip install mmgp` or pick a non-FP8 dtype."
        ) from e

    dit = getattr(getattr(model, "pipe", None), "dit", None)
    if dit is None:
        raise RuntimeError(
            "Cannot find model.pipe.dit to quantize — UniVidX pipeline "
            "structure may have changed."
        )
    _mmgp_offload.quantize(
        dit,
        weights=qtype,
        exclude=["*lora_A*", "*lora_B*", "*lora_embedding_A*", "*lora_embedding_B*"],
    )


def clear_cache() -> None:
    """Drop cached models. Call before unloading the plugin."""
    with _LOAD_LOCK:
        _MODEL_CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
