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
import logging
import os
import sys
import threading
from pathlib import Path

# Fail fast on stale ComfyUI installs that drop this folder into a
# Python 3.9 runtime — the `str | None` PEP 604 union syntax later in
# this module would otherwise raise a cryptic SyntaxError that surfaces
# as a vague node-load failure in the ComfyUI log.
if sys.version_info < (3, 10):
    raise RuntimeError(
        "UniVidX_ComfyUI requires Python 3.10+. "
        f"Current interpreter: {sys.version.split()[0]}"
    )

import torch

_log = logging.getLogger("unividx")

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
               compile_dit: bool = False, prefer_sage_attn: bool = False):
    """
    Load (or return from cache) UniVidIntrinsic or UniVidAlpha.

    variant: 'intrinsic' or 'alpha'.
    vram_buffer: GB of VRAM kept free for activations / KV cache / VAE
        decode. Passed straight to UniVidX's pipeline-level
        `enable_vram_management(vram_buffer=...)` (see
        `vendor/UniVidX/src/pipelines/univid_intrinsic.py`), which wraps
        the text encoder, DiT, and VAE through DiffSynth's low-level
        `enable_vram_management()` helper so layers live on CPU and
        stream to GPU only during forward. Higher values = more free
        VRAM but slower (more streaming); lower values pack more model
        in residency. On a 32 GB GPU, ~4.0 is a sane default for BF16
        Wan2.1-14B + LoRAs.
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
                 quantize_fp8, bool(compile_dit), bool(prefer_sage_attn))

    with _LOAD_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        initialize()

        # Always undo any global F.scaled_dot_product_attention pollution
        # done by other custom nodes (Stable3DGen monkey-patches it to
        # sage at import time when sageattention is installed). UniVidX's
        # VAE has 1-head SDPA where head_dim=channel_count which hits 384
        # in some blocks — sage raises ValueError on that, taking down
        # the entire run. We restore native SDPA on every load_model
        # call to stay self-healing.
        _restore_native_sdpa_if_polluted()

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
            # Apply the SageAttention dispatcher AFTER model construction
            # so UniVidX's vendored wan_video_dit modules are imported
            # and present in sys.modules — they have their own
            # flash_attention() copy that the CMSA path uses, separate
            # from DiffSynth's. We patch both.
            if prefer_sage_attn:
                if not _force_sage_over_fa2():
                    _log.warning(
                        "prefer_sage_attn=True but the SageAttention "
                        "dispatcher could not be installed. Most likely "
                        "cause: `sageattention` is not importable in "
                        "this venv (try `pip install sageattention` or "
                        "the woct0rdho prebuilt wheel — see README). "
                        "Continuing with default attention backend."
                    )

            if quantize_fp8:
                _quantize_dit_fp8(model, qtype=quantize_fp8)

            # Wire vram_buffer to UniVidX's pipeline-level VRAM manager.
            # UniVidIntrinsic / UniVidAlpha define enable_vram_management()
            # themselves (vendor/UniVidX/src/pipelines/univid_*.py) — it
            # wraps text encoder + DiT + VAE through DiffSynth's low-level
            # helper. The bf16 DiT is ~28 GB; without this the model
            # alone saturates a 32 GB card and per-step balloons to
            # 2-3+ min. With it, per-step is ~30 sec.
            if hasattr(model, "enable_vram_management"):
                try:
                    model.enable_vram_management(vram_buffer=float(vram_buffer))
                    _log.info("VRAM management enabled with vram_buffer=%.1f GB",
                              float(vram_buffer))
                except TypeError as exc:
                    _log.warning(
                        "model.enable_vram_management(vram_buffer=...) "
                        "rejected the kwarg: %s. VRAM management was NOT "
                        "applied — sampling may OOM or be memory-bound.",
                        exc,
                    )
            else:
                _log.warning(
                    "Model class %s lacks enable_vram_management(); "
                    "vram_buffer_gb has no effect on this build.",
                    type(model).__name__,
                )

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


def _restore_native_sdpa_if_polluted() -> bool:
    """Defensive: undo Stable3DGen's global F.scaled_dot_product_attention
    pollution.

    ComfyUI-3D-Pack's `Stable3DGen/trellis/backend_config.py` does
    `F.scaled_dot_product_attention = sageattn` at import time IF the
    `sageattention` package is importable. That hostile global swap
    breaks every other custom node that uses SDPA with head_dim outside
    sage's supported set (UniVidX's VAE has 1-head SDPA where
    head_dim=channel_count, hitting 384 in some block — sage raises
    Unsupported head_dim).

    `torch._C._nn.scaled_dot_product_attention` is the underlying C++
    implementation and is NOT touched by the Python-level swap, so we
    use it as the trusted reference for restoration. Idempotent: if
    F.SDPA is already native, this is a no-op.

    Returns True if a restore happened, False if no pollution detected.
    """
    import torch
    import torch.nn.functional as F
    native = torch._C._nn.scaled_dot_product_attention
    if F.scaled_dot_product_attention is native:
        return False
    F.scaled_dot_product_attention = native
    return True


# De-duplication map: (backend, head_dim, exc-type-name) -> True once warned.
_attention_fallback_warned: set = set()


def _warn_attention_fallback(backend: str, head_dim: int, exc: BaseException) -> None:
    """Log once per (backend, head_dim, exc-type) so we don't spam the
    console at every sampler step but still surface real failures."""
    key = (backend, head_dim, type(exc).__name__)
    if key in _attention_fallback_warned:
        return
    _attention_fallback_warned.add(key)
    _log.warning(
        "%s attention failed for head_dim=%d, falling back to next "
        "backend in chain. Cause: %s: %s",
        backend, head_dim, type(exc).__name__, exc,
    )


def _force_sage_over_fa2() -> bool:
    """Replace DiffSynth's Wan DiT attention with a SageAttention-first
    wrapper that falls back to FA2 (then SDPA) per-call.

    DiffSynth's stock `flash_attention()` is a hardcoded elif-chain:
    FA3 > FA2 > SAGE > SDPA. Two problems on Blackwell + sage 1.x:
      1. FA2 wins by default (FA3 is Hopper-only).
      2. SageAttention 1.x only supports head_dim in {64, 96, 128}; the
         Wan2.1-14B + UniVidX stack has at least one attention call
         with a different head_dim (cross-attention vs the T5 text
         encoder, typically), and sage raises hard if asked to handle
         it.

    We swap in our own function that:
      - tries SageAttention when head_dim ∈ {64, 96, 128}
      - falls back to FA2 for any other head_dim
      - falls back to torch SDPA if FA2 isn't available either

    Returns True if the wrapper was installed, False otherwise.
    """
    try:
        import sageattention  # noqa: F401
        from sageattention import sageattn
    except ImportError:
        return False
    try:
        from diffsynth.models import wan_video_dit
    except ImportError:
        return False
    if not getattr(wan_video_dit, "SAGE_ATTN_AVAILABLE", False):
        return False

    import torch.nn.functional as F
    from einops import rearrange

    # Capture FA2 reference if present so we can use it as the fallback.
    _FA2 = None
    if getattr(wan_video_dit, "FLASH_ATTN_2_AVAILABLE", False):
        try:
            import flash_attn
            _FA2 = flash_attn.flash_attn_func
        except ImportError:
            _FA2 = None

    # Cascading attention dispatcher: try sage → try FA2 → SDPA last.
    # Each backend is wrapped in try/except because UniVidX's CMSA
    # uses head_dim values (e.g. 384) that neither sage nor FA2
    # support — only SDPA can handle them. We also pre-skip sage/FA2
    # for head_dim > 256 to avoid the per-call exception cost on
    # known-bad shapes.
    #
    # `drop_out` and `**_kwargs` are accepted defensively: UniVidX's
    # vendored wan_video_dit_intrinsic / wan_video_dit_alpha pass a
    # drop_out kwarg through this signature for their CMSA routing
    # (cross-batch K/V concat). We don't patch those CMSA modules by
    # default (see the for-loop below), but if a future upstream rev
    # routes through our function we want the call to succeed rather
    # than fail with a TypeError on the unrecognised kwarg.
    def _wrapped_flash_attention(q, k, v, num_heads, compatibility_mode=False,
                                  drop_out=None, **_kwargs):
        head_dim = q.shape[-1] // num_heads

        if compatibility_mode or head_dim > 256:
            q2 = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k2 = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v2 = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            x = F.scaled_dot_product_attention(q2, k2, v2)
            return rearrange(x, "b n s d -> b s (n d)", n=num_heads)

        try:
            q2 = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k2 = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v2 = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            x = sageattn(q2, k2, v2)
            return rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        except Exception as exc:
            # Log once per (head_dim, exc-type) tuple so we don't spam the
            # console on every step but still surface real numerical
            # failures (CUDA OOM, NaN, dtype mismatch) instead of
            # silently degrading to SDPA.
            _warn_attention_fallback("sage", head_dim, exc)

        if _FA2 is not None:
            try:
                q2 = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
                k2 = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
                v2 = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
                x = _FA2(q2, k2, v2)
                return rearrange(x, "b s n d -> b s (n d)", n=num_heads)
            except Exception as exc:
                _warn_attention_fallback("FA2", head_dim, exc)

        q2 = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k2 = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v2 = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q2, k2, v2)
        return rearrange(x, "b n s d -> b s (n d)", n=num_heads)

    # Patch DiffSynth's Wan DiT (used by the pipeline's main DiT).
    wan_video_dit.flash_attention = _wrapped_flash_attention

    # ALSO patch UniVidX's vendored base wan_video_dit module — its
    # flash_attention() defaults to SDPA when FA3 is absent (always on
    # Blackwell). UniVidIntrinsic and UniVidAlpha import their own
    # variant-specific DiT modules (wan_video_dit_intrinsic /
    # wan_video_dit_alpha) which we deliberately leave UNPATCHED:
    #   - Both variant-specific modules call flash_attention with a
    #     `drop_out` kwarg that gates CMSA (cross-modal self-attention)
    #     routing — when active, K/V are reshaped across the batch
    #     dimension via `rearrange(... -> 1 n (b s) d).repeat(b, 1, 1, 1)`
    #     to make each batch element attend to every batch's K/V.
    #   - Mathematically sage/FA2 would still compute correct attention
    #     on the reshaped tensors, but the upstream behaviour is the
    #     load-bearing definition of UniVidX's multi-modality attention.
    #     Until we have a regression test pinning numerical equivalence
    #     of sage vs SDPA on that exact pattern, we keep the CMSA path
    #     on the original SDPA implementation for safety.
    # The non-CMSA paths (text cross-attention, single-modality self-
    # attention) still benefit from sage via the diffsynth and base
    # wan_video_dit patches above.
    import sys as _sys
    for mod_name in ("src.models.wan_video_dit",):
        mod = _sys.modules.get(mod_name)
        if mod is None or not hasattr(mod, "flash_attention"):
            continue
        mod.flash_attention = _wrapped_flash_attention
    return True


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
