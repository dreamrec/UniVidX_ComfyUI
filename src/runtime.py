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
from collections import OrderedDict
from pathlib import Path
from typing import Optional

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

# LRU-bounded model cache. Each entry is ~14 GB FP8 or ~28 GB BF16
# in steady-state VRAM + host RAM, so growing this dict unbounded is
# how a multi-condition bench can thrash a 32 GB card. Default cap of
# 2 covers the common cases (single-loader graphs + two-loader graphs
# with different settings) and forces an LRU eviction on the third
# distinct cache key.
#
# Settable via the UNIVIDX_MODEL_CACHE_MAX env var for advanced users
# running headless validation matrices on cards with more VRAM. Tests
# rewrite _MODEL_CACHE_MAX_SIZE directly.
_MODEL_CACHE: "OrderedDict" = OrderedDict()
_MODEL_CACHE_MAX_SIZE: int = int(os.environ.get("UNIVIDX_MODEL_CACHE_MAX", "2"))


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
               compile_dit: bool = False, prefer_sage_attn: bool = False,
               dit_weight_mode: str = "bf16_shards",
               step_distill_lora: str = "none",
               step_distill_strength: float = 1.0):
    """
    Load (or return from cache) UniVidIntrinsic or UniVidAlpha.

    variant: 'intrinsic' or 'alpha'.
    vram_buffer: GB of VRAM kept free for activations / KV cache / VAE
        decode. Passed to `model.pipe.enable_vram_management(vram_buffer=...)`
        — the method lives on UniVidX's OWN WanVideoPipeline subclass
        (vendor/UniVidX/src/pipelines/univid_intrinsic.py:24, method at
        line 210), NOT on UniVidIntrinsic itself or on DiffSynth's stock
        WanVideoPipeline. It wraps text encoder + DiT + VAE through
        DiffSynth's low-level enable_vram_management() helper so layers
        live on CPU and stream to GPU only during forward. Higher =
        more free VRAM but slower (more streaming); lower = more
        residency. 4.0 is a sane default for BF16 Wan2.1-14B + LoRAs on
        32 GB cards.
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
                 quantize_fp8, bool(compile_dit), bool(prefer_sage_attn),
                 dit_weight_mode, step_distill_lora,
                 float(step_distill_strength))

    with _LOAD_LOCK:
        if cache_key in _MODEL_CACHE:
            # Touch the entry so it becomes most-recently-used.
            _MODEL_CACHE.move_to_end(cache_key)
            return _MODEL_CACHE[cache_key]

        # Cache miss. Evict LRU entries BEFORE constructing the new
        # model so we never have two big models in VRAM simultaneously
        # (peak host RAM during cold-load is ~28 GB; doubling that
        # OOMs on a 32 GB card with anything else resident).
        # Index map for cache_key tuple — keep in sync with the
        # cache_key construction below. Indexed access here so the
        # diagnostic log stays correct when cache_key gains new fields.
        # 0=variant, 1=ckpt, 2=device, 3=dtype, 4=vram_buffer,
        # 5=quantize_fp8, 6=compile_dit, 7=prefer_sage_attn,
        # 8=dit_weight_mode, 9=step_distill_lora, 10=step_distill_strength
        while len(_MODEL_CACHE) >= max(1, _MODEL_CACHE_MAX_SIZE):
            evicted_key, evicted_model = _MODEL_CACHE.popitem(last=False)
            _log.info(
                "Evicting LRU model cache entry (variant=%s, "
                "dit_weight_mode=%s, vram_buffer=%s, "
                "step_distill_lora=%s, step_distill_strength=%s) "
                "to stay within UNIVIDX_MODEL_CACHE_MAX=%d",
                evicted_key[0],     # variant
                evicted_key[8],     # dit_weight_mode (was -1, that's step_distill_strength)
                evicted_key[4],     # vram_buffer
                evicted_key[9],     # step_distill_lora
                evicted_key[10],    # step_distill_strength
                _MODEL_CACHE_MAX_SIZE,
            )
            del evicted_model
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

            # Tier C: step-distill merge MUST happen before
            # enable_vram_management(). Why: enable_vram_management wraps
            # non-Linear modules (RMSNorms, patch_embedding, head) in
            # AutoWrappedModule whose internal layout indirects through
            # `.module.weight` — at which point LightX2V's `.diff` keys
            # (addressed against bare `<base>.weight` paths) silently
            # skip. The lora_merge resolver now descends through both
            # PEFT `.base_layer` AND vram-mgmt `.module` for robustness,
            # but doing the merge first means we never depend on the
            # second descent (less surface for the same kind of bug
            # to recur in a future refactor).
            if step_distill_lora and step_distill_lora != "none":
                _apply_step_distill_merge(
                    model, lora_kind=step_distill_lora,
                    strength=float(step_distill_strength),
                )

            # Wire vram_buffer to UniVidX's pipeline-level VRAM manager.
            # The method lives on `model.pipe` — an instance of UniVidX's
            # OWN WanVideoPipeline subclass (vendor/UniVidX/src/pipelines/
            # univid_intrinsic.py:24, with enable_vram_management() at
            # line 210), NOT DiffSynth's stock WanVideoPipeline.
            # UniVidIntrinsic / UniVidAlpha themselves don't define it —
            # they delegate to their `.pipe` attribute. The method wraps
            # text encoder + DiT + VAE through DiffSynth's low-level
            # offload helper. The bf16 DiT is ~28 GB; without this the
            # model alone saturates a 32 GB card and per-step balloons
            # to 2-3+ min. With it, per-step is ~30 sec.
            pipe = getattr(model, "pipe", None)
            if pipe is not None and hasattr(pipe, "enable_vram_management"):
                try:
                    pipe.enable_vram_management(vram_buffer=float(vram_buffer))
                    _log.info("VRAM management enabled with vram_buffer=%.1f GB",
                              float(vram_buffer))
                except TypeError as exc:
                    _log.warning(
                        "model.pipe.enable_vram_management(vram_buffer=...) "
                        "rejected the kwarg: %s. VRAM management was NOT "
                        "applied — sampling may OOM or be memory-bound.",
                        exc,
                    )
            else:
                _log.warning(
                    "model.pipe.enable_vram_management() not found on "
                    "%s; vram_buffer_gb has no effect on this build.",
                    type(model).__name__,
                )

            if dit_weight_mode == "fp8_prequantized":
                _apply_fp8_substitution(model, variant)

            if compile_dit:
                _compile_dit(model)

        _MODEL_CACHE[cache_key] = model
        return model


def _resolve_step_distill_path(lora_kind: str) -> Optional[str]:
    """Locate a step-distill LoRA file on disk by kind. Returns None
    if not found so the caller can surface a clear error.

    Search candidates per kind:
      lightx2v: models/loras/lightx2v/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
                (the layout `hf download lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v
                 loras/...  --local-dir models/loras/lightx2v` produces)
                Falls back to a flat copy at models/loras/lightx2v/<filename>.
    """
    root = Path(_comfy_root())
    if lora_kind == "lightx2v":
        candidates = [
            root / "models" / "loras" / "lightx2v" / "loras"
            / "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
            root / "models" / "loras" / "lightx2v"
            / "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
            root / "models" / "loras"
            / "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        ]
    else:
        return None
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _apply_step_distill_merge(model, *, lora_kind: str, strength: float) -> None:
    """Load a step-distill LoRA safetensors and merge it into the DiT's
    base weights in place.

    For LightX2V: keys are prefixed with `diffusion_model.`, use Kohya
    `lora_down/lora_up` naming for Linear deltas, plus `.diff_b` for
    bias deltas and `.diff` for direct weight deltas on non-Linear
    modules (RMSNorm, patch_embedding, head).

    PEFT-aware via lora_merge._descend_peft — the merge applies to the
    .base_layer Linear inside UniVidX's per-modality wrappers; the
    LoRA siblings (lora_A_rgb/albedo/irradiance/normal etc.) stay
    untouched.
    """
    path = _resolve_step_distill_path(lora_kind)
    if path is None:
        raise FileNotFoundError(
            f"Step-distill LoRA '{lora_kind}' not found on disk. "
            f"Expected layout:\n"
            f"  models/loras/lightx2v/loras/"
            f"Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors\n\n"
            f"Download via:\n"
            f"  hf download lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v "
            f"loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors "
            f"--local-dir ComfyUI/models/loras/lightx2v\n\n"
            f"Or set step_distill_lora='none' to skip the merge."
        )
    from safetensors.torch import load_file as _load_safetensors
    from .lora_merge import merge_lora_into_base

    _log.info("Loading step-distill LoRA (%s) from %s", lora_kind, path)
    sd = _load_safetensors(path)
    try:
        report = merge_lora_into_base(
            model.pipe.dit, sd,
            strength=strength,
            strip_prefix="diffusion_model.",
        )
    finally:
        del sd
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _log.info(
        "Step-distill merge complete (%s, strength=%.2f): "
        "%d Linears merged, %d biases patched, %d weights patched, "
        "%d unmatched, %d skipped",
        lora_kind, strength,
        report["merged"], report["biases_patched"],
        report["weights_patched"], report["unmatched"], report["skipped"],
    )
    if report["merged"] == 0 and report["weights_patched"] == 0:
        _log.warning(
            "Step-distill merge applied 0 deltas — the LoRA's key "
            "structure may not match UniVidX's WanModel layout."
        )


def _resolve_fp8_weights_path() -> str | None:
    """Probe for a Kijai pre-quantized Wan2.1-T2V-14B FP8 weights file
    with PER-TENSOR SCALES (the `_scaled` filename suffix).

    Returns the path if a suitable file is present; returns ``None``
    if not — the caller falls back to the runtime-quantize path
    (which is the 0.4.0 default, computes per-tensor scales from the
    BF16 cold-load weights itself).

    Rationale for only the `_scaled` variant:
      - The bare-cast `Wan2_1-T2V-14B_fp8_e4m3fn.safetensors` (no
        `_scaled` suffix) is what's available on Kijai/WanVideo_comfy
        today, but it ships without per-tensor scales — every tested
        modality measured 21-31 dB PSNR vs BF16 at tiny step counts.
      - The `_scaled` variant DOES exist for Wan2.2 (e.g.
        `wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors`)
        but NOT yet for Wan2.1-T2V-14B as of 0.4.0 release.
      - When a `_scaled` Wan2.1 file lands upstream, dropping it into
        models/diffusion_models/ will auto-enable the file-based load
        (faster cold-load than runtime quantize, same quality).
    """
    candidates = [
        Path(_comfy_root()) / "models" / "diffusion_models"
        / "Wan2_1-T2V-14B_fp8_e4m3fn_scaled.safetensors",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _apply_fp8_substitution(model, variant: str) -> None:
    """Convert the DiT's nn.Linear modules to FP8 e4m3fn.

    Two implementations, picked at runtime:
      1. **File-based** — if a Kijai `_scaled` Wan2.1-T2V-14B FP8
         safetensors is present at
         models/diffusion_models/Wan2_1-T2V-14B_fp8_e4m3fn_scaled.safetensors,
         load the FP8 weights + per-tensor scales directly via
         load_fp8_state_dict_into. Faster cold-load (no quantize pass),
         identical numerics to (2). As of 0.4.0 release, this file
         doesn't exist upstream for Wan2.1 — drop one in to opt in
         when it does.
      2. **Runtime quantize (default)** — compute per-tensor absmax
         scales from the BF16 cold-load weights and cast to FP8 in
         place via quantize_dit_inplace. Same memory + quality result
         as (1) but adds ~30 sec to cold load (walks 400+ Linears).

    Both paths land at the same end state: ~400 FP8Linear modules
    inside the DiT, LoRA adapters preserved at BF16 (B5 contract),
    ~14 GB steady-state VRAM (vs ~28 GB BF16).

    History: an earlier B3 design tried Kijai's `_fp8_e4m3fn` file
    (without `_scaled` suffix) — that's a bare BF16->FP8 cast and
    gave 21-31 dB PSNR. Path (1) only triggers on the `_scaled`
    variant for that reason. See CHANGELOG 0.4.0-rc1.
    """
    fp8_path = _resolve_fp8_weights_path()
    if fp8_path is not None:
        _apply_fp8_substitution_from_file(model, fp8_path)
    else:
        _apply_fp8_substitution_runtime_quantize(model)


def _apply_fp8_substitution_runtime_quantize(model) -> None:
    """Compute per-tensor absmax scales from the BF16 cold-load weights
    and cast to FP8 in place. The 0.4.0 default path."""
    from .fp8_loader import quantize_dit_inplace

    _log.info("Quantizing DiT to FP8 e4m3fn (per-tensor absmax scaling)")
    report = quantize_dit_inplace(model.pipe.dit)
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log.info(
        "FP8 substitution complete: %d Linears quantized to FP8Linear "
        "(per-tensor scaled, runtime)",
        report["linears_quantized"],
    )
    if report["linears_quantized"] == 0:
        _log.warning(
            "FP8 substitution replaced 0 Linears — the DiT may not "
            "contain any nn.Linear modules at the expected positions, "
            "or PEFT wrapping made them unreachable via attribute "
            "walk. Sampling will run at full BF16."
        )


def _apply_fp8_substitution_from_file(model, fp8_path: str) -> None:
    """Stream-load a Kijai `_scaled` FP8 safetensors and replace DiT
    Linears with FP8Linear carrying the pre-baked weight + scales.

    Triggered automatically when the resolver finds a matching file.
    Doesn't currently fire for any shipped Kijai Wan2.1 file (the
    `_scaled` variant doesn't exist upstream yet) — included as the
    forward-compat path for when it does."""
    from safetensors.torch import load_file as _load_safetensors
    from .fp8_loader import load_fp8_state_dict_into

    _log.info("Loading Kijai _scaled FP8 base from %s", fp8_path)
    fp8_sd = _load_safetensors(fp8_path)
    try:
        report = load_fp8_state_dict_into(model.pipe.dit, fp8_sd)
    finally:
        del fp8_sd
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _log.info(
        "FP8 substitution complete: %d Linears -> FP8Linear "
        "(file-based, %d aux loaded, %d unmatched)",
        report["fp8_linears_replaced"],
        report["aux_keys_loaded"],
        len(report["unmatched_keys"]),
    )
    if report["fp8_linears_replaced"] == 0:
        _log.warning(
            "FP8 substitution (file-based) replaced 0 Linears — the "
            "file's key convention may not match UniVidX's WanModel "
            "layout. Run examples/_audit_kijai_fp8.py %s to diagnose.",
            fp8_path,
        )


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
