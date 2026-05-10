# Changelog

## 0.3.0 — 2026-05-11

Tier-A correctness fixes from `ROADMAP_v0.3.md`, **with the roadmap's root-cause diagnosis corrected mid-flight** (see "Diagnosis correction" below). The headline number from Tier A5: **`vram_buffer_gb` 4.0 vs 12.0 is +65% wall** (10.36 min → 17.10 min on baseline R2AIN_video, RTX 5090, no sage/compile) — making it the single biggest perf knob in the system, not the "deprecated" one 0.2.1 docs claimed.

History note: this release ships as `0.3.0` after a transient buggy `0.3.0-rc1` (commit [`a73cdca`](https://github.com/dreamrec/UniVidX_ComfyUI/commit/a73cdca)) followed the roadmap's misdiagnosis and retargeted the call from `model.pipe.enable_vram_management(...)` (correct) to `model.enable_vram_management(...)` (wrong — the method doesn't exist on the outer class). The retarget was reverted in commit [`1f04a9b`](https://github.com/dreamrec/UniVidX_ComfyUI/commit/1f04a9b). Live-runtime sanity probe + 30-min bench in this commit confirm the corrected wiring on real hardware.

### Fixed

- **`vram_buffer` re-added to the cache key.** 0.2.0 dropped it on the (mistaken) belief that the underlying call was a no-op, so two `UniVidXLoader` nodes with different `vram_buffer_gb` settings collided into a single cache entry — the second node silently inherited whichever value loaded first, contradicting its own UI. Restored to the tuple: `(variant, ckpt, device, dtype, float(vram_buffer), quantize_fp8, compile_dit, prefer_sage_attn)`.
- **`if/else` with explicit WARNING-on-missing-method** around the `enable_vram_management` call. The original 0.2.1 code used a bare `hasattr` check that silently skipped if the method was missing — exactly the failure mode that led to the misdiagnosis the roadmap was written against (see below). New structure logs `INFO` on success ("VRAM management enabled with vram_buffer=4.0 GB") and `WARNING` if the method is genuinely absent (future upstream rev). A regression here will be visible in the ComfyUI console.
- **Docs corrected.** Loader widget tooltip and README no longer claim `vram_buffer_gb` is "deprecated, no-op." The "What does NOT help on Blackwell" section drops the bullet. The Node Overview cache-key tuple lists `vram_buffer` explicitly.

### Diagnosis correction (vs ROADMAP_v0.3.md)

The roadmap claimed `vram_buffer_gb` was a silent no-op in 0.1.0–0.2.1 because `runtime.py` called `model.pipe.enable_vram_management(...)` and `model.pipe` was supposedly DiffSynth's stock `WanVideoPipeline` (no such method). **That diagnosis was wrong.** `model.pipe` is an instance of UniVidX's own `WanVideoPipeline` subclass defined locally at `vendor/UniVidX/src/pipelines/univid_intrinsic.py:24`, and `enable_vram_management()` is a method on that class at line 210 — exactly what the original code targeted. Live runtime probe (Tier-A1 sanity check) confirmed: after pointing the call at `model` per the roadmap, a `WARNING: Model class UniVidIntrinsic lacks enable_vram_management()` fired — proving the method does NOT exist on the outer class, contrary to the roadmap. The fix reverts to `model.pipe.enable_vram_management(...)` (the proven-working 0.2.1 target). What 0.3.0 actually fixes is **the cache-key bug + the doc inaccuracy**; the wiring was correct all along.

### Added

- **4 new unit tests in `tests/test_runtime_cache_key.py`**: pins `model.pipe.enable_vram_management(vram_buffer=...)` as the call target, asserts a WARNING fires when `.pipe` lacks the method, and asserts cache-hit/cache-miss behaviour for matching/distinct `vram_buffer` values. 35 → 39 unit tests.
- **`examples/_bench_vram_buffer.py`** — wall + per-step measurement harness used to produce the +65% delta number. Queues both conditions back-to-back via `/prompt`, polls `/history` to completion, extracts timings from the status-messages timeline.
- **`examples/_sanity_a1.py`** — one-shot tiny-workflow probe that confirms the corrected code is live by grepping the ComfyUI log for the new `VRAM management enabled with vram_buffer=4.0 GB` INFO line.

### Tier A5 measurement (RTX 5090, R2AIN_video baseline 480×640×21×20, no sage/compile)

| `vram_buffer_gb` | Wall (server) | Δ |
|---|---|---|
| 4.0 | 10.36 min (621.6 s) | baseline |
| 12.0 | 17.10 min (1026.1 s) | **+6.74 min / +65.1%** |

Lower buffer = more residency = faster. 4.0 is near-optimal on 32 GB; raise only if you hit OOM. **Translation: in 0.1.0–0.2.1, anyone who saw the "deprecated, no-op" tooltip and left `vram_buffer_gb` at the default got lucky** — the call DID work, the docs just lied about it. Anyone who *raised* the value thinking it was inert was silently making their runs up to 65% slower.

## 0.2.1 — 2026-05-11

External-review fixes (h/t the second-pass code review against `4c1f282`):

### Fixed

- **Latent crash in alpha-variant + `prefer_sage_attn=True`.** UniVidX's vendored `wan_video_dit_alpha.py:133` calls `flash_attention(..., drop_out=drop_out)`, but our wrapper signature didn't accept `drop_out` — any alpha-mode run with sage enabled would have raised `TypeError`. Wrapper now accepts `drop_out=None` and `**_kwargs` defensively. (We never benched alpha-with-sage so the bug went undetected; latent.)
- **Sage patch was missing the variant-specific CMSA modules.** UniVidIntrinsic / UniVidAlpha import their attention from `wan_video_dit_intrinsic` / `wan_video_dit_alpha`, NOT the base `wan_video_dit` we patched. The 0.2.0 measured 18 % sage win came from non-CMSA paths only (text cross-attention, single-modality self-attention). **Decision: continue NOT patching the CMSA modules** — they reshape K/V across the batch dimension via `repeat(batch_size, 1, 1, 1)` which mathematically would still be valid sage attention, but until we have a numerical-equivalence test we keep the CMSA path on the original SDPA implementation for safety. New regression test asserts the CMSA modules are NOT touched.
- **`__init__.py` silent ImportError.** The `try/except ImportError` chain swallowed both errors with no log, so a real install bug (missing torch, broken submodule, etc.) made the nodes silently disappear from the ComfyUI sidebar with no clue why. Now logs both errors via the `unividx` logger.
- **README cache-key drift.** Node Overview section claimed cache is `(variant, dtype, device, vram_buffer, ...)` but 0.2.0 explicitly removed `vram_buffer` from the key. Updated to match the actual tuple.
- **README install.py drift.** Claimed `install.py` creates symlinks; it doesn't — it verifies the submodule, copies workflows, and prints download hints. The actual symlink/junction creation happens lazily at first model load via `runtime.initialize() → path_resolver.ensure_symlinks()`. Now documented accurately.

### Added

- 2 new regression tests in `tests/test_runtime_drop_out.py` pinning the `drop_out` kwarg acceptance + the variant-DiT patch-selection invariant.
- README Roadmap → FP8: expanded into a 7-step refactor plan (split `dtype` into `compute_dtype` + `dit_weight_mode`, add pre-quantized weight loader bypassing the hardcoded BF16 six-shard loop, keep PEFT LoRAs in BF16, two-phase rollout for correctness then speed).

### Known limitation (not fixed in this release)

- **`unividx_cwd()` is process-global**, not thread-safe. Concurrent ComfyUI queue runs that both call `unividx_cwd()` will race on the process `cwd`. Realistic fix requires patching upstream UniVidX's hardcoded relative paths (out of scope) or a per-call `threading.Lock` (deferred). Documented; tracked in Roadmap.

## 0.2.0 — 2026-05-10

### Added

- **Loader perf knobs** for RTX 50-series and 4090-class GPUs:
  - `compile_dit` (BOOLEAN) — `torch.compile(dit, mode='reduce-overhead', dynamic=True)` after model load. Measured **−17 % wall, −28 % per-step** on a 5090 R2AIN run. First sampler step pays ~90 s graph capture.
  - `prefer_sage_attn` (BOOLEAN) — installs a sage→FA2→SDPA cascade dispatcher across DiffSynth's *and* UniVidX's vendored `wan_video_dit.flash_attention()`. Measured **−18 % wall, −21 % per-step** with quality verified visually identical to FA2.
  - `dtype` adds `fp8_e4m3fn` and `fp8_e5m2` (post-quantize via `mmgp.offload.quantize`). **EXPERIMENTAL** — quantize pass hung in our cold-load test. Tooltip and README flag the risk; see Roadmap for the planned pre-quantized-weights replacement (option B).
- **`R2AIN_video_api.json` / `R2PFB_video_api.json`** — proper video-conditioned workflows using `VHS_LoadVideoPath` instead of the previous single-still-repeated-21× pattern. Generated by the new self-contained `examples/_build_video_workflows.py`.
- **AGGRESSIVE 4-step preview preset** documented — `prefer_sage_attn=True` + `num_inference_steps=4` + `cfg_scale=1.0` on RGB-conditioned modes. Measured **1.35 min wall vs 17.6 min baseline (13× speedup)** with quality verified usable on portrait test clip.
- **9 new mock-based unit tests** (`tests/test_runtime_patches.py`) for `_restore_native_sdpa_if_polluted`, `_force_sage_over_fa2`, and `_warn_attention_fallback` — no GPU required.
- **5 new SVG node cards** + **1 PNG workflow diagram** + result quad images for the README's Visual Tour section.
- **`examples/_bench_perf.py`** — small CLI to queue R2AIN_video benchmarks with chosen perf settings.

### Changed

- **README rewritten** for clarity — 417 → 229 lines. Two-preset (PRODUCTION / PREVIEW) recommendation surfaced near the top. Optimization knobs consolidated into one section with measured numbers.
- `examples/_build_ui_workflows.py` no longer generates the deleted `_basic_` workflows.
- Loader cache key updated to include the new perf-knob fields. Old saved workflows still validate (new widgets are in the optional dict with defaults).

### Removed

- `examples/R2AIN_basic.json` + `_api.json` and `examples/R2PFB_basic.json` + `_api.json` — replaced by the new `_video_api.json` workflows. The test_matrix's `C_RA2IN.json` / `D_RAI2N.json` / `F_R2PFB.json` continue to cover the regression-test path the `_basic_` files implicitly served.
- `examples/_smoke_runner_R2AIN.py` — specific to deleted workflow.

### Fixed

- **Cross-plugin SDPA pollution** — ComfyUI-3D-Pack's `Stable3DGen/trellis/backend_config.py` does `F.scaled_dot_product_attention = sageattn` *globally* at module import when `sageattention` is importable. That hostile swap broke UniVidX's VAE 1-head SDPA where head_dim=channel_count=384. New `_restore_native_sdpa_if_polluted()` runs at every `load_model()` and `sample()` call, restoring `F.scaled_dot_product_attention` from `torch._C._nn.scaled_dot_product_attention` (the C++ impl, immune to Python alias rebinding).
- **Silent sage no-op** — `prefer_sage_attn=True` now logs a `WARNING` if the dispatcher can't be installed (sageattention missing or DiffSynth's flag is False) instead of silently doing nothing.
- **Bare-except in attention fallback** — the sage→FA2→SDPA cascade now logs each fallback once per `(backend, head_dim, exc-type)` so genuine numerical failures (CUDA OOM, NaN) don't silently degrade output to SDPA.
- **`vram_buffer` dropped from cache key** — it was a no-op on current DiffSynth; including it in the key caused spurious double-loads when two loader nodes differed only in this dead value.
- **Dead local removed** in `loader.load()` (unused `quantize_fp8 = bool` shadowed the meaningful `fp8_variant` string).
- **Python 3.10 runtime guard** in `runtime.py` so stale ComfyUI installs fail with a clear `RuntimeError` instead of a cryptic `SyntaxError` from PEP-604 union syntax.

### Roadmap (next iteration)

- **FP8 via pre-quantized Kijai weights** (replaces the hung runtime-quantize path). Load FP8 safetensors directly from [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) instead of running `mmgp.offload.quantize()` on cold-load. Targets ~14 GB DiT residency, ~30-40 % additional speedup on top of `prefer_sage_attn`.
- **Step-distill LoRA stacking** — try [LightX2V `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32`](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v) on top of UniVidX's per-modality LoRA. Could close the small quality gap of the PREVIEW 4-step preset.
- **DiffSynth `vram_limit` re-wire** — current diffsynth replaced `enable_vram_management` with `ModelConfig`-driven `vram_limit` + per-config `offload_dtype`/`onload_dtype`. Re-wire our `vram_buffer_gb` knob to the new API instead of the no-op `hasattr`-guarded legacy call.
- **Thread-safe `unividx_cwd()`** — `os.chdir()` in this context manager is process-global and races across concurrent ComfyUI queue threads. Currently a known limitation; fix likely requires patching upstream UniVidX's hardcoded relative paths.

## 0.1.0 — initial release

- Strategy A wrapper for UniVidX (SIGGRAPH 2026): 5 nodes, 30 task modes across two model variants.
- Vendored UniVidX as pinned git submodule under `vendor/UniVidX/`.
- Three Windows-specific patches (JSON path escaping, mmgp readonly mmap, junctions instead of symlinks).
- 24 unit tests + 10-entry integration matrix.
