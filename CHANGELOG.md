# Changelog

## 0.3.0-rc1 — 2026-05-11

Tier-A correctness fixes from `ROADMAP_v0.3.md`. Tagged as a release candidate because the perf delta on real hardware (Tier A5) is pending re-bench; the code change is complete and unit-test-covered.

### Fixed

- **`vram_buffer_gb` was a silent no-op in 0.1.0–0.2.1.** `src/runtime.py` called `model.pipe.enable_vram_management(vram_buffer=...)` — but `model.pipe` is DiffSynth's `WanVideoPipeline`, which has no such method. The `hasattr` guard returned False every time, so the call was skipped without logging. The actual entrypoint is `model.enable_vram_management(...)` on UniVidX's pipeline class (`vendor/UniVidX/src/pipelines/univid_intrinsic.py:210` and `univid_alpha.py:203`), which wraps text encoder + DiT + VAE through DiffSynth's low-level offload helper. The fix is a one-token change (`model.pipe` → `model`), wrapped in an `if/else` that logs `WARNING` when the method is genuinely absent (future upstream rev) so the no-op cannot recur silently.
- **`vram_buffer` re-added to the cache key.** 0.2.0 removed it on the assumption it was dead, which was correct given the silent no-op. With the wiring fix above it's load-affecting again, so two loader nodes with different `vram_buffer_gb` settings now get distinct cache entries (previously: the second node silently inherited whichever value loaded first).
- **Loader tooltip + README** drop the "DEPRECATED, no-op" framing. Tooltip now describes the actual mechanic (GB of free VRAM, controls layer streaming aggressiveness). The "What does NOT help on Blackwell" section no longer claims `vram_buffer_gb` is dead.

### Added

- **4 new unit tests in `tests/test_runtime_cache_key.py`**: pins the new `enable_vram_management` target (model, not model.pipe), the WARNING log when the method is absent, the cache hit on identical `vram_buffer`, and the cache miss on distinct values. 35 → 39 unit tests.

### Pending before tagging `0.3.0`

- **Tier A5 re-bench.** Now that `vram_buffer_gb` is load-affecting, the README perf table needs an honest measurement of `vram_buffer=4` vs `vram_buffer=12` on a 32 GB card. The RC ships with "perf Δ un-benched" in that table row; the final 0.3.0 tag will replace it with measured numbers. (Cannot reliably re-bench from an automated session — the run takes ~30 min and needs the user's GPU.)

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
