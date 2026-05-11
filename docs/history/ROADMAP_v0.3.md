# Roadmap — v0.3

> **Errata (2026-05-11, post-implementation):** Tier A1's stated root cause is wrong. The roadmap claims `model.pipe` is DiffSynth's stock `WanVideoPipeline` (no `enable_vram_management`), so 0.2.1's call was silently no-op'ing. **A live runtime probe disproved this:** when the runtime was changed to target `model.enable_vram_management(...)` per the roadmap, ComfyUI emitted `WARNING: Model class UniVidIntrinsic lacks enable_vram_management()` — proving the outer class lacks the method. `model.pipe` is in fact an instance of UniVidX's OWN `WanVideoPipeline` subclass at `vendor/UniVidX/src/pipelines/univid_intrinsic.py:24` (method at line 210), which is exactly what 0.2.1 was already targeting. The actual fixes in `0.3.0-rc1` are narrower than this document anticipated: the cache-key omission (real bug — A2 stands) + the documentation accuracy (the "deprecated, no-op" framing was based on this same misdiagnosis). A1's `model.pipe`→`model` change has been reverted; see CHANGELOG `0.3.0-rc1 / Diagnosis correction`.

Self-contained execution plan. Picks up cold from `main` at `4dc9f99` (v0.2.1).

This document was assembled after a second-pass external review identified that two roadmap items shipped in v0.2.0/0.2.1 are mis-prioritized:

1. **`vram_buffer_gb` is mis-described as "deprecated, no-op"** — it's no-op only because we call it on the wrong object. A 1-char fix unlocks working VRAM management. **This is a present correctness issue ahead of the more glamorous FP8/LightX2V work.**
2. **Step-distill LoRA stacking is more involved than "load another LoRA"** — UniVidX already injects 4 modality-specific PEFT adapters with per-sample `adapter_names`; a LightX2V distill LoRA needs to be globally active *in addition to* those, which means a compatibility scanner + merge-into-base path tried first.

Both findings are reflected in the priority order below.

---

## Tier A — Correctness fixes (must-do, day 1)

Tiny code, large cleanup. Do these first.

### A1. Wire `vram_buffer_gb` to the actually-existing API

**Problem:** `src/runtime.py` calls
```python
if hasattr(model, "pipe") and hasattr(model.pipe, "enable_vram_management"):
    model.pipe.enable_vram_management(vram_buffer=float(vram_buffer))
```
`model.pipe` is DiffSynth's `WanVideoPipeline` — that class has no `enable_vram_management` method. So both `hasattr` paths skip the call.

But UniVidX's pipeline classes (`model` itself, not `model.pipe`) DO define one. See [`vendor/UniVidX/src/pipelines/univid_intrinsic.py:210`](vendor/UniVidX/src/pipelines/univid_intrinsic.py) and [`vendor/UniVidX/src/pipelines/univid_alpha.py:203`](vendor/UniVidX/src/pipelines/univid_alpha.py):

```python
def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
    self.vram_management_enabled = True
    ...
    vram_limit = vram_limit - vram_buffer
    if self.text_encoder is not None:
        enable_vram_management(self.text_encoder, ...)  # diffsynth's low-level helper
    ...
```

It wraps text encoder + DiT + VAE through DiffSynth's `enable_vram_management()` low-level helper with an offload policy parameterized by `vram_buffer`.

**Fix in `src/runtime.py`:**
```python
# Was:
if hasattr(model, "pipe") and hasattr(model.pipe, "enable_vram_management"):
    model.pipe.enable_vram_management(vram_buffer=float(vram_buffer))

# Becomes:
if hasattr(model, "enable_vram_management"):
    try:
        model.enable_vram_management(vram_buffer=float(vram_buffer))
        _log.info("VRAM management enabled with vram_buffer=%.1f GB", vram_buffer)
    except TypeError as exc:
        _log.warning(
            "model.enable_vram_management(vram_buffer=...) rejected the kwarg: %s. "
            "VRAM management was NOT applied.", exc,
        )
else:
    _log.warning(
        "Model class %s lacks enable_vram_management(); "
        "vram_buffer_gb has no effect on this build.", type(model).__name__,
    )
```

The `try`/`except TypeError` covers the case where a future upstream rev changes the signature.

### A2. Restore `vram_buffer` to the cache key

**Problem:** `0.2.0` removed `vram_buffer` from the cache key on the assumption it was dead. Now that A1 makes it active, two loader nodes with different `vram_buffer_gb` values would silently share a model configured by whichever loaded first.

**Fix in `src/runtime.py`:**
```python
# Was (current):
cache_key = (variant, ckpt, device, dtype,
             quantize_fp8, bool(compile_dit), bool(prefer_sage_attn))

# Becomes:
cache_key = (variant, ckpt, device, dtype, float(vram_buffer),
             quantize_fp8, bool(compile_dit), bool(prefer_sage_attn))
```

### A3. Update tooltip + README + CHANGELOG

- `nodes/loader.py` widget tooltip: drop the "DEPRECATED" framing; explain `vram_buffer_gb` is the GB of VRAM kept free for activations (passed straight to UniVidX's `enable_vram_management`).
- `README.md` perf table row: replace "no-op" with the actual measured impact at e.g. `vram_buffer=4` vs `vram_buffer=12` on a 32 GB card.
- `README.md` "What does NOT help on Blackwell" section: drop the `vram_buffer_gb` bullet.
- `CHANGELOG.md` `0.3.0` Fixed section: "Wire `vram_buffer_gb` to the actual UniVidX-pipeline `enable_vram_management()`. Was a no-op in 0.1.0–0.2.1 because the runtime called it on `model.pipe` (DiffSynth's `WanVideoPipeline`, no such method) instead of `model` itself."

### A4. Cache-correctness unit test

New test in `tests/test_runtime_cache_key.py`:
- Mock UniVidX class with `enable_vram_management` accepting `vram_buffer`.
- Call `load_model(variant, vram_buffer=4.0)` twice — assert one cache hit.
- Call `load_model(variant, vram_buffer=4.0)` then `load_model(variant, vram_buffer=12.0)` — assert two distinct cache entries (two model loads).
- Mock-call to `enable_vram_management` should be observed with the right `vram_buffer` value.

### A5. Re-bench

Re-run the 17.6 min R2AIN baseline with `vram_buffer=4` (default) vs `vram_buffer=12` on a 32 GB GPU. Update README perf table with the real measured Δ. Now-true that this knob actually does something, the perf table needs an honest update.

**Estimated Tier A time: 2-4 hours code + 30 min benchmark.**

---

## Tier B — Pre-quantized FP8 (the big one)

The current `dtype=fp8_e4m3fn`/`fp8_e5m2` runtime-quantize path hangs in cold-load and destroys the main FP8 benefit (the BF16 cold load is what we wanted to avoid). The right fix is a deeper refactor.

### B1. Audit Kijai's FP8 weights

Inspect [`Kijai/WanVideo_comfy/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors`](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors):
- List of keys + dtypes + shapes (`safetensors.safe_open` then `.keys()` + `.get_slice(...).get_dtype()`).
- Are scale tensors per-tensor or per-channel? Stored as separate `*.scale_weight` keys?
- What's the key prefix vs UniVidX's `WanModel` state dict (e.g. `dit.blocks.0.self_attn.q.weight` vs `model.diffusion_model.blocks.0.self_attn.q.weight`)?
- Which layers are kept in BF16/FP32? Typically norms, biases, time embedding, text projection, patch embed, head.

### B2. State-dict key adapter

Build `src/fp8_loader.py` with a `remap_state_dict(kijai_sd, target_model)` that:
- Strips/adds prefixes to align Kijai keys with UniVidX's `WanModel.state_dict()` keys.
- Handles missing/extra keys with explicit logging.
- Returns the remapped dict + a list of FP8-stored keys + a list of full-precision keys.

### B3. Alternate DiT loader

Bypass the hardcoded six-shard BF16 loop in [`univid_intrinsic.py:447`](vendor/UniVidX/src/pipelines/univid_intrinsic.py) and `univid_alpha.py:425`. The current loop is `WanVideoPipeline.from_pretrained(..., model_configs=[ModelConfig(path=t5), ModelConfig(path=vae), ...])` which auto-discovers the six DiT shards.

New code path in `src/runtime.py`:
1. Construct an empty `WanModel` directly (skip `WanVideoPipeline.from_pretrained` for the DiT only).
2. Load T5 + VAE through normal channels (those stay BF16).
3. Stream-load the FP8 safetensors via `safetensors.safe_open` + iterate → `model.dit.load_state_dict(remapped_dict, strict=False)`.
4. Keep specific layer types (norms, biases, time/text/patch/head) in BF16 by NOT replacing them.
5. Attach the remapped pipeline as `model.pipe`.

### B4. Split the public `dtype` knob

In `nodes/loader.py`:
- Replace `dtype` (current: `bf16 / fp16 / fp8_e4m3fn / fp8_e5m2`) with two clearer fields:
  - `compute_dtype`: `bf16` (default) | `fp16`
  - `dit_weight_mode`: `bf16_shards` (default) | `fp8_prequantized` | `fp8_runtime_experimental`
- The runtime-quantize path becomes `fp8_runtime_experimental` so its known hang risk is explicit in the name.
- For backwards compat: keep `dtype=fp8_e4m3fn`/`fp8_e5m2` as a hidden alias that maps to `dit_weight_mode=fp8_runtime_experimental` + correct sub-variant.

### B5. Keep PEFT LoRAs in BF16

UniVidX's `add_multiple_loras_to_model()` wires four per-modality adapters with dynamic `adapter_names=[...]` switching at forward time (see [`wan_video_dit_intrinsic.py:150`](vendor/UniVidX/src/models/wan_video_dit_intrinsic.py)). Quantizing those LoRA `Linear` layers would (a) save no memory (rank=32 is tiny), (b) interact poorly with PEFT's adapter dispatch, and (c) almost certainly degrade the modality conditioning quality. **Quantize ONLY the base DiT `Linear` weights; LoRA stays BF16.**

### B6. Phase 1 — "memory-safe FP8" (correctness first)

Implement the simplest correct path:
- FP8 base weights stored as `torch.float8_e4m3fn`.
- BF16 LoRA adapters attached via existing UniVidX machinery.
- Forward pass: dequantize FP8 → BF16 just before matmul (slower than `_scaled_mm` but correct).
- Validate per-modality SSIM/PSNR vs BF16 reference on tiny R2AIN/R2PFB workflows. Acceptance threshold: ≥ 0.99 SSIM on each of R/A/I/N (intrinsic) and R/P/F/B (alpha).

### B7. Phase 2 — "fast FP8" (after correctness)

Replace dequantize-on-forward with adapter-aware `torch._scaled_mm`:
- For the base projection: `_scaled_mm(input_fp8, weight_fp8, bias=None, scale_a=in_scale, scale_b=weight_scale, out_dtype=bf16)`.
- For the LoRA residual: standard BF16 `nn.Linear` applied separately, then summed.
- Needs a custom `AdapterAwareFP8Linear` class that subclasses `nn.Linear` (or replaces the LoRA-wrapping logic). Cannot use Kijai's no-LoRA fast path verbatim.

### B8. Validation matrix

Before merging Phase 1:
- Tiny R2AIN: 256×256 × 5 frames × 3 steps. SSIM/PSNR per modality vs BF16.
- Tiny R2PFB: same. Plus alpha-matte edge SSIM specifically.
- Cold load time: must be ≤ BF16 + 30 sec.
- Peak VRAM during sampling: must be ≤ 18 GB on 32 GB card (vs ~32 GB BF16).

Before merging Phase 2:
- Add per-step time. Target: ≥ 30% faster than Phase 1 (ideally ≥ 25% faster than `prefer_sage_attn`-only baseline).
- All Phase 1 quality bars still met.

**Estimated Tier B time: 3-5 days. High confidence on Phase 1 (mechanical), medium on Phase 2 (depends on PEFT integration cleanness).**

---

## Tier C — Step-distill LoRA stacking (after FP8 stable)

LightX2V's [`Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors`](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v) is a Wan2.1-trained distill LoRA targeting 4-step + cfg=1 inference. Ours is the harder case: it needs to be active GLOBALLY in addition to UniVidX's 4 per-modality adapters.

### C1. Compatibility scan

`examples/_scan_lightx2v_compat.py` (new):
- Load the LightX2V LoRA via safetensors.
- List LoRA keys → infer target_modules.
- Compare to UniVidX's `target_modules="self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2"`.
- Flag mismatches (e.g. LightX2V also targets `self_attn.norm_q`, modality difference).
- Verify rank (32 in our case — same as UniVidX's per-modality adapters).
- Output a yes/no compatibility report + a list of layers that would be touched by both.

### C2. Try merge-into-base FIRST

Safest path: fold LightX2V's LoRA INTO the base Wan2.1 weights *before* UniVidX's per-modality adapters are attached. This avoids stacking two PEFT adapters on the same Linears.

```python
# Pseudocode in src/runtime.py:
base_state_dict = load_wan21_state_dict()
if step_distill_lora == "lightx2v":
    distill_lora = load_safetensors("...lightx2v...rank32.safetensors")
    base_state_dict = merge_lora_into_base(base_state_dict, distill_lora,
                                            scale=distill_strength)
# Then UniVidX construction uses the modified base_state_dict.
```

`merge_lora_into_base` is `weight + scale * (lora_B @ lora_A)` for each `(weight, lora_A, lora_B)` triple. Standard PEFT-merge math.

### C3. Loader controls (only after C2 works)

In `nodes/loader.py`:
- `step_distill_lora`: enum `none` (default) | `lightx2v` | `custom`
- `step_distill_strength`: float, default 1.0, range 0.0-2.0
- Tooltip warning: "Trained for 4-6 step inference + cfg_scale=1. Quality unverified on UniVidX's intrinsic/alpha modalities."

If `custom`, expose `step_distill_path` (string) for users to point at their own distill LoRA.

### C4. Validation per-modality

Tiny R2AIN + R2PFB at fixed seed, comparing 20-step BF16 baseline vs 4/6/8-step distill at each `distill_strength` ∈ {0.5, 0.75, 1.0}:
- **RGB (R2AIN with text-only fallback)**: aesthetic comparison, prompt adherence
- **Albedo**: color stability across frames (no flicker), no oversaturation
- **Irradiance**: smooth gradients (no banding), candle-highlight preservation
- **Normal**: facial geometry sharpness, no spurious gradients
- **Alpha matte (R2PFB)**: edge IoU vs baseline, no halo
- **Foreground/Background**: composite-back PSNR vs BF16

Record observed per-modality degradation in `examples/test_matrix/REPORT.md`.

### C5. Document quality-vs-speed trade-offs honestly

Same format as the current PRODUCTION/PREVIEW preset table. Add a "DISTILLED PREVIEW" row:
```
DISTILLED PREVIEW: prefer_sage_attn=True + step_distill_lora=lightx2v +
                   num_inference_steps=4 + cfg_scale=1.0
                   wall: ~?? min · per-modality quality: see REPORT.md
```

### C6. ONLY THEN test FP8 + step-distill together

Combination test is last because each ingredient adds quality risk independently. Want to know the marginal effect of EACH change isolated before stacking.

**Estimated Tier C time: 3-4 days. High risk on quality validation (must be honest about per-modality degradation), low risk on code.**

---

## Tier D — Quality of life (interleaved)

### D1. Thread-safe `unividx_cwd()`

`os.chdir()` is process-global. Concurrent ComfyUI queue runs that both call this race on the cwd. Either:
- Add a `threading.Lock` around the chdir pair (simplest, serializes UniVidX runs but they're already GPU-bound serially).
- Patch upstream UniVidX to accept absolute paths (correct fix, larger surgery).

Pick #1 for v0.3.

### D2. Windows CI

`.github/workflows/smoke.yml` currently `runs-on: ubuntu-latest`. Add a `windows-latest` matrix entry that runs `python -m compileall .` + `pytest tests/` (excluding GPU tests). Catches the kind of path-handling regressions our three Windows-specific patches were written to avoid.

### D3. Modes validator

`examples/_validate_modes.py` (new): assert every mode string in `README.md`'s mode reference appears in `INTRINSIC_MODES` / `ALPHA_MODES` from `src/modes.py`. Wire into `smoke.yml` so README/code drift gets caught.

### D4. PERFORMANCE.md split

Move the Optimization knobs section + the install-sage steps into `docs/PERFORMANCE.md`. Keep only the two-preset table + a one-liner "see PERFORMANCE.md for the full perf knob set" in the main README. Doc reviewer's recommendation; defers the deep dive without hiding it.

### D5. Examples gallery

Embed a 5-10 second mp4 (or GIF) per modality (RGB, Albedo, Irradiance, Normal, Alpha matte, Foreground, Background composite) in the README. Use the existing LTX clip's `vendor/UniVidX/assets/` outputs or generate fresh from a curated input.

---

## Tier E — Deferred to 0.4+

- **SageAttention 3** sm_120-native kernels when the upstream issue [thu-ml/SageAttention#291](https://github.com/thu-ml/SageAttention/issues/291) lands real Blackwell support (currently sm89 fallback only).
- **Flash Attention 4** integration when DiffSynth detects it via the right module name (`flash_attn.cute`).
- **CMSA optimization** (multi-week, requires upstream fork to make cross-modal attention more efficient than the current quadratic-in-modality-count cost).

---

## Suggested execution order for v0.3

| Day | Tier | Task |
|---|---|---|
| 1 | A | Fix vram_buffer wiring + cache key + tooltip + README + test |
| 1 | A5 | Re-bench vram_buffer=4 vs 12, update README perf table |
| 2-3 | B1-B3 | Audit Kijai FP8 + state-dict adapter + alternate DiT loader (Phase 1 plumbing) |
| 4 | B4-B5 | Loader knob split + LoRA-stays-BF16 wiring |
| 5 | B6 + B8 | Phase 1 dequantize-on-forward + validation matrix |
| 6-7 | B7 + B8 | Phase 2 `_scaled_mm` adapter-aware Linear + perf benchmark |
| 8 | C1-C2 | LightX2V compat scanner + merge-into-base proof-of-concept |
| 9 | C3-C5 | Loader controls + per-modality validation + REPORT.md update |
| 10 | C6 | FP8 + step-distill combination test |
| Interleaved | D1-D3 | Thread-safe cwd + Windows CI + modes validator |
| At cut | D4-D5 | PERFORMANCE.md split + gallery + 0.3.0 release notes |

**Total: ~10 working days of focused work for a polished v0.3.**

---

## Risk register

| Risk | Tier | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| Kijai FP8 weight key conventions don't match UniVidX's WanModel | B2 | Medium | Blocks Tier B | Build a fuzzy remapper, test fast before going deeper |
| Phase 2 `_scaled_mm` adapter-aware Linear interacts badly with PEFT | B7 | Medium | Phase 2 fallback to Phase 1 | Keep Phase 1 path as the production default; Phase 2 is opt-in |
| LightX2V LoRA degrades modality decomposition more than expected | C4 | High | Demote step-distill to "experimental" tier in docs | Ship clear per-modality quality numbers, let users decide |
| Thread-safe cwd serialization tanks throughput for users running parallel queue | D1 | Low | Single-user impact (most users don't parallelize anyway) | Document the serialization in README; add a benchmark note |
| Windows CI matrix discovers a real path-handling bug we missed | D2 | Medium | Win-only fix in 0.3.x patch release | Acceptable — finding it is the point |

---

## When this is done

A new `CHANGELOG.md` `## 0.3.0` section should claim:
- `vram_buffer_gb` actually works (cite the perf delta from A5)
- `dit_weight_mode = fp8_prequantized` is the recommended FP8 path; runtime-quantize moved to `fp8_runtime_experimental`
- Per-modality quality numbers documented for FP8 and for step-distill at 4/6/8 steps
- Thread-safe `unividx_cwd()` (with a brief note about throughput trade-off)
- Windows CI green
- README split into main + `docs/PERFORMANCE.md`

Cut the release, bump `pyproject.toml` to `0.3.0`, push tag.

---

## Bonus: review-traceability

Findings from the second-pass external review (against `4c1f282`) addressed by this plan:

- **P0 vram_buffer cache/doc inconsistency** → A1 + A2 + A3 + A4 + A5
- **P0 FP8 runtime-quantize wrong production path** → B1-B8
- **P1 Step-distill needs compatibility scan** → C1-C6
- **P1 Doc drift on install.py** → already fixed in 0.2.1
- **P1 Doc drift on cache key** → already fixed in 0.2.1
- **P2 `__init__.py` swallows ImportError** → already fixed in 0.2.1
- **P2 FP8/Sage tests light** → covered by A4 + B8 + C4 (per-modality validation tests)

Reviewer's suggested priority order — exactly what Tiers A→B→C above implement.
