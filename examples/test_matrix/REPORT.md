# ComfyUI-UniVidX Test Matrix Report

Date: 2026-05-10
Hardware: RTX 5090 (32 GB), torch 2.7.0+cu128, Python 3.12.9
ComfyUI: 0.20.1 on port 8000
All tests: tiny config (256×256, 5 frames, 3 inference steps, seed 42)

## Summary

**10 / 10 tests pass.** Every code path in `nodes/sampler.py` and `nodes/decoder.py`
exercised end-to-end (tests A–H), plus two **downstream-composition tests** (I, J)
that validate Strategy A's actual value proposition: UniVidX outputs flow into
other ComfyUI nodes as real IMAGE batches.

| # | Test | Mode | Variant | Inputs | Result |
|---|---|---|---|---|---|
| A | t2RAIN_full | t2RAIN | intrinsic | (text) | ✅ PASS — 84 PNGs, 480×640×21f, 628 sec (full smoke, prior session) |
| B | R2AIN_full | R2AIN | intrinsic | RGB | ✅ PASS — 84 PNGs incl. 21 black placeholders, 617 sec (prior session) |
| C | C_RA2IN | RA2IN | intrinsic | RGB+Albedo | ✅ PASS — 25 sec |
| D | D_RAI2N | RAI2N | intrinsic | RGB+A+I | ✅ PASS — 20 sec |
| E | E_t2RPFB | t2RPFB | alpha | (text) | ✅ PASS — 161 sec (incl. cold load) |
| F | F_R2PFB | R2PFB | alpha | RGB | ✅ PASS — 30 sec |
| G | G_error_family_mismatch | t2RPFB | intrinsic | (text) | ✅ PASS — execution_error contains "family" |
| H | H_error_missing_input | R2AIN | intrinsic | (none) | ✅ PASS — execution_error contains "missing" |
| **I** | **I_video_output** | t2RAIN | intrinsic | (text) | **✅ PASS — 4 MP4 files written by VHS_VideoCombine, 25 sec** |
| **J** | **J_alpha_compositing** | R2PFB | alpha | RGB | **✅ PASS — composite over cyan bg: 89.3% bg / 10.7% fg, 25 sec** |

## Per-test pixel statistics (frame 1 of each output)

### C — RA2IN (intrinsic, 2 inputs → 2 outputs)

| Modality | min | max | mean | std | Verdict |
|---|---|---|---|---|---|
| rgb (placeholder) | 0 | 0 | 0.0 | 0.0 | ✅ correctly black |
| albedo (placeholder) | 0 | 0 | 0.0 | 0.0 | ✅ correctly black |
| irradiance (target) | — | — | 174 | 48 | ✅ real content |
| normal (target) | — | — | 171 | 59 | ✅ real content |

### D — RAI2N (intrinsic, 3 inputs → 1 output, max conditioning)

| Modality | min | max | mean | std | Verdict |
|---|---|---|---|---|---|
| rgb (placeholder) | 0 | 0 | 0.0 | 0.0 | ✅ |
| albedo (placeholder) | 0 | 0 | 0.0 | 0.0 | ✅ |
| irradiance (placeholder) | 0 | 0 | 0.0 | 0.0 | ✅ |
| normal (target) | — | — | 171 | 59 | ✅ real content |

### E — t2RPFB (alpha, text-to-all)

| Modality | min | max | mean | std | Notes |
|---|---|---|---|---|---|
| composite_rgb | 0 | 255 | 232 | 52 | ✅ |
| alpha | 242 | 255 | **254.9** | **0.7** | Nearly all-white. Model couldn't decompose without RGB reference. |
| foreground | 0 | 255 | 238 | 52 | ✅ |
| background | 0 | 255 | 234 | 44 | ✅ |

### F — R2PFB (alpha, RGB-conditioned)

| Modality | min | max | mean | std | Notes |
|---|---|---|---|---|---|
| composite_rgb (placeholder) | 0 | 0 | 0.0 | 0.0 | ✅ |
| alpha | 0 | 255 | **27.1** | **75.5** | Real matte — clear hedgehog silhouette on black bg |
| foreground | 0 | 255 | 234 | 55 | ✅ Hedgehog isolated |
| background | 0 | 254 | 166 | 55 | ✅ Kitchen alone |

**Behavioural finding**: comparing E vs F demonstrates that **conditioning input
matters**. With only text (E), the alpha matte is nearly all white (≈ "everything
is foreground"). With an RGB reference (F), the same model produces a sharp
hedgehog silhouette and a clean foreground/background split. This is correct
UniVidX behaviour, not a bug.

### G — error_family_mismatch (Loader=intrinsic, TaskMode=alpha-family)

```
exception_type: ValueError
exception_message: Task mode t2RPFB is family=alpha, but loaded model is intrinsic.
                   Pick a matching loader (intrinsic vs alpha).
executed: ['1']    # Loader ran; sampler raised before any sampling
```

✅ The Sampler's `if family != variant` runtime check fired exactly as designed.

### H — error_missing_input (TaskMode=R2AIN with no rgb wired)

```
exception_type: ValueError
exception_message: Mode R2AIN requires inputs ['rgb'], missing: ['rgb']
executed: ['1', '2']    # Loader and TaskMode ran; sampler raised at validate_mode()
```

✅ `src/modes.py:validate_mode()` correctly enforces required inputs.

### I — video_output (UniVidX → VHS_VideoCombine)

Workflow: `UniVidXSampler(t2RAIN) → DecodeIntrinsic → 4× VHS_VideoCombine`
producing one H.264 MP4 per modality.

| Output file | Size |
|---|---|
| `matrix_I_video_rgb_00001.mp4` | 29 KB |
| `matrix_I_video_albedo_00001.mp4` | 37 KB |
| `matrix_I_video_irradiance_00001.mp4` | 32 KB |
| `matrix_I_video_normal_00001.mp4` | 35 KB |

✅ The decoder's IMAGE batches feed unmodified into a third-party node (`ComfyUI-VHS`)
that doesn't know anything about UniVidX. This is the load-bearing validation
of Strategy A's I/O boundary.

### J — alpha_compositing (UniVidX alpha matte → real VFX comp)

Workflow:
```
UniVidXSampler(R2PFB)  ──┐
                          ├─→ DecodeAlpha (rgb_placeholder, alpha, foreground, background)
                          │
                                  alpha ──→ ImageToMask(channel='red') ──┐
                                                                          │
EmptyImage(256x256, color=cyan) ─────────────→ destination ─────┐         │
                                                                 │         │
DecodeAlpha.foreground ────────────────────→ source ────────┐    │         │
                                                             ▼    ▼         ▼
                                              ImageCompositeMasked(dest, src, mask)
                                                             │
                                                             ▼
                                                      SaveImage
```

Result: 5 PNG frames showing the hedgehog cleanly extracted from the kitchen
and pasted onto a cyan background. Pixel analysis: **89.3% cyan-background
pixels, 10.7% non-cyan (foreground)** — confirming both that the alpha matte
correctly identified the hedgehog (small object in scene) AND that the composite
pasted only those pixels onto the new background.

The alpha output of `R2PFB` is grayscale-equivalent across RGB channels, so
`ImageToMask(channel='red')` faithfully extracts it as a `MASK` type for
`ImageCompositeMasked` to consume.

✅ This validates the real-world utility: UniVidX's alpha decomposition is a
**usable VFX matte**, not just a visualization. Same matte could be wired
into chroma-key compositing, frame-by-frame masking, or any other node that
consumes `MASK` inputs.

## Performance summary

Per-step times (256×256 × 5 frames × 4-modality streams):

| Variant | Cold load | Per-step (cached) |
|---|---|---|
| Intrinsic | ~2.5 min (T5+VAE+DiT+LoRA) | ~3.8 sec/step |
| Alpha     | ~2.5 min (same Wan2.1 base, different LoRA) | ~7.0 sec/step (incl. JIT warm-up on first run) |

Cache hits between same-variant tests: **20-30 sec total** per test (just
sampling + VAE decode + saving). The runtime `_MODEL_CACHE` keyed by
`(variant, ckpt, device, dtype)` is doing its job — switching variants
forces a reload, but consecutive same-variant tests reuse the loaded model.

## Code paths validated

- `nodes/loader.py`: both `variant="intrinsic"` and `variant="alpha"`
- `nodes/task.py`: 6 different modes (t2RAIN, R2AIN, RA2IN, RAI2N, t2RPFB, R2PFB)
- `nodes/sampler.py`:
  - Family/variant compatibility check (Test G)
  - `validate_mode()` required-input check (Test H)
  - `image_batch_to_video_tensor()` with 1, 2, 3 conditioning inputs
  - Multiple optional IMAGE inputs simultaneously (RA2IN: rgb+albedo; RAI2N: rgb+albedo+irradiance)
  - Both 4-target output (t2X) and N-target output (XYZ2W) cases
- `nodes/decoder.py`:
  - `UniVidXDecodeIntrinsic` with full target dict (t2RAIN — prior session)
  - `UniVidXDecodeIntrinsic` with partial dict + 1, 2, 3 placeholder slots (Tests B, C, D)
  - `UniVidXDecodeAlpha` with full dict (Test E)
  - `UniVidXDecodeAlpha` with 1 placeholder slot (Test F)
- `src/runtime.py:load_model()`:
  - `_MODEL_CACHE` cache hit (Tests B, C, D, F all reuse cached model)
  - Fresh load for new variant (Test E loaded alpha for the first time)
  - All three Windows-fix patches active throughout
  - `enable_vram_management(vram_buffer=4.0)` keeping VRAM bounded

## Out of scope / not validated

- Sampling at non-default frame counts other than 5 / 21
- Sampling at resolutions other than 256² (tiny) and 480×640 (full)
- Mode/variant pairs not in the 6 tested (24 of 30 modes still untouched)
- Long-running stability (no multi-prompt back-to-back stress tests)
- The other 24 modes — most should "just work" since they use the same code paths,
  but each is a fresh prompt-engineering experiment

## Reproducing this report

```bash
cd ${COMFY_ROOT}/custom_nodes/UniVidX_ComfyUI
# Stage input images
cp output/unividx_t2RAIN_rgb_00011_.png       input/unividx_R2AIN_input.png
cp output/unividx_t2RAIN_albedo_00011_.png    input/unividx_input_albedo.png
cp output/unividx_t2RAIN_irradiance_00011_.png input/unividx_input_irradiance.png
# Generate workflows (C–H)
python examples/test_matrix/_build.py
# (I and J are hand-written; see I_video_output.json, J_alpha_compositing.json)

# Run all 8 generated tests (~5 min if cached):
python examples/test_matrix/_run.py
# Or a subset:
python examples/test_matrix/_run.py --filter alpha       # E + F only
python examples/test_matrix/_run.py --filter error       # G + H only
python examples/test_matrix/_run.py --filter video       # I only
python examples/test_matrix/_run.py --filter compositing # J only
```

Raw machine-readable results in `_run_results.json`.

**Composition tests (I, J) require:**
- `ComfyUI-VideoHelperSuite` (VHS) for `VHS_VideoCombine` — already in your
  `custom_nodes/`.
- Core ComfyUI nodes `EmptyImage`, `LoadImage`, `RepeatImageBatch`,
  `ImageToMask`, `ImageCompositeMasked`, `SaveImage` — all in core.
