# Roadmap

Living plan for ongoing work. Replaces the earlier numbered
`ROADMAP_v0.3.md` (archived under
[`docs/history/ROADMAP_v0.3.md`](docs/history/ROADMAP_v0.3.md)),
which became outdated after Tier B's mid-flight pivot from a
file-based FP8 design to runtime quantization.

## Status — 2026-05-11 (post-0.4.0-rc1)

What's shipped on `origin/main`:

| Layer | Status | Notes |
|---|---|---|
| Tier A (correctness — `vram_buffer_gb`) | **shipped 0.3.0** | +65% wall going 4→12 measured |
| Tier B Phase 1 (FP8 runtime quantize) | **shipped 0.4.0-rc1** | -13% wall, -50% DiT VRAM, PSNR ≥30 dB per modality at production |
| Tier B7 (Phase 2 `_scaled_mm`) | **deferred** | Phase 1 already delivered the speedup; only worth doing if profiling shows dequant is the bottleneck |
| Tier B file-based FP8 path | **dormant — auto-enables when a Kijai `_scaled` Wan2.1 file lands upstream** | Code retained at `src/fp8_loader.py:load_fp8_state_dict_into`; runtime resolver looks for `Wan2_1-T2V-14B_fp8_e4m3fn_scaled.safetensors` |

## Status — 0.4.0 final (2026-05-11)

All six close-out validation matrix conditions measured. See CHANGELOG
`0.4.0` section for the headline + the full table. Tagged + pushed.

Key findings that update prior docs:

- [x] **BF16+sage = 14.48 min** — validates the original "~14.5 min"
      claim; the README was correct, not stale.
- [x] **`prefer_sage_attn=True` is +33% wall on R2AIN_video** — the
      0.2.0 perf-table's "−18%" was on a different config. **Sage
      now hurts on R2AIN-style workloads.**
- [x] **FP8 + sage = 11.75 min** (sage adds +25% on top of FP8 baseline).
- [x] **FP8 + compile_dit = 11.65 min** (compile adds +24% on top of
      FP8 baseline). Graph-captures cleanly on FP8Linear; doesn't
      help.
- [x] **FP8 alpha variant = 12.36 min** — works, slightly slower
      than intrinsic.
- [x] **FP8 PREVIEW + sage = 6.20 min** (cold-load dominates short
      sample run).
- [x] **FP8 text-only tiny = 4.66 min** — smoke-test config.

The 0.4.0 release notes now lead with "just set
`dit_weight_mode=fp8_prequantized`, leave everything else default"
because that's the empirically best configuration on this hardware.

## Next milestone candidates — `0.5.0`

In priority order based on user value × implementation cost:

### ~~Tier C — Step-distill LoRA stacking (LightX2V)~~ **SHIPPED in 0.5.0**

See CHANGELOG `0.5.0` for the full measurement set + framing.
Headline outcome: **`FP8 + distill = 4.59 min wall, 3.15× speedup vs old
PRODUCTION**. PSNR ~22-26 dB vs BF16 (visibly different decompositions,
but plausible content). Ships as **fast preview / iteration mode**,
NOT as a production replacement for the FP8 baseline.

Original Tier C analysis below for archive purposes:

LightX2V's
[`Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors`](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v)
(note: rank 64, not 32 as originally claimed) is a Wan2.1-trained
distill LoRA targeting 4-step + cfg=1 inference. Our complication:
it must be active GLOBALLY on top of UniVidX's 4 per-modality LoRA
adapters.

Approach (per `docs/history/ROADMAP_v0.3.md` § Tier C, still valid):
1. Compatibility scanner (`examples/_scan_lightx2v_compat.py`):
   confirm target_modules overlap, rank parity (32 ≡ 32), no
   `norm_q/k` collisions.
2. **Merge-into-base** first try — fold LightX2V into the Wan2.1
   weights BEFORE UniVidX's adapters are wired, avoiding stacked
   PEFT adapters. Standard `W' = W + α·B·A` math.
3. Loader controls: new `step_distill_lora` enum widget
   (`none / lightx2v / custom`) + `step_distill_strength` float.
4. **Per-modality quality validation matrix** — distilled outputs
   compared to BF16 baseline at 4/6/8 steps, each `strength ∈
   {0.5, 0.75, 1.0}`, for R/A/I/N and R/P/F/B. SSIM ≥ 0.99 RGB,
   ≥ 0.97 synthetic.
5. **Combine with FP8** only after both individually pass quality.

Open question: FP8 base may shift activation distribution in ways
LightX2V wasn't trained against. **The FP8 + step-distill combination
is the highest-quality-risk pairing in the project.**

### Tier D1 — Thread-safe `unividx_cwd()`

<1 day; low risk, mechanical.

`os.chdir()` in `unividx_cwd()` is process-global. Concurrent
ComfyUI queue runs that both call this race on cwd. Either:
- Add `threading.Lock` around the chdir pair — serializes UniVidX
  runs but they're already GPU-bound serially. **Recommended for
  0.5.x.**
- Or patch upstream UniVidX to accept absolute paths — correct but
  large surgery; deferred indefinitely.

### Tier D2 — Windows CI

<1 day; medium value.

`.github/workflows/smoke.yml` currently `runs-on: ubuntu-latest`.
Add `windows-latest` matrix entry running `python -m compileall .`
+ `pytest tests/` (excluding GPU tests). Catches the kind of
path-handling regressions our three Windows-specific patches were
written against (Win backslashes in JSON, paging-file commit on
mmap, junction-vs-symlink).

### Tier D3 — Modes validator

<1 day; low value.

`examples/_validate_modes.py` (new) asserts every mode string in
README's mode reference appears in `INTRINSIC_MODES` / `ALPHA_MODES`
from `src/modes.py`. Wire into `smoke.yml` so README/code drift
gets caught.

### Tier D4 — `docs/PERFORMANCE.md` split

<1 day; useful as more perf knobs land.

Move the Optimization knobs section + install-sage steps from
README into `docs/PERFORMANCE.md`. Keep only the two-preset table +
a "see PERFORMANCE.md" one-liner in the main README. The FP8
addition makes the perf section dense enough to warrant its own
file.

### Tier D5 — Examples gallery

Per-modality 5-10 second mp4 (or GIF) embedded in the README:
RGB, Albedo, Irradiance, Normal, Alpha matte, Foreground, Background
composite. Use the existing LTX clip outputs or generate fresh.

### Tier B7 — Phase 2 `_scaled_mm` (deferred)

3-5 days; medium risk; **lower priority** than originally planned.

Phase 1's runtime quantize accidentally delivered the per-step
speedup (-13% wall) because the BF16 baseline pays heavy
`enable_vram_management` streaming. The motivating gap for Phase 2
(per-step throughput) is therefore much smaller than the roadmap
expected.

Phase 2 would replace `FP8Linear.forward`'s dequant-then-matmul
with `torch._scaled_mm(input_fp8, weight_fp8, scale_a=in_scale,
scale_b=weight_scale, out_dtype=bf16)` for the base projection,
plus standard BF16 LoRA residual on the side. Needs a custom
adapter-aware Linear that PEFT can dispatch through cleanly. Open
to revisiting if a profile shows the FP8→BF16 cast is the
bottleneck.

### Tier E — Deferred to `0.5+` or later

- **SageAttention 3 sm_120-native kernels** when [thu-ml/SageAttention#291](https://github.com/thu-ml/SageAttention/issues/291) lands real Blackwell support.
- **Flash Attention 4** when DiffSynth detects it via the right module name (`flash_attn.cute`).
- **CMSA optimization** — multi-week, requires upstream fork to
  make cross-modal attention more efficient than its current
  quadratic-in-modality-count cost.

## Risk register

| Risk | Tier | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| `torch.compile` doesn't graph-capture FP8Linear cleanly | 0.4.0 validation | Medium | Falls through to eager, lose the compile_dit speedup but FP8 still works | Validate as part of 0.4.0 close-out matrix |
| FP8 + sage_attn numerical issues | 0.4.0 validation | Low | Quality regression in attention pathways | Validate quality + perf both |
| LightX2V step-distill produces visible per-modality degradation on Albedo/Irradiance/Normal | C4 | High | Demote step-distill to "experimental" tier in docs | Ship clear per-modality quality numbers, let users decide per workload |
| Thread-safe cwd serialization tanks throughput for parallel queue users | D1 | Low | Single-user impact (most users don't parallelize anyway) | Document the serialization in README |

## When this roadmap is done

- `0.4.0` final tagged with full perf-table verified across
  FP8 × {sage, compile, alpha, PREVIEW, text-only}
- `0.5.0` ships with at minimum Tier C (step-distill) or Tier D
  cleanup batch, depending on which the user prioritizes
- A new ROADMAP entry replaces this one when 0.5+ is in scope
