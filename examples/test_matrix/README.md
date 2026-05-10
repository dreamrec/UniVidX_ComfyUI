# Test Matrix

Eight focused workflows that exercise every code path in `nodes/sampler.py` and `nodes/decoder.py`, plus two downstream-composition tests that prove Strategy A's value at the I/O boundary.

All matrix workflows use the **tiny config** (256×256, 5 frames, 3 inference steps, seed 42) so each one runs in 25-30 sec. This keeps the matrix fast enough to use as a regression suite — the full set finishes in under 5 min on an RTX 5090 (excluding cold model load).

## Files

| File | Purpose |
|---|---|
| [`_build.py`](_build.py) | Regenerates the eight `*.json` workflows from in-source templates. Run after editing node schemas. |
| [`_run.py`](_run.py) | Submits each workflow to a running ComfyUI on `localhost:8000`, asserts the expected outcome (success / specific error), writes `_run_results.json`. |
| [`_run_results.json`](_run_results.json) | Latest run's machine-readable results — wall time, output filenames, error text where applicable. |
| [`REPORT.md`](REPORT.md) | Latest run's human-readable report including per-modality pixel statistics. |
| `C_RA2IN.json` … `J_alpha_compositing.json` | The eight workflow JSONs (UI format). |

## Running

```bash
# regenerate JSONs (only after editing node schemas or _build.py templates)
python examples/test_matrix/_build.py

# run all eight + assert
python examples/test_matrix/_run.py

# run a subset
python examples/test_matrix/_run.py --filter alpha
python examples/test_matrix/_run.py --filter error

# require ComfyUI running on a non-default port
python examples/test_matrix/_run.py --host http://localhost:8188
```

## What each test validates

| # | File | Mode | Variant | Inputs | What it proves |
|---|---|---|---|---|---|
| C | `C_RA2IN.json` | `RA2IN` | intrinsic | RGB + Albedo | Multi-input intrinsic conditioning works; non-target slots come back as black placeholders. |
| D | `D_RAI2N.json` | `RAI2N` | intrinsic | RGB + A + I | Maximum-conditioning case — 3 IMAGE inputs, 1 generated target. |
| E | `E_t2RPFB.json` | `t2RPFB` | alpha | (text) | Alpha variant cold-loads correctly; `DecodeAlpha` splays the result into the right four IMAGE outputs. |
| F | `F_R2PFB.json` | `R2PFB` | alpha | RGB | Sharp alpha matte from RGB conditioning — the production-quality alpha path. |
| G | `G_error_family_mismatch.json` | `t2RPFB` | intrinsic | (text) | Sampler rejects a task whose family doesn't match the loaded model variant. Error message contains `"family"`. |
| H | `H_error_missing_input.json` | `R2AIN` | intrinsic | (none) | `validate_mode()` rejects a mode whose required inputs aren't wired. Error message contains `"missing"`. |
| **I** | **`I_video_output.json`** | `t2RAIN` | intrinsic | (text) | UniVidX outputs flow into `VHS_VideoCombine` and produce four valid MP4 files. **Strategy A boundary test.** |
| **J** | **`J_alpha_compositing.json`** | `R2PFB` | alpha | RGB | UniVidX alpha outputs flow into `ImageToMask` + `ImageCompositeMasked` and composite cleanly onto a synthetic background. **Strategy A boundary test.** |

Tests **I** and **J** are the load-bearing claims of Strategy A — they show that UniVidX outputs are real ComfyUI `IMAGE` batches that compose with arbitrary downstream nodes that know nothing about UniVidX.

## Adding a new test

1. Add a template entry to `_build.py` — pick a unique single-letter prefix, set the mode and variant, list the IMAGE inputs to wire in (the builder synthesises a `LoadImage` + `RepeatImageBatch` chain per input).
2. Run `python examples/test_matrix/_build.py` to materialise the JSON.
3. Run `python examples/test_matrix/_run.py --filter <prefix>` to verify it passes.
4. Re-run the full matrix and refresh `REPORT.md` with `python examples/test_matrix/_run.py && python -c "from _run import write_report; write_report()"` (or just hand-edit).

## Companion node packs

Test **I** requires [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for `VHS_VideoCombine`. The runner skips test **I** with a clear message if the node isn't installed; everything else uses ComfyUI core nodes only.
