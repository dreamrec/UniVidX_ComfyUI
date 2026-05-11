"""Tests for chunked_clip_sampler.plan_chunks — the deterministic
chunk-planning math.

The driver itself (queueing workflows, polling /history, stitching
PNGs into MP4s) is best validated against a real ComfyUI instance,
not in unit tests. But the chunk plan is pure arithmetic — invariants
are easy to assert and worth pinning."""
from __future__ import annotations

import os
import sys

import pytest

HERE = __file__
_REPO = os.path.abspath(os.path.join(os.path.dirname(HERE), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Import the chunk planner directly; avoid importing the rest of the
# script (which would try to pull in numpy + PIL).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "chunked_clip_sampler",
    os.path.join(_REPO, "examples", "chunked_clip_sampler.py"),
)
_mod = importlib.util.module_from_spec(_spec)
# Stub the heavy imports the file's __init__ would pull in.
import numpy as _np  # type: ignore
sys.modules.setdefault("numpy", _np)
try:
    _spec.loader.exec_module(_mod)
except ImportError:
    # PIL or imageio missing in test env — load just the function we need
    # via direct AST exec. Simpler workaround: re-implement here and pin
    # both to match.
    raise


def test_plan_covers_all_frames():
    """Every frame in [0, total) must be covered by at least one chunk."""
    plan = _mod.plan_chunks(total_frames=1440, chunk_size=21, overlap=5)
    covered = set()
    for start, end in plan:
        covered.update(range(start, end))
    assert covered == set(range(1440))


def test_plan_chunks_uniform_size():
    """Every chunk must be exactly chunk_size frames."""
    plan = _mod.plan_chunks(total_frames=1440, chunk_size=21, overlap=5)
    sizes = {end - start for start, end in plan}
    assert sizes == {21}


def test_plan_overlaps_match_overlap_param_except_last():
    """Consecutive chunks should overlap by exactly `overlap` frames,
    except possibly the final pair (which may overlap more to anchor
    to total_frames)."""
    plan = _mod.plan_chunks(total_frames=1440, chunk_size=21, overlap=5)
    for i in range(len(plan) - 2):
        prev_end = plan[i][1]
        next_start = plan[i + 1][0]
        assert prev_end - next_start == 5, (
            f"chunk {i}→{i+1} overlap is {prev_end - next_start}, expected 5"
        )


def test_plan_anchors_final_chunk_to_total():
    """The final chunk must end exactly at total_frames (no frames
    dropped at the tail)."""
    plan = _mod.plan_chunks(total_frames=1440, chunk_size=21, overlap=5)
    assert plan[-1][1] == 1440


def test_plan_first_chunk_starts_at_zero():
    plan = _mod.plan_chunks(total_frames=1440, chunk_size=21, overlap=5)
    assert plan[0][0] == 0


def test_plan_with_no_overlap_is_strided():
    """overlap=0 means stride==chunk_size; chunks tile non-overlapping."""
    plan = _mod.plan_chunks(total_frames=63, chunk_size=21, overlap=0)
    assert plan == [(0, 21), (21, 42), (42, 63)]


def test_plan_with_exact_multiple_no_anchor_chunk():
    """When total_frames is exactly covered by stride-aligned chunks,
    no extra anchor chunk is appended."""
    # 21-frame chunks with overlap=5 → stride=16. total=37 covers
    # chunks (0,21) and (16,37) exactly with no tail.
    plan = _mod.plan_chunks(total_frames=37, chunk_size=21, overlap=5)
    assert plan == [(0, 21), (16, 37)]


def test_plan_anchor_chunk_overlaps_more_when_tail_short():
    """When the last regular stride doesn't reach total_frames, the
    anchor chunk may overlap the previous chunk by more than `overlap`
    frames in order to end at total_frames."""
    # stride=16. total=40: chunk0=(0,21), next stride goes to (16,37).
    # 37 < 40 so anchor chunk (40-21=19, 40) overlaps (16,37) by 18 frames.
    plan = _mod.plan_chunks(total_frames=40, chunk_size=21, overlap=5)
    assert plan[-1] == (19, 40)
    prev_end = plan[-2][1]
    next_start = plan[-1][0]
    assert prev_end - next_start == 18  # > 5 (the configured overlap)


def test_plan_rejects_invalid_overlap():
    """overlap >= chunk_size is nonsensical and must raise."""
    with pytest.raises(ValueError):
        _mod.plan_chunks(total_frames=100, chunk_size=21, overlap=21)
    with pytest.raises(ValueError):
        _mod.plan_chunks(total_frames=100, chunk_size=21, overlap=-1)


def test_plan_rejects_total_below_chunk_size():
    """A source shorter than chunk_size has no meaningful plan."""
    with pytest.raises(ValueError):
        _mod.plan_chunks(total_frames=10, chunk_size=21, overlap=5)


def test_plan_exact_chunk_size():
    """total_frames == chunk_size produces a single chunk."""
    plan = _mod.plan_chunks(total_frames=21, chunk_size=21, overlap=5)
    assert plan == [(0, 21)]


def test_plan_chunk_count_for_one_minute_clip():
    """Quick sanity: 1 min @ 24 fps = 1440 frames, chunk=21 overlap=5
    should produce ~90 chunks (1440 / stride=16 = 90)."""
    plan = _mod.plan_chunks(total_frames=1440, chunk_size=21, overlap=5)
    # stride=16, 1440/16 = 90; possibly +1 for an anchor chunk if the
    # last stride doesn't reach 1440 exactly.
    # 1440 - 21 = 1419; 1419 / 16 = 88.6875; floor = 88 regular chunks
    # starting at 0..88*16=1408. (1408, 1429). 1429 < 1440 → anchor.
    # Total = 89 regular + 1 anchor = 90. Tight check:
    assert 89 <= len(plan) <= 91
