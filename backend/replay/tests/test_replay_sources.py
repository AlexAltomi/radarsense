import numpy as np
import pytest
from conftest import SPLITS, write_cache

from radarsense_dsp.windows import CLASS_MAP, CLASS_NAMES, WINDOW, far_span
from replay import codec
from replay.index import (
    ReplayError,
    ReplayIndex,
    background_ranges,
    usable_background_runs,
)
from replay.sources import labels_from_targets, load_clip, next_clip


@pytest.fixture(scope="module")
def index(cache_dir):
    return ReplayIndex(cache_dir, SPLITS)


def frame_numbers(clip):
    """Recover the frame index inside the recording from channel 0 (which is f / 99)."""
    return np.rint(codec.decode_frames(clip.frames)[:, 0, 0, 0] * 99).astype(int)


def test_labels_use_class_map_and_255_for_unknown():
    t = np.array([0, 1, 2, 3, 6, 7, 4, 5, 9, -1, 127])
    expected = [CLASS_MAP.get(int(v), 255) for v in t]
    assert labels_from_targets(t).tolist() == expected
    assert labels_from_targets(t).tolist() == [0, 1, 2, 3, 4, 5, 255, 255, 255, 255, 255]


def test_a_full_recording_clip_has_100_frames_and_per_frame_labels(index):
    rec = index.recs[(index.recs["user"] == 3) & (index.recs["label"] == 6)][0]
    clip = load_clip(index, rec, False, np.random.default_rng(0))
    assert clip.frames.shape == (100, 3, 32, 32) and clip.frames.dtype == np.uint8
    assert frame_numbers(clip).tolist() == list(range(100))
    s, length = int(rec["s"]), int(rec["length"])
    expected = np.zeros(100, dtype=np.uint8)
    expected[s : s + length] = CLASS_MAP[6]
    assert np.array_equal(clip.labels, expected)


def test_background_ranges_stay_outside_the_far_span():
    for s, length in [(40, 8), (5, 8), (60, 10), (10, 80), (30, 40)]:
        lo, hi = far_span(s, length)
        for r in background_ranges(s, length):
            assert all(f < lo or f > hi for f in r)
    assert [(r.start, r.stop) for r in background_ranges(40, 8)] == [(0, 20), (58, 100)]
    assert background_ranges(10, 80) == []


def test_background_frames_never_fall_inside_the_extended_span(index):
    rng = np.random.default_rng(1)
    seen_low = seen_high = False
    for _ in range(300):
        rec, background = index.pick(rng, "any", "background")
        assert background
        clip = load_clip(index, rec, True, rng)
        lo, hi = far_span(int(rec["s"]), int(rec["length"]))
        frames = frame_numbers(clip)
        assert len(frames) >= WINDOW  # only runs that fill the detector buffer are streamed
        assert len(frames) == len(clip.labels)
        assert ((frames < lo) | (frames > hi)).all()
        assert clip.first_frame == frames[0]
        assert (clip.labels == 0).all()  # outside the gesture, the target is 0 and CLASS_MAP[0] is 0
        seen_low |= frames[0] == 0
        seen_high |= frames[-1] == 99
    assert seen_low and seen_high  # both sides of the gesture are used


def test_next_clip_for_a_named_gesture(index):
    clip = next_clip(index, np.random.default_rng(2), "user3", CLASS_NAMES[3])
    assert len(clip.frames) == 100
    assert set(np.unique(clip.labels).tolist()) == {0, 3}


def test_usable_background_runs_keep_only_runs_of_at_least_one_window():
    assert [(r.start, r.stop) for r in usable_background_runs(40, 8)] == [(58, 100)]  # the run 0..19 is too short
    assert [(r.start, r.stop) for r in usable_background_runs(50, 8)] == [(68, 100)]  # exactly WINDOW frames is kept
    assert [(r.start, r.stop) for r in usable_background_runs(30, 30)] == []  # the only run has 30 frames
    assert all(len(r) >= WINDOW for s, n in [(5, 8), (60, 10), (35, 8)] for r in usable_background_runs(s, n))
    assert len(usable_background_runs(50, 8)[0]) == WINDOW


def test_both_usable_runs_of_a_recording_are_used_and_short_ones_never(tmp_path):
    # s=55, L=3: the zone is 35..67, so the runs are 0..34 (35 frames) and 68..99 (32 frames). Both are usable.
    cache = write_cache(tmp_path / "c", {"user3_e1": (3, 1, None, "complete", [(2, 55, 3)])})
    idx = ReplayIndex(cache, SPLITS)
    rng = np.random.default_rng(0)
    starts = [load_clip(index=idx, rec=idx.recs[0], background=True, rng=rng).first_frame for _ in range(200)]
    assert set(starts) == {0, 68}
    assert 0.4 < starts.count(0) / 200 < 0.7  # chosen by length: 35 of 67 frames


def test_a_user_with_only_short_background_runs_gets_an_error(tmp_path):
    # s=30, L=30: the zone is 10..69, so the only run is 70..99 with 30 frames, which is shorter than WINDOW.
    cache = write_cache(tmp_path / "c", {"user3_e1": (3, 1, None, "complete", [(2, 30, 30)])})
    idx = ReplayIndex(cache, SPLITS)
    assert idx.background_frames.tolist() == [0]
    with pytest.raises(ReplayError) as e:
        next_clip(idx, np.random.default_rng(0), "user3", "background")
    assert str(e.value) == f"No background run of at least {WINDOW} frames matches user 'user3'."
    next_clip(idx, np.random.default_rng(0), "user3", "any")  # gestures still work
