import numpy as np
import pytest
from conftest import SPLITS

from radarsense_dsp.windows import CLASS_MAP, CLASS_NAMES, far_span
from replay import codec
from replay.index import ReplayIndex, background_ranges
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
        assert len(frames) > 0
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
