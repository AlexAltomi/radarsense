import itertools
import json

import numpy as np
import pytest

from radarsense_dsp import windows
from radarsense_dsp.windows import (
    CLASS_MAP,
    BalancedUserSampler,
    CacheReader,
    build_window_index,
    count_table,
    gesture_span,
    incomplete_members,
    index_digest,
    load_complete_members,
    no_gesture_recording_share,
    scarcity_warnings,
    user_balanced_weights,
    window_classes,
)

SPLITS = {
    "train": ["user1", "user2"],
    "val": ["user3"],
    "test": ["user10", "user12"],
}


def row(label, s, length, frames=100):
    r = np.zeros(frames, dtype=np.int8)
    r[s : s + length] = label
    return r


def groups(r):
    out: dict[int, list[int]] = {}
    for w, cls in window_classes(r):
        out.setdefault(cls, []).append(w)
    return out


def test_stride_and_window_fit_in_100_frames():
    starts = list(windows.window_starts())
    assert starts[0] == 0 and starts[-1] == 68 and len(starts) == 18
    assert {b - a for a, b in itertools.pairwise(starts)} == {4}


@pytest.mark.parametrize(
    "s, length, gesture, none",
    [
        (40, 8, [16, 20, 24, 28], [60, 64, 68]),  # The gesture is in the middle.
        (21, 8, [0, 4, 8], list(range(40, 69, 4))),  # The gesture is near the start.
        (5, 8, [0], list(range(24, 69, 4))),  # The gesture starts before frame 10, so the extended span is clipped at frame 0.
        (60, 10, [40, 44, 48], [0, 4, 8]),  # The span has 10 frames.
        (88, 8, [64, 68], list(range(0, 37, 4))),  # The gesture is at the end.
    ],
)
def test_window_sets_hand_computed(s, length, gesture, none):
    g = groups(row(3, s, length))
    assert g.get(CLASS_MAP[3], []) == gesture
    assert g.get(0, []) == none


def test_partial_overlap_windows_are_excluded_and_never_no_gesture():
    s, length = 40, 8
    labelled = dict(window_classes(row(2, s, length)))
    excluded = [w for w in windows.window_starts() if w not in labelled]
    assert excluded == [0, 4, 8, 12] + list(range(32, 57, 4))
    for w, cls in labelled.items():
        if cls == 0:  # A no-gesture window is at least 10 frames away from the extended span [s-10, s+L-1].
            assert w + 31 < s - 20 or w > s + length + 9
        else:  # A gesture window contains the extended span completely.
            assert w <= s - 10 and w + 31 >= s + length - 1


def test_no_window_labelled_zero_inside_the_guard():
    for s in range(15, 80, 3):
        for length in (8, 10):
            if s + length > 100:
                continue
            for w, cls in window_classes(row(1, s, length)):
                if cls == 0:
                    assert not (w <= s + length + 9 and w + 31 >= s - 20)


def test_span_length_is_read_from_the_recording_itself():
    assert gesture_span(row(6, 30, 8)) == (6, 30, 8)
    assert gesture_span(row(7, 30, 10)) == (7, 30, 10)
    assert groups(row(6, 40, 8)) != groups(row(6, 40, 10))


@pytest.mark.parametrize("bad", ["none", "two_values", "split", "unknown_label"])
def test_bad_recordings_raise(bad):
    r = row(2, 40, 8)
    if bad == "none":
        r[:] = 0
    elif bad == "two_values":
        r[44:48] = 3
    elif bad == "split":
        r[44] = 0
    else:
        r[:] = 0
        r[40:48] = 5
    with pytest.raises(ValueError):
        window_classes(r)


def test_class_map_is_the_single_definition():
    assert CLASS_MAP == {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 7: 5}
    assert len(windows.CLASS_NAMES) == windows.N_CLASSES == len(set(CLASS_MAP.values()))
    j = windows.class_map_json()
    assert j["label_to_class"] == {"0": 0, "1": 1, "2": 2, "3": 3, "6": 4, "7": 5}
    for label, cls in CLASS_MAP.items():
        if label == 0:
            continue
        classes = {c for _, c in window_classes(row(label, 40, 8)) if c != 0}
        assert classes == {cls}


# Tests that use a synthetic cache.


def make_cache(tmp_path, spec: dict[str, dict]):
    """Write a synthetic cache from a spec.

    The spec maps each stem to a dict with the keys "user", "anomaly", "status" (default "complete"), "spans" (a list of
    (label, s, L) tuples) and "files" (default True).
    """
    cache = tmp_path / "processed"
    cache.mkdir()
    index = {"version": 1, "files": {}}
    for stem, m in spec.items():
        spans = m["spans"]
        n = len(spans)
        entry = {"user": m["user"], "session": 1, "anomaly": m.get("anomaly"), "n": n, "status": m.get("status", "complete")}
        index["files"][stem] = entry
        if m.get("files", True) and entry["status"] == "complete":
            targets = np.stack([row(*sp) for sp in spans])
            frames = np.zeros((n, 100, 3, 32, 32), dtype=np.float16)
            frames[:, :, 0, 0, 0] = np.arange(100) / 100  # This channel stores the frame number.
            frames[:, :, 1, 0, 0] = np.arange(n)[:, None]  # This channel stores the recording number.
            np.save(cache / f"{stem}.targets.npy", targets)
            np.save(cache / f"{stem}.npy", frames)
    (cache / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return cache


def spec_basic() -> dict[str, dict]:
    return {
        "user1_e1": {"user": 1, "spans": [(1, 40, 8), (2, 40, 8), (6, 60, 10)]},
        "user2_e1": {"user": 2, "spans": [(3, 40, 8)]},
        "user3_e1": {"user": 3, "spans": [(7, 40, 8), (1, 30, 8)]},
    }


def test_loader_keeps_only_complete_nominal_members_with_files(tmp_path):
    spec = spec_basic()
    spec["user1_e2"] = {"user": 1, "status": "violation", "spans": [(1, 40, 8)]}
    spec["user2_e1_fast"] = {"user": 2, "anomaly": "fast", "spans": [(1, 40, 8)]}
    spec["user1_e3"] = {"user": 1, "spans": [(1, 40, 8)], "files": False}  # The index says complete, but the files are missing.
    cache = make_cache(tmp_path, spec)
    stems = [m.stem for m in load_complete_members(cache, SPLITS)]
    assert stems == ["user1_e1", "user2_e1", "user3_e1"]
    assert [m.stem for m in incomplete_members(cache, SPLITS)] == ["user1_e2", "user1_e3"]
    with_anomalous = [m.stem for m in load_complete_members(cache, SPLITS, nominal_only=False)]
    assert "user2_e1_fast" in with_anomalous
    split_of = {m.stem: m.split for m in load_complete_members(cache, SPLITS)}
    assert split_of == {"user1_e1": "train", "user2_e1": "train", "user3_e1": "val"}


def test_test_users_are_refused(tmp_path):
    spec = spec_basic()
    spec["user10_e1"] = {"user": 10, "spans": [(1, 40, 8)]}
    cache = make_cache(tmp_path, spec)
    with pytest.raises(PermissionError):
        load_complete_members(cache, SPLITS)


def test_window_index_and_counts(tmp_path):
    cache = make_cache(tmp_path, spec_basic())
    members = load_complete_members(cache, SPLITS)
    wi = build_window_index(cache, members)
    # The recordings (1,40,8), (2,40,8) and (3,40,8) give 4 gesture and 3 no-gesture windows each, and (6,60,10) gives 3 and 3.
    train = count_table(wi)["train"]
    assert train == [3 + 3 + 3 + 3, 4, 4, 4, 3, 0]
    val = count_table(wi)["val"]
    assert val[0] == 3 + len([w for w, c in window_classes(row(1, 30, 8)) if c == 0])
    assert val[5] == 4 and val[1] == len([w for w, c in window_classes(row(1, 30, 8)) if c == 1])
    assert set(wi.split_rows("val")["user"]) == {3}
    assert set(wi.rows["user"]) == {1, 2, 3}
    assert no_gesture_recording_share(wi, "train") == 1.0


def test_scarcity_warnings():
    assert scarcity_warnings({"train": [100, 50, 60, 70, 80, 90], "val": [10, 5, 5, 5, 5, 5]}) == []
    assert "has no windows of the no-gesture class" in scarcity_warnings({"train": [0, 5, 5, 5, 5, 5]})[0]
    assert "fewer" in scarcity_warnings({"val": [3, 5, 5, 5, 5, 5]})[0]


def test_index_digest_depends_only_on_the_given_members(tmp_path):
    cache = make_cache(tmp_path, spec_basic())
    members = load_complete_members(cache, SPLITS)
    d = index_digest(members)
    assert d == index_digest(list(reversed(members)))
    assert d != index_digest(members[:2])
    assert len(d) == 16


def test_mmap_reads_return_the_requested_slice(tmp_path):
    cache = make_cache(tmp_path, spec_basic())
    members = load_complete_members(cache, SPLITS)
    wi = build_window_index(cache, members)
    reader = CacheReader(cache, members)
    i = int(np.flatnonzero((wi.rows["member"] == 0) & (wi.rows["recording"] == 2) & (wi.rows["start"] == 40))[0])
    w = reader.window(wi.rows[i])
    assert w.shape == (32, 3, 32, 32) and w.dtype == np.float16
    assert np.allclose(w[:, 0, 0, 0], np.arange(40, 72) / 100, atol=1e-3)
    assert (w[:, 1, 0, 0] == 2).all()
    assert reader.recording(0, 1).shape == (100, 3, 32, 32)
    assert isinstance(reader._map(0), np.memmap)


# Tests of the user balancing.


def test_sampler_gives_each_user_the_same_count_every_epoch():
    users = np.array([1] * 570 + [2] * 100 + [3] * 130)
    s = BalancedUserSampler(users, seed=4)
    assert s.per_user == 100 and len(s) == 300
    for epoch in range(3):
        idx = s.epoch_indices(epoch)
        assert len(idx) == 300 and idx.min() >= 0 and idx.max() < len(users)
        assert {int(u): int((users[idx] == u).sum()) for u in (1, 2, 3)} == {1: 100, 2: 100, 3: 100}
        assert len(set(idx.tolist())) == 300  # Windows are drawn without replacement.


def test_sampler_is_deterministic_and_reshuffles_per_epoch():
    users = np.array([1] * 50 + [2] * 20)
    a, b = BalancedUserSampler(users, seed=7), BalancedUserSampler(users, seed=7)
    assert list(a) == list(b)
    assert list(a) != list(b.epoch_indices(0).tolist())  # The sampler a is on epoch 1 now, while epoch 0 of b is replayed.
    assert not np.array_equal(a.epoch_indices(0), a.epoch_indices(1))
    assert not np.array_equal(BalancedUserSampler(users, seed=8).epoch_indices(0), a.epoch_indices(0))


def test_sampler_oversamples_a_small_user_when_asked():
    users = np.array([1] * 50 + [2] * 5)
    idx = BalancedUserSampler(users, per_user=20).epoch_indices(0)
    assert (users[idx] == 2).sum() == 20 and (users[idx] == 1).sum() == 20


def test_user_balanced_weights_sum_equally_per_user():
    users = np.array([1] * 57 + [2] * 30 + [3] * 13)
    w = user_balanced_weights(users)
    assert w.mean() == pytest.approx(1.0)
    assert [w[users == u].sum() for u in (1, 2, 3)] == pytest.approx([100 / 3] * 3)


# Tests of the per-start rule, the margin and the class-balanced sampler.


def test_window_class_at_agrees_with_window_classes_for_every_start():
    for s, length in [(40, 8), (5, 8), (88, 8), (60, 10)]:
        r = row(3, s, length)
        by_start = dict(window_classes(r))
        for w in windows.window_starts():
            assert windows.window_class_at(3, s, length, w) == by_start.get(w)


def test_window_margin_counts_frames_to_the_edge_of_the_rule():
    # s=40, L=8: the extended span is [30, 47]. A gesture window at 16 covers [16, 47], so it has 14 free frames on the
    # left and none on the right.
    assert windows.window_margin(40, 8, 16) == 0
    assert windows.window_margin(40, 8, 28) == 2  # It covers [28, 59]: 2 free frames on the left, 12 on the right.
    assert windows.window_margin(40, 8, 20) == 4  # It covers [20, 51]: 10 free frames on the left, 4 on the right.
    # The no-gesture window [60, 91] has to stay beyond 47 + 10, so it could move 2 frames towards the gesture.
    assert windows.window_margin(40, 8, 60) == 60 - 57 - 1
    assert windows.window_margin(40, 8, 40) is None  # An excluded window has no margin.


def test_class_balanced_sampler_gives_gestures_equal_shares_and_none_twice_that():
    users = np.array([1] * 600 + [2] * 200)
    classes = np.concatenate([np.repeat([0, 1, 2, 3, 4, 5], [400, 40, 50, 50, 30, 30]), np.repeat([0, 1, 2], [100, 50, 50])])
    s = BalancedUserSampler(users, per_user=70, seed=3, classes=classes, none_ratio=2.0)
    idx = s.epoch_indices(0)
    assert len(idx) == 140
    for u, present in ((1, [0, 1, 2, 3, 4, 5]), (2, [0, 1, 2])):
        got = np.bincount(classes[idx[users[idx] == u]], minlength=6)
        assert got.sum() == 70
        n_gesture = len(present) - 1
        unit = 70 / (n_gesture + 2)  # The no-gesture class counts twice.
        assert abs(got[0] - 2 * unit) <= 1 and all(abs(got[c] - unit) <= 1 for c in present[1:])
        assert all(got[c] == 0 for c in range(6) if c not in present)
