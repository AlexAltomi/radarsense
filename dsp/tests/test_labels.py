import numpy as np

from radarsense_dsp.labels import analyze_targets


def make(*recs):
    """Each rec is a list of (value, start, length) runs on an otherwise-zero 100-frame recording."""
    t = np.zeros((len(recs), 100), dtype=np.int32)
    for i, runs in enumerate(recs):
        for value, start, length in runs:
            t[i, start : start + length] = value
    return t


def test_clean_contiguous_recordings():
    a = analyze_targets(make([(3, 20, 10)], [(1, 0, 10)], [(7, 90, 10)]))
    assert a["n"] == 3
    assert a["labelled_hist"] == {10: 3}
    assert a["zero_gesture"] == 0
    assert a["multi_value"] == {"count": 0, "first_indices": []}
    assert a["start"] == {"min": 0, "median": 20, "max": 90, "hist10": {0: 1, 20: 1, 90: 1}}
    assert a["span_hist"] == {10: 3}
    assert a["runs_hist"] == {1: 3}
    assert a["contiguous"] == 3


def test_empty_recording_counts_as_zero_gesture_and_is_excluded_from_start_stats():
    a = analyze_targets(make([], [(2, 40, 10)]))
    assert a["labelled_hist"] == {0: 1, 10: 1}
    assert a["zero_gesture"] == 1
    assert a["start"]["min"] == a["start"]["max"] == 40
    assert a["runs_hist"] == {1: 1}  # zero-gesture recordings have no runs and are not counted here
    assert a["contiguous"] == 1


def test_split_run_is_not_contiguous():
    a = analyze_targets(make([(3, 10, 5), (3, 30, 5)]))
    assert a["labelled_hist"] == {10: 1}
    assert a["runs_hist"] == {2: 1}
    assert a["contiguous"] == 0
    assert a["span_hist"] == {25: 1}  # frames 10..34
    assert a["multi_value"]["count"] == 0


def test_multiple_gesture_values_flagged_with_indices():
    a = analyze_targets(make([(1, 0, 10)], [(1, 5, 5), (6, 50, 5)], [(2, 10, 10)], [(3, 0, 5), (7, 60, 5)]))
    assert a["multi_value"] == {"count": 2, "first_indices": [1, 3]}


def test_first_indices_capped_at_ten():
    a = analyze_targets(make(*[[(1, 0, 5), (2, 50, 5)] for _ in range(15)]))
    assert a["multi_value"]["count"] == 15
    assert a["multi_value"]["first_indices"] == list(range(10))


def test_all_zero_input_has_no_start_stats():
    a = analyze_targets(make([], []))
    assert a["zero_gesture"] == 2
    assert a["start"] is None
    assert a["span_hist"] == {}
