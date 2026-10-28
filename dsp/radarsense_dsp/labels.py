"""Analyze the structure of the per-frame `targets` arrays.

The functions here report how many frames of each recording are labelled, where the labelled frames start, and how
many separate runs they form. Label 0 means "no gesture", and every other value is a gesture. The module works on
the targets only and never touches radar data.
"""
from collections import Counter

import numpy as np

MAX_LISTED_INDICES = 10
START_BIN = 10


def _sorted(counter: Counter) -> dict[int, int]:
    return {int(k): int(v) for k, v in sorted(counter.items())}


def analyze_targets(targets: np.ndarray) -> dict:
    """Summarize a targets array of shape (recordings, frames).

    The start and span statistics cover only the recordings that contain a gesture. The entry `runs_hist` counts the
    maximal runs of consecutive labelled (non-zero) frames of each recording. A single run means that the gesture
    is contiguous.
    """
    labelled_hist: Counter = Counter()
    span_hist: Counter = Counter()
    runs_hist: Counter = Counter()
    start_hist: Counter = Counter()
    starts: list[int] = []
    multi_value: list[int] = []

    for i, row in enumerate(targets):
        mask = row != 0
        n_labelled = int(mask.sum())
        labelled_hist[n_labelled] += 1
        if n_labelled == 0:
            continue
        idx = np.flatnonzero(mask)
        start, end = int(idx[0]), int(idx[-1])
        starts.append(start)
        start_hist[start // START_BIN * START_BIN] += 1
        span_hist[end - start + 1] += 1
        runs_hist[int(1 + np.count_nonzero(np.diff(idx) > 1))] += 1
        if len(np.unique(row[mask])) > 1:
            multi_value.append(i)

    start_stats = None
    if starts:
        start_stats = {
            "min": min(starts),
            "median": float(np.median(starts)),
            "max": max(starts),
            "hist10": _sorted(start_hist),
        }
    return {
        "n": len(targets),
        "labelled_hist": _sorted(labelled_hist),
        "zero_gesture": labelled_hist[0],
        "multi_value": {"count": len(multi_value), "first_indices": multi_value[:MAX_LISTED_INDICES]},
        "start": start_stats,
        "span_hist": _sorted(span_hist),
        "runs_hist": _sorted(runs_hist),
        "contiguous": runs_hist[1],
    }


def span_violations(targets: np.ndarray) -> list[tuple[int, str]]:
    """List the violations as pairs of recording index and kind.

    A recording is a violation unless it holds exactly one contiguous gesture of a single label value. The kinds are
    `no_gesture`, `multiple_values` and `split_span`. A recording with several values is reported as
    `multiple_values` even if it is also split.
    """
    out: list[tuple[int, str]] = []
    for i, row in enumerate(targets):
        idx = np.flatnonzero(row)
        if len(idx) == 0:
            out.append((i, "no_gesture"))
        elif len(np.unique(row[idx])) > 1:
            out.append((i, "multiple_values"))
        elif idx[-1] - idx[0] + 1 != len(idx):
            out.append((i, "split_span"))
    return out
