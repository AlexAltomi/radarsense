import numpy as np
import pytest
from compute_constants import (
    LO,
    NBINS,
    balanced_percentile,
    db_histogram,
    percentile_from_counts,
)


def test_percentile_of_uniform_dB():
    h = db_histogram(np.linspace(10, 60, 100001))
    assert percentile_from_counts(h, 50) == pytest.approx(35.0, abs=0.02)
    assert percentile_from_counts(h, 99.9) == pytest.approx(59.95, abs=0.02)


def test_out_of_range_values_are_clipped_into_edge_bins():
    h = db_histogram(np.array([-1000.0, 1000.0]))
    assert h.shape == (NBINS,) and h.sum() == 2 and h[0] == 1 and h[-1] == 1
    assert percentile_from_counts(h, 1) == pytest.approx(LO, abs=0.01)


def test_balanced_percentile_weights_users_equally():
    big = db_histogram(np.full(1000, 20.0))
    small = db_histogram(np.full(10, 40.0))
    assert percentile_from_counts(big + small, 75) == pytest.approx(20.0, abs=0.01)  # pooled: big user dominates
    assert balanced_percentile({"a": big, "b": small}, 75) == pytest.approx(40.0, abs=0.01)
