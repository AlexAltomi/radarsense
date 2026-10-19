import zipfile

import compute_constants as cc
import numpy as np
import pytest
from compute_constants import (
    LO,
    NBINS,
    balanced_percentile,
    db_histogram,
    per_user,
    percentile_from_counts,
    process_file,
)

from radarsense_dsp.io import Member


def make_zip(tmp_path, stem, labels):
    """One train-user member with len(labels) recordings; recording i has an 8-frame gesture of labels[i]."""
    rng = np.random.default_rng(0)
    inputs = rng.standard_normal((len(labels), 100, 3, 32, 64)).astype(np.float32)
    targets = np.zeros((len(labels), 100), dtype=np.int32)
    for i, label in enumerate(labels):
        targets[i, 40:48] = label
    npz = tmp_path / f"{stem}.npz"
    np.savez(npz, inputs=inputs, targets=targets)
    zip_path = tmp_path / "radar_dataset.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(npz, f"fulldataset/{stem}.npz")
    return zip_path


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


def test_per_user_sums_files_and_percentiles_come_from_histograms_only():
    cache = {
        "hist__user1_e1": db_histogram(np.full(100, 20.0)),
        "hist__user1_e2": db_histogram(np.full(100, 20.0)),
        "hist__user2_e1": db_histogram(np.full(10, 40.0)),
    }
    members = [
        Member("fulldataset/user1_e1.npz", 1, 1, None),
        Member("fulldataset/user1_e2.npz", 1, 2, None),
        Member("fulldataset/user2_e1.npz", 2, 1, None),
    ]
    users = per_user(cache, members)
    assert users[1].sum() == 200 and users[2].sum() == 10
    assert balanced_percentile({str(u): h for u, h in users.items()}, 75) == pytest.approx(40.0, abs=0.01)


def test_process_file_returns_one_fixed_size_histogram_and_stops_early(tmp_path, monkeypatch):
    labels = [1, 2, 3, 6, 7, 1, 2, 3, 6, 7, 1, 2]  # with KEEP=2 every label is full after 10 recordings
    monkeypatch.setattr(cc, "ZIP", make_zip(tmp_path, "user4_e1", labels))
    monkeypatch.setattr(cc, "KEEP", 2)
    hist, kept = process_file(Member("fulldataset/user4_e1.npz", 4, 1, None))
    assert hist.shape == (NBINS,)
    assert kept == {1: 2, 2: 2, 3: 2, 6: 2, 7: 2}
    assert hist.sum() == 10 * 100 * 32 * 32  # recordings 11 and 12 were never processed


def test_select_members():
    ms = [Member("fulldataset/user4_e1.npz", 4, 1, None), Member("fulldataset/user8_e1.npz", 8, 1, None)]
    assert cc.select_members(ms, None) == ms
    assert cc.select_members(ms, "user8_e1") == [ms[1]]
    assert cc.select_members(ms, "fulldataset/user4_e1.npz") == [ms[0]]
    with pytest.raises(SystemExit):
        cc.select_members(ms, "user10_e1")  # a test user is not a train nominal member
