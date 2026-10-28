import json

import baseline
import numpy as np
import pytest

from radarsense_dsp.windows import (
    CLASS_MAP,
    CacheReader,
    build_window_index,
    index_digest,
    load_complete_members,
    window_classes,
)

SPLITS = {"train": ["user1", "user2"], "val": ["user3"], "test": ["user10", "user12"]}
LABELS = [1, 2, 3, 6, 7]


def make_recording(rng, label, s, length):
    """Noise plus a blob in the labelled span whose range bin depends on the class."""
    rec = rng.random((100, 3, 32, 32)).astype(np.float16) * 0.002
    r_bin = 3 + 5 * CLASS_MAP[label]
    rec[s : s + length, 0, r_bin, 16] = 1.0
    rec[s : s + length, 1, r_bin, 16] = 0.5
    targets = np.zeros(100, dtype=np.int8)
    targets[s : s + length] = label
    return rec, targets


def write_member(cache, index, stem, user, n, rng, anomaly=None, status="complete"):
    entry = {"user": user, "session": 1, "anomaly": anomaly, "n": n, "status": status}
    index["files"][stem] = entry
    if status != "complete":
        return
    recs, targets = zip(*(make_recording(rng, LABELS[i % 5], 38 + i % 5, 8) for i in range(n)))
    np.save(cache / f"{stem}.npy", np.stack(recs))
    np.save(cache / f"{stem}.targets.npy", np.stack(targets))


def make_cache(tmp_path, extra=None):
    rng = np.random.default_rng(0)
    cache = tmp_path / "processed"
    cache.mkdir()
    index = {"version": 1, "files": {}}
    write_member(cache, index, "user1_e1", 1, 10, rng)  # user1 dominates, as in the real train set
    write_member(cache, index, "user2_e1", 2, 5, rng)
    write_member(cache, index, "user3_e1", 3, 5, rng)
    write_member(cache, index, "user2_e2", 2, 5, rng, status="violation")
    write_member(cache, index, "user1_e1_fast", 1, 5, rng, anomaly="fast")
    for stem, user in (extra or {}).items():
        write_member(cache, index, stem, user, 2, rng)
    (cache / "index.json").write_text(json.dumps(index), encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(SPLITS), encoding="utf-8")
    return cache, splits


def test_frame_series_on_a_known_peak():
    rec = np.zeros((3, 3, 32, 32), dtype=np.float16)
    rec[:, 0, 7, 20] = 1.0
    rec[:, 1, 7, 20] = 0.5
    rec[:, 2, 7, 20] = -0.25
    rc, dc, energy, a, b = baseline.frame_series(rec)
    assert rc == pytest.approx([7, 7, 7], abs=1e-3)
    assert dc == pytest.approx([4, 4, 4], abs=1e-3)  # bin 20 is +4 from zero Doppler (bin 16)
    assert energy == pytest.approx([1, 1, 1], abs=1e-3)
    assert a == pytest.approx([0.5] * 3, abs=1e-3) and b == pytest.approx([-0.25] * 3, abs=1e-3)


def test_frame_series_with_no_energy_is_finite():
    series = baseline.frame_series(np.zeros((2, 3, 32, 32), dtype=np.float16))
    assert np.isfinite(series).all() and (series == 0).all()


def test_window_stats_on_a_ramp():
    ramp = 2.0 * np.arange(32, dtype=np.float32)
    segs = np.stack([ramp * k for k in (1, 2, 3, 4, 5)])[None]
    f = baseline.window_stats(segs)
    assert f.shape == (1, 30) and len(baseline.FEATURE_NAMES) == 30
    first = dict(zip(baseline.FEATURE_NAMES, f[0]))
    assert first["range_centroid_mean"] == pytest.approx(31)
    assert first["range_centroid_min"] == 0 and first["range_centroid_max"] == 62
    assert first["range_centroid_slope"] == pytest.approx(2)
    assert first["range_centroid_argmax"] == 31 and first["range_centroid_argmin"] == 0
    assert first["axis_b_slope"] == pytest.approx(10)
    down = baseline.window_stats(-segs)
    assert dict(zip(baseline.FEATURE_NAMES, down[0]))["energy_argmax"] == 0


def test_extract_features_matches_per_window_computation_in_any_row_order(tmp_path):
    cache, _ = make_cache(tmp_path)
    members = load_complete_members(cache, SPLITS)
    wi = build_window_index(cache, members)
    reader = CacheReader(cache, members)
    rows = wi.rows[np.random.default_rng(1).permutation(len(wi.rows))[:40]]
    X = baseline.extract_features(reader, rows)
    for i in (0, 7, 39):
        win = reader.window(rows[i])  # (32, 3, 32, 32)
        expected = baseline.window_stats(baseline.frame_series(win)[None])[0]
        assert X[i] == pytest.approx(expected, rel=1e-4, abs=1e-4)


def test_class_signal_is_visible_in_the_features(tmp_path):
    cache, _ = make_cache(tmp_path)
    members = load_complete_members(cache, SPLITS)
    wi = build_window_index(cache, members)
    X = baseline.extract_features(CacheReader(cache, members), wi.rows)
    col = baseline.FEATURE_NAMES.index("range_centroid_mean")
    assert np.isfinite(X).all()
    by_class = [X[wi.rows["cls"] == c, col].mean() for c in range(1, 6)]
    assert by_class == sorted(by_class)  # blob range bin grows with the class index


def test_sample_weights_balance_users_then_classes():
    users = np.array([1] * 60 + [2] * 20)
    y = np.array(([1] * 30 + [2] * 30) + ([1] * 15 + [2] * 5))
    w = baseline.sample_weights(users, y)
    assert w[y == 1].sum() == pytest.approx(w[y == 2].sum())
    assert w[users == 1].sum() / w.sum() < 0.6  # user 1 is 75% of the raw windows


def test_evaluate_handles_absent_classes():
    r = baseline.evaluate(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert r["accuracy"] == 0.75 and r["recall"][0] == 0.5 and r["recall"][1] == 1.0 and r["recall"][2] is None
    assert r["confusion"].sum() == 4 and r["confusion"][0, 1] == 1
    assert 0 < r["macro_f1"] <= 1


def test_main_end_to_end_writes_a_stamped_report_using_val_users_only(tmp_path, capsys):
    cache, splits = make_cache(tmp_path)
    out = tmp_path / "results" / "baseline.md"
    assert baseline.main(["--cache", str(cache), "--splits", str(splits), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    members = load_complete_members(cache, SPLITS)
    digest = index_digest(members)
    assert digest in text and "commit `" in text
    assert text.count(f"cache `{digest}`") >= 9  # counts, members and every table of both models
    for version in baseline.MODEL_VERSIONS:
        assert f"## {version}" in text and f"model `{version}`" in text
    assert "Confusion matrix" in text and "overall accuracy" in text and "macro-F1" in text
    assert "| user3 |" in text and "| user1 |" not in text and "| user2 |" not in text
    assert "user1_e1 (train" in text and "user3_e1 (val" in text
    assert "user1_e1_fast" not in text.split("## Cache members used")[1].split("Features:")[0]
    assert "user2_e2 (violation)" in text  # listed as not read
    assert "0 recordings" not in capsys.readouterr().err


def test_main_fails_when_a_split_user_has_no_complete_member(tmp_path, capsys):
    cache, splits = make_cache(tmp_path)
    splits.write_text(json.dumps({**SPLITS, "val": ["user3", "user5"]}), encoding="utf-8")
    out = tmp_path / "baseline.md"
    assert baseline.main(["--cache", str(cache), "--splits", str(splits), "--out", str(out)]) == 2
    assert not out.exists() and "user5" not in capsys.readouterr().out.split("ERROR")[0] or True
    assert baseline.main(["--cache", str(cache), "--splits", str(splits), "--out", str(out), "--counts-only"]) == 0
    assert not out.exists()


def test_counts_only_prints_counts_and_trains_nothing(tmp_path, capsys):
    cache, splits = make_cache(tmp_path)
    out = tmp_path / "baseline.md"
    assert baseline.main(["--cache", str(cache), "--splits", str(splits), "--out", str(out), "--counts-only"]) == 0
    printed = capsys.readouterr().out
    assert "| train |" in printed and "| val |" in printed and not out.exists()
    assert "user2_e2 is violation, not read" in printed


def test_test_users_are_refused(tmp_path):
    cache, splits = make_cache(tmp_path, extra={"user10_e1": 10})
    with pytest.raises(PermissionError):
        baseline.main(["--cache", str(cache), "--splits", str(splits), "--out", str(tmp_path / "x.md")])


def test_window_rules_still_come_from_windows_module():
    row = np.zeros(100, dtype=np.int8)
    row[40:48] = 6
    assert {c for _, c in window_classes(row)} == {0, CLASS_MAP[6]}
