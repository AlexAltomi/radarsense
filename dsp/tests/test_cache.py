import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from radarsense_dsp import cache
from radarsense_dsp.cache import (
    Constants,
    StaleConstantsError,
    build_file,
    estimate_cache_bytes,
    plan_members,
)
from radarsense_dsp.frames import process_frames
from radarsense_dsp.io import list_members, parse_member

N_REC = 3
CONSTANTS = Constants(db_min=0.0, db_max=60.0, pair_a=(0, 2), pair_b=(1, 2), source_commit="abc123")


def make_zip(tmp_path, stems, labels=(1, 2, 3), split_recording=None):
    """Create a zip in the style of radar_dataset. Every recording has an 8-frame gesture, except `split_recording`, which has two runs."""
    rng = np.random.default_rng(0)
    originals = {}
    zip_path = tmp_path / "radar_dataset.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for stem in stems:
            inputs = rng.standard_normal((N_REC, 100, 3, 32, 64)).astype(np.float32)
            targets = np.zeros((N_REC, 100), dtype=np.int32)
            for i, label in enumerate(labels):
                targets[i, 40:48] = label
            if split_recording is not None:
                targets[split_recording, 60:64] = labels[split_recording]
            npz = tmp_path / f"{stem}.npz"
            np.savez(npz, inputs=inputs, targets=targets)
            z.write(npz, f"fulldataset/{stem}.npz")
            originals[stem] = (inputs, targets)
    return zip_path, originals


def member(stem):
    return parse_member(f"fulldataset/{stem}.npz")


@pytest.fixture(autouse=True)
def no_test_override(monkeypatch):
    monkeypatch.delenv("RADARSENSE_ALLOW_TEST", raising=False)


def test_build_file_writes_frames_targets_and_index(tmp_path):
    zip_path, originals = make_zip(tmp_path, ["user4_e1"])
    out = tmp_path / "processed"
    result = build_file(zip_path, member("user4_e1"), out, CONSTANTS, build_commit="def456")
    assert result.outcome == "built"

    inputs, targets = originals["user4_e1"]
    frames = np.load(out / "user4_e1.npy", mmap_mode="r")
    assert frames.shape == (N_REC, 100, 3, 32, 32) and frames.dtype == np.float16
    expected = np.stack([process_frames(r, 0.0, 60.0, (0, 2), (1, 2)) for r in inputs])
    assert np.array_equal(frames, expected)
    saved_targets = np.load(out / "user4_e1.targets.npy")
    assert saved_targets.dtype == np.int8 and np.array_equal(saved_targets, targets)
    assert sorted(p.name for p in out.iterdir()) == ["index.json", "user4_e1.npy", "user4_e1.targets.npy"]

    entry = json.loads((out / "index.json").read_text())["files"]["user4_e1"]
    assert entry["status"] == "complete" and entry["n"] == N_REC
    assert (entry["user"], entry["session"], entry["anomaly"]) == (4, 1, None)
    assert entry["shape"] == [N_REC, 100, 3, 32, 32] and entry["dtype"] == "float16"
    assert entry["span_hist"] == {"8": N_REC}
    assert entry["violations"] == {"count": 0, "kinds": {}, "first": []}
    assert entry["constants"] == {
        "db_min": 0.0, "db_max": 60.0, "pair_a": [0, 2], "pair_b": [1, 2], "source_commit": "abc123",
    }
    assert entry["build_commit"] == "def456"


def test_second_run_skips_a_complete_file_without_opening_the_zip(tmp_path):
    zip_path, _ = make_zip(tmp_path, ["user4_e1"])
    out = tmp_path / "processed"
    build_file(zip_path, member("user4_e1"), out, CONSTANTS, "c")
    again = build_file(tmp_path / "missing.zip", member("user4_e1"), out, CONSTANTS, "c")
    assert again.outcome == "skipped"


def test_stale_constants_are_not_silently_skipped(tmp_path):
    zip_path, _ = make_zip(tmp_path, ["user4_e1"])
    out = tmp_path / "processed"
    build_file(zip_path, member("user4_e1"), out, CONSTANTS, "c")
    other = Constants(db_min=1.0, db_max=60.0, pair_a=(0, 2), pair_b=(1, 2), source_commit="abc123")
    with pytest.raises(StaleConstantsError, match="user4_e1"):
        build_file(zip_path, member("user4_e1"), out, other, "c")


def test_span_violation_skips_the_file_and_records_it(tmp_path):
    zip_path, _ = make_zip(tmp_path, ["user4_e1"], split_recording=1)
    out = tmp_path / "processed"
    result = build_file(zip_path, member("user4_e1"), out, CONSTANTS, "c")
    assert result.outcome == "violation"
    assert sorted(p.name for p in out.iterdir()) == ["index.json"]  # The directory has no frames, no targets and no partial file.
    entry = json.loads((out / "index.json").read_text())["files"]["user4_e1"]
    assert entry["status"] == "violation"
    assert entry["violations"] == {"count": 1, "kinds": {"split_span": 1}, "first": [[1, "split_span"]]}
    assert "shape" not in entry
    # A member with a known violation is not read again.
    again = build_file(tmp_path / "missing.zip", member("user4_e1"), out, CONSTANTS, "c")
    assert again.outcome == "violation"


def test_test_user_is_refused_and_nothing_is_written(tmp_path):
    zip_path, _ = make_zip(tmp_path, ["user10_e1"])
    out = tmp_path / "processed"
    with pytest.raises(PermissionError):
        build_file(zip_path, member("user10_e1"), out, CONSTANTS, "c")
    assert not out.exists() or list(out.iterdir()) == []


def test_interrupted_build_leaves_only_partial_files_and_a_rerun_completes(tmp_path, monkeypatch):
    zip_path, _ = make_zip(tmp_path, ["user4_e1"])
    out = tmp_path / "processed"
    calls = {"n": 0}
    real = cache.process_frames

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("interrupted")
        return real(*args, **kwargs)

    monkeypatch.setattr(cache, "process_frames", flaky)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_file(zip_path, member("user4_e1"), out, CONSTANTS, "c")
    assert not (out / "user4_e1.npy").exists()
    assert not (out / "index.json").exists()
    assert (out / "user4_e1.npy.partial").exists()

    monkeypatch.setattr(cache, "process_frames", real)
    assert build_file(zip_path, member("user4_e1"), out, CONSTANTS, "c").outcome == "built"
    assert not list(out.glob("*.partial"))
    assert np.load(out / "user4_e1.npy", mmap_mode="r").shape[0] == N_REC


def test_non_finite_output_is_an_error_and_nothing_is_published(tmp_path, monkeypatch):
    zip_path, _ = make_zip(tmp_path, ["user4_e1"])
    out = tmp_path / "processed"
    monkeypatch.setattr(cache, "process_frames", lambda *a, **k: np.full((100, 3, 32, 32), np.nan, np.float16))
    with pytest.raises(FloatingPointError):
        build_file(zip_path, member("user4_e1"), out, CONSTANTS, "c")
    assert not (out / "user4_e1.npy").exists() and not (out / "user4_e1.targets.npy").exists()


def test_plan_members_orders_nominal_first_and_never_includes_test_users():
    names = [
        "user12_e1", "user10_e1_wrist", "user9_e1_fast", "user9_e1", "user2_e1", "user1_e2", "user1_e1",
        "user1_e1_slow", "user3_e1", "user12_e1_slow",
    ]
    members = [member(n) for n in names]
    planned = [Path(m.name).stem for m in plan_members(members, nominal_only=False)]
    assert planned == ["user1_e1", "user1_e2", "user2_e1", "user3_e1", "user9_e1", "user1_e1_slow", "user9_e1_fast"]
    assert [Path(m.name).stem for m in plan_members(members, nominal_only=True)] == planned[:5]


def test_estimate_cache_bytes_from_member_sizes(tmp_path):
    zip_path, _ = make_zip(tmp_path, ["user4_e1", "user9_e1_fast"])
    members = list_members(zip_path)
    assert estimate_cache_bytes(zip_path, members) == 2 * N_REC * (100 * 3 * 32 * 32 * 2 + 100)
