import zipfile

import numpy as np
import pytest

from radarsense_dsp.io import (
    Member,
    iter_recordings,
    list_members,
    open_recordings,
    parse_member,
    read_meta,
)

N_REC = 3
SHAPE = (N_REC, 100, 3, 32, 64)


def _npz_bytes(path, inputs, targets):
    np.savez(path, inputs=inputs, targets=targets)


@pytest.fixture
def synthetic_zip(tmp_path):
    rng = np.random.default_rng(0)
    originals = {}
    zip_path = tmp_path / "radar_dataset.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.mkdir("fulldataset")
        for name in ("user4_e1", "user10_e1"):
            inputs = rng.standard_normal(SHAPE).astype(np.float32)
            targets = rng.integers(0, 4, size=(N_REC, 100)).astype(np.int32)
            npz = tmp_path / f"{name}.npz"
            _npz_bytes(npz, inputs, targets)
            z.write(npz, f"fulldataset/{name}.npz")
            originals[name] = (inputs, targets)
    return zip_path, originals


def test_parse_member_names():
    assert parse_member("fulldataset/user1_e3.npz") == Member(
        "fulldataset/user1_e3.npz", 1, 3, None
    )
    m = parse_member("fulldataset/user12_e1_wrist.npz")
    assert (m.user, m.session, m.anomaly) == (12, 1, "wrist")
    assert parse_member("user10_e1_slow.npz").anomaly == "slow"
    with pytest.raises(ValueError):
        parse_member("fulldataset/readme.txt")


def test_list_members(synthetic_zip):
    zip_path, _ = synthetic_zip
    members = list_members(zip_path)
    assert [(m.user, m.session, m.anomaly) for m in members] == [
        (10, 1, None),
        (4, 1, None),
    ]


def test_stream_equals_original(synthetic_zip, monkeypatch):
    monkeypatch.delenv("RADARSENSE_ALLOW_TEST", raising=False)
    zip_path, originals = synthetic_zip
    inputs, targets = originals["user4_e1"]
    out = list(iter_recordings(zip_path, "fulldataset/user4_e1.npz"))
    assert len(out) == N_REC
    for i, (rec, tgt) in enumerate(out):
        assert rec.dtype == np.float32 and rec.shape == (100, 3, 32, 64)
        assert tgt.dtype == np.int32 and tgt.shape == (100,)
        assert np.array_equal(rec, inputs[i])
        assert np.array_equal(tgt, targets[i])


def test_read_meta(synthetic_zip, monkeypatch):
    monkeypatch.delenv("RADARSENSE_ALLOW_TEST", raising=False)
    zip_path, originals = synthetic_zip
    n, targets = read_meta(zip_path, "fulldataset/user4_e1.npz")
    assert n == N_REC
    assert np.array_equal(targets, originals["user4_e1"][1])
    with pytest.raises(PermissionError):
        read_meta(zip_path, "fulldataset/user10_e1.npz")


@pytest.mark.parametrize("env", [None, "0"])
def test_test_user_refused(synthetic_zip, monkeypatch, env):
    if env is None:
        monkeypatch.delenv("RADARSENSE_ALLOW_TEST", raising=False)
    else:
        monkeypatch.setenv("RADARSENSE_ALLOW_TEST", env)
    zip_path, _ = synthetic_zip
    with pytest.raises(PermissionError):
        next(iter_recordings(zip_path, "fulldataset/user10_e1.npz"))


def test_test_user_allowed_with_env(synthetic_zip, monkeypatch):
    monkeypatch.setenv("RADARSENSE_ALLOW_TEST", "1")
    zip_path, originals = synthetic_zip
    inputs, targets = originals["user10_e1"]
    out = list(iter_recordings(zip_path, "fulldataset/user10_e1.npz"))
    assert len(out) == N_REC
    for i, (rec, tgt) in enumerate(out):
        assert np.array_equal(rec, inputs[i])
        assert np.array_equal(tgt, targets[i])


def test_open_recordings_gives_targets_up_front_then_streams_inputs(synthetic_zip, monkeypatch):
    monkeypatch.delenv("RADARSENSE_ALLOW_TEST", raising=False)
    zip_path, originals = synthetic_zip
    inputs, targets = originals["user4_e1"]
    with open_recordings(zip_path, "fulldataset/user4_e1.npz") as (t, recordings):
        assert np.array_equal(t, targets)  # available before any recording is read
        out = list(recordings)
    assert len(out) == N_REC
    assert all(np.array_equal(rec, inputs[i]) for i, rec in enumerate(out))


def test_open_recordings_refuses_test_user_on_entry(synthetic_zip, monkeypatch):
    monkeypatch.delenv("RADARSENSE_ALLOW_TEST", raising=False)
    zip_path, _ = synthetic_zip
    with pytest.raises(PermissionError), open_recordings(zip_path, "fulldataset/user10_e1.npz"):
        pass
