import json
import zipfile

import numpy as np
from build_cache import main

N_REC = 2


def make_zip(tmp_path, split_stems=()):
    rng = np.random.default_rng(0)
    zip_path = tmp_path / "radar_dataset.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for stem in ("user4_e1", "user9_e1_fast", "user10_e1"):
            inputs = rng.standard_normal((N_REC, 100, 3, 32, 64)).astype(np.float32)
            targets = np.zeros((N_REC, 100), dtype=np.int32)
            targets[:, 40:48] = [[1], [2]]
            if stem in split_stems:
                targets[1, 70:73] = 2
            npz = tmp_path / f"{stem}.npz"
            np.savez(npz, inputs=inputs, targets=targets)
            z.write(npz, f"fulldataset/{stem}.npz")
    return zip_path


def statuses(out):
    files = json.loads((out / "index.json").read_text())["files"]
    return {stem: e["status"] for stem, e in files.items()}


def test_full_run_builds_nominal_then_anomalous_and_never_touches_test_users(tmp_path, monkeypatch):
    monkeypatch.delenv("RADARSENSE_ALLOW_TEST", raising=False)
    zip_path, out = make_zip(tmp_path), tmp_path / "processed"
    assert main(["--zip", str(zip_path), "--out", str(out)]) == 0
    assert statuses(out) == {"user4_e1": "complete", "user9_e1_fast": "complete"}
    assert not list(out.glob("user10*"))


def test_nominal_only(tmp_path):
    zip_path, out = make_zip(tmp_path), tmp_path / "processed"
    assert main(["--zip", str(zip_path), "--out", str(out), "--nominal-only"]) == 0
    assert statuses(out) == {"user4_e1": "complete"}


def test_violation_is_skipped_recorded_and_exit_code_is_1(tmp_path, capsys):
    zip_path, out = make_zip(tmp_path, split_stems=("user9_e1_fast",)), tmp_path / "processed"
    assert main(["--zip", str(zip_path), "--out", str(out)]) == 1
    assert statuses(out) == {"user4_e1": "complete", "user9_e1_fast": "violation"}
    assert not (out / "user9_e1_fast.npy").exists()
    assert "user9_e1_fast" in capsys.readouterr().out
    # A second run keeps reporting the violation and does not touch the finished file.
    assert main(["--zip", str(zip_path), "--out", str(out)]) == 1


def test_only_rejects_unknown_and_test_members(tmp_path):
    zip_path, out = make_zip(tmp_path), tmp_path / "processed"
    for stem in ("user10_e1", "user99_e1"):
        try:
            main(["--zip", str(zip_path), "--out", str(out), "--only", stem])
        except SystemExit as e:
            assert e.code not in (0, None)
        else:
            raise AssertionError(f"{stem} was accepted")
    assert not out.exists()
