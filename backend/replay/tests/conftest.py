"""A small synthetic cache. Frame f of every recording has channel 0 equal to f / 99, so a decoded frame tells which
frame of the recording it came from."""
import json
from pathlib import Path

import numpy as np
import pytest

SPLITS = {
    "train": ["user1", "user2", "user4"],
    "val": ["user3", "user5"],
    "test": ["user10", "user12"],
}

# stem: (user, session, anomaly, status, [(label, s, length), ...])
MEMBERS = {
    "user1_e1": (1, 1, None, "complete", [(2, 40, 8), (7, 30, 8)]),
    "user2_e1": (2, 1, None, "complete", [(3, 40, 8)]),
    "user3_e1": (3, 1, None, "complete", [(2, 40, 8), (3, 45, 8), (6, 35, 8), (1, 50, 8)]),
    "user5_e1": (5, 1, None, "complete", [(7, 40, 10), (6, 55, 10), (1, 10, 80)]),
    "user4_e1": (4, 1, None, "violation", [(2, 40, 8)]),  # Never read: the status is not "complete".
    "user3_e1_fast": (3, 1, "fast", "complete", [(2, 40, 8)]),  # Anomalous files are not streamed.
}
FRAME_VALUES = np.arange(100, dtype=np.float32) / 99


def _targets(recs):
    t = np.zeros((len(recs), 100), dtype=np.int8)
    for i, (label, s, length) in enumerate(recs):
        t[i, s : s + length] = label
    return t


def write_cache(cache_dir: Path, members: dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for stem, (user, session, anomaly, status, recs) in members.items():
        files[stem] = {"user": user, "session": session, "anomaly": anomaly, "status": status, "n": len(recs)}
        if status != "complete":
            continue
        frames = np.zeros((len(recs), 100, 3, 32, 32), dtype=np.float16)
        frames[:, :, 0] = FRAME_VALUES[None, :, None, None]
        np.save(cache_dir / f"{stem}.npy", frames)
        np.save(cache_dir / f"{stem}.targets.npy", _targets(recs))
    (cache_dir / "index.json").write_text(json.dumps({"version": 1, "files": files}), encoding="utf-8")
    return cache_dir


@pytest.fixture(scope="session")
def splits_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("splits") / "splits.json"
    path.write_text(json.dumps(SPLITS), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    return write_cache(tmp_path_factory.mktemp("cache"), MEMBERS)
