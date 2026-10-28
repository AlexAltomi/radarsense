"""Build the processed cache: raw radar recordings become float16 range-Doppler and phase frames on disk.

One member (one .npz file of the dataset) is processed at a time. The output directory contains these files:

  <stem>.npy          float16 frames with shape (N, 100, 3, 32, 32)
  <stem>.targets.npy  int8 targets with shape (N, 100)
  index.json          a description of every member, in the form {"version": 1, "files": {stem: entry}}

Code that reads the cache must use a file only if its index entry has the status "complete".

A member is first checked for span violations using its targets alone. Every recording must contain exactly one
contiguous gesture with a single label value. A member with a violation is skipped, recorded with the status
"violation" and never streamed. Otherwise the frames are written to `<stem>.npy.partial`. The targets file is renamed
first and the frames file last, and the index entry is written after both. If `<stem>.npy` exists, the member is
therefore complete. The loader in `radarsense_dsp.io` refuses the test users, so they never enter the cache.
"""
import json
import os
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radarsense_dsp.frames import process_frames
from radarsense_dsp.io import TEST_USERS, Member, open_recordings
from radarsense_dsp.labels import analyze_targets, span_violations

CONSTANTS_PATH = Path(__file__).with_name("constants.json")
INDEX_NAME = "index.json"
OUT_SHAPE = (100, 3, 32, 32)
RAW_RECORD_BYTES = 100 * 3 * 32 * 64 * 4
OUT_RECORD_BYTES = 100 * 3 * 32 * 32 * 2
TARGET_RECORD_BYTES = 100
MAX_LISTED_VIOLATIONS = 10


class StaleConstantsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Constants:
    db_min: float
    db_max: float
    pair_a: tuple[int, int]
    pair_b: tuple[int, int]
    source_commit: str

    def as_dict(self) -> dict:
        return {
            "db_min": self.db_min,
            "db_max": self.db_max,
            "pair_a": list(self.pair_a),
            "pair_b": list(self.pair_b),
            "source_commit": self.source_commit,
        }


@dataclass(frozen=True)
class BuildResult:
    outcome: str  # One of "built", "skipped" or "violation".
    entry: dict


def load_constants(path: str | Path = CONSTANTS_PATH) -> Constants:
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    return Constants(
        c["db_min"],
        c["db_max"],
        (c["pair_a"][0], c["pair_a"][1]),
        (c["pair_b"][0], c["pair_b"][1]),
        c["git_commit"],
    )


def load_index(out_dir: Path) -> dict:
    path = out_dir / INDEX_NAME
    if not path.exists():
        return {"version": 1, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_index(out_dir: Path, index: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (INDEX_NAME + ".tmp")
    tmp.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out_dir / INDEX_NAME)


def plan_members(members: list[Member], nominal_only: bool = False) -> list[Member]:
    """Order the members for building: nominal files first, then anomalous ones. Test users are never included."""
    usable = [m for m in members if m.user not in TEST_USERS]
    nominal = sorted((m for m in usable if m.anomaly is None), key=lambda m: (m.user, m.session))
    if nominal_only:
        return nominal
    anomalous = sorted((m for m in usable if m.anomaly is not None), key=lambda m: (m.user, m.session, m.anomaly or ""))
    return nominal + anomalous


def estimate_cache_bytes(zip_path: str | Path, members: list[Member]) -> int:
    """Estimate the disk space in bytes that these members need. Only the uncompressed member sizes are read."""
    with zipfile.ZipFile(zip_path) as z:
        sizes = {i.filename: i.file_size for i in z.infolist()}
    return sum(
        round(sizes[m.name] / RAW_RECORD_BYTES) * (OUT_RECORD_BYTES + TARGET_RECORD_BYTES) for m in members
    )


def _base_entry(member: Member, targets: np.ndarray, constants: Constants, build_commit: str, violations: list) -> dict:
    analysis = analyze_targets(targets)
    return {
        "user": member.user,
        "session": member.session,
        "anomaly": member.anomaly,
        "n": len(targets),
        "constants": constants.as_dict(),
        "build_commit": build_commit,
        "labelled_hist": analysis["labelled_hist"],
        "span_hist": analysis["span_hist"],
        "start": analysis["start"],
        "violations": {
            "count": len(violations),
            "kinds": dict(Counter(kind for _, kind in violations)),
            "first": [[i, kind] for i, kind in violations[:MAX_LISTED_VIOLATIONS]],
        },
    }


def _record(out_dir: Path, index: dict, stem: str, entry: dict, t0: float) -> None:
    entry["seconds"] = round(time.perf_counter() - t0, 1)
    index["files"][stem] = entry
    _save_index(out_dir, index)


def build_file(
    zip_path: str | Path, member: Member, out_dir: str | Path, constants: Constants, build_commit: str
) -> BuildResult:
    out_dir = Path(out_dir)
    stem = Path(member.name).stem
    frames_path = out_dir / f"{stem}.npy"
    targets_path = out_dir / f"{stem}.targets.npy"
    index = load_index(out_dir)

    entry = index["files"].get(stem)
    if entry is not None:
        if entry["status"] == "violation":
            return BuildResult("violation", entry)
        if entry["status"] == "complete" and frames_path.exists() and targets_path.exists():
            if entry["constants"] != constants.as_dict():
                raise StaleConstantsError(
                    f"{stem} was cached with the constants {entry['constants']}, but the current constants are "
                    f"{constants.as_dict()}. Delete {out_dir} (or the files and index entry of this member) "
                    "and run the build again."
                )
            return BuildResult("skipped", entry)

    t0 = time.perf_counter()
    with open_recordings(zip_path, member) as (targets, recordings):
        violations = span_violations(targets)
        entry = _base_entry(member, targets, constants, build_commit, violations)
        if violations:
            entry["status"] = "violation"
            _record(out_dir, index, stem, entry, t0)
            return BuildResult("violation", entry)
        if targets.min() < 0 or targets.max() > np.iinfo(np.int8).max:
            raise ValueError(f"The target values of {stem} do not fit into int8, so they cannot be cached.")

        n = len(targets)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / f"{stem}.npy.partial"
        with open(partial, "wb") as f:
            np.lib.format.write_array_header_1_0(f, {"descr": "<f2", "fortran_order": False, "shape": (n, *OUT_SHAPE)})
            header_len = f.tell()
            for rec in recordings:
                frames = process_frames(rec, constants.db_min, constants.db_max, constants.pair_a, constants.pair_b)
                if not np.isfinite(frames).all():
                    raise FloatingPointError(f"The processed frames of {stem} contain NaN or infinite values. Check the raw recording.")
                f.write(frames.tobytes())
            f.flush()
            os.fsync(f.fileno())
            if f.tell() != header_len + n * OUT_RECORD_BYTES:
                raise OSError(f"Wrote {f.tell()} bytes for {stem}, but expected {header_len + n * OUT_RECORD_BYTES}. Check the free disk space.")

        targets_partial = out_dir / f"{stem}.targets.npy.partial"
        with open(targets_partial, "wb") as tf:
            np.save(tf, targets.astype(np.int8))
        os.replace(targets_partial, targets_path)
        os.replace(partial, frames_path)

    entry.update(status="complete", shape=[n, *OUT_SHAPE], dtype="float16")
    _record(out_dir, index, stem, entry, t0)
    return BuildResult("built", entry)
