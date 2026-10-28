"""Compute the dB clip range of channel 0 and write it to dsp/radarsense_dsp/constants.json.

Only the nominal files of the train users are used (see "Data rules" in docs/project-rules.md).

Usage:
  python training/compute_constants.py            Stream the missing files and print the numbers. Nothing is written.
  python training/compute_constants.py --write    Write constants.json from a complete cache.
The option --cache PATH sets the histogram cache. It defaults to data/tmp/constants_cache.npz, which is
D:/radarsense-data/tmp/constants_cache.npz because data/ is a junction.

For each nominal file of a train user (listed in training/splits.json) the script keeps up to KEEP recordings per
label value. It adds the raw dB values of all 100 frames, unclipped and averaged over the antennas, to a histogram
with 0.01 dB bins and caches that histogram per file. The option --write therefore never reads the dataset again.
The values db_min and db_max are the DB_MIN_Q and DB_MAX_Q percentiles of a mixture in which every user has equal
weight. Without this weighting user1, who has six nominal files, would dominate.
The option --write refuses to run when tracked files are modified, so that the recorded git commit is the code that
produced the numbers.
"""
import argparse
import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dsp"))

from radarsense_dsp.frames import magnitude_db, range_doppler_spectra
from radarsense_dsp.io import TEST_USERS, iter_recordings, list_members

ZIP = ROOT / "data" / "radar_dataset.zip"
SPLITS = ROOT / "training" / "splits.json"
OUT = ROOT / "dsp" / "radarsense_dsp" / "constants.json"
DEFAULT_CACHE = ROOT / "data" / "tmp" / "constants_cache.npz"  # data/ is a junction to D:\radarsense-data.
LABELS = (1, 2, 3, 6, 7)
KEEP = 40
DB_MIN_Q, DB_MAX_Q = 50.0, 99.9
PAIR_A, PAIR_B = (0, 2), (1, 2)
BIN = 0.01
LO = -250.0
NBINS = 40000  # The bins cover -250 to 150 dB. The floor that comes from the epsilon term is -240 dB.


def db_histogram(db: np.ndarray) -> np.ndarray:
    idx = np.clip(np.floor((np.asarray(db).ravel() - LO) / BIN).astype(np.int64), 0, NBINS - 1)
    return np.bincount(idx, minlength=NBINS)


def percentile_from_counts(counts: np.ndarray, q: float) -> float:
    """Return the centre of the first bin whose cumulative share reaches q percent."""
    cum = np.cumsum(counts) / counts.sum()
    return LO + (int(np.searchsorted(cum, q / 100.0)) + 0.5) * BIN


def balanced_percentile(hists: dict[str, np.ndarray], q: float) -> float:
    """Return the percentile of the mixture in which every histogram (every user) has equal weight."""
    mix = np.mean([h / h.sum() for h in hists.values()], axis=0)
    return percentile_from_counts(mix, q)


def train_nominal_members() -> list:
    train = {int(u.removeprefix("user")) for u in json.loads(SPLITS.read_text(encoding="utf-8"))["train"]}
    assert not train & TEST_USERS, "splits.json lists a test user as a train user. Fix the split before computing constants."
    return [m for m in list_members(ZIP) if m.user in train and m.anomaly is None]


def select_members(members: list, only: str | None) -> list:
    """Return all members, or only the one whose name or stem equals `only`. A test user is never in the list."""
    if only is None:
        return members
    picked = [m for m in members if only in (m.name, Path(m.name).stem)]
    if not picked:
        sys.exit(f"{only!r} is not a nominal member of a train user. Choose one of {[Path(m.name).stem for m in members]}.")
    return picked


def process_file(member) -> tuple[np.ndarray, dict[int, int]]:
    hist = np.zeros(NBINS, dtype=np.int64)
    kept: dict[int, int] = dict.fromkeys(LABELS, 0)
    gen = iter_recordings(ZIP, member)
    for raw, targets in gen:
        nz = np.flatnonzero(targets)
        label = int(targets[nz[0]]) if len(nz) else 0
        if label not in kept or kept[label] >= KEEP:
            continue
        hist += db_histogram(magnitude_db(range_doppler_spectra(raw)))
        kept[label] += 1
        if all(v >= KEEP for v in kept.values()):
            break
    gen.close()
    return hist, kept


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def per_user(cache: dict[str, np.ndarray], members: list) -> dict[int, np.ndarray]:
    users: dict[int, np.ndarray] = {}
    for m in members:
        h = cache[f"hist__{Path(m.name).stem}"]
        users[m.user] = users.get(m.user, np.zeros(NBINS, dtype=np.int64)) + h
    return users


def constants_dict(users: dict[int, np.ndarray], members: list, commit: str) -> dict:
    hists = {str(u): h for u, h in users.items()}
    return {
        "db_min": round(balanced_percentile(hists, DB_MIN_Q), 2),
        "db_max": round(balanced_percentile(hists, DB_MAX_Q), 2),
        "percentiles": {"db_min": DB_MIN_Q, "db_max": DB_MAX_Q},
        "weighting": "equal per user",
        "pair_a": list(PAIR_A),
        "pair_b": list(PAIR_B),
        "source_users": sorted(users),
        "source_files": [Path(m.name).stem for m in members],
        "sampling": f"nominal files only, up to {KEEP} recordings per label per file, all 100 frames, "
        f"raw antenna-averaged dB in {BIN} dB bins",
        "per_user": {
            str(u): {
                "db_min": round(percentile_from_counts(h, DB_MIN_Q), 2),
                "db_max": round(percentile_from_counts(h, DB_MAX_Q), 2),
            }
            for u, h in sorted(users.items())
        },
        "script": "training/compute_constants.py",
        "git_commit": commit,
    }


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help=f"file that caches the histograms (default {DEFAULT_CACHE})")
    ap.add_argument("--write", action="store_true", help="write constants.json from a complete cache")
    ap.add_argument("--only", metavar="MEMBER", help="process only this nominal member of a train user, given as a stem such as user8_e1")
    args = ap.parse_args()
    if args.only and args.write:
        sys.exit("The option --write needs the complete cache. Run again without --only.")

    members = train_nominal_members()
    cache = load_cache(args.cache)
    if args.write and any(f"hist__{Path(m.name).stem}" not in cache for m in members):
        sys.exit("The histogram cache is incomplete. Run once without --write to fill it.")
    tracemalloc.start()  # psutil is not installed. tracemalloc sees Python and NumPy allocations, but not zlib or the operating system file cache.
    for m in select_members(members, args.only):
        stem = Path(m.name).stem
        if f"hist__{stem}" in cache:
            print(f"Skipping {stem}, which is already cached.", flush=True)
            continue
        print(f"Reading {stem} one recording at a time.", flush=True)
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        hist, kept = process_file(m)
        peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
        cache[f"hist__{stem}"] = hist
        cache[f"kept__{stem}"] = np.array([kept[label] for label in LABELS])
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, **cache)
        print(f"Finished {stem}: kept per label {dict(zip(LABELS, cache[f'kept__{stem}'].tolist(), strict=True))}, "
              f"took {time.perf_counter() - t0:.0f} s, peak traced memory {peak_mb:.0f} MB.", flush=True)
    if args.only:
        print("Partial run (--only): the percentiles are computed only from a complete cache.")
        return

    users = per_user(cache, members)
    db_min = balanced_percentile({str(u): h for u, h in users.items()}, DB_MIN_Q)
    db_max = balanced_percentile({str(u): h for u, h in users.items()}, DB_MAX_Q)
    pooled = np.sum(list(users.values()), axis=0)
    print(f"\nUsers: {sorted(users)}. Files: {[Path(m.name).stem for m in members]}.")
    print(f"Raw dB percentiles per user ({DB_MIN_Q:g}th and {DB_MAX_Q:g}th):")
    for u, h in sorted(users.items()):
        print(f"  user{u}: {percentile_from_counts(h, DB_MIN_Q):.2f} / {percentile_from_counts(h, DB_MAX_Q):.2f}")
    print(f"Pooled over all recordings, where user1 dominates, for comparison: {percentile_from_counts(pooled, DB_MIN_Q):.2f} and "
          f"{percentile_from_counts(pooled, DB_MAX_Q):.2f}.")
    print(f"Values used, with every train user weighted equally: db_min = {db_min:.2f} dB, db_max = {db_max:.2f} dB.")

    if not args.write:
        return
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("Tracked files are modified. Commit them first, so that the recorded git commit is accurate.")
    OUT.write_text(json.dumps(constants_dict(users, members, git("rev-parse", "HEAD")), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}.")


if __name__ == "__main__":
    main()
