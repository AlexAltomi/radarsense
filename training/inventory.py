"""Dataset inventory: N, nominal/anomaly type and label value counts per non-test member.

Usage: python training/inventory.py [--only SUBSTRING] [--out PATH]
  --only  restrict to members whose name contains SUBSTRING (prints only; writes no file unless --out is given)
  --out   markdown output (default training/dataset_inventory.md, written only for a full run)

N and label counts come from the .npy headers and the small targets array; no inputs data is read, except the
single first recording of user4_e1 that is read once to time the stream start.
Test users (user10/user12) are listed as locked and never opened.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dsp"))

from radarsense_dsp.io import TEST_USERS, iter_recordings, list_members, read_meta

ZIP = ROOT / "data" / "radar_dataset.zip"
TIMED_MEMBER = "user4_e1"
LOCKED = "locked until Week 11"


def time_first_recording(member_name: str) -> float:
    t0 = time.perf_counter()
    gen = iter_recordings(ZIP, member_name)
    next(gen)
    elapsed = time.perf_counter() - t0
    gen.close()
    return elapsed


def label_counts(targets: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(targets, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts, strict=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    members = list_members(ZIP)
    if args.only:
        members = [m for m in members if args.only in m.name]
    if not members:
        sys.exit("no members match")

    rows = []
    for m in members:
        stem = Path(m.name).stem
        kind = m.anomaly or "nominal"
        if m.user in TEST_USERS:
            print(f"{stem}: {LOCKED}")
            rows.append((stem, m.user, m.session, kind, LOCKED, ""))
            continue
        t0 = time.perf_counter()
        n, targets = read_meta(ZIP, m)
        elapsed = time.perf_counter() - t0
        counts = label_counts(targets)
        print(f"{stem}: N={n} {kind} labels={counts} (meta read {elapsed:.1f} s)")
        rows.append((stem, m.user, m.session, kind, str(n), " ".join(f"{k}:{v}" for k, v in counts.items())))
        if stem == TIMED_MEMBER:
            print(f"time to first recording of {stem}: {time_first_recording(m.name):.1f} s")

    out = args.out or (None if args.only else ROOT / "training" / "dataset_inventory.md")
    if out is None:
        return
    lines = [
        "# Dataset inventory",
        "",
        "Counts only. Label counts are over all frames of all recordings in the member (`value:frames`).",
        "",
        "| member | user | session | type | N | label counts |",
        "|---|---|---|---|---|---|",
        *(f"| {s} | {u} | e{e} | {k} | {n} | {c} |" for s, u, e, k, n, c in rows),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
