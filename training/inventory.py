"""Write the dataset inventory: for every member, the number of recordings, the type and the label counts.

Usage: python training/inventory.py [--only SUBSTRING] [--out PATH]
  --only  Restrict the run to members whose name contains SUBSTRING. The script then only prints and writes no file,
          unless --out is given.
  --out   The markdown output file. It defaults to training/dataset_inventory.md and is written only for a full run.

The number of recordings and the label counts come from the .npy headers and the small targets array. No inputs
data is read, except for the first recording of user4_e1, which is read once to time how long a stream takes to
start. The test users user10 and user12 are listed as locked and are never opened.
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

INTRO = [
    "# Dataset inventory",
    "",
    ("This file describes the Infineon 60 GHz radar gesture dataset that RadarSense uses. It contains counts only and "
    "no radar data. The scripts in `training/` generate it. `inventory.py` writes the glossary and the member list, "
    "`label_structure.py` and `infer_labels_antennas.py` add their own sections, and the section on constants is "
    "kept by hand from the output of `compute_constants.py`."),
    "",
    "## Glossary",
    "",
    ("- **Member**: one `.npz` file of the dataset, for example `user4_e1` or `user9_e1_fast`. The name gives the user, "
    "the session and, for anomalous members, the type."),
    "- **Recording**: one gesture example inside a member. A member holds N recordings.",
    "- **Frame**: one radar snapshot. Every recording consists of 100 frames, and every frame has a label.",
    "- **Nominal**: a member whose gestures were performed in the normal way.",
    ("- **Anomalous**: a member that deviates from the normal way, namely the fast, slow and wrist variants. These "
    "members are used only to test robustness."),
    ("- **Labelled span**: the run of consecutive frames in a recording that carry a non-zero label. Label 0 means "
    "that no gesture is happening."),
    ("- **Train, validation and test users**: the split of the people in the dataset, frozen in "
    "`training/splits.json`. The train users are user1, user2, user4, user6, user7, user8, user9 and user11. The "
    "validation users are user3 and user5. The test users are user10 and user12, who stay locked until Week 11."),
    "",
    "## Members",
    "",
    ("This table lists every member of the dataset with its user, session, type, number of recordings N and label "
    "counts. The label counts cover all frames of all recordings in the member and are written as `value:frames`. "
    "The members of the test users are listed as locked, because their data is not opened."),
    "",
]


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
    ap.add_argument("--only", help="restrict the run to members whose name contains this text")
    ap.add_argument("--out", type=Path, help="markdown output file (default training/dataset_inventory.md)")
    args = ap.parse_args()

    members = list_members(ZIP)
    if args.only:
        members = [m for m in members if args.only in m.name]
    if not members:
        sys.exit("No member name contains the --only text. Check the spelling, for example --only user4_e1.")

    rows = []
    for m in members:
        stem = Path(m.name).stem
        kind = m.anomaly or "nominal"
        if m.user in TEST_USERS:
            print(f"The member {stem} is {LOCKED} and was not opened.")
            rows.append((stem, m.user, m.session, kind, LOCKED, ""))
            continue
        t0 = time.perf_counter()
        n, targets = read_meta(ZIP, m)
        elapsed = time.perf_counter() - t0
        counts = label_counts(targets)
        print(f"Read {stem}: {n} recordings, type {kind}, label counts {counts}. Reading the headers took {elapsed:.1f} s.")
        rows.append((stem, m.user, m.session, kind, str(n), " ".join(f"{k}:{v}" for k, v in counts.items())))
        if stem == TIMED_MEMBER:
            print(f"Reading the first recording of {stem} took {time_first_recording(m.name):.1f} s.")

    out = args.out or (None if args.only else ROOT / "training" / "dataset_inventory.md")
    if out is None:
        return
    lines = [
        *INTRO,
        "| member | user | session | type | N | label counts |",
        "|---|---|---|---|---|---|",
        *(f"| {s} | {u} | e{e} | {k} | {n} | {c} |" for s, u, e, k, n, c in rows),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}.")


if __name__ == "__main__":
    main()
