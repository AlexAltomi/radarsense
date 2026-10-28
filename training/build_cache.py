"""Build the processed cache in data/processed from the zip file, one member at a time. The build can be resumed.

Usage: python training/build_cache.py [--nominal-only] [--only STEM] [--zip PATH] [--out DIR]

Nominal members are processed first and anomalous members afterwards. The test users user10 and user12 are never
included. A member with a span violation, which means that a recording does not contain exactly one contiguous
gesture, is skipped and recorded in index.json with the status "violation". The run then continues with the other
members and exits with code 1 and a summary. The exit code is 2 if a cached member was built with different DSP
constants. Code that reads the cache must use a member only if its index status is "complete" (see
dsp/radarsense_dsp/cache.py).
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dsp"))

from radarsense_dsp.cache import (
    StaleConstantsError,
    build_file,
    estimate_cache_bytes,
    load_constants,
    load_index,
    plan_members,
)
from radarsense_dsp.io import list_members

ZIP = ROOT / "data" / "radar_dataset.zip"
OUT = ROOT / "data" / "processed"
HEADROOM = 1.1


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def build_commit() -> str:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return git("rev-parse", "HEAD") + ("+dirty" if dirty else "")


def existing_parent(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal-only", action="store_true", help="skip the anomalous members (fast, slow and wrist)")
    ap.add_argument("--only", metavar="STEM", help="build only this member, given as a stem such as user4_e1")
    ap.add_argument("--zip", type=Path, default=ZIP, help="the dataset zip file")
    ap.add_argument("--out", type=Path, default=OUT, help="the cache directory")
    args = ap.parse_args(argv)

    members = plan_members(list_members(args.zip), args.nominal_only)
    if args.only:
        members = [m for m in members if Path(m.name).stem == args.only]
        if not members:
            sys.exit(f"{args.only!r} is not a cacheable member. It is unknown, belongs to a test user, or is excluded by --nominal-only.")

    constants = load_constants()
    known = load_index(args.out)["files"]
    pending = [m for m in members if Path(m.name).stem not in known]
    need = estimate_cache_bytes(args.zip, pending)
    free = shutil.disk_usage(existing_parent(args.out)).free
    print(f"The plan has {len(members)} members, and {len(pending)} of them are not yet cached. About "
          f"{need / 1e9:.1f} GB of disk space are needed, and {free / 1e9:.1f} GB are free.", flush=True)
    if free < HEADROOM * need:
        sys.exit("There is not enough free disk space for the cache. Free some space or choose another --out directory.")
    commit = build_commit()

    built, skipped, violations = [], [], []
    for i, m in enumerate(members, 1):
        stem = Path(m.name).stem
        print(f"Member {i} of {len(members)}: starting {stem}.", flush=True)
        t0 = time.perf_counter()
        try:
            result = build_file(args.zip, m, args.out, constants, commit)
        except StaleConstantsError as e:
            print("Stopped: this member was cached with different DSP constants.")
            print(e, file=sys.stderr)
            return 2
        entry = result.entry
        if result.outcome == "built":
            built.append(stem)
            print(f"Built {stem} with {entry['n']} recordings in {time.perf_counter() - t0:.0f} s.", flush=True)
        elif result.outcome == "skipped":
            skipped.append(stem)
            print(f"{stem} is already complete and was skipped.", flush=True)
        else:
            violations.append((stem, entry["violations"]))
            print(
                f"{stem} has span violations {entry['violations']['kinds']}, and the first ones are "
                f"{entry['violations']['first']}. The member was not cached.",
                flush=True,
            )

    print(f"\nSummary: built {len(built)}, skipped {len(skipped)}, members with span violations {len(violations)}.")
    for stem, v in violations:
        print(f"  {stem} has {v['count']} recordings with violations {v['kinds']}, and the first ones are {v['first']}.")
    if violations:
        print("Members with violations are not cached. Their index status is 'violation', and downstream code must not read them.")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
