"""Build the processed cache (data/processed) from the zip, one member at a time. Resumable.

Usage: python training/build_cache.py [--nominal-only] [--only STEM] [--zip PATH] [--out DIR]
Nominal members are processed first, then anomalous ones. user10/user12 are never included.
A member with a span violation (not exactly one contiguous gesture per recording) is skipped and recorded in
index.json with status "violation"; the run continues and exits 1 with a summary. Exit 2: stale constants.
Downstream code must read a cached member only if its index status is "complete" (see dsp/radarsense_dsp/cache.py).
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
    ap.add_argument("--nominal-only", action="store_true", help="skip the anomalous (fast/slow/wrist) members")
    ap.add_argument("--only", metavar="STEM", help="build just this member, e.g. user4_e1")
    ap.add_argument("--zip", type=Path, default=ZIP)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    members = plan_members(list_members(args.zip), args.nominal_only)
    if args.only:
        members = [m for m in members if Path(m.name).stem == args.only]
        if not members:
            sys.exit(f"{args.only!r} is not a cacheable member (unknown, a test user, or excluded by --nominal-only)")

    constants = load_constants()
    known = load_index(args.out)["files"]
    pending = [m for m in members if Path(m.name).stem not in known]
    need = estimate_cache_bytes(args.zip, pending)
    free = shutil.disk_usage(existing_parent(args.out)).free
    print(f"{len(members)} members, {len(pending)} not yet cached; need about {need / 1e9:.1f} GB, "
          f"{free / 1e9:.1f} GB free", flush=True)
    if free < HEADROOM * need:
        sys.exit("not enough free disk space")
    commit = build_commit()

    built, skipped, violations = [], [], []
    for i, m in enumerate(members, 1):
        stem = Path(m.name).stem
        print(f"[{i}/{len(members)}] {stem}: ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            result = build_file(args.zip, m, args.out, constants, commit)
        except StaleConstantsError as e:
            print("STALE CONSTANTS")
            print(e, file=sys.stderr)
            return 2
        entry = result.entry
        if result.outcome == "built":
            built.append(stem)
            print(f"built {entry['n']} recordings in {time.perf_counter() - t0:.0f} s", flush=True)
        elif result.outcome == "skipped":
            skipped.append(stem)
            print("already complete, skipped", flush=True)
        else:
            violations.append((stem, entry["violations"]))
            print(f"SPAN VIOLATION {entry['violations']['kinds']}, first {entry['violations']['first']}", flush=True)

    print(f"\nsummary: built {len(built)}, skipped {len(skipped)}, span violations {len(violations)}")
    for stem, v in violations:
        print(f"  VIOLATION {stem}: {v['count']} recordings {v['kinds']}, first {v['first']}")
    if violations:
        print("violating members are NOT cached (index status 'violation'); downstream must not read them")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
