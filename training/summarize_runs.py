"""Summarize training runs in one markdown table.

Usage: python training/summarize_runs.py RUN [RUN ...] --out training/results/cnn-seeds.md
           [--seed-group RUN [RUN ...]] [--label RUN=text ...]

Reads training/runs/<run>/log.csv and config.json only. Nothing is recomputed and the data loader is not used. Each run
gets one row with its seed, the settings that differ from the train.py defaults (max-epochs and weight-decay), its commit
stamp, the best epoch with its validation macro-F1 and the last epoch with its validation macro-F1. If the log has
per-user accuracy columns (any column whose name contains `user<N>`), they are added for the best and the last epoch.
For the runs of --seed-group the table ends with a mean row and a range (min to max) row.

The best epoch is the first epoch with the highest validation macro-F1, which is the epoch that train.py checkpoints.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

from build_cache import build_commit

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "training" / "runs"
DEFAULTS = {"max_epochs": 30, "weight_decay": 0.01}  # The defaults of train.py (a test keeps them in sync).
USER_COLUMN = re.compile(r"user\d+")


def load_run(run: str, runs_dir: Path = RUNS) -> dict:
    """Read one run. Return its settings and the best and the last epoch of its log."""
    d = runs_dir / run
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    with open(d / "log.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"The log of {run} has no epochs.")
    f1 = [float(r["val_macro_f1"]) for r in rows]
    best = f1.index(max(f1))
    user_cols = [c for c in rows[0] if USER_COLUMN.search(c)]
    return {
        "run": run,
        "seed": config["seed"],
        "max_epochs": config["max_epochs"],
        "weight_decay": config["weight_decay"],
        "commit": rows[0].get("commit") or config.get("commit", ""),
        "cache": rows[0].get("cache") or config.get("cache_id", ""),
        "epochs_run": len(rows),
        "best": rows[best],
        "last": rows[-1],
        "best_f1": f1[best],
        "last_f1": f1[-1],
        "user_cols": user_cols,
    }


def differences(run: dict) -> str:
    out = []
    if run["max_epochs"] != DEFAULTS["max_epochs"]:
        out.append(f"max-epochs {run['max_epochs']}")
    if run["weight_decay"] != DEFAULTS["weight_decay"]:
        out.append(f"weight-decay {run['weight_decay']:g}")
    return ", ".join(out) or "none"


def stamp(run: dict) -> str:
    return f"`{run['commit'][:7]}` · cache `{run['cache']}`"


def table(runs: list[dict], labels: dict[str, str], seed_group: list[str]) -> list[str]:
    user_cols = [c for c in runs[0]["user_cols"] if all(c in r["user_cols"] for r in runs)]
    head = ["run", "note", "seed", "differs from the defaults", "commit stamp", "epochs run", "best epoch", "best val macro-F1",
            "last epoch", "last val macro-F1"]
    head += [f"{c} (best)" for c in user_cols] + [f"{c} (last)" for c in user_cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        cells = [
            f"`{r['run']}`", labels.get(r["run"], ""), str(r["seed"]), differences(r), stamp(r), str(r["epochs_run"]),
            r["best"]["epoch"], f"{r['best_f1']:.4f}", r["last"]["epoch"], f"{r['last_f1']:.4f}",
        ]
        cells += [f"{float(r['best'][c]):.4f}" for c in user_cols] + [f"{float(r['last'][c]):.4f}" for c in user_cols]
        lines.append("| " + " | ".join(cells) + " |")
    group = [r for r in runs if r["run"] in seed_group]
    if len(group) > 1:
        pad = [""] * len(user_cols) * 2
        for name, fn in (("mean", lambda v: f"{sum(v) / len(v):.4f}"), ("range", lambda v: f"{min(v):.4f} to {max(v):.4f}")):
            cells = [f"**{name}**", f"seeds {', '.join(str(r['seed']) for r in group)}", "", "", "", "", "",
                     fn([r["best_f1"] for r in group]), "", fn([r["last_f1"] for r in group])]
            lines.append("| " + " | ".join(cells + pad) + " |")
    return lines


def render(runs: list[dict], labels: dict[str, str], seed_group: list[str], produced_at: str) -> str:
    intro = (
        f"This table was produced at commit `{produced_at}` from the logs of the runs "
        "(`training/runs/<run>/log.csv` and `config.json`). Nothing was recomputed."
    )
    out = ["# cnn-v1 training runs", "", intro, ""]
    out += table(runs, labels, seed_group)
    note = (
        "The best epoch is the first epoch with the highest validation macro-F1 (the checkpointed epoch). "
        "The validation users user3 and user5 also choose the epoch, so the best value is optimistic. "
        "The last epoch is where training stopped, by early stopping or at max-epochs."
    )
    if len([r for r in runs if r["run"] in seed_group]) > 1:
        note += " The mean and range rows cover the seed group only."
    out += ["", note]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", metavar="RUN")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-group", nargs="+", default=[], metavar="RUN", help="runs for the mean and range rows")
    ap.add_argument("--label", action="append", default=[], metavar="RUN=text", help="a note for a run (repeatable)")
    ap.add_argument("--runs-dir", type=Path, default=RUNS)
    args = ap.parse_args(argv)

    labels = {}
    for item in args.label:
        run, sep, text = item.partition("=")
        if not sep or run not in args.runs:
            print(f"ERROR: --label needs RUN=text with RUN among the runs, got {item!r}.", file=sys.stderr)
            return 2
        labels[run] = text
    unknown = [r for r in args.seed_group if r not in args.runs]
    missing = [r for r in args.runs if not (args.runs_dir / r / "log.csv").exists()]
    if unknown or missing:
        print(f"ERROR: --seed-group runs must be among the runs ({unknown}); no log.csv for {missing}.", file=sys.stderr)
        return 2

    runs = [load_run(r, args.runs_dir) for r in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(runs, labels, args.seed_group, build_commit()), encoding="utf-8")
    print(f"Wrote {args.out} with {len(runs)} runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
