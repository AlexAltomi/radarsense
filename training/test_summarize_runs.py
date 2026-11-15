"""Tests for summarize_runs.py on synthetic logs."""
import csv
import json
import re
from pathlib import Path

import pytest
import summarize_runs as sr

FIELDS = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_accuracy", "lr", "epoch_seconds", "model", "commit", "cache"]


def make_run(runs_dir: Path, name: str, f1: list[float], seed=0, max_epochs=30, wd=0.01, extra=None):
    d = runs_dir / name
    d.mkdir(parents=True)
    fields = FIELDS + list(extra or {})
    with open(d / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for i, v in enumerate(f1, 1):
            row = {
                "epoch": i, "train_loss": 0.1, "val_loss": 0.2, "val_macro_f1": v, "val_accuracy": v, "lr": 0.001,
                "epoch_seconds": 1.0, "model": "cnn-v1", "commit": "abcdef0123456789", "cache": "cafe",
            }
            for c in extra or {}:
                row[c] = extra[c][i - 1]
            w.writerow(row)
    (d / "config.json").write_text(json.dumps({"seed": seed, "max_epochs": max_epochs, "weight_decay": wd,
                                               "commit": "abcdef0123456789", "cache_id": "cafe"}), encoding="utf-8")


@pytest.fixture
def runs_dir(tmp_path):
    d = tmp_path / "runs"
    make_run(d, "a", [0.80, 0.90, 0.95, 0.93, 0.94], seed=0, max_epochs=12)
    make_run(d, "b", [0.70, 0.92, 0.92, 0.90], seed=1, max_epochs=12)  # a tie: the first epoch is the best
    make_run(d, "c", [0.60, 0.88, 0.91], seed=2, max_epochs=12)
    make_run(d, "wd", [0.85, 0.94], wd=0.05, max_epochs=12)
    make_run(d, "default", [0.5, 0.6])
    return d


def test_load_run_picks_the_first_best_epoch_and_the_last_epoch(runs_dir):
    b = sr.load_run("b", runs_dir)
    assert b["best"]["epoch"] == "2" and b["best_f1"] == 0.92 and b["last"]["epoch"] == "4" and b["last_f1"] == 0.90
    assert b["seed"] == 1 and b["epochs_run"] == 4 and b["commit"].startswith("abcdef0")


def test_differences_from_the_defaults(runs_dir):
    assert sr.differences(sr.load_run("a", runs_dir)) == "max-epochs 12"
    assert sr.differences(sr.load_run("wd", runs_dir)) == "max-epochs 12, weight-decay 0.05"
    assert sr.differences(sr.load_run("default", runs_dir)) == "none"


def test_defaults_match_train_py():
    src = (Path(sr.ROOT) / "training" / "train.py").read_text(encoding="utf-8")
    assert int(re.search(r'"--max-epochs", type=int, default=(\d+)', src)[1]) == sr.DEFAULTS["max_epochs"]
    assert float(re.search(r'"--weight-decay", type=float, default=([\d.e-]+)', src)[1]) == sr.DEFAULTS["weight_decay"]


def test_table_rows_labels_mean_and_range(runs_dir, tmp_path):
    out = tmp_path / "t.md"
    argv = ["a", "b", "c", "wd", "default", "--out", str(out), "--runs-dir", str(runs_dir), "--seed-group", "a", "b", "c",
            "--label", "wd=weight decay 5e-2, not adopted", "--label", "default=default 30-epoch schedule"]
    assert sr.main(argv) == 0
    text = out.read_text(encoding="utf-8")
    rows = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("|")}
    assert "weight decay 5e-2, not adopted" in rows["`wd`"] and "default 30-epoch schedule" in rows["`default`"]
    assert "| 0 |" in rows["`a`"] and "| 1 |" in rows["`b`"] and "| 3 | 0.9500 | 5 | 0.9400 |" in rows["`a`"]
    mean_best, mean_last = (0.95 + 0.92 + 0.91) / 3, (0.94 + 0.90 + 0.91) / 3
    assert f"{mean_best:.4f}" in rows["**mean**"] and f"{mean_last:.4f}" in rows["**mean**"]
    assert "0.9100 to 0.9500" in rows["**range**"] and "0.9000 to 0.9400" in rows["**range**"]
    assert "seeds 0, 1, 2" in rows["**mean**"]
    assert "`abcdef0` · cache `cafe`" in rows["`a`"]
    header = next(line for line in text.splitlines() if line.startswith("| run |"))
    assert "(best)" not in header and "user" not in header  # no per-user columns without per-user data in the log
    assert "produced at commit" in text


def test_per_user_columns_appear_only_when_the_log_has_them(tmp_path):
    d = tmp_path / "runs"
    extra = {"val_acc_user3": [0.8, 0.9], "val_acc_user5": [0.7, 0.85]}
    make_run(d, "u", [0.5, 0.6], extra=extra)
    make_run(d, "v", [0.5, 0.7], extra=extra)
    runs = [sr.load_run("u", d), sr.load_run("v", d)]
    text = "\n".join(sr.table(runs, {}, []))
    assert "val_acc_user3 (best)" in text and "val_acc_user5 (last)" in text and "0.8500" in text
    make_run(d, "w", [0.5, 0.7])  # one run without them: the columns are left out for all runs
    assert "user" not in "\n".join(sr.table([*runs, sr.load_run("w", d)], {}, []))


def test_no_mean_row_for_a_single_run_and_errors_for_bad_arguments(runs_dir, tmp_path, capsys):
    out = tmp_path / "t.md"
    assert sr.main(["a", "--out", str(out), "--runs-dir", str(runs_dir), "--seed-group", "a"]) == 0
    assert "**mean**" not in out.read_text(encoding="utf-8")
    assert sr.main(["a", "--out", str(out), "--runs-dir", str(runs_dir), "--label", "zzz=x"]) == 2
    assert sr.main(["a", "--out", str(out), "--runs-dir", str(runs_dir), "--seed-group", "b"]) == 2
    assert sr.main(["nope", "--out", str(out), "--runs-dir", str(runs_dir)]) == 2
    assert "nope" in capsys.readouterr().err
