"""Evaluate a cnn-v1 checkpoint on the validation users and write training/results/cnn.md.

Usage: python training/eval.py [--checkpoint FILE] [--out FILE] [--figure FILE]

Only the validation users user3 and user5 are loaded. The loader refuses the test users, and this script never refers
to them. The report has the same layout as training/results/baseline.md and adds a comparison with the two baselines
and a split of the macro-F1 into windows well inside their rule and windows near the edge of their rule. Every number
is stamped with the git commit, the cache id and the model name.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dsp"))
sys.path.insert(0, str(ROOT / "training"))

from baseline import cls_label, evaluate, per_user_accuracy
from build_cache import build_commit
from model import MODEL_NAME, GestureCNN
from train_data import MAX_SHIFT, predict, span_table

from radarsense_dsp.windows import (
    CLASS_NAMES,
    N_CLASSES,
    CacheReader,
    build_window_index,
    class_map_json,
    index_digest,
    load_complete_members,
    load_splits,
    window_margin,
)

CACHE = ROOT / "data" / "processed"
SPLITS = ROOT / "training" / "splits.json"
CHECKPOINT = ROOT / "training" / "checkpoints" / f"{MODEL_NAME}-seed0" / "best.pt"
OUT = ROOT / "training" / "results" / "cnn.md"
FIGURE = ROOT / "training" / "figures" / "cnn_confusion.png"
BASELINE = ROOT / "training" / "results" / "baseline.md"
EDGE_MARGIN = MAX_SHIFT  # A window is near the edge when a shift of up to MAX_SHIFT frames could cross its rule.


def window_margins(rows: np.ndarray, spans: list[np.ndarray]) -> np.ndarray:
    """Return the margin of every window row (see `window_margin`) as an int array."""
    out = np.empty(len(rows), dtype=int)
    for i, r in enumerate(rows):
        _, s, length = spans[int(r["member"])][int(r["recording"])]
        margin = window_margin(int(s), int(length), int(r["start"]))
        if margin is None:
            raise ValueError("A window row is excluded by the window rule, so the index is inconsistent.")
        out[i] = margin
    return out


def read_baseline(path: Path) -> dict[str, dict]:
    """Read the accuracy, macro-F1, per-class recall and stamp of each baseline model from baseline.md."""
    text = path.read_text(encoding="utf-8")
    out = {}
    for section in text.split("\n## ")[1:]:
        name = section.split("\n", 1)[0].strip()
        if not name.startswith("baseline-"):
            continue
        acc = re.search(r"overall accuracy: ([\d.]+)", section)
        f1 = re.search(r"macro-F1 \(classes present in val\): ([\d.]+)", section)
        stamp = re.search(r"produced at (commit `[0-9a-f+dirty]+` · cache `\w+`)", section)
        recall: dict[str, str] = {}
        for c, v in re.findall(r"^\| (\d) \w+ \| ([\d.]+|n/a) \|", section, flags=re.MULTILINE):
            recall.setdefault(c, v)  # The recall table comes before the confusion matrix, whose rows match too.
        users = {int(u): float(a) for u, a in re.findall(r"^\| user(\d+) \| ([\d.]+) \|", section, flags=re.MULTILINE)}
        if acc and f1 and stamp:
            out[name] = {
                "accuracy": float(acc[1]), "macro_f1": float(f1[1]), "stamp": stamp[1],
                "recall": [recall.get(str(c)) for c in range(N_CLASSES)], "users": users,
            }
    return out


def confusion_figure(cm: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    labels = [cls_label(c) for c in range(N_CLASSES)]
    ax.set_xticks(range(N_CLASSES), labels, rotation=45, ha="right")
    ax.set_yticks(range(N_CLASSES), labels)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title, fontsize=9)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if share[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="share of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--splits", type=Path, default=SPLITS)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--figure", type=Path, default=FIGURE)
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if ckpt["model"] != MODEL_NAME:
        print(f"ERROR: The checkpoint holds the model {ckpt['model']}, not {MODEL_NAME}.", file=sys.stderr)
        return 2
    model = GestureCNN().to(device)
    model.load_state_dict(ckpt["model_state"])

    splits = load_splits(args.splits)
    members = load_complete_members(args.cache, splits)
    cache_id = index_digest(members)
    commit = build_commit()
    stamp = f"commit `{commit}` · cache `{cache_id}` · model `{MODEL_NAME}`"
    if cache_id != ckpt["cache_id"]:
        print(f"WARNING: The checkpoint was trained on cache {ckpt['cache_id']}, but the cache is now {cache_id}.")

    wi = build_window_index(args.cache, members)
    val_rows = wi.split_rows("val")
    y = val_rows["cls"].astype(int)
    pred = predict(model, CacheReader(args.cache, members), val_rows, device).argmax(axis=1)
    res = evaluate(y, pred)
    users = per_user_accuracy(val_rows["user"], y, pred)

    margins = window_margins(val_rows, span_table(args.cache, members))
    near = margins < EDGE_MARGIN
    groups = {
        f"inside (margin of {EDGE_MARGIN} frames or more)": ~near,
        f"near the edge (margin below {EDGE_MARGIN} frames)": near,
    }
    group_res = {name: evaluate(y[mask], pred[mask]) | {"windows": int(mask.sum())} for name, mask in groups.items()}

    confusion_figure(res["confusion"], args.figure, f"{MODEL_NAME}, validation users user3 and user5\n{stamp.replace('`', '')}")
    baselines = read_baseline(args.baseline)

    s = f"({stamp})"
    out = [f"# {MODEL_NAME}: per-frame CNN with a temporal head", ""]
    out += [f"This report was produced at {stamp}.", ""]
    out += [
        (
            f"The checkpoint is the best epoch ({ckpt['epoch']}) of a run trained at commit `{ckpt['commit']}` on cache "
            f"`{ckpt['cache_id']}`, chosen by the macro-F1 on the validation users. It is evaluated on the nominal "
            "files of the validation users user3 and user5 only. Test users are never loaded."
        ),
        "",
    ]
    note = "The mapping is defined only in `dsp/radarsense_dsp/windows.py`. Not yet verified: the real gesture names."
    out += ["## Class map", "", f"`{class_map_json()}`. {note}", ""]
    out += [f"## {MODEL_NAME}", "", f"These results were produced at {stamp}.", ""]
    out += [f"- overall accuracy: {res['accuracy']:.4f}", f"- macro-F1 (classes present in val): {res['macro_f1']:.4f}", ""]
    out += [f"Per-class recall {s}:", "", "| class | recall | val windows |", "|---|---|---|"]
    out += [f"| {cls_label(c)} | {fmt(res['recall'][c])} | {res['support'][c]} |" for c in range(N_CLASSES)]
    out += ["", f"Confusion matrix with the true class in the rows and the predicted class in the columns {s}:", ""]
    out.append("| true \\ pred | " + " | ".join(cls_label(c) for c in range(N_CLASSES)) + " |")
    out.append("|" + "---|" * (N_CLASSES + 1))
    out += [f"| {cls_label(c)} | " + " | ".join(str(n) for n in res["confusion"][c]) + " |" for c in range(N_CLASSES)]
    out += ["", f"The matrix is also saved as `{args.figure.relative_to(ROOT).as_posix()}`.", ""]
    out += [f"Accuracy for each validation user {s}:", "", "| user | accuracy | windows |", "|---|---|---|"]
    out += [f"| user{u} | {acc:.4f} | {n} |" for u, (acc, n) in sorted(users.items())]

    out += ["", "## Comparison with the baselines", ""]
    out += [
        (
            "All three models are evaluated on the same validation windows. Each row carries the stamp of the run "
            "that produced it."
        ),
        "",
        "| model | accuracy | macro-F1 | user3 | user5 | stamp |",
        "|---|---|---|---|---|---|",
    ]
    out.append(
        f"| {MODEL_NAME} | {res['accuracy']:.4f} | {res['macro_f1']:.4f} | {users[3][0]:.4f} | {users[5][0]:.4f} | {stamp} |"
    )
    for name, b in baselines.items():
        out.append(
            f"| {name} | {b['accuracy']:.4f} | {b['macro_f1']:.4f} | {b['users'].get(3, float('nan')):.4f} | "
            f"{b['users'].get(5, float('nan')):.4f} | {b['stamp']} · model `{name}` |"
        )
    out += ["", f"Per-class recall {s}:", "", "| class | " + " | ".join([MODEL_NAME, *baselines]) + " |"]
    out.append("|" + "---|" * (2 + len(baselines)))
    for c in range(N_CLASSES):
        cells = [fmt(res["recall"][c]), *(b["recall"][c] or "n/a" for b in baselines.values())]
        out.append(f"| {cls_label(c)} | " + " | ".join(cells) + " |")
    for name, b in baselines.items():
        out.append("")
        out.append(f"{MODEL_NAME} minus {name} in macro-F1: {res['macro_f1'] - b['macro_f1']:+.4f}.")

    out += ["", "## Windows inside the rule and near its edge", ""]
    out += [
        (
            "The margin of a window is the number of frames it could slide towards the edge of its rule before the "
            "label rule of `window_class_at` changes its class. For a gesture window it is the smaller free space "
            "around the extended span. For a no-gesture window it is the number of frames it could move towards the "
            f"gesture and stay 10 frames away. A window is near the edge when its margin is below {EDGE_MARGIN} frames, "
            "which is the largest shift used in training. The macro-F1 of a group averages over the classes that occur "
            "in that group, with the predictions of the whole model."
        ),
        "",
        f"Windows by group and class {s}:",
        "",
        "| group | windows | accuracy | macro-F1 | " + " | ".join(f"{c} {CLASS_NAMES[c]}" for c in range(N_CLASSES)) + " |",
        "|" + "---|" * (N_CLASSES + 4),
    ]
    for name, g in group_res.items():
        out.append(
            f"| {name} | {g['windows']} | {g['accuracy']:.4f} | {g['macro_f1']:.4f} | "
            + " | ".join(str(n) for n in g["support"])
            + " |"
        )
    out.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"{MODEL_NAME}: accuracy {res['accuracy']:.4f}, macro-F1 {res['macro_f1']:.4f}.")
    for name, g in group_res.items():
        print(f"  {name}: {g['windows']} windows, macro-F1 {g['macro_f1']:.4f}.")
    print(f"Wrote {args.out} and {args.figure}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
