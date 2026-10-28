"""Week 3 baseline: hand-crafted window features + LogisticRegression / RandomForest.

Usage: python training/baseline.py [--cache DIR] [--splits FILE] [--out FILE] [--counts-only]
Trains on the train users' nominal windows, evaluates on the val users (user3, user5) only. Test users are refused by
the loader and never referenced here. Reads only cache members whose index.json status is "complete".
Window rules and the class map live in dsp/radarsense_dsp/windows.py. Exit 2: a train/val user has no complete member.
Every reported number is stamped with the git commit, the model version and the cache index hash.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dsp"))

from build_cache import build_commit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from radarsense_dsp.windows import (
    CLASS_NAMES,
    N_CLASSES,
    WINDOW,
    CacheMember,
    CacheReader,
    WindowIndex,
    build_window_index,
    class_map_json,
    count_table,
    incomplete_members,
    index_digest,
    load_complete_members,
    load_splits,
    no_gesture_recording_share,
    scarcity_warnings,
    user_balanced_weights,
)

CACHE = ROOT / "data" / "processed"
SPLITS = ROOT / "training" / "splits.json"
OUT = ROOT / "training" / "results" / "baseline.md"
EPS = 1e-6
SERIES = ("range_centroid", "doppler_centroid", "energy", "axis_a", "axis_b")
STATS = ("mean", "min", "max", "slope", "argmax", "argmin")
FEATURE_NAMES = [f"{s}_{t}" for s in SERIES for t in STATS]
MODEL_VERSIONS = ("baseline-lr-v1", "baseline-rf-v1")


def frame_series(rec: np.ndarray) -> np.ndarray:
    """(frames, 3, 32, 32) -> (5, frames): range centroid, signed Doppler centroid, energy, axis A, axis B.

    Centroids are weighted by ch0 (scaled magnitude). ch1/ch2 are already weighted by ch0, so their sum over
    ch0's sum is the magnitude-weighted mean of sin(delta phi). Layout per frame is [channel, range, Doppler];
    Doppler bin 16 is zero Doppler (fftshift).
    """
    f = rec.astype(np.float32)
    ch0 = f[:, 0]
    energy = ch0.sum(axis=(1, 2))
    denom = energy + EPS
    rng = np.arange(32, dtype=np.float32)
    dop = np.arange(32, dtype=np.float32) - 16
    range_c = (ch0.sum(axis=2) * rng).sum(axis=1) / denom
    doppler_c = (ch0.sum(axis=1) * dop).sum(axis=1) / denom
    axis_a = f[:, 1].sum(axis=(1, 2)) / denom
    axis_b = f[:, 2].sum(axis=(1, 2)) / denom
    return np.stack([range_c, doppler_c, energy, axis_a, axis_b])


def window_stats(segs: np.ndarray) -> np.ndarray:
    """(W, 5, T) -> (W, 30): mean, min, max, least-squares slope, argmax, argmin time per series (FEATURE_NAMES)."""
    t = np.arange(segs.shape[-1], dtype=np.float32)
    t -= t.mean()
    mean = segs.mean(axis=-1)
    slope = ((segs - mean[..., None]) * t).sum(axis=-1) / (t * t).sum()
    stats = [mean, segs.min(axis=-1), segs.max(axis=-1), slope, segs.argmax(axis=-1), segs.argmin(axis=-1)]
    return np.stack(stats, axis=-1).reshape(len(segs), -1).astype(np.float32)


def extract_features(reader: CacheReader, rows: np.ndarray) -> np.ndarray:
    """(len(rows), 30); each recording is read once and all its windows share the per-frame series."""
    X = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    if len(rows) == 0:
        return X
    order = np.lexsort((rows["start"], rows["recording"], rows["member"]))
    keys = rows[order]
    change = np.flatnonzero((np.diff(keys["member"]) != 0) | (np.diff(keys["recording"]) != 0)) + 1
    bounds = [0, *change.tolist(), len(keys)]
    for a, b in itertools.pairwise(bounds):
        grp = keys[a:b]
        series = frame_series(reader.recording(int(grp["member"][0]), int(grp["recording"][0])))
        segs = np.stack([series[:, int(w) : int(w) + WINDOW] for w in grp["start"]])
        X[order[a:b]] = window_stats(segs)
    return X


def sample_weights(users: np.ndarray, y: np.ndarray) -> np.ndarray:
    """User-balanced weights, rescaled per class so every present class has equal total weight.

    Class totals are exact; user totals are only approximately equal after the class rescaling.
    """
    w = user_balanced_weights(users)
    totals = np.bincount(y, weights=w, minlength=N_CLASSES)
    present = totals > 0
    class_w = np.zeros(N_CLASSES)
    class_w[present] = w.sum() / (present.sum() * totals[present])
    return w * class_w[y]


def make_models() -> dict:
    return {
        "baseline-lr-v1": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "baseline-rf-v1": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    }


def fit(model, X, y, w) -> None:
    if hasattr(model, "steps"):
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    else:
        model.fit(X, y, sample_weight=w)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro-F1 over the classes present in y_true, per-class recall (None if absent) and the confusion matrix."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    recall = [float(tp[c] / support[c]) if support[c] else None for c in range(N_CLASSES)]
    f1 = [2 * tp[c] / (support[c] + predicted[c]) for c in np.flatnonzero(support)]
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(np.mean(f1)),
        "recall": recall,
        "support": support.tolist(),
        "confusion": cm,
    }


def per_user_accuracy(users: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, tuple[float, int]]:
    return {
        int(u): (float((y_true[users == u] == y_pred[users == u]).mean()), int((users == u).sum()))
        for u in np.unique(users)
    }


def cls_label(c: int) -> str:
    return f"{c} {CLASS_NAMES[c]}"


def counts_markdown(counts: dict[str, list[int]], wi: WindowIndex) -> list[str]:
    lines = ["| split | " + " | ".join(cls_label(c) for c in range(N_CLASSES)) + " | total | recordings with a no-gesture window |"]
    lines.append("|" + "---|" * (N_CLASSES + 3))
    for split, c in counts.items():
        share = no_gesture_recording_share(wi, split)
        lines.append(f"| {split} | " + " | ".join(str(n) for n in c) + f" | {sum(c)} | {share:.1%} |")
    return lines


def report_markdown(
    stamp: str,
    members: list[CacheMember],
    skipped: list[CacheMember],
    wi: WindowIndex,
    counts: dict[str, list[int]],
    warnings: list[str],
    results: dict[str, dict],
    shape: dict,
) -> str:
    out = ["# Baseline: hand-crafted window features", ""]
    out += [f"Stamp: {stamp}", ""]
    out += ["Train users: nominal files of the train split; evaluation: val users only (nominal). Test users are never loaded.", ""]
    class_map_note = "single definition in `dsp/radarsense_dsp/windows.py`; gesture names are UNVERIFIED"
    out += ["## Class map", "", f"`{class_map_json()}` ({class_map_note})", ""]
    out += [f"## Window counts ({stamp})", ""] + counts_markdown(counts, wi) + [""]
    if warnings:
        out += ["**Warnings**", ""] + [f"- {w}" for w in warnings] + [""]
    out += [f"## Cache members used ({stamp})", ""]
    out += [f"- {m.stem} ({m.split}, n={m.entry['n']})" for m in members]
    if skipped:
        out += ["", "Listed in index.json but NOT complete, not read:"] + [f"- {m.stem} ({m.entry['status']})" for m in skipped]
    features = f"{len(FEATURE_NAMES)} per window ({', '.join(SERIES)} x {', '.join(STATS)})"
    out += ["", f"Features: {features}; train windows {shape['train']}, val windows {shape['val']}.", ""]
    for version, r in results.items():
        s = f"{stamp} · model `{version}`"
        out += [f"## {version}", "", f"Stamp: {s}", ""]
        out += [f"- overall accuracy: {r['accuracy']:.4f}", f"- macro-F1 (classes present in val): {r['macro_f1']:.4f}", ""]
        out += [f"Per-class recall ({s})", "", "| class | recall | val windows |", "|---|---|---|"]
        for c in range(N_CLASSES):
            rec = "n/a" if r["recall"][c] is None else f"{r['recall'][c]:.4f}"
            out.append(f"| {cls_label(c)} | {rec} | {r['support'][c]} |")
        out += ["", f"Confusion matrix, rows = true, columns = predicted ({s})", ""]
        out.append("| true \\ pred | " + " | ".join(cls_label(c) for c in range(N_CLASSES)) + " |")
        out.append("|" + "---|" * (N_CLASSES + 1))
        for c in range(N_CLASSES):
            out.append(f"| {cls_label(c)} | " + " | ".join(str(n) for n in r["confusion"][c]) + " |")
        out += ["", f"Per-val-user accuracy ({s})", "", "| user | accuracy | windows |", "|---|---|---|"]
        out += [f"| user{u} | {acc:.4f} | {n} |" for u, (acc, n) in sorted(r["per_user"].items())]
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--splits", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--counts-only", action="store_true", help="print window counts and warnings, train nothing")
    args = ap.parse_args(argv)

    splits = load_splits(args.splits)
    members = load_complete_members(args.cache, splits)
    skipped = incomplete_members(args.cache, splits)
    digest = index_digest(members)
    wi = build_window_index(args.cache, members)
    counts = count_table(wi)
    warnings = scarcity_warnings(counts)
    stamp = f"commit `{build_commit()}` · cache `{digest}`"

    print(f"{stamp}; {len(members)} complete members, {len(skipped)} listed but not complete")
    print("\n".join(counts_markdown(counts, wi)))
    for w in warnings:
        print(f"WARNING: {w}")
    for m in skipped:
        print(f"WARNING: {m.stem} is {m.entry['status']}, not read")

    expected = {int(u.removeprefix("user")) for split in ("train", "val") for u in splits[split]}
    missing = sorted(expected - {m.user for m in members})
    if missing:
        print(f"{'WARNING' if args.counts_only else 'ERROR'}: no complete member for users {missing}", file=sys.stderr)
        if not args.counts_only:
            return 2
    if args.counts_only:
        return 0

    reader = CacheReader(args.cache, members)
    train_rows, val_rows = wi.split_rows("train"), wi.split_rows("val")
    X_train, y_train = extract_features(reader, train_rows), train_rows["cls"].astype(int)
    X_val, y_val = extract_features(reader, val_rows), val_rows["cls"].astype(int)
    w_train = sample_weights(train_rows["user"], y_train)

    results = {}
    for version, model in make_models().items():
        fit(model, X_train, y_train, w_train)
        pred = model.predict(X_val).astype(int)
        results[version] = evaluate(y_val, pred) | {"per_user": per_user_accuracy(val_rows["user"], y_val, pred)}
        print(f"{version}: accuracy {results[version]['accuracy']:.4f}, macro-F1 {results[version]['macro_f1']:.4f}")

    shape = {"train": len(train_rows), "val": len(val_rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report_markdown(stamp, members, skipped, wi, counts, warnings, results, shape), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
