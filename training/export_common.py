"""Shared helpers of the export scripts: window selection, the wire round trip, metrics and the model card.

This module needs only numpy (and the repository's dsp and replay packages), so export_check.py can use it in the
.venv-export environment, which has no CUDA and no scikit-learn.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dsp"))
sys.path.insert(0, str(ROOT / "backend" / "replay"))

from radarsense_dsp.io import TEST_USERS
from radarsense_dsp.windows import (
    CLASS_MAP,
    CLASS_NAMES,
    N_CLASSES,
    user_balanced_weights,
)
from replay import codec

DATASET_DOI = "10.5281/zenodo.15178095"
N_CALIB = 500
N_EVAL = 1000
SEED = 0
EVAL_USERS = ("user3", "user5")
FP32_NAME, INT8_NAME, FP16_NAME = "model_fp32.tflite", "model_int8.tflite", "model_fp16.tflite"
INPUTS_NAME, GOLDEN_NAME, CARD_NAME, STAMP_NAME = "export_inputs.npz", "golden.npz", "model_card.json", "commit_stamp.json"
MAX_INT8_DROP = 0.01  # An INT8 macro-F1 more than 1 point below FP32 makes float16 the fallback.


def assert_no_test_users(users: np.ndarray) -> None:
    """Refuse window rows of the locked test users. They are not loaded before Week 11."""
    bad = sorted({int(u) for u in np.unique(users)} & set(TEST_USERS))
    if bad:
        raise PermissionError(f"The rows contain the test users {bad}, who may not be used before Week 11.")


def select_balanced(rows: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Return n indices into `rows`, with the same number of windows from every class (the first `n % 6` classes get one
    more). Inside a class the users share the draw equally. The result is shuffled and depends only on the seed.
    """
    assert_no_test_users(rows["user"])
    rng = np.random.default_rng(seed)
    quota = [n // N_CLASSES + (1 if c < n % N_CLASSES else 0) for c in range(N_CLASSES)]
    picks = []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(rows["cls"] == c)
        if len(idx) < quota[c]:
            raise ValueError(f"Class {c} ({CLASS_NAMES[c]}) has {len(idx)} windows, fewer than the {quota[c]} needed.")
        w = user_balanced_weights(rows["user"][idx])
        picks.append(rng.choice(idx, quota[c], replace=False, p=w / w.sum()))
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


def wire_roundtrip(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode windows of shape (N, 32, 3, 32, 32) to uint8 with replay.codec and decode them back.

    Returns (uint8 windows, float32 windows). All three models (PyTorch, FP32 and INT8) get the float32 windows, so the
    numbers describe what the app sees.
    """
    u8 = codec.encode_frames(np.asarray(windows, dtype=np.float32))
    return u8, codec.decode_frames(u8)


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float((np.asarray(y) == np.asarray(pred)).mean())


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """The macro-F1 over the classes that occur in y. It gives the same value as baseline.evaluate."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(cm, (np.asarray(y), np.asarray(pred)), 1)
    support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    return float(np.mean([2 * tp[c] / (support[c] + predicted[c]) for c in np.flatnonzero(support)]))


def per_class_recall(y: np.ndarray, pred: np.ndarray) -> list[float | None]:
    y, pred = np.asarray(y), np.asarray(pred)
    return [float((pred[y == c] == c).mean()) if (y == c).any() else None for c in range(N_CLASSES)]


def model_metrics(y: np.ndarray, logits: np.ndarray) -> dict:
    pred = logits.argmax(axis=1)
    return {"accuracy": accuracy(y, pred), "macro_f1": macro_f1(y, pred), "recall": per_class_recall(y, pred)}


def top1_agreement(a_logits: np.ndarray, b_logits: np.ndarray) -> float:
    return float((a_logits.argmax(axis=1) == b_logits.argmax(axis=1)).mean())


def int8_needs_fallback(f1_fp32: float, f1_int8: float) -> bool:
    return f1_int8 < f1_fp32 - MAX_INT8_DROP


def build_model_card(
    version: str, stamp: dict, constants: dict, metrics: dict, sizes: dict, agreement: dict, fallback: str | None
) -> dict:
    """Return the model card. `metrics` maps a model name to model_metrics(...) on the evaluation windows."""
    return {
        "version": version,
        "model": stamp["model"],
        "input": {
            "shape": [1, 32, 3, 32, 32],
            "layout": "(batch, frame, channel, range, Doppler); NCHW inside each frame",
            "dtype": "float32",
            "frames_per_window": 32,
            "wire_dtype": "uint8",
        },
        "output": {"shape": [1, N_CLASSES], "kind": "logits"},
        "classes": list(CLASS_NAMES),
        "label_to_class": {str(k): v for k, v in CLASS_MAP.items()},
        "uint8_decode": {
            "channel_0": {"formula": "x = u / 255", "range": [0.0, 1.0]},
            "channel_1": {"formula": "x = u / 255 * 2 - 1", "range": [-1.0, 1.0]},
            "channel_2": {"formula": "x = u / 255 * 2 - 1", "range": [-1.0, 1.0]},
        },
        "db_range": {
            "db_min": constants["db_min"],
            "db_max": constants["db_max"],
            "pair_a": constants["pair_a"],
            "pair_b": constants["pair_b"],
            "source": "dsp/radarsense_dsp/constants.json",
            "source_commit": constants["git_commit"],
            "source_users": constants["source_users"],
        },
        "dataset": {"doi": DATASET_DOI, "license": "CC BY 4.0"},
        "validation": {
            "users": list(EVAL_USERS),
            "windows": stamp["n_eval"],
            "selection": f"class-balanced, seed {stamp['seed']}, after the uint8 wire round trip",
            "checkpoint_epoch": stamp["epoch"],
            "checkpoint_val_macro_f1": stamp["val_macro_f1"],
            "models": metrics,
            "top1_agreement": agreement,
        },
        "test": {"macro_f1": None, "accuracy": None, "note": "Not evaluated before Week 11."},
        "sizes_bytes": sizes,
        "fallback": fallback,
        "stamp": {
            "checkpoint_commit": stamp["checkpoint_commit"],
            "cache_id": stamp["cache_id"],
            "prepared_at_commit": stamp["prepared_at_commit"],
        },
    }


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
