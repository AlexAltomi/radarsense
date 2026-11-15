"""Prepare the inputs of the LiteRT export (Windows, uses the main .venv).

Usage: python training/export_prepare.py --checkpoint training/checkpoints/cnn-v1-e12/best.pt

Writes data/models/<version>/ (git-ignored), where <version> is the run name (the folder name of the checkpoint):
  export_inputs.npz   500 calibration windows from the TRAIN users and 1,000 evaluation windows from the VALIDATION
                      users user3 and user5. Both sets are class-balanced, drawn with a fixed seed, and passed through
                      the uint8 wire round trip of replay.codec. The npz holds the uint8 windows, their classes and users.
  best.pt             a copy of the checkpoint
  commit_stamp.json   the commit, cache id, epoch and validation macro-F1 of the checkpoint, and the hashes of the files

Then convert_litert.py (Linux, Colab) turns best.pt into .tflite files, and export_check.py compares the models.
The test users are never loaded: the window loader refuses them, and select_balanced checks the rows again.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from build_cache import build_commit
from export_common import (
    EVAL_USERS,
    INPUTS_NAME,
    N_CALIB,
    N_EVAL,
    ROOT,
    SEED,
    STAMP_NAME,
    select_balanced,
    wire_roundtrip,
    write_json,
)
from model import MODEL_NAME

from radarsense_dsp.windows import (
    CLASS_NAMES,
    CacheReader,
    build_window_index,
    index_digest,
    load_complete_members,
    load_splits,
)

CACHE = ROOT / "data" / "processed"
SPLITS = ROOT / "training" / "splits.json"
MODELS = ROOT / "data" / "models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_windows(reader: CacheReader, rows: np.ndarray) -> np.ndarray:
    return np.stack([reader.window(r) for r in rows]).astype(np.float32)


def prepare(
    cache: Path, splits: dict, n_calib: int = N_CALIB, n_eval: int = N_EVAL, seed: int = SEED
) -> tuple[dict, str]:
    """Select the windows. Return the arrays of export_inputs.npz and the digest of the cache index."""
    members = load_complete_members(cache, splits)  # refuses the test users
    wi = build_window_index(cache, members)
    reader = CacheReader(cache, members)
    out = {}
    for name, split, n in (("calib", "train", n_calib), ("eval", "val", n_eval)):
        rows = wi.split_rows(split)
        chosen = rows[select_balanced(rows, n, seed)]
        u8, _ = wire_roundtrip(read_windows(reader, chosen))
        out[f"{name}_u8"] = u8
        out[f"{name}_classes"] = chosen["cls"].astype(np.int8)
        out[f"{name}_users"] = chosen["user"].astype(np.int16)
        out[f"{name}_rows"] = np.stack([chosen["member"], chosen["recording"], chosen["start"]], axis=1).astype(np.int32)
    return out, index_digest(members)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--splits", type=Path, default=SPLITS)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: data/models/<run name>")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-calib", type=int, default=N_CALIB)
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    args = ap.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt["model"] != MODEL_NAME:
        print(f"ERROR: The checkpoint holds the model {ckpt['model']}, not {MODEL_NAME}.", file=sys.stderr)
        return 2
    version = args.checkpoint.resolve().parent.name
    out_dir = args.out_dir or MODELS / version
    out_dir.mkdir(parents=True, exist_ok=True)

    data, cache_id = prepare(args.cache, load_splits(args.splits), args.n_calib, args.n_eval, args.seed)
    if cache_id != ckpt["cache_id"]:
        print(f"WARNING: The checkpoint was trained on cache {ckpt['cache_id']}, but the cache is now {cache_id}.")
    npz = out_dir / INPUTS_NAME
    np.savez_compressed(npz, **data)
    shutil.copy2(args.checkpoint, out_dir / "best.pt")

    stamp = {
        "version": version,
        "model": ckpt["model"],
        "checkpoint_commit": ckpt["commit"],
        "cache_id": ckpt["cache_id"],
        "cache_id_now": cache_id,
        "epoch": ckpt["epoch"],
        "val_macro_f1": ckpt["val_macro_f1"],
        "seed": args.seed,
        "n_calib": len(data["calib_u8"]),
        "n_eval": len(data["eval_u8"]),
        "eval_users": list(EVAL_USERS),
        "calib_users": sorted({f"user{u}" for u in np.unique(data["calib_users"])}),
        "class_names": list(CLASS_NAMES),
        "eval_class_counts": np.bincount(data["eval_classes"], minlength=len(CLASS_NAMES)).tolist(),
        "calib_class_counts": np.bincount(data["calib_classes"], minlength=len(CLASS_NAMES)).tolist(),
        "prepared_at_commit": build_commit(),
        "sha256": {"best.pt": sha256(out_dir / "best.pt"), INPUTS_NAME: sha256(npz)},
        "config": ckpt["config"],
    }
    write_json(out_dir / STAMP_NAME, stamp)
    mb = 1024 * 1024
    print(f"Wrote {out_dir}: {INPUTS_NAME} {npz.stat().st_size / mb:.1f} MB, best.pt, {STAMP_NAME}.")
    print(f"Calibration: {stamp['n_calib']} windows, users {stamp['calib_users']}, classes {stamp['calib_class_counts']}.")
    print(f"Evaluation:  {stamp['n_eval']} windows, users {stamp['eval_users']}, classes {stamp['eval_class_counts']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
