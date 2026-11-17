"""Writes a small parity set for the Android classifier test from golden.npz.

Output (in data/models/<version>/parity/, which git ignores):
  parity_u8.bin      N windows of uint8, each (32, 3, 32, 32): the wire values
  parity_logits.bin  N x 6 float32, little-endian: the float32 model's logits from golden.npz

Push both into the app's external files folder, then run the instrumented test:
  adb push data/models/cnn-v1-e12/parity/. /sdcard/Android/data/com.radarsense.app/files/

Run with the repository venv: .venv\\Scripts\\python.exe scripts/make_parity_subset.py
"""
import sys
from pathlib import Path

import numpy as np

VERSION = "cnn-v1-e12"
PER_CLASS = 8  # 6 classes x 8 = 48 windows, about 4.7 MB


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "data" / "models" / VERSION
    golden = np.load(model_dir / "golden.npz")
    u8, classes, logits = golden["eval_u8"], golden["classes"], golden["logits_fp32"]

    picks = np.concatenate([np.flatnonzero(classes == c)[:PER_CLASS] for c in range(6)])
    out = model_dir / "parity"
    out.mkdir(exist_ok=True)
    np.ascontiguousarray(u8[picks], dtype=np.uint8).tofile(out / "parity_u8.bin")
    np.ascontiguousarray(logits[picks], dtype="<f4").tofile(out / "parity_logits.bin")
    print(f"{len(picks)} windows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
