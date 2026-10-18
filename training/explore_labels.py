"""Exploration, TRAIN USER ONLY: which target value is which motion axis.
Copies user4_e1.npz from the zip to D: (streamed, low RAM), then reads only the
first N_REC recordings. Delete the temp file afterwards.
"""
import shutil
import zipfile
from pathlib import Path

import numpy as np

ZIP = Path(r"D:\radarsense-data\radar_dataset.zip")
MEMBER = "fulldataset/user4_e1.npz"
TMP = ZIP.parent / "tmp_user4_e1.npz"
N_REC = 300
RANGE_WIN = np.hanning(64)
DOPP_WIN = np.hanning(32)

if not TMP.exists():
    print("copying member to D: (about 2.4 GB, takes a minute)...")
    with zipfile.ZipFile(ZIP) as z, z.open(MEMBER) as src, open(TMP, "wb") as dst:
        shutil.copyfileobj(src, dst, 16 * 1024 * 1024)

with zipfile.ZipFile(TMP) as zz:
    with zz.open("inputs.npy") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        assert not fortran, "unexpected Fortran order"
        per_rec = int(np.prod(shape[1:])) * dtype.itemsize
        buf = f.read(N_REC * per_rec)
        inputs = np.frombuffer(buf, dtype=dtype).reshape((N_REC, *shape[1:]))
    with zz.open("targets.npy") as f:
        targets = np.lib.format.read_array(f)[:N_REC]

print(f"inputs full shape {shape}, using first {N_REC}; targets {targets.shape}")


def frame_features(x):
    """x: (3 ant, 32 chirps, 64 samples) -> range bin, signed Doppler bin, pair phases."""
    x = x - x.mean(axis=-1, keepdims=True)
    r = np.fft.fft(x * RANGE_WIN, axis=-1)[..., :32]
    r = r - r.mean(axis=1, keepdims=True)
    rb = int(np.abs(r).sum(axis=(0, 1)).argmax())
    d = np.fft.fftshift(np.fft.fft(r[:, :, rb] * DOPP_WIN, axis=-1), axes=-1)
    db = int(np.abs(d).sum(axis=0).argmax())
    a = d[:, db]
    ph = [
        np.angle(a[0] * np.conj(a[1])),
        np.angle(a[0] * np.conj(a[2])),
        np.angle(a[1] * np.conj(a[2])),
    ]
    return rb, db - 16, ph


def slope(y):
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


per_class = {}
skipped = 0
for rec in range(N_REC):
    labels = np.unique(targets[rec][targets[rec] != 0])
    if len(labels) != 1:
        skipped += 1
        continue
    c = int(labels[0])
    idx = np.flatnonzero(targets[rec] == c)
    feats = [frame_features(inputs[rec, i]) for i in idx]
    rb = [ft[0] for ft in feats]
    db = [ft[1] for ft in feats]
    ph = np.unwrap(np.array([ft[2] for ft in feats]), axis=0)
    per_class.setdefault(c, []).append(
        [slope(rb), float(np.mean(db)), slope(ph[:, 0]), slope(ph[:, 1]), slope(ph[:, 2])]
    )

print(f"recordings with exactly one gesture label: {N_REC - skipped}/{N_REC}")
cols = ["range_slope", "doppler_mean", "ph01_slope", "ph02_slope", "ph12_slope"]
print("class   n   " + "   ".join(f"{c:>15}" for c in cols))
for c in sorted(per_class):
    arr = np.array(per_class[c])
    cells = "   ".join(f"{m:+6.2f}+/-{s:5.2f}" for m, s in zip(arr.mean(0), arr.std(0)))
    print(f"{c:>5} {len(arr):>3}   {cells}")
