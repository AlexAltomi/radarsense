"""Week 1 check: list the zip, then read ONE .npz in memory and print dtypes/label values.

Usage: python training/inspect_zip.py [substring-of-member-name]
Never extracts the zip to disk.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

ZIP = Path(r"D:\radarsense-data\radar_dataset.zip")
LISTING = ZIP.parent / "zip_listing.txt"

with zipfile.ZipFile(ZIP) as z:
    infos = [i for i in z.infolist() if i.filename.endswith(".npz")]
    print(f"{len(infos)} .npz members")
    LISTING.write_text(
        "\n".join(f"{i.file_size / 1e6:10.1f} MB  {i.filename}" for i in infos),
        encoding="utf-8",
    )
    print(f"full listing written to {LISTING}")

    if len(sys.argv) > 1:
        pick = next(i for i in infos if sys.argv[1] in i.filename)
    else:
        pick = min(infos, key=lambda i: i.file_size)
    print(f"reading: {pick.filename} ({pick.file_size / 1e6:.1f} MB)")

    with z.open(pick) as f:
        data = np.load(io.BytesIO(f.read()))
    for k in data.files:
        print(f"  {k}: dtype={data[k].dtype} shape={data[k].shape}")
    vals, counts = np.unique(data["targets"], return_counts=True)
    print("target values:", dict(zip(vals.tolist(), counts.tolist())))
