"""Turn an index entry into a clip of encoded frames with a true label for every frame."""
from dataclasses import dataclass

import numpy as np

from radarsense_dsp.windows import CLASS_MAP
from replay import codec
from replay.index import ReplayIndex, background_ranges

_LABEL_LUT = np.full(256, codec.UNKNOWN_LABEL, dtype=np.uint8)
for _target, _cls in CLASS_MAP.items():
    _LABEL_LUT[_target] = _cls


@dataclass(frozen=True)
class Clip:
    frames: np.ndarray  # uint8, shape (n, 3, 32, 32)
    labels: np.ndarray  # uint8, shape (n,): the class of the frame's target value, or 255 when unknown
    first_frame: int  # The index of the first frame inside its recording.


def labels_from_targets(targets: np.ndarray) -> np.ndarray:
    """Map target values through CLASS_MAP. Values that are not in CLASS_MAP (including negative ones) give 255."""
    t = np.asarray(targets).astype(np.int64)
    valid = (t >= 0) & (t < 256)
    return np.where(valid, _LABEL_LUT[np.clip(t, 0, 255)], codec.UNKNOWN_LABEL).astype(np.uint8)


def load_clip(index: ReplayIndex, rec: np.void, background: bool, rng: np.random.Generator) -> Clip:
    """Read one recording, or one stretch of its background frames, and encode it for the wire."""
    member, r = int(rec["member"]), int(rec["recording"])
    lo, hi = 0, 100
    if background:
        ranges = background_ranges(int(rec["s"]), int(rec["length"]))
        chosen = ranges[int(rng.choice(len(ranges), p=np.array([len(x) for x in ranges]) / sum(map(len, ranges))))]
        lo, hi = chosen.start, chosen.stop
    frames = codec.encode_frames(index.reader.recording(member, r)[lo:hi].astype(np.float32))
    labels = labels_from_targets(index.targets(member)[r][lo:hi])
    return Clip(frames, labels, lo)


def next_clip(index: ReplayIndex, rng: np.random.Generator, user: str, gesture: str) -> Clip:
    rec, background = index.pick(rng, user, gesture)
    return load_clip(index, rec, background, rng)
