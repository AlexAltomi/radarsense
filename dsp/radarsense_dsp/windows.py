"""Training windows over the processed cache.

This module decides which 32-frame windows count as gesture, no-gesture or excluded. It also builds the dataset
index over the complete cache members, reads windows from memory-mapped files, and balances the users. The rules
are described under "Training windows" in docs/project-rules.md.

For a recording with the labelled span [s, s+L-1] and the label g, a window is classified as follows:

  gesture     The window contains the extended span [s-PRE, s+L-1] completely. Its class is CLASS_MAP[g].
  no gesture  The window does not overlap [s-PRE-GUARD, s+L-1+GUARD]. Its class is 0.
  otherwise   The window is excluded. It is never labelled as no-gesture.

Dataset labels are never shifted. CLASS_MAP is the only place that defines the mapping from label values to class
numbers. Test users are refused. Anomalous members are skipped unless `nominal_only=False` is passed. Only index.json
entries with the status "complete" are read.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radarsense_dsp.cache import load_index
from radarsense_dsp.io import TEST_USERS

WINDOW = 32
STRIDE = 4
PRE = 10  # The number of frames before s that a gesture window must also contain.
GUARD = 10  # No-gesture windows stay at least this many frames away from the extended span.
FRAMES = 100

CLASS_MAP: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 7: 5}
CLASS_NAMES = ("none", "label1", "label2", "label3", "label6", "label7")  # Not yet verified: the real gesture names.
N_CLASSES = len(CLASS_NAMES)

ROW_DTYPE = np.dtype([("member", "i4"), ("recording", "i4"), ("start", "i2"), ("cls", "i1"), ("user", "i2")])


def class_map_json() -> dict:
    return {
        "label_to_class": {str(k): v for k, v in CLASS_MAP.items()},
        "class_names": list(CLASS_NAMES),
    }


def window_starts() -> range:
    return range(0, FRAMES - WINDOW + 1, STRIDE)


def gesture_span(row: np.ndarray) -> tuple[int, int, int]:
    """Return (label, s, L) for a recording that holds exactly one contiguous gesture of a single label value."""
    idx = np.flatnonzero(row)
    if len(idx) == 0:
        raise ValueError("The recording has no gesture, so it has no labelled span.")
    label = int(row[idx[0]])
    if (row[idx] != label).any():
        raise ValueError("The recording has more than one label value, so it has no single labelled span.")
    s, length = int(idx[0]), len(idx)
    if int(idx[-1]) - s + 1 != length:
        raise ValueError("The gesture span of the recording is not contiguous.")
    return label, s, length


def window_classes(row: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, class) for every gesture and no-gesture window of one recording. Excluded windows are left out."""
    label, s, length = gesture_span(row)
    if label not in CLASS_MAP:
        raise ValueError(f"The label {label} is not in CLASS_MAP, so it has no class number.")
    gesture_cls = CLASS_MAP[label]
    ext_lo, ext_hi = max(0, s - PRE), s + length - 1
    far_lo, far_hi = s - PRE - GUARD, s + length - 1 + GUARD
    out = []
    for w in window_starts():
        w_hi = w + WINDOW - 1
        if w <= ext_lo and w_hi >= ext_hi:
            out.append((w, gesture_cls))
        elif w_hi < far_lo or w > far_hi:
            out.append((w, 0))
    return out


def load_splits(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CacheMember:
    stem: str
    user: int
    split: str
    anomaly: str | None
    entry: dict


def _split_of_user(splits: dict, user: int) -> str:
    name = f"user{user}"
    for split in ("train", "val", "test"):
        if name in splits.get(split, []):
            return split
    raise ValueError(f"{name} is not listed in any split of splits.json.")


def _cache_members(cache_dir: Path, splits: dict, nominal_only: bool, complete: bool) -> list[CacheMember]:
    out = []
    for stem, entry in sorted(load_index(cache_dir)["files"].items()):
        if nominal_only and entry["anomaly"] is not None:
            continue
        user = int(entry["user"])
        if user in TEST_USERS:
            raise PermissionError(
                f"The cache member {stem} belongs to the test user user{user}, who may not be loaded before Week 11. "
                "Remove this member from the cache index."
            )
        split = _split_of_user(splits, user)
        is_complete = (
            entry["status"] == "complete"
            and (cache_dir / f"{stem}.npy").exists()
            and (cache_dir / f"{stem}.targets.npy").exists()
        )
        if is_complete == complete:
            out.append(CacheMember(stem, user, split, entry["anomaly"], entry))
    return out


def load_complete_members(cache_dir: str | Path, splits: dict, nominal_only: bool = True) -> list[CacheMember]:
    """Return the members whose index status is "complete" and whose files exist, sorted by stem."""
    return _cache_members(Path(cache_dir), splits, nominal_only, complete=True)


def incomplete_members(cache_dir: str | Path, splits: dict, nominal_only: bool = True) -> list[CacheMember]:
    """Return the members listed in index.json that cannot be used because of a violation or missing files."""
    return _cache_members(Path(cache_dir), splits, nominal_only, complete=False)


def index_digest(members: list[CacheMember]) -> str:
    """Return the first 16 hex digits of the SHA-256 hash of the index entries of exactly these members.

    The hash stays the same while other members are still being built.
    """
    payload = json.dumps({m.stem: m.entry for m in sorted(members, key=lambda m: m.stem)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class WindowIndex:
    members: list[CacheMember]
    rows: np.ndarray  # ROW_DTYPE

    def split_rows(self, split: str) -> np.ndarray:
        ids = [i for i, m in enumerate(self.members) if m.split == split]
        return self.rows[np.isin(self.rows["member"], ids)]


def build_window_index(cache_dir: str | Path, members: list[CacheMember]) -> WindowIndex:
    """Build the window rows from the targets files only. The frames are never touched."""
    cache_dir = Path(cache_dir)
    chunks = []
    for member_id, m in enumerate(members):
        targets = np.load(cache_dir / f"{m.stem}.targets.npy")
        rows = []
        for r, row in enumerate(targets):
            rows.extend((member_id, r, w, cls, m.user) for w, cls in window_classes(row))
        chunks.append(np.array(rows, dtype=ROW_DTYPE))
    rows = np.concatenate(chunks) if chunks else np.empty(0, dtype=ROW_DTYPE)
    return WindowIndex(members, rows)


class CacheReader:
    """Read frames through memory-mapped files, which are opened lazily for each member.

    Only the requested slice is copied into memory.
    """

    def __init__(self, cache_dir: str | Path, members: list[CacheMember]):
        self._dir = Path(cache_dir)
        self._members = members
        self._maps: dict[int, np.ndarray] = {}

    def _map(self, member: int) -> np.ndarray:
        if member not in self._maps:
            self._maps[member] = np.load(self._dir / f"{self._members[member].stem}.npy", mmap_mode="r")
        return self._maps[member]

    def recording(self, member: int, recording: int) -> np.ndarray:
        """Return one recording as a float16 array of shape (100, 3, 32, 32)."""
        return np.array(self._map(member)[recording])

    def window(self, row) -> np.ndarray:
        """Return the window of one WindowIndex row as a float16 array of shape (32, 3, 32, 32)."""
        start = int(row["start"])
        return np.array(self._map(int(row["member"]))[int(row["recording"]), start : start + WINDOW])


def count_table(wi: WindowIndex) -> dict[str, list[int]]:
    """Count the windows of each class and return them as a list per split."""
    return {
        split: np.bincount(wi.split_rows(split)["cls"], minlength=N_CLASSES).tolist()
        for split in ("train", "val")
    }


def no_gesture_recording_share(wi: WindowIndex, split: str) -> float:
    """Return the fraction of the split's recordings that contribute at least one no-gesture window."""
    ids = [i for i, m in enumerate(wi.members) if m.split == split]
    total = sum(wi.members[i].entry["n"] for i in ids)
    rows = wi.split_rows(split)
    rows = rows[rows["cls"] == 0]
    with_none = len(np.unique(np.stack([rows["member"], rows["recording"]], axis=1), axis=0)) if len(rows) else 0
    return with_none / total if total else 0.0


def scarcity_warnings(counts: dict[str, list[int]]) -> list[str]:
    out = []
    for split, c in counts.items():
        gestures = [n for n in c[1:] if n > 0]
        if c[0] == 0:
            out.append(
                f"The {split} split has no windows of the no-gesture class, so the model cannot learn it. "
                "Check the window rules in docs/project-rules.md before training."
            )
        elif gestures and c[0] < min(gestures):
            out.append(
                f"The {split} split has only {c[0]} no-gesture windows, which is fewer than the smallest gesture "
                f"class ({min(gestures)}). The no-gesture class may be under-represented."
            )
    return out


def user_balanced_weights(users: np.ndarray) -> np.ndarray:
    """Return per-sample weights that give every user the same total weight. The mean weight is 1."""
    _, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w * (len(users) / w.sum())


class BalancedUserSampler:
    """Draw the same number of windows from every user in each epoch.

    By default the number equals the window count of the smallest user.
    """

    def __init__(self, users: np.ndarray, per_user: int | None = None, seed: int = 0):
        self._by_user = {int(u): np.flatnonzero(users == u) for u in np.unique(users)}
        self.per_user = per_user if per_user is not None else min(len(v) for v in self._by_user.values())
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self.per_user * len(self._by_user)

    def epoch_indices(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng([self._seed, epoch])
        picks = [
            rng.choice(idx, self.per_user, replace=self.per_user > len(idx)) for _, idx in sorted(self._by_user.items())
        ]
        out = np.concatenate(picks)
        rng.shuffle(out)
        return out

    def __iter__(self):
        indices = self.epoch_indices(self._epoch)
        self._epoch += 1
        return iter(indices.tolist())

