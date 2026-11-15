"""The in-memory index of the processed cache, built once at startup.

Only members whose index.json status is "complete" are used, and only through `radarsense_dsp.windows`, which refuses
the test users. Frames and targets are opened with np.load(mmap_mode="r"), so nothing is read until a recording is
streamed. The cache never contains user10 or user12, and this module has no way to name them.
"""
import re
from pathlib import Path

import numpy as np

from radarsense_dsp.windows import (
    CLASS_MAP,
    CLASS_NAMES,
    FRAMES,
    WINDOW,
    CacheMember,
    CacheReader,
    far_span,
    gesture_span,
    load_complete_members,
)

REC_DTYPE = np.dtype(
    [
        ("member", "i4"),
        ("recording", "i4"),
        ("user", "i2"),
        ("session", "i2"),
        ("label", "i1"),  # The dataset label value of the gesture.
        ("cls", "i1"),  # CLASS_MAP[label].
        ("s", "i2"),
        ("length", "i2"),
    ]
)

BACKGROUND = "background"
ANY = "any"
GESTURE_NAMES = CLASS_NAMES[1:]  # Class 0 ("none") has no recordings of its own. Use "background" for it.
_USER_RE = re.compile(r"user(\d+)")


class ReplayError(Exception):
    """A request that cannot be served. The message goes to the app in an error message."""


def background_ranges(s: int, length: int) -> list[range]:
    """Return the frame ranges of a recording that lie outside the zone [s-20, s+L+9], in order."""
    far_lo, far_hi = far_span(s, length)
    out = [range(min(far_lo, FRAMES)), range(max(far_hi + 1, 0), FRAMES)]
    return [r for r in out if len(r) > 0]


def usable_background_runs(s: int, length: int) -> list[range]:
    """Return the background runs that fill the detector buffer: only runs of at least WINDOW frames."""
    return [r for r in background_ranges(s, length) if len(r) >= WINDOW]


class ReplayIndex:
    def __init__(self, cache_dir: str | Path, splits: dict):
        self.cache_dir = Path(cache_dir)
        self.val_users = sorted(int(u.removeprefix("user")) for u in splits.get("val", []))
        self.members: list[CacheMember] = load_complete_members(self.cache_dir, splits, nominal_only=True)
        self.reader = CacheReader(self.cache_dir, self.members)
        self._targets: dict[int, np.ndarray] = {}
        rows = []
        self.recording_counts: dict[tuple[int, int], int] = {}
        for member_id, m in enumerate(self.members):
            targets = self.targets(member_id)
            key = (m.user, int(m.entry["session"]))
            self.recording_counts[key] = self.recording_counts.get(key, 0) + len(targets)
            for r in range(len(targets)):
                try:
                    label, s, length = gesture_span(np.asarray(targets[r]))
                except ValueError:
                    continue  # The cache builder rejects such files, so this is not expected.
                if label not in CLASS_MAP:
                    continue
                rows.append((member_id, r, m.user, key[1], label, CLASS_MAP[label], s, length))
        self.recs = np.array(rows, dtype=REC_DTYPE)
        self.background_frames = np.array(
            [sum(len(r) for r in usable_background_runs(int(x["s"]), int(x["length"]))) for x in self.recs], dtype=np.int32
        )
        self.users = sorted({int(u) for u in np.unique(self.recs["user"])})

    def targets(self, member: int) -> np.ndarray:
        if member not in self._targets:
            path = self.cache_dir / f"{self.members[member].stem}.targets.npy"
            self._targets[member] = np.load(path, mmap_mode="r")
        return self._targets[member]

    def __len__(self) -> int:
        return len(self.recs)

    def pick(self, rng: np.random.Generator, user: str, gesture: str) -> tuple[np.void, bool]:
        """Choose a random recording. Return it and whether its background frames are wanted."""
        if gesture not in (ANY, BACKGROUND, *GESTURE_NAMES):
            raise ReplayError(f"Unknown gesture {gesture!r}. Use 'any', 'background' or one of {list(GESTURE_NAMES)}.")
        mask = np.isin(self.recs["user"], self._users_for(user))
        background = gesture == BACKGROUND
        if background:
            mask &= self.background_frames > 0
        elif gesture != ANY:
            mask &= self.recs["cls"] == CLASS_NAMES.index(gesture)
        candidates = np.flatnonzero(mask)
        if len(candidates) == 0:
            what = f"a background run of at least {WINDOW} frames" if background else "a recording"
            raise ReplayError(f"No {what} matches user {user!r} and gesture {gesture!r}.")
        return self.recs[int(rng.choice(candidates))], background

    def _users_for(self, user: str) -> list[int]:
        if user == ANY:
            # The default users are the validation users. Other cached users must be named.
            default = [u for u in self.val_users if u in self.users]
            if not default:
                raise ReplayError("No validation user is cached.")
            return default
        match = _USER_RE.fullmatch(user) if isinstance(user, str) else None
        if match is None or int(match.group(1)) not in self.users:
            names = [f"user{u}" for u in self.users]
            raise ReplayError(f"Unknown user {user!r}. Use 'any' or one of {names}.")
        return [int(match.group(1))]
