"""Data access for training and evaluating the CNN.

Training windows come from the memory-mapped cache, one window at a time, so the cache is never loaded into memory.
The frames in the cache are already scaled: channel 0 lies in 0 to 1 and channels 1 and 2 lie in -1 to 1, so the
model needs no extra normalization.

Augmentation (train windows only, see "Data rules" in docs/project-rules.md):
  - Frame shift: the window start moves by up to +-3 frames and the window is relabelled with the window rule of
    `radarsense_dsp.windows.window_class_at`. If the rule excludes the shifted window, the unshifted window is used.
  - Amplitude: one gain between -3 dB and +3 dB per window, applied on the 0-1 scale of channel 0.
  - Noise: small Gaussian noise.
  There are no flips, no mirroring and no channel swaps.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dsp"))

from radarsense_dsp.windows import (
    FRAMES,
    N_CLASSES,
    WINDOW,
    BalancedUserSampler,
    CacheMember,
    CacheReader,
    window_class_at,
)

MAX_SHIFT = 3
MAX_DB = 3.0
NOISE_STD = 0.01


def span_table(cache_dir: str | Path, members: list[CacheMember]) -> list[np.ndarray]:
    """Return one int32 array of shape (recordings, 3) per member with the columns label, s and L.

    Every recording of a complete cache member holds exactly one contiguous gesture with a single label value (the
    cache builder checks this), so these values can be read from the targets without a loop.
    """
    out = []
    for m in members:
        targets = np.load(Path(cache_dir) / f"{m.stem}.targets.npy")
        nonzero = targets > 0
        out.append(np.stack([targets.max(axis=1), nonzero.argmax(axis=1), nonzero.sum(axis=1)], axis=1).astype(np.int32))
    return out


def apply_shift(label: int, s: int, length: int, start: int, cls: int, shift: int) -> tuple[int, int]:
    """Return (start, class) after shifting a window by `shift` frames.

    The shifted window is relabelled with the window rule. If it falls outside the recording or the rule excludes it,
    the unshifted window and its original class are returned.
    """
    new = start + shift
    if 0 <= new <= FRAMES - WINDOW:
        new_cls = window_class_at(label, s, length, new)
        if new_cls is not None:
            return new, new_cls
    return start, cls


class TrainStream(Dataset):
    """A stream of training windows in which epoch e is the sampler's draw for epoch e.

    Index i belongs to epoch i // epoch_len. The plan of an epoch (which windows, which shifts) depends only on the
    seed and the epoch number, so it is the same for any number of data loader workers.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        members: list[CacheMember],
        rows: np.ndarray,
        sampler: BalancedUserSampler,
        seed: int,
        epoch_len: int,
        n_epochs: int,
        max_shift: int = MAX_SHIFT,
    ):
        self.rows = rows
        self.spans = span_table(cache_dir, members)
        self.reader = CacheReader(cache_dir, members)
        self.sampler = sampler
        self.seed = seed
        self.epoch_len = epoch_len
        self.n_epochs = n_epochs
        self.max_shift = max_shift
        self._plan_cache: tuple[int, np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return self.epoch_len * self.n_epochs

    def plan(self, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        if self._plan_cache is None or self._plan_cache[0] != epoch:
            idx = self.sampler.epoch_indices(epoch)[: self.epoch_len]
            rng = np.random.default_rng([self.seed, 1, epoch])
            shifts = rng.integers(-self.max_shift, self.max_shift + 1, size=len(idx))
            self._plan_cache = (epoch, idx, shifts)
        return self._plan_cache[1], self._plan_cache[2]

    def item(self, epoch: int, pos: int) -> tuple[np.ndarray, int]:
        idx, shifts = self.plan(epoch)
        row = self.rows[idx[pos]].copy()
        label, s, length = (int(v) for v in self.spans[int(row["member"])][int(row["recording"])])
        start, cls = apply_shift(label, s, length, int(row["start"]), int(row["cls"]), int(shifts[pos]))
        row["start"] = start
        return self.reader.window(row), cls

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        window, cls = self.item(*divmod(i, self.epoch_len))
        return torch.from_numpy(window), cls


def augment_batch(x: torch.Tensor, db_range: float, max_db: float = MAX_DB, noise_std: float = NOISE_STD) -> torch.Tensor:
    """Change the amplitude of each window by up to +-max_db and add small Gaussian noise.

    x has shape (batch, frames, 3, 32, 32). Channel 0 is the magnitude in dB scaled to 0-1 over `db_range` dB, so a
    gain of g dB moves it by g / db_range and clips it to 0-1. Cells at 0 hold no reflection and stay at 0. Channels 1
    and 2 are sin(phase difference) times channel 0, so they are scaled by the same factor as channel 0.
    """
    gain = (torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * 2 - 1) * (max_db / db_range)
    mag = x[:, :, 0:1]
    new_mag = torch.where(mag > 0, (mag + gain).clamp(0, 1), mag)
    ratio = new_mag / mag.clamp_min(1e-6)
    out = torch.cat([new_mag, x[:, :, 1:] * ratio], dim=2)
    return out + noise_std * torch.randn_like(out)


@torch.no_grad()
def predict(model: torch.nn.Module, reader: CacheReader, rows: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    """Return the logits of shape (len(rows), N_CLASSES) for the window rows, in the order of `rows`.

    Each recording is read once and all of its windows are cut from it, so a validation pass reads every frame about
    once instead of once for every overlapping window.
    """
    model.eval()
    logits = np.empty((len(rows), N_CLASSES), dtype=np.float32)
    order = np.lexsort((rows["start"], rows["recording"], rows["member"]))
    keys = rows[order]
    change = np.flatnonzero((np.diff(keys["member"]) != 0) | (np.diff(keys["recording"]) != 0)) + 1
    bounds = [0, *change.tolist(), len(keys)]
    pending_x: list[np.ndarray] = []
    pending_pos: list[int] = []

    def flush() -> None:
        if not pending_x:
            return
        x = torch.from_numpy(np.stack(pending_x)).to(device).float()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
        logits[np.array(pending_pos)] = out.float().cpu().numpy()
        pending_x.clear()
        pending_pos.clear()

    for a, b in itertools.pairwise(bounds):
        group = keys[a:b]
        rec = reader.recording(int(group["member"][0]), int(group["recording"][0]))
        for w, pos in zip(group["start"], order[a:b], strict=True):
            pending_x.append(rec[int(w) : int(w) + WINDOW])
            pending_pos.append(int(pos))
        if len(pending_x) >= batch:
            flush()
    flush()
    return logits
