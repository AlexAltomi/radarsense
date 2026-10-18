"""Streaming access to the radar_dataset zip, one recording at a time.

Never extracts the zip and never loads a whole `inputs` array (CLAUDE.md hard rules 1 and 6).
"""
import os
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEST_USERS = frozenset({10, 12})
ALLOW_TEST_ENV = "RADARSENSE_ALLOW_TEST"

FRAME_SHAPE = (100, 3, 32, 64)
_NAME_RE = re.compile(r"^(?:.*/)?user(\d+)_e(\d+)(?:_(fast|slow|wrist))?\.npz$")


@dataclass(frozen=True)
class Member:
    name: str
    user: int
    session: int
    anomaly: str | None


def parse_member(name: str) -> Member:
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"not a radar_dataset member name: {name!r}")
    return Member(name, int(m[1]), int(m[2]), m[3])


def list_members(zip_path: str | Path) -> list[Member]:
    """Names only; nothing is opened, so this is allowed for every user."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".npz")]
    return sorted((parse_member(n) for n in names), key=lambda m: m.name)


def _check_allowed(user: int) -> None:
    if user in TEST_USERS and os.environ.get(ALLOW_TEST_ENV) != "1":
        raise PermissionError(f"user{user} is a test user and may not be loaded before Week 11")


def _read_header(f) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    raise ValueError(f"unsupported .npy format version {version}")


def _read_into(f, buf: memoryview) -> None:
    """Fill buf completely; a short read before that means a truncated file."""
    got = 0
    while got < len(buf):
        n = f.readinto(buf[got:])
        if not n:
            raise EOFError("unexpected end of inputs.npy")
        got += n


def iter_recordings(
    zip_path: str | Path, member: str | Member
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (inputs[100,3,32,64] float32, targets[100] int32), one recording at a time."""
    m = member if isinstance(member, Member) else parse_member(member)
    _check_allowed(m.user)
    with zipfile.ZipFile(zip_path) as outer, outer.open(m.name) as raw, zipfile.ZipFile(raw) as npz:
        with npz.open("targets.npy") as tf:
            targets = np.lib.format.read_array(tf)
        with npz.open("inputs.npy") as f:
            shape, fortran_order, dtype = _read_header(f)
            if dtype != np.dtype("<f4") or fortran_order:
                raise ValueError(f"unexpected inputs layout: dtype={dtype} fortran={fortran_order}")
            if shape[1:] != FRAME_SHAPE or shape[0] != len(targets):
                raise ValueError(f"inputs {shape} does not match targets {targets.shape}")
            for i in range(shape[0]):
                rec = np.empty(FRAME_SHAPE, dtype=np.float32)
                _read_into(f, memoryview(rec).cast("B"))
                yield rec, targets[i].copy()
