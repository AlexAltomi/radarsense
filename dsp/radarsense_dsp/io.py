"""Streaming access to the radar_dataset zip, one recording at a time.

This module never extracts the zip file and never loads a whole `inputs` array into memory. It also refuses to open
the test users (user10 and user12) unless the environment variable RADARSENSE_ALLOW_TEST is set to 1. Both rules are
described in docs/project-rules.md under "Data rules".
"""
import os
import re
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
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
        raise ValueError(f"{name!r} is not a radar_dataset member name. Expected a name like 'fulldataset/user1_e1.npz'.")
    return Member(name, int(m[1]), int(m[2]), m[3])


def list_members(zip_path: str | Path) -> list[Member]:
    """List the member names. Nothing is opened, so this is allowed for every user."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".npz")]
    return sorted((parse_member(n) for n in names), key=lambda m: m.name)


def _check_allowed(user: int) -> None:
    if user in TEST_USERS and os.environ.get(ALLOW_TEST_ENV) != "1":
        raise PermissionError(
            f"user{user} is a test user and may not be loaded before Week 11. "
            "See the data rules in docs/project-rules.md."
        )


def _read_header(f) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    raise ValueError(f"The .npy format version {version} is not supported. Only versions 1.0 and 2.0 can be read.")


def _read_into(f, buf: memoryview) -> None:
    """Fill the buffer completely. A short read means that the file is truncated."""
    got = 0
    while got < len(buf):
        n = f.readinto(buf[got:])
        if not n:
            raise EOFError("The inputs.npy stream ended early, so the archive is probably truncated. Check the zip file.")
        got += n


@contextmanager
def _open_member(zip_path: str | Path, member: str | Member):
    """Yield the inner .npz ZipFile of one member after checking that the user may be loaded."""
    m = member if isinstance(member, Member) else parse_member(member)
    _check_allowed(m.user)
    with zipfile.ZipFile(zip_path) as outer, outer.open(m.name) as raw, zipfile.ZipFile(raw) as npz:
        yield npz


def _read_targets(npz: zipfile.ZipFile) -> np.ndarray:
    with npz.open("targets.npy") as tf:
        return np.lib.format.read_array(tf)


def _check_inputs_header(f, targets: np.ndarray) -> tuple[int, ...]:
    shape, fortran_order, dtype = _read_header(f)
    if dtype != np.dtype("<f4") or fortran_order:
        raise ValueError(f"The inputs array must be little-endian float32 in C order, but it has dtype {dtype} and fortran_order={fortran_order}.")
    if shape[1:] != FRAME_SHAPE or shape[0] != len(targets):
        raise ValueError(f"The inputs array has shape {shape}, which does not match the targets array of shape {targets.shape}.")
    return shape


def read_meta(zip_path: str | Path, member: str | Member) -> tuple[int, np.ndarray]:
    """Return the number of recordings N and the targets array of shape (N, 100). Only headers are read, never the inputs data."""
    with _open_member(zip_path, member) as npz:
        targets = _read_targets(npz)
        with npz.open("inputs.npy") as f:
            shape = _check_inputs_header(f, targets)
    return shape[0], targets


def _iter_inputs(f, n: int) -> Iterator[np.ndarray]:
    for _ in range(n):
        rec = np.empty(FRAME_SHAPE, dtype=np.float32)
        _read_into(f, memoryview(rec).cast("B"))
        yield rec


@contextmanager
def open_recordings(zip_path: str | Path, member: str | Member):
    """Yield the targets array of shape (N, frames) and a one-pass iterator over the recordings.

    The targets are read first, so that a caller can inspect them before paying for the inputs. Each recording
    has shape (100, 3, 32, 64) and dtype float32.
    """
    with _open_member(zip_path, member) as npz:
        targets = _read_targets(npz)
        with npz.open("inputs.npy") as f:
            shape = _check_inputs_header(f, targets)
            yield targets, _iter_inputs(f, shape[0])


def iter_recordings(
    zip_path: str | Path, member: str | Member
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield one recording at a time as a pair of inputs (100, 3, 32, 64) float32 and targets (100,) int32."""
    with open_recordings(zip_path, member) as (targets, recordings):
        for i, rec in enumerate(recordings):
            yield rec, targets[i].copy()
