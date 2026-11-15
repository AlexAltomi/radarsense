"""Wire format of the replay stream: a 16-byte header followed by one 3x32x32 uint8 frame (3,088 bytes in total).

The header is little-endian, struct format "<IQBB2x": a uint32 sequence number, a uint64 server time in milliseconds,
a uint8 true label (255 when unknown), a uint8 flags byte and 2 reserved bytes. Bit 0 of the flags is set on the first
frame of a clip, which is a discontinuity: the receiver must clear its frame buffer. Channel 0 holds values in [0, 1]
and is stored as
round(x * 255). Channels 1 and 2 hold sin(phase) values in [-1, 1] and are stored as round((x + 1) / 2 * 255).
Encoding and decoding live only in this module.
"""
import struct
from typing import NamedTuple

import numpy as np

HEADER = struct.Struct("<IQBB2x")
HEADER_SIZE = HEADER.size
FRAME_SHAPE = (3, 32, 32)
PAYLOAD_SIZE = 3 * 32 * 32
FRAME_SIZE = HEADER_SIZE + PAYLOAD_SIZE
UNKNOWN_LABEL = 255
FLAG_CLIP_START = 0x01  # Bit 0 of the flags byte: the first frame of a clip (a discontinuity).
SEQ_MASK = 0xFFFFFFFF


def encode_frames(x: np.ndarray) -> np.ndarray:
    """Encode float frames of shape (..., 3, 32, 32) to uint8 of the same shape. Values outside the range are clipped."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-3:] != FRAME_SHAPE:
        raise ValueError(f"Expected frames with shape (..., 3, 32, 32), got {x.shape}.")
    out = np.empty(x.shape, dtype=np.uint8)
    out[..., 0, :, :] = np.rint(np.clip(x[..., 0, :, :], 0.0, 1.0) * 255.0)
    out[..., 1:, :, :] = np.rint((np.clip(x[..., 1:, :, :], -1.0, 1.0) + 1.0) / 2.0 * 255.0)
    return out


def decode_frames(u: np.ndarray) -> np.ndarray:
    """Decode uint8 frames of shape (..., 3, 32, 32) back to float32 (channel 0 in [0, 1], channels 1 and 2 in [-1, 1])."""
    u = np.asarray(u)
    if u.dtype != np.uint8 or u.shape[-3:] != FRAME_SHAPE:
        raise ValueError(f"Expected uint8 frames with shape (..., 3, 32, 32), got {u.dtype} {u.shape}.")
    f = u.astype(np.float32)
    out = np.empty(u.shape, dtype=np.float32)
    out[..., 0, :, :] = f[..., 0, :, :] / 255.0
    out[..., 1:, :, :] = f[..., 1:, :, :] / 255.0 * 2.0 - 1.0
    return out


def pack_frame(seq: int, time_ms: int, label: int, payload: bytes | np.ndarray, flags: int = 0) -> bytes:
    """Return header + payload. `payload` is one encoded frame (3,072 bytes)."""
    data = payload if isinstance(payload, bytes) else np.ascontiguousarray(payload, dtype=np.uint8).tobytes()
    if len(data) != PAYLOAD_SIZE:
        raise ValueError(f"A frame payload has {PAYLOAD_SIZE} bytes, got {len(data)}.")
    if not 0 <= label <= UNKNOWN_LABEL:
        raise ValueError(f"The label must fit into a uint8, got {label}.")
    if not 0 <= flags <= 255:
        raise ValueError(f"The flags must fit into a uint8, got {flags}.")
    return HEADER.pack(seq & SEQ_MASK, time_ms, label, flags) + data


class Frame(NamedTuple):
    seq: int
    time_ms: int
    label: int
    flags: int
    frame: np.ndarray  # uint8, shape (3, 32, 32)

    @property
    def clip_start(self) -> bool:
        return bool(self.flags & FLAG_CLIP_START)


def unpack_frame(message: bytes) -> Frame:
    """Split a binary message into its header fields and the uint8 frame of shape (3, 32, 32)."""
    if len(message) != FRAME_SIZE:
        raise ValueError(f"A binary frame has {FRAME_SIZE} bytes, got {len(message)}.")
    seq, time_ms, label, flags = HEADER.unpack_from(message)
    frame = np.frombuffer(message, dtype=np.uint8, offset=HEADER_SIZE).reshape(FRAME_SHAPE)
    return Frame(seq, time_ms, label, flags, frame)
