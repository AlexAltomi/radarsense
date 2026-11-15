"""Wire format of the replay stream: a 16-byte header followed by one 3x32x32 uint8 frame (3,088 bytes in total).

The header is little-endian, struct format "<IQB3x": a uint32 sequence number, a uint64 server time in milliseconds,
a uint8 true label (255 when unknown) and 3 reserved bytes. Channel 0 holds values in [0, 1] and is stored as
round(x * 255). Channels 1 and 2 hold sin(phase) values in [-1, 1] and are stored as round((x + 1) / 2 * 255).
Encoding and decoding live only in this module.
"""
import struct

import numpy as np

HEADER = struct.Struct("<IQB3x")
HEADER_SIZE = HEADER.size
FRAME_SHAPE = (3, 32, 32)
PAYLOAD_SIZE = 3 * 32 * 32
FRAME_SIZE = HEADER_SIZE + PAYLOAD_SIZE
UNKNOWN_LABEL = 255
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


def pack_frame(seq: int, time_ms: int, label: int, payload: bytes | np.ndarray) -> bytes:
    """Return header + payload. `payload` is one encoded frame (3,072 bytes)."""
    data = payload if isinstance(payload, bytes) else np.ascontiguousarray(payload, dtype=np.uint8).tobytes()
    if len(data) != PAYLOAD_SIZE:
        raise ValueError(f"A frame payload has {PAYLOAD_SIZE} bytes, got {len(data)}.")
    if not 0 <= label <= UNKNOWN_LABEL:
        raise ValueError(f"The label must fit into a uint8, got {label}.")
    return HEADER.pack(seq & SEQ_MASK, time_ms, label) + data


def unpack_frame(message: bytes) -> tuple[int, int, int, np.ndarray]:
    """Return (seq, time_ms, label, frame) with the frame as uint8 of shape (3, 32, 32)."""
    if len(message) != FRAME_SIZE:
        raise ValueError(f"A binary frame has {FRAME_SIZE} bytes, got {len(message)}.")
    seq, time_ms, label = HEADER.unpack_from(message)
    frame = np.frombuffer(message, dtype=np.uint8, offset=HEADER_SIZE).reshape(FRAME_SHAPE)
    return seq, time_ms, label, frame
