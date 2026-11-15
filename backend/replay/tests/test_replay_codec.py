import numpy as np
import pytest

from replay import codec


def test_sizes():
    assert codec.HEADER_SIZE == 16
    assert codec.PAYLOAD_SIZE == 3072
    assert codec.FRAME_SIZE == 3088


def test_header_is_little_endian_with_the_documented_layout():
    msg = codec.pack_frame(0x01020304, 0x1122334455667788, 7, bytes(3072))
    assert codec.pack_frame(0, 0, 0, bytes(3072), codec.FLAG_CLIP_START)[13] == 1
    assert len(msg) == 3088
    assert msg[:4] == bytes([4, 3, 2, 1])
    assert msg[4:12] == bytes([0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11])
    assert msg[12] == 7
    assert msg[13] == 0
    assert msg[14:16] == bytes(2)


def test_pack_unpack_roundtrip():
    payload = np.arange(3072, dtype=np.uint32).astype(np.uint8).reshape(3, 32, 32)
    seq, ms, label, flags, frame = codec.unpack_frame(codec.pack_frame(123456, 1_700_000_000_123, 255, payload, 1))
    assert (seq, ms, label, flags) == (123456, 1_700_000_000_123, 255, 1)
    assert np.array_equal(frame, payload)


def test_sequence_wraps_at_32_bits():
    seq = codec.unpack_frame(codec.pack_frame(2**32 + 5, 0, 0, bytes(3072))).seq
    assert seq == 5


def test_bad_sizes_are_rejected():
    with pytest.raises(ValueError):
        codec.pack_frame(0, 0, 0, bytes(10))
    with pytest.raises(ValueError):
        codec.unpack_frame(bytes(3087))
    with pytest.raises(ValueError):
        codec.pack_frame(0, 0, 256, bytes(3072))


def test_encode_decode_roundtrip_error_is_at_most_one_step():
    rng = np.random.default_rng(0)
    x = np.empty((50, 3, 32, 32), dtype=np.float32)
    x[:, 0] = rng.random((50, 32, 32))
    x[:, 1:] = rng.uniform(-1, 1, (50, 2, 32, 32))
    x[0, :, 0, 0] = [0.0, -1.0, 1.0]  # the end points
    x[0, :, 0, 1] = [1.0, 1.0, -1.0]
    u = codec.encode_frames(x)
    assert u.dtype == np.uint8 and u.shape == x.shape
    back = codec.decode_frames(u)
    assert np.abs(back[:, 0] - x[:, 0]).max() <= 1 / 255  # the range of channel 0 is 1
    assert np.abs(back[:, 1:] - x[:, 1:]).max() <= 2 / 255  # the range of channels 1 and 2 is 2


def test_encoding_formula_and_clipping():
    x = np.zeros((3, 32, 32), dtype=np.float32)
    x[0, 0, 0], x[0, 0, 1], x[0, 0, 2] = 0.5, 1.5, -0.2
    x[1, 0, 0], x[2, 0, 0] = -1.0, 1.0
    x[1, 0, 1] = 0.0
    u = codec.encode_frames(x)
    assert u[0, 0, 0] == round(0.5 * 255)
    assert u[0, 0, 1] == 255 and u[0, 0, 2] == 0
    assert u[1, 0, 0] == 0 and u[2, 0, 0] == 255
    assert u[1, 0, 1] == round(0.5 * 255)
    assert u[1, 5, 5] == round(0.5 * 255)  # a zero phase value is the middle of the range


def test_encode_rejects_wrong_shape():
    with pytest.raises(ValueError):
        codec.encode_frames(np.zeros((3, 32, 64)))
