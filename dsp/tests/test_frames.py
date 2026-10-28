import numpy as np
import pytest

from radarsense_dsp.frames import magnitude_db, process_frames, range_doppler_spectra

# A tone with unit amplitude peaks at |X| of about 244 (47.75 dB) after both windowed FFTs.
DB_MIN, DB_MAX = 0.0, 60.0  # The peak stays unclipped, so the position of the maximum is unambiguous.
DB_CLIP_MIN, DB_CLIP_MAX = 0.0, 40.0  # The peak is clipped to 1, so the phase weight is exactly 1.
PAIR_A, PAIR_B = (0, 1), (0, 2)
ATOL = 2e-3  # This tolerance is a few float16 steps for values between 0.5 and 1.


def tone(k, d, phi=(0.0, 0.0, 0.0), amp=1.0):
    n = np.arange(64)[None, None, :]
    c = np.arange(32)[None, :, None]
    ph = np.asarray(phi)[:, None, None]
    return (amp * np.cos(2 * np.pi * k * n / 64 + 2 * np.pi * d * c / 32 + ph)).astype(np.float32)


def run(raw, db_min=DB_MIN, db_max=DB_MAX):
    return process_frames(raw, db_min, db_max, PAIR_A, PAIR_B)


def test_peak_lands_at_range_and_doppler_bin():
    out = run(tone(10, 3))
    assert np.unravel_index(np.argmax(out[0]), (32, 32)) == (10, 16 + 3)


def test_static_target_removed_and_moving_survives():
    moving = tone(10, 3)
    both = moving + tone(20, 0, amp=5.0)  # The static tone would reach about 62 dB if the clutter removal did not suppress it.
    out_both, out_moving = run(both), run(moving)
    assert out_both[0, 20, 16] == 0
    assert np.unravel_index(np.argmax(out_both[0]), (32, 32)) == (10, 19)
    assert np.allclose(out_both, out_moving, rtol=0, atol=ATOL)


def test_phase_channels_follow_pair_difference():
    peak = (10, 19)
    h = run(tone(10, 3, phi=(0.0, 0.5, 0.0)), DB_CLIP_MIN, DB_CLIP_MAX)
    assert h[1][peak] == pytest.approx(np.sin(0.5), abs=ATOL)
    assert h[2][peak] == pytest.approx(0.0, abs=ATOL)

    flipped = run(tone(10, 3, phi=(0.0, -0.5, 0.0)), DB_CLIP_MIN, DB_CLIP_MAX)
    assert flipped[1][peak] == pytest.approx(-np.sin(0.5), abs=ATOL)

    v = run(tone(10, 3, phi=(0.0, 0.0, 0.5)), DB_CLIP_MIN, DB_CLIP_MAX)
    assert v[1][peak] == pytest.approx(0.0, abs=ATOL)
    assert v[2][peak] == pytest.approx(np.sin(0.5), abs=ATOL)


def test_phase_channels_are_weighted_by_scaled_ch0():
    out = run(tone(10, 3, phi=(0.0, 0.5, 0.0)))
    peak_mag = 0.5 * np.hanning(64).sum() * np.hanning(32).sum()  # This is the peak of a unit cosine after both windowed FFTs.
    w = 20 * np.log10(peak_mag) / (DB_MAX - DB_MIN)
    assert out[0][10, 19] == pytest.approx(w, abs=ATOL)
    assert out[1][10, 19] == pytest.approx(w * np.sin(0.5), abs=ATOL)


def test_spectra_phase_sign_and_shape():
    s = range_doppler_spectra(tone(10, 3, phi=(0.0, 0.5, -0.25)))
    assert s.shape == (3, 32, 32) and np.iscomplexobj(s)
    peak = (10, 19)
    # The phase difference is delta_phi = angle(ant_b) - angle(ant_a).
    assert np.angle(s[1][peak] * np.conj(s[0][peak])) == pytest.approx(0.5, abs=1e-3)
    assert np.angle(s[2][peak] * np.conj(s[0][peak])) == pytest.approx(-0.25, abs=1e-3)


def test_magnitude_db_peak_and_batch_shape():
    peak_mag = 0.5 * np.hanning(64).sum() * np.hanning(32).sum()
    db = magnitude_db(range_doppler_spectra(np.stack([tone(10, 3), tone(5, -2)])))
    assert db.shape == (2, 32, 32)
    assert np.unravel_index(np.argmax(db[0]), (32, 32)) == (10, 19)
    assert db[0][10, 19] == pytest.approx(20 * np.log10(peak_mag), abs=1e-2)
    assert np.unravel_index(np.argmax(db[1]), (32, 32)) == (5, 16 - 2)


def test_shape_dtype_and_ranges():
    rng = np.random.default_rng(1)
    raw = tone(10, 3) + 0.5 * rng.standard_normal((3, 32, 64)).astype(np.float32)
    out = run(raw)
    assert out.shape == (3, 32, 32)
    assert out.dtype == np.float16
    assert np.isfinite(out).all()
    assert out[0].min() >= 0 and out[0].max() <= 1
    assert np.abs(out[1:]).max() <= 1


def test_batch_equals_frame_by_frame():
    rng = np.random.default_rng(2)
    raw = np.stack([tone(k, d, phi=(0.0, 0.3, -0.2)) for k, d in [(5, -4), (10, 3), (22, 0), (30, 7)]])
    raw += 0.2 * rng.standard_normal(raw.shape).astype(np.float32)
    batch = run(raw)
    assert batch.shape == (4, 3, 32, 32)
    one_by_one = np.stack([run(f) for f in raw])
    assert np.allclose(batch, one_by_one, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    "db_min, db_max, pair_a, pair_b",
    [(10.0, 10.0, (0, 1), (0, 2)), (10.0, 5.0, (0, 1), (0, 2)), (0.0, 60.0, (1, 1), (0, 2)),
     (0.0, 60.0, (0, 1), (0, 3))],
)
def test_bad_arguments_raise(db_min, db_max, pair_a, pair_b):
    with pytest.raises(ValueError):
        process_frames(tone(10, 3), db_min, db_max, pair_a, pair_b)
