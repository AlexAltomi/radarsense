"""Radar frame -> 3x32x32 range-Doppler/phase image (the DSP spec in CLAUDE.md).

Input is a raw real ADC frame (3 antennas, 32 chirps, 64 samples); leading batch dimensions are allowed.
Output layout is [channel, range, Doppler] = (3, 32, 32), float16:
  ch0  magnitude in dB averaged over the 3 antennas, clipped to [db_min, db_max] and scaled to 0..1
  ch1  sin(delta_phi) of h_pair, weighted by ch0 (the scaled magnitude, so empty cells give 0)
  ch2  sin(delta_phi) of v_pair, weighted by ch0

Conventions:
  delta_phi = angle(ant_b) - angle(ant_a) for a pair (a, b), taken per range-Doppler cell.
  np.fft sign convention (e^{-j...}); a tone cos(2*pi*k*n/64 + 2*pi*d*c/32) peaks at range bin k and Doppler
  bin 16 + d (after fftshift, so bin 16 is zero Doppler).

UNVERIFIED (to be confirmed on train-user gestures in the label/antenna task):
  - the Doppler sign versus the physical direction of motion (positive d = towards or away from the radar);
  - which antenna pair is horizontal and which is vertical, and the sign of each pair's delta_phi.
"""
import numpy as np

N_ANT, N_CHIRP, N_SAMPLE, N_RANGE = 3, 32, 64, 32
_RAW_SHAPE = (N_ANT, N_CHIRP, N_SAMPLE)

_HANN_SAMPLE = np.hanning(N_SAMPLE).astype(np.float32)
_HANN_CHIRP = np.hanning(N_CHIRP).astype(np.float32)[:, None]


def _check_pair(pair: tuple[int, int], name: str) -> tuple[int, int]:
    a, b = pair
    if a == b or not (0 <= a < N_ANT and 0 <= b < N_ANT):
        raise ValueError(f"{name} must be two different antenna indices in 0..{N_ANT - 1}, got {pair}")
    return a, b


def process_frames(
    raw: np.ndarray,
    db_min: float,
    db_max: float,
    h_pair: tuple[int, int],
    v_pair: tuple[int, int],
    eps: float = 1e-12,
) -> np.ndarray:
    """Process raw frames (..., 3, 32, 64) into float16 images (..., 3, 32, 32)."""
    if not db_min < db_max:
        raise ValueError(f"need db_min < db_max, got {db_min}, {db_max}")
    ha, hb = _check_pair(h_pair, "h_pair")
    va, vb = _check_pair(v_pair, "v_pair")
    x = np.asarray(raw, dtype=np.float32)
    if x.shape[-3:] != _RAW_SHAPE:
        raise ValueError(f"expected (..., {_RAW_SHAPE}) frames, got {x.shape}")

    x = x - x.mean(axis=-1, keepdims=True)  # per-chirp DC removal
    r = np.fft.fft(x * _HANN_SAMPLE, axis=-1)[..., :N_RANGE]  # (..., ant, chirp, range)
    r = r - r.mean(axis=-2, keepdims=True)  # clutter: mean over chirps per range bin
    d = np.fft.fftshift(np.fft.fft(r * _HANN_CHIRP, axis=-2), axes=-2)  # (..., ant, doppler, range)
    d = np.swapaxes(d, -1, -2)  # (..., ant, range, doppler)

    db = (20 * np.log10(np.abs(d) + eps)).mean(axis=-3)
    ch0 = np.clip((db - db_min) / (db_max - db_min), 0.0, 1.0)

    def phase(a: int, b: int) -> np.ndarray:
        z = d[..., b, :, :] * np.conj(d[..., a, :, :])
        return z.imag / (np.abs(z) + eps)

    out = np.stack([ch0, ch0 * phase(ha, hb), ch0 * phase(va, vb)], axis=-3)
    return out.astype(np.float16)
