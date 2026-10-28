"""Convert one raw radar frame into the 3x32x32 image that the classifier reads.

The processing steps are described under "Signal processing" in docs/project-rules.md. The input is a raw real ADC
(analog-to-digital converter) frame with 3 antennas, 32 chirps and 64 samples per chirp. Leading batch dimensions are
allowed. The output is a float16 array with the layout [channel, range, Doppler], which is (3, 32, 32).

A range-Doppler map shows, for every distance (range bin) and every radial speed (Doppler bin), how strongly the
radar sees a reflection. It is computed with two FFTs (fast Fourier transforms) with a Hann window, a smooth taper
that reduces spectral leakage. The three output channels are:

  ch0  The magnitude in dB averaged over the 3 antennas, clipped to [db_min, db_max] and scaled to 0..1.
  ch1  sin(delta_phi) of the antenna pair given by `pair_a`, called axis A (labels 2 and 3), weighted by ch0.
       Because ch0 is the scaled magnitude, empty cells give 0.
  ch2  sin(delta_phi) of the antenna pair given by `pair_b`, called axis B (labels 6 and 7), weighted by ch0.

Conventions used in this module:

  For an antenna pair (a, b), delta_phi = angle(ant_b) - angle(ant_a), taken per range-Doppler cell.
  The FFT uses the NumPy sign convention e^{-j...}. A tone cos(2*pi*k*n/64 + 2*pi*d*c/32) therefore peaks at range
  bin k and at Doppler bin 16 + d after fftshift, so Doppler bin 16 is zero Doppler.
  Antenna 2 is the corner of the L-shaped arrangement. The pairs (0,2) and (1,2) are the orthogonal baselines, and the
  diagonal pair (0,1) is not used. This was observed on train users 2, 4, 9 and 11 (figure produced at commit
  2bca1f5).

Not yet verified: whether an approaching hand has a negative signed Doppler (bins below 16). The same figure
suggests it.
Not yet verified: which axis (A or B) is horizontal and which is vertical, and the direction within each axis.
For this reason the code contains no left, right, up or down names.
"""
import numpy as np

N_ANT, N_CHIRP, N_SAMPLE, N_RANGE = 3, 32, 64, 32
_RAW_SHAPE = (N_ANT, N_CHIRP, N_SAMPLE)

_HANN_SAMPLE = np.hanning(N_SAMPLE).astype(np.float32)
_HANN_CHIRP = np.hanning(N_CHIRP).astype(np.float32)[:, None]


def _check_pair(pair: tuple[int, int], name: str) -> tuple[int, int]:
    a, b = pair
    if a == b or not (0 <= a < N_ANT and 0 <= b < N_ANT):
        raise ValueError(f"{name} must be two different antenna indices between 0 and {N_ANT - 1}, but it is {pair}.")
    return a, b


def range_doppler_spectra(raw: np.ndarray) -> np.ndarray:
    """Return the complex range-Doppler spectra, shape (..., 3, range, Doppler), of raw frames (..., 3, 32, 64)."""
    x = np.asarray(raw, dtype=np.float32)
    if x.shape[-3:] != _RAW_SHAPE:
        raise ValueError(f"Expected frames with trailing shape {_RAW_SHAPE}, but the input has shape {x.shape}.")
    x = x - x.mean(axis=-1, keepdims=True)  # Remove the constant (DC) offset of every chirp.
    r = np.fft.fft(x * _HANN_SAMPLE, axis=-1)[..., :N_RANGE]  # Shape: (..., antenna, chirp, range).
    r = r - r.mean(axis=-2, keepdims=True)  # Remove static clutter by subtracting the mean over the chirps for each range bin.
    d = np.fft.fftshift(np.fft.fft(r * _HANN_CHIRP, axis=-2), axes=-2)  # Shape: (..., antenna, Doppler, range).
    return np.swapaxes(d, -1, -2)  # Shape: (..., antenna, range, Doppler).


def magnitude_db(spectra: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return the unclipped magnitude in dB, averaged over the antennas, with shape (..., range, Doppler)."""
    return (20 * np.log10(np.abs(spectra) + eps)).mean(axis=-3)


def process_frames(
    raw: np.ndarray,
    db_min: float,
    db_max: float,
    pair_a: tuple[int, int],
    pair_b: tuple[int, int],
    eps: float = 1e-12,
) -> np.ndarray:
    """Turn raw frames of shape (..., 3, 32, 64) into float16 images of shape (..., 3, 32, 32)."""
    if not db_min < db_max:
        raise ValueError(f"db_min must be smaller than db_max, but they are {db_min} and {db_max}.")
    a0, a1 = _check_pair(pair_a, "pair_a")
    b0, b1 = _check_pair(pair_b, "pair_b")
    d = range_doppler_spectra(raw)

    db = magnitude_db(d, eps)
    ch0 = np.clip((db - db_min) / (db_max - db_min), 0.0, 1.0)

    def phase(a: int, b: int) -> np.ndarray:
        z = d[..., b, :, :] * np.conj(d[..., a, :, :])
        return np.imag(z) / (np.abs(z) + eps)

    out = np.stack([ch0, ch0 * phase(a0, a1), ch0 * phase(b0, b1)], axis=-3)
    return out.astype(np.float16)
