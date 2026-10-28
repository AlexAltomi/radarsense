import numpy as np
import pytest
from infer_labels_antennas import PRE, window_features

from radarsense_dsp.frames import range_doppler_spectra

START, LENGTH, SLOPE = 40, 8, 0.1


def synthetic_recording():
    rng = np.random.default_rng(0)
    raw = 1e-3 * rng.standard_normal((100, 3, 32, 64)).astype(np.float32)
    n = np.arange(64)[None, None, :]
    c = np.arange(32)[None, :, None]
    for t in range(LENGTH):
        phi = np.array([0.0, SLOPE * t, 0.0])[:, None, None]
        raw[START + t] += np.cos(2 * np.pi * 12 * n / 64 + 2 * np.pi * 3 * c / 32 + phi).astype(np.float32)
    return raw


def test_window_features_recover_range_doppler_and_phase_slope():
    f = window_features(range_doppler_spectra(synthetic_recording()), START, LENGTH)
    assert f["range"].shape == (31,) and f["phasors"].shape == (3, 31)
    assert (f["range"][PRE : PRE + LENGTH] == 12).all()
    assert f["doppler"][PRE : PRE + LENGTH] == pytest.approx(3, abs=0.5)
    assert f["slopes"][0] == pytest.approx(SLOPE, abs=1e-2)  # For the pair (0,1), delta_phi = angle(ant1) - angle(ant0).
    assert f["slopes"][1] == pytest.approx(0, abs=1e-2)  # This is the pair (0,2).
    assert f["slopes"][2] == pytest.approx(-SLOPE, abs=1e-2)  # For the pair (1,2), delta_phi = angle(ant2) - angle(ant1).
