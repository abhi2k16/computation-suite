"""
test_random_vibration_time_history.py -- Wave 12 item 116 (docs/
consolidated_future_roadmap.md, ICE-ROM/GAP_ANALYSIS.md gap #5):
validates band_limited_gaussian_time_history(), promoted unchanged in
method from ICE-ROM/validation/dynamic_comparison.py's own ad hoc
prototype (already used there to validate a real flat-beam ICE-ROM
dynamic reproduction).
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from rom_engine import band_limited_gaussian_time_history


def test_output_length_and_rms_match_request():
    rng = np.random.default_rng(0)
    n = 4001
    x = band_limited_gaussian_time_history(n, dt=5e-5, f_max=1500.0, rms=0.72, rng=rng)
    assert x.shape == (n,)
    assert np.isclose(np.sqrt(np.mean(x ** 2)), 0.72, rtol=1e-9)


def test_spectrum_has_no_power_above_f_max():
    rng = np.random.default_rng(1)
    n, dt, f_max = 2001, 5e-5, 1500.0
    x = band_limited_gaussian_time_history(n, dt=dt, f_max=f_max, rms=1.0, rng=rng)
    freqs = np.fft.rfftfreq(n, dt)
    X = np.fft.rfft(x)
    power_above = np.sum(np.abs(X[freqs > f_max]) ** 2)
    power_total = np.sum(np.abs(X) ** 2)
    assert power_above / power_total < 1e-20   # exactly zeroed bins, not just attenuated


def test_deterministic_given_same_rng_state():
    x1 = band_limited_gaussian_time_history(500, dt=1e-4, f_max=1000.0, rms=1.0,
                                              rng=np.random.default_rng(42))
    x2 = band_limited_gaussian_time_history(500, dt=1e-4, f_max=1000.0, rms=1.0,
                                              rng=np.random.default_rng(42))
    assert np.array_equal(x1, x2)


def test_different_seeds_give_different_signals():
    x1 = band_limited_gaussian_time_history(500, dt=1e-4, f_max=1000.0, rms=1.0,
                                              rng=np.random.default_rng(1))
    x2 = band_limited_gaussian_time_history(500, dt=1e-4, f_max=1000.0, rms=1.0,
                                              rng=np.random.default_rng(2))
    assert not np.allclose(x1, x2)


def test_rejects_f_max_above_nyquist():
    with pytest.raises(ValueError, match="nyquist"):
        band_limited_gaussian_time_history(100, dt=1e-3, f_max=600.0, rms=1.0)


def test_rejects_non_positive_f_max():
    with pytest.raises(ValueError):
        band_limited_gaussian_time_history(100, dt=1e-3, f_max=0.0, rms=1.0)


def test_rejects_bad_n_samples_and_dt():
    with pytest.raises(ValueError):
        band_limited_gaussian_time_history(1, dt=1e-3, f_max=100.0, rms=1.0)
    with pytest.raises(ValueError):
        band_limited_gaussian_time_history(100, dt=0.0, f_max=100.0, rms=1.0)


def test_unseeded_call_still_works():
    x = band_limited_gaussian_time_history(200, dt=1e-3, f_max=100.0, rms=2.0)
    assert x.shape == (200,)
    assert np.isclose(np.sqrt(np.mean(x ** 2)), 2.0, rtol=1e-9)
