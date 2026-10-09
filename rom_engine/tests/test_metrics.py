"""
test_metrics.py -- validates rom_engine.metrics.modal_assurance_criterion
and rom_engine.metrics.r_squared.

Checks (modal_assurance_criterion):
  1. A vector against itself gives MAC = 1.
  2. Two orthogonal vectors give MAC = 0.
  3. MAC is invariant to real scaling and (for complex vectors) to an
     arbitrary overall phase rotation -- the two invariances the metric
     is specifically designed to have.
  4. A real eigenbasis from a real fea_engine cantilever: each mode
     against itself is 1, distinct modes (K/M-orthogonal, hence also
     Euclidean-independent for a generic beam) have MAC well below 1 --
     a real-model sanity check, not just a synthetic-vector check.
  5. Mismatched shapes and a zero vector both raise, rather than
     silently returning a meaningless number.

Checks (r_squared):
  6. A perfect prediction gives R^2 = 1.
  7. Predicting the constant mean of y_true for every sample gives
     R^2 = 0 exactly (the metric's own defining baseline).
  8. A prediction worse than the constant-mean baseline gives a
     NEGATIVE R^2 -- a real, meaningful outcome this function must not
     clip away.
  9. Multi-dimensional (e.g. (n_modes, n_samples)) inputs are handled
     via flattening, not rejected.
  10. Mismatched shapes and a zero-variance y_true both raise.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest
from rom_engine.metrics import modal_assurance_criterion, r_squared
from scipy.linalg import eigh
import fea_fixtures as ff


def test_mac_of_vector_with_itself_is_one():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(10) + 1j * rng.standard_normal(10)
    mac = modal_assurance_criterion(v, v)
    print(f"MAC(v, v) = {mac:.12f}")
    assert abs(mac - 1.0) < 1e-12


def test_mac_of_orthogonal_vectors_is_zero():
    v1 = np.array([1.0, 0.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0, 0.0])
    mac = modal_assurance_criterion(v1, v2)
    print(f"MAC(orthogonal) = {mac:.3e}")
    assert mac < 1e-12


def test_mac_is_invariant_to_real_scale():
    rng = np.random.default_rng(1)
    v1 = rng.standard_normal(8)
    v2 = 3.7 * v1 + 0.0  # exact same direction, different scale
    mac = modal_assurance_criterion(v1, v2)
    print(f"MAC(v, 3.7*v) = {mac:.12f}")
    assert abs(mac - 1.0) < 1e-10


def test_mac_is_invariant_to_complex_phase():
    rng = np.random.default_rng(2)
    v1 = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    phase = np.exp(1j * 0.73)
    v2 = phase * v1
    mac = modal_assurance_criterion(v1, v2)
    print(f"MAC(v, e^(i*0.73)*v) = {mac:.12f}")
    assert abs(mac - 1.0) < 1e-10


def test_mac_on_real_cantilever_eigenmodes():
    fx = ff.cantilever_beam_system(n=16)
    free = fx["free_dofs"]
    K_ff = fx["K"][np.ix_(free, free)]
    M_ff = fx["M"][np.ix_(free, free)]
    _, eigvecs = eigh(K_ff, M_ff)

    n_modes = 5
    print(f"{'mode i':>6} {'mode j':>6} {'MAC':>8}")
    for i in range(n_modes):
        mac_self = modal_assurance_criterion(eigvecs[:, i], eigvecs[:, i])
        assert abs(mac_self - 1.0) < 1e-10, "a mode against itself must give MAC = 1"
        for j in range(i + 1, n_modes):
            mac = modal_assurance_criterion(eigvecs[:, i], eigvecs[:, j])
            print(f"{i:>6} {j:>6} {mac:>8.4f}")
            assert mac < 0.5, (
                "distinct eigenmodes of a generic (non-degenerate) beam should be "
                "well-separated in MAC, not near-collinear")


def test_mac_rejects_mismatched_shapes_and_zero_vector():
    with pytest.raises(ValueError):
        modal_assurance_criterion(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        modal_assurance_criterion(np.zeros(3), np.array([1.0, 0.0, 0.0]))


def test_r_squared_perfect_prediction_is_one():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(30)
    assert abs(r_squared(y, y) - 1.0) < 1e-12


def test_r_squared_constant_mean_prediction_is_zero():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(30)
    y_pred = np.full_like(y, y.mean())
    assert abs(r_squared(y, y_pred)) < 1e-12


def test_r_squared_can_go_negative_for_a_bad_fit():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(30)
    y_pred = -3.0 * y + rng.standard_normal(30) * 5.0   # deliberately anti-correlated + noisy
    r2 = r_squared(y, y_pred)
    print(f"deliberately bad fit R^2 = {r2:.4f}")
    assert r2 < 0


def test_r_squared_handles_multidimensional_input_via_flattening():
    rng = np.random.default_rng(3)
    y_true = rng.standard_normal((4, 10))
    y_pred = y_true + 0.01 * rng.standard_normal((4, 10))
    r2 = r_squared(y_true, y_pred)
    assert r2 > 0.99


def test_r_squared_rejects_mismatched_shapes_and_zero_variance_target():
    with pytest.raises(ValueError):
        r_squared(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        r_squared(np.full(5, 3.0), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
