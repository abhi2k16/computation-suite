"""
Tests for differentiable_correction.py (Wave 10 item 103). See
docs/differentiable_rom_correction_design.md for the design note this
prototype implements. reduced_residual() runs UNCONDITIONALLY (pure
NumPy); calibrate_reduced_correction_explicit()/ScalarModalCorrection
are torch-gated.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from rom_engine.differentiable_correction import (
    _HAS_TORCH, reduced_residual,
)

if _HAS_TORCH:
    from rom_engine.differentiable_correction import (
        ScalarModalCorrection, calibrate_reduced_correction_explicit,
    )


class _NumpyToyCorrection:
    """Pure-NumPy correction stand-in for the unconditional test below
    -- satisfies reduced_residual()'s own `.value(q)` protocol without
    needing torch at all."""

    def __init__(self, offset):
        self.offset = np.asarray(offset, dtype=float)

    def value(self, q):
        return self.offset


class TestReducedResidual:
    def test_zero_at_true_equilibrium_no_correction(self):
        Lambda = np.array([100.0, 64.0])

        def F_nl_fn(q):
            return 5.0 * q ** 3

        q_true = np.array([0.3, -0.2])
        F_ext = Lambda * q_true + F_nl_fn(q_true)
        R = reduced_residual(q_true, Lambda, F_nl_fn, F_ext)
        assert np.allclose(R, 0.0, atol=1e-12)

    def test_correction_shifts_the_residual_by_exactly_its_value(self):
        Lambda = np.array([10.0, 8.0])

        def F_nl_fn(q):
            return q ** 2

        q = np.array([0.1, 0.2])
        F_ext = np.array([1.0, 1.0])
        corr = _NumpyToyCorrection(offset=[0.05, -0.02])

        R_plain = reduced_residual(q, Lambda, F_nl_fn, F_ext, correction=None)
        R_corrected = reduced_residual(q, Lambda, F_nl_fn, F_ext, correction=corr)
        assert np.allclose(R_plain - R_corrected, corr.offset)


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
class TestCalibrateReducedCorrectionExplicit:
    def test_recovers_exact_closed_form_target(self):
        # f_theta(q) = theta is CONSTANT w.r.t. q, so minimizing
        # ||target - theta||^2 has a known, closed-form global optimum
        # theta = target -- decisive, not just "loss went down" (the
        # same style of check fea_engine's own item-100 test uses).
        Lambda = np.array([10.0, 8.0])

        def F_nl_fn(q):
            return q ** 3

        q_reference = np.array([0.2, -0.1])
        F_nl = F_nl_fn(q_reference)
        F_ext = np.array([5.0, -3.0])
        target = F_ext - Lambda * q_reference - F_nl

        corr = ScalarModalCorrection(n_modes=2, init=0.0)
        calibrate_reduced_correction_explicit(
            q_reference, Lambda, F_nl_fn, F_ext, corr, n_epochs=400, lr=0.5)
        assert np.allclose(corr.value(q_reference), target, rtol=1e-2, atol=1e-3)

    def test_loss_history_decreases(self):
        Lambda = np.array([10.0])

        def F_nl_fn(q):
            return q ** 3

        q_reference = np.array([0.1])
        F_ext = np.array([2.0])
        corr = ScalarModalCorrection(n_modes=1, init=0.0)
        history = calibrate_reduced_correction_explicit(
            q_reference, Lambda, F_nl_fn, F_ext, corr, n_epochs=200, lr=0.3)
        assert history[-1] < history[0]
