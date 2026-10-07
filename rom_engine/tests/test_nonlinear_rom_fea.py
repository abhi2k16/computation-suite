"""
Real-fixture validation of nonlinear_rom.py against fea_engine's own
geometrically nonlinear Beam2DCorotational element -- the strongest
check available for this module: everything in test_nonlinear_rom.py
uses a synthetic ground-truth function (a known cubic/Nash-form
polynomial) or a hand-built decoupled-cubic-spring stand-in FOM, which
proves the REGRESSION math is right but not that the whole pipeline
(mass-normalized modal projection, applied-load training-data
generation, surrogate fit) survives contact with a real, coupled,
geometrically nonlinear structural model.

Uses tests/fea_fixtures.py's clamped_clamped_nonlinear_beam_system() --
the standard clamped-clamped benchmark geometry this module's reference
literature (He et al. 2023 and its ICE/STEP predecessors) targets, as
opposed to test_nonlinear_beam.py's own cantilever fixture (built to
validate the ELEMENT against the Euler elastica, a different purpose).
"""
import numpy as np
import pytest

from rom_engine.nonlinear_rom import (
    MultiFidelitySurrogate,
    PolynomialModalROM,
    AppliedLoadStrategy,
    _HAS_TORCH,
    NeuralSurrogate,
)
from rom_engine.metrics import r_squared
from fea_fixtures import clamped_clamped_nonlinear_beam_system


@pytest.fixture(scope="module")
def cc_beam_training_data():
    """Builds the clamped-clamped beam fixture once per test module
    (Newton-Raphson training solves are the expensive part of this
    test, ~40 nonlinear static solves) and generates ONE shared
    AppliedLoadStrategy training set that every test in this file
    reuses -- consistent with this test suite's existing convention of
    factoring expensive fea_engine setup out of individual test
    functions (see test_frequency.py's fixture usage)."""
    fx = clamped_clamped_nonlinear_beam_system(n_elem=10, n_modes=2)
    n_modes = fx["V"].shape[1]

    rng = np.random.default_rng(42)
    strategy = AppliedLoadStrategy(
        target_fracs=(0.3, 2.0),        # fraction of the reference scale (beam "thickness")
        reference_scale=0.01,            # a nominal thickness-like length scale, meters
        n_samples=40,
        rng=rng,
    )
    q_l, q_nl, F_nl = strategy.generate(
        fx["V"], fx["M_ff"], fx["freq_hz"], fx["mode_shape_peaks"], fx["fom_solver"])

    n_train = 30
    return {
        "fx": fx, "n_modes": n_modes,
        "q_l_train": q_l[:n_train], "q_nl_train": q_nl[:n_train], "F_nl_train": F_nl[:n_train],
        "q_l_test": q_l[n_train:], "q_nl_test": q_nl[n_train:], "F_nl_test": F_nl[n_train:],
    }


class TestAppliedLoadStrategyOnRealBeam:
    def test_generated_q_nl_is_nonzero_and_bounded(self, cc_beam_training_data):
        # a degenerate all-zero or blown-up training set would silently
        # make every downstream R^2 check meaningless
        q_nl = cc_beam_training_data["q_nl_train"]
        assert np.all(np.isfinite(q_nl))
        assert np.max(np.abs(q_nl)) > 0.0
        assert np.max(np.abs(q_nl)) < 1.0   # sanity bound -- not a wild extrapolation blow-up

    def test_generated_F_nl_is_nonzero(self, cc_beam_training_data):
        # a linear (or near-linear) response would make F_nl ~ 0 everywhere,
        # which would trivially pass any R^2 check without testing anything
        F_nl = cc_beam_training_data["F_nl_train"]
        assert np.max(np.abs(F_nl)) > 1e-6


class TestMultiFidelitySurrogateOnRealBeam:
    @pytest.mark.parametrize("kernel", ["multiquadric", "cubic", "thin_plate_spline"])
    def test_held_out_r_squared_is_high(self, cc_beam_training_data, kernel):
        d = cc_beam_training_data
        model = MultiFidelitySurrogate(kernel=kernel).fit(d["q_l_train"], d["F_nl_train"])
        pred = model.predict(d["q_l_test"])
        r2 = r_squared(d["F_nl_test"], pred)
        assert r2 > 0.95

    def test_jacobian_matches_finite_difference_at_a_training_point(self, cc_beam_training_data):
        d = cc_beam_training_data
        model = MultiFidelitySurrogate(kernel="cubic").fit(d["q_l_train"], d["F_nl_train"])
        q0 = d["q_l_train"][0]
        J = model.jacobian(q0)
        eps = 1e-8
        n = d["n_modes"]
        J_fd = np.zeros((n, n))
        for j in range(n):
            dq = np.zeros(n)
            dq[j] = eps
            J_fd[:, j] = (model.predict(q0 + dq) - model.predict(q0 - dq)) / (2 * eps)
        assert np.max(np.abs(J - J_fd)) < 1.0   # loose bound: real force scale is O(1-200) here


class TestPolynomialModalROMOnRealBeam:
    def test_held_out_r_squared_is_high(self, cc_beam_training_data):
        d = cc_beam_training_data
        model = PolynomialModalROM(d["n_modes"]).fit(d["q_nl_train"], d["F_nl_train"])
        pred = model.force(d["q_nl_test"])
        r2 = r_squared(d["F_nl_test"], pred)
        assert r2 > 0.99

    def test_newton_predict_recovers_a_training_equilibrium(self, cc_beam_training_data):
        # round-trip check: Lambda*q + theta(q) = F_ext at a KNOWN
        # training point should Newton-solve back to that same q
        d = cc_beam_training_data
        model = PolynomialModalROM(d["n_modes"]).fit(d["q_nl_train"], d["F_nl_train"])
        Lambda = (2 * np.pi * d["fx"]["freq_hz"]) ** 2
        q_true = d["q_nl_train"][0]
        F_ext = Lambda * q_true + model.force(q_true)
        q_sol, converged = model.predict(F_ext=F_ext, Lambda=Lambda)
        assert converged
        assert np.max(np.abs(q_sol - q_true)) < 1e-6


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
class TestNeuralSurrogateOnRealBeam:
    """Mirrors TestMultiFidelitySurrogateOnRealBeam -- same real
    clamped-clamped beam training/held-out split, swapping the RBF
    surrogate for the MLP-based NeuralSurrogate to confirm it is a
    genuine drop-in ReducedForceModel alternative, not just correct on
    the decoupled synthetic cases in test_nonlinear_rom.py. Torch-gated;
    needs real pass/fail validation on a machine with PyTorch installed
    (see NeuralSurrogate's own docstring and the sandbox note in
    docs/nonlinear_surrogate_rom_roadmap.md)."""

    def test_held_out_r_squared_is_high(self, cc_beam_training_data):
        d = cc_beam_training_data
        model = NeuralSurrogate(n_modes=d["n_modes"], hidden_sizes=(32, 32),
                                 n_epochs=3000, lr=1e-2, seed=0).fit(
            d["q_l_train"], d["F_nl_train"])
        pred = model.predict(d["q_l_test"])
        r2 = r_squared(d["F_nl_test"], pred)
        assert r2 > 0.9

    def test_jacobian_matches_finite_difference_at_a_training_point(self, cc_beam_training_data):
        d = cc_beam_training_data
        model = NeuralSurrogate(n_modes=d["n_modes"], hidden_sizes=(32, 32),
                                 n_epochs=3000, lr=1e-2, seed=1).fit(
            d["q_l_train"], d["F_nl_train"])
        q0 = d["q_l_train"][0]
        J = model.jacobian(q0)
        eps = 1e-6
        n = d["n_modes"]
        J_fd = np.zeros((n, n))
        for j in range(n):
            dq = np.zeros(n)
            dq[j] = eps
            J_fd[:, j] = (model.predict(q0 + dq) - model.predict(q0 - dq)) / (2 * eps)
        assert np.max(np.abs(J - J_fd)) < 1.0   # same loose bound as the MFS comparison above
