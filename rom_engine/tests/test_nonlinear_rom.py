"""
Tests for nonlinear_rom.py -- KERNEL_REGISTRY, MultiFidelitySurrogate,
PolynomialModalROM, and the two TrainingStrategy variants.

Every model check below has two layers: (1) a SYNTHETIC ground-truth
function with a known closed form, so "did the fit recover the right
answer" and "is the analytic jacobian correct" are both checkable
without any FE package; (2) for the training strategies, a hand-built
one-DOF-per-mode nonlinear "full order model" (cubic springs) standing
in for a real FE model, exercising the exact `fom_solver` callable
contract `AppliedLoadStrategy`/`EnforcedDisplacementStrategy` expect
without importing fea_engine (this package's library code never does,
per rom_engine/__init__.py's stated design).

Real-fixture validation against an actual `fea_engine`
`Beam2DCorotational` clamped-clamped model lives in
tests/test_nonlinear_rom_fea.py (Step 2's later "real fea_engine
validation" task), not here.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from rom_engine.nonlinear_rom import (
    KERNEL_REGISTRY,
    MultiFidelitySurrogate,
    PolynomialModalROM,
    AppliedLoadStrategy,
    EnforcedDisplacementStrategy,
    ICEROM,
    ShiMeiROM,
    EnforcedDisplacementROM,
    _HAS_TORCH,
    _fit_normalization,
    _normalize,
    _denormalize,
    NeuralSurrogate,
)


def _finite_diff_jacobian(predict_fn, q0, r_out, eps=1e-6):
    r_in = len(q0)
    J = np.zeros((r_out, r_in))
    for j in range(r_in):
        dq = np.zeros(r_in)
        dq[j] = eps
        J[:, j] = (predict_fn(q0 + dq) - predict_fn(q0 - dq)) / (2 * eps)
    return J


# =====================================================================
# KERNEL_REGISTRY
# =====================================================================
class TestKernelRegistry:
    def test_all_four_paper_kernels_present(self):
        assert set(KERNEL_REGISTRY) == {"multiquadric", "cubic", "thin_plate_spline", "gaussian"}

    @pytest.mark.parametrize("kernel", list(KERNEL_REGISTRY))
    def test_dpsi_matches_finite_difference_of_psi(self, kernel):
        psi, dpsi = KERNEL_REGISTRY[kernel]
        sigma = 0.7
        R = np.array([0.05, 0.3, 1.0, 2.5])
        eps = 1e-6
        fd = (psi(R + eps, sigma) - psi(R - eps, sigma)) / (2 * eps)
        analytic = dpsi(R, sigma)
        assert np.max(np.abs(fd - analytic)) < 1e-4

    @pytest.mark.parametrize("kernel", list(KERNEL_REGISTRY))
    def test_dpsi_vanishes_at_zero_no_singularity(self, kernel):
        # every registered kernel's dpsi/dR -> 0 as R -> 0 (needed so
        # MultiFidelitySurrogate.jacobian()'s (q-center)/R direction
        # term is safe to zero out at R=0 without a 0/0 mismatch)
        _, dpsi = KERNEL_REGISTRY[kernel]
        assert dpsi(np.array([0.0]), 0.7)[0] == pytest.approx(0.0, abs=1e-9)

    def test_thin_plate_spline_psi_zero_at_origin(self):
        # R^2 log(R) -> 0 as R -> 0, not NaN (log(0) would be -inf * 0)
        psi, _ = KERNEL_REGISTRY["thin_plate_spline"]
        assert psi(np.array([0.0]), 1.0)[0] == 0.0


# =====================================================================
# MultiFidelitySurrogate
# =====================================================================
class TestMultiFidelitySurrogate:
    def _synthetic_cubic(self, rng, n=60, r=2):
        q_l = rng.uniform(-1, 1, size=(n, r))
        F_nl = 5.0 * q_l ** 3
        return q_l, F_nl

    @pytest.mark.parametrize("kernel", list(KERNEL_REGISTRY))
    def test_fit_predict_recovers_synthetic_cubic_force(self, kernel):
        rng = np.random.default_rng(0)
        q_l, F_nl = self._synthetic_cubic(rng)
        model = MultiFidelitySurrogate(kernel=kernel).fit(q_l, F_nl)
        pred = model.predict(q_l)
        ss_res = np.sum((F_nl - pred) ** 2)
        ss_tot = np.sum((F_nl - F_nl.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.999

    @pytest.mark.parametrize("kernel", list(KERNEL_REGISTRY))
    def test_jacobian_matches_finite_difference(self, kernel):
        rng = np.random.default_rng(1)
        q_l, F_nl = self._synthetic_cubic(rng)
        model = MultiFidelitySurrogate(kernel=kernel).fit(q_l, F_nl)
        q0 = np.array([0.3, -0.2])
        J = model.jacobian(q0)
        J_fd = _finite_diff_jacobian(model.predict, q0, r_out=2)
        assert np.max(np.abs(J - J_fd)) < 5e-3

    def test_predict_single_point_returns_1d(self):
        rng = np.random.default_rng(2)
        q_l, F_nl = self._synthetic_cubic(rng, n=30)
        model = MultiFidelitySurrogate(kernel="cubic").fit(q_l, F_nl)
        out = model.predict(q_l[0])
        assert out.shape == (2,)

    def test_predict_batch_returns_2d(self):
        rng = np.random.default_rng(3)
        q_l, F_nl = self._synthetic_cubic(rng, n=30)
        model = MultiFidelitySurrogate(kernel="cubic").fit(q_l, F_nl)
        out = model.predict(q_l)
        assert out.shape == q_l.shape

    def test_unknown_kernel_rejected(self):
        with pytest.raises(ValueError):
            MultiFidelitySurrogate(kernel="not_a_kernel")

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            MultiFidelitySurrogate().predict(np.array([0.0, 0.0]))

    def test_jacobian_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            MultiFidelitySurrogate().jacobian(np.array([0.0, 0.0]))

    def test_mismatched_sample_rows_rejected(self):
        rng = np.random.default_rng(4)
        q_l, F_nl = self._synthetic_cubic(rng, n=30)
        with pytest.raises(ValueError):
            MultiFidelitySurrogate().fit(q_l, F_nl[:-1])

    def test_reconstruct_q_nl_matches_eq11_rearrangement(self):
        # q_nl = q_l - F_nl(q_l)/Lambda (Eq. 11 rearranged); check it's
        # internally consistent with predict() rather than an
        # independent formula
        rng = np.random.default_rng(5)
        q_l, F_nl = self._synthetic_cubic(rng)
        model = MultiFidelitySurrogate(kernel="cubic").fit(q_l, F_nl)
        Lambda = np.array([10.0, 8.0])
        q0 = np.array([0.4, -0.1])
        q_nl = model.reconstruct_q_nl(q0, Lambda)
        expected = q0 - model.predict(q0) / Lambda
        assert np.allclose(q_nl, expected)

    def test_ridge_regularization_prevents_singular_solve(self):
        # duplicate training points would make the raw kernel matrix
        # singular without the ridge term
        q_l = np.array([[0.1, 0.1], [0.1, 0.1], [0.5, -0.3], [0.2, 0.4]])
        F_nl = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
        model = MultiFidelitySurrogate(kernel="multiquadric").fit(q_l, F_nl)
        assert model.weights is not None
        assert np.all(np.isfinite(model.weights))


# =====================================================================
# PolynomialModalROM
# =====================================================================
class TestPolynomialModalROM:
    def _synthetic_nash_form(self, rng, n=80, n_modes=2):
        q_nl = rng.uniform(-1, 1, size=(n, n_modes))
        theta1 = 2 * q_nl[:, 0] ** 2 + 3 * q_nl[:, 0] * q_nl[:, 1] + 4 * q_nl[:, 0] ** 3
        theta2 = -1 * q_nl[:, 1] ** 2 + 2 * q_nl[:, 0] ** 2 * q_nl[:, 1]
        F_nl = np.column_stack([theta1, theta2])
        return q_nl, F_nl

    def test_force_recovers_synthetic_nash_form_polynomial(self):
        rng = np.random.default_rng(10)
        q_nl, F_nl = self._synthetic_nash_form(rng)
        model = PolynomialModalROM(n_modes=2).fit(q_nl, F_nl)
        pred = model.force(q_nl)
        ss_res = np.sum((F_nl - pred) ** 2)
        ss_tot = np.sum((F_nl - F_nl.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.999999

    def test_jacobian_matches_finite_difference(self):
        rng = np.random.default_rng(11)
        q_nl, F_nl = self._synthetic_nash_form(rng)
        model = PolynomialModalROM(n_modes=2).fit(q_nl, F_nl)
        q0 = np.array([0.4, -0.3])
        J = model.jacobian(q0)
        J_fd = _finite_diff_jacobian(model.force, q0, r_out=2)
        assert np.max(np.abs(J - J_fd)) < 1e-4

    def test_predict_with_q_is_direct_evaluation(self):
        rng = np.random.default_rng(12)
        q_nl, F_nl = self._synthetic_nash_form(rng)
        model = PolynomialModalROM(n_modes=2).fit(q_nl, F_nl)
        assert np.allclose(model.predict(q=q_nl[0]), model.force(q_nl[0]))

    def test_predict_with_force_ext_solves_newton_and_recovers_q(self):
        rng = np.random.default_rng(13)
        q_nl, F_nl = self._synthetic_nash_form(rng)
        model = PolynomialModalROM(n_modes=2).fit(q_nl, F_nl)
        Lambda = np.array([10.0, 8.0])
        q_true = np.array([0.4, -0.3])
        F_ext = Lambda * q_true + model.force(q_true)
        q_sol, converged = model.predict(F_ext=F_ext, Lambda=Lambda)
        assert converged
        assert np.allclose(q_sol, q_true, atol=1e-8)

    def test_predict_rejects_both_q_and_force_ext(self):
        model = PolynomialModalROM(n_modes=2)
        with pytest.raises(ValueError):
            model.predict(q=np.zeros(2), F_ext=np.zeros(2), Lambda=np.ones(2))

    def test_predict_rejects_neither_q_nor_force_ext(self):
        model = PolynomialModalROM(n_modes=2)
        with pytest.raises(ValueError):
            model.predict()

    def test_predict_force_ext_requires_lambda(self):
        model = PolynomialModalROM(n_modes=2)
        with pytest.raises(ValueError):
            model.predict(F_ext=np.zeros(2))

    def test_force_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            PolynomialModalROM(n_modes=2).force(np.zeros(2))

    def test_wrong_number_of_modes_in_training_data_rejected(self):
        rng = np.random.default_rng(14)
        q_nl, F_nl = self._synthetic_nash_form(rng, n_modes=2)
        with pytest.raises(ValueError):
            PolynomialModalROM(n_modes=3).fit(q_nl, F_nl)


# =====================================================================
# TrainingStrategy variants -- validated against a hand-built
# synthetic nonlinear "full order model" (decoupled cubic springs per
# mode), NOT fea_engine (that's the separate real-fixture test file).
# =====================================================================
class _SyntheticCubicSpringFOM:
    """theta(u) = k3 * u^3 elementwise, decoupled per DOF -- a minimal
    stand-in nonlinear structural model with a known closed-form
    nonlinear restoring force, used only to exercise the
    `fom_solver` callable contract each TrainingStrategy expects."""

    def __init__(self, Lambda, k3):
        self.Lambda = Lambda
        self.k3 = k3

    def solve_applied_load(self, F_free):
        # Newton solve of Lambda*u + k3*u^3 = F_free, elementwise
        u = np.zeros_like(F_free)
        for _ in range(50):
            R = F_free - (self.Lambda * u + self.k3 * u ** 3)
            if np.max(np.abs(R)) < 1e-12:
                break
            Kt = self.Lambda + 3 * self.k3 * u ** 2
            u = u + R / Kt
        return u

    def solve_enforced_displacement(self, u_target):
        return self.Lambda * u_target + self.k3 * u_target ** 3


class TestAppliedLoadStrategy:
    def _setup(self):
        n_modes = 2
        Lambda = np.array([100.0, 64.0])
        k3 = np.array([5.0, 8.0])
        fom = _SyntheticCubicSpringFOM(Lambda, k3)
        M_ff = np.eye(n_modes)
        V = np.eye(n_modes)
        basis_freqs_hz = np.sqrt(Lambda) / (2 * np.pi)
        mode_shape_peaks = np.array([1.0, 1.0])
        return fom, M_ff, V, basis_freqs_hz, mode_shape_peaks, Lambda, k3

    def test_generate_returns_correctly_shaped_triple(self):
        fom, M_ff, V, freqs, peaks, Lambda, k3 = self._setup()
        strat = AppliedLoadStrategy(target_fracs=(0.2, 0.8), reference_scale=1.0,
                                     n_samples=25, rng=np.random.default_rng(0))
        q_l, q_nl, F_nl = strat.generate(V, M_ff, freqs, peaks, fom.solve_applied_load)
        assert q_l.shape == (25, 2)
        assert q_nl.shape == (25, 2)
        assert F_nl.shape == (25, 2)

    def test_downstream_multifidelity_surrogate_fits_generated_data(self):
        fom, M_ff, V, freqs, peaks, Lambda, k3 = self._setup()
        strat = AppliedLoadStrategy(target_fracs=(0.2, 0.8), reference_scale=1.0,
                                     n_samples=40, rng=np.random.default_rng(1))
        q_l, q_nl, F_nl = strat.generate(V, M_ff, freqs, peaks, fom.solve_applied_load)
        model = MultiFidelitySurrogate(kernel="cubic").fit(q_l, F_nl)
        pred = model.predict(q_l)
        r2 = 1 - np.sum((F_nl - pred) ** 2) / np.sum((F_nl - F_nl.mean(axis=0)) ** 2)
        assert r2 > 0.999

    def test_downstream_polynomial_rom_fits_generated_data(self):
        fom, M_ff, V, freqs, peaks, Lambda, k3 = self._setup()
        strat = AppliedLoadStrategy(target_fracs=(0.2, 0.8), reference_scale=1.0,
                                     n_samples=40, rng=np.random.default_rng(2))
        q_l, q_nl, F_nl = strat.generate(V, M_ff, freqs, peaks, fom.solve_applied_load)
        model = PolynomialModalROM(n_modes=2).fit(q_nl, F_nl)
        pred = model.force(q_nl)
        r2 = 1 - np.sum((F_nl - pred) ** 2) / np.sum((F_nl - F_nl.mean(axis=0)) ** 2)
        assert r2 > 0.999


class TestEnforcedDisplacementStrategy:
    def _setup(self):
        n_modes = 2
        Lambda = np.array([100.0, 64.0])
        k3 = np.array([5.0, 8.0])
        fom = _SyntheticCubicSpringFOM(Lambda, k3)
        M_ff = np.eye(n_modes)
        V = np.eye(n_modes)
        basis_freqs_hz = np.sqrt(Lambda) / (2 * np.pi)
        return fom, M_ff, V, basis_freqs_hz, Lambda, k3

    def test_generate_returns_correctly_shaped_triple(self):
        fom, M_ff, V, freqs, Lambda, k3 = self._setup()
        strat = EnforcedDisplacementStrategy(q_range=(-1.0, 1.0), n_samples=25,
                                              rng=np.random.default_rng(3))
        q_l, q_nl, F_nl = strat.generate(V, M_ff, freqs, fom.solve_enforced_displacement)
        assert q_l.shape == (25, 2)
        assert q_nl.shape == (25, 2)
        assert F_nl.shape == (25, 2)

    def test_prescribed_displacement_matches_q_nl_exactly(self):
        # this strategy prescribes q_nl directly -- unlike AppliedLoadStrategy,
        # there is no equilibrium-solving ambiguity in what q_nl ends up being
        fom, M_ff, V, freqs, Lambda, k3 = self._setup()
        strat = EnforcedDisplacementStrategy(q_range=(-1.0, 1.0), n_samples=20,
                                              rng=np.random.default_rng(4))
        q_l, q_nl, F_nl = strat.generate(V, M_ff, freqs, fom.solve_enforced_displacement)
        F_nl_expected = k3 * q_nl ** 3  # theta(u) = k3*u^3 is exactly the nonlinear part
        assert np.allclose(F_nl, F_nl_expected, atol=1e-10)

    def test_downstream_polynomial_rom_generalizes_to_held_out_points(self):
        fom, M_ff, V, freqs, Lambda, k3 = self._setup()
        strat = EnforcedDisplacementStrategy(q_range=(-1.0, 1.0), n_samples=40,
                                              rng=np.random.default_rng(5))
        q_l, q_nl, F_nl = strat.generate(V, M_ff, freqs, fom.solve_enforced_displacement)
        model = PolynomialModalROM(n_modes=2).fit(q_nl, F_nl)

        q_test = np.random.default_rng(99).uniform(-0.8, 0.8, size=(30, 2))
        F_test = k3 * q_test ** 3
        pred = model.force(q_test)
        r2 = 1 - np.sum((F_test - pred) ** 2) / np.sum((F_test - F_test.mean(axis=0)) ** 2)
        assert r2 > 0.999

    def test_scalar_q_range_broadcasts_to_every_mode(self):
        fom, M_ff, V, freqs, Lambda, k3 = self._setup()
        strat = EnforcedDisplacementStrategy(q_range=(-0.5, 0.5), n_samples=10,
                                              rng=np.random.default_rng(6))
        q_l, q_nl, F_nl = strat.generate(V, M_ff, freqs, fom.solve_enforced_displacement)
        assert np.all(q_nl >= -0.5 - 1e-9) and np.all(q_nl <= 0.5 + 1e-9)

    def test_bad_q_range_shape_rejected(self):
        strat = EnforcedDisplacementStrategy(q_range=np.zeros((5, 2)), n_samples=10,
                                              rng=np.random.default_rng(7))
        with pytest.raises(ValueError):
            strat.generate(np.eye(2), np.eye(2), np.array([1.0, 2.0]), lambda u: u)


# =====================================================================
# Convenience constructors
# =====================================================================
class TestConvenienceConstructors:
    def test_ice_rom_and_shi_mei_rom_are_polynomial_modal_rom(self):
        assert isinstance(ICEROM(2), PolynomialModalROM)
        assert isinstance(ShiMeiROM(2), PolynomialModalROM)

    def test_enforced_displacement_rom_is_polynomial_modal_rom(self):
        assert isinstance(EnforcedDisplacementROM(2), PolynomialModalROM)

    def test_ice_rom_fits_and_predicts_like_bare_polynomial_modal_rom(self):
        rng = np.random.default_rng(20)
        q_nl = rng.uniform(-1, 1, size=(50, 2))
        F_nl = 3.0 * q_nl ** 3
        model = ICEROM(2).fit(q_nl, F_nl)
        pred = model.force(q_nl)
        r2 = 1 - np.sum((F_nl - pred) ** 2) / np.sum((F_nl - F_nl.mean(axis=0)) ** 2)
        assert r2 > 0.999


# =====================================================================
# NeuralSurrogate normalization helpers -- these run UNCONDITIONALLY
# (no torch import needed), unlike every NeuralSurrogate-proper test
# below. Isolating the plain-NumPy normalization math this way means
# it gets checked in every environment, including this sandbox where
# torch itself is not installable (see NeuralSurrogate's own docstring
# for why the split exists).
# =====================================================================
class TestNormalizationHelpers:
    def test_normalize_then_denormalize_is_identity(self):
        rng = np.random.default_rng(30)
        X = rng.uniform(-5, 5, size=(40, 3))
        mean, std = _fit_normalization(X)
        rt = _denormalize(_normalize(X, mean, std), mean, std)
        assert np.allclose(rt, X, atol=1e-10)

    def test_normalized_data_has_zero_mean_unit_std(self):
        rng = np.random.default_rng(31)
        X = rng.uniform(-5, 5, size=(200, 2))
        mean, std = _fit_normalization(X)
        Xn = _normalize(X, mean, std)
        assert np.allclose(Xn.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(Xn.std(axis=0), 1.0, atol=1e-10)

    def test_constant_column_does_not_divide_by_zero(self):
        # a column with zero variance would give std=0 without the
        # eps-guard in _fit_normalization -- check it's clamped to 1.0
        # instead of producing inf/nan
        X = np.column_stack([np.full(10, 3.0), np.linspace(-1, 1, 10)])
        mean, std = _fit_normalization(X)
        assert std[0] == 1.0
        Xn = _normalize(X, mean, std)
        assert np.all(np.isfinite(Xn))
        assert np.allclose(Xn[:, 0], 0.0)


# =====================================================================
# NeuralSurrogate -- torch-gated. Cannot run in this sandbox (torch is
# not pip-installable here, per this project's established sandbox-
# then-user-machine convention -- see fea_engine's torch_sparse_solver
# and autograd_tangent modules for prior instances of the same split).
# These tests are written to the exact same conventions as
# TestMultiFidelitySurrogate above and need real pass/fail validation
# on a machine with PyTorch installed.
# =====================================================================
@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
class TestNeuralSurrogate:
    def _synthetic_cubic(self, rng, n=200, r=2):
        q_l = rng.uniform(-1, 1, size=(n, r))
        F_nl = 5.0 * q_l ** 3
        return q_l, F_nl

    def test_fit_predict_recovers_synthetic_cubic_force(self):
        rng = np.random.default_rng(0)
        q_l, F_nl = self._synthetic_cubic(rng)
        model = NeuralSurrogate(n_modes=2, hidden_sizes=(32, 32),
                                 n_epochs=1500, lr=1e-2, seed=0).fit(q_l, F_nl)
        pred = model.predict(q_l)
        ss_res = np.sum((F_nl - pred) ** 2)
        ss_tot = np.sum((F_nl - F_nl.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.99

    def test_jacobian_matches_finite_difference(self):
        rng = np.random.default_rng(1)
        q_l, F_nl = self._synthetic_cubic(rng)
        model = NeuralSurrogate(n_modes=2, hidden_sizes=(32, 32),
                                 n_epochs=1500, lr=1e-2, seed=1).fit(q_l, F_nl)
        q0 = np.array([0.3, -0.2])
        J = model.jacobian(q0)
        J_fd = _finite_diff_jacobian(model.predict, q0, r_out=2)
        assert np.max(np.abs(J - J_fd)) < 5e-2

    def test_predict_single_point_returns_1d(self):
        rng = np.random.default_rng(2)
        q_l, F_nl = self._synthetic_cubic(rng, n=60)
        model = NeuralSurrogate(n_modes=2, n_epochs=200, seed=2).fit(q_l, F_nl)
        out = model.predict(q_l[0])
        assert out.shape == (2,)

    def test_predict_batch_returns_2d(self):
        rng = np.random.default_rng(3)
        q_l, F_nl = self._synthetic_cubic(rng, n=60)
        model = NeuralSurrogate(n_modes=2, n_epochs=200, seed=3).fit(q_l, F_nl)
        out = model.predict(q_l)
        assert out.shape == q_l.shape

    def test_unknown_activation_rejected(self):
        with pytest.raises(ValueError):
            NeuralSurrogate(n_modes=2, activation="not_an_activation")

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            NeuralSurrogate(n_modes=2).predict(np.array([0.0, 0.0]))

    def test_jacobian_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            NeuralSurrogate(n_modes=2).jacobian(np.array([0.0, 0.0]))

    def test_jacobian_rejects_batch_input(self):
        rng = np.random.default_rng(4)
        q_l, F_nl = self._synthetic_cubic(rng, n=60)
        model = NeuralSurrogate(n_modes=2, n_epochs=200, seed=4).fit(q_l, F_nl)
        with pytest.raises(ValueError):
            model.jacobian(q_l)

    def test_mismatched_sample_rows_rejected(self):
        rng = np.random.default_rng(5)
        q_l, F_nl = self._synthetic_cubic(rng, n=30)
        with pytest.raises(ValueError):
            NeuralSurrogate(n_modes=2).fit(q_l, F_nl[:-1])

    def test_wrong_number_of_modes_in_training_data_rejected(self):
        rng = np.random.default_rng(6)
        q_l, F_nl = self._synthetic_cubic(rng, n=30, r=2)
        with pytest.raises(ValueError):
            NeuralSurrogate(n_modes=3).fit(q_l, F_nl)

    def test_seed_gives_reproducible_fit(self):
        rng = np.random.default_rng(7)
        q_l, F_nl = self._synthetic_cubic(rng, n=40)
        m1 = NeuralSurrogate(n_modes=2, n_epochs=100, seed=42).fit(q_l, F_nl)
        m2 = NeuralSurrogate(n_modes=2, n_epochs=100, seed=42).fit(q_l, F_nl)
        assert np.allclose(m1.predict(q_l), m2.predict(q_l))

    def test_downstream_applied_load_strategy_data_fits_neural_surrogate(self):
        # same synthetic cubic-spring FOM used for MultiFidelitySurrogate
        # above, now fed into NeuralSurrogate instead -- confirms
        # NeuralSurrogate is a drop-in ReducedForceModel alternative to
        # MultiFidelitySurrogate for TrainingStrategy-generated data
        n_modes = 2
        Lambda = np.array([100.0, 64.0])
        k3 = np.array([5.0, 8.0])
        fom = _SyntheticCubicSpringFOM(Lambda, k3)
        M_ff = np.eye(n_modes)
        V = np.eye(n_modes)
        freqs = np.sqrt(Lambda) / (2 * np.pi)
        peaks = np.array([1.0, 1.0])
        strat = AppliedLoadStrategy(target_fracs=(0.2, 0.8), reference_scale=1.0,
                                     n_samples=60, rng=np.random.default_rng(8))
        q_l, q_nl, F_nl = strat.generate(V, M_ff, freqs, peaks, fom.solve_applied_load)
        model = NeuralSurrogate(n_modes=2, n_epochs=1500, lr=1e-2, seed=8).fit(q_l, F_nl)
        pred = model.predict(q_l)
        r2 = 1 - np.sum((F_nl - pred) ** 2) / np.sum((F_nl - F_nl.mean(axis=0)) ** 2)
        assert r2 > 0.99
