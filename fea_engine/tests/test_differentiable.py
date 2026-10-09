# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_differentiable.py -- Wave 10 (docs/consolidated_future_
roadmap.md), source Saverio et al. 2026: validates fea_engine.
differentiable's items 98-101 (generic trainable additive-correction
Newton driver, explicit residual-minimization calibration, the
whole-solver implicit-differentiation adjoint layer, and metric-
consistent gradient scaling), plus item 102 (batched vmap autograd
tangents, tested in test_autograd_tangent.py instead, next to the
per-element functions it batches).

Two-tier structure, matching this module's own docstring:
  - TestAdditiveCorrectionDriver / TestAdjointMath / TestMetricConsistent
    Scaling run UNCONDITIONALLY (no torch needed) -- they validate the
    actual mathematical/numerical content (does a Newton solve with a
    correction term converge to a self-consistent root; does the
    adjoint formula match a brute-force finite-difference reference;
    does the metric reparametrization identity hold) using a hand-
    coded pure-NumPy correction (`_NumpyToyCorrection` below) wherever
    a "correction" object is needed.
  - Every class below that constructs a REAL torch-backed correction
    (ScalarFieldCorrection/MLPCorrection/MetricScaledField) or calls a
    torch-optimizer-driven function (calibrate_correction_explicit,
    implicit_correction_solve) is torch-gated (`_HAS_TORCH`), following
    the same skip-cleanly convention test_autograd_tangent.py/test_
    torch_sparse_solver.py already established.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls

from fea_engine.differentiable import (
    _HAS_TORCH,
    AdditiveCorrection,
    corrected_residual,
    corrected_tangent,
    _newton_equilibrium,
    solve_nonlinear_static_corrected,
    adjoint_gradient,
    mass_cholesky_factor,
    metric_step_matches_euclidean_reparametrized_step,
)

if _HAS_TORCH:
    import torch
    from fea_engine.differentiable import (
        ScalarFieldCorrection, MLPCorrection, MetricScaledField,
        calibrate_correction_explicit, implicit_correction_solve,
    )


# =====================================================================
# Shared fixtures
# =====================================================================
def _single_bar_system(E=200e9, A=1e-4, L0=2.0, F=5.0e3):
    """The simplest possible nonlinear fea_engine model with exactly
    ONE free dof: a single TrussTL2D bar along the global x-axis, node
    0 fully fixed, node 1's y-dof ALSO fixed (a bar exactly along one
    axis has zero perpendicular stiffness at u=0 -- a classical
    mechanism -- so leaving that dof free would make the tangent
    singular; fixing it isolates the one genuinely load-bearing dof,
    exactly what every test below needs and nothing more). Returns
    (fesystem, mat, F_ext)."""
    nodes = np.array([[0.0, 0.0], [L0, 0.0]])
    elements = np.array([[0, 1]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0], [0, 1])
    fes.fix_dofs([1], [1])
    fes.add_nodal_force([1], 0, F)
    return fes, (E, A), fes.F.copy()


def _two_bar_system(a=1.0, h0=0.6, E=210e9, A=2e-4, F=2.0e3):
    """A two-bar truss (apex node free in BOTH x and y -- 2 free dofs),
    used for tests that need a genuinely coupled/non-diagonal
    correction Jacobian rather than the single-bar fixture's trivial
    1-DOF case. h0 chosen tall enough, and F small enough, that this
    stays well away from the von Mises snap-through limit point
    test_nonlinear.py's own two-bar fixture is built to explore --
    plain Newton (no line search) must converge outright here."""
    nodes = np.array([[-a, 0.0], [0.0, h0], [a, 0.0]])
    elements = np.array([[0, 1], [1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1])
    fes.add_nodal_force([1], 1, -F)
    return fes, (E, A), fes.F.copy()


class _NumpyToyCorrection(AdditiveCorrection):
    """Pure-NumPy, hand-differentiable trainable correction used ONLY
    by this file's own unconditional (no-torch) tests: f_theta(u) =
    theta * (A @ u) for a fixed matrix A (identity by default) and a
    scalar theta. Chosen because both jacobian() (theta*A, constant)
    and d f_theta/d theta (A@u, needed by TestAdjointMath's own hand-
    coded vjp_fn) have simple closed forms -- no torch.autograd needed
    to differentiate it, which is the whole point of keeping this
    class torch-free."""
    scope = "nonlocal"

    def __init__(self, theta, A=None, n=None):
        self.theta = theta
        self.A = A if A is not None else np.eye(n)

    def value(self, u_free):
        return self.theta * (self.A @ u_free)

    def jacobian(self, u_free):
        return self.theta * self.A

    def dvalue_dtheta(self, u_free):
        return self.A @ u_free


# =====================================================================
# Item 99: additive-correction Newton driver -- unconditional (no
# torch needed: _NumpyToyCorrection is pure NumPy).
# =====================================================================
class TestAdditiveCorrectionDriver:
    def test_correction_none_matches_plain_newton(self):
        fes1, mat, _ = _two_bar_system()
        fes2, _, _ = _two_bar_system()
        load_factors = np.linspace(0.0, 1.0, 6)

        lf1, U1 = nls.solve_nonlinear_static(fes1, mat, load_factors=load_factors,
                                              line_search=False)
        lf2, U2 = solve_nonlinear_static_corrected(fes2, mat, correction=None,
                                                    load_factors=load_factors)
        assert np.allclose(lf1, lf2)
        assert np.max(np.abs(U1 - U2)) < 1e-10

    def test_solve_with_correction_converges_to_self_consistent_root(self):
        fes, mat, F_ext = _two_bar_system()
        free = fes.free_dofs
        corr = _NumpyToyCorrection(theta=0.15, n=len(free))
        load_factors, U_hist = solve_nonlinear_static_corrected(
            fes, mat, correction=corr, load_factors=np.array([0.0, 1.0]))
        u_final = U_hist[-1]
        R_free, _ = corrected_residual(fes, u_final, mat, corr)
        assert np.max(np.abs(R_free)) < 1e-6

    def test_correction_changes_the_equilibrium_solution(self):
        fes1, mat, _ = _two_bar_system()
        fes2, _, _ = _two_bar_system()
        free = fes1.free_dofs
        # theta scaled to be a real (~14%) fraction of the bars' own
        # EA/L0 (~3.6e7) -- a plain theta=O(1) is utterly negligible
        # next to that stiffness and would leave the solution
        # unchanged to within Newton's own convergence tolerance,
        # which is not what this test is checking.
        corr = _NumpyToyCorrection(theta=5.0e6, n=len(free))

        _, U_plain = solve_nonlinear_static_corrected(fes1, mat, correction=None)
        _, U_corrected = solve_nonlinear_static_corrected(fes2, mat, correction=corr)
        assert np.max(np.abs(U_plain[-1] - U_corrected[-1])) > 1e-6

    def test_including_correction_jacobian_reduces_iteration_count(self):
        # Newton with the correction's own jacobian wired into the
        # tangent should need FEWER (or equal) iterations to reach the
        # same tolerance than a "lazy" version that always reports a
        # zero jacobian (still converges to the same root -- see
        # AdditiveCorrection's own docstring -- just no longer at
        # quadratic Newton speed).
        fes1, mat, F_ext = _two_bar_system()
        fes2, _, _ = _two_bar_system()
        free = fes1.free_dofs

        class _LazyJacobianCorrection(_NumpyToyCorrection):
            def jacobian(self, u_free):
                return np.zeros((len(u_free), len(u_free)))

        corr_full = _NumpyToyCorrection(theta=5.0e6, n=len(free))
        corr_lazy = _LazyJacobianCorrection(theta=5.0e6, n=len(free))

        *_, converged_full, _ = _newton_equilibrium(fes1, mat, corr_full, F_ext, tol=1e-10, max_iter=50)
        n_iter_full = _count_iterations(fes1, mat, corr_full, F_ext)
        n_iter_lazy = _count_iterations(fes2, mat, corr_lazy, F_ext)
        assert converged_full
        assert n_iter_full <= n_iter_lazy


def _count_iterations(fesystem, mat, correction, F_ext, tol=1e-12, max_iter=50):
    """Re-runs _newton_equilibrium()'s own loop logic just to count
    iterations to convergence (the function itself doesn't return a
    count) -- a small, self-contained re-instrumentation used only by
    the iteration-count comparison test above."""
    free = fesystem.free_dofs
    u = np.zeros(fesystem.n_dof)
    ref = max(np.linalg.norm(np.asarray(F_ext)[free]), 1e-30)

    def _residual(u_free_trial):
        u[free] = u_free_trial
        return corrected_residual(fesystem, u, mat, correction, F_ext=F_ext)

    u_free = u[free].copy()
    R_free, _ = _residual(u_free)
    Rn = np.linalg.norm(R_free)
    for it in range(max_iter):
        if Rn < tol * ref or Rn < tol:
            return it
        K_eff = corrected_tangent(fesystem, u, mat, correction)
        du = np.linalg.solve(K_eff, R_free)
        u_free = u_free + du
        R_free, _ = _residual(u_free)
        Rn = np.linalg.norm(R_free)
    return max_iter


# =====================================================================
# Item 98: adjoint math -- unconditional (no torch needed).
# =====================================================================
class TestAdjointMath:
    def test_adjoint_gradient_matches_finite_difference_single_dof(self):
        def solve_and_state(theta):
            fes, mat, F_ext = _single_bar_system()
            corr = _NumpyToyCorrection(theta=theta, n=len(fes.free_dofs))
            u_full, u_free, K_eff, F_int, converged, Rn = _newton_equilibrium(
                fes, mat, corr, F_ext, tol=1e-8, max_iter=30)
            assert converged
            return u_free, K_eff, corr

        # theta scaled relative to the bar's own EA/L0 (~1e7 here) --
        # O(1) would be negligible next to that stiffness and give a
        # finite-difference signal swamped by Newton's own tol=1e-8
        # convergence noise floor; ~1% of K is a real, well-resolved
        # perturbation.
        theta0 = 1.0e5
        u_free0, K_eff0, corr0 = solve_and_state(theta0)
        loss0 = 0.5 * np.sum(u_free0 ** 2)
        grad_w = u_free0.copy()   # dL/du_free for L = 0.5*sum(u_free**2)

        def vjp_fn(lam):
            return np.array([np.dot(lam, corr0.dvalue_dtheta(u_free0))])

        grad_theta_adjoint = adjoint_gradient(K_eff0, grad_w, vjp_fn)[0]

        eps = theta0 * 1e-3
        u_p, _, _ = solve_and_state(theta0 + eps)
        u_m, _, _ = solve_and_state(theta0 - eps)
        loss_p = 0.5 * np.sum(u_p ** 2)
        loss_m = 0.5 * np.sum(u_m ** 2)
        grad_theta_fd = (loss_p - loss_m) / (2 * eps)

        rel = abs(grad_theta_adjoint - grad_theta_fd) / max(abs(grad_theta_fd), 1e-30)
        assert rel < 1e-3

    def test_adjoint_gradient_matches_finite_difference_coupled_two_dof(self):
        # a genuinely NONLOCAL (off-diagonal) correction on a 2-free-
        # dof system -- exercises adjoint_gradient() on a real matrix
        # solve, not a 1x1 scalar division.
        rng = np.random.default_rng(3)
        A = rng.standard_normal((2, 2))

        def solve_and_state(theta):
            fes, mat, F_ext = _two_bar_system()
            corr = _NumpyToyCorrection(theta=theta, A=A)
            u_full, u_free, K_eff, F_int, converged, Rn = _newton_equilibrium(
                fes, mat, corr, F_ext, tol=1e-8, max_iter=30)
            assert converged
            return u_free, K_eff, corr

        theta0 = 1.0e5
        u_free0, K_eff0, corr0 = solve_and_state(theta0)
        w = np.array([1.3, -0.7])   # arbitrary fixed loss-gradient direction
        loss_grad_w = w

        def vjp_fn(lam):
            return np.array([np.dot(lam, corr0.dvalue_dtheta(u_free0))])

        grad_theta_adjoint = adjoint_gradient(K_eff0, loss_grad_w, vjp_fn)[0]

        eps = theta0 * 1e-3
        u_p, _, _ = solve_and_state(theta0 + eps)
        u_m, _, _ = solve_and_state(theta0 - eps)
        # L(theta) := w . u_free(theta)  =>  dL/dtheta via FD directly
        L_p = np.dot(w, u_p)
        L_m = np.dot(w, u_m)
        grad_theta_fd = (L_p - L_m) / (2 * eps)

        rel = abs(grad_theta_adjoint - grad_theta_fd) / max(abs(grad_theta_fd), 1e-30)
        assert rel < 1e-3


# =====================================================================
# Item 101: metric-consistent gradient scaling -- unconditional (no
# torch needed).
# =====================================================================
class TestMetricConsistentScaling:
    def test_reparametrized_step_matches_mass_preconditioned_step(self):
        rng = np.random.default_rng(0)
        n = 5
        B = rng.standard_normal((n, n))
        M = B @ B.T + n * np.eye(n)   # guaranteed SPD
        N = mass_cholesky_factor(M)
        assert np.allclose(N @ N.T, M)

        grad_theta = rng.standard_normal(n)
        d_reparam, d_metric = metric_step_matches_euclidean_reparametrized_step(
            N, grad_theta, lr=0.05)
        assert np.allclose(d_reparam, d_metric)

    def test_mass_cholesky_factor_rejects_non_spd(self):
        M_bad = np.array([[1.0, 2.0], [2.0, 1.0]])   # indefinite (eigenvalues -1, 3)
        with pytest.raises(np.linalg.LinAlgError):
            mass_cholesky_factor(M_bad)


# =====================================================================
# Torch-gated: concrete trainable corrections, explicit calibration
# (item 100), and the implicit adjoint layer's torch.autograd.Function
# wrapper (item 98).
# =====================================================================
@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
class TestScalarFieldCorrection:
    def test_value_and_jacobian_shapes(self):
        corr = ScalarFieldCorrection(n_free=3, init=0.5)
        u = np.zeros(3)
        v = corr.value(u)
        assert v.shape == (3,)
        assert np.allclose(v, 0.5)
        J = corr.jacobian(u)
        assert J.shape == (3, 3)
        assert np.allclose(J, 0.0)


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
class TestCalibrateCorrectionExplicit:
    def test_recovers_exact_closed_form_offset(self):
        # f_theta(u) = theta (CONSTANT w.r.t. u), so minimizing
        # ||target - theta||^2 has a known, closed-form global optimum
        # theta=target -- a decisive check, not just "loss went down".
        fes, mat, F_ext = _single_bar_system()
        free = fes.free_dofs
        w_ref = np.zeros(fes.n_dof)
        w_ref[free] = 1e-5
        F_int = fes.assemble_internal_force(w_ref, mat)
        target = F_ext[free] - F_int[free]

        corr = ScalarFieldCorrection(n_free=len(free), init=0.0)
        calibrate_correction_explicit(fes, mat, corr, w_ref, F_ext=F_ext,
                                       n_epochs=400, lr=max(1.0, abs(target[0]) * 0.1))
        assert np.allclose(corr.value(w_ref[free]), target, rtol=1e-2)


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
class TestMLPCorrection:
    def test_jacobian_matches_finite_difference_and_is_diagonal(self):
        corr = MLPCorrection(n_free=4, hidden_sizes=(8, 8), seed=0)
        rng = np.random.default_rng(1)
        u = rng.uniform(-1, 1, size=4)
        J = corr.jacobian(u)

        eps = 1e-6
        J_fd = np.zeros((4, 4))
        for j in range(4):
            du = np.zeros(4)
            du[j] = eps
            J_fd[:, j] = (corr.value(u + du) - corr.value(u - du)) / (2 * eps)
        assert np.max(np.abs(J - J_fd)) < 1e-5

        off_diag = J - np.diag(np.diag(J))
        assert np.allclose(off_diag, 0.0)


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
class TestMetricScaledFieldTorch:
    def test_theta_reconstructed_from_init(self):
        rng = np.random.default_rng(2)
        n = 3
        B = rng.standard_normal((n, n))
        M = B @ B.T + n * np.eye(n)
        field = MetricScaledField(n_field=n, M_sub=M, init=0.7)
        theta = field.value(None)
        assert np.allclose(theta, 0.7, atol=1e-8)

    def test_jacobian_is_zero(self):
        rng = np.random.default_rng(4)
        n = 2
        B = rng.standard_normal((n, n))
        M = B @ B.T + n * np.eye(n)
        field = MetricScaledField(n_field=n, M_sub=M, init=0.0)
        assert np.allclose(field.jacobian(None), 0.0)


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
class TestImplicitCorrectionSolve:
    def test_matches_plain_newton_when_theta_zero(self):
        fes, mat, F_ext = _single_bar_system()
        corr = ScalarFieldCorrection(n_free=len(fes.free_dofs), init=0.0)
        w_free_t = implicit_correction_solve(fes, mat, corr, F_ext=F_ext, tol=1e-10, max_iter=50)

        fes2, mat2, F_ext2 = _single_bar_system()
        u_full, u_free_np, K_eff, F_int, converged, Rn = _newton_equilibrium(
            fes2, mat2, None, F_ext2, tol=1e-10, max_iter=50)
        assert converged
        assert np.allclose(w_free_t.detach().numpy(), u_free_np, atol=1e-8)

    def test_gradient_matches_finite_difference(self):
        def solve(theta_val):
            fes, mat, F_ext = _single_bar_system()
            corr = ScalarFieldCorrection(n_free=len(fes.free_dofs), init=theta_val)
            w_t = implicit_correction_solve(fes, mat, corr, F_ext=F_ext, tol=1e-12, max_iter=50)
            loss = 0.5 * torch.sum(w_t ** 2)
            return loss, corr

        theta0 = 10.0
        loss0, corr0 = solve(theta0)
        loss0.backward()
        grad_analytic = float(corr0.theta.grad.detach().numpy().sum())

        eps = 1e-2
        loss_p, _ = solve(theta0 + eps)
        loss_m, _ = solve(theta0 - eps)
        grad_fd = (loss_p.item() - loss_m.item()) / (2 * eps)

        rel = abs(grad_analytic - grad_fd) / max(abs(grad_fd), 1e-30)
        assert rel < 1e-2
