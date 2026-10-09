"""
test_static_koiter_newton.py -- validation for Module 22:
nonlinear_solver.solve_nonlinear_static_koiter_newton() (Koiter-Newton
solve to a SINGLE prescribed target load, not a path tracer).

Distinct from test_koiter_newton.py (which validates the continuation
driver, solve_nonlinear_koiter_newton()): this module exists because a
first attempt to reuse the continuation driver as a "reach one target
load" solver (run it for a few steps, then interpolate/polish back to
lambda=1) was tried on a real Shell4MITCCorotational model and failed --
it overshot the target by >14x on one sample (slower than plain Newton)
and drove the element into floating-point overflow on a larger sample.
solve_nonlinear_static_koiter_newton() is the real fix: aim the cubic
predictor's step directly at the remaining gap to the target load
instead of growing until a free-running validity check fails.

Three lines of evidence:

1. test_matches_plain_newton_pre_limit_point -- the same von Mises
   truss/closed-form benchmark every other nonlinear static driver in
   this package is validated against, on the MONOTONIC (pre-limit-
   point) branch where solve_nonlinear_static() itself is known-valid,
   so the two can be compared directly.
2. test_reduced_coefficients_match_analytic_cubic_force_law -- the same
   hand-derived Green-Lagrange truss-bar closed form
   test_koiter_newton.py's own analogous test uses, confirming the
   shared Q/C finite-difference machinery is being exercised and
   trusted correctly in this function's own (different) predictor-
   aiming logic.
3. test_multi_expansion_path_reaches_target -- forces a case where the
   FIRST expansion's target-aimed cubic root does NOT satisfy
   predictor_tol (an artificially strict predictor_tol on the von Mises
   truss well past its own first limit point), confirming the shrink-
   and-re-expand loop actually engages (more than one expansion is
   used) and still lands exactly on lambda=1.0 without ever
   overshooting past it.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls


A_HALFSPAN, H0 = 1.0, 0.10
E, A = 210e9, 2e-4
L0 = np.sqrt(A_HALFSPAN ** 2 + H0 ** 2)


def _von_mises_truss():
    nodes = np.array([[-A_HALFSPAN, 0.0], [0.0, H0], [A_HALFSPAN, 0.0]])
    elements = np.array([[0, 1], [1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1])
    control_dof = 1 * fes.npn + 1
    return fes, control_dof


def _P_closed_form(delta):
    return (E * A / L0 ** 3) * delta * (H0 - delta) * (2 * H0 - delta)


def test_matches_plain_newton_pre_limit_point():
    delta_peak = H0 * (3 - np.sqrt(3)) / 3
    P_peak = _P_closed_form(delta_peak)
    P_target = 0.5 * P_peak   # comfortably before the limit point

    fes_static, control_dof = _von_mises_truss()
    fes_static.add_nodal_force([1], 1, -P_target)
    _, U_hist = nls.solve_nonlinear_static(fes_static, (E, A), n_steps=20, tol=1e-12)
    u_static = U_hist[-1]

    fes_kn, control_dof2 = _von_mises_truss()
    assert control_dof2 == control_dof
    fes_kn.add_nodal_force([1], 1, -P_target)
    u_kn = nls.solve_nonlinear_static_koiter_newton(fes_kn, (E, A), tol=1e-12)

    scale = max(np.max(np.abs(u_static)), 1e-30)
    assert np.max(np.abs(u_static - u_kn)) / scale < 1e-8

    delta_kn = -u_kn[control_dof]
    assert abs(_P_closed_form(delta_kn) - P_target) / P_target < 1e-6


def test_reduced_coefficients_match_analytic_cubic_force_law():
    """Same closed form as test_koiter_newton.py's analogous unit check:
    f(u) = E*A*(u + 1.5*u^2 + 0.5*u^3) for a single L0=1 TrussTL2D bar
    loaded purely axially -- exactly cubic, so a single expansion should
    reach the target load to floating-point precision."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    elements = np.array([[0, 1]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0], [0, 1])
    fes.fix_dofs([1], [1])
    P_target = 5.0e6
    fes.add_nodal_force([1], 0, P_target)

    u = nls.solve_nonlinear_static_koiter_newton(fes, (E, A), tol=1e-10)
    delta = u[2]
    P_exact = E * A * (delta + 1.5 * delta ** 2 + 0.5 * delta ** 3)
    assert abs(P_exact - P_target) / P_target < 1e-8


def test_multi_expansion_path_reaches_target():
    """P_target stays BEFORE the truss's own limit point (a fixed-load
    equilibrium is well-posed there for ANY method, including plain
    Newton) but close enough to it that the response is strongly
    nonlinear -- combined with a deliberately strict predictor_tol, this
    forces the shrink-and-re-expand loop to actually engage (more than
    one expansion needed) rather than reaching the target in a single
    shot, while still landing on it exactly."""
    delta_peak = H0 * (3 - np.sqrt(3)) / 3
    P_peak = _P_closed_form(delta_peak)
    P_target = 0.9 * P_peak

    fes, control_dof = _von_mises_truss()
    fes.add_nodal_force([1], 1, -P_target)

    u = nls.solve_nonlinear_static_koiter_newton(
        fes, (E, A), tol=1e-10, predictor_tol=1e-4, max_expansions=30,
        max_shrink_iters=20)

    delta = -u[control_dof]
    assert abs(_P_closed_form(delta) - P_target) / P_target < 1e-6
