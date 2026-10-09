# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_wave1_newton_robustness.py -- Wave 1 (docs/consolidated_future_
roadmap.md, source nonlinear_fem_lessons.md's appendix), items 9/10:
displacement-increment/energy-error convergence criteria and Armijo
line search, added to solve_nonlinear_static(), solve_nonlinear_
displacement_control(), solve_nonlinear_arc_length(), and solve_
nonlinear_static_koiter_newton()'s corrector (see nonlinear_solver.py's
own module docstring "Wave 1" section for the full design and the two
shared helpers, _extra_convergence_ok()/_armijo_line_search_step(),
every one of those drivers now calls).

Two things are checked here, split deliberately:

1. CHECK 1/2 unit-test the two shared helpers DIRECTLY against hand-
   computed, independently-verifiable numbers -- this is the actual
   new algorithmic core Wave 1 introduces, and is fully controllable
   (unlike a real FE model's convergence behavior).

2. CHECK 3+ are integration tests on real drivers, confirming (a) the
   new du_tol/energy_tol/line_search parameters, passed with their
   DEFAULT values (None/None/True-but-never-triggered), reproduce
   EXACTLY the same converged answer every existing test already
   confirms (this file adds no NEW default-path regression coverage
   beyond what already re-passed unmodified -- see the Wave 1
   implementation notes -- so it isn't repeated here), and (b)
   explicitly setting du_tol/energy_tol to a loose-enough value still
   reaches the SAME, closed-form-verified answer.

HONESTY NOTE: a natural, real-FE-model reproduction of "plain Newton
fails within max_iter, line search rescues it" (the same kind of
evidence solve_nonlinear_transient()'s own docstring documents for its
own line-search fallback) was tried here directly, on the closed-form
von Mises truss (both load- and displacement-control) -- not found:
this benchmark's low DOF count and smooth cubic-in-displacement force
law make plain Newton converge (or fail) essentially identically to
line search at every max_iter/load-level combination tried, so there
is no natural case in this small/fast test suite where the fallback
actually changes the outcome. This is reported honestly rather than
manufactured; CHECK 2 instead verifies the fallback mechanism itself,
directly and deterministically, against a synthetic residual (tanh)
chosen specifically because full Newton is KNOWN to overshoot on it.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls
from fea_engine.nonlinear_solver import _extra_convergence_ok, _armijo_line_search_step


def test_extra_convergence_ok_off_by_default():
    print("=" * 70)
    print("CHECK 1: _extra_convergence_ok() -- both criteria are trivially")
    print("satisfied (True, True) when du_tol/energy_tol are both None")
    print("(the default, 'this criterion is off' state)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    R, du, u, F_ext, F_int = (rng.standard_normal(5) for _ in range(5))
    du_ok, energy_ok = _extra_convergence_ok(R, du, u, F_ext, F_int, None, None)
    assert du_ok is True and energy_ok is True
    print("  PASS")

    print()
    print("  -- and correctly discriminate large vs. small du_tol on a")
    print("  hand-built case --")
    u = np.array([1.0, 0.0])
    du_small = np.array([1e-9, 0.0])      # ||du||/||u|| = 1e-9
    du_large = np.array([0.5, 0.0])        # ||du||/||u|| = 0.5
    du_ok_small, _ = _extra_convergence_ok(R, du_small, u, F_ext, F_int, 1e-6, None)
    du_ok_large, _ = _extra_convergence_ok(R, du_large, u, F_ext, F_int, 1e-6, None)
    assert du_ok_small and not du_ok_large
    print("  PASS")

    print()
    print("  -- and correctly discriminate large vs. small energy_tol --")
    R2 = np.array([1.0, 0.0])
    du_tiny_energy = np.array([1e-9, 0.0])   # |du.R| = 1e-9, tiny
    du_big_energy = np.array([10.0, 0.0])    # |du.R| = 10, large
    F_ext2, F_int2 = np.array([1.0, 0.0]), np.array([0.5, 0.0])
    u2 = np.array([1.0, 0.0])   # W_ext=|1|, W_int=|0.5| -> ref_w=1.0
    _, e_ok_tiny = _extra_convergence_ok(R2, du_tiny_energy, u2, F_ext2, F_int2, None, 1e-6)
    _, e_ok_big = _extra_convergence_ok(R2, du_big_energy, u2, F_ext2, F_int2, None, 1e-6)
    assert e_ok_tiny and not e_ok_big
    print("  PASS")


def test_armijo_line_search_step_rescues_a_known_newton_overshoot():
    print()
    print("=" * 70)
    print("CHECK 2: _armijo_line_search_step() actually backtracks and")
    print("accepts alpha<1 on a residual where the FULL Newton step is")
    print("KNOWN (by hand-computation) to make ||R|| WORSE")
    print("=" * 70)
    # R(x) = tanh(x), root at x=0. At x0=1.2, Newton's linear model
    # badly over-extrapolates (tanh flattens out): R'(x)=1-tanh(x)^2.
    x0 = 1.2
    R0 = np.tanh(x0)
    Rp0 = 1.0 - np.tanh(x0) ** 2
    du_newton = np.array([-R0 / Rp0])   # full Newton step
    u_free = np.array([x0])
    Rn0 = abs(R0)

    # Hand-verify the overshoot BEFORE calling the function under test,
    # so this test would fail loudly (not silently) if numpy/tanh
    # semantics ever changed underneath it.
    x_full = x0 + du_newton[0]
    Rn_full = abs(np.tanh(x_full))
    print(f"  x0={x0}  R(x0)={R0:.6f}  full Newton step -> x={x_full:.6f}, "
          f"R(x_full)={np.tanh(x_full):.6f}")
    assert Rn_full > Rn0, "test setup error: expected the full step to overshoot"

    def residual_fn(x_trial):
        return np.array([np.tanh(x_trial[0])]), None

    u_new, R_new, _aux, Rn_new, alpha = _armijo_line_search_step(
        residual_fn, u_free, du_newton, Rn0, tol=1e-10, ref=0.0)

    print(f"  _armijo_line_search_step: alpha_used={alpha:.4f}  "
          f"|R_new|={Rn_new:.6f}  (|R_full| would have been {Rn_full:.6f})")
    assert alpha < 1.0, "expected backtracking to reject the full (overshooting) step"
    assert Rn_new < Rn0, "expected a genuine residual decrease after backtracking"
    # independently recompute at the returned alpha, confirm self-consistency
    x_check = x0 + alpha * du_newton[0]
    assert abs(u_new[0] - x_check) < 1e-12
    assert abs(Rn_new - abs(np.tanh(x_check))) < 1e-12
    print("  PASS -- rejected the overshooting full step and accepted a"
          " smaller, genuinely-decreasing one, self-consistently")


def _von_mises_truss():
    a, h0 = 1.0, 0.10
    E, A = 210e9, 2e-4
    L0 = np.sqrt(a ** 2 + h0 ** 2)
    nodes = np.array([[-a, 0.0], [0.0, h0], [a, 0.0]])
    elements = np.array([[0, 1], [1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)

    def P_closed_form(delta):
        return (E * A / L0 ** 3) * delta * (h0 - delta) * (2 * h0 - delta)

    delta_peak = h0 * (3 - np.sqrt(3)) / 3
    return mesh, E, A, P_closed_form, delta_peak


def test_static_du_tol_energy_tol_still_match_closed_form():
    print()
    print("=" * 70)
    print("CHECK 3: solve_nonlinear_static() with du_tol/energy_tol SET")
    print("(loose enough not to block real convergence) still reaches the")
    print("SAME closed-form-verified answer as the default (both None)")
    print("=" * 70)
    mesh, E, A, P_closed_form, delta_peak = _von_mises_truss()
    P_apply = 0.5 * P_closed_form(delta_peak)

    from scipy.optimize import brentq
    delta_cf = brentq(lambda d: P_closed_form(d) - P_apply, 1e-9, delta_peak)

    def _run(**extra):
        fes = FESystem(mesh, elmod.TrussTL2D())
        fes.fix_dofs([0, 2], [0, 1])
        fes.add_nodal_force([1], 1, -P_apply)
        control_dof = 1 * fes.npn + 1
        load_factors, U_hist = nls.solve_nonlinear_static(
            fes, (E, A), n_steps=20, tol=1e-12, **extra)
        return -U_hist[-1, control_dof]

    delta_default = _run()
    delta_extra = _run(du_tol=1e-4, energy_tol=1e-4)

    rel_default = abs(delta_default - delta_cf) / abs(delta_cf)
    rel_extra = abs(delta_extra - delta_cf) / abs(delta_cf)
    print(f"  delta_closed_form   = {delta_cf:.8f}")
    print(f"  delta (defaults)    = {delta_default:.8f}  rel_err={rel_default:.2e}")
    print(f"  delta (du/energy_tol=1e-4) = {delta_extra:.8f}  rel_err={rel_extra:.2e}")
    assert rel_default < 1e-6
    assert rel_extra < 1e-6
    print("  PASS")


def test_static_line_search_false_vs_true_agree_on_a_normal_case():
    print()
    print("=" * 70)
    print("CHECK 4: line_search=False and line_search=True (default) give")
    print("IDENTICAL results whenever plain Newton already converges fine")
    print("(the common case, per solve_nonlinear_transient's own")
    print("'fallback, not default path' design this reuses) -- confirms")
    print("the new fallback is truly inert unless actually needed")
    print("=" * 70)
    mesh, E, A, P_closed_form, delta_peak = _von_mises_truss()
    P_apply = 0.5 * P_closed_form(delta_peak)

    def _run(line_search):
        fes = FESystem(mesh, elmod.TrussTL2D())
        fes.fix_dofs([0, 2], [0, 1])
        fes.add_nodal_force([1], 1, -P_apply)
        control_dof = 1 * fes.npn + 1
        _, U_hist = nls.solve_nonlinear_static(
            fes, (E, A), n_steps=20, tol=1e-12, line_search=line_search)
        return U_hist[-1, control_dof]

    u_false = _run(False)
    u_true = _run(True)
    print(f"  line_search=False: u={u_false:.10e}")
    print(f"  line_search=True:  u={u_true:.10e}")
    assert u_false == pytest.approx(u_true, abs=1e-14)
    print("  PASS")


def test_static_koiter_newton_du_tol_energy_tol_still_matches_target():
    print()
    print("=" * 70)
    print("CHECK 5: solve_nonlinear_static_koiter_newton()'s corrector,")
    print("with du_tol/energy_tol set, still reaches the target load with")
    print("a physically sensible (closed-form-consistent) displacement")
    print("=" * 70)
    mesh, E, A, P_closed_form, delta_peak = _von_mises_truss()
    P_apply = 0.5 * P_closed_form(delta_peak)

    from scipy.optimize import brentq
    delta_cf = brentq(lambda d: P_closed_form(d) - P_apply, 1e-9, delta_peak)

    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1])
    fes.add_nodal_force([1], 1, -P_apply)
    control_dof = 1 * fes.npn + 1

    u_final = nls.solve_nonlinear_static_koiter_newton(
        fes, (E, A), tol=1e-10, du_tol=1e-4, energy_tol=1e-4)
    delta_fe = -u_final[control_dof]
    rel = abs(delta_fe - delta_cf) / abs(delta_cf)
    print(f"  delta_closed_form = {delta_cf:.8f}  delta_FE = {delta_fe:.8f}  rel_err={rel:.2e}")
    assert rel < 1e-5
    print("  PASS")
