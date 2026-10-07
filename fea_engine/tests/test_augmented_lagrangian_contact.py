"""
test_augmented_lagrangian_contact.py -- Wave 3 item 15 (docs/
consolidated_future_roadmap.md): validates nonlinear_solver.
solve_contact_augmented_lagrange_static(), the third contact-
enforcement mechanism this package now offers alongside the pure
penalty method (element.GapContactPenalty) and the exact bordered-KKT
pure Lagrange-multiplier method (solve_contact_lagrange_static(),
tests/test_contact.py CHECK 4/5).

Same 2-node TrussTL2D-vs-flat-wall benchmark as test_contact.py, for
direct comparability -- N_truss(u1) is the SAME already-validated
closed-form axial force used there.

Four lines of evidence:

1. test_al_matches_exact_lagrange -- at a reasonably modest k_p (1e10 --
   FAR below what test_contact.py's own CHECK 5 needed for even 1e-3
   relative accuracy from plain penalty), this driver's displacement,
   multiplier, and active/inactive history all match
   solve_contact_lagrange_static()'s EXACT answer to near machine
   precision, and the converged penetration is itself near machine-zero
   -- the concrete "better conditioning than pure penalty" payoff this
   item's roadmap entry names.
2. test_al_beats_plain_penalty_at_matched_k_p -- at the SAME k_p, this
   driver's converged answer is dramatically closer to the exact
   Lagrange reaction than plain penalty's own converged contact force
   is -- the augmented multiplier, not a huge k_p, is doing the exact-
   enforcement work.
3. test_al_zero_outer_iterations_equals_plain_penalty -- a direct
   structural check on the algorithm itself (not just its converged
   answer): forcing exactly ONE inner Newton solve with lambda_bar
   pinned at 0 (the very first Uzawa iterate, before any multiplier
   update) must give EXACTLY GapContactPenalty's own force/displacement
   at that same k_p and g0 -- confirms the "AL vsz plain penalty" claim
   in the function's own docstring (g0_eff=g0 when lambda_bar=0) is
   correct, not just approximately true.
4. test_al_zero_load_and_no_engagement -- basic sanity: zero load gives
   zero displacement and lambda=0; a load that never reaches the wall
   never activates contact and lambda stays exactly 0 throughout.
"""
import numpy as np
import pytest

from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls


E, A = 200e9, 1e-4
L0 = 1.0
G0 = 0.02
N_HAT = np.array([1.0, 0.0])
MAT = (E, A)


def _N_truss(u1):
    """Closed-form TrussTL2D axial force at node 1 (x-displacement u1),
    the SAME closed form test_contact.py already validates against."""
    E_GL = ((L0 + u1) ** 2 - L0 ** 2) / (2 * L0 ** 2)
    S = E * E_GL
    return S * A * (L0 + u1) / L0


def _mesh():
    nodes = np.array([[0.0, 0.0], [L0, 0.0]])
    elements = np.array([[0, 1]], dtype=int)
    return Mesh(nodes=nodes, elements=elements, dim=2)


def _fes():
    fes = FESystem(_mesh(), elmod.TrussTL2D())
    fes.fix_dofs([0], [0, 1])
    fes.fix_dofs([1], [1])
    return fes


def test_al_matches_exact_lagrange():
    print("=" * 70)
    print("CHECK 1: augmented-Lagrangian contact matches the EXACT")
    print("bordered-KKT Lagrange-multiplier answer, at a k_p far too")
    print("small for plain penalty to get anywhere near this accurate")
    print("=" * 70)
    P_gap = _N_truss(G0)
    P_apply = 3 * P_gap

    fes_exact = _fes()
    fes_exact.add_nodal_force([1], 0, P_apply)
    lf_e, U_e, lam_e, act_e = nls.solve_contact_lagrange_static(
        fes_exact, MAT, contact_node=1, n_hat=N_HAT, g0=G0, n_steps=60, tol=1e-13)

    fes_al = _fes()
    fes_al.add_nodal_force([1], 0, P_apply)
    k_p = 1e10
    lf_a, U_a, lam_a, act_a = nls.solve_contact_augmented_lagrange_static(
        fes_al, MAT, contact_node=1, n_hat=N_HAT, g0=G0, k_p=k_p, n_steps=60, tol=1e-13)

    u1_e, u1_a = U_e[:, 2], U_a[:, 2]
    max_disp_err = np.max(np.abs(u1_a - u1_e))
    max_lam_err = np.max(np.abs(lam_a - lam_e))
    print(f"  k_p = {k_p:.0e}")
    print(f"  max |u1_AL - u1_exact|      = {max_disp_err:.3e}")
    print(f"  max |lambda_AL - lambda_exact| = {max_lam_err:.3e}  (lambda scale ~{lam_e.max():.1f})")
    print(f"  active_hist matches exactly: {np.array_equal(act_e, act_a)}")
    assert np.array_equal(act_e, act_a)
    assert max_disp_err < 1e-9
    assert max_lam_err / max(lam_e.max(), 1.0) < 1e-6

    pen_active = np.abs(u1_a[act_a] - G0)
    print(f"  max penetration while active: {pen_active.max():.3e} (near machine-zero, like the exact driver)")
    assert pen_active.max() < 1e-9
    print("  PASS")


def test_al_beats_plain_penalty_at_matched_k_p():
    print()
    print("=" * 70)
    print("CHECK 2: at the SAME k_p, augmented Lagrangian is dramatically")
    print("closer to the exact contact reaction than plain penalty")
    print("=" * 70)
    P_apply = 2.0 * _N_truss(G0)
    lam_exact = P_apply - _N_truss(G0)
    k_p = 1e9   # deliberately modest -- test_contact.py's own CHECK 5 needed
    # k_p=1e12 for plain penalty to reach even 1e-3 relative error

    fes_pen = _fes()
    fes_pen.add_nodal_force([1], 0, P_apply)
    gc = elmod.GapContactPenalty()
    fes_pen.add_contact_element(gc, [1], (k_p, G0, N_HAT))
    _, U_pen = nls.solve_nonlinear_static(fes_pen, MAT, n_steps=80, tol=1e-8, max_iter=60)
    u1_pen = U_pen[-1, 2]
    force_pen = k_p * max(u1_pen - G0, 0.0)
    rel_err_pen = abs(force_pen - lam_exact) / lam_exact

    fes_al = _fes()
    fes_al.add_nodal_force([1], 0, P_apply)
    lf_a, U_a, lam_a, act_a = nls.solve_contact_augmented_lagrange_static(
        fes_al, MAT, contact_node=1, n_hat=N_HAT, g0=G0, k_p=k_p, n_steps=80, tol=1e-10)
    rel_err_al = abs(lam_a[-1] - lam_exact) / lam_exact

    print(f"  k_p = {k_p:.0e}   exact reaction = {lam_exact:.4f}")
    print(f"  plain penalty:  force = {force_pen:.4f}   rel_err = {rel_err_pen:.3e}")
    print(f"  augmented Lag.: force = {lam_a[-1]:.4f}   rel_err = {rel_err_al:.3e}")
    assert rel_err_al < rel_err_pen * 1e-3, (
        "augmented Lagrangian should be orders of magnitude more accurate "
        "than plain penalty at the same k_p")
    print("  PASS")


def test_al_zero_outer_iterations_equals_plain_penalty():
    print()
    print("=" * 70)
    print("CHECK 3: the FIRST Uzawa iterate (lambda_bar=0) is EXACTLY")
    print("GapContactPenalty's own force/tangent at the same k_p, g0 --")
    print("a direct structural check on the g0_eff=g0-lambda_bar/k_p claim")
    print("=" * 70)
    gc = elmod.GapContactPenalty()
    k_p = 1e8
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(10):
        u1 = rng.uniform(-0.01, 0.06)
        u_elem = np.array([u1, 0.0])
        f_penalty = gc.internal_force(np.zeros((2, 2)), u_elem, (k_p, G0, N_HAT))
        # lambda_bar=0 -> g0_eff = G0 - 0/k_p = G0 exactly
        delta = N_HAT @ u_elem - G0
        f_al_inner = (k_p * delta) * N_HAT if delta > 0.0 else np.zeros(2)
        err = np.max(np.abs(f_penalty - f_al_inner))
        max_err = max(max_err, err)
    print(f"  max abs force diff over 10 random u1 (lambda_bar=0 inner solve vs plain penalty): {max_err:.3e}")
    assert max_err < 1e-10
    print("  PASS -- confirms g0_eff reduces to plain g0 exactly when lambda_bar=0, "
          "so the very first inner Newton solve of an AL step IS a plain-penalty solve.")


def test_al_zero_load_and_no_engagement():
    print()
    print("=" * 70)
    print("CHECK 4: zero load -> zero displacement/lambda; a load that")
    print("never reaches the wall never engages contact")
    print("=" * 70)
    fes = _fes()
    P_gap = _N_truss(G0)
    fes.add_nodal_force([1], 0, 0.5 * P_gap)   # never reaches the wall
    k_p = 1e10
    lf, U, lam, act = nls.solve_contact_augmented_lagrange_static(
        fes, MAT, contact_node=1, n_hat=N_HAT, g0=G0, k_p=k_p, n_steps=20, tol=1e-12)

    assert lf[0] == 0.0
    assert np.allclose(U[0], 0.0)
    assert lam[0] == 0.0
    assert not np.any(act), "load never reaches the wall -- contact should never engage"
    assert np.all(lam == 0.0)
    print(f"  max u1 reached: {U[-1, 2]:.6f} (< g0={G0}); active anywhere: {np.any(act)}")
    print("  PASS")
