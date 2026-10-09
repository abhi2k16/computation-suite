# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"

import numpy as np
from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls



def test_contact():
    np.set_printoptions(precision=6, suppress=True)

    E, A = 200e9, 1e-4
    L0 = 1.0
    g0 = 0.02
    n_hat = np.array([1.0, 0.0])
    mat = (E, A)   # TrussTL2D convention

    elem_coords = np.array([[0.0, 0.0], [L0, 0.0]])
    gc = elmod.GapContactPenalty()


    def N_truss(u1):
        """Closed-form TrussTL2D axial force at node 1 (x-displacement u1),
        already validated in Module 8 (validate_nonlinear.py CHECK 1)."""
        E_GL = ((L0 + u1) ** 2 - L0 ** 2) / (2 * L0 ** 2)
        S = E * E_GL
        return S * A * (L0 + u1) / L0


    print("=" * 70)
    print("CHECK 1: GapContactPenalty internal_force/tangent_stiffness -- open & active")
    print("=" * 70)
    k_p = 5e9
    max_err1 = 0.0
    for u1 in [0.0, 0.01, 0.019999, 0.02, 0.025, 0.05]:
        u_elem = np.array([u1, 0.0])
        f = gc.internal_force(elem_coords, u_elem, (k_p, g0, n_hat))
        delta = u1 - g0
        f_cf = np.array([k_p * delta, 0.0]) if delta > 0 else np.zeros(2)
        err = np.max(np.abs(f - f_cf)) / max(k_p * 0.01, 1.0)
        max_err1 = max(max_err1, err)
        print(f"  u1={u1:8.5f}  delta={delta:+.5f}  f_fe={f}  f_cf={f_cf}  err={err:.2e}")
    assert max_err1 < 1e-12
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 2: tangent_stiffness vs finite difference (open & active branches)")
    print("=" * 70)
    max_err2 = 0.0
    for u1, label in [(0.01, "open"), (0.03, "active")]:
        u_elem = np.array([u1, 0.0])
        K_a = gc.tangent_stiffness(elem_coords, u_elem, (k_p, g0, n_hat))
        h = 1e-6
        K_fd = np.zeros((2, 2))
        for j in range(2):
            du = np.zeros(2); du[j] = h
            fp = gc.internal_force(elem_coords, u_elem + du, (k_p, g0, n_hat))
            fm = gc.internal_force(elem_coords, u_elem - du, (k_p, g0, n_hat))
            K_fd[:, j] = (fp - fm) / (2 * h)
        err = np.max(np.abs(K_a - K_fd)) / max(np.max(np.abs(K_a)), 1.0)
        max_err2 = max(max_err2, err)
        print(f"  {label:8s} (u1={u1}): max abs/rel err = {err:.2e}")
    assert max_err2 < 1e-6
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 3: penalty-contact FESystem solve vs closed-form equilibrium")
    print("=" * 70)
    nodes = np.array([[0.0, 0.0], [L0, 0.0]])
    elements = np.array([[0, 1]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)

    P_gap = N_truss(g0)
    print(f"  Force needed to just reach the wall (u1=g0): P_gap = {P_gap:.4f} N")

    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0], [0, 1])
    fes.fix_dofs([1], [1])
    fes.add_nodal_force([1], 0, 3 * P_gap)   # ramp well past the gap
    fes.add_contact_element(gc, [1], (k_p, g0, n_hat))

    load_factors, U_hist = nls.solve_nonlinear_static(fes, mat, n_steps=60, tol=1e-13)
    u1_hist = U_hist[:, 2]
    P_hist = load_factors * 3 * P_gap
    P_check = np.array([N_truss(u1) + k_p * max(u1 - g0, 0.0) for u1 in u1_hist])
    err3 = np.abs(P_hist - P_check)
    rel3 = err3 / max(P_hist.max(), 1.0)
    print(f"  {'u1':>10} {'P_applied':>12} {'P=N_truss+k_p*pen':>20} {'abs_err':>10}")
    for i in range(0, len(u1_hist), 10):
        print(f"  {u1_hist[i]:10.6f} {P_hist[i]:12.4f} {P_check[i]:20.4f} {err3[i]:10.2e}")
    print(f"  -> max abs equilibrium error: {err3.max():.3e} (max relative: {rel3.max():.2e})")
    assert rel3.max() < 1e-8
    n_open = np.sum(u1_hist < g0)
    n_active = np.sum(u1_hist >= g0)
    print(f"  -> {n_open} steps with gap open, {n_active} steps in contact")
    assert n_open > 3 and n_active > 3, "sweep should visit both regimes"
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 4: Lagrange-multiplier contact -- exact zero penetration")
    print("=" * 70)
    fes2 = FESystem(mesh, elmod.TrussTL2D())
    fes2.fix_dofs([0], [0, 1])
    fes2.fix_dofs([1], [1])
    fes2.add_nodal_force([1], 0, 3 * P_gap)
    # NOTE: no add_contact_element() here -- the Lagrange driver manages the
    # single contact constraint itself, on top of the plain structural system.

    load_factors2, U_hist2, lambda_hist, active_hist = nls.solve_contact_lagrange_static(
        fes2, mat, contact_node=1, n_hat=n_hat, g0=g0, n_steps=60, tol=1e-13)
    u1_hist2 = U_hist2[:, 2]
    P_hist2 = load_factors2 * 3 * P_gap

    print(f"  {'u1':>10} {'active':>7} {'lambda':>12} {'P-N_truss(g0)':>16} {'lam_err':>10}")
    max_pen = 0.0
    max_lam_err = 0.0
    for i in range(0, len(u1_hist2), 10):
        lam_cf = P_hist2[i] - N_truss(g0) if active_hist[i] else 0.0
        lam_err = abs(lambda_hist[i] - lam_cf)
        max_lam_err = max(max_lam_err, lam_err)
        print(f"  {u1_hist2[i]:10.6f} {str(active_hist[i]):>7} {lambda_hist[i]:12.4f} "
              f"{lam_cf:16.4f} {lam_err:10.2e}")
        if active_hist[i]:
            max_pen = max(max_pen, abs(u1_hist2[i] - g0))

    # full-resolution checks (not just the printed subsample)
    pen_active = np.abs(u1_hist2[active_hist] - g0)
    lam_cf_full = np.where(active_hist, P_hist2 - N_truss(g0), 0.0)
    lam_err_full = np.abs(lambda_hist - lam_cf_full)

    print(f"\n  -> max penetration while active: {pen_active.max():.3e} (should be ~0, EXACT constraint)")
    print(f"  -> max |lambda_FE - (P-N_truss(g0))| while active: {lam_err_full.max():.3e}")
    print(f"  -> lambda while inactive: max|lambda|={np.abs(lambda_hist[~active_hist]).max():.3e} (should be 0)")
    assert pen_active.max() < 1e-9, "Lagrange multiplier contact should give EXACT zero penetration"
    assert lam_err_full.max() < 1e-4
    assert np.abs(lambda_hist[~active_hist]).max() < 1e-9
    print("  PASS -- Lagrange multiplier enforces zero penetration to ~1e-9 (vs. the")
    print("          penalty method's finite, k_p-dependent penetration in CHECK 3).")

    print()
    print("=" * 70)
    print("CHECK 5: penalty -> Lagrange convergence as k_p increases")
    print("=" * 70)
    P_apply = 2.0 * P_gap   # comfortably in contact
    lam_exact = P_apply - N_truss(g0)
    print(f"  Applied P = {P_apply:.4f}, exact contact reaction (Lagrange) = {lam_exact:.4f}")
    print(f"  {'k_p':>12} {'penetration':>14} {'penalty force':>16} {'rel_err_vs_exact':>18}")
    prev_pen = None
    for k_p_test in [1e7, 1e8, 1e9, 1e10, 1e11, 1e12]:
        fes3 = FESystem(mesh, elmod.TrussTL2D())
        fes3.fix_dofs([0], [0, 1])
        fes3.fix_dofs([1], [1])
        fes3.add_nodal_force([1], 0, P_apply)
        fes3.add_contact_element(gc, [1], (k_p_test, g0, n_hat))
        # looser tol at extreme k_p: the scalar residual floors out around
        # machine-epsilon * (force scale) well before 1e-13*ref is reachable
        # -- doesn't affect the physics, only how tight Newton can converge.
        _, U_hist3 = nls.solve_nonlinear_static(fes3, mat, n_steps=80, tol=1e-8, max_iter=60)
        u1_final = U_hist3[-1, 2]
        penetration = u1_final - g0
        contact_force = k_p_test * penetration
        rel_err = abs(contact_force - lam_exact) / lam_exact
        print(f"  {k_p_test:12.1e} {penetration:14.3e} {contact_force:16.4f} {rel_err:18.2e}")
        if prev_pen is not None:
            assert penetration < prev_pen, "penetration should shrink as k_p grows"
        prev_pen = penetration
    assert rel_err < 1e-3, "penalty contact force should converge to the exact Lagrange value"
    print("  PASS -- penetration shrinks and the penalty contact force converges to the")
    print("          exact Lagrange-multiplier reaction as k_p is increased.")

    print()
    print("ALL CHECKS PASSED")
