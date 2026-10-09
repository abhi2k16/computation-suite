# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"

import numpy as np
from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls



def test_nonlinear():
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 70)
    print("CHECK 1: single-bar pure axial stretch vs closed-form St Venant-Kirchhoff")
    print("=" * 70)
    E, A = 200e9, 1e-4
    L0 = 2.0
    elem_coords = np.array([[0.0, 0.0], [L0, 0.0]])
    tr = elmod.TrussTL2D()

    max_rel_err = 0.0
    for stretch_frac in [0.001, 0.01, 0.05, 0.10, 0.20, -0.05, -0.10]:
        u = stretch_frac * L0
        u_elem = np.array([0.0, 0.0, u, 0.0])
        f = tr.internal_force(elem_coords, u_elem, (E, A))
        N_fe = f[2]   # x-force at node 2 = axial force magnitude

        E_GL_cf = ((L0 + u) ** 2 - L0 ** 2) / (2 * L0 ** 2)
        S_cf = E * E_GL_cf     # 2nd Piola-Kirchhoff stress
        # The FE nodal force is NOT S*A -- it's the physical (current-
        # configuration) axial force, i.e. 1st Piola-Kirchhoff stress times
        # reference area: P = F*S (1-D deformation gradient F = l/L0), so
        # N = P*A = S*A*(l/L0). This is the standard TL-truss result (e.g.
        # Crisfield Vol 1); S*A alone is only the force conjugate to a
        # REFERENCE-length measure, not the actual nodal force.
        N_cf = S_cf * A * (L0 + u) / L0

        rel_err = abs(N_fe - N_cf) / max(abs(N_cf), 1e-30)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  stretch={stretch_frac:+.3f}  N_fe={N_fe: .6e}  N_closed_form={N_cf: .6e}"
              f"  rel_err={rel_err:.2e}")

    print(f"  -> max relative error over all cases: {max_rel_err:.2e}")
    assert max_rel_err < 1e-10, "single-bar internal_force does not match closed form"
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 2: analytical tangent_stiffness vs finite-difference of internal_force")
    print("=" * 70)
    rng = np.random.default_rng(0)
    max_rel_err2 = 0.0
    for trial in range(5):
        elem_coords_t = np.array([[0.0, 0.0], [1.3, 0.7]]) + 0.1 * rng.standard_normal((2, 2))
        u_elem_t = 0.05 * rng.standard_normal(4)
        K_analytical = tr.tangent_stiffness(elem_coords_t, u_elem_t, (E, A))

        h = 1e-7
        K_fd = np.zeros((4, 4))
        for j in range(4):
            du = np.zeros(4); du[j] = h
            fp = tr.internal_force(elem_coords_t, u_elem_t + du, (E, A))
            fm = tr.internal_force(elem_coords_t, u_elem_t - du, (E, A))
            K_fd[:, j] = (fp - fm) / (2 * h)

        err = np.max(np.abs(K_analytical - K_fd)) / max(np.max(np.abs(K_analytical)), 1e-30)
        max_rel_err2 = max(max_rel_err2, err)
        print(f"  trial {trial}: max |K_analytical - K_fd| / max|K_analytical| = {err:.2e}")

    print(f"  -> max relative error over all trials: {max_rel_err2:.2e}")
    assert max_rel_err2 < 1e-5, "tangent_stiffness does not match finite-difference"
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 3: two-bar (von Mises) truss snap-through vs closed-form P(delta)")
    print("=" * 70)
    a = 1.0
    h0 = 0.10
    E2, A2 = 210e9, 2e-4
    L0_2 = np.sqrt(a ** 2 + h0 ** 2)

    nodes = np.array([[-a, 0.0], [0.0, h0], [a, 0.0]])
    elements = np.array([[0, 1], [1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)

    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1])          # both supports fully fixed
    control_dof = 1 * fes.npn + 1          # apex node, y-dof

    n_steps = 41
    delta = np.linspace(0.0, 2 * h0, n_steps)   # delta = downward motion of apex, 0 -> full inversion
    u_targets = -delta   # apex y-DISPLACEMENT is NEGATIVE as it moves down -- this is
                          # the control_dof value passed to the FE solver; delta itself
                          # (always >= 0) is only the bookkeeping variable used in the
                          # closed-form P(delta) derivation below.

    U_hist, reaction_raw = nls.solve_nonlinear_displacement_control(
        fes, (E2, A2), control_dof, u_targets, tol=1e-12, max_iter=50)
    # F_int at the control dof is the UPWARD restoring force; the applied
    # force needed to hold the apex there is its negative.
    reaction_hist = -reaction_raw

    # closed form (derived via total potential energy, see nonlinear_solver.py /
    # README): P(delta) = (E*A/L0^3) * delta * (h0-delta) * (2h0-delta)
    def P_closed_form(delta):
        return (E2 * A2 / L0_2 ** 3) * delta * (h0 - delta) * (2 * h0 - delta)

    u_targets_report = delta   # for the printout below
    P_cf = P_closed_form(delta)
    abs_scale = max(np.max(np.abs(P_cf)), 1e-30)
    err = np.abs(reaction_hist - P_cf)
    rel_err = err / abs_scale
    max_rel_err3 = np.max(rel_err)

    print(f"  {'delta':>10} {'P_FE':>14} {'P_closed_form':>16} {'abs_err':>12}")
    for i in range(0, n_steps, 4):
        print(f"  {delta[i]:10.4f} {reaction_hist[i]:14.4f} {P_cf[i]:16.4f} {err[i]:12.3e}")

    print(f"  -> max |P_FE - P_closed_form| / max|P_closed_form| over all {n_steps} steps: "
          f"{max_rel_err3:.2e}")
    print(f"  -> peak P (pre-snap):  FE={reaction_hist.max():.4f}  closed-form={P_cf.max():.4f}")
    print(f"  -> trough P (post-snap): FE={reaction_hist.min():.4f}  closed-form={P_cf.min():.4f}")
    assert max_rel_err3 < 1e-6, "displacement-controlled snap-through does not match closed form"
    print("  PASS -- full snap-through path (including the descending/negative-P branch)"
          " traced correctly by displacement control, matching closed form at every step.")

    print()
    print("=" * 70)
    print("CHECK 4: solve_nonlinear_static (load control) on the SAME truss, pre-limit-point only")
    print("=" * 70)
    # Load control only works up to the limit point (max P); use a small
    # fraction of the peak load so this is a genuine independent check of
    # assemble_internal_force/assemble_tangent_stiffness/solve_nonlinear_static
    # via a totally different code path (Newton-Raphson on lambda*F_ext,
    # not prescribed-displacement), compared against the SAME closed form.
    P_peak = P_cf.max()
    P_apply = 0.5 * P_peak   # comfortably pre-limit-point
    fes2 = FESystem(mesh, elmod.TrussTL2D())
    fes2.fix_dofs([0, 2], [0, 1])
    fes2.add_nodal_force([1], 1, -P_apply)   # downward force at apex (y-dof), matches +delta sign convention

    load_factors, U_hist2 = nls.solve_nonlinear_static(fes2, (E2, A2), n_steps=20, tol=1e-12)
    delta_fe = -U_hist2[-1, control_dof]     # apex moves down (negative y) under downward load

    # invert closed form P(delta)=P_apply for the small (first) root near delta=0
    from scipy.optimize import brentq
    # P(delta) rises from 0, peaks at delta_peak = h0*(3-sqrt(3))/3 (root of
    # dP/ddelta=0), then falls back to 0 at delta=h0 -- so P_apply=0.5*peak
    # is crossed TWICE in (0, h0); restrict the bracket to the ascending
    # branch only (0, delta_peak) to get the physically relevant root.
    delta_peak = h0 * (3 - np.sqrt(3)) / 3
    delta_cf = brentq(lambda d: P_closed_form(d) - P_apply, 1e-9, delta_peak)

    rel_err4 = abs(delta_fe - delta_cf) / abs(delta_cf)
    print(f"  applied P = {P_apply:.4f}  (peak P = {P_peak:.4f})")
    print(f"  delta_FE (load control)    = {delta_fe:.8f}")
    print(f"  delta_closed_form (brentq) = {delta_cf:.8f}")
    print(f"  relative error = {rel_err4:.2e}")
    assert rel_err4 < 1e-6, "load-controlled Newton-Raphson does not match closed form"
    print("  PASS")

    print()
    print("ALL CHECKS PASSED")
