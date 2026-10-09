# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
arc_length_snap_through_demo.py -- Phase 5 (general-purpose extensions
roadmap) worked example: nonlinear_solver.solve_nonlinear_arc_length()
on the classic von Mises (two-bar) snap-through truss.

Three-part demonstration, in increasing order of what each solver
CAN'T do: (1) load control (solve_nonlinear_static()) fails once the
applied load passes the truss's limit point -- exactly as its own
docstring says; (2) displacement control sails through that same limit
point by prescribing the apex's own displacement instead of the load;
(3) arc-length continuation gets the SAME full path as displacement
control WITHOUT prescribing anything at a single DOF -- both the
displacement increment and the load-factor increment are solved for
together every step, checked the whole way against the same closed-
form P(delta) = (E*A/L0^3)*delta*(h0-delta)*(2*h0-delta) used to
validate every solver in this module (see test_arc_length.py /
nonlinear_solver.py for the derivation).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls


A_HALFSPAN, H0 = 1.0, 0.10
E, A = 210e9, 2e-4
L0 = np.sqrt(A_HALFSPAN**2 + H0**2)


def von_mises_truss():
    nodes = np.array([[-A_HALFSPAN, 0.0], [0.0, H0], [A_HALFSPAN, 0.0]])
    elements = np.array([[0, 1], [1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1])
    control_dof = 1 * fes.npn + 1
    return fes, control_dof


def P_closed_form(delta):
    return (E * A / L0**3) * delta * (H0 - delta) * (2 * H0 - delta)


def main():
    delta_peak = H0 * (3 - np.sqrt(3)) / 3
    P_peak = P_closed_form(delta_peak)
    print("=" * 72)
    print("von Mises truss: a=%.2f m, h0=%.3f m, peak snap-through load P_peak=%.2f N"
          % (A_HALFSPAN, H0, P_peak))
    print("=" * 72)

    print("\n1) Load control, applied P = 1.1 * P_peak (past the limit point):")
    fes1, _ = von_mises_truss()
    fes1.add_nodal_force([1], 1, -1.1 * P_peak)
    try:
        nls.solve_nonlinear_static(fes1, (E, A), n_steps=20, tol=1e-12)
        print("   UNEXPECTED: converged (should have failed)")
    except RuntimeError as e:
        print(f"   FAILED to converge, as expected: {str(e).splitlines()[0][:90]}...")

    print("\n2) Displacement control, apex driven from delta=0 to delta=2*h0:")
    fes2, control_dof = von_mises_truss()
    delta_dc = np.linspace(0.0, 2 * H0, 41)
    U_dc, reaction_dc = nls.solve_nonlinear_displacement_control(
        fes2, (E, A), control_dof, -delta_dc, tol=1e-12, max_iter=50)
    P_dc = -reaction_dc
    err_dc = np.max(np.abs(P_dc - P_closed_form(delta_dc))) / max(np.abs(P_dc).max(), 1e-30)
    print(f"   traced full path, max rel. error vs closed form: {err_dc:.2e}")

    print("\n3) Arc-length continuation, SAME truss, only a reference load direction given:")
    fes3, control_dof3 = von_mises_truss()
    fes3.add_nodal_force([1], 1, -1.0)
    load_factors, U_al = nls.solve_nonlinear_arc_length(
        fes3, (E, A), delta_L=0.01, n_steps=60, tol=1e-10, max_iter=60)
    delta_al = -U_al[:, control_dof3]
    P_al = load_factors
    scale = max(np.abs(P_closed_form(delta_al)).max(), 1e-30)
    significant = np.abs(P_closed_form(delta_al)) > 1e-4 * scale
    err_al = np.max(np.abs(P_al - P_closed_form(delta_al))[significant]
                     / np.abs(P_closed_form(delta_al))[significant])
    print(f"   traced delta from {delta_al.min():.4f} to {delta_al.max():.4f} m "
          f"(past full inversion at {2*H0:.3f} m)")
    print(f"   max rel. error vs closed form: {err_al:.2e}")
    print(f"   (no delta ever prescribed -- lambda AND every displacement dof")
    print(f"    were solved for simultaneously, every step)")

    print("\nAll three solvers agree with the SAME closed-form P(delta) wherever")
    print("they're each capable of reaching it -- load control only up to the")
    print("limit point, displacement control and arc-length across the full path.")


if __name__ == "__main__":
    main()
