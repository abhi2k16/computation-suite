# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
arc_length_snap_through_plot.py -- Phase 5 (general-purpose extensions
roadmap) worked example: visual verification companion to
arc_length_snap_through_demo.py.

One panel, but a decisive one: the full P vs delta equilibrium path of
the von Mises truss, with load control's incomplete path (stops dead
at the limit point), displacement control's full path, and arc-
length's full path (going further still, past full inversion) all
plotted together against the closed-form curve -- the "S-shaped"
snap-through curve made visible, and each solver's actual REACH along
it shown directly rather than described in words.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import matplotlib.pyplot as plt

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

    # ---- 1) load control: only reaches up to (just before) the limit point ----
    fes1, control_dof = von_mises_truss()
    P_apply = 0.9 * P_peak
    fes1.add_nodal_force([1], 1, -P_apply)
    load_factors1, U_lc = nls.solve_nonlinear_static(fes1, (E, A), n_steps=40, tol=1e-8)
    delta_lc = -U_lc[:, control_dof]
    P_lc = load_factors1 * P_apply

    # ---- 2) displacement control: full path, 0 -> 2*h0 ----
    fes2, control_dof2 = von_mises_truss()
    delta_dc = np.linspace(0.0, 2 * H0, 81)
    U_dc, reaction_dc = nls.solve_nonlinear_displacement_control(
        fes2, (E, A), control_dof2, -delta_dc, tol=1e-12, max_iter=50)
    P_dc = -reaction_dc

    # ---- 3) arc-length: full path, continues past full inversion ----
    fes3, control_dof3 = von_mises_truss()
    fes3.add_nodal_force([1], 1, -1.0)
    load_factors3, U_al = nls.solve_nonlinear_arc_length(
        fes3, (E, A), delta_L=0.01, n_steps=60, tol=1e-10, max_iter=60)
    delta_al = -U_al[:, control_dof3]
    P_al = load_factors3

    # ---- closed form, for reference, over the arc-length solver's own reach ----
    delta_ref = np.linspace(delta_al.min(), delta_al.max(), 400)
    P_ref = P_closed_form(delta_ref)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    for ax, xlim in ((ax1, (-0.01, 0.22)), (ax2, (delta_al.min(), delta_al.max()))):
        ax.plot(delta_ref, P_ref, '-', color='k', linewidth=1.2, alpha=0.6,
                label='closed form: $P(\\delta)$', zorder=1)
        ax.plot(delta_lc, P_lc, 'o-', color='tomato', markersize=4,
                label=f'load control (stops at $P$={P_apply:.0f} N,\npre-limit-point)', zorder=3)
        ax.plot(delta_dc, P_dc, 's-', color='steelblue', markersize=3,
                label='displacement control (full path)', zorder=2)
        ax.plot(delta_al, P_al, '^-', color='seagreen', markersize=4,
                label='arc-length (full path, continues\npast full inversion)', zorder=4)
        ax.axhline(0.0, color='0.7', linewidth=0.8)
        ax.set_xlim(*xlim)
        ax.set_xlabel(r'apex downward displacement $\delta$ (m)')
        ax.grid(True, alpha=0.35)

    ax1.axvline(H0, color='0.8', linewidth=0.8, linestyle=':')
    ax1.axvline(2 * H0, color='0.8', linewidth=0.8, linestyle=':')
    ax1.set_ylim(-P_peak * 1.4, P_peak * 1.4)   # zoom the y-axis to this
    # region's own scale -- otherwise the snap-through S-curve (order
    # P_peak ~ 1.6e4 N) is invisible next to the full path's order-1e6 N end
    ax1.annotate('limit point', xy=(delta_peak, P_peak), xytext=(delta_peak + 0.03, P_peak * 1.15),
                 fontsize=9, color='0.3', arrowprops=dict(arrowstyle='->', color='0.5'))
    ax1.annotate('fully inverted\n(P=0)', xy=(2 * H0, 0), xytext=(2 * H0 - 0.10, -P_peak * 1.15),
                 fontsize=9, color='0.3', arrowprops=dict(arrowstyle='->', color='0.5'))
    ax1.set_ylabel(r'applied load $P$ (N)')
    ax1.set_title('snap-through region (zoomed)')
    ax1.legend(loc='upper right', fontsize=8)

    ax2.set_title('full traced path')

    fig.suptitle('von Mises truss snap-through: three solvers, one closed-form curve')
    fig.savefig('arc_length_snap_through.png', dpi=150)
    print("Saved arc_length_snap_through.png")


if __name__ == "__main__":
    main()
