# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
fea_03_hertzian_contact.py -- Example Gallery: an elastic block pressed
onto a rigid circular obstacle, using GapContactCurvedFriction
(elements/contact.py) attached to every bottom-row node -- mirrors
TensorMesh's Example Gallery / Solid Mechanics / Hertzian Contact
page.

Each bottom node gets its own contact element referencing the SAME
circle (center, R); the bulk block is an ordinary Quad4PlaneStress
mesh whose linear internal_force()=K@u (the base Element class's
generic default) combines with the contact elements' genuinely
nonlinear status-switch force through solve_nonlinear_static() --
the same fes.add_contact_element()/solve_nonlinear_static() pattern
tests/test_contact.py's own validated Check 3 uses, just on a mesh
instead of a single truss.

Panel 1: deformed block wrapping around the circular obstacle.
Panel 2: contact pressure (pn, at converged equilibrium) along the
bottom edge -- the classic Hertzian-contact pressure-distribution
shape (peaked under the obstacle, tapering to zero at the contact
patch edges). This is the element's own penalty-law pressure, not a
claim of quantitative agreement with analytical Hertz theory (a
different, non-penalty contact formulation) -- the SHAPE is the
point, and is checked against zero force outside the patch, not
against a closed-form Hertz pressure profile.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PolyCollection, LineCollection

from fea_engine import elements as elmod
from fea_engine.solver import FESystem
from fea_engine.mesh import rectangle_mesh
from fea_engine.material import Material, D_plane_stress
from fea_engine import nonlinear_solver as nls

STEEL = Material(E=8.0e7, nu=0.45, rho=1200.0)   # soft rubber-like, for a visible contact patch
LX, LY = 4.0, 1.0
NX, NY = 20, 5
R_OBSTACLE = 15.0
INITIAL_PENETRATION = 0.015   # pre-penetration at x=LX/2 so contact is active at u=0, avoiding a pre-contact rigid-body mode in y
K_P, K_T, MU = 3.0e7, 1.5e7, 0.3


def main():
    mesh = rectangle_mesh(Lx=LX, Ly=LY, nx=NX, ny=NY)
    D = D_plane_stress(STEEL)
    fs = FESystem(mesh, elmod.Quad4PlaneStress())

    top = mesh.nodes_on_line(axis=1, value=LY)
    for n in top:
        fs.fix_dofs([n], [0])                      # top edge: roller (x fixed)
        fs.add_nodal_force([n], 1, -1.8e6 / len(top))   # pushed straight down

    bottom = sorted(mesh.nodes_on_line(axis=0, value=0.0)) if False else None
    bottom = mesh.nodes_on_line(axis=1, value=0.0)
    center = np.array([LX / 2.0, -(R_OBSTACLE - INITIAL_PENETRATION)])
    ce = elmod.GapContactCurvedFriction()
    for n in bottom:
        fs.add_contact_element(ce, [n], (K_P, K_T, MU, center, R_OBSTACLE))

    load_factors, U_hist = nls.solve_nonlinear_static(fs, D, n_steps=40, tol=1e-3, max_iter=60)
    U = U_hist[-1]

    # ---- recover final contact pressure at every bottom node ----
    pressures = []
    for n in bottom:
        elem_coords = mesh.nodes[[n]]
        g = fs._global_dofs([n])
        u_elem = U[g]
        r = ce._trial(elem_coords, u_elem, (K_P, K_T, MU, center, R_OBSTACLE), ce.init_state())
        pressures.append(r["pn"] if r["active"] else 0.0)
    pressures = np.array(pressures)
    x_bottom = mesh.nodes[bottom, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # ---- Panel 1: deformed shape wrapping the obstacle ----
    undeformed = mesh.nodes[mesh.elements]
    ax1.add_collection(LineCollection([np.vstack([q, q[0]]) for q in undeformed],
                                       colors='0.8', linewidths=0.5, linestyles='dashed'))
    disp = np.zeros_like(mesh.nodes)
    for n in range(len(mesh.nodes)):
        disp[n] = U[fs._global_dofs([n])]
    deformed_nodes = mesh.nodes + disp
    quads = deformed_nodes[mesh.elements]
    pc = PolyCollection(quads, facecolor='steelblue', alpha=0.7, edgecolors='0.2', linewidths=0.3)
    ax1.add_collection(pc)
    circ = Circle(center, R_OBSTACLE, facecolor='0.85', edgecolor='k', linewidth=1.2, zorder=0)
    ax1.add_patch(circ)
    ax1.set_xlim(-0.3, LX + 0.3)
    ax1.set_ylim(-0.3, LY + 0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Deformed block wrapping a rigid circular obstacle')
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')

    # ---- Panel 2: contact pressure distribution along the bottom edge ----
    ax2.plot(x_bottom, pressures / 1e6, 'o-', color='tomato', markersize=5)
    ax2.fill_between(x_bottom, 0, pressures / 1e6, color='tomato', alpha=0.2)
    ax2.set_xlabel('x along bottom edge (m)')
    ax2.set_ylabel('contact pressure $p_n$ (MPa)')
    ax2.set_title('Contact pressure distribution (final load step)')
    ax2.grid(True, alpha=0.35)
    ax2.axhline(0, color='k', linewidth=0.6)

    fig.suptitle('Hertzian-style Contact -- GapContactCurvedFriction (fea_engine)', fontsize=13)
    fig.savefig('fea_03_hertzian_contact.png', dpi=150)
    print("Saved fea_03_hertzian_contact.png")
    n_active = np.sum(pressures > 0)
    print(f"Converged {len(load_factors)-1} load steps; {n_active}/{len(bottom)} bottom nodes "
          f"in contact at final load, peak pressure {pressures.max()/1e6:.2f} MPa")


if __name__ == "__main__":
    main()
