"""
higher_order_elements_plot.py -- Phase 1 (general-purpose extensions
roadmap) worked example: visual verification companion to
higher_order_elements_demo.py.

Reuses that script's exact coarse pure-bending setup (2 elements long,
1 through the height -- the standard shear-locking stress test) and
adds two panels: (1) a bar chart of tip-deflection ratio to the exact
elasticity solution, making the ~33% locking error and Quad8's
near-exact match immediately visible side by side; (2) the two
elements' DEFORMED shapes (displacement exaggerated) drawn over their
undeformed outlines, so the locking is visible as a shape, not just a
number -- Quad4 barely curves, Quad8 traces the smooth bent beam.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from fea_engine import (
    FESystem, Material, D_plane_stress, Quad4PlaneStress, Quad8PlaneStress, mesh,
)


def run_case(E, nu, L, h, t, M_applied):
    D2 = D_plane_stress(Material(E=E, nu=nu))
    F = M_applied / h

    # ---- Quad4: 2 elements long x 1 tall ----
    m4 = mesh.rectangle_mesh(L, h, 2, 1)
    sys4 = FESystem(m4, Quad4PlaneStress(), thickness=t)
    sys4.assemble_stiffness(D2)
    for n in m4.nodes_on_line(0, 0.0):
        sys4.fix_dofs([n], [0, 1])
    top4 = np.where((np.abs(m4.nodes[:, 0] - L) < 1e-9) & (np.abs(m4.nodes[:, 1] - h) < 1e-9))[0][0]
    bot4 = np.where((np.abs(m4.nodes[:, 0] - L) < 1e-9) & (np.abs(m4.nodes[:, 1] - 0.0) < 1e-9))[0][0]
    sys4.F[2 * top4] = F
    sys4.F[2 * bot4] = -F
    U4 = sys4.solve_static()

    # ---- Quad8: SAME coarse layout, hand-built (matches the demo script) ----
    nodes8 = np.array([
        (0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5),
        (2, 0), (2, 1), (1.5, 0), (2, 0.5), (1.5, 1)], dtype=float)
    elems8 = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 10, 11, 12, 5]])
    m8 = mesh.Mesh(nodes=nodes8, elements=elems8, dim=2)
    sys8 = FESystem(m8, Quad8PlaneStress(), thickness=t)
    sys8.assemble_stiffness(D2)
    for n in np.where(np.abs(m8.nodes[:, 0] - 0.0) < 1e-9)[0]:
        sys8.fix_dofs([n], [0, 1])
    top8 = np.where((np.abs(m8.nodes[:, 0] - L) < 1e-9) & (np.abs(m8.nodes[:, 1] - h) < 1e-9))[0][0]
    bot8 = np.where((np.abs(m8.nodes[:, 0] - L) < 1e-9) & (np.abs(m8.nodes[:, 1] - 0.0) < 1e-9))[0][0]
    sys8.F[2 * top8] = F
    sys8.F[2 * bot8] = -F
    U8 = sys8.solve_static()

    return (m4, U4), (m8, U8)


def draw_deformed(ax, m, U, scale, color, label):
    n_nodes = m.nodes.shape[0]
    disp = U.reshape(n_nodes, 2)
    deformed = m.nodes + scale * disp
    patches = []
    for elem in m.elements:
        # use only the 4 CORNER nodes for a clean outline, even for Quad8
        # (corners are always listed first in this package's node ordering)
        corner_idx = elem[:4] if len(elem) > 4 else elem
        patches.append(Polygon(deformed[corner_idx], closed=True))
        ax.plot(*m.nodes[np.append(corner_idx, corner_idx[0])].T,
                 '--', color='0.7', linewidth=0.8)
    pc = PatchCollection(patches, facecolor=color, edgecolor='k', alpha=0.35, linewidth=1.2)
    ax.add_collection(pc)
    ax.plot([], [], color=color, alpha=0.6, linewidth=6, label=label)


def main():
    E, nu = 210e9, 0.0
    L, h, t = 2.0, 1.0, 1.0
    I = t * h**3 / 12.0
    M_applied = 1.0e6
    w_exact = M_applied * L**2 / (2 * E * I)

    (m4, U4), (m8, U8) = run_case(E, nu, L, h, t, M_applied)
    tip4 = np.where(np.abs(m4.nodes[:, 0] - L) < 1e-9)[0]
    tip_uy4 = U4[2 * tip4 + 1].mean()
    tip8 = np.where(np.abs(m8.nodes[:, 0] - L) < 1e-9)[0]
    tip_uy8 = U8[2 * tip8 + 1].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Panel 1: ratio bar chart (magnitude -- the deflection sign just
    # reflects the applied moment's direction, not accuracy)
    labels = ['Quad4\n(4 elem, 2x1)', 'Quad8\n(2 elem, 2x1)']
    ratios = [abs(tip_uy4 / w_exact), abs(tip_uy8 / w_exact)]
    bars = ax1.bar(labels, ratios, color=['tomato', 'seagreen'], width=0.5)
    ax1.axhline(1.0, color='k', linestyle='--', linewidth=1.2, label='exact elasticity solution')
    for b, r in zip(bars, ratios):
        ax1.text(b.get_x() + b.get_width() / 2, r + 0.02, f'{r:.3f}', ha='center', fontsize=11)
    ax1.set_ylabel('FEM tip deflection / exact tip deflection')
    ax1.set_title('Coarse pure-bending patch test (same element count)')
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, axis='y', alpha=0.4)
    ax1.legend(loc='lower right')

    # Panel 2: deformed shapes, exaggerated, overlaid on undeformed outline
    scale = 0.3 * h / max(abs(tip_uy4), abs(tip_uy8))  # normalize so the LARGER
    # deflection (Quad8, closer to exact) reaches a fixed visual amplitude --
    # this makes Quad4's under-deflection (locking) visually obvious rather
    # than washing it out with an arbitrary fixed scale factor.
    draw_deformed(ax2, m4, U4, scale, 'tomato', 'Quad4 (locked)')
    draw_deformed(ax2, m8, U8, scale, 'seagreen', 'Quad8')
    ax2.set_xlim(-0.1, L + 0.1)
    ax2.set_ylim(-0.4, h + 0.15)
    ax2.set_aspect('equal')
    ax2.set_title('Deformed shape under pure bending (displacement exaggerated)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Quad4 vs Quad8 under coarse pure bending -- shear locking, visually')
    fig.savefig('higher_order_elements_locking.png', dpi=150)
    print("Saved higher_order_elements_locking.png")


if __name__ == "__main__":
    main()
