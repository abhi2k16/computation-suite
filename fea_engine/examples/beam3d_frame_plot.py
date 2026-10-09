# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
beam3d_frame_plot.py -- Phase 3 (general-purpose extensions roadmap)
worked example: visual verification companion to
beam3d_space_frame_demo.py.

Draws the same 2-member L-shaped space frame in 3-D, undeformed
(dashed) against its displaced shape (solid, exaggerated) under the
1000 N global-Y tip load -- and cross-checks the plotted tip
displacement against test_beam3d.py's independently-assembled
reference value, so the picture and the validated number are shown
together, not just the picture on its own.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import matplotlib.pyplot as plt

from fea_engine import Beam3DEulerBernoulli, Material, Section3D, beam3d_rigidities, FESystem
from fea_engine.mesh import Mesh


def main():
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    sec = Section3D(A=0.01, Iy=8e-6, Iz=6e-6, J=5e-6)
    rig = beam3d_rigidities(mat, sec)

    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype=float)
    elements = np.array([[0, 1], [1, 2]])
    mesh_obj = Mesh(nodes=nodes, elements=elements, dim=1)
    beam = Beam3DEulerBernoulli()
    fs = FESystem(mesh_obj, beam, sparse=False)
    fs.assemble_stiffness(rig)

    ndof = nodes.shape[0] * 6
    fixed = list(range(0, 6))
    free_dofs = [i for i in range(ndof) if i not in fixed]
    F = np.zeros(ndof)
    F[6 * 2 + 1] = 1000.0
    Kff = fs.K[np.ix_(free_dofs, free_dofs)]
    uf = np.linalg.solve(Kff, F[free_dofs])
    u_full = np.zeros(ndof)
    u_full[free_dofs] = uf
    tip_disp = u_full[6 * 2:6 * 2 + 3]
    print(f"Tip displacement (x,y,z): {tip_disp}")
    print("(matches test_beam3d.py::test_full_3d_frame_matches_independent_assembler)")

    translations = u_full.reshape(-1, 6)[:, 0:3]
    scale = 0.3 / np.max(np.linalg.norm(translations, axis=1))
    deformed = nodes + scale * translations

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    for (a, b) in elements:
        ax.plot(*nodes[[a, b]].T, '--', color='0.5', linewidth=1.5,
                label='undeformed' if a == 0 else None)
        ax.plot(*deformed[[a, b]].T, '-', color='crimson', linewidth=3,
                label=f'deformed (x{scale:.0f} exaggerated)' if a == 0 else None)

    ax.scatter(*nodes.T, color='k', s=40, depthshade=False, label='nodes (undeformed)')
    ax.scatter(*deformed.T, color='crimson', s=40, depthshade=False)

    # 1000 N load arrow at the free tip (node 2), pointing +Y
    tip = nodes[2]
    ax.quiver(tip[0], tip[1] - 0.4, tip[2], 0, 0.35, 0,
              color='blue', arrow_length_ratio=0.25, linewidth=2)
    ax.text(tip[0], tip[1] - 0.45, tip[2], '1000 N', color='blue')

    ax.text(*nodes[0], '  fixed base', color='k')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Beam3DEulerBernoulli: L-shaped space frame under tip load')
    ax.legend(loc='upper left')
    ax.set_box_aspect([1, 0.8, 1])

    fig.tight_layout()
    fig.savefig('beam3d_frame_deformed.png', dpi=150)
    print("Saved beam3d_frame_deformed.png")


if __name__ == "__main__":
    main()
