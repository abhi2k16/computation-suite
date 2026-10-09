# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
shell_plot.py -- Phase 7 (general-purpose extensions roadmap) worked
example: visual verification companion to shell_demo.py.

Two panels: (1) the flat-plate-limit ratio (Shell4MITC / Quad4MindlinPlate
tip deflection) converging to 1 as the mesh refines -- a converging
curve, not just a final small number, is the "convincing" signature
here; (2) the curved cylindrical-arc cantilever's mid-surface mesh,
undeformed (wireframe) against its exaggerated deformed shape under the
tip load -- a genuinely curved, non-flat shell actually bending, shown
in 3-D, the qualitative check a table of numbers alone can't give.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import matplotlib.pyplot as plt

from fea_engine import Shell4MITC, D_shell, Material, D_mindlin_plate, Quad4MindlinPlate, FESystem
from fea_engine.mesh import Mesh, rectangle_mesh


E, NU, RHO = 2.1e11, 0.3, 7850.0


def main():
    mat = Material(E=E, nu=NU, rho=RHO)

    # ---- Panel 1: flat-plate-limit convergence ----
    h, L, W = 0.01, 1.0, 0.2
    Db, Ds = D_mindlin_plate(mat, h)
    D_sh = D_shell(mat, h)
    nx_list = np.array([4, 8, 16, 32, 64])
    ratios = []
    for nx in nx_list:
        ny = max(2, nx // 8)
        mesh_p = rectangle_mesh(L, W, nx, ny)
        sys_p = FESystem(mesh_p, Quad4MindlinPlate(), sparse=True)
        sys_p.assemble_stiffness((Db, Ds))
        left = mesh_p.nodes_on_plane(axis=0, value=0.0)
        sys_p.fix_dofs(left, [0, 1, 2])
        tip = mesh_p.nodes_on_plane(axis=0, value=L)
        sys_p.add_nodal_force(tip, 0, -1000.0)
        w_p = sys_p.solve_static()[3 * tip[0]]

        nodes3 = np.column_stack([mesh_p.nodes, np.zeros(len(mesh_p.nodes))])
        mesh_s = Mesh(nodes3, mesh_p.elements, dim=3)
        sys_s = FESystem(mesh_s, Shell4MITC(), sparse=True)
        sys_s.assemble_stiffness(D_sh)
        sys_s.fix_dofs(left, [0, 1, 2, 3, 4, 5])
        sys_s.add_nodal_force(tip, 2, -1000.0)
        w_s = sys_s.solve_static()[6 * tip[0] + 2]
        ratios.append(w_s / w_p)
    ratios = np.array(ratios)

    # ---- Panel 2: curved shell deformed shape ----
    R, h2, Wd, theta_max = 1.0, 0.02, 0.3, 0.3
    n_theta, n_y = 16, 3
    raw = rectangle_mesh(theta_max, Wd, n_theta, n_y)
    theta, y = raw.nodes[:, 0], raw.nodes[:, 1]
    nodes = np.column_stack([R * np.cos(theta), y, -R * np.sin(theta)])
    mesh = Mesh(nodes, raw.elements, dim=3)

    D_curved = D_shell(mat, h2)
    sys = FESystem(mesh, Shell4MITC(), sparse=False)
    sys.assemble_stiffness(D_curved)
    fixed = np.where(np.isclose(raw.nodes[:, 0], 0.0))[0]
    sys.fix_dofs(fixed, [0, 1, 2, 3, 4, 5])
    tip = np.where(np.isclose(raw.nodes[:, 0], theta_max))[0]
    rad_dir = np.array([np.cos(theta_max), 0.0, -np.sin(theta_max)])
    for d in range(3):
        sys.add_nodal_force(tip, d, -1500.0 * rad_dir[d])
    U = sys.solve_static()
    disp = U.reshape(-1, 6)[:, :3]

    scale = 40.0
    deformed = mesh.nodes + scale * disp

    fig = plt.figure(figsize=(12.5, 5.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(nx_list, ratios, 'o-', color='seagreen')
    ax1.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.6)
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('elements along length (nx)')
    ax1.set_ylabel(r'$w_{shell} / w_{plate}$')
    ax1.set_title('Flat-plate limit: Shell4MITC vs. Quad4MindlinPlate')
    ax1.grid(True, alpha=0.35)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for elem in mesh.elements:
        for i0, i1 in edges:
            p0, p1 = mesh.nodes[elem[i0]], mesh.nodes[elem[i1]]
            ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                      color='0.7', lw=0.6, linestyle='--')
    for elem in mesh.elements:
        for i0, i1 in edges:
            p0, p1 = deformed[elem[i0]], deformed[elem[i1]]
            ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                      color='tomato', lw=1.2)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_zlabel('z (m)')
    ax2.set_title(f'Curved cantilever strip: undeformed (dashed) vs.\n'
                   f'deformed (x{scale:.0f} exaggerated), radial tip load')
    ax2.view_init(elev=18, azim=-60)

    fig.suptitle('Shell4MITC (MITC4) verification')
    fig.tight_layout()
    fig.savefig('shell_verification.png', dpi=150)
    print("Saved shell_verification.png")


if __name__ == "__main__":
    main()
