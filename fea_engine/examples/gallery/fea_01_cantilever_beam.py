# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
fea_01_cantilever_beam.py -- Example Gallery: linear-elastic cantilever
beam (Quad4PlaneStress), the baseline "hello world" of the solid-
mechanics ladder -- mirrors TensorMesh's own Example Gallery /
Solid Mechanics / Cantilever Beam page, using real fea_engine code
(FESystem.solve_static(), no toy/standalone physics).

Panel 1: deformed shape (magnified) colored by element-centroid von
Mises stress, sigma = D @ B(0,0) @ u_elem evaluated with the SAME
B_matrix()/D_plane_stress() the stiffness assembly itself already
uses -- not a new, separately-validated stress-recovery module, just
the constitutive law applied once more after the solve (standard
FEA post-processing).

Panel 2: mesh-refinement convergence of the FE tip deflection toward
the Euler-Bernoulli analytical prediction. A 2-D continuum cantilever
does not converge EXACTLY to 1-D beam theory (root effects / shear
flexibility a slender-beam formula ignores), so the honest claim here
is "approaches," not "matches" -- shown directly rather than asserted.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

from fea_engine import Quad4PlaneStress, FESystem, Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh

STEEL = Material(E=2.1e11, nu=0.3, rho=7850.0)
L, H = 10.0, 1.0          # beam length, depth (m)
P_TOTAL = -5.0e4           # total tip load (N), downward


def solve_cantilever(nx, ny):
    mesh = rectangle_mesh(Lx=L, Ly=H, nx=nx, ny=ny)
    D = D_plane_stress(STEEL)
    fs = FESystem(mesh, Quad4PlaneStress())
    fs.assemble_stiffness(D)
    left = mesh.nodes_on_line(axis=0, value=0.0)
    for n in left:
        fs.fix_dofs([n], [0, 1])
    tip = mesh.nodes_on_line(axis=0, value=L)
    for n in tip:
        fs.F[fs._global_dofs([n])[1]] += P_TOTAL / len(tip)
    U = fs.solve_static()
    return mesh, fs, D, U


def element_von_mises(mesh, fs, D, U):
    elem = fs.elem
    vm = np.zeros(len(mesh.elements))
    for e, conn in enumerate(mesh.elements):
        g = fs._global_dofs(conn)
        u_elem = U[g]
        B, _ = elem.B_matrix((0.0, 0.0), mesh.nodes[conn])
        sx, sy, txy = D @ (B @ u_elem)
        vm[e] = np.sqrt(sx**2 - sx * sy + sy**2 + 3 * txy**2)
    return vm


def main():
    mesh, fs, D, U = solve_cantilever(nx=40, ny=6)
    vm = element_von_mises(mesh, fs, D, U)

    tip_dof = fs._global_dofs(mesh.nodes_on_line(axis=0, value=L))[1]
    tip_defl = U[tip_dof].mean()
    mag = 0.15 * H / max(abs(tip_defl), 1e-12)   # magnify so deflection ~ 15% of depth

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # ---- Panel 1: deformed shape colored by von Mises stress ----
    undeformed = mesh.nodes[mesh.elements]
    lc = LineCollection([np.vstack([q, q[0]]) for q in undeformed],
                         colors='0.75', linewidths=0.6, linestyles='dashed')
    ax1.add_collection(lc)

    disp = np.zeros_like(mesh.nodes)
    for n in range(len(mesh.nodes)):
        g = fs._global_dofs([n])
        disp[n] = U[g]
    deformed_nodes = mesh.nodes + mag * disp
    quads = deformed_nodes[mesh.elements]
    pc = PolyCollection(quads, array=vm / 1e6, cmap='viridis', edgecolors='0.2', linewidths=0.3)
    ax1.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax1, shrink=0.8)
    cb.set_label('von Mises stress (MPa)')
    ax1.set_xlim(-0.5, L + 0.5)
    ax1.set_ylim(-1.5, H + 2.5)
    ax1.set_aspect('equal')
    ax1.set_title(f'Deformed shape ({mag:.0f}x magnified) -- Quad4PlaneStress, {len(mesh.elements)} elements')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')

    # ---- Panel 2: mesh-refinement convergence vs Euler-Bernoulli ----
    I = H**3 / 12.0
    delta_analytic = P_TOTAL * L**3 / (3 * STEEL.E * I)
    nx_list = [4, 8, 16, 32, 64, 128]
    fe_defl = []
    for nx in nx_list:
        ny = max(2, round(nx * H / L))
        _, fs_i, _, U_i = solve_cantilever(nx, ny)
        mesh_i = fs_i.mesh
        tip_i = fs_i._global_dofs(mesh_i.nodes_on_line(axis=0, value=L))[1]
        fe_defl.append(U_i[tip_i].mean())
    fe_defl = np.array(fe_defl)

    ax2.axhline(delta_analytic, color='k', linestyle='--', linewidth=1,
                label=f'Euler-Bernoulli: {delta_analytic*1000:.3f} mm')
    ax2.plot(nx_list, fe_defl * 1000, 'o-', color='steelblue', markersize=6,
              label='FE tip deflection (Quad4PlaneStress)')
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('elements along length (nx)')
    ax2.set_ylabel('tip deflection (mm)')
    ax2.set_title('Convergence toward beam-theory prediction')
    ax2.grid(True, alpha=0.35)
    ax2.legend()

    fig.suptitle('Cantilever Beam -- linear elasticity (fea_engine)', fontsize=13)
    fig.savefig('fea_01_cantilever_beam.png', dpi=150)
    print("Saved fea_01_cantilever_beam.png")
    print(f"Final FE tip deflection: {fe_defl[-1]*1000:.4f} mm  vs  analytical: {delta_analytic*1000:.4f} mm"
          f"  (rel. diff {abs(fe_defl[-1]-delta_analytic)/abs(delta_analytic)*100:.2f}%)")


if __name__ == "__main__":
    main()
