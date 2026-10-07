"""
fea_02_hyperelastic_neo_hookean.py -- Example Gallery: large-strain
Neo-Hookean hyperelasticity (Tet4NeoHookean), mirroring TensorMesh's
Example Gallery / Solid Mechanics / Hyperelastic Beam page.

Single Tet4NeoHookean element (this package's only hyperelastic
element -- 3-D solid only, no multi-tet mesh generator exists in this
codebase, so a single element is the SAME scope the package's own
existing validation demo (examples/hyperelastic_neo_hookean_demo.py)
already uses -- built here for the visual, not as new physics).

Panel 1: 3-D wireframe of the undeformed vs. large-strain deformed
tetrahedron.
Panel 2: load-deflection curve -- the Neo-Hookean incremental
Newton-Raphson solve (solve_nonlinear_static) vs. a ONE-SHOT LINEAR
solve of the SAME tet/material via the ordinary Tet4Solid3D element
(D_solid3d), scaled by load factor -- the two must agree at small
load and visibly diverge as the load grows, the real, honest
"hyperelastic vs. linear" signature (not asserted, shown).
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from fea_engine import (Tet4NeoHookean, Tet4Solid3D, NeoHookeanMaterial,
                         FESystem, D_solid3d, Material)
from fea_engine.mesh import Mesh
from fea_engine import nonlinear_solver as nls

E, NU = 1.0e6, 0.4   # rubber-like
COORDS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
CONN = np.array([[0, 1, 2, 3]])
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
F_TOTAL = np.array([1.3e5, 9e4, -6.5e4])   # corner load on node 3


def main():
    mat = NeoHookeanMaterial(E=E, nu=NU)
    mesh = Mesh(nodes=COORDS, elements=CONN, dim=3)

    # ---- Nonlinear (Neo-Hookean) incremental solve ----
    elem_nh = Tet4NeoHookean()
    fs_nh = FESystem(mesh, elem_nh, sparse=False)
    fs_nh.fix_dofs([0, 1, 2], [0, 1, 2])
    fs_nh.F[9:12] = F_TOTAL
    n_steps = 40
    load_factors, U_hist = nls.solve_nonlinear_static(fs_nh, mat, n_steps=n_steps, tol=1e-10, max_iter=60)
    tip_nh = U_hist[:, 9:12]

    # ---- Linear-elastic reference (same tet, same E/nu, load-proportional) ----
    D_lin = D_solid3d(Material(E=E, nu=NU))
    elem_lin = Tet4Solid3D()
    fs_lin = FESystem(mesh, elem_lin, sparse=False)
    fs_lin.assemble_stiffness(D_lin)
    fs_lin.fix_dofs([0, 1, 2], [0, 1, 2])
    fs_lin.F[9:12] = F_TOTAL
    U_lin_full = fs_lin.solve_static()
    tip_lin_full = U_lin_full[9:12]
    tip_lin = np.outer(load_factors, tip_lin_full)   # linear => proportional to load

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    # ---- Panel 1: undeformed vs deformed tet ----
    u_final = U_hist[-1].reshape(4, 3)
    deformed = COORDS + u_final
    lc_u = Line3DCollection([[COORDS[i], COORDS[j]] for i, j in EDGES],
                             colors='0.7', linestyles='dashed', linewidths=1.2)
    lc_d = Line3DCollection([[deformed[i], deformed[j]] for i, j in EDGES],
                             colors='tomato', linewidths=2.2)
    ax1.add_collection3d(lc_u)
    ax1.add_collection3d(lc_d)
    ax1.scatter(*COORDS.T, color='0.5', s=25)
    ax1.scatter(*deformed.T, color='tomato', s=35)
    all_pts = np.vstack([COORDS, deformed])
    for i, setlim in enumerate([ax1.set_xlim, ax1.set_ylim, ax1.set_zlim]):
        setlim(all_pts[:, i].min() - 0.2, all_pts[:, i].max() + 0.2)
    ax1.set_title('Undeformed (gray) vs. deformed (red)\nsingle Tet4NeoHookean, full load')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')

    # ---- Panel 2: load-deflection curve, nonlinear vs linear ----
    disp_mag_nh = np.linalg.norm(tip_nh, axis=1)
    disp_mag_lin = np.linalg.norm(tip_lin, axis=1)
    ax2.plot(load_factors, disp_mag_nh, 'o-', color='tomato', markersize=4,
              label='Neo-Hookean (nonlinear, Tet4NeoHookean)')
    ax2.plot(load_factors, disp_mag_lin, '--', color='steelblue', linewidth=2,
              label='linear elastic (Tet4Solid3D, load-proportional)')
    ax2.set_xlabel(r'load factor $\lambda$')
    ax2.set_ylabel('corner-node displacement magnitude (m)')
    ax2.set_title('Load-deflection: hyperelastic vs. linear reference')
    ax2.grid(True, alpha=0.35)
    ax2.legend()

    fig.suptitle('Hyperelastic Beam -- Neo-Hookean large-strain solid (fea_engine)', fontsize=13)
    fig.savefig('fea_02_hyperelastic_neo_hookean.png', dpi=150)
    print("Saved fea_02_hyperelastic_neo_hookean.png")
    rel = abs(disp_mag_nh[-1] - disp_mag_lin[-1]) / disp_mag_lin[-1]
    print(f"At full load: nonlinear disp = {disp_mag_nh[-1]:.4f} m, "
          f"linear disp = {disp_mag_lin[-1]:.4f} m ({rel*100:.1f}% apart)")


if __name__ == "__main__":
    main()
