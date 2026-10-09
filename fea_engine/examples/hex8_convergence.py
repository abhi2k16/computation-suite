# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
hex8_convergence.py -- mesh refinement study for item 4 in main.py.

Fixes the beam geometry/load/BCs used in main.py's Hex8Solid3D case and
sweeps the number of elements through the cross-section (ny=nz) to show
how the shear-locking error (FEM/Euler-Bernoulli ratio) shrinks as the
through-thickness mesh is refined. Axial element count (nx) is also
scaled up alongside ny/nz so element aspect ratio stays roughly constant
across the sweep (otherwise refining only ny/nz while holding nx fixed
makes elements more slender in x, which is a different effect).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import matplotlib.pyplot as plt

from fea_engine.material import Material, D_solid3d
from fea_engine.mesh import box_mesh
from fea_engine.elements import Hex8Solid3D
from fea_engine.solver import FESystem

steel = Material(E=2.1e11, nu=0.3, rho=7850.0)

L3, h_cs, w_cs = 1.0, 0.05, 0.05
F_tip3 = -1000.0
I3 = w_cs * h_cs**3 / 12
w_tip3_EB = F_tip3 * L3**3 / (3 * steel.E * I3)

through_thickness_counts = [1, 2, 3, 4, 6, 8, 12]
ratios = []
ndofs = []

elem4 = Hex8Solid3D()
for n_tt in through_thickness_counts:
    nx3 = 5 * n_tt  # keep axial:through-thickness element aspect ratio ~constant
    mesh4 = box_mesh(L3, w_cs, h_cs, nx3, n_tt, n_tt)
    sys4 = FESystem(mesh4, elem4)
    sys4.assemble_stiffness(D_solid3d(steel))
    tip_nodes3 = mesh4.nodes_on_plane(axis=0, value=L3)
    sys4.add_nodal_force(tip_nodes3, dof_index=2, total_force=F_tip3)
    fixed_nodes3 = mesh4.nodes_on_plane(axis=0, value=0.0)
    sys4.fix_dofs(fixed_nodes3, [0, 1, 2])
    U4 = sys4.solve_static()
    tip_center3 = tip_nodes3[np.argmin(np.abs(mesh4.nodes[tip_nodes3, 1] - w_cs / 2))]
    w_tip3_fem = U4[3 * tip_center3 + 2]
    ratio = w_tip3_fem / w_tip3_EB
    ratios.append(ratio)
    ndofs.append(mesh4.nodes.shape[0] * 3)
    print(f"  through-thickness n={n_tt:2d} (nx={nx3:3d}, {mesh4.nodes.shape[0]:5d} nodes): "
          f"ratio = {ratio:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(through_thickness_counts, ratios, 'o-', color='blue', label='Hex8Solid3D (full 2x2x2 integration)')
ax.axhline(1.0, color='k', linestyle='--', linewidth=1, label='Euler-Bernoulli (exact, ratio=1)')
ax.set_xlabel('elements through cross-section thickness (ny = nz)')
ax.set_ylabel('FEM / Euler-Bernoulli tip deflection ratio')
ax.set_title('Hex8Solid3D shear-locking: convergence with through-thickness refinement')
ax.set_ylim(0.6, 1.05)
ax.grid(True, alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig('hex8_convergence.png', dpi=150)
print("\nSaved hex8_convergence.png")
