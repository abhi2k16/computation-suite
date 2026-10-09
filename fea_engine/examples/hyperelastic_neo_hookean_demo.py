# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
hyperelastic_neo_hookean_demo.py -- Phase 6 (general-purpose
extensions roadmap) worked example: Tet4NeoHookean (compressible
Neo-Hookean hyperelasticity, large-strain Total Lagrangian).

Two demonstrations: (1) the small-strain limit reducing to
D_solid3d()'s linear-elastic tangent, with the relative error
shrinking as strain shrinks (the roadmap's headline hyperelasticity
requirement); (2) a full incremental Newton-Raphson solve of a single
element pulled by a corner force, checked via total-potential-energy
conservation (external work = stored strain energy) -- a strong,
independent physical check that the FINITE-DIFFERENCE tangent_
stiffness() (see Tet4NeoHookean's docstring for why it's FD, not a
hand-derived closed form) is good enough for Newton to actually reach
the correct equilibrium, not just self-consistent in isolation.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from fea_engine import Tet4NeoHookean, NeoHookeanMaterial, FESystem, D_solid3d, Material
from fea_engine.mesh import Mesh
from fea_engine.material import neo_hookean_pk2_stress
from fea_engine import nonlinear_solver as nls


E, NU = 1.0e6, 0.4   # rubber-like


def main():
    mat = NeoHookeanMaterial(E=E, nu=NU)
    D_el = D_solid3d(Material(E=E, nu=NU))

    print("=" * 72)
    print(f"Tet4NeoHookean: E={E:.1e}, nu={NU}")
    print("=" * 72)
    print("\n1) Small-strain limit -> linear elasticity:")
    print(f"{'strain':>10} {'S11 (Neo-Hookean)':>20} {'sigma11 (linear)':>18} {'rel err':>10}")
    for eps_tiny in (1e-2, 1e-3, 1e-4, 1e-5):
        F = np.eye(3) + eps_tiny * np.diag([1.0, -NU, -NU])
        S, _ = neo_hookean_pk2_stress(F, mat)
        sigma_lin = D_el[0] @ np.array([eps_tiny, -NU * eps_tiny, -NU * eps_tiny, 0, 0, 0])
        rel = abs(S[0, 0] - sigma_lin) / abs(sigma_lin)
        print(f"{eps_tiny:10.0e} {S[0,0]:20.6e} {sigma_lin:18.6e} {rel:10.2e}")
    print("(relative error shrinks ~linearly with strain -- the nonlinear")
    print(" correction vanishes proportionally, as required for a correct")
    print(" small-strain limit, not just 'gets smaller somehow')")

    print("\n2) Full nonlinear solve + energy conservation check:")
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3]])
    mesh = Mesh(nodes=coords, elements=elements, dim=3)
    elem = Tet4NeoHookean()
    fs = FESystem(mesh, elem, sparse=False)
    fs.fix_dofs([0, 1, 2], [0, 1, 2])
    F_applied_total = np.array([5e4, 3e4, -2e4])
    fs.F[9:12] = F_applied_total

    n_steps = 50
    load_factors, U_hist = nls.solve_nonlinear_static(fs, mat, n_steps=n_steps, tol=1e-10, max_iter=60)
    print(f"   Converged all {n_steps} incremental Newton-Raphson steps.")

    u3_hist = U_hist[:, 9:12]
    work = 0.0
    for i in range(1, len(load_factors)):
        F_avg = 0.5 * (load_factors[i] + load_factors[i - 1]) * F_applied_total
        work += F_avg @ (u3_hist[i] - u3_hist[i - 1])

    u_final = U_hist[-1]
    F_def, _, vol = elem._deformation_gradient(mesh.nodes, u_final)
    _, W = neo_hookean_pk2_stress(F_def, mat)
    energy_stored = vol * W

    print(f"   Final tip displacement: {u3_hist[-1]}")
    print(f"   External work (path integral of F.du): {work:.6f} J")
    print(f"   Stored strain energy V0*W(F_final):     {energy_stored:.6f} J")
    print(f"   Relative difference: {abs(work-energy_stored)/abs(energy_stored):.2e}")
    print("   (a conservative system: work supplied must equal energy stored,")
    print("    computed two completely independent ways)")


if __name__ == "__main__":
    main()
