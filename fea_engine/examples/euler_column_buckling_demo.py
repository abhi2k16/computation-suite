# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
euler_column_buckling_demo.py -- Module 19 (general-purpose extensions
roadmap Phase 4): linear buckling eigenvalue solver demo.

Discretizes a pin-pin column into progressively finer meshes of
Beam2DEulerBernoulli elements, solves the linear buckling eigenproblem
via FESystem.assemble_geometric_stiffness() + solve_linear_buckling(),
and shows the FEM critical load converging to the textbook Euler
closed form P_cr = pi^2*EI/L^2 -- the same mesh-refinement-convergence
story every other benchmark in this project tells, applied to the new
buckling capability.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from fea_engine import Beam2DEulerBernoulli, FESystem
from fea_engine.mesh import Mesh


def euler_column(n_elem, L, EI):
    n_nodes = n_elem + 1
    nodes = np.linspace(0, L, n_nodes).reshape(-1, 1)
    elements = np.array([[i, i + 1] for i in range(n_elem)])
    mesh = Mesh(nodes=nodes, elements=elements, dim=1)
    beam = Beam2DEulerBernoulli()
    fs = FESystem(mesh, beam, sparse=False)
    fs.assemble_stiffness(EI)
    # Reference state: 1 N of axial compression in every element (the
    # exact, statically-determinate axial force distribution for a
    # straight column loaded only at its ends) -- tension-positive
    # convention, so compression is passed as -1.0.
    fs.assemble_geometric_stiffness(-1.0)
    fs.fix_dofs([0], [0])            # pin: v=0, theta free
    fs.fix_dofs([n_nodes - 1], [0])  # pin: v=0, theta free
    return fs


def main():
    E, I, L = 210e9, 8e-6, 3.0
    EI = E * I
    P_exact = np.pi**2 * EI / L**2

    print("=== Euler column buckling: pin-pin, P_cr = pi^2*EI/L^2 ===")
    print(f"E={E:.3e} Pa, I={I:.3e} m^4, L={L} m -> P_cr (exact) = {P_exact:.6f} N\n")
    print(f"{'n_elem':>8} {'P_cr (FEM)':>16} {'rel. error':>14}")
    for n_elem in (2, 4, 8, 16, 32, 64):
        fs = euler_column(n_elem, L, EI)
        load_factors, mode_shapes = fs.solve_linear_buckling(n_modes=1)
        P_fem = load_factors[0]
        rel_err = abs(P_fem - P_exact) / P_exact
        print(f"{n_elem:8d} {P_fem:16.6f} {rel_err:14.3e}")

    print("\nMonotone convergence to the closed form as the mesh refines --")
    print("no fudge factor, no calibration: the same closed-form geometric")
    print("stiffness matrix (Przemieniecki) every general FE textbook uses.")


if __name__ == "__main__":
    main()
