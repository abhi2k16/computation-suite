"""
beam3d_space_frame_demo.py -- Module 18 (general-purpose extensions
roadmap Phase 3): Beam3DEulerBernoulli demo.

Builds a small L-shaped space frame (one member along global X, one
along global Z, rigid joint between them), fixes the base, applies a
transverse tip load, and solves for the tip deflection -- a minimal,
readable version of the same problem test_beam3d.py's
test_full_3d_frame_matches_independent_assembler checks bit-for-bit
against an independent reference assembler. Also prints the flat-plane
reduction check (this element's bending block matches
Beam2DEulerBernoulli exactly) to make that guarantee visible without
reading the test file.
"""
__author__ = "Abhijeet"
import numpy as np

from fea_engine import (Beam3DEulerBernoulli, Beam2DEulerBernoulli, Material,
                         Section3D, beam3d_rigidities, beam3d_mass_props, FESystem)
from fea_engine.mesh import Mesh


def main():
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    sec = Section3D(A=0.01, Iy=8e-6, Iz=6e-6, J=5e-6)
    rig = beam3d_rigidities(mat, sec)
    EA, GJ, EIy, EIz = rig

    print("=== Beam3DEulerBernoulli: L-shaped space frame ===")
    print(f"Section rigidities: EA={EA:.3e}, GJ={GJ:.3e}, EIy={EIy:.3e}, EIz={EIz:.3e}\n")

    # Node 0: fixed base. Node 1: corner (rigid joint). Node 2: free tip.
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype=float)
    elements = np.array([[0, 1], [1, 2]])
    mesh = Mesh(nodes=nodes, elements=elements, dim=1)
    beam = Beam3DEulerBernoulli()
    fs = FESystem(mesh, beam, sparse=False)
    fs.assemble_stiffness(rig)

    ndof = nodes.shape[0] * 6
    fixed = list(range(0, 6))
    free_dofs = [i for i in range(ndof) if i not in fixed]
    F = np.zeros(ndof)
    F[6 * 2 + 1] = 1000.0   # 1000 N, global Y, at the free tip
    Kff = fs.K[np.ix_(free_dofs, free_dofs)]
    uf = np.linalg.solve(Kff, F[free_dofs])
    u_full = np.zeros(ndof)
    u_full[free_dofs] = uf
    tip = u_full[6 * 2:6 * 2 + 3]

    print("2-member frame: (0,0,0) -[member1: +X]-> (1,0,0) -[member2: +Z]-> (1,0,1)")
    print("Fixed at (0,0,0); 1000 N global-Y load at the free tip (1,0,1).")
    print(f"Tip displacement (x,y,z): {tip}\n")

    # Flat-plane reduction, made visible directly (not just asserted in a test):
    # a single beam along global X reduces exactly to Beam2DEulerBernoulli's
    # own (v, theta_z) bending block once its local-to-global sign convention
    # is accounted for.
    L = 2.0
    coords = np.array([[0, 0, 0], [L, 0, 0]], dtype=float)
    mprops = beam3d_mass_props(mat, sec)
    ke = beam.stiffness(coords, rig)
    b2 = Beam2DEulerBernoulli()
    k2 = b2.stiffness(np.array([[0.0], [L]]), EIz)
    idx = [2, 4, 8, 10]
    S = np.diag([1, -1, 1, -1])
    k_extracted = S @ ke[np.ix_(idx, idx)] @ S
    print("Flat-plane reduction check (Beam3D -> Beam2D, same EIz):")
    print(f"  max |Beam3D block - Beam2D.stiffness()| = {np.abs(k_extracted - k2).max():.3e}"
          f" (exact 0.0 expected: same closed-form Hermite formula)")


if __name__ == "__main__":
    main()
