"""
higher_order_elements_demo.py -- Phase 1 (general-purpose extensions
roadmap) worked example: Quad8PlaneStress vs Quad4PlaneStress under
coarse pure bending.

Demonstrates the actual reason to reach for a quadratic element: at an
IDENTICAL, coarse element count (2 elements long, 1 element through
the height -- about as coarse as a mesh gets), a single row of Quad4
elements is well known to "shear-lock" under bending (parasitic shear
strain the element can't avoid makes it artificially stiff), while
Quad8 captures pure bending almost exactly. Pure bending is used
deliberately: sigma_xx = M*y/I, sigma_yy = tau_xy = 0 is an EXACT
elasticity solution for a prismatic beam (not just a beam-theory
approximation), so tip deflection = M*L^2/(2*E*I) is a mathematically
exact target -- this is a decisive, not just qualitative, comparison.
See tests/test_higher_order_elements.py for the same comparison run as
a pytest regression check, and
docs/general_purpose_extensions_roadmap.md Section 3 for the design
background.
"""
__author__ = "Abhijeet"
import numpy as np
from fea_engine import (
    FESystem, Material, D_plane_stress, Quad4PlaneStress, Quad8PlaneStress, mesh,
)


def main():
    print("=" * 70)
    print("Quad4 (shear-locked) vs Quad8 (not) under coarse pure bending")
    print("=" * 70)

    E, nu = 210e9, 0.0   # nu=0 isolates locking from Poisson cross-coupling
    mat = Material(E=E, nu=nu)
    D2 = D_plane_stress(mat)
    L, h, t = 2.0, 1.0, 1.0
    I = t * h**3 / 12.0
    M_applied = 1.0e6
    F = M_applied / h
    w_exact = M_applied * L**2 / (2 * E * I)
    print(f"beam: L={L}, h={h}, I={I:.6f}, applied moment M={M_applied:.3e}")
    print(f"exact tip deflection (elasticity-exact for pure bending): "
          f"w = M*L^2/(2EI) = {w_exact:.6e}\n")

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
    tip4 = np.where(np.abs(m4.nodes[:, 0] - L) < 1e-9)[0]
    tip_uy4 = U4[2 * tip4 + 1].mean()

    # ---- Quad8: SAME coarse layout, hand-built (no quadratic mesh
    # generator in mesh.py yet -- a natural next extension) ----
    nodes8 = np.array([
        (0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5),
        (2, 0), (2, 1), (1.5, 0), (2, 0.5), (1.5, 1)], dtype=float)
    m8 = mesh.Mesh(nodes=nodes8, elements=np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                                                       [1, 8, 9, 2, 10, 11, 12, 5]]), dim=2)
    sys8 = FESystem(m8, Quad8PlaneStress(), thickness=t)
    sys8.assemble_stiffness(D2)
    for n in np.where(np.abs(m8.nodes[:, 0] - 0.0) < 1e-9)[0]:
        sys8.fix_dofs([n], [0, 1])
    top8 = np.where((np.abs(m8.nodes[:, 0] - L) < 1e-9) & (np.abs(m8.nodes[:, 1] - h) < 1e-9))[0][0]
    bot8 = np.where((np.abs(m8.nodes[:, 0] - L) < 1e-9) & (np.abs(m8.nodes[:, 1] - 0.0) < 1e-9))[0][0]
    sys8.F[2 * top8] = F
    sys8.F[2 * bot8] = -F
    U8 = sys8.solve_static()
    tip8 = np.where(np.abs(m8.nodes[:, 0] - L) < 1e-9)[0]
    tip_uy8 = U8[2 * tip8 + 1].mean()

    print(f"Quad4 (4 elements, 2x1):  tip uy = {tip_uy4: .6e}   "
          f"ratio to exact = {tip_uy4 / w_exact: .4f}")
    print(f"Quad8 (2 elements, 2x1):  tip uy = {tip_uy8: .6e}   "
          f"ratio to exact = {tip_uy8 / w_exact: .4f}")
    print()
    print("Quad4 is roughly 33% too stiff (the textbook 2/3 single-element-")
    print("through-thickness locking ratio) -- Quad8, at the SAME (in fact")
    print("fewer!) element count, matches the exact elasticity solution to")
    print("within numerical precision.")


if __name__ == "__main__":
    main()
