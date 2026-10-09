"""
test_tri6_element.py -- validates Tri6PlaneStress (Wave 0 item 7,
docs/consolidated_future_roadmap.md, source geometry_meshing_
alternatives_research.md item 4): the quadratic-triangle plane-stress
element added for row-3 parity with the existing Quad8PlaneStress.

Same validation depth as test_higher_order_elements.py's Quad8 checks
(isoparametric-identity + partition-of-unity, exact constant-strain
patch test, rigid-body-mode count, consistent-mass total, and a genuine
TWO-ELEMENT interior-node patch test) -- Tri6's mid-edge shape
functions vanish on the two edges NOT containing that midpoint (the
same reasoning as Quad8's (1-xi^2)-type edge functions, NOT like
Tet10/Hex20 in 3-D, where a minimal 2-element mesh has no genuinely
interior node -- see that file's CHECK 5 docstring for the geometric
argument), so a shared-edge midpoint really is interior after just two
triangles.
"""
__author__ = "Abhijeet"
import numpy as np
from fea_engine import FESystem, Material, D_plane_stress, Tri6PlaneStress, mesh


# Reference element: natural coordinates used directly as physical
# coordinates, so the isoparametric map is the identity -- isolates
# shape-function correctness from Jacobian bookkeeping. Corners
# 0,1,2 at (0,0),(1,0),(0,1) (Tri3PlaneStress's own convention);
# mid-edges 3,4,5 on edges (0,1),(1,2),(2,0) respectively.
_REF_T6 = np.array([
    [0, 0], [1, 0], [0, 1],
    [0.5, 0], [0.5, 0.5], [0, 0.5]], dtype=float)


def test_tri6_plane_stress():
    np.set_printoptions(precision=6, suppress=True)
    t6 = Tri6PlaneStress()
    mat = Material(E=200e9, nu=0.3)
    D2 = D_plane_stress(mat)

    print("=" * 70)
    print("CHECK 1: Tri6 partition of unity + isoparametric identity")
    print("=" * 70)
    for p in [(1 / 3, 1 / 3), (0.2, 0.3), (0.1, 0.7), (0.6, 0.1), (0.0, 0.0)]:
        N, dN = t6.shape_and_derivs(p)
        assert abs(N.sum() - 1.0) < 1e-12
        assert np.allclose(dN.sum(axis=1), 0.0, atol=1e-12)
        x = N @ _REF_T6
        assert np.allclose(x, p, atol=1e-12)
    print("  PASS -- sum(N)=1, sum(dN)=0, and the reference map is the identity"
          " at every sampled point.")

    print()
    print("=" * 70)
    print("CHECK 2: Tri6 exact constant-strain patch test")
    print("=" * 70)
    eps_xx, eps_yy, gamma_xy = 0.002, -0.0015, 0.003
    u_elem = np.zeros(12)
    for i, (x, y) in enumerate(_REF_T6):
        u_elem[2 * i] = eps_xx * x + 0.5 * gamma_xy * y
        u_elem[2 * i + 1] = eps_yy * y + 0.5 * gamma_xy * x
    expected = np.array([eps_xx, eps_yy, gamma_xy])
    for p in [(1 / 3, 1 / 3), (0.1, 0.2), (0.7, 0.1)]:
        B, detJ = t6.B_matrix(p, _REF_T6)
        strain = B @ u_elem
        err = np.max(np.abs(strain - expected))
        print(f"  p={p}: strain={strain}, err={err:.3e}")
        assert err < 1e-12
    print("  PASS -- a linear (constant-strain) displacement field is reproduced"
          " EXACTLY at every sampled point, as required for convergence.")

    print()
    print("=" * 70)
    print("CHECK 3: Tri6 rigid-body-mode count (3 in 2-D: 2 translations + 1 rotation)")
    print("=" * 70)
    ke = t6.full_stiffness(_REF_T6, D2, 1.0)
    assert np.allclose(ke, ke.T, atol=1e-6 * np.abs(ke).max())
    eig = np.linalg.eigvalsh(ke)
    n_zero = int(np.sum(np.abs(eig) < 1e-6 * np.abs(eig).max()))
    print(f"  n_zero_eig = {n_zero} (expect 3); min/max eig = {eig[0]:.3e}/{eig[-1]:.3e}")
    assert n_zero == 3
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 4: Tri6 consistent mass -- total mass identity")
    print("=" * 70)
    rho = 7800.0
    me = t6.mass(_REF_T6, rho * np.eye(2), thickness=1.0)
    total = me[0::2, 0::2].sum()
    expected_total = rho * 0.5   # area of the natural triangle (0,0),(1,0),(0,1)
    print(f"  total mass (x-block sum) = {total:.6f}, expected = {expected_total:.6f}")
    assert np.isclose(total, expected_total, rtol=1e-8)
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 5: Tri6 TWO-ELEMENT interior-node patch test (via FESystem)")
    print("=" * 70)
    # Two Tri6 elements splitting the unit square [0,1]x[0,1] along its
    # diagonal -- node 5 (physical (0.5, 0.5), the midpoint of the
    # shared hypotenuse) is the ONLY node not on the outer boundary of
    # the combined domain.
    nodes = np.array([
        (0, 0), (1, 0), (1, 1), (0.5, 0), (1, 0.5), (0.5, 0.5),
        (0, 1), (0.5, 1), (0, 0.5)], dtype=float)
    elemA = [0, 1, 2, 3, 4, 5]           # corners (0,0),(1,0),(1,1)
    elemB = [0, 2, 6, 5, 7, 8]           # corners (0,0),(1,1),(0,1)
    m = mesh.Mesh(nodes=nodes, elements=np.array([elemA, elemB]), dim=2)
    sysT = FESystem(m, t6, thickness=1.0)
    sysT.assemble_stiffness(D2)

    def field2(x, y):
        return np.array([0.001 + 0.7 * x - 0.4 * y, -0.3 * x + 0.6 * y - 0.1]) * 1e-3

    free_nodes = [5]
    presc_nodes = [n for n in range(9) if n not in free_nodes]
    free_dofs = np.array([2 * n + k for n in free_nodes for k in range(2)])
    presc_dofs = np.array([2 * n + k for n in presc_nodes for k in range(2)])
    u_full = np.zeros(sysT.n_dof)
    for n in presc_nodes:
        u_full[2 * n:2 * n + 2] = field2(*m.nodes[n])
    K = sysT.K
    rhs = -K[np.ix_(free_dofs, presc_dofs)] @ u_full[presc_dofs]
    u_full[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], rhs)
    got = u_full[free_dofs]
    exact = field2(*m.nodes[5])
    err = np.max(np.abs(got - exact))
    print(f"  interior node 5: solved = {got}, exact = {exact}, max abs err = {err:.3e}")
    assert err < 1e-10
    print("  PASS -- assembled two-element system recovers the exact field at a"
          " genuinely free interior node, to near machine precision.")
