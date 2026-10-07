"""
test_higher_order_elements.py -- validates the Phase 1 (general-purpose
extensions roadmap) higher-order elements: Quad8PlaneStress,
Hex20Solid3D, Tet10Solid3D (docs/general_purpose_extensions_roadmap.md
Section 3).

Same validation depth as the existing Tri3PlaneStress/Tet4Solid3D
checks (test_simplex_elements.py): isoparametric-identity + exact
constant-strain patch test on a single element, rigid-body-mode count,
consistent-mass total. Quad8 additionally gets a genuine TWO-ELEMENT
interior-node patch test (see CHECK 5) -- Hex20/Tet10 do NOT get the
tetrahedral/hexahedral equivalent here, and that is a deliberate,
documented scoping decision, not an oversight: for a serendipity
solid element with no interior/bubble nodes, a node is only truly
"interior" (shape-function support entirely away from any outer
boundary face) if enough SURROUNDING elements exist that none of them
contribute an outer face at that node -- a minimal 2-element mesh
(the simplest assembly that exists at all) can't provide that for a
tet or a hex, since every node of a 2-tet/2-hex assembly is still a
vertex of at least one outer face. (This was discovered by actually
attempting the test: a first version left the shared-face nodes free,
and the assembled internal force was zero at the shared CORNER nodes
but large and nonzero at the shared MID-EDGE nodes -- tracked down to
exactly this geometric fact via manual-vs-assembled force comparison,
single-element re-verification, and a literature-grounded shape-
function-support argument, not a bug in the elements: manual scatter-
assembly matched FESystem's own assembly exactly, and each element
individually reproduced the imposed field's strain exactly.) Quad8
doesn't have this problem: a quadrilateral's edge-midpoint shape
function has a (1-xi^2)-type factor that vanishes identically on the
element's OTHER three edges, so a shared edge's midpoint really is
interior after just two elements -- confirmed by CHECK 5 passing at
literal machine precision (~1e-19).

CHECK 6 is the headline general-purpose demonstration the roadmap
itself called for: a coarse (1-element-through-the-height) cantilever
under a PURE bending moment -- the classic textbook case where a
single row of Quad4 elements is well known to shear-lock (parasitic
shear strain makes it artificially stiff), while quadratic elements
capture pure bending almost exactly. Pure bending is used specifically
because sigma_xx=M*y/I, sigma_yy=tau_xy=0 is an EXACT elasticity
solution for a prismatic beam (not just a beam-theory approximation),
so the target isn't an engineering estimate -- it's mathematically
exact, making this a decisive, not just qualitative, comparison.
"""
import numpy as np
from fea_engine import (
    FESystem, Material, D_plane_stress, D_solid3d,
    Quad4PlaneStress, Quad8PlaneStress, Hex20Solid3D, Tet10Solid3D, mesh,
)


# ---------------------------------------------------------------------
# Shared reference-element node coordinates (natural == physical, so
# the isoparametric map is the identity -- the simplest possible
# geometry for isolating shape-function correctness from Jacobian
# bookkeeping).
# ---------------------------------------------------------------------
_REF_Q8 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1],
                     [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float)
_REF_H20 = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    [0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1],
    [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
    [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
_REF_T10 = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0], [0, 0, 0.5],
    [0.5, 0, 0.5], [0, 0.5, 0.5]], dtype=float)


def test_quad8_plane_stress():
    np.set_printoptions(precision=6, suppress=True)
    q8 = Quad8PlaneStress()
    mat = Material(E=200e9, nu=0.3)
    D2 = D_plane_stress(mat)

    print("=" * 70)
    print("CHECK 1: Quad8 partition of unity + isoparametric identity")
    print("=" * 70)
    for p in [(0, 0), (0.4, -0.6), (-0.9, 0.3), (1, 1), (-1, -1)]:
        N, dN = q8.shape_and_derivs(p)
        assert abs(N.sum() - 1.0) < 1e-12
        assert np.allclose(dN.sum(axis=1), 0.0, atol=1e-12)
        x = N @ _REF_Q8
        assert np.allclose(x, p, atol=1e-12)
    print("  PASS -- sum(N)=1, sum(dN)=0, and the reference map is the identity"
          " at every sampled point.")

    print()
    print("=" * 70)
    print("CHECK 2: Quad8 exact constant-strain patch test")
    print("=" * 70)
    eps_xx, eps_yy, gamma_xy = 0.002, -0.0015, 0.003
    u_elem = np.zeros(16)
    for i, (x, y) in enumerate(_REF_Q8):
        u_elem[2 * i] = eps_xx * x + 0.5 * gamma_xy * y
        u_elem[2 * i + 1] = eps_yy * y + 0.5 * gamma_xy * x
    expected = np.array([eps_xx, eps_yy, gamma_xy])
    for p in [(0, 0), (0.4, -0.6), (-0.9, 0.3), (1, 1)]:
        B, _ = q8.B_matrix(p, _REF_Q8)
        strain = B @ u_elem
        assert np.allclose(strain, expected, atol=1e-10)
    print(f"  strain reproduced exactly at every sampled point: {expected}")
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 3: Quad8 rigid-body-mode count (3 in 2-D: 2 translations + 1 rotation)")
    print("=" * 70)
    ke = q8.full_stiffness(_REF_Q8, D2, 1.0)
    assert np.allclose(ke, ke.T, atol=1e-6 * np.abs(ke).max())
    eig = np.linalg.eigvalsh(ke)
    n_zero = int(np.sum(np.abs(eig) < 1e-6 * np.abs(eig).max()))
    print(f"  n_zero_eig = {n_zero} (expect 3); min/max eig = {eig[0]:.3e}/{eig[-1]:.3e}")
    assert n_zero == 3
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 4: Quad8 consistent mass -- total mass identity")
    print("=" * 70)
    rho = 7800.0
    me = q8.mass(_REF_Q8, rho * np.eye(2), thickness=1.0)
    total = me[0::2, 0::2].sum()
    expected_total = rho * 4.0   # area of the [-1,1]^2 reference square
    print(f"  total mass (x-block sum) = {total:.6f}, expected = {expected_total:.6f}")
    assert np.isclose(total, expected_total, rtol=1e-8)
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 5: Quad8 TWO-ELEMENT interior-node patch test (via FESystem)")
    print("=" * 70)
    # Two unit-square Quad8 elements side by side, spanning [0,2]x[0,1] --
    # node 5 (physical (1, 0.5), the midpoint of the shared edge) is the
    # ONLY node not on the outer boundary of the combined domain.
    nodes = np.array([
        (0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5),
        (2, 0), (2, 1), (1.5, 0), (2, 0.5), (1.5, 1)], dtype=float)
    elemA = [0, 1, 2, 3, 4, 5, 6, 7]
    elemB = [1, 8, 9, 2, 10, 11, 12, 5]
    m = mesh.Mesh(nodes=nodes, elements=np.array([elemA, elemB]), dim=2)
    sysQ = FESystem(m, q8, thickness=1.0)
    sysQ.assemble_stiffness(D2)

    def field2(x, y):
        return np.array([0.001 + 0.7 * x - 0.4 * y, -0.3 * x + 0.6 * y - 0.1]) * 1e-3

    free_nodes = [5]
    presc_nodes = [n for n in range(13) if n not in free_nodes]
    free_dofs = np.array([2 * n + k for n in free_nodes for k in range(2)])
    presc_dofs = np.array([2 * n + k for n in presc_nodes for k in range(2)])
    u_full = np.zeros(sysQ.n_dof)
    for n in presc_nodes:
        u_full[2 * n:2 * n + 2] = field2(*m.nodes[n])
    K = sysQ.K
    rhs = -K[np.ix_(free_dofs, presc_dofs)] @ u_full[presc_dofs]
    u_full[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], rhs)
    got = u_full[free_dofs]
    exact = field2(*m.nodes[5])
    err = np.max(np.abs(got - exact))
    print(f"  interior node 5: solved = {got}, exact = {exact}, max abs err = {err:.3e}")
    assert err < 1e-12
    print("  PASS -- assembled two-element system recovers the exact field at a"
          " genuinely free interior node, to machine precision.")


def test_hex20_solid3d():
    np.set_printoptions(precision=6, suppress=True)
    h20 = Hex20Solid3D()
    mat = Material(E=200e9, nu=0.3)
    D3 = D_solid3d(mat)

    print("=" * 70)
    print("CHECK 1: Hex20 partition of unity + isoparametric identity")
    print("=" * 70)
    for p in [(0, 0, 0), (0.2, -0.3, 0.5), (1, 1, 1), (-1, -1, -1), (0.5, 0, -0.5)]:
        N, dN = h20.shape_and_derivs(p)
        assert abs(N.sum() - 1.0) < 1e-12
        assert np.allclose(dN.sum(axis=1), 0.0, atol=1e-12)
        x = N @ _REF_H20
        assert np.allclose(x, p, atol=1e-10)
    print("  PASS -- sum(N)=1, sum(dN)=0, and the reference map is the identity"
          " at every sampled point.")

    print()
    print("=" * 70)
    print("CHECK 2: Hex20 exact constant-strain patch test")
    print("=" * 70)
    coefs = np.array([[0.1, 0.05, -0.02, 0.03],
                       [0.02, -0.03, 0.06, 0.01],
                       [-0.01, 0.02, -0.015, 0.04]]) * 1e-2
    u_elem = np.zeros(60)
    for i, (x, y, z) in enumerate(_REF_H20):
        for comp in range(3):
            c0, c1, c2, c3 = coefs[comp]
            u_elem[3 * i + comp] = c0 + c1 * x + c2 * y + c3 * z
    exx, eyy, ezz = coefs[0, 1], coefs[1, 2], coefs[2, 3]
    gxy = coefs[0, 2] + coefs[1, 1]
    gyz = coefs[1, 3] + coefs[2, 2]
    gzx = coefs[2, 1] + coefs[0, 3]
    expected = np.array([exx, eyy, ezz, gxy, gyz, gzx])
    for p in [(0, 0, 0), (0.3, -0.2, 0.5), (-0.8, 0.6, 0.1)]:
        B, _ = h20.B_matrix(p, _REF_H20)
        strain = B @ u_elem
        assert np.allclose(strain, expected, atol=1e-9)
    print(f"  strain reproduced exactly at every sampled point: {expected}")
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 3: Hex20 rigid-body-mode count (6 in 3-D: 3 translations + 3 rotations)")
    print("=" * 70)
    ke = h20.full_stiffness(_REF_H20, D3, 1.0)
    assert np.allclose(ke, ke.T, atol=1e-6 * np.abs(ke).max())
    eig = np.linalg.eigvalsh(ke)
    n_zero = int(np.sum(np.abs(eig) < 1e-6 * np.abs(eig).max()))
    print(f"  n_zero_eig = {n_zero} (expect 6); min/max eig = {eig[0]:.3e}/{eig[-1]:.3e}")
    assert n_zero == 6
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 4: Hex20 consistent mass -- total mass identity")
    print("=" * 70)
    rho = 7800.0
    me = h20.mass(_REF_H20, rho * np.eye(3), thickness=1.0)
    total = me[0::3, 0::3].sum()
    expected_total = rho * 8.0   # volume of the [-1,1]^3 reference cube
    print(f"  total mass (x-block sum) = {total:.6f}, expected = {expected_total:.6f}")
    assert np.isclose(total, expected_total, rtol=1e-8)
    print("  PASS")
    print()
    print("  NOTE: no multi-element interior-node patch test here -- see this")
    print("  file's module docstring for why a minimal 2-hex assembly can't")
    print("  provide a genuinely interior node for a serendipity solid element")
    print("  (a real limitation of the TEST, not the element; single-element")
    print("  checks above are the same depth the existing Tet4/Tri3 elements")
    print("  were validated to).")


def test_tet10_solid3d():
    np.set_printoptions(precision=6, suppress=True)
    t10 = Tet10Solid3D()
    mat = Material(E=200e9, nu=0.3)
    D3 = D_solid3d(mat)

    print("=" * 70)
    print("CHECK 1: Tet10 partition of unity + isoparametric identity")
    print("=" * 70)
    for p in [(0.1, 0.1, 0.1), (0.5, 0.2, 0.1), (0, 0, 0), (1, 0, 0),
              (0, 0, 1), (0.25, 0.25, 0.25)]:
        N, dN = t10.shape_and_derivs(p)
        assert abs(N.sum() - 1.0) < 1e-12
        assert np.allclose(dN.sum(axis=1), 0.0, atol=1e-12)
        x = N @ _REF_T10
        assert np.allclose(x, p, atol=1e-12)
    print("  PASS -- sum(N)=1, sum(dN)=0, and the reference map is the identity"
          " at every sampled point.")

    print()
    print("=" * 70)
    print("CHECK 2: Tet10 exact constant-strain patch test, and constant detJ")
    print("=" * 70)
    coefs = np.array([[0.1, 0.05, -0.02, 0.03],
                       [0.02, -0.03, 0.06, 0.01],
                       [-0.01, 0.02, -0.015, 0.04]]) * 1e-2
    u_elem = np.zeros(30)
    for i, (x, y, z) in enumerate(_REF_T10):
        for comp in range(3):
            c0, c1, c2, c3 = coefs[comp]
            u_elem[3 * i + comp] = c0 + c1 * x + c2 * y + c3 * z
    exx, eyy, ezz = coefs[0, 1], coefs[1, 2], coefs[2, 3]
    gxy = coefs[0, 2] + coefs[1, 1]
    gyz = coefs[1, 3] + coefs[2, 2]
    gzx = coefs[2, 1] + coefs[0, 3]
    expected = np.array([exx, eyy, ezz, gxy, gyz, gzx])
    detJs = []
    for p in [(0.1, 0.1, 0.1), (0.5, 0.2, 0.1), (0.25, 0.25, 0.25), (0.6, 0.1, 0.2)]:
        B, detJ = t10.B_matrix(p, _REF_T10)
        detJs.append(detJ)
        strain = B @ u_elem
        assert np.allclose(strain, expected, atol=1e-9)
    print(f"  strain reproduced exactly at every sampled point: {expected}")
    print(f"  detJ at 4 different natural points: {detJs} (spread = "
          f"{max(detJs) - min(detJs):.2e}) -- confirms the straight-sided "
          f"element's isoparametric map is genuinely AFFINE despite the "
          f"quadratic shape functions, exactly as tet_quadrature_4pt()'s "
          f"docstring claims")
    assert max(detJs) - min(detJs) < 1e-10
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 3: Tet10 rigid-body-mode count (6 in 3-D) + exact strain-energy check")
    print("=" * 70)
    ke = t10.stiffness(_REF_T10, D3, 1.0)
    assert np.allclose(ke, ke.T, atol=1e-6 * np.abs(ke).max())
    eig = np.linalg.eigvalsh(ke)
    n_zero = int(np.sum(np.abs(eig) < 1e-6 * np.abs(eig).max()))
    print(f"  n_zero_eig = {n_zero} (expect 6); min/max eig = {eig[0]:.3e}/{eig[-1]:.3e}")
    assert n_zero == 6

    # independent closed-form check: strain energy U = 0.5*u^T*ke*u for a
    # constant-strain field must equal Volume * 0.5*strain^T*D*strain
    # exactly -- this checks ke's absolute SCALE (not just its rank/strain
    # reconstruction), which is exactly the kind of bug the 1/6 natural-
    # tetrahedron-volume factor could get wrong (and, during development,
    # briefly did -- see tet_quadrature_4pt()'s docstring).
    strain = np.array([1e-4, -0.5e-4, 0.2e-4, 0.3e-4, -0.1e-4, 0.15e-4])
    exx, eyy, ezz, gxy, gyz, gzx = strain

    def disp(x, y, z):
        return np.array([
            exx * x + 0.5 * gxy * y + 0.5 * gzx * z,
            eyy * y + 0.5 * gxy * x + 0.5 * gyz * z,
            ezz * z + 0.5 * gyz * y + 0.5 * gzx * x])

    u2 = np.concatenate([disp(*c) for c in _REF_T10])
    U_fem = 0.5 * u2 @ ke @ u2
    V = 1.0 / 6.0
    U_exact = V * 0.5 * strain @ D3 @ strain
    print(f"  strain energy: FEM = {U_fem:.6f}, exact closed-form = {U_exact:.6f}")
    assert np.isclose(U_fem, U_exact, rtol=1e-8)
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 4: Tet10 consistent mass -- total mass identity")
    print("=" * 70)
    rho = 7800.0
    me = t10.mass(_REF_T10, rho, thickness=1.0)
    total = me[0::3, 0::3].sum()
    expected_total = rho * (1.0 / 6.0)   # volume of the reference tetrahedron
    print(f"  total mass (x-block sum) = {total:.6f}, expected = {expected_total:.6f}")
    assert np.isclose(total, expected_total, rtol=1e-6)
    print("  PASS")
    print()
    print("  NOTE: no multi-element interior-node patch test here -- same")
    print("  reason as Hex20 (see this file's module docstring).")


def test_quad8_vs_quad4_pure_bending_locking_comparison():
    """The headline general-purpose demonstration
    docs/general_purpose_extensions_roadmap.md Section 3 called for:
    quadratic elements should need far fewer DOF (here, literally the
    SAME coarse 2-element-long, 1-element-tall mesh) to capture a
    bending-dominated response that a linear element is known to
    shear-lock on.

    Pure bending (equal-and-opposite end forces forming a moment, zero
    net shear) is used deliberately: sigma_xx = M*y/I with sigma_yy =
    tau_xy = 0 is an EXACT elasticity solution for a prismatic beam
    (not just a beam-theory approximation), so tip deflection =
    M*L^2/(2*E*I) is a mathematically exact target for ANY sufficiently
    rich element -- making this a decisive, not just qualitative,
    comparison. nu=0 removes Poisson cross-coupling so the 2-D
    plane-stress elasticity solution matches the 1-D beam formula
    exactly, isolating the locking behavior itself."""
    np.set_printoptions(precision=6, suppress=True)
    print("=" * 70)
    print("Quad4 (shear-locked) vs Quad8 (not) under coarse pure bending")
    print("=" * 70)

    E, nu = 210e9, 0.0
    mat = Material(E=E, nu=nu)
    D2 = D_plane_stress(mat)
    L, h, t = 2.0, 1.0, 1.0
    I = t * h**3 / 12.0
    M_applied = 1.0e6
    F = M_applied / h   # end force-couple magnitude giving moment M = F*h
    w_exact = M_applied * L**2 / (2 * E * I)
    print(f"  L={L}, h={h}, I={I:.6f}, M={M_applied:.3e}, exact tip deflection "
          f"w = M*L^2/(2EI) = {w_exact:.6e}")

    # ---- Quad4: 2 elements long x 1 tall, via the existing rectangle_mesh() ----
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

    # ---- Quad8: SAME coarse layout (2 elements long x 1 tall), hand-built
    # (mesh.py has no quadratic-mesh generator yet) ----
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

    ratio4 = tip_uy4 / w_exact
    ratio8 = tip_uy8 / w_exact
    print(f"  Quad4 (8 dof/element-row): tip uy = {tip_uy4:.6e}, ratio to exact = {ratio4:.4f}")
    print(f"  Quad8 (same 2 elements):   tip uy = {tip_uy8:.6e}, ratio to exact = {ratio8:.4f}")

    # Quad4's well-known single-element-through-thickness shear-locking
    # ratio for this exact setup is 2/3 (a textbook number, e.g.
    # Cook/Malkus/Plesha) -- assert it's clearly, badly off (not just
    # "a bit less accurate"), and that Quad8 is essentially exact.
    assert abs(ratio4) < 0.75, f"expected Quad4 to be badly locked (~0.667), got {ratio4}"
    assert abs(abs(ratio8) - 1.0) < 1e-6, f"expected Quad8 to be essentially exact, got {ratio8}"
    print("  PASS -- Quad4 is badly shear-locked (~33% too stiff) at this coarse "
          "resolution; Quad8, at the IDENTICAL element count, matches the exact "
          "elasticity solution to within numerical precision.")
