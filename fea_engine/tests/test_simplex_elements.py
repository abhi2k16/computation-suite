__author__ = "Abhijeet"

import numpy as np
from fea_engine import elements as elmod
from fea_engine.material import Material, D_plane_stress, D_solid3d



def test_simplex_elements():
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 70)
    print("CHECK 1: Tri3PlaneStress -- patch test (exact for ANY linear disp field)")
    print("=" * 70)
    mat = Material(E=200e9, nu=0.3)
    D2 = D_plane_stress(mat)
    tri = elmod.Tri3PlaneStress()

    # an arbitrary (non-right, irregular) triangle
    elem_coords = np.array([[0.2, 0.1], [1.3, 0.4], [0.6, 1.1]])

    # impose a KNOWN, arbitrary uniform strain field via nodal displacements
    # u = eps_xx*x + 0.5*gamma_xy*y ; v = eps_yy*y + 0.5*gamma_xy*x
    eps_xx, eps_yy, gamma_xy = 0.002, -0.0015, 0.003
    u_elem = np.zeros(6)
    for i, (x, y) in enumerate(elem_coords):
        u_elem[2 * i] = eps_xx * x + 0.5 * gamma_xy * y
        u_elem[2 * i + 1] = eps_yy * y + 0.5 * gamma_xy * x

    B, detJ = tri.B_matrix((1 / 3, 1 / 3), elem_coords)
    strain_fe = B @ u_elem
    strain_exact = np.array([eps_xx, eps_yy, gamma_xy])
    err1 = np.max(np.abs(strain_fe - strain_exact))
    print(f"  strain_fe    = {strain_fe}")
    print(f"  strain_exact = {strain_exact}")
    print(f"  max abs err  = {err1:.3e}")
    assert err1 < 1e-12
    print("  PASS -- constant-strain element reproduces an imposed uniform strain exactly.")

    print()
    print("=" * 70)
    print("CHECK 2: Tri3PlaneStress stiffness() vs an independently hand-coded ke")
    print("=" * 70)
    (x1, y1), (x2, y2), (x3, y3) = elem_coords
    A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    B_hand = np.zeros((3, 6))
    for i in range(3):
        B_hand[0, 2 * i] = b[i] / (2 * A)
        B_hand[1, 2 * i + 1] = c[i] / (2 * A)
        B_hand[2, 2 * i] = c[i] / (2 * A)
        B_hand[2, 2 * i + 1] = b[i] / (2 * A)
    ke_hand = B_hand.T @ D2 @ B_hand * A * 1.0
    ke_fe = tri.stiffness(elem_coords, D2, thickness=1.0)
    err2 = np.max(np.abs(ke_hand - ke_fe)) / np.max(np.abs(ke_hand))
    print(f"  max relative error vs hand formula: {err2:.3e}")
    assert err2 < 1e-10
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 3: Tri3PlaneStress consistent + lumped mass -- total mass identity")
    print("=" * 70)
    rho = 7850.0
    thickness = 0.01
    M = tri.mass(elem_coords, rho, thickness)
    total_mass_exact = rho * A * thickness
    # CONSISTENT mass: total mass per direction = sum of the FULL x-x (or y-y)
    # block, not just its diagonal -- integral(N_i)*integral(N_j) over the
    # whole element sums to Area via partition of unity (sum_i N_i = 1).
    x_block_sum = M[0::2, 0::2].sum()
    y_block_sum = M[1::2, 1::2].sum()
    print(f"  consistent mass: x-block sum={x_block_sum:.6f}  y-block sum={y_block_sum:.6f}  "
          f"exact rho*A*t={total_mass_exact:.6f}")
    assert abs(x_block_sum - total_mass_exact) < 1e-9
    assert abs(y_block_sum - total_mass_exact) < 1e-9

    # LUMPED mass: HRZ rescaling should make the DIAGONAL sum per direction
    # exactly rho*A*t too (checked directly, not assumed).
    Mlump = tri.lumped_mass(elem_coords, rho, thickness)
    diag_lump = np.diag(Mlump)
    x_diag_sum, y_diag_sum = diag_lump[0::2].sum(), diag_lump[1::2].sum()
    print(f"  lumped mass:     x-diag sum={x_diag_sum:.6f}  y-diag sum={y_diag_sum:.6f}")
    assert abs(x_diag_sum - total_mass_exact) < 1e-9
    assert abs(y_diag_sum - total_mass_exact) < 1e-9
    print("  PASS -- consistent AND lumped mass both conserve total mass exactly, per direction.")

    print()
    print("=" * 70)
    print("CHECK 4: Tet4Solid3D -- patch test (exact for ANY linear disp field)")
    print("=" * 70)
    D3 = D_solid3d(mat)
    tet = elmod.Tet4Solid3D()
    elem_coords3 = np.array([[0.1, 0.0, 0.2], [1.2, 0.1, 0.0], [0.3, 1.1, 0.1], [0.2, 0.2, 1.3]])

    exx, eyy, ezz = 0.001, -0.0008, 0.0012
    gxy, gyz, gzx = 0.0015, -0.0011, 0.0009
    u_elem3 = np.zeros(12)
    for i, (x, y, z) in enumerate(elem_coords3):
        u_elem3[3 * i + 0] = exx * x + 0.5 * gxy * y + 0.5 * gzx * z
        u_elem3[3 * i + 1] = eyy * y + 0.5 * gxy * x + 0.5 * gyz * z
        u_elem3[3 * i + 2] = ezz * z + 0.5 * gyz * y + 0.5 * gzx * x

    B3, detJ3 = tet.B_matrix((0.25, 0.25, 0.25), elem_coords3)
    strain_fe3 = B3 @ u_elem3
    strain_exact3 = np.array([exx, eyy, ezz, gxy, gyz, gzx])
    err4 = np.max(np.abs(strain_fe3 - strain_exact3))
    print(f"  strain_fe    = {strain_fe3}")
    print(f"  strain_exact = {strain_exact3}")
    print(f"  max abs err  = {err4:.3e}")
    assert err4 < 1e-11
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 5: Tet4Solid3D stiffness() vs an independently hand-coded ke")
    print("=" * 70)
    X = elem_coords3
    vol = abs(np.linalg.det(X[1:] - X[0])) / 6.0
    # hand-coded gradient via Cramer's-rule-free direct linear solve (independent
    # of B_matrix()'s Jacobian-based path, using explicit coefficient formulas)
    Mmat = np.hstack([np.ones((4, 1)), X])   # [1, x, y, z] per node
    coeffs = np.linalg.inv(Mmat)             # rows -> a_i,b_i,c_i,d_i s.t. N_i = a_i+b_i x+c_i y+d_i z
    grads = coeffs[1:, :]                    # (3,4): d N_i/dx, dy, dz
    B3_hand = np.zeros((6, 12))
    for i in range(4):
        dx, dy, dz = grads[:, i]
        c0 = 3 * i
        B3_hand[0, c0] = dx
        B3_hand[1, c0 + 1] = dy
        B3_hand[2, c0 + 2] = dz
        B3_hand[3, c0] = dy; B3_hand[3, c0 + 1] = dx
        B3_hand[4, c0 + 1] = dz; B3_hand[4, c0 + 2] = dy
        B3_hand[5, c0] = dz; B3_hand[5, c0 + 2] = dx
    ke_hand3 = B3_hand.T @ D3 @ B3_hand * vol
    ke_fe3 = tet.stiffness(elem_coords3, D3, thickness=1.0)
    err5 = np.max(np.abs(ke_hand3 - ke_fe3)) / np.max(np.abs(ke_hand3))
    print(f"  max relative error vs hand formula (independent gradient derivation): {err5:.3e}")
    assert err5 < 1e-8
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 6: Tet4Solid3D consistent + lumped mass -- total mass identity")
    print("=" * 70)
    rho3 = 2700.0
    M3 = tet.mass(elem_coords3, rho3, 1.0)
    total_mass_exact3 = rho3 * vol
    block_sums3 = np.array([M3[k::3, k::3].sum() for k in range(3)])
    print(f"  consistent mass: block sums (x,y,z)={block_sums3}  exact rho*V={total_mass_exact3:.6f}")
    assert np.max(np.abs(block_sums3 - total_mass_exact3)) < 1e-9

    Mlump3 = tet.lumped_mass(elem_coords3, rho3, 1.0)
    diag_lump3 = np.diag(Mlump3)
    diag_sums3 = np.array([diag_lump3[k::3].sum() for k in range(3)])
    print(f"  lumped mass:     diag sums (x,y,z)={diag_sums3}")
    assert np.max(np.abs(diag_sums3 - total_mass_exact3)) < 1e-9
    print("  PASS -- consistent AND lumped mass both conserve total mass exactly, per direction.")

    print()
    print("ALL CHECKS PASSED")
