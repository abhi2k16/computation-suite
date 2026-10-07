"""
test_hex8_bbar.py -- Wave 2 item 11 (docs/consolidated_future_roadmap.
md, source general_purpose_extensions_roadmap.md / nonlinear_fem_
lessons.md / tensormesh_comparative_analysis.md Sec.6.4): validates
Hex8SolidBbar (elements/solids.py), the mean-dilatation B-bar cure for
volumetric locking as Poisson's ratio nu -> 0.5.

Three lines of evidence:

1. EXACT reduction to plain Hex8Solid3D on any affine/constant-strain
   field -- proven in the class's own docstring (any such field has an
   IDENTICAL B at every Gauss point, so B-bar's volume-averaging is a
   no-op); checked here directly rather than just asserted.
2. A genuine patch test (uniform stress/strain state reproduced
   exactly), same standard this package uses for every other new
   element (see test_simplex_elements.py / test_higher_order_
   elements.py).
3. The actual locking cure: the SAME cantilever bending benchmark
   examples/main.py and test_hourglass_stabilization.py already use
   (Hex8Solid3D's own documented ~0.71 shear-locking ratio at nu=0.3),
   swept toward nu=0.4999 -- plain Hex8Solid3D should lock much WORSE
   (ratio collapsing further from 1.0) as nu->0.5, while Hex8SolidBbar
   should stay comparatively stable.
"""
import numpy as np
from fea_engine import Hex8Solid3D, Hex8SolidBbar, FESystem, D_solid3d, Material
from fea_engine.mesh import box_mesh


def _steel(nu=0.3):
    return Material(E=2.1e11, nu=nu, rho=7850.0)


def _single_hex8():
    return np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])


def test_bbar_reduces_exactly_to_hex8solid3d_on_affine_fields():
    print("=" * 70)
    print("CHECK 1: Hex8SolidBbar's stiffness matches Hex8Solid3D's own")
    print("EXACTLY (bit-for-bit, to machine precision) when applied to any")
    print("affine displacement field -- the 'limitation principle' this")
    print("class's docstring claims, checked directly")
    print("=" * 70)
    coords = _single_hex8()
    D = D_solid3d(_steel(nu=0.3))
    ke_std = Hex8Solid3D().full_stiffness(coords, D)
    ke_bbar = Hex8SolidBbar().full_stiffness(coords, D)

    rng = np.random.default_rng(1)
    max_rel = 0.0
    for _ in range(20):
        A = rng.standard_normal((3, 3)) * 0.01
        b = rng.standard_normal(3) * 0.01
        u = np.zeros(24)
        for a in range(8):
            u[3 * a:3 * a + 3] = A @ coords[a] + b
        f_std = ke_std @ u
        f_bbar = ke_bbar @ u
        rel = np.max(np.abs(f_bbar - f_std)) / max(np.max(np.abs(f_std)), 1e-30)
        max_rel = max(max_rel, rel)
    print(f"  max relative force difference over 20 random affine fields: {max_rel:.3e}")
    assert max_rel < 1e-10
    print("  PASS")


def test_bbar_patch_test():
    print()
    print("=" * 70)
    print("CHECK 2: Hex8SolidBbar patch test -- a uniform strain field")
    print("imposed via boundary displacements on a small multi-element")
    print("mesh reproduces that EXACT uniform strain/stress everywhere")
    print("=" * 70)
    steel = _steel(nu=0.3)
    D = D_solid3d(steel)
    mesh = box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
    sys = FESystem(mesh, Hex8SolidBbar())
    sys.assemble_stiffness(D)

    eps_imposed = np.array([0.001, -0.0004, 0.0002, 0.0003, -0.0001, 0.00015])   # [e11,e22,e33,g12,g23,g13]

    def u_affine(x, y, z):
        return np.array([
            eps_imposed[0] * x + 0.5 * eps_imposed[3] * y + 0.5 * eps_imposed[5] * z,
            0.5 * eps_imposed[3] * x + eps_imposed[1] * y + 0.5 * eps_imposed[4] * z,
            0.5 * eps_imposed[5] * x + 0.5 * eps_imposed[4] * y + eps_imposed[2] * z,
        ])

    n_nodes = len(mesh.nodes)
    U_all = np.zeros(3 * n_nodes)
    boundary_nodes = []
    for i, (x, y, z) in enumerate(mesh.nodes):
        on_boundary = (np.isclose(x, 0) or np.isclose(x, 1.0) or
                       np.isclose(y, 0) or np.isclose(y, 1.0) or
                       np.isclose(z, 0) or np.isclose(z, 1.0))
        if on_boundary:
            boundary_nodes.append(i)
    sys.fix_dofs(boundary_nodes, [0, 1, 2])
    for i in boundary_nodes:
        x, y, z = mesh.nodes[i]
        U_all[3 * i:3 * i + 3] = u_affine(x, y, z)

    free = sys.free_dofs
    fixed_vals = U_all
    F_eff = sys.F.copy() - sys.K @ fixed_vals
    Kff = sys.K[np.ix_(free, free)]
    U_all[free] = np.linalg.solve(Kff, F_eff[free])

    max_disp_err = 0.0
    for i, (x, y, z) in enumerate(mesh.nodes):
        u_exact = u_affine(x, y, z)
        u_fem = U_all[3 * i:3 * i + 3]
        max_disp_err = max(max_disp_err, np.max(np.abs(u_fem - u_exact)))
    scale = np.max(np.abs(U_all))
    rel_err = max_disp_err / scale
    print(f"  max nodal displacement error (relative to max |u|): {rel_err:.3e}")
    assert rel_err < 1e-10
    print("  PASS -- every interior node exactly reproduces the imposed uniform strain field")


def _cantilever_ratio(elem_cls, nu, nx=20, ny=4, nz=4):
    steel = _steel(nu=nu)
    L3, h_cs, w_cs = 1.0, 0.05, 0.05
    F_tip3 = -1000.0
    mesh = box_mesh(L3, w_cs, h_cs, nx, ny, nz)
    elem = elem_cls()
    tip_nodes = mesh.nodes_on_plane(axis=0, value=L3)
    fixed_nodes = mesh.nodes_on_plane(axis=0, value=0.0)
    tip_center = tip_nodes[np.argmin(np.abs(mesh.nodes[tip_nodes, 1] - w_cs / 2))]

    sys = FESystem(mesh, elem)
    sys.assemble_stiffness(D_solid3d(steel))
    sys.add_nodal_force(tip_nodes, dof_index=2, total_force=F_tip3)
    sys.fix_dofs(fixed_nodes, [0, 1, 2])
    U = sys.solve_static()
    w_tip = U[3 * tip_center + 2]

    I3 = w_cs * h_cs ** 3 / 12
    w_EB = F_tip3 * L3 ** 3 / (3 * steel.E * I3)
    return w_tip / w_EB


def test_bbar_relieves_additional_volumetric_locking_near_incompressible():
    print()
    print("=" * 70)
    print("CHECK 3: as nu -> 0.5, plain Hex8Solid3D's cantilever-bending")
    print("ratio (already ~0.71 at nu=0.3 from shear locking) collapses")
    print("FURTHER from 1.0 due to added volumetric locking, while")
    print("Hex8SolidBbar's ratio stays comparatively stable")
    print("=" * 70)
    ratio_std_03 = _cantilever_ratio(Hex8Solid3D, nu=0.3)
    ratio_std_05 = _cantilever_ratio(Hex8Solid3D, nu=0.4999)
    ratio_bbar_03 = _cantilever_ratio(Hex8SolidBbar, nu=0.3)
    ratio_bbar_05 = _cantilever_ratio(Hex8SolidBbar, nu=0.4999)

    print(f"  Hex8Solid3D   nu=0.3:    ratio={ratio_std_03:.4f}")
    print(f"  Hex8Solid3D   nu=0.4999: ratio={ratio_std_05:.4f}")
    print(f"  Hex8SolidBbar nu=0.3:    ratio={ratio_bbar_03:.4f}")
    print(f"  Hex8SolidBbar nu=0.4999: ratio={ratio_bbar_05:.4f}")

    assert 0.60 < ratio_std_03 < 0.80, "sanity: matches the documented ~0.71 shear-locking ratio"
    # plain Hex8 should get WORSE (further from 1.0) approaching incompressibility
    assert ratio_std_05 < ratio_std_03 - 0.05, \
        "expected plain Hex8Solid3D to lock further as nu -> 0.5"
    # B-bar should not exhibit anywhere near the same collapse
    assert abs(1.0 - ratio_bbar_05) < abs(1.0 - ratio_std_05), \
        "expected Hex8SolidBbar to stay closer to the reference than plain Hex8Solid3D at nu=0.4999"
    print("  PASS")


def test_bbar_reduced_stiffness_raises():
    print()
    print("=" * 70)
    print("CHECK 4: Hex8SolidBbar has no reduced-integration variant --")
    print("reduced_stiffness() raises rather than silently doing")
    print("something not meaningful")
    print("=" * 70)
    coords = _single_hex8()
    D = D_solid3d(_steel())
    try:
        Hex8SolidBbar().reduced_stiffness(coords, D)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
    print("  PASS")
