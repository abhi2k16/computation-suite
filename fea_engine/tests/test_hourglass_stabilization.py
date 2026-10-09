"""
test_hourglass_stabilization.py -- Wave 2 item 12 (docs/consolidated_
future_roadmap.md, source nonlinear_fem_lessons.md's appendix Sec.
8.7.3-8.7.6): validates Element.hourglass_stabilized_stiffness()
(elements/base.py) and FESystem.assemble_stiffness(method=
"hourglass_stabilized") (solver.py), on Hex8Solid3D -- the element
examples/main.py's own "Full vs. reduced integration" section already
demonstrates the underlying problem for (12 spurious modes per element,
mesh-level near-zero eigenvalues even after boundary conditions).

Four independent lines of evidence, matching this project's usual
per-feature validation depth:

1. Single-element RANK check -- reduced integration's 12 spurious
   modes (the exact, hand-checkable number from spurious_zero_energy_
   modes()'s own docstring) are gone after stabilization; only the 6
   genuine rigid-body modes remain near-zero.
2. PATCH-TEST INVARIANCE -- the stabilization stiffness is proven (see
   hourglass_stabilized_stiffness()'s own docstring) to vanish exactly
   on any affine/constant-strain displacement field; checked directly
   on a random affine field, not just asserted.
3. MESH-level near-zero eigenvalues -- the same cantilever mesh/BCs
   examples/main.py uses to show reduced integration is unsafe at
   assembly scale, re-run with method="hourglass_stabilized": no
   near-zero eigenvalues should remain.
4. ACCURACY -- the same cantilever's tip deflection, run three ways
   (full/locking, plain reduced/hourglass-prone-but-not-checked-here,
   hourglass_stabilized), confirms the stabilized-reduced path relieves
   Hex8's documented shear locking (ratio vs. Euler-Bernoulli moves
   closer to 1.0 than full integration's own ~0.71).
"""
__author__ = "Abhijeet"
import numpy as np
from fea_engine import Hex8Solid3D, FESystem, D_solid3d, Material
from fea_engine.mesh import box_mesh


def _steel():
    return Material(E=2.1e11, nu=0.3, rho=7850.0)


def _single_hex8():
    # unit cube, standard trilinear node ordering (same convention
    # used throughout this package's own single-element fixtures)
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])
    return nodes


def test_stabilization_restores_full_rank_on_a_single_element():
    print("=" * 70)
    print("CHECK 1: hourglass_stabilized_stiffness() eliminates all 12")
    print("spurious modes reduced integration introduces on a single Hex8")
    print("=" * 70)
    elem = Hex8Solid3D()
    coords = _single_hex8()
    D = D_solid3d(_steel())

    ke_full = elem.full_stiffness(coords, D)
    ke_reduced = elem.reduced_stiffness(coords, D)
    ke_stab = elem.hourglass_stabilized_stiffness(coords, D)

    eig_full = np.sort(np.abs(np.linalg.eigvalsh(ke_full)))
    eig_reduced = np.sort(np.abs(np.linalg.eigvalsh(ke_reduced)))
    eig_stab = np.sort(np.abs(np.linalg.eigvalsh(ke_stab)))
    scale = eig_full.max()

    n_zero_full = int(np.sum(eig_full < 1e-6 * scale))
    n_zero_reduced = int(np.sum(eig_reduced < 1e-6 * scale))
    n_zero_stab = int(np.sum(eig_stab < 1e-6 * scale))
    print(f"  near-zero eigenvalues: full={n_zero_full}  reduced={n_zero_reduced}  "
          f"stabilized={n_zero_stab}")
    assert n_zero_full == 6, "sanity: full integration should show exactly 6 rigid-body modes"
    assert n_zero_reduced == 18, "sanity: reduced integration should show 6 rigid + 12 spurious"
    assert n_zero_stab == 6, "stabilization should restore full rank (only 6 rigid-body modes left)"
    print("  PASS")


def test_stabilization_vanishes_exactly_on_affine_fields():
    print()
    print("=" * 70)
    print("CHECK 2: the stabilization correction (ke_stab - ke_reduced) is")
    print("EXACTLY zero when applied to ANY affine (constant-strain,")
    print("including rigid-body) nodal displacement field -- proven")
    print("mathematically in the function's own docstring; checked here")
    print("directly on 20 random affine fields")
    print("=" * 70)
    elem = Hex8Solid3D()
    coords = _single_hex8()
    D = D_solid3d(_steel())

    ke_reduced = elem.reduced_stiffness(coords, D)
    ke_stab = elem.hourglass_stabilized_stiffness(coords, D)
    K_correction = ke_stab - ke_reduced

    rng = np.random.default_rng(0)
    max_rel = 0.0
    for _ in range(20):
        A = rng.standard_normal((3, 3)) * 0.01     # small displacement gradient
        b = rng.standard_normal(3) * 0.01           # rigid translation
        u = np.zeros(24)
        for a in range(8):
            u[3 * a:3 * a + 3] = A @ coords[a] + b
        f_correction = K_correction @ u
        rel = np.max(np.abs(f_correction)) / max(np.max(np.abs(ke_reduced @ u)), 1e-30)
        max_rel = max(max_rel, rel)
    print(f"  max relative stabilization-force over 20 random affine fields: {max_rel:.3e}")
    assert max_rel < 1e-8
    print("  PASS -- stabilization is a genuine no-op on every affine field, so it")
    print("  cannot corrupt a patch test")


def test_mesh_level_near_zero_eigenvalues_eliminated():
    print()
    print("=" * 70)
    print("CHECK 3: on the SAME cantilever mesh/BCs examples/main.py uses")
    print("to show plain reduced integration is unsafe at assembly scale,")
    print("method='hourglass_stabilized' leaves NO near-zero eigenvalues")
    print("=" * 70)
    steel = _steel()
    L3, h_cs, w_cs = 1.0, 0.05, 0.05
    nx3, ny3, nz3 = 6, 2, 2   # coarser than main.py's demo -- cheap enough for a test, same qualitative issue
    mesh = box_mesh(L3, w_cs, h_cs, nx3, ny3, nz3)
    elem = Hex8Solid3D()
    fixed_nodes = mesh.nodes_on_plane(axis=0, value=0.0)

    # A slender, coarse cantilever's genuine softest bending eigenvalue
    # can legitimately sit many orders of magnitude below its stiffest
    # (axial) one -- confirmed directly below against plain "reduced"
    # (truly singular directions land at ~1e-7 ABSOLUTE, i.e.
    # machine-precision zero relative to a ~1e11 stiffness scale,
    # dramatically smaller than any genuine physical mode), so a
    # tighter tolerance than the single-element check above is used
    # here specifically to separate "singular" from "just soft."
    sys_reduced = FESystem(mesh, elem)
    sys_reduced.assemble_stiffness(D_solid3d(steel), method="reduced")
    sys_reduced.fix_dofs(fixed_nodes, [0, 1, 2])
    free_r = sys_reduced.free_dofs
    eigs_reduced = np.linalg.eigvalsh(sys_reduced.K[np.ix_(free_r, free_r)])
    scale_r = np.max(np.abs(eigs_reduced))
    n_singular_reduced = int(np.sum(np.abs(eigs_reduced) < 1e-9 * scale_r))
    print(f"  plain 'reduced' (unstabilized): {n_singular_reduced} truly-singular "
          f"directions (min|eig|={np.min(np.abs(eigs_reduced)):.3e}) -- the known problem")
    assert n_singular_reduced > 0, "sanity: plain reduced integration should still be singular here"

    sys_stab = FESystem(mesh, elem)
    sys_stab.assemble_stiffness(D_solid3d(steel), method="hourglass_stabilized")
    sys_stab.fix_dofs(fixed_nodes, [0, 1, 2])
    free = sys_stab.free_dofs
    eigs = np.linalg.eigvalsh(sys_stab.K[np.ix_(free, free)])
    scale = np.max(np.abs(eigs))
    n_singular_stab = int(np.sum(np.abs(eigs) < 1e-9 * scale))
    print(f"  hourglass_stabilized: {n_singular_stab} truly-singular directions "
          f"(min|eig|={np.min(np.abs(eigs)):.3e}, max|eig|={scale:.3e})")
    assert n_singular_stab == 0
    print("  PASS")


def test_hourglass_stabilized_relieves_shear_locking_vs_full_integration():
    print()
    print("=" * 70)
    print("CHECK 4: hourglass-stabilized reduced integration gives a tip")
    print("deflection CLOSER to Euler-Bernoulli than full integration's")
    print("own documented ~0.71 ratio (examples/main.py) -- confirms this")
    print("is a genuine locking cure, not just a rank fix")
    print("=" * 70)
    steel = _steel()
    L3, h_cs, w_cs = 1.0, 0.05, 0.05
    F_tip3 = -1000.0
    nx3, ny3, nz3 = 20, 4, 4
    mesh = box_mesh(L3, w_cs, h_cs, nx3, ny3, nz3)
    elem = Hex8Solid3D()
    tip_nodes = mesh.nodes_on_plane(axis=0, value=L3)
    fixed_nodes = mesh.nodes_on_plane(axis=0, value=0.0)
    tip_center = tip_nodes[np.argmin(np.abs(mesh.nodes[tip_nodes, 1] - w_cs / 2))]

    def _solve(method):
        sys = FESystem(mesh, elem)
        sys.assemble_stiffness(D_solid3d(steel), method=method)
        sys.add_nodal_force(tip_nodes, dof_index=2, total_force=F_tip3)
        sys.fix_dofs(fixed_nodes, [0, 1, 2])
        U = sys.solve_static()
        return U[3 * tip_center + 2]

    w_full = _solve("full")
    w_stab = _solve("hourglass_stabilized")

    I3 = w_cs * h_cs ** 3 / 12
    w_EB = F_tip3 * L3 ** 3 / (3 * steel.E * I3)
    ratio_full = w_full / w_EB
    ratio_stab = w_stab / w_EB
    print(f"  Euler-Bernoulli reference: {w_EB:.6e} m")
    print(f"  full integration:         {w_full:.6e} m  (ratio={ratio_full:.4f})")
    print(f"  hourglass-stabilized:      {w_stab:.6e} m  (ratio={ratio_stab:.4f})")
    assert 0.60 < ratio_full < 0.80, "sanity: full integration should show its documented ~0.71 locking ratio"
    assert abs(1.0 - ratio_stab) < abs(1.0 - ratio_full), \
        "hourglass-stabilized reduced integration should be closer to the EB reference than full integration"
    print("  PASS")


def test_unknown_assemble_stiffness_method_raises():
    print()
    print("=" * 70)
    print("CHECK 5: FESystem.assemble_stiffness(method=...) validates its")
    print("argument the same way FESystem(backend=...) does")
    print("=" * 70)
    steel = _steel()
    mesh = box_mesh(1.0, 0.05, 0.05, 2, 1, 1)
    sys = FESystem(mesh, Hex8Solid3D())
    try:
        sys.assemble_stiffness(D_solid3d(steel), method="bogus")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "method" in str(e)
    print("  PASS")
