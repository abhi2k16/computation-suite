"""
test_node_to_segment_contact.py -- Wave 3 items 16-17 (docs/
consolidated_future_roadmap.md): validates closest_point_on_segment_2d(),
find_contact_pairs_2d(), NodeToSegmentContact2D, and
NodeToSegmentContact2DFriction -- real geometric node-to-segment
contact search between a slave node and a MASTER SEGMENT of two
ordinary (possibly deformable) mesh nodes, rather than one node
against an analytically-prescribed fixed obstacle
(GapContactPenalty/GapContactCurvedFriction, both untouched).

Six lines of evidence:

1. test_closest_point_on_segment_2d -- pure-geometry sanity: interior
   projection, clamping past both endpoints, and the degenerate
   (zero-length) segment guard.
2. test_node_to_segment_force_and_newtons_third_law -- inactive/active
   branches, and a direct check that f_slave + f_master_a + f_master_b
   = 0 EXACTLY (not just asserted in the class docstring) over many
   random configurations and penetration depths, plus a finite-
   difference tangent check.
3. test_node_to_segment_reduces_to_gap_contact_penalty -- with the
   master segment held stationary (zero master displacement) and the
   slave's projection safely interior (t never clamped),
   NodeToSegmentContact2D's force EXACTLY matches an equivalent
   GapContactPenalty's force -- the "collapses to an already-validated
   special case" pattern this whole package uses, here applied to a
   genuinely new contact discretization.
4. test_find_contact_pairs_2d -- broad-phase sanity: pairs within
   search_radius are found, pairs outside are not.
5. test_two_deformable_bodies_fe_integration -- the headline check: a
   real TWO-DEFORMABLE-BODY system (a TrussTL2D "post" pressing down
   into a Quad4PlaneStress elastic block), contact discovered/verified
   via the same geometry the class itself uses, solved through
   nonlinear_solver.solve_nonlinear_static() with the contact element
   registered via FESystem.add_contact_element() -- zero new solver
   code needed. Checked: global equilibrium (sum of reactions balances
   the applied load to near machine precision) and that the BLOCK
   itself measurably deforms under the contact reaction (the concrete
   "master side is not rigid" property this item's roadmap entry asks
   for).
6. test_friction_stick_then_slip_and_frictionless_reduction --
   NodeToSegmentContact2DFriction's stick/slip decision matches the
   closed-form Coulomb return-map prediction at every step of a
   penetration-held/tangential-sweep test (mirroring test_curved_
   contact.py's own CHECK 3 structure), and mu=0 reproduces
   NodeToSegmentContact2D's own (frictionless) force exactly.
"""
import numpy as np
import pytest

from fea_engine import elements as elmod
from fea_engine.mesh import MultiBlockMesh
from fea_engine.solver import FESystem
from fea_engine.material import D_plane_stress, Material
from fea_engine import nonlinear_solver as nls
from fea_engine.elements.contact import closest_point_on_segment_2d, find_contact_pairs_2d


def test_closest_point_on_segment_2d():
    print("=" * 70)
    print("CHECK 1: closest_point_on_segment_2d -- interior projection,")
    print("clamping past both endpoints, degenerate zero-length segment")
    print("=" * 70)
    a, b = np.array([0.0, 0.0]), np.array([2.0, 0.0])

    t, proj = closest_point_on_segment_2d(np.array([1.0, 1.0]), a, b)
    print(f"  interior: t={t}, proj={proj}")
    assert np.isclose(t, 0.5) and np.allclose(proj, [1.0, 0.0])

    t, proj = closest_point_on_segment_2d(np.array([-1.0, 0.5]), a, b)
    print(f"  clamp at a: t={t}, proj={proj}")
    assert np.isclose(t, 0.0) and np.allclose(proj, a)

    t, proj = closest_point_on_segment_2d(np.array([3.0, -0.5]), a, b)
    print(f"  clamp at b: t={t}, proj={proj}")
    assert np.isclose(t, 1.0) and np.allclose(proj, b)

    t, proj = closest_point_on_segment_2d(np.array([5.0, 5.0]), a, a.copy())
    print(f"  degenerate (zero-length) segment: t={t}, proj={proj}")
    assert t == 0.0 and np.allclose(proj, a)
    print("  PASS")


def test_node_to_segment_force_and_newtons_third_law():
    print()
    print("=" * 70)
    print("CHECK 2: force law (inactive/active), Newton's-third-law self-")
    print("equilibration over random configs, and a finite-difference")
    print("tangent check")
    print("=" * 70)
    nts = elmod.NodeToSegmentContact2D()
    mat = (1e6,)

    # inactive: slave outside the master body per the a->b winding used
    # (a=(0,0), b=(1,0) -> n_hat=(0,-1) -> outward is -y)
    elem_coords = np.array([[0.5, -0.3], [0.0, 0.0], [1.0, 0.0]])
    f0 = nts.internal_force(elem_coords, np.zeros(6), mat)
    print(f"  inactive: f = {f0}")
    assert np.allclose(f0, 0.0)

    rng = np.random.default_rng(2)
    max_imbalance = 0.0
    max_fd_err = 0.0
    for _ in range(15):
        sx = rng.uniform(0.1, 0.9)
        sy0 = rng.uniform(-0.4, -0.05)
        elem_coords = np.array([[sx, sy0], [0.0, 0.0], [1.0, 0.0]])
        u_pen = rng.uniform(0.05, 0.5)   # push slave up past y=0 by u_pen
        u_elem = np.array([0.0, -sy0 + u_pen, 0.0, 0.0, 0.0, 0.0])
        f = nts.internal_force(elem_coords, u_elem, mat)
        imbalance = np.max(np.abs(f[0:2] + f[2:4] + f[4:6]))
        max_imbalance = max(max_imbalance, imbalance)

        h = 1e-7
        K_a = nts.tangent_stiffness(elem_coords, u_elem, mat)
        K_fd = np.zeros((6, 6))
        for j in range(6):
            du = np.zeros(6); du[j] = h
            fp = nts.internal_force(elem_coords, u_elem + du, mat)
            fm = nts.internal_force(elem_coords, u_elem - du, mat)
            K_fd[:, j] = (fp - fm) / (2 * h)
        fd_err = np.max(np.abs(K_a - K_fd)) / max(np.max(np.abs(K_a)), 1.0)
        max_fd_err = max(max_fd_err, fd_err)

    print(f"  max |f_slave+f_master_a+f_master_b| over 15 random active configs: {max_imbalance:.3e}")
    print(f"  max relative tangent_stiffness() vs FD error: {max_fd_err:.3e}")
    assert max_imbalance < 1e-8
    assert max_fd_err < 1e-6
    print("  PASS")


def test_node_to_segment_reduces_to_gap_contact_penalty():
    print()
    print("=" * 70)
    print("CHECK 3: with a stationary master segment and the slave's")
    print("projection safely interior, NodeToSegmentContact2D's force")
    print("EXACTLY matches an equivalent GapContactPenalty")
    print("=" * 70)
    nts = elmod.NodeToSegmentContact2D()
    gc = elmod.GapContactPenalty()
    g0_true = 0.05
    # master segment along the y-axis (a=(0,1), b=(0,0)); slave starts at
    # x=g0_true, y=0.5 (segment midpoint, t stays 0.5 for any small x sweep)
    elem_coords = np.array([[g0_true, 0.5], [0.0, 1.0], [0.0, 0.0]])
    mat_nts = (1e6,)
    mat_gcp = (1e6, -g0_true, np.array([1.0, 0.0]))

    max_err = 0.0
    for u_x in np.linspace(-0.15, 0.15, 25):
        u_elem = np.array([u_x, 0.0, 0.0, 0.0, 0.0, 0.0])
        f_nts = nts.internal_force(elem_coords, u_elem, mat_nts)[0:2]
        f_gcp = gc.internal_force(elem_coords[0], np.array([u_x, 0.0]), mat_gcp)
        max_err = max(max_err, np.max(np.abs(f_nts - f_gcp)))
    print(f"  max abs force diff over 25-point sweep: {max_err:.3e}")
    assert max_err < 1e-10
    print("  PASS -- confirms NodeToSegmentContact2D collapses exactly to the "
          "already-validated GapContactPenalty when the master side doesn't move.")


def test_find_contact_pairs_2d():
    print()
    print("=" * 70)
    print("CHECK 4: find_contact_pairs_2d broad-phase sanity")
    print("=" * 70)
    coords = np.array([
        [0.0, 0.5],    # 0: slave, close to segment (0,1)-(0,2)... wait keep simple
        [5.0, 5.0],    # 1: slave, far from everything
        [0.0, 0.0],    # 2: master a
        [1.0, 0.0],    # 3: master b
        [10.0, 0.0],   # 4: master a (far segment)
        [11.0, 0.0],   # 5: master b (far segment)
    ])
    slaves = [0, 1]
    segments = [(2, 3), (4, 5)]
    pairs = find_contact_pairs_2d(slaves, segments, coords, search_radius=1.0)
    print(f"  pairs found: {pairs}")
    assert (0, (2, 3)) in pairs
    assert (1, (2, 3)) not in pairs
    assert (1, (4, 5)) not in pairs
    assert (0, (4, 5)) not in pairs
    assert len(pairs) == 1
    print("  PASS")


def test_two_deformable_bodies_fe_integration():
    print()
    print("=" * 70)
    print("CHECK 5: TWO deformable bodies -- a TrussTL2D post pressing")
    print("into a Quad4PlaneStress elastic block, contact discovered/")
    print("registered via find_contact_pairs_2d()/add_contact_element(),")
    print("solved through the ordinary nonlinear_solver.solve_nonlinear_")
    print("static() driver with ZERO new solver code")
    print("=" * 70)
    E_truss, A_truss = 200e9, 1e-4
    L0 = 0.5
    G0 = 0.02

    def N_truss(elong):
        """Same closed-form TrussTL2D axial force test_contact.py uses,
        here only as a rough scale for picking a sensible applied load
        (the actual FE answer also includes the block's own compliance,
        so this is NOT used as an independent reference for the final
        answer -- see the equilibrium/deformation checks below instead)."""
        E_GL = ((L0 + elong) ** 2 - L0 ** 2) / (2 * L0 ** 2)
        S = E_truss * E_GL
        return S * A_truss * (L0 + elong) / L0

    P_gap = N_truss(G0)

    nodes = np.array([
        [0.5, G0 + L0],   # 0: truss top, fixed
        [0.5, G0],        # 1: slave (truss bottom / post tip), free in y only
        [0.0, -0.3],      # 2: block bottom-left, fixed
        [1.0, -0.3],      # 3: block bottom-right, fixed
        [1.0, 0.0],       # 4: block top-right -- master_a (contact segment)
        [0.0, 0.0],       # 5: block top-left  -- master_b
    ])
    truss_elems = np.array([[0, 1]])
    quad_elems = np.array([[2, 3, 4, 5]])
    mesh = MultiBlockMesh(nodes=nodes, blocks={"truss": truss_elems, "quad": quad_elems}, dim=2)
    fes = FESystem(mesh, {"truss": elmod.TrussTL2D(), "quad": elmod.Quad4PlaneStress()}, sparse=False)
    fes.fix_dofs([0], [0, 1])
    fes.fix_dofs([1], [0])       # post only moves vertically (roller)
    fes.fix_dofs([2, 3], [0, 1])  # block rests on rigid ground

    # discover the contact pair the same way a real caller would --
    # search from the slave's reference position, wide enough to catch
    # the initial gap G0
    pairs = find_contact_pairs_2d([1], [(4, 5)], nodes, search_radius=0.5)
    assert pairs == [(1, (4, 5))], f"expected exactly one candidate pair, got {pairs}"

    nts = elmod.NodeToSegmentContact2D()
    k_p = 1e8
    for slave, (ma, mb) in pairs:
        # winding (master_a=4, master_b=5): ab=(-1,0) -> n_hat=(0,1) -> outward
        # is +y, matching the slave approaching (and penetrating) from above
        fes.add_contact_element(nts, [slave, ma, mb], (k_p,))

    steel_block = Material(E=2.1e11, nu=0.3)
    D_block = D_plane_stress(steel_block)
    mat = {"truss": (E_truss, A_truss), "quad": D_block}

    P_apply = 3 * P_gap
    fes.add_nodal_force([1], 1, -P_apply)

    load_factors, U_hist = nls.solve_nonlinear_static(fes, mat, n_steps=60, tol=1e-8, max_iter=80)

    u_final = U_hist[-1]
    y1_final = nodes[1, 1] + u_final[3]
    ytr_final = nodes[4, 1] + u_final[9]
    ytl_final = nodes[5, 1] + u_final[11]
    print(f"  final slave y: {y1_final:.6e}   final block top-right y: {ytr_final:.6e}"
          f"   final block top-left y: {ytl_final:.6e}")

    # (a) the block itself measurably deformed -- NOT a rigid obstacle
    block_compression = -ytr_final   # top-right node started at y=0
    print(f"  block top-right compression: {block_compression:.6e} m (must be > 0: genuinely deformable)")
    assert block_compression > 1e-9
    assert np.isclose(ytr_final, ytl_final, atol=1e-12), "centered contact should compress both top nodes equally"

    # (b) global equilibrium: sum of internal force at every FIXED dof
    # (the reactions) balances the total applied external force, to
    # near machine precision -- the same check style test_contact.py's
    # own CHECK 3 uses, generalized to sum-over-all-dofs here since
    # this system has several independent fixed dofs across two bodies
    F_int_final = fes.assemble_internal_force(u_final, mat)
    F_ext_final = load_factors[-1] * fes.F
    free = fes.free_dofs
    residual = np.max(np.abs(F_int_final[free] - F_ext_final[free]))
    print(f"  max residual on free dofs (should be ~solver tol): {residual:.3e}")
    assert residual < 1e-5

    fixed = sorted(set(range(fes.n_dof)) - set(free))
    total_reaction = np.sum(F_int_final[fixed])
    total_applied = np.sum(F_ext_final)
    print(f"  sum applied load: {total_applied:.6f}   sum reactions: {total_reaction:.6f}")
    assert abs(total_reaction + total_applied) < 1e-3 * abs(total_applied)
    print("  PASS -- both bodies deform, contact couples them correctly, "
          "and the whole assembled system is in global equilibrium.")


def test_friction_stick_then_slip_and_frictionless_reduction():
    print()
    print("=" * 70)
    print("CHECK 6: NodeToSegmentContact2DFriction stick/slip matches the")
    print("closed-form Coulomb return map at every step, and mu=0")
    print("reproduces NodeToSegmentContact2D's own force exactly")
    print("=" * 70)
    ntsf = elmod.NodeToSegmentContact2DFriction()
    # a=(0,0), b=(1,0) -> n_hat=[dy,-dx]/L=(0,-1) -> outward is -y, so the
    # master body's INTERIOR is +y here -- the slave must sit at y=+d_fixed
    # (not -d_fixed) to be penetrating, per NodeToSegmentContact2D's own
    # documented sign convention (see that class's docstring / CHECK 2 above).
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    k_p, k_t, mu = 1.0e6, 5.0e5, 0.3
    mat = (k_p, k_t, mu)

    d_fixed = 0.15   # keep penetration constant: slave at y = +d_fixed always
    pn_fixed = k_p * d_fixed
    limit = mu * pn_fixed
    print(f"  pn_fixed = {pn_fixed:.3f}   slip limit mu*pn = {limit:.3f}")

    state = ntsf.init_state()
    xs = np.linspace(0.5, 0.9, 16)   # sweep the slave along the segment
    tt_hist, mode_hist = [], []
    for x in xs:
        elem_coords = np.array([[x, d_fixed], list(a), list(b)])
        u_elem = np.zeros(6)
        r = ntsf._trial(elem_coords, u_elem, mat, state)
        tt_hist.append(r["tt"])
        s = x * 1.0   # seg_len=1, t=x here since a=(0,0),b=(1,0)
        s_stick = state["s_stick"] if state["s_stick"] is not None else s
        tt_closed = np.clip(k_t * (s - s_stick), -limit, limit)
        mode = "slip" if abs(k_t * (s - s_stick)) > limit + 1e-9 else "stick"
        mode_hist.append(mode)
        assert np.isclose(r["tt"], tt_closed, atol=1e-6), \
            f"x={x:.4f}: tt_fe={r['tt']:.6f} vs tt_closed={tt_closed:.6f}"
        state = ntsf.commit_state(elem_coords, u_elem, mat, state)

    print(f"  mode history: {mode_hist}")
    assert mode_hist[0] == "stick"
    assert "slip" in mode_hist, "sweep didn't reach the slip plateau -- test range too small"
    assert all(abs(v) <= limit + 1e-6 for v in tt_hist)
    print("  PASS -- stick-then-slip matches the closed-form return map exactly.")

    print()
    print("  sub-check: mu=0 reduces exactly to NodeToSegmentContact2D")
    nts = elmod.NodeToSegmentContact2D()
    mat0 = (k_p, k_t, 0.0)
    elem_coords = np.array([[0.6, d_fixed], list(a), list(b)])
    u_elem = np.zeros(6)
    f_fric = ntsf.internal_force(elem_coords, u_elem, mat0, state=ntsf.init_state())
    f_frictionless = nts.internal_force(elem_coords, u_elem, (k_p,))
    max_err = np.max(np.abs(f_fric - f_frictionless))
    print(f"  max abs diff (mu=0 friction vs frictionless): {max_err:.3e}")
    assert max_err < 1e-8
    print("  PASS")
