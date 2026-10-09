__author__ = "Abhijeet"

import numpy as np
from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem


def test_curved_contact():
    from fea_engine.nonlinear_solver import solve_nonlinear_static

    np.set_printoptions(precision=6, suppress=True)

    ce = elmod.GapContactCurvedFriction()

    print("=" * 70)
    print("CHECK 1: hand-value check of pn / n_hat -- inside vs outside the circle")
    print("=" * 70)
    center = np.array([0.0, 0.0])
    R = 1.0
    k_p, k_t, mu = 1.0e5, 5.0e4, 0.3
    mat0 = (k_p, k_t, mu, center, R)

    # node's undeformed position is (2, 0) -- 1.0 outside the circle boundary
    elem_coords = np.array([[2.0, 0.0]])

    # (a) u=0 -> position (2,0), d=2 > R=1 -> INACTIVE
    f_inactive = ce.internal_force(elem_coords, np.array([0.0, 0.0]), mat0)
    print(f"  inactive: f_int = {f_inactive}  (expect [0, 0])")
    assert np.allclose(f_inactive, [0.0, 0.0])

    # (b) u=(-1.5,0) -> position (0.5,0), d=0.5 < R=1 -> ACTIVE, penetration=0.5
    u_active = np.array([-1.5, 0.0])
    f_active = ce.internal_force(elem_coords, u_active, mat0)
    pn_expect = k_p * (R - 0.5)          # = k_p * 0.5
    f_expect = np.array([-pn_expect, 0.0])   # points toward center (-x), mu=0 trial => tt=0 (first contact, s_stick=None -> stick with e_t=0)
    print(f"  active:   f_int = {f_active}   expected {f_expect}")
    assert np.allclose(f_active, f_expect, rtol=1e-10)
    print("  PASS -- inactive gives zero force; active force points TOWARD the "
          "circle's center (increasing-penetration direction), matching the "
          "package's F_int = dU/du convention (same sign convention as "
          "GapContactPenalty, NOT the naive 'push the node away' formula).")

    print()
    print("=" * 70)
    print("CHECK 2: frictionless (mu=0) closed-form tangent vs the element's own "
          "finite-difference tangent_stiffness()")
    print("=" * 70)
    mat_nofric = (k_p, k_t, 0.0, center, R)
    u_test = np.array([-1.3, 0.4])     # position (0.7, 0.4), d = sqrt(0.85) ~ 0.9219 < R
    pos = elem_coords[0] + u_test
    d_vec = pos - center
    d = np.linalg.norm(d_vec)
    assert d < R, "test point must be inside the circle to exercise the active branch"
    n_hat = d_vec / d
    I2 = np.eye(2)
    K_closed = k_p * (1 - R / d) * I2 + k_p * (R / d) * np.outer(n_hat, n_hat)
    K_fd = ce.tangent_stiffness(elem_coords, u_test, mat_nofric)
    err2 = np.max(np.abs(K_closed - K_fd)) / np.max(np.abs(K_closed))
    print(f"  K_closed =\n{K_closed}")
    print(f"  K_fd (element's own FD tangent) =\n{K_fd}")
    print(f"  max relative error = {err2:.3e}")
    assert err2 < 1e-6
    # also check the documented eigenvalues directly: +k_p (normal), k_p*(1-R/d) (tangential, negative)
    eigvals = np.sort(np.linalg.eigvalsh(K_closed))
    print(f"  eigenvalues = {eigvals}  (expect one = k_p*(1-R/d) = {k_p*(1-R/d):.4f} < 0, "
          f"one = k_p = {k_p:.4f})")
    assert np.isclose(eigvals[0], k_p * (1 - R / d), rtol=1e-8)
    assert np.isclose(eigvals[1], k_p, rtol=1e-8)
    print("  PASS -- independently-derived closed-form curvature tangent matches "
          "the element's finite-difference tangent_stiffness() to high precision; "
          "the negative tangential eigenvalue is the expected curvature effect "
          "for a convex obstacle (Newton-Raphson only needs K_T invertible, not "
          "positive-definite, at each iterate).")

    print()
    print("=" * 70)
    print("CHECK 3: stick-then-slip sequence along a fixed-penetration arc sweep")
    print("=" * 70)
    # keep penetration depth constant (d=0.5 => pn=k_p*0.5 fixed) and sweep the
    # node ALONG the circle's tangential direction via increasing theta -- this
    # is exactly the "updating normal, sliding along a curved obstacle" motion
    # the flat-wall GapContactPenalty element cannot represent.
    d_fixed = 0.5
    pn_fixed = k_p * (R - d_fixed)
    limit = mu * pn_fixed
    print(f"  pn_fixed = {pn_fixed:.3f}   slip limit mu*pn = {limit:.3f}")

    state = ce.init_state()
    thetas = np.linspace(0.0, 0.6, 16)   # radians; s = R*theta grows monotonically
                                          # past the slip threshold s = limit/k_t = 0.3
    tt_hist, mode_hist = [], []
    for theta in thetas:
        pos = center + d_fixed * np.array([np.cos(theta), np.sin(theta)])
        u_elem = pos - elem_coords[0]
        r = ce._trial(elem_coords, u_elem, mat0, state)
        tt_hist.append(r["tt"])
        s = R * theta
        s_stick = state["s_stick"] if state["s_stick"] is not None else s
        e_t_trial = s - s_stick
        tt_trial_closed = k_t * e_t_trial
        mode = "slip" if abs(tt_trial_closed) > limit + 1e-9 else "stick"
        mode_hist.append(mode)
        tt_closed = np.clip(tt_trial_closed, -limit, limit)
        assert np.isclose(r["tt"], tt_closed, atol=1e-8), \
            f"theta={theta:.4f}: tt_fe={r['tt']:.6f} vs tt_closed={tt_closed:.6f}"
        state = ce.commit_state(elem_coords, u_elem, mat0, state)

    print(f"  tt history (first 6): {[f'{v:.2f}' for v in tt_hist[:6]]}")
    print(f"  mode history: {mode_hist}")
    assert mode_hist[0] == "stick", "small initial sweep should still be sticking"
    assert "slip" in mode_hist, "sweep didn't reach the slip plateau -- test range too small"
    assert all(abs(v) <= limit + 1e-8 for v in tt_hist), "tt exceeded the Coulomb limit"
    slip_vals = [v for v, m in zip(tt_hist, mode_hist) if m == "slip"]
    assert np.allclose(np.abs(slip_vals), limit, atol=1e-6), \
        "slip-phase tangential force should sit exactly on the mu*pn plateau"
    print("  PASS -- tangential force ramps linearly (stick, tt=k_t*e_t) then "
          "clamps exactly at the Coulomb limit mu*pn (slip plateau), matching "
          "the closed-form return-mapping prediction at every step, via the "
          "SAME state/commit_state mechanism as TrussPlastic2D (Module 9).")

    print()
    print("=" * 70)
    print("CHECK 4: full FESystem + nonlinear_solver integration, exact equilibrium")
    print("=" * 70)
    # a shallow 3-node, 2-element "V" truss (node0, node2 fixed supports; node1
    # the free apex, offset BELOW the line of supports) pushed upward into a
    # circular obstacle centered above it. A single horizontal 2-node truss has
    # ZERO transverse (y) tangent stiffness at u=0 (a mechanism -- K_T singular),
    # so a "V" shape is used instead: with the two members at different angles,
    # their combined stiffness at the free node is full rank from the very first
    # Newton iteration, same as the classic von-Mises-truss benchmark already
    # used elsewhere in this package. add_contact_element() registers
    # GapContactCurvedFriction on node 1 with ZERO changes to solver.py's main
    # assembly loop beyond the contact-state passthrough fix made this session.
    nodes = np.array([[0.0, 0.0], [1.0, -0.1], [2.0, 0.0]])
    elements = np.array([[0, 1], [1, 2]])
    mesh = Mesh(nodes=nodes, elements=elements, dim=1)
    truss = elmod.TrussTL2D()
    sysc = FESystem(mesh, truss)
    sysc.fix_dofs([0, 2], [0, 1])
    sysc.init_state()   # harmless no-op for TrussTL2D (no init_state), but
                         # required so add_contact_element()'s state slot exists
                         # alongside self.state machinery being exercised too

    # obstacle: circle centered at (1.0, 0.15), radius 0.10 -- node 1's rest
    # position (1,-0.1) is a gap of 0.15 below the circle's near edge; pushing
    # node 1 up toward y~0.24 (well within reach at this force/stiffness, per
    # the exploratory single-truss-arm calculation) closes the gap and engages
    # contact partway through the load ramp.
    center2 = np.array([1.0, 0.15])
    R2 = 0.10
    mat_contact = (k_p, k_t, mu, center2, R2)
    sysc.add_contact_element(ce, [1], mat_contact)
    sysc.add_nodal_force([1], 1, 300.0)   # push node 1 upward, toward the obstacle

    mat_truss = (210e9, 1e-6)
    load_factors, U_hist = solve_nonlinear_static(sysc, mat_truss, n_steps=20)
    U_final = U_hist[-1]
    print(f"  node 1 final displacement: {U_final[2:4]}")
    pos1_final = nodes[1] + U_final[2:4]
    gap_final = np.linalg.norm(pos1_final - center2) - R2
    print(f"  final gap (should be <= ~0, contact engaged): {gap_final:.4e}")
    assert gap_final < 1e-3, "expected the contact to have engaged by the final load step"

    F_int_final = sysc.assemble_internal_force(U_final, mat_truss)
    resid = sysc.F - F_int_final
    free = sysc.free_dofs
    print(f"  |residual| at free dofs (Newton converged) = {np.linalg.norm(resid[free]):.3e}")
    assert np.linalg.norm(resid[free]) < 1e-4

    # exact equilibrium -- but NOT the usual "sum(applied)+sum(reaction)=0"
    # rigid-body-mode identity used everywhere else in this project, because
    # that identity relies on every element's internal_force() being
    # self-equilibrated under a uniform translation (true for TrussTL2D,
    # which returns [-f2, f2] -- always sums to zero over its own two nodes,
    # by inspection, independent of u). GapContactCurvedFriction is NOT
    # translation-invariant: it acts against a FIXED obstacle anchored in
    # space, so it behaves like a grounded spring/support, not a normal
    # element -- its force does not cancel internally. The correct identity
    # instead follows from: (a) summing TrussTL2D's self-equilibrated
    # contribution over BOTH elements and ALL 3 nodes gives EXACTLY zero,
    # always, so summing F_int_total over all 3 nodes leaves only the
    # contact force at node 1; (b) Newton convergence gives F_ext=F_int at
    # every FREE dof (node 1); (c) no external load acts on the fixed nodes.
    # Combining: reaction(node0) + reaction(node2) == f_contact(node1) -
    # F_ext(node1), an exact, independently-checkable equilibrium statement.
    f_contact_alone = ce.internal_force(np.array([nodes[1]]), U_final[2:4],
                                         mat_contact, state=sysc.contact_state[0])
    lhs = F_int_final[0:2] + F_int_final[4:6]          # reaction(node0) + reaction(node2)
    rhs = f_contact_alone - sysc.F[2:4]                # f_contact(node1) - F_ext(node1)
    err4 = np.max(np.abs(lhs - rhs))
    print(f"  reaction(node0)+reaction(node2) = {lhs}")
    print(f"  f_contact(node1) - F_ext(node1) = {rhs}")
    print(f"  |difference| = {err4:.3e}")
    assert err4 < 1e-4
    print("  PASS -- add_contact_element() + solve_nonlinear_static() integrate "
          "GapContactCurvedFriction with ZERO changes beyond the contact-state "
          "passthrough fix made in solver.py this session; Newton residual at "
          "convergence is ~0 at every free DOF, and the (non-self-equilibrated, "
          "grounded-support-like) contact force independently reconciles with "
          "the two support reactions exactly.")

    print()
    print("=" * 70)
    print("CHECK 5: updating normal -- n_hat differs at two different tangential "
          "positions on the SAME obstacle (the property GapContactPenalty's flat "
          "wall structurally cannot have)")
    print("=" * 70)
    d_fixed2 = 0.6
    theta_a, theta_b = 0.0, np.pi / 2
    pos_a = center + d_fixed2 * np.array([np.cos(theta_a), np.sin(theta_a)])
    pos_b = center + d_fixed2 * np.array([np.cos(theta_b), np.sin(theta_b)])
    u_a = pos_a - elem_coords[0]
    u_b = pos_b - elem_coords[0]
    r_a = ce._trial(elem_coords, u_a, mat0, ce.init_state())
    r_b = ce._trial(elem_coords, u_b, mat0, ce.init_state())
    print(f"  n_hat at theta=0:    {r_a['n_hat']}")
    print(f"  n_hat at theta=pi/2: {r_b['n_hat']}")
    assert not np.allclose(r_a["n_hat"], r_b["n_hat"]), \
        "expected the contact normal to update with tangential position on a curved obstacle"
    assert np.isclose(np.linalg.norm(r_a["n_hat"]), 1.0)
    assert np.isclose(np.linalg.norm(r_b["n_hat"]), 1.0)
    print("  PASS -- n_hat rotates with the node's position along the obstacle, "
          "confirming the 'updating contact normal' half of this module's scope "
          "(GapContactPenalty's mat=(k_p, g0, n_hat) instead FIXES n_hat as a "
          "constant material parameter, correct only for a flat wall).")

    print()
    print("ALL CHECKS PASSED")
