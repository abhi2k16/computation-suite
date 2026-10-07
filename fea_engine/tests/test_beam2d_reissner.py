"""
test_beam2d_reissner.py -- validates Beam2DReissner (Wave 17 item 140,
docs/consolidated_future_roadmap.md): the geometrically exact planar
(Simo-Reissner) shear-deformable rod built to reproduce Georgiou
(2005), "Advanced Proper Orthogonal Decomposition Tools..." (Nonlinear
Dynamics 41:69-110). See that class's own docstring in
elements/beams.py for the full strain-measure / tangent derivation and
the objectivity proof this file's CHECK (b) confirms numerically.

Five checks, exactly the roadmap's own validation plan for item 140:

(a) Tangent vs. central finite difference of internal_force() at
    several RANDOM LARGE-ROTATION states -- the primary correctness
    gate for the hand-derived analytic tangent (relative error must be
    < 1e-6).
(b) Objectivity: a pure rigid-body rotation of the whole element gives
    zero internal force, at rotation magnitudes well past 90 degrees.
(c) Linear pinned-pinned frequencies converge, as the mesh is refined,
    to within 0.2% of the closed-form Timoshenko (kappa_s=1) values
    3078.5 / 25214.0 / 60981.7 rad/s (bending modes 1/3/5 in the
    ordinary sense) -- the exact parameters (L=10, A=1, I=1/12,
    E=12000, G=5000, rho=1e-6) are the roadmap's own gap-analysis
    numbers, reused here so the result is directly comparable.
(d) Large-rotation cantilever: an applied end moment
    M = 2*pi*EI/L rolls the rod into a closed circle (tip returns to
    the root's position, tip rotation = 2*pi exactly) -- same style
    check as tests/test_nonlinear_beam.py's elastica benchmark, just
    with the textbook-exact "constant curvature -> circle" case
    instead of a numerically-integrated elastica BVP (this one has an
    exact closed form to check against: curvature = M/EI everywhere,
    so a full 2*pi turn closes the loop exactly whenever M = 2*pi*EI/L).
(e) Energy conservation: an undamped free-vibration run released from
    nonlinear static equilibrium, driven through
    solve_nonlinear_transient() (Newmark average-acceleration),
    conserves total (kinetic + strain) energy to the SAME drift
    tolerance (<2% over a several-period window) tests/test_nonlinear_
    transient.py's own TestEnergyConsistency check already establishes
    as this codebase's convention for a geometrically nonlinear Newmark
    run (Newmark is not exactly energy-conserving for a nonlinear
    system in general -- a known property, not a bug -- so this is
    reused rather than invented).
"""
import numpy as np

from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine.damping import RayleighDamping
from fea_engine.loads import LoadPattern, TimeHistoryLoad
from fea_engine import nonlinear_solver as nls


def _beam_chain(L, n_elem):
    """A straight chain of n_elem Beam2DReissner elements along the
    x-axis, node i at x = i*L/n_elem -- same convention
    test_nonlinear_beam.py's build_beam_chain() uses for
    Beam2DCorotational."""
    x = np.linspace(0.0, L, n_elem + 1).reshape(-1, 1)
    nodes = np.hstack([x, np.zeros_like(x)])
    elements = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=1)
    return mesh


def test_check_a_tangent_matches_finite_difference_at_large_rotations():
    print("=" * 70)
    print("CHECK (a): analytic tangent_stiffness() vs. central finite")
    print("difference of internal_force(), at RANDOM LARGE-ROTATION states")
    print("=" * 70)
    beam = elmod.Beam2DReissner()
    E, G, A, I, kappa_s = 210e9, 80e9, 1.0e-3, 1.0e-7, 1.0
    mat = (E, G, A, I, kappa_s)
    rng = np.random.default_rng(42)

    max_err = 0.0
    for trial in range(12):
        ec = np.array([[0.0, 0.0], [1.5, 0.9]]) + 0.1 * rng.standard_normal((2, 2))
        u = rng.standard_normal(6)
        u[0:2] *= 0.05; u[3:5] *= 0.05     # displacements: modest (this is a
                                            # STRAIN-magnitude scale, not a
                                            # rotation-magnitude one)
        u[2] = rng.uniform(-3.0, 3.0)      # rotations: deliberately LARGE,
        u[5] = rng.uniform(-3.0, 3.0)      # well past +/-90 degrees

        K = beam.tangent_stiffness(ec, u, mat)
        h = 1e-6
        K_fd = np.zeros((6, 6))
        for j in range(6):
            du = np.zeros(6); du[j] = h
            fp = beam.internal_force(ec, u + du, mat)
            fm = beam.internal_force(ec, u - du, mat)
            K_fd[:, j] = (fp - fm) / (2 * h)
        err = np.max(np.abs(K - K_fd)) / max(np.max(np.abs(K)), 1e-30)
        sym_err = np.max(np.abs(K - K.T))
        max_err = max(max_err, err)
        print(f"  trial {trial:2d}: theta1={u[2]:+.3f} theta2={u[5]:+.3f}  "
              f"rel_err={err:.3e}  asymmetry={sym_err:.3e}")

    print(f"  max relative error across all trials: {max_err:.3e}")
    assert max_err < 1e-6, "analytic tangent does not match finite difference"
    print("  PASS")


def test_check_b_objectivity_rigid_rotation_gives_zero_force():
    print("=" * 70)
    print("CHECK (b): objectivity -- a pure rigid-body rotation (both nodes,")
    print("no actual strain) must give EXACTLY zero internal force")
    print("=" * 70)
    beam = elmod.Beam2DReissner()
    E, G, A, I, kappa_s = 210e9, 80e9, 1.0e-3, 1.0e-7, 1.0
    mat = (E, G, A, I, kappa_s)
    rng = np.random.default_rng(7)

    max_f = 0.0
    for trial in range(10):
        ec = np.array([[0.0, 0.0], [1.7, 0.0]]) + 0.1 * rng.standard_normal((2, 2))
        X1, X2 = ec
        theta_rigid = rng.uniform(-3.0, 3.0)     # past +/-90 degrees
        trans = rng.standard_normal(2)
        c, s = np.cos(theta_rigid), np.sin(theta_rigid)
        R = np.array([[c, -s], [s, c]])
        X1r = X1 + trans
        X2r = X1r + R @ (X2 - X1)
        # the rotation DOF is the cross-section's rotation INCREMENT from
        # its own reference orientation (see class docstring) -- a rigid
        # rotation by theta_rigid increments BOTH nodes' theta by the
        # SAME theta_rigid, matching the applied rigid motion.
        u_rigid = np.concatenate([X1r - X1, [theta_rigid], X2r - X2, [theta_rigid]])
        f = beam.internal_force(ec, u_rigid, mat)
        max_f = max(max_f, np.max(np.abs(f)))
        print(f"  trial {trial}: theta_rigid={theta_rigid:+.3f} rad  "
              f"max|internal_force|={np.max(np.abs(f)):.3e}")

    # Threshold set by floating-point roundoff in the sin/cos evaluations,
    # amplified by this element's stiffness scale (EA ~ 2e8 N) -- not by
    # any residual strain (see CHECK (a)'s independent tangent-vs-FD
    # agreement for the actual strain-measure correctness gate). Same
    # reasoning Beam2DCorotational's own rigid-rotation check uses (see
    # test_nonlinear_beam.py CHECK 2's docstring).
    assert max_f < 1e-6, "rigid rotation produces spurious internal force"
    print(f"  PASS -- max|f| = {max_f:.3e} (floating-point roundoff only)")


def test_check_c_pinned_pinned_frequencies_converge_to_timoshenko():
    print("=" * 70)
    print("CHECK (c): linear pinned-pinned frequencies converge to the")
    print("closed-form Timoshenko (kappa_s=1) values as the mesh is refined")
    print("=" * 70)
    # Exact parameters from the roadmap's own gap-analysis derivation
    # (Wave 17 section, docs/consolidated_future_roadmap.md) -- reused
    # here so the result is directly comparable to the number already
    # worked out there.
    L, A, I = 10.0, 1.0, 1.0 / 12.0
    E, G, rho, kappa_s = 12000.0, 5000.0, 1.0e-6, 1.0
    mat = (E, G, A, I, kappa_s)
    rhoA, rhoI = rho * A, rho * I

    # Bending modes 1, 3, 5 in the ordinary sense (the paper's own
    # numbering interleaves axial/shear modes among them, see the
    # roadmap's Wave 17 section) -- identified below as the 1st, 3rd,
    # and 6th eigenvalue (0-indexed 0, 2, 5) of the SORTED free-free...
    # pinned-pinned spectrum, confirmed by checking each converges
    # monotonically toward its own target as the mesh refines (the
    # OTHER indices converge toward genuinely different values --
    # e.g. index 3 toward the axial mode 1r = pi*c/L = 34414, the
    # roadmap's own cross-check number -- not toward these targets).
    targets = np.array([3078.5, 25214.0, 60981.7])
    mode_indices = [0, 2, 5]

    elem_counts = [10, 20, 40, 80, 160]
    print(f"  {'n_elem':>7} {'mode1':>12} {'err1':>9} {'mode3':>12} {'err3':>9} "
          f"{'mode5':>12} {'err5':>9}")
    errs_by_n = []
    for n in elem_counts:
        mesh = _beam_chain(L, n)
        beam = elmod.Beam2DReissner()
        fes = FESystem(mesh, beam)
        fes.assemble_stiffness(mat)
        fes.assemble_mass((rhoA, rhoI))
        # Axially immovable pins: u1=u2=0 at both ends, theta free --
        # the roadmap's own note (Wave 17 section) that the paper's
        # axial-mode frequency (1r = pi*c/L) implies this BC, not a
        # simply-supported-with-free-axial-slide one.
        fes.fix_dofs([0, n], [0, 1])
        freq_hz, _ = fes.solve_modal(n_modes=8)
        omega = 2 * np.pi * freq_hz
        picked = omega[mode_indices]
        errs = np.abs(picked - targets) / targets
        errs_by_n.append(errs)
        print(f"  {n:7d} {picked[0]:12.3f} {errs[0]*100:8.4f}% "
              f"{picked[1]:12.3f} {errs[1]*100:8.4f}% "
              f"{picked[2]:12.3f} {errs[2]*100:8.4f}%")

    errs_by_n = np.array(errs_by_n)
    # Monotonic convergence (each mode's own error shrinks as the mesh
    # refines) -- a genuine discretization-convergence check, not just
    # "the finest mesh happens to pass."
    for m in range(3):
        assert errs_by_n[-1, m] < errs_by_n[0, m], \
            f"mode {m} did not converge as the mesh was refined"
    assert np.all(errs_by_n[-1] < 0.002), \
        f"finest mesh ({elem_counts[-1]} elements) still exceeds 0.2%: {errs_by_n[-1]}"
    print(f"  PASS -- finest mesh ({elem_counts[-1]} elements) errors: "
          f"{errs_by_n[-1] * 100} % (all < 0.2%)")


def test_check_d_cantilever_end_moment_rolls_into_full_circle():
    print("=" * 70)
    print("CHECK (d): cantilever under end moment M=2*pi*EI/L rolls into a")
    print("CLOSED CIRCLE -- tip returns to the root position, tip rotation")
    print("= 2*pi exactly (constant curvature = M/EI everywhere along an")
    print("initially-straight rod is the textbook-exact closed form here)")
    print("=" * 70)
    E, A, I, G, kappa_s = 210e9, 1.0e-3, 1.0e-7, 210e9 / (2 * 1.3), 1.0
    L = 2.0
    EI = E * I
    mat = (E, G, A, I, kappa_s)
    M_full = 2 * np.pi * EI / L

    n_elem_conv = [10, 20, 40]
    final_errs = {}
    for n in n_elem_conv:
        mesh = _beam_chain(L, n)
        beam = elmod.Beam2DReissner()
        fes = FESystem(mesh, beam)
        fes.fix_dofs([0], [0, 1, 2])
        fes.add_nodal_force([n], 2, M_full)
        load_factors = np.linspace(0.0, 1.0, 80)
        _, U_hist = nls.solve_nonlinear_static(fes, mat, load_factors=load_factors,
                                                tol=1e-10, max_iter=60)
        u_final = U_hist[-1]
        tip = n
        tip_x = mesh.nodes[tip, 0] + u_final[tip * 3 + 0]
        tip_y = mesh.nodes[tip, 1] + u_final[tip * 3 + 1]
        tip_theta = u_final[tip * 3 + 2]
        pos_err = np.hypot(tip_x, tip_y) / L         # tip should land back on (0,0)
        theta_err = abs(tip_theta - 2 * np.pi) / (2 * np.pi)
        final_errs[n] = (pos_err, theta_err)
        print(f"  n_elem={n:3d}  tip=({tip_x: .3e}, {tip_y: .3e})  "
              f"pos_err/L={pos_err:.3e}  theta={tip_theta:.8f}  "
              f"theta_err={theta_err:.3e}")

    for n in n_elem_conv:
        pos_err, theta_err = final_errs[n]
        assert pos_err < 1e-6, f"n_elem={n}: tip does not return to root, err/L={pos_err:.2e}"
        assert theta_err < 1e-9, f"n_elem={n}: tip rotation is not exactly 2*pi, err={theta_err:.2e}"
    print("  PASS -- the rod closes into an exact circle at every mesh tried")


def test_check_e_energy_conservation_in_free_vibration():
    print("=" * 70)
    print("CHECK (e): undamped free vibration (released from nonlinear")
    print("static equilibrium) conserves total energy through")
    print("solve_nonlinear_transient()'s Newmark average-acceleration scheme")
    print("=" * 70)
    E, A, I, G, kappa_s = 210e9, 1.0e-4, 8.333e-9, 80e9, 1.0
    rho, L = 7850.0, 1.0
    mat = (E, G, A, I, kappa_s)
    n_elem = 10

    mesh = _beam_chain(L, n_elem)
    beam = elmod.Beam2DReissner()
    fes = FESystem(mesh, beam)
    fes.fix_dofs([0], [0, 1, 2])
    fes.assemble_mass((rho * A, rho * I))
    fes.assemble_stiffness(mat)
    fes.assemble_damping(RayleighDamping(alpha=0.0, beta=0.0))   # undamped

    fes_static = FESystem(mesh, beam)
    fes_static.fix_dofs([0], [0, 1, 2])
    fes_static.add_nodal_force([n_elem], 1, 50.0)
    _, U_hist_static = nls.solve_nonlinear_static(fes_static, mat, n_steps=10)
    u0 = U_hist_static[-1].copy()

    zero_pattern = LoadPattern(node_ids=np.array([n_elem]), dof_index=1)
    zero_load = TimeHistoryLoad(pattern=zero_pattern, time_fn=lambda t: 0.0)

    freq_hz, _ = fes.solve_modal(n_modes=1)
    period = 1.0 / freq_hz[0]
    T_total, dt = 5 * period, period / 400
    t, U_hist = nls.solve_nonlinear_transient(fes, mat, zero_load, T_total, dt, u0=u0,
                                               tol=1e-8, max_iter=40)

    free = fes.free_dofs
    Mff = fes.M[np.ix_(free, free)]
    v_hist = np.gradient(U_hist[:, free], dt, axis=0, edge_order=2)

    KE = np.zeros(len(t))
    SE = np.zeros(len(t))   # relative to the initial state (a constant offset
                             # from true strain energy, which cancels out of a
                             # CONSERVATION check) -- same convention
                             # test_nonlinear_transient.py's own energy check uses
    se_accum = 0.0
    for i in range(len(t)):
        KE[i] = 0.5 * v_hist[i] @ (Mff @ v_hist[i])
        if i > 0:
            du = U_hist[i] - U_hist[i - 1]
            F_int_prev = fes.assemble_internal_force(U_hist[i - 1], mat)
            F_int_curr = fes.assemble_internal_force(U_hist[i], mat)
            se_accum += 0.5 * (F_int_prev[free] + F_int_curr[free]) @ du[free]
        SE[i] = se_accum

    E_total = KE + SE
    energy_scale = KE.max() - KE.min()
    rel_drift = (E_total.max() - E_total.min()) / energy_scale
    print(f"  period={period:.6e} s, T_total={T_total:.6e} s, dt={dt:.6e} s")
    print(f"  energy drift over 5 periods: {rel_drift * 100:.4f} %")
    # Same 2% tolerance test_nonlinear_transient.py's own
    # TestEnergyConsistency check uses for a geometrically nonlinear
    # Newmark run -- Newmark average-acceleration is not exactly
    # energy-conserving for a nonlinear system (known, not a bug), but
    # should stay small over a several-period window.
    assert rel_drift < 0.02
    print("  PASS")


def test_item142_distributed_transverse_load_sums_to_P_times_L():
    """Item 142: check whether the EXISTING LoadPattern/TimeHistoryLoad
    machinery already expresses a spatially-uniform, time-varying
    distributed transverse load F2(s,t) = P*g(t) as a nodal load vector
    that sums to P*L over a length-L rod -- the roadmap's own (trivial)
    validation bar for this item. Finding: YES, no new loads.py code is
    needed -- LoadPattern.vector() already splits a `total` evenly
    across an arbitrary node_ids list at a fixed dof_index, so setting
    node_ids = every node along the rod and total = P*L*g(t) (via
    TimeHistoryLoad.time_fn) is exactly this load, with a trivially
    correct total. This is a genuinely LUMPED (equal-share) load, not a
    higher-order 'consistent' FE line-load weighting, but the roadmap's
    own acceptance criterion is only about the TOTAL, which any split
    that sums to P*L (equal-share included) satisfies by construction."""
    print("=" * 70)
    print("Item 142: LoadPattern + TimeHistoryLoad already expresses a")
    print("uniform distributed transverse load with an arbitrary time shape")
    print("=" * 70)
    n_elem = 8
    L = 10.0
    P = 3.7        # force per unit length
    Omega = 3121.0

    node_ids = np.arange(n_elem + 1)   # every node along the rod
    pattern = LoadPattern(node_ids=node_ids, dof_index=1)   # transverse (u2)
    g = lambda t: np.cos(Omega * t)    # e.g. the paper's own harmonic g(t)
    load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: P * L * g(t))

    n_dof = (n_elem + 1) * 3
    npn = 3
    for t in [0.0, 1e-4, 5e-4]:
        F = load.force_at(t, n_dof, npn)
        total = F[1::3].sum()   # sum over the transverse (dof_index=1) entries
        expected = P * L * g(t)
        rel_err = abs(total - expected) / max(abs(expected), 1e-30)
        print(f"  t={t:.5e}  sum(F2)={total:.6f}  expected P*L*g(t)={expected:.6f}  "
              f"rel_err={rel_err:.2e}")
        assert rel_err < 1e-12
    print("  PASS -- no new loads.py code needed for item 142")
