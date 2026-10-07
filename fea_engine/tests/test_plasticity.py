import numpy as np
from fea_engine import elements as elmod
from fea_engine.material import PlasticMaterial1D
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls



def test_plasticity():
    np.set_printoptions(precision=6, suppress=True)

    E, sigma_y, H = 200e9, 250e6, 20e9
    A = 1e-4
    plas_mat = PlasticMaterial1D(E=E, sigma_y=sigma_y, H=H)
    mat = (plas_mat, A)

    L0 = 1.0
    elem_coords = np.array([[0.0, 0.0], [L0, 0.0]])
    tr = elmod.TrussPlastic2D()

    print("=" * 70)
    print("CHECK 1: single-element internal_force vs hand return-mapping")
    print("=" * 70)


    def hand_return_map(eps_total, eps_p_n, alpha_n):
        """Independent (re-derived, not calling element.py) 1-D return map."""
        eps_e_trial = eps_total - eps_p_n
        sigma_trial = E * eps_e_trial
        f_trial = abs(sigma_trial) - (sigma_y + H * alpha_n)
        if f_trial <= 0.0:
            return sigma_trial, eps_p_n, alpha_n, E
        dgamma = f_trial / (E + H)
        sign = np.sign(sigma_trial)
        sigma = sigma_trial - E * dgamma * sign
        return sigma, eps_p_n + dgamma * sign, alpha_n + dgamma, E * H / (E + H)


    max_err1 = 0.0
    state0 = tr.init_state()
    eps_y = sigma_y / E
    for eps in [0.0005, eps_y * 0.5, eps_y * 0.99, eps_y * 1.5, eps_y * 3.0, -eps_y * 2.0]:
        u_elem = np.array([0.0, 0.0, eps * L0, 0.0])
        f = tr.internal_force(elem_coords, u_elem, mat, state=state0)
        sigma_fe = f[2] / A
        sigma_cf, _, _, _ = hand_return_map(eps, state0["eps_p"], state0["alpha"])
        err = abs(sigma_fe - sigma_cf) / max(abs(sigma_cf), 1.0)
        max_err1 = max(max_err1, err)
        regime = "ELASTIC" if abs(eps) <= eps_y else "PLASTIC"
        print(f"  eps={eps:+.6f} ({regime:8s})  sigma_fe={sigma_fe: .4e}  "
              f"sigma_cf={sigma_cf: .4e}  rel_err={err:.2e}")
    print(f"  -> max relative error: {max_err1:.2e}")
    assert max_err1 < 1e-10
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 2: analytical tangent_stiffness vs finite-difference (elastic & plastic)")
    print("=" * 70)
    rng = np.random.default_rng(1)
    max_err2 = 0.0
    for eps_center, label in [(eps_y * 0.4, "elastic branch"), (eps_y * 2.0, "plastic branch")]:
        elem_coords_t = np.array([[0.0, 0.0], [1.0, 0.0]])
        u_elem_t = np.array([0.0, 0.0, eps_center * L0, 0.0])
        state_t = tr.init_state()
        K_analytical = tr.tangent_stiffness(elem_coords_t, u_elem_t, mat, state=state_t)
        h = 1e-8
        K_fd = np.zeros((4, 4))
        for j in range(4):
            du = np.zeros(4); du[j] = h
            fp = tr.internal_force(elem_coords_t, u_elem_t + du, mat, state=state_t)
            fm = tr.internal_force(elem_coords_t, u_elem_t - du, mat, state=state_t)
            K_fd[:, j] = (fp - fm) / (2 * h)
        err = np.max(np.abs(K_analytical - K_fd)) / max(np.max(np.abs(K_analytical)), 1e-30)
        max_err2 = max(max_err2, err)
        print(f"  {label:16s} (eps={eps_center:+.6f}): max rel err = {err:.2e}")
    print(f"  -> max relative error: {max_err2:.2e}")
    assert max_err2 < 1e-5
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 3: load-unload-reload history on a fixed-free bar (FESystem +")
    print("         solve_nonlinear_static with a non-monotonic load_factors seq)")
    print("=" * 70)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    elements = np.array([[0, 1]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)

    fes = FESystem(mesh, elmod.TrussPlastic2D())
    fes.fix_dofs([0], [0, 1])   # node 0 fully fixed
    fes.fix_dofs([1], [1])      # node 1: y fixed, x free (pure uniaxial bar)
    F_ref = 1.5 * sigma_y * A   # trial stress at lambda=1 would be 1.5*sigma_y
    fes.add_nodal_force([1], 0, F_ref)
    fes.init_state()

    lambda_seq = np.array([0.0, 0.3, 0.667, 0.8, 1.0,       # load into tension, past yield
                            0.8, 0.6, 0.3, 0.0,               # unload to zero
                            -0.3, -0.6, -0.8, -1.0,           # reverse into compression, past new yield
                            -0.6, 0.0])                        # unload again

    load_factors, U_hist = nls.solve_nonlinear_static(fes, mat, load_factors=lambda_seq,
                                                        tol=1e-12, max_iter=50)
    eps_hist = U_hist[:, 2] / L0   # node 1 x-displacement / L0 = eps_total

    # Independent replay of the SAME eps_total trajectory through a standalone
    # (not element.py-derived) return map, to cross-check both the constitutive
    # formulas (already checked in CHECK 1/2) AND the state persistence/commit
    # bookkeeping across a long non-monotonic sequence (new in this module).
    eps_p, alpha = 0.0, 0.0
    sigma_hand = np.zeros(len(lambda_seq))
    eps_p_hand = np.zeros(len(lambda_seq))
    alpha_hand = np.zeros(len(lambda_seq))
    for i, eps in enumerate(eps_hist):
        sigma_hand[i], eps_p, alpha, _ = hand_return_map(eps, eps_p, alpha)
        eps_p_hand[i], alpha_hand[i] = eps_p, alpha

    sigma_fe_equilibrium = lambda_seq * F_ref / A   # F_int must equal F_ext at convergence
    err3 = np.abs(sigma_hand - sigma_fe_equilibrium)
    rel3 = err3 / max(sigma_y, 1.0)

    print(f"  {'lambda':>8} {'eps_total':>12} {'sigma_hand':>14} {'sigma=lam*Fref/A':>18} {'abs_err':>10}")
    for i in range(len(lambda_seq)):
        print(f"  {lambda_seq[i]:8.3f} {eps_hist[i]:12.6f} {sigma_hand[i]:14.4e} "
              f"{sigma_fe_equilibrium[i]:18.4e} {err3[i]:10.2e}")

    print(f"  -> max |sigma_hand - lambda*F_ref/A| : {err3.max():.4e}  "
          f"(max relative to sigma_y: {rel3.max():.2e})")
    assert rel3.max() < 1e-6, "FE equilibrium does not match independent return-map replay"

    # Permanent set: at every FULLY UNLOADED step (lambda==0), sigma=0 so
    # eps_total must equal eps_p exactly (eps_e = eps_total - eps_p = sigma/E = 0).
    unloaded_idx = np.where(lambda_seq == 0.0)[0]
    print(f"\n  Permanent-set check at lambda=0 steps (indices {list(unloaded_idx)}):")
    for i in unloaded_idx:
        print(f"    step {i}: eps_total={eps_hist[i]:.6f}  eps_p(hand)={eps_p_hand[i]:.6f}  "
              f"diff={abs(eps_hist[i]-eps_p_hand[i]):.2e}")
        if i > 0:
            assert abs(eps_hist[i]) > 1e-4, \
                "expected a nonzero PERMANENT plastic set after yielding, found ~0"
            assert abs(eps_hist[i] - eps_p_hand[i]) < 1e-9

    # Isotropic hardening: the yield threshold that stopped the FIRST tensile
    # plastic excursion (sigma ~ sigma_y + H*alpha at that point) must be
    # LARGER than sigma_y, and the compressive branch must not re-yield until
    # |sigma_trial| exceeds that SAME expanded threshold (not the virgin sigma_y).
    alpha_after_tension = alpha_hand[4]   # after the lambda=1.0 step
    expanded_yield = sigma_y + H * alpha_after_tension
    print(f"\n  Isotropic hardening: virgin sigma_y={sigma_y:.4e}, expanded yield after "
          f"first tensile excursion = {expanded_yield:.4e} (alpha={alpha_after_tension:.6e})")
    assert expanded_yield > sigma_y * 1.05, "hardening did not measurably raise the yield surface"
    # and the compressive stress at lambda=-1.0 (index 12) should sit at
    # -(sigma_y + H*alpha_at_that_point), i.e. back on the (further-expanded) yield surface
    print(f"  Compressive stress at lambda=-1.0: {sigma_hand[12]:.4e}  "
          f"(-(sigma_y+H*alpha) at that step = {-(sigma_y + H*alpha_hand[12]):.4e})")
    assert abs(sigma_hand[12] - (-(sigma_y + H * alpha_hand[12]))) / sigma_y < 1e-4

    print("  PASS -- equilibrium matches an independent return-map replay at every step,")
    print("          permanent plastic set confirmed after unload, and isotropic hardening")
    print("          correctly raises the yield threshold in BOTH tension and compression.")

    print()
    print("ALL CHECKS PASSED")
