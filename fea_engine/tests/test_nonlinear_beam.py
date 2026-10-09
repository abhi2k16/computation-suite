# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
validate_nonlinear_beam.py -- validates fea_engine's new geometrically
nonlinear BEAM element, element.Beam2DCorotational (a 2-node planar
corotational beam-column: small local strain, unlimited rigid rotation
-- see that class's docstring in element.py for the formulation).

Unlike validate_nonlinear_cantilever.py (which had to fake a "beam"
out of a Pratt-truss lattice of TrussTL2D bars, because no nonlinear
beam element existed yet), this script drives a plain end-to-end CHAIN
of the real beam element and can therefore be checked against the
textbook-exact large-deflection cantilever solution: the Euler
ELASTICA (Bisshopp & Drucker 1945). That is the strongest validation
available for a geometrically nonlinear beam, and it is only possible
now that a real bending element exists.

Six checks:
  1. Pure axial stretch (no rotation): internal_force matches the
     trivial closed form N = EA/L0 * e, M1=M2=0, to near machine
     precision.
  2. Rigid-body rotation invariance: translate+rotate BOTH nodes
     together (zero local strain by construction) -> internal_force
     must be exactly zero. The single most important sanity check for
     any corotational element -- get the frame-invariance wrong and
     this fails immediately, no matter how plausible everything else
     looks.
  3. Finite-difference check of the closed-form tangent_stiffness()
     against internal_force(), at generic (non-equilibrium, arbitrarily
     rotated/stretched) states -- the geometric-stiffness term was
     hand-derived (see element.py) and is verified here, not assumed.
  4. Small-load cantilever CHAIN (n elements, real nonlinear_solver.py
     API, mat=(E,A,I) -- no per-element-area workaround needed, unlike
     the truss-lattice script): matches Euler-Bernoulli P*L^3/(3EI) AND
     an independent K0 linear solve.
  5. Mesh independence: the FE small-load deflection matches the
     Euler-Bernoulli closed form at EVERY element count tried, including
     just 2 -- the linearized beam stiffness is nodally exact for a
     tip-loaded prismatic cantilever, so (unlike the truss-lattice
     version of this validation) there is no discretization error to
     converge away in the first place.
  6. Large-load elastica benchmark: the nonlinear load-deflection curve
     from a chain of Beam2DCorotational elements is compared against
     the EXACT Euler elastica solution (obtained independently here by
     shooting the elastica ODE with scipy.integrate.solve_bvp -- not a
     hand-derived formula, so there's no hand-derivation to get wrong),
     across a wide range of load levels reaching order-one rotations.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, cumulative_trapezoid
from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls



def test_nonlinear_beam():
    np.set_printoptions(precision=6, suppress=True)
    beam = elmod.Beam2DCorotational()


    def build_beam_chain(L, n_elem):
        """A straight chain of n_elem Beam2DCorotational elements along the
        x-axis, node i at x = i*L/n_elem -- the FE idealization of a
        continuum cantilever, fixed (u=v=theta=0) at node 0."""
        x = np.linspace(0.0, L, n_elem + 1).reshape(-1, 1)
        nodes = np.hstack([x, np.zeros_like(x)])
        elements = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)
        mesh = Mesh(nodes=nodes, elements=elements, dim=1)
        fes = FESystem(mesh, beam)
        fes.fix_dofs([0], [0, 1, 2])
        return fes, n_elem   # tip node index == n_elem


    def elastica_reference(P, EI, L, n_mesh=400, guess_sol=None):
        """Exact (to BVP-solver tolerance) large-deflection cantilever under
        a transverse tip point load P that stays vertical ("dead" load),
        obtained by shooting the classical elastica ODE

            d^2(phi)/ds^2 = (P/EI) * cos(phi),   phi(0) = 0,  phi'(L) = 0

        (phi(s) = tangent angle at arc length s; phi'(L)=0 <=> zero moment
        at the free tip) -- see Bisshopp & Drucker (1945), Quarterly of
        Applied Mathematics, "Large deflection of cantilever beams". This
        is an INDEPENDENT numerical solve (not a hand-typed closed-form
        elliptic-integral formula, so there's nothing here to transcribe
        wrong), used purely as a reference to check the FE beam chain
        against. Returns (x_tip, y_tip, sol) -- y_tip < 0 for a downward
        load, so the downward tip DEFLECTION is -y_tip; `guess_sol` (a
        previous solve_bvp solution) seeds continuation to the next load
        level, needed for solve_bvp to converge reliably as P grows."""
        def odes(s, y):
            phi, phip = y
            return np.vstack([phip, (P / EI) * np.cos(phi)])

        def bc(ya, yb):
            return np.array([ya[0], yb[1]])

        s = np.linspace(0.0, L, n_mesh)
        y0 = np.zeros((2, s.size)) if guess_sol is None else guess_sol.sol(s)
        sol = solve_bvp(odes, bc, s, y0, max_nodes=50000, tol=1e-10)
        if not sol.success:
            raise RuntimeError(f"elastica solve_bvp failed at P={P}: {sol.message}")

        ss = np.linspace(0.0, L, 4000)
        phi = sol.sol(ss)[0]
        xs = cumulative_trapezoid(np.cos(phi), ss, initial=0.0)
        ys = cumulative_trapezoid(np.sin(phi), ss, initial=0.0)
        return xs[-1], ys[-1], sol


    # =====================================================================
    # Material / geometry (a slender, nearly-inextensible section so the FE
    # beam's axial flexibility doesn't contaminate the comparison against
    # the elastica, which assumes an INEXTENSIBLE rod)
    # =====================================================================
    E = 210e9
    A = 1.0e-2      # m^2 -- deliberately stiff in tension/compression
    I = 2.0e-6      # m^4
    L = 2.0
    EI = E * I
    print(f"EI = {EI:.4e} N*m^2, EA = {E*A:.4e} N (EA/EI*L^2 = {E*A/EI*L**2:.2e} -- "
          f"axial-to-bending stiffness ratio, large by design)")


    print()
    print("=" * 70)
    print("CHECK 1: pure axial stretch (no rotation) vs. closed form N=EA/L0*e")
    print("=" * 70)
    elem_coords = np.array([[0.0, 0.0], [L, 0.0]])
    max_err1 = 0.0
    for stretch_frac in [0.0001, 0.001, 0.01, -0.005]:
        e = stretch_frac * L
        u = np.array([0.0, 0.0, 0.0, e, 0.0, 0.0])
        f = beam.internal_force(elem_coords, u, (E, A, I))
        N_fe = f[3]
        N_cf = E * A / L * e
        err = abs(N_fe - N_cf) / max(abs(N_cf), 1e-30)
        max_err1 = max(max_err1, err)
        print(f"  e/L={stretch_frac:+.4f}  N_fe={N_fe: .6e}  N_cf={N_cf: .6e}  "
              f"rel_err={err:.2e}  M1={f[2]:.2e}  M2={f[5]:.2e}")
    assert max_err1 < 1e-10, "axial stretch does not match closed form"
    print("  PASS")


    print()
    print("=" * 70)
    print("CHECK 2: rigid-body rotation invariance -- internal_force must be")
    print("         EXACTLY zero under a superposed rigid translation+rotation")
    print("=" * 70)
    rng = np.random.default_rng(1)
    max_f = 0.0
    for trial in range(6):
        ec = np.array([[0.0, 0.0], [1.7, 0.0]]) + 0.1 * rng.standard_normal((2, 2))
        X1, X2 = ec
        theta_rigid = rng.uniform(-2.5, 2.5)      # includes rotations > 90 deg
        trans = rng.standard_normal(2)
        c, s = np.cos(theta_rigid), np.sin(theta_rigid)
        R = np.array([[c, -s], [s, c]])
        X1r = X1 + trans
        X2r = X1r + R @ (X2 - X1)
        u_rigid = np.concatenate([X1r - X1, [theta_rigid], X2r - X2, [theta_rigid]])
        f_rigid = beam.internal_force(ec, u_rigid, (E, A, I))
        max_f = max(max_f, np.max(np.abs(f_rigid)))
        print(f"  trial {trial}: theta_rigid={theta_rigid:+.3f} rad  "
              f"max|internal_force|={np.max(np.abs(f_rigid)):.3e} N (or N*m)")
    # threshold set by floating-point roundoff in the atan2/sin/cos
    # round-trip through _wrap(), amplified by this element's large
    # stiffness scale (EA~2e9, EI/L~2e5) -- not by any residual strain.
    assert max_f < 1e-4, "rigid rotation produces spurious internal force -- frame invariance broken"
    print("  PASS -- zero (to floating-point roundoff) force/moment under arbitrary rigid motion")


    print()
    print("=" * 70)
    print("CHECK 3: analytical tangent_stiffness vs. finite difference of")
    print("         internal_force, at generic (rotated, stretched) states")
    print("=" * 70)
    max_err3 = 0.0
    for trial in range(8):
        ec = np.array([[0.0, 0.0], [1.5, 0.9]]) + 0.1 * rng.standard_normal((2, 2))
        u_t = 0.3 * rng.standard_normal(6)
        K = beam.tangent_stiffness(ec, u_t, (E, A, I))
        hfd = 1e-6
        K_fd = np.zeros((6, 6))
        for j in range(6):
            du = np.zeros(6); du[j] = hfd
            fp = beam.internal_force(ec, u_t + du, (E, A, I))
            fm = beam.internal_force(ec, u_t - du, (E, A, I))
            K_fd[:, j] = (fp - fm) / (2 * hfd)
        err = np.max(np.abs(K - K_fd)) / max(np.max(np.abs(K)), 1e-30)
        sym_err = np.max(np.abs(K - K.T))
        max_err3 = max(max_err3, err)
        print(f"  trial {trial}: max rel err vs FD = {err:.2e}   max asymmetry = {sym_err:.2e}")
    assert max_err3 < 1e-5, "tangent_stiffness does not match finite-difference"
    print("  PASS")


    print()
    print("=" * 70)
    print("CHECK 4: small-load cantilever CHAIN (real nonlinear_solver.py API)")
    print("         vs. Euler-Bernoulli closed form AND an independent linear solve")
    print("=" * 70)
    n_elem_baseline = 20
    fes4, tip4 = build_beam_chain(L, n_elem_baseline)
    delta_target_frac = 0.001
    P_small = 3 * EI * (delta_target_frac * L) / L**3
    fes4.add_nodal_force([tip4], 1, -P_small)

    K0 = fes4.assemble_tangent_stiffness(np.zeros(fes4.n_dof), (E, A, I))
    free4 = fes4.free_dofs
    u_lin = np.zeros(fes4.n_dof)
    u_lin[free4] = np.linalg.solve(K0[np.ix_(free4, free4)], fes4.F[free4])

    _, U_hist4 = nls.solve_nonlinear_static(fes4, (E, A, I), n_steps=1, tol=1e-6)
    u_nl = U_hist4[-1]

    delta_nl = -u_nl[tip4 * fes4.npn + 1]
    delta_lin_direct = -u_lin[tip4 * fes4.npn + 1]
    delta_eb = P_small * L**3 / (3 * EI)
    rel_err4 = abs(delta_nl - delta_eb) / delta_eb
    rel_err4_vs_lin = np.max(np.abs(u_nl - u_lin)) / max(np.max(np.abs(u_lin)), 1e-30)
    print(f"  P_small = {P_small:.6f} N")
    print(f"  tip deflection: Newton-Raphson={delta_nl:.10e} m, "
          f"K0-linear={delta_lin_direct:.10e} m, Euler-Bernoulli={delta_eb:.10e} m")
    print(f"  rel err vs Euler-Bernoulli = {rel_err4:.3e}   rel err vs K0-linear solve = {rel_err4_vs_lin:.3e}")
    assert rel_err4 < 1e-4, "small-load chain does not match Euler-Bernoulli"
    # vs-K0-linear tolerance is set by the Newton residual floor (tol=1e-6
    # above), not by the physics -- see validate_nonlinear_cantilever.py's
    # CHECK 1 for the same floating-point-noise-floor discussion.
    assert rel_err4_vs_lin < 2e-3, "small-load nonlinear result does not match the independent linear solve"
    print("  PASS")


    print()
    print("=" * 70)
    print("CHECK 5: mesh independence -- small-load deflection matches the")
    print("         Euler-Bernoulli closed form REGARDLESS of element count")
    print("=" * 70)
    # For a prismatic cantilever loaded only at its tip (no distributed
    # load along the span), the linearized Euler-Bernoulli beam stiffness
    # matrix is NODALLY EXACT -- a single element already reproduces the
    # closed-form tip deflection exactly, so there is no discretization
    # error left to converge away by refining. This check therefore does
    # NOT expect error to shrink with more elements (that premise is wrong
    # for this problem); it expects every mesh, including 2 elements, to
    # already agree with Euler-Bernoulli to roughly the same tight,
    # Newton-tolerance-limited accuracy -- the truss-lattice version of
    # this script, by contrast, only ever approximated a beam and needed
    # real mesh refinement to approach the same target.
    elem_counts = [2, 4, 8, 16, 32]
    rel_errs5 = []
    for ne in elem_counts:
        fes_r, tip_r = build_beam_chain(L, ne)
        fes_r.add_nodal_force([tip_r], 1, -P_small)
        _, U_hist_r = nls.solve_nonlinear_static(fes_r, (E, A, I), n_steps=1, tol=1e-6)
        delta_r = -U_hist_r[-1, tip_r * fes_r.npn + 1]
        err_r = abs(delta_r - delta_eb) / delta_eb
        rel_errs5.append(err_r)
        print(f"  n_elem={ne:3d}  delta_FE={delta_r:.10e} m  rel_err={err_r:.3e}")
    print(f"  -> error stays flat across mesh sizes (Newton-tolerance noise, not discretization "
          f"error): {['%.2e' % e for e in rel_errs5]}")
    assert max(rel_errs5) < 1e-4, f"some mesh size disagrees with Euler-Bernoulli by {max(rel_errs5):.2e}"
    print("  PASS -- every mesh already matches Euler-Bernoulli to ~1e-6, confirming nodal exactness")


    print()
    print("=" * 70)
    print("CHECK 6: large-load ELASTICA benchmark -- FE beam chain vs. the")
    print("         exact large-deflection cantilever solution (Bisshopp-Drucker)")
    print("=" * 70)
    P_ref = 3 * EI * L / L**3   # load at which linear theory predicts delta = L
    load_fracs = np.linspace(0.05, 1.0, 20)

    n_elem_conv = [5, 10, 20, 40]
    final_errs = {}
    fe_curve_by_ne = {}
    for ne in n_elem_conv:
        fes6, tip6 = build_beam_chain(L, ne)
        fes6.add_nodal_force([tip6], 1, -P_ref)
        load_factors, U_hist6 = nls.solve_nonlinear_static(fes6, (E, A, I), load_factors=load_fracs, tol=1e-8)
        delta_fe = -U_hist6[:, tip6 * fes6.npn + 1]
        fe_curve_by_ne[ne] = delta_fe

    guess = None
    elastica_deltas = np.zeros(len(load_fracs))
    elastica_xtip = np.zeros(len(load_fracs))
    for i, frac in enumerate(load_fracs):
        P = frac * P_ref
        xt, yt, guess = elastica_reference(P, EI, L, guess_sol=guess)
        elastica_deltas[i] = -yt
        elastica_xtip[i] = xt

    print(f"  {'P/P_ref':>8} {'delta_elastica':>15}", end='')
    for ne in n_elem_conv:
        print(f"{'FE(n='+str(ne)+')':>14}", end='')
    print()
    for i in range(0, len(load_fracs), 4):
        print(f"  {load_fracs[i]:8.3f} {elastica_deltas[i]:15.6f}", end='')
        for ne in n_elem_conv:
            print(f"{fe_curve_by_ne[ne][i]:14.6f}", end='')
        print()

    for ne in n_elem_conv:
        err = np.max(np.abs(fe_curve_by_ne[ne] - elastica_deltas)) / L
        final_errs[ne] = err
        print(f"  n_elem={ne:3d}: max |delta_FE - delta_elastica| / L = {err:.3e}")

    assert final_errs[n_elem_conv[-1]] < final_errs[n_elem_conv[0]], \
        "refining the FE chain did not converge toward the elastica solution"
    assert final_errs[n_elem_conv[-1]] < 5e-4, \
        f"finest chain ({n_elem_conv[-1]} elements) still {final_errs[n_elem_conv[-1]]:.2e} (as fraction of L) off the elastica"
    print("  PASS -- FE beam chain converges to the exact elastica large-deflection solution")

    print()
    print("ALL CHECKS PASSED")


    # =====================================================================
    # Plots
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    n_elem_plot = 20
    fes_p, tip_p = build_beam_chain(L, n_elem_plot)
    fes_p.add_nodal_force([tip_p], 1, -P_ref)
    plot_fracs = [0.0, 0.3, 0.6, 1.0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(plot_fracs)))
    for frac, c in zip(plot_fracs, colors):
        if frac == 0.0:
            u_plot = np.zeros(fes_p.n_dof)
        else:
            lf = np.linspace(0.0, frac, 25)
            _, U_p = nls.solve_nonlinear_static(fes_p, (E, A, I), load_factors=lf, tol=1e-8)
            u_plot = U_p[-1]
        xdef = fes_p.mesh.nodes[:, 0] + u_plot[0::3]
        ydef = fes_p.mesh.nodes[:, 1] + u_plot[1::3]
        ax.plot(xdef, ydef, '-o', color=c, ms=3, label=f'FE, P/P_ref={frac}')

        if frac > 0.0:
            _, _, sol_plot = elastica_reference(frac * P_ref, EI, L)
            ss = np.linspace(0, L, 200)
            phi = sol_plot.sol(ss)[0]
            xs = cumulative_trapezoid(np.cos(phi), ss, initial=0.0)
            ys = cumulative_trapezoid(np.sin(phi), ss, initial=0.0)
            ax.plot(xs, ys, 'k--', lw=1.0)
    ax.plot([], [], 'k--', lw=1.0, label='elastica (exact)')
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Corotational beam chain ({n_elem_plot} elements)\nvs. exact elastica')
    ax.legend(fontsize=8)

    ax2 = axes[1]
    delta_linear_vals = load_fracs * P_ref * L**3 / (3 * EI)
    ax2.plot(delta_linear_vals / L, load_fracs * P_ref, 'k--', label='linear (EI) extrapolation')
    ax2.plot(elastica_deltas / L, load_fracs * P_ref, 'k-', lw=2, label='elastica (exact)')
    ax2.plot(fe_curve_by_ne[n_elem_conv[-1]] / L, load_fracs * P_ref, 'o', color='tab:red',
              ms=4, label=f'FE (Beam2DCorotational, n={n_elem_conv[-1]})')
    ax2.set_xlabel('tip deflection / L')
    ax2.set_ylabel('tip load P (N)')
    ax2.set_title('Large-deflection cantilever: FE vs. exact elastica')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "cantilever_beam_element_validation.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")
