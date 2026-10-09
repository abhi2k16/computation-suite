"""
test_shell_corotational_elastica.py -- Phase C of
docs/geometric_nonlinear_shell_roadmap.md (Section 4 items 3-4): a
large-deflection benchmark against a published/sourced reference
solution, plus quasi-static ramp agreement, for
elements.Shell4MITCCorotational.

Directly mirrors tests/test_nonlinear_beam.py's own elastica benchmark
for Beam2DCorotational -- same reference solution (Bisshopp & Drucker
1945's cantilever elastica, obtained independently here via
scipy.integrate.solve_bvp, not a hand-typed closed form), same "shoot
the exact large-deflection cantilever, ramp the FE model through
solve_nonlinear_static() at the same load levels, compare" pattern --
applied to a NARROW cantilever strip meshed with
Shell4MITCCorotational elements instead of a 1-D beam chain. A strip
narrow enough (L/b = 20 here) that its free long edges are not
constrained by cross-strip continuity behaves, to good approximation,
like an Euler-Bernoulli beam of the same rectangular cross-section
(width b, thickness h) rather than a stiffened wide plate: MITC4's
bending block, unconstrained across the width, develops the same
Poisson-relieving anticlastic curvature a real free-edged strip would,
which is what makes I = b*h^3/12 (not the plate flexural rigidity
Eh^3/12/(1-nu^2)) the correct comparison stiffness -- confirmed
directly by CHECK 1 below, not merely assumed.

Four checks:
  1. Small-load regression against the closed-form Euler-Bernoulli tip
     deflection P*L^3/(3*E*I) -- confirms the "narrow strip behaves like
     a beam" assumption above holds for THIS element/mesh, not just in
     principle, before trusting anything built on top of it.
  2. Mesh-independence within the SMALL-TO-MODERATE rotation regime --
     the FE tip deflection at a fixed load must already be converged
     (not still shifting with more elements), so any elastica-vs-FE
     disagreement seen in CHECK 3 can be attributed to a genuine
     modeling limitation, not unconverged discretization.
  3. The elastica benchmark itself, restricted to the regime this
     element's own docstring (shells.py, "Known remaining limitation",
     "Phase C finding") documents as trustworthy -- tip rotation up to
     roughly 13 degrees, where tip-deflection agreement with the exact
     elastica solution stays within a few percent.
  4. A canary for the SAME known limitation, run deliberately BEYOND
     the trustworthy regime, UPDATED 2026-09-03 after a partial fix
     (shells.py's _bending_membrane_coupling_force(), a mean-deviation
     von Karman coupling -- see that method's own docstring): confirms
     (a) the element now predicts NONZERO, correctly-signed but still
     badly UNDER-magnitude axial (in-plane) foreshortening (a per-
     element-only effect, deliberately weak to preserve exact rigid-
     tilt invariance -- see shells.py's "Design history"; recovering
     the FULL elastica-scale foreshortening needs a stateful,
     incrementally-tracked reference frame across the assembled mesh,
     not yet attempted) at every load level tested, even where the true
     elastica foreshortens substantially more -- and (b) tip-deflection
     disagreement with the elastica still grows, rather than staying
     bounded, once tip rotation exceeds the CHECK-3 regime (confirmed
     unchanged by the partial fix, whose magnitude is too small to move
     this number). If a future kinematics change (the stateful frame
     tracking above) ever closes the REMAINING gap, THIS test is
     expected to start failing -- that would be good news, and the fix
     should come with an update here, not a workaround.
"""
__author__ = "Abhijeet"
import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
from scipy.integrate import solve_bvp, cumulative_trapezoid

from fea_engine import Material, D_shell, Shell4MITCCorotational, rectangle_mesh, FESystem
from fea_engine.mesh import Mesh
from fea_engine.nonlinear_solver import solve_nonlinear_static


E = 210e9
NU = 0.3
H = 0.005    # m, strip thickness
B = 0.05     # m, strip width -- narrow: L/B = 20
L = 1.0      # m, strip length
I_BEAM = B * H ** 3 / 12
EI = E * I_BEAM
MAT = Material(E=E, nu=NU, rho=7800.0)
D = D_shell(MAT, H)
COROTATIONAL = Shell4MITCCorotational()


def _build_strip(nx, ny=2):
    """A narrow rectangular strip in the z=0 plane, x in [0, L], meshed
    nx-by-ny with Shell4MITCCorotational, fixed (all 6 DOFs) along the
    x=0 edge -- the shell idealization of a cantilever, loaded via a
    transverse (z-direction, out-of-plane) tip force spread evenly
    across the tip-edge nodes."""
    mesh2d = rectangle_mesh(L, B, nx, ny)
    nodes3d = np.hstack([mesh2d.nodes, np.zeros((mesh2d.nodes.shape[0], 1))])
    mesh = Mesh(nodes=nodes3d, elements=mesh2d.elements, dim=2)
    fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    tip_nodes = np.where(np.abs(mesh.nodes[:, 0] - L) < 1e-9)[0]
    fes = FESystem(mesh, COROTATIONAL, thickness=H)
    fes.fix_dofs(fixed_nodes, [0, 1, 2, 3, 4, 5])
    return fes, tip_nodes


def _tip_w(u_hist_row, tip_nodes, fes):
    return np.mean([u_hist_row[fes.npn * int(n) + 2] for n in tip_nodes])


def _tip_u(u_hist_row, tip_nodes, fes):
    return np.mean([u_hist_row[fes.npn * int(n) + 0] for n in tip_nodes])


def elastica_reference(P, EI, L, n_mesh=400, guess_sol=None):
    """Exact (to BVP-solver tolerance) large-deflection cantilever under
    a transverse tip point load P that stays vertical ("dead" load),
    via the classical elastica ODE d^2(phi)/ds^2 = (P/EI)*cos(phi),
    phi(0)=0, phi'(L)=0 -- Bisshopp & Drucker (1945), Quarterly of
    Applied Mathematics, "Large deflection of cantilever beams". Same
    formula and solve_bvp shooting approach as tests/
    test_nonlinear_beam.py's own elastica_reference() (duplicated, not
    imported, to keep this file independently runnable -- the two are
    intentionally byte-for-byte the same physics). Returns (x_tip,
    y_tip, tip_angle_deg, sol)."""
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
    return xs[-1], ys[-1], np.degrees(phi[-1]), sol


P_REF = 3 * EI * L / L ** 3   # linear theory predicts delta = L at this load


def test_shell_strip_small_load_matches_euler_bernoulli():
    """A narrow (L/b=20) shell strip, even mesh-converged (confirmed:
    ny=2 vs ny=4 across the width change this by <0.01%, so this is not
    a discretization artifact), does not reproduce Euler-Bernoulli's
    P*L^3/(3EI) to floating-point precision -- MITC4's transverse-shear
    flexibility and the strip's finite (not infinitesimal) width both
    add a small residual stiffness difference relative to an idealized
    1-D beam, measured at ~0.7% even in the linear (near-zero-load)
    regime. The 1% bound below is set by THAT measured baseline (with
    margin), not by 0 -- this check exists to confirm the "narrow strip
    behaves approximately like a beam of I=b*h^3/12" assumption holds
    to within a known, small, load-INDEPENDENT offset, which is what
    licenses comparing this strip's LARGE-load behavior against the
    elastica in the tests below (the offset does not grow with load in
    the small-rotation regime, so it does not contaminate the
    nonlinear comparison)."""
    fes, tip_nodes = _build_strip(nx=20)
    delta_target_frac = 0.001
    P_small = 3 * EI * (delta_target_frac * L) / L ** 3
    fes.add_nodal_force(tip_nodes, 2, -P_small)
    # tol=1e-6, not 1e-8 (2026-09-03, bending-membrane coupling term
    # added to Shell4MITCCorotational -- see shells.py's
    # _bending_membrane_coupling_force()): coupling the membrane and
    # bending blocks raises this system's condition number to ~1e10
    # (their native stiffness scales differ by the same ~1e6-1e7 ratio
    # already documented elsewhere in this codebase), so plain
    # double-precision Newton plateaus at a genuine noise floor around
    # 1.5e-7-5e-7 -- verified directly (bouncing, not trending down,
    # after 80 iterations; disabling the new term reproduces the old
    # clean 1e-10 convergence). tol=1e-6 sits comfortably above that
    # floor; CHECK 1's own accuracy assertion below is what actually
    # matters, not the residual digit count.
    _, U_hist = solve_nonlinear_static(fes, D, n_steps=1, tol=1e-6)
    w_fe = _tip_w(U_hist[-1], tip_nodes, fes)
    w_eb = -P_small * L ** 3 / (3 * EI)
    rel_err = abs(w_fe - w_eb) / abs(w_eb)
    assert rel_err < 0.01, (
        f"narrow-strip small-load deflection ({w_fe:.6e}) does not match "
        f"Euler-Bernoulli ({w_eb:.6e}, rel_err={rel_err:.2e}) -- the "
        f"'narrow strip behaves like a beam of I=b*h^3/12' assumption "
        f"this benchmark depends on does not hold for this mesh")


def test_shell_strip_mesh_independence_moderate_rotation():
    """At a FIXED moderate load (tip rotation ~10 degrees, well inside
    the regime test_elastica_benchmark_small_to_moderate_rotation below
    validates), the FE tip deflection must already be converged --
    refining the mesh should change the answer by only a small fraction
    of a percent, confirming any elastica disagreement is a MODELING
    gap, not leftover discretization error."""
    P = 0.11 * P_REF
    results = {}
    for nx in [10, 20, 40]:
        fes, tip_nodes = _build_strip(nx=nx)
        fes.add_nodal_force(tip_nodes, 2, -P)
        # tol=1e-6, not 1e-7 -- see test_shell_strip_small_load_matches_
        # euler_bernoulli's comment above for why (bending-membrane
        # coupling raises this system's condition number to ~1e10,
        # plateauing plain Newton at a genuine ~1.5e-7-5e-7 floor).
        _, U_hist = solve_nonlinear_static(fes, D, n_steps=6, tol=1e-6)
        results[nx] = _tip_w(U_hist[-1], tip_nodes, fes)
    finest, coarsest = results[40], results[10]
    rel_spread = abs(finest - coarsest) / abs(finest)
    assert rel_spread < 0.01, (
        f"tip deflection is still changing with mesh refinement "
        f"(nx=10: {coarsest:.6f}, nx=40: {finest:.6f}, "
        f"rel_spread={rel_spread:.2e}) at a load this benchmark treats "
        f"as converged")


def test_elastica_benchmark_small_to_moderate_rotation():
    """The actual Phase C acceptance test: a ramp of load levels up to
    ~13 degrees of tip rotation (shells.py's own documented trustworthy
    regime), compared point-by-point against the exact elastica tip
    deflection -- this IS the "quasi-static ramp agreement" check
    (Section 4 item 4) and the "large-deflection benchmark" check
    (item 3) together, exactly how test_nonlinear_beam.py's own CHECK 6
    combines both for Beam2DCorotational."""
    fes, tip_nodes = _build_strip(nx=20)
    load_fracs = np.linspace(0.02, 0.15, 10)
    fes.add_nodal_force(tip_nodes, 2, -P_REF)
    _, U_hist = solve_nonlinear_static(fes, D, load_factors=load_fracs, tol=1e-7, max_iter=40)

    guess = None
    max_err = 0.0
    for i, frac in enumerate(load_fracs):
        w_fe = _tip_w(U_hist[i], tip_nodes, fes)
        P = frac * P_REF
        _, y_elastica, tip_deg, guess = elastica_reference(P, EI, L, guess_sol=guess)
        err = abs(w_fe - y_elastica) / abs(y_elastica)
        max_err = max(max_err, err)
        assert err < 0.03, (
            f"at P/P_ref={frac:.3f} (tip rotation ~{tip_deg:.1f} deg), FE "
            f"tip deflection {w_fe:.6f} disagrees with the exact elastica "
            f"{y_elastica:.6f} by {err:.2%} -- outside the documented "
            f"small-to-moderate-rotation regime")
    assert max_err < 0.03


def test_known_limitation_large_rotation_regime():
    """Canary for the documented (shells.py, 'Phase C finding')
    limitation, UPDATED 2026-09-03 after a partial fix (see shells.py's
    _bending_membrane_coupling_force()): axial foreshortening is no
    longer EXACTLY zero (part (a) below), but the fix is a per-element-
    only, mean-deviation-based von Karman coupling -- deliberately weak
    by construction, to stay exactly zero under a single-element rigid
    tilt (the property that sank three earlier, stronger attempts, see
    shells.py's "Design history") -- so the recovered foreshortening is
    2-3 orders of magnitude SMALLER than the true elastica value (a
    within-element-curvature-only effect, not the dominant accumulated-
    rotation effect a real cantilever shows), and tip-deflection
    agreement with the elastica still degrades once rotation grows past
    the small-to-moderate regime (part (b), unchanged from before this
    fix -- confirmed directly, the fix's magnitude is too small to move
    this number). Closing the remaining gap needs a STATEFUL,
    incrementally-updated per-element reference frame (tracking
    accumulated rotation the way Beam2DCorotational tracks its own
    chord angle) so real neighbor-to-neighbor accumulated rotation can
    be distinguished from a single element's own rigid tilt -- a
    materially bigger architecture change (a per-element persisted
    state, via init_state()/commit_all_states()) than this kinematics
    addition, not yet attempted."""
    fes, tip_nodes = _build_strip(nx=20)
    load_fracs = np.array([0.05, 0.30])
    fes.add_nodal_force(tip_nodes, 2, -P_REF)
    _, U_hist = solve_nonlinear_static(fes, D, load_factors=load_fracs, tol=1e-6, max_iter=40)

    # (a) foreshortening is now NONZERO and correctly SIGNED (matching
    # the elastica's own contraction direction) at every load level, but
    # still tiny relative to the true elastica foreshortening -- measured
    # directly at ~0.05%-0.06% of the true value at these two load
    # levels, nowhere close to closing the gap.
    guess_a = None
    for i in range(len(load_fracs)):
        u_fe = _tip_u(U_hist[i], tip_nodes, fes)
        assert abs(u_fe) > 1e-9, (
            f"expected the new bending-membrane coupling term to produce "
            f"SOME nonzero foreshortening at P/P_ref={load_fracs[i]}, got "
            f"exactly {u_fe:.3e} -- if the coupling term was removed or "
            f"disabled, this whole test needs reconsidering")
        x_e, y_e, _, guess_a = elastica_reference(load_fracs[i] * P_REF, EI, L, guess_sol=guess_a)
        u_elastica = x_e - L
        assert u_fe * u_elastica > 0, (
            f"expected foreshortening to have the SAME sign as the exact "
            f"elastica's own contraction at P/P_ref={load_fracs[i]} "
            f"(u_fe={u_fe:.3e}, u_elastica={u_elastica:.3e})")
        assert abs(u_fe) < 0.01 * abs(u_elastica), (
            f"expected the current per-element-only coupling to still "
            f"badly UNDER-predict foreshortening (this is the documented, "
            f"expected gap, not a bug) at P/P_ref={load_fracs[i]} "
            f"(u_fe={u_fe:.3e}, u_elastica={u_elastica:.3e}, "
            f"ratio={abs(u_fe/u_elastica):.2%}) -- if this ratio has grown "
            f"well past 1%, the underlying kinematics may have improved; "
            f"update this test and shells.py's docstring together")

    # (b) tip-deflection error grows once rotation leaves the regime
    # test_elastica_benchmark_small_to_moderate_rotation validated.
    w_small = _tip_w(U_hist[0], tip_nodes, fes)
    _, y_small, _, guess = elastica_reference(load_fracs[0] * P_REF, EI, L)
    err_small = abs(w_small - y_small) / abs(y_small)

    w_large = _tip_w(U_hist[1], tip_nodes, fes)
    _, y_large, tip_deg_large, _ = elastica_reference(load_fracs[1] * P_REF, EI, L, guess_sol=guess)
    err_large = abs(w_large - y_large) / abs(y_large)

    assert err_large > 5 * err_small, (
        f"expected disagreement with the elastica to grow sharply beyond "
        f"the small-rotation regime (err at P/P_ref={load_fracs[0]}: "
        f"{err_small:.2%}, err at P/P_ref={load_fracs[1]} / tip rotation "
        f"~{tip_deg_large:.1f} deg: {err_large:.2%}) -- if this no longer "
        f"grows, the limitation this test guards may have been fixed")


# =========================================================================
# Plot (visual companion, not a pytest assertion) -- mirrors
# tests/test_nonlinear_beam.py's own two-panel figure: deformed shapes
# at increasing load, and the load-deflection curve vs. the exact
# elastica, extended far enough past the validated regime to make the
# divergence documented above visible, not just asserted.
# =========================================================================
def _make_plot():
    fes_p, tip_p = _build_strip(nx=20)
    fes_p.add_nodal_force(tip_p, 2, -P_REF)
    plot_fracs = [0.0, 0.1, 0.2, 0.35]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(plot_fracs)))
    for frac, c in zip(plot_fracs, colors):
        if frac == 0.0:
            u_plot = np.zeros(fes_p.n_dof)
        else:
            lf = np.linspace(0.0, frac, 25)
            _, U_p = solve_nonlinear_static(fes_p, D, load_factors=lf, tol=1e-7, max_iter=40)
            u_plot = U_p[-1]
        xdef = fes_p.mesh.nodes[tip_p, 0] + u_plot[6 * tip_p + 0]
        zdef = fes_p.mesh.nodes[tip_p, 2] + u_plot[6 * tip_p + 2]
        ax.plot([0] + list(xdef), [0] + list(zdef), '-o', color=c, ms=3,
                label=f'FE, P/P_ref={frac}')
        if frac > 0.0:
            _, _, _, sol_plot = elastica_reference(frac * P_REF, EI, L)
            ss = np.linspace(0, L, 200)
            phi = sol_plot.sol(ss)[0]
            xs = cumulative_trapezoid(np.cos(phi), ss, initial=0.0)
            zs = cumulative_trapezoid(np.sin(phi), ss, initial=0.0)
            ax.plot(xs, zs, 'k--', lw=1.0)
    ax.plot([], [], 'k--', lw=1.0, label='elastica (exact)')
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('w, tip deflection (m)')
    ax.set_title('Shell4MITCCorotational strip (nx=20)\nvs. exact elastica')
    ax.legend(fontsize=8)

    ax2 = axes[1]
    load_fracs_plot = np.linspace(0.02, 0.35, 25)
    fes_c, tip_c = _build_strip(nx=20)
    fes_c.add_nodal_force(tip_c, 2, -P_REF)
    _, U_c = solve_nonlinear_static(fes_c, D, load_factors=load_fracs_plot, tol=1e-6, max_iter=40)
    w_fe_curve = np.array([_tip_w(U_c[i], tip_c, fes_c) for i in range(len(load_fracs_plot))])

    guess = None
    w_elastica_curve = np.zeros(len(load_fracs_plot))
    for i, frac in enumerate(load_fracs_plot):
        _, yt, _, guess = elastica_reference(frac * P_REF, EI, L, guess_sol=guess)
        w_elastica_curve[i] = yt

    delta_linear = load_fracs_plot * P_REF * L ** 3 / (3 * EI)
    ax2.plot(delta_linear / L, load_fracs_plot * P_REF, 'k--', label='linear (EI) extrapolation')
    ax2.plot(-w_elastica_curve / L, load_fracs_plot * P_REF, 'k-', lw=2, label='elastica (exact)')
    ax2.plot(-w_fe_curve / L, load_fracs_plot * P_REF, 'o', color='tab:red', ms=4,
              label='FE (Shell4MITCCorotational, nx=20)')
    ax2.axvline(-np.interp(0.15, load_fracs_plot, w_fe_curve) / L, color='gray', lw=0.8, ls=':')
    ax2.set_xlabel('tip deflection / L')
    ax2.set_ylabel('tip load P (N)')
    ax2.set_title('Narrow cantilever strip: FE vs. exact elastica\n(validated regime left of dotted line)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "shell_corotational_elastica_validation.png")
    fig.savefig(out_path, dpi=150)
    return out_path


if __name__ == "__main__":
    test_shell_strip_small_load_matches_euler_bernoulli()
    test_shell_strip_mesh_independence_moderate_rotation()
    test_elastica_benchmark_small_to_moderate_rotation()
    test_known_limitation_large_rotation_regime()
    path = _make_plot()
    print(f"Saved plot: {path}")
    print("ALL CHECKS PASSED")
