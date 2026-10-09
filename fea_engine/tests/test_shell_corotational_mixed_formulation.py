# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_shell_corotational_mixed_formulation.py -- validation for the
Module 23 mixed-formulation ATTEMPT at Shell4MITCCorotational's von
Karman bending-membrane coupling (D5 in the NonLin-HyROM paper-
reproduction project's own terms; see shells.py's "Design history"
comment, "DEAD END 7" for the full account).

RESULT: REJECTED, 2026-09-04. Switching to absolute rotation (dead end
4's own correct physics) plus a mixed formulation tracking the coupling
stress as a lagged (one-call-behind) internal unknown does NOT close
the foreshortening gap without reintroducing catastrophic Newton
divergence -- confirmed to be a genuine limitation of the LAGGED update
strategy itself (every analytic piece -- the Schur-complement tangent
correction, the compatibility residual -- independently verified
correct via complex-step; this is not a bug), not a tunable parameter.
The two checks below that exercise a real solve are marked
`xfail(strict=True)` for exactly this reason: they encode what SHOULD
be true if this specific approach worked, so that either fixing it for
real (see shells.py's "Design history" for the two open paths) or
leaving it broken both surface clearly here, rather than a silently
skipped or deleted test hiding the attempt.

Reuses the exact same narrow-strip/elastica benchmark
test_shell_corotational_elastica.py validates the DEFAULT (non-mixed)
formulation against, so results are directly comparable to that file's
own numbers.

Three checks:
1. test_mixed_formulation_recovers_elastica_scale_foreshortening
   (xfail) -- the payoff this approach was meant to deliver:
   foreshortening as a substantial fraction of the true elastica value
   (not the ~0.05%-0.06% the non-mixed canary test documents). Does not
   even get the chance to be checked -- the solve itself fails to
   converge first (RuntimeError from solve_nonlinear_static).
2. test_mixed_formulation_converges_where_dead_end_4_failed (xfail) --
   Newton robustness at the SAME kind of load levels/step sizes that
   made raw displacement-only absolute rotation diverge (dead end 4).
   FAILS: diverges even faster and more severely than dead end 4 itself
   (a floating-point overflow within the very first load step -- see
   shells.py's "DEAD END 7" account for the measured |R| trace).
3. test_single_element_rigid_tilt_spurious_force -- PASSES (a pure
   diagnostic, not a claim of success): confirms computing the mixed-
   mode force at a rigid-tilt state at least returns a finite result,
   and prints the actual spurious-force numbers for the record. This
   does not exercise a real Newton solve, so it is unaffected by the
   divergence documented above.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.integrate import solve_bvp, cumulative_trapezoid

from fea_engine import Material, D_shell, Shell4MITCCorotational, rectangle_mesh, FESystem
from fea_engine.mesh import Mesh
from fea_engine.nonlinear_solver import solve_nonlinear_static


E = 210e9
NU = 0.3
H = 0.005
B = 0.05
L = 1.0
I_BEAM = B * H ** 3 / 12
EI = E * I_BEAM
MAT = Material(E=E, nu=NU, rho=7800.0)
D = D_shell(MAT, H)


def _build_strip(nx, ny=2):
    mesh2d = rectangle_mesh(L, B, nx, ny)
    nodes3d = np.hstack([mesh2d.nodes, np.zeros((mesh2d.nodes.shape[0], 1))])
    mesh = Mesh(nodes=nodes3d, elements=mesh2d.elements, dim=2)
    fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    tip_nodes = np.where(np.abs(mesh.nodes[:, 0] - L) < 1e-9)[0]
    fes = FESystem(mesh, Shell4MITCCorotational(), thickness=H)
    fes.fix_dofs(fixed_nodes, [0, 1, 2, 3, 4, 5])
    return fes, tip_nodes


def _tip_w(u_hist_row, tip_nodes, fes):
    return np.mean([u_hist_row[fes.npn * int(n) + 2] for n in tip_nodes])


def _tip_u(u_hist_row, tip_nodes, fes):
    return np.mean([u_hist_row[fes.npn * int(n) + 0] for n in tip_nodes])


def elastica_reference(P, EI, L, n_mesh=400, guess_sol=None):
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


P_REF = 3 * EI * L / L ** 3


@pytest.mark.xfail(
    strict=True, reason="Dead end 7 (shells.py 'Design history'): the mixed "
    "formulation's lagged N_add update diverges before this can even be "
    "checked -- solve_nonlinear_static itself raises RuntimeError.")
def test_mixed_formulation_recovers_elastica_scale_foreshortening():
    fes, tip_nodes = _build_strip(nx=20)
    fes.init_iter_state()
    load_fracs = np.array([0.05, 0.30])
    fes.add_nodal_force(tip_nodes, 2, -P_REF)
    _, U_hist = solve_nonlinear_static(fes, D, load_factors=load_fracs, tol=1e-6, max_iter=60)

    guess = None
    ratios = []
    for i in range(len(load_fracs)):
        u_fe = _tip_u(U_hist[i], tip_nodes, fes)
        x_e, y_e, _, guess = elastica_reference(load_fracs[i] * P_REF, EI, L, guess_sol=guess)
        u_elastica = x_e - L
        assert u_fe * u_elastica > 0, (
            f"foreshortening sign mismatch at P/P_ref={load_fracs[i]}: "
            f"u_fe={u_fe:.3e}, u_elastica={u_elastica:.3e}")
        ratio = abs(u_fe / u_elastica)
        ratios.append(ratio)
        print(f"P/P_ref={load_fracs[i]}: u_fe={u_fe:.4e}, u_elastica={u_elastica:.4e}, "
              f"ratio={ratio:.2%}")

    # the non-mixed default recovers ~0.05%-0.06% (test_known_limitation_
    # large_rotation_regime) -- require the mixed formulation to do
    # dramatically better, i.e. actually close most of that gap, not
    # just move the number by some small multiple.
    assert all(r > 0.5 for r in ratios), (
        f"expected the mixed formulation to recover at least half the "
        f"true elastica foreshortening magnitude, got ratios {ratios}")


@pytest.mark.xfail(
    strict=True, reason="Dead end 7 (shells.py 'Design history'): diverges "
    "even faster/worse than dead end 4 itself -- a floating-point overflow "
    "within the very first load step, not merely a non-convergence.")
def test_mixed_formulation_converges_where_dead_end_4_failed():
    """Dead end 4 (raw absolute rotation, no mixed formulation) failed
    catastrophically even at the smallest load step, on a properly-
    resolved (nx=20) mesh, at a modest ~1.7 degree local rotation.
    Confirm the mixed formulation converges cleanly on a fine load ramp
    reaching well past that rotation, with NO exceptions raised."""
    fes, tip_nodes = _build_strip(nx=20)
    fes.init_iter_state()
    load_fracs = np.linspace(0.01, 0.3, 30)
    fes.add_nodal_force(tip_nodes, 2, -P_REF)
    # must not raise
    _, U_hist = solve_nonlinear_static(fes, D, load_factors=load_fracs, tol=1e-6, max_iter=60)
    assert np.all(np.isfinite(U_hist))
    w_final = _tip_w(U_hist[-1], tip_nodes, fes)
    assert abs(w_final) > 0   # sanity: the structure actually deflected


def test_single_element_rigid_tilt_spurious_force():
    """Single isolated Shell4MITCCorotational element, all 4 nodes given
    the SAME rigid rotation (a pure tilt, zero real strain) -- measures
    the spurious membrane force this produces with the mixed formulation
    (absolute rotation) active, honestly compared to the deviation-based
    default's own (near-zero-by-construction) result at the same state."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements = np.array([[0, 1, 2, 3]])
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    elem = Shell4MITCCorotational()

    for tilt_deg in (0.5, 6.0, 17.0):
        tilt = np.radians(tilt_deg)
        u = np.zeros(24)
        for a in range(4):
            u[6 * a + 4] = tilt   # theta_y = betax, uniform -> pure tilt about y-axis

        f_default = elem.internal_force(nodes, u, D)
        n_gauss = 4
        N_add0 = np.zeros((n_gauss, 3))
        f_mixed_at_zero_Nadd = elem.internal_force(nodes, u, D, iter_state=N_add0)

        ref_scale = max(np.linalg.norm(f_default), 1e-30)
        rel_default = np.linalg.norm(f_default) / ref_scale
        rel_mixed = np.linalg.norm(f_mixed_at_zero_Nadd) / max(np.linalg.norm(f_mixed_at_zero_Nadd), ref_scale, 1e-30)
        print(f"tilt={tilt_deg} deg: |f_default|={np.linalg.norm(f_default):.3e}, "
              f"|f_mixed(N_add=0)|={np.linalg.norm(f_mixed_at_zero_Nadd):.3e}")

    # this test is diagnostic (prints the actual numbers for the record)
    # -- the one hard assertion is that computing the mixed-mode force
    # does not raise and returns a finite result.
    assert np.all(np.isfinite(f_mixed_at_zero_Nadd))
