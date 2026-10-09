# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_shell4director_elastica.py -- Wave 4 item 47, Phase 3, roadmap doc
steps 9-10 (docs/director_based_shell_element_roadmap.md Section 5):
the actual PAYOFF check for the whole director-based-element effort --
does `Shell4Director` recover the elastica's real, accumulated-rotation
membrane-bending coupling (axial foreshortening) that every attempt on
the flat-shell architecture (`Shell4MITCCorotational`, 8 "Design
history" dead ends plus this wave's own building blocks A/B/C) could
not, per `test_shell_corotational_elastica.py`'s own documented ceiling
of ~0.05%-0.06% of the true elastica foreshortening?

SCOPE NOTE, read before extending this file: `Shell4Director.internal_
force()`/`tangent_stiffness()` are REAL central-finite-difference (see
that class's own "TANGENT STRATEGY" docstring) -- correct by
construction, but ~50 seconds per Newton iteration's tangent for even a
SINGLE element (measured directly), because each `tangent_stiffness()`
call is 48 `internal_force()` calls, each itself 48 `strain_energy()`
calls (2304 total per element per iteration). This is why this file
uses deliberately SMALL meshes (1 and 3 elements, not the existing
elastica test's nx=20) and few load steps -- a full mesh-converged
elastica benchmark at this file's tangent-evaluation cost would take
many minutes to hours, well beyond what this test suite can afford.
This is an explicit, documented scope boundary (an analytic B-matrix,
avoiding the FD cost entirely, is future work -- see the roadmap doc's
own Section 4 "Tangent strategy" discussion), not a hidden shortcut:
what IS checked below (a clear, monotonic CONVERGENCE TREND toward the
elastica as elements are added, going from ~49% to ~94% of the true
foreshortening between 1 and 3 elements) is enough to demonstrate the
underlying kinematics are right, even without full mesh convergence.

Two checks:

1. test_single_element_foreshortening_far_exceeds_old_element_ceiling
   -- a SINGLE `Shell4Director` element, at a load giving ~13 degrees
   tip rotation (the same regime `test_shell_corotational_elastica.py`
   documents as its own trustworthy boundary), recovers roughly HALF
   the true elastica foreshortening -- three orders of magnitude more
   than the OLD element's own documented ~0.05%-0.06% ceiling at a
   comparable load, on a SINGLE coarse element with no mesh refinement
   helping it.

2. test_three_element_mesh_converges_toward_elastica -- a 3-element
   chain (still tiny, but no longer a single element) at the SAME load,
   converged via `solve_nonlinear_static()` with NO indefiniteness/
   divergence (itself a real check -- every "Design history" dead end
   on the old architecture either failed to converge or needed a
   redesign at exactly this kind of moderate-rotation, multi-element
   state), showing BOTH tip deflection and foreshortening within a few
   percent of the true elastica value, and foreshortening recovery
   improving from ~49% (1 element) to ~94% (3 elements) -- a genuine
   convergence trend, not a coincidence at one specific mesh size.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.integrate import solve_bvp, cumulative_trapezoid

from fea_engine import Material, D_shell, rectangle_mesh, FESystem
from fea_engine.mesh import Mesh
from fea_engine.nonlinear_solver import solve_nonlinear_static
from fea_engine.elements.shells_director import Shell4Director

E = 210e9
NU = 0.3
H = 0.005
B = 0.05
L = 1.0
I_BEAM = B * H ** 3 / 12
EI = E * I_BEAM
MAT = Material(E=E, nu=NU, rho=7800.0)
D = D_shell(MAT, H)
P_REF = 3 * EI * L / L ** 3


def _build_strip(nx, ny=1):
    mesh2d = rectangle_mesh(L, B, nx, ny)
    nodes3d = np.hstack([mesh2d.nodes, np.zeros((mesh2d.nodes.shape[0], 1))])
    mesh = Mesh(nodes=nodes3d, elements=mesh2d.elements, dim=2)
    fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    tip_nodes = np.where(np.abs(mesh.nodes[:, 0] - L) < 1e-9)[0]
    fes = FESystem(mesh, Shell4Director(), thickness=H)
    fes.fix_dofs(fixed_nodes, [0, 1, 2, 3, 4, 5])
    return fes, tip_nodes


def _tip_w(u_row, tip_nodes, fes):
    return np.mean([u_row[fes.npn * int(n) + 2] for n in tip_nodes])


def _tip_u(u_row, tip_nodes, fes):
    return np.mean([u_row[fes.npn * int(n) + 0] for n in tip_nodes])


def elastica_reference(P, EI, L, n_mesh=400):
    """Same construction as test_shell_corotational_elastica.py's own
    elastica_reference() -- duplicated, not imported, for the same
    "independently runnable" reason that file gives."""
    def odes(s, y):
        phi, phip = y
        return np.vstack([phip, (P / EI) * np.cos(phi)])

    def bc(ya, yb):
        return np.array([ya[0], yb[1]])

    s = np.linspace(0.0, L, n_mesh)
    y0 = np.zeros((2, s.size))
    sol = solve_bvp(odes, bc, s, y0, max_nodes=50000, tol=1e-10)
    if not sol.success:
        raise RuntimeError(f"elastica solve_bvp failed at P={P}: {sol.message}")
    ss = np.linspace(0.0, L, 4000)
    phi = sol.sol(ss)[0]
    xs = cumulative_trapezoid(np.cos(phi), ss, initial=0.0)
    ys = cumulative_trapezoid(np.sin(phi), ss, initial=0.0)
    return xs[-1], ys[-1], np.degrees(phi[-1])


LOAD_FRAC = 0.15   # ~13 deg tip rotation -- the old element's own trustworthy-regime boundary


def test_single_element_foreshortening_far_exceeds_old_element_ceiling():
    fes, tip_nodes = _build_strip(nx=1, ny=1)
    fes.add_nodal_force(tip_nodes, 2, -LOAD_FRAC * P_REF)
    _, U_hist = solve_nonlinear_static(fes, D, n_steps=3, tol=1e-6, max_iter=20)

    w_fe = _tip_w(U_hist[-1], tip_nodes, fes)
    u_fe = _tip_u(U_hist[-1], tip_nodes, fes)
    x_e, y_e, tip_deg = elastica_reference(LOAD_FRAC * P_REF, EI, L)
    u_elastica = x_e - L

    assert u_fe * u_elastica > 0, (
        f"expected foreshortening with the SAME sign as the exact elastica's "
        f"own contraction (u_fe={u_fe:.3e}, u_elastica={u_elastica:.3e})"
    )
    ratio = abs(u_fe / u_elastica)
    assert ratio > 0.10, (
        f"single-element foreshortening recovery ({ratio:.1%} of the true "
        f"elastica value, tip rotation ~{tip_deg:.1f} deg) should far exceed "
        f"Shell4MITCCorotational's own documented ~0.05%-0.06% ceiling at a "
        f"comparable load (test_shell_corotational_elastica.py's "
        f"test_known_limitation_large_rotation_regime) -- if this has "
        f"dropped near that old ceiling, the director-based membrane "
        f"coupling this whole element exists to deliver may have regressed"
    )


def test_three_element_mesh_converges_toward_elastica():
    # Single-element foreshortening at this SAME load, from
    # test_single_element_foreshortening_far_exceeds_old_element_ceiling
    # above -- NOT re-solved here (each Newton solve at this element's
    # FD-tangent cost is expensive; see this file's own module
    # docstring "SCOPE NOTE") -- reused as a fixed baseline to check the
    # convergence TREND without paying for a second full solve. This
    # value is fully deterministic (same mesh, load, solver settings) --
    # verified directly before being hardcoded. Re-measured 2026-09-24
    # after the Green-Lagrange curvature change (was -0.0063647... with
    # the old "moderate" curvature).
    u_fe_1elem = -0.006044830531827686

    fes3, tip3 = _build_strip(nx=3, ny=1)
    fes3.add_nodal_force(tip3, 2, -LOAD_FRAC * P_REF)
    # solve_nonlinear_static's own return signature: (converged_flag_or_
    # similar, U_hist) -- a clean return (no exception, no NaN/inf in the
    # history) IS itself part of the claim here: every "Design history"
    # dead end on the OLD flat-shell architecture either failed to
    # converge or needed a redesigned coupling term at exactly this kind
    # of moderate-rotation, multi-element assembled state (see shells.py's
    # own "ABSOLUTE-ROTATION ATTEMPT" dead end 4 account).
    _, U3 = solve_nonlinear_static(fes3, D, n_steps=3, tol=1e-6, max_iter=20)
    assert np.all(np.isfinite(U3[-1])), "3-element solve produced non-finite displacements"

    w_fe = _tip_w(U3[-1], tip3, fes3)
    u_fe = _tip_u(U3[-1], tip3, fes3)
    x_e, y_e, tip_deg = elastica_reference(LOAD_FRAC * P_REF, EI, L)
    u_elastica = x_e - L

    w_err = abs(w_fe - y_e) / abs(y_e)
    u_err = abs(u_fe - u_elastica) / abs(u_elastica)
    assert w_err < 0.05, f"3-element tip deflection error {w_err:.2%} vs exact elastica (tip rotation ~{tip_deg:.1f} deg)"
    # 0.12 (was 0.10) since the 2026-09-24 Green-Lagrange curvature change.
    # The old "moderate" curvature hit 6.3% here partly through error
    # CANCELLATION: its foreshortening recovery went 93.7% (3 el) ->
    # 100.7% (6) -> 102.7% (10), converging to the wrong answer, and its
    # transverse stiffening was ~0% at any mesh. The Green-Lagrange
    # measure converges monotonically, 89.7% -> 96.2% -> 98.1%, and
    # matches the elastica's stiffening on a 20-element strip to <0.1%
    # (test_shell4director_gl_curvature.py). 3 elements is simply coarse.
    assert u_err < 0.12, f"3-element foreshortening error {u_err:.2%} vs exact elastica"

    ratio_1elem = abs(u_fe_1elem / u_elastica)
    ratio_3elem = abs(u_fe / u_elastica)
    assert ratio_3elem > ratio_1elem, (
        f"expected foreshortening recovery to IMPROVE with mesh refinement "
        f"(1 element: {ratio_1elem:.1%}, 3 elements: {ratio_3elem:.1%}) -- a "
        f"genuine convergence trend, not a coincidence at one mesh size"
    )
