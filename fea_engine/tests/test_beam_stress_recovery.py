"""
test_beam_stress_recovery.py -- Wave 12 item 115 (docs/consolidated_
future_roadmap.md, ICE-ROM/GAP_ANALYSIS.md gap #4): validates
Beam2DCorotational.recover_stress(), the fiber-level axial-stress
recovery this element family was missing (internal_force() only ever
returned nodal forces).

Checks, in increasing order of how much of the formula they exercise:

  1. Algebraic identity: the axial term recover_stress computes,
     E*e_bar/L0, must equal N/A where N is EXACTLY the same axial force
     internal_force()'s own B^T @ [N, M1, M2] transform uses -- both
     read off the assembled nodal force directly, so this is a real
     cross-check of the two independently-written code paths agreeing,
     not a tautology.
  2. Pure end moment (single element, moment applied directly as a
     nodal dof force, no transverse load at all): closed-form Euler-
     Bernoulli theory says a cantilever under a pure end moment has
     CONSTANT moment M(x) = M_applied along its whole length -- so
     recover_stress(xi=0), recover_stress(xi=0.5), recover_stress(xi=1)
     must all agree with sigma = y_fiber*M_applied/I to near machine
     precision, independent of xi. This is the single most decisive
     check: it isolates the bending term from any geometric-
     nonlinearity effects (small rotation) and from any xi-dependence
     ambiguity (constant moment removes it).
  3. Small transverse tip load (linear/small-rotation regime): matches
     the classic P*L*y/I cantilever-root-stress formula to a tight
     (but not machine-precision) tolerance -- FE discretization of a
     single 2-node element is exact for a prismatic cantilever's linear
     bending solution, so the residual here is purely Newton/load-step
     tolerance, not discretization error.
  4. LARGE transverse tip load (order-one tip rotation, ~0.28 rad):
     deviates from the same closed-form linear formula by ~1-2%. This
     is EXPECTED, not a bug -- confirmed by checking (a) it does NOT
     happen in the pure-moment case at the same load scale (check 2
     stays exact even for large M_applied, because uniform bending has
     no other kinematic coupling to worry about) and (b) the deviation
     grows monotonically with rotation, the signature of the standard
     large-deflection cantilever stiffening effect (see
     test_nonlinear_beam.py's own elastica benchmark for the tip-
     deflection version of the same effect). Documented here rather
     than asserted near zero, matching this project's "narrow and
     document, don't overclaim" validation convention.
  5. Broadcasting: y_fiber and xi both accept scalars or arrays and
     combine via ordinary numpy broadcasting; verified against a
     scalar-call-in-a-loop reference (an independent, dumb, obviously-
     correct implementation of "the same thing done N times").
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine.elements.beams import Beam2DCorotational
from fea_engine import nonlinear_solver as nls


E = 2.1e11
B_SEC, H_SEC = 0.02, 0.01
A = B_SEC * H_SEC
I = B_SEC * H_SEC ** 3 / 12.0
MAT = (E, A, I)
Y_TOP = H_SEC / 2.0


def _single_element_cantilever():
    """One Beam2DCorotational element along x in [0, 1], fixed
    (u=v=theta=0) at node 0 -- the same single-element cantilever
    convention test_nonlinear_beam.py's own checks 1-3 use for exact
    closed-form comparisons (no discretization error to average away)."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    elems = np.array([[0, 1]])
    mesh = Mesh(nodes=nodes, elements=elems, dim=1)
    elem = Beam2DCorotational()
    fes = FESystem(mesh, elem)
    fes.fix_dofs([0], [0, 1, 2])
    return fes, nodes, elem


# =====================================================================
# 1. Algebraic identity: sigma_axial term vs internal_force()'s own N
# =====================================================================
def test_sigma_axial_term_matches_internal_force_axial_force_over_area():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    elem = Beam2DCorotational()
    # a generic (non-equilibrium) displacement state with pure axial
    # stretch, no rotation, no transverse motion -- isolates N cleanly
    u_elem = np.array([0.0, 0.0, 0.0, 2.0e-4, 0.0, 0.0])

    sigma_axial = Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=0.0, xi=0.0)
    f_int = elem.internal_force(nodes, u_elem, MAT)
    N_from_internal_force = f_int[3]   # x-force at node 2 == +N for this orientation
    assert np.isclose(sigma_axial, N_from_internal_force / A, rtol=1e-12, atol=0.0)


# =====================================================================
# 2. Pure end moment: constant-moment closed form, near machine precision
# =====================================================================
def test_pure_end_moment_matches_constant_moment_closed_form():
    fes, nodes, elem = _single_element_cantilever()
    M_applied = 500.0
    fes.F[:] = 0.0
    fes.F[5] = M_applied   # rotation dof at node 1 (tip)
    _, U_hist = nls.solve_nonlinear_static(fes, MAT, n_steps=5, tol=1e-10, max_iter=50)
    u_elem = U_hist[-1][[0, 1, 2, 3, 4, 5]]

    closed_top = Y_TOP * M_applied / I
    for xi in (0.0, 0.5, 1.0):
        sigma_top = Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=Y_TOP, xi=xi)
        assert np.isclose(sigma_top, closed_top, rtol=1e-8, atol=0.0), f"xi={xi}"

    # opposite fiber must be the exact negative (pure bending, no axial term)
    sigma_bot = Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=-Y_TOP, xi=0.0)
    assert np.isclose(sigma_bot, -closed_top, rtol=1e-8, atol=0.0)


def test_pure_end_moment_large_magnitude_still_exact():
    """Same check at a MUCH larger applied moment (still exact, unlike
    the transverse-load case below) -- confirms the ~1-2% deviation
    seen under large transverse load (check 4) is specifically a
    transverse-load / large-ROTATION effect, not a general breakdown
    of recover_stress at large internal moments."""
    fes, nodes, elem = _single_element_cantilever()
    M_applied = 800.0   # tip rotation ~2.3 rad -- well past the ~0.28 rad
                         # rotation where check 4's transverse-load case
                         # already shows a ~1-2% deviation
    fes.F[:] = 0.0
    fes.F[5] = M_applied
    _, U_hist = nls.solve_nonlinear_static(fes, MAT, n_steps=30, tol=1e-8, max_iter=100)
    u_elem = U_hist[-1][[0, 1, 2, 3, 4, 5]]
    assert abs(u_elem[5]) > 2.0   # confirm this really is a large rotation

    closed_top = Y_TOP * M_applied / I
    sigma_top = Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=Y_TOP, xi=0.0)
    assert np.isclose(sigma_top, closed_top, rtol=1e-10, atol=0.0)


# =====================================================================
# 3. Small transverse tip load: linear-regime closed form
# =====================================================================
def test_small_transverse_tip_load_matches_linear_closed_form():
    fes, nodes, elem = _single_element_cantilever()
    P = 0.2
    fes.F[:] = 0.0
    fes.F[4] = -P   # downward tip force, y-dof of node 1
    _, U_hist = nls.solve_nonlinear_static(fes, MAT, n_steps=10, tol=1e-8, max_iter=100)
    u_elem = U_hist[-1][[0, 1, 2, 3, 4, 5]]

    theta_tip = u_elem[5]
    assert abs(theta_tip) < 1e-3   # confirm this really is the small-rotation regime

    closed_root = -P * 1.0 * Y_TOP / I   # M(0) = P*L in Euler-Bernoulli theory
    sigma_root = Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=Y_TOP, xi=0.0)
    assert np.isclose(sigma_root, closed_root, rtol=1e-5, atol=0.0)

    # free tip carries ~zero moment
    sigma_tip = Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=Y_TOP, xi=1.0)
    assert abs(sigma_tip) < 1.0


# =====================================================================
# 4. Large transverse tip load: documented, EXPECTED geometric-
#    nonlinearity deviation from the linear closed form
# =====================================================================
def test_large_transverse_tip_load_deviates_from_linear_closed_form_as_expected():
    fes, nodes, elem = _single_element_cantilever()
    P = 200.0
    fes.F[:] = 0.0
    fes.F[4] = -P
    _, U_hist = nls.solve_nonlinear_static(fes, MAT, n_steps=10, tol=1e-8, max_iter=100)
    u_elem = U_hist[-1][[0, 1, 2, 3, 4, 5]]

    theta_tip = u_elem[5]
    assert abs(theta_tip) > 0.1   # confirm this really is a large-rotation case (~0.28 rad)

    closed_root = -P * 1.0 * Y_TOP / I
    sigma_root = Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=Y_TOP, xi=0.0)
    rel_dev = abs(sigma_root - closed_root) / abs(closed_root)

    # EXPECTED: bounded, non-trivial deviation from small-rotation theory
    # (order the tip rotation itself would suggest, ~O(theta^2) ~ a few
    # percent) -- neither near-machine-precision (that would mean the
    # nonlinear solve wasn't actually nonlinear) nor huge (that would
    # mean something is actually broken).
    assert 0.005 < rel_dev < 0.05


# =====================================================================
# 5. Broadcasting: y_fiber / xi arrays vs scalar-call-in-a-loop reference
# =====================================================================
def test_broadcasting_matches_scalar_calls_in_a_loop():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    u_elem = np.array([0.0, 0.0, 0.0, 0.0, -3.0e-4, -0.02])

    y_fibers = np.array([Y_TOP, -Y_TOP])
    sig_arr = Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=y_fibers, xi=0.0)
    sig_loop = np.array([
        Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=y, xi=0.0)
        for y in y_fibers
    ])
    assert np.allclose(sig_arr, sig_loop, rtol=0.0, atol=0.0)
    # opposite fibers of a bent (non-axially-loaded... here there IS some
    # axial coupling from the rotation, but the two should still differ)
    assert sig_arr[0] != sig_arr[1]

    xis = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sig_xi_arr = Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=Y_TOP, xi=xis)
    sig_xi_loop = np.array([
        Beam2DCorotational.recover_stress(nodes, u_elem, MAT, y_fiber=Y_TOP, xi=x)
        for x in xis
    ])
    assert np.allclose(sig_xi_arr, sig_xi_loop, rtol=0.0, atol=0.0)

    # moment varies linearly in xi (no distributed load along the
    # element) -> stress-from-bending varies linearly too, so the
    # xi-swept values must be exactly evenly spaced
    diffs = np.diff(sig_xi_arr)
    assert np.allclose(diffs, diffs[0], rtol=1e-10, atol=1e-6)
