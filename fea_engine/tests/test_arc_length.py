"""
test_arc_length.py -- validation for Module 19 (general-purpose
extensions roadmap Phase 5): nonlinear_solver.solve_nonlinear_arc_length()
(Crisfield cylindrical arc-length continuation).

Reuses the SAME von Mises (two-bar) snap-through truss benchmark
test_nonlinear.py already validates solve_nonlinear_static() and
solve_nonlinear_displacement_control() against -- same nodes, same
closed-form P(delta) = (E*A/L0^3) * delta * (h0-delta) * (2*h0-delta)
(derived via total potential energy; see nonlinear_solver.py /
test_nonlinear.py for the full derivation) -- so all three drivers are
checked against ONE shared ground truth, not three separately-derived
closed forms that might quietly disagree with each other.

Three lines of evidence:

1. test_load_control_fails_past_limit_point -- the genuine regression/
   contrast test the roadmap's own validation plan calls for:
   solve_nonlinear_static() (load control) is confirmed to fail
   EXACTLY as its own docstring predicts, at a load past the limit
   point on this truss -- establishing that this problem really does
   need something other than plain load control, not just asserting
   it in a docstring.
2. test_arc_length_traces_full_path_matching_closed_form -- the
   decisive check: arc-length continuation, with NO knowledge of
   where the limit points are, traces delta from 0 out past the fully-
   inverted state (delta=2*h0) and matches the closed-form P(delta) at
   every recorded point along the way, through BOTH sign changes of
   dP/ddelta (the ascending branch's peak, and the descending
   branch's trough) -- this is "the reversal" the roadmap's validation
   plan asks to see traced, not just approached.
3. test_arc_length_matches_displacement_control_where_both_apply -- an
   independent cross-check between two DIFFERENT solution strategies
   on the same problem: displacement control (already validated
   against the closed form in test_nonlinear.py) and arc-length should
   land on the same equilibrium points, interpolated to a shared delta
   grid, wherever their paths overlap.
"""
import numpy as np
import pytest

from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls


A_HALFSPAN, H0 = 1.0, 0.10
E, A = 210e9, 2e-4
L0 = np.sqrt(A_HALFSPAN ** 2 + H0 ** 2)


def _von_mises_truss():
    nodes = np.array([[-A_HALFSPAN, 0.0], [0.0, H0], [A_HALFSPAN, 0.0]])
    elements = np.array([[0, 1], [1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1])
    control_dof = 1 * fes.npn + 1   # apex node, y-dof
    return fes, control_dof


def _P_closed_form(delta):
    return (E * A / L0 ** 3) * delta * (H0 - delta) * (2 * H0 - delta)


def test_load_control_fails_past_limit_point():
    delta_peak = H0 * (3 - np.sqrt(3)) / 3
    P_peak = _P_closed_form(delta_peak)

    fes, _ = _von_mises_truss()
    fes.add_nodal_force([1], 1, -1.1 * P_peak)   # past the limit point

    with pytest.raises(RuntimeError, match="limit point"):
        nls.solve_nonlinear_static(fes, (E, A), n_steps=20, tol=1e-12)


def test_arc_length_traces_full_path_matching_closed_form():
    fes, control_dof = _von_mises_truss()
    fes.add_nodal_force([1], 1, -1.0)   # reference load direction/pattern only

    delta_L = 0.01
    load_factors, U_hist = nls.solve_nonlinear_arc_length(
        fes, (E, A), delta_L, n_steps=60, tol=1e-10, max_iter=60)

    delta_fe = -U_hist[:, control_dof]
    P_fe = load_factors   # F_ref at control_dof is -1, so lambda*(-1) downward
                           # force equals +lambda in the delta-positive-downward sign convention
    P_cf = _P_closed_form(delta_fe)

    # sanity: the path actually goes past full inversion (delta > 2*H0),
    # i.e. genuinely continues through BOTH limit points, not just up to one
    assert delta_fe.max() > 2 * H0

    scale = max(np.max(np.abs(P_cf)), 1e-30)
    significant = np.abs(P_cf) > 1e-4 * scale
    rel_err = np.abs(P_fe - P_cf)[significant] / np.abs(P_cf)[significant]
    assert np.max(rel_err) < 1e-6


def test_arc_length_matches_displacement_control_where_both_apply():
    fes_al, control_dof = _von_mises_truss()
    fes_al.add_nodal_force([1], 1, -1.0)
    load_factors, U_al = nls.solve_nonlinear_arc_length(
        fes_al, (E, A), delta_L=0.01, n_steps=30, tol=1e-10, max_iter=60)
    delta_al = -U_al[:, control_dof]

    fes_dc, control_dof2 = _von_mises_truss()
    assert control_dof2 == control_dof
    delta_dc = np.linspace(0.0, delta_al.max(), 50)
    U_dc, _ = nls.solve_nonlinear_displacement_control(
        fes_dc, (E, A), control_dof, -delta_dc, tol=1e-12, max_iter=50)

    # both solvers report the SAME structure's total potential energy
    # response; compare via the reaction/internal force at the control
    # dof, interpolated onto the arc-length solver's own delta samples
    # (skip the very first, delta=0, point -- interpolation there is
    # exact by construction, not a meaningful check).
    from fea_engine.material import Material  # noqa: F401  (mat not needed directly)
    F_int_dc = np.array([
        fes_dc.assemble_internal_force(U_dc[i], (E, A))[control_dof]
        for i in range(len(delta_dc))
    ])
    P_dc = -F_int_dc

    mask = (delta_al > 1e-6) & (delta_al <= delta_dc.max())
    P_dc_interp = np.interp(delta_al[mask], delta_dc, P_dc)
    P_al = load_factors[mask]

    scale = max(np.max(np.abs(P_dc)), 1e-30)
    rel_err = np.abs(P_al - P_dc_interp) / scale
    assert np.max(rel_err) < 5e-3   # interpolation onto different sample grids,
    # so a looser tolerance than the closed-form comparisons above is expected
