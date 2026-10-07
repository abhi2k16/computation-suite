"""
test_iter_state.py -- validation for Module 23: FESystem.init_iter_state()/
update_iter_states(), the mixed-formulation internal-state mechanism
added to nonlinear_solver.py's static drivers to support
Shell4MITCCorotational's von Karman coupling fix (see that class's
"Design history"/D5 investigation).

A minimal TOY element (not a real structural formulation) is used here
DELIBERATELY, to isolate and directly verify the PLUMBING itself
(does init_iter_state() populate self.iter_state correctly, does it
reach internal_force()/tangent_stiffness() as the iter_state= kwarg,
does update_iter_states() call update_iter_state() with the right
elem_coords/u_elem/delta_u_elem/current-iter_state, does the realized
correction accumulate correctly across several iterations) BEFORE
trusting it under the shell's own considerably more complex physics.

The toy element: a 1-DOF-per-node "spring" whose OWN internal force
secretly also depends on an accumulator (iter_state) that just sums up
every delta_u it's ever been given -- since delta_u summed over a
converged Newton loop must equal u_final - u_initial exactly (for
u_initial=0, this is just u_final), the accumulator's final value is a
DIRECT, exactly-checkable fingerprint of every solve_nonlinear_static()
correction actually reaching update_iter_states() with the correct
elem-local delta_u slice.
"""
import numpy as np
import pytest

from fea_engine.elements.base import Element
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls


class _AccumulatingSpring(Element):
    """A 2-node, 1-dof/node linear spring (k*(u2-u1)) -- ordinary linear
    behavior, EXCEPT it also maintains an iter_state accumulator (one
    scalar per node) that update_iter_state() sums delta_u into, purely
    so the test can verify the accumulator ends up EXACTLY equal to the
    final converged displacement (since delta_u summed from u=0 across
    a converged Newton loop telescopes to u_final)."""
    dofs_per_node = 1
    n_nodes = 2

    def stiffness(self, elem_coords, D, thickness=1.0, **kwargs):
        k = D
        return np.array([[k, -k], [-k, k]], dtype=float)

    def internal_force(self, elem_coords, u_elem, D, thickness=1.0, **kwargs):
        return self.stiffness(elem_coords, D) @ u_elem

    def tangent_stiffness(self, elem_coords, u_elem, D, thickness=1.0, **kwargs):
        return self.stiffness(elem_coords, D)

    def init_iter_state(self):
        return np.zeros(2)

    def update_iter_state(self, elem_coords, u_elem, delta_u_elem, D, iter_state, **kwargs):
        return iter_state + delta_u_elem


def _single_spring_system(k=5.0, F=10.0):
    nodes = np.array([[0.0], [1.0]])
    elements = np.array([[0, 1]])
    mesh = Mesh(nodes=nodes, elements=elements, dim=1)
    fes = FESystem(mesh, _AccumulatingSpring())
    fes.fix_dofs([0], [0])
    fes.add_nodal_force([1], 0, F)
    return fes, k


def test_iter_state_defaults_to_none_and_is_noop():
    """Without calling init_iter_state(), solve_nonlinear_static() must
    behave EXACTLY as if the mechanism didn't exist -- fesystem.iter_state
    stays None, and nothing about the converged answer changes."""
    fes, k = _single_spring_system()
    assert fes.iter_state is None
    load_factors, U_hist = nls.solve_nonlinear_static(fes, k, n_steps=4, tol=1e-12)
    assert fes.iter_state is None
    assert abs(U_hist[-1, 1] - 10.0 / k) < 1e-10


def test_iter_state_accumulator_matches_final_displacement():
    """With init_iter_state() called, the toy accumulator must end up
    EXACTLY equal to the converged tip displacement (every accepted
    Newton correction's delta_u, summed from u=0, telescopes to
    u_final) -- a direct, exact fingerprint that update_iter_states()
    is reaching the element with the correct per-element delta_u slice
    on every iteration, not just once or with the wrong indices."""
    fes, k = _single_spring_system()
    fes.init_iter_state()
    assert fes.iter_state is not None
    load_factors, U_hist = nls.solve_nonlinear_static(fes, k, n_steps=4, tol=1e-12)
    u_final = U_hist[-1, 1]
    acc = fes.iter_state[None][0]   # single block (name=None), single element
    assert acc[0] == pytest.approx(0.0, abs=1e-12)     # fixed node, never moves
    assert acc[1] == pytest.approx(u_final, abs=1e-10)  # matches final displacement exactly


def test_iter_state_multi_step_accumulates_across_steps():
    """The accumulator must keep summing across MULTIPLE load steps
    (not reset each step) -- confirms iter_state genuinely persists
    like self.state does, not just within one step's own Newton loop."""
    fes, k = _single_spring_system(F=20.0)
    fes.init_iter_state()
    load_factors, U_hist = nls.solve_nonlinear_static(fes, k, n_steps=10, tol=1e-12)
    u_final = U_hist[-1, 1]
    acc = fes.iter_state[None][0]
    assert acc[1] == pytest.approx(u_final, abs=1e-9)


def test_arc_length_and_static_koiter_newton_also_wire_iter_state():
    """Spot-check the other two drivers touched by Module 23
    (solve_nonlinear_arc_length, solve_nonlinear_static_koiter_newton)
    -- same accumulator fingerprint, on the same toy spring (a linear
    element, so both drivers converge trivially; the point here is
    purely to confirm the plumbing, not to re-validate each driver's
    own already-tested nonlinear behavior)."""
    fes, k = _single_spring_system(F=10.0)
    fes.init_iter_state()
    load_factors, U_hist = nls.solve_nonlinear_arc_length(fes, k, delta_L=0.5, n_steps=3, tol=1e-10)
    u_final = U_hist[-1, 1]
    acc = fes.iter_state[None][0]
    assert acc[1] == pytest.approx(u_final, abs=1e-8)

    fes2, k2 = _single_spring_system(F=10.0)
    fes2.init_iter_state()
    u = nls.solve_nonlinear_static_koiter_newton(fes2, k2, tol=1e-10)
    acc2 = fes2.iter_state[None][0]
    assert acc2[1] == pytest.approx(u[1], abs=1e-8)
