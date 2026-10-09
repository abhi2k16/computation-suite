# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_transient_displacement_control.py -- validates nonlinear_solver.
solve_transient_displacement_control(), the dynamic (Newmark-implicit)
displacement-control driver (Wave 6 item 34, docs/consolidated_future_
roadmap.md: "Arc-length / displacement-control transient variant
(dynamic snap-through)").

Fixture: the same two-bar (von Mises) truss test_nonlinear.py's own
"CHECK 3" uses to validate the STATIC displacement-control driver
against a closed-form P(delta) = (E*A/L0^3)*delta*(h0-delta)*(2h0-delta)
-- reused here (not re-derived) so this file's own checks compare
against the SAME already-trusted ground truth, just in the dynamic
generalization.

1. Quasi-static-limit check (mirrors test_nonlinear_transient.py's own
   TestQuasiStaticLimit for the load-controlled implicit driver): with
   heavy damping and a slow ramp, the dynamic reaction should track the
   static closed-form P(delta) closely -- EXCEPT deliberately restricted
   to delta comfortably away from the limit point itself (delta up to
   ~1.2x delta_peak), because near a limit point the local natural
   frequency of the controlled mode goes to zero (period -> infinity),
   so no finite ramp rate / damping combination reproduces the static
   curve arbitrarily well exactly there -- a real, expected physical
   effect (confirmed directly: an earlier, lighter-damping/faster-ramp
   attempt showed 40%+ deviation growing specifically as the apex
   approached the limit point, while adding damping and restricting the
   comparison window away from the peak collapsed the SAME comparison
   to <1%), not a bug in the driver being papered over by a lenient
   tolerance.

2. Fast dynamic snap-through: a MUCH faster ramp, light damping, driven
   all the way through full inversion (delta: 0 -> 2*h0). Confirms the
   driver runs end-to-end through the whole path (unlike load control,
   which cannot pass the limit point at all) and produces a genuinely
   DIFFERENT reaction-vs-delta trace than the static curve (peak
   reaction measurably higher, from inertia -- this is real dynamic
   snap-through, not a degenerate case that happens to coincide with
   statics).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import elements as elmod
from fea_engine.solver import FESystem
from fea_engine.mesh import Mesh
from fea_engine.damping import RayleighDamping
from fea_engine import nonlinear_solver as nls

A_SPAN = 1.0
H0 = 0.10
E_TRUSS, A_TRUSS = 210e9, 2e-4
L0 = np.sqrt(A_SPAN ** 2 + H0 ** 2)
RHO = 7850.0
DELTA_PEAK = H0 * (3 - np.sqrt(3)) / 3


def _von_mises_truss():
    nodes = np.array([[-A_SPAN, 0.0], [0.0, H0], [A_SPAN, 0.0]])
    elements = np.array([[0, 1], [1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1])
    control_dof = 1 * fes.npn + 1
    fes.assemble_mass(RHO * A_TRUSS * np.eye(2))
    return fes, control_dof


def _P_closed_form(delta):
    return (E_TRUSS * A_TRUSS / L0 ** 3) * delta * (H0 - delta) * (2 * H0 - delta)


class TestQuasiStaticLimit:
    def test_heavy_damping_slow_ramp_matches_closed_form_away_from_limit_point(self):
        fes, control_dof = _von_mises_truss()
        # Damping near the fundamental mode's OWN critical value (alpha
        # = 2*omega_1, omega_1 ~ 1024 rad/s for this truss -- computed
        # directly via solve_modal(), not guessed), not an arbitrarily
        # huge one: found directly that going much PAST critical damping
        # makes the approach to quasi-static equilibrium SLOWER, not
        # faster (the classic overdamped-pendulum effect -- the slow
        # eigenvalue's own time constant grows with damping past
        # critical) -- an earlier attempt at alpha=20000 (~10x critical)
        # showed the tracking error still at 2-5% even a THIRD of the
        # way into the ramp, while critical damping here settles to
        # <1% almost immediately. "Heavily damped" for driving a system
        # toward its quasi-static path means near-critical, not "as
        # much damping as possible."
        fes.assemble_damping(RayleighDamping(alpha=2048.0, beta=0.0))

        delta_max = 1.2 * DELTA_PEAK   # stop short of/just past the peak,
                                        # never touching the limit point itself
        T_ramp = 5.0

        # A smooth (cosine ease-in) ramp, not a plain linear one: a
        # linear ramp has a discontinuous VELOCITY at t=0 (jumping from
        # 0 to the ramp rate instantly), which is a genuine, physically
        # real kick that takes this heavily-overdamped system several
        # step's worth of its own relaxation time to settle out of --
        # confirmed directly (an earlier version of this test used a
        # plain linear ramp and saw a real, physically-explainable
        # startup transient: ~40% error at the very first few points,
        # shrinking smoothly and monotonically to <0.1% by the end of
        # the ramp -- not a bug, just the true dynamic response to an
        # instantaneous velocity onset). A smooth ease-in has ZERO
        # velocity/acceleration at t=0, avoiding that startup transient
        # so the ENTIRE trajectory tracks the quasi-static curve well.
        def u_target_fn(t):
            frac = min(t / T_ramp, 1.0)
            smooth = 0.5 * (1 - np.cos(np.pi * frac))
            return -smooth * delta_max

        n_steps = 100
        dt = T_ramp / n_steps
        t, U_hist, reaction_hist = nls.solve_transient_displacement_control(
            fes, (E_TRUSS, A_TRUSS), control_dof, u_target_fn, T_ramp, dt,
            tol=1e-10, max_iter=50)

        assert np.all(np.isfinite(U_hist))
        assert np.all(np.isfinite(reaction_hist))

        delta = -U_hist[:, control_dof]
        P_cf = _P_closed_form(delta)
        reaction_applied = -reaction_hist
        # Compare only from delta > 0.05*delta_max onward: right near
        # delta=0, P_cf itself is tiny, so even a small absolute
        # discretization error inflates the RELATIVE error hugely --
        # an artifact of the comparison metric near zero, not a real
        # discrepancy (this is the same reason test_nonlinear_transient
        # .py's own quasi-static check compares final states rather
        # than a pointwise trajectory).
        mask = delta > 0.05 * delta_max
        rel_err = np.abs(reaction_applied[mask] - P_cf[mask]) / np.abs(P_cf[mask])
        assert np.max(rel_err) < 0.02   # measured max ~0.5% at critical damping

    def test_reduces_to_static_reaction_at_rest(self):
        # With an explicit at-rest initial condition (v0_control=
        # a0_control=0.0, matching a u_target_fn that itself starts
        # with zero velocity/acceleration at t=0 -- the smooth ease-in
        # profile, not a plain linear ramp, see the test above for why
        # that distinction matters here), the dynamic reaction at the
        # very first recorded step must equal the static F_int[control
        # _dof] - F[control_dof] -- zero here since the truss starts
        # undeformed and unloaded.
        fes, control_dof = _von_mises_truss()
        fes.assemble_damping(RayleighDamping(alpha=1.0, beta=0.0))

        def u_target_fn(t):
            frac = min(t / 1.0, 1.0)
            smooth = 0.5 * (1 - np.cos(np.pi * frac))
            return -0.01 * smooth

        t, U_hist, reaction_hist = nls.solve_transient_displacement_control(
            fes, (E_TRUSS, A_TRUSS), control_dof, u_target_fn, 1.0, 0.01,
            v0_control=0.0, a0_control=0.0, tol=1e-10, max_iter=50)
        assert abs(reaction_hist[0]) < 1e-8


class TestDynamicSnapThrough:
    def test_fast_ramp_traverses_full_inversion_and_differs_from_static(self):
        fes, control_dof = _von_mises_truss()
        fes.assemble_damping(RayleighDamping(alpha=10.0, beta=0.0))

        T_ramp = 0.02   # fast relative to the ~6ms fundamental period --
                         # a genuinely dynamic traverse, not quasi-static

        def u_target_fn(t):
            frac = min(t / T_ramp, 1.0)
            return -frac * 2 * H0

        n_steps = 200
        dt = T_ramp / n_steps
        t, U_hist, reaction_hist = nls.solve_transient_displacement_control(
            fes, (E_TRUSS, A_TRUSS), control_dof, u_target_fn, T_ramp, dt,
            tol=1e-8, max_iter=50)

        assert np.all(np.isfinite(U_hist))
        assert np.all(np.isfinite(reaction_hist))
        assert U_hist.shape == (n_steps + 1, fes.n_dof)

        delta = -U_hist[:, control_dof]
        # reaches (near) full inversion
        assert delta[-1] > 1.9 * H0

        reaction_applied = -reaction_hist
        P_cf = _P_closed_form(delta)
        # a genuinely dynamic run should show a LARGER peak-magnitude
        # reaction than the static curve's own peak (inertia adds to
        # the force needed to drive the apex through as fast as this
        # ramp demands) -- confirms this is real dynamic behavior, not
        # a degenerate near-static coincidence.
        assert np.max(np.abs(reaction_applied)) > np.max(np.abs(P_cf))

        # fixed dofs stay exactly zero at every recorded step
        fixed_idx = sorted(fes.fixed_dofs)
        assert np.all(U_hist[:, fixed_idx] == 0.0)
