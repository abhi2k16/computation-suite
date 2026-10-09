"""
test_nonlinear_transient.py -- validates nonlinear_solver.solve_nonlinear_transient(),
the Newmark-Newton nonlinear implicit time integrator (Module 15 /
docs/nonlinear_transient_dynamics_roadmap.md).

Three checks, following the roadmap doc's own validation plan (Section 4):

1. Linear-limit regression (the single most important test): on a
   LINEAR element (Beam2DEulerBernoulli, whose internal_force()/
   tangent_stiffness() are the base Element class's default `ke @ u`/
   `ke`, i.e. it does not override the nonlinear-extension-point
   methods), solve_nonlinear_transient() must agree with the existing,
   already-validated FESystem.solve_transient_implicit() to machine
   precision -- this proves the Newmark bookkeeping (predictor
   constants, effective-stiffness assembly, re-factoring every Newton
   iteration) is correct independent of any actual nonlinear behavior.
2. Quasi-static limit: driving the SAME nonlinear (Beam2DCorotational)
   fixture test_nonlinear_beam.py uses, but as a slow transient ramp
   (T_ramp long relative to the structure's fundamental period),
   through solve_nonlinear_transient() converges to the SAME final
   state as the already-validated solve_nonlinear_static() -- confirms
   the two nonlinear solvers agree in the quasi-static limit, not just
   that each independently runs without error.
3. Energy consistency: an undamped, unforced free-vibration case
   (released from a nonlinear static-equilibrium initial condition)
   should conserve total mechanical energy (kinetic + strain, the
   latter from the cumulative trapezoidal work integral of F_int . du,
   since Beam2DCorotational has no separate strain_energy() evaluation)
   to within Newmark's own numerical-damping/drift tolerance over a
   several-period window.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from fea_engine.material import EI_beam, Material, Section
from fea_engine.geometry import generate_mesh
from fea_engine.solver import FESystem
from fea_engine.damping import RayleighDamping
from fea_engine.mesh import Mesh
from fea_engine import elements as elmod
from fea_engine.loads import LoadPattern, TimeHistoryLoad
from fea_engine import nonlinear_solver as nls


def _clamped_free_nonlinear_beam_chain(n_elem=10, L=1.0, E=210e9, A=1e-4, I=8.333e-9, rho=7850.0):
    """The same clamped-FREE (cantilever) Beam2DCorotational chain
    test_nonlinear_beam.py's build_beam_chain() builds -- a real
    geometrically nonlinear model, reused here rather than re-derived,
    so this file's quasi-static/energy checks are validated against
    the same already-trusted element/fixture convention."""
    mat = (E, A, I)
    x = np.linspace(0.0, L, n_elem + 1).reshape(-1, 1)
    nodes = np.hstack([x, np.zeros_like(x)])
    elem_conn = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elem_conn, dim=1)
    beam = elmod.Beam2DCorotational()
    fes = FESystem(mesh, beam)
    fes.fix_dofs([0], [0, 1, 2])
    fes.assemble_mass(rho * A)
    fes.assemble_stiffness(mat)   # only needed for solve_modal()'s eigenproblem below
    fes.assemble_damping(RayleighDamping(alpha=0.0, beta=0.0))   # undamped
    return fes, mesh, beam, mat, n_elem


class TestLinearLimitRegression:
    def test_matches_solve_transient_implicit_to_machine_precision(self):
        n = 10
        mesh = generate_mesh(dim=1, L=1.0, n=n)
        elem = elmod.Beam2DEulerBernoulli()
        sysobj = FESystem(mesh, elem)
        mat = Material(E=210e9, nu=0.3, rho=7800.0)
        sec = Section(A=0.01, I=8.33e-6)
        EI = EI_beam(mat, sec)
        sysobj.assemble_stiffness(EI)
        sysobj.assemble_mass(7800.0 * 0.01)
        sysobj.assemble_damping(RayleighDamping(alpha=2.0, beta=1e-5))
        sysobj.fix_dofs([0], [0, 1])

        pattern = LoadPattern(node_ids=np.array([n]), dof_index=0)
        load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: 1000.0 * min(t / 0.01, 1.0))

        T_total, dt = 0.05, 0.0005
        t1, U1 = sysobj.solve_transient_implicit(load, T_total, dt)
        t2, U2 = nls.solve_nonlinear_transient(sysobj, EI, load, T_total, dt, tol=1e-12, max_iter=30)

        assert np.array_equal(t1, t2)
        rel_diff = np.max(np.abs(U1 - U2)) / np.max(np.abs(U1))
        assert rel_diff < 1e-9

    def test_returns_same_shape_convention_as_solve_transient_implicit(self):
        n = 6
        mesh = generate_mesh(dim=1, L=1.0, n=n)
        elem = elmod.Beam2DEulerBernoulli()
        sysobj = FESystem(mesh, elem)
        mat = Material(E=210e9, nu=0.3, rho=7800.0)
        sec = Section(A=0.01, I=8.33e-6)
        EI = EI_beam(mat, sec)
        sysobj.assemble_stiffness(EI)
        sysobj.assemble_mass(7800.0 * 0.01)
        sysobj.assemble_damping(RayleighDamping(alpha=1.0, beta=1e-5))
        sysobj.fix_dofs([0], [0, 1])

        pattern = LoadPattern(node_ids=np.array([n]), dof_index=0)
        load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: 500.0)
        t, U_hist = nls.solve_nonlinear_transient(sysobj, EI, load, 0.02, 0.001)
        n_steps = round(0.02 / 0.001)
        assert U_hist.shape == (n_steps + 1, sysobj.n_dof)
        assert np.allclose(U_hist[0], 0.0)
        # fixed dofs stay exactly zero at every recorded step
        assert np.all(U_hist[:, [0, 1]] == 0.0)


class TestQuasiStaticLimit:
    def test_slow_ramp_converges_to_solve_nonlinear_static_answer(self):
        fes, mesh, beam, mat, n_elem = _clamped_free_nonlinear_beam_chain()
        freq_hz, _ = fes.solve_modal(n_modes=1)
        period = 1.0 / freq_hz[0]

        P_final = 500.0
        T_ramp = 300.0 * period   # far longer than the fundamental period -> quasi-static
        pattern = LoadPattern(node_ids=np.array([n_elem]), dof_index=1)
        load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: P_final * min(t / T_ramp, 1.0))

        # A coarse-enough dt that each step's load increment stays well
        # above this problem's own floating-point residual noise floor
        # (~1e-7 N, from the beam's E*A/L0 ~ 1e8-scale stiffness terms)
        # relative to tol=1e-8 -- see nonlinear_solver.solve_nonlinear_transient's
        # docstring for why the reference-scale convention is widened
        # for near-zero loads, which this coarse-dt choice avoids
        # needing to rely on in the first place.
        n_steps = 40
        dt = T_ramp / n_steps
        t, U_hist = nls.solve_nonlinear_transient(fes, mat, load, T_ramp, dt, tol=1e-8, max_iter=40)
        u_dynamic = U_hist[-1]

        fes_static = FESystem(mesh, beam)
        fes_static.fix_dofs([0], [0, 1, 2])
        fes_static.add_nodal_force([n_elem], 1, P_final)
        _, U_hist_static = nls.solve_nonlinear_static(fes_static, mat, n_steps=20)
        u_static = U_hist_static[-1]

        rel_diff = np.max(np.abs(u_dynamic - u_static)) / np.max(np.abs(u_static))
        # A generous (0.1%), deliberately non-brittle threshold: a coarse
        # (40-step) ramp still carries a small residual free-vibration
        # ripple sampled at only 40 points, so the exact relative-diff
        # value is not perfectly monotonic in T_ramp/dt -- the point of
        # this check is confirming AGREEMENT IN THE QUASI-STATIC LIMIT
        # (both solvers converge to the same physical answer), not
        # chasing arbitrary numerical precision against solver-specific
        # sampling artifacts.
        assert rel_diff < 1e-3


class TestEnergyConsistency:
    def test_undamped_free_vibration_conserves_energy(self):
        fes, mesh, beam, mat, n_elem = _clamped_free_nonlinear_beam_chain()
        free = fes.free_dofs

        # nonlinear static equilibrium under a tip transverse load -> release
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

        Mff = fes.M[np.ix_(free, free)]
        v_hist = np.gradient(U_hist[:, free], dt, axis=0, edge_order=2)

        KE = np.zeros(len(t))
        SE = np.zeros(len(t))   # relative to the initial state, not absolute -- a
                                 # constant offset from true strain energy, which
                                 # cancels out of a CONSERVATION check
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
        energy_scale = KE.max() - KE.min()   # the problem's own natural energy scale
        rel_drift = (E_total.max() - E_total.min()) / energy_scale
        assert rel_drift < 0.02   # < 2% drift over 5 periods -- Newmark on a
                                   # geometrically nonlinear system is not exactly
                                   # energy-conserving in general (a known property,
                                   # not a bug), but should stay small over this window
