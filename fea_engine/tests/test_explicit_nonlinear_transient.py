# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_explicit_nonlinear_transient.py -- validates nonlinear_solver.
solve_transient_explicit_nonlinear(), the nonlinear central-difference
explicit time integrator (Wave 6 item 31, docs/consolidated_future_
roadmap.md).

Mirrors test_nonlinear_transient.py's own validation plan for the
IMPLICIT nonlinear driver, adapted to the explicit method:

1. Linear-limit regression (the single most important test): on a
   LINEAR element (Beam2DEulerBernoulli), solve_transient_explicit_
   nonlinear() must agree with the existing, already-validated
   FESystem.solve_transient_explicit() to machine precision -- proves
   the central-difference bookkeeping (diag_fast path AND the general-C
   fallback path) is correct independent of any actual nonlinear
   behavior, exactly the role test_nonlinear_transient.py's own linear-
   limit test plays for the implicit driver.
2. Shape/convention check: return shape, zero initial state, fixed dofs
   stay exactly zero.
3. Energy consistency: an undamped, unforced free-vibration case
   (Beam2DCorotational, released from a nonlinear static-equilibrium
   initial condition -- the SAME fixture test_nonlinear_transient.py's
   TestEnergyConsistency uses) should conserve total mechanical energy
   to within a small drift over several periods -- confirms the
   nonlinear internal-force substitution is physically sound, not just
   "runs without error."
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine.material import EI_beam, Material, Section
from fea_engine.geometry import generate_mesh
from fea_engine.solver import FESystem
from fea_engine.damping import RayleighDamping
from fea_engine.mesh import Mesh, line_mesh
from fea_engine import elements as elmod
from fea_engine.loads import LoadPattern, TimeHistoryLoad
from fea_engine import nonlinear_solver as nls


def _clamped_free_nonlinear_beam_chain(n_elem=10, L=1.0, E=210e9, A=1e-4, I=8.333e-9, rho=7850.0):
    """Identical fixture to test_nonlinear_transient.py's own helper of
    the same name -- reused (not re-derived) so this file's energy
    check is validated against the same already-trusted element/
    fixture convention as the implicit driver's own energy test."""
    mat = (E, A, I)
    x = np.linspace(0.0, L, n_elem + 1).reshape(-1, 1)
    nodes = np.hstack([x, np.zeros_like(x)])
    elem_conn = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elem_conn, dim=1)
    beam = elmod.Beam2DCorotational()
    fes = FESystem(mesh, beam)
    fes.fix_dofs([0], [0, 1, 2])
    fes.assemble_mass(rho * A)
    fes.assemble_lumped_mass(rho * A)
    fes.assemble_stiffness(mat)   # only needed for solve_modal()'s eigenproblem below
    fes.assemble_damping(RayleighDamping(alpha=0.0, beta=0.0))   # undamped
    return fes, mesh, beam, mat, n_elem


class TestLinearLimitRegression:
    def _build_linear_beam(self, damping):
        n = 10
        # Uses line_mesh() (1-column x-coordinates), not generate_mesh()
        # -- matching examples/main.py's own working assemble_lumped_
        # mass() setup for Beam2DEulerBernoulli: the base Element.
        # lumped_mass() HRZ-rescaling pass reuses the generic isoparametric
        # jacobian()/gauss_product() machinery (see elements/base.py),
        # which needs elem_coords in the same coordinate convention
        # gauss_product(self.gauss_order, self.dim=1) expects -- found
        # directly (generate_mesh(dim=1,...)'s 2-column, x/y-padded node
        # array broke that internal jacobian() call, line_mesh()'s
        # single-column array does not).
        mesh = line_mesh(1.0, n=n)
        elem = elmod.Beam2DEulerBernoulli()
        sysobj = FESystem(mesh, elem)
        mat = Material(E=210e9, nu=0.3, rho=7800.0)
        sec = Section(A=0.01, I=8.33e-6)
        EI = EI_beam(mat, sec)
        sysobj.assemble_stiffness(EI)
        sysobj.assemble_mass(7800.0 * 0.01)
        sysobj.assemble_lumped_mass(7800.0 * 0.01)
        sysobj.assemble_damping(damping)
        sysobj.fix_dofs([0], [0, 1])
        return sysobj, EI, n

    def test_matches_solve_transient_explicit_to_machine_precision_diag_fast(self):
        # RayleighDamping with beta=0 -> C is mass-proportional -> diagonal
        # -> exercises the diag_fast path in BOTH drivers.
        sysobj, EI, n = self._build_linear_beam(RayleighDamping(alpha=2.0, beta=0.0))
        pattern = LoadPattern(node_ids=np.array([n]), dof_index=0)
        load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: 1000.0 * min(t / 0.01, 1.0))

        dt_crit = sysobj.critical_timestep()
        dt = 0.2 * dt_crit
        T_total = 200 * dt

        t1, U1 = sysobj.solve_transient_explicit(load, T_total, dt)
        t2, U2 = nls.solve_transient_explicit_nonlinear(sysobj, EI, load, T_total, dt)

        assert np.array_equal(t1, t2)
        rel_diff = np.max(np.abs(U1 - U2)) / np.max(np.abs(U1))
        assert rel_diff < 1e-9

    def test_matches_solve_transient_explicit_to_machine_precision_general_C(self):
        # RayleighDamping with beta!=0 -> C = alpha*M + beta*K is NOT
        # diagonal -> exercises the general (linear-solve) fallback path
        # in BOTH drivers.
        sysobj, EI, n = self._build_linear_beam(RayleighDamping(alpha=1.0, beta=1e-6))
        pattern = LoadPattern(node_ids=np.array([n]), dof_index=0)
        load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: 800.0 * min(t / 0.01, 1.0))

        dt_crit = sysobj.critical_timestep()
        dt = 0.2 * dt_crit
        T_total = 150 * dt

        t1, U1 = sysobj.solve_transient_explicit(load, T_total, dt)
        t2, U2 = nls.solve_transient_explicit_nonlinear(sysobj, EI, load, T_total, dt)

        assert np.array_equal(t1, t2)
        rel_diff = np.max(np.abs(U1 - U2)) / np.max(np.abs(U1))
        assert rel_diff < 1e-9

    def test_returns_same_shape_convention_as_solve_transient_explicit(self):
        sysobj, EI, n = self._build_linear_beam(RayleighDamping(alpha=1.0, beta=0.0))
        pattern = LoadPattern(node_ids=np.array([n]), dof_index=0)
        load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: 500.0)

        dt_crit = sysobj.critical_timestep()
        dt = 0.2 * dt_crit
        n_steps = 50
        T_total = n_steps * dt
        t, U_hist = nls.solve_transient_explicit_nonlinear(sysobj, EI, load, T_total, dt)

        assert U_hist.shape == (n_steps + 1, sysobj.n_dof)
        assert np.allclose(U_hist[0], 0.0)
        assert np.all(U_hist[:, [0, 1]] == 0.0)


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
        dt_crit = fes.critical_timestep()
        # Explicit stability requires dt well below dt_crit; also want
        # enough steps per period to resolve the motion for the energy
        # integral below -- take whichever constraint is tighter.
        dt = min(0.4 * dt_crit, period / 400)
        T_total = 5 * period

        t, U_hist = nls.solve_transient_explicit_nonlinear(fes, mat, zero_load, T_total, dt, u0=u0)

        Mff = fes.M_lumped[np.ix_(free, free)]
        v_hist = np.gradient(U_hist[:, free], dt, axis=0, edge_order=2)

        KE = np.zeros(len(t))
        SE = np.zeros(len(t))
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
        energy_scale = KE.max() - KE.min()
        rel_drift = (E_total.max() - E_total.min()) / energy_scale
        # Explicit central difference is also not exactly energy-
        # conserving on a nonlinear system; use the same generous,
        # non-brittle threshold as the implicit driver's own check.
        assert rel_drift < 0.02
