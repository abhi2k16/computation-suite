"""
test_critical_timestep_local.py -- validates FESystem.critical_timestep_local(),
the cheap O(n_elements) local CFL estimate (Wave 6 item 32, docs/
consolidated_future_roadmap.md: "Δt_crit = α·min_e(ℓ_e/c_e)", the
book's cheaper alternative to critical_timestep()'s exact but
O(n_dof^3) global spectral estimate).

Uses TrussTL2D (a 2-node axial bar element) as the validation fixture,
not a beam: a uniform bar-chain's exact central-difference stability
limit for LUMPED mass is the textbook closed form dt_crit = h/c (h =
element length, c = sqrt(E/rho), the axial wave speed) -- exactly the
physics critical_timestep_local()'s l_e/c_e formula assumes, so this
element lets the local estimate be checked against an independent,
analytically-known ground truth rather than just "runs without
error". (A bending-dominated element like Beam2DEulerBernoulli is
deliberately NOT used here: its critical timestep is governed by
flexural, not axial-bar, wave physics -- a very different h^2 relation
-- so l_e/c_e is not the right formula for it in the first place; that
mismatch is a real, expected limitation of the naive local-CFL
heuristic for bending elements, not something this test should paper
over by picking a formula that happens to match.)

1. Exact-formula check: critical_timestep_local(wave_speed=c, alpha=1.0)
   on a uniform TrussTL2D chain must equal h/c to near machine
   precision (h = L/n, the minimum pairwise node distance the method
   computes internally).
2. Stability boundary check: driving solve_transient_explicit_nonlinear()
   at dt slightly BELOW the exact critical value stays bounded/finite;
   at dt slightly ABOVE it blows up (NaN) -- confirms the estimate is a
   real, physically meaningful stability boundary, not just a number
   that happens to satisfy check 1's algebra.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from fea_engine.solver import FESystem
from fea_engine.mesh import Mesh
from fea_engine import elements as elmod
from fea_engine.loads import LoadPattern, TimeHistoryLoad
from fea_engine import nonlinear_solver as nls


def _truss_chain(n=10, L=1.0, E=210e9, A=1e-4, rho=7850.0):
    x = np.linspace(0.0, L, n + 1).reshape(-1, 1)
    nodes = np.hstack([x, np.zeros_like(x)])
    elem_conn = np.array([[i, i + 1] for i in range(n)], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elem_conn, dim=1)
    truss = elmod.TrussTL2D()
    fes = FESystem(mesh, truss)
    fes.fix_dofs([0], [0, 1])
    fes.assemble_lumped_mass(rho * A * np.eye(2))
    mat = (E, A)
    return fes, mat, n, L, E, A, rho


class TestExactFormula:
    def test_matches_analytic_bar_chain_cfl_limit(self):
        fes, mat, n, L, E, A, rho = _truss_chain()
        c = np.sqrt(E / rho)
        h = L / n

        dt_local = fes.critical_timestep_local(wave_speed=c, alpha=1.0)
        rel_err = abs(dt_local - h / c) / (h / c)
        assert rel_err < 1e-12

    def test_alpha_scales_linearly(self):
        fes, mat, n, L, E, A, rho = _truss_chain()
        c = np.sqrt(E / rho)
        dt_full = fes.critical_timestep_local(wave_speed=c, alpha=1.0)
        dt_scaled = fes.critical_timestep_local(wave_speed=c, alpha=0.5)
        assert abs(dt_scaled - 0.5 * dt_full) < 1e-15 * dt_full

    def test_per_block_wave_speed_dict_matches_scalar(self):
        # A single-block mesh: passing {None: c} (the block name FESystem
        # uses for an un-named single-block mesh) must agree with the
        # plain-scalar call -- same convention _per_block_arg() already
        # establishes for D/mat/rho throughout this class.
        fes, mat, n, L, E, A, rho = _truss_chain()
        c = np.sqrt(E / rho)
        dt_scalar = fes.critical_timestep_local(wave_speed=c, alpha=1.0)
        block_name = fes._blocks[0][0]
        dt_dict = fes.critical_timestep_local(wave_speed={block_name: c}, alpha=1.0)
        assert dt_scalar == dt_dict


class TestStabilityBoundary:
    def test_below_critical_stays_bounded(self):
        fes, mat, n, L, E, A, rho = _truss_chain()
        c = np.sqrt(E / rho)
        dt_exact = (L / n) / c
        dt = 0.9 * dt_exact

        pattern = LoadPattern(node_ids=np.array([n]), dof_index=0)
        load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: 5000.0)
        t, U = nls.solve_transient_explicit_nonlinear(fes, mat, load, 200 * dt, dt)

        assert np.all(np.isfinite(U))
        assert np.max(np.abs(U)) < 1.0   # sane physical displacement scale

    def test_above_critical_blows_up(self):
        fes, mat, n, L, E, A, rho = _truss_chain()
        c = np.sqrt(E / rho)
        dt_exact = (L / n) / c
        dt = 1.3 * dt_exact

        pattern = LoadPattern(node_ids=np.array([n]), dof_index=0)
        load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: 5000.0)
        with np.errstate(over="ignore", invalid="ignore"):
            t, U = nls.solve_transient_explicit_nonlinear(fes, mat, load, 200 * dt, dt)

        assert not np.all(np.isfinite(U))
