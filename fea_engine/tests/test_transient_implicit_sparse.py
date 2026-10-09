# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_transient_implicit_sparse.py -- Wave 16 item 134 (docs/
consolidated_future_roadmap.md, source: TensorMesh's `Time Integration`
documentation page): validates FESystem.solve_transient_implicit()'s
new sparse=True support and its dense-path switch from an explicit
matrix inversion (np.linalg.inv()) to a one-time LU factorization
(scipy.linalg.lu_factor()/lu_solve()) reused every time step.

Before this item, solve_transient_implicit() unconditionally called
np.linalg.inv() on Kff+a0c*Mff+a1c*Cff, which silently assumed a dense
Kff/Mff/Cff -- calling it on a sparse=True FESystem raised
`LinAlgError: 0-dimensional array given. Array must be at least
two-dimensional` (numpy trying to treat a scipy.sparse matrix as a
0-d array). Every test in this file that constructs a sparse=True
FESystem and calls solve_transient_implicit() successfully is itself
a regression check against that crash -- it could not have been
written before this item.

Decisive checks throughout: agreement between the new sparse and dense
paths (not just "both run"), agreement of the new dense lu_factor()
path against an INDEPENDENTLY reimplemented old-style np.linalg.inv()
reference (not just against itself), and a physically meaningful
free-vibration energy-decay check (not just formula-matching) to catch
a plausible-but-wrong implementation the machine-precision numerical
checks alone might miss.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import Quad4PlaneStress, FESystem, Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh
from fea_engine.damping import RayleighDamping


def _steel():
    return Material(E=2.1e11, nu=0.3, rho=7850.0)


def _build(sparse, nx=6, ny=6):
    mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=nx, ny=ny)
    D = D_plane_stress(_steel())
    fs = FESystem(mesh, Quad4PlaneStress(), sparse=sparse)
    fs.assemble_stiffness(D)
    fs.assemble_mass(_steel().rho * np.eye(2))
    fs.assemble_damping(RayleighDamping(alpha=0.5, beta=1e-4))
    left = mesh.nodes_on_line(axis=0, value=0.0)
    for n in left:
        fs.fix_dofs([n], [0, 1])
    return fs, mesh


class _RampLoad:
    """A simple ramped tip load -- force_at(t, n_dof, npn) convention."""

    def __init__(self, dof, magnitude, ramp_time):
        self.dof, self.magnitude, self.ramp_time = dof, magnitude, ramp_time

    def force_at(self, t, n_dof, npn):
        F = np.zeros(n_dof)
        F[self.dof] = min(t / self.ramp_time, 1.0) * self.magnitude
        return F


def _independent_old_style_reference(fs, load, T_total, dt, beta=0.25, gamma=0.5,
                                      u0=None, v0=None):
    """Reimplements the PRE-item-134 algorithm from scratch (explicit
    np.linalg.inv(), dense arrays only) as an independent reference --
    not a call into the module under test."""
    free = fs.free_dofs
    n_steps = int(round(T_total / dt))
    Mff = np.asarray(fs.M[np.ix_(free, free)].toarray() if fs.sparse
                      else fs.M[np.ix_(free, free)])
    Cff = np.asarray(fs.C[np.ix_(free, free)].toarray() if fs.sparse
                      else fs.C[np.ix_(free, free)])
    Kff = np.asarray(fs.K[np.ix_(free, free)].toarray() if fs.sparse
                      else fs.K[np.ix_(free, free)])
    a0c = 1 / (beta * dt**2); a1c = gamma / (beta * dt); a2c = 1 / (beta * dt)
    a3c = 1 / (2 * beta) - 1; a4c = gamma / beta - 1; a5c = dt / 2 * (gamma / beta - 2)
    a6c = dt * (1 - gamma); a7c = dt * gamma
    Keff_inv = np.linalg.inv(Kff + a0c * Mff + a1c * Cff)
    d = np.zeros(len(free)) if u0 is None else np.asarray(u0)[free].copy()
    v = np.zeros(len(free)) if v0 is None else np.asarray(v0)[free].copy()

    def F_free(tt):
        return load.force_at(tt, fs.n_dof, fs.npn)[free]

    a = np.linalg.solve(Mff, F_free(0.0) - Cff @ v - Kff @ d)
    U_hist = np.zeros((n_steps + 1, fs.n_dof))
    U_hist[0, free] = d
    for step in range(n_steps):
        rhs = (F_free((step + 1) * dt) + Mff @ (a0c * d + a2c * v + a3c * a)
               + Cff @ (a1c * d + a4c * v + a5c * a))
        d_new = Keff_inv @ rhs
        a_new = a0c * (d_new - d) - a2c * v - a3c * a
        v_new = v + a6c * a + a7c * a_new
        d, v, a = d_new, v_new, a_new
        U_hist[step + 1, free] = d
    return U_hist


class TestSparseSupport:
    """Every test here constructs sparse=True and calls
    solve_transient_implicit() -- this crashed unconditionally before
    item 134 (LinAlgError from np.linalg.inv() on a scipy.sparse
    matrix), so successfully running at all is itself the regression
    check for the fixed crash."""

    def test_sparse_runs_without_error(self):
        fs, mesh = _build(sparse=True)
        tip = [n for n in mesh.nodes_on_line(axis=0, value=1.0)][0]
        load = _RampLoad(fs._global_dofs([tip])[1], -5000.0, ramp_time=0.01)
        t, U = fs.solve_transient_implicit(load, T_total=0.02, dt=0.001)
        assert U.shape == (21, fs.n_dof)
        assert np.all(np.isfinite(U))

    def test_sparse_matches_dense(self):
        fs_dense, mesh_d = _build(sparse=False)
        fs_sparse, mesh_s = _build(sparse=True)
        tip_d = [n for n in mesh_d.nodes_on_line(axis=0, value=1.0)][0]
        tip_s = [n for n in mesh_s.nodes_on_line(axis=0, value=1.0)][0]
        load_d = _RampLoad(fs_dense._global_dofs([tip_d])[1], -5000.0, ramp_time=0.01)
        load_s = _RampLoad(fs_sparse._global_dofs([tip_s])[1], -5000.0, ramp_time=0.01)

        t_d, U_dense = fs_dense.solve_transient_implicit(load_d, T_total=0.05, dt=0.001)
        t_s, U_sparse = fs_sparse.solve_transient_implicit(load_s, T_total=0.05, dt=0.001)

        assert np.allclose(U_dense, U_sparse, atol=1e-10, rtol=1e-8)

    def test_sparse_matches_independent_reference(self):
        fs, mesh = _build(sparse=True)
        tip = [n for n in mesh.nodes_on_line(axis=0, value=1.0)][0]
        load = _RampLoad(fs._global_dofs([tip])[1], -3000.0, ramp_time=0.005)
        t, U = fs.solve_transient_implicit(load, T_total=0.03, dt=0.0005)
        U_ref = _independent_old_style_reference(fs, load, T_total=0.03, dt=0.0005)
        assert np.allclose(U, U_ref, atol=1e-8, rtol=1e-8)


class TestDenseMatchesIndependentReference:
    """Confirms the switch from np.linalg.inv() to lu_factor()/
    lu_solve() didn't change the answer -- checked against an
    independently reimplemented old-style algorithm, not against the
    module's own prior output."""

    def test_ramped_load_matches_reference(self):
        fs, mesh = _build(sparse=False)
        tip = [n for n in mesh.nodes_on_line(axis=0, value=1.0)][0]
        load = _RampLoad(fs._global_dofs([tip])[1], -5000.0, ramp_time=0.01)
        t, U = fs.solve_transient_implicit(load, T_total=0.05, dt=0.001)
        U_ref = _independent_old_style_reference(fs, load, T_total=0.05, dt=0.001)
        assert np.allclose(U, U_ref, atol=1e-9, rtol=1e-8)

    @pytest.mark.parametrize("beta,gamma", [(0.25, 0.5), (0.3025, 0.6)])
    def test_alternate_beta_gamma_matches_reference(self, beta, gamma):
        fs, mesh = _build(sparse=False)
        tip = [n for n in mesh.nodes_on_line(axis=0, value=1.0)][0]
        load = _RampLoad(fs._global_dofs([tip])[1], -2000.0, ramp_time=0.008)
        t, U = fs.solve_transient_implicit(load, T_total=0.04, dt=0.001,
                                            beta=beta, gamma=gamma)
        U_ref = _independent_old_style_reference(fs, load, T_total=0.04, dt=0.001,
                                                   beta=beta, gamma=gamma)
        assert np.allclose(U, U_ref, atol=1e-8, rtol=1e-8)

    def test_nonzero_initial_conditions_match_reference(self):
        fs, mesh = _build(sparse=False)
        tip = [n for n in mesh.nodes_on_line(axis=0, value=1.0)][0]
        load = _RampLoad(fs._global_dofs([tip])[1], -1000.0, ramp_time=0.005)
        rng = np.random.default_rng(0)
        u0 = np.zeros(fs.n_dof)
        v0 = np.zeros(fs.n_dof)
        u0[fs.free_dofs] = rng.normal(scale=1e-5, size=len(fs.free_dofs))
        v0[fs.free_dofs] = rng.normal(scale=1e-4, size=len(fs.free_dofs))
        t, U = fs.solve_transient_implicit(load, T_total=0.02, dt=0.001, u0=u0, v0=v0)
        U_ref = _independent_old_style_reference(fs, load, T_total=0.02, dt=0.001,
                                                   u0=u0, v0=v0)
        assert np.allclose(U, U_ref, atol=1e-8, rtol=1e-8)


class TestPhysicalSanity:
    """A decisive physical check independent of the numerical
    cross-checks above: an undamped-ish free-vibration decay under
    Rayleigh damping should monotonically lose energy over many
    periods, for both the sparse and dense paths -- catches a
    plausible-but-wrong implementation (e.g. a sign error in the new
    solve dispatch) that agreement-with-a-buggy-reference could not."""

    @pytest.mark.parametrize("sparse", [False, True])
    def test_free_decay_loses_energy(self, sparse):
        fs, mesh = _build(sparse=sparse, nx=4, ny=4)

        class ZeroLoad:
            def force_at(self, t, n_dof, npn):
                return np.zeros(n_dof)

        u0 = np.zeros(fs.n_dof)
        tip = [n for n in mesh.nodes_on_line(axis=0, value=1.0)][0]
        u0[fs._global_dofs([tip])[1]] = 0.01
        t, U = fs.solve_transient_implicit(ZeroLoad(), T_total=0.2, dt=0.001, u0=u0)

        tip_dof = fs._global_dofs([tip])[1]
        peak_early = np.max(np.abs(U[:50, tip_dof]))
        peak_late = np.max(np.abs(U[-50:, tip_dof]))
        assert peak_late < peak_early, (
            f"expected amplitude decay under Rayleigh damping, got "
            f"early peak {peak_early:.3e} <= late peak {peak_late:.3e}")
