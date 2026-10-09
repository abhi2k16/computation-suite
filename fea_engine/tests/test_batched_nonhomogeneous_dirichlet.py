# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_batched_nonhomogeneous_dirichlet.py -- Wave 16 item 133: closes a
gap item 109's solve_static_batched() (batched_solve.py) created
relative to item 131 (non-homogeneous Dirichlet BCs). solve_static_
batched() predates item 131 and, until this item, silently treated
EVERY fixed DOF as zero regardless of fix_dofs(..., value=...) -- a
batched load-sweep on a mesh with a nonzero prescribed displacement got
a silently wrong answer at every column.

Decisive checks throughout: agreement with an independent per-column
solve_static() reference (not just "the prescribed value shows up
somewhere"), across every column of the batch (not just column 0), and
confirmed backward compatibility (an all-zero-BC batch is bit-for-bit
unaffected by this item, matching solve_static()'s own IDENTICAL-cost-
when-unused convention).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import Quad4PlaneStress, FESystem, Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh
from fea_engine.batched_solve import solve_static_batched


def _steel():
    return Material(E=2.1e11, nu=0.3, rho=7850.0)


def _build(nonzero_value=None, sparse=False):
    mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=6, ny=3)
    D = D_plane_stress(_steel())
    fs = FESystem(mesh, Quad4PlaneStress(), sparse=sparse)
    fs.assemble_stiffness(D)
    left = mesh.nodes_on_line(axis=0, value=0.0)
    right = mesh.nodes_on_line(axis=0, value=2.0)
    for n in left:
        fs.fix_dofs([n], [0, 1])
    if nonzero_value is not None:
        for n in right:
            fs.fix_dofs([n], [0], value=nonzero_value)
    return fs, mesh, right


def _random_load_batch(n_dof, n_batch, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_dof, n_batch)) * 1000.0


class TestNonHomogeneousDirichletHonoredInBatch:
    def test_matches_independent_per_column_solve_static(self):
        fs, mesh, right = _build(nonzero_value=0.01)
        n_batch = 6
        F_batch = _random_load_batch(fs.n_dof, n_batch)

        U_batch = solve_static_batched(fs, F_batch)

        U_ref = np.zeros_like(U_batch)
        for i in range(n_batch):
            fs_i, _, _ = _build(nonzero_value=0.01)
            fs_i.F[:] = F_batch[:, i]
            U_ref[:, i] = fs_i.solve_static()

        assert np.allclose(U_batch, U_ref, atol=1e-8, rtol=1e-8)

    def test_prescribed_value_appears_in_every_column(self):
        fs, mesh, right = _build(nonzero_value=0.015)
        n_batch = 4
        F_batch = _random_load_batch(fs.n_dof, n_batch)
        U_batch = solve_static_batched(fs, F_batch)
        right_x = [fs._global_dofs([n])[0] for n in right]
        assert np.allclose(U_batch[right_x, :], 0.015, atol=1e-10)

    def test_sparse_matches_independent_per_column_solve_static(self):
        fs, mesh, right = _build(nonzero_value=0.02, sparse=True)
        n_batch = 5
        F_batch = _random_load_batch(fs.n_dof, n_batch, seed=1)

        U_batch = solve_static_batched(fs, F_batch)

        U_ref = np.zeros_like(U_batch)
        for i in range(n_batch):
            fs_i, _, _ = _build(nonzero_value=0.02, sparse=True)
            fs_i.F[:] = F_batch[:, i]
            U_ref[:, i] = fs_i.solve_static()

        assert np.allclose(U_batch, U_ref, atol=1e-8, rtol=1e-8)

    def test_single_column_batch_matches_solve_static(self):
        """n_batch=1 is the degenerate case a caller might hit first --
        confirm it isn't special-cased incorrectly."""
        fs, mesh, right = _build(nonzero_value=0.01)
        F = np.zeros(fs.n_dof)
        tip = mesh.nodes_on_line(axis=0, value=2.0)
        for n in tip:
            F[fs._global_dofs([n])[1]] += -500.0 / len(tip)
        U_batch = solve_static_batched(fs, F[:, None])

        fs_ref, _, _ = _build(nonzero_value=0.01)
        fs_ref.F[:] = F
        U_ref = fs_ref.solve_static()

        assert np.allclose(U_batch[:, 0], U_ref, atol=1e-8, rtol=1e-8)


class TestBackwardCompatibility:
    """An all-homogeneous (value=0.0 / no value=) BC batch must be
    unaffected by this item -- same convention as every other item in
    this Wave (128-132): IDENTICAL behavior when the new feature isn't
    used."""

    def test_zero_bc_batch_matches_pre_item_133_reference(self):
        fs, mesh, right = _build(nonzero_value=None)
        n_batch = 5
        F_batch = _random_load_batch(fs.n_dof, n_batch, seed=2)

        U_batch = solve_static_batched(fs, F_batch)

        U_ref = np.zeros_like(U_batch)
        for i in range(n_batch):
            fs_i, _, _ = _build(nonzero_value=None)
            fs_i.F[:] = F_batch[:, i]
            U_ref[:, i] = fs_i.solve_static()

        assert np.allclose(U_batch, U_ref, atol=1e-10, rtol=1e-10)

    def test_explicit_zero_value_matches_omitted_value(self):
        mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=6, ny=3)
        D = D_plane_stress(_steel())

        fs1 = FESystem(mesh, Quad4PlaneStress())
        fs1.assemble_stiffness(D)
        left = mesh.nodes_on_line(axis=0, value=0.0)
        for n in left:
            fs1.fix_dofs([n], [0, 1])

        fs2 = FESystem(mesh, Quad4PlaneStress())
        fs2.assemble_stiffness(D)
        for n in left:
            fs2.fix_dofs([n], [0, 1], value=0.0)

        F_batch = _random_load_batch(fs1.n_dof, 3, seed=3)
        U1 = solve_static_batched(fs1, F_batch)
        U2 = solve_static_batched(fs2, F_batch)
        assert np.array_equal(U1, U2)
