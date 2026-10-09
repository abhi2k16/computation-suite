# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_batched_solve.py -- Wave 11 item 109 (docs/consolidated_future_
roadmap.md): validates solve_batched()/solve_static_batched()
(batched_solve.py) -- agreement with N independent np.linalg.solve()/
FESystem.solve_static() calls (the thing this item is a throughput
optimization OF, so every batched answer must match its own
independently-solved reference exactly), for both dense and sparse K,
and the real end-to-end multi-load-case FESystem use case this item was
built for (a ROM/surrogate training-snapshot sweep).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fea_engine import Quad4PlaneStress, FESystem, Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh
from fea_engine.batched_solve import solve_batched, solve_static_batched


def test_solve_batched_dense_matches_independent_solves():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    K = A @ A.T + 5 * np.eye(6)   # SPD, but solve_batched doesn't require SPD
    B = rng.standard_normal((6, 4))

    X = solve_batched(K, B)
    for i in range(4):
        x_ref = np.linalg.solve(K, B[:, i])
        assert np.allclose(X[:, i], x_ref, atol=1e-10)


def test_solve_batched_1d_input_returns_1d_output():
    K = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x = solve_batched(K, b)
    assert x.ndim == 1
    assert np.allclose(x, np.linalg.solve(K, b), atol=1e-12)


def test_solve_batched_sparse_matches_dense():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 8))
    K_dense = A @ A.T + 8 * np.eye(8)
    K_sparse = csr_matrix(K_dense)
    B = rng.standard_normal((8, 3))

    X_dense = solve_batched(K_dense, B)
    X_sparse = solve_batched(K_sparse, B)
    assert np.allclose(X_dense, X_sparse, atol=1e-9)


def test_solve_batched_unknown_backend_raises():
    K = np.eye(3)
    B = np.ones((3, 2))
    with pytest.raises(ValueError, match="unknown backend"):
        solve_batched(K, B, backend="bogus")


def _steel():
    return Material(E=2.1e11, nu=0.3, rho=7850.0)


def test_solve_static_batched_matches_n_independent_solve_static_calls():
    mesh = rectangle_mesh(Lx=3.0, Ly=1.0, nx=6, ny=3)
    D = D_plane_stress(_steel())
    left = mesh.nodes_on_line(axis=0, value=0.0)
    tip = mesh.nodes_on_line(axis=0, value=3.0)

    fs = FESystem(mesh, Quad4PlaneStress())
    fs.assemble_stiffness(D)
    for n in left:
        fs.fix_dofs([n], [0, 1])

    n_batch = 5
    rng = np.random.default_rng(2)
    F_batch = np.zeros((fs.n_dof, n_batch))
    tip_dof_y = [fs._global_dofs([n])[1] for n in tip]
    scales = rng.uniform(-2000.0, -200.0, size=n_batch)
    for j in range(n_batch):
        for d in tip_dof_y:
            F_batch[d, j] = scales[j] / len(tip)

    U_batch = solve_static_batched(fs, F_batch)

    for j in range(n_batch):
        fs.F = F_batch[:, j].copy()
        U_ref = fs.solve_static()
        assert np.allclose(U_batch[:, j], U_ref, atol=1e-8, rtol=1e-7)


def test_solve_static_batched_sparse_system():
    mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=4, ny=2)
    D = D_plane_stress(_steel())
    left = mesh.nodes_on_line(axis=0, value=0.0)
    tip = mesh.nodes_on_line(axis=0, value=2.0)

    fs = FESystem(mesh, Quad4PlaneStress(), sparse=True)
    fs.assemble_stiffness(D)
    for n in left:
        fs.fix_dofs([n], [0, 1])

    F_batch = np.zeros((fs.n_dof, 3))
    tip_dof_y = [fs._global_dofs([n])[1] for n in tip]
    for j, scale in enumerate([-100.0, -500.0, -900.0]):
        for d in tip_dof_y:
            F_batch[d, j] = scale / len(tip)

    U_batch = solve_static_batched(fs, F_batch)
    for j in range(3):
        fs.F = F_batch[:, j].copy()
        U_ref = fs.solve_static()
        assert np.allclose(U_batch[:, j], U_ref, atol=1e-8, rtol=1e-7)
