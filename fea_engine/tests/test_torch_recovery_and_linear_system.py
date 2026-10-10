# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_torch_recovery_and_linear_system.py -- batched (vectorized) stress recovery against the per-element
reference loop, and the optional torch paths (torch recovery, differentiable stress, ReducedSystem.solve).

The batched NumPy tests always run; the torch tests skip when torch is missing or fails to import.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import itertools
import time

import numpy as np
import pytest

from fea_engine import (FESystem, Material, D_plane_stress, D_solid3d, Quad4PlaneStress, Quad8PlaneStress,
                        Tri3PlaneStress, Tri6PlaneStress, Hex8Solid3D, Hex20Solid3D, Tet4Solid3D, Tet10Solid3D,
                        coefficients as cf)
from fea_engine.mesh import Mesh, rectangle_mesh, box_mesh

try:
    import torch
    _HAS_TORCH = True
except Exception:                                   # missing, or a CUDA wheel without CUDA runtime
    torch = None
    _HAS_TORCH = False
needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")

MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)
ALL_ELEMENTS = [Quad4PlaneStress, Quad8PlaneStress, Tri3PlaneStress, Tri6PlaneStress,
                Hex8Solid3D, Hex20Solid3D, Tet4Solid3D, Tet10Solid3D]


def _natural_nodes(f):
    grid = [-1.0, 0.0, 1.0] if f.quadrature_family == "tensor" else [0.0, 0.5, 1.0]
    out = []
    for a in range(f.n_nodes):
        for p in itertools.product(grid, repeat=f.dim):
            N = f.shape_and_derivs(p)[0]
            if abs(N[a] - 1.0) < 1e-12 and abs(np.abs(N).sum() - 1.0) < 1e-12:
                out.append(p)
                break
    return np.array(out)


def _single_element_system(cls):
    f = cls()
    p = _natural_nodes(f)
    A = np.array([[2.0, 0.3, 0.1], [0.1, 1.5, 0.2], [0.05, 0.2, 1.2]])[:f.dim, :f.dim]
    nodes = p @ A.T + np.array([0.3, -0.2, 0.5])[:f.dim]
    mesh = Mesh(nodes=nodes, elements=np.arange(f.n_nodes)[None, :], dim=f.dim)
    s = FESystem(mesh, f, thickness=1.0)
    s.assemble_stiffness(D_plane_stress(MAT) if f.dim == 2 else D_solid3d(MAT), thickness=1.0)
    U = np.random.default_rng(3).normal(size=s.n_dof) * 1e-4
    return s, U


def _quad_system(nx=4, ny=3, tri=False):
    mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=nx, ny=ny)
    f = Quad4PlaneStress()
    if tri:
        q = np.asarray(mesh.elements)
        mesh = Mesh(nodes=mesh.nodes, elements=np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]), dim=2)
        f = Tri3PlaneStress()
    s = FESystem(mesh, f, thickness=0.1)
    s.assemble_stiffness(D_plane_stress(MAT), thickness=0.1)
    U = np.random.default_rng(5).normal(size=s.n_dof) * 1e-4
    return s, U


def _hex_system():
    mesh = box_mesh(1.0, 1.0, 1.0, 3, 2, 2)
    s = FESystem(mesh, Hex8Solid3D())
    s.assemble_stiffness(D_solid3d(MAT))
    U = np.random.default_rng(7).normal(size=s.n_dof) * 1e-4
    return s, U


def _close(a, b, tol=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    return np.allclose(a, b, rtol=tol, atol=tol * max(1.0, np.abs(b).max()))


# ====================================================================== batched NumPy == reference loop
class TestBatchedMatchesLoop:
    @pytest.mark.parametrize("cls", ALL_ELEMENTS)
    def test_single_element(self, cls):
        s, U = _single_element_system(cls)
        for fn in ("stress", "strain"):
            for at in ("nodes", "elements"):
                a = getattr(s, fn)(U, at=at)
                b = getattr(s, fn)(U, at=at, vectorized=False)
                assert _close(a, b), (cls.__name__, fn, at)
        assert _close(s.von_mises(U), s.von_mises(U, vectorized=False))

    @pytest.mark.parametrize("make", [_quad_system, lambda: _quad_system(tri=True), _hex_system])
    def test_shared_node_meshes(self, make):
        s, U = make()
        assert _close(s.stress(U), s.stress(U, vectorized=False))
        assert _close(s.stress(U, at="elements"), s.stress(U, at="elements", vectorized=False))
        assert _close(s.von_mises(U, at="elements"), s.von_mises(U, at="elements", vectorized=False))

    def test_coefficient_falls_back_to_loop(self):
        s, U = _quad_system()
        E = cf.from_material(lambda x: Material(E=1e11 * (1 + x[0]), nu=0.3, rho=1.0), D_plane_stress, at="gauss")
        s.assemble_stiffness(E, thickness=0.1)
        assert _close(s.stress(U), s.stress(U, vectorized=False))

    def test_bad_backend_and_args(self):
        s, U = _quad_system()
        with pytest.raises(ValueError, match="backend"):
            s.stress(U, backend="jax")

    def test_batched_is_faster(self):
        s, U = _quad_system(nx=40, ny=40)
        t0 = time.perf_counter(); s.stress(U); fast = time.perf_counter() - t0
        t0 = time.perf_counter(); s.stress(U, vectorized=False); slow = time.perf_counter() - t0
        assert fast < slow


# ====================================================================== torch recovery
@needs_torch
class TestTorchRecovery:
    @pytest.mark.parametrize("cls", ALL_ELEMENTS)
    def test_matches_numpy(self, cls):
        s, U = _single_element_system(cls)
        assert _close(s.stress(U, backend="torch"), s.stress(U))
        assert _close(s.strain(U, backend="torch"), s.strain(U))
        assert _close(s.von_mises(U, backend="torch", at="elements"), s.von_mises(U, at="elements"))

    def test_shared_nodes_and_tensor_input(self):
        s, U = _hex_system()
        assert _close(s.stress(torch.as_tensor(U), backend="torch"), s.stress(U))

    def test_requires_batched_and_constant_D(self):
        s, U = _quad_system()
        with pytest.raises(ValueError, match="vectorized"):
            s.stress(U, backend="torch", vectorized=False)
        E = cf.from_material(lambda x: MAT, D_plane_stress, at="gauss")
        s.assemble_stiffness(E, thickness=0.1)
        with pytest.raises(NotImplementedError, match="coefficients"):
            s.stress(U, backend="torch")

    def test_stress_tensor_gradient_matches_finite_difference(self):
        s, U = _quad_system(nx=2, ny=2)
        Ut = torch.as_tensor(U, dtype=torch.float64).clone().requires_grad_(True)
        loss = (s.stress_tensor(Ut) ** 2).sum()
        loss.backward()
        g = Ut.grad.numpy()

        def f(u):
            return float((s.stress_tensor(torch.as_tensor(u)) ** 2).sum())
        rng = np.random.default_rng(1)
        for i in rng.choice(len(U), 4, replace=False):
            h = 1e-9
            up, um = U.copy(), U.copy()
            up[i] += h; um[i] -= h
            fd = (f(up) - f(um)) / (2 * h)
            assert g[i] == pytest.approx(fd, rel=1e-5, abs=1e-6 * np.abs(g).max())

    def test_von_mises_tensor_backward_and_values(self):
        s, U = _quad_system(nx=2, ny=2)
        Ut = torch.as_tensor(U, dtype=torch.float64).clone().requires_grad_(True)
        vm = s.von_mises_tensor(Ut)
        assert np.allclose(vm.detach().numpy(), np.asarray(s.von_mises(U)).ravel(), rtol=1e-10)
        vm.sum().backward()
        assert Ut.grad is not None and np.isfinite(Ut.grad.numpy()).all()


# ====================================================================== ReducedSystem torch solve
def _plate(constrained=False):
    mesh = rectangle_mesh(Lx=0.8, Ly=0.4, nx=8, ny=4)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
    s.assemble_stiffness(D_plane_stress(MAT), thickness=0.02)
    mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=0.8, name="tip")
    s.fix_dofs("root", ["ux", "uy"])
    s.add_nodal_force("tip", "uy", -1.0e4)
    if constrained:
        a, b = int(mesh.select_nodes(x=0.8, y=0.0)[0]), int(mesh.select_nodes(x=0.8, y=0.4)[0])
        s.add_constraint([(a, "ux", 1.0), (b, "ux", -1.0)])
    return s


class TestReducedSystemSolve:
    def test_unknown_backend_and_method(self):
        rs = _plate().form_linear_system()
        with pytest.raises(ValueError, match="backend"):
            rs.solve(backend="jax")

    def test_numpy_input_helpers_accept_array_likes(self):
        rs = _plate().form_linear_system()
        u = np.random.default_rng(0).normal(size=rs.n_free)
        assert rs.expand(list(u)).shape == (rs.n_dof,)
        assert np.allclose(rs.residual(list(u)), rs.K @ u - rs.F)

    @needs_torch
    @pytest.mark.parametrize("constrained", [False, True])
    @pytest.mark.parametrize("method", ["dense", "cg", "auto"])
    def test_torch_matches_scipy(self, constrained, method):
        s = _plate(constrained)
        rs = s.form_linear_system()
        ref = np.asarray(rs.solve())
        got = np.asarray(rs.solve(backend="torch", method=method, tol=1e-12))
        assert np.allclose(got, ref, rtol=1e-6, atol=1e-9 * np.abs(ref).max())

    @needs_torch
    def test_to_torch_and_tensor_roundtrip(self):
        rs = _plate(True).form_linear_system()
        K, F = rs.to_torch()
        assert K.dtype == torch.float64 and tuple(F.shape) == (rs.n_free,)
        uf = torch.linalg.solve(K.to_dense() if K.layout != torch.strided else K, F)
        full = rs.expand(uf)                                  # tensors accepted
        assert np.allclose(full, np.asarray(rs.solve()), rtol=1e-8, atol=1e-12)
        Ks, _ = rs.to_torch(sparse=True)
        assert Ks.layout == torch.sparse_csr

    @needs_torch
    def test_cg_failure_raises(self):
        rs = _plate().form_linear_system()
        with pytest.raises(RuntimeError, match="did not converge"):
            rs.solve(backend="torch", method="cg", max_iter=1, tol=1e-14)
