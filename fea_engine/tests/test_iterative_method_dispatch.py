# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_iterative_method_dispatch.py -- Wave 16 item 132 (docs/
consolidated_future_roadmap.md, source: TensorMesh's `Sparse Solvers`
documentation page): validates FESystem.solve_static(method="cg"/"pcg")
-- routing the free-DOF solve through Wave 8's own preconditioned_cg()
(iterative_solvers.py) instead of the existing direct/eigen path.

Decisive checks throughout: agreement with the existing direct-solve
path (not just "runs without erroring"), each of the three
preconditioners actually being exercised and agreeing too, correct
interaction with item 131's non-homogeneous Dirichlet correction, the
method=None default being a true no-op (nothing about this item changes
the untouched code path), and the explicit ValueError guards (unknown
method, method= combined with backend='auto') -- matching this
project's own "fail loudly, never silently reinterpret" precedent.

TestTorchBackendGuard (updated for Wave 9 addendum item 137, docs/
consolidated_future_roadmap.md): method="cg"/"pcg" combined with
backend="torch" used to raise ValueError (the ORIGINAL, narrower scope
of this item); item 137 closes that gap by giving backend="torch" its
own CG dispatch (iterative_solvers.preconditioned_cg(backend="torch"))
-- so this class now confirms AGREEMENT with the direct-solve path on
backend="torch", the same decisive standard every other class in this
file already holds backend="scipy" to, plus the one still-scoped-out
case (preconditioner="ic0" combined with backend="torch") raising
NotImplementedError rather than silently falling back or breaking.
backend="auto" combined with method= still raises ValueError
unchanged -- see TestGuardRails::test_method_with_backend_auto_raises_
value_error below, untouched by this item.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import Quad4PlaneStress, FESystem, Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh


def _steel():
    return Material(E=2.1e11, nu=0.3, rho=7850.0)


def _cantilever(sparse=False, backend="scipy"):
    mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=8, ny=4)
    D = D_plane_stress(_steel())
    fs = FESystem(mesh, Quad4PlaneStress(), sparse=sparse, backend=backend)
    fs.assemble_stiffness(D)
    left = mesh.nodes_on_line(axis=0, value=0.0)
    tip = mesh.nodes_on_line(axis=0, value=2.0)
    for n in left:
        fs.fix_dofs([n], [0, 1])
    for n in tip:
        fs.F[fs._global_dofs([n])[1]] += -1000.0 / len(tip)
    return fs


class TestCGMatchesDirectSolve:
    """The decisive correctness check: method="cg"/"pcg" must agree
    with the existing, already-validated direct-solve path -- not just
    converge to *something*."""

    def test_dense_cg_matches_direct(self):
        fs_ref = _cantilever(sparse=False)
        U_ref = fs_ref.solve_static()
        fs = _cantilever(sparse=False)
        U_cg = fs.solve_static(method="cg", tol=1e-10, maxiter=20000)
        assert np.allclose(U_cg, U_ref, atol=1e-8, rtol=1e-8)

    def test_sparse_cg_matches_direct(self):
        fs_ref = _cantilever(sparse=True)
        U_ref = fs_ref.solve_static()
        fs = _cantilever(sparse=True)
        U_cg = fs.solve_static(method="cg", tol=1e-10, maxiter=20000)
        assert np.allclose(U_cg, U_ref, atol=1e-8, rtol=1e-8)

    def test_pcg_default_preconditioner_matches_direct(self):
        fs_ref = _cantilever()
        U_ref = fs_ref.solve_static()
        fs = _cantilever()
        U_pcg = fs.solve_static(method="pcg", tol=1e-10)
        assert np.allclose(U_pcg, U_ref, atol=1e-8, rtol=1e-8)

    @pytest.mark.parametrize("precond", ["jacobi", "ssor", "ic0"])
    def test_pcg_every_preconditioner_matches_direct(self, precond):
        fs_ref = _cantilever()
        U_ref = fs_ref.solve_static()
        fs = _cantilever()
        U_pcg = fs.solve_static(method="pcg", preconditioner=precond, tol=1e-10)
        assert np.allclose(U_pcg, U_ref, atol=1e-8, rtol=1e-8)


class TestNonHomogeneousDirichletInteraction:
    """method="cg" must honor item 131's -Kio*uo RHS correction
    identically to the direct path -- the correction is a property of
    the linear system, not of how it's solved."""

    def test_cg_with_nonzero_prescribed_value_matches_direct(self):
        mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=8, ny=4)
        D = D_plane_stress(_steel())

        def build():
            fs = FESystem(mesh, Quad4PlaneStress())
            fs.assemble_stiffness(D)
            left = mesh.nodes_on_line(axis=0, value=0.0)
            right = mesh.nodes_on_line(axis=0, value=2.0)
            for n in left:
                fs.fix_dofs([n], [0, 1])
            for n in right:
                fs.fix_dofs([n], [0], value=0.02)
            return fs, right

        fs_direct, right = build()
        U_direct = fs_direct.solve_static()

        fs_cg, _ = build()
        U_cg = fs_cg.solve_static(method="cg", tol=1e-12)

        assert np.allclose(U_cg, U_direct, atol=1e-8, rtol=1e-8)
        right_x = [fs_direct._global_dofs([n])[0] for n in right]
        assert np.allclose(U_cg[right_x], 0.02, atol=1e-8)


class TestGuardRails:
    def test_method_none_default_is_unchanged_direct_solve(self):
        """method=None (the default) must be bit-for-bit identical to
        never having passed method= at all -- true no-op, matching
        this project's own IDENTICAL-cost/behavior-when-unused
        convention (first established for Wave 1's du_tol/energy_tol)."""
        fs1 = _cantilever()
        U1 = fs1.solve_static()
        fs2 = _cantilever()
        U2 = fs2.solve_static(method=None)
        assert np.array_equal(U1, U2)

    def test_method_with_backend_auto_raises_value_error(self):
        fs = _cantilever(backend="auto")
        with pytest.raises(ValueError, match="only supported with backend='scipy'"):
            fs.solve_static(method="cg")

    def test_unknown_method_raises_value_error(self):
        fs = _cantilever()
        with pytest.raises(ValueError, match="unknown method"):
            fs.solve_static(method="not_a_real_method")

    def test_stationary_iterations_are_not_exposed_as_a_method(self):
        """Deliberate scope decision (see _solve_static_iterative()'s
        own docstring): Jacobi/Gauss-Seidel/SOR are NOT valid method=
        choices here, even though iterative_solvers.py implements them
        -- confirmed to diverge on an ordinary (non-diagonally-
        dominant) Quad4 stiffness matrix during this item's own
        development, so exposing them as a top-level solve method
        would be handing a caller a solver likely to fail silently."""
        fs = _cantilever()
        with pytest.raises(ValueError, match="unknown method"):
            fs.solve_static(method="jacobi")

    def test_unknown_preconditioner_raises_value_error(self):
        fs = _cantilever()
        with pytest.raises(ValueError, match="unknown preconditioner"):
            fs.solve_static(method="pcg", preconditioner="not_a_real_preconditioner")


class TestTorchBackendGuard:
    """method="cg"/"pcg" combined with backend='torch' (Wave 9 addendum
    item 137) -- gated on torch actually being usable, same convention
    as test_nonhomogeneous_dirichlet.py's own TestTorchBackendGuard
    (backend='torch' fails at FESystem construction time in any
    environment without a working torch install, including this
    project's own sandbox -- see torch_sparse_solver.py's docstring).
    Not executed end-to-end in the sandbox this file was authored in
    for that exact reason; written to run for real, and SHOULD be run
    at least once in an environment with a working torch install."""

    def _torch_usable(self):
        try:
            from fea_engine.torch_sparse_solver import _HAS_TORCH
            return _HAS_TORCH
        except ImportError:
            return False

    def test_torch_cg_matches_direct_solve(self):
        if not self._torch_usable():
            pytest.skip("torch not usable in this environment")
        fs_direct = _cantilever(backend="torch")
        fs_cg = _cantilever(backend="torch")
        U_direct = fs_direct.solve_static()
        U_cg = fs_cg.solve_static(method="cg", tol=1e-10)
        assert np.allclose(U_direct, U_cg, atol=1e-8, rtol=1e-6)

    def test_torch_pcg_jacobi_matches_direct_solve(self):
        if not self._torch_usable():
            pytest.skip("torch not usable in this environment")
        fs_direct = _cantilever(backend="torch")
        fs_pcg = _cantilever(backend="torch")
        U_direct = fs_direct.solve_static()
        U_pcg = fs_pcg.solve_static(method="pcg", preconditioner="jacobi", tol=1e-10)
        assert np.allclose(U_direct, U_pcg, atol=1e-8, rtol=1e-6)

    def test_torch_pcg_ssor_matches_direct_solve(self):
        if not self._torch_usable():
            pytest.skip("torch not usable in this environment")
        fs_direct = _cantilever(backend="torch")
        fs_pcg = _cantilever(backend="torch")
        U_direct = fs_direct.solve_static()
        U_pcg = fs_pcg.solve_static(method="pcg", preconditioner="ssor", tol=1e-10)
        assert np.allclose(U_direct, U_pcg, atol=1e-8, rtol=1e-6)

    def test_torch_pcg_ic0_raises_not_implemented(self):
        if not self._torch_usable():
            pytest.skip("torch not usable in this environment")
        fs = _cantilever(backend="torch")
        with pytest.raises(NotImplementedError, match="ic0"):
            fs.solve_static(method="pcg", preconditioner="ic0")

    def test_torch_cg_sparse_matches_direct_solve(self):
        if not self._torch_usable():
            pytest.skip("torch not usable in this environment")
        fs_direct = _cantilever(sparse=True, backend="torch")
        fs_cg = _cantilever(sparse=True, backend="torch")
        U_direct = fs_direct.solve_static()
        U_cg = fs_cg.solve_static(method="cg", tol=1e-10)
        assert np.allclose(U_direct, U_cg, atol=1e-8, rtol=1e-6)
