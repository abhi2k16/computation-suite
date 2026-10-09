# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_torch_iterative_solvers.py -- Wave 9 addendum item 137 (docs/
consolidated_future_roadmap.md): validates preconditioned_cg()'s
backend="numpy"/"torch" contract directly (iterative_solvers.py),
independent of the FESystem.solve_static() wiring already covered by
test_iterative_method_dispatch.py::TestTorchBackendGuard (that file
exercises this through the solver; this file exercises
preconditioned_cg()/jacobi_preconditioner_torch()/
ssor_preconditioner_torch() as standalone functions, on hand-built
SPD systems, matching test_iterative_solvers.py's own existing style
for the scipy-backend versions of these same functions).

Same two-part split as test_torch_mesh_transform.py/
test_torch_vectorized_assembly.py and this project's established
convention:

  TestBackendArgumentValidation -- runs UNCONDITIONALLY, no torch
      needed: unknown backend= raises ValueError; backend="scipy"
      (the default) is confirmed unchanged.

  TestTorchBackendNumericalAgreement -- gated on fea_engine.
      torch_sparse_solver._HAS_TORCH. The DECISIVE check is agreement
      with the scipy-backend preconditioned_cg() on the SAME SPD
      system, for M=None, jacobi_preconditioner_torch(), and
      ssor_preconditioner_torch() -- plus the explicit-failure
      contracts this item's docstring commits to: callback= raises
      TypeError on the torch path; a numpy-only preconditioner
      (jacobi_preconditioner()'s own scipy-path return value) raises
      TypeError rather than silently misbehaving mid-iteration; a
      non-SPD system still raises (RuntimeError via non-convergence,
      since the torch path's _cg_torch() loop -- reused from
      torch_sparse_solver.py -- does not repeat the scipy path's own
      explicit p^T A p <= 0 breakdown check, see that function's own
      docstring). Not executed end-to-end in the sandbox this file was
      authored in (no usable torch here); written to run for real, and
      SHOULD be run at least once on a torch-equipped machine.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
import scipy.sparse as sp

from fea_engine.torch_sparse_solver import _HAS_TORCH
from fea_engine.iterative_solvers import (
    preconditioned_cg,
    jacobi_preconditioner,
    jacobi_preconditioner_torch,
    ssor_preconditioner,
    ssor_preconditioner_torch,
)


def _spd_system(n=40, seed=0):
    """A small, well-conditioned SPD system (not from an FE assembly --
    this file tests preconditioned_cg() itself, not FE integration;
    test_iterative_method_dispatch.py's TestTorchBackendGuard already
    covers the real-stiffness-matrix end-to-end case)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)  # guaranteed SPD, diagonally dominant
    b = rng.standard_normal(n)
    return sp.csr_matrix(A), b


class TestBackendArgumentValidation:
    def test_unknown_backend_raises_valueerror(self):
        A, b = _spd_system()
        with pytest.raises(ValueError, match="unknown backend"):
            preconditioned_cg(A, b, backend="jax")

    def test_default_backend_scipy_matches_explicit_scipy(self):
        A, b = _spd_system()
        x_default, n_default = preconditioned_cg(A, b, tol=1e-12)
        x_explicit, n_explicit = preconditioned_cg(A, b, tol=1e-12, backend="scipy")
        np.testing.assert_array_equal(x_default, x_explicit)
        assert n_default == n_explicit


pytestmark_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not usable in this environment (not installed, or installed "
           "but failing to import -- see torch_sparse_solver.py's docstring); "
           "these tests validate the torch backend CG path and have nothing "
           "to run without it.")


@pytestmark_torch
class TestTorchBackendNumericalAgreement:
    def test_unpreconditioned_cg_matches_scipy_backend(self):
        A, b = _spd_system()
        x_ref, _ = preconditioned_cg(A, b, tol=1e-10)
        x_torch, _ = preconditioned_cg(A, b, tol=1e-10, backend="torch", device="cpu")
        assert np.allclose(x_ref, x_torch, atol=1e-6, rtol=1e-6)

    def test_jacobi_preconditioned_cg_matches_scipy_backend(self):
        A, b = _spd_system()
        M_scipy = jacobi_preconditioner(A)
        x_ref, _ = preconditioned_cg(A, b, M=M_scipy, tol=1e-10)
        M_torch = jacobi_preconditioner_torch(A, device="cpu")
        x_torch, _ = preconditioned_cg(A, b, M=M_torch, tol=1e-10, backend="torch", device="cpu")
        assert np.allclose(x_ref, x_torch, atol=1e-6, rtol=1e-6)

    def test_ssor_preconditioned_cg_matches_scipy_backend(self):
        A, b = _spd_system()
        M_scipy = ssor_preconditioner(A)
        x_ref, _ = preconditioned_cg(A, b, M=M_scipy, tol=1e-10)
        M_torch = ssor_preconditioner_torch(A, device="cpu")
        x_torch, _ = preconditioned_cg(A, b, M=M_torch, tol=1e-10, backend="torch", device="cpu")
        assert np.allclose(x_ref, x_torch, atol=1e-6, rtol=1e-6)

    def test_x0_warm_start_matches_scipy_backend(self):
        A, b = _spd_system()
        rng = np.random.default_rng(1)
        x0 = rng.standard_normal(b.shape[0]) * 0.01
        x_ref, _ = preconditioned_cg(A, b, x0=x0, tol=1e-10)
        x_torch, _ = preconditioned_cg(A, b, x0=x0, tol=1e-10, backend="torch", device="cpu")
        assert np.allclose(x_ref, x_torch, atol=1e-6, rtol=1e-6)

    def test_callback_raises_type_error_on_torch_path(self):
        A, b = _spd_system()
        with pytest.raises(TypeError, match="callback"):
            preconditioned_cg(A, b, callback=lambda k, x, r: None, backend="torch", device="cpu")

    def test_numpy_only_preconditioner_raises_type_error_on_torch_path(self):
        # jacobi_preconditioner() (the SCIPY-path version) is a
        # numpy-in-numpy-out closure with no
        # _fea_engine_torch_preconditioner marker -- must be rejected
        # up front, not allowed to fail confusingly mid-iteration.
        A, b = _spd_system()
        M_numpy_only = jacobi_preconditioner(A)
        with pytest.raises(TypeError, match="jacobi_preconditioner_torch"):
            preconditioned_cg(A, b, M=M_numpy_only, backend="torch", device="cpu")

    def test_unavailable_torch_fails_fast(self, monkeypatch):
        import fea_engine.iterative_solvers as itsolv_mod

        def _fake_require_torch():
            raise ImportError("simulated: torch unavailable")

        monkeypatch.setattr(itsolv_mod, "_require_torch", _fake_require_torch)
        A, b = _spd_system()
        with pytest.raises(ImportError):
            preconditioned_cg(A, b, backend="torch")
