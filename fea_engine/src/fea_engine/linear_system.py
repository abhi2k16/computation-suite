# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
linear_system.py -- the reduced (constrained) linear system, exposed as an object.

``FESystem.form_linear_system()`` returns a :class:`ReducedSystem` holding the free-DOF stiffness
``K`` and right-hand side ``F`` (Dirichlet values already moved to the right-hand side), plus
``recover`` to expand a reduced solution back to a full ``FEField``. Use it to plug in your own
solver, preconditioner, eigen-solver or reduced-order model without re-implementing static
condensation. It is the counterpart of MFEM's ``FormLinearSystem`` / ``RecoverFEMSolution``.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from dataclasses import dataclass, field

import numpy as np


def _np(x):
    """ndarray view of ``x``; torch tensors are detached and moved to the CPU first."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class ReducedSystem:
    """Constrained linear system ``K u_free = F`` and the maps to and from the full DOF vector.

    Attributes
    ----------
    K : ndarray or scipy.sparse.csr_matrix, shape (n_free, n_free)
    F : ndarray, shape (n_free,)
        Load on free DOFs minus ``K[free, fixed] @ u_fixed``.
    free, fixed : int ndarray
        Global indices of free and constrained DOFs (both sorted).
    u_fixed : ndarray
        Prescribed values at ``fixed`` (zeros for homogeneous constraints).
    n_dof : int
        Size of the full system.
    """
    K: object
    F: np.ndarray
    free: np.ndarray
    fixed: np.ndarray
    u_fixed: np.ndarray
    n_dof: int
    _system: object = field(default=None, repr=False, compare=False)
    # linear constraints (None when there are none): dependent DOFs, u_slave = g - C u_free
    slave: np.ndarray = field(default=None, repr=False, compare=False)
    slave_C: object = field(default=None, repr=False, compare=False)
    slave_g: np.ndarray = field(default=None, repr=False, compare=False)
    _T: object = field(default=None, repr=False, compare=False)          # free_all -> free (independent)
    _free_all: np.ndarray = field(default=None, repr=False, compare=False)

    @property
    def n_free(self):
        return len(self.free)

    # ------------------------------------------------------------------ maps
    def restrict(self, v):
        """Full vector (or matrix of column vectors) -> free-DOF part."""
        return _np(v)[self.free]

    def reduce_matrix(self, A):
        """Free-free block of another full matrix (e.g. ``M``, ``C``, a geometric stiffness)."""
        if self._system is not None:
            A = self._system._as_solve_matrix(A)
        if self._T is not None:                       # constrained: T^T A_ff T
            Aff = A[np.ix_(self._free_all, self._free_all)]
            return self._T.T @ (Aff @ self._T)
        return A[np.ix_(self.free, self.free)]

    def expand(self, u_free, include_prescribed=True):
        """Free-DOF vector (or ``(n_free, k)`` columns) -> full-size array, zeros at constrained DOFs.

        With ``include_prescribed=True`` the prescribed values are placed at the constrained DOFs
        (1-D input only). Pass ``False`` for modes and increments, which vanish there.
        """
        u_free = _np(u_free)
        if u_free.shape[0] != self.n_free:
            raise ValueError(f"expected {self.n_free} free-DOF rows, got {u_free.shape[0]}")
        full = np.zeros((self.n_dof,) + u_free.shape[1:], dtype=np.result_type(u_free.dtype, float))
        full[self.free] = u_free
        if self.slave is not None and len(self.slave):
            dep = -(self.slave_C @ u_free)
            if include_prescribed and u_free.ndim == 1:
                dep = dep + self.slave_g
            full[self.slave] = dep
        if include_prescribed and u_free.ndim == 1:
            full[self.fixed] = self.u_fixed
        return full

    def recover(self, u_free, label="displacement", include_prescribed=True):
        """Like :meth:`expand`, but returns an ``FEField`` when the parent system is known."""
        full = self.expand(u_free, include_prescribed=include_prescribed)
        if self._system is not None:
            return self._system.field(full, label=label)
        return full

    # ------------------------------------------------------------------ convenience
    def solve(self, label="displacement", backend="scipy", device="cpu", method="auto", tol=1e-8, max_iter=None):
        """Solve and recover.

        ``backend="scipy"`` (default) uses the system's own static path (Cholesky, or sparse LU).
        ``backend="torch"`` uses PyTorch (optional dependency) on ``device`` (``"cpu"`` or ``"cuda"``), in
        float64: ``method="dense"`` is a direct solve, ``method="cg"`` Jacobi-preconditioned conjugate
        gradients (SPD only, raises if it does not converge), ``"auto"`` picks ``"dense"`` below 2000 free
        DOFs and ``"cg"`` above. Constraints are already eliminated in ``K``, so both work with them.
        """
        s = self._system
        if s is None:
            raise RuntimeError("this ReducedSystem has no parent FESystem; solve it yourself")
        if backend == "torch":
            uf = self._solve_torch(device, method, tol, max_iter)
            return self.recover(uf, label=label)
        if backend != "scipy":
            raise ValueError(f"backend must be 'scipy' or 'torch', got {backend!r}.")
        if s.sparse:
            uf = s._sparse_lu_solve(self.K.tocsc(), self.F)
        else:
            try:
                uf = s._dense_spd_solve(self.K, self.F)
            except np.linalg.LinAlgError:
                uf = s._eigen_solve(self.K, self.F)
        return self.recover(uf, label=label)

    def _solve_torch(self, device, method, tol, max_iter):
        from . import torch_sparse_solver as tss
        tss._require_torch()
        if method == "auto":
            method = "dense" if self.n_free < 2000 else "cg"
        if method == "dense":
            return tss.torch_dense_solve(self.K, self.F, device=device)
        if method == "cg":
            uf, n_iter, ok = tss.torch_sparse_cg_solve(self.K, self.F, tol=tol, max_iter=max_iter, device=device)
            if not ok:
                raise RuntimeError(f"ReducedSystem.solve: CG did not converge in {n_iter} iterations "
                                   f"(tol={tol}); K may not be SPD. Try method='dense'.")
            return uf
        raise ValueError(f"method must be 'auto', 'dense' or 'cg', got {method!r}.")

    def to_torch(self, device="cpu", dtype=None, sparse=None):
        """``(K, F)`` as torch tensors on ``device`` (float64 by default) for your own torch solver or
        differentiable pipeline. ``K`` is a sparse CSR tensor when the system is sparse (or ``sparse=True``),
        otherwise dense."""
        from . import torch_sparse_solver as tss
        tss._require_torch()
        import torch
        dtype = dtype or torch.float64
        import scipy.sparse as sp
        use_sparse = sp.issparse(self.K) if sparse is None else bool(sparse)
        if use_sparse:
            K = tss._scipy_csr_to_torch_sparse(tss._to_csr_scipy(self.K), device, dtype)
        else:
            Kd = self.K.toarray() if sp.issparse(self.K) else np.asarray(self.K)
            K = torch.as_tensor(Kd, dtype=dtype, device=device)
        F = torch.as_tensor(np.asarray(self.F, dtype=float), dtype=dtype, device=device)
        return K, F

    def residual(self, u_free):
        """``K u_free - F`` for a candidate reduced solution."""
        return self.K @ _np(u_free) - self.F
