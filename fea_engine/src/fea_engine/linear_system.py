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
        return np.asarray(v)[self.free]

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
        u_free = np.asarray(u_free)
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
    def solve(self, label="displacement"):
        """Solve with the system's own static solver path (Cholesky, or sparse LU) and recover."""
        s = self._system
        if s is None:
            raise RuntimeError("this ReducedSystem has no parent FESystem; solve it yourself")
        if s.sparse:
            uf = s._sparse_lu_solve(self.K.tocsc(), self.F)
        else:
            try:
                uf = s._dense_spd_solve(self.K, self.F)
            except np.linalg.LinAlgError:
                uf = s._eigen_solve(self.K, self.F)
        return self.recover(uf, label=label)

    def residual(self, u_free):
        """``K u_free - F`` for a candidate reduced solution."""
        return self.K @ np.asarray(u_free) - self.F
