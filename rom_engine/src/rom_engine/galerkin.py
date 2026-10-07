"""
galerkin.py -- intrusive Galerkin projection: given a reduced basis V
(from pod.py, or an ordinary eigenbasis, or any other orthonormal-ish
set of column vectors), project a full-order linear system onto it and
solve the tiny reduced system instead.

"Intrusive" means this needs the ACTUAL system matrices (K, M, F) of
the full-order model, unlike the non-intrusive surrogate approach
(fit a response surface to input/output SAMPLES without ever touching
K/M/F directly) that a future module of this package will add for
nonlinear problems. For a LINEAR problem, intrusive projection is
exact in the limit V -> full rank, and needs no training samples at
all -- one matrix projection replaces the whole "collect training
data" step.

The mathematics: seeking x ~= V @ q for some small q (n_modes,), the
full residual K x - F is, in general, NOT zero for every x in the
column space of V (V doesn't span the exact solution unless it happens
to). Galerkin's condition picks q so the residual is ORTHOGONAL to the
subspace V spans: V^T (K V q - F) = 0, i.e.

    (V^T K V) q = V^T F   <=>   K_r q = F_r

This is the unique best approximation to the true solution IN THE
V-COLUMN-SPACE, in the energy norm induced by K (for K symmetric
positive definite) -- the same variational argument that makes the
ordinary finite element method itself a Galerkin method one level up
(there, V is the FE shape functions; here, V is the reduced basis).

This module is deliberately FE-package-agnostic: K, M, F are plain
numpy arrays (or anything duck-typing an ndarray for @ and .T), so it
works with matrices from fea_engine, from any other FE code, or from a
hand-built K/M/F for a toy problem, as long as they're already
assembled on the SAME (full) degree-of-freedom numbering the basis V
was extracted on.
"""
import numpy as np
from scipy.linalg import eigh


def _as_array(x):
    """np.asarray(x), preserving complex dtype -- see pod._as_array for
    the full rationale (identical helper, duplicated rather than
    imported to keep this module's only dependency scipy/numpy, per
    the project's "no cross-module rom_engine imports in the core
    library" convention)."""
    x = np.asarray(x)
    return x if np.iscomplexobj(x) else x.astype(float, copy=False)


class GalerkinROM:
    """A reduced-order model built by Galerkin-projecting a full-order
    linear system onto a fixed basis V.

    Parameters
    ----------
    basis : ndarray (n_dof, n_modes), or an object with a .V attribute
        (e.g. a fitted pod.PodBasis) -- either form works, so this
        class can be built directly from an ordinary eigenvector
        matrix without going through PodBasis at all.
    """

    def __init__(self, basis):
        self.V = basis.V if hasattr(basis, "V") else _as_array(basis)
        if self.V.ndim != 2:
            raise ValueError(f"basis must be 2-D (n_dof, n_modes), got shape {self.V.shape}")
        self.n_dof, self.n_modes = self.V.shape
        self.K_r = None
        self.M_r = None
        self.F_r = None

    # -----------------------------------------------------------------
    # Projection primitives -- exposed directly too, since a caller may
    # want to project a matrix/vector without going through
    # reduce_system() (e.g. affine.py projects many component matrices
    # individually during its own offline stage).
    # -----------------------------------------------------------------
    def project_matrix(self, A):
        """V^T A V -- the reduced counterpart of a full (n_dof, n_dof)
        matrix (stiffness, mass, damping, ...)."""
        A = _as_array(A)
        return self.V.T @ A @ self.V

    def project_vector(self, b):
        """V^T b -- the reduced counterpart of a full (n_dof,) or
        (n_dof, k) load/right-hand-side vector (or batch of vectors)."""
        b = _as_array(b)
        return self.V.T @ b

    def expand(self, q):
        """V @ q -- reduced coordinates back to a full-order
        approximation. q may be (n_modes,) or (n_modes, k)."""
        return self.V @ _as_array(q)

    # -----------------------------------------------------------------
    # Building the reduced system
    # -----------------------------------------------------------------
    def reduce_system(self, K, M=None, F=None):
        """Project the given full-order matrices/vector once and cache
        the results as K_r/M_r/F_r, so repeated solve_static()/
        solve_modal() calls (e.g. under a load sweep, with F changing
        but K/M fixed) don't re-project K/M every time.

        M and F are optional independently: pass M for solve_modal(),
        F (or pass it directly to solve_static()) for solve_static().

        Returns self, so this can be chained with the constructor:
        ``rom = GalerkinROM(basis).reduce_system(K, M, F)``.
        """
        self.K_r = self.project_matrix(K)
        if M is not None:
            self.M_r = self.project_matrix(M)
        if F is not None:
            self.F_r = self.project_vector(F)
        return self

    # -----------------------------------------------------------------
    # Solves
    # -----------------------------------------------------------------
    def solve_static(self, F=None):
        """Solve K_r q = F_r (F_r from the argument if given, else the
        cached one from reduce_system()) and expand back to full
        coordinates.

        Returns (x_full, q_reduced): x_full = V @ q, the (n_dof,)
        (or (n_dof, k) for a batch of right-hand sides) full-space
        approximation, and q_reduced the raw (n_modes,) (or (n_modes,
        k)) reduced solution, returned too since it's often useful on
        its own (e.g. as a compact feature vector for a downstream
        surrogate model)."""
        if self.K_r is None:
            raise RuntimeError("call reduce_system(K, ...) before solve_static()")
        F_r = self.project_vector(F) if F is not None else self.F_r
        if F_r is None:
            raise ValueError("no F given here or cached via reduce_system(..., F=...)")
        q = np.linalg.solve(self.K_r, F_r)
        return self.expand(q), q

    def solve_modal(self, n_modes=None):
        """Reduced generalized eigenproblem K_r phi = omega^2 M_r phi
        -- the SAME calculation solve_modal() on a full FESystem would
        do, just on the tiny (n_modes, n_modes) reduced system instead
        of the full (n_dof, n_dof) one. If the basis V spans (close to)
        the true low-frequency eigenspace, these frequencies should
        closely match the full-order model's own low modes -- exactly
        the convergence check this module's test suite runs against
        fea_engine's own solve_modal().

        Returns (freq_hz, mode_shapes_full, mode_shapes_reduced):
        mode_shapes_full = V @ mode_shapes_reduced, expanded back to
        the (n_dof, n_modes_kept) full space for direct comparison
        against a full-order model's own mode shapes.
        """
        if self.K_r is None or self.M_r is None:
            raise RuntimeError("call reduce_system(K, M) before solve_modal()")
        eigvals, eigvecs_r = eigh(self.K_r, self.M_r)
        eigvals = np.clip(eigvals, 0, None)   # guard tiny negative numerical noise
        omega = np.sqrt(eigvals)
        freq_hz = omega / (2 * np.pi)
        if n_modes is not None:
            freq_hz = freq_hz[:n_modes]
            eigvecs_r = eigvecs_r[:, :n_modes]
        mode_shapes_full = self.expand(eigvecs_r)
        return freq_hz, mode_shapes_full, eigvecs_r
