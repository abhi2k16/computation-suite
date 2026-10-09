# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
hyper_reduction.py -- hyper-reduction for nonlinear ROMs: ECSW, DEIM/QDEIM, gappy reconstruction.

WHY. `intrusive_nonlinear_rom.IntrusiveNonlinearROM` projects the equations of motion onto a small
basis V, but every time step still calls the FULL-order internal force routine (all elements) and
then projects the result, so the cost per step stays proportional to the size of the finite
element model. Hyper-reduction removes that last full-order dependence: the nonlinear reduced force

    f_r(q) = V^T f_int(V q)

is approximated from a SMALL subset of the model, chosen offline.

TWO FAMILIES (both provided, neither replaces the other):

  * ECSW  (Energy-Conserving Sampling and Weighting, Farhat et al. 2014/2015). Keeps the reduced
    force as a weighted sum over a few ELEMENTS:   f_r(q) ~= sum_{e in E} w_e V_e^T f_e(V_e q).
    The weights come from a non-negative least squares fit to training states. Because it sums
    real element contributions with positive weights it preserves the structure of the model
    (symmetric tangent, positive-definite stiffness). Needs per-element force callbacks.

  * DEIM / QDEIM  (Chaturantabut & Sorensen 2010; Drmac & Gugercin 2016). Approximates the full
    nonlinear force vector f(u) from its values at a few interpolation DOFs using a POD basis of
    force snapshots:  f ~= U (P^T U)^-1 P^T f.  Works with any black-box force function that can be
    evaluated at selected rows, but does not by itself keep the reduced tangent symmetric.

`gappy_reconstruct()` is the general least-squares form (more sample points than basis vectors).

HONEST LIMITS.
  - Hyper-reduction is only as good as its training states: states far outside the training range
    can give large errors (check with `reduced_force_error()` on held-out states).
  - ECSW as implemented here is a standard Lawson-Hanson active-set NNLS with an early stop at a
    relative tolerance; very small tolerances select many elements and reduce the speed-up.
  - The speed-up exists only when `elem_force_fn` costs time proportional to the number of
    selected elements, which is how the callbacks in the tests are written.

This module never imports fea_engine. Callers pass plain arrays and callbacks (see the tests for a
fea_engine-based wiring).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import qr, lu_factor, lu_solve

from .intrusive_nonlinear_rom import IntrusiveNonlinearROM


def _as_array(x):
    return np.asarray(x)


# ----------------------------------------------------------------------------- DEIM family
def deim_indices(U):
    """Greedy DEIM point selection for a force basis U (n x m, orthonormal columns preferred).

    Returns an integer array of m distinct row indices. The first point is the largest entry of
    the first basis vector; each next point is the largest entry of the residual of the next basis
    vector after interpolating it with the points chosen so far."""
    U = _as_array(U)
    n, m = U.shape
    idx = [int(np.argmax(np.abs(U[:, 0])))]
    for j in range(1, m):
        c = np.linalg.solve(U[np.ix_(idx, range(j))], U[idx, j])
        r = U[:, j] - U[:, :j] @ c
        idx.append(int(np.argmax(np.abs(r))))
    if len(set(idx)) != m:
        raise ValueError("DEIM selected a repeated point; the basis is probably rank-deficient")
    return np.array(idx, dtype=int)


def qdeim_indices(U):
    """QDEIM point selection (Drmac & Gugercin 2016): column-pivoted QR of U^T.

    Same output as `deim_indices` but it needs no sequential solves and usually gives a better
    conditioned interpolation matrix."""
    U = _as_array(U)
    _, _, piv = qr(U.T, mode="economic", pivoting=True)
    return np.array(piv[: U.shape[1]], dtype=int)


class DEIM:
    """Discrete empirical interpolation of a force vector from its values at a few DOFs.

    Parameters
    ----------
    U : (n, m) force basis (for example left singular vectors of force snapshots).
    method : "deim" (greedy) or "qdeim" (pivoted QR).

    Attributes
    ----------
    indices : (m,) selected rows.
    interp : (n, m) matrix U (P^T U)^-1, so that  f ~= interp @ f[indices].
    """

    def __init__(self, U, method="deim"):
        U = _as_array(U)
        if method == "deim":
            self.indices = deim_indices(U)
        elif method == "qdeim":
            self.indices = qdeim_indices(U)
        else:
            raise ValueError("method must be 'deim' or 'qdeim'")
        self.U = U
        self._lu = lu_factor(U[self.indices, :].T)
        self.interp = U @ np.linalg.inv(U[self.indices, :])
        self.method = method

    @classmethod
    def from_snapshots(cls, force_snapshots, n_points=None, energy=None, method="deim"):
        """Build the force basis by SVD of force snapshots (columns), then select points."""
        F = _as_array(force_snapshots)
        Us, s, _ = np.linalg.svd(F, full_matrices=False)
        if n_points is None:
            if energy is None:
                energy = 1.0 - 1e-10
            cum = np.cumsum(s**2) / np.sum(s**2)
            n_points = int(min(np.searchsorted(cum, energy) + 1, len(s)))
        return cls(Us[:, :n_points], method=method)

    def approximate(self, f_at_indices):
        """Approximate the full vector from its values at `self.indices`."""
        return self.interp @ _as_array(f_at_indices)

    def approximate_reduced(self, V, f_at_indices):
        """V^T f without ever forming the full f:  (V^T interp) @ f_P  (precompute for speed)."""
        return (_as_array(V).T @ self.interp) @ _as_array(f_at_indices)


def gappy_reconstruct(U, indices, f_at_indices):
    """Least-squares gappy-POD reconstruction  f ~= U argmin_c || f_P - U_P c ||  (len(indices) >= m)."""
    U = _as_array(U)
    c, *_ = np.linalg.lstsq(U[np.asarray(indices), :], _as_array(f_at_indices), rcond=None)
    return U @ c


# ----------------------------------------------------------------------------- ECSW
def ecsw_weights(G, b, tol=1e-6, max_elements=None):
    """Sparse non-negative weights w >= 0 with  || G w - b || <= tol * ||b||  (Lawson-Hanson NNLS
    with an early stop at the tolerance, which is what keeps the selected set small).

    Parameters
    ----------
    G : (n_rows, n_elem) training matrix; column e holds element e's reduced force contributions
        stacked over all training states.
    b : (n_rows,) the exact target (sum of all columns for ECSW, i.e. weights of one reproduce it).
    tol : relative residual at which to stop adding elements.
    max_elements : optional cap on the number of selected elements.

    Returns
    -------
    w : (n_elem,) weights (zero for unselected elements)
    info : dict with 'residual' (relative), 'n_selected', 'iterations'
    """
    G = _as_array(G).astype(float)
    b = _as_array(b).astype(float)
    n = G.shape[1]
    bn = np.linalg.norm(b)
    if bn == 0:
        return np.zeros(n), dict(residual=0.0, n_selected=0, iterations=0)
    w = np.zeros(n)
    passive = []
    r = b.copy()
    it = 0
    cap = n if max_elements is None else int(max_elements)
    while np.linalg.norm(r) > tol * bn and len(passive) < cap and it < 5 * n + 50:
        it += 1
        grad = G.T @ r
        grad[passive] = -np.inf
        j = int(np.argmax(grad))
        if not np.isfinite(grad[j]) or grad[j] <= 1e-14 * np.linalg.norm(G.T @ b, np.inf):
            break
        passive.append(j)
        while True:
            s = np.zeros(n)
            s[passive], *_ = np.linalg.lstsq(G[:, passive], b, rcond=None)
            if np.all(s[passive] > 0):
                w = s
                break
            neg = [k for k in passive if s[k] <= 0]
            alpha = min(w[k] / (w[k] - s[k]) for k in neg if w[k] - s[k] != 0) if neg else 0.0
            w = w + alpha * (s - w)
            passive = [k for k in passive if w[k] > 1e-14]
            w[[k for k in range(n) if k not in passive]] = 0.0
            if not passive:
                break
        r = b - G @ w
    return w, dict(residual=float(np.linalg.norm(r) / bn), n_selected=int(np.count_nonzero(w)), iterations=it)


def ecsw_training_matrix(V, snapshots, elem_dofs, elem_force_fn):
    """Build (G, b) for `ecsw_weights`.

    Parameters
    ----------
    V : (n_dof, r) basis.
    snapshots : (n_dof, n_snap) full-order states (columns) at which the force is sampled.
    elem_dofs : list of integer arrays; global DOFs of each element (length n_elem).
    elem_force_fn : callable(e, u_local) -> (len(elem_dofs[e]),) element internal force.
    """
    V = _as_array(V)
    S = _as_array(snapshots)
    n_elem = len(elem_dofs)
    r = V.shape[1]
    G = np.zeros((S.shape[1] * r, n_elem))
    for k in range(S.shape[1]):
        u = S[:, k]
        for e in range(n_elem):
            g = np.asarray(elem_dofs[e])
            fe = _as_array(elem_force_fn(e, u[g]))
            G[k * r:(k + 1) * r, e] = V[g, :].T @ fe
    return G, G.sum(axis=1)


class ECSW:
    """Hyper-reduced reduced internal force from a weighted element subset.

    Build with `ECSW.fit(...)`; evaluate with `reduced_force(q)` (and `reduced_tangent(q)` if an
    element tangent callback was given). Only the selected elements are evaluated.
    """

    def __init__(self, V, elem_dofs, elem_force_fn, weights, elem_tangent_fn=None, info=None):
        self.V = _as_array(V)
        self.weights = _as_array(weights)
        self.elements = np.flatnonzero(self.weights > 0)
        self._dofs = [np.asarray(elem_dofs[e]) for e in self.elements]
        self._w = self.weights[self.elements]
        self._Ve = [self.V[g, :] for g in self._dofs]
        self.elem_force_fn = elem_force_fn
        self.elem_tangent_fn = elem_tangent_fn
        self.n_total = len(elem_dofs)
        self.info = info or {}

    @classmethod
    def fit(cls, V, snapshots, elem_dofs, elem_force_fn, elem_tangent_fn=None, tol=1e-6, max_elements=None):
        G, b = ecsw_training_matrix(V, snapshots, elem_dofs, elem_force_fn)
        w, info = ecsw_weights(G, b, tol=tol, max_elements=max_elements)
        return cls(V, elem_dofs, elem_force_fn, w, elem_tangent_fn, info)

    @property
    def n_selected(self):
        return len(self.elements)

    def reduced_force(self, q):
        q = _as_array(q)
        f = np.zeros(self.V.shape[1])
        for e, wv, Ve in zip(self.elements, self._w, self._Ve):
            f += wv * (Ve.T @ _as_array(self.elem_force_fn(int(e), Ve @ q)))
        return f

    def reduced_tangent(self, q):
        if self.elem_tangent_fn is None:
            raise RuntimeError("ECSW was built without elem_tangent_fn")
        q = _as_array(q)
        K = np.zeros((self.V.shape[1],) * 2)
        for e, wv, Ve in zip(self.elements, self._w, self._Ve):
            K += wv * (Ve.T @ _as_array(self.elem_tangent_fn(int(e), Ve @ q)) @ Ve)
        return K


def reduced_force_error(ecsw, full_reduced_force_fn, q_test):
    """Relative error of the hyper-reduced reduced force at held-out reduced states `q_test` (rows).

    full_reduced_force_fn(q) must return V^T f_int(V q) with the FULL model."""
    errs = []
    for q in np.atleast_2d(q_test):
        ref = _as_array(full_reduced_force_fn(q))
        errs.append(np.linalg.norm(ecsw.reduced_force(q) - ref) / max(np.linalg.norm(ref), 1e-300))
    return np.array(errs)


class HyperReducedNonlinearROM(IntrusiveNonlinearROM):
    """`IntrusiveNonlinearROM` whose reduced internal force/tangent come from an ECSW element subset.

    All integrators of the parent class (`integrate_rk4`, `integrate_solve_ivp`,
    `integrate_newton_newmark`, ...) work unchanged because they call `reduced_internal_force()` /
    `reduced_tangent()`, which are overridden here.

    Parameters
    ----------
    V, M, C, load_fn : as for `IntrusiveNonlinearROM`.
    K0 : (n_dof, n_dof) full zero-state tangent. Used ONCE to build the linear reduced stiffness
        `K_r` (an offline cost).
    ecsw : a fitted `ECSW`.
    """

    def __init__(self, V, M, C, load_fn, K0, ecsw):
        K0 = _as_array(K0)
        super().__init__(V, M, C, internal_force_fn=lambda u: np.zeros(K0.shape[0]), load_fn=load_fn,
                         tangent_fn=lambda u: K0)
        self.ecsw = ecsw

    def reduced_internal_force(self, q):
        return self.ecsw.reduced_force(q)

    def reduced_tangent(self, q):
        return self.ecsw.reduced_tangent(q)
