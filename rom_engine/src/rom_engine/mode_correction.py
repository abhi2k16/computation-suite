# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
mode_correction.py -- mode acceleration and modal truncation augmentation
(Besselink, Tabak, Lutowska, van de Wouw, Nijmeijer, Rixen, Hochstenbach
& Schilders, JSV 2013, Section 2 / eqs. 13-20): two cheap, closed-form
corrections to plain mode displacement (x ~= V @ eta, exactly what
galerkin.GalerkinROM already gives from a modal basis) that recover the
STATIC contribution of whatever modes a truncated basis V leaves out.

Why this matters: for a static (or slowly-varying) load, a truncated
modal basis is missing not just the truncated modes' DYNAMIC response
(which is fine to drop if those modes are far above the frequency band
of interest) but also their STATIC contribution to the total deflection
-- and for a smooth load, that missing static piece is often the biggest
single source of truncation error, independent of how many modes are
kept. Both corrections below add that missing static piece back in,
using only ONE extra full-order static solve (K^-1 F), which is a
one-time cost per load pattern, not a per-query cost.

This module is deliberately additive to, not a replacement for,
galerkin.GalerkinROM: mode_acceleration_response() is a post-hoc
correction applied on top of an ordinary GalerkinROM.solve_static() call,
and augmented_basis() produces a plain ndarray that feeds straight back
into GalerkinROM(augmented_basis(...)).reduce_system(...) -- no new
solver logic is needed for either method, matching this package's
"compose with existing modules, don't duplicate their algebra"
convention (the same reasoning frequency.py already follows by composing
affine.py + galerkin.py instead of reimplementing either).

Like every other module in this package, this is FE-package-agnostic:
K, F, and the basis are plain numpy arrays (or a basis object with a .V
attribute, e.g. a fitted pod.PodBasis or a galerkin.GalerkinROM's own
.V), on the same full DOF numbering throughout.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np


def _as_array(x):
    """np.asarray(x), preserving complex dtype -- see pod._as_array for
    the full rationale (identical helper, duplicated rather than
    imported to keep this module's only dependency scipy/numpy, per the
    project's "no cross-module rom_engine imports in the core library"
    convention)."""
    x = np.asarray(x)
    return x if np.iscomplexobj(x) else x.astype(float, copy=False)


def _basis_matrix(basis):
    """Accept either a plain ndarray or anything with a .V attribute
    (a fitted pod.PodBasis, a galerkin.GalerkinROM, ...) -- the same
    duck-typing convention galerkin.GalerkinROM.__init__ already uses."""
    V = basis.V if hasattr(basis, "V") else _as_array(basis)
    if V.ndim != 2:
        raise ValueError(f"basis must be 2-D (n_dof, n_modes), got shape {V.shape}")
    return V


# -----------------------------------------------------------------------
# Mode acceleration (eqs. 13-16)
# -----------------------------------------------------------------------
def mode_acceleration_correction(K, basis, F):
    """The static correction term

        q_cor = K^-1 F - V (V^T K V)^-1 V^T F

    -- the part of the static response the modes OUTSIDE the kept basis
    V would have contributed, computed without ever forming those modes
    explicitly. Requires exactly one full-order static solve (K^-1 F);
    the second term reuses the already-cheap reduced solve.

    Parameters
    ----------
    K : ndarray (n_dof, n_dof)
        Full-order stiffness matrix.
    basis : ndarray (n_dof, n_modes), or an object with a .V attribute
        The (typically modal) basis mode displacement is being
        truncated to.
    F : ndarray (n_dof,) or (n_dof, k)
        Static load (or a batch of k loads, one per column).

    Returns
    -------
    q_cor : ndarray, same shape as F
        The static correction vector -- feed to mode_acceleration_
        response() (post-hoc correction) or augmented_basis() (folded
        into the basis itself, modal truncation augmentation).
    """
    K = _as_array(K)
    V = _basis_matrix(basis)
    F = _as_array(F)
    x_static = np.linalg.solve(K, F)
    K_r = V.T @ K @ V
    F_r = V.T @ F
    q_r = np.linalg.solve(K_r, F_r)
    return x_static - V @ q_r


def mode_acceleration_response(basis, eta, q_cor):
    """x_MA = V @ eta + q_cor -- the mode-acceleration-corrected
    response (eq. 13), given the ordinary reduced modal coordinates eta
    (e.g. from GalerkinROM(basis).solve_static()'s second return value)
    and the static correction q_cor from mode_acceleration_correction()
    (computed for the SAME load F that produced eta).
    """
    V = _basis_matrix(basis)
    return V @ _as_array(eta) + _as_array(q_cor)


# -----------------------------------------------------------------------
# Modal truncation augmentation (eq. 20)
# -----------------------------------------------------------------------
def augmented_basis(basis, q_cor, M=None, tol=1e-10):
    """Psi = [V, q_cor_orthogonalized] -- fold the static correction
    INTO the basis (eq. 20) instead of applying it as a post-hoc
    correction, so it can be used with ordinary GalerkinROM machinery
    (static OR modal solves, unlike mode_acceleration_response() which
    only makes sense for the static case it was computed for).

    q_cor is first orthogonalized against the existing basis columns --
    plain Euclidean Gram-Schmidt if M is None, or M-orthogonalization
    (Gram-Schmidt in the M inner product, matching the inner product
    GalerkinROM.solve_modal()'s eigh(K_r, M_r) implicitly uses) if a
    mass matrix M is given -- and then normalized, so the appended
    column can't leave the augmented reduced mass/stiffness matrices
    ill-conditioned when q_cor happens to already be nearly in the span
    of V (a real possibility: if V already captures the load shape well,
    the static correction is small by construction).

    Parameters
    ----------
    basis : ndarray (n_dof, n_modes), or an object with a .V attribute
    q_cor : ndarray (n_dof,)
        A SINGLE static correction vector (mode_acceleration_correction()'s
        output for one load pattern) -- eq. 20 augments by exactly one
        column per load pattern being specifically accounted for.
    M : ndarray (n_dof, n_dof), optional
        Mass matrix, for M-orthogonalization. Recommended whenever the
        augmented basis will subsequently be used for solve_modal() (a
        generalized eigenproblem in the M inner product) -- omit only
        for a purely static use of the augmented basis.
    tol : float
        If q_cor's component orthogonal to V has norm below tol times
        q_cor's own norm, it is judged to already be (numerically) in
        the span of V, and a ValueError is raised rather than silently
        appending a near-zero, ill-conditioning column.

    Returns
    -------
    Psi : ndarray (n_dof, n_modes + 1)
    """
    V = _basis_matrix(basis)
    q_cor = _as_array(q_cor)
    if q_cor.ndim != 1:
        raise ValueError(f"q_cor must be a single (n_dof,) vector, got shape {q_cor.shape}")
    q_norm = np.linalg.norm(q_cor)

    if M is None:
        coeffs = V.T @ q_cor
        resid = q_cor - V @ coeffs
        resid_norm = np.linalg.norm(resid)
    else:
        M = _as_array(M)
        V_M_V = V.T @ M @ V
        V_M_q = V.T @ (M @ q_cor)
        coeffs = np.linalg.solve(V_M_V, V_M_q)
        resid = q_cor - V @ coeffs
        resid_norm = float(np.sqrt(resid @ (M @ resid)))

    if resid_norm < tol * max(q_norm, 1e-300):
        raise ValueError(
            "q_cor lies (numerically) entirely in the span of the existing "
            "basis -- the static correction adds nothing new here, so "
            "modal truncation augmentation has no augmented column to add "
            "(this can happen when the basis already captures the load "
            "shape well; mode_acceleration_response() is still valid, or "
            "increase tol if this is expected)."
        )
    return np.hstack([V, (resid / resid_norm).reshape(-1, 1)])
