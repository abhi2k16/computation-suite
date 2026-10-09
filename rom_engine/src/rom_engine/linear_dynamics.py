# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
linear_dynamics.py -- time-domain response of LINEAR (full or reduced) structural systems.

`GalerkinROM` only offered static and modal solves, and `nonlinear_dynamics.py`'s Newmark integrator
is tied to the nonlinear surrogate force models. This module fills the plain linear gap:

    M q'' + C q' + K q = F(t)

  * `newmark_linear`        implicit Newmark-beta on any (M, C, K) (default average acceleration:
                             unconditionally stable, second order). The effective matrix is factored
                             ONCE, so each step costs one back-substitution.
  * `modal_superposition`   classical modal truncation: solve the lowest `n_modes` modes of (K, M),
                             integrate each modal equation EXACTLY for a load that is piecewise linear
                             in time (no time-step error from the integrator), then sum. Damping:
                             constant ratio `zeta` or Rayleigh `(alpha, beta)`.
  * `galerkin_transient`    convenience wrapper: Newmark on the reduced matrices of a `GalerkinROM`
                             built with `reduce_system(K, M, F)`, returning the response expanded to
                             full-order coordinates.
  * `piecewise_linear_exact` the exact discrete propagator used by `modal_superposition`, exposed
                             because it is useful on its own (any linear state-space system).

The load may be given as an array of shape (n_steps+1, n) sampled at t_k = k*dt, or as a callable
`f(t) -> (n,)` evaluated at those times.

CHECKED against closed-form single-DOF solutions (free decay, step load) and against each other and
the full-order response on a real fea_engine beam (see tests/test_linear_dynamics.py).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import eigh, expm, lu_factor, lu_solve


def _arr(x):
    return np.asarray(x, dtype=float)


def _load_array(load, n, dt, n_steps):
    t = np.arange(n_steps + 1) * dt
    if callable(load):
        F = np.array([_arr(load(tk)) for tk in t])
    else:
        F = _arr(load)
        if F.ndim == 1 and F.shape[0] == n:
            F = np.tile(F, (n_steps + 1, 1))
    if F.shape != (n_steps + 1, n):
        raise ValueError(f"load must have shape ({n_steps + 1}, {n}) or be a callable/constant vector, got {F.shape}")
    return t, F


def newmark_linear(M, C, K, load, dt, n_steps, q0=None, v0=None, beta=0.25, gamma=0.5):
    """Newmark-beta integration of M q'' + C q' + K q = F(t).

    Returns (t, q, v, a), each of shape (n_steps+1, n) except t (n_steps+1,)."""
    M, K = _arr(M), _arr(K)
    n = M.shape[0]
    C = np.zeros_like(M) if C is None else _arr(C)
    t, F = _load_array(load, n, dt, n_steps)
    q = np.zeros((n_steps + 1, n)); v = np.zeros_like(q); a = np.zeros_like(q)
    if q0 is not None:
        q[0] = q0
    if v0 is not None:
        v[0] = v0
    a[0] = np.linalg.solve(M, F[0] - C @ v[0] - K @ q[0])
    a0 = 1.0 / (beta * dt**2); a1 = gamma / (beta * dt); a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2 * beta) - 1.0; a4 = gamma / beta - 1.0; a5 = dt * (gamma / (2 * beta) - 1.0)
    lu = lu_factor(K + a1 * C + a0 * M)
    for k in range(n_steps):
        rhs = (F[k + 1] + M @ (a0 * q[k] + a2 * v[k] + a3 * a[k]) + C @ (a1 * q[k] + a4 * v[k] + a5 * a[k]))
        q[k + 1] = lu_solve(lu, rhs)
        a[k + 1] = a0 * (q[k + 1] - q[k]) - a2 * v[k] - a3 * a[k]
        v[k + 1] = v[k] + dt * ((1 - gamma) * a[k] + gamma * a[k + 1])
    return t, q, v, a


def piecewise_linear_exact(A, B, U, dt, x0):
    """Exact solution of x' = A x + B u(t) at t_k = k dt, for u linear between the samples U[k].

    Uses one matrix exponential of the augmented system [[A, B, 0], [0, 0, I], [0, 0, 0]]; no
    time-stepping error beyond the (assumed) linearity of u within a step. Returns X (n_steps+1, n)."""
    A, B, U = _arr(A), _arr(B), _arr(U)
    n, m = B.shape
    big = np.zeros((n + 2 * m, n + 2 * m))
    big[:n, :n] = A; big[:n, n:n + m] = B; big[n:n + m, n + m:] = np.eye(m)
    E = expm(big * dt)
    Phi, G1, G2 = E[:n, :n], E[:n, n:n + m], E[:n, n + m:]
    X = np.zeros((U.shape[0], n)); X[0] = x0
    for k in range(U.shape[0] - 1):
        w = (U[k + 1] - U[k]) / dt
        X[k + 1] = Phi @ X[k] + G1 @ U[k] + G2 @ w
    return X


def modal_superposition(K, M, load, dt, n_steps, n_modes, zeta=0.0, rayleigh=None, u0=None, v0=None):
    """Modal-superposition transient with exact modal integration.

    Parameters
    ----------
    K, M : (n, n) symmetric, M positive definite.
    n_modes : number of lowest modes kept.
    zeta : constant modal damping ratio (used when `rayleigh` is None).
    rayleigh : (alpha, beta) so that C = alpha M + beta K gives zeta_i = alpha/(2 w_i) + beta w_i / 2.

    Returns (t, u, info) with u of shape (n_steps+1, n) and info = dict(freq_hz, zeta, Phi)."""
    K, M = _arr(K), _arr(M)
    n = K.shape[0]
    w2, Phi = eigh(K, M)
    k = int(min(n_modes, n))
    w = np.sqrt(np.clip(w2[:k], 0.0, None)); Phi = Phi[:, :k]
    if rayleigh is not None:
        al, be = rayleigh
        z = al / (2 * np.maximum(w, 1e-300)) + be * w / 2
    else:
        z = np.full(k, float(zeta))
    t, F = _load_array(load, n, dt, n_steps)
    P = F @ Phi                                  # modal loads (n_steps+1, k)
    X0 = np.zeros((k, 2))
    if u0 is not None:
        X0[:, 0] = Phi.T @ (M @ _arr(u0))
    if v0 is not None:
        X0[:, 1] = Phi.T @ (M @ _arr(v0))
    Q = np.zeros((n_steps + 1, k))
    for i in range(k):
        A = np.array([[0.0, 1.0], [-w[i] ** 2, -2 * z[i] * w[i]]])
        Bm = np.array([[0.0], [1.0]])
        Q[:, i] = piecewise_linear_exact(A, Bm, P[:, i:i + 1], dt, X0[i])[:, 0]
    return t, Q @ Phi.T, dict(freq_hz=w / (2 * np.pi), zeta=z, Phi=Phi)


def galerkin_transient(rom, load, dt, n_steps, C=None, M=None, u0=None, v0=None, beta=0.25, gamma=0.5):
    """Newmark on the reduced matrices of a `GalerkinROM` (built with `reduce_system(K, M, F)`).

    `load` is the FULL-order load (array (n_steps+1, n_dof), callable, or constant vector); `C` an
    optional full-order damping matrix (projected here). Full-order initial conditions `u0`, `v0`
    are projected through M_r and need the full-order mass `M`. Returns (t, U_full), U_full (n_steps+1, n_dof)."""
    if rom.K_r is None or rom.M_r is None:
        raise ValueError("call rom.reduce_system(K, M=M, F=F) first (M is required for dynamics)")
    V = rom.V
    n = V.shape[0]
    t, F = _load_array(load, n, dt, n_steps)
    Fr = F @ V
    Cr = None if C is None else rom.project_matrix(C)
    if (u0 is not None or v0 is not None) and M is None:
        raise ValueError("full-order initial conditions need the full-order mass matrix M")
    q0 = None if u0 is None else np.linalg.solve(rom.M_r, V.T @ (_arr(M) @ _arr(u0)))
    qd0 = None if v0 is None else np.linalg.solve(rom.M_r, V.T @ (_arr(M) @ _arr(v0)))
    t, q, _, _ = newmark_linear(rom.M_r, Cr, rom.K_r, Fr, dt, n_steps, q0, qd0, beta, gamma)
    return t, q @ V.T

