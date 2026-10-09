# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
loewner_fixtures.py -- synthetic ground-truth system + FRF-sampling
helpers for validating rom_engine.loewner / rom_engine.screening.

This mirrors the working prototype (rom_modal_identification.py)'s
build_system()/frf()/ground_truth_modes() functions, with ONE
deliberate change: `rng` is an explicit parameter here (not a hidden
module-global default), matching the roadmap's Section 2 deviation and
the "no hidden state" convention every other rom_engine module already
follows.

This file -- like fea_fixtures.py -- is TEST-ONLY code: it builds a
system and simulates a full-order FRF solve purely to GENERATE data and
to check the identification method's answers against a known ground
truth. None of this belongs in loewner.py itself, which never sees M,
C, or K (see loewner.py's module docstring) -- these functions are
exactly the boundary that keeps that true: everything on this side of
`frf()` "knows" the system, everything that consumes its OUTPUT does
not.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import eigh


def build_system(n=15, k0=8.0e5, m0=1.0, alpha=0.8, beta=2.0e-5, rng=None):
    """n-DOF fixed-fixed mass-spring chain with Rayleigh damping
    C = alpha*M + beta*K. Random-ish spring stiffnesses break symmetry
    so no measurement DOF sits exactly at a modal node.

    rng : numpy.random.Generator, required (explicit, no hidden
        default -- see module docstring).
    """
    if rng is None:
        raise ValueError("build_system requires an explicit rng (np.random.default_rng(seed))")
    M = m0 * np.eye(n)
    k = k0 * (1.0 + 0.15 * rng.standard_normal(n + 1))
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k[i] + k[i + 1]
        if i > 0:
            K[i, i - 1] -= k[i]
            K[i - 1, i] -= k[i]
    C = alpha * M + beta * K
    return M, C, K, alpha, beta


def frf(omega, F, M, C, K):
    """Direct FRF solve x(omega) = (K + i*omega*C - omega^2*M)^-1 F.
    Plays the role of "obtain vibration responses through simulation or
    measurement" -- nothing downstream of this function's OUTPUT ever
    sees M, C, K, or F again."""
    omega = np.atleast_1d(omega).astype(complex)
    n = M.shape[0]
    X = np.zeros((n, len(omega)), dtype=complex)
    for k, w in enumerate(omega):
        D = K + 1j * w * C - w ** 2 * M
        X[:, k] = np.linalg.solve(D, F)
    return X  # shape (n_dof, n_omega)


def ground_truth_modes(M, C, K, alpha, beta, fmin, fmax):
    """Closed-form modal solution for a proportionally-damped system,
    used ONLY to check identification results -- never consumed by the
    identification method itself."""
    wn2, Phi = eigh(K, M)
    wn = np.sqrt(np.abs(wn2))
    zeta = (alpha + beta * wn ** 2) / (2 * wn)
    f = wn * np.sqrt(np.clip(1 - zeta ** 2, 0, None)) / (2 * np.pi)
    eta = zeta / np.sqrt(np.clip(1 - zeta ** 2, 1e-12, None))
    mask = (f >= fmin) & (f <= fmax)
    order = np.argsort(f[mask])
    return f[mask][order], eta[mask][order], Phi[:, mask][:, order]
