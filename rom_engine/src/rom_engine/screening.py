"""
screening.py -- automated cross-ROM stability screening for
non-intrusive Loewner-pencil modal identification.

A single LoewnerROM.fit() call (loewner.py) with an interpolation order
`n_interp` larger than the true number of modes in-band -- deliberately
oversized, per the reference paper's own guidance, since undersizing it
risks MISSING a genuine mode entirely -- generically returns some
NUMBER of extra, non-physical eigenpairs alongside the genuine ones.
Those spurious eigenpairs are artifacts of that one particular
interpolation-point choice, not of the underlying structure: a
DIFFERENT random choice of interpolation points produces a DIFFERENT
set of spurious eigenpairs, while the genuine physical modes show up
-- at very nearly the same (f, eta) -- in every ROM, regardless of
which interpolation points were used to build it.

This module turns that observation into an automated filter (Eqs.
30-32 of the reference paper's "Automated Screening of Physical Modes"
algorithm): build MANY LoewnerROMs from independent random
interpolation-frequency subsets of a shared frequency pool, connect
eigenpairs that agree closely across DIFFERENT ROMs into clusters (a
graph, via scipy.sparse.csgraph.connected_components), then keep only
the clusters that are both tightly clustered (low coefficient of
variation in both f and eta) and recur in a large fraction of the ROMs
-- exactly the two properties a numerical artifact of one specific
interpolation choice would NOT have.

Reference: Liu, J. & Li, S. (2026), J. Vib. Eng. Technol. 14:335,
Eqs. (30)-(32). Direct, from-scratch port of the working prototype
`rom_modal_identification.py`'s `multi_rom_screening()`, with ONE
deliberate change: `rng` is a required, explicit parameter here, not a
hidden module-global default (see
docs/loewner_modal_identification_roadmap.md Section 2) -- every other
randomized construction in this package (there are none yet outside
this module, but the convention is set here for any future one) should
follow the same pattern: no function silently reads global RNG state a
caller cannot see or control.
"""
from typing import NamedTuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .loewner import LoewnerROM


class ScreenedMode(NamedTuple):
    """One physical mode surviving cross-ROM stability screening.
    Behaves as a plain (f, eta, s_total, occurrence_count) tuple (so
    existing unpacking code keeps working) while also being
    field-addressable (mode.f, mode.eta, ...) for readability."""
    f: float
    eta: float
    s_total: float
    occurrence_count: int


def screen_physical_modes(freq_pool, x_pool, fmin, fmax, rng,
                           n_roms=20, n_interp=9,
                           eps_f=0.005, eps_eta=0.10, tau_s=0.95):
    """Eqs. (30)-(32): build `n_roms` LoewnerROMs from random
    interpolation-frequency subsets of `freq_pool`, then use the
    cross-ROM stability criterion to separate genuine physical modes
    from spurious numerical ones.

    Parameters
    ----------
    freq_pool : array_like, shape (m,)
        Pool of frequencies [rad/s] at which single-DOF responses were
        measured/simulated (m must be >= 2*n_interp).
    x_pool : array_like, shape (m,), complex
        Corresponding complex response samples.
    fmin, fmax : float
        Frequency band [Hz] a mode must fall in to be considered.
    rng : numpy.random.Generator
        Required, explicit source of randomness for the n_roms
        interpolation-subset draws (see module docstring) -- pass
        np.random.default_rng(seed) for a reproducible run.
    n_roms : int, default 20
        Number of independent LoewnerROMs to build and cross-check.
    n_interp : int, default 9
        Interpolation order per ROM (deliberately oversized relative
        to the expected number of true in-band modes -- see module
        docstring).
    eps_f, eps_eta : float
        Relative-difference thresholds (Eq. 30) below which two
        eigenpairs FROM DIFFERENT ROMS are considered "the same mode"
        and connected in the stability graph.
    tau_s : float
        Minimum per-cluster stability index S_total = Sf * Se
        (Eqs. 31-32) required to accept a cluster as physical.

    Returns
    -------
    list of ScreenedMode
        Sorted by frequency. Each entry is (f, eta, s_total,
        occurrence_count) -- occurrence_count is the number of DISTINCT
        ROMs (out of n_roms) that produced an eigenpair in this
        cluster; a cluster seen only once or twice can never pass the
        n_roms // 2 recurrence requirement below and is discarded in
        Step 1 of the loop, before ever reaching the stability check.
    """
    freq_pool = np.asarray(freq_pool, dtype=float)
    x_pool = np.asarray(x_pool, dtype=complex)
    if len(freq_pool) != len(x_pool):
        raise ValueError(
            f"freq_pool and x_pool must have the same length, got "
            f"{len(freq_pool)} and {len(x_pool)}")
    if len(freq_pool) < 2 * n_interp:
        raise ValueError(
            f"freq_pool has only {len(freq_pool)} points, need at least "
            f"2*n_interp = {2 * n_interp} to draw {n_roms} independent "
            f"interpolation subsets")

    all_f, all_eta, rom_id = [], [], []

    for r in range(n_roms):
        idx = rng.choice(len(freq_pool), size=2 * n_interp, replace=False)
        idx_a, idx_b = idx[:n_interp], idx[n_interp:]
        rom = LoewnerROM.fit(freq_pool[idx_a], freq_pool[idx_b],
                              x_pool[idx_a], x_pool[idx_b])

        # Step 1, "Preprocessing": drop obviously nonphysical solutions
        keep = (rom.f >= fmin) & (rom.f <= fmax) & (rom.eta >= 0) & (rom.eta <= 1)
        all_f.extend(rom.f[keep]); all_eta.extend(rom.eta[keep])
        rom_id.extend([r] * int(keep.sum()))

    all_f = np.array(all_f); all_eta = np.array(all_eta); rom_id = np.array(rom_id)
    n_pts = len(all_f)

    # Step 2, "Graph construction": pairing condition Eq. (30), only
    # between eigenpairs coming from DIFFERENT ROM configurations
    rows, cols = [], []
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if rom_id[i] == rom_id[j]:
                continue
            df = abs(all_f[i] - all_f[j]) / max(all_f[i], all_f[j])
            de = abs(all_eta[i] - all_eta[j]) / max(all_eta[i], all_eta[j], 1e-12)
            if df < eps_f and de < eps_eta:
                rows.append(i); cols.append(j)

    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_pts, n_pts))
    n_clusters, labels = connected_components(adj, directed=False)

    # Steps 3-5: per-cluster stability index (Eq. 31-32) and decision
    physical = []
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        if len(members) < 2:
            continue  # never tracked across ROMs -> discard outright
        mu_f, sig_f = all_f[members].mean(), all_f[members].std()
        mu_e, sig_e = all_eta[members].mean(), all_eta[members].std()
        Sf = 1 - sig_f / mu_f
        Se = 1 - sig_e / mu_e if mu_e > 0 else 0.0
        S_total = Sf * Se
        n_unique_roms = len(set(rom_id[members]))
        if S_total > tau_s and n_unique_roms >= n_roms // 2:
            physical.append(ScreenedMode(mu_f, mu_e, S_total, n_unique_roms))

    physical.sort(key=lambda mode: mode.f)
    return physical
