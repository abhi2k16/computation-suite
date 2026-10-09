# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
rom_01_pod_basis.py -- Example Gallery: Proper Orthogonal Decomposition
basis extraction (pod.py) on a set of static cantilever-beam load
cases -- the "offline" step every other ROM example in this gallery
builds on: run the full-order model a handful of times, extract the
subspace it actually explores, use that subspace (not the raw dof
count) as the ROM's dimension.

Panel 1: singular value decay of the snapshot matrix -- the elbow
that motivates truncating to a handful of POD modes instead of
keeping the full dof count.
Panel 2: reconstruction relative error vs. number of retained modes,
for two DIFFERENT held-out loads: one that is a random LINEAR
COMBINATION of the training loads (must reach ~machine precision once
enough modes are retained -- since a PodBasis built from n_loads
independent snapshots spans an n_loads-dimensional subspace, and any
combination of the training solutions lies exactly in it, this is a
before-and-after regression test, not tuned artwork) and one that is
a genuinely INDEPENDENT random load, outside that trained subspace
(must plateau above zero -- correctly showing the limits of a basis
fit to a different load distribution, not a subtle bug).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
from fea_fixtures import cantilever_static_snapshots  # noqa: E402

from rom_engine.pod import PodBasis  # noqa: E402

N_ELEM = 20
N_LOADS = 6
SEED = 0


def relative_error_by_rank(basis, K_free, free, x_true_full, max_rank):
    """For ranks 1..max_rank, project x_true onto the first `rank`
    columns of basis.V, reconstruct, and return the relative l2 error
    -- using ONLY the basis (no re-solve), exactly how an online ROM
    query would use a POD basis."""
    x_free = x_true_full[free]
    errs = np.zeros(max_rank)
    for r in range(1, max_rank + 1):
        Vr = basis.V[free, :r]
        q = Vr.T @ x_free                 # least-squares projection (Vr has orthonormal columns)
        x_rec = Vr @ q
        errs[r - 1] = np.linalg.norm(x_rec - x_free) / np.linalg.norm(x_free)
    return errs


def main():
    fx, snaps, loads = cantilever_static_snapshots(n=N_ELEM, n_loads=N_LOADS, seed=SEED)
    free = fx["free_dofs"]
    K_free = fx["K"][np.ix_(free, free)]

    basis = PodBasis().fit(snaps, n_modes=N_LOADS)
    print(f"Snapshot matrix: {snaps.shape[0]} dof x {snaps.shape[1]} snapshots")
    print(f"Singular values: {np.round(basis.singular_values, 6)}")
    print(f"Orthonormality error (V^T V - I): {basis.orthonormality_error():.2e}")
    print(f"Energy captured by all {N_LOADS} modes: {basis.energy_captured()*100:.4f}%")

    rng = np.random.default_rng(SEED + 1)

    # Held-out load #1: a random LINEAR COMBINATION of the training loads
    # -- lies exactly in the span of the training snapshots.
    combo_weights = rng.standard_normal(N_LOADS)
    F_combo = np.zeros(fx["n_dof"])
    F_combo[free] = loads[free] @ combo_weights
    x_combo_full = np.zeros(fx["n_dof"])
    x_combo_full[free] = np.linalg.solve(K_free, F_combo[free])

    # Held-out load #2: a genuinely INDEPENDENT random load -- outside
    # the span the basis was trained on.
    F_indep = np.zeros(fx["n_dof"])
    F_indep[free] = rng.standard_normal(len(free))
    x_indep_full = np.zeros(fx["n_dof"])
    x_indep_full[free] = np.linalg.solve(K_free, F_indep[free])

    err_combo = relative_error_by_rank(basis, K_free, free, x_combo_full, N_LOADS)
    err_indep = relative_error_by_rank(basis, K_free, free, x_indep_full, N_LOADS)

    print(f"\nReconstruction error at full rank ({N_LOADS} modes):")
    print(f"  in-span combination load:     {err_combo[-1]:.3e}  (expect ~machine precision)")
    print(f"  out-of-span independent load: {err_indep[-1]:.3e}  (expect > 0, genuinely unresolved)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ranks = np.arange(1, len(basis.singular_values) + 1)
    ax1.semilogy(ranks, basis.singular_values, 'o-', color='steelblue', markersize=6)
    ax1.set_xlabel('mode index')
    ax1.set_ylabel('singular value')
    ax1.set_title(f'POD singular value decay\n({N_LOADS} independent static-load snapshots, '
                  f'{fx["n_dof"]} dof cantilever)')
    ax1.grid(True, alpha=0.35, which='both')
    ax1.set_xticks(ranks)

    ax2.semilogy(ranks, np.maximum(err_combo, 1e-17), 'o-', color='seagreen', markersize=6,
                 label='held-out load IN the training span (linear combo)')
    ax2.semilogy(ranks, np.maximum(err_indep, 1e-17), 's--', color='tomato', markersize=6,
                 label='held-out load OUTSIDE the training span (independent)')
    ax2.set_xlabel('number of retained POD modes')
    ax2.set_ylabel('relative reconstruction error')
    ax2.set_title('Reconstruction error vs. basis size')
    ax2.grid(True, alpha=0.35, which='both')
    ax2.set_xticks(ranks)
    ax2.legend(fontsize=9)

    fig.suptitle('Proper Orthogonal Decomposition -- cantilever static-load snapshots (rom_engine)',
                 fontsize=13)
    out = os.path.join(os.path.dirname(__file__), 'rom_01_pod_basis.png')
    fig.savefig(out, dpi=150)
    print("\nSaved rom_01_pod_basis.png")


if __name__ == "__main__":
    main()
