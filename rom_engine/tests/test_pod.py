# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_pod.py -- validates rom_engine.pod.PodBasis.

Checks:
  1. Mass-weighted POD, given the EXACT eigenmodes of a real fea_engine
     cantilever beam as its snapshot set, exactly recovers that
     eigenspace (reconstruction error ~0) and produces an M-orthonormal
     basis (orthonormality_error ~0).
  2. Standard (Euclidean) POD on the same snapshots also reconstructs
     them exactly at full rank, but is NOT M-orthonormal in general --
     confirms the two variants are actually doing different things,
     not just relabeling the same computation.
  3. A snapshot set built from N independent static load cases has
     numerical rank exactly N (generic loads, no special structure to
     cause rank deficiency); PodBasis with a tight energy_threshold
     recovers that same rank.
  4. A synthetic, exactly-low-rank problem (snapshots built from a
     KNOWN small set of orthogonal directions with geometrically
     decaying singular values) is truncated by energy_threshold to the
     expected number of modes.
  5. Reconstruction error is monotonically non-increasing as n_modes
     grows, and exactly zero once n_modes reaches the snapshot
     matrix's own rank (Eckart-Young sanity check).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from rom_engine import PodBasis
from scipy.linalg import eigh
import fea_fixtures as ff


def test_mass_weighted_pod_recovers_exact_eigenmodes():
    fx = ff.cantilever_beam_system(n=16)
    free = fx["free_dofs"]
    K_ff = fx["K"][np.ix_(free, free)]
    M_ff = fx["M"][np.ix_(free, free)]
    n_modes = 5

    eigvals, eigvecs = eigh(K_ff, M_ff)
    Phi_exact = eigvecs[:, :n_modes]   # already M-orthonormal (scipy/eigh convention)

    basis = PodBasis().fit(Phi_exact, n_modes=n_modes, M=M_ff)
    print(f"mass-weighted POD orthonormality error: {basis.orthonormality_error():.3e}")
    assert basis.orthonormality_error() < 1e-8, "mass-weighted POD basis should be M-orthonormal"

    recon_err = basis.reconstruction_error(Phi_exact)
    print(f"reconstruction error of the exact eigenmodes themselves: {recon_err:.3e}")
    assert recon_err < 1e-8, (
        "POD basis extracted FROM the exact eigenmodes, at full rank, "
        "must reconstruct them essentially exactly")

    # the recovered subspace should be numerically IDENTICAL to the true
    # eigenspace (not just "some rotation of a similar-energy subspace").
    # Both Phi_exact and basis.V are M-ORTHONORMAL (Phi^T M Phi = I), not
    # Euclidean-orthonormal, so the principal-angle cosines between the two
    # subspaces are the singular values of Phi_exact^T @ M @ V (the
    # M-weighted inner product) -- using a plain Phi_exact^T @ V here would
    # NOT be bounded by 1 and would silently test the wrong thing.
    overlap = np.linalg.svd(Phi_exact.T @ M_ff @ basis.V, compute_uv=False)
    print(f"subspace principal angles (M-weighted cosines, should all be ~1): {np.round(overlap, 8)}")
    assert np.min(overlap) > 1 - 1e-6, "POD subspace should exactly match the true eigenspace"
    print("PASS -- mass-weighted POD exactly recovers the true eigenspace")


def test_standard_pod_reconstructs_but_is_not_mass_orthonormal():
    fx = ff.cantilever_beam_system(n=16)
    free = fx["free_dofs"]
    K_ff = fx["K"][np.ix_(free, free)]
    M_ff = fx["M"][np.ix_(free, free)]
    n_modes = 5
    eigvals, eigvecs = eigh(K_ff, M_ff)
    Phi_exact = eigvecs[:, :n_modes]

    basis_std = PodBasis().fit(Phi_exact, n_modes=n_modes)   # no M -> Euclidean POD
    recon_err = basis_std.reconstruction_error(Phi_exact)
    print(f"standard POD reconstruction error: {recon_err:.3e}")
    assert recon_err < 1e-8, "standard POD at full rank should still reconstruct its own snapshots exactly"

    euclid_orth_err = np.max(np.abs(basis_std.V.T @ basis_std.V - np.eye(n_modes)))
    mass_orth_err = np.max(np.abs(basis_std.V.T @ M_ff @ basis_std.V - np.eye(n_modes)))
    print(f"standard POD: Euclidean orthonormality error = {euclid_orth_err:.3e}, "
          f"mass orthonormality error = {mass_orth_err:.3e}")
    assert euclid_orth_err < 1e-8, "standard POD basis should be Euclidean-orthonormal"
    assert mass_orth_err > 1e-3, (
        "a Euclidean-orthonormal basis of these particular eigenmodes should NOT "
        "generally also be mass-orthonormal -- if this fails, the test fixture's "
        "M is suspiciously close to the identity, not a real check")
    print("PASS -- standard POD reconstructs correctly but confirms Euclidean != mass-orthonormal")


def test_pod_rank_matches_independent_load_count():
    n_loads = 5
    fx, snaps, loads = ff.cantilever_static_snapshots(n=20, n_loads=n_loads, seed=1)
    numerical_rank = np.linalg.matrix_rank(snaps)
    print(f"snapshot matrix numerical rank: {numerical_rank} (built from {n_loads} independent loads)")
    assert numerical_rank == n_loads

    basis = PodBasis().fit(snaps, energy_threshold=1 - 1e-12)
    print(f"PodBasis energy_threshold=1-1e-12 selected n_modes={basis.n_modes}")
    assert basis.n_modes == n_loads, (
        "an (almost) full-energy threshold on a rank-n_loads snapshot set "
        "should retain exactly n_loads modes, not more or fewer")
    assert basis.reconstruction_error(snaps) < 1e-8
    print("PASS -- POD rank matches the known generating rank of the snapshot set")


def test_energy_threshold_truncation_on_synthetic_low_rank_problem():
    rng = np.random.default_rng(2)
    n, k = 40, 10
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)))   # k orthonormal directions
    decay = 0.3 ** np.arange(k)                          # geometric singular-value decay
    X = Q * decay[None, :] @ rng.standard_normal((k, 15))  # (n, 15) snapshots, exactly rank k

    basis_loose = PodBasis().fit(X, energy_threshold=0.90)
    basis_tight = PodBasis().fit(X, energy_threshold=0.999999)
    print(f"loose threshold (0.90) -> n_modes={basis_loose.n_modes}, "
          f"tight threshold (0.999999) -> n_modes={basis_tight.n_modes}")
    assert basis_loose.n_modes <= basis_tight.n_modes
    assert basis_tight.n_modes <= k, "should never need more modes than the true rank"
    assert basis_loose.energy_captured() >= 0.90 - 1e-9
    assert basis_tight.energy_captured() >= 0.999999 - 1e-9
    print("PASS -- energy_threshold truncation behaves sensibly on a known-decay problem")


def test_reconstruction_error_monotonic_and_zero_at_full_rank():
    fx, snaps, loads = ff.cantilever_static_snapshots(n=14, n_loads=6, seed=3)
    errs = []
    for r in range(1, 7):
        basis = PodBasis().fit(snaps, n_modes=r)
        errs.append(basis.reconstruction_error(snaps))
    print("reconstruction error vs n_modes:", [f"{e:.3e}" for e in errs])
    assert all(errs[i] >= errs[i + 1] - 1e-10 for i in range(len(errs) - 1)), (
        "reconstruction error must be non-increasing as more modes are retained")
    assert errs[-1] < 1e-8, "at n_modes == snapshot rank, reconstruction should be essentially exact"
    print("PASS -- reconstruction error decreases monotonically and vanishes at full rank")


def test_fit_requires_n_modes_or_energy_threshold():
    X = np.random.default_rng(71).standard_normal((10, 4))
    with pytest.raises(ValueError, match="n_modes, energy_threshold"):
        PodBasis().fit(X)
    print("PASS -- fit() with neither n_modes nor energy_threshold raises ValueError")


def test_fit_rejects_non_2d_snapshots():
    X = np.random.default_rng(72).standard_normal(10)   # 1-D, not (n_dof, n_snapshots)
    with pytest.raises(ValueError, match="must be 2-D"):
        PodBasis().fit(X, n_modes=2)
    print("PASS -- fit() with 1-D snapshots raises ValueError")


def test_all_zero_snapshots_fall_back_to_rank_one():
    # a degenerate, all-zero snapshot matrix has zero total energy --
    # _select_rank's energy_threshold branch must fall back to rank 1
    # instead of dividing by zero / selecting rank 0.
    X = np.zeros((6, 3))
    basis = PodBasis().fit(X, energy_threshold=0.99)
    print(f"all-zero snapshots -> n_modes={basis.n_modes}")
    assert basis.n_modes == 1, "zero-energy snapshots should fall back to a 1-mode basis, not crash"
    print("PASS -- an all-zero snapshot matrix falls back to a rank-1 basis")


def test_energy_captured_zero_energy_fallback():
    # same degenerate all-zero case, but hitting energy_captured()'s own
    # zero-total-energy guard (reached via an explicit n_modes fit,
    # bypassing _select_rank's energy_threshold branch entirely).
    X = np.zeros((6, 3))
    basis = PodBasis().fit(X, n_modes=2)
    print(f"all-zero snapshots, energy_captured()={basis.energy_captured()}")
    assert basis.energy_captured() == 1.0, (
        "energy_captured() on a zero-energy snapshot set should report 1.0 "
        "(vacuously fully captured) rather than divide by zero")
    print("PASS -- energy_captured() handles a zero-total-energy basis without dividing by zero")


def test_reconstruction_error_absolute_matches_relative_times_norm():
    fx, snaps, loads = ff.cantilever_static_snapshots(n=12, n_loads=5, seed=81)
    basis = PodBasis().fit(snaps, n_modes=2)   # deliberately under-ranked -> nonzero error

    err_abs = basis.reconstruction_error(snaps, relative=False)
    err_rel = basis.reconstruction_error(snaps, relative=True)
    norm_X = np.linalg.norm(snaps, ord="fro")

    print(f"absolute error={err_abs:.3e}, relative error={err_rel:.3e}, "
          f"relative*||X||={err_rel * norm_X:.3e}")
    assert err_abs > 0, "an under-ranked basis should have nonzero absolute reconstruction error"
    assert abs(err_abs - err_rel * norm_X) < 1e-8 * max(norm_X, 1.0), (
        "relative=False must return the raw Frobenius error, i.e. relative error * ||X||")
    print("PASS -- reconstruction_error(relative=False) returns the correct raw (unnormalized) error")
