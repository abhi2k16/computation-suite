# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_galerkin.py -- validates rom_engine.galerkin.GalerkinROM against
real fea_engine cantilever-beam models.

Checks:
  1. A full-RANK basis (an orthonormal basis spanning the ENTIRE free
     dof space, e.g. from QR of a random full-rank matrix) makes the
     Galerkin static solve reproduce the exact full-order solution --
     projecting onto "everything" should lose nothing.
  2. If the basis is built from POD of a specific SET of static-load
     snapshots, then solving for ANY load that is a linear combination
     of those same snapshot loads is reproduced EXACTLY (not just
     approximately) -- the defining property of Galerkin projection
     when the true solution actually lies in the basis's span.
  3. A POD basis built from STATIC-load snapshots (not eigenmode
     snapshots -- using eigenmodes as both the training data and the
     validation target would be circular), when used for solve_modal(),
     produces frequencies that converge towards the full model's own
     fea_engine solve_modal() frequencies as the basis rank grows. This
     is the real test that a Galerkin ROM built for one purpose (static
     response) still captures meaningful dynamics, not just a
     software-plumbing check.
  4. reduce_system()'s caching behaves correctly: calling solve_static()
     with a NEW F re-projects only F_r, without needing K_r/M_r
     recomputed, and gives the same answer as passing F to
     reduce_system() directly.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.linalg import eigh, qr
from rom_engine import PodBasis, GalerkinROM
import fea_fixtures as ff


def test_full_rank_basis_reproduces_exact_static_solution():
    fx = ff.cantilever_beam_system(n=10)
    free = fx["free_dofs"]
    n_free = len(free)
    K_ff = fx["K"][np.ix_(free, free)]

    rng = np.random.default_rng(7)
    Q, _ = qr(rng.standard_normal((n_free, n_free)))   # full-rank orthonormal basis

    F_free = rng.standard_normal(n_free)
    x_exact = np.linalg.solve(K_ff, F_free)

    rom = GalerkinROM(Q).reduce_system(K_ff, F=F_free)
    x_full, q = rom.solve_static()
    err = np.max(np.abs(x_full - x_exact)) / np.max(np.abs(x_exact))
    print(f"full-rank basis static solve, max relative error: {err:.3e}")
    assert err < 1e-8, "a full-rank orthonormal basis must reproduce the exact solution"
    print("PASS -- full-rank Galerkin ROM matches the exact full-order static solve")


def test_load_in_span_reproduced_exactly():
    fx, snaps, loads = ff.cantilever_static_snapshots(n=16, n_loads=5, seed=11)
    free = fx["free_dofs"]
    K_ff = fx["K"][np.ix_(free, free)]
    snaps_free = snaps[free, :]
    loads_free = loads[free, :]

    basis = PodBasis().fit(snaps_free, n_modes=5)   # full rank of the snapshot set
    rom = GalerkinROM(basis)

    # a NEW load that's a random linear combination of the training loads
    rng = np.random.default_rng(12)
    coeffs = rng.standard_normal(5)
    F_new = loads_free @ coeffs
    x_exact = np.linalg.solve(K_ff, F_new)

    rom.reduce_system(K_ff)
    x_full, q = rom.solve_static(F=F_new)
    err = np.max(np.abs(x_full - x_exact)) / np.max(np.abs(x_exact))
    print(f"in-span load static solve, max relative error: {err:.3e}")
    assert err < 1e-8, (
        "Galerkin projection must be exact for any load whose response lies "
        "in the span of the POD basis (here, by construction, a linear "
        "combination of the training loads' own responses)")
    print("PASS -- Galerkin ROM exactly reproduces the response to an in-span load")


def test_modal_frequencies_converge_with_static_snapshot_basis_rank():
    fx, snaps, loads = ff.cantilever_static_snapshots(n=20, n_loads=12, seed=21)
    free = fx["free_dofs"]
    K_ff = fx["K"][np.ix_(free, free)]
    M_ff = fx["M"][np.ix_(free, free)]
    snaps_free = snaps[free, :]

    # ground truth: fea_engine's own full-order modal solve
    eigvals_full, _ = eigh(K_ff, M_ff)
    freq_full = np.sqrt(np.clip(eigvals_full, 0, None)) / (2 * np.pi)
    n_check = 3   # compare the lowest 3 modes
    freq_full_low = freq_full[:n_check]
    print(f"full-order lowest {n_check} frequencies (Hz): {freq_full_low}")

    ranks = [2, 4, 6, 8, 10, 12]
    rel_errs = []
    for r in ranks:
        basis = PodBasis().fit(snaps_free, n_modes=r)
        rom = GalerkinROM(basis).reduce_system(K_ff, M=M_ff)
        freq_rom, _, _ = rom.solve_modal(n_modes=min(n_check, r))
        # pad with nan if the ROM has fewer than n_check modes at low rank
        padded = np.full(n_check, np.nan)
        padded[:len(freq_rom)] = freq_rom
        err = np.abs(padded - freq_full_low) / freq_full_low
        rel_errs.append(err)
        print(f"rank={r:2d}  ROM freqs={np.round(padded, 3)}  rel err={np.round(err, 4)}")

    rel_errs = np.array(rel_errs)   # (len(ranks), n_check)
    # mode 0 (lowest, best-excited by generic static loads) should show a
    # clear overall improvement from the smallest to the largest basis
    mode0_first, mode0_last = rel_errs[0, 0], rel_errs[-1, 0]
    print(f"mode 0 relative error: rank={ranks[0]} -> {mode0_first:.4f}, "
          f"rank={ranks[-1]} -> {mode0_last:.4f}")
    assert mode0_last < mode0_first, (
        "the lowest mode's frequency error should shrink as more static-load "
        "snapshots (and hence more POD modes) are included in the basis")
    assert mode0_last < 0.05, (
        "at the largest tested basis rank, the lowest mode should be within "
        "5% of the true full-order frequency")
    print("PASS -- modal frequencies from a STATIC-snapshot POD basis converge "
          "towards the true full-order frequencies as rank grows")


def test_reduce_system_caching_and_direct_F_agree():
    fx, snaps, loads = ff.cantilever_static_snapshots(n=12, n_loads=4, seed=31)
    free = fx["free_dofs"]
    K_ff = fx["K"][np.ix_(free, free)]
    snaps_free = snaps[free, :]
    basis = PodBasis().fit(snaps_free, n_modes=4)

    rng = np.random.default_rng(32)
    F_new = rng.standard_normal(len(free))

    rom_a = GalerkinROM(basis).reduce_system(K_ff, F=F_new)
    x_a, q_a = rom_a.solve_static()

    rom_b = GalerkinROM(basis).reduce_system(K_ff)   # no F cached
    x_b, q_b = rom_b.solve_static(F=F_new)            # F passed directly

    diff = np.max(np.abs(x_a - x_b))
    print(f"max diff between cached-F and directly-passed-F solves: {diff:.3e}")
    assert diff < 1e-12, "caching F via reduce_system() or passing it to solve_static() must agree exactly"
    print("PASS -- reduce_system() caching is consistent with direct solve_static(F=...)")


def test_constructor_rejects_non_2d_basis():
    with pytest.raises(ValueError, match="must be 2-D"):
        GalerkinROM(np.zeros(5))   # 1-D, not (n_dof, n_modes)
    print("PASS -- GalerkinROM(1-D basis) raises ValueError")


def test_solve_static_before_reduce_system_raises():
    Q, _ = qr(np.eye(4))
    rom = GalerkinROM(Q)
    with pytest.raises(RuntimeError, match="reduce_system"):
        rom.solve_static()
    print("PASS -- solve_static() before reduce_system() raises RuntimeError")


def test_solve_static_with_no_F_anywhere_raises():
    fx = ff.cantilever_beam_system(n=8)
    free = fx["free_dofs"]
    K_ff = fx["K"][np.ix_(free, free)]
    rng = np.random.default_rng(61)
    Q, _ = qr(rng.standard_normal((len(free), len(free))))

    rom = GalerkinROM(Q).reduce_system(K_ff)   # no F given here or cached
    with pytest.raises(ValueError, match="no F given"):
        rom.solve_static()
    print("PASS -- solve_static() with no F given or cached raises ValueError")


def test_solve_modal_before_reduce_system_raises():
    Q, _ = qr(np.eye(4))
    rom = GalerkinROM(Q)
    with pytest.raises(RuntimeError, match="reduce_system"):
        rom.solve_modal()
    print("PASS -- solve_modal() before reduce_system(K, M) raises RuntimeError")
