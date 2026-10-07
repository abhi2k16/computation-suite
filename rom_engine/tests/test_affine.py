"""
test_affine.py -- validates rom_engine.affine.AffineDecomposition against
a real two-region fea_engine beam model.

Checks:
  1. assemble(mu) -- the FULL-order affine reconstruction
     theta_1*K_1 + theta_2*K_2 -- matches fea_engine's own INDEPENDENT
     per-block reassembly (two_region_beam_components()'s K_direct(),
     which calls FESystem.assemble_stiffness() fresh, not via affine.py
     at all) to near machine precision, across MULTIPLE random
     (EI1, EI2) draws -- not just the one pair already smoke-tested
     during development.
  2. "project-then-assemble" equals "assemble-then-project": projecting
     the affine components once and then combining them online
     (project(basis).assemble_reduced(mu)) gives the SAME reduced
     matrix as assembling the full K(mu) first and projecting that
     single result directly (GalerkinROM.project_matrix(assemble(mu))).
     This is the algebraic identity the whole offline-online speedup
     depends on: V^T(aA+bB)V = a(V^T A V) + b(V^T B V), and it's worth
     checking explicitly against REAL, not toy, matrices.
  3. A basic timing demonstration: after the one-time project() cost,
     repeated assemble_reduced() online queries are substantially
     cheaper than repeated full assemble() calls -- loosely checked
     (order-of-magnitude, not a tight bound) to avoid a flaky test on
     a shared/slow sandbox machine.
"""
import time
import numpy as np
import pytest
from scipy.linalg import qr
from rom_engine import AffineDecomposition, GalerkinROM
import fea_fixtures as ff


def _theta_two_region(mu):
    return [mu[0], mu[1]]


def test_full_order_assembly_matches_independent_reassembly():
    comps = ff.two_region_beam_components(n=20)
    K1, K2 = comps["K1"], comps["K2"]
    K_direct = comps["K_direct"]

    affine = AffineDecomposition([K1, K2], _theta_two_region)

    rng = np.random.default_rng(41)
    max_errs = []
    for _ in range(6):
        EI1, EI2 = rng.uniform(0.5, 5.0, size=2)
        K_affine = affine.assemble((EI1, EI2))
        K_true = K_direct(EI1, EI2)
        err = np.max(np.abs(K_affine - K_true)) / np.max(np.abs(K_true))
        max_errs.append(err)
        print(f"EI1={EI1:.4f}, EI2={EI2:.4f} -> max relative diff vs independent "
              f"reassembly: {err:.3e}")

    assert all(e < 1e-9 for e in max_errs), (
        "affine.assemble(mu) must match fea_engine's own independent per-block "
        "reassembly to near machine precision for every parameter draw -- this "
        "is an EXACT algebraic identity, not an approximation")
    print("PASS -- affine full-order assembly matches independent fea_engine "
          "reassembly across multiple random parameter draws")


def test_project_then_assemble_equals_assemble_then_project():
    comps = ff.two_region_beam_components(n=16)
    K1, K2 = comps["K1"], comps["K2"]
    n_dof = comps["n_dof"]

    rng = np.random.default_rng(42)
    # mode='economic' is essential here: scipy.linalg.qr defaults to the
    # FULL decomposition (Q is (n_dof, n_dof)) unlike numpy.linalg.qr's
    # default -- without 'economic' this silently builds a full-rank
    # "basis" instead of the actually-reduced (n_dof, 6) one intended.
    Q, _ = qr(rng.standard_normal((n_dof, 6)), mode="economic")   # arbitrary orthonormal reduced basis

    affine = AffineDecomposition([K1, K2], _theta_two_region).project(Q)
    rom = GalerkinROM(Q)

    max_errs = []
    for _ in range(5):
        EI1, EI2 = rng.uniform(0.2, 8.0, size=2)
        Kr_online = affine.assemble_reduced((EI1, EI2))          # project-then-sum
        Kr_direct = rom.project_matrix(affine.assemble((EI1, EI2)))  # sum-then-project
        err = np.max(np.abs(Kr_online - Kr_direct))
        max_errs.append(err)
        print(f"EI1={EI1:.4f}, EI2={EI2:.4f} -> max abs diff, "
              f"project-then-sum vs sum-then-project: {err:.3e}")

    assert all(e < 1e-8 for e in max_errs), (
        "V^T(theta1*K1+theta2*K2)V must equal theta1*(V^T K1 V)+theta2*(V^T K2 V) "
        "-- the linearity identity the offline-online speedup relies on")
    print("PASS -- offline projection commutes with the affine sum, as required")


def test_reduced_online_queries_are_faster_than_full_reassembly():
    comps = ff.two_region_beam_components(n=600)   # a larger model, so the
    # O(n_dof^2) cost of full reassembly is big enough to clearly dominate
    # Python call overhead and show the O(n_modes^2) reduced query's speedup
    K1, K2 = comps["K1"], comps["K2"]
    n_dof = comps["n_dof"]

    rng = np.random.default_rng(43)
    # mode='economic': see note in test_project_then_assemble_equals_assemble_then_project
    Q, _ = qr(rng.standard_normal((n_dof, 8)), mode="economic")

    affine = AffineDecomposition([K1, K2], _theta_two_region).project(Q)

    n_queries = 200
    mus = [tuple(rng.uniform(0.5, 5.0, size=2)) for _ in range(n_queries)]

    t0 = time.perf_counter()
    for mu in mus:
        affine.assemble(mu)
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    for mu in mus:
        affine.assemble_reduced(mu)
    t_reduced = time.perf_counter() - t0

    speedup = t_full / max(t_reduced, 1e-12)
    print(f"{n_queries} queries: full assemble()={t_full*1e3:.2f}ms, "
          f"assemble_reduced()={t_reduced*1e3:.2f}ms, speedup={speedup:.1f}x")
    # loose bound -- just confirm the reduced path is meaningfully faster,
    # not a tight performance regression test that could flake on a busy
    # shared sandbox
    assert speedup > 3, (
        "online reduced queries should be substantially faster than full "
        "(n_dof x n_dof) reassembly once the basis is small relative to n_dof")
    print("PASS -- reduced online queries are substantially faster than full assembly")


def test_constructor_rejects_empty_components():
    with pytest.raises(ValueError, match="at least one component"):
        AffineDecomposition([], _theta_two_region)
    print("PASS -- AffineDecomposition([], ...) raises ValueError")


def test_constructor_rejects_mismatched_component_shapes():
    K1 = np.eye(4)
    K2 = np.eye(5)
    with pytest.raises(ValueError, match="same shape"):
        AffineDecomposition([K1, K2], _theta_two_region)
    print("PASS -- mismatched component shapes raise ValueError")


def test_theta_func_wrong_length_raises():
    K1, K2 = np.eye(4), np.eye(4)
    affine = AffineDecomposition([K1, K2], lambda mu: [mu[0]])   # only 1, need 2
    with pytest.raises(ValueError, match="expected 2"):
        affine.assemble((1.0,))
    print("PASS -- theta_func(mu) returning the wrong number of coefficients raises ValueError")


def test_project_rejects_basis_with_wrong_row_count():
    comps = ff.two_region_beam_components(n=10)
    K1, K2 = comps["K1"], comps["K2"]
    n_dof = comps["n_dof"]
    affine = AffineDecomposition([K1, K2], _theta_two_region)

    bad_basis = np.zeros((n_dof + 1, 3))   # wrong number of rows
    with pytest.raises(ValueError, match="basis has"):
        affine.project(bad_basis)
    print("PASS -- project() with a row-count-mismatched basis raises ValueError")


def test_assemble_reduced_before_project_raises():
    K1, K2 = np.eye(4), np.eye(4)
    affine = AffineDecomposition([K1, K2], _theta_two_region)
    with pytest.raises(RuntimeError, match="call project"):
        affine.assemble_reduced((1.0, 2.0))
    print("PASS -- assemble_reduced() before project() raises RuntimeError")


def test_solve_reduced_matches_manual_assemble_and_solve():
    comps = ff.two_region_beam_components(n=16)
    K1, K2 = comps["K1"], comps["K2"]
    n_dof = comps["n_dof"]

    rng = np.random.default_rng(51)
    Q, _ = qr(rng.standard_normal((n_dof, 6)), mode="economic")
    affine = AffineDecomposition([K1, K2], _theta_two_region).project(Q)

    mu = (1.5, 2.5)
    F_r = rng.standard_normal(6)

    q_solve_reduced = affine.solve_reduced(mu, F_r)
    Kr = affine.assemble_reduced(mu)
    q_manual = np.linalg.solve(Kr, F_r)

    err = np.max(np.abs(q_solve_reduced - q_manual))
    print(f"solve_reduced() vs manual assemble_reduced()+solve, max abs diff: {err:.3e}")
    assert err < 1e-12, "solve_reduced(mu, F_r) must equal np.linalg.solve(assemble_reduced(mu), F_r)"
    print("PASS -- solve_reduced() is consistent with a manual assemble_reduced()+solve")
