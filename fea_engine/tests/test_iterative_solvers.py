"""
test_iterative_solvers.py -- Wave 8 items 36-39 (docs/consolidated_
future_roadmap.md): fill-reducing reordering, preconditioners,
conjugate gradients, and classical stationary iterations
(iterative_solvers.py).
"""
__author__ = "Abhijeet"
import numpy as np
import pytest
import scipy.sparse as sp

from fea_engine.iterative_solvers import (
    reverse_cuthill_mckee, fill_in_count, permuted_solve,
    jacobi_preconditioner, ssor_preconditioner, incomplete_cholesky0,
    preconditioned_cg, jacobi_solve, gauss_seidel_solve, sor_solve,
)


def _grid_laplacian(n):
    """n x n 5-point-stencil grid Laplacian, natural row-major
    numbering -- a genuinely FE-like sparse SPD matrix (more nonzeros
    per row than a 1-D tridiagonal system, so IC(0)/RCM have real
    (not vacuously-exact/vacuously-unchanged) work to do)."""
    N = n * n

    def idx(i, j):
        return i * n + j
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            rows.append(k); cols.append(k); vals.append(4.0)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    rows.append(k); cols.append(idx(ii, jj)); vals.append(-1.0)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


class TestFillReducingReordering:
    def test_rcm_reduces_fill_on_scrambled_ordering(self):
        """The regime item 36 targets: an arbitrary (here, randomly
        scrambled) node numbering has much worse fill than a natural
        one, and RCM recovers most of that loss -- fem_implementation_
        lessons.md's own "cuts factor fill by half" framing, checked
        directly rather than assumed."""
        A_natural = _grid_laplacian(12)
        rng = np.random.default_rng(0)
        perm_scramble = rng.permutation(A_natural.shape[0])
        A_scrambled = A_natural[perm_scramble][:, perm_scramble]

        fill_nat = fill_in_count(A_natural)
        fill_scr = fill_in_count(A_scrambled)
        perm = reverse_cuthill_mckee(A_scrambled)
        fill_rcm = fill_in_count(A_scrambled[perm][:, perm])

        assert fill_scr > fill_nat, "a scrambled ordering should have MORE fill than natural"
        assert fill_rcm < fill_scr, "RCM should reduce fill relative to the scrambled ordering"
        # RCM should recover most of the way back toward the natural-order fill.
        assert fill_rcm < 0.5 * (fill_nat + fill_scr)

    def test_permuted_solve_matches_direct(self):
        A = _grid_laplacian(10)
        rng = np.random.default_rng(1)
        b = rng.standard_normal(A.shape[0])
        x_direct = sp.linalg.spsolve(A.tocsc(), b)
        x_perm, perm = permuted_solve(A, b)
        assert np.allclose(x_perm, x_direct, atol=1e-9)
        assert sorted(perm.tolist()) == list(range(A.shape[0]))


class TestPreconditioners:
    def test_jacobi_matches_dense_diag_inverse(self):
        A = _grid_laplacian(6)
        M = jacobi_preconditioner(A)
        r = np.arange(A.shape[0], dtype=float)
        assert np.allclose(M(r), r / 4.0)

    def test_ssor_and_ic0_reduce_cg_iterations(self):
        """The comparative claim fem_implementation_lessons.md cites
        (SSOR/IC(0) meaningfully beat plain and Jacobi-preconditioned
        CG) checked directly on the same matrix/RHS."""
        A = _grid_laplacian(14)
        rng = np.random.default_rng(2)
        b = rng.standard_normal(A.shape[0])

        _, it_none = preconditioned_cg(A, b, tol=1e-10)
        _, it_jacobi = preconditioned_cg(A, b, M=jacobi_preconditioner(A), tol=1e-10)
        _, it_ssor = preconditioned_cg(A, b, M=ssor_preconditioner(A, omega=1.5), tol=1e-10)
        _, it_ic0 = preconditioned_cg(A, b, M=incomplete_cholesky0(A), tol=1e-10)

        # Jacobi is a no-op here (constant diagonal) -- matches the book's
        # own "frequently not very helpful" framing for a constant diagonal.
        assert it_jacobi == it_none
        assert it_ssor < it_none
        assert it_ic0 < it_none

    def test_incomplete_cholesky_solution_is_correct(self):
        A = _grid_laplacian(8)
        rng = np.random.default_rng(3)
        b = rng.standard_normal(A.shape[0])
        M = incomplete_cholesky0(A)
        x, _ = preconditioned_cg(A, b, M=M, tol=1e-10)
        assert np.allclose(x, np.linalg.solve(A.toarray(), b), atol=1e-7)


class TestConjugateGradients:
    def test_cg_matches_dense_solve(self):
        A = _grid_laplacian(10)
        rng = np.random.default_rng(4)
        b = rng.standard_normal(A.shape[0])
        x, n_iter = preconditioned_cg(A, b, tol=1e-10)
        assert np.allclose(x, np.linalg.solve(A.toarray(), b), atol=1e-7)
        assert 0 < n_iter <= A.shape[0]

    def test_cg_raises_on_non_spd(self):
        # symmetric, indefinite (eigenvalues +1, -1); b chosen so the
        # very first search direction p=r=b hits p^T A p = 0 exactly,
        # rather than a value that happens to still be positive for
        # this particular b (indefiniteness is a property of A, not
        # of any specific b, but CG's failure mode is detected via
        # the search direction actually encountered).
        A = np.array([[1.0, 0.0], [0.0, -1.0]])
        b = np.array([1.0, 1.0])
        with pytest.raises(np.linalg.LinAlgError):
            preconditioned_cg(A, b, tol=1e-10)


class TestStationaryIterations:
    def test_jacobi_gs_sor_all_match_direct_solve(self):
        A = _grid_laplacian(8)
        rng = np.random.default_rng(5)
        b = rng.standard_normal(A.shape[0])
        x_direct = np.linalg.solve(A.toarray(), b)

        xj, _ = jacobi_solve(A, b, tol=1e-9, maxiter=20000)
        xgs, _ = gauss_seidel_solve(A, b, tol=1e-9, maxiter=20000)
        xsor, _ = sor_solve(A, b, omega=1.6, tol=1e-9, maxiter=20000)

        assert np.allclose(xj, x_direct, atol=1e-6)
        assert np.allclose(xgs, x_direct, atol=1e-6)
        assert np.allclose(xsor, x_direct, atol=1e-6)

    def test_gauss_seidel_converges_faster_than_jacobi(self):
        """fem_implementation_lessons.md's own Chapter 12 summary:
        Gauss-Seidel/SOR are meant to converge faster than plain
        Jacobi -- checked directly by iteration count on the same
        problem."""
        A = _grid_laplacian(10)
        rng = np.random.default_rng(6)
        b = rng.standard_normal(A.shape[0])
        _, itj = jacobi_solve(A, b, tol=1e-8, maxiter=50000)
        _, itgs = gauss_seidel_solve(A, b, tol=1e-8, maxiter=50000)
        assert itgs < itj

    def test_cg_converges_faster_than_stationary_iterations(self):
        """The headline Chapter 11 claim: CG's iteration count tracks
        sqrt(cond(K)), stationary iterations track cond(K) directly --
        so CG should need far fewer iterations on the same problem."""
        A = _grid_laplacian(14)
        rng = np.random.default_rng(7)
        b = rng.standard_normal(A.shape[0])
        _, it_cg = preconditioned_cg(A, b, tol=1e-8)
        _, it_gs = gauss_seidel_solve(A, b, tol=1e-8, maxiter=50000)
        assert it_cg < it_gs
