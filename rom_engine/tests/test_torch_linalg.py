"""
test_torch_linalg.py -- Wave 9 addendum item 138 (docs/consolidated_
future_roadmap.md): validates rom_engine.torch_linalg.py's standalone
primitives directly (the Kronecker-vectorization Lyapunov/Sylvester
solvers, the Gramian square root, and the balancing step), independent
of balanced_truncation.py's own backend="torch" wiring (covered by
test_balanced_truncation_torch.py).

Same two-part split as fea_engine's own test_torch_*.py files and this
project's established convention:

  TestKroneckerVectorizationHelpers -- runs UNCONDITIONALLY, no torch
      needed: the pure-NumPy re-derivation of _vec_f/_unvec_f's own
      column-major convention (checked directly against scipy's
      solve_continuous_lyapunov()/solve_sylvester() using the SAME
      Kronecker-sum construction torch_linalg.py's own torch functions
      use internally) -- this is the actual mathematical claim the
      whole module rests on, worth checking independent of whether
      torch itself is installed in this environment.

  TestTorchPrimitives -- gated on rom_engine.torch_linalg._HAS_TORCH.
      Validates each torch-native primitive directly against its SciPy/
      NumPy counterpart on hand-built systems: torch_solve_continuous_
      lyapunov() vs. scipy.linalg.solve_continuous_lyapunov(), torch_
      solve_sylvester() vs. scipy.linalg.solve_sylvester() (including
      the rectangular case), torch_gramian_square_root() vs. the
      Cholesky/eigh-fallback reference reconstruction P = L @ L.T, and
      torch_balance_from_gramians() vs. balanced_truncation.py's own
      NumPy _balance_from_gramians() (same sigma/T/Tinv, checked via
      the balanced-realization identity Tinv @ T = I and diag(sigma)
      agreement, not just "runs"). Not executed end-to-end in the
      sandbox this file was authored in (no usable torch here); written
      to run for real, and SHOULD be run at least once on a torch-
      equipped machine.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov, solve_sylvester

from rom_engine.torch_linalg import (
    _HAS_TORCH,
    _require_torch,
    torch_solve_continuous_lyapunov,
    torch_solve_sylvester,
    torch_gramian_square_root,
    torch_balance_from_gramians,
)
from rom_engine.balanced_truncation import _balance_from_gramians, _gramian_square_root


def _stable_A(n=6, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return -(M @ M.T) - n * np.eye(n)   # Hurwitz-stable, symmetric-ish


def _spd(n=6, seed=1):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


class TestKroneckerVectorizationHelpers:
    """The pure-NumPy re-derivation of the exact vec_F/Kronecker-sum
    construction torch_linalg.py's own torch functions use internally
    -- this is the mathematical claim the whole module rests on, so it
    gets its own unconditional (no-torch-needed) check, independent of
    whether torch itself is available in this environment."""

    @staticmethod
    def _vec_f(M):
        return M.T.reshape(-1)

    @staticmethod
    def _unvec_f(y, n_rows, n_cols):
        return y.reshape(n_cols, n_rows).T

    def test_lyapunov_kronecker_construction_matches_scipy(self):
        n = 5
        A = _stable_A(n=n, seed=10)
        rng = np.random.default_rng(11)
        Q = rng.standard_normal((n, n))
        Q = Q + Q.T
        X_ref = solve_continuous_lyapunov(A, Q)

        I_n = np.eye(n)
        M = np.kron(I_n, A) + np.kron(A, I_n)
        y = np.linalg.solve(M, self._vec_f(Q))
        X_mine = self._unvec_f(y, n, n)
        assert np.allclose(X_ref, X_mine, atol=1e-8, rtol=1e-8)

    def test_sylvester_kronecker_construction_matches_scipy_rectangular(self):
        m, p = 4, 6
        rng = np.random.default_rng(12)
        A = rng.standard_normal((m, m))
        B = rng.standard_normal((p, p))
        Q = rng.standard_normal((m, p))
        X_ref = solve_sylvester(A, B, Q)

        I_m, I_p = np.eye(m), np.eye(p)
        M = np.kron(I_p, A) + np.kron(B.T, I_m)
        y = np.linalg.solve(M, self._vec_f(Q))
        X_mine = self._unvec_f(y, m, p)
        assert np.allclose(X_ref, X_mine, atol=1e-8, rtol=1e-8)


pytestmark_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not usable in this environment (not installed, or installed "
           "but failing to import -- see torch_linalg.py's own docstring); "
           "these tests validate the torch-native dense linalg primitives "
           "and have nothing to run without it.")


@pytestmark_torch
class TestTorchPrimitives:
    def test_solve_continuous_lyapunov_matches_scipy(self):
        n = 6
        A = _stable_A(n=n, seed=20)
        rng = np.random.default_rng(21)
        Q = rng.standard_normal((n, n))
        Q = Q + Q.T
        X_ref = solve_continuous_lyapunov(A, Q)
        X_torch = torch_solve_continuous_lyapunov(A, Q, device="cpu").cpu().numpy()
        assert np.allclose(X_ref, X_torch, atol=1e-6, rtol=1e-6)

    def test_solve_sylvester_matches_scipy_rectangular(self):
        m, p = 5, 7
        rng = np.random.default_rng(22)
        A = rng.standard_normal((m, m))
        B = rng.standard_normal((p, p))
        Q = rng.standard_normal((m, p))
        X_ref = solve_sylvester(A, B, Q)
        X_torch = torch_solve_sylvester(A, B, Q, device="cpu").cpu().numpy()
        assert np.allclose(X_ref, X_torch, atol=1e-6, rtol=1e-6)

    def test_gramian_square_root_reconstructs_P(self):
        n = 6
        P = _spd(n=n, seed=23)
        L_torch = torch_gramian_square_root(P, device="cpu").cpu().numpy()
        assert np.allclose(L_torch @ L_torch.T, P, atol=1e-6, rtol=1e-6)

    def test_gramian_square_root_matches_numpy_reference_up_to_rotation(self):
        # Cholesky is unique (triangular, positive diagonal) so the
        # torch and numpy Cholesky factors should agree directly (both
        # call the same underlying LAPACK-family algorithm family on a
        # genuinely SPD P, no jitter fallback triggered).
        n = 6
        P = _spd(n=n, seed=24)
        L_numpy = _gramian_square_root(P)
        L_torch = torch_gramian_square_root(P, device="cpu").cpu().numpy()
        assert np.allclose(L_numpy, L_torch, atol=1e-6, rtol=1e-6)

    def test_balance_from_gramians_matches_numpy_backend(self):
        # NOTE: this test originally also compared T_ref/T_torch and
        # Tinv_ref/Tinv_torch elementwise and FAILED for real on a
        # torch-equipped machine -- not a bug in torch_balance_from_
        # gramians() itself, but a wrong assumption in the test: an
        # SVD's singular VECTORS are only unique up to a simultaneous
        # sign flip of a (u_i, v_i) pair, even for perfectly simple
        # (non-degenerate) singular VALUES -- different LAPACK-family
        # backends (NumPy's vs. torch's own SVD, which need not even be
        # the same underlying routine) are not required to agree on
        # that sign choice. Confirmed directly (both analytically and
        # by constructing an artificially sign-flipped T/Tinv pair in a
        # throwaway script) that the sign choice has NO effect on any
        # of the properties that actually matter -- Tinv @ T = I, and
        # the balanced-Gramian diagonalization identities
        # Tinv @ P @ Tinv.T = diag(sigma) = T.T @ Q @ T -- so THOSE are
        # what this test now checks, not the raw T/Tinv arrays.
        n = 6
        P = _spd(n=n, seed=25)
        Q = _spd(n=n, seed=26)
        sigma_ref, T_ref, Tinv_ref = _balance_from_gramians(P, Q)
        sigma_torch, T_torch, Tinv_torch = torch_balance_from_gramians(P, Q, device="cpu")
        assert np.allclose(sigma_ref, sigma_torch, atol=1e-6, rtol=1e-6)
        # Decisive, sign-independent structural checks: Tinv @ T = I
        # (a genuine balancing-transform property)...
        assert np.allclose(Tinv_torch @ T_torch, np.eye(n), atol=1e-6, rtol=1e-6)
        # ...and the actual balancing property both T_ref/Tinv_ref and
        # T_torch/Tinv_torch are SUPPOSED to achieve: transforming P, Q
        # by either pair diagonalizes them to the SAME diag(sigma).
        P_bal_ref = Tinv_ref @ P @ Tinv_ref.T
        Q_bal_ref = T_ref.T @ Q @ T_ref
        P_bal_torch = Tinv_torch @ P @ Tinv_torch.T
        Q_bal_torch = T_torch.T @ Q @ T_torch
        assert np.allclose(P_bal_ref, P_bal_torch, atol=1e-5, rtol=1e-5)
        assert np.allclose(Q_bal_ref, Q_bal_torch, atol=1e-5, rtol=1e-5)
        assert np.allclose(np.diag(P_bal_torch), sigma_torch, atol=1e-6, rtol=1e-6)

    def test_unavailable_torch_fails_fast(self, monkeypatch):
        import rom_engine.torch_linalg as tl_mod

        def _fake_require_torch():
            raise ImportError("simulated: torch unavailable")

        monkeypatch.setattr(tl_mod, "_require_torch", _fake_require_torch)
        A = _stable_A(n=4, seed=27)
        Q = np.eye(4)
        with pytest.raises(ImportError):
            tl_mod.torch_solve_continuous_lyapunov(A, Q)
