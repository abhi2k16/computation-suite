# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_balanced_truncation_torch.py -- Wave 9 addendum item 138 (docs/
consolidated_future_roadmap.md): validates balanced_truncation.py's
(and hankel_norm.py's) backend="numpy"/"torch" contract end-to-end --
the actual class-level entry points (BalancedTruncationROM.from_MCK,
SingularPerturbationROM.from_MCK, FrequencyWeightedBalancedTruncation
ROM.from_MCK, hankel_norm.OptimalHankelNormROM.from_MCK), not just the
standalone torch_linalg.py primitives (covered directly by
test_torch_linalg.py).

Same two-part split as this project's established convention:

  TestBackendArgumentValidation -- runs UNCONDITIONALLY, no torch
      needed: unknown backend= raises ValueError on every function that
      gained one; backend="numpy" (the default) is confirmed to produce
      IDENTICAL results to calling with no backend= argument at all
      (the pre-item-138 call signature).

  TestTorchBackendNumericalAgreement -- gated on rom_engine.
      torch_linalg._HAS_TORCH. The DECISIVE check throughout is
      agreement between backend="torch" and backend="numpy" on the SAME
      hand-built, stable, damped mass-spring-damper chain -- Hankel
      singular values, the reduced (A_r, B_r, Cout_r) system, and (for
      BalancedTruncationROM specifically) the actual frequency response
      and h_infinity_error_bound(), the same decisive standard this
      package holds every other backend addition to. Not executed
      end-to-end in the sandbox this file was authored in (no usable
      torch here); written to run for real, and SHOULD be run at least
      once on a torch-equipped machine.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine.torch_linalg import _HAS_TORCH
from rom_engine.balanced_truncation import (
    BalancedTruncationROM,
    SingularPerturbationROM,
    FrequencyWeightedBalancedTruncationROM,
    hankel_singular_values,
    lowpass_weight,
)
from rom_engine.hankel_norm import OptimalHankelNormROM


def _mck_chain(n_dof=6, k=200.0, m=1.0, alpha=0.5, beta=1e-4):
    """A small, stable, Rayleigh-damped mass-spring-damper chain
    (tridiagonal K, diagonal M, C = alpha*M + beta*K) -- a self-
    contained fixture with no fea_engine dependency, matching the
    small-synthetic-system style test_torch_iterative_solvers.py (the
    fea_engine sibling item's own torch test file) already uses rather
    than test_balanced_truncation.py's real-fea_engine-beam fixture
    (that file's own scale note about needing a WELL-CONDITIONED,
    modally-pre-truncated system applies here too -- this chain is
    built small and well-conditioned from the start, so no modal
    pre-truncation step is needed)."""
    K = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        K[i, i] = 2.0 * k
        if i > 0:
            K[i, i - 1] = -k
            K[i - 1, i] = -k
    K[-1, -1] = k   # free end
    M = m * np.eye(n_dof)
    C = alpha * M + beta * K
    B = np.zeros((n_dof, 1))
    B[-1, 0] = 1.0
    Cout = np.zeros((1, n_dof))
    Cout[0, -1] = 1.0
    return M, K, C, B, Cout


class TestBackendArgumentValidation:
    def test_hankel_singular_values_unknown_backend_raises(self):
        M, K, C, B, Cout = _mck_chain()
        A = -np.linalg.solve(M, K)   # not a real state-space A, just needs SOME array here
        with pytest.raises(ValueError, match="unknown backend"):
            hankel_singular_values(np.eye(4), np.ones((4, 1)), np.ones((1, 4)), backend="jax")

    def test_balanced_truncation_default_backend_matches_explicit_numpy(self):
        M, K, C, B, Cout = _mck_chain()
        rom_default = BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4)
        rom_explicit = BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4, backend="numpy")
        np.testing.assert_array_equal(rom_default.hsv, rom_explicit.hsv)
        np.testing.assert_array_equal(rom_default.A_r, rom_explicit.A_r)

    def test_singular_perturbation_default_backend_matches_explicit_numpy(self):
        M, K, C, B, Cout = _mck_chain()
        rom_default = SingularPerturbationROM.from_MCK(M, K, B, Cout, C=C, r=4)
        rom_explicit = SingularPerturbationROM.from_MCK(M, K, B, Cout, C=C, r=4, backend="numpy")
        np.testing.assert_array_equal(rom_default.A_r, rom_explicit.A_r)

    def test_frequency_weighted_default_backend_matches_explicit_numpy(self):
        M, K, C, B, Cout = _mck_chain()
        Wi = lowpass_weight(50.0)
        rom_default = FrequencyWeightedBalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4, Wi=Wi)
        rom_explicit = FrequencyWeightedBalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4, Wi=Wi, backend="numpy")
        np.testing.assert_array_equal(rom_default.hsv, rom_explicit.hsv)

    def test_optimal_hankel_norm_default_backend_matches_explicit_numpy(self):
        M, K, C, B, Cout = _mck_chain()
        rom_default = OptimalHankelNormROM.from_MCK(M, K, B, Cout, C=C, r=4)
        rom_explicit = OptimalHankelNormROM.from_MCK(M, K, B, Cout, C=C, r=4, backend="numpy")
        np.testing.assert_array_equal(rom_default.A_r, rom_explicit.A_r)
        assert rom_default.sigma_r1 == rom_explicit.sigma_r1


pytestmark_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not usable in this environment (not installed, or installed "
           "but failing to import -- see torch_linalg.py's own docstring); "
           "these tests validate the torch backend end-to-end through the "
           "balanced_truncation.py/hankel_norm.py class entry points and "
           "have nothing to run without it.")


@pytestmark_torch
class TestTorchBackendNumericalAgreement:
    def test_balanced_truncation_hsv_matches_numpy_backend(self):
        M, K, C, B, Cout = _mck_chain()
        rom_np = BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4)
        rom_torch = BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4, backend="torch", device="cpu")
        assert np.allclose(rom_np.hsv, rom_torch.hsv, atol=1e-6, rtol=1e-6)

    def test_balanced_truncation_reduced_system_matches_numpy_backend(self):
        M, K, C, B, Cout = _mck_chain()
        rom_np = BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4)
        rom_torch = BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4, backend="torch", device="cpu")
        assert np.allclose(sorted(np.linalg.eigvals(rom_np.A_r).real),
                            sorted(np.linalg.eigvals(rom_torch.A_r).real), atol=1e-4, rtol=1e-4)
        assert rom_torch.is_stable()

    def test_balanced_truncation_frequency_response_matches_numpy_backend(self):
        M, K, C, B, Cout = _mck_chain()
        rom_np = BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4)
        rom_torch = BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4, backend="torch", device="cpu")
        omega = np.linspace(0.1, 50.0, 25)
        H_np = rom_np.frequency_response(omega)
        H_torch = rom_torch.frequency_response(omega)
        assert np.allclose(H_np, H_torch, atol=1e-5, rtol=1e-4)

    def test_balanced_truncation_h_infinity_bound_matches_numpy_backend(self):
        M, K, C, B, Cout = _mck_chain()
        rom_np = BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4)
        rom_torch = BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4, backend="torch", device="cpu")
        assert np.isclose(rom_np.h_infinity_error_bound(), rom_torch.h_infinity_error_bound(),
                           atol=1e-6, rtol=1e-5)

    def test_singular_perturbation_matches_numpy_backend(self):
        M, K, C, B, Cout = _mck_chain()
        rom_np = SingularPerturbationROM.from_MCK(M, K, B, Cout, C=C, r=4)
        rom_torch = SingularPerturbationROM.from_MCK(M, K, B, Cout, C=C, r=4, backend="torch", device="cpu")
        assert np.allclose(rom_np.transfer_function(0.0), rom_torch.transfer_function(0.0),
                            atol=1e-5, rtol=1e-4)

    def test_frequency_weighted_matches_numpy_backend(self):
        M, K, C, B, Cout = _mck_chain()
        Wi = lowpass_weight(50.0)
        rom_np = FrequencyWeightedBalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4, Wi=Wi)
        rom_torch = FrequencyWeightedBalancedTruncationROM.from_MCK(
            M, K, B, Cout, C=C, r=4, Wi=Wi, backend="torch", device="cpu")
        assert np.allclose(rom_np.hsv, rom_torch.hsv, atol=1e-6, rtol=1e-6)

    def test_optimal_hankel_norm_matches_numpy_backend(self):
        M, K, C, B, Cout = _mck_chain()
        rom_np = OptimalHankelNormROM.from_MCK(M, K, B, Cout, C=C, r=4)
        rom_torch = OptimalHankelNormROM.from_MCK(M, K, B, Cout, C=C, r=4, backend="torch", device="cpu")
        assert np.isclose(rom_np.sigma_r1, rom_torch.sigma_r1, atol=1e-6, rtol=1e-5)
        assert np.isclose(rom_np.measured_hankel_norm_error, rom_torch.measured_hankel_norm_error,
                           atol=1e-6, rtol=1e-3)
        assert rom_torch.numerically_reliable == rom_np.numerically_reliable

    def test_unavailable_torch_fails_fast(self, monkeypatch):
        # Patch balanced_truncation's OWN `_require_torch` name, not
        # torch_linalg's -- balanced_truncation.py imports it BY
        # REFERENCE ("from .torch_linalg import ..., _require_torch,
        # ..."), so controllability_gramian()/observability_gramian()
        # (which is what BalancedTruncationROM.from_MCK(backend="torch")
        # actually calls) resolve the name through balanced_
        # truncation's own module globals, not torch_linalg's -- the
        # exact by-reference-import lesson fea_engine's Wave 9 item 135
        # test (test_torch_mesh_transform.py) learned the hard way on
        # real hardware; applied proactively here instead.
        import rom_engine.balanced_truncation as bt_mod

        def _fake_require_torch():
            raise ImportError("simulated: torch unavailable")

        monkeypatch.setattr(bt_mod, "_require_torch", _fake_require_torch)
        M, K, C, B, Cout = _mck_chain()
        with pytest.raises(ImportError):
            BalancedTruncationROM.from_MCK(M, K, B, Cout, C=C, r=4, backend="torch")
