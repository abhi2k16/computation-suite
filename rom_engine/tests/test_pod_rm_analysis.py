# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_pod_rm_analysis.py -- validates rom_engine.intrusive_nonlinear_rom.
pod_rm_analysis() (Wave 17 item 145) against the roadmap's own two
named validation criteria:

  1. Diagonal M_r/K_r gives E_hat = I EXACTLY (machine precision) and
     RM frequencies exactly equal to the uncoupled 1-DOF frequencies
     (a diagonal system already IS its own uncoupled representation).
  2. A KNOWN 3x3 orthogonal rotation of a diagonal system recovers that
     rotation in E_hat (up to this module's own sign convention) and
     gives RM frequencies exactly matching the original (unrotated)
     diagonal system's frequencies (frequencies are rotation-invariant
     -- a strong, decisive check that the generalized eigenproblem
     solve itself is correct, not merely plausible-looking).

Plus a small extra check that the symmetry/positive-definiteness report
(`spd_report`) actually flags a genuinely non-SPD damping-like matrix
as such, so that code path is exercised too.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine import pod_rm_analysis


def _align_signs_to_reference(E, R):
    """Flip each column of E independently so its dominant-overlap sign
    with the matching column of R agrees -- eigenvectors (and this
    module's own sign convention) are only defined up to an overall
    per-column sign, so a direct comparison needs this alignment step
    first, exactly as the roadmap's own item 145 row anticipates
    ("up to your own documented sign convention")."""
    E = E.copy()
    for m in range(E.shape[1]):
        if np.dot(E[:, m], R[:, m]) < 0:
            E[:, m] = -E[:, m]
    return E


class TestDiagonalSystemGivesIdentity:
    @pytest.fixture(scope="class")
    def setup(self):
        rng = np.random.default_rng(0)
        n = 4
        # Chosen so sqrt(K_diag/M_diag) is already ASCENDING in index
        # order -- eigh's own ascending-eigenvalue order then coincides
        # with the original coordinate order too, so BOTH E_hat_sorted
        # and E_hat_pod_order (and both frequency orderings) reduce to
        # the identity/no-op case, letting this test check the sorted
        # ordering as well, not only the POD-mode-order one.
        M_diag = np.array([2.0, 1.6, 2.1, 0.9])
        K_diag = np.array([50.0, 90.0, 189.0, 260.0])
        M_r = np.diag(M_diag)
        K_r = np.diag(K_diag)
        # D_r deliberately non-diagonal and non-symmetric -- the
        # near-identity/frequency checks below don't depend on D_r at
        # all, only the spd_report does (checked separately).
        D_r = 0.05 * rng.standard_normal((n, n))
        result = pod_rm_analysis(M_r, D_r, K_r)
        return {"result": result, "M_diag": M_diag, "K_diag": K_diag, "n": n}

    def test_E_hat_is_identity_to_machine_precision(self, setup):
        result = setup["result"]
        err_sorted = np.max(np.abs(result.E_hat_sorted - np.eye(setup["n"])))
        err_pod = np.max(np.abs(result.E_hat_pod_order - np.eye(setup["n"])))
        print(f"diagonal system: max|E_hat_sorted - I| = {err_sorted:.3e}, "
              f"max|E_hat_pod_order - I| = {err_pod:.3e}")
        assert err_sorted < 1e-12
        assert err_pod < 1e-12

    def test_diag_E_hat_is_ones(self, setup):
        assert np.allclose(setup["result"].diag_E_hat, 1.0, atol=1e-12)

    def test_pod_order_index_is_identity_permutation(self, setup):
        assert np.array_equal(setup["result"].pod_order_index, np.arange(setup["n"]))

    def test_rm_frequencies_equal_uncoupled_frequencies_exactly(self, setup):
        result = setup["result"]
        expected = np.sqrt(setup["K_diag"] / setup["M_diag"])
        expected_sorted = np.sort(expected)
        err_uncoupled = np.max(np.abs(result.uncoupled_freq - expected))
        err_sorted = np.max(np.abs(result.freq_sorted - expected_sorted))
        err_pod = np.max(np.abs(result.freq_pod_order - expected))
        print(f"diagonal system: max|uncoupled_freq - expected| = {err_uncoupled:.3e}, "
              f"max|freq_sorted - expected_sorted| = {err_sorted:.3e}, "
              f"max|freq_pod_order - expected| = {err_pod:.3e}")
        assert err_uncoupled < 1e-10
        assert err_sorted < 1e-8
        assert err_pod < 1e-8
        # decisive: RM (coupled-solve) frequencies equal the uncoupled
        # ones for a genuinely diagonal system, not just close.
        assert np.max(np.abs(result.freq_pod_order - result.uncoupled_freq)) < 1e-8


class TestRotatedDiagonalSystemRecoversRotation:
    @pytest.fixture(scope="class")
    def setup(self):
        # Chosen so sqrt(K_diag/M_diag) is already ASCENDING in index
        # order (frequencies 4.47, 15.49, 100.0 rad/s) -- see the
        # identity-case fixture's own comment for why this matters:
        # it makes eigh's ascending-eigenvalue order on the ROTATED
        # system coincide, column-for-column, with the ORIGINAL
        # coordinate order, so E_hat_sorted's raw column m is directly
        # comparable to R^T's column m with no extra re-permutation
        # needed beyond the documented sign convention.
        M_diag = np.array([2.0, 1.25, 0.5])
        K_diag = np.array([40.0, 300.0, 5000.0])
        M0 = np.diag(M_diag)
        K0 = np.diag(K_diag)

        # A known orthogonal rotation (Rodrigues formula about a fixed
        # axis, angle pi/5 -- genuinely couples all three coordinates,
        # not a permutation or a rotation about a coordinate axis).
        axis = np.array([1.0, 2.0, -1.0])
        axis = axis / np.linalg.norm(axis)
        theta = np.pi / 5
        K_mat = np.array([[0, -axis[2], axis[1]],
                           [axis[2], 0, -axis[0]],
                           [-axis[1], axis[0], 0]])
        R = (np.eye(3) + np.sin(theta) * K_mat
             + (1 - np.cos(theta)) * (K_mat @ K_mat))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)  # sanity: genuinely orthogonal

        M_r = R.T @ M0 @ R
        K_r = R.T @ K0 @ R
        D_r = np.zeros((3, 3))

        result = pod_rm_analysis(M_r, D_r, K_r)

        # Original (unrotated) diagonal system's own frequencies, sorted
        # -- rotation-invariant ground truth.
        expected_freq_sorted = np.sort(np.sqrt(K_diag / M_diag))

        return {"result": result, "R": R, "expected_freq_sorted": expected_freq_sorted}

    def test_rm_frequencies_match_original_diagonal_system_exactly(self, setup):
        result, expected = setup["result"], setup["expected_freq_sorted"]
        err = np.max(np.abs(result.freq_sorted - expected))
        print(f"rotated system: max|freq_sorted - expected (unrotated) freq| = {err:.3e}")
        assert err < 1e-9

    def test_E_hat_recovers_known_rotation(self, setup):
        result, R = setup["result"], setup["R"]
        # E_hat_sorted's columns are (up to sign) R^T's columns, since
        # for M_r=R^T M0 R / K_r=R^T K0 R, eigenvectors are R^T @ (the
        # diagonal system's own Euclidean-unit eigenvectors, i.e. the
        # standard basis) = R^T's own columns = R's own rows.
        E_signed = _align_signs_to_reference(result.E_hat_sorted, R.T)
        err = np.max(np.abs(E_signed - R.T))
        print(f"rotated system: max|sign-aligned E_hat_sorted - R^T| = {err:.3e}")
        assert err < 1e-9


class TestSpdReportFlagsNonSpdMatrix:
    def test_spd_report_symmetric_pd_for_diagonal_M_K_and_flags_asymmetric_D(self):
        n = 3
        M_r = np.diag([1.0, 2.0, 3.0])
        K_r = np.diag([10.0, 20.0, 30.0])
        D_r = np.array([[1.0, 5.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])  # deliberately non-symmetric
        result = pod_rm_analysis(M_r, D_r, K_r)

        assert result.spd_report["M_r"].symmetric
        assert result.spd_report["M_r"].positive_definite
        assert result.spd_report["K_r"].symmetric
        assert result.spd_report["K_r"].positive_definite
        assert not result.spd_report["D_r"].symmetric
        print(f"D_r asymmetry (expected large, non-symmetric by construction): "
              f"{result.spd_report['D_r'].asymmetry:.3e}")
