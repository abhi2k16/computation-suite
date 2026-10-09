# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_membrane_expansion.py -- Wave 12 item 112 (docs/consolidated_
future_roadmap.md, ICE-ROM/GAP_ANALYSIS.md gap #1): validates
MembraneBasis's Eq. 15/16 least-squares fit and Eq. 16/11 expansion.

Three independent lines of evidence:
  1. A SYNTHETIC exact-recovery check (a known T_m_true, noise-free
     residual built directly from Eq. 16's own quadratic design matrix)
     -- decisive on the linear algebra itself, for n_modes > 1, with no
     dependence on any FE model or mode-selection choice.
  2. An independent, hand-transcribed reference of this project's own
     ALREADY-VALIDATED n_modes=1-only ad hoc prototype
     (ICE-ROM/validation/membrane_expansion.py, itself validated
     against a real flat-beam FOM reproduction), kept deliberately
     SEPARATE from MembraneBasis's own code -- agreement is a genuine
     cross-check of the generalization, not a tautology.
  3. AppliedLoadStrategy's new return_snapshots=True path, checked for
     shape/orientation and for reducing the bending-only reconstruction
     residual on the SAME (training) data the membrane basis was fit
     from -- the actual "does this do what it says" sanity check on
     real fea_engine data.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine.membrane_expansion import MembraneBasis
from rom_engine.nonlinear_rom import _monomial_indices, AppliedLoadStrategy

from fea_fixtures import clamped_clamped_nonlinear_beam_system


# =====================================================================
# 1. Synthetic exact recovery (n_modes > 1, no FE model involved)
# =====================================================================
def test_exact_recovery_of_a_known_membrane_basis():
    rng = np.random.default_rng(0)
    n_modes, n_free, n_samples = 3, 20, 400
    quad_idx = _monomial_indices(n_modes, 2)
    n_quad = len(quad_idx)

    V = rng.standard_normal((n_free, n_modes))
    Tm_true = rng.standard_normal((n_quad, n_free))
    q_nl = rng.uniform(-1.0, 1.0, size=(n_samples, n_modes))

    Q = np.array([q_nl[:, i] * q_nl[:, j] for (i, j) in quad_idx])   # (n_quad, n_samples)
    W = V @ q_nl.T + Tm_true.T @ Q                                    # Eq. 14, noise-free

    mb = MembraneBasis().fit(V, q_nl, W)
    assert mb.quad_idx == quad_idx
    assert np.allclose(mb.Tm, Tm_true, atol=1e-8, rtol=1e-6)

    # and expand() on a fresh, un-fitted p_hist reproduces the exact
    # synthetic construction (Eq. 16's quadratic expansion applied to
    # NEW points, not just the training set)
    p_hist = rng.uniform(-1.0, 1.0, size=(50, n_modes))
    Q_new = np.array([p_hist[:, i] * p_hist[:, j] for (i, j) in quad_idx])
    w_m_expected = (Tm_true.T @ Q_new).T
    w_m = mb.expand(p_hist)
    assert np.allclose(w_m, w_m_expected, atol=1e-7, rtol=1e-6)


# =====================================================================
# 2. Cross-check against an independent, hand-transcribed reference of
#    this project's own already-validated n_modes=1 ad hoc prototype
#    (ICE-ROM/validation/membrane_expansion.py -- kept separate here so
#    agreement is a real cross-check, not a tautology).
# =====================================================================
def _reference_fit_membrane_basis_n1(V_free, P, W):
    """P: (1, n_samples). Direct transcription of the validated ad hoc
    prototype's own fit_membrane_basis(), restricted to n_modes=1."""
    residual = W - V_free @ P
    Q = (P[0] ** 2)[None, :]
    Tm, *_ = np.linalg.lstsq(Q.T, residual.T, rcond=None)
    return Tm.ravel()


def _reference_expand_membrane_displacement_n1(Tm, p_hist):
    q_hist = p_hist ** 2
    return np.outer(Tm, q_hist)


def test_matches_independent_reference_of_validated_n1_prototype_on_real_beam_data():
    fix = clamped_clamped_nonlinear_beam_system(n_elem=10, n_modes=1)
    V = fix["V"]
    M_ff = fix["M_ff"]
    freq_hz = fix["freq_hz"]
    mode_shape_peaks = fix["mode_shape_peaks"]
    fom_solver = fix["fom_solver"]

    rng = np.random.default_rng(0)
    strat = AppliedLoadStrategy(target_fracs=(-3.0, 3.0), reference_scale=0.05,
                                 n_samples=24, rng=rng)
    q_l, q_nl, F_nl, W = strat.generate(V, M_ff, freq_hz, mode_shape_peaks, fom_solver,
                                         return_snapshots=True)
    assert q_nl.shape[1] == 1

    mb = MembraneBasis().fit(V, q_nl, W)

    P_ref = q_nl.T   # (1, n_samples), the ad hoc prototype's own orientation
    Tm_ref = _reference_fit_membrane_basis_n1(V, P_ref, W)

    assert mb.Tm.shape == (1, V.shape[0])
    assert np.allclose(mb.Tm[0], Tm_ref, atol=1e-10, rtol=1e-8)

    p_hist = np.linspace(-0.04, 0.04, 30).reshape(-1, 1)
    w_m = mb.expand(p_hist)                                    # (n_steps, n_free)
    w_m_ref = _reference_expand_membrane_displacement_n1(Tm_ref, p_hist[:, 0])   # (n_free, n_steps)
    assert np.allclose(w_m.T, w_m_ref, atol=1e-10, rtol=1e-8)


# =====================================================================
# 3. return_snapshots=True + the actual "does the expansion help"
#    reconstruction check on real fea_engine training data
# =====================================================================
def test_applied_load_strategy_return_snapshots_shape_and_default_unchanged():
    fix = clamped_clamped_nonlinear_beam_system(n_elem=6, n_modes=2)
    V, M_ff, freq_hz = fix["V"], fix["M_ff"], fix["freq_hz"]
    mode_shape_peaks, fom_solver = fix["mode_shape_peaks"], fix["fom_solver"]

    rng = np.random.default_rng(0)
    strat = AppliedLoadStrategy(target_fracs=(-2.0, 2.0), reference_scale=0.05,
                                 n_samples=10, rng=rng)

    # default: unchanged 3-tuple, no behavior change for existing callers
    out_default = strat.generate(V, M_ff, freq_hz, mode_shape_peaks, fom_solver)
    assert len(out_default) == 3

    rng2 = np.random.default_rng(0)   # same seed -> same samples
    strat2 = AppliedLoadStrategy(target_fracs=(-2.0, 2.0), reference_scale=0.05,
                                  n_samples=10, rng=rng2)
    q_l, q_nl, F_nl, W = strat2.generate(V, M_ff, freq_hz, mode_shape_peaks, fom_solver,
                                          return_snapshots=True)
    assert W.shape == (V.shape[0], 10)
    # the first 3 return values are bit-for-bit identical to the default path
    assert np.array_equal(q_l, out_default[0])
    assert np.array_equal(q_nl, out_default[1])
    assert np.array_equal(F_nl, out_default[2])


def test_membrane_expansion_reduces_training_reconstruction_residual_on_real_beam():
    """The actual payoff this item exists for: adding the membrane term
    should reconstruct the TRUE full-order training snapshots better
    than the bending-only prediction alone (a straight/flat beam's
    bending modes carry ~zero axial content, so the bending-only
    residual is dominated by exactly the membrane signal T_m exists to
    capture). Checked on the TRAINING set the basis was fit from --
    generalization to held-out data is a separate question this single
    check does not claim to answer."""
    fix = clamped_clamped_nonlinear_beam_system(n_elem=10, n_modes=1)
    V, M_ff, freq_hz = fix["V"], fix["M_ff"], fix["freq_hz"]
    mode_shape_peaks, fom_solver = fix["mode_shape_peaks"], fix["fom_solver"]

    rng = np.random.default_rng(2)
    strat = AppliedLoadStrategy(target_fracs=(-3.0, 3.0), reference_scale=0.05,
                                 n_samples=30, rng=rng)
    q_l, q_nl, F_nl, W = strat.generate(V, M_ff, freq_hz, mode_shape_peaks, fom_solver,
                                         return_snapshots=True)

    mb = MembraneBasis().fit(V, q_nl, W)
    w_m = mb.expand(q_nl)   # (n_samples, n_free)

    resid_bending_only = np.linalg.norm(W - V @ q_nl.T)
    resid_with_membrane = np.linalg.norm(W - (V @ q_nl.T + w_m.T))
    assert resid_with_membrane < resid_bending_only


# =====================================================================
# Input validation
# =====================================================================
def test_fit_rejects_mismatched_shapes():
    V = np.zeros((10, 2))
    q_nl_bad = np.zeros((5, 3))   # wrong n_modes
    W = np.zeros((10, 5))
    with pytest.raises(ValueError):
        MembraneBasis().fit(V, q_nl_bad, W)

    q_nl = np.zeros((5, 2))
    W_bad = np.zeros((11, 5))   # wrong n_free
    with pytest.raises(ValueError):
        MembraneBasis().fit(V, q_nl, W_bad)

    W_bad2 = np.zeros((10, 4))   # wrong n_samples
    with pytest.raises(ValueError):
        MembraneBasis().fit(V, q_nl, W_bad2)


def test_expand_before_fit_raises():
    with pytest.raises(RuntimeError):
        MembraneBasis().expand(np.zeros((5, 2)))


def test_expand_rejects_wrong_n_modes():
    V = np.eye(4)[:, :2]
    q_nl = np.random.default_rng(0).standard_normal((10, 2))
    W = V @ q_nl.T
    mb = MembraneBasis().fit(V, q_nl, W)
    with pytest.raises(ValueError):
        mb.expand(np.zeros((5, 3)))
