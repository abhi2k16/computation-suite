# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_wave13_validation.py -- Wave 13 item 120 (docs/consolidated_
future_roadmap.md): validates the cross-configuration harness that
ties items 117/118/119 together on ONE real fea_engine fixture pair
(`clamped_clamped_nonlinear_beam_system`, n_elem=10 for training /
n_elem=20 for cross-resolution validation), rather than re-proving any
one item's own already-tested claim in isolation.

Empirical numbers from this file's own development (recorded here so
a future reader can tell "still working as designed" apart from
"regressed" without re-deriving thresholds from scratch):

  - Two independent integrators (Newton-Newmark, item 114; RK4, item
    119) of the SAME fitted reduced dynamics, at dt=2e-5/n_steps=2500:
    max relative trajectory difference ~0.37%, well under the 2%
    bound asserted below. At the module's own former default
    (dt=2e-4/n_steps=250) they only agreed to ~42% -- a resolution
    issue in BOTH integrators (confirmed by dt-halving convergence),
    not a bug, and the reason the finer default was chosen.
  - item 117's residual, evaluated on both trajectories: ~0.02%
    (Newmark) / ~0.007% (RK4) of the applied-force scale.
  - item 118's cross-mesh-resolution reconstruction: ~0.03% same-
    resolution vs ~0.5%-2.2% cross-resolution (n_elem=10 -> n_elem=20),
    matching item 118's own test file's numbers almost exactly (same
    underlying fixture, same AppliedLoadStrategy convention).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine.wave13_validation import (
    cross_check_reduced_dynamics, cross_check_basis_generalization,
    run_cross_configuration_validation, _HAS_TORCH,
)
from fea_fixtures import clamped_clamped_nonlinear_beam_system


@pytest.fixture(scope="module")
def fix10():
    return clamped_clamped_nonlinear_beam_system(n_elem=10, n_modes=2,
                                                   damping_alpha=0.5, damping_beta=1e-5)


@pytest.fixture(scope="module")
def fix20():
    return clamped_clamped_nonlinear_beam_system(n_elem=20, n_modes=2,
                                                   damping_alpha=0.5, damping_beta=1e-5)


# =====================================================================
# 1. Reduced-dynamics cross-check: Newton-Newmark vs item 119's RK4,
#    both checked against item 117's own residual function.
# =====================================================================
def test_two_independent_integrators_of_the_same_reduced_model_agree(fix10):
    rep = cross_check_reduced_dynamics(fix10, n_train=40, seed=0)
    assert rep["max_rel_diff_newmark_vs_rk4"] < 0.02
    # both integrators' own trajectories should satisfy the continuous
    # reduced EOM to a small fraction of the applied-force scale
    assert rep["eom_residual_rms_newmark"] < 0.01 * rep["force_scale"]
    assert rep["eom_residual_rms_rk4"] < 0.01 * rep["force_scale"]


def test_coarser_resolution_shows_the_two_integrators_diverge_then_reconverge(fix10):
    """Decisive control: confirms the agreement above isn't a fluke of
    one particular dt -- deliberately using a too-coarse step first
    shows LARGE disagreement, then halving dt repeatedly shows the
    disagreement shrink monotonically (both integrators converging to
    the same continuous solution, the standard signature of a
    resolution effect rather than a bug in either one)."""
    Lambda_amp = 0.05 * np.array([1115939.46, 8483305.29]) * 0.02  # matches fixture's own Lambda scale
    diffs = []
    for dt, n_steps in [(2e-4, 63), (1e-4, 125), (5e-5, 250)]:
        rep = cross_check_reduced_dynamics(fix10, n_train=40, dt=dt, n_steps=n_steps,
                                            F_ext_amplitude=Lambda_amp, seed=0)
        diffs.append(rep["max_rel_diff_newmark_vs_rk4"])
    assert diffs[0] > diffs[1] > diffs[2]


# =====================================================================
# 2. Basis cross-resolution generalization (item 118), re-run as part
#    of the combined harness.
# =====================================================================
def test_basis_cross_resolution_error_stays_small_and_exceeds_same_resolution_error(fix10, fix20):
    rep = cross_check_basis_generalization(fix10, fix20)
    assert rep["err_same_resolution"] < 0.01
    assert rep["err_cross_resolution_max"] < 0.05
    # cross-resolution reconstruction is a genuinely harder task than
    # same-resolution round trip -- its own error should be larger
    assert rep["err_cross_resolution_max"] > rep["err_same_resolution"]


# =====================================================================
# 3. Combined report
# =====================================================================
def test_run_cross_configuration_validation_returns_both_reports(fix10, fix20):
    report = run_cross_configuration_validation(fix10, fix20, n_train=40, seed=0)
    assert "reduced_dynamics" in report and "basis_generalization" in report
    assert report["reduced_dynamics"]["max_rel_diff_newmark_vs_rk4"] < 0.02
    assert report["basis_generalization"]["err_cross_resolution_max"] < 0.05


# =====================================================================
# torch-gated: ExcitationResponseOperator + ParameterizedLatentODE
# cross-checked against the same Newton-Newmark reference, on the same
# real fixture.
# =====================================================================
@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
def test_operator_generalizes_to_unseen_frequency_on_real_fixture(fix10):
    from rom_engine.wave13_validation import cross_check_operators_torch
    rep = cross_check_operators_torch(fix10, n_train_freqs=4, n_time=64, seed=0)
    assert rep["r2_unseen_frequency"] > 0.0
