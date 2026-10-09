# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
Tests for ensemble_uq.py (Wave 10 item 105). EnsembleUQ itself is
tested UNCONDITIONALLY (no torch needed) via a hand-built, seed-
dependent toy model (`_RandomOffsetToyModel` below) that validates the
actual aggregation machinery (does fitting N seeds give N distinct
members; does predict_mean_std's mean/std match a hand-computed
reference) without needing a real stochastically-trained model.
neural_surrogate_ensemble() -- the real NeuralSurrogate-based use case
-- is torch-gated, mirroring every other NeuralSurrogate-touching test
in this package (test_nonlinear_rom.py::TestNeuralSurrogate).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine.ensemble_uq import EnsembleUQ
from rom_engine.nonlinear_rom import _HAS_TORCH

if _HAS_TORCH:
    from rom_engine.ensemble_uq import neural_surrogate_ensemble


class _RandomOffsetToyModel:
    """A trivial seed-dependent model used ONLY by these tests: fits a
    constant (the training targets' own mean) plus a small SEED-
    DEPENDENT random offset -- exercises EnsembleUQ's own seed-to-
    member wiring and mean/std aggregation without needing a real
    stochastically-trained model like NeuralSurrogate (torch-gated)."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.value = None
        self.offset = None

    def fit(self, X, y):
        self.value = np.asarray(y, dtype=float).mean(axis=0)
        self.offset = self.rng.normal(scale=0.1, size=self.value.shape)
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.tile(self.value + self.offset, (X.shape[0], 1))


class TestEnsembleUQ:
    def test_different_seeds_give_different_members(self):
        X = np.zeros((10, 2))
        y = np.tile([1.0, 2.0], (10, 1))
        ens = EnsembleUQ(_RandomOffsetToyModel, n_members=4).fit(X, y)
        offsets = [m.offset for m in ens.members]
        # every pair of members should differ (different seeds -> different
        # random offsets) -- a real check that fit() didn't accidentally
        # reuse one instance across every "seed"
        for i in range(len(offsets)):
            for j in range(i + 1, len(offsets)):
                assert not np.allclose(offsets[i], offsets[j])

    def test_predict_mean_std_matches_hand_computed_reference(self):
        X = np.zeros((5, 2))
        y = np.tile([1.0, 2.0], (5, 1))
        ens = EnsembleUQ(_RandomOffsetToyModel, n_members=6, seeds=list(range(6))).fit(X, y)

        X_query = np.zeros((3, 2))
        mean, std = ens.predict_mean_std(X_query)
        assert mean.shape == (3, 2)
        assert std.shape == (3, 2)

        # hand-computed reference: every member predicts a CONSTANT
        # (value + offset) regardless of X, so re-deriving the same
        # per-member offsets independently and averaging them must
        # match ens's own aggregation exactly.
        expected_per_member = np.array([
            [1.0, 2.0] + np.random.default_rng(seed).normal(scale=0.1, size=2)
            for seed in range(6)
        ])
        assert np.allclose(mean[0], expected_per_member.mean(axis=0))
        assert np.allclose(std[0], expected_per_member.std(axis=0))
        # every query row should be identical (the toy model ignores X)
        assert np.allclose(mean[0], mean[1])
        assert np.allclose(mean[0], mean[2])

    def test_predict_all_returns_stacked_per_member_predictions(self):
        X = np.zeros((4, 1))
        y = np.ones((4, 1)) * 3.0
        ens = EnsembleUQ(_RandomOffsetToyModel, n_members=3).fit(X, y)
        preds = ens.predict_all(np.zeros((2, 1)))
        assert preds.shape == (3, 2, 1)

    def test_mismatched_seeds_length_rejected(self):
        with pytest.raises(ValueError):
            EnsembleUQ(_RandomOffsetToyModel, n_members=3, seeds=[0, 1])

    def test_predict_before_fit_raises(self):
        ens = EnsembleUQ(_RandomOffsetToyModel, n_members=2)
        with pytest.raises(RuntimeError):
            ens.predict_all(np.zeros((1, 1)))

    def test_zero_variance_model_gives_zero_std(self):
        # a DETERMINISTIC model family (ignores its seed entirely) must
        # give exactly zero ensemble std -- a decisive sanity check
        # that predict_mean_std() isn't manufacturing spurious spread.
        class _DeterministicModel:
            def __init__(self, seed):
                pass

            def fit(self, X, y):
                self.value = np.asarray(y, dtype=float).mean(axis=0)
                return self

            def predict(self, X):
                X = np.atleast_2d(X)
                return np.tile(self.value, (X.shape[0], 1))

        X = np.zeros((5, 1))
        y = np.ones((5, 1)) * 7.0
        ens = EnsembleUQ(_DeterministicModel, n_members=5).fit(X, y)
        _, std = ens.predict_mean_std(np.zeros((2, 1)))
        assert np.allclose(std, 0.0)


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
class TestNeuralSurrogateEnsemble:
    def test_ensemble_of_neural_surrogates_has_nonzero_spread(self):
        rng = np.random.default_rng(0)
        q = rng.uniform(-1, 1, size=(60, 2))
        F = 3.0 * q ** 3

        ens = neural_surrogate_ensemble(
            n_modes=2, q_samples=q, F_samples=F, n_members=3,
            hidden_sizes=(16, 16), n_epochs=150, lr=1e-2)
        mean, std = ens.predict_mean_std(q[:5])
        assert mean.shape == (5, 2)
        assert std.shape == (5, 2)
        # different random NN initializations trained for a short,
        # deliberately under-converged run should NOT all land on
        # exactly the same function -- real, nonzero ensemble spread.
        assert np.max(std) > 0.0
