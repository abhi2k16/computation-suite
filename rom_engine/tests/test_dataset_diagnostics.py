"""
Tests for dataset_diagnostics.py (Wave 10 item 104). Everything here
runs unconditionally -- no torch dependency anywhere in this module.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from rom_engine.dataset_diagnostics import (
    parameter_coverage_report, pca_dimensionality, GPSurrogate, sobol_indices,
)


# =====================================================================
# parameter_coverage_report
# =====================================================================
class TestParameterCoverageReport:
    def test_full_coverage_has_no_gap(self):
        rng = np.random.default_rng(0)
        # a genuine space-filling design (uniform over the whole range,
        # many samples) should show no empty bins
        table = rng.uniform(0, 1, size=(500, 2))
        report = parameter_coverage_report(table, n_bins=10)
        assert len(report) == 2
        for entry in report:
            assert not entry["has_gap"]
            assert entry["counts"].sum() == 500

    def test_clustered_design_shows_a_gap(self):
        # all samples crammed into the lower half of the range -- the
        # upper-half bins must show up as empty (has_gap=True)
        rng = np.random.default_rng(1)
        table = rng.uniform(0, 0.3, size=(200, 1))
        # force the reported range to span [0, 1] by adding two
        # single endpoint samples, then check the INTERIOR still gaps
        table_full_range = np.vstack([table, [[0.0]], [[1.0]]])
        report = parameter_coverage_report(table_full_range, n_bins=10)
        assert report[0]["has_gap"]

    def test_custom_param_names(self):
        table = np.zeros((5, 2))
        report = parameter_coverage_report(table, param_names=["E", "nu"])
        assert [r["name"] for r in report] == ["E", "nu"]

    def test_mismatched_param_names_rejected(self):
        table = np.zeros((5, 2))
        with pytest.raises(ValueError):
            parameter_coverage_report(table, param_names=["only_one"])

    def test_degenerate_constant_column_is_not_an_error(self):
        table = np.column_stack([np.full(10, 3.0), np.linspace(0, 1, 10)])
        report = parameter_coverage_report(table, n_bins=5)
        assert report[0]["min"] == report[0]["max"] == 3.0
        assert not report[0]["has_gap"]


# =====================================================================
# pca_dimensionality
# =====================================================================
class TestPcaDimensionality:
    def test_perfectly_correlated_columns_need_only_one_component(self):
        rng = np.random.default_rng(2)
        base = rng.standard_normal(200)
        X = np.column_stack([base, 2.0 * base, -0.5 * base])   # rank-1 data
        result = pca_dimensionality(X, variance_threshold=0.99)
        assert result["n_components"] == 1
        assert result["cumulative_variance_ratio"][0] > 0.999

    def test_independent_gaussian_columns_need_all_components(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((2000, 4))   # ~isotropic -> needs ~all 4
        result = pca_dimensionality(X, variance_threshold=0.95)
        assert result["n_components"] >= 3

    def test_explained_variance_ratio_sums_to_one(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((100, 5))
        result = pca_dimensionality(X)
        assert np.isclose(result["explained_variance_ratio"].sum(), 1.0)

    def test_requires_at_least_two_samples(self):
        with pytest.raises(ValueError):
            pca_dimensionality(np.array([[1.0, 2.0]]))


# =====================================================================
# GPSurrogate
# =====================================================================
class TestGPSurrogate:
    def test_fits_smooth_1d_function_well(self):
        rng = np.random.default_rng(5)
        X = rng.uniform(-3, 3, size=(40, 1))
        y = np.sin(X[:, 0])
        gp = GPSurrogate(noise=1e-8).fit(X, y)

        X_test = np.linspace(-3, 3, 50).reshape(-1, 1)
        y_test = np.sin(X_test[:, 0])
        pred = gp.predict(X_test)
        r2 = 1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        assert r2 > 0.95

    def test_predict_returns_std_when_requested(self):
        rng = np.random.default_rng(6)
        X = rng.uniform(0, 1, size=(20, 2))
        y = X[:, 0] + X[:, 1]
        gp = GPSurrogate().fit(X, y)
        mean, std = gp.predict(X[:5], return_std=True)
        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert np.all(std >= 0.0)

    def test_std_is_near_zero_at_training_points(self):
        # a training point (with tiny noise) should have near-zero
        # posterior std -- the GP interpolates its own data
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, size=(15, 1))
        y = np.cos(3 * X[:, 0])
        gp = GPSurrogate(noise=1e-10).fit(X, y)
        _, std = gp.predict(X, return_std=True)
        assert np.max(std) < 1e-3

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            GPSurrogate().predict(np.zeros((1, 1)))

    def test_mismatched_rows_rejected(self):
        with pytest.raises(ValueError):
            GPSurrogate().fit(np.zeros((5, 2)), np.zeros(4))


# =====================================================================
# sobol_indices
# =====================================================================
class TestSobolIndices:
    def test_additive_linear_function_known_closed_form(self):
        # f(x1,x2) = x1 + 2*x2, x1,x2 ~ U(0,1) iid: Var(x1)=Var(x2)=1/12,
        # Var(f) = 1/12 + 4/12 = 5/12 (no interaction term) ->
        # S1 = ST = [0.2, 0.8] exactly in the infinite-sample limit.
        def model_fn(X):
            return X[:, 0] + 2.0 * X[:, 1]

        rng = np.random.default_rng(8)
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = sobol_indices(bounds, model_fn, n_samples=20000, rng=rng)

        assert np.allclose(result["S1"], [0.2, 0.8], atol=0.03)
        assert np.allclose(result["ST"], [0.2, 0.8], atol=0.03)
        # no interaction -- total and first order should closely agree
        assert np.allclose(result["S1"], result["ST"], atol=0.03)

    def test_interaction_term_makes_total_exceed_first_order(self):
        # f(x1,x2) = x1*x2 has a genuine interaction (no purely
        # additive part at all) -- ST_i must exceed S1_i for both
        # dimensions by a real margin.
        def model_fn(X):
            return X[:, 0] * X[:, 1]

        rng = np.random.default_rng(9)
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = sobol_indices(bounds, model_fn, n_samples=20000, rng=rng)
        assert np.all(result["ST"] > result["S1"] + 0.05)

    def test_via_gp_surrogate_recovers_similar_indices(self):
        # the realistic use case: fit a GP to a modest number of
        # training samples of the SAME additive function above, then
        # run Sobol' through the surrogate instead of the true
        # function -- should recover indices in the right ballpark
        # (looser tolerance than the exact-function test above, since
        # this adds real surrogate-fit error on top of Monte Carlo
        # noise).
        def true_fn(X):
            return X[:, 0] + 2.0 * X[:, 1]

        rng = np.random.default_rng(10)
        X_train = rng.uniform(0, 1, size=(60, 2))
        y_train = true_fn(X_train)
        gp = GPSurrogate(noise=1e-6).fit(X_train, y_train)

        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = sobol_indices(bounds, gp.predict, n_samples=5000, rng=np.random.default_rng(11))
        assert np.allclose(result["S1"], [0.2, 0.8], atol=0.1)

    def test_requires_explicit_rng(self):
        with pytest.raises(ValueError):
            sobol_indices(np.array([[0.0, 1.0]]), lambda X: X[:, 0], rng=None)
