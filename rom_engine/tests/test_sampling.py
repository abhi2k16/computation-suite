"""
test_sampling.py -- validates rom_engine.sampling.optimal_lhs and
rom_engine.sampling.modal_force_samples.

Checks:
  1. optimal_lhs returns the right shape, stays within [0,1]^n_dims,
     and is a genuine Latin Hypercube (exactly one sample per stratum
     along each dimension).
  2. More candidate iterations (n_iter) never makes the maximin score
     WORSE -- checked as a guaranteed inequality (same rng seed, so the
     n_iter=1 result is exactly the n_iter>1 run's first candidate),
     not just "seems to usually help".
  3. Same rng seed -> identical design (reproducibility); an explicit
     rng is required (no hidden default).
  4. Unsupported criterion / degenerate n_samples/n_dims/n_iter are
     rejected loudly.
  5. modal_force_samples: output shape, the round-trip
     Q_max = f_hat * phi_max / lambda lands inside the requested
     [frac_min, frac_max] * reference_scale range for every sample and
     mode, and per-mode target_fracs ranges are honored independently.
"""
import numpy as np
import pytest
from scipy.spatial.distance import pdist

from rom_engine.sampling import optimal_lhs, modal_force_samples


def test_optimal_lhs_shape_and_bounds():
    rng = np.random.default_rng(0)
    X = optimal_lhs(20, 3, rng=rng)
    assert X.shape == (20, 3)
    assert np.all(X >= 0.0) and np.all(X <= 1.0)


def test_optimal_lhs_is_a_genuine_latin_hypercube():
    """Latin Hypercube property: along EACH dimension, the n_samples
    points fall into n_samples equal-width strata with exactly one
    point per stratum (no two points share a stratum, no stratum is
    empty)."""
    rng = np.random.default_rng(1)
    n = 15
    X = optimal_lhs(n, 4, rng=rng)
    for d in range(4):
        strata = np.floor(X[:, d] * n).astype(int)
        strata = np.clip(strata, 0, n - 1)
        assert sorted(strata) == list(range(n)), (
            f"dimension {d}: not a genuine LHS stratification, strata={sorted(strata)}")


def test_more_iterations_never_makes_maximin_score_worse():
    # Same seed for both calls: the n_iter=1 run's single candidate is
    # EXACTLY the first candidate the n_iter=25 run also draws (same
    # rng calls in the same order), so the n_iter=25 run's best score
    # is guaranteed >= the n_iter=1 run's score, not just usually true.
    rng1 = np.random.default_rng(42)
    X1 = optimal_lhs(10, 2, n_iter=1, rng=rng1)
    score1 = pdist(X1).min()

    rng25 = np.random.default_rng(42)
    X25 = optimal_lhs(10, 2, n_iter=25, rng=rng25)
    score25 = pdist(X25).min()

    print(f"maximin score: n_iter=1 -> {score1:.4f}, n_iter=25 -> {score25:.4f}")
    assert score25 >= score1 - 1e-12


def test_optimal_lhs_reproducible_with_same_seed():
    X1 = optimal_lhs(10, 3, rng=np.random.default_rng(7))
    X2 = optimal_lhs(10, 3, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(X1, X2)


def test_optimal_lhs_requires_explicit_rng():
    with pytest.raises(ValueError):
        optimal_lhs(10, 2)


def test_optimal_lhs_rejects_unsupported_criterion():
    with pytest.raises(ValueError):
        optimal_lhs(10, 2, criterion="correlation", rng=np.random.default_rng(0))


def test_optimal_lhs_rejects_degenerate_sizes():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        optimal_lhs(0, 2, rng=rng)
    with pytest.raises(ValueError):
        optimal_lhs(5, 0, rng=rng)
    with pytest.raises(ValueError):
        optimal_lhs(5, 2, n_iter=0, rng=rng)


def test_modal_force_samples_shape_and_target_range():
    rng = np.random.default_rng(3)
    basis_freqs_hz = np.array([120.0, 340.0, 610.0])
    mode_shape_peaks = np.array([1.0, 0.8, 1.2])
    reference_scale = 0.002   # e.g. a beam thickness [m]
    target_fracs = (0.8, 1.2)
    n_samples = 25

    f_hat = modal_force_samples(basis_freqs_hz, mode_shape_peaks, target_fracs,
                                 reference_scale, n_samples, rng=rng)
    assert f_hat.shape == (n_samples, 3)

    lam = (2 * np.pi * basis_freqs_hz) ** 2
    Q_max = f_hat * mode_shape_peaks / lam   # invert Eq. 46-47 back to the sampled displacement target
    frac = Q_max / reference_scale
    print(f"\nrecovered fraction range per mode: "
          f"{[f'{frac[:, m].min():.3f}-{frac[:, m].max():.3f}' for m in range(3)]}")
    assert np.all(frac >= 0.8 - 1e-9) and np.all(frac <= 1.2 + 1e-9)


def test_modal_force_samples_per_mode_ranges_are_independent():
    rng = np.random.default_rng(4)
    basis_freqs_hz = np.array([100.0, 200.0])
    mode_shape_peaks = np.array([1.0, 1.0])
    reference_scale = 1.0
    # mode 0: narrow range near 1.0; mode 1: wide range near 0.1
    target_fracs = np.array([[0.95, 1.05], [0.05, 0.20]])
    n_samples = 20

    f_hat = modal_force_samples(basis_freqs_hz, mode_shape_peaks, target_fracs,
                                 reference_scale, n_samples, rng=rng)
    lam = (2 * np.pi * basis_freqs_hz) ** 2
    Q_max = f_hat * mode_shape_peaks / lam
    assert Q_max[:, 0].min() >= 0.95 - 1e-9 and Q_max[:, 0].max() <= 1.05 + 1e-9
    assert Q_max[:, 1].min() >= 0.05 - 1e-9 and Q_max[:, 1].max() <= 0.20 + 1e-9


def test_modal_force_samples_requires_explicit_rng():
    with pytest.raises(ValueError):
        modal_force_samples(np.array([100.0]), np.array([1.0]), (0.8, 1.2), 1.0, 10)


def test_modal_force_samples_rejects_mismatched_lengths():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        modal_force_samples(np.array([100.0, 200.0]), np.array([1.0]), (0.8, 1.2), 1.0, 10, rng=rng)


def test_modal_force_samples_rejects_bad_frac_range():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        modal_force_samples(np.array([100.0]), np.array([1.0]), (1.2, 0.8), 1.0, 10, rng=rng)
