# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
dataset_diagnostics.py -- Wave 10 item 104 (fea_engine/docs/
consolidated_future_roadmap.md, source: Saverio, Bucci, Farro, Content
& Sipp, "An end-to-end PyTorch interface for differentiable PDE
solvers: A RANS model-correction study," Data-Centric Engineering
7:e37, 2026, Appendix B). A small, standalone, low-risk utility: given
a design-of-experiments parameter table (e.g. `rom_engine.sampling.
optimal_lhs()`'s own output, rescaled to physical units) and the
resulting converged-case outputs, report

  - per-parameter coverage (histograms + a "has this range got a gap"
    flag) -- `parameter_coverage_report()`
  - a PCA-based dimensionality estimate of the input or output
    variability -- `pca_dimensionality()`
  - Sobol' sensitivity indices, via a cheap Gaussian-Process surrogate
    on the (already-reduced/converged) outputs, quantifying which
    input parameters actually drive output variance -- `GPSurrogate`
    + `sobol_indices()`

Lives in rom_engine (not fea_engine) because a ROM/surrogate TRAINING
dataset -- exactly what `nonlinear_rom.py`'s `TrainingStrategy`
subclasses and `sampling.optimal_lhs()` already build -- is this
item's natural, concrete use case (the paper's own framing: "useful the
moment any future parameter-sweep training dataset is generated"), not
an FEA-specific concern; nothing here imports fea_engine or is coupled
to any particular solver, matching this package's own stated design
principle (rom_engine never takes fea_engine as a library dependency,
only as a test/example dependency).

Independent of torch entirely -- no optional-dependency gating needed
here; every function is plain NumPy/SciPy, fully testable in any
environment.

Not blocking anything else in this wave -- a "nice to have" the next
time a training dataset is built, per the roadmap item's own framing.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.spatial.distance import cdist, pdist


# =====================================================================
# Parameter coverage
# =====================================================================
def parameter_coverage_report(param_table, n_bins=10, param_names=None):
    """Per-parameter coverage histogram + gap flag for a design-of-
    experiments parameter table.

    Parameters
    ----------
    param_table : array_like, shape (n_samples, n_params)
    n_bins : int, default 10
        Number of equal-width histogram bins per parameter.
    param_names : list of str, optional
        Defaults to "param_0", "param_1", ...

    Returns
    -------
    list of dict, one per parameter, each with keys "name", "min",
    "max", "counts" (n_bins,), "bin_edges" (n_bins+1,), and "has_gap"
    (True iff at least one bin within [min, max] has zero samples --
    a cheap, direct signal that the design under-samples some region
    of that parameter's own range, worth flagging before spending
    expensive full-order solves elsewhere in the same range)."""
    param_table = np.atleast_2d(np.asarray(param_table, dtype=float))
    n_samples, n_params = param_table.shape
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]
    elif len(param_names) != n_params:
        raise ValueError(
            f"param_names has {len(param_names)} entries, expected {n_params}")

    report = []
    for j in range(n_params):
        col = param_table[:, j]
        lo, hi = float(col.min()), float(col.max())
        if hi > lo:
            counts, edges = np.histogram(col, bins=n_bins, range=(lo, hi))
            has_gap = bool(np.any(counts == 0))
        else:
            # every sample has the identical value -- degenerate but
            # not an error (a fixed/non-swept parameter is legitimate).
            counts = np.array([n_samples])
            edges = np.array([lo, hi])
            has_gap = False
        report.append({
            "name": param_names[j], "min": lo, "max": hi,
            "counts": counts, "bin_edges": edges, "has_gap": has_gap,
        })
    return report


# =====================================================================
# PCA-based dimensionality estimate
# =====================================================================
def pca_dimensionality(X, variance_threshold=0.95):
    """SVD-based PCA dimensionality estimate -- no scikit-learn
    dependency (this package already avoids adding one for a single
    small utility; SVD of the centered data matrix is the exact same
    computation scikit-learn's own PCA performs internally).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Either the input (geometry/parameter) table or the output
        (converged state / reduced-mode) table -- this function makes
        no assumption about which.
    variance_threshold : float in (0, 1], default 0.95

    Returns
    -------
    dict with keys "n_components" (smallest number of principal
    components whose CUMULATIVE explained-variance ratio reaches
    variance_threshold), "explained_variance_ratio" (n_features,),
    "cumulative_variance_ratio" (n_features,)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("pca_dimensionality needs at least 2 samples")
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    explained_variance = s ** 2 / (n_samples - 1)
    total = explained_variance.sum()
    ratio = explained_variance / total if total > 0 else np.zeros_like(explained_variance)
    cumulative = np.cumsum(ratio)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_components = min(n_components, len(ratio))
    return {
        "n_components": n_components,
        "explained_variance_ratio": ratio,
        "cumulative_variance_ratio": cumulative,
    }


# =====================================================================
# Cheap Gaussian-Process surrogate (for Sobol sampling only)
# =====================================================================
class GPSurrogate:
    """A minimal, hand-rolled Gaussian-Process regressor -- isotropic
    squared-exponential kernel, median-heuristic length scale by
    default, small noise/nugget regularization, exact (Cholesky)
    training. Deliberately NOT a full-featured GP library (no kernel
    selection, no marginal-likelihood hyperparameter optimization):
    the ONLY reason this class exists is to give sobol_indices() below
    a CHEAP, repeatedly-callable stand-in for an expensive full-order
    solver (Sobol' needs O(n_samples * (n_dims+2)) evaluations, far
    more than any real solver sweep affords) -- matching this
    project's general preference for owning small, well-understood
    numerics over pulling in a heavy new dependency for one utility.

    Parameters
    ----------
    length_scale : float, optional
        Kernel length scale. Defaults to the median pairwise training-
        point distance (the standard "median heuristic").
    noise : float, default 1e-6
        Diagonal regularization ("nugget") added to the kernel matrix
        before Cholesky factorization -- keeps the solve well-posed
        even for near-duplicate training points.
    signal_var : float, optional
        Kernel output-scale (variance). Defaults to the training
        targets' own sample variance.
    """

    def __init__(self, length_scale=None, noise=1e-6, signal_var=None):
        self.length_scale_param = length_scale
        self.noise = noise
        self.signal_var_param = signal_var
        self.X_train = None
        self.y_mean = None
        self.length_scale = None
        self.signal_var_ = None
        self.L_ = None
        self.alpha_ = None

    def _kernel(self, X1, X2):
        d2 = cdist(X1, X2, "sqeuclidean")
        return self.signal_var_ * np.exp(-0.5 * d2 / self.length_scale ** 2)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"GPSurrogate.fit: X has {X.shape[0]} rows, y has {y.shape[0]}")
        self.X_train = X
        self.y_mean = float(y.mean())
        y_centered = y - self.y_mean

        if self.length_scale_param is None:
            d = pdist(X)
            self.length_scale = float(np.median(d)) if len(d) and np.median(d) > 0 else 1.0
        else:
            self.length_scale = self.length_scale_param

        if self.signal_var_param is None:
            var = float(y_centered.var())
            self.signal_var_ = var if var > 0 else 1.0
        else:
            self.signal_var_ = self.signal_var_param

        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.L_ = np.linalg.cholesky(K)
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, y_centered))
        return self

    def predict(self, X, return_std=False):
        if self.X_train is None:
            raise RuntimeError("GPSurrogate.predict() called before fit()")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Ks = self._kernel(X, self.X_train)
        mean = Ks @ self.alpha_ + self.y_mean
        if not return_std:
            return mean
        v = np.linalg.solve(self.L_, Ks.T)
        var = self.signal_var_ - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 0.0)
        return mean, np.sqrt(var)


# =====================================================================
# Sobol' sensitivity indices
# =====================================================================
def sobol_indices(bounds, model_fn, n_samples=512, rng=None):
    """First-order (Saltelli 2010) and total-order (Jansen 1999) Sobol'
    sensitivity indices for `model_fn`, a cheap callable (n, d) ->
    (n,). Intended to be called with a `GPSurrogate.predict` bound
    above (fit to a real, expensive parameter sweep's own outputs) as
    `model_fn` -- Sobol' needs O(n_samples*(d+2)) evaluations, far more
    than a real full-order-model sweep affords directly, but cheap
    enough on a surrogate trained on that same sweep's own data.

    Estimator (Saltelli 2010's improved formulas, the standard modern
    choice over the original 1993 estimator for both stability and
    reuse of the same sample set for both indices): draw two
    independent design matrices A, B ~ Uniform(bounds), (n_samples, d)
    each; for each dimension i, build AB_i = A with column i replaced
    by B's column i. Then

        S_i  = mean(f_B * (f_ABi - f_A)) / V          (first-order)
        ST_i = mean((f_A - f_ABi)**2) / (2*V)          (total-order)

    where V = Var([f_A; f_B]). S_i measures dimension i's OWN
    contribution to output variance; ST_i additionally includes every
    interaction effect involving dimension i -- ST_i > S_i signals a
    real interaction with some other input, ST_i ~= S_i signals
    dimension i acts nearly additively.

    Parameters
    ----------
    bounds : array_like, shape (n_dims, 2)
        (low, high) range for each input dimension.
    model_fn : callable, (n, n_dims) ndarray -> (n,) ndarray
    n_samples : int, default 512
        Base sample size N -- total model_fn evaluations = N*(d+2).
    rng : numpy.random.Generator
        Required, explicit (matching this package's own sampling.py
        convention).

    Returns
    -------
    dict with keys "S1" (n_dims,), "ST" (n_dims,), "variance" (float,
    the V used to normalize both -- reported so a caller can sanity-
    check it against the raw output spread)."""
    if rng is None:
        raise ValueError("sobol_indices requires an explicit rng (np.random.default_rng(seed))")
    bounds = np.atleast_2d(np.asarray(bounds, dtype=float))
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]

    A = lo + rng.random((n_samples, d)) * (hi - lo)
    B = lo + rng.random((n_samples, d)) * (hi - lo)
    f_A = np.asarray(model_fn(A), dtype=float).reshape(-1)
    f_B = np.asarray(model_fn(B), dtype=float).reshape(-1)
    V = float(np.concatenate([f_A, f_B]).var())

    S1 = np.zeros(d)
    ST = np.zeros(d)
    for i in range(d):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        f_ABi = np.asarray(model_fn(AB_i), dtype=float).reshape(-1)
        if V > 0:
            S1[i] = np.mean(f_B * (f_ABi - f_A)) / V
            ST[i] = np.mean((f_A - f_ABi) ** 2) / (2 * V)
        else:
            S1[i] = 0.0
            ST[i] = 0.0

    return {"S1": S1, "ST": ST, "variance": V}
