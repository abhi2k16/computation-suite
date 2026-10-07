"""
ensemble_uq.py -- Wave 10 item 105 (fea_engine/docs/consolidated_
future_roadmap.md, source Saverio et al. 2026 Section 4.4.2/Fig 16):
"once ANY of items 99/100/103 introduces a trainable NN or RBF
correction, train it N times from different random seeds/mini-batch
orderings and report the ensemble mean/standard-deviation of its
predictions -- a cheap, well-understood epistemic-uncertainty proxy."
The paper shows this ensemble spread correlates spatially with regions
of higher true prediction error; this module provides the aggregation
machinery a caller checks that correlation with, not the correlation
check itself (which is inherently data/problem-specific).

A thin wrapper around whatever training loop item 99/100/103 already
uses, not a new algorithm -- exactly the roadmap item's own framing.
Works with ANY model exposing `fit(X, y) -> self` / `predict(X) ->
array` (this package's own established `ReducedForceModel` duck-typed
protocol, nonlinear_rom.py) via a caller-supplied `model_factory(seed)
-> fresh model instance` -- re-instantiating per seed rather than
re-fitting one shared instance, since a real ensemble member (e.g. an
NN with random weight initialization) generally needs a FRESH object
per seed, not just a re-run of `.fit()` on the same one.

Independent of torch at the module level -- `EnsembleUQ` itself never
imports torch; only `neural_surrogate_ensemble()`'s convenience
constructor touches `NeuralSurrogate` (which is itself torch-optional
at IMPORT time, per nonlinear_rom.py's own `_HAS_TORCH` pattern --
constructing a `NeuralSurrogate` is what actually requires torch, not
importing the module that defines it)."""
import numpy as np

from .nonlinear_rom import NeuralSurrogate


class EnsembleUQ:
    """Trains `n_members` independently-seeded copies of the same
    model family and reports mean/std across the ensemble at predict
    time -- a cheap, well-understood epistemic-uncertainty proxy (see
    this module's own docstring).

    Parameters
    ----------
    model_factory : callable, seed -> a fresh model instance exposing
        fit(X, y, **kwargs) -> self and predict(X) -> array.
    n_members : int, default 5
    seeds : list of int, optional
        Defaults to range(n_members). Must have length n_members if
        supplied explicitly.
    """

    def __init__(self, model_factory, n_members=5, seeds=None):
        if seeds is None:
            seeds = list(range(n_members))
        elif len(seeds) != n_members:
            raise ValueError(
                f"seeds has {len(seeds)} entries, expected n_members={n_members}")
        self.model_factory = model_factory
        self.n_members = n_members
        self.seeds = seeds
        self.members = []

    def fit(self, X, y, **fit_kwargs):
        """Fits `n_members` fresh model instances, one per seed, each
        on the SAME (X, y) training data -- the ensemble's spread then
        reflects only the model family's own training stochasticity
        (random init, mini-batch order, etc.), not a data-sampling
        effect (bootstrap/bagging ensembles are a different, also
        legitimate, technique -- out of scope for this item, which the
        roadmap frames specifically around seed variation)."""
        self.members = []
        for seed in self.seeds:
            model = self.model_factory(seed)
            model.fit(X, y, **fit_kwargs)
            self.members.append(model)
        return self

    def predict_all(self, X):
        """Returns (n_members, n_points, n_out) -- every member's own
        prediction, un-aggregated (a caller wanting the raw per-member
        spread rather than just mean/std uses this directly)."""
        if not self.members:
            raise RuntimeError("EnsembleUQ.predict_all() called before fit()")
        preds = [np.atleast_2d(np.asarray(m.predict(X), dtype=float)) for m in self.members]
        return np.stack(preds, axis=0)

    def predict_mean_std(self, X):
        """Returns (mean, std), each (n_points, n_out) -- the ensemble
        mean prediction and its per-point, per-output-component
        standard deviation across members (the epistemic-uncertainty
        proxy itself)."""
        preds = self.predict_all(X)
        return preds.mean(axis=0), preds.std(axis=0)


def neural_surrogate_ensemble(n_modes, q_samples, F_samples, n_members=5, seeds=None,
                               **surrogate_kwargs):
    """Convenience constructor: builds and fits an EnsembleUQ of
    `NeuralSurrogate` instances (rom_engine's own trainable, seed-
    sensitive `ReducedForceModel` -- Wave 7 item 35) -- the concrete,
    real use case this whole item exists for ("once ANY of items 99/
    100/103 introduces a trainable NN ... correction"). Requires torch
    (NeuralSurrogate's own constructor does; this function itself adds
    no additional torch dependency beyond that).

    surrogate_kwargs are forwarded to every NeuralSurrogate(...)
    construction (hidden_sizes, activation, lr, n_epochs, ...) except
    `seed`, which this function supplies itself, one per ensemble
    member.

    Returns a fitted EnsembleUQ."""
    def factory(seed):
        return NeuralSurrogate(n_modes=n_modes, seed=seed, **surrogate_kwargs)

    ensemble = EnsembleUQ(factory, n_members=n_members, seeds=seeds)
    ensemble.fit(q_samples, F_samples)
    return ensemble
