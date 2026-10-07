"""
rom_06_dataset_diagnostics_uq.py -- Example Gallery: training-dataset
diagnostics (dataset_diagnostics.py: coverage, PCA dimensionality,
Sobol' sensitivity) and ensemble-based epistemic uncertainty
quantification (ensemble_uq.py's EnsembleUQ) on the SAME two-region
cantilever-beam parameter study examples 01/02 already built --
reusing rom_engine.sampling.optimal_lhs() for the design-of-
experiments table, exactly as nonlinear_rom.py's own training
strategies do internally.

This is the "am I about to train a ROM/surrogate on a good dataset,
and how much should I trust its predictions" companion to the direct
reduction examples: before spending a training budget, check (1) does
the design actually cover the parameter range without gaps, and (2)
which parameters actually drive the output variance (so a caller
knows whether it is safe to skip sweeping a parameter that barely
matters). Then, once a small surrogate IS trained, EnsembleUQ turns
"how much do independently-seeded fits of the SAME data disagree" into
a per-query uncertainty estimate that requires no torch, no analytic
error bound, and no ground truth at prediction time -- only re-fits.

EnsembleUQ's own docstring frames it around a trainable NN surrogate
(NeuralSurrogate, which needs torch and is unavailable in this
sandbox); this script uses a small hand-rolled RandomFeatureRidge
regressor instead -- Rahimi & Recht 2007 random Fourier features, a
genuinely stochastic-per-seed model family (like a NN's random weight
init) fit by a single linear solve -- so the SAME EnsembleUQ machinery
is demonstrated torch-free, on real data from this repository's own
fea_engine two-region beam model.

Panel 1: parameter-space coverage (the LHS design in (EI1,EI2)) plus
the PCA-estimated effective dimensionality of the resulting tip-
deflection dataset, printed to stdout.
Panel 2: Sobol' first-order/total-order sensitivity of tip deflection
to EI1 vs. EI2 (bar chart) -- checked against the physical
expectation that EI1 (the ROOT-side region of a cantilever, where
bending moment and curvature are largest) should dominate.
Panel 3: EnsembleUQ mean +/- std envelope over an EI1 sweep at fixed
EI2, with the true full-order fea_engine curve overlaid, showing the
ensemble spread widening in the low-EI1 (highly nonlinear 1/EI-ish
response) region where the surrogate is genuinely least certain.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from two_region_beam_rom import build_two_region_beam  # noqa: E402

from rom_engine.sampling import optimal_lhs  # noqa: E402
from rom_engine.dataset_diagnostics import (  # noqa: E402
    parameter_coverage_report, pca_dimensionality, GPSurrogate, sobol_indices,
)
from rom_engine.ensemble_uq import EnsembleUQ  # noqa: E402

EI_LO, EI_HI = 0.5, 5.0
N_TRAIN = 24
TIP_LOAD = -1000.0


class RandomFeatureRidge:
    """Random Fourier features (Rahimi & Recht 2007) + ridge regression:
    a cheap, genuinely stochastic-per-seed regressor (the random
    feature directions/phases depend on `seed`, exactly like a small
    NN's random weight initialization) fit by a single linear solve --
    used here purely as a torch-free stand-in for EnsembleUQ's usual
    NeuralSurrogate ensemble member, so the SAME ensemble machinery
    this module ships is demonstrated without requiring torch."""

    def __init__(self, seed, n_features=60, length_scale=2.0, reg=1e-4):
        self.rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.length_scale = length_scale
        self.reg = reg

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        d = X.shape[1]
        self.W = self.rng.standard_normal((d, self.n_features)) / self.length_scale
        self.b = self.rng.uniform(0, 2 * np.pi, self.n_features)
        Phi = np.sqrt(2.0 / self.n_features) * np.cos(X @ self.W + self.b)
        A = Phi.T @ Phi + self.reg * np.eye(self.n_features)
        self.coef_ = np.linalg.solve(A, Phi.T @ y)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Phi = np.sqrt(2.0 / self.n_features) * np.cos(X @ self.W + self.b)
        return Phi @ self.coef_


def main():
    model = build_two_region_beam(n=120)
    free = model["free_dofs"]
    tip_dof_local = len(free) - 2
    F_free = np.zeros(len(free)); F_free[tip_dof_local] = TIP_LOAD

    def tip_deflection(EI1, EI2):
        K_ff = model["K_direct"](EI1, EI2)[np.ix_(free, free)]
        return np.linalg.solve(K_ff, F_free)[tip_dof_local]

    # ---- 1. Design-of-experiments table (rom_engine.sampling.optimal_lhs) ----
    rng = np.random.default_rng(0)
    unit_design = optimal_lhs(N_TRAIN, 2, rng=rng)
    params = EI_LO + unit_design * (EI_HI - EI_LO)     # rescale to (EI1, EI2) range
    y_train = np.array([tip_deflection(EI1, EI2) for EI1, EI2 in params])

    coverage = parameter_coverage_report(params, n_bins=6, param_names=["EI1", "EI2"])
    for rep in coverage:
        print(f"{rep['name']}: range [{rep['min']:.2f}, {rep['max']:.2f}], "
              f"gap in coverage: {rep['has_gap']}")

    pca = pca_dimensionality(y_train.reshape(-1, 1), variance_threshold=0.95)
    print(f"Output (tip deflection) PCA dimensionality: {pca['n_components']} "
          f"component(s) explain >=95% variance (expected: 1, it's a scalar)")

    # ---- 2. Sobol' sensitivity via a GP surrogate fit to the same data ----
    gp = GPSurrogate().fit(params, y_train)
    sobol_rng = np.random.default_rng(1)
    sobol = sobol_indices(bounds=[[EI_LO, EI_HI], [EI_LO, EI_HI]], model_fn=gp.predict,
                          n_samples=512, rng=sobol_rng)
    print(f"Sobol' indices -- S1 (first-order): EI1={sobol['S1'][0]:.3f}, "
          f"EI2={sobol['S1'][1]:.3f}")
    print(f"Sobol' indices -- ST (total-order): EI1={sobol['ST'][0]:.3f}, "
          f"EI2={sobol['ST'][1]:.3f}")
    dominant = "EI1 (root-side region)" if sobol['ST'][0] > sobol['ST'][1] else "EI2 (tip-side region)"
    print(f"-> {dominant} drives more of the tip-deflection variance "
          f"(physically expected: the ROOT region carries the larger bending moment)")

    # ---- 3. EnsembleUQ: torch-free RandomFeatureRidge ensemble ----
    def factory(seed):
        return RandomFeatureRidge(seed=seed)

    ensemble = EnsembleUQ(factory, n_members=15).fit(params, y_train)

    EI2_fixed = 1.5
    ei1_sweep = np.linspace(EI_LO, EI_HI, 60)
    sweep_X = np.stack([ei1_sweep, np.full_like(ei1_sweep, EI2_fixed)], axis=1)
    mean, std = ensemble.predict_mean_std(sweep_X)
    mean, std = mean.reshape(-1), std.reshape(-1)
    y_true_sweep = np.array([tip_deflection(EI1, EI2_fixed) for EI1 in ei1_sweep])
    abs_err = np.abs(mean - y_true_sweep)
    corr = np.corrcoef(std, abs_err)[0, 1]
    print(f"\nEnsembleUQ ({ensemble.n_members} members) at EI2={EI2_fixed}: "
          f"correlation(ensemble std, true error) = {corr:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    ax1, ax2, ax3 = axes

    ax1.scatter(params[:, 0], params[:, 1], c='steelblue', s=35, edgecolor='0.3', zorder=3)
    ax1.set_xlabel(r'$EI_1$')
    ax1.set_ylabel(r'$EI_2$')
    ax1.set_title(f'Design-of-experiments coverage\n'
                  f'({N_TRAIN} optimal-LHS points via rom_engine.sampling)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(EI_LO - 0.2, EI_HI + 0.2)
    ax1.set_ylim(EI_LO - 0.2, EI_HI + 0.2)

    x = np.arange(2)
    width = 0.35
    ax2.bar(x - width / 2, sobol['S1'], width, color='steelblue', label='first-order (S1)')
    ax2.bar(x + width / 2, sobol['ST'], width, color='tomato', label='total-order (ST)')
    ax2.set_xticks(x, [r'$EI_1$ (root region)', r'$EI_2$ (tip region)'])
    ax2.set_ylabel('Sobol\' sensitivity index')
    ax2.set_title('Sensitivity of tip deflection\nto each region\'s rigidity')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(fontsize=9)

    ax3.plot(ei1_sweep, y_true_sweep, '-', color='0.2', linewidth=2.0,
              label='true (fea_engine full-order)', zorder=3)
    ax3.plot(ei1_sweep, mean, '--', color='tomato', linewidth=1.6,
              label='EnsembleUQ mean', zorder=3)
    ax3.fill_between(ei1_sweep, mean - 2 * std, mean + 2 * std, color='tomato', alpha=0.2,
                      label=r'EnsembleUQ mean $\pm 2\sigma$')
    ax3.set_xlabel(r'$EI_1$ (EI2 fixed at %.1f)' % EI2_fixed)
    ax3.set_ylabel('tip deflection (m)')
    ax3.set_title(f'Ensemble epistemic-UQ envelope\n'
                  f'corr(std, true error) = {corr:.2f}, {ensemble.n_members} members')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    fig.suptitle('Dataset Diagnostics + Ensemble Epistemic UQ -- two-region cantilever beam (rom_engine)',
                 fontsize=13)
    out = os.path.join(os.path.dirname(__file__), 'rom_06_dataset_diagnostics_uq.png')
    fig.savefig(out, dpi=150)
    print("Saved rom_06_dataset_diagnostics_uq.png")


if __name__ == "__main__":
    main()
