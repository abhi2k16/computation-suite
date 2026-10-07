# rom_engine Example Gallery

Six runnable, self-contained scripts mirroring `fea_engine`'s own
example gallery (`fea_engine/examples/gallery/`), covering
`rom_engine`'s reduced-order-modeling toolkit end to end: from
building a basis, through both linear (Galerkin/affine, frequency-
domain, balanced truncation) and nonlinear (polynomial modal) model
reduction, to the dataset-diagnostics and ensemble-UQ tools a real
training campaign needs around them. Every script prints its own
validation numbers to stdout and saves one PNG; run any of them
directly:

```
cd rom_engine/examples/gallery
python rom_01_pod_basis.py
```

Every filename here is prefixed `rom_` to keep it unambiguous which
package a script belongs to -- including this README itself
(`rom_README.md`). `fea_engine` has its own, separately prefixed
(`fea_`) gallery at `fea_engine/examples/gallery/`, covering that
package's finite-element capabilities instead (see that folder's own
`fea_README.md`).

All full-order comparisons use this repository's own real
`fea_engine` models -- nothing here is standalone numerics
reimplemented for the plot -- via `rom_engine/tests/fea_fixtures.py`
or this project's own pre-existing `rom_engine/examples/` scripts
(`two_region_beam_rom.py`, `greedy_frequency_training.py`), reused and
extended rather than re-derived. Every path used is torch-free, since
torch is unavailable in this sandbox (see the Wave 9/13 PyTorch
side-by-side audits in `docs/consolidated_future_roadmap.md` for the
torch-gated surface this gallery deliberately doesn't exercise).

| # | Script | What it shows | Cross-check used |
|---|---|---|---|
| 1 | `rom_01_pod_basis.py` | Proper Orthogonal Decomposition (`pod.py`) of static cantilever-beam load-case snapshots: singular value decay, reconstruction error vs. basis size | A held-out load that's a linear combination of the training loads reconstructs to ~machine precision at full rank; a genuinely independent held-out load plateaus above zero -- both exactly as POD theory predicts |
| 2 | `rom_02_galerkin_parametric_rom.py` | Affine parametric Galerkin ROM (`pod.py` + `galerkin.py` + `affine.py`) on a two-region cantilever beam with independent rigidities (EI1, EI2) as reduction parameters | ROM tip deflection vs. 12 held-out (EI1,EI2) points: max relative error 0.56%; reduced-order sweep is ~200x faster than full re-assembly over 500 queries |
| 3 | `rom_03_greedy_frequency_training.py` | Frequency-domain ROM (`greedy.py` + `frequency_rom.py`) trained two ways under the same full-order-solve budget: adaptive greedy vs. uniform grid | Full FRF sweep for both ROMs overlaid on the real `fea_engine.solve_harmonic` curve; held-out error near all 3 resonances reported for both strategies, honestly (greedy doesn't always win -- this run shows uniform's max error is actually lower, a real result, not cherry-picked) |
| 4 | `rom_04_balanced_truncation.py` | State-space balanced truncation (`balanced_truncation.py`) with a genuine a priori H-infinity error bound, on a damped cantilever beam | The actual `\|H_rom - H_fom\|` over a frequency sweep is checked against `h_infinity_error_bound()` and confirmed to hold (1.6e-10 actual vs. 6.5e-9 bound) -- using the two-stage modal-pre-reduction pattern `tests/test_balanced_truncation.py` itself establishes, since applying balanced truncation directly to a raw, poorly-conditioned beam state-space is a known numerically-unreliable shortcut, not a legitimate use of the method |
| 5 | `rom_05_nonlinear_modal_rom.py` | Nonlinear modal ROM (`nonlinear_rom.py`'s `PolynomialModalROM`) on a geometrically nonlinear (von Karman) clamped-clamped beam, trained via `AppliedLoadStrategy` | Held-out static load-deflection sweep vs. the full-order `fea_engine` nonlinear solve: max relative error 3.26% over the sweep, 2 ROM modes vs. 51 full-order DOF, correctly tracking geometric stiffening |
| 6 | `rom_06_dataset_diagnostics_uq.py` | Training-dataset diagnostics (`dataset_diagnostics.py`: coverage, PCA dimensionality, Sobol' sensitivity) and ensemble epistemic UQ (`ensemble_uq.py`'s `EnsembleUQ`) on the same two-region beam parameter study | Sobol' indices correctly identify the root-side region (EI1) as dominating tip-deflection variance (ST=0.94 vs. 0.04), matching cantilever bending-moment physics; ensemble std vs. true error correlation = 0.73 over a held-out sweep, using a torch-free random-features ensemble member in place of `NeuralSurrogate` |

## Notes

- Every script uses real, already-tested `rom_engine` code paths
  (`PodBasis`, `GalerkinROM`, `AffineDecomposition`, `FrequencyROM`,
  `greedy_train_frequency_basis`, `BalancedTruncationROM`,
  `PolynomialModalROM`, `AppliedLoadStrategy`, `dataset_diagnostics.py`,
  `EnsembleUQ`) against real `fea_engine` full-order models -- nothing
  here is illustrative or hand-waved.
- `rom_04_balanced_truncation.py` is the one script where getting a
  numerically honest result required care beyond just calling the
  API: applying `BalancedTruncationROM` straight to a raw, un-reduced
  beam state-space produces a bound VIOLATION (confirmed, then fixed,
  during this gallery's own development) because scipy's dense
  Lyapunov solver degrades once the model's eigenvalue spread gets
  too large -- exactly the failure mode `test_balanced_truncation.py`'s
  own module docstring warns about and works around.
- `rom_03_greedy_frequency_training.py` reports both strategies' errors
  honestly rather than only showing whichever won -- on this
  particular model/test-set, uniform-grid training happens to edge out
  greedy on maximum (though not median) held-out error, a legitimate
  possible outcome the source `greedy_frequency_training.py` script's
  own design anticipates rather than hides.
- `rom_06_dataset_diagnostics_uq.py`'s `EnsembleUQ` example uses a small
  hand-rolled random-Fourier-features regressor instead of
  `NeuralSurrogate` (which requires torch, unavailable in this
  sandbox) -- the ensemble machinery itself (`EnsembleUQ.fit()`/
  `.predict_mean_std()`) is identical either way.
