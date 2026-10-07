# Differentiable ROM correction: design note (Wave 10 item 103)

Source: `fea_engine/docs/consolidated_future_roadmap.md` Wave 10, item
103, itself sourced from Saverio, Bucci, Farro, Content & Sipp, "An
end-to-end PyTorch interface for differentiable PDE solvers: A RANS
model-correction study" (*Data-Centric Engineering* 7:e37, 2026,
doi:10.1017/dce.2026.10066). Written 2026-09-13, alongside the
`differentiable_correction.py` prototype this document describes.

The roadmap item's own framing: *"applies items 98-100's machinery to
the sibling ROM projects' own problem... this item is a DESIGN NOTE +
prototype, not a full reimplementation of either ROM project --
scoping the actual interface boundary between fea_engine's new
item-98/100 machinery and the ROM projects' own snapshot/training
pipeline is the first concrete step."* This document is that scoping
pass.

## 1. The analogy, stated precisely

The paper couples an existing RANS solver to a trainable additive
closure correction `f_theta`, calibrated either explicitly (against a
full reference flow field) or implicitly (through an adjoint-wrapped
solve, for partial/indirect reference data). `fea_engine`'s Wave 10
(`differentiable.py`) built the direct structural-mechanics analogue:
`F_ext - F_int(u) - f_theta(u) = 0`, with an explicit calibration path
(item 100) and an implicit whole-solver adjoint layer (item 98).

`rom_engine`'s own nonlinear-ROM machinery (`nonlinear_rom.py`) has an
EXACT structural counterpart at the REDUCED-coordinate level:
`PolynomialModalROM.predict(F_ext=..., Lambda=...)`'s own equilibrium
condition is

    Lambda * q + F_nl(q) = F_ext,  i.e.  F_ext - Lambda*q - F_nl(q) = 0

which is already a Newton-solved nonlinear residual, exactly the shape
`fea_engine.differentiable`'s machinery was built for -- just at
reduced-coordinate scale (a handful of modes) instead of full mesh
scale (thousands of dofs). The RANS-closure-correction analogy the
paper draws (`fea_engine`'s own docstring already quotes this: "the
paper itself names solid-mechanics constitutive modeling as a direct
analogue, Sec 2.3") extends one level further, cleanly, to *reduced
structural model correction*.

## 2. Why NOT literally import `fea_engine.differentiable`

`rom_engine`'s own stated design principle (`rom_engine/__init__.py`):
`fea_engine` is a TEST/EXAMPLE dependency only, never a LIBRARY
dependency -- this package's actual source code must import and run
with zero `fea_engine` present (many users of `rom_engine`'s ROM
machinery will never touch a real FE mesh at all; the ROM layer is
useful standing alone, e.g. driven by an externally-supplied
`fom_solver` callable, as `AppliedLoadStrategy`/`EnforcedDisplacement
Strategy` already do). This is the SAME reasoning that led
`NeuralSurrogate` (Wave 7 item 35) to independently re-implement
`_HAS_TORCH`/`_require_torch()` rather than importing anything from
`fea_engine.autograd_tangent`, even though the underlying technique
(row-by-row `torch.autograd.grad`) is identical.

`differentiable_correction.py` therefore reimplements the SAME
"`R(w)+f_theta(w)=0`, calibrate explicitly against a known reference
state" pattern independently, at the reduced-coordinate level:
`reduced_residual()` mirrors `corrected_residual()`,
`calibrate_reduced_correction_explicit()` mirrors `calibrate_
correction_explicit()`, and `ScalarModalCorrection` mirrors
`ScalarFieldCorrection` -- same math, same sign convention, no shared
code, no shared import.

## 3. What's built vs. not built

**Built** (`differentiable_correction.py`, tested in
`tests/test_differentiable_correction.py`):

- `reduced_residual(q, Lambda, F_nl_fn, F_ext, correction=None)` --
  pure NumPy, no torch needed. The reduced-coordinate analogue of
  `corrected_residual()`.
- `calibrate_reduced_correction_explicit(...)` -- the reduced-
  coordinate analogue of `calibrate_correction_explicit()` (Wave 10
  item 100's own pattern): calibrates a trainable correction against a
  KNOWN reference equilibrium `q_m`, no Newton solve needed, one
  forward+backward through `f_theta` alone per optimizer step.
- `ScalarModalCorrection` -- the simplest possible trainable
  correction at this level, mirroring `ScalarFieldCorrection`.

**Deliberately NOT built** (the item's own "DESIGN NOTE... first
concrete step" framing, not a full reimplementation):

- The reduced-coordinate analogue of item 98's whole-solver implicit-
  adjoint layer -- needed to calibrate a correction against PARTIAL or
  INDIRECT reference data (e.g. a handful of sparse sensor
  measurements of `q`, not a full `q_m`) by differentiating through an
  UNCONVERGED-at-calibration-time `PolynomialModalROM.predict(F_ext=
  ..., Lambda=...)` Newton solve, the same way `implicit_correction_
  solve()` differentiates through `_newton_equilibrium()`. Building
  this is a genuine follow-on item, not attempted here -- it needs its
  own adjoint derivation against `PolynomialModalROM`'s specific
  Newton solve (`nonlinear_rom.py`'s own `predict()` internals), which
  is real, separate work.
- A `MLPCorrection`/`NeuralSurrogate`-style NONLINEAR trainable
  correction at reduced-coordinate scale (`ScalarModalCorrection` is
  constant-only, matching `ScalarFieldCorrection`'s own scope) --
  a natural next step once the explicit-calibration path above has a
  real use case driving it, not built speculatively here.
- Any actual END-TO-END demonstration against a real fea_engine
  high-fidelity reference (this document's own examples are synthetic,
  matching `test_differentiable_correction.py`'s own "synthetic
  ground-truth first" convention, the same one `test_nonlinear_rom.py`
  already established for `MultiFidelitySurrogate`/`PolynomialModal
  ROM` before their own separate real-fixture file, `test_nonlinear_
  rom_fea.py`, was written). A real-fixture version of this item's own
  tests is a reasonable follow-on, not attempted in this pass.
- Item 101's metric-consistent gradient scaling has no analogue built
  here either -- `ScalarModalCorrection`'s parameter is a handful of
  per-mode scalars (already Euclidean-natural, per `MetricScaledField`
  's own docstring), not a spatially-varying field, so it isn't
  needed yet at this scale.

## 4. When to revisit this

Per the roadmap's own framing, this item exists to answer "is the
paper relevant to where the ROM packages are headed" -- the answer is
yes, and the interface boundary above is now concrete enough to extend
should a real use case need it: a genuine partial/sparse-measurement
ROM calibration problem showing up in `NonLin-HyROM` or
`Multi_Fidelity_NL_Structural_ROM` is the natural trigger for building
the implicit-adjoint counterpart described in Section 3 above.
