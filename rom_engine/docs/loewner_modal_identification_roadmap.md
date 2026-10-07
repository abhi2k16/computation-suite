# Non-intrusive Loewner-pencil modal identification for `rom_engine` -- roadmap

Status: **all four phases are implemented and validated.**
`metrics.py`, `loewner.py` (`LoewnerROM`), and `screening.py`
(`screen_physical_modes`) exist and are tested against a synthetic
mass-spring fixture, a real fea_engine damped cantilever, AND the
paper's own "Example 1" 867-DOF plate FE model (`tests/plate_fixtures.py`)
-- 66 tests total, all passing (`test_metrics.py`, `test_loewner.py`,
`test_screening.py`, `test_plate_fixtures.py`, `test_loewner_plate.py`).
Phase 4's examples (`examples/plate_modal_identification.py`,
`examples/mode_shape_vibration_recovery.py`) and the README updates are
also done. Phases 1-3 were built first (per the scope note originally
here); Phase 4 -- initially deferred as "a specific case to be
specified separately later" -- was subsequently confirmed and
completed in full, exactly as originally scoped in Section 7 below.

## 1. What this module is and why it's a new kind of module for rom_engine

Every module `rom_engine` has today (`pod`, `galerkin`, `affine`,
`frequency`, `greedy`, `scm`) is **intrusive**: every one of them takes
the actual system matrices `K`/`M`/`C` (or an affine decomposition of
them) as input. This module is **non-intrusive**: it identifies natural
frequencies, damping ratios, and mode shapes from sampled frequency-
response data `x(omega)` alone -- it never sees `K`, `M`, or `C`. That
makes it usable directly on real measured/experimental data or on a
proprietary/black-box FE model's exported harmonic response, not just
on models this package can assemble itself. `rom_engine/__init__.py`'s
existing "Roadmap (not yet built)" paragraph does not currently mention
this capability at all -- it lists a nonlinear-surrogate module, a PSD
random-vibration analyzer, full LP-based SCM, and Krylov/SOAR as the
only unbuilt items. This document (and the `__init__.py` edit in Phase
4 below) adds non-intrusive modal identification as a new, distinct
roadmap entry.

Reference: Liu, J. & Li, S. (2026), "A Reduced-Order Model for Modal
Parameter Identification of Fluid-Structure Interaction Systems Based
on Vibration Responses," *J. Vib. Eng. Technol.* 14:335. Method: the
**Loewner framework** (Mayo & Antoulas, *Linear Algebra Appl.* 2007) --
given transfer-function samples at two disjoint interpolation-point
sets, build a matrix pencil whose generalized eigenvalues are the
system's poles, with no state-space realization ever assembled.

## 2. What the working prototype actually does (verified against the real file)

`rom_modal_identification.py` (301 lines, confirmed by direct read) has
four parts:

1. **Synthetic ground-truth system** (`build_system`, `frf`,
   `ground_truth_modes`) -- a stand-in for "the structure," used only to
   generate test data and to check the ROM's answers. None of this is
   part of the method itself and none of it should be ported into
   `loewner.py` as library code -- it belongs in test fixtures /
   examples only, exactly like `build_system` played the paper's role of
   "the ANSYS+BEM model."
2. **`build_rom(omega_alpha, omega_beta, x_alpha, x_beta)`** -- Eqs.
   (21)-(22): assembles the Loewner pencil `(A_tilde, E_tilde)` directly
   from four 1-D arrays (two interpolation-frequency sets, two complex
   response-sample sets) via a double loop over `denom = wa2[a] - wb2[b]`.
   Solves the generalized eigenproblem `scipy.linalg.eig(A_t, E_t)`
   (Eq. 25), converts eigenvalues to `omega_tilde = sqrt(lambda)`, then
   to `f` (Eq. 26) and a loss-factor-like damping ratio `eta` (Eq. 27).
   Returns `(f, eta, Phi_tilde)`.
3. **`reconstruct_mode_shapes(omega_beta, X_beta_multi, Phi_tilde)`** --
   Eq. (28): `Phi = X_beta_multi @ Phi_tilde`, a single matrix product
   projecting multi-DOF response samples onto the ROM eigenvectors.
4. **`modal_assurance_criterion(phi1, phi2)`** -- standard complex MAC,
   `|<phi1,phi2>|^2 / (<phi1,phi1> <phi2,phi2>)`.
5. **`multi_rom_screening(freq_pool, x_pool, fmin, fmax, n_roms, n_interp, eps_f, eps_eta, tau_s)`**
   -- Eqs. (30)-(32) plus the paper's "Automated Screening of Physical
   Modes" algorithm: builds `n_roms` Loewner ROMs from random
   interpolation-frequency subsets, keeps in-band/physically-plausible
   eigenpairs, connects eigenpairs across *different* ROMs whose
   `(f, eta)` are close (a graph via `scipy.sparse.csgraph.connected_components`),
   then computes a per-cluster stability index `S_total = Sf * Se` from
   each cluster's frequency/damping coefficient-of-variation and keeps
   clusters that are both stable (`S_total > tau_s`) and occur in at
   least half the ROMs. Returns a sorted list of
   `(f, eta, S_total, occurrence_count)` tuples -- this is what actually
   separates genuine physical modes from an oversized/arbitrary
   interpolation-order's spurious numerical artifacts.

**Confirmed deviation to make when porting**: the prototype uses a
module-level global `rng = np.random.default_rng(42)`, read inside
`build_system`, `multi_rom_screening`, and `main()`. Every other
`rom_engine` module takes randomness (or avoids it) explicitly through
its own arguments/state -- no module has hidden global RNG state today.
`screening.py`'s screening function should take `rng` as an explicit
parameter (no default global), both for reproducible testing and for
consistency with the rest of the package.

## 3. What the three existing validation scripts already provide (confirmed to exist, read in full) -- now ported (Phase 4)

Three scripts originally sat in the outputs scratchpad, written against
`rom_modal_identification.py` as an importable module (`import
rom_modal_identification as rom1`). `plate_fe_model.py` and
`fem_rom_comparison.py` have since been ported (Phase 4, Section 7) --
`plate_fe_model.py`'s plate mesh/assembly became
`tests/plate_fixtures.py`, and `fem_rom_comparison.py`'s
identify-and-compare workflow became `examples/plate_modal_identification.py`,
re-pointed at `LoewnerROM`/`screen_physical_modes`/
`modal_assurance_criterion`. `mode_shape_vibration_recovery.py` became
`examples/mode_shape_vibration_recovery.py`, ported onto the synthetic
mass-spring fixture (inlined, self-contained, matching this package's
existing per-example convention) rather than the plate model. The
descriptions below are kept as-written (they describe the ORIGINAL
prototype scripts) since they're still an accurate record of what was
ported from:

- **`plate_fe_model.py`** -- builds the paper's actual "Example 1"
  benchmark: a from-scratch 4-node Reissner-Mindlin plate FE model
  (867 structural DOF, 16x16 mesh) matching the paper's stated
  geometry/material/mesh exactly, validated against the closed-form
  Navier solution for a simply-supported plate. Adds an approximate,
  honestly-caveated fluid added-mass model (air/water) since the
  paper's own BEM-coupled impedance matrix is proprietary and can't be
  reproduced. Saves `plate_fe_model_results.npz` (K, M variants, C,
  Table-1 node map) for reuse.
- **`fem_rom_comparison.py`** -- loads that saved model, generates
  synthetic response data at the paper's own Table-1 excitation/
  measurement node locations, runs `build_rom` + `multi_rom_screening`
  on ONLY that response data, and reports identified vs. true frequency/
  damping/MAC for both air- and water-loaded cases, plus a summary
  plot. This is a materially stronger validation target than a generic
  mass-spring chain: a real, paper-matched FE plate model.
- **`mode_shape_vibration_recovery.py`** -- re-runs the identification
  pipeline (same seed) on the synthetic mass-spring system, then adds
  two things not in the base prototype: full-field mode-shape recovery
  (Eq. 28 applied at every DOF, not just a few measurement points) and
  time-domain vibration recovery (propagate the identified modal model
  forward in time from an observed initial condition via least-squares
  modal-coordinate projection, `pinv(Phi_model) @ x0`, then compare
  against a direct full-order time integration).

These three scripts would be good *content* to port into `rom_engine`'s
`examples/` and `tests/` in a future phase -- they currently import the
prototype module directly (`import rom_modal_identification as rom1`)
and call its functions by their prototype names/signatures, so any such
port would need to re-point them at the new `loewner`/`screening`/
`rom_engine.metrics` APIs rather than copying them verbatim. That port
is deferred; nothing in Section 6/7 below depends on it.

## 4. Concrete architectural fit: what's reusable vs. what's new

Unlike `frequency.py`/`greedy.py`, which compose `affine.py` +
`galerkin.py` because `A(omega)` is exactly affine in a matrix triple
this package already has machinery for, **the Loewner method shares no
machinery with the existing intrusive modules** -- there is no `K`/`M`/`C`
to project, no basis to build, no affine decomposition. It is a
genuinely independent code path. What *is* shared is the package's
conventions, not its algebra:

- Plain functions/classes operating on numpy arrays, not
  `fea_engine`-specific types (matches `pod.py`/`galerkin.py`/`affine.py`'s
  design principle stated in `rom_engine/__init__.py`).
- `fea_engine` used only in `tests/`/`examples/`, never imported by the
  library modules themselves.
- Explicit dtype handling: `build_rom`'s eigenproblem is inherently
  complex (`x_alpha`/`x_beta` are complex FRF samples); this is native
  to the method from the start, unlike `frequency.py`'s retrofit
  (Section 4 of the frequency-domain roadmap) -- no dtype-generalization
  prerequisite phase is needed here.
- Explicit `rng` threading (Section 2's deviation) instead of hidden
  module state, matching the "no hidden state" principle implicit in
  every other module's design (each takes its inputs as arguments).

## 5. Proposed API

```python
# rom_engine/loewner.py -- new module

class LoewnerROM:
    """Non-intrusive modal model identified directly from complex
    frequency-response samples via the Loewner pencil (Mayo & Antoulas
    2007; Liu & Li 2026 Eqs. 21-28). Never constructed from M/C/K --
    only from sampled x(omega) data, real (experimental) or FE-derived,
    it makes no difference to this class."""

    def __init__(self, f, eta, Phi_tilde, omega_beta, x_beta_ref):
        ...  # holds the raw ROM eigen-solution; enough state to call
             # reconstruct_mode_shapes() without re-solving

    @classmethod
    def fit(cls, omega_alpha, omega_beta, x_alpha, x_beta):
        """Eqs. (21)-(27): assemble the Loewner pencil (A~, E~) from
        two interpolation-frequency sets and their single-DOF complex
        response samples, solve the generalized eigenproblem, convert
        eigenvalues to (f, eta). Direct port of build_rom(), returns a
        LoewnerROM instance instead of a bare (f, eta, Phi_tilde) tuple."""

    def reconstruct_mode_shapes(self, X_beta_multi):
        """Eq. (28): Phi = X_beta_multi @ Phi_tilde -- multi-DOF mode
        shape reconstruction from response data sampled at the SAME
        omega_beta used to fit() this ROM."""
```

```python
# rom_engine/screening.py -- new module

def screen_physical_modes(freq_pool, x_pool, fmin, fmax, rng,
                           n_roms=20, n_interp=9,
                           eps_f=0.005, eps_eta=0.10, tau_s=0.95):
    """Eqs. (30)-(32): build n_roms LoewnerROMs from random
    interpolation-frequency subsets of freq_pool, cluster eigenpairs
    that agree across DIFFERENT ROMs (cross-ROM stability criterion),
    and keep only clusters that are both low-variance (S_total > tau_s)
    and recur in at least half the ROMs. Direct port of
    multi_rom_screening(), with rng threaded explicitly (see Section 2)
    instead of a module-global default. Returns a sorted list of
    (f, eta, S_total, occurrence_count) tuples -- the genuinely physical
    modes, separated from numerical artifacts of any one interpolation-
    point choice."""
```

```python
# rom_engine/metrics.py -- new module (generic modal QA, not Loewner-specific)

def modal_assurance_criterion(phi1, phi2):
    """Standard MAC between two (possibly complex) mode-shape vectors.
    Lives in its own module, not loewner.py, because MAC is a generic
    modal-comparison metric useful anywhere two mode shapes need
    comparing (e.g. a future intrusive-vs-non-intrusive cross-check,
    or comparing galerkin.py's modal basis against measured shapes) --
    not tied to how either mode shape was obtained."""
```

## 6. Validation plan (Phases 1-3 scope; same standard as every other module in this project)

- **Synthetic-system sanity check**: a small proportionally-damped
  mass-spring-damper fixture (adapt `build_system`/`ground_truth_modes`
  into `tests/loewner_fixtures.py`) with closed-form modal parameters --
  the cheapest, fastest correctness check, run first.
- **Real fea_engine fixture**: reuse `tests/fea_fixtures.py`'s existing
  damped cantilever (already built for `frequency.py`'s tests) --
  generate synthetic FRF samples with `fea_engine`'s own harmonic solve
  (never a hand-built matrix), feed ONLY those samples to `LoewnerROM.fit()`,
  and compare identified `(f, eta)` against `fea_engine`'s modal solve.
  This directly parallels how every other rom_engine module is validated
  against a real structural model, not just a toy system.
- **Non-intrusiveness as a checked property, not just a design intent**:
  a structural test asserting `LoewnerROM.fit()`'s signature/implementation
  never references `M`, `C`, or `K` -- e.g. by constructing the fixture's
  FRF samples in one function and passing only the resulting arrays into
  a *separate* scope that has no access to the system matrices, so a
  future accidental intrusive dependency would be a `NameError`, not a
  silent assumption.
- **Screening correctness**: verify `screen_physical_modes()` actually
  rejects spurious modes -- deliberately oversize `n_interp` (as the
  paper recommends and the prototype's `main()` demonstrates) so the raw
  single-ROM identification contains extra, non-physical eigenpairs, and
  confirm screening's retained count converges to the true mode count
  while the raw single-ROM count does not.
- **Basic mode-shape reconstruction check**: `reconstruct_mode_shapes()`
  against the damped-cantilever fixture's own true mode shapes via MAC
  -- the Phases 1-3 scope's version of this check; the fuller full-field
  + time-domain recovery demonstration (`mode_shape_vibration_recovery.py`,
  Section 3) is deferred along with the rest of the plate-benchmark
  material.
- **RNG-threading regression**: same `rng` seed reproduces identical
  screening results; different seeds are permitted to differ (since
  the algorithm is randomized by construction) but should still converge
  to the same physical-mode set on the fixtures above.

## 7. Phased roadmap (Phases 1-3 in scope now; Phase 4 deferred)

**Phase 1 -- `metrics.py` (small, standalone, immediately useful).**
Port `modal_assurance_criterion()`. Trivial in isolation, but unblocks
every later phase's tests, and is reusable outside this module per
Section 5's rationale.

**Phase 2 -- `loewner.py` core (the main deliverable).**
Implement `LoewnerROM.fit()` and `.reconstruct_mode_shapes()` per
Section 5. Validate against the synthetic mass-spring fixture and the
real damped-cantilever `fea_engine` fixture (Section 6, first two
bullets). This alone delivers a working, validated non-intrusive
identification capability and is a defensible stopping point if scope
needs to be kept small, mirroring how Phase 1 was scoped as the
"natural stopping point" in the frequency-domain roadmap.

**Phase 3 -- `screening.py` (automated physical-mode screening).**
Implement `screen_physical_modes()` with explicit `rng` threading
(Section 2). Validate the spurious-mode-rejection property and the
RNG-threading regression (Section 6).

**Phase 4 -- plate benchmark + examples + docs. DONE.**
`plate_fe_model.py` ported into `tests/plate_fixtures.py` (validated
against the closed-form Navier solution and a physically-obvious
fluid-loading ordering check, `test_plate_fixtures.py`);
`fem_rom_comparison.py` ported into `examples/plate_modal_identification.py`
(re-pointed at `LoewnerROM`/`screen_physical_modes`/
`modal_assurance_criterion`; matches 6/7 air-loaded and 7/7 water-loaded
true modes to within 3% frequency error); `mode_shape_vibration_recovery.py`
ported into `examples/mode_shape_vibration_recovery.py` (full-field MAC
> 0.99 on 5/6 plotted modes, time-domain recovery RMS error 1-2.4% at
both excited and unexcited DOFs); a dedicated `test_loewner_plate.py`
also validates the identify-then-screen pipeline directly against the
plate fixture (not just the example script's console output). README
updated (module table, intrusive-vs-non-intrusive section, quick-start
snippet, package layout, test-suite summary, example descriptions).
`rom_engine/__init__.py`'s "Roadmap (not yet built)" paragraph was left
unchanged, as planned -- non-intrusive identification was never listed
there (it's a Modules-list entry now, like every other shipped module,
not a roadmap item).

Each phase was confirmed complete (tests passing, synced to the
canonical `computation-suite/rom_engine/` location) before starting the
next, consistent with this project's established
phase-by-phase-with-confirmation workflow.

## Sources

- Mayo, A.J. & Antoulas, A.C. (2007). "A framework for the solution of
  the generalized realization problem." *Linear Algebra and its
  Applications*, 425(2-3):634-662.
- Liu, J. & Li, S. (2026). "A Reduced-Order Model for Modal Parameter
  Identification of Fluid-Structure Interaction Systems Based on
  Vibration Responses." *Journal of Vibration Engineering &
  Technologies*, 14:335. (Eqs. 21-28: Loewner ROM construction,
  eigenvalue-to-modal-parameter conversion, mode-shape reconstruction.
  Eqs. 30-32 + "Automated Screening of Physical Modes": cross-ROM
  stability criterion.)
- `rom_modal_identification.py` (this project's working prototype,
  301 lines, read in full for this document) -- the direct source for
  Section 2's function-by-function mapping.
- `plate_fe_model.py`, `fem_rom_comparison.py`,
  `mode_shape_vibration_recovery.py` (this project's existing scripts,
  read in full for this document) -- the source for Section 3's
  deferred-phase reference material.
- `docs/frequency_domain_rom_roadmap.md`, `docs/phase4_error_bounds_greedy_roadmap.md`
  (this package) -- format and phasing-convention precedent.
