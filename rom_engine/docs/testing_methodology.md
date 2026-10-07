# Testing methodology -- `rom_engine`

Status: current as of the `frequency.py` module addition. Test suite:
33 tests, `tests/`, run via `pytest tests/ -v`. Last full run: 33
passed, 0 failed.

## 1. Purpose of this document

This describes how `rom_engine` is validated, why it is validated that
way, what the current results are, and -- deliberately -- what this
testing regime does and does not prove. It is written to be read
alongside `docs/frequency_domain_rom_roadmap.md` (the design document
for the frequency-domain module) and the module docstrings themselves,
which contain the same reasoning in smaller pieces, closer to the code
it applies to.

## 2. Method

### 2.1 Validate against a real, independently-computed reference, never a synthetic or hand-derived one

Every correctness check in this package's test suite is run against a
real `fea_engine` structural model -- a cantilever beam, a two-region
beam with independent material properties, a damped cantilever -- and
the reference/expected answer always comes from `fea_engine`
**independently computing its own solution**: a direct linear solve, a
`scipy.linalg.eigh` eigendecomposition, `fea_engine`'s own
`solve_harmonic()`/`solve_frequency_sweep()`. It never comes from a
value derived by hand alongside the implementation under test.

This distinction matters. If the "expected" value in a test is derived
by the same reasoning (and the same potential mistake) as the
implementation, a shared bug in that reasoning can appear in both and
the test will pass anyway. Using a second, independently implemented
code path as ground truth closes that loophole: `fea_engine`'s solver
and `rom_engine`'s reduced-order machinery were written, and are
tested, as genuinely separate implementations of related but distinct
mathematics (full-order assembly and solve vs. basis projection and
reduced solve), so agreement between them is real evidence, not a
tautology.

### 2.2 One import boundary, enforced structurally

Only one file in the test suite, `tests/fea_fixtures.py`, imports
`fea_engine`. Every other test file (`test_pod.py`, `test_galerkin.py`,
`test_affine.py`, `test_frequency.py`) consumes plain `numpy` arrays
(`K`, `M`, `C`, snapshot matrices) that `fea_fixtures.py` produces, and
`rom_engine`'s own library code (`pod.py`, `galerkin.py`, `affine.py`,
`frequency.py`) never imports `fea_engine` at all. This is not
incidental tidiness -- it is what makes "validated against a real
model" and "FE-package-agnostic core" simultaneously true: the library
works with matrices from any source, and the test suite proves it
works correctly on matrices from one real, non-trivial source.

### 2.3 Real numeric assertions, not "did it run"

Every test asserts a specific, checkable numeric claim -- a relative
or absolute error under an explicit tolerance, or exact agreement to
machine precision where the underlying mathematics guarantees
exactness (e.g. a full-rank basis must reproduce the exact full-order
solution; a load that is a linear combination of a POD basis's own
training loads must be reproduced exactly by Galerkin projection).
Tests print the actual measured numbers (error magnitudes, timings,
energy captured) alongside the pass/fail assertion, so a human
reviewing test output sees the real evidence, not just a checkmark.

### 2.4 Test the failure modes deliberately, not just the happy path

A dedicated subset of tests exists specifically to catch classes of
bugs that produce a plausible-looking WRONG answer rather than a
crash -- the more dangerous failure mode in numerical code, because
nothing signals that anything is wrong. The concrete example from this
package's own history: `affine.py`'s `_theta()` method used to
hardcode `np.asarray(theta_func(mu), dtype=float)`. For the
frequency-domain module, `theta(omega) = i*omega` (the damping
coefficient) is purely imaginary -- that cast would have silently
discarded the entire imaginary part, with no exception and no warning,
silently solving the undamped problem regardless of what the damping
matrix actually was. This was fixed (Section 4 below), and
`test_frequency.py::test_general_damping_matches_fea_engine_
complex_solve` now asserts directly that a damped response's imaginary
(phase-lag) component is actually present and non-negligible, so this
specific failure mode cannot silently reappear undetected.

A second, related category tests **misuse**, not correctness: calling
`solve_static()` before `reduce_system()`, constructing an
`AffineDecomposition` with zero or mismatched-shape components, a
`theta_func` that returns the wrong number of coefficients. These
assert that the code fails LOUDLY (a raised, specific exception) in
situations where silent wrong output would otherwise be possible.

### 2.5 Standard test framework, `pytest`

`tests/` uses ordinary `pytest` conventions (`test_*.py` files,
`test_*` functions, plain `assert` statements) -- no custom test
runner or DSL. Run via `pytest tests/ -v` from the package root
(`pip install -e ".[fea,dev]"` first).

## 3. Why this matters (importance)

A reduced-order model is, by construction, an approximation -- the
question is never "is it exact" but "is it correctly approximating the
right thing, by a known and acceptable amount." Two failure modes are
possible, and they need different defenses:

- **A wrong reduction** (an algebra/implementation bug: a transposed
  matrix, a wrong sign, an off-by-one in an index) produces an answer
  that is not just approximate but categorically incorrect. Comparing
  against `fea_engine`'s independent full-order solve, with a tight
  tolerance, is the direct defense.
- **A silently degraded reduction** (the dtype bug in Section 2.4 is
  the concrete example; a basis that's technically valid but too small
  for the physics at hand is another) produces an answer that LOOKS
  plausible -- right order of magnitude, right general shape -- but is
  quietly, substantially wrong in a way a cursory glance would miss.
  This is the harder failure mode to catch, and it is why several
  tests assert something more specific than "close to the reference"
  -- e.g. that a damped response's imaginary part is actually
  non-trivial, not just that its magnitude looks reasonable.

Testing against a second, real, independently-computed system (rather
than, say, a value computed by the same formula the code under test
uses) is what makes a passing test meaningful evidence for either
failure mode, instead of only for the first.

## 4. Results (current, as of this document)

33 tests pass, 0 fail, across four modules, all validated against real
`fea_engine` models. Selected concrete figures (all pulled directly
from a fresh test run, not from memory):

**`pod.py`** -- Mass-weighted POD, given the exact eigenmodes of a real
`fea_engine` cantilever as its snapshot set, recovers the true
eigenspace with orthonormality error `6.661e-16` and reconstruction
error `1.318e-15` (machine precision). Standard (Euclidean) POD is
confirmed to differ genuinely from mass-weighted POD on the same data
(Euclidean orthonormality error `5.551e-16`, but MASS orthonormality
error `9.843e-01` -- i.e. a real, large difference, not a relabeling).
Reconstruction error is confirmed to decrease monotonically with rank
and vanish (`4.686e-16`) at full rank.

**`galerkin.py`** -- A full-rank basis reproduces the exact full-order
static solution to relative error `8.307e-13`. A load in the span of a
POD basis's training loads is reproduced to relative error `4.081e-13`
(both effectively machine precision, as the underlying math
guarantees). Modal frequencies from a basis built from STATIC (not
eigenmode) snapshots are shown to converge as rank grows: mode-0
relative error falls from `0.0046` at rank 2 to `0.0012` at rank 12.

**`affine.py`** -- The affine full-order reconstruction matches
`fea_engine`'s own independent per-block reassembly to relative error
between `9.19e-17` and `2.53e-16` across six random parameter draws
(machine precision). The "project-then-sum" and "sum-then-project"
orders of operation are confirmed algebraically identical (max abs
diff `~1e-10`, at the precision floor of the random reduced basis used
in that check). A measured (not assumed) online-query speedup of
`1635.5x` over full reassembly on a 3200-dof model.

**`frequency.py`** -- Matches `fea_engine`'s own `solve_harmonic()`
with relative error `8.95e-6` near a resonance and `8.08e-4` off
resonance (both well under the test's `5e-3` threshold; note the
NEAR-resonance case is, here, the MORE accurate one, because
proportional damping keeps the operator well-conditioned even at
resonance -- an example of a result worth reporting exactly as
measured rather than assumed in advance). A POD-on-FRF-snapshots basis
at rank 8 achieves relative error `8.5e-13` at a held-out frequency,
compared to `6.7e-4` for a same-rank modal basis on the same query --
both individually validated as accurate, reported side by side rather
than one asserted to "win." The proportional-damping 2-term collapse
matches the general 3-term form to max abs diff `9.5e-7` across six
frequencies. A measured sweep speedup of `60.3x` (100-point sweep, 300
free dof).

## 5. What this does -- and does not -- guarantee

Passing this suite is strong evidence the implementation is correct
for the cases it exercises, and reasonable inductive evidence it is
sound more generally, given that the checks span multiple mesh sizes,
random parameter draws, and both near- and off-resonance regimes
rather than one cherry-picked case each. It is not a proof of
correctness for every input, and this document says so explicitly
rather than implying otherwise:

- **Finite coverage.** The suite exercises specific mesh sizes (n=14
  to n=600 elements depending on the test), specific damping ratios,
  specific basis ranks. It does not fuzz across the input space or use
  property-based testing; an edge case entirely outside what's been
  tried (e.g. a near-singular mass matrix, an extremely ill-conditioned
  basis) is not covered by anything here.
- **Loose timing bounds, on purpose.** Speedup assertions (e.g.
  `assert speedup > 3`) are deliberately loose rather than tight, to
  avoid flaky failures on a shared/variable-load machine. The numbers
  reported in Section 4 are real measurements from one run; a rerun
  will differ somewhat, though not in a way that changes the
  conclusion.
- **`error_estimate()` is explicitly NOT a certified bound.**
  `FrequencyROM.error_estimate()` returns a numerical residual norm,
  useful as a relative ranking signal, but is deliberately named
  `error_estimate` rather than `error_bound` because it lacks a
  rigorous lower bound on the operator's smallest singular value --
  see `docs/frequency_domain_rom_roadmap.md` Section 2c/Phase 4 for
  what a certified version would require.
- **Known, previously documented unresolved discrepancies elsewhere in
  this project are not hidden.** The earlier Fig. 5 (MFS-NLROM paper)
  reproduction has one panel with an honestly-reported, unresolved
  ~6x mismatch against the published figure -- reported as such rather
  than force-fit, and not silently dropped from the project's history
  because a later module happened to go well.
- **Sandbox/environment limits are separate from code correctness.**
  e.g. `fea_engine`'s `hex8_convergence.py` example hits a memory
  ceiling on very large meshes in this development sandbox -- a
  resource limit of the environment it happened to be run in, not a
  defect the test suite is designed to catch or hide.

The honest summary: this package is extensively validated against
real, independently-computed structural-analysis results, with the
actual numbers and the actual limitations stated plainly. That is a
stronger and more useful claim than "foolproof" -- which no finite
test suite can truthfully claim for numerical software -- because it
tells you specifically what has been checked, what hasn't, and how to
extend the checking if a new case matters.
