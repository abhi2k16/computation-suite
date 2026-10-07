# Frequency-domain intrusive ROM for `rom_engine` -- research + roadmap

Status: **implemented and validated** -- this began as a design
document with no code written yet, but every phase described below,
including Phase 5's SOAR/TOAR second-order Krylov addendum (Section 8),
is now implemented (`frequency.py`, `greedy.py`, `scm.py`, `scm_lp.py`,
`random_vibration.py`, `soar.py`) and covered by `rom_engine`'s test
suite. The "design document" framing below is preserved as originally
written for its own historical record; see each phase's own section for
its current implementation status.

## 1. The problem

A linear structural model under harmonic excitation `F(t) = Re[F e^{i*omega*t}]`
has steady-state response `x(t) = Re[X(omega) e^{i*omega*t}]` where

```
A(omega) X(omega) = F,      A(omega) = -omega^2 * M + i*omega*C + K
```

`A(omega)` is complex, dense (or sparse) and, critically, **ill-conditioned
near resonance** (`A` becomes singular as `omega` approaches an undamped
natural frequency of the corresponding conservative system, and stays
poorly conditioned nearby even with damping). A "frequency-domain
intrusive ROM" is a reduced model built directly from `M`, `C`, `K` (as
opposed to a non-intrusive surrogate trained only on input/output
samples) that lets `A(omega) X(omega) = F` be solved cheaply at many
`omega` values -- exactly the situation a frequency sweep, a random-
vibration (PSD) analysis, or a resonance search needs.

## 2. What the literature actually does (three families)

### 2a. Modal / reduced-basis projection (closest to what `rom_engine` already has)

Project onto a fixed subspace `V` (undamped mode shapes, POD modes from
training-frequency snapshots, or both) and solve the tiny reduced
system `(V^T A(omega) V) q = V^T F` instead. This is a direct
generalization of `galerkin.py`'s existing static/modal solves to a
complex, frequency-dependent system matrix.

The key structural fact this design leans on: **`A(omega)` is exactly
affine in a 3-term basis** `{M, C, K}` with parameter-dependent
coefficients `theta(omega) = [-omega^2, i*omega, 1]`. That is *precisely*
the abstraction `affine.py` already implements (`K(mu) = sum theta_q(mu)
K_q`) -- just with `mu = omega` (a single real "parameter"), `Q = 3`, and
complex-valued `theta_q`. No new mathematical machinery is needed for
this path, only a dtype generalization (Section 4).

For **proportional (Rayleigh) damping**, `C = alpha*M + beta*K`, the
3-term decomposition collapses to 2 terms:
`theta_M(omega) = -omega^2 + i*omega*alpha`, `theta_K(omega) = 1 + i*omega*beta`.
This is exactly what `fea_engine.damping.RayleighDamping` already
represents, so it is the natural first real-world validation case
(reuse `fea_fixtures.py`'s existing cantilever, just add damping). For
**modal (hysteretic) damping**, `K` itself becomes complex,
`K(1+i*eta)`, still a 2-term affine form. Only a genuinely
non-proportional/physical damping matrix needs the full 3-term case.

For the *basis* `V` itself, two well-established choices, and this
project should support both since they trade off differently:

- **Undamped mode shapes** (`scipy.linalg.eigh(K, M)`, already exactly
  what `galerkin.solve_modal()` returns): cheap, physically
  interpretable, standard "modal truncation" / "modal superposition"
  method, works well away from resonance and for lightly damped
  structures, degrades if a *forced* response has significant
  content outside the retained modes.
- **POD on FRF snapshots** at a handful of *training* frequencies
  spanning the band of interest (solve `A(omega_k) x = F` fully for a
  handful of `omega_k`, `PodBasis.fit()` the resulting *complex*
  snapshot matrix): captures the actual (possibly non-modal, load-
  dependent) response shape better, is the method the applied POD-FRF
  literature (Section 6) reports outperforming plain modal truncation
  once enough energy accuracy is demanded. Costs more offline (real
  solves at each training frequency) and needs a frequency-sampling
  strategy (uniform grid to start; greedy, error-driven sampling is a
  natural later refinement -- see 2c).

### 2b. Krylov subspace / moment-matching (Padé-via-Arnoldi, SOAR, TOAR)

Instead of picking a physically-motivated basis, match the transfer
function's Taylor moments around one or more *expansion points*
`omega_0` via a (rational) Krylov subspace built from `A(omega_0)` and
its derivatives. For **second-order systems** (as opposed to a
first-order state-space linearization, which doubles the DOF count and
can destroy passivity/stability structure), the structure-preserving
variant is the **Second-Order Arnoldi method (SOAR)**, Bai & Su
(2005), later stabilized as **TOAR** (SOAR is known to be numerically
unstable in its original form). Reference survey: Bai (2002),
*"Krylov subspace techniques for reduced-order modeling of large-scale
dynamical systems"*, and the SOAR papers below.

This family is the *industry-standard* approach for very large models
(automotive/vibroacoustic FE models with 10^5-10^7 DOF) precisely
because it needs no full eigen-decomposition and matches the transfer
function's local behavior extremely accurately near the expansion
point(s) with a very small basis. It is materially harder to implement
correctly (moment matching for a *quadratic* eigenvalue-adjacent
problem, shift selection, breakdown/deflation handling) than the
affine-reuse path in 2a, and is not needed until 2a's accuracy or
basis-size is shown to be insufficient on a real target model -- so
this is scoped as a **later, optional phase**, not the initial
deliverable.

### 2c. Certified reduced basis method (rigorous error bounds + greedy sampling)

The reduced-basis-method (RBM) literature (Rozza/Huynh/Patera and
follow-ons, applied to the Helmholtz/frequency-response setting)
wraps exactly the 2a construction with two additions that turn "a ROM
that seems to work" into "a ROM with a computable guarantee":

- A **residual-based a posteriori error estimator**:
  `||x(omega) - V q(omega)|| <= ||A(omega)^{-1}|| * ||F - A(omega) V q(omega)||`.
  The residual norm is cheap to compute at any `omega` without a full
  solve; the hard part -- and the honest caveat to document rather
  than paper over, consistent with how this project has always
  handled unresolved difficulty (see the Fig. 5 reproduction's
  documented panel-b discrepancy) -- is a *rigorous, cheap* lower
  bound on `A(omega)`'s smallest singular value, which is exactly what
  degrades near resonance. A defensible starting point is a coarse
  numerical estimate (e.g. min singular value on a coarse `omega` grid,
  interpolated) rather than a certified bound, with that limitation
  stated plainly rather than implied to be rigorous.
- A **greedy offline sampling algorithm**: instead of a fixed uniform
  training-frequency grid, iteratively add the `omega` where the
  current error estimator is largest, re-solve the full model only
  there, and repeat -- concentrating training solves where they are
  actually needed (near resonances) instead of wasting them on flat
  regions of the FRF.

This is the natural **Phase 4** refinement of the same 2a machinery,
not a different code path.

## 3. Implementation-pattern reference: how existing Python ROM
   packages structure this

[pyMOR](https://docs.pymor.org) (the most complete open-source Python
MOR library) is a useful API precedent, not a dependency to add:

- It keeps a **`TransferFunction`** object (frequency response
  evaluation, Bode plots, H2-norm) *separate* from the **model**
  object that owns `M`, `C`, `K`/`E`, `B`, `C` -- newer pyMOR versions
  moved `SecondOrderModel`'s frequency-response methods onto
  `model.transfer_function` rather than the model itself, specifically
  so "give me `H(omega)`" and "here is my `(M,C,K)` system" are
  decoupled concerns. `rom_engine` should follow the same split: a
  `FrequencyROM` (owns the affine `M,C,K` decomposition + basis,
  analogous to today's `GalerkinROM`) and a thin
  `frequency_response(omega_array, input_dofs, output_dofs)` method/
  helper that returns a `(n_omega, n_out, n_in)` complex array --
  matching pyMOR's own returned-shape convention, which is a
  reasonable de facto standard to match for anyone coming from that
  ecosystem.
- pyMOR represents second-order systems with an explicit
  `SecondOrderModel(M, E, K, B, Cp, Cv, D)` class rather than folding
  everything into a generic first-order LTI system --
  structure-preservation (not linearizing to `2n` states) is treated
  as a first-class design choice there, reinforcing that `rom_engine`'s
  natural fit is extending `affine.py`/`galerkin.py` (which already
  operate on `(K, M, F)` triples) rather than introducing a first-order
  state-space abstraction rom_engine doesn't otherwise have.

## 4. Concrete architectural fit: what's reusable today vs. what
   must change

Read directly from the current source in
`computation-suite/rom_engine/src/rom_engine/`:

**Reusable almost as-is** (the offline/online split, the affine
component-sum algebra, the Galerkin projection machinery, the whole
`reduce_system()`/caching pattern) -- the *shape* of the problem
(`A(mu) = sum theta_q(mu) A_q`, project once, solve many) is identical
to what `affine.py` + `galerkin.py` already do for the real-valued
two-region-stiffness case validated in `test_affine.py`.

**Must change -- all three core modules currently hardcode
`dtype=float`, which will silently corrupt a frequency-domain
computation rather than error out:**

- `affine.py.__init__`: `self.components = [np.asarray(Kq, dtype=float)
  for Kq in components]` -- would silently discard the imaginary part
  of a complex damping/stiffness component.
- `affine.py._theta`: `theta = np.asarray(self.theta_func(mu),
  dtype=float).ravel()` -- **this is the critical one**: `theta_C(omega)
  = i*omega` is *purely imaginary*, so this line as written would
  silently zero out the entire damping contribution to every reduced
  query. This is exactly the kind of silent-not-loud failure mode this
  project's validation culture exists to catch -- it must be fixed
  before any complex `theta_func` is used, and a regression test should
  assert a complex `theta_func` round-trips correctly (not just "the
  code runs").
- `galerkin.py.__init__`, `.project_matrix()`, `.project_vector()`:
  same `dtype=float` casts on `basis`/`A`/`b`.
- `pod.py.fit()`, `.project()`, `.expand()`, `.reconstruction_error()`:
  same pattern -- needed if training snapshots are complex FRF
  solutions (Section 2a's second basis option); not needed if the
  basis stays real (undamped mode shapes, or POD on real static
  snapshots) and only the *reduced system* becomes complex.

The fix is mechanical and low-risk: replace the hardcoded `dtype=float`
with dtype inferred from the input (`np.asarray(x)` and let numpy's
type promotion do the right thing on `V.T @ complex_matrix @ V`), or an
explicit `dtype=complex` opt-in path. Every existing real-valued test
must keep passing bit-for-bit after this change -- a natural first
regression check.

## 5. Proposed API

```python
# rom_engine/frequency.py -- new module

class FrequencyROM:
    """Intrusive frequency-domain ROM: A(omega) = sum theta_q(omega) A_q,
    projected onto a reduced basis, following the same offline/online
    split as affine.AffineDecomposition (this class composes one
    internally rather than duplicating the algebra)."""

    def __init__(self, components, theta_func, basis):
        ...  # wraps AffineDecomposition(components, theta_func).project(basis)

    @classmethod
    def from_MCK(cls, M, C, K, basis, proportional_damping=None):
        """Convenience constructor for the M, C, K case -- builds the
        3-component (or 2-component, if proportional_damping folds C
        into M/K) affine decomposition and theta_func(omega) automatically,
        so a caller never has to hand-write theta(omega) = [-omega**2, ...]."""

    def solve(self, omega, F_r):
        """K_r(omega) q = F_r -- one frequency, reduced coordinates."""

    def frequency_response(self, omega_array, F, output_dofs=None):
        """Sweep omega_array, return (n_omega, n_out) complex array --
        the main "many-query" entry point, mirroring pyMOR's
        (n_omega, n_out, n_in) transfer-function convention."""

    def error_estimate(self, omega, F_r):
        """Residual-based a posteriori bound (Phase 4) -- returns a
        numerical estimate, explicitly documented as NOT a certified
        bound unless/until a real coercivity lower-bound estimator is
        added (Section 2c)."""
```

`build_pod_basis_from_frf_snapshots(training_omegas, A_func, F)` is a
free function (not a method) in the same module, since it needs a full-
order solve at each training frequency -- keeping it a plain function
that returns a snapshot matrix for `PodBasis.fit()` keeps `FrequencyROM`
itself free of any full-order-model dependency, preserving the
decoupling principle already stated in `rom_engine/__init__.py`.

## 6. Validation plan (same standard as every other module in this project)

Extend `tests/fea_fixtures.py` with a **damped** cantilever fixture:
reuse `cantilever_beam_system()`'s `K`, `M`, add
`fea_engine.damping.RayleighDamping` to build `C`, and a
`cantilever_frequency_sweep()` helper that does full-order complex
solves of `A(omega) x = F` at a set of `omega` (the ground truth every
ROM check below is measured against -- never a synthetic matrix,
matching the standard already set by `test_pod.py`/`test_galerkin.py`/
`test_affine.py`).

Planned checks, mirroring the existing suite's structure exactly:

- **Correctness at a resonance and off resonance**: reduced solution
  vs. full-order solution at both a frequency near a known undamped
  natural frequency (where `A(omega)` is nearly singular -- the hard
  case) and a frequency far from any mode (the easy case) -- report
  both, don't only report the easy one.
- **Convergence with basis rank**, both for the modal-truncation basis
  and the POD-on-FRF-snapshots basis -- directly comparable to
  `test_galerkin.py::test_modal_frequencies_converge_with_static_snapshot_basis_rank`'s
  existing pattern.
- **Complex `theta_func` round-trip regression**: assert the fixed
  `dtype=float` bug (Section 4) cannot silently reappear -- a
  purely-imaginary `theta_q` must actually affect `assemble_reduced()`'s
  output, checked by direct numerical comparison, not just "does it
  run without raising."
- **Sweep speedup**, mirroring
  `test_affine.py::test_reduced_online_queries_are_faster_than_full_reassembly`'s
  already-established pattern (loose, non-flaky bound, not a tight
  timing assertion).
- **Proportional-damping affine collapse**: verify the 2-term
  (`M`,`K`-only) decomposition derived from `RayleighDamping` matches a
  direct 3-term (`M`,`C`,`K`) decomposition to machine precision --
  confirms the algebraic simplification in Section 2a is implemented
  correctly, not just assumed.

## 7. Phased roadmap

**Phase 0 -- dtype generalization (prerequisite, small, high-value).**
Fix the `dtype=float` hardcoding in `pod.py`, `galerkin.py`, `affine.py`
identified in Section 4. Re-run the full existing 27-test suite
unchanged (must stay green) plus one new tiny complex-arithmetic
smoke test. Low risk, unlocks everything else.

**Phase 1 -- `FrequencyROM` via affine reuse (the core deliverable).**
Implement the `frequency.py` module from Section 5, `from_MCK()`
convenience constructor, proportional-damping collapse, undamped-modal
basis option. Validate against the damped-cantilever fixture (Section
6, first three checks). This alone delivers a working, validated
frequency-domain intrusive ROM and is the natural stopping point if
scope needs to be kept small.

**Phase 2 -- POD-on-FRF-snapshots basis option.**
Add `build_pod_basis_from_frf_snapshots()`, wire it as an alternative
to the modal basis in Phase 1, validate the convergence-with-rank
check for this basis choice too, and compare its accuracy-per-mode
against the modal basis on the same fixture (an honest empirical
comparison, not an assumed winner).

**Phase 3 -- example + docs.**
An end-to-end `examples/` script (mirroring
`two_region_beam_rom.py`'s existing shape: build a damped fea_engine
model, build the ROM, sweep, plot/report accuracy and speedup) plus a
README section, matching the documentation standard already set for
`pod`/`galerkin`/`affine`.

**Phase 4 -- certified error estimation + greedy frequency sampling
(optional, advanced).**
Section 2c's residual-based estimator and greedy training-frequency
selection, with the coercivity-bound caveat stated explicitly rather
than glossed over.

**Phase 5 -- Krylov/SOAR moment-matching path. DONE (SOAR half; see
Section 8).**
Not pursued because the affine/POD approach above needed a larger
basis on any real target model tried -- rather, taken up afterward at
the user's explicit request, as a second-order-structure-preserving
complement to `krylov.py`'s existing first-order path, and specifically
to test whether it inherits `passivity.py`'s already-proven second-
order-Galerkin passivity guarantee where `krylov.py`'s own first-order
`KrylovROM` provably does not.

Update: a plain, FIRST-ORDER (non-structure-preserving) Krylov
moment-matching module already existed -- `krylov.py`
(`docs/classical_mor_roadmap.md`), one- and two-sided, built to
reproduce Besselink et al. 2013's own method on a general
`(M,C,K,B,Cout)` I/O port. It solves a different problem than this
Phase 5 item (which is specifically about `FrequencyROM`'s own
resonance-handling in the affine/POD basis this package already uses)
-- worth restating precisely because IT'S ALSO the item SOAR/TOAR
below is directly compared against, being the other reduction of the
SAME `(M,C,K,B,Cout)` port. SOAR -- the second-order-structure-
preserving variant this Phase 5 is actually about, avoiding first-order
form's 2x state-count blowup -- is now implemented, in a new module
`soar.py`, as `soar_basis()` + `SOARROM`; TOAR (the further numerical-
stabilization of SOAR's own known fragility) is NOT implemented, a
deliberate scope decision explained in Section 8 below.

**Phase 6 -- PSD-driven random-vibration stress analyzer.**

Status: design addendum below, no code written yet -- pulled forward
from the `__init__.py` "possible future work" list. Unlike Phases 0-5
above, this one needs almost no new research: `fea_engine`'s own
`FESystem.solve_random_vibration()` already implements and validates
the exact formula this ROM analog reuses, so the design question here
is purely "how does this compose with `FrequencyROM`", not "what is the
right math".

*The problem.* Given a PSD input spectrum `S_in(f)` (e.g. base
excitation, turbulent buffet, road/rail roughness) applied through a
fixed spatial load pattern `F0`, the steady-state OUTPUT PSD at a
sensor DOF is `S_out(f) = |H(f)|^2 * S_in(f)`, where `H(f)` is the
system's own transfer function from `F0` to that DOF -- a direct
consequence of linear systems theory (PSDs transform through the
squared transfer-function magnitude), needing no new derivation.
`fea_engine.solver.FESystem.solve_random_vibration(freqs_hz, psd_input,
F0_pattern, output_dof)` already implements exactly this, at full
order: it builds `H` via `solve_frequency_sweep()`, forms `S_out`, and
integrates it (trapezoidal) for the response RMS `sigma_out`. The ROM
version is the same three lines, with `H` coming from
`FrequencyROM.frequency_response()` instead of a full-order solve --
literally the same "many-query, one basis" savings every other module
in this package already provides, applied to one more downstream
quantity.

*Proposed API* (new module `random_vibration.py`, since this is a
distinct capability composed ON TOP of `frequency.py` rather than
belonging inside it, matching the one-concept-per-module convention
already used throughout this package):

```python
def psd_response(rom_freq, freqs_hz, psd_input, F0_pattern, output_dofs):
    """S_out(f) = |H(f)|^2 * S_in(f), H from rom_freq.frequency_response().
    output_dofs: a single int (fea_engine-signature-compatible: returns
    scalar sigma_out) or an array-like of several DOFs (returns one
    sigma_out per DOF). Returns (S_out, sigma_out)."""
```

*Validation plan*: build a `FrequencyROM` from a real, damped
`fea_engine` cantilever (the SAME fixture `test_frequency.py` already
uses), and check `psd_response()`'s `S_out`/`sigma_out` against
`fea_engine`'s own `solve_random_vibration()` at full order -- both for
a PSD band containing a resonance (the hard case, where `|H(f)|^2`
peaks sharply and the trapezoidal integration is most sensitive to
frequency-grid resolution) and one that doesn't (the easy case), plus a
multi-output-DOF call to check the generalization beyond
`fea_engine`'s own single-`output_dof` signature.

## 8. Phase 5 addendum: SOAR second-order Krylov (`soar.py`)

### Derivation (why the recurrence is what it is)

`krylov.py` matches Taylor moments of `H(s) = Cout (sE-A)^-1 B`, the
FIRST-ORDER linearized transfer function, at the cost of doubling the
state count (`state_space.py`'s `[q; q_dot]` construction). SOAR (Bai &
Su 2005) matches the SAME moments of the SAME port's transfer function,
but works directly with the second-order pencil

    H(s) = Cout (s^2 M + s C + K)^-1 B,

never forming the `2n x 2n` first-order system at all. Expanding around
an expansion point `s0` (`s = s0 + sigma`), and defining
`K0 = K + s0*C + s0^2*M` (assumed invertible -- generically true for
`s0` not itself a system pole; `s0=0` gives `K0=K`, matching
`krylov.py`'s own default):

    s^2 M + s C + K = K0 * [I - sigma*A1 - sigma^2*A2],
    A1 = -K0^-1 (C + 2*s0*M),   A2 = -K0^-1 M

so `(s^2 M+sC+K)^-1 B = K0^-1 B * sum_j sigma^j Phi_j`, where `Phi_j`
is defined by the TWO-TERM recurrence `Phi_0=I`, `Phi_1=A1`,
`Phi_j = A1*Phi_{j-1} + A2*Phi_{j-2}` for `j>=2` -- the second-order
generalization of the ONE-term recurrence (`v_j = A v_{j-1}`) ordinary
Arnoldi builds its basis from. Applying this to `K0^-1 B` directly
(rather than tracking the operator `Phi_j` itself) gives the vector
recurrence SOAR's basis is built to span:

    r_1 = K0^-1 B,   r_0 := 0,   r_j = A1 r_{j-1} + A2 r_{j-2}  (j>=2)

An orthonormal basis `V` of `span{r_1,...,r_k}`, used exactly like
every other Galerkin projection in this package
(`M_r=V^T M V`, `C_r=V^T C V`, `K_r=V^T K V`, `B_r=V^T B`,
`Cout_r=Cout V`), gives a REDUCED SECOND-ORDER system of order `k`
(not `2k`) whose transfer function matches `H(s)`'s first `k` Taylor
moments at `s0` -- SOAR's whole point, and the source of its "avoids
the first-order 2x blowup" advantage over `krylov.py`'s own
`arnoldi_basis()`.

### SOAR vs. TOAR: what's built, and what's honestly not

Bai & Su's original SOAR construction (built here, in `soar_basis()`)
computes this basis via a block-generational Gram-Schmidt process
directly analogous to `krylov.py`'s own `arnoldi_basis()` -- each
"generation" of block-input columns is orthogonalized (full
reorthogonalization against every previously-kept basis vector, not
just the current generation, the standard mitigation for Arnoldi-type
loss-of-orthogonality) and deflated exactly as `arnoldi_basis()`
already does, generalized to require the PREVIOUS TWO generations
(not one) to form each new block of raw candidate directions, per the
two-term recurrence above.

SOAR is literature-documented as numerically fragile in its original
form (Bai & Su themselves; the module docstring's own citation of the
TOAR stability paper) -- particularly for BLOCK (multiple simultaneous
input directions) starting vectors, which is exactly this package's
use case (`B` with `n_in` columns). TOAR (Lu, Su, Bai) fixes this with
a "two-level" representation: every Krylov direction is stored as a
SHARED position-space basis `V` plus a small per-direction coefficient
pair, rather than as full-length vectors, both reducing storage and
(more importantly here) improving numerical stability by keeping the
[position; derivative]-style bookkeeping consistent across the whole
basis-building process rather than letting it drift.

This addendum deliberately does NOT implement that two-level
bookkeeping -- an honest scope decision, stated plainly rather than
glossed over (matching this project's established precedent: the
natural-norm SCM variant in `scm_lp.py`, the `D!=0` positive-real ARE
case in `passivity.py`, subharmonic-multiplier support in `nnm.py`).
TOAR's specific numerical advantage over plain SOAR matters most at the
very large (`10^5-10^7` DOF) industrial scale Section 2b's own
"industry-standard" framing is about; this package's real validation
fixtures are two to three orders of magnitude smaller, where full
reorthogonalization (already the standard, effective fragility
mitigation for Arnoldi-family methods generally, and already used
throughout `krylov.py`'s own `arnoldi_basis()`) is expected to behave
well -- checked directly below, not assumed. If a real target model
ever needs genuine TOAR, it is now a much narrower, well-scoped
follow-up (replace `soar_basis()`'s vector storage/orthogonalization
internals only -- the moment-matching recurrence and `SOARROM`'s
projection/API stay the same) rather than the whole SOAR-from-scratch
effort this addendum already did.

### API

New module `soar.py`, mirroring `krylov.py`'s own shape:

```python
def soar_basis(M, C, K, B, k, s0=0.0, tol=1e-12):
    """Orthonormal (n_dof, k') basis of the order-k second-order Krylov
    subspace spanned by r_1=K0^-1 B, r_j = A1 r_{j-1} + A2 r_{j-2}."""

class SOARROM:
    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, s0=0.0, k=10): ...
    def transfer_function(self, s): ...          # Cout_r(s^2 M_r+sC_r+K_r)^-1 B_r
    def frequency_response(self, omega_array): ...
    def is_stable(self): ...       # quadratic eigenvalue problem on (M_r,C_r,K_r)
```

`SOARROM.k` here means the same thing as `KrylovROM`'s `k` means
SECOND-ORDER states (`n_dof`-sized), not first-order ones -- so a fair
"does structure-preservation actually save states" comparison against
`KrylovROM` is `SOARROM(k=k)` vs. `KrylovROM(k=2*k)` (same number of
first-order-equivalent states), not the same raw `k` value on both.

### Validation plan

1. **Moment-matching self-consistency**: a finite-difference/repeated-
   solve oracle (mirroring `test_krylov.py`'s own `_moments()` helper,
   generalized to the second-order transfer function) on a small,
   well-conditioned synthetic system -- confirm `SOARROM`'s reduced
   transfer function matches the full one's first `k` (or so) Taylor
   moments at `s0` to near machine precision, then diverges, the same
   defining signature `test_krylov.py` already established for
   `KrylovROM`.
2. **Cross-check against `krylov.py`'s own `KrylovROM`**: same real
   fea_engine damped-cantilever fixture and port `test_krylov.py`
   already uses; `SOARROM(k)` and `KrylovROM(k=2k)` (same total
   first-order-equivalent state count) should give comparably accurate
   frequency responses near `s0` -- both are matching the same
   underlying moments of the same transfer function, so close agreement
   here is itself a correctness check on `soar_basis()`, independent of
   the synthetic-oracle check above.
3. **`is_stable()`, checked not assumed**: same honest convention as
   `KrylovROM.is_stable()` -- no a priori guarantee is claimed, the
   reduced quadratic pencil's own poles are what get checked.
4. **The actual motivating question -- passivity**: `passivity.py`'s
   own closed-form theorem (module docstring) proves ANY second-order
   Galerkin projection (`M_r=Phi^T M Phi`, ...) of an SPD-`M`/PSD-`C`/
   SPD-`K` system preserves passivity, for ANY basis -- `soar_basis()`'s
   `V` is exactly such a basis, so `SOARROM` should ALSO provably
   preserve passivity, unlike `KrylovROM` (first-order, no such
   guarantee -- `test_passivity.py` already found a first-order
   reduction method violating it in practice). Checked directly via
   `passivity.py`'s own `velocity_transfer_function()`/`is_passive()`
   machinery, at multiple `k` on the real fixture, including `k` values
   in the range where `test_krylov.py`'s own `is_stable()` sweep showed
   `KrylovROM` behaving least favorably -- the sharpest, most literal
   test of this addendum's actual motivation.

### Outcome

Implemented (`soar_basis()`, `SOARROM`, in the new module `soar.py`)
and validated all four ways above, all passing (`tests/test_soar.py`,
7 tests; 251 tests package-wide).

A real, self-caught implementation bug is worth recording rather than
silently fixing (matching this project's established practice, e.g.
the omega-scaling bug in `nnm.py`'s Phase 6 addendum): the first
version of `soar_basis()` fed the ORTHONORMALIZED basis vectors back
into the two-term recurrence (`r_j = A1 v_{j-1} + A2 v_{j-2}`, using
`v`, not raw `r`), reasoning by (incorrect) analogy to ordinary
Arnoldi, where doing exactly that is harmless. It is NOT harmless here:
unlike a single-operator recurrence, orthogonalizing away a component
before feeding a two-generation recurrence forward discards real
content the next step needs. The telltale sign was that moment-matching
accuracy PLATEAUED after moment 1 regardless of how large k was made
-- a correct method matches MORE moments as k grows, so a hard plateau
independent of k is a bug signature, not a benign limitation. The fix
tracks the raw (pre-orthogonalization) recursion on its own, using
orthogonalization only to decide what is genuinely new for `V` (see
`soar.py`'s own module docstring for the full account) -- confirmed to
restore the expected "k basis vectors match k moments exactly" property
(`test_soar_basis_matches_moments_exactly_through_k_minus_1`,
`test_soarrom_transfer_function_matches_moments_at_nonzero_s0`, both at
`s0=0` and at a nonzero complex `s0`).

Check 2 (block port vs. `KrylovROM`, same total state count): on the
real fea_engine damped cantilever, a 3-DOF collocated block port, at
`k=2` (2 second-order states vs. `KrylovROM`'s 4 first-order states --
the same total count), `SOARROM`'s relative error was 0.135 against
`KrylovROM`'s 0.885 -- nearly an order of magnitude better at the same
state budget, the actual "avoids the first-order 2x blowup" claim this
addendum exists to demonstrate, not just assert.

Check 3, an honest, genuine SOAR fragility finding (not a bug -- traced
directly to a specific numerical mechanism, and reported plainly):
`soar_basis()`'s size PLATEAUS well below the requested `k` for a
SINGLE-DOF port on this beam (stuck at 1, regardless of whether `k=4`
or `k=24` is requested). The mechanism: with a single starting vector,
the raw recurrence is dominated by `A2 = -K0^-1 M`, which is exactly
the operator INVERSE POWER ITERATION on the generalized eigenproblem
`(K,M)` uses -- so successive raw generations converge geometrically
toward the dominant (first) mode shape, at a rate set by the
eigenvalue ratio (`(omega1/omega2)^2 ~ 1/39` on this beam,
`omega2/omega1 ~ 6.27`), and every new direction beyond the first is
numerically indistinguishable from what's already spanned within a
handful of steps. This is exactly the literature-documented reason
real SOAR/TOAR usage favors BLOCK (multi-column) starting vectors --
several simultaneous directions delay the same collapse -- and Check 2
above (a 3-column block) confirms that directly: no such plateau there
at the tested `k`.

Check 4, the actual motivation, holding cleanly: `SOARROM` stayed
passive (`passivity.py`'s `is_passive()`) at every tested `k` (1, 4, 8,
16 -- including the plateaued single-DOF-port case) and every tested
`s0` (0, and two nonzero points spanning the first resonance) on the
real fixture, and `M_r`/`K_r` stayed SPD, `C_r` stayed PSD, directly
confirming the theorem's premise rather than only its consequence.
`KrylovROM` also happened to stay passive on this specific fixture/port
at the orders tested here -- consistent with, not contradicting,
`test_passivity.py`'s own existing finding that first-order reduction
methods are not ALWAYS passive (its violation was at a different
method/order/port); the point of this addendum is the PROOF `SOARROM`
carries regardless of order or port, not that `KrylovROM` is
demonstrably worse on every fixture tried.

TOAR remains unbuilt, per the honest scope decision above -- a future
follow-up if a real target model's block-input case needs it (Section
2b's own "materially harder to implement and validate" framing, now
concretely borne out: getting even plain SOAR's core recurrence
correct took a real, self-caught bug fix along the way).

## Sources

- Bai, Z. (2002). ["Krylov subspace techniques for reduced-order modeling of large-scale dynamical systems."](https://www.cs.ucdavis.edu/~bai/publications/bai02.pdf)
- Bai, Z. & Su, Y. ["SOAR: A Second-order Arnoldi Method for the Solution of the Quadratic Eigenvalue Problem."](https://www.researchgate.net/publication/220656877_SOAR_A_Second-order_Arnoldi_Method_for_the_Solution_of_the_Quadratic_Eigenvalue_Problem) / ["Dimension Reduction of Large-Scale Second-Order Dynamical Systems via a Second-Order Arnoldi Method."](https://www.researchgate.net/publication/220412227_Dimension_Reduction_of_Large-Scale_Second-Order_Dynamical_Systems_via_a_Second-Order_Arnoldi_Method)
- Lu & Su et al. ["Stability analysis of the two-level orthogonal Arnoldi procedure"](https://web.cs.ucdavis.edu/~bai/publications/lusubai15.pdf) (TOAR, the numerically-stabilized successor to SOAR).
- Rozza, Huynh, Patera. ["Reduced Basis Approximation and a Posteriori Error Estimation for Affinely Parametrized Elliptic Coercive Partial Differential Equations."](https://link.springer.com/article/10.1007/s11831-008-9019-9)
- ["Reduced Basis Method for the Convected Helmholtz Equation"](https://arxiv.org/pdf/1506.02901) -- RB applied directly to a frequency-response/Helmholtz setting.
- ["Fast A Posteriori State Error Estimation for Reliable Frequency Sweeping in Microwave Circuits via the Reduced-Basis Method"](https://arxiv.org/pdf/2110.05925) -- greedy frequency-sweep sampling in practice.
- Lu, Y. et al. (2021). ["A Review of Model Order Reduction Methods for Large-Scale Structure Systems."](https://onlinelibrary.wiley.com/doi/10.1155/2021/6631180) Shock and Vibration.
- pyMOR documentation: [`pymor.models.transfer_function`](https://docs.pymor.org/main/autoapi/pymor/models/transfer_function/index.html), [`pymor.models.iosys`](https://docs.pymor.org/latest/autoapi/pymor/models/iosys/index.html) -- API precedent for `SecondOrderModel`/`TransferFunction` separation.
