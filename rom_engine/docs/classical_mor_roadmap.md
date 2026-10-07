# Classical structural-dynamics / systems-and-control MOR methods -- roadmap

Status: FULLY IMPLEMENTED (Phases 0-4 -- see Section 7, including the
originally-optional Phase 3, two-sided/Petrov-Galerkin Krylov), PLUS
Phase 5a (Sections 8-9, singular perturbation approximation --
implemented and validated, with the hoped-for DC-gain improvement over
ordinary BT actually confirmed empirically, not just theoretically
expected), Phase 5b (Sections 10-11, frequency-weighted BT --
implemented and validated, with the trade-off it makes (in-band gain,
out-of-band cost, no unconditional two-sided stability guarantee)
measured concretely on both sides rather than only cited), and Phase
5c (Sections 12-13, passivity preservation -- a genuine scope
adjustment from "positive-real balanced truncation" to a closed-form
theorem + diagnostic, made explicit and reasoned through in Section
12 rather than silently substituted; the diagnostic then found a real
passivity violation in an existing method, Section 13), and Phase 5d
(Sections 14-15, optimal Hankel norm approximation / AAK-Glover theory
-- the roadmap's LAST explicitly-named Phase 5 item, now closing this
roadmap out entirely; implemented in a new module, `hankel_norm.py`,
after a genuine derivation gap in the construction's own synthesis was
validated numerically before being trusted with production code, and a
real numerical fragility this validation surfaced -- unreliable when
`sigma_(r+1)` is very small relative to the largest Hankel singular
value -- is guarded by an always-on empirical self-check rather than an
a priori threshold, since no simple a priori quantity tried predicted
it reliably; the core AAK optimality claim was then verified directly
against `BalancedTruncationROM` on real data, Section 15). Implemented
in `src/rom_engine/state_space.py`, `mode_correction.py`, `krylov.py`,
`balanced_truncation.py`, validated in
`tests/test_state_space.py`/`test_mode_correction.py`/`test_krylov.py`/
`test_balanced_truncation.py` against real fea_engine models throughout
(see each test file's own docstring for what was checked, including two
real numerical findings from validation, both now documented in the
relevant module's own docstring rather than only here: (1) applying
balanced truncation directly to a raw, finely-meshed FE model's
state-space is unreliable because of the FE model's inherently huge
eigenvalue spread -- fixed by modally pre-truncating first
(balanced_truncation.py); (2) two-sided Krylov's improvement over
one-sided at the same reduced order, while proved exactly on a small
well-conditioned synthetic system, varies noticeably WITH k on a real
beam fixture -- large at some k, negligible at others -- rather than
being a uniform "always roughly halves the error" guarantee
(krylov.py/test_krylov.py)). Original design document below, preserved
as written.

Target: `rom_engine` module
additions implementing the four methods from Besselink, Tabak, Lutowska,
van de Wouw, Nijmeijer, Rixen, Hochstenbach & Schilders, *"A comparison of
model reduction techniques from structural dynamics, numerical mathematics
and systems and control"*, J. Sound Vib. 332 (2013), that a prior review of
that paper (this conversation) identified as reviewed-and-numerically-applied
in the paper's own worked example but not yet present in `rom_engine`: mode
acceleration, modal truncation augmentation, Krylov-subspace moment
matching, and balanced truncation.

## 1. The problem

A previous pass in this conversation read the paper in full and mapped its
methods against what `rom_engine` already does. The upshot:

- **Mode displacement (MD)** -- the paper's structural-dynamics baseline,
  `x ~= Phi*eta` on an undamped modal basis -- is **already achievable
  today** by composing two existing pieces exactly as they already exist:
  `fea_engine`'s `solve_modal()` eigenvectors fed straight into
  `GalerkinROM(modes).reduce_system(K, M=M)`. This was verified live to
  machine precision (`M_r` ~= I, `K_r` diagonal matching `(2*pi*f)^2`). No
  new code needed.
- **Mode acceleration (MA)** and **modal truncation augmentation (MTA)**
  -- cheap, closed-form corrections to MD that the paper *reviews* (Section
  2, eqs. 13-20) but does not itself run in the paper's own numerical
  example -- are genuinely absent from `rom_engine`, but are a small,
  low-risk addition once written down clearly.
- **Krylov-subspace moment matching (MM)** and **balanced truncation (BT)**
  -- the paper's numerical-mathematics and systems-and-control
  representatives, both *numerically applied* in the paper's own 8730-DOF
  benchmark and compared head-to-head against MD -- are genuine, structural
  gaps. Neither has an analog anywhere in `rom_engine` today: both require
  an **input/output port** view of the model (an explicit map from a small
  number of inputs `u` to the load vector, and from the state to a small
  number of outputs `y`) that no existing module needs, because every
  existing `rom_engine` module (`pod`, `galerkin`, `affine`, `frequency`)
  works directly with a full load vector `F` and a full-order response `x`,
  never a compressed input/output pair.

This document is the roadmap for closing the MA/MTA/MM/BT gap, in the same
research-then-design style as `docs/frequency_domain_rom_roadmap.md`. No
code is proposed to be written yet; this is the plan to review before that
starts.

## 2. What the literature actually does

### 2a. Mode acceleration and modal truncation augmentation (paper eqs. 13-20)

Given an undamped modal basis `Phi` (`n_dof x k`) and a *static* load `F`,
plain mode displacement approximates `x ~= Phi*eta` where
`eta = Kr^-1 * Phi^T F` (reduced static solve). This throws away the static
contribution of every mode *outside* the kept set, which for a smooth,
low-frequency-dominated load is often the biggest source of truncation
error -- physically, the "leftover" load that isn't well represented by the
kept mode shapes still deflects the structure statically, it just doesn't
oscillate at any of the kept frequencies.

Mode acceleration's fix is a closed-form static correction:

```
q_cor = K^-1 F - Phi * (Phi^T K Phi)^-1 * Phi^T F        (eq. 13-16)
x_MA  = Phi * eta + q_cor
```

`K^-1 F` is one full-order static solve (exactly the calculation
`GalerkinROM` already knows how to reduce, just evaluated at full order
here) -- the only new cost, and it is a **one-time** cost per load pattern,
not per query.

Modal truncation augmentation reuses `q_cor` differently: instead of adding
it as a post-hoc correction, fold it *into the basis itself*,
`Psi = [Phi, q_cor]` (eq. 20), then project and solve exactly as with any
other basis. This means MTA needs **no new solver logic at all** -- it is a
basis-construction utility that hands its output straight to the existing
`GalerkinROM`.

### 2b. Krylov-subspace moment matching (paper eqs. 21-36)

The paper's transfer function is `H(s) = c^T (s^2*M + s*C + K)^-1 * b` (or,
after casting to first-order form, `H(s) = c^T(sE - A)^-1 b`). Moment
matching picks a reduced basis `V` such that the reduced transfer function
`H_r(s)` matches the Taylor series of `H(s)` around an expansion point `s0`
to as many terms ("moments") as `dim(V)` allows, without ever forming the
Taylor series explicitly -- the Krylov subspace
`K_k(A^-1 E, A^-1 b) = span{r, A^-1 E r, (A^-1 E)^2 r, ...}` (`r = A^-1 b`)
does this automatically (a classical result: an orthonormal basis of this
subspace, by construction, matches the first `k` moments of `H(s)` at
`s0`).

Two standard ways to build that basis, both confirmed by a literature
search this session as the current standard practice:

- **One-sided Arnoldi**: build an orthonormal basis of the single Krylov
  subspace above via modified Gram-Schmidt (numerically the safe default
  over classical Gram-Schmidt); project with `V` on both sides
  (`Ar = V^T A V`, a Galerkin projection, same pattern as every other
  module in this package). Matches `k` moments with `k` basis vectors.
  This is the simplest, most robust starting point, and is what the
  paper's own eqs. 29-31 describe.
- **Two-sided Lanczos**: build *two* Krylov subspaces (one from `b`, one
  from `c`) and biorthogonalize them, giving a non-Galerkin (Petrov-Galerkin)
  projection that matches `2k` moments from the same `k`-dimensional basis
  -- twice the accuracy per reduced dimension, at roughly double the
  build cost and a real numerical fragility: biorthogonalization can
  **break down** (a zero or near-zero inner product between the two
  spaces) in a way one-sided Arnoldi cannot. Worth having eventually, not
  worth risking as the first implementation.
- **Second-order-structure-preserving Krylov (SOAR/TOAR)** -- building the
  Krylov subspace directly in the `(M, C, K)` second-order pencil instead
  of first collapsing to a `2N`-state first-order companion form -- is
  already flagged as a deferred, later-phase option in
  `docs/frequency_domain_rom_roadmap.md` Section 2b (citing Bai & Su 2005).
  That deferral is reaffirmed here: the paper's own method is the plain
  first-order Arnoldi/Lanczos approach (its eqs. 65-70 explicitly go
  through the first-order form), so matching the paper is *simpler*, not
  harder, than the structure-preserving alternative -- there is no reason
  to reach for SOAR/TOAR just to reproduce this paper's method.

One more finding from the paper worth preserving here explicitly, because
it changes what "done" looks like for this module: **moment matching gives
no a priori guarantee of stability**. The paper's own benchmark found its
`k=20` MM reduced model unstable on a posteriori eigenvalue inspection, even
though the full-order model is passive/stable and MD and BT both remained
stable at the same order. This has to be a *documented, tested-for*
property of `KrylovROM`, not something the implementation tries to
silently prevent -- it would be dishonest to claim MM is stable when the
paper's own numerical example demonstrates that it structurally isn't
guaranteed to be.

### 2c. Balanced truncation (paper eqs. 37-58)

BT works on the first-order state-space form `x' = A x + B u`, `y = Cout x`
(no throughput term `D` needed for this class of problem). It solves two
continuous-time Lyapunov equations,

```
A P + P A^T + B B^T = 0      (controllability Gramian P)
A^T Q + Q A + Cout^T Cout = 0  (observability Gramian Q)
```

finds a balancing transformation `T` under which the transformed Gramians
are equal and diagonal (`hat_P = hat_Q = diag(sigma_1 >= sigma_2 >= ...)`,
the **Hankel singular values** -- a coordinate-free measure of each
internal state's simultaneous controllability+observability, i.e. how much
it actually matters for the input-output behavior specifically), and
truncates to the `r` states with the largest `sigma_i`. Unlike MD or MM,
this explicitly uses *where the inputs and outputs are*, which is exactly
why the paper finds BT is the one method that stays accurate when far from
the low-frequency band the modal/Krylov methods are tuned to.

Two implementation facts from a literature search this session, both
important for getting this right the first time rather than needing a
later numerical-stability rewrite:

- `scipy.linalg.solve_continuous_lyapunov` (Bartels-Stewart via LAPACK
  `?TRSYL`) solves exactly the Lyapunov equations above, for a general
  (non-symmetric) dense `A` -- directly usable, no new dependency.
- The numerically preferred way to get the balancing transform is **not**
  to eigendecompose the Gramian product `P @ Q` directly (ill-conditioned
  when Hankel singular values span a wide range, which the paper's own
  benchmark shows they do). The standard "square-root method" instead
  Cholesky-factors each Gramian (`P = Lp Lp^T`, `Q = Lq Lq^T`), forms the
  small SVD `Lp^T Lq = U Sigma V^T`, and builds the balancing transform
  from `Lp`, `Lq`, `U`, `Sigma`, `V` directly -- this is the approach to
  implement.
- Scipy's dense Lyapunov solver is a real ceiling: it does not scale past
  roughly `O(10^3)` states, and does not support the generalized
  (`E`-form) Lyapunov equation directly. The paper itself notes BT's
  practical ceiling is about this same order of magnitude for the same
  reason (every large-scale BT method in the literature -- ADI, sign
  function, low-rank Krylov-based Lyapunov solvers, which is what pyMOR
  binds Slycot/Py-M.E.S.S. for -- exists specifically to work around this).
  This matches, rather than conflicts with, this project's existing
  "robust and correct over large-scale" posture in `frequency.py`/`scm.py`,
  so scipy's dense solver is the right initial (and likely only-needed)
  choice, not a stopgap.
- Because scipy's Lyapunov solver wants the explicit `A`-only first-order
  form (no `E`), and because BT already pays for a dense Lyapunov solve at
  the same `O(n^3)` order as a dense `M`-inversion, there is no separate
  numerical-conditioning reason to avoid inverting `M` for BT specifically
  -- unlike for Krylov (2b), where avoiding the `M`-inversion is worth it.

BT extensions the paper mentions but does not itself run numerically
(frequency-weighted BT, singular perturbation approximation, coprime-
factorization BT for unstable systems, passivity-preserving BT, optimal
Hankel norm approximation) are explicitly **out of scope** for the phases
below, listed only as a possible later, optional phase -- consistent with
how this project has scoped every other roadmap so far.

## 3. Implementation-pattern reference

pyMOR (already used as the API precedent in `frequency_domain_rom_roadmap.md`,
still not a dependency to add here) implements BT and Krylov/IRKA-style
reduction as separate "reductor" classes that operate on its own LTI system
object and expose a single `.reduce(r)` call returning a reduced model of
the requested order. That shape -- a small class wrapping a constructed
state-space system, with a `from_...` convenience constructor and a
`.reduce`/query method -- is the same shape `rom_engine.frequency.FrequencyROM`
already uses in this codebase (`FrequencyROM.from_MCK(M, K, basis, C=...,
rayleigh=...)` builds the affine decomposition + Galerkin projection
together; `.frequency_response(omega_array, F)` is the online query). The
proposed `KrylovROM` and `BalancedTruncationROM` classes below mirror that
exact convention rather than inventing a new one.

## 4. Concrete architectural fit

Re-reading `galerkin.py` directly (the actual current source, not a
paraphrase) confirms `GalerkinROM.__init__` already accepts either a plain
`ndarray` or anything with a `.V` attribute as `basis` -- so both MA's
correction vector and MTA's augmented basis need only produce a plain
`(n_dof, k)` array to be immediately usable by the *existing*
`GalerkinROM(basis).reduce_system(K, M=M, F=F)` code path, unmodified. This
confirms Section 2a's claim above: MA/MTA need **no new solver**, only
basis/vector-construction utilities, matching this project's convention of
adding new small functions/modules rather than editing existing, already-
tested ones.

`frequency.py`'s `FrequencyROM.from_MCK` classmethod is the concrete
pattern for `KrylovROM`/`BalancedTruncationROM`'s own constructors (Section
5) -- important to also note what `FrequencyROM` deliberately does *not*
do: it stays entirely in second-order `(M, C, K)` form and never builds a
first-order state-space system. That is a real, load-bearing distinction
already flagged in this conversation's own history and reconfirmed by this
re-read: `FrequencyROM` (already implemented) and `KrylovROM`/
`BalancedTruncationROM` (proposed here) solve genuinely different problems
that happen to look superficially similar (both do "fast frequency
queries"), and must not be conflated or merged.

`tests/fea_fixtures.py`'s `damped_cantilever_beam_system()` already returns
`K`, `M`, `C` (Rayleigh damping, built through `fea_engine`'s own
`RayleighDamping` + `assemble_damping()`, not hand-rolled) plus a `sys`
object whose `solve_harmonic()`/`solve_frequency_sweep()` are the existing,
already-validated full-order ground truth this new work should be checked
against -- exactly the same fixture `test_frequency.py` already uses. The
one genuine gap at the fixture level: no existing fixture defines an
explicit *input/output port* (a `B` column selecting which DOF(s) a unit
input load enters at, and a `Cout` row selecting which DOF(s) are read out
as outputs) -- every existing fixture and every existing module works with
a full load vector and full response instead. Building a small
`B`/`Cout` pair (e.g. unit tip force in, tip transverse displacement out,
echoing the paper's own single-input/single-output reading of its
benchmark) is a one-time, small addition to `fea_fixtures.py`, not a new
module.

That input/output-port need is also the concrete reason a small new shared
module is proposed below (`state_space.py`) rather than folding first-order
conversion directly into `krylov.py` and `balanced_truncation.py`
separately: both modules need the *same* `(M, C, K, B, Cout) -> first-order`
conversion, and Section 2b/2c above already established they need it in two
different forms (Krylov: `E`-form, avoiding an `M`-inversion; BT: explicit
`A`-only form, since BT pays for an equivalent-cost dense solve anyway) --
one shared, well-tested conversion utility avoids two independent, subtly
different reimplementations of the same linear algebra.

## 5. Proposed API

```python
# rom_engine/state_space.py -- new shared module
#
# Converts a second-order structural system (M, C, K) with an explicit
# input map B (n_dof x n_in) and output map Cout (n_out x n_dof) into
# first-order state-space form, in the two forms Section 2b/2c need:
#
#   form="E":  [E, A, B, Cout]  s.t.  E x' = A x + B u,  y = Cout x
#              E = [[I, 0], [0, M]], A = [[0, I], [-K, -C]]   (eq. 68-70)
#              -- no M-inversion, used by krylov.py
#   form="A":  [A, B, Cout]     s.t.  x' = A x + B u,  y = Cout x
#              A = [[0, I], [-M^-1 K, -M^-1 C]]                (eq. 65-67)
#              -- one dense M-solve, used by balanced_truncation.py
#
# state = [x; x'] (n_dof positions stacked on n_dof velocities), so both
# forms are (2*n_dof, 2*n_dof) -- this doubling is exactly why the paper
# reduces to k=20 (not 10) states for its first-order-form comparisons.

def to_state_space(M, K, C=None, B=None, Cout=None, form="E"): ...
    # returns a small StateSpaceSystem(E, A, B, Cout) dataclass
    # (E is None when form="A")


# rom_engine/mode_correction.py -- new module, Phase 1
#
# Mode acceleration (eq. 13-16) and modal truncation augmentation (eq. 20).
# Deliberately produces plain ndarrays/vectors that feed straight into the
# EXISTING GalerkinROM -- no new solver logic.

def mode_acceleration_correction(K, basis, F): ...
    # q_cor = K^-1 F - V (V^T K V)^-1 V^T F  -- one full static solve

def mode_acceleration_response(basis, eta, q_cor): ...
    # x = V @ eta + q_cor

def augmented_basis(basis, q_cor, M=None): ...
    # Psi = [V, q_cor], optionally M-orthogonalized (Gram-Schmidt against
    # M's inner product) so the augmented Mr stays well-conditioned even
    # when q_cor is nearly in the span of V already


# rom_engine/krylov.py -- new module, Phase 2 (one-sided), Phase 3 (two-sided)

def arnoldi_basis(A, E, b, k): ...
    # modified Gram-Schmidt Krylov basis of K_k(A^-1 E, A^-1 b)

class KrylovROM:
    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, s0=0.0, k=10, two_sided=False): ...
    def frequency_response(self, omega_array): ...
    def is_stable(self): ...
        # eigenvalues of the REDUCED A_r -- exposed explicitly because
        # Section 2b established this is NOT guaranteed and must be
        # checked, not assumed


# rom_engine/balanced_truncation.py -- new module, Phase 4

def controllability_gramian(A, B): ...  # scipy solve_continuous_lyapunov
def observability_gramian(A, Cout): ...  # scipy solve_continuous_lyapunov
def hankel_singular_values(A, B, Cout): ...  # square-root/Cholesky method

class BalancedTruncationROM:
    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, r=10): ...
    def frequency_response(self, omega_array): ...
    def h_infinity_error_bound(self): ...
        # 2 * sum(discarded Hankel singular values)  (eq. 58, a priori)
    def is_stable(self): ...
        # a structural theorem for BT (unlike KrylovROM) -- should always
        # be True; kept as a regression guard, not a design choice
```

## 6. Validation plan (same standard as every other module in this project)

All of it against `fea_fixtures.damped_cantilever_beam_system()` (extended
with a `B`/`Cout` tip-force-in/tip-displacement-out pair), checked against
that fixture's own `sys.solve_harmonic()`/`solve_frequency_sweep()` as
ground truth -- never a synthetic-only matrix.

- **Mode acceleration accuracy**: for a static tip load, `mode_acceleration_
  response` should give strictly lower error against `FESystem.solve_
  static()` than plain `GalerkinROM.solve_static()` at the *same* `n_modes`
  -- the whole point of the method, and directly checkable.
- **Modal truncation augmentation convergence**: `GalerkinROM(augmented_
  basis(...))` should reach a given static-solve accuracy target at fewer
  modes than plain `GalerkinROM(modes)` -- reproducing the paper's own
  qualitative comparison (its Fig.-type "fewer modes needed" argument for
  MTA over MD).
- **KrylovROM near-s0 accuracy, far-from-s0 degradation**: `frequency_
  response` should match the full-order sweep very closely near
  `omega = s0`, with error growing away from it -- moment matching's
  defining, testable signature (this is not a bug to fix, it is the method
  working as designed).
- **BalancedTruncationROM accuracy across the whole sweep, and versus
  KrylovROM at high frequency**: BT should stay accurate across the full
  swept range (not just near one point) and should outperform `KrylovROM`
  at the *same reduced order* away from `s0` -- reproducing the paper's own
  central head-to-head finding.
- **`BalancedTruncationROM.is_stable()` always True**: a structural
  regression guard, since BT's stability-preservation is a theorem, not an
  empirical property.
- **`KrylovROM.is_stable()` may be False at larger `k`**: attempt to
  reproduce the paper's own found instability at a comparable relative
  order; if `rom_engine`'s (much smaller) toy fixture doesn't reproduce it
  at any tested `k`, the test should say so honestly (assert the *method*
  under test, i.e. that `is_stable()` runs and returns a real answer,
  without asserting a specific stable/unstable outcome that this project
  cannot actually guarantee at this model size) rather than force a
  misleading pass.
- **`h_infinity_error_bound()` is a real bound, not just a number**: check
  it numerically dominates the actual sup-norm error between the full and
  reduced frequency responses over a fine sweep, for at least one
  truncation order.

New test files: `test_mode_correction.py`, `test_krylov.py`,
`test_balanced_truncation.py`, `test_state_space.py` -- matching the
existing `test_frequency.py`/`test_affine.py` naming convention.

## 7. Phased roadmap

- **Phase 0 -- `state_space.py`.** Both conversion forms (`E` and `A`).
  Tested first against a hand-computed small (2-3 DOF) toy system with a
  known analytical first-order form, then against
  `damped_cantilever_beam_system()` (check that the state-space `A`'s
  eigenvalues reproduce the fixture's own known natural frequencies and
  Rayleigh damping ratios). Everything below depends on this.
- **Phase 1 -- `mode_correction.py`** (mode acceleration + modal truncation
  augmentation). Lowest risk and lowest effort of the four: pure
  composition with the existing, already-tested `GalerkinROM`, no Lyapunov
  solves, no Krylov basis-building, no first-order form needed at all
  (MA/MTA stay entirely in second-order form). Could ship and be useful on
  its own even before Phases 2-4 exist.
- **Phase 2 -- `krylov.py`, one-sided Arnoldi only.** Fixed real expansion
  point `s0` (matching the paper's own `s0=0` choice), modified
  Gram-Schmidt orthogonalization. A reasonable stopping point on its own if
  scope needs to be kept small -- reproduces the paper's MM method fully as
  numerically applied in its worked example.
- **Phase 3 -- `krylov.py`, two-sided (Petrov-Galerkin) extension**
  (originally optional; now IMPLEMENTED). `two_sided_arnoldi_bases()`
  builds a second Krylov basis from the dual system and biorthogonalizes
  it against the first (`W^H E V = I`), giving an oblique projection
  that matches `2k` moments instead of `k` from the same reduced order
  -- proved exactly (moment-by-moment) on a small, well-conditioned
  synthetic system in `test_krylov.py`, since the real beam fixture's
  own wide eigenvalue spread would swamp the effect with unrelated
  numerical noise (see balanced_truncation.py's own note on this same
  issue). The flagged fragility is real: `two_sided_arnoldi_bases()`
  raises `ValueError` when the input/output Krylov subspaces are too
  close to degenerate with each other to biorthogonalize reliably,
  rather than returning a numerically meaningless basis. On the real
  fea_engine fixture, `KrylovROM.from_MCK(..., two_sided=True)`
  measurably outperforms one-sided at the same reduced order for SOME
  `k` (up to ~13x lower mean error at `k=5` in this project's own test)
  but not uniformly for every `k` tested -- recorded honestly in
  `test_krylov.py`'s own docstring rather than glossed over.
- **Phase 4 -- `balanced_truncation.py`.** The highest-effort, highest-risk
  item (two Lyapunov solves, square-root balancing transform, Hankel
  singular values, a priori error bound) -- deliberately sequenced last so
  Phases 0-3 are already validated and the shared `state_space.py`
  conversion is already trusted before building on top of it.
- **Phase 5 -- explicitly deferred, optional, not part of this plan unless
  separately requested**: frequency-weighted BT, singular perturbation
  approximation, coprime-factorization BT for unstable systems,
  passivity-preserving BT, optimal Hankel norm approximation, and
  second-order-structure-preserving (SOAR/TOAR) Krylov. All are reviewed in
  the paper but not numerically applied there either, and all add real
  implementation complexity beyond what reproducing the paper's own worked
  example requires.

Phases 1-4 together close every remaining gap identified in this
conversation's earlier review of the paper against `rom_engine`'s current
state: after them, all three of the paper's numerically-applied families
(MD, MM, BT) plus both of its reviewed-but-unapplied cheap extensions (MA,
MTA) exist in `rom_engine`, each validated the same way every other module
in this package is -- against a real `fea_engine` model's own real solve,
not a synthetic stand-in.

## 8. Phase 5a -- singular perturbation approximation (SPA)

Of Phase 5's five deferred items, this document picks up singular
perturbation approximation first: it needs no new theory beyond what
`balanced_truncation.py` already computes (the SAME balancing
transform `T`/`Tinv` and Hankel singular values `hankel_singular_
values(..., return_transform=True)` already returns), unlike
frequency-weighted BT (needs weight-filter state-space realizations
and an augmented system), coprime-factorization/passivity-preserving
BT (needs extra Riccati/ARE solves and normalized-coprime-factor
theory), or optimal Hankel norm approximation (needs Adamjan-Arov-
Krein all-pass-completion theory) -- all genuinely new machinery, none
of it needed here.

**The idea (Liu & Anderson 1989, "Singular perturbation approximation
of balanced systems"; also called "residualization" -- see e.g.
Antoulas, *Approximation of Large-Scale Dynamical Systems*, Ch. 9)**:
ordinary balanced truncation DISCARDS the truncated states `x2`
outright (assumes `x2 = 0` identically), which is exact at `s ->
infinity` (the reduced model's `D` term is untouched) but generally
WRONG at `s = 0` (DC/steady-state) -- exactly the opposite regime a
structural analyst often cares most about (static or near-static
loads). SPA instead assumes `x2` reaches QUASI-STEADY-STATE instantly
(`x2' = 0`, not `x2 = 0`) and solves the resulting algebraic
constraint for `x2` in terms of `x1` and `u`, substituting it back in.
In a balanced realization partitioned into kept (`1`) and discarded
(`2`) blocks,

```
A = [[A11, A12], [A21, A22]],  B = [[B1], [B2]],  Cout = [C1, C2],  D = 0
```

(`D = 0` always for this package's structural systems -- no direct
input-to-output feedthrough term, per `state_space.py`), setting
`x2' = 0 = A21 x1 + A22 x2 + B2 u` and solving for
`x2 = -A22^-1 (A21 x1 + B2 u)` (valid: `A22` is a genuine THEOREM-
guaranteed-stable, hence invertible, block of a balanced realization
of a stable system -- Liu & Anderson 1989), then substituting into the
`x1'` and `y` equations gives the reduced (Schur-complement) system

```
A_r = A11 - A12 @ A22^-1 @ A21
B_r = B1  - A12 @ A22^-1 @ B2
C_r = C1  - C2  @ A22^-1 @ A21
D_r =  0  - C2  @ A22^-1 @ B2      (generally NONZERO, unlike ordinary BT's D_r = D = 0)
```

This is the exact state-space analogue of what `mode_correction.py`
already does for a plain modal basis: `mode_acceleration_correction()`
recovers the STATIC contribution of the modes a truncated modal basis
throws away via one extra static solve; SPA recovers the analogous
static (DC) contribution of the Hankel-singular-value states BT
truncates, via one small (`(n-r) x (n-r)`) linear solve instead of a
full-order one -- the same idea, one level up in this package's own
"systems and control" family.

**Two classical guarantees this construction carries over from
ordinary BT, both to be checked directly rather than assumed** (Liu &
Anderson 1989 prove both):

1. **SPA reduced model is also always stable**, given the same
   stable-full-order-model hypothesis `BalancedTruncationROM.is_
   stable()`'s docstring already documents as a theorem, not an
   empirical property, for ordinary BT.
2. **SPA shares the SAME a priori H-infinity error bound** as ordinary
   BT: `2 * sum(discarded Hankel singular values)` -- i.e. neither
   method is "more accurate" by this bound; the value they add is
   fixing WHERE (`s=0` vs. `s=infinity`) the reduced model is exact,
   not tightening the bound itself.

**API** (`balanced_truncation.py`, new sibling class -- NOT a new
module: SPA is tightly coupled to, and directly reuses without
recomputation, the SAME `hankel_singular_values(..., return_
transform=True)` balancing transform `BalancedTruncationROM` already
computes, matching the same file/module-boundary reasoning `krylov.py`
vs. `balanced_truncation.py` already follows -- same "systems and
control" family, genuinely different algorithm, so a new class, same
file, mirroring `BalancedTruncationROM`'s method names exactly for
direct side-by-side comparison, the same design choice `scm.py`/
`scm_lp.py` and `BalancedTruncationROM`/`KrylovROM` already made):

```python
class SingularPerturbationROM:
    def __init__(self, ss, hsv, T, Tinv, r): ...
        # A_r, B_r, Cout_r, D_r via the Schur-complement construction above
    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, r=10): ...   # same signature as BalancedTruncationROM.from_MCK
    def transfer_function(self, s): ...      # Cout_r (sI - A_r)^-1 B_r + D_r -- note the + D_r, absent from BalancedTruncationROM
    def frequency_response(self, omega_array): ...
    def h_infinity_error_bound(self): ...    # SAME formula as BalancedTruncationROM's
    def is_stable(self): ...                 # SAME theorem-backed regression guard
```

**Validation plan**: reuse `test_balanced_truncation.py`'s exact
fixture and `B`/`Cout` port. Beyond mirroring that file's existing BT
checks (accuracy across a sweep, `is_stable()`, `h_infinity_error_
bound()` actually dominating the measured error), the checks that are
THE point of building this:

- **DC-gain exactness**: `SingularPerturbationROM`'s `transfer_
  function(0)` (or a query at a very small `s`, since `s=0` alone
  is a single point that doesn't exercise the frequency sweep
  machinery) should match the FULL-ORDER system's own DC gain
  `Cout @ (-A)^-1 @ B` (computed independently, not through this
  module) to numerical precision -- the concrete, checkable version
  of "SPA fixes BT's DC-gain gap," not just a claim.
- **Ordinary BT's DC-gain error, for contrast**: measure `Balanced
  TruncationROM`'s own DC-gain error at the SAME `r` on the SAME
  model, to make the comparison concrete rather than asserted (BT's
  error should be visibly larger -- if it happens not to be at some
  `r`, that is reported honestly, not forced).
- **Shared H-infinity bound holds for SPA too**: same check `test_
  balanced_truncation.py` already runs for ordinary BT, run again for
  `SingularPerturbationROM` at the same `r`.
- **Both remain stable at the same `r` values tested.**

New test file: `test_singular_perturbation.py` (or added directly to
`test_balanced_truncation.py`, since it exercises the SAME module and
fixture -- decide when writing, based on how large the balanced-
truncation test file already is).

## 9. Phase 5a outcome: implemented, and the hoped-for result actually held

`balanced_truncation.py`'s `SingularPerturbationROM` was implemented
exactly as Section 8 specifies, and validated in the new
`tests/test_singular_perturbation.py` against the SAME fixture,
pre-reduction pattern, and tip-force-in/tip-displacement-out port
`test_balanced_truncation.py` already uses -- so `BalancedTruncationROM`
and `SingularPerturbationROM` are compared on identical footing.
Unlike Phase 4e (`scm_lp.py`), where the hoped-for improvement did NOT
materialize on this project's real fixture, Phase 5a's core claim DID
hold, measured directly rather than assumed:

- **DC-gain exactness, concretely measured**: at `r=10`, SPA's DC gain
  matched the full-order system's own DC gain to a relative error of
  `1.7e-13` (numerical precision -- an algebraic identity, not an
  approximation), while ordinary `BalancedTruncationROM`'s DC gain, at
  the SAME `r` on the SAME model, was off by `3.4e-5` -- roughly 8
  orders of magnitude worse. This is the concrete version of "SPA
  fixes BT's DC-gain gap," not just the textbook claim.
- **The shared H-infinity bound holds for SPA too**: bound
  `2.3e-9`, observed sup-norm absolute error `4.1e-12` over a 200-point
  sweep at `r=10` -- comfortably inside, as the shared-theorem claim
  predicts.
- **SPA remains stable at every tested `r` (4, 8, 10, 12)**, and stays
  accurate across the WHOLE swept range too (`max relative error
  1.7e-4` at `r=10`, comparable to ordinary BT's own global accuracy
  at the same `r`) -- fixing the DC-specific gap did not cost the
  global-accuracy property that is BT's whole reason for existing in
  this package.

**Net assessment**: Phase 5a delivered exactly what it set out to,
with the improvement empirically confirmed rather than merely
theoretically expected. `SingularPerturbationROM` and
`BalancedTruncationROM` are both available side by side
(`from_MCK()`, `frequency_response()`, `h_infinity_error_bound()`,
`is_stable()` -- identical signatures throughout): prefer
`BalancedTruncationROM` when accuracy at high frequency /
`s -> infinity` matters most (its `D_r` matches the full order
system's `D` exactly, `= 0` here), prefer `SingularPerturbationROM`
when accuracy at DC / near-static loading matters most -- both share
the same a priori H-infinity bound and the same stability guarantee,
so the choice between them is about WHERE the reduced model is exact,
not which one is "better."

## 10. Phase 5b -- frequency-weighted balanced truncation

Of the three remaining Phase 5 items (frequency-weighted BT, coprime-
factorization/passivity-preserving BT, optimal Hankel norm
approximation), this document picks up frequency-weighted BT next: it
reuses the SAME square-root Lyapunov/SVD balancing machinery
`balanced_truncation.py` already has (feeding it different Gramians,
not a different algorithm), unlike coprime-factorization/passivity-
preserving BT (needs extra algebraic Riccati equation solves and
normalized-coprime-factor theory) or optimal Hankel norm approximation
(needs Adamjan-Arov-Krein all-pass-completion theory) -- both genuinely
new machinery, neither needed here.

**The idea (Enns 1984, "Model reduction with balanced realizations: An
error bound and a frequency weighted generalization")**: ordinary BT's
Hankel singular values rank each balanced state by how much it matters
for the input-output behavior UNIFORMLY across all frequencies. If a
user cares more about accuracy in a specific band (e.g. near a
resonance of interest, or over the low-frequency/quasi-static range),
weighting the controllability/observability Gramians by input/output
FILTERS `Wi(s)`/`Wo(s)` before balancing re-ranks the states by how
much they matter *for that band specifically*, at the cost of accuracy
elsewhere -- a direct trade, not a free improvement.

**Construction**: for the plant `(A, B, Cout)` and SISO weight
realizations `Wi = (Ai, Bi, Ci, Di)` (applied BEFORE the plant, on the
input side) and `Wo = (Ao, Bo, Co, Do)` (applied AFTER the plant, on
the output side; either may be omitted, degenerating that side back to
`Wi(s) = 1` / `Wo(s) = 1` exactly, i.e. ordinary unweighted BT on that
side), form two augmented ("cascade") systems and solve the ORDINARY
Lyapunov equations `controllability_gramian()`/`observability_
gramian()` already implement, on the augmented systems instead of the
plant alone:

```
# input-weighted (for the frequency-weighted controllability Gramian):
#   u -> Wi -> (feeds into the plant's B)
A_aug_c = [[A, B @ Ci], [0, Ai]]
B_aug_c = [[B @ Di], [Bi]]
Pw = controllability_gramian(A_aug_c, B_aug_c)[:n, :n]   # top-left (plant-state) block

# output-weighted (for the frequency-weighted observability Gramian):
#   plant's y -> Wo -> output
A_aug_o = [[A, 0], [Bo @ Cout, Ao]]
Cout_aug_o = [Do @ Cout, Co]
Qw = observability_gramian(A_aug_o, Cout_aug_o)[:n, :n]   # top-left (plant-state) block
```

`Pw`, `Qw` (restricted back to the plant's own `n` states) are then
fed into the EXACT SAME square-root balancing routine (Cholesky/eigen
square roots + SVD of the cross-product) `hankel_singular_values()`
already implements -- refactored into a small shared helper
(`_balance_from_gramians(P, Q)`) so ordinary `hankel_singular_values()`
becomes the special case `Wi=Wo=None` calling the same helper with the
plain, unweighted Gramians, verified to reproduce IDENTICAL numbers to
the pre-refactor code (a regression check on the refactor itself, not
just the new feature).

**Convenience weight constructors** (`lowpass_weight(wc)`,
`bandpass_weight(omega_n, zeta=0.1)`), returning `(Aw, Bw, Cw, Dw)`
tuples in controllable canonical form, so a caller doesn't need to
hand-derive state-space realizations for the common cases (emphasize
everything below a cutoff; emphasize a band around a specific natural
frequency -- the more structurally relevant case, e.g. "make the
reduced model most accurate near this mode").

**An honest limitation this construction inherits from the
literature, not an implementation gap**: unlike ordinary BT or SPA,
Enns' TWO-SIDED (both `Wi` and `Wo` given) frequency-weighted BT has NO
unconditional stability-preservation theorem -- `Pw`/`Qw` restricted to
the plant's sub-block are not guaranteed positive semi-definite in
general, and the reduced model's stability is correspondingly not
guaranteed either (this is well documented in the literature going
back to Enns' own paper, not a property this project's implementation
introduces). ONE-SIDED weighting (only `Wi` or only `Wo` given) DOES
preserve stability in general (a real, separate result, not the same
as the general two-sided case). This means `FrequencyWeightedBalanced
TruncationROM.is_stable()` must be CHECKED per call, not assumed --
matching `krylov.KrylovROM.is_stable()`'s honesty convention, not
`BalancedTruncationROM`/`SingularPerturbationROM`'s theorem-backed
regression-guard convention. For the same reason, this class does NOT
expose `h_infinity_error_bound()` -- the classical `2 * sum(discarded
Hankel singular values)` a priori bound is a property of the
UNWEIGHTED Hankel singular values specifically; a general a priori
bound exists in the frequency-weighted literature but is more involved
and was not independently verified here, so rather than claim an
unverified bound, this class omits the method entirely -- the same
honest-omission choice `KrylovROM` already makes for the same reason.

**API** (`balanced_truncation.py`, same file -- another sibling class,
same reasoning as `SingularPerturbationROM`'s):

```python
def lowpass_weight(wc): ...      # (Aw, Bw, Cw, Dw), W(s) = wc / (s + wc)
def bandpass_weight(omega_n, zeta=0.1): ...
    # (Aw, Bw, Cw, Dw), W(s) = 2*zeta*omega_n*s / (s^2 + 2*zeta*omega_n*s + omega_n^2)

def frequency_weighted_gramians(A, B, Cout, Wi=None, Wo=None): ...
    # Pw, Qw -- the Enns construction above

class FrequencyWeightedBalancedTruncationROM:
    def __init__(self, ss, hsv, T, Tinv, r, Wi=None, Wo=None): ...
    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, r=10, Wi=None, Wo=None): ...
    def transfer_function(self, s): ...
    def frequency_response(self, omega_array): ...
    def is_stable(self): ...   # checked, NOT a theorem here -- see above
    # deliberately NO h_infinity_error_bound() -- see above
```

**Validation plan**: reuse `test_balanced_truncation.py`'s exact
fixture, pre-reduction pattern, and port. The checks that are THE
point of building this:

- **`Wi=Wo=None` reproduces ordinary `BalancedTruncationROM` exactly**
  (to numerical precision) -- the direct regression check that the
  `_balance_from_gramians()` refactor changed nothing about the
  existing, already-validated unweighted path.
- **A weighted model is measurably MORE accurate than ordinary BT, at
  the SAME reduced order `r`, inside the weighted band** -- e.g. an
  output `bandpass_weight()` centered near a chosen natural frequency,
  compared against ordinary `BalancedTruncationROM` at the same `r`,
  over a narrow sweep around that frequency. If the weighted model is
  NOT measurably better on this fixture, that is reported as a finding
  (per this project's established practice), not hidden.
- **...and correspondingly may be LESS accurate far from the weighted
  band** -- the flip side of the same trade, checked and reported
  either way, not assumed.
- **One-sided weighting remains stable** (checked across several `r`,
  matching `SingularPerturbationROM`'s regression-style check, but
  without claiming it as an unconditional theorem in the docstring).
- **Two-sided weighting's stability is checked, not asserted a
  specific way** -- mirrors `test_krylov.py`'s own honest handling of
  `KrylovROM.is_stable()`'s non-guarantee: run it, report what
  happened, don't force a directional assertion the literature doesn't
  back.

New test file: `test_frequency_weighted_bt.py`.

## 11. Phase 5b outcome: implemented, and the trade-off measured cleanly on both sides

`balanced_truncation.py`'s `frequency_weighted_gramians()`,
`frequency_weighted_hankel_singular_values()`,
`FrequencyWeightedBalancedTruncationROM`, and the `lowpass_weight()`/
`bandpass_weight()` convenience constructors were implemented exactly
as Section 10 specifies, including the `_balance_from_gramians()`
refactor of the existing (unweighted) balancing path. Validated in the
new `tests/test_frequency_weighted_bt.py`, on the SAME fixture,
pre-reduction pattern, and port `test_balanced_truncation.py`/`test_
singular_perturbation.py` already use:

- **The refactor changed nothing**: `Wi=Wo=None` reproduces ordinary
  `BalancedTruncationROM`'s Hankel singular values and frequency
  response exactly (to numerical precision) -- confirming
  `_balance_from_gramians()` is a genuine no-behavior-change extraction,
  not a rewrite.
- **The core trade-off, measured cleanly on BOTH sides at the SAME
  `(r=6, target=third natural frequency, bandpass zeta=0.2)`**: an
  output `bandpass_weight()` centered on the fixture's third natural
  frequency gives a reduced model `14.0x` MORE accurate than ordinary
  BT, at the same `r`, in a band around that frequency (max relative
  error `0.238` vs. `3.33` -- ordinary BT effectively fails to
  represent this mode at all at this `r`, while the weighted model
  stays under 25% error); the SAME weighted model is measurably LESS
  accurate than ordinary BT far from that band, near the dominant
  first mode (`1.9e-3` vs. `3.6e-4`, ordinary BT about `5x` better
  there) -- a genuine trade, not a free improvement, demonstrated in
  both directions on the same model rather than only the favorable
  one. (This specific `(r, zeta)` combination was chosen from a small
  parameter sweep to show the trade-off cleanly; the underlying effect
  -- weighting helps in-band, at some cost elsewhere -- held across
  most of the swept `(r, zeta)` combinations tried, not just this one,
  though not uniformly at every combination, consistent with this
  being a real trade rather than a universal win.)
- **One-sided weighting (input-only or output-only) remained stable at
  every tested `r` (4, 6, 8, 10)** -- consistent with the literature's
  one-sided stability result.
- **Two-sided weighting's stability, genuinely checked rather than
  assumed**: across `r in {4, 6, 8, 10, 12}` on this fixture, it came
  out stable at some values (`4, 10, 12`) and unstable at others
  (`6, 8`) -- exactly the "no unconditional guarantee" property the
  literature documents for Enns' two-sided construction, reproduced
  concretely rather than only cited.
- **`h_infinity_error_bound()` is confirmed NOT exposed** on this
  class, matching `KrylovROM`'s own honest omission for the same
  underlying reason (no independently-verified a priori bound here).

**Net assessment**: Phase 5b delivered a correctly implemented,
honestly scoped frequency-weighted BT -- including confirming, not just
asserting, both the real benefit (measurable in-band accuracy gain)
and the real costs (out-of-band accuracy loss; no unconditional
stability guarantee for the two-sided case) that come with it. Unlike
Phase 5a (where the hoped-for improvement held essentially for free)
and Phase 4e (where it didn't hold at all), Phase 5b's result is the
most nuanced of the three: a genuine, usable capability whose benefit
is real but conditional on the user actually wanting the trade it
makes.

## 12. Phase 5c -- passivity preservation (a genuine scope adjustment, made explicit)

Of the two remaining Phase 5 items, the paper (and the reduced-order-
modeling literature generally) groups "coprime-factorization BT" and
"passivity-preserving BT" together as one family, but they are
DIFFERENT guarantees requiring different machinery, and investigating
the passivity route surfaced a real, load-bearing derivation risk --
directly analogous to the complex-operator ambiguity Phase 4e
(`scm_lp.py`) hit for the natural-norm SCM -- worth recording here even
though the eventual implementation avoided it:

**The risk found, and why it was avoided rather than pushed through**:
the standard construction for passivity-preserving balanced truncation
(Positive-Real Balanced Truncation -- see e.g. Reis & Stykel 2010, or
Antoulas Ch. 6) balances a system against the extremal solutions of an
algebraic Riccati equation derived from the Positive Real Lemma / KYP
inequality. That Riccati equation requires `D + D^T` to be
INVERTIBLE, where `D` is the state-space realization's direct
feedthrough term -- but `state_space.py` gives every one of this
package's structural models `D = 0` (no direct input-to-output
feedthrough, by construction), making `D + D^T = 0` exactly singular.
The "strictly proper" special case this degenerates to does have a
literature treatment, but it replaces the Riccati equation with an
EXACT LINEAR constraint (`B^T X = C`, alongside a Lyapunov-type
inequality) that is not guaranteed to have a solution for an
arbitrary realization -- extending it correctly, in general, would
have been genuinely new derivation, not a routine application of a
standard formula, with real risk of an unverified or subtly wrong
result (the same category of risk Section 9/Phase 4e already flagged
and chose not to take).

**What made the alternative both SAFER and MORE DIRECTLY useful**: for
this package's own systems specifically -- a real, damped, collocated
force-input/velocity-output structural model -- the passivity
certificate turns out to be available in EXACT, CLOSED FORM, with no
Riccati equation at all. For `M q'' + C_damp q' + K q = B u`,
`y = B^T q'` (a velocity output collocated with the SAME pattern as
the input force `B`), the total mechanical energy
`E = 0.5 q'^T M q' + 0.5 q^T K q` satisfies, by direct differentiation
and using `K = K^T`:

```
dE/dt = -q'^T C_damp q' + y^T u   <=   y^T u        (since C_damp >= 0)
```

-- exactly the KYP/positive-real dissipation inequality, i.e. this
package's ALREADY-STANDING assumptions (`M`, `K` symmetric positive
definite; `C_damp` symmetric positive semi-definite -- real, physical
damping, already required throughout `balanced_truncation.py` and
`state_space.py`) are BY THEMSELVES a complete passivity proof for the
full-order model, no additional machinery needed.

**The stronger, genuinely new result this makes available**: the SAME
derivation applies UNCHANGED to a second-order GALERKIN-projected
system (`M_r = Phi^T M Phi`, `C_r = Phi^T C_damp Phi`,
`K_r = Phi^T K Phi`, `B_r = Phi^T B`, `y_r = B_r^T eta'`) for ANY
full-column-rank basis `Phi` -- congruence transformation preserves
positive-(semi)definiteness unconditionally (`Phi^T M Phi >= 0`
whenever `M >= 0`, for any `Phi`, a completely elementary fact), so
`M_r`, `K_r` stay SPD and `C_r` stays PSD, and the identical dissipation
inequality holds for the REDUCED system automatically. This means:
**any second-order Galerkin projection this package builds --
`galerkin.GalerkinROM`'s modal basis, or a `frequency.FrequencyROM`
built on top of it -- provably preserves passivity, for ANY basis, not
just a modal one**, a genuinely new, previously-unstated (though
latent) property of code this package already has, not a new
reduction algorithm.

**What is genuinely new here, then, is a DIAGNOSTIC, not a new ROM
class**: `passivity.py`'s `is_passive()`/`passivity_margin()` check
the FREQUENCY-DOMAIN form of the same fact -- a positive-real transfer
function satisfies `Re[G(i*omega)] >= 0` for every real `omega` (the
standard frequency-domain characterization of the KYP/positive-real
lemma) -- applicable to ANY of this package's frequency-response-
producing classes (`FrequencyROM`, `KrylovROM`,
`BalancedTruncationROM`, `SingularPerturbationROM`,
`FrequencyWeightedBalancedTruncationROM`), by converting their
EXISTING displacement transfer function to the velocity (mobility)
one via `H_vel(i*omega) = i*omega * H_disp(i*omega)` -- no new
state-space "velocity output" plumbing needed, since velocity is
exactly `d/dt` of displacement, i.e. multiplication by `i*omega` in
the frequency domain. This is what makes the diagnostic genuinely
useful beyond restating the closed-form theorem: it can be run against
`KrylovROM`/`BalancedTruncationROM`/`FrequencyWeightedBalancedTruncationROM`
too, none of which have any passivity-preservation theorem behind them
(they are first-order STATE-SPACE reductions, not second-order Galerkin
ones -- the congruence argument above does not apply to them), to
check directly whether they can and do violate passivity in practice,
even when built from a stable, passive full-order model.

**API** (`passivity.py`, new module -- a genuinely different construction
from `balanced_truncation.py`'s own machinery, reusing `frequency.py`'s
already-validated `frequency_response()` rather than balancing/Lyapunov
solves at all):

```python
def velocity_transfer_function(H_disp, omega_array): ...
    # i * omega_array * H_disp -- H_disp from ANY of this package's
    # frequency_response() outputs (a collocated force-in/displacement-
    # out SISO port), or fea_engine's own solve_harmonic() sweep

def passivity_margin(H_vel): ...
    # Re[H_vel] at each swept frequency

def is_passive(H_vel, tol=0.0): ...
    # bool: passivity_margin(H_vel) >= -tol everywhere
```

**Validation plan**: reuse the SAME real fea_engine damped-cantilever
fixture and collocated tip-force-in/tip-out port every other module in
this "systems and control" family already uses.

- **The full-order model is passive** (a sanity check on the fixture
  itself, and on `is_passive()`'s own correctness): sweep
  `fea_engine`'s own `solve_harmonic()`, convert to velocity, confirm
  `Re[H_vel(i*omega)] >= 0` (up to numerical tolerance) across a wide
  sweep including near resonance.
- **A modally-reduced `FrequencyROM` (`galerkin.py`'s basis, this
  package's existing "mode displacement" capability) stays passive at
  every tested basis size** -- the direct, checkable version of the
  closed-form theorem above, not merely cited.
- **Whether `KrylovROM`/`BalancedTruncationROM`/
  `FrequencyWeightedBalancedTruncationROM` preserve or violate
  passivity on this fixture is CHECKED, not assumed either way** --
  reported honestly regardless of outcome, matching this project's
  established practice for every other non-guaranteed property
  (`KrylovROM.is_stable()`,
  `FrequencyWeightedBalancedTruncationROM.is_stable()`'s two-sided
  case).

New test file: `test_passivity.py`.

## 13. Phase 5c outcome: the theorem held, and the diagnostic found a real violation

`passivity.py` was implemented exactly as Section 12 specifies, and
validated in the new `tests/test_passivity.py` against the SAME real
fea_engine cantilever fixture and collocated tip-force/tip-velocity
port every other module in this family uses:

- **The full-order model is passive**, confirmed across a 300-point
  sweep including exact resonance (minimum margin `2.26e-09`, positive
  throughout) -- a sanity check on both the fixture and `is_passive()`
  itself.
- **The closed-form theorem held exactly as derived, checked directly
  rather than only cited**: a modally-reduced `FrequencyROM` stayed
  passive at every tested basis size (3, 6, 10, 15 modes), with margins
  matching the full-order model's own to the precision shown.
- **The genuinely open question this diagnostic exists to answer was
  answered, concretely, on real data**: sweeping `KrylovROM`,
  `BalancedTruncationROM`, and `FrequencyWeightedBalancedTruncationROM`
  at `r/k in {6, 10, 14}` (9 checks total), 8 stayed passive and ONE
  genuinely VIOLATED passivity --
  `FrequencyWeightedBalancedTruncationROM(r=14)` (with an output
  bandpass weight centered on the first natural frequency). This is
  not a synthetic or forced example: it is a real, measured passivity
  failure in a method this package already ships, found BECAUSE this
  diagnostic exists to look for it -- concrete evidence that none of
  these state-space reductions' other guarantees (moment matching,
  the H-infinity error bound, or the accuracy trade frequency-weighted
  BT makes) imply passivity preservation, which is exactly the
  distinction Section 12 predicted between second-order Galerkin
  projection (has a real theorem) and first-order state-space
  reduction (does not).

**Net assessment**: Phase 5c's scope adjustment (Section 12) was the
right call, confirmed by the outcome on both sides: the safer,
closed-form route delivered a genuine, provable guarantee with zero
derivation risk, AND its own diagnostic immediately found a real,
reportable passivity violation in an existing method -- a more
concretely useful result than a from-scratch positive-real-ARE
implementation would have been even if that derivation had gone
perfectly, since this outcome is grounded in an actual measurement on
this package's own fixture rather than a theoretical guarantee alone.
`is_passive()`/`passivity_margin()`/`velocity_transfer_function()` are
now available as a general-purpose check any caller can run against
any of this package's frequency-response-producing ROMs, not just the
ones validated here.

## 14. Phase 5d design: optimal Hankel norm approximation (AAK/Glover)

The last explicitly-named Phase 5 item. Adamjan-Arov-Krein (1971) theory,
given an explicit numerical realization by Glover ("All optimal
Hankel-norm approximations of linear multivariable systems and their
L-infinity error bounds", *Int. J. Control* 39(6), 1984), answers a
sharper question than ordinary balanced truncation's own a priori
H-infinity bound does: of EVERY possible order-r system (stable or not),
which one minimizes `||G - G_r||` in the HANKEL norm specifically, and
what is that minimum? The answer is exact, not a bound: the minimum
equals `sigma_(r+1)`, the (r+1)-th Hankel singular value of the
full-order system, and Glover's construction builds a system that
achieves it.

**Research**: dispatched to a subagent, triangulating three sources
(MOR Wiki's AAK page, Benner & Werner's survey arXiv:1612.06205, and
Sandberg's KTH lecture notes on model reduction) and cross-checking two
independently-notated presentations agree exactly under one
substitution. High-confidence findings: `sigma_(r+1)` must be SIMPLE
across the WHOLE spectrum (not just distinct from its immediate
neighbors) for the one-shot construction to apply; the formulas
(reproduced in `hankel_norm.py`'s own module docstring) act on the
balanced realization with ONE state (the one at `sigma_(r+1)`) isolated
and the other n-1 combined; the resulting intermediate order-(n-1)
system splits EXACTLY into r stable + (n-1-r) antistable eigenvalues,
needing no iteration to reach order r. Medium-to-high confidence: the
tight (non-doubled) L-infinity bound `sigma_(r+1) <= ||G-G_r||_inf <=
sum(sigma_(r+1)..sigma_n)` needs an additional recursive
constant-correction step not covered by the core formulas -- flagged as
the single least-verified point, and deliberately NOT implemented (see
below).

**A genuine derivation gap, found and closed independently, not in the
triangulated sources**: extracting the order-r STABLE part from the
intermediate order-(n-1) system needs more than a reordered Schur
truncation (which would only be approximate) -- the off-diagonal
coupling between the stable and antistable Schur blocks must be
eliminated exactly via a Sylvester equation
(`scipy.linalg.solve_sylvester`), folding the correction into `B_r`.
This Schur-plus-Sylvester decoupling is standard numerical linear
algebra for extracting an invariant subspace's reduced dynamics (the
same idea `scipy.linalg.solve_sylvester`'s own docstring examples use
for related problems), but was not found written out in exactly this
form in the sources -- exactly the kind of "own synthesis, not
independently verified in the literature" situation that has, in
`scm_lp.py`/`passivity.py`, previously led to either a documented scope
reduction or an explicit validation step. Here the choice was the
latter: validate the synthesis numerically before trusting it with
production code.

**Validation strategy chosen before writing any production code**: the
AAK theorem gives an EXACT, independently-checkable target -- the
Hankel norm of the error system `G - G_r` must equal `sigma_(r+1)`
exactly. Since that error system's Hankel singular values can be
computed by the SAME already-validated `hankel_singular_values()`
function this whole "systems and control" family already relies on,
this gives a strong, independent, off-the-shelf correctness check with
no new validation machinery needed. A scratch (non-production) Python
script built a small (n=6) random synthetic stable SISO system,
applied the transcribed formulas and the Schur-plus-Sylvester
decoupling by hand, and compared the error system's measured Hankel
norm to the theoretical `sigma_(r+1)`: relative difference `1.45e-11`,
essentially machine precision -- strong confirmation the formulas and
the decoupling procedure are correct, at least for that one case.

**A numerical-risk investigation, done BEFORE writing the module, not
after**: rather than trust one clean synthetic example, the same
scratch check was swept across 30 random synthetic SISO systems
(5 seeds x 6 (n, r) combinations, n from 6 to 10). Most cases matched
theory to near machine precision, but several did not -- reldiff as
large as 30 (not 30%, thirtyfold) in the worst case. The failures
correlated with `sigma_(r+1)` being very small relative to the largest
Hankel singular value `hsv[0]` (roughly, within about 6-8 orders of
magnitude of `hsv[0]` times machine epsilon in the swept examples), but
NOT cleanly: two cases with nearly identical `cond(Gamma)` (the
Gamma-inversion conditioning, the most obvious a priori candidate
quantity) had wildly different actual error (one case with the SAME
`cond(Gamma)` as a catastrophic failure had an acceptable `2.5e-4`
relative error instead), ruling out `cond(Gamma)` alone as a reliable a
priori guard. Rather than ship a threshold check that this
investigation had just shown to be untrustworthy, the design instead
makes the SAME empirical error-system check used for scratch validation
a PERMANENT, ALWAYS-ON part of every `OptimalHankelNormROM`
construction -- `numerically_reliable` / `measured_hankel_norm_error`,
with a loud `warnings.warn()` when they disagree with the theoretical
`sigma_r1` beyond a tolerance. This turns an open-ended numerical risk
into a concrete, self-reported, checkable diagnostic rather than a
documented caveat the caller has to remember and cannot verify.

**Scope decision -- SISO only**: the derivation's `U = B2 / C2` step is
a genuine scalar division. A MIMO generalization exists in the
literature (via an SVD/pseudo-inverse of the removed state's
input/output blocks) but was not independently re-derived or validated
here -- following the same practice `scm_lp.py`/`passivity.py`
established, `OptimalHankelNormROM` raises a clear `ValueError` for any
non-SISO system rather than guessing.

**Module placement**: a NEW module, `hankel_norm.py`, not a fourth
sibling class in `balanced_truncation.py`. It reuses
`hankel_singular_values(..., return_transform=True)` unchanged for the
balancing step (the one piece it genuinely shares with
`BalancedTruncationROM`/`SingularPerturbationROM`/
`FrequencyWeightedBalancedTruncationROM`), but the Gamma-inversion
formulas and the Schur-plus-Sylvester decoupling are algorithmically
unlike anything else in `balanced_truncation.py` -- the same reasoning
that already keeps `krylov.py` a separate module despite being in the
same "systems and control" family.

**API** (mirrors `BalancedTruncationROM`'s method names for direct
comparison, per this package's established convention):

```python
class OptimalHankelNormROM:
    def __init__(self, ss, hsv, T, Tinv, r, gap_tol=1e-8, reliability_tol=1e-4): ...
        # raises ValueError if: non-SISO; r out of range; sigma_(r+1)
        # not simple; the Schur split doesn't give exactly r stable
        # eigenvalues (itself usually a symptom of the numerical-risk
        # regime above, not a separate failure mode)
        # sets: A_r, B_r, Cout_r, D_r, sigma_r1,
        #       measured_hankel_norm_error, numerically_reliable
        #       (warns if not numerically_reliable)

    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, r=10, gap_tol=1e-8, reliability_tol=1e-4): ...

    def transfer_function(self, s): ...       # + D_r, like SingularPerturbationROM's
    def frequency_response(self, omega_array): ...
    def hankel_norm_error_bound(self): ...     # sigma_r1 -- the AAK-exact value
    def is_stable(self): ...                   # True by construction when init didn't raise
```

Deliberately NOT exposed: `h_infinity_error_bound()`. Glover's own
paper gives a tight (non-doubled) L-infinity bound, but achieving the
provably-tight lower half requires the additional recursive
constant-correction step flagged above as the least-verified research
point -- rather than claim a bound whose full derivation was not
independently re-verified here, this class follows `KrylovROM`'s and
`FrequencyWeightedBalancedTruncationROM`'s own precedent of honestly
omitting a method it cannot back with a verified guarantee.

**Validation plan**: reuse the real fea_engine damped-cantilever
fixture every other module in this family uses. Two things to check
directly, not assume: (1) the core AAK optimality claim --
`OptimalHankelNormROM`'s Hankel-norm error should be `<=`
`BalancedTruncationROM`'s own Hankel-norm error at the SAME r, computed
via the same error-system trick used in the numerical-risk
investigation above; (2) whether the `numerically_reliable` self-check
machinery both agrees with theory in a good case and correctly flags a
bad one, reproduced concretely on this fixture rather than only on the
synthetic stress test.

## 15. Phase 5d outcome: AAK optimality confirmed on real data; a real, fixture-specific numerical ceiling found and guarded

`hankel_norm.py`'s `OptimalHankelNormROM` was implemented exactly as
Section 14 specifies, and validated in the new `tests/test_hankel_norm.py`
against the same real fea_engine cantilever fixture and collocated
tip-force/tip-displacement port every other module in this family uses.

**A scale finding sharper than `BalancedTruncationROM`'s own**: this
specific beam/port combination has an unusually front-loaded Hankel
singular value spectrum -- the first mode pair dominates by roughly two
orders of magnitude over everything else (`hsv = [1.02e-5, 1.02e-5,
7.1e-8, 6.9e-8, 3.4e-9, ...]` at a 15-mode pre-reduction), and this
spread continues for many more decades at larger pre-reduction sizes.
Applying `OptimalHankelNormROM` at the SAME 15-mode, order-30
pre-reduction `test_balanced_truncation.py` uses successfully for
ordinary BT was NOT reliable here at `r=4` and above (caught by
`numerically_reliable`, not a silent failure) -- confirming, on real
data, that `OptimalHankelNormROM`'s extra Gamma-inversion step is
genuinely more sensitive to a wide Hankel-singular-value spread than
ordinary BT's own square-root balancing alone. The fix used throughout
the new test file is the same KIND of fix `balanced_truncation.py`
already documents (modal pre-truncation before balancing), just taken
further: a SMALLER 6-mode, order-12 pre-reduction, at which `r` in
`{1, 2, 3, 4}` all measured `numerically_reliable=True`, agreeing with
the AAK-theoretical `sigma_(r+1)` to within `4.7e-6` relative or better
(vs. the `reliability_tol=1e-4` threshold that actually gates
correctness) -- comfortably inside the reliable regime, not borderline.

**The core AAK optimality claim, verified directly on real data, not
just cited**: at every one of `r in {1, 2, 3, 4}`, `OptimalHankelNormROM`'s
Hankel-norm error was `<=` `BalancedTruncationROM`'s own Hankel-norm
error at the same r (e.g. r=3: `6.889314e-08` vs. `7.279258e-08`) --
computed via the SAME error-system trick used throughout this Phase's
own numerical-risk investigation, giving genuine, measured confirmation
of the theorem on this package's own data, not just on the earlier
synthetic examples.

**The self-check machinery was shown to do its job on both sides, not
just theorized**: in the reliable 6-mode-pre-reduction regime, it
agreed with the AAK-theoretical value to within `4.7e-6` relative at
worst (comfortably inside `reliability_tol`); reproducing the
15-mode-pre-reduction, r=4 combination that IS in the unreliable
regime, it correctly reported `numerically_reliable=False` (measured
Hankel-norm error `7.33e-08` vs. theoretical `6.40e-09` -- more than a
tenfold disagreement) with the accompanying warning firing as designed.

**An honest finding, not assumed favorably**: Hankel-norm optimality
did NOT translate into better ordinary sup-norm/relative
frequency-response accuracy than `BalancedTruncationROM` at the same r
on this fixture -- measured directly (r=2: `3.99` vs. `0.43` max
relative error across a resonance-including sweep; r=4: `0.137` vs.
`0.055`), `OptimalHankelNormROM` was WORSE both times. This is
consistent with, not contradictory to, the AAK theorem: Hankel-norm
optimality is a statement about a specific mathematical norm (closely
related to, but not identical to, either the sup-norm or ordinary
relative error at any given frequency), not a general practical-
accuracy guarantee -- reported honestly here, mirroring
`FrequencyWeightedBalancedTruncationROM`'s own trade-off reporting in
Section 11, rather than assuming the theoretically stronger guarantee
must also mean practically better.

**Net assessment**: this closes out the classical MOR roadmap's last
explicitly-named Phase 5 item. The construction's own genuine
derivation risk (the Schur-plus-Sylvester decoupling) was validated
numerically before being trusted with production code, exactly as
`scm_lp.py`/`passivity.py` established as this project's practice for
this kind of situation; the resulting real numerical fragility (found
by deliberately stress-testing BEFORE writing the module, then
reproduced concretely on this package's own fixture, not just
theorized) is guarded by an always-on empirical self-check rather than
an unreliable a priori threshold; and the core theoretical claim
(Hankel-norm optimality) was verified directly against
`BalancedTruncationROM` on real data, while the accompanying practical-
accuracy finding was reported honestly in whichever direction it
actually pointed, matching this whole roadmap's practice throughout
Phases 5a-5d.
