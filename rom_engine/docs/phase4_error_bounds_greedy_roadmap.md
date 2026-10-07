# Phase 4 -- certified error bounds + greedy sampling for `rom_engine`

Status: **Phases 4a, 4b, 4d, and 4e are implemented and validated**
(Sections 2, 4-6, 8-10 below) -- `AffineDecomposition.assemble_action()`,
`FrequencyROM.residual_norm()`/`.hierarchical_error_indicator()`,
`greedy.py`'s `greedy_train_frequency_basis()`, `scm.py`'s
`SingularValueLowerBound`/`certified_error_bound()`, and `scm_lp.py`'s
`LPSingularValueLowerBound`/`certified_error_bound()` (the genuine
classical LP-based SCM) all exist, tested against real fea_engine
models in
`test_frequency.py`/`test_greedy.py`/`test_scm.py`/`test_scm_lp.py`
(14 new tests since the original Phase 4a/4b work, all passing). Phase
4c and the Phase-4d-equivalent example + docs are also done -- see
`examples/greedy_frequency_training.py` and
`examples/certified_bound_cantilever.py`. **Phase 4d's outcome (Section
8) is a genuinely certified bound whose certification guarantee holds
unconditionally, but whose practical usefulness for a real FE
stiffness matrix is much narrower than hoped; Phase 4e's outcome
(Sections 9-10) is the real classical LP-based SCM, also
unconditionally certified, measured to be narrower still on the same
model** -- read Sections 8 and 10 before choosing between `scm.py` and
`scm_lp.py` for anything beyond a tight neighborhood of a known
reference frequency.

This document was the direct follow-on to
`docs/frequency_domain_rom_roadmap.md` Section 2c/Phase 4 -- that
document deferred this topic with the note that a rigorous error bound
"is exactly the standard difficulty in the literature especially near
resonances." This document researched that difficulty properly instead
of leaving it deferred, and the roadmap below (now implemented through
Phase 4c) was honest about which parts are and are not solved by the
classical theory -- a distinction the implementation preserves: see
`residual_norm()`/`hierarchical_error_indicator()`'s docstrings, both
still named to signal they are NOT certified bounds.

## 1. What Phase 4 is actually for

`FrequencyROM.error_estimate()` currently exists (see
`src/rom_engine/frequency.py`) and is deliberately named
`error_ESTIMATE`, not `error_bound`: it returns the raw full-order
residual norm `||F - A(omega) @ (V @ q)||`, explicitly documented as a
ranking signal only, with two acknowledged gaps:

1. It costs one full `(n_dof, n_dof)` assembly per call
   (`AffineDecomposition.assemble(omega)`), which is fine for
   occasional diagnostic use but far too expensive to call at every
   candidate frequency inside a greedy training loop -- the very use
   case Phase 4 exists for.
2. It has no rigorous relationship to the TRUE error
   `||x_true(omega) - V q(omega)||` -- turning a residual into a
   genuine upper bound needs dividing by a lower bound on `A(omega)`'s
   smallest singular value, which was flagged but not solved.

Phase 4 has two genuinely separate deliverables that get conflated in
casual descriptions of "certified ROMs," so this document keeps them
separate throughout:

- **(A) An error INDICATOR cheap enough to call thousands of times**
  -- needed for greedy sampling, where the loop repeatedly asks "which
  untried frequency is worst-represented by the current basis?" This
  does NOT need to be a rigorous bound to be useful -- it needs to be
  cheap and to correctly RANK candidate frequencies by how bad the
  current ROM is there.
- **(B) A genuinely certified error BOUND** -- a number that is
  provably `>= true error`, usable to say "this ROM's answer is
  guaranteed accurate to within X" without checking against a
  full-order solve. This is the harder, classical reduced-basis-method
  goal, and the literature is explicit that it is NOT reliably
  achievable near resonance with the standard approach (Section 3).

## 2. Efficient residual evaluation (solves deliverable A's cost problem)

The current `error_estimate()`'s cost problem has a concrete, fully
correct (not approximate) fix that doesn't require any new theory:
precompute `A_q @ V` (an `(n_dof, n_modes)` matrix per affine
component `q`, computed ONCE, offline) instead of only the doubly-
projected `V^T A_q V` that `AffineDecomposition.project()` already
caches. Online, at a new `omega`:

```
A(omega) @ x  =  A(omega) @ (V @ q)  =  sum_q theta_q(omega) * (A_q @ V) @ q
```

which costs `O(Q * n_dof * n_modes)` (dominated by the cached
`(A_q @ V) @ q` mat-vecs) instead of the current
`O(n_dof^2)` (building the full `A(omega)` via
`AffineDecomposition.assemble()`). Since `n_modes << n_dof` by
construction, this is a real, substantial speedup, not a
micro-optimization -- and it computes the EXACT residual, not an
approximation of it, so it's a strict improvement over the current
`error_estimate()` with no accuracy trade-off. This is the natural
`Q2` follow-up to `AffineDecomposition.project()`, which already
caches `V^T A_q V`; caching `A_q @ V` alongside it is the same idea
one projection short of where it currently stops.

Going further -- to a residual norm costing `O(Q^2 * n_modes^2)`,
independent of `n_dof` even for the vector operations, exactly the
classical reduced-basis "fully offline-online" residual dual norm --
needs precomputing the `Q^2` cross-Gram quantities
`(A_p @ V)^H (A_q @ V)` (or, more precisely, the Riesz-representor
inner products the classical theory uses) offline. This is well-
established (Rozza/Huynh/Patera and follow-ons, Section 3) but adds
real implementation complexity for a further constant-factor gain once
`n_dof` is already out of the cost; it is scoped as an optional
refinement (Section 6, Phase 4c) rather than a Phase-4 prerequisite,
since the `O(Q n_dof n_modes)` version above is already a large,
correctness-preserving improvement on its own.

## 3. What the literature says about certified bounds near resonance
   (deliverable B) -- and why this document recommends NOT starting there

The classical reduced-basis a posteriori bound is
`||x_true - V q|| <= ||residual|| / beta_LB(omega)`, where
`beta_LB(omega)` is a rigorous, cheaply-computable LOWER bound on
`A(omega)`'s smallest singular value (its "coercivity" or "inf-sup"
constant). Computing `beta_LB` rigorously and cheaply is itself a
whole sub-field: the **Successive Constraint Method** (SCM; Huynh,
Rozza, Sen, Patera, and the improved "natural-norm" SCM of Chen et
al.) formulates it as a small online linear program, using an offline-
selected set of "control points" via its own inner greedy procedure.
It is a real, workable, well-cited method -- but the literature is
explicit and specific about its limitation for exactly this package's
use case: **inf-sup-constant-based error estimators cannot provide
tight error bounds near resonant frequencies, and the residual norm
alone cannot provide sharp error estimation there either** (Section
"Main Challenges Near Resonance" in the sources below) -- because
`beta_LB(omega) -> 0` at a resonance is precisely the physical
phenomenon being modeled, not a numerical artifact to be estimated
away. A bound that divides by a quantity approaching zero necessarily
blows up, correctly reflecting large uncertainty but not usefully
CERTIFYING anything useful there.

Given that, implementing a full SCM pipeline as Phase 4's first
deliverable would be substantial effort spent on a technique the
literature already says under-delivers exactly where `FrequencyROM` is
most interesting (near a structure's resonances -- the peaks are the
point of a frequency sweep). This document instead recommends the
**hierarchical a posteriori error estimator** (Hain, Ohlberger, Radic,
Urban) as Phase 4's primary deliverable: instead of a coercivity-
constant-based bound, build TWO reduced models at different accuracy
(e.g. basis size `r` and `r + delta`) and use their DISAGREEMENT as the
error indicator for the smaller one. This needs no coercivity constant
at all, is reported as sharp (efficiency index near 1) when the
Kolmogorov N-width decays quickly -- true for smooth, well-separated
structural resonances -- and is directly implementable with machinery
this package already has (`FrequencyROM` at two basis sizes, compared
via `frequency_response()`). It is NOT a rigorous, provable bound in
the SCM sense (it relies on a "saturation assumption" that the larger
model actually IS more accurate, not a certificate) -- that honest
caveat is carried into the API naming in Section 4, exactly as
`error_estimate()` already does today.

**Recommendation**: Phase 4a/4b (Sections 5-6) implement the cheap
efficient residual (Section 2, unconditionally useful) and the
hierarchical estimator (practical, implementable now, good near
resonance). A true SCM-based certified bound is scoped as Phase 4c,
optional, explicitly labeled as sharing the near-resonance limitation
the literature documents -- not silently promised as a solution to a
problem the field itself hasn't fully solved for this class of
problem.

## 4. Greedy sampling: the weak greedy algorithm

The standard construction (widely used, convergence theory
established for both Hilbert and Banach-space settings): starting from
a small seed basis, repeatedly (1) evaluate a cheap error INDICATOR
(Section 2 or 3's estimator, not a full-order solve) across a
candidate "training set" of parameter values, (2) full-order-solve
ONLY at the worst-indicated candidate, (3) add that solution to the
basis (extend the POD snapshot set and re-fit), (4) repeat until the
worst indicated error drops below a tolerance or a maximum basis size
is reached. This concentrates expensive full-order solves where the
basis is actually weak (near resonances, for `FrequencyROM`) instead
of wasting them on a uniform grid's flat regions -- directly the
motivation already stated in
`docs/frequency_domain_rom_roadmap.md` Section 2c.

**Implementation-pattern reference**: pyMOR's `weak_greedy(surrogate,
training_set, atol=..., rtol=..., max_extensions=...)` decouples the
generic greedy LOOP from the problem-specific error surrogate via a
small `WeakGreedySurrogate`-style interface (an object with an
`evaluate(candidate_mus)` method) and a reductor-like object with
`extend_basis()`. `rom_engine` should borrow the DECOUPLING, not the
class hierarchy -- consistent with this package's existing style
(`pod.py`/`galerkin.py`/`affine.py`/`frequency.py` are plain classes
and functions composed together, not an abstract-base-class
framework), a plain callback function is a better fit than an
abstract surrogate class here.

## 5. Proposed API

```python
# rom_engine/frequency.py -- extend FrequencyROM

class FrequencyROM:
    ...
    def residual_norm(self, omega, F):
        """Efficient (O(Q*n_dof*n_modes), NOT O(n_dof^2)) exact
        residual norm -- see Section 2. Replaces error_estimate()'s
        reliance on a full assemble() call; same value, cheaper to
        compute, callable inside a greedy loop's inner sweep over many
        candidate omegas."""

    def hierarchical_error_indicator(self, omega, F, richer_rom):
        """||self.frequency_response([omega], F) - richer_rom.
        frequency_response([omega], F)|| -- Section 3's practical,
        coercivity-constant-free error indicator. `richer_rom` is a
        second FrequencyROM built on a LARGER basis (same M, K, C).
        Explicitly NOT a certified bound (relies on richer_rom being a
        materially better approximation, unverified) -- named
        "indicator", matching error_estimate()'s existing honest-naming
        convention, not "bound"."""
```

```python
# rom_engine/greedy.py -- new module

def greedy_train_frequency_basis(training_omegas, M, K, F, C=None,
                                  seed_omegas=None, tol=1e-3,
                                  max_modes=30, richer_extra_modes=4):
    """Weak-greedy construction of a POD-on-FRF-snapshots basis for
    FrequencyROM (docs/phase4_error_bounds_greedy_roadmap.md Section 4):
    starts from a small seed snapshot set (seed_omegas, or a handful of
    evenly spaced points in training_omegas if not given), then
    repeatedly (1) builds a FrequencyROM at the current basis AND a
    slightly richer one (+richer_extra_modes) for the hierarchical
    indicator, (2) evaluates that indicator across every UNTRIED
    omega in training_omegas, (3) full-order-solves ONLY at the worst-
    indicated omega and adds it to the snapshot set, (4) repeats until
    the worst indicator drops below tol or max_modes is reached.

    Returns (basis, history): basis is a fitted PodBasis; history is a
    list of (omega_selected, indicator_value_before_adding_it) pairs,
    kept so a caller can plot/audit which frequencies the greedy loop
    actually chose -- e.g. confirming they cluster near resonances,
    the qualitative claim this module's test suite checks directly.
    """
```

Kept as a separate `greedy.py` module (not folded into `frequency.py`)
because the weak-greedy LOOP itself is not frequency-specific --
`affine.py`'s general material-parameter case could use the same
pattern with a different error indicator and full-order solver, a
natural future generalization noted here but not implemented now
(scope stays to what Phase 4 was actually asked for: the frequency-
domain case).

## 6. Validation plan

Extend `tests/test_frequency.py` (residual_norm/hierarchical_error_
indicator checks) and add `tests/test_greedy.py`, using the SAME
`damped_cantilever_beam_system()` fixture and `fea_engine`-computed
ground truth this package's whole test suite already relies on:

- **`residual_norm()` matches the current full-`assemble()`-based
  residual to machine precision** -- a direct regression check that
  the Section 2 optimization is a cost improvement only, not an
  accuracy change (compare against today's `error_estimate()`
  computation at several omegas).
- **The hierarchical indicator is large near resonance, small away
  from it** -- checked directly against the TRUE error (from
  `fea_engine.solve_harmonic()`), i.e. confirm the indicator actually
  correlates with real error rather than just asserting it looks
  reasonable; report both series (indicator vs. true error) across a
  sweep, not a single spot check.
- **Greedy-selected training frequencies cluster near resonance** --
  a directly checkable, honest claim: the selected `omega` history
  should have higher density near the fixture's known first resonance
  than a uniform grid would, not merely "seems reasonable."
- **A greedy-trained basis beats a same-rank uniform-grid-trained
  basis on a held-out test set** -- the key claim that justifies
  greedy sampling's extra complexity at all, checked by actually
  comparing both (not assumed): build both at the same final rank,
  measure max relative error against `fea_engine` ground truth on
  frequencies NEITHER training method selected.
- **Phase 4c (if built): SCM-based `beta_LB(omega)` is a genuine lower
  bound** -- checked directly against `np.linalg.svd(A(omega))`'s true
  smallest singular value across a sweep including a near-resonance
  point, with the expected degradation near resonance reported
  explicitly rather than the check being quietly scoped to avoid that
  region.

## 7. Phased roadmap

**Phase 4a -- efficient exact residual (Section 2).** Add
`FrequencyROM.residual_norm()`. Small, self-contained, zero accuracy
trade-off, immediately useful on its own (cheaper diagnostics) and a
prerequisite for 4b/4d's inner loops.

**Phase 4b -- hierarchical error indicator + greedy training (Sections
3-5).** Add `FrequencyROM.hierarchical_error_indicator()` and the new
`greedy.py` module's `greedy_train_frequency_basis()`. This is the
main deliverable: a working, validated (Section 6) greedy-sampling
capability that concentrates training solves near resonances, without
depending on the SCM machinery Section 3 explains is weak exactly
there.

**Phase 4c -- example + docs.** An `examples/` script comparing a
greedy-trained basis against a uniform-grid one on the damped
cantilever (mirroring `frequency_sweep_cantilever.py`'s existing
shape), plus a README section, matching this package's established
documentation standard.

**Phase 4d -- SCM-based certified lower bound.** Built as a
deliberately SIMPLIFIED SCM-family method (`scm.py`'s
`SingularValueLowerBound`, based on the Weyl/Mirsky singular-value
perturbation inequality) rather than the full classical LP-based
natural-norm SCM -- see Section 8 for what this delivers, and its
concretely measured limitation.

## 8. Phase 4d outcome: a real bound, correctly implemented, narrower
   than hoped

`scm.py` implements `sigma_min(A(omega)) >= sigma_min(A(omega_j)) -
sum_q |theta_q(omega) - theta_q(omega_j)| * ||A_q||_2` from a set of
offline reference points `omega_j` (Weyl/Mirsky perturbation
inequality + the affine triangle inequality -- rigorous, not a
heuristic). This is genuinely a member of the SCM family (an
offline-selected reference/control-point construction bounding a
parametric stability quantity), but a much simpler one than the
classical LP-based natural-norm SCM (Section 3's sources): no linear
program, no "attainable set" -- just a single global Lipschitz
constant per affine component, `||A_q||_2`.

**What was verified to hold, unconditionally** (`test_scm.py`,
`examples/certified_bound_cantilever.py`):

- `lower_bound(omega)` never exceeds the true `sigma_min(A(omega))` --
  checked at 300-400 points including exact resonance, worst observed
  violation exactly `0.0`. This is the actual certification promise,
  and it holds by construction (Weyl/Mirsky is unconditional), not by
  luck on this particular test model.
- A finite `certified_error_bound()` never undershoots the true error
  measured against fea_engine's own `solve_harmonic()`.
- The bound is exact (not merely non-violating) precisely at its own
  reference point -- `lower_bound(omega_j) == sigma_min(A(omega_j))`
  to machine precision.

**What was discovered, and is now documented rather than hidden**: for
this package's real cantilever fixture, `||K||_2` (the stiffness
component's spectral norm, ~1e11-1e12 for a realistic beam
discretization) is roughly 8 orders of magnitude larger than
`sigma_min(A(omega))` itself (~1e3) at any physically relevant
frequency. Since the perturbation term scales with the RAW component
norm, `lower_bound()` collapses to the trivial `0` within roughly
`1e-7` relative frequency of a reference point -- confirmed by direct
measurement (`test_useful_radius_is_narrow_for_this_model_and_is_
reported_honestly` in `test_scm.py`, and Part 2 of
`certified_bound_cantilever.py`), not asserted or assumed. This is
FAR narrower than the >1%-of-first-resonance spacing that suffices for
`greedy.py`'s hierarchical-indicator-driven basis training to work
well (Section 6's bullet on that).

This was checked to make sure it wasn't a bug: `rom.affine.components`
correctly holds the FULL-ORDER (not reduced) `M`/`K` matrices, which
IS the mathematically correct thing to bound (`x_true - x_ROM =
A(omega)^{-1} @ residual` needs the full operator's `sigma_min`, not a
reduced one). The narrowness is a genuine, expected consequence of a
realistic FE stiffness matrix's enormous eigenvalue spread interacting
with a Lipschitz-type bound that doesn't exploit any problem structure
beyond the raw operator norm -- precisely the scale-mismatch weakness
the classical "natural-norm" SCM (Chen et al. 2010) was invented to
address, by choosing a smarter, problem-adapted norm instead of the
raw spectral norm. Re-implementing that sharper (but substantially
more complex, LP-based) method remains scoped as optional future work
(README's Roadmap section), pursued only if a real use case needs a
certified bound informative across a band wider than a reference
point's immediate neighborhood.

**Net assessment**: Phase 4d delivered exactly what a "genuinely
certified" bound has to deliver -- a mathematically rigorous,
verified-never-violated guarantee -- and delivered it honestly,
including the discovery that this simplified construction's practical
reach is narrow for a realistic structural model. That narrowness is
reported as a measured, load-bearing fact in the module's docstring,
tests, example, and README caveat, not smoothed over. `scm.py` is
correctly scoped as a rigorous, narrow safety margin around specific
reference frequencies -- a genuine, useful capability -- not as a
general-purpose replacement for `residual_norm()`/`hierarchical_error_
indicator()`'s practical, band-wide (if uncertified) error signal.

## 9. Phase 4e -- a genuine LP-based SCM, scoped around a real derivation risk

Section 8's honest finding -- `scm.py`'s simplified bound collapses to
`0` within ~1e-7 relative frequency of a reference point, because it
uses each component's raw, global spectral norm `||A_q||_2` as a
worst-case Lipschitz constant -- is exactly the scale-mismatch problem
the classical **natural-norm SCM** (Chen et al. 2010/2015) was invented
to fix, by replacing the raw norm with a problem-adapted norm measured
through a local "supremizer" operator `T^mu_bar`. Implementing that
variant for `rom_engine`'s actual operator was investigated directly
(a literature-research pass on both the original SCM and the natural-
norm SCM, focused specifically on the online LP's exact structure) and
surfaced a real, not merely inconvenient, obstacle: **the natural-norm
construction's literature is written for real symmetric/coercive or
real non-symmetric operators; `rom_engine`'s `A(omega) = -omega^2 M +
i*omega*C + K` is complex and non-Hermitian for any `omega != 0` with
damping present, and neither source paper treats this case explicitly**.
Extending it requires guessing a real-part convention
(`y_q = Re[a_q(w, T^mu_bar w)] / ||T^mu_bar w||^2`) to keep the LP's
data real, since `scipy.optimize.linprog` requires real `c`/`A_ub`/
`b_ub` -- a plausible but literally unverified extension. Given that a
"certified" bound is only trustworthy if its derivation is actually
correct (a subtly wrong certified bound is worse than an honestly
narrow one), this document does NOT implement that unverified
extension.

**What this section implements instead** is the genuine, textbook
classical SCM (Huynh, Rozza, Sen, Patera 2007) -- a real online linear
program over offline-selected reference/"control" points, exactly the
machinery Section 3 and `scm.py`'s own docstring described as "not
implemented here" -- applied via a standard, well-established reduction
that sidesteps the complex-operator ambiguity entirely: instead of
bounding `sigma_min(A(omega))` directly, bound
`sigma_min(A(omega))^2 = lambda_min(A(omega)^H A(omega))`, the smallest
eigenvalue of a matrix that is REAL, SYMMETRIC, and PSD by construction
for any `A(omega)`, complex and non-Hermitian or not.

**The affine expansion stays small.** `A(omega)^H A(omega) =
sum_p sum_q conj(theta_p(omega)) theta_q(omega) A_p^H A_q`. Pairing
each `(p, q)` term with its `(q, p)` mirror collapses this into
`Q(Q+1)/2` REAL-coefficient terms over REAL SYMMETRIC component
matrices:

```
B_qq = A_q^T A_q                              (coefficient |theta_q(omega)|^2)
B_pq = A_p^T A_q + A_q^T A_p,   p < q          (coefficient Re[conj(theta_p(omega)) theta_q(omega)])
```

(using `A_q^T`, not `A_q^H`, since `M`/`C`/`K` are themselves real).
This IS the literature's "SCM-squared" reduction, and the literature's
own caveat about it -- that squaring blows up the online LP to
`Q(Q+1)/2` variables and loses some tightness -- is a real cost for
problems with many components (`Q` in the tens); for `rom_engine`'s
`Q <= 3` (`{M, C, K}` or `{M, K}` under proportional damping), that is
at most 6 variables, i.e. no practical blow-up at all. This is a
genuine instance of the classical (non-natural-norm) SCM -- not a new
invention -- applied in a regime (small `Q`) where its known weakness
doesn't bite.

**Online LP, exactly the classical SCM's structure** (Section 3's
sources, Part B): variables `y in R^{Q(Q+1)/2}`, one per `B_pq`.

- **Box bounds** (offline, `Q(Q+1)/2` small dense eigendecompositions,
  done once): `y_pq in [lambda_min(B_pq), lambda_max(B_pq)]` -- valid
  for ANY unit vector `w`, since a Rayleigh quotient always lies
  between a symmetric matrix's extreme eigenvalues.
- **Reference-point constraints** (one per stored reference `omega_j`,
  where `alpha_j := lambda_min(A(omega_j)^H A(omega_j))` was computed
  exactly, offline, from a single SVD -- the SAME cost `scm.py`'s
  `add_reference()` already pays): `sum_pq c_pq(omega_j) y_pq >=
  alpha_j`, where `c_pq(omega)` is the coefficient vector above. This
  holds for the SAME reason `scm.py`'s Lipschitz bound holds: for
  ANY vector `w` (in particular, the unknown minimizer at the query
  `omega`), its Rayleigh quotient at a DIFFERENT operator can never be
  smaller than that operator's own smallest eigenvalue.
- **Objective**: minimize `sum_pq c_pq(omega) y_pq`, i.e.
  `scipy.optimize.linprog(c=c(omega), A_ub=-C_ref, b_ub=-alpha_ref,
  bounds=box_bounds)` where `C_ref` stacks `c(omega_j)` for every
  stored reference. `Q(Q+1)/2` variables, `Q(Q+1)/2 * 2 + n_references`
  constraints -- sub-millisecond, exactly the classical SCM's online
  cost profile.
- **Bound**: `sigma_min(A(omega)) >= sqrt(max(0, LP_optimum))`.

This is unconditionally rigorous by the same two facts `scm.py` already
leans on (Rayleigh-quotient bracketing by extreme eigenvalues; a
Rayleigh quotient's value at another operator lower-bounds nothing
except that OTHER operator's own smallest eigenvalue) -- no new,
unverified mathematical step is introduced, unlike the natural-norm
extension above.

**API** (`scm_lp.py`, new module -- kept separate from `scm.py` rather
than added to `SingularValueLowerBound`, since it is a genuinely
different algorithm with different offline cost and data, not a tweak
to the existing one; mirrors `SingularValueLowerBound`'s method names
deliberately, so the two can be swapped in and compared directly on
the same model):

```python
class LPSingularValueLowerBound:
    def __init__(self, components, theta_func): ...
    @classmethod
    def from_affine(cls, affine): ...
    def add_reference(self, omega): ...      # one SVD; stores alpha_j and c(omega_j)
    def lower_bound(self, omega): ...          # solves the LP, returns sqrt(max(0, LP optimum))
    def greedy_train(self, candidate_omegas, tol=..., max_references=..., seed_omega=None): ...
```

**Validation plan -- the point of building this at all**: reuse
`test_scm.py`'s exact fixture
(`damped_cantilever_beam_system(n=20)`, 10-mode basis) and its
never-exceeds-true-sigma_min certification check, PLUS the same
useful-radius measurement `test_useful_radius_is_narrow_for_this_
model_and_is_reported_honestly` performs for `SingularValueLowerBound`
-- run on `LPSingularValueLowerBound` too, on the SAME reference
point, and report both radii side by side. If the LP-based bound's
radius is not measurably wider, that is reported as a finding, not
assumed away -- consistent with Section 8's own practice of measuring
rather than asserting.

**Honest scope note**: this delivers the real, textbook classical SCM
-- a genuine online LP over offline reference/control points -- which
IS "the full LP-based Successive Constraint Method" in the sense
`scm.py`'s docstring flagged as unimplemented. It is the classical
(coercivity-style, applied via the standard SCM² reduction), not the
natural-norm variant, and that distinction is deliberate: the natural-
norm variant's sharper, problem-adapted norm is exactly what would be
needed if this LP-based bound turns out to still be scale-limited for
this model (plausible, since `B_pq`'s eigenvalue spread inherits
`K`'s own ~1e11-1e12 spread, squared) -- if so, that becomes a new,
concretely measured finding for a future Phase 4f, not a claim made in
advance of measuring it.

## 10. Phase 4e outcome: correctly implemented, and (as flagged above)
   scale-limited MORE severely than the simplified bound, on this model

`scm_lp.py`'s `LPSingularValueLowerBound` was implemented, wired to a
real `scipy.optimize.linprog` online LP exactly as Section 9 specifies,
and validated in `test_scm_lp.py` against the same
`damped_cantilever_beam_system(n=20)` fixture `test_scm.py` uses.

**What was verified to hold, unconditionally** (`test_scm_lp.py`):

- `lower_bound(omega)` never exceeds the true `sigma_min(A(omega))`,
  checked at 200 points including exact resonance.
- A finite `certified_error_bound()` (the `scm_lp` module-level
  function, mirroring `scm.certified_error_bound()`) never undershoots
  the true error against fea_engine's own `solve_harmonic()`.
- At its own reference point, `lower_bound()` matches the true value
  to numerical precision (e.g. `17046495296` vs. the true
  `17046495269` -- a ~1e-6 relative gap, consistent with going through
  an LP solve and a `sqrt` rather than a closed-form formula, unlike
  `scm.py`'s exact-to-machine-precision case).

**A real numerical-conditioning bug found and fixed during
validation**: the naive online LP, run with `scipy.optimize.linprog`'s
default HiGHS backend, reported "unbounded" for the mirror-image
maximization used to compute `upper_bound()` (the greedy-selection
heuristic, not the certified bound itself) -- on a feasible region
that is PROVABLY compact (a finite box intersected with half-spaces),
so a genuine solver numerical failure, not a real unboundedness. Root
cause, confirmed directly: `B_pq`'s eigenvalue-based box bounds span
`>30` orders of magnitude for this model's real stiffness matrix (from
`~1e-10` to `~4.5e23`, since `B_pq` is built from `A_p^T A_q` and
inherits `K`'s own `~1e11-1e12` spectral norm, SQUARED) -- beyond what
HiGHS's internal scaling reliably handles. A per-variable rescaling fix
was tried first and made things WORSE (it relocates the same extreme
magnitude from the bounds into the objective/constraint-matrix data,
where it then triggers a DIFFERENT HiGHS failure, "Model error", by
exceeding its `large_matrix_value` validation threshold) -- confirming
this is a structural consequence of the real magnitude spread, not a
formatting artifact fixable by rescaling. The actual fix: `lower_bound()`
keeps the full box+reference-constrained LP (needed for the certified
value, and verified to solve correctly and match the true value at
reference points); `upper_bound()` -- used only to rank candidates for
greedy selection, where a looser but still-valid bound is an acceptable
trade -- was rewritten as a closed-form box-only maximization (drops
the reference constraints, which can only loosen, never invalidate,
a maximum), sidestepping the solver instability entirely. See
`scm_lp.py`'s `upper_bound()` docstring for the full account.

**The finding this module set out to measure, reported honestly**:
`test_useful_radius_compared_directly_against_simplified_bound` runs
both `scm.SingularValueLowerBound` and `scm_lp.LPSingularValueLowerBound`
from the SAME reference point on the SAME model and sweeps the same
relative-frequency grid used in Section 8's own radius measurement.
Result: the simplified (Lipschitz) bound's useful radius (`~1e-7`,
matching Section 8) was WIDER than the LP-based bound's, which was
already `0` at the narrowest step tested (`1e-8`). This is the
opposite of the hoped-for direction, but it is exactly the risk Section
9 flagged in advance, not a surprise discovered after the fact: the
SCM-squared reduction bounds `sigma_min(A(omega))^2` via box bounds on
`B_pq = A_p^T A_q + A_q^T A_p`, whose own spectral spread is roughly
the SQUARE of `A_q`'s (`~1e22-1e24` here, vs. `~1e11-1e12` for the raw
components `scm.py`'s bound uses) -- squaring a bound that was already
scale-limited by an 8-order-of-magnitude spread makes the scale
mismatch worse, not better. This is precisely the failure mode the
natural-norm SCM (Chen et al.) exists to avoid, by never squaring the
operator in the first place -- confirming, with a concrete measurement
rather than a forecast, that Section 9's unverified-derivation-risk
reason for not attempting the natural-norm variant is also the reason
a real gain from a genuine classical LP-based SCM was not available
here without it.

**Net assessment**: Phase 4e delivered what it set out to -- the
actual classical, LP-based Successive Constraint Method, correctly
implemented and unconditionally certified, closing the "not
implemented here" gap `scm.py`'s own docstring names. It does NOT
deliver a wider useful radius than the simpler bound on this package's
real cantilever model; that negative result is reported as a measured
fact (`test_scm_lp.py`, this section), not smoothed over, mirroring
Section 8's own practice for `scm.py`. Both `SingularValueLowerBound`
and `LPSingularValueLowerBound` remain available side by side (matching
method names, by design) for future models where the balance may
differ, or as the concrete, honest baseline against which a future
natural-norm SCM implementation (Phase 4f, if ever pursued, once the
complex-operator derivation risk Section 9 flagged is independently
resolved) would need to demonstrate real improvement.

## Sources

- Huynh, D.B.P., Rozza, G., Sen, S., Patera, A.T. ["A successive constraint linear optimization method for lower bounds of parametric coercivity and inf-sup stability constants."](https://www.sciencedirect.com/science/article/pii/S1631073X07003871)
- Chen, Y. et al. ["A natural-norm Successive Constraint Method for inf-sup lower bounds."](https://www.sciencedirect.com/science/article/abs/pii/S0045782510000691) / ["A Certified Natural-Norm Successive Constraint Method for Parametric Inf-Sup Lower Bounds."](https://arxiv.org/abs/1503.04760)
- ["Fast A Posteriori State Error Estimation for Reliable Frequency Sweeping in Microwave Circuits via the Reduced-Basis Method"](https://arxiv.org/pdf/2110.05925) -- the direct source for "inf-sup/residual-based estimators cannot provide tight bounds near resonance," in a frequency-sweep setting mathematically analogous to `FrequencyROM`'s.
- Hain, S., Ohlberger, M., Radic, M., Urban, K. ["A Hierarchical A-Posteriori Error Estimator for the Reduced Basis Method."](https://arxiv.org/abs/1802.03298) Advances in Computational Mathematics.
- ["Convergence Rates for Greedy Algorithms in Reduced Basis Methods"](https://www.researchgate.net/publication/220132479_Convergence_Rates_for_Greedy_Algorithms_in_Reduced_Basis_Methods) -- weak-greedy convergence theory.
- pyMOR documentation: [`pymor.algorithms.greedy`](https://docs.pymor.org/latest/autoapi/pymor/algorithms/greedy/index.html) -- `weak_greedy`/`WeakGreedySurrogate` API precedent for decoupling the greedy loop from the error surrogate.
- `docs/frequency_domain_rom_roadmap.md` (this package) -- Section 2c/Phase 4, the original deferred scope this document follows up on.
