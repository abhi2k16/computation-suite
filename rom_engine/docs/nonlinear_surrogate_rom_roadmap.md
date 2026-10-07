# Generalized nonlinear-ROM / surrogate-correction capability for `rom_engine` -- roadmap

Status: **Phases 1-5 (all phases) are implemented and validated.** `rom_engine/sampling.py`
(`optimal_lhs()`, `modal_force_samples()`) and `rom_engine/metrics.r_squared()`
(Phase 1) exist and are tested. `rom_engine/nonlinear_rom.py` (Phase 2)
now provides `KERNEL_REGISTRY` (4 kernels + analytic Jacobians),
`ReducedForceModel`, `MultiFidelitySurrogate` (RBF fit of `F_nl(q_l)`),
`PolynomialModalROM` (Nash-form fit of `F_nl(q_nl)` + Newton solve, with
`ICEROM`/`ShiMeiROM`/`EnforcedDisplacementROM` convenience constructors),
and both `AppliedLoadStrategy`/`EnforcedDisplacementStrategy` training
strategies -- ported from, and validated against, this project's own
earlier `mfs-nlrom-beam` prototype, a hand-built synthetic nonlinear
system, AND a real fea_engine clamped-clamped `Beam2DCorotational` model
(`tests/test_nonlinear_rom.py`: 46 tests, `tests/test_nonlinear_rom_fea.py`:
8 tests) -- 137 tests total in the package, all passing, synced to the
canonical `computation-suite/rom_engine/` location. Phase 3, the
companion `fea_engine.nonlinear_solver.solve_nonlinear_transient()`
(Newmark-Newton implicit nonlinear transient integration), is also DONE
-- validated against `solve_transient_implicit()` in the linear limit
(machine precision), `solve_nonlinear_static()` in the quasi-static
limit (<0.1%), and undamped free-vibration energy conservation (<0.1%
drift over 5 periods); see `fea_engine/docs/nonlinear_transient_dynamics_roadmap.md`
for that package's own status. Phase 4, `rom_engine/nonlinear_dynamics.py`
(`integrate_newmark_surrogate()`, all three `correction` modes), is
also DONE -- validated against closed-form/scipy-ODE references at
the unit level and against a real fea_engine clamped-clamped
free-decay run (via Phase 3's `solve_nonlinear_transient()`) at the
real-fixture level; 161 tests total, synced to canonical. Phase 5
(`rom_engine/nnm.py`, `HarmonicBalanceSystem` + `solve_nnm_backbone()`)
is also DONE -- built from first-principles HBM/AFT theory (no literal
prototype to port from, since the earlier `mfs-nlrom-beam` reproduction
deliberately used time-integration + FFT period measurement instead of
a literal HBM solver, due to OCR-damaged source equations for that
paper section), validated against `frequency.FrequencyROM` (exact
`n_harmonics=1` cross-check), an independent scipy-ODE Duffing-
oscillator backbone, and a real fea_engine clamped-clamped beam
backbone measured the same time-integration + FFT way; 187 tests total
in the package, all passing, synced to canonical. Section 12
(`NeuralSurrogate`, a neural-network `ReducedForceModel` alternative to
`MultiFidelitySurrogate`, Wave 7 item 35 of `fea_engine`'s consolidated
roadmap) is implemented and unit-tested as of 2026-09-13, with its
torch-gated tests (14 of them) awaiting real pass/fail confirmation on
a machine with PyTorch installed -- see Section 12 for the full status.
Follows the format of
`docs/loewner_modal_identification_roadmap.md`,
`docs/frequency_domain_rom_roadmap.md`, and
`docs/phase4_error_bounds_greedy_roadmap.md`.

**Reference case, not the spec.** He, Yang, Li, Pang, Kan & Song (2023),
"A novel geometric nonlinear reduced order modeling method using
multi-fidelity surrogate for real-time structural analysis" (*Struct
Multidisc Optim* 66:233) -- the MFS-NLROM paper -- is used throughout
this document to ground design choices in a concrete worked example
(its flat-beam benchmark is the Phase 5 validation target below), and
its equations are cited by number where a design choice traces
directly to one. But every module specified here is written against
the *general* problem each piece solves, not against this one paper's
specific kernel choices, mode-selection convention, or beam geometry.
Where the paper made an arbitrary or paper-specific choice (e.g. fixing
the multi-fidelity scaling factor rho=1, or hand-deriving a Jacobian
for one kernel only), this document generalizes it and says so
explicitly, rather than porting the special case as if it were the
general one.

## 1. What kind of problem this actually is (why it's not "port MFS-NLROM")

A structural nonlinear ROM built by fitting a data-driven model of the
nonlinear restoring force in reduced coordinates is a **hybrid**, not a
new species: the linear part stays fully intrusive (a real modal or POD
basis, projected from the actual `K`/`M`/`C`), and only the nonlinear
correction term is a black-box surrogate fit from sampled input/output
pairs. Every published method in this family --

- **STEP / E-STEP / HR-STEP** (Enforced Displacement, Muravyov & Rizzi
  2003; Lee et al. 2021) -- polynomial (quadratic+cubic) nonlinear
  modal stiffness fit from enforced-displacement static solves,
  element-wise + hyper-reduced in the newer variants,
- **ICE, Implicit Condensation and Expansion** (Hollkamp & Gordon 2008)
  -- the same polynomial form fit from *applied-load* static solves
  with a membrane-augmented basis,
- **RBF-based STEP** (2024, radial-basis stiffness evaluation --
  confirms polynomial and RBF regression are two interchangeable
  regression backends for the *same* underlying problem, not
  competing architectures),
- **MFS-NLROM** (this project's reference case) -- an RBF surrogate of
  the nonlinear modal *force* (not stiffness coefficients) as a
  function of the *linear* modal displacement, using the multi-fidelity
  identity `q_nl = q_l - F_nl/Lambda` (Eq. 11) to avoid the
  Newton-Raphson step every polynomial-coefficient method still needs,

-- share one shape: **basis (intrusive) + nonlinear-term regression
(non-intrusive) + a closed-form or iterative way to recover the full
nonlinear reduced state from the fitted term.** This is exactly why
`rom_engine` is the right home (see the prior conversation's
conclusion): the intrusive half is `pod.py`/`galerkin.py` territory
already, and `rom_engine` already has a non-intrusive, data-fit module
(`loewner.py`) living next to the intrusive ones. What's being added
here is the missing *nonlinear* non-intrusive piece, generalized across
the whole family above instead of hardcoded to one paper's kernel
menu.

Consequence for the design: **one shared regression protocol** should
serve both the RBF surrogate and the polynomial/STEP/ICE family (Section
3), and **one shared reduced-dynamics/NNM machinery** (Sections 5-6)
should consume *any* model implementing that protocol -- including,
for validation, the true nonlinear force computed directly from
`fea_engine` (Section 7) with no surrogate at all. A design that only
worked for "the RBF fit to modal force" would fail the very validation
plan this document proposes (Shi & Mei / ICE / Enforced-Displacement as
baselines *of the same machinery*, not three separate one-off scripts).

## 2. Architectural fit: what's reused vs. what's new

Reused as-is, no changes:
- `pod.PodBasis` / an ordinary eigenbasis from a full-order modal
  solve -- either already produces the projection basis `V`/`phi` this
  whole family needs; nothing here requires changing basis extraction.
- `galerkin.GalerkinROM.reduce_system(K, M)` -- produces `Lambda`
  (`K_r`) and the modal mass/damping projections the reduced equations
  of motion need. The new modules consume `GalerkinROM`'s output; they
  do not reimplement projection.
- `metrics.py` -- gets one new, generic, method-agnostic function
  (`r_squared`, Section 8); it lives here for the same reason
  `modal_assurance_criterion` does (useful anywhere two response sets
  are compared, not tied to any one ROM method).

New (this document's scope):
- `rom_engine.sampling` -- OLHS design-of-experiments core, decoupled
  from any one loading convention (Section 4).
- `rom_engine.nonlinear_rom` -- the shared regression protocol plus two
  concrete model families: an RBF/multi-fidelity surrogate and a
  polynomial (Nash-form) ROM (Section 3).
- `rom_engine.nonlinear_dynamics` -- a reduced nonlinear time integrator
  parametrized by *any* model implementing the Section 3 protocol
  (Section 5).
- `rom_engine.nnm` -- a multi-harmonic-balance continuation solver, also
  parametrized by the Section 3 protocol (Section 6).
- `fea_engine.nonlinear_solver.solve_nonlinear_transient` -- the one
  full-order addition needed to generate genuinely nonlinear dynamic/NNM
  ground truth; specified in a companion document,
  `fea_engine/docs/nonlinear_transient_dynamics_roadmap.md`, since it
  belongs to the other package (Section 7 here only describes the
  *interface* `rom_engine` needs from it).

None of the new modules import `fea_engine`. Per `rom_engine`'s existing
design principle (stated in its own `__init__.py`), `fea_engine` is used
only in `tests/`/`examples/` to generate real validation fixtures (the
flat-beam model, Section 9) -- every library module here takes plain
numpy arrays / callables.

## 3. `rom_engine.nonlinear_rom` -- the shared regression protocol + two model families

### 3.1 The protocol

```python
# rom_engine/nonlinear_rom.py

class ReducedForceModel:
    """Protocol every nonlinear-force model in this family implements,
    whether it's a data-driven surrogate or a polynomial coefficient
    fit. `nonlinear_dynamics.py` and `nnm.py` are written ONLY against
    this protocol -- they never know or care whether the concrete model
    is `MultiFidelitySurrogate`, `PolynomialModalROM`, or a hand-wrapped
    call into a full nonlinear FE model's own internal-force assembly.

    Not an ABC by inheritance requirement -- any object with these three
    methods (predict always, jacobian and fit where meaningful) works,
    consistent with this package's existing duck-typed style (e.g.
    galerkin.GalerkinROM accepting either a PodBasis or a bare ndarray).
    """

    def fit(self, q_samples, F_samples, **kwargs):
        """Train the model from paired reduced-displacement / reduced
        nonlinear-force samples. Returns self (chainable, matching
        pod.PodBasis.fit()/galerkin.GalerkinROM.reduce_system())."""

    def predict(self, q):
        """Reduced displacement(s) -> predicted nonlinear force(s).
        q may be (r,) or (r, k) for a batch."""

    def jacobian(self, q):
        """dF_nl/dq at q, shape (r, r) (or a batch). Required by
        nnm.py's Newton solve and OPTIONALLY used by
        nonlinear_dynamics.py; a model that cannot provide one
        analytically may compute it by finite differences on predict()
        (see Section 6.2) -- callers should not assume analytic-only."""
```

### 3.2 `MultiFidelitySurrogate` -- generalizing MFS-NLROM's RBF fit

Generalizes Eq. 5-8, 11-15 beyond the paper's fixed choices:

```python
class MultiFidelitySurrogate(ReducedForceModel):
    """y_H(x) = rho * y_L(x) + d(x)  (Eq. 8), specialized here to
    y_L = q_l (linear reduced displacement, i.e. rho fixed to the
    identity map, matching Eq. 9-10's own q_l/q_nl relation) and d(x)
    an RBF fit of the nonlinear reduced force. Unlike the paper, `rho`
    is a free scalar-or-diagonal fit parameter, not hardcoded to 1 --
    Eq. 8 only reduces to the paper's Eq. 11 special case when rho=1;
    leaving it free costs nothing (one extra least-squares column) and
    stops the class from silently assuming that special case holds for
    every future use.

    kernel: name in KERNEL_REGISTRY (Table 1's four -- multiquadric,
    cubic, thin-plate-spline, gaussian -- registered by name so a fifth
    kernel is an ADDITION, not an edit, matching fea_engine's
    ELEMENT_REGISTRY/CONSTITUTIVE_REGISTRY convention). Every registered
    kernel supplies both psi(R) and its analytic dpsi/dR, so jacobian()
    is available for every kernel, not just MQ (the paper only derives
    Eq. 44 for MQ and is silent on the other three)."""

    def __init__(self, kernel="multiquadric", sigma=None):
        ...

    def fit(self, q_l_samples, F_nl_samples, q_nl_samples=None):
        """Eq. 13-15: builds H (Eq. 14) and solves beta = (H^T H)^-1 H^T q_nl
        for the DISPLACEMENT-domain fit (as literally written in Eq.
        13-15), OR, if q_nl_samples is None, fits d(x) directly against
        F_nl_samples (the FORCE-domain fit implied by Eq. 12 and used in
        Fig. 5/Table 2) -- both are legitimate readings of the paper and
        are kept as two supported fit TARGETS rather than picking one
        silently; see Section 9's validation plan for which is checked
        against Table 2's numbers."""

    def predict(self, q_l):
        """Eq. 11: q_nl = q_l - F_nl(q_l)/Lambda needs Lambda -- passed
        at construction or to predict(), not hidden global state."""


KERNEL_REGISTRY = {
    "multiquadric": (psi_mq, dpsi_mq),
    "cubic": (psi_cubic, dpsi_cubic),           # paper's "BH" kernel
    "thin_plate_spline": (psi_tps, dpsi_tps),
    "gaussian": (psi_gaussian, dpsi_gaussian),
}
```

### 3.3 `PolynomialModalROM` -- generalizing ICE / Shi & Mei / Enforced-Displacement

These three (plus E-STEP/HR-STEP's newer hyper-reduced variants) are the
*same* Nash-form fit (Appendix Eq. 45, 48-49) with a different
**training-data strategy**, not three different regression methods --
generalized here as a strategy object instead of three near-duplicate
scripts:

```python
class PolynomialModalROM(ReducedForceModel):
    """theta_r = sum_ij B_r(i,j) q_i q_j + sum_ijk A_r(i,j,k) q_i q_j q_k
    (Eq. 45), coefficients identified by least squares (Eq. 48-49) from
    whatever (q, F) pairs a TrainingStrategy produced. predict() is
    NON-closed-form (unlike the RBF surrogate): evaluating requires
    solving the nonlinear reduced equilibrium, so predict() performs its
    own small Newton-Raphson (the "still needs Newton-Raphson" property
    the paper repeatedly contrasts MFS-NLROM against -- reproduced
    honestly here, not hidden)."""

    def fit(self, q_samples, F_samples):
        """Assembles Z (Eq. 49) from q_samples, solves theta = Z+ @ F_samples
        per output mode independently."""

    def predict(self, q_target=None, F_ext=None, tol=1e-8, max_iter=30):
        """Either evaluate theta(q) directly (q_target given) or solve
        Lambda@q + theta(q) = F_ext for q via Newton-Raphson (F_ext
        given) -- the second form is what static/dynamic analysis
        actually needs."""

    def jacobian(self, q):
        """d theta/dq, analytic from the polynomial form -- always
        available, unlike the RBF surrogate's kernel-dependent case."""


class TrainingStrategy:
    """Pluggable training-DATA generation, not pluggable regression.
    Each strategy returns (q_samples, F_samples) given a full-order
    model callable and a basis -- PolynomialModalROM.fit() doesn't know
    or care which strategy produced its input."""

class AppliedLoadStrategy(TrainingStrategy):
    """Eq. 46-47: F_t = (1/m) M (phi_1 f_hat_1 + ... + phi_m f_hat_m),
    f_hat_r = (Q_max,r / phi_r,max) * lambda_r^2 -- the ICE / Shi & Mei
    convention. Solve the FULL nonlinear FOM at each F_t, project the
    resulting displacement onto the basis for q_samples."""

class EnforcedDisplacementStrategy(TrainingStrategy):
    """STEP's own convention: prescribe q directly (a displacement
    PATTERN in the shape of one or a combination of retained modes),
    solve the full-order model in DISPLACEMENT CONTROL (needs
    fea_engine.solve_nonlinear_displacement_control or equivalent), read
    back the reaction force. No load-scaling formula needed at all --
    this is the strategy that removes Eq. 46-47 from the picture
    entirely, which is exactly what distinguishes "enforced
    displacement" from "applied loads" in the literature (see Section
    1's citations)."""
```

`ICEROM`, `ShiMeiROM`, and `EnforcedDisplacementROM` (the paper's three
named baselines) are then thin, honestly-labeled convenience
constructors -- `PolynomialModalROM(basis=..., membrane_augmented=True)`
fit with `AppliedLoadStrategy` for ICE, the same class without membrane
augmentation for Shi & Mei, and `EnforcedDisplacementStrategy` for the
third -- not three copies of the same 40 lines of least-squares code.

## 4. `rom_engine.sampling` -- generalized OLHS training-design generator

Decoupled from the modal-force convention specifically, so a future
training strategy (Section 3.3) that samples something other than a
per-mode force scale can reuse the same core:

```python
# rom_engine/sampling.py

def optimal_lhs(n_samples, n_dims, criterion="maximin", n_iter=200, rng=None):
    """General-purpose OLHS sampler in [0,1]^n_dims (scipy.stats.qmc.
    LatinHypercube with an explicit optimization criterion and an
    EXPLICIT rng argument -- matching this package's existing "no hidden
    RNG state" convention from the Loewner roadmap, Section 2). Returns
    an (n_samples, n_dims) array in [0,1]; callers rescale to their own
    parameter ranges."""

def modal_force_samples(basis_freqs_hz, mode_shape_peaks, target_fracs,
                         n_samples, rng=None):
    """Eq. 16-17 / 46-47, generalized: given the retained modes'
    frequencies, mode-shape peak components, and a target displacement
    FRACTION per mode (e.g. 0.8-1.2x thickness, the paper's own
    "obvious nonlinear effect" range, Eq. 48's gamma_r), OLHS-samples
    the per-mode scale factors a_r and returns the resulting f_hat_r
    matrix (n_samples, n_modes) -- one thin, named function built ON
    TOP of optimal_lhs(), not a second sampling implementation."""
```

## 5. `rom_engine.nonlinear_dynamics` -- generalized reduced nonlinear time integration

Generalizes Eq. 19-33 to accept any `ReducedForceModel`, and separates
the METHOD (Newmark + surrogate-lag correction) from the MODEL (which
`ReducedForceModel` supplies `F_nl`/its Jacobian):

```python
# rom_engine/nonlinear_dynamics.py

def integrate_newmark_surrogate(Lambda, C_r, force_model, F_ext,
                                 q0, qdot0, dt, n_steps,
                                 alpha=0.25, beta=0.5,
                                 correction="sign_deviation"):
    """Eq. 21-33, decoupled from any one force_model implementation:
    - Eq. 24-30: standard Newmark predictor + effective-stiffness solve,
      exactly as written, with q_nl(t)/qdot_nl(t)/qddot_nl(t) the state.
    - Eq. 31: F_nl(t+dt) = force_model.predict(q_l(t+dt)) -- ANY
      ReducedForceModel plugs in here (MultiFidelitySurrogate,
      PolynomialModalROM evaluated in "direct" mode, or a thin wrapper
      around fea_engine's own nonlinear force assembly for a
      no-surrogate reference run -- see Section 7).
    - Eq. 32: `correction="sign_deviation"` reproduces the paper's own
      sign(qddot_nl(t))*(F_nl(t+dt)-F_nl(t)) term exactly;
      `correction="none"` (predict at face value, no lag correction) and
      `correction="fixed_point"` (iterate predict() a few times per step
      before accepting F_nl(t+dt), the earlier ad hoc reproduction's
      substitute -- kept as an option for the accuracy/cost tradeoff
      comparison, NOT as the default) are also selectable, so the
      dedicated MFS-NLROM scheme is one option among honestly-labeled
      alternatives, not the only path through the code.
    Returns (t, q_nl_hist, q_l_hist, F_nl_hist)."""
```

## 6. `rom_engine.nnm` -- generalized multi-harmonic-balance continuation

Eq. 34-44, generalized per the NLvib/MANLAB precedent (predictor-
corrector pseudo-arclength continuation with either analytic or
numerically-differentiated Jacobians -- see Sources):

```python
# rom_engine/nnm.py

class HarmonicBalanceSystem:
    """Eq. 35-39: assembles A(omega) (linear + damping operator in the
    harmonic domain) for a GIVEN Lambda, C_r, n_harmonics, and
    subharmonic multiplier -- pure linear-algebra bookkeeping, no
    nonlinear force involved yet. Deliberately mirrors
    frequency.FrequencyROM's own A(omega)=-omega^2 M + i*omega*C + K
    construction (same idea, harmonic-balance domain instead of a single
    frequency) -- reuses that module's Fourier/inverse-Fourier
    conventions where they overlap rather than inventing new ones."""

def solve_nnm_backbone(hb_system, force_model, q_amplitude_range,
                        n_points=20, jacobian="auto"):
    """Eq. 34, 40-44: h(z, omega) = A(omega) z + F_nl(z) = 0, Newton-
    solved at each target amplitude, continued across q_amplitude_range
    to trace the frequency-energy backbone.

    jacobian="auto" (default): uses force_model.jacobian() when the
    model provides one (analytic, e.g. PolynomialModalROM always,
    MultiFidelitySurrogate for its registered kernels -- Section 3.2);
    falls back to a numerical directional-derivative (central
    differences on force_model.predict(), i.e. an AFT-style Jacobian --
    see NLvib's own precedent, Sources) for any model that doesn't.
    This removes the paper's own admitted limitation (Eq. 44 derived
    for the MQ kernel only, "may be inaccurate" for the rest, Sect.
    4.1's NNM discussion) as an ARCHITECTURAL default rather than a
    per-kernel derivation debt every new kernel/model would otherwise
    incur.

    Works identically whether force_model is a surrogate, a polynomial
    ROM, or a full-order reference (Section 7) -- backbone comparison
    across all three (the paper's own Fig. 9 layout) is then "call this
    once per model," not three separate solvers."""
```

## 7. The one `fea_engine` dependency: full-order nonlinear reference generation

`rom_engine`'s own modules above never import `fea_engine`. But
producing a genuine "nonlinear FE" ground-truth curve (Figs. 7/9's
reference lines) needs a nonlinear TRANSIENT solve, which does not
exist anywhere in `fea_engine` today (both `solve_transient_implicit`
and `solve_transient_explicit` factor a fixed LINEAR `Keff` once and
never re-touch it -- confirmed by reading `solver.py` directly). That
addition, `nonlinear_solver.solve_nonlinear_transient()` (a Newmark
predictor + per-step Newton correction reusing the existing
`assemble_internal_force`/`assemble_tangent_stiffness`), is specified
in a companion document: `fea_engine/docs/nonlinear_transient_dynamics_roadmap.md`.

The only thing `rom_engine`'s test suite needs from it is a thin
adapter satisfying Section 3.1's protocol:

```python
# tests/ or examples/ only -- NOT part of the rom_engine library
class FullOrderForceModel:
    """Wraps an fea_engine FESystem so it can be dropped into
    integrate_newmark_surrogate()/solve_nnm_backbone() as the
    "nonlinear FE, no surrogate" reference case, with NO surrogate
    module needing to know fea_engine exists."""
    def predict(self, q_l):
        u_full = basis.V @ q_l
        F_int = fesystem.assemble_internal_force(u_full, mat)
        return basis.V.T @ F_int
    def jacobian(self, q_l):
        u_full = basis.V @ q_l
        Kt = fesystem.assemble_tangent_stiffness(u_full, mat)
        return basis.V.T @ Kt @ basis.V
```

## 8. `rom_engine.metrics` addition

```python
def r_squared(y_true, y_pred):
    """Eq. 18, generic: 1 - sum((y-yhat)^2) / sum((y-mean(y))^2).
    Method-agnostic (works on force samples, displacement samples, or
    NNM amplitude samples alike) -- every accuracy table/figure in the
    reference paper (Tables 2-3, Fig. 8, the NNM comparisons) uses this
    one statistic."""
```

## 9. Validation plan

- **Unit level, per module, synthetic fixtures first** (matching every
  prior `rom_engine` module's own pattern): `sampling.optimal_lhs`
  against known LHS space-filling properties; each `KERNEL_REGISTRY`
  entry's `dpsi/dR` checked against finite differences;
  `PolynomialModalROM.jacobian` checked against finite differences on
  `predict()`; `HarmonicBalanceSystem`'s `A(omega)` checked to reduce to
  `frequency.FrequencyROM`'s own `A(omega)` at a single harmonic
  (`n_harmonics=1` should recover ordinary linear FRF machinery exactly
  -- a strong cross-module consistency check `frequency.py` didn't
  previously have a partner for).
- **Real fixture -- the flat-beam reference case**, built with
  `fea_engine` exactly as `flat_beam_fem.py` (this project's existing
  script) already does: 40-element clamped-clamped beam, modes {1,3,5}
  as the retained basis (confirmed against Fig. 5's own axis labels --
  not the body text's literal "first three modes," see the prior
  conversation), N=30 OLHS training / Nt=21 test static cases via
  `sampling.modal_force_samples`.
- **Cross-model agreement, not just paper-number matching**: fit
  `MultiFidelitySurrogate` (all four kernels) AND `PolynomialModalROM`
  (both `AppliedLoadStrategy` and `EnforcedDisplacementStrategy`) on the
  SAME training set, and require they agree with each other and with
  direct nonlinear FE on held-out test cases to a stated tolerance --
  reproducing Table 2's actual R^2 numbers exactly is a stretch goal
  (this project's existing fidelity notes on OLHS-range sensitivity
  apply here too), but relative ranking (RBF close to ICE, both far
  above the reference's own admittedly-weak Gaussian kernel) is a firm
  requirement.
- **`integrate_newmark_surrogate`'s three `correction` modes**,
  compared against `FullOrderForceModel`-driven
  `fea_engine.solve_nonlinear_transient` ground truth, at both of the
  paper's own initial conditions (`F_nl(0)=-2068 Pa`,
  `x_nl(0)=0.5*th*phi_1/max(phi_1)`) -- reproducing Table 3's ranking
  (`sign_deviation` should outperform `fixed_point`) is the check, not
  the literal R^2 values.
- **`solve_nnm_backbone`'s jacobian="auto"`** run once with each
  `ReducedForceModel` family AND with `FullOrderForceModel` (the
  no-surrogate reference), confirming the reference and the polynomial
  ROM's backbones closely track each other (Fig. 9's own finding) while
  documenting where the RBF surrogate's backbone degrades, exactly as
  honestly as the existing project's fidelity notes already do for the
  {1,3,5}-basis dynamic/backbone results.

## 10. Phased roadmap

**Phase 1 -- `sampling.py` + `metrics.r_squared` (small, standalone,
unblocks everything else's training data and every later accuracy
check). DONE.**
`optimal_lhs()` implements the maximin criterion as a best-of-`n_iter`
candidate-LHS selection (scipy's `LatinHypercube` per candidate, no
hidden `rng` default); `modal_force_samples()` generalizes Eq. 16-17/
46-47 with an explicit `reference_scale` parameter instead of the
paper's hardcoded thickness. `metrics.r_squared()` added per Eq. 18.
Validated: genuine-LHS stratification property, a guaranteed (same-seed)
inequality that more candidate iterations never worsens the maximin
score, rng reproducibility, and (for `modal_force_samples`) that the
sampled displacement targets land inside the requested per-mode
fraction range. No `fea_engine`-based validation was needed for this
phase (it's pure sampling/statistics, nothing structural to check yet).

**Phase 2 -- `nonlinear_rom.py` core**: the `ReducedForceModel`
protocol, `MultiFidelitySurrogate` (all four kernels, with newly-derived
analytic Jacobians the earlier `mfs-nlrom-beam` prototype did not need),
`PolynomialModalROM` (Nash-form + Newton solve + jacobian, with
`ICEROM`/`ShiMeiROM`/`EnforcedDisplacementROM` convenience constructors)
+ both `TrainingStrategy` variants (`AppliedLoadStrategy`,
`EnforcedDisplacementStrategy`). DONE.**
Ported faithfully from this project's own earlier, validated
`mfs-nlrom-beam` skill (`mfs_nlrom.py`'s `RBFSurrogate`, `ice_rom.py`'s
`IceRom`) -- including the confirmed protocol-domain split
(`MultiFidelitySurrogate` fits against `q_l`, `PolynomialModalROM`
against `q_nl`) and the exact kernel formulas from that validated code.
Validated three ways: (1) synthetic ground-truth cubic/Nash-form
functions with finite-difference-checked analytic Jacobians
(`test_nonlinear_rom.py`, 46 tests); (2) a hand-built synthetic
nonlinear (decoupled cubic-spring) full-order model exercising both
`TrainingStrategy` variants' `fom_solver` callable contract; and (3) a
REAL fea_engine clamped-clamped `Beam2DCorotational` model
(`test_nonlinear_rom_fea.py`, 8 tests) -- `AppliedLoadStrategy` driving
`fea_engine.nonlinear_solver.solve_nonlinear_static()` directly (no
Phase 3 solver needed for this, since static training data suffices),
reaching held-out R² > 0.95 (RBF, all 3 non-cubic kernels tested) /
> 0.99 (polynomial) on genuinely nonlinear response data. This alone is
a defensible stopping point, mirroring how Phase 2 was the main
deliverable in the Loewner roadmap.

**Phase 3 -- `fea_engine.nonlinear_solver.solve_nonlinear_transient`**
(companion document/package). DONE.** Newmark-Newton implicit nonlinear
transient integration, re-factoring `K_eff(d) = K_T(d) + a0c*M + a1c*C`
every Newton iteration of every step. Validated per that document's own
plan: linear-limit regression against `solve_transient_implicit()`
(machine precision), quasi-static-limit agreement against
`solve_nonlinear_static()` (<0.1%), and undamped free-vibration energy
conservation (<0.1% drift over 5 periods) --
`fea_engine/tests/test_nonlinear_transient.py`, 4 tests, synced to the
canonical `computation-suite/fea_engine/` location. This unblocks
Phases 4-5 below: `rom_engine.nonlinear_dynamics`/`nnm` can now be
validated against a real nonlinear-FE dynamic ground truth (via
`solve_nonlinear_transient` as the `fom_solver` callable) rather than
only against synthetic systems or each other.

Deliberately deferred (documented, not silently dropped): the paper's
free-`rho` generalization of the multi-fidelity identity (Eq. 8) --
only the `rho=1` specialization is implemented, because the source
PDF's OCR-damaged equations couldn't recover the general form reliably
(same gap the `mfs-nlrom-beam` skill's own fidelity notes record).
`EnforcedDisplacementStrategy` was validated only against the synthetic
system, not fea_engine, because `solve_nonlinear_displacement_control()`
is single-DOF-control only -- a real fea_engine API gap for a
full-pattern-controlled training strategy, not a `nonlinear_rom.py` bug.

**Phase 4 -- `nonlinear_dynamics.py`**: `integrate_newmark_surrogate()`,
all three `correction` modes (`"none"`, `"fixed_point"`,
`"sign_deviation"`, default). DONE.**
**2026-09-24 update: the default is now `"fixed_point"`.** `"sign_deviation"`
diverges on the NonLin-HyROM Case 2 cylindrical shell (non-finite by
t~0.30 s for every mode set, earlier than the uncorrected `"none"`).
The cause is its `sign(qddot)` factor, which turns the correction into
an anti-Picard step for decelerating modes. The literal Eqs. 28-32
(they do extract with pymupdf) diverge the same way. See
`nonlinear_dynamics.py`'s docstring.
`"fixed_point"` is ported faithfully from this project's own earlier,
validated MFS-NLROM dynamic reproduction (`mfs-nlrom-beam` skill's
`scripts/dynamic_newmark.py`, `modal_newmark_mfs`, `n_fixed_point=4`).
`"sign_deviation"` is this project's own honest, clearly-labeled
reconstruction of the paper's Eq. 32 idea (NOT a literal transcription
-- that equation range's OCR-damaged source text is the same gap
`dynamic_newmark.py`'s own docstring already records): a single
predict-then-correct pass (two linear solves, two surrogate
evaluations, still zero Newton iterations) applying the deviation
between a lagged-force prediction and a fresh one, directed by the
previous step's acceleration sign. Validated per Section 9 against
Phase 3's ground truth: unit-level, the linear limit (zero or exactly-
linear force models) reduces EXACTLY to plain Newmark regardless of
`correction` (`fixed_point` to near machine precision); a `q_l`-domain
linear case and a genuinely nonlinear cubic-spring case are both
checked against independent `scipy.integrate.solve_ivp` references
(`tests/test_nonlinear_dynamics.py`, 17 tests). Real-fixture: a
clamped-clamped `Beam2DCorotational` free-decay run, `MultiFidelitySurrogate`
and `PolynomialModalROM` both trained via `AppliedLoadStrategy`, compared
against `fea_engine.nonlinear_solver.solve_nonlinear_transient()` as
full-order ground truth (`tests/test_nonlinear_dynamics_fea.py`, 7
tests) -- short-window (~2 period) R² > 0.9 and matching amplitude-
decay envelopes for both model families, with the expected,
honestly-reported phase-drift-driven R² degradation over the full
8-period window (a known property of fitted-vs-exact nonlinear force
models, not a bug). 161 tests total in the package, all passing,
synced to the canonical `computation-suite/rom_engine/` location.

**Phase 5 -- `nnm.py`**, `HarmonicBalanceSystem` + `solve_nnm_backbone`
with the `jacobian="auto"` fallback. DONE.**

Built from first-principles multi-harmonic-balance (HBM) + Alternating
Frequency-Time (AFT) theory generalizing Eq. 34-44, per the NLvib/
MANLAB precedent (Sources): `HarmonicBalanceSystem` represents a
periodic state as real trigonometric coefficients `Z` and assembles the
linear-in-`Z` operator `A(omega)`, deliberately mirroring
`frequency.FrequencyROM`'s own complex `A(omega) = -omega^2*M +
i*omega*C + K` construction; the AFT nonlinear-force evaluation and its
Jacobian (chain-ruled through the `q_l = q_nl + F_nl(q_l)/Lambda`
domain translation via the implicit function theorem when needed) live
in `solve_nnm_backbone()` itself, which Newton-continues the
periodic-orbit family across a prescribed master-mode amplitude range.
Unlike Phases 2/4, there was no literal HBM prototype to port from --
the earlier `mfs-nlrom-beam` skill's own `backbone_nnm.py` explicitly
did NOT implement the paper's HBM equations (that section's source PDF
did not survive OCR), instead measuring the backbone the honestly
simpler, slower way (undamped free-vibration time integration + FFT-
based period measurement) and flagging the simplification directly in
its own module docstring. This module fills that gap with a genuine
HBM/AFT implementation rather than reproducing the same workaround
again.

Two scope simplifications, stated explicitly (see the module's own
docstring for the full reasoning): `solve_nnm_backbone()` targets the
classical UNDAMPED, UNFORCED autonomous periodic-orbit backbone
(Peeters et al. 2009's own convention, and what the reference paper's
Fig. 8/9 backbones are) even though `HarmonicBalanceSystem.apply_A()`
itself does support a damping term `C_r`; and continuation is
NATURAL-PARAMETER (prescribed-amplitude, Newton-solved, seeded from the
previous point) rather than full pseudo-arclength, sufficient for the
non-folding (monotonically hardening/softening) backbones this
project's validation targets but not for a backbone with a genuine
fold. Deriving the correct degrees-of-freedom bookkeeping for the
Newton continuation (which two rows of `h(Z,omega)=0` to fix/drop,
given the autonomous problem's 1-D phase-symmetry-driven rank
deficiency) required first-principles reasoning about the periodic-
orbit solution manifold, not a formula copied from the paper.

Validated four ways, from the most basic linear-algebra self-
consistency up to a real structural model, per Section 9's plan:
(1) the AFT trigonometric transform's exact invertibility and
`apply_A`/`apply_dA_domega` self-consistency against their own dense
matrix forms; (2) an EXACT cross-check against `frequency.FrequencyROM`
at `n_harmonics=1` -- the real cos/sin harmonic-balance solve
reproduces that module's own complex FRF solve's real/imaginary parts
directly, a genuine cross-module consistency check `frequency.py`
didn't previously have; (3) an independent `scipy.integrate.solve_ivp`
+ peak-measured-period ground truth for the classical Duffing
oscillator, matched to <1% across 6 amplitude points with the correct
hardening trend, plus a `domain="q_l"` linear-limit check against a
closed-form effective-stiffness frequency (`tests/test_nnm.py`, 17
tests); and (4), the strongest check, a REAL fea_engine clamped-clamped
`Beam2DCorotational` backbone, `MultiFidelitySurrogate` and
`PolynomialModalROM` both trained via `AppliedLoadStrategy`, with an
INDEPENDENT ground truth measured the same time-integration + FFT way
`backbone_nnm.py` validated its own backbone -- both model families
match that measurement to <10% at every point, correctly reproduce the
hardening trend, and sit above the linear natural frequency
(`tests/test_nnm_fea.py`, 9 tests). 187 tests total in the package, all
passing, synced to the canonical `computation-suite/rom_engine/`
location.

Each phase confirmed complete (tests passing, synced to the canonical
`computation-suite/rom_engine/` location) before the next starts, per
this project's established workflow. All 5 phases of this roadmap are
now complete; Section 11 below documents a Phase 6 addendum to
`nnm.py` taken up afterward, at the user's request.

## 11. Phase 6 addendum: pseudo-arclength continuation
(`solve_nnm_backbone_arclength`)

Phase 5's `solve_nnm_backbone()` is NATURAL-PARAMETER continuation:
amplitude is prescribed, Newton solves for the rest. That fails at a
fold -- an amplitude value where the true backbone curve turns back on
itself, so no solution exists for amplitude values past the turn no
matter how the equations are solved. This addendum adds a second
continuation driver, `solve_nnm_backbone_arclength()`, built on the
classical fix for exactly this failure mode: pseudo-arclength
continuation (Keller 1977, Sources).

**Design.** Given the same HBM/AFT residual `h(Z,omega)=0` Phase 5
already assembles, pseudo-arclength continuation prescribes ARCLENGTH
DISTANCE from the previous converged point instead of prescribing
amplitude, via a linear constraint `t_hat . (v - v_prev) - ds = 0`
(`t_hat` a unit tangent direction) appended to `h`, and solves the
AUGMENTED system for every unknown jointly -- including the amplitude
that natural-parameter continuation would otherwise have fixed. This
can pass through a fold because nothing in the augmented system
requires amplitude to be monotonic; only the arclength distance is.

Three implementation choices, each stated in `nnm.py`'s own
docstrings and repeated here for the roadmap record:

- **Secant predictor, not the true local tangent.** The full Keller
  method computes `t_hat` from a null-space solve of the linearized
  system at each point. This implementation instead uses the SECANT
  direction between the two most recently converged points. This is a
  well-precedented simplification: the corrector's arclength
  constraint only needs `t_hat` to be roughly transversal to the
  curve, not exactly tangent to it, since the augmented Newton
  corrector re-converges onto the true curve regardless of exactly
  where the predictor lands.
- **Bootstrap from two natural-parameter points.** No secant exists
  before two points are known, so the first two points of every trace
  are found by ordinary prescribed-amplitude Newton solves (reusing
  Phase 5's own inner loop, refactored out as `_newton_fixed_amplitude`
  with a full-suite zero-regression check before any new code was
  added). Genuine arclength stepping begins from the third point on.
- **Omega-rescaling to fix a real mixed-units failure.** The raw
  augmented unknown vector mixes amplitude-sized components (`Z`,
  `a1`, typically ~0.005-0.03) with `omega` (typically ~10). An
  unscaled Euclidean arclength metric is dominated by whichever has
  larger raw magnitude -- in practice always `omega` -- which starves
  the amplitude direction of step budget. This was not a hypothetical
  concern: an early unscaled version of this code measurably advanced
  amplitude only from 0 to 0.0089 over 6 points at `ds=0.005`, versus
  natural-parameter continuation's fixed 0.005-per-point steps over
  the same range. The fix rescales `omega` by the master mode's own
  linear natural frequency (`omega_scale = sqrt(Lambda[master_mode])`
  by default) and does all tangent/predictor/constraint bookkeeping on
  the rescaled vector, converting back to physical `omega` only when
  evaluating the equilibrium residual and Jacobian (with the
  Jacobian's last column chain-rule-scaled back). After the fix, the
  same comparison tracked amplitude from 0.005 to 0.0263 over a
  comparable number of points, agreeing with natural-parameter
  continuation's own trace to <2e-4 relative error on a densely
  sampled overlap.

  This is a different fix than fea_engine's own arc-length/Riks solver
  uses for the analogous displacement/load-factor scale mismatch
  (`fea_engine/src/fea_engine/nonlinear_solver.py`,
  `solve_nonlinear_arc_length`) -- worth recording accurately since an
  earlier draft of this code's docstring claimed otherwise. fea_engine
  uses a Crisfield-style spherical arc-length constraint built purely
  from the displacement increment's own norm, EXCLUDING the load
  factor from the metric entirely and solving for it afterward via a
  separate quadratic root selection. That works there because
  displacements dominate and the load factor is a minor bookkeeping
  unknown. NNM backbone continuation cannot use the same trick: both
  `omega` and amplitude are physically meaningful, actively-varying
  unknowns along the curve, and excluding either from the arclength
  metric would bias the traced path toward the other. Rescaling both
  into comparable units, rather than excluding one, is the correct fix
  for this problem.

**Outcome.** Implemented (`_augmented_residual_jacobian`,
`_augmented_residual_jacobian_scaled`, `_arclength_corrector`,
`solve_nnm_backbone_arclength`) and validated three ways, all passing
(`tests/test_nnm.py`, 26 tests in that file alone; 244 tests package-
wide):

1. **A fully analytic fold, hand-verified with total certainty.**
   The unit circle `x^2+y^2=1`, with `x` playing the role of
   "prescribed amplitude": the exact fold locations (`x=+-1`, where no
   real `y` exists for `|x|>1`) are known algebraically, not just
   numerically. The core predictor/corrector/step-halving algorithm
   traces 260 points through BOTH folds, residual at machine precision
   (2.2e-16) throughout, `x` reaching both `+1.0` and `-1.0` within
   `atol=1e-2` -- direct proof the algorithm turns a fold correctly,
   independent of any NNM-specific machinery.
2. **Agreement with `solve_nnm_backbone` on a non-folding case.** A
   Duffing-oscillator backbone traced both ways (60 points each,
   `ds=0.002`) agrees to <1e-3 relative error over the overlapping
   amplitude range (densely sampled, `np.interp`-matched) -- confirms
   the new driver reduces to the same physics as the validated Phase 5
   driver when there is no fold to navigate.
3. **Exact reduction check on a genuinely 2-mode system.** A
   `TwoCoupledDuffing` system (two cubic oscillators, cubic
   cross-coupling) bootstrapped from a zero-seeded guess correctly
   finds the invariant PLANAR branch, where the satellite mode's own
   amplitude `a2` stays exactly zero (<1e-9) throughout an 80-point
   trace spanning `a1` from +0.19 through 0 to -0.58; on that branch,
   the traced solution matches the closed-form single-DOF Duffing
   governing relation `(w1^2-omega^2)*a1 + 0.75*c1*a1^3 = 0` to <1e-6
   relative -- confirms the augmented multi-mode machinery collapses
   correctly onto a known-exact reduced case rather than introducing
   spurious multi-mode coupling.

**Honestly incomplete: a real, hand-verified NNM fold.** Beyond the
analytic circle proof above, this addendum also explored whether the
`TwoCoupledDuffing` system (and related synthetic 2-mode variants)
exhibits a genuine amplitude fold reachable by this solver, as a
physically-motivated demonstration beyond the circle toy problem.
Extensive parameter exploration found real multi-branch complexity --
including cases where naive natural-parameter continuation loses track
of a branch and jumps discontinuously, exactly the failure mode this
addendum exists to fix -- but did not, within the scope of this
addendum, conclusively pin down a fully characterized, hand-verified
fold on a real NNM backbone (e.g. the kind expected near an internal
resonance). This is reported honestly as an incomplete finding rather
than overclaimed, matching this project's established practice
(`scm_lp.py`'s Phase 4e negative finding, `passivity.py`'s honest
violation-finding). It does not block this addendum's correctness
claim, which rests on the three independent checks above (an exact
analytic fold, exact agreement with the validated Phase 5 driver, and
an exact reduction to a known closed-form case) -- it is a natural next
validation step, not a gap in the algorithm itself.

## 12. `NeuralSurrogate` -- neural-network `ReducedForceModel` (Wave 7 item 35)

STATUS: implemented and unit-tested 2026-09-13; `NeuralSurrogate`-specific
tests are torch-gated (this sandbox cannot `pip install torch` -- proxy
blocks the PyTorch wheel index) and still need real pass/fail
validation on a machine with PyTorch installed, per this project's
established sandbox-then-user-machine convention (see `fea_engine`'s
`torch_sparse_solver.py`/`autograd_tangent.py` for the prior instances
of the same split).

**Naming correction from the source roadmap item.** `fea_engine/docs/
consolidated_future_roadmap.md`'s Wave 7 item 35 (itself drawn from
`tensormesh_comparative_analysis.md` Section 6.2, written before this
file's Section 3 architecture existed to inspect) proposed a class
called "NeuralSurrogateTrainingStrategy". That name doesn't fit the
real architecture: `TrainingStrategy` (Section 3) is data-generation
ONLY -- it produces `(q_l, q_nl, F_nl)` triples and is deliberately
decoupled from which regression method later consumes them. A neural
regression model is therefore a new `ReducedForceModel`, a sibling of
`MultiFidelitySurrogate` and `PolynomialModalROM`, not a new
`TrainingStrategy`. Built and named accordingly: `NeuralSurrogate`,
in `nonlinear_rom.py` next to the other two model families, consuming
the exact same `(q_l_samples, F_nl_samples)` contract
`MultiFidelitySurrogate.fit()` does (drop-in replacement, confirmed by
`test_nonlinear_rom.py::TestNeuralSurrogate::
test_downstream_applied_load_strategy_data_fits_neural_surrogate`
feeding `AppliedLoadStrategy`-generated data straight into it).

**Design.** A small MLP (`torch.nn.Sequential`, configurable
`hidden_sizes`/`activation` from a `{"tanh","relu","silu"}` registry),
trained by Adam + MSE on Z-SCORE-NORMALIZED inputs and outputs. The
normalization matters for correctness here in a way it doesn't for
`MultiFidelitySurrogate`: an RBF kernel's `sigma` is itself fit to the
data's own length scale, but a plain MLP's gradient-descent training
is sensitive to raw input/output scale, so `_fit_normalization()`/
`_normalize()`/`_denormalize()` (plain NumPy, factored out so they run
UNCONDITIONALLY without torch -- see `test_nonlinear_rom.py::
TestNormalizationHelpers`, 3 tests, all passing in every environment
including this sandbox) are a real modeling choice, not cosmetic
plumbing. `jacobian()` uses row-by-row `torch.autograd.grad` (one
reverse-mode pass per output component) with the normalization's
chain-rule scaling applied explicitly afterward -- the same technique
`fea_engine.autograd_tangent` already uses for FE element tangent
stiffnesses, now reused for a ROM force-surrogate Jacobian instead of
an FE internal-force Jacobian; documented as a deliberate stylistic/
technical parallel in the class's own docstring.

**Why offer this alongside the RBF/polynomial families rather than
replacing them.** `MultiFidelitySurrogate`'s kernel matrix is O(N^2) to
build and O(N^3) to factor once at fit time (fine for the paper's
scale, tens of training points); an MLP is O(N) per training epoch and
O(1) at predict time regardless of how many training points were used,
at the cost of needing an iterative (non-closed-form) fit and losing
the RBF's analytic kernel-derivative Jacobian in favor of an autograd
one. Neither dominates the other -- this is a genuine trade-off,
offered as an additional registry entry, matching this package's
stated extension-point design principle (`__init__.py`: "extending the
package means ADDING a function/class, not editing existing ones").

**Validation.** Synthetic cubic ground-truth fit/predict/R^2 (R^2 >
0.99 target, matching `TestMultiFidelitySurrogate`'s own bar),
jacobian-vs-central-finite-difference, single-point vs. batch shape
contracts, before-fit/wrong-shape/wrong-mode-count error checks, a
fit-reproducibility-under-fixed-seed check, and the cross-strategy
check described above -- 12 tests total in
`tests/test_nonlinear_rom.py::TestNeuralSurrogate`. Real-fixture
validation against the same clamped-clamped `Beam2DCorotational`
training/held-out split `TestMultiFidelitySurrogateOnRealBeam` uses
lives in `tests/test_nonlinear_rom_fea.py::TestNeuralSurrogateOnRealBeam`
(2 tests: held-out R^2 > 0.9, jacobian-vs-finite-difference at a
training point). All 14 NeuralSurrogate-specific tests are currently
SKIPPED in this sandbox (torch not importable) and PASS/FAIL status is
unconfirmed until run on a machine with PyTorch -- everything else in
the package (254 tests across `rom_engine/tests/`, including the new
unconditional `TestNormalizationHelpers`) was re-run in full after this
change and shows zero regressions.

## Sources

- He, X., Yang, L., Li, K., Pang, Y., Kan, Z., Song, X. (2023). "A novel
  geometric nonlinear reduced order modeling method using multi-fidelity
  surrogate for real-time structural analysis." *Structural and
  Multidisciplinary Optimization*, 66:233. (Reference case throughout;
  full text re-extracted directly from the source PDF via PyMuPDF for
  this document, recovering Eq. 16-17 and 21-33 which an earlier OCR
  pass on this project had left unrecoverable; Fig. 5 rendered and
  inspected directly to confirm the {1,3,5} retained-mode reading.)
- Muravyov, A.A. & Rizzi, S.A. (2003). "Determination of nonlinear
  stiffness with application to random vibration of geometrically
  nonlinear structures." *Computers & Structures*, 81(15). (STEP.)
- Hollkamp, J.J. & Gordon, R.W. (2008). "Reduced-order models for
  nonlinear response prediction: Implicit condensation and expansion."
  *J. Sound and Vibration*, 318(4-5). (ICE, this paper's own Appendix
  source.)
- Lee, K. et al. (2021). Hyper-reduction stiffness evaluation procedure
  (HE-STEP ROM) via element-wise STEP + hyper-reduction. *Comput Mech*
  67. (E-STEP/hyper-reduction lineage, Section 1's generalization
  argument.)
- "Non-intrusive reduced-order modeling for nonlinear structural systems
  via radial basis function-based stiffness evaluation procedure"
  (2024), *Computers & Structures* (ScienceDirect). Confirms RBF and
  polynomial regression are interchangeable backends for the same STEP
  problem -- direct support for Section 3's shared-protocol design.
  https://www.sciencedirect.com/science/article/abs/pii/S0045794924002293
- Krack, M. & Gross, J. -- *NLvib* (Matlab tool for nonlinear vibration
  analysis via harmonic balance, predictor-corrector Newton continuation
  with AFT/analytic Jacobians). Precedent for Section 6's
  `jacobian="auto"` design.
  https://github.com/maltekrack/NLvib
- Cochelin, B. et al. -- *MANLAB* (harmonic-balance + asymptotic
  numerical method continuation). Precedent for treating HB continuation
  as a reusable numerical core independent of the nonlinear model being
  continued. https://www.researchgate.net/publication/331998890
- Keller, H.B. (1977). "Numerical solution of bifurcation and nonlinear
  eigenvalue problems." In *Applications of Bifurcation Theory*
  (Rabinowitz, ed.), Academic Press. Original pseudo-arclength
  continuation reference, Section 11's `solve_nnm_backbone_arclength`.
- `rom_engine/__init__.py`, `docs/loewner_modal_identification_roadmap.md`,
  `docs/frequency_domain_rom_roadmap.md` (this package) -- format,
  phasing, and "intrusive vs. non-intrusive" precedent this document
  follows directly.
