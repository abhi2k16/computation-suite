<div align="center">

# rom_engine

**Reduced-order modeling for structural analysis: POD, Galerkin, affine, frequency-domain, state-space, nonlinear**

![version](https://img.shields.io/badge/version-0.1.0-7c3aed?style=for-the-badge)
![status](https://img.shields.io/badge/status-alpha-f59e0b?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.9+-3776ab?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-1.22+-013243?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-1.8+-0054a6?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-optional-ee4c2c?style=for-the-badge)
![license](https://img.shields.io/badge/license-MIT-16a34a?style=for-the-badge)

</div>

## 🧭 Contents

| | |
|---|---|
| 🎯 [Design principle: fea_engine-agnostic core](#s-1) | 🧪 [Running the tests](#s-6) |
| ✅ [What's implemented (linear core)](#s-2) | 🧪 [Running the examples](#s-7) |
| ⚙️ [Installation](#s-3) | ⚠️ [A certified-bound caveat (read before using `scm.py`)](#s-8) |
| 🚀 [Quick start](#s-4) | 🔢 [A dtype note (why this matters more than it sounds)](#s-9) |
| 🗂️ [Package layout](#s-5) | 🗺️ [Roadmap](#s-10) |

---


A small, from-scratch reduced-order-modeling (ROM) package for
structural engineering analysis. It is the companion package to
[`fea_engine`](../fea_engine): fea_engine assembles and solves
full-order finite element models, rom_engine compresses and
accelerates them.

<a id="s-1"></a>

## 🎯 Design principle: fea_engine-agnostic core

`rom_engine`'s library code (`pod.py`, `galerkin.py`, `affine.py`,
`frequency.py`) never imports `fea_engine`. Every function and class
works on plain `numpy` arrays -- a stiffness matrix `K`, a mass matrix
`M`, a load vector `F`, a matrix of snapshot vectors. This means
rom_engine works with matrices assembled by fea_engine, by any other FE
code, or by a hand-built toy model -- it makes no assumptions about
where `K`/`M`/`F` came from. Complex-valued arrays (needed by
`frequency.py`'s `A(omega) = -omega^2*M + i*omega*C + K`) are handled
transparently throughout -- every array-ingesting method infers
real-vs-complex dtype from its input rather than forcing `float`.

`fea_engine` is used in exactly one place in this package: as the
source of realistic, physically meaningful test fixtures
(`tests/fea_fixtures.py`) and worked examples (`examples/`). Every
correctness check in this package's test suite is validated against a
real fea_engine structural model -- a cantilever beam, a two-region
beam with independent material properties -- not just against
synthetic matrices, so the numbers in the test output mean something
physically, not just algebraically.

### Intrusive vs. non-intrusive

Every module above (`pod`/`galerkin`/`affine`/`frequency`/`greedy`/`scm`)
is **intrusive**: each is handed the actual system matrices `K`/`M`/`C`
(or an affine decomposition of them). `loewner.py`/`screening.py` are
the package's first **non-intrusive** capability: `LoewnerROM.fit()`
identifies natural frequencies, damping ratios, and mode shapes
directly from sampled complex frequency-response data `x(omega)` --
it never sees `K`, `M`, or `C`, and its signature makes that a checked
property, not just a design intent (see `test_loewner.py`). This makes
it usable on real measured/experimental vibration data or a
proprietary/black-box model's exported harmonic response, not just on
a system this package (or any FE code) can assemble directly. See
`docs/loewner_modal_identification_roadmap.md` for the full design
background.

<a id="s-2"></a>

## ✅ What's implemented (linear core)

| Module | Purpose |
|---|---|
| `pod.py` | Proper Orthogonal Decomposition: extract a reduced basis from a matrix of snapshot vectors. Standard (Euclidean/SVD) and mass-weighted (M-orthonormal, via a Cholesky-based algorithm) variants; truncate by explicit rank or an energy-capture threshold. |
| `galerkin.py` | Intrusive Galerkin projection: project a full-order linear system `(K, M, F)` onto a reduced basis and solve the reduced system (static: `K_r q = F_r`; modal: reduced generalized eigenproblem). |
| `affine.py` | Affine parametric decomposition: `K(mu) = theta_1(mu)*K_1 + ... + theta_Q(mu)*K_Q`, projected onto a reduced basis ONCE offline so that new parameter queries cost `O(Q * n_modes^2)` instead of a full reassembly + solve. |
| `frequency.py` | Intrusive frequency-domain ROM: `A(omega) = -omega^2*M + i*omega*C + K` is exactly an affine-in-omega system, so `FrequencyROM` composes `affine.py` + `galerkin.py` (rather than duplicating their algebra) for fast harmonic-response sweeps, with either a modal or a POD-on-FRF-snapshots basis. Also provides an efficient exact residual (`residual_norm()`) and a coercivity-constant-free error indicator (`hierarchical_error_indicator()`), used by `greedy.py`. |
| `greedy.py` | Weak-greedy adaptive training for `FrequencyROM`: instead of a fixed training-frequency grid, adaptively picks which frequencies to full-order-solve at, spending that budget where the current basis is weakest -- using the hierarchical indicator instead of a classical coercivity-bound estimator, because the literature documents that approach as weak exactly near resonance. |
| `scm.py` | The one genuinely **certified** (rigorously provable, not just a heuristic) error bound in this package: `SingularValueLowerBound` lower-bounds `sigma_min(A(omega))` from a handful of offline reference points via a Weyl/Mirsky matrix-perturbation inequality, turning `residual_norm()` into a true a posteriori bound (`certified_error_bound()`) rather than an estimate. Deliberately a SIMPLIFIED SCM-family method, not the classical LP-based (natural-norm) Successive Constraint Method -- see the caveats below. |
| `scm_lp.py` | The genuine classical LP-based Successive Constraint Method (Huynh, Rozza, Sen, Patera 2007): `LPSingularValueLowerBound` (same API as `scm.py`'s `SingularValueLowerBound` -- `from_affine()`, `add_reference()`, `lower_bound()`, `upper_bound()`, `greedy_train()`) bounds `sigma_min(A(omega))^2 = lambda_min(A(omega)^H A(omega))` via a genuine online linear program (`scipy.optimize.linprog`, HiGHS) built from offline-selected reference/control points, exposed as `certified_error_bound_lp()`. Squaring turns the problem into a real, symmetric, PSD eigenvalue problem, sidestepping a literature-acknowledged derivation ambiguity the sharper natural-norm variant has for complex, non-Hermitian operators like this package's `A(omega)` -- see the caveats below and `docs/phase4_error_bounds_greedy_roadmap.md` Section 9. |
| `metrics.py` | Generic, method-agnostic comparison metrics: `modal_assurance_criterion()` (MAC) -- scale- and phase-invariant collinearity between two mode-shape vectors -- and `r_squared()` -- the coefficient of determination, used by the nonlinear-ROM roadmap's validation but not tied to it. |
| `loewner.py` | **Non-intrusive** modal parameter identification: `LoewnerROM.fit()` builds a Loewner pencil (Mayo & Antoulas 2007) directly from sampled complex frequency-response data and solves a generalized eigenproblem for natural frequencies/damping ratios, never touching `K`/`M`/`C`. `.reconstruct_mode_shapes()` recovers full mode shapes from multi-DOF response data. |
| `screening.py` | Automated cross-ROM stability screening for `LoewnerROM`: `screen_physical_modes()` builds many `LoewnerROM`s from independent random interpolation-frequency subsets and keeps only the eigenpairs that recur, tightly clustered, across most of them -- separating genuine physical modes from an oversized interpolation order's spurious numerical artifacts. |
| `sampling.py` | Generalized Optimal Latin Hypercube Sampling (OLHS) design-of-experiments core: `optimal_lhs()` (general-purpose, unit-hypercube, best-of-`n_iter` maximin selection) and `modal_force_samples()` (a thin rescaling into the per-mode training-load convention a nonlinear-ROM training strategy needs). Phase 1 of `docs/nonlinear_surrogate_rom_roadmap.md`. |
| `nonlinear_rom.py` | Hybrid nonlinear structural ROM: an intrusive linear modal basis plus a non-intrusive, black-box regression of the nonlinear reduced restoring force. `MultiFidelitySurrogate` (RBF fit of `F_nl(q_l)`, 4 kernels each with an analytic Jacobian -- the MFS-NLROM/He et al. 2023 approach) and `PolynomialModalROM` (Nash-form quadratic+cubic fit of `F_nl(q_nl)` + Newton solve -- the ICE/Shi & Mei/STEP family, with `ICEROM`/`ShiMeiROM`/`EnforcedDisplacementROM` convenience constructors), both implementing the shared `ReducedForceModel` protocol. `AppliedLoadStrategy`/`EnforcedDisplacementStrategy` generate training data via a caller-supplied `fom_solver` callable. Phase 2 of `docs/nonlinear_surrogate_rom_roadmap.md`. |
| `nonlinear_dynamics.py` | Generalized reduced nonlinear time integration: `integrate_newmark_surrogate()` separates the Newmark-beta predictor/corrector METHOD from the MODEL (any `nonlinear_rom.ReducedForceModel`). Four selectable `correction` modes. `"none"` is the cheapest baseline. `"fixed_point"` (ported from this project's own validated `mfs-nlrom-beam` prototype) is the default since 2026-09-24. `"sign_deviation"` reconstructs the MFS-NLROM paper's Eq. 32 idea but is known to diverge on strongly nonlinear problems (its `sign(qddot)` factor reverses the correction for decelerating modes), so it is kept only for reproducing that paper. `"newton"` (`domain="q_nl"` only) runs full Newton-Raphson each step. Phase 4 of `docs/nonlinear_surrogate_rom_roadmap.md`. |
| `nnm.py` | Multi-harmonic-balance (HBM) + Alternating Frequency-Time (AFT) nonlinear normal mode (NNM) backbone continuation. `HarmonicBalanceSystem` assembles the linear-in-`Z` operator `A(omega)` (real cos/sin block form, deliberately mirroring `frequency.FrequencyROM`'s own complex `A(omega)`) for any `nonlinear_rom.ReducedForceModel` via AFT. Two continuation drivers: `solve_nnm_backbone()` Newton-continues via NATURAL-PARAMETER (prescribed-amplitude) continuation, simple and fast for the common non-folding (monotonically hardening/softening) case; `solve_nnm_backbone_arclength()` (Phase 6 addendum) instead uses Keller (1977) PSEUDO-ARCLENGTH continuation -- a secant-predictor + augmented-Newton-corrector scheme with an omega-rescaling fix for a real mixed-units step-size issue -- able to trace THROUGH a genuine amplitude fold, where natural-parameter continuation cannot. Built from first-principles HBM/AFT theory (no literal prototype to port from) and validated against `frequency.FrequencyROM` (exact `n_harmonics=1` cross-check), an independent scipy-ODE Duffing-oscillator backbone, an analytic unit-circle fold-turning proof, an exact-reduction check on a 2-mode coupled system, and a REAL fea_engine clamped-clamped beam backbone measured via time-integration + FFT period. Step 5 of `docs/nonlinear_surrogate_rom_roadmap.md`; pseudo-arclength continuation is that document's Section 11 (Phase 6) addendum. |
| `state_space.py` | Converts a second-order structural system `(M, C, K)` plus an explicit INPUT map `B` and OUTPUT map `Cout` into first-order state-space form -- `to_state_space(..., form="E")` (descriptor, no `M`-inversion) or `form="A"` (explicit ODE form, one dense `M`-solve). The shared prerequisite `krylov.py` and `balanced_truncation.py` both need. |
| `mode_correction.py` | Mode acceleration + modal truncation augmentation (Besselink et al. 2013, eqs. 13-20): cheap, closed-form corrections that recover the STATIC contribution a truncated modal basis leaves out, using one extra full-order static solve. `mode_acceleration_response()` corrects post-hoc; `augmented_basis()` folds the correction into the basis for use with the existing `GalerkinROM`. Both proved (and tested) to reproduce the exact full-order static solution to machine precision. |
| `krylov.py` | Krylov-subspace moment matching (Besselink et al. 2013, eqs. 21-36): `arnoldi_basis()` builds a block-Arnoldi basis via one reused LU factorization; `KrylovROM.from_MCK()` projects onto it. One-sided (Galerkin, default) matches the transfer function's Taylor moments EXACTLY at the expansion point `s0`, with accuracy degrading away from it -- expected, not a flaw. Two-sided (Petrov-Galerkin, `from_MCK(..., two_sided=True)`) via `two_sided_arnoldi_bases()` matches roughly TWICE as many moments at the same reduced order, at the cost of a documented biorthogonalization fragility that raises `ValueError` rather than degrading silently. `KrylovROM.is_stable()` checks (does not assume) stability either way, since moment matching gives no a priori guarantee. |
| `soar.py` | SOAR (Second-Order Arnoldi, Bai & Su 2005): `soar_basis()` builds an orthonormal basis of the second-order Krylov subspace (`r_1=K0^-1 B`, `r_j=A1 r_{j-1}+A2 r_{j-2}`) directly from `(M,C,K)`, NEVER forming the `2*n_dof` first-order pencil `krylov.py` needs; `SOARROM.from_MCK()` projects `M/C/K/B/Cout` onto it -- an ordinary Galerkin projection, which is exactly what makes `passivity.py`'s second-order-Galerkin passivity theorem apply, unlike `krylov.py`'s first-order `KrylovROM`. Only SOAR is implemented, deliberately not TOAR (the further numerical stabilization of SOAR's own known fragility) -- mitigated instead via full reorthogonalization, matching `krylov.py`'s own `arnoldi_basis()`. A real self-caught bug (feeding orthonormalized rather than raw vectors into the two-term recurrence silently capped moment-matching accuracy at moment 1 regardless of basis size) was found and fixed during development -- see the module's own docstring. Validated: exact moment matching through moment `k-1` for a `k`-vector basis (synthetic system, at `s0=0` and a nonzero complex `s0`); on a real fea_engine beam with a 3-DOF block port, `SOARROM(k)` measurably OUTPERFORMS `KrylovROM(2k)` (same total reduced-state count) at small `k`; and, the actual motivation, `SOARROM` stays passive at every tested order and expansion point (unlike `KrylovROM`, which has no such guarantee). A genuine, honestly-reported SOAR fragility: for a SINGLE-DOF port on this beam's well-separated eigenvalues, the raw recursion collapses toward the dominant mode like inverse power iteration, plateauing the basis size well below any requested `k` -- exactly why real SOAR/TOAR usage favors block ports, confirmed directly (the same block port does not plateau). Phase 5 addendum of `docs/frequency_domain_rom_roadmap.md` (Section 8). |
| `random_vibration.py` | PSD (power spectral density) random-vibration response: `psd_response(rom_freq, freqs_hz, psd_input, F0_pattern, output_dofs)` reuses `FrequencyROM.frequency_response()` to compute `S_out(f) = \|H(f)\|^2 * S_in(f)` and the response RMS, mirroring `fea_engine`'s own already-validated `solve_random_vibration()` formula exactly (a single `output_dofs` int matches its signature 1:1; an array-like of several DOFs is a small generalization). Phase 6 of `docs/frequency_domain_rom_roadmap.md`. |
| `balanced_truncation.py` | Balanced truncation (Besselink et al. 2013, eqs. 37-58): `controllability_gramian()`/`observability_gramian()` solve the two Lyapunov equations (scipy's dense Bartels-Stewart solver); `hankel_singular_values()` uses the numerically preferred square-root method; `BalancedTruncationROM.from_MCK()` balances and truncates. The one method that stays accurate globally (not just near one point) because it uses where the ports are. `is_stable()` is a genuine theorem here (unlike `krylov.py`); `h_infinity_error_bound()` gives a real, checked a priori worst-case error bound `krylov.py` has no equivalent of. See the module's own docstring for a practical scale note (modally pre-truncate a raw FE model before balancing it). `SingularPerturbationROM` (singular perturbation approximation / residualization, Liu & Anderson 1989) is a sibling reduction in the SAME module with an IDENTICAL API (`from_MCK()`, `transfer_function()`, `frequency_response()`, `h_infinity_error_bound()`, `is_stable()`), reusing the SAME balancing transform (`hankel_singular_values(..., return_transform=True)`) rather than recomputing anything. Where ordinary `BalancedTruncationROM` simply discards the truncated ("fast") balanced states -- exact at `s=infinity`, generally wrong at DC/`s=0` -- SPA instead holds them at algebraic quasi-steady-state (derivative=0, not state=0) and solves the resulting Schur-complement system, giving a nonzero `D_r` (always exactly 0 for ordinary BT here) that makes it exact at DC instead: the state-space analogue of what `mode_correction.py` already does for a plain modal basis. Both share the exact same a priori H-infinity error bound and stability-preservation guarantee; on the same real fea_engine damped-cantilever fixture at `r=10`, SPA's DC-gain error measures 1.7e-13 relative (machine precision -- an algebraic identity) vs. ordinary BT's 3.4e-5 relative at the same `r`, roughly 8 orders of magnitude worse, without costing BT's own whole-sweep accuracy (max relative error 1.7e-4 across the frequency sweep at `r=10`, comparable to ordinary BT's). Prefer `BalancedTruncationROM` when high-frequency accuracy matters most; prefer `SingularPerturbationROM` when DC/near-static-load accuracy matters most -- the choice is about WHERE the reduced model is exact, not which is "better" overall. `FrequencyWeightedBalancedTruncationROM` (Enns 1984) is a third sibling with the same `from_MCK()`/`transfer_function()`/`frequency_response()`/`is_stable()` API shape, plus `lowpass_weight(wc)`/`bandpass_weight(omega_n, zeta=0.1)` convenience constructors for the input/output weight filters `Wi(s)`/`Wo(s)`. Where ordinary BT ranks states by importance UNIFORMLY across all frequencies, this class weights the controllability/observability Gramians by `Wi`/`Wo` BEFORE balancing (`frequency_weighted_gramians()`, `frequency_weighted_hankel_singular_values()`), ranking states by importance in a CHOSEN band instead; with `Wi=Wo=None` it reduces EXACTLY to ordinary `BalancedTruncationROM` (a regression test, sharing the same `_balance_from_gramians()` balancing step factored out of `hankel_singular_values()` -- zero behavior change to the unweighted path). This is a genuine trade-off, not a free improvement: on the same fixture at `r=6`, targeting the beam's third natural frequency with `bandpass_weight(omega3, zeta=0.2)` as an output weight, it is 14.0x MORE accurate than ordinary BT IN the targeted band (max relative error 0.238 vs 3.33 -- ordinary BT essentially fails to represent this mode at all at `r=6`) but ~5.3x LESS accurate FAR from it, near the dominant first mode (1.9e-3 vs 3.6e-4) -- both measured directly on the same (r, zeta), not just cited from theory; the effect held across most of a small parameter sweep, not uniformly at every combination tried. Unlike `BalancedTruncationROM`/`SingularPerturbationROM`, stability is NOT unconditionally guaranteed here -- a real, literature-documented limitation of Enns' two-sided (both `Wi` and `Wo` given) construction. One-sided weighting (only `Wi` or only `Wo`) preserved stability at every tested `r` (4, 6, 8, 10); two-sided weighting, genuinely checked (not assumed) across `r` in {4,6,8,10,12} on the same fixture, came out stable at `r=4,10,12` and UNSTABLE at `r=6,8` -- so `is_stable()` on this class must be called and checked per-use (like `krylov.KrylovROM.is_stable()`), not treated as a theorem-backed guard. It deliberately does NOT expose `h_infinity_error_bound()`: the classical a priori bound formula applies to the UNWEIGHTED Hankel singular values specifically, and a frequency-weighted a priori bound exists in the literature but wasn't independently verified here -- the same honest-omission choice `KrylovROM` already makes (confirmed by a `hasattr` test). |
| `passivity.py` | Passivity diagnostics, usable against ANY of this package's frequency-response-producing classes (`FrequencyROM`, `KrylovROM`, `BalancedTruncationROM`, `SingularPerturbationROM`, `FrequencyWeightedBalancedTruncationROM`) -- plain functions, not a ROM class: `velocity_transfer_function(H_disp, omega_array)` converts an already-computed DISPLACEMENT frequency response to a VELOCITY (mobility) one via multiplication by `i*omega` (velocity is the time-derivative of displacement, so no new state-space "velocity output" plumbing is needed); `passivity_margin(H_vel)` returns `Re[H_vel(i*omega)]` at each frequency; `is_passive(H_vel, tol=0.0)` checks it stays `>= -tol` everywhere. Originally scoped as coprime-factorization/passivity-preserving balanced truncation, but the standard construction needs an algebraic Riccati equation requiring `D + D^T` invertible, and this package's structural systems always have `D=0` (no direct feedthrough) -- singular. Pivoted instead to a closed-form result: for a collocated force-in/velocity-out system, the mechanical energy `E = 0.5 qdot^T M qdot + 0.5 q^T K q` satisfies the KYP/positive-real dissipation inequality `dE/dt <= y^T u` exactly whenever `M`, `K` are SPD and `C` is PSD -- properties this package already assumes everywhere -- so `galerkin.py`/`frequency.py`'s existing second-order Galerkin reduction ALREADY provably preserves passivity, for any basis, with no Riccati equation needed. Validated in `test_passivity.py`: the full-order fea_engine fixture is passive (sanity check); a modally-reduced `FrequencyROM` stays passive at every tested basis size; and a sweep of the state-space (first-order) reduction methods found 8 of 9 checks passive but ONE genuine violation -- `FrequencyWeightedBalancedTruncationROM` at `r=14` with a bandpass output weight -- concrete evidence that first-order reduction carries no passivity-preservation guarantee, unlike second-order Galerkin projection. |
| `hankel_norm.py` | Optimal Hankel norm approximation (Adamjan-Arov-Krein theory; explicit realization due to Glover 1984): `OptimalHankelNormROM.from_MCK()` builds, in ONE application of the AAK formulas (reusing `balanced_truncation.hankel_singular_values()` for the balancing step), an order-`r` model with `\|\|G-G_r\|\|_Hankel = sigma_(r+1)` EXACTLY -- the true minimum over EVERY possible order-`r` system (stable or not), a stronger optimality claim (in the Hankel-norm sense) than `BalancedTruncationROM`'s own a priori H-infinity bound. SISO only (a scalar division in the derivation -- raises `ValueError` for MIMO systems rather than guessing at an unverified generalization, matching `scm_lp.py`/`passivity.py`'s precedent). A genuine numerical fragility was found via deliberate stress-testing BEFORE the module was written (unreliable when `sigma_(r+1)` is very small relative to the largest Hankel singular value, and no simple a priori quantity -- including `cond(Gamma)` -- reliably predicted it), so EVERY construction runs an always-on EMPIRICAL self-check instead (`numerically_reliable`/`measured_hankel_norm_error`, comparing the theoretical `sigma_(r+1)` against the independently-measured Hankel norm of the error system `G-G_r`, itself computed via the same already-validated `hankel_singular_values()`), warning loudly rather than failing silently. Validated on the real fea_engine cantilever fixture (`test_hankel_norm.py`): the core AAK optimality claim was checked DIRECTLY (`OptimalHankelNormROM`'s Hankel-norm error `<=` `BalancedTruncationROM`'s own at the same `r`, at every tested `r`), and the self-check machinery was shown to both agree with theory to near machine precision in the reliable regime AND correctly flag a genuinely unreliable case reproduced on this fixture (a 15-mode pre-reduction at `r=4`, vs. the 6-mode pre-reduction the reliable tests use). An honest finding, not assumed favorably: Hankel-norm optimality did NOT translate into better sup-norm/relative frequency-response accuracy than ordinary `BalancedTruncationROM` at the same `r` on this fixture (mirrors `FrequencyWeightedBalancedTruncationROM`'s own honest trade-off reporting). `h_infinity_error_bound()` is deliberately NOT exposed (the tight L-infinity bound needs an additional recursive correction step not independently re-derived here) -- the same honest-omission precedent `KrylovROM`/`FrequencyWeightedBalancedTruncationROM` already set. |
| `cms.py` | **Component mode synthesis.** `guyan()` (static condensation) and `craig_bampton()` (fixed-interface modes + constraint modes, interface DOFs kept physical), and `couple()` to join reduced substructures at shared interface DOFs into one model with `solve_modal()`. Validated on a real fea_engine clamped-clamped beam split into two halves: with all fixed-interface modes it reproduces the full model's frequencies (max relative error ~1e-9), and the error (relative to the first frequency) falls monotonically as modes are added, about 8e-3 with 5 modes per half and 4e-4 with 10. |
| `hyper_reduction.py` | **Hyper-reduction** so a nonlinear ROM no longer evaluates the full-order force. `ECSW` (weighted element subset from a Lawson-Hanson NNLS fit, `ecsw_weights()`), `DEIM`/`deim_indices()`/`qdeim_indices()` (interpolation of a force vector from a few DOFs), `gappy_reconstruct()`, and `HyperReducedNonlinearROM`, a drop-in subclass of `IntrusiveNonlinearROM` whose integrators all work unchanged. On a 40-element nonlinear fea_engine beam, ECSW picked 16 of 40 elements, matched the full reduced force to ~3e-5 on held-out states (about 3x faster per force evaluation), and a 300-step RK4 trajectory agreed with the full intrusive ROM to 1.1e-5 relative. End to end the run was only about 1.4x faster on this small model, because the other per-step costs dominate there; the gain grows with model size. Accuracy depends on the training states covering the range you will use. |
| `linear_dynamics.py` | **Linear time-domain response.** `newmark_linear()` (one factorisation, second-order accurate), `modal_superposition()` (exact integration of each mode for piecewise-linear loads, constant or Rayleigh damping), `piecewise_linear_exact()` (the exact discrete propagator) and `galerkin_transient()` (Newmark on a `GalerkinROM`'s reduced matrices). Checked against closed-form single-DOF solutions (halving dt divides the Newmark error by ~4) and against the full-order response of a real fea_engine beam. |
| `validation.py` | **One standard validation report for any ROM.** `validate_rom(rom_predict, fom_predict, test_inputs, tol=...)` returns per-case errors, timing, speed-up and pass/fail; `convergence_study()` gives error versus reduced size and `select_basis_size()` the smallest size meeting a tolerance. Works for any predictor (static vectors, time histories, complex frequency responses). |

### Why these three together

A parametric structural model -- say, a beam whose two halves have
independently uncertain or design-varying material stiffness -- needs
many solves at different parameter values (a sweep, an optimization
loop, an uncertainty-quantification run). The three modules chain
together to make that cheap:

1. **POD** turns a handful of full-order solves into a small basis
   that captures how the structure actually deforms.
2. **Galerkin projection** turns the full-order system into a tiny
   one, on that basis.
3. **Affine decomposition** exploits the fact that ordinary linear
   elasticity is exactly linear in a scaling modulus/rigidity, so the
   expensive "project the big matrices onto the small basis" step only
   needs to happen once -- ever -- no matter how many parameter values
   are later queried.

The result: a full re-assembly + solve of a real fea_engine model
costs single-digit milliseconds; the corresponding reduced query costs
microseconds. `examples/two_region_beam_rom.py` demonstrates and
measures this directly (see below) -- roughly a 200x speedup on a
120-element two-region cantilever, at a maximum tip-deflection error
under 0.2% across held-out parameter points.

<a id="s-3"></a>

## ⚙️ Installation

```bash
pip install -e .              # core (numpy, scipy only)
pip install -e ".[fea]"       # + fea_engine, for running tests/examples
pip install -e ".[dev]"       # + pytest, for running the test suite
```

<a id="s-4"></a>

## 🚀 Quick start

```python
import numpy as np
from rom_engine import PodBasis, GalerkinROM, AffineDecomposition

# K, M, F below are plain numpy arrays -- from fea_engine, or anywhere else.

# 1. POD from a set of snapshot vectors (e.g. static solves at several
#    training loads or parameter values), mass-weighted if M is given:
basis = PodBasis().fit(snapshots, n_modes=10, M=M)

# 2. Galerkin-project and solve the reduced system:
rom = GalerkinROM(basis).reduce_system(K, M=M, F=F)
x_full, q_reduced = rom.solve_static()
freq_hz, mode_shapes_full, mode_shapes_reduced = rom.solve_modal()

# 3. Affine decomposition for fast many-query parameter sweeps, when
#    K(mu) = theta_1(mu)*K_1 + theta_2(mu)*K_2 + ...:
affine = AffineDecomposition([K1, K2], theta_func=lambda mu: [mu[0], mu[1]])
affine.project(basis.V)                       # offline, once
for mu in many_parameter_values:
    K_r = affine.assemble_reduced(mu)          # online, cheap
    q = np.linalg.solve(K_r, rom.project_vector(F))

# 4. Frequency-domain ROM: A(omega) = -omega^2*M + i*omega*C + K,
#    fast harmonic sweeps via the SAME affine machinery under the hood.
from rom_engine import FrequencyROM
rom_freq = FrequencyROM.from_MCK(M, K, basis.V, rayleigh=(alpha, beta))  # or C=C_matrix
response = rom_freq.frequency_response(omega_array, F)   # (n_omega, n_dof) complex

# 5. Greedy training: instead of a fixed training-frequency grid, adaptively
#    pick which frequencies to solve at, based on a cheap error indicator:
from rom_engine import greedy_train_frequency_basis
basis, history = greedy_train_frequency_basis(
    candidate_omegas, M, K, F, rayleigh=(alpha, beta), max_modes=20)

# 6. A genuinely CERTIFIED error bound (not just an estimate), from a
#    handful of offline reference points -- see the caveats below before
#    relying on this for anything beyond a narrow band around a reference:
from rom_engine import SingularValueLowerBound, certified_error_bound
scm = SingularValueLowerBound.from_affine(rom_freq.affine)
scm.add_reference(omega_of_interest)              # offline, one SVD
bound = certified_error_bound(rom_freq, scm, omega_of_interest, F)  # online, cheap

# 7. NON-INTRUSIVE modal identification: given only sampled complex
#    response data x(omega) (measured, or from any solver) at two
#    disjoint interpolation-frequency sets -- no M/C/K required:
from rom_engine import LoewnerROM, screen_physical_modes, modal_assurance_criterion
rom_id = LoewnerROM.fit(omega_alpha, omega_beta, x_alpha, x_beta)
print(rom_id.f, rom_id.eta)                        # identified frequencies [Hz], damping ratios
mode_shapes = rom_id.reconstruct_mode_shapes(X_beta_multi, omega_beta=omega_beta)

# Automated screening removes spurious eigenpairs from an oversized
# interpolation order by cross-checking many random ROMs:
physical = screen_physical_modes(freq_pool, x_pool, fmin, fmax,
                                  rng=np.random.default_rng(0))

# 8. NONLINEAR reduced-force model: fit a black-box surrogate of the
#    nonlinear modal restoring force from sampled training data (here,
#    an RBF fit against the linear modal displacement q_l -- the
#    MFS-NLROM approach), then use it in place of a full-order
#    nonlinear internal-force evaluation:
from rom_engine import MultiFidelitySurrogate, AppliedLoadStrategy
strategy = AppliedLoadStrategy(target_fracs=(0.3, 2.0), reference_scale=thickness,
                                n_samples=40, rng=np.random.default_rng(0))
q_l, q_nl, F_nl = strategy.generate(V, M_ff, basis_freqs_hz, mode_shape_peaks, fom_solver)
surrogate = MultiFidelitySurrogate(kernel="cubic").fit(q_l, F_nl)
F_nl_pred = surrogate.predict(q_l_new)             # closed-form, no Newton-Raphson
J = surrogate.jacobian(q_l_new)                    # analytic dF_nl/dq_l

# Or the polynomial/ICE family (Nash-form, fit against q_nl instead):
from rom_engine import PolynomialModalROM
poly_rom = PolynomialModalROM(n_modes=len(basis_freqs_hz)).fit(q_nl, F_nl)
q_solution, converged = poly_rom.predict(F_ext=F_ext, Lambda=Lambda)  # Newton solve

# 9. Reduced NONLINEAR TIME INTEGRATION: Newmark-beta on the modal
#    equation of motion, using EITHER model above as the nonlinear
#    force -- Newton-free by default (correction="fixed_point"):
from rom_engine import integrate_newmark_surrogate
C_r = 2 * zeta * (2 * np.pi * basis_freqs_hz)   # e.g. Rayleigh-calibrated modal damping
t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
    Lambda, C_r, surrogate, F_ext=0.0,          # free decay
    q0=q_nl0, qdot0=np.zeros_like(q_nl0),
    dt=dt, n_steps=n_steps, domain="q_l")        # domain="q_nl" for PolynomialModalROM

# 10. NNM BACKBONE CONTINUATION: trace frequency-vs-amplitude for the
#     autonomous (unforced, undamped) periodic-orbit family of a given
#     master mode, using EITHER nonlinear force model above via AFT:
from rom_engine import HarmonicBalanceSystem, solve_nnm_backbone
hb = HarmonicBalanceSystem(Lambda, n_harmonics=5)          # C_r=None -> undamped backbone
amplitudes, omegas, Z_hist, converged = solve_nnm_backbone(
    hb, surrogate, q_amplitude_range=(a_min, a_max),
    n_points=20, domain="q_l", master_mode=0)               # domain="q_nl" for PolynomialModalROM

# 11. HYPER-REDUCTION: evaluate the nonlinear reduced force from a few weighted elements
#     (elem_dofs: global DOFs per element; elem_force_fn(e, u_local): element internal force):
from rom_engine import ECSW, HyperReducedNonlinearROM
ecsw = ECSW.fit(V, training_states, elem_dofs, elem_force_fn, elem_tangent_fn, tol=1e-4)
rom_h = HyperReducedNonlinearROM(V, M, C, load_fn, K0_full, ecsw)   # same integrators as IntrusiveNonlinearROM
t, q_hist, qdot_hist = rom_h.integrate_rk4(q0, qdot0, dt, n_steps)

# 12. COMPONENT MODE SYNTHESIS: reduce two substructures and join them at their interface DOFs:
from rom_engine import craig_bampton, couple
cb1 = craig_bampton(K1, M1, interface1, n_modes=10)
cb2 = craig_bampton(K2, M2, interface2, n_modes=10)
freq_hz, shapes = couple([cb1, cb2], [[0, 1], [0, 1]]).solve_modal(5)

# 13. LINEAR TRANSIENT + one validation report for any ROM:
from rom_engine import modal_superposition, validate_rom
t, u, info = modal_superposition(K, M, load, dt, n_steps, n_modes=8, rayleigh=(alpha, beta))
report = validate_rom(rom_predict, fom_predict, held_out_inputs, tol=1e-3)
print(report.summary())
```

<a id="s-5"></a>

## 🗂️ Package layout

```
rom_engine/
├── pyproject.toml
├── docs/
│   ├── frequency_domain_rom_roadmap.md              # research + design doc for frequency.py
│   ├── phase4_error_bounds_greedy_roadmap.md        # research + design doc for greedy.py
│   ├── loewner_modal_identification_roadmap.md      # research + design doc for loewner.py/screening.py
│   ├── nonlinear_surrogate_rom_roadmap.md           # research + design doc for sampling.py/nonlinear_rom.py/...
│   ├── classical_mor_roadmap.md                     # research + design doc for state_space/mode_correction/krylov/balanced_truncation.py
│   └── testing_methodology.md                       # how/why this package is validated
├── src/rom_engine/
│   ├── __init__.py       # re-exports PodBasis, GalerkinROM, AffineDecomposition, FrequencyROM, ...
│   ├── pod.py
│   ├── galerkin.py
│   ├── affine.py
│   ├── frequency.py
│   ├── greedy.py
│   ├── scm.py
│   ├── scm_lp.py          # LPSingularValueLowerBound, certified_error_bound_lp()
│   ├── metrics.py        # modal_assurance_criterion(), r_squared()
│   ├── loewner.py         # NON-INTRUSIVE: LoewnerROM
│   ├── screening.py       # NON-INTRUSIVE: screen_physical_modes()
│   ├── sampling.py        # optimal_lhs(), modal_force_samples()
│   ├── nonlinear_rom.py   # MultiFidelitySurrogate, PolynomialModalROM, TrainingStrategy
│   ├── nonlinear_dynamics.py  # integrate_newmark_surrogate()
│   ├── nnm.py             # HarmonicBalanceSystem, solve_nnm_backbone(), solve_nnm_backbone_arclength()
│   ├── state_space.py     # to_state_space(), StateSpaceSystem
│   ├── mode_correction.py # mode_acceleration_correction/response(), augmented_basis()
│   ├── krylov.py          # arnoldi_basis(), two_sided_arnoldi_bases(), KrylovROM
│   ├── soar.py            # soar_basis(), SOARROM (second-order-structure-preserving Krylov)
│   ├── balanced_truncation.py  # controllability/observability_gramian(), hankel_singular_values(), BalancedTruncationROM, SingularPerturbationROM, FrequencyWeightedBalancedTruncationROM, lowpass_weight(), bandpass_weight()
│   ├── passivity.py       # velocity_transfer_function(), passivity_margin(), is_passive()
│   ├── hankel_norm.py     # OptimalHankelNormROM
│   ├── cms.py             # guyan(), craig_bampton(), couple()
│   ├── hyper_reduction.py # ECSW, DEIM/QDEIM, gappy_reconstruct(), HyperReducedNonlinearROM
│   ├── linear_dynamics.py # newmark_linear(), modal_superposition(), galerkin_transient()
│   ├── validation.py      # validate_rom(), convergence_study(), select_basis_size()
│   └── random_vibration.py     # psd_response()
├── tests/
│   ├── fea_fixtures.py    # fea_engine-based fixtures (cantilever, damped cantilever, clamped-clamped nonlinear beam, ...)
│   ├── loewner_fixtures.py  # synthetic mass-spring-damper fixture for loewner/screening
│   ├── plate_fixtures.py    # the paper's "Example 1" plate FE model (fea_engine-independent)
│   ├── test_pod.py
│   ├── test_galerkin.py
│   ├── test_affine.py
│   ├── test_frequency.py
│   ├── test_greedy.py
│   ├── test_scm.py
│   ├── test_scm_lp.py
│   ├── test_metrics.py
│   ├── test_loewner.py
│   ├── test_screening.py
│   ├── test_plate_fixtures.py
│   ├── test_loewner_plate.py
│   ├── test_sampling.py
│   ├── test_nonlinear_rom.py       # kernels, models, synthetic training strategies
│   ├── test_nonlinear_rom_fea.py   # real fea_engine clamped-clamped Beam2DCorotational validation
│   ├── test_nonlinear_dynamics.py      # Newmark bookkeeping, correction modes, closed-form/scipy-ODE checks
│   ├── test_nonlinear_dynamics_fea.py  # real fea_engine free-decay validation
│   ├── test_nnm.py       # AFT bookkeeping, FrequencyROM cross-check, Duffing backbone, domain translation
│   ├── test_nnm_fea.py   # real fea_engine clamped-clamped beam backbone validation
│   ├── test_state_space.py          # toy + real-fixture pole checks, form="E"/"A" equivalence
│   ├── test_mode_correction.py      # exact-static-reproduction proofs, refusal on degenerate q_cor
│   ├── test_krylov.py               # near/far-from-s0 accuracy, is_stable() self-consistency
│   ├── test_soar.py                 # exact moment matching, block-port vs. KrylovROM at same state count, passivity
│   ├── test_balanced_truncation.py  # whole-sweep accuracy, stability theorem, H-infinity bound check
│   ├── test_singular_perturbation.py  # SPA DC-gain exactness vs. ordinary BT, shared H-infinity bound, stability, whole-sweep accuracy
│   ├── test_frequency_weighted_bt.py  # unweighted-reduction regression, in-band/out-of-band trade-off, one-/two-sided stability, missing h_infinity_error_bound()
│   ├── test_passivity.py            # full-order + modally-reduced FrequencyROM passivity, 8/9-passive-1-violation sweep across Krylov/BT/SPA/FW-BT
│   ├── test_hankel_norm.py          # AAK optimality vs BT verified directly, self-check reliability (agrees when reliable, flags known-unreliable case), SISO restriction, honest accuracy trade-off finding
│   ├── test_random_vibration.py     # PSD S_out/sigma_out vs fea_engine's solve_random_vibration(), on/off resonance
│   ├── test_cms.py                  # Craig-Bampton/Guyan vs full beam frequencies, exactness limit, monotone convergence
│   ├── test_hyper_reduction.py      # ECSW/DEIM/QDEIM, NNLS weights, hyper-reduced vs full intrusive ROM trajectory
│   ├── test_linear_dynamics.py      # SDOF closed forms, Newmark convergence order, modal vs Newmark on a real beam
│   └── test_validation.py           # report, convergence study, basis-size selection
└── examples/
    ├── two_region_beam_rom.py             # POD + Galerkin + affine offline/online walkthrough
    ├── frequency_sweep_cantilever.py      # frequency-domain ROM walkthrough
    ├── greedy_frequency_training.py       # adaptive vs. uniform training-frequency selection
    ├── certified_bound_cantilever.py      # certified SCM-family bound: guarantee + honest reach
    ├── plate_modal_identification.py      # non-intrusive ID on a real plate FE model (air/water)
    └── mode_shape_vibration_recovery.py   # full-field mode shapes + time-domain recovery
```

<a id="s-6"></a>

## 🧪 Running the tests

```bash
pip install -e ".[fea,dev]"
pytest tests/ -v
```

All 410 tests pass (46 further tests need PyTorch and skip when it is not installed; 456 in total). The intrusive modules (below) are validated against
real fea_engine models (not synthetic matrices); the non-intrusive
`loewner`/`screening` modules are validated against BOTH a synthetic
mass-spring fixture and two real, materially different structural
models (a real fea_engine damped cantilever, and a from-scratch
867-DOF plate FE model matching a published paper's own benchmark) --
see the `loewner`/`screening` bullet below for details:

- **POD**: mass-weighted POD exactly recovers the true eigenspace of a
  cantilever beam (subspace principal angles = 1.0, machine precision);
  standard POD is confirmed to differ from mass-weighted POD (not just
  a relabeling); POD rank matches the known rank of an
  independent-load-case snapshot set; energy-threshold truncation and
  reconstruction-error monotonicity checked on both real and synthetic
  data.
- **Galerkin**: a full-rank basis exactly reproduces the full-order
  static solve; a POD basis exactly reproduces the response to any load
  in the span of its training loads; modal frequencies from a basis
  built from STATIC (not eigenmode) snapshots are shown to converge
  towards the true fea_engine modal frequencies as basis rank grows.
- **Affine**: the full-order affine reconstruction matches an
  independent fea_engine per-block reassembly to ~1e-16 relative error
  across multiple random parameter draws; the "project-then-sum" and
  "sum-then-project" orders are confirmed algebraically identical; the
  reduced online query is measured (not assumed) to be ~1800x faster
  than full reassembly on a 3200-dof model.
- **Frequency**: `FrequencyROM` matches fea_engine's own
  `solve_harmonic()`/`solve_frequency_sweep()` both near a resonance
  (the hard, near-singular case) and off resonance; accuracy improves
  with basis rank; a POD-on-FRF-snapshots basis is validated as a real
  alternative to a modal basis at a held-out frequency; a dedicated
  regression test confirms the complex `theta(omega) = i*omega` damping
  term survives (guards against the dtype bug described below); the
  proportional-damping 2-term collapse is confirmed algebraically
  identical to the general 3-term `{M,C,K}` form; a reduced sweep is
  measured substantially faster than fea_engine's own full sweep;
  `residual_norm()`'s efficient (`assemble_action`-based) computation
  is confirmed to match an independently, slowly computed residual;
  the hierarchical error indicator is confirmed to be large exactly
  where the true error (vs. fea_engine) is large, and small where it's
  small, not just plausible-looking.
- **Greedy**: greedy-selected training frequencies are confirmed to
  cluster measurably closer to the model's real resonances than a
  uniform-random baseline; a greedy-trained basis is confirmed to beat
  EVERY one of 7 differently-phased uniform-grid bases at the same
  rank (same full-order-solve budget) on a held-out, resonance-
  concentrated test set -- the actual justification for greedy's
  extra complexity, checked rather than assumed.
- **SCM (certified bound)**: `lower_bound()` is confirmed to NEVER
  exceed the true `sigma_min(A(omega))` (direct SVD ground truth) at
  300 swept frequencies including exact resonance -- the actual
  certification property, not just a plausibility check; a finite
  `certified_error_bound()` is confirmed to NEVER undershoot the true
  error against fea_engine's own `solve_harmonic()`; the bound is
  confirmed EXACT (not just non-violating) precisely at its own
  reference point; and, honestly, its practical "useful radius" for
  this real beam's stiffness matrix is directly measured (not
  asserted away) at roughly `1e-7` relative frequency from a
  reference -- see the caveat below.
- **Loewner / screening (non-intrusive)**: `LoewnerROM.fit()`'s
  signature is confirmed to structurally reject M/C/K (no such
  parameter exists to pass), not just avoid using them by convention;
  identified frequencies/damping match closed-form ground truth on a
  synthetic mass-spring fixture (max error ~0.002%) AND a real
  fea_engine damped cantilever (max error ~0.13%); `reconstruct_mode_shapes()`
  recovers true mode shapes with MAC > 0.99 on the synthetic fixture;
  `screen_physical_modes()` is confirmed to actually reject spurious
  eigenpairs from a deliberately oversized interpolation order
  (screened count converges exactly to the true mode count while the
  raw single-ROM count does not), with same-seed reproducibility and
  different-seed convergence to the same physical-mode set both
  checked directly; on the 867-DOF plate FE model (matching a published
  paper's own "Example 1" benchmark, air- and water-loaded), the full
  identify-then-screen pipeline matches 6/7 (air) and 7/7 (water) true
  modes to within 3% frequency error and MAC > 0.7.
- **Sampling (`optimal_lhs`/`modal_force_samples`)**: every design
  returned is confirmed to be a genuine Latin Hypercube (exactly one
  sample per stratum along each dimension, not just "looks spread
  out"); more candidate iterations is confirmed to NEVER make the
  maximin space-filling score worse (a guaranteed same-seed inequality,
  not a "usually helps" claim); `modal_force_samples()`'s sampled
  per-mode displacement targets are confirmed to land inside the
  requested fraction-of-reference-scale range, independently per mode.
- **`r_squared`**: confirmed to equal exactly 1 for a perfect
  prediction, exactly 0 for the constant-mean baseline (its own
  defining property), and a genuinely NEGATIVE value for a fit worse
  than that baseline -- not clipped to `[0, 1]`, since a negative R^2
  is a real, meaningful outcome a nonlinear-ROM fit can produce.
- **`nonlinear_rom` (nonlinear ROM)**: every kernel's analytic `dpsi/dR`
  is confirmed to match a finite-difference check of `psi`, and to
  vanish at `R=0` (no singularity in `jacobian()`'s direction term);
  `MultiFidelitySurrogate` and `PolynomialModalROM` both recover known
  synthetic cubic/Nash-form ground-truth functions to R² > 0.999, with
  their analytic Jacobians confirmed against finite differences and
  `PolynomialModalROM`'s Newton solve confirmed to recover a known
  displacement from its own predicted equilibrium force; both
  `TrainingStrategy` variants are validated end-to-end against a
  hand-built synthetic nonlinear (cubic-spring) full-order model AND
  -- the strongest check -- against a REAL fea_engine clamped-clamped
  `Beam2DCorotational` model (`test_nonlinear_rom_fea.py`), where both
  model families reach held-out R² > 0.95 (RBF) / > 0.99 (polynomial)
  on genuinely nonlinear training data generated by fea_engine's own
  `solve_nonlinear_static()`.
- **`nonlinear_dynamics` (reduced nonlinear time integration)**: with a
  zero or exactly-linear force model, `integrate_newmark_surrogate()`
  reduces EXACTLY to plain linear Newmark regardless of `correction`
  mode (`fixed_point` to near machine precision, since it Picard-
  iterates to convergence even for a linear residual); a `q_l`-domain
  linear force model is checked against an independent
  `scipy.integrate.solve_ivp` reference after analytically inverting
  the Eq. 9-11 domain relation; a genuinely nonlinear (cubic-spring)
  case is checked against `scipy.integrate.solve_ivp` for all three
  correction modes with no blow-up, confirming `fixed_point` is at
  least as accurate as `none` (the documented motivation for its extra
  cost). The strongest check: a REAL fea_engine clamped-clamped
  `Beam2DCorotational` free-decay run, compared against
  `nonlinear_solver.solve_nonlinear_transient()` as full-order ground
  truth (`test_nonlinear_dynamics_fea.py`) -- both `MultiFidelitySurrogate`
  and `PolynomialModalROM`, trained via `AppliedLoadStrategy`, reproduce
  the true response's short-window (~2 period) trajectory (R² > 0.9)
  and amplitude-decay envelope, with the expected, honestly-reported
  phase-drift-driven degradation over the full 8-period window (a
  known property of fitted-vs-exact nonlinear force models, not a bug
  -- see the module's own docstring).
- **`nnm` (NNM backbone continuation)**: the AFT trigonometric
  transform is confirmed exactly invertible (`Tinv @ T = I`) and
  self-consistent (`apply_A`/`apply_dA_domega` match their own dense
  matrix forms, and `dA_domega` matches a finite-difference check);
  at `n_harmonics=1`, `HarmonicBalanceSystem`'s real cos/sin solve is
  confirmed to EXACTLY reproduce `frequency.FrequencyROM`'s own
  complex FRF solve (real/imaginary parts) at several frequencies --
  a genuine cross-module consistency check `frequency.py` didn't
  previously have; `solve_nnm_backbone()` on a synthetic Duffing
  oscillator matches an INDEPENDENT `scipy.integrate.solve_ivp` +
  peak-measured-period ground truth to <1% across 6 amplitude points,
  reproduces the expected hardening trend, and gives matching answers
  whether `jacobian="analytic"` or a numerical fallback is used; the
  `domain="q_l"` translation is confirmed against a closed-form
  effective-stiffness linear limit to ~3e-11 relative error. The
  strongest check: a REAL fea_engine clamped-clamped
  `Beam2DCorotational` backbone, with an INDEPENDENT ground truth
  measured the same way this project's earlier `mfs-nlrom-beam`
  prototype validated its own backbone (undamped free-vibration time
  integration + FFT-based period measurement, not a second HBM solve)
  -- both `MultiFidelitySurrogate` and `PolynomialModalROM` backbones
  match that independent measurement to <10% at every point, correctly
  reproduce the hardening trend, and sit above the linear natural
  frequency (`test_nnm_fea.py`). The Phase 6 addendum,
  `solve_nnm_backbone_arclength()` (pseudo-arclength continuation, for
  backbones with a genuine amplitude fold), is validated separately:
  an analytic fold with total certainty (the unit circle `x^2+y^2=1`,
  traced through both `x=+-1` folds at machine-precision residual);
  agreement with `solve_nnm_backbone()` to <1e-3 relative on a
  non-folding Duffing backbone; and an exact reduction to the
  closed-form single-DOF Duffing relation on a 2-mode coupled system's
  invariant planar branch. A fully characterized, hand-verified fold on
  a real multi-mode NNM backbone was explored but not conclusively
  established within scope -- reported honestly as a natural next
  validation step, not a blocker, in
  `docs/nonlinear_surrogate_rom_roadmap.md` Section 11.

<a id="s-7"></a>

## 🧪 Running the examples

```bash
pip install -e ".[fea]"
python examples/two_region_beam_rom.py
python examples/frequency_sweep_cantilever.py
python examples/greedy_frequency_training.py
python examples/certified_bound_cantilever.py
python examples/plate_modal_identification.py
python examples/mode_shape_vibration_recovery.py
```

`two_region_beam_rom.py` builds a two-region cantilever beam in
fea_engine, trains a POD basis from 10 full-order static solves,
projects the affine stiffness components onto it, then validates the
ROM against 8 held-out full-order solves and times a 500-point
parameter sweep against full reassembly.

`frequency_sweep_cantilever.py` builds a damped cantilever in
fea_engine, builds a `FrequencyROM` from 20 undamped mode shapes,
sweeps 400 frequencies and compares against fea_engine's own
`solve_frequency_sweep()` (both overall and right at the resonance
peak), then repeats the comparison with a same-rank POD-on-FRF-
snapshots basis instead.

`greedy_frequency_training.py` compares a greedy-trained basis against
a uniform-grid basis at the SAME rank (same full-order-solve budget)
on a damped cantilever, printing which frequencies each strategy
actually trained on and how each performs on a held-out test set
concentrated near the model's real resonances.

`certified_bound_cantilever.py` builds `SingularValueLowerBound` on
the same damped cantilever, proves the certification guarantee holds
across a 400-point sweep including exact resonance (and confirms the
end-to-end `certified_error_bound()` never undershoots the true error
against fea_engine's own solve), then measures and reports the bound's
practical "useful radius" honestly -- it is narrow for this model, and
the script says so directly rather than only showing favorable cases.

`plate_modal_identification.py` builds the paper's "Example 1"
simply-supported plate FE model (867 DOF, air- and water-loaded),
generates synthetic response data at the paper's own Table-1 sensor
locations, and runs the full non-intrusive `LoewnerROM.fit()` +
`screen_physical_modes()` pipeline on ONLY that data -- reporting and
plotting identified vs. true frequency/damping for both loading cases.

`mode_shape_vibration_recovery.py` goes one step further on a synthetic
mass-spring-damper system: recovers FULL-FIELD mode shapes (Eq. 28
applied at every DOF, not just a few sensors) purely from response data
+ the identified ROM eigenvectors, then propagates the identified modal
model forward in time from an observed initial condition and compares
against a direct full-order time integration -- demonstrating the
identified model is actually usable, not just numerically close.

<a id="s-8"></a>

## ⚠️ A certified-bound caveat (read before using `scm.py`)

`SingularValueLowerBound`'s certification guarantee is rigorous and
holds unconditionally (proven in `test_scm.py`, demonstrated in
`certified_bound_cantilever.py`): its `lower_bound()` never exceeds
the true `sigma_min(A(omega))`, so a finite `certified_error_bound()`
is always a valid upper bound on the true error. But for this real
beam's stiffness matrix, the bound's simplified (raw spectral-norm,
Lipschitz-triangle-inequality) construction is only *informative*
within roughly `1e-7` relative frequency of an offline reference point
-- because `||K||_2` (~1e11-1e12) is many orders of magnitude larger
than `sigma_min(A(omega))` itself (~1e3), so the perturbation term
swamps the bound almost immediately away from an exact match. This is
a genuine, literature-anticipated property of this simplified method
(not a bug -- see `scm.py`'s module docstring), and is measurably
narrower than the >1%-of-first-resonance spacing that suffices for
`greedy.py`'s (uncertified) hierarchical-indicator-driven training.
Use `scm.py` for a rigorous, narrow safety margin around specific,
already-computed reference frequencies -- not as a general-purpose
error signal across a band; `residual_norm()` /
`hierarchical_error_indicator()` remain the practical tools for that,
with the explicit understanding that they are estimates, not bounds.
The classical "natural-norm" SCM (Chen et al. 2010) exists specifically
to close this gap with a smarter, problem-adapted norm; re-implementing
it is noted as optional future work below rather than attempted here.

<a id="s-9"></a>

## 🔢 A dtype note (why this matters more than it sounds)

Every array-ingesting method in `pod.py`/`galerkin.py`/`affine.py`
used to hardcode `np.asarray(x, dtype=float)`. That was invisible for
this package's original real-valued use cases, but it is a silent
correctness bug for `frequency.py`: `affine.py`'s
`theta_func(omega) = i*omega` is PURELY imaginary, so the old code
would have silently zeroed out the entire damping contribution to
every reduced query -- no error, no warning, just a quietly-wrong
answer. This was caught and fixed (each module now infers real-vs-
complex dtype from its input instead of forcing `float`) before
`frequency.py` was built on top of it; `test_frequency.py` includes a
dedicated regression test (`test_general_damping_matches_fea_engine_
complex_solve`) asserting the imaginary/phase-lag part of a damped
response is actually present, specifically to keep this bug from
silently reappearing.

<a id="s-10"></a>

## 🗺️ Roadmap

The 5-phase nonlinear-surrogate-ROM plan in
`docs/nonlinear_surrogate_rom_roadmap.md` is now fully implemented and
validated: `sampling.py` (Phase 1), `nonlinear_rom.py` (Phase 2 -- the
shared `ReducedForceModel` protocol, `MultiFidelitySurrogate`/
`PolynomialModalROM`, and both `TrainingStrategy` variants),
`fea_engine.nonlinear_solver.solve_nonlinear_transient()` (Phase 3 --
lives in the `fea_engine` package, see that package's own README),
`nonlinear_dynamics.py` (Phase 4 -- `integrate_newmark_surrogate()` and
its three correction modes), and `nnm.py` (Phase 5, final phase --
`HarmonicBalanceSystem` + `solve_nnm_backbone()`), each with real
fea_engine validation.

Not currently planned, but possible future work:

- ~~Within `nnm.py` itself: full pseudo-arclength continuation (needed
  for backbones with a genuine fold, i.e. non-monotonic amplitude vs.
  frequency)~~ -- now implemented as a Phase 6 addendum:
  `solve_nnm_backbone_arclength()`, Keller (1977) pseudo-arclength
  continuation with a secant predictor, an augmented Newton corrector,
  and an omega-rescaling fix for a real mixed-units step-size issue
  (`docs/nonlinear_surrogate_rom_roadmap.md` Section 11). Validated
  against an analytic fold (the unit circle, traced through both
  folds at machine precision), agreement with `solve_nnm_backbone()`
  on a non-folding Duffing backbone (<1e-3 relative), and an exact
  reduction to a closed-form single-DOF case on a 2-mode coupled
  system's invariant planar branch; a fully characterized real NNM
  fold was explored but not conclusively established within scope,
  reported honestly as a natural next validation step rather than a
  blocker. Subharmonic-multiplier support remains unimplemented --
  a materially larger AFT-scheme extension than pseudo-arclength
  turned out to be, still explicitly scoped out (see `nnm.py`'s own
  module docstring for why).
- ~~A PSD-driven random-vibration stress analyzer~~ -- now implemented:
  `random_vibration.py`'s `psd_response()`, validated against
  `fea_engine`'s own `solve_random_vibration()` (`test_random_vibration.py`).
- ~~The full classical LP-based (natural-norm) Successive Constraint
  Method (Chen et al. 2010), as a sharper replacement for `scm.py`'s
  simplified Weyl/Mirsky-perturbation bound~~ -- the classical LP-based
  SCM (Huynh, Rozza, Sen, Patera 2007) is now implemented: `scm_lp.py`'s
  `LPSingularValueLowerBound` (same API as `scm.py`'s
  `SingularValueLowerBound`) plus `certified_error_bound_lp()`, using a
  genuine online linear program (`scipy.optimize.linprog`, HiGHS) built
  from offline-selected reference/control points -- validated in
  `test_scm_lp.py` against the same real fea_engine damped-cantilever
  fixture `test_scm.py` uses (certification holds unconditionally,
  checked at 200 points including exact resonance). Honestly, this did
  NOT close the gap it was meant to close: compared directly against
  `scm.py` on the SAME reference point of the SAME real cantilever
  model, `scm_lp.py`'s useful (non-trivial) radius measured NARROWER,
  not wider -- already 0 at the narrowest step tested (`1e-8` relative
  frequency), vs. `scm.py` remaining nonzero out to ~`1e-7`. The reason:
  bounding `sigma_min(A(omega))^2 = lambda_min(A(omega)^H A(omega))`
  instead of `sigma_min(A(omega))` directly (the standard "SCM-squared"
  reduction, needed to keep the eigenvalue problem real, symmetric, and
  PSD for a complex, non-Hermitian `A(omega)`) makes `B_pq = A_p^T A_q +
  A_q^T A_p` inherit roughly the SQUARE of the raw component
  spectral-norm spread (~1e22-1e24, vs. ~1e11-1e12 for the raw `M`/`K`
  matrices themselves) -- worsening the same scale-mismatch problem
  described in the caveat above instead of fixing it. (A real
  implementation bug surfaced along the way, too: HiGHS spuriously
  reported "unbounded" for `upper_bound()`'s maximization -- used only
  for greedy-selection heuristics, not the certified bound itself -- on
  a provably compact region, because of that same ~1e23-magnitude
  coefficient range; fixed by making `upper_bound()` a closed-form
  box-only maximization instead of relying on the fragile LP there.
  `lower_bound()`, the actually certified quantity, solves correctly
  via the LP as-is.) Both bounds remain available side by side. The
  sharper **natural-norm** SCM variant (Chen et al. 2010) -- which
  bounds `sigma_min(A(omega))` directly, without the squaring that
  worsens the scale mismatch above -- remains genuinely future work: it
  has a real, literature-acknowledged derivation ambiguity for complex,
  non-Hermitian operators like this package's
  `A(omega) = -omega^2*M + i*omega*C + K`, which is why it was
  deliberately not attempted here (see
  `docs/phase4_error_bounds_greedy_roadmap.md` Section 9 for the full
  reasoning). `residual_norm()`/`hierarchical_error_indicator()` remain
  the practical, UNcertified estimates/indicators `greedy.py` actually
  uses; `scm.py`/`scm_lp.py`'s bounds are certified-but-narrow
  complements to them, not replacements.
- Within `balanced_truncation.py`: `docs/classical_mor_roadmap.md`
  explicitly deferred five BT extensions the source paper reviews but
  never runs numerically -- frequency-weighted BT, singular perturbation
  approximation, coprime-factorization BT for unstable systems,
  passivity-preserving BT, and optimal Hankel norm approximation -- as
  the classical MOR roadmap's own explicitly-deferred Phase 5. Of those,
  ~~singular perturbation approximation~~ -- now implemented: SPA
  (Liu & Anderson 1989, also called residualization), as
  `SingularPerturbationROM`, a sibling class in the SAME module reusing
  the SAME balancing transform `BalancedTruncationROM` already computes,
  with an identical `from_MCK()`/`transfer_function()`/
  `frequency_response()`/`h_infinity_error_bound()`/`is_stable()` API.
  Validated in the new `test_singular_perturbation.py` (4 tests) against
  the same real fea_engine damped-cantilever fixture
  `test_balanced_truncation.py` uses. Unlike Phase 4e (`scm_lp.py`)
  above, where the hoped-for improvement did NOT materialize, this one
  did: at `r=10`, SPA's DC-gain error measures 1.7e-13 relative
  (machine precision) against the true full-order DC gain, vs. ordinary
  BT's 3.4e-5 relative at the same `r` on the same model -- roughly 8
  orders of magnitude worse -- while sharing the exact same a priori
  H-infinity error bound (2.3e-9 bound vs. 4.1e-12 observed sup-norm
  absolute error) and the same stability-preservation guarantee as
  ordinary BT (stable at every tested `r`: 4, 8, 10, 12), and without
  costing BT's own whole-sweep accuracy (max relative error 1.7e-4
  across the frequency sweep at `r=10`, comparable to ordinary BT's own
  global accuracy). ~~Frequency-weighted BT~~ -- now implemented: Enns
  (1984) frequency-weighted balanced truncation, as
  `FrequencyWeightedBalancedTruncationROM`, a third sibling class in the
  SAME module reusing the same balancing math via a newly factored-out
  `_balance_from_gramians()` (confirmed zero behavior change to the
  existing unweighted path -- `Wi=Wo=None` reduces exactly to ordinary
  `BalancedTruncationROM`, checked as a regression test). This one is
  the most nuanced of the three additions: a genuine, measured trade,
  not a clean win like SPA above or a dead end like Phase 4e (`scm_lp.py`)
  above. On the same fixture at `r=6`, targeting the beam's third
  natural frequency with `bandpass_weight(omega3, zeta=0.2)` as an
  output weight: 14.0x MORE accurate than ordinary BT in the targeted
  band (max relative error 0.238 vs 3.33, where ordinary BT at `r=6`
  essentially fails to represent this mode at all), but ~5.3x LESS
  accurate far from it, near the dominant first mode (1.9e-3 vs 3.6e-4)
  -- both measured directly, and the effect held across most (not all)
  of a small parameter sweep. Unlike ordinary BT/SPA, stability is NOT
  unconditionally guaranteed for two-sided (both `Wi` and `Wo`)
  weighting -- a real, literature-documented limitation of Enns'
  construction, genuinely checked (not assumed) across `r` in
  {4,6,8,10,12}: stable at `r=4,10,12`, unstable at `r=6,8`. One-sided
  weighting (`Wi` or `Wo` alone) preserved stability at every tested `r`
  (4, 6, 8, 10). `h_infinity_error_bound()` is deliberately not exposed
  on this class, since the classical a priori bound formula applies to
  the unweighted Hankel singular values specifically and a
  frequency-weighted version wasn't independently verified here --
  confirmed by a `hasattr` test, the same honest omission `krylov.py`
  already makes. Validated in the new `test_frequency_weighted_bt.py`
  (6 tests) against the same real fea_engine damped-cantilever fixture.
  ~~Coprime-factorization/passivity-preserving BT~~ -- investigated and
  closed out via a different, safer route rather than built as originally
  scoped: the standard positive-real-BT construction needs an algebraic
  Riccati equation requiring `D + D^T` invertible, but this package's
  structural systems always have `D=0` (no direct feedthrough), making
  that equation singular -- the same category of derivation risk
  `scm_lp.py`'s natural-norm SCM variant carries for complex,
  non-Hermitian operators (see Phase 4e above). Rather than push through
  an unverified derivation, the project pivoted to a closed-form result
  instead: for a collocated force-in/velocity-out system
  (`M qdotdot + C qdot + K q = B u`, `y = B^T qdot`), the mechanical
  energy `E = 0.5 qdot^T M qdot + 0.5 q^T K q` satisfies the KYP/
  positive-real dissipation inequality `dE/dt <= y^T u` exactly, given
  only that `M`, `K` are SPD and `C` is PSD -- properties this package
  already assumes everywhere, no Riccati equation needed. Because
  congruence transformation (`Phi^T M Phi`) preserves
  positive-(semi)definiteness unconditionally for ANY basis, this
  derivation applies unchanged to this package's EXISTING second-order
  Galerkin projection (`galerkin.py`/`frequency.py`), which therefore
  already provably preserves passivity -- a previously-unstated but
  latent property, not a new reduction algorithm. The genuinely new
  piece is a small diagnostic toolkit, `passivity.py`:
  `velocity_transfer_function()` converts any already-computed
  displacement frequency response to velocity (mobility) via
  multiplication by `i*omega`; `passivity_margin()` returns
  `Re[H_vel(i*omega)]`; `is_passive()` checks it stays non-negative.
  Validated in `test_passivity.py` (3 tests) against the same real
  fea_engine cantilever fixture used throughout this suite: the
  full-order model is passive (sanity check); a modally-reduced
  `FrequencyROM` stays passive at every tested basis size (3, 6, 10, 15
  modes) -- the closed-form theorem confirmed directly, not just cited;
  and sweeping the state-space (first-order) reduction methods
  (`KrylovROM`/`BalancedTruncationROM`/
  `FrequencyWeightedBalancedTruncationROM` at `r`/`k` in {6, 10, 14}, 9
  checks total) found 8 stayed passive but ONE genuinely violated it --
  `FrequencyWeightedBalancedTruncationROM` at `r=14` with a bandpass
  output weight centered on the first natural frequency -- real,
  measured evidence (not synthetic or forced) that none of these
  first-order reduction methods carry any passivity-preservation
  guarantee, unlike second-order Galerkin projection. This effectively
  closes out the originally-planned coprime-factorization BT item too:
  the specific coprime-factor construction itself was never built, but
  the underlying goal -- passivity preservation, verified rather than
  assumed -- is now addressed via this safer, more directly useful
  route. ~~Optimal Hankel norm approximation~~ -- now implemented:
  Adamjan-Arov-Krein/Glover (1984) theory, as `hankel_norm.py`'s
  `OptimalHankelNormROM` (a separate module, not a sibling class in this
  file, since its Gamma-inversion/Schur-plus-Sylvester construction is
  algorithmically unlike anything else here -- the same reasoning that
  already keeps `krylov.py` separate). Builds, in one application of the
  AAK formulas, an order-`r` model achieving `\|\|G-G_r\|\|_Hankel =
  sigma_(r+1)` exactly -- the true minimum over every possible order-`r`
  system, a stronger optimality claim than this module's own a priori
  H-infinity bound. SISO only, and guarded by an always-on empirical
  self-check (rather than an a priori threshold, since deliberate
  pre-implementation stress-testing showed no simple a priori quantity
  reliably predicts when the construction becomes numerically
  unreliable). Validated on the real fea_engine cantilever fixture in
  the new `test_hankel_norm.py` (6 tests): the AAK optimality claim was
  verified directly against `BalancedTruncationROM` at the same `r`, the
  self-check was shown to both agree with theory in the reliable regime
  and correctly flag a reproduced unreliable case, and an honest finding
  was reported -- Hankel-norm optimality did not translate into better
  practical (sup-norm/relative) accuracy than ordinary BT at the same
  `r` on this fixture. This closes out the classical MOR roadmap's Phase
  5 entirely (see `docs/classical_mor_roadmap.md` Sections 14-15).
- ~~A SECOND-ORDER-structure-preserving SOAR/TOAR Krylov path
  specifically~~ (Phase 5 of `docs/frequency_domain_rom_roadmap.md`) --
  the SOAR half is now implemented: `soar.py`'s `soar_basis()` +
  `SOARROM`, Section 8 of that doc, taken up at the user's request after
  the 5-phase roadmap above was already complete. Note this solves a
  DIFFERENT problem than what that Phase 5 item was originally scoped
  for (`FrequencyROM`'s own resonance-handling in the affine/POD basis)
  -- it instead gives a second-order-structure-preserving ALTERNATIVE to
  `krylov.py`'s own first-order moment matching, on the same general
  `(M,C,K,B,Cout)` I/O-port problem, avoiding the first-order form's 2x
  state-count blowup. Validated: exact moment matching through moment
  `k-1` for a `k`-vector basis on a synthetic system (catching, along
  the way, a real self-caught bug where feeding orthonormalized rather
  than raw vectors into the two-term recurrence silently capped accuracy
  at moment 1 regardless of `k`); on a real fea_engine beam with a 3-DOF
  block port, `SOARROM(k)` measurably OUTPERFORMS `KrylovROM(2k)` (same
  total reduced-state count) at small `k` (relative error 0.135 vs.
  0.885 at `k=2`); and, the actual motivation, `SOARROM` stays passive
  (per `passivity.py`'s own second-order-Galerkin theorem) at every
  tested order and expansion point on the real fixture, a guarantee
  `KrylovROM` does not have. An honest, literature-consistent limitation
  was found and reported, not hidden: for a SINGLE-DOF port on this
  beam's well-separated eigenvalues (`omega2/omega1 ~ 6.3`), the raw
  recursion collapses toward the dominant mode like inverse power
  iteration, and the achievable basis size plateaus at 1 regardless of
  how large `k` is requested -- exactly why real SOAR/TOAR usage favors
  block (multi-column) ports, confirmed directly (the 3-DOF block port
  above does not plateau at the tested orders). TOAR (the further
  numerical stabilization of SOAR's own known fragility) remains a
  deliberately-scoped-out follow-up, mitigated here instead via full
  reorthogonalization (matching `krylov.py`'s own `arnoldi_basis()`) --
  see `soar.py`'s own module docstring for the full reasoning.
- ~~Hyper-reduction, component mode synthesis, linear transient solvers and a standard validation report~~ -- now implemented: `hyper_reduction.py`, `cms.py`, `linear_dynamics.py` and `validation.py` (see the module table above; each validated against real fea_engine models in its own test file). Still open nearby: hyper-reduction of the TANGENT for very large models is done element-by-element here but not yet compared against DEIM on the same problem; component mode synthesis covers Craig-Bampton and Guyan only (no Hurty/Rubin free-interface or Craig-Chang variants); `validate_rom()` has no built-in extrapolation test, so include test inputs outside the training range yourself.
- Sphinx documentation (alongside fea_engine's).
