"""
rom_engine -- a small, from-scratch reduced-order-modeling package for
structural engineering analysis.

This is the companion package to fea_engine: fea_engine assembles and
solves full-order finite element models, rom_engine compresses and
accelerates them. The two are deliberately DECOUPLED -- rom_engine's
core modules take plain numpy arrays (K, M, F, snapshot matrices), not
fea_engine objects, so this package works with matrices from any FE
source, not just fea_engine. fea_engine models are used as the
VALIDATION fixtures in this package's own test suite (see tests/), and
as the worked examples in examples/, but the library code itself never
imports fea_engine.

Modules
-------
pod       : Proper Orthogonal Decomposition -- extract a reduced basis
            from a snapshot matrix (standard Euclidean, or mass-weighted
            for correct handling of physically dissimilar DOFs).
            MultiFieldPOD adds, on top of PodBasis, the per-field
            diagnostics (component norms, normalized component shapes,
            amplitude histories, field-separated reconstruction) a
            multi-physical-field snapshot (e.g. a rod's axial/
            transverse/rotation state) needs, per Georgiou (2005).
galerkin  : intrusive Galerkin projection -- project a full-order linear
            system (K, M, F) onto a reduced basis and solve the tiny
            reduced system (static and modal)
affine    : affine parametric decomposition -- the offline-online split
            that makes reduced-basis methods fast for many-query
            problems (parameter sweeps, optimization, UQ)
frequency : intrusive frequency-domain ROM -- A(omega) = -omega^2*M +
            i*omega*C + K is exactly an affine-in-omega system, so this
            module composes affine + galerkin (rather than duplicating
            their algebra) for fast harmonic-response sweeps, with
            either a modal or a POD-on-FRF-snapshots reduced basis.
            See docs/frequency_domain_rom_roadmap.md for the design
            background.
greedy    : weak-greedy adaptive training for FrequencyROM --
            FrequencyROM.residual_norm() (an efficient, exact residual)
            and .hierarchical_error_indicator() (a coercivity-constant-
            free error signal, deliberately chosen over the classical
            certified-bound approach because the literature documents
            that approach as weak near resonance) drive
            greedy_train_frequency_basis() to spend full-order solves
            where the current basis is weakest instead of on a fixed
            grid. See docs/phase4_error_bounds_greedy_roadmap.md.
scm       : a genuinely CERTIFIED lower bound on sigma_min(A(omega)),
            turning residual_norm() into a true a posteriori error
            BOUND (not just an estimate) via a simplified,
            Weyl/Mirsky-perturbation-based SCM-family method --
            rigorous by construction, honestly NOT the full classical
            LP-based Successive Constraint Method, and honestly weaker
            near resonance than away from it (this is a property of
            the underlying mathematics, not an implementation gap --
            see the module docstring). SingularValueLowerBound +
            certified_error_bound().
scm_lp    : the genuine classical LP-based Successive Constraint Method
            (Huynh, Rozza, Sen, Patera 2007) scm.py's own docstring
            flags as NOT implemented there -- a real online
            scipy.optimize.linprog LP over offline reference/control
            points, applied via the standard "SCM-squared" reduction
            (bound sigma_min(A(omega))^2 = lambda_min(A(omega)^H
            A(omega)), a REAL SYMMETRIC PSD eigenvalue problem for any
            complex, non-Hermitian A(omega), sidestepping a genuine,
            literature-acknowledged derivation ambiguity the sharper
            "natural-norm" SCM variant has for complex operators -- see
            docs/phase4_error_bounds_greedy_roadmap.md Section 9 for
            why that variant was deliberately NOT attempted).
            LPSingularValueLowerBound mirrors SingularValueLowerBound's
            API exactly (from_affine/add_reference/lower_bound/
            greedy_train) for direct comparison; its own
            certified_error_bound() is the LP-based analogue of scm.py's
            (import both under their module names to disambiguate, e.g.
            `from rom_engine.scm_lp import certified_error_bound as
            certified_error_bound_lp`). Measured finding (Section 10):
            on this package's real cantilever fixture, this bound's
            useful radius is narrower than scm.py's simplified one, not
            wider -- the SCM-squared reduction's squaring makes the
            same scale mismatch worse, exactly the failure mode the
            (unattempted) natural-norm variant exists to avoid. A real
            HiGHS numerical-conditioning issue in the greedy-selection
            upper_bound() (not the certified lower_bound() itself,
            which was verified correct) was found and fixed along the
            way -- see the module docstring and Section 10.
metrics   : generic, method-agnostic comparison metrics -- currently
            modal_assurance_criterion() (MAC) and r_squared() (the
            coefficient of determination, used by nonlinear_rom.py's
            validation but not tied to it).
loewner   : NON-INTRUSIVE modal parameter identification -- unlike
            every module above, LoewnerROM.fit() never sees M/C/K, only
            sampled complex frequency-response data x(omega). Builds a
            Loewner pencil (Mayo & Antoulas 2007) whose generalized
            eigenvalues are the system's identified poles, converts
            them to natural frequencies / damping ratios, and
            reconstructs mode shapes (Eq. 28) from multi-DOF response
            data. See docs/loewner_modal_identification_roadmap.md.
screening : automated cross-ROM stability screening for LoewnerROM --
            builds many LoewnerROMs from independent random
            interpolation-frequency subsets of a shared frequency pool
            and keeps only the eigenpairs that recur, tightly
            clustered, across a large fraction of them (Eqs. 30-32),
            separating genuine physical modes from an oversized
            interpolation order's spurious numerical artifacts.
            screen_physical_modes() + ScreenedMode.
sampling  : generalized Optimal Latin Hypercube Sampling (OLHS) design-
            of-experiments core -- optimal_lhs() (general-purpose, unit-
            hypercube, maximin criterion) and modal_force_samples()
            (a thin rescaling for the per-mode training-load convention
            nonlinear_rom.py's AppliedLoadStrategy needs). Step 1 of
            docs/nonlinear_surrogate_rom_roadmap.md.
nonlinear_rom : hybrid nonlinear structural ROM -- an intrusive linear
            modal basis (from pod.py/galerkin.py, or any external
            source) plus a non-intrusive, black-box regression of the
            nonlinear reduced restoring force. KERNEL_REGISTRY (4
            kernels, each with an analytic Jacobian) +
            MultiFidelitySurrogate (RBF fit of F_nl(q_l), the
            MFS-NLROM/He et al. 2023 approach) and PolynomialModalROM
            (Nash-form quadratic+cubic fit of F_nl(q_nl) + Newton
            solve, the ICE/Shi & Mei/STEP family -- with ICEROM/
            ShiMeiROM/EnforcedDisplacementROM convenience constructors),
            both implementing the shared ReducedForceModel protocol.
            AppliedLoadStrategy + EnforcedDisplacementStrategy generate
            training data via a caller-supplied fom_solver callable
            (never imports fea_engine, per this module's own design
            principle below). Ported from, and validated against, this
            project's own earlier mfs-nlrom-beam prototype plus a real
            fea_engine clamped-clamped Beam2DCorotational model (see
            tests/test_nonlinear_rom.py / test_nonlinear_rom_fea.py).
            Step 2 of docs/nonlinear_surrogate_rom_roadmap.md. Also in
            this module: NeuralSurrogate (Wave 7 item 35, docs/
            consolidated_future_roadmap.md, the fea_engine-side
            planning document) -- a plain torch.nn MLP alternative to
            MultiFidelitySurrogate, fitting the SAME F_nl(q_l) protocol
            (any TrainingStrategy's output feeds it unchanged), with
            jacobian() computed via torch.autograd.grad row-by-row --
            the same technique fea_engine.autograd_tangent already
            uses for element tangent stiffnesses, applied here to a
            ROM force surrogate. Optional dependency (only
            instantiating NeuralSurrogate needs torch; importing
            rom_engine/nonlinear_rom.py never does), matching
            fea_engine's own _HAS_TORCH/_require_torch() convention.
nonlinear_dynamics : generalized reduced nonlinear time integration --
            integrate_newmark_surrogate() separates the METHOD (Newmark-
            beta predictor + a surrogate-lag correction scheme, entirely
            Newton-free) from the MODEL (any nonlinear_rom.ReducedForceModel).
            Correction modes -- "none" (cheapest baseline),
            "fixed_point" (Picard-iterated, ported from this project's
            own validated mfs-nlrom-beam prototype; DEFAULT since
            2026-09-24), "sign_deviation" (reconstruction of the MFS-
            NLROM paper's Eq. 32 idea; known to diverge on strongly
            nonlinear problems, kept for reproducing that paper only),
            "newton" (q_nl domain only) -- all validated against
            closed-form/scipy-ODE references (test_nonlinear_dynamics.py)
            and against a REAL fea_engine clamped-clamped Beam2DCorotational
            free-decay run via nonlinear_solver.solve_nonlinear_transient
            (test_nonlinear_dynamics_fea.py). Step 4 of
            docs/nonlinear_surrogate_rom_roadmap.md.
nnm       : multi-harmonic-balance (HBM) + Alternating Frequency-Time (AFT)
            nonlinear normal mode (NNM) backbone continuation --
            HarmonicBalanceSystem assembles the linear-in-Z harmonic-balance
            operator A(omega) (real cos/sin block form, deliberately
            mirroring frequency.FrequencyROM's own complex A(omega) --
            checked directly by an exact n_harmonics=1 cross-check against
            it) for any nonlinear_rom.ReducedForceModel via AFT. TWO
            continuation drivers share this same AFT/HBM machinery:
            solve_nnm_backbone() Newton-continues the periodic-orbit family
            at a DIRECTLY PRESCRIBED master-mode amplitude (natural-
            parameter continuation -- simple, but cannot pass a genuine
            amplitude fold), and solve_nnm_backbone_arclength() instead
            uses genuine PSEUDO-ARCLENGTH continuation (Keller 1977, with
            a documented secant-predictor simplification and an omega-
            rescaling fix for a real mixed-units arclength-metric issue
            found during its own validation -- see module docstring) that
            CAN continue through such a fold, at the cost of amplitude
            becoming a measured output rather than a prescribed input.
            No literal HBM prototype existed to port from (the earlier
            mfs-nlrom-beam reproduction used time-integration + FFT period
            measurement instead, due to OCR-damaged source equations for
            the paper's own HBM section), so this module was built from
            first-principles HBM/AFT theory and validated multiple ways:
            AFT/linear-algebra self-consistency, the FrequencyROM cross-
            check above, an independent scipy-ODE Duffing-oscillator
            backbone, a fully analytic textbook fold (the unit circle,
            validating the arclength algorithm's core "turn a fold" claim
            with total certainty) plus a direct agreement check against
            solve_nnm_backbone on the SAME non-folding backbone, and an
            exact-reduction check on a genuinely 2-mode coupled system's
            invariant planar branch (test_nnm.py) -- then against a REAL
            fea_engine clamped-clamped Beam2DCorotational backbone,
            measured independently via the same time-integration +
            FFT-period convention as that earlier prototype
            (test_nnm_fea.py). Step 5 (original, final phase of the
            nonlinear-surrogate-ROM plan) plus a later addendum (pseudo-
            arclength) of docs/nonlinear_surrogate_rom_roadmap.md
            Section 11.

state_space : convert a second-order structural system (M, C, K) plus an
            explicit INPUT map B and OUTPUT map Cout into first-order
            state-space form -- the shared prerequisite krylov.py and
            balanced_truncation.py both need (neither pod/galerkin/
            affine/frequency ever needed a compressed input/output
            "port" view before). Two forms: "E" (descriptor, no
            M-inversion -- krylov.py's choice) and "A" (explicit ODE
            form, one dense M-solve -- balanced_truncation.py's choice,
            since scipy's Lyapunov solver has no generalized-E variant).
            See docs/classical_mor_roadmap.md.
mode_correction : mode acceleration + modal truncation augmentation
            (Besselink et al. 2013, eqs. 13-20) -- two cheap, closed-
            form corrections that recover the STATIC contribution a
            truncated modal basis leaves out, using one extra full-order
            static solve. mode_acceleration_response() is a post-hoc
            correction on top of an ordinary GalerkinROM.solve_static();
            augmented_basis() instead folds the correction INTO the
            basis (Psi=[V, q_cor]), so it works directly with the
            EXISTING GalerkinROM -- no new solver logic either way. Both
            are proved (and tested) to reproduce the exact full-order
            static solution to machine precision for the load the
            correction was built from -- an algebraic identity, not an
            approximation.
krylov    : Krylov-subspace moment matching (Besselink et al. 2013
            Section 3 / eqs. 21-36) -- arnoldi_basis() builds a
            block-Arnoldi basis of the shifted Krylov subspace
            K_k((A-s0*E)^-1 E, (A-s0*E)^-1 B) via one reused LU
            factorization; KrylovROM.from_MCK() projects the
            state_space.py "E" form onto it. ONE-SIDED (Galerkin,
            default) matches the full-order transfer function's Taylor
            moments EXACTLY at the expansion point s0, with error
            growing away from it -- a documented, tested, EXPECTED
            property, not a flaw. TWO-SIDED (Petrov-Galerkin,
            from_MCK(..., two_sided=True)) -- two_sided_arnoldi_bases()
            biorthogonalizes a second basis built from the dual system
            (A^H, E^H, Cout^H) -- matches roughly TWICE as many moments
            from the same reduced order (proved exactly, on a small
            well-conditioned synthetic system, in
            test_krylov.py::test_two_sided_matches_twice_as_many_moments_as_one_sided),
            at the cost of a genuinely fragile biorthogonalization that
            raises ValueError rather than silently degrading when it
            breaks down. Unlike balanced_truncation.py, gives NO a
            priori stability guarantee either way (KrylovROM.is_stable()
            checks this directly rather than assuming it, matching a
            real instability the paper's own benchmark found at larger
            order). Second-order-structure-preserving SOAR Krylov is now
            implemented separately -- see soar below -- since it solves
            a genuinely different problem (matching the SAME transfer
            function's moments WITHOUT the 2x first-order state blowup);
            krylov.py's own module docstring still notes this as
            historically out of scope for krylov.py ITSELF.
soar      : SOAR (Second-Order Arnoldi, Bai & Su 2005) -- a SECOND-ORDER-
            structure-preserving alternative to krylov.py's own
            first-order moment matching, built as a Phase 5 addendum to
            docs/frequency_domain_rom_roadmap.md (Section 8).
            soar_basis() builds an orthonormal basis of the second-order
            Krylov subspace defined by r_1=K0^-1 B,
            r_j=A1 r_{j-1}+A2 r_{j-2} (A1, A2 derived from a Taylor
            expansion of (s^2 M+sC+K)^-1 around s0) directly from
            (M,C,K), never forming the 2*n_dof first-order pencil;
            SOARROM.from_MCK() projects M/C/K/B/Cout onto it -- an
            ordinary Galerkin projection (M_r=V^T M V, ...), which is
            exactly what makes passivity.py's own second-order-Galerkin
            passivity theorem apply, unlike krylov.py's first-order
            KrylovROM. Only SOAR is implemented, deliberately NOT TOAR
            (the further numerical stabilization of SOAR's own known
            fragility) -- a stated scope decision (soar.py's own module
            docstring), mitigated instead via full reorthogonalization
            (as krylov.py's own arnoldi_basis() already does). A real
            implementation bug was caught and fixed during development
            (feeding orthonormalized, rather than raw, vectors into the
            two-term recurrence silently capped moment-matching accuracy
            regardless of basis size -- see soar.py's module docstring
            for the honest account) and a genuine SOAR fragility was
            found and reported, not hidden: for a SINGLE-DOF port on a
            well-separated-eigenvalue structure, the raw recursion
            collapses toward the dominant mode like inverse power
            iteration, and the achievable basis size plateaus well below
            any requested k -- exactly why real SOAR/TOAR usage favors
            BLOCK (multi-column) ports, confirmed directly: a 3-DOF
            block port on the same real fea_engine beam does NOT
            plateau at the tested orders, and SOARROM(k) there
            measurably outperforms KrylovROM(2k) (the same total
            reduced-state count) at small k.
random_vibration : PSD (power spectral density) random-vibration response --
            psd_response(rom_freq, freqs_hz, psd_input, F0_pattern,
            output_dofs) reuses FrequencyROM.frequency_response()
            (rather than re-deriving anything) to compute S_out(f) =
            |H(f)|^2 * S_in(f) and the response RMS, mirroring
            fea_engine.solver.FESystem.solve_random_vibration()'s own
            already-validated formula exactly. A single output_dofs int
            matches that method's signature 1:1 (scalar return); an
            array-like of several DOFs is a small generalization it
            doesn't offer. See
            docs/frequency_domain_rom_roadmap.md Phase 6.
balanced_truncation : balanced truncation (Besselink et al. 2013 Section
            4 / eqs. 37-58) -- the one of the paper's three numerically-
            applied methods that stays accurate GLOBALLY (not just near
            one point) because it explicitly uses where the inputs and
            outputs are. controllability_gramian()/observability_
            gramian() solve the two continuous Lyapunov equations via
            scipy's dense Bartels-Stewart solver; hankel_singular_values()
            uses the numerically preferred SQUARE-ROOT method (Cholesky/
            eigen factors + one small SVD, never an explicit
            eigendecomposition of the Gramian product); BalancedTruncationROM.
            from_MCK() balances and truncates the state_space.py "A"
            form. is_stable() is a genuine THEOREM for BT (Pernebo &
            Silverman 1982), unlike krylov.KrylovROM's; h_infinity_error_
            bound() gives a real, checked-to-actually-hold a priori
            bound on the worst-case ABSOLUTE transfer-function error --
            a certificate krylov.KrylovROM structurally has no
            equivalent of. A practical scale note worth knowing before
            calling this on a raw FE model directly: see
            balanced_truncation.py's own module docstring and
            tests/test_balanced_truncation.py for why (and how) to
            modally pre-truncate a finely-meshed FE model first.
            Also in this module: SingularPerturbationROM (singular
            perturbation approximation / "residualization" -- Liu &
            Anderson 1989), a sibling reduction reusing the SAME
            balancing transform BalancedTruncationROM computes. Fixes
            ordinary BT's one real weakness: BT's D_r matches the full-
            order D exactly (= 0 here) so it is exact at s=infinity but
            generally wrong at s=0 (DC/steady-state); SPA instead
            solves the truncated states' algebraic quasi-steady-state
            constraint (x2'=0, not x2=0) and gets a nonzero D_r that
            makes IT exact at s=0 instead -- the state-space analogue
            of what mode_correction.py already does for a plain modal
            basis. Shares BT's SAME a priori H-infinity error bound and
            SAME stability-preservation theorem (both re-verified, not
            assumed, in tests/test_singular_perturbation.py); identical
            from_MCK()/frequency_response()/h_infinity_error_bound()/
            is_stable() signatures to BalancedTruncationROM's, for
            direct side-by-side use. Measured finding (docs/classical_
            mor_roadmap.md Section 9): on this package's real
            cantilever fixture, SPA's DC-gain error was ~8 orders of
            magnitude smaller than ordinary BT's at the same r
            (1.7e-13 vs. 3.4e-5) -- the hoped-for improvement, actually
            confirmed empirically here (unlike scm_lp.py's Phase 4e,
            where the analogous hoped-for improvement did NOT hold).
            Also in this module: FrequencyWeightedBalancedTruncationROM
            (Enns 1984 frequency-weighted BT), a third sibling reusing
            _balance_from_gramians() -- the shared square-root
            balancing step now factored out of hankel_singular_values()
            itself, fed frequency_weighted_gramians()'s weighted
            Gramians (from a cascade-system construction with input/
            output filter realizations Wi/Wo -- lowpass_weight()/
            bandpass_weight() build common ones) instead of the plain
            ones. Ranks balanced states by importance in a CHOSEN
            frequency band rather than uniformly across all frequencies
            -- a real trade (better in-band, worse out-of-band), not a
            free improvement, and measured as such: on this package's
            real cantilever fixture, a bandpass output weight gave a
            14x more accurate reduced model in the targeted band at the
            same r, but ~5x LESS accurate far from it (docs/classical_
            mor_roadmap.md Section 11). With Wi=Wo=None this reduces
            exactly to ordinary BalancedTruncationROM (checked as a
            regression test on the _balance_from_gramians() refactor
            itself). Two-sided weighting (both Wi and Wo) has NO
            unconditional stability guarantee (a real literature
            limitation of Enns' method, not an implementation gap) --
            reproduced concretely (stable at some r, unstable at
            others on the same fixture) rather than only cited;
            one-sided weighting DOES preserve stability and was
            checked to. is_stable() is therefore genuinely checked here
            (like krylov.KrylovROM's), not a theorem-backed regression
            guard (unlike BalancedTruncationROM/SingularPerturbationROM's);
            h_infinity_error_bound() is deliberately NOT exposed, for
            the same reason KrylovROM omits it (no independently-
            verified a priori bound for this method here).
passivity : passivity (positive-realness) diagnostics for a collocated
            force-in/velocity-out structural port -- velocity_transfer_
            function(), passivity_margin(), is_passive(). Built as a
            deliberate scope adjustment from "positive-real balanced
            truncation" (docs/classical_mor_roadmap.md Section 12):
            that construction needs an algebraic Riccati equation that
            is singular for this package's D=0 structural realizations
            (an analogous derivation risk to scm_lp.py's complex-
            operator ambiguity), so this module instead uses a CLOSED-
            FORM passivity certificate (the total mechanical energy
            0.5 q'^T M q' + 0.5 q^T K q satisfies the KYP dissipation
            inequality exactly, given only M/K SPD and C PSD -- already-
            standing assumptions throughout this package) that
            transfers UNCHANGED to ANY second-order Galerkin projection
            (galerkin.py's basis, or a frequency.FrequencyROM on top of
            it), for ANY basis, not just a modal one -- a genuinely new
            property of code this package already has, not a new
            reduction algorithm. The actual new capability is the
            DIAGNOSTIC: converts any of this package's displacement
            frequency_response() outputs to a velocity/mobility one
            (multiply by i*omega -- no new state-space plumbing needed)
            and checks Re[H_vel(i*omega)] >= 0 (the frequency-domain
            KYP condition). Validated (tests/test_passivity.py) against
            a real fea_engine fixture: the full-order model and every
            tested modally-reduced FrequencyROM stayed passive
            (confirming the theorem directly, not just citing it), and
            -- the genuinely open question this diagnostic exists to
            answer -- sweeping KrylovROM/BalancedTruncationROM/
            FrequencyWeightedBalancedTruncationROM at several reduced
            orders found ONE real passivity violation
            (FrequencyWeightedBalancedTruncationROM at r=14 with an
            output weight), concrete evidence that none of those
            methods' other guarantees imply passivity preservation.

hankel_norm : optimal Hankel norm approximation (Adamjan-Arov-Krein
            theory; explicit realization due to Glover 1984) --
            OptimalHankelNormROM, the classical MOR roadmap's last
            explicitly-named Phase 5 balanced-truncation-family
            extension (docs/classical_mor_roadmap.md Section 14). Given
            a SIMPLE Hankel singular value sigma_(r+1) (checked, not
            assumed), builds -- in ONE application of the AAK formulas,
            reusing balanced_truncation.hankel_singular_values() for
            the balancing step -- an order-r model with
            ||G-G_r||_Hankel = sigma_(r+1) EXACTLY, the true minimum
            over EVERY possible order-r system (stable or not), not
            just a bound: a genuinely stronger (in the Hankel-norm
            sense) optimality claim than BalancedTruncationROM's own a
            priori H-infinity bound. SISO only (a scalar division in
            the derivation, not re-derived for MIMO -- raises
            ValueError otherwise, per this package's established
            practice of declining to guess at an unverified
            generalization, e.g. scm_lp.py/passivity.py). A real,
            measured numerical limitation (found via a deliberate
            random-synthetic-system stress test before this module was
            written, not merely anticipated): the construction can
            become unreliable when sigma_(r+1) is very small relative
            to the largest Hankel singular value, and no single simple
            a priori quantity tried (including cond(Gamma)) reliably
            predicted it -- so EVERY construction instead runs an exact
            EMPIRICAL self-check (building the error system G-G_r and
            independently measuring ITS Hankel norm via the same
            hankel_singular_values(), which the AAK theorem says must
            equal sigma_(r+1) exactly), exposed as
            numerically_reliable/measured_hankel_norm_error, with a
            loud warning rather than a silent wrong answer when it
            fails. Validated on this package's real cantilever fixture
            (tests/test_hankel_norm.py): the core AAK optimality claim
            was checked DIRECTLY (OptimalHankelNormROM's Hankel-norm
            error is <= BalancedTruncationROM's own at the same r, at
            every tested r) and the self-check machinery was shown to
            both agree with theory to near machine precision in the
            reliable regime AND correctly flag a genuinely unreliable
            case reproduced on this fixture. An honest finding
            (mirroring frequency_weighted BT's own trade-off
            reporting): Hankel-norm optimality did NOT translate into
            better sup-norm/relative frequency-response accuracy than
            ordinary BalancedTruncationROM at the same r on this
            fixture -- measured directly, not assumed either way. The
            tight L-infinity bound Glover's theory also gives is
            deliberately NOT exposed as a method (needs an additional
            recursive correction step not independently re-derived
            here) -- the same honest-omission precedent
            KrylovROM/FrequencyWeightedBalancedTruncationROM already
            set.

dataset_diagnostics : Wave 10 item 104 (fea_engine/docs/consolidated_
            future_roadmap.md, source Saverio et al. 2026 Appendix B) --
            a standalone, torch-free design-of-experiments diagnostics
            utility, useful the moment any future parameter-sweep
            training dataset is generated (a ROM training set spanning
            geometry/material variations, in the spirit of mfs-nlrom-
            beam's own static-pressure-sweep dataset). parameter_
            coverage_report() (per-parameter histograms + gap flags),
            pca_dimensionality() (SVD-based dimensionality estimate,
            no scikit-learn dependency), and a minimal hand-rolled
            GPSurrogate + sobol_indices() (Saltelli 2010/Jansen 1999
            first- and total-order Sobol' sensitivity estimators,
            using the GP as a cheap stand-in for an expensive full-
            order solver's own repeated evaluation).
ensemble_uq : Wave 10 item 105 -- a thin wrapper (EnsembleUQ), not a
            new algorithm: trains N independently-seeded copies of any
            fit()/predict() model family (this package's own
            ReducedForceModel duck-typed protocol) and reports
            predictive mean/std across the ensemble, a cheap
            epistemic-uncertainty proxy (Saverio et al. 2026 Sec
            4.4.2/Fig 16). neural_surrogate_ensemble() is the concrete
            NeuralSurrogate-based convenience constructor (torch-
            optional at import time, like NeuralSurrogate itself).
neural_operator : Wave 13 item 117 (docs/consolidated_future_roadmap.md,
            reviewed against Yang et al. 2025's FRINO paper and this
            project's own sibling `L-NeuralODE` repository) --
            reduced_eom_residual_trajectory() (pure NumPy, unconditional:
            a central-difference residual of the continuous reduced EOM
            on any discrete trajectory, decisively order-of-convergence
            validated) + ExcitationResponseOperator, a compact 1-D
            Fourier Neural Operator (SpectralConv1d/FNO1dBlock) mapping
            a whole modal-force-excitation waveform directly to the
            reduced response trajectory, with an optional physics-
            residual training penalty built on the same residual
            function. UNLIKE NeuralSurrogate above, ExcitationResponse
            Operator genuinely subclasses torch.nn.Module (an FNO's
            spectral-conv weights need real Parameter registration), so
            it -- and LatentRefinementNet/ParameterizedODEFunc/
            ParameterizedLatentODE below -- can only be DEFINED when
            torch is importable, not just instantiated; rom_engine's own
            top-level `__init__.py` therefore does NOT re-export these
            four classes (unlike every torch-optional class elsewhere in
            this package) -- import them directly from their own module,
            e.g. `from rom_engine.neural_operator import
            ExcitationResponseOperator`, on a machine with PyTorch
            installed.
reduced_basis_operator : Wave 13 item 118 -- RegularizedProjectionEncoder
            (pure NumPy, unconditional): represents a discrete POD basis
            as a continuous field via a thin-plate-spline RBF
            interpolant, making encode()/decode() callable at ANY query
            coordinates, not just the basis's own training nodes --
            genuinely NEW to this package: a basis fit at one mesh
            resolution correctly reconstructs a full-order snapshot
            solved at a DIFFERENT resolution (measured ~0.5%-2.2% cross-
            resolution error vs ~0.03% same-resolution, on a real
            fea_engine fixture -- see the module's own docstring).
            LatentRefinementNet (torch-gated, see neural_operator's own
            note above) is a small residual bias-correction network.
parameterized_latent_ode : Wave 13 item 119 -- rk4_step()/integrate_rk4()
            (pure NumPy, unconditional): a CORRECTLY implemented generic
            4-stage RK4 integrator, written specifically to fix a real
            bug found while reviewing the sibling `L-NeuralODE`
            repository's own rk4() (k2/k3 evaluated at the un-advanced
            state, silently degrading a claimed 4th-order method to
            1st-order -- reproduced and measured directly in this
            module's own test file: ~16x error reduction per dt-halving
            for the fix vs ~2x for the bug). CurriculumSchedule (pure
            NumPy) is a small, independently-tested progressive-horizon
            bookkeeping state machine. ParameterizedODEFunc/
            ParameterizedLatentODE (torch-gated) train dz/dt=f(t,xi,z)
            across a sweep of a held-fixed physical parameter xi via
            this curriculum schedule, predicting whole trajectories from
            (z0, xi) directly -- a genuinely different, less physically-
            constrained trade-off than nonlinear_rom.ReducedForceModel's
            instantaneous state->force protocol, not a replacement for it.
wave13_validation : Wave 13 item 120 -- a cross-configuration validation
            harness tying items 117/118/119 together on ONE real
            fea_engine fixture pair (rather than re-proving each item's
            own already-tested claim in isolation): two independent
            integrators of the same fitted reduced dynamics (Newton-
            Newmark, item 114; item 119's own RK4) are cross-checked
            against each other AND against item 117's own EOM residual
            function, and item 118's cross-mesh-resolution reconstruction
            is re-run as part of the same combined report. The torch-
            gated cross_check_operators_torch() (same non-export note as
            neural_operator's own classes) additionally trains an
            ExcitationResponseOperator on a real fixture's reduced
            dynamics and checks it against a held-out excitation
            frequency's Newton-Newmark reference.
differentiable_correction : Wave 10 item 103 -- a DESIGN NOTE (docs/
            differentiable_rom_correction_design.md) + prototype
            applying fea_engine.differentiable's item-100 explicit
            residual-minimization calibration pattern to this
            package's OWN reduced equilibrium equation (Lambda*q +
            F_nl(q) = F_ext, PolynomialModalROM's own Newton-solved
            condition) instead of a full finite-element residual.
            Independently reimplemented (not imported from
            fea_engine.differentiable), per this module's own
            docstring, matching NeuralSurrogate's own precedent for
            never taking fea_engine as a library dependency.
            reduced_residual() is pure NumPy; calibrate_reduced_
            correction_explicit()/ScalarModalCorrection are torch-
            optional. Deliberately scoped as a first interface-
            boundary step, not a full port of fea_engine item 98's
            implicit-adjoint layer -- see the design doc's own "not
            built here" section.

intrusive_nonlinear_rom : Wave 17 item 144 (fea_engine/docs/
            consolidated_future_roadmap.md, Georgiou 2005) --
            IntrusiveNonlinearROM(V, M, C, internal_force_fn, load_fn,
            tangent_fn=None): an intrusive nonlinear Galerkin ROM built
            on a FULL (non-diagonal, non-mass-normalized) M_r/D_r/K_r
            -- deliberately separate from nonlinear_dynamics.py, whose
            M_r=I / diagonal-Lambda / surrogate-force contract does not
            fit this method (see this module's own docstring for the
            full reasoning). The reduced nonlinear force
            f_int_r(q) = V^T @ internal_force_fn(V@q) is evaluated
            through the REAL full-order internal-force routine every
            query (no hyper-reduction -- out of scope per this item's
            own roadmap row). Three explicitly separate integrators,
            none replacing another: integrate_rk4() (fixed-step,
            reusing parameterized_latent_ode.rk4_step()'s already-
            validated generic RK4), integrate_solve_ivp() (scipy on the
            identical first-order right-hand side), and
            integrate_newton_newmark() (a genuine per-step Newton-
            Newmark corrector, generalizing nonlinear_dynamics.
            integrate_newmark_surrogate()'s Wave 12 item 114
            correction="newton" mode from a mass-normalized M_r=I to a
            general M_r). initial_conditions() projects through M_r
            (mass-weighted, not a naive V^T u0); the module-level
            impulse_to_reduced_velocity() implements Georgiou's own
            Eqs. 47-48 impulse-to-velocity mapping generically.

            Wave 17 item 145 lives in the SAME module:
            pod_rm_analysis(M_r, D_r, K_r) -- POD-RM modal diagnostics
            (Georgiou 2005 Eqs. 41-44, 50-60). Generalized eigenproblem
            eigh(K_r, M_r) for the RM frequencies (correct for a full,
            non-diagonal M_r); a Euclidean-unit-normalized, sign-aligned
            (positive largest-magnitude entry per column, matching item
            143's MultiFieldPOD convention) eigenvector matrix E_hat,
            returned BOTH in eigh's own ascending-frequency order and in
            a "POD-mode order" found by an optimal one-to-one assignment
            (scipy.optimize.linear_sum_assignment) of eigenpairs to
            original POD coordinates; diag(E_hat) in POD-mode order as
            the paper's own near-identity diagnostic (exposed directly,
            not buried); uncoupled 1-DOF frequencies sqrt(K_mm/M_mm)
            per original POD mode (Eq. 55) as a coupled-vs-uncoupled
            contrast; and a symmetry/positive-definiteness report per
            matrix (MatrixSPDReport), reusing item 144's own test suite's
            eigenvalue-margin style. Returns a PodRMAnalysis NamedTuple.

Note: pod/galerkin/affine's array-handling now preserves complex dtype
(needed by frequency.py's complex A(omega)) instead of the earlier
hardcoded dtype=float -- every pre-existing real-valued use is
unaffected (see each module's _as_array() docstring).

Roadmap: the 5-phase nonlinear-surrogate-ROM plan in
docs/nonlinear_surrogate_rom_roadmap.md (sampling.py, nonlinear_rom.py,
nonlinear_dynamics.py, and nnm.py above) is now fully implemented and
validated. The 4-phase classical-MOR plan in
docs/classical_mor_roadmap.md (state_space.py, mode_correction.py,
krylov.py, balanced_truncation.py above -- closing the gap between this
package and Besselink et al. 2013's own numerically-applied methods) is
now FULLY implemented and validated, including its originally-optional
Phase 3 (two-sided/Petrov-Galerkin Krylov, added to krylov.py). The
PSD-driven random-vibration stress analyzer built on top of
frequency.py's sweep (random_vibration.py, Phase 6 of
docs/frequency_domain_rom_roadmap.md) is also now implemented and
validated. The genuine classical LP-based Successive Constraint Method
(scm_lp.py, Phase 4e of docs/phase4_error_bounds_greedy_roadmap.md) is
also now implemented and validated -- unconditionally certified, but
measured to be narrower than scm.py's simplified bound on this
package's real cantilever fixture (Section 10 of that doc), not the
hoped-for improvement; both remain available side by side. Singular
perturbation approximation (SingularPerturbationROM, Phase 5a of
docs/classical_mor_roadmap.md Section 8-9) is also now implemented and
validated -- here the hoped-for improvement DID hold empirically (SPA's
DC-gain error ~8 orders of magnitude smaller than ordinary BT's at the
same r on the real cantilever fixture). Frequency-weighted BT
(FrequencyWeightedBalancedTruncationROM, Phase 5b of docs/classical_
mor_roadmap.md Section 10-11) is also now implemented and validated --
its trade-off (in-band gain, out-of-band cost, no unconditional two-
sided stability guarantee) was measured concretely on both sides
rather than assumed favorable. Passivity preservation (passivity.py,
Phase 5c of docs/classical_mor_roadmap.md Sections 12-13) is also now
implemented and validated -- via a deliberate scope adjustment away
from positive-real balanced truncation's singular-for-D=0 Riccati
equation, toward a closed-form theorem (any second-order Galerkin
projection preserves passivity) plus a diagnostic; that diagnostic
then found a real passivity violation in
FrequencyWeightedBalancedTruncationROM at r=14 on the real fixture,
concrete evidence for exactly the distinction the scope adjustment
predicted. Optimal Hankel norm approximation (hankel_norm.py's
OptimalHankelNormROM, Phase 5d of docs/classical_mor_roadmap.md Section
14) is also now implemented and validated -- the classical MOR
roadmap's last explicitly-named Phase 5 item, closing that roadmap out
entirely (SPA, frequency-weighted BT, passivity preservation, and now
optimal Hankel norm approximation are all done; coprime-factorization
BT specifically was scoped out in favor of the passivity route, see
Section 12's reasoning). Its AAK-optimality claim was verified directly
against BalancedTruncationROM on the real fixture, and a genuine
numerical fragility (found via deliberate stress-testing before the
module was written -- unreliable when sigma_(r+1) is very small
relative to the largest Hankel singular value) is guarded by an
always-on empirical self-check rather than an a priori threshold, since
no simple a priori quantity tried predicted it reliably. Possible
future work, not currently planned: the natural-norm SCM variant
(deliberately not attempted for scm_lp.py -- see that doc's Section 9
for the literature-acknowledged complex-operator derivation risk this
would need to resolve first) if a real use case needs a wider
certified radius than either existing bound gives. ~~Second-order-
structure-preserving SOAR/TOAR Krylov, if plain first-order krylov.py
proves insufficient for a large target model~~ -- the SOAR half is now
implemented (soar.py, docs/frequency_domain_rom_roadmap.md Section 8);
TOAR remains a deliberately-scoped-out follow-up (see soar.py's own
module docstring). A MIMO generalization of
OptimalHankelNormROM's AAK construction, if a real use case needs it
(not attempted here -- see hankel_norm.py's own module docstring for
why); a from-scratch positive-real-ARE implementation for the general
D!=0 case, if a future non-structural use case actually needs it (not
planned, since this package's models are always D=0). Within nnm.py
itself: pseudo-arclength continuation (for backbones with a genuine
amplitude fold) is now implemented -- solve_nnm_backbone_arclength(),
docs/nonlinear_surrogate_rom_roadmap.md Section 11 -- validated on a
fully analytic textbook fold (the unit circle), a direct agreement
check against solve_nnm_backbone on a non-folding backbone, and an
exact-reduction check on a genuinely 2-mode coupled system's invariant
planar branch; a fully characterized, hand-verified FOLD on a real
multi-mode NNM backbone (e.g. near an internal resonance) was explored
but not conclusively established within scope -- honestly noted as a
natural next validation step in that section, not a blocker, given the
algorithm's own correctness is independently confirmed by the other
three checks. Subharmonic-multiplier support remains unimplemented
(explicitly scoped out of Step 5, see nnm.py's own module docstring) --
a materially larger extension (non-integer-multiple harmonics in the
AFT scheme) than pseudo-arclength turned out to be, not attempted here.
"""
from .pod import PodBasis, MultiFieldPOD, assemble_field_weight_matrix, trapezoidal_field_gram
from .galerkin import GalerkinROM
from .affine import AffineDecomposition
from .frequency import FrequencyROM, build_pod_basis_from_frf_snapshots
from .greedy import greedy_train_frequency_basis
from .random_vibration import psd_response, band_limited_gaussian_time_history
from .scm import SingularValueLowerBound, certified_error_bound
from .scm_lp import LPSingularValueLowerBound, certified_error_bound as certified_error_bound_lp
from .metrics import modal_assurance_criterion, r_squared
from .loewner import LoewnerROM
from .screening import screen_physical_modes, ScreenedMode
from .sampling import optimal_lhs, modal_force_samples
from .nonlinear_rom import (
    KERNEL_REGISTRY, ReducedForceModel, MultiFidelitySurrogate, PolynomialModalROM,
    NeuralSurrogate,
    TrainingStrategy, AppliedLoadStrategy, EnforcedDisplacementStrategy,
    TrajectoryPilotedStrategy,
    ICEROM, ShiMeiROM, EnforcedDisplacementROM,
)
from .nonlinear_dynamics import integrate_newmark_surrogate
from .nnm import HarmonicBalanceSystem, solve_nnm_backbone, solve_nnm_backbone_arclength
from .state_space import to_state_space, StateSpaceSystem
from .mode_correction import (
    mode_acceleration_correction, mode_acceleration_response, augmented_basis,
)
from .krylov import arnoldi_basis, two_sided_arnoldi_bases, KrylovROM
from .soar import soar_basis, SOARROM
from .balanced_truncation import (
    controllability_gramian, observability_gramian, hankel_singular_values,
    BalancedTruncationROM, SingularPerturbationROM,
    frequency_weighted_gramians, frequency_weighted_hankel_singular_values,
    FrequencyWeightedBalancedTruncationROM, lowpass_weight, bandpass_weight,
)
from .passivity import velocity_transfer_function, passivity_margin, is_passive
from .hankel_norm import OptimalHankelNormROM
from .dataset_diagnostics import (
    parameter_coverage_report, pca_dimensionality, GPSurrogate, sobol_indices,
)
from .ensemble_uq import EnsembleUQ, neural_surrogate_ensemble
from .differentiable_correction import (
    reduced_residual, ScalarModalCorrection, calibrate_reduced_correction_explicit,
)
from .membrane_expansion import MembraneBasis
from .neural_operator import reduced_eom_residual_trajectory
from .reduced_basis_operator import RegularizedProjectionEncoder
from .parameterized_latent_ode import rk4_step, integrate_rk4, CurriculumSchedule
from .wave13_validation import (
    cross_check_reduced_dynamics, cross_check_basis_generalization,
    run_cross_configuration_validation,
)
from .intrusive_nonlinear_rom import (
    IntrusiveNonlinearROM, impulse_to_reduced_velocity,
    pod_rm_analysis, PodRMAnalysis, MatrixSPDReport,
)
from .attractor import (
    stroboscopic_sample, steady_state_amplitude, midspan_transverse_probe,
    master_slave_data, dominant_frequency,
    SweepPoint, SweepResult, frequency_sweep, amplitude_sweep,
)
# Wave 13's torch-gated classes (ExcitationResponseOperator,
# LatentRefinementNet, ParameterizedODEFunc, ParameterizedLatentODE,
# wave13_validation.cross_check_operators_torch) genuinely subclass
# torch.nn.Module and so, unlike NeuralSurrogate above, cannot be
# DEFINED at all without torch importable -- deliberately NOT
# re-exported here (see neural_operator's own module docstring note);
# import them directly from their own module when torch is installed.

__all__ = [
    "PodBasis", "MultiFieldPOD", "assemble_field_weight_matrix", "trapezoidal_field_gram",
    "GalerkinROM", "AffineDecomposition",
    "FrequencyROM", "build_pod_basis_from_frf_snapshots",
    "greedy_train_frequency_basis",
    "psd_response", "band_limited_gaussian_time_history",
    "SingularValueLowerBound", "certified_error_bound",
    "LPSingularValueLowerBound", "certified_error_bound_lp",
    "modal_assurance_criterion", "r_squared",
    "LoewnerROM",
    "screen_physical_modes", "ScreenedMode",
    "optimal_lhs", "modal_force_samples",
    "KERNEL_REGISTRY", "ReducedForceModel", "MultiFidelitySurrogate", "PolynomialModalROM",
    "NeuralSurrogate",
    "TrainingStrategy", "AppliedLoadStrategy", "EnforcedDisplacementStrategy",
    "TrajectoryPilotedStrategy",
    "ICEROM", "ShiMeiROM", "EnforcedDisplacementROM",
    "integrate_newmark_surrogate",
    "HarmonicBalanceSystem", "solve_nnm_backbone", "solve_nnm_backbone_arclength",
    "to_state_space", "StateSpaceSystem",
    "mode_acceleration_correction", "mode_acceleration_response", "augmented_basis",
    "arnoldi_basis", "two_sided_arnoldi_bases", "KrylovROM",
    "soar_basis", "SOARROM",
    "controllability_gramian", "observability_gramian", "hankel_singular_values",
    "BalancedTruncationROM", "SingularPerturbationROM",
    "frequency_weighted_gramians", "frequency_weighted_hankel_singular_values",
    "FrequencyWeightedBalancedTruncationROM", "lowpass_weight", "bandpass_weight",
    "velocity_transfer_function", "passivity_margin", "is_passive",
    "OptimalHankelNormROM",
    "parameter_coverage_report", "pca_dimensionality", "GPSurrogate", "sobol_indices",
    "EnsembleUQ", "neural_surrogate_ensemble",
    "reduced_residual", "ScalarModalCorrection", "calibrate_reduced_correction_explicit",
    "MembraneBasis",
    "reduced_eom_residual_trajectory",
    "RegularizedProjectionEncoder",
    "rk4_step", "integrate_rk4", "CurriculumSchedule",
    "cross_check_reduced_dynamics", "cross_check_basis_generalization",
    "run_cross_configuration_validation",
    "IntrusiveNonlinearROM", "impulse_to_reduced_velocity",
    "pod_rm_analysis", "PodRMAnalysis", "MatrixSPDReport",
    "stroboscopic_sample", "steady_state_amplitude", "midspan_transverse_probe",
    "master_slave_data", "dominant_frequency",
    "SweepPoint", "SweepResult", "frequency_sweep", "amplitude_sweep",
]

__version__ = "0.1.0"
