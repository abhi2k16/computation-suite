"""
nonlinear_solver.py -- Module 8: incremental Newton-Raphson drivers for
geometric nonlinearity (large displacement / large rotation).

This module knows NOTHING about trusses, beams, or continua -- it only
calls fesystem.assemble_internal_force(u, mat) and
fesystem.assemble_tangent_stiffness(u, mat), which in turn only call
element.internal_force()/element.tangent_stiffness() (see element.py's
"Nonlinear extension point" docstring). So adding a second nonlinear
element later (a large-rotation beam, a Total-Lagrangian Quad4 for the
large-displacement/large-strain case) means this file needs ZERO
changes -- the same additive property assemble_stiffness()/solve_static()
already have for linear elements.

Two drivers are provided, because a single "ramp the load" strategy
cannot trace every equilibrium path:

solve_nonlinear_static()
    Load control: increment the external force F_ext = lambda * F,
    lambda: 0 -> 1, Newton-Raphson to equilibrium at each step. Fails
    (Newton won't converge, K_T becomes singular) at or past a LIMIT
    POINT (dP/ddelta = 0) -- fine for monotonic problems (e.g. a
    large-deflection cantilever), wrong tool for snap-through.

solve_nonlinear_displacement_control()
    Prescribe the displacement at one DOF directly (instead of the
    load), Newton-Raphson the reaction there and equilibrium
    everywhere else. Since delta itself is now the control parameter
    (not derived from equilibrium), this sails straight through a
    limit point / snap-through with no special continuation algorithm
    needed -- the price is that it only works when a single DOF is a
    valid path parameter for the whole structure (true for the
    classic von Mises truss benchmark used to validate this module;
    NOT a substitute for general arc-length/Riks continuation on an
    arbitrary structure -- see solve_nonlinear_arc_length() below).

solve_nonlinear_arc_length()
    Module 19 (general-purpose extensions roadmap Phase 5): Crisfield's
    cylindrical arc-length method. Neither the load NOR one displacement
    component is prescribed directly -- instead BOTH the displacement
    increment Delta_u and the load-factor increment Delta_lambda for a
    step are solved for TOGETHER, constrained to a fixed "arc length"
    radius ||Delta_u|| = delta_L. This traces the full equilibrium path
    through limit points AND snap-back (where even the loaded DOF's own
    displacement reverses direction, the one case displacement control
    cannot handle -- see that driver's docstring) without needing to
    know in advance where any of that happens, at the cost of one new
    piece of math per Newton iteration: with the linearized update
    du = du_r + dlambda*du_t (du_r: correction from the current residual,
    du_t: response to a unit load, both K_T^-1 solves against the SAME
    tangent), the arc-length constraint becomes a QUADRATIC equation in
    dlambda -- solve it, then pick whichever of its two roots keeps the
    new increment most aligned with the step's own predictor direction
    (the standard root-selection criterion; the other root would double
    back on the path just traced).

solve_nonlinear_koiter_newton()
    Module 21: single-branch Koiter-Newton predictor/corrector
    continuation (Liang & Sun, "A reduced-order modeling technique for
    nonlinear buckling analysis," ICCM2017; Liang, Abdalla & Gurdal,
    "A Koiter-Newton approach for nonlinear structural analysis," IJNME
    2013). Same primary-path equilibrium problem as
    solve_nonlinear_arc_length() (lambda AND u solved for together,
    every step), but replaces that driver's LINEAR tangent predictor
    with a CUBIC one, built from a genuine Koiter asymptotic expansion
    of the equilibrium path at the current point -- see that function's
    own docstring for the full derivation and the ONE scope reduction
    versus the published method (m=0: no separate closely-spaced-
    buckling-mode perturbation directions, primary path only).

solve_nonlinear_static_koiter_newton()
    Module 22: Koiter-Newton solve to a SINGLE prescribed target load
    (F_ext = 1.0*fesystem.F), NOT a path tracer -- the actual usage
    pattern Yang et al. 2019's Fig. 1 flowchart and Section 4.1 describe
    for generating nonlinear-static ROM-training data (one static test
    per prescribed modal-load combination), as opposed to
    solve_nonlinear_koiter_newton()'s open-ended continuation. Reuses
    the SAME cubic asymptotic predictor machinery, but aims each
    expansion's perturbation parameter DIRECTLY at the remaining gap to
    the target load (the real root of the reduced cubic polynomial
    closest to the linear estimate), shrinking only if that lands
    somewhere the model doesn't trust -- it NEVER grows past the
    target, unlike the continuation driver's open-ended growth (which,
    tried here first, was found to overshoot arbitrarily -- up to 14x
    the target load in one test -- and could drive the full model into
    a floating-point overflow, since nothing bounds a free-running
    predictor's growth toward a caller's actual target). Because the
    target load is externally KNOWN (not a free unknown the way
    continuation's lambda is), the corrector needs no bordering trick --
    just ordinary frozen-tangent Newton at the fixed target load.

solve_nonlinear_koiter_newton_generic()
    Module 24: the GENERIC (m>=1) Koiter-Newton continuation driver --
    lifts solve_nonlinear_koiter_newton()'s disclosed m=0 scope
    reduction (see that function's own "SCOPE" section) by adding
    automatic detection of a near-critical tangent-stiffness mode at
    each step (a relative smallest-eigenvalue check on K_T) and, when
    one is found, a SECOND perturbation direction built from that
    mode -- the actual multi-direction reduced basis the published
    method's own m>=1 case describes, including the genuine
    mode-interaction/branch-selection behavior (the reduced
    equilibrium equation for the extra direction becomes a cubic
    polynomial with up to 3 real roots, not a single predicted value).
    A brand-new function, not a modification of
    solve_nonlinear_koiter_newton() above -- that driver is unchanged
    and remains the one NonLin-HyROM's existing training pipeline
    depends on; see this function's own docstring for the full
    derivation, and docs/general_purpose_extensions_roadmap.md Section
    11 for the correctness-review finding (no bugs in the m=0 driver,
    only its disclosed scope was incomplete) and the fea_engine-vs-
    rom_engine placement reasoning.

solve_nonlinear_transient()
    Newmark-beta implicit TIME integration for a geometrically (or
    materially) nonlinear structure -- the dynamic counterpart to the
    four STATIC drivers above. `FESystem.solve_transient_implicit()`
    factors one effective stiffness ONCE, before the time loop, and
    reuses it unchanged at every step: correct for a linear,
    time-invariant system, silently wrong the moment the internal
    force/tangent should depend on the current displacement (which for
    ANY of this module's other four drivers, or any nonlinear element,
    it does). This driver instead runs Newton-Raphson to convergence
    EVERY step, using the current assemble_internal_force()/
    assemble_tangent_stiffness() exactly as the static drivers above
    do -- the residual is the standard Newmark-implicit dynamic
    residual R(d) = M @ a(d) + C @ v(d) + F_int(d) - F_ext(t+dt), with
    a(d)/v(d) affine in the unknown trial displacement d via the same
    Newmark predictor constants FESystem.solve_transient_implicit()
    already uses, so K_eff(d) = K_T(d) + a0c*M + a1c*C is the exact
    Newton tangent, not an approximation. See docs/
    nonlinear_transient_dynamics_roadmap.md for the full design
    rationale and validation plan this implements.

solve_transient_explicit_nonlinear()
    Wave 6 item 31 (docs/consolidated_future_roadmap.md): the EXPLICIT
    counterpart to solve_nonlinear_transient() above -- same idea
    (generalize a LINEAR FESystem transient driver by substituting the
    real nonlinear internal-force vector for the linear Kff@d term),
    applied to FESystem.solve_transient_explicit()'s central-difference
    recursion instead of solve_transient_implicit()'s Newmark-average-
    acceleration one. The two generalizations differ in one important
    way: central difference already has d_new in closed form (it never
    appears inside the unknown-implicit F_int(d_new) term the way
    Newmark's a(d)/v(d) do), so NO Newton loop, and hence no tangent
    stiffness assembly at all, is needed here -- this driver's entire
    per-step cost is one assemble_internal_force() call plus either an
    elementwise divide (diagonal/no damping, the common case) or one
    CONSTANT-matrix linear solve (general C), mirroring FESystem.
    solve_transient_explicit()'s own diag_fast split exactly. See that
    method's docstring for the underlying difference-equation algebra;
    see this function's own docstring below for why it lives in this
    module (not as a FESystem method) despite needing no Newton loop.

solve_transient_displacement_control()
    Wave 6 item 34 (docs/consolidated_future_roadmap.md: "Arc-length /
    displacement-control transient variant (dynamic snap-through)"):
    the dynamic (Newmark-implicit) counterpart to solve_nonlinear_
    displacement_control() above, generalized the same way solve_
    nonlinear_transient() generalized solve_nonlinear_static() -- one
    DOF's motion is prescribed directly (via a plain t -> float
    callable, not a load object) instead of solved for, every other
    free dof is solved via the SAME trust-region/line-search Newton
    ladder every implicit driver in this module uses, and the actuator
    force needed to hold the prescribed motion is recorded as a
    reaction at every step. See this function's own docstring for why
    displacement control (not literal arc-length) is what this item
    actually builds: a transient problem's Newmark effective stiffness
    carries a mass term that stays positive definite straight through
    a static limit point, so the singular-Jacobian failure mode that
    motivates arc-length continuation in the STATIC case does not
    transfer to the dynamic one the same way.

Wave 1 (docs/consolidated_future_roadmap.md, source nonlinear_fem_
lessons.md's appendix): two globalization/convergence upgrades, added
to solve_nonlinear_static(), solve_nonlinear_displacement_control(),
solve_nonlinear_arc_length(), and solve_nonlinear_static_koiter_
newton()'s frozen-tangent corrector -- see _extra_convergence_ok() and
_armijo_line_search_step() just below for the two shared helpers every
one of those drivers now calls, so the algorithm is written ONCE and
reused, not re-derived per driver. Both upgrades are OFF (du_tol/
energy_tol=None) or a FALLBACK-ONLY retry (line_search=True, tried
only after plain Newton exhausts max_iter) by default, so every
existing caller of every driver above gets IDENTICAL behavior, cost,
and results to before this wave -- this is the same "add it as an
explicit, opt-in choice, never silently replace the existing path"
principle the backend= parameter on FESystem (solver.py) established
for the SciPy-vs-PyTorch solve choice.

  Item 9 -- displacement-increment / energy-error convergence
  criteria: the force-residual-norm check above was, until this wave,
  the ONLY convergence criterion anywhere in this module -- adequate
  for the smooth (linear-tangent, e.g. TrussTL2D) elements this
  package's tests mostly exercise, but nonlinear_fem_lessons.md's own
  Sec.6.3.9 warns a residual-only check can be satisfied prematurely
  for non-smooth constitutive laws (plasticity yielding, contact) where
  the Jacobian itself loses regularity. `du_tol`/`energy_tol` add the
  book's other two named criteria as OPTIONAL, ADDITIONAL (AND-
  combined, never a replacement for) requirements -- see
  _extra_convergence_ok() below for the exact formulas.

  Item 10 -- line search in the static/arc-length drivers: Armijo
  backtracking already existed (2026-09-01) as a last-resort fallback
  inside solve_nonlinear_transient() (see that function's own
  extensive "Line search" docstring section for the full derivation
  and the real Tet10SolidTL wing-cantilever failure that motivated it)
  but was absent from every static/arc-length driver. _armijo_line_
  search_step() below is that SAME algorithm, factored out into a
  reusable, mode-agnostic helper (residual_fn callback instead of a
  hardcoded Newmark residual) so each driver's own Newton loop can
  retry through it without re-deriving or duplicating the logic.

  SCOPE NOTE on the Koiter-Newton family specifically: item 10 was
  added to solve_nonlinear_static_koiter_newton()'s corrector because
  that corrector is, by its own docstring, "ORDINARY frozen-tangent
  (chord) Newton" -- structurally identical to this module's other
  plain-Newton loops, so the same helper applies directly. It was
  DELIBERATELY NOT added to solve_nonlinear_koiter_newton()'s or
  solve_nonlinear_koiter_newton_generic()'s continuation correctors:
  those solve a BORDERED system for (du, dlambda) TOGETHER against a
  frozen tangent already reused, unmodified, across every iteration
  and retry of a step (see solve_nonlinear_koiter_newton()'s own
  "CORRECTOR" docstring section) -- a real, previously-debugged sign-
  error war story lives in that exact code path, and it already has
  its own, different globalization (shrinking the predictor's
  perturbation parameter `a`, not backtracking a Newton step). Bolting
  a second, structurally different globalization mechanism onto
  already-delicate, already-validated math was judged a materially
  riskier change than what item 10 actually asks for, so it is left as
  documented future work rather than attempted here -- item 9's
  convergence-criteria addition, being purely additive (an extra CHECK,
  no change to the update math itself), was judged safe and IS applied
  to those two drivers as well.
"""
__author__ = "Abhijeet"
import copy
import warnings
import numpy as np
from scipy.linalg import lu_factor, lu_solve, eigh


def _extra_convergence_ok(R_free, du_free, u_free, F_ext_free, F_int_free,
                           du_tol, energy_tol):
    """Wave 1 item 9 (nonlinear_fem_lessons.md Sec.6.3.9, Belytschko &
    Schoeberle 1975): the displacement-increment and energy-error
    convergence criteria, evaluated ONLY for whichever of du_tol/
    energy_tol is not None (None means "this criterion is off" -- the
    caller's own force-residual check stays the primary/default
    criterion regardless; these are ADDITIONAL, AND-combined
    requirements on top of it, never a substitute for it).

    R_free: the residual that PRODUCED du_free (i.e. K_T @ du_free =
    R_free -- both evaluated at the SAME trial point, u_free, before
    this correction is applied). F_ext_free/F_int_free: the external/
    internal force at that same point -- used as the two "work" terms
    nonlinear_fem_lessons.md's energy criterion normalizes against
    (its general max(W_ext, W_int, W_kin) with W_kin=0, since none of
    this module's static drivers carry an inertial term).

    Displacement-increment: ||du|| / max(||u||, floor) <= du_tol.
    Energy-error:            |du . R| <= energy_tol * max(|F_ext.u|, |F_int.u|, floor).

    Returns (du_ok, energy_ok), each True when its own criterion is
    off (nothing to fail) or satisfied."""
    du_ok = True
    if du_tol is not None:
        u_norm = max(np.linalg.norm(u_free), 1e-30)
        du_ok = np.linalg.norm(du_free) / u_norm <= du_tol
    energy_ok = True
    if energy_tol is not None:
        w_ext = abs(F_ext_free @ u_free)
        w_int = abs(F_int_free @ u_free)
        ref_w = max(w_ext, w_int, 1e-30)
        energy_ok = abs(du_free @ R_free) <= energy_tol * ref_w
    return du_ok, energy_ok


def _armijo_line_search_step(residual_fn, u_free, du_newton, Rn, tol, ref,
                              c1=1e-4, max_backtrack=19, verbose=False,
                              verbose_prefix=""):
    """Wave 1 item 10: ONE Armijo-backtracked Newton correction from
    u_free along the (already-computed) Newton direction du_newton --
    the exact algorithm solve_nonlinear_transient() validated as its
    own 'line_search' mode (see that function's docstring for the full
    derivation/history), factored out here so every OTHER driver in
    this module reuses it instead of re-deriving it.

    du_newton is guaranteed a descent direction for the merit function
    m(alpha) = ||R(u_free + alpha*du_newton)||^2 at alpha=0 whenever
    the Jacobian used to build it is a reasonable one (m'(0) =
    2 R^T J du = -2||R||^2 < 0 when J@du_newton = -R by construction --
    true whether J is a FRESH tangent (this module's incremental
    static/arc-length drivers) or a FROZEN one (a chord/modified-Newton
    corrector, e.g. solve_nonlinear_static_koiter_newton()'s), since
    that identity only needs J@du_newton=-R, not J itself being exact).

    residual_fn(u_trial_free) -> (R_trial_free, aux) must, as a SIDE
    EFFECT, leave whatever state the caller's own next K_T/F_int
    computation depends on updated at u_trial_free (mirroring
    solve_nonlinear_transient's own _residual() closure) -- `aux` is
    any extra value the caller wants carried through unchanged from
    whichever trial point is finally accepted (e.g. F_int, for
    _extra_convergence_ok()'s next call), so this helper never needs
    to know what the caller wants beyond R itself.

    Accepts the first alpha in {1, 1/2, 1/4, ...} (halved up to
    max_backtrack times) satisfying Armijo sufficient decrease
    m(alpha) <= (1 - 2*c1*alpha)*m0, or immediately accepts a trial
    that already satisfies the caller's OWN outer convergence
    tolerance at any alpha (no reason to keep hunting for "more
    decrease" past that point). Falls back to the FULL step (alpha=1)
    if no alpha satisfies genuine sufficient decrease -- NOT the
    smallest-alpha trial or whichever trial had the lowest residual;
    solve_nonlinear_transient's own docstring documents why those two
    simpler-looking alternatives are real bugs, not just style choices
    (a tight/near-machine-precision tol can make every alpha look
    equally flat, and a strict-decrease-only rule then spuriously
    freezes the iterate at a near-zero alpha).

    Returns (u_new_free, R_new_free, aux_new, Rn_new, alpha_used)."""
    m0 = Rn ** 2
    u_full = u_free + du_newton
    R_full, aux_full = residual_fn(u_full)
    Rn_full = np.linalg.norm(R_full)
    already_good = Rn_full < tol * ref or Rn_full < tol
    sufficient_decrease = Rn_full ** 2 <= (1.0 - 2.0 * c1) * m0
    if already_good or sufficient_decrease:
        return u_full, R_full, aux_full, Rn_full, 1.0

    alpha = 1.0
    u_cand, R_cand, aux_cand, accepted = u_full, R_full, aux_full, False
    for _ in range(max_backtrack):
        alpha *= 0.5
        u_try = u_free + alpha * du_newton
        R_try, aux_try = residual_fn(u_try)
        Rn_try = np.linalg.norm(R_try)
        if Rn_try < tol * ref or Rn_try < tol or \
                Rn_try ** 2 <= (1.0 - 2.0 * c1 * alpha) * m0:
            u_cand, R_cand, aux_cand, accepted = u_try, R_try, aux_try, True
            break
    if not accepted:
        u_cand, R_cand, aux_cand, alpha = u_full, R_full, aux_full, 1.0
    elif verbose:
        print(f"{verbose_prefix}line search backtracked to alpha={alpha:.3e}")
    return u_cand, R_cand, aux_cand, np.linalg.norm(R_cand), alpha


def solve_contact_lagrange_static(fesystem, mat, contact_node, n_hat, g0,
                                   n_steps=10, tol=1e-10, max_iter=50,
                                   verbose=False, load_factors=None, **kwargs):
    """Module 10, Lagrange-multiplier contact: EXACT (not approximate)
    enforcement of zero penetration at a single contact node, via one
    extra scalar unknown lambda (the contact reaction magnitude) --
    the "extra equation in the global matrix" that distinguishes this
    from the penalty method (element.GapContactPenalty), which instead
    accepts a small, k_p-dependent penetration in exchange for staying
    inside the ordinary n_dof x n_dof system.

    `fesystem` should be a plain structural system with NO contact
    elements registered via add_contact_element() -- this function
    manages the single contact constraint itself, on top of
    fesystem.assemble_internal_force()/assemble_tangent_stiffness()
    (which, with no contact elements added, are exactly the
    structural-only internal force/tangent).

    Two-phase ACTIVE-SET strategy at each load step (standard for a
    single, monotonically-engaging unilateral constraint -- see the
    module docstring below for what this does NOT generalize to):
      1. Solve the UNCONSTRAINED structural problem (ignore contact).
         If the resulting gap is still >= 0 (not penetrating), that IS
         the solution for this step -- contact is inactive, lambda=0.
      2. If the unconstrained trial WOULD penetrate, re-solve with the
         contact constraint ACTIVE: an augmented KKT Newton-Raphson on
         [u_free; lambda] enforcing gap(u) = 0 exactly:

             [ K_struct   n_hat ] [du]   [F_ext - F_int(u) - lambda*n_hat]
             [ n_hat^T      0   ] [dλ] = [        g0 - n_hat . u_contact  ]

         (derived from stationarity of Pi(u,lambda) = Pi_struct(u) -
         lambda*(n_hat.u_contact - g0); lambda comes out with the sign
         of a physical compressive contact force for a monotonically
         increasing load into the obstacle).

    Returns (load_factors, U_hist, lambda_hist, active_hist).

    Scope: ONE contact constraint, monotonic engagement (contact turns
    on at most once per call, never off again mid-run). A structure
    with multiple simultaneous contacts, or contact that can both
    engage AND release within one analysis, needs a general multi-
    constraint active-set (or semismooth-Newton) solver -- a natural
    extension of the same augmented-KKT idea, not implemented here."""
    free = list(fesystem.free_dofs)
    F_total = fesystem.F.copy()
    n_dof = fesystem.n_dof
    npn = fesystem.npn
    n_hat = np.asarray(n_hat, dtype=float)

    contact_dofs = [npn * contact_node + k for k in range(npn)]
    n_hat_free = np.zeros(len(free))
    for k, d in enumerate(contact_dofs):
        if d in free:
            n_hat_free[free.index(d)] = n_hat[k]

    if load_factors is None:
        load_factors = np.linspace(0.0, 1.0, n_steps + 1)
    else:
        load_factors = np.asarray(load_factors, dtype=float)

    u = np.zeros(n_dof)
    U_hist = np.zeros((len(load_factors), n_dof))
    lambda_hist = np.zeros(len(load_factors))
    active_hist = np.zeros(len(load_factors), dtype=bool)

    def gap_of(u_vec):
        return n_hat @ u_vec[contact_dofs] - g0

    start = 0
    if load_factors[0] == 0.0:
        U_hist[0] = u
        fesystem.commit_all_states(u, mat, **kwargs)
        start = 1

    for step in range(start, len(load_factors)):
        F_ext = load_factors[step] * F_total
        ref = max(np.linalg.norm(F_ext[free]), 1e-30)

        # --- Phase 1: unconstrained structural trial ---
        u_trial = u.copy()
        converged = False
        for it in range(max_iter):
            F_int = fesystem.assemble_internal_force(u_trial, mat, **kwargs)
            R = F_ext - F_int
            Rn = np.linalg.norm(R[free])
            if Rn < tol * ref or Rn < tol:
                converged = True
                break
            K_T = fesystem.assemble_tangent_stiffness(u_trial, mat, **kwargs)
            du = np.linalg.solve(K_T[np.ix_(free, free)], R[free])
            u_trial[free] += du
        if not converged:
            raise RuntimeError(
                f"solve_contact_lagrange_static: unconstrained Phase-1 solve "
                f"failed to converge at step {step}.")

        if gap_of(u_trial) <= 0.0:
            u, lam, active = u_trial, 0.0, False
        else:
            # --- Phase 2: augmented KKT solve, constraint active ---
            u_c = u_trial.copy()
            lam_c = 0.0
            nf = len(free)
            converged2 = False
            for it in range(max_iter):
                F_int = fesystem.assemble_internal_force(u_c, mat, **kwargs)
                R_u = F_ext[free] - F_int[free] - lam_c * n_hat_free
                R_lam = -gap_of(u_c)
                Rn = np.sqrt(np.linalg.norm(R_u) ** 2 + R_lam ** 2)
                if verbose:
                    print(f"  step {step:3d} [active] it {it:2d}  |R|={Rn:.3e}")
                if Rn < tol * max(ref, 1.0):
                    converged2 = True
                    break
                K_T = fesystem.assemble_tangent_stiffness(u_c, mat, **kwargs)
                K_aug = np.zeros((nf + 1, nf + 1))
                K_aug[:nf, :nf] = K_T[np.ix_(free, free)]
                K_aug[:nf, nf] = n_hat_free
                K_aug[nf, :nf] = n_hat_free
                R_aug = np.concatenate([R_u, [R_lam]])
                dx = np.linalg.solve(K_aug, R_aug)
                u_c[free] += dx[:nf]
                lam_c += dx[nf]
            if not converged2:
                raise RuntimeError(
                    f"solve_contact_lagrange_static: constrained Phase-2 solve "
                    f"failed to converge at step {step}.")
            u, lam, active = u_c, lam_c, True

        U_hist[step] = u
        lambda_hist[step] = lam
        active_hist[step] = active
        fesystem.commit_all_states(u, mat, **kwargs)

    return load_factors, U_hist, lambda_hist, active_hist


def solve_contact_augmented_lagrange_static(fesystem, mat, contact_node, n_hat, g0, k_p,
                                             n_steps=10, tol=1e-10, max_iter=50,
                                             al_tol=1e-9, al_max_iter=30,
                                             verbose=False, load_factors=None, **kwargs):
    """Wave 3 item 15 (docs/consolidated_future_roadmap.md): augmented-
    Lagrangian contact enforcement -- the third contact-enforcement
    option this module now offers, sitting alongside (never replacing)
    the pure penalty method (element.GapContactPenalty, added via
    fesystem.add_contact_element()) and the exact bordered-KKT pure
    Lagrange-multiplier method (solve_contact_lagrange_static() above).
    Same single-constraint, monotonic-engagement SCOPE as
    solve_contact_lagrange_static() -- see that function's own
    docstring's "Scope" paragraph, which applies here unchanged -- this
    is a genuinely different ENFORCEMENT mechanism for the identical
    problem class, not a generalization of it.

    THE IDEA (Wriggers, "Computational Contact Mechanics", 2nd ed.,
    Ch. 4; Simo & Laursen 1992): augment the penalty method with an
    explicit multiplier estimate lambda_bar, updated between outer
    (Uzawa) iterations, so the FINAL converged penetration is driven to
    (numerically) exactly zero regardless of how large k_p is -- unlike
    plain penalty, whose converged penetration is ALWAYS k_p-dependent
    and nonzero (see solve_contact_lagrange_static()'s own CHECK 5 in
    tests/test_contact.py: penetration shrinks but never vanishes as
    k_p grows), and unlike the exact bordered-KKT Lagrange method above,
    without needing to build/factor an augmented (n_free+1)x(n_free+1)
    system every Newton iteration.

    Derivation, worked from the augmented Lagrangian functional
    L_c(u, lambda_bar) = Pi_struct(u) + [(lambda_bar*delta +
    0.5*k_p*delta^2) if (lambda_bar+k_p*delta)>0 else -0.5*lambda_bar^2/k_p],
    delta(u) = n_hat.u_contact - g0 (identical sign convention to
    element.GapContactPenalty: delta>0 means penetrating). Taking
    dL_c/du gives a contact force of exactly

        f_c = max(lambda_bar + k_p*delta(u), 0) * n_hat

    -- i.e. an ordinary PENALTY contact force, just evaluated against a
    SHIFTED effective gap g0_eff = g0 - lambda_bar/k_p instead of the
    true g0 (since lambda_bar + k_p*delta = k_p*(n_hat.u - g0_eff), a
    two-line algebraic rearrangement). This is why the inner solve
    below is implemented as literally GapContactPenalty's own force/
    tangent formula (not re-derived, not imported either -- inlined
    directly, matching the sibling solve_contact_lagrange_static()'s
    own convention of not importing a specific Element class into this
    otherwise element-agnostic module) evaluated at g0_eff: reusing an
    already-validated closed form as a black box, rather than writing
    new contact-force math, is the deliberate choice here, the same
    "minimize new/untested math" principle used throughout this
    project's Wave 2 additions.

    ALGORITHM per load step: an outer Uzawa loop --
      1. With lambda_bar fixed, Newton-solve the structural problem to
         equilibrium with the (fixed-g0_eff) penalty contact force
         above included -- an ordinary, well-conditioned Newton solve
         (k_p need NOT be huge, since lambda_bar -- not k_p alone --
         does the exact-enforcement work).
      2. Update lambda_bar <- max(lambda_bar + k_p*delta(u), 0) (the
         standard Uzawa/projected-gradient multiplier update for a
         non-negative contact pressure).
      3. Repeat until lambda_bar stops changing (al_tol, relative to
         k_p*max(gap scale, 1) -- see the convergence check below) or
         al_max_iter is exhausted.
    lambda_bar is WARM-STARTED across load steps (not reset to 0 each
    step) since the converged multiplier from one step is normally a
    good starting guess for the next, small load increment.

    Returns (load_factors, U_hist, lambda_hist, active_hist) -- the
    IDENTICAL return signature as solve_contact_lagrange_static(), so
    the two are directly interchangeable in a caller/test and their
    lambda_hist/active_hist can be compared entry-for-entry. Validated
    in tests/test_contact.py against exactly that comparison: the same
    problem run through both drivers should agree on lambda_hist to
    tight tolerance and on active_hist exactly, while this driver's own
    penetration stays near machine-zero using a k_p MANY ORDERS OF
    MAGNITUDE smaller than plain penalty needs for comparable accuracy
    (see solve_contact_lagrange_static()'s CHECK 5 for the penalty
    baseline this is compared against) -- the concrete "better
    conditioning than pure penalty" payoff this item's roadmap entry
    names."""
    free = list(fesystem.free_dofs)
    F_total = fesystem.F.copy()
    n_dof = fesystem.n_dof
    npn = fesystem.npn
    n_hat = np.asarray(n_hat, dtype=float)

    contact_dofs = [npn * contact_node + k for k in range(npn)]

    if load_factors is None:
        load_factors = np.linspace(0.0, 1.0, n_steps + 1)
    else:
        load_factors = np.asarray(load_factors, dtype=float)

    u = np.zeros(n_dof)
    U_hist = np.zeros((len(load_factors), n_dof))
    lambda_hist = np.zeros(len(load_factors))
    active_hist = np.zeros(len(load_factors), dtype=bool)

    def gap_of(u_vec):
        return n_hat @ u_vec[contact_dofs] - g0

    def contact_force_and_tangent(u_vec, g0_eff):
        delta = n_hat @ u_vec[contact_dofs] - g0_eff
        if delta <= 0.0:
            return np.zeros(npn), np.zeros((npn, npn))
        return (k_p * delta) * n_hat, k_p * np.outer(n_hat, n_hat)

    lambda_bar = 0.0
    start = 0
    if load_factors[0] == 0.0:
        U_hist[0] = u
        fesystem.commit_all_states(u, mat, **kwargs)
        start = 1

    for step in range(start, len(load_factors)):
        F_ext = load_factors[step] * F_total
        ref = max(np.linalg.norm(F_ext[free]), 1e-30)

        for al_it in range(al_max_iter):
            g0_eff = g0 - lambda_bar / k_p

            # --- inner Newton: ordinary penalty solve at fixed g0_eff ---
            converged = False
            for it in range(max_iter):
                F_int = fesystem.assemble_internal_force(u, mat, **kwargs)
                f_c, _ = contact_force_and_tangent(u, g0_eff)
                F_int[contact_dofs] += f_c
                R = F_ext - F_int
                Rn = np.linalg.norm(R[free])
                if Rn < tol * ref or Rn < tol:
                    converged = True
                    break
                K_T = fesystem.assemble_tangent_stiffness(u, mat, **kwargs)
                _, k_c = contact_force_and_tangent(u, g0_eff)
                K_T[np.ix_(contact_dofs, contact_dofs)] += k_c
                du = np.linalg.solve(K_T[np.ix_(free, free)], R[free])
                u[free] += du
            if not converged:
                raise RuntimeError(
                    f"solve_contact_augmented_lagrange_static: inner Newton "
                    f"solve failed to converge at step {step}, AL outer "
                    f"iteration {al_it}.")

            # --- outer Uzawa multiplier update, against the TRUE gap ---
            delta_true = gap_of(u)
            lambda_new = max(lambda_bar + k_p * delta_true, 0.0)
            al_scale = max(k_p * max(abs(g0), 1.0), 1.0)
            if verbose:
                print(f"  step {step:3d}  AL it {al_it:2d}  lambda={lambda_new:.6e}  "
                      f"delta={delta_true:+.3e}")
            if abs(lambda_new - lambda_bar) < al_tol * al_scale:
                lambda_bar = lambda_new
                break
            lambda_bar = lambda_new
        else:
            raise RuntimeError(
                f"solve_contact_augmented_lagrange_static: outer Uzawa loop "
                f"did not converge within {al_max_iter} iterations at step "
                f"{step}.")

        U_hist[step] = u
        lambda_hist[step] = lambda_bar
        active_hist[step] = lambda_bar > 0.0
        fesystem.commit_all_states(u, mat, **kwargs)

    return load_factors, U_hist, lambda_hist, active_hist


def solve_nonlinear_static(fesystem, mat, n_steps=10, tol=1e-8, max_iter=30,
                            verbose=False, load_factors=None,
                            du_tol=None, energy_tol=None, line_search=True,
                            **kwargs):
    """Load-controlled incremental Newton-Raphson. F_ext at step i is
    load_factors[i] * fesystem.F (whatever add_nodal_force()/etc.
    already built).

    load_factors: if None (default), uses the original monotonic
    np.linspace(0, 1, n_steps+1) -- a plain "ramp the load up once"
    analysis, and n_steps controls its resolution. Pass your OWN array
    instead for anything else: a non-monotonic load HISTORY (ramp up,
    unload, reload -- needed to see plasticity's permanent set/
    hysteresis, since that's only visible across multiple load
    reversals, not a single monotonic ramp), a finer/coarser step size
    near a nonlinearity, etc. n_steps is then ignored; U_hist has one
    row per entry of your load_factors, in the same order, and NO
    implicit zero-state row is prepended -- include your own starting
    lambda (usually 0.0) as load_factors[0] if you want it recorded.

    mat is forwarded to assemble_internal_force()/assemble_tangent_stiffness()
    unchanged (e.g. (E, A) for TrussTL2D, or (PlasticMaterial1D(...), A)
    for TrussPlastic2D). After EVERY converged step, calls
    fesystem.commit_all_states(u, mat, **kwargs) -- a no-op unless
    fesystem.init_state() was called first, so this is transparent to
    every element from before Module 9. After EVERY Newton CORRECTION
    (not just converged steps), also calls fesystem.update_iter_states()
    with the REALIZED correction (Module 23, mixed-formulation internal
    state, e.g. Shell4MITCCorotational's von Karman coupling stress) --
    likewise a no-op unless fesystem.init_iter_state() was called first.

    du_tol/energy_tol/line_search (Wave 1 items 9/10 -- see this
    module's own docstring, "Wave 1" section, for the shared helpers
    both of these call and the full design rationale). All three
    default to their original-behavior values (du_tol=energy_tol=None,
    line_search=True but never invoked unless plain Newton actually
    fails) -- an existing caller passing none of these gets IDENTICAL
    results to before this wave, at identical cost."""
    free = fesystem.free_dofs
    F_total = fesystem.F.copy()
    n_dof = fesystem.n_dof

    if load_factors is None:
        load_factors = np.linspace(0.0, 1.0, n_steps + 1)
    else:
        load_factors = np.asarray(load_factors, dtype=float)

    u = np.zeros(n_dof)
    U_hist = np.zeros((len(load_factors), n_dof))

    start = 0
    if load_factors[0] == 0.0:
        # zero load -> zero displacement trivially, no Newton needed;
        # still commit (harmless no-op unless this system has state).
        U_hist[0] = u
        fesystem.commit_all_states(u, mat, **kwargs)
        start = 1

    for step in range(start, len(load_factors)):
        lam = load_factors[step]
        F_ext = lam * F_total
        ref = max(np.linalg.norm(F_ext[free]), 1e-30)
        u_step_start = u.copy()
        iter_state_at_step_start = copy.deepcopy(fesystem.iter_state)

        def _residual(u_trial_free):
            u[free] = u_trial_free
            F_int_trial = fesystem.assemble_internal_force(u, mat, **kwargs)
            return F_ext[free] - F_int_trial[free], F_int_trial

        def _attempt(mode):
            # Every mode retries from the step's own start state, not
            # wherever a previous (failed) attempt left off -- mirrors
            # solve_nonlinear_transient's own _newton_attempt(mode), and
            # the iter_state snapshot/restore below (needed because
            # update_iter_states() below may have been called several
            # times by a failed prior attempt) mirrors solve_nonlinear_
            # static_koiter_newton's own per-attempt snapshot/restore.
            fesystem.iter_state = copy.deepcopy(iter_state_at_step_start)
            u_free = u_step_start[free].copy()
            R_free, F_int = _residual(u_free)
            Rn = np.linalg.norm(R_free)
            converged = False
            for it in range(max_iter):
                if verbose:
                    tag = " [line search]" if mode == "line_search" else ""
                    print(f"  step {step:3d} it {it:2d}{tag}  |R|={Rn:.3e}")
                residual_ok = Rn < tol * ref or Rn < tol
                if residual_ok and du_tol is None and energy_tol is None:
                    converged = True
                    break
                K_T = fesystem.assemble_tangent_stiffness(u, mat, **kwargs)
                du = np.linalg.solve(K_T[np.ix_(free, free)], R_free)
                if residual_ok:
                    du_ok, energy_ok = _extra_convergence_ok(
                        R_free, du, u_free, F_ext[free], F_int[free],
                        du_tol, energy_tol)
                    if du_ok and energy_ok:
                        converged = True
                        break

                if mode == "line_search":
                    u_new, R_new, F_int_new, Rn_new, _alpha = _armijo_line_search_step(
                        _residual, u_free, du, Rn, tol, ref, verbose=verbose,
                        verbose_prefix=f"    step {step:3d} it {it:2d}  ")
                else:
                    u_new = u_free + du
                    R_new, F_int_new = _residual(u_new)
                    Rn_new = np.linalg.norm(R_new)

                delta_u = np.zeros(n_dof)
                delta_u[free] = u_new - u_free
                fesystem.update_iter_states(u, delta_u, mat, **kwargs)
                u_free, R_free, F_int, Rn = u_new, R_new, F_int_new, Rn_new
            return converged, u_free, Rn

        converged, u_free, Rn = _attempt("plain")
        if not converged and line_search:
            if verbose:
                print(f"  step {step:3d}: plain Newton did not converge in "
                      f"{max_iter} iterations (|R|={Rn:.3e}) -- retrying "
                      f"with line search")
            converged, u_free, Rn = _attempt("line_search")

        if not converged:
            raise RuntimeError(
                f"solve_nonlinear_static: Newton-Raphson failed to converge "
                f"at load step {step} (lambda={lam:.4f}), |R|={Rn:.3e} after "
                f"{max_iter} iterations -- likely at/past a limit point; try "
                f"solve_nonlinear_displacement_control() instead, or more steps.")
        u[free] = u_free
        U_hist[step] = u
        fesystem.commit_all_states(u, mat, **kwargs)

    return load_factors, U_hist


def solve_nonlinear_displacement_control(fesystem, mat, control_dof,
                                          u_target_array, tol=1e-10,
                                          max_iter=50, verbose=False,
                                          du_tol=None, energy_tol=None,
                                          line_search=True, **kwargs):
    """Prescribes u[control_dof] = u_target_array[i] at each step i
    (control_dof is treated as an extra support with a nonzero,
    changing prescribed value -- it must NOT already be in
    fesystem.fixed_dofs), Newton-Raphson-solves for every other free
    DOF, and records the reaction force needed to hold control_dof at
    that value: reaction = F_int[control_dof] - fesystem.F[control_dof]
    (zero if no other external load acts directly on control_dof, the
    usual case -- see nonlinear_solver module docstring / README for
    the potential-energy derivation dU/ddelta = P this reaction
    reduces to for the validated single-DOF truss benchmark).

    du_tol/energy_tol/line_search: Wave 1 items 9/10, identical
    meaning and IDENTICAL-cost-when-unused convention as
    solve_nonlinear_static()'s own parameters of the same name -- see
    this module's own docstring ("Wave 1" section) and
    _extra_convergence_ok()/_armijo_line_search_step() for the shared
    algorithm. This driver's own convergence check stays the ABSOLUTE
    `Rn < tol` it always used (no `ref`-relative form, unlike the other
    three drivers) -- du_tol/energy_tol/line_search do not change that.

    Returns (U_hist (n_steps, n_dof), reaction_hist (n_steps,)) -- no
    trivial zero-state prepended (u_target_array should start at its
    own first increment, e.g. include 0.0 explicitly if you want it)."""
    assert control_dof not in fesystem.fixed_dofs, \
        "control_dof must not already be a fixed support"
    free = np.array([d for d in fesystem.free_dofs if d != control_dof])
    n_dof = fesystem.n_dof
    n_steps = len(u_target_array)
    n_free = len(free)

    u = np.zeros(n_dof)
    U_hist = np.zeros((n_steps, n_dof))
    reaction_hist = np.zeros(n_steps)
    F_ext = fesystem.F   # fixed external load every step -- only
                          # control_dof's prescribed value changes

    for i, u_target in enumerate(u_target_array):
        u_step_start = u.copy()
        u_step_start[control_dof] = u_target
        iter_state_at_step_start = copy.deepcopy(fesystem.iter_state)

        def _residual(u_trial_free):
            u[free] = u_trial_free
            F_int_trial = fesystem.assemble_internal_force(u, mat, **kwargs)
            R_free = F_ext[free] - F_int_trial[free] if n_free else np.zeros(0)
            return R_free, F_int_trial

        def _attempt(mode):
            fesystem.iter_state = copy.deepcopy(iter_state_at_step_start)
            u[control_dof] = u_target
            u_free = u_step_start[free].copy() if n_free else np.zeros(0)
            R_free, F_int = _residual(u_free)
            Rn = np.linalg.norm(R_free) if n_free else 0.0
            converged = False
            for it in range(max_iter):
                if verbose:
                    tag = " [line search]" if mode == "line_search" else ""
                    print(f"  step {i:3d} it {it:2d}{tag}  |R|={Rn:.3e}")
                residual_ok = Rn < tol
                if residual_ok and du_tol is None and energy_tol is None:
                    converged = True
                    break
                if not n_free:
                    # nothing left to correct -- whatever residual_ok
                    # says (trivially True, Rn is always 0.0 here) is final.
                    converged = residual_ok
                    break
                K_T = fesystem.assemble_tangent_stiffness(u, mat, **kwargs)
                du = np.linalg.solve(K_T[np.ix_(free, free)], R_free)
                if residual_ok:
                    du_ok, energy_ok = _extra_convergence_ok(
                        R_free, du, u_free, F_ext[free], F_int[free],
                        du_tol, energy_tol)
                    if du_ok and energy_ok:
                        converged = True
                        break

                if mode == "line_search":
                    u_new, R_new, F_int_new, Rn_new, _alpha = _armijo_line_search_step(
                        _residual, u_free, du, Rn, tol, 0.0, verbose=verbose,
                        verbose_prefix=f"    step {i:3d} it {it:2d}  ")
                else:
                    u_new = u_free + du
                    R_new, F_int_new = _residual(u_new)
                    Rn_new = np.linalg.norm(R_new)

                delta_u = np.zeros(n_dof)
                delta_u[free] = u_new - u_free
                fesystem.update_iter_states(u, delta_u, mat, **kwargs)
                u_free, R_free, F_int, Rn = u_new, R_new, F_int_new, Rn_new
            return converged, u_free, Rn

        converged, u_free, Rn = _attempt("plain")
        if not converged and line_search:
            if verbose:
                print(f"  step {i:3d}: plain Newton did not converge in "
                      f"{max_iter} iterations (|R|={Rn:.3e}) -- retrying "
                      f"with line search")
            converged, u_free, Rn = _attempt("line_search")

        if not converged:
            raise RuntimeError(
                f"solve_nonlinear_displacement_control: Newton-Raphson failed "
                f"to converge at step {i} (u_target={u_target:.6g}), "
                f"|R|={Rn:.3e} after {max_iter} iterations.")

        if n_free:
            u[free] = u_free
        F_int = fesystem.assemble_internal_force(u, mat, **kwargs)
        reaction_hist[i] = F_int[control_dof] - fesystem.F[control_dof]
        U_hist[i] = u
        fesystem.commit_all_states(u, mat, **kwargs)

    return U_hist, reaction_hist


def solve_nonlinear_arc_length(fesystem, mat, delta_L, n_steps=50, tol=1e-8,
                                max_iter=30, verbose=False,
                                du_tol=None, energy_tol=None, line_search=True,
                                **kwargs):
    """Module 19 (general-purpose extensions roadmap Phase 5): Crisfield
    cylindrical arc-length continuation -- see the module docstring
    above for the method. F_ext = lambda * fesystem.F, same reference-
    load convention as solve_nonlinear_static(); lambda is now a SOLVED
    unknown at every step, not a prescribed input.

    delta_L: the fixed arc-length radius per step -- constrains
    ||Delta_u||^2 (over the FREE dofs only) = delta_L^2 every step (the
    "cylindrical" variant: the load-factor increment does NOT enter the
    constraint itself, only the equilibrium equation, which is Crisfield's
    original 1981 formulation and the simplest of the arc-length family).
    Choosing delta_L is problem-dependent -- too large and the predictor
    can jump past a sharp turn in the path (Newton correction may then
    fail to converge or converge to the wrong branch), too small and
    n_steps needs to grow to cover the same total path; this function
    does NOT auto-adapt delta_L step to step (e.g. from the previous
    step's iteration count, a common refinement in production arc-length
    codes) -- a fixed radius, chosen by the caller, for this first pass.

    Root selection (which of the quadratic's two dlambda roots to take)
    uses the standard criterion: whichever root keeps the resulting
    increment most closely aligned (largest dot product) with the
    increment accumulated so far THIS step -- for the very first
    (predictor) "iteration" of a step that means aligning with the
    predictor itself (trivially satisfied, just fixes its sign against
    the PREVIOUS step's converged total increment, so the path keeps
    moving forward instead of retracing itself); for every correction
    iteration after that it discourages the large, spurious jumps a
    naive root choice can produce.

    du_tol/energy_tol (Wave 1 item 9): same meaning as
    solve_nonlinear_static()'s parameters of the same name, evaluated
    against whatever CORRECTION (du_iter, the Crisfield-quadratic root
    already selected above) a given iteration is about to apply --
    None (default) for both means the force-residual check above stays
    the ONLY criterion, identical to before this wave.

    line_search (Wave 1 item 10, default True): the predictor above is
    NEVER retried (mirrors solve_nonlinear_transient(), where line
    search only ever retries the corrector, not the Newmark predictor)
    -- only the CORRECTOR loop is retried, from the SAME predicted
    point, if it exhausts max_iter without converging. Because a
    correction here is the COUPLED pair (du_iter, dlam) -- not a plain
    K_T^-1 @ R the way the other drivers' corrections are -- the
    backtrack scales BOTH by the same alpha (the natural generalization
    of _armijo_line_search_step()'s single-vector direction to this
    driver's own (displacement, load-factor) unknown, done here by
    handing it the AUGMENTED vector [u_free; lambda] and direction
    [du_iter; dlam]). One honest caveat vs. the pure-displacement case
    in the other drivers: the clean m'(0)<0 descent-direction proof
    there relies on K_T@du = -R exactly; here du_iter/dlam jointly
    satisfy the arc-length quadratic rather than a plain linear solve
    against R alone, so descent is the standard practical assumption
    for this generalization, not a proof of the same rigor -- validated
    empirically instead (tests/test_arc_length.py's line-search-forced
    regression case), same as every other globalization choice in this
    module that started from evidence rather than a guarantee (see e.g.
    solve_nonlinear_transient()'s own "trust region" docstring section).

    Returns (load_factors (n_steps+1,), U_hist (n_steps+1, n_dof)) --
    ALWAYS includes the trivial zero-state as row 0 (unlike
    solve_nonlinear_static()/solve_nonlinear_displacement_control(),
    there is no load_factors array to control this -- lambda is solved
    for, not prescribed, so there is nothing else to put there)."""
    free = list(fesystem.free_dofs)
    F_total = fesystem.F.copy()
    F_ref = F_total[free]
    n_dof = fesystem.n_dof

    u = np.zeros(n_dof)
    lam = 0.0
    load_factors = np.zeros(n_steps + 1)
    U_hist = np.zeros((n_steps + 1, n_dof))
    U_hist[0] = u
    fesystem.commit_all_states(u, mat, **kwargs)

    prev_total_du = None   # previous step's converged Delta_u, for predictor sign

    def _residual(u_trial_free, lam_trial):
        u[free] = u_trial_free
        F_int_trial = fesystem.assemble_internal_force(u, mat, **kwargs)
        return lam_trial * F_ref - F_int_trial[free], F_int_trial

    for step in range(1, n_steps + 1):
        u_n = u.copy()
        lam_n = lam
        ref = max(np.linalg.norm(F_ref), 1e-30)

        # ---- predictor (iteration 0): pure tangent response to F_ref,
        # scaled to exactly hit the arc-length radius -- computed ONCE
        # per step, common ground for every corrector attempt below
        # (never itself retried/line-searched -- see this function's
        # own docstring addendum). ----
        K_T0 = fesystem.assemble_tangent_stiffness(u, mat, **kwargs)
        du_t0 = np.linalg.solve(K_T0[np.ix_(free, free)], F_ref)
        sign = 1.0 if prev_total_du is None or np.dot(prev_total_du, du_t0) >= 0 else -1.0
        dlam0 = sign * delta_L / np.linalg.norm(du_t0)
        u_pred_free = u_n[free] + dlam0 * du_t0
        lam_pred = lam_n + dlam0
        delta_u0 = np.zeros(n_dof)
        delta_u0[free] = dlam0 * du_t0
        u[free] = u_pred_free
        fesystem.update_iter_states(u, delta_u0, mat, **kwargs)
        iter_state_after_predictor = copy.deepcopy(fesystem.iter_state)
        du_total0 = dlam0 * du_t0

        def _attempt(mode):
            # Both attempts start fresh from the SAME predicted point
            # and iter_state (see docstring addendum) -- a failed
            # "plain" attempt must not leak its own trajectory into the
            # "line_search" retry, mirroring every other driver's own
            # per-attempt reset in this module.
            fesystem.iter_state = copy.deepcopy(iter_state_after_predictor)
            u_free = u_pred_free.copy()
            lam_local = lam_pred
            du_total = du_total0.copy()
            R_free, F_int = _residual(u_free, lam_local)
            Rn = np.linalg.norm(R_free)
            converged = False
            for it in range(max_iter):
                if verbose:
                    tag = " [line search]" if mode == "line_search" else ""
                    print(f"  step {step:3d} it {it:2d}{tag}  lambda={lam_local:.6f}  |R|={Rn:.3e}")
                residual_ok = Rn < tol * ref or Rn < tol
                if residual_ok and du_tol is None and energy_tol is None:
                    converged = True
                    break

                K_T = fesystem.assemble_tangent_stiffness(u, mat, **kwargs)
                du_r = np.linalg.solve(K_T[np.ix_(free, free)], R_free)
                du_t = np.linalg.solve(K_T[np.ix_(free, free)], F_ref)
                a1 = du_t @ du_t
                a2 = 2.0 * (du_total + du_r) @ du_t
                a3 = (du_total + du_r) @ (du_total + du_r) - delta_L ** 2
                disc = a2 ** 2 - 4.0 * a1 * a3
                if disc < 0.0:
                    raise RuntimeError(
                        f"solve_nonlinear_arc_length: no real root at step "
                        f"{step}, iteration {it}{' [line search]' if mode == 'line_search' else ''} "
                        f"(discriminant={disc:.3e} < 0) -- delta_L is probably "
                        f"too large for the curvature of this part of the "
                        f"path; try a smaller delta_L or more steps.")
                sq = np.sqrt(disc)
                dlam1 = (-a2 + sq) / (2.0 * a1)
                dlam2 = (-a2 - sq) / (2.0 * a1)
                cos1 = (du_total + du_r + dlam1 * du_t) @ du_total
                cos2 = (du_total + du_r + dlam2 * du_t) @ du_total
                dlam = dlam1 if cos1 >= cos2 else dlam2
                du_iter = du_r + dlam * du_t

                if residual_ok:
                    F_ext_free = lam_local * F_ref
                    du_ok, energy_ok = _extra_convergence_ok(
                        R_free, du_iter, u_free, F_ext_free, F_int[free],
                        du_tol, energy_tol)
                    if du_ok and energy_ok:
                        converged = True
                        break

                if mode == "line_search":
                    x_free = np.concatenate([u_free, [lam_local]])
                    dx = np.concatenate([du_iter, [dlam]])

                    def _res_x(x_trial):
                        return _residual(x_trial[:-1], x_trial[-1])

                    x_new, R_new, F_int_new, Rn_new, _alpha = _armijo_line_search_step(
                        _res_x, x_free, dx, Rn, tol, ref, verbose=verbose,
                        verbose_prefix=f"    step {step:3d} it {it:2d}  ")
                    u_new_free, lam_new = x_new[:-1], x_new[-1]
                    applied_du, applied_dlam = u_new_free - u_free, lam_new - lam_local
                else:
                    u_new_free = u_free + du_iter
                    lam_new = lam_local + dlam
                    R_new, F_int_new = _residual(u_new_free, lam_new)
                    Rn_new = np.linalg.norm(R_new)
                    applied_du, applied_dlam = du_iter, dlam

                delta_u = np.zeros(n_dof)
                delta_u[free] = applied_du
                fesystem.update_iter_states(u, delta_u, mat, **kwargs)
                du_total = du_total + applied_du
                u_free, lam_local = u_new_free, lam_new
                R_free, F_int, Rn = R_new, F_int_new, Rn_new
            return converged, u_free, lam_local, Rn, du_total

        converged, u_free, lam, Rn, du_total = _attempt("plain")
        if not converged and line_search:
            if verbose:
                print(f"  step {step:3d}: plain Newton did not converge in "
                      f"{max_iter} iterations (|R|={Rn:.3e}) -- retrying "
                      f"with line search")
            converged, u_free, lam, Rn, du_total = _attempt("line_search")

        if not converged:
            raise RuntimeError(
                f"solve_nonlinear_arc_length: Newton-Raphson failed to converge "
                f"at step {step} (lambda={lam:.4f}), |R|={Rn:.3e} after "
                f"{max_iter} iterations -- try a smaller delta_L.")

        u[free] = u_free
        prev_total_du = du_total
        load_factors[step] = lam
        U_hist[step] = u
        fesystem.commit_all_states(u, mat, **kwargs)

    return load_factors, U_hist


def solve_nonlinear_koiter_newton(fesystem, mat, delta_L, n_steps=50, tol=1e-8,
                                   max_iter=30, predictor_tol=0.1, fd_rel=1e-2,
                                   growth_factor=2.0, max_growth_iters=6,
                                   shrink_factor=0.5, max_shrink_iters=8,
                                   verbose=False, du_tol=None, energy_tol=None,
                                   **kwargs):
    """Module 21 (Koiter-Newton continuation): single-branch predictor/
    corrector path-following, replacing solve_nonlinear_arc_length()'s
    LINEAR tangent predictor with a CUBIC one built from a genuine
    Koiter asymptotic expansion of the equilibrium path at the current
    point -- the actual method Liang, Abdalla & Gurdal published as
    "the Koiter-Newton (KN) approach" (IJNME 2013, 10.1002/nme.4581),
    reproduced here to match the SPECIFIC formulation of the paper this
    solver was written for: Yang, Liang, Rong & Sun, "A hybrid reduced-
    order modeling technique for nonlinear structural dynamic
    simulation," Aerospace Science and Technology 84 (2019), Section
    4.2 (their own related-work section cites the KN method [44]-[46]
    as the technique behind this exact static-ROM construction; Eqs.
    16-22 there are what the predictor/corrector below reproduce almost
    verbatim -- NOT the shorter, more generic derivation in Liang & Sun,
    ICCM2017, which an earlier draft of this docstring/implementation
    followed before the exact target paper's own Eqs. 16-22 were
    re-examined directly, see the "CORRECTOR" section below for the one
    place this changed something real, not just cosmetic).

    THE ASYMPTOTIC EXPANSION (single perturbation direction, i.e. the
    published method's m=0 case -- see "SCOPE" below): write the exact
    nonlinear equilibrium equations, Taylor-expanded to cubic order
    about the CURRENT converged point (u_n, lambda_n), as

        L(w) + Q(w,w) + C(w,w,w) = lambda_rel * F_ref            (*)

    (w = u - u_n, lambda_rel = lambda - lambda_n, L = K_T(u_n) the
    current tangent stiffness, Q/C the quadratic/cubic terms of
    F_int(u_n + w) - F_int(u_n)'s own Taylor series -- exactly the
    paper's Eq. 1 specialized to ONE sub-load F_ref, i.e. m+1=1). Both
    w and lambda_rel are expanded in a SINGLE scalar perturbation
    parameter a (the paper's Eq. 3-4 with every index collapsed to 1):

        w(a)      = a*u1 + a^2*u11 + O(a^3)
        lambda(a) = Lbar1*a + Qbar11*a^2 + Cbar1111*a^3 + O(a^4)

    Substituting into (*) and collecting powers of a gives three linear
    systems (paper Eqs. 5-7), all sharing the SAME (nf+1)x(nf+1)
    "bordered" coefficient matrix [[K_T, -F_ref], [-F_ref^T, 0]] (built
    and reused, not re-factored, for both u1 and u11 below):

      Eq.5:  K_T u1  - F_ref*Lbar1  = 0,        F_ref^T u1  = 1
      Eq.6:  K_T u11 - F_ref*Qbar11 = -Q(u1,u1), F_ref^T u11 = 0
      Eq.7:  Cbar1111 = u1.C(u1,u1,u1) - 2*u11.(K_T @ u11)

    (Eq.7 collapses to this simple form because the paper's general
    "-2/3*(u_ij.L(u_pk) + u_jk.L(u_pi) + u_ki.L(u_pj))" correction has
    all three terms identical, i,j,k,p all being the same single index
    -- a direct simplification of the published formula, not a
    different one.) Eq.5's F_ref^T u1 = 1 normalization is the paper's
    own convention, not an arbitrary choice -- it fixes what "a" means
    (roughly, a generalized arc-length along the load direction) and
    is exactly what makes u1 -- and therefore a itself -- shrink
    automatically near a limit point (K_T nearly singular -> u1 blows
    up -> the SAME target delta_L is reached at a much smaller a), the
    mechanism behind the method's large-step efficiency claim.

    GETTING Q AND C WITHOUT HAND-DERIVING THEM PER ELEMENT: the
    published method computes Q/C analytically from each element's own
    strain-energy polynomial (its Eq. 7's "up to fourth order" strain-
    energy derivatives) -- impractical to redo for every element class
    in this package (trusses, beams, shells, solids, each with its own
    kinematics). Instead, since Q(u1,u1) and C(u1,u1,u1) are exactly
    the 2nd/3rd Taylor coefficients of the SCALAR-parametrized vector
    function g(eps) = F_int(u_n + eps*u1) - F_int(u_n), they are
    extracted with a symmetric finite-difference stencil against
    fesystem.assemble_internal_force() -- the SAME "differentiate the
    existing internal-force/tangent callable numerically" trick this
    module's own transient driver and elements/shells.py's corotational
    coupling tangent already use elsewhere in this package for exactly
    this reason (no per-element analytic re-derivation needed, works
    identically for every element already registered in `elements/`):

        Q(u1,u1)    = [g(h) + g(-h) - 2*g(0)] / (2*h^2)
        C(u1,u1,u1) = [g(h) - g(-h) - 2*h*(K_T@u1)] / (2*h^3)

    (central differences of a cubic-accurate Taylor series; g(0)=0 and
    g'(0)=K_T@u1 by construction). The ACTUAL probe displacement h*u1
    has norm fd_rel * max(||u_n||, delta_L) (default fd_rel=1e-2 --
    see _fd_probe_length for why this is tied to the step size rather
    than an absolute floor), independent of u1's own (normalization-
    dependent, potentially large or tiny) magnitude.

    ADAPTIVE STEP SIZE ("the unbalanced-force criterion", paper's own
    Section on the KN algorithm): unlike solve_nonlinear_arc_length's
    FIXED delta_L, this driver grows/shrinks the perturbation parameter
    a directly from how well the cubic ROM predicts the TRUE (full
    finite-element) residual -- exactly the paper's own description
    ("the exact unbalanced force is calculated using the full finite
    element model... if the criterion is not satisfied, the initial
    prediction will be stopped"). Each trial costs only ONE
    assemble_internal_force() call (no tangent assembly, no linear
    solve) -- cheap enough to try several a's per step: start from
    a0 = delta_L/||u1|| (so a "typical" first trial reaches roughly the
    same displacement-space radius as solve_nonlinear_arc_length's own
    delta_L, for direct comparability), then either GROW a by
    growth_factor (up to max_growth_iters times) while the resulting
    predictor's residual ratio ||R_pred||/||F_ref|| stays under
    predictor_tol, or -- if even a0 already exceeds predictor_tol --
    SHRINK a by shrink_factor (up to max_shrink_iters times) until it
    doesn't. Raises RuntimeError if shrinking max_shrink_iters times
    still can't satisfy predictor_tol (an extremely locally nonlinear
    region -- try a smaller delta_L or looser predictor_tol).

    CORRECTOR (Yang et al. 2019, Eqs. 21-22, reproduced almost
    verbatim -- an earlier version of this function instead reused
    solve_nonlinear_arc_length()'s fresh-tangent Crisfield corrector;
    re-examining the target paper's OWN corrector directly turned up a
    real, better-justified alternative, not just a stylistic one, so
    this was replaced): once a is chosen, (u_pred, lambda_pred) seeds a
    chord/modified-Newton iteration that solves a REMAINDER correction
    (du_tilde, dphi_tilde) at every iteration from the SAME bordered
    system already factored above for u1/u11 (paper Eq. 21's own
    matrix, [[L, -F_ref], [-F_ref^T, 0]] -- L frozen at u_n), against a
    fresh right-hand side built from the CURRENT full residual:

        [[K_T(u_n), -F_ref], [-F_ref^T, 0]] @ [du_tilde; dphi_tilde]
            = [r_free; 0]

    where r = lambda_c*F_total - F_int(u_c). This is exactly the
    first-order (frozen-Jacobian) Newton linearization of that
    residual -- re-derived directly rather than trusted verbatim from
    the source PDF's own equation (21), whose image-based OCR
    transcription of a sign inside the unknown vector (the paper prints
    the second unknown as "-Delta phi-tilde," which would flip the sign
    on the F_ref@dphi_tilde term above and turn this into a DIVERGING,
    not converging, fixed-point map -- confirmed directly: implementing
    it exactly as transcribed makes every corrector iteration's
    residual grow by a clean, repeatable ~2x per iteration, the
    signature of a sign error, not a convergence-rate problem). The
    LEFT-HAND matrix is the SAME K_T(u_n)-based bordered matrix already
    factored for u1/u11, reused UNCHANGED (never re-assembled at the
    trial state, unlike solve_nonlinear_arc_length's corrector) for
    every iteration of every retry this whole step -- this part of the
    paper's own description IS implemented verbatim: "the coefficient
    matrix is exactly the same as that used for construction of the
    reduced-order model[.] This indicates that no new matrix
    factorizations are produced in the correction phase." Trading fresh-tangent quadratic Newton
    convergence for a frozen-tangent linear (chord) one is the
    deliberate cost/accuracy trade this driver's whole efficiency case
    rests on -- consistent with the paper's own claim that per-step
    cost is "roughly the same" between the two methods, with the actual
    speedup coming entirely from needing FEWER, LARGER steps (the cubic
    predictor), not cheaper correction inside each step. A failed
    correction (paper: "if the iteration number exceeds a given
    threshold (such as 15)...") is handled here by SHRINKING the step a
    and retrying from the SAME frozen factorization -- no new tangent
    assembly even on retry, since the frozen matrix doesn't depend on a
    at all, only the predictor RHS does.

    SCOPE (read before trusting this on a buckling-sensitive problem):
    this implements the PRIMARY-PATH-ONLY case of the published method
    (their own m=0: the reduced model has exactly one perturbation
    direction, u1, spanning the primary equilibrium path). The
    published method's actual headline strength -- tracing THROUGH a
    bifurcation with closely spaced buckling modes by adding m>=1 EXTRA
    perturbation directions (the near-critical eigenmodes of K_T) to
    the same reduced basis -- is NOT implemented here. On a structure
    whose limit points are simple (one mode at a time, not closely
    spaced/interacting), this reduces to exactly what the method's own
    published single-mode numerical examples do (see e.g. Liang & Sun
    ICCM2017's six-beam examples, each run with total reduced dimension
    m+1=2 -- one primary-path DOF plus one buckling DOF -- collapsing
    to exactly m+1=1, this function's case, whenever a problem has no
    bifurcation to add a second direction for at all, e.g. any snap-
    through/limit-point path with no branching). Extending to m>=1
    would mean: (a) an eigenvalue check on K_T to detect an
    approaching/crossed bifurcation, (b) adding each near-critical
    eigenvector as an extra perturbation direction in the SAME bordered
    system (a straightforward but real generalization of the m=0 system
    above, not implemented), and (c) mode-interaction terms in Q/C
    across those extra directions. Left as a documented extension, not
    attempted here.

    du_tol/energy_tol (Wave 1 item 9 -- see this module's own docstring
    "Wave 1" section): applied to the corrector below (_run_corrector)
    as an ADDITIONAL, AND-combined requirement on top of the force-
    residual check it already had, via the SAME _extra_convergence_ok()
    helper every other driver in this module now uses. Deliberately
    NOT given a line_search parameter (item 10) -- see this module's
    own docstring for why bolting Armijo backtracking onto this
    specific bordered (du, dlambda) corrector was judged a materially
    riskier change than the shrink-`a` retry it already has; both
    default to None (off), so an existing caller's results/cost are
    unchanged.

    Returns (load_factors, U_hist), same convention as
    solve_nonlinear_arc_length() (row 0 is the trivial zero state)."""
    free = list(fesystem.free_dofs)
    F_total = fesystem.F.copy()
    F_ref = F_total[free]
    n_dof = fesystem.n_dof
    nf = len(free)

    u_n = np.zeros(n_dof)
    lam_n = 0.0
    load_factors = np.zeros(n_steps + 1)
    U_hist = np.zeros((n_steps + 1, n_dof))
    U_hist[0] = u_n
    fesystem.commit_all_states(u_n, mat, **kwargs)

    for step in range(1, n_steps + 1):
        ref = max(np.linalg.norm(F_ref), 1e-30)

        # ---- build the reduced (Koiter asymptotic) cubic predictor ----
        K_T = fesystem.assemble_tangent_stiffness(u_n, mat, **kwargs)
        K_ff = K_T[np.ix_(free, free)]
        A_aug = np.zeros((nf + 1, nf + 1))
        A_aug[:nf, :nf] = K_ff
        A_aug[:nf, nf] = -F_ref
        A_aug[nf, :nf] = -F_ref
        # ONE factorization, reused for u1/u11 below AND for every
        # corrector iteration/retry this whole step -- see the
        # corrector's own comment for why this matters (it is literally
        # what the paper's "no new matrix factorizations... in the
        # correction phase" claim refers to).
        lu_and_piv = lu_factor(A_aug)

        rhs1 = np.zeros(nf + 1)
        rhs1[nf] = -1.0
        x1 = lu_solve(lu_and_piv, rhs1)
        u1, Lbar1 = x1[:nf], x1[nf]

        u1_norm = np.linalg.norm(u1)
        h = _fd_probe_length(u_n[free], delta_L, fd_rel) / max(u1_norm, 1e-300)
        w_plus = u_n.copy(); w_plus[free] += h * u1
        w_minus = u_n.copy(); w_minus[free] -= h * u1
        F0 = fesystem.assemble_internal_force(u_n, mat, **kwargs)
        F_plus = fesystem.assemble_internal_force(w_plus, mat, **kwargs)
        F_minus = fesystem.assemble_internal_force(w_minus, mat, **kwargs)
        g_plus = F_plus[free] - F0[free]
        g_minus = F_minus[free] - F0[free]
        Q11 = (g_plus + g_minus) / (2.0 * h ** 2)
        C111 = (g_plus - g_minus - 2.0 * h * (K_ff @ u1)) / (2.0 * h ** 3)

        rhs2 = np.zeros(nf + 1)
        rhs2[:nf] = -Q11
        x2 = lu_solve(lu_and_piv, rhs2)
        u11, Qbar11 = x2[:nf], x2[nf]

        Cbar1111 = u1 @ C111 - 2.0 * (u11 @ (K_ff @ u11))

        def _predict(a):
            lam_p = lam_n + Lbar1 * a + Qbar11 * a ** 2 + Cbar1111 * a ** 3
            u_p = u_n.copy()
            u_p[free] = u_n[free] + a * u1 + a ** 2 * u11
            return u_p, lam_p

        def _residual_ratio(u_p, lam_p):
            F_int_p = fesystem.assemble_internal_force(u_p, mat, **kwargs)
            R = lam_p * F_total - F_int_p
            return np.linalg.norm(R[free]) / ref

        # ---- adaptive predictor sizing: grow/shrink 'a' from the true
        # (full finite-element) residual the cubic ROM predicts -- the
        # paper's own "unbalanced force" validity criterion ----
        a0 = delta_L / max(u1_norm, 1e-300)
        u_ok, lam_ok = _predict(a0)
        ratio0 = _residual_ratio(u_ok, lam_ok)
        a_ok = a0
        if ratio0 <= predictor_tol:
            for _ in range(max_growth_iters):
                a_try = a_ok * growth_factor
                u_try, lam_try = _predict(a_try)
                if _residual_ratio(u_try, lam_try) <= predictor_tol:
                    a_ok, u_ok, lam_ok = a_try, u_try, lam_try
                else:
                    break
        else:
            found = False
            a_try = a0
            for _ in range(max_shrink_iters):
                a_try *= shrink_factor
                u_try, lam_try = _predict(a_try)
                if _residual_ratio(u_try, lam_try) <= predictor_tol:
                    a_ok, u_ok, lam_ok, found = a_try, u_try, lam_try, True
                    break
            if not found:
                raise RuntimeError(
                    f"solve_nonlinear_koiter_newton: could not shrink the "
                    f"asymptotic predictor to satisfy predictor_tol={predictor_tol:.3g} "
                    f"at step {step} after {max_shrink_iters} halvings -- try a "
                    f"smaller delta_L or a looser predictor_tol.")

        if verbose:
            print(f"  step {step:3d}  predictor a={a_ok:.4e}  "
                  f"|dlambda|={abs(lam_ok - lam_n):.4e}  "
                  f"|du|={np.linalg.norm(u_ok[free] - u_n[free]):.4e}")

        # ---- corrector: Yang et al. 2019 Eq. 21-22's own remainder-
        # correction scheme, reproduced almost verbatim -- NOT a re-use
        # of solve_nonlinear_arc_length()'s corrector. The tangent L is
        # FROZEN at the expansion point u_n for the ENTIRE corrector
        # (every iteration, every retry): each correction (d_utilde,
        # d_phitilde) solves the SAME bordered matrix A_aug already
        # factored above (lu_and_piv), just against a new right-hand
        # side [r_free; 0] built from the CURRENT full-model residual
        # r = lam_c*F_total - F_int(u_c) -- "the coefficient matrix is
        # exactly the same as that used for construction of the
        # reduced-order model[,] no new matrix factorizations are
        # produced in the correction phase" (paper, verbatim). This is
        # a genuine chord/modified-Newton iteration (linear, not
        # quadratic, convergence rate), matching the paper's own
        # documented threshold (verbose text below cites 15 as their
        # example cutoff) -- and is the reason a FAILED correction is
        # handled by SHRINKING THE STEP AND RETRYING (bringing the
        # trial state back closer to the expansion point, where the
        # frozen tangent is a better approximation to the true one)
        # rather than by falling back to fresh-tangent full Newton,
        # which is what a genuinely different (and, per the same
        # "roughly the same per-step cost" efficiency claim, cheaper
        # each iteration) implementation choice would do.
        def _run_corrector(a_seed):
            u_c, lam_c = _predict(a_seed)
            predictor_delta_u = np.zeros(n_dof)
            predictor_delta_u[free] = u_c[free] - u_n[free]
            fesystem.update_iter_states(u_c, predictor_delta_u, mat, **kwargs)
            for it in range(max_iter):
                F_int = fesystem.assemble_internal_force(u_c, mat, **kwargs)
                R = lam_c * F_total - F_int
                Rn = np.linalg.norm(R[free])
                if verbose:
                    print(f"    it {it:2d}  lambda={lam_c:.6f}  |R|={Rn:.3e}")
                residual_ok = Rn < tol * ref or Rn < tol
                if residual_ok and du_tol is None and energy_tol is None:
                    return True, u_c, lam_c, Rn

                rhs3 = np.zeros(nf + 1)
                rhs3[:nf] = R[free]
                x3 = lu_solve(lu_and_piv, rhs3)
                du_tilde, dphi_tilde = x3[:nf], x3[nf]

                if residual_ok:
                    # Wave 1 item 9 -- displacement/energy criteria,
                    # ADDITIONAL to the residual check above (see this
                    # function's own docstring). No line_search here --
                    # see the docstring's own explanation.
                    du_ok, energy_ok = _extra_convergence_ok(
                        R[free], du_tilde, u_c[free], lam_c * F_ref,
                        F_int[free], du_tol, energy_tol)
                    if du_ok and energy_ok:
                        return True, u_c, lam_c, Rn

                delta_u = np.zeros(n_dof)
                delta_u[free] = du_tilde
                u_c[free] += du_tilde
                lam_c += dphi_tilde
                fesystem.update_iter_states(u_c, delta_u, mat, **kwargs)

            return False, u_c, lam_c, Rn

        a_corr = a_ok
        converged = False
        # snapshot iter_state (Module 23) BEFORE each retry attempt and
        # restore it on failure -- _run_corrector mutates iter_state as
        # it goes (every accepted correction, including the predictor's
        # own jump), and a REJECTED attempt's mutations must not leak
        # into the NEXT (smaller-a_corr) retry's own fresh start from
        # u_n, exactly the "rejected trial corrupts state" risk this
        # module's own docstring for update_iter_states() warns about.
        iter_state_at_step_start = copy.deepcopy(fesystem.iter_state)
        for _retry in range(max_shrink_iters):
            converged, u, lam, Rn = _run_corrector(a_corr)
            if converged:
                break
            fesystem.iter_state = copy.deepcopy(iter_state_at_step_start)
            if verbose:
                print(f"  step {step:3d}: corrector did not converge at "
                      f"a={a_corr:.4e} (|R|={Rn:.3e}) -- shrinking and retrying "
                      f"(SAME frozen tangent, no re-assembly)")
            a_corr *= shrink_factor

        if not converged:
            raise RuntimeError(
                f"solve_nonlinear_koiter_newton: chord corrector failed to "
                f"converge at step {step} even after {max_shrink_iters} "
                f"predictor shrinks (last |R|={Rn:.3e}) -- try a smaller "
                f"delta_L or a tighter predictor_tol.")

        u_n, lam_n = u, lam
        load_factors[step] = lam_n
        U_hist[step] = u_n
        fesystem.commit_all_states(u_n, mat, **kwargs)

    return load_factors, U_hist


# ---------------------------------------------------------------------------
# Module 24 helpers: shared by solve_nonlinear_koiter_newton_generic() only.
# Kept private (leading underscore) and separate from the m=0 driver above,
# which is deliberately left untouched -- see that function's own docstring
# for why (NonLin-HyROM's existing training pipeline depends on it exactly
# as validated).
# ---------------------------------------------------------------------------
def _fd_probe_length(u_n_free, step_scale, fd_rel):
    """Length (free-DOF displacement norm) of the finite-difference probe
    the Koiter-Newton drivers use to extract Q/C Taylor coefficients:
    fd_rel * max(||u_n||, step_scale), where step_scale is the size of
    the displacement step the cubic model is about to be used over
    (delta_L for the continuation drivers; the linear estimate of the
    displacement to the target load for the target-load driver).

    WHY RELATIVE TO THE STEP, NOT AN ABSOLUTE FLOOR (fixed 2026-09-23):
    the probe used to be fd_rel * max(1.0, ||u_n||) with fd_rel=1e-4 --
    i.e. an ABSOLUTE 1e-4 length units from an undeformed state,
    whatever the model's units or size. On a curved shell whose nodes sit
    far from the origin (Case 2 cylindrical shell, R=2540mm, crown at
    Z~2540mm), floating-point rounding in the internal force at that
    probe size swamped the cubic term: C111 came out with the WRONG SIGN
    and ~1000x too large (verified: Cbar1111=-3.4e-9 at a 1e-4mm probe
    vs. a stable +3.28e-12 for any probe between 1e-2 and 1e-1mm). That
    left the target-load driver's reduced cubic with only a NEGATIVE
    real root, so it stepped AWAY from the target and oscillated for
    all max_expansions (20 expansions, lambda wandering in [-0.76, 0])
    on a load whose linear displacement was only ~0.4mm. Tying the
    probe to the step's own displacement scale makes it unit- and
    size-independent: the cubic model is only ever evaluated over
    ~step_scale anyway, so a probe of 1% of that (default fd_rel=1e-2)
    keeps central-difference truncation error at O(fd_rel^2) of the
    step while staying far above rounding noise."""
    return fd_rel * max(np.linalg.norm(u_n_free), step_scale)


def _directional_QC(fesystem, mat, u_n, free, direction, K_ff, h_scale, F0_free, **kwargs):
    """Central-difference extraction of the quadratic/cubic Taylor
    coefficients Q(d,d), C(d,d,d) of g(eps) = F_int(u_n + eps*d) -
    F_int(u_n) along a single direction d -- EXACTLY the stencil
    solve_nonlinear_koiter_newton() uses inline for its own (single-
    direction) Q11/C111, factored out here so it can be reused both for
    the m=0 fallback below and as the building block of
    _two_direction_QC()'s polarization-identity construction."""
    dnorm = np.linalg.norm(direction)
    h = h_scale / max(dnorm, 1e-300)
    w_plus = u_n.copy(); w_plus[free] += h * direction
    w_minus = u_n.copy(); w_minus[free] -= h * direction
    F_plus = fesystem.assemble_internal_force(w_plus, mat, **kwargs)
    F_minus = fesystem.assemble_internal_force(w_minus, mat, **kwargs)
    g_plus = F_plus[free] - F0_free
    g_minus = F_minus[free] - F0_free
    Q = (g_plus + g_minus) / (2.0 * h ** 2)
    C = (g_plus - g_minus - 2.0 * h * (K_ff @ direction)) / (2.0 * h ** 3)
    return Q, C


def _two_direction_QC(fesystem, mat, u_n, free, u0, u1, K_ff, fd_rel, F0_free, step_scale=1.0,
                      **kwargs):
    """Extract the full symmetric quadratic (Q_ij, i,j in {0,1}) and
    cubic (C_ijk, i,j,k in {0,1}) Taylor-coefficient tensors of
    g(eps0,eps1) = F_int(u_n + eps0*u0 + eps1*u1) - F_int(u_n) from
    FOUR single-direction probes (u0 alone, u1 alone, u0+u1, u0-u1),
    each run through the SAME already-validated 1-D stencil above --
    a polarization-identity construction, not a new finite-difference
    formula. Writing h(t) = g(t,t) and k(t) = g(t,-t) as their own
    scalar-parametrized Taylor series,

        h(t) = t*(g0+g1) + t^2*(Q00+2Q01+Q11) + t^3*(C000+3C001+3C011+C111) + ...
        k(t) = t*(g0-g1) + t^2*(Q00-2Q01+Q11) + t^3*(C000-3C001+3C011-C111) + ...

    the two DIRECT single-direction probes (u0 alone gives Q00/C000,
    u1 alone gives Q11/C111) plus these two SUM probes give exactly
    enough independent linear equations to solve for the two remaining
    unknowns Q01 and (C001, C011) -- pure algebra on four already-
    validated stencil outputs, so no new numerical-differentiation risk
    is introduced. Q01 is computed twice, independently (once from the
    diagonal probe, once from the anti-diagonal one), and their
    disagreement is returned as 'Q01_consistency' -- since the two are
    mathematically identical, any real difference is pure finite-
    difference truncation/step-size error, a cheap built-in sanity
    check with no analytic-benchmark needed.

    step_scale: see _fd_probe_length (the generic driver passes delta_L;
    the default 1.0 reproduces the pre-2026-09-23 absolute-floor probe,
    kept only so direct unit-level callers are unaffected)."""
    h_scale = _fd_probe_length(u_n[free], step_scale, fd_rel)

    Q00, C000 = _directional_QC(fesystem, mat, u_n, free, u0, K_ff, h_scale, F0_free, **kwargs)
    Q11, C111 = _directional_QC(fesystem, mat, u_n, free, u1, K_ff, h_scale, F0_free, **kwargs)
    Qd, Cd = _directional_QC(fesystem, mat, u_n, free, u0 + u1, K_ff, h_scale, F0_free, **kwargs)
    Qa, Ca = _directional_QC(fesystem, mat, u_n, free, u0 - u1, K_ff, h_scale, F0_free, **kwargs)

    Q01_from_diag = (Qd - Q00 - Q11) / 2.0
    Q01_from_anti = (Q00 + Q11 - Qa) / 2.0
    Q01 = 0.5 * (Q01_from_diag + Q01_from_anti)

    D_plus = Cd - C000 - C111        # = 3*(C001 + C011)
    D_minus = Ca - C000 + C111       # = 3*(C011 - C001)
    C011 = (D_plus + D_minus) / 6.0
    C001 = (D_plus - D_minus) / 6.0

    denom = max(np.linalg.norm(Q01), 1e-300)
    consistency = np.linalg.norm(Q01_from_diag - Q01_from_anti) / denom

    return dict(Q00=Q00, Q01=Q01, Q11=Q11, C000=C000, C001=C001, C011=C011,
                C111=C111, Q01_consistency=consistency)


def _generic_cbar_term(p, i, j, k, u_dir, u_quad, C_lin, K_ff):
    """Eq.7 of the Koiter-Newton papers, in its GENERAL (not m=0-
    collapsed) form -- the algebraic formula
    solve_nonlinear_koiter_newton()'s own docstring quotes and then
    simplifies for the single-direction case:

        Cbar[p; i,j,k] = u_p . C(u_i,u_j,u_k)
            - (2/3)*[ u_ij.K(u_pk) + u_jk.K(u_pi) + u_ki.K(u_pj) ]

    (K(x) := K_ff @ x). Implemented generically against dictionaries
    keyed by (sorted) index tuples rather than hand-substituting each
    of the four needed (i,j,k) triples separately, to avoid an algebra
    mistake in that substitution -- as an internal consistency check,
    calling this with p=i=j=k=0 reproduces
    solve_nonlinear_koiter_newton()'s own "u1.C(u1,u1,u1) -
    2*u11.(K_T@u11)" formula exactly (all three correction terms
    collapse to the same scalar u_00.K(u_00) via K_ff's symmetry,
    a . K(b) == b . K(a) for any a, b)."""
    def uq(a, b):
        return u_quad[tuple(sorted((a, b)))]

    def cl(a, b, c):
        return C_lin[tuple(sorted((a, b, c)))]

    correction = (uq(i, j) @ (K_ff @ uq(p, k)) +
                  uq(j, k) @ (K_ff @ uq(p, i)) +
                  uq(k, i) @ (K_ff @ uq(p, j)))
    return u_dir[p] @ cl(i, j, k) - (2.0 / 3.0) * correction


def solve_nonlinear_koiter_newton_generic(fesystem, mat, delta_L, n_steps=50, tol=1e-8,
                                           max_iter=30, predictor_tol=0.1, fd_rel=1e-2,
                                           growth_factor=2.0, max_growth_iters=6,
                                           shrink_factor=0.5, max_shrink_iters=8,
                                           mode_detect_rel_tol=0.05,
                                           enable_mode_interaction=True,
                                           u0=None, lam0=0.0,
                                           verbose=False, **kwargs):
    """Wave 1 items 9/10 (docs/consolidated_future_roadmap.md) SCOPE
    NOTE: neither was added to this function's correctors. This is the
    most delicate driver in the module -- two corrector branches
    (single- and mode-interaction), the latter solving a cubic with up
    to 3 real roots per step, on top of an already-real sign-error war
    story in the closely related solve_nonlinear_koiter_newton()'s own
    corrector (see this module's own docstring "Wave 1" section for
    the full reasoning that also excluded THAT function's corrector
    from item 10). Touching this one under the same time budget was
    judged the higher-risk choice for the lower-confidence payoff, so
    it is left as documented future work rather than attempted here --
    a caller who needs either convergence criterion or line-search
    globalization on the m>=1 case should use solve_nonlinear_koiter_
    newton() (m=0, both item 9 and, via its own docstring's
    explanation, NOT item 10) or solve_nonlinear_arc_length() (item 9
    AND item 10) instead for now.

    Module 24 (generic Koiter-Newton continuation, m>=1): lifts
    solve_nonlinear_koiter_newton()'s disclosed m=0 scope reduction by
    adding, at each step, an automatic check for a near-critical
    tangent-stiffness mode and -- when one is found -- a SECOND
    perturbation direction built from it, exactly the published
    method's own m=1 case (Liang, Abdalla & Gurdal 2013; Liang & Sun,
    ICCM2017; the same papers solve_nonlinear_koiter_newton()'s own
    docstring cites for the m=0 case this driver generalizes).

    WHY A NEW FUNCTION, NOT AN EXTRA PARAMETER ON THE EXISTING ONE:
    solve_nonlinear_koiter_newton() is already validated (4 passing
    tests, including a von Mises truss snap-through cross-check against
    both arc-length and closed-form) and already in production use by
    NonLin-HyROM's training pipeline. Generalizing it in place would
    risk regressing that exact, working code path for a capability
    (mode interaction) most callers of the m=0 driver don't need. This
    function instead reproduces the m=0 formulas as its OWN fallback
    path (see "m=0 FALLBACK" below) -- verified in
    tests/test_koiter_newton_generic.py to numerically reproduce
    solve_nonlinear_koiter_newton()'s own results on the same problem
    when no critical mode is ever detected -- and only diverges into
    the new two-direction machinery when the eigenvalue check actually
    fires.

    AUTOMATIC MODE DETECTION: at each step, after assembling K_T(u_n),
    the two smallest eigenvalues of K_ff are found via
    scipy.linalg.eigh(K_ff, subset_by_index=[0,1]) (cheap relative to
    the O(nf^3) factorization already needed below -- a partial
    symmetric eigensolve, not a full one). The smallest eigenvalue is
    compared, in a RELATIVE sense, against lam_ref = |lambda_min| of the
    UNDEFORMED structure's own K_ff(u=0) (computed once per call, always
    at u=0, so a restart from an already-near-critical u0 still detects
    it): a near-critical mode is declared whenever
    |lambda_min| < mode_detect_rel_tol * lam_ref, i.e. when the softest
    mode has lost that fraction of its initial stiffness. CHANGED
    2026-09-25: the scale used to be k_scale = trace(K_ff)/nf, which is
    not scale-free. A thin shell's trace is dominated by membrane/drilling
    stiffness, so |lambda_min|/k_scale is tiny in the UNDEFORMED state
    (5.6e-6 on NonLin-HyROM's Case 2 panel), and m=1 fired at step 1 with no
    instability anywhere. That misrouted every shell into the m=1 branch
    (tests/test_koiter_newton_generic.py::
    test_mode_detection_ignores_stiffness_contrast). trace/nf is kept
    only as a fallback if K_ff(0) is itself (near-)singular. The
    Liang et al. 2014 paper instead selects perturbation modes from a
    linear buckling analysis (buckling loads within 20% of the first);
    this per-step, scale-free test is this driver's lighter-weight
    analogue. A band that catches
    BOTH an approaching bifurcation (lambda_min -> 0 from above) and one
    just crossed (lambda_min -> 0 from below), while excluding a
    tangent stiffness that is simply deep past a limit point (large
    negative lambda_min) -- a regime this predictor extension does not
    attempt to fix (the frozen-tangent corrector below still assumes
    the expansion point is a reasonable local approximation). No
    hysteresis is applied across steps -- the check is re-run fresh
    every step, so the active dimension can in principle toggle step to
    step near the detection threshold; this is a documented
    simplification, not a claimed-away limitation (see "SCOPE" below).

    THE EXTRA PERTURBATION LOAD: once a critical mode phi_crit is found
    (the eigenvector for lambda_min, unit-normalized), the paper's own
    prescription is to add ANOTHER sub-load and expand in a second
    scalar parameter for it (Eq. 2's F = sum_p lambda_p*f_p generalized
    to p=0,1). This driver's own choice of that second sub-load is
    f_1 := (K_ff @ phi_crit), rescaled to the same norm as ||F_ref|| --
    NOT phi_crit itself. This is deliberate: the bordering technique
    needs f_1^T u = const to pin down a well-posed SECOND direction u_1
    (see Eq.5 below); with f_1 built this way, u_1 solves
    K_ff @ u_1 - f_1*Lbar1 = 0, f_1^T u_1 = 1, i.e. u_1 is pulled
    directly toward K_ff's own near-null direction (since f_1 IS
    K_ff@phi_crit, u_1 ~ phi_crit to leading order whenever K_ff is
    genuinely near-singular there) -- physically the buckling-mode
    direction the method's own description calls for, while staying
    exactly inside the same well-posed bordered-matrix machinery as the
    real load direction, rather than requiring a separate, differently-
    normalized construction.

    THE GENERALIZED ASYMPTOTIC EXPANSION (m+1=2 directions; collapses
    to solve_nonlinear_koiter_newton()'s own m+1=1 equations when only
    one direction is active): with w = u - u_n expanded in TWO scalar
    parameters a0 (primary/load direction) and a1 (extra/mode
    direction),

        w(a0,a1) = a0*u_0 + a1*u_1
                   + a0^2*u_00 + 2*a0*a1*u_01 + a1^2*u_11 + O(a^3)
        lambda(a0,a1) = Lbar[0,0]*a0 + Lbar[0,1]*a1
                        + Qbar_00[0]*a0^2 + 2*Qbar_01[0]*a0*a1 + Qbar_11[0]*a1^2
                        + Cbar[0;0,0,0]*a0^3 + 3*Cbar[0;0,0,1]*a0^2*a1
                        + 3*Cbar[0;0,1,1]*a0*a1^2 + Cbar[0;1,1,1]*a1^3 + O(a^4)

    the FIRST-order directions u_0, u_1 and the 2x2 matrix of Lagrange
    multipliers Lbar[k,i] solve the SAME bordered system generalized to
    TWO border rows/columns,

        [[K_ff, -f_0, -f_1], [-f_0^T, 0, 0], [-f_1^T, 0, 0]]
            @ [u_i; Lbar[0,i]; Lbar[1,i]] = [0; -delta_{0i}; -delta_{1i}]   (i=0,1)

    ONE factorization of this (nf+2)x(nf+2) matrix, reused for both
    columns i=0,1 AND every second-order solve AND the corrector below
    -- exactly the same "one factorization, many right-hand-sides"
    structure as the m=0 driver, just one row/column larger. The
    quadratic-order solves (u_00, u_01, u_11, and their own Lagrange
    parts Qbar_ij[k]) use the SAME matrix against RHS=[-Q(u_i,u_j); 0; 0]
    for each of the three distinct index pairs -- Q(u_i,u_j) (i,j in
    {0,1}) extracted via _two_direction_QC() above. The cubic
    coefficients Cbar[p;i,j,k] (p is the OUTPUT row, (i,j,k) the
    symmetric index triple) are then pure algebra via
    _generic_cbar_term() -- no further linear solves needed, exactly as
    in the m=0 case's own Eq.7.

    ROW 0 vs ROW 1 -- WHY THIS IS THE PART THAT ACTUALLY DELIVERS MODE
    INTERACTION: row p=0 above (built from f_0=F_ref, the real applied
    load) is lambda(a0,a1) -- an OUTPUT, exactly as in the m=0 case.
    Row p=1 (built from f_1, which has NO real external load behind it)
    is instead a genuine EQUILIBRIUM CONSTRAINT: "lambda_1(a0,a1) = 0"
    for the fictitious sub-load direction, i.e. for a FIXED a0 it is a
    CUBIC POLYNOMIAL in a1 (coefficients Cbar[1;1,1,1], 3*Cbar[1;0,1,1]*a0,
    Lbar[1,1]+2*Qbar_01[1]*a0+3*Cbar[1;0,0,1]*a0^2, and
    Lbar[1,0]*a0+Qbar_00[1]*a0^2+Cbar[1;0,0,0]*a0^3) with up to THREE
    real roots -- this is the actual mode-interaction/bifurcation-
    branch behavior the published method provides beyond a plain
    tangent predictor: multiple candidate equilibrium states at the
    same load-direction step size a0, exactly what "tracing through a
    bifurcation with closely-spaced buckling modes" means. Solved via
    numpy.roots(); when more than one real root exists, the one closest
    to the PREVIOUS step's accepted a1 (0.0 the first time a mode
    activates) is selected -- a continuity/branch-following heuristic,
    not a claim of resolving genuine branch-switching decisions
    automatically (see "SCOPE" below).

    m=0 FALLBACK: when no critical mode is detected, this driver runs
    EXACTLY solve_nonlinear_koiter_newton()'s own formulas (same
    bordered-matrix construction, same u1/u11/Lbar1/Qbar11/Cbar1111,
    same adaptive-a growth/shrink search against the true residual, same
    frozen-tangent chord corrector) -- reproduced inline rather than
    calling that function as a subroutine, so this driver owns its own
    factorization and both branches share the surrounding step/commit
    bookkeeping below. See tests/test_koiter_newton_generic.py for the
    numerical cross-check against solve_nonlinear_koiter_newton() this
    equivalence claim rests on.

    CORRECTOR, m=0 branch: the SAME frozen-tangent chord/modified-Newton
    scheme as solve_nonlinear_koiter_newton()'s own corrector: each
    iteration solves [[K_ff,-f_0],[-f_0^T,0]] @ [du; dlambda] = [r; 0]
    against the CURRENT full residual r = lambda_c*F_total - F_int(u_c),
    reusing the predictor's factorization.

    CORRECTOR, m=1 branch (CHANGED 2026-09-25): Newton on the REAL
    residual, bordered by the REAL load only, [[K_T(u_c),-f_0],
    [-f_0^T,0]] @ [du; dlambda] = [r; 0], with K_T re-assembled at the
    current iterate every iteration. This follows Liang et al. 2014 Sec. 3:
    Koiter's reduced model (including the fictitious perturbation load f_1)
    is only the PREDICTOR, and the corrector is a "Newton arc-length" scheme
    with "general Newton steps of the full FE system". The previous version
    reused the predictor's (nf+2) matrix lu2 and dropped the f_1 multiplier.
    Its f_1 row (f_1 . du = 0) froze the a1 amplitude, so the residual along
    f_1 was absorbed by the dropped multiplier and never removed (for nf=2,
    du was forced to 0). It only "converged" when the predictor alone met
    tol, and otherwise failed after max_shrink_iters retries with |R| exactly
    constant (NonLin-HyROM Case 2;
    tests/test_koiter_newton_generic.py::
    test_m1_corrector_converges_off_the_predictor). The frozen K_ff(u_n)
    is also near-singular exactly when this branch is active, which is why
    the tangent is re-assembled. A failed correction still shrinks a0,
    re-solves a1 from it (preserving branch continuity), and retries.

    SCOPE (read before trusting this beyond a single simple
    bifurcation): (1) at most ONE extra direction (m<=1) is added per
    step -- two or more simultaneously near-critical modes (deeply
    closely-spaced buckling, not just a single isolated bifurcation)
    would need a further generalization of the branch-selection step
    this implementation does not attempt. (2) branch selection at a
    genuine bifurcation point (more than one real root to the row-1
    cubic) uses simple continuity-to-the-previous-step, not a
    stability/energy criterion -- adequate for tracing a single
    physically continuous path through a simple bifurcation, not for
    automatically discovering or choosing between multiple stable
    post-buckling branches. (3) no hysteresis on the detection check
    (see above). These are documented simplifications of an already
    substantial generalization, not silent gaps -- see
    docs/general_purpose_extensions_roadmap.md Section 11 for the
    validation this rests on and what remains open.

    u0/lam0 (restart support): a converged (displacement, load-factor)
    state to re-expand from, instead of the undeformed state -- directly
    implements the paper's own remedy for a step this driver can't resolve
    (Section 4: "we stop the current expansion step and reconstruct the
    reduced-order model at the obtained equilibrium state to initiate a
    new expansion step"). A caller that catches this function's own
    RuntimeError can restart it with u0/lam0 set to the LAST state it
    itself successfully committed (U_hist[i], load_factors[i] for the
    last converged i), chaining several calls into one longer trace one
    re-expansion at a time -- exactly the paper's own multi-expansion-step
    procedure, e.g. Section 5.2.1's "three numerical steps". Both default
    to None/0.0, which reproduces this function's PRE-EXISTING behavior
    (start from the undeformed state) exactly -- every existing caller is
    unaffected. k_scale and the mode-interaction state (a1_prev) are
    always recomputed fresh at the first step of THIS call regardless of
    u0/lam0, never carried in from a hypothetical prior call -- correct
    for a genuine restart (a fresh reference stiffness scale and no
    assumed mode-interaction history at the new expansion point).

    Returns (load_factors, U_hist), same convention as
    solve_nonlinear_koiter_newton()."""
    free = list(fesystem.free_dofs)
    F_total = fesystem.F.copy()
    F_ref = F_total[free]
    n_dof = fesystem.n_dof
    nf = len(free)

    u_n = u0.copy() if u0 is not None else np.zeros(n_dof)
    lam_n = lam0
    load_factors = np.zeros(n_steps + 1)
    U_hist = np.zeros((n_steps + 1, n_dof))
    load_factors[0] = lam_n
    U_hist[0] = u_n
    fesystem.commit_all_states(u_n, mat, **kwargs)

    k_scale = None
    lam_ref = None
    a1_prev = 0.0

    for step in range(1, n_steps + 1):
        ref = max(np.linalg.norm(F_ref), 1e-30)

        K_T = fesystem.assemble_tangent_stiffness(u_n, mat, **kwargs)
        K_ff = K_T[np.ix_(free, free)]

        if k_scale is None:
            k_scale = np.trace(K_ff) / max(nf, 1)

        use_extra = False
        phi_crit = None
        if enable_mode_interaction:
            if lam_ref is None:
                # Detection scale = the UNDEFORMED structure's own smallest
                # tangent eigenvalue (see docstring "AUTOMATIC MODE
                # DETECTION"). Always taken at u = 0, so a restart (u0) at an
                # already-near-critical state still detects it. Falls back
                # to trace/n only if K_ff(0) is itself (near-)singular.
                K0_ff = fesystem.assemble_tangent_stiffness(np.zeros(n_dof), mat, **kwargs)[np.ix_(free, free)]
                try:
                    lam_ref = abs(eigh(K0_ff, eigvals_only=True, subset_by_index=[0, 0])[0])
                except Exception:
                    lam_ref = abs(np.linalg.eigvalsh(K0_ff)[0])
                if lam_ref < 1e-12 * max(abs(k_scale), 1e-30):
                    lam_ref = abs(k_scale)
            try:
                eigvals, eigvecs = eigh(K_ff, subset_by_index=[0, 1])
            except Exception:
                eigvals_all, eigvecs_all = eigh(K_ff)
                eigvals, eigvecs = eigvals_all[:2], eigvecs_all[:, :2]
            lam_min = eigvals[0]
            if abs(lam_min) < mode_detect_rel_tol * max(lam_ref, 1e-30):
                phi_crit = eigvecs[:, 0]
                phi_crit = phi_crit / max(np.linalg.norm(phi_crit), 1e-300)
                # K_ff @ phi_crit is the extra direction's own sub-load
                # (see the function docstring's "THE EXTRA PERTURBATION
                # LOAD" section) -- only actually activate the second
                # direction if that load is numerically well-conditioned;
                # a near-zero K_ff @ phi_crit would make the second
                # border row singular, so this step falls back to the
                # plain m=0 predictor instead (same as never having
                # detected a critical mode at all).
                if np.linalg.norm(K_ff @ phi_crit) >= 1e-300 * max(ref, 1.0):
                    use_extra = True

        F0 = fesystem.assemble_internal_force(u_n, mat, **kwargs)

        # =================================================================
        # m=0 FALLBACK -- identical formulas to solve_nonlinear_koiter_
        # newton(), see that function and this one's own docstring.
        # =================================================================
        if not use_extra:
            A_aug = np.zeros((nf + 1, nf + 1))
            A_aug[:nf, :nf] = K_ff
            A_aug[:nf, nf] = -F_ref
            A_aug[nf, :nf] = -F_ref
            lu_and_piv = lu_factor(A_aug)

            rhs1 = np.zeros(nf + 1); rhs1[nf] = -1.0
            x1 = lu_solve(lu_and_piv, rhs1)
            u1, Lbar1 = x1[:nf], x1[nf]

            h_scale = _fd_probe_length(u_n[free], delta_L, fd_rel)
            Q11, C111 = _directional_QC(fesystem, mat, u_n, free, u1, K_ff,
                                         h_scale, F0[free], **kwargs)

            rhs2 = np.zeros(nf + 1); rhs2[:nf] = -Q11
            x2 = lu_solve(lu_and_piv, rhs2)
            u11, Qbar11 = x2[:nf], x2[nf]
            Cbar1111 = u1 @ C111 - 2.0 * (u11 @ (K_ff @ u11))

            def _predict(a):
                lam_p = lam_n + Lbar1 * a + Qbar11 * a ** 2 + Cbar1111 * a ** 3
                u_p = u_n.copy()
                u_p[free] = u_n[free] + a * u1 + a ** 2 * u11
                return u_p, lam_p

            def _residual_ratio(u_p, lam_p):
                F_int_p = fesystem.assemble_internal_force(u_p, mat, **kwargs)
                R = lam_p * F_total - F_int_p
                return np.linalg.norm(R[free]) / ref

            u1_norm = np.linalg.norm(u1)
            a0 = delta_L / max(u1_norm, 1e-300)
            u_ok, lam_ok = _predict(a0)
            ratio0 = _residual_ratio(u_ok, lam_ok)
            a_ok = a0
            if ratio0 <= predictor_tol:
                for _ in range(max_growth_iters):
                    a_try = a_ok * growth_factor
                    u_try, lam_try = _predict(a_try)
                    if _residual_ratio(u_try, lam_try) <= predictor_tol:
                        a_ok, u_ok, lam_ok = a_try, u_try, lam_try
                    else:
                        break
            else:
                found = False
                a_try = a0
                for _ in range(max_shrink_iters):
                    a_try *= shrink_factor
                    u_try, lam_try = _predict(a_try)
                    if _residual_ratio(u_try, lam_try) <= predictor_tol:
                        a_ok, u_ok, lam_ok, found = a_try, u_try, lam_try, True
                        break
                if not found:
                    raise RuntimeError(
                        f"solve_nonlinear_koiter_newton_generic: could not shrink "
                        f"the m=0 predictor to satisfy predictor_tol={predictor_tol:.3g} "
                        f"at step {step} after {max_shrink_iters} halvings.")

            if verbose:
                print(f"  step {step:3d}  [m=0]  a={a_ok:.4e}  "
                      f"|dlambda|={abs(lam_ok - lam_n):.4e}")

            def _run_corrector(a_seed):
                u_c, lam_c = _predict(a_seed)
                predictor_delta_u = np.zeros(n_dof)
                predictor_delta_u[free] = u_c[free] - u_n[free]
                fesystem.update_iter_states(u_c, predictor_delta_u, mat, **kwargs)
                for it in range(max_iter):
                    F_int = fesystem.assemble_internal_force(u_c, mat, **kwargs)
                    R = lam_c * F_total - F_int
                    Rn = np.linalg.norm(R[free])
                    if verbose:
                        print(f"    it {it:2d}  lambda={lam_c:.6f}  |R|={Rn:.3e}")
                    if Rn < tol * ref or Rn < tol:
                        return True, u_c, lam_c, Rn
                    rhs3 = np.zeros(nf + 1)
                    rhs3[:nf] = R[free]
                    x3 = lu_solve(lu_and_piv, rhs3)
                    du_tilde, dphi_tilde = x3[:nf], x3[nf]
                    delta_u = np.zeros(n_dof)
                    delta_u[free] = du_tilde
                    u_c[free] += du_tilde
                    lam_c += dphi_tilde
                    fesystem.update_iter_states(u_c, delta_u, mat, **kwargs)
                return False, u_c, lam_c, Rn

            a_corr = a_ok

        # =================================================================
        # m=1 -- two-direction bordered system
        # =================================================================
        else:
            f0 = F_ref
            f1_raw = K_ff @ phi_crit
            f1_norm = np.linalg.norm(f1_raw)
            if f1_norm < 1e-300 * max(ref, 1.0):
                # K_ff @ phi_crit is (numerically) zero -- phi_crit pairs
                # with a load that doesn't excite it at all in this
                # direction; bordering on it would be singular. Fall back
                # to the plain m=0 predictor/corrector for this step only
                # (same code path as "not use_extra" above).
                use_extra = False

            if use_extra:
                f1 = f1_raw * (ref / f1_norm)

                A_aug2 = np.zeros((nf + 2, nf + 2))
                A_aug2[:nf, :nf] = K_ff
                A_aug2[:nf, nf] = -f0
                A_aug2[:nf, nf + 1] = -f1
                A_aug2[nf, :nf] = -f0
                A_aug2[nf + 1, :nf] = -f1
                lu2 = lu_factor(A_aug2)

                u_dir = {}
                Lbar = np.zeros((2, 2))
                for i in range(2):
                    rhs = np.zeros(nf + 2); rhs[nf + i] = -1.0
                    x = lu_solve(lu2, rhs)
                    u_dir[i] = x[:nf]
                    Lbar[:, i] = x[nf:nf + 2]

                qc = _two_direction_QC(fesystem, mat, u_n, free, u_dir[0], u_dir[1],
                                        K_ff, fd_rel, F0[free], step_scale=delta_L, **kwargs)
                if verbose and qc['Q01_consistency'] > 1e-2:
                    print(f"  step {step:3d}  [m=1]  WARNING: Q01 cross-check "
                          f"disagreement={qc['Q01_consistency']:.3e} (consider a "
                          f"larger fd_rel)")

                Q_lookup = {(0, 0): qc['Q00'], (0, 1): qc['Q01'], (1, 1): qc['Q11']}
                u_quad = {}
                Qbar = {}
                for pair in [(0, 0), (0, 1), (1, 1)]:
                    rhs = np.zeros(nf + 2)
                    rhs[:nf] = -Q_lookup[pair]
                    x = lu_solve(lu2, rhs)
                    u_quad[pair] = x[:nf]
                    Qbar[pair] = x[nf:nf + 2]

                u_dir_map = {0: u_dir[0], 1: u_dir[1]}
                C_lin_map = {(0, 0, 0): qc['C000'], (0, 0, 1): qc['C001'],
                             (0, 1, 1): qc['C011'], (1, 1, 1): qc['C111']}
                triples = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
                Cbar = {(p, t): _generic_cbar_term(p, t[0], t[1], t[2],
                                                    u_dir_map, u_quad, C_lin_map, K_ff)
                        for p in (0, 1) for t in triples}

                def _reduced_value(p, a0_, a1_):
                    return (Lbar[p, 0] * a0_ + Lbar[p, 1] * a1_
                            + Qbar[(0, 0)][p] * a0_ ** 2 + 2 * Qbar[(0, 1)][p] * a0_ * a1_
                            + Qbar[(1, 1)][p] * a1_ ** 2
                            + Cbar[(p, (0, 0, 0))] * a0_ ** 3
                            + 3 * Cbar[(p, (0, 0, 1))] * a0_ ** 2 * a1_
                            + 3 * Cbar[(p, (0, 1, 1))] * a0_ * a1_ ** 2
                            + Cbar[(p, (1, 1, 1))] * a1_ ** 3)

                def _solve_a1(a0_, a1_seed):
                    c3 = Cbar[(1, (1, 1, 1))]
                    c2 = 3.0 * Cbar[(1, (0, 1, 1))] * a0_
                    c1 = (Lbar[1, 1] + 2.0 * Qbar[(0, 1)][1] * a0_
                          + 3.0 * Cbar[(1, (0, 0, 1))] * a0_ ** 2)
                    c0 = (Lbar[1, 0] * a0_ + Qbar[(0, 0)][1] * a0_ ** 2
                          + Cbar[(1, (0, 0, 0))] * a0_ ** 3)
                    if abs(c3) < 1e-300 and abs(c2) < 1e-300:
                        if abs(c1) < 1e-300:
                            return 0.0
                        return -c0 / c1
                    roots = np.roots([c3, c2, c1, c0])
                    real_roots = [r.real for r in roots
                                  if abs(r.imag) < 1e-6 * max(abs(r.real), 1.0)]
                    if not real_roots:
                        return 0.0
                    return min(real_roots, key=lambda r: abs(r - a1_seed))

                def _predict2(a0_, a1_seed):
                    a1_ = _solve_a1(a0_, a1_seed)
                    lam_p = lam_n + _reduced_value(0, a0_, a1_)
                    u_p = u_n.copy()
                    u_p[free] = (u_n[free] + a0_ * u_dir[0] + a1_ * u_dir[1]
                                 + a0_ ** 2 * u_quad[(0, 0)]
                                 + 2 * a0_ * a1_ * u_quad[(0, 1)]
                                 + a1_ ** 2 * u_quad[(1, 1)])
                    return u_p, lam_p, a1_

                def _residual_ratio2(u_p, lam_p):
                    F_int_p = fesystem.assemble_internal_force(u_p, mat, **kwargs)
                    R = lam_p * F_total - F_int_p
                    return np.linalg.norm(R[free]) / ref

                u0_norm = np.linalg.norm(u_dir[0])
                a0_try0 = delta_L / max(u0_norm, 1e-300)
                u_ok, lam_ok, a1_ok = _predict2(a0_try0, a1_prev)
                ratio0 = _residual_ratio2(u_ok, lam_ok)
                a0_ok = a0_try0
                if ratio0 <= predictor_tol:
                    for _ in range(max_growth_iters):
                        a0_try = a0_ok * growth_factor
                        u_try, lam_try, a1_try = _predict2(a0_try, a1_ok)
                        if _residual_ratio2(u_try, lam_try) <= predictor_tol:
                            a0_ok, u_ok, lam_ok, a1_ok = a0_try, u_try, lam_try, a1_try
                        else:
                            break
                else:
                    found = False
                    a0_try = a0_try0
                    for _ in range(max_shrink_iters):
                        a0_try *= shrink_factor
                        u_try, lam_try, a1_try = _predict2(a0_try, a1_ok)
                        if _residual_ratio2(u_try, lam_try) <= predictor_tol:
                            a0_ok, u_ok, lam_ok, a1_ok, found = a0_try, u_try, lam_try, a1_try, True
                            break
                    if not found:
                        raise RuntimeError(
                            f"solve_nonlinear_koiter_newton_generic: could not "
                            f"shrink the m=1 predictor to satisfy "
                            f"predictor_tol={predictor_tol:.3g} at step {step} "
                            f"after {max_shrink_iters} halvings.")

                if verbose:
                    print(f"  step {step:3d}  [m=1]  a0={a0_ok:.4e}  a1={a1_ok:.4e}  "
                          f"|dlambda|={abs(lam_ok - lam_n):.4e}")

                def _run_corrector2(a0_seed, a1_seed):
                    # Newton on the REAL residual, bordered by the real load only
                    # (Liang et al. 2014 Sec. 3: "Newton arc-length" corrector with
                    # "general Newton steps of the full FE system"). NOT lu2: its
                    # f_1 row froze the a1 amplitude (f_1 . du = 0) and the dropped
                    # f_1 multiplier absorbed the residual along f_1, so the old
                    # corrector could not move off the predictor (2026-09-25).
                    # The tangent is re-assembled every iteration because K_ff(u_n)
                    # is near-singular exactly when this branch is active.
                    u_c, lam_c, a1_used = _predict2(a0_seed, a1_seed)
                    predictor_delta_u = np.zeros(n_dof)
                    predictor_delta_u[free] = u_c[free] - u_n[free]
                    fesystem.update_iter_states(u_c, predictor_delta_u, mat, **kwargs)
                    Rn = np.inf
                    for it in range(max_iter):
                        F_int = fesystem.assemble_internal_force(u_c, mat, **kwargs)
                        R = lam_c * F_total - F_int
                        Rn = np.linalg.norm(R[free])
                        if verbose:
                            print(f"    it {it:2d}  lambda={lam_c:.6f}  |R|={Rn:.3e}")
                        if not np.isfinite(Rn):
                            return False, u_c, lam_c, Rn, a1_used
                        if Rn < tol * ref or Rn < tol:
                            return True, u_c, lam_c, Rn, a1_used
                        K_c = fesystem.assemble_tangent_stiffness(u_c, mat, **kwargs)[np.ix_(free, free)]
                        A_c = np.zeros((nf + 1, nf + 1))
                        A_c[:nf, :nf] = K_c
                        A_c[:nf, nf] = -f0
                        A_c[nf, :nf] = -f0
                        rhs = np.zeros(nf + 1)
                        rhs[:nf] = R[free]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            x = lu_solve(lu_factor(A_c), rhs)
                        if not np.all(np.isfinite(x)):
                            return False, u_c, lam_c, Rn, a1_used
                        du_tilde, dphi_tilde = x[:nf], x[nf]
                        delta_u = np.zeros(n_dof)
                        delta_u[free] = du_tilde
                        u_c[free] += du_tilde
                        lam_c += dphi_tilde
                        fesystem.update_iter_states(u_c, delta_u, mat, **kwargs)
                    return False, u_c, lam_c, Rn, a1_used

                a0_corr = a0_ok

        # =================================================================
        # shared retry/commit bookkeeping for both branches
        # =================================================================
        converged = False
        iter_state_at_step_start = copy.deepcopy(fesystem.iter_state)
        a1_used_final = a1_prev
        for _retry in range(max_shrink_iters):
            if use_extra:
                converged, u, lam, Rn, a1_used_final = _run_corrector2(a0_corr, a1_ok)
            else:
                converged, u, lam, Rn = _run_corrector(a_corr)
            if converged:
                break
            fesystem.iter_state = copy.deepcopy(iter_state_at_step_start)
            if verbose:
                print(f"  step {step:3d}: corrector did not converge "
                      f"(|R|={Rn:.3e}) -- shrinking and retrying")
            if use_extra:
                a0_corr *= shrink_factor
            else:
                a_corr *= shrink_factor

        if not converged:
            raise RuntimeError(
                f"solve_nonlinear_koiter_newton_generic: chord corrector failed "
                f"to converge at step {step} even after {max_shrink_iters} "
                f"predictor shrinks (last |R|={Rn:.3e}).")

        a1_prev = a1_used_final if use_extra else 0.0
        u_n, lam_n = u, lam
        load_factors[step] = lam_n
        U_hist[step] = u_n
        fesystem.commit_all_states(u_n, mat, **kwargs)

    return load_factors, U_hist


def solve_nonlinear_static_koiter_newton(fesystem, mat, tol=1e-8, max_iter=30,
                                          max_expansions=20, predictor_tol=0.1,
                                          fd_rel=1e-2, shrink_factor=0.5,
                                          max_shrink_iters=10, verbose=False,
                                          du_tol=None, energy_tol=None,
                                          line_search=True, full_tangent_fallback=True,
                                          **kwargs):
    """Module 22: Koiter-Newton solve to the SINGLE prescribed target load
    F_ext = 1.0*fesystem.F (same load-scale convention as
    solve_nonlinear_static()) -- NOT a path tracer like
    solve_nonlinear_koiter_newton() above. Use this one when what you
    actually want is "the equilibrium at this one known load level, as
    cheaply as possible" (e.g. generating nonlinear-static training data
    for a dynamic ROM's polynomial modal-stiffness fit -- Yang et al.
    2019's own Fig. 1 flowchart shows exactly this: "determination of
    nonlinear static test cases" -> "construction of static reduced-
    order model" -> "calculating static tests", one static-ROM expansion
    PER prescribed test load, not one continuous path); use
    solve_nonlinear_koiter_newton() instead when you want to trace an
    entire equilibrium path with no specific target in mind.

    WHY THIS IS A SEPARATE FUNCTION, NOT A THIN WRAPPER AROUND
    solve_nonlinear_koiter_newton(): tried the wrapper first --
    run the continuation driver for a fixed handful of steps, then
    interpolate/polish back to lambda=1 -- and it failed on a real
    Shell4MITCCorotational model in two ways: (1) with nothing telling
    it to stop, the continuation driver's OWN adaptive growth happily
    ran to lambda=14.3 for a target of lambda=1.0 (a >14x overshoot),
    making it SLOWER than plain solve_nonlinear_static() despite the
    supposedly-cheaper predictor, because all that extra growth was
    pure waste; (2) for a larger target load, the same unbounded growth
    drove the shell element into floating-point overflow (NaN/Inf in
    its own internal-force computation) before the wrapper ever got a
    chance to interpolate back down -- an outright crash, not just an
    inefficiency. Both failures trace to the same root cause: the
    continuation driver's step-growth criterion ("does the cubic model
    still predict the true residual well?") has no idea a caller wants
    to stop at a SPECIFIC load level -- it only knows to keep growing
    while the model still looks locally trustworthy, which for a
    genuinely-cubic force law (exactly true for e.g. a Green-Lagrange
    truss bar, approximately true for many hardening shells over a wide
    range) can be "forever."

    THE FIX: aim, don't grow. At each expansion, after building the
    SAME reduced cubic predictor as solve_nonlinear_koiter_newton()
    (u1/Lbar1, u11/Qbar11, Cbar1111 -- see that function's own docstring
    for the full Koiter-asymptotic derivation, identical here), solve
    the reduced cubic polynomial DIRECTLY for the perturbation
    parameter a_target that lands exactly at the REMAINING gap to the
    target load (mu_target = 1.0 - lambda_current):

        Cbar1111*a^3 + Qbar11*a^2 + Lbar1*a - mu_target = 0

    (`numpy.roots`, picking the REAL root closest to the linear estimate
    mu_target/Lbar1 -- the natural "closest to what plain Newton would
    guess" tie-break when more than one real root exists). If the
    resulting predicted point's true residual satisfies predictor_tol,
    this expansion reaches the target in one shot -- no further growth
    is ever attempted, so there is nothing left to overshoot with. If
    it does NOT satisfy predictor_tol (the target is "too far" from this
    expansion point for the cubic model to trust), shrink a TOWARD ZERO
    from a_target (never past it, and never in a different direction)
    until it does, accept that PARTIAL step, and re-expand from the new
    corrected point -- exactly the paper's own multi-step behavior
    (Fig. 3's star markers: each star is a `predictor_tol`-limited
    partial step toward the eventual target, not an attempt to overshoot
    it), just applied toward a caller-known target instead of an
    open-ended path.

    CORRECTOR: because the target load for this expansion (lambda_used,
    either the true target on the last expansion or an intermediate
    value on an earlier one) is EXTERNALLY KNOWN once the predictor
    step is chosen -- unlike solve_nonlinear_koiter_newton()'s
    continuation corrector, which must solve for lambda itself along
    with u -- there is no need for Eq. 21's bordered/remainder trick at
    all. This reduces to ORDINARY frozen-tangent (chord) Newton at the
    FIXED load F_ext = lambda_used*fesystem.F, reusing the plain
    (non-bordered) K_T(u_n) factorization -- simpler, and a genuine
    simplification of Eq. 21 for the case where phi is already known,
    not a deviation from the paper's method.

    Raises RuntimeError if no trustworthy step can be found within
    max_shrink_iters at some expansion, or if max_expansions is
    exhausted without reaching lambda=1.0 -- in both cases, try a
    looser predictor_tol or more max_expansions/max_shrink_iters.

    du_tol/energy_tol/line_search (Wave 1 items 9/10 -- see this
    module's own docstring "Wave 1" section): applied to THIS
    corrector specifically, because -- unlike solve_nonlinear_koiter_
    newton()'s continuation corrector below, deliberately left alone
    -- it reduces to "ORDINARY frozen-tangent (chord) Newton" (see
    "CORRECTOR" above), the same structural shape as this module's
    other plain-Newton loops, so the shared helpers apply directly.
    This composes with, rather than replaces, the expansion loop's OWN
    existing globalization (shrinking the perturbation parameter `a`
    toward zero when a step isn't trustworthy) -- line search is tried
    INSIDE one fixed-lambda corrector attempt, before that attempt is
    reported as failed up to the shrink-`a` retry above it, the same
    "escalate through layers, cheapest first" structure solve_
    nonlinear_transient() itself uses (trust region, then line search).
    All three default to their original-behavior values -- an existing
    caller passing none of them gets IDENTICAL results to before this
    wave.

    full_tangent_fallback (default True): a THIRD, more expensive
    corrector rung, tried only if BOTH "plain" and "line_search" chord
    Newton (both of which reuse ONE tangent factorization, lu_plain,
    frozen at u_n -- see CORRECTOR above) fail to converge. Added
    2026-09-13 after this exact chord-Newton corrector was found to
    DIVERGE GEOMETRICALLY (residual growing ~3.3x every iteration,
    confirmed by an iteration-by-iteration trace, not just observed as a
    final crash) when combined with `Shell4Director` -- a new,
    genuinely-nonlinear-in-bending shell element (docs/shells.md Section
    1.3) whose tangent stiffness actually moves meaningfully away from
    its own u=0 value at ordinary training-data deflections. This is NOT
    a defect in the frozen-tangent design itself: it was, and remains,
    an exact, efficient corrector for any case where the true tangent
    stays close to K_T(u_n) across the predictor's step (verified true
    for `Shell4MITCCorotational` on this same class of problem, where the
    D5 architectural limitation -- see shells.md Section 1.2 -- means the
    internal force is, for pure-bending load patterns, ALMOST EXACTLY
    LINEAR in u, so freezing the tangent is nearly free of error). It is
    a correctness gap for any element/load combination where that
    implicit assumption doesn't hold, which this driver had never been
    exercised against before. This fallback reassembles and refactors
    the FULL tangent stiffness EVERY iteration (mirroring solve_
    nonlinear_static()'s own inner loop, confirmed independently to
    converge cleanly -- 3 iterations, residual to ~2e-6 -- on the exact
    load/element combination that broke the frozen-tangent corrector),
    with line search enabled unconditionally for extra robustness on
    this already-expensive rung. STRICTLY ADDITIVE: only ever invoked
    after both existing modes have already failed, so every previously-
    passing case (the truss benchmarks in tests/test_static_koiter_
    newton.py, and Shell4MITCCorotational's own validated training
    samples) reaches convergence in "plain" mode exactly as before and
    never reaches this code path -- confirmed by the full existing
    Koiter-Newton test suite passing unchanged. A companion fix in this
    same change, UNCONDITIONAL (not gated by full_tangent_fallback):
    BOTH chord-Newton modes ("plain"/"line_search") now detect a
    non-finite residual immediately after each trial step and return
    converged=False instead of feeding it into the next iteration's
    lu_solve() -- which previously raised an uncaught ValueError ("array
    must not contain infs or NaNs"), aborting the ENTIRE solve before
    even this function's OWN existing shrink-a-toward-zero-and-re-expand
    recovery loop (the mechanism "CORRECTOR"/"THE FIX" above already
    describes for an untrustworthy predictor step) ever got a chance to
    engage. This alone lets a genuinely bad predictor step recover the
    way the expansion loop was always meant to; full_tangent_fallback is
    the separate, additional safety net for cases (like Shell4Director
    above) where the predictor step itself is fine but the frozen-tangent
    corrector diverging is the actual problem, which shrinking `a` cannot
    fix (a smaller step from the SAME starting tangent hits the SAME
    instability). Set full_tangent_fallback=False only to skip paying
    for this specific rung (e.g. to fail fast during development); the
    non-finite-residual fix above still applies either way.

    Returns u (shape (n_dof,), the full displacement vector at
    F_ext = 1.0*fesystem.F) -- NOT a (load_factors, U_hist) path history
    like every other driver in this module, since this one solves a
    single equilibrium problem, not a path."""
    free = list(fesystem.free_dofs)
    F_total = fesystem.F.copy()
    F_ref = F_total[free]
    n_dof = fesystem.n_dof
    nf = len(free)
    ref = max(np.linalg.norm(F_ref), 1e-30)

    u_n = np.zeros(n_dof)
    lam_n = 0.0

    for expansion in range(max_expansions):
        K_T = fesystem.assemble_tangent_stiffness(u_n, mat, **kwargs)
        K_ff = K_T[np.ix_(free, free)]
        A_aug = np.zeros((nf + 1, nf + 1))
        A_aug[:nf, :nf] = K_ff
        A_aug[:nf, nf] = -F_ref
        A_aug[nf, :nf] = -F_ref
        lu_aug = lu_factor(A_aug)
        lu_plain = lu_factor(K_ff)

        rhs1 = np.zeros(nf + 1)
        rhs1[nf] = -1.0
        x1 = lu_solve(lu_aug, rhs1)
        u1, Lbar1 = x1[:nf], x1[nf]

        u1_norm = np.linalg.norm(u1)
        # probe scale = linear estimate of the displacement still needed
        # to reach the target (see _fd_probe_length)
        a_lin_step = (1.0 - lam_n) / Lbar1 if abs(Lbar1) > 1e-300 else (1.0 - lam_n)
        h = _fd_probe_length(u_n[free], abs(a_lin_step) * u1_norm, fd_rel) / max(u1_norm, 1e-300)
        w_plus = u_n.copy(); w_plus[free] += h * u1
        w_minus = u_n.copy(); w_minus[free] -= h * u1
        F0 = fesystem.assemble_internal_force(u_n, mat, **kwargs)
        F_plus = fesystem.assemble_internal_force(w_plus, mat, **kwargs)
        F_minus = fesystem.assemble_internal_force(w_minus, mat, **kwargs)
        g_plus = F_plus[free] - F0[free]
        g_minus = F_minus[free] - F0[free]
        Q11 = (g_plus + g_minus) / (2.0 * h ** 2)
        C111 = (g_plus - g_minus - 2.0 * h * (K_ff @ u1)) / (2.0 * h ** 3)

        rhs2 = np.zeros(nf + 1)
        rhs2[:nf] = -Q11
        x2 = lu_solve(lu_aug, rhs2)
        u11, Qbar11 = x2[:nf], x2[nf]

        Cbar1111 = u1 @ C111 - 2.0 * (u11 @ (K_ff @ u11))

        def _predict(a):
            lam_p = lam_n + Lbar1 * a + Qbar11 * a ** 2 + Cbar1111 * a ** 3
            u_p = u_n.copy()
            u_p[free] = u_n[free] + a * u1 + a ** 2 * u11
            return u_p, lam_p

        def _residual_ratio(u_p, lam_p):
            F_int_p = fesystem.assemble_internal_force(u_p, mat, **kwargs)
            R = lam_p * F_total - F_int_p
            return np.linalg.norm(R[free]) / ref

        def _fixed_lambda_newton(u_seed, lam_fixed):
            u_c = u_seed.copy()
            F_ext = lam_fixed * F_total
            ref_ext = max(np.linalg.norm(F_ext[free]), 1e-30)

            predictor_delta_u = np.zeros(n_dof)
            predictor_delta_u[free] = u_c[free] - u_n[free]
            fesystem.update_iter_states(u_c, predictor_delta_u, mat, **kwargs)
            iter_state_after_predictor_delta = copy.deepcopy(fesystem.iter_state)

            def _residual(u_trial_free):
                u_c[free] = u_trial_free
                F_int_trial = fesystem.assemble_internal_force(u_c, mat, **kwargs)
                return F_ext[free] - F_int_trial[free], F_int_trial

            def _newton(mode):
                fesystem.iter_state = copy.deepcopy(iter_state_after_predictor_delta)
                u_free = u_seed[free].copy()
                R_free, F_int = _residual(u_free)
                Rn = np.linalg.norm(R_free)
                if not np.all(np.isfinite(R_free)):
                    return False, u_free, np.inf
                converged = False
                for it in range(max_iter):
                    if verbose:
                        tag = {"line_search": " [line search]",
                               "full_tangent": " [full tangent]"}.get(mode, "")
                        print(f"    it {it:2d}{tag}  |R|={Rn:.3e}")
                    residual_ok = Rn < tol * ref_ext or Rn < tol
                    if residual_ok and du_tol is None and energy_tol is None:
                        converged = True
                        break
                    if mode == "full_tangent":
                        # Reassemble + refactor EVERY iteration -- the
                        # expensive fallback rung for an element/load
                        # combination where the frozen tangent below is
                        # not a convergent chord direction (see this
                        # function's own docstring, "full_tangent_
                        # fallback": found necessary for Shell4Director,
                        # whose true tangent moves enough from K_T(u_n)
                        # to make the frozen factorization diverge).
                        K_T_now = fesystem.assemble_tangent_stiffness(u_c, mat, **kwargs)
                        lu_now = lu_factor(K_T_now[np.ix_(free, free)])
                        du = lu_solve(lu_now, R_free)
                    else:
                        # frozen tangent (lu_plain, factored ONCE per
                        # expansion above) -- still a valid backtracking
                        # direction for line search (see _armijo_line_
                        # search_step's own docstring: only needs
                        # J@du=-R at THIS trial point, not J being exact).
                        du = lu_solve(lu_plain, R_free)
                    if residual_ok:
                        du_ok, energy_ok = _extra_convergence_ok(
                            R_free, du, u_free, F_ext[free], F_int[free],
                            du_tol, energy_tol)
                        if du_ok and energy_ok:
                            converged = True
                            break

                    if mode in ("line_search", "full_tangent"):
                        u_new, R_new, F_int_new, Rn_new, _alpha = _armijo_line_search_step(
                            _residual, u_free, du, Rn, tol, ref_ext, verbose=verbose,
                            verbose_prefix=f"      it {it:2d}  ")
                    else:
                        u_new = u_free + du
                        R_new, F_int_new = _residual(u_new)
                        Rn_new = np.linalg.norm(R_new)

                    if not np.all(np.isfinite(R_new)):
                        # Diverged (e.g. a rotation-based element's
                        # exponential-map state has left its valid
                        # range) -- report "not converged" instead of
                        # feeding a non-finite residual into the NEXT
                        # iteration's lu_solve(), which raises rather
                        # than letting the caller's escalation ladder
                        # (plain -> line_search -> full_tangent, then the
                        # expansion loop's own shrink-a-toward-zero
                        # retry) get a chance to recover.
                        return False, u_new, np.inf

                    delta_u = np.zeros(n_dof)
                    delta_u[free] = u_new - u_free
                    fesystem.update_iter_states(u_c, delta_u, mat, **kwargs)
                    u_free, R_free, F_int, Rn = u_new, R_new, F_int_new, Rn_new
                return converged, u_free, Rn

            ok, u_free, Rn = _newton("plain")
            if not ok and line_search:
                if verbose:
                    print(f"    plain chord-Newton did not converge in "
                          f"{max_iter} iterations (|R|={Rn:.3e}) -- retrying "
                          f"with line search")
                ok, u_free, Rn = _newton("line_search")
            if not ok and full_tangent_fallback:
                if verbose:
                    print(f"    frozen-tangent Newton (plain + line search) "
                          f"did not converge (|R|={Rn:.3e}) -- retrying with "
                          f"a reassembled tangent every iteration")
                ok, u_free, Rn = _newton("full_tangent")
            u_c[free] = u_free
            return ok, u_c, Rn

        mu_target = 1.0 - lam_n
        coeffs = [Cbar1111, Qbar11, Lbar1, -mu_target]
        roots = np.roots(coeffs)
        tol_imag = 1e-8 * max(np.max(np.abs(roots.real)), 1.0)
        real_roots = roots[np.abs(roots.imag) < tol_imag].real
        a_lin = mu_target / Lbar1 if abs(Lbar1) > 1e-300 else mu_target
        # Only roots on the SAME side as the linear estimate move the load
        # toward the target; an opposite-sign root means the cubic predicts
        # a limit point before the target (or its coefficients are
        # untrustworthy -- the 2026-09-23 bug, see _fd_probe_length), and
        # stepping that way drives lambda AWAY from the target. Fall back
        # to the linear estimate; the shrink loop below still enforces
        # predictor_tol on it.
        real_roots = real_roots[np.sign(real_roots) == np.sign(a_lin)]
        a_target = (real_roots[np.argmin(np.abs(real_roots - a_lin))]
                    if len(real_roots) else a_lin)

        a_try = a_target
        u_c, lam_used, reached_target = None, None, False
        # snapshot/restore iter_state (Module 23) around each attempt --
        # same "a rejected trial must not leak into the next, smaller
        # a_try" reasoning as solve_nonlinear_koiter_newton's corrector.
        iter_state_at_expansion_start = copy.deepcopy(fesystem.iter_state)
        for attempt in range(max_shrink_iters + 1):
            u_pred, lam_pred = _predict(a_try)
            ratio = _residual_ratio(u_pred, lam_pred)
            if ratio <= predictor_tol:
                ok, u_c, Rn = _fixed_lambda_newton(u_pred, lam_pred)
                if ok:
                    lam_used = lam_pred
                    reached_target = (a_try == a_target)
                    break
            fesystem.iter_state = copy.deepcopy(iter_state_at_expansion_start)
            if verbose:
                print(f"  expansion {expansion}: a={a_try:.4e} rejected "
                      f"(predictor ratio={ratio:.3e}) -- shrinking")
            a_try *= shrink_factor
        else:
            raise RuntimeError(
                f"solve_nonlinear_static_koiter_newton: could not find a "
                f"trustworthy, convergent step at expansion {expansion} after "
                f"{max_shrink_iters} shrinks toward the target -- try a "
                f"looser predictor_tol.")

        if verbose:
            print(f"  expansion {expansion}: accepted a={a_try:.4e}, "
                  f"lambda={lam_used:.6f}, reached_target={reached_target}")

        u_n, lam_n = u_c, lam_used
        fesystem.commit_all_states(u_n, mat, **kwargs)
        if reached_target:
            return u_n

    raise RuntimeError(
        f"solve_nonlinear_static_koiter_newton: did not reach the target load "
        f"(lambda=1.0) within {max_expansions} expansions (reached "
        f"lambda={lam_n:.6f}) -- try more max_expansions or a looser "
        f"predictor_tol.")


def solve_nonlinear_transient(fesystem, mat, load, T_total, dt, beta=0.25, gamma=0.5,
                               u0=None, v0=None, tol=1e-8, max_iter=30, verbose=False,
                               line_search=True, trust_region=True,
                               backend="scipy", device="cpu", **kwargs):
    """Newmark-beta implicit time integration for a GEOMETRICALLY (or
    materially) NONLINEAR structure -- see the module docstring above
    and docs/nonlinear_transient_dynamics_roadmap.md for the full design
    rationale. At every step, predicts d/v/a exactly as
    FESystem.solve_transient_implicit() does, then runs Newton-Raphson
    to convergence using the CURRENT tangent stiffness
    (assemble_tangent_stiffness(u, mat)) and the CURRENT internal force
    (assemble_internal_force(u, mat)) -- i.e. the effective residual at
    each Newton iteration is

        R(d) = M @ a(d) + C @ v(d) + F_int(d) - F_ext(t+dt)

    with a(d) = a0c*d - d_pred_const and v(d) = v_pred_const + a1c*d -
    a7c*d_pred_const the standard Newmark relations, AFFINE in the
    unknown trial displacement d (d_pred_const/v_pred_const are built
    from the previous step's converged d/v/a and the SAME a0c..a7c
    constants FESystem.solve_transient_implicit() uses), so the exact
    Newton tangent is

        K_eff(d) = K_T(d) + a0c*M + a1c*C

    and is re-factored EVERY Newton iteration (unlike
    solve_transient_implicit's single, one-time factorization) -- this
    is the entire fix. On a LINEAR mat/element (whose
    tangent_stiffness()/internal_force() reduce to the base Element
    class's default `ke @ u`/`ke`, i.e. the element does not override
    them), Newton converges in exactly one correction step and this
    function agrees with solve_transient_implicit() to machine
    precision -- the primary regression test (see
    tests/test_nonlinear_transient.py).

    `load` takes the same interface FESystem.solve_transient_implicit()
    already uses (`force_at(t, n_dof, npn)`, see loads.TimeHistoryLoad),
    so existing load definitions in loads.py need no changes.

    `mat` is forwarded to assemble_internal_force()/
    assemble_tangent_stiffness() unchanged, exactly like
    solve_nonlinear_static(); `tol`/`max_iter`/`verbose` follow that
    same static driver's own convergence convention (relative
    tolerance against a reference force scale), with ONE deliberate
    adaptation: the reference is `max(||F_ext||, ||R_0||)`, not just
    `||F_ext||` -- `R_0` (the residual at the FIRST, pre-correction
    Newton iterate) also captures the inertial (`M@a`) and damping
    (`C@v`) terms that dominate an UNFORCED free-vibration step or an
    early step of a fine quasi-static ramp through a near-zero external
    load, neither of which `solve_nonlinear_static`'s pure load-ramp
    convention ever has to handle. Real, non-degenerate cases are
    unaffected by this widening. After every converged step, calls
    fesystem.commit_all_states(u, mat,
    **kwargs) -- a no-op unless fesystem.init_state() was called first,
    matching every other driver in this module.

    Note on stability: `beta`/`gamma` default to 0.25/0.5 (average
    acceleration), matching solve_transient_implicit()'s own defaults
    for direct comparability -- but unlike the LINEAR case, this
    combination is NOT guaranteed unconditionally stable for a general
    nonlinear system (true of every implicit nonlinear Newmark scheme,
    not a defect specific to this implementation); pick dt with the
    structure's fundamental period in mind, same as any other implicit
    nonlinear time integrator.

    Trust region (added 2026-09-01, `trust_region=True` by default;
    demoted from PRIMARY to an ESCALATION on 2026-09-02, see below) --
    caps each Newton correction's norm to an adaptive radius Delta,
    accepting or rejecting the (possibly capped) step by comparing its
    ACTUAL residual-norm-squared reduction against what the local
    linear model (R + K_eff@du) PREDICTED. Added because line search
    (below) turned out NOT to catch a real failure mode: a plain Newton
    step can satisfy the
    outer convergence tolerance -- so nothing "fails," line search's
    fallback never even triggers -- while landing on a mathematically
    valid but PHYSICALLY SPURIOUS root, one disconnected from the true
    continuous trajectory. Found directly, not hypothesized, on this
    same `Tet10SolidTL` wing-cantilever pipeline: a lightly-damped,
    small (0.5x thickness) initial displacement, with zero external
    forcing, spontaneously grew to >100x its own amplitude over ~30
    time steps -- physically impossible (damping only removes energy),
    yet every single step's Newton loop "converged" to its own
    tolerance. The smoking gun was the per-iteration residual pattern
    at the step where the trajectory first jumps: |R| = 1.8e4 -> 223 ->
    1.1e5 -> 71 -> 1.2e4 -> 1.5 -> 206 -> 5e-4 (converged) -- wildly
    non-monotonic bouncing across many orders of magnitude, the
    signature of an unglobalized full step landing somewhere different
    each try and eventually satisfying tolerance more by luck than by
    approaching the correct nearby equilibrium.

    Delta starts at the FIRST iteration's own full-Newton step norm
    (so, exactly like line search, a well-behaved step is never
    artificially restricted), then adapts by the standard trust-region
    rule (see e.g. Nocedal & Wright, "Numerical Optimization," Algorithm
    4.1, adapted here to ||R||^2 as the equation-solving merit function
    rather than a general objective): accept any step with ratio
    rho = actual_reduction/predicted_reduction > 0.1, growing Delta
    (up to 1e3x the first step's own norm) when rho > 0.75 and the cap
    was active, shrinking it to 0.25*||du|| when rho < 0.25; reject
    (rho <= 0.1) and retry from the SAME, unmoved state with the
    shrunk Delta -- reusing the already-assembled K_eff, no extra
    tangent-assembly cost, up to 30 reject/retry attempts before
    falling through to the outer max_iter budget. This is the
    mechanism that actually prevents the spurious-root failure: a step
    whose predicted-vs-actual reduction disagree wildly (exactly what
    the bouncing pattern above shows) gets REJECTED before it is ever
    committed, rather than being silently accepted because it happened
    to satisfy the residual tolerance.

    **Update (2026-09-02): trust region is no longer tried on every
    step.** Making it the PRIMARY strategy (ahead of plain Newton) was
    tested directly against a previously-working case (the wing
    cantilever's released-static-equilibrium free decay) and broke it:
    plain Newton for that case routinely takes a single correction that
    makes |R| ~100x WORSE before the very next correction collapses it
    to converged (a genuine trace: |R| = 1.081e-1 -> 1.195e+1 ->
    5.660e-7) -- normal, fast, quadratic Newton behavior for this
    element/mesh. Trust region's ratio test REJECTS that first step
    (the local linear model predicts near-total convergence; actual |R|
    instead grew 110x, so rho is wildly negative), forcing a tiny,
    well-predicted-but-nearly-useless capped correction instead --
    destroying quadratic convergence and turning a 2-correction step
    into 100+ iterations without even reaching tol. Plain Newton is now
    PRIMARY again; trust region is an ESCALATION, tried only when plain
    Newton's own attempt either fails outright or shows >= 2
    non-monotone |R| increases (not just the single spike-then-collapse
    above) -- directly distinguishing the two cases by their actual
    residual-history SHAPE rather than a single bad-looking step: the
    genuine spurious-root trace above has THREE increases (223->1.1e5,
    71->1.2e4, 1.5->206), not one. See the driver logic below (search
    "PRIMARY strategy") for the exact escalation condition.

    Pass trust_region=False to skip trust region entirely and fall
    through straight to line_search (if enabled) whenever plain Newton
    fails to converge outright -- note this reintroduces exposure to
    the spurious-root failure mode above, since without trust region
    nothing catches a converged-but-wrong step.

    Line search (added 2026-09-01, `line_search=True` by default) --
    a SEPARATE, final fallback, tried only if the trust-region
    escalation above (or plain Newton directly, if trust_region=False)
    exhausts max_iter without converging outright:
    plain Newton (always take the full du = -K_eff^-1 R step) can fail
    to converge within max_iter for a large single-step state change --
    found directly, not hypothesized, while driving `Tet10SolidTL`
    (elements/nonlinear_solids.py) through a large-amplitude,
    non-equilibrium initial condition in the sibling Multi_Fidelity_NL_
    Structural_ROM project's wing-cantilever dynamic pipeline (see that
    project's paper_notes.md 2026-09-01 write-up). Also verified
    directly that this was NOT fixed by making the tangent itself exact
    (Tet10SolidTL's tangent_stiffness() switched to complex-step
    differentiation the same day, see that class's own docstring) --
    the failures are a genuine lack of globalization in plain Newton,
    not tangent inaccuracy, which is what a line search addresses.

    Each retried correction step backtracks along the Newton direction
    du with an Armijo sufficient-decrease condition on the residual-
    norm-squared merit function m(alpha) = ||R(d + alpha*du)||^2: du is
    guaranteed a descent direction for m at alpha=0 whenever K_eff is a
    reasonable Jacobian, since m'(0) = 2 R^T K_eff du = -2||R||^2 < 0
    (K_eff du = -R by construction) -- so backtracking from the full
    Newton step (alpha=1, tried first) is always well-posed, not a
    heuristic bolted on. Accepts the first alpha in {1, 1/2, 1/4, ...}
    (halved up to 19 times) satisfying m(alpha) <= (1 - 2*c1*alpha)*m0
    with c1=1e-4 (or, at any alpha including 1, immediately accepts a
    trial that already satisfies the OUTER convergence tolerance --
    there is no reason to keep hunting for "more decrease" past that
    point). If NO alpha satisfies genuine sufficient decrease, falls
    back to the FULL step (alpha=1), not to the smallest-alpha trial or
    to whichever trial happened to have the lowest residual -- both of
    those simpler-looking alternatives were tried and found to be real
    bugs (see tests/test_nonlinear_transient.py's tight-tolerance,
    tol=1e-12, linear-limit regression, which caught the first one
    directly): once Rn is already at a problem's own floating-point
    noise floor (an over-tight tol, or simply very close to the true
    solution), every alpha gives ~the same residual up to roundoff, so
    a strict decrease requirement can spuriously reject alpha=1 and
    settle on a near-zero alpha whose candidate barely differs from the
    current state -- freezing the iterate there.

    WHY LINE SEARCH IS A FALLBACK, NOT THE DEFAULT PATH: found directly
    (the second, more consequential bug this feature went through) that
    running EVERY correction through line search, unconditionally,
    actively BROKE a case that converged cleanly without it. Plain
    Newton on this problem regularly overshoots to a much WORSE
    residual on a step's FIRST correction, then converges in one more
    step from that seemingly-worse point (observed directly: |R| going
    0.1 -> 12 -> 6e-7 across three plain corrections, a real, repeatable
    pattern, not a fluke) -- a "one step back, two steps forward" basin-
    hopping behavior a monotone-decrease line search categorically
    forbids, since it will never ACCEPT the worse intermediate point in
    the first place. Forcing every step through line search from the
    start turned that 2-3-iteration convergence into a 150-iteration
    crawl at a near-zero step size that never actually reached
    convergence. So: every step's Newton loop tries the PRIMARY strategy
    FIRST (trust-region Newton by default, or plain unconditional
    Newton if trust_region=False -- either way, max_iter iterations,
    identical in the trust_region=False/line_search=False case to the
    very first, unsafeguarded version of this function), and only
    RETRIES that same step with line-search-guided backtracking if the
    primary attempt exhausts max_iter without converging -- giving
    genuinely well-behaved steps (the common case, including ones that
    benefit from temporary overshoot, which trust region -- unlike line
    search -- does NOT categorically forbid) zero extra cost, while
    still providing a globalization safety net of last resort for steps
    that even trust-region Newton cannot resolve.

    Costs at most max_iter extra assemble_tangent_stiffness()/
    assemble_internal_force() calls PER STEP -- only for a step that
    already failed to converge in max_iter iterations of the primary
    strategy, i.e. only when the extra cost is actually earned. Pass
    line_search=False to disable this fallback retry entirely.

    Returns (t, U_hist), same shape/convention as
    solve_transient_implicit() -- U_hist shape (n_steps+1, n_dof), zero
    at every fixed dof for every recorded step (homogeneous boundary
    conditions, same assumption every driver in this module makes).

    backend="scipy" (default, UNCHANGED behavior) / backend="torch"
    (opt-in, PyTorch side-by-side extension item 94, docs/
    consolidated_future_roadmap.md "Wave 9"): dispatches ONLY the
    per-Newton-iteration linear solve `du_newton = solve(K_eff, -R)`
    inside _newton_attempt() (every one of "plain"/"trust_region"/
    "line_search" mode goes through this same single call site) through
    torch_sparse_solver.torch_dense_solve() instead of
    np.linalg.solve() -- K_eff is already a small dense free-dof matrix
    by construction (K_T[free,free] + a0c*Mff + a1c*Cff), the same
    shape/role FESystem.solve_static()'s own backend="torch" dispatch
    (solver.py) already treats as a dense torch.linalg.solve() case, so
    this reuses that exact helper rather than inventing a new one.
    device="cuda" runs each of those solves GPU-resident (worthwhile
    once K_eff is large enough that GPU dense LU beats a CPU one -- for
    a small system the transfer overhead can dominate, same caveat
    FESystem's own backend="torch" already documents). Every other part
    of this driver (residual assembly, trust-region ratio test, line-
    search backtracking, state commits) is completely unaffected --
    only the raw linear-algebra call changes, exactly mirroring item
    3's "format-bridging happens once at the boundary, caller-facing
    contract stays backend-independent" design. Unknown backend values
    raise ValueError immediately (checked below, before the time-
    stepping loop starts) rather than failing deep inside step 500;
    backend="torch" with torch unavailable likewise fails immediately
    via torch_sparse_solver's own _require_torch(), not mid-solve.
    Validated (tests/test_torch_transient_backend.py) against the
    scipy path to near machine precision on a linear-limit case, same
    validation strategy solve_transient_explicit_nonlinear() (item 31)
    used against solve_transient_explicit()."""
    if backend not in ("scipy", "torch"):
        raise ValueError(
            f"solve_nonlinear_transient: unknown backend={backend!r} -- "
            "expected 'scipy' (default) or 'torch'.")
    if backend == "torch":
        from .torch_sparse_solver import _require_torch, torch_dense_solve
        _require_torch()
    assert fesystem.M is not None and fesystem.C is not None, \
        "call assemble_mass() and assemble_damping() first"
    free = fesystem.free_dofs
    n_dof = fesystem.n_dof
    n_steps = int(round(T_total / dt))
    t = np.arange(n_steps + 1) * dt

    Mff = fesystem.M[np.ix_(free, free)]
    Cff = fesystem.C[np.ix_(free, free)]

    a0c = 1 / (beta * dt**2); a1c = gamma / (beta * dt); a2c = 1 / (beta * dt)
    a3c = 1 / (2 * beta) - 1; a6c = dt * (1 - gamma); a7c = dt * gamma

    def F_free(tt):
        return load.force_at(tt, n_dof, fesystem.npn)[free]

    u = np.zeros(n_dof)
    d = np.zeros(len(free)) if u0 is None else np.asarray(u0)[free].copy()
    v = np.zeros(len(free)) if v0 is None else np.asarray(v0)[free].copy()
    u[free] = d

    F_int0 = fesystem.assemble_internal_force(u, mat, **kwargs)
    a = np.linalg.solve(Mff, F_free(0.0) - Cff @ v - F_int0[free])

    U_hist = np.zeros((n_steps + 1, n_dof))
    U_hist[0, free] = d
    fesystem.commit_all_states(u, mat, **kwargs)

    for step in range(n_steps):
        d_old, v_old, a_old = d, v, a
        F_ext = F_free(t[step + 1])
        ref = max(np.linalg.norm(F_ext), 1e-30)

        # Predictor constants: a(d) = a0c*d - d_pred_const,
        # v(d) = v_pred_const - a7c*d_pred_const + a1c*d -- both affine
        # in the unknown trial d, derived from the SAME Newmark
        # relations solve_transient_implicit() applies once d is known.
        d_pred_const = a0c * d_old + a2c * v_old + a3c * a_old
        v_pred_const = v_old + a6c * a_old

        def _residual(d_val):
            """R(d_val) = M@a(d_val) + C@v(d_val) + F_int(d_val) - F_ext,
            the affine-in-d Newmark residual -- also leaves u[free] set
            to d_val as a side effect, which assemble_tangent_stiffness()
            below relies on (it must see the SAME state _residual() was
            just evaluated at)."""
            u[free] = d_val
            F_int_val = fesystem.assemble_internal_force(u, mat, **kwargs)
            a_val = a0c * d_val - d_pred_const
            v_val = v_pred_const - a7c * d_pred_const + a1c * d_val
            return Mff @ a_val + Cff @ v_val + F_int_val[free] - F_ext

        d_trial0 = d_old.copy()
        R0 = _residual(d_trial0)
        Rn0 = np.linalg.norm(R0)
        # Bump the reference scale to include the FIRST (pre-correction)
        # residual norm, not just ||F_ext||: unlike solve_nonlinear_
        # static's ramp (whose load_factors typically jump straight to
        # a sizable fraction of the full load), a dynamic residual's
        # dominant term can easily be inertial/internal-force rather
        # than external-force -- an UNFORCED free-vibration step, or an
        # early step of a fine-grained quasi-static ramp through a
        # near-zero external load, would otherwise compare a real (if
        # already-tiny) out-of-balance force against an unreachably
        # strict ||F_ext||-only floor. This is a genuine relative-
        # tolerance issue, not a convergence bug: the pre-bump Rn/ref
        # check below still runs (Rn == the reference itself only if it
        # dominates, in which case tol < 1 keeps it from converging
        # trivially), so real cases are unaffected.
        ref = max(ref, Rn0)

        def _newton_attempt(mode):
            """Runs up to max_iter Newton corrections from d_trial0 in
            one of three modes -- 'plain' (unconditional full steps, the
            default PRIMARY strategy -- see the ordering logic below for
            why), 'trust_region' (capped Newton, an escalation used when
            plain Newton looks untrustworthy), or 'line_search' (Armijo
            backtracking, the final fallback). Returns (converged,
            d_trial, R, Rn, n_increases), n_increases counting how many
            iterations had their accepted candidate's |R| INCREASE over
            the previous iteration's -- see the ordering logic below for
            what this is used for."""
            d_trial, R, Rn = d_trial0, R0, Rn0
            Delta = None   # trust radius, initialized from the FIRST
                            # iteration's own full Newton step (see below)
            converged = False
            n_increases = 0
            for it in range(max_iter):
                if verbose:
                    tag = {"trust_region": " [trust region]",
                           "line_search": " [line search]", "plain": ""}[mode]
                    print(f"  step {step:4d} it {it:2d}{tag}  |R|={Rn:.3e}")
                if Rn < tol * ref or Rn < tol:
                    converged = True
                    break

                K_T = fesystem.assemble_tangent_stiffness(u, mat, **kwargs)
                K_eff = K_T[np.ix_(free, free)] + a0c * Mff + a1c * Cff
                if backend == "torch":
                    du_newton = torch_dense_solve(K_eff, -R, device=device)
                else:
                    du_newton = np.linalg.solve(K_eff, -R)

                if mode == "trust_region":
                    # Trust-region-capped Newton -- see this function's
                    # own docstring "Trust region" section for the full
                    # derivation and why this exists (line search alone
                    # does NOT catch a step that satisfies the outer
                    # convergence tolerance while landing on a spurious,
                    # physically-nonsensical root -- it never even gets
                    # a chance to intervene, since nothing "fails").
                    # Reuses the SAME K_eff/du_newton across a bounded
                    # reject-and-shrink retry loop (no re-assembly needed
                    # -- the state hasn't moved on a rejected trial).
                    du_norm = np.linalg.norm(du_newton)
                    if Delta is None:
                        Delta = du_norm   # first try == the full Newton step, unbiased
                    m0 = Rn ** 2
                    stuck = False
                    for _tr in range(30):
                        if du_norm <= Delta:
                            du, at_boundary = du_newton, False
                        else:
                            du, at_boundary = (Delta / du_norm) * du_newton, True
                        d_cand = d_trial + du
                        R_cand = _residual(d_cand)
                        Rn_cand = np.linalg.norm(R_cand)
                        actual_reduction = m0 - Rn_cand ** 2
                        # Predicted reduction from the LOCAL LINEAR
                        # model R_lin(du) = R + K_eff@du -- exact for
                        # this residual's affine (Newmark) part, so this
                        # ratio test measures specifically how well the
                        # NONLINEAR internal-force term is approximated
                        # by that linearization over this step size.
                        Rn_pred = np.linalg.norm(R + K_eff @ du)
                        predicted_reduction = m0 - Rn_pred ** 2
                        rho = (actual_reduction / predicted_reduction
                               if predicted_reduction > 0 else -1.0)
                        if rho > 0.1:   # ACCEPT
                            if rho > 0.75 and at_boundary:
                                Delta = min(2.0 * Delta, 1e3 * du_norm)
                            elif rho < 0.25:
                                Delta = 0.25 * np.linalg.norm(du)
                            d_cand_accepted, R_cand_accepted = d_cand, R_cand
                            break
                        # REJECT -- shrink and retry from the SAME
                        # (unmoved) state; standard trust-region
                        # convention shrinks based on the norm of the
                        # step just tried, not Delta itself.
                        Delta = 0.25 * np.linalg.norm(du)
                    else:
                        # Exhausted the reject/shrink budget without any
                        # accepted step -- Delta has collapsed toward
                        # (near-)zero without finding improvement. This
                        # state is PROVABLY unrecoverable within this
                        # attempt: nothing about it/d_trial/R/K_eff
                        # changes if the outer loop just tries again (no
                        # correction was accepted, so the next outer
                        # iteration re-assembles the SAME tangent at the
                        # SAME state and re-runs this SAME exhausted
                        # search) -- confirmed directly (2026-09-02): on
                        # a genuinely hard wing-cantilever transient
                        # step, this used to burn the ENTIRE max_iter=150
                        # budget with |R| completely frozen before
                        # falling through to line search, wasting up to
                        # 30*150 residual evaluations for zero progress.
                        # Bail out of the OUTER loop immediately instead
                        # -- the line-search fallback below still gets a
                        # chance, just without the wasted iterations.
                        d_cand_accepted, R_cand_accepted = d_trial, R
                        stuck = True
                    d_cand, R_cand = d_cand_accepted, R_cand_accepted
                    if stuck:
                        d_trial, R, Rn = d_cand, R_cand, np.linalg.norm(R_cand)
                        break

                elif mode == "line_search":
                    # Backtracking line search (Armijo sufficient
                    # decrease on m(alpha) = ||R(d_trial+alpha*du)||^2)
                    # -- see this function's own docstring "Line
                    # search" section for the full derivation.
                    du = du_newton
                    m0 = Rn ** 2
                    d_full = d_trial + du
                    R_full = _residual(d_full)
                    Rn_full = np.linalg.norm(R_full)
                    already_good = Rn_full < tol * ref or Rn_full < tol
                    sufficient_decrease = Rn_full ** 2 <= (1.0 - 2e-4) * m0
                    if already_good or sufficient_decrease:
                        d_cand, R_cand = d_full, R_full
                    else:
                        d_cand, R_cand, alpha, accepted = d_full, R_full, 1.0, False
                        for _ls in range(19):
                            alpha *= 0.5
                            d_try = d_trial + alpha * du
                            R_try = _residual(d_try)
                            Rn_try = np.linalg.norm(R_try)
                            if Rn_try < tol * ref or Rn_try < tol or \
                                    Rn_try ** 2 <= (1.0 - 2e-4 * alpha) * m0:
                                d_cand, R_cand, accepted = d_try, R_try, True
                                break
                        if not accepted:
                            d_cand, R_cand = d_full, R_full   # fall back to the full step
                        elif verbose:
                            print(f"    line search backtracked to alpha={alpha:.3e}")
                else:   # mode == "plain"
                    d_cand = d_trial + du_newton
                    R_cand = _residual(d_cand)

                Rn_new = np.linalg.norm(R_cand)
                if Rn_new > Rn:
                    n_increases += 1
                d_trial, R, Rn = d_cand, R_cand, Rn_new
            return converged, d_trial, R, Rn, n_increases

        # Plain unconstrained Newton is the PRIMARY strategy (added
        # 2026-09-02, replacing an earlier "trust-region-as-primary"
        # design after direct evidence it broke a previously-working
        # case -- see below). It is fast and, for THIS element/mesh
        # combination, routinely well-behaved even when a single
        # correction transiently makes |R| much WORSE: e.g. a genuine
        # trace from this same wing-cantilever problem, |R| =
        # 1.081e-1 -> 1.195e+1 -> 5.660e-7 (converged) -- a 110x spike
        # on the FIRST correction, fully absorbed by the SECOND, in
        # exactly 2 corrections. Rejecting that first step (which is
        # what a trust-region ratio test does: the local linear model
        # predicts almost total convergence, actual |R| instead grew
        # 110x, rho << 0.1) forces a tiny, well-predicted-but-nearly-
        # useless capped correction instead, destroying Newton's
        # quadratic convergence -- confirmed directly to make a
        # previously-fast, correct case take 100+ iterations per step
        # without even reaching tol.
        converged, d_trial, R, Rn, n_increases = _newton_attempt("plain")

        # Trust-region-capped Newton is an ESCALATION, tried only when
        # plain Newton's own trajectory looks untrustworthy -- either it
        # failed outright, or (the case a bare non-convergence check
        # CANNOT catch, since nothing "fails") its |R| history shows
        # REPEATED non-monotone increases, not just the single benign
        # spike-then-collapse above. This distinguishes the two failure
        # modes directly from evidence, not a guessed threshold: the
        # genuine spurious-root bug that motivated trust region in the
        # first place traced as |R| = 1.8e4 -> 223 -> 1.1e5 -> 71 ->
        # 1.2e4 -> 1.5 -> 206 -> 5e-4(converged) -- THREE separate
        # increases (223->1.1e5, 71->1.2e4, 1.5->206), a persistently
        # oscillating trajectory that "converges" more by luck than by
        # approach, vs. the single-spike pattern above that is routine
        # and benign for this problem. Trust region's ratio test
        # (rejecting a step whose actual-vs-predicted reduction disagree
        # badly, retrying smaller BEFORE committing -- see this
        # function's own docstring "Trust region" section) still exists
        # specifically to catch this repeated-oscillation case; it is
        # just no longer asked to referee every single ordinary step.
        used_escalation = False
        if trust_region and (not converged or n_increases >= 2):
            if verbose:
                reason = (f"did not converge in {max_iter} iterations (|R|={Rn:.3e})"
                          if not converged else
                          f"converged but with {n_increases} non-monotone |R| increases "
                          f"(possible spurious root)")
                print(f"  step {step:4d}: plain Newton {reason} -- "
                      f"retrying with trust-region-capped Newton")
            converged, d_trial, R, Rn, _ = _newton_attempt("trust_region")
            used_escalation = True

        # Line search remains a SEPARATE, final fallback for the rare
        # case even the escalation above exhausts max_iter without
        # converging.
        if not converged and line_search:
            if verbose:
                print(f"  step {step:4d}: {'trust-region' if used_escalation else 'plain'} "
                      f"Newton did not converge in {max_iter} iterations "
                      f"(|R|={Rn:.3e}) -- retrying with line search")
            converged, d_trial, R, Rn, _ = _newton_attempt("line_search")

        if not converged:
            # Rn here is the residual AFTER the last attempted
            # correction of whichever attempt (fallback retries
            # included) ran last.
            raise RuntimeError(
                f"solve_nonlinear_transient: Newton-Raphson failed to converge "
                f"at step {step} (t={t[step + 1]:.6g}), |R|={Rn:.3e} after "
                f"{max_iter} iterations -- try a smaller dt.")

        d = d_trial
        a = a0c * d - d_pred_const
        v = v_pred_const - a7c * d_pred_const + a1c * d
        u[free] = d
        U_hist[step + 1, free] = d
        fesystem.commit_all_states(u, mat, **kwargs)

    return t, U_hist


def solve_transient_explicit_nonlinear(fesystem, mat, load, T_total, dt,
                                        u0=None, v0=None, **kwargs):
    """Wave 6 item 31 (docs/consolidated_future_roadmap.md): nonlinear
    generalization of FESystem.solve_transient_explicit() -- the SAME
    central-difference recursion on a LUMPED (diagonal) mass matrix,
    with the linear Kff @ d term replaced everywhere by the current
    nonlinear internal-force vector fesystem.assemble_internal_force(
    u, mat, **kwargs)[free], evaluated at the CURRENT displacement d --
    exactly the substitution solve_nonlinear_transient() makes relative
    to solve_transient_implicit(), just applied to the explicit method
    instead of the implicit one.

    WHY NO NEWTON LOOP (the one genuine algorithmic difference from
    every other driver in this module): central difference's update
    solves for d_new from a residual that is AFFINE in d_new with a
    CONSTANT (d_new-independent) coefficient matrix -- F_int(d) here is
    evaluated at the already-known CURRENT step d, not at the unknown
    d_new, unlike Newmark-average-acceleration's a(d_new)/v(d_new) terms
    inside solve_nonlinear_transient()'s residual. So d_new is already
    in closed form (one divide, or one solve against a matrix that
    never changes step to step): there is no nonlinear equation in
    d_new left to iterate on, hence no residual/tangent/Newton machinery
    at all -- this driver's entire per-step cost is exactly ONE
    assemble_internal_force() call, never assemble_tangent_stiffness().
    This is the actual content of item 31's stated formula a = (F_ext -
    F_int(d) - C@v) / m_diag: nothing about that formula is implicit in
    the unknown new state, so nothing about solving it needs Newton.

    WHY THIS LIVES HERE, NOT AS A FESystem METHOD (unlike the linear
    FESystem.solve_transient_explicit() it generalizes): mirrors
    solve_nonlinear_transient()'s and solve_nonlinear_arc_length()'s own
    placement convention -- every driver that must thread `mat` through
    to assemble_internal_force()/commit_all_states() (i.e. every driver
    whose per-step behavior can depend on a nonlinear or path-dependent
    material, which for an explicit method still includes plasticity
    return-mapping, contact, or any state-dependent constitutive law
    even though no TANGENT is ever assembled) belongs in this module;
    FESystem's own methods stay linear-only, matching solver.py's own
    scope statement.

    mat, **kwargs: passed straight through to assemble_internal_force()
    and commit_all_states(), same convention as every other driver in
    this module (e.g. for a plane-stress-J2 `mat` or a contact `kwargs`
    payload).

    Conditionally stable, same caveat as FESystem.solve_transient_
    explicit(): dt must be below the mesh's critical time step or the
    solution blows up -- no automatic check is done here (an expensive
    one for a large system), call fesystem.critical_timestep() (global
    spectral estimate) or the cheaper per-element estimate (Wave 6 item
    32) once yourself and pick dt as a safe fraction of it.

    NOTE on tangent stiffness NOT being required at all: this means an
    element formulation only needs a correct internal_force() (mass
    matrix + internal_force(), specifically -- no tangent_stiffness())
    to run under this driver, unlike every implicit/static driver in
    this module. This is item 33's actual content for the corotational
    shell family (Wave 6) -- see that item's notes for what "explicit
    dynamics for the corotational shell" really requires verifying.

    Returns (t, U_hist), same shape/convention as every other transient
    driver in this module: U_hist shape (n_steps+1, n_dof), zero at
    every fixed dof for every recorded step.

    Validated (tests/test_explicit_nonlinear_transient.py) to agree
    with FESystem.solve_transient_explicit() to near machine precision
    on a LINEAR element/material driven through this nonlinear code
    path -- the same cross-check solve_nonlinear_transient() itself
    documents against solve_transient_implicit()."""
    assert fesystem.M_lumped is not None, "call assemble_lumped_mass() first"
    free = fesystem.free_dofs
    n_dof = fesystem.n_dof
    n_steps = int(round(T_total / dt))
    t = np.arange(n_steps + 1) * dt

    Mff = fesystem.M_lumped[np.ix_(free, free)]
    Cff = None if fesystem.C is None else fesystem.C[np.ix_(free, free)]
    diag_fast = (Cff is None) or np.allclose(Cff, np.diag(np.diag(Cff)))

    def F_free(tt):
        return load.force_at(tt, n_dof, fesystem.npn)[free]

    u = np.zeros(n_dof)
    d = np.zeros(len(free)) if u0 is None else np.asarray(u0)[free].copy()
    v = np.zeros(len(free)) if v0 is None else np.asarray(v0)[free].copy()
    u[free] = d

    m_diag = np.diag(Mff)
    Cv0 = (Cff @ v) if Cff is not None else 0.0
    F_int0 = fesystem.assemble_internal_force(u, mat, **kwargs)
    a = (F_free(0.0) - Cv0 - F_int0[free]) / m_diag
    d_prev = d - dt * v + 0.5 * dt**2 * a

    U_hist = np.zeros((n_steps + 1, n_dof))
    U_hist[0, free] = d
    fesystem.commit_all_states(u, mat, **kwargs)

    if diag_fast:
        c_diag = np.zeros(len(free)) if Cff is None else np.diag(Cff)
        keff_diag = m_diag / dt**2 + c_diag / (2 * dt)
        for step in range(n_steps):
            Fn = F_free(t[step])
            u[free] = d
            F_int = fesystem.assemble_internal_force(u, mat, **kwargs)
            rhs = (Fn - F_int[free] + (2 * m_diag / dt**2) * d
                   - (m_diag / dt**2 - c_diag / (2 * dt)) * d_prev)
            d_new = rhs / keff_diag
            d_prev, d = d, d_new
            u[free] = d
            U_hist[step + 1, free] = d
            fesystem.commit_all_states(u, mat, **kwargs)
    else:
        Keff = Mff / dt**2 + Cff / (2 * dt)
        A2 = 2 * Mff / dt**2
        A3 = Mff / dt**2 - Cff / (2 * dt)
        for step in range(n_steps):
            Fn = F_free(t[step])
            u[free] = d
            F_int = fesystem.assemble_internal_force(u, mat, **kwargs)
            rhs = Fn - F_int[free] + A2 @ d - A3 @ d_prev
            d_new = np.linalg.solve(Keff, rhs)
            d_prev, d = d, d_new
            u[free] = d
            U_hist[step + 1, free] = d
            fesystem.commit_all_states(u, mat, **kwargs)

    return t, U_hist


def solve_transient_displacement_control(fesystem, mat, control_dof, u_target_fn,
                                          T_total, dt, beta=0.25, gamma=0.5,
                                          u0=None, v0=None, v0_control=None,
                                          a0_control=None, tol=1e-8, max_iter=30,
                                          verbose=False, line_search=True,
                                          trust_region=True,
                                          backend="scipy", device="cpu", **kwargs):
    """Item 34 (Wave 6, docs/consolidated_future_roadmap.md: "Arc-length
    / displacement-control transient variant (dynamic snap-through)"):
    the dynamic (Newmark-implicit) counterpart to solve_nonlinear_
    displacement_control() above -- exactly the same generalization
    solve_nonlinear_transient() already made for load-controlled
    solve_nonlinear_static(): keep the same physical control strategy
    (prescribe one DOF's motion directly, solve for equilibrium
    everywhere else and the reaction it takes), replace the static
    residual with the Newmark-implicit dynamic one.

    WHY DISPLACEMENT CONTROL, NOT ARC-LENGTH, FOR THE TRANSIENT CASE:
    static arc-length (solve_nonlinear_arc_length() above) exists
    because a STATIC limit point (dP/ddelta = 0) makes load control's
    own K_T singular right at the point of interest, and displacement
    control itself only fails at a SNAP-BACK (where even the controlled
    DOF's own displacement reverses direction -- see that function's
    own docstring). Neither failure mode transfers cleanly to a
    TRANSIENT problem: the Newmark effective stiffness K_eff(d) =
    K_T(d) + a0c*M + a1c*C carries a mass term that stays positive
    definite even exactly AT a static limit point (a0c*M dominates for
    any physically reasonable dt) -- inertia itself regularizes the
    Jacobian a static analysis of the same structure would lose. This
    is the actual reason a transient snap-through problem is usually
    easier to drive through than its static counterpart, not a
    coincidence of this particular implementation. A genuine "arc-
    length in time" would also need a different continuation parameter
    than lambda -- there is no free load-factor unknown once load.
    force_at(t) is fully prescribed at every t the way every other
    driver in this module assumes -- so it is a real but separate
    problem, left for a future item if an actual use case needs it
    (matching the roadmap's own framing of this item as low-priority
    and deferred; nothing in this package's own reference cases does).

    control_dof: prescribed via u_target_fn (a plain callable t ->
    float, NOT a fesystem-wide load.force_at(t, n_dof, npn) object --
    this DOF is excluded from the Newton unknown vector entirely,
    mirroring solve_nonlinear_displacement_control()'s own `free =
    [d for d in fesystem.free_dofs if d != control_dof]`) at every
    step. Every OTHER free dof is solved via the SAME trust-region/
    line-search Newton escalation ladder solve_nonlinear_transient()
    uses (see that function's own extensive docstring for the full
    derivation of "plain" -> "trust_region" -> "line_search"; this
    driver reuses that exact ladder, just with control_dof excluded
    from the unknown set and its own displacement history known rather
    than solved for).

    MASS/DAMPING COUPLING TO control_dof (the one genuinely new piece
    versus the static driver, which never touches M or C at all):
    a(d_free)/v(d_free) for the FREE dofs enter the residual through
    Mff/Cff exactly as in solve_nonlinear_transient(), but the full
    mass/damping matrices also generally COUPLE control_dof's own
    acceleration/velocity into every free dof's equation of motion
    (M[free, control_dof] and C[free, control_dof] are not zero in
    general, even for a diagonal lumped mass, whenever control_dof
    shares an element with a free dof) -- ignoring that coupling would
    silently drop real inertial/damping forces the actuator induces in
    the rest of the structure. control_dof's own (d_c, v_c, a_c) are
    tracked with the SAME Newmark corrector recursion FESystem.
    solve_transient_implicit() and every other driver in this module
    already use (a_new = a0c*(d_new-d_old) - a2c*v_old - a3c*a_old;
    v_new = v_old + a6c*a_old + a7c*a_new) -- the only difference from
    a normal free dof is that d_c's value at each step comes directly
    from u_target_fn(t) (known), not from a Newton solve, so this
    recursion is a plain forward evaluation, not an unknown to solve
    for. v0_control/a0_control seed that recursion at t=0 -- default
    (None) ESTIMATES them from u_target_fn itself via forward
    differences at the SAME dt the integration uses (v0_control =
    (u_target_fn(dt) - u_target_fn(0))/dt; a0_control from the forward
    second difference), not a naive 0.0/"starts from rest" default.
    This matters more than it looks: found directly (not assumed) that
    seeding a0_control=v0_control=0.0 for an ordinary ramp that starts
    moving immediately at t=0 (u_target_fn has a nonzero one-sided
    derivative there -- the overwhelmingly common case for a "ramp from
    0 to some target" profile) leaves the Newmark acceleration-recovery
    recursion permanently inconsistent with the TRUE motion: average-
    acceleration Newmark has no numerical damping at all, so that one
    wrong initial value does not decay -- it rings forever, alternating
    sign every step with slowly GROWING amplitude (reproduced in
    isolation: v_c oscillating between 0 and 2x the true ramp rate every
    single step, a_c growing linearly and unboundedly, for a plain
    constant-velocity ramp that has an EXACTLY zero true acceleration).
    The forward-difference estimate instead seeds the recursion at
    (near-)its own true fixed point for a smooth or piecewise-linear
    u_target_fn, eliminating the ringing (confirmed directly: the same
    reproduction above, seeded correctly, agrees with the analytic
    constant-velocity/zero-acceleration solution to machine precision).
    Pass v0_control/a0_control explicitly only if u_target_fn genuinely
    starts from a state other than what its own near-t=0 values imply
    (e.g. a deliberate initial-velocity impulse the caller wants
    distinct from the forward-difference estimate).

    Returns (t, U_hist, reaction_hist): t/U_hist as in every other
    transient driver in this module (U_hist shape (n_steps+1, n_dof),
    row 0 the t=0 initial condition with u_target_fn(0) already placed
    at control_dof); reaction_hist (n_steps+1,) is the actuation force
    control_dof needs at EVERY recorded step -- the dynamic equilibrium
    residual AT control_dof, M[control_dof]@a_full + C[control_dof]@
    v_full + F_int(u)[control_dof] - fesystem.F[control_dof], the
    direct dynamic generalization of the static driver's F_int[control
    _dof] - fesystem.F[control_dof] reaction: it now also includes
    whatever inertial/damping force the actuator must supply to
    produce control_dof's own prescribed acceleration/velocity, not
    just hold its static equilibrium.

    backend="scipy" (default, UNCHANGED behavior) / backend="torch"
    (opt-in, PyTorch side-by-side extension item 95, docs/
    consolidated_future_roadmap.md "Wave 9"): dispatches ONLY the
    per-Newton-iteration linear solve inside this driver's own Newton
    loop (the same "plain"/"trust_region"/"line_search" ladder solve_
    nonlinear_transient() uses) through torch_sparse_solver.
    torch_dense_solve() instead of np.linalg.solve() -- identical
    dispatch point/mechanism to solve_nonlinear_transient()'s own item
    94 addition (see that function's docstring for the full rationale);
    not repeated here beyond noting it applies verbatim to this
    driver's K_eff as well. Unknown backend values raise ValueError
    immediately, before the time-stepping loop starts. Validated
    (tests/test_torch_transient_backend.py) against the scipy path on
    the same von Mises truss fixture test_transient_displacement_
    control.py's own quasi-static check uses."""
    if backend not in ("scipy", "torch"):
        raise ValueError(
            f"solve_transient_displacement_control: unknown backend={backend!r} "
            "-- expected 'scipy' (default) or 'torch'.")
    if backend == "torch":
        from .torch_sparse_solver import _require_torch, torch_dense_solve
        _require_torch()
    assert fesystem.M is not None and fesystem.C is not None, \
        "call assemble_mass() and assemble_damping() first"
    assert control_dof not in fesystem.fixed_dofs, \
        "control_dof must not already be a fixed support"
    free = np.array([d for d in fesystem.free_dofs if d != control_dof])
    n_dof = fesystem.n_dof
    n_steps = int(round(T_total / dt))
    t = np.arange(n_steps + 1) * dt

    def _dense_vec(mat_like, rows, cols):
        """rows/cols: one is an array, the other a scalar int -- always
        returns a plain 1-D numpy array, regardless of whether M/C are
        stored dense (sparse=False, the default) or as scipy.sparse
        (sparse=True), since np.asarray() alone does not densify a
        scipy sparse matrix the way it does a plain ndarray."""
        sub = mat_like[rows, cols]
        if hasattr(sub, "toarray"):
            sub = sub.toarray()
        return np.asarray(sub).reshape(-1)

    Mff = fesystem.M[np.ix_(free, free)]
    Cff = fesystem.C[np.ix_(free, free)]
    Mfc = _dense_vec(fesystem.M, free, control_dof)
    Cfc = _dense_vec(fesystem.C, free, control_dof)
    Mcf = _dense_vec(fesystem.M, control_dof, free)
    Ccf = _dense_vec(fesystem.C, control_dof, free)
    Mcc = fesystem.M[control_dof, control_dof]
    Ccc = fesystem.C[control_dof, control_dof]

    a0c = 1 / (beta * dt**2); a1c = gamma / (beta * dt); a2c = 1 / (beta * dt)
    a3c = 1 / (2 * beta) - 1; a6c = dt * (1 - gamma); a7c = dt * gamma

    F_ext = fesystem.F   # fixed external load, same convention as
                          # solve_nonlinear_displacement_control() --
                          # only control_dof's own prescribed VALUE
                          # changes step to step, not F_ext itself

    u = np.zeros(n_dof)
    d = np.zeros(len(free)) if u0 is None else np.asarray(u0)[free].copy()
    v = np.zeros(len(free)) if v0 is None else np.asarray(v0)[free].copy()
    u[free] = d
    d_c = u_target_fn(0.0)
    # Forward-difference estimates of control_dof's own initial
    # velocity/acceleration from u_target_fn itself, at the SAME dt the
    # integration uses -- see this function's own docstring for why
    # the naive 0.0/"starts from rest" default rings forever instead.
    if v0_control is None or a0_control is None:
        d_c1 = u_target_fn(dt)
    if v0_control is None:
        v_c = (d_c1 - d_c) / dt
    else:
        v_c = v0_control
    if a0_control is None:
        d_c2 = u_target_fn(2 * dt)
        a_c = (d_c2 - 2 * d_c1 + d_c) / dt ** 2
    else:
        a_c = a0_control
    u[control_dof] = d_c

    F_int0 = fesystem.assemble_internal_force(u, mat, **kwargs)
    a = np.linalg.solve(
        Mff, F_ext[free] - Mfc * a_c - Cff @ v - Cfc * v_c - F_int0[free])

    U_hist = np.zeros((n_steps + 1, n_dof))
    reaction_hist = np.zeros(n_steps + 1)
    U_hist[0] = u
    reaction_hist[0] = (Mcc * a_c + Mcf @ a + Ccc * v_c + Ccf @ v
                         + F_int0[control_dof] - F_ext[control_dof])
    fesystem.commit_all_states(u, mat, **kwargs)

    for step in range(n_steps):
        d_old, v_old, a_old = d, v, a
        d_c_old, v_c_old, a_c_old = d_c, v_c, a_c

        d_c = u_target_fn(t[step + 1])
        a_c = a0c * (d_c - d_c_old) - a2c * v_c_old - a3c * a_c_old
        v_c = v_c_old + a6c * a_c_old + a7c * a_c

        F_ext_eff = F_ext[free] - Mfc * a_c - Cfc * v_c
        ref = max(np.linalg.norm(F_ext_eff), 1e-30)

        d_pred_const = a0c * d_old + a2c * v_old + a3c * a_old
        v_pred_const = v_old + a6c * a_old

        def _residual(d_val):
            u[free] = d_val
            u[control_dof] = d_c
            F_int_val = fesystem.assemble_internal_force(u, mat, **kwargs)
            a_val = a0c * d_val - d_pred_const
            v_val = v_pred_const - a7c * d_pred_const + a1c * d_val
            return Mff @ a_val + Cff @ v_val + F_int_val[free] - F_ext_eff

        d_trial0 = d_old.copy()
        R0 = _residual(d_trial0)
        Rn0 = np.linalg.norm(R0)
        ref = max(ref, Rn0)   # see solve_nonlinear_transient()'s own
                               # docstring for why this bump exists

        def _newton_attempt(mode):
            d_trial, R, Rn = d_trial0, R0, Rn0
            Delta = None
            converged = False
            n_increases = 0
            for it in range(max_iter):
                if verbose:
                    tag = {"trust_region": " [trust region]",
                           "line_search": " [line search]", "plain": ""}[mode]
                    print(f"  step {step:4d} it {it:2d}{tag}  |R|={Rn:.3e}")
                if Rn < tol * ref or Rn < tol:
                    converged = True
                    break

                u[free] = d_trial
                u[control_dof] = d_c
                K_T = fesystem.assemble_tangent_stiffness(u, mat, **kwargs)
                K_eff = K_T[np.ix_(free, free)] + a0c * Mff + a1c * Cff
                if backend == "torch":
                    du_newton = torch_dense_solve(K_eff, -R, device=device)
                else:
                    du_newton = np.linalg.solve(K_eff, -R)

                if mode == "trust_region":
                    du_norm = np.linalg.norm(du_newton)
                    if Delta is None:
                        Delta = du_norm
                    m0 = Rn ** 2
                    stuck = False
                    for _tr in range(30):
                        if du_norm <= Delta:
                            du, at_boundary = du_newton, False
                        else:
                            du, at_boundary = (Delta / du_norm) * du_newton, True
                        d_cand = d_trial + du
                        R_cand = _residual(d_cand)
                        Rn_cand = np.linalg.norm(R_cand)
                        actual_reduction = m0 - Rn_cand ** 2
                        Rn_pred = np.linalg.norm(R + K_eff @ du)
                        predicted_reduction = m0 - Rn_pred ** 2
                        rho = (actual_reduction / predicted_reduction
                               if predicted_reduction > 0 else -1.0)
                        if rho > 0.1:
                            if rho > 0.75 and at_boundary:
                                Delta = min(2.0 * Delta, 1e3 * du_norm)
                            elif rho < 0.25:
                                Delta = 0.25 * np.linalg.norm(du)
                            d_cand_accepted, R_cand_accepted = d_cand, R_cand
                            break
                        Delta = 0.25 * np.linalg.norm(du)
                    else:
                        d_cand_accepted, R_cand_accepted = d_trial, R
                        stuck = True
                    d_cand, R_cand = d_cand_accepted, R_cand_accepted
                    if stuck:
                        d_trial, R, Rn = d_cand, R_cand, np.linalg.norm(R_cand)
                        break

                elif mode == "line_search":
                    du = du_newton
                    m0 = Rn ** 2
                    d_full = d_trial + du
                    R_full = _residual(d_full)
                    Rn_full = np.linalg.norm(R_full)
                    already_good = Rn_full < tol * ref or Rn_full < tol
                    sufficient_decrease = Rn_full ** 2 <= (1.0 - 2e-4) * m0
                    if already_good or sufficient_decrease:
                        d_cand, R_cand = d_full, R_full
                    else:
                        d_cand, R_cand, alpha, accepted = d_full, R_full, 1.0, False
                        for _ls in range(19):
                            alpha *= 0.5
                            d_try = d_trial + alpha * du
                            R_try = _residual(d_try)
                            Rn_try = np.linalg.norm(R_try)
                            if Rn_try < tol * ref or Rn_try < tol or \
                                    Rn_try ** 2 <= (1.0 - 2e-4 * alpha) * m0:
                                d_cand, R_cand, accepted = d_try, R_try, True
                                break
                        if not accepted:
                            d_cand, R_cand = d_full, R_full
                        elif verbose:
                            print(f"    line search backtracked to alpha={alpha:.3e}")
                else:
                    d_cand = d_trial + du_newton
                    R_cand = _residual(d_cand)

                Rn_new = np.linalg.norm(R_cand)
                if Rn_new > Rn:
                    n_increases += 1
                d_trial, R, Rn = d_cand, R_cand, Rn_new
            return converged, d_trial, R, Rn, n_increases

        converged, d_trial, R, Rn, n_increases = _newton_attempt("plain")

        used_escalation = False
        if trust_region and (not converged or n_increases >= 2):
            if verbose:
                reason = (f"did not converge in {max_iter} iterations (|R|={Rn:.3e})"
                          if not converged else
                          f"converged but with {n_increases} non-monotone |R| increases")
                print(f"  step {step:4d}: plain Newton {reason} -- "
                      f"retrying with trust-region-capped Newton")
            converged, d_trial, R, Rn, _ = _newton_attempt("trust_region")
            used_escalation = True

        if not converged and line_search:
            if verbose:
                print(f"  step {step:4d}: {'trust-region' if used_escalation else 'plain'} "
                      f"Newton did not converge in {max_iter} iterations "
                      f"(|R|={Rn:.3e}) -- retrying with line search")
            converged, d_trial, R, Rn, _ = _newton_attempt("line_search")

        if not converged:
            raise RuntimeError(
                f"solve_transient_displacement_control: Newton-Raphson failed to "
                f"converge at step {step} (t={t[step + 1]:.6g}, "
                f"control_dof target={d_c:.6g}), |R|={Rn:.3e} after "
                f"{max_iter} iterations -- try a smaller dt.")

        d = d_trial
        a = a0c * d - d_pred_const
        v = v_pred_const - a7c * d_pred_const + a1c * d
        u[free] = d
        u[control_dof] = d_c
        F_int = fesystem.assemble_internal_force(u, mat, **kwargs)
        reaction_hist[step + 1] = (Mcc * a_c + Mcf @ a + Ccc * v_c + Ccf @ v
                                    + F_int[control_dof] - F_ext[control_dof])
        U_hist[step + 1] = u
        fesystem.commit_all_states(u, mat, **kwargs)

    return t, U_hist, reaction_hist


# ---------------------------------------------------------------------------
# v1.0.1 (P2): shared convergence options. Every driver below gains a keyword-only
# `options=NewtonOptions(...)`; explicit keywords still win, old calls are unchanged.
# Applied here (not as decorators on each def) so the driver bodies stay untouched.
# ---------------------------------------------------------------------------
from .newton_options import NewtonOptions, accepts_options   # noqa: E402

for _name in ("solve_contact_lagrange_static", "solve_contact_augmented_lagrange_static",
              "solve_nonlinear_static", "solve_nonlinear_displacement_control",
              "solve_nonlinear_arc_length", "solve_nonlinear_koiter_newton",
              "solve_nonlinear_koiter_newton_generic", "solve_nonlinear_static_koiter_newton",
              "solve_nonlinear_transient", "solve_transient_displacement_control"):
    globals()[_name] = accepts_options(globals()[_name])
del _name
