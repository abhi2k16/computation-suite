"""
nonlinear_dynamics.py -- generalized reduced nonlinear time integration.

Generalizes the MFS-NLROM paper's dynamic algorithm (Eq. 19-33) to
accept ANY `nonlinear_rom.ReducedForceModel`, separating the METHOD
(Newmark-beta predictor + a surrogate-lag correction scheme, all of it
NEWTON-FREE -- the entire point of this model family versus a full
nonlinear Newmark-Newton step every timestep, see
`fea_engine.nonlinear_solver.solve_nonlinear_transient` for that
alternative) from the MODEL (whatever concrete `ReducedForceModel`
supplies `F_nl` via `.predict()`).

Works in mass-normalized modal coordinates (`M_r = I`, this whole
model family's standing convention -- see `nonlinear_rom.py`'s module
docstring and `pod.PodBasis`/`galerkin.GalerkinROM`'s own
mass-orthonormalization), solving

    qddot_nl + C_r @ qdot_nl + Lambda*q_nl + F_nl(...) = F_ext(t)

where `q_nl` is the physical reduced state (the actual Newmark
predictor/corrector variable throughout) and `F_nl` is supplied by
`force_model.predict(q)` at whatever domain that model's own
convention expects -- see `domain` in `integrate_newmark_surrogate`'s
docstring for the two cases (`MultiFidelitySurrogate`'s `q_l` vs
`PolynomialModalROM`'s own native `q_nl`, evaluated directly / bypassing
its own internal Newton solve, exactly matching
`docs/nonlinear_surrogate_rom_roadmap.md` Section 5's "PolynomialModalROM
evaluated in 'direct' mode" phrasing).
"""
import numpy as np


def integrate_newmark_surrogate(Lambda, C_r, force_model, F_ext,
                                 q0, qdot0, dt, n_steps,
                                 domain="q_l",
                                 beta=0.25, gamma=0.5,
                                 correction="fixed_point",
                                 n_fixed_point=4,
                                 newton_tol=1e-9, newton_max_iter=30,
                                 newton_max_backtrack=30, q_bound=None):
    """Reduced nonlinear Newmark-beta time integration, decoupled from
    any one `ReducedForceModel` implementation.

    Four `correction` strategies. The first three call
    `force_model.predict()` only, never `.jacobian()`; "newton" also
    uses `.jacobian()`:

    - "none": accept the PREVIOUS step's converged `F_nl` as a frozen,
      external-like force in THIS step's linear Newmark solve, then
      evaluate `F_nl` once more at the resulting state purely to record
      it (and to lag into the NEXT step) -- never re-solved with that
      fresh value. The cheapest, least accurate option; included as the
      honest baseline every other mode is compared against, matching
      the roadmap's own "predict at face value, no lag correction"
      description.
    - "fixed_point": Picard-iterate the same lagged-force linear solve
      `n_fixed_point` times, re-evaluating `F_nl` at each iterate before
      re-solving. This project's own earlier, VALIDATED MFS-NLROM
      dynamic reproduction (the `mfs-nlrom-beam` skill's
      `scripts/dynamic_newmark.py`, `modal_newmark_mfs`) used exactly
      this scheme (with `n_fixed_point=4`) after finding that a raw
      one-shot lag ("none") accumulates enough one-step error over
      hundreds of steps to blow up on its own flat-beam benchmark --
      ported faithfully here as a validated fallback, not re-derived.
      DEFAULT (since 2026-09-24; was "sign_deviation", see below) --
      the safe Newton-free choice for either `domain`. Each Picard
      iteration contracts the force error by rho(J_nl @ Keff^-1), which
      the Newmark a0*I term keeps small at practical dt (<=0.11 on the
      Case 2 cylindrical shell, where 4 iterations match "newton" to
      printed precision).
    - "sign_deviation": a two-solve predict/correct pass modelled on
      MFS-NLROM (He et al. 2023) Eq. 32,
      `dF(t+dt) = sign(qddot_nl(t)) * (F_nl(t+dt) - F_nl(t))`: predict
      once with the lagged force, then re-solve ONCE with
      `F_lag + sign(qddot) * (F_pred - F_lag)`. NOT a literal
      transcription: the paper's Eqs. 28-32 (which DO extract cleanly
      with pymupdf, contrary to what this docstring used to say) apply
      the sign-scaled increment as a one-solve linear EXTRAPOLATION of
      the force history, `F_pred(t+dt) = F_nl(t) + dF(t)`.
      **Known to diverge; kept only for reproducing that paper.** For
      any mode with negative acceleration the sign turns the correction
      into an anti-Picard step (error propagation `2I - A` instead of
      `A`, A = -J_nl Keff^-1), i.e. it pushes the force FURTHER into the
      past, and a lagging restoring force pumps energy into the system.
      On the NonLin-HyROM Case 2 cylindrical shell (PolynomialModalROM,
      2-6 modes, dt=0.005) it goes non-finite by t~0.30 s for every mode
      set -- earlier than the uncorrected "none" (~0.49 s) -- and the
      literal Eq. 28-32 scheme diverges the same way (~0.31 s), while
      one Picard step ("fixed_point", n_fixed_point=1) stays within
      ~1.1% of "newton". Details: NonLin-HyROM/case2_signdev_diagnostic.py.
      Weakly nonlinear runs may not blow up, but it is never more
      accurate than "fixed_point" (Case 1 flat plate: 21.7% vs 19.8%).
    - "newton": Wave 12 item 114 (docs/consolidated_future_roadmap.md,
      `ICE-ROM/GAP_ANALYSIS.md` gap #3) -- genuine Newton-Raphson on the
      full nonlinear reduced equation every step, using `force_model`'s
      own analytic `.jacobian()`, with backtracking line search and an
      optional amplitude clamp (`q_bound`). Promoted, generalized only
      to accept any `q_nl`-domain `ReducedForceModel` rather than one
      hard-coded polynomial fit, from a validated ad hoc prototype in
      this project's own `ICE-ROM/validation/dynamic_comparison.py::
      integrate_ice_rom_newton`. **Requires `domain="q_nl"`** (raises
      `ValueError` for `domain="q_l"`) -- the other three correction
      modes' whole point is staying Newton-FREE by using the `q_l`-
      domain's closed-form `q_l = q_nl + F_nl(q_l)/Lambda` relation
      (Eq. 9-11) instead of iterating, which is exactly what a genuine
      per-step Newton solve on `q_l` would need to unwind self-
      consistently (the map from `q_nl` to `q_l` is itself only defined
      implicitly through `F_nl`); scoping this mode to `q_nl` keeps its
      residual `Keff @ q + F_nl(q) - rhs_base = 0` a direct, unambiguous
      function of the single unknown `q`, with no implicit domain
      translation nested inside the Newton loop. This is genuinely more
      ROBUST than the three Newton-free modes for a stiff polynomial fit
      (`PolynomialModalROM`) evaluated directly in its own native
      domain: a single lagged/corrected linear predictor step can land
      outside the fitted polynomial's well-behaved region with no way
      back (confirmed empirically -- see the gap analysis), whereas
      exact Newton convergence at every step stays on the TRUE nonlinear
      equilibrium as long as it's a fair extrapolation of the fit's own
      trend. `newton_tol`/`newton_max_iter`/`newton_max_backtrack`/
      `q_bound` are used only by this mode (ignored otherwise, same
      convention as `n_fixed_point` being unused outside
      `"fixed_point"`).

    Parameters
    ----------
    Lambda : (n_modes,) ndarray
        Linear modal stiffness (natural-frequency-squared, diagonal).
    C_r : (n_modes,) or (n_modes, n_modes) ndarray
        Modal damping. A (n_modes,) array is the diagonal of a diagonal
        damping matrix (e.g. `2*zeta*omega`, the standard
        Rayleigh/proportional-damping modal form -- see
        `nonlinear_rom.py`'s `AppliedLoadStrategy`/training-fixture
        convention); a full `(n_modes, n_modes)` matrix is also
        accepted for the non-proportional case.
    force_model : nonlinear_rom.ReducedForceModel
        Supplies `F_nl` via `.predict(q)`.
    F_ext : callable(t) -> (n_modes,) ndarray, or (n_modes,) array_like
        External modal force history. A constant array/scalar-broadcast
        is accepted directly (e.g. `F_ext=0` for a free-decay run).
    q0, qdot0 : (n_modes,) array_like
        Initial nonlinear modal displacement / velocity.
    dt : float
    n_steps : int
    domain : {"q_l", "q_nl"}, default "q_l"
        Which domain `force_model.predict()` expects. `"q_l"` (matching
        `MultiFidelitySurrogate`) triggers the `q_l = q_nl + F_nl(q_l)/Lambda`
        algebraic relation (Eq. 9-11) every step; `"q_nl"` (matching
        `PolynomialModalROM` evaluated directly, bypassing its own
        internal Newton solve -- see `nonlinear_rom.PolynomialModalROM.force()`)
        calls `force_model.predict(q_nl)` with no translation at all.
    beta, gamma : float, default 0.25 / 0.5
        Newmark parameters -- average-acceleration by default, matching
        `fea_engine.solve_transient_implicit()`/`solve_nonlinear_transient()`'s
        own defaults (this module intentionally uses the same `beta`/
        `gamma` naming as that package rather than the roadmap draft's
        own `alpha`/`beta` names, for cross-package consistency).
    n_fixed_point : int, default 4
        Number of Picard iterations for `correction="fixed_point"`
        (unused by the other modes) -- matches the validated
        prototype's own default.
    newton_tol : float, default 1e-9
        Relative residual-norm convergence tolerance for
        `correction="newton"` (unused otherwise).
    newton_max_iter : int, default 30
        Max Newton iterations per step for `correction="newton"`
        (unused otherwise).
    newton_max_backtrack : int, default 30
        Max step-halving attempts per Newton iteration's backtracking
        line search, for `correction="newton"` (unused otherwise).
    q_bound : (n_modes,) array_like, optional
        Per-mode amplitude clamp applied to each Newton trial iterate
        for `correction="newton"` (unused otherwise) -- a physically-
        motivated safeguard against a full (or line-searched) Newton
        step wandering into a region the fitted polynomial was never
        trained on and producing a spurious, badly-extrapolated root.
        `None` (default) applies no clamp.

    Returns
    -------
    t : (n_steps+1,) ndarray
    q_nl_hist, q_l_hist, F_nl_hist : (n_steps+1, n_modes) ndarray
        `q_l_hist` equals `q_nl_hist` exactly when `domain="q_nl"` (no
        translation is ever applied in that case).
    """
    Lambda = np.asarray(Lambda, dtype=float)
    n = len(Lambda)

    C_r = np.asarray(C_r, dtype=float)
    if C_r.ndim == 1:
        C_mat = np.diag(C_r)
    elif C_r.ndim == 2:
        C_mat = C_r
    else:
        raise ValueError(f"C_r must be 1-D (diagonal) or 2-D, got ndim={C_r.ndim}")

    if domain not in ("q_l", "q_nl"):
        raise ValueError(f"domain={domain!r} must be 'q_l' or 'q_nl'")
    if correction not in ("none", "fixed_point", "sign_deviation", "newton"):
        raise ValueError(
            f"correction={correction!r} must be 'none', 'fixed_point', 'sign_deviation', or 'newton'")
    if correction == "newton" and domain != "q_nl":
        raise ValueError(
            "correction='newton' requires domain='q_nl' -- see this function's own "
            "docstring (correction parameter, 'newton' bullet) for why a genuine "
            "per-step Newton solve is only well-posed with no implicit q_l<->q_nl "
            "translation nested inside it.")
    if q_bound is not None:
        q_bound = np.asarray(q_bound, dtype=float)

    if callable(F_ext):
        F_ext_fn = F_ext
    else:
        F_ext_const = np.broadcast_to(np.asarray(F_ext, dtype=float), (n,))
        F_ext_fn = lambda t: F_ext_const

    a0c = 1 / (beta * dt ** 2); a1c = gamma / (beta * dt); a2c = 1 / (beta * dt)
    a3c = 1 / (2 * beta) - 1; a4c = gamma / beta - 1; a5c = dt / 2 * (gamma / beta - 2)
    a6c = dt * (1 - gamma); a7c = dt * gamma

    Keff = np.diag(Lambda) + a0c * np.eye(n) + a1c * C_mat

    def native_input(q_nl_trial, F_nl_est):
        """Translate the physical q_nl into force_model's own predict()
        domain -- identity when domain="q_nl" (PolynomialModalROM,
        direct mode), the Eq. 9-11 algebraic relation when domain="q_l"
        (MultiFidelitySurrogate)."""
        if domain == "q_nl":
            return q_nl_trial
        return q_nl_trial + F_nl_est / Lambda

    q = np.asarray(q0, dtype=float).copy()
    qdot = np.asarray(qdot0, dtype=float).copy()

    # Consistent initial acceleration + F_nl(0), refined once (the
    # F_nl=0 seed used for the very first native_input() call is only
    # ever a starting guess for the domain="q_l" translation -- a single
    # refinement avoids biasing q_l(0) on a strongly nonlinear IC).
    q_native0 = native_input(q, np.zeros(n))
    F_nl0 = force_model.predict(q_native0)
    q_native0 = native_input(q, F_nl0)
    F_nl0 = force_model.predict(q_native0)
    qddot = F_ext_fn(0.0) - C_mat @ qdot - Lambda * q - F_nl0

    t = np.arange(n_steps + 1) * dt
    q_nl_hist = np.zeros((n_steps + 1, n))
    q_l_hist = np.zeros((n_steps + 1, n))
    F_nl_hist = np.zeros((n_steps + 1, n))
    q_nl_hist[0] = q
    q_l_hist[0] = q_native0
    F_nl_hist[0] = F_nl0

    def solve_q_nl(rhs_base, F_nl_est):
        return np.linalg.solve(Keff, rhs_base - F_nl_est)

    for step in range(n_steps):
        rhs_base = (F_ext_fn(t[step + 1])
                    + a0c * q + a2c * qdot + a3c * qddot
                    + C_mat @ (a1c * q + a4c * qdot + a5c * qddot))
        F_nl_lag = F_nl_hist[step]

        if correction == "none":
            q_nl_new = solve_q_nl(rhs_base, F_nl_lag)
            q_native = native_input(q_nl_new, F_nl_lag)
            F_nl_new = force_model.predict(q_native)

        elif correction == "fixed_point":
            F_nl_iter = F_nl_lag
            q_native = None
            for _ in range(n_fixed_point):
                q_nl_iter = solve_q_nl(rhs_base, F_nl_iter)
                q_native = native_input(q_nl_iter, F_nl_iter)
                F_nl_iter = force_model.predict(q_native)
            q_nl_new = solve_q_nl(rhs_base, F_nl_iter)
            F_nl_new = F_nl_iter

        elif correction == "sign_deviation":
            q_nl_pred = solve_q_nl(rhs_base, F_nl_lag)
            q_native_pred = native_input(q_nl_pred, F_nl_lag)
            F_nl_pred = force_model.predict(q_native_pred)
            deviation = F_nl_pred - F_nl_lag
            F_nl_corrected = F_nl_lag + np.sign(qddot) * deviation
            q_nl_new = solve_q_nl(rhs_base, F_nl_corrected)
            q_native = native_input(q_nl_new, F_nl_corrected)
            F_nl_new = force_model.predict(q_native)

        else:  # "newton" -- genuine Newton-Raphson, domain="q_nl" only
            # (validated by the guard clause above). Residual and its
            # exact Jacobian, both direct functions of q (no implicit
            # domain translation): G(q) = Keff@q - rhs_base + F_nl(q).
            def residual(qv):
                return Keff @ qv - rhs_base + force_model.predict(qv)

            q_trial = q.copy()
            G = residual(q_trial)
            ref = max(1.0, np.max(np.abs(rhs_base)))
            for _ in range(newton_max_iter):
                if np.max(np.abs(G)) < newton_tol * ref:
                    break
                J = Keff + force_model.jacobian(q_trial)
                try:
                    dq = np.linalg.solve(J, G)
                except np.linalg.LinAlgError:
                    break
                # Backtracking line search: a plain full Newton step on
                # a strongly hardening/softening restoring force can
                # badly overshoot and converge to a distant, spurious
                # root instead of the true nearby one -- halve the step
                # until the residual norm actually decreases (the exact
                # failure mode fea_engine.nonlinear_solver's own
                # line_search/trust_region safeguards exist for).
                step_scale = 1.0
                Gn = np.max(np.abs(G))
                q_new_trial, G_new_trial = q_trial, G
                for _ in range(newton_max_backtrack):
                    cand = q_trial - step_scale * dq
                    if q_bound is not None:
                        cand = np.clip(cand, -q_bound, q_bound)
                    G_cand = residual(cand)
                    if np.max(np.abs(G_cand)) < Gn or step_scale < 1e-6:
                        q_new_trial, G_new_trial = cand, G_cand
                        break
                    step_scale *= 0.5
                q_trial, G = q_new_trial, G_new_trial
            q_nl_new = q_trial
            q_native = q_nl_new   # domain="q_nl": identity, same as native_input() above
            F_nl_new = force_model.predict(q_native)

        qddot_new = a0c * (q_nl_new - q) - a2c * qdot - a3c * qddot
        qdot_new = qdot + a6c * qddot + a7c * qddot_new
        q, qdot, qddot = q_nl_new, qdot_new, qddot_new

        q_nl_hist[step + 1] = q
        q_l_hist[step + 1] = q_native
        F_nl_hist[step + 1] = F_nl_new

    return t, q_nl_hist, q_l_hist, F_nl_hist
