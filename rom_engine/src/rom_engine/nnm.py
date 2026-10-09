"""
nnm.py -- generalized multi-harmonic-balance (HBM) continuation for
nonlinear normal mode (NNM) backbone curves.

Generalizes the MFS-NLROM paper's Eq. 34-44 per the NLvib/MANLAB
precedent (predictor-corrector continuation with either analytic or
numerically-differentiated Jacobians, Alternating Frequency-Time (AFT)
evaluation of the nonlinear force in the harmonic domain), parametrized
by ANY `nonlinear_rom.ReducedForceModel` -- exactly like
`nonlinear_dynamics.py`, this module never knows or cares whether the
concrete model is `MultiFidelitySurrogate`, `PolynomialModalROM`, or a
hand-wrapped full-order reference.

Deliberate scope, stated honestly:

- **Undamped-backbone convention.** `HarmonicBalanceSystem.apply_A()`
  DOES support a damping term `C_r` (mirroring `frequency.FrequencyROM`'s
  own `A(omega) = -omega^2*M + i*omega*C + K` construction -- see that
  class's docstring for the real cos/sin block form this generalizes
  it to), but `solve_nnm_backbone()`'s own continuation is written for
  the classical, LITERATURE-STANDARD NNM backbone problem: an
  UNFORCED, and in practice usually UNDAMPED (`C_r=0`), autonomous
  periodic-orbit family (Peeters et al. 2009's own convention, and
  exactly what the reference paper's Fig. 8/9 backbones are). A caller
  may pass nonzero `C_r` to `HarmonicBalanceSystem` for other uses of
  the class (e.g. the `n_harmonics=1` cross-check against
  `frequency.FrequencyROM` this module's own test suite performs), but
  `solve_nnm_backbone()` does not claim to trace a damped-forced
  resonance curve -- that is a materially different, larger
  continuation problem (two free parameters, forcing amplitude AND
  frequency, not addressed here).
- **Two continuation drivers, sharing the same AFT/HBM machinery but
  making a genuinely different trade-off.** `solve_nnm_backbone()` uses
  NATURAL-PARAMETER (prescribed-amplitude) continuation: each point is
  solved at a DIRECTLY PRESCRIBED master-mode amplitude (Newton-
  converged, seeded from the previous point's converged state). This is
  simpler and fully robust for the non-folding (monotonically
  hardening/softening) backbones this function's own validation
  targets, but it CANNOT continue past a point where the master mode's
  own amplitude is non-monotonic along the true curve (a genuine
  "fold" in amplitude) -- the implicit constraint `a1 = target` has no
  solution beyond such a point for a `target` on the far side of it.
  `solve_nnm_backbone_arclength()` instead uses genuine PSEUDO-
  ARCLENGTH continuation (Keller 1977, with a documented secant-
  predictor simplification -- see its own docstring), which CAN
  continue through such a fold, at the cost of tracing arclength rather
  than a caller-chosen amplitude grid (so `amplitudes`, unlike
  `solve_nnm_backbone`'s, is a MEASURED output, not a prescribed input).
  Neither function is strictly better: `solve_nnm_backbone` is simpler
  to call and reason about (a fixed amplitude grid) when the backbone
  is known not to fold; `solve_nnm_backbone_arclength` is the one to
  reach for once a fold is suspected or must be traced through (e.g.
  near an internal resonance).
- **No subharmonic multiplier.** Only the fundamental (period = 2*pi/omega)
  is traced -- subharmonic resonance capture (period = m*2*pi/omega) is
  a real extension of the same harmonic-balance machinery, not
  implemented here.

See `docs/nonlinear_surrogate_rom_roadmap.md` Section 6 (original design)
and Section 11 (pseudo-arclength addendum) for the full design
background this module implements against.
"""
__author__ = "Abhijeet"
import numpy as np


class HarmonicBalanceSystem:
    """Assembles the LINEAR (+ damping) harmonic-balance operator
    `A(omega)` for a given modal stiffness `Lambda`, modal damping
    `C_r`, and number of retained harmonics -- pure linear-algebra
    bookkeeping, no nonlinear force involved yet (that lives in
    `solve_nnm_backbone`'s own AFT evaluation, since it needs a
    `ReducedForceModel` this class deliberately does not depend on).

    A periodic state is represented as real trigonometric Fourier
    coefficients `Z`, shape `(1 + 2*n_harmonics, n_modes)`:
    `Z[0]` the DC (mean) term, `Z[2*k-1]`/`Z[2*k]` the cosine/sine
    coefficients of harmonic `k` (`k=1..n_harmonics`), for
    `q(t) = Z[0] + sum_k Z[2k-1]*cos(k*omega*t) + Z[2k]*sin(k*omega*t)`.

    Deliberately mirrors `frequency.FrequencyROM`'s own
    `A(omega) = -omega^2*M + i*omega*C + K` construction: harmonic-balancing
    `qddot_nl + C_r@qdot_nl + Lambda*q_nl = f(t)` term-by-term against
    `cos(k*omega*t)`/`sin(k*omega*t)` at each harmonic `k` gives, for
    the `(cos_k, sin_k)` pair (real 2x2-per-mode block, `M_r=I`):

        [ Lambda-(k*omega)^2      k*omega*C_r        ] [a_k]   [f_cos_k]
        [ -k*omega*C_r            Lambda-(k*omega)^2 ] [b_k] = [f_sin_k]

    which is EXACTLY the real (cos, sin) decomposition of
    `FrequencyROM`'s complex `A(k*omega)` acting on a real
    `f(t)=f_cos*cos(k*omega*t)+f_sin*sin(k*omega*t)` forcing -- checked
    directly in `test_nnm.py`'s `n_harmonics=1` cross-check, a strong
    cross-module consistency check `frequency.py` didn't previously
    have a partner for. The DC row (`k=0`) is simply `Lambda*a0 = f_DC`
    (no `qddot`/`qdot` contribution from a time-constant term).
    """

    def __init__(self, Lambda, C_r=None, n_harmonics=5, n_time_samples=None):
        self.Lambda = np.asarray(Lambda, dtype=float)
        self.n = len(self.Lambda)
        if C_r is None:
            self.C_r = np.zeros(self.n)
        else:
            C_r = np.asarray(C_r, dtype=float)
            if C_r.ndim == 2:
                # Only proportional (diagonal-in-this-basis) damping is
                # supported by the real cos/sin block form above -- a
                # full non-diagonal C_r would couple different MODES'
                # cos/sin harmonics nontrivially, a real generalization
                # not needed for this module's own (undamped-backbone)
                # validation target, so deliberately not implemented.
                if not np.allclose(C_r, np.diag(np.diag(C_r))):
                    raise ValueError(
                        "HarmonicBalanceSystem only supports diagonal (proportional) "
                        "C_r -- a full matrix would couple different modes' cos/sin "
                        "harmonics, not implemented (see class docstring)")
                self.C_r = np.diag(C_r)
            else:
                self.C_r = C_r
        if n_harmonics < 1:
            raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")
        self.n_harmonics = int(n_harmonics)
        self.n_coeffs = 1 + 2 * self.n_harmonics   # real coefficients per mode

        NT = n_time_samples or max(4 * self.n_harmonics + 4, 32)
        NT = max(NT, 2 * self.n_harmonics + 1)     # Nyquist: NT >= 2*n_harmonics+1
        if NT % 2 == 1:
            NT += 1
        self.n_time_samples = NT

        theta = np.arange(NT) * 2 * np.pi / NT     # omega-independent AFT sample grid
        self.theta = theta
        T = np.zeros((NT, self.n_coeffs))
        T[:, 0] = 1.0
        for k in range(1, self.n_harmonics + 1):
            T[:, 2 * k - 1] = np.cos(k * theta)
            T[:, 2 * k] = np.sin(k * theta)
        self.T = T                                  # (NT, n_coeffs): coeffs -> time samples

        Tinv = np.zeros((self.n_coeffs, NT))
        Tinv[0, :] = 1.0 / NT
        for k in range(1, self.n_harmonics + 1):
            Tinv[2 * k - 1, :] = (2.0 / NT) * np.cos(k * theta)
            Tinv[2 * k, :] = (2.0 / NT) * np.sin(k * theta)
        self.Tinv = Tinv                             # (n_coeffs, NT): time samples -> coeffs
        # Tinv @ T == I_{n_coeffs} exactly for this uniform grid whenever
        # NT >= 2*n_harmonics+1 (orthogonality of sampled sinusoids over
        # one full period) -- checked in test_nnm.py, not asserted here.

    def reconstruct(self, Z):
        """Z, shape (n_coeffs, n) -> q(theta_i), shape (n_time_samples, n)."""
        return self.T @ Z

    def project(self, q_time):
        """q(theta_i), shape (n_time_samples, n) -> Z, shape (n_coeffs, n)
        (exact trigonometric-interpolation coefficients, truncated to
        n_harmonics -- the AFT "inverse" step)."""
        return self.Tinv @ q_time

    def apply_A(self, Z, omega):
        """Z, shape (n_coeffs, n) -> A(omega) @ Z, same shape."""
        out = np.zeros_like(Z)
        out[0] = self.Lambda * Z[0]
        for k in range(1, self.n_harmonics + 1):
            a_k, b_k = Z[2 * k - 1], Z[2 * k]
            diag_term = self.Lambda - (k * omega) ** 2
            out[2 * k - 1] = diag_term * a_k + (k * omega) * (self.C_r * b_k)
            out[2 * k] = -(k * omega) * (self.C_r * a_k) + diag_term * b_k
        return out

    def apply_dA_domega(self, Z, omega):
        """d(A(omega) @ Z)/domega, same shape as Z."""
        out = np.zeros_like(Z)
        for k in range(1, self.n_harmonics + 1):
            a_k, b_k = Z[2 * k - 1], Z[2 * k]
            d_diag = -2 * (k ** 2) * omega
            out[2 * k - 1] = d_diag * a_k + k * (self.C_r * b_k)
            out[2 * k] = -k * (self.C_r * a_k) + d_diag * b_k
        return out

    def linear_matrix(self, omega):
        """Dense (n_coeffs*n, n_coeffs*n) matrix form of apply_A(), for
        Newton-Jacobian assembly. Flattening convention: flat index
        `c*n + m` for coefficient row `c`, mode `m` (numpy's default
        row-major flatten of a (n_coeffs, n) array) -- matched exactly
        by `_nl_harmonics_jacobian`'s `np.kron`-based assembly."""
        n, NC = self.n, self.n_coeffs
        NCn = NC * n
        Amat = np.zeros((NCn, NCn))
        Amat[0:n, 0:n] = np.diag(self.Lambda)
        C_mat = np.diag(self.C_r)
        for k in range(1, self.n_harmonics + 1):
            idx_c = slice((2 * k - 1) * n, (2 * k) * n)
            idx_s = slice((2 * k) * n, (2 * k + 1) * n)
            diag_term = np.diag(self.Lambda - (k * omega) ** 2)
            Amat[idx_c, idx_c] = diag_term
            Amat[idx_c, idx_s] = (k * omega) * C_mat
            Amat[idx_s, idx_c] = -(k * omega) * C_mat
            Amat[idx_s, idx_s] = diag_term
        return Amat

    def dA_domega_matrix(self, omega):
        """Dense (n_coeffs*n, n_coeffs*n) matrix form of apply_dA_domega()."""
        n, NC = self.n, self.n_coeffs
        NCn = NC * n
        dAmat = np.zeros((NCn, NCn))
        C_mat = np.diag(self.C_r)
        for k in range(1, self.n_harmonics + 1):
            idx_c = slice((2 * k - 1) * n, (2 * k) * n)
            idx_s = slice((2 * k) * n, (2 * k + 1) * n)
            d_diag = np.diag(np.full(n, -2 * (k ** 2) * omega))
            dAmat[idx_c, idx_c] = d_diag
            dAmat[idx_c, idx_s] = k * C_mat
            dAmat[idx_s, idx_c] = -k * C_mat
            dAmat[idx_s, idx_s] = d_diag
        return dAmat


def _finite_diff_jacobian(force_model, q, eps=1e-7):
    """Numerical fallback for force_model.jacobian(q) via central
    differences on predict() -- used by jacobian="auto" whenever a
    model has no jacobian() (or it raises NotImplementedError), and
    always when jacobian="numerical" is requested explicitly."""
    n = len(q)
    J = np.zeros((n, n))
    for j in range(n):
        dq = np.zeros(n)
        step = eps * max(1.0, abs(q[j]))
        dq[j] = step
        J[:, j] = (force_model.predict(q + dq) - force_model.predict(q - dq)) / (2 * step)
    return J


def _model_jacobian(force_model, q, jacobian):
    if jacobian == "numerical":
        return _finite_diff_jacobian(force_model, q)
    # jacobian == "auto" (or "analytic", trusting the model to raise if it can't)
    try:
        return force_model.jacobian(q)
    except (NotImplementedError, AttributeError):
        if jacobian == "analytic":
            raise
        return _finite_diff_jacobian(force_model, q)


def _native_q_for_model(q_nl, Lambda, force_model, domain, F_nl_guess=None,
                         n_fixed_point=6, tol=1e-12):
    """Given the PHYSICAL nonlinear modal displacement q_nl at one AFT
    time sample, return (q_native, F_nl): the input force_model.predict()
    expects and the resulting nonlinear force.

    domain="q_nl": identity (PolynomialModalROM's own native domain,
    "direct" mode per nonlinear_rom.py's Section 3.3 docstring) -- one
    predict() call.

    domain="q_l": solves q_l = q_nl + F_nl(q_l)/Lambda (Eq. 9-11) via a
    short Picard iteration, mirroring nonlinear_dynamics.py's own
    established fixed-point convention for this exact relation --
    warm-started from F_nl_guess (the previous AFT sample's own F_nl,
    since AFT samples are typically close together along one period)."""
    if domain == "q_nl":
        F_nl = force_model.predict(q_nl)
        return q_nl, F_nl
    F_nl = np.zeros_like(q_nl) if F_nl_guess is None else F_nl_guess
    q_l = q_nl + F_nl / Lambda
    for _ in range(n_fixed_point):
        F_nl_new = force_model.predict(q_l)
        q_l_new = q_nl + F_nl_new / Lambda
        converged = np.max(np.abs(q_l_new - q_l)) < tol
        q_l, F_nl = q_l_new, F_nl_new
        if converged:
            break
    return q_l, F_nl


def _nl_harmonics(hb, Z, force_model, domain, need_jacobian, jacobian="auto"):
    """AFT: Z (n_coeffs, n) -> (F_nl harmonics (n_coeffs, n), dFnl_dZ or
    None). dFnl_dZ, when requested, is chain-ruled through the domain
    translation (identity for "q_nl"; implicit-function-theorem on
    q_l = q_nl + F_nl(q_l)/Lambda for "q_l") and assembled in the SAME
    (n_coeffs*n, n_coeffs*n) flattening HarmonicBalanceSystem.linear_matrix()
    uses, via np.kron per time sample (n_time_samples is small -- a
    dense per-sample n x n Jacobian contribution is cheap)."""
    q_time = hb.reconstruct(Z)               # (NT, n)
    NT, n, NC = hb.n_time_samples, hb.n, hb.n_coeffs
    F_time = np.zeros_like(q_time)
    q_native_time = np.zeros_like(q_time)
    F_nl_guess = None
    for i in range(NT):
        q_native, F_nl = _native_q_for_model(q_time[i], hb.Lambda, force_model, domain, F_nl_guess)
        F_time[i] = F_nl
        q_native_time[i] = q_native
        F_nl_guess = F_nl
    Fnl_harm = hb.project(F_time)             # (n_coeffs, n)

    if not need_jacobian:
        return Fnl_harm, None

    NCn = NC * n
    dFnl_dZ = np.zeros((NCn, NCn))
    invLam = 1.0 / hb.Lambda
    for i in range(NT):
        J_native = _model_jacobian(force_model, q_native_time[i], jacobian)
        if domain == "q_nl":
            J_qnl = J_native
        else:
            # d(q_l)/d(q_nl) via implicit function theorem on
            # q_l - F_nl(q_l)/Lambda = q_nl:
            #   (I - diag(1/Lambda) @ J_native) @ dq_l/dq_nl = I
            M = np.eye(n) - invLam[:, None] * J_native
            dql_dqnl = np.linalg.solve(M, np.eye(n))
            J_qnl = J_native @ dql_dqnl
        outer_coef = np.outer(hb.Tinv[:, i], hb.T[i, :])   # (n_coeffs, n_coeffs)
        dFnl_dZ += np.kron(outer_coef, J_qnl)
    return Fnl_harm, dFnl_dZ


def _newton_fixed_amplitude(hb, force_model, target_amplitude, Z_free0, omega0,
                             master_mode, domain, jacobian, tol, max_iter):
    """One Newton solve of `h(Z, omega) = 0` with the master mode's
    first-harmonic amplitude PRESCRIBED to `target_amplitude` and its
    sine coefficient fixed at 0 (the phase convention -- see
    `solve_nnm_backbone`'s own docstring for why) -- the natural-
    parameter corrector step, factored out so both `solve_nnm_backbone`'s
    own continuation loop AND `solve_nnm_backbone_arclength`'s bootstrap
    (its first two points, before genuine arclength stepping begins --
    see that function's docstring) share ONE Newton solve, rather than
    two independently-written near-duplicates of the same corrector.

    Parameters
    ----------
    Z_free0 : ndarray, shape (n_coeffs*n - 2,)
        Initial guess for the "free" coefficients (every entry of the
        flattened `Z` EXCEPT the prescribed `a1[master_mode]` and the
        phase-fixed `b1[master_mode]`).
    omega0 : float
        Initial guess for omega.

    Returns
    -------
    Z_free : ndarray, same shape as Z_free0 -- the converged (or, if
        `converged` is False, last-iterate) free coefficients.
    omega : float
    converged : bool
    """
    n, NC = hb.n, hb.n_coeffs
    NCn = NC * n
    a1_flat = 1 * n + master_mode
    b1_flat = 2 * n + master_mode
    free_idx = np.array([i for i in range(NCn) if i not in (a1_flat, b1_flat)])
    eq_idx = np.array([i for i in range(NCn) if i != b1_flat])

    u = np.concatenate([Z_free0, [omega0]])
    conv = False
    for it in range(max_iter):
        Z_flat = np.zeros(NCn)
        Z_flat[free_idx] = u[:-1]
        Z_flat[a1_flat] = target_amplitude
        Z_flat[b1_flat] = 0.0
        omega_it = u[-1]
        Z = Z_flat.reshape(NC, n)

        Fnl_harm, dFnl_dZ = _nl_harmonics(hb, Z, force_model, domain,
                                           need_jacobian=True, jacobian=jacobian)
        Alin = hb.apply_A(Z, omega_it)
        h_flat = (Alin + Fnl_harm).flatten()
        R = h_flat[eq_idx]
        ref = max(np.max(np.abs(h_flat)), 1e-30)
        if np.max(np.abs(R)) < tol * ref or np.max(np.abs(R)) < tol:
            conv = True
            break

        Amat = hb.linear_matrix(omega_it)
        dh_dZ_full = Amat + dFnl_dZ
        dh_domega_full = hb.apply_dA_domega(Z, omega_it).flatten()

        J_reduced = np.zeros((len(eq_idx), len(free_idx) + 1))
        J_reduced[:, :len(free_idx)] = dh_dZ_full[np.ix_(eq_idx, free_idx)]
        J_reduced[:, -1] = dh_domega_full[eq_idx]

        try:
            du = np.linalg.solve(J_reduced, -R)
        except np.linalg.LinAlgError:
            break
        u = u + du

    return u[:-1], u[-1], conv


def solve_nnm_backbone(hb, force_model, q_amplitude_range, n_points=20,
                        domain="q_nl", master_mode=0, omega0=None,
                        jacobian="auto", tol=1e-8, max_iter=30):
    """Traces an NNM backbone (frequency vs. amplitude) for the master
    mode `master_mode`, over `n_points` amplitudes linearly spaced
    across `q_amplitude_range = (a_min, a_max)`.

    h(Z, omega) = A(omega)@Z + F_nl_harmonics(Z) = 0 (Eq. 34, F_ext=0 --
    see module docstring for the undamped-autonomous-backbone scope),
    Newton-solved at each prescribed amplitude via `jacobian="auto"`
    (analytic `force_model.jacobian()` when available, a numerical
    AFT-consistent fallback otherwise -- Section 3.1's protocol makes
    `jacobian()` optional, not every model provides one), continued
    across `q_amplitude_range` using each converged point as the next
    point's initial guess (natural-parameter continuation -- see module
    docstring for why this is not full pseudo-arclength).

    Phase/amplitude convention (autonomous periodic orbits have a free
    time-shift AND a free amplitude parametrizing the 1-D backbone
    family; 2 extra scalar constraints pin down one point): the master
    mode's own first-harmonic SINE coefficient is fixed at 0 (kills the
    time-shift symmetry -- an arbitrary but standard phase choice) and
    its COSINE coefficient is prescribed to the target amplitude
    directly; the corresponding SINE-harmonic-1 row of h=0 is dropped
    (redundant given the phase symmetry) while the COSINE-harmonic-1
    row is KEPT as a real equation -- with `a_1[master]` no longer a
    free unknown, that equation instead determines `omega`, which is
    exactly the frequency-vs-amplitude relationship a backbone curve is.

    Parameters
    ----------
    hb : HarmonicBalanceSystem
    force_model : nonlinear_rom.ReducedForceModel
    q_amplitude_range : (float, float)
        (a_min, a_max) for the master mode's first-harmonic amplitude.
    n_points : int, default 20
    domain : {"q_nl", "q_l"}, default "q_nl"
        Matches nonlinear_dynamics.integrate_newmark_surrogate's own
        convention -- see that module's docstring.
    master_mode : int, default 0
    omega0 : float, optional
        Initial frequency guess for the FIRST amplitude point (default:
        the linear natural frequency `sqrt(Lambda[master_mode])`).
        Subsequent points are seeded from the previous point's own
        converged state.
    jacobian : {"auto", "analytic", "numerical"}, default "auto"
    tol, max_iter : Newton convergence controls, per point.

    Returns
    -------
    amplitudes : (n_points,) ndarray
    omegas : (n_points,) ndarray (NaN at any point that failed to converge)
    Z_hist : (n_points, n_coeffs, n_modes) ndarray -- the full periodic
        state at each backbone point (for reconstructing waveforms, or
        feeding into downstream analysis).
    converged : (n_points,) bool ndarray
    """
    if domain not in ("q_l", "q_nl"):
        raise ValueError(f"domain={domain!r} must be 'q_l' or 'q_nl'")
    n, NC = hb.n, hb.n_coeffs
    NCn = NC * n
    if not (0 <= master_mode < n):
        raise ValueError(f"master_mode={master_mode} out of range for n={n} modes")

    amplitudes = np.linspace(q_amplitude_range[0], q_amplitude_range[1], n_points)
    a1_flat = 1 * n + master_mode
    b1_flat = 2 * n + master_mode
    free_idx = np.array([i for i in range(NCn) if i not in (a1_flat, b1_flat)])

    if omega0 is None:
        omega0 = float(np.sqrt(hb.Lambda[master_mode]))

    omegas = np.full(n_points, np.nan)
    converged_flags = np.zeros(n_points, dtype=bool)
    Z_hist = np.zeros((n_points, NC, n))

    Z_free = np.zeros(len(free_idx))
    omega = omega0

    for p, target in enumerate(amplitudes):
        Z_free_new, omega_new, conv = _newton_fixed_amplitude(
            hb, force_model, target, Z_free, omega, master_mode, domain, jacobian, tol, max_iter)

        omegas[p] = omega_new
        converged_flags[p] = conv
        Z_flat_final = np.zeros(NCn)
        Z_flat_final[free_idx] = Z_free_new
        Z_flat_final[a1_flat] = target
        Z_flat_final[b1_flat] = 0.0
        Z_hist[p] = Z_flat_final.reshape(NC, n)

        if conv:
            # natural-parameter continuation: seed the next amplitude
            # from this point's own converged state
            Z_free, omega = Z_free_new, omega_new
        # if not converged, keep the previous Z_free/omega as the seed
        # for the next point anyway -- often still a reasonable guess,
        # and better than resetting to the linear-mode guess every time

    return amplitudes, omegas, Z_hist, converged_flags


def _augmented_residual_jacobian(hb, force_model, Z_free_all, omega, free_idx_all,
                                  master_mode, domain, jacobian):
    """h(Z, omega) and its Jacobian w.r.t. [Z_free_all, omega], with
    `Z_free_all` now including the master mode's own first-harmonic
    COSINE coefficient (`a1[master]`) as a free unknown -- unlike
    `_newton_fixed_amplitude`'s `free_idx`, which excludes it because
    that corrector prescribes it directly. Only the phase-fixing
    `b1[master] = 0` row/column is excluded here (the SAME single
    constraint `solve_nnm_backbone` uses to kill the autonomous orbit's
    free time-shift -- see that function's docstring)."""
    n, NC = hb.n, hb.n_coeffs
    NCn = NC * n
    b1_flat = 2 * n + master_mode
    eq_idx = free_idx_all   # square (before the arclength row is added): drop only the sin_1[master] row

    Z_flat = np.zeros(NCn)
    Z_flat[free_idx_all] = Z_free_all
    Z_flat[b1_flat] = 0.0
    Z = Z_flat.reshape(NC, n)

    Fnl_harm, dFnl_dZ = _nl_harmonics(hb, Z, force_model, domain,
                                       need_jacobian=True, jacobian=jacobian)
    Alin = hb.apply_A(Z, omega)
    h_flat = (Alin + Fnl_harm).flatten()
    R_eq = h_flat[eq_idx]

    Amat = hb.linear_matrix(omega)
    dh_dZ_full = Amat + dFnl_dZ
    dh_domega_full = hb.apply_dA_domega(Z, omega).flatten()

    J_eq = np.zeros((len(eq_idx), len(free_idx_all) + 1))
    J_eq[:, :len(free_idx_all)] = dh_dZ_full[np.ix_(eq_idx, free_idx_all)]
    J_eq[:, -1] = dh_domega_full[eq_idx]
    return R_eq, J_eq, h_flat


def _augmented_residual_jacobian_scaled(hb, force_model, w, free_idx_all, master_mode,
                                         domain, jacobian, omega_scale):
    """`_augmented_residual_jacobian`, but operating on the RESCALED
    unknown vector `w = [Z_free_all, omega/omega_scale]` instead of the
    physical `v = [Z_free_all, omega]` -- see
    `solve_nnm_backbone_arclength`'s docstring for why this rescaling is
    necessary (a raw Euclidean arclength metric mixing amplitude-sized
    and omega-sized components is dominated by whichever happens to have
    the larger RAW magnitude, not whichever is physically more
    significant). The equilibrium residual itself still needs the
    PHYSICAL omega (unscaled internally here before calling
    `hb.apply_A`/`_nl_harmonics`); the returned Jacobian's LAST COLUMN is
    scaled back up by `omega_scale` (chain rule:
    `d/d(omega/omega_scale) = omega_scale * d/domega`) so the whole
    system stays consistent in `w`-space."""
    v = w.copy()
    v[-1] = w[-1] * omega_scale
    R_eq, J_eq, h_flat = _augmented_residual_jacobian(
        hb, force_model, v[:-1], v[-1], free_idx_all, master_mode, domain, jacobian)
    J_eq = J_eq.copy()
    J_eq[:, -1] *= omega_scale
    return R_eq, J_eq, h_flat


def _arclength_corrector(hb, force_model, w_pred, w_prev, t_hat, ds, free_idx_all,
                          master_mode, domain, jacobian, tol, max_iter, omega_scale):
    """Newton-correct the AUGMENTED system [h_eq(w) = 0; pseudo-arclength
    constraint] back onto the solution curve, starting from the linear
    predictor `w_pred = w_prev + ds*t_hat`. `w` packs the RESCALED
    unknowns `[Z_free_all (including a1[master]), omega/omega_scale]`
    -- see `_augmented_residual_jacobian_scaled`'s own docstring for why.

    The pseudo-arclength constraint (Keller 1977) is the LINEAR equation
    `t_hat . (w - w_prev) - ds = 0` -- geometrically, "stay a distance
    `ds` from the previous point, measured along the tangent direction,
    in the RESCALED metric" -- which is what lets this corrector
    converge to a point PAST a fold in `a1` (amplitude), unlike
    `_newton_fixed_amplitude`'s corrector, whose implicit constraint
    (`a1 = target`, exactly) has NO solution on the far side of an
    `a1`-turning-point for a `target` beyond the fold. See
    `solve_nnm_backbone_arclength`'s own docstring for how `t_hat`
    itself is obtained (a SECANT direction from the two most recently
    converged points, not a from-scratch tangent computed via a separate
    null-space solve -- a deliberate, documented simplification, see
    that docstring)."""
    w = w_pred.copy()
    conv = False
    for it in range(max_iter):
        R_eq, J_eq, h_flat = _augmented_residual_jacobian_scaled(
            hb, force_model, w, free_idx_all, master_mode, domain, jacobian, omega_scale)
        R_arc = np.dot(t_hat, w - w_prev) - ds
        ref = max(np.max(np.abs(h_flat)), 1e-30)
        arc_ref = max(abs(ds), 1.0)
        if (np.max(np.abs(R_eq)) < tol * ref or np.max(np.abs(R_eq)) < tol) and \
                abs(R_arc) < tol * arc_ref:
            conv = True
            break
        J = np.vstack([J_eq, t_hat[None, :]])
        R = np.concatenate([R_eq, [R_arc]])
        try:
            dw = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            break
        w = w + dw
    return w, conv


def solve_nnm_backbone_arclength(hb, force_model, a0, n_points=40, direction=1.0,
                                  ds=None, domain="q_nl", master_mode=0, omega0=None,
                                  jacobian="auto", tol=1e-8, max_iter=30,
                                  max_step_halvings=4, omega_scale=None):
    """Traces the SAME kind of NNM backbone as `solve_nnm_backbone`
    (frequency vs. master-mode amplitude, undamped-autonomous
    convention -- see module docstring), but via genuine PSEUDO-
    ARCLENGTH continuation (Keller, "Numerical solution of bifurcation
    and nonlinear eigenvalue problems", 1977) in the combined
    `[Z_free, a1[master], omega]` space, rather than
    `solve_nnm_backbone`'s natural-parameter (prescribed-amplitude)
    continuation. The practical difference: `solve_nnm_backbone` cannot
    continue PAST a point where the master mode's own amplitude is
    non-monotonic along the true backbone curve (a genuine "fold" in
    amplitude -- the Newton corrector's implicit constraint, `a1 =
    target`, has no solution beyond such a point for a `target` on the
    far side of it); this function CAN, because arclength -- not
    amplitude -- is what it prescribes at each step.

    A GENUINE, DELIBERATE SIMPLIFICATION relative to the fullest
    "textbook" Keller method, stated explicitly (matching this
    project's practice of scoping deviations honestly rather than
    silently -- see `scm_lp.py`/`passivity.py`/`hankel_norm.py`'s own
    scope notes for precedent): the PREDICTOR tangent `t_hat` at each
    step is a SECANT direction -- `(v_new - v_prev) / ||v_new - v_prev||`
    from the two most recently converged points -- not the true local
    tangent obtained by solving the Jacobian's null space (a separate
    linear solve full Keller pseudo-arclength normally uses). This is a
    well-precedented simpler variant (secant-predictor arclength
    continuation), reusing the SAME corrector machinery either way,
    since the corrector's own arclength constraint only needs `t_hat`
    to be roughly transversal to the curve near the fold, not exactly
    tangent to it -- the corrector's own Newton iterations, not the
    predictor, are what actually re-converge onto the true curve at
    each step. The trade-off: a slightly less accurate PREDICTOR very
    close to a sharp fold (more step-halvings may be needed there), at
    the benefit of no new null-space linear algebra -- see
    `docs/nonlinear_surrogate_rom_roadmap.md` Section 11 for the full
    reasoning and the validation this choice was checked against.

    BOOTSTRAP: the first two points are found via ORDINARY
    prescribed-amplitude Newton solves (`_newton_fixed_amplitude`, the
    SAME corrector `solve_nnm_backbone` itself uses) at `a0` and
    `a0 + direction*step0` -- genuine pseudo-arclength stepping only
    begins from the THIRD point onward, once an initial secant direction
    exists. If `ds` is not given, it defaults to the distance between
    these first two bootstrap points, a natural, self-consistent choice
    of step size for the arclength stepping that follows.

    STEP-SIZE CONTROL: if the corrector fails to converge at a given
    step, `ds` is HALVED and the step retried (up to `max_step_halvings`
    times) before giving up on that point entirely; `ds` is NOT grown
    back afterward (a deliberate simplicity choice -- aggressive
    step-growth heuristics are their own tuning problem, not needed to
    demonstrate the core "can continue past a fold" capability this
    function exists for). If a point still fails to converge after all
    halvings, tracing STOPS there -- remaining requested points are left
    `NaN`/`False`, reported honestly, matching `solve_nnm_backbone`'s
    own convention rather than silently padding or extrapolating.

    A SECOND genuine numerical wrinkle, found and fixed BEFORE this
    function's own real-fixture validation (not left for a caller to
    discover): the raw unknown vector mixes amplitude-sized components
    (`Z`, `a1` -- typically small) with `omega` (typically much larger
    in absolute terms). An UNSCALED Euclidean arclength metric is
    dominated by whichever happens to have the larger raw magnitude --
    `omega` in practice -- which starves the amplitude direction of
    "step budget" and makes tracing crawl. Classical FEA arc-length/Riks
    solvers face an analogous mixed-units problem (a load FACTOR next to
    displacement DOFs) but `fea_engine`'s own solver sidesteps it
    entirely rather than rescale: its arclength radius is defined purely
    from the DISPLACEMENT increment's own norm (the load factor is
    excluded from the metric, solved for afterward via a separate
    quadratic constraint equation) -- a choice that works there because
    displacements are the dominant, physically meaningful quantity and
    the load factor is a comparatively minor bookkeeping unknown. That
    option is not available here: `omega` and amplitude are BOTH
    actively-varying, physically meaningful unknowns along an NNM
    backbone (the whole curve is "frequency vs. amplitude"), so
    excluding either from the metric would bias the traced path, not
    just avoid a units mismatch. This function instead explicitly
    RESCALES `omega` before combining it with `Z`/`a1` in the metric:
    internally, all tangent/predictor/arclength-constraint bookkeeping
    is done on `w = [Z_free_all, omega/omega_scale]`, not the physical
    `v = [Z_free_all, omega]` -- `omega_scale` defaults to the master
    mode's own LINEAR natural frequency (`sqrt(Lambda[master_mode])`), a
    natural, dimensionally sensible non-dimensionalization. The
    equilibrium residual/Jacobian are still evaluated in PHYSICAL
    `omega` internally (`_augmented_residual_jacobian_scaled` converts
    back before calling `hb.apply_A`/AFT, then chain-rules the returned
    Jacobian's last column) -- this scaling changes only the ARCLENGTH
    METRIC, not the physics being solved. See
    `docs/nonlinear_surrogate_rom_roadmap.md` Section 11 for the measured
    before/after this fix made on this module's own validation.

    Parameters
    ----------
    hb : HarmonicBalanceSystem
    force_model : nonlinear_rom.ReducedForceModel
    a0 : float
        Starting master-mode first-harmonic amplitude (the bootstrap
        point) -- same physical quantity `q_amplitude_range[0]` is for
        `solve_nnm_backbone`.
    n_points : int, default 40
        Total points requested, INCLUDING the two bootstrap points.
    direction : float, default 1.0
        Sign (and, if `ds` is None, an implicit small relative scale via
        `_newton_fixed_amplitude`'s own step) of the initial bootstrap
        step in amplitude, away from `a0`. Only sets the STARTING
        direction -- once arclength stepping begins, the traced curve's
        actual direction (including through a fold, where amplitude
        itself may locally reverse) is determined by the tangent, not by
        this argument.
    ds : float, optional
        Arclength step size. Defaults to the distance between the two
        bootstrap points (see above).
    domain, master_mode, omega0, jacobian, tol, max_iter :
        Same meaning as `solve_nnm_backbone`'s own parameters.
    max_step_halvings : int, default 4
    omega_scale : float, optional
        Characteristic frequency used to non-dimensionalize `omega`
        before it enters the arclength metric (see above). Defaults to
        `sqrt(hb.Lambda[master_mode])` -- the master mode's own linear
        natural frequency.

    Returns
    -------
    amplitudes : (n_points,) ndarray
        The master mode's own first-harmonic amplitude AT EACH TRACED
        POINT -- unlike `solve_nnm_backbone`'s `amplitudes` (a fixed,
        externally chosen grid), this is now a MEASURED OUTPUT, since
        amplitude is no longer prescribed -- it can be, and near a fold
        generally will be, non-monotonic across `n_points`.
    omegas : (n_points,) ndarray (NaN at any point not reached)
    Z_hist : (n_points, n_coeffs, n_modes) ndarray
    converged : (n_points,) bool ndarray
    """
    if domain not in ("q_l", "q_nl"):
        raise ValueError(f"domain={domain!r} must be 'q_l' or 'q_nl'")
    n, NC = hb.n, hb.n_coeffs
    NCn = NC * n
    if not (0 <= master_mode < n):
        raise ValueError(f"master_mode={master_mode} out of range for n={n} modes")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2 (need both bootstrap points), got {n_points}")

    a1_flat = 1 * n + master_mode
    b1_flat = 2 * n + master_mode
    free_idx_all = np.array([i for i in range(NCn) if i != b1_flat])   # a1 now free
    a1_pos = int(np.searchsorted(free_idx_all, a1_flat))

    if omega0 is None:
        omega0 = float(np.sqrt(hb.Lambda[master_mode]))
    if omega_scale is None:
        omega_scale = float(np.sqrt(hb.Lambda[master_mode]))
    omega_scale = max(abs(omega_scale), 1e-300)

    amplitudes = np.full(n_points, np.nan)
    omegas = np.full(n_points, np.nan)
    Z_hist = np.zeros((n_points, NC, n))
    converged = np.zeros(n_points, dtype=bool)

    def _pack_w(Z_free_no_a1, a1_val, omega_val):
        v = np.empty(NCn - 1)
        no_a1_idx = np.array([i for i in range(len(free_idx_all)) if i != a1_pos])
        v[no_a1_idx] = Z_free_no_a1
        v[a1_pos] = a1_val
        return np.concatenate([v, [omega_val / omega_scale]])   # w-space: omega rescaled

    def _record(idx, w):
        v = w.copy()
        v[-1] = w[-1] * omega_scale   # unscale back to physical omega
        Z_flat = np.zeros(NCn)
        Z_flat[free_idx_all] = v[:-1]
        Z_flat[b1_flat] = 0.0
        Z_hist[idx] = Z_flat.reshape(NC, n)
        amplitudes[idx] = v[a1_pos]
        omegas[idx] = v[-1]

    # --- bootstrap point 0
    Zfree0 = np.zeros(NCn - 2)
    Zfree0_out, om0, conv0 = _newton_fixed_amplitude(
        hb, force_model, a0, Zfree0, omega0, master_mode, domain, jacobian, tol, max_iter)
    if not conv0:
        return amplitudes, omegas, Z_hist, converged   # can't even bootstrap -- all NaN/False, reported honestly
    w0 = _pack_w(Zfree0_out, a0, om0)
    _record(0, w0)
    converged[0] = True

    # --- bootstrap point 1: a small step in amplitude away from a0
    step0 = ds if ds is not None else max(abs(a0) * 0.05, 1e-3)
    a1_target = a0 + direction * step0
    Zfree1_out, om1, conv1 = _newton_fixed_amplitude(
        hb, force_model, a1_target, Zfree0_out, om0, master_mode, domain, jacobian, tol, max_iter)
    if not conv1:
        return amplitudes, omegas, Z_hist, converged   # only the first point was reached
    w1 = _pack_w(Zfree1_out, a1_target, om1)
    _record(1, w1)
    converged[1] = True

    t_hat = w1 - w0
    t_norm = np.linalg.norm(t_hat)
    if t_norm < 1e-300:
        return amplitudes, omegas, Z_hist, converged   # degenerate bootstrap step -- stop, reported honestly
    t_hat = t_hat / t_norm
    if ds is None:
        ds = t_norm
    ds = max(abs(ds), 1e-10)

    # --- genuine pseudo-arclength stepping for the remaining points
    # (all in w-space, the rescaled [Z_free_all, omega/omega_scale] metric)
    w_prev = w1
    for p in range(2, n_points):
        ds_try = ds
        accepted = False
        w_new = None
        for _halving in range(max_step_halvings + 1):
            w_pred = w_prev + ds_try * t_hat
            w_new, conv = _arclength_corrector(
                hb, force_model, w_pred, w_prev, t_hat, ds_try,
                free_idx_all, master_mode, domain, jacobian, tol, max_iter, omega_scale)
            if conv:
                accepted = True
                break
            ds_try *= 0.5
        if not accepted:
            break   # stop tracing here -- remaining points stay NaN/False

        t_new = w_new - w_prev
        t_new_norm = np.linalg.norm(t_new)
        if t_new_norm < 1e-300:
            break
        t_new = t_new / t_new_norm
        if np.dot(t_new, t_hat) < 0:
            t_new = -t_new   # keep the tangent oriented FORWARD along the curve

        _record(p, w_new)
        converged[p] = True
        w_prev, t_hat, ds = w_new, t_new, ds_try

    return amplitudes, omegas, Z_hist, converged
