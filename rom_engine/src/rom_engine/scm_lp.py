"""
scm_lp.py -- the genuine classical Successive Constraint Method (SCM;
Huynh, Rozza, Sen, Patera 2007): a rigorous lower bound on
sigma_min(A(omega)) computed via a small ONLINE LINEAR PROGRAM over
offline-selected reference ("control") points, rather than
scm.py's simplified single-Lipschitz-constant construction. See
docs/phase4_error_bounds_greedy_roadmap.md Section 9 for the full
derivation and the honest scoping decision this module makes -- read
that before using this module for anything beyond what its docstrings
below already say.

Scoping decision, in brief (full reasoning: Section 9 of the roadmap
doc above): the sharper "natural-norm" SCM (Chen et al. 2010/2015) is
NOT implemented here, because its literature is written for real
symmetric/real non-symmetric operators, and rom_engine's
A(omega) = -omega^2 M + i*omega*C + K is complex and non-Hermitian for
any omega with damping present -- extending the natural-norm
construction to this case requires an unverified real-part convention
neither source paper treats explicitly. Rather than implement an
unverified "certified" bound (worse than an honestly narrow one, if
subtly wrong), this module implements the ORIGINAL classical SCM,
applied via the standard "SCM-squared" reduction:

    sigma_min(A(omega))^2 = lambda_min(A(omega)^H A(omega))

A(omega)^H A(omega) is REAL, SYMMETRIC, and PSD by construction for any
A(omega) -- complex and non-Hermitian or not -- so this sidesteps the
complex-operator ambiguity entirely and lands exactly in the setting
the classical SCM was designed for. The literature's own caveat about
this reduction (it blows up the online LP to Q(Q+1)/2 variables for
large Q) is a non-issue here: rom_engine's Q <= 3 ({M, C, K} or {M, K}
under proportional damping) gives at most 6 LP variables.

The affine expansion (real M, C, K, so A_q^H = A_q^T):

    A(omega)^H A(omega) = sum_pq conj(theta_p(omega)) theta_q(omega) A_p^T A_q
                         = sum_q  |theta_q(omega)|^2                * B_qq
                         + sum_{p<q} Re[conj(theta_p(omega)) theta_q(omega)] * B_pq

    B_qq = A_q^T A_q                      (q)
    B_pq = A_p^T A_q + A_q^T A_p    (p < q)

both REAL SYMMETRIC, giving Q(Q+1)/2 real-coefficient affine terms.

Online LP (classical SCM structure, scipy.optimize.linprog):
  - box bounds y_pq in [lambda_min(B_pq), lambda_max(B_pq)] (offline,
    Q(Q+1)/2 small dense eigendecompositions, done once) -- valid for
    ANY unit vector w, since a Rayleigh quotient always lies between a
    symmetric matrix's extreme eigenvalues;
  - one inequality constraint per stored reference omega_j:
    sum_pq c_pq(omega_j) y_pq >= alpha_j, where
    alpha_j = lambda_min(A(omega_j)^H A(omega_j)) was computed exactly
    offline (one SVD, the same cost scm.py's add_reference() already
    pays) -- valid because ANY vector's Rayleigh quotient at operator
    j can never be smaller than operator j's own smallest eigenvalue;
  - objective: minimize sum_pq c_pq(omega) y_pq.

The LP's optimal value is a rigorous lower bound on
lambda_min(A(omega)^H A(omega)); sqrt(max(0, that)) rigorously lower-
bounds sigma_min(A(omega)).

This module deliberately mirrors scm.SingularValueLowerBound's method
names (from_affine/add_reference/lower_bound/greedy_train) so the two
can be swapped in and compared directly on the same model -- see
tests/test_scm_lp.py, which measures this bound's useful radius
against scm.py's already-measured ~1e-7 one on the SAME reference
point of the SAME real fea_engine fixture, rather than assuming this
LP-based construction is automatically sharper.
"""
__author__ = "Abhijeet"
import numpy as np
from scipy.optimize import linprog


def _pair_indices(Q):
    """[(0,0), (0,1), ..., (Q-1,Q-1)] -- the Q(Q+1)/2 (p, q) pairs with
    p <= q, in a fixed order shared by every array this module builds."""
    return [(p, q) for p in range(Q) for q in range(p, Q)]


class LPSingularValueLowerBound:
    """A certified lower bound on sigma_min(A(omega)) for an affine
    A(omega) = sum_q theta_q(omega) A_q, computed via the genuine
    classical Successive Constraint Method's online linear program
    (see module docstring for the SCM-squared reduction this uses, and
    docs/phase4_error_bounds_greedy_roadmap.md Section 9 for the full
    derivation and honest scoping note).

    Parameters
    ----------
    components : sequence of ndarray (n_dof, n_dof), REAL
        The same affine component matrices (e.g. [M, K] or [M, C, K])
        an affine.AffineDecomposition for this A(omega) was built from
        -- see from_affine() to build directly from one instead.
    theta_func : callable
        The same theta_func(mu) -> coefficients (possibly complex)
        used by the corresponding AffineDecomposition.
    """

    def __init__(self, components, theta_func):
        self.components = [np.asarray(Aq, dtype=float) for Aq in components]
        self.theta_func = theta_func
        self.Q = len(self.components)
        self.pairs = _pair_indices(self.Q)
        self.n_pairs = len(self.pairs)

        # Offline, ONE-TIME cost: build every B_pq and its eigenvalue
        # range -- Q(Q+1)/2 real symmetric (n_dof, n_dof) matrices and
        # dense eigendecompositions, done once regardless of how many
        # omega queries follow.
        self._B = []
        y_min = np.empty(self.n_pairs)
        y_max = np.empty(self.n_pairs)
        for idx, (p, q) in enumerate(self.pairs):
            Ap, Aq = self.components[p], self.components[q]
            if p == q:
                B = Ap.T @ Aq
            else:
                B = Ap.T @ Aq + Aq.T @ Ap
            B = 0.5 * (B + B.T)  # symmetrize away any roundoff asymmetry
            self._B.append(B)
            eigvals = np.linalg.eigvalsh(B)
            y_min[idx] = eigvals[0]
            y_max[idx] = eigvals[-1]
        self.box_bounds = list(zip(y_min.tolist(), y_max.tolist()))
        self._y_min = y_min
        self._y_max = y_max

        self.reference_omegas = []      # list of omega_j
        self.reference_c = []           # list of c(omega_j), the pair-coefficient vector
        self.reference_alpha = []       # list of TRUE lambda_min(A(omega_j)^H A(omega_j))

    @classmethod
    def from_affine(cls, affine):
        """Build directly from an already-constructed
        affine.AffineDecomposition (e.g. rom.affine on a FrequencyROM)
        -- reuses its components/theta_func rather than requiring the
        caller to pass them again."""
        return cls(affine.components, affine.theta_func)

    def _theta(self, omega):
        return np.asarray(self.theta_func(omega)).ravel()

    def _coefficients(self, omega):
        """c_pq(omega) for every stored (p, q) pair, in self.pairs
        order: |theta_q|^2 on the diagonal, Re[conj(theta_p) theta_q]
        off it -- the real coefficients A(omega)^H A(omega)'s affine
        expansion needs (see module docstring)."""
        theta = self._theta(omega)
        c = np.empty(self.n_pairs)
        for idx, (p, q) in enumerate(self.pairs):
            if p == q:
                c[idx] = (theta[q] * np.conj(theta[q])).real
            else:
                c[idx] = (np.conj(theta[p]) * theta[q]).real
        return c

    def _assemble_AhA(self, omega):
        """Full-order (n_dof, n_dof) A(omega)^H A(omega), built the
        direct way (not via the affine B_pq expansion) -- used only by
        add_reference() to get the TRUE alpha_j, independent of the
        expansion this class's online path relies on, so a bug in the
        expansion couldn't silently self-certify."""
        theta = self._theta(omega)
        A = theta[0] * self.components[0]
        for q in range(1, self.Q):
            A = A + theta[q] * self.components[q]
        return A.conj().T @ A

    # -----------------------------------------------------------------
    # Offline: add a reference point (expensive -- one full eigenvalue
    # computation of the (n_dof, n_dof) A(omega_j)^H A(omega_j) -- do
    # this rarely, only at points a greedy procedure has identified as
    # worth the cost. Same cost profile as scm.py's add_reference()).
    # -----------------------------------------------------------------
    def add_reference(self, omega):
        """Full-order-compute the TRUE
        lambda_min(A(omega)^H A(omega)) = sigma_min(A(omega))^2 at this
        omega and store it (plus c(omega)) as a reference/control point
        for lower_bound()'s LP at OTHER omegas. Returns the true
        sigma_min(A(omega)) directly, for parity with
        scm.SingularValueLowerBound.add_reference()."""
        AhA = self._assemble_AhA(omega)
        AhA = 0.5 * (AhA + AhA.conj().T)
        alpha = float(np.linalg.eigvalsh(AhA)[0])
        alpha = max(alpha, 0.0)  # PSD by construction; clip tiny negative roundoff
        self.reference_omegas.append(float(np.real(omega)))
        self.reference_c.append(self._coefficients(omega))
        self.reference_alpha.append(alpha)
        return float(np.sqrt(alpha))

    # -----------------------------------------------------------------
    # Online: the classical SCM's LP, solved fresh at each omega using
    # every reference added so far. Q(Q+1)/2 variables, that many box
    # constraints plus one per reference -- sub-millisecond for
    # rom_engine's Q <= 3.
    # -----------------------------------------------------------------
    def lower_bound(self, omega):
        """sqrt(max(0, LP optimum)), where the LP minimizes
        sum_pq c_pq(omega) y_pq subject to the box bounds (valid for
        ANY unit vector) and, for every stored reference omega_j,
        sum_pq c_pq(omega_j) y_pq >= alpha_j (valid because a Rayleigh
        quotient at operator j can never undercut operator j's own
        smallest eigenvalue) -- the genuine classical SCM online LP
        (see module docstring). RIGOROUS for any reference set; more
        references only ever tighten this, never invalidate it. With
        no references yet, returns 0.0 (trivially valid, uninformative
        -- same convention as scm.SingularValueLowerBound)."""
        if not self.reference_omegas:
            return 0.0
        c = self._coefficients(omega)
        A_ub = -np.array(self.reference_c)          # (n_ref, n_pairs)
        b_ub = -np.array(self.reference_alpha)       # (n_ref,)
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=self.box_bounds,
                          method="highs")
        if not result.success:
            # Infeasible/unbounded LP shouldn't happen for a MINIMIZATION
            # over a bounded box intersected with reference half-spaces --
            # if it ever does, fail safe with the trivial, always-valid
            # bound rather than silently returning something wrong.
            return 0.0
        alpha_lb = max(float(result.fun), 0.0)
        return float(np.sqrt(alpha_lb))

    def upper_bound(self, omega):
        """A companion upper bound, via a closed-form (NOT LP-based)
        box-only maximization: sum_pq c_pq(omega) * (y_max_pq if
        c_pq(omega) >= 0 else y_min_pq). This DROPS the reference-point
        constraints lower_bound()'s LP uses, so it is a looser upper
        bound than the full constrained maximum would give -- but
        dropping constraints from a maximization can only ever
        INCREASE the true optimum, so this remains a valid, rigorous
        upper bound, just not the tightest one available in principle.

        This is a deliberate choice, not an oversight: maximizing
        sum_pq c_pq(omega) y_pq over the SAME box + reference-
        constraint feasible region lower_bound() uses was tried via
        linprog first, and HiGHS was observed to spuriously report
        "unbounded" on this package's real cantilever fixture -- a
        numerically provable impossibility, since the feasible region
        is a bounded box intersected with half-spaces (hence compact),
        caused by the ~1e23-magnitude coefficient range this package's
        real FE stiffness matrix produces after the SCM-squared
        reduction's squaring (see module docstring) exceeding HiGHS's
        internal numerical-stability assumptions. Rather than paper
        over that with fragile rescaling (tried, and shown to just
        relocate the same extreme magnitude from the bounds into the
        objective/constraint-matrix data instead of removing it), this
        closed form sidesteps the solver entirely for the one place
        (the greedy-selection heuristic) where a looser, still-valid
        bound is an acceptable trade -- lower_bound() itself, the
        actually CERTIFIED quantity, keeps the full LP."""
        if not self.reference_omegas:
            return np.inf
        c = self._coefficients(omega)
        per_term = np.where(c >= 0, c * self._y_max, c * self._y_min)
        alpha_ub = max(float(np.sum(per_term)), 0.0)
        return float(np.sqrt(alpha_ub))

    # -----------------------------------------------------------------
    # Greedy reference-point selection -- directly analogous to
    # scm.SingularValueLowerBound.greedy_train().
    # -----------------------------------------------------------------
    def greedy_train(self, candidate_omegas, tol=0.05, max_references=15, seed_omega=None):
        """Adaptively add reference points where the certified gap
        (upper_bound - lower_bound) is currently largest -- see
        scm.SingularValueLowerBound.greedy_train()'s docstring, which
        this mirrors exactly (same stopping rule, same history format),
        for direct comparability between the two bounds' greedy runs
        on the same candidate pool."""
        candidate_omegas = np.asarray(candidate_omegas, dtype=float)
        if not self.reference_omegas:
            if seed_omega is None:
                seed_omega = float(candidate_omegas[len(candidate_omegas) // 2])
            self.add_reference(seed_omega)

        history = []
        while len(self.reference_omegas) < max_references:
            gaps = []
            for om in candidate_omegas:
                lb = self.lower_bound(om)
                ub = self.upper_bound(om)
                rel_gap = (ub - lb) / ub if np.isfinite(ub) and ub > 0 else np.inf
                gaps.append(rel_gap)
            gaps = np.array(gaps)
            worst_idx = int(np.argmax(gaps))
            worst_gap = float(gaps[worst_idx])
            worst_omega = float(candidate_omegas[worst_idx])
            history.append((worst_omega, worst_gap))

            if worst_gap < tol:
                break
            self.add_reference(worst_omega)

        return history


def certified_error_bound(rom, scm_bound, omega, F):
    """||x_true(omega) - x_ROM(omega)|| <= residual_norm(omega) /
    sigma_min(A(omega)), evaluated with an LPSingularValueLowerBound's
    RIGOROUS lower bound on sigma_min(A(omega)) -- the LP-based
    analogue of scm.certified_error_bound(), identical in structure
    (only the bound source differs), so the two can be compared
    directly on the same (rom, omega, F).

    Parameters
    ----------
    rom : frequency.FrequencyROM
    scm_bound : LPSingularValueLowerBound
        Should already have at least one reference point added.
    omega, F :
        As in FrequencyROM.residual_norm().

    Returns
    -------
    float
        A rigorous upper bound on the true error, or +inf if
        scm_bound's lower bound at this omega is <= 0.
    """
    residual = rom.residual_norm(omega, F)
    beta_lb = scm_bound.lower_bound(omega)
    if beta_lb <= 0:
        return float("inf")
    return residual / beta_lb
