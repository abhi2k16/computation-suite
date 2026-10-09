"""
scm.py -- a genuinely CERTIFIED (rigorously provable, not just a
practical indicator) lower bound on sigma_min(A(omega)), the quantity
that turns FrequencyROM.residual_norm() into a true a posteriori error
BOUND rather than an estimate: ||x_true(omega) - x_ROM(omega)|| <=
residual_norm(omega) / sigma_min(A(omega)).

Honest relationship to the classical Successive Constraint Method
(SCM): the reduced-basis-method literature's SCM (Huynh, Rozza, Sen,
Patera 2007; the "natural-norm" SCM of Chen et al. 2010 for the
general non-coercive/non-symmetric case, which is the relevant variant
for our complex, non-Hermitian A(omega)) computes a rigorous lower
bound via a small LINEAR PROGRAM at each query, built from an
offline-selected set of "control points" and continuity/stability
constraints on an "attainable set" in R^(Q^2). That is the sharpest
version of this idea, but its LP formulation for a general complex,
non-normal affine operator is substantial machinery to implement
correctly from scratch (see
docs/phase4_error_bounds_greedy_roadmap.md Section 3's caveat about
this).

THIS module implements a SIMPLER, still fully rigorous, SCM-FAMILY
bound instead, built on a single classical matrix-perturbation fact
(Weyl/Mirsky's inequality for singular values -- see e.g. Horn &
Johnson, "Matrix Analysis", or Golub & Van Loan): for any two matrices
A, B of the same shape,

    |sigma_min(A) - sigma_min(B)| <= ||A - B||_2

i.e. the smallest singular value is 1-Lipschitz with respect to
operator-norm perturbation. Since A(omega) is affine,
A(omega) - A(omega_j) = sum_q (theta_q(omega) - theta_q(omega_j)) A_q,
so by the triangle inequality for the operator norm,

    ||A(omega) - A(omega_j)||_2 <= sum_q |theta_q(omega) - theta_q(omega_j)| * ||A_q||_2

Combining the two gives a computable, RIGOROUS lower bound at any
omega from a single "reference" point omega_j where sigma_min(A(omega_j))
has been computed exactly (offline, one full SVD -- expensive, done
rarely):

    sigma_min(A(omega)) >= sigma_min(A(omega_j)) - sum_q |theta_q(omega)-theta_q(omega_j)| * ||A_q||_2

Taking the MAX of this over several reference points (each contributing
a valid lower bound) gives the tightest bound available from that
reference set -- and the corresponding MIN of the mirror-image upper
bound gives a matching upper bound, whose gap to the lower bound is
exactly what a greedy procedure (Section "greedy reference selection"
below) can shrink by adding more references where the gap is largest --
the same OFFLINE-EXPENSIVE / ONLINE-CHEAP philosophy real SCM uses,
just with a provably-correct-by-inspection bound instead of an LP.

This is deliberately named "scm.py" (this IS an SCM-family method --
lower-bounding a parametric stability/coercivity-like quantity via
offline-selected reference/control points) while being explicit,
here and in every docstring below, that it is a SIMPLIFIED variant,
not the classical LP-based algorithm -- so nothing here should be
confused with, or cited as, the Huynh/Rozza/Sen/Patera or Chen et al.
methods themselves.
"""
__author__ = "Abhijeet"
import numpy as np


class SingularValueLowerBound:
    """A certified, Lipschitz-perturbation-based lower bound on
    sigma_min(A(omega)) for an affine A(omega) = sum_q theta_q(omega) A_q.

    Parameters
    ----------
    components : sequence of ndarray (n_dof, n_dof)
        The SAME affine component matrices an
        affine.AffineDecomposition for this A(omega) was built from
        (e.g. [M, K] for the proportional-damping case, [M, C, K] for
        the general case) -- see from_affine() for building this
        directly from an existing AffineDecomposition/FrequencyROM
        instead of passing components again by hand.
    theta_func : callable
        The SAME theta_func(mu) -> coefficients used by the
        corresponding AffineDecomposition.
    """

    def __init__(self, components, theta_func):
        self.components = [np.asarray(Kq) for Kq in components]
        self.theta_func = theta_func
        self.Q = len(self.components)
        # Offline, ONE-TIME cost: the operator (spectral) norm of each
        # component, ||A_q||_2 -- the largest singular value. Q of
        # these, each an O(n_dof^3) SVD, but done ONCE regardless of
        # how many omega queries follow.
        self.component_norms = np.array([
            np.linalg.norm(Aq, ord=2) for Aq in self.components
        ])
        self.reference_omegas = []       # list of omega_j
        self.reference_theta = []        # list of theta(omega_j), cached
        self.reference_sigma_min = []    # list of TRUE sigma_min(A(omega_j))

    @classmethod
    def from_affine(cls, affine):
        """Build directly from an already-constructed
        affine.AffineDecomposition (e.g. rom.affine on a FrequencyROM)
        -- reuses its components/theta_func rather than requiring the
        caller to pass them again."""
        return cls(affine.components, affine.theta_func)

    def _theta(self, omega):
        return np.asarray(self.theta_func(omega)).ravel()

    # -----------------------------------------------------------------
    # Offline: add a reference point (expensive -- one full SVD of the
    # (n_dof, n_dof) A(omega_j) -- do this rarely, only at points a
    # greedy procedure has identified as worth the cost).
    # -----------------------------------------------------------------
    def add_reference(self, omega):
        """Full-order-compute the TRUE sigma_min(A(omega)) at this
        omega (one O(n_dof^3) SVD) and store it as a reference point
        for lower_bound()/upper_bound() at OTHER, nearby omegas.
        Returns the computed sigma_min, in case the caller wants it
        directly too."""
        theta = self._theta(omega)
        A = theta[0] * self.components[0]
        for q in range(1, self.Q):
            A = A + theta[q] * self.components[q]
        sigma_min = float(np.linalg.svd(A, compute_uv=False).min())
        self.reference_omegas.append(float(np.real(omega)))
        self.reference_theta.append(theta)
        self.reference_sigma_min.append(sigma_min)
        return sigma_min

    # -----------------------------------------------------------------
    # Online: cheap (O(Q) per reference, no full-order operation at
    # all) evaluation at any omega, using every reference added so far.
    # -----------------------------------------------------------------
    def lower_bound(self, omega):
        """max_j [ sigma_min(A(omega_j)) - sum_q |theta_q(omega) -
        theta_q(omega_j)| * ||A_q||_2 ], clipped at 0 (sigma_min is
        never negative, so 0 is always a trivially valid, if
        uninformative, lower bound). RIGOROUS for any set of
        references with correctly computed sigma_min -- more
        references (spread appropriately, see greedy_train() below)
        only ever TIGHTEN this, never invalidate it."""
        if not self.reference_omegas:
            return 0.0
        theta = self._theta(omega)
        best = 0.0
        for theta_j, sigma_j in zip(self.reference_theta, self.reference_sigma_min):
            perturbation = np.sum(np.abs(theta - theta_j) * self.component_norms)
            best = max(best, sigma_j - perturbation)
        return float(best)

    def upper_bound(self, omega):
        """min_j [ sigma_min(A(omega_j)) + sum_q |theta_q(omega) -
        theta_q(omega_j)| * ||A_q||_2 ] -- the mirror-image bound from
        the SAME Lipschitz fact, used only to drive greedy reference
        selection (the gap upper_bound - lower_bound is a cheap,
        rigorous indicator of how much a NEW reference point could
        possibly help at this omega -- unlike frequency.py's
        hierarchical indicator, this gap is itself certified, not a
        heuristic)."""
        if not self.reference_omegas:
            return np.inf
        theta = self._theta(omega)
        best = np.inf
        for theta_j, sigma_j in zip(self.reference_theta, self.reference_sigma_min):
            perturbation = np.sum(np.abs(theta - theta_j) * self.component_norms)
            best = min(best, sigma_j + perturbation)
        return float(best)

    # -----------------------------------------------------------------
    # Greedy reference-point selection
    # -----------------------------------------------------------------
    def greedy_train(self, candidate_omegas, tol=0.05, max_references=15, seed_omega=None):
        """Adaptively add reference points where the CERTIFIED gap
        (upper_bound - lower_bound) is currently largest, i.e. where
        the bound is least informative -- directly analogous to
        greedy.greedy_train_frequency_basis()'s selection logic, but
        driven by a rigorous gap instead of a heuristic indicator,
        since that gap is available here for free from the same
        Lipschitz argument the bounds themselves use.

        Parameters
        ----------
        candidate_omegas : array-like of float
            The pool of omegas this loop may pick references from.
        tol : float
            Stop once the worst RELATIVE gap
            ((upper_bound - lower_bound) / upper_bound) across all
            candidates falls below this.
        max_references : int
            Hard cap on the number of (expensive, O(n_dof^3)) full SVDs
            this loop will perform.
        seed_omega : float, optional
            Where to add the FIRST reference (no bound exists before
            at least one reference is added, since lower_bound()
            returns the trivial 0 with none). Defaults to the middle
            of candidate_omegas.

        Returns
        -------
        history : list of (omega, gap) -- every candidate selected, in
            order, with the relative gap that justified selecting it,
            kept for the same audit/plotting purpose
            greedy_train_frequency_basis()'s history serves.
        """
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
    sigma_min(A(omega)), evaluated with scm_bound's RIGOROUS lower
    bound on sigma_min(A(omega)) in place of the true (expensive)
    value -- the one genuinely CERTIFIED bound in this package (every
    other error signal -- error_estimate()/residual_norm(),
    hierarchical_error_indicator() -- is explicitly an estimate or
    indicator, not a bound; see their docstrings).

    Parameters
    ----------
    rom : frequency.FrequencyROM
        Must have been built from the SAME (M, K, C/rayleigh) as
        scm_bound (typically: build scm_bound via
        SingularValueLowerBound.from_affine(rom.affine)).
    scm_bound : SingularValueLowerBound
        Should already have at least one reference point added (via
        add_reference() or greedy_train()) -- with none, lower_bound()
        returns the trivial 0, making this function return +inf
        (a valid but useless bound, correctly signaling "no
        information yet" rather than silently returning something
        that looks like a real number but isn't backed by anything).
    omega, F :
        As in FrequencyROM.residual_norm().

    Returns
    -------
    float
        A RIGOROUS upper bound on the true error, or +inf if
        scm_bound's lower bound at this omega is <= 0 (uninformative;
        happens near resonance more often than not with a small
        reference set -- see docs/phase4_error_bounds_greedy_roadmap.md
        Section 3's discussion of why this whole family of bounds is
        fundamentally weaker there, an honest limitation this function
        surfaces as +inf rather than hides).
    """
    residual = rom.residual_norm(omega, F)
    beta_lb = scm_bound.lower_bound(omega)
    if beta_lb <= 0:
        return float("inf")
    return residual / beta_lb
