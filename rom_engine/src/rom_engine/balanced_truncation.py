"""
balanced_truncation.py -- balanced truncation (Besselink et al. 2013,
Section 4 / eqs. 37-58), the paper's "systems and control" family, and
the one of its three numerically-applied methods that stays accurate
FAR from any single expansion point (unlike krylov.py's moment
matching, which is only locally accurate) because it explicitly uses
WHERE the inputs and outputs are, not just the system's own internal
dynamics.

The idea: solve two continuous-time Lyapunov equations for the
controllability Gramian P and observability Gramian Q of the first-order
state-space form (state_space.py, form="A" -- see below for why this
form specifically),

    A P + P A^T + B B^T = 0          (controllability)
    A^T Q + Q A + Cout^T Cout = 0    (observability)

find a "balancing" state transformation T under which the transformed
Gramians become equal and diagonal, diag(sigma_1 >= sigma_2 >= ...) --
the HANKEL SINGULAR VALUES, a coordinate-free measure of how much each
internal state simultaneously matters for controllability AND
observability, i.e. for the actual input-output behavior specifically
(this is exactly why BT, unlike mode displacement or Krylov moment
matching, "knows" about the ports) -- and truncate to the r states with
the largest sigma_i.

This module implements the numerically preferred SQUARE-ROOT method
(Cholesky/eigen factors of the Gramians, an SVD of their cross product,
never an explicit eigendecomposition of the Gramian PRODUCT P@Q, which
is the classical textbook presentation but genuinely ill-conditioned
when the Hankel singular values span a wide range -- exactly what real
structural models, including this paper's own benchmark, produce).

Why form="A" (the explicit M-inverted state-space form), not form="E"
like krylov.py: scipy.linalg.solve_continuous_lyapunov only solves the
ORDINARY Lyapunov equation above, not a generalized (E-weighted) one --
there is no separate scipy entry point for that. Since balanced
truncation already pays for a dense O(n^3) Lyapunov solve regardless,
paying for one more dense M-solve to get to the ordinary form first adds
no new order-of-magnitude cost, unlike in krylov.py where avoiding the
M-inversion was worth it for its own sake (see state_space.py's module
docstring for the full comparison).

A structural guarantee this module's tests hold it to, and that
krylov.KrylovROM explicitly does NOT share: for a stable full-order
system, a balanced-and-truncated reduced model is ALWAYS stable too --
this is a genuine theorem (Pernebo & Silverman 1982), not an empirical
property, so BalancedTruncationROM.is_stable() should always report
True and is kept mainly as a regression guard on that theorem, not a
real per-call check the way krylov.KrylovROM's version is.

Extensions the paper mentions but does not itself run numerically
(coprime-factorization BT for unstable systems, passivity-preserving
BT, optimal Hankel norm approximation) were originally out of scope --
see docs/classical_mor_roadmap.md Section 2c/7 -- but ALL THREE are now
covered, two of them elsewhere in this package rather than as sibling
classes here: singular perturbation approximation (SPA,
SingularPerturbationROM below -- see its own docstring and
docs/classical_mor_roadmap.md Section 8) and frequency-weighted BT
(FrequencyWeightedBalancedTruncationROM below -- see its own docstring
and Section 10) ARE sibling classes in THIS file, since both reuse the
SAME square-root balancing machinery this module already implements
(_balance_from_gramians()), just fed different Gramians. Passivity
preservation was instead scoped into its own module, passivity.py (a
closed-form theorem plus a diagnostic, not a new ROM class -- see that
module's docstring and Section 12 for why). Optimal Hankel norm
approximation was also scoped into its own module,
hankel_norm.py's OptimalHankelNormROM (see its own docstring and
Section 14) -- it reuses hankel_singular_values() for its balancing
step but its Gamma-inversion/Schur-plus-Sylvester construction is
algorithmically unlike anything else in this file, the same reasoning
that already keeps krylov.py separate from this module.

A PRACTICAL SCALE NOTE, found while validating this module against a
real fea_engine beam: don't call from_MCK() directly on a raw,
finely-meshed FE model's full free-dof system. Ordinary structural FE
discretizations (Euler-Bernoulli beam elements included) always carry a
very wide eigenvalue spread -- the highest, most-refined-element modes
sit many orders of magnitude above the lowest structural ones -- and the
two dense Lyapunov solves this module relies on (scipy's
solve_continuous_lyapunov, a dense Bartels-Stewart solver) lose real
accuracy as that spread grows, well before hitting scipy's raw size
ceiling. The fix is cheap and standard: modally truncate the raw model
down to a moderate size first (a plain eigh(K, M) on the lowest N modes
-- exactly what galerkin.GalerkinROM's own solve_modal() already does,
no new machinery needed), THEN balance-and-truncate that well-
conditioned intermediate model. See tests/test_balanced_truncation.py
for a worked example of this two-stage pattern, including the numerical
symptoms (an artificial noise floor a few orders of magnitude above
machine epsilon in the smallest computed Hankel singular values) that
show up if this step is skipped.
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, cholesky, eigh, LinAlgError

from .state_space import to_state_space
from .torch_linalg import _HAS_TORCH, _require_torch, torch_solve_continuous_lyapunov, torch_balance_from_gramians

# GPU/torch side-by-side path (Wave 9 addendum item 138, docs/
# consolidated_future_roadmap.md) -- see torch_linalg.py's own module
# docstring for the full design (why a shared helper module, what is
# and is not ported, the Kronecker-vectorization Lyapunov-solve method
# used in place of scipy's Bartels-Stewart). Every function below that
# gains a backend="numpy" (default, unchanged) / backend="torch" +
# device= parameter ALWAYS returns plain NumPy arrays regardless of
# backend -- torch is used only to accelerate the internal O(n^3)-and-
# up compute (the two Lyapunov solves plus the SVD), never changes this
# module's own data contract, since nothing downstream (BalancedTrunc
# ationROM/SingularPerturbationROM/FrequencyWeightedBalancedTruncation
# ROM/hankel_norm.OptimalHankelNormROM) is torch-native.


def controllability_gramian(A, B, backend="numpy", device="cpu"):
    """Solve A P + P A^T + B B^T = 0 for P. backend="numpy" (default,
    unchanged) uses scipy's dense Bartels-Stewart solver (LAPACK
    ?TRSYL). backend="torch" (Wave 9 addendum item 138) routes through
    torch_linalg.torch_solve_continuous_lyapunov() instead -- a
    genuinely different numerical method (Kronecker-sum vectorization,
    see that function's own docstring), cross-validated against the
    scipy path rather than assumed equivalent. Always returns a plain
    NumPy array. A must be Hurwitz-stable (all eigenvalues strictly
    negative real part) -- an undamped or marginally-damped model will
    make this equation singular or ill-posed; see state_space.
    to_state_space()'s own docstring note about needing real damping
    for this reason."""
    if backend not in ("numpy", "torch"):
        raise ValueError(
            f"controllability_gramian: unknown backend={backend!r} -- "
            f"expected 'numpy' (default) or 'torch'.")
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if backend == "torch":
        _require_torch()
        X = torch_solve_continuous_lyapunov(A, -(B @ B.T), device=device)
        return X.cpu().numpy()
    return solve_continuous_lyapunov(A, -(B @ B.T))


def observability_gramian(A, Cout, backend="numpy", device="cpu"):
    """Solve A^T Q + Q A + Cout^T Cout = 0 for Q -- the dual of
    controllability_gramian(), same solver dispatch, same backend=/
    device= convention, same A-must-be-stable requirement."""
    if backend not in ("numpy", "torch"):
        raise ValueError(
            f"observability_gramian: unknown backend={backend!r} -- "
            f"expected 'numpy' (default) or 'torch'.")
    A = np.asarray(A, dtype=float)
    Cout = np.asarray(Cout, dtype=float)
    if backend == "torch":
        _require_torch()
        X = torch_solve_continuous_lyapunov(A.T, -(Cout.T @ Cout), device=device)
        return X.cpu().numpy()
    return solve_continuous_lyapunov(A.T, -(Cout.T @ Cout))


def _gramian_square_root(P, jitter_tries=(0.0, 1e-12, 1e-9, 1e-6)):
    """A factor L with P = L @ L.T. Tries an ordinary (triangular)
    Cholesky factorization first -- the cheap, standard case for a
    genuinely positive-definite Gramian -- and falls back to an
    eigenvalue-based square root (clipping any tiny/negative
    eigenvalues to zero, which is expected: an eigenvalue near zero
    here means a nearly uncontrollable/unobservable state, exactly the
    kind of state balanced truncation is supposed to identify and
    discard, not a numerical failure to hide) if Cholesky fails because
    numerical noise from the Lyapunov solve left P not quite positive
    definite. The result L need not be triangular -- the square-root
    balancing algorithm below only uses P = L @ L.T, never L's own
    structure."""
    P = 0.5 * (P + P.T)   # symmetrize away solver round-off asymmetry
    last_err = None
    for jitter in jitter_tries:
        try:
            return cholesky(P + jitter * np.eye(P.shape[0]), lower=True)
        except LinAlgError as e:
            last_err = e
            continue
    eigvals, eigvecs = eigh(P)
    eigvals = np.clip(eigvals, 0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def _balance_from_gramians(P, Q, backend="numpy", device="cpu"):
    """The shared square-root balancing step: Gramian square roots
    P=Lp Lp^T, Q=Lq Lq^T, then an SVD of Lq^T Lp (NOT an
    eigendecomposition of P@Q -- see module docstring for why). Used by
    both hankel_singular_values() (P, Q = the plain, unweighted
    Gramians) and frequency_weighted_hankel_singular_values() (P, Q =
    the Enns frequency-weighted Gramians restricted to the plant's own
    states) -- the SAME balancing math either way, only the Gramians
    fed into it differ. Always returns the transform (unlike
    hankel_singular_values()'s optional return_transform -- both public
    callers decide for themselves whether to expose it).

    backend="numpy" (default, unchanged) / backend="torch" (Wave 9
    addendum item 138) -- torch path routes through torch_linalg.
    torch_balance_from_gramians(), same formulas, always returns
    NumPy arrays either way.

    Returns
    -------
    sigma : ndarray, descending
    T, Tinv : ndarray (n, n)
    """
    if backend == "torch":
        _require_torch()
        return torch_balance_from_gramians(P, Q, device=device)
    Lp = _gramian_square_root(P)
    Lq = _gramian_square_root(Q)

    U, sigma, Vt = np.linalg.svd(Lq.T @ Lp)
    sigma_safe = np.where(sigma > 1e-300, sigma, 1e-300)
    T = Lp @ Vt.T @ np.diag(sigma_safe ** -0.5)
    Tinv = np.diag(sigma_safe ** -0.5) @ U.T @ Lq.T
    return sigma, T, Tinv


def hankel_singular_values(A, B, Cout, return_transform=False, backend="numpy", device="cpu"):
    """The system's Hankel singular values, via the square-root method
    (_balance_from_gramians(), fed the plain, unweighted controllability/
    observability Gramians -- see frequency_weighted_hankel_singular_
    values() for the Enns frequency-weighted generalization of this
    same balancing step).

    Parameters
    ----------
    A, B, Cout : ndarray
        A stable state-space system's matrices (state_space.py's
        form="A" convention -- ordinary ODE form, no E).
    return_transform : bool
        If True, also return the balancing transform (T, Tinv) such
        that Tinv @ A @ T is a balanced realization (equal, diagonal
        Gramians = diag(hsv)) -- what BalancedTruncationROM uses
        internally to build its reduced model.
    backend : "numpy" (default) or "torch"
        Wave 9 addendum item 138: backend="torch" routes the two
        Lyapunov solves and the balancing SVD through torch_linalg.py
        (device= selects "cpu"/"cuda") -- accelerates the genuinely
        expensive, full-order-sized compute this function does. Always
        returns plain NumPy arrays either way; see torch_linalg.py's
        own module docstring for the method and its scope.

    Returns
    -------
    hsv : ndarray, descending
        Hankel singular values, sigma_1 >= sigma_2 >= ... >= 0.
    T, Tinv : ndarray (n, n), only if return_transform
    """
    if backend not in ("numpy", "torch"):
        raise ValueError(
            f"hankel_singular_values: unknown backend={backend!r} -- "
            f"expected 'numpy' (default) or 'torch'.")
    P = controllability_gramian(A, B, backend=backend, device=device)
    Q = observability_gramian(A, Cout, backend=backend, device=device)
    sigma, T, Tinv = _balance_from_gramians(P, Q, backend=backend, device=device)
    if not return_transform:
        return sigma
    return sigma, T, Tinv


def lowpass_weight(wc):
    """A first-order low-pass SISO weight W(s) = wc / (s + wc), in
    controllable canonical form, DC gain 1 -- a convenience for
    frequency_weighted_gramians()/FrequencyWeightedBalancedTruncationROM's
    Wi/Wo arguments when the goal is "emphasize accuracy below this
    frequency", without hand-deriving a state-space realization.

    Returns
    -------
    (Aw, Bw, Cw, Dw) : ndarray, each (1, 1) except Dw a float
    """
    wc = float(wc)
    Aw = np.array([[-wc]])
    Bw = np.array([[wc]])
    Cw = np.array([[1.0]])
    Dw = np.array([[0.0]])
    return Aw, Bw, Cw, Dw


def bandpass_weight(omega_n, zeta=0.1):
    """A second-order bandpass SISO weight
    W(s) = 2*zeta*omega_n*s / (s^2 + 2*zeta*omega_n*s + omega_n^2),
    peaking at s=i*omega_n, in controllable canonical form -- a
    convenience for emphasizing balanced-truncation accuracy near a
    specific frequency of interest (e.g. a structural resonance), the
    more structurally relevant case than lowpass_weight() for this
    package.

    Returns
    -------
    (Aw, Bw, Cw, Dw) : ndarray, each (2, 2)/(2, 1)/(1, 2) except Dw a
        (1, 1) zero
    """
    omega_n = float(omega_n)
    zeta = float(zeta)
    a1 = 2.0 * zeta * omega_n
    a0 = omega_n ** 2
    Aw = np.array([[0.0, 1.0], [-a0, -a1]])
    Bw = np.array([[0.0], [1.0]])
    Cw = np.array([[0.0, a1]])
    Dw = np.array([[0.0]])
    return Aw, Bw, Cw, Dw


def frequency_weighted_gramians(A, B, Cout, Wi=None, Wo=None, backend="numpy", device="cpu"):
    """Enns (1984) frequency-weighted controllability/observability
    Gramians, restricted to the plant's own n states -- see module
    docstring / docs/classical_mor_roadmap.md Section 10 for the
    cascade-system derivation. Either Wi or Wo (or both) may be None,
    degenerating that side back to Wi(s)=1 / Wo(s)=1 exactly (i.e. the
    plain, unweighted Gramian on that side).

    Parameters
    ----------
    A, B, Cout : ndarray
        The plant's state-space matrices (state_space.py form="A").
    Wi, Wo : (Aw, Bw, Cw, Dw) tuple of ndarray, or None
        SISO weight realizations -- see lowpass_weight()/
        bandpass_weight() for convenience constructors. Wi is applied
        BEFORE the plant (on the input side); Wo is applied AFTER the
        plant (on the output side).
    backend, device
        Wave 9 addendum item 138 -- passed through to the underlying
        controllability_gramian()/observability_gramian() calls (both
        potentially on an AUGMENTED, larger-than-n system when Wi/Wo
        are given, making the torch path more valuable here, not less).

    Returns
    -------
    Pw, Qw : ndarray (n, n)
        The frequency-weighted Gramians, restricted to the plant's own
        states -- feed directly into _balance_from_gramians().
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Cout = np.asarray(Cout, dtype=float)
    n = A.shape[0]

    if Wi is not None:
        Ai, Bi, Ci, Di = (np.asarray(x, dtype=float) for x in Wi)
        ni = Ai.shape[0]
        A_aug_c = np.block([[A, B @ Ci], [np.zeros((ni, n)), Ai]])
        B_aug_c = np.vstack([B @ Di, Bi])
        P_aug = controllability_gramian(A_aug_c, B_aug_c, backend=backend, device=device)
        Pw = P_aug[:n, :n]
    else:
        Pw = controllability_gramian(A, B, backend=backend, device=device)

    if Wo is not None:
        Ao, Bo, Co, Do = (np.asarray(x, dtype=float) for x in Wo)
        no = Ao.shape[0]
        A_aug_o = np.block([[A, np.zeros((n, no))], [Bo @ Cout, Ao]])
        Cout_aug_o = np.hstack([Do @ Cout, Co])
        Q_aug = observability_gramian(A_aug_o, Cout_aug_o, backend=backend, device=device)
        Qw = Q_aug[:n, :n]
    else:
        Qw = observability_gramian(A, Cout, backend=backend, device=device)

    return Pw, Qw


def frequency_weighted_hankel_singular_values(A, B, Cout, Wi=None, Wo=None,
                                               return_transform=False,
                                               backend="numpy", device="cpu"):
    """The Enns frequency-weighted analogue of hankel_singular_values()
    -- same _balance_from_gramians() step, fed frequency_weighted_
    gramians()'s Pw/Qw instead of the plain Gramians. With Wi=Wo=None
    this reproduces hankel_singular_values() exactly (frequency_
    weighted_gramians() returns the SAME plain Gramians in that case) --
    see test_frequency_weighted_bt.py's direct regression check of
    this. backend/device: Wave 9 addendum item 138, same convention as
    hankel_singular_values()."""
    if backend not in ("numpy", "torch"):
        raise ValueError(
            f"frequency_weighted_hankel_singular_values: unknown "
            f"backend={backend!r} -- expected 'numpy' (default) or 'torch'.")
    Pw, Qw = frequency_weighted_gramians(A, B, Cout, Wi=Wi, Wo=Wo, backend=backend, device=device)
    sigma, T, Tinv = _balance_from_gramians(Pw, Qw, backend=backend, device=device)
    if not return_transform:
        return sigma
    return sigma, T, Tinv


class BalancedTruncationROM:
    """A reduced-order model built by balancing then truncating a
    stable first-order state-space realization. Most callers should use
    from_MCK() rather than this constructor.

    Attributes
    ----------
    ss : state_space.StateSpaceSystem
        The full-order form="A" realization this was built from.
    hsv : ndarray
        ALL of the full-order system's Hankel singular values
        (descending) -- kept in full (not just the retained r) so
        h_infinity_error_bound() can sum the DISCARDED ones.
    r : int
        The retained reduced order.
    A_r, B_r, Cout_r : ndarray
        The reduced (balanced-then-truncated) system.
    """

    def __init__(self, ss, hsv, T, Tinv, r):
        self.ss = ss
        self.hsv = hsv
        self.r = r
        T_r = T[:, :r]
        Tinv_r = Tinv[:r, :]
        self.A_r = Tinv_r @ ss.A @ T_r
        self.B_r = Tinv_r @ ss.B
        self.Cout_r = ss.Cout @ T_r

    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, r=10, backend="numpy", device="cpu"):
        """Build a BalancedTruncationROM directly from second-order
        mass/stiffness (and, optionally, damping) matrices plus an
        input map B and output map Cout -- mirrors
        krylov.KrylovROM.from_MCK / frequency.FrequencyROM.from_MCK's
        convention.

        Parameters
        ----------
        M, K : ndarray (n_dof, n_dof)
        B : ndarray (n_dof, n_in) or (n_dof,)
        Cout : ndarray (n_out, n_dof) or (n_dof,)
        C : ndarray (n_dof, n_dof), optional
            Damping. Genuinely REQUIRED in practice, not just optional
            bookkeeping -- an undamped model's poles sit on the
            imaginary axis, which is not Hurwitz-stable, and the
            Lyapunov equations this method solves assume Hurwitz
            stability. Pass at least light Rayleigh damping.
        r : int
            Retained reduced STATE-SPACE order (like krylov.KrylovROM,
            this is DOUBLE the "number of second-order modes" a
            mode-displacement basis of comparable size would use --
            see state_space.py's module docstring).
        backend, device
            Wave 9 addendum item 138: backend="numpy" (default,
            unchanged) / backend="torch" (+ device="cpu"/"cuda"),
            passed straight through to hankel_singular_values() -- see
            that function's own docstring and torch_linalg.py's module
            docstring for the method and its scope.
        """
        ss = to_state_space(M, K, C=C, B=B, Cout=Cout, form="A")
        hsv, T, Tinv = hankel_singular_values(ss.A, ss.B, ss.Cout, return_transform=True,
                                               backend=backend, device=device)
        return cls(ss, hsv, T, Tinv, r)

    # -----------------------------------------------------------------
    def transfer_function(self, s):
        """H_r(s) = Cout_r (s*I - A_r)^-1 B_r. Returns (n_out, n_in)."""
        M = s * np.eye(self.r) - self.A_r
        X = np.linalg.solve(M, self.B_r)
        return self.Cout_r @ X

    def frequency_response(self, omega_array):
        """H_r(i*omega) swept over omega_array -- same convention and
        return-shape rule as krylov.KrylovROM.frequency_response()
        (squeezed to a 1-D complex array for a SISO port), directly
        comparable to it and to a full-order fea_engine
        solve_harmonic() trace at the same port."""
        omega_array = np.atleast_1d(np.asarray(omega_array, dtype=float))
        n_out, n_in = self.Cout_r.shape[0], self.B_r.shape[1]
        H = np.empty((len(omega_array), n_out, n_in), dtype=complex)
        for i, omega in enumerate(omega_array):
            H[i] = self.transfer_function(1j * omega)
        if n_out == 1 and n_in == 1:
            return H[:, 0, 0]
        return H

    def h_infinity_error_bound(self):
        """2 * sum(discarded Hankel singular values) -- the classical
        a priori H-infinity error bound (eq. 58): ||H(s) - H_r(s)||_inf
        <= this value, for EVERY frequency at once, known BEFORE ever
        evaluating a single frequency response. This is what makes
        balanced truncation's accuracy claim fundamentally different
        from krylov.KrylovROM's (which has no such a priori,
        frequency-independent bound at all -- moment matching's
        accuracy can only be assessed locally, a posteriori)."""
        return float(2.0 * np.sum(self.hsv[self.r:]))

    def is_stable(self):
        """True iff every pole of the reduced model has negative real
        part. For balanced truncation this is a THEOREM (Pernebo &
        Silverman 1982) given a stable full-order model, not an
        empirical property -- see module docstring. Kept as an explicit,
        checkable regression guard on that theorem, matching
        krylov.KrylovROM.is_stable()'s interface even though the two
        methods' actual guarantees are very different."""
        poles = np.linalg.eigvals(self.A_r)
        return bool(np.all(poles.real < 0))


class SingularPerturbationROM:
    """Singular perturbation approximation (SPA; Liu & Anderson 1989,
    "Singular perturbation approximation of balanced systems" -- also
    called "residualization", e.g. Antoulas, *Approximation of
    Large-Scale Dynamical Systems*, Ch. 9) -- docs/classical_mor_
    roadmap.md Section 8 has the full derivation this docstring
    summarizes.

    Ordinary BalancedTruncationROM DISCARDS the truncated ("fast")
    balanced states `x2` outright (assumes `x2 = 0`), which is exact at
    `s -> infinity` but generally WRONG at `s = 0` (DC/steady-state) --
    the opposite of what a structural analyst often cares most about
    (static or near-static loads). SPA instead assumes `x2` reaches
    QUASI-STEADY-STATE instantly (`x2' = 0`, not `x2 = 0`), solves the
    resulting algebraic constraint for `x2` in terms of the kept state
    `x1` and the input `u`, and substitutes it back in. In balanced
    coordinates, partitioned into kept (1, size r) and discarded
    (2, size n-r) blocks with `D = 0` (this package's structural
    systems never have a direct input-to-output feedthrough term -- see
    state_space.py):

        A = [[A11, A12], [A21, A22]],  B = [[B1], [B2]],  Cout = [C1, C2]
        0 = x2' = A21 x1 + A22 x2 + B2 u   =>   x2 = -A22^-1 (A21 x1 + B2 u)

    (valid: A22 is a genuine THEOREM-guaranteed-stable, hence
    invertible, block of a balanced realization of a stable system --
    Liu & Anderson 1989), substituting into the x1' and y equations
    gives the Schur-complement reduced system

        A_r = A11 - A12 @ A22^-1 @ A21
        B_r = B1  - A12 @ A22^-1 @ B2
        C_r = C1  - C2  @ A22^-1 @ A21
        D_r =  0  - C2  @ A22^-1 @ B2      (generally NONZERO)

    This is the exact state-space analogue of what mode_correction.py
    already does for a plain modal basis: mode_acceleration_
    correction() recovers the static contribution a truncated modal
    basis throws away via one extra FULL-order static solve; SPA
    recovers the analogous static (DC) contribution of the truncated
    Hankel-singular-value states via one small (n-r, n-r) linear solve
    instead -- the same idea, one level up in this package's own
    "systems and control" family.

    Two classical guarantees carried over from ordinary BT (Liu &
    Anderson 1989 prove both; test_balanced_truncation.py checks both
    directly rather than assuming them): SPA is ALSO always stable
    given a stable full-order model (same theorem BalancedTruncationROM.
    is_stable() relies on), and shares the SAME a priori H-infinity
    error bound, `2 * sum(discarded Hankel singular values)`, as
    ordinary BT -- SPA's value is fixing WHERE (s=0 vs. s=infinity) the
    reduced model is exact, not tightening that bound.

    Deliberately a separate class in this SAME module (not a new
    file): tightly coupled to, and reuses without recomputation, the
    SAME hankel_singular_values(..., return_transform=True) balancing
    transform BalancedTruncationROM already computes -- same "systems
    and control" family, genuinely different algorithm, mirroring
    BalancedTruncationROM's method names exactly for direct side-by-
    side comparison (the same design choice krylov.KrylovROM/
    BalancedTruncationROM and scm.SingularValueLowerBound/
    scm_lp.LPSingularValueLowerBound already made).

    Attributes
    ----------
    ss, hsv, r : same meaning as BalancedTruncationROM's.
    A_r, B_r, Cout_r, D_r : ndarray
        The reduced (balanced-then-residualized) system. D_r is
        generally nonzero, unlike BalancedTruncationROM's (always 0,
        since this package's full-order D is always 0).
    """

    def __init__(self, ss, hsv, T, Tinv, r):
        self.ss = ss
        self.hsv = hsv
        self.r = r

        T1, T2 = T[:, :r], T[:, r:]
        Tinv1, Tinv2 = Tinv[:r, :], Tinv[r:, :]

        A11 = Tinv1 @ ss.A @ T1
        A12 = Tinv1 @ ss.A @ T2
        A21 = Tinv2 @ ss.A @ T1
        A22 = Tinv2 @ ss.A @ T2
        B1 = Tinv1 @ ss.B
        B2 = Tinv2 @ ss.B
        C1 = ss.Cout @ T1
        C2 = ss.Cout @ T2

        # A22 is a theorem-guaranteed-stable (hence invertible) block
        # of a balanced realization of a stable system (Liu & Anderson
        # 1989) -- solve rather than form an explicit inverse, same
        # numerical-hygiene convention this package uses elsewhere.
        A22_inv_A21 = np.linalg.solve(A22, A21)
        A22_inv_B2 = np.linalg.solve(A22, B2)

        self.A_r = A11 - A12 @ A22_inv_A21
        self.B_r = B1 - A12 @ A22_inv_B2
        self.Cout_r = C1 - C2 @ A22_inv_A21
        self.D_r = -(C2 @ A22_inv_B2)

    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, r=10, backend="numpy", device="cpu"):
        """Build a SingularPerturbationROM directly from second-order
        mass/stiffness (and, optionally, damping) matrices plus an
        input map B and output map Cout -- identical signature to
        BalancedTruncationROM.from_MCK (including backend=/device=,
        Wave 9 addendum item 138), for direct side-by-side use."""
        ss = to_state_space(M, K, C=C, B=B, Cout=Cout, form="A")
        hsv, T, Tinv = hankel_singular_values(ss.A, ss.B, ss.Cout, return_transform=True,
                                               backend=backend, device=device)
        return cls(ss, hsv, T, Tinv, r)

    # -----------------------------------------------------------------
    def transfer_function(self, s):
        """H_r(s) = Cout_r (s*I - A_r)^-1 B_r + D_r. Returns (n_out,
        n_in). Note the "+ D_r" term, absent from BalancedTruncationROM.
        transfer_function() (whose D_r is always exactly 0) -- this is
        precisely the term that makes H_r(0) match the full-order
        system's DC gain exactly (see module/class docstring and
        test_balanced_truncation.py)."""
        M = s * np.eye(self.r) - self.A_r
        X = np.linalg.solve(M, self.B_r)
        return self.Cout_r @ X + self.D_r

    def frequency_response(self, omega_array):
        """H_r(i*omega) swept over omega_array -- same convention and
        return-shape rule as BalancedTruncationROM.frequency_response()/
        krylov.KrylovROM.frequency_response()."""
        omega_array = np.atleast_1d(np.asarray(omega_array, dtype=float))
        n_out, n_in = self.Cout_r.shape[0], self.B_r.shape[1]
        H = np.empty((len(omega_array), n_out, n_in), dtype=complex)
        for i, omega in enumerate(omega_array):
            H[i] = self.transfer_function(1j * omega)
        if n_out == 1 and n_in == 1:
            return H[:, 0, 0]
        return H

    def h_infinity_error_bound(self):
        """2 * sum(discarded Hankel singular values) -- the SAME
        formula, and the SAME theorem (Liu & Anderson 1989 prove SPA
        shares BT's a priori H-infinity error bound), as
        BalancedTruncationROM.h_infinity_error_bound()."""
        return float(2.0 * np.sum(self.hsv[self.r:]))

    def is_stable(self):
        """True iff every pole of the reduced model has negative real
        part. Also a THEOREM for SPA given a stable full-order model
        (Liu & Anderson 1989) -- kept as an explicit regression guard,
        matching BalancedTruncationROM.is_stable()'s interface and
        reasoning exactly."""
        poles = np.linalg.eigvals(self.A_r)
        return bool(np.all(poles.real < 0))


class FrequencyWeightedBalancedTruncationROM:
    """Frequency-weighted balanced truncation (Enns 1984, "Model
    reduction with balanced realizations: An error bound and a
    frequency weighted generalization") -- docs/classical_mor_
    roadmap.md Section 10 has the full derivation this docstring
    summarizes.

    Ordinary BalancedTruncationROM's Hankel singular values rank each
    balanced state by how much it matters for the input-output
    behavior UNIFORMLY across all frequencies. Weighting the
    controllability/observability Gramians by input/output filters
    `Wi(s)`/`Wo(s)` (via frequency_weighted_gramians()) BEFORE
    balancing re-ranks the states by how much they matter for a
    SPECIFIC band instead -- a direct trade (more accurate in that
    band, potentially less accurate elsewhere), not a free
    improvement, and this class's own validation
    (tests/test_frequency_weighted_bt.py) measures both sides of that
    trade rather than only the favorable one.

    With `Wi=Wo=None` (the default), this reduces EXACTLY to ordinary
    BalancedTruncationROM (frequency_weighted_gramians() returns the
    plain, unweighted Gramians in that case) -- checked directly as a
    regression test, not just claimed.

    An HONEST LIMITATION inherited from the Enns construction itself,
    not an implementation gap: unlike BalancedTruncationROM/
    SingularPerturbationROM, TWO-SIDED weighting (both Wi and Wo given)
    has NO unconditional stability-preservation theorem -- the
    weighted Gramians restricted to the plant's sub-block are not
    guaranteed positive semi-definite in general, so the reduced
    model's stability is correspondingly not guaranteed either (well
    documented in the literature since Enns' own paper). ONE-SIDED
    weighting (only Wi or only Wo) DOES preserve stability in general.
    This means is_stable() below must be CHECKED per call, not
    assumed -- matching krylov.KrylovROM.is_stable()'s honesty
    convention, NOT BalancedTruncationROM/SingularPerturbationROM's
    theorem-backed regression-guard convention. For the same reason,
    this class deliberately does NOT expose h_infinity_error_bound():
    the classical `2 * sum(discarded Hankel singular values)` a priori
    bound is a property of the UNWEIGHTED Hankel singular values
    specifically; a frequency-weighted a priori bound exists in the
    literature but is more involved and was not independently verified
    here, so rather than claim an unverified bound, this class omits
    the method entirely -- the same honest-omission choice KrylovROM
    already makes for the same reason.

    Attributes
    ----------
    ss, hsv, r : same meaning as BalancedTruncationROM's (hsv here are
        the FREQUENCY-WEIGHTED Hankel singular values, not comparable
        in magnitude to BalancedTruncationROM's own -- both are
        "sigma_i" but of different Gramians).
    Wi, Wo : the weight realizations this instance was built with (or
        None), kept for inspection/reporting.
    A_r, B_r, Cout_r : ndarray
        The reduced (weighted-balance-then-truncated) system. D_r is
        NOT modeled here (this class truncates, like
        BalancedTruncationROM, rather than residualizing like
        SingularPerturbationROM -- the two ideas are independent and
        could in principle be combined, not attempted here).
    """

    def __init__(self, ss, hsv, T, Tinv, r, Wi=None, Wo=None):
        self.ss = ss
        self.hsv = hsv
        self.r = r
        self.Wi = Wi
        self.Wo = Wo
        T_r = T[:, :r]
        Tinv_r = Tinv[:r, :]
        self.A_r = Tinv_r @ ss.A @ T_r
        self.B_r = Tinv_r @ ss.B
        self.Cout_r = ss.Cout @ T_r

    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, r=10, Wi=None, Wo=None, backend="numpy", device="cpu"):
        """Build a FrequencyWeightedBalancedTruncationROM directly from
        second-order mass/stiffness (and, optionally, damping) matrices
        plus an input map B and output map Cout -- same signature as
        BalancedTruncationROM.from_MCK, plus the Wi/Wo weight
        arguments (see lowpass_weight()/bandpass_weight() for
        convenience constructors, or module docstring for the raw
        (Aw, Bw, Cw, Dw) tuple convention). backend/device: Wave 9
        addendum item 138, same convention as BalancedTruncationROM.
        from_MCK -- the augmented (plant + weight) systems this class's
        own Gramians are built from are LARGER than the plant alone,
        making the torch path more valuable here, not less."""
        ss = to_state_space(M, K, C=C, B=B, Cout=Cout, form="A")
        hsv, T, Tinv = frequency_weighted_hankel_singular_values(
            ss.A, ss.B, ss.Cout, Wi=Wi, Wo=Wo, return_transform=True,
            backend=backend, device=device)
        return cls(ss, hsv, T, Tinv, r, Wi=Wi, Wo=Wo)

    # -----------------------------------------------------------------
    def transfer_function(self, s):
        """H_r(s) = Cout_r (s*I - A_r)^-1 B_r. Returns (n_out, n_in) --
        same convention as BalancedTruncationROM.transfer_function()
        (no D_r term, unlike SingularPerturbationROM's)."""
        M = s * np.eye(self.r) - self.A_r
        X = np.linalg.solve(M, self.B_r)
        return self.Cout_r @ X

    def frequency_response(self, omega_array):
        """H_r(i*omega) swept over omega_array -- same convention and
        return-shape rule as BalancedTruncationROM.frequency_response()."""
        omega_array = np.atleast_1d(np.asarray(omega_array, dtype=float))
        n_out, n_in = self.Cout_r.shape[0], self.B_r.shape[1]
        H = np.empty((len(omega_array), n_out, n_in), dtype=complex)
        for i, omega in enumerate(omega_array):
            H[i] = self.transfer_function(1j * omega)
        if n_out == 1 and n_in == 1:
            return H[:, 0, 0]
        return H

    def is_stable(self):
        """True iff every pole of the reduced model has negative real
        part. UNLIKE BalancedTruncationROM/SingularPerturbationROM,
        this is NOT a theorem in general for two-sided (both Wi and Wo
        given) weighting -- see class docstring. Genuinely checked
        here, matching krylov.KrylovROM.is_stable()'s honesty
        convention, not asserted or assumed."""
        poles = np.linalg.eigvals(self.A_r)
        return bool(np.all(poles.real < 0))
