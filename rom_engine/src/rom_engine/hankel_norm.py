"""
hankel_norm.py -- optimal Hankel norm approximation (Adamjan-Arov-Krein
1971 theory; explicit state-space realization due to Glover, "All
optimal Hankel-norm approximations of linear multivariable systems and
their L-infinity error bounds", 1984) -- the last of the classical MOR
roadmap's explicitly-named Phase 5 balanced-truncation-family
extensions. docs/classical_mor_roadmap.md Section 14 has the full
derivation and the numerical-risk investigation this docstring
summarizes.

THE IDEA: balanced_truncation.BalancedTruncationROM answers "which
order-r model minimizes some norm of the approximation error" only
implicitly, via its own a priori H-infinity bound (2 * sum of discarded
Hankel singular values) -- a bound the theory does not claim is tight.
AAK theory answers a SHARPER, EXACT question directly: of every
possible order-r system (stable OR unstable), the one this module
builds achieves

    ||G - G_r||_Hankel = sigma_(r+1)

EXACTLY -- not a bound, the exact minimum, where sigma_(r+1) is the
(r+1)-th Hankel singular value of the FULL-order system G. This is a
genuinely different (and stronger, in the Hankel-norm sense) optimality
property than anything BalancedTruncationROM/SingularPerturbationROM/
FrequencyWeightedBalancedTruncationROM claim.

THE CONSTRUCTION (SISO only -- see below for why): balance the
full-order system exactly like balanced_truncation.py does
(hankel_singular_values(..., return_transform=True) is reused
unchanged, not reimplemented), then, in balanced coordinates, isolate
the SINGLE state associated with sigma_(r+1) = hsv[r] (already sorted
descending, so this is just index r -- no extra permutation needed) and
combine ALL the other n-1 states -- both the ones with LARGER Hankel
singular values (indices < r, which will end up kept) and the ones with
SMALLER ones (indices > r, which will end up discarded) -- into one
intermediate order-(n-1) system via the AAK formulas (Glover 1984 eqs.
3.5-3.8, the "Sigma_check"/"Gamma" notation below matches Benner &
Werner's arXiv:1612.06205 restatement, cross-checked against Sandberg's
KTH lecture notes -- both independently notated sources agree exactly):

    U = B2 / C2                                    (scalar, SISO)
    Gamma = Sigma_check^2 - sigma^2 * I
    Atil = Gamma^-1 (sigma^2 A11^T + Sigma_check A11 Sigma_check
                      + sigma C1^T U B1^T)
    Btil = Gamma^-1 (Sigma_check B1 - sigma C1^T U)
    Ctil = C1 Sigma_check - sigma U B1^T
    Dtil = sigma * U

where (A11, B1, C1) is the balanced realization restricted to the n-1
KEPT states, and (B2, C2) are the removed state's own row/column. A
deep theorem (not verified from scratch here, taken from Glover 1984 and
cross-checked via the secondary sources above) guarantees Atil's n-1
eigenvalues split EXACTLY into r stable and (n-1-r) antistable ones
whenever sigma_(r+1) is a SIMPLE Hankel singular value of the full
system (simple across the WHOLE spectrum, not just distinct from its
immediate neighbors) -- so ONE application of these formulas reaches
the target order r directly, no iteration needed. The order-r STABLE
part is then extracted by a real Schur decomposition of Atil sorted
stable-first (scipy.linalg.schur(..., sort=...)), which by itself would
only be an APPROXIMATE truncation (it drops the off-diagonal coupling
block between the stable and antistable Schur blocks); the exact
decoupling this module actually needs solves the Sylvester equation
`Ah @ X - X @ Af = -A12` for that coupling (scipy.linalg.
solve_sylvester) and folds X into B_r -- this Schur-plus-Sylvester step
is this module's own synthesis (standard numerical linear algebra for
extracting an invariant subspace's reduced dynamics, not something
independently found written out in the three triangulated sources in
exactly this form), which is exactly why the self-check below exists
rather than trusting the algebra alone.

WHY SISO ONLY: the U = B2 / C2 step above is a genuine SCALAR division
in the derivation this module implements. A MIMO generalization exists
in the literature (replacing the scalar U with a matrix built from an
SVD/pseudo-inverse of the removed states' input/output blocks) but was
not independently re-derived or validated here -- rather than guess at
that generalization the way scm_lp.py and passivity.py both explicitly
declined to guess at their own harder derivations, this class raises a
clear ValueError for any non-SISO (M, B, Cout) combination instead of
silently doing something unverified.

WHY A SEPARATE MODULE, not a sibling class alongside
BalancedTruncationROM/SingularPerturbationROM/
FrequencyWeightedBalancedTruncationROM in balanced_truncation.py: this
construction reuses hankel_singular_values() for its balancing step
(the one piece it genuinely shares) but everything after that --
isolating a single state, the Gamma-inversion formulas, and especially
the Schur-plus-Sylvester decoupling -- is algorithmically unlike
anything else in that file (no other class needs a Schur decomposition
or a Sylvester solve at all). This mirrors why krylov.py is kept
separate from balanced_truncation.py despite both being "systems and
control" ROMs: shared FAMILY, genuinely different algorithm.

A GENUINE, MEASURED NUMERICAL LIMITATION (found via a deliberate
stress-testing sweep across many random synthetic SISO systems before
this module was written -- see docs/classical_mor_roadmap.md Section 14
for the full sweep): this construction can become numerically
unreliable -- the stable/antistable eigenvalue split silently returning
the wrong COUNT, or (more insidiously) returning the RIGHT count with a
WRONG reduced model -- when sigma_(r+1) is very small relative to the
largest Hankel singular value hsv[0] (roughly, within about 6-8 orders
of magnitude of hsv[0] times machine epsilon in the swept examples,
though the failure was NOT cleanly predicted by any single simple a
priori quantity tried, including cond(Gamma) -- two cases with
identical cond(Gamma) had wildly different actual error, ruling that
out as a reliable a priori guard on its own). Rather than ship an
unreliable a priori conditioning threshold, THIS MODULE INSTEAD RUNS AN
EXACT EMPIRICAL SELF-CHECK ON EVERY CONSTRUCTION: the AAK theorem
guarantees the Hankel norm of the ERROR system G - G_r equals
sigma_(r+1) exactly, and since that error system's Hankel singular
values can be computed by the SAME already-validated
hankel_singular_values() function used for balancing, every
OptimalHankelNormROM checks this agreement itself (see
measured_hankel_norm_error / numerically_reliable below) and warns
loudly, rather than silently, when it does not hold -- turning an
open-ended numerical risk into a concrete, always-on, self-reported
diagnostic. In practice, for the moderately-sized, modally-pre-truncated
structural models this package's own workflow already recommends (see
balanced_truncation.py's "PRACTICAL SCALE NOTE"), the Hankel singular
value spread stays far from this danger zone -- see
tests/test_hankel_norm.py for the measurement on this package's own
real fea_engine fixture.

GPU/torch side-by-side path (Wave 9 addendum item 138, docs/
consolidated_future_roadmap.md): from_MCK()/__init__() gained a
backend="numpy" (default) / backend="torch" + device= parameter,
threaded into every hankel_singular_values() call this class makes --
both the main balancing step AND the constructor's own empirical
self-check (whose augmented Ae system is LARGER than the full-order
model alone, making it the single most expensive call here). The
Schur-plus-Sylvester decoupling construction itself is NOT ported --
it operates on the small, r-sized reduced system regardless of the
full-order model's size, so there is no real GPU win to chase there;
see torch_linalg.py's own module docstring for the full reasoning.

THE L-INFINITY BOUND -- DELIBERATELY NOT EXPOSED: Glover's own paper
also gives a TIGHT (non-doubled) L-infinity bound, sigma_(r+1) <=
||G - G_r||_inf <= sum(sigma_(r+1) .. sigma_n), but achieving the tight
lower half of that in general requires an additional recursive
constant-correction step this module does not implement (flagged as
the single least-verified point in the research behind this module).
Rather than expose an h_infinity_error_bound() method whose claimed
bound was not independently re-derived here, this class follows
KrylovROM's and FrequencyWeightedBalancedTruncationROM's own precedent
of honestly omitting a method it cannot back with a verified guarantee
-- callers wanting an L-infinity estimate should measure
frequency_response() directly against the full-order model, exactly
like those two classes' own tests already do.
"""
import warnings

import numpy as np
from scipy.linalg import schur, solve_sylvester

from .state_space import to_state_space
from .balanced_truncation import hankel_singular_values


class OptimalHankelNormROM:
    """An order-r reduced model achieving ||G - G_r||_Hankel =
    sigma_(r+1) EXACTLY (the Adamjan-Arov-Krein/Glover optimum) -- see
    module docstring for the full construction, the SISO restriction,
    and the numerical self-check this class always runs. Most callers
    should use from_MCK() rather than this constructor.

    Attributes
    ----------
    ss : state_space.StateSpaceSystem
        The full-order form="A" realization this was built from.
    hsv : ndarray
        ALL of the full-order system's Hankel singular values
        (descending).
    r : int
        The retained reduced order.
    A_r, B_r, Cout_r, D_r : ndarray
        The reduced (AAK-optimal) system. D_r is generally nonzero
        (like SingularPerturbationROM's, unlike BalancedTruncationROM's).
    sigma_r1 : float
        sigma_(r+1) -- the AAK-theoretical, EXACT Hankel norm of
        G - G_r. See hankel_norm_error_bound().
    measured_hankel_norm_error : float
        The SAME quantity, computed empirically and independently by
        building the error system G - G_r and taking its largest
        Hankel singular value (via the already-validated
        hankel_singular_values()) -- should equal sigma_r1 to close to
        machine precision for a numerically reliable construction; see
        numerically_reliable.
    numerically_reliable : bool
        False if measured_hankel_norm_error disagrees with sigma_r1 by
        more than reliability_tol (relative) -- see module docstring's
        numerical-limitation discussion. A warnings.warn() is also
        raised in that case.
    """

    def __init__(self, ss, hsv, T, Tinv, r, gap_tol=1e-8, reliability_tol=1e-4,
                 backend="numpy", device="cpu"):
        self.ss = ss
        self.hsv = np.asarray(hsv, dtype=float)
        self.r = r
        self._backend = backend
        self._device = device

        n_out, n_in = ss.Cout.shape[0], ss.B.shape[1]
        if n_out != 1 or n_in != 1:
            raise ValueError(
                f"OptimalHankelNormROM is SISO-only (the AAK/Glover "
                f"construction implemented here needs a scalar "
                f"U = B2/C2 division at the removed balanced state -- "
                f"see module docstring) -- got n_out={n_out}, "
                f"n_in={n_in}. Use BalancedTruncationROM or KrylovROM "
                f"for MIMO systems."
            )

        n_tot = ss.A.shape[0]
        if not (1 <= r <= n_tot - 1):
            raise ValueError(
                f"r={r} out of range for OptimalHankelNormROM -- need "
                f"1 <= r <= n_total-1 (n_total={n_tot}): this "
                f"construction removes exactly one balanced state "
                f"(index r) and then splits the remaining n_total-1 "
                f"states into r stable + (n_total-1-r) antistable ones."
            )

        sigma = float(self.hsv[r])
        other = np.delete(self.hsv, r)
        rel_gaps = np.abs(other - sigma) / sigma
        min_gap = float(np.min(rel_gaps))
        if min_gap < gap_tol:
            raise ValueError(
                f"hsv[{r}]={sigma:.6e} is not a SIMPLE Hankel singular "
                f"value of this system (the closest OTHER Hankel "
                f"singular value -- not necessarily an immediate "
                f"neighbor in the sorted list -- is within relative "
                f"{min_gap:.3e} of it, tol={gap_tol:.0e}) -- the "
                f"one-shot AAK construction requires sigma_(r+1) to be "
                f"simple across the WHOLE spectrum. Pick a different r."
            )

        Ab = Tinv @ ss.A @ T
        Bb = Tinv @ ss.B
        Cb = ss.Cout @ T
        keep = [i for i in range(n_tot) if i != r]
        A11 = Ab[np.ix_(keep, keep)]
        B1 = Bb[keep, :]
        C1 = Cb[:, keep]
        B2 = Bb[[r], :]
        C2 = Cb[:, [r]]
        Sigma_check = np.diag(self.hsv[keep])

        # SISO sanity check the theory guarantees: in a genuinely
        # balanced realization, |B2| == |C2| exactly (both come from
        # the same singular vector of the same Gramian pair) -- a
        # mismatch here would mean T/Tinv is not actually a balancing
        # transform, not just numerical noise.
        b2_mag, c2_mag = abs(B2[0, 0]), abs(C2[0, 0])
        if abs(b2_mag - c2_mag) > 1e-6 * max(b2_mag, c2_mag, 1e-300):
            warnings.warn(
                f"OptimalHankelNormROM: balanced-realization sanity "
                f"check failed at the removed state -- |B2|={b2_mag:.6e} "
                f"!= |C2|={c2_mag:.6e} (should be exactly equal). "
                f"Results may not be reliable.",
                stacklevel=2,
            )

        U = B2 / C2
        Gamma = Sigma_check ** 2 - sigma ** 2 * np.eye(n_tot - 1)
        Atil = np.linalg.solve(
            Gamma,
            sigma ** 2 * A11.T + Sigma_check @ A11 @ Sigma_check
            + sigma * C1.T @ U @ B1.T,
        )
        Btil = np.linalg.solve(Gamma, Sigma_check @ B1 - sigma * C1.T @ U)
        Ctil = C1 @ Sigma_check - sigma * U @ B1.T
        Dtil = sigma * U

        Ts, Z, sdim = schur(Atil, output="real", sort=lambda x: x.real < 0)
        if sdim != r:
            raise ValueError(
                f"AAK stable/antistable split gave {sdim} stable "
                f"eigenvalues out of {n_tot - 1}, expected exactly "
                f"r={r} -- Glover's theorem guarantees this split when "
                f"sigma_(r+1) is simple, so a mismatch here signals a "
                f"numerical failure. Very likely cause: sigma_(r+1)="
                f"{sigma:.3e} is too small relative to the largest "
                f"Hankel singular value hsv[0]={self.hsv[0]:.3e} "
                f"(ratio {sigma / self.hsv[0]:.3e}) -- see module "
                f"docstring's numerical-limitation discussion. Try a "
                f"different r, or modally pre-truncate the model first "
                f"(same fix balanced_truncation.py's own module "
                f"docstring recommends for raw, finely-meshed FE "
                f"models)."
            )

        Ah = Ts[:r, :r]
        Af = Ts[r:, r:]
        A12 = Ts[:r, r:]
        Bt = Z.T @ Btil
        Ct = Ctil @ Z
        Bh = Bt[:r, :]
        Bf = Bt[r:, :]
        Ch = Ct[:, :r]

        if Af.shape[0] > 0:
            # exact decoupling of the stable/antistable coupling block
            # (a plain reordered-Schur truncation alone would only be
            # approximate) -- see module docstring.
            X = solve_sylvester(Ah, -Af, -A12)
            Bh_new = Bh - X @ Bf
        else:
            Bh_new = Bh   # r == n_tot - 1: nothing antistable to decouple

        self.A_r = Ah
        self.B_r = Bh_new
        self.Cout_r = Ch
        self.D_r = Dtil
        self.sigma_r1 = sigma

        # --- independent empirical self-check (always run -- see
        # module docstring for why an a priori conditioning threshold
        # alone was not trustworthy enough during development).
        n_full = ss.A.shape[0]
        n_red = self.A_r.shape[0]
        Ae = np.block([[ss.A, np.zeros((n_full, n_red))],
                       [np.zeros((n_red, n_full)), self.A_r]])
        Be = np.vstack([ss.B, self.B_r])
        Ce = np.hstack([ss.Cout, -self.Cout_r])
        # This self-check's own Ae is LARGER than the full-order system
        # alone (n_full + n_red states) -- the single most expensive
        # hankel_singular_values() call this class makes, so it gets
        # the same backend=/device= (Wave 9 addendum item 138) this
        # instance was built with, not left hardcoded to NumPy.
        hsv_err = hankel_singular_values(Ae, Be, Ce, backend=self._backend, device=self._device)
        measured = float(hsv_err[0])
        self.measured_hankel_norm_error = measured
        rel_discrepancy = abs(measured - sigma) / sigma
        self.numerically_reliable = rel_discrepancy < reliability_tol
        if not self.numerically_reliable:
            warnings.warn(
                f"OptimalHankelNormROM: the EMPIRICALLY measured Hankel "
                f"norm of the error system ({measured:.6e}) disagrees "
                f"with the AAK-theoretical value sigma_(r+1)="
                f"{sigma:.6e} by a relative {rel_discrepancy:.3e} "
                f"(tol={reliability_tol:.0e}) -- this reduced model is "
                f"likely NUMERICALLY UNRELIABLE, not just imprecise. "
                f"This was observed in development specifically when "
                f"sigma_(r+1) was very small relative to hsv[0] (a very "
                f"wide Hankel-singular-value spread) -- try a different "
                f"r, or modally pre-truncate the model first. "
                f"hankel_norm_error_bound() still returns the "
                f"theoretical value, but it should not be trusted here "
                f"-- check measured_hankel_norm_error instead.",
                stacklevel=2,
            )

    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, r=10, gap_tol=1e-8, reliability_tol=1e-4,
                 backend="numpy", device="cpu"):
        """Build an OptimalHankelNormROM directly from second-order
        mass/stiffness (and, optionally, damping) matrices plus a
        SINGLE input map B and a SINGLE output map Cout (SISO only --
        see module docstring) -- same signature convention as
        BalancedTruncationROM.from_MCK.

        Parameters
        ----------
        M, K : ndarray (n_dof, n_dof)
        B : ndarray (n_dof, 1) or (n_dof,)
        Cout : ndarray (1, n_dof) or (n_dof,)
        C : ndarray (n_dof, n_dof), optional
            Damping -- genuinely required in practice, same reasoning
            as BalancedTruncationROM.from_MCK.
        r : int
            Target reduced STATE-SPACE order (same convention as
            BalancedTruncationROM/KrylovROM -- double the "number of
            second-order modes" a mode-displacement basis would use).
        gap_tol, reliability_tol : float
            Passed through to the constructor -- see its docstring and
            the module docstring's numerical-limitation discussion.
        backend, device
            Wave 9 addendum item 138: backend="numpy" (default,
            unchanged) / backend="torch" (+ device="cpu"/"cuda").
            Accelerates BOTH the main hankel_singular_values() balancing
            call below AND the constructor's own empirical self-check
            (see __init__'s own comment) -- the Schur-plus-Sylvester
            decoupling step itself stays NumPy/SciPy-only regardless
            (see module docstring for why that is not a GPU-worthwhile
            target).
        """
        ss = to_state_space(M, K, C=C, B=B, Cout=Cout, form="A")
        hsv, T, Tinv = hankel_singular_values(ss.A, ss.B, ss.Cout, return_transform=True,
                                               backend=backend, device=device)
        return cls(ss, hsv, T, Tinv, r, gap_tol=gap_tol, reliability_tol=reliability_tol,
                   backend=backend, device=device)

    # -----------------------------------------------------------------
    def transfer_function(self, s):
        """H_r(s) = Cout_r (s*I - A_r)^-1 B_r + D_r. Returns a (1, 1)
        array (always SISO -- see module docstring)."""
        Msys = s * np.eye(self.r) - self.A_r
        X = np.linalg.solve(Msys, self.B_r)
        return self.Cout_r @ X + self.D_r

    def frequency_response(self, omega_array):
        """H_r(i*omega) swept over omega_array -- same convention as
        BalancedTruncationROM.frequency_response(), always squeezed to
        a 1-D complex array (this class is SISO-only)."""
        omega_array = np.atleast_1d(np.asarray(omega_array, dtype=float))
        H = np.empty(len(omega_array), dtype=complex)
        for i, omega in enumerate(omega_array):
            H[i] = self.transfer_function(1j * omega)[0, 0]
        return H

    def hankel_norm_error_bound(self):
        """sigma_(r+1) -- the AAK-theoretical EXACT Hankel norm of
        G - G_r (not just a bound: the exact minimum over every
        possible order-r system, per the AAK theorem). See
        numerically_reliable / measured_hankel_norm_error for this
        package's own empirical check of that claim on THIS
        construction -- if numerically_reliable is False, treat this
        value with real suspicion, not just noted skepticism."""
        return self.sigma_r1

    def is_stable(self):
        """True iff every pole of the reduced model has negative real
        part. By CONSTRUCTION here (the stable Schur block is what A_r
        is built from), this should always be True when the
        constructor did not raise -- kept as an explicit, checkable
        regression guard, matching BalancedTruncationROM.is_stable()'s
        interface and reasoning."""
        if self.A_r.size == 0:
            return True
        poles = np.linalg.eigvals(self.A_r)
        return bool(np.all(poles.real < 0))
