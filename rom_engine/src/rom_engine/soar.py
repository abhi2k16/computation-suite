"""
soar.py -- SOAR (Second-Order Arnoldi) moment-matching ROM: a
SECOND-ORDER-structure-preserving alternative to krylov.py's own
first-order Krylov moment matching, built as a Phase 5 addendum to
docs/frequency_domain_rom_roadmap.md (Section 8 there has the full
derivation and validation plan -- this docstring summarizes it).

krylov.py matches Taylor moments of H(s) = Cout (s*E-A)^-1 B, the
FIRST-ORDER linearized transfer function, at the cost of doubling the
state count (state_space.py's [q; q_dot] construction: a k-state
first-order Krylov ROM needs a 2n_dof-sized pencil to build from, and
represents roughly k/2 second-order "modes" worth of information). SOAR
(Bai & Su 2005) matches the SAME moments of the SAME port's transfer
function directly from the second-order pencil

    H(s) = Cout (s^2*M + s*C + K)^-1 B,

never forming the first-order system at all. Expanding around an
expansion point s0 (s = s0 + sigma), and defining
K0 = K + s0*C + s0^2*M (assumed invertible -- generically true for s0
not itself a system pole; s0=0 gives K0=K, matching krylov.py's own
default):

    s^2*M + s*C + K = K0 * [I - sigma*A1 - sigma^2*A2],
    A1 = -K0^-1 (C + 2*s0*M),   A2 = -K0^-1 M

so (s^2*M+s*C+K)^-1 B = K0^-1 B * sum_j sigma^j Phi_j, where Phi_j is
defined by the TWO-TERM recurrence Phi_0=I, Phi_1=A1,
Phi_j = A1*Phi_{j-1} + A2*Phi_{j-2} for j>=2 -- the second-order
generalization of the one-term recurrence (v_j = A v_{j-1}) ordinary
Arnoldi builds its basis from. Applying this to K0^-1 B directly gives
the vector recurrence soar_basis() builds an orthonormal basis of:

    r_1 = K0^-1 B,   r_0 := 0,   r_j = A1 r_{j-1} + A2 r_{j-2}  (j>=2)

An orthonormal basis V of span{r_1,...,r_k}, used exactly like every
other Galerkin projection in this package (M_r=V^T M V, C_r=V^T C V,
K_r=V^T K V, B_r=V^T B, Cout_r=Cout V), gives a REDUCED SECOND-ORDER
system of order k (not 2k) whose transfer function matches H(s)'s
first several Taylor moments at s0 -- SOAR's whole point, and the
source of its "avoids the first-order 2x blowup" advantage over
krylov.py's own arnoldi_basis().

SOAR vs. TOAR, stated honestly: Bai & Su's original SOAR construction
(built here) is literature-documented as numerically fragile,
particularly for BLOCK (multiple simultaneous input columns) starting
vectors -- exactly this package's use case (B with n_in columns). TOAR
(Lu, Su, Bai) fixes this with a "two-level" representation (every
Krylov direction stored as a shared position-space basis plus a small
per-direction coefficient pair, rather than as full-length vectors).
This module does NOT implement that two-level bookkeeping -- a
deliberate, stated scope decision (matching this project's established
precedent of honestly-scoped-out extensions: scm_lp.py's natural-norm
SCM variant, passivity.py's D!=0 positive-real ARE case, nnm.py's
subharmonic-multiplier support). soar_basis() instead mitigates SOAR's
known fragility the standard way Arnoldi-family methods generally do --
FULL reorthogonalization against every previously-kept basis vector at
every step (already how krylov.py's own arnoldi_basis() works) -- which
is expected to behave well at the modest scale this package's real
validation fixtures actually reach (two to three orders of magnitude
below the 10^5-10^7-DOF industrial scale where TOAR's extra machinery
specifically earns its keep), checked directly in test_soar.py rather
than assumed. If a real target model ever needs genuine TOAR, replacing
soar_basis()'s internals is a much narrower follow-up than this SOAR
implementation was, since SOARROM's projection/API stays the same.

A subtlety that cost a real, self-caught bug during development (worth
recording rather than silently fixing): the two-term recurrence MUST be
applied to the RAW (pre-orthogonalization) candidate vectors at each
generation, not to the orthonormalized basis vectors V accumulates.
Unlike ordinary first-order Arnoldi -- where replacing r_{j-1} with its
normalized form v_{j-1} only rescales the whole subsequent chain by a
constant (harmless, since the moment-matching theorem is a statement
about spans) -- orthogonalizing r_{j-1} against v_1,...,v_{j-2}
SUBTRACTS components, which is not a harmless rescaling once fed back
into a recurrence that mixes TWO different generations (A2's term
needs the generation-before-last, and its "new content" relative to
the current span is not preserved by using the truncated/orthogonalized
version). Concretely, feeding orthonormalized v's back into the
recurrence (the first version of this code) reproduced the exact
moments through order 1 only, REGARDLESS of how large k was made --
a telltale sign of a real bug, not a benign matching-order limitation,
since a correct method should match MORE moments as k grows, not
plateau. soar_basis() therefore tracks the raw two-term recursion on a
SEPARATE, un-orthogonalized sequence, and only orthogonalizes a COPY of
each new generation against the accumulated basis to decide what (if
anything) is genuinely new -- confirmed to restore the expected
"k basis vectors match k moments exactly" property (test_soar.py),
matching ordinary Arnoldi's own exact-through-moment-(k-1) result.

A second consequence of this fix, also worth stating plainly: because a
given input "slot" can still contribute new, previously-unseen basis
directions in a LATER generation even after one particular generation's
candidate for that slot deflates (unlike ordinary single-operator
Krylov, where once a whole generation deflates the subspace is provably
invariant and further iteration is provably pointless), soar_basis()
does NOT stop iterating a slot just because one generation added
nothing new -- it keeps advancing the raw recursion (up to a generous
cap; see max_generations) until either the target k is reached or BOTH
of the two most recently tracked raw generations are genuinely
numerically zero (the one case where the recursion truly cannot
produce anything new from then on).

A genuine numerical finding from validating this on a REAL fea_engine
beam (test_soar.py), stated honestly rather than tuned away: for a
SINGLE-vector port (one input DOF) on a structure with well-separated
eigenvalues (this package's cantilever fixture has omega2/omega1 ~ 6.3),
the raw recursion is DOMINATED by A2 = -K0^-1 M -- which is exactly the
operator inverse power iteration on the generalized eigenproblem
(K, M) uses -- so successive generations converge toward the FIRST
mode shape geometrically (ratio ~ (omega1/omega2)^2 ~ 0.025 per step
here), and the basis genuinely stops growing (new directions deflate
below tol) after just a few generations, regardless of how large k is
requested. This is NOT a bug (confirmed directly: soar_basis() still
matches every Taylor moment its actual basis size supports, to machine
precision, exactly as designed) -- it is a real, literature-consistent
property of SOAR (this is precisely why real SOAR/TOAR usage almost
always uses BLOCK (multi-column) starting vectors: a block port has
several independent directions feeding the same dominant-mode
convergence, so the basis keeps growing usefully for several more
generations before saturating). test_soar.py validates both regimes
honestly: a multi-DOF block port, where SOARROM(k) is confirmed to
noticeably OUTPERFORM KrylovROM(2k) (same total state count) at small
k, and a single-DOF port, where the basis size is confirmed to plateau
well below the requested k -- a real, reported limitation, not a
silently-avoided one.

THE ACTUAL MOTIVATION for building this (not just "because SOAR/TOAR
was on the list"): passivity.py's own closed-form theorem proves ANY
second-order Galerkin projection (M_r=Phi^T M Phi, C_r=Phi^T C Phi,
K_r=Phi^T K Phi) of an SPD-M/PSD-C/SPD-K system preserves passivity,
for ANY basis Phi -- congruence transformation preserves
positive-(semi)definiteness unconditionally. soar_basis()'s V is
exactly such a basis, so SOARROM should ALSO provably preserve
passivity -- unlike krylov.py's KrylovROM, which is a FIRST-ORDER
state-space reduction the congruence argument does not apply to, and
which test_passivity.py already found violating passivity in practice
at a real reduced order on this package's own fixture. This makes
SOARROM the first Krylov-family (moment-matching) method in this
package with a provable passivity guarantee -- checked directly in
test_soar.py, not just asserted from the theorem.
"""
__author__ = "Abhijeet"
import numpy as np
from scipy.linalg import lu_factor, lu_solve, eig


def _as_2d_input_map(x):
    """Coerce a 1-D selection vector to (n, 1); pass a genuine 2-D map
    straight through. Small, deliberately duplicated helper (not
    imported from state_space.py) -- this module has no dependency on
    state_space.py at all, since it never linearizes to first-order
    form, and keeping it that way is the point."""
    x = np.asarray(x)
    x = x if np.iscomplexobj(x) else x.astype(float, copy=False)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _as_2d_output_map(x):
    x = np.asarray(x)
    x = x if np.iscomplexobj(x) else x.astype(float, copy=False)
    return x.reshape(1, -1) if x.ndim == 1 else x


def soar_basis(M, K, B, k, C=None, s0=0.0, tol=1e-12, max_generations=None):
    """Orthonormal (n_dof, k') basis (k' <= k) of the order-k second-order
    Krylov subspace spanned by r_1=K0^-1 B, r_j = A1 r_{j-1} + A2 r_{j-2}
    -- see module docstring for the derivation of A1, A2, K0, and for
    why the raw (pre-orthogonalization) sequence, not the orthonormal
    basis itself, must drive the recurrence.

    Parameters
    ----------
    M, K : ndarray (n_dof, n_dof)
    B : ndarray (n_dof, n_in) or (n_dof,)
        ALL n_in columns seed the block recursion (block SOAR): each
        generation advances every input direction's own raw sequence at
        once; a column that stops contributing NEW basis directions can
        still resume doing so in a later generation (see module
        docstring), so slots are never permanently retired early.
    k : int
        Target basis size (total columns, not per input -- matching
        krylov.py's own arnoldi_basis() convention).
    C : ndarray (n_dof, n_dof), optional
        Damping matrix. Defaults to zero (undamped).
    s0 : float or complex
        Expansion point. s0=0 (the default) matches krylov.py's own
        default and gives K0=K.
    tol : float
        A candidate direction with norm below this (after being
        orthogonalized against everything already kept) is judged
        numerically dependent and not added to the basis -- see module
        docstring for why this does not, by itself, stop the raw
        recursion.
    max_generations : int, optional
        Hard cap on how many recurrence steps to attempt before giving
        up on reaching k (default n_dof + 10 -- generous, since the
        subspace cannot have genuinely new content beyond n_dof
        dimensions). Exists only to guarantee termination; not expected
        to bind in practice for k <= n_dof.

    Returns
    -------
    V : ndarray (n_dof, k') with k' <= k
        k' can be smaller than k if the raw recursion is confirmed
        exhausted (see module docstring) before reaching the requested
        size -- reported via V's own actual shape rather than silently
        padded, matching arnoldi_basis()'s convention.
    """
    n = M.shape[0]
    B = _as_2d_input_map(B)
    complex_dtype = (np.iscomplexobj(M) or np.iscomplexobj(K) or np.iscomplexobj(B)
                      or np.iscomplexobj(s0) or (C is not None and np.iscomplexobj(C)))
    dtype = complex if complex_dtype else float
    M = np.asarray(M, dtype=dtype)
    K = np.asarray(K, dtype=dtype)
    B = np.asarray(B, dtype=dtype)
    C = np.zeros((n, n), dtype=dtype) if C is None else np.asarray(C, dtype=dtype)
    if max_generations is None:
        max_generations = n + 10

    K0 = K + s0 * C + (s0 ** 2) * M
    lu, piv = lu_factor(K0)
    D1 = C + 2 * s0 * M   # A1 = -K0^-1 D1, applied via lu_solve below

    V_cols = []

    def _add_new_directions(raw_block):
        """Orthogonalize a COPY of raw_block's columns (MGS against
        V_cols, then against each other -- matching arnoldi_basis()'s
        own pattern) and append genuinely new directions to V_cols
        (stopping once len(V_cols)==k). Does NOT modify raw_block --
        the raw recursion continues from the UN-orthogonalized vectors
        (see module docstring). Returns the number of directions added
        (0 means this generation contributed nothing new right now --
        not necessarily a stopping condition, see caller)."""
        added = 0
        accepted = []
        for j in range(raw_block.shape[1]):
            if len(V_cols) == k:
                break
            w = raw_block[:, j].copy()
            for v in V_cols:
                w = w - np.vdot(v, w) * v
            for v in accepted:
                w = w - np.vdot(v, w) * v
            if np.linalg.norm(w) > tol:
                w = w / np.linalg.norm(w)
                accepted.append(w)
                V_cols.append(w)
                added += 1
        return added

    raw_prev = lu_solve((lu, piv), B)          # raw r_1 = K0^-1 B
    raw_prevprev = np.zeros_like(raw_prev)     # raw r_0 := 0
    _add_new_directions(raw_prev)
    if not V_cols:
        raise ValueError("SOAR subspace is empty -- B is (numerically) zero?")

    generations = 0
    while len(V_cols) < k and generations < max_generations:
        rhs = np.hstack([D1 @ raw_prev, M @ raw_prevprev])
        sol = lu_solve((lu, piv), rhs)
        n_w = raw_prev.shape[1]
        raw_new = -sol[:, :n_w] - sol[:, n_w:]
        _add_new_directions(raw_new)
        raw_prevprev = raw_prev
        raw_prev = raw_new
        generations += 1
        # Stop only once TWO CONSECUTIVE genuinely-computed generations
        # are exactly zero. generations>=2 guards against the r_0:=0
        # placeholder ever counting toward this: for an UNDAMPED system
        # at s0=0, D1=0 identically, so the very first raw_new (r_2) is
        # EXACTLY zero too -- a real, expected fact (H(s) is even in s
        # for an undamped system, so odd moments vanish identically),
        # not exhaustion; without this guard the loop would stop right
        # there and never reach the perfectly good, nonzero r_3.
        if (generations >= 2 and np.linalg.norm(raw_new) < tol
                and np.linalg.norm(raw_prevprev) < tol):
            break   # both tracked raw generations are genuinely zero --
                     # the recursion cannot produce anything new from here

    return np.column_stack(V_cols)


class SOARROM:
    """A SOAR second-order-structure-preserving Krylov moment-matching
    ROM. Unlike krylov.KrylovROM, this projects M/C/K DIRECTLY (no
    first-order linearization), giving a reduced SECOND-ORDER system of
    order k (not 2k) -- see module docstring for the derivation and the
    passivity-preservation motivation. Most callers should use
    from_MCK() rather than this constructor.

    Attributes
    ----------
    V : ndarray (n_dof, k')
        The SOAR basis.
    s0 : complex
        The expansion point V was built around.
    M_r, C_r, K_r, B_r, Cout_r : ndarray
        The reduced second-order system -- an ordinary Galerkin
        projection (Phi=V), exactly like galerkin.py/frequency.py's own
        M_r=Phi^T M Phi etc. This is what makes passivity.py's
        second-order-Galerkin passivity theorem apply here (see module
        docstring), unlike krylov.py's first-order KrylovROM.
    """

    def __init__(self, M, C, K, B, Cout, V, s0):
        self.V = V
        self.s0 = s0
        self.M_r = V.conj().T @ M @ V
        self.C_r = V.conj().T @ C @ V
        self.K_r = V.conj().T @ K @ V
        self.B_r = V.conj().T @ B
        self.Cout_r = Cout @ V

    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, s0=0.0, k=10, tol=1e-12):
        """Build a SOARROM directly from second-order mass/stiffness
        (and, optionally, damping) matrices plus an input map B and
        output map Cout -- mirrors krylov.KrylovROM.from_MCK's
        convention, EXCEPT k here means second-order (n_dof-sized)
        reduced states, not first-order (2*n_dof-sized) ones -- a fair
        "same total state count" comparison against KrylovROM is
        SOARROM(k=k) vs. KrylovROM(k=2*k), not the same raw k on both
        (see module docstring).

        Parameters
        ----------
        M, K : ndarray (n_dof, n_dof)
        B : ndarray (n_dof, n_in) or (n_dof,)
        Cout : ndarray (n_out, n_dof) or (n_dof,)
        C : ndarray (n_dof, n_dof), optional
        s0 : float or complex
            Expansion point (default 0, matching krylov.py's own).
        k : int
            Target reduced SECOND-ORDER state count.
        tol : float
            Passed through to soar_basis().
        """
        n = M.shape[0]
        Cout = _as_2d_output_map(Cout)
        dtype = complex if (np.iscomplexobj(M) or np.iscomplexobj(K) or np.iscomplexobj(B)
                             or np.iscomplexobj(Cout) or np.iscomplexobj(s0)
                             or (C is not None and np.iscomplexobj(C))) else float
        M = np.asarray(M, dtype=dtype)
        K = np.asarray(K, dtype=dtype)
        B = _as_2d_input_map(B).astype(dtype, copy=False)
        Cout = Cout.astype(dtype, copy=False)
        C_mat = np.zeros((n, n), dtype=dtype) if C is None else np.asarray(C, dtype=dtype)

        V = soar_basis(M, K, B, k, C=C_mat, s0=s0, tol=tol)
        return cls(M, C_mat, K, B, Cout, V, s0)

    # -----------------------------------------------------------------
    def transfer_function(self, s):
        """H_r(s) = Cout_r (s^2*M_r + s*C_r + K_r)^-1 B_r, the reduced
        model's own SECOND-ORDER transfer function, at one (possibly
        complex) frequency s. Returns an (n_out, n_in) array."""
        Mat = (s ** 2) * self.M_r + s * self.C_r + self.K_r
        X = np.linalg.solve(Mat, self.B_r)
        return self.Cout_r @ X

    def frequency_response(self, omega_array):
        """H_r(i*omega) swept over omega_array -- directly comparable to
        a full-order model's own solve_harmonic()/solve_frequency_sweep()
        output at the SAME output DOF(s), for a unit-amplitude input at
        the SAME input DOF(s) B selects.

        Returns
        -------
        H : ndarray, complex
            Shape (n_omega,) if this system is SISO (n_in==n_out==1),
            else (n_omega, n_out, n_in) -- same convention as
            krylov.KrylovROM.frequency_response().
        """
        omega_array = np.atleast_1d(np.asarray(omega_array, dtype=float))
        n_out, n_in = self.Cout_r.shape[0], self.B_r.shape[1]
        H = np.empty((len(omega_array), n_out, n_in), dtype=complex)
        for i, omega in enumerate(omega_array):
            H[i] = self.transfer_function(1j * omega)
        if n_out == 1 and n_in == 1:
            return H[:, 0, 0]
        return H

    def is_stable(self):
        """True iff every pole of the REDUCED quadratic eigenvalue
        problem det(s^2*M_r + s*C_r + K_r)=0 has negative real part,
        found by linearizing ONLY the small already-reduced (k x k)
        pencil to a cheap 2k x 2k first-order generalized eigenproblem
        -- NOT the full-order 2*n_dof linearization SOAR exists to
        avoid; k is small by construction, so this costs nothing like
        what avoiding first-order form at full order saves.

        Like krylov.KrylovROM.is_stable(), NOT guaranteed True in
        general for an arbitrary basis -- but see module docstring and
        test_soar.py for the passivity-based argument (a DIFFERENT,
        provable guarantee this ROM does have) for why it is expected
        to fare better in practice than KrylovROM's own first-order
        reduction on the same real fixture."""
        k = self.M_r.shape[0]
        dtype = self.M_r.dtype
        Zero = np.zeros((k, k), dtype=dtype)
        I = np.eye(k, dtype=dtype)
        A = np.block([[Zero, I], [-self.K_r, -self.C_r]])
        E = np.block([[I, Zero], [Zero, self.M_r]])
        poles = eig(A, E, right=False)
        poles = poles[np.isfinite(poles)]
        return bool(np.all(poles.real < 0))
