"""
krylov.py -- Krylov-subspace moment-matching ROM (Besselink et al. 2013,
Section 3 / eqs. 21-36), the paper's "numerical mathematics" family.

The transfer function of a first-order state-space system (see
state_space.py) H(s) = Cout (s*E - A)^-1 B has a Taylor expansion around
any point s0 whose coefficients ("moments") are, up to bookkeeping, the
vectors of the Krylov subspace

    K_k((A - s0*E)^-1 E, (A - s0*E)^-1 B)

so an ORTHONORMAL basis V of that subspace, used as a one-sided
(Galerkin) projection (A_r = V^T A V, E_r = V^T E V, B_r = V^T B,
Cout_r = Cout V -- the SAME projection pattern galerkin.py already uses,
just on the first-order form instead of K/M directly), gives a reduced
model whose transfer function matches the first several moments of the
full-order one AT s0 -- exactly at s0, with error growing away from it.
This is fundamentally a LOCAL approximation (unlike balanced truncation,
which is accurate globally), and this module deliberately does not try
to hide that: KrylovROM.frequency_response() should be expected to look
very good near s0 and progressively worse away from it.

This module builds the basis via ONE-SIDED (Galerkin) block Arnoldi by
default -- matching the paper's own eqs. 21-31 exactly, and its own
worked-example method (the paper's numerical results ARE one-sided
Arnoldi at s0=0). A TWO-SIDED (Petrov-Galerkin) extension is also
available (KrylovROM.from_MCK(..., two_sided=True)): it builds a SECOND
Krylov basis W from the dual system (A^H, E^H, Cout^H) and
biorthogonalizes it against V (W^H E V = I), giving an oblique
projection that matches roughly TWICE as many moments from the same
basis size -- k' input-side moments from V, k' more output-side moments
from W (Grimme 1997; Bai 2002 Section 4). This is exactly the
"genuinely fragile" extension flagged in docs/classical_mor_roadmap.md
Section 7 Phase 3: biorthogonalizing two INDEPENDENTLY built Krylov
bases can be ill-conditioned when the input and output Krylov subspaces
are nearly degenerate with each other for a given port/s0/k, and
two_sided_arnoldi_bases() raises a clear ValueError rather than
silently returning a numerically meaningless basis when that happens --
callers hitting it should fall back to one-sided (the default), try a
different s0, or request fewer moments (a smaller k). A genuinely
SECOND-ORDER-structure-preserving Krylov method (SOAR/TOAR, building
the subspace directly in the (M,C,K) pencil instead of first collapsing
to the 2*n_dof first-order form) is still out of scope here, already
flagged as a deferred later phase in
docs/frequency_domain_rom_roadmap.md Section 2b -- the paper's own
method is the plain first-order approach implemented here, so matching
the paper is simpler, not harder, than the structure-preserving
alternative.

IMPORTANT, and worth repeating from the module built on top of this one:
Krylov moment matching gives NO a priori stability guarantee. The
paper's own benchmark found its k=20 reduced model unstable on a
posteriori inspection, despite the full-order model being passive/stable
and both its other two methods (mode displacement, balanced truncation)
remaining stable at the same order. KrylovROM.is_stable() checks this
directly -- callers should check it, not assume a Galerkin-projected
model inherits the full-order model's stability (it generally does NOT,
unlike balanced truncation, where stability preservation is a theorem).
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve, eig

from .state_space import to_state_space


def arnoldi_basis(A, E, B, k, s0=0.0, tol=1e-12):
    """Orthonormal (modified Gram-Schmidt) basis of the order-k block
    Krylov subspace K_k((A - s0*E)^-1 E, (A - s0*E)^-1 B), built from
    ONE dense LU factorization of (A - s0*E), reused for every
    iteration (each new "generation" of directions costs one E-multiply
    plus one triangular back-substitution per column, not a fresh
    solve).

    Parameters
    ----------
    A, E : ndarray (n, n)
        The state-space pencil (state_space.to_state_space(..., form="E")).
    B : ndarray (n, n_in)
        Input map -- ALL n_in columns seed the block Krylov recursion
        (this is "block Arnoldi": for n_in > 1, each iteration advances
        every input direction at once, deflating any candidate direction
        that has become numerically dependent on what's already been
        kept -- a real occurrence once multiple input sequences start
        overlapping the same dominant subspace).
    k : int
        Target basis size (total columns, NOT per input -- e.g. k=10
        with 2 inputs gives roughly 5 "generations" of the block
        recursion, not 10 each).
    s0 : float or complex
        Expansion point. s0=0 (the default) matches the paper's own
        choice; any other point is the ordinary shift-and-invert
        generalization of the same construction.
    tol : float
        A candidate direction with norm below this (after being
        orthogonalized against everything already kept) is judged
        numerically dependent and dropped (deflated) rather than
        normalized and kept, since dividing by a ~0 norm would inject
        numerical noise into the basis, not real new subspace content.

    Returns
    -------
    V : ndarray (n, k') with k' <= k
        k' can be smaller than k if the Krylov subspace closes
        (deflates completely) before reaching the requested size --
        this happens for a genuinely low-order system, and is reported
        via V's own actual shape rather than silently padded.
    """
    n, n_in = B.shape
    A = np.asarray(A, dtype=complex if np.iscomplexobj(A) or np.iscomplexobj(E) or np.iscomplexobj(B) or np.iscomplexobj(s0) else float)
    E = np.asarray(E, dtype=A.dtype)
    B = np.asarray(B, dtype=A.dtype)
    As = A - s0 * E
    lu, piv = lu_factor(As)

    V_cols = []
    generation = lu_solve((lu, piv), B)
    while len(V_cols) < k:
        new_gen = []
        for j in range(generation.shape[1]):
            w = generation[:, j].copy()
            for v in V_cols:
                w = w - (np.vdot(v, w)) * v
            for v in new_gen:
                w = w - (np.vdot(v, w)) * v
            norm_w = np.linalg.norm(w)
            if norm_w > tol:
                w = w / norm_w
                new_gen.append(w)
                V_cols.append(w)
                if len(V_cols) == k:
                    break
        if not new_gen:
            break   # subspace numerically exhausted before reaching k
        generation = lu_solve((lu, piv), E @ np.column_stack(new_gen))
    if not V_cols:
        raise ValueError("Krylov subspace is empty -- B is (numerically) zero?")
    return np.column_stack(V_cols)


def two_sided_arnoldi_bases(A, E, B, Cout, k, s0=0.0, tol=1e-12, cond_tol=1e12):
    """Build a biorthogonalized pair of Krylov bases (V, W) for TWO-SIDED
    (Petrov-Galerkin) moment matching:

      - V spans the "input"/right Krylov subspace
        K_k((A-s0*E)^-1 E, (A-s0*E)^-1 B) -- IDENTICAL to what
        arnoldi_basis() itself builds.
      - W_tilde spans the "output"/left Krylov subspace, built the SAME
        way but from the DUAL system (A^H, E^H, Cout^H) -- the standard
        trick that lets one Arnoldi routine build both bases (Cout^H
        plays the role B plays for V).
      - V and W_tilde are then biorthogonalized (W = W_tilde @ X for an
        X chosen so W^H E V = I exactly) rather than used as-is, because
        an oblique (Petrov-Galerkin) projection needs a genuinely
        biorthogonal pair, not just two independently-orthonormal bases.

    This is the classical two-sided Krylov construction (Grimme 1997;
    Bai 2002 Section 4): projecting with W on the left and V on the
    right (A_r = W^H A V, ...) matches roughly TWICE as many moments of
    H(s) around s0 as the same-size ONE-SIDED (V-only, W=V) projection
    arnoldi_basis() feeds into KrylovROM by default -- k' from the
    input-side Krylov sequence, k' more from the output-side one.

    Parameters
    ----------
    A, E : ndarray (n, n)
    B : ndarray (n, n_in)
    Cout : ndarray (n_out, n)
    k : int
        Target size for EACH of the two one-sided bases before
        biorthogonalization (so the final basis pair has size
        min(k'_V, k'_W) <= k, not 2*k -- see Returns).
    s0 : float or complex
    tol : float
        Passed through to both underlying arnoldi_basis() calls.
    cond_tol : float
        If the biorthogonalization matrix G = W_tilde^H E V is
        ill-conditioned beyond this threshold, raises ValueError instead
        of returning a numerically meaningless basis pair -- this IS the
        "genuinely fragile" failure mode the module docstring warns
        about (it happens when the input and output Krylov subspaces
        are nearly degenerate with each other for this port/s0/k), not
        a bug to silently paper over.

    Returns
    -------
    V, W : ndarray (n, k'), k' <= k
        Biorthogonalized so W.conj().T @ E @ V is the k'x k' identity
        (checked, not just asserted, by the caller's own tests). If
        block deflation (see arnoldi_basis()) triggers at a different
        point for the input- and output-side sequences, both are
        truncated to the smaller common size k' = min(...) to keep them
        the same shape -- a simplification: no separate re-optimization
        is done to realign columns beyond that common size.
    """
    V = arnoldi_basis(A, E, B, k, s0=s0, tol=tol)
    A_H = A.conj().T if np.iscomplexobj(A) else A.T
    E_H = E.conj().T if np.iscomplexobj(E) else E.T
    Cout_H = Cout.conj().T if np.iscomplexobj(Cout) else Cout.T
    W_tilde = arnoldi_basis(A_H, E_H, Cout_H, k, s0=np.conj(s0), tol=tol)

    kk = min(V.shape[1], W_tilde.shape[1])
    V = V[:, :kk]
    W_tilde = W_tilde[:, :kk]

    G = W_tilde.conj().T @ E @ V
    cond = np.linalg.cond(G)
    if cond > cond_tol:
        raise ValueError(
            f"two-sided biorthogonalization is ill-conditioned "
            f"(cond(G)={cond:.3e} > {cond_tol:.0e}) -- the input and "
            f"output Krylov subspaces are nearly degenerate with each "
            f"other for this port/s0/k. This is the documented "
            f"fragility of two-sided moment matching (see module "
            f"docstring), not a bug -- try one_sided (two_sided=False, "
            f"the default), a different s0, or a smaller k."
        )
    # solve for X with G^H X = I (X = inv(G^H)), then W = W_tilde @ X,
    # giving W^H E V = X^H (W_tilde^H E V) = X^H G = (G^-H)^H G = G^-1 G = I
    X = np.linalg.solve(G.conj().T, np.eye(kk, dtype=G.dtype))
    W = W_tilde @ X
    return V, W


class KrylovROM:
    """A Krylov-subspace moment-matching ROM -- one-sided (Galerkin) by
    default, or two-sided (Petrov-Galerkin, matching roughly twice as
    many moments from the same basis size) if built with
    from_MCK(..., two_sided=True). Most callers should use from_MCK()
    rather than this constructor.

    Attributes
    ----------
    ss : state_space.StateSpaceSystem
        The full-order first-order (descriptor) form this was built from.
    V : ndarray (2*n_dof, k')
        The (right) Arnoldi basis.
    W : ndarray (2*n_dof, k'), or None
        The (left) biorthogonalized basis, only when two_sided=True.
        None for a one-sided ROM -- see two_sided property.
    s0 : complex
        The expansion point V (and W, if present) were built around.
    A_r, E_r, B_r, Cout_r : ndarray
        The reduced pencil -- Galerkin- (W=V) or Petrov-Galerkin-
        (W != V) projected depending on two_sided.
    """

    def __init__(self, ss, V, s0, W=None):
        self.ss = ss
        self.V = V
        self.W = W
        self.s0 = s0
        W_eff = V if W is None else W
        self.A_r = W_eff.conj().T @ ss.A @ V
        self.E_r = W_eff.conj().T @ ss.E @ V
        self.B_r = W_eff.conj().T @ ss.B
        self.Cout_r = ss.Cout @ V

    @property
    def two_sided(self):
        return self.W is not None

    @classmethod
    def from_MCK(cls, M, K, B, Cout, C=None, s0=0.0, k=10, two_sided=False):
        """Build a KrylovROM directly from second-order mass/stiffness
        (and, optionally, damping) matrices plus an input map B and
        output map Cout -- the usual entry point, mirroring
        frequency.FrequencyROM.from_MCK's convention.

        Parameters
        ----------
        M, K : ndarray (n_dof, n_dof)
        B : ndarray (n_dof, n_in) or (n_dof,)
        Cout : ndarray (n_out, n_dof) or (n_dof,)
        C : ndarray (n_dof, n_dof), optional
        s0 : float or complex
            Expansion point (default 0, matching the paper).
        k : int
            Target reduced STATE-SPACE order (i.e. the size of the
            first-order pencil after reduction -- this is DOUBLE the
            "number of second-order modes" a mode-displacement basis of
            comparable accuracy would use, since the state here is
            [q; q_dot], not q alone; see module docstring of
            state_space.py). For two_sided=True, this is the target
            size of EACH one-sided basis before biorthogonalization
            (see two_sided_arnoldi_bases()) -- the resulting reduced
            order can be smaller if block deflation triggers, and the
            number of MOMENTS matched is roughly 2x this reduced order,
            not 2x k.
        two_sided : bool
            False (default): one-sided Galerkin projection (matches the
            paper's own worked-example method exactly). True: two-sided
            Petrov-Galerkin projection (two_sided_arnoldi_bases()) --
            matches more moments per reduced state, at the cost of a
            documented, real biorthogonalization fragility (see that
            function's own docstring) that can raise ValueError for
            some port/s0/k combinations.
        """
        ss = to_state_space(M, K, C=C, B=B, Cout=Cout, form="E")
        if two_sided:
            V, W = two_sided_arnoldi_bases(ss.A, ss.E, ss.B, ss.Cout, k, s0=s0)
            return cls(ss, V, s0, W=W)
        V = arnoldi_basis(ss.A, ss.E, ss.B, k, s0=s0)
        return cls(ss, V, s0)

    # -----------------------------------------------------------------
    def transfer_function(self, s):
        """H_r(s) = Cout_r (s*E_r - A_r)^-1 B_r, the reduced model's
        own transfer function, at one (possibly complex) frequency s.
        Returns an (n_out, n_in) array."""
        M = s * self.E_r - self.A_r
        X = np.linalg.solve(M, self.B_r)
        return self.Cout_r @ X

    def frequency_response(self, omega_array):
        """H_r(i*omega) swept over omega_array -- the reduced model's
        harmonic transfer function, directly comparable to a full-order
        model's own solve_harmonic()/solve_frequency_sweep() output at
        the SAME output DOF(s), for a unit-amplitude input at the SAME
        input DOF(s) B selects.

        Returns
        -------
        H : ndarray, complex
            Shape (n_omega,) if this system is SISO (n_in == n_out == 1,
            the common case for comparing directly against a single
            fea_engine solve_harmonic() trace), else
            (n_omega, n_out, n_in).
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
        """True iff every pole of the REDUCED model (generalized
        eigenvalues of (A_r, E_r)) has negative real part. Unlike
        balanced_truncation.BalancedTruncationROM.is_stable(), this is
        NOT guaranteed to be True -- see module docstring -- and is
        provided specifically so callers can check it rather than
        assume it."""
        poles = eig(self.A_r, self.E_r, right=False)
        poles = poles[np.isfinite(poles)]
        return bool(np.all(poles.real < 0))
