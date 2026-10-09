# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
pod.py -- Proper Orthogonal Decomposition (POD) basis extraction.

Given a "snapshot matrix" X (n_dof, n_snapshots) -- each column a
full-order solution (a static displacement, a mode shape, a transient
state at one time step, whatever is being compressed) -- POD finds the
r-dimensional subspace that captures the most "energy" (Frobenius-norm
variance) of that specific snapshot set, in the least-squares-optimal
sense (Eckart-Young theorem: the best rank-r approximation of X in the
Frobenius norm is exactly its own truncated SVD). This is the
"offline" half of the reduced-basis method: run the full-order model a
handful of times, extract the subspace it actually explores, then let
galerkin.py project the full system onto that subspace for fast
"online" queries.

Two variants are implemented:

STANDARD POD: plain economy SVD of X, X = U*Sigma*V^T, basis = the
first r left singular vectors U[:, :r]. Optimal in the Euclidean
(l2) norm -- correct when every degree of freedom is physically
comparable (e.g. all translational displacements in consistent units).

MASS-WEIGHTED POD: for a structural FE model, this is usually the
WRONG norm to truncate in. A Hermite beam's rotation DOFs (radians)
and translation DOFs (meters) are not on a comparable numerical scale,
and even among translations, physically "important" (high-mass,
high-energy) directions shouldn't be judged by displacement magnitude
alone. The natural, physically meaningful inner product for a
structural model is the ENERGY (mass) inner product <x,y>_M = x^T M y
-- the same inner product that makes ordinary eigenmodes M-orthonormal
(phi_i^T M phi_j = delta_ij). Mass-weighted POD finds the basis that is
optimal in THAT norm instead: Cholesky-factor M = L L^T, SVD the
TRANSFORMED snapshots L^T X, then map the resulting left singular
vectors back through L^{-T} -- a standard trick (e.g. Chatterjee 2000,
"An introduction to the proper orthogonal decomposition") that makes
the extracted POD modes exactly M-orthonormal, i.e. behave under
projection exactly like a truncated eigenbasis would.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np


def _as_array(x):
    """np.asarray(x), preserving complex dtype instead of silently
    discarding the imaginary part.

    Every array-ingesting method in this module used to hardcode
    ``np.asarray(x, dtype=float)``. That's correct for the static/modal
    snapshots this module was originally built for, but it is a silent
    correctness bug the moment a caller passes COMPLEX snapshots --
    e.g. frequency-response solutions x(omega), which are generically
    complex even when M/K/C are real, needed by the frequency-domain
    ROM module. ``dtype=float`` on a complex array raises in modern
    numpy (good, loud) for a genuinely complex array, but a caller
    passing e.g. a float array that's ABOUT to be combined with complex
    data elsewhere would get no warning at all until much later. Using
    plain ``np.asarray`` and letting numpy's own type promotion handle
    real/complex mixing keeps every existing (real) use identical
    (the array already IS float64/int, asarray is a no-op) while making
    complex snapshots simply work instead of silently breaking. Real
    (including integer) input is still promoted to float64, exactly
    matching the old behavior -- only a genuinely complex input skips
    that cast, instead of being truncated by it.
    """
    x = np.asarray(x)
    return x if np.iscomplexobj(x) else x.astype(float, copy=False)


class PodBasis:
    """A POD basis extracted from a snapshot matrix, with either a
    Euclidean or mass-weighted inner product.

    Attributes
    ----------
    V : ndarray, shape (n_dof, n_modes)
        The retained basis vectors (columns), after fit().
    singular_values : ndarray, shape (min(n_dof, n_snapshots),)
        The FULL singular value spectrum (not just the retained part)
        -- kept for diagnostics: plotting the decay, deciding in
        hindsight whether more/fewer modes should have been retained.
    n_modes : int
        Number of retained modes (columns of V).
    mass_weighted : bool
        Whether this basis was extracted with a mass inner product.
    M : ndarray or None
        The mass matrix used, if mass_weighted -- stored so
        orthonormality_error()/project() know which inner product to
        use without the caller having to pass M again.
    """

    def __init__(self):
        self.V = None
        self.singular_values = None
        self.n_modes = None
        self.mass_weighted = False
        self.M = None

    def fit(self, snapshots, n_modes=None, energy_threshold=None, M=None):
        """Extract the POD basis from a snapshot matrix.

        Parameters
        ----------
        snapshots : ndarray, shape (n_dof, n_snapshots)
            Each column is one full-order state to compress.
        n_modes : int, optional
            Explicit truncation rank. Mutually exclusive with
            energy_threshold in spirit (only one is needed), but if
            BOTH are given, the smaller of the two resulting ranks
            wins -- a convenient safety cap ("at most n_modes, unless
            the energy target is reached sooner").
        energy_threshold : float, optional
            Retain the smallest r such that the first r singular
            values capture at least this fraction (e.g. 0.9999) of
            the total "energy" sum(sigma_i^2). Common in RB-method
            practice as the primary truncation criterion, since it is
            interpretable independent of problem size.
        M : ndarray, shape (n_dof, n_dof), optional
            Mass matrix (must be symmetric positive definite). If
            given, uses mass-weighted POD (see module docstring);
            otherwise standard Euclidean POD.

        Returns
        -------
        self, so this can be chained: ``basis = PodBasis().fit(X, n_modes=5)``.
        """
        if n_modes is None and energy_threshold is None:
            raise ValueError("fit() needs at least one of n_modes, energy_threshold")
        X = _as_array(snapshots)
        if X.ndim != 2:
            raise ValueError(f"snapshots must be 2-D (n_dof, n_snapshots), got shape {X.shape}")

        if M is not None:
            self._fit_mass_weighted(X, _as_array(M), n_modes, energy_threshold)
        else:
            self._fit_standard(X, n_modes, energy_threshold)
        return self

    def _fit_standard(self, X, n_modes, energy_threshold):
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        r = self._select_rank(S, n_modes, energy_threshold)
        self.V = U[:, :r]
        self.singular_values = S
        self.n_modes = r
        self.mass_weighted = False
        self.M = None

    def _fit_mass_weighted(self, X, M, n_modes, energy_threshold):
        L = np.linalg.cholesky(M)          # M = L L^T
        X_tilde = L.T @ X                   # transform to the M-orthonormal frame
        U_tilde, S, _ = np.linalg.svd(X_tilde, full_matrices=False)
        r = self._select_rank(S, n_modes, energy_threshold)
        # Phi = L^{-T} @ U_tilde[:, :r], solved rather than inverted explicitly
        V = np.linalg.solve(L.T, U_tilde[:, :r])
        self.V = V
        self.singular_values = S
        self.n_modes = r
        self.mass_weighted = True
        self.M = M

    @staticmethod
    def _select_rank(S, n_modes, energy_threshold):
        candidates = []
        if n_modes is not None:
            candidates.append(min(int(n_modes), len(S)))
        if energy_threshold is not None:
            energy = S**2
            total = np.sum(energy)
            if total <= 0:
                candidates.append(1)
            else:
                cum = np.cumsum(energy) / total
                r = int(np.searchsorted(cum, energy_threshold) + 1)
                candidates.append(min(r, len(S)))
        return max(1, min(candidates))

    def energy_captured(self):
        """Fraction of total snapshot 'energy' (sum of squared singular
        values) captured by the retained n_modes -- a diagnostic for
        whether the truncation was reasonable, always computable even
        if fit() was called with an explicit n_modes rather than an
        energy_threshold."""
        energy = self.singular_values**2
        total = np.sum(energy)
        if total <= 0:
            return 1.0
        return float(np.sum(energy[:self.n_modes]) / total)

    def orthonormality_error(self):
        """max|V^T V - I| (standard) or max|V^T M V - I| (mass-weighted)
        -- should be at machine precision for any basis this class
        produced itself; a nonzero value on a basis assembled by hand
        elsewhere is a real warning sign before using it in
        galerkin.py (a non-orthonormal basis silently breaks the
        M-orthonormal-mode-shape identities galerkin.solve_modal()
        relies on)."""
        G = self.V.T @ (self.M @ self.V) if self.mass_weighted else self.V.T @ self.V
        return float(np.max(np.abs(G - np.eye(self.n_modes))))

    def project(self, x):
        """Full-order state(s) -> reduced coordinates. x may be a
        single vector (n_dof,) or a batch (n_dof, k). Uses the SAME
        inner product the basis was extracted with (V^T M x if
        mass-weighted, V^T x otherwise) -- using the wrong one here is
        a common, silent correctness bug (it still runs, it's just not
        the orthogonal projection the basis was built for)."""
        x = _as_array(x)
        if self.mass_weighted:
            return self.V.T @ (self.M @ x)
        return self.V.T @ x

    def expand(self, q):
        """Reduced coordinates -> full-order approximation. q may be a
        single vector (n_modes,) or a batch (n_modes, k)."""
        return self.V @ _as_array(q)

    def reconstruction_error(self, snapshots, relative=True):
        """||X - V @ project(X)|| (Frobenius), optionally normalized by
        ||X|| -- the direct, model-agnostic measure of how much of the
        ORIGINAL snapshot set this truncated basis can reproduce; the
        Eckart-Young theorem guarantees this is the SMALLEST possible
        reconstruction error among all rank-n_modes subspaces, for
        standard POD in the Euclidean norm (and for mass-weighted POD
        in the M norm -- see reconstruction_error's M-norm variant
        below if that distinction matters for a given check)."""
        X = _as_array(snapshots)
        X_hat = self.expand(self.project(X))
        err = np.linalg.norm(X - X_hat, ord="fro")
        if not relative:
            return float(err)
        denom = np.linalg.norm(X, ord="fro")
        return float(err / denom) if denom > 0 else float(err)


def assemble_field_weight_matrix(n_dof, field_slices, field_grams, scale=1.0):
    """Assemble a full (n_dof, n_dof) block-diagonal weighting/Gram matrix
    `W` for MultiFieldPOD out of one small per-field Gram matrix per field,
    scattered into that field's own DOF positions.

    This is the seam MultiFieldPOD is deliberately built around: per this
    package's own decoupling principle (see the package docstring --
    rom_engine's core modules take plain numpy arrays, never fea_engine
    objects), MultiFieldPOD itself never touches shape functions or
    quadrature. A caller who wants the field Gram matrices built from a
    REAL finite element's shape functions (e.g. a
    ``fea_engine.elements.beams.Beam2DReissner`` rod, whose 3 fields --
    axial displacement, transverse displacement, rotation -- are exactly
    Georgiou (2005)'s setup) computes each field's own consistent Gram
    matrix there (the same ``integral N_k^T N_k ds`` construction
    fea_engine's own element ``mass()`` methods already use, restricted to
    one field's shape functions and Gauss quadrature) and passes the
    pieces in here -- rom_engine never has to import fea_engine to make
    that composition work.

    Parameters
    ----------
    n_dof : int
        Total number of DOFs in one snapshot column (the full multi-field
        state vector length).
    field_slices : sequence of array-like of int
        One integer index array per field, selecting that field's DOF
        positions within a snapshot column. Need not cover every DOF and
        need not be contiguous, but the field index sets must be
        pairwise DISJOINT for the block-diagonal assembly (and for the
        ``sum_k C_km^2 = 1`` identity MultiFieldPOD relies on) to make
        sense.
    field_grams : sequence of ndarray
        One small Gram/mass matrix per field, shape
        ``(len(field_slices[k]), len(field_slices[k]))``, each already
        including whatever normalization convention the caller wants
        (e.g. Georgiou's own bare ``integral a.b ds`` with no extra
        prefactor -- the ``2/L`` scalar is applied separately below, by
        `scale`, so the same per-field Gram can be reused unscaled for a
        different normalization convention later).
    scale : float, optional
        A single scalar applied to the whole assembled matrix -- this is
        where Georgiou's ``2/L`` inner-product prefactor (or any other
        global normalization convention) is plugged in. Deliberately a
        plain keyword, not a hardcoded constant, per the roadmap's own
        note that item 143 should generalize the paper's specific
        normalization rather than lock it in permanently.

    Returns
    -------
    W : ndarray, shape (n_dof, n_dof)
    """
    W = np.zeros((n_dof, n_dof))
    for idx, Gk in zip(field_slices, field_grams):
        idx = np.asarray(idx)
        Gk = _as_array(Gk)
        W[np.ix_(idx, idx)] += Gk
    return scale * W


def trapezoidal_field_gram(s, scale=1.0):
    """A simple, FE-shape-function-free example of a per-field Gram
    matrix: the LUMPED (diagonal) composite-trapezoidal-rule integration
    weights for a field sampled at 1-D coordinates `s` -- i.e. the
    diagonal approximation to ``integral phi_i(s) phi_j(s) ds`` that
    treats each node's own contribution as independent of its neighbors'
    (exact for a piecewise-constant field centered on each node's own
    trapezoid panel; an approximation, not the exact consistent Gram, for
    a piecewise-linear field -- good enough for synthetic/example use and
    for cases where only a rough spatial weighting is available, but a
    real FE Gram matrix from `fea_engine` shape functions, fed through
    `assemble_field_weight_matrix`, is more accurate for production use).

    Parameters
    ----------
    s : array-like, shape (n,)
        Strictly increasing 1-D nodal coordinates for this one field.
    scale : float, optional
        Extra scalar factor (e.g. folded into the `scale` of
        `assemble_field_weight_matrix`, or applied here directly).

    Returns
    -------
    G : ndarray, shape (n, n)
        Diagonal Gram matrix; ``G[i, i]`` is half the sum of the two
        panel widths adjoining node i (the standard trapezoidal-rule
        nodal weight), or the single adjoining half-panel at the two
        ends.
    """
    s = np.asarray(s, dtype=float)
    n = len(s)
    w = np.zeros(n)
    d = np.diff(s)
    w[:-1] += d / 2
    w[1:] += d / 2
    return scale * np.diag(w)


class MultiFieldPOD:
    """Multi-field POD diagnostics on top of `PodBasis`, for snapshots
    whose state vector interleaves several physically DISTINCT fields
    (Georgiou 2005, "Advanced Proper Orthogonal Decomposition Tools...
    Nonlinear Rods," *Nonlinear Dynamics* 41:69-110, Eqs. 6-19) -- e.g. a
    planar rod's axial displacement, transverse displacement, and
    rotation, but genuinely generic in the NUMBER of fields, not
    hardcoded to exactly three.

    This class does not replace `PodBasis` or duplicate its SVD/Cholesky
    algebra: it fits one, using the caller-supplied weighting/Gram matrix
    `W` as `PodBasis`'s own `M` (mass-weighted POD, module docstring
    above), on RAW -- explicitly NOT mean-subtracted -- snapshots (there
    is no mean-subtraction anywhere in this module; `PodBasis.fit()`
    never centers its input, so passing raw snapshots straight through
    already gives the physically meaningful, uncentered component norms
    this class's diagnostics need), then adds the per-field
    decompositions (component norms, normalized component shapes,
    amplitude histories, field-separated reconstruction, and a
    deterministic cross-run sign convention) the paper's own analysis
    needs on top of that fitted basis.

    THE INNER PRODUCT. The paper's own weighted inner product is
    ``<a,b> = (2/L) * integral_0^L a . b ds`` -- a single global scalar
    (``2/L``) times a plain dot-product spatial integral summed over all
    three fields. This class does not hardcode that -- it takes an
    already-assembled weighting/Gram matrix `W` (plain ndarray, built
    however the caller likes; see `assemble_field_weight_matrix` and
    `trapezoidal_field_gram` above for one concrete, dependency-free way
    to build one, or a real FE shape-function-based Gram from
    `fea_engine` for production use) and uses it exactly as `PodBasis`
    already uses a mass matrix. Georgiou's own ``2/L`` normalization is
    just one particular choice of `W`'s overall scale -- a caller
    reproducing the paper folds ``2/L`` into `W` (e.g. via
    `assemble_field_weight_matrix`'s own `scale=` argument); a caller
    who wants a different weighting convention (a true FE-consistent
    Gram, a different global normalization, or no weighting at all,
    i.e. `W = identity`) uses a differently-built `W` instead. Nothing
    about the paper's specific ``2/L`` choice is baked into this class.

    CRITICAL ASSUMPTION: `W` must be block-diagonal ACROSS fields (no
    coupling between, say, an axial DOF and a rotation DOF in `W`) --
    exactly what Georgiou's own inner product is, since it sums three
    independent per-field dot products rather than mixing them. This is
    what makes the ``sum_k C_km^2 = 1`` identity (Eq. 10 in spirit) hold:
    the whole-vector M-orthonormality `PodBasis` already guarantees
    (``V[:, m]^T W V[:, m] = 1``) only splits cleanly into a sum of
    per-field quadratic forms when the cross-field blocks of `W` are
    exactly zero. `fit()` checks this (see `check_block_diagonal`) and
    raises if it is not satisfied within tolerance, rather than silently
    returning component norms that don't actually sum to 1.

    SIGN CONVENTION. POD modes are only defined up to an overall sign
    (`V` and `-V` are equally valid left singular vectors), which makes
    per-mode component shapes incomparable across independent runs (a
    re-fit on a different-but-equivalent snapshot set could flip any
    mode's sign) unless something pins it down. This class's convention:
    for each retained mode `m`, look at the SUB-VECTOR of that mode
    restricted to `sign_reference_field` (field 0 by default -- e.g. the
    rod's axial field, chosen as the reference here since the paper's
    Table 17 axial component norms are its most consistently reported
    diagnostic across modes; use whichever field is physically most
    "always present" for a different application via the
    `sign_reference_field=` constructor argument), find the entry with
    the LARGEST MAGNITUDE in that sub-vector, and flip the whole column's
    sign (and the matching row of the amplitude history `Q`) if that
    entry is negative. This is the roadmap's own suggested "positive
    midspan transverse component" convention, generalized: a literal
    midspan node index is not universally meaningful for arbitrary field
    layouts, but "the largest-magnitude entry of a fixed reference field"
    is the same idea (pin the sign using the spatial location where the
    mode's presence in that field is least ambiguous) made well-defined
    for any field layout, including fields with an even number of nodes
    (no single literal midpoint) or non-uniform spacing.

    Attributes (after fit())
    -------------------------
    basis : PodBasis
        The underlying fitted mass-weighted PodBasis (``M = W``), with
        the sign convention above already applied to `basis.V`.
    Q : ndarray, shape (n_modes, n_snapshots)
        Amplitude histories `Q_m(t)` -- the mass-weighted projection
        coefficients of each snapshot onto each retained mode
        (``V^T W X``, i.e. `basis.project(X)`), sign-consistent with
        `basis.V`.
    C : ndarray, shape (n_fields, n_modes)
        Per-mode, per-field component norms `C_km`
        (``C[k, m] = sqrt(V[idx_k, m]^T @ W_kk @ V[idx_k, m])``, always
        >= 0), satisfying ``sum_k C[k, m]**2 == 1`` for every mode `m`.
    component_shapes : list of ndarray
        One array per field, shape ``(len(field_slices[k]), n_modes)``:
        the normalized component shapes `phi_km(s)`
        (``component_shapes[k][:, m] = V[idx_k, m] / C[k, m]``),
        satisfying ``phi_km^T @ W_kk @ phi_km == 1`` whenever
        ``C[k, m] > 0`` (a mode with essentially zero presence in a
        field has an ill-defined shape there and is left as all-zero
        rather than divided by a near-zero norm).
    energy_fractions : ndarray, shape (n_modes,)
        ``lambda_m / sum(lambda)`` for each retained mode, with the
        denominator the FULL (untruncated) singular-value spectrum's
        total energy -- so these fractions sum to 1 only when every mode
        is retained (n_modes == full rank), by construction, matching
        the roadmap's own validation criterion.
    """

    def __init__(self, field_slices, W, sign_reference_field=0):
        self.field_slices = [np.asarray(idx, dtype=int) for idx in field_slices]
        if not (0 <= sign_reference_field < len(self.field_slices)):
            raise ValueError(
                f"sign_reference_field={sign_reference_field} out of range "
                f"for {len(self.field_slices)} fields")
        self.sign_reference_field = sign_reference_field
        self.W = _as_array(W)
        if self.W.ndim != 2 or self.W.shape[0] != self.W.shape[1]:
            raise ValueError(f"W must be square, got shape {self.W.shape}")

        self.basis = None
        self.Q = None
        self.C = None
        self.component_shapes = None
        self.energy_fractions = None

    def _check_block_diagonal(self, tol=1e-9):
        scale = np.max(np.abs(self.W)) if self.W.size else 0.0
        if scale <= 0:
            return
        for i, idx_i in enumerate(self.field_slices):
            for j, idx_j in enumerate(self.field_slices):
                if i == j:
                    continue
                block = self.W[np.ix_(idx_i, idx_j)]
                if block.size and np.max(np.abs(block)) > tol * scale:
                    raise ValueError(
                        "W has non-negligible coupling between field "
                        f"{i} and field {j} (max |W_ij| = "
                        f"{np.max(np.abs(block)):.3e}, vs overall scale "
                        f"{scale:.3e}). MultiFieldPOD's per-field "
                        "decomposition (sum_k C_km^2 == 1) assumes W is "
                        "block-diagonal across fields -- pass "
                        "check_block_diagonal=False to fit() only if this "
                        "coupling is deliberate and the caller does not "
                        "need that identity to hold exactly.")

    def fit(self, snapshots, n_modes=None, energy_threshold=None, check_block_diagonal=True):
        """Fit the underlying mass-weighted PodBasis (M=W) on RAW
        (not mean-subtracted) snapshots, then compute every multi-field
        diagnostic on top of it.

        Parameters
        ----------
        snapshots : ndarray, shape (n_dof, n_snapshots)
            RAW full-state snapshots (each column one time instant / load
            case), on the same DOF numbering `field_slices` indexes into.
            Deliberately NOT mean-subtracted before fitting -- see the
            class docstring.
        n_modes, energy_threshold :
            Passed straight through to `PodBasis.fit()`.
        check_block_diagonal : bool, optional
            If True (default), verify `W` has no cross-field coupling
            before proceeding (see `_check_block_diagonal`).

        Returns
        -------
        self
        """
        if check_block_diagonal:
            self._check_block_diagonal()

        X = _as_array(snapshots)
        basis = PodBasis().fit(X, n_modes=n_modes, energy_threshold=energy_threshold, M=self.W)

        V = basis.V.copy()
        ref_idx = self.field_slices[self.sign_reference_field]
        signs = np.ones(V.shape[1])
        for m in range(V.shape[1]):
            col = V[ref_idx, m]
            k = int(np.argmax(np.abs(col)))
            if col[k] < 0:
                signs[m] = -1.0
        V = V * signs[None, :]
        basis.V = V
        self.basis = basis

        # basis.V already has the sign convention baked in, so
        # basis.project(X) (== V^T W X with the CORRECTED V) already
        # reflects the sign flip -- do not apply `signs` a second time
        # here, or modes whose sign was flipped would be double-flipped
        # back and no longer match V @ Q == the original snapshots.
        self.Q = basis.project(X)
        self._compute_component_norms_and_shapes()
        self._compute_energy_fractions()
        return self

    def _compute_component_norms_and_shapes(self):
        V = self.basis.V
        n_modes = self.basis.n_modes
        n_fields = len(self.field_slices)
        C = np.zeros((n_fields, n_modes))
        shapes = [np.zeros((len(idx), n_modes)) for idx in self.field_slices]
        for m in range(n_modes):
            for k, idx in enumerate(self.field_slices):
                Wk = self.W[np.ix_(idx, idx)]
                vk = V[idx, m]
                c2 = float(vk @ (Wk @ vk))
                c2 = max(c2, 0.0)   # guard tiny negative roundoff
                c = np.sqrt(c2)
                C[k, m] = c
                if c > 1e-13:
                    shapes[k][:, m] = vk / c
                # else: this mode has ~zero presence in this field --
                # its "normalized shape" is ill-defined; leave it zero
                # rather than blow up dividing by ~0.
        self.C = C
        self.component_shapes = shapes

    def _compute_energy_fractions(self):
        S = self.basis.singular_values
        total = float(np.sum(S**2))
        if total <= 0:
            self.energy_fractions = np.zeros(self.basis.n_modes)
            return
        self.energy_fractions = (S[:self.basis.n_modes]**2) / total

    def reconstruct_field(self, field_index, Q=None):
        """Field-separated reconstruction: the time history of just ONE
        field, reconstructed from POD amplitudes -- e.g. pull out just
        the transverse-displacement time history for plotting, without
        the axial/rotation components cluttering the array.

        Parameters
        ----------
        field_index : int
            Which field (index into `field_slices`) to reconstruct.
        Q : ndarray, shape (n_modes, n_snapshots), optional
            Amplitude history to use. Defaults to `self.Q` (the fitted
            snapshots' own amplitudes); pass a different `Q` (e.g. from
            `IntrusiveNonlinearROM`-integrated reduced coordinates, item
            144) to reconstruct a field's time history from a REDUCED
            SIMULATION instead of the original training snapshots.

        Returns
        -------
        ndarray, shape (len(field_slices[field_index]), n_snapshots)
        """
        if Q is None:
            Q = self.Q
        else:
            Q = _as_array(Q)
        idx = self.field_slices[field_index]
        return self.basis.V[idx, :] @ Q
