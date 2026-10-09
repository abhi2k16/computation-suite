# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
frequency.py -- intrusive frequency-domain reduced-order modeling for
linear structural dynamics.

Under harmonic excitation F(t) = Re[F * e^{i*omega*t}], the steady-
state response x(t) = Re[X(omega) * e^{i*omega*t}] satisfies

    A(omega) X(omega) = F,      A(omega) = -omega^2*M + i*omega*C + K

`A(omega)` is complex and, critically, ill-conditioned near a
resonance (it becomes exactly singular, at the corresponding
undamped natural frequency, if C were zero there). A frequency sweep
-- evaluate X(omega) at hundreds or thousands of omega values, e.g.
for a Bode plot, a resonance search, or a PSD/random-vibration
analysis -- is exactly the kind of "same structure, many parameter
queries" problem the rest of this package (pod.py, galerkin.py,
affine.py) already exists to accelerate.

The key structural fact this module leans on: A(omega) is EXACTLY
affine in the 3-term basis {M, C, K}, with parameter-dependent
coefficients theta(omega) = [-omega^2, i*omega, 1]. That is precisely
the abstraction affine.py already implements -- this module does not
introduce new projection algebra, it COMPOSES AffineDecomposition (for
the offline/online split) with GalerkinROM (for basis projection of
the load vector and expansion back to full coordinates) around that
one physically-specific theta(omega). See
docs/frequency_domain_rom_roadmap.md for the full research/design
background (Krylov/SOAR alternatives, certified error estimation,
POD-on-FRF-snapshots vs. modal bases) -- this module implements that
document's Phase 1 (affine-reuse core) and Phase 2 (POD-on-FRF-
snapshots basis option).

Damping models
--------------
Three cases, in increasing generality (all handled by from_MCK()):

  - Undamped (C = 0): 2-term {M, K} decomposition,
    theta(omega) = [-omega^2, 1].
  - Proportional/Rayleigh damping, C = alpha*M + beta*K (the case
    fea_engine.damping.RayleighDamping already represents): the i*omega*C
    term folds additively into the M and K coefficients, so this is
    STILL a 2-term {M, K} decomposition,
    theta(omega) = [-omega^2 + i*omega*alpha, 1 + i*omega*beta] --
    algebraically identical to the general 3-term form (see
    test_frequency.py's collapse-equivalence check) but half the
    offline projection cost.
  - General (non-proportional) damping, an explicit C matrix that
    isn't a linear combination of M and K: the full 3-term {M, C, K}
    decomposition, theta(omega) = [-omega^2, i*omega, 1].
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .affine import AffineDecomposition
from .galerkin import GalerkinROM
from .pod import PodBasis


def _theta_undamped(omega):
    omega = float(np.real(omega))
    return np.array([-omega**2, 1.0], dtype=complex)


def _theta_rayleigh(omega, alpha, beta):
    omega = float(np.real(omega))
    return np.array([-omega**2 + 1j * omega * alpha, 1.0 + 1j * omega * beta], dtype=complex)


def _theta_general(omega):
    omega = float(np.real(omega))
    return np.array([-omega**2, 1j * omega, 1.0], dtype=complex)


class FrequencyROM:
    """An intrusive reduced-order model of A(omega) = -omega^2*M +
    i*omega*C + K, built by composing affine.AffineDecomposition (for
    the {M, C, K} affine-in-omega structure) with galerkin.GalerkinROM
    (for projecting loads and expanding solutions).

    Most callers should use the from_MCK() classmethod rather than
    this constructor directly -- it builds the right theta_func
    automatically for the undamped / proportional / general-damping
    cases described in the module docstring, instead of requiring the
    caller to hand-derive theta(omega).

    Parameters
    ----------
    components : sequence of ndarray (n_dof, n_dof)
        The affine component matrices (2 or 3 of them -- see module
        docstring). Passed straight through to AffineDecomposition.
    theta_func : callable
        omega -> array-like of length len(components), matching
        AffineDecomposition's theta_func convention exactly (mu is
        simply a scalar omega here instead of a tuple of material
        parameters -- AffineDecomposition itself makes no assumption
        about what mu "is").
    basis : ndarray (n_dof, n_modes), or a fitted pod.PodBasis
        The reduced basis to project onto -- e.g. undamped mode shapes
        from a GalerkinROM(...).solve_modal()-style eigensolve, or
        PodBasis.fit() on FRF snapshots from
        build_pod_basis_from_frf_snapshots() below.
    """

    def __init__(self, components, theta_func, basis):
        self.affine = AffineDecomposition(components, theta_func).project(basis)
        self.galerkin = GalerkinROM(basis)

    @classmethod
    def from_MCK(cls, M, K, basis, C=None, rayleigh=None):
        """Build a FrequencyROM directly from mass/stiffness (and,
        optionally, damping) matrices -- the usual entry point.

        Parameters
        ----------
        M, K : ndarray (n_dof, n_dof)
            Mass and stiffness matrices.
        basis : ndarray or PodBasis
            Reduced basis to project onto.
        C : ndarray (n_dof, n_dof), optional
            An explicit (possibly non-proportional) damping matrix.
            If given, builds the full 3-term {M, C, K} decomposition.
        rayleigh : (alpha, beta) tuple, optional
            Proportional damping C = alpha*M + beta*K. If given
            (instead of C), builds the algebraically-equivalent but
            cheaper 2-term {M, K} decomposition described in the
            module docstring. Mutually exclusive with C.

        If neither C nor rayleigh is given, the system is treated as
        undamped (a 2-term {M, K} decomposition with
        theta(omega) = [-omega^2, 1]) -- a reasonable default for a
        first look at a structure's resonances, though real damping
        should be added before trusting amplitudes near resonance,
        where an undamped model is singular.
        """
        if C is not None and rayleigh is not None:
            raise ValueError("give at most one of C, rayleigh (they're two ways to "
                              "specify the same physical quantity)")
        if C is not None:
            components = [M, C, K]
            theta_func = _theta_general
        elif rayleigh is not None:
            alpha, beta = rayleigh
            components = [M, K]
            theta_func = lambda omega, a=alpha, b=beta: _theta_rayleigh(omega, a, b)
        else:
            components = [M, K]
            theta_func = _theta_undamped
        return cls(components, theta_func, basis)

    # -----------------------------------------------------------------
    # Online queries
    # -----------------------------------------------------------------
    def solve(self, omega, F_r):
        """Solve the reduced system A_r(omega) q = F_r for q, at one
        frequency. F_r must already be in reduced coordinates (e.g.
        from self.galerkin.project_vector(F)) -- this mirrors
        affine.AffineDecomposition.solve_reduced()'s existing
        convention exactly, just under this class's name."""
        A_r = self.affine.assemble_reduced(omega)
        return np.linalg.solve(A_r, F_r)

    def frequency_response(self, omega_array, F, output_dofs=None):
        """Sweep omega_array, returning the FULL-order (expanded)
        response at each frequency -- the main "many-query" entry
        point this whole module exists for.

        Parameters
        ----------
        omega_array : array-like of float
            Driving frequencies (rad/s) to sweep.
        F : ndarray (n_dof,)
            The (frequency-INDEPENDENT) load vector -- projected onto
            the reduced basis ONCE, outside the sweep loop. A
            frequency-DEPENDENT load needs a per-omega call to
            solve() directly instead (project F(omega) fresh each
            time), since this convenience method assumes a fixed F.
        output_dofs : array-like of int, optional
            If given, only these full-order DOF indices are returned
            (e.g. a single sensor/output location) instead of the
            full (n_dof,) state at every frequency -- much cheaper
            when only a few outputs matter, since the (n_dof,)
            expansion V @ q is still done, but only the requested rows
            are kept, matching the shape convention pyMOR's transfer-
            function evaluation uses (leading axis = frequency).

        Returns
        -------
        ndarray, complex, shape (len(omega_array), n_out)
            n_out = n_dof if output_dofs is None, else
            len(output_dofs).
        """
        F_r = self.galerkin.project_vector(F)
        V = self.galerkin.V
        n_dof = V.shape[0]
        out_dim = n_dof if output_dofs is None else len(output_dofs)
        result = np.zeros((len(omega_array), out_dim), dtype=complex)
        for i, omega in enumerate(omega_array):
            q = self.solve(omega, F_r)
            x = V @ q
            result[i, :] = x if output_dofs is None else x[output_dofs]
        return result

    def residual_norm(self, omega, F):
        """The EXACT full-order residual norm
        ||F - A(omega) @ (V @ q)|| of the expanded reduced solution at
        this omega -- computed EFFICIENTLY, in O(Q * n_dof * n_modes),
        via AffineDecomposition.assemble_action() rather than
        O(n_dof^2) via assemble() (see
        docs/phase4_error_bounds_greedy_roadmap.md Section 2 for the
        derivation). Since n_modes << n_dof for any basis worth
        reducing to, this is cheap enough to call many times inside a
        greedy training loop's inner sweep over candidate frequencies
        -- the use case error_estimate() (below) was originally too
        slow for.

        This is a RANKING signal only -- e.g. "which of these omegas
        has the roughest reduced solution, so more training data
        should go there" -- not an absolute error BOUND. A true a
        posteriori bound needs this residual divided by a rigorous
        lower bound on A(omega)'s smallest singular value (the
        "coercivity constant" of the reduced-basis-method literature),
        which is NOT implemented here; see
        docs/phase4_error_bounds_greedy_roadmap.md Section 3 for why
        that is a genuinely hard, still-open problem near resonance
        (the literature is explicit that residual/coercivity-based
        estimators do not give tight bounds there), and why this
        package's greedy training uses the hierarchical indicator
        below instead of chasing a certified bound as the first cut.
        """
        F_r = self.galerkin.project_vector(F)
        q = self.solve(omega, F_r)
        Ax = self.affine.assemble_action(omega, q)
        residual = np.asarray(F, dtype=complex) - Ax
        return float(np.linalg.norm(residual))

    def error_estimate(self, omega, F):
        """Backward-compatible alias for residual_norm() -- kept under
        its original name since it predates residual_norm() and existing
        callers/docs refer to it, but now backed by the efficient
        O(Q*n_dof*n_modes) implementation instead of the original
        O(n_dof^2) one (same value, computed differently -- see
        residual_norm()'s docstring for the full explanation, including
        why this is an ESTIMATE, not a certified BOUND)."""
        return self.residual_norm(omega, F)

    def hierarchical_error_indicator(self, omega, F, comparison_rom):
        """||self.frequency_response([omega], F) -
        comparison_rom.frequency_response([omega], F)|| -- the
        DISAGREEMENT between this ROM and a second FrequencyROM (same
        M, K, C; a DIFFERENT basis) at this omega, used as a practical
        error indicator that needs no coercivity constant at all (see
        docs/phase4_error_bounds_greedy_roadmap.md Section 3).

        This sidesteps residual_norm()'s fundamental limitation
        (dividing by a near-zero singular value near resonance to turn
        a residual into a bound) by never trying to form a bound at
        all -- it only asks "do two different-fidelity ROMs agree,"
        which stays well-behaved even where the true error is largest.
        It is explicitly NOT a certified bound: it relies on an
        unverified "saturation assumption" (errors trend down as more
        information is added) from the reduced-basis-method literature,
        not a proof. Two calling patterns are both legitimate, and the
        method is symmetric in what "comparison_rom" means between them:

          - comparison_rom is a GENUINELY richer model (a basis
            meaningfully larger than self's, same training data plus
            more) -- the "textbook" reduced-basis-method usage, a
            direct estimate of self's own error.
          - comparison_rom is the PREVIOUS iteration's model in a
            greedy loop (a smaller, older basis) -- the usage
            greedy.greedy_train_frequency_basis() makes: "how much did
            the last real snapshot change the prediction here" as a
            proxy for "how much would the next one still change it,"
            the standard weak-greedy convergence-trend heuristic.

        Named "indicator", matching residual_norm()/error_estimate()'s
        existing honest-naming convention -- never call this a bound.
        """
        x_self = self.frequency_response([omega], F)[0]
        x_other = comparison_rom.frequency_response([omega], F)[0]
        return float(np.linalg.norm(x_self - x_other))


def build_pod_basis_from_frf_snapshots(training_omegas, M, K, F, C=None,
                                        n_modes=None, energy_threshold=None):
    """Full-order-solve A(omega) x = F at each of a handful of TRAINING
    frequencies and POD the resulting (complex) snapshot matrix -- an
    alternative to a modal (undamped-eigenmode) basis for
    FrequencyROM, better suited to capturing the actual load-dependent
    response shape near resonance (see
    docs/frequency_domain_rom_roadmap.md Section 2a: the applied POD-
    FRF literature reports this outperforming plain modal truncation
    once enough accuracy is demanded).

    This function does the expensive, full (n_dof, n_dof) solves --
    len(training_omegas) of them. The caller decides how many/which
    training frequencies to use; a grid spanning (and a bit beyond)
    the frequency BAND of interest, with extra density near any known
    resonances, is a reasonable starting point. A smarter, error-
    driven (greedy) alternative is Phase 4 of the roadmap, not
    implemented here.

    Parameters
    ----------
    training_omegas : array-like of float
        Frequencies (rad/s) to solve the FULL-ORDER system at.
    M, K : ndarray (n_dof, n_dof)
        Mass and stiffness.
    F : ndarray (n_dof,)
        Load vector used for every training solve (matching
        FrequencyROM.frequency_response()'s fixed-load assumption).
    C : ndarray (n_dof, n_dof), optional
        Damping matrix; omitted means undamped training solves (still
        useful as a basis for a SUBSEQUENTLY damped reduced query --
        the basis just needs to span the relevant response shapes,
        it doesn't need to be built from the exact same A(omega) the
        reduced model will later use).
    n_modes, energy_threshold :
        Passed straight through to PodBasis.fit().

    Returns
    -------
    A fitted PodBasis (complex-valued V, since FRF solutions are
    complex even when M, C, K are all real).
    """
    M = np.asarray(M, dtype=float)
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float) if C is not None else None
    n_dof = M.shape[0]
    snaps = np.zeros((n_dof, len(training_omegas)), dtype=complex)
    for i, omega in enumerate(training_omegas):
        A = -(omega**2) * M + K
        if C is not None:
            A = A + 1j * omega * C
        snaps[:, i] = np.linalg.solve(A, F)
    return PodBasis().fit(snaps, n_modes=n_modes, energy_threshold=energy_threshold)
