"""
loewner.py -- non-intrusive modal parameter identification via the
Loewner-pencil framework.

Every other module in this package (`galerkin.py`, `affine.py`,
`frequency.py`) is INTRUSIVE: each one is handed the actual system
matrices `K`/`M`/`C` (or an affine decomposition of them) and projects
them onto a reduced basis. This module is the opposite: it identifies
natural frequencies, damping ratios, and mode shapes directly from
SAMPLED complex frequency-response data `x(omega)` -- `LoewnerROM.fit()`
never receives `K`, `M`, or `C`, and never will, by construction (it
only accepts frequency/response arrays). That makes it usable on real
measured/experimental vibration data or on a proprietary/black-box
model's exported harmonic response, not just on a system this package
(or any FE code) can assemble directly.

Method (Eqs. 21-28 of the reference below): given complex response
samples `x_alpha` at one interpolation-frequency set `omega_alpha` and
`x_beta` at a disjoint second set `omega_beta`, build the Loewner pencil
`(A~, E~)` (Eqs. 21-22) whose generalized eigenvalues are exactly the
system's (squared, complex) poles -- no state-space realization is ever
formed. Solving `A~ Phi~ = E~ Phi~ Lambda~` (Eq. 25) and converting each
eigenvalue `lambda_i` to `omega_i = sqrt(lambda_i)` gives a natural
frequency `f_i = Re(omega_i) / 2*pi` (Eq. 26) and a loss-factor-like
modal damping ratio `eta_i = Im(omega_i) / Re(omega_i)` (Eq. 27). The
ROM eigenvectors `Phi~` can then be used to reconstruct full mode
shapes from multi-DOF response data sampled at the same `omega_beta`
points (Eq. 28).

Reference
---------
Liu, J. & Li, S. (2026). "A Reduced-Order Model for Modal Parameter
Identification of Fluid-Structure Interaction Systems Based on
Vibration Responses." Journal of Vibration Engineering & Technologies,
14:335. (Eqs. 21-28.)

Mayo, A.J. & Antoulas, A.C. (2007). "A framework for the solution of
the generalized realization problem." Linear Algebra and its
Applications, 425(2-3):634-662. (The underlying Loewner framework.)

This module is a direct, from-scratch port of the working prototype
`rom_modal_identification.py`'s `build_rom()`/`reconstruct_mode_shapes()`
functions, wrapped as a class (`LoewnerROM.fit()` /
`.reconstruct_mode_shapes()`) so a caller keeps the fitted ROM's state
(the eigenvectors, the omega_beta grid it was fit against) instead of
re-threading a bare tuple through every later call. See
docs/loewner_modal_identification_roadmap.md for the full design
background and phased plan.
"""
__author__ = "Abhijeet"
import numpy as np
from scipy.linalg import eig


class LoewnerROM:
    """A non-intrusive modal model identified from complex frequency-
    response samples via the Loewner pencil (see module docstring).

    Attributes (all set by fit(), read-only in spirit)
    ---------------------------------------------------
    f : ndarray, shape (n,)
        Identified natural frequencies [Hz] (Eq. 26). Complex-valued
        input, real-valued output -- these are always real numbers by
        construction (`.real` is taken explicitly in fit()).
    eta : ndarray, shape (n,)
        Identified modal damping ratios (loss-factor-like, Eq. 27).
        Real by construction, same reasoning as f.
    Phi_tilde : ndarray, shape (n, n), complex
        The raw ROM eigenvector matrix from the generalized
        eigenproblem (Eq. 25) -- needed by reconstruct_mode_shapes()
        (Eq. 28), not generally meaningful to a caller on its own.
    omega_alpha, omega_beta : ndarray, shape (n,)
        The two interpolation-frequency sets [rad/s] this ROM was
        fit() with. Stored so reconstruct_mode_shapes() can check a
        caller's response data was sampled at the SAME omega_beta
        points the method requires (Eq. 28's precondition).
    """

    def __init__(self, f, eta, Phi_tilde, omega_alpha, omega_beta):
        self.f = f
        self.eta = eta
        self.Phi_tilde = Phi_tilde
        self.omega_alpha = omega_alpha
        self.omega_beta = omega_beta

    @classmethod
    def fit(cls, omega_alpha, omega_beta, x_alpha, x_beta):
        """Build a LoewnerROM from single-DOF complex response samples
        (Eqs. 21-27). This is the ONLY place in this class that touches
        raw data, and the data it touches is exactly (and only) four
        1-D arrays -- no system matrices, no fea_engine objects, no
        knowledge of what physical structure or measurement produced
        these numbers.

        Parameters
        ----------
        omega_alpha, omega_beta : array_like, shape (n,)
            Two DISJOINT sets of interpolation frequencies [rad/s]
            forming the Loewner pencil's two Krylov interpolation
            point sets S_alpha, S_beta. Must be the same length (the
            resulting pencil is square, n x n).
        x_alpha, x_beta : array_like, shape (n,), complex
            The measured/simulated complex displacement response at a
            single reference DOF, sampled at omega_alpha and
            omega_beta respectively.

        Returns
        -------
        LoewnerROM
            Fitted model, with f/eta/Phi_tilde/omega_alpha/omega_beta
            set as described in the class docstring.
        """
        omega_alpha = np.asarray(omega_alpha, dtype=float)
        omega_beta = np.asarray(omega_beta, dtype=float)
        x_alpha = np.asarray(x_alpha, dtype=complex)
        x_beta = np.asarray(x_beta, dtype=complex)

        n = len(omega_alpha)
        if not (len(omega_beta) == n and len(x_alpha) == n and len(x_beta) == n):
            raise ValueError(
                "omega_alpha, omega_beta, x_alpha, x_beta must all have the "
                f"same length; got {len(omega_alpha)}, {len(omega_beta)}, "
                f"{len(x_alpha)}, {len(x_beta)}")
        if n < 1:
            raise ValueError("need at least 1 interpolation point per set")

        wa2 = omega_alpha ** 2
        wb2 = omega_beta ** 2

        A_t = np.zeros((n, n), dtype=complex)
        E_t = np.zeros((n, n), dtype=complex)
        for a in range(n):
            for b in range(n):
                denom = wa2[a] - wb2[b]          # Eqs. (21)-(22)
                if denom == 0:
                    raise ValueError(
                        "omega_alpha and omega_beta must be DISJOINT frequency "
                        f"sets -- found a shared/coincident value at index "
                        f"({a}, {b}) (omega_alpha[{a}] == omega_beta[{b}]), "
                        "which makes the Loewner pencil entry undefined "
                        "(0/0). Re-sample with distinct interpolation points.")
                A_t[a, b] = (wa2[a] * x_alpha[a] - wb2[b] * x_beta[b]) / denom
                E_t[a, b] = (x_alpha[a] - x_beta[b]) / denom

        lam_tilde, Phi_tilde = eig(A_t, E_t)      # Eq. (25): lambda ~ omega^2
        omega_tilde = np.sqrt(lam_tilde)           # take sqrt -> omega (Eq. 25 text)
        omega_tilde = np.where(omega_tilde.real < 0, -omega_tilde, omega_tilde)

        f = omega_tilde.real / (2 * np.pi)          # Eq. (26)
        eta = omega_tilde.imag / omega_tilde.real   # Eq. (27)
        return cls(f, eta, Phi_tilde, omega_alpha, omega_beta)

    def reconstruct_mode_shapes(self, X_beta_multi, omega_beta=None):
        """Eq. (28): Phi = X_beta_multi @ Phi_tilde -- reconstruct
        multi-DOF mode shapes from response data sampled at MULTIPLE
        DOFs, but at the SAME omega_beta interpolation frequencies used
        to fit() this ROM (Eq. 28's actual precondition -- reusing
        Phi_tilde against response data sampled at different
        frequencies is a silent misuse this method guards against
        below rather than assumes the caller got right).

        Parameters
        ----------
        X_beta_multi : array_like, shape (n_meas_dof, n)
            Complex response samples at the measurement/reconstruction
            DOFs, at the same n omega_beta frequencies this ROM was
            fit() with (n = len(self.omega_beta)).
        omega_beta : array_like, shape (n,), optional
            If given, checked against the omega_beta this ROM was
            fit() with (element-wise, via np.allclose) -- an explicit,
            loud way to confirm the caller sampled X_beta_multi at the
            right frequencies, instead of a shape match alone (which
            would not catch e.g. an accidentally reordered or
            differently-spaced frequency array of the same length).

        Returns
        -------
        ndarray, shape (n_meas_dof, n), complex
            Reconstructed mode-shape matrix: column j is (proportional
            to) the j-th identified mode's shape at the n_meas_dof
            measurement DOFs.
        """
        X_beta_multi = np.asarray(X_beta_multi, dtype=complex)
        n = self.Phi_tilde.shape[0]
        if X_beta_multi.ndim != 2 or X_beta_multi.shape[1] != n:
            raise ValueError(
                f"X_beta_multi must have shape (n_meas_dof, {n}) -- one "
                f"column per omega_beta interpolation point this ROM was "
                f"fit() with; got shape {X_beta_multi.shape}")
        if omega_beta is not None:
            omega_beta = np.asarray(omega_beta, dtype=float)
            if omega_beta.shape != self.omega_beta.shape or not np.allclose(omega_beta, self.omega_beta):
                raise ValueError(
                    "omega_beta does not match the interpolation frequencies "
                    "this ROM was fit() with -- Eq. (28) requires reconstructing "
                    "from response data sampled at the SAME omega_beta points, "
                    "not merely the same number of points.")
        return X_beta_multi @ self.Phi_tilde
