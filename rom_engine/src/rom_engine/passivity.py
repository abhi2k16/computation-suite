"""
passivity.py -- passivity (positive-realness) diagnostics for a
collocated force-in/velocity-out structural port. See
docs/classical_mor_roadmap.md Section 12 for the full derivation and
an honest account of the scope adjustment this module makes (from
"positive-real balanced truncation," which needs a Riccati equation
that is singular for this package's D=0 structural realizations, to a
closed-form theorem + a diagnostic reusing already-existing machinery
-- a genuinely safer, and no less useful, path).

THE CLOSED-FORM PASSIVITY CERTIFICATE (no Riccati equation needed):
for `M q'' + C_damp q' + K q = B u`, `y = B^T q'` (a velocity output
collocated with the SAME pattern as the input force `B`), the total
mechanical energy `E = 0.5 q'^T M q' + 0.5 q^T K q` satisfies, using
`K = K^T` (so `q'^T K q = q^T K q'`, cancelling in the identity below):

    dE/dt = q'^T M q'' + q^T K q'
          = q'^T(-C_damp q' - K q + B u) + q^T K q'
          = -q'^T C_damp q' + q'^T B u
          = -q'^T C_damp q' + y^T u
          <= y^T u                      (since C_damp is PSD -- real,
                                          physical damping, already
                                          assumed throughout this
                                          package's state_space.py/
                                          balanced_truncation.py)

-- exactly the KYP/positive-real dissipation inequality. THIS SAME
DERIVATION APPLIES UNCHANGED to a second-order GALERKIN-projected
system (`M_r = Phi^T M Phi`, `C_r = Phi^T C_damp Phi`,
`K_r = Phi^T K Phi`, `B_r = Phi^T B`, `y_r = B_r^T eta'`) for ANY
full-column-rank basis `Phi`, since congruence transformation
preserves positive-(semi)definiteness unconditionally (`Phi^T M Phi >=
0` whenever `M >= 0`, for any `Phi` -- an elementary fact, no
orthogonality or special basis structure required). So: **any
second-order Galerkin projection this package builds --
`galerkin.GalerkinROM`'s basis, or a `frequency.FrequencyROM` on top
of it -- provably preserves passivity, for ANY basis, not just a
modal one.** This is a genuinely new, previously-unstated (though
latent) property of code this package already has, not a new
reduction algorithm -- nothing in this module builds a new ROM class.

THE DIAGNOSTIC this module actually adds: the FREQUENCY-DOMAIN form of
the same fact -- a positive-real transfer function satisfies
`Re[G(i*omega)] >= 0` for every real `omega` (the standard frequency-
domain characterization of the KYP/positive-real lemma) -- applicable
to ANY of this package's frequency-response-producing classes
(`FrequencyROM`, `KrylovROM`, `BalancedTruncationROM`,
`SingularPerturbationROM`, `FrequencyWeightedBalancedTruncationROM`),
by converting their EXISTING displacement transfer function to the
velocity (mobility) one via `H_vel(i*omega) = i*omega * H_disp(i*omega)`
-- no new state-space "velocity output" plumbing needed, since
velocity is exactly `d/dt` of displacement, i.e. multiplication by
`i*omega` in the frequency domain. This is what makes the diagnostic
useful beyond restating the closed-form theorem: it can be run against
`KrylovROM`/`BalancedTruncationROM`/
`FrequencyWeightedBalancedTruncationROM` too, NONE of which have any
passivity-preservation theorem behind them (they are first-order
STATE-SPACE reductions, not second-order Galerkin ones -- the
congruence argument above does not apply to them), to check directly
whether they can and do violate passivity in practice, even when built
from a stable, passive full-order model -- see
tests/test_passivity.py for what was actually found on this package's
real fixture, reported honestly either way.
"""
__author__ = "Abhijeet"
import numpy as np


def velocity_transfer_function(H_disp, omega_array):
    """H_vel(i*omega) = i*omega * H_disp(i*omega) -- the mobility/
    admittance transfer function for a collocated force-in/
    displacement-out SISO port, derived from H_disp WITHOUT any new
    state-space "velocity output" construction, since velocity is
    exactly the time-derivative of displacement (multiplication by
    i*omega in the frequency domain).

    Parameters
    ----------
    H_disp : array-like, complex, shape (n_freq,)
        A displacement transfer function already swept over
        omega_array -- from any of this package's frequency_response()
        outputs (FrequencyROM/KrylovROM/BalancedTruncationROM/
        SingularPerturbationROM/FrequencyWeightedBalancedTruncationROM,
        called with a SINGLE collocated force-in/displacement-out
        port), or fea_engine's own solve_harmonic() sweep at a single
        DOF.
    omega_array : array-like of float, shape (n_freq,)
        The SAME frequency grid H_disp was evaluated on.

    Returns
    -------
    H_vel : ndarray, complex, shape (n_freq,)
    """
    omega_array = np.asarray(omega_array, dtype=float)
    H_disp = np.asarray(H_disp, dtype=complex)
    return 1j * omega_array * H_disp


def passivity_margin(H_vel):
    """Re[H_vel(i*omega)] at each swept frequency -- the standard
    frequency-domain positive-realness/passivity margin for a one-port
    (SISO, collocated force-in/velocity-out) system: a transfer
    function is positive real iff this margin is >= 0 at every real
    omega (the frequency-domain form of the KYP/positive-real lemma --
    see module docstring for the time-domain dissipation-inequality
    version this is equivalent to). A margin that dips negative at
    some tested omega is a genuine passivity VIOLATION at that
    frequency, not just numerical noise, if the dip is well resolved
    (see is_passive()'s tol argument for how this module distinguishes
    a real violation from solver/floating-point noise near zero).

    Parameters
    ----------
    H_vel : array-like, complex, shape (n_freq,)
        A velocity transfer function, e.g. from
        velocity_transfer_function().

    Returns
    -------
    ndarray, real, shape (n_freq,)
    """
    return np.real(np.asarray(H_vel, dtype=complex))


def is_passive(H_vel, tol=0.0):
    """True iff passivity_margin(H_vel) >= -tol everywhere swept.

    Parameters
    ----------
    H_vel : array-like, complex, shape (n_freq,)
    tol : float
        Defaults to 0.0 (a strict check: the margin must never dip
        below exactly zero). A small positive tol (matching the
        numerical noise floor of the specific ROM/solve being checked)
        is appropriate when checking a REDUCED (not exact/full-order)
        model, since floating-point/solver noise can otherwise flag a
        spurious violation that isn't a real passivity failure --
        callers should choose tol deliberately (e.g. from a known
        solver tolerance), not just to make a test pass.

    Returns
    -------
    bool
    """
    return bool(np.all(passivity_margin(H_vel) >= -tol))
