"""
postprocess.py -- Module 7: derived dynamic-response quantities.

These are quantities computed FROM an already-solved system (modal
results, a frequency sweep, a PSD) rather than part of solving it --
kept out of solver.py (which owns assembly/solution) and out of
main.py (which owns orchestration/demo execution), so main.py doesn't
turn into a dumping ground as dynamics capability grows. Add a new
derived quantity by adding one function here.

Deliberately general, not tied to a single-DOF closed form: an MDOF
FEM system doesn't have a simple Miles'-equation-style shortcut, so
variance_from_psd() does the numerical integration directly rather than
assuming a formula that would only be valid for a idealized SDOF.
"""
import numpy as np


def modal_participation_factors(mode_shapes_free, Mff, influence_vector):
    """L_i = phi_i^T M r, where r is the spatial 'influence vector'
    (e.g. a unit vector in the excited DOF direction for base
    excitation, or a load pattern for a distributed force). Returns an
    array of length n_modes."""
    return mode_shapes_free.T @ Mff @ influence_vector


def effective_modal_mass(mode_shapes_free, Mff, influence_vector):
    """Effective modal mass per retained mode: how much of the total
    excited mass each mode captures. If the modes are mass-normalized
    (phi_i^T M phi_i = 1, true for solver.solve_modal()'s output), this
    reduces to L_i^2. Summed over ALL modes of the system (not just the
    retained few), this equals the total mass moved by influence_vector
    -- the standard diagnostic for deciding how many modes are 'enough'
    for modal superposition."""
    L = modal_participation_factors(mode_shapes_free, Mff, influence_vector)
    denom = np.einsum('ik,ij,jk->k', mode_shapes_free, Mff, mode_shapes_free)
    return L**2 / denom


def variance_from_psd(freqs, S):
    """Response variance (= RMS^2) from a one-sided PSD via trapezoidal
    integration over frequency (Hz). General for any DOF/PSD shape."""
    return np.trapz(S, freqs)


def rms_from_psd(freqs, S):
    return np.sqrt(variance_from_psd(freqs, S))
