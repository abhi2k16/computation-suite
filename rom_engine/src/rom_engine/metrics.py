"""
metrics.py -- generic, method-agnostic comparison metrics.

This module holds metrics for comparing two response sets, kept
deliberately separate from any one ROM method's module so they stay
reusable wherever two response sets need comparing, not tied to how
either one was obtained:

  - `modal_assurance_criterion` (MAC) -- used by ``loewner.py``'s
    validation/tests to check identified mode shapes against ground
    truth, but equally applicable to comparing two INTRUSIVE mode
    shapes (e.g. a coarse vs. a fine ``galerkin.py`` modal solve), or
    two experimentally measured shapes against each other.
  - `r_squared` -- the coefficient of determination, used by
    ``nonlinear_rom.py``'s validation (every accuracy table/figure in
    the He et al. 2023 reference paper this module traces to uses this
    one statistic) but equally applicable to any (y_true, y_pred) pair
    -- force samples, displacement samples, NNM amplitude samples,
    whatever a caller is checking a regression fit against.
"""
__author__ = "Abhijeet"
import numpy as np


def modal_assurance_criterion(phi1, phi2):
    """Modal Assurance Criterion (MAC) between two (possibly complex)
    mode-shape vectors.

    MAC(phi1, phi2) = |phi1^H phi2|^2 / ( (phi1^H phi1) (phi2^H phi2) )

    A standard scalar measure of how collinear two mode shapes are,
    independent of their (arbitrary) scale and, for complex vectors,
    their arbitrary overall phase: MAC = 1 means the two vectors are
    exactly parallel (the same mode shape, up to complex scale); MAC
    near 0 means they are essentially unrelated. It does NOT require
    the two vectors to be real or unit-normalized first -- the complex
    inner products and the division by each vector's own norm already
    make the result scale- and phase-invariant.

    Parameters
    ----------
    phi1, phi2 : array_like, shape (n,)
        Two mode-shape vectors, evaluated at the SAME set of degrees of
        freedom (comparing shapes sampled at different DOF sets is a
        common, silent misuse of MAC -- the caller is responsible for
        ensuring `phi1[i]` and `phi2[i]` refer to the same physical
        DOF).

    Returns
    -------
    float
        MAC value in [0, 1] (up to roundoff).
    """
    phi1 = np.asarray(phi1)
    phi2 = np.asarray(phi2)
    if phi1.shape != phi2.shape:
        raise ValueError(f"phi1 and phi2 must have the same shape, got {phi1.shape} and {phi2.shape}")
    num = np.abs(np.vdot(phi1, phi2)) ** 2
    den = np.vdot(phi1, phi1).real * np.vdot(phi2, phi2).real
    if den <= 0:
        raise ValueError("modal_assurance_criterion: a zero-norm mode shape was given")
    return float(num / den)


def r_squared(y_true, y_pred):
    """Coefficient of determination (Eq. 18 of He et al. 2023):

        R^2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)

    Generic and method-agnostic -- works on force samples, displacement
    samples, or NNM amplitude samples alike, whichever a caller is
    checking a regression/surrogate fit against. R^2 = 1 means a
    perfect fit; R^2 = 0 means the model does no better than predicting
    the mean of `y_true` for every sample; R^2 can go negative for a
    fit worse than that constant-mean baseline (a real, meaningful
    outcome this function does not clip away).

    Parameters
    ----------
    y_true : array_like
        Observed/ground-truth values (any shape; flattened internally,
        so a caller comparing e.g. a (n_modes, n_samples) prediction
        matrix against its own ground truth does not need to reshape
        first).
    y_pred : array_like
        Predicted/fitted values, same shape as `y_true`.

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        raise ValueError(
            "r_squared: y_true has zero variance (all values identical) -- "
            "R^2 is undefined (division by zero) for a constant target")
    return float(1.0 - ss_res / ss_tot)
