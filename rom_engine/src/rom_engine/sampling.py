"""
sampling.py -- generalized Optimal Latin Hypercube Sampling (OLHS)
design-of-experiments core.

Every nonlinear-force-model training strategy in `nonlinear_rom.py`
(Section 4 of docs/nonlinear_surrogate_rom_roadmap.md) needs a way to
pick WHERE in some parameter space to run expensive full-order training
solves -- a per-mode force-scale factor for `AppliedLoadStrategy`, a
displacement-pattern amplitude for `EnforcedDisplacementStrategy`, or
(in the future) something else entirely. Rather than hand-roll that
sampling logic once per strategy, this module provides ONE
general-purpose OLHS sampler in the unit hypercube (`optimal_lhs`),
decoupled from any physical convention, plus ONE thin, named
convenience function (`modal_force_samples`) that rescales its output
into the specific per-mode force-scale quantity Eq. 16-17/46-47 of the
reference paper (He et al. 2023) needs. A future training strategy that
samples something else entirely reuses `optimal_lhs` directly rather
than duplicating the space-filling-design logic.

Explicit `rng` throughout (no hidden global state), matching the
convention already established for this package's other randomized
constructions (`rom_engine.loewner`/`rom_engine.screening`, see
docs/loewner_modal_identification_roadmap.md Section 2).
"""
__author__ = "Abhijeet"
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist


def _min_pairwise_distance(sample):
    """The maximin criterion's score: the SMALLEST pairwise distance
    between any two points in the design. A space-filling design wants
    this as LARGE as possible (no two training points crowded close
    together, wasting an expensive full-order solve on a point that
    tells the regression little it didn't already know from its
    neighbor)."""
    if sample.shape[0] < 2:
        return np.inf
    return float(pdist(sample).min())


def optimal_lhs(n_samples, n_dims, criterion="maximin", n_iter=200, rng=None):
    """General-purpose Optimal Latin Hypercube Sampling design in
    [0, 1]^n_dims.

    Implementation: draws `n_iter` independent (scrambled) Latin
    Hypercube candidate designs via `scipy.stats.qmc.LatinHypercube`
    and keeps the one that scores best under `criterion` -- the
    standard, simple "best-of-many-random-LHS" construction of an OLHS
    design (as opposed to a gradient-based/simulated-annealing swap
    optimizer, which would also work but adds real implementation
    complexity for a marginal further improvement once `n_iter` is
    reasonably large; this is the same practical trade-off SciPy's own
    `optimization="random-cd"` option makes for its own criterion).

    Parameters
    ----------
    n_samples : int
        Number of design points.
    n_dims : int
        Dimensionality of the design space.
    criterion : {"maximin"}, default "maximin"
        Space-filling criterion to optimize. Only "maximin" (maximize
        the smallest pairwise distance between any two design points,
        `_min_pairwise_distance` above) is implemented -- the most
        common OLHS criterion in the structural-ROM training-design
        literature this package's nonlinear-ROM modules target. Any
        other value raises `ValueError` rather than silently falling
        back to a different, unrequested criterion.
    n_iter : int, default 200
        Number of candidate LHS designs to draw and score. Larger
        values can only find an equal-or-better design (see
        `test_optimal_lhs_more_iterations_never_makes_the_design_worse`)
        at the cost of more candidate draws -- there is no
        overfitting/instability risk from raising it, only added cost.
    rng : numpy.random.Generator
        Required, explicit source of randomness (see module
        docstring) -- pass `np.random.default_rng(seed)` for a
        reproducible design.

    Returns
    -------
    ndarray, shape (n_samples, n_dims)
        Design points in [0, 1]^n_dims. Callers rescale to their own
        parameter ranges (see `modal_force_samples` below for one such
        rescaling).
    """
    if rng is None:
        raise ValueError("optimal_lhs requires an explicit rng (np.random.default_rng(seed))")
    if criterion != "maximin":
        raise ValueError(
            f"criterion={criterion!r} is not supported -- only 'maximin' is "
            f"implemented (see docstring)")
    if n_samples < 1 or n_dims < 1:
        raise ValueError(f"n_samples and n_dims must both be >= 1, got {n_samples}, {n_dims}")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")

    best_sample = None
    best_score = -np.inf
    for _ in range(n_iter):
        sampler = qmc.LatinHypercube(d=n_dims, rng=rng)
        candidate = sampler.random(n=n_samples)
        score = _min_pairwise_distance(candidate)
        if score > best_score:
            best_score = score
            best_sample = candidate
    return best_sample


def modal_force_samples(basis_freqs_hz, mode_shape_peaks, target_fracs,
                         reference_scale, n_samples, rng=None):
    """Eq. 16-17 / 46-47 of the reference paper, generalized: OLHS-
    samples a per-mode target peak displacement within a fraction range
    of a caller-supplied reference length scale, then converts each
    sampled displacement target into the per-mode FORCE-scale factor
    `f_hat_r` the paper's `AppliedLoadStrategy` (ICE / Shi & Mei
    training-load convention) needs:

        Q_max_r = a_r * reference_scale_r     (a_r in [frac_min_r, frac_max_r], OLHS-sampled)
        f_hat_r = (Q_max_r / phi_r_max) * lambda_r^2

    where `lambda_r = (2*pi*basis_freqs_hz[r])^2` is the retained
    mode's squared natural frequency (rad/s) and `phi_r_max =
    mode_shape_peaks[r]` is that mode's own peak (max-abs) shape
    component -- the paper's own normalization, so `Q_max_r` is exactly
    the requested modal PEAK displacement, not a dimensionless quantity.

    Deliberately generalized beyond the paper's own beam-thickness-
    specific convention: `reference_scale` is a caller-supplied
    physical length (e.g. a beam or plate thickness for the "0.8-1.2x
    thickness" range the paper itself uses), not hardcoded to
    "thickness" -- this function makes no assumption about what
    structure or geometry it's being used for.

    Parameters
    ----------
    basis_freqs_hz : array_like, shape (n_modes,)
        Natural frequencies [Hz] of the retained modes.
    mode_shape_peaks : array_like, shape (n_modes,)
        Each retained mode's own peak (max-abs) mode-shape component.
    target_fracs : (frac_min, frac_max) tuple, or array_like shape
        (n_modes, 2)
        Fraction of `reference_scale` each mode's target peak
        displacement should span. A single (frac_min, frac_max) pair
        is broadcast to every mode; a (n_modes, 2) array gives an
        independent range per mode.
    reference_scale : float or array_like, shape (n_modes,)
        Physical length scale `target_fracs` is relative to (e.g. a
        beam/plate thickness). A scalar is broadcast to every mode.
    n_samples : int
        Number of OLHS training points.
    rng : numpy.random.Generator
        Required, explicit (see module docstring / `optimal_lhs`).

    Returns
    -------
    ndarray, shape (n_samples, n_modes)
        The per-mode force-scale matrix `f_hat`, ready to build
        `F_t = (1/m) * M @ (phi_1 * f_hat[:, 0] + ... + phi_m * f_hat[:, m-1])`
        in `nonlinear_rom.AppliedLoadStrategy`.
    """
    basis_freqs_hz = np.asarray(basis_freqs_hz, dtype=float)
    mode_shape_peaks = np.asarray(mode_shape_peaks, dtype=float)
    n_modes = len(basis_freqs_hz)
    if len(mode_shape_peaks) != n_modes:
        raise ValueError(
            f"basis_freqs_hz and mode_shape_peaks must have the same length, "
            f"got {n_modes} and {len(mode_shape_peaks)}")

    fracs = np.asarray(target_fracs, dtype=float)
    if fracs.shape == (2,):
        fracs = np.tile(fracs, (n_modes, 1))
    elif fracs.shape != (n_modes, 2):
        raise ValueError(
            f"target_fracs must be a (2,) pair or an (n_modes, 2)={n_modes, 2} "
            f"array, got shape {fracs.shape}")
    frac_min, frac_max = fracs[:, 0], fracs[:, 1]
    if np.any(frac_max <= frac_min):
        raise ValueError("target_fracs: every (frac_min, frac_max) pair must have frac_max > frac_min")

    ref = np.broadcast_to(np.asarray(reference_scale, dtype=float), (n_modes,))

    unit_design = optimal_lhs(n_samples, n_modes, criterion="maximin", rng=rng)
    a = frac_min + unit_design * (frac_max - frac_min)   # (n_samples, n_modes), each column in its own [frac_min_r, frac_max_r]

    Q_max = a * ref                                       # (n_samples, n_modes)
    lam = (2 * np.pi * basis_freqs_hz) ** 2                # (n_modes,)
    f_hat = (Q_max / mode_shape_peaks) * lam                # broadcasts (n_modes,) across rows
    return f_hat
