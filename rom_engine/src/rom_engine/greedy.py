"""
greedy.py -- weak-greedy reduced-basis training for
frequency.FrequencyROM: instead of training a POD-on-FRF-snapshots
basis from a fixed (e.g. uniform) grid of frequencies, ADAPTIVELY pick
which frequencies to full-order-solve at, spending that (expensive)
budget where the current basis is actually weak.

The algorithm (see docs/phase4_error_bounds_greedy_roadmap.md Section
4 for the full research background): start from a small seed set of
frequencies, then repeatedly (1) evaluate a CHEAP error indicator
across every untried candidate frequency, (2) full-order-solve ONLY at
the worst-indicated one and add it to the training set, (3) rebuild
the basis, (4) repeat until the worst indicated error is small enough
or a basis-size budget is reached. This concentrates expensive
full-order solves near resonances (where a frequency response is
hardest to represent) instead of wasting them on a uniform grid's flat
regions.

The error indicator: this iteration's FrequencyROM vs. the PREVIOUS
iteration's, both evaluated (cheaply -- reduced queries only, no
full-order solve) at every untried candidate --
FrequencyROM.hierarchical_error_indicator(). This is the standard
weak-greedy hierarchical construction (Hain et al.): the previous
iteration's model is a genuinely SMALLER-INFORMATION model (it was
built before the most recently selected snapshot was added), so
"how much did the last real snapshot change the prediction at this
candidate" is a reasonable proxy for "how much would the NEXT snapshot
change it" under the usual (unverified, but standard) saturation
assumption that error decreases roughly monotonically as snapshots are
added. This is deliberately used INSTEAD OF a certified coercivity-
bound estimator, because the literature documents the coercivity-bound
approach as weak exactly near resonance (Section 3 of the roadmap doc)
-- precisely where this loop most needs a reliable signal. The very
first iteration has no "previous" model yet to compare against, so it
falls back to FrequencyROM.residual_norm() (Phase 4a's efficient exact
residual) instead -- a real ranking signal even without a comparison
point.

This module is deliberately NOT specific to any one FrequencyROM
construction detail beyond from_MCK()'s signature -- the greedy LOOP
itself (evaluate indicator, pick worst, solve, extend, repeat) is a
generic pattern that a future affine.py material-parameter greedy
trainer could reuse with a different error indicator and full-order
solver, not implemented here but noted as a natural generalization.
"""
import numpy as np

from .frequency import FrequencyROM, build_pod_basis_from_frf_snapshots


def greedy_train_frequency_basis(training_omegas, M, K, F, C=None, rayleigh=None,
                                  n_seed=3, tol=1e-4, max_modes=30):
    """Weak-greedy construction of a POD-on-FRF-snapshots basis for
    FrequencyROM.

    Parameters
    ----------
    training_omegas : array-like of float
        The CANDIDATE pool of frequencies (rad/s) the greedy loop is
        allowed to choose from -- typically a fine grid spanning the
        band of interest. Full-order solves only ever happen at
        frequencies actually SELECTED from this pool, never at every
        candidate (that's the whole point).
    M, K : ndarray (n_dof, n_dof)
        Mass and stiffness.
    F : ndarray (n_dof,)
        Load vector (matching FrequencyROM.frequency_response()'s
        fixed-load assumption).
    C : ndarray (n_dof, n_dof), optional
        Explicit damping matrix (general 3-term case).
    rayleigh : (alpha, beta) tuple, optional
        Proportional damping (2-term collapse). At most one of C /
        rayleigh should be given, matching FrequencyROM.from_MCK().
    n_seed : int
        Number of initial (evenly spaced through training_omegas)
        frequencies to seed the training set with, before the
        adaptive loop starts picking. The first ADAPTIVE selection
        (after the seed) always uses the residual-norm fallback (no
        "previous" model exists yet); every selection after that uses
        the hierarchical indicator.
    tol : float
        Stop once the worst indicator (relative to the current model's
        own response norm at that candidate) across all untried
        candidates falls below this.
    max_modes : int
        Hard cap on basis size regardless of tol -- a runaway-training
        safety limit.

    Returns
    -------
    (basis, history) : (PodBasis, list of (omega, indicator, method))
        basis is the final fitted PodBasis (built from every selected
        frequency, at full available rank up to max_modes). history
        records every SELECTED frequency in order, together with the
        indicator value that justified selecting it and which
        indicator ("hierarchical" or "residual_fallback") was used --
        kept so a caller can audit/plot which frequencies the loop
        actually chose (e.g. confirming they cluster near a resonance),
        the direct claim this module's test suite checks.
    """
    if C is not None and rayleigh is not None:
        raise ValueError("give at most one of C, rayleigh")

    M = np.asarray(M, dtype=float)
    K = np.asarray(K, dtype=float)
    F = np.asarray(F)
    training_omegas = np.asarray(training_omegas, dtype=float)
    if len(training_omegas) < n_seed + 1:
        raise ValueError("training_omegas needs more points than n_seed, so there's "
                          "something left for the greedy loop to choose from")

    def _rom_from(omegas):
        basis = build_pod_basis_from_frf_snapshots(
            omegas, M, K, F, C=C, n_modes=min(len(omegas), max_modes))
        rom = FrequencyROM.from_MCK(M, K, basis.V, C=C, rayleigh=rayleigh)
        return basis, rom

    seed_idx = np.linspace(0, len(training_omegas) - 1, n_seed).astype(int)
    selected = list(training_omegas[seed_idx])
    remaining = [om for i, om in enumerate(training_omegas) if i not in set(seed_idx)]

    basis, rom_curr = _rom_from(selected)
    rom_prev = None   # no earlier model yet -- first iteration falls back
    history = []

    while remaining and len(selected) < max_modes:
        if rom_prev is None:
            method = "residual_fallback"
            F_norm = max(np.linalg.norm(F), 1e-30)
            rel_indicators = np.array([
                rom_curr.residual_norm(om, F) / F_norm for om in remaining
            ])
        else:
            method = "hierarchical"
            indicators = np.array([
                rom_curr.hierarchical_error_indicator(om, F, comparison_rom=rom_prev)
                for om in remaining
            ])
            scales = np.array([
                np.linalg.norm(rom_curr.frequency_response([om], F)[0])
                for om in remaining
            ])
            rel_indicators = indicators / np.maximum(scales, 1e-30)

        worst_local = int(np.argmax(rel_indicators))
        worst_omega = remaining[worst_local]
        worst_val = float(rel_indicators[worst_local])
        history.append((worst_omega, worst_val, method))

        if worst_val < tol:
            break

        selected.append(worst_omega)
        remaining.pop(worst_local)
        rom_prev = rom_curr
        basis, rom_curr = _rom_from(selected)

    return basis, history
