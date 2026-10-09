"""
random_vibration.py -- PSD (power spectral density) random-vibration
response, built as a thin layer on top of frequency.FrequencyROM's
existing sweep. Reuses, rather than re-derives, a formula
`fea_engine.solver.FESystem.solve_random_vibration()` already
implements and validates at full order: for a linear system driven by a
fixed spatial load pattern `F0` with PSD input spectrum `S_in(f)`, the
steady-state output PSD at a sensor DOF is

    S_out(f) = |H(f)|^2 * S_in(f)

where `H(f)` is the system's own transfer function from `F0` to that
DOF -- a direct, standard consequence of linear-systems PSD propagation
(no new derivation needed here; see
docs/frequency_domain_rom_roadmap.md Phase 6 for the design note this
module implements). The response RMS is the square root of the
trapezoidal-integrated output PSD, `sigma_out = sqrt(trapz(S_out, f))`.

This module is deliberately a SEPARATE, small module rather than a
method added to `FrequencyROM` itself, matching this package's
one-concept-per-module convention (the same reasoning `greedy.py`/
`scm.py` already follow on top of `frequency.py`, rather than growing
`FrequencyROM` into a do-everything class): `psd_response()` composes
`FrequencyROM.frequency_response()` (already handles the affine+Galerkin
machinery) with the two extra lines PSD propagation needs, and nothing
about it is specific to how the underlying ROM was built.
"""
__author__ = "Abhijeet"
import numpy as np


def psd_response(rom_freq, freqs_hz, psd_input, F0_pattern, output_dofs):
    """Random-vibration (PSD) response of a FrequencyROM, at one or
    several output DOFs.

    Parameters
    ----------
    rom_freq : frequency.FrequencyROM
        An already-built ROM (``FrequencyROM.from_MCK(...)``).
    freqs_hz : array-like of float
        The frequency grid (Hz) the input PSD is defined on -- also
        the grid `S_out` is returned on and integrated over.
    psd_input : array-like of float, same length as freqs_hz
        The input PSD `S_in(f)`, e.g. from a measured base-excitation
        or turbulence spectrum.
    F0_pattern : ndarray (n_dof,)
        The fixed spatial load pattern the PSD input is applied
        through (same convention as
        ``FrequencyROM.frequency_response()``'s own `F` argument --
        a frequency-INDEPENDENT shape, projected once).
    output_dofs : int, or array-like of int
        A SINGLE full-order DOF index (matching
        `fea_engine.solver.FESystem.solve_random_vibration()`'s own
        signature exactly, for direct 1:1 comparability -- see that
        method's docstring) returns a scalar `sigma_out`; an array-like
        of several DOF indices returns one `sigma_out` per DOF instead
        (a generalization `fea_engine`'s own single-output method
        doesn't offer).

    Returns
    -------
    S_out : ndarray
        Output PSD, shape (n_freq,) for a single output_dofs int, or
        (n_freq, n_out) for an array-like of several.
    sigma_out : float or ndarray
        Response RMS -- a float for a single output_dofs int, or an
        (n_out,) array for several.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    omega_array = 2 * np.pi * freqs_hz
    psd_input = np.asarray(psd_input, dtype=float)

    scalar_output = np.ndim(output_dofs) == 0
    dofs = [output_dofs] if scalar_output else output_dofs

    H = rom_freq.frequency_response(omega_array, F0_pattern, output_dofs=dofs)   # (n_freq, n_out)
    S_out = np.abs(H) ** 2 * psd_input[:, None]                                   # (n_freq, n_out)
    sigma_out = np.sqrt(np.trapz(S_out, freqs_hz, axis=0))                        # (n_out,)

    if scalar_output:
        return S_out[:, 0], float(sigma_out[0])
    return S_out, sigma_out


def band_limited_gaussian_time_history(n_samples, dt, f_max, rms, rng=None):
    """Wave 12 item 116 (docs/consolidated_future_roadmap.md, sourced
    from `ICE-ROM/GAP_ANALYSIS.md` gap #5): a band-limited (0 to
    `f_max` Hz), Gaussian-random, time-domain forcing signal -- the
    driving input Hollkamp & Gordon (2008)'s own dynamic-response
    reproduction needs (Sec. 4, a band-limited random pressure driving
    both the full FOM and the trained ROM identically so their transient
    responses are directly comparable), and a standing need for ANY
    random-vibration time-domain simulation this package's PSD-domain
    `psd_response()` above doesn't itself produce (that function stays
    in the frequency domain on purpose; this one is the time-domain
    counterpart when a caller needs an actual signal to integrate a
    nonlinear transient driver with, not just a PSD envelope).

    Promoted, unchanged in method, from the ad hoc ~10-line prototype
    this project's own `ICE-ROM/validation/dynamic_comparison.py::
    band_limited_gaussian` used to validate a real flat-beam ICE-ROM
    reproduction -- spectral synthesis via one `rfft`, zeroing every
    bin above `f_max`, one `irfft` back to the time domain, then
    rescaled to the requested RMS. This is the standard, simplest way
    to generate a band-limited stationary Gaussian process on a
    UNIFORM time grid of a KNOWN, fixed length -- deliberately not a
    general streaming/online generator, which would need a different
    (e.g. filter-based) approach.

    Parameters
    ----------
    n_samples : int
        Number of time samples to generate (the output is exactly this
        length, unlike an FFT-based filter that might need padding).
    dt : float
        Time step (s) -- together with `n_samples` fixes the frequency
        resolution (`1/(n_samples*dt)`) and the Nyquist limit
        (`1/(2*dt)`); `f_max` must not exceed the Nyquist limit or this
        raises `ValueError` rather than silently doing nothing (a
        `f_max` above Nyquist would leave every bin already below the
        signal's own frequency ceiling, i.e. the band-limiting step
        would have no effect at all -- an easy, silent mistake to make
        when choosing `dt` and `f_max` independently).
    f_max : float
        Upper cutoff frequency (Hz) -- every FFT bin strictly above
        this is zeroed (an ideal brick-wall low-pass, not a physically
        realizable analog filter's rolloff -- appropriate here since
        this is a SIMULATED input signal, not a measured one being
        filtered after the fact).
    rms : float
        Target root-mean-square value of the returned signal -- the
        raw filtered white-noise realization is rescaled (a single
        multiply, using its OWN sample RMS -- not its std, which would
        silently differ if the DC bin were left in the signal) so the
        returned signal's sample RMS matches this exactly, not just in
        expectation over infinitely many realizations.
    rng : numpy.random.Generator, optional
        Explicit generator (this package's standing convention, see
        `sampling.py`'s own module docstring on why every randomized
        construction here takes an explicit `rng`, never hidden global
        state) -- a fresh, unseeded `np.random.default_rng()` is used
        if omitted.

    Returns
    -------
    x : ndarray, shape (n_samples,)
        The band-limited, RMS-normalized time-domain signal.
    """
    if n_samples < 2:
        raise ValueError(f"band_limited_gaussian_time_history: n_samples must be >= 2, got {n_samples}")
    if dt <= 0:
        raise ValueError(f"band_limited_gaussian_time_history: dt must be > 0, got {dt}")
    nyquist = 1.0 / (2.0 * dt)
    if f_max <= 0 or f_max > nyquist:
        raise ValueError(
            f"band_limited_gaussian_time_history: f_max={f_max} must be in (0, nyquist={nyquist}] "
            f"-- a value above the Nyquist limit would leave every frequency bin unfiltered, "
            f"silently defeating the whole point of this function.")
    if rng is None:
        rng = np.random.default_rng()

    x = rng.standard_normal(n_samples)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n_samples, dt)
    # Zero the DC bin too, not just everything above f_max: a physical
    # random forcing signal (the paper's own band-limited pressure load)
    # has no static offset, and zeroing DC also makes the RMS-vs-std
    # distinction moot below (RMS == std exactly once the mean is zero).
    X[(freqs > f_max) | (freqs == 0.0)] = 0.0
    x_filt = np.fft.irfft(X, n=n_samples)
    rms_raw = np.sqrt(np.mean(x_filt ** 2))
    if rms_raw < 1e-300:
        raise RuntimeError(
            "band_limited_gaussian_time_history: filtered signal has ~zero variance -- "
            "f_max is likely too small relative to the frequency resolution 1/(n_samples*dt) "
            "for any bin to survive the band-limiting step.")
    x_filt *= rms / rms_raw
    return x_filt
