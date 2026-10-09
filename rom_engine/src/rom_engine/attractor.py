# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
attractor.py -- Wave 17 item 146 (fea_engine/docs/consolidated_future_
roadmap.md): attractor / Poincare tooling for FE (full-order) and
reduced (IntrusiveNonlinearROM) models, needed to reproduce Georgiou
(2005)'s Figs. 15, 18, 19, 20-22 and Eq. 61.

WHY THIS IS ITS OWN MODULE, NOT PART OF nnm.py. nnm.py already covers
one flavor of "periodic response" analysis -- harmonic-balance NNM
BACKBONE continuation (an autonomous, undamped, unforced periodic-orbit
family). This item is a genuinely different animal: FORCED, damped
steady-state response, characterized by TIME-DOMAIN integration plus
stroboscopic (Poincare) sampling at the forcing period, not harmonic
balance. Checked directly before writing anything here: nnm.py has no
stroboscopic/Poincare/attractor-adjacent helper at all (its own
docstring's own "Deliberate scope" section explicitly limits it to
undamped-autonomous backbones) -- so there is nothing to reuse or
duplicate from it; this module is additive, not a generalization of
existing code.

CONTENTS
--------
- `stroboscopic_sample()` -- samples a trajectory at the forcing period
  `2*pi/Omega`, WITH a genuine settling check (compares consecutive
  stroboscopic samples to a relative tolerance and reports the actual
  number of periods discarded, rather than assuming any fixed discard
  count) -- see its own docstring for why a fixed count is wrong here
  (the roadmap's own Wave 17 gap analysis: mode-1-scale problems settle
  in ~10 periods, mode-3-scale problems can need 50-80).
- `steady_state_amplitude()` -- 0.5*(max-min) of the FULL trajectory
  over the last settled forcing period, the standard steady-state
  forced-response amplitude definition used by an FRF.
- `SweepPoint`/`SweepResult` -- the common FS(full-order)-vs-RS(reduced)
  sweep-result data layout (see `frequency_sweep()`'s own docstring for
  the design reasoning).
- `frequency_sweep()`/`amplitude_sweep()` -- share one driver
  (`_sweep_driver()`), dispatching on whether `model` is an
  `IntrusiveNonlinearROM` (Wave 17 item 144) or a plain FE-model
  callable, per this item's own roadmap row.
- `midspan_transverse_probe()` -- transverse-displacement probe,
  generalized to any node count/dof layout (not hardcoded to one mesh).
- `master_slave_data()` -- Q2 vs Q1 / Q2 vs (Q1, Q1_dot) extraction from
  POD amplitude time series, for the paper's own Figs. 20-22 slow-
  invariant-manifold plots.
- `dominant_frequency()` -- FFT-based dominant frequency of a Q_m(t)
  series, for the paper's "slaved mode responds at 2x the master
  frequency" claim (Section 11) and its Section 7 amplitude-frequency
  table.

PARALLELISM. `n_jobs=` on both sweep drivers, via the standard library's
`concurrent.futures.ProcessPoolExecutor` -- checked directly (not
assumed) that neither `rom_engine` nor `fea_engine` uses
`multiprocessing`/`joblib`/`concurrent.futures` ANYWHERE else in either
package before choosing this, so this module is the first user of any
parallelism mechanism in either package. `ProcessPoolExecutor` (stdlib,
no new dependency) was chosen over `joblib` for exactly that reason --
both packages' `pyproject.toml` declare only `numpy`/`scipy` as runtime
dependencies, and this project's standing convention (see e.g.
`intrusive_nonlinear_rom.py`'s own "no cross-module rom_engine imports"
note) is to avoid adding a dependency when the standard library already
covers the need. HONEST LIMITATION, stated up front rather than
discovered by a confused caller: `ProcessPoolExecutor` pickles the
worker function and its arguments, so `n_jobs>1` requires `model`
(including, for an `IntrusiveNonlinearROM`, its own
`internal_force_fn`/`tangent_fn`/`load_fn` attributes),
`load_fn_builder`, and `probe_fn` to all be picklable -- plain
module-level functions or picklable callable objects, NOT lambdas or
closures over local variables (the standard Python multiprocessing
constraint, not specific to this module). `n_jobs=1` (the default) runs
serially with no such restriction.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .intrusive_nonlinear_rom import IntrusiveNonlinearROM


def _as_2d(y):
    y = np.asarray(y, dtype=float)
    return y[:, None] if y.ndim == 1 else y


# =====================================================================
# Stroboscopic (Poincare) sampling, with a genuine settling check.
# =====================================================================
def stroboscopic_sample(t, y, Omega, tol=1e-3, min_periods=3, norm_ord=np.inf):
    """Stroboscopic sampling of a trajectory `y(t)` at the forcing
    period `T = 2*pi/Omega`, with an AUTOMATIC settling check -- NOT a
    fixed arbitrary discard count. Rationale (the roadmap's own gap
    analysis, Wave 17 intro): a mode-1-scale problem (damping ratio
    ~0.08) settles in ~10 forcing periods, a mode-3-scale problem
    (damping ratio ~0.01) can need 50-80; a fixed discard window would
    be either wrong (too short for the slow case) or wasteful (too long
    for the fast one). This function instead samples the trajectory at
    EVERY forcing period available, then finds the SMALLEST discard
    count `p` (`p >= min_periods`) such that every subsequent
    stroboscopic sample is within `tol` (relative to the largest sample
    magnitude) of its predecessor -- i.e. genuinely checks for
    convergence onto a periodic (or slowly-varying quasi-periodic)
    attractor, rather than assuming it.

    Parameters
    ----------
    t : (n_time,) array_like
        Trajectory time samples (need not be uniform -- linear
        interpolation is used to evaluate `y` exactly at each
        stroboscopic instant `t0 + k*T`).
    y : (n_time,) or (n_time, n_dof) array_like
        Trajectory state history (displacement, reduced amplitude, or
        any other quantity sampled at `t`).
    Omega : float
        Forcing angular frequency (rad/s); the stroboscopic period is
        `T = 2*pi/Omega`.
    tol : float, default 1e-3
        Relative settling tolerance (fraction of the largest observed
        stroboscopic-sample magnitude).
    min_periods : int, default 3
        Minimum number of periods discarded regardless of how fast the
        raw samples appear to settle (guards against declaring
        "settled" from an accidental early near-repeat during a
        genuinely still-decaying transient).
    norm_ord : passed to `np.linalg.norm`'s `ord`, default `np.inf`
        (max-abs-component norm across dofs).

    Returns
    -------
    dict with keys:
      strobe_t, strobe_y : ALL stroboscopic samples (including the
          discarded transient ones), shape (n_strobe,), (n_strobe, n_dof).
      attractor_t, attractor_y : the samples AFTER the detected
          transient (`strobe_t/y[discard_periods:]`) -- for a converged
          period-1 attractor this is a small, tightly clustered set of
          points; for a genuinely non-converged/chaotic case it stays a
          scattered cloud (see this module's own Duffing test for both
          outcomes).
      discard_periods : int, the number of periods discarded.
      settled : bool, whether the settling criterion was actually met
          before running out of available periods (False means
          `discard_periods` is just "as many periods as were available
          minus `min_periods`" and the tail samples should NOT be
          trusted as a converged attractor).
      rel_diffs : (n_strobe-1,) ndarray, the relative sample-to-sample
          differences actually computed (for diagnostics/plotting).
    """
    t = np.asarray(t, dtype=float)
    y = _as_2d(y)
    T = 2.0 * np.pi / Omega
    t0, tf = t[0], t[-1]
    n_periods_avail = int(np.floor((tf - t0) / T))
    if n_periods_avail < min_periods:
        raise ValueError(
            f"trajectory spans only {n_periods_avail} forcing periods, "
            f"need at least min_periods={min_periods} to even attempt a "
            f"settling check -- integrate longer.")

    k = np.arange(n_periods_avail + 1)
    strobe_t = t0 + k * T
    strobe_y = np.column_stack([np.interp(strobe_t, t, y[:, j]) for j in range(y.shape[1])])

    diffs = np.linalg.norm(np.diff(strobe_y, axis=0), ord=norm_ord, axis=1)
    scale = max(float(np.max(np.linalg.norm(strobe_y, ord=norm_ord, axis=1))), 1e-300)
    rel_diffs = diffs / scale

    settled = False
    discard = max(n_periods_avail - min_periods, min_periods)
    for p in range(min_periods, len(rel_diffs) + 1):
        if np.all(rel_diffs[p:] < tol):
            discard = p
            settled = True
            break

    return {
        "strobe_t": strobe_t, "strobe_y": strobe_y,
        "attractor_t": strobe_t[discard:], "attractor_y": strobe_y[discard:],
        "discard_periods": discard, "settled": settled,
        "rel_diffs": rel_diffs,
    }


def steady_state_amplitude(t, y, Omega, tol=1e-3, min_periods=3, norm_ord=np.inf):
    """Steady-state forced-response amplitude(s): `0.5*(max-min)` of the
    FULL (not stroboscopically decimated) trajectory `y`, evaluated over
    the LAST fully-settled forcing period (per `stroboscopic_sample()`'s
    own settling check) -- the standard per-dof steady-state amplitude
    definition an FRF curve is built from. Returns an (n_dof,) ndarray
    (or a scalar if `y` was 1-D).

    Raises ValueError if the settling check never converged (a scattered
    cloud, or an insufficiently long run) -- amplitude is only a
    meaningful concept once the trajectory is genuinely periodic.
    """
    y2 = _as_2d(y)
    strobe = stroboscopic_sample(t, y2, Omega, tol=tol, min_periods=min_periods, norm_ord=norm_ord)
    if not strobe["settled"]:
        raise ValueError(
            "steady_state_amplitude: trajectory did not settle to the "
            "requested tolerance -- integrate longer or loosen tol.")
    t_last = strobe["strobe_t"][-2]
    T = 2.0 * np.pi / Omega
    mask = (t >= t_last - 1e-9 * T) & (t <= t_last + T + 1e-9 * T)
    if np.count_nonzero(mask) < 3:
        # fall back to interpolating a dense window if the raw samples
        # are too sparse across the last period
        t_dense = np.linspace(t_last, t_last + T, 200)
        y_window = np.column_stack([np.interp(t_dense, t, y2[:, j]) for j in range(y2.shape[1])])
    else:
        y_window = y2[mask]
    amp = 0.5 * (y_window.max(axis=0) - y_window.min(axis=0))
    return amp[0] if np.asarray(y).ndim == 1 else amp


# =====================================================================
# Midspan transverse-displacement probe.
# =====================================================================
def midspan_transverse_probe(state, n_nodes, dofs_per_node=3, transverse_dof=1, V=None):
    """Transverse-displacement value at (or nearest) the midpoint node
    of a rod/beam model, generalized to any node count and dof layout
    (not hardcoded to one specific mesh size, per this item's own
    roadmap row).

    Parameters
    ----------
    state : (..., n_state) array_like
        Either a FULL-ORDER state (`n_state == n_nodes*dofs_per_node`,
        `V=None`) or a REDUCED state (`n_state == n_modes`, with `V`
        supplied -- the full-order state is reconstructed as
        `state @ V.T` before probing). Any number of leading axes is
        supported (e.g. a whole `(n_time, n_state)` trajectory, or a
        `(n_sweep, n_time, n_state)` batch of sweep trajectories).
    n_nodes : int
        Total node count along the rod/beam (used to locate the
        midpoint node as `n_nodes // 2`, generalizing "positive midspan
        transverse component" to any mesh).
    dofs_per_node : int, default 3
        Dofs per node (matches `Beam2DReissner`'s own (u1, u2, theta)
        convention by default).
    transverse_dof : int, default 1
        Local dof index of the transverse displacement within each
        node's dof block (u2 is index 1 in the (u1, u2, theta) ordering).
    V : (n_dof, n_modes) array_like, optional
        Reduced basis, required (and used) only when `state` is a
        reduced-coordinate array.

    Returns
    -------
    ndarray, shape `state.shape[:-1]` -- the probed transverse
    displacement at every leading-axis entry of `state`.
    """
    state = np.asarray(state, dtype=float)
    mid_node = n_nodes // 2
    dof_index = mid_node * dofs_per_node + transverse_dof
    if V is not None:
        V = np.asarray(V, dtype=float)
        full = state @ V.T
        return full[..., dof_index]
    return state[..., dof_index]


# =====================================================================
# Master-slave (slow-invariant-manifold) data extraction.
# =====================================================================
def master_slave_data(t, Q, master=0, slave=1):
    """`Q2` vs `Q1` and `Q2` vs `(Q1, Q1_dot)` extraction from a POD
    amplitude time series `Q(t)`, for reproducing the paper's own
    master-slave slow-invariant-manifold plots (Figs. 20-22), where the
    slaved mode's amplitude is shown as approximately a function of the
    master mode's amplitude and velocity.

    Parameters
    ----------
    t : (n_time,) array_like
    Q : (n_time, n_modes) array_like
        POD amplitude time series (e.g. `MultiFieldPOD`'s own `Q_m(t)`,
        or an `IntrusiveNonlinearROM` reduced trajectory).
    master, slave : int
        Mode indices for the master/slave roles (default 0/1, matching
        the paper's own mode-1-master / mode-2-slave convention).

    Returns
    -------
    dict with keys:
      Q1, Q2 : (n_time,) ndarray -- the master/slave amplitude series.
      Q1_dot : (n_time,) ndarray -- `np.gradient(Q1, t)`, the master
          mode's own velocity (central differences, one-sided at the
          endpoints).
    """
    t = np.asarray(t, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Q1 = Q[:, master]
    Q2 = Q[:, slave]
    Q1_dot = np.gradient(Q1, t)
    return {"Q1": Q1, "Q2": Q2, "Q1_dot": Q1_dot}


# =====================================================================
# FFT-based dominant-frequency extraction.
# =====================================================================
def dominant_frequency(t, Q_m, n_peaks=1, rtol=1e-6):
    """FFT-based dominant (angular) frequency of a single mode's own
    amplitude time series `Q_m(t)`, needed for the paper's "the slaved
    mode responds at 2x the master frequency" claim (Section 11) and its
    Section 7 amplitude-frequency numbers (3067.21 / 25764.58 /
    62571.12 rad/s).

    Parameters
    ----------
    t : (n_time,) array_like
        Must be UNIFORMLY sampled (checked to `rtol`) -- `np.fft.rfft`
        assumes a fixed sample spacing.
    Q_m : (n_time,) array_like
    n_peaks : int, default 1
        Number of top FFT-magnitude peaks to return (by descending
        magnitude), excluding the DC (zero-frequency) bin.
    rtol : float, default 1e-6
        Relative tolerance for the uniform-sampling check.

    Returns
    -------
    float (if n_peaks == 1) or (n_peaks,) ndarray -- dominant angular
    frequency/frequencies in rad/s.
    """
    t = np.asarray(t, dtype=float)
    Q_m = np.asarray(Q_m, dtype=float)
    dt_arr = np.diff(t)
    dt0 = dt_arr[0]
    if not np.allclose(dt_arr, dt0, rtol=rtol, atol=rtol * max(abs(dt0), 1e-300)):
        raise ValueError("dominant_frequency needs a uniformly-sampled time series")

    N = len(Q_m)
    Qw = Q_m - np.mean(Q_m)
    F = np.fft.rfft(Qw)
    freqs_hz = np.fft.rfftfreq(N, d=dt0)
    mags = np.abs(F)
    mags[0] = -np.inf   # exclude DC from the peak search
    order = np.argsort(mags)[::-1]
    top = order[:n_peaks]
    omega_top = 2.0 * np.pi * freqs_hz[top]
    return float(omega_top[0]) if n_peaks == 1 else omega_top


# =====================================================================
# Common FS(full-order)-vs-RS(reduced) sweep data layout.
# =====================================================================
@dataclass
class SweepPoint:
    """One point of a frequency/amplitude sweep -- IDENTICAL fields
    regardless of whether it came from a full-order (FS) FE-model run or
    a reduced-order (RS) `IntrusiveNonlinearROM` run, which is the whole
    point of this common layout (direct FS-vs-RS overlay/comparison)."""
    param: float
    t: np.ndarray
    y: np.ndarray                 # (n_time, n_dof_or_modes)
    ydot: "np.ndarray | None"
    strobe_t: np.ndarray
    strobe_y: np.ndarray
    attractor_t: np.ndarray
    attractor_y: np.ndarray
    discard_periods: int
    settled: bool
    probe: "np.ndarray | float | None"
    is_reduced: bool


@dataclass
class SweepResult:
    """A whole frequency- or amplitude-sweep: `param_name`/`params`
    record what was swept, `points` holds one `SweepPoint` per value, in
    the SAME order as `params`."""
    param_name: str
    params: np.ndarray
    points: list = field(default_factory=list)

    def probes(self):
        """Stacked array of `point.probe` across the sweep, shape
        `(n_points,)` (scalar probes) or `(n_points, ...)` -- convenience
        for plotting an FRF-style curve directly."""
        return np.array([p.probe for p in self.points])


def _unpack_trajectory(result):
    """Normalizes a trajectory_fn / ROM-integrator return into
    (t, y, ydot)."""
    if len(result) == 3:
        t, y, ydot = result
    else:
        t, y = result
        ydot = None
    return np.asarray(t, dtype=float), _as_2d(y), (None if ydot is None else _as_2d(ydot))


def _run_point(model, param, strobe_omega, q0, qdot0, dt, n_steps,
                load_fn_builder, integrator, integrator_kwargs, probe_fn, strobe_kwargs):
    """Module-level (picklable) worker: runs ONE sweep point and returns
    a `SweepPoint`. Shared by `frequency_sweep()`/`amplitude_sweep()`
    via `_sweep_driver()` -- see that function's docstring for the
    FS-vs-RS dispatch logic this implements."""
    strobe_kwargs = strobe_kwargs or {}
    is_reduced = isinstance(model, IntrusiveNonlinearROM)

    if is_reduced:
        if q0 is None or qdot0 is None or dt is None or n_steps is None or load_fn_builder is None:
            raise ValueError(
                "sweeping an IntrusiveNonlinearROM needs q0, qdot0, dt, "
                "n_steps, and load_fn_builder (dt/n_steps may be callables "
                "of the swept parameter).")
        dt_p = dt(param) if callable(dt) else dt
        n_steps_p = n_steps(param) if callable(n_steps) else n_steps
        integrator_kwargs = integrator_kwargs or {}
        old_load_fn = model.load_fn
        model.load_fn = load_fn_builder(param)
        try:
            if integrator == "rk4":
                result = model.integrate_rk4(q0, qdot0, dt_p, n_steps_p)
            elif integrator == "newton_newmark":
                result = model.integrate_newton_newmark(q0, qdot0, dt_p, n_steps_p, **integrator_kwargs)
            elif integrator == "solve_ivp":
                sol = model.integrate_solve_ivp(q0, qdot0, (0.0, dt_p * n_steps_p), **integrator_kwargs)
                result = sol[:3]
            else:
                raise ValueError(f"unknown integrator {integrator!r}")
        finally:
            model.load_fn = old_load_fn
    else:
        result = model(param)

    t, y, ydot = _unpack_trajectory(result)
    strobe = stroboscopic_sample(t, y, strobe_omega, **strobe_kwargs)
    probe_val = probe_fn(strobe["attractor_y"]) if probe_fn is not None else None

    return SweepPoint(
        param=param, t=t, y=y, ydot=ydot,
        strobe_t=strobe["strobe_t"], strobe_y=strobe["strobe_y"],
        attractor_t=strobe["attractor_t"], attractor_y=strobe["attractor_y"],
        discard_periods=strobe["discard_periods"], settled=strobe["settled"],
        probe=probe_val, is_reduced=is_reduced,
    )


def _sweep_driver(model, param_values, param_name, strobe_omega_fn, *,
                   q0=None, qdot0=None, dt=None, n_steps=None, load_fn_builder=None,
                   integrator="rk4", integrator_kwargs=None,
                   probe_fn=None, strobe_kwargs=None, n_jobs=1):
    """Shared engine for `frequency_sweep()`/`amplitude_sweep()`.

    `model` dispatch (per this item's own roadmap row -- "accept EITHER
    a FE-model callable OR an IntrusiveNonlinearROM"):
      - `IntrusiveNonlinearROM` instance: each point is integrated with
        the ROM's OWN integrator (`integrator=` selects `"rk4"`
        (default -- the paper's own choice, per item 144),
        `"solve_ivp"`, or `"newton_newmark"`), with `load_fn_builder(param)`
        supplying that point's forcing (temporarily substituted for
        `model.load_fn`, restored afterward).
      - plain callable: called as `model(param) -> (t, y)` or
        `(t, y, ydot)` -- e.g. a closure wrapping
        `fea_engine.nonlinear_solver.solve_nonlinear_transient()` through
        the FAST vectorized element path (item 147's
        `beam2d_reissner_vectorized.py`), NOT the slow looped path --
        see this module's own docstring / the roadmap's item 147 row for
        why (a 20-40 run sweep is ~14-27 min on the vectorized path vs.
        1.5-7.5 hours looped).

    Both paths funnel into the SAME `SweepPoint`/`SweepResult` layout
    (`_run_point()`), so a full-order (FS) sweep and a reduced-order (RS)
    sweep can be overlaid directly -- the whole point of building a ROM.
    """
    def _args(param):
        return (model, param, strobe_omega_fn(param), q0, qdot0, dt, n_steps,
                load_fn_builder, integrator, integrator_kwargs, probe_fn, strobe_kwargs)

    if n_jobs and n_jobs > 1:
        # See module docstring: model/load_fn_builder/probe_fn must be
        # picklable for this path (standard multiprocessing constraint).
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_run_point, *_args(p)) for p in param_values]
            points = [f.result() for f in futures]
    else:
        points = [_run_point(*_args(p)) for p in param_values]

    return SweepResult(param_name=param_name, params=np.asarray(param_values), points=points)


def frequency_sweep(model, Omega_values, **kwargs):
    """Frequency sweep: runs one trajectory per `Omega` in `Omega_values`
    and stroboscopically samples EACH at that same `Omega` (the natural
    choice -- the forcing period is `2*pi/Omega`). See `_sweep_driver()`'s
    docstring for the full `model`/`q0`/`qdot0`/`dt`/`n_steps`/
    `load_fn_builder`/`integrator`/`probe_fn`/`strobe_kwargs`/`n_jobs`
    parameter set (all forwarded here as keyword arguments).

    Returns
    -------
    SweepResult (param_name="Omega")
    """
    return _sweep_driver(model, Omega_values, "Omega", lambda Omega: Omega, **kwargs)


def amplitude_sweep(model, amplitude_values, Omega, **kwargs):
    """Amplitude sweep: runs one trajectory per forcing amplitude in
    `amplitude_values`, ALL at the SAME forcing frequency `Omega`
    (needed to define the stroboscopic period, since amplitude alone
    doesn't). `load_fn_builder(amplitude)` must build that point's
    load_fn (e.g. `lambda a: lambda t: a * pattern * np.cos(Omega*t)`).
    See `_sweep_driver()`'s docstring for the rest of the parameter set.

    Returns
    -------
    SweepResult (param_name="amplitude")
    """
    return _sweep_driver(model, amplitude_values, "amplitude", lambda _a: Omega, **kwargs)
