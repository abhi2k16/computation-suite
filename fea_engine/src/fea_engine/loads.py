"""
loads.py -- Module 6: load definitions.

Decouples WHAT a load is (a spatial pattern plus a time/frequency/
statistical description) from HOW it gets applied to a mesh (solver.py's
add_nodal_force) and HOW the resulting response is solved for (solver.py's
solve_* methods). A static point load only needs a pattern; a transient,
harmonic, or random-vibration load needs that same pattern PLUS a
time/frequency-domain description of its magnitude -- which is exactly
the distinction encoded here.

To add a new load type (e.g. a moving load, a base-excitation load):
add one small class with a `.pattern` (a LoadPattern) and whatever
magnitude description it needs. solver.py's dynamics methods only ever
call `load.force_at(t, n_dof, npn)` (transient) or read `.pattern`/
`.F0`/`.freqs`/`.psd` directly (harmonic/PSD) -- nothing else in the
package needs to change.
"""
__author__ = "Abhijeet"
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class LoadPattern:
    """The spatial distribution of a load: `total` is split evenly
    across `node_ids` at local DOF `dof_index` -- the same simplified
    lumping convention used throughout this project's earlier scripts."""
    node_ids: np.ndarray
    dof_index: int

    def vector(self, n_dof: int, npn: int, total: float = 1.0) -> np.ndarray:
        v = np.zeros(n_dof)
        share = total / len(self.node_ids)
        for n in self.node_ids:
            v[npn * int(n) + self.dof_index] += share
        return v


@dataclass
class TimeHistoryLoad:
    """An arbitrary time-varying load: spatial pattern held fixed,
    magnitude given by any callable time_fn(t) -> float. Used by
    solver.solve_transient_implicit()/solve_transient_explicit()."""
    pattern: LoadPattern
    time_fn: Callable[[float], float]

    def force_at(self, t: float, n_dof: int, npn: int) -> np.ndarray:
        return self.pattern.vector(n_dof, npn, total=self.time_fn(t))


@dataclass
class HarmonicLoad:
    """F(t) = F0 * e^{i*Omega*t}, magnitude/pattern only -- the
    frequency itself is supplied per-call to solve_harmonic()/
    solve_frequency_sweep(), not stored here, since a sweep evaluates
    many frequencies against the same load."""
    pattern: LoadPattern
    F0: float

    def force_vector(self, n_dof: int, npn: int) -> np.ndarray:
        return self.pattern.vector(n_dof, npn, total=self.F0)


@dataclass
class PSDLoad:
    """A statistically-defined load: spatial pattern plus a one-sided
    input power spectral density S_input(f) sampled at `freqs` (Hz).
    Used by solver.solve_random_vibration()."""
    pattern: LoadPattern
    freqs: np.ndarray
    psd: np.ndarray

    def force_vector(self, n_dof: int, npn: int) -> np.ndarray:
        """Unit-magnitude pattern -- solve_random_vibration() builds the
        transfer function H(f) from this and multiplies by psd itself,
        so the load magnitude here is always 1.0."""
        return self.pattern.vector(n_dof, npn, total=1.0)
