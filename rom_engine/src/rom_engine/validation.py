# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
validation.py -- one standard way to check ANY reduced model against its full-order model.

Every method in this package reports accuracy in its own way. This module gives a single,
method-agnostic report: hand it two callables (the reduced predictor and the full-order predictor)
and a list of held-out test inputs (loads, parameters, excitations, ...) and it returns errors,
timing and speed-up, and a pass/fail against a tolerance you choose.

  * `validate_rom(rom_predict, fom_predict, test_inputs, tol=None)`  -> `ValidationReport`
  * `convergence_study(build_rom, sizes, fom_predict, test_inputs)`   -> error versus reduced size
  * `select_basis_size(study, tol)`                                   -> smallest size meeting `tol`

Predictors return arrays of any shape (a displacement vector, a time history, a frequency response,
complex values are fine); the error is the relative 2-norm of the difference by default.

Honest limits: the report is only as meaningful as the test inputs. Use inputs that were NOT used for
training, and include a few outside the training range to see extrapolation behaviour; the report
does not know what is inside or outside the training set.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from dataclasses import dataclass, field
import time
import numpy as np


def _err(a, b, norm):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"predictions must have the same shape, got {a.shape} and {b.shape}")
    d = np.linalg.norm((a - b).ravel())
    n = np.linalg.norm(b.ravel())
    if norm == "rel_l2":
        return d / n if n > 0 else d
    if norm == "abs_l2":
        return d
    if norm == "rel_max":
        m = np.max(np.abs(b))
        return np.max(np.abs(a - b)) / m if m > 0 else np.max(np.abs(a - b))
    raise ValueError("norm must be 'rel_l2', 'abs_l2' or 'rel_max'")


@dataclass
class ValidationReport:
    errors: np.ndarray
    norm: str
    tol: float = None
    t_fom: float = float("nan")
    t_rom: float = float("nan")
    worst_index: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def mean(self):
        return float(np.mean(self.errors))

    @property
    def median(self):
        return float(np.median(self.errors))

    @property
    def max(self):
        return float(np.max(self.errors))

    @property
    def speedup(self):
        return self.t_fom / self.t_rom if self.t_rom > 0 else float("inf")

    @property
    def passed(self):
        return None if self.tol is None else bool(self.max <= self.tol)

    def summary(self):
        s = (f"{len(self.errors)} test cases, {self.norm}: mean {self.mean:.3e}, median {self.median:.3e}, "
             f"max {self.max:.3e} (case {self.worst_index})")
        if not np.isnan(self.t_fom):
            s += f"; FOM {1e3 * self.t_fom:.3g} ms, ROM {1e3 * self.t_rom:.3g} ms, speed-up {self.speedup:.3g}x"
        if self.tol is not None:
            s += f"; tolerance {self.tol:.1e}: {'PASS' if self.passed else 'FAIL'}"
        return s

    def as_table(self):
        return {"case": list(range(len(self.errors))), "error": [float(e) for e in self.errors]}


def validate_rom(rom_predict, fom_predict, test_inputs, tol=None, norm="rel_l2", timing=True):
    """Compare a reduced predictor with the full-order predictor over held-out inputs.

    rom_predict, fom_predict : callable(x) -> array (same shape for the same x)
    test_inputs : iterable of inputs x
    tol : optional pass/fail threshold on the MAXIMUM error
    timing : also time both predictors (mean wall time per call)
    """
    xs = list(test_inputs)
    if not xs:
        raise ValueError("test_inputs is empty")
    errs, tf, tr = [], [], []
    for x in xs:
        t0 = time.perf_counter(); ref = fom_predict(x); tf.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); app = rom_predict(x); tr.append(time.perf_counter() - t0)
        errs.append(_err(app, ref, norm))
    errs = np.array(errs)
    rep = ValidationReport(errs, norm, tol, float(np.mean(tf)) if timing else float("nan"),
                           float(np.mean(tr)) if timing else float("nan"), int(np.argmax(errs)))
    if not np.all(np.isfinite(errs)):
        rep.extra["warning"] = "non-finite errors: a predictor returned NaN/inf"
    return rep


def convergence_study(build_rom, sizes, fom_predict, test_inputs, norm="rel_l2"):
    """Error versus reduced size. `build_rom(size)` must return a predictor callable(x).

    Returns dict(sizes, max, mean, reports). Use it to see whether more modes actually help."""
    xs = list(test_inputs)
    out = dict(sizes=list(sizes), max=[], mean=[], reports=[])
    for r in sizes:
        rep = validate_rom(build_rom(r), fom_predict, xs, norm=norm, timing=False)
        out["max"].append(rep.max); out["mean"].append(rep.mean); out["reports"].append(rep)
    return out


def select_basis_size(study, tol, use="max"):
    """Smallest size in a `convergence_study` whose error (`use` = 'max' or 'mean') is <= tol.

    Returns None if no tested size reaches the tolerance (never extrapolates)."""
    for r, e in zip(study["sizes"], study[use]):
        if e <= tol:
            return r
    return None
