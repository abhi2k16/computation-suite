# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
convergence.py -- small helpers for mesh-convergence studies.

    from fea_engine import convergence as cv

    study = cv.run_study(lambda n: tip_deflection(n), resolutions=[8, 16, 32], exact=1.0)
    study.orders          # pairwise observed orders, e.g. [1.93, 1.98]
    study.order           # least-squares order over all points
    print(study)          # table of h, value, error, order

``resolutions`` are the mesh parameters you vary (elements per side); ``h`` defaults to ``1 / resolution``.
Without ``exact`` the error of each level is measured against the finest one (the finest level then has no
error and is left out of the order fit) -- good enough to see a rate, but a known solution is better.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from dataclasses import dataclass, field

import numpy as np


def pairwise_orders(h, errors):
    """Observed order between consecutive levels: ``log(e_i / e_{i+1}) / log(h_i / h_{i+1})``."""
    h = np.asarray(h, dtype=float)
    e = np.abs(np.asarray(errors, dtype=float))
    if h.shape != e.shape or h.ndim != 1 or len(h) < 2:
        raise ValueError("pairwise_orders needs two equal-length 1-D sequences with at least 2 entries.")
    if np.any(h <= 0) or np.any(e <= 0):
        raise ValueError("h and errors must be positive (an exactly zero error has no order).")
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])


def observed_order(h, errors):
    """Least-squares slope of ``log(error)`` against ``log(h)``."""
    h = np.asarray(h, dtype=float)
    e = np.abs(np.asarray(errors, dtype=float))
    if h.shape != e.shape or h.ndim != 1 or len(h) < 2:
        raise ValueError("observed_order needs two equal-length 1-D sequences with at least 2 entries.")
    if np.any(h <= 0) or np.any(e <= 0):
        raise ValueError("h and errors must be positive (an exactly zero error has no order).")
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


def richardson(h1, u1, h2, u2, order):
    """Extrapolate to ``h -> 0`` from two levels assuming ``u(h) = u0 + c h^order``."""
    r = (h1 / h2) ** order
    if r == 1.0:
        raise ValueError("richardson needs two different mesh sizes.")
    return (r * u2 - u1) / (r - 1.0)


@dataclass
class ConvergenceStudy:
    resolutions: np.ndarray
    h: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    orders: np.ndarray = field(default=None)
    order: float = float("nan")
    reference: float = float("nan")

    def __str__(self):
        rows = ["  n        h          value          error      order"]
        for i, (n, h, v, e) in enumerate(zip(self.resolutions, self.h, self.values, self.errors)):
            o = "" if i == 0 or self.orders is None or i - 1 >= len(self.orders) else f"{self.orders[i - 1]:7.2f}"
            rows.append(f"{n:>3}  {h:10.4g}  {v:14.8g}  {e:12.4e}  {o}")
        rows.append(f"fitted order {self.order:.3f}")
        return "\n".join(rows)


def run_study(fn, resolutions, exact=None, h_of=None):
    """Evaluate ``fn(resolution)`` on each resolution and return a :class:`ConvergenceStudy`."""
    res = np.asarray(list(resolutions))
    if len(res) < 2:
        raise ValueError("run_study needs at least two resolutions.")
    if np.any(np.diff(res) == 0):
        raise ValueError("resolutions must be distinct.")
    h = np.array([h_of(r) if h_of else 1.0 / r for r in res], dtype=float)
    vals = np.array([float(fn(r)) for r in res])
    if exact is None:
        ref = vals[np.argmin(h)]
        err = np.abs(vals - ref)
        keep = err > 0
    else:
        ref = float(exact)
        err = np.abs(vals - ref)
        keep = err > 0
    if keep.sum() >= 2:
        order = observed_order(h[keep], err[keep])
        orders = pairwise_orders(h[keep], err[keep])
    else:
        order, orders = float("nan"), np.array([])
    return ConvergenceStudy(res, h, vals, err, orders, order, ref)
