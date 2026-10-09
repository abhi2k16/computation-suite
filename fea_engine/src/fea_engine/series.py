"""
series.py -- FieldSeries: one container for multi-step results.

The solvers return histories in different shapes: ``(load_factors, U_hist)`` from the
nonlinear drivers, ``(t, U_hist)`` from the transient ones, ``(freq_hz, mode_shapes)`` from
the modal solver (columns, not rows). A `FieldSeries` gives all of them the same access
style, built on `FEField`, without changing what those solvers return::

    lf, hist = solve_nonlinear_static(system, mat, n_steps=20)
    path = system.series(hist, steps=lf, step_name="load factor")

    path[-1]                                  # last step as an FEField
    path.at(0.5)                              # step nearest to load factor 0.5
    path.history("uy", "tip", reduce="mean")  # load-displacement curve, shape (n_steps,)
    path.component("uy", nodes="tip")         # (n_steps, n_tip_nodes)
    path.peak("uy")                           # per-node max |uy| over all steps
    path.plot_history("uy", "tip", reduce="mean")

    t, hist = system.solve_transient_implicit(load, T_total=1.0, dt=1e-3)
    resp = system.series(hist, steps=t, step_name="time")

    modes = system.modal_series(n_modes=5)    # steps = frequencies in Hz
    modes.at(120.0).plot("uy")                # mode closest to 120 Hz

The series stores ONE ``(n_steps, n_dof)`` array; each step is a view of a row, so slicing
and iterating copy nothing.
"""
from __future__ import annotations
__author__ = "Abhijeet"

import numpy as np

from .fields import FEField

__all__ = ["FieldSeries"]


class FieldSeries:
    """An ordered sequence of `FEField` snapshots with a step coordinate.

    Parameters
    ----------
    data : array (n_steps, n_dof)
        One row per step.
    template : FEField
        Any field of the same system; supplies the DOF layout, mesh and units.
    steps : array (n_steps,), optional
        Step values (load factor, time, frequency, ...). Default ``0..n_steps-1``.
    step_name : str
        What the steps are, e.g. ``"load factor"``, ``"time"``, ``"frequency"``.
    step_unit : str, optional
        Unit label of the steps (``"s"``, ``"Hz"``), used in plot labels.
    label : str, optional
        Name of the quantity (``"displacement"``).

    Normally created with ``FESystem.series`` or ``FESystem.modal_series``.
    """

    def __init__(self, data, template, steps=None, step_name="step", step_unit=None, label=None):
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"FieldSeries: data must be 2-D (n_steps, n_dof), got shape {data.shape}.")
        if not isinstance(template, FEField) or template._n_nodes is None:
            raise ValueError("FieldSeries: template must be an FEField with nodal metadata.")
        n_dof = template._n_nodes * template._npn
        if data.shape[1] != n_dof:
            raise ValueError(f"FieldSeries: rows have {data.shape[1]} entries but the system has "
                             f"{n_dof} DOFs. If the history is stored as columns (n_dof, n_steps), "
                             f"pass axis=1 to FESystem.series.")
        if steps is None:
            steps = np.arange(data.shape[0], dtype=float)
        steps = np.asarray(steps)
        if steps.ndim != 1 or steps.shape[0] != data.shape[0]:
            raise ValueError(f"FieldSeries: {data.shape[0]} steps of data but steps has shape "
                             f"{steps.shape}.")
        self._data = data
        self._template = template
        self.steps = steps
        self.step_name = step_name
        self.step_unit = step_unit
        self.label = label if label is not None else template.label

    # ---- sequence protocol --------------------------------------------------------------------
    def __len__(self):
        return self._data.shape[0]

    def _wrap(self, row):
        t = self._template
        return FEField(row, t._n_nodes, t._npn, t._dof_names, t._aliases, t._translational,
                       t._mesh, self.label, t.units)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return FieldSeries(self._data[i], self._template, self.steps[i], self.step_name,
                               self.step_unit, self.label)
        if isinstance(i, (int, np.integer)):
            return self._wrap(self._data[i])
        raise TypeError("FieldSeries indices must be integers or slices; use .at(step_value) to "
                        "select by step value.")

    def __iter__(self):
        for row in self._data:
            yield self._wrap(row)

    def __repr__(self):
        lo, hi = (self.steps[0], self.steps[-1]) if len(self) else (None, None)
        return (f"FieldSeries({self.label or 'field'}: {len(self)} steps of {self._data.shape[1]} DOFs, "
                f"{self.step_name} {lo!r} .. {hi!r})")

    @property
    def first(self):
        return self[0]

    @property
    def last(self):
        return self[-1]

    @property
    def dof_names(self):
        return self._template.dof_names

    @property
    def units(self):
        return self._template.units

    # ---- selecting steps ------------------------------------------------------------------------
    def index_of(self, value, tol=None):
        """Index of the step closest to ``value``. With ``tol`` given, raises ValueError if the
        closest step is farther than that."""
        if len(self) == 0:
            raise ValueError("FieldSeries.index_of: the series is empty.")
        k = int(np.argmin(np.abs(self.steps - value)))
        if tol is not None and abs(self.steps[k] - value) > tol:
            raise ValueError(f"FieldSeries.at: no step within {tol} of {value} "
                             f"(closest {self.step_name} is {self.steps[k]}).")
        return k

    def at(self, value, tol=None):
        """The `FEField` of the step nearest to ``value`` (see `index_of`)."""
        return self[self.index_of(value, tol)]

    # ---- bulk access ------------------------------------------------------------------------
    def _nodal(self):
        t = self._template
        return np.asarray(self._data).reshape(len(self), t._n_nodes, t._npn)

    def component(self, name, nodes=None):
        """Values of one DOF at the selected nodes for every step: shape (n_steps, n_nodes_selected).
        Names, aliases, ``"x"/"y"/"z"`` and node-set names work exactly as in `FEField.component`."""
        t = self._template
        j = t._dof_index(name, "component")
        ids = t._node_ids(nodes, "component")
        arr = self._nodal()[:, :, j]
        return arr if ids is None else arr[:, ids]

    def magnitude(self, nodes=None):
        """Translation magnitude per step and node: shape (n_steps, n_nodes_selected)."""
        t = self._template
        mask = t._translational or (True,) * t._npn
        idx = [j for j, tr in enumerate(mask) if tr]
        mag = np.sqrt(np.sum(np.abs(self._nodal()[:, :, idx]) ** 2, axis=2))
        ids = t._node_ids(nodes, "magnitude")
        return mag if ids is None else mag[:, ids]

    def history(self, name, node, reduce=None):
        """One curve against the steps: shape (n_steps,).

        ``node`` is a node id or a node-set name. If it selects several nodes, say how to combine
        them with ``reduce="mean"``, ``"max"``, ``"min"`` or ``"absmax"``; otherwise ValueError.
        """
        vals = self.component(name, nodes=[node] if isinstance(node, (int, np.integer)) else node)
        if vals.shape[1] == 1:
            return vals[:, 0]
        if reduce is None:
            raise ValueError(f"FieldSeries.history: {vals.shape[1]} nodes selected; pass reduce="
                             f"'mean', 'max', 'min' or 'absmax', or select a single node.")
        ops = {"mean": lambda a: a.mean(axis=1), "max": lambda a: a.max(axis=1),
               "min": lambda a: a.min(axis=1), "absmax": lambda a: np.abs(a).max(axis=1)}
        if reduce not in ops:
            raise ValueError(f"FieldSeries.history: reduce must be one of {sorted(ops)}, got {reduce!r}.")
        return ops[reduce](vals)

    def peak(self, name, nodes=None):
        """Maximum absolute value over all steps, per node: shape (n_nodes_selected,)."""
        return np.abs(self.component(name, nodes)).max(axis=0)

    def to_array(self, layout="steps_first"):
        """Plain ndarray of the whole series: ``"steps_first"`` -> (n_steps, n_dof) (the shape the
        transient/nonlinear drivers return), ``"dofs_first"`` -> (n_dof, n_steps) (the shape of
        ``solve_modal`` mode shapes)."""
        if layout == "steps_first":
            return np.asarray(self._data)
        if layout == "dofs_first":
            return np.asarray(self._data).T
        raise ValueError("FieldSeries.to_array: layout must be 'steps_first' or 'dofs_first'.")

    def to_dataframe(self, component, nodes=None):
        """pandas DataFrame indexed by the steps, one column per selected node (``node<id>``).
        Requires pandas."""
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("FieldSeries.to_dataframe() needs pandas (pip install pandas).") from err
        t = self._template
        ids = t._node_ids(nodes, "to_dataframe")
        ids = np.arange(t._n_nodes) if ids is None else ids
        vals = self.component(component, nodes=ids)
        return pd.DataFrame(vals, index=pd.Index(self.steps, name=self.step_name),
                            columns=[f"node{int(i)}" for i in ids])

    # ---- plotting -----------------------------------------------------------------------------
    def plot_history(self, component, nodes=None, ax=None, reduce=None, title=None):
        """Plot component-vs-step curves and return the Matplotlib axes: one line per selected node,
        or a single line when ``reduce`` is given (see `history`)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError("FieldSeries.plot_history() needs matplotlib.") from err
        t = self._template
        j = t._dof_index(component, "plot_history")
        name = t._dof_names[j] if t._dof_names else f"dof {j}"
        unit = t.unit_of(component)
        if ax is None:
            _, ax = plt.subplots()
        if reduce is not None:
            ax.plot(self.steps, self.history(component, nodes, reduce=reduce), label=f"{reduce} {name}")
        else:
            vals = self.component(component, nodes)
            ids = t._node_ids(nodes, "plot_history")
            ids = np.arange(t._n_nodes) if ids is None else ids
            for k in range(vals.shape[1]):
                ax.plot(self.steps, vals[:, k], label=f"node {int(ids[k])}")
        if 0 < len(ax.lines) <= 8:                   # more lines than that: a legend is just noise
            ax.legend()
        ax.set_xlabel(f"{self.step_name} [{self.step_unit}]" if self.step_unit else self.step_name)
        ax.set_ylabel(f"{name} [{unit}]" if unit else name)
        ax.set_title(title if title is not None else (self.label or "history"))
        return ax
