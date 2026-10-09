"""
fields.py -- FEField: a solution vector that knows what its entries mean.

WHY. FESystem's solvers return flat NumPy vectors, so reading one value means hand-written index
arithmetic (`U[2*node + 1]`), which silently goes wrong when the element changes. PyMAPDL's
`post_processing.nodal_displacement("X")` and PyDPF's self-describing `Field` (data + scoping +
location) show the better pattern: keep the data, say what it is, and select by name.

HOW. `FEField` is a thin `numpy.ndarray` SUBCLASS, so it is a drop-in replacement for the plain
array the solvers returned before: slicing, arithmetic, `np.linalg.norm`, `U[2*i+1]`, saving, passing
to SciPy/Torch all behave as they did. On top of that it adds named access:

    U = system.solve_static()              # FEField, shape (n_dof,)
    U.nodal                                # (n_nodes, dofs_per_node) view
    U.component("uy", nodes="tip")         # values of DOF "uy" at the nodes of the set "tip"
    U.displacement("y", nodes=tip_ids)     # same, translational shorthand ("x"/"y"/"z" or "ux"...)
    U.magnitude(nodes="tip")               # |translation| per node
    U.to_dataframe()                       # pandas table (optional dependency)

A mode-shape matrix from `solve_modal()` has shape (n_dof, n_modes) and the same methods return
(n_nodes, n_modes) data, one column per mode.

Metadata is only valid while the array still has the full nodal length; a slice such as `U[free]` is
still an FEField but the nodal accessors then raise a clear error instead of returning wrong data.
"""
import numpy as np


def _rebuild_fefield(data, meta):
    return FEField(data, **meta)


class FEField(np.ndarray):
    """ndarray of nodal DOF values with names. See the module docstring."""

    def __new__(cls, data, n_nodes=None, dofs_per_node=None, dof_names=None, aliases=None,
                translational=None, mesh=None, label=None):
        obj = np.asarray(data).view(cls)
        obj._n_nodes = n_nodes
        obj._npn = dofs_per_node
        obj._dof_names = tuple(dof_names) if dof_names is not None else None
        obj._aliases = dict(aliases) if aliases else {}
        obj._translational = None if translational is None else tuple(bool(t) for t in translational)
        obj._mesh = mesh
        obj.label = label
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        for name in ("_n_nodes", "_npn", "_dof_names", "_aliases", "_translational", "_mesh", "label"):
            setattr(self, name, getattr(obj, name, None if name != "_aliases" else {}))

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        # Reductions / dot products of a plain ndarray give NumPy scalars; a bare subclass would give
        # 0-d FEFields instead (not JSON-serialisable, np.isscalar False, ...). Keep the old behaviour.
        if getattr(out_arr, "ndim", 1) == 0:
            return out_arr[()]
        return np.ndarray.__array_wrap__(self, out_arr, context)

    # ---- pickling: keep the small metadata, drop the (possibly large) mesh reference ---------
    def __reduce__(self):
        meta = dict(n_nodes=self._n_nodes, dofs_per_node=self._npn, dof_names=self._dof_names,
                    aliases=self._aliases, translational=self._translational, mesh=None,
                    label=self.label)
        return (_rebuild_fefield, (np.asarray(self), meta))

    # ---- structure -------------------------------------------------------------------------
    @property
    def dof_names(self):
        """Per-node DOF names in local order, e.g. ('ux', 'uy') (None if unknown)."""
        return self._dof_names

    @property
    def is_nodal(self):
        """True while the array still has the full (n_nodes * dofs_per_node) leading length."""
        return (self._n_nodes is not None and self._npn is not None
                and self.ndim >= 1 and self.shape[0] == self._n_nodes * self._npn)

    def _require_nodal(self, who):
        if self._n_nodes is None or self._npn is None:
            raise ValueError(f"FEField.{who}: this array carries no nodal metadata.")
        if not self.is_nodal:
            raise ValueError(
                f"FEField.{who}: the array has leading length {self.shape[0] if self.ndim else '()'} "
                f"but a full nodal field needs {self._n_nodes * self._npn} "
                f"({self._n_nodes} nodes x {self._npn} DOFs) -- it was probably sliced "
                f"(e.g. U[free]). Use the full vector.")

    @property
    def nodal(self):
        """Plain ndarray view with shape (n_nodes, dofs_per_node[, n_columns])."""
        self._require_nodal("nodal")
        return np.asarray(self).reshape((self._n_nodes, self._npn) + tuple(self.shape[1:]))

    # ---- name resolution ---------------------------------------------------------------------
    def _dof_index(self, component, who):
        if isinstance(component, (int, np.integer)):
            if not 0 <= int(component) < self._npn:
                raise ValueError(f"FEField.{who}: DOF index {int(component)} out of range "
                                 f"0..{self._npn - 1}.")
            return int(component)
        names = self._dof_names
        if names is None:
            raise ValueError(f"FEField.{who}: DOF names are unknown for this field; "
                             f"pass an integer index.")
        key = str(component).strip().lower()
        low = [n.lower() for n in names]
        if key in low:
            return low.index(key)
        # "x" / "y" / "z" shorthand for the translations
        if key in ("x", "y", "z") and f"u{key}" in low:
            return low.index(f"u{key}")
        canon = self._aliases.get(key)
        if canon is not None and canon.lower() in low:
            return low.index(canon.lower())
        raise ValueError(f"FEField.{who}: unknown DOF {component!r}; valid names: {list(names)}"
                         + (f", aliases: {sorted(self._aliases)}" if self._aliases else "") + ".")

    def _node_ids(self, nodes, who):
        if nodes is None:
            return None
        if isinstance(nodes, str) or (isinstance(nodes, (list, tuple)) and nodes
                                      and all(isinstance(x, str) for x in nodes)):
            sets = getattr(self._mesh, "node_sets", None)
            if sets is None:
                raise ValueError(f"FEField.{who}: node set {nodes!r} requested but this field has "
                                 f"no mesh attached (it was pickled or created without one).")
            names = [nodes] if isinstance(nodes, str) else list(nodes)
            missing = [n for n in names if n not in sets]
            if missing:
                raise ValueError(f"FEField.{who}: unknown node set(s) {missing}; "
                                 f"available: {sorted(sets)}.")
            ids = np.unique(np.concatenate([np.asarray(sets[n], dtype=int) for n in names]))
        else:
            ids = np.asarray(nodes).reshape(-1)
            if ids.size and ids.dtype.kind not in "iu":
                raise ValueError(f"FEField.{who}: node ids must be integers.")
            ids = ids.astype(int)
        if ids.size == 0:
            raise ValueError(f"FEField.{who}: no nodes selected.")
        if ids.min() < 0 or ids.max() >= self._n_nodes:
            raise ValueError(f"FEField.{who}: node id outside the mesh (0..{self._n_nodes - 1}).")
        return ids

    # ---- access ------------------------------------------------------------------------------
    def component(self, name, nodes=None):
        """Values of one DOF (by name such as 'uy', alias, 'x'/'y'/'z' shorthand, or integer
        index). `nodes`: None (all nodes), an array of node ids, or the name(s) of mesh node sets.
        Returns a plain ndarray of shape (n_selected[, n_columns])."""
        self._require_nodal("component")
        j = self._dof_index(name, "component")
        data = self.nodal[:, j]
        ids = self._node_ids(nodes, "component")
        return data if ids is None else data[ids]

    displacement = component      # translational shorthand: U.displacement("y", nodes="tip")

    def at(self, nodes):
        """All DOFs at the given nodes: shape (n_selected, dofs_per_node[, n_columns])."""
        self._require_nodal("at")
        return self.nodal[self._node_ids(nodes, "at")]

    def magnitude(self, nodes=None):
        """Euclidean norm of the TRANSLATIONAL DOFs at each node (rotations excluded)."""
        self._require_nodal("magnitude")
        mask = self._translational
        if mask is None:
            mask = (True,) * self._npn
        idx = [j for j, t in enumerate(mask) if t]
        data = self.nodal[:, idx]
        mag = np.sqrt(np.sum(np.abs(data) ** 2, axis=1))
        ids = self._node_ids(nodes, "magnitude")
        return mag if ids is None else mag[ids]

    def to_dataframe(self, column=0):
        """pandas DataFrame with one row per node: node id, coordinates (if a mesh is attached)
        and one column per DOF name. For a 2-D field (modes/steps) pick the column with `column`.
        Requires pandas."""
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("FEField.to_dataframe() needs pandas (pip install pandas).") from err
        self._require_nodal("to_dataframe")
        nod = self.nodal
        if nod.ndim == 3:
            nod = nod[:, :, column]
        names = self._dof_names or tuple(f"d{j}" for j in range(self._npn))
        cols = {"node": np.arange(self._n_nodes)}
        nodes_xyz = getattr(self._mesh, "nodes", None)
        if nodes_xyz is not None:
            for a, ax in enumerate("xyz"[:nodes_xyz.shape[1]]):
                cols[ax] = nodes_xyz[:, a]
        for j, n in enumerate(names):
            cols[n] = nod[:, j]
        return pd.DataFrame(cols)
