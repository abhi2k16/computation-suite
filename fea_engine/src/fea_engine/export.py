# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
export.py -- write meshes and results for ParaView / VisIt (VTK XML ``.vtu`` and ``.pvd``).

No dependency beyond NumPy. Typical use::

    U = system.solve_static()
    fea_engine.export.write_vtu("beam.vtu", U)                       # one result
    fea_engine.export.write_vtu("modes.vtu", shapes)                 # one array per mode
    fea_engine.export.write_vtu("run.vtu", {"u": U, "T": temp})      # several fields, one mesh
    fea_engine.export.write_series("run/disp", system.series(hist))  # run/disp.pvd + run/disp_0000.vtu ...

What is written
---------------
* Points: the mesh nodes (2-D meshes are padded to z = 0).
* Cells: every element block, with the VTK cell type chosen from the node count and mesh dimension
  (line, triangle, quad, tetra, hexahedron; see ``high_order``).
* Point data per field: translational DOFs as one 3-component vector named after the field label
  (``"displacement"``), every other DOF (rotations, ...) as a scalar named after the DOF, and for
  multi-column fields (mode shapes) one set per column suffixed ``_mode<k>``. Complex fields give
  ``_re`` / ``_im`` / ``_abs`` arrays.
* Extra ``point_data`` / ``cell_data`` dictionaries pass through arrays you computed yourself.

``high_order``: ``"linear"`` (default) draws Tri6/Quad8/Quad9/Tet10/Hex20 elements through their corner
nodes only, which is correct for every node-ordering convention; ``"native"`` writes VTK quadratic
cells and assumes the VTK node order (right for Tri6 and Quad8/9 in the usual counter-clockwise
corner-then-midside layout; check Tet10 / Hex20 on a small model before trusting it).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import base64
import os
import struct
from xml.sax.saxutils import quoteattr

import numpy as np

from .fields import FEField
from .series import FieldSeries

# VTK cell type ids
_LINE, _TRI, _QUAD, _TET, _HEX = 3, 5, 9, 10, 12
_QUAD_LINE, _QUAD_TRI, _QUAD_QUAD, _QUAD_TET, _QUAD_HEX, _BIQUAD_QUAD = 21, 22, 23, 24, 25, 28


def _cell_layout(k, dim, name, high_order):
    """(vtk_type, n_nodes_written) for an element with k nodes."""
    name = (name or "").lower()
    native = high_order == "native"
    if k == 2:
        return _LINE, 2
    if k == 3 and (dim == 1 or name.startswith(("beam", "line"))):
        return (_QUAD_LINE, 3) if native else (_LINE, 2)
    if k == 3:
        return _TRI, 3
    if k == 4:
        return (_TET, 4) if (dim == 3 or name.startswith("tet")) else (_QUAD, 4)
    if k == 6:
        return (_QUAD_TRI, 6) if native else (_TRI, 3)
    if k == 8:
        if dim == 3 or name.startswith("hex"):
            return _HEX, 8
        return (_QUAD_QUAD, 8) if native else (_QUAD, 4)
    if k == 9:
        return (_BIQUAD_QUAD, 9) if native else (_QUAD, 4)
    if k == 10:
        return (_QUAD_TET, 10) if native else (_TET, 4)
    if k == 20:
        return (_QUAD_HEX, 20) if native else (_HEX, 8)
    raise NotImplementedError(f"export: no VTK cell type for an element with {k} nodes (mesh dim {dim}).")


def _blocks(mesh):
    blocks = getattr(mesh, "blocks", None)
    if blocks:
        return list(blocks.items())
    return [("", mesh.elements)]


class _Writer:
    def __init__(self, binary):
        self.binary = binary
        self.parts = []

    def array(self, name, arr, vtk_type, ncomp=1):
        arr = np.ascontiguousarray(arr)
        head = f'<DataArray type="{vtk_type}" Name={quoteattr(name)} NumberOfComponents="{ncomp}"'
        if self.binary:
            raw = arr.tobytes()
            payload = base64.b64encode(struct.pack("<I", len(raw)) + raw).decode("ascii")
            self.parts.append(f'{head} format="binary">{payload}</DataArray>')
        else:
            if arr.dtype.kind == "f":
                text = " ".join(f"{v:.9g}" for v in arr.ravel())
            else:
                text = " ".join(str(int(v)) for v in arr.ravel())
            self.parts.append(f'{head} format="ascii">{text}</DataArray>')


def _field_arrays(field, name=None):
    """Expand one FEField into {array name: (n_nodes[,3]) ndarray}."""
    if not isinstance(field, FEField) or field._n_nodes is None:
        raise ValueError("export: fields must be FEField objects with nodal metadata "
                         "(use system.field(...) to wrap a plain vector).")
    base = name or field.label or "u"
    nod = field.nodal                                  # (n_nodes, npn[, k])
    ncols = nod.shape[2] if nod.ndim == 3 else 1
    if nod.ndim == 2:
        nod = nod[:, :, None]
    names = field.dof_names or tuple(f"d{j}" for j in range(field._npn))
    mask = field._translational or (True,) * field._npn
    trans = [j for j, t in enumerate(mask) if t]
    out = {}
    for c in range(ncols):
        suffix = f"_mode{c}" if ncols > 1 else ""
        parts = {}
        vec = np.zeros((field._n_nodes, 3), dtype=nod.dtype)
        for slot, j in enumerate(trans[:3]):
            vec[:, slot] = nod[:, j, c]
        if trans:
            parts[base] = vec
        for j in range(field._npn):
            if j not in trans:
                parts[f"{names[j]}"] = nod[:, j, c]
        for key, val in parts.items():
            key = f"{key}{suffix}"
            if np.iscomplexobj(val):
                out[key + "_re"] = np.ascontiguousarray(val.real)
                out[key + "_im"] = np.ascontiguousarray(val.imag)
                out[key + "_abs"] = np.abs(val)
            else:
                out[key] = np.ascontiguousarray(val)
    return out


def write_vtu(path, fields=None, mesh=None, point_data=None, cell_data=None,
              high_order="linear", binary=False):
    """Write a mesh plus results to a VTK XML unstructured-grid file (open it in ParaView).

    Parameters
    ----------
    path : str or path-like
        Output file; ``.vtu`` is appended when missing.
    fields : FEField or dict of name -> FEField, optional
        Results to store as point data. A single field uses its label as the name.
    mesh : Mesh or MultiBlockMesh, optional
        Defaults to the mesh attached to the first field.
    point_data : dict, optional
        Extra arrays of shape ``(n_nodes,)`` or ``(n_nodes, 3)``.
    cell_data : dict, optional
        Arrays with one value per element, in block order (``(n_elements,)`` or ``(n_elements, k)``).
    high_order : {"linear", "native"}
        How quadratic elements are written (see the module docstring).
    binary : bool
        Base64 binary arrays instead of ASCII (smaller and faster for big meshes).

    Returns
    -------
    str
        The path written.
    """
    if high_order not in ("linear", "native"):
        raise ValueError("export: high_order must be 'linear' or 'native'.")
    if isinstance(fields, FEField):
        fields = {None: fields}
    fields = dict(fields or {})
    if mesh is None:
        for f in fields.values():
            mesh = getattr(f, "_mesh", None)
            if mesh is not None:
                break
    if mesh is None or getattr(mesh, "nodes", None) is None:
        raise ValueError("export: no mesh available -- pass mesh= or use fields made by system.field() "
                         "/ solve_*() (a pickled field has no mesh).")
    coords = np.asarray(mesh.nodes, dtype=float)
    n_nodes = coords.shape[0]
    pts = np.zeros((n_nodes, 3))
    pts[:, :coords.shape[1]] = coords[:, :3]
    dim = int(getattr(mesh, "dim", coords.shape[1]))

    conn, offsets, types = [], [], []
    total = 0
    n_elems = 0
    for bname, elems in _blocks(mesh):
        elems = np.asarray(elems, dtype=int)
        vtype, nw = _cell_layout(elems.shape[1], dim, bname, high_order)
        for row in elems:
            conn.extend(int(i) for i in row[:nw])
            total += nw
            offsets.append(total)
            types.append(vtype)
        n_elems += len(elems)

    arrays = {}
    for name, f in fields.items():
        for key, val in _field_arrays(f, name).items():
            if val.shape[0] != n_nodes:
                raise ValueError(f"export: field array {key!r} has {val.shape[0]} nodes, mesh has {n_nodes}.")
            arrays[key] = val
    for key, val in (point_data or {}).items():
        val = np.asarray(val)
        if val.shape[0] != n_nodes:
            raise ValueError(f"export: point_data {key!r} has {val.shape[0]} rows, mesh has {n_nodes} nodes.")
        arrays[key] = val
    cell_arrays = {}
    for key, val in (cell_data or {}).items():
        val = np.asarray(val)
        if val.shape[0] != n_elems:
            raise ValueError(f"export: cell_data {key!r} has {val.shape[0]} rows, mesh has {n_elems} elements.")
        cell_arrays[key] = val

    w =_Writer(binary)
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">',
             "<UnstructuredGrid>",
             f'<Piece NumberOfPoints="{n_nodes}" NumberOfCells="{n_elems}">',
             "<Points>"]
    w.array("Points", pts, "Float64", 3)
    lines += w.parts; w.parts = []
    lines += ["</Points>", "<Cells>"]
    w.array("connectivity", np.asarray(conn, dtype=np.int64), "Int64")
    w.array("offsets", np.asarray(offsets, dtype=np.int64), "Int64")
    w.array("types", np.asarray(types, dtype=np.uint8), "UInt8")
    lines += w.parts; w.parts = []
    lines.append("</Cells>")

    def emit(tag, data):
        if not data:
            return
        lines.append(f"<{tag}>")
        for key, val in data.items():
            arr = val.astype(np.float64) if val.dtype.kind == "f" else (
                val.astype(np.int64) if val.dtype.kind in "iub" else val.astype(np.float64))
            ncomp = 1 if arr.ndim == 1 else int(np.prod(arr.shape[1:]))
            w.array(key, arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr,
                    "Float64" if arr.dtype.kind == "f" else "Int64", ncomp)
        lines.extend(w.parts); w.parts = []
        lines.append(f"</{tag}>")

    emit("PointData", arrays)
    emit("CellData", cell_arrays)
    lines += ["</Piece>", "</UnstructuredGrid>", "</VTKFile>"]

    path = os.fspath(path)
    if not path.lower().endswith(".vtu"):
        path += ".vtu"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_series(basename, series, mesh=None, extra=None, high_order="linear", binary=False):
    """Write a ``FieldSeries`` as ``<basename>_0000.vtu`` ... plus ``<basename>.pvd`` (ParaView time series).

    Parameters
    ----------
    basename : str or path-like
        Path without extension.
    series : FieldSeries
    extra : dict of name -> FieldSeries, optional
        More series with the same number of steps, written into the same files.
    Time values in the ``.pvd`` are the series ``steps``.

    Returns
    -------
    str
        The ``.pvd`` path.
    """
    if not isinstance(series, FieldSeries):
        raise TypeError("write_series expects a FieldSeries (see FESystem.series).")
    extra = dict(extra or {})
    for k, s in extra.items():
        if len(s) != len(series):
            raise ValueError(f"export: extra series {k!r} has {len(s)} steps, expected {len(series)}.")
    base = os.fspath(basename)
    if base.lower().endswith((".pvd", ".vtu")):
        base = base[:-4]
    entries = []
    for i in range(len(series)):
        fields = {None: series[i]}
        fields.update({k: s[i] for k, s in extra.items()})
        fpath = write_vtu(f"{base}_{i:04d}.vtu", fields, mesh=mesh, high_order=high_order, binary=binary)
        entries.append((float(np.real(series.steps[i])), os.path.basename(fpath)))
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">', "<Collection>"]
    lines += [f'<DataSet timestep="{t:.12g}" group="" part="0" file={quoteattr(f)}/>' for t, f in entries]
    lines += ["</Collection>", "</VTKFile>"]
    pvd = base + ".pvd"
    with open(pvd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return pvd
