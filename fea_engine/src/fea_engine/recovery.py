# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
recovery.py -- derived results as FEFields: strain, stress, von Mises and support reactions.

Reached through ``FESystem.stress(U)``, ``.strain(U)``, ``.von_mises(U)`` and ``.reactions(U)``::

    U = system.solve_static()
    S = system.stress(U)                      # FEField, components sxx, syy, sxy (2-D) or sxx..sxz (3-D)
    S.component("sxx", nodes="root")          # same named access as displacements
    S.plot("syy")                             # contour, colour bar "syy [Pa]"
    vm = system.von_mises(U)                  # scalar FEField "von_mises"
    R = system.reactions(U)                   # FEField of support forces (zero at free DOFs)
    R.component("uy", nodes="root").sum()     # total vertical reaction at the root

How nodal stress is obtained
----------------------------
Stress is computed at the integration points (``sigma = D(x) B u``) and moved to the corner nodes with a
lumped L2 projection using the linear shape functions: ``sigma_a = sum(N_a sigma |J| w) / sum(N_a |J| w)``
over the elements sharing node ``a``. It is positive-weighted, so it cannot overshoot the integration-point
values, and for constant-strain triangles/tetrahedra it reduces to the usual volume-weighted average.
Mid-side nodes of quadratic elements are interpolated from their corners (a quadratic shape function
integrates to zero or less at a corner, so projecting with it directly would be ill-posed). Values on a
boundary are smoothed estimates, not exact boundary stresses. ``at="elements"`` returns the volume-averaged element
values instead (an ndarray, one row per element in block order, convenient for ``cell_data`` export).

Supported elements: Quad4/Quad8/Tri3/Tri6 plane stress and Hex8/Hex20/Tet4/Tet10 solids (full integration).
D comes from the last ``assemble_stiffness`` call (coefficients included); pass ``D=`` to override.
Voigt order: 2-D ``[xx, yy, xy]``; 3-D ``[xx, yy, zz, xy, yz, xz]``; strains use engineering shear.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from . import coefficients as cf
from .fields import FEField

_NAMES = {
    ("stress", 3): ("sxx", "syy", "sxy"),
    ("stress", 6): ("sxx", "syy", "szz", "sxy", "syz", "sxz"),
    ("strain", 3): ("exx", "eyy", "gxy"),
    ("strain", 6): ("exx", "eyy", "ezz", "gxy", "gyz", "gxz"),
}


def _resolve_D(system, D):
    if D is not None:
        return D
    args = getattr(system, "_stiffness_args", None)
    if args is None:
        raise RuntimeError("stress/strain recovery needs the constitutive matrix: call assemble_stiffness() "
                           "first (it is remembered) or pass D=.")
    return args[0]


def _check_vector(system, U):
    U = np.asarray(U)
    if U.ndim != 1 or U.shape[0] != system.n_dof:
        raise ValueError(f"expected a displacement vector of length {system.n_dof}, got shape {U.shape}.")
    if np.iscomplexobj(U):
        raise NotImplementedError("recovery of complex (harmonic) fields: pass U.real or U.imag.")
    return U


_LAYOUT_CACHE = {}


def _layout(f):
    """(linear element, n_corner, T): the linear companion element, how many leading nodes are corners, and
    T (n_nodes - n_corner, n_corner) interpolating corner values to the remaining (mid-side) nodes."""
    key = type(f)
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    from .elements.solids import Quad4PlaneStress, Hex8Solid3D, Tri3PlaneStress, Tet4Solid3D
    simplex = f.quadrature_family == "simplex"
    lin = {(2, False): Quad4PlaneStress, (3, False): Hex8Solid3D,
           (2, True): Tri3PlaneStress, (3, True): Tet4Solid3D}[(f.dim, simplex)]()
    nc = lin.n_nodes
    T = None
    if f.n_nodes != nc:
        import itertools
        grid = [0.0, 0.5, 1.0] if simplex else [-1.0, 0.0, 1.0]
        pos = []
        for a in range(f.n_nodes):
            for p in itertools.product(grid, repeat=f.dim):
                N = f.shape_and_derivs(p)[0]
                if abs(N[a] - 1.0) < 1e-12 and abs(np.abs(N).sum() - 1.0) < 1e-12:
                    pos.append(p)
                    break
            else:
                raise NotImplementedError(f"recovery: cannot locate node {a} of {type(f).__name__}.")
        for a in range(nc):          # the first nc nodes must be the corners, in the linear element's order
            if not np.allclose(lin.shape_and_derivs(pos[a])[0], np.eye(nc)[a]):
                raise NotImplementedError(f"recovery: {type(f).__name__} corner nodes are not its first {nc}.")
        T = np.array([lin.shape_and_derivs(pos[a])[0] for a in range(nc, f.n_nodes)])
    _LAYOUT_CACHE[key] = (lin, nc, T)
    return _LAYOUT_CACHE[key]


def _recover(system, U, D, kind):
    """-> (nodal (n_nodes, ncomp), element-average (n_elements, ncomp)).

    Integration-point values are projected to the corner nodes with the (positive) linear shape functions;
    mid-side nodes of quadratic elements are interpolated from their corners."""
    U = _check_vector(system, U)
    D = _resolve_D(system, D) if kind == "stress" else None
    n_nodes = len(system.mesh.nodes)
    num = den = None
    elem_rows = []
    mids = []
    ncomp = None
    for name, f, conn in system._blocks:
        cf.check_supported(f, f"{kind} recovery")
        nc = 3 if f.dim == 2 else 6
        if ncomp is None:
            ncomp = nc
            num = np.zeros((n_nodes, ncomp))
            den = np.zeros(n_nodes)
        elif nc != ncomp:
            raise NotImplementedError("recovery needs all blocks to be 2-D or all 3-D.")
        block_D = system._per_block_arg(D, name) if D is not None else None
        rule = cf.quadrature(f, "recovery")
        lin, nc, T = _layout(f)
        for i, e in enumerate(conn):
            coords = system.mesh.nodes[e]
            ue = U[system._global_dofs(e)]
            acc, vol = np.zeros(ncomp), 0.0
            for p, w in rule:
                B, detJ = f.B_matrix(p, coords)
                N = f._cached_shape_and_derivs(p)[0]
                Nl = lin.shape_and_derivs(p)[0]
                val = B @ ue
                if kind == "stress":
                    Dg = cf._val(block_D, i, coords, N @ coords)
                    val = np.asarray(Dg) @ val
                wt = abs(detJ) * w
                num[e[:nc]] += wt * Nl[:, None] * val[None, :]
                den[e[:nc]] += wt * Nl
                acc += wt * val
                vol += wt
            elem_rows.append(acc / vol)
            if T is not None:
                mids.append(e)
        if T is not None:
            mids_T = T
    nodal = np.divide(num, den[:, None], out=np.zeros_like(num), where=den[:, None] > 0)
    if mids:
        acc = np.zeros_like(nodal)
        cnt = np.zeros(n_nodes)
        for e in mids:
            acc[e[nc:]] += mids_T @ nodal[e[:nc]]
            cnt[e[nc:]] += 1
        only_mid = (den == 0) & (cnt > 0)
        nodal[only_mid] = acc[only_mid] / cnt[only_mid, None]
    return nodal, np.array(elem_rows)


def _field(system, data, names, label, unit, translational=None):
    n = len(system.mesh.nodes)
    k = len(names)
    return FEField(np.ascontiguousarray(data).reshape(n * k), n_nodes=n, dofs_per_node=k, dof_names=names,
                   aliases={}, translational=translational or (False,) * k, mesh=system.mesh, label=label,
                   units=unit, dof_units=(unit or "",) * k if unit else None)


def _stress_unit(system):
    return (system.units.stress or None) if system.units is not None else None


def stress(system, U, D=None, at="nodes"):
    """Stress components; see the module docstring. ``at``: ``"nodes"`` (FEField) or ``"elements"``."""
    nodal, elem = _recover(system, U, D, "stress")
    if at == "elements":
        return elem
    if at != "nodes":
        raise ValueError("at must be 'nodes' or 'elements'.")
    return _field(system, nodal, _NAMES[("stress", nodal.shape[1])], "stress", _stress_unit(system))


def strain(system, U, at="nodes"):
    """Strain components (engineering shear), same layout as :func:`stress`."""
    nodal, elem = _recover(system, U, None, "strain")
    if at == "elements":
        return elem
    if at != "nodes":
        raise ValueError("at must be 'nodes' or 'elements'.")
    return _field(system, nodal, _NAMES[("strain", nodal.shape[1])], "strain", None)


def _mises(s, plane, nu):
    if s.shape[-1] == 6:
        sxx, syy, szz, sxy, syz, sxz = (s[..., j] for j in range(6))
        return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                       + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))
    sxx, syy, sxy = (s[..., j] for j in range(3))
    if plane == "stress":
        return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)
    szz = nu * (sxx + syy)
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * sxy ** 2)


def von_mises(system, U, D=None, plane="stress", nu=None, at="nodes"):
    """Von Mises equivalent stress. 2-D: ``plane="stress"`` (default) or ``"strain"`` (needs ``nu``)."""
    if plane not in ("stress", "strain"):
        raise ValueError("plane must be 'stress' or 'strain'.")
    if plane == "strain" and nu is None:
        raise ValueError("von_mises(plane='strain') needs the Poisson ratio nu= to form sigma_zz.")
    nodal, elem = _recover(system, U, D, "stress")
    if at == "elements":
        return _mises(elem, plane, nu)
    if at != "nodes":
        raise ValueError("at must be 'nodes' or 'elements'.")
    unit = _stress_unit(system)
    f = _field(system, _mises(nodal, plane, nu), ("von_mises",), "von Mises stress", unit, translational=(True,))
    f._aliases = {"vm": "von_mises", "mises": "von_mises"}
    f.units = unit
    return f


def reactions(system, U=None):
    """Support reactions ``R = K U - F`` at the constrained DOFs (zero elsewhere) as an FEField.

    Forces for translational DOFs, moments for rotations. Sum a component over a node set to get the
    total reaction; with the applied loads in ``system.F`` it balances to rounding error.
    """
    if U is None:
        U = system.solve_static()
    U = _check_vector(system, U)
    system._require_stiffness("reactions")
    K = system._as_solve_matrix(system.K)
    r = np.asarray(K @ U).ravel() - np.asarray(system.F).ravel()
    out = np.zeros(system.n_dof)
    fixed = system.fixed_dofs_array
    out[fixed] = r[fixed]
    units = system.units
    elem = system._blocks[0][1]
    mask = elem.translational_dof_mask
    mask = mask if mask is not None else (True,) * system.npn
    dof_units = None
    if units is not None:
        dof_units = tuple(units.force if t else f"{units.force}*{units.length}" for t in mask)
    f = system.field(out, label="reaction")
    if isinstance(f, FEField):
        f._dof_units = dof_units
        f.units = units.force if units is not None else None
    return f
