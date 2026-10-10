# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
coefficients.py -- material and section data that vary in space (MFEM-style coefficients).

``assemble_stiffness`` / ``assemble_mass`` / ``assemble_lumped_mass`` normally take ONE constitutive
matrix or density for the whole mesh (or a dict with one per block). A ``Coefficient`` lets any of
them, and the thickness, depend on position::

    from fea_engine import coefficients as cf

    # Young's modulus grows with height; D rebuilt from the local material
    D = cf.from_material(lambda x: Material(E=70e9 * (1 + 2 * x[1]), nu=0.3), D_plane_stress, at="gauss")
    system.assemble_stiffness(D, thickness=0.02)

    # tapered plate: thickness depends on x; density graded the same way
    system.assemble_stiffness(D_plane_stress(mat), thickness=cf.by_position(lambda x: 0.02 * (1 - 0.5 * x[0])))
    system.assemble_mass(cf.by_position(lambda x: 2700 * (1 + x[1]) * np.eye(2)), thickness=0.02)

    # one value per element (e.g. from a topology-optimisation density field)
    system.assemble_stiffness(cf.by_element([D_plane_stress(m) for m in element_materials]), thickness=0.02)

Two evaluation modes (``at=``):

* ``"centroid"`` (default): the function is evaluated once per element at the average of its nodes, so
  the coefficient is piecewise constant. Works with every element type and with the existing element
  stiffness/mass routines, so the result for a constant function is identical to the plain call.
* ``"gauss"``: evaluated at each integration point (``x = N(xi) . node_coords``), so a smoothly graded
  material is integrated properly inside each element. Supported for Quad4/Quad8/Tri3/Tri6 plane stress
  and Hex8/Hex20/Tet4/Tet10 solids with full integration; other elements raise ``NotImplementedError``.

Stress recovery (``FESystem.stress`` and friends) re-uses the coefficient that was assembled.
Scope: linear assembly only; the nonlinear drivers still take one material per block.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .elements.base import gauss_product, tri_quadrature, tri_quadrature_3pt, tet_quadrature, \
    tet_quadrature_4pt, jacobian
from .elements.solids import (Quad4PlaneStress, Quad8PlaneStress, Hex8Solid3D, Hex20Solid3D,
                              Tri3PlaneStress, Tri6PlaneStress, Tet4Solid3D, Tet10Solid3D)

SUPPORTED_ELEMENTS = (Quad4PlaneStress, Quad8PlaneStress, Hex8Solid3D, Hex20Solid3D,
                      Tri3PlaneStress, Tri6PlaneStress, Tet4Solid3D, Tet10Solid3D)


class Coefficient:
    """A value (matrix or scalar) that depends on the element or on position. Build with
    :func:`by_position`, :func:`by_element` or :func:`from_material`."""

    def __init__(self, fn=None, values=None, at="centroid"):
        if at not in ("centroid", "gauss"):
            raise ValueError("Coefficient: at must be 'centroid' or 'gauss'.")
        if (fn is None) == (values is None):
            raise ValueError("Coefficient: give exactly one of fn or values.")
        if values is not None and at == "gauss":
            raise ValueError("Coefficient: per-element values cannot be evaluated at Gauss points; "
                             "use by_position(fn, at='gauss').")
        if fn is not None and not callable(fn):
            raise TypeError("Coefficient: fn must be callable, fn(x) -> value.")
        self.fn, self.values, self.at = fn, values, at

    @property
    def gauss(self):
        return self.at == "gauss"

    def element_value(self, i, coords):
        """Piecewise-constant value for element ``i`` (position in its block) with node coords."""
        if self.values is not None:
            if i >= len(self.values):
                raise ValueError(f"Coefficient.by_element: {len(self.values)} values but element index {i}.")
            return np.asarray(self.values[i])
        return np.asarray(self.fn(np.asarray(coords, dtype=float).mean(axis=0)))

    def point_value(self, x):
        return np.asarray(self.fn(np.asarray(x, dtype=float)))

    def __repr__(self):
        kind = "by_element" if self.values is not None else f"by_position(at={self.at!r})"
        return f"Coefficient({kind})"


def by_position(fn, at="centroid"):
    """Coefficient ``fn(x) -> value`` where ``x`` is the coordinate vector of the evaluation point."""
    return Coefficient(fn=fn, at=at)


def by_element(values):
    """Coefficient with one value per element of a block, in connectivity order."""
    return Coefficient(values=list(values))


def from_material(material_fn, builder, at="centroid"):
    """Coefficient ``builder(material_fn(x))``, e.g. ``from_material(fn, D_plane_stress)``."""
    return Coefficient(fn=lambda x: builder(material_fn(x)), at=at)


def has_coefficient(*objs):
    """True when any argument (or any value of a dict argument) is a Coefficient."""
    for o in objs:
        vals = o.values() if isinstance(o, dict) else (o,)
        if any(isinstance(v, Coefficient) for v in vals):
            return True
    return False


def check_supported(formulation, what):
    if type(formulation) not in SUPPORTED_ELEMENTS:
        raise NotImplementedError(
            f"{what} is not available for {type(formulation).__name__}; supported: "
            f"{', '.join(c.__name__ for c in SUPPORTED_ELEMENTS)}. Use at='centroid' instead.")


def quadrature(formulation, purpose="stiffness"):
    """List of ``(natural_point, weight)`` with the simplex area/volume factor folded into the weight, so
    ``sum(f * abs(detJ) * w)`` is the element integral."""
    if formulation.quadrature_family == "tensor":
        pts, wts = gauss_product(formulation.gauss_order, formulation.dim)
        return list(zip(pts, wts))
    dim = formulation.dim
    linear = formulation.n_nodes == dim + 1
    if dim == 2:
        if linear:
            pts, wts = tri_quadrature(2) if purpose == "mass" else ([(1 / 3, 1 / 3)], [1.0])
        else:
            pts, wts = tri_quadrature_3pt()
        factor = 0.5
    else:
        if linear:
            pts, wts = tet_quadrature(2) if purpose == "mass" else ([(0.25, 0.25, 0.25)], [1.0])
        else:
            pts, wts = tet_quadrature_4pt()
        factor = 1.0 / 6.0
    return [(tuple(p), w * factor) for p, w in zip(pts, wts)]


def _val(c, i, coords, x):
    if not isinstance(c, Coefficient):
        return c
    return c.point_value(x) if c.gauss else c.element_value(i, coords)


def resolve_centroid(arg, i, coords):
    """Value of a possibly-Coefficient argument for element ``i`` (centroid / per-element mode)."""
    return arg.element_value(i, coords) if isinstance(arg, Coefficient) else arg


def stiffness(formulation, coords, D, kwargs, i, method="full"):
    """Element stiffness when ``D`` and/or ``kwargs['thickness']`` may be Coefficients."""
    kw = dict(kwargs)
    has_th = "thickness" in kw
    th = kw.pop("thickness", 1.0)
    if any(isinstance(c, Coefficient) and c.gauss for c in (D, th)):
        if method != "full":
            raise ValueError("gauss-point coefficients need method='full'.")
        check_supported(formulation, "Gauss-point coefficients")
        if kw:
            raise TypeError(f"gauss-point coefficients do not support extra kwargs {sorted(kw)}.")
        ke = 0.0
        for p, w in quadrature(formulation, "stiffness"):
            B, detJ = formulation.B_matrix(p, coords)
            N = formulation._cached_shape_and_derivs(p)[0]
            x = N @ coords
            ke = ke + (B.T @ _val(D, i, coords, x) @ B) * abs(detJ) * w * float(_val(th, i, coords, x))
        return ke
    D_e = resolve_centroid(D, i, coords)
    call = {"full": formulation.stiffness, "reduced": getattr(formulation, "reduced_stiffness", None),
            "hourglass_stabilized": getattr(formulation, "hourglass_stabilized_stiffness", None)}[method]
    if has_th:
        kw["thickness"] = float(resolve_centroid(th, i, coords))
    return call(coords, D_e, **kw)


def mass(formulation, coords, rho, kwargs, i, element_matrix, kind="mass"):
    """Element (consistent or lumped) mass when ``rho`` and/or ``thickness`` may be Coefficients.

    ``element_matrix(formulation, kind, coords, rho_value, kwargs)`` is the system's own evaluator."""
    kw = dict(kwargs)
    th = kw.get("thickness", 1.0)
    gauss = any(isinstance(c, Coefficient) and c.gauss for c in (rho, th))
    if gauss:
        if kind != "mass":
            raise NotImplementedError("gauss-point coefficients are available for the consistent mass only; "
                                      "use at='centroid' for assemble_lumped_mass.")
        check_supported(formulation, "Gauss-point coefficients")
        extra = set(kw) - {"thickness"}
        if extra:
            raise TypeError(f"gauss-point coefficients do not support extra kwargs {sorted(extra)}.")
        npn = formulation.dofs_per_node
        me = 0.0
        for p, w in quadrature(formulation, "mass"):
            N, dN = formulation._cached_shape_and_derivs(p)
            detJ = jacobian(dN, coords)[1]
            x = N @ coords
            r = np.asarray(_val(rho, i, coords, x), dtype=float)
            if r.ndim == 0:
                r = r * np.eye(npn)
            Nm = formulation.N_matrix(N)
            me = me + (Nm.T @ r @ Nm) * abs(detJ) * w * float(_val(th, i, coords, x))
        return me
    if "thickness" in kw:
        kw["thickness"] = float(resolve_centroid(th, i, coords))
    return element_matrix(formulation, kind, coords, resolve_centroid(rho, i, coords), kw)
