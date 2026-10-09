# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
facet_loads.py -- Wave 15 item 127 (docs/consolidated_future_roadmap.md,
source: TensorMesh's "Elements and Quadrature" documentation page's
facet_quadrature()/facet_shape_val()/nanson_scale()/FxW machinery): a
bounded, quadrature-based facet (edge for a 2-D element, face for a 3-D
solid) traction/pressure loader for the EXISTING element library.

WHAT THIS IS NOT (read this before reaching for it): NOT a generic
weak-form assembler -- Wave 14's own Meshes-page comparison already
found that gap (a facet-based Neumann/traction element is "inert
without the generic weak-form assembler abstraction fea_engine doesn't
have at all") and deliberately left it out of scope as a materially
bigger undertaking than anything else in either wave; this module does
NOT change that. What it DOES close is the narrower, concrete gap the
Elements-and-Quadrature comparison found on the QUADRATURE side of the
same issue: FESystem.add_nodal_force() (a flat even-split, no shape
functions at all) and FESystem.add_consistent_edge_load() (a
closed-form linear formula for straight 2-node edges only) have no way
to correctly load a CURVED/quadratic edge (Tri6/Quad8) or a 3-D element
FACE (Tet4/Tet10/Hex8/Hex20) with a properly shape-function-weighted,
Gauss-quadrature-integrated consistent nodal load. Same scalar-
traction-along-one-global-DOF convention add_consistent_edge_load()
already uses -- not a directional/pressure-normal-to-facet convention
(that would need a documented, consistent outward-normal orientation
across the whole element library, a bigger undertaking not attempted
here; a caller wanting an approximately-normal load can still call this
once per Cartesian direction with the traction component in that
direction).

THE MECHANICS, briefly: every facet (edge/face) of a solid/plane-stress
element in this package's library is itself EXACTLY one of the six
lower-dimensional element types this package ALREADY has real,
already-validated shape functions for -- line2/line3 (a facet's own
1-D restriction, written directly below since no standalone Line
element class exists in this package) for 2-D-element edges, and
Tri3PlaneStress/Tri6PlaneStress/Quad4PlaneStress/Quad8PlaneStress's own
shape_and_derivs() (reused AS-IS, not re-derived) for 3-D-element
faces. So "restrict shape_and_derivs() to a facet" reduces to: (1) know
which of the parent element's local node indices, in which order,
belong to that facet (EDGE_TABLES/FACE_TABLES below), (2) evaluate the
matching lower-dimensional element's OWN shape functions at facet-local
quadrature points (item 125's tri_quadrature()/gauss_product() for
2-D faces, gauss_legendre() for 1-D edges), (3) get the facet's own
physical length/area measure via jacobian_measure() (elements/base.py,
Wave 6's embedded-element fix -- ALREADY handles the "facet dim <
embedding dim" case a 1-D-edge-in-2-D or 2-D-face-in-3-D needs, exactly
the same formula Wave 6 found TrussTL2D/every shell formulation
needed). No new Jacobian math, no new shape-function math -- this
module is pure REUSE-and-restrict, wired together.

EDGE_TABLES/FACE_TABLES themselves are NOT hand-trusted -- every single
entry (which local nodes, in which order, belong to each facet) was
checked directly against the parent element's own shape_and_derivs()
before being written here: at several points spanning each facet, (a)
every node NOT listed for that facet has shape value ~0 there (checked
directly, "on this facet" isn't guessed), and (b) the listed nodes' own
shape values, evaluated via the facet's natural_coord_map into the
parent's full natural coordinates, reproduce the matching lower-
dimensional element's shape function values to machine precision (not
merely "close") -- see tests/test_facet_loads.py's own
TestFacetTablesAreCorrect class, which re-runs exactly this check as
part of the normal test suite, not just once by hand while writing this
module. This caught (and this file's own final natural_coord_map
lambdas reflect the fix for) three genuinely wrong first-guess Hex20
face parametrizations during development -- exactly the kind of subtle
node-ordering mistake this project's history (GMSH_NODE_ORDER,
meshio's node-order finding) has repeatedly found by checking directly
rather than trusting a hand derivation.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import math

import numpy as np

from .elements.base import gauss_legendre, gauss_product, tri_quadrature, jacobian_measure
from .elements.solids import (Tri3PlaneStress, Tri6PlaneStress,
                               Quad4PlaneStress, Quad8PlaneStress,
                               Tet4Solid3D, Tet10Solid3D,
                               Hex8Solid3D, Hex20Solid3D)


def _line2_shape(s):
    """2-node linear facet ('Line', TensorMesh's own naming) -- no
    standalone Line element class exists elsewhere in this package
    (truss/beam elements have a different, physics-specific dofs_per_
    node convention unsuitable for reuse here), so this is written
    directly, the same standard 1-D Lagrange formula every FEM text
    gives. s in [-1, 1]; N sums to 1 and dN sums to 0 at every s, the
    same partition-of-unity/consistency checks this package's other
    shape-function tests already run."""
    N = np.array([(1.0 - s) / 2.0, (1.0 + s) / 2.0])
    dN = np.array([-0.5, 0.5])
    return N, dN


def _line3_shape(s):
    """3-node quadratic facet (corner0, corner1, mid) -- the 1-D
    sibling of Tri6PlaneStress's own quadratic corner+mid convention,
    s in [-1, 1]."""
    N = np.array([s * (s - 1.0) / 2.0, s * (s + 1.0) / 2.0, 1.0 - s ** 2])
    dN = np.array([s - 0.5, s + 0.5, -2.0 * s])
    return N, dN


_FACET_SHAPE_FNS = {
    "line2": _line2_shape,
    "line3": _line3_shape,
    "tri3": lambda p: Tri3PlaneStress().shape_and_derivs(p),
    "tri6": lambda p: Tri6PlaneStress().shape_and_derivs(p),
    "quad4": lambda p: Quad4PlaneStress().shape_and_derivs(p),
    "quad8": lambda p: Quad8PlaneStress().shape_and_derivs(p),
}


# =====================================================================
# EDGE_TABLES -- 2-D parent elements. Each entry: (local_node_indices,
# family, natural_coord_map). natural_coord_map(s) -> (xi, eta) in the
# PARENT's own natural coordinates, used only by tests/test_facet_
# loads.py's verification pass (see this module's own docstring) --
# NOT needed at runtime, since consistent_facet_load_shares() below
# operates entirely in the facet's own local family coordinates.
# =====================================================================
EDGE_TABLES = {
    Tri3PlaneStress: [
        ((0, 1), "line2", lambda s: ((1 + s) / 2, 0.0)),
        ((1, 2), "line2", lambda s: ((1 - s) / 2, (1 + s) / 2)),
        ((2, 0), "line2", lambda s: (0.0, (1 - s) / 2)),
    ],
    Tri6PlaneStress: [
        ((0, 1, 3), "line3", lambda s: ((1 + s) / 2, 0.0)),
        ((1, 2, 4), "line3", lambda s: ((1 - s) / 2, (1 + s) / 2)),
        ((2, 0, 5), "line3", lambda s: (0.0, (1 - s) / 2)),
    ],
    Quad4PlaneStress: [
        ((0, 1), "line2", lambda s: (s, -1.0)),
        ((1, 2), "line2", lambda s: (1.0, s)),
        ((2, 3), "line2", lambda s: (-s, 1.0)),
        ((3, 0), "line2", lambda s: (-1.0, -s)),
    ],
    Quad8PlaneStress: [
        ((0, 1, 4), "line3", lambda s: (s, -1.0)),
        ((1, 2, 5), "line3", lambda s: (1.0, s)),
        ((2, 3, 6), "line3", lambda s: (-s, 1.0)),
        ((3, 0, 7), "line3", lambda s: (-1.0, -s)),
    ],
}

# =====================================================================
# FACE_TABLES -- 3-D solid parent elements. Tet4/Tet10's four faces are
# the natural tetrahedron's own coordinate-plane faces (t=0, s=0, r=0,
# r+s+t=1); Hex8/Hex20's six faces reuse mesh.py's own already-
# validated _TET4_FACES/_HEX8_FACES corner ordering (Wave 0 item 8),
# extended with each element's own mid-edge nodes for the quadratic
# case (Tet10Solid3D._edges, Hex20Solid3D's own documented 8-19
# bottom/top/vertical mid-edge grouping).
# =====================================================================
FACE_TABLES = {
    Tet4Solid3D: [
        ((0, 1, 2), "tri3", lambda a, b: (a, b, 0.0)),
        ((0, 1, 3), "tri3", lambda a, b: (a, 0.0, b)),
        ((0, 2, 3), "tri3", lambda a, b: (0.0, a, b)),
        ((1, 2, 3), "tri3", lambda a, b: (1 - a - b, a, b)),
    ],
    Tet10Solid3D: [
        ((0, 1, 2, 4, 5, 6), "tri6", lambda a, b: (a, b, 0.0)),
        ((0, 1, 3, 4, 8, 7), "tri6", lambda a, b: (a, 0.0, b)),
        ((0, 2, 3, 6, 9, 7), "tri6", lambda a, b: (0.0, a, b)),
        ((1, 2, 3, 5, 9, 8), "tri6", lambda a, b: (1 - a - b, a, b)),
    ],
    Hex8Solid3D: [
        ((0, 3, 2, 1), "quad4", lambda a, b: (b, a, -1.0)),
        ((4, 5, 6, 7), "quad4", lambda a, b: (a, b, 1.0)),
        ((0, 1, 5, 4), "quad4", lambda a, b: (a, -1.0, b)),
        ((1, 2, 6, 5), "quad4", lambda a, b: (1.0, a, b)),
        ((2, 3, 7, 6), "quad4", lambda a, b: (-a, 1.0, b)),
        ((3, 0, 4, 7), "quad4", lambda a, b: (-1.0, -a, b)),
    ],
    Hex20Solid3D: [
        ((0, 3, 2, 1, 11, 10, 9, 8), "quad8", lambda a, b: (b, a, -1.0)),
        ((4, 5, 6, 7, 12, 13, 14, 15), "quad8", lambda a, b: (a, b, 1.0)),
        ((0, 1, 5, 4, 8, 17, 12, 16), "quad8", lambda a, b: (a, -1.0, b)),
        ((1, 2, 6, 5, 9, 18, 13, 17), "quad8", lambda a, b: (1.0, a, b)),
        ((2, 3, 7, 6, 10, 19, 14, 18), "quad8", lambda a, b: (-a, 1.0, b)),
        ((3, 0, 4, 7, 11, 16, 15, 19), "quad8", lambda a, b: (-1.0, -a, b)),
    ],
}


def list_facets(formulation):
    """Returns the list of (local_node_indices, family, natural_coord_
    map) triples for every facet (edge for a 2-D parent, face for a
    3-D solid parent) of `formulation`'s own type -- the lookup a
    caller uses to turn one mesh element's own connectivity row into
    the correctly-ordered GLOBAL node-id tuples add_consistent_facet_
    load()/consistent_facet_load_shares() expect: `conn[list(idx)]`
    for each `(idx, family, _)` entry below.

    Raises ValueError for any element type with no facet table --
    currently Tri3PlaneStress/Tri6PlaneStress/Quad4PlaneStress/
    Quad8PlaneStress (edges) and Tet4Solid3D/Tet10Solid3D/
    Hex8Solid3D/Hex20Solid3D (faces); see this module's own docstring
    for scope."""
    table = EDGE_TABLES if formulation.dim == 2 else FACE_TABLES
    cls = type(formulation)
    if cls not in table:
        raise ValueError(
            f"facet_loads: no facet table for {cls.__name__} -- see this "
            f"module's own docstring for the currently-supported element "
            f"list (plane-stress triangles/quads for edges, tet/hex "
            f"solids for faces).")
    return table[cls]


def facet_family(parent_dim, n_facet_nodes):
    """Resolves a facet's own shape-function family from (parent_dim,
    facet-node count) -- e.g. (2, 3) is unambiguous as line3 (a 2-D
    parent's quadratic edge) even though 3 nodes ALSO describes tri3
    (a 3-D parent's linear face); parent_dim disambiguates."""
    table = {(2, 2): "line2", (2, 3): "line3",
             (3, 3): "tri3", (3, 6): "tri6",
             (3, 4): "quad4", (3, 8): "quad8"}
    key = (parent_dim, n_facet_nodes)
    if key not in table:
        raise ValueError(
            f"facet_loads.facet_family: no known facet family for "
            f"parent_dim={parent_dim}, {n_facet_nodes} facet nodes.")
    return table[key]


def facet_quadrature(family, order=2):
    """Returns (points, weights) on the facet's own natural domain,
    weights already scaled to the SAME 'final, ready to multiply
    directly by the physical measure' convention gauss_product()/
    tri_quadrature()'s own callers use (see mesh_transform.py's
    _reference_quadrature() for the identical pattern one level up) --
    i.e. NOT barycentric-normalized for the tri family; the 0.5 area
    factor is applied here, once, so every caller of this function
    never has to know which family produced these weights.

    order: exact polynomial degree to integrate (item 125's own
    order-parameterized tri_quadrature()/the plain n=ceil((order+1)/2)
    point-count rule for line/quad families) -- default 2, matching
    this package's other default-2 fixed rules (a uniform traction is
    itself degree-0, so order=2 is already generous for that case;
    pass a higher order for a spatially-varying, higher-degree
    traction field, the same reasoning item 125's own quad_order=
    parameter documents)."""
    if family in ("line2", "line3"):
        n = max(1, math.ceil((order + 1) / 2))
        pts, wts = gauss_legendre(n)
        return list(pts), list(wts)
    if family in ("tri3", "tri6"):
        pts, wts = tri_quadrature(order)
        return pts, [w * 0.5 for w in wts]
    if family in ("quad4", "quad8"):
        n = max(1, math.ceil((order + 1) / 2))
        return gauss_product(n, 2)
    raise ValueError(f"facet_loads.facet_quadrature: unknown family {family!r}")


def consistent_facet_load_shares(parent_dim, facet_coords, traction, quad_order=2):
    """Returns an (n_facet_nodes,) array of consistent nodal load
    'shares' for a uniform scalar traction/pressure MAGNITUDE over one
    facet, via real Gauss quadrature restricted to that facet -- the
    generalization of FESystem.add_consistent_edge_load()'s own
    length-weighted 2-node formula to curved/quadratic edges and 3-D
    element faces (see this module's own docstring for the mechanics).

    parent_dim: 2 (this facet is an EDGE of a 2-D element) or 3 (this
    facet is a FACE of a 3-D solid).
    facet_coords: (n_facet_nodes, parent_dim) physical coordinates of
    ONLY this facet's own nodes, in the corner-then-midside order
    list_facets() documents (2 or 3 nodes for an edge; 3, 4, 6, or 8
    for a face) -- NOT the parent element's full node array; a facet
    call only ever needs its own nodes' coordinates, since the facet's
    own shape functions (line2/line3/tri3/tri6/quad4/quad8) don't
    reference any other node.
    traction: scalar magnitude (matches add_consistent_edge_load()'s
    own convention).
    quad_order: see facet_quadrature()'s own docstring.

    Every quadrature point's contribution uses jacobian_measure()
    (elements/base.py) rather than plain jacobian() -- a facet is
    ALWAYS embedded (facet dim < parent_dim: 1-D-in-2-D for an edge,
    2-D-in-3-D for a face), the exact case jacobian_measure()'s own
    Gram-determinant branch exists for (Wave 6's TrussTL2D/shell mass-
    matrix fix), so this measure is always non-negative by
    construction -- no abs()/orientation convention to get wrong, the
    way the simplex-family volumetric case (Wave 15 item 126's own
    module docstring note) needed one."""
    facet_coords = np.asarray(facet_coords, dtype=float)
    n_facet = facet_coords.shape[0]
    family = facet_family(parent_dim, n_facet)
    shape_fn = _FACET_SHAPE_FNS[family]
    pts, wts = facet_quadrature(family, quad_order)

    share = np.zeros(n_facet)
    for p, w in zip(pts, wts):
        N, dN = shape_fn(p)
        dN = np.atleast_2d(dN)   # line family returns a 1-D dN -> (1, n_facet)
        measure = jacobian_measure(dN, facet_coords)
        share += N * traction * measure * w
    return share
