# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
mesh.py -- Module 2: geometric mesh generation and boundary-node
selection.

Structured/mapped mesh generators for simple 1-D, 2-D, and 3-D
geometries, wrapped in a small Mesh container, plus boundary-node
selector helpers. This is a direct port of the standalone
mesh_generation.py module into the package, with two additions:
    - line_mesh() for 1-D beam elements
    - Mesh is now a dataclass with convenience methods, and
      check_quality() is generic: it takes any Element (from
      element.py) and asks IT for shape derivatives, so mesh.py never
      needs to know about a specific element's shape functions.

To add a new geometry (e.g. a cylinder, an ellipse-with-notch, ...):
write one function that returns a Mesh -- nothing else here changes.

MultiBlockMesh (Module 14: mixed-element-type assembly) is a SEPARATE
container, not a variant of Mesh, for meshes where more than one
element TOPOLOGY coexists (e.g. triangles and quads in the same 2-D
region -- the case a Gmsh "recombine" pass typically produces: quads
where the algorithm could pair triangles up, plain leftover triangles
where it couldn't). Mesh.elements is a single rectangular
(n_elements, nodes_per_element) array -- correct for a uniform mesh,
but structurally unable to hold rows of different width, which is why
this is a new class rather than a generalization of Mesh itself. Every
existing generator (line_mesh, rectangle_mesh, box_mesh, ...) and every
existing script that constructs a Mesh directly or reads .elements
continues to work completely unchanged -- MultiBlockMesh is something
you opt into (built by geometry_engine.py when Gmsh actually returns
more than one element type, see that module), not something imposed on
the single-type case.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree

try:
    from .grading import graded_partition   # used as `import fea_engine`
except ImportError:
    from grading import graded_partition     # used as a flat script directory


# =====================================================================
# 3-D face-adjacency tables, Wave 0 item 8 (docs/consolidated_future_
# roadmap.md, source generalized_mesh_grading_roadmap.md): the
# per-topology face table Mesh.check_grading() previously raised
# NotImplementedError for want of. Two topologies, exactly what the
# roadmap named -- Tet4's 4 triangular faces and Hex8's 6 quadrilateral
# faces -- expressed as CORNER-node-index tuples into one element's own
# connectivity row, matched as frozensets (adjacency only needs the
# SET of shared corner nodes, not winding order).
#
# Reused as-is for the 10-node Tet10 and 20-node Hex20 (see
# check_grading()'s dispatch by node count below): both formulations
# put their corner nodes FIRST, mid-edge/mid-face nodes after (an
# established, documented convention throughout elements/solids.py --
# see e.g. Tet10Solid3D's own GMSH_NODE_ORDER docstring), so slicing
# off the first 4 or 8 entries of a 10- or 20-node connectivity row
# recovers exactly the corner sub-tetrahedron/sub-hexahedron this table
# already knows how to handle -- no separate quadratic-element face
# table needed.
_TET4_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

# Outward-oriented (for _hex_volume()'s sign, though that function
# takes abs() regardless -- kept correctly wound anyway since it's
# cheap and self-documenting) quadrilateral faces for the Hex8 corner
# convention used throughout this package (elements/solids.py
# Hex8Solid3D.shape_and_derivs(): nodes 0-3 the zeta=-1 face going
# around counterclockwise as seen from OUTSIDE (i.e. from -zeta), 4-7
# the zeta=+1 face directly above 0-3 respectively). Verified
# numerically against the reference unit cube (see this module's git
# history / tests/test_grading_3d.py) to reproduce volume=8 for the
# [-1,1]^3 reference element exactly.
_HEX8_FACES = [
    (0, 3, 2, 1),   # bottom, zeta=-1
    (4, 5, 6, 7),   # top, zeta=+1
    (0, 1, 5, 4),   # front, eta=-1
    (1, 2, 6, 5),   # right, xi=+1
    (2, 3, 7, 6),   # back, eta=+1
    (3, 0, 4, 7),   # left, xi=-1
]


def _tet_volume(pts):
    """Exact closed-form volume of a straight-sided tetrahedron from
    its 4 corner points (scalar triple product / 6) -- unlike a hex, a
    tet's 4 triangular faces are always planar, so there is no
    decomposition ambiguity to worry about the way _hex_volume() below
    has to. abs() because this is used only as a SIZE metric (Wave 0
    item 8's grading diagnostic); orientation/inversion is already
    Mesh.check_quality()'s job, not this one's."""
    p0, p1, p2, p3 = pts
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def _hex_volume(pts):
    """Volume of a (possibly non-parallelepiped, straight-edged)
    hexahedron from its 8 corner points, via the divergence theorem
    applied to its triangulated boundary (each of the 6 quad faces
    split into 2 triangles): V = (1/6) * sum_faces sum_tris
    dot(a, cross(b, c)) for triangle vertices (a, b, c) in outward-
    oriented order. This is exact for any straight-edged hexahedron
    with PLANAR faces (the standard assumption for a structured/mapped
    FE mesh element -- the same assumption Mesh.check_quality()'s own
    positive-Jacobian check is already screening for elsewhere), and
    -- unlike picking one fixed tet decomposition (e.g. always
    splitting from corner 0) -- doesn't silently mis-handle a
    hexahedron shaped such that a naive decomposition's tets would
    overlap or leave a gap; the divergence-theorem sum has no such
    failure mode for a genuinely closed, consistently-oriented
    boundary. Only used as a SIZE metric here (see _tet_volume()'s
    docstring for the same abs() reasoning)."""
    total = 0.0
    for quad in _HEX8_FACES:
        a, b, c, d = (pts[i] for i in quad)
        total += np.dot(a, np.cross(b, c))
        total += np.dot(a, np.cross(c, d))
    return abs(total) / 6.0


def _corner_count_for_face_table(n_nodes):
    """4/10-node elements are tet-family (4 corners), 8/20-node
    elements are hex-family (8 corners) -- see _TET4_FACES/_HEX8_FACES'
    module docstring for why the higher-order counts reuse the same
    corner-only tables. Returns None (rather than raising directly) so
    callers can produce ONE consistent, specific error message whether
    the failure happens while building faces or while sizing an
    element."""
    if n_nodes in (4, 10):
        return 4
    if n_nodes in (8, 20):
        return 8
    return None


def _unsupported_topology_error(n_nodes):
    return NotImplementedError(
        f"check_grading(): no 3-D face table for a {n_nodes}-node "
        "element -- supported node counts are 4 or 10 (Tet4/Tet10, "
        "tetrahedral faces) and 8 or 20 (Hex8/Hex20, hexahedral "
        "faces). See docs/generalized_mesh_grading_roadmap.md.")


def _elem_faces_3d(elem):
    """Corner-node face signatures (as frozensets, for adjacency
    matching) for one element's connectivity row."""
    n_corners = _corner_count_for_face_table(len(elem))
    if n_corners is None:
        raise _unsupported_topology_error(len(elem))
    corners = elem[:n_corners]
    faces = _TET4_FACES if n_corners == 4 else _HEX8_FACES
    return [frozenset(int(corners[i]) for i in f) for f in faces]


def _elem_size_3d(elem, nodes):
    """Characteristic length scale: volume**(1/3), the 3-D analogue of
    check_grading()'s existing 2-D sqrt(area) -- both give a single
    length-dimensioned number per element, comparable across elements
    of different shapes, for the neighbor-to-neighbor SIZE ratio this
    method reports (not an exact edge length, just a scale proxy, the
    same spirit as the 2-D sqrt(area) it mirrors)."""
    n_corners = _corner_count_for_face_table(len(elem))
    if n_corners is None:
        raise _unsupported_topology_error(len(elem))
    pts = nodes[elem[:n_corners]]
    vol = _tet_volume(pts) if n_corners == 4 else _hex_volume(pts)
    return vol ** (1.0 / 3.0)


def _max_neighbor_ratio(sizes, adjacency_to_elems, growth_ratio_cap):
    """Shared by check_grading()'s 2-D (edge-adjacency) and 3-D
    (face-adjacency) branches: given each element's scalar size and a
    map from a shared-boundary signature (edge or face) to the list of
    elements touching it, find the worst neighbor-to-neighbor size
    ratio and (if a cap was given) how many boundaries exceed it.
    Factored out so the two branches can't silently drift apart on
    this shared bookkeeping the way two independent copies risk
    doing."""
    max_ratio = 1.0
    n_over = 0
    for elist in adjacency_to_elems.values():
        if len(elist) != 2:
            continue   # exterior boundary -- only one adjacent element
        s1, s2 = sizes[elist[0]], sizes[elist[1]]
        r = max(s1, s2) / min(s1, s2)
        max_ratio = max(max_ratio, r)
        if growth_ratio_cap is not None and r > growth_ratio_cap:
            n_over += 1
    return max_ratio, n_over


class _NodeSelectionMixin:
    """Named node sets and coordinate-based selection shared by Mesh and MultiBlockMesh (v1.0.1).

    A node set is a name for a group of node ids, stored in `self.node_sets`. Once registered, the
    name can be used anywhere a node selection is accepted, e.g.
        mesh.select_nodes(x=0.0, name="root")           # or mesh.add_node_set("root", ids)
        system.fix_dofs("root", ["ux", "uy"])
        U.component("uy", nodes="tip")
    The pattern follows PyMAPDL components / PyDPF scopings: say WHAT is selected once, reuse it."""

    def add_node_set(self, name, node_ids):
        """Register `node_ids` (non-empty, within the mesh) under `name`. Returns self (chainable).
        An existing name is replaced."""
        if not isinstance(name, str) or not name:
            raise ValueError("add_node_set: the set name must be a non-empty string.")
        ids = np.asarray(node_ids).reshape(-1)
        if ids.size == 0:
            raise ValueError(f"add_node_set({name!r}): no nodes given -- the selection matched nothing "
                             f"(check the coordinate value / tolerance).")
        if ids.dtype.kind not in "iu":
            if ids.dtype.kind == "f" and np.all(ids == np.round(ids)):
                ids = ids.astype(int)
            else:
                raise ValueError(f"add_node_set({name!r}): node ids must be integers.")
        ids = np.unique(ids.astype(int))
        if ids[0] < 0 or ids[-1] >= len(self.nodes):
            raise ValueError(f"add_node_set({name!r}): node id outside the mesh "
                             f"(valid range 0..{len(self.nodes) - 1}).")
        self.node_sets[name] = ids
        return self

    def node_set(self, name):
        """The sorted node-id array registered under `name` (ValueError listing the available
        names if there is no such set)."""
        try:
            return self.node_sets[name]
        except KeyError:
            raise ValueError(f"no node set named {name!r}; available: {sorted(self.node_sets)}") from None

    def select_nodes(self, x=None, y=None, z=None, tol=1e-9, name=None):
        """Node ids whose coordinates satisfy ALL given conditions. Each of x, y, z is either a
        value (|coord - value| < tol) or a (lo, hi) pair (lo - tol <= coord <= hi + tol); axes left
        as None are unconstrained. With `name`, the result is also registered as a node set (an
        empty selection then raises). Without `name`, an empty array is returned, as nodes_on_line does.

            mesh.select_nodes(x=0.0)                       # a line / plane
            mesh.select_nodes(x=(0.0, 0.1), y=0.2)         # a segment
            mesh.select_nodes(y=0.0, name="bottom")
        """
        dim = self.nodes.shape[1]
        mask = np.ones(len(self.nodes), dtype=bool)
        for axis, spec in enumerate((x, y, z)):
            if spec is None:
                continue
            if axis >= dim:
                raise ValueError(f"select_nodes: '{'xyz'[axis]}' given but the mesh has only {dim} "
                                 f"coordinate(s).")
            col = self.nodes[:, axis]
            if np.ndim(spec) == 0:
                mask &= np.abs(col - spec) < tol
            else:
                lo, hi = spec
                if lo > hi:
                    raise ValueError(f"select_nodes: {'xyz'[axis]}=({lo}, {hi}) has lo > hi.")
                mask &= (col >= lo - tol) & (col <= hi + tol)
        ids = np.where(mask)[0]
        if name is not None:
            self.add_node_set(name, ids)
        return ids


@dataclass
class Mesh(_NodeSelectionMixin):
    nodes: np.ndarray       # (n_nodes, dim)
    elements: np.ndarray    # (n_elements, nodes_per_element), int
    dim: int                # 1, 2, or 3

    # Wave 14 item 121 (docs/consolidated_future_roadmap.md), scoped
    # directly against TensorMesh's own Meshes documentation page
    # (https://docs.tensor-mesh.com/user_guide/meshes.html): per-node
    # (`point_data`) and per-element (`cell_data`) field attachment
    # directly on the mesh object, plus mesh-global metadata
    # (`field_data`) -- so a solution field (today only ever returned
    # as a SEPARATE array from solve_static()/solve_modal()/etc.) can
    # OPTIONALLY be attached back onto the mesh itself, the way
    # TensorMesh's register_point_data()/register_element_data() work.
    # Purely additive: every pre-existing caller that only ever touches
    # .nodes/.elements/.dim (the entire codebase, before this item) is
    # completely unaffected -- these three fields default to empty
    # dicts, and nothing downstream (FESystem, solve_*, check_quality,
    # ...) reads them. `field(default_factory=dict)` (not `= {}`) is
    # required here, not a style choice -- a plain mutable default on a
    # dataclass field is shared across every instance that doesn't
    # override it, which would silently leak point_data/cell_data
    # between unrelated Mesh objects.
    point_data: dict = field(default_factory=dict)
    cell_data: dict = field(default_factory=dict)
    field_data: dict = field(default_factory=dict)
    node_sets: dict = field(default_factory=dict)    # {name: sorted node-id array}, see add_node_set()

    def register_point_data(self, name, array):
        """Attach a per-node field (any array whose first axis has
        length == len(self.nodes)) under `name` in self.point_data,
        mirroring TensorMesh's Mesh.register_point_data(). Returns
        self (chainable), matching that same convention, so a caller
        can build up a mesh-with-results via
        `mesh.register_point_data("u", u).register_point_data("v", v)`
        before saving/plotting it."""
        array = np.asarray(array)
        if array.shape[0] != len(self.nodes):
            raise ValueError(
                f"register_point_data({name!r}): array's first axis has length "
                f"{array.shape[0]}, expected {len(self.nodes)} (one entry per node)")
        self.point_data[name] = array
        return self

    def register_element_data(self, name, array):
        """Attach a per-element field (any array whose first axis has
        length == len(self.elements)) under `name` in self.cell_data,
        mirroring TensorMesh's Mesh.register_element_data(). Returns
        self (chainable), same reasoning as register_point_data()
        above."""
        array = np.asarray(array)
        if array.shape[0] != len(self.elements):
            raise ValueError(
                f"register_element_data({name!r}): array's first axis has length "
                f"{array.shape[0]}, expected {len(self.elements)} (one entry per element)")
        self.cell_data[name] = array
        return self

    def nodes_on_line(self, axis, value, tol=1e-9):
        """1-D/2-D selector: axis in {0:x, 1:y}."""
        return np.where(np.abs(self.nodes[:, axis] - value) < tol)[0]

    def nodes_on_plane(self, axis, value, tol=1e-9):
        """3-D selector: axis in {0:x, 1:y, 2:z}."""
        return np.where(np.abs(self.nodes[:, axis] - value) < tol)[0]

    def compute_boundary_mask(self, store_as="is_boundary"):
        """Wave 14 item 123 (docs/consolidated_future_roadmap.md):
        topology-derived boundary-node detection, complementing
        nodes_on_line()/nodes_on_plane() above -- those only work when
        the boundary of interest happens to be axis/plane-aligned (a
        straight coordinate value); this works for ANY mesh shape,
        including curved or otherwise non-axis-aligned domains (a
        Gmsh-authored fillet/hole boundary, an off-center hole, ...),
        the same "is_boundary" concept TensorMesh's own generators
        precompute (Meshes page, "Boundary identification").

        Mechanism: an edge (2-D) or face (3-D) shared by exactly ONE
        element is, by construction, a boundary edge/face -- the exact
        same edge-/face-adjacency bookkeeping check_grading() already
        builds (see _elem_faces_3d()/the 2-D edge loop inside
        check_grading() above) repurposed here for a different
        question (which NODES touch an unshared boundary, not the
        worst neighbor-size ratio). A node is flagged boundary if it
        appears in at least one boundary edge/face.

        Returns a bool array of shape (len(self.nodes),), and ALSO
        stores it in self.point_data[store_as] (default "is_boundary",
        matching TensorMesh's own key) unless store_as is None.

        1-D meshes have no edge/face-adjacency notion (same limitation
        check_grading() already documents) -- for dim=1, every node
        with only one incident line-element side is boundary, handled
        as a direct special case below rather than forcing dim=1
        through the 2-D/3-D adjacency machinery it doesn't have."""
        n_nodes = len(self.nodes)
        mask = np.zeros(n_nodes, dtype=bool)

        if self.dim == 1:
            # A 1-D line mesh's own node ordering already tells us the
            # two ends directly: any node used by only one element
            # (i.e. not an interior node shared by two consecutive
            # line elements) is boundary.
            counts = np.zeros(n_nodes, dtype=int)
            for elem in self.elements:
                for n in elem:
                    counts[n] += 1
            mask[counts == 1] = True
        elif self.dim == 2:
            adjacency_to_elems = {}
            for elem in self.elements:
                n = len(elem)
                for k in range(n):
                    edge = frozenset((int(elem[k]), int(elem[(k + 1) % n])))
                    adjacency_to_elems.setdefault(edge, []).append(True)
            for edge, elist in adjacency_to_elems.items():
                if len(elist) == 1:
                    for node_id in edge:
                        mask[node_id] = True
        elif self.dim == 3:
            adjacency_to_elems = {}
            for elem in self.elements:
                for face in _elem_faces_3d(elem):
                    adjacency_to_elems.setdefault(face, []).append(True)
            for face, elist in adjacency_to_elems.items():
                if len(elist) == 1:
                    for node_id in face:
                        mask[node_id] = True
        else:
            raise NotImplementedError(
                f"compute_boundary_mask(): not meaningful for a {self.dim}-D mesh "
                "-- supported dimensions are 1, 2, and 3.")

        if store_as is not None:
            self.point_data[store_as] = mask
        return mask

    def check_quality(self, elem_formulation, verbose=True):
        """Evaluate the Jacobian determinant at every Gauss point of every
        element using elem_formulation's OWN shape functions and Gauss
        order (see element.Element) -- generic across any element type,
        no per-shape special-casing needed here."""
        try:
            from .elements import gauss_product   # used as `import fea_engine`
        except ImportError:
            from elements import gauss_product     # used as a flat script directory
        pts, wts = gauss_product(elem_formulation.gauss_order, elem_formulation.dim)
        detJ_all = []
        n_bad = 0
        for elem in self.elements:
            elem_coords = self.nodes[elem]
            bad = False
            for p in pts:
                _, dN = elem_formulation.shape_and_derivs(p)
                J = dN @ elem_coords
                detJ = np.linalg.det(J) if J.shape[0] > 1 else J[0, 0]
                detJ_all.append(detJ)
                if detJ <= 0:
                    bad = True
            if bad:
                n_bad += 1
        detJ_all = np.array(detJ_all)
        ratio = np.max(np.abs(detJ_all)) / np.min(np.abs(detJ_all))
        if verbose:
            print(f"  {len(self.elements)} elements, {len(self.nodes)} nodes, dim={self.dim}")
            print(f"  detJ range: [{detJ_all.min():.3e}, {detJ_all.max():.3e}]  "
                  f"(max/min |detJ| ratio: {ratio:.1f})")
            print("  OK: all Gauss-point Jacobians positive (no inverted elements)"
                  if n_bad == 0 else
                  f"  *** {n_bad} element(s) with non-positive detJ -- INVERTED/DEGENERATE ***")
        return n_bad == 0

    def check_grading(self, growth_ratio_cap=None, verbose=True):
        """Realized neighbor-to-neighbor element-size-ratio diagnostic
        (docs/generalized_mesh_grading_roadmap.md, Section 4's "Quality"
        validation item) -- complements check_quality()'s detJ-based
        inversion/degeneracy check with the OTHER thing mesh grading can
        get wrong: every element can be individually well-shaped
        (check_quality() green) and the mesh can still jump abruptly in
        size between neighbors, which check_quality() alone never flags.

        Builds an edge-adjacency map (any polygon, since edges are just
        consecutive node pairs -- works for Tri3 and Quad4/Quad8
        uniformly) and reports the max ratio between sqrt(area)-based
        sizes of any two elements sharing an edge.

        `growth_ratio_cap` given -> returns True/False (pass/fail, like
        check_quality()) and prints how many edges exceed it.
        `growth_ratio_cap=None` (default) -> returns the realized max
        ratio itself, for callers that want the number rather than a
        verdict.

        Wave 0 item 8 (docs/consolidated_future_roadmap.md, source
        generalized_mesh_grading_roadmap.md): extended to 3-D via a
        per-topology face-adjacency table (Tet4/Tet10's 4 triangular
        faces, Hex8/Hex20's 6 quadrilateral faces -- see the
        module-level _TET4_FACES/_HEX8_FACES tables and their
        docstrings). The 3-D "size" is volume**(1/3) (see
        _elem_size_3d()), the natural length-dimensioned analogue of
        the 2-D branch's sqrt(area) below; both feed the SAME shared
        worst-ratio bookkeeping (_max_neighbor_ratio()) so the two
        branches can't drift apart on that logic. Any 3-D element
        topology outside Tet4/Tet10/Hex8/Hex20 (e.g. a wedge/prism, not
        currently in this package's element library at all) still
        raises NotImplementedError rather than guessing.

        1-D meshes aren't meaningfully "graded" in this sense (a line
        mesh's neighbor structure is just its own 1-D ordering, not an
        edge/face adjacency problem) and also raise NotImplementedError
        here."""
        if self.dim == 2:
            def _elem_size_2d(elem):
                x, y = self.nodes[elem, 0], self.nodes[elem, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
                return np.sqrt(area)

            sizes = np.array([_elem_size_2d(e) for e in self.elements])
            adjacency_to_elems = {}
            for ei, elem in enumerate(self.elements):
                n = len(elem)
                for k in range(n):
                    edge = frozenset((int(elem[k]), int(elem[(k + 1) % n])))
                    adjacency_to_elems.setdefault(edge, []).append(ei)
            unit = "edge"
        elif self.dim == 3:
            sizes = np.array([_elem_size_3d(e, self.nodes) for e in self.elements])
            adjacency_to_elems = {}
            for ei, elem in enumerate(self.elements):
                for face in _elem_faces_3d(elem):
                    adjacency_to_elems.setdefault(face, []).append(ei)
            unit = "face"
        else:
            raise NotImplementedError(
                f"check_grading(): not meaningful for a {self.dim}-D mesh "
                "-- supported dimensions are 2 and 3.")

        max_ratio, n_over = _max_neighbor_ratio(sizes, adjacency_to_elems, growth_ratio_cap)

        if verbose:
            msg = f"  max neighbor-to-neighbor size ratio: {max_ratio:.3f}"
            if growth_ratio_cap is not None:
                msg += f"  ({n_over} {unit}(s) exceed cap {growth_ratio_cap})"
            print(msg)

        return (n_over == 0) if growth_ratio_cap is not None else max_ratio


@dataclass
class MultiBlockMesh(_NodeSelectionMixin):
    """A mesh with MORE THAN ONE element topology sharing one node
    array. `blocks` maps a block name (any hashable key -- Module 14's
    own producers use short topology names like "tri3"/"quad4"/"tet4",
    deliberately NOT tied to any specific element.py class, since the
    mesh doesn't know or care which Element formulation you'll pair
    with each block -- see solver.FESystem) to that block's own
    (n_elements_in_block, nodes_per_element) connectivity array, all
    indexing into the SAME shared `nodes`.

    Constraint: every block that will be assembled together in one
    FESystem must use an element formulation with the SAME
    dofs_per_node. Global DOFs are numbered per NODE (see
    FESystem._global_dofs()), not per block, so a node shared between
    a 2-dof/node block and a 3-dof/node block would need two different
    numbers of DOFs at once -- not supported (see FESystem's
    docstring). This is not a restriction on TOPOLOGY (tri3 and quad4
    are different topologies with the same dofs_per_node=2, which is
    exactly the case this class exists for) -- it rules out mixing,
    say, a plane-stress block with a plate-bending block in one mesh,
    which is a fundamentally different (multi-field) problem.
    """
    nodes: np.ndarray       # (n_nodes, dim)
    blocks: dict             # {block_name: (n_elements_in_block, nodes_per_element) int array}
    dim: int                 # 1, 2, or 3

    # Wave 14 item 121 -- same field-attachment idea as Mesh's own
    # point_data/cell_data/field_data above, adapted to this class's
    # per-block connectivity: point_data is shared across the whole
    # mesh (one node array), but cell_data is necessarily nested by
    # block name first (matching `blocks` itself), since two different
    # blocks can have different element counts.
    point_data: dict = field(default_factory=dict)
    cell_data: dict = field(default_factory=dict)   # {block_name: {field_name: array}}
    field_data: dict = field(default_factory=dict)
    node_sets: dict = field(default_factory=dict)    # {name: sorted node-id array}, see add_node_set()

    def register_point_data(self, name, array):
        """Same contract as Mesh.register_point_data() -- see that
        method's own docstring."""
        array = np.asarray(array)
        if array.shape[0] != len(self.nodes):
            raise ValueError(
                f"register_point_data({name!r}): array's first axis has length "
                f"{array.shape[0]}, expected {len(self.nodes)} (one entry per node)")
        self.point_data[name] = array
        return self

    def register_element_data(self, block_name, name, array):
        """Per-block analogue of Mesh.register_element_data() -- the
        block whose field is being attached must be named explicitly
        (block_name), since MultiBlockMesh has no single "the
        elements" array to infer a length from the way Mesh does."""
        if block_name not in self.blocks:
            raise KeyError(
                f"register_element_data: no block named {block_name!r} "
                f"(blocks: {sorted(self.blocks)})")
        array = np.asarray(array)
        n_block = len(self.blocks[block_name])
        if array.shape[0] != n_block:
            raise ValueError(
                f"register_element_data({block_name!r}, {name!r}): array's first "
                f"axis has length {array.shape[0]}, expected {n_block} (one entry "
                f"per element in block {block_name!r})")
        self.cell_data.setdefault(block_name, {})[name] = array
        return self

    def nodes_on_line(self, axis, value, tol=1e-9):
        return np.where(np.abs(self.nodes[:, axis] - value) < tol)[0]

    def nodes_on_plane(self, axis, value, tol=1e-9):
        return np.where(np.abs(self.nodes[:, axis] - value) < tol)[0]

    def compute_boundary_mask(self, store_as="is_boundary"):
        """MultiBlockMesh analogue of Mesh.compute_boundary_mask() --
        same edge-/face-adjacency mechanism (an edge/face shared by
        exactly one element, across ALL blocks combined into one
        shared adjacency map, since two elements from DIFFERENT blocks
        can legitimately share a boundary -- e.g. a Gmsh "recombine"
        mesh's leftover triangles bordering its paired quads), but
        `_elem_faces_3d()`'s own corner-count dispatch already handles
        mixed element node-counts per call, so no extra per-block
        branching is needed here beyond iterating every block's own
        connectivity into the SAME adjacency dict before checking
        which edges/faces are unshared. See Mesh.compute_boundary_mask()
        for the full mechanism/rationale; this is the same logic,
        just iterating over every block's connectivity rather than one
        flat .elements array."""
        n_nodes = len(self.nodes)
        mask = np.zeros(n_nodes, dtype=bool)

        if self.dim == 1:
            counts = np.zeros(n_nodes, dtype=int)
            for conn in self.blocks.values():
                for elem in conn:
                    for n in elem:
                        counts[n] += 1
            mask[counts == 1] = True
        elif self.dim == 2:
            adjacency_to_elems = {}
            for conn in self.blocks.values():
                for elem in conn:
                    n = len(elem)
                    for k in range(n):
                        edge = frozenset((int(elem[k]), int(elem[(k + 1) % n])))
                        adjacency_to_elems.setdefault(edge, []).append(True)
            for edge, elist in adjacency_to_elems.items():
                if len(elist) == 1:
                    for node_id in edge:
                        mask[node_id] = True
        elif self.dim == 3:
            adjacency_to_elems = {}
            for conn in self.blocks.values():
                for elem in conn:
                    for face in _elem_faces_3d(elem):
                        adjacency_to_elems.setdefault(face, []).append(True)
            for face, elist in adjacency_to_elems.items():
                if len(elist) == 1:
                    for node_id in face:
                        mask[node_id] = True
        else:
            raise NotImplementedError(
                f"compute_boundary_mask(): not meaningful for a {self.dim}-D mesh "
                "-- supported dimensions are 1, 2, and 3.")

        if store_as is not None:
            self.point_data[store_as] = mask
        return mask

    def total_elements(self):
        return sum(len(conn) for conn in self.blocks.values())

    def elements_of_type(self, block_name):
        """Convenience accessor for code (e.g. plotting) that wants one
        block's raw connectivity array without reaching into .blocks
        directly -- mirrors the flat .elements array every single-type
        Mesh already exposes, one block at a time."""
        return self.blocks[block_name]

    def check_quality(self, elem_formulations, verbose=True):
        """Generalizes Mesh.check_quality() across blocks: elem_formulations
        is a dict with the SAME keys as self.blocks, each value the
        Element instance to check that block against. Reports a single
        pass/fail across the whole mesh, plus a per-block element count."""
        try:
            from .elements import gauss_product
        except ImportError:
            from elements import gauss_product
        detJ_all = []
        n_bad = 0
        n_elements = 0
        for name, connectivity in self.blocks.items():
            if name not in elem_formulations:
                raise KeyError(
                    f"check_quality: no element formulation given for block {name!r} "
                    f"(elem_formulations keys: {sorted(elem_formulations)})")
            formulation = elem_formulations[name]
            pts, wts = gauss_product(formulation.gauss_order, formulation.dim)
            n_elements += len(connectivity)
            for elem in connectivity:
                elem_coords = self.nodes[elem]
                bad = False
                for p in pts:
                    _, dN = formulation.shape_and_derivs(p)
                    J = dN @ elem_coords
                    detJ = np.linalg.det(J) if J.shape[0] > 1 else J[0, 0]
                    detJ_all.append(detJ)
                    if detJ <= 0:
                        bad = True
                if bad:
                    n_bad += 1
            if verbose:
                print(f"  block {name!r}: {len(connectivity)} elements "
                      f"({formulation.__class__.__name__})")
        detJ_all = np.array(detJ_all)
        ratio = np.max(np.abs(detJ_all)) / np.min(np.abs(detJ_all))
        if verbose:
            print(f"  TOTAL: {n_elements} elements, {len(self.nodes)} nodes, "
                  f"{len(self.blocks)} block(s), dim={self.dim}")
            print(f"  detJ range: [{detJ_all.min():.3e}, {detJ_all.max():.3e}]  "
                  f"(max/min |detJ| ratio: {ratio:.1f})")
            print("  OK: all Gauss-point Jacobians positive (no inverted elements)"
                  if n_bad == 0 else
                  f"  *** {n_bad} element(s) with non-positive detJ -- INVERTED/DEGENERATE ***")
        return n_bad == 0


# =====================================================================
# 1-D
# =====================================================================
def line_mesh(L, n, x0=0.0):
    """n 2-node line elements spanning [x0, x0+L] -- for 1-D beam elements."""
    x = np.linspace(x0, x0 + L, n + 1).reshape(-1, 1)
    elements = np.array([[i, i + 1] for i in range(n)], dtype=int)
    return Mesh(x, elements, dim=1)


# =====================================================================
# 2-D
# =====================================================================
def rectangle_mesh(Lx, Ly, nx, ny, x0=0.0, y0=0.0):
    dx, dy = Lx / nx, Ly / ny
    n_nx, n_ny = nx + 1, ny + 1

    def node_id(i, j):
        return i * n_ny + j

    nodes = np.zeros((n_nx * n_ny, 2))
    for i in range(n_nx):
        for j in range(n_ny):
            nodes[node_id(i, j)] = [x0 + i * dx, y0 + j * dy]

    elements = []
    for i in range(nx):
        for j in range(ny):
            elements.append([node_id(i, j), node_id(i + 1, j),
                              node_id(i + 1, j + 1), node_id(i, j + 1)])
    return Mesh(nodes, np.array(elements, dtype=int), dim=2)


def rectangle_with_hole_mesh_quarter(a, b, R, nr, ntheta, grade_p=2.0):
    """One quadrant of a rectangle with a circular hole at the origin,
    mapped/transfinite mesh (hole boundary -> outer rectangle edge)."""
    theta_vals = np.linspace(0.0, np.pi / 2, ntheta + 1)
    s_vals = np.linspace(0.0, 1.0, nr + 1) ** grade_p
    n_nx, n_ny = nr + 1, ntheta + 1

    def node_id(i, j):
        return i * n_ny + j

    nodes = np.zeros((n_nx * n_ny, 2))
    for i in range(n_nx):
        for j in range(n_ny):
            th = theta_vals[j]
            ct, st = np.cos(th), np.sin(th)
            inner = np.array([R * ct, R * st])
            t_out = min(a / ct if ct > 1e-9 else np.inf,
                        b / st if st > 1e-9 else np.inf)
            outer = np.array([t_out * ct, t_out * st])
            nodes[node_id(i, j)] = inner + s_vals[i] * (outer - inner)

    elements = []
    for i in range(nr):
        for j in range(ntheta):
            elements.append([node_id(i, j), node_id(i + 1, j),
                              node_id(i + 1, j + 1), node_id(i, j + 1)])
    return Mesh(nodes, np.array(elements, dtype=int), dim=2)


# graded_partition() now lives in grading.py (imported above) -- moved
# there as part of the generalized mesh-grading module so both this
# structured front end and geometry/gmsh_engine.py's unstructured front
# end share one implementation. Still importable as
# `fea_engine.mesh.graded_partition` for backward compatibility with any
# existing caller.


def rectangle_mesh_from_partitions(x_coords, y_coords):
    """Structured Quad4 mesh from EXPLICIT (possibly non-uniform) x/y
    coordinate arrays -- generalizes rectangle_mesh()'s uniform
    np.linspace to any strictly increasing partition, so callers can
    grade element size (fine near a feature, coarse in the far field)
    without leaving the structured, axis-aligned-quad family. This
    matters beyond convenience: Shell4MITC (elements/shells.py) was
    found to develop severe spurious stiffening -- 40x-1500x too stiff
    in cases checked -- whenever neighboring elements' local frames are
    mutually rotated, which happens on ANY unstructured mesh (Gmsh's
    recombine pass included) or even a lightly-perturbed structured
    grid, but NOT on a structured axis-aligned grid regardless of how
    non-uniform its spacing is (see NonLin-HyROM's Case 1 verification
    session, 2026-09-03, for the isolating experiments). Grading via
    THIS function instead of an unstructured mesher is therefore not
    just an efficiency choice -- it's the only currently-reliable way
    to vary element size and still get correct Shell4MITC results."""
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    if np.any(np.diff(x_coords) <= 0) or np.any(np.diff(y_coords) <= 0):
        raise ValueError("x_coords and y_coords must be strictly increasing")
    n_nx, n_ny = len(x_coords), len(y_coords)

    def node_id(i, j):
        return i * n_ny + j

    nodes = np.zeros((n_nx * n_ny, 2))
    for i in range(n_nx):
        for j in range(n_ny):
            nodes[node_id(i, j)] = [x_coords[i], y_coords[j]]

    elements = []
    for i in range(n_nx - 1):
        for j in range(n_ny - 1):
            elements.append([node_id(i, j), node_id(i + 1, j),
                              node_id(i + 1, j + 1), node_id(i, j + 1)])
    return Mesh(nodes, np.array(elements, dtype=int), dim=2)


def weld_meshes(mesh_a, mesh_b, tol=1e-9):
    """Merge two 2-D Quad4 meshes sharing one node array's worth of
    coincident boundary nodes into a single connected Mesh -- e.g.
    stitching a near-feature mapped block (rectangle_with_hole_mesh_
    quarter) to a plain/graded extension block along their shared
    edge. Every mesh_b node within `tol` of a mesh_a node is welded
    onto it; other mesh_b nodes are appended after mesh_a's (same
    matching scheme mirror_mesh() uses internally, generalized to two
    independent meshes instead of one mesh and its own reflection)."""
    tree = cKDTree(mesh_a.nodes)
    dist, idx = tree.query(mesh_b.nodes, k=1)
    node_map = np.where(dist < tol, idx, -1)
    combined_nodes = list(mesh_a.nodes)
    for i in range(len(mesh_b.nodes)):
        if node_map[i] == -1:
            node_map[i] = len(combined_nodes)
            combined_nodes.append(mesh_b.nodes[i])
    combined_nodes = np.array(combined_nodes)
    combined_elements = np.vstack([mesh_a.elements, node_map[mesh_b.elements]])
    return Mesh(combined_nodes, combined_elements, dim=mesh_a.dim)


def rectangle_with_hole_mesh_quarter_full(a, b, R, nr, ntheta, grade_p=2.0,
                                           n_extend=None, extend_grade=2.0):
    """rectangle_with_hole_mesh_quarter(), fixed for a != b (elongated
    quadrants): that function traces its outer boundary at UNIFORMLY
    SPACED ANGLES, so for a >> b or b >> a almost every ray lands on
    the SHORT edge and only a couple reach the LONG edge -- e.g. for
    a=0.5, b=0.1, ntheta=10, only 2 of the 11 rays reach x=a, leaving
    that edge (often the physically important one, e.g. a clamped
    boundary) almost unresolved and a few percent of the rectangle's
    area unmeshed between the traced boundary and the true corner.

    Fix: mesh a SQUARE near-hole block at size min(a,b) x min(a,b) with
    the original function (already exact for a==b -- the two edges
    split evenly at the diagonal), then extend the long direction from
    min(a,b) out to max(a,b) with a separate rectangle_mesh_from_
    partitions() block, graded fine-near-hole to coarse-far-field via
    extend_grade, and weld it to the near-hole block's outer edge. For
    a==b this returns exactly rectangle_with_hole_mesh_quarter()'s own
    result (no extension block needed). n_extend defaults to nr."""
    if np.isclose(a, b):
        return rectangle_with_hole_mesh_quarter(a, b, R, nr, ntheta, grade_p)
    if n_extend is None:
        n_extend = nr

    s = min(a, b)
    hole_block = rectangle_with_hole_mesh_quarter(s, s, R, nr, ntheta, grade_p)

    if a > b:
        edge_mask = np.isclose(hole_block.nodes[:, 0], s, atol=1e-9)
        y_partition = np.sort(hole_block.nodes[edge_mask, 1])
        x_partition = graded_partition(s, a, n_extend, grade=extend_grade, dense_at="start")
        ext_block = rectangle_mesh_from_partitions(x_partition, y_partition)
    else:
        edge_mask = np.isclose(hole_block.nodes[:, 1], s, atol=1e-9)
        x_partition = np.sort(hole_block.nodes[edge_mask, 0])
        y_partition = graded_partition(s, b, n_extend, grade=extend_grade, dense_at="start")
        ext_block = rectangle_mesh_from_partitions(x_partition, y_partition)

    return weld_meshes(hole_block, ext_block)


def mirror_mesh(mesh, mirror_x=False, mirror_y=False, tol=1e-9):
    """Mirror a 2-D Quad4 mesh across x=0 and/or y=0, welding nodes that
    land on the mirror line so the result is one clean connected mesh."""
    nodes, elements = mesh.nodes, mesh.elements

    def _mirror_once(nodes, elements, axis):
        n_orig = len(nodes)
        mirrored = nodes.copy()
        mirrored[:, axis] *= -1.0
        tree = cKDTree(nodes)
        dist, idx = tree.query(mirrored, k=1)
        node_map = np.where(dist < tol, idx, -1)
        combined = list(nodes)
        for i in range(n_orig):
            if node_map[i] == -1:
                node_map[i] = len(combined)
                combined.append(mirrored[i])
        combined = np.array(combined)
        mirrored_elements = node_map[elements][:, ::-1]  # restore CCW winding
        return combined, np.vstack([elements, mirrored_elements])

    if mirror_x:
        nodes, elements = _mirror_once(nodes, elements, axis=0)
    if mirror_y:
        nodes, elements = _mirror_once(nodes, elements, axis=1)
    return Mesh(nodes, elements, dim=2)


# =====================================================================
# Generalized structured grading (Phase 2 of docs/
# generalized_mesh_grading_roadmap.md): a hole ANYWHERE inside a
# rectangle, not just at the origin of an already-quarter-symmetric
# domain. Built by composition on top of rectangle_with_hole_mesh_
# quarter_full() (unchanged, still validated by its own existing
# callers -- e.g. NonLin-HyROM's case1_flat_plate_with_hole.py -- rather
# than rewritten in place, since it already IS this function's
# symmetric-hole-at-center special case).
# =====================================================================
def _reflect_mesh(mesh, axis):
    """Pure reflection of a 2-D Quad4 mesh about x=0 (axis=0) or y=0
    (axis=1) -- unlike mirror_mesh(), this does NOT weld the reflection
    back onto the original; it returns an independent, disjoint mesh.
    Building block for placing independently-built quadrant blocks into
    position around an off-center feature (hole_in_rectangle_mesh()
    below), which welds them together itself afterward."""
    nodes = mesh.nodes.copy()
    nodes[:, axis] *= -1.0
    elements = mesh.elements[:, ::-1]  # reflection flips winding -- reverse it
    return Mesh(nodes, elements, dim=mesh.dim)


def _translate_mesh(mesh, dx, dy):
    """Rigid translation of a 2-D mesh (no other change)."""
    nodes = mesh.nodes.copy()
    nodes[:, 0] += dx
    nodes[:, 1] += dy
    return Mesh(nodes, mesh.elements.copy(), dim=mesh.dim)


def hole_in_rectangle_mesh(Lx, Ly, hole_center, R, nr, ntheta, grade_p=2.0,
                            n_extend=None, extend_grade=2.0, weld_tol=1e-9):
    """A circular hole ANYWHERE inside a (not necessarily symmetric)
    rectangle [0,Lx] x [0,Ly] -- the generalization of
    rectangle_with_hole_mesh_quarter_full() beyond "hole at the origin
    of an already-quarter-symmetric domain" (that function's own
    docstring explains ITS generalization, over the plain
    rectangle_with_hole_mesh_quarter(), to a != b; this one generalizes
    over hole PLACEMENT).

    Built from 4 independent quadrant blocks (one per direction from the
    hole center), each reusing rectangle_with_hole_mesh_quarter_full()'s
    own near-hole + graded-extension construction with THAT quadrant's
    own distances to the rectangle's edges, welded together.

    Why the weld is exact, not approximate: two quadrants that share an
    edge always share the SAME distance to whichever rectangle edge that
    shared edge runs into -- e.g. the +x,+y and -x,+y quadrants both
    border the vertical ray x=hole_center[0], and both are "+y"
    quadrants, so both are built with the same b=Ly-hole_center[1]. As
    long as every quadrant also uses the same nr/ntheta/grade_p/
    n_extend/extend_grade (one hole, one consistent resolution -- the
    natural choice, and the only one this function offers), the shared
    ray's node coordinates come out bit-for-bit identical on both sides,
    not just close, so weld_meshes()'s tolerance only has to absorb
    floating-point roundoff, never a real geometric mismatch.

    hole_center=(Lx/2, Ly/2) is the symmetric case rectangle_with_hole_
    mesh_quarter_full() already covers (see tests/
    test_mesh_grading_generalized.py for the cross-check against that
    existing function via total mesh area, which is placement-agnostic
    and therefore checks the general off-center case directly rather
    than only the symmetric reduction)."""
    cx, cy = hole_center
    a_plus, a_minus = Lx - cx, cx
    b_plus, b_minus = Ly - cy, cy
    if min(a_plus, a_minus, b_plus, b_minus) <= R:
        raise ValueError(
            f"hole_center={hole_center}, R={R} -- the hole must be fully "
            f"interior with clearance > R on all four sides of the "
            f"rectangle [0,{Lx}]x[0,{Ly}], got clearances "
            f"{(a_plus, a_minus, b_plus, b_minus)}")

    def _quadrant(a, b, flip_x, flip_y):
        m = rectangle_with_hole_mesh_quarter_full(
            a, b, R, nr, ntheta, grade_p, n_extend, extend_grade)
        if flip_x:
            m = _reflect_mesh(m, axis=0)
        if flip_y:
            m = _reflect_mesh(m, axis=1)
        return m

    q_pp = _quadrant(a_plus, b_plus, False, False)
    q_np = _quadrant(a_minus, b_plus, True, False)
    q_nn = _quadrant(a_minus, b_minus, True, True)
    q_pn = _quadrant(a_plus, b_minus, False, True)

    combined = q_pp
    for q in (q_np, q_nn, q_pn):
        combined = weld_meshes(combined, q, tol=weld_tol)

    return _translate_mesh(combined, cx, cy)


def hole_in_rectangle_mesh_graded(Lx, Ly, hole_center, R, nr, ntheta, h_far,
                                   growth_ratio=None, n_extend=None,
                                   weld_tol=1e-9):
    """hole_in_rectangle_mesh(), but `grade_p`/`extend_grade` are DERIVED
    from a target growth ratio (grading.py, Section 1b of the roadmap
    doc) instead of hand-picked -- the actual "growth-ratio-driven
    grading" capability the generalized-grading design calls for, not
    just the geometric generalization above. `h_far` is the caller's
    target element size away from the hole (used only to size the
    far-field extension's grading via n_extend, not to change the
    mesh's node count); `nr`/`n_extend` (defaulting to nr) are still the
    caller's element-count budget -- this function decides how
    AGGRESSIVELY to grade within that budget, not how many elements to
    use."""
    try:
        from .grading import grade_for_growth_ratio, DEFAULT_GROWTH_RATIO
    except ImportError:
        from grading import grade_for_growth_ratio, DEFAULT_GROWTH_RATIO
    if growth_ratio is None:
        growth_ratio = DEFAULT_GROWTH_RATIO
    n_ext = n_extend if n_extend is not None else nr
    grade_p = grade_for_growth_ratio(nr, growth_ratio, dense_at="start")
    extend_grade = grade_for_growth_ratio(n_ext, growth_ratio, dense_at="start")
    return hole_in_rectangle_mesh(Lx, Ly, hole_center, R, nr, ntheta,
                                   grade_p=grade_p, n_extend=n_extend,
                                   extend_grade=extend_grade, weld_tol=weld_tol)


# =====================================================================
# Fillet and Notch on the structured path (Wave 5 items 26/27, docs/
# consolidated_future_roadmap.md) -- the two features generalized_mesh_
# grading_roadmap.md's Phase 2 explicitly deferred ("Fillet/Notch on
# the structured path: NOT implemented this session ... Real, concrete
# future work, not silently dropped."). Both reuse the SAME
# block-decomposition-and-weld pattern hole_in_rectangle_mesh() already
# established: independent rectangular (or, for Fillet, one curved)
# blocks, built so their shared edges coincide EXACTLY (same element
# count and spacing law on both sides of a shared boundary, not merely
# close), then stitched with weld_meshes() -- the same exactness
# argument hole_in_rectangle_mesh()'s own docstring makes for its
# 4-quadrant weld.
# =====================================================================
def _quarter_disk_sector_mesh(R, nr, ntheta, grade_p=1.0):
    """Solid quarter-disk sector of radius R, centered at the origin,
    material in the quadrant x>=0, y>=0 -- the corner block a filleted
    rectangle corner needs (see fillet_in_rectangle_mesh()'s own
    docstring for the geometry). Meshed as a COLLAPSED polar grid: all
    ntheta+1 nodes at the innermost radius (i=0) coincide at the single
    origin node -- the standard, well-established way to mesh a solid
    disk/sector with quadrilaterals (the same "O-grid center"
    degeneracy every structured circular/polar mesh, from CFD boundary-
    layer O-grids to cylindrical mesh cores, resolves one way or
    another; collapsing to one node is the simplest of the standard
    choices, and the only one implemented here). Each of the ntheta
    elements touching the origin is therefore a DEGENERATE quad (2 of
    its 4 connectivity entries are the same node index, equivalent to a
    triangle) -- Mesh.check_quality() evaluates detJ only at INTERIOR
    Gauss points, never exactly at a node, so this stays numerically
    well-posed (a positive, finite Jacobian everywhere it's evaluated)
    as long as the two other, distinct corners of that element are not
    themselves coincident, which they never are for R > 0 -- verified
    directly against check_quality() in this module's own tests, not
    just argued.

    grade_p defaults to 1.0 (uniform radial spacing), unlike the hole
    case's default clustering toward the feature boundary -- there is
    no thin near-boundary feature to grade toward here (the disk's own
    boundary IS the arc, not an infinitesimally-thin hole), and grading
    toward r=0 would only shrink the already-small collapsed elements
    further for no benefit."""
    theta_vals = np.linspace(0.0, np.pi / 2, ntheta + 1)
    r_vals = np.linspace(0.0, 1.0, nr + 1) ** grade_p * R
    n_ny = ntheta + 1

    def node_id(i, j):
        if i == 0:
            return 0
        return 1 + (i - 1) * n_ny + j

    n_nodes = 1 + nr * n_ny
    nodes = np.zeros((n_nodes, 2))
    for i in range(nr + 1):
        for j in range(n_ny):
            r, th = r_vals[i], theta_vals[j]
            nodes[node_id(i, j)] = [r * np.cos(th), r * np.sin(th)]

    elements = []
    for i in range(nr):
        for j in range(ntheta):
            elements.append([node_id(i, j), node_id(i + 1, j),
                              node_id(i + 1, j + 1), node_id(i, j + 1)])
    return Mesh(nodes, np.array(elements, dtype=int), dim=2)


def fillet_in_rectangle_mesh(Lx, Ly, corner, R, nx, ny, nr, ntheta=8, corner_grade_p=1.0):
    """A rectangle [0,Lx] x [0,Ly] with ONE corner rounded off (Wave 5
    item 27, grading.Fillet) -- `corner` is one of the rectangle's own 4
    corner points (Lx,Ly), (0,Ly), (0,0), or (Lx,0) (checked against the
    actual rectangle, not guessed from a name/index).

    Geometry: the sharp corner is replaced by an arc of radius R,
    tangent to both edges at distance R from the corner along each --
    the standard engineering fillet. The arc's CENTER sits R inward
    from the corner along BOTH axes (not at the corner itself); the
    material inside the R x R corner square is exactly the quarter-disk
    of radius R around that center (_quarter_disk_sector_mesh() above),
    and the tiny sliver of the original sharp corner beyond the arc is
    what filleting removes.

    Built from 4 blocks for the canonical "+x,+y" corner (Lx,Ly), then
    reflected via the existing _reflect_mesh()/_translate_mesh() helpers
    (the identical technique hole_in_rectangle_mesh() already uses for
    its own 4 quadrants) for the other 3 corners:
        - main_body:     [0,Lx-R] x [0,Ly-R], plain rectangle_mesh(nx,ny)
        - right_strip:   [Lx-R,Lx] x [0,Ly-R], rectangle_mesh(nr,ny)
        - top_strip:     [0,Lx-R] x [Ly-R,Ly], rectangle_mesh(nx,nr)
        - corner_sector: the quarter-disk above, translated to
          (Lx-R, Ly-R)
    Every shared edge between these 4 blocks has the SAME element count
    (nr along both fillet-square directions, nx/ny along the two
    far-field directions) and the SAME (uniform, since corner_grade_p
    defaults to 1.0 on both the sector's own radial law and the strips'
    plain rectangle_mesh spacing) coordinate law, so weld_meshes() only
    ever has floating-point roundoff to absorb, never a real mismatch.

    R must leave room on both edges meeting at `corner` (0 < R < Lx and
    0 < R < Ly, strictly), else main_body/the opposite strip would have
    non-positive extent.

    SAFETY NOTE for rotational-DOF elements (Shell4MITC/
    Shell4MITCCorotational/Quad4MindlinPlate): unlike notch_in_
    rectangle_mesh() below (which stays perfectly axis-aligned
    throughout, including its 90-degree-rotated left/right variants --
    a GLOBAL rotation of an already-mutually-aligned grid, not a
    per-element misalignment), this mesh's corner_sector block has
    genuinely CURVED, non-axis-aligned element edges near the fillet
    arc -- the same category of topology docs/generalized_mesh_grading_
    roadmap.md's Section 2 flags as not verified safe for those
    elements (the existing rectangle_with_hole_mesh_quarter() family
    has this same caveat, for the same reason: its near-hole elements
    are curved/mapped too). This function does NOT run the
    ROTATIONAL_DOF_ELEMENTS safety gate build_mesh.py's dispatcher uses
    for Hole features -- extending that gate to Fillet/Notch is Wave 5
    items 28/29, deliberately not attempted this pass (still blocked on
    the same underlying Wave 4 shell-formulation risk, see the roadmap
    doc). Safe to use freely with any translational-DOF element
    (Quad4PlaneStress, Tri3PlaneStress, ...); do not pair with a
    rotational-DOF element without the same kind of verification Wave 4
    items 20/22 attempted (and deferred) for the hole case."""
    if R <= 0:
        raise ValueError(f"R must be > 0, got {R}")
    if R >= Lx or R >= Ly:
        raise ValueError(
            f"fillet radius R={R} must be strictly less than both Lx={Lx} "
            f"and Ly={Ly} -- otherwise the fillet consumes the whole "
            "rectangle in that direction")

    corners = {(Lx, Ly): (False, False), (0.0, Ly): (True, False),
               (0.0, 0.0): (True, True), (Lx, 0.0): (False, True)}
    key = min(corners, key=lambda c: (c[0] - corner[0]) ** 2 + (c[1] - corner[1]) ** 2)
    if (key[0] - corner[0]) ** 2 + (key[1] - corner[1]) ** 2 > 1e-9 * (Lx ** 2 + Ly ** 2):
        raise ValueError(
            f"corner={corner} does not match any of the rectangle's own 4 "
            f"corners {list(corners)} (within tolerance) -- fillet_in_"
            "rectangle_mesh only rounds an actual rectangle corner")
    flip_x, flip_y = corners[key]

    main_body = rectangle_mesh(Lx - R, Ly - R, nx, ny)
    right_strip = _translate_mesh(rectangle_mesh(R, Ly - R, nr, ny), Lx - R, 0.0)
    top_strip = _translate_mesh(rectangle_mesh(Lx - R, R, nx, nr), 0.0, Ly - R)
    sector = _translate_mesh(
        _quarter_disk_sector_mesh(R, nr, ntheta, grade_p=corner_grade_p), Lx - R, Ly - R)

    combined = main_body
    for blk in (right_strip, top_strip, sector):
        combined = weld_meshes(combined, blk)

    if flip_x:
        combined = _reflect_mesh(combined, axis=0)
        combined = _translate_mesh(combined, Lx, 0.0)
    if flip_y:
        combined = _reflect_mesh(combined, axis=1)
        combined = _translate_mesh(combined, 0.0, Ly)
    return combined


def notch_in_rectangle_mesh(Lx, Ly, edge, s0, s1, depth,
                             n_left, n_mid, n_right, n_depth, n_wall):
    """A rectangle [0,Lx] x [0,Ly] with a rectangular slot (Wave 5 item
    27, grading.Notch) cut inward from one boundary edge -- the
    straight-walled notch geometry grading.Notch's own docstring scopes
    this feature to (see that class for why an arbitrary notch path was
    narrowed down to this).

    `edge` in {"top","bottom","left","right"} names which boundary edge
    the notch opens into; `s0 < s1` are the mouth's two endpoints in
    that edge's own tangential coordinate (x for top/bottom, y for
    left/right); `depth` is how far inward (perpendicular to `edge`)
    the slot cuts -- must leave material on both sides of the mouth
    (0 < s0 < s1 < the edge's own length) and below the slot (0 < depth
    < the rectangle's own extent perpendicular to `edge`).

    Built canonically for edge="top" as 3 blocks -- a left column and a
    right column (both FULL height [0,Ly], via rectangle_mesh_from_
    partitions with a y-partition that includes Ly-depth as an exact
    breakpoint, so the notch's own floor sits on an already-existing
    node row rather than needing a new weld seam there), and a middle
    block spanning only [0, Ly-depth] (the material directly under the
    slot -- there is no block for x in [s0,s1], y in [Ly-depth,Ly]:
    that IS the slot, empty) -- then welded. The other 3 edges reuse
    this exact same canonical mesh, mapped onto the true rectangle by
    an explicit affine map per edge (derived and orientation-checked
    below, not just assumed): "bottom" is a plain y-reflection (the
    existing _reflect_mesh()/_translate_mesh() composition); "left" and
    "right" both need x<->y swapped, but one is a 90-degree ROTATION
    (determinant +1, winding preserved) and the other a pure AXIS SWAP
    (determinant -1, winding reversed) -- confirmed by direct 2x2
    Jacobian sign, then verified numerically via check_quality() in
    this module's own tests, not trusted from the algebra alone."""
    if edge not in ("top", "bottom", "left", "right"):
        raise ValueError(f"edge must be 'top'/'bottom'/'left'/'right', got {edge!r}")
    span = Lx if edge in ("top", "bottom") else Ly
    perp = Ly if edge in ("top", "bottom") else Lx
    if not (0.0 < s0 < s1 < span):
        raise ValueError(
            f"notch mouth [{s0}, {s1}] must lie strictly inside the "
            f"{edge!r} edge's own extent (0, {span}) -- got s0={s0}, s1={s1}")
    if not (0.0 < depth < perp):
        raise ValueError(
            f"notch depth={depth} must be strictly between 0 and the "
            f"rectangle's extent perpendicular to {edge!r} ({perp})")

    # Canonical construction: notch opens DOWNWARD into a rectangle of
    # width `span` (x) and height `perp` (y) from its TOP edge (y=perp),
    # mouth at x in [s0,s1]. edge="top" IS this canonical case;
    # "bottom"/"left"/"right" map this same shape onto the true
    # [0,Lx]x[0,Ly] rectangle below.
    y_body = graded_partition(0.0, perp - depth, n_depth, grade=1.0)
    y_wall = graded_partition(perp - depth, perp, n_wall, grade=1.0)
    y_full = np.concatenate([y_body, y_wall[1:]])

    x_left = graded_partition(0.0, s0, n_left, grade=1.0)
    x_mid = graded_partition(s0, s1, n_mid, grade=1.0)
    x_right = graded_partition(s1, span, n_right, grade=1.0)

    left_col = rectangle_mesh_from_partitions(x_left, y_full)
    right_col = rectangle_mesh_from_partitions(x_right, y_full)
    mid_block = rectangle_mesh_from_partitions(x_mid, y_body)

    combined = weld_meshes(left_col, right_col)
    combined = weld_meshes(combined, mid_block)

    if edge == "top":
        return combined
    if edge == "bottom":
        combined = _reflect_mesh(combined, axis=1)
        return _translate_mesh(combined, 0.0, perp)
    if edge == "left":
        # canonical (x,y) -> true (perp - y, x): a 90-degree rotation
        # (Jacobian det = +1) -- winding is preserved.
        nodes = np.column_stack([perp - combined.nodes[:, 1], combined.nodes[:, 0]])
        return Mesh(nodes, combined.elements.copy(), dim=2)
    # "right": canonical (x,y) -> true (y, x): a pure axis swap
    # (Jacobian det = -1) -- winding DOES flip, reverse connectivity.
    nodes = np.column_stack([combined.nodes[:, 1], combined.nodes[:, 0]])
    elements = combined.elements[:, ::-1].copy()
    return Mesh(nodes, elements, dim=2)


# =====================================================================
# 3-D (composed from 2-D via extrusion)
# =====================================================================
def extrude_mesh(mesh2d, Lz, nz, z0=0.0):
    """Turn any Quad4 Mesh into a Hex8 Mesh by stacking nz layers along z."""
    nodes2d, elements2d = mesh2d.nodes, mesh2d.elements
    n2d = len(nodes2d)
    z_vals = z0 + np.linspace(0.0, Lz, nz + 1)
    nodes3d = np.zeros((n2d * (nz + 1), 3))
    for k, z in enumerate(z_vals):
        nodes3d[k * n2d:(k + 1) * n2d, 0:2] = nodes2d
        nodes3d[k * n2d:(k + 1) * n2d, 2] = z

    elements3d = []
    for k in range(nz):
        offset_bot, offset_top = k * n2d, (k + 1) * n2d
        for quad in elements2d:
            elements3d.append(list(quad + offset_bot) + list(quad + offset_top))
    return Mesh(nodes3d, np.array(elements3d, dtype=int), dim=3)


def box_mesh(Lx, Ly, Lz, nx, ny, nz):
    return extrude_mesh(rectangle_mesh(Lx, Ly, nx, ny), Lz, nz)


def box_with_hole_mesh(a, b, R, Lz, nr, ntheta, nz, grade_p=2.0):
    m2d = rectangle_with_hole_mesh_quarter(a, b, R, nr, ntheta, grade_p)
    m2d = mirror_mesh(m2d, mirror_x=True, mirror_y=True)
    return extrude_mesh(m2d, Lz, nz)


# =====================================================================
# Plotting (QA, not physics post-processing -- kept here since it only
# needs nodes/elements, not any solved field)
# =====================================================================
def plot_mesh_2d(mesh, ax, color='0.4', lw=0.5):
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for elem in mesh.elements:
        for i0, i1 in edges:
            p0, p1 = mesh.nodes[elem[i0]], mesh.nodes[elem[i1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw)
    ax.set_aspect('equal')


def plot_mesh_3d(mesh, ax, color='0.4', lw=0.4):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    segments = [(mesh.nodes[elem[i0]], mesh.nodes[elem[i1]])
                for elem in mesh.elements for i0, i1 in edges]
    ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=lw))


_HATCH_DIRECTIONS = {'left': (-1, 0), 'right': (1, 0), 'below': (0, -1), 'above': (0, 1)}


def plot_mesh_annotated(mesh, annotations=None, title="", figsize=(8, 6)):
    """2-D mesh plot (grid via plot_mesh_2d) with a set of highlighted
    node groups drawn on top -- boundary conditions, loads, symmetry/free
    edges -- each described by one dict in `annotations`:

        nodes      : array of node indices to highlight (required)
        label      : legend label (optional; omit for no legend entry)
        color      : highlight color (default 'red')
        marker     : matplotlib marker string (e.g. 's', '^') to scatter
                     at each node; None (default) draws no markers
        connect    : if True (default) and marker is None, draw a line
                     through `nodes` (ordered along whichever coordinate
                     varies most across them) -- e.g. a symmetry/free
                     edge. Set False to suppress (e.g. when only
                     markers/hatch/arrows are wanted, as for a small
                     cluster of point-loaded/fixed nodes rather than a
                     whole edge)
        linestyle  : line style for the `connect` line (default '-')
        hatch      : 'left'/'right'/'above'/'below' -- draws a short
                     perpendicular tick at each node, the standard
                     "fixed wall" support symbol
        arrow      : (dx, dy) direction (need not be unit length) --
                     draws an arrow at each node in that direction,
                     e.g. an applied load or traction
        arrow_scale: arrow length as a fraction of the mesh's bounding-
                     box diagonal (default 0.08)

    Returns (fig, ax) -- fig is NOT saved/shown here (same convention as
    every other plot function in this package); the caller does that,
    typically via fig.savefig(...) / plt.show()."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    plot_mesh_2d(mesh, ax, color='0.75', lw=0.5)

    nodes = mesh.nodes
    diag = float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0))) or 1.0
    hatch_len = 0.02 * diag

    for ann in (annotations or []):
        idx = np.asarray(ann['nodes'], dtype=int)
        if len(idx) == 0:
            continue
        pts = nodes[idx]
        color = ann.get('color', 'red')
        marker = ann.get('marker', None)
        label = ann.get('label', None)
        connect = ann.get('connect', True)
        linestyle = ann.get('linestyle', '-')
        hatch = ann.get('hatch', None)
        arrow = ann.get('arrow', None)
        arrow_scale = ann.get('arrow_scale', 0.08)

        used_label = False

        if marker is not None:
            ax.plot(pts[:, 0], pts[:, 1], marker, color=color, linestyle='None',
                     ms=5, label=label, zorder=5)
            used_label = True

        if connect and marker is None:
            spread = pts.max(axis=0) - pts.min(axis=0)
            axis = int(np.argmax(spread))
            order = np.argsort(pts[:, axis])
            ax.plot(pts[order, 0], pts[order, 1], linestyle=linestyle, color=color,
                     lw=1.5, label=None if used_label else label, zorder=4)
            used_label = True

        if hatch is not None:
            dx, dy = _HATCH_DIRECTIONS[hatch]
            for p in pts:
                ax.plot([p[0], p[0] + dx * hatch_len], [p[1], p[1] + dy * hatch_len],
                        color=color, lw=1.0, zorder=3)
            if not used_label and label is not None:
                ax.plot([], [], color=color, lw=1.0, label=label)
                used_label = True

        if arrow is not None:
            dx, dy = arrow
            norm = np.hypot(dx, dy) or 1.0
            dx, dy = dx / norm, dy / norm
            alen = arrow_scale * diag
            for p in pts:
                ax.annotate('', xy=(p[0] + dx * alen, p[1] + dy * alen), xytext=(p[0], p[1]),
                             arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=6)
            if not used_label and label is not None:
                ax.plot([], [], color=color, lw=1.5, label=label)
                used_label = True

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    return fig, ax
