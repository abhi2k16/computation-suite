"""
mesh_io.py -- Wave 14 item 122 (docs/consolidated_future_roadmap.md),
scoped directly against TensorMesh's own Meshes documentation page
(https://docs.tensor-mesh.com/user_guide/meshes.html) and the earlier
question this project answered about importing an Abaqus/ANSYS/other
FE mesh (as opposed to CAD geometry, which geometry.gmsh_engine's own
`generate_from_step()` already handles via STEP/IGES/BREP): a thin
wrapper around the `meshio` package, opening up the ~30 mesh FILE
formats it understands (Abaqus INP, Nastran BDF, VTK/VTU, Gmsh MSH,
UNV, MED, ...) as real mesh-file imports/exports for this package,
matching this exact mechanism `geometry_meshing_alternatives_research.md`
already recommended (its own §4, "the practical adapter") but never
implemented.

Requires the `meshio` package (`pip install meshio`, or
`pip install fea_engine[meshio]`) -- NOT a dependency of the rest of
fea_engine: __init__.py does not import this module, so `import
fea_engine` still works with no meshio installed. Import this module
explicitly when you need it: `from fea_engine import mesh_io`.

NODE-ORDERING FINDING (checked directly against meshio's own source and
with real round-trip runs through this module's own test file, not
assumed): geometry.gmsh_engine.py's own `_GMSH_NODE_ORDER` table exists
because Gmsh's RAW/native node order for a 2nd-order element does not
always match this package's own element-formulation convention (see
that module's top-of-file comment). meshio maintains its OWN canonical
per-cell-type node order across every format it reads/writes (its own
internal `_gmsh_to_meshio_order()` table, `meshio/gmsh/common.py`) --
and inspecting that table directly shows it applies EXACTLY the same
permutation this package's own GMSH_NODE_ORDER attributes already do:
`tetra10`: `[0,1,2,3,4,5,6,7,9,8]` in both; `hexahedron20`:
`[0,1,2,3,4,5,6,7,8,11,13,9,16,18,19,17,10,12,14,15]` in both (compare
Tet10Solid3D.GMSH_NODE_ORDER / Hex20Solid3D.GMSH_NODE_ORDER); and
`triangle6`/`quad8` are left untouched by BOTH conventions (identity in
Tri6PlaneStress.GMSH_NODE_ORDER / Quad8PlaneStress.GMSH_NODE_ORDER, and
simply absent from meshio's own reorder table, meaning "pass through
Gmsh's own native order unchanged" there too -- which is itself already
this package's own order for those two types). In other words: meshio's
own canonical node order for every one of this package's four
quadratic element types IS ALREADY this package's own order. No further
reindexing is applied by this module -- confirmed empirically by
building a hand-written raw Gmsh v2.2 `.msh` file with a single tetra10
and a single hexahedron20 element (known node-tag order per Gmsh's own
documented convention), reading each through REAL meshio 5.3.5, and
comparing byte-for-byte against `raw_gmsh_order[:, GMSH_NODE_ORDER]` --
see tests/test_mesh_io.py's TestQuadraticNodeOrderAgreesWithGmsh class
for the reproducible version of that same check, plus a VTU round-trip
confirming meshio's canonical order survives a non-Gmsh format
unchanged too.

SCOPE, stated honestly rather than silently assumed: the check above
was run for the Gmsh-format (`.msh`) read path and a VTU round-trip of
an already-in-meshio-order array. meshio's architecture normalizes
every format it supports into this ONE shared canonical order (that is
the whole point of the library, and why format-specific `_order`
tables exist in its `abaqus`/`nastran`/`vtk` submodules too -- each
normalizes ITS format into the same canonical convention this module
relies on) -- but a caller reading a genuinely different format's
quadratic elements for the FIRST time (an Abaqus INP, a Nastran BDF)
has not had that specific path checked here. `Mesh.check_quality()`
(all-positive Jacobians) is a cheap, direct way to spot-check any new
format/element-type combination before trusting it, the same safety
net every other node-ordering claim in this package rests on ultimately
being checkable against.

Field-data round-tripping (item 121): `mesh_to_meshio()`/`write_mesh()`
DO carry over `.point_data`/`.cell_data` when saving a `Mesh`/
`MultiBlockMesh` that has any registered (via `register_point_data()`/
`register_element_data()`). `read_mesh()`/`mesh_from_meshio()`
deliberately do NOT populate `.point_data`/`.cell_data` on the mesh
they return -- reading a file's own field data back onto the (possibly
node-renumbered, see the orphan-node cleanup below) returned mesh is
real, scoped-out future work, not silently half-done; attach whatever
fields you need via `register_point_data()`/`register_element_data()`
after reading.
"""
import numpy as np
from .mesh import Mesh, MultiBlockMesh

try:
    import meshio
    _HAS_MESHIO = True
except ImportError:
    _HAS_MESHIO = False


def _require_meshio():
    if not _HAS_MESHIO:
        raise ImportError(
            "mesh_io.py requires the optional 'meshio' package: "
            "pip install meshio (or `pip install fea_engine[meshio]`) -- "
            "not required for the rest of fea_engine, which stays "
            "numpy/scipy only.")


# meshio cell-type name -> (nodes_per_element, topological dimension,
# short block-name). The short names deliberately match geometry.
# gmsh_engine.py's own _GMSH_TYPE_NAMES convention exactly ("tri3",
# "quad4", "tet4", "hex8", "tri6", "tet10", "quad8", "hex20") -- not
# meshio's own naming -- so a MultiBlockMesh built by THIS module looks
# identical, key-naming-wise, to one built by that one; the same
# FESystem usage pattern for a MultiBlockMesh applies regardless of
# which module produced it. Deliberately the SAME set of nine types
# geometry.gmsh_engine.py's own _GMSH_ELEMENT_NODE_COUNTS supports --
# not more -- since going beyond that would let this module hand back
# a mesh this package has no Element formulation to actually consume
# (the exact same scoping reasoning that module's own comment gives for
# leaving hex27/quad9 out).
_MESHIO_CELL_INFO = {
    "line":         (2, 1, "line2"),
    "triangle":     (3, 2, "tri3"),
    "quad":         (4, 2, "quad4"),
    "tetra":        (4, 3, "tet4"),
    "hexahedron":   (8, 3, "hex8"),
    "triangle6":    (6, 2, "tri6"),
    "tetra10":      (10, 3, "tet10"),
    "quad8":        (8, 2, "quad8"),
    "hexahedron20": (20, 3, "hex20"),
}

_FEA_BLOCK_TO_MESHIO_TYPE = {short: meshio_t for meshio_t, (_, _, short) in _MESHIO_CELL_INFO.items()}


def mesh_from_meshio(raw, dim):
    """Convert an in-memory `meshio.Mesh` into a `Mesh` (single element
    type) or `MultiBlockMesh` (more than one), the meshio-object
    analogue of geometry.gmsh_engine.UnifiedGeometryEngine._extract_
    mesh() -- deliberately mirroring that method's own two behaviors:

        dim: which TOPOLOGICAL dimension of cell to keep (1/2/3) --
            e.g. dim=2 keeps only triangle/triangle6/quad/quad8 cells,
            dropping any "line" cells the file also carries as boundary
            markers, the same way _extract_mesh(dim=2) only ever fetches
            Gmsh elements of that dimension. Required, not inferred --
            matching generate_from_step()'s own "explicit over magic"
            convention for exactly the same reason: a file can
            legitimately carry cells of more than one topological
            dimension (a volume mesh with its own boundary faces also
            tagged as 2-D cells) and guessing which one the caller
            wants would risk silently picking the wrong one.

        orphan-node cleanup: any node not used by any kept cell (a
            construction/marker point, or a cell type filtered out by
            the dim mismatch above) is dropped and the survivors
            re-indexed sequentially -- the identical cleanup
            _extract_mesh() already applies to Gmsh's own output, for
            the identical reason (an unreferenced node leaves a zero
            row/column in the assembled K, a singular system).

    Cell types outside `_MESHIO_CELL_INFO` (this package's own nine
    supported element topologies) are silently skipped, NOT dropped
    into `blocks` -- matching Gmsh's own construction-point handling,
    not `_extract_mesh()`'s unsupported-type RuntimeError, because a
    real-world mesh file legitimately carries element types this
    package has no formulation for at all (e.g. a wedge/prism region in
    an otherwise-hex Abaqus mesh) that the caller may simply not need
    for the part of the model they're importing. If NO cell of the
    requested `dim` survives at all, that IS treated as an error (see
    below) -- silently returning an empty mesh would be worse than
    raising."""
    _require_meshio()
    points = np.asarray(raw.points, dtype=float)
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2, or 3, got {dim!r}")
    nodes = points[:, :dim].copy()

    raw_blocks = {}
    for cell_type, conn in raw.cells_dict.items():
        if cell_type not in _MESHIO_CELL_INFO:
            continue
        npe, topo_dim, short_name = _MESHIO_CELL_INFO[cell_type]
        if topo_dim != dim:
            continue
        conn = np.asarray(conn, dtype=int)
        if conn.shape[1] != npe:
            raise RuntimeError(
                f"mesh_from_meshio(): meshio cell type {cell_type!r} reported "
                f"{conn.shape[1]} nodes/element, expected {npe} -- malformed input?")
        if short_name in raw_blocks:
            raw_blocks[short_name] = np.vstack([raw_blocks[short_name], conn])
        else:
            raw_blocks[short_name] = conn

    if not raw_blocks:
        raise RuntimeError(
            f"mesh_from_meshio(): no cells of topological dimension {dim} found "
            f"among the file's own types {sorted(raw.cells_dict)} -- this module "
            f"supports {sorted(k for k, (_, d, _) in _MESHIO_CELL_INFO.items() if d == dim)} "
            f"at dim={dim}.")

    # Orphan-node cleanup -- identical logic and identical reason to
    # geometry.gmsh_engine.UnifiedGeometryEngine._extract_mesh()'s own.
    used = np.unique(np.concatenate([c.ravel() for c in raw_blocks.values()]))
    if len(used) < len(nodes):
        remap = -np.ones(len(nodes), dtype=int)
        remap[used] = np.arange(len(used))
        nodes = nodes[used]
        raw_blocks = {name: remap[conn] for name, conn in raw_blocks.items()}

    if len(raw_blocks) == 1:
        (elements,) = raw_blocks.values()
        return Mesh(nodes, elements, dim=dim)
    return MultiBlockMesh(nodes, raw_blocks, dim=dim)


def read_mesh(filepath, dim, **kwargs):
    """Read any file format `meshio` understands (Abaqus INP, Nastran
    BDF, VTK/VTU, Gmsh MSH, UNV, MED, ...) into a `Mesh`/`MultiBlockMesh`.
    `dim` is required -- see mesh_from_meshio()'s own docstring for why.
    Extra `**kwargs` are forwarded to `meshio.read()` (e.g.
    `file_format=` to override format auto-detection)."""
    _require_meshio()
    raw = meshio.read(filepath, **kwargs)
    return mesh_from_meshio(raw, dim=dim)


def mesh_to_meshio(mesh):
    """Convert a `Mesh` or `MultiBlockMesh` into an in-memory
    `meshio.Mesh`, carrying over `.point_data` and, per-block,
    `.cell_data` if the mesh has any registered (Wave 14 item 121's
    register_point_data()/register_element_data()) -- the reverse
    direction of mesh_from_meshio()."""
    _require_meshio()
    nodes = np.asarray(mesh.nodes, dtype=float)
    if nodes.shape[1] < 3:
        # meshio/VTK convention: pad 2-D (or 1-D) coordinates to 3
        # columns -- the identical padding TensorMesh's own save()
        # does, and for the same reason (most meshio-consuming formats
        # and viewers assume 3-D points).
        padded = np.zeros((nodes.shape[0], 3))
        padded[:, :nodes.shape[1]] = nodes
        nodes = padded

    point_data = dict(mesh.point_data) if mesh.point_data else {}

    if isinstance(mesh, MultiBlockMesh):
        cells = []
        cell_data = {}
        for block_name, conn in mesh.blocks.items():
            if block_name not in _FEA_BLOCK_TO_MESHIO_TYPE:
                raise ValueError(
                    f"mesh_to_meshio(): block name {block_name!r} is not one of "
                    f"this module's known short topology names "
                    f"({sorted(_FEA_BLOCK_TO_MESHIO_TYPE)}) -- blocks built by "
                    "geometry.gmsh_engine or mesh_io.py already use these names; "
                    "a hand-built MultiBlockMesh with a different key convention "
                    "must be relabeled before it can be written out.")
            meshio_type = _FEA_BLOCK_TO_MESHIO_TYPE[block_name]
            cells.append((meshio_type, np.asarray(conn, dtype=int)))
            for field_name, arr in mesh.cell_data.get(block_name, {}).items():
                cell_data.setdefault(field_name, []).append(np.asarray(arr))
        return meshio.Mesh(nodes, cells, point_data=point_data,
                            cell_data=(cell_data or None))

    npe = mesh.elements.shape[1]
    candidates = [k for k, (n, d, _) in _MESHIO_CELL_INFO.items()
                  if n == npe and d == mesh.dim]
    if len(candidates) != 1:
        raise ValueError(
            f"mesh_to_meshio(): cannot uniquely infer a meshio cell type for "
            f"{npe} nodes/element at dim={mesh.dim} (candidates: {candidates}) -- "
            "every Mesh this package's own generators produce has exactly one "
            "match; a hand-built connectivity with an unusual element shape "
            "should use meshio.Mesh(...) directly instead of this helper.")
    meshio_type = candidates[0]
    cell_data = ({name: [np.asarray(arr)] for name, arr in mesh.cell_data.items()}
                 if mesh.cell_data else None)
    return meshio.Mesh(nodes, [(meshio_type, np.asarray(mesh.elements, dtype=int))],
                        point_data=point_data, cell_data=cell_data)


def write_mesh(mesh, filepath, **kwargs):
    """Write a `Mesh`/`MultiBlockMesh` to any file format `meshio` can
    write (inferred from `filepath`'s extension, or forced via
    `file_format=` in `**kwargs`)."""
    _require_meshio()
    raw = mesh_to_meshio(mesh)
    raw.write(filepath, **kwargs)
