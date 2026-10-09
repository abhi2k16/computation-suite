"""
test_mesh_io.py -- Wave 14 item 122 (docs/consolidated_future_roadmap.
md): meshio-based mesh file I/O. All tests here are torch-analogue
"meshio-gated": skip cleanly if the optional `meshio` package isn't
installed, matching this project's established `_HAS_TORCH`/`_HAS_GMSH`
skip convention for every other optional dependency.

Covers, in order: the node-ordering claim mesh_io.py's own module
docstring makes (checked directly here too, not just cited), a plain
linear round trip (no ordering ambiguity at all -- the sanity floor),
a MultiBlockMesh round trip, point_data/cell_data carrying over on
write, the orphan-node cleanup, and the explicit dim-mismatch/
unsupported-type error paths.
"""
__author__ = "Abhijeet"
import os
import tempfile

import numpy as np
import pytest

from fea_engine.mesh import Mesh, MultiBlockMesh, rectangle_mesh, box_mesh
from fea_engine.elements.solids import Tet10Solid3D, Hex20Solid3D

try:
    import meshio
    from fea_engine import mesh_io
    _HAS_MESHIO = True
except ImportError:
    _HAS_MESHIO = False

pytestmark = pytest.mark.skipif(not _HAS_MESHIO, reason="meshio not installed")


# =====================================================================
# The node-ordering claim, re-derived directly (not just trusted from
# the module docstring's citation of it) -- a hand-built raw Gmsh v2.2
# .msh file with KNOWN node-tag declaration order, read through REAL
# meshio, compared against this package's own GMSH_NODE_ORDER applied
# to that same raw order.
# =====================================================================
class TestQuadraticNodeOrderAgreesWithGmsh:
    def _write_raw_msh(self, path, gmsh_type, n_nodes):
        lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(n_nodes)]
        for i in range(1, n_nodes + 1):
            lines.append(f"{i} {i * 0.1} {i * 0.2} {i * 0.3}")
        lines += ["$EndNodes", "$Elements", "1"]
        tags = " ".join(str(i) for i in range(1, n_nodes + 1))
        lines.append(f"1 {gmsh_type} 2 0 0 {tags}")
        lines.append("$EndElements")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def test_tetra10_matches_fea_engine_own_order(self, tmp_path):
        print("=" * 70)
        print("CHECK 1: meshio's own parsed tetra10 connectivity, from a raw")
        print("hand-built Gmsh .msh file, matches Tet10Solid3D.GMSH_NODE_ORDER")
        print("applied to the SAME raw (0-indexed) declaration order")
        print("=" * 70)
        path = str(tmp_path / "tet10.msh")
        self._write_raw_msh(path, gmsh_type=11, n_nodes=10)
        raw = meshio.read(path)
        got = raw.cells_dict["tetra10"]
        raw_order = np.array([list(range(10))])
        expected = raw_order[:, Tet10Solid3D.GMSH_NODE_ORDER]
        assert np.array_equal(got, expected)
        print(f"  meshio: {got.tolist()}")
        print(f"  fea_engine-expected: {expected.tolist()}")
        print("  OK: identical, no extra reorder needed")

    def test_hexahedron20_matches_fea_engine_own_order(self, tmp_path):
        print("=" * 70)
        print("CHECK 2: same check for hexahedron20 (the more complex")
        print("permutation) against Hex20Solid3D.GMSH_NODE_ORDER")
        print("=" * 70)
        path = str(tmp_path / "hex20.msh")
        self._write_raw_msh(path, gmsh_type=17, n_nodes=20)
        raw = meshio.read(path)
        got = raw.cells_dict["hexahedron20"]
        raw_order = np.array([list(range(20))])
        expected = raw_order[:, Hex20Solid3D.GMSH_NODE_ORDER]
        assert np.array_equal(got, expected)
        print("  OK: identical, no extra reorder needed")

    def test_meshio_canonical_order_survives_a_non_gmsh_format(self, tmp_path):
        print("=" * 70)
        print("CHECK 3: meshio's canonical hexahedron20 order round-trips")
        print("UNCHANGED through a non-Gmsh format (VTU) -- format-independence")
        print("=" * 70)
        conn = np.array([list(range(20))])
        m = meshio.Mesh(np.random.default_rng(0).random((20, 3)), [("hexahedron20", conn)])
        path = str(tmp_path / "hex20.vtu")
        m.write(path)
        m2 = meshio.read(path)
        assert np.array_equal(m2.cells_dict["hexahedron20"], conn)
        print("  OK: VTU round-trip preserves meshio's own canonical order")


# =====================================================================
# mesh_io.py's own conversion functions
# =====================================================================
class TestMeshFromMeshioAndReadMesh:
    def test_linear_round_trip_via_msh(self, tmp_path):
        print("=" * 70)
        print("CHECK 1: a plain Quad4 rectangle_mesh survives a full")
        print("write_mesh() -> read_mesh() round trip through a real .msh file")
        print("(no ordering ambiguity at all for a linear element -- the sanity")
        print("floor every other check here builds on)")
        print("=" * 70)
        m = rectangle_mesh(2.0, 1.0, 4, 3)
        path = str(tmp_path / "rect.msh")
        mesh_io.write_mesh(m, path, file_format="gmsh22", binary=False)
        m2 = mesh_io.read_mesh(path, dim=2)
        assert isinstance(m2, Mesh)
        assert m2.nodes.shape == m.nodes.shape
        assert np.allclose(np.sort(m2.nodes, axis=0), np.sort(m.nodes, axis=0))
        assert m2.elements.shape == m.elements.shape
        print(f"  OK: {len(m2.nodes)} nodes, {len(m2.elements)} elements round-tripped")

    def test_multiblockmesh_round_trip(self, tmp_path):
        print("=" * 70)
        print("CHECK 2: a MultiBlockMesh (two element types sharing one node")
        print("array) round-trips through write_mesh()/read_mesh() as a")
        print("MultiBlockMesh again, with both blocks intact")
        print("=" * 70)
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]], dtype=float)
        blocks = {"quad4": np.array([[0, 1, 2, 3]]),
                  "tri3": np.array([[1, 4, 5], [1, 5, 2]])}
        mb = MultiBlockMesh(nodes, blocks, dim=2)
        path = str(tmp_path / "mixed.vtu")
        mesh_io.write_mesh(mb, path)
        mb2 = mesh_io.read_mesh(path, dim=2)
        assert isinstance(mb2, MultiBlockMesh)
        assert set(mb2.blocks) == {"quad4", "tri3"}
        assert mb2.blocks["quad4"].shape == (1, 4)
        assert mb2.blocks["tri3"].shape == (2, 3)
        print(f"  OK: blocks {list(mb2.blocks)} both survived round trip")

    def test_point_data_and_cell_data_carried_over_on_write(self, tmp_path):
        print("=" * 70)
        print("CHECK 3: register_point_data()/register_element_data() fields")
        print("(Wave 14 item 121) actually reach the written file")
        print("=" * 70)
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        u = np.linspace(0, 1, len(m.nodes))
        m.register_point_data("u", u)
        e = np.arange(len(m.elements), dtype=float)
        m.register_element_data("energy", e)
        path = str(tmp_path / "with_fields.vtu")
        mesh_io.write_mesh(m, path)
        raw = meshio.read(path)
        assert "u" in raw.point_data
        assert np.allclose(raw.point_data["u"], u)
        assert "energy" in raw.cell_data
        print("  OK: point_data and cell_data both present in the written file")

    def test_orphan_node_dropped_and_reindexed(self, tmp_path):
        print("=" * 70)
        print("CHECK 4: a node not used by any dim-matching cell (a stray marker")
        print("point) is dropped and the rest re-indexed, same as gmsh_engine's")
        print("own _extract_mesh() cleanup -- an unreferenced node would leave a")
        print("singular row/column in K if it survived into FESystem")
        print("=" * 70)
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [5, 5]], dtype=float)
        cells = [("quad", np.array([[0, 1, 2, 3]])), ("vertex", np.array([[4]]))]
        m = meshio.Mesh(nodes, cells)
        got = mesh_io.mesh_from_meshio(m, dim=2)
        assert len(got.nodes) == 4, "the orphan marker point (node 4) must be dropped"
        assert got.elements.max() == 3
        print(f"  OK: 5 raw nodes -> {len(got.nodes)} kept, orphan point dropped")

    def test_dim_mismatch_raises_clear_error(self):
        print("=" * 70)
        print("CHECK 5: requesting a dim with no matching cells raises, rather")
        print("than silently returning an empty/wrong mesh")
        print("=" * 70)
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        m = meshio.Mesh(nodes, [("quad", np.array([[0, 1, 2, 3]]))])
        with pytest.raises(RuntimeError, match="no cells of topological dimension"):
            mesh_io.mesh_from_meshio(m, dim=3)
        print("  OK: RuntimeError raised as documented")

    def test_bad_dim_argument_raises(self):
        m = meshio.Mesh(np.zeros((1, 3)), [("vertex", np.array([[0]]))])
        with pytest.raises(ValueError):
            mesh_io.mesh_from_meshio(m, dim=4)

    def test_multiblockmesh_unknown_block_name_raises_on_write(self):
        print("=" * 70)
        print("CHECK 6: a MultiBlockMesh whose block keys don't match this")
        print("module's known short topology names fails clearly on write,")
        print("rather than silently mis-mapping to the wrong meshio cell type")
        print("=" * 70)
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        mb = MultiBlockMesh(nodes, {"not_a_real_block_name": np.array([[0, 1, 2, 3]])}, dim=2)
        with pytest.raises(ValueError, match="not one of"):
            mesh_io.mesh_to_meshio(mb)
        print("  OK: ValueError raised as documented")
