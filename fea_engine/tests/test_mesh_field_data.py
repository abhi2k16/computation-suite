"""
test_mesh_field_data.py -- Wave 14 items 121 and 123 (docs/consolidated_
future_roadmap.md), scoped directly against TensorMesh's own Meshes
documentation page (https://docs.tensor-mesh.com/user_guide/meshes.html):

    item 121: point_data/cell_data/field_data attached directly onto
              Mesh/MultiBlockMesh, plus the register_point_data()/
              register_element_data() convenience methods.
    item 123: Mesh.compute_boundary_mask()/MultiBlockMesh.compute_
              boundary_mask() -- topology-derived (edge-/face-adjacency)
              boundary-node detection, working on any mesh shape, not
              just axis/plane-aligned boundaries the way nodes_on_line()/
              nodes_on_plane() are restricted to.

Both are purely additive: every pre-existing Mesh/MultiBlockMesh caller
in this package (the entire codebase before this wave) only ever touches
.nodes/.elements/.blocks/.dim, none of which changed shape or meaning
here.
"""
import numpy as np
import pytest
from fea_engine.mesh import (
    Mesh, MultiBlockMesh, rectangle_mesh, box_mesh, line_mesh,
    rectangle_with_hole_mesh_quarter, weld_meshes,
)


# =====================================================================
# Item 121: point_data / cell_data / field_data
# =====================================================================
class TestFieldDataAttachment:
    def test_mesh_defaults_to_empty_dicts(self):
        print("=" * 70)
        print("CHECK 1: a plain Mesh(nodes, elements, dim) call (no field-data")
        print("args) still works, with empty dicts, not shared mutable state")
        print("=" * 70)
        m1 = rectangle_mesh(1.0, 1.0, 2, 2)
        m2 = rectangle_mesh(1.0, 1.0, 2, 2)
        assert m1.point_data == {} and m1.cell_data == {} and m1.field_data == {}
        m1.point_data["u"] = np.zeros(len(m1.nodes))
        # the classic Python mutable-default-argument bug would leak this
        # into m2's own point_data too -- confirm it does NOT.
        assert m2.point_data == {}, "point_data dict leaked across Mesh instances!"
        print("  OK: independent empty dicts per instance")

    def test_register_point_data_length_check_and_chaining(self):
        print("=" * 70)
        print("CHECK 2: register_point_data() validates length, stores under")
        print("the given name, and returns self (chainable)")
        print("=" * 70)
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        n = len(m.nodes)
        u = np.linspace(0, 1, n)
        v = np.linspace(1, 2, n)
        result = m.register_point_data("u", u).register_point_data("v", v)
        assert result is m, "register_point_data must return self for chaining"
        assert np.array_equal(m.point_data["u"], u)
        assert np.array_equal(m.point_data["v"], v)
        with pytest.raises(ValueError):
            m.register_point_data("bad", np.zeros(n - 1))
        print(f"  OK: {list(m.point_data)} registered, wrong-length input rejected")

    def test_register_element_data_length_check(self):
        print("=" * 70)
        print("CHECK 3: register_element_data() validates against len(elements),")
        print("not len(nodes)")
        print("=" * 70)
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        n_elem = len(m.elements)
        energy = np.arange(n_elem, dtype=float)
        m.register_element_data("strain_energy", energy)
        assert np.array_equal(m.cell_data["strain_energy"], energy)
        with pytest.raises(ValueError):
            m.register_element_data("bad", np.zeros(n_elem + 1))
        print(f"  OK: cell_data={list(m.cell_data)}, mismatched length rejected")

    def test_multiblockmesh_point_data_shared_cell_data_nested_by_block(self):
        print("=" * 70)
        print("CHECK 4: MultiBlockMesh.point_data is shared (one node array);")
        print("cell_data is nested {block_name: {field: array}}")
        print("=" * 70)
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        mb = MultiBlockMesh(m.nodes.copy(), {"quad4": m.elements.copy()}, dim=2)
        mb.register_point_data("u", np.zeros(len(mb.nodes)))
        assert "u" in mb.point_data
        mb.register_element_data("quad4", "energy", np.zeros(len(mb.blocks["quad4"])))
        assert mb.cell_data["quad4"]["energy"].shape == (len(mb.blocks["quad4"]),)
        with pytest.raises(KeyError):
            mb.register_element_data("no_such_block", "x", np.zeros(1))
        with pytest.raises(ValueError):
            mb.register_element_data("quad4", "bad", np.zeros(999))
        print("  OK: nested cell_data, unknown-block and wrong-length both rejected")


# =====================================================================
# Item 123: topology-derived boundary-node detection
# =====================================================================
class TestBoundaryMask:
    def test_2d_uniform_grid_exact_boundary_count(self):
        print("=" * 70)
        print("CHECK 1: a 2x2-element (3x3-node) rectangle_mesh has exactly")
        print("8 boundary nodes and 1 interior node (the center)")
        print("=" * 70)
        m = rectangle_mesh(1.0, 1.0, 2, 2)
        mask = m.compute_boundary_mask()
        assert mask.dtype == bool
        assert mask.sum() == 8
        center_id = 1 * 3 + 1  # node_id(i=1, j=1) in rectangle_mesh's own indexing
        assert not mask[center_id]
        assert m.point_data["is_boundary"] is mask
        print(f"  OK: {mask.sum()}/9 boundary, center node {center_id} correctly interior")

    def test_2d_matches_nodes_on_line_on_axis_aligned_rectangle(self):
        print("=" * 70)
        print("CHECK 2: on a plain axis-aligned rectangle, compute_boundary_mask()'s")
        print("flagged set equals the union of all 4 nodes_on_line() edges -- the")
        print("topology-derived and coordinate-derived answers must agree on a")
        print("shape both methods can handle")
        print("=" * 70)
        m = rectangle_mesh(2.0, 1.0, 5, 4)
        topo_mask = m.compute_boundary_mask()
        coord_mask = np.zeros(len(m.nodes), dtype=bool)
        for idx in (m.nodes_on_line(0, 0.0), m.nodes_on_line(0, 2.0),
                    m.nodes_on_line(1, 0.0), m.nodes_on_line(1, 1.0)):
            coord_mask[idx] = True
        assert np.array_equal(topo_mask, coord_mask)
        print(f"  OK: {topo_mask.sum()} boundary nodes, exact agreement")

    def test_2d_curved_boundary_not_reachable_by_nodes_on_line(self):
        print("=" * 70)
        print("CHECK 3: on a mesh with a genuinely curved boundary (the near-hole")
        print("quarter block), compute_boundary_mask() still finds a sane boundary")
        print("set -- this is precisely the case nodes_on_line()/nodes_on_plane()")
        print("CANNOT handle (no single coordinate value describes an arc)")
        print("=" * 70)
        m = rectangle_with_hole_mesh_quarter(a=2.0, b=1.0, R=0.3, nr=4, ntheta=6)
        mask = m.compute_boundary_mask()
        # every node on the inner (hole) arc and the outer mapped boundary
        # should be flagged; interior (non-edge) nodes should not.
        assert 0 < mask.sum() < len(m.nodes)
        # the hole boundary is i=0 in rectangle_with_hole_mesh_quarter's own
        # indexing (node_id(0, j) for all j) -- confirm every one of those
        # is flagged, since it's a genuine mesh boundary (nothing inside it).
        n_ny = 6 + 1
        hole_nodes = [0 * n_ny + j for j in range(n_ny)]
        assert mask[hole_nodes].all(), "every hole-arc node must be on the boundary"
        print(f"  OK: {mask.sum()}/{len(m.nodes)} boundary nodes, all hole-arc nodes included")

    def test_3d_box_exact_boundary_count(self):
        print("=" * 70)
        print("CHECK 4: a 2x2x2-element (3x3x3-node) box_mesh has exactly 26")
        print("boundary nodes and 1 interior node (the very center)")
        print("=" * 70)
        m = box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
        mask = m.compute_boundary_mask()
        assert mask.sum() == 26
        print(f"  OK: {mask.sum()}/27 boundary")

    def test_1d_line_mesh_only_the_two_ends(self):
        print("=" * 70)
        print("CHECK 5: a 1-D line_mesh flags exactly its two end nodes")
        print("=" * 70)
        m = line_mesh(2.0, 5)
        mask = m.compute_boundary_mask()
        assert mask.sum() == 2
        assert mask[0] and mask[-1]
        assert not mask[1:-1].any()
        print("  OK: exactly the two endpoints")

    def test_multiblockmesh_boundary_across_two_welded_blocks(self):
        print("=" * 70)
        print("CHECK 6: MultiBlockMesh.compute_boundary_mask() correctly treats")
        print("an edge shared between TWO DIFFERENT blocks as interior (not")
        print("boundary) -- the real reason this needs one shared adjacency map")
        print("across all blocks, not one map per block")
        print("=" * 70)
        left = rectangle_mesh(1.0, 1.0, 2, 2)
        right = rectangle_mesh(1.0, 1.0, 2, 2, x0=1.0)
        combined = weld_meshes(left, right)
        # as a MultiBlockMesh with the SAME elements split into two
        # differently-named blocks straddling the weld seam, to genuinely
        # exercise the cross-block adjacency path (not just reproduce the
        # single-block Mesh case with a different container type).
        half = len(combined.elements) // 2
        mb = MultiBlockMesh(
            combined.nodes.copy(),
            {"block_a": combined.elements[:half].copy(),
             "block_b": combined.elements[half:].copy()},
            dim=2)
        mesh_equiv = Mesh(combined.nodes.copy(), combined.elements.copy(), dim=2)
        mask_mb = mb.compute_boundary_mask()
        mask_mesh = mesh_equiv.compute_boundary_mask()
        assert np.array_equal(mask_mb, mask_mesh), (
            "splitting the SAME connectivity across two MultiBlockMesh blocks "
            "must not change which nodes are flagged boundary")
        # sanity: the welded seam (x=1.0) is only genuinely INTERIOR away
        # from the top/bottom edges -- its own endpoints (y=0, y=1) sit on
        # the combined rectangle's real outer boundary regardless of the
        # seam, so only the strictly-interior seam node (y=0.5, touched by
        # elements from BOTH blocks on both sides) should read as False.
        seam_interior_node = combined.nodes_on_line(0, 1.0)[
            np.isclose(combined.nodes[combined.nodes_on_line(0, 1.0), 1], 0.5)]
        assert len(seam_interior_node) == 1
        assert not mask_mb[seam_interior_node[0]], (
            "the strictly-interior welded seam node must not be boundary")
        print(f"  OK: {mask_mb.sum()} boundary nodes, identical across Mesh/MultiBlockMesh, "
              f"welded seam correctly interior")
