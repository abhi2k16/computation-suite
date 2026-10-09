# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_facet_loads.py -- Wave 15 item 127 (docs/consolidated_future_
roadmap.md, source: TensorMesh's "Elements and Quadrature" documentation
page's facet_quadrature()/nanson_scale() machinery): validates
facet_loads.py (the bounded, quadrature-based facet/traction loader)
and FESystem.add_consistent_facet_load().

Three layers of decisive checks, not "runs without raising":

1. TestFacetTablesAreCorrect -- re-derives, for every entry in
   EDGE_TABLES/FACE_TABLES, that (a) every parent-element node NOT
   listed for a facet has ~zero shape value there, and (b) the listed
   nodes' own values match the matching lower-dimensional element's
   (Tri3/Tri6/Quad4/Quad8's own already-validated) shape functions
   exactly -- the EXACT verification this project's own history
   (GMSH_NODE_ORDER, mesh_io.py's node-order finding) demands before
   trusting a hand-derived topology table, run here as a real,
   reproducible test rather than a one-off check while writing the
   module.

2. TestConsistentFacetLoadShares -- partition-of-unity/total-load
   conservation (sum(share) == traction * true physical measure) for
   every one of the six facet families, PLUS the genuinely decisive
   curved-edge claim: a bowed (non-straight) quadratic edge's total
   load CONVERGES, as quad_order increases, to the curve's own TRUE
   arc length (cross-checked against an independent fine-resolution
   numerical integral) and is measurably DIFFERENT from the straight-
   line chord length -- proof this is really integrating the curved
   geometry, not silently falling back to a linear approximation.

3. TestFESystemFacetLoad -- end-to-end FESystem.add_consistent_facet_
   load() usage, including an exact cross-check against add_consistent_
   edge_load() on a straight edge (both methods must agree to floating-
   point precision when the geometry is one both can handle).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import facet_loads
from fea_engine.elements.solids import (
    Tri3PlaneStress, Tri6PlaneStress, Quad4PlaneStress, Quad8PlaneStress,
    Tet4Solid3D, Tet10Solid3D, Hex8Solid3D, Hex20Solid3D)
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem


# =====================================================================
# 1. Facet table correctness
# =====================================================================
class TestFacetTablesAreCorrect:
    @pytest.mark.parametrize("formulation_cls", [
        Tri3PlaneStress, Tri6PlaneStress, Quad4PlaneStress, Quad8PlaneStress])
    def test_2d_edge_tables(self, formulation_cls):
        print("=" * 70)
        print(f"CHECK: {formulation_cls.__name__}'s EDGE_TABLES entries -- "
              f"off-edge nodes vanish, on-edge nodes match the facet family")
        print("=" * 70)
        elem = formulation_cls()
        n_nodes = elem.n_nodes
        rng = np.random.default_rng(hash(formulation_cls.__name__) % 2**32)
        for local_idx, family, natcoord_map in facet_loads.list_facets(elem):
            others = [i for i in range(n_nodes) if i not in local_idx]
            shape_fn = facet_loads._FACET_SHAPE_FNS[family]
            max_off = 0.0
            max_mismatch = 0.0
            for s in np.linspace(-0.9, 0.9, 7):
                xi, eta = natcoord_map(s)
                N, _ = elem.shape_and_derivs((xi, eta))
                max_off = max(max_off, np.max(np.abs(N[others])) if others else 0.0)
                N_facet, _ = shape_fn(s)
                max_mismatch = max(max_mismatch, np.max(np.abs(N[list(local_idx)] - N_facet)))
            print(f"  edge {local_idx} ({family}): off-edge-max={max_off:.2e}, "
                  f"facet-match-max={max_mismatch:.2e}")
            assert max_off < 1e-10
            assert max_mismatch < 1e-9

    @pytest.mark.parametrize("formulation_cls", [Tet4Solid3D, Tet10Solid3D])
    def test_tet_face_tables(self, formulation_cls):
        elem = formulation_cls()
        n_nodes = elem.n_nodes
        rng = np.random.default_rng(hash(formulation_cls.__name__) % 2**32)
        for local_idx, family, natcoord_map in facet_loads.list_facets(elem):
            others = [i for i in range(n_nodes) if i not in local_idx]
            shape_fn = facet_loads._FACET_SHAPE_FNS[family]
            max_off = 0.0
            max_mismatch = 0.0
            for _ in range(8):
                a = rng.uniform(0.05, 0.85)
                b = rng.uniform(0.05, 0.9 - a)
                r, s, t = natcoord_map(a, b)
                N, _ = elem.shape_and_derivs((r, s, t))
                max_off = max(max_off, np.max(np.abs(N[others])) if others else 0.0)
                N_facet, _ = shape_fn((a, b))
                max_mismatch = max(max_mismatch, np.max(np.abs(N[list(local_idx)] - N_facet)))
            assert max_off < 1e-10
            assert max_mismatch < 1e-9

    @pytest.mark.parametrize("formulation_cls", [Hex8Solid3D, Hex20Solid3D])
    def test_hex_face_tables(self, formulation_cls):
        elem = formulation_cls()
        n_nodes = elem.n_nodes
        rng = np.random.default_rng(hash(formulation_cls.__name__) % 2**32)
        for local_idx, family, natcoord_map in facet_loads.list_facets(elem):
            others = [i for i in range(n_nodes) if i not in local_idx]
            shape_fn = facet_loads._FACET_SHAPE_FNS[family]
            max_off = 0.0
            max_mismatch = 0.0
            for _ in range(8):
                a, b = rng.uniform(-0.9, 0.9, 2)
                nc = natcoord_map(a, b)
                N, _ = elem.shape_and_derivs(nc)
                max_off = max(max_off, np.max(np.abs(N[others])) if others else 0.0)
                N_facet, _ = shape_fn((a, b))
                max_mismatch = max(max_mismatch, np.max(np.abs(N[list(local_idx)] - N_facet)))
            assert max_off < 1e-10
            assert max_mismatch < 1e-9


# =====================================================================
# 2. consistent_facet_load_shares() -- geometric decisiveness
# =====================================================================
class TestConsistentFacetLoadShares:
    def test_line2_straight_edge_total_equals_traction_times_length(self):
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])   # length 5
        share = facet_loads.consistent_facet_load_shares(2, coords, traction=2.0, quad_order=2)
        assert share.sum() == pytest.approx(10.0, abs=1e-10)
        assert np.all(share > 0)

    def test_line3_straight_edge_total_equals_traction_times_length(self):
        coords = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 0.0]])   # straight, mid at chord midpoint
        share = facet_loads.consistent_facet_load_shares(2, coords, traction=1.0, quad_order=2)
        assert share.sum() == pytest.approx(4.0, abs=1e-9)

    def test_curved_line3_edge_converges_to_true_arc_length_not_chord(self):
        print("=" * 70)
        print("CHECK: a BOWED quadratic edge (mid-node off the chord midpoint)")
        print("has a total consistent load that CONVERGES, as quad_order")
        print("increases, to the curve's own true arc length -- cross-checked")
        print("against an independent fine-resolution numerical integral --")
        print("and is measurably different from the straight-line chord")
        print("=" * 70)
        coords = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 1.0]])   # bowed
        chord_length = 4.0

        totals = {}
        for qo in (2, 6, 10):
            share = facet_loads.consistent_facet_load_shares(2, coords, traction=1.0, quad_order=qo)
            totals[qo] = share.sum()
        print(f"  totals by quad_order: {totals}")

        # Independent reference: fine-resolution numerical arc-length
        # integral of the SAME quadratic parametrization, not reusing
        # this module's own quadrature machinery at all.
        s = np.linspace(-1, 1, 200001)
        N0, N1, N2 = s * (s - 1) / 2, s * (s + 1) / 2, 1 - s ** 2
        x = N1 * 4.0 + N2 * 2.0
        y = N2 * 1.0
        dx = np.gradient(x, s)
        dy = np.gradient(y, s)
        ref_length = np.trapezoid(np.sqrt(dx ** 2 + dy ** 2), s) if hasattr(np, "trapezoid") \
            else np.trapz(np.sqrt(dx ** 2 + dy ** 2), s)
        print(f"  independent fine-resolution reference length = {ref_length:.6f}")

        assert totals[10] == pytest.approx(ref_length, abs=1e-3)
        assert abs(totals[6] - totals[10]) < 1e-3, "should have converged by quad_order=6"
        assert abs(totals[10] - chord_length) > 0.1, \
            "curved-edge total must differ meaningfully from the straight chord"

    def test_tri3_flat_face_total_equals_traction_times_area(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        share = facet_loads.consistent_facet_load_shares(3, coords, traction=3.0, quad_order=2)
        assert share.sum() == pytest.approx(3.0 * 0.5, abs=1e-10)

    def test_quad4_flat_face_total_equals_traction_times_area(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                            [2.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        share = facet_loads.consistent_facet_load_shares(3, coords, traction=1.5, quad_order=2)
        assert share.sum() == pytest.approx(1.5 * 6.0, abs=1e-9)

    def test_tri6_flat_face_total_equals_traction_times_area(self):
        c0, c1, c2 = np.array([0., 0, 0]), np.array([2., 0, 0]), np.array([0., 3, 0])
        coords = np.array([c0, c1, c2, (c0 + c1) / 2, (c1 + c2) / 2, (c2 + c0) / 2])
        share = facet_loads.consistent_facet_load_shares(3, coords, traction=1.0, quad_order=2)
        true_area = 0.5 * 2.0 * 3.0
        assert share.sum() == pytest.approx(true_area, abs=1e-9)

    def test_quad8_flat_face_total_equals_traction_times_area(self):
        c0, c1, c2, c3 = (np.array([0., 0, 0]), np.array([2., 0, 0]),
                           np.array([2., 3, 0]), np.array([0., 3, 0]))
        coords = np.array([c0, c1, c2, c3, (c0 + c1) / 2, (c1 + c2) / 2,
                            (c2 + c3) / 2, (c3 + c0) / 2])
        share = facet_loads.consistent_facet_load_shares(3, coords, traction=1.0, quad_order=2)
        assert share.sum() == pytest.approx(6.0, abs=1e-9)

    def test_bad_facet_family_raises_clear_error(self):
        coords = np.zeros((5, 3))   # 5 nodes -- not a recognized facet family
        with pytest.raises(ValueError, match="no known facet family"):
            facet_loads.consistent_facet_load_shares(3, coords, traction=1.0)


# =====================================================================
# 3. FESystem.add_consistent_facet_load()
# =====================================================================
class TestFESystemFacetLoad:
    def test_matches_add_consistent_edge_load_exactly_on_a_straight_edge(self):
        nodes = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 0.0]])
        conn = np.array([[0, 1, 2]])
        mesh = Mesh(nodes, conn, dim=2)
        elem = Tri3PlaneStress()

        fes_old = FESystem(mesh, elem)
        fes_old.add_consistent_edge_load([(0, 1)], dof_index=0, traction=2.5)

        fes_new = FESystem(mesh, elem)
        fes_new.add_consistent_facet_load(elem, [(0, 1)], dof_index=0, traction=2.5)

        assert np.allclose(fes_old.F, fes_new.F, atol=1e-12)

    def test_quad8_curved_edge_end_to_end_via_fesystem(self):
        print("=" * 70)
        print("CHECK: FESystem.add_consistent_facet_load() end-to-end on a")
        print("real Quad8PlaneStress element with one bowed (curved) edge")
        print("=" * 70)
        # A Quad8 with corners at a unit square, but the bottom edge's
        # mid-node bowed downward (a curved bottom edge).
        nodes = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1],        # corners 0-3
            [0.5, -0.2], [1, 0.5], [0.5, 1], [0, 0.5],   # mid-edges 4-7
        ], dtype=float)
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        mesh = Mesh(nodes, conn, dim=2)
        elem = Quad8PlaneStress()
        fes = FESystem(mesh, elem)

        facets = []
        for local_idx, family, _ in facet_loads.list_facets(elem):
            if family == "line3" and set(local_idx) == {0, 1, 4}:
                facets.append(tuple(conn[0, i] for i in local_idx))
        assert len(facets) == 1, "expected exactly the bottom (0,1,4) edge"

        fes.add_consistent_facet_load(elem, facets, dof_index=1, traction=1.0)
        total = fes.F.sum()
        print(f"  total consistent load on the bowed bottom edge = {total:.6f}")
        # Must exceed the straight chord length (1.0) since the edge bows outward.
        assert total > 1.0
        assert total < 1.5   # sanity upper bound -- not wildly wrong
        # Only the three bottom-edge nodes (0, 1, 4) should carry any load.
        for node_id in (0, 1, 4):
            assert fes.F[2 * node_id + 1] > 0
        for node_id in (2, 3, 5, 6, 7):
            assert fes.F[2 * node_id + 1] == 0.0

    def test_hex20_face_end_to_end_via_fesystem(self):
        print("=" * 70)
        print("CHECK: FESystem.add_consistent_facet_load() end-to-end on a")
        print("real Hex20Solid3D element's flat top face -- total load must")
        print("equal traction * true face area (a decisive, independently")
        print("computable target, not just 'ran without raising')")
        print("=" * 70)
        xi_i = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1, -1, 1, 1, -1], dtype=float)
        eta_i = np.array([-1, -1, 1, 1, -1, -1, 1, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1, -1, 1, 1], dtype=float)
        zeta_i = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
        Lx, Ly, Lz = 2.0, 3.0, 1.0
        nodes = np.column_stack([
            (xi_i + 1) / 2 * Lx, (eta_i + 1) / 2 * Ly, (zeta_i + 1) / 2 * Lz])
        conn = np.array([list(range(20))])
        mesh = Mesh(nodes, conn, dim=3)
        elem = Hex20Solid3D()
        fes = FESystem(mesh, elem)

        top_face = [entry for entry in facet_loads.list_facets(elem)
                    if entry[0] == (4, 5, 6, 7, 12, 13, 14, 15)][0]
        local_idx = top_face[0]
        facets = [tuple(conn[0, i] for i in local_idx)]
        fes.add_consistent_facet_load(elem, facets, dof_index=2, traction=5.0)
        total = fes.F.sum()
        true_area = Lx * Ly
        print(f"  total load = {total:.6f}, expected = traction*area = {5.0*true_area:.6f}")
        assert total == pytest.approx(5.0 * true_area, abs=1e-8)
