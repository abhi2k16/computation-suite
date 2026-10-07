"""
test_grading_3d.py -- Wave 0 item 8 (docs/consolidated_future_roadmap.
md, source generalized_mesh_grading_roadmap.md): validates the 3-D
extension of Mesh.check_grading(), which previously raised
NotImplementedError unconditionally for any 3-D mesh (no per-topology
face-adjacency table existed yet).

Covers, in order: the raw volume helpers directly (closed-form
correctness on a known box/tet, independent of any mesh-adjacency
bookkeeping), Hex8 grading on a uniform and a deliberately-graded box
mesh, Tet4 grading on a hand-built two-tet mesh, the Tet10/Hex20
corner-truncation reuse (see _TET4_FACES/_HEX8_FACES's module
docstring in mesh.py for why this is expected to just work), and the
still-correctly-raised unsupported-topology case (so the new support
for Tet4/Hex8/Tet10/Hex20 didn't accidentally turn into "never raises
at all").
"""
import numpy as np
import pytest
from fea_engine.mesh import (
    Mesh, box_mesh, rectangle_mesh, extrude_mesh,
    _tet_volume, _hex_volume, _elem_size_3d, _elem_faces_3d,
)


def test_hex_volume_matches_closed_form_box_volume():
    print("=" * 70)
    print("CHECK 1: _hex_volume() matches a*b*c for an axis-aligned box,")
    print("including a NON-cube (anisotropic) box")
    print("=" * 70)
    a, b, c = 2.0, 3.0, 0.5
    pts = np.array([
        (0, 0, 0), (a, 0, 0), (a, b, 0), (0, b, 0),
        (0, 0, c), (a, 0, c), (a, b, c), (0, b, c)], dtype=float)
    vol = _hex_volume(pts)
    print(f"  computed volume = {vol:.6f}, expected = {a*b*c:.6f}")
    assert vol == pytest.approx(a * b * c, rel=1e-12)

    # translated far from the origin -- the divergence-theorem formula
    # must be translation-invariant for a closed, watertight boundary
    pts_shifted = pts + np.array([100.0, -50.0, 7.0])
    vol_shifted = _hex_volume(pts_shifted)
    print(f"  translated-box volume = {vol_shifted:.6f} (must be unchanged)")
    assert vol_shifted == pytest.approx(a * b * c, rel=1e-12)
    print("  PASS")


def test_tet_volume_matches_closed_form():
    print()
    print("=" * 70)
    print("CHECK 2: _tet_volume() matches the known 1/6 reference-tet volume,")
    print("and scales correctly for a stretched tet")
    print("=" * 70)
    ref = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=float)
    vol = _tet_volume(ref)
    print(f"  reference tet volume = {vol:.6f}, expected = {1/6:.6f}")
    assert vol == pytest.approx(1.0 / 6.0, rel=1e-12)

    stretched = ref * np.array([2.0, 3.0, 4.0])   # scales volume by 2*3*4=24
    vol_s = _tet_volume(stretched)
    print(f"  stretched tet volume = {vol_s:.6f}, expected = {24/6:.6f}")
    assert vol_s == pytest.approx(24.0 / 6.0, rel=1e-12)
    print("  PASS")


def test_check_grading_hex8_uniform_box_ratio_near_one():
    print()
    print("=" * 70)
    print("CHECK 3: check_grading() on a uniform Hex8 box mesh -> ratio ~ 1")
    print("=" * 70)
    m = box_mesh(1.0, 1.0, 1.0, 4, 4, 4)
    ratio = m.check_grading(verbose=False)
    print(f"  realized ratio = {ratio:.6f}")
    assert ratio == pytest.approx(1.0, abs=1e-9)
    print("  PASS")


def test_check_grading_hex8_detects_size_jump():
    print()
    print("=" * 70)
    print("CHECK 4: check_grading() on TWO Hex8 boxes of different height")
    print("sharing a face -- ratio must match the known size ratio")
    print("=" * 70)
    # Box A: unit cube. Box B: same footprint, HALF the height, stacked
    # directly on top -- they share the top face of A / bottom face of B.
    nodesA = np.array([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], dtype=float)
    nodesB_top = nodesA[4:8].copy()
    nodesB_top[:, 2] = 1.5   # box B has height 0.5, half of box A's height 1.0
    nodes = np.vstack([nodesA, nodesB_top])
    elemA = [0, 1, 2, 3, 4, 5, 6, 7]
    elemB = [4, 5, 6, 7, 8, 9, 10, 11]
    m = Mesh(nodes=nodes, elements=np.array([elemA, elemB]), dim=3)

    volA = _elem_size_3d(np.array(elemA), m.nodes) ** 3
    volB = _elem_size_3d(np.array(elemB), m.nodes) ** 3
    print(f"  volume A = {volA:.6f} (expect 1.0), volume B = {volB:.6f} (expect 0.5)")
    assert volA == pytest.approx(1.0, rel=1e-9)
    assert volB == pytest.approx(0.5, rel=1e-9)

    ratio = m.check_grading(verbose=False)
    expected_ratio = (volA ** (1 / 3)) / (volB ** (1 / 3))
    print(f"  realized check_grading() ratio = {ratio:.6f}, expected = {expected_ratio:.6f}")
    assert ratio == pytest.approx(expected_ratio, rel=1e-9)

    assert m.check_grading(growth_ratio_cap=expected_ratio * 1.01, verbose=False) is True
    assert m.check_grading(growth_ratio_cap=expected_ratio * 0.99, verbose=False) is False
    print("  PASS -- realized ratio matches the known volume ratio, and the"
          " cap pass/fail dispatch is correct on both sides")


def test_check_grading_tet4_hand_built_two_tet_mesh():
    print()
    print("=" * 70)
    print("CHECK 5: check_grading() on a hand-built Tet4 mesh -- two")
    print("reference tets of different size sharing a face")
    print("=" * 70)
    # Tet A: the standard unit reference tet.
    nodesA = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=float)
    # Tet B: shares face {1,2,3} (i.e. nodesA[1], nodesA[2], nodesA[3])
    # with Tet A, extended outward to a DIFFERENT (smaller) size by
    # picking a 4th point on the opposite side, scaled down.
    apex_B = np.array([-0.3, -0.3, -0.3])
    nodes = np.vstack([nodesA, apex_B])
    elemA = [0, 1, 2, 3]
    elemB = [1, 2, 3, 4]   # shares the face {1,2,3} with elemA
    m = Mesh(nodes=nodes, elements=np.array([elemA, elemB]), dim=3)

    faces_A = _elem_faces_3d(np.array(elemA))
    faces_B = _elem_faces_3d(np.array(elemB))
    shared = set(faces_A) & set(faces_B)
    print(f"  shared face signature(s) between the two tets: {shared}")
    assert len(shared) == 1
    assert shared == {frozenset((1, 2, 3))}

    ratio = m.check_grading(verbose=False)
    volA = _tet_volume(nodes[elemA])
    volB = _tet_volume(nodes[elemB])
    expected_ratio = max(volA, volB) ** (1 / 3) / min(volA, volB) ** (1 / 3)
    print(f"  volA={volA:.6f}, volB={volB:.6f}, realized ratio={ratio:.6f}, "
          f"expected={expected_ratio:.6f}")
    assert ratio == pytest.approx(expected_ratio, rel=1e-9)
    print("  PASS -- face-adjacency correctly identified the single shared"
          " triangular face, and the size ratio matches the closed-form volumes")


def test_check_grading_quadratic_elements_reuse_corner_tables():
    print()
    print("=" * 70)
    print("CHECK 6: Tet10/Hex20 connectivity (extra mid-nodes appended after")
    print("the corners) reuses the SAME corner-only face/volume logic --")
    print("check_grading() must give the identical answer to the pure")
    print("corner (Tet4/Hex8) mesh, since extra columns are structurally")
    print("irrelevant to this diagnostic")
    print("=" * 70)
    m_hex8 = box_mesh(1.0, 1.0, 1.0, 3, 2, 2)
    ratio_hex8 = m_hex8.check_grading(verbose=False)

    # Pad every element's connectivity with 12 dummy extra columns (as if
    # it were Hex20 connectivity) -- the VALUES are irrelevant since
    # check_grading() must only ever look at elem[:8].
    dummy_cols = np.zeros((m_hex8.elements.shape[0], 12), dtype=int)
    elements_hex20_shaped = np.hstack([m_hex8.elements, dummy_cols])
    m_hex20_shaped = Mesh(nodes=m_hex8.nodes, elements=elements_hex20_shaped, dim=3)
    ratio_hex20_shaped = m_hex20_shaped.check_grading(verbose=False)

    print(f"  Hex8 ratio = {ratio_hex8:.9f}, Hex20-shaped ratio = {ratio_hex20_shaped:.9f}")
    assert ratio_hex20_shaped == pytest.approx(ratio_hex8, rel=1e-12)
    print("  PASS")


def test_check_grading_unsupported_topology_still_raises():
    print()
    print("=" * 70)
    print("CHECK 7: an unsupported 3-D element topology (e.g. a 6-node")
    print("wedge/prism -- not in this package's element library) still")
    print("raises NotImplementedError, not a silently wrong answer")
    print("=" * 70)
    nodes = np.random.default_rng(0).random((6, 3))
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2, 3, 4, 5]]), dim=3)
    with pytest.raises(NotImplementedError):
        m.check_grading(verbose=False)
    print("  PASS")


def test_check_grading_1d_still_raises():
    print()
    print("=" * 70)
    print("CHECK 8: a 1-D mesh raises NotImplementedError (not meaningfully")
    print("'graded' in the edge/face-adjacency sense this method uses)")
    print("=" * 70)
    from fea_engine.mesh import line_mesh
    m = line_mesh(1.0, 5)
    with pytest.raises(NotImplementedError):
        m.check_grading(verbose=False)
    print("  PASS")
