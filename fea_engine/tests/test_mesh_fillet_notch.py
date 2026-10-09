"""
tests/test_mesh_fillet_notch.py -- Wave 5 items 26/27 (docs/
consolidated_future_roadmap.md): grading.Notch (real implementation,
replacing the previous MeshGradingError-on-construction placeholder)
and mesh.fillet_in_rectangle_mesh()/notch_in_rectangle_mesh() (the
structured-path Fillet/Notch support docs/generalized_mesh_grading_
roadmap.md's Phase 2 explicitly deferred).

Checks are geometry-level (no element formulation beyond Quad4PlaneStress
for check_quality(), no solver): mesh area via the shoelace formula
against each shape's exact analytic target, element-inversion-freedom,
and -- since fillet_in_rectangle_mesh()/notch_in_rectangle_mesh() both
rely on hand-derived orientation/winding logic for their reflected and
axis-swapped corner/edge variants (see each function's own docstring) --
every one of the 4 corners/edges is exercised, not just the canonical
one, so a winding bug in any single reflection path would show up as a
check_quality() failure (inverted elements) rather than being silently
missed."""
__author__ = "Abhijeet"
import numpy as np
import pytest
from scipy.spatial import cKDTree

from fea_engine.mesh import (
    fillet_in_rectangle_mesh, notch_in_rectangle_mesh, _quarter_disk_sector_mesh,
)
from fea_engine.grading import Notch, Fillet, MeshGradingError
from fea_engine.elements import Quad4PlaneStress


def _quad_area(nodes, elem):
    x, y = nodes[elem, 0], nodes[elem, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _total_area(mesh):
    return sum(_quad_area(mesh.nodes, e) for e in mesh.elements)


def _min_nonzero_pairwise_dist(nodes):
    tree = cKDTree(nodes)
    dist, _ = tree.query(nodes, k=2)
    return dist[:, 1].min()


ELEM = Quad4PlaneStress()


# =====================================================================
# grading.Notch: real implementation
# =====================================================================
def test_notch_dataclass_now_constructs():
    n = Notch(path=[(0.0, 1.0), (0.4, 1.0)], depth=0.2, n_ring=4)
    assert n.h_min == pytest.approx(0.4 / 4)


def test_notch_rejects_wrong_path_length():
    with pytest.raises(MeshGradingError, match="exactly 2 points"):
        Notch(path=[(0.0, 1.0), (0.2, 1.0), (0.4, 1.0)], depth=0.2)
    with pytest.raises(MeshGradingError, match="exactly 2 points"):
        Notch(path=[(0.0, 1.0)], depth=0.2)


def test_notch_rejects_degenerate_or_bad_geometry():
    with pytest.raises(MeshGradingError, match="distinct"):
        Notch(path=[(0.1, 1.0), (0.1, 1.0)], depth=0.2)
    with pytest.raises(MeshGradingError, match="depth"):
        Notch(path=[(0.0, 1.0), (0.4, 1.0)], depth=0.0)


# =====================================================================
# _quarter_disk_sector_mesh: the fillet corner's collapsed-polar block
# =====================================================================
def test_quarter_disk_sector_quality_and_area():
    R = 0.5
    for nr, ntheta in [(3, 5), (8, 10)]:
        m = _quarter_disk_sector_mesh(R, nr, ntheta)
        assert m.check_quality(ELEM, verbose=False)
        area = _total_area(m)
        target = np.pi * R ** 2 / 4.0
        # Polygonal chord approximation of the arc slightly under-covers
        # the true quarter-disk; converges as ntheta grows.
        assert area < target
        assert abs(area - target) / target < 0.05


def test_quarter_disk_sector_center_node_is_shared():
    m = _quarter_disk_sector_mesh(0.3, nr=4, ntheta=6)
    # every element touching i=0 shares node 0 as two of its own corners
    n_touching = sum(1 for e in m.elements if (e == 0).sum() == 2)
    assert n_touching == 6   # exactly ntheta elements are collapsed


# =====================================================================
# fillet_in_rectangle_mesh: all 4 corners
# =====================================================================
@pytest.mark.parametrize("corner", [(2.0, 1.5), (0.0, 1.5), (0.0, 0.0), (2.0, 0.0)])
def test_fillet_area_and_quality_all_corners(corner):
    Lx, Ly, R = 2.0, 1.5, 0.3
    m = fillet_in_rectangle_mesh(Lx, Ly, corner, R, nx=8, ny=6, nr=6, ntheta=8)
    assert m.check_quality(ELEM, verbose=False)
    area = _total_area(m)
    expected = Lx * Ly - (R ** 2 - np.pi * R ** 2 / 4.0)
    assert abs(area - expected) / expected < 1e-3
    assert _min_nonzero_pairwise_dist(m.nodes) > 1e-6   # no failed-weld duplicates


def test_fillet_area_converges_with_refinement():
    Lx, Ly, R = 2.0, 1.5, 0.3
    expected = Lx * Ly - (R ** 2 - np.pi * R ** 2 / 4.0)
    errs = []
    for nr, ntheta in [(4, 6), (8, 12), (16, 24)]:
        m = fillet_in_rectangle_mesh(Lx, Ly, (Lx, Ly), R, nx=8, ny=6, nr=nr, ntheta=ntheta)
        errs.append(abs(_total_area(m) - expected) / expected)
    assert errs[1] < errs[0]
    assert errs[2] < errs[1]


def test_fillet_rejects_radius_too_large():
    with pytest.raises(ValueError, match="strictly less than"):
        fillet_in_rectangle_mesh(1.0, 1.0, (1.0, 1.0), R=1.0, nx=4, ny=4, nr=4)


def test_fillet_rejects_non_corner_point():
    with pytest.raises(ValueError, match="does not match"):
        fillet_in_rectangle_mesh(1.0, 1.0, (0.5, 0.5), R=0.2, nx=4, ny=4, nr=4)


# =====================================================================
# notch_in_rectangle_mesh: all 4 edges
# =====================================================================
@pytest.mark.parametrize("edge", ["top", "bottom", "left", "right"])
def test_notch_area_and_quality_all_edges(edge):
    Lx, Ly = 2.0, 1.5
    span = Lx if edge in ("top", "bottom") else Ly
    perp = Ly if edge in ("top", "bottom") else Lx
    s0, s1, depth = 0.3 * span, 0.6 * span, 0.4 * perp
    m = notch_in_rectangle_mesh(Lx, Ly, edge, s0, s1, depth,
                                 n_left=4, n_mid=3, n_right=4, n_depth=5, n_wall=3)
    assert m.check_quality(ELEM, verbose=False)
    area = _total_area(m)
    expected = Lx * Ly - (s1 - s0) * depth
    assert area == pytest.approx(expected, rel=1e-10)   # all-straight edges: exact
    assert _min_nonzero_pairwise_dist(m.nodes) > 1e-6


def test_notch_all_four_edges_give_identical_area_and_count():
    """A symmetric setup (mouth/depth chosen proportionally to each
    edge's own span/perp) should give IDENTICAL element/node counts and
    area regardless of which edge -- a direct cross-check that the
    rotation ("left") and axis-swap ("right") mappings are geometrically
    equivalent to the plain/reflected ones ("top"/"bottom"), not just
    each individually plausible."""
    Lx, Ly = 2.0, 1.5
    results = {}
    for edge in ["top", "bottom", "left", "right"]:
        span = Lx if edge in ("top", "bottom") else Ly
        perp = Ly if edge in ("top", "bottom") else Lx
        s0, s1, depth = 0.3 * span, 0.6 * span, 0.4 * perp
        m = notch_in_rectangle_mesh(Lx, Ly, edge, s0, s1, depth,
                                     n_left=4, n_mid=3, n_right=4, n_depth=5, n_wall=3)
        results[edge] = (len(m.elements), len(m.nodes), _total_area(m))
    values = list(results.values())
    for v in values[1:]:
        assert v[0] == values[0][0]
        assert v[1] == values[0][1]
        assert v[2] == pytest.approx(values[0][2], rel=1e-10)


def test_notch_rejects_bad_geometry():
    with pytest.raises(ValueError, match="edge must be"):
        notch_in_rectangle_mesh(1.0, 1.0, "diagonal", 0.2, 0.5, 0.2,
                                 n_left=2, n_mid=2, n_right=2, n_depth=2, n_wall=2)
    with pytest.raises(ValueError, match="mouth"):
        notch_in_rectangle_mesh(1.0, 1.0, "top", 0.6, 0.2, 0.2,   # s0 > s1
                                 n_left=2, n_mid=2, n_right=2, n_depth=2, n_wall=2)
    with pytest.raises(ValueError, match="depth"):
        notch_in_rectangle_mesh(1.0, 1.0, "top", 0.2, 0.6, 1.5,   # depth > Ly
                                 n_left=2, n_mid=2, n_right=2, n_depth=2, n_wall=2)
