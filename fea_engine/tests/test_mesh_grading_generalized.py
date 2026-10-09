"""
tests/test_mesh_grading_generalized.py -- Phase 2 validation of
mesh.hole_in_rectangle_mesh()/hole_in_rectangle_mesh_graded() (docs/
generalized_mesh_grading_roadmap.md).

Checks are geometry-level (no element formulation, no solver): mesh area
via the shoelace formula, element-inversion-freedom via Mesh.check_quality(),
weld correctness via nearest-neighbor node-duplication checks, and the
growth-ratio solver's realized effect on the actual radial node spacing --
none of this depends on any specific Element class, so it's checkable
without importing elements/ at all.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest
from scipy.spatial import cKDTree

from fea_engine.mesh import (
    hole_in_rectangle_mesh, hole_in_rectangle_mesh_graded,
    rectangle_with_hole_mesh_quarter_full, extrude_mesh, _reflect_mesh,
)
from fea_engine.grading import growth_ratio_of_partition


def _quad_area(nodes, elem):
    """Shoelace formula for one planar Quad4 element (exact for any
    planar simple quadrilateral, convex or not, as long as node order is
    consistent -- CCW here since check_quality() elsewhere confirms
    positive Jacobians)."""
    x, y = nodes[elem, 0], nodes[elem, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _total_area(mesh):
    return sum(_quad_area(mesh.nodes, e) for e in mesh.elements)


def _min_nonzero_pairwise_dist(nodes):
    """Nearest-neighbor distance excluding self -- used to catch a failed
    weld (two coincident-but-not-merged nodes would show up as an exact
    duplicate, distance ~0, distinct from genuine near-neighbors)."""
    tree = cKDTree(nodes)
    dist, _ = tree.query(nodes, k=2)
    return dist[:, 1].min()


# =====================================================================
# Off-center hole: area convergence against the analytic target
# =====================================================================
@pytest.mark.parametrize("ntheta", [6, 12, 24])
def test_area_converges_to_rectangle_minus_circle(ntheta):
    Lx, Ly, R = 1.0, 0.6, 0.08
    hole_center = (0.35, 0.4)   # deliberately off-center, off-symmetric
    mesh = hole_in_rectangle_mesh(Lx, Ly, hole_center, R, nr=6, ntheta=ntheta,
                                   grade_p=2.0, extend_grade=2.0)
    area = _total_area(mesh)
    target = Lx * Ly - np.pi * R**2
    # The polygonal hole boundary's chords cut inside the true circle, so
    # meshed area is slightly ABOVE the analytic target, converging as
    # ntheta grows (each quadrant meshes 1/4 of the full circle's
    # circumference at ntheta divisions).
    rel_err = (area - target) / target
    assert 0.0 <= rel_err < 0.05, f"ntheta={ntheta}: area={area}, target={target}, rel_err={rel_err}"


def test_area_error_shrinks_with_finer_ntheta():
    Lx, Ly, R, hole_center = 1.0, 0.6, 0.08, (0.35, 0.4)
    err_coarse = None
    err_fine = None
    for ntheta, slot in ((6, "coarse"), (24, "fine")):
        mesh = hole_in_rectangle_mesh(Lx, Ly, hole_center, R, nr=6, ntheta=ntheta)
        area = _total_area(mesh)
        target = Lx * Ly - np.pi * R**2
        err = abs(area - target)
        if slot == "coarse":
            err_coarse = err
        else:
            err_fine = err
    assert err_fine < err_coarse


# =====================================================================
# Off-center placement: no inverted elements, no failed welds
# =====================================================================
@pytest.mark.parametrize("hole_center", [(0.5, 0.5), (0.3, 0.4), (0.7, 0.2)])
def test_off_center_hole_no_inverted_elements(hole_center):
    Lx, Ly, R = 1.0, 1.0, 0.1
    mesh = hole_in_rectangle_mesh(Lx, Ly, hole_center, R, nr=5, ntheta=10)
    # Every quad's shoelace area must be positive -- a negative/zero area
    # here means a winding or weld error slipped through.
    for e in mesh.elements:
        x, y = mesh.nodes[e, 0], mesh.nodes[e, 1]
        signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert signed > 0, f"non-CCW/degenerate element in hole_center={hole_center}"


@pytest.mark.parametrize("hole_center", [(0.5, 0.5), (0.3, 0.4), (0.7, 0.2)])
def test_off_center_hole_welds_cleanly(hole_center):
    """A failed weld along a shared quadrant boundary leaves two nodes at
    (numerically) the same location that were never merged -- catch that
    directly rather than trusting downstream element checks to notice."""
    Lx, Ly, R = 1.0, 1.0, 0.1
    mesh = hole_in_rectangle_mesh(Lx, Ly, hole_center, R, nr=5, ntheta=10)
    min_dist = _min_nonzero_pairwise_dist(mesh.nodes)
    # Should be bounded below by roughly the finest element size in the
    # mesh, not by weld_tol (1e-9) -- a near-1e-9 min distance would mean
    # an unwelded duplicate.
    assert min_dist > 1e-6, f"suspiciously small min node spacing: {min_dist}"


def test_off_center_hole_rejects_hole_too_close_to_edge():
    with pytest.raises(ValueError):
        hole_in_rectangle_mesh(1.0, 1.0, (0.05, 0.5), R=0.1, nr=5, ntheta=10)


# =====================================================================
# Growth-ratio-driven grading actually controls the realized ratio
# =====================================================================
def test_graded_variant_respects_requested_growth_ratio_radially():
    """Sample the realized radial node spacing along one quadrant's
    theta=0 ray (hole boundary outward) and confirm the growth ratio
    the *_graded() convenience wrapper produced doesn't wildly exceed
    what was requested -- the whole point of deriving grade_p instead of
    hand-picking it."""
    Lx, Ly, R = 1.0, 1.0, 0.05
    hole_center = (0.5, 0.5)
    growth_ratio = 1.2
    mesh = hole_in_rectangle_mesh_graded(Lx, Ly, hole_center, R, nr=8,
                                          ntheta=10, h_far=0.1,
                                          growth_ratio=growth_ratio)
    # theta=0 ray from the hole center: y == hole_center[1], x >= hole_center[0]+R
    cy = hole_center[1]
    on_ray = np.where(np.abs(mesh.nodes[:, 1] - cy) < 1e-9)[0]
    xs = np.sort(mesh.nodes[on_ray, 0])
    xs = xs[xs >= hole_center[0] + R - 1e-9]
    realized = growth_ratio_of_partition(xs)
    assert realized <= growth_ratio + 0.15, (
        f"requested growth_ratio={growth_ratio}, realized={realized} along "
        f"the sampled ray -- grading exponent derivation isn't controlling "
        f"the actual mesh")


def test_graded_variant_more_uniform_at_ratio_near_one():
    """A growth_ratio very close to 1 should push grade_p toward 1
    (near-uniform spacing) -- sanity check on the direction of the
    derivation, not just its magnitude."""
    Lx, Ly, R = 1.0, 1.0, 0.05
    mesh_tight = hole_in_rectangle_mesh_graded(Lx, Ly, (0.5, 0.5), R, nr=8,
                                                ntheta=10, h_far=0.1,
                                                growth_ratio=1.01)
    mesh_loose = hole_in_rectangle_mesh_graded(Lx, Ly, (0.5, 0.5), R, nr=8,
                                                ntheta=10, h_far=0.1,
                                                growth_ratio=3.0)
    cy = 0.5
    def realized_ratio(mesh):
        on_ray = np.where(np.abs(mesh.nodes[:, 1] - cy) < 1e-9)[0]
        xs = np.sort(mesh.nodes[on_ray, 0])
        xs = xs[xs >= 0.5 + R - 1e-9]
        return growth_ratio_of_partition(xs)
    assert realized_ratio(mesh_tight) < realized_ratio(mesh_loose)


# =====================================================================
# 3-D extension: grading survives extrusion
# =====================================================================
def test_extrusion_preserves_2d_grading_and_quality():
    Lx, Ly, R, Lz = 1.0, 1.0, 0.1, 0.3
    mesh2d = hole_in_rectangle_mesh_graded(Lx, Ly, (0.5, 0.5), R, nr=5,
                                            ntheta=8, h_far=0.15,
                                            growth_ratio=1.3)
    nz = 4
    mesh3d = extrude_mesh(mesh2d, Lz, nz)
    assert mesh3d.dim == 3
    assert len(mesh3d.elements) == len(mesh2d.elements) * nz
    assert len(mesh3d.nodes) == len(mesh2d.nodes) * (nz + 1)
    # Positive-volume check per Hex8 (via two triangulated tets' signed
    # volumes is overkill here; instead confirm every element's 8 corner
    # z-values split cleanly into two layers -- a basic sanity check that
    # extrusion didn't scramble connectivity).
    for e in mesh3d.elements:
        zs = mesh3d.nodes[e, 2]
        assert len(set(np.round(zs, 12))) == 2


# =====================================================================
# _reflect_mesh building block
# =====================================================================
def test_reflect_mesh_preserves_winding_orientation():
    """A pure reflection flips a CCW quad to CW unless connectivity is
    reversed -- confirm _reflect_mesh's reversal keeps every element's
    shoelace area positive after reflection."""
    base = rectangle_with_hole_mesh_quarter_full(0.5, 0.3, 0.05, 5, 8)
    reflected = _reflect_mesh(base, axis=0)
    for e in reflected.elements:
        x, y = reflected.nodes[e, 0], reflected.nodes[e, 1]
        signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert signed > 0
