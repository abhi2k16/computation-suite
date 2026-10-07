"""
test_field_interpolation.py -- Wave 16 item 128 (docs/consolidated_
future_roadmap.md, source: TensorMesh's `Forms` documentation page):
validates interpolate_point_data()/interpolate_point_data_gradient()
(mesh_transform.py) -- the bounded, non-dispatch version of TensorMesh's
automatic point_data-to-quadrature-point interpolation.

The decisive checks throughout are exactness claims, not "runs without
erroring": on any straight-sided element (tensor OR simplex family),
interpolating a genuinely AFFINE nodal field must reproduce the true
field value exactly at every quadrature point's own physical location
(every element formulation this package has is complete through affine
fields -- that's the whole point of the patch test), and the field's
gradient must be recovered as the exact constant (b, c[, d]) coefficient
vector, not merely "something plausible". A self-consistency check
(interpolating the coordinate field mesh.nodes itself) confirms the
utility's own quadrature-point placement agrees with the isoparametric
mapping every element already uses internally.
"""
import numpy as np
import pytest

from fea_engine import Quad4PlaneStress, Hex8Solid3D, Tet4Solid3D, Tri3PlaneStress
from fea_engine.mesh import Mesh, rectangle_mesh, box_mesh
from fea_engine.mesh_transform import MeshTransformation, interpolate_point_data, \
    interpolate_point_data_gradient


def _affine_scalar(coords, a, b):
    """a + b . coords, coords: (..., dim)."""
    return a + coords @ np.asarray(b)


class TestCoordinateSelfConsistency:
    """interpolate_point_data(mt, mesh.nodes) must reproduce the exact
    physical quadrature-point locations every element's own isoparametric
    mapping (N . elem_coords) already computes -- mesh.nodes IS a valid
    point_data-shaped array (n_nodes, dim), so this is both a genuine use
    case and a self-consistency check on shape_val's own indexing."""

    def test_quad4_physical_points_match_isoparametric_mapping(self):
        mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=4, ny=3)
        elem = Quad4PlaneStress()
        mt = MeshTransformation(mesh, elem)
        phys = interpolate_point_data(mt, mesh.nodes)
        assert phys.shape == (mt.n_elements, mt.n_gauss, 2)

        from fea_engine.elements.base import gauss_product
        pts, _ = gauss_product(elem.gauss_order, elem.dim)
        for e, conn in enumerate(mesh.elements):
            elem_coords = mesh.nodes[conn]
            for q, p in enumerate(pts):
                N, _ = elem._cached_shape_and_derivs(tuple(p))
                expected = N @ elem_coords
                assert np.allclose(phys[e, q], expected, atol=1e-13)

    def test_tet4_physical_points_match_isoparametric_mapping(self):
        mesh = box_mesh(Lx=1.0, Ly=1.0, Lz=1.0, nx=1, ny=1, nz=1)
        # box_mesh is Hex8 -- build a genuine Tet4 mesh directly instead.
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        conn = np.array([[0, 1, 2, 3]])
        tet_mesh = Mesh(nodes, conn, dim=3)
        elem = Tet4Solid3D()
        mt = MeshTransformation(tet_mesh, elem)
        phys = interpolate_point_data(mt, tet_mesh.nodes)

        # Tet4Solid3D is closed-form constant-strain (its own stiffness()
        # doesn't loop over tet_quadrature_4pt() at all -- see Wave 15's
        # own "Tri3/Tet4 untouched" note), so MeshTransformation's
        # simplex-family dispatch (_reference_quadrature) is the actual
        # source of truth for which points mt.shape_val corresponds to,
        # not the legacy fixed-order tet_quadrature_4pt() rule -- reuse
        # it directly rather than assuming which rule was used.
        from fea_engine.mesh_transform import _reference_quadrature
        pts, _ = _reference_quadrature(elem, elem.gauss_order)
        for q, p in enumerate(pts):
            N, _ = elem._cached_shape_and_derivs(tuple(p))
            expected = N @ nodes
            assert np.allclose(phys[0, q], expected, atol=1e-13)


class TestAffineFieldExactness:
    """Every element formulation in this package reproduces affine
    fields exactly (the patch-test property); interpolate_point_data()
    reuses the SAME cached shape values those elements' own stiffness
    loops use, so it must inherit that exactness."""

    def test_quad4_scalar_affine_field_and_gradient(self):
        mesh = rectangle_mesh(Lx=2.0, Ly=1.5, nx=5, ny=4)
        elem = Quad4PlaneStress()
        mt = MeshTransformation(mesh, elem)

        a, b = 3.0, np.array([1.7, -0.9])
        field = _affine_scalar(mesh.nodes, a, b)   # (n_nodes,)

        values = interpolate_point_data(mt, field)
        grads = interpolate_point_data_gradient(mt, field)
        assert values.shape == (mt.n_elements, mt.n_gauss)
        assert grads.shape == (mt.n_elements, mt.n_gauss, 2)

        phys = interpolate_point_data(mt, mesh.nodes)
        expected_values = _affine_scalar(phys, a, b)
        assert np.allclose(values, expected_values, atol=1e-11)
        # gradient of a + b.x is the constant vector b, everywhere
        assert np.allclose(grads, np.broadcast_to(b, grads.shape), atol=1e-10)

    def test_hex8_scalar_affine_field_and_gradient(self):
        mesh = box_mesh(Lx=1.0, Ly=1.0, Lz=1.0, nx=2, ny=2, nz=2)
        elem = Hex8Solid3D()
        mt = MeshTransformation(mesh, elem)

        a, b = -1.2, np.array([0.4, 0.6, -1.1])
        field = _affine_scalar(mesh.nodes, a, b)

        values = interpolate_point_data(mt, field)
        grads = interpolate_point_data_gradient(mt, field)
        phys = interpolate_point_data(mt, mesh.nodes)
        assert np.allclose(values, _affine_scalar(phys, a, b), atol=1e-10)
        assert np.allclose(grads, np.broadcast_to(b, grads.shape), atol=1e-9)

    def test_tri3_scalar_affine_field_and_gradient(self):
        nodes = np.array([[0, 0], [2, 0], [2, 1.5], [0, 1.5], [1, 0.7]], dtype=float)
        conn = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        mesh = Mesh(nodes, conn, dim=2)
        elem = Tri3PlaneStress()
        mt = MeshTransformation(mesh, elem)

        a, b = 2.0, np.array([-0.5, 1.3])
        field = _affine_scalar(nodes, a, b)
        values = interpolate_point_data(mt, field)
        grads = interpolate_point_data_gradient(mt, field)
        phys = interpolate_point_data(mt, nodes)
        assert np.allclose(values, _affine_scalar(phys, a, b), atol=1e-11)
        assert np.allclose(grads, np.broadcast_to(b, grads.shape), atol=1e-10)

    def test_tet4_scalar_affine_field_and_gradient(self):
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        conn = np.array([[0, 1, 2, 3]])
        mesh = Mesh(nodes, conn, dim=3)
        elem = Tet4Solid3D()
        mt = MeshTransformation(mesh, elem)

        a, b = 0.5, np.array([1.0, -2.0, 0.3])
        field = _affine_scalar(nodes, a, b)
        values = interpolate_point_data(mt, field)
        grads = interpolate_point_data_gradient(mt, field)
        phys = interpolate_point_data(mt, nodes)
        assert np.allclose(values, _affine_scalar(phys, a, b), atol=1e-11)
        assert np.allclose(grads, np.broadcast_to(b, grads.shape), atol=1e-10)


class TestVectorFieldShapes:
    """A (dim,)-trailing-shape vector point_data field (e.g. a
    displacement field from a previous solve, TensorMesh's own example
    use case) must carry its trailing dim through untouched, with an
    extra trailing gradient-direction axis for interpolate_point_data_
    gradient()."""

    def test_vector_field_value_and_gradient_shapes_quad4(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=3, ny=2)
        elem = Quad4PlaneStress()
        mt = MeshTransformation(mesh, elem)

        # A genuinely affine VECTOR field: component k = a_k + b_k . x
        a = np.array([1.0, -2.0])
        B = np.array([[0.5, 0.25], [-0.3, 0.7]])   # rows = per-component gradient
        field = a[None, :] + mesh.nodes @ B.T   # (n_nodes, 2)

        values = interpolate_point_data(mt, field)
        grads = interpolate_point_data_gradient(mt, field)
        assert values.shape == (mt.n_elements, mt.n_gauss, 2)
        assert grads.shape == (mt.n_elements, mt.n_gauss, 2, 2)

        phys = interpolate_point_data(mt, mesh.nodes)
        expected_values = a[None, None, :] + phys @ B.T
        assert np.allclose(values, expected_values, atol=1e-10)
        assert np.allclose(grads, np.broadcast_to(B, grads.shape), atol=1e-9)


class TestErrorHandling:
    def test_field_too_short_raises_on_value_interpolation(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        elem = Quad4PlaneStress()
        mt = MeshTransformation(mesh, elem)
        short_field = np.zeros(2)   # mesh has more than 2 nodes
        with pytest.raises(ValueError, match="field must be indexed by the"):
            interpolate_point_data(mt, short_field)

    def test_field_too_short_raises_on_gradient_interpolation(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        elem = Quad4PlaneStress()
        mt = MeshTransformation(mesh, elem)
        short_field = np.zeros(2)
        with pytest.raises(ValueError, match="field must be indexed by the"):
            interpolate_point_data_gradient(mt, short_field)


def test_reuses_cached_shape_data_not_recomputed():
    """interpolate_point_data()/_gradient() must consume mt.shape_val/
    mt.shape_grad as-is (no re-evaluation of shape functions) -- checked
    directly by monkeypatching the MeshTransformation's own cached
    arrays to a hand-crafted stand-in and confirming the interpolation
    result changes accordingly (proves the function actually reads that
    attribute rather than recomputing shape values from elem_coords)."""
    mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
    elem = Quad4PlaneStress()
    mt = MeshTransformation(mesh, elem)

    field = np.arange(len(mesh.nodes), dtype=float)
    baseline = interpolate_point_data(mt, field).copy()

    # Perturb the cached shape_val in place and confirm the interpolated
    # result changes -- proves this function reads shape_val directly.
    mt.shape_val = mt.shape_val * 2.0
    perturbed = interpolate_point_data(mt, field)
    assert not np.allclose(baseline, perturbed)
    assert np.allclose(perturbed, baseline * 2.0, atol=1e-10)
