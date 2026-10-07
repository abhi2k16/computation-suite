"""
test_mesh_transform.py -- Wave 11 item 106 (docs/consolidated_future_
roadmap.md): validates MeshTransformation (mesh_transform.py) against
the existing, already-validated per-element B_matrix()/jacobian() path
it precomputes -- the physical gradients and JxW it produces must agree
EXACTLY (to floating-point solve-vs-solve precision, not merely
"close") with what Quad4PlaneStress.B_matrix()/Hex8Solid3D.B_matrix()
already compute one element at a time, since it is the SAME formula,
just batched.

TestSimplexFamily (Wave 15 item 126, docs/consolidated_future_roadmap.
md, source: TensorMesh's "Elements and Quadrature" documentation page)
extends this same validation to Tri3PlaneStress/Tri6PlaneStress/
Tet4Solid3D/Tet10Solid3D -- previously explicitly OUT of scope (see
this file's own now-updated test_tet4_and_tri3_now_in_scope_via_
simplex_family, which replaces the old test_out_of_scope_element_
raises_clear_error's Tet4 example). The decisive check for this family
is not just "constructs without erroring" but that
vectorized_assembly.tensorized_element_stiffness()'s batched result
matches Tri3PlaneStress.stiffness()/Tri6PlaneStress.stiffness()/
Tet4Solid3D.stiffness()/Tet10Solid3D.stiffness()'s own per-element loop
EXACTLY -- including the abs(detJ) vs. signed-detJ convention
difference between the two quadrature families (see mesh_transform.py's
own module docstring), which a naive "just reuse the tensor-family
JxW formula" implementation would silently get wrong for any element
whose natural-coordinate node winding gives a negative detJ.
"""
import numpy as np
import pytest

from fea_engine import Quad4PlaneStress, Hex8Solid3D, Tet4Solid3D, Tri3PlaneStress
from fea_engine.elements.solids import Tri6PlaneStress, Tet10Solid3D
from fea_engine.elements.plates import Quad4MindlinPlate
from fea_engine.mesh import Mesh, rectangle_mesh, box_mesh
from fea_engine.mesh_transform import MeshTransformation
from fea_engine.vectorized_assembly import tensorized_element_stiffness


def test_shape_grad_matches_per_element_B_matrix_on_quad4_patch():
    mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=4, ny=3)
    elem = Quad4PlaneStress()
    mt = MeshTransformation(mesh, elem)

    assert mt.n_elements == mesh.elements.shape[0]
    assert mt.shape_grad.shape == (mt.n_elements, mt.n_gauss, 4, 2)
    assert mt.JxW.shape == (mt.n_elements, mt.n_gauss)

    from fea_engine.elements.base import gauss_product
    pts, wts = gauss_product(elem.gauss_order, elem.dim)

    for e, conn in enumerate(mesh.elements):
        elem_coords = mesh.nodes[conn]
        for q, (p, w) in enumerate(zip(pts, wts)):
            B_ref, detJ_ref = elem.B_matrix(p, elem_coords)
            # Reconstruct the physical gradient the reference B encodes:
            # B_ref[0, 2k] = dNx[k], B_ref[1, 2k+1] = dNy[k]
            dNx_ref = B_ref[0, 0::2]
            dNy_ref = B_ref[1, 1::2]
            assert np.allclose(mt.shape_grad[e, q, :, 0], dNx_ref, atol=1e-13)
            assert np.allclose(mt.shape_grad[e, q, :, 1], dNy_ref, atol=1e-13)
            assert np.isclose(mt.JxW[e, q], detJ_ref * w, atol=1e-13)


def test_shape_grad_matches_per_element_B_matrix_on_hex8_patch():
    mesh = box_mesh(Lx=1.0, Ly=1.0, Lz=1.0, nx=2, ny=2, nz=2)
    elem = Hex8Solid3D()
    mt = MeshTransformation(mesh, elem)

    from fea_engine.elements.base import gauss_product
    pts, wts = gauss_product(elem.gauss_order, elem.dim)

    for e, conn in enumerate(mesh.elements):
        elem_coords = mesh.nodes[conn]
        for q, (p, w) in enumerate(zip(pts, wts)):
            B_ref, detJ_ref = elem.B_matrix(p, elem_coords)
            dNx_ref = B_ref[0, 0::3]
            dNy_ref = B_ref[1, 1::3]
            dNz_ref = B_ref[2, 2::3]
            assert np.allclose(mt.shape_grad[e, q, :, 0], dNx_ref, atol=1e-12)
            assert np.allclose(mt.shape_grad[e, q, :, 1], dNy_ref, atol=1e-12)
            assert np.allclose(mt.shape_grad[e, q, :, 2], dNz_ref, atol=1e-12)
            assert np.isclose(mt.JxW[e, q], detJ_ref * w, atol=1e-12)


def test_out_of_scope_element_raises_clear_error():
    # Quad4MindlinPlate overrides Element.stiffness() with its own
    # selective-reduced-integration scheme while still declaring
    # quadrature_family="tensor" (the default -- it never opted into
    # "simplex") -- still explicitly excluded, see mesh_transform.py's
    # own Scope docstring. (Tet4Solid3D/Tet10Solid3D/Tri3PlaneStress/
    # Tri6PlaneStress used to be the example here before Wave 15 item
    # 126 brought the whole simplex family INTO scope -- see
    # TestSimplexFamily below for their own, now-passing coverage.)
    mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
    elem = Quad4MindlinPlate()
    with pytest.raises(ValueError, match="overrides Element.stiffness"):
        MeshTransformation(mesh, elem, connectivity=np.zeros((1, 4), dtype=int))


def test_unrecognized_quadrature_family_raises_clear_error():
    class _BogusElement(Quad4PlaneStress):
        quadrature_family = "not_a_real_family"

    mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=1, ny=1)
    with pytest.raises(ValueError, match="unrecognized quadrature_family"):
        MeshTransformation(mesh, _BogusElement())


class TestSimplexFamily:
    """Wave 15 item 126 (docs/consolidated_future_roadmap.md): the
    generalization of MeshTransformation/vectorized_assembly beyond the
    tensor-product (Quad4/Quad8/Hex8/Hex20) family to Tri3PlaneStress/
    Tri6PlaneStress/Tet4Solid3D/Tet10Solid3D, once item 125 gave the
    simplex family a real order-parameterized quadrature rule to
    qualify against _check_scope(). The decisive check throughout: the
    batched vectorized_assembly.tensorized_element_stiffness() result
    must match each element's own, already-validated per-element
    stiffness() loop EXACTLY -- not just "MeshTransformation constructs
    without raising"."""

    def _straight_sided_tri6_mesh(self):
        c0, c1, c2 = np.array([0.0, 0.0]), np.array([2.0, 0.2]), np.array([0.3, 1.5])
        nodes = np.array([c0, c1, c2, (c0 + c1) / 2, (c1 + c2) / 2, (c2 + c0) / 2])
        conn = np.array([[0, 1, 2, 3, 4, 5]])
        return Mesh(nodes, conn, dim=2)

    def _straight_sided_tet10_mesh(self):
        p0, p1, p2, p3 = (np.array([0, 0, 0.0]), np.array([1.2, 0.1, 0.05]),
                           np.array([0.1, 1.3, 0.05]), np.array([0.05, 0.05, 1.1]))
        nodes = np.array([p0, p1, p2, p3, (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p0) / 2,
                           (p0 + p3) / 2, (p1 + p3) / 2, (p2 + p3) / 2])
        conn = np.array([list(range(10))])
        return Mesh(nodes, conn, dim=3)

    def test_tri3_batched_stiffness_matches_per_element_loop(self):
        print("=" * 70)
        print("CHECK: Tri3PlaneStress via MeshTransformation/tensorized_")
        print("element_stiffness() matches the per-element .stiffness() loop")
        print("exactly, on a small fan-of-triangles patch (a genuinely")
        print("multi-element mesh, not a single reference element)")
        print("=" * 70)
        nodes = np.array([[0, 0], [2, 0], [2, 1.5], [0, 1.5], [1, 0.7]], dtype=float)
        conn = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        mesh = Mesh(nodes, conn, dim=2)
        elem = Tri3PlaneStress()
        D = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.35]])

        mt = MeshTransformation(mesh, elem)
        ke_batch = tensorized_element_stiffness(mt, D)
        for e, c in enumerate(conn):
            ke_ref = elem.stiffness(nodes[c], D)
            assert np.allclose(ke_batch[e], ke_ref, atol=1e-10)
        print(f"  PASS -- all {len(conn)} elements match exactly")

    def test_tri6_batched_stiffness_matches_per_element_loop(self):
        mesh = self._straight_sided_tri6_mesh()
        elem = Tri6PlaneStress()
        D = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.35]])
        mt = MeshTransformation(mesh, elem)
        ke_batch = tensorized_element_stiffness(mt, D)
        ke_ref = elem.stiffness(mesh.nodes, D)
        assert np.allclose(ke_batch[0], ke_ref, atol=1e-9)

    def test_tet4_batched_stiffness_matches_per_element_loop(self):
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        conn = np.array([[0, 1, 2, 3]])
        mesh = Mesh(nodes, conn, dim=3)
        elem = Tet4Solid3D()
        D6 = np.eye(6) + 0.1
        mt = MeshTransformation(mesh, elem)
        ke_batch = tensorized_element_stiffness(mt, D6)
        ke_ref = elem.stiffness(nodes, D6)
        assert np.allclose(ke_batch[0], ke_ref, atol=1e-10)

    def test_tet10_batched_stiffness_matches_per_element_loop(self):
        mesh = self._straight_sided_tet10_mesh()
        elem = Tet10Solid3D()
        D6 = np.eye(6) + 0.1
        mt = MeshTransformation(mesh, elem)
        ke_batch = tensorized_element_stiffness(mt, D6)
        ke_ref = elem.stiffness(mesh.nodes, D6)
        assert np.allclose(ke_batch[0], ke_ref, atol=1e-9)

    def test_explicit_higher_gauss_order_matches_elements_own_quad_order(self):
        """MeshTransformation's gauss_order= override must dispatch to
        the SAME tri_quadrature(order)/tet_quadrature(order) call the
        element's own quad_order= parameter (item 125) would use --
        confirming the two new item-125/126 surfaces are actually wired
        to the same underlying quadrature, not two independently-
        plausible-looking implementations that happen to agree only by
        coincidence at the default order."""
        mesh = self._straight_sided_tri6_mesh()
        elem = Tri6PlaneStress()
        D = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.35]])
        mt_hi = MeshTransformation(mesh, elem, gauss_order=4)
        ke_batch_hi = tensorized_element_stiffness(mt_hi, D)
        ke_ref_hi = elem.stiffness(mesh.nodes, D, quad_order=4)
        assert np.allclose(ke_batch_hi[0], ke_ref_hi, atol=1e-9)

    def test_JxW_uses_abs_detJ_for_simplex_family_not_signed(self):
        """The one subtle convention difference the module docstring
        flags explicitly: simplex stiffness()/mass() loops use
        abs(detJ); the tensor family's generic Element.stiffness() loop
        uses signed detJ. Confirms MeshTransformation.JxW is always
        non-negative for the simplex family (abs), even though jac's
        own raw det() -- reconstructible from mt.jac -- need not be."""
        mesh = self._straight_sided_tet10_mesh()
        elem = Tet10Solid3D()
        mt = MeshTransformation(mesh, elem)
        raw_detJ = np.linalg.det(mt.jac)
        assert np.all(mt.JxW >= 0.0)
        # And JxW really is |raw_detJ| * (barycentric weight * 1/6), not
        # just "happens to be positive" -- reconstruct it independently.
        from fea_engine.mesh_transform import _reference_quadrature
        _, wts = _reference_quadrature(elem, elem.gauss_order)
        expected = np.abs(raw_detJ) * np.asarray(wts)[None, :]
        assert np.allclose(mt.JxW, expected, atol=1e-12)


def test_mismatched_connectivity_node_count_raises():
    mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
    elem = Quad4PlaneStress()
    bad_conn = mesh.elements[:, :3]   # wrong node count for Quad4
    with pytest.raises(ValueError, match="connectivity has"):
        MeshTransformation(mesh, elem, connectivity=bad_conn)
