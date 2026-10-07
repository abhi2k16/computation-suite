"""
test_vectorized_assembly.py -- Wave 11 item 107 (docs/consolidated_
future_roadmap.md): validates assemble_stiffness_vectorized() /
FESystem.assemble_stiffness(vectorized=True) (vectorized_assembly.py)
against the existing, already-validated per-element Python-loop path --
the item's own headline claim is "changes only the SCHEDULING, never
the numbers" (the same phrasing this project used for Wave 10 item
102's vmap batching), so the decisive check here is numerical agreement
with the default path, not merely "produces something plausible".
"""
import numpy as np
import pytest

from fea_engine import (Quad4PlaneStress, Hex8Solid3D, FESystem, Material,
                         D_plane_stress, D_solid3d)
from fea_engine.mesh import rectangle_mesh, box_mesh
from fea_engine.vectorized_assembly import (
    build_B_batched, tensorized_element_stiffness, assemble_stiffness_vectorized)
from fea_engine.mesh_transform import MeshTransformation


def _steel():
    return Material(E=2.1e11, nu=0.3, rho=7850.0)


def test_vectorized_matches_looped_on_quad4_patch_dense():
    mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=5, ny=4)
    D = D_plane_stress(_steel())

    fs_loop = FESystem(mesh, Quad4PlaneStress())
    fs_loop.assemble_stiffness(D)

    fs_vec = FESystem(mesh, Quad4PlaneStress())
    fs_vec.assemble_stiffness(D, vectorized=True)

    # Stiffness entries here are O(1e11) (steel, SI units) -- the two
    # paths sum the same per-element contributions in a different
    # order (einsum reduction vs. sequential scatter-add), so agreement
    # is to float64 summation-order noise (~1e-4 absolute here), not
    # exact bit-for-bit; rtol=1e-8 comfortably covers that at this scale.
    assert np.allclose(fs_loop.K, fs_vec.K, atol=1e-3, rtol=1e-8)
    # symmetry sanity, independent of the comparison above
    assert np.allclose(fs_vec.K, fs_vec.K.T, atol=1e-3)


def test_vectorized_matches_looped_on_quad4_patch_sparse():
    mesh = rectangle_mesh(Lx=1.5, Ly=1.0, nx=4, ny=3)
    D = D_plane_stress(_steel())

    fs_loop = FESystem(mesh, Quad4PlaneStress(), sparse=True)
    fs_loop.assemble_stiffness(D)

    fs_vec = FESystem(mesh, Quad4PlaneStress(), sparse=True)
    fs_vec.assemble_stiffness(D, vectorized=True)

    assert np.allclose(fs_loop.K.toarray(), fs_vec.K.toarray(), atol=1e-3, rtol=1e-8)


def test_vectorized_matches_looped_on_hex8_patch():
    mesh = box_mesh(Lx=1.0, Ly=1.0, Lz=1.0, nx=2, ny=2, nz=2)
    D = D_solid3d(_steel())

    fs_loop = FESystem(mesh, Hex8Solid3D())
    fs_loop.assemble_stiffness(D)

    fs_vec = FESystem(mesh, Hex8Solid3D())
    fs_vec.assemble_stiffness(D, vectorized=True)

    assert np.allclose(fs_loop.K, fs_vec.K, atol=1.0, rtol=1e-9)


def test_vectorized_static_solve_matches_looped_end_to_end():
    # The real headline check: solve a cantilever both ways, compare
    # tip displacement -- not just the assembled K in isolation.
    mesh = rectangle_mesh(Lx=4.0, Ly=1.0, nx=8, ny=3)
    D = D_plane_stress(_steel())
    left = mesh.nodes_on_line(axis=0, value=0.0)
    tip_nodes = mesh.nodes_on_line(axis=0, value=4.0)

    def _solve(vectorized):
        fs = FESystem(mesh, Quad4PlaneStress())
        fs.assemble_stiffness(D, vectorized=vectorized)
        for n in left:
            fs.fix_dofs([n], [0, 1])
        for n in tip_nodes:
            fs.F[fs._global_dofs([n])[1]] += -1000.0 / len(tip_nodes)
        return fs.solve_static()

    U_loop = _solve(False)
    U_vec = _solve(True)
    assert np.allclose(U_loop, U_vec, atol=1e-9, rtol=1e-8)
    assert np.max(np.abs(U_loop)) > 1e-6   # sanity: something actually deflected


def test_reduced_method_rejected_for_vectorized():
    mesh = box_mesh(Lx=1.0, Ly=1.0, Lz=1.0, nx=1, ny=1, nz=1)
    D = D_solid3d(_steel())
    fs = FESystem(mesh, Hex8Solid3D())
    with pytest.raises(ValueError, match="vectorized=True only supports"):
        fs.assemble_stiffness(D, method="reduced", vectorized=True)


def test_build_B_batched_shapes():
    mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
    elem = Quad4PlaneStress()
    mt = MeshTransformation(mesh, elem)
    B = build_B_batched(mt.shape_grad, dofs_per_node=2)
    assert B.shape == (mt.n_elements, mt.n_gauss, 3, 8)
    with pytest.raises(ValueError, match="unsupported dofs_per_node"):
        build_B_batched(mt.shape_grad, dofs_per_node=5)


class TestMemoryChunking:
    """Wave 16 item 129 (docs/consolidated_future_roadmap.md, source:
    TensorMesh's `Forms` page `batch_size` argument): chunk_size must
    produce the SAME fesystem.K as chunk_size=None regardless of the
    chunk size chosen -- including chunk sizes that don't evenly divide
    the element count (forcing a shorter tail chunk), a chunk size of 1
    (the maximally-chunked case), and a chunk size larger than the whole
    mesh (degenerates to the unchunked single pass).

    Agreement is checked to a numerical tolerance, NOT exact bit-for-bit
    equality -- verified directly (not assumed) by first trying exact
    `np.array_equal` here and finding it genuinely fails for some chunk
    sizes (e.g. chunk_size=3 on the quad4 case below differs from
    chunk_size=None by ~1.2e-4 absolute on ~1e11-scale entries): chunking
    changes which entries get summed together inside ONE scipy
    coo_matrix duplicate-accumulation call (within a chunk) versus
    across separate Python-level `fesystem.K = fesystem.K + K_block`
    additions (between chunks) for any global dof shared by elements
    landing in different chunks -- floating-point addition is not
    associative, so a different grouping can give a different rounding,
    exactly the same "changes the SCHEDULING, not the numbers, but
    summation order is part of scheduling" caveat item 107's own
    vectorized-vs-looped comparison already documents and tests at
    atol=1e-3/rtol=1e-8 for these same steel/SI-unit magnitudes -- the
    identical tolerance is reused here for the identical reason."""

    def test_chunk_sizes_agree_on_quad4_dense(self):
        mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=5, ny=4)   # 20 elements
        D = D_plane_stress(_steel())

        fs_unchunked = FESystem(mesh, Quad4PlaneStress())
        fs_unchunked.assemble_stiffness(D, vectorized=True)

        for chunk_size in (1, 3, 7, 20, 1000):
            fs = FESystem(mesh, Quad4PlaneStress())
            fs.assemble_stiffness(D, vectorized=True, chunk_size=chunk_size)
            assert np.allclose(fs.K, fs_unchunked.K, atol=1e-3, rtol=1e-8), (
                f"chunk_size={chunk_size} disagreed with chunk_size=None beyond "
                f"float64 summation-order noise")

    def test_chunk_sizes_agree_on_quad4_sparse(self):
        mesh = rectangle_mesh(Lx=1.5, Ly=1.0, nx=4, ny=3)   # 12 elements
        D = D_plane_stress(_steel())

        fs_unchunked = FESystem(mesh, Quad4PlaneStress(), sparse=True)
        fs_unchunked.assemble_stiffness(D, vectorized=True)

        for chunk_size in (1, 5, 12):
            fs = FESystem(mesh, Quad4PlaneStress(), sparse=True)
            fs.assemble_stiffness(D, vectorized=True, chunk_size=chunk_size)
            assert np.allclose(fs.K.toarray(), fs_unchunked.K.toarray(), atol=1e-3, rtol=1e-8), (
                f"chunk_size={chunk_size} disagreed with chunk_size=None beyond "
                f"float64 summation-order noise")

    def test_chunk_sizes_agree_on_hex8(self):
        mesh = box_mesh(Lx=1.0, Ly=1.0, Lz=1.0, nx=2, ny=2, nz=2)   # 8 elements
        D = D_solid3d(_steel())

        fs_unchunked = FESystem(mesh, Hex8Solid3D())
        fs_unchunked.assemble_stiffness(D, vectorized=True)

        for chunk_size in (1, 3, 8):
            fs = FESystem(mesh, Hex8Solid3D())
            fs.assemble_stiffness(D, vectorized=True, chunk_size=chunk_size)
            assert np.allclose(fs.K, fs_unchunked.K, atol=1.0, rtol=1e-9), (
                f"chunk_size={chunk_size} disagreed with chunk_size=None beyond "
                f"float64 summation-order noise")

    def test_chunk_sizes_exact_when_no_element_spans_a_chunk_boundary_at_shared_dof(self):
        # chunk_size=1 (every element its own chunk) and chunk_size >=
        # n_elements (degenerates to one chunk) are the two cases where
        # the grouping genuinely CAN'T differ from the unchunked
        # single-coo_matrix summation -- confirmed exact, not just close,
        # as the decisive positive control for the tolerance used above.
        mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=5, ny=4)
        D = D_plane_stress(_steel())
        fs_unchunked = FESystem(mesh, Quad4PlaneStress())
        fs_unchunked.assemble_stiffness(D, vectorized=True)
        for chunk_size in (1, 1000):
            fs = FESystem(mesh, Quad4PlaneStress())
            fs.assemble_stiffness(D, vectorized=True, chunk_size=chunk_size)
            assert np.array_equal(fs.K, fs_unchunked.K)

    def test_chunked_static_solve_matches_unchunked_end_to_end(self):
        mesh = rectangle_mesh(Lx=4.0, Ly=1.0, nx=8, ny=3)
        D = D_plane_stress(_steel())
        left = mesh.nodes_on_line(axis=0, value=0.0)
        tip_nodes = mesh.nodes_on_line(axis=0, value=4.0)

        def _solve(chunk_size):
            fs = FESystem(mesh, Quad4PlaneStress())
            fs.assemble_stiffness(D, vectorized=True, chunk_size=chunk_size)
            for n in left:
                fs.fix_dofs([n], [0, 1])
            for n in tip_nodes:
                fs.F[fs._global_dofs([n])[1]] += -1000.0 / len(tip_nodes)
            return fs.solve_static()

        U_unchunked = _solve(None)
        U_chunked = _solve(3)
        assert np.allclose(U_unchunked, U_chunked, atol=1e-12, rtol=1e-8)
        assert np.max(np.abs(U_unchunked)) > 1e-6

    def test_nonpositive_chunk_size_raises(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        fs = FESystem(mesh, Quad4PlaneStress())
        with pytest.raises(ValueError, match="chunk_size must be a positive int"):
            fs.assemble_stiffness(D, vectorized=True, chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be a positive int"):
            fs.assemble_stiffness(D, vectorized=True, chunk_size=-5)

    def test_chunk_size_ignored_without_vectorized_true_raises_typeerror(self):
        # chunk_size is only meaningful with vectorized=True -- passing
        # it through the plain per-element path should surface as an
        # ordinary unexpected-keyword TypeError from the element's own
        # stiffness(), not be silently accepted and ignored.
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        fs = FESystem(mesh, Quad4PlaneStress())
        with pytest.raises(TypeError):
            fs.assemble_stiffness(D, chunk_size=3)

    def test_chunk_size_on_simplex_family(self):
        # Wave 15 item 126 brought Tri3/Tet4/Tri6/Tet10 into
        # MeshTransformation's scope; chunking must work identically
        # there since it's built on the same connectivity= mechanism.
        from fea_engine import Tri3PlaneStress
        nodes = np.array([[0, 0], [2, 0], [2, 1.5], [0, 1.5], [1, 0.7]], dtype=float)
        conn = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        from fea_engine.mesh import Mesh
        tri_mesh = Mesh(nodes, conn, dim=2)
        D = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.35]])

        fs_unchunked = FESystem(tri_mesh, Tri3PlaneStress())
        fs_unchunked.assemble_stiffness(D, vectorized=True)

        fs_chunked = FESystem(tri_mesh, Tri3PlaneStress())
        fs_chunked.assemble_stiffness(D, vectorized=True, chunk_size=1)

        assert np.array_equal(fs_unchunked.K, fs_chunked.K)
