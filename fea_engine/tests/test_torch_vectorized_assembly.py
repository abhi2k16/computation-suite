"""
test_torch_vectorized_assembly.py -- Wave 9 addendum item 136 (docs/
consolidated_future_roadmap.md): validates assemble_stiffness_
vectorized()'s backend="numpy"/"torch" contract (vectorized_
assembly.py), building directly on item 135's MeshTransformation
backend contract.

Same two-part split as test_torch_mesh_transform.py (item 135's own
test file) and this project's established convention:

  TestBackendArgumentValidation -- runs UNCONDITIONALLY, no torch
      needed: unknown backend= raises ValueError; backend="numpy"
      (the default) is confirmed unchanged.

  TestTorchBackendNumericalAgreement -- gated on fea_engine.
      torch_sparse_solver._HAS_TORCH. The DECISIVE check here is not
      just "the torch path produces a matrix that resembles the numpy
      one" but that assemble_stiffness_vectorized(..., backend="torch")
      produces the SAME fesystem.K as backend="numpy" (to solve-vs-
      solve floating point precision) on FESystem instances built both
      dense and sparse -- since scatter_global_stiffness()'s own
      torch.sparse_coo_tensor(...).coalesce() duplicate-index-summation
      path is a genuinely different code path from scipy.sparse.
      coo_matrix's own accumulation (this module's own docstring flags
      it as needing its own dedicated validation, not an assumed
      match). Not executed end-to-end in the sandbox this file was
      authored in (no usable torch here); written to run for real, and
      SHOULD be run at least once on a torch-equipped machine.
"""
import numpy as np
import pytest

from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, Hex8Solid3D, mesh as mesh_mod
from fea_engine.torch_sparse_solver import _HAS_TORCH
from fea_engine.vectorized_assembly import assemble_stiffness_vectorized


def _build_quad4_system(nx=5, ny=3, sparse=False):
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    D2 = D_plane_stress(mat)
    m = mesh_mod.rectangle_mesh(2.0, 1.0, nx, ny)
    fs = FESystem(m, Quad4PlaneStress(), sparse=sparse)
    return fs, D2


def _build_hex8_system(nx=2, ny=2, nz=2, sparse=False):
    from fea_engine.material import D_solid3d
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    D3 = D_solid3d(mat)
    m = mesh_mod.box_mesh(1.0, 1.0, 1.0, nx, ny, nz)
    fs = FESystem(m, Hex8Solid3D(), sparse=sparse)
    return fs, D3


class TestBackendArgumentValidation:
    def test_unknown_backend_raises_valueerror(self):
        fs, D2 = _build_quad4_system()
        with pytest.raises(ValueError, match="unknown backend"):
            assemble_stiffness_vectorized(fs, D2, backend="jax")

    def test_default_backend_numpy_matches_omitted_backend(self):
        fs_default, D2 = _build_quad4_system()
        fs_explicit, _ = _build_quad4_system()
        assemble_stiffness_vectorized(fs_default, D2)
        assemble_stiffness_vectorized(fs_explicit, D2, backend="numpy")
        np.testing.assert_array_equal(fs_default.K, fs_explicit.K)

    def test_numpy_backend_still_matches_per_element_loop(self):
        # Regression check: this item must not perturb the pre-existing,
        # already-validated numpy path (Wave 11 item 107's own
        # test_vectorized_matches_looped_on_quad4_patch_dense covers this
        # too -- repeated here as a direct smoke test scoped to this
        # item's own touch points). Same atol=1e-3 that test uses: the
        # einsum-reduction vs. sequential-scatter-add paths sum the same
        # per-element contributions in a different order, so agreement
        # is to float64 summation-order noise (~1e-4 absolute at this
        # O(1e11) stiffness scale), not exact bit-for-bit.
        fs_vec, D2 = _build_quad4_system()
        fs_loop, _ = _build_quad4_system()
        assemble_stiffness_vectorized(fs_vec, D2, backend="numpy")
        fs_loop.assemble_stiffness(D2)
        assert np.allclose(fs_vec.K, fs_loop.K, atol=1e-3, rtol=1e-8)

    def test_fesystem_assemble_stiffness_accepts_backend_kwarg_on_numpy_path(self):
        # FESystem.assemble_stiffness(vectorized=True, backend=...) --
        # confirms the kwarg threads through solver.py without torch
        # needing to be installed for the numpy case.
        fs, D2 = _build_quad4_system()
        fs.assemble_stiffness(D2, vectorized=True, backend="numpy")
        fs_loop, _ = _build_quad4_system()
        fs_loop.assemble_stiffness(D2)
        assert np.allclose(fs.K, fs_loop.K, atol=1e-3, rtol=1e-8)


pytestmark_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not usable in this environment (not installed, or installed "
           "but failing to import -- see torch_sparse_solver.py's docstring); "
           "these tests validate the torch backend assembly path and have "
           "nothing to run without it.")


@pytestmark_torch
class TestTorchBackendNumericalAgreement:
    def test_quad4_dense_matches_numpy_backend(self):
        # Same floating-point summation-order noise the numpy-vs-loop
        # comparison above needs atol=1e-3 for (O(1e11) stiffness
        # entries, einsum vs. scatter-add reduction order differs) --
        # numpy-vs-torch is the identical situation, one more
        # independently-implemented reduction order added to the mix.
        fs_np, D2 = _build_quad4_system(sparse=False)
        fs_torch, _ = _build_quad4_system(sparse=False)
        assemble_stiffness_vectorized(fs_np, D2, backend="numpy")
        assemble_stiffness_vectorized(fs_torch, D2, backend="torch", device="cpu")
        assert np.allclose(fs_np.K, fs_torch.K, atol=1e-3, rtol=1e-8)

    def test_quad4_sparse_matches_numpy_backend(self):
        fs_np, D2 = _build_quad4_system(sparse=True)
        fs_torch, _ = _build_quad4_system(sparse=True)
        assemble_stiffness_vectorized(fs_np, D2, backend="numpy")
        assemble_stiffness_vectorized(fs_torch, D2, backend="torch", device="cpu")
        assert np.allclose(fs_np.K.toarray(), fs_torch.K.toarray(), atol=1e-3, rtol=1e-8)

    def test_hex8_dense_matches_numpy_backend(self):
        fs_np, D3 = _build_hex8_system(sparse=False)
        fs_torch, _ = _build_hex8_system(sparse=False)
        assemble_stiffness_vectorized(fs_np, D3, backend="numpy")
        assemble_stiffness_vectorized(fs_torch, D3, backend="torch", device="cpu")
        assert np.allclose(fs_np.K, fs_torch.K, atol=1e-3, rtol=1e-8)

    def test_torch_backend_end_to_end_static_solve_matches_numpy_backend(self):
        # The decisive, end-to-end check (mirroring item 107's own
        # test_vectorized_static_solve_matches_looped_end_to_end): build
        # K via the torch assembly path, solve with the ORDINARY scipy
        # solve_static() (assembly backend and solve backend are
        # independent, per solver.py's own updated docstring), and
        # confirm the displacement field matches the numpy-assembled,
        # scipy-solved reference.
        fs_np, D2 = _build_quad4_system(sparse=False)
        fs_torch, _ = _build_quad4_system(sparse=False)
        assemble_stiffness_vectorized(fs_np, D2, backend="numpy")
        assemble_stiffness_vectorized(fs_torch, D2, backend="torch", device="cpu")

        left = fs_np.mesh.nodes_on_line(axis=0, value=0.0)
        for n in left:
            fs_np.fix_dofs([n], [0, 1])
            fs_torch.fix_dofs([n], [0, 1])
        tip = fs_np.mesh.nodes_on_line(axis=0, value=2.0)
        for n in tip:
            g = fs_np._global_dofs([n])
            fs_np.F[g[1]] += -1000.0 / len(tip)
            fs_torch.F[g[1]] += -1000.0 / len(tip)

        U_np = fs_np.solve_static()
        U_torch = fs_torch.solve_static()
        assert np.allclose(U_np, U_torch, atol=1e-9, rtol=1e-5)

    def test_chunked_torch_assembly_matches_unchunked_numpy(self):
        fs_np, D2 = _build_quad4_system(nx=8, ny=4, sparse=False)
        fs_torch, _ = _build_quad4_system(nx=8, ny=4, sparse=False)
        assemble_stiffness_vectorized(fs_np, D2, backend="numpy")
        assemble_stiffness_vectorized(fs_torch, D2, backend="torch", device="cpu", chunk_size=7)
        # Chunked accumulation isn't bit-for-bit even on the numpy path
        # (see assemble_stiffness_vectorized()'s own docstring on
        # floating-point summation-order noise) -- same tolerance
        # TestMemoryChunking already established.
        assert np.allclose(fs_np.K, fs_torch.K, atol=1e-3, rtol=1e-8)

    def test_fesystem_assemble_stiffness_accepts_backend_kwarg_on_torch_path(self):
        fs, D2 = _build_quad4_system()
        fs.assemble_stiffness(D2, vectorized=True, backend="torch", device="cpu")
        fs_ref, _ = _build_quad4_system()
        fs_ref.assemble_stiffness(D2, vectorized=True, backend="numpy")
        assert np.allclose(fs.K, fs_ref.K, atol=1e-3, rtol=1e-8)

    def test_unavailable_torch_fails_fast_at_first_chunk(self, monkeypatch):
        import fea_engine.mesh_transform as mt_mod

        def _fake_require_torch():
            raise ImportError("simulated: torch unavailable")

        monkeypatch.setattr(mt_mod, "_require_torch", _fake_require_torch)
        fs, D2 = _build_quad4_system()
        with pytest.raises(ImportError):
            assemble_stiffness_vectorized(fs, D2, backend="torch")
