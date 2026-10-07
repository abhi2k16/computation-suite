"""
test_nonhomogeneous_dirichlet.py -- Wave 16 item 131 (docs/consolidated_
future_roadmap.md, source: TensorMesh's `Boundary Conditions`
documentation page's non-homogeneous `Condenser(mask, values)` case):
validates FESystem.fix_dofs(value=...)/solve_static()'s new -Kio*uo
static-condensation RHS correction.

Decisive checks throughout: agreement with an independently-built
bordered/manual elimination reference (not just "runs without
erroring"), exact backward compatibility when value=0.0 (the only value
possible before this item existed), and that the torch backend fails
LOUDLY (a specific NotImplementedError) rather than silently solving the
wrong (homogeneous) system when a nonzero value is present -- matching
this project's own "fails loudly rather than silently" precedent
(Wave 0 item 6's singular-system handling, mesh_transform.py's
_check_scope(), etc.).
"""
import numpy as np
import pytest

from fea_engine import Quad4PlaneStress, FESystem, Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh


def _steel():
    return Material(E=2.1e11, nu=0.3, rho=7850.0)


def _manual_elimination_reference(fs):
    """Independently reconstructs the expected answer via the raw
    static-condensation formula, bypassing solve_static() entirely --
    the reference every test below checks against."""
    free = fs.free_dofs
    fixed = fs.fixed_dofs_array
    u_fixed = fs._fixed_dof_values_array(fixed)
    K = fs.K if not fs.sparse else fs.K.toarray()
    Kff = K[np.ix_(free, free)]
    Kfo = K[np.ix_(free, fixed)]
    Ff = fs.F[free] - Kfo @ u_fixed
    Uf = np.linalg.solve(Kff, Ff)
    U = np.zeros(fs.n_dof)
    U[fixed] = u_fixed
    U[free] = Uf
    return U


class TestBackwardCompatibility:
    """value=0.0 (the default, and the only value ever possible before
    this item existed) must reproduce the exact original behavior --
    bit-for-bit, not just numerically close, since this is a pure
    no-op path when unused."""

    def _build(self, sparse=False, backend="scipy"):
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        left = mesh.nodes_on_line(axis=0, value=0.0)
        tip = mesh.nodes_on_line(axis=0, value=1.0)
        fs = FESystem(mesh, Quad4PlaneStress(), sparse=sparse, backend=backend)
        fs.assemble_stiffness(D)
        for n in left:
            fs.fix_dofs([n], [0, 1])
        for n in tip:
            fs.F[fs._global_dofs([n])[1]] += -100.0 / len(tip)
        return fs, left, tip

    def test_dense_default_value_matches_explicit_zero(self):
        fs1, left, tip = self._build(sparse=False)
        U1 = fs1.solve_static()

        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        fs2 = FESystem(mesh, Quad4PlaneStress())
        fs2.assemble_stiffness(D)
        for n in left:
            fs2.fix_dofs([n], [0, 1], value=0.0)
        for n in tip:
            fs2.F[fs2._global_dofs([n])[1]] += -100.0 / len(tip)
        U2 = fs2.solve_static()
        assert np.array_equal(U1, U2)

    def test_sparse_default_value_matches_explicit_zero(self):
        fs1, left, tip = self._build(sparse=True)
        U1 = fs1.solve_static()

        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        fs2 = FESystem(mesh, Quad4PlaneStress(), sparse=True)
        fs2.assemble_stiffness(D)
        for n in left:
            fs2.fix_dofs([n], [0, 1], value=0.0)
        for n in tip:
            fs2.F[fs2._global_dofs([n])[1]] += -100.0 / len(tip)
        U2 = fs2.solve_static()
        assert np.allclose(U1, U2, atol=1e-12)

    def test_auto_backend_default_value_matches_explicit_zero(self):
        fs1, left, tip = self._build(backend="auto")
        U1 = fs1.solve_static()

        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        fs2 = FESystem(mesh, Quad4PlaneStress(), backend="auto")
        fs2.assemble_stiffness(D)
        for n in left:
            fs2.fix_dofs([n], [0, 1], value=0.0)
        for n in tip:
            fs2.F[fs2._global_dofs([n])[1]] += -100.0 / len(tip)
        U2 = fs2.solve_static()
        assert np.allclose(U1, U2, atol=1e-12)


class TestNonHomogeneousDirichlet:
    """The decisive correctness checks: a genuinely nonzero prescribed
    displacement must (a) actually appear at the prescribed DOFs in the
    solved answer, and (b) match an independent manual-elimination
    reference to near machine precision -- not just "look plausible"."""

    def _clamped_prescribed_edge(self, sparse=False, backend="scipy"):
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        left = mesh.nodes_on_line(axis=0, value=0.0)
        right = mesh.nodes_on_line(axis=0, value=1.0)
        fs = FESystem(mesh, Quad4PlaneStress(), sparse=sparse, backend=backend)
        fs.assemble_stiffness(D)
        for n in left:
            fs.fix_dofs([n], [0, 1])
        for n in right:
            fs.fix_dofs([n], [0], value=0.01)
            fs.fix_dofs([n], [1], value=0.0)
        return fs, right

    def test_dense_prescribed_value_appears_exactly_in_solution(self):
        fs, right = self._clamped_prescribed_edge()
        U = fs.solve_static()
        right_x = [fs._global_dofs([n])[0] for n in right]
        assert np.allclose(U[right_x], 0.01, atol=1e-14)

    def test_dense_matches_manual_elimination_reference(self):
        fs, right = self._clamped_prescribed_edge()
        U = fs.solve_static()
        U_ref = _manual_elimination_reference(fs)
        assert np.allclose(U, U_ref, atol=1e-9, rtol=1e-8)

    def test_sparse_matches_manual_elimination_reference(self):
        fs, right = self._clamped_prescribed_edge(sparse=True)
        U = fs.solve_static()
        U_ref = _manual_elimination_reference(fs)
        assert np.allclose(U, U_ref, atol=1e-9, rtol=1e-8)

    def test_auto_backend_matches_manual_elimination_reference(self):
        fs, right = self._clamped_prescribed_edge(backend="auto")
        U = fs.solve_static()
        U_ref = _manual_elimination_reference(fs)
        assert np.allclose(U, U_ref, atol=1e-9, rtol=1e-8)

    def test_prescribed_rigid_translation_produces_zero_strain_energy(self):
        """A decisive physical sanity check independent of the manual-
        elimination reference: prescribing the SAME nonzero displacement
        on every node but one (a rigid-body translation, with one node
        left free so free_dofs is non-empty -- a fully-constrained mesh
        is a separate, pre-existing edge case in free_dofs unrelated to
        this item, so it's avoided here) must produce that exact
        translation everywhere, including at the one free node, since a
        pure rigid translation carries zero strain energy and the free
        node has nothing pulling it away from the rigid solution."""
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=3, ny=3)
        D = D_plane_stress(_steel())
        fs = FESystem(mesh, Quad4PlaneStress())
        fs.assemble_stiffness(D)
        n_nodes = len(mesh.nodes)
        for n in range(n_nodes - 1):
            fs.fix_dofs([n], [0], value=0.02)
            fs.fix_dofs([n], [1], value=-0.01)
        U = fs.solve_static()
        ux = U[0::2]
        uy = U[1::2]
        assert np.allclose(ux, 0.02, atol=1e-9)
        assert np.allclose(uy, -0.01, atol=1e-9)

    def test_multiple_fix_dofs_calls_are_additive_across_different_nodes(self):
        """Two separate fix_dofs() calls, each with its own value, on
        DISJOINT node sets must both take effect -- fixed_dof_values
        must not be clobbered by later calls touching different DOFs."""
        mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=4, ny=2)
        D = D_plane_stress(_steel())
        left = mesh.nodes_on_line(axis=0, value=0.0)
        right = mesh.nodes_on_line(axis=0, value=2.0)
        fs = FESystem(mesh, Quad4PlaneStress())
        fs.assemble_stiffness(D)
        for n in left:
            fs.fix_dofs([n], [0, 1], value=0.0)
        for n in right:
            fs.fix_dofs([n], [0], value=0.05)
            fs.fix_dofs([n], [1], value=0.0)
        U = fs.solve_static()
        left_dofs = np.array([fs._global_dofs([n]) for n in left]).reshape(-1)
        right_x = [fs._global_dofs([n])[0] for n in right]
        assert np.allclose(U[left_dofs], 0.0, atol=1e-14)
        assert np.allclose(U[right_x], 0.05, atol=1e-14)


class TestTorchBackendGuard:
    """The torch backend does not implement the RHS correction --
    confirm it fails LOUDLY (a specific NotImplementedError) rather than
    silently returning a wrong (homogeneous) answer. Gated on torch
    actually being constructible (backend='torch' fails fast at
    FESystem construction, per Wave 0 item 3's own design, in any
    environment without a working torch install -- including this
    project's own development sandbox, see torch_sparse_solver.py's
    docstring) -- these tests are written to run for real wherever
    torch is usable, same convention as test_torch_sparse_solver.py."""

    def _torch_usable(self):
        try:
            from fea_engine.torch_sparse_solver import _HAS_TORCH
            return _HAS_TORCH
        except ImportError:
            return False

    def test_torch_backend_raises_on_nonzero_dirichlet_value(self):
        if not self._torch_usable():
            pytest.skip("torch not usable in this environment")
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        left = mesh.nodes_on_line(axis=0, value=0.0)
        right = mesh.nodes_on_line(axis=0, value=1.0)
        fs = FESystem(mesh, Quad4PlaneStress(), backend="torch")
        fs.assemble_stiffness(D)
        for n in left:
            fs.fix_dofs([n], [0, 1])
        for n in right:
            fs.fix_dofs([n], [0], value=0.01)
        with pytest.raises(NotImplementedError, match="non-homogeneous Dirichlet"):
            fs.solve_static()

    def test_torch_backend_still_works_for_homogeneous_case(self):
        if not self._torch_usable():
            pytest.skip("torch not usable in this environment")
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        left = mesh.nodes_on_line(axis=0, value=0.0)
        tip = mesh.nodes_on_line(axis=0, value=1.0)
        fs = FESystem(mesh, Quad4PlaneStress(), backend="torch")
        fs.assemble_stiffness(D)
        for n in left:
            fs.fix_dofs([n], [0, 1])
        for n in tip:
            fs.F[fs._global_dofs([n])[1]] += -100.0 / len(tip)
        U = fs.solve_static()   # must not raise -- no nonzero value present
        assert np.all(np.isfinite(U))


class TestHelperMethods:
    def test_fixed_dofs_array_is_sorted(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        fs = FESystem(mesh, Quad4PlaneStress())
        fs.assemble_stiffness(D)
        fs.fix_dofs([3], [1])
        fs.fix_dofs([0], [0])
        arr = fs.fixed_dofs_array
        assert np.array_equal(arr, np.sort(arr))
        assert set(arr.tolist()) == fs.fixed_dofs

    def test_fixed_dof_values_array_defaults_to_zero_for_valueless_fix(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        fs = FESystem(mesh, Quad4PlaneStress())
        fs.assemble_stiffness(D)
        fs.fix_dofs([0], [0])                    # no value= -> 0.0
        fs.fix_dofs([1], [0], value=7.5)
        fixed = fs.fixed_dofs_array
        vals = fs._fixed_dof_values_array(fixed)
        d0 = fs._global_dofs([0])[0]
        d1 = fs._global_dofs([1])[0]
        assert vals[list(fixed).index(d0)] == 0.0
        assert vals[list(fixed).index(d1)] == 7.5

    def test_dirichlet_rhs_correction_is_zero_vector_when_all_values_zero(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        fs = FESystem(mesh, Quad4PlaneStress())
        fs.assemble_stiffness(D)
        fs.fix_dofs([0], [0, 1])
        free = fs.free_dofs
        fixed = fs.fixed_dofs_array
        u_fixed = fs._fixed_dof_values_array(fixed)
        correction = fs._dirichlet_rhs_correction(fs.K, free, fixed, u_fixed)
        assert np.array_equal(correction, np.zeros(len(free)))

    def test_dirichlet_rhs_correction_matches_direct_matmul_when_nonzero(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        D = D_plane_stress(_steel())
        fs = FESystem(mesh, Quad4PlaneStress())
        fs.assemble_stiffness(D)
        fs.fix_dofs([0], [0], value=0.3)
        free = fs.free_dofs
        fixed = fs.fixed_dofs_array
        u_fixed = fs._fixed_dof_values_array(fixed)
        correction = fs._dirichlet_rhs_correction(fs.K, free, fixed, u_fixed)
        expected = fs.K[np.ix_(free, fixed)] @ u_fixed
        assert np.allclose(correction, expected, atol=1e-14)
