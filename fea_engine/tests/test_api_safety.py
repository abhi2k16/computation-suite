"""
test_api_safety.py -- v1.0.1 input-validation / safe re-assembly guards on FESystem.

Each test pins one behaviour found by probing the interface against PyMAPDL/PyDPF design practice
(see FEA_ENGINE_API_REVIEW.md): silent wrong answers became clear errors or idempotent behaviour,
and previously valid usage is unchanged (checked against the README example numbers).
"""
import warnings

import numpy as np
import pytest

from fea_engine import Material, D_plane_stress, Quad4PlaneStress, FESystem
from fea_engine.mesh import rectangle_mesh
from fea_engine.geometry import generate_mesh
from fea_engine.material import Section, EI_beam
from fea_engine.elements import Beam2DEulerBernoulli

MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)
D = D_plane_stress(MAT)


def _plane(nx=6, ny=3, sparse=False):
    mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=nx, ny=ny)
    return mesh, FESystem(mesh, Quad4PlaneStress(), thickness=0.02, sparse=sparse)


def _dense(K):
    return K.toarray() if hasattr(K, "toarray") else np.asarray(K)


# ------------------------------------------------------------------ fix_dofs validation
class TestFixDofsValidation:
    def test_local_dof_out_of_range_raises_and_names_the_valid_range(self):
        mesh, s = _plane()
        with pytest.raises(ValueError, match=r"Quad4PlaneStress has 2 DOF\(s\) per node.*0\.\.1"):
            s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 7])
        assert not s.fixed_dofs, "a rejected call must not leave partial constraints behind"

    def test_negative_local_dof_rejected(self):
        mesh, s = _plane()
        with pytest.raises(ValueError, match="out of range"):
            s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [-1])

    def test_non_integer_dof_rejected(self):
        mesh, s = _plane()
        with pytest.raises(ValueError, match="integers"):
            s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0.5])

    def test_empty_selection_raises_clear_error(self):
        mesh, s = _plane()
        with pytest.raises(ValueError, match="no nodes selected"):
            s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.41), [0, 1])

    def test_node_id_outside_mesh_rejected(self):
        mesh, s = _plane()
        with pytest.raises(ValueError, match="outside the mesh"):
            s.fix_dofs([len(mesh.nodes)], [0])
        with pytest.raises(ValueError, match="outside the mesh"):
            s.fix_dofs([-1], [0])

    def test_valid_calls_unchanged(self):
        mesh, s = _plane()
        nodes = mesh.nodes_on_line(axis=0, value=0.0)
        s.fix_dofs(nodes, [0, 1])
        assert s.fixed_dofs == {2 * int(n) + k for n in nodes for k in (0, 1)}
        s.fix_dofs(list(nodes[:1]), (1,), value=1e-3)          # list / tuple / nonzero value still fine
        assert s.fixed_dof_values[2 * int(nodes[0]) + 1] == 1e-3
        s.fix_dofs(set(int(n) for n in nodes), np.array([0]))  # set of ids, ndarray dofs
        s.fix_dofs(range(2), 0)                                # scalar dof index


# ------------------------------------------------------------------ load validation
class TestLoadValidation:
    def test_add_nodal_force_empty_selection_is_not_a_zero_division(self):
        mesh, s = _plane()
        with pytest.raises(ValueError, match="no nodes selected"):
            s.add_nodal_force(mesh.nodes_on_line(axis=0, value=0.41), dof_index=1, total_force=-1.0)

    def test_add_nodal_force_bad_dof_rejected_and_F_untouched(self):
        mesh, s = _plane()
        with pytest.raises(ValueError, match="out of range"):
            s.add_nodal_force(mesh.nodes_on_line(axis=0, value=0.4), dof_index=2, total_force=-1.0)
        with pytest.raises(ValueError, match="out of range"):
            s.add_nodal_force(mesh.nodes_on_line(axis=0, value=0.4), dof_index=-1, total_force=-1.0)
        assert not np.any(s.F)

    def test_add_nodal_force_valid_unchanged(self):
        mesh, s = _plane()
        tip = mesh.nodes_on_line(axis=0, value=0.4)
        s.add_nodal_force(tip, dof_index=1, total_force=-20000.0)
        assert s.F.sum() == pytest.approx(-20000.0)

    def test_edge_load_validation(self):
        mesh, s = _plane()
        with pytest.raises(ValueError, match="no nodes selected"):
            s.add_consistent_edge_load([], dof_index=1, traction=1.0)
        with pytest.raises(ValueError, match="out of range"):
            s.add_consistent_edge_load([(0, 1)], dof_index=5, traction=1.0)
        s.add_consistent_edge_load([(0, 1)], dof_index=1, traction=1.0)   # valid still works
        assert np.any(s.F)


# ------------------------------------------------------------------ safe re-assembly
class TestReassembly:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_assembling_stiffness_twice_replaces_instead_of_doubling(self, sparse):
        _, s = _plane(sparse=sparse)
        s.assemble_stiffness(D, thickness=0.02)
        K1 = _dense(s.K).copy()
        with pytest.warns(UserWarning, match="already assembled"):
            s.assemble_stiffness(D, thickness=0.02)
        assert np.allclose(_dense(s.K), K1)

    def test_warning_is_emitted_once_per_matrix(self):
        _, s = _plane()
        s.assemble_stiffness(D, thickness=0.02)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.assemble_stiffness(D, thickness=0.02)
            s.assemble_stiffness(D, thickness=0.02)
        assert len([x for x in w if "already assembled" in str(x.message)]) == 1

    def test_accumulate_true_still_adds_explicitly_and_silently(self):
        _, s = _plane()
        s.assemble_stiffness(D, thickness=0.02)
        K1 = _dense(s.K).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s.assemble_stiffness(D, thickness=0.02, accumulate=True)
        assert np.allclose(_dense(s.K), 2 * K1)

    def test_mass_and_lumped_mass_are_idempotent_too(self):
        _, s = _plane()
        rho = MAT.rho * np.eye(2)
        s.assemble_mass(rho, thickness=0.02)
        M1 = _dense(s.M).copy()
        with pytest.warns(UserWarning, match="M was already assembled"):
            s.assemble_mass(rho, thickness=0.02)
        assert np.allclose(_dense(s.M), M1)
        s.assemble_lumped_mass(rho, thickness=0.02)
        L1 = _dense(s.M_lumped).copy()
        with pytest.warns(UserWarning, match="M_lumped was already assembled"):
            s.assemble_lumped_mass(rho, thickness=0.02)
        assert np.allclose(_dense(s.M_lumped), L1)

    def test_first_assembly_emits_no_warning_and_vectorized_path_also_replaces(self):
        _, s = _plane()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s.assemble_stiffness(D, thickness=0.02, vectorized=True)
        K1 = _dense(s.K).copy()
        with pytest.warns(UserWarning, match="already assembled"):
            s.assemble_stiffness(D, thickness=0.02, vectorized=True)
        assert np.allclose(_dense(s.K), K1)

    def test_failed_reassembly_leaves_the_previous_matrix_intact(self):
        _, s = _plane()
        s.assemble_stiffness(D, thickness=0.02, vectorized=True)
        K1 = _dense(s.K).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError):
                s.assemble_stiffness(D, thickness=0.02, vectorized=True, chunk_size=-5)
            assert np.allclose(_dense(s.K), K1), "an error during re-assembly must not wipe K"
            s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
            M1 = _dense(s.M).copy()
            with pytest.raises(ValueError):
                s.assemble_mass(MAT.rho, thickness=0.02)           # scalar rho on a plane element
            assert np.allclose(_dense(s.M), M1)

    def test_failed_first_assembly_is_not_reported_as_a_reassembly(self):
        _, s = _plane()
        with pytest.raises(ValueError):
            s.assemble_stiffness(D, thickness=0.02, vectorized=True, chunk_size=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s.assemble_stiffness(D, thickness=0.02, vectorized=True)    # first successful assembly: no warning
        assert np.any(s.K)

    def test_warning_points_at_the_callers_line(self):
        _, s = _plane()
        s.assemble_stiffness(D, thickness=0.02)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.assemble_stiffness(D, thickness=0.02)
        assert w and w[0].filename.endswith("test_api_safety.py")

    def test_a_system_built_without_init_state_still_assembles(self):
        _, s = _plane()
        del s._assembled, s._warned_reassembly           # e.g. an object unpickled from an older version
        s.assemble_stiffness(D, thickness=0.02)
        assert np.any(s.K)


# ------------------------------------------------------------------ solve guards
class TestSolveGuards:
    def test_solve_static_before_assembly_raises(self):
        mesh, s = _plane()
        s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
        with pytest.raises(RuntimeError, match="assemble_stiffness"):
            s.solve_static()

    def test_zero_load_vector_warns(self):
        mesh, s = _plane()
        s.assemble_stiffness(D, thickness=0.02)
        s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
        with pytest.warns(UserWarning, match="all zeros"):
            U = s.solve_static()
        assert not np.any(U)

    def test_prescribed_displacement_with_zero_load_does_not_warn(self):
        mesh, s = _plane()
        s.assemble_stiffness(D, thickness=0.02)
        s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
        s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.4), [0], value=1e-4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            U = s.solve_static()
        assert np.any(U)

    def test_unconstrained_pure_neumann_solve_remains_supported(self):
        """solve_static documents support for singular/free-body systems; the guards must not break it."""
        mesh, s = _plane()
        s.assemble_stiffness(D, thickness=0.02)
        # axial tension: equal and opposite x-forces on the two ends, collinear -> zero net force and moment
        s.add_nodal_force(mesh.nodes_on_line(axis=0, value=0.4), dof_index=0, total_force=10.0)
        s.add_nodal_force(mesh.nodes_on_line(axis=0, value=0.0), dof_index=0, total_force=-10.0)
        U = s.solve_static()                      # self-equilibrated load on a free body
        assert np.all(np.isfinite(U))
        # and the solver's own existing check for a NON-equilibrated free body is untouched
        s.add_nodal_force(mesh.nodes_on_line(axis=0, value=0.4), dof_index=1, total_force=-10.0)
        with pytest.raises(np.linalg.LinAlgError, match="under-constrained"):
            s.solve_static()

    def test_modal_without_mass_or_stiffness_raises_runtime_error(self):
        mesh, s = _plane()
        s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
        with pytest.raises(RuntimeError, match="assemble_mass"):
            s.solve_modal(n_modes=2)
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        with pytest.raises(RuntimeError, match="assemble_stiffness"):
            s.solve_modal(n_modes=2)


# ------------------------------------------------------------------ scalar density message
class TestScalarDensity:
    def test_scalar_rho_on_plane_element_gives_actionable_message(self):
        _, s = _plane()
        with pytest.raises(ValueError, match=r"np\.eye\(n\)"):
            s.assemble_mass(MAT.rho, thickness=0.02)
        with pytest.raises(ValueError, match="density MATRIX"):
            s.assemble_lumped_mass(MAT.rho, thickness=0.02)

    def test_density_matrix_works_and_beams_still_take_a_scalar(self):
        mesh, s = _plane()
        s.assemble_stiffness(D, thickness=0.02)
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
        f, _ = s.solve_modal(n_modes=2)
        assert f[0] > 0 and f[1] > f[0]
        # README E3: Euler-Bernoulli cantilever, scalar rho*A, first mode 83.80 Hz
        m = Material(E=210e9, nu=0.3, rho=7800.0)
        beam = FESystem(generate_mesh(dim=1, L=1.0, n=40), Beam2DEulerBernoulli())
        beam.assemble_stiffness(EI_beam(m, Section(A=0.01, I=8.33e-6)))
        beam.assemble_mass(m.rho * 0.01)
        beam.fix_dofs([0], [0, 1])
        fb, _ = beam.solve_modal(n_modes=3)
        assert fb[0] == pytest.approx(83.80, abs=0.01)


# ------------------------------------------------------------------ numerics unchanged
def test_readme_example_e1_numbers_unchanged():
    """README / API_REFERENCE E1: 24x12 plane-stress cantilever, mean tip U_y = -1.7969e-4 m."""
    mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=24, ny=12)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
    s.assemble_stiffness(D, thickness=0.02)
    tip = mesh.nodes_on_line(axis=0, value=0.4)
    s.add_nodal_force(tip, dof_index=1, total_force=-20000.0)
    s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
    U = s.solve_static()
    assert len(mesh.nodes) == 325 and s.n_dof == 650
    assert U[2 * tip + 1].mean() == pytest.approx(-1.7969e-4, rel=1e-3)
