# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_supports_and_constraints.py -- springs, elastic foundations (Robin terms) and linear constraints.

Every constraint result is checked against an independent Lagrange-multiplier (KKT) solve or a null-space
basis built in the test, not against the code under test.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import warnings

import numpy as np
import pytest
from scipy.linalg import eigh, null_space

import fea_engine
from fea_engine import (FESystem, Material, D_plane_stress, D_solid3d, Quad4PlaneStress, Hex8Solid3D,
                        Beam2DReissner, NewtonOptions)
from fea_engine import facet_loads, nonlinear_solver as ns
from fea_engine.mesh import rectangle_mesh, box_mesh, Mesh

MAT0 = Material(E=2.1e11, nu=0.0, rho=7850.0)
MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)
L, H, T = 1.0, 0.1, 0.05


def _bar(nx=8, ny=1, sparse=False, mat=MAT0, load=1.0e5, fix_uy=True):
    mesh = rectangle_mesh(Lx=L, Ly=H, nx=nx, ny=ny)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=T, sparse=sparse)
    s.assemble_stiffness(D_plane_stress(mat), thickness=T)
    mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=L, name="tip")
    s.fix_dofs("root", ["ux"])
    if fix_uy:
        s.fix_dofs(mesh.select_nodes(x=0.0, y=0.0), ["uy"])
    if load:
        s.add_nodal_force("tip", "ux", load)
    return mesh, s


K_BAR = MAT0.E * H * T / L


# ====================================================================== springs
class TestSprings:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_parallel_spring_stiffens_the_tip(self, sparse):
        P, k = 1.0e5, 3.0e8
        mesh, s = _bar(sparse=sparse, load=P)
        s.add_spring("tip", "ux", k)                       # one spring per tip node -> 2k in total
        u = s.solve_static().component("ux", nodes="tip").mean()
        assert u == pytest.approx(P / (K_BAR + 2 * k), rel=1e-9)

    def test_spring_to_displaced_support(self):
        P, k, u0 = 1.0e5, 3.0e8, 2.0e-4
        mesh, s = _bar(load=P)
        s.add_spring("tip", "ux", k, u_ref=u0)
        u = s.solve_static().component("ux", nodes="tip").mean()
        assert u == pytest.approx((P + 2 * k * u0) / (K_BAR + 2 * k), rel=1e-9)

    def test_per_node_stiffness_and_multiple_dofs(self):
        mesh, s = _bar(load=0.0)
        n = s._check_nodes("tip", "t")
        K0 = np.array(s.K, copy=True)
        s.add_spring("tip", ["ux", "uy"], np.array([1.0e6, 2.0e6]))
        dK = np.array(s.K) - K0
        for node, kk in zip(n, [1.0e6, 2.0e6]):
            for d in (0, 1):
                assert dK[2 * node + d, 2 * node + d] == pytest.approx(kk)
        assert np.count_nonzero(dK) == 4

    def test_order_independent_and_survives_reassembly(self):
        mesh = rectangle_mesh(Lx=L, Ly=H, nx=4, ny=1)
        D = D_plane_stress(MAT0)
        a = FESystem(mesh, Quad4PlaneStress(), thickness=T)
        a.add_spring([3], "uy", 1.0e7)                     # before assembly
        a.assemble_stiffness(D, thickness=T)
        b = FESystem(mesh, Quad4PlaneStress(), thickness=T)
        b.assemble_stiffness(D, thickness=T)
        b.add_spring([3], "uy", 1.0e7)                     # after assembly
        assert np.allclose(a.K, b.K)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b.assemble_stiffness(D, thickness=T)           # replacement re-applies the spring, once
        assert np.allclose(a.K, b.K)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b.assemble_stiffness(D, thickness=T, accumulate=True)   # adds a second plain copy only
        assert b.K[7, 7] == pytest.approx(a.K[7, 7] + (a.K[7, 7] - 1.0e7))

    def test_modal_frequencies_rise_with_a_spring(self):
        mesh, s = _bar(nx=6, load=0.0)
        s.assemble_mass(MAT0.rho * np.eye(2), thickness=T)
        f0, _ = s.solve_modal(n_modes=3)
        s.add_spring("tip", "ux", 5.0e9)
        f1, _ = s.solve_modal(n_modes=3)
        assert np.all(f1 > f0 * 1.05)
        free = s.free_dofs
        ref = np.sqrt(eigh(np.asarray(s.K)[np.ix_(free, free)], np.asarray(s.M)[np.ix_(free, free)],
                           eigvals_only=True)[:3]) / (2 * np.pi)
        assert np.allclose(f1, ref, rtol=1e-9)

    def test_validation(self):
        mesh, s = _bar(load=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            s.add_spring("tip", "ux", -1.0)
        with pytest.raises(ValueError, match="unknown DOF"):
            s.add_spring("tip", "uz", 1.0)
        with pytest.raises(ValueError, match="unknown node set"):
            s.add_spring("nowhere", "ux", 1.0)


# ====================================================================== foundations / Robin
class TestFoundation:
    def test_facet_matrix_line2_and_total(self):
        c = np.array([[0.0, 0.0], [2.0, 0.0]])
        m = facet_loads.consistent_facet_mass(2, c, 3.0)
        assert np.allclose(m, 3.0 * 2.0 / 6.0 * np.array([[2, 1], [1, 2]]))
        mk = facet_loads.consistent_facet_mass(2, c, lambda x: 1.0 + x[0])
        assert mk.sum() == pytest.approx(2.0 + 2.0)        # integral of (1 + x) over [0, 2]

    def test_quadratic_edge_and_3d_face_totals(self):
        c3 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
        assert facet_loads.consistent_facet_mass(2, c3, 5.0).sum() == pytest.approx(5.0)
        quad = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0.0]])
        assert facet_loads.consistent_facet_mass(3, quad, 4.0).sum() == pytest.approx(4.0 * 6.0)

    def _column_on_foundation(self, kf, u_ref=0.0, P=1.0e5):
        mesh = rectangle_mesh(Lx=H, Ly=L, nx=1, ny=4)       # width H, height L, nu = 0: uniform strain
        s = FESystem(mesh, Quad4PlaneStress(), thickness=T)
        s.assemble_stiffness(D_plane_stress(MAT0), thickness=T)
        mesh.select_nodes(y=0.0, name="bottom"); mesh.select_nodes(y=L, name="top")
        s.fix_dofs("bottom", ["ux"])
        s.fix_dofs(mesh.select_nodes(y=L, x=0.0), ["ux"])
        s.add_elastic_foundation("bottom", "uy", kf, thickness=T, u_ref=u_ref)
        s.add_nodal_force("top", "uy", -P)
        return mesh, s

    @pytest.mark.parametrize("sparse", [False])
    def test_column_on_winkler_foundation_matches_hand_solution(self, sparse):
        kf, P = 4.0e9, 1.0e5                                  # per unit area
        mesh, s = self._column_on_foundation(kf, P=P)
        U = s.solve_static()
        u_bottom = -P / (kf * H * T)
        u_top = u_bottom - P * L / (MAT0.E * H * T)
        assert U.component("uy", nodes="bottom").mean() == pytest.approx(u_bottom, rel=1e-9)
        assert U.component("uy", nodes="top").mean() == pytest.approx(u_top, rel=1e-9)

    def test_robin_with_reference_displacement(self):
        kf, P, u0 = 4.0e9, 1.0e5, 3.0e-4
        mesh, s = self._column_on_foundation(kf, u_ref=u0, P=P)
        U = s.solve_static()
        assert U.component("uy", nodes="bottom").mean() == pytest.approx(u0 - P / (kf * H * T), rel=1e-9)

    def test_explicit_facets_and_facets_on(self):
        from fea_engine.boundary import facets_on
        mesh = rectangle_mesh(Lx=1.0, Ly=0.2, nx=3, ny=1)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=1.0)
        s.assemble_stiffness(D_plane_stress(MAT0), thickness=1.0)
        bottom = mesh.select_nodes(y=0.0, name="bottom")
        facets = facets_on(s, bottom)
        assert len(facets) == 3 and all(len(f) == 2 for f in facets)
        K0 = np.array(s.K, copy=True)
        s.add_elastic_foundation(facets, "ux", 2.0)
        assert (np.array(s.K) - K0).sum() == pytest.approx(2.0 * 1.0, abs=1e-3)   # k times boundary length
        with pytest.raises(ValueError, match="no element facet"):
            s.add_elastic_foundation([int(bottom[0])], "ux", 2.0)

    def test_3d_face_foundation(self):
        mesh = box_mesh(1.0, 2.0, 0.5, 2, 2, 1)
        s = FESystem(mesh, Hex8Solid3D())
        s.assemble_stiffness(D_solid3d(MAT))
        mesh.select_nodes(z=0.0, name="floor")
        K0 = np.array(s.K, copy=True)
        s.add_elastic_foundation("floor", "uz", 3.0)
        dK = np.array(s.K) - K0
        assert dK.sum() == pytest.approx(3.0 * 1.0 * 2.0)                    # k times floor area
        assert np.allclose(dK, dK.T)

    def test_validation(self):
        mesh, s = _bar(load=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            s.add_elastic_foundation("tip", "ux", -2.0)


# ====================================================================== springs in nonlinear drivers
def _cantilever_reissner(n=8):
    E, nu, A, I, Lb = 210e9, 0.3, 1e-3, 8.33e-7, 1.0
    G = E / (2 * (1 + nu))
    x = np.linspace(0, Lb, n + 1).reshape(-1, 1)
    s = FESystem(Mesh(nodes=np.hstack([x, 0 * x]), elements=np.array([[i, i + 1] for i in range(n)]), dim=1),
                 Beam2DReissner())
    s.fix_dofs([0], ["ux", "uy", "rz"])
    mat = (E, G, A, I, 1.0)
    s.assemble_stiffness(mat)
    return s, mat, 3 * E * I / Lb ** 3


class TestSpringsInNonlinearDrivers:
    def test_small_load_matches_linear_spring_beam(self):
        s, mat, kb = _cantilever_reissner()
        ks = 2.0 * kb
        s.add_spring([8], "uy", ks)
        P = 1.0e-3 * kb                                       # deflection ~1e-3 of the length
        s.add_nodal_force([8], "uy", P)
        lf, hist = ns.solve_nonlinear_static(s, mat, n_steps=2, options=NewtonOptions(tol=1e-8, max_iter=50))
        tip = s.field(hist[-1]).component("uy", nodes=[8])[0]
        assert tip == pytest.approx(P / (kb + ks), rel=2e-3)

    def test_vectorized_beam_assembly_refuses_springs(self):
        from fea_engine.beam2d_reissner_vectorized import assemble_internal_force_vectorized
        s, mat, kb = _cantilever_reissner(4)
        s.add_spring([4], "uy", kb)
        with pytest.raises(NotImplementedError, match="springs"):
            assemble_internal_force_vectorized(s, np.zeros(s.n_dof), mat)


# ====================================================================== constraints
def _kkt(s, rows, values):
    """Independent reference: Lagrange-multiplier solve with prescribed DOFs handled by partitioning."""
    K = s.K.toarray() if hasattr(s.K, "toarray") else np.asarray(s.K)
    n = s.n_dof
    fixed = s.fixed_dofs_array
    free = s.free_dofs
    uc = np.array([s.fixed_dof_values.get(int(d), 0.0) for d in fixed])
    C = np.zeros((len(rows), n))
    for i, r in enumerate(rows):
        for d, c in r.items():
            C[i, d] = c
    Cf, Cc = C[:, free], C[:, fixed]
    g = np.asarray(values, dtype=float) - Cc @ uc
    Kff = K[np.ix_(free, free)]
    Ff = s.F[free] - K[np.ix_(free, fixed)] @ uc
    m = len(rows)
    sc = np.abs(np.diag(Kff)).mean()                 # scale the constraint rows to the size of K
    A = np.block([[Kff, sc * Cf.T], [sc * Cf, np.zeros((m, m))]])
    sol = np.linalg.solve(A, np.concatenate([Ff, sc * g]))
    U = np.zeros(n)
    U[fixed] = uc
    U[free] = sol[:len(free)]
    return U


def _plate(nx=4, ny=2, load=2.0e4):
    mesh = rectangle_mesh(Lx=0.8, Ly=0.4, nx=nx, ny=ny)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
    s.assemble_stiffness(D_plane_stress(MAT), thickness=0.02)
    mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=0.8, name="tip")
    s.fix_dofs("root", ["ux", "uy"])
    s.add_nodal_force("tip", "uy", -load)
    return mesh, s


class TestConstraints:
    def test_equal_displacement_constraint_matches_kkt(self):
        mesh, s = _plate()
        a, b = int(mesh.select_nodes(x=0.8, y=0.0)[0]), int(mesh.select_nodes(x=0.8, y=0.4)[0])
        s.add_constraint([(a, "ux", 1.0), (b, "ux", -1.0)])
        U = s.solve_static()
        ref = _kkt(s, [{2 * a: 1.0, 2 * b: -1.0}], [0.0])
        assert np.allclose(np.asarray(U), ref, rtol=1e-8, atol=1e-14)
        assert U[2 * a] == pytest.approx(U[2 * b], abs=1e-14)

    def test_constraint_changes_the_answer(self):
        mesh, s = _plate()
        free_u = np.asarray(s.solve_static())
        a, b = int(mesh.select_nodes(x=0.8, y=0.0)[0]), int(mesh.select_nodes(x=0.8, y=0.4)[0])
        s.add_constraint([(a, "ux", 1.0), (b, "ux", -1.0)])
        assert not np.allclose(free_u, np.asarray(s.solve_static()))
        s.clear_constraints()
        assert np.allclose(free_u, np.asarray(s.solve_static()))

    def test_inhomogeneous_weighted_and_chained_constraints(self):
        mesh, s = _plate(nx=3, ny=2)
        n = len(mesh.nodes)
        a, b, c = 5, 9, 11
        rows = [{2 * a: 1.0, 2 * b + 1: -2.0}, {2 * b + 1: 1.0, 2 * c: 0.5, 2 * a + 1: 1.0}]
        vals = [1.0e-4, -2.0e-4]
        s.add_constraint([(a, "ux", 1.0), (b, "uy", -2.0)], 1.0e-4)
        s.add_constraint([(b, "uy", 1.0), (c, "ux", 0.5), (a, "uy", 1.0)], -2.0e-4)   # shares uy of b
        U = np.asarray(s.solve_static())
        assert np.allclose(U, _kkt(s, rows, vals), rtol=1e-7, atol=1e-13)
        assert U[2 * a] - 2 * U[2 * b + 1] == pytest.approx(1.0e-4, abs=1e-12)

    def test_constraint_on_prescribed_dof(self):
        mesh, s = _plate(nx=3, ny=2)
        s.fix_dofs("tip", ["ux"], value=1.0e-4)             # prescribed (nonzero) tip ux
        a = int(mesh.select_nodes(x=0.0, y=0.4)[0])          # a root node, ux fixed at 0
        b = 6
        t = int(mesh.select_nodes(x=0.8, y=0.0)[0])
        rows = [{2 * b: 1.0, 2 * t: -1.0, 2 * a: 0.3}]
        s.add_constraint([(b, "ux", 1.0), (t, "ux", -1.0), (a, "ux", 0.3)], 5.0e-5)
        U = np.asarray(s.solve_static())
        assert np.allclose(U, _kkt(s, rows, [5.0e-5]), rtol=1e-7, atol=1e-13)
        assert U[2 * b] == pytest.approx(5.0e-5 + 1.0e-4, abs=1e-12)

    def test_periodic_tie_with_offset(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=0.2, nx=4, ny=2)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=0.1)
        s.assemble_stiffness(D_plane_stress(MAT), thickness=0.1)
        left = mesh.select_nodes(x=0.0, name="left"); right = mesh.select_nodes(x=1.0, name="right")
        s.fix_dofs(mesh.select_nodes(x=0.5, y=0.0), ["ux", "uy"])
        s.fix_dofs(mesh.select_nodes(x=0.5, y=0.2), ["ux"])
        s.add_nodal_force(mesh.select_nodes(x=0.25, y=0.2), "ux", 1.0e4)
        s.tie("right", "left", ["ux", "uy"], offset=(-1.0, 0.0))
        U = np.asarray(s.solve_static())
        for r in right:
            partner = int(left[np.argmin(np.abs(mesh.nodes[left, 1] - mesh.nodes[r, 1]))])
            assert U[2 * r:2 * r + 2] == pytest.approx(U[2 * partner:2 * partner + 2], abs=1e-13)
        assert len(s._constraints) == 2 * len(right)
        with pytest.raises(ValueError, match="no master node"):
            s.tie("right", "left", "ux")                     # offset omitted: nothing coincides

    def test_redundant_consistent_constraints_are_accepted_and_conflicting_rejected(self):
        mesh, s = _plate()
        a, b = 7, 8
        for _ in range(2):
            s.add_constraint([(a, "uy", 1.0), (b, "uy", -1.0)])
        s.add_constraint([(a, "uy", 2.0), (b, "uy", -2.0)])
        U = np.asarray(s.solve_static())
        assert U[2 * a + 1] == pytest.approx(U[2 * b + 1], abs=1e-14)
        s.add_constraint([(a, "uy", 1.0), (b, "uy", -1.0)], 1.0e-3)
        with pytest.raises(ValueError, match="inconsistent"):
            s.solve_static()

    def test_form_linear_system_with_constraints(self):
        mesh, s = _plate()
        a, b = 7, 8
        s.add_constraint([(a, "ux", 1.0), (b, "ux", -1.0)], 2.0e-5)
        rs = s.form_linear_system()
        assert rs.n_free == len(s.free_dofs) - 1 and len(rs.slave) == 1
        U = rs.recover(np.linalg.solve(rs.K, rs.F))
        assert U[2 * a] - U[2 * b] == pytest.approx(2.0e-5, abs=1e-12)
        assert np.allclose(np.asarray(U), np.asarray(s.solve_static()))
        modes = rs.expand(np.ones((rs.n_free, 2)), include_prescribed=False)
        assert modes.shape == (s.n_dof, 2) and np.allclose(modes[2 * a] - modes[2 * b], 0.0)
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        assert rs.reduce_matrix(s.M).shape == rs.K.shape

    @pytest.mark.parametrize("sparse", [False, True])
    def test_modal_matches_null_space_reference(self, sparse):
        mesh = rectangle_mesh(Lx=0.8, Ly=0.2, nx=6, ny=2)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02, sparse=sparse)
        s.assemble_stiffness(D_plane_stress(MAT), thickness=0.02)
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        mesh.select_nodes(x=0.0, name="root")
        s.fix_dofs("root", ["ux", "uy"])
        a, b, c = 20, 13, 9
        s.add_constraint([(a, "uy", 1.0), (b, "uy", -1.0)])
        s.add_constraint([(c, "ux", 1.0), (b, "ux", 1.0), (a, "ux", -2.0)])
        f, modes = s.solve_modal(n_modes=4)
        K = s.K.toarray() if sparse else np.asarray(s.K)
        M = s.M.toarray() if sparse else np.asarray(s.M)
        free = s.free_dofs
        pos = {int(d): i for i, d in enumerate(free)}
        C = np.zeros((2, len(free)))
        C[0, [pos[2 * a + 1], pos[2 * b + 1]]] = [1, -1]
        C[1, [pos[2 * c], pos[2 * b], pos[2 * a]]] = [1, 1, -2]
        Z = null_space(C)
        w2 = eigh(Z.T @ K[np.ix_(free, free)] @ Z, Z.T @ M[np.ix_(free, free)] @ Z, eigvals_only=True)[:4]
        assert np.allclose(f, np.sqrt(w2) / (2 * np.pi), rtol=1e-8)
        m0 = np.asarray(modes)[:, 0]
        assert m0[2 * a + 1] == pytest.approx(m0[2 * b + 1], abs=1e-12)
        assert m0[2 * c] + m0[2 * b] - 2 * m0[2 * a] == pytest.approx(0.0, abs=1e-12)

    def test_constraints_combined_with_springs(self):
        mesh, s = _plate()
        s.add_spring("tip", "uy", 4.0e6)
        a, b = int(mesh.select_nodes(x=0.8, y=0.0)[0]), int(mesh.select_nodes(x=0.8, y=0.4)[0])
        s.add_constraint([(a, "uy", 1.0), (b, "uy", -1.0)])
        U = np.asarray(s.solve_static())
        assert np.allclose(U, _kkt(s, [{2 * a + 1: 1.0, 2 * b + 1: -1.0}], [0.0]), rtol=1e-8, atol=1e-14)

    def test_validation_and_refusals(self):
        mesh, s = _plate()
        with pytest.raises(ValueError, match="no terms"):
            s.add_constraint([])
        with pytest.raises(ValueError, match="appears twice"):
            s.add_constraint([(3, "ux", 1.0), (3, "ux", 2.0)])
        with pytest.raises(ValueError, match="all coefficients are zero"):
            s.add_constraint([(3, "ux", 0.0), (4, "ux", 0.0)])
        with pytest.raises(ValueError, match="outside the mesh"):
            s.add_constraint([(999, "ux", 1.0)])
        s.add_constraint([(3, "ux", 1.0), (4, "ux", -1.0)])
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        with pytest.raises(NotImplementedError, match="solve_static and solve_modal only"):
            s.solve_harmonic(10.0, np.zeros(s.n_dof))
        with pytest.raises(NotImplementedError):
            s.solve_linear_buckling()
        with pytest.raises(NotImplementedError, match="default direct"):
            s.solve_static(method="cg")
        with pytest.raises(NotImplementedError, match="nonlinear"):
            s.assemble_internal_force(np.zeros(s.n_dof), D_plane_stress(MAT))
        s.clear_constraints()
        s.solve_static()
        assert s.solve_harmonic.__name__ == "solve_harmonic"


def test_top_level_exports():
    assert fea_engine.boundary.facets_on is not None
