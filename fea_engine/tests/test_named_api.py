# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_named_api.py -- v1.0.1 P1 interface: named DOFs, named node sets, FEField results.

Everything here is ADDITIVE: integer DOF indices, node-id arrays and plain-array behaviour of the
solver outputs keep working (the older tests cover that); these tests pin the new named access.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import inspect
import json
import pickle
import warnings

import numpy as np
import pytest

import fea_engine
from fea_engine import (Material, D_plane_stress, Quad4PlaneStress, FESystem, FEField, Element,
                        Beam2DEulerBernoulli, Beam2DReissner, Quad4MindlinPlate, Hex8Solid3D)
from fea_engine.material import Section, EI_beam
from fea_engine.mesh import rectangle_mesh, MultiBlockMesh, Mesh, line_mesh
from fea_engine.geometry import generate_mesh

MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)
D = D_plane_stress(MAT)


def _cantilever(nx=24, ny=12):
    mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=nx, ny=ny)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
    s.assemble_stiffness(D, thickness=0.02)
    return mesh, s


# ====================================================================== element DOF names
def _all_element_classes():
    out = []
    for name in dir(fea_engine.elements):
        obj = getattr(fea_engine.elements, name)
        if inspect.isclass(obj) and issubclass(obj, Element) and obj is not Element \
                and getattr(obj, "dofs_per_node", None):
            out.append(obj)
    return out


@pytest.mark.parametrize("cls", _all_element_classes(), ids=lambda c: c.__name__)
def test_every_element_has_consistent_dof_names(cls):
    try:
        elem = cls()
    except TypeError:
        pytest.skip(f"{cls.__name__} needs constructor arguments")
    names = elem.local_dof_names()
    assert names is not None, f"{cls.__name__}: no names for {elem.dofs_per_node} DOFs/node"
    assert len(names) == elem.dofs_per_node
    assert len(set(names)) == len(names)
    for j, n in enumerate(names):
        assert elem.dof_index(n) == j
        assert elem.dof_index(n.upper()) == j          # case-insensitive
        assert elem.dof_index(j) == j                   # integers pass through


def test_documented_dof_conventions():
    assert Quad4PlaneStress().local_dof_names() == ("ux", "uy")
    assert Hex8Solid3D().local_dof_names() == ("ux", "uy", "uz")
    assert Beam2DEulerBernoulli().local_dof_names() == ("uy", "rz")          # [v, theta]
    assert Beam2DReissner().local_dof_names() == ("ux", "uy", "rz")          # [u, v, theta]
    assert Quad4MindlinPlate().local_dof_names() == ("w", "betax", "betay")  # as documented
    assert fea_engine.Shell4MITC().local_dof_names() == ("ux", "uy", "uz", "rx", "ry", "rz")
    assert fea_engine.Beam3DEulerBernoulli().local_dof_names() == ("ux", "uy", "uz", "rx", "ry", "rz")


def test_aliases_and_shorthand():
    q = Quad4PlaneStress()
    assert q.dof_index("x") == 0 and q.dof_index("y") == 1
    assert q.dof_index("u") == 0 and q.dof_index("v") == 1
    b = Beam2DEulerBernoulli()
    assert b.dof_index("theta") == 1 and b.dof_index("w") == 0 and b.dof_index("v") == 0
    assert Quad4MindlinPlate().dof_index("uz") == 0
    assert Beam2DReissner().dof_index("theta") == 2


def test_unknown_name_error_lists_valid_names():
    with pytest.raises(ValueError, match=r"valid names: \['ux', 'uy'\]"):
        Quad4PlaneStress().dof_index("uz")
    with pytest.raises(ValueError, match="out of range"):
        Quad4PlaneStress().dof_index(5)


# ====================================================================== named DOFs in the system
class TestNamedDofsInFESystem:
    def test_dof_names_property_and_dof_index(self):
        _, s = _cantilever(4, 2)
        assert s.dof_names == ("ux", "uy")
        assert s.dof_index("y") == 1
        with pytest.raises(ValueError, match="valid names"):
            s.dof_index("rz")

    def test_fix_dofs_by_name_equals_by_index(self):
        mesh, a = _cantilever(4, 2)
        _, b = _cantilever(4, 2)
        nodes = mesh.nodes_on_line(axis=0, value=0.0)
        a.fix_dofs(nodes, [0, 1])
        b.fix_dofs(nodes, ["ux", "uy"])
        assert a.fixed_dofs == b.fixed_dofs

    def test_mixed_names_and_indices_and_single_name(self):
        mesh, s = _cantilever(4, 2)
        nodes = mesh.nodes_on_line(axis=0, value=0.0)
        s.fix_dofs(nodes, ["ux", 1])
        assert s.fixed_dofs == {2 * int(n) + k for n in nodes for k in (0, 1)}
        s.fixed_dofs.clear()
        s.fix_dofs(nodes, "uy")
        assert s.fixed_dofs == {2 * int(n) + 1 for n in nodes}
        s.fixed_dofs.clear()
        s.fix_dofs(nodes, {"ux"})                      # a set of names
        assert s.fixed_dofs == {2 * int(n) for n in nodes}

    def test_unknown_name_in_fix_dofs_is_rejected_and_names_the_call(self):
        mesh, s = _cantilever(4, 2)
        with pytest.raises(ValueError, match=r"fix_dofs: unknown DOF name 'uz'"):
            s.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), ["ux", "uz"])
        assert not s.fixed_dofs

    def test_loads_by_name(self):
        mesh, a = _cantilever(4, 2)
        _, b = _cantilever(4, 2)
        tip = mesh.nodes_on_line(axis=0, value=0.4)
        a.add_nodal_force(tip, 1, -100.0)
        b.add_nodal_force(tip, "uy", -100.0)
        assert np.array_equal(a.F, b.F)
        a.add_consistent_edge_load([(0, 1)], 1, 5.0)
        b.add_consistent_edge_load([(0, 1)], "uy", 5.0)
        assert np.array_equal(a.F, b.F)

    def test_names_unavailable_for_element_without_standard_names(self):
        class Odd(Element):
            n_nodes, dofs_per_node, dim = 2, 4, 1
        s = FESystem(line_mesh(1.0, 2), Odd())
        assert s.dof_names is None
        with pytest.raises(ValueError, match="without standard names"):
            s.fix_dofs([0], ["ux"])
        s.fix_dofs([0], [0, 3])                         # integers still fine
        assert s.fixed_dofs == {0, 3}

    def test_beam_with_named_dofs_reproduces_analytic_frequency(self):
        m = Material(E=210e9, nu=0.3, rho=7800.0)
        beam = FESystem(generate_mesh(dim=1, L=1.0, n=40), Beam2DEulerBernoulli())
        beam.assemble_stiffness(EI_beam(m, Section(A=0.01, I=8.33e-6)))
        beam.assemble_mass(m.rho * 0.01)
        beam.fix_dofs([0], ["uy", "rz"])               # clamped end by name
        f, shapes = beam.solve_modal(n_modes=3)
        assert f[0] == pytest.approx(83.80, abs=0.01)
        assert shapes.dof_names == ("uy", "rz")
        tip_shape = shapes.component("uy", nodes=[40])
        assert tip_shape.shape == (1, 3) and np.all(np.abs(tip_shape) > 0)


# ====================================================================== named node sets
class TestNodeSets:
    def test_select_nodes_matches_nodes_on_line(self):
        mesh, _ = _cantilever(6, 3)
        assert np.array_equal(mesh.select_nodes(x=0.0), mesh.nodes_on_line(axis=0, value=0.0))
        assert np.array_equal(mesh.select_nodes(y=0.2), mesh.nodes_on_line(axis=1, value=0.2))

    def test_select_nodes_ranges_and_intersections(self):
        mesh, _ = _cantilever(4, 2)               # x: 0,.1,.2,.3,.4  y: 0,.1,.2
        ids = mesh.select_nodes(x=(0.1, 0.3), y=0.0)
        assert np.allclose(sorted(mesh.nodes[ids, 0]), [0.1, 0.2, 0.3]) and np.allclose(mesh.nodes[ids, 1], 0.0)
        assert len(mesh.select_nodes(x=0.0, y=(0.0, 0.2))) == 3
        assert len(mesh.select_nodes()) == len(mesh.nodes)       # no condition -> all nodes

    def test_select_nodes_errors(self):
        mesh, _ = _cantilever(4, 2)
        with pytest.raises(ValueError, match="only 2 coordinate"):
            mesh.select_nodes(z=0.0)
        with pytest.raises(ValueError, match="lo > hi"):
            mesh.select_nodes(x=(0.3, 0.1))
        assert mesh.select_nodes(x=0.77).size == 0                # unnamed empty selection returns empty
        with pytest.raises(ValueError, match="matched nothing"):
            mesh.select_nodes(x=0.77, name="nothing")

    def test_add_node_set_validation_and_chaining(self):
        mesh, _ = _cantilever(4, 2)
        assert mesh.add_node_set("a", [3, 1, 3, 2]) is mesh
        assert list(mesh.node_set("a")) == [1, 2, 3]                # sorted, unique
        with pytest.raises(ValueError, match="no nodes given"):
            mesh.add_node_set("b", [])
        with pytest.raises(ValueError, match="outside the mesh"):
            mesh.add_node_set("b", [len(mesh.nodes)])
        with pytest.raises(ValueError, match="integers"):
            mesh.add_node_set("b", [0.5])
        with pytest.raises(ValueError, match="non-empty string"):
            mesh.add_node_set("", [0])
        mesh.add_node_set("a", [0])                                 # replacing is allowed
        assert list(mesh.node_set("a")) == [0]
        with pytest.raises(ValueError, match=r"available: \['a'\]"):
            mesh.node_set("zzz")

    def test_node_sets_are_not_shared_between_meshes(self):
        a, _ = _cantilever(2, 1)
        b, _ = _cantilever(2, 1)
        a.add_node_set("only_a", [0])
        assert "only_a" not in b.node_sets

    def test_sets_work_in_fix_dofs_and_loads(self):
        mesh, s = _cantilever(6, 3)
        mesh.select_nodes(x=0.0, name="root")
        mesh.select_nodes(x=0.4, name="tip")
        s.fix_dofs("root", ["ux", "uy"])
        s.add_nodal_force("tip", "uy", -1000.0)
        assert s.fixed_dofs == {2 * int(n) + k for n in mesh.nodes_on_line(axis=0, value=0.0) for k in (0, 1)}
        assert s.F.sum() == pytest.approx(-1000.0)

    def test_list_of_set_names_is_a_union_and_errors_list_the_available_sets(self):
        mesh, s = _cantilever(4, 2)
        mesh.select_nodes(x=0.0, name="left")
        mesh.select_nodes(x=0.4, name="right")
        s.fix_dofs(["left", "right"], "uy")
        both = np.union1d(mesh.node_set("left"), mesh.node_set("right"))
        assert s.fixed_dofs == {2 * int(n) + 1 for n in both}
        with pytest.raises(ValueError, match=r"unknown node set\(s\) \['top'\]; available: \['left', 'right'\]"):
            s.fix_dofs("top", "ux")

    def test_multiblock_mesh_supports_sets(self):
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
        mb = MultiBlockMesh(nodes, {"tri3": np.array([[0, 1, 2], [0, 2, 3]])}, dim=2)
        assert mb.add_node_set("left", mb.select_nodes(x=0.0)) is mb
        assert list(mb.node_set("left")) == [0, 3]
        s = FESystem(mb, {"tri3": fea_engine.Tri3PlaneStress()}, thickness=1.0)
        s.fix_dofs("left", ["ux", "uy"])
        assert s.fixed_dofs == {0, 1, 6, 7}

    def test_existing_selectors_unchanged(self):
        mesh, _ = _cantilever(4, 2)
        assert len(mesh.nodes_on_line(axis=0, value=0.0)) == 3
        assert mesh.node_sets == {}


# ====================================================================== FEField results
class TestFEField:
    def setup_method(self):
        self.mesh, self.s = _cantilever()
        self.mesh.select_nodes(x=0.0, name="root")
        self.mesh.select_nodes(x=0.4, name="tip")
        self.s.fix_dofs("root", ["ux", "uy"])
        self.s.add_nodal_force("tip", "uy", -20000.0)
        self.U = self.s.solve_static()
        self.tip = self.mesh.nodes_on_line(axis=0, value=0.4)

    def test_solve_static_returns_a_drop_in_ndarray(self):
        U = self.U
        assert isinstance(U, FEField) and isinstance(U, np.ndarray)
        assert U.shape == (self.s.n_dof,) and U.dtype == float
        # old-style indexing gives the README E1 number
        assert U[2 * self.tip + 1].mean() == pytest.approx(-1.7969e-4, rel=1e-3)
        raw = self.s._solve_static_raw()
        assert np.array_equal(np.asarray(U), raw), "wrapping must not change the numbers"

    def test_named_access_matches_manual_indexing(self):
        U = self.U
        assert np.array_equal(U.component("uy", nodes="tip"), U[2 * self.tip + 1])
        assert np.array_equal(U.component("y", nodes=self.tip), U[2 * self.tip + 1])      # x/y shorthand, ids
        assert np.array_equal(U.component(1, nodes=self.tip), U[2 * self.tip + 1])        # integer
        assert np.array_equal(U.displacement("v", nodes="tip"), U[2 * self.tip + 1])      # alias + method alias
        assert U.nodal.shape == (len(self.mesh.nodes), 2)
        assert np.array_equal(U.nodal[:, 0], U[0::2])
        assert U.component("ux").shape == (len(self.mesh.nodes),)
        assert U.at("tip").shape == (len(self.tip), 2)

    def test_magnitude_uses_translations_only(self):
        U = self.U
        mag = U.magnitude()
        assert np.allclose(mag, np.hypot(U[0::2], U[1::2]))
        # rotations excluded: build a beam field with a huge rotation and check it is ignored
        beam = FESystem(generate_mesh(dim=1, L=1.0, n=2), Beam2DReissner())
        v = np.zeros(beam.n_dof); v[0:3] = [3.0, 4.0, 1e6]
        f = beam.field(v)
        assert f.magnitude(nodes=[0])[0] == pytest.approx(5.0)

    def test_errors_are_clear(self):
        with pytest.raises(ValueError, match="unknown DOF 'uz'; valid names: \\['ux', 'uy'\\]"):
            self.U.component("uz")
        with pytest.raises(ValueError, match="unknown node set"):
            self.U.component("ux", nodes="nowhere")
        with pytest.raises(ValueError, match="no nodes selected"):
            self.U.component("ux", nodes=[])
        with pytest.raises(ValueError, match="outside the mesh"):
            self.U.component("ux", nodes=[10 ** 6])
        with pytest.raises(ValueError, match="probably sliced"):
            self.U[self.s.free_dofs].nodal
        with pytest.raises(ValueError, match="out of range"):
            self.U.component(7)

    def test_behaves_like_a_plain_array(self):
        U = self.U
        assert isinstance(U[::2], FEField) and isinstance(U + 1.0, FEField)
        assert (U + 1.0).is_nodal and not U[::2].is_nodal
        # reductions give NumPy scalars exactly like a plain ndarray (json, np.isscalar, formatting)
        for val in (U.max(), U.sum(), U @ U, np.linalg.norm(U), U[3]):
            assert np.isscalar(val) and not isinstance(val, np.ndarray)
        json.dumps([float(U.max()), U.max(), U.sum()])
        assert isinstance((U > 0).any(), (bool, np.bool_))
        assert np.allclose(U, np.asarray(U))
        assert U.copy().dof_names == ("ux", "uy")

    def test_pickle_roundtrip_keeps_names_but_drops_the_mesh(self):
        U2 = pickle.loads(pickle.dumps(self.U))
        assert isinstance(U2, FEField) and np.array_equal(np.asarray(U2), np.asarray(self.U))
        assert np.array_equal(U2.component("uy", nodes=self.tip), self.U.component("uy", nodes=self.tip))
        with pytest.raises(ValueError, match="no mesh attached"):
            U2.component("uy", nodes="tip")

    def test_field_wrapper_only_wraps_valid_vectors(self):
        s = self.s
        v = np.zeros(s.n_dof)
        assert isinstance(s.field(v), FEField)
        assert isinstance(s.field(np.zeros((s.n_dof, 3))), FEField)
        wrong = np.zeros(s.n_dof - 1)
        assert s.field(wrong) is wrong                                   # wrong length: untouched
        ints = np.zeros(s.n_dof, dtype=int)
        assert s.field(ints) is ints                                     # non-float: untouched
        f = s.field(v)
        assert s.field(f) is f                                           # already wrapped

    def test_modal_shapes_and_harmonic_results_are_fields(self):
        s = self.s
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        f, shapes = s.solve_modal(n_modes=3)
        assert isinstance(shapes, FEField) and shapes.shape == (s.n_dof, 3)
        assert f.shape == (3,) and not isinstance(f, FEField)
        assert shapes.component("uy", nodes="tip").shape == (len(self.tip), 3)
        assert shapes.nodal.shape == (len(self.mesh.nodes), 2, 3)
        Fvec = np.zeros(s.n_dof); Fvec[2 * self.tip + 1] = 1.0
        from fea_engine import RayleighDamping
        s.assemble_damping(RayleighDamping(alpha=1.0, beta=1e-6))
        Uh = s.solve_harmonic(100.0, Fvec)
        assert isinstance(Uh, FEField) and np.iscomplexobj(Uh)
        assert Uh.component("uy", nodes="tip").dtype.kind == "c"

    def test_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        df = self.U.to_dataframe()
        assert list(df.columns) == ["node", "x", "y", "ux", "uy"]
        assert len(df) == len(self.mesh.nodes)
        assert df.loc[self.tip[0], "uy"] == pytest.approx(self.U[2 * self.tip[0] + 1])

    def test_nonlinear_history_can_be_wrapped_with_field(self):
        from fea_engine.mesh import Mesh
        from fea_engine import elements
        from fea_engine.nonlinear_solver import solve_nonlinear_static
        E, nu, A, I, L, n = 210e9, 0.3, 1e-3, 8.33e-7, 1.0, 8
        G = E / (2 * (1 + nu)); x = np.linspace(0, L, n + 1).reshape(-1, 1)
        nb = FESystem(Mesh(nodes=np.hstack([x, 0 * x]),
                           elements=np.array([[i, i + 1] for i in range(n)]), dim=1),
                      elements.Beam2DReissner())
        nb.fix_dofs([0], ["ux", "uy", "rz"])
        matb = (E, G, A, I, 1.0)
        nb.assemble_stiffness(matb)
        nb.add_nodal_force([n], "uy", 0.8 * 3 * E * I / L ** 2)
        lf, Uh = solve_nonlinear_static(nb, matb, n_steps=20)
        last = nb.field(np.asarray(Uh[-1]))
        assert last.component("uy", nodes=[n])[0] == pytest.approx(0.5472, abs=1e-3)   # README E4
