# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_linear_system_and_export.py -- ReducedSystem (form_linear_system) and the VTK exporter.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import base64
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import fea_engine
from fea_engine import (FESystem, Material, D_plane_stress, Quad4PlaneStress, Tri3PlaneStress,
                        Beam2DEulerBernoulli, Hex8Solid3D, ReducedSystem, export)
from fea_engine.material import Section, EI_beam
from fea_engine.mesh import rectangle_mesh, box_mesh, MultiBlockMesh
from fea_engine.geometry import generate_mesh

MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)


def _cantilever(sparse=False, prescribed=0.0):
    mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=8, ny=4)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02, sparse=sparse)
    s.assemble_stiffness(D_plane_stress(MAT), thickness=0.02)
    mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=0.4, name="tip")
    s.fix_dofs("root", ["ux", "uy"])
    if prescribed:
        s.fix_dofs("tip", ["uy"], value=prescribed)
    else:
        s.add_nodal_force("tip", "uy", -2e4)
    return mesh, s


# ====================================================================== ReducedSystem
class TestReducedSystem:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_solve_matches_solve_static(self, sparse):
        _, s = _cantilever(sparse)
        rs = s.form_linear_system()
        assert isinstance(rs, ReducedSystem) and rs.n_free == len(s.free_dofs)
        assert rs.K.shape == (rs.n_free, rs.n_free) and rs.F.shape == (rs.n_free,)
        assert np.allclose(np.asarray(rs.solve()), np.asarray(s.solve_static()), rtol=1e-9, atol=1e-14)

    def test_external_solver_plus_recover(self):
        _, s = _cantilever()
        rs = s.form_linear_system()
        uf = np.linalg.solve(rs.K, rs.F)
        U = rs.recover(uf)
        assert isinstance(U, fea_engine.FEField) and U.label == "displacement"
        assert np.allclose(np.asarray(U), np.asarray(s.solve_static()))
        assert np.linalg.norm(rs.residual(uf)) < 1e-6 * np.linalg.norm(rs.F)

    def test_prescribed_values_move_to_rhs_and_come_back(self):
        _, s = _cantilever(prescribed=-1e-4)
        rs = s.form_linear_system()
        assert np.any(rs.u_fixed != 0.0) and np.any(rs.F != 0.0)
        U = rs.solve()
        assert np.allclose(np.asarray(U), np.asarray(s.solve_static()), rtol=1e-8, atol=1e-14)
        tip_uy = U.component("uy", nodes="tip")
        assert np.allclose(tip_uy, -1e-4)

    def test_expand_restrict_roundtrip_and_modes(self):
        _, s = _cantilever()
        rs = s.form_linear_system()
        v = np.arange(s.n_dof, dtype=float)
        assert np.array_equal(rs.restrict(v), v[rs.free])
        cols = np.ones((rs.n_free, 3))
        full = rs.expand(cols)
        assert full.shape == (s.n_dof, 3) and not full[rs.fixed].any()
        with pytest.raises(ValueError, match="free-DOF rows"):
            rs.expand(np.ones(rs.n_free + 1))

    def test_reduce_matrix_and_modal_use(self):
        _, s = _cantilever()
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        rs = s.form_linear_system()
        Mff = rs.reduce_matrix(s.M)
        from scipy.linalg import eigh
        w2 = eigh(rs.K, Mff, eigvals_only=True)[:2]
        f_ref, _ = s.solve_modal(n_modes=2)
        assert np.allclose(np.sqrt(w2) / (2 * np.pi), f_ref, rtol=1e-8)

    def test_custom_load_and_matrix_and_errors(self):
        _, s = _cantilever()
        F2 = 2 * s.F
        rs = s.form_linear_system(F=F2)
        assert np.allclose(rs.F, 2 * s.form_linear_system().F)
        rs3 = s.form_linear_system(K=2 * s.K)
        assert np.allclose(rs3.K, 2 * s.form_linear_system().K)
        with pytest.raises(ValueError, match="shape"):
            s.form_linear_system(F=np.zeros(3))
        mesh = rectangle_mesh(Lx=1, Ly=1, nx=2, ny=2)
        with pytest.raises(RuntimeError):
            FESystem(mesh, Quad4PlaneStress(), thickness=1.0).form_linear_system()


# ====================================================================== VTK export
def _parse(path):
    root = ET.parse(path).getroot()
    piece = root.find("UnstructuredGrid/Piece")
    return root, piece


def _arr(piece, section, name, binary=False):
    node = [d for d in piece.find(section).iter("DataArray") if d.get("Name") == name][0]
    ncomp = int(node.get("NumberOfComponents"))
    if node.get("format") == "binary":
        raw = base64.b64decode(node.text)
        n = struct.unpack("<I", raw[:4])[0]
        dt = {"Float64": "<f8", "Int64": "<i8", "UInt8": "u1"}[node.get("type")]
        data = np.frombuffer(raw[4:4 + n], dtype=dt)
    else:
        dt = float if node.get("type") == "Float64" else int
        data = np.array(node.text.split(), dtype=dt)
    return data.reshape(-1, ncomp) if ncomp > 1 else data


class TestWriteVtu:
    @pytest.mark.parametrize("binary", [False, True])
    def test_quad_mesh_roundtrip(self, tmp_path, binary):
        mesh, s = _cantilever()
        U = s.solve_static()
        p = export.write_vtu(tmp_path / "cant", U, binary=binary)
        assert p.endswith("cant.vtu")
        _, piece = _parse(p)
        assert int(piece.get("NumberOfPoints")) == len(mesh.nodes)
        assert int(piece.get("NumberOfCells")) == len(mesh.elements)
        pts = _arr(piece, "Points", "Points")
        assert np.allclose(pts[:, :2], mesh.nodes) and not pts[:, 2].any()
        assert set(_arr(piece, "Cells", "types")) == {9}
        assert np.array_equal(_arr(piece, "Cells", "connectivity"), np.asarray(mesh.elements).ravel())
        u = _arr(piece, "PointData", "displacement")
        assert u.shape == (len(mesh.nodes), 3)
        assert np.allclose(u[:, :2], U.nodal, rtol=1e-8) and not u[:, 2].any()

    def test_beam_has_rotation_scalar_and_line_cells(self, tmp_path):
        beam = FESystem(generate_mesh(dim=1, L=1.0, n=5), Beam2DEulerBernoulli())
        beam.assemble_stiffness(EI_beam(Material(E=210e9, nu=0.3, rho=7800.0), Section(A=0.01, I=8.33e-6)))
        beam.fix_dofs([0], ["uy", "rz"]); beam.add_nodal_force([5], "uy", -1000.0)
        U = beam.solve_static()
        _, piece = _parse(export.write_vtu(tmp_path / "b.vtu", U))
        assert set(_arr(piece, "Cells", "types")) == {3}
        assert np.allclose(_arr(piece, "PointData", "rz"), U.component("rz"), rtol=1e-8)
        assert _arr(piece, "PointData", "displacement").shape[1] == 3

    def test_mode_shapes_one_array_per_mode(self, tmp_path):
        _, s = _cantilever()
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        _, shapes = s.solve_modal(n_modes=3)
        _, piece = _parse(export.write_vtu(tmp_path / "m.vtu", shapes))
        names = {d.get("Name") for d in piece.find("PointData").iter("DataArray")}
        assert {"mode shapes_mode0", "mode shapes_mode1", "mode shapes_mode2"} <= names

    def test_complex_field_gives_re_im_abs(self, tmp_path):
        _, s = _cantilever()
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        s.assemble_damping(fea_engine.RayleighDamping(alpha=1.0, beta=1e-6))
        F = np.zeros(s.n_dof); F[1] = 1.0
        Uh = s.solve_harmonic(50.0, F)
        _, piece = _parse(export.write_vtu(tmp_path / "h.vtu", {"uh": Uh}))
        names = {d.get("Name") for d in piece.find("PointData").iter("DataArray")}
        assert {"uh_re", "uh_im", "uh_abs"} <= names

    def test_3d_hex_and_cell_and_point_data(self, tmp_path):
        mesh = box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
        s = FESystem(mesh, Hex8Solid3D())
        f = s.field(np.zeros(s.n_dof))
        n_el = len(mesh.elements)
        p = export.write_vtu(tmp_path / "h", f, point_data={"T": np.arange(len(mesh.nodes), dtype=float)},
                             cell_data={"id": np.arange(n_el)})
        _, piece = _parse(p)
        assert set(_arr(piece, "Cells", "types")) == {12}
        assert np.array_equal(_arr(piece, "CellData", "id"), np.arange(n_el))
        assert np.array_equal(_arr(piece, "PointData", "T"), np.arange(len(mesh.nodes)))

    def test_multiblock_mixed_cells(self, tmp_path):
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1.0]])
        mb = MultiBlockMesh(nodes, {"tri3": np.array([[0, 1, 2], [0, 2, 3]]),
                                    "quad4": np.array([[1, 4, 5, 2]])}, dim=2)
        s = FESystem(mb, {"tri3": Tri3PlaneStress(), "quad4": Quad4PlaneStress()}, thickness=1.0)
        _, piece = _parse(export.write_vtu(tmp_path / "mb", s.field(np.arange(s.n_dof, dtype=float))))
        assert list(_arr(piece, "Cells", "types")) == [5, 5, 9]
        assert list(_arr(piece, "Cells", "offsets")) == [3, 6, 10]

    def test_high_order_modes(self, tmp_path):
        from fea_engine import Quad8PlaneStress
        from fea_engine.mesh import Mesh
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [.5, 0], [1, .5], [.5, 1], [0, .5]])
        mesh = Mesh(nodes=nodes, elements=np.arange(8).reshape(1, 8), dim=2)
        s = FESystem(mesh, Quad8PlaneStress(), thickness=1.0)
        f = s.field(np.zeros(s.n_dof))
        t_lin = set(_arr(_parse(export.write_vtu(tmp_path / "a", f))[1], "Cells", "types"))
        t_nat = set(_arr(_parse(export.write_vtu(tmp_path / "b", f, high_order="native"))[1], "Cells", "types"))
        assert t_lin == {9} and t_nat == {23}

    def test_errors(self, tmp_path):
        mesh, s = _cantilever()
        U = s.solve_static()
        import pickle
        with pytest.raises(ValueError, match="no mesh available"):
            export.write_vtu(tmp_path / "x", pickle.loads(pickle.dumps(U)))
        with pytest.raises(ValueError, match="high_order"):
            export.write_vtu(tmp_path / "x", U, high_order="cubic")
        with pytest.raises(ValueError, match="point_data"):
            export.write_vtu(tmp_path / "x", U, point_data={"bad": np.zeros(3)})
        with pytest.raises(ValueError, match="FEField"):
            export.write_vtu(tmp_path / "x", {"u": np.zeros(s.n_dof)}, mesh=mesh)

    def test_mesh_only(self, tmp_path):
        mesh, _ = _cantilever()
        _, piece = _parse(export.write_vtu(tmp_path / "mesh", mesh=mesh))
        assert piece.find("PointData") is None


class TestWriteSeries:
    def test_pvd_and_files(self, tmp_path):
        mesh, s = _cantilever()
        U = np.asarray(s.solve_static())
        hist = np.array([k * U for k in (0.0, 0.5, 1.0)])
        ser = s.series(hist, steps=[0.0, 0.5, 1.0], step_name="load factor")
        pvd = export.write_series(tmp_path / "run" / "disp", ser)
        assert pvd.endswith("disp.pvd")
        coll = ET.parse(pvd).getroot().find("Collection")
        sets = coll.findall("DataSet")
        assert [float(d.get("timestep")) for d in sets] == [0.0, 0.5, 1.0]
        assert [d.get("file") for d in sets] == ["disp_0000.vtu", "disp_0001.vtu", "disp_0002.vtu"]
        _, piece = _parse(tmp_path / "run" / "disp_0002.vtu")
        assert np.allclose(_arr(piece, "PointData", "displacement")[:, :2], ser[2].nodal)

    def test_extra_series_and_errors(self, tmp_path):
        _, s = _cantilever()
        U = np.asarray(s.solve_static())
        a = s.series(np.array([U, 2 * U]))
        b = s.series(np.array([3 * U, 4 * U]))
        export.write_series(tmp_path / "two", a, extra={"other": b})
        _, piece = _parse(tmp_path / "two_0001.vtu")
        names = {d.get("Name") for d in piece.find("PointData").iter("DataArray")}
        assert {"displacement", "other"} <= names
        with pytest.raises(TypeError):
            export.write_series(tmp_path / "bad", U)
        with pytest.raises(ValueError, match="steps"):
            export.write_series(tmp_path / "bad", a, extra={"x": b[:1]})


def test_top_level_exports():
    assert fea_engine.export is export and fea_engine.ReducedSystem is ReducedSystem
