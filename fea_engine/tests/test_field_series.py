"""
test_field_series.py -- item C: FieldSeries, one container for load paths, time histories, modes.
Additive: solver return values are unchanged, `system.series(...)` wraps them.
"""
__author__ = "Abhijeet"
import pickle

import numpy as np
import pytest

import fea_engine
from fea_engine import (FESystem, FEField, FieldSeries, Material, D_plane_stress, Quad4PlaneStress,
                        Beam2DReissner, RayleighDamping)
from fea_engine.mesh import Mesh, rectangle_mesh
from fea_engine.nonlinear_solver import solve_nonlinear_static
from fea_engine.loads import TimeHistoryLoad, LoadPattern

MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)


def _plane():
    mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=8, ny=4)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
    s.assemble_stiffness(D_plane_stress(MAT), thickness=0.02)
    s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
    mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=0.4, name="tip")
    mesh.add_node_set("tip_top", mesh.select_nodes(x=0.4, y=0.2))
    s.fix_dofs("root", ["ux", "uy"]); s.add_nodal_force("tip", "uy", -2e4)
    return mesh, s


def _beam(n=6):
    E, nu, A, I, L = 210e9, 0.3, 1e-3, 8.33e-7, 1.0
    G = E / (2 * (1 + nu))
    x = np.linspace(0, L, n + 1).reshape(-1, 1)
    mesh = Mesh(nodes=np.hstack([x, 0 * x]), elements=np.array([[i, i + 1] for i in range(n)]), dim=1)
    s = FESystem(mesh, Beam2DReissner())
    mesh.add_node_set("tip", [n])
    s.fix_dofs([0], ["ux", "uy", "rz"])
    mat = (E, G, A, I, 1.0)
    s.assemble_stiffness(mat)
    s.add_nodal_force("tip", "uy", 0.8 * 3 * E * I / L ** 2)
    return s, mat, n


@pytest.fixture(scope="module")
def path():
    s, mat, n = _beam()
    lf, hist = solve_nonlinear_static(s, mat, n_steps=5)
    return s, n, lf, hist, s.series(hist, steps=lf, step_name="load factor")


# ================================================================== structure and access
class TestFieldSeriesBasics:
    def test_wraps_driver_output_without_changing_numbers(self, path):
        s, n, lf, hist, ser = path
        assert isinstance(ser, FieldSeries) and len(ser) == len(lf) == hist.shape[0]
        assert np.array_equal(ser.steps, lf) and ser.step_name == "load factor"
        assert np.array_equal(ser.to_array(), hist)
        assert np.array_equal(ser.to_array("dofs_first"), hist.T)
        with pytest.raises(ValueError, match="layout"):
            ser.to_array("sideways")

    def test_indexing_iteration_slicing(self, path):
        s, n, lf, hist, ser = path
        assert isinstance(ser[-1], FEField) and np.array_equal(np.asarray(ser[2]), hist[2])
        assert ser.first is not None and np.array_equal(np.asarray(ser.last), hist[-1])
        assert [np.array_equal(np.asarray(f), h) for f, h in zip(ser, hist)] == [True] * len(ser)
        sub = ser[1:4]
        assert isinstance(sub, FieldSeries) and len(sub) == 3 and np.array_equal(sub.steps, lf[1:4])
        with pytest.raises(TypeError, match="at\\(step_value\\)"):
            ser[0.5]

    def test_steps_are_views_not_copies(self, path):
        _, _, _, _, ser = path
        assert np.shares_memory(np.asarray(ser[1]), ser.to_array())

    def test_each_step_is_a_named_field(self, path):
        s, n, lf, hist, ser = path
        f = ser[-1]
        assert f.dof_names == ("ux", "uy", "rz") and f.is_nodal
        assert f.component("uy", nodes="tip")[0] == pytest.approx(hist[-1][3 * n + 1])

    def test_at_selects_nearest_step_with_optional_tolerance(self, path):
        _, _, lf, hist, ser = path
        assert np.array_equal(np.asarray(ser.at(0.5)), hist[int(np.argmin(abs(lf - 0.5)))])
        assert ser.index_of(10.0) == len(ser) - 1
        with pytest.raises(ValueError, match="no step within"):
            ser.at(10.0, tol=1e-3)
        assert ser.at(lf[2], tol=1e-12) is not None

    def test_repr_mentions_size_and_step_name(self, path):
        assert "6 steps" in repr(path[4]) and "load factor" in repr(path[4])


# ================================================================== bulk access
class TestBulkAccess:
    def test_component_shapes_and_values(self, path):
        s, n, lf, hist, ser = path
        c = ser.component("uy", nodes="tip")
        assert c.shape == (len(ser), 1)
        assert np.array_equal(c[:, 0], hist[:, 3 * n + 1])
        assert ser.component("y").shape == (len(ser), n + 1)
        assert np.array_equal(ser.component(0, nodes=[0, 1]), hist[:, [0, 3]])

    def test_history_single_node_and_reduce(self):
        mesh, s = _plane()
        hist = np.stack([k * np.asarray(s.solve_static()) for k in (1.0, 2.0, 3.0)])
        ser = s.series(hist, steps=[1, 2, 3], step_name="load factor")
        top = ser.history("uy", int(mesh.node_set("tip_top")[0]))
        assert top.shape == (3,) and top[2] == pytest.approx(3 * top[0])
        with pytest.raises(ValueError, match="pass reduce="):
            ser.history("uy", "tip")
        m = ser.history("uy", "tip", reduce="mean")
        assert np.allclose(m, ser.component("uy", "tip").mean(axis=1))
        assert np.allclose(ser.history("uy", "tip", reduce="absmax"), np.abs(ser.component("uy", "tip")).max(axis=1))
        assert ser.history("uy", "tip", reduce="min").shape == (3,) and ser.history("uy", "tip", reduce="max").shape == (3,)
        with pytest.raises(ValueError, match="reduce must be one of"):
            ser.history("uy", "tip", reduce="median")

    def test_magnitude_excludes_rotations(self, path):
        s, n, lf, hist, ser = path
        mag = ser.magnitude(nodes="tip")
        expect = np.hypot(hist[:, 3 * n], hist[:, 3 * n + 1])
        assert mag.shape == (len(ser), 1) and np.allclose(mag[:, 0], expect)

    def test_peak_is_max_abs_over_steps(self, path):
        s, n, lf, hist, ser = path
        assert np.allclose(ser.peak("uy"), np.abs(hist[:, 1::3]).max(axis=0))
        assert ser.peak("uy", nodes="tip").shape == (1,)

    def test_unknown_names_fail_clearly(self, path):
        _, _, _, _, ser = path
        with pytest.raises(ValueError, match="unknown DOF"):
            ser.component("uz")
        with pytest.raises(ValueError, match="unknown node set"):
            ser.component("uy", nodes="nowhere")

    def test_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        mesh, s = _plane()
        u = np.asarray(s.solve_static())
        ser = s.series(np.stack([u, 2 * u]), steps=[0.5, 1.0], step_name="load factor")
        df = ser.to_dataframe("uy", "tip")
        assert df.index.name == "load factor" and list(df.index) == [0.5, 1.0]
        assert df.shape == (2, len(mesh.node_set("tip"))) and df.columns[0].startswith("node")
        assert df.shape[1] == len(mesh.node_set("tip"))
        assert ser.to_dataframe("ux").shape == (2, len(mesh.nodes))


# ================================================================== construction and validation
class TestConstruction:
    def test_list_of_vectors_and_int_history_are_accepted(self):
        mesh, s = _plane()
        u = np.asarray(s.solve_static())
        ser = s.series([u, 2 * u])
        assert len(ser) == 2 and np.array_equal(ser.steps, [0.0, 1.0])
        z = s.series(np.zeros((3, s.n_dof), dtype=int))
        assert z.to_array().dtype.kind == "f"

    def test_bad_shapes_are_explained(self):
        _, s = _plane()
        with pytest.raises(ValueError, match="must be 2-D"):
            s.series(np.zeros(s.n_dof))
        with pytest.raises(ValueError, match="axis=1"):
            s.series(np.zeros((s.n_dof, 3)))                       # columns without axis=1
        with pytest.raises(ValueError, match="steps has shape"):
            s.series(np.zeros((3, s.n_dof)), steps=[0, 1])
        with pytest.raises(ValueError, match="axis must be"):
            s.series(np.zeros((3, s.n_dof)), axis=2)

    def test_series_direct_constructor_needs_a_template(self):
        with pytest.raises(ValueError, match="template"):
            FieldSeries(np.zeros((2, 4)), np.zeros(4))

    def test_unit_inheritance_and_time_unit(self):
        mesh, s = _plane()
        s.units = "SI"
        u = np.asarray(s.solve_static())
        ser = s.series(np.stack([u, u]), steps=[0.0, 0.1], step_name="time")
        assert ser.step_unit == "s" and ser.units == "m" and ser[0].units == "m"
        assert s.series(np.stack([u, u]), step_name="time", step_unit="ms").step_unit == "ms"

    def test_pickle_roundtrip_keeps_values_and_steps(self, path):
        _, _, lf, hist, ser = path
        ser2 = pickle.loads(pickle.dumps(ser))
        assert np.array_equal(ser2.to_array(), hist) and np.array_equal(ser2.steps, lf)


# ================================================================== the three solver families
class TestSolverFamilies:
    def test_modal_series(self):
        mesh, s = _plane()
        f, shapes = s.solve_modal(n_modes=3)
        ms = s.modal_series(n_modes=3)
        assert len(ms) == 3 and np.allclose(ms.steps, f) and ms.step_name == "frequency" and ms.step_unit == "Hz"
        assert np.allclose(np.asarray(ms[1]), np.asarray(shapes)[:, 1])
        assert ms.label == "mode shapes" and ms.units is None
        assert ms.at(f[2] + 1e-6)[0] == np.asarray(shapes)[0, 2] or np.allclose(np.asarray(ms.at(f[2])), np.asarray(shapes)[:, 2])
        assert ms.component("uy", nodes="tip").shape == (3, len(mesh.node_set("tip")))

    def test_modal_series_from_solve_modal_columns_via_axis(self):
        mesh, s = _plane()
        f, shapes = s.solve_modal(n_modes=2)
        ser = s.series(shapes, steps=f, axis=1)
        assert np.allclose(ser.to_array("dofs_first"), np.asarray(shapes))

    def test_transient_history(self):
        mesh, s = _plane()
        s.assemble_damping(RayleighDamping(alpha=1.0, beta=1e-6))
        load = TimeHistoryLoad(LoadPattern(mesh.node_set("tip"), 1), lambda t: -1e3)
        t, hist = s.solve_transient_implicit(load, T_total=0.005, dt=0.001)
        assert hist.shape == (len(t), s.n_dof)
        ser = s.series(hist, steps=t, step_name="time")
        assert len(ser) == len(t) and ser.step_name == "time"
        assert ser.last.component("uy", nodes="tip").shape[0] == len(mesh.node_set("tip"))

    def test_nonlinear_path_matches_load_displacement_values(self, path):
        s, n, lf, hist, ser = path
        curve = ser.history("uy", "tip")
        assert curve[0] == 0.0 and 0.54 < curve[-1] < 0.55        # README E4: ~0.547 at 80% of the linear load
        assert np.all(np.diff(curve) > 0)


# ================================================================== plotting
class TestPlotHistory:
    def test_single_curve_with_labels_and_units(self, path):
        matplotlib = pytest.importorskip("matplotlib"); matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s, n, lf, hist, ser = path
        s.units = "SI"
        ser2 = s.series(hist, steps=lf, step_name="load factor")
        ax = ser2.plot_history("uy", "tip")
        assert ax.get_xlabel() == "load factor" and ax.get_ylabel() == "uy [m]"
        assert np.allclose(ax.lines[0].get_ydata(), ser2.history("uy", "tip"))
        assert ax.get_title() == "displacement"
        plt.close("all"); s.units = None

    def test_reduce_and_time_axis_unit(self):
        matplotlib = pytest.importorskip("matplotlib"); matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mesh, s = _plane()
        u = np.asarray(s.solve_static())
        s.units = "SI"
        ser = s.series(np.stack([u, 2 * u, 3 * u]), steps=[0, 1, 2], step_name="time")
        ax = ser.plot_history("uy", "tip", reduce="mean")
        assert len(ax.lines) == 1 and ax.get_xlabel() == "time [s]"
        ax2 = ser.plot_history("uy", "tip")
        assert len(ax2.lines) == len(mesh.node_set("tip"))
        plt.close("all")


def test_exports():
    assert fea_engine.FieldSeries is FieldSeries
