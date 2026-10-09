# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_units_and_plot.py -- v1.0.1 P3: unit labels (no conversion) and FEField.plot.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import pickle
import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fea_engine  # noqa: E402
from fea_engine import (FESystem, Material, D_plane_stress, Quad4PlaneStress, Beam2DEulerBernoulli,  # noqa: E402
                        Beam2DReissner, Tri3PlaneStress, Hex8Solid3D, units)
from fea_engine.material import Section, EI_beam  # noqa: E402
from fea_engine.mesh import rectangle_mesh, MultiBlockMesh, box_mesh  # noqa: E402
from fea_engine.geometry import generate_mesh  # noqa: E402

MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _cantilever(unit=None):
    mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=12, ny=6)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
    s.assemble_stiffness(D_plane_stress(MAT), thickness=0.02)
    mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=0.4, name="tip")
    s.fix_dofs("root", ["ux", "uy"]); s.add_nodal_force("tip", "uy", -2e4)
    if unit is not None:
        s.units = unit
    return mesh, s


# ====================================================================== UnitSystem
class TestUnitSystem:
    def test_presets_and_derived_labels(self):
        assert units.SI.stress == "Pa" and units.SI.density == "kg/m^3" and units.SI.frequency == "Hz"
        assert units.MM_N_TONNE.stress == "MPa" and units.MM_N_TONNE.length == "mm"
        custom = units.UnitSystem("in", "lbf", "slinch", "s")
        assert custom.stress == "lbf/in^2"
        assert units.SI.unit_of("rotation") == "rad" and units.SI.unit_of("displacement") == "m"
        with pytest.raises(ValueError, match="unknown quantity"):
            units.SI.unit_of("temperature")

    def test_resolve(self):
        assert units.resolve(None) is None and units.resolve(units.SI) is units.SI
        assert units.resolve("si") is units.SI and units.resolve("MM_N_TONNE") is units.MM_N_TONNE
        with pytest.raises(ValueError, match="units must be"):
            units.resolve("furlongs")

    def test_frozen_and_comparable(self):
        with pytest.raises(Exception):
            units.SI.length = "mm"
        assert units.UnitSystem("m", "N", "kg", "s") == units.SI

    def test_check_consistent(self):
        a = Material(E=1.0, nu=0.3); b = Material(E=1.0, nu=0.3)
        a.units, b.units = units.SI, units.MM_N_TONNE
        with pytest.raises(ValueError, match="unit systems differ"):
            units.check_consistent(a, b)
        b.units = units.SI
        units.check_consistent(a, b)
        units.check_consistent(a, Material(E=1.0, nu=0.3))      # unlabelled objects are skipped


class TestUnitsOnObjects:
    def test_material_units_is_optional_and_does_not_affect_equality(self):
        assert Material(E=1.0, nu=0.3).units is None
        assert Material(E=1.0, nu=0.3, units=units.SI) == Material(E=1.0, nu=0.3)
        assert "units" not in repr(Material(E=1.0, nu=0.3, units=units.SI))
        assert Material(2.1e11, 0.3, 7850.0).rho == 7850.0          # positional construction intact

    def test_system_units_default_setter_and_chaining(self):
        _, s = _cantilever()
        assert s.units is None
        assert s.set_units("SI") is s and s.units is units.SI
        s.units = units.MM_N_TONNE
        assert s.units.length == "mm"
        s.units = None
        assert s.units is None
        with pytest.raises(ValueError):
            s.units = "bogus"

    def test_unlabelled_results_have_no_units(self):
        _, s = _cantilever()
        U = s.solve_static()
        assert U.units is None and U.unit_of("uy") is None

    def test_labelled_results_carry_length_unit_through_ops_and_pickle(self):
        _, s = _cantilever("SI")
        U = s.solve_static()
        assert U.units == "m" and U.unit_of("uy") == "m"
        assert (U * 2).units == "m" and U[::2].units == "m" and U.copy().units == "m"
        assert pickle.loads(pickle.dumps(U)).units == "m"
        _, s2 = _cantilever(units.MM_N_TONNE)
        assert s2.solve_static().units == "mm"

    def test_rotation_dofs_are_radians(self):
        beam = FESystem(generate_mesh(dim=1, L=1.0, n=4), Beam2DEulerBernoulli())
        beam.units = "SI"
        f = beam.field(np.ones(beam.n_dof))
        f.label = "displacement"
        beam.assemble_stiffness(EI_beam(Material(E=210e9, nu=0.3, rho=7800.0), Section(A=0.01, I=8.33e-6)))
        f = beam.field(np.ones(beam.n_dof), label="displacement")
        assert f.unit_of("uy") == "m" and f.unit_of("rz") == "rad"

    def test_modal_shapes_have_no_unit_and_harmonic_does(self):
        _, s = _cantilever("SI")
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        _, shapes = s.solve_modal(n_modes=2)
        assert shapes.units is None
        s.assemble_damping(fea_engine.RayleighDamping(alpha=1.0, beta=1e-6))
        F = np.zeros(s.n_dof); F[1] = 1.0
        assert s.solve_harmonic(50.0, F).units == "m"

    def test_numbers_are_unchanged_by_labels(self):
        _, a = _cantilever()
        _, b = _cantilever("SI")
        assert np.array_equal(np.asarray(a.solve_static()), np.asarray(b.solve_static()))


# ====================================================================== FEField.plot
class TestPlot:
    def test_2d_contour_with_unit_in_colorbar_and_title(self):
        mesh, s = _cantilever("SI")
        U = s.solve_static()
        ax = U.plot("uy", title="tip load")
        assert ax.get_title() == "tip load" and ax.get_xlabel() == "x"
        labels = [a.get_ylabel() for a in ax.figure.axes]
        assert "uy [m]" in labels                               # colour bar carries name and unit
        assert ax.get_aspect() == 1.0

    def test_default_component_is_magnitude_and_title_is_label(self):
        _, s = _cantilever()
        ax = s.solve_static().plot()
        assert ax.get_title() == "displacement"
        assert "|u|" in [a.get_ylabel() for a in ax.figure.axes]

    def test_deform_moves_the_mesh_and_options_work(self):
        mesh, s = _cantilever()
        U = s.solve_static()
        fig, ax = plt.subplots()
        out = U.plot("uy", ax=ax, deform=True, scale=1e3, show_mesh=True, colorbar=False, cmap="plasma")
        assert out is ax and len(ax.figure.axes) == 1           # no colour bar axes added
        ymin = ax.get_ylim()[0]
        assert ymin < -0.0                                      # tip deflects below y=0 (undeformed min is 0)

    def test_deform_needs_ux_uy(self):
        from fea_engine import Quad4MindlinPlate
        from fea_engine.material import D_mindlin_plate
        plate = FESystem(rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2), Quad4MindlinPlate())
        with pytest.raises(ValueError, match="needs 'ux' and 'uy'"):
            plate.field(np.zeros(plate.n_dof)).plot(deform=True)

    def test_mode_shape_plot_picks_a_column(self):
        _, s = _cantilever()
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        _, shapes = s.solve_modal(n_modes=3)
        ax = shapes.plot("uy", mode=2)
        assert ax.get_title() == "mode shapes"
        with pytest.raises(ValueError, match="mode 5 out of range"):
            shapes.plot("uy", mode=5)

    def test_complex_field_is_drawn_as_absolute_value(self):
        _, s = _cantilever()
        s.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
        s.assemble_damping(fea_engine.RayleighDamping(alpha=1.0, beta=1e-6))
        F = np.zeros(s.n_dof); F[1] = 1.0
        Uh = s.solve_harmonic(50.0, F)
        ax = Uh.plot("uy")
        assert "|uy|" in [a.get_ylabel() for a in ax.figure.axes]

    def test_triangle_and_multiblock_meshes(self):
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1.0]])
        mb = MultiBlockMesh(nodes, {"tri3": np.array([[0, 1, 2], [0, 2, 3]]),
                                    "quad4": np.array([[1, 4, 5, 2]])}, dim=2)
        s = FESystem(mb, {"tri3": Tri3PlaneStress(), "quad4": Quad4PlaneStress()}, thickness=1.0)
        f = s.field(np.arange(s.n_dof, dtype=float))
        tris = f._triangles()
        assert tris.shape == (4, 3)                              # 2 triangles + quad split in two
        f.plot("ux")

    def test_1d_beam_line_plot(self):
        m = Material(E=210e9, nu=0.3, rho=7800.0)
        beam = FESystem(generate_mesh(dim=1, L=1.0, n=10), Beam2DEulerBernoulli())
        beam.units = "SI"
        beam.assemble_stiffness(EI_beam(m, Section(A=0.01, I=8.33e-6)))
        beam.fix_dofs([0], ["uy", "rz"]); beam.add_nodal_force([10], "uy", -1000.0)
        U = beam.solve_static()
        ax = U.plot("uy")
        line = ax.lines[0]
        assert len(line.get_xdata()) == 11 and ax.get_ylabel() == "uy [m]"
        assert np.allclose(line.get_ydata(), U.component("uy"))
        with pytest.raises(ValueError, match="only available for 2-D"):
            U.plot("uy", deform=True)
        U.plot("rz")
        assert ax.figure is not None

    def test_errors(self):
        mesh, s = _cantilever()
        U = s.solve_static()
        with pytest.raises(ValueError, match="unknown DOF"):
            U.plot("uz")
        with pytest.raises(ValueError, match="probably sliced"):
            U[s.free_dofs].plot()
        U2 = pickle.loads(pickle.dumps(U))
        with pytest.raises(ValueError, match="no mesh attached"):
            U2.plot()
        box = box_mesh(1.0, 1.0, 1.0, 2, 2, 2) if callable(box_mesh) else None
        s3 = FESystem(box, Hex8Solid3D())
        with pytest.raises(NotImplementedError, match="3-D"):
            s3.field(np.zeros(s3.n_dof)).plot()
