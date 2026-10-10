# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_coefficients_and_recovery.py -- spatially varying coefficients and derived results
(stress, strain, von Mises, reactions).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import itertools
import pickle

import numpy as np
import pytest

import fea_engine
from fea_engine import (FESystem, Material, D_plane_stress, D_plane_strain, D_solid3d, Quad4PlaneStress,
                        Quad8PlaneStress, Tri3PlaneStress, Tri6PlaneStress, Hex8Solid3D, Hex20Solid3D,
                        Tet4Solid3D, Tet10Solid3D, Beam2DEulerBernoulli, coefficients as cf, export)
from fea_engine.mesh import Mesh, rectangle_mesh, box_mesh
from fea_engine.geometry import generate_mesh
from fea_engine.material import Section, EI_beam

MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)
MAT0 = Material(E=2.1e11, nu=0.0, rho=7850.0)


# ------------------------------------------------------------------ helpers
def _natural_nodes(f):
    """Natural coordinates of an element's nodes, found from its own shape functions."""
    grid = [-1.0, 0.0, 1.0] if f.quadrature_family == "tensor" else [0.0, 0.5, 1.0]
    out = []
    for a in range(f.n_nodes):
        for p in itertools.product(grid, repeat=f.dim):
            N = f.shape_and_derivs(p)[0]
            if abs(N[a] - 1.0) < 1e-12 and abs(np.abs(N).sum() - 1.0) < 1e-12:
                out.append(p)
                break
        else:
            raise AssertionError(f"node {a} of {type(f).__name__} not found on the grid")
    return np.array(out)


def _distorted_coords(f):
    p = _natural_nodes(f)
    A = np.array([[2.0, 0.3, 0.1], [0.1, 1.5, 0.2], [0.05, 0.2, 1.2]])[:f.dim, :f.dim]
    return p @ A.T + np.array([0.3, -0.2, 0.5])[:f.dim]


ALL_ELEMENTS = [Quad4PlaneStress, Quad8PlaneStress, Tri3PlaneStress, Tri6PlaneStress,
                Hex8Solid3D, Hex20Solid3D, Tet4Solid3D, Tet10Solid3D]


def _D(f):
    return D_plane_stress(MAT) if f.dim == 2 else D_solid3d(MAT)


def _bar(nx=40, ny=2, mat=MAT0, sparse=False, E_of_x=None, at="centroid", L=1.0, H=0.1, t=0.05,
         tri=False, load=1.0e5):
    """Cantilevered tension bar along x, force at the right edge."""
    mesh = rectangle_mesh(Lx=L, Ly=H, nx=nx, ny=ny)
    if tri:
        q = np.asarray(mesh.elements)
        mesh = Mesh(nodes=mesh.nodes, elements=np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]), dim=2)
        f = Tri3PlaneStress()
    else:
        f = Quad4PlaneStress()
    s = FESystem(mesh, f, thickness=t, sparse=sparse)
    if E_of_x is None:
        D = D_plane_stress(mat)
    else:
        D = cf.from_material(lambda x: Material(E=E_of_x(x[0]), nu=0.0, rho=mat.rho), D_plane_stress, at=at)
    s.assemble_stiffness(D, thickness=t)
    mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=L, name="tip")
    s.fix_dofs("root", ["ux"])
    s.fix_dofs(mesh.select_nodes(x=0.0, y=0.0), ["uy"])
    s.add_nodal_force("tip", "ux", load)
    return mesh, s


# ====================================================================== Coefficient objects
class TestCoefficientObjects:
    def test_validation(self):
        with pytest.raises(ValueError, match="exactly one"):
            cf.Coefficient()
        with pytest.raises(ValueError, match="at must be"):
            cf.by_position(lambda x: 1.0, at="node")
        with pytest.raises(ValueError, match="per-element values"):
            cf.Coefficient(values=[1.0], at="gauss")
        with pytest.raises(TypeError, match="callable"):
            cf.Coefficient(fn=3.0)

    def test_values_and_helpers(self):
        c = cf.by_position(lambda x: x[0] + 2 * x[1])
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        assert c.element_value(0, coords) == pytest.approx(3.0) and c.point_value([1.0, 1.0]) == 3.0
        e = cf.by_element([1.0, 5.0])
        assert e.element_value(1, coords) == 5.0
        with pytest.raises(ValueError, match="element index"):
            e.element_value(2, coords)
        assert cf.has_coefficient(c) and cf.has_coefficient({"a": 1, "b": e}) and not cf.has_coefficient(1.0, {"a": 2})
        assert "by_position" in repr(c) and "by_element" in repr(e)


# ====================================================================== assembly equivalence
class TestConstantCoefficientEqualsPlain:
    @pytest.mark.parametrize("cls", ALL_ELEMENTS)
    @pytest.mark.parametrize("at", ["centroid", "gauss"])
    def test_stiffness(self, cls, at):
        f = cls(); coords = _distorted_coords(f); D = _D(f)
        kw = {"thickness": 0.7} if f.dim == 2 else {}      # Tet10 ignores thickness; solids need none
        plain = f.stiffness(coords, D, **kw)
        got = cf.stiffness(f, coords, cf.by_position(lambda x: D, at=at), kw, 0)
        assert np.allclose(got, plain, rtol=1e-9, atol=1e-6 * np.abs(plain).max())

    @pytest.mark.parametrize("cls", [Quad4PlaneStress, Quad8PlaneStress, Tri6PlaneStress, Hex8Solid3D,
                                     Hex20Solid3D, Tri3PlaneStress, Tet4Solid3D, Tet10Solid3D])
    def test_mass_gauss(self, cls):
        f = cls(); coords = _distorted_coords(f); n = f.dofs_per_node
        rho = 7850.0 * np.eye(n)
        kw = {"thickness": 0.7} if f.dim == 2 else {}
        plain = f.mass(coords, rho, **kw)
        el = lambda ff, k, c, r, kwargs: getattr(ff, k)(c, r, **kwargs)
        got = cf.mass(f, coords, cf.by_position(lambda x: rho, at="gauss"), kw, 0, el)
        assert np.allclose(got, plain, rtol=1e-9, atol=1e-9 * np.abs(plain).max())

    def test_system_level_k_and_m_identical_to_plain(self):
        for sparse in (False, True):
            mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=6, ny=3)
            a = FESystem(mesh, Quad4PlaneStress(), thickness=0.02, sparse=sparse)
            b = FESystem(mesh, Quad4PlaneStress(), thickness=0.02, sparse=sparse)
            D = D_plane_stress(MAT)
            a.assemble_stiffness(D, thickness=0.02)
            b.assemble_stiffness(cf.by_position(lambda x: D, at="gauss"),
                                 thickness=cf.by_position(lambda x: 0.02))
            a.assemble_mass(MAT.rho * np.eye(2), thickness=0.02)
            b.assemble_mass(cf.by_position(lambda x: MAT.rho * np.eye(2), at="gauss"), thickness=0.02)
            a.assemble_lumped_mass(MAT.rho * np.eye(2), thickness=0.02)
            b.assemble_lumped_mass(cf.by_position(lambda x: MAT.rho * np.eye(2)), thickness=0.02)
            dense = lambda m: m.toarray() if hasattr(m, "toarray") else m
            assert np.allclose(dense(a.K), dense(b.K), rtol=1e-10, atol=1e-3)
            assert np.allclose(dense(a.M), dense(b.M), rtol=1e-10, atol=1e-6)
            assert np.allclose(dense(a.M_lumped), dense(b.M_lumped), rtol=1e-10, atol=1e-6)

    def test_by_element_matches_per_element_values(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=0.5, nx=4, ny=2)
        n_el = len(mesh.elements)
        a = FESystem(mesh, Quad4PlaneStress(), thickness=0.1)
        b = FESystem(mesh, Quad4PlaneStress(), thickness=0.1)
        a.assemble_stiffness(D_plane_stress(MAT), thickness=0.1)
        b.assemble_stiffness(cf.by_element([D_plane_stress(MAT)] * n_el), thickness=0.1)
        assert np.allclose(a.K, b.K)
        # stiffer second half of the elements really changes the matrix
        vals = [D_plane_stress(MAT) * (1.0 if i < n_el // 2 else 3.0) for i in range(n_el)]
        c = FESystem(mesh, Quad4PlaneStress(), thickness=0.1)
        c.assemble_stiffness(cf.by_element(vals), thickness=0.1)
        assert np.trace(c.K) > np.trace(a.K) * 1.5

    def test_dict_per_block_with_coefficient(self):
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1.0]])
        from fea_engine.mesh import MultiBlockMesh
        mb = MultiBlockMesh(nodes, {"tri3": np.array([[0, 1, 2], [0, 2, 3]]),
                                    "quad4": np.array([[1, 4, 5, 2]])}, dim=2)
        els = {"tri3": Tri3PlaneStress(), "quad4": Quad4PlaneStress()}
        D = D_plane_stress(MAT)
        a = FESystem(mb, els, thickness=1.0); a.assemble_stiffness(D, thickness=1.0)
        b = FESystem(mb, els, thickness=1.0)
        b.assemble_stiffness({"tri3": cf.by_position(lambda x: D), "quad4": D}, thickness=1.0)
        assert np.allclose(a.K, b.K)


# ====================================================================== physics of graded materials
class TestGradedMaterial:
    @staticmethod
    def _tip_error(at, nx):
        E0, L, H, t, P = 2.1e11, 1.0, 0.1, 0.05, 1.0e5
        E = lambda x: E0 * (1.0 + 2.0 * x)
        mesh, s = _bar(nx=nx, ny=1, E_of_x=E, at=at, L=L, H=H, t=t, load=P)
        U = s.solve_static()
        exact = P / (H * t) / (2.0 * E0) * np.log(1.0 + 2.0 * L)          # integral of sigma / E(x)
        return abs(U.component("ux", nodes="tip").mean() - exact) / exact

    def test_graded_bar_converges_and_gauss_beats_centroid(self):
        e_c, e_g = self._tip_error("centroid", 12), self._tip_error("gauss", 12)
        assert e_g < 3e-3 and e_g < e_c
        assert self._tip_error("gauss", 24) < e_g / 2          # refinement helps

    def test_tapered_thickness(self):
        L, H, P = 1.0, 0.1, 1.0e5
        mesh = rectangle_mesh(Lx=L, Ly=H, nx=20, ny=1)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=0.05)
        s.assemble_stiffness(D_plane_stress(MAT0),
                             thickness=cf.by_position(lambda x: 0.05 * (1.0 - 0.5 * x[0]), at="gauss"))
        mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=L, name="tip")
        s.fix_dofs("root", ["ux"]); s.fix_dofs(mesh.select_nodes(x=0.0, y=0.0), ["uy"])
        s.add_nodal_force("tip", "ux", P)
        U = s.solve_static()
        exact = P / (H * 0.05 * 2.1e11) * (-2.0) * np.log(1.0 - 0.5 * L) * 1.0   # int dx / (1 - x/2)
        assert U.component("ux", nodes="tip").mean() == pytest.approx(exact, rel=5e-3)

    def test_graded_density_total_mass(self):
        L, H, t = 1.0, 0.2, 0.1
        mesh = rectangle_mesh(Lx=L, Ly=H, nx=10, ny=2)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=t)
        s.assemble_mass(cf.by_position(lambda x: 1000.0 * (1 + x[0]) * np.eye(2), at="gauss"), thickness=t)
        ones = np.zeros(s.n_dof); ones[0::2] = 1.0
        total = ones @ s.M @ ones
        assert total == pytest.approx(1000.0 * (L + L ** 2 / 2) * H * t, rel=1e-10)

    @pytest.mark.filterwarnings("ignore:.*already assembled")
    def test_errors(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=0.1, nx=2, ny=1)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=0.1)
        D = D_plane_stress(MAT)
        with pytest.raises(ValueError, match="vectorized"):
            s.assemble_stiffness(cf.by_position(lambda x: D), vectorized=True, thickness=0.1)
        with pytest.raises(ValueError, match="method='full'"):
            s.assemble_stiffness(cf.by_position(lambda x: D, at="gauss"), method="reduced", thickness=0.1)
        with pytest.raises(TypeError, match="extra kwargs"):
            s.assemble_stiffness(cf.by_position(lambda x: D, at="gauss"), thickness=0.1, gauss_order=3)
        with pytest.raises(NotImplementedError, match="lumped"):
            s.assemble_lumped_mass(cf.by_position(lambda x: 1.0 * np.eye(2), at="gauss"), thickness=0.1)
        beam = FESystem(generate_mesh(dim=1, L=1.0, n=2), Beam2DEulerBernoulli())
        with pytest.raises(NotImplementedError, match="not available for Beam2DEulerBernoulli"):
            beam.assemble_stiffness(cf.by_position(lambda x: np.eye(2), at="gauss"))
        assert s.K is not None
        # a failed assembly leaves the earlier matrix intact
        s.assemble_stiffness(D, thickness=0.1)
        before = np.array(s.K, copy=True)
        with pytest.raises(ValueError):
            s.assemble_stiffness(cf.by_position(lambda x: D, at="gauss"), method="reduced", thickness=0.1)
        assert np.array_equal(before, s.K)


# ====================================================================== stress / strain / von Mises
class TestRecoveryUniformStates:
    @pytest.mark.parametrize("tri", [False, True])
    def test_uniform_tension_2d(self, tri):
        P, H, t = 1.0e5, 0.1, 0.05
        mesh, s = _bar(nx=8, ny=1, mat=MAT, tri=tri, H=H, t=t, load=P)
        U = s.solve_static()
        S = s.stress(U)
        sig = P / (H * t)
        assert isinstance(S, fea_engine.FEField) and S.label == "stress"
        assert S.dof_names == ("sxx", "syy", "sxy")
        assert np.allclose(S.component("sxx"), sig, rtol=1e-8)
        assert np.abs(S.component("syy")).max() < 1e-6 * sig and np.abs(S.component("sxy")).max() < 1e-6 * sig
        vm = s.von_mises(U)
        assert np.allclose(vm.component("von_mises"), sig, rtol=1e-8)
        assert np.allclose(vm.component("vm"), sig, rtol=1e-8)
        e = s.strain(U)
        assert np.allclose(e.component("exx"), sig / MAT.E, rtol=1e-8)
        assert np.allclose(e.component("eyy"), -MAT.nu * sig / MAT.E, rtol=1e-8)

    def test_default_displacement_is_solved_when_omitted(self):
        _, s = _bar(nx=4, ny=1, mat=MAT)
        assert np.allclose(np.asarray(s.stress()), np.asarray(s.stress(s.solve_static())))

    def test_element_averages_shape_and_values(self):
        P, H, t = 1.0e5, 0.1, 0.05
        mesh, s = _bar(nx=6, ny=1, mat=MAT, H=H, t=t, load=P)
        el = s.stress(at="elements")
        assert el.shape == (len(mesh.elements), 3) and np.allclose(el[:, 0], P / (H * t), rtol=1e-8)
        assert s.von_mises(at="elements").shape == (len(mesh.elements),)
        with pytest.raises(ValueError, match="at must be"):
            s.stress(at="gauss")

    def test_uniform_tension_3d_all_solid_families(self):
        P = 1.0e5
        mesh = box_mesh(1.0, 0.2, 0.2, 4, 1, 1)
        s = FESystem(mesh, Hex8Solid3D())
        s.assemble_stiffness(D_solid3d(MAT))
        mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=1.0, name="tip")
        s.fix_dofs("root", ["ux"])
        s.fix_dofs(mesh.select_nodes(x=0.0, y=0.0), ["uy"])
        s.fix_dofs(mesh.select_nodes(x=0.0, y=0.0, z=0.0), ["uz"])
        s.fix_dofs(mesh.select_nodes(x=0.0, y=0.2, z=0.0), ["uz"])
        s.add_nodal_force("tip", "ux", P)
        S = s.stress()
        assert S.dof_names == ("sxx", "syy", "szz", "sxy", "syz", "sxz")
        assert np.allclose(S.component("sxx"), P / (0.2 * 0.2), rtol=1e-6)
        assert np.allclose(s.von_mises().component("von_mises"), P / 0.04, rtol=1e-6)

    @pytest.mark.parametrize("cls", ALL_ELEMENTS)
    def test_recovery_reproduces_linear_field_on_distorted_element(self, cls):
        """sigma = D eps for an imposed affine displacement must be recovered exactly at the nodes."""
        f = cls(); coords = _distorted_coords(f); dim = f.dim
        mesh = Mesh(nodes=coords, elements=np.arange(f.n_nodes).reshape(1, -1), dim=dim)
        s = FESystem(mesh, f, thickness=1.0) if dim == 2 else FESystem(mesh, f)
        D = _D(f)
        s.assemble_stiffness(D, thickness=1.0) if dim == 2 else s.assemble_stiffness(D)
        G = np.array([[1e-3, 2e-4, 1e-4], [3e-4, -5e-4, 2e-4], [1e-4, 4e-4, 8e-4]])[:dim, :dim]
        U = (coords @ G.T).ravel()
        eps_tensor = 0.5 * (G + G.T)
        if dim == 2:
            eps = np.array([eps_tensor[0, 0], eps_tensor[1, 1], 2 * eps_tensor[0, 1]])
        else:
            eps = np.array([eps_tensor[0, 0], eps_tensor[1, 1], eps_tensor[2, 2],
                            2 * eps_tensor[0, 1], 2 * eps_tensor[1, 2], 2 * eps_tensor[0, 2]])
        S = s.stress(s.field(U))
        assert np.allclose(S.nodal, D @ eps, rtol=1e-8, atol=1e-6)


class TestRecoveryWithCoefficients:
    def test_graded_bar_stress_is_uniform(self):
        P, H, t = 1.0e5, 0.1, 0.05
        mesh, s = _bar(nx=30, ny=1, E_of_x=lambda x: 2.1e11 * (1 + 2 * x), at="gauss", H=H, t=t, load=P)
        S = s.stress()
        interior = S.component("sxx")[(mesh.nodes[:, 0] > 0.1) & (mesh.nodes[:, 0] < 0.9)]
        assert np.allclose(interior, P / (H * t), rtol=5e-3)    # E(x) du/dx = P/A needs D(x) in recovery

    def test_explicit_D_override_and_missing_D(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=0.1, nx=2, ny=1)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=0.1)
        with pytest.raises(RuntimeError, match="constitutive matrix"):
            s.stress(np.zeros(s.n_dof))
        s.assemble_stiffness(D_plane_stress(MAT), thickness=0.1)
        U = np.zeros(s.n_dof); U[0::2] = mesh.nodes[:, 0] * 1e-3
        a = s.stress(U); b = s.stress(U, D=2 * D_plane_stress(MAT))
        assert np.allclose(np.asarray(b), 2 * np.asarray(a))


class TestVonMisesFormulas:
    def test_pure_shear_and_plane_strain(self):
        from fea_engine.recovery import _mises
        assert _mises(np.array([0.0, 0.0, 100.0]), "stress", None) == pytest.approx(100.0 * np.sqrt(3.0))
        # plane strain, equibiaxial: szz = nu*(2s) -> vm = |s*(1-2nu)|
        s, nu = 100.0, 0.3
        assert _mises(np.array([s, s, 0.0]), "strain", nu) == pytest.approx(abs(s * (1 - 2 * nu)))
        assert _mises(np.array([s, s, s, 0.0, 0.0, 0.0]), "stress", None) == pytest.approx(0.0)  # hydrostatic, 6-comp
        assert _mises(np.array([s, 0.0, 0.0, 0.0, 0.0, 0.0]), "stress", None) == pytest.approx(s)

    def test_plane_strain_needs_nu(self):
        _, s = _bar(nx=2, ny=1, mat=MAT)
        with pytest.raises(ValueError, match="needs the Poisson"):
            s.von_mises(plane="strain")
        with pytest.raises(ValueError, match="plane must be"):
            s.von_mises(plane="axisym")
        assert s.von_mises(plane="strain", nu=0.3).label == "von Mises stress"


class TestRecoveryErrorsAndUnits:
    def test_unsupported_element_and_bad_input(self):
        beam = FESystem(generate_mesh(dim=1, L=1.0, n=4), Beam2DEulerBernoulli())
        beam.assemble_stiffness(EI_beam(Material(E=210e9, nu=0.3, rho=7800.0), Section(A=0.01, I=8.33e-6)))
        with pytest.raises(NotImplementedError, match="not available for Beam2DEulerBernoulli"):
            beam.stress(np.zeros(beam.n_dof))
        _, s = _bar(nx=2, ny=1, mat=MAT)
        with pytest.raises(ValueError, match="expected a displacement vector"):
            s.stress(np.zeros(3))
        with pytest.raises(NotImplementedError, match="complex"):
            s.stress(np.zeros(s.n_dof, dtype=complex))

    def test_units_labels_and_plot(self):
        pytest.importorskip("matplotlib").use("Agg")
        import matplotlib.pyplot as plt
        mesh, s = _bar(nx=6, ny=2, mat=MAT)
        s.units = "SI"
        S, vm, R = s.stress(), s.von_mises(), s.reactions()
        assert S.unit_of("sxx") == "Pa" and vm.unit_of("von_mises") == "Pa" and R.unit_of("ux") == "N"
        ax = vm.plot()
        assert "von_mises [Pa]" in [a.get_ylabel() for a in ax.figure.axes]
        S.plot("sxx")
        with pytest.raises(ValueError, match="no translational components"):
            S.plot()
        plt.close("all")
        s.units = "MM_N_TONNE"
        assert s.stress().unit_of("syy") == "MPa"

    def test_pickle_keeps_unit_labels(self):
        _, s = _bar(nx=2, ny=1, mat=MAT)
        s.units = "SI"
        S = pickle.loads(pickle.dumps(s.stress()))
        assert S.unit_of("sxx") == "Pa" and S.dof_names == ("sxx", "syy", "sxy")

    def test_export_of_derived_fields(self, tmp_path):
        import xml.etree.ElementTree as ET
        mesh, s = _bar(nx=4, ny=1, mat=MAT)
        p = export.write_vtu(tmp_path / "d", {"stress": s.stress(), "vm": s.von_mises(),
                                              "R": s.reactions(), "u": s.solve_static()},
                             cell_data={"vm_el": s.von_mises(at="elements")})
        piece = ET.parse(p).getroot().find("UnstructuredGrid/Piece")
        names = {d.get("Name") for d in piece.find("PointData").iter("DataArray")}
        assert {"sxx", "syy", "sxy", "vm", "R", "u"} <= names
        vm = [d for d in piece.find("PointData").iter("DataArray") if d.get("Name") == "vm"][0]
        assert vm.get("NumberOfComponents") == "1"
        assert {d.get("Name") for d in piece.find("CellData").iter("DataArray")} == {"vm_el"}


# ====================================================================== reactions
class TestReactions:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_reactions_balance_applied_load(self, sparse):
        P = 1.0e5
        mesh, s = _bar(nx=6, ny=2, mat=MAT, sparse=sparse, load=P)
        R = s.reactions()
        assert R.label == "reaction" and R.dof_names == ("ux", "uy")
        assert R.component("ux", nodes="root").sum() == pytest.approx(-P, rel=1e-9)
        assert np.abs(R.component("uy")).max() < 1e-6 * P
        free = s.free_dofs
        assert not np.asarray(R)[free].any()                      # zero at free DOFs
        total = np.asarray(R).reshape(-1, 2).sum(axis=0) + np.asarray(s.F).reshape(-1, 2).sum(axis=0)
        assert np.allclose(total, 0.0, atol=1e-6 * P)

    def test_prescribed_displacement_gives_spring_reaction(self):
        L, H, t, delta = 1.0, 0.1, 0.05, 1e-4
        mesh = rectangle_mesh(Lx=L, Ly=H, nx=10, ny=2)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=t)
        s.assemble_stiffness(D_plane_stress(MAT0), thickness=t)
        mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=L, name="tip")
        s.fix_dofs("root", ["ux"]); s.fix_dofs(mesh.select_nodes(x=0.0, y=0.0), ["uy"])
        s.fix_dofs("tip", ["ux"], value=delta)
        R = s.reactions()
        k = MAT0.E * H * t / L
        assert R.component("ux", nodes="tip").sum() == pytest.approx(k * delta, rel=1e-9)
        assert R.component("ux", nodes="root").sum() == pytest.approx(-k * delta, rel=1e-9)

    def test_beam_reaction_moment_unit(self):
        beam = FESystem(generate_mesh(dim=1, L=2.0, n=4), Beam2DEulerBernoulli())
        beam.units = "SI"
        beam.assemble_stiffness(EI_beam(Material(E=210e9, nu=0.3, rho=7800.0), Section(A=0.01, I=8.33e-6)))
        beam.fix_dofs([0], ["uy", "rz"]); beam.add_nodal_force([4], "uy", -1000.0)
        R = beam.reactions()
        assert R.component("uy", nodes=[0])[0] == pytest.approx(1000.0, rel=1e-8)
        assert R.component("rz", nodes=[0])[0] == pytest.approx(2000.0, rel=1e-8)    # P*L moment
        assert R.unit_of("uy") == "N" and R.unit_of("rz") == "N*m"

    def test_requires_assembled_stiffness(self):
        mesh = rectangle_mesh(Lx=1.0, Ly=0.1, nx=2, ny=1)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=0.1)
        with pytest.raises(RuntimeError, match="all zeros"):
            s.reactions(np.zeros(s.n_dof))


def test_top_level_exports():
    assert fea_engine.coefficients is cf and hasattr(fea_engine.recovery, "stress")
