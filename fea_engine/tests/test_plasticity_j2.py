"""
test_plasticity_j2.py -- validation for Module 19 (general-purpose
extensions roadmap Phase 6): material.PlasticMaterialJ2 /
material.j2_radial_return_3d() / elements.Hex8PlasticJ2.

Five lines of evidence, following the roadmap's own validation plan
(Section 4) plus the closed-form-tangent cross-check every hand-derived
tangent in this project gets:

1. test_elastic_branch_matches_D_solid3d -- below yield, the return
   map must degenerate EXACTLY to ordinary linear elasticity (both the
   stress AND the tangent) -- a trivial but necessary check.
2. test_consistent_tangent_matches_finite_difference -- the hand-
   derived closed-form algorithmic tangent (re-derived from scratch by
   directly differentiating THIS discrete return-map algorithm, not
   copied from a possibly-misremembered textbook formula -- see
   j2_radial_return_3d()'s docstring) checked against a finite
   difference of the SAME function's own stress output, in the
   plastic regime.
3. test_hex8_uniaxial_stress_matches_plastic_material_1d -- the
   roadmap's headline check: a single Hex8PlasticJ2 element, loaded
   and constrained to reproduce a genuinely UNIAXIAL STRESS state
   (axial load, lateral faces traction-free), must match
   PlasticMaterial1D's already-validated 1-D closed form EXACTLY --
   the "collapse to an existing validated special case" pattern used
   throughout this roadmap, one dimension up.
4. test_monotonic_then_unload_shows_correct_permanent_set -- ramp the
   SAME element past yield, then unload: the elastic unloading slope
   must equal E exactly (no reverse yielding), and the strain
   remaining at (near-)zero load must equal the closed-form permanent
   plastic strain (sigma_peak-sigma_y)/H.
5. test_rigid_body_modes_and_zero_state -- basic structural sanity
   (symmetric tangent, 6 rigid-body modes at u=0, zero internal force
   at u=0) -- the same minimum-correctness check every other element
   in this package gets.
"""
import numpy as np
import pytest

from fea_engine import Hex8PlasticJ2, PlasticMaterialJ2, FESystem, D_solid3d, Material
from fea_engine.mesh import Mesh
from fea_engine.material import j2_radial_return_3d
from fea_engine import nonlinear_solver as nls


E, NU, SIGMA_Y, H = 200e9, 0.3, 250e6, 1e9


def _cube_mesh_and_elem():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mesh = Mesh(nodes=nodes, elements=elements, dim=3)
    return mesh, Hex8PlasticJ2()


def test_elastic_branch_matches_D_solid3d():
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    eps_small = np.array([1e-5, -0.3e-5, -0.3e-5, 0.0, 0.0, 0.0])
    eps_p0, alpha0 = np.zeros(6), 0.0

    sigma, D, eps_p_new, alpha_new, yielded = j2_radial_return_3d(eps_small, eps_p0, alpha0, mat)
    assert not yielded

    D_el = D_solid3d(Material(E=E, nu=NU))
    assert np.allclose(D, D_el, rtol=1e-12)
    assert np.allclose(sigma, D_el @ eps_small, rtol=1e-10)


def test_consistent_tangent_matches_finite_difference():
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    eps_big = np.array([0.01, -0.003, -0.003, 0.002, 0.001, -0.0015])
    eps_p0, alpha0 = np.zeros(6), 0.0

    sigma, D, _, _, yielded = j2_radial_return_3d(eps_big, eps_p0, alpha0, mat)
    assert yielded

    h = 1e-8
    D_fd = np.zeros((6, 6))
    for j in range(6):
        dE = np.zeros(6)
        dE[j] = h
        sp, *_ = j2_radial_return_3d(eps_big + dE, eps_p0, alpha0, mat)
        sm, *_ = j2_radial_return_3d(eps_big - dE, eps_p0, alpha0, mat)
        D_fd[:, j] = (sp - sm) / (2 * h)

    rel_err = np.abs(D - D_fd) / np.abs(D).max()
    assert np.max(rel_err) < 1e-8


def test_hex8_uniaxial_stress_matches_plastic_material_1d():
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    mesh, elem = _cube_mesh_and_elem()
    fs = FESystem(mesh, elem, sparse=False)

    x0_nodes = [0, 3, 4, 7]
    fs.fix_dofs(x0_nodes, [0])
    fs.fix_dofs([0], [1, 2])   # kill rigid translation in y, z
    fs.fix_dofs([3], [2])      # kill rotation about x
    fs.fix_dofs([4], [1])      # kill remaining rotation

    x1_nodes = [1, 2, 5, 6]
    sigma_target = 1.8 * SIGMA_Y   # well into the plastic regime
    fs.add_nodal_force(x1_nodes, 0, sigma_target)

    fs.init_state()
    load_factors, U_hist = nls.solve_nonlinear_static(fs, mat, n_steps=60, tol=1e-9, max_iter=50)

    eps_total_fe = U_hist[-1, 3 * np.array(x1_nodes)].mean()   # L0 = 1

    # closed-form 1-D uniaxial response (elastic-then-linear-hardening)
    eps_e_final = sigma_target / E
    alpha = (sigma_target - SIGMA_Y) / H
    eps_1d = eps_e_final + alpha

    assert eps_total_fe == pytest.approx(eps_1d, rel=1e-10)


def test_monotonic_then_unload_shows_correct_permanent_set():
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    mesh, elem = _cube_mesh_and_elem()
    fs = FESystem(mesh, elem, sparse=False)

    x0_nodes = [0, 3, 4, 7]
    fs.fix_dofs(x0_nodes, [0])
    fs.fix_dofs([0], [1, 2])
    fs.fix_dofs([3], [2])
    fs.fix_dofs([4], [1])

    x1_nodes = [1, 2, 5, 6]
    sigma_peak = 1.8 * SIGMA_Y
    fs.add_nodal_force(x1_nodes, 0, sigma_peak)

    fs.init_state()
    lf_up = np.linspace(0.0, 1.0, 41)
    lf_down = np.linspace(1.0, 0.001, 40)   # stop just short of exactly zero
    # (load_factors[i]=0 exactly hits solve_nonlinear_static's absolute-
    # tolerance floor at this problem's ~1e8 N force scale -- see
    # nonlinear_solver.py's ref=max(norm(F_ext),1e-30) fallback; not a
    # plasticity bug, just a mismatch between an absolute tol=1e-9 N and
    # a force scale of ~1e8 N when the reference load is exactly zero)
    load_factors = np.concatenate([lf_up, lf_down])
    _, U_hist = nls.solve_nonlinear_static(fs, mat, load_factors=load_factors,
                                            tol=1e-9, max_iter=50)

    eps_hist = U_hist[:, 3 * np.array(x1_nodes)].mean(axis=1)
    sigma_hist = load_factors * sigma_peak

    eps_p_expected = (sigma_peak - SIGMA_Y) / H
    assert eps_hist[-1] == pytest.approx(eps_p_expected, rel=1e-3)

    unload_slope = (sigma_hist[-1] - sigma_hist[-2]) / (eps_hist[-1] - eps_hist[-2])
    assert unload_slope == pytest.approx(E, rel=1e-10)


def test_rigid_body_modes_and_zero_state():
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    mesh, elem = _cube_mesh_and_elem()
    coords = mesh.nodes
    u0 = np.zeros(24)
    state = elem.init_state()

    ke = elem.tangent_stiffness(coords, u0, mat, state=state)
    assert np.allclose(ke, ke.T)

    eig = np.linalg.eigvalsh(ke)
    n_zero = np.sum(np.abs(eig) < 1e-6 * np.abs(eig).max())
    assert n_zero == 6

    f0 = elem.internal_force(coords, u0, mat, state=state)
    assert np.abs(f0).max() < 1e-8
