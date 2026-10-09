# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_plasticity_plane_stress.py -- Wave 2 item 13 (docs/consolidated_
future_roadmap.md, source general_purpose_extensions_roadmap.md):
validates material.j2_radial_return_plane_stress() and
elements.Quad4PlasticJ2PlaneStress.

Mirrors tests/test_plasticity_j2.py's five-line-of-evidence structure
(same roadmap, same validation standard), one dimension down:

1. test_elastic_branch_matches_D_plane_stress -- below yield, the
   eps_33-Newton-condensed return map must degenerate EXACTLY to
   ordinary linear plane-stress elasticity (both stress AND tangent).
2. test_consistent_tangent_matches_finite_difference -- the Schur-
   complement-condensed tangent D_ps checked against a finite
   difference of THIS function's own (eps33-Newton-condensed) stress
   output, in the plastic regime -- validates the condensation
   formula's correctness for real, not just on faith (see the
   function's own docstring).
3. test_quad4_uniaxial_stress_matches_plastic_material_1d -- the
   headline check, exactly mirroring test_hex8_uniaxial_stress_
   matches_plastic_material_1d one dimension down: a single
   Quad4PlasticJ2PlaneStress element, loaded axially with its lateral
   edges left traction-free, must match PlasticMaterial1D's already-
   validated 1-D closed form EXACTLY. Since plane stress by
   construction already enforces sigma_33=0, and traction-free lateral
   edges enforce sigma_22=sigma_12=0 too, this is a genuine uniaxial
   STRESS state -- precisely what PlasticMaterial1D models.
4. test_monotonic_then_unload_shows_correct_permanent_set -- ramp past
   yield then unload: elastic unloading slope must equal E exactly,
   and the permanent strain must equal the closed-form (sigma_peak-
   sigma_y)/H.
5. test_rigid_body_modes_and_zero_state -- symmetric tangent, 3 rigid-
   body modes (2 translations + 1 rotation) at u=0, zero internal
   force at u=0.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import (Quad4PlasticJ2PlaneStress, PlasticMaterialJ2, PlasticMaterial1D,
                         FESystem, D_plane_stress, Material)
from fea_engine.mesh import Mesh
from fea_engine.material import j2_radial_return_plane_stress
from fea_engine import nonlinear_solver as nls


E, NU, SIGMA_Y, H = 200e9, 0.3, 250e6, 1e9


def _unit_square_mesh_and_elem():
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3]])
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    return mesh, Quad4PlasticJ2PlaneStress()


def test_elastic_branch_matches_D_plane_stress():
    print("=" * 70)
    print("CHECK 1: below yield, j2_radial_return_plane_stress() degenerates")
    print("EXACTLY to ordinary linear plane-stress elasticity")
    print("=" * 70)
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    eps_small = np.array([1e-5, -0.3e-5, 0.4e-5])   # [e11, e22, g12]
    eps_p0, alpha0, eps33_0 = np.zeros(6), 0.0, 0.0

    sigma, D, eps_p_new, alpha_new, eps33_new, yielded = j2_radial_return_plane_stress(
        eps_small, eps_p0, alpha0, eps33_0, mat)
    assert not yielded

    D_el = D_plane_stress(Material(E=E, nu=NU))
    print(f"  max |D - D_elastic|/max|D_elastic| = "
          f"{np.max(np.abs(D - D_el)) / np.max(np.abs(D_el)):.3e}")
    assert np.allclose(D, D_el, rtol=1e-8, atol=1e-3)
    assert np.allclose(sigma, D_el @ eps_small, rtol=1e-8)
    # elastic branch must also leave eps_33 at its (linear-elastic) value,
    # not stuck at the eps33_n=0 seed
    eps33_expected = -NU / (1.0 - NU) * (eps_small[0] + eps_small[1])
    assert eps33_new == pytest.approx(eps33_expected, rel=1e-6)
    print("  PASS")


def test_consistent_tangent_matches_finite_difference():
    print()
    print("=" * 70)
    print("CHECK 2: the Schur-complement-condensed tangent D_ps matches a")
    print("finite difference of this SAME function's own stress output,")
    print("in the plastic regime")
    print("=" * 70)
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    eps_big = np.array([0.01, -0.002, 0.003])   # well into the plastic regime
    eps_p0, alpha0, eps33_0 = np.zeros(6), 0.0, 0.0

    sigma0, D, _, _, eps33_1, yielded = j2_radial_return_plane_stress(
        eps_big, eps_p0, alpha0, eps33_0, mat)
    assert yielded

    h = 1e-8
    D_fd = np.zeros((3, 3))
    for j in range(3):
        dE = np.zeros(3)
        dE[j] = h
        sp, *_ = j2_radial_return_plane_stress(eps_big + dE, eps_p0, alpha0, eps33_0, mat)
        sm, *_ = j2_radial_return_plane_stress(eps_big - dE, eps_p0, alpha0, eps33_0, mat)
        D_fd[:, j] = (sp - sm) / (2 * h)

    rel_err = np.abs(D - D_fd) / np.abs(D).max()
    print(f"  max relative tangent error vs. FD: {np.max(rel_err):.3e}")
    assert np.max(rel_err) < 1e-6
    print("  PASS")


def test_quad4_uniaxial_stress_matches_plastic_material_1d():
    print()
    print("=" * 70)
    print("CHECK 3: a single Quad4PlasticJ2PlaneStress element, loaded")
    print("axially with lateral edges traction-free, matches")
    print("PlasticMaterial1D's closed-form uniaxial response EXACTLY")
    print("=" * 70)
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    mesh, elem = _unit_square_mesh_and_elem()
    fs = FESystem(mesh, elem, sparse=False)

    x0_nodes = [0, 3]   # x=0 edge
    fs.fix_dofs(x0_nodes, [0])     # ux=0 -- kills x-translation AND rotation
    fs.fix_dofs([0], [1])          # uy=0 at one node only -- kills y-translation,
                                    # leaves the y=0/y=1 edges traction-free

    x1_nodes = [1, 2]   # x=1 edge -- axial load, lateral (y) edges untouched
    sigma_target = 1.8 * SIGMA_Y
    fs.add_nodal_force(x1_nodes, 0, sigma_target)

    fs.init_state()
    load_factors, U_hist = nls.solve_nonlinear_static(fs, mat, n_steps=60, tol=1e-9, max_iter=50)

    eps_total_fe = U_hist[-1, 2 * np.array(x1_nodes)].mean()   # L0 = 1

    eps_e_final = sigma_target / E
    alpha = (sigma_target - SIGMA_Y) / H
    eps_1d = eps_e_final + alpha

    print(f"  FE axial strain:       {eps_total_fe:.10e}")
    print(f"  1-D closed-form strain: {eps_1d:.10e}")
    assert eps_total_fe == pytest.approx(eps_1d, rel=1e-8)
    print("  PASS")


def test_monotonic_then_unload_shows_correct_permanent_set():
    print()
    print("=" * 70)
    print("CHECK 4: monotonic loading past yield then unloading shows the")
    print("correct elastic unload slope (=E) and permanent plastic set")
    print("=" * 70)
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    mesh, elem = _unit_square_mesh_and_elem()
    fs = FESystem(mesh, elem, sparse=False)

    x0_nodes = [0, 3]
    fs.fix_dofs(x0_nodes, [0])
    fs.fix_dofs([0], [1])

    x1_nodes = [1, 2]
    sigma_peak = 1.8 * SIGMA_Y
    fs.add_nodal_force(x1_nodes, 0, sigma_peak)

    fs.init_state()
    lf_up = np.linspace(0.0, 1.0, 41)
    lf_down = np.linspace(1.0, 0.001, 40)   # stop just short of exactly zero,
    # same reason as test_plasticity_j2.py's own version of this test:
    # load_factors[i]=0 exactly hits solve_nonlinear_static's absolute-
    # tolerance floor at this problem's force scale, not a plasticity bug
    load_factors = np.concatenate([lf_up, lf_down])
    _, U_hist = nls.solve_nonlinear_static(fs, mat, load_factors=load_factors,
                                            tol=1e-9, max_iter=50)

    eps_hist = U_hist[:, 2 * np.array(x1_nodes)].mean(axis=1)
    sigma_hist = load_factors * sigma_peak

    eps_p_expected = (sigma_peak - SIGMA_Y) / H
    print(f"  permanent set: FE={eps_hist[-1]:.6e}  expected={eps_p_expected:.6e}")
    assert eps_hist[-1] == pytest.approx(eps_p_expected, rel=1e-3)

    unload_slope = (sigma_hist[-1] - sigma_hist[-2]) / (eps_hist[-1] - eps_hist[-2])
    print(f"  unload slope: FE={unload_slope:.6e}  E={E:.6e}")
    assert unload_slope == pytest.approx(E, rel=1e-10)
    print("  PASS")


def test_rigid_body_modes_and_zero_state():
    print()
    print("=" * 70)
    print("CHECK 5: symmetric tangent, 3 rigid-body modes at u=0 (2")
    print("translations + 1 rotation, plane-stress DOF count), zero")
    print("internal force at u=0")
    print("=" * 70)
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    mesh, elem = _unit_square_mesh_and_elem()
    coords = mesh.nodes
    u0 = np.zeros(8)
    state = elem.init_state()

    ke = elem.tangent_stiffness(coords, u0, mat, state=state)
    assert np.allclose(ke, ke.T)

    eig = np.linalg.eigvalsh(ke)
    n_zero = np.sum(np.abs(eig) < 1e-6 * np.abs(eig).max())
    assert n_zero == 3

    f0 = elem.internal_force(coords, u0, mat, state=state)
    assert np.abs(f0).max() < 1e-8
    print("  PASS")
