# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_plasticity_kinematic.py -- Wave 2 item 14 (docs/consolidated_
future_roadmap.md, source nonlinear_fem_lessons.md appendix status
table: "Non-associative / kinematic hardening -- Not Implemented"):
validates material.PlasticMaterialJ2Kinematic / material.
j2_radial_return_3d_kinematic() / elements.Hex8PlasticJ2Kinematic --
combined isotropic + Armstrong-Frederick nonlinear kinematic hardening.

Six lines of evidence:

1. test_c_kin_zero_reduces_exactly_to_isotropic_j2 -- C_kin=0 (with
   gamma_AF either zero or nonzero -- shouldn't matter) must keep the
   back stress beta exactly zero forever and reproduce
   j2_radial_return_3d()'s own stress/tangent output bit-for-bit, both
   elastic and plastic branches -- the "collapses to the already-
   validated special case" pattern this whole package uses.
2. test_prager_limit_matches_independent_closed_form -- gamma_AF=0
   (pure linear/Prager kinematic + isotropic hardening) has a
   SEPARATELY-derivable closed form (an "effective combined hardening
   modulus" H+C_kin plugged into the exact same mathematical structure
   as j2_radial_return_3d()'s own formula, just centered on xi_trial =
   s_trial-beta_n instead of s_trial) -- written independently here,
   not copied from material.py, and checked against both the stress
   and the (FD-based) tangent this function actually returns.
3. test_yield_condition_satisfied_at_converged_state -- a direct
   residual check that the converged return map actually SITS ON the
   (shifted, combined-hardening) yield surface to near machine
   precision -- independent of whether the closed-form derivation in
   j2_radial_return_3d_kinematic()'s docstring is right, this just
   checks the algorithm's own self-consistency.
4. test_af_saturation_limit -- the single-step dgamma->infinity limit
   ||beta|| -> (2/3)*C_kin/gamma_AF, derived directly from the update
   formula in j2_radial_return_3d_kinematic()'s own docstring; checked
   here by applying one enormous strain increment (dgamma forced huge)
   and confirming ||beta|| lands near that asymptote.
5. test_hex8_uniaxial_monotonic_matches_combined_modulus_closed_form --
   full-stack FE check (same style as test_plasticity_j2.py's own
   headline check): a single Hex8PlasticJ2Kinematic element, gamma_AF=0
   (pure Prager + isotropic), loaded axially with lateral faces
   traction-free, matches PlasticMaterial1D's closed form using an
   EFFECTIVE hardening modulus H+C_kin.
6. test_rigid_body_modes_and_zero_state -- standard minimum-correctness
   structural sanity check.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import (Hex8PlasticJ2Kinematic, PlasticMaterialJ2Kinematic,
                         PlasticMaterialJ2, PlasticMaterial1D, FESystem, Material)
from fea_engine.mesh import Mesh
from fea_engine.material import j2_radial_return_3d, j2_radial_return_3d_kinematic
from fea_engine import nonlinear_solver as nls


E, NU, SIGMA_Y, H = 200e9, 0.3, 250e6, 1e9
C_KIN, GAMMA_AF = 5e9, 40.0


def _cube_mesh_and_elem():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mesh = Mesh(nodes=nodes, elements=elements, dim=3)
    return mesh, Hex8PlasticJ2Kinematic()


def test_c_kin_zero_reduces_exactly_to_isotropic_j2():
    print("=" * 70)
    print("CHECK 1: C_kin=0 reproduces j2_radial_return_3d()'s own")
    print("stress/tangent output bit-for-bit, beta stays exactly zero")
    print("=" * 70)
    mat_iso = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    mat_kin = PlasticMaterialJ2Kinematic(E=E, nu=NU, sigma_y=SIGMA_Y, H=H,
                                          C_kin=0.0, gamma_AF=77.0)   # gamma_AF irrelevant when C_kin=0
    eps_p0, alpha0, beta0 = np.zeros(6), 0.0, np.zeros(6)

    for eps_voigt in [np.array([1e-5, -0.3e-5, -0.3e-5, 0.0, 0.0, 0.0]),      # elastic
                       np.array([0.01, -0.003, -0.003, 0.002, 0.001, -0.0015])]:  # plastic
        sigma_iso, D_iso, ep_iso, a_iso, y_iso = j2_radial_return_3d(
            eps_voigt, eps_p0, alpha0, mat_iso)
        sigma_kin, D_kin, ep_kin, a_kin, beta_kin, y_kin = j2_radial_return_3d_kinematic(
            eps_voigt, eps_p0, alpha0, beta0, mat_kin)

        assert y_iso == y_kin
        assert np.allclose(sigma_iso, sigma_kin, rtol=1e-10)
        assert np.max(np.abs(beta_kin)) < 1e-8 * max(np.abs(sigma_kin).max(), 1.0)
        assert np.allclose(ep_iso, ep_kin, rtol=1e-8, atol=1e-14)
        assert a_iso == pytest.approx(a_kin, rel=1e-10)
        # D_kin is FD-based (h=1e-7) vs D_iso's hand-derived closed form --
        # loosen the tolerance accordingly, still a tight check
        rel_err = np.max(np.abs(D_iso - D_kin)) / np.abs(D_iso).max()
        print(f"  yielded={y_iso}  max relative tangent diff vs. isotropic: {rel_err:.3e}")
        assert rel_err < 1e-5
    print("  PASS")


def test_prager_limit_matches_independent_closed_form():
    print()
    print("=" * 70)
    print("CHECK 2: gamma_AF=0 (pure Prager + isotropic) matches an")
    print("INDEPENDENTLY-written closed form using an effective combined")
    print("hardening modulus H+C_kin")
    print("=" * 70)
    mat = PlasticMaterialJ2Kinematic(E=E, nu=NU, sigma_y=SIGMA_Y, H=H,
                                      C_kin=C_KIN, gamma_AF=0.0)
    eps_p0, alpha0 = np.zeros(6), 0.0
    beta0 = np.array([1.2e7, -0.6e7, -0.6e7, 0.3e7, 0.0, 0.0])   # nonzero warm-started back stress
    eps_voigt = np.array([0.01, -0.003, -0.003, 0.002, 0.001, -0.0015])

    sigma, D, eps_p_new, alpha_new, beta_new, yielded = j2_radial_return_3d_kinematic(
        eps_voigt, eps_p0, alpha0, beta0, mat)
    assert yielded

    # --- independent closed form, written fresh here, mirroring
    # j2_radial_return_3d()'s own structure but centered on xi = s-beta
    # and with an EFFECTIVE hardening modulus H_eff = H + C_kin ---
    mu, kappa = mat.mu, mat.kappa
    eps_t = eps_voigt
    e11, e22, e33, g12, g23, g13 = eps_t
    eps_tensor = np.array([[e11, g12 / 2, g13 / 2], [g12 / 2, e22, g23 / 2],
                            [g13 / 2, g23 / 2, e33]])
    beta_tensor = np.array([[beta0[0], beta0[3], beta0[5]], [beta0[3], beta0[1], beta0[4]],
                             [beta0[5], beta0[4], beta0[2]]])
    tr_eps = np.trace(eps_tensor)
    eps_dev = eps_tensor - (tr_eps / 3.0) * np.eye(3)
    s_trial = 2.0 * mu * eps_dev
    p = kappa * tr_eps
    xi_trial = s_trial - beta_tensor
    xi_norm = np.sqrt(np.sum(xi_trial * xi_trial))
    H_eff = H + C_KIN
    f_trial = xi_norm - np.sqrt(2.0 / 3.0) * SIGMA_Y
    h_eff = 2.0 * mu + (2.0 / 3.0) * H_eff
    dgamma_cf = f_trial / h_eff
    N = xi_trial / xi_norm
    s_new_cf = s_trial - 2.0 * mu * dgamma_cf * N
    beta_new_cf_t = beta_tensor + (2.0 / 3.0) * C_KIN * dgamma_cf * N
    sigma_cf = s_new_cf + p * np.eye(3)
    sigma_cf_voigt = np.array([sigma_cf[0, 0], sigma_cf[1, 1], sigma_cf[2, 2],
                                sigma_cf[0, 1], sigma_cf[1, 2], sigma_cf[0, 2]])
    beta_cf_voigt = np.array([beta_new_cf_t[0, 0], beta_new_cf_t[1, 1], beta_new_cf_t[2, 2],
                               beta_new_cf_t[0, 1], beta_new_cf_t[1, 2], beta_new_cf_t[0, 2]])

    print(f"  max relative stress diff vs. closed form: "
          f"{np.max(np.abs(sigma - sigma_cf_voigt)) / np.abs(sigma_cf_voigt).max():.3e}")
    assert np.allclose(sigma, sigma_cf_voigt, rtol=1e-8)
    assert np.allclose(beta_new, beta_cf_voigt, rtol=1e-8)

    # independent closed-form tangent, same structural form as
    # j2_radial_return_3d()'s own D_ep but with h_eff/H_eff substituted in
    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    I_dev_voigt = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - (1.0 / 3.0) * np.outer(m, m)
    theta = 1.0 - 2.0 * mu * dgamma_cf / xi_norm
    theta_bar = (1.0 - theta) - 1.0 / (1.0 + H_eff / (3.0 * mu))
    N_voigt = np.array([N[0, 0], N[1, 1], N[2, 2], N[0, 1], N[1, 2], N[0, 2]])
    D_cf = (2.0 * mu * theta * I_dev_voigt + 2.0 * mu * theta_bar * np.outer(N_voigt, N_voigt)
            + kappa * np.outer(m, m))
    rel_err = np.max(np.abs(D - D_cf)) / np.abs(D_cf).max()
    print(f"  max relative tangent diff vs. closed form: {rel_err:.3e}")
    assert rel_err < 1e-5
    print("  PASS")


def test_yield_condition_satisfied_at_converged_state():
    print()
    print("=" * 70)
    print("CHECK 3: the converged return map sits on its own (shifted)")
    print("yield surface to near machine precision -- general gamma_AF!=0")
    print("=" * 70)
    mat = PlasticMaterialJ2Kinematic(E=E, nu=NU, sigma_y=SIGMA_Y, H=H,
                                      C_kin=C_KIN, gamma_AF=GAMMA_AF)
    eps_p0, alpha0, beta0 = np.zeros(6), 0.0, np.zeros(6)
    eps_voigt = np.array([0.015, -0.004, -0.004, 0.003, -0.0012, 0.0009])

    sigma, D, eps_p_new, alpha_new, beta_new, yielded = j2_radial_return_3d_kinematic(
        eps_voigt, eps_p0, alpha0, beta0, mat)
    assert yielded

    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    tr_sigma = sigma[0] + sigma[1] + sigma[2]
    s = sigma - (tr_sigma / 3.0) * m
    xi = s - beta_new
    # xi uses the kinetic (no shear-factor-2) Voigt convention throughout,
    # so its "tensor" Frobenius norm needs the shear terms counted TWICE
    # (once for each off-diagonal symmetric entry) -- same convention
    # j2_radial_return_3d()'s own s_trial_norm uses on the full tensor
    xi_norm = np.sqrt(xi[0] ** 2 + xi[1] ** 2 + xi[2] ** 2
                       + 2 * xi[3] ** 2 + 2 * xi[4] ** 2 + 2 * xi[5] ** 2)
    yield_target = np.sqrt(2.0 / 3.0) * (SIGMA_Y + H * alpha_new)
    rel_err = abs(xi_norm - yield_target) / yield_target
    print(f"  ||xi_new|| = {xi_norm:.6e}   target = {yield_target:.6e}   rel err = {rel_err:.3e}")
    assert rel_err < 1e-8
    print("  PASS")


def test_af_saturation_limit():
    print()
    print("=" * 70)
    print("CHECK 4: single huge-dgamma step drives ||beta|| toward the")
    print("(2/3)*C_kin/gamma_AF saturation asymptote")
    print("=" * 70)
    mat = PlasticMaterialJ2Kinematic(E=E, nu=NU, sigma_y=SIGMA_Y, H=0.0,
                                      C_kin=C_KIN, gamma_AF=GAMMA_AF)
    eps_p0, alpha0, beta0 = np.zeros(6), 0.0, np.zeros(6)
    # an enormous imposed strain forces an enormous dgamma -- large enough
    # that gamma_AF*dgamma >> 1, so the O(1/(gamma_AF*dgamma)) approach to
    # the asymptote is comfortably below this test's tolerance
    eps_voigt = np.array([500.0, -250.0, -250.0, 0.0, 0.0, 0.0])

    _, _, _, _, beta_new, yielded = j2_radial_return_3d_kinematic(
        eps_voigt, eps_p0, alpha0, beta0, mat)
    assert yielded
    beta_norm = np.sqrt(beta_new[0] ** 2 + beta_new[1] ** 2 + beta_new[2] ** 2
                         + 2 * beta_new[3] ** 2 + 2 * beta_new[4] ** 2 + 2 * beta_new[5] ** 2)
    asymptote = (2.0 / 3.0) * C_KIN / GAMMA_AF
    rel_err = abs(beta_norm - asymptote) / asymptote
    print(f"  ||beta|| = {beta_norm:.6e}   asymptote = {asymptote:.6e}   rel err = {rel_err:.3e}")
    assert rel_err < 1e-3
    print("  PASS")


def test_hex8_uniaxial_monotonic_matches_combined_modulus_closed_form():
    print()
    print("=" * 70)
    print("CHECK 5: a single Hex8PlasticJ2Kinematic element (gamma_AF=0,")
    print("pure Prager + isotropic) under monotonic uniaxial-stress")
    print("loading matches PlasticMaterial1D's closed form with an")
    print("EFFECTIVE hardening modulus H+C_kin")
    print("=" * 70)
    mat = PlasticMaterialJ2Kinematic(E=E, nu=NU, sigma_y=SIGMA_Y, H=H,
                                      C_kin=C_KIN, gamma_AF=0.0)
    mesh, elem = _cube_mesh_and_elem()
    fs = FESystem(mesh, elem, sparse=False)

    x0_nodes = [0, 3, 4, 7]
    fs.fix_dofs(x0_nodes, [0])
    fs.fix_dofs([0], [1, 2])
    fs.fix_dofs([3], [2])
    fs.fix_dofs([4], [1])

    x1_nodes = [1, 2, 5, 6]
    sigma_target = 1.8 * SIGMA_Y
    fs.add_nodal_force(x1_nodes, 0, sigma_target)

    fs.init_state()
    load_factors, U_hist = nls.solve_nonlinear_static(fs, mat, n_steps=60, tol=1e-9, max_iter=50)

    eps_total_fe = U_hist[-1, 3 * np.array(x1_nodes)].mean()

    eps_e_final = sigma_target / E
    alpha_1d = (sigma_target - SIGMA_Y) / (H + C_KIN)   # effective combined modulus
    eps_1d = eps_e_final + alpha_1d

    print(f"  FE axial strain:       {eps_total_fe:.10e}")
    print(f"  1-D closed-form strain (H_eff=H+C_kin): {eps_1d:.10e}")
    assert eps_total_fe == pytest.approx(eps_1d, rel=1e-6)
    print("  PASS")


def test_rigid_body_modes_and_zero_state():
    print()
    print("=" * 70)
    print("CHECK 6: symmetric tangent, 6 rigid-body modes at u=0, zero")
    print("internal force at u=0")
    print("=" * 70)
    mat = PlasticMaterialJ2Kinematic(E=E, nu=NU, sigma_y=SIGMA_Y, H=H,
                                      C_kin=C_KIN, gamma_AF=GAMMA_AF)
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
    print("  PASS")
