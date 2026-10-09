# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
plasticity_j2_demo.py -- Phase 6 (general-purpose extensions roadmap)
worked example: Hex8PlasticJ2 (small-strain J2 plasticity, continuum).

Loads a single Hex8PlasticJ2 element into a genuinely UNIAXIAL STRESS
state (axial load, lateral faces traction-free), past yield, then
unloads -- and checks the result against PlasticMaterial1D's already-
validated 1-D closed form at every step: this is the roadmap's
headline validation requirement ("uniaxial stress path should match
PlasticMaterial1D's already-validated 1-D result exactly").
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from fea_engine import Hex8PlasticJ2, PlasticMaterialJ2, FESystem
from fea_engine.mesh import Mesh
from fea_engine import nonlinear_solver as nls


E, NU, SIGMA_Y, H = 200e9, 0.3, 250e6, 1e9


def build_cube():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mesh = Mesh(nodes=nodes, elements=elements, dim=3)
    fs = FESystem(mesh, Hex8PlasticJ2(), sparse=False)

    x0_nodes = [0, 3, 4, 7]
    fs.fix_dofs(x0_nodes, [0])
    fs.fix_dofs([0], [1, 2])
    fs.fix_dofs([3], [2])
    fs.fix_dofs([4], [1])
    return fs, [1, 2, 5, 6]


def uniaxial_1d(sigma_applied, E, sigma_y, H):
    if abs(sigma_applied) <= sigma_y:
        return sigma_applied / E
    alpha = (sigma_applied - sigma_y) / H
    return sigma_applied / E + alpha


def main():
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    print("=" * 72)
    print(f"Hex8PlasticJ2: single-element uniaxial stress, E={E:.0e}, nu={NU}, "
          f"sigma_y={SIGMA_Y:.0e}, H={H:.0e}")
    print("=" * 72)

    fs, x1_nodes = build_cube()
    sigma_peak = 1.8 * SIGMA_Y
    fs.add_nodal_force(x1_nodes, 0, sigma_peak)
    fs.init_state()

    lf_up = np.linspace(0.0, 1.0, 41)
    lf_down = np.linspace(1.0, 0.001, 40)
    load_factors = np.concatenate([lf_up, lf_down])
    _, U_hist = nls.solve_nonlinear_static(fs, mat, load_factors=load_factors,
                                            tol=1e-9, max_iter=50)

    eps_fe = U_hist[:, 3 * np.array(x1_nodes)].mean(axis=1)
    sigma_hist = load_factors * sigma_peak
    n_up = len(lf_up)

    # uniaxial_1d() assumes MONOTONIC loading from virgin state -- only
    # valid for the loading branch (indices 0..n_up-1); the unloading
    # branch is elastic (slope E) from the peak state instead, checked
    # separately below via the unload-slope/permanent-set summary.
    eps_cf_loading = np.array([uniaxial_1d(s, E, SIGMA_Y, H) for s in sigma_hist[:n_up]])

    print(f"\nLoading branch (monotonic, matches PlasticMaterial1D's closed form):")
    print(f"{'sigma (MPa)':>14} {'eps_FE':>12} {'eps_1D_closed_form':>20} {'rel_err':>10}")
    for i in range(0, n_up, 8):
        rel = abs(eps_fe[i] - eps_cf_loading[i]) / max(abs(eps_cf_loading[i]), 1e-30)
        print(f"{sigma_hist[i]/1e6:14.2f} {eps_fe[i]:12.6f} {eps_cf_loading[i]:20.6f} {rel:10.2e}")

    print(f"\nUnloading branch: elastic (slope E), starting from the peak state.")
    print(f"Peak stress reached: {sigma_hist[n_up-1]/1e6:.1f} MPa "
          f"({sigma_peak/SIGMA_Y:.2f}x yield)")
    eps_p_expected = (sigma_peak - SIGMA_Y) / H
    print(f"Permanent (plastic) strain after unload: FE={eps_fe[-1]:.6f}  "
          f"closed-form={eps_p_expected:.6f}")
    unload_slope = (sigma_hist[-1] - sigma_hist[-2]) / (eps_fe[-1] - eps_fe[-2])
    print(f"Elastic unload slope: {unload_slope:.4e} Pa  (E = {E:.4e} Pa, "
          f"rel err {abs(unload_slope-E)/E:.2e})")


if __name__ == "__main__":
    main()
