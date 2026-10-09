# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
rom_05_nonlinear_modal_rom.py -- Example Gallery: geometrically nonlinear
reduced-order modeling (nonlinear_rom.py's PolynomialModalROM, the
ICE-ROM/He-et-al-2023 style Nash-form polynomial reduced force model),
trained on a real fea_engine full-order model -- the ROM counterpart
of the fea_engine gallery's Hyperelastic Beam page, here showing
"a small modal basis + a fitted polynomial nonlinearity reproduces the
FULL nonlinear static response," not "large deformation exists."

Uses tests/fea_fixtures.py's clamped_clamped_nonlinear_beam_system()
(the standard clamped-clamped Beam2DCorotational benchmark this
module's reference literature targets -- membrane-stretching-induced
geometric stiffening under central loading, no snap-through) and
AppliedLoadStrategy's already-validated training-data pipeline
(test_nonlinear_rom_fea.py) -- nothing here is a new physics path,
just a fresh, plotted run of it.

Panel 1: mid-span static load-deflection curve, ROM (2-mode
PolynomialModalROM, solved via its own Newton-Raphson predict(F_ext=))
vs. an INDEPENDENT full-order fea_engine solve at every load level
(not the training data -- a held-out load sweep).
Panel 2: relative error of the ROM's mid-span deflection vs. load
level, on a log scale -- the actual accuracy claim, not just "the
curves look close."
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import numpy as np
import matplotlib.pyplot as plt

from rom_engine.nonlinear_rom import PolynomialModalROM, AppliedLoadStrategy
from fea_fixtures import clamped_clamped_nonlinear_beam_system

N_ELEM, N_MODES = 16, 2


def main():
    fx = clamped_clamped_nonlinear_beam_system(n_elem=N_ELEM, n_modes=N_MODES, L=1.0)
    V, M_ff, freq_hz = fx["V"], fx["M_ff"], fx["freq_hz"]
    Lambda = (2 * np.pi * freq_hz) ** 2
    free = fx["free_dofs"]
    mid_node = N_ELEM // 2
    mid_dof_local = list(free).index(mid_node * 3 + 1)   # v-dof at mid-span node

    print(f"Clamped-clamped Beam2DCorotational, {N_ELEM} elements, "
          f"{N_MODES} retained modes, freq = {np.round(freq_hz, 2)} Hz")

    # ---- Train: AppliedLoadStrategy + PolynomialModalROM (already-
    #      validated pipeline, test_nonlinear_rom_fea.py) ----
    rng = np.random.default_rng(0)
    strategy = AppliedLoadStrategy(target_fracs=(0.3, 3.0), reference_scale=0.01,
                                    n_samples=45, rng=rng)
    q_l, q_nl, F_nl = strategy.generate(V, M_ff, freq_hz, fx["mode_shape_peaks"], fx["fom_solver"])
    rom = PolynomialModalROM(N_MODES).fit(q_nl, F_nl)

    # ---- Independent held-out static load sweep: central point load ----
    F_shape_free = np.zeros(len(free))
    F_shape_free[mid_dof_local] = -1.0
    f_hat_shape = V.T @ F_shape_free    # modal projection of the unit load shape

    load_mags = np.linspace(0.0, 900.0, 18)
    mid_rom, mid_fom = [], []
    q_prev = np.zeros(N_MODES)
    for P in load_mags:
        F_free = P * F_shape_free
        u_fom_free = fx["fom_solver"](F_free, n_steps=10)
        mid_fom.append(u_fom_free[mid_dof_local])

        f_hat = P * f_hat_shape
        q_sol, converged = rom.predict(F_ext=f_hat, Lambda=Lambda, q0=q_prev)
        assert converged, f"ROM Newton solve failed to converge at P={P}"
        q_prev = q_sol
        u_rom_free = V @ q_sol
        mid_rom.append(u_rom_free[mid_dof_local])

    mid_rom, mid_fom = np.array(mid_rom), np.array(mid_fom)
    rel_err = np.abs(mid_rom - mid_fom) / np.maximum(np.abs(mid_fom), 1e-12)
    rel_err[0] = np.nan   # zero-load point is 0/0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax1.plot(load_mags, mid_fom * 1000, 'o-', color='steelblue', markersize=5,
              label='full-order fea_engine (Beam2DCorotational)')
    ax1.plot(load_mags, mid_rom * 1000, 's--', color='tomato', markersize=4,
              label=f'{N_MODES}-mode PolynomialModalROM')
    ax1.set_xlabel('central point load (N)')
    ax1.set_ylabel('mid-span deflection (mm)')
    ax1.set_title('Static load-deflection: geometric stiffening\nfrom clamped-clamped membrane stretching')
    ax1.grid(True, alpha=0.35)
    ax1.legend()

    ax2.semilogy(load_mags[1:], rel_err[1:] * 100, 'o-', color='darkorange')
    ax2.set_xlabel('central point load (N)')
    ax2.set_ylabel('relative error in mid-span deflection (%)')
    ax2.set_title('ROM accuracy vs. independent full-order solves\n(held-out load sweep, not training data)')
    ax2.grid(True, alpha=0.35, which='both')

    fig.suptitle('Nonlinear Modal ROM -- PolynomialModalROM on a clamped-clamped beam (rom_engine)', fontsize=13)
    fig.savefig(os.path.join(os.path.dirname(__file__), 'rom_05_nonlinear_modal_rom.png'), dpi=150)
    print("Saved rom_05_nonlinear_modal_rom.png")
    print(f"Max relative error over the sweep: {np.nanmax(rel_err)*100:.3f}%")
    print(f"Full-order solves used: {strategy.n_samples} (training) + {len(load_mags)} (independent sweep)")
    print(f"Reduced model size: {N_MODES} modes vs. {fx['n_dof']} full-order DOF")


if __name__ == "__main__":
    main()
