# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
e1_rom_speedup.py -- POD + Galerkin + affine ROM of a two-region cantilever (claim E1).

Reuses ``rom_engine/examples/two_region_beam_rom.py`` (same model, same training and test draws), so the numbers are
those of the shipped example, and adds repeated timing and a figure.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from common import ROOT, best_of, load_module, plt, save_fig, save_result

N_ELEM, N_TRAIN, N_MODES, TIP_LOAD = 120, 10, 10, -1000.0
N_TEST, N_SWEEP = 8, 500


def main():
    ex = load_module(ROOT / "rom_engine" / "examples" / "two_region_beam_rom.py", "two_region_beam_rom")
    model = ex.build_two_region_beam(n=N_ELEM)
    free, n_dof = model["free_dofs"], model["n_dof"]
    basis, galerkin, affine, tip_local = ex.build_rom(model, n_train=N_TRAIN, n_modes=N_MODES, tip_load=TIP_LOAD, seed=0)

    F_free = np.zeros(len(free)); F_free[tip_local] = TIP_LOAD
    F_full = np.zeros(n_dof); F_full[free] = F_free
    F_r = galerkin.project_vector(F_full)

    def rom_tip(EI1, EI2):
        q = np.linalg.solve(affine.assemble_reduced((EI1, EI2)), F_r)
        return (basis.V @ q)[free][tip_local]

    def fom_tip(EI1, EI2):
        return np.linalg.solve(model["K_direct"](EI1, EI2)[np.ix_(free, free)], F_free)[tip_local]

    rng = np.random.default_rng(99)
    test = [tuple(rng.uniform(0.5, 5.0, size=2)) for _ in range(N_TEST)]
    errs = [abs(rom_tip(*p) - fom_tip(*p)) / abs(fom_tip(*p)) for p in test]

    mus = [tuple(rng.uniform(0.5, 5.0, size=2)) for _ in range(N_SWEEP)]
    t_rom, _ = best_of(lambda: [np.linalg.solve(affine.assemble_reduced(m), F_r) for m in mus], repeats=5)
    t_fom, _ = best_of(lambda: [np.linalg.solve(model["K_direct"](*m)[np.ix_(free, free)], F_free) for m in mus],
                       repeats=3)

    data = {"n_elements": N_ELEM, "n_dof_full": int(n_dof), "n_dof_reduced": int(basis.n_modes), "n_train": N_TRAIN,
            "pod_energy_captured": float(basis.energy_captured()), "n_test": N_TEST,
            "max_rel_tip_error": float(max(errs)), "mean_rel_tip_error": float(np.mean(errs)),
            "n_sweep": N_SWEEP, "t_fom_s": t_fom, "t_rom_s": t_rom, "speedup": t_fom / t_rom,
            "timing_rule": "minimum over repeats of the whole sweep"}
    save_result("e1_rom_speedup", data)

    P = plt()
    ratios = np.linspace(0.1, 10.0, 20)
    fig, ax = P.subplots(figsize=(4.2, 3.0))
    for EI2, c in zip((0.5, 1.0, 2.0), ("C0", "C1", "C2")):
        ax.plot(ratios, [fom_tip(r * EI2, EI2) for r in ratios], "o", ms=3, color=c, label=f"FOM, $EI_2={EI2:g}$")
        ax.plot(ratios, [rom_tip(r * EI2, EI2) for r in ratios], "-", color=c, label=f"ROM, $EI_2={EI2:g}$")
    ax.set_xlabel("$EI_1/EI_2$"); ax.set_ylabel("tip deflection (m)"); ax.legend(fontsize=7)
    save_fig(fig, "e1_rom_speedup")
    print(f"max rel. error {data['max_rel_tip_error']:.2e}, speed-up {data['speedup']:.1f}x")


if __name__ == "__main__":
    main()
