"""
rom_02_galerkin_parametric_rom.py -- Example Gallery: affine parametric
Galerkin ROM (pod.py + galerkin.py + affine.py) on a two-region
cantilever beam whose two independent bending rigidities (EI1, EI2)
are the reduction PARAMETERS, not just a single fixed model like
example 01 -- the "many-query" use case reduced-basis methods exist
for: build the reduced model ONCE offline, then answer a whole design
sweep or UQ study online for a fraction of a full re-assembly's cost.

Reuses this repository's own existing, previously-validated
two_region_beam_rom.py end-to-end script (build_two_region_beam(),
build_rom()) rather than re-deriving the affine-decomposition setup,
and adds: a POD singular-value-decay panel, and a saved (not just
plt.show()'d) tip-deflection-vs-rigidity-ratio comparison panel with
the FULL-ORDER (fea_engine) curve plotted UNDERNEATH the ROM curve so
any visible mismatch would be obvious, plus a printed online/offline
timing speedup exactly like the source script reports.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from two_region_beam_rom import build_two_region_beam, build_rom  # noqa: E402

N_ELEM = 120
N_TRAIN = 10
N_MODES = 10
TIP_LOAD = -1000.0


def main():
    model = build_two_region_beam(n=N_ELEM)
    free = model["free_dofs"]

    basis, galerkin, affine, tip_dof_local = build_rom(
        model, n_train=N_TRAIN, n_modes=N_MODES, tip_load=TIP_LOAD, seed=0)
    print(f"POD basis: {basis.n_modes} modes (from {N_TRAIN} training snapshots), "
          f"energy captured = {basis.energy_captured():.6f}")

    F_free = np.zeros(len(free))
    F_free[tip_dof_local] = TIP_LOAD
    F_full = np.zeros(model["n_dof"])
    F_full[free] = F_free
    F_r = galerkin.project_vector(F_full)

    # Held-out accuracy check (same pattern as the source script).
    rng = np.random.default_rng(99)
    n_test = 12
    rel_errs = []
    for _ in range(n_test):
        EI1, EI2 = rng.uniform(0.5, 5.0, size=2)
        K_r = affine.assemble_reduced((EI1, EI2))
        q = np.linalg.solve(K_r, F_r)
        x_rom_free = (basis.V @ q)[free]
        K_fom_ff = model["K_direct"](EI1, EI2)[np.ix_(free, free)]
        x_fom_free = np.linalg.solve(K_fom_ff, F_free)
        rel_errs.append(abs(x_rom_free[tip_dof_local] - x_fom_free[tip_dof_local])
                         / abs(x_fom_free[tip_dof_local]))
    print(f"Max relative tip-deflection error over {n_test} held-out (EI1,EI2) points: "
          f"{max(rel_errs):.3e}")

    import time
    n_sweep = 500
    mus = [tuple(rng.uniform(0.5, 5.0, size=2)) for _ in range(n_sweep)]
    t0 = time.perf_counter()
    for EI1, EI2 in mus:
        K_r = affine.assemble_reduced((EI1, EI2))
        np.linalg.solve(K_r, F_r)
    t_rom = time.perf_counter() - t0
    t0 = time.perf_counter()
    for EI1, EI2 in mus:
        K_fom_ff = model["K_direct"](EI1, EI2)[np.ix_(free, free)]
        np.linalg.solve(K_fom_ff, F_free)
    t_fom = time.perf_counter() - t0
    speedup = t_fom / t_rom
    print(f"{n_sweep}-point (EI1,EI2) sweep: full-order {t_fom*1e3:.2f} ms, "
          f"reduced-order {t_rom*1e3:.2f} ms -> {speedup:.1f}x speedup")

    # Sweep for the plot: tip deflection vs rigidity ratio, several EI2.
    rigidity_ratios = np.linspace(0.1, 10.0, 25)
    fixed_EI2s = (0.5, 1.0, 2.0)
    curves = {}
    for EI2 in fixed_EI2s:
        tip_roms, tip_foms = [], []
        for ratio in rigidity_ratios:
            EI1 = ratio * EI2
            K_r = affine.assemble_reduced((EI1, EI2))
            q = np.linalg.solve(K_r, F_r)
            tip_roms.append((basis.V @ q)[free][tip_dof_local])
            K_fom_ff = model["K_direct"](EI1, EI2)[np.ix_(free, free)]
            x_fom_free = np.linalg.solve(K_fom_ff, F_free)
            tip_foms.append(x_fom_free[tip_dof_local])
        curves[EI2] = (np.array(tip_foms), np.array(tip_roms))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ranks = np.arange(1, len(basis.singular_values) + 1)
    ax1.semilogy(ranks, basis.singular_values, 'o-', color='steelblue', markersize=5)
    ax1.axvline(basis.n_modes, color='tomato', linestyle='--', linewidth=1.2,
                label=f'retained: {basis.n_modes} modes')
    ax1.set_xlabel('mode index')
    ax1.set_ylabel('singular value')
    ax1.set_title(f'POD singular value decay\n({N_TRAIN} training (EI1,EI2) draws, two-region beam)')
    ax1.grid(True, alpha=0.35, which='both')
    ax1.legend()

    colors = ['steelblue', 'seagreen', 'tomato']
    for (EI2, (fom, rom)), c in zip(curves.items(), colors):
        ax2.plot(rigidity_ratios, fom, 'o', color=c, markersize=4,
                  label=f'FEA (full-order), EI2={EI2:g}')
        ax2.plot(rigidity_ratios, rom, '-', color=c, linewidth=1.6, alpha=0.85,
                  label=f'Galerkin ROM, EI2={EI2:g}')
    ax2.set_xlabel(r'$EI_1 / EI_2$')
    ax2.set_ylabel('tip deflection (m)')
    ax2.set_title(f'Tip deflection vs. rigidity ratio\n'
                  f'max held-out error {max(rel_errs):.2e}, {speedup:.0f}x online speedup')
    ax2.grid(True, alpha=0.35)
    ax2.legend(fontsize=8, ncol=1)

    fig.suptitle('Affine Parametric Galerkin ROM -- two-region cantilever beam (rom_engine)',
                 fontsize=13)
    out = os.path.join(os.path.dirname(__file__), 'rom_02_galerkin_parametric_rom.png')
    fig.savefig(out, dpi=150)
    print("Saved rom_02_galerkin_parametric_rom.png")


if __name__ == "__main__":
    main()
