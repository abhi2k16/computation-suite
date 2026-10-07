"""
rom_03_greedy_frequency_training.py -- Example Gallery: adaptive
(greedy) vs. uniform-grid training for a frequency-domain ROM
(FrequencyROM + greedy_train_frequency_basis, greedy.py) on a damped
cantilever beam.

Both strategies spend the SAME number of expensive full-order
solves (the training budget); the only difference is WHICH
frequencies each one chooses to solve at. The greedy loop uses a
cheap error indicator -- never a second full-order solve -- to find
where the current basis is weakest and spends the next solve there,
so it should preferentially land near the model's actual resonances
without ever being told where they are.

Reuses this repository's own existing, previously-validated
greedy_frequency_training.py end-to-end script (build_damped_cantilever(),
the greedy-vs-uniform training + held-out comparison logic) rather
than re-deriving the training/comparison pipeline, and adds: a
full-order-vs-both-ROMs frequency response sweep panel (so the greedy
basis's resonance-seeking behavior is visible directly, not just in a
bar chart), plus a saved PNG instead of a plt.show() bar chart.
"""
import os
import sys
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from greedy_frequency_training import build_damped_cantilever  # noqa: E402

from rom_engine import (  # noqa: E402
    FrequencyROM,
    build_pod_basis_from_frf_snapshots,
    greedy_train_frequency_basis,
)

import matplotlib.pyplot as plt


def main():
    sysobj, alpha, beta = build_damped_cantilever(n=20)
    free = sysobj.free_dofs
    n_dof = sysobj.n_dof
    Kff = sysobj.K[np.ix_(free, free)]
    Mff = sysobj.M[np.ix_(free, free)]
    Cff = sysobj.C[np.ix_(free, free)]

    eigvals, _ = eigh(Kff, Mff)
    omegas_n = np.sqrt(np.clip(eigvals, 0, None))
    print(f"model: {n_dof} dof ({len(free)} free), first 3 resonances (rad/s): {omegas_n[:3]}")

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    training_band = np.linspace(0.1, 20.0, 400) * omegas_n[0]

    greedy_basis, history = greedy_train_frequency_basis(
        training_band, Mff, Kff, F_free, rayleigh=(alpha, beta),
        n_seed=3, tol=1e-9, max_modes=5)
    final_rank = greedy_basis.n_modes
    greedy_omegas = np.array([om for om, _, _ in history])
    print(f"Greedy training: final basis rank = {final_rank}, "
          f"selections (omega/omega1) = {np.round(greedy_omegas / omegas_n[0], 3)}")

    uniform_omegas = np.linspace(training_band[0], training_band[-1], final_rank)
    uniform_basis = build_pod_basis_from_frf_snapshots(
        uniform_omegas, Mff, Kff, F_free, C=Cff, n_modes=final_rank)

    rom_greedy = FrequencyROM.from_MCK(Mff, Kff, greedy_basis.V, rayleigh=(alpha, beta))
    rom_uniform = FrequencyROM.from_MCK(Mff, Kff, uniform_basis.V, rayleigh=(alpha, beta))

    rng = np.random.default_rng(0)
    test_omegas = np.concatenate([
        omegas_n[i] * (1.0 + 0.05 * rng.standard_normal(20)) for i in range(3)
    ])
    test_omegas = test_omegas[test_omegas > 0]

    def median_max_rel_error(rom):
        errs = []
        for om in test_omegas:
            U_true = sysobj.solve_harmonic(om, F_full)[free]
            x_rom = rom.frequency_response([om], F_free)[0]
            errs.append(np.linalg.norm(x_rom - U_true) / max(np.linalg.norm(U_true), 1e-30))
        return float(np.median(errs)), float(np.max(errs))

    med_g, max_g = median_max_rel_error(rom_greedy)
    med_u, max_u = median_max_rel_error(rom_uniform)
    print(f"Held-out test set ({len(test_omegas)} pts near resonances): "
          f"greedy median={med_g:.3e}/max={max_g:.3e}, "
          f"uniform median={med_u:.3e}/max={max_u:.3e}")

    # Full FRF sweep, all three (FOM, greedy ROM, uniform ROM), tip dof.
    tip_dof_local = len(free) - 2
    sweep_omegas = np.linspace(0.1, 15.0, 250) * omegas_n[0]
    H_fom = np.array([sysobj.solve_harmonic(om, F_full)[free][tip_dof_local] for om in sweep_omegas])
    H_greedy = rom_greedy.frequency_response(sweep_omegas, F_free)[:, tip_dof_local]
    H_uniform = rom_uniform.frequency_response(sweep_omegas, F_free)[:, tip_dof_local]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax1.semilogy(sweep_omegas / omegas_n[0], np.abs(H_fom), '-', color='0.3', linewidth=2.0,
                  label='full-order (fea_engine solve_harmonic)', zorder=2)
    ax1.semilogy(sweep_omegas / omegas_n[0], np.abs(H_greedy), '--', color='seagreen',
                  linewidth=1.6, label=f'greedy-trained ROM (rank {final_rank})', zorder=3)
    ax1.semilogy(sweep_omegas / omegas_n[0], np.abs(H_uniform), ':', color='tomato',
                  linewidth=1.8, label=f'uniform-grid ROM (rank {final_rank})', zorder=3)
    for i, om in enumerate(greedy_omegas / omegas_n[0]):
        ax1.axvline(om, color='seagreen', alpha=0.15, linewidth=3,
                    label='greedy-selected training points' if i == 0 else None)
    ax1.set_xlabel(r'$\omega / \omega_1$')
    ax1.set_ylabel('|tip displacement / tip force|')
    ax1.set_title('Frequency response: FOM vs. both ROMs\n'
                  '(shaded columns = frequencies greedy chose to train on)')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=8)

    x = np.arange(2)
    width = 0.35
    ax2.bar(x - width / 2, [med_g, med_u], width, color=['seagreen', 'tomato'], alpha=0.55,
            label='median relative error')
    ax2.bar(x + width / 2, [max_g, max_u], width, color=['seagreen', 'tomato'], alpha=0.95,
            label='maximum relative error')
    ax2.set_yscale('log')
    ax2.set_xticks(x, ['greedy', 'uniform'])
    ax2.set_ylabel('relative error')
    ax2.set_title(f'Held-out test set ({len(test_omegas)} pts scattered around\n'
                  f'the first 3 resonances) -- same training budget, rank {final_rank}')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle('Greedy vs. Uniform Frequency-Domain ROM Training -- damped cantilever (rom_engine)',
                 fontsize=13)
    out = os.path.join(os.path.dirname(__file__), 'rom_03_greedy_frequency_training.png')
    fig.savefig(out, dpi=150)
    print("Saved rom_03_greedy_frequency_training.png")


if __name__ == "__main__":
    main()
