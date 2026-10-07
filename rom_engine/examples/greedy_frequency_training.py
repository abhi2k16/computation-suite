"""
greedy_frequency_training.py -- end-to-end rom_engine.greedy example.

Compares two ways of training a FrequencyROM's POD-on-FRF-snapshots
basis under the SAME full-order-solve budget (same final rank):

  1. A UNIFORM grid of training frequencies across the band of interest
     -- the obvious, simple default (used throughout
     frequency_sweep_cantilever.py and rom_engine's other examples).
  2. greedy_train_frequency_basis() -- adaptively picks WHICH
     frequencies to solve at, using a cheap error indicator (never a
     full-order solve) to find where the current basis is weakest,
     spending the (expensive) full-order-solve budget there instead.

The uniform grid doesn't know in advance where the model's resonances
are; the greedy loop finds them without being told, purely from how
much a candidate frequency's prediction changes as the training set
grows (docs/phase4_error_bounds_greedy_roadmap.md Section 4). This
script shows both: which frequencies each strategy actually trains on,
and how each performs on a held-out test set concentrated near the
model's real resonances.
"""
import numpy as np
from scipy.linalg import eigh

from fea_engine.material import EI_beam, Material, Section
from fea_engine.geometry import generate_mesh
from fea_engine.solver import FESystem
from fea_engine.damping import RayleighDamping
from fea_engine import elements

from rom_engine import (
    FrequencyROM,
    build_pod_basis_from_frf_snapshots,
    greedy_train_frequency_basis,
)


def build_damped_cantilever(n=20, L=1.0, E=210e9, rho=7800.0, A=0.01, I=8.33e-6,
                             alpha=2.0, beta=1e-5):
    mesh = generate_mesh(dim=1, L=L, n=n)
    elem = elements.Beam2DEulerBernoulli()
    sysobj = FESystem(mesh, elem)
    mat = Material(E=E, nu=0.3, rho=rho)
    sec = Section(A=A, I=I)
    EI = EI_beam(mat, sec)
    sysobj.assemble_stiffness(EI)
    sysobj.assemble_mass(rho * A)
    sysobj.assemble_damping(RayleighDamping(alpha=alpha, beta=beta))
    sysobj.fix_dofs([0], [0, 1])
    return sysobj, alpha, beta


def main():
    print("=" * 70)
    print("rom_engine.greedy example: adaptive vs. uniform frequency training")
    print("=" * 70)

    sysobj, alpha, beta = build_damped_cantilever(n=20)
    free = sysobj.free_dofs
    n_dof = sysobj.n_dof
    Kff = sysobj.K[np.ix_(free, free)]
    Mff = sysobj.M[np.ix_(free, free)]
    Cff = sysobj.C[np.ix_(free, free)]

    eigvals, _ = eigh(Kff, Mff)
    omegas_n = np.sqrt(np.clip(eigvals, 0, None))
    print(f"model: {n_dof} dof ({len(free)} free), first 3 resonances "
          f"(rad/s): {omegas_n[:3]}")

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    training_band = np.linspace(0.1, 20.0, 400) * omegas_n[0]

    # ---- 1. Greedy training ------------------------------------------
    greedy_basis, history = greedy_train_frequency_basis(
        training_band, Mff, Kff, F_free, rayleigh=(alpha, beta),
        n_seed=3, tol=1e-9, max_modes=5)
    final_rank = greedy_basis.n_modes
    print(f"\nGreedy training: final basis rank = {final_rank}")
    print("Selections (in order), relative to the 1st resonance:")
    for om, val, method in history:
        print(f"  omega/omega1 = {om / omegas_n[0]:6.3f}   "
              f"indicator = {val:.3e}   ({method})")

    # ---- 2. Uniform-grid training, SAME rank/budget -------------------
    uniform_omegas = np.linspace(training_band[0], training_band[-1], final_rank)
    uniform_basis = build_pod_basis_from_frf_snapshots(
        uniform_omegas, Mff, Kff, F_free, C=Cff, n_modes=final_rank)
    print(f"\nUniform-grid training (same rank {final_rank}), "
          f"relative to the 1st resonance:")
    print("  " + ", ".join(f"{om / omegas_n[0]:.3f}" for om in uniform_omegas))

    rom_greedy = FrequencyROM.from_MCK(Mff, Kff, greedy_basis.V, rayleigh=(alpha, beta))
    rom_uniform = FrequencyROM.from_MCK(Mff, Kff, uniform_basis.V, rayleigh=(alpha, beta))

    # ---- 3. Compare on a held-out test set near the resonances --------
    rng = np.random.default_rng(0)
    test_omegas = np.concatenate([
        omegas_n[i] * (1.0 + 0.05 * rng.standard_normal(20)) for i in range(3)
    ])
    test_omegas = test_omegas[test_omegas > 0]

    def median_rel_error(rom):
        errs = []
        for om in test_omegas:
            U_true = sysobj.solve_harmonic(om, F_full)[free]
            x_rom = rom.frequency_response([om], F_free)[0]
            errs.append(np.linalg.norm(x_rom - U_true) / max(np.linalg.norm(U_true), 1e-30))
        return float(np.median(errs)), float(np.max(errs))

    med_g, max_g = median_rel_error(rom_greedy)
    med_u, max_u = median_rel_error(rom_uniform)

    print(f"\nHeld-out test set ({len(test_omegas)} points, scattered around "
          f"the first 3 resonances):")
    print(f"  greedy-trained basis:  median rel err = {med_g:.3e}, max rel err = {max_g:.3e}")
    print(f"  uniform-grid basis:    median rel err = {med_u:.3e}, max rel err = {max_u:.3e}")

    print("\n" + "=" * 70)
    print("Done. Both strategies spent the SAME number of expensive full-order")
    print("solves -- greedy just chose WHICH frequencies to spend them on.")
    print("=" * 70)

    # plot the held-out test set results
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        x = np.arange(2)
        width = 0.35
        plt.bar(x - width / 2, [med_g, med_u], width, label="median relative error")
        plt.bar(x + width / 2, [max_g, max_u], width, label="maximum relative error")
        plt.yscale("log")
        plt.xticks(x, ["greedy", "uniform"])
        plt.ylabel("relative error")
        plt.title("Held-out Test Set Results")
        plt.legend()
        plt.grid(True, axis='y')
        plt.show()
    except ImportError:
        print("\nmatplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
