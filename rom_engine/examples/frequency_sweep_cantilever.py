# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
frequency_sweep_cantilever.py -- end-to-end rom_engine.frequency example.

Pipeline demonstrated:
    fea_engine damped full-order model  -->  modal (or POD-on-FRF-
    snapshots) reduced basis  -->  FrequencyROM  -->  fast harmonic
    frequency sweep, cross-checked against fea_engine's own
    solve_frequency_sweep() the whole way.

Physical setup: the same cantilever beam used throughout this
package's examples/tests, now with Rayleigh (proportional) damping
C = alpha*M + beta*K, built via fea_engine's own RayleighDamping. A
frequency sweep -- the response amplitude at a fixed output point,
swept across hundreds of driving frequencies -- is exactly the "many
queries against the same structure" situation rom_engine's whole
offline/online design exists for: build the reduced basis and project
{M, C, K} onto it ONCE, then every subsequent frequency costs a tiny
reduced solve instead of a full-order factorization.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import time
import numpy as np
from scipy.linalg import eigh

from fea_engine.material import EI_beam, Material, Section
from fea_engine.geometry import generate_mesh
from fea_engine.solver import FESystem
from fea_engine.damping import RayleighDamping
from fea_engine import elements

from rom_engine import FrequencyROM, build_pod_basis_from_frf_snapshots


def build_damped_cantilever(n=150, L=1.0, E=210e9, rho=7800.0, A=0.01, I=8.33e-6,
                             alpha=2.0, beta=1e-5):
    """A fixed-free Euler-Bernoulli cantilever with Rayleigh damping,
    built entirely through fea_engine's own API (assemble_stiffness,
    assemble_mass, assemble_damping) -- the fixture used throughout
    this module mirrors this exactly, so this example is a faithful
    stand-in for the validated test suite's setup, just with a larger
    mesh to make the reduced-order speedup obvious."""
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
    print("rom_engine.frequency example: damped cantilever frequency sweep")
    print("=" * 70)

    sysobj, alpha, beta = build_damped_cantilever(n=150)
    free = sysobj.free_dofs
    n_dof = sysobj.n_dof
    Kff = sysobj.K[np.ix_(free, free)]
    Mff = sysobj.M[np.ix_(free, free)]

    eigvals, eigvecs = eigh(Kff, Mff)
    omega_n = np.sqrt(np.clip(eigvals, 0, None))
    print(f"model: {n_dof} dof ({len(free)} free), "
          f"first natural frequency = {omega_n[0]:.2f} rad/s "
          f"({omega_n[0] / (2 * np.pi):.2f} Hz)")

    # tip transverse load, tip transverse output -- the classic FRF setup
    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    # --- build the ROM: modal basis (first 20 undamped mode shapes) ---
    n_modes = 20
    basis = eigvecs[:, :n_modes]
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(alpha, beta))
    print(f"\nFrequencyROM built from {n_modes} undamped mode shapes "
          f"(proportional-damping collapse: 2-term {{M,K}} decomposition)")

    # --- sweep, ROM vs fea_engine's own full-order sweep -------------
    omegas = np.linspace(0.1, 3.5, 400) * omega_n[0]

    t0 = time.perf_counter()
    resp_rom = rom.frequency_response(omegas, F_free, output_dofs=[len(free) - 2])[:, 0]
    t_rom = time.perf_counter() - t0

    t0 = time.perf_counter()
    U_full = sysobj.solve_frequency_sweep(omegas, F_full)
    resp_full = U_full[:, n_dof - 2]
    t_full = time.perf_counter() - t0

    rel_err = np.abs(resp_rom - resp_full) / np.max(np.abs(resp_full))
    print(f"\n{len(omegas)}-point sweep, tip transverse response:")
    print(f"  fea_engine full solve_frequency_sweep(): {t_full * 1e3:8.2f} ms")
    print(f"  rom_engine FrequencyROM.frequency_response(): {t_rom * 1e3:8.2f} ms")
    print(f"  speedup: {t_full / t_rom:.1f}x")
    print(f"  max relative error across the whole sweep: {np.max(rel_err):.3e}")
    print(f"  relative error AT the first resonance peak: {rel_err[np.argmax(np.abs(resp_full))]:.3e}")

    # --- POD-on-FRF-snapshots basis, same rank, held-out comparison ---
    print(f"\nComparing to a POD-on-FRF-snapshots basis, same rank ({n_modes}):")
    training_omegas = np.linspace(0.1, 3.5, 10) * omega_n[0]
    pod_basis = build_pod_basis_from_frf_snapshots(
        training_omegas, Mff, Kff, F_free, C=sysobj.C[np.ix_(free, free)], n_modes=n_modes)
    rom_pod = FrequencyROM.from_MCK(Mff, Kff, pod_basis.V, rayleigh=(alpha, beta))
    resp_pod = rom_pod.frequency_response(omegas, F_free, output_dofs=[len(free) - 2])[:, 0]
    rel_err_pod = np.abs(resp_pod - resp_full) / np.max(np.abs(resp_full))
    print(f"  POD basis energy captured: {pod_basis.energy_captured():.6f}")
    print(f"  max relative error across the whole sweep (POD basis): {np.max(rel_err_pod):.3e}")
    print(f"  (modal basis, for comparison): {np.max(rel_err):.3e}")

    print("\n" + "=" * 70)
    print("Done. Both basis choices track fea_engine's own full-order sweep")
    print("closely, at a large speedup once the basis is built.")
    print("=" * 70)

    # Plot the results if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(omegas / (2 * np.pi), np.abs(resp_full), label="Full-order FEA", color='black')
        plt.plot(omegas / (2 * np.pi), np.abs(resp_rom), '--', label="ROM (modal basis)", color='blue')
        plt.plot(omegas / (2 * np.pi), np.abs(resp_pod), ':', label="ROM (POD basis)", color='orange')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Tip Transverse Response Amplitude")
        plt.title("Frequency Sweep of Damped Cantilever Beam")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nmatplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
