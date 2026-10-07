"""
rom_04_balanced_truncation.py -- Example Gallery: state-space balanced
truncation model-order reduction (balanced_truncation.py) on a
damped cantilever beam -- the linear-systems-theory sibling of the
modal/POD-based reduction the other gallery examples use, with its
own distinguishing feature: a genuine, frequency-independent A PRIORI
error bound (h_infinity_error_bound(), Pernebo & Silverman 1982),
known BEFORE a single frequency response is ever evaluated.

Panel 1: Hankel singular value decay -- the "how many states actually
matter" signature this whole method is built on; the sharp elbow is
what makes a small r a good truncation choice, not an arbitrary one.
Panel 2: frequency response at the tip (tip-force-to-tip-displacement)
-- a small (r=8, i.e. 4 structural "modes" worth of states)
BalancedTruncationROM vs. an INDEPENDENT full-order fea_engine
solve_frequency_sweep(), with the a priori H-infinity error bound
checked (not assumed) to actually dominate the observed error.

Two-stage reduction, matching this package's own established fix
(tests/test_balanced_truncation.py's module docstring): applying
BalancedTruncationROM directly to the raw beam's full free-dof
state-space is numerically unreliable, NOT a bug -- an Euler-Bernoulli
beam's natural frequencies span many orders of magnitude (the most-
refined-element modes sit far above the lowest structural ones), and
scipy's dense Bartels-Stewart Lyapunov solver loses accuracy as that
spread grows. The standard fix, used here exactly as that test file
establishes it: first modally truncate to a moderate, well-conditioned
number of undamped modes (plain eigh(K,M)), THEN balance-and-truncate
further from that intermediate model.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from fea_engine.material import EI_beam, Material, Section
from fea_engine.geometry import generate_mesh
from fea_engine.solver import FESystem
from fea_engine.damping import RayleighDamping
from fea_engine import elements
from scipy.linalg import eigh

from rom_engine.balanced_truncation import BalancedTruncationROM

N, L = 24, 1.0
E, RHO, A_SEC, I_SEC = 210e9, 7800.0, 0.01, 8.33e-6
ALPHA, BETA = 2.0, 1e-5
N_MODES_PRE = 15    # stage 1: modal pre-reduction (well-conditioned, matches test_balanced_truncation.py)
R_REDUCED = 8        # stage 2: balanced truncation on the pre-reduced system


def main():
    mesh = generate_mesh(dim=1, L=L, n=N)
    elem = elements.Beam2DEulerBernoulli()
    fes = FESystem(mesh, elem)
    mat = Material(E=E, nu=0.3, rho=RHO)
    sec = Section(A=A_SEC, I=I_SEC)
    EI = EI_beam(mat, sec)
    fes.assemble_stiffness(EI)
    fes.assemble_mass(RHO * A_SEC)
    fes.assemble_damping(RayleighDamping(alpha=ALPHA, beta=BETA))
    fes.fix_dofs([0], [0, 1])
    free = np.asarray(fes.free_dofs)
    n_dof = fes.n_dof

    Kff = fes.K[np.ix_(free, free)]
    Mff = fes.M[np.ix_(free, free)]
    Cff = fes.C[np.ix_(free, free)]

    tip_dof_local = len(free) - 2   # tip transverse displacement, free-dof numbering
    B_in = np.zeros(len(free)); B_in[tip_dof_local] = 1.0
    Cout = B_in.copy()   # collocated: force in, displacement out, same dof

    # Stage 1 -- modal pre-reduction: the raw 2*len(free) free-dof beam
    # state-space spans a huge eigenvalue range (many orders of magnitude
    # between the lowest bending mode and the most-refined-element modes),
    # which degrades scipy's dense Bartels-Stewart Lyapunov solver used
    # internally by balanced truncation. Truncating to a moderate number of
    # undamped modes first gives BalancedTruncationROM a well-conditioned
    # system to work with -- exactly the pattern
    # tests/test_balanced_truncation.py's _reduced_port_system() uses.
    eigvals, Vfull = eigh(Kff, Mff)
    n_modes_pre = min(N_MODES_PRE, len(free))
    Vt = Vfull[:, :n_modes_pre]
    Kr = Vt.T @ Kff @ Vt
    Mr = Vt.T @ Mff @ Vt
    Cr = Vt.T @ Cff @ Vt
    Br = Vt.T @ B_in
    Coutr = Vt.T @ Cout

    # Stage 2 -- balanced truncation on the well-conditioned modal model.
    rom = BalancedTruncationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=R_REDUCED)
    print(f"Full-order state dimension: {2*len(free)}  ->  modal pre-reduction: "
          f"{2*n_modes_pre} ({n_modes_pre} modes)  ->  balanced truncation: {R_REDUCED} states")
    print(f"Hankel singular values (first 10): {np.round(rom.hsv[:10], 6)}")
    hinf_bound = rom.h_infinity_error_bound()
    print(f"A priori H-infinity error bound (before any frequency evaluated): {hinf_bound:.4e}")

    omega1 = np.sqrt(max(eigvals[0], 0.0))
    omegas = np.linspace(0.05, 8.0, 400) * omega1

    H_rom = rom.frequency_response(omegas)

    F0_full = np.zeros(n_dof)
    F0_full[free[tip_dof_local]] = 1.0
    U_fom = fes.solve_frequency_sweep(omegas, F0_full)
    H_fom = U_fom[:, free[tip_dof_local]]

    actual_err = np.abs(H_rom - H_fom)
    within_bound = np.all(actual_err <= hinf_bound * 1.001)   # tiny slack for float roundoff

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax1.semilogy(np.arange(1, len(rom.hsv) + 1), rom.hsv, 'o-', color='steelblue', markersize=4)
    ax1.axvline(R_REDUCED, color='tomato', linestyle='--', linewidth=1.2,
                label=f'truncation at r={R_REDUCED}')
    ax1.set_xlabel('index')
    ax1.set_ylabel('Hankel singular value')
    ax1.set_title('Hankel singular value decay')
    ax1.grid(True, alpha=0.35, which='both')
    ax1.legend()

    ax2.semilogy(omegas / omega1, np.abs(H_fom), '-', color='steelblue', linewidth=2.2,
                  label='full-order fea_engine (solve_frequency_sweep)')
    ax2.semilogy(omegas / omega1, np.abs(H_rom), '--', color='tomato', linewidth=1.6,
                  label=f'BalancedTruncationROM (r={R_REDUCED})')
    ax2.set_xlabel(r'$\omega / \omega_1$')
    ax2.set_ylabel('|tip displacement / tip force|')
    ax2.set_title(f'Frequency response -- max actual error {actual_err.max():.2e}\n'
                   f'vs. a priori H-inf bound {hinf_bound:.2e} '
                   f'({"bound holds" if within_bound else "BOUND VIOLATED"})')
    ax2.grid(True, alpha=0.35, which='both')
    ax2.legend(fontsize=9)

    fig.suptitle('Balanced Truncation -- damped cantilever beam (rom_engine)', fontsize=13)
    out = os.path.join(os.path.dirname(__file__), 'rom_04_balanced_truncation.png')
    fig.savefig(out, dpi=150)
    print(f"Saved rom_04_balanced_truncation.png")
    print(f"Max actual |H_rom - H_fom| over the sweep: {actual_err.max():.4e} "
          f"(bound: {hinf_bound:.4e}, {'HOLDS' if within_bound else 'VIOLATED'})")


if __name__ == "__main__":
    main()
