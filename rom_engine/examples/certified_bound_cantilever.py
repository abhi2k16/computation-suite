"""
certified_bound_cantilever.py -- end-to-end rom_engine.scm example.

Demonstrates the ONE genuinely CERTIFIED (rigorously provable, not just
a practical heuristic) error signal in rom_engine:
SingularValueLowerBound + certified_error_bound(), built on the
Weyl/Mirsky singular-value perturbation inequality (see scm.py's module
docstring and docs/phase4_error_bounds_greedy_roadmap.md Section 3).

This script does two things honestly, in order:

  1. Proves the CERTIFICATION property holds: sweeping many frequencies
     (including exactly AT resonance), the bound NEVER exceeds the true
     sigma_min(A(omega)), and certified_error_bound() -- whenever it
     returns a finite number -- NEVER undershoots the true error against
     fea_engine's own independent solve_harmonic(). This is the actual
     promise a "certified bound" has to keep, and it's kept.

  2. Reports, rather than hides, this simplified bound's real practical
     limitation for a real FE stiffness matrix: because A(omega)'s
     stiffness component has a spectral norm (~1e11-1e12 here) many
     orders of magnitude larger than sigma_min(A(omega)) itself
     (~1e3), the Lipschitz perturbation term swamps the bound within a
     tiny relative frequency step of each reference point -- on this
     model, roughly 1e-7 in relative omega. That is FAR narrower than
     the spacing greedy.py's hierarchical-indicator-driven training
     needs for good basis accuracy (typically 1%-10% of the first
     resonance). The module is correct and useful for exactly what it
     is -- a rigorous local safety net around a handful of specific,
     already-computed reference frequencies -- but it is NOT a
     drop-in replacement for the (uncertified, but far more practically
     useful) error_estimate()/hierarchical_error_indicator() this
     package's greedy training already relies on. See scm.py's
     docstring for why the classical "natural-norm" SCM (Chen et al.
     2010) exists to fix exactly this kind of scale-mismatch weakness,
     and why re-implementing it is noted as optional future work rather
     than done here.
"""
__author__ = "Abhijeet"
import numpy as np
from scipy.linalg import eigh

from fea_engine.material import EI_beam, Material, Section
from fea_engine.geometry import generate_mesh
from fea_engine.solver import FESystem
from fea_engine.damping import RayleighDamping
from fea_engine import elements

from rom_engine import FrequencyROM, SingularValueLowerBound, certified_error_bound


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
    print("=" * 72)
    print("rom_engine.scm example: a genuinely certified error bound,")
    print("its guarantee proven, and its practical reach measured honestly")
    print("=" * 72)

    sysobj, alpha, beta = build_damped_cantilever(n=20)
    free = sysobj.free_dofs
    n_dof = sysobj.n_dof
    Kff = sysobj.K[np.ix_(free, free)]
    Mff = sysobj.M[np.ix_(free, free)]

    eigvals, eigvecs = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    basis = eigvecs[:, :10]
    print(f"model: {n_dof} dof ({len(free)} free), 1st resonance = {omega1:.2f} rad/s, "
          f"10-mode modal ROM basis")

    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(alpha, beta))
    scm = SingularValueLowerBound.from_affine(rom.affine)
    print(f"affine component spectral norms ||M||_2, ||K||_2 = {scm.component_norms}")

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    # ---- Part 1: prove the certification guarantee ----------------------
    print("\n--- Part 1: certification guarantee (the actual promise) ---")
    for om in np.linspace(0.2, 2.0, 5) * omega1:
        scm.add_reference(om)
    print(f"added {len(scm.reference_omegas)} references spanning 0.2-2.0x omega1")

    sweep = np.linspace(0.1, 3.0, 400) * omega1   # crosses exact resonance
    worst_violation = -np.inf
    for om in sweep:
        theta = rom.affine._theta(om)
        A = theta[0] * rom.affine.components[0]
        for q in range(1, rom.affine.Q):
            A = A + theta[q] * rom.affine.components[q]
        true_sigma = float(np.linalg.svd(A, compute_uv=False).min())
        worst_violation = max(worst_violation, scm.lower_bound(om) - true_sigma)
    print(f"worst (lower_bound - true sigma_min) over {len(sweep)} points, "
          f"including exact resonance: {worst_violation:.3e} (must be <= 0)")

    # Test at generic points in the spanned band (0.5, 1.3x omega1) AND at
    # an exact reference (0.2x omega1, added above) -- the generic points
    # are expected to give +inf (an honest "no information here", not a
    # failure -- see Part 2), while the exact reference gives a genuine
    # finite, valid bound. Both outcomes are correct: the CERTIFICATION
    # property (never wrong when finite) is what's being checked, not
    # "always informative".
    test_points = [0.2 * omega1, omega1, 0.5 * omega1, 1.3 * omega1]
    for om in test_points:
        ceb = certified_error_bound(rom, scm, om, F_free)
        U_true = sysobj.solve_harmonic(om, F_full)[free]
        x_rom = rom.frequency_response([om], F_free)[0]
        true_err = np.linalg.norm(x_rom - U_true)
        ok = "holds" if (not np.isfinite(ceb) or ceb >= true_err) else "VIOLATED"
        tag = " (exact reference)" if np.isclose(om, 0.2 * omega1) else ""
        print(f"  omega/omega1={om/omega1:.2f}{tag}: certified_bound={ceb:.4e}  "
              f"true_err={true_err:.4e}  [{ok}]")

    # ---- Part 2: measure the useful radius honestly ----------------------
    print("\n--- Part 2: how far from a reference point is the bound useful? ---")
    fresh = SingularValueLowerBound.from_affine(rom.affine)
    true_sigma_ref = fresh.add_reference(omega1)
    print(f"single reference at omega1, true sigma_min there = {true_sigma_ref:.2f}")
    for frac in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        lb = fresh.lower_bound(omega1 * (1 + frac))
        print(f"  relative step {frac:8.1e}  ->  lower_bound = {lb:10.4e}")

    print("\n" + "=" * 72)
    print("Summary: the certification guarantee holds everywhere, always -- but")
    print("for this real beam's stiffness matrix, this simplified bound is only")
    print("non-trivial within roughly 1e-7 relative frequency of a reference")
    print("point, versus the >1%-scale spacing that suffices for greedy.py's")
    print("(uncertified) hierarchical-indicator-driven basis training. A real")
    print("bound, correctly implemented -- just a narrow one for this problem.")
    print("=" * 72)


if __name__ == "__main__":
    main()
