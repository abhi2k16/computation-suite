"""
test_passivity.py -- validates rom_engine.passivity's diagnostics
against a REAL, damped fea_engine cantilever beam, using the SAME
collocated tip-force-in/tip-out port convention every other module in
this "systems and control" family (state_space.py/krylov.py/
balanced_truncation.py) already uses (docs/classical_mor_roadmap.md
Section 12).

Checks:
  1. Sanity check on the fixture and on is_passive() itself: the
     FULL-ORDER fea_engine model (velocity output) is passive across a
     wide sweep including near resonance.
  2. THE closed-form theorem, checked directly rather than only cited:
     a modally-reduced FrequencyROM (galerkin.py's basis) stays passive
     at every tested basis size.
  3. Whether KrylovROM/BalancedTruncationROM/
     FrequencyWeightedBalancedTruncationROM preserve or violate
     passivity is CHECKED, not assumed either way, and reported
     honestly -- none of them have a passivity-preservation theorem
     behind them (first-order state-space reductions, not second-order
     Galerkin ones), so this is a genuine, open question this test
     answers empirically for this fixture.
"""
__author__ = "Abhijeet"
import numpy as np
from scipy.linalg import eigh
from rom_engine import FrequencyROM
from rom_engine.krylov import KrylovROM
from rom_engine.balanced_truncation import (
    BalancedTruncationROM, FrequencyWeightedBalancedTruncationROM, bandpass_weight,
)
from rom_engine.passivity import velocity_transfer_function, passivity_margin, is_passive
import fea_fixtures as ff


def _fixture(n=25, alpha=2.0, beta=1e-5):
    fx = ff.damped_cantilever_beam_system(n=n, alpha=alpha, beta=beta)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]
    n_free = len(free)
    tip_local = n_free - 2
    return fx, free, Kff, Mff, Cff, tip_local


def _full_order_H_disp(fx, free, tip_local, omega_array):
    n_dof = fx["n_dof"]
    tip_dof_global = free[tip_local]
    F_full = np.zeros(n_dof)
    F_full[tip_dof_global] = 1.0
    sysobj = fx["sys"]
    return np.array([sysobj.solve_harmonic(om, F_full)[tip_dof_global] for om in omega_array])


def test_full_order_is_passive():
    fx, free, Kff, Mff, Cff, tip_local = _fixture()
    eigvals, _ = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    omega_array = np.linspace(0.05 * omega1, 3.0 * omega1, 300)   # includes resonance

    H_disp = _full_order_H_disp(fx, free, tip_local, omega_array)
    H_vel = velocity_transfer_function(H_disp, omega_array)
    margin = passivity_margin(H_vel)

    print(f"full-order model: min passivity margin across {len(omega_array)} points "
          f"(incl. resonance) = {margin.min():.3e}")
    assert is_passive(H_vel, tol=1e-9), (
        "the full-order fea_engine model (collocated tip force/tip velocity) "
        "must be passive -- this is the closed-form theorem (module docstring) "
        "applied to the UN-reduced system, and a failure here would mean "
        "either a fixture/damping-sign bug or a bug in is_passive() itself"
    )
    print("PASS -- full-order model is passive across a wide sweep including resonance")


def test_modal_frequency_rom_stays_passive():
    fx, free, Kff, Mff, Cff, tip_local = _fixture()
    eigvals, eigvecs = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    omega_array = np.linspace(0.05 * omega1, 3.0 * omega1, 300)

    F_free = np.zeros(len(free)); F_free[tip_local] = 1.0

    for n_modes in (3, 6, 10, 15):
        basis = eigvecs[:, :n_modes]
        rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))
        H_disp = rom.frequency_response(omega_array, F_free, output_dofs=[tip_local])[:, 0]
        H_vel = velocity_transfer_function(H_disp, omega_array)
        margin = passivity_margin(H_vel)
        print(f"n_modes={n_modes}: min passivity margin = {margin.min():.3e}")
        assert is_passive(H_vel, tol=1e-9), (
            f"a modally-reduced FrequencyROM (n_modes={n_modes}) should stay "
            f"passive -- this is the closed-form Galerkin-congruence theorem "
            f"(module docstring), checked directly here, not just cited"
        )
    print("PASS -- modally-reduced FrequencyROM stays passive at every tested basis size")


def test_state_space_reductions_passivity_checked_not_assumed():
    """KrylovROM/BalancedTruncationROM/FrequencyWeightedBalancedTruncationROM
    have NO passivity-preservation theorem behind them -- this test
    genuinely checks (does not assume) whether each one preserves or
    violates passivity on this real fixture, and reports the finding
    honestly either way, exactly like test_krylov.py's own handling of
    KrylovROM.is_stable()'s non-guarantee."""
    fx, free, Kff, Mff, Cff, tip_local = _fixture()
    eigvals, eigvecs = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    omega_array = np.linspace(0.05 * omega1, 3.0 * omega1, 300)

    B = np.zeros((len(free), 1)); B[tip_local, 0] = 1.0
    Cout = np.zeros((1, len(free))); Cout[0, tip_local] = 1.0

    results = {}

    for r in (6, 10, 14):
        krylov = KrylovROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=0.0, k=r)
        H_disp = krylov.frequency_response(omega_array)
        H_vel = velocity_transfer_function(H_disp, omega_array)
        results[f"KrylovROM(k={r})"] = is_passive(H_vel, tol=1e-9)

        bt = BalancedTruncationROM.from_MCK(Mff, Kff, B, Cout, C=Cff, r=r)
        H_disp = bt.frequency_response(omega_array)
        H_vel = velocity_transfer_function(H_disp, omega_array)
        results[f"BalancedTruncationROM(r={r})"] = is_passive(H_vel, tol=1e-9)

        Wo = bandpass_weight(omega1, zeta=0.2)
        fwbt = FrequencyWeightedBalancedTruncationROM.from_MCK(Mff, Kff, B, Cout, C=Cff, r=r, Wo=Wo)
        H_disp = fwbt.frequency_response(omega_array)
        H_vel = velocity_transfer_function(H_disp, omega_array)
        results[f"FrequencyWeightedBalancedTruncationROM(r={r})"] = is_passive(H_vel, tol=1e-9)

    print("passivity by method/order (checked, not assumed, on this real fixture):")
    for name, passive in results.items():
        print(f"  {name}: {'passive' if passive else 'VIOLATES passivity'}")

    n_violations = sum(1 for v in results.values() if not v)
    if n_violations > 0:
        print(f"FINDING: {n_violations}/{len(results)} state-space reductions tested "
              f"VIOLATE passivity on this fixture at at least one tested order -- "
              f"concrete evidence that these methods' other guarantees (H-infinity "
              f"error bound, moment matching) do NOT imply passivity preservation, "
              f"which is exactly why the closed-form second-order-Galerkin route "
              f"(test_modal_frequency_rom_stays_passive) is the one with an actual "
              f"passivity theorem behind it.")
    else:
        print("FINDING: none of the tested state-space reductions violated passivity "
              "at the orders tested on this fixture -- reported honestly; this does "
              "NOT mean they are guaranteed to (no such theorem exists for them), "
              "only that no violation was observed here.")
    # no directional assertion here by design -- see docstring
    assert all(isinstance(v, bool) for v in results.values())
