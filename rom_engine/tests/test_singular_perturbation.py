# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_singular_perturbation.py -- validates
rom_engine.balanced_truncation.SingularPerturbationROM (SPA) against a
REAL, damped fea_engine cantilever beam, using the SAME fixture,
pre-reduction pattern, and port (tip-force-in/tip-displacement-out) as
test_balanced_truncation.py, so BalancedTruncationROM and
SingularPerturbationROM are compared on identical footing throughout
(docs/classical_mor_roadmap.md Section 8).

Checks:
  1. DC-gain exactness: SPA's transfer_function() at a very small s
     matches the FULL-ORDER system's own DC gain
     (Cout @ (-A)^-1 @ B, computed independently of both ROM classes)
     to numerical precision -- the concrete version of "SPA fixes
     ordinary BT's DC-gain gap", not just a claim.
  2. Ordinary BT's DC-gain error, for contrast, measured at the SAME r
     on the SAME model -- making the comparison a real measurement,
     not an assumption (if BT's error happened not to be larger, that
     would be reported honestly here too).
  3. The shared H-infinity error bound (2 * sum(discarded Hankel
     singular values)) actually bounds SPA's true absolute error too,
     mirroring test_balanced_truncation.py's own check for ordinary BT.
  4. SPA remains stable at every tested r -- also a theorem (Liu &
     Anderson 1989), checked as a regression guard exactly like
     BalancedTruncationROM.is_stable().
  5. SPA stays accurate across the WHOLE swept range too (not just at
     DC) -- it should not have traded away BT's global-accuracy
     property to fix the DC-specific one.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import eigh
from rom_engine.balanced_truncation import BalancedTruncationROM, SingularPerturbationROM
import fea_fixtures as ff


def _reduced_port_system(n=25, n_modes=15, alpha=2.0, beta=1e-5):
    """Identical to test_balanced_truncation.py's own helper -- see
    that file's module docstring for why this pre-reduction step
    (modally truncate the raw FE model before balancing) matters."""
    fx = ff.damped_cantilever_beam_system(n=n, alpha=alpha, beta=beta)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]
    n_free = len(free)
    tip_local = n_free - 2
    B = np.zeros((n_free, 1)); B[tip_local, 0] = 1.0
    Cout = np.zeros((1, n_free)); Cout[0, tip_local] = 1.0

    eigvals, V = eigh(Kff, Mff)
    Vt = V[:, :n_modes]
    Kr = Vt.T @ Kff @ Vt
    Mr = Vt.T @ Mff @ Vt
    Cr = Vt.T @ Cff @ Vt
    Br = Vt.T @ B
    Coutr = Cout @ Vt
    return fx, Kr, Mr, Cr, Br, Coutr, eigvals


def _true_H(fx, omega_array):
    n_dof = fx["n_dof"]
    tip_dof_global = n_dof - 2
    F_full = np.zeros(n_dof)
    F_full[tip_dof_global] = 1.0
    sysobj = fx["sys"]
    return np.array([sysobj.solve_harmonic(om, F_full)[tip_dof_global] for om in omega_array])


def _full_order_dc_gain(Kr, Mr, Cr, Br, Coutr):
    """Cout @ (-A)^-1 @ B for the SAME reduced-but-not-yet-balanced
    (Kr, Mr, Cr, Br, Coutr) intermediate model both ROM classes are
    built from, computed independently of SingularPerturbationROM/
    BalancedTruncationROM (direct first-order assembly + solve, not
    through either class) -- the ground truth DC gain both ROMs are
    compared against."""
    from rom_engine.state_space import to_state_space
    ss = to_state_space(Mr, Kr, C=Cr, B=Br, Cout=Coutr, form="A")
    x_dc = np.linalg.solve(-ss.A, ss.B)
    return (ss.Cout @ x_dc)[0, 0]


def test_dc_gain_exactness():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    r = 10
    spa = SingularPerturbationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
    bt = BalancedTruncationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)

    true_dc = _full_order_dc_gain(Kr, Mr, Cr, Br, Coutr)
    spa_dc = spa.transfer_function(1e-8)[0, 0]
    bt_dc = bt.transfer_function(1e-8)[0, 0]

    spa_err = abs(spa_dc - true_dc) / abs(true_dc)
    bt_err = abs(bt_dc - true_dc) / abs(true_dc)
    print(f"true DC gain={true_dc:.6e}  SPA DC gain={spa_dc:.6e} "
          f"(rel err={spa_err:.3e})  ordinary-BT DC gain={bt_dc:.6e} "
          f"(rel err={bt_err:.3e})")

    assert spa_err < 1e-6, (
        "SingularPerturbationROM should match the full-order DC gain "
        "to numerical precision -- this is the whole point of SPA, "
        "an algebraic identity, not an approximation"
    )
    print(f"FINDING: SPA's DC-gain error ({spa_err:.3e}) vs. ordinary BT's "
          f"({bt_err:.3e}) -- SPA {'is' if spa_err < bt_err else 'is NOT'} "
          f"more accurate at DC on this model, reported as measured.")


def test_shared_h_infinity_bound_holds_for_spa():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    r = 10
    spa = SingularPerturbationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
    bound = spa.h_infinity_error_bound()

    omega_fine = np.linspace(0.5, 2000.0, 200)
    H_spa = spa.frequency_response(omega_fine)
    H_true = _true_H(fx, omega_fine)
    true_sup_err = np.max(np.abs(H_spa - H_true))

    print(f"r={r}: SPA's (shared-with-BT) a priori H-infinity bound={bound:.6e}, "
          f"observed sup-norm ABSOLUTE error over {len(omega_fine)}-point "
          f"sweep={true_sup_err:.6e}")
    assert true_sup_err <= bound, (
        "SPA shares BalancedTruncationROM's a priori H-infinity error bound "
        "(Liu & Anderson 1989) -- it must actually bound the observed error"
    )


def test_is_always_stable():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    for r in (4, 8, 10, 12):
        rom = SingularPerturbationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
        assert rom.is_stable(), (
            f"SPA must ALWAYS preserve stability (a theorem, not an "
            f"empirical property, given a stable full-order model) -- "
            f"failed at r={r}"
        )
    print("PASS -- SingularPerturbationROM stable at every tested reduced order")


def test_accurate_across_whole_sweep():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    rom = SingularPerturbationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=10)

    omega_array = np.linspace(1.0, 1500.0, 25)
    H_rom = rom.frequency_response(omega_array)
    H_true = _true_H(fx, omega_array)
    rel_err = np.abs(H_rom - H_true) / np.abs(H_true)

    print(f"SingularPerturbationROM (r=10, from a 15-mode pre-reduction) "
          f"relative error across full sweep: max={rel_err.max():.3e}, "
          f"mean={rel_err.mean():.3e}")
    assert rel_err.max() < 1e-2, (
        "SPA should stay accurate across the WHOLE swept range too -- "
        "fixing the DC-specific gap should not cost the global-accuracy "
        "property BT already has"
    )
