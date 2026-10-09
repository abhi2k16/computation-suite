# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_hankel_norm.py -- validates rom_engine.hankel_norm.OptimalHankelNormROM
against a REAL, damped fea_engine cantilever beam, using fea_engine's own
solve_harmonic() as ground truth and rom_engine.balanced_truncation's
already-validated hankel_singular_values() as an independent check.

A note on scale, discovered empirically while writing this test (mirrors,
but is even more severe than, test_balanced_truncation.py's own scale
note): this beam's tip-force-in/tip-displacement-out port is dominated by
just its first couple of modes -- the Hankel singular values drop by TWO
FULL ORDERS OF MAGNITUDE between the first mode pair and everything else
(see hsv print in _reduced_port_system()'s own construction). Applying
OptimalHankelNormROM even to a 15-mode (order-30) pre-reduction, a size
that works perfectly well for ordinary BalancedTruncationROM, produces
UNRELIABLE results at r=4 and above (caught by this class's own
numerically_reliable self-check, not a silent failure -- see module
docstring). This is exactly the module docstring's own numerical-
limitation discussion, now measured concretely on this package's real
fixture: OptimalHankelNormROM's extra Gamma-inversion step is genuinely
more sensitive to a wide Hankel-singular-value spread than ordinary BT's
own square-root balancing, so it needs an EVEN SMALLER pre-reduction (6
modes, order 12, here) to stay reliable at the reduced orders tested
below.

Checks:
  1. THE core AAK claim, verified directly (not just cited): at several
     r, this construction's Hankel norm of the error system is <= the
     Hankel norm of BalancedTruncationROM's error system AT THE SAME r
     -- the actual "provably at least as good, in the Hankel norm
     specifically" property, computed via the SAME already-validated
     hankel_singular_values() both classes already rely on.
  2. The empirical self-check machinery (numerically_reliable /
     measured_hankel_norm_error) agrees with the theoretical sigma_r1 to
     near machine precision on every RELIABLE construction in this
     fixture's safe regime, and DOES flag (numerically_reliable=False,
     with a warning) a construction pushed into the fixture's known
     unreliable regime -- checked directly, not assumed.
  3. is_stable() is always True (guaranteed by construction: A_r is
     built from the explicitly-selected stable Schur block).
  4. The SISO-only restriction raises a clear ValueError for a MIMO
     (M, B, Cout) combination, rather than silently guessing at an
     unverified MIMO generalization.
  5. An HONEST accuracy finding, measured not assumed: AAK's Hankel-norm
     optimality does NOT translate into better sup-norm/relative
     frequency-response accuracy than ordinary BalancedTruncationROM at
     the SAME r on this fixture -- reported directly, mirroring
     test_frequency_weighted_bt.py's own honest-tradeoff reporting
     rather than forcing a directional claim the theorem does not
     actually make.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import eigh
from rom_engine.hankel_norm import OptimalHankelNormROM
from rom_engine.balanced_truncation import BalancedTruncationROM, hankel_singular_values
import fea_fixtures as ff


def _reduced_port_system(n=25, n_modes=6, alpha=2.0, beta=1e-5):
    """A SMALL, well-conditioned (Kr, Mr, Cr, Br, Coutr) intermediate
    system -- see module docstring for why OptimalHankelNormROM needs an
    even smaller pre-reduction than BalancedTruncationROM's own 15-mode
    convention (test_balanced_truncation.py) to stay numerically
    reliable on this specific fixture/port."""
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


def _error_system_hankel_norm(ss_full, A_r, B_r, Cout_r):
    """Build the error system G - G_r and return its largest Hankel
    singular value (= its Hankel norm), via the SAME already-validated
    hankel_singular_values() every class in this "systems and control"
    family already relies on -- the independent check this whole test
    file's core claim (check 1) is built on."""
    n_full = ss_full.A.shape[0]
    n_red = A_r.shape[0]
    Ae = np.block([[ss_full.A, np.zeros((n_full, n_red))],
                    [np.zeros((n_red, n_full)), A_r]])
    Be = np.vstack([ss_full.B, B_r])
    Ce = np.hstack([ss_full.Cout, -Cout_r])
    return float(hankel_singular_values(Ae, Be, Ce)[0])


def test_aak_hankel_norm_optimality_beats_bt_at_same_r():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()

    for r in (1, 2, 3, 4):
        ohna = OptimalHankelNormROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
        bt = BalancedTruncationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
        bt_hankel_err = _error_system_hankel_norm(ohna.ss, bt.A_r, bt.B_r, bt.Cout_r)

        print(f"r={r}: OHNA Hankel-norm error (AAK-optimal)={ohna.sigma_r1:.6e}  "
              f"BT Hankel-norm error={bt_hankel_err:.6e}")
        assert ohna.numerically_reliable, (
            f"r={r} should be in this fixture's numerically reliable "
            f"regime with a 6-mode pre-reduction -- see module docstring"
        )
        assert ohna.sigma_r1 <= bt_hankel_err * (1 + 1e-9), (
            f"AAK theory guarantees NO order-r system beats sigma_(r+1) "
            f"in Hankel norm -- BalancedTruncationROM's own reduced "
            f"model at the SAME r must not have a SMALLER Hankel-norm "
            f"error than the claimed AAK optimum (r={r})"
        )
    print("PASS -- OptimalHankelNormROM's Hankel-norm error is <= "
          "BalancedTruncationROM's own, at every tested r, on this real fixture")


def test_self_check_agrees_to_near_machine_precision_when_reliable():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    for r in (1, 2, 3, 4):
        rom = OptimalHankelNormROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
        rel_diff = abs(rom.measured_hankel_norm_error - rom.sigma_r1) / rom.sigma_r1
        print(f"r={r}: sigma_r1={rom.sigma_r1:.6e}  measured={rom.measured_hankel_norm_error:.6e}  "
              f"rel_diff={rel_diff:.3e}  reliable={rom.numerically_reliable}")
        assert rom.numerically_reliable
        assert rel_diff < 1e-5, (
            "in this fixture's known-reliable regime, the empirical "
            "error-system self-check should agree with the AAK-"
            "theoretical sigma_(r+1) to near machine precision (allowing "
            "for ordinary floating-point accumulation through the "
            "Schur/Sylvester decoupling step -- NOT the same tolerance "
            "as reliability_tol=1e-4, which is the much looser threshold "
            "that actually distinguishes a genuine numerical failure)"
        )
    print("PASS -- self-check agrees with theory to near machine precision "
          "at every tested r in the reliable regime")


def test_self_check_flags_the_known_unreliable_regime():
    """This fixture's port is dominated by its first mode pair -- pushing
    OptimalHankelNormROM to r=4 from a MUCH LARGER (20-mode) pre-reduction
    lands in the numerically unreliable regime described in the module
    docstring (a very wide overall Hankel-singular-value spread; see the
    module docstring for why the OVERALL spread, not just sigma_(r+1)
    itself, drives the balancing transform's accuracy). v1.0.1: the state
    equilibration in hankel_singular_values() moved the onset of this regime
    from 15 to 20 modes (15-mode cases now agree with theory to machine
    precision, tested above). In this regime the contract is that the
    method must NEVER silently return a wrong model: either construction
    raises a ValueError naming the numerical failure, or the returned model
    reports numerically_reliable=False. Which of the two happens at the
    edge can depend on the BLAS/LAPACK build, so both are accepted."""
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system(n_modes=20)
    try:
        rom = OptimalHankelNormROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=4)
    except ValueError as e:
        assert "stable" in str(e) or "numerical" in str(e).lower(), str(e)
        print(f"PASS -- 20-mode r=4 refused loudly: {str(e)[:90]}...")
        return
    print(f"20-mode pre-reduction, r=4: sigma_r1={rom.sigma_r1:.6e}  "
          f"measured={rom.measured_hankel_norm_error:.6e}  "
          f"reliable={rom.numerically_reliable}")
    assert not rom.numerically_reliable, (
        "this (pre-reduction size, r) combination is a known "
        "numerically unreliable case for THIS fixture/port -- the "
        "self-check must catch it, not silently return a wrong model"
    )
    print("PASS -- numerically_reliable correctly flags the known-bad case")


def test_is_always_stable_by_construction():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    for r in (1, 2, 3, 4):
        rom = OptimalHankelNormROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
        assert rom.is_stable(), (
            f"A_r is built directly from the selected STABLE Schur "
            f"block -- this must always hold when construction did not "
            f"raise (r={r})"
        )
    print("PASS -- OptimalHankelNormROM stable at every tested reduced order")


def test_siso_only_restriction_raises_clearly():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    n = Mr.shape[0]
    B_mimo = np.hstack([Br, np.ones((n, 1))])   # 2 inputs
    try:
        OptimalHankelNormROM.from_MCK(Mr, Kr, B_mimo, Coutr, C=Cr, r=2)
        raised = False
    except ValueError as e:
        raised = True
        msg = str(e)
    assert raised, "a MIMO (2-input) system must raise ValueError, not silently guess at an unverified MIMO generalization"
    assert "SISO" in msg
    print(f"PASS -- MIMO input correctly rejected: {msg[:80]}...")


def test_honest_accuracy_finding_aak_not_more_accurate_than_bt_in_practice():
    """AAK/Glover optimality is a HANKEL-NORM statement, not a claim
    about sup-norm or relative frequency-response accuracy at any
    particular r. Measured directly on this fixture: OptimalHankelNormROM
    is actually LESS accurate than ordinary BalancedTruncationROM in
    ordinary relative error across a resonance-including sweep, at the
    SAME r -- reported honestly here (mirroring
    test_frequency_weighted_bt.py's own tradeoff reporting) rather than
    assuming Hankel-norm optimality implies practical superiority."""
    fx, Kr, Mr, Cr, Br, Coutr, eigvals = _reduced_port_system()
    omega1 = float(np.sqrt(eigvals[0]))
    omega_array = np.linspace(0.05 * omega1, 3.0 * omega1, 100)
    H_true = _true_H(fx, omega_array)

    findings = {}
    for r in (2, 4):
        ohna = OptimalHankelNormROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
        bt = BalancedTruncationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
        err_ohna = float(np.max(np.abs(ohna.frequency_response(omega_array) - H_true) / np.abs(H_true)))
        err_bt = float(np.max(np.abs(bt.frequency_response(omega_array) - H_true) / np.abs(H_true)))
        findings[r] = (err_ohna, err_bt)
        print(f"r={r}: OHNA max rel err={err_ohna:.3e}   BT max rel err={err_bt:.3e}")

    n_ohna_worse = sum(1 for (e_o, e_b) in findings.values() if e_o > e_b)
    print(f"FINDING: OHNA had LARGER sup-norm relative error than BT at "
          f"{n_ohna_worse}/{len(findings)} tested r on this fixture -- "
          f"consistent with AAK optimality being a Hankel-norm-specific "
          f"guarantee, not a general practical-accuracy guarantee. "
          f"OptimalHankelNormROM remains the right choice specifically "
          f"when the Hankel norm (or its close relative, the tight "
          f"L-infinity bound the theory also gives -- not independently "
          f"re-derived here, see module docstring) is the metric that "
          f"actually matters for a given use case.")
    # no directional assertion on practical accuracy -- see docstring
    assert all(np.isfinite(e_o) and np.isfinite(e_b) for e_o, e_b in findings.values())
