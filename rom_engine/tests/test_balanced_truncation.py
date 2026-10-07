"""
test_balanced_truncation.py -- validates
rom_engine.balanced_truncation.BalancedTruncationROM against a REAL,
damped fea_engine cantilever beam (fea_fixtures.damped_cantilever_beam_system()),
using fea_engine's own solve_harmonic() as ground truth.

A note on scale, discovered empirically while writing this test: applying
BalancedTruncationROM directly to the RAW fea_engine beam's full free-dof
state-space (order ~2*n_free) makes the two dense Lyapunov solves
numerically unreliable, NOT because of a bug, but because Euler-Bernoulli
beam finite elements always carry a very wide range of natural
frequencies (the highest, most-refined-element "hourglass-adjacent" modes
sit many orders of magnitude above the lowest structural ones -- e.g. a
25-element beam here spans eigenvalues from -2.4 to -3*10^8), and scipy's
dense Bartels-Stewart solver (like any dense Lyapunov solver) loses
accuracy as that spread grows. This is exactly the "BT has a practical
size/conditioning ceiling" limitation flagged in
docs/classical_mor_roadmap.md Section 2c -- the fix used throughout this
file, and the recommended real-world pattern, is to first modally
truncate the raw FE model down to a moderate size (here: the lowest 15
undamped modes, via a plain eigh(K,M) -- no new machinery, this is
exactly what galerkin.GalerkinROM's own solve_modal() already does) and
apply BalancedTruncationROM to THAT well-conditioned intermediate model.
This two-stage pattern (coarse modal truncation, then balance-and-
truncate further) is standard practice for large-scale BT in the
literature, not a workaround specific to this test.

Checks (mirroring docs/classical_mor_roadmap.md Section 6):
  1. Accuracy stays good across the WHOLE swept range (not just near one
     point) -- the property mode-displacement/plain-Krylov methods don't
     have in general.
  2. is_stable() is always True -- a structural theorem for BT, checked
     here as a regression guard, not an empirical property.
  3. h_infinity_error_bound() actually BOUNDS (numerically dominates)
     the true sup-norm ABSOLUTE error over a fine frequency sweep (the
     bound is on |H(s)-H_r(s)| directly, NOT on relative error -- the
     test is careful to compare like with like).
  4. BalancedTruncationROM provides a genuine, checkable a priori,
     frequency-independent error CERTIFICATE (h_infinity_error_bound())
     that krylov.KrylovROM structurally has no equivalent of -- this is
     the real, fixture-independent distinction between the two methods
     this project can honestly stand behind (an attempt to also show
     "BT numerically beats Krylov at a specific far frequency" was tried
     while writing this test and dropped: for THIS particular beam and
     port, moment matching's local accuracy actually degrades gracefully
     enough that it remains competitive with, or better than, BT at the
     specific far frequencies tried, at these specific reduced orders --
     a real, fixture-dependent finding worth recording honestly rather
     than forcing a directional assertion the paper's own DIFFERENT
     benchmark system doesn't guarantee in general).
"""
import numpy as np
from scipy.linalg import eigh
from rom_engine.balanced_truncation import BalancedTruncationROM
from rom_engine.krylov import KrylovROM
import fea_fixtures as ff


def _reduced_port_system(n=25, n_modes=15, alpha=2.0, beta=1e-5):
    """A moderate-size, well-conditioned (Kr, Mr, Cr, Br, Coutr)
    intermediate system: the lowest n_modes undamped modes of a real
    fea_engine damped cantilever, with a unit tip-force-in / tip-
    displacement-out port projected through the same basis -- see
    module docstring for why this pre-reduction step matters here."""
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


def test_accurate_across_whole_sweep():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    rom = BalancedTruncationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=10)

    omega_array = np.linspace(1.0, 1500.0, 25)
    H_rom = rom.frequency_response(omega_array)
    H_true = _true_H(fx, omega_array)
    rel_err = np.abs(H_rom - H_true) / np.abs(H_true)

    print(f"BalancedTruncationROM (r=10, from a 15-mode pre-reduction) "
          f"relative error across full sweep: max={rel_err.max():.3e}, "
          f"mean={rel_err.mean():.3e}")
    assert rel_err.max() < 1e-2, "BT should stay accurate across the WHOLE swept range"


def test_is_always_stable():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    for r in (4, 8, 10, 12):
        rom = BalancedTruncationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
        assert rom.is_stable(), (
            f"balanced truncation must ALWAYS preserve stability (a theorem, "
            f"not an empirical property) -- failed at r={r}"
        )
    print("PASS -- BalancedTruncationROM stable at every tested reduced order")


def test_h_infinity_error_bound_actually_bounds_the_true_absolute_error():
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    r = 10
    rom = BalancedTruncationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
    bound = rom.h_infinity_error_bound()

    omega_fine = np.linspace(0.5, 2000.0, 200)
    H_rom = rom.frequency_response(omega_fine)
    H_true = _true_H(fx, omega_fine)
    true_sup_err = np.max(np.abs(H_rom - H_true))   # ABSOLUTE error -- the
                                                       # bound is on |H-H_r|,
                                                       # not a relative error

    print(f"r={r}: a priori H-infinity bound={bound:.6e}, observed sup-norm "
          f"ABSOLUTE error over {len(omega_fine)}-point sweep={true_sup_err:.6e}")
    assert true_sup_err <= bound, (
        "the a priori H-infinity error bound must actually bound the "
        "observed ABSOLUTE error, not just correlate with it"
    )


def test_bt_provides_a_certificate_krylov_structurally_cannot():
    """The real, fixture-independent distinction this project can stand
    behind (see module docstring for what was tried and dropped
    instead): BalancedTruncationROM can certify, BEFORE evaluating a
    single frequency, a hard ceiling on its worst-case error anywhere on
    the imaginary axis. krylov.KrylovROM has no equivalent method --
    its accuracy can only be assessed locally/a posteriori, exactly
    because moment matching carries no global energy-ranking argument
    the way balancing does."""
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    r = 10
    bt = BalancedTruncationROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, r=r)
    mm = KrylovROM.from_MCK(Mr, Kr, Br, Coutr, C=Cr, s0=0.0, k=r)

    assert hasattr(bt, "h_infinity_error_bound")
    bound = bt.h_infinity_error_bound()
    assert np.isfinite(bound) and bound >= 0

    assert not hasattr(mm, "h_infinity_error_bound"), (
        "KrylovROM should NOT expose a global a priori error bound -- "
        "moment matching genuinely has no such certificate, and this "
        "project should not pretend otherwise"
    )
