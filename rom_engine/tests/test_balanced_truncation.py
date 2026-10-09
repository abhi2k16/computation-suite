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
__author__ = "Abhijeet"
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


def test_result_is_insensitive_to_roundoff_level_input_noise():
    """Regression for the v1.0.1 fix. Before it, the Gramian square root's Cholesky-with-absolute-
    jitter ladder made BT bistable on this model: ~1 in 12 runs with only ulp-level input noise
    (and every run on some BLAS/LAPACK builds, e.g. Windows MKL) gave a 3e-2 sweep error instead of
    ~5e-4. With state equilibration the answer must not depend on round-off, so all noisy runs agree."""
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system()
    omega = np.linspace(1.0, 1500.0, 25)
    H_true = _true_H(fx, omega)
    rng = np.random.default_rng(7)
    errs = []
    for eps in (0.0, 1e-15, 1e-14, 1e-13, 1e-12, 1e-13, 1e-14, 1e-15, 1e-12, 1e-13):
        pert = lambda A: A * (1.0 + eps * rng.standard_normal(A.shape))
        rom = BalancedTruncationROM.from_MCK(pert(Mr), pert(Kr), Br, Coutr, C=pert(Cr), r=10)
        errs.append(np.max(np.abs(rom.frequency_response(omega) - H_true) / np.abs(H_true)))
    errs = np.array(errs)
    print(f"sweep error under round-off noise: min={errs.min():.3e} max={errs.max():.3e}")
    assert errs.max() < 5e-3
    assert errs.max() / errs.min() < 1.5, "result must not branch on round-off"


def test_state_equilibration_leaves_hankel_singular_values_unchanged():
    """The power-of-two similarity scaling inside hankel_singular_values() is a similarity
    transform: the leading Hankel singular values (the well-resolved ones) must equal an
    unscaled computation, and the returned transform must balance the ORIGINAL system."""
    from rom_engine.balanced_truncation import (hankel_singular_values, controllability_gramian,
                                                observability_gramian, _balance_from_gramians)
    from rom_engine.state_space import to_state_space
    fx, Kr, Mr, Cr, Br, Coutr, _ = _reduced_port_system(n_modes=8)
    ss = to_state_space(Mr, Kr, C=Cr, B=Br, Cout=Coutr, form="A")
    hsv, T, Tinv = hankel_singular_values(ss.A, ss.B, ss.Cout, return_transform=True)
    P = controllability_gramian(ss.A, ss.B)
    Q = observability_gramian(ss.A, ss.Cout)
    hsv_plain, _, _ = _balance_from_gramians(P, Q)
    assert np.allclose(hsv[:6], hsv_plain[:6], rtol=1e-5)          # well-resolved values agree
    # T maps back to the original state coordinates
    assert np.allclose(Tinv @ T, np.eye(T.shape[0]), atol=1e-6)
    Ab = Tinv @ ss.A @ T
    assert np.all(np.linalg.eigvals(Ab).real < 0)


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
