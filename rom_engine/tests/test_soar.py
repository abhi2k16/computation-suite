"""
test_soar.py -- validates rom_engine.soar's SOARROM against a small
synthetic system (exact moment-matching self-consistency) and a REAL,
damped fea_engine cantilever beam (docs/frequency_domain_rom_roadmap.md
Section 8), using the SAME fixture and port conventions test_krylov.py/
test_passivity.py already established for this "systems and control"
family.

Checks:
  1. THE core mathematical claim, proved directly on a small synthetic
     second-order system via a repeated-solve moment oracle (mirroring
     test_krylov.py's own _moments() helper, generalized to the
     second-order transfer function H(s)=Cout(s^2 M+sC+K)^-1 B): a
     SOARROM built from a k-column soar_basis() matches the full
     system's Taylor moments at s0 EXACTLY through moment k-1, then
     diverges -- exactly analogous to ordinary Arnoldi's own
     exact-through-moment-(k-1) property, and the property that a
     first, buggy version of this code (see soar.py's own module
     docstring) failed at moment 2 regardless of k, catching a real
     implementation bug before it shipped.
  2. On the real fea_engine fixture, with a genuinely BLOCK (multi-DOF)
     port, SOARROM(k) is confirmed to noticeably OUTPERFORM
     KrylovROM(2k) -- the SAME total reduced-state count, since
     KrylovROM's states are first-order ([q;q_dot]-sized) and SOARROM's
     are second-order (q-sized only) -- the actual "avoids the 2x
     blowup" claim this module exists to make, checked rather than
     assumed.
  3. Also on the real fixture, with a SINGLE-DOF port: soar_basis()'s
     own basis size is confirmed to plateau well below the requested k
     -- the genuine, literature-consistent SOAR fragility (single-vector
     starting vectors on a well-separated-eigenvalue structure converge
     like inverse power iteration toward the dominant mode) documented
     honestly in soar.py's own module docstring, not hidden by this
     test suite.
  4. is_stable(), checked not assumed -- same honest convention as
     KrylovROM.is_stable(): no outcome is asserted a priori, only that
     the check itself is self-consistent with directly inspecting the
     reduced quadratic pencil's own poles.
  5. THE actual motivation for this module: passivity.py's closed-form
     theorem says ANY second-order Galerkin projection of an
     SPD-M/PSD-C/SPD-K system preserves passivity. Checked directly,
     not just cited: SOARROM stays passive at every tested k (including
     the plateaued single-DOF-port case, k'=1) and at every tested s0,
     on the real fea_engine fixture -- a provable guarantee KrylovROM
     (a first-order reduction) does not have, per its own module
     docstring and per test_passivity.py's existing, honestly-reported
     8-passive/1-violation sweep of first-order reduction methods.
"""
import numpy as np
from scipy.linalg import eig, eigh
from rom_engine.soar import soar_basis, SOARROM
from rom_engine.krylov import KrylovROM
from rom_engine.passivity import velocity_transfer_function, passivity_margin, is_passive
import fea_fixtures as ff


# -----------------------------------------------------------------------
# 1. Exact moment matching (synthetic system, test oracle)
# -----------------------------------------------------------------------
def _moments_2nd_order(M, C, K, B, Cout, s0, n_moments):
    """Test oracle (not part of the library): the scalar/matrix Taylor-
    moment sequence of H(s)=Cout(s^2 M+sC+K)^-1 B around s0, via the
    same repeated-solve recursion soar.py's own module docstring
    derives (r_1=K0^-1 B, r_j=A1 r_{j-1}+A2 r_{j-2}), applied directly
    with dense solves rather than soar_basis()'s own machinery -- an
    independent re-derivation, not a call into the library under test."""
    K0 = K + s0 * C + s0 ** 2 * M
    D1 = C + 2 * s0 * M
    prev = np.linalg.solve(K0, B)
    prevprev = np.zeros_like(prev)
    out = [Cout @ prev]
    for _ in range(1, n_moments):
        raw = -np.linalg.solve(K0, D1 @ prev) - np.linalg.solve(K0, M @ prevprev)
        out.append(Cout @ raw)
        prevprev = prev
        prev = raw
    return np.array(out)


def _small_well_conditioned_second_order_system(seed=0, n=40):
    """A small, well-conditioned, genuinely non-modal-degenerate
    synthetic second-order system -- deliberately NOT fea_engine-
    derived, for the same reason test_krylov.py's own
    _small_well_conditioned_system() isn't: this check is about the
    EXACT moment-matching property, which needs a system without the
    real beam fixture's well-separated eigenvalues (see check 3 below,
    where that separation is instead the point)."""
    rng = np.random.default_rng(seed)
    diag = rng.uniform(1, 5, n)
    K = np.diag(diag) + 0.01 * rng.standard_normal((n, n))
    K = (K + K.T) / 2 + n * np.eye(n)          # SPD, well-conditioned
    M = np.eye(n) + 0.01 * rng.standard_normal((n, n))
    M = (M + M.T) / 2 + n * np.eye(n)          # SPD, well-conditioned
    C = np.diag(rng.uniform(0.05, 0.5, n))     # PSD, NOT proportional to M or K
    B = rng.standard_normal((n, 1))
    Cout = rng.standard_normal((1, n))
    return M, C, K, B, Cout


def test_soar_basis_matches_moments_exactly_through_k_minus_1():
    M, C, K, B, Cout = _small_well_conditioned_second_order_system()
    s0 = 0.0
    n_check = 10
    m_full = _moments_2nd_order(M, C, K, B, Cout, s0, n_check).flatten()

    for k in (3, 5, 7):
        V = soar_basis(M, K, B, k, C=C, s0=s0)
        assert V.shape[1] == k, "no deflation expected for this well-conditioned, non-degenerate system"
        Mr, Cr, Kr = V.T @ M @ V, V.T @ C @ V, V.T @ K @ V
        Br, Coutr = V.T @ B, Cout @ V
        m_red = _moments_2nd_order(Mr, Cr, Kr, Br, Coutr, s0, n_check).flatten()
        err = np.abs(m_red - m_full) / np.abs(m_full)
        print(f"k={k}: moment errors {np.array2string(err, precision=2)}")
        assert np.all(err[:k] < 1e-8), f"k={k} should match moments 0..{k-1} exactly"
        # "diverges starting at moment k" means a sharp (many-orders-of-
        # magnitude) JUMP relative to the still-exact moment k-1, not a
        # fixed absolute threshold -- matching test_krylov.py's own
        # two-sided test convention, since how far past machine precision
        # the very first missed moment lands can vary a bit by system
        assert err[k] > 1e3 * max(err[k - 1], 1e-16), (
            f"k={k} should show a sharp jump in error starting at moment {k}"
        )


def test_soarrom_transfer_function_matches_moments_at_nonzero_s0():
    """Same claim, but through the public SOARROM.from_MCK() API rather
    than soar_basis() + hand-rolled projection, and at a NONZERO
    (complex) expansion point -- confirms the class wiring itself
    (from_MCK's dtype handling, transfer_function()) is correct, not
    just the underlying basis-building function."""
    M, C, K, B, Cout = _small_well_conditioned_second_order_system(seed=1)
    s0 = 2.0 + 1.0j
    n_check = 6
    m_full = _moments_2nd_order(M, C, K, B, Cout, s0, n_check).flatten()

    k = 4
    rom = SOARROM.from_MCK(M, K, B, Cout, C=C, s0=s0, k=k)
    assert rom.V.shape[1] == k
    m_red = _moments_2nd_order(rom.M_r, rom.C_r, rom.K_r, rom.B_r, rom.Cout_r, s0, n_check).flatten()
    err = np.abs(m_red - m_full) / np.abs(m_full)
    print(f"nonzero s0: moment errors {np.array2string(err, precision=2)}")
    assert np.all(err[:k] < 1e-8)
    assert err[k] > 1e3 * max(err[k - 1], 1e-16), f"should show a sharp jump in error starting at moment {k}"


# -----------------------------------------------------------------------
# 2/3/4. Real fea_engine fixture
# -----------------------------------------------------------------------
def _fixture(n=25, alpha=2.0, beta=1e-5):
    fx = ff.damped_cantilever_beam_system(n=n, alpha=alpha, beta=beta)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]
    return fx, free, Kff, Mff, Cff


def _true_H_block(fx, free, dof_idx, omega_array):
    """True H(i*omega) (n_omega, n_out, n_in) for a multi-DOF collocated
    port, from fea_engine's own solve_harmonic() at each input column in
    turn -- the same ground-truth convention test_krylov.py's own
    _true_H() uses, generalized to a block port."""
    n_dof = fx["n_dof"]
    sysobj = fx["sys"]
    n_port = len(dof_idx)
    H = np.empty((len(omega_array), n_port, n_port), dtype=complex)
    for i, om in enumerate(omega_array):
        for j, d in enumerate(dof_idx):
            F_full = np.zeros(n_dof)
            F_full[free[d]] = 1.0
            x = sysobj.solve_harmonic(om, F_full)
            for o, do in enumerate(dof_idx):
                H[i, o, j] = x[free[do]]
    return H


def _true_H_siso(fx, free, dof, omega_array):
    n_dof = fx["n_dof"]
    tip_dof_global = free[dof]
    F_full = np.zeros(n_dof)
    F_full[tip_dof_global] = 1.0
    sysobj = fx["sys"]
    return np.array([sysobj.solve_harmonic(om, F_full)[tip_dof_global] for om in omega_array])


def test_block_port_soarrom_outperforms_krylovrom_at_same_state_count():
    fx, free, Kff, Mff, Cff = _fixture()
    n_free = len(free)
    dof_idx = np.array([2, n_free // 2, n_free - 2])   # 3 spread-out free DOFs
    B = np.zeros((n_free, 3))
    for i, d in enumerate(dof_idx):
        B[d, i] = 1.0
    Cout = B.T.copy()   # collocated multi-port

    omega_near = np.linspace(1.0, 50.0, 4)
    H_true = _true_H_block(fx, free, dof_idx, omega_near)

    k = 2   # small order, where the 2x-states-for-free-with-Krylov gap is starkest
    soar_rom = SOARROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=0.0, k=k)
    kry_rom = KrylovROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=0.0, k=2 * k)
    assert soar_rom.V.shape[1] == k, "no deflation expected yet at this small k for a 3-column block"

    H_soar = soar_rom.frequency_response(omega_near)
    H_kry = kry_rom.frequency_response(omega_near)
    err_soar = np.max(np.abs(H_soar - H_true) / np.abs(H_true))
    err_kry = np.max(np.abs(H_kry - H_true) / np.abs(H_true))
    print(f"k={k} (SOAR {k} second-order states vs Krylov {2*k} first-order states -- "
          f"same total state count): err_soar={err_soar:.3e}, err_kry={err_kry:.3e}")
    assert err_soar < err_kry, (
        "at the SAME total reduced-state count, SOARROM should measurably "
        "outperform KrylovROM at small k -- the actual 'avoids the "
        "first-order 2x blowup' claim this module exists to make"
    )


def test_single_dof_port_basis_plateaus_below_requested_k():
    """The honest, documented SOAR fragility (soar.py's own module
    docstring): for a single-vector port on this well-separated-
    eigenvalue beam, the raw recursion converges like inverse power
    iteration toward the dominant mode, and soar_basis()'s actual size
    plateaus well below whatever k is requested -- NOT a bug (checked
    separately above that moment-matching itself is exact up to
    whatever size IS reached), a genuine numerical property, confirmed
    directly rather than glossed over."""
    fx, free, Kff, Mff, Cff = _fixture()
    n_free = len(free)
    tip_local = n_free - 2
    B = np.zeros((n_free, 1))
    B[tip_local, 0] = 1.0

    eigvals, _ = eigh(Kff, Mff)
    print(f"omega2/omega1 = {np.sqrt(eigvals[1] / eigvals[0]):.3f} (eigenvalue separation "
          f"driving the power-iteration-like collapse)")

    sizes = [soar_basis(Mff, Kff, B, k, C=Cff, s0=0.0).shape[1] for k in (4, 8, 16, 24)]
    print(f"requested k=(4,8,16,24) -> actual basis size {sizes}")
    assert sizes[-1] < 24, "single-DOF port on a well-separated-eigenvalue beam should plateau, not reach k=24"
    assert sizes == sorted(sizes), "actual size should never DECREASE as more is requested"


def test_is_stable_is_self_consistent_on_real_fixture():
    fx, free, Kff, Mff, Cff = _fixture()
    n_free = len(free)
    dof_idx = np.array([2, n_free // 2, n_free - 2])
    B = np.zeros((n_free, 3))
    for i, d in enumerate(dof_idx):
        B[d, i] = 1.0
    Cout = B.T.copy()

    for k in (2, 4, 6):
        rom = SOARROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=0.0, k=k)
        kk = rom.M_r.shape[0]
        Zero = np.zeros((kk, kk))
        I = np.eye(kk)
        A = np.block([[Zero, I], [-rom.K_r, -rom.C_r]])
        E = np.block([[I, Zero], [Zero, rom.M_r]])
        poles = eig(A, E, right=False)
        poles = poles[np.isfinite(poles)]
        expected = bool(np.all(poles.real < 0))
        assert rom.is_stable() == expected, (
            "is_stable() must agree with directly inspecting the reduced "
            "quadratic pencil's own poles -- no hidden logic beyond that check"
        )
        print(f"k={k} (actual size {kk}): is_stable()={rom.is_stable()}")


# -----------------------------------------------------------------------
# 5. The actual motivation: passivity
# -----------------------------------------------------------------------
def test_soarrom_stays_passive_across_orders_and_expansion_points():
    fx, free, Kff, Mff, Cff = _fixture()
    n_free = len(free)
    tip_local = n_free - 2
    B = np.zeros((n_free, 1))
    B[tip_local, 0] = 1.0
    Cout = B.T.copy()   # collocated force-in/velocity-out port, per passivity.py's own convention

    eigvals, _ = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    omega_sweep = np.linspace(0.05 * omega1, 3.0 * omega1, 200)   # includes resonance

    for s0 in (0.0, 0.5 * omega1, 1.5 * omega1):
        for k in (1, 4, 8, 16):
            rom = SOARROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=s0, k=k)
            H_disp = rom.frequency_response(omega_sweep)
            H_vel = velocity_transfer_function(H_disp, omega_sweep)
            margin = passivity_margin(H_vel)
            print(f"s0={s0:.1f}, k={k} (actual size {rom.V.shape[1]}): "
                  f"passive={is_passive(H_vel)}, min margin={margin.min():.3e}")
            assert is_passive(H_vel), (
                "SOARROM is a second-order Galerkin projection (M_r=V^T M V, "
                "...) of an SPD-M/PSD-C/SPD-K system, so passivity.py's own "
                "closed-form theorem says it MUST stay passive for ANY basis "
                "V, at ANY order, ANY s0 -- this is the actual point of "
                "building this module (see soar.py's own docstring); a "
                "violation here would mean the theorem's premise (M_r,K_r "
                "SPD, C_r PSD by congruence) is somehow not actually met by "
                "this projection, a real bug."
            )


def test_soarrom_projected_matrices_stay_spd_psd():
    """A more direct, structural check of the SAME theorem's premise
    (rather than only its frequency-domain consequence above): M_r, K_r
    must stay SPD and C_r must stay PSD for ANY basis V, by congruence
    -- checked directly via eigenvalues, on the real fixture, at several
    k including the plateaued single-DOF-port case."""
    fx, free, Kff, Mff, Cff = _fixture()
    n_free = len(free)
    tip_local = n_free - 2
    B = np.zeros((n_free, 1))
    B[tip_local, 0] = 1.0
    Cout = B.T.copy()

    for k in (1, 4, 8, 16):
        rom = SOARROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=0.0, k=k)
        eig_M = np.linalg.eigvalsh(rom.M_r)
        eig_K = np.linalg.eigvalsh(rom.K_r)
        eig_C = np.linalg.eigvalsh(rom.C_r)
        print(f"k={k} (actual size {rom.V.shape[1]}): min eig M_r={eig_M.min():.3e}, "
              f"K_r={eig_K.min():.3e}, C_r={eig_C.min():.3e}")
        assert np.all(eig_M > -1e-10 * max(abs(eig_M).max(), 1.0))
        assert np.all(eig_K > -1e-10 * max(abs(eig_K).max(), 1.0))
        assert np.all(eig_C > -1e-10 * max(abs(eig_C).max(), 1.0))
