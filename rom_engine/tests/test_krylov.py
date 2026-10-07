"""
test_krylov.py -- validates rom_engine.krylov.KrylovROM against a REAL,
damped fea_engine cantilever beam (fea_fixtures.damped_cantilever_beam_system()),
using fea_engine's own solve_harmonic() (already-validated, already-shipped
functionality) as ground truth throughout -- exactly the same fixture and
ground-truth convention test_frequency.py already uses for FrequencyROM,
so the two ROM families are directly, fairly comparable.

Input/output port: unit force in at the tip transverse DOF, tip
transverse displacement read out -- a genuine SISO port (not the whole
state), matching the "port-dependent accuracy" property Krylov moment
matching and balanced truncation are both specifically about, and that
mode displacement/frequency.FrequencyROM (which see the WHOLE response,
not a compressed port) are not.

Checks (mirroring docs/classical_mor_roadmap.md Section 6):
  1. KrylovROM.frequency_response() matches the true H(i*omega) (from
     fea_engine's own solve_harmonic(), the tip-DOF component divided by
     the unit input amplitude) EXTREMELY closely near the expansion
     point s0 -- the defining property of moment matching.
  2. Accuracy measurably DEGRADES far from s0 at the SAME reduced order
     -- also the defining, EXPECTED property (not a bug), explicitly
     checked rather than glossed over.
  3. Accuracy improves as k (reduced order) grows, at a FIXED evaluation
     point.
  4. is_stable() runs and returns an honest, checkable answer; the test
     does not assert a specific stable/unstable outcome (this project
     cannot promise Krylov moment matching preserves stability -- that
     is precisely the point of the method, see module docstring), only
     that the check itself is meaningful (poles are finite, and the
     answer is self-consistent with directly inspecting the reduced
     pencil's own eigenvalues).

Two-sided (Petrov-Galerkin) extension checks (docs/classical_mor_roadmap.md
Section 7, Phase 3):
  5. THE central mathematical claim, proved directly rather than taken
     on faith: on a small, well-conditioned synthetic system (so the
     comparison isn't muddied by the real beam fixture's own numerical
     conditioning -- see balanced_truncation.py's module docstring for
     why that matters), a one-sided basis of size k matches the full
     system's transfer-function moments around s0 EXACTLY up through
     moment k-1 and then diverges sharply; a two-sided basis of the SAME
     size k matches EXACTLY through moment 2k-1 -- twice as many.
  6. Biorthogonality: the two_sided_arnoldi_bases() pair satisfies
     W^H E V = I to near machine precision on the real fea_engine
     fixture, not just in principle.
  7. two_sided_arnoldi_bases() actually raises ValueError (the
     documented fragility, not silently returning nonsense) when its
     conditioning threshold is exceeded.
  8. On the real fea_engine fixture, two_sided=True measurably
     outperforms one-sided at the SAME reduced order k, for at least one
     k where this is a large, unambiguous effect -- the "fuller
     comparison" this extension exists to enable. (Note, found while
     writing this test and worth recording honestly: the SIZE of this
     improvement varies noticeably with k for this particular beam/port
     -- large at k=3 and k=5, negligible at k=4 and k=6 -- plausibly
     because whether an additional matched moment actually helps depends
     on that moment's specific contribution to THIS port's transfer
     function, not a universal guarantee that every extra k always
     halves the error.)
"""
import numpy as np
from scipy.linalg import eig, eigh
from rom_engine.krylov import KrylovROM, arnoldi_basis, two_sided_arnoldi_bases
import fea_fixtures as ff


def _build_port(fx):
    """Unit tip-force-in / tip-displacement-out port, on the FREE-dof
    numbering (matching test_frequency.py's own tip-dof convention:
    tip transverse dof is free[-2])."""
    free = fx["free_dofs"]
    n_free = len(free)
    tip_local = n_free - 2
    B = np.zeros((n_free, 1)); B[tip_local, 0] = 1.0
    Cout = np.zeros((1, n_free)); Cout[0, tip_local] = 1.0
    return B, Cout, tip_local


def _true_H(fx, omega_array):
    """True H(i*omega) at the tip port, from fea_engine's own
    solve_harmonic() -- since a unit force at the tip DOF was applied,
    the tip-DOF component of the resulting complex displacement IS
    H(i*omega) directly (no further scaling needed)."""
    free = fx["free_dofs"]
    n_dof = fx["n_dof"]
    tip_dof_global = n_dof - 2
    F_full = np.zeros(n_dof)
    F_full[tip_dof_global] = 1.0
    sysobj = fx["sys"]
    H = np.array([sysobj.solve_harmonic(om, F_full)[tip_dof_global] for om in omega_array])
    return H


def test_accurate_near_s0_degrades_away_from_it():
    fx = ff.damped_cantilever_beam_system(n=25)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]
    B, Cout, _ = _build_port(fx)

    s0 = 0.0
    rom = KrylovROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=s0, k=12)

    # near s0: low frequencies
    omega_near = np.linspace(1.0, 50.0, 5)
    H_rom_near = rom.frequency_response(omega_near)
    H_true_near = _true_H(fx, omega_near)
    err_near = np.abs(H_rom_near - H_true_near) / np.abs(H_true_near)

    # far from s0: near/above the higher structural resonances
    eigvals, _ = eigh(Kff, Mff)
    omega_high = np.linspace(0.9, 1.3, 5) * np.sqrt(eigvals[-1])
    H_rom_far = rom.frequency_response(omega_high)
    H_true_far = _true_H(fx, omega_high)
    err_far = np.abs(H_rom_far - H_true_far) / np.abs(H_true_far)

    print(f"KrylovROM (k=12, s0=0) relative error near s0: max={err_near.max():.3e}, "
          f"far from s0 (near highest mode): max={err_far.max():.3e}")
    assert err_near.max() < 1e-6, "moment matching should be extremely accurate AT/NEAR s0"
    assert err_far.max() > err_near.max(), (
        "accuracy should measurably degrade far from s0 -- this is the "
        "defining signature of a LOCAL approximation method, not a bug"
    )


def test_accuracy_improves_with_larger_k():
    fx = ff.damped_cantilever_beam_system(n=25)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]
    B, Cout, _ = _build_port(fx)

    omega_probe = np.array([200.0])   # a fixed, moderately-far-from-0 point
    errs = []
    for k in (4, 8, 16):
        rom = KrylovROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=0.0, k=k)
        H_rom = rom.frequency_response(omega_probe)
        H_true = _true_H(fx, omega_probe)
        err = float((np.abs(H_rom - H_true) / np.abs(H_true))[0])
        errs.append(err)
    print(f"KrylovROM relative error at omega=200 vs k=(4,8,16): {errs}")
    assert errs[0] >= errs[1] >= errs[2], "error should not increase as more moments are matched"
    assert errs[2] < errs[0]


def test_is_stable_is_self_consistent():
    fx = ff.damped_cantilever_beam_system(n=25)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]
    B, Cout, _ = _build_port(fx)

    for k in (4, 12, 24):
        rom = KrylovROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=0.0, k=k)
        poles = eig(rom.A_r, rom.E_r, right=False)
        poles = poles[np.isfinite(poles)]
        assert len(poles) > 0
        expected = bool(np.all(poles.real < 0))
        assert rom.is_stable() == expected, (
            "is_stable() must agree with directly inspecting the reduced "
            "pencil's own eigenvalues -- no hidden logic beyond that check"
        )
        print(f"k={k}: is_stable()={rom.is_stable()} "
              f"(max real part of reduced poles = {poles.real.max():.4g})")


# -----------------------------------------------------------------------
# Two-sided (Petrov-Galerkin) extension
# -----------------------------------------------------------------------
def _moments(A, E, B, Cout, s0, n_moments):
    """The scalar Taylor-moment sequence of H(s)=Cout(sE-A)^-1 B around
    s0, via the same repeated-solve recursion the module docstring
    describes -- used here purely as a TEST oracle (not part of the
    library), on small well-conditioned systems where its own numerical
    error stays far below the effect being measured."""
    As0 = A - s0 * E
    term = np.linalg.solve(As0, B)
    out = []
    for _ in range(n_moments):
        out.append((Cout @ term)[0, 0])
        term = np.linalg.solve(As0, E @ term)
    return np.array(out)


def _small_well_conditioned_system(seed=0, n=40):
    """A small, well-conditioned (NOT fea_engine-derived) synthetic
    stable SISO system -- deliberately used ONLY for the exact-moment-
    count proof below, where the real beam fixture's own wide
    eigenvalue spread (see balanced_truncation.py's module docstring)
    would swamp the effect being measured with unrelated numerical
    noise. The real fea_engine fixture is still used for every other
    check in this file (biorthogonality, ill-conditioning, and the
    actual frequency-response comparison)."""
    rng = np.random.default_rng(seed)
    diag = rng.uniform(1, 5, n)
    skew = rng.standard_normal((n, n)) * 0.05
    skew = skew - skew.T
    A = -np.diag(diag) + skew
    E = np.eye(n)
    B = rng.standard_normal((n, 1))
    Cout = rng.standard_normal((1, n))
    return A, E, B, Cout


def test_two_sided_matches_twice_as_many_moments_as_one_sided():
    A, E, B, Cout = _small_well_conditioned_system()
    n_check = 14
    m_full = _moments(A, E, B, Cout, 0.0, n_check)

    for k in (3, 5):
        V = arnoldi_basis(A, E, B, k, s0=0.0)
        Ar, Er, Br, Cr = V.conj().T @ A @ V, V.conj().T @ E @ V, V.conj().T @ B, Cout @ V
        m_one = _moments(Ar, Er, Br, Cr, 0.0, n_check)

        Vt, Wt = two_sided_arnoldi_bases(A, E, B, Cout, k, s0=0.0)
        Ar2 = Wt.conj().T @ A @ Vt
        Er2 = Wt.conj().T @ E @ Vt
        Br2 = Wt.conj().T @ B
        Cr2 = Cout @ Vt
        m_two = _moments(Ar2, Er2, Br2, Cr2, 0.0, n_check)

        err_one = np.abs(m_one - m_full) / np.abs(m_full)
        err_two = np.abs(m_two - m_full) / np.abs(m_full)
        print(f"k={k}: one-sided moment errors {np.array2string(err_one, precision=2)}")
        print(f"k={k}: two-sided moment errors {np.array2string(err_two, precision=2)}")

        # one-sided: EXACT through moment k-1, then diverges
        assert np.all(err_one[:k] < 1e-8), f"one-sided (k={k}) should match moments 0..{k-1} exactly"
        assert err_one[k] > 1e-4, f"one-sided (k={k}) should visibly diverge starting at moment {k}"
        # two-sided: EXACT through moment 2k-1 (twice as many), then a
        # sharp (many-orders-of-magnitude) jump at moment 2k -- the jump
        # size, not a fixed absolute threshold, is what distinguishes
        # "still matching" from "starting to diverge", since the actual
        # post-divergence error grows gradually from there rather than
        # snapping instantly to some fixed magnitude
        assert np.all(err_two[:2 * k] < 1e-8), f"two-sided (k={k}) should match moments 0..{2*k-1} exactly"
        assert err_two[2 * k] > 1e3 * max(err_two[2 * k - 1], 1e-16), (
            f"two-sided (k={k}) should show a sharp jump in error starting at moment {2*k}"
        )


def test_two_sided_biorthogonality_on_real_fixture():
    fx = ff.damped_cantilever_beam_system(n=25)
    Kff = fx["K"][np.ix_(fx["free_dofs"], fx["free_dofs"])]
    Mff = fx["M"][np.ix_(fx["free_dofs"], fx["free_dofs"])]
    Cff = fx["C"][np.ix_(fx["free_dofs"], fx["free_dofs"])]
    B, Cout, _ = _build_port(fx)

    for k in (4, 8, 12):
        rom = KrylovROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=0.0, k=k, two_sided=True)
        assert rom.two_sided
        G = rom.W.conj().T @ rom.ss.E @ rom.V
        err = np.max(np.abs(G - np.eye(G.shape[0])))
        print(f"k={k}: biorthogonality (W^H E V vs I) max error = {err:.3e}")
        assert err < 1e-8


def test_two_sided_raises_on_ill_conditioned_biorthogonalization():
    fx = ff.damped_cantilever_beam_system(n=25)
    Kff = fx["K"][np.ix_(fx["free_dofs"], fx["free_dofs"])]
    Mff = fx["M"][np.ix_(fx["free_dofs"], fx["free_dofs"])]
    Cff = fx["C"][np.ix_(fx["free_dofs"], fx["free_dofs"])]
    B, Cout, _ = _build_port(fx)

    # a deliberately unreasonable conditioning threshold (>=1 always
    # holds for any nontrivial matrix) exercises the exact same failure
    # path a genuinely near-degenerate input/output Krylov-subspace pair
    # would hit -- this tests the mechanism directly and reliably rather
    # than trying to engineer a naturally-occurring degenerate case.
    from rom_engine.state_space import to_state_space
    ss = to_state_space(Mff, Kff, C=Cff, B=B, Cout=Cout, form="E")
    try:
        two_sided_arnoldi_bases(ss.A, ss.E, ss.B, ss.Cout, k=6, s0=0.0, cond_tol=1.0)
        raised = False
    except ValueError as e:
        raised = True
        assert "ill-conditioned" in str(e)
    assert raised, "an unreasonably strict cond_tol should trigger the documented ValueError"


def test_two_sided_outperforms_one_sided_at_same_k_on_real_fixture():
    fx = ff.damped_cantilever_beam_system(n=25)
    Kff = fx["K"][np.ix_(fx["free_dofs"], fx["free_dofs"])]
    Mff = fx["M"][np.ix_(fx["free_dofs"], fx["free_dofs"])]
    Cff = fx["C"][np.ix_(fx["free_dofs"], fx["free_dofs"])]
    B, Cout, _ = _build_port(fx)

    k = 5   # a k where this beam/port shows a large, unambiguous
            # improvement -- see module docstring note on why this
            # varies by k
    one = KrylovROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=0.0, k=k, two_sided=False)
    two = KrylovROM.from_MCK(Mff, Kff, B, Cout, C=Cff, s0=0.0, k=k, two_sided=True)

    omega_far = np.linspace(500.0, 3000.0, 10)
    H_true = _true_H(fx, omega_far)
    H_one = one.frequency_response(omega_far)
    H_two = two.frequency_response(omega_far)
    err_one = np.mean(np.abs(H_one - H_true) / np.abs(H_true))
    err_two = np.mean(np.abs(H_two - H_true) / np.abs(H_true))

    print(f"k={k}: mean relative error far from s0 -- one-sided={err_one:.3e}, "
          f"two-sided={err_two:.3e} ({err_one / err_two:.1f}x better)")
    assert err_two < err_one, (
        "at k=5 on this fixture, two-sided should clearly outperform "
        "one-sided at the same reduced order -- the 'fuller comparison' "
        "this extension exists to enable"
    )
