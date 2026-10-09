"""
test_frequency.py -- validates rom_engine.frequency.FrequencyROM and
build_pod_basis_from_frf_snapshots() against a REAL, damped fea_engine
cantilever beam (fea_fixtures.damped_cantilever_beam_system()), using
fea_engine's own solve_harmonic()/solve_frequency_sweep() (already
validated, already-shipped fea_engine functionality -- not hand-rolled
here) as ground truth throughout.

Checks (mirroring docs/frequency_domain_rom_roadmap.md Section 6):
  1. Correctness both NEAR a resonance (where A(omega) is closest to
     singular -- the hard case) and OFF resonance (the easy case) --
     both reported, not just the easy one.
  2. Accuracy improves as the modal basis rank grows.
  3. A POD-on-FRF-snapshots basis is a genuine alternative to a modal
     basis -- built and checked on its own terms, not assumed to win.
  4. The dtype fix from affine.py/_as_array can't silently regress:
     a general (explicit-C) FrequencyROM's response has a nonzero,
     correct imaginary/damping-driven component, checked directly
     against fea_engine's own complex solve.
  5. The proportional-damping 2-term collapse (rayleigh=(alpha,beta))
     is algebraically identical to the general 3-term ({M,C,K})
     decomposition, checked to near machine precision.
  6. A frequency sweep via the reduced model is substantially faster
     than one via fea_engine's own full-order solve_frequency_sweep().
"""
__author__ = "Abhijeet"
import numpy as np
from scipy.linalg import eigh
from rom_engine import FrequencyROM, build_pod_basis_from_frf_snapshots
import fea_fixtures as ff


def _modal_basis(Kff, Mff, n_modes):
    """The first n_modes undamped mode shapes, M-orthonormal (scipy's
    eigh convention) -- FrequencyROM's "modal basis" option."""
    eigvals, eigvecs = eigh(Kff, Mff)
    return eigvecs[:, :n_modes]


def test_correctness_near_and_off_resonance():
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    sysobj = fx["sys"]

    eigvals, _ = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))   # first undamped natural frequency

    basis = _modal_basis(Kff, Mff, n_modes=12)
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))

    n_dof = fx["n_dof"]
    tip_dof = n_dof - 2   # tip transverse dof, full numbering
    F_full = np.zeros(n_dof)
    F_full[tip_dof] = 1000.0
    F_free = F_full[free]

    for label, omega in [("near resonance", omega1), ("off resonance (0.3*omega1)", 0.3 * omega1)]:
        x_rom_free = rom.frequency_response([omega], F_free)[0]
        U_true = sysobj.solve_harmonic(omega, F_full)
        x_true_free = U_true[free]
        err = np.max(np.abs(x_rom_free - x_true_free)) / np.max(np.abs(x_true_free))
        print(f"{label}: omega={omega:.2f} rad/s, max relative error = {err:.3e}")
        assert err < 5e-3, f"{label}: ROM should track fea_engine's own solve_harmonic() closely"
    print("PASS -- FrequencyROM matches fea_engine's own harmonic solve both near and off resonance")


def test_modal_basis_accuracy_improves_with_rank():
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    sysobj = fx["sys"]
    n_dof = fx["n_dof"]

    eigvals, _ = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    omega_probe = 1.5 * omega1   # a frequency needing more than just mode 1 to represent well

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]
    U_true_free = sysobj.solve_harmonic(omega_probe, F_full)[free]

    ranks = [2, 4, 8, 16]
    errs = []
    for r in ranks:
        basis = _modal_basis(Kff, Mff, n_modes=r)
        rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))
        x_rom = rom.frequency_response([omega_probe], F_free)[0]
        err = np.max(np.abs(x_rom - U_true_free)) / np.max(np.abs(U_true_free))
        errs.append(err)
        print(f"rank={r:3d}  relative error = {err:.3e}")

    assert errs[-1] < errs[0], "accuracy should improve (error shrink) as basis rank grows"
    assert errs[-1] < 1e-3, "at rank=16 (of 42 free dof), error should be small"
    print("PASS -- modal-basis FrequencyROM accuracy improves monotonically-ish with rank")


def test_pod_basis_from_frf_snapshots_is_a_real_alternative():
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]
    sysobj = fx["sys"]
    n_dof = fx["n_dof"]

    eigvals, _ = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    # train the POD basis on a handful of frequencies spanning [0.2, 2.5]*omega1
    training_omegas = np.linspace(0.2, 2.5, 8) * omega1
    pod_basis = build_pod_basis_from_frf_snapshots(
        training_omegas, Mff, Kff, F_free, C=Cff, n_modes=8)
    print(f"POD-on-FRF-snapshots basis: {pod_basis.n_modes} modes, "
          f"energy captured = {pod_basis.energy_captured():.6f}")

    rom_pod = FrequencyROM.from_MCK(Mff, Kff, pod_basis.V, C=Cff)
    modal_basis = _modal_basis(Kff, Mff, n_modes=8)
    rom_modal = FrequencyROM.from_MCK(Mff, Kff, modal_basis, C=Cff)

    # test at a HELD-OUT frequency (not one of the training points)
    omega_test = 0.83 * omega1
    U_true_free = sysobj.solve_harmonic(omega_test, F_full)[free]

    x_pod = rom_pod.frequency_response([omega_test], F_free)[0]
    x_modal = rom_modal.frequency_response([omega_test], F_free)[0]
    err_pod = np.max(np.abs(x_pod - U_true_free)) / np.max(np.abs(U_true_free))
    err_modal = np.max(np.abs(x_modal - U_true_free)) / np.max(np.abs(U_true_free))
    print(f"same rank (8), held-out omega={omega_test:.2f}: "
          f"POD-basis relative error = {err_pod:.3e}, modal-basis relative error = {err_modal:.3e}")

    # both should be small (an "empirical comparison, not an assumed winner" per the roadmap --
    # we assert each is individually accurate, not that one beats the other)
    assert err_pod < 1e-3, "POD-on-FRF-snapshots basis should accurately predict a held-out frequency"
    assert err_modal < 1e-3, "modal basis should also be accurate here, as a sanity cross-check"
    print("PASS -- POD-on-FRF-snapshots basis is a working, accurate alternative to a modal basis")


def test_general_damping_matches_fea_engine_complex_solve():
    fx = ff.damped_cantilever_beam_system(n=20, alpha=5.0, beta=2e-5)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]
    sysobj = fx["sys"]
    n_dof = fx["n_dof"]

    basis = _modal_basis(Kff, Mff, n_modes=14)
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, C=Cff)   # general (explicit C) path

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    eigvals, _ = eigh(Kff, Mff)
    omega_test = 1.1 * float(np.sqrt(eigvals[0]))
    x_rom = rom.frequency_response([omega_test], F_free)[0]
    U_true_free = sysobj.solve_harmonic(omega_test, F_full)[free]

    assert np.max(np.abs(x_rom.imag)) > 1e-6 * np.max(np.abs(x_rom)), (
        "a damped response at a frequency near resonance MUST have a substantial "
        "imaginary (phase-lag) component -- if this is ~0, the i*omega*C term is "
        "being silently dropped (exactly the affine._as_array dtype bug the "
        "roadmap flagged)")
    err = np.max(np.abs(x_rom - U_true_free)) / np.max(np.abs(U_true_free))
    print(f"general (explicit-C) FrequencyROM vs fea_engine solve_harmonic(): "
          f"relative error = {err:.3e}, max|Im(x_rom)|/max|x_rom| = "
          f"{np.max(np.abs(x_rom.imag)) / np.max(np.abs(x_rom)):.4f}")
    assert err < 5e-3
    print("PASS -- general-damping FrequencyROM correctly reproduces the complex "
          "(magnitude AND phase) response, confirming the dtype fix holds")


def test_proportional_damping_collapse_matches_general_form():
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]   # = alpha*Mff + beta*Kff, from fea_engine's own assemble_damping

    basis = _modal_basis(Kff, Mff, n_modes=10)
    rom_general = FrequencyROM.from_MCK(Mff, Kff, basis, C=Cff)
    rom_collapsed = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))

    eigvals, _ = eigh(Kff, Mff)
    rng_omegas = np.linspace(0.3, 3.0, 6) * float(np.sqrt(eigvals[0]))
    max_diff = 0.0
    for omega in rng_omegas:
        A_general = rom_general.affine.assemble(omega)
        A_collapsed = rom_collapsed.affine.assemble(omega)
        diff = np.max(np.abs(A_general - A_collapsed))
        max_diff = max(max_diff, diff)
    print(f"max abs diff, general 3-term {{M,C,K}} vs collapsed 2-term {{M,K}} "
          f"assembly across {len(rng_omegas)} frequencies: {max_diff:.3e}")
    assert max_diff < 1e-6, (
        "the proportional-damping collapse (Section 2a of the roadmap) must be "
        "algebraically IDENTICAL to the general 3-term form, not just similar")
    print("PASS -- proportional-damping 2-term collapse matches the general 3-term form exactly")


def test_reduced_sweep_faster_than_fea_engine_full_sweep():
    import time
    fx = ff.damped_cantilever_beam_system(n=200)   # a bigger model so the full sweep is slow enough to time meaningfully
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    sysobj = fx["sys"]
    n_dof = fx["n_dof"]

    basis = _modal_basis(Kff, Mff, n_modes=15)
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    eigvals, _ = eigh(Kff, Mff)
    omegas = np.linspace(0.2, 3.0, 100) * float(np.sqrt(eigvals[0]))

    t0 = time.perf_counter()
    rom.frequency_response(omegas, F_free)
    t_rom = time.perf_counter() - t0

    t0 = time.perf_counter()
    sysobj.solve_frequency_sweep(omegas, F_full)
    t_full = time.perf_counter() - t0

    speedup = t_full / max(t_rom, 1e-12)
    print(f"{len(omegas)}-point sweep: fea_engine full solve_frequency_sweep() = "
          f"{t_full*1e3:.2f} ms, FrequencyROM.frequency_response() = {t_rom*1e3:.2f} ms, "
          f"speedup = {speedup:.1f}x")
    assert speedup > 3, "the reduced sweep should be substantially faster on a model this size"
    print("PASS -- reduced-order frequency sweep is substantially faster than fea_engine's full sweep")


def test_residual_norm_matches_independent_full_assembly():
    """Phase 4a regression check: residual_norm() computes A(omega)@x
    via AffineDecomposition.assemble_action() (O(Q*n_dof*n_modes)) --
    this must match an INDEPENDENTLY computed residual built the slow
    way (via the full assemble() this package's own error_estimate()
    used before Phase 4a), confirming the optimization changed only
    the cost, not the value."""
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]

    basis = _modal_basis(Kff, Mff, n_modes=10)
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))

    n_dof = fx["n_dof"]
    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    eigvals, _ = eigh(Kff, Mff)
    omega_test = 1.3 * float(np.sqrt(eigvals[0]))

    fast = rom.residual_norm(omega_test, F_free)

    # independently rebuild the residual the SLOW way, deliberately not
    # reusing residual_norm()'s own code path
    F_r = rom.galerkin.project_vector(F_free)
    q = rom.solve(omega_test, F_r)
    x = rom.galerkin.expand(q)
    A_full = rom.affine.assemble(omega_test)
    slow = float(np.linalg.norm(F_free - A_full @ x))

    rel_diff = abs(fast - slow) / max(slow, 1e-30)
    print(f"residual_norm() (fast, assemble_action) = {fast:.6f}, "
          f"independent slow (full assemble()) = {slow:.6f}, relative diff = {rel_diff:.3e}")
    assert rel_diff < 1e-6, "the efficient residual must match the slow, independently computed one"
    print("PASS -- Phase 4a's efficient residual matches an independent full-assembly computation")

    # also confirm error_estimate() (the pre-existing public name) now
    # delegates to residual_norm() and gives the identical value
    assert rom.error_estimate(omega_test, F_free) == fast
    print("PASS -- error_estimate() delegates to residual_norm() exactly")


def test_hierarchical_indicator_correlates_with_true_error():
    """The hierarchical indicator (comparing a small-basis ROM against
    a genuinely richer one) should be LARGE where the small ROM is
    actually inaccurate (near resonance, at low rank) and SMALL where
    it's already accurate (off resonance, or at a rank that's already
    converged) -- checked directly against fea_engine's own
    solve_harmonic() ground truth, not just asserted to look sensible."""
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    sysobj = fx["sys"]
    n_dof = fx["n_dof"]

    eigvals, _ = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    small_basis = _modal_basis(Kff, Mff, n_modes=2)   # spans modes 1-2 only
    richer_basis = _modal_basis(Kff, Mff, n_modes=16)
    rom_small = FrequencyROM.from_MCK(Mff, Kff, small_basis, rayleigh=(fx["alpha"], fx["beta"]))
    rom_richer = FrequencyROM.from_MCK(Mff, Kff, richer_basis, rayleigh=(fx["alpha"], fx["beta"]))

    # "good": right AT the 1st resonance -- mode 1 is IN the rank-2 basis,
    # so this should be represented well despite being a resonance.
    # "bad": AT the 3rd resonance -- mode 3 is NOT in the rank-2 basis
    # (which only spans modes 1-2), so this should be represented badly.
    # Both are resonances, deliberately -- this isolates "is the relevant
    # mode IN the basis" as the actual driver, not just "is it a resonance."
    probe_points = {
        "3rd resonance (mode 3, NOT in the rank-2 basis)": float(np.sqrt(eigvals[2])),
        "1st resonance (mode 1, IS in the rank-2 basis)": omega1,
    }
    results = {}
    for label, omega in probe_points.items():
        indicator = rom_small.hierarchical_error_indicator(omega, F_free, comparison_rom=rom_richer)
        U_true = sysobj.solve_harmonic(omega, F_full)[free]
        x_small = rom_small.frequency_response([omega], F_free)[0]
        rel_true_err = np.linalg.norm(x_small - U_true) / np.linalg.norm(U_true)
        rel_indicator = indicator / np.linalg.norm(x_small)
        results[label] = (rel_indicator, rel_true_err)
        print(f"{label}: omega={omega:.1f}  relative hierarchical indicator={rel_indicator:.4e}  "
              f"relative true error (vs fea_engine)={rel_true_err:.4e}")

    ind_bad, err_bad = results["3rd resonance (mode 3, NOT in the rank-2 basis)"]
    ind_good, err_good = results["1st resonance (mode 1, IS in the rank-2 basis)"]
    assert err_bad > err_good, "sanity check on the probe points themselves: the 'bad' case must really be worse"
    assert ind_bad > ind_good, (
        "the hierarchical indicator must be larger at the genuinely worse-represented "
        "frequency -- it should track true error qualitatively, not just look plausible")
    print("PASS -- hierarchical indicator is large exactly where the true error is large")
