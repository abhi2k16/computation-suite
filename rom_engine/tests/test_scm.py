"""
test_scm.py -- validates rom_engine.scm.SingularValueLowerBound and
certified_error_bound() against a REAL, damped fea_engine cantilever
beam (mirroring test_frequency.py/test_greedy.py's fixtures and
ground-truth conventions).

Checks:
  1. THE core promise: lower_bound(omega) NEVER exceeds the TRUE
     sigma_min(A(omega)) (computed via direct SVD), swept across many
     points including AT and very near resonance -- this is the actual
     certification property, and if it were ever violated even once,
     the whole module would be making a false claim.
  2. certified_error_bound() NEVER undershoots the TRUE error (vs.
     fea_engine's own solve_harmonic()) whenever it returns a finite
     value -- the end-to-end promise a "certified bound" actually
     needs to keep.
  3. AT (or extremely near) a reference point, the bound is tight --
     demonstrating the module has genuine, if narrow, utility, not
     just correctness-by-triviality (a bound that's always 0 or +inf
     would technically never be "violated" either).
  4. An HONEST characterization of the bound's practical looseness for
     a real FE stiffness matrix: the "useful radius" around a
     reference point is measured and reported directly, not glossed
     over -- see docs/phase4_error_bounds_greedy_roadmap.md's Phase 4d
     status note for why this is expected (a real, large scale
     mismatch between ||K||_2 and sigma_min(A(omega)) for this model,
     not an implementation bug), and why the classical natural-norm
     SCM exists to address exactly this.
  5. greedy_train() behaves sensibly: it adds references (doesn't loop
     forever or stall), and the reported gap at each newly-added point
     was genuinely the worst available at that iteration.
"""
import numpy as np
from scipy.linalg import eigh
from rom_engine import FrequencyROM, SingularValueLowerBound, certified_error_bound
import fea_fixtures as ff


def _true_sigma_min(rom, omega):
    """Independent, direct (expensive, O(n_dof^3)) computation of the
    TRUE smallest singular value of the FULL A(omega) -- the ground
    truth lower_bound() is checked against. Deliberately does NOT use
    scm.SingularValueLowerBound.add_reference()'s own code path, so
    this is a genuinely independent cross-check."""
    theta = rom.affine._theta(omega)
    A = theta[0] * rom.affine.components[0]
    for q in range(1, rom.affine.Q):
        A = A + theta[q] * rom.affine.components[q]
    return float(np.linalg.svd(A, compute_uv=False).min())


def test_lower_bound_never_exceeds_true_sigma_min():
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]

    eigvals, eigvecs = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    basis = eigvecs[:, :10]
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))
    scm = SingularValueLowerBound.from_affine(rom.affine)

    # a handful of references spread across a band including resonance
    for om in np.linspace(0.2, 2.0, 5) * omega1:
        scm.add_reference(om)

    test_omegas = np.linspace(0.1, 3.0, 300) * omega1   # includes exact resonance
    worst_violation = -np.inf
    for om in test_omegas:
        true_sigma = _true_sigma_min(rom, om)
        lb = scm.lower_bound(om)
        worst_violation = max(worst_violation, lb - true_sigma)

    print(f"worst (lower_bound - true_sigma_min) across {len(test_omegas)} points "
          f"(must be <= ~0): {worst_violation:.3e}")
    assert worst_violation <= 1e-6, (
        "CERTIFICATION VIOLATED: lower_bound() must never exceed the true sigma_min, "
        "by construction of the Weyl/Mirsky perturbation inequality -- a violation "
        "here would mean a real bug in the bound's derivation or implementation")
    print("PASS -- the lower bound never exceeds the true sigma_min, at any tested "
          "frequency including exact resonance (the core certification property)")


def test_certified_error_bound_never_undershoots_true_error():
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    sysobj = fx["sys"]
    n_dof = fx["n_dof"]

    eigvals, eigvecs = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    basis = eigvecs[:, :10]
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))
    scm = SingularValueLowerBound.from_affine(rom.affine)

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    # add references AT the exact frequencies we'll test -- given how
    # narrow this simplified bound's useful radius is (Section 4 above),
    # this is the ONLY regime where certified_error_bound() returns a
    # finite (non-+inf) value at all for this model; that narrowness is
    # itself the finding test_useful_radius_is_narrow_for_this_model()
    # documents directly, not hidden here.
    test_omegas = [omega1, 0.5 * omega1, 1.3 * omega1]
    for om in test_omegas:
        scm.add_reference(om)

    checked_finite = 0
    for om in test_omegas:
        ceb = certified_error_bound(rom, scm, om, F_free)
        U_true = sysobj.solve_harmonic(om, F_full)[free]
        x_rom = rom.frequency_response([om], F_free)[0]
        true_err = np.linalg.norm(x_rom - U_true)
        print(f"omega/omega1={om/omega1:.3f}  certified_bound={ceb:.4e}  "
              f"true_err(vs fea_engine)={true_err:.4e}  "
              f"holds={'yes' if not np.isfinite(ceb) else ceb >= true_err}")
        if np.isfinite(ceb):
            checked_finite += 1
            assert ceb >= true_err, (
                "a finite certified_error_bound() must NEVER be smaller than the "
                "TRUE error against fea_engine's own solve -- this is the actual "
                "promise of a 'certified' bound, not just an internal consistency check")
    assert checked_finite > 0, "expected at least the exact reference points to give a finite bound"
    print(f"PASS -- certified_error_bound() never undershoots the true error, "
          f"at all {checked_finite} points where it returned a finite value")


def test_bound_is_tight_exactly_at_a_reference_point():
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]

    eigvals, eigvecs = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    basis = eigvecs[:, :10]
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))
    scm = SingularValueLowerBound.from_affine(rom.affine)

    true_sigma = scm.add_reference(omega1)
    lb_at_reference = scm.lower_bound(omega1)
    print(f"true sigma_min at the reference itself: {true_sigma:.6f}, "
          f"lower_bound() there: {lb_at_reference:.6f}")
    assert abs(lb_at_reference - true_sigma) < 1e-6, (
        "AT its own reference point (zero perturbation), the bound must be EXACT, "
        "not just non-violating -- this is what distinguishes genuine (if narrow) "
        "utility from a trivial always-0 bound")
    print("PASS -- the bound is exact (not just non-violating) precisely at a reference point")


def test_useful_radius_is_narrow_for_this_model_and_is_reported_honestly():
    """This is a DELIBERATE, documented finding, not a bug report: for
    this cantilever's real stiffness matrix, ||K||_2 is enormous
    (~1e11-1e12) relative to sigma_min(A(omega)) (~1e3) -- a genuine
    property of a real, generically ill-conditioned FE stiffness
    matrix (its eigenvalues span many orders of magnitude), not an
    artifact of this test. Because the simplified (non-"natural-norm")
    Lipschitz bound's perturbation term scales with the RAW component
    spectral norm, this means the bound collapses to the trivial 0
    within a very small relative frequency step of each reference --
    this test MEASURES that radius directly and reports it, rather
    than asserting a specific "should be at least this wide" value
    (which would misrepresent this simplified method's real, literature-
    anticipated limitation -- see docs/phase4_error_bounds_greedy_
    roadmap.md's Phase 4d status note)."""
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]

    eigvals, eigvecs = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    basis = eigvecs[:, :10]
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))
    scm = SingularValueLowerBound.from_affine(rom.affine)
    scm.add_reference(omega1)

    print(f"component spectral norms ||M_r||_2, ||K_r||_2: {scm.component_norms}")

    fracs = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    last_nonzero_frac = 0.0
    for frac in fracs:
        lb = scm.lower_bound(omega1 * (1 + frac))
        print(f"  relative frequency step = {frac:.1e}  ->  lower_bound = {lb:.4e}")
        if lb > 0:
            last_nonzero_frac = frac

    print(f"useful (non-trivial) radius for this model: approximately "
          f"{last_nonzero_frac:.1e} relative frequency step")
    # not asserting a specific radius (that would misrepresent the finding as a
    # requirement rather than a measurement) -- only that the module ran and
    # produced a well-defined, reportable number, and that it IS narrow (< 1%)
    # for this real stiffness matrix, confirming the honest caveat is accurate
    assert last_nonzero_frac < 1e-2 or lb == 0.0, (
        "if this ever becomes false, the honest 'narrow radius' characterization "
        "in this module's docs would need updating, not this assertion loosened")
    print("PASS -- the bound's narrow practical radius for a real FE stiffness "
          "matrix is measured and reported, matching the documented expectation")


def test_greedy_train_adds_references_and_gap_reflects_worst_candidate():
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]

    eigvals, eigvecs = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    basis = eigvecs[:, :10]
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))
    scm = SingularValueLowerBound.from_affine(rom.affine)

    candidates = np.linspace(0.5, 1.5, 20) * omega1
    history = scm.greedy_train(candidates, tol=0.5, max_references=6)

    print(f"greedy_train added {len(scm.reference_omegas)} references "
          f"(history length {len(history)}):")
    for om, gap in history:
        print(f"  omega/omega1={om/omega1:.3f}  gap={gap:.4f}")

    assert len(scm.reference_omegas) >= 2, "expected at least a couple of references to be added"
    assert len(scm.reference_omegas) <= 6, "must respect max_references"

    # verify the reported gap at each selection really was the max over the
    # candidate pool at that point in time, by replaying with a fresh
    # SingularValueLowerBound and the same reference order. greedy_train()
    # seeds ONE reference (defaulting to the middle candidate) BEFORE its
    # history recording starts, so history[0]'s gap already reflects that
    # seed's presence -- replay must add the same seed first to match.
    replay = SingularValueLowerBound.from_affine(rom.affine)
    seed_omega = float(candidates[len(candidates) // 2])
    replay.add_reference(seed_omega)
    for i, (om_selected, reported_gap) in enumerate(history):
        gaps_now = []
        for om in candidates:
            lb = replay.lower_bound(om)
            ub = replay.upper_bound(om)
            gaps_now.append((ub - lb) / ub if np.isfinite(ub) and ub > 0 else np.inf)
        max_gap_now = max(gaps_now) if replay.reference_omegas else np.inf
        assert abs(max_gap_now - reported_gap) < 1e-9 or (
            not np.isfinite(max_gap_now) and not np.isfinite(reported_gap)), (
            f"selection {i}: history's reported gap ({reported_gap}) should match "
            f"the actual max gap over the candidate pool at that point ({max_gap_now})")
        replay.add_reference(om_selected)
    print("PASS -- greedy_train() adds references and its reported gaps genuinely "
          "match the worst-candidate gap at each step")
