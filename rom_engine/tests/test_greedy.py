"""
test_greedy.py -- validates rom_engine.greedy.greedy_train_frequency_basis()
against a REAL, damped fea_engine cantilever beam (mirroring
test_frequency.py's fixtures and ground-truth conventions).

Checks (per docs/phase4_error_bounds_greedy_roadmap.md Section 6):
  1. Greedy-selected training frequencies CLUSTER near the model's
     actual resonances -- a directly checkable claim (density near a
     known resonance vs. a uniform grid's density), not "seems
     reasonable."
  2. A greedy-trained basis achieves LOWER error than a same-rank
     uniform-grid-trained basis on a HELD-OUT test set -- the claim
     that justifies greedy sampling's extra complexity, checked by
     actually comparing both, not assumed.
"""
__author__ = "Abhijeet"
import numpy as np
from scipy.linalg import eigh
from rom_engine import greedy_train_frequency_basis, build_pod_basis_from_frf_snapshots, FrequencyROM
import fea_fixtures as ff


def test_greedy_selections_cluster_near_resonance():
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    n_dof = fx["n_dof"]

    eigvals, _ = eigh(Kff, Mff)
    omega1 = float(np.sqrt(eigvals[0]))
    omega2 = float(np.sqrt(eigvals[1]))

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    # a band spanning both the 1st and 2nd resonance
    training_omegas = np.linspace(0.1, 7.0, 150) * omega1
    basis, history = greedy_train_frequency_basis(
        training_omegas, Mff, Kff, F_free, rayleigh=(fx["alpha"], fx["beta"]),
        n_seed=3, tol=1e-4, max_modes=18)

    selected = np.array([h[0] for h in history[:-1]])   # exclude the final "converged" probe, which wasn't actually added
    # actually re-derive exactly which omegas were ADDED: every history
    # entry except possibly the last (if it triggered the tol stop and
    # was never appended) corresponds to an added point
    n_added = basis.n_modes - 3   # started from n_seed=3, grew by n_added real additions
    added_omegas = np.array([h[0] for h in history[:max(n_added, 0)]])
    print(f"greedy added {len(added_omegas)} frequencies beyond the 3 seeds "
          f"(final basis rank {basis.n_modes})")
    assert len(added_omegas) >= 2, "expected the greedy loop to add at least a couple of points on this band"

    # distance from each ADDED omega to the nearest of the two known resonances
    resonances = np.array([omega1, omega2])
    dist_to_nearest_resonance = np.min(
        np.abs(added_omegas[:, None] - resonances[None, :]), axis=1) / omega1
    band_width = training_omegas[-1] - training_omegas[0]

    print("added omegas (relative to omega1) and their distance to the nearest resonance:")
    for om, d in zip(added_omegas, dist_to_nearest_resonance):
        print(f"  omega/omega1={om/omega1:.3f}  dist/omega1={d:.3f}")

    median_dist = float(np.median(dist_to_nearest_resonance))
    # a uniform random point in the band would have a MUCH larger typical
    # distance to the nearest of just 2 resonance points -- compare against
    # that as the "no clustering" null hypothesis
    rng = np.random.default_rng(0)
    uniform_samples = rng.uniform(training_omegas[0], training_omegas[-1], 2000)
    uniform_dist = np.min(np.abs(uniform_samples[:, None] - resonances[None, :]), axis=1) / omega1
    median_uniform_dist = float(np.median(uniform_dist))

    print(f"median distance/omega1 to nearest resonance: greedy-selected = {median_dist:.3f}, "
          f"uniform-random baseline = {median_uniform_dist:.3f}")
    assert median_dist < 0.5 * median_uniform_dist, (
        "greedy-selected frequencies should cluster measurably closer to the "
        "known resonances than a uniform/random baseline would")
    print("PASS -- greedy-selected training frequencies cluster near the model's actual resonances")


def test_greedy_basis_beats_uniform_grid_basis_at_same_rank():
    """Compares a greedy-trained basis against SEVERAL uniform-grid
    bases (different phase offsets, same rank/budget) rather than just
    one -- a single arbitrary uniform grid can get lucky or unlucky
    depending on whether it happens to land near a resonance, so a fair
    ("you don't know in advance where the resonances are") comparison
    is against the uniform strategy's TYPICAL behavior, not one draw of
    it. The claim checked is on the MEDIAN held-out error, since a
    tight training budget can leave a worse worst-case error for either
    strategy depending on which single frequency it happens to miss --
    median is the more robust, less noise-sensitive statistic here."""
    fx = ff.damped_cantilever_beam_system(n=20)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]
    sysobj = fx["sys"]
    n_dof = fx["n_dof"]

    eigvals, _ = eigh(Kff, Mff)
    omegas_n = np.sqrt(eigvals)
    omega1 = float(omegas_n[0])

    F_full = np.zeros(n_dof)
    F_full[n_dof - 2] = 1000.0
    F_free = F_full[free]

    # a wide band spanning the first three resonances, with a TIGHT
    # basis-size budget -- forces genuine competition for where the
    # limited full-order solves get spent
    training_omegas = np.linspace(0.1, 20.0, 400) * omega1
    greedy_basis, history = greedy_train_frequency_basis(
        training_omegas, Mff, Kff, F_free, rayleigh=(fx["alpha"], fx["beta"]),
        n_seed=3, tol=1e-9, max_modes=5)
    final_rank = greedy_basis.n_modes
    print(f"greedy basis final rank: {final_rank}")
    rom_greedy = FrequencyROM.from_MCK(Mff, Kff, greedy_basis.V, rayleigh=(fx["alpha"], fx["beta"]))

    # held-out test frequencies scattered around the first three
    # resonances -- where a poor basis is most exposed
    rng = np.random.default_rng(1)
    test_omegas = np.concatenate([
        omegas_n[i] * (1.0 + 0.05 * rng.standard_normal(15)) for i in range(3)
    ])
    test_omegas = test_omegas[test_omegas > 0]

    def median_rel_error(rom):
        errs = []
        for om in test_omegas:
            U_true = sysobj.solve_harmonic(om, F_full)[free]
            x_rom = rom.frequency_response([om], F_free)[0]
            errs.append(np.linalg.norm(x_rom - U_true) / max(np.linalg.norm(U_true), 1e-30))
        return float(np.median(errs))

    err_greedy = median_rel_error(rom_greedy)

    # several uniform grids at different phase offsets, same rank/budget
    n_phases = 7
    uniform_medians = []
    for phase in np.linspace(0, 1, n_phases, endpoint=False):
        offset = phase * (training_omegas[-1] - training_omegas[0]) / final_rank
        uniform_omegas = np.linspace(training_omegas[0] + offset, training_omegas[-1], final_rank)
        uniform_basis = build_pod_basis_from_frf_snapshots(
            uniform_omegas, Mff, Kff, F_free, C=Cff, n_modes=final_rank)
        rom_uniform = FrequencyROM.from_MCK(Mff, Kff, uniform_basis.V, rayleigh=(fx["alpha"], fx["beta"]))
        u_err = median_rel_error(rom_uniform)
        uniform_medians.append(u_err)
        print(f"  uniform grid, phase={phase:.2f}: median rel err = {u_err:.3e}")

    mean_uniform_median = float(np.mean(uniform_medians))
    print(f"greedy-trained basis: median rel err = {err_greedy:.3e}")
    print(f"mean of {n_phases} uniform-grid medians: {mean_uniform_median:.3e}")

    assert err_greedy < mean_uniform_median, (
        "at the SAME basis rank (same full-order-solve budget), the greedy-trained "
        "basis should beat the uniform strategy's TYPICAL (mean-over-phase-offsets) "
        "performance on a held-out, resonance-concentrated test set -- this is the "
        "whole justification for greedy's extra complexity")
    assert err_greedy < min(uniform_medians), (
        "the greedy basis should beat EVERY tested uniform-grid phase here, not just "
        "the average -- a weaker result would still be plausible in general, but this "
        "particular run should show a clean win given the tight, multi-resonance budget")
    print("PASS -- greedy-trained basis outperforms uniform-grid bases (every tested "
          "phase offset) on held-out, resonance-concentrated test frequencies")
