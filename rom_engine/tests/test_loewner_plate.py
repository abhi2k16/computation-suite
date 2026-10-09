"""
test_loewner_plate.py -- validates rom_engine.loewner/rom_engine.screening
against plate_fixtures.build_paper_plate_system(): a materially harder,
more realistic real-model target than the mass-spring-chain and beam
fixtures used elsewhere (867 structural DOF, 6 measurement DOFs, a
real added-mass fluid-loading effect), matching the paper's own
"Example 1" excitation/measurement node layout (Table 1).

This mirrors the working prototype's fem_rom_comparison.py workflow,
re-pointed at rom_engine's LoewnerROM / screen_physical_modes /
modal_assurance_criterion instead of the prototype's flat functions.
"""
__author__ = "Abhijeet"
import numpy as np

from rom_engine.loewner import LoewnerROM
from rom_engine.screening import screen_physical_modes
from rom_engine.metrics import modal_assurance_criterion
import plate_fixtures as pf
import loewner_fixtures as lf


def _identify_case(fx, M_wet, f_wet_undamped, excite_pid, meas_pids, rng, n_target_modes=6):
    free = fx["free"]
    K = fx["K"]
    alpha_R, beta_R = fx["alpha_R"], fx["beta_R"]
    K_ff = K[np.ix_(free, free)]
    M_ff = M_wet[np.ix_(free, free)]
    C_ff = alpha_R * M_ff + beta_R * K_ff   # self-consistent w.r.t. (M_wet, K)

    free_index_of = -np.ones(fx["n_dof"], dtype=int)
    free_index_of[free] = np.arange(len(free))

    def w_dof_free_index(fe_node):
        idx = free_index_of[3 * fe_node]
        if idx < 0:
            raise ValueError("node's w-DOF is constrained (on the simply-supported edge)")
        return idx

    fmin, fmax = 0.7 * f_wet_undamped[0], 1.2 * f_wet_undamped[n_target_modes - 1]
    f_true, eta_true, Phi_true = lf.ground_truth_modes(M_ff, C_ff, K_ff, alpha_R, beta_R, fmin, fmax)
    n_modes = len(f_true)

    excite_dof = w_dof_free_index(fx["table1_fe_node"][excite_pid])
    meas_dofs = [w_dof_free_index(fx["table1_fe_node"][p]) for p in meas_pids]
    F_ff = np.zeros(len(free)); F_ff[excite_dof] = 1.0

    n_pool = 60
    f_pool_hz = np.linspace(fmin, fmax, n_pool) * (1 + 1e-3 * rng.standard_normal(n_pool))
    omega_pool = 2 * np.pi * f_pool_hz
    # x_pool_ref sampled at meas_dofs[0] (node 58) rather than a node
    # that sits near a nodal line for some of the target modes (node 32
    # does, for this excitation -- see the reference validation
    # scripts) -- a real modal-testing pitfall, not a method limitation.
    x_pool_ref = lf.frf(omega_pool, F_ff, M_ff, C_ff, K_ff)[meas_dofs[0], :]

    n_interp = n_modes + 3
    idx = rng.choice(n_pool, size=2 * n_interp, replace=False)
    idx_a, idx_b = idx[:n_interp], idx[n_interp:]
    rom = LoewnerROM.fit(omega_pool[idx_a], omega_pool[idx_b], x_pool_ref[idx_a], x_pool_ref[idx_b])

    physical = screen_physical_modes(omega_pool, x_pool_ref, fmin, fmax,
                                      rng=rng, n_roms=25, n_interp=n_interp)

    X_beta_multi = lf.frf(omega_pool[idx_b], F_ff, M_ff, C_ff, K_ff)[meas_dofs, :]
    Phi_ident = rom.reconstruct_mode_shapes(X_beta_multi, omega_beta=omega_pool[idx_b])

    return dict(f_true=f_true, eta_true=eta_true, Phi_true=Phi_true[meas_dofs, :],
                physical=physical, rom=rom, Phi_ident=Phi_ident, n_modes=n_modes)


def _report_and_check(result, label, f_tol=0.03, mac_min=0.7, min_matched_frac=0.6):
    f_true, eta_true, Phi_true = result["f_true"], result["eta_true"], result["Phi_true"]
    physical, rom, Phi_ident = result["physical"], result["rom"], result["Phi_ident"]
    n_modes = result["n_modes"]

    phys_f = np.array([m.f for m in physical]) if physical else np.array([])
    print(f"\n{label}: {len(physical)} screened physical modes vs {n_modes} true modes")
    print(f"{'mode':>4} {'f_true':>10} {'f_id':>10} {'err%':>7} {'MAC':>7}")

    n_matched = 0
    matched_true = set()
    for i in range(n_modes):
        if len(phys_f) == 0:
            break
        k = np.argmin(np.abs(phys_f - f_true[i]))
        if k in matched_true or abs(phys_f[k] - f_true[i]) / f_true[i] > f_tol:
            print(f"{i:>4} {f_true[i]:>10.3f}    -- not identified within {100*f_tol:.0f}% --")
            continue
        matched_true.add(k)
        j = np.argmin(np.abs(rom.f - f_true[i]))
        mac = modal_assurance_criterion(Phi_ident[:, j], Phi_true[:, i])
        print(f"{i:>4} {f_true[i]:>10.3f} {physical[k].f:>10.3f} "
              f"{100*abs(physical[k].f-f_true[i])/f_true[i]:>7.3f} {mac:>7.4f}")
        n_matched += 1
        # a matched mode's mode shape should be at least loosely aligned
        # with the true one -- not held to the >0.9 standard of the
        # cleaner cantilever/mass-spring fixtures, since this is a much
        # larger, more realistic model with only 6 measurement DOFs
        assert mac > mac_min, f"{label} mode {i}: MAC={mac:.3f} too low"

    print(f"{label}: matched {n_matched}/{n_modes} true modes")
    assert n_matched >= min_matched_frac * n_modes, (
        f"{label}: only matched {n_matched}/{n_modes} true modes "
        f"(need at least {min_matched_frac*100:.0f}%)")


def test_air_loaded_plate_identification_matches_ground_truth():
    fx = pf.build_paper_plate_system(nex=16, ney=16)
    rng = np.random.default_rng(21)
    result = _identify_case(fx, fx["M_air"], fx["f_air"], excite_pid=50,
                             meas_pids=[58, 32, 136, 150, 193, 265], rng=rng)
    _report_and_check(result, "air-loaded")


def test_water_loaded_plate_identification_matches_ground_truth():
    fx = pf.build_paper_plate_system(nex=16, ney=16)
    rng = np.random.default_rng(22)
    result = _identify_case(fx, fx["M_water"], fx["f_water"], excite_pid=50,
                             meas_pids=[58, 32, 136, 150, 193, 265], rng=rng)
    _report_and_check(result, "water-loaded")
