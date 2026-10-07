"""
Real-fixture validation of nonlinear_dynamics.integrate_newmark_surrogate()
against fea_engine's own geometrically nonlinear Beam2DCorotational
element, via nonlinear_solver.solve_nonlinear_transient() (Step 3) as
the full-order dynamic ground truth -- the strongest check available
for this module: test_nonlinear_dynamics.py's synthetic/scipy-ODE
checks prove the Newmark bookkeeping and correction-mode logic are
right, but not that the whole pipeline (mass-normalized modal
projection, applied-load training data, a fitted surrogate, THEN
reduced nonlinear time integration) survives contact with a real,
coupled, geometrically nonlinear structural model released into free
vibration.

Free-decay setup, mirroring the reference paper's own Fig. 7 dynamic
comparison and this project's earlier `mfs-nlrom-beam` skill
reproduction of it (`scripts/dynamic_newmark.py`): light (~1%) modal
Rayleigh damping calibrated across the two retained modes, a
mode-1-dominated initial condition within the surrogate's training
amplitude range, F_ext=0 for t>0. Both R^2 windows are reported and
checked, matching that prior work's own honest convention: a SHORT
window (~2 periods) where a small per-step frequency mismatch has not
yet compounded into visible phase drift, and the FULL window, where
phase drift (not amplitude/decay-envelope error) typically dominates a
pointwise R^2 for a fitted-vs-exact nonlinear force model -- both are
checked so a real degradation in the FULL window isn't silently
excused by only looking at the short one.
"""
import numpy as np
import pytest

from rom_engine.nonlinear_rom import AppliedLoadStrategy, MultiFidelitySurrogate, PolynomialModalROM
from rom_engine.nonlinear_dynamics import integrate_newmark_surrogate
from rom_engine.metrics import r_squared
from fea_engine import nonlinear_solver as nls
from fea_fixtures import clamped_clamped_nonlinear_beam_system


class _ZeroLoad:
    """A trivial TimeHistoryLoad-like object (force_at(t, n_dof, npn) ->
    zeros) for the free-decay case -- F_ext(t)=0 for all t>0, matching
    nls.solve_nonlinear_transient()'s `load` interface without needing
    fea_engine.loads.TimeHistoryLoad's LoadPattern machinery for a
    genuinely zero load."""
    def force_at(self, t, n_dof, npn):
        return np.zeros(n_dof)


@pytest.fixture(scope="module")
def free_decay_comparison():
    """Builds the clamped-clamped beam fixture with light calibrated
    Rayleigh damping, trains BOTH model families on applied-load static
    data, runs the real fea_engine nonlinear Newmark-Newton free-decay
    ground truth, and runs both reduced models' own
    integrate_newmark_surrogate() over the SAME initial condition and
    duration -- built once per module (each of these is a real,
    non-trivial solve) and shared across every test in this file."""
    n_modes = 2
    zeta_target = 0.01

    # Pass 1 (undamped) purely to get the natural frequencies needed to
    # calibrate Rayleigh damping to a target modal damping ratio at both
    # retained modes -- the same two-point calibration the validated
    # mfs-nlrom-beam prototype's dynamic_newmark.py uses.
    fx0 = clamped_clamped_nonlinear_beam_system(n_elem=10, n_modes=n_modes)
    omega = 2 * np.pi * fx0["freq_hz"]
    w1, w2 = omega[0], omega[-1]
    A_cal = np.array([[1 / (2 * w1), w1 / 2], [1 / (2 * w2), w2 / 2]])
    alpha_R, beta_R = np.linalg.solve(A_cal, [zeta_target, zeta_target])

    fx = clamped_clamped_nonlinear_beam_system(
        n_elem=10, n_modes=n_modes, damping_alpha=alpha_R, damping_beta=beta_R)
    V, M_ff, C_ff = fx["V"], fx["M_ff"], fx["C_ff"]
    freq_hz, free, mat, fes, n_dof = fx["freq_hz"], fx["free_dofs"], fx["mat"], fx["sys"], fx["n_dof"]
    Lambda = (2 * np.pi * freq_hz) ** 2
    # Rayleigh damping is exactly diagonalized by the mass-normalized
    # modes, so V.T @ C_ff @ V is diagonal to numerical precision --
    # extract the diagonal directly as the modal damping array
    # nonlinear_dynamics.integrate_newmark_surrogate's C_r expects.
    C_r = np.diag(V.T @ C_ff @ V)

    rng = np.random.default_rng(7)
    strategy = AppliedLoadStrategy(target_fracs=(0.3, 2.0), reference_scale=0.01,
                                    n_samples=40, rng=rng)
    q_l, q_nl, F_nl = strategy.generate(V, M_ff, freq_hz, fx["mode_shape_peaks"], fx["fom_solver"])

    mfs = MultiFidelitySurrogate(kernel="cubic").fit(q_l, F_nl)
    poly = PolynomialModalROM(n_modes).fit(q_nl, F_nl)

    # Mode-1-dominated IC well within the training amplitude range.
    q_nl0 = np.zeros(n_modes)
    q_nl0[0] = np.max(np.abs(q_nl[:, 0])) * 0.6
    qdot0 = np.zeros(n_modes)
    u0_full = np.zeros(n_dof)
    u0_full[free] = V @ q_nl0

    # >=20 steps per period of the HIGHEST retained mode (matching the
    # validated prototype's own resolution requirement -- a coarser dt
    # under-resolves the higher mode and both ROMs pick up spurious
    # high-frequency artifacts as a direct consequence, not a modeling
    # error).
    dt = 1.0 / (freq_hz[-1] * 25)
    n_periods = 8
    T_total = n_periods / freq_hz[0]
    n_steps = int(round(T_total / dt))

    t_fe, U_hist = nls.solve_nonlinear_transient(
        fes, mat, _ZeroLoad(), T_total, dt, u0=u0_full, tol=1e-8, max_iter=40)
    # mass-orthogonal projection onto the SAME retained modal basis
    # used to train the surrogates (V.T @ M_ff @ u, matching Step 2's
    # own projection convention throughout this package).
    q_nl_fe = (V.T @ M_ff @ U_hist[:, free].T).T

    results = {}
    for name, model, domain in [("mfs", mfs, "q_l"), ("poly", poly, "q_nl")]:
        t_r, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
            Lambda, C_r, model, F_ext=0.0, q0=q_nl0, qdot0=qdot0,
            dt=dt, n_steps=n_steps, domain=domain, correction="sign_deviation")
        results[name] = q_nl_hist

    n_short = round(0.25 * T_total / dt)   # ~2 periods of mode 1
    return {
        "q_nl_fe": q_nl_fe, "q_nl_mfs": results["mfs"], "q_nl_poly": results["poly"],
        "n_short": n_short, "n_modes": n_modes,
    }


class TestFreeDecayAgainstFullOrderFE:
    @pytest.mark.parametrize("key", ["q_nl_mfs", "q_nl_poly"])
    def test_short_window_r_squared_is_high(self, free_decay_comparison, key):
        d = free_decay_comparison
        r2 = r_squared(d["q_nl_fe"][:d["n_short"]], d[key][:d["n_short"]])
        assert r2 > 0.9

    @pytest.mark.parametrize("key", ["q_nl_mfs", "q_nl_poly"])
    def test_full_window_r_squared_is_reasonable(self, free_decay_comparison, key):
        # honest, looser threshold than the short window -- phase drift
        # over 8 periods is an EXPECTED property of a fitted-vs-exact
        # nonlinear force model (see module docstring), not a bug; this
        # still catches a genuinely broken integration (blow-up, wrong
        # sign, wrong frequency band) rather than just amplitude/decay
        # agreement, which the next test checks directly.
        d = free_decay_comparison
        r2 = r_squared(d["q_nl_fe"], d[key])
        assert r2 > 0.5

    @pytest.mark.parametrize("key", ["q_nl_mfs", "q_nl_poly"])
    def test_amplitude_decay_envelope_matches(self, free_decay_comparison, key):
        # a physically meaningful check independent of phase: the
        # PEAK mode-1 amplitude near the start and near the end of the
        # run should decay by a comparable amount as the true FE
        # response's does, confirming the fitted model reproduces the
        # right DAMPING/DECAY RATE, not just an initially-correct
        # trajectory that later diverges in amplitude too.
        d = free_decay_comparison
        fe_start = np.max(np.abs(d["q_nl_fe"][:d["n_short"], 0]))
        fe_end = np.max(np.abs(d["q_nl_fe"][-d["n_short"]:, 0]))
        rom_start = np.max(np.abs(d[key][:d["n_short"], 0]))
        rom_end = np.max(np.abs(d[key][-d["n_short"]:, 0]))
        fe_decay_frac = fe_end / fe_start
        rom_decay_frac = rom_end / rom_start
        assert abs(fe_decay_frac - rom_decay_frac) < 0.15

    def test_no_blow_up(self, free_decay_comparison):
        d = free_decay_comparison
        for key in ["q_nl_mfs", "q_nl_poly"]:
            assert np.all(np.isfinite(d[key]))
            assert np.max(np.abs(d[key])) < 5 * np.max(np.abs(d["q_nl_fe"]))
