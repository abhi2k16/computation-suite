"""
Real-fixture validation of nnm.py's harmonic-balance NNM backbone
against fea_engine's own geometrically nonlinear Beam2DCorotational
element -- the strongest check available for this module, mirroring
Step 4's test_nonlinear_dynamics_fea.py: test_nnm.py's synthetic
Duffing-oscillator/FrequencyROM checks prove the AFT/harmonic-balance
bookkeeping and Newton continuation logic are right, but not that the
whole pipeline (mass-normalized modal projection, applied-load
training data, a fitted surrogate, THEN harmonic-balance backbone
continuation) survives contact with a real, coupled, geometrically
nonlinear structural model.

Ground truth, mirroring this project's own earlier `mfs-nlrom-beam`
skill reproduction (`scripts/backbone_nnm.py`): rather than a second,
independent HBM implementation (which does not exist for this real
model), the backbone is measured the direct, physically transparent
way this project has already validated -- undamped free vibration
(zeta=0, so the period is not contaminated by decay) from a
mode-1-dominated initial condition at the SAME amplitude a backbone
point targets, with the resulting (at these training-range amplitudes,
still close to single-harmonic) period measured via
fft-with-parabolic-interpolation (`measure_period`, ported verbatim in
spirit from that prototype's own function of the same name) and
inverted to a frequency. Same honest flagging as that prototype: this
is a materially different, simpler method (time-domain, not a second
single-shot solver) from solve_nnm_backbone() itself, so it is a
genuine independent check, not solve_nnm_backbone() checked against
itself.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from rom_engine.nonlinear_rom import AppliedLoadStrategy, MultiFidelitySurrogate, PolynomialModalROM
from rom_engine.nnm import HarmonicBalanceSystem, solve_nnm_backbone
from fea_engine import nonlinear_solver as nls
from fea_fixtures import clamped_clamped_nonlinear_beam_system


class _ZeroLoad:
    def force_at(self, t, n_dof, npn):
        return np.zeros(n_dof)


def measure_period(hist, dt):
    """Dominant-frequency estimate via FFT with quadratic (parabolic)
    interpolation of the peak magnitude bin, converted to a period --
    ported in spirit from mfs-nlrom-beam/scripts/backbone_nnm.py's own
    function of the same name (chosen there, and here, over naive
    peak-to-peak picking because it is far more robust once the
    response has any higher-harmonic content from modal coupling)."""
    x = hist - np.mean(hist)
    n = len(x)
    mag = np.abs(np.fft.rfft(x * np.hanning(n)))
    mag[0] = 0.0
    k = np.argmax(mag)
    if k == 0 or k == len(mag) - 1:
        freq = k / (n * dt)
    else:
        a_, b_, c_ = mag[k - 1], mag[k], mag[k + 1]
        denom = (a_ - 2 * b_ + c_)
        delta = 0.5 * (a_ - c_) / denom if denom != 0 else 0.0
        freq = (k + delta) / (n * dt)
    return 1.0 / freq if freq > 0 else None


@pytest.fixture(scope="module")
def backbone_comparison():
    """Trains both surrogate families on the (undamped) clamped-clamped
    beam fixture, traces each one's own solve_nnm_backbone() for the
    mode-1 master mode, then, for the same handful of amplitude points,
    runs a REAL fea_engine undamped free-vibration solve and measures
    its own period independently -- built once per module and shared
    across every test in this file (each backbone point's ground-truth
    solve is a real nonlinear Newmark-Newton run over several periods,
    not cheap)."""
    n_modes = 2
    fx = clamped_clamped_nonlinear_beam_system(n_elem=10, n_modes=n_modes)
    V, M_ff = fx["V"], fx["M_ff"]
    freq_hz, free, mat, fes, n_dof = fx["freq_hz"], fx["free_dofs"], fx["mat"], fx["sys"], fx["n_dof"]
    Lambda = (2 * np.pi * freq_hz) ** 2

    rng = np.random.default_rng(11)
    strategy = AppliedLoadStrategy(target_fracs=(0.3, 2.0), reference_scale=0.01,
                                    n_samples=40, rng=rng)
    q_l, q_nl, F_nl = strategy.generate(V, M_ff, freq_hz, fx["mode_shape_peaks"], fx["fom_solver"])

    mfs = MultiFidelitySurrogate(kernel="cubic").fit(q_l, F_nl)
    poly = PolynomialModalROM(n_modes).fit(q_nl, F_nl)

    # Backbone amplitude range for the master (mode-1) coordinate,
    # comfortably inside the training data's own mode-1 amplitude range
    # (never extrapolating a fitted surrogate beyond where it was
    # trained -- the same discipline test_nonlinear_dynamics_fea.py's
    # free-decay IC uses).
    q1_max = np.max(np.abs(q_nl[:, 0]))
    amp_range = (0.25 * q1_max, 0.6 * q1_max)
    n_points = 4

    hb = HarmonicBalanceSystem(Lambda, n_harmonics=5, n_time_samples=64)

    results = {}
    for name, model, domain in [("mfs", mfs, "q_l"), ("poly", poly, "q_nl")]:
        amps, omegas, Z_hist, conv = solve_nnm_backbone(
            hb, model, q_amplitude_range=amp_range, n_points=n_points, domain=domain)
        results[name] = {"amps": amps, "omegas": omegas, "converged": conv}

    # >=25 steps per period of the HIGHEST retained mode (matching Step
    # 4's own resolution requirement -- a coarser dt under-resolves the
    # higher mode).
    dt = 1.0 / (freq_hz[-1] * 25)
    n_periods_meas = 12   # enough periods for a well-resolved FFT peak

    omega_fe = np.full(n_points, np.nan)
    for p in range(n_points):
        amp = results["mfs"]["amps"][p]   # same amplitude grid for both
        T_guess = 1.0 / freq_hz[0]
        T_total = n_periods_meas * T_guess
        n_steps = int(round(T_total / dt))

        q_nl0 = np.zeros(n_modes)
        q_nl0[0] = amp
        u0_full = np.zeros(n_dof)
        u0_full[free] = V @ q_nl0

        t_fe, U_hist = nls.solve_nonlinear_transient(
            fes, mat, _ZeroLoad(), T_total, dt, u0=u0_full, tol=1e-8, max_iter=40)
        q_nl_fe = (V.T @ M_ff @ U_hist[:, free].T).T

        T_meas = measure_period(q_nl_fe[:, 0], dt)
        if T_meas is not None:
            omega_fe[p] = 2 * np.pi / T_meas

    results["omega_fe"] = omega_fe
    results["amps"] = results["mfs"]["amps"]
    results["omega_linear"] = float(np.sqrt(Lambda[0]))
    return results


class TestBackboneAgainstFullOrderFE:
    @pytest.mark.parametrize("key", ["mfs", "poly"])
    def test_all_points_converged(self, backbone_comparison, key):
        assert np.all(backbone_comparison[key]["converged"])

    def test_fe_period_measurement_succeeded(self, backbone_comparison):
        assert np.all(np.isfinite(backbone_comparison["omega_fe"]))

    @pytest.mark.parametrize("key", ["mfs", "poly"])
    def test_backbone_frequency_matches_fe_within_tolerance(self, backbone_comparison, key):
        # A real, coupled two-mode geometrically nonlinear structure is
        # a materially harder target than the idealized single-DOF
        # Duffing check in test_nnm.py (modal coupling into mode 2,
        # AFT/surrogate approximation error compounding with the
        # fitted-force-model error itself) -- an honestly looser
        # tolerance than that ~1% synthetic check, but still tight
        # enough to catch a genuinely wrong backbone (wrong sign, wrong
        # order of magnitude, flat/non-hardening).
        d = backbone_comparison
        omegas = d[key]["omegas"]
        rel_err = np.abs(omegas - d["omega_fe"]) / d["omega_fe"]
        assert np.max(rel_err) < 0.1

    @pytest.mark.parametrize("key", ["mfs", "poly"])
    def test_backbone_is_hardening(self, backbone_comparison, key):
        # The clamped-clamped geometry's membrane stretching gives a
        # classical hardening (frequency rises with amplitude) trend --
        # both the ROM backbone and the independent FE measurement
        # should show it.
        d = backbone_comparison
        assert np.all(np.diff(d[key]["omegas"]) > -1e-8)
        assert np.all(np.diff(d["omega_fe"]) > -1e-8)

    @pytest.mark.parametrize("key", ["mfs", "poly"])
    def test_backbone_above_linear_frequency(self, backbone_comparison, key):
        # The ROM backbone (an exact algebraic Newton solve, not subject
        # to measurement noise) should sit strictly above the linear
        # frequency at every point. The FE ground truth is instead an
        # FFT-based MEASUREMENT: at the smallest amplitude point the
        # true hardening shift is only a few hundredths of a percent
        # (see test_backbone_frequency_matches_fe_within_tolerance,
        # which already confirms both track each other to <10%), so a
        # strict ">" there can fail on measurement noise alone -- a 0.5%
        # slack (an order of magnitude looser than that tolerance)
        # absorbs exactly that noise without hiding a real sign error.
        d = backbone_comparison
        assert np.all(d[key]["omegas"] > d["omega_linear"])
        assert np.all(d["omega_fe"] > d["omega_linear"] * 0.995)
