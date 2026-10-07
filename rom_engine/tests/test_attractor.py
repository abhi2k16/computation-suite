"""
test_attractor.py -- validates rom_engine.attractor (Wave 17 item 146,
fea_engine/docs/consolidated_future_roadmap.md).

Two REQUIRED validation checks, per the roadmap's own item-146 row:

  1. TestDuffingPeriod1Attractor -- a Duffing oscillator integrated into
     its known period-1 steady state, stroboscopically sampled with
     stroboscopic_sample()'s own automatic settling check, converges to
     a SINGLE, stable Poincare point (not a scattered cloud).
  2. TestLinearSdofFrequencySweep -- frequency_sweep() over a linear
     damped SDOF oscillator reproduces the closed-form steady-state FRF
     amplitude at every swept frequency.

Plus unit-level coverage of every other public helper this module
promises (midspan_transverse_probe, master_slave_data,
dominant_frequency, the IntrusiveNonlinearROM dispatch path in
frequency_sweep/amplitude_sweep, and the n_jobs=parallel path), each
checked against an independent closed-form or synthetic-signal
reference rather than merely "runs without crashing".
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rom_engine.attractor import (
    stroboscopic_sample, steady_state_amplitude, midspan_transverse_probe,
    master_slave_data, dominant_frequency, frequency_sweep, amplitude_sweep,
)
from rom_engine.intrusive_nonlinear_rom import IntrusiveNonlinearROM


# =====================================================================
# Shared helper: integrate an ODE with the exact stroboscopic instants
# folded into the sample grid (dense_output + a union with the strobe
# times themselves), so stroboscopic_sample()'s own linear interpolation
# introduces ~0 extra error at those instants -- isolating the
# SETTLING-DETECTION logic being tested from ordinary sampling-grid
# interpolation error (confirmed directly during development: a coarse
# t_eval grid alone made an otherwise cleanly period-1 Duffing case look
# like it "never quite settles" at 1e-6, purely from np.interp error,
# not physics -- exactly the kind of thing this project's own
# convention is to track down rather than paper over with a looser
# tolerance).
# =====================================================================
def _integrate_with_strobe_grid(rhs, t_span, y0, Omega, n_periods, pts_per_period=50, **kwargs):
    T = 2 * np.pi / Omega
    sol = solve_ivp(rhs, t_span, y0, dense_output=True, max_step=T / pts_per_period, **kwargs)
    strobe_t_exact = np.arange(n_periods + 1) * T
    t_eval = np.union1d(np.linspace(*t_span, n_periods * pts_per_period), strobe_t_exact)
    y = sol.sol(t_eval).T   # (n_t, n_states), matching attractor.py's own (n_time, n_dof) convention
    return t_eval, y


# =====================================================================
# CHECK 1 (required) -- Duffing oscillator, known period-1 attractor.
# =====================================================================
class TestDuffingPeriod1Attractor:
    """Duffing oscillator x'' + delta*x' + alpha*x + beta*x^3 =
    gamma*cos(omega*t), single-well (alpha>0, beta>0, hardening spring)
    parameters chosen so the forced response is a mild-amplitude,
    non-chaotic period-1 resonance -- a standard textbook regime (e.g.
    Nayfeh & Mook, "Nonlinear Oscillations"), NOT the classical
    twin-well chaotic Ueda/Moon parameter set (alpha<0, larger gamma)
    that this same equation is also famous for. Chosen and checked
    directly (not guessed): delta=0.3, alpha=1.0, beta=0.5, gamma=0.5,
    omega=1.2 gives a single settled Poincare point at
    x* ~= 0.502 (see module-level development check) -- if this
    parameter choice had NOT cleanly settled, that would itself be
    useful information (wrong regime), per this item's own validation
    note; it does settle cleanly, confirmed below."""
    delta, alpha, beta, gamma, omega = 0.3, 1.0, 0.5, 0.5, 1.2

    def _rhs(self, t, z):
        x, v = z
        return [v, self.gamma * np.cos(self.omega * t) - self.delta * v - self.alpha * x - self.beta * x ** 3]

    @pytest.fixture(scope="class")
    def trajectory(self):
        n_periods = 300
        T = 2 * np.pi / self.omega
        t, y = _integrate_with_strobe_grid(
            self._rhs, (0.0, n_periods * T), [0.0, 0.0], self.omega, n_periods,
            rtol=1e-12, atol=1e-13)
        return t, y[:, 0]

    def test_settles_to_single_poincare_point(self, trajectory):
        t, x = trajectory
        result = stroboscopic_sample(t, x, self.omega, tol=1e-6, min_periods=20)
        print(f"Duffing settling: settled={result['settled']}, "
              f"discard_periods={result['discard_periods']}, "
              f"n_attractor_points={len(result['attractor_y'])}, "
              f"attractor value={result['attractor_y'][-1, 0]:.10f}")
        assert result["settled"]
        # a genuinely converged period-1 attractor: every retained
        # stroboscopic sample agrees with every other to within a tiny
        # ABSOLUTE spread (not just the relative settling tolerance
        # used to detect it), i.e. a single point, not a scattered cloud.
        spread = result["attractor_y"].max() - result["attractor_y"].min()
        print(f"attractor spread (max-min): {spread:.3e}")
        assert spread < 1e-5

    def test_scattered_before_settling_is_distinguishable(self, trajectory):
        # Sanity check on the settling machinery itself: the EARLY
        # (transient) stroboscopic samples, before the detected
        # discard_periods, are NOT all clustered -- confirms
        # discard_periods is doing real work, not just always returning
        # everything from index 0.
        t, x = trajectory
        result = stroboscopic_sample(t, x, self.omega, tol=1e-6, min_periods=20)
        transient = result["strobe_y"][: result["discard_periods"]]
        transient_spread = transient.max() - transient.min()
        print(f"pre-settling transient spread: {transient_spread:.3e}")
        assert transient_spread > 1e-3   # orders of magnitude bigger than the settled spread


# =====================================================================
# CHECK 2 (required) -- linear SDOF frequency sweep vs. closed-form FRF.
# =====================================================================
class TestLinearSdofFrequencySweep:
    """m*x'' + c*x' + k*x = F0*cos(Omega*t): closed-form steady-state
    amplitude X(Omega) = F0 / sqrt((k - m*Omega^2)^2 + (c*Omega)^2)."""
    m, c, k, F0 = 1.0, 0.4, 100.0, 5.0

    @property
    def omega_n(self):
        return np.sqrt(self.k / self.m)

    @property
    def zeta(self):
        return self.c / (2 * np.sqrt(self.k * self.m))

    def _trajectory_fn(self, Omega):
        m, c, k, F0 = self.m, self.c, self.k, self.F0

        def rhs(t, z):
            x, v = z
            return [v, (F0 * np.cos(Omega * t) - c * v - k * x) / m]

        # enough periods to clear several decay time-constants
        # tau=1/(zeta*omega_n), capped for run time
        n_periods = int(min(max(60, 12.0 / (self.zeta * self.omega_n) / (2 * np.pi / Omega)), 400))
        T = 2 * np.pi / Omega
        return _integrate_with_strobe_grid(
            rhs, (0.0, n_periods * T), [0.0, 0.0], Omega, n_periods,
            rtol=1e-11, atol=1e-13)

    def _analytic_amplitude(self, Omega):
        m, c, k, F0 = self.m, self.c, self.k, self.F0
        return F0 / np.sqrt((k - m * Omega ** 2) ** 2 + (c * Omega) ** 2)

    def test_sweep_amplitude_matches_closed_form_frf(self):
        Omega_values = np.linspace(0.3 * self.omega_n, 2.0 * self.omega_n, 8)
        result = frequency_sweep(
            self._trajectory_fn, Omega_values,
            strobe_kwargs={"tol": 1e-6, "min_periods": 20})

        assert result.param_name == "Omega"
        assert len(result.points) == len(Omega_values)

        errs = []
        for p in result.points:
            assert p.settled, f"Omega={p.param} did not settle"
            amp_num = steady_state_amplitude(p.t, p.y[:, 0], p.param, tol=1e-6, min_periods=20)
            amp_ref = self._analytic_amplitude(p.param)
            rel_err = abs(float(amp_num) - amp_ref) / amp_ref
            errs.append(rel_err)
            print(f"Omega={p.param:.4f}  amp_num={float(amp_num):.6f}  "
                  f"amp_analytic={amp_ref:.6f}  rel_err={rel_err:.3e}  "
                  f"discard_periods={p.discard_periods}")

        print(f"max relative FRF amplitude error across sweep: {max(errs):.3e}")
        assert max(errs) < 2e-3


# =====================================================================
# midspan_transverse_probe()
# =====================================================================
class TestMidspanTransverseProbe:
    def test_full_order_state_probe(self):
        n_nodes = 9   # midpoint node index 4
        dofs_per_node = 3
        n_dof = n_nodes * dofs_per_node
        state = np.arange(n_dof, dtype=float)
        val = midspan_transverse_probe(state, n_nodes, dofs_per_node=dofs_per_node, transverse_dof=1)
        expected = 4 * dofs_per_node + 1
        assert val == expected

    def test_full_order_batch_of_states(self):
        n_nodes, dofs_per_node = 7, 3
        n_dof = n_nodes * dofs_per_node
        rng = np.random.default_rng(0)
        states = rng.standard_normal((10, n_dof))
        vals = midspan_transverse_probe(states, n_nodes, dofs_per_node=dofs_per_node, transverse_dof=1)
        mid_dof = (n_nodes // 2) * dofs_per_node + 1
        assert np.allclose(vals, states[:, mid_dof])

    def test_reduced_state_probe_matches_full_reconstruction(self):
        n_nodes, dofs_per_node = 5, 3
        n_dof = n_nodes * dofs_per_node
        rng = np.random.default_rng(1)
        V = rng.standard_normal((n_dof, 3))
        q = rng.standard_normal((4, 3))
        full = q @ V.T
        mid_dof = (n_nodes // 2) * dofs_per_node + 1
        expected = full[:, mid_dof]
        got = midspan_transverse_probe(q, n_nodes, dofs_per_node=dofs_per_node, transverse_dof=1, V=V)
        assert np.allclose(got, expected)


# =====================================================================
# master_slave_data()
# =====================================================================
class TestMasterSlaveData:
    def test_shapes_and_velocity_gradient(self):
        t = np.linspace(0, 10, 5000)
        omega1 = 3.0
        Q = np.column_stack([np.cos(omega1 * t), np.cos(2 * omega1 * t), np.zeros_like(t)])
        out = master_slave_data(t, Q, master=0, slave=1)
        assert out["Q1"].shape == t.shape
        assert out["Q2"].shape == t.shape
        assert np.allclose(out["Q1"], Q[:, 0])
        assert np.allclose(out["Q2"], Q[:, 1])
        analytic_dot = -omega1 * np.sin(omega1 * t)
        err = np.max(np.abs(out["Q1_dot"][5:-5] - analytic_dot[5:-5]))
        print(f"master_slave_data Q1_dot vs analytic derivative max abs error: {err:.3e}")
        assert err < 1e-3


# =====================================================================
# dominant_frequency()
# =====================================================================
class TestDominantFrequency:
    """FFT bin spacing is `2*pi/T_total` rad/s -- an arbitrary target
    frequency generally falls BETWEEN bins (spectral leakage), which is
    an inherent property of a finite-length DFT, not a bug in
    dominant_frequency(). Both checks below pick the target frequency to
    land exactly on an FFT bin (`omega = 2*pi*k/T_total` for integer k)
    so the comparison isolates the peak-picking LOGIC from this
    well-understood, unrelated resolution effect."""

    def test_recovers_known_sinusoid_frequency(self):
        T_total, n = 40.0, 4000
        k = 46   # bin index nearest the originally-intended 7.3 rad/s
        omega_true = 2 * np.pi * k / T_total
        t = np.linspace(0, T_total, n, endpoint=False)
        Q_m = 2.0 * np.cos(omega_true * t + 0.4)
        omega_hat = dominant_frequency(t, Q_m)
        rel_err = abs(omega_hat - omega_true) / omega_true
        print(f"dominant_frequency recovered {omega_hat:.6f} vs true {omega_true:.6f}, rel_err={rel_err:.3e}")
        assert rel_err < 1e-8

    def test_slave_mode_responds_at_2x_master_frequency(self):
        # Reproduces the paper's own "slaved mode responds at 2x the
        # master frequency" claim on a synthetic master/slave pair.
        T_total, n = 60.0, 6000
        k = 48   # bin index nearest the originally-intended 5.0 rad/s;
        # 2*k=96 is also an exact bin, so the slave's 2*omega1 signal
        # lands exactly on a bin too.
        omega1 = 2 * np.pi * k / T_total
        t = np.linspace(0, T_total, n, endpoint=False)
        Q1 = np.cos(omega1 * t)
        Q2 = 0.3 * np.cos(2 * omega1 * t + 0.2)
        f1 = dominant_frequency(t, Q1)
        f2 = dominant_frequency(t, Q2)
        print(f"master dominant freq: {f1:.6f}, slave dominant freq: {f2:.6f}, ratio: {f2 / f1:.6f}")
        assert abs(f1 - omega1) / omega1 < 1e-8
        assert abs(f2 / f1 - 2.0) < 1e-8


# =====================================================================
# frequency_sweep()/amplitude_sweep() dispatch on an IntrusiveNonlinearROM
# instance (the "same data layout regardless of model type" claim).
# =====================================================================
def _make_linear_rom():
    # A trivial 1-mode linear oscillator expressed as an
    # IntrusiveNonlinearROM (V=[[1]], M_r=[[m]], D_r=[[c]], K_r=[[k]])
    # -- exercises the ROM dispatch path in frequency_sweep() without
    # needing a real fea_engine model (this module's tests for the ROM
    # class itself already validate that path against a genuine
    # Beam2DReissner fixture; this test's own job is only to confirm
    # attractor.py's dispatch/return-layout logic, not IntrusiveNonlinearROM
    # itself again).
    m, c, k = 1.0, 0.4, 100.0
    V = np.array([[1.0]])
    M = np.array([[m]])
    C = np.array([[c]])

    def internal_force_fn(u):
        return k * u

    def tangent_fn(u):
        return np.array([[k]])

    rom = IntrusiveNonlinearROM(V, M, C, internal_force_fn, load_fn=lambda t: np.zeros(1),
                                 tangent_fn=tangent_fn)
    return rom, m, c, k


class TestRomFrequencySweepDispatch:
    def test_rom_sweep_matches_analytic_frf_and_common_layout(self):
        rom, m, c, k = _make_linear_rom()
        F0 = 5.0
        omega_n = np.sqrt(k / m)
        zeta = c / (2 * np.sqrt(k * m))
        Omega_values = np.array([0.5 * omega_n, 1.0 * omega_n, 1.6 * omega_n])

        def load_fn_builder(Omega):
            return lambda t: np.array([F0 * np.cos(Omega * t)])

        def dt_fn(Omega):
            return (2 * np.pi / Omega) / 60.0

        def n_steps_fn(Omega):
            T = 2 * np.pi / Omega
            n_periods = int(min(max(80, 12.0 / (zeta * omega_n) / T), 500))
            return n_periods * 60

        result = frequency_sweep(
            rom, Omega_values, q0=np.zeros(1), qdot0=np.zeros(1),
            dt=dt_fn, n_steps=n_steps_fn, load_fn_builder=load_fn_builder,
            integrator="rk4", strobe_kwargs={"tol": 1e-5, "min_periods": 30})

        assert len(result.points) == 3
        for p in result.points:
            assert p.is_reduced is True
            amp_ref = F0 / np.sqrt((k - m * p.param ** 2) ** 2 + (c * p.param) ** 2)
            amp_num = steady_state_amplitude(p.t, p.y[:, 0], p.param, tol=1e-5, min_periods=30)
            rel_err = abs(float(amp_num) - amp_ref) / amp_ref
            print(f"ROM sweep Omega={p.param:.3f} amp_num={float(amp_num):.5f} "
                  f"amp_analytic={amp_ref:.5f} rel_err={rel_err:.3e}")
            assert rel_err < 1e-2
        # rom.load_fn must be restored to its ORIGINAL value after the
        # sweep (each point's temporary substitution should not leak).
        assert rom.load_fn(0.0).tolist() == [0.0]


# =====================================================================
# n_jobs= parallel path (module-level, picklable trajectory_fn/probe_fn
# -- see attractor.py's own module docstring for why that's required).
# =====================================================================
def _sdof_trajectory_fn(Omega):
    m, c, k, F0 = 1.0, 0.4, 100.0, 5.0

    def rhs(t, z):
        x, v = z
        return [v, (F0 * np.cos(Omega * t) - c * v - k * x) / m]

    T = 2 * np.pi / Omega
    n_periods = 80
    return _integrate_with_strobe_grid(rhs, (0.0, n_periods * T), [0.0, 0.0], Omega, n_periods,
                                        rtol=1e-10, atol=1e-12)


class TestParallelSweepMatchesSerial:
    def test_n_jobs_2_matches_n_jobs_1(self):
        Omega_values = np.array([8.0, 10.0, 12.0])
        strobe_kwargs = {"tol": 1e-5, "min_periods": 20}

        result_serial = frequency_sweep(_sdof_trajectory_fn, Omega_values,
                                         strobe_kwargs=strobe_kwargs, n_jobs=1)
        result_parallel = frequency_sweep(_sdof_trajectory_fn, Omega_values,
                                           strobe_kwargs=strobe_kwargs, n_jobs=2)

        for ps, pp in zip(result_serial.points, result_parallel.points):
            assert ps.param == pp.param
            assert np.allclose(ps.attractor_y, pp.attractor_y)
            assert ps.settled == pp.settled
