"""
Tests for nonlinear_dynamics.py -- integrate_newmark_surrogate().

Every check has a synthetic ground truth with a known closed form or an
independent high-accuracy scipy.integrate.solve_ivp reference, so "is
the Newmark bookkeeping/domain-translation/correction logic right" is
checkable without any FE package. Real fea_engine validation lives in
tests/test_nonlinear_dynamics_fea.py (Section 9's flat-beam dynamic
comparison), not here.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rom_engine.nonlinear_dynamics import integrate_newmark_surrogate


class _ZeroForceModel:
    """F_nl == 0 everywhere -- the pure LINEAR limit, domain-agnostic."""
    def predict(self, q):
        q = np.asarray(q, dtype=float)
        return np.zeros_like(q)

    def jacobian(self, q):
        return np.zeros((len(q), len(q)))


class _LinearQnlForceModel:
    """F_nl(q_nl) = k_extra * q_nl -- an exactly LINEAR augmentation of
    the modal stiffness, expressed directly in the q_nl domain
    (domain="q_nl", matching PolynomialModalROM's own native domain)."""
    def __init__(self, k_extra):
        self.k_extra = np.asarray(k_extra, dtype=float)

    def predict(self, q):
        return self.k_extra * np.asarray(q, dtype=float)

    def jacobian(self, q):
        return np.diag(self.k_extra)


class _LinearQlForceModel:
    """F_nl(q_l) = k_extra * q_l -- an exactly LINEAR augmentation
    expressed in the q_l domain (domain="q_l", MultiFidelitySurrogate's
    own native domain), so the TRUE force as a function of q_nl is
    itself nonlinear-looking algebra (q_l = q_nl/(1 - k_extra/Lambda))
    even though the underlying physics is still linear."""
    def __init__(self, k_extra):
        self.k_extra = np.asarray(k_extra, dtype=float)

    def predict(self, q_l):
        return self.k_extra * np.asarray(q_l, dtype=float)

    def jacobian(self, q_l):
        return np.diag(self.k_extra)


class _CubicQnlForceModel:
    """F_nl(q_nl) = k3 * q_nl^3 -- a genuinely nonlinear (cubic-spring)
    force, domain="q_nl"."""
    def __init__(self, k3):
        self.k3 = np.asarray(k3, dtype=float)

    def predict(self, q):
        return self.k3 * np.asarray(q, dtype=float) ** 3

    def jacobian(self, q):
        return np.diag(3 * self.k3 * np.asarray(q, dtype=float) ** 2)


def _exact_linear_newmark(Lambda_eff, C_r, q0, qdot0, dt, n_steps, beta=0.25, gamma=0.5):
    """Independent, hand-rolled linear Newmark reference (no
    force_model/correction machinery at all) -- used to prove the
    shared Newmark predictor/corrector bookkeeping in
    integrate_newmark_surrogate() is correct, isolated from any
    correction-mode logic."""
    n = len(Lambda_eff)
    a0c = 1 / (beta * dt ** 2); a1c = gamma / (beta * dt); a2c = 1 / (beta * dt)
    a3c = 1 / (2 * beta) - 1; a4c = gamma / beta - 1; a5c = dt / 2 * (gamma / beta - 2)
    a6c = dt * (1 - gamma); a7c = dt * gamma
    Keff = np.diag(Lambda_eff) + a0c * np.eye(n) + a1c * np.diag(C_r)
    q, qdot = np.asarray(q0, dtype=float).copy(), np.asarray(qdot0, dtype=float).copy()
    qddot = -C_r * qdot - Lambda_eff * q
    hist = np.zeros((n_steps + 1, n))
    hist[0] = q
    for step in range(n_steps):
        rhs = a0c * q + a2c * qdot + a3c * qddot + C_r * (a1c * q + a4c * qdot + a5c * qddot)
        q_new = np.linalg.solve(Keff, rhs)
        qddot_new = a0c * (q_new - q) - a2c * qdot - a3c * qddot
        qdot_new = qdot + a6c * qddot + a7c * qddot_new
        q, qdot, qddot = q_new, qdot_new, qddot_new
        hist[step + 1] = q
    return hist


class TestLinearLimit:
    """domain="q_nl" with a zero or exactly-linear force model should
    reduce EXACTLY to plain linear Newmark, independent of `correction`
    -- the fixed_point mode should reach the reference to machine
    precision (it fully converges within each step even for a linear
    residual); none/sign_deviation carry a small, bounded one-step-lag
    error, which is the expected cost of never solving a residual to
    convergence, not a bug."""

    @pytest.mark.parametrize("correction", ["none", "fixed_point", "sign_deviation", "newton"])
    def test_zero_force_model_matches_pure_linear_newmark(self, correction):
        n = 2
        Lambda = np.array([100.0, 400.0])
        C_r = np.array([0.5, 0.8])
        q0 = np.array([0.01, -0.005])
        qdot0 = np.zeros(n)
        dt, n_steps = 0.001, 200

        ref = _exact_linear_newmark(Lambda, C_r, q0, qdot0, dt, n_steps)
        t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
            Lambda, C_r, _ZeroForceModel(), F_ext=0.0, q0=q0, qdot0=qdot0,
            dt=dt, n_steps=n_steps, domain="q_nl", correction=correction)

        assert np.allclose(q_nl_hist, ref, atol=1e-10)
        assert np.array_equal(q_nl_hist, q_l_hist)   # domain="q_nl" -> no translation
        assert np.allclose(F_nl_hist, 0.0)

    def test_linear_qnl_force_model_newton_matches_effective_stiffness_exactly(self):
        # correction="newton" fully converges the residual every step
        # (same as fixed_point does for this linear case), so it too
        # should match the exact effective-stiffness Newmark reference
        # to near machine precision -- in fact a SINGLE Newton iteration
        # suffices here (the residual is exactly linear in q).
        n = 2
        Lambda = np.array([100.0, 400.0])
        k_extra = np.array([5.0, 8.0])
        C_r = np.array([0.5, 0.8])
        q0 = np.array([0.01, -0.005])
        qdot0 = np.zeros(n)
        dt, n_steps = 0.001, 500

        ref = _exact_linear_newmark(Lambda + k_extra, C_r, q0, qdot0, dt, n_steps)
        t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
            Lambda, C_r, _LinearQnlForceModel(k_extra), F_ext=0.0, q0=q0, qdot0=qdot0,
            dt=dt, n_steps=n_steps, domain="q_nl", correction="newton")
        assert np.max(np.abs(q_nl_hist - ref)) < 1e-10

    def test_linear_qnl_force_model_fixed_point_matches_effective_stiffness_exactly(self):
        n = 2
        Lambda = np.array([100.0, 400.0])
        k_extra = np.array([5.0, 8.0])
        C_r = np.array([0.5, 0.8])
        q0 = np.array([0.01, -0.005])
        qdot0 = np.zeros(n)
        dt, n_steps = 0.001, 500

        ref = _exact_linear_newmark(Lambda + k_extra, C_r, q0, qdot0, dt, n_steps)
        t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
            Lambda, C_r, _LinearQnlForceModel(k_extra), F_ext=0.0, q0=q0, qdot0=qdot0,
            dt=dt, n_steps=n_steps, domain="q_nl", correction="fixed_point")

        # fixed_point Picard-iterates to convergence every step, so for
        # a genuinely LINEAR force model it should match the exact
        # effective-stiffness Newmark reference to near machine precision
        assert np.max(np.abs(q_nl_hist - ref)) < 1e-12

    @pytest.mark.parametrize("correction", ["none", "fixed_point", "sign_deviation", "newton"])
    def test_linear_qnl_force_model_stays_close_to_effective_stiffness(self, correction):
        n = 2
        Lambda = np.array([100.0, 400.0])
        k_extra = np.array([5.0, 8.0])
        C_r = np.array([0.5, 0.8])
        q0 = np.array([0.01, -0.005])
        qdot0 = np.zeros(n)
        dt, n_steps = 0.001, 500

        ref = _exact_linear_newmark(Lambda + k_extra, C_r, q0, qdot0, dt, n_steps)
        t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
            Lambda, C_r, _LinearQnlForceModel(k_extra), F_ext=0.0, q0=q0, qdot0=qdot0,
            dt=dt, n_steps=n_steps, domain="q_nl", correction=correction)
        rel = np.max(np.abs(q_nl_hist - ref)) / np.max(np.abs(ref))
        assert rel < 1e-3


class TestQlDomainTranslation:
    """A force model defined in the q_l domain (MultiFidelitySurrogate's
    own convention) should still track the TRUE physical q_nl response,
    once the Eq. 9-11 translation is inverted analytically for a linear
    F_nl(q_l) = k_extra*q_l case and checked against an independent
    scipy ODE integration."""

    def test_linear_ql_force_model_tracks_scipy_ode_reference(self):
        n = 2
        Lambda = np.array([100.0, 400.0])
        k_extra = np.array([5.0, 8.0])
        C_r = np.array([0.5, 0.8])
        q0 = np.array([0.01, -0.005])
        qdot0 = np.zeros(n)
        dt, n_steps = 0.0005, 1000
        T = dt * n_steps

        # Closed-form true force as a function of q_nl: q_l = q_nl/(1 - k_extra/Lambda)
        # (solving q_l = q_nl + k_extra*q_l/Lambda for q_l), so
        # Fnl_true(q_nl) = k_extra * q_nl / (1 - k_extra/Lambda) --
        # an exactly linear relation once inverted, giving an effective
        # stiffness this test can check against an independent ODE solve.
        Lambda_eff = Lambda + k_extra / (1 - k_extra / Lambda)

        def rhs(t, y):
            q, qd = y[:n], y[n:]
            qdd = -C_r * qd - Lambda_eff * q
            return np.concatenate([qd, qdd])

        y0 = np.concatenate([q0, qdot0])
        sol = solve_ivp(rhs, [0, T], y0, t_eval=np.arange(n_steps + 1) * dt,
                         rtol=1e-12, atol=1e-14)
        ref_q = sol.y[:n].T

        for correction in ["none", "fixed_point", "sign_deviation"]:
            t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
                Lambda, C_r, _LinearQlForceModel(k_extra), F_ext=0.0, q0=q0, qdot0=qdot0,
                dt=dt, n_steps=n_steps, domain="q_l", correction=correction)
            rel = np.max(np.abs(q_nl_hist - ref_q)) / np.max(np.abs(ref_q))
            assert rel < 1e-3, f"correction={correction} rel_err={rel}"

    def test_q_l_hist_differs_from_q_nl_hist_when_domain_is_q_l(self):
        n = 1
        Lambda = np.array([100.0])
        k_extra = np.array([5.0])
        C_r = np.array([0.5])
        q0 = np.array([0.02])
        qdot0 = np.zeros(n)
        t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
            Lambda, C_r, _LinearQlForceModel(k_extra), F_ext=0.0, q0=q0, qdot0=qdot0,
            dt=0.001, n_steps=50, domain="q_l", correction="fixed_point")
        assert not np.allclose(q_nl_hist, q_l_hist)


class TestNonlinearCubicSpring:
    """A genuinely nonlinear (cubic) force model, checked against an
    independent scipy ODE solve -- confirms all three correction modes
    remain stable and reasonably accurate for real nonlinear dynamics,
    not just the linear-limit special cases above."""

    @pytest.mark.parametrize("correction", ["none", "fixed_point", "sign_deviation", "newton"])
    def test_tracks_scipy_ode_reference_without_blowing_up(self, correction):
        n = 1
        Lambda = np.array([100.0])
        C_r = np.array([0.2])
        k3 = np.array([2000.0])
        q0 = np.array([0.05])
        qdot0 = np.array([0.0])
        dt, n_steps = 0.001, 2000
        T = dt * n_steps

        def rhs(t, y):
            q, qd = y[:n], y[n:]
            qdd = -C_r * qd - Lambda * q - k3 * q ** 3
            return np.concatenate([qd, qdd])

        y0 = np.concatenate([q0, qdot0])
        sol = solve_ivp(rhs, [0, T], y0, t_eval=np.arange(n_steps + 1) * dt,
                         rtol=1e-11, atol=1e-13)
        ref_q = sol.y[:n].T

        t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
            Lambda, C_r, _CubicQnlForceModel(k3), F_ext=0.0, q0=q0, qdot0=qdot0,
            dt=dt, n_steps=n_steps, domain="q_nl", correction=correction)

        assert np.all(np.isfinite(q_nl_hist))
        assert np.max(np.abs(q_nl_hist)) < 10 * np.max(np.abs(ref_q))   # no blow-up
        rel = np.max(np.abs(q_nl_hist - ref_q)) / np.max(np.abs(ref_q))
        assert rel < 0.01

    def test_fixed_point_is_at_least_as_accurate_as_none(self):
        # the whole motivation for fixed_point (per the module docstring
        # and the validated prototype it was ported from) is that "none"
        # can accumulate lag error a few Picard iterations would remove
        n = 1
        Lambda = np.array([100.0])
        C_r = np.array([0.2])
        k3 = np.array([2000.0])
        q0 = np.array([0.05])
        qdot0 = np.array([0.0])
        dt, n_steps = 0.001, 2000
        T = dt * n_steps

        def rhs(t, y):
            q, qd = y[:n], y[n:]
            qdd = -C_r * qd - Lambda * q - k3 * q ** 3
            return np.concatenate([qd, qdd])

        y0 = np.concatenate([q0, qdot0])
        sol = solve_ivp(rhs, [0, T], y0, t_eval=np.arange(n_steps + 1) * dt,
                         rtol=1e-11, atol=1e-13)
        ref_q = sol.y[:n].T

        errs = {}
        for correction in ["none", "fixed_point"]:
            _, q_nl_hist, _, _ = integrate_newmark_surrogate(
                Lambda, C_r, _CubicQnlForceModel(k3), F_ext=0.0, q0=q0, qdot0=qdot0,
                dt=dt, n_steps=n_steps, domain="q_nl", correction=correction)
            errs[correction] = np.max(np.abs(q_nl_hist - ref_q))
        assert errs["fixed_point"] <= errs["none"]


class TestInputValidation:
    def test_rejects_unknown_domain(self):
        with pytest.raises(ValueError):
            integrate_newmark_surrogate(
                np.array([1.0]), np.array([0.1]), _ZeroForceModel(), 0.0,
                np.array([0.0]), np.array([0.0]), 0.01, 10, domain="bogus")

    def test_rejects_unknown_correction(self):
        with pytest.raises(ValueError):
            integrate_newmark_surrogate(
                np.array([1.0]), np.array([0.1]), _ZeroForceModel(), 0.0,
                np.array([0.0]), np.array([0.0]), 0.01, 10, correction="bogus")

    def test_accepts_full_damping_matrix(self):
        n = 2
        Lambda = np.array([100.0, 400.0])
        C_full = np.array([[0.5, 0.05], [0.05, 0.8]])   # non-diagonal
        q0 = np.array([0.01, -0.005])
        qdot0 = np.zeros(n)
        t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
            Lambda, C_full, _ZeroForceModel(), F_ext=0.0, q0=q0, qdot0=qdot0,
            dt=0.001, n_steps=50, domain="q_nl", correction="fixed_point")
        assert np.all(np.isfinite(q_nl_hist))

    def test_callable_F_ext_is_honored(self):
        n = 1
        Lambda = np.array([100.0])
        C_r = np.array([0.1])
        calls = []

        def F_ext(t):
            calls.append(t)
            return np.array([1.0])

        integrate_newmark_surrogate(
            Lambda, C_r, _ZeroForceModel(), F_ext, q0=np.zeros(1), qdot0=np.zeros(1),
            dt=0.01, n_steps=5, domain="q_nl", correction="none")
        # one call for the initial consistent-acceleration setup (t=0),
        # plus exactly one per step thereafter (cached into rhs_base and
        # reused across any correction-mode's internal iterations) --
        # NOT re-called per Newton/fixed-point/correction pass
        assert len(calls) == 1 + 5


def _reference_newton_newmark(Lambda, C_r, force_model, F_ext_fn, q0, qdot0, dt, n_steps,
                               beta=0.25, gamma=0.5, tol=1e-9, max_iter=30, q_bound=None):
    """Independent, hand-rolled Newton-Newmark reference -- a direct
    transcription of the validated ad hoc prototype (ICE-ROM/validation/
    dynamic_comparison.py::integrate_ice_rom_newton, domain="q_nl" only)
    kept SEPARATE from integrate_newmark_surrogate()'s own
    correction="newton" implementation, so agreement between the two
    is a genuine cross-check of the promoted/generalized code against
    its own original source, not a tautology."""
    n = len(Lambda)
    if q_bound is None:
        q_bound = np.full(n, np.inf)
    q_bound = np.asarray(q_bound, dtype=float)
    a0c = 1 / (beta * dt ** 2); a1c = gamma / (beta * dt); a2c = 1 / (beta * dt)
    a3c = 1 / (2 * beta) - 1; a4c = gamma / beta - 1; a5c = dt / 2 * (gamma / beta - 2)
    a6c = dt * (1 - gamma); a7c = dt * gamma

    q = np.asarray(q0, dtype=float).copy()
    qdot = np.asarray(qdot0, dtype=float).copy()
    theta0 = force_model.predict(q)
    qddot = F_ext_fn(0.0) - C_r * qdot - Lambda * q - theta0

    q_hist = np.zeros((n_steps + 1, n))
    q_hist[0] = q
    Keff_lin = np.diag(Lambda) + a0c * np.eye(n) + a1c * np.diag(C_r)

    for step in range(n_steps):
        F_ext = F_ext_fn((step + 1) * dt)
        rhs_base = (F_ext + a0c * q + a2c * qdot + a3c * qddot
                    + C_r * (a1c * q + a4c * qdot + a5c * qddot))

        def residual(qv):
            return Keff_lin @ qv - rhs_base + force_model.predict(qv)

        q_trial = q.copy()
        G = residual(q_trial)
        ref = max(1.0, np.max(np.abs(rhs_base)))
        for _ in range(max_iter):
            if np.max(np.abs(G)) < tol * ref:
                break
            J = Keff_lin + force_model.jacobian(q_trial)
            dq = np.linalg.solve(J, G)
            step_scale = 1.0
            Gn = np.max(np.abs(G))
            for _ in range(30):
                q_new_trial = np.clip(q_trial - step_scale * dq, -q_bound, q_bound)
                G_new_trial = residual(q_new_trial)
                if np.max(np.abs(G_new_trial)) < Gn or step_scale < 1e-6:
                    break
                step_scale *= 0.5
            q_trial, G = q_new_trial, G_new_trial
        q_new = q_trial
        qddot_new = a0c * (q_new - q) - a2c * qdot - a3c * qddot
        qdot_new = qdot + a6c * qddot + a7c * qddot_new
        q, qdot, qddot = q_new, qdot_new, qddot_new
        q_hist[step + 1] = q
    return q_hist


class TestNewtonCorrection:
    """Wave 12 item 114 (docs/consolidated_future_roadmap.md,
    ICE-ROM/GAP_ANALYSIS.md gap #3)-specific behavior:
    correction="newton"'s domain restriction, q_bound clamp, and
    agreement with the independent hand-rolled reference above (which
    is itself a direct transcription of the validated ad hoc prototype
    this item promotes)."""

    def test_rejects_domain_q_l(self):
        with pytest.raises(ValueError, match="domain='q_nl'"):
            integrate_newmark_surrogate(
                np.array([100.0]), np.array([0.1]), _LinearQlForceModel(np.array([5.0])),
                F_ext=0.0, q0=np.array([0.0]), qdot0=np.array([0.0]),
                dt=0.001, n_steps=5, domain="q_l", correction="newton")

    def test_matches_independent_hand_rolled_reference_on_cubic_spring(self):
        n = 2
        Lambda = np.array([100.0, 300.0])
        C_r = np.array([0.3, 0.5])
        k3 = np.array([3000.0, 1500.0])
        q0 = np.array([0.03, -0.02])
        qdot0 = np.zeros(n)
        dt, n_steps = 0.001, 300

        def F_ext_fn(t):
            return np.array([5.0 * np.sin(20.0 * t), 2.0 * np.cos(15.0 * t)])

        ref = _reference_newton_newmark(
            Lambda, C_r, _CubicQnlForceModel(k3), F_ext_fn, q0, qdot0, dt, n_steps)
        t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
            Lambda, C_r, _CubicQnlForceModel(k3), F_ext_fn, q0=q0, qdot0=qdot0,
            dt=dt, n_steps=n_steps, domain="q_nl", correction="newton")

        assert np.allclose(q_nl_hist, ref, atol=1e-9, rtol=1e-8)

    def test_q_bound_clamps_every_iterate(self):
        n = 1
        Lambda = np.array([50.0])
        C_r = np.array([0.1])
        k3 = np.array([50000.0])   # deliberately stiff, to drive large trial steps
        q_bound = np.array([0.02])

        t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
            Lambda, C_r, _CubicQnlForceModel(k3), F_ext=np.array([50.0]),
            q0=np.array([0.0]), qdot0=np.array([0.0]), dt=0.001, n_steps=200,
            domain="q_nl", correction="newton", q_bound=q_bound)

        assert np.all(np.isfinite(q_nl_hist))
        assert np.all(np.abs(q_nl_hist) <= q_bound[0] + 1e-12)

    def test_more_accurate_than_lagged_corrections_for_a_stiff_polynomial_fit(self):
        # The item's own headline motivation: a stiff cubic-spring model
        # driven by a strong external force should stay CLOSER to a
        # high-accuracy scipy ODE reference under exact Newton
        # convergence than under any of the three Newton-free lagged
        # corrections, which only approximate the same per-step
        # residual rather than solving it.
        n = 1
        Lambda = np.array([100.0])
        C_r = np.array([0.2])
        k3 = np.array([8000.0])
        q0 = np.array([0.0])
        qdot0 = np.array([0.0])
        dt, n_steps = 0.001, 800
        T = dt * n_steps

        def F_ext_scalar(t):
            return 40.0 * np.sin(30.0 * t)

        def rhs(tt, y):
            q, qd = y[:n], y[n:]
            qdd = F_ext_scalar(tt) - C_r * qd - Lambda * q - k3 * q ** 3
            return np.concatenate([qd, qdd])

        y0 = np.concatenate([q0, qdot0])
        sol = solve_ivp(rhs, [0, T], y0, t_eval=np.arange(n_steps + 1) * dt,
                         rtol=1e-12, atol=1e-14)
        ref_q = sol.y[:n].T

        def F_ext(t):
            return np.array([F_ext_scalar(t)])

        errs = {}
        for correction in ["none", "fixed_point", "sign_deviation", "newton"]:
            _, q_nl_hist, _, _ = integrate_newmark_surrogate(
                Lambda, C_r, _CubicQnlForceModel(k3), F_ext, q0=q0, qdot0=qdot0,
                dt=dt, n_steps=n_steps, domain="q_nl", correction=correction)
            errs[correction] = np.max(np.abs(q_nl_hist - ref_q))

        assert errs["newton"] <= errs["none"]
        assert errs["newton"] <= errs["sign_deviation"]
        # fixed_point (4 Picard iterations) already nearly fully
        # converges the per-step residual in this mild-enough regime,
        # so the two land within noise of each other at this dt (both
        # dominated by Newmark's own O(dt^2) truncation error, not by
        # correction-scheme convergence) -- newton should still be
        # comparably accurate, not measurably worse.
        assert errs["newton"] <= 1.05 * errs["fixed_point"]
        # a healthy, non-blown-up track of the reference (this dt's own
        # O(dt^2) Newmark truncation error dominates here, not the
        # correction scheme -- ~0.15% relative is the expected floor,
        # not a bug)
        assert errs["newton"] < 1e-2 * np.max(np.abs(ref_q))


class TestDefaultCorrectionAndSignDeviationDivergence:
    """2026-09-24: the default moved from "sign_deviation" to "fixed_point".
    sign_deviation's sign(qddot) factor turns its correction into an
    anti-Picard step for decelerating modes (NonLin-HyROM Case 2 diverged
    for every mode set). A forced 1-DOF hardening cubic reproduces this
    with no FE data."""

    W = 2 * np.pi * 5.0
    DT, N_STEPS = 0.005, 200

    class _Cubic:
        def __init__(self, k3):
            self.k3 = k3

        def predict(self, q):
            return self.k3 * np.asarray(q, dtype=float) ** 3

        def jacobian(self, q):
            return np.diag(3 * self.k3 * np.asarray(q, dtype=float) ** 2)

    def _run(self, k3, F0, **kw):
        with np.errstate(all="ignore"):
            return integrate_newmark_surrogate(
                np.array([self.W ** 2]), np.zeros(1), self._Cubic(k3),
                lambda t: np.array([F0 * np.sin(2 * np.pi * 2.5 * t)]),
                np.zeros(1), np.zeros(1), self.DT, self.N_STEPS, domain="q_nl", **kw)[1][:, 0]

    def test_default_is_fixed_point(self):
        q_default = self._run(1e4, 50.0)
        q_fp = self._run(1e4, 50.0, correction="fixed_point")
        np.testing.assert_array_equal(q_default, q_fp)

    def test_fixed_point_stable_where_sign_deviation_diverges(self):
        # nonlinear stiffness ~8x linear at peak
        ref = self._run(1e5, 200.0, correction="newton")
        q_fp = self._run(1e5, 200.0, correction="fixed_point")
        q_sd = self._run(1e5, 200.0, correction="sign_deviation")
        assert np.max(np.abs(q_fp - ref)) / np.max(np.abs(ref)) < 1e-4
        assert not np.all(np.isfinite(q_sd))

    def test_sign_deviation_less_accurate_than_no_correction(self):
        # moderately nonlinear (~0.26x linear stiffness at peak): both
        # lagged schemes stay finite, but sign_deviation's correction
        # makes it WORSE than the uncorrected lag
        ref = self._run(1e4, 50.0, correction="newton")
        err = {c: np.max(np.abs(self._run(1e4, 50.0, correction=c) - ref))
               for c in ("none", "sign_deviation", "fixed_point")}
        assert err["sign_deviation"] > err["none"] > err["fixed_point"]
