# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_parameterized_latent_ode.py -- Wave 13 item 119 (fea_engine/docs/
consolidated_future_roadmap.md).

The decisive evidence for this item is the head-to-head order-of-
convergence comparison below: the CORRECT `rk4_step()`/`integrate_rk4()`
in this package versus an inline reimplementation of the
`L-NeuralODE` reference's own bug (k2/k3 evaluated at the un-advanced
state). Empirically (this file's own development, exponential decay
dy/dt=-2.5y, t in [0,1], halving n_steps 10->20->40->80):

    correct RK4 error-halving ratio: 17.8, 16.9, 16.4  (-> 4th order,
        the textbook ratio for halving dt is 2**4 = 16)
    buggy  RK4 error-halving ratio:  2.02, 2.01, 2.01  (-> 1st order,
        no better than forward Euler)

i.e. the reference implementation's bug doesn't just add a little
noise -- it silently throws away three full orders of accuracy. The
test below reproduces this exact comparison and asserts on it
directly, rather than merely checking the corrected version's error
against an arbitrary tolerance.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine.parameterized_latent_ode import (
    rk4_step, integrate_rk4, CurriculumSchedule, _HAS_TORCH,
)


# =====================================================================
# 1. Decisive: correct vs buggy RK4, order-of-convergence comparison.
# =====================================================================
def _buggy_rk4_step(func, t, dt, y, *args):
    """Reproduces the `L-NeuralODE` reference's own `rk4()` bug for
    comparison purposes only (not part of the public API): k2 and k3
    are BOTH evaluated at the un-advanced state `y`, instead of at the
    midpoint states a genuine RK4 step requires."""
    k1 = func(t, y, *args)
    k2 = func(t + 0.5 * dt, y, *args)          # BUG: should be y + 0.5*dt*k1
    k3 = func(t + 0.5 * dt, y, *args)          # BUG: should be y + 0.5*dt*k2
    k4 = func(t + dt, y + dt * k3, *args)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate(step_fn, func, y0, t, *args):
    y = np.array(y0, dtype=float)
    n = len(t)
    hist = np.zeros((n,) + y.shape)
    hist[0] = y
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        y = step_fn(func, t[i], dt, y, *args)
        hist[i + 1] = y
    return hist


def test_rk4_step_matches_manual_exponential_decay_to_machine_precision_for_one_step():
    # a single RK4 step's local truncation error is O(dt^5); over one
    # tiny step it should match the exact solution to near machine
    # precision, decisive on the formula itself (not just convergence
    # trend).
    k = 2.5
    f = lambda t, y: -k * y
    dt = 1e-3
    y1 = rk4_step(f, 0.0, dt, np.array([1.0]))
    exact = np.exp(-k * dt)
    assert abs(y1[0] - exact) < 1e-14


def test_correct_rk4_converges_at_fourth_order_while_the_reference_bug_degrades_to_first_order():
    k = 2.5
    f = lambda t, y: -k * y
    y0 = np.array([1.0])
    t_final = 1.0
    exact = y0[0] * np.exp(-k * t_final)

    errs_correct, errs_buggy = [], []
    for n_steps in (10, 20, 40, 80, 160):
        t = np.linspace(0.0, t_final, n_steps + 1)
        y_correct = _integrate(rk4_step, f, y0, t)[-1, 0]
        y_buggy = _integrate(_buggy_rk4_step, f, y0, t)[-1, 0]
        errs_correct.append(abs(y_correct - exact))
        errs_buggy.append(abs(y_buggy - exact))

    ratios_correct = [errs_correct[i - 1] / errs_correct[i] for i in range(1, len(errs_correct))]
    ratios_buggy = [errs_buggy[i - 1] / errs_buggy[i] for i in range(1, len(errs_buggy))]

    # correct RK4: error should shrink by ~16x (2**4) per dt-halving
    assert all(12.0 < r < 20.0 for r in ratios_correct)
    # the reference's own bug: error shrinks by only ~2x per halving
    # (first-order, no better than forward Euler) -- and, decisively,
    # strictly worse convergence than the corrected version at every
    # resolution tested
    assert all(1.7 < r < 2.3 for r in ratios_buggy)
    for rc, rb in zip(ratios_correct, ratios_buggy):
        assert rc > 5.0 * rb


def test_integrate_rk4_matches_scipy_solve_ivp_on_a_harmonic_oscillator():
    """Cross-checks integrate_rk4 against an INDEPENDENT reference
    integrator (scipy's own adaptive RK45), not merely against itself
    at a different resolution."""
    scipy_integrate = pytest.importorskip("scipy.integrate")
    omega = 3.0

    def f_shm(t, z):
        x, v = z[0], z[1]
        return np.array([v, -omega ** 2 * x])

    def f_shm_scipy(t, z):
        x, v = z[0], z[1]
        return [v, -omega ** 2 * x]

    z0 = np.array([1.0, 0.0])
    t_final = 2.0
    t = np.linspace(0.0, t_final, 201)

    z_ours = integrate_rk4(f_shm, z0, t)
    sol = scipy_integrate.solve_ivp(f_shm_scipy, [0.0, t_final], z0, t_eval=t,
                                     rtol=1e-10, atol=1e-12)

    assert np.max(np.abs(z_ours[:, 0] - sol.y[0])) < 1e-6


def test_integrate_rk4_handles_nonuniform_time_grid():
    # per-step dt is read directly from consecutive t entries, so a
    # non-uniform grid should still integrate correctly (checked
    # against the exact exponential-decay solution at each grid point)
    k = 1.5
    f = lambda t, y: -k * y
    t = np.sort(np.concatenate([np.linspace(0, 0.5, 8), np.linspace(0.5, 1.5, 15)]))
    t = np.unique(t)
    hist = integrate_rk4(f, np.array([2.0]), t)
    exact = 2.0 * np.exp(-k * t)
    assert np.max(np.abs(hist[:, 0] - exact)) < 1e-6


# =====================================================================
# 2. CurriculumSchedule -- pure index bookkeeping, unconditional.
# =====================================================================
def test_curriculum_schedule_starts_short_and_reaches_full_horizon():
    sched = CurriculumSchedule(n_steps=101, n_folds=10)
    assert sched.upper_index == 10
    assert not sched.is_full_horizon
    seen = [sched.upper_index]
    for _ in range(15):
        sched.advance()
        seen.append(sched.upper_index)
    assert seen[-1] == 101
    assert sched.is_full_horizon
    # monotonically non-decreasing throughout
    assert all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1))


def test_curriculum_schedule_advance_is_a_noop_once_full_horizon_reached():
    sched = CurriculumSchedule(n_steps=20, n_folds=3)
    for _ in range(20):
        sched.advance()
    assert sched.is_full_horizon
    upper_before = sched.upper_index
    changed = sched.advance()
    assert not changed
    assert sched.upper_index == upper_before == 20


def test_curriculum_schedule_maybe_advance_only_advances_below_tolerance():
    sched = CurriculumSchedule(n_steps=101, n_folds=5)
    start = sched.upper_index
    advanced_high_loss = sched.maybe_advance(loss=1.0, tol=0.1)
    assert not advanced_high_loss
    assert sched.upper_index == start
    advanced_low_loss = sched.maybe_advance(loss=0.01, tol=0.1)
    assert advanced_low_loss
    assert sched.upper_index > start


def test_curriculum_schedule_rejects_invalid_construction():
    with pytest.raises(ValueError):
        CurriculumSchedule(n_steps=1)
    with pytest.raises(ValueError):
        CurriculumSchedule(n_steps=10, n_folds=0)


# =====================================================================
# 3. torch-gated: ParameterizedODEFunc / ParameterizedLatentODE
# =====================================================================
@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
class TestParameterizedLatentODE:
    def test_ode_func_forward_shape(self):
        from rom_engine.parameterized_latent_ode import ParameterizedODEFunc
        import torch
        func = ParameterizedODEFunc(n_xi_features=1, n_vars=2, hidden_sizes=(8, 8))
        t = torch.tensor(0.3)
        xi = torch.zeros((5, 1), dtype=torch.float64)
        z = torch.zeros((5, 2), dtype=torch.float64)
        dz = func(t, xi, z)
        assert dz.shape == (5, 2)

    def test_generalizes_to_an_unseen_parameter_value(self):
        """Trains on trajectories of a parameterized Duffing-like
        cubic-spring oscillator (single mode, state z=[q, qdot]),
        dq/dt=qdot, dqdot/dt=-omega0^2*q - xi*q^3, sweeping the cubic
        stiffness xi across several TRAINING values, then evaluates on
        an UNSEEN xi strictly between two training values -- the same
        honest, non-inflated generalization bar item 117's own
        ExcitationResponseOperator test uses (R^2 > 0, not a tight
        accuracy target)."""
        from rom_engine.parameterized_latent_ode import ParameterizedLatentODE

        omega0 = 2.0
        rng = np.random.default_rng(0)

        def f_duffing(t, z, xi):
            q, qd = z[0], z[1]
            return np.array([qd, -omega0 ** 2 * q - xi * q ** 3])

        t = np.linspace(0.0, 2.0, 41)
        xi_train = np.array([1.0, 3.0, 5.0, 7.0])
        z0 = np.array([0.5, 0.0])

        z_traj_train = np.stack(
            [integrate_rk4(f_duffing, z0, t, xi) for xi in xi_train], axis=0)
        z0_train = np.tile(z0, (len(xi_train), 1))
        xi_train_col = xi_train.reshape(-1, 1)

        model = ParameterizedLatentODE(n_xi_features=1, n_vars=2,
                                        hidden_sizes=(32, 32), lr=3e-3, seed=0)
        model.fit(t, z0_train, xi_train_col, z_traj_train,
                  n_iters=400, n_folds=4, curric_tol=1e-2)

        xi_unseen = 4.0
        z_true_unseen = integrate_rk4(f_duffing, z0, t, xi_unseen)
        z_pred_unseen = model.predict(t, z0, np.array([xi_unseen]))

        ss_res = np.sum((z_pred_unseen[:, 0] - z_true_unseen[:, 0]) ** 2)
        ss_tot = np.sum((z_true_unseen[:, 0] - np.mean(z_true_unseen[:, 0])) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.0

    def test_fit_rejects_mismatched_shapes(self):
        from rom_engine.parameterized_latent_ode import ParameterizedLatentODE
        model = ParameterizedLatentODE(n_xi_features=1, n_vars=2, hidden_sizes=(4,))
        t = np.linspace(0, 1, 5)
        with pytest.raises(ValueError):
            model.fit(t, np.zeros((3, 2)), np.zeros((3, 1)), np.zeros((4, 5, 2)), n_iters=1)
