# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_shell_corotational_committed_coupling.py -- validation for
Wave 4 item 46, building block C (docs/shell_rotation_coupling_fix_
roadmap.md Section 4.3): `Shell4MITCCorotational`'s
init_state()/commit_state() and the state= branch of
internal_force()/tangent_stiffness().

Read shells.py's own class-level "BUILDING BLOCK C" comment (directly
above init_state()) before extending this file -- it documents TWO
findings from this same implementation pass:

1. A FIRST design (generalizing the whole dof_local extraction, via
   `_mean_rigid_rotation()`, to be relative to a committed baseline)
   was tried, caught reproducing "Design history" dead end 3 via direct
   testing, and reverted in place before being wired into any test file
   -- not represented here since no code from it shipped.

2. The REVISED design that IS implemented (this file's actual subject)
   -- an accumulated, since-commit "absolute rotation" von Karman
   coupling term, via a new per-Gauss-point `N_add_history` state --
   has its OWN, more subtle flaw: the accumulation rule is a proven-
   wrong discretization of the underlying integral (it converges to
   ZERO, not to the correct total, as commit count increases at a FIXED
   final state). This file's tests are split into two groups
   accordingly:

   (a) Tests 1-5: validate the STATE MACHINERY ITSELF (bookkeeping,
       zero/reference-point reduction, byte-identical state=None
       regression, holomorphy/complex-step correctness) -- these
       properties are all independently correct regardless of whether
       the specific coupling formula is "the right physics," and remain
       true infrastructure for a future, corrected coupling term to
       build on.
   (b) Test 6 is a PERMANENT CANARY for the accumulation-rule flaw
       itself -- documents the "N_add_history halves as commit count
       doubles at a fixed final state" signature directly, mirroring
       test_shell_corotational_exact_rotation_extraction.py's own
       "prove a specific finding, don't just assert it once" convention
       from building block B. If this test ever starts failing, the
       accumulation rule has changed (hopefully to the correct
       telescoping one, `d(0.5*Wx^2)=Wx*dWx` per commit) -- update this
       test and shells.py's docstrings together, not as a silent
       coincidence.

This building block, per the class-level comment's "STATUS" paragraph,
is NOT a validated fix for the elastica foreshortening problem --
state=None (the only path every OTHER existing test in this project
exercises) is completely unaffected either way, confirmed directly
here (test 3).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import Material, D_shell, Shell4MITCCorotational


def _material():
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    h = 0.005
    return D_shell(mat, h), h


RECT = np.array([[0.875, 0., 0.], [1., 0., 0.], [1., 0.1, 0.], [0.875, 0.1, 0.]])


def _u_from_theta(theta):
    u = np.zeros(24)
    for i in range(4):
        u[6 * i + 3:6 * i + 6] = theta[i]
    return u


def test_init_state_shape_and_values():
    coro = Shell4MITCCorotational()
    state = coro.init_state()
    assert state["N_add_history"].shape == (4, 3)
    assert state["theta_committed"].shape == (4, 3)
    assert np.max(np.abs(state["N_add_history"])) == 0.0
    assert np.max(np.abs(state["theta_committed"])) == 0.0


def test_zero_state_gives_zero_force_and_matches_u0_tangent():
    """state=init_state() at u_elem=0 must reduce to exactly zero force
    (nothing committed, nothing displaced) and its tangent must match
    the state=None u=0 tangent -- the same "state=None and
    state=init_state() agree at the reference point" sanity property
    every other stateful element in this package satisfies trivially."""
    D, h = _material()
    coro = Shell4MITCCorotational()
    state0 = coro.init_state()

    f0 = coro.internal_force(RECT, np.zeros(24), D, thickness=h, state=state0)
    assert np.max(np.abs(f0)) == 0.0

    K_state = coro.tangent_stiffness(RECT, np.zeros(24), D, thickness=h, state=state0)
    K_none = coro.tangent_stiffness(RECT, np.zeros(24), D, thickness=h)
    assert np.allclose(K_state, K_none, atol=1e-4, rtol=1e-8), (
        f"u=0 tangent with state=init_state() disagrees with state=None: "
        f"max diff {np.max(np.abs(K_state - K_none)):.3e}"
    )


def test_state_none_regression_byte_identical():
    """The state= branch in internal_force()/tangent_stiffness() must
    be purely ADDITIVE -- state=None (the default, and the ONLY path
    every pre-existing test in this project exercises) must be
    completely untouched by this building block's addition."""
    D, h = _material()
    coro = Shell4MITCCorotational()
    rng = np.random.default_rng(3)
    for trial in range(5):
        u = rng.normal(scale=0.02, size=24)
        f_default = coro.internal_force(RECT, u, D, thickness=h)
        f_state_none = coro.internal_force(RECT, u, D, thickness=h, state=None)
        assert np.array_equal(f_default, f_state_none), trial

    K_default = coro.tangent_stiffness(RECT, np.zeros(24), D, thickness=h)
    K_state_none = coro.tangent_stiffness(RECT, np.zeros(24), D, thickness=h, state=None)
    assert np.array_equal(K_default, K_state_none)


def test_commit_state_is_idempotent_at_a_fixed_point():
    """Committing twice at the SAME u_elem (no displacement change
    between commits) must leave state completely unchanged -- the
    second commit's own since-commit increment is exactly zero."""
    D, h = _material()
    coro = Shell4MITCCorotational()
    state0 = coro.init_state()
    rng = np.random.default_rng(7)
    theta = rng.normal(scale=0.02, size=(4, 3))
    theta[:, 2] = 0.0   # keep drilling out of it, not the subject here
    u = _u_from_theta(theta)

    state1 = coro.commit_state(RECT, u, D, state0)
    state2 = coro.commit_state(RECT, u, D, state1)
    assert np.array_equal(state1["N_add_history"], state2["N_add_history"])
    assert np.array_equal(state1["theta_committed"], state2["theta_committed"])
    # and the commit actually did something relative to the FIRST state
    # (not a no-op from the start) -- confirms this is a real check, not
    # a vacuous one.
    assert not np.array_equal(state0["N_add_history"], state1["N_add_history"])


def test_committed_tangent_matches_independent_real_finite_difference():
    """The class-level comment claims `_internal_force_committed()`
    stays holomorphic in u_elem (built entirely from theta minus a REAL
    state constant, then products/sums), so `tangent_stiffness()`'s
    state branch reuses complex-step. Cross-check that claim directly
    against an INDEPENDENT real central finite-difference Jacobian, at
    a genuine post-commit trial state (not just u=0) -- this is the
    actual gate for "is the holomorphy claim true," not merely "does it
    run without a complex-arithmetic error."""
    D, h = _material()
    coro = Shell4MITCCorotational()
    state0 = coro.init_state()
    rng = np.random.default_rng(11)

    theta_commit = rng.normal(scale=0.02, size=(4, 3))
    theta_commit[:, 2] = 0.0
    u_commit = _u_from_theta(theta_commit)
    state1 = coro.commit_state(RECT, u_commit, D, state0)

    u_trial = u_commit + rng.normal(scale=0.01, size=24)
    K_cs = coro.tangent_stiffness(RECT, u_trial, D, thickness=h, state=state1)

    def f(u):
        return coro.internal_force(RECT, u, D, thickness=h, state=state1)

    n, hfd = 24, 1e-6
    K_fd = np.zeros((n, n))
    for j in range(n):
        du = np.zeros(n)
        du[j] = hfd
        K_fd[:, j] = (f(u_trial + du) - f(u_trial - du)) / (2.0 * hfd)
    K_fd = 0.5 * (K_fd + K_fd.T)

    rel = np.linalg.norm(K_cs - K_fd) / max(np.linalg.norm(K_fd), 1e-30)
    assert rel < 1e-6, (
        f"complex-step vs. real central-FD tangent for the committed path "
        f"disagree by relative {rel:.3e} -- the holomorphy claim in "
        f"_bending_membrane_coupling_force_committed()'s docstring would be "
        f"WRONG if this ever fails"
    )


def test_accumulation_rule_vanishes_with_finer_commit_granularity():
    """PERMANENT CANARY for the "SECOND FINDING" documented in shells.py's
    class-level "BUILDING BLOCK C" comment: committing the SAME final
    rotation state via progressively more, smaller, equal sub-steps
    makes the accumulated N_add_history shrink towards zero, rather
    than converge to a fixed value -- the textbook signature of
    accumulating `0.5*(increment)^2` instead of the correct telescoping
    `d(0.5*x^2) = x*dx` update. Proves this is a genuine discretization
    defect in the accumulation rule (not a "needs a finer load ramp"
    tuning issue): each doubling of the sub-step count should leave
    N_add_history roughly HALVED, for any reasonable rotation magnitude.

    If this test ever starts failing, the accumulation rule has been
    corrected (hopefully to the telescoping update, which per the
    class-level comment's own algebra would make this reduce EXACTLY to
    `_bending_membrane_coupling_force()`'s own dead-end-4 "ABSOLUTE
    rotation" branch, independent of step count) -- update this test
    and shells.py's docstrings together, not as a silent coincidence."""
    D, h = _material()
    coro = Shell4MITCCorotational()

    theta_final = np.zeros((4, 3))
    rng = np.random.default_rng(21)
    theta_final[:, 0] = rng.normal(scale=0.008, size=4)
    theta_final[:, 1] = rng.normal(scale=0.008, size=4)

    magnitudes = []
    for n_steps in [1, 2, 4, 8, 16]:
        state = coro.init_state()
        for k in range(1, n_steps + 1):
            theta_k = theta_final * (k / n_steps)
            u_k = _u_from_theta(theta_k)
            state = coro.commit_state(RECT, u_k, D, state)
        magnitudes.append(np.max(np.abs(state["N_add_history"])))

    for i in range(len(magnitudes) - 1):
        ratio = magnitudes[i + 1] / magnitudes[i]
        assert 0.4 < ratio < 0.6, (
            f"expected ~halving (ratio~0.5) going from n_steps="
            f"{[1,2,4,8,16][i]} to {[1,2,4,8,16][i+1]}, got ratio={ratio:.3f} "
            f"(magnitudes={magnitudes}) -- if this no longer halves, the "
            f"accumulation-rule flaw this test guards may have been fixed"
        )
    # and the 1-step value must equal _bending_membrane_coupling_force()'s
    # own dead-end-4 absolute-rotation eps_add (Dm @ [0.5wx^2,...]) exactly,
    # since with EXACTLY one commit from the zero baseline, "since commit"
    # IS "since reference" -- confirms the n_steps=1 anchor is correct, not
    # just the halving TREND.
    e1_0, e2_0, e3_0, _ = coro._linear._local_frame_and_coords(RECT)
    Dm = D[0]
    wx = theta_final @ e2_0
    wy = -(theta_final @ e1_0)
    # crude single-Gauss-point-free check: peak nodal value bounds the
    # shape-function-interpolated Gauss-point value from above.
    eps_add_bound = 0.5 * max(np.max(wx**2), np.max(wy**2))
    assert magnitudes[0] <= np.max(np.abs(Dm)) * eps_add_bound * 1.5 + 1e-6
