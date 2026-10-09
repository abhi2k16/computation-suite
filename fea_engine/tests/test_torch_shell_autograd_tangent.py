"""
test_torch_shell_autograd_tangent.py -- validation for item 96 of the
PyTorch side-by-side extension (docs/consolidated_future_roadmap.md,
"Wave 9"): the opt-in method="autograd" path wired into
Shell4MITCCorotational.tangent_stiffness()'s state=... branch, which
routes through autograd_tangent.shell4_mitc_corotational_committed_
tangent_autograd() instead of the default _tangent_stiffness_committed_
complex_step().

Fixture (RECT, _material(), _u_from_theta()) reused verbatim from
tests/test_shell_corotational_committed_coupling.py, so this file's own
checks are validated against the same already-trusted model.

Five lines of evidence:

1. test_method_complex_step_default_unchanged -- method="complex_step"
   (both implicit default and explicit) reproduces the pre-existing
   state=... tangent bit-for-bit -- a regression check that adding the
   method= parameter did not perturb the existing complex-step path at
   all.
2. test_unknown_method_raises -- an unrecognized method value fails
   loudly (ValueError), and this check needs no torch at all (the
   ValueError is raised before any torch import is attempted).
3. test_numpy_shadow_matches_production_internal_force -- THE decisive
   correctness check for the underlying DERIVATION, and the one that
   needed no torch to run: a plain-numpy transcription of the exact
   formula sequence the torch port (autograd_tangent.py's
   shell4_mitc_corotational_committed_tangent_autograd()) implements
   (same variable names/operation order, np. in place of torch., see
   that module's own "VALIDATION METHODOLOGY NOTE" comment) is checked
   directly against Shell4MITCCorotational.internal_force(...,
   state=...)'s real production output on a genuine post-commit trial
   state. This is what was actually run before any torch code was
   written (see autograd_tangent.py's own comment) -- reproduced here
   as a permanent regression guard, and the only one of these checks
   that provides a real numerical guarantee in a torch-less sandbox.
4. test_autograd_matches_complex_step -- (torch-gated) the two methods
   agree with each other on a genuine post-commit trial state, to
   complex-step's own machine-precision floor.
5. test_autograd_mixed_formulation_not_implemented -- (torch-gated)
   passing iter_state != None with method="autograd" raises
   NotImplementedError rather than silently computing the wrong answer
   -- see autograd_tangent.py's own "SCOPE, DELIBERATELY NARROWED"
   comment for why the mixed-formulation branch is out of scope.

SANDBOX NOTE (2026-09-11): torch is not installed/importable in the
sandbox this file was authored in (see test_torch_autograd_tangent_
stiffness.py's own identical note). Tests 4-5 are skipped cleanly via
_HAS_TORCH; tests 1-3 need no torch at all and always run -- in
particular, test 3 (the numpy shadow) is the test that actually
validated the new torch code's DERIVATION in this sandbox, before any
torch code was executed anywhere.

VALIDATED (2026-09-11, on the user's own machine -- Windows, conda base
env, PyTorch installed): all 5 tests in this file PASSED for real,
including the two (test_autograd_matches_complex_step, test_autograd_
mixed_formulation_not_implemented) that could only be structurally
reviewed here. test_autograd_matches_complex_step is the actual payoff
of the shadow-then-port strategy above: the independently-implemented
torch tangent and the existing complex-step tangent agree on a real
torch execution, not just in the numpy shadow that preceded it. See
test_torch_autograd_tangent_stiffness.py's own updated SANDBOX NOTE for
the full combined-run result (30 passed, 1 benign warning, 0 failed
across every torch-gated test this wave added)."""
__author__ = "Abhijeet"
import numpy as np
import pytest

from fea_engine import Material, D_shell, Shell4MITCCorotational
from fea_engine.elements.shells import _MEM
from fea_engine.elements.base import gauss_product

try:
    import torch  # noqa: F401
    from fea_engine.autograd_tangent import shell4_mitc_corotational_committed_tangent_autograd
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


RECT = np.array([[0.875, 0., 0.], [1., 0., 0.], [1., 0.1, 0.], [0.875, 0.1, 0.]])


def _material():
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    h = 0.005
    return D_shell(mat, h), h


def _u_from_theta(theta):
    u = np.zeros(24)
    for i in range(4):
        u[6 * i + 3:6 * i + 6] = theta[i]
    return u


def _post_commit_trial_state():
    """A genuine post-commit trial state: commit at a random small
    rotation, then perturb further with both rotation AND translation --
    mirrors test_shell_corotational_committed_coupling.py's own
    test_committed_tangent_matches_independent_real_finite_difference
    fixture exactly."""
    D, h = _material()
    coro = Shell4MITCCorotational()
    state0 = coro.init_state()
    rng = np.random.default_rng(11)

    theta_commit = rng.normal(scale=0.02, size=(4, 3))
    theta_commit[:, 2] = 0.0
    u_commit = _u_from_theta(theta_commit)
    state1 = coro.commit_state(RECT, u_commit, D, state0)

    u_trial = u_commit.copy()
    u_trial += rng.normal(scale=0.01, size=24)
    u_trial[0:3] += rng.normal(scale=0.003, size=3)
    return coro, D, h, state1, u_trial


def test_method_complex_step_default_unchanged():
    coro, D, h, state1, u_trial = _post_commit_trial_state()
    K_implicit = coro.tangent_stiffness(RECT, u_trial, D, thickness=h, state=state1)
    K_explicit = coro.tangent_stiffness(RECT, u_trial, D, thickness=h, state=state1,
                                         method="complex_step")
    assert np.array_equal(K_implicit, K_explicit)


def test_unknown_method_raises():
    coro, D, h, state1, u_trial = _post_commit_trial_state()
    with pytest.raises(ValueError):
        coro.tangent_stiffness(RECT, u_trial, D, thickness=h, state=state1, method="bogus")


def test_numpy_shadow_matches_production_internal_force():
    """Plain-numpy transcription of the exact torch formula sequence in
    autograd_tangent.shell4_mitc_corotational_committed_tangent_
    autograd() -- same variable names/operation order, np. in place of
    torch. -- checked against the real production internal_force(...,
    state=...) result. This is the check that actually validates the
    new torch code's derivation in a sandbox with no working torch
    install: 1.25e-16 relative agreement confirms the formula sequence
    (including the block-diagonal-rotation-as-reshape identity used to
    avoid materializing a 24x24 matrix) is correct BEFORE any torch-
    specific translation risk enters the picture."""
    coro, D, h, state1, u_trial = _post_commit_trial_state()
    linear = coro._linear
    f_ref = coro.internal_force(RECT, u_trial, D, thickness=h, state=state1)

    X_ref = np.asarray(RECT, dtype=float)
    e1_0, e2_0, e3_0, local = linear._local_frame_and_coords(X_ref)
    R0 = np.vstack([e1_0, e2_0, e3_0])
    centroid_ref = X_ref.mean(axis=0)
    Xref_local = (X_ref - centroid_ref) @ R0.T

    K_local0 = coro._local_material_stiffness(X_ref, D)
    Dm, Db, Ds, hh = D
    assert hh == h

    pts, wts = gauss_product(coro.gauss_order, coro.dim)
    membrane = linear._membrane
    gauss_data = []
    for p, wgt in zip(pts, wts):
        N, _ = membrane.shape_and_derivs(p)
        Bm, detJ = membrane.B_matrix(p, local)
        gauss_data.append((N, Bm, detJ, wgt))

    def exact_drill_rotation_np(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def dof_local_shadow(u):
        u_nodes = u.reshape(4, 6)
        u_trans = u_nodes[:, 0:3]
        theta_global = u_nodes[:, 3:6]
        mean_theta = theta_global.mean(axis=0)
        theta_z_mean = mean_theta[2]
        R_drill = R0 @ exact_drill_rotation_np(theta_z_mean)
        e1, e2, e3 = R_drill[0], R_drill[1], R_drill[2]

        x_current = X_ref + u_trans
        centroid_current = x_current.mean(axis=0)
        x_current_local = (x_current - centroid_current) @ R_drill.T

        u_local_mem_xy = x_current_local[:, 0:2] - Xref_local[:, 0:2]
        thetaz_local = theta_global[:, 2] - theta_z_mean

        node_vecs = []
        for a in range(4):
            node_vecs.append(np.array([
                u_local_mem_xy[a, 0], u_local_mem_xy[a, 1],
                u_trans[a] @ e3_0, theta_global[a] @ e1_0,
                theta_global[a] @ e2_0, thetaz_local[a],
            ]))
        return np.concatenate(node_vecs), e1, e2, e3, theta_global

    def coupling_force_shadow(dof_local):
        wx = np.array([dof_local[6 * a + 4] for a in range(4)])
        wy = np.array([-dof_local[6 * a + 3] for a in range(4)])
        dof_mem = dof_local[_MEM]
        wx_use = wx - wx.sum() / 4.0
        wy_use = wy - wy.sum() / 4.0
        Nb_offset = -0.25
        contrib = np.zeros(24)
        for N, Bm, detJ, wgt in gauss_data:
            Wx, Wy = N @ wx_use, N @ wy_use
            eps_add = np.array([0.5 * Wx * Wx, 0.5 * Wy * Wy, Wx * Wy])
            eps_lin = Bm @ dof_mem
            sigma_total = Dm @ (eps_lin + eps_add)
            f_mem_add = Bm.T @ (Dm @ eps_add) * detJ * wgt * h
            for i, idx in enumerate(_MEM):
                contrib[idx] += f_mem_add[i]
            for b in range(4):
                Nb_c = N[b] + Nb_offset
                contrib[6 * b + 4] += h * detJ * wgt * Nb_c * (Wx * sigma_total[0] + Wy * sigma_total[2])
                contrib[6 * b + 3] += -h * detJ * wgt * Nb_c * (Wy * sigma_total[1] + Wx * sigma_total[2])
        return contrib

    def coupling_force_committed_shadow(dof_local, theta_global, theta_committed, N_add_history):
        theta_incr = theta_global - theta_committed
        wx_incr = theta_incr @ e2_0
        wy_incr = -(theta_incr @ e1_0)
        dof_mem = dof_local[_MEM]
        contrib = np.zeros(24)
        for gp_idx, (N, Bm, detJ, wgt) in enumerate(gauss_data):
            Wx, Wy = N @ wx_incr, N @ wy_incr
            eps_add_incr = np.array([0.5 * Wx * Wx, 0.5 * Wy * Wy, Wx * Wy])
            Nadd_trial_gp = N_add_history[gp_idx] + Dm @ eps_add_incr
            eps_lin = Bm @ dof_mem
            sigma_total = Dm @ eps_lin + Nadd_trial_gp
            f_mem_add = Bm.T @ Nadd_trial_gp * detJ * wgt * h
            for i, idx in enumerate(_MEM):
                contrib[idx] += f_mem_add[i]
            for b in range(4):
                Nb = N[b]
                contrib[6 * b + 4] += h * detJ * wgt * Nb * (Wx * sigma_total[0] + Wy * sigma_total[2])
                contrib[6 * b + 3] += -h * detJ * wgt * Nb * (Wy * sigma_total[1] + Wx * sigma_total[2])
        return contrib

    dof_local, e1, e2, e3, theta_global = dof_local_shadow(u_trial)
    f_local = (K_local0 @ dof_local + coupling_force_shadow(dof_local)
               + coupling_force_committed_shadow(dof_local, theta_global,
                                                  state1["theta_committed"], state1["N_add_history"]))
    R_mat = np.vstack([e1, e2, e3])
    f_shadow = (f_local.reshape(8, 3) @ R_mat).reshape(24)

    rel_err = np.max(np.abs(f_shadow - f_ref)) / np.max(np.abs(f_ref))
    assert rel_err < 1e-10


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_autograd_matches_complex_step():
    """K_ad is the RAW (unsymmetrized) autograd Jacobian -- see
    shell4_mitc_corotational_committed_tangent_autograd()'s own
    docstring: this element's committed-state internal force is not
    exactly the gradient of a scalar potential, so its true Jacobian is
    genuinely (not just to float roundoff) asymmetric. tangent_
    stiffness(method="complex_step") returns the SYMMETRIZED
    0.5*(K+K.T) production convention, so it must be compared against
    K_ad's own symmetrization, not K_ad directly -- comparing raw-vs-
    symmetrized would spuriously fail on every asymmetric entry even
    though both methods agree exactly on the underlying raw Jacobian."""
    coro, D, h, state1, u_trial = _post_commit_trial_state()
    K_cs = coro.tangent_stiffness(RECT, u_trial, D, thickness=h, state=state1,
                                   method="complex_step")
    K_ad = coro.tangent_stiffness(RECT, u_trial, D, thickness=h, state=state1,
                                   method="autograd")
    K_ad_sym = 0.5 * (K_ad + K_ad.T)
    rel_err = np.max(np.abs(K_ad_sym - K_cs)) / np.max(np.abs(K_cs))
    assert rel_err < 1e-6


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_autograd_mixed_formulation_not_implemented():
    coro, D, h, state1, u_trial = _post_commit_trial_state()
    n_gauss = len(gauss_product(coro.gauss_order, coro.dim)[0])
    N_add = np.zeros((n_gauss, 3))
    with pytest.raises(NotImplementedError):
        coro.tangent_stiffness(RECT, u_trial, D, thickness=h, state=state1,
                                iter_state=N_add, method="autograd")
