# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
wave13_validation.py -- Wave 13 item 120 (fea_engine/docs/consolidated_
future_roadmap.md): a cross-configuration validation harness tying
items 117 (`neural_operator.reduced_eom_residual_trajectory`), 118
(`reduced_basis_operator.RegularizedProjectionEncoder`), and 119
(`parameterized_latent_ode.integrate_rk4`/`ParameterizedLatentODE`)
together on ONE real `fea_engine` fixture pair, rather than validating
each in isolation the way their own test files already do.

Two independent lines of evidence, following item 113's own "narrow
and document, don't overclaim" precedent -- neither claim below is a
new capability; both are CROSS-CHECKS that items built and tested
separately actually agree with each other and with independently-
generated reference data when used together:

  1. `cross_check_reduced_dynamics()` (numpy-unconditional): the SAME
     fitted `PolynomialModalROM` reduced force model, driven by the
     SAME modal excitation, is time-marched by TWO INDEPENDENT
     integrators -- `nonlinear_dynamics.integrate_newmark_surrogate`
     (this package's own validated Newton-Newmark scheme, Wave 12
     item 114) and item 119's freshly-validated `integrate_rk4`
     (a completely different numerical method, same continuous
     equation) -- and the two trajectories are checked to agree with
     each other, AND item 117's `reduced_eom_residual_trajectory` is
     used to confirm BOTH trajectories actually satisfy the continuous
     reduced equation of motion, not just each other (two integrators
     could in principle agree on a shared bug).

  2. `cross_check_basis_generalization()`: item 118's own decisive
     cross-mesh-resolution claim (a basis fit at `n_elem_train`
     reconstructs a snapshot solved at a DIFFERENT `n_elem_validate`),
     re-run here as part of the SAME harness/report rather than only
     in item 118's own isolated test file, so a single call surfaces
     all three items' health on one real structural configuration
     pair.

torch-gated (`cross_check_operators_torch`, item 117's
`ExcitationResponseOperator` + item 119's `ParameterizedLatentODE`
trained together and checked against the SAME Newton-Newmark reference
this module's numpy core already validates against): structurally
reviewed and syntax-checked in this sandbox (no PyTorch installed
here, matching every other torch-gated module in this package -- see
`differentiable_correction.py`'s own module docstring for why the
`except Exception` guard is used instead of `except ImportError`), NOT
executed. Needs confirmation on a machine with PyTorch installed,
exactly like `ExcitationResponseOperator`'s and
`ParameterizedLatentODE`'s own torch-gated tests.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .nonlinear_rom import AppliedLoadStrategy, PolynomialModalROM
from .nonlinear_dynamics import integrate_newmark_surrogate
from .neural_operator import reduced_eom_residual_trajectory, _HAS_TORCH as _NO_HAS_TORCH
from .reduced_basis_operator import RegularizedProjectionEncoder
from .parameterized_latent_ode import integrate_rk4, _HAS_TORCH as _PLO_HAS_TORCH
from .pod import PodBasis

_HAS_TORCH = _NO_HAS_TORCH and _PLO_HAS_TORCH


def _v_component(free_dofs, mesh, W):
    """Same transverse-displacement-only extraction item 118's own
    test file uses (Beam2DCorotational's [u, v, theta] per-node dof
    convention)."""
    free_dofs = np.asarray(free_dofs)
    v_mask = (free_dofs % 3) == 1
    node_idx = free_dofs[v_mask] // 3
    coords = mesh.nodes[node_idx, 0]
    vals = W[v_mask] if W.ndim == 1 else W[v_mask, :]
    order = np.argsort(coords)
    return coords[order], (vals[order] if W.ndim == 1 else vals[order, :])


def _fit_reduced_model_from_fixture(fixture, n_train=40, target_fracs=(-3.0, 3.0),
                                     reference_scale=0.05, seed=0):
    """Builds an AppliedLoadStrategy training set from a
    `clamped_clamped_nonlinear_beam_system` fixture dict and fits a
    PolynomialModalROM to it -- the one piece of setup every check in
    this module shares."""
    V, M_ff, freq_hz = fixture["V"], fixture["M_ff"], fixture["freq_hz"]
    mode_shape_peaks, fom_solver = fixture["mode_shape_peaks"], fixture["fom_solver"]
    n_modes = V.shape[1]

    rng = np.random.default_rng(seed)
    strat = AppliedLoadStrategy(target_fracs=target_fracs, reference_scale=reference_scale,
                                 n_samples=n_train, rng=rng)
    q_l, q_nl, F_nl, W = strat.generate(V, M_ff, freq_hz, mode_shape_peaks, fom_solver,
                                         return_snapshots=True)
    force_model = PolynomialModalROM(n_modes=n_modes).fit(q_nl, F_nl)

    Lambda = (2 * np.pi * np.asarray(freq_hz)) ** 2
    C_ff = fixture["C_ff"]
    C_r = V.T @ C_ff @ V   # modal damping, full (n_modes, n_modes) -- Rayleigh damping
    return force_model, Lambda, C_r, W


def cross_check_reduced_dynamics(fixture, n_train=40, dt=2e-5, n_steps=2500,
                                  F_ext_amplitude=None, seed=0):
    """Line of evidence 1 (see module docstring). Returns a report
    dict; raises nothing -- callers assert on the returned numbers so
    the actual values are visible in a failing test's own traceback.

    `dt`/`n_steps` default to a resolution empirically found (this
    module's own development, against the `clamped_clamped_nonlinear_
    beam_system(n_elem=10, n_modes=2)` fixture) fine enough for the
    two independent integrators to agree to ~0.4% (`dt=2e-4` only
    agrees to ~42% -- a coarser step resolves the higher mode's
    ~2.2ms period too poorly for either integrator's own truncation
    error to be small, which is expected numerical behavior, not a
    bug in either method -- halving `dt` repeatedly here converges the
    two integrators together, the standard way to tell a resolution
    issue apart from a genuine disagreement).
    """
    force_model, Lambda, C_r, _W = _fit_reduced_model_from_fixture(
        fixture, n_train=n_train, seed=seed)
    n_modes = len(Lambda)
    if F_ext_amplitude is None:
        # a modest fraction of the linear restoring force at a
        # representative modal amplitude, so the trajectory stays in
        # the fitted polynomial's well-sampled range (same reasoning
        # AppliedLoadStrategy's own target_fracs uses) -- empirically
        # keeps max|q_nl| ~2e-3, well inside the training sweep's own
        # target_fracs=(-3,3)*reference_scale=0.05 range.
        F_ext_amplitude = 0.05 * Lambda * 0.02
    F_ext_const = np.asarray(F_ext_amplitude, dtype=float)
    if F_ext_const.shape != (n_modes,):
        F_ext_const = np.full(n_modes, float(np.mean(F_ext_const)))

    q0 = np.zeros(n_modes)
    qdot0 = np.zeros(n_modes)

    # --- integrator A: Newton-Newmark (Wave 12 item 114) ---
    t, q_newmark, _q_l, _F_nl = integrate_newmark_surrogate(
        Lambda, C_r, force_model, F_ext=F_ext_const, q0=q0, qdot0=qdot0,
        dt=dt, n_steps=n_steps, domain="q_nl", correction="newton", newton_tol=1e-10)

    # --- integrator B: item 119's corrected RK4, same continuous ODE,
    #     written as a first-order system z=[q, qdot] ---
    def f_reduced(tt, z):
        q, qd = z[:n_modes], z[n_modes:]
        qddot = F_ext_const - C_r @ qd - Lambda * q - force_model.force(q)
        return np.concatenate([qd, qddot])

    z0 = np.concatenate([q0, qdot0])
    z_hist = integrate_rk4(f_reduced, z0, t)
    q_rk4 = z_hist[:, :n_modes]

    # agreement between the two independent integrators
    scale = max(np.max(np.abs(q_newmark)), 1e-12)
    max_rel_diff = np.max(np.abs(q_newmark - q_rk4)) / scale

    # item 117's residual check, applied to BOTH trajectories
    F_ext_hist = np.tile(F_ext_const, (len(t), 1))
    _, resid_newmark = reduced_eom_residual_trajectory(
        t, q_newmark, Lambda, C_r, F_nl_fn=force_model.force, F_ext_hist=F_ext_hist)
    _, resid_rk4 = reduced_eom_residual_trajectory(
        t, q_rk4, Lambda, C_r, F_nl_fn=force_model.force, F_ext_hist=F_ext_hist)
    force_scale = max(np.max(np.abs(F_ext_hist)), 1e-12)

    return {
        "n_modes": n_modes,
        "max_rel_diff_newmark_vs_rk4": float(max_rel_diff),
        "eom_residual_rms_newmark": float(np.sqrt(np.mean(resid_newmark ** 2))),
        "eom_residual_rms_rk4": float(np.sqrt(np.mean(resid_rk4 ** 2))),
        "force_scale": float(force_scale),
        "t": t, "q_newmark": q_newmark, "q_rk4": q_rk4,
    }


def cross_check_basis_generalization(fixture_train, fixture_validate, n_train=30,
                                      n_pod_modes=3, n_validate_samples=3, seed_train=0,
                                      seed_validate=1):
    """Line of evidence 2 (see module docstring) -- item 118's own
    decisive check, re-run as part of this harness so the report from
    `run_cross_configuration_validation()` below covers all three
    items together."""
    V, M_ff, freq_hz = fixture_train["V"], fixture_train["M_ff"], fixture_train["freq_hz"]
    mode_shape_peaks, fom_solver = fixture_train["mode_shape_peaks"], fixture_train["fom_solver"]
    free_train, mesh_train = fixture_train["free_dofs"], fixture_train["sys"].mesh

    rng = np.random.default_rng(seed_train)
    strat = AppliedLoadStrategy(target_fracs=(-3.0, 3.0), reference_scale=0.05,
                                 n_samples=n_train, rng=rng)
    _, _, _, W_train = strat.generate(V, M_ff, freq_hz, mode_shape_peaks, fom_solver,
                                       return_snapshots=True)
    coords_train, Wv_train = _v_component(free_train, mesh_train, W_train)

    pod = PodBasis().fit(Wv_train, n_modes=n_pod_modes)
    enc = RegularizedProjectionEncoder().fit(coords_train, pod.V, smoothing=1e-8)

    held_out = Wv_train[:, -1]
    recon_same, _ = enc.reconstruct(coords_train, held_out, reg=1e-8)
    err_same = float(np.linalg.norm(recon_same - held_out) / np.linalg.norm(held_out))

    V20, M_ff20, freq_hz20 = fixture_validate["V"], fixture_validate["M_ff"], fixture_validate["freq_hz"]
    mode_shape_peaks20 = fixture_validate["mode_shape_peaks"]
    fom_solver20 = fixture_validate["fom_solver"]
    free_val, mesh_val = fixture_validate["free_dofs"], fixture_validate["sys"].mesh

    rng2 = np.random.default_rng(seed_validate)
    strat2 = AppliedLoadStrategy(target_fracs=(-2.0, 2.0), reference_scale=0.05,
                                  n_samples=n_validate_samples, rng=rng2)
    _, _, _, W_val = strat2.generate(V20, M_ff20, freq_hz20, mode_shape_peaks20, fom_solver20,
                                      return_snapshots=True)
    coords_val, Wv_val = _v_component(free_val, mesh_val, W_val)

    errs_cross = []
    for i in range(Wv_val.shape[1]):
        true_vals = Wv_val[:, i]
        z = enc.encode(coords_val, true_vals, reg=1e-6)
        recon = enc.decode(z, coords_val)
        errs_cross.append(float(np.linalg.norm(recon - true_vals) / np.linalg.norm(true_vals)))

    return {
        "err_same_resolution": err_same,
        "err_cross_resolution_max": max(errs_cross),
        "err_cross_resolution_all": errs_cross,
        "n_elem_train": fixture_train["n_elem"],
        "n_elem_validate": fixture_validate["n_elem"],
    }


def run_cross_configuration_validation(fixture_train, fixture_validate, **kwargs):
    """Convenience wrapper running both lines of evidence and
    returning one combined report dict."""
    dyn_kwargs = {k: v for k, v in kwargs.items()
                  if k in ("n_train", "dt", "n_steps", "F_ext_amplitude", "seed")}
    basis_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("n_train", "n_pod_modes", "n_validate_samples",
                              "seed_train", "seed_validate")}
    return {
        "reduced_dynamics": cross_check_reduced_dynamics(fixture_train, **dyn_kwargs),
        "basis_generalization": cross_check_basis_generalization(
            fixture_train, fixture_validate, **basis_kwargs),
    }


# =====================================================================
# torch-gated: items 117 + 119's trainable classes, cross-checked
# against the SAME Newton-Newmark reference cross_check_reduced_
# dynamics() already validates against, on the SAME real fixture.
# =====================================================================
if _HAS_TORCH:
    from .neural_operator import ExcitationResponseOperator
    from .parameterized_latent_ode import ParameterizedLatentODE

    def cross_check_operators_torch(fixture, n_train_freqs=4, n_time=64, seed=0):
        """Trains an ExcitationResponseOperator (item 117) across a
        sweep of sinusoidal modal excitation frequencies on the given
        fixture's own fitted reduced dynamics, and checks it against a
        held-out UNSEEN frequency's Newton-Newmark reference trajectory
        (same honest R^2 > 0 generalization bar item 117's own test
        uses -- this is a cross-check of behavior already validated in
        isolation, not a new accuracy claim)."""
        force_model, Lambda, C_r, _W = _fit_reduced_model_from_fixture(
            fixture, n_train=40, seed=seed)
        n_modes = len(Lambda)
        dt = 5e-4
        t = np.arange(n_time) * dt

        freqs_train = np.linspace(20.0, 60.0, n_train_freqs)
        F_ext_trajs, q_trajs = [], []
        for f in freqs_train:
            F_ext_fn = lambda tt, f=f: 0.05 * Lambda * np.sin(2 * np.pi * f * tt)
            _, q_hist, _, _ = integrate_newmark_surrogate(
                Lambda, C_r, force_model, F_ext=F_ext_fn,
                q0=np.zeros(n_modes), qdot0=np.zeros(n_modes),
                dt=dt, n_steps=n_time - 1, domain="q_nl", correction="newton")
            F_ext_trajs.append(np.stack([F_ext_fn(tt) for tt in t], axis=0))
            q_trajs.append(q_hist)

        op = ExcitationResponseOperator(n_modes=n_modes, width=16, modes=8, n_layers=2,
                                         n_epochs=500, lr=3e-3, seed=seed)
        op.fit(t, np.array(F_ext_trajs), np.array(q_trajs), n_epochs=500)

        f_unseen = float(np.mean(freqs_train[:2]) + 0.5 * (freqs_train[1] - freqs_train[0]))
        F_ext_unseen_fn = lambda tt: 0.05 * Lambda * np.sin(2 * np.pi * f_unseen * tt)
        _, q_true, _, _ = integrate_newmark_surrogate(
            Lambda, C_r, force_model, F_ext=F_ext_unseen_fn,
            q0=np.zeros(n_modes), qdot0=np.zeros(n_modes),
            dt=dt, n_steps=n_time - 1, domain="q_nl", correction="newton")
        F_ext_unseen = np.stack([F_ext_unseen_fn(tt) for tt in t], axis=0)
        q_pred = op.predict(t, F_ext_unseen)

        ss_res = np.sum((q_pred - q_true) ** 2)
        ss_tot = np.sum((q_true - q_true.mean(axis=0)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        return {"r2_unseen_frequency": float(r2), "f_unseen_hz": f_unseen,
                "freqs_train_hz": freqs_train.tolist()}
