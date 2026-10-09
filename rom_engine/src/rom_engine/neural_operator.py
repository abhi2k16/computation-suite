# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
neural_operator.py -- Wave 13 item 117 (fea_engine/docs/consolidated_
future_roadmap.md): an excitation-to-response operator for the
NONLINEAR reduced dynamics, adapting FRINO's (Yang et al. 2025,
"FE reduced-order model-informed neural operator for structural
dynamic response prediction") architectural pattern -- train ONE
operator mapping a reduced excitation time series directly to a
predicted response trajectory, so a single trained model generalizes
to excitation WAVEFORMS it never saw in training, unlike every
existing `ReducedForceModel` in this package (each of which is an
instantaneous state->force map, re-time-marched by
`nonlinear_dynamics.integrate_newmark_surrogate` for every new load
case).

WHY THIS IS NOT A DIRECT PORT OF FRINO (see the roadmap's own "Key
mismatches" note for Wave 13): FRINO's own physics loss (its Eqs.
15-19) is a closed-form FREQUENCY-DOMAIN relation between excitation
and modal response that only holds for a LINEAR, non-state-dependent
M/C/K -- it has no nonlinear force term at all. This package's whole
reason for existing is nonlinear structural response, so the physics
loss here is instead a direct finite-difference residual of the real
reduced equation of motion already used throughout this package
(`nonlinear_dynamics.py`'s own mass-normalized convention,
`qddot + C_r@qdot + Lambda*q + F_nl(q) = F_ext(t)`), evaluated on the
predicted trajectory itself -- not a linear frequency-response
formula. What IS kept from FRINO: (a) the excitation-in/response-out
operator framing (as opposed to a state->force map needing external
time-marching), and (b) the "Fourier layer" architecture itself
(spectral convolution truncated to a fixed number of Fourier modes,
FRINO's Eq. 12-15) as the operator's core building block -- a genuine,
if compact, from-scratch FNO1d, not a stand-in dense network, since
FRINO's own name and contribution centers specifically on that layer.

Two independently useful pieces, split the same numpy-core/torch-
optional way every other trainable component in this package is
(`nonlinear_rom.NeuralSurrogate`, `differentiable_correction.py`):

  - `reduced_eom_residual_trajectory()`: PURE NUMPY. Central-difference
    residual of the real reduced EOM along a whole (n_steps, n_modes)
    trajectory. Needs no torch at all, and is the actual physics-
    fidelity claim this item makes -- tested directly against known
    closed-form and `nonlinear_dynamics`-integrated references.
  - `SpectralConv1d` / `FNO1dBlock` / `ExcitationResponseOperator`:
    torch-optional (only instantiating `ExcitationResponseOperator`
    needs torch; importing this module never does -- the same
    `_HAS_TORCH`/`_require_torch()` convention `nonlinear_rom.py`
    already establishes).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from itertools import combinations_with_replacement
from collections import Counter

_HAS_TORCH = False
try:
    import torch
    import torch.nn as _nn
    _HAS_TORCH = True
except Exception:
    # See nonlinear_rom.NeuralSurrogate's own comment on this exact
    # except clause -- a CUDA-linked wheel can install but fail to
    # import with OSError/ValueError on a machine with no matching
    # CUDA runtime, not just ImportError.
    pass


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "ExcitationResponseOperator requires PyTorch, which is not "
            "importable in this environment. Install it (`pip install "
            "torch`) -- reduced_eom_residual_trajectory() itself does NOT "
            "need torch (pure NumPy) and works unconditionally.")


# =====================================================================
# 1. Physics residual along a trajectory -- pure NumPy, unconditional.
# =====================================================================
def reduced_eom_residual_trajectory(t, q_hist, Lambda, C_r, F_nl_fn, F_ext_hist):
    """Central-difference residual of the mass-normalized reduced EOM

        qddot(t) + C_r @ qdot(t) + Lambda*q(t) + F_nl(q(t)) - F_ext(t) = 0

    evaluated at every INTERIOR time point of a given trajectory (the
    first and last samples have no centered derivative and are
    excluded, matching the standard central-difference convention).

    Parameters
    ----------
    t : (n_steps,) array_like
        Time stamps. Must be uniformly spaced (checked directly,
        `np.diff(t)` constant to within `rtol=1e-8`) -- central
        differences below assume a single scalar `dt`.
    q_hist : (n_steps, n_modes) array_like
        The reduced-coordinate trajectory to residualize (a training
        target, a model's own prediction, or anything else in this
        shape -- this function has no opinion on where it came from).
    Lambda : (n_modes,) array_like
        Linear modal stiffness, same convention as `nonlinear_dynamics.
        integrate_newmark_surrogate`.
    C_r : (n_modes,) or (n_modes, n_modes) array_like
        Modal damping -- diagonal vector or full matrix, same
        convention as `integrate_newmark_surrogate`.
    F_nl_fn : callable, (n, n_modes) -> (n, n_modes), or None
        Nonlinear reduced force, e.g. a fitted
        `nonlinear_rom.PolynomialModalROM.force` (batched). `None`
        (default via omission at the call site) treats the system as
        purely linear (F_nl == 0) -- useful for the decisive linear
        closed-form cross-check this function's own tests use.
    F_ext_hist : (n_steps, n_modes) array_like
        External modal force history, sampled at the SAME `t`.

    Returns
    -------
    t_interior : (n_steps-2,) ndarray
    residual : (n_steps-2, n_modes) ndarray
    """
    t = np.asarray(t, dtype=float)
    q_hist = np.asarray(q_hist, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    F_ext_hist = np.asarray(F_ext_hist, dtype=float)
    n_steps, n_modes = q_hist.shape
    if n_steps < 3:
        raise ValueError("reduced_eom_residual_trajectory needs at least 3 time samples")
    if t.shape != (n_steps,):
        raise ValueError(f"t must have shape ({n_steps},), got {t.shape}")
    if F_ext_hist.shape != (n_steps, n_modes):
        raise ValueError(f"F_ext_hist must have shape {(n_steps, n_modes)}, got {F_ext_hist.shape}")
    dts = np.diff(t)
    dt = float(dts[0])
    if dt <= 0 or not np.allclose(dts, dt, rtol=1e-8, atol=1e-12):
        raise ValueError("t must be uniformly spaced with a positive step")

    C_r = np.asarray(C_r, dtype=float)
    if C_r.ndim == 1:
        C_mat = np.diag(C_r)
    elif C_r.ndim == 2:
        C_mat = C_r
    else:
        raise ValueError(f"C_r must be 1-D (diagonal) or 2-D, got ndim={C_r.ndim}")

    qdot = (q_hist[2:] - q_hist[:-2]) / (2.0 * dt)           # central 1st derivative
    qddot = (q_hist[2:] - 2.0 * q_hist[1:-1] + q_hist[:-2]) / (dt ** 2)  # central 2nd derivative
    q_mid = q_hist[1:-1]

    F_nl = F_nl_fn(q_mid) if F_nl_fn is not None else np.zeros_like(q_mid)
    residual = qddot + qdot @ C_mat.T + q_mid * Lambda[None, :] + F_nl - F_ext_hist[1:-1]
    return t[1:-1], residual


# =====================================================================
# 2. FNO1d building blocks -- torch-optional.
# =====================================================================
if _HAS_TORCH:

    class SpectralConv1d(_nn.Module):
        """One Fourier layer's kernel-integral operator (FRINO Eq.
        12-15 / Li et al. 2020's own FNO): truncate the input's rFFT
        (along the TIME axis) to the lowest `modes` frequencies,
        multiply each retained mode by an independently-learned
        complex weight (a full `(in_channels, out_channels)` matrix
        per mode, not a shared scalar), zero every higher frequency,
        then invert. This is exactly what makes an FNO layer
        discretization-aware in the FOURIER sense (though NOT the same
        as RONOM's own discretization-ROBUSTNESS property, item 118 --
        here the truncation is still tied to a fixed number of time
        steps at both train and inference).
        """

        def __init__(self, in_channels, out_channels, modes):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes = modes
            scale = 1.0 / (in_channels * out_channels)
            # Real/imag parts stored separately -- torch's own complex
            # autograd support is solid, but keeping this in real
            # tensors avoids any complex-dtype/optimizer edge cases.
            self.weight_re = _nn.Parameter(
                scale * torch.randn(modes, in_channels, out_channels, dtype=torch.float64))
            self.weight_im = _nn.Parameter(
                scale * torch.randn(modes, in_channels, out_channels, dtype=torch.float64))

        def forward(self, x):
            """x : (batch, in_channels, n_time) -> (batch, out_channels, n_time)."""
            batch, _, n_time = x.shape
            x_ft = torch.fft.rfft(x, dim=-1)                 # (batch, in_channels, n_freq)
            n_freq = x_ft.shape[-1]
            m = min(self.modes, n_freq)

            weight = torch.complex(self.weight_re[:m], self.weight_im[:m])  # (m, in, out)
            out_ft = torch.zeros(batch, self.out_channels, n_freq,
                                  dtype=torch.complex128, device=x.device)
            # einsum over the retained modes only: for each of the m
            # lowest frequencies, out[:, o, k] = sum_i x_ft[:, i, k] * weight[k, i, o]
            out_ft[:, :, :m] = torch.einsum("bik,kio->bok", x_ft[:, :, :m], weight)
            return torch.fft.irfft(out_ft, n=n_time, dim=-1)


    class FNO1dBlock(_nn.Module):
        """SpectralConv1d + a pointwise (kernel=1) Conv1d skip
        connection + nonlinearity -- one Fourier-layer iteration of
        FRINO Eq. 13, `nu_{i+1} = sigma(W nu_i + K_phi(b) nu_i)`."""

        def __init__(self, width, modes, activation=None):
            super().__init__()
            self.spectral = SpectralConv1d(width, width, modes)
            self.skip = _nn.Conv1d(width, width, kernel_size=1, dtype=torch.float64)
            self.act = activation or _nn.GELU()

        def forward(self, x):
            return self.act(self.spectral(x) + self.skip(x))


    class ExcitationResponseOperator(_nn.Module):
        """FRINO-style operator: lifts a reduced excitation time series
        `F_ext(t)` (shape `(n_modes, n_time)`) into a wider latent
        channel space, applies `n_layers` `FNO1dBlock`s, and projects
        back down to a predicted `q_nl(t)` trajectory -- one trained
        model, many excitation waveforms, no re-time-marching needed
        at inference (`predict()` is a single forward pass).

        Parameters
        ----------
        n_modes : int
            Number of retained reduced modes (both input and output
            channel count).
        width : int, default 16
            Latent channel width inside the Fourier layers.
        modes : int, default 8
            Number of retained Fourier modes per `SpectralConv1d` --
            must be <= n_time//2 + 1 at both train and predict time
            (checked, not silently truncated further, in `fit()`).
        n_layers : int, default 3
            Number of stacked `FNO1dBlock`s.
        lr, n_epochs, seed, device : same convention as
            `nonlinear_rom.NeuralSurrogate`.
        """

        def __init__(self, n_modes, width=16, modes=8, n_layers=3,
                     lr=1e-3, n_epochs=1000, seed=None, device="cpu"):
            _require_torch()
            super().__init__()
            if seed is not None:
                torch.manual_seed(seed)
            self.n_modes = n_modes
            self.width = width
            self.modes = modes
            self.n_layers = n_layers
            self.lr = lr
            self.n_epochs = n_epochs
            self.device = device

            self.lift = _nn.Conv1d(n_modes, width, kernel_size=1, dtype=torch.float64)
            self.blocks = _nn.ModuleList([FNO1dBlock(width, modes) for _ in range(n_layers)])
            self.project = _nn.Sequential(
                _nn.Conv1d(width, width, kernel_size=1, dtype=torch.float64),
                _nn.GELU(),
                _nn.Conv1d(width, n_modes, kernel_size=1, dtype=torch.float64),
            )
            self.to(device)
            self.q_mean = self.q_std = self.F_mean = self.F_std = None
            self.loss_history = None

        def forward(self, F_ext_t):
            """F_ext_t : (batch, n_modes, n_time) torch tensor (already
            normalized, if the caller wants normalization) -> predicted
            q_nl, same shape."""
            x = self.lift(F_ext_t)
            for block in self.blocks:
                x = block(x)
            return self.project(x)

        def fit(self, t, F_ext_trajectories, q_nl_trajectories,
                Lambda=None, C_r=None, force_model=None, phys_weight=0.0,
                n_epochs=None, verbose=False):
            """Train on a SWEEP of excitation trajectories (the whole
            point of this class: one fit() call, many waveforms).

            Parameters
            ----------
            t : (n_time,) array_like, shared uniform time grid.
            F_ext_trajectories, q_nl_trajectories : (n_runs, n_time, n_modes)
                array_like -- paired excitation/response trajectories,
                e.g. produced by `nonlinear_dynamics.integrate_newmark_
                surrogate` at several different excitation waveforms
                (single-frequency, multi-frequency, band-limited random
                via `random_vibration.band_limited_gaussian_time_
                history`, ...) -- mirroring FRINO's own excitation-
                sweep training set.
            Lambda, C_r, force_model : optional, needed only when
                `phys_weight > 0`. `force_model` must expose a
                `.coeffs`/`.quad_idx`/`.cub_idx` triple (i.e. a fitted
                `nonlinear_rom.PolynomialModalROM`) -- its ALREADY-
                FITTED polynomial coefficients are evaluated via a
                torch-differentiable mirror (`_polynomial_force_torch`
                below) so the physics-residual term backpropagates
                into this operator's own weights, not just the data
                loss. A `None` force_model with `phys_weight > 0`
                penalizes the LINEAR part of the residual only
                (`F_nl == 0`).
            phys_weight : float, default 0.0
                Soft physics-residual loss weight (FRINO's own `beta`,
                Eq. 30) -- 0 (default) trains on data loss alone.
                Applied to the residual's RAW physical units (`data_
                loss` is on normalized data, `phys_loss` is not) --
                deliberately NOT auto-scaled to be "comparable" to
                data_loss, after a real test run showed that an
                auto-scaling attempt measurably hurt results on a real
                fixture (see `test_neural_operator.py::
                TestExcitationResponseOperator::test_generalizes_to_an_
                unseen_excitation_frequency`'s own docstring for the
                full trail). In practice this means a useful
                `phys_weight` value is fixture-specific (force/mass
                unit scale dependent) and needs tuning per problem, not
                a fixed default that "just works" -- an honest, current
                limitation of this loss interface.
            n_epochs : int, optional
                Overrides the constructor default for this call.

            Returns
            -------
            self
            """
            t = np.asarray(t, dtype=float)
            F_ext_trajectories = np.asarray(F_ext_trajectories, dtype=float)
            q_nl_trajectories = np.asarray(q_nl_trajectories, dtype=float)
            if F_ext_trajectories.ndim != 3 or q_nl_trajectories.ndim != 3:
                raise ValueError("F_ext_trajectories/q_nl_trajectories must be 3-D (n_runs, n_time, n_modes)")
            n_runs, n_time, n_modes = F_ext_trajectories.shape
            if n_modes != self.n_modes:
                raise ValueError(f"trajectories have n_modes={n_modes}, expected {self.n_modes}")
            if q_nl_trajectories.shape != F_ext_trajectories.shape:
                raise ValueError("F_ext_trajectories and q_nl_trajectories must have the same shape")
            n_freq = n_time // 2 + 1
            if self.modes > n_freq:
                raise ValueError(
                    f"modes={self.modes} exceeds n_time//2+1={n_freq} for n_time={n_time} "
                    "-- reduce `modes` or supply a longer trajectory")
            if t.shape != (n_time,):
                raise ValueError(f"t must have shape ({n_time},), got {t.shape}")
            dts = np.diff(t)
            if len(dts) and (dts[0] <= 0 or not np.allclose(dts, dts[0], rtol=1e-8, atol=1e-12)):
                raise ValueError("t must be uniformly spaced with a positive step")

            self.F_mean, self.F_std = _fit_normalization_2d(F_ext_trajectories)
            self.q_mean, self.q_std = _fit_normalization_2d(q_nl_trajectories)
            F_n = _normalize_2d(F_ext_trajectories, self.F_mean, self.F_std)
            q_n = _normalize_2d(q_nl_trajectories, self.q_mean, self.q_std)

            F_t = torch.tensor(F_n, dtype=torch.float64, device=self.device).permute(0, 2, 1)
            q_t = torch.tensor(q_n, dtype=torch.float64, device=self.device).permute(0, 2, 1)

            use_phys = phys_weight > 0.0
            if use_phys:
                if Lambda is None or C_r is None:
                    raise ValueError("phys_weight > 0 requires Lambda and C_r")
                Lambda_t = torch.tensor(np.asarray(Lambda, dtype=float), dtype=torch.float64)
                C_r_arr = np.asarray(C_r, dtype=float)
                C_mat_t = torch.tensor(
                    np.diag(C_r_arr) if C_r_arr.ndim == 1 else C_r_arr, dtype=torch.float64)
                dt = float(t[1] - t[0])
                F_ext_t_raw = torch.tensor(F_ext_trajectories, dtype=torch.float64, device=self.device)

            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            epochs = self.n_epochs if n_epochs is None else n_epochs
            history = []
            self.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                pred_n = self.forward(F_t)                        # (n_runs, n_modes, n_time)
                data_loss = torch.mean((pred_n - q_t) ** 2)
                loss = data_loss
                if use_phys:
                    q_std_t = torch.tensor(self.q_std, dtype=torch.float64)
                    q_mean_t = torch.tensor(self.q_mean, dtype=torch.float64)
                    pred_phys = pred_n.permute(0, 2, 1) * q_std_t + q_mean_t  # de-normalized, (n_runs, n_time, n_modes)
                    qdot = (pred_phys[:, 2:, :] - pred_phys[:, :-2, :]) / (2.0 * dt)
                    qddot = (pred_phys[:, 2:, :] - 2.0 * pred_phys[:, 1:-1, :] + pred_phys[:, :-2, :]) / (dt ** 2)
                    q_mid = pred_phys[:, 1:-1, :]
                    F_nl = (_polynomial_force_torch(force_model, q_mid)
                            if force_model is not None else torch.zeros_like(q_mid))
                    resid = (qddot + torch.einsum("btm,cm->btc", qdot, C_mat_t)
                              + q_mid * Lambda_t[None, None, :] + F_nl
                              - F_ext_t_raw[:, 1:-1, :])
                    # NOTE: `resid` is on RAW physical units, not
                    # divided by any normalization scale, and
                    # `phys_weight` is applied directly to its mean
                    # square. An earlier version of this code divided
                    # by the per-mode F_std used for input
                    # normalization first, reasoning that `data_loss`
                    # (on zero-mean/unit-std data) and a raw-units
                    # `phys_loss` are not directly comparable -- a
                    # reasonable-sounding hypothesis that a REAL
                    # torch-equipped test run directly falsified: on
                    # `test_neural_operator.py`'s own generalization
                    # fixture, the "corrected" scaled version measured
                    # WORSE (R^2=-1.65) than this raw version
                    # (R^2=-0.66) at the same phys_weight=1.0 and
                    # modes=20 -- meaning the raw residual's own scale
                    # already put it in a useful range here, and
                    # dividing by F_std (small on this fixture) made
                    # the physics term dominate MORE, not less,
                    # over-constraining training. Reverted rather than
                    # kept as a "more correct but empirically worse"
                    # default -- see that test's own docstring for the
                    # full trail of evidence. `phys_weight` therefore
                    # remains a fixture-specific, raw-units multiplier
                    # to be tuned empirically, not a normalized
                    # relative weight -- a real, documented limitation
                    # of this training-loss interface, not a bug to
                    # silently paper over.
                    phys_loss = torch.mean(resid ** 2)
                    loss = data_loss + phys_weight * phys_loss
                loss.backward()
                optimizer.step()
                history.append(float(loss.item()))
                if verbose and (epoch % max(1, epochs // 10) == 0):
                    print(f"  ExcitationResponseOperator.fit: epoch {epoch}/{epochs} loss={loss.item():.6e}")
            self.loss_history = np.array(history)
            self.eval()
            return self

        def predict(self, t, F_ext_trajectory):
            """t : (n_time,) ; F_ext_trajectory : (n_time, n_modes), a
            SINGLE excitation waveform, possibly one never seen during
            training (the actual generalization claim this item
            exists to support) -> predicted (n_time, n_modes) q_nl."""
            if self.F_mean is None:
                raise RuntimeError("ExcitationResponseOperator.predict() called before fit()")
            F_ext_trajectory = np.asarray(F_ext_trajectory, dtype=float)
            F_n = _normalize_2d(F_ext_trajectory[None, :, :], self.F_mean, self.F_std)
            with torch.no_grad():
                F_t = torch.tensor(F_n, dtype=torch.float64, device=self.device).permute(0, 2, 1)
                pred_n = self.forward(F_t).permute(0, 2, 1).cpu().numpy()[0]
            return _denormalize_2d(pred_n, self.q_mean, self.q_std)


    def _polynomial_force_torch(force_model, q_t):
        """Torch-differentiable mirror of a FITTED
        `nonlinear_rom.PolynomialModalROM.force()`, evaluated on a
        torch tensor `q_t` of shape (..., n_modes) instead of a numpy
        array, using the SAME `quad_idx`/`cub_idx` monomial convention
        (`nonlinear_rom._monomial_indices`) and the model's OWN
        already-fitted `.coeffs` (held as a constant, not
        differentiated -- only `q_t`, and hence whatever produced it,
        is differentiated through). Independently re-implemented here
        rather than imported, matching this package's standing
        "reimplement the same math in the domain that needs
        differentiability" precedent (`differentiable_correction.py`'s
        own docstring states this principle explicitly for the
        fea_engine/rom_engine boundary; here the boundary is numpy
        least-squares fit vs. torch-differentiable evaluation of that
        SAME fit, within one module)."""
        coeffs_t = torch.tensor(force_model.coeffs, dtype=torch.float64)
        cols = []
        for (i, j) in force_model.quad_idx:
            cols.append(q_t[..., i] * q_t[..., j])
        for (i, j, k) in force_model.cub_idx:
            cols.append(q_t[..., i] * q_t[..., j] * q_t[..., k])
        X = torch.stack(cols, dim=-1)                # (..., n_terms)
        return torch.einsum("...t,tm->...m", X, coeffs_t)


def _fit_normalization_2d(X):
    """Per-mode mean/std over a (n_runs, n_time, n_modes) array,
    pooling across BOTH n_runs and n_time -- the shape-agnostic
    generalization of nonlinear_rom._fit_normalization (which only
    ever saw a plain (N, n_modes) table) to a batch of trajectories.
    Deliberately factored out as a plain-NumPy function so it stays
    unit-testable without torch, mirroring nonlinear_rom._fit_
    normalization's own docstring rationale."""
    X = np.asarray(X, dtype=float)
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def _normalize_2d(X, mean, std):
    return (X - mean) / std


def _denormalize_2d(Xn, mean, std):
    return Xn * std + mean
