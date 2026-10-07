"""
parameterized_latent_ode.py -- Wave 13 item 119 (fea_engine/docs/
consolidated_future_roadmap.md): a parameterized latent-ODE model,
`dz/dt = f(t, xi, z)`, trained across a SWEEP of a held-fixed-per-
trajectory physical parameter `xi` (a load amplitude, a geometric or
material property, ...) via curriculum learning -- adapted from the
sibling `generative-prediction-of-laser-induced-dynamic-latent-space-
representations/L-NeuralODE` repository's architecture, reviewed
during Wave 13's own scoping pass.

WHY THIS IS NOT A DIRECT PORT (see the roadmap's own "Key mismatches"
note for Wave 13): that repository's own `models.py::rk4()` has a real
bug -- `k2`/`k3` are BOTH evaluated at the un-advanced state `y`
instead of at the midpoint-corrected states a genuine 4-stage RK4
step needs, silently degrading the integrator below its claimed
4th-order accuracy. `rk4_step()`/`integrate_rk4()` below are a
CORRECTLY implemented, independently-derived generic RK4 (validated
directly against the exact order-of-convergence a real RK4 step must
have -- see this module's own tests), not a transcription of the
buggy version.

Deliberately does NOT claim to satisfy `nonlinear_rom.
ReducedForceModel`'s protocol: that protocol is an instantaneous
state->force map, re-time-marched by `nonlinear_dynamics.
integrate_newmark_surrogate` for every new load case. This class
instead predicts a WHOLE TRAJECTORY directly from `(z0, xi)` in one
forward pass, with no external M/C/K structure imposed on `f` at
all -- a genuinely different, less physically-constrained trade-off,
discussed honestly in this module's own docstring and in item 120's
cross-configuration validation.

Two independently useful pieces, split the same numpy-core/torch-
optional way every other trainable component in this package is:

  - `rk4_step()`/`integrate_rk4()`/`CurriculumSchedule`: PURE NUMPY,
    unconditional. The corrected integrator and the curriculum-
    learning index bookkeeping, both fully unit-testable without
    torch.
  - `ParameterizedODEFunc`/`ParameterizedLatentODE`: torch-optional
    (only instantiating `ParameterizedLatentODE` needs torch;
    importing this module never does).
"""
import numpy as np

_HAS_TORCH = False
try:
    import torch
    import torch.nn as _nn
    _HAS_TORCH = True
except Exception:
    # See nonlinear_rom.NeuralSurrogate's own comment on this exact
    # except clause.
    pass


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "ParameterizedLatentODE requires PyTorch, which is not "
            "importable in this environment. Install it (`pip install "
            "torch`) -- rk4_step()/integrate_rk4()/CurriculumSchedule "
            "themselves do NOT need torch (pure NumPy) and work "
            "unconditionally.")


# =====================================================================
# 1. Correctly-implemented generic RK4 -- pure NumPy, unconditional.
# =====================================================================
def rk4_step(func, t, dt, y, *args):
    """One step of the classical 4-stage, 4th-order Runge-Kutta method
    for `dy/dt = func(t, y, *args)`.

    THE FIX (see module docstring): a genuine RK4 step evaluates `k2`
    and `k3` at states ADVANCED by the previous stage's own slope --
    `k2 = func(t+dt/2, y+dt/2*k1, ...)`, `k3 = func(t+dt/2,
    y+dt/2*k2, ...)` -- not at the same un-advanced `y` for all four
    stages (the `L-NeuralODE` reference implementation's own bug,
    which collapses k2 and k3 to identical values and gives up the
    4th-order accuracy the formula's own name promises).

    Returns
    -------
    y_new : same shape as y
    """
    k1 = func(t, y, *args)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
    k4 = func(t + dt, y + dt * k3, *args)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_rk4(func, y0, t, *args):
    """Integrate `dy/dt = func(t, y, *args)` from `y0` over the time
    vector `t` (not necessarily uniformly spaced -- each step uses its
    own local `dt`), via repeated `rk4_step()` calls.

    Parameters
    ----------
    func : callable(t, y, *args) -> dy/dt, same shape as y
    y0 : (n_vars,) array_like
    t : (n_steps,) array_like

    Returns
    -------
    y_hist : (n_steps, n_vars) ndarray
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y0, dtype=float).copy()
    n_steps = len(t)
    y_hist = np.zeros((n_steps,) + y.shape)
    y_hist[0] = y
    for i in range(n_steps - 1):
        dt = t[i + 1] - t[i]
        y = rk4_step(func, t[i], dt, y, *args)
        y_hist[i + 1] = y
    return y_hist


# =====================================================================
# 2. Curriculum-learning schedule -- pure NumPy index bookkeeping.
# =====================================================================
class CurriculumSchedule:
    """Progressively extends the trained integration horizon as the
    training loss drops below a tolerance -- the `L-NeuralODE`
    reference's own stabilization technique for a fully black-box
    `dz/dt=f(...)` (no M/C/K structure to exploit, unlike every other
    `TrainingStrategy` in this package, which all sample STATES rather
    than extend TIME horizons -- see the roadmap's own item 119 row).

    Reimplemented here as a small, independently-testable state
    machine (not a direct transcription of the reference's own
    `determine_curriculum_times()`/training-loop interleaving, which
    used a `-1` sentinel for "full horizon reached" and always
    advanced on the very first call regardless of loss -- both
    replaced here by explicit, unambiguous state: `is_full_horizon`
    is a real boolean, and the FIRST call to `maybe_advance()` only
    advances if the caller's own initial loss is already below `tol`
    OR via the explicit `advance()` a caller can use to seed stage 1
    before any training has happened, matching how `train()` below
    actually uses it).

    Parameters
    ----------
    n_steps : int
        Total number of time steps in the FULL trajectory (matching
        `t`'s own length, i.e. `len(t) - 1` intervals).
    n_folds : int, default 10
        Number of curriculum stages between the shortest and the full
        horizon.
    """

    def __init__(self, n_steps, n_folds=10):
        if n_steps < 2:
            raise ValueError("n_steps must be >= 2")
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        self.n_steps = n_steps
        self.n_folds = n_folds
        self._fold_upper = np.unique(
            np.clip(np.round(np.linspace(1, n_folds, n_folds) / n_folds * n_steps).astype(int),
                    2, n_steps))
        self.stage = 0
        self.upper_index = int(self._fold_upper[0])

    @property
    def is_full_horizon(self):
        return self.upper_index >= self.n_steps

    def advance(self):
        """Unconditionally move to the next stage (a no-op once
        `is_full_horizon` is already True). Returns whether the stage
        actually changed."""
        if self.is_full_horizon:
            return False
        self.stage = min(self.stage + 1, len(self._fold_upper) - 1)
        new_upper = int(self._fold_upper[self.stage])
        changed = new_upper != self.upper_index
        self.upper_index = new_upper
        return changed

    def maybe_advance(self, loss, tol):
        """Advance one stage if `loss < tol` and not already at full
        horizon. Returns whether it advanced."""
        if self.is_full_horizon:
            return False
        if loss < tol:
            return self.advance()
        return False


# =====================================================================
# 3. Parameterized ODE function + trainable latent-ODE -- torch-optional.
# =====================================================================
if _HAS_TORCH:

    class ParameterizedODEFunc(_nn.Module):
        """`dz/dt = f(t, xi, z)`: a small feedforward MLP, input
        `[t, xi, z]` concatenated -- the SAME architecture the
        `L-NeuralODE` reference's own `ODEFunc` uses, but with `xi`
        kept as a torch tensor throughout `forward()` (the reference's
        own `forward()` called `torch.from_numpy(xi.transpose())`
        INSIDE the ODE function, evaluated at every RK4 stage of every
        step -- a real inefficiency and a design smell this
        reimplementation avoids by keeping everything in torch from
        the caller's own `fit()`/`predict()` boundary inward)."""

        def __init__(self, n_xi_features, n_vars, hidden_sizes=(64, 64), activation="tanh"):
            super().__init__()
            self.n_xi_features = n_xi_features
            self.n_vars = n_vars
            act_cls = {"tanh": _nn.Tanh, "relu": _nn.ReLU, "silu": _nn.SiLU}[activation]
            sizes = (n_xi_features + 1 + n_vars,) + tuple(hidden_sizes) + (n_vars,)
            layers = []
            for i in range(len(sizes) - 1):
                layers.append(_nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:
                    layers.append(act_cls())
            self.net = _nn.Sequential(*layers).double()

        def forward(self, t, xi, z):
            """t : a Python float/int OR a 0-D tensor -- `_rk4_step_torch`
            below passes a plain Python float (RK4's own `t+0.5*dt`
            arithmetic on the float `t` threaded through `_rollout()`),
            so this must accept both rather than assume a tensor (a
            real bug caught by an actual torch-equipped test run: `t`
            arriving as a bare float made `t.reshape(...)` raise
            `AttributeError: 'float' object has no attribute
            'reshape'` -- `torch.as_tensor` below fixes this for
            either input type, and is a no-op wrap for an existing
            tensor). xi : (batch, n_xi_features). z : (batch, n_vars).
            Returns dz/dt, (batch, n_vars)."""
            batch = z.shape[0]
            t_tensor = torch.as_tensor(t, dtype=torch.float64, device=z.device)
            t_col = t_tensor.reshape(1, 1).expand(batch, 1).double()
            txiz = torch.cat([t_col, xi, z], dim=1)
            return self.net(txiz)


    def _rk4_step_torch(func, t, dt, z, xi):
        """Torch mirror of rk4_step() above, same corrected formula,
        differentiable end-to-end through `func`'s own parameters."""
        k1 = func(t, xi, z)
        k2 = func(t + 0.5 * dt, xi, z + 0.5 * dt * k1)
        k3 = func(t + 0.5 * dt, xi, z + 0.5 * dt * k2)
        k4 = func(t + dt, xi, z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


    class ParameterizedLatentODE:
        """Trains and evaluates a `ParameterizedODEFunc` across a
        SWEEP of trajectories, each with its own held-fixed parameter
        vector `xi`, via curriculum learning (`CurriculumSchedule`).

        Parameters
        ----------
        n_xi_features, n_vars : int
            Dimensions of `xi` and the latent state `z`.
        hidden_sizes, activation : passed to `ParameterizedODEFunc`.
        lr, seed, device : same convention as `nonlinear_rom.
            NeuralSurrogate`.
        """

        def __init__(self, n_xi_features, n_vars, hidden_sizes=(64, 64),
                     activation="tanh", lr=1e-3, seed=None, device="cpu"):
            _require_torch()
            if seed is not None:
                torch.manual_seed(seed)
            self.n_xi_features = n_xi_features
            self.n_vars = n_vars
            self.lr = lr
            self.device = device
            self.func = ParameterizedODEFunc(n_xi_features, n_vars, hidden_sizes, activation).to(device)
            self.loss_history = None

        def _rollout(self, z0_t, xi_t, t, upper_index):
            """z0_t : (batch, n_vars). xi_t : (batch, n_xi_features).
            t : (n_steps,) ndarray (full horizon). Returns predicted
            trajectory (upper_index, batch, n_vars) as a torch tensor,
            differentiable."""
            z = z0_t
            traj = [z]
            for i in range(upper_index - 1):
                dt = float(t[i + 1] - t[i])
                z = _rk4_step_torch(self.func, float(t[i]), dt, z, xi_t)
                traj.append(z)
            return torch.stack(traj, dim=0)

        def fit(self, t, z0_samples, xi_samples, z_traj_samples,
                n_iters=2000, n_folds=10, curric_tol=1e-3, batch_size=None,
                verbose=False):
            """Train across a sweep of trajectories.

            Parameters
            ----------
            t : (n_steps,) array_like, shared uniform-ISH time grid
                (per-step dt is read directly from `t`, so non-uniform
                grids work too, matching integrate_rk4()'s own
                generality).
            z0_samples : (n_runs, n_vars) array_like
            xi_samples : (n_runs, n_xi_features) array_like
            z_traj_samples : (n_runs, n_steps, n_vars) array_like
                Ground-truth trajectories, one per run, starting at
                the matching row of `z0_samples` under the matching
                row of `xi_samples`.
            n_iters : int, default 2000
            n_folds : int, default 10
                `CurriculumSchedule`'s own `n_folds`.
            curric_tol : float, default 1e-3
                Loss threshold to advance the curriculum stage.
            batch_size : int, optional
                Defaults to all runs (full-batch), matching
                `NeuralSurrogate.fit()`'s own default reasoning for
                this package's typically small (O(10-100)) training
                sets.

            Returns
            -------
            self
            """
            t = np.asarray(t, dtype=float)
            z0_samples = np.asarray(z0_samples, dtype=float)
            xi_samples = np.asarray(xi_samples, dtype=float)
            z_traj_samples = np.asarray(z_traj_samples, dtype=float)
            n_runs, n_steps, n_vars = z_traj_samples.shape
            if n_vars != self.n_vars:
                raise ValueError(f"z_traj_samples has n_vars={n_vars}, expected {self.n_vars}")
            if z0_samples.shape != (n_runs, n_vars):
                raise ValueError(f"z0_samples must have shape {(n_runs, n_vars)}, got {z0_samples.shape}")
            if xi_samples.shape != (n_runs, self.n_xi_features):
                raise ValueError(
                    f"xi_samples must have shape {(n_runs, self.n_xi_features)}, got {xi_samples.shape}")
            if t.shape != (n_steps,):
                raise ValueError(f"t must have shape ({n_steps},), got {t.shape}")

            z0_t = torch.tensor(z0_samples, dtype=torch.float64, device=self.device)
            xi_t = torch.tensor(xi_samples, dtype=torch.float64, device=self.device)
            z_traj_t = torch.tensor(z_traj_samples, dtype=torch.float64, device=self.device)

            schedule = CurriculumSchedule(n_steps, n_folds=n_folds)
            bs = batch_size or n_runs
            optimizer = torch.optim.Adam(self.func.parameters(), lr=self.lr)

            history = []
            loss_val = float("inf")
            for it in range(n_iters):
                schedule.maybe_advance(loss_val, curric_tol)
                idx = np.random.default_rng(it).choice(n_runs, size=min(bs, n_runs), replace=False)
                z0_b = z0_t[idx]
                xi_b = xi_t[idx]
                target = z_traj_t[idx, :schedule.upper_index, :]

                optimizer.zero_grad()
                pred = self._rollout(z0_b, xi_b, t, schedule.upper_index)   # (upper, batch, n_vars)
                pred = pred.permute(1, 0, 2)                                 # (batch, upper, n_vars)
                loss = torch.mean((pred - target) ** 2)
                loss.backward()
                optimizer.step()
                loss_val = float(loss.item())
                history.append(loss_val)
                if verbose and it % max(1, n_iters // 10) == 0:
                    print(f"  ParameterizedLatentODE.fit: iter {it}/{n_iters} "
                          f"stage_upper_index={schedule.upper_index}/{n_steps} loss={loss_val:.6e}")

            self.loss_history = np.array(history)
            return self

        def predict(self, t, z0, xi):
            """t : (n_steps,). z0 : (n_vars,). xi : (n_xi_features,).
            -> predicted (n_steps, n_vars) trajectory, via the SAME
            corrected RK4 integrator (evaluated with `torch.no_grad()`,
            no batching needed for a single rollout)."""
            t = np.asarray(t, dtype=float)
            z0_t = torch.tensor(np.asarray(z0, dtype=float)[None, :], dtype=torch.float64, device=self.device)
            xi_t = torch.tensor(np.asarray(xi, dtype=float)[None, :], dtype=torch.float64, device=self.device)
            with torch.no_grad():
                traj = self._rollout(z0_t, xi_t, t, len(t))   # (n_steps, 1, n_vars)
            return traj[:, 0, :].cpu().numpy()
