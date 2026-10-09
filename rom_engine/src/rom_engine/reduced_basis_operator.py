# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
reduced_basis_operator.py -- Wave 13 item 118 (fea_engine/docs/
consolidated_future_roadmap.md): a discretization-robust encoder/
decoder pair, adapting RONOM's (Dummer, Ye & Brune, "RONOM: Reduced-
Order Neural Operator Modeling") architectural pattern -- encode a
field sampled at ANY set of points into a finite-dimensional latent
code via a regularized L2 projection onto a CONTINUOUS basis, then
decode that latent code back to a field value at ANY OTHER set of
points, including points never seen during fitting.

WHY THIS MATTERS (see the roadmap's own Wave 13 framing): every
existing `rom_engine` reduced model -- `pod.PodBasis`,
`nonlinear_rom.PolynomialModalROM`/`MultiFidelitySurrogate`/
`NeuralSurrogate` -- is tied to the exact mesh/point ordering its
basis or training data was built from. `PodBasis.project(x)` requires
`x` to be the SAME length and ordering as the snapshots it was fit
from; there is no way to fit a basis at one mesh resolution and query
it at a genuinely different one. RONOM's own contribution is making
that possible, at the cost of the basis functions needing a
CONTINUOUS (not just tabulated-at-nodes) representation.

SCOPE LIMITATION, stated up front rather than discovered later
(matching this project's "narrow and document" convention): this
module handles a SINGLE SCALAR FIELD per spatial coordinate (e.g. the
transverse-displacement component of a beam's deformation), not a
full multi-DOF-per-node vector field. A caller wanting to reconstruct
every DOF component (u, v, theta, ...) calls this once PER component.
This keeps the regularized-projection linear algebra a single,
unambiguous least-squares problem; RONOM's own paper (whose three
numerical examples are all single-field PDEs -- Burgers, wave
equation, and a single field of Navier-Stokes) does not attempt the
multi-field case either.

WHY A CONTINUOUS BASIS (not FE shape functions directly, though the
roadmap's own item 118 row names that as a future option): building a
literal isoparametric FE interpolant would need this module to
understand element connectivity/shape functions, which is exactly the
`fea_engine`-dependency this package's whole design (see package
docstring) forbids in library code. Fitting each POD-basis column as
an independent, smooth, mesh-free interpolant via `scipy.interpolate.
RBFInterpolator` is a defensible, package-boundary-respecting
alternative that is EXACTLY as evaluable at an arbitrary query
coordinate as a real FE shape function would be, at the cost of not
being the exact same interpolant a specific finite-element formulation
would use element-by-element.

Two pieces, split the same numpy-core/torch-optional way every other
trainable component in this package is:

  - `RegularizedProjectionEncoder`/`decode()`: PURE NUMPY. The actual
    RONOM-pattern encoder/decoder and this item's own decisive claim
    (cross-mesh-resolution reconstruction) -- needs no torch at all.
  - `LatentRefinementNet`: torch-optional. RONOM's own `E_phi` stage
    (a small NN applied AFTER the linear projection, Eq. 2.4) -- here
    used as a bias-correction network trained to nudge the linear
    projection's own latent estimate toward a KNOWN true reduced
    coordinate (e.g. an exact POD projection), when such ground truth
    is available for training.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.interpolate import RBFInterpolator

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
            "LatentRefinementNet requires PyTorch, which is not importable "
            "in this environment. Install it (`pip install torch`) -- "
            "RegularizedProjectionEncoder itself does NOT need torch (pure "
            "NumPy) and works unconditionally.")


class RegularizedProjectionEncoder:
    """RONOM's `E := (E_phi o M o P_V^lambda)` encoder / optimal-
    recovery decoder pair (Section 2.3/2.5 of the paper), specialized
    to a basis built from an EXISTING discrete POD basis (any
    `pod.PodBasis.V` column, or a bare ndarray) made continuous via
    per-column RBF interpolation.

    Parameters (constructor takes none -- `fit()` sets everything, the
    same `PodBasis`/`MembraneBasis` convention this package already
    uses).
    """

    def __init__(self):
        self.n_modes = None
        self.coords = None
        self.V = None
        self._basis = None   # RBFInterpolator: (n_query, dim) -> (n_query, n_modes)
        self.kernel = None
        self.smoothing = None

    def fit(self, coords, V, kernel="thin_plate_spline", smoothing=1e-8):
        """Build the continuous basis representation.

        Parameters
        ----------
        coords : (n_nodes,) or (n_nodes, dim) array_like
            Spatial coordinate of each row of `V` (e.g. the x-position
            of each beam node the scalar field `V`'s rows correspond
            to). A 1-D array is treated as `dim=1` and reshaped
            internally.
        V : (n_nodes, n_modes) array_like
            A discrete basis (e.g. `pod.PodBasis.V`'s own columns,
            already restricted to ONE scalar field per node -- see
            module docstring's scope note) tabulated at `coords`.
        kernel, smoothing : passed straight to
            `scipy.interpolate.RBFInterpolator` -- `thin_plate_spline`
            (default) needs no length-scale parameter to tune, and
            `smoothing > 0` allows a small residual at the training
            nodes themselves (useful if `V`'s own columns carry any
            numerical noise), matching RONOM's own regularized `P_V^
            lambda` projection (Eq. 2.3) at the BASIS-FITTING stage,
            distinct from the regularization `encode()` applies at the
            LATENT-CODE-FITTING stage below.

        Returns
        -------
        self
        """
        coords = np.asarray(coords, dtype=float)
        if coords.ndim == 1:
            coords = coords[:, None]
        V = np.asarray(V, dtype=float)
        if V.ndim != 2:
            raise ValueError(f"V must be 2-D (n_nodes, n_modes), got shape {V.shape}")
        if coords.shape[0] != V.shape[0]:
            raise ValueError(
                f"coords has {coords.shape[0]} rows, V has {V.shape[0]} -- must match")
        self.coords = coords
        self.V = V
        self.n_modes = V.shape[1]
        self.kernel = kernel
        self.smoothing = smoothing
        self._basis = RBFInterpolator(coords, V, kernel=kernel, smoothing=smoothing)
        return self

    def _eval_basis(self, query_coords):
        query_coords = np.asarray(query_coords, dtype=float)
        if query_coords.ndim == 1:
            query_coords = query_coords[:, None]
        return self._basis(query_coords)   # (n_query, n_modes)

    def encode(self, query_coords, values, reg=1e-6):
        """Point samples `{x_i, f_i}` -> latent code `z` (RONOM Eq.
        2.5's discretized regularized projection): solves

            min_z  sum_i (phi(x_i) . z - f_i)^2 + reg * |z|^2

        via ordinary Tikhonov-regularized least squares on the basis
        evaluated at the QUERY points -- `query_coords` need not be
        `self.coords` at all (a different resolution, a subset, extra
        points, ...), which is exactly the discretization-robustness
        claim this class exists to support.

        Parameters
        ----------
        query_coords : (n_pts,) or (n_pts, dim) array_like
        values : (n_pts,) array_like
            Sampled scalar field values at `query_coords` (see module
            docstring's single-scalar-field scope note).
        reg : float, default 1e-6
            Tikhonov regularization weight.

        Returns
        -------
        z : (n_modes,) ndarray
        """
        if self._basis is None:
            raise RuntimeError("RegularizedProjectionEncoder.encode() called before fit()")
        values = np.asarray(values, dtype=float)
        Phi = self._eval_basis(query_coords)   # (n_pts, n_modes)
        if Phi.shape[0] != values.shape[0]:
            raise ValueError(
                f"query_coords has {Phi.shape[0]} points, values has {values.shape[0]} -- must match")
        A = Phi.T @ Phi + reg * np.eye(self.n_modes)
        b = Phi.T @ values
        return np.linalg.solve(A, b)

    def decode(self, z, query_coords):
        """Latent code `z` -> field value(s) at `query_coords`
        (RONOM's own "optimal recovery" decoder, Eq. 2.5's dual side) --
        again, `query_coords` may be ANY points, not just `self.coords`.

        Parameters
        ----------
        z : (n_modes,) array_like
        query_coords : (n_pts,) or (n_pts, dim) array_like

        Returns
        -------
        values : (n_pts,) ndarray
        """
        if self._basis is None:
            raise RuntimeError("RegularizedProjectionEncoder.decode() called before fit()")
        z = np.asarray(z, dtype=float)
        if z.shape != (self.n_modes,):
            raise ValueError(f"z must have shape ({self.n_modes},), got {z.shape}")
        Phi = self._eval_basis(query_coords)
        return Phi @ z

    def reconstruct(self, query_coords, values, reg=1e-6):
        """encode() then immediately decode() at the SAME points --
        the direct "how well does this basis explain this data"
        round-trip check, returned alongside the fitted `z` for
        convenience."""
        z = self.encode(query_coords, values, reg=reg)
        return self.decode(z, query_coords), z


if _HAS_TORCH:

    class LatentRefinementNet(_nn.Module):
        """RONOM's own `E_phi` stage (Eq. 2.4): a small MLP applied
        AFTER the linear regularized-projection latent estimate,
        trained here as a bias-correction network -- nudges
        `RegularizedProjectionEncoder.encode()`'s own linear-algebra
        estimate toward a KNOWN true reduced coordinate (e.g. an exact
        `pod.PodBasis.project()` value, when the caller has access to
        the full-order snapshot the sampled points came from and can
        compute that ground truth for training). Purely additive: a
        `RegularizedProjectionEncoder` used alone (no refinement net)
        is unaffected, exactly matching every other `NeuralSurrogate`-
        style torch-optional addition in this package.
        """

        def __init__(self, n_modes, hidden_sizes=(16, 16), lr=1e-3, n_epochs=1000,
                     seed=None, device="cpu"):
            _require_torch()
            super().__init__()
            if seed is not None:
                torch.manual_seed(seed)
            self.n_modes = n_modes
            self.lr = lr
            self.n_epochs = n_epochs
            self.device = device
            sizes = (n_modes,) + tuple(hidden_sizes) + (n_modes,)
            layers = []
            for i in range(len(sizes) - 1):
                layers.append(_nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:
                    layers.append(_nn.Tanh())
            # Residual/bias-correction convention: initialize the FINAL
            # layer to (near) zero so the network starts as the
            # identity map (z_refined ~= z_linear) before any training
            # -- a principled starting point, not an arbitrary one,
            # since the linear projection alone is already a
            # reasonable estimate this net should only ever nudge.
            with torch.no_grad():
                layers[-1].weight.mul_(0.01)
                layers[-1].bias.mul_(0.0)
            self.net = _nn.Sequential(*layers).double().to(device)
            self.loss_history = None

        def forward(self, z_t):
            return z_t + self.net(z_t)   # residual form

        def fit(self, z_linear_samples, z_true_samples, n_epochs=None, verbose=False):
            """z_linear_samples : (N, n_modes) -- RegularizedProjection
            Encoder.encode() outputs. z_true_samples : (N, n_modes) --
            the corresponding KNOWN true reduced coordinates. Trains
            via Adam + MSE, matching NeuralSurrogate.fit()'s own
            convention exactly."""
            z_linear_samples = np.asarray(z_linear_samples, dtype=float)
            z_true_samples = np.asarray(z_true_samples, dtype=float)
            if z_linear_samples.shape != z_true_samples.shape:
                raise ValueError("z_linear_samples and z_true_samples must have the same shape")
            if z_linear_samples.shape[1] != self.n_modes:
                raise ValueError(
                    f"samples have {z_linear_samples.shape[1]} columns, expected n_modes={self.n_modes}")

            zl_t = torch.tensor(z_linear_samples, dtype=torch.float64, device=self.device)
            zt_t = torch.tensor(z_true_samples, dtype=torch.float64, device=self.device)
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            loss_fn = _nn.MSELoss()
            epochs = self.n_epochs if n_epochs is None else n_epochs
            history = []
            self.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                pred = self.forward(zl_t)
                loss = loss_fn(pred, zt_t)
                loss.backward()
                optimizer.step()
                history.append(float(loss.item()))
                if verbose and (epoch % max(1, epochs // 10) == 0):
                    print(f"  LatentRefinementNet.fit: epoch {epoch}/{epochs} loss={loss.item():.6e}")
            self.loss_history = np.array(history)
            self.eval()
            return self

        def refine(self, z_linear):
            """z_linear : (n_modes,) or (n, n_modes) -> refined z, same
            shape convention as nonlinear_rom.NeuralSurrogate.predict()."""
            z_linear = np.asarray(z_linear, dtype=float)
            single = z_linear.ndim == 1
            Z = np.atleast_2d(z_linear)
            with torch.no_grad():
                z_t = torch.tensor(Z, dtype=torch.float64, device=self.device)
                out = self.forward(z_t).cpu().numpy()
            return out[0] if single else out
