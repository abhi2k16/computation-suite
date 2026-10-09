"""
nonlinear_rom.py -- shared regression protocol for reduced nonlinear-
force models, plus two concrete model families.

A structural nonlinear ROM built by fitting a data-driven model of the
nonlinear restoring force in reduced coordinates is a HYBRID: the
linear part stays fully intrusive (a real modal or POD basis, projected
from the actual K/M/C via pod.py/galerkin.py), and only the nonlinear
CORRECTION term is a non-intrusive, black-box surrogate fit from
sampled input/output pairs. Every published method in this family --
STEP/E-STEP/HR-STEP, ICE, RBF-based STEP, and MFS-NLROM -- shares this
one shape (see docs/nonlinear_surrogate_rom_roadmap.md Section 1 for
the literature survey this module follows).

This module provides:

  - `ReducedForceModel`: the shared, duck-typed protocol every concrete
    model implements (`fit`/`predict`/`jacobian`). Not an ABC by
    inheritance requirement -- any object with these methods works,
    matching this package's existing style (e.g. `galerkin.GalerkinROM`
    accepting either a `PodBasis` or a bare ndarray).
  - `MultiFidelitySurrogate`: an RBF surrogate of the nonlinear modal
    force AS A FUNCTION OF THE LINEAR modal displacement `q_l` (the
    MFS-NLROM approach) -- ported from this project's own validated
    `mfs-nlrom-beam` skill (`scripts/mfs_nlrom.py`'s `RBFSurrogate`),
    which reproduces He et al. (2023) on a real clamped-clamped beam
    benchmark (R^2 = 0.9996 on held-out static test data, cubic
    kernel). `KERNEL_REGISTRY` adds an analytic `dpsi/dR` for every
    kernel (needed for `jacobian()`, which the original prototype did
    not need and did not implement) and generalizes the ridge
    regularization into an explicit parameter.
  - `PolynomialModalROM`: a Nash-form quadratic+cubic polynomial fit of
    the nonlinear modal force AS A FUNCTION OF THE NONLINEAR modal
    displacement `q_nl` itself (the ICE / Shi & Mei / STEP family) --
    ported from the same skill's `scripts/ice_rom.py`'s `IceRom`.
    Unlike `MultiFidelitySurrogate`, evaluating `predict()` against an
    external force needs its own Newton-Raphson solve (the accuracy/
    speed trade-off the whole MFS-NLROM paper is built around) --
    reproduced honestly here, not hidden.
  - `TrainingStrategy` + two concrete strategies (`AppliedLoadStrategy`,
    `EnforcedDisplacementStrategy`): pluggable training-DATA generation,
    decoupled from the regression method. Neither strategy imports
    `fea_engine` -- both take a caller-supplied `fom_solver` callable
    (matching this package's design principle, stated in
    `rom_engine/__init__.py`, that `fea_engine` is used only in
    `tests/`/`examples/`).

Fidelity note (matching the mfs-nlrom-beam skill's own honesty
convention): the reference paper's Eq. 8 describes a general
multi-fidelity identity `y_H(x) = rho*y_L(x) + d(x)` with a free scaling
factor `rho`; the skill's own validated reproduction (and this module,
which ports it) uses the `rho=1` specialization throughout, because the
free-`rho` generalization's exact intended form could not be recovered
from the source PDF's OCR-damaged equation blocks (the same gap
documented in the skill's own `SKILL.md` fidelity notes). This is
recorded here rather than silently assumed away.
"""
__author__ = "Abhijeet"
import numpy as np
from itertools import combinations_with_replacement
from collections import Counter

from .sampling import optimal_lhs, modal_force_samples


# =====================================================================
# 1. Kernel registry (Section 3.2) -- psi(R) and its analytic dpsi/dR
# =====================================================================
def _psi_multiquadric(R, sigma):
    return np.sqrt(R ** 2 + sigma ** 2)


def _dpsi_multiquadric(R, sigma):
    return R / np.sqrt(R ** 2 + sigma ** 2)


def _psi_cubic(R, sigma):
    return R ** 3


def _dpsi_cubic(R, sigma):
    return 3.0 * R ** 2


def _psi_thin_plate_spline(R, sigma):
    out = np.zeros_like(R)
    nz = R > 1e-12
    out[nz] = R[nz] ** 2 * np.log(R[nz])
    return out


def _dpsi_thin_plate_spline(R, sigma):
    # d/dR [R^2 log R] = R(2 log R + 1) -> 0 as R -> 0 (the correct limit,
    # not a singularity, so R=0 is safe to set to 0 explicitly).
    out = np.zeros_like(R)
    nz = R > 1e-12
    out[nz] = R[nz] * (2.0 * np.log(R[nz]) + 1.0)
    return out


def _psi_gaussian(R, sigma):
    return np.exp(-R ** 2 / (2.0 * sigma ** 2))


def _dpsi_gaussian(R, sigma):
    return -(R / sigma ** 2) * np.exp(-R ** 2 / (2.0 * sigma ** 2))


KERNEL_REGISTRY = {
    "multiquadric": (_psi_multiquadric, _dpsi_multiquadric),
    "cubic": (_psi_cubic, _dpsi_cubic),               # the paper's "BH" kernel
    "thin_plate_spline": (_psi_thin_plate_spline, _dpsi_thin_plate_spline),
    "gaussian": (_psi_gaussian, _dpsi_gaussian),
}


# =====================================================================
# 2. The shared protocol (Section 3.1) -- documentation only, no
#    enforced base class, matching this package's duck-typed style.
# =====================================================================
class ReducedForceModel:
    """Protocol every nonlinear-force model in this family implements.
    `nonlinear_dynamics.py` and `nnm.py` (future modules) are written
    ONLY against this protocol -- they never know or care whether the
    concrete model is `MultiFidelitySurrogate`, `PolynomialModalROM`,
    or a hand-wrapped call into a full nonlinear FE model's own
    internal-force assembly.

    Not meant to be subclassed for enforcement -- any object with
    `predict()` (always) and `jacobian()` (where meaningful) works.
    """

    def fit(self, q_samples, F_samples, **kwargs):
        """Train the model from paired reduced-displacement / reduced
        nonlinear-force samples. Returns self (chainable, matching
        `pod.PodBasis.fit()`/`galerkin.GalerkinROM.reduce_system()`).
        NOTE: which q-domain (`q_l` or `q_nl`) `q_samples` must be in
        is MODEL-SPECIFIC -- see each concrete class's own docstring."""
        raise NotImplementedError

    def predict(self, q):
        """Reduced displacement(s) -> predicted nonlinear force(s). `q`
        may be `(r,)` or `(n, r)` for a batch."""
        raise NotImplementedError

    def jacobian(self, q):
        """dF_nl/dq at a single point q, shape (r, r)."""
        raise NotImplementedError


# =====================================================================
# 3. MultiFidelitySurrogate -- RBF fit of F_nl(q_l) (Section 3.2)
# =====================================================================
class MultiFidelitySurrogate(ReducedForceModel):
    """RBF interpolant of the nonlinear modal force as a function of
    the LINEAR modal displacement `q_l` -- ported from the validated
    `mfs-nlrom-beam` skill's `RBFSurrogate`. Fits ONE shared kernel
    matrix (all outputs share the same centers = training `q_l`
    samples) with an independent weight column per retained mode, then
    solves `predict()` in closed form -- no Newton-Raphson needed,
    exactly the property that makes this method fast at evaluation
    time (contrast `PolynomialModalROM.predict(F_ext=...)` below).

    Parameters
    ----------
    kernel : str, one of KERNEL_REGISTRY's keys, default "multiquadric"
        Which radial basis function to use.
    sigma : float, optional
        Kernel shape/width parameter (unused by the "cubic" kernel).
        If None (default), set at fit() time to the mean nonzero
        pairwise distance between training centers -- a standard,
        data-adaptive default (matches the validated prototype).
    ridge : float, default 1e-10
        Relative ridge regularization added to the kernel matrix's
        diagonal before solving (`ridge * trace(Psi)/N`) -- kernel
        matrices can be near-singular with few training points,
        especially for the thin-plate-spline/cubic kernels; this
        matches the validated prototype's own stabilization.
    """

    def __init__(self, kernel="multiquadric", sigma=None, ridge=1e-10):
        if kernel not in KERNEL_REGISTRY:
            raise ValueError(f"kernel={kernel!r} not in KERNEL_REGISTRY {list(KERNEL_REGISTRY)}")
        self.kernel = kernel
        self.sigma = sigma
        self.ridge = ridge
        self.centers = None
        self.weights = None

    def fit(self, q_l_samples, F_nl_samples):
        """q_l_samples : (N, r) linear modal displacement training
        points. F_nl_samples : (N, r) corresponding nonlinear modal
        force samples (see module docstring / TrainingStrategy for how
        these are generated from a full-order model)."""
        q_l_samples = np.asarray(q_l_samples, dtype=float)
        F_nl_samples = np.asarray(F_nl_samples, dtype=float)
        if q_l_samples.ndim != 2 or F_nl_samples.ndim != 2:
            raise ValueError("q_l_samples and F_nl_samples must both be 2-D (N, r)")
        if q_l_samples.shape[0] != F_nl_samples.shape[0]:
            raise ValueError(
                f"q_l_samples and F_nl_samples must have the same number of rows, "
                f"got {q_l_samples.shape[0]} and {F_nl_samples.shape[0]}")
        n = q_l_samples.shape[0]
        self.centers = q_l_samples.copy()
        R = np.linalg.norm(q_l_samples[:, None, :] - q_l_samples[None, :, :], axis=-1)
        if self.sigma is None:
            self.sigma = float(np.mean(R[R > 0])) if np.any(R > 0) else 1.0
        psi, _ = KERNEL_REGISTRY[self.kernel]
        Psi = psi(R, self.sigma)
        Psi_reg = Psi + self.ridge * np.eye(n) * np.trace(Psi) / n
        self.weights = np.linalg.solve(Psi_reg, F_nl_samples)  # (N, r)
        return self

    def predict(self, q_l):
        """q_l : (r,) or (n, r). Returns matching-shape F_nl."""
        if self.centers is None:
            raise RuntimeError("MultiFidelitySurrogate.predict() called before fit()")
        q_l = np.asarray(q_l, dtype=float)
        single = q_l.ndim == 1
        Q = np.atleast_2d(q_l)
        R = np.linalg.norm(Q[:, None, :] - self.centers[None, :, :], axis=-1)
        psi, _ = KERNEL_REGISTRY[self.kernel]
        F_nl = psi(R, self.sigma) @ self.weights
        return F_nl[0] if single else F_nl

    def jacobian(self, q_l):
        """dF_nl/dq_l at a single point q_l, shape (r_out, r_in)."""
        if self.centers is None:
            raise RuntimeError("MultiFidelitySurrogate.jacobian() called before fit()")
        q_l = np.asarray(q_l, dtype=float)
        if q_l.ndim != 1:
            raise ValueError("jacobian() takes a single point (r,), not a batch")
        diff = q_l[None, :] - self.centers            # (N, r_in)
        R = np.linalg.norm(diff, axis=1)               # (N,)
        _, dpsi = KERNEL_REGISTRY[self.kernel]
        dpsi_dR = dpsi(R, self.sigma)                   # (N,)
        # direction (q - c_j)/R_j, safely 0 where R_j == 0 (every
        # registered kernel's dpsi/dR -> 0 as R -> 0, so the product is
        # correctly 0 there too, not a 0/0 indeterminate form)
        with np.errstate(invalid="ignore", divide="ignore"):
            direction = np.where(R[:, None] > 0, diff / np.where(R[:, None] > 0, R[:, None], 1.0), 0.0)
        # d psi(R_j)/dq_in = dpsi_dR[j] * direction[j, :]   -> (N, r_in)
        dpsi_dq = dpsi_dR[:, None] * direction
        # F_nl_out = sum_j weights[j, out] * psi(R_j)
        # dF_nl_out/dq_in = sum_j weights[j, out] * dpsi_dq[j, in]
        return self.weights.T @ dpsi_dq                 # (r_out, r_in)

    def reconstruct_q_nl(self, q_l, Lambda):
        """Eq. 11 (rearranged): q_nl = q_l - F_nl(q_l)/Lambda, using
        the trained surrogate's F_nl prediction -- no Newton-Raphson
        needed, the defining efficiency property of this method."""
        q_l = np.asarray(q_l, dtype=float)
        Lambda = np.asarray(Lambda, dtype=float)
        F_nl = self.predict(q_l)
        return q_l - F_nl / Lambda


# =====================================================================
# 3b. NeuralSurrogate -- torch.nn (MLP) fit of F_nl(q_l) (Wave 7 item
#     35, docs/consolidated_future_roadmap.md): a native PyTorch
#     alternative to MultiFidelitySurrogate's RBF interpolant, sharing
#     the exact same q_l -> F_nl protocol and predict()/jacobian()
#     shape conventions, so `nonlinear_dynamics.py`/`nnm.py` (and any
#     `TrainingStrategy`) accept either interchangeably -- neither
#     knows or cares which one produced the `ReducedForceModel` it was
#     handed. See this class's own docstring for the design rationale
#     and why this is NOT a new `TrainingStrategy` (a common
#     misreading of the roadmap item's own name).
# =====================================================================
_HAS_TORCH = False
try:
    import torch
    import torch.nn as _nn
    _HAS_TORCH = True
except Exception:
    # Bare `except Exception`, not `except ImportError`: a CUDA-linked
    # PyPI wheel with no matching CUDA runtime can raise OSError/
    # ValueError at import time, not ImportError -- the exact same
    # defensive pattern fea_engine/autograd_tangent.py's own
    # `_HAS_TORCH` uses, independently re-implemented here since
    # rom_engine does not depend on fea_engine (rom_engine/__init__.py's
    # own stated design: fea_engine is a test/example dependency only).
    pass


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "NeuralSurrogate requires PyTorch, which is not importable in "
            "this environment. Install it (`pip install torch`) or use "
            "MultiFidelitySurrogate/PolynomialModalROM instead -- both are "
            "pure NumPy and need no optional dependency.")


_ACTIVATION_REGISTRY = {}
if _HAS_TORCH:
    _ACTIVATION_REGISTRY = {"tanh": _nn.Tanh, "relu": _nn.ReLU, "silu": _nn.SiLU}


def _fit_normalization(X, eps=1e-12):
    """Per-column mean/std, with std floored at `eps` so a constant
    (or single-sample) column never produces a divide-by-zero --
    plain NumPy, deliberately factored out of NeuralSurrogate so this
    piece of the logic is unit-testable WITHOUT torch (see
    tests/test_nonlinear_rom.py::TestNormalizationHelpers, which runs
    unconditionally, unlike every torch-gated NeuralSurrogate test)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def _normalize(X, mean, std):
    return (X - mean) / std


def _denormalize(Xn, mean, std):
    return Xn * std + mean


class NeuralSurrogate(ReducedForceModel):
    """MLP surrogate of the nonlinear modal force as a function of the
    LINEAR modal displacement `q_l` -- same protocol and role as
    `MultiFidelitySurrogate` (Section 3 above), swapping the RBF
    interpolant for a small feedforward `torch.nn` network. Neither
    `TrainingStrategy` needs to know or change anything to produce
    data for this class instead of the RBF one: both already return
    plain `(q_l_samples, q_nl_samples, F_nl_samples)` arrays, and this
    class's `fit(q_l_samples, F_nl_samples)` signature matches
    `MultiFidelitySurrogate.fit()` exactly.

    Naming note: the roadmap item that scoped this (Wave 7 item 35,
    `docs/consolidated_future_roadmap.md`, tracing back to
    `tensormesh_comparative_analysis.md` Section 6.2) called the
    planned class "NeuralSurrogateTrainingStrategy." Reading the
    actual architecture here (this module's own Section 5 docstring:
    "`TrainingStrategy` ... pluggable training-DATA generation, not
    pluggable regression") makes clear that name described the wrong
    layer -- `TrainingStrategy` governs how FULL-ORDER-MODEL solves
    are turned into `(q_l, q_nl, F_nl)` triples, which has nothing to
    do with which regression method later fits a model to them.
    `AppliedLoadStrategy`/`EnforcedDisplacementStrategy` already work
    unchanged with ANY `ReducedForceModel`, including this one -- what
    was actually missing, and what this class provides, is a new
    `ReducedForceModel`, the same architectural slot
    `MultiFidelitySurrogate` occupies. Documented here rather than
    silently building something misnamed.

    Why a neural net alongside the already-validated RBF surrogate:
    an MLP scales more gracefully than an RBF interpolant to LARGER
    training sets (RBF's kernel matrix is `(N, N)`, revisited and
    re-solved in full on every `fit()`; a fixed-size MLP's training
    cost per epoch is `O(N)`, not `O(N^2)`/`O(N^3)`) and, unlike RBF's
    similarity-to-training-centers structure, can represent smooth
    global trends across a larger, higher-dimensional `q_l` domain
    with fewer parameters once `r` (retained-mode count) grows -- the
    trade-off is needing an iterative training loop (this class) in
    place of one linear solve (`MultiFidelitySurrogate.fit()`).

    Parameters
    ----------
    n_modes : int
        Number of retained modes (both input and output dimension of
        the network, matching `q_l`/`F_nl`'s own shared `r`).
    hidden_sizes : tuple of int, default (32, 32)
        Hidden-layer widths of the feedforward network.
    activation : str, one of _ACTIVATION_REGISTRY's keys, default "tanh"
        Hidden-layer nonlinearity. "tanh" (smooth, infinitely
        differentiable) is the default specifically because
        `jacobian()` differentiates THROUGH it -- "relu"'s kink at 0
        still autograds correctly (a valid subgradient) but gives a
        piecewise-constant-slope Jacobian locally, a real but
        deliberately accepted trade-off if chosen.
    lr : float, default 1e-3
        Adam learning rate.
    n_epochs : int, default 2000
        Full-batch training epochs (`fit()`'s own training sets are
        the same O(10-100)-sample scale every `TrainingStrategy` in
        this module already produces -- small enough that full-batch,
        not mini-batch, gradient descent is the right, simplest choice).
    weight_decay : float, default 0.0
        Adam L2 penalty (optional light regularization for a small
        training set).
    seed : int, optional
        `torch.manual_seed()` value for reproducible weight
        initialization -- unseeded (PyTorch's own default RNG state)
        if None.
    device : str, default "cpu"
        Passed straight to `.to(device)` -- "cuda" opens a GPU path on
        a machine with a working CUDA-enabled torch install, mirroring
        `fea_engine.torch_sparse_solver`'s own `device=` convention.
    """

    def __init__(self, n_modes, hidden_sizes=(32, 32), activation="tanh",
                 lr=1e-3, n_epochs=2000, weight_decay=0.0, seed=None, device="cpu"):
        _require_torch()
        if activation not in _ACTIVATION_REGISTRY:
            raise ValueError(
                f"activation={activation!r} not in {list(_ACTIVATION_REGISTRY)}")
        self.n_modes = n_modes
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
        layers = []
        sizes = (n_modes,) + self.hidden_sizes + (n_modes,)
        act_cls = _ACTIVATION_REGISTRY[activation]
        for i in range(len(sizes) - 1):
            layers.append(_nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act_cls())
        self.net = _nn.Sequential(*layers).double().to(device)

        self.q_mean = self.q_std = self.F_mean = self.F_std = None
        self.loss_history = None

    def fit(self, q_l_samples, F_nl_samples, n_epochs=None, verbose=False):
        """q_l_samples, F_nl_samples : (N, n_modes) -- identical
        contract to `MultiFidelitySurrogate.fit()`. Trains via Adam +
        MSE loss on Z-SCORE-NORMALIZED inputs/outputs (`_fit_
        normalization()`/`_normalize()` above): unlike an RBF kernel
        (whose `sigma` is itself fit to the data's own length scale),
        a plain MLP's gradient-descent training is sensitive to input/
        output scale, so normalizing first is a real correctness/
        convergence-quality choice here, not cosmetic. `n_epochs`
        overrides the constructor's default for this call only.
        Returns self (chainable, matching every other `fit()` in this
        module)."""
        q_l_samples = np.asarray(q_l_samples, dtype=float)
        F_nl_samples = np.asarray(F_nl_samples, dtype=float)
        if q_l_samples.ndim != 2 or F_nl_samples.ndim != 2:
            raise ValueError("q_l_samples and F_nl_samples must both be 2-D (N, r)")
        if q_l_samples.shape[0] != F_nl_samples.shape[0]:
            raise ValueError(
                f"q_l_samples and F_nl_samples must have the same number of rows, "
                f"got {q_l_samples.shape[0]} and {F_nl_samples.shape[0]}")
        if q_l_samples.shape[1] != self.n_modes:
            raise ValueError(
                f"q_l_samples has {q_l_samples.shape[1]} columns, expected "
                f"n_modes={self.n_modes}")

        self.q_mean, self.q_std = _fit_normalization(q_l_samples)
        self.F_mean, self.F_std = _fit_normalization(F_nl_samples)
        q_n = _normalize(q_l_samples, self.q_mean, self.q_std)
        F_n = _normalize(F_nl_samples, self.F_mean, self.F_std)

        q_t = torch.tensor(q_n, dtype=torch.float64, device=self.device)
        F_t = torch.tensor(F_n, dtype=torch.float64, device=self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        loss_fn = _nn.MSELoss()

        epochs = self.n_epochs if n_epochs is None else n_epochs
        history = []
        self.net.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.net(q_t)
            loss = loss_fn(pred, F_t)
            loss.backward()
            optimizer.step()
            history.append(float(loss.item()))
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"  NeuralSurrogate.fit: epoch {epoch}/{epochs} loss={loss.item():.6e}")
        self.loss_history = np.array(history)
        self.net.eval()
        return self

    def predict(self, q_l):
        """q_l : (r,) or (n, r). Returns matching-shape F_nl -- same
        shape convention as `MultiFidelitySurrogate.predict()`."""
        if self.q_mean is None:
            raise RuntimeError("NeuralSurrogate.predict() called before fit()")
        q_l = np.asarray(q_l, dtype=float)
        single = q_l.ndim == 1
        Q = np.atleast_2d(q_l)
        q_n = _normalize(Q, self.q_mean, self.q_std)
        with torch.no_grad():
            q_t = torch.tensor(q_n, dtype=torch.float64, device=self.device)
            F_n = self.net(q_t).cpu().numpy()
        F_nl = _denormalize(F_n, self.F_mean, self.F_std)
        return F_nl[0] if single else F_nl

    def jacobian(self, q_l):
        """dF_nl/dq_l at a single point q_l, shape (r_out, r_in) --
        same contract as `MultiFidelitySurrogate.jacobian()`, computed
        via `torch.autograd.grad` (one exact reverse-mode pass per
        output row) instead of an analytic kernel-derivative formula --
        the same row-by-row `torch.autograd.grad` technique this
        project's `fea_engine.autograd_tangent` module already uses
        for element tangent stiffnesses, applied here to a ROM force
        surrogate instead of an FE internal-force function. The
        normalization layers are included in the differentiated graph
        via explicit chain-rule scaling (`F_std[:, None] * dF_n/dq_n /
        q_std[None, :]`), not re-derived by hand outside it."""
        if self.q_mean is None:
            raise RuntimeError("NeuralSurrogate.jacobian() called before fit()")
        q_l = np.asarray(q_l, dtype=float)
        if q_l.ndim != 1:
            raise ValueError("jacobian() takes a single point (r,), not a batch")
        q_n = _normalize(q_l, self.q_mean, self.q_std)
        q_t = torch.tensor(q_n, dtype=torch.float64, device=self.device, requires_grad=True)
        F_n_t = self.net(q_t)
        n_out = F_n_t.shape[0]
        rows = []
        for i in range(n_out):
            grad_i, = torch.autograd.grad(F_n_t[i], q_t, retain_graph=(i < n_out - 1))
            rows.append(grad_i.detach().cpu().numpy())
        J_norm = np.stack(rows, axis=0)                 # dF_n/dq_n, (r_out, r_in)
        return (self.F_std[:, None] * J_norm) / self.q_std[None, :]


# =====================================================================
# 4. PolynomialModalROM -- Nash-form quadratic+cubic fit of F_nl(q_nl)
#    (Section 3.3: generalizes ICE / Shi & Mei / Enforced-Displacement)
# =====================================================================
def _monomial_indices(n_modes, order):
    return list(combinations_with_replacement(range(n_modes), order))


def _design_matrix(Q, quad_idx, cub_idx):
    cols = [Q[:, i] * Q[:, j] for (i, j) in quad_idx]
    cols += [Q[:, i] * Q[:, j] * Q[:, k] for (i, j, k) in cub_idx]
    return np.column_stack(cols) if cols else np.zeros((Q.shape[0], 0))


class PolynomialModalROM(ReducedForceModel):
    """theta_r(q) = sum_{i<=j} B_r(i,j) q_i q_j + sum_{i<=j<=k} A_r(i,j,k) q_i q_j q_k
    (Nash-form, Eq. 45), fit by ordinary least squares per retained
    mode -- ported from the validated `mfs-nlrom-beam` skill's
    `IceRom`. Fits F_nl as a function of the NONLINEAR modal
    displacement q_nl itself (unlike `MultiFidelitySurrogate`, which
    fits against q_l) -- this is exactly why `predict(F_ext=...)`
    needs its own Newton-Raphson solve.

    `ICEROM`/`ShiMeiROM`/`EnforcedDisplacementROM` (module-level
    convenience constructors below) are thin, honestly-labeled wrappers
    around this one class + a specific `TrainingStrategy` -- not three
    copies of the same regression code.
    """

    def __init__(self, n_modes):
        self.n_modes = n_modes
        self.quad_idx = _monomial_indices(n_modes, 2)
        self.cub_idx = _monomial_indices(n_modes, 3)
        self.coeffs = None   # (n_terms, n_modes)

    def fit(self, q_nl_samples, F_nl_samples):
        """q_nl_samples : (N, n_modes) NONLINEAR modal displacement
        training points (the actual projected full-order solution, not
        the linear estimate). F_nl_samples : (N, n_modes)."""
        q_nl_samples = np.asarray(q_nl_samples, dtype=float)
        F_nl_samples = np.asarray(F_nl_samples, dtype=float)
        if q_nl_samples.shape[1] != self.n_modes:
            raise ValueError(
                f"q_nl_samples has {q_nl_samples.shape[1]} columns, "
                f"expected n_modes={self.n_modes}")
        X = _design_matrix(q_nl_samples, self.quad_idx, self.cub_idx)
        self.coeffs = np.linalg.lstsq(X, F_nl_samples, rcond=None)[0]
        return self

    def force(self, q):
        """theta(q): direct polynomial evaluation, no Newton-Raphson.
        q may be (n_modes,) or (n, n_modes)."""
        if self.coeffs is None:
            raise RuntimeError("PolynomialModalROM.force() called before fit()")
        single = np.asarray(q).ndim == 1
        Q = np.atleast_2d(q)
        X = _design_matrix(Q, self.quad_idx, self.cub_idx)
        out = X @ self.coeffs
        return out[0] if single else out

    def jacobian(self, q):
        """d(theta_out)/d(q_in) at a single point q, shape (n_modes, n_modes)."""
        if self.coeffs is None:
            raise RuntimeError("PolynomialModalROM.jacobian() called before fit()")
        q = np.asarray(q, dtype=float)
        n = self.n_modes
        KT = np.zeros((n, n))
        nq = len(self.quad_idx)
        for c, (i, j) in enumerate(self.quad_idx):
            coef = self.coeffs[c]           # (n_modes,) contribution to each output
            if i == j:
                KT[:, i] += coef * 2.0 * q[i]
            else:
                KT[:, i] += coef * q[j]
                KT[:, j] += coef * q[i]
        for c, (i, j, k) in enumerate(self.cub_idx):
            coef = self.coeffs[nq + c]
            cnt = Counter([i, j, k])
            for idx, mult in cnt.items():
                remaining = [i, j, k]
                for _ in range(mult):
                    remaining.remove(idx)
                # d/dq_idx of q_i*q_j*q_k = mult * q_idx^(mult-1) * prod(remaining factors)
                prod_rest = 1.0
                for m in remaining:
                    prod_rest *= q[m]
                KT[:, idx] += coef * mult * (q[idx] ** (mult - 1)) * prod_rest
        return KT

    def predict(self, q=None, F_ext=None, Lambda=None, q0=None, tol=1e-10, max_iter=50):
        """Either evaluate theta(q) directly (q given, no Newton-
        Raphson -- just the polynomial force at that displacement), or
        solve Lambda@q + theta(q) = F_ext for q via Newton-Raphson
        (F_ext AND Lambda given) -- the second form is what static/
        dynamic analysis actually needs, and is the accuracy-for-speed
        trade-off this whole model family makes relative to
        `MultiFidelitySurrogate`.

        Returns
        -------
        q given: F_nl prediction (same shape convention as `force()`).
        F_ext given: (q_solution, converged) -- q_solution is the last
        Newton iterate even if converged is False, so a caller can
        inspect how far off it ended up rather than only get an
        exception.
        """
        if q is not None and F_ext is not None:
            raise ValueError("predict(): pass EITHER q OR F_ext, not both")
        if q is not None:
            return self.force(q)
        if F_ext is None:
            raise ValueError("predict(): must pass either q or F_ext")
        if Lambda is None:
            raise ValueError("predict(F_ext=...) requires Lambda (the linear modal stiffness)")
        Lambda = np.asarray(Lambda, dtype=float)
        F_ext = np.asarray(F_ext, dtype=float)
        n = len(Lambda)
        q_iter = np.zeros(n) if q0 is None else np.asarray(q0, dtype=float).copy()
        converged = False
        for _ in range(max_iter):
            theta = self.force(q_iter)
            R = F_ext - (Lambda * q_iter + theta)
            if np.max(np.abs(R)) < tol * max(1.0, np.max(np.abs(F_ext))):
                converged = True
                break
            K_T = np.diag(Lambda) + self.jacobian(q_iter)
            try:
                dq = np.linalg.solve(K_T, R)
            except np.linalg.LinAlgError:
                break
            q_iter = q_iter + dq
        return q_iter, converged


# =====================================================================
# 5. TrainingStrategy -- pluggable training-DATA generation (Section 4)
# =====================================================================
class TrainingStrategy:
    """Pluggable training-DATA generation, not pluggable regression.
    Each concrete strategy's `generate()` returns
    `(q_l_samples, q_nl_samples, F_nl_samples)` given a full-order model
    solver callable and a (mass-normalized) basis -- neither
    `MultiFidelitySurrogate.fit()` (which wants `q_l_samples`) nor
    `PolynomialModalROM.fit()` (which wants `q_nl_samples`) knows or
    cares which strategy produced its input; a caller picks whichever
    column this returns matches the model it's about to fit."""

    def generate(self, *args, **kwargs):
        raise NotImplementedError


class AppliedLoadStrategy(TrainingStrategy):
    """Eq. 46-47 / the ICE / Shi & Mei training-load convention: build
    a per-mode force-scale matrix via `sampling.modal_force_samples()`
    (OLHS-sampled per-mode target displacement fractions), assemble
    each training sample's full free-DOF force vector as
    `F = M_ff @ (V @ f_hat_row)` (so that `V.T @ F == f_hat_row`
    exactly, for mass-normalized modes -- `V.T @ M_ff @ V == I`), solve
    the FULL nonlinear FOM at each `F`, and project the resulting
    displacement back onto the basis.
    """

    def __init__(self, target_fracs, reference_scale, n_samples, rng):
        self.target_fracs = target_fracs
        self.reference_scale = reference_scale
        self.n_samples = n_samples
        self.rng = rng

    def generate(self, V, M_ff, basis_freqs_hz, mode_shape_peaks, fom_solver,
                 return_snapshots=False):
        """
        Parameters
        ----------
        V : ndarray, shape (n_free, n_modes)
            Mass-normalized retained mode shapes (V.T @ M_ff @ V == I).
        M_ff : ndarray, shape (n_free, n_free)
            Free-DOF mass matrix.
        basis_freqs_hz, mode_shape_peaks : see `sampling.modal_force_samples`.
        fom_solver : callable(F_free) -> u_free
            Solves the FULL nonlinear order model's static equilibrium
            at the given free-DOF force vector, returning the free-DOF
            displacement. The only place this function (or the caller
            constructing it) may know about a specific FE package --
            this module never imports one.
        return_snapshots : bool, default False
            Wave 12 item 112 (docs/consolidated_future_roadmap.md,
            `ICE-ROM/GAP_ANALYSIS.md` gap #1): when True, ALSO returns
            the full free-DOF displacement snapshots `W` this method
            already computes internally (as `u_free`) but previously
            discarded after projecting onto the retained modal basis.
            `membrane_expansion.MembraneBasis.fit()` needs these full
            snapshots (Eq. 13-15 of Hollkamp & Gordon 2008) -- they are
            not extra full-order solves, just data this method was
            already producing and throwing away. Default False keeps
            every existing caller's return-tuple shape unchanged.

        Returns
        -------
        q_l_samples, q_nl_samples, F_nl_samples : each (n_samples, n_modes)
        W_samples : ndarray, shape (n_free, n_samples), ONLY when
            `return_snapshots=True` -- appended as a fourth return
            value, never inserted in the middle, so existing 3-tuple
            unpacking (`q_l, q_nl, F_nl = strategy.generate(...)`)
            keeps working unchanged when this flag is left at its
            default.
        """
        n_modes = V.shape[1]
        n_free = V.shape[0]
        Lambda = (2 * np.pi * np.asarray(basis_freqs_hz)) ** 2
        f_hat = modal_force_samples(basis_freqs_hz, mode_shape_peaks, self.target_fracs,
                                     self.reference_scale, self.n_samples, rng=self.rng)

        q_l = f_hat / Lambda[None, :]                  # Eq. 10: Lambda q_l = modal_force = f_hat
        q_nl = np.zeros((self.n_samples, n_modes))
        F_nl = np.zeros((self.n_samples, n_modes))
        W = np.zeros((n_free, self.n_samples)) if return_snapshots else None
        for i in range(self.n_samples):
            F_free = M_ff @ (V @ f_hat[i])
            u_free = np.asarray(fom_solver(F_free))
            q_nl[i] = V.T @ (M_ff @ u_free)             # mass-orthogonal projection
            F_nl[i] = f_hat[i] - Lambda * q_nl[i]        # Eq. 9 rearranged
            if return_snapshots:
                W[:, i] = u_free
        if return_snapshots:
            return q_l, q_nl, F_nl, W
        return q_l, q_nl, F_nl


class EnforcedDisplacementStrategy(TrainingStrategy):
    """STEP's own convention: prescribe q directly (a displacement
    PATTERN in the shape of one or a combination of retained modes),
    solve the full-order model in DISPLACEMENT CONTROL, read back the
    reaction force. No load-scaling formula needed at all -- this is
    the strategy that removes Eq. 46-47 from the picture entirely,
    exactly what distinguishes "enforced displacement" from "applied
    loads" in the literature (module docstring, Section 1)."""

    def __init__(self, q_range, n_samples, rng):
        """q_range : (n_modes, 2) array (or a (2,) pair broadcast to
        every mode) of [q_min, q_max] per retained modal coordinate to
        OLHS-sample over. n_samples, rng: see `sampling.optimal_lhs`."""
        self.q_range = q_range
        self.n_samples = n_samples
        self.rng = rng

    def generate(self, V, M_ff, basis_freqs_hz, fom_solver):
        """
        Parameters
        ----------
        V : ndarray, shape (n_free, n_modes)
        M_ff : ndarray, shape (n_free, n_free)
        basis_freqs_hz : array_like, shape (n_modes,)
        fom_solver : callable(u_free_target) -> F_reaction_free
            Solves the full-order model in DISPLACEMENT CONTROL at the
            prescribed free-DOF displacement pattern, returning the
            free-DOF reaction force needed to hold it there. As with
            `AppliedLoadStrategy`, the only fea_engine-aware code lives
            in whatever the caller passes here, never in this module.

        Returns
        -------
        q_l_samples, q_nl_samples, F_nl_samples : each (n_samples, n_modes)
        """
        n_modes = V.shape[1]
        Lambda = (2 * np.pi * np.asarray(basis_freqs_hz)) ** 2

        q_range = np.asarray(self.q_range, dtype=float)
        if q_range.shape == (2,):
            q_range = np.tile(q_range, (n_modes, 1))
        elif q_range.shape != (n_modes, 2):
            raise ValueError(f"q_range must be a (2,) pair or (n_modes, 2), got {q_range.shape}")
        q_min, q_max = q_range[:, 0], q_range[:, 1]

        unit_design = optimal_lhs(self.n_samples, n_modes, criterion="maximin", rng=self.rng)
        q_nl_prescribed = q_min + unit_design * (q_max - q_min)   # (n_samples, n_modes)

        q_nl = np.zeros((self.n_samples, n_modes))
        q_l = np.zeros((self.n_samples, n_modes))
        F_nl = np.zeros((self.n_samples, n_modes))
        for i in range(self.n_samples):
            q_target = q_nl_prescribed[i]
            u_target = V @ q_target
            F_reaction = np.asarray(fom_solver(u_target))
            modal_force = V.T @ F_reaction
            q_nl[i] = q_target
            q_l[i] = modal_force / Lambda
            F_nl[i] = modal_force - Lambda * q_target
        return q_l, q_nl, F_nl


def _farthest_point_subsample(candidates, n_target, rng):
    """Greedy farthest-point ("maximin") subset selection: pick a
    random seed point, then repeatedly add whichever remaining
    candidate maximizes its OWN distance to the nearest point already
    selected -- the standard, simplest space-filling way to thin a
    large point cloud down to `n_target` well-SPREAD representatives
    (as opposed to `sampling.optimal_lhs`, which DESIGNS new points in
    a box; this instead SELECTS a subset of points that already exist,
    which is what `TrajectoryPilotedStrategy` needs -- the trajectory
    a dynamic simulation actually visited isn't a box to sample from,
    it's a fixed, irregularly-shaped point cloud to pick well-spread
    representatives out of).

    Returns the selected row INDICES into `candidates` (not the points
    themselves), so a caller can use them to also select any parallel
    array (e.g. a corresponding time-stamp array) consistently.
    """
    candidates = np.asarray(candidates, dtype=float)
    n_candidates = candidates.shape[0]
    if n_target >= n_candidates:
        return np.arange(n_candidates)
    seed = int(rng.integers(n_candidates))
    selected = [seed]
    min_dist = np.linalg.norm(candidates - candidates[seed], axis=1)
    for _ in range(n_target - 1):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        d = np.linalg.norm(candidates - candidates[next_idx], axis=1)
        min_dist = np.minimum(min_dist, d)
    return np.array(selected)


class TrajectoryPilotedStrategy(TrainingStrategy):
    """Wave 12 item 113 (docs/consolidated_future_roadmap.md,
    `ICE-ROM/GAP_ANALYSIS.md` gap #2): training-data generation piloted
    by states a dynamic simulation actually VISITS, rather than by
    independently-sampled per-mode static targets.

    The problem this fixes (found empirically, not by code inspection
    -- see the gap analysis's own account): `AppliedLoadStrategy`
    samples each mode's own target displacement fraction independently
    (an OLHS design over per-mode force scales). For a single retained
    mode this is the whole space there is to cover. For SEVERAL
    retained modes, it covers only the axis-aligned "independent"
    corners of the true multi-mode state space -- it does not, and
    structurally cannot, cover the coupled, sign-varying, time-
    correlated COMBINATIONS a real coupled dynamic response actually
    visits. Two architecturally unrelated regression methods
    (`PolynomialModalROM`, `NeuralSurrogate`) were both found to reach
    excellent static held-out R^2 on `AppliedLoadStrategy` data and
    still diverge catastrophically the moment they were driven
    dynamically -- decisive evidence the defect is upstream of the
    regression method, in WHERE the training data was sampled from,
    not in how it is fit.

    This strategy instead takes the multi-mode states a caller-supplied
    exploratory dynamic trajectory (or several) actually passed
    through, thins them down to a manageable, well-spread training set
    via `_farthest_point_subsample()`, and reuses
    `EnforcedDisplacementStrategy`'s own per-sample mechanism exactly
    (prescribe each selected `q_nl` combination as a displacement
    target, solve in displacement control, read back the reaction
    force) -- not a new full-order-model interface, just a different,
    trajectory-informed choice of WHICH `q_nl` combinations to solve
    at.

    Deliberately does NOT run the exploratory simulation itself (same
    "the only fea_engine-aware code lives in whatever the caller
    passes in" principle `AppliedLoadStrategy`/`EnforcedDisplacementStrategy`
    already establish): the caller supplies the visited trajectory
    or trajectories directly, whether produced by a full nonlinear
    transient FOM solve, a crude/under-trained ROM's own reduced
    integrator, or anything else -- this class stays integration-
    method-agnostic.
    """

    def __init__(self, n_target, rng):
        """n_target : int, the number of training points to keep after
        farthest-point subsampling (fewer than the number of visited
        trajectory samples supplied to `generate()`, typically). rng :
        numpy.random.Generator, this package's standing explicit-rng
        convention (see `sampling.py`'s own module docstring)."""
        self.n_target = n_target
        self.rng = rng

    def generate(self, V, basis_freqs_hz, trajectories, fom_solver):
        """
        Parameters
        ----------
        V : ndarray, shape (n_free, n_modes)
            Mass-normalized retained mode shapes.
        basis_freqs_hz : array_like, shape (n_modes,)
        trajectories : array_like (n_visited, n_modes), or a list/tuple
            of such arrays (one per exploratory run, concatenated
            internally) -- the `q_nl(t)` states ACTUALLY VISITED by one
            or more cheap exploratory dynamic simulations. Typically
            `nonlinear_dynamics.integrate_newmark_surrogate()`'s own
            `q_nl_hist` return value from a crude/under-trained ROM, or
            a full-order transient solve's displacement history
            projected onto `V` (`V.T @ (M_ff @ u_free_hist.T)`, mass-
            orthogonal, matching `AppliedLoadStrategy`'s own
            projection convention).
        fom_solver : callable(u_free_target) -> F_reaction_free
            SAME contract as `EnforcedDisplacementStrategy`'s own
            `fom_solver` -- solves the full-order model in displacement
            control at the prescribed free-DOF pattern, returns the
            free-DOF reaction force. `M_ff` is not needed here (same
            reason `EnforcedDisplacementStrategy` doesn't use it
            either: `q_nl` is prescribed directly, not obtained via a
            mass-orthogonal projection of a solved displacement).

        Returns
        -------
        q_l_samples, q_nl_samples, F_nl_samples : each
            (min(n_target, n_visited), n_modes).
        """
        if isinstance(trajectories, (list, tuple)):
            trajectories = np.concatenate(
                [np.asarray(tr, dtype=float) for tr in trajectories], axis=0)
        else:
            trajectories = np.asarray(trajectories, dtype=float)
        n_modes = V.shape[1]
        if trajectories.ndim != 2 or trajectories.shape[1] != n_modes:
            raise ValueError(
                f"trajectories must be (n_visited, n_modes={n_modes}), "
                f"got shape {trajectories.shape}")
        Lambda = (2 * np.pi * np.asarray(basis_freqs_hz)) ** 2

        idx = _farthest_point_subsample(trajectories, self.n_target, self.rng)
        q_nl_targets = trajectories[idx]
        n_selected = len(idx)

        q_nl = np.zeros((n_selected, n_modes))
        q_l = np.zeros((n_selected, n_modes))
        F_nl = np.zeros((n_selected, n_modes))
        for i in range(n_selected):
            q_target = q_nl_targets[i]
            u_target = V @ q_target
            F_reaction = np.asarray(fom_solver(u_target))
            modal_force = V.T @ F_reaction
            q_nl[i] = q_target
            q_l[i] = modal_force / Lambda
            F_nl[i] = modal_force - Lambda * q_target
        return q_l, q_nl, F_nl


# =====================================================================
# 6. Convenience constructors -- the paper's three named baselines,
#    as thin PolynomialModalROM(...) + TrainingStrategy combinations,
#    not three copies of the same ~40 lines of least-squares code.
# =====================================================================
def ICEROM(n_modes):
    """Implicit Condensation and Expansion (Hollkamp & Gordon 2008):
    `PolynomialModalROM` fit with `AppliedLoadStrategy`-generated
    training data (the applied-load convention). Identical class to
    `ShiMeiROM` -- the two differ only in training-data STRATEGY, not
    regression method (see module docstring, Section 3.3)."""
    return PolynomialModalROM(n_modes)


def ShiMeiROM(n_modes):
    """Same regression as `ICEROM` -- kept as a separate, honestly-
    labeled name because the literature (Section 1's citations)
    distinguishes ICE and Shi & Mei by training-data convention details
    (e.g. membrane augmentation) this module does not yet implement as
    a separate option; both currently map to the same
    `PolynomialModalROM` class."""
    return PolynomialModalROM(n_modes)


def EnforcedDisplacementROM(n_modes):
    """STEP's own convention (Muravyov & Rizzi 2003): `PolynomialModalROM`
    fit with `EnforcedDisplacementStrategy`-generated training data
    (no load-scaling formula needed)."""
    return PolynomialModalROM(n_modes)
