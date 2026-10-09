"""
differentiable.py -- Wave 10 (docs/consolidated_future_roadmap.md),
sourced from Saverio, Bucci, Farro, Content & Sipp, "An end-to-end
PyTorch interface for differentiable PDE solvers: A RANS model-
correction study" (Data-Centric Engineering 7:e37, 2026,
doi:10.1017/dce.2026.10066). Items 98-101 of that wave:

  - Item 99: a generic trainable additive-correction API (`Additive
    Correction` + concrete `ScalarFieldCorrection`/`MLPCorrection`)
    bolted onto FESystem's own nonlinear static residual, promoting
    Wave 9's ad hoc per-element `method="autograd"` opt-in into a
    first-class R(u)+f_theta(u)=0 hook. `solve_nonlinear_static_
    corrected()` is the concrete Newton driver every other item in
    this module trains THROUGH.
  - Item 100: `calibrate_correction_explicit()` -- the cheap, lower-
    risk calibration path for when a full reference state is already
    available: minimize ||R(w_m)+f_theta(w_m)||^2 directly, no Newton
    solve needed. Built and validated FIRST, per this wave's own
    stated risk-ordering.
  - Item 98: `implicit_correction_solve()` / `adjoint_gradient()` --
    the whole-solver implicit-differentiation (DEQ-style adjoint) layer
    that differentiates an ENTIRE converged Newton solve w.r.t.
    correction.parameters() by solving one adjoint linear system,
    without unrolling the Newton iteration. The single biggest item in
    this wave.
  - Item 101: `mass_cholesky_factor()` / `MetricScaledField` -- the
    Cholesky change-of-variables (Appendix C) that makes a plain
    Euclidean optimizer step on a REPARAMETRIZED spatial-field
    parameter equal a mass-matrix-metric-consistent step on the field
    itself.

Item 102 (batched vmap element-tangent assembly) lives in
autograd_tangent.py instead, next to the per-element autograd tangents
it batches (Wave 9 items 93/96) -- see that module's own new section.
Items 103-105 (ROM correction design/prototype, dataset diagnostics,
ensemble UQ) live in the sibling rom_engine package (differentiable_
correction.py, dataset_diagnostics.py, ensemble_uq.py) since none of
them are FEA-specific -- see each module's own docstring.

CONVENTION CARRIED OVER UNCHANGED from autograd_tangent.py/torch_
sparse_solver.py (Wave 0/9): torch stays an OPTIONAL dependency
(`_HAS_TORCH`/`_require_torch()`, bare `except Exception` for the same
CUDA-linked-wheel reason documented in those modules' own docstrings).

DELIBERATE DESIGN CHOICE, made so this module's most important new
capability could get REAL test coverage in a sandbox where torch is
not importable at all: the additive-correction Newton driver (item 99)
and the adjoint math (item 98's `adjoint_gradient()`) are written
against a plain `value()`/`jacobian()` protocol ANY object can
satisfy, including a pure-NumPy one with a hand-coded closed-form
Jacobian -- they do not import or require torch themselves. This means
tests/test_differentiable.py can validate the actual mathematical
content (does a Newton solve with a correction term converge to a
self-consistent root; does the adjoint formula for differentiating
through that root match a brute-force finite-difference reference)
WITHOUT needing a working torch install. Only the concrete NN/tensor-
parameterized correction classes, the explicit-calibration training
LOOP (needs a torch optimizer), and the torch.autograd.Function
wrapper around the adjoint layer actually require torch -- exactly the
same "numpy-shadow-first" split Wave 7's NeuralSurrogate established
(`_fit_normalization`/`_normalize`/`_denormalize` there; `adjoint_
gradient()`/`solve_nonlinear_static_corrected()` here).

SIGN CONVENTION, stated once here since every function below depends
on it: this package's own nonlinear residual is R(u) = F_ext - F_int
(u) = 0 (see nonlinear_solver.solve_nonlinear_static()'s own
_residual() closure). The paper's own convention is R(w)+f_theta(w)=0
with R meaning THEIR forward operator (our F_int-F_ext). Translating
their form into ours: (F_int(u)-F_ext) + f_theta(u) = 0, i.e.

    F_ext - F_int(u) - f_theta(u) = 0

-- f_theta enters our residual with a MINUS sign, i.e. behaves exactly
like an ADDITIONAL, trainable internal-force term. Every function
below (corrected_residual, corrected_tangent, adjoint_gradient's own
derivation) is consistent with this one convention.
"""
__author__ = "Abhijeet"
import numpy as np

try:
    import torch
    import torch.nn as _nn
    _HAS_TORCH = True
except Exception:
    # Deliberately a bare `except Exception`, not `except ImportError`
    # -- see autograd_tangent.py's own comment on this exact except
    # clause (a CUDA-linked wheel can install but fail to import with
    # OSError/ValueError on a machine with no CUDA runtime).
    _HAS_TORCH = False


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "differentiable.py's torch-dependent functionality requires "
            "the 'torch' package: pip install torch -- this, like gmsh "
            "and autograd_tangent.py/torch_sparse_solver.py, is an "
            "OPTIONAL dependency of fea_engine. See this module's own "
            "docstring for which pieces (the Newton driver, the adjoint "
            "math) do NOT need torch at all.")


# =====================================================================
# Item 99: generic trainable additive-correction interface
# =====================================================================
class AdditiveCorrection:
    """Duck-typed protocol (not an ABC by inheritance requirement --
    matching this package's own established style; see the sibling
    rom_engine package's `nonlinear_rom.ReducedForceModel` for the same
    convention) for a trainable additive correction f_theta(u) inserted
    into FESystem's own nonlinear static residual:

        R(u) = F_ext - F_int(u) - f_theta(u) = 0

    -- see this module's own docstring, "SIGN CONVENTION," for why the
    minus sign is the right translation of the paper's own R(w)+
    f_theta(w)=0 into this package's F_ext-F_int=0 convention.

    A concrete correction MUST implement:
      value(u_free) -> (len(u_free),) numpy array: f_theta evaluated at
          the CURRENT free-dof state. Must always return a plain numpy
          array -- a torch-backed correction detaches internally --
          since this is called from inside the plain-NumPy Newton loop
          below (solve_nonlinear_static_corrected/_newton_equilibrium).
      jacobian(u_free) -> (len(u_free), len(u_free)) numpy array:
          d f_theta / d u_free at the current state, added into the
          Newton tangent so the linearization stays consistent with
          the residual above (see corrected_tangent()'s own docstring
          for the exact sign). A correction that returns zeros here
          when it actually has real u-dependence only degrades
          Newton's convergence RATE (no longer exactly quadratic), not
          what it converges to -- still a legitimate, if suboptimal,
          implementation.
      scope: "local" (each output component only depends on its own
          input component / the owning element's own state -- cheap,
          often-diagonal Jacobian) or "nonlocal" (may couple across
          dofs) -- self-descriptive metadata, matching the paper's own
          local f(w,alpha_theta o phi(w)) vs. nonlocal f(w,alpha_theta
          (w)) distinction (Eq. 2.1-2.14); not enforced by this base
          class.

    A TRAINABLE (torch-backed) correction additionally implements:
      torch_value(u_free_t) -> torch tensor, DIFFERENTIABLE w.r.t. both
          u_free_t and this object's own trainable parameters -- what
          calibrate_correction_explicit() (item 100) and the implicit
          adjoint layer (item 98) both need instead of value().
      parameters() -> list of torch leaf tensors (requires_grad=True).
    A pure-NumPy, non-trainable correction (see this module's own test
    file's `_NumpyToyCorrection`) never defines these two -- usable
    with solve_nonlinear_static_corrected()/adjoint_gradient() (whose
    vjp_fn can be any hand-coded closed form) but not with calibrate_
    correction_explicit()/implicit_correction_solve(), which is
    correct: those are training-time concerns and a hand-fixed
    (non-trainable) correction has nothing to train.
    """
    scope = "nonlocal"

    def value(self, u_free):
        raise NotImplementedError

    def jacobian(self, u_free):
        raise NotImplementedError


def corrected_residual(fesystem, u, mat, correction, F_ext=None, **kwargs):
    """R_free(u) = F_ext_free - F_int_free(u) - f_theta(u_free) -- the
    free-standing equivalent of nonlinear_solver.solve_nonlinear_
    static()'s own `_residual()` closure, usable outside a Newton loop
    (calibrate_correction_explicit() below calls the F_int half of
    this directly). Returns (R_free, F_int_full). F_ext defaults to
    fesystem.F; correction=None recovers EXACTLY the plain, uncorrected
    residual (f_theta==0 everywhere)."""
    free = fesystem.free_dofs
    if F_ext is None:
        F_ext = fesystem.F
    F_int = fesystem.assemble_internal_force(u, mat, **kwargs)
    u_free = u[free]
    f_corr = correction.value(u_free) if correction is not None else 0.0
    R_free = F_ext[free] - F_int[free] - f_corr
    return R_free, F_int


def corrected_tangent(fesystem, u, mat, correction, **kwargs):
    """K_eff = K_T(u) + d f_theta/du_free -- the free-dof tangent
    Newton needs to stay consistent with corrected_residual() above:
    d/du of -f_theta(u) contributes +correction.jacobian(u_free) (two
    minus signs: R has a -f_theta term, and Newton solves
    -dR/du @ du = R, so d(-f_theta)/du flips sign twice back to +).
    correction=None recovers exactly K_T(u)'s own free-dof block."""
    free = fesystem.free_dofs
    K_T = fesystem.assemble_tangent_stiffness(u, mat, **kwargs)
    K_free = K_T[np.ix_(free, free)]
    if correction is not None:
        u_free = u[free]
        K_free = K_free + correction.jacobian(u_free)
    return K_free


def _newton_equilibrium(fesystem, mat, correction, F_ext, u0=None,
                         tol=1e-8, max_iter=30, verbose=False, **kwargs):
    """Single-load-vector plain Newton-Raphson solve of the corrected
    residual F_ext - F_int(u) - f_theta(u) = 0, starting from u0
    (defaults to the zero state). Shared core behind both
    solve_nonlinear_static_corrected()'s per-load-step loop AND item
    98's implicit_correction_solve() (which needs a SINGLE converged
    equilibrium state and its final K_eff, not a load-factor sweep) --
    factored out once so both call sites are guaranteed to agree on
    what "the corrected Newton solve" means.

    Deliberately narrower than nonlinear_solver.solve_nonlinear_
    static()'s own per-step ladder: no line search, no du_tol/
    energy_tol extra convergence criteria, no mixed-formulation
    iter_state snapshot/restore. Adding a trainable correction term is
    scoped as an ADDITIVE capability on top of the plain-Newton case
    only here -- exactly the same "narrow, document explicitly"
    convention Wave 5's Notch/Fillet items and Wave 8's multigrid
    scoping already established -- not a claim that every existing
    nonlinear_solver.py robustness feature has been reproduced.

    Returns (u_full, u_free, K_eff, F_int_full, converged, Rn) --
    K_eff is the FINAL free-dof corrected tangent (K_T(u*) +
    correction.jacobian(u_free*) at convergence), returned so a caller
    (the implicit adjoint layer) can reuse it directly instead of
    reassembling."""
    free = fesystem.free_dofs
    n_dof = fesystem.n_dof
    u = np.zeros(n_dof) if u0 is None else np.asarray(u0, dtype=float).copy()
    ref = max(np.linalg.norm(np.asarray(F_ext)[free]), 1e-30)

    def _residual(u_free_trial):
        u[free] = u_free_trial
        R_free, F_int = corrected_residual(fesystem, u, mat, correction, F_ext=F_ext, **kwargs)
        return R_free, F_int

    u_free = u[free].copy()
    R_free, F_int = _residual(u_free)
    Rn = np.linalg.norm(R_free)
    K_eff = None
    converged = False
    for it in range(max_iter):
        if verbose:
            print(f"  [corrected Newton] it {it:2d}  |R|={Rn:.3e}")
        if Rn < tol * ref or Rn < tol:
            converged = True
            K_eff = corrected_tangent(fesystem, u, mat, correction, **kwargs)
            break
        K_eff = corrected_tangent(fesystem, u, mat, correction, **kwargs)
        du = np.linalg.solve(K_eff, R_free)
        u_free = u_free + du
        R_free, F_int = _residual(u_free)
        Rn = np.linalg.norm(R_free)
    u[free] = u_free
    return u, u_free, K_eff, F_int, converged, Rn


def solve_nonlinear_static_corrected(fesystem, mat, correction=None, n_steps=10, tol=1e-8,
                                      max_iter=30, verbose=False, load_factors=None, **kwargs):
    """Item 99's concrete Newton driver: the SAME load-controlled
    incremental-Newton ladder as nonlinear_solver.solve_nonlinear_
    static() (identical load_factors convention, identical commit_all_
    states() call after every converged step), generalized to solve
    F_ext - F_int(u) - f_theta(u) = 0 at every step instead of
    F_ext - F_int(u) = 0.

    correction=None recovers plain, uncorrected Newton -- verified
    directly (tests/test_differentiable.py::test_correction_none_
    matches_plain_newton) to match nonlinear_solver.solve_nonlinear_
    static(..., line_search=False)'s own U_hist bit-for-bit on a case
    where plain Newton converges outright, since this driver has no
    line-search fallback of its own (see _newton_equilibrium()'s own
    docstring for exactly what is and isn't reproduced from that
    richer driver).

    Returns (load_factors, U_hist) -- identical shape/convention to
    nonlinear_solver.solve_nonlinear_static()."""
    free = fesystem.free_dofs
    F_total = fesystem.F.copy()
    n_dof = fesystem.n_dof

    if load_factors is None:
        load_factors = np.linspace(0.0, 1.0, n_steps + 1)
    else:
        load_factors = np.asarray(load_factors, dtype=float)

    u = np.zeros(n_dof)
    U_hist = np.zeros((len(load_factors), n_dof))

    start = 0
    if load_factors[0] == 0.0:
        U_hist[0] = u
        fesystem.commit_all_states(u, mat, **kwargs)
        start = 1

    for step in range(start, len(load_factors)):
        lam = load_factors[step]
        F_ext = lam * F_total
        u_new, u_free, K_eff, F_int, converged, Rn = _newton_equilibrium(
            fesystem, mat, correction, F_ext, u0=u, tol=tol, max_iter=max_iter,
            verbose=verbose, **kwargs)
        if not converged:
            raise RuntimeError(
                f"solve_nonlinear_static_corrected: Newton-Raphson failed to "
                f"converge at load step {step} (lambda={lam:.4f}), |R|={Rn:.3e} "
                f"after {max_iter} iterations.")
        u = u_new
        U_hist[step] = u
        fesystem.commit_all_states(u, mat, **kwargs)

    return load_factors, U_hist


# =====================================================================
# Item 100: explicit residual-minimization calibration
# =====================================================================
def calibrate_correction_explicit(fesystem, mat, correction, w_reference, F_ext=None,
                                   n_epochs=500, lr=1e-2, optimizer_cls=None, verbose=False,
                                   **kwargs):
    """Item 100 -- the cheap, lower-risk alternative to item 98's
    implicit adjoint layer: when a full reference state w_m is already
    available (a finer-mesh solution, a full-field DIC/experimental
    measurement, or a high-fidelity ROM snapshot -- w_reference here),
    calibrate correction.parameters() by minimizing

        ||F_ext - F_int(w_m) - f_theta(w_m)||^2

    directly. NO Newton solve, no adjoint solve -- F_int(w_m) is
    computed ONCE in plain NumPy (mat's own internal-force law is not
    trainable, so nothing needs to differentiate through it) and held
    as a CONSTANT torch tensor; only f_theta(w_m) is differentiated
    w.r.t. theta, every optimizer step. This is exactly why this item
    is cheap relative to item 98: one forward+backward through f_theta
    ALONE per step, reusing the same row-by-row torch.autograd.grad-
    adjacent machinery autograd_tangent.py already established, just
    applied to a training loop instead of a single tangent evaluation.

    Built and validated FIRST, before item 98, per this wave's own
    documented risk-ordering ("Ordering logic," docs/consolidated_
    future_roadmap.md Wave 10).

    correction: any AdditiveCorrection with torch_value()/parameters()
    (a "trainable" one -- see AdditiveCorrection's own docstring).
    w_reference: (n_dof,) array-like, the FULL reference state.
    F_ext: defaults to fesystem.F.

    Returns a (n_epochs,) numpy array of the loss history -- the same
    convention rom_engine.nonlinear_rom.NeuralSurrogate.fit()'s own
    loss_history uses."""
    _require_torch()
    free = fesystem.free_dofs
    if F_ext is None:
        F_ext = fesystem.F
    w_reference = np.asarray(w_reference, dtype=float)
    F_int = fesystem.assemble_internal_force(w_reference, mat, **kwargs)
    R_const = np.asarray(F_ext, dtype=float)[free] - F_int[free]
    R_const_t = torch.tensor(R_const, dtype=torch.float64)
    w_free_t = torch.tensor(w_reference[free], dtype=torch.float64)

    params = list(correction.parameters())
    optimizer = (optimizer_cls or torch.optim.Adam)(params, lr=lr)
    history = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        f_theta_t = correction.torch_value(w_free_t)
        residual_t = R_const_t - f_theta_t
        loss = torch.sum(residual_t ** 2)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        if verbose and epoch % max(1, n_epochs // 10) == 0:
            print(f"  calibrate_correction_explicit: epoch {epoch}/{n_epochs} "
                  f"loss={loss.item():.6e}")
    return np.array(history)


# =====================================================================
# Concrete trainable corrections (torch-backed)
# =====================================================================
class ScalarFieldCorrection(AdditiveCorrection):
    """Simplest possible trainable correction: one free trainable
    scalar per ACTIVE free dof, f_theta(u) = theta (a constant additive
    term, independent of u -- jacobian() is therefore exactly zero).
    'local' by construction: theta_i only ever affects free-dof i. The
    "handful of scalars" case AdditiveCorrection's own docstring and
    item 101's MetricScaledField both contrast against a spatially-
    varying field or NN correction."""
    scope = "local"

    def __init__(self, n_free, init=0.0, dtype=None):
        _require_torch()
        self.n_free = n_free
        dtype = dtype or torch.float64
        self.theta = torch.full((n_free,), float(init), dtype=dtype, requires_grad=True)

    def value(self, u_free):
        return self.theta.detach().numpy().copy()

    def jacobian(self, u_free):
        return np.zeros((self.n_free, self.n_free))

    def torch_value(self, u_free_t):
        return self.theta

    def parameters(self):
        return [self.theta]


class MLPCorrection(AdditiveCorrection):
    """Local, feature-based trainable correction: a small shared-
    weight torch.nn MLP applied POINTWISE to each free dof's own
    scalar feature (u_free itself by default, or a caller-supplied
    feature_fn) -- f_theta(u)_i = MLP(phi(u)_i) -- matching the paper's
    LOCAL correction form f(w, alpha_theta o phi(w)) (Eq. 2.1-2.14):
    every dof is corrected by the SAME small network applied to its
    own feature, not a network that reads the whole field (that would
    be the 'nonlocal' case, not implemented here -- a caller wanting
    that can supply a feature_fn that concatenates neighbor features,
    at the cost of a denser jacobian()).

    Because the network is applied pointwise/independently to each
    dof's own scalar feature, jacobian() is exactly DIAGONAL (d f_i /
    d u_j = 0 for i != j) -- computed via n_free independent 1-D
    torch.autograd.grad calls, the same row-by-row technique autograd_
    tangent.py already uses, specialized here to a case where most
    entries are known to be zero a priori.

    Reuses the same small-MLP machinery rom_engine.nonlinear_rom.
    NeuralSurrogate already established, independently re-implemented
    here exactly like that class independently re-implements _HAS_
    TORCH/_require_torch() rather than importing rom_engine -- fea_
    engine and rom_engine deliberately share no runtime dependency
    (see rom_engine/__init__.py's own stated design principle)."""
    scope = "local"

    def __init__(self, n_free, hidden_sizes=(16, 16), feature_fn=None, seed=None, dtype=None):
        _require_torch()
        self.n_free = n_free
        self.feature_fn = feature_fn if feature_fn is not None else (lambda u: u)
        self.dtype = dtype or torch.float64
        if seed is not None:
            torch.manual_seed(seed)
        sizes = (1,) + tuple(hidden_sizes) + (1,)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(_nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(_nn.Tanh())
        self.net = _nn.Sequential(*layers).to(self.dtype)

    def torch_value(self, u_free_t):
        phi = self.feature_fn(u_free_t)
        return self.net(phi.reshape(-1, 1)).reshape(-1)

    def value(self, u_free):
        with torch.no_grad():
            u_t = torch.tensor(np.asarray(u_free, dtype=float), dtype=self.dtype)
            return self.torch_value(u_t).numpy()

    def jacobian(self, u_free):
        u_t = torch.tensor(np.asarray(u_free, dtype=float), dtype=self.dtype, requires_grad=True)
        out = self.torch_value(u_t)
        n = out.shape[0]
        diag = np.zeros(n)
        for i in range(n):
            (g,) = torch.autograd.grad(out[i], u_t, retain_graph=(i < n - 1))
            diag[i] = g[i].item()
        return np.diag(diag)

    def parameters(self):
        return list(self.net.parameters())


# =====================================================================
# Item 98: whole-solver implicit-differentiation adjoint layer
# =====================================================================
def adjoint_gradient(K_eff, grad_w, vjp_fn):
    """Shared numeric core of the whole-solver implicit-differentiation
    layer. Given the converged corrected tangent K_eff = K_T(w*) +
    d f_theta/du|_{w*} (already available for free at the end of any
    converged _newton_equilibrium() call -- no NEW tangent assembly
    needed) and an upstream gradient grad_w = dL/dw* for some scalar
    loss L, solves the adjoint system

        K_eff^T @ lam = -grad_w

    ONCE, then returns vjp_fn(lam) -- a caller-supplied vector-
    Jacobian-product callable computing lam^T @ (d f_theta/d theta)
    |_{w*}, which by the implicit function theorem equals dL/dtheta
    exactly. Derivation (root condition G(w,theta) = F_ext - F_int(w)
    - f_theta(w,theta) = 0; theta enters G ONLY through f_theta):

        dw*/dtheta = -(dG/dw)^-1 (dG/dtheta)
        dL/dtheta  = (dL/dw*) @ dw*/dtheta
                   = -(dL/dw*) (dG/dw)^-1 (dG/dtheta)
        define lam by (dG/dw)^T @ lam = (dL/dw*)^T = grad_w
        dG/dw = -K_eff  =>  -K_eff^T @ lam = grad_w
                         =>  K_eff^T @ lam = -grad_w
        dL/dtheta = -lam^T (dG/dtheta) = -lam^T (-d f_theta/d theta)
                  = lam^T (d f_theta/d theta) = vjp_fn(lam)

    Deliberately factored out of BOTH the pure-NumPy finite-difference
    cross-check (tests/test_differentiable.py::TestAdjointMath, no
    torch needed -- vjp_fn is a hand-coded closed form there) and the
    torch.autograd.Function wrapper below (vjp_fn calls torch.autograd.
    grad there) -- validating this ONE function validates the
    mathematical content behind both."""
    K_eff = np.asarray(K_eff, dtype=float)
    grad_w = np.asarray(grad_w, dtype=float)
    lam = np.linalg.solve(K_eff.T, -grad_w)
    return vjp_fn(lam)


def _params_to_vector(params):
    return torch.cat([p.reshape(-1) for p in params])


def _vector_to_params(vec, params):
    i = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(vec[i:i + n].reshape(p.shape))
            i += n


class _ImplicitCorrectedSolve(torch.autograd.Function if _HAS_TORCH else object):
    """torch.autograd.Function wrapping ONE call to _newton_
    equilibrium() (item 99's plain-NumPy corrected Newton solve) as a
    black-box forward pass, with a backward() that calls adjoint_
    gradient() above instead of differentiating through the Newton
    iteration -- the actual mechanism item 98 exists to provide (see
    this module's own docstring, and the roadmap's own framing:
    "without this, any correction calibrated against partial/indirect
    data has no way to get a gradient back to theta short of unrolling
    the whole Newton iteration by hand").

    Static-method torch.autograd.Function convention: forward()/
    backward() cannot be ordinary bound methods with closures over
    per-call state, so fesystem/mat/correction/kwargs are threaded
    through as extra (non-differentiated, `None`-gradient) forward()
    arguments instead -- implicit_correction_solve() below is the
    actual public entry point that hides this mechanical detail.

    Class only actually defined with torch.autograd.Function as its
    base when torch is importable (`_HAS_TORCH`) -- subclassing a
    real `object` otherwise keeps this module importable without
    torch, consistent with every other class/function here."""

    @staticmethod
    def forward(ctx, theta_vec, fesystem, mat, correction, F_ext_np, u0_np, tol, max_iter):
        _vector_to_params(theta_vec.detach(), list(correction.parameters()))
        u_full, u_free, K_eff, F_int, converged, Rn = _newton_equilibrium(
            fesystem, mat, correction, F_ext_np, u0=u0_np, tol=tol, max_iter=max_iter)
        if not converged:
            raise RuntimeError(
                f"_ImplicitCorrectedSolve: corrected Newton solve did not "
                f"converge (|R|={Rn:.3e} after {max_iter} iterations) -- "
                "cannot differentiate through a non-converged forward pass.")
        ctx.correction = correction
        ctx.K_eff = K_eff
        ctx.dtype = theta_vec.dtype
        w_free_t = torch.tensor(u_free, dtype=theta_vec.dtype)
        ctx.save_for_backward(theta_vec, w_free_t)
        return w_free_t.clone()

    @staticmethod
    def backward(ctx, grad_w):
        theta_vec, w_free_t = ctx.saved_tensors
        correction = ctx.correction
        _vector_to_params(theta_vec.detach(), list(correction.parameters()))
        for p in correction.parameters():
            p.requires_grad_(True)
        w_free_t_const = w_free_t.detach().clone()

        def vjp_fn(lam_np):
            lam_t = torch.tensor(lam_np, dtype=ctx.dtype)
            f_theta_t = correction.torch_value(w_free_t_const)
            params = list(correction.parameters())
            grads = torch.autograd.grad(f_theta_t, params, grad_outputs=lam_t, retain_graph=False)
            return _params_to_vector(grads)

        grad_theta_vec = adjoint_gradient(ctx.K_eff, grad_w.detach().numpy(), vjp_fn)
        return (grad_theta_vec, None, None, None, None, None, None, None)


def implicit_correction_solve(fesystem, mat, correction, F_ext=None, u0=None,
                               tol=1e-8, max_iter=30):
    """Public entry point for item 98: solves the corrected equilibrium
    F_ext - F_int(w) - f_theta(w) = 0 for w* via the ORDINARY plain-
    NumPy Newton ladder (_newton_equilibrium(), fully reused from item
    99 -- no reimplementation), and returns w*_free as a torch tensor
    that IS differentiable w.r.t. correction.parameters() via the
    adjoint backward pass above, despite the forward pass never having
    been run through autograd at all -- the actual "train a correction
    against partial/indirect data, through the converged state"
    capability this whole item exists for.

    F_ext: defaults to fesystem.F (the load already built via
    add_nodal_force()/etc.) -- pass an explicit array for a scaled/
    different load case. u0: initial guess, defaults to zeros.

    Returns a torch tensor of shape (len(free_dofs),) -- call
    .detach().numpy() for the plain equilibrium displacement, or keep
    it inside a torch loss computation (e.g. matching a handful of
    sparse sensor dofs) to train correction.parameters() through it
    via ordinary loss.backward()."""
    _require_torch()
    if F_ext is None:
        F_ext = fesystem.F
    params = list(correction.parameters())
    theta_vec = _params_to_vector(params)
    return _ImplicitCorrectedSolve.apply(
        theta_vec, fesystem, mat, correction, np.asarray(F_ext, dtype=float), u0, tol, max_iter)


# =====================================================================
# Item 101: metric-consistent gradient scaling for spatial-field
# parameters
# =====================================================================
def mass_cholesky_factor(M_sub):
    """Cholesky factor N of a (symmetric positive definite) mass
    submatrix M_sub = N @ N.T (Saverio et al. 2026 Appendix C) -- the
    change-of-variables matrix behind MetricScaledField below. M_sub:
    (n,n) dense numpy array or SciPy-sparse (densified here; a
    spatial-field correction's own dof count is expected to be small
    relative to the full mesh -- see MetricScaledField's own docstring
    for why this is a reasonable scope limit, not an oversight). Pure
    NumPy -- no torch needed, usable and tested unconditionally."""
    M_dense = np.asarray(M_sub.todense() if hasattr(M_sub, "todense") else M_sub, dtype=float)
    return np.linalg.cholesky(M_dense)


def metric_step_matches_euclidean_reparametrized_step(N, grad_theta, lr=0.1):
    """Decisive, torch-free identity check of MetricScaledField's whole
    point: ONE ordinary Euclidean gradient-descent step in the
    reparametrized coordinate theta_tilde = N^-1 @ theta (equivalently
    theta = N @ theta_tilde) is EXACTLY a mass-matrix-metric-
    preconditioned ("natural gradient") step in theta itself.

        Euclidean step in theta_tilde (chain rule, theta = N @
        theta_tilde => dL/dtheta_tilde = N.T @ dL/dtheta):
            theta_tilde_new = theta_tilde_old - lr * N.T @ grad_theta
        Mapping back:
            theta_new = N @ theta_tilde_new
                      = theta_old - lr * (N @ N.T) @ grad_theta
                      = theta_old - lr * M @ grad_theta

    -- an M-preconditioned step, exactly Appendix C's point: a plain
    Euclidean optimizer never needs to know M exists once theta_tilde
    is what it actually updates. Returns (delta_theta_reparam,
    delta_theta_metric) -- a caller/test checks these match to machine
    precision directly, rather than trusting this docstring's algebra
    alone."""
    N = np.asarray(N, dtype=float)
    grad_theta = np.asarray(grad_theta, dtype=float)
    M = N @ N.T
    delta_theta_reparam = -lr * (N @ (N.T @ grad_theta))
    delta_theta_metric = -lr * (M @ grad_theta)
    return delta_theta_reparam, delta_theta_metric


class MetricScaledField(AdditiveCorrection):
    """Wraps a plain per-dof/per-element scalar-field correction (theta
    itself, i.e. ScalarFieldCorrection's own shape) so its trainable
    parameter is theta_tilde = N^-1 @ theta rather than raw theta -- see
    mass_cholesky_factor()/metric_step_matches_euclidean_reparametrized
    _step() above for why a raw-Euclidean optimizer step on theta_tilde
    becomes a metric-consistent step on theta automatically, with NO
    change needed to the optimizer or training loop itself (the whole
    appeal of a change-of-variables fix over a custom optimizer).

    WHY THIS MATTERS, not cosmetic: the moment item 99/100 calibrates a
    SPATIALLY-VARYING field parameter (one scalar per node/element,
    analogous to the paper's mu_t(x)/beta(x)) rather than a handful of
    scalars or NN weights (whose natural coordinate already IS
    Euclidean), a standard optimizer's implicit Euclidean metric makes
    the effective per-location step size MESH-DENSITY-DEPENDENT -- a
    locally refined region's parameters move at a different EFFECTIVE
    rate than a coarse region's, purely as a parametrization artifact,
    not the physics. Rescaling by the mesh's own mass-matrix-consistent
    discrete L2 inner product (<a,b>_M = a^T M b, already used
    elsewhere in this package as a discrete L2 inner product -- see
    FESystem's consistent/lumped mass matrices) removes that artifact.

    M_sub: the (n_field, n_field) mass submatrix restricted to this
    field's own dofs (e.g. fesystem.M[np.ix_(field_dofs, field_dofs)]
    after assemble_mass()) -- the caller's own responsibility to supply
    (this class doesn't assume a specific FESystem layout for what "the
    field's own dofs" means)."""
    scope = "local"

    def __init__(self, n_field, M_sub, init=0.0, dtype=None):
        _require_torch()
        self.n_field = n_field
        self.dtype = dtype or torch.float64
        self.N = mass_cholesky_factor(M_sub)
        self.N_t = torch.tensor(self.N, dtype=self.dtype)
        theta0 = np.full(n_field, float(init))
        theta_tilde0 = np.linalg.solve(self.N, theta0)
        self.theta_tilde = torch.tensor(theta_tilde0, dtype=self.dtype, requires_grad=True)

    def _theta_t(self):
        return self.N_t @ self.theta_tilde

    def value(self, u_free):
        return self._theta_t().detach().numpy().copy()

    def jacobian(self, u_free):
        return np.zeros((self.n_field, self.n_field))

    def torch_value(self, u_free_t):
        return self._theta_t()

    def parameters(self):
        return [self.theta_tilde]
