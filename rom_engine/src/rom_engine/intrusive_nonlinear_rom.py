"""
intrusive_nonlinear_rom.py -- Wave 17 item 144 (fea_engine/docs/
consolidated_future_roadmap.md): an INTRUSIVE nonlinear Galerkin ROM
that keeps the reduced mass/damping/stiffness matrices FULL (not
mass-normalized, not diagonal), matching Georgiou (2005)'s Eqs. 29-39
POD-Galerkin construction of a planar nonlinear rod's reduced
equations of motion.

WHY THIS IS A SEPARATE MODULE FROM nonlinear_dynamics.py. That module
(and the `nonlinear_rom.ReducedForceModel` protocol it drives) has a
STANDING, documented convention -- see both modules' own docstrings --
of working in MASS-NORMALIZED modal coordinates: `M_r = I` exactly, by
construction (the basis is always `pod.PodBasis`'s own mass-weighted,
M-orthonormal output, or an ordinary mass-normalized eigenbasis), so
`integrate_newmark_surrogate()` never needs to solve a mass-matrix
linear system at all -- every `a0c*I`/`Keff = diag(Lambda) + ...` term
in that module hardcodes the identity in place of a real `M_r`. This
item's own roadmap entry is explicit that `M_r` must NOT be
normalized here: the POD-RM diagnostics item 145 builds on top of this
module need the RAW `M_r`/`D_r`/`K_r` (Georgiou's own Eqs. 41-44,
50-60 are stated directly in terms of the un-normalized reduced
matrices, not a normalized ratio of them). Forcing this module through
`nonlinear_dynamics.py`'s `M_r=I` contract would mean silently
mass-normalizing the basis inside this class, exactly what the roadmap
says not to do -- so this is a deliberately separate, sibling module,
not a generalization bolted onto the existing one. Where the algebra
genuinely IS the same generalization (the reduced Newton-Newmark
integrator, see `integrate_newton_newmark()` below), that generalization
is derived explicitly in this module's own docstrings, keyed back to
`nonlinear_dynamics.integrate_newmark_surrogate`'s own "newton"
correction mode, rather than re-derived from scratch.

THE MODEL. Given a fixed reduced basis `V` (n_dof x n_modes -- from
`pod.PodBasis`, an eigenbasis, or any other caller-supplied basis; NOT
assumed mass-orthonormal), a full-order mass matrix `M`, a full-order
damping matrix `C`, and two full-order callbacks --
`internal_force_fn(u_full) -> f_int_full` and (optionally)
`tangent_fn(u_full) -> K_T_full` -- this class builds

    M_r = V^T M V          D_r = V^T C V          K_r = V^T K_T(0) V

(`K_r` -- the LINEARIZED reduced stiffness at the undeformed state --
needs one call to `tangent_fn` at `u_full = 0`, done once here at
construction) and integrates the reduced nonlinear equation of motion

    M_r qddot + D_r qdot + f_int_r(q) = F_ext_r(t)

where `f_int_r(q) = V^T @ internal_force_fn(V @ q)` is the FULL
(linear + nonlinear) reduced restoring force, evaluated through the
REAL full-order element's own `internal_force()` at every query -- the
FE-consistent analogue of the paper's own Gauss-Legendre integral
nonlinearity (Eqs. 29-33), with no separate reduced-order
approximation of the nonlinearity itself (no hyper-reduction: a 1-D
rod with a few hundred dofs is cheap enough that a full-order
`internal_force_fn` call every step/stage is fine -- ECSW/DEIM/gappy-POD
are deliberately out of scope here, per this item's own roadmap row).
`f_nl(q) = f_int_r(q) - K_r @ q` (the pure NONLINEAR part, `K_r @ q`
being the linear part already captured by the zero-state tangent) is
exposed separately as `f_nl()` below since it is the quantity item
145's diagnostics, and the paper's own Eq. 38 property statement, are
phrased in terms of.

INTEGRATORS. Three, all explicitly named, none replacing another (this
project's standing "keep every valid implementation available" rule --
see e.g. `nonlinear_dynamics.integrate_newmark_surrogate`'s own four
named `correction` modes for the precedent):

  - `integrate_rk4()` -- fixed-step, classical 4-stage RK4 on the
    first-order form `[qdot; qddot]`, the paper's OWN integration
    choice (Eqs. 34-36 describe exactly this). Built on
    `parameterized_latent_ode.rk4_step()`/`integrate_rk4()` (Wave 13
    item 119's own CORRECTLY-implemented, order-of-convergence-
    validated generic RK4 -- reused directly, not re-derived, per this
    project's "don't re-derive already-validated math" convention).
  - `integrate_solve_ivp()` -- `scipy.integrate.solve_ivp` on the
    IDENTICAL first-order right-hand side (`_rhs()` below), so the two
    integrators can be cross-checked against each other directly (see
    this module's own test suite) as well as against the full-order
    ground truth.
  - `integrate_newton_newmark()` -- a genuine Newton-Raphson-corrected
    implicit Newmark-beta step every step, for stiff cases where a
    fixed-step explicit/adaptive-explicit integrator would need a
    prohibitively small `dt`. See its own docstring for the exact
    generalization from `nonlinear_dynamics.integrate_newmark_surrogate`'s
    `correction="newton"` mode (Wave 12 item 114) to a non-diagonal
    `M_r`/`D_r`.

Both `M_r` and the Newton-Newmark effective stiffness are LU-factored
once (mass) or reassembled analytically (Newmark `Keff`, itself
constant across steps since only the *nonlinear* part of the tangent
changes) rather than re-inverted from scratch every call.

TORCH BACKEND (additive addendum to item 144, same roadmap row --
fea_engine/docs/consolidated_future_roadmap.md). Per this project's
standing "whenever more than one valid implementation/backend exists
for the same capability, keep all of them available and give the
caller an explicit, named choice, never silently prefer or replace
one" rule (the same rule behind FESystem's own `backend="torch"`
option in fea_engine/solver.py, fea_engine/torch_sparse_solver.py, and
this same package's own `torch_linalg.py`), `IntrusiveNonlinearROM`
gets a handful of `torch_*`/`*_torch` METHODS alongside its existing
plain-NumPy ones -- not a `device=` parameter bolted onto
`integrate_rk4()` etc. (that would make the torch path implicit rather
than an explicit opt-in), matching `torch_linalg.py`'s own convention
of a separate, explicitly-named parallel function (there: module-level
`torch_solve_continuous_lyapunov()` beside `scipy.linalg.
solve_continuous_lyapunov`; here: `integrate_rk4_torch()` beside
`integrate_rk4()`, etc.) rather than a same-name dispatch parameter.
Every existing method above is completely untouched -- `import
rom_engine` and every existing test still needs no torch install at
all.

  - `torch_reduced_matrices()` -- an INDEPENDENT torch re-derivation of
    `M_r = V^T M V`, `D_r = V^T C V`, `K_r = V^T K_T(0) V` (via genuine
    torch matmuls on `self._K0_full`, the full zero-state tangent
    stored at construction time for exactly this reuse -- not merely
    `torch.as_tensor(self.M_r)`, which would just be a dtype cast of an
    already-trusted NumPy result, not a real cross-check). Kept FULL /
    non-diagonal, exactly like the NumPy path -- no basis mass-
    normalization creeps in here either, since it is the identical
    `V^T A V` projection, just executed with torch ops instead of NumPy
    ones.
  - `integrate_rk4_torch()` -- mirrors `integrate_rk4()`'s exact step
    logic. Reuses `parameterized_latent_ode.rk4_step()` VERBATIM (not
    reimplemented for torch): that function only ever does `+`, `*`,
    and calls `func` -- no NumPy-specific operation anywhere in it --
    so it already works unchanged on `torch.Tensor` state, the same
    "don't re-derive already-validated math" reasoning the NumPy
    `integrate_rk4()` itself gives for reusing it. Accepts OPTIONAL
    `internal_force_fn_torch`/`load_fn_torch` torch-native callbacks;
    if omitted, falls back to this instance's own plain-NumPy
    `internal_force_fn`/`load_fn` via a NumPy round-trip every RK4
    stage (correct, but no GPU-residency benefit). Passing torch-native
    callables directly is the actual point of this method: it lets a
    future torch-native full-order model (e.g. a `Beam2DReissner`
    torch-autograd internal-force/tangent -- checked directly as of
    this writing, `fea_engine/autograd_tangent.py` covers `Tet4NeoHookean`
    /`Tet10SolidTL`/`Shell4MITCCorotational` only, NOT `Beam2DReissner`,
    so this interface is designed for that future addition rather than
    consuming it now) integrate every RK4 stage on-device with zero
    NumPy round-trips.
  - `integrate_newton_newmark_torch()` -- mirrors
    `integrate_newton_newmark()`'s exact `a0c..a7c`/`Keff`/`G`/`J`
    algebra (not re-derived, ported mechanically: `np.linalg.solve` ->
    `torch.linalg.solve`, `np.max(np.abs(.))` -> `torch.max(torch.abs(.))`,
    etc.), including the same Newton-with-backtracking loop. Same
    optional `internal_force_fn_torch`/`tangent_fn_torch`/
    `load_fn_torch` override pattern as `integrate_rk4_torch()`.
  - `integrate_solve_ivp()` gets NO torch counterpart, deliberately --
    `scipy.integrate.solve_ivp` has no meaningful torch-tensor
    equivalent to swap in (torch's own ODE-solver ecosystem lives in a
    separate `torchdiffeq`-style package, out of scope here and not a
    bare-torch op the way this module's other torch additions are); the
    NumPy `integrate_solve_ivp()` above is unaffected and remains the
    only solve_ivp-based path.

HONESTY NOTE (development-sandbox limitation -- see torch_linalg.py's
and torch_sparse_solver.py's own identical notes for the full
explanation of WHY: a CUDA-linked PyPI `torch` wheel cannot actually be
imported in this project's own development sandbox, `ModuleNotFoundError`/
import failure confirmed directly). Every `torch_*`/`*_torch` method
above was written and reasoned through carefully (dtype consistency,
`.detach().cpu().numpy()`/`torch.as_tensor()` conversions placed only
at the documented NumPy-fallback boundary, gradients never required to
flow through the integrator's own step-count bookkeeping) but NOT yet
executed on a real torch install -- `tests/test_intrusive_nonlinear_rom.py`'s
own torch-gated tests (see `_HAS_TORCH` there) are written to run for
real and SHOULD be run at least once on a torch-equipped machine, the
same status every other torch addition in this codebase carries until
that happens.
"""
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, lu_factor, lu_solve
from scipy.optimize import linear_sum_assignment

from .parameterized_latent_ode import rk4_step

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except Exception:
    # Bare `except Exception`, not `except ImportError` -- see
    # torch_linalg.py's own identical block (same package, same reason:
    # a CUDA-linked PyPI wheel with no matching CUDA runtime can raise
    # OSError/ValueError at import time, not ImportError). Duplicated
    # here rather than imported from torch_linalg.py, per this module's
    # own `_as_array()` docstring's "no cross-module rom_engine imports
    # in the core library" convention.
    pass


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "intrusive_nonlinear_rom.py: PyTorch is not importable in this "
            "environment -- install it (`pip install torch`) to use "
            "IntrusiveNonlinearROM's torch_reduced_matrices()/"
            "integrate_rk4_torch()/integrate_newton_newmark_torch(); every "
            "other method on this class (the default NumPy path) works "
            "without it.")


def _as_array(x):
    """np.asarray(x), preserving complex dtype -- see pod._as_array's
    own docstring for the full rationale (identical helper, duplicated
    rather than imported to keep every rom_engine core module's own
    dependency footprint to scipy/numpy only, per this package's
    standing "no cross-module rom_engine imports in the core library"
    convention -- see galerkin.py's own identical duplication note)."""
    x = np.asarray(x)
    return x if np.iscomplexobj(x) else x.astype(float, copy=False)


def impulse_to_reduced_velocity(V, M_r, P_full, tau0):
    """Georgiou (2005) Eqs. 47-48, generalized: an impulsive full-order
    load of magnitude/pattern `P_full` (a plain (n_dof,) force vector)
    applied over a short duration `tau0` imparts a full-order momentum
    change `M @ v0 = P_full * tau0`, i.e. `v0 = M^-1 @ (P_full * tau0)`
    -- the paper's own scalar `v0 = P*tau0/mass` special case for a
    single-dof lumped-mass system. Reducing THAT relation the same way
    `IntrusiveNonlinearROM.initial_conditions()` reduces an initial
    velocity (mass-weighted projection through `M_r`) gives

        M_r @ qdot0 = V^T @ (M @ v0) = V^T @ (P_full * tau0)

    -- note the full mass matrix `M` and its inverse never need to be
    formed or applied at all: `V^T @ (P_full * tau0)` is already the
    right-hand side of the REDUCED system, so only the small `M_r`
    solve (already LU-factored by the caller) is needed. A free
    (module-level) function, not a method tied to one class instance's
    own stored `M`, so it can be reused directly on any `(V, M_r)` pair
    -- e.g. from a script that never builds a full `IntrusiveNonlinearROM`
    -- and so it is generic to ANY full-order load PATTERN `P_full`
    (not hardcoded to the paper's own single-dof rod problem).

    Parameters
    ----------
    V : (n_dof, n_modes) ndarray
        The reduced basis.
    M_r : (n_modes, n_modes) ndarray, or a `scipy.linalg.lu_factor`
        tuple
        The reduced mass matrix (or its pre-factored LU form, as
        `IntrusiveNonlinearROM._M_r_lu` already is -- passing that
        directly avoids a redundant re-factorization).
    P_full : (n_dof,) array_like
        The full-order impulsive force PATTERN (not yet scaled by
        `tau0`).
    tau0 : float
        The impulse duration.

    Returns
    -------
    qdot0 : (n_modes,) ndarray
        The reduced initial velocity.
    """
    V = _as_array(V)
    P_full = _as_array(P_full)
    rhs = V.T @ (P_full * tau0)
    if isinstance(M_r, tuple):
        return lu_solve(M_r, rhs)
    return np.linalg.solve(_as_array(M_r), rhs)


class IntrusiveNonlinearROM:
    """Intrusive nonlinear Galerkin ROM with a full (non-diagonal,
    non-mass-normalized) reduced mass/damping/stiffness system -- see
    module docstring for the full derivation and the reasoning behind
    keeping this deliberately separate from `nonlinear_dynamics.py`.

    Parameters
    ----------
    V : (n_dof, n_modes) ndarray, or an object with a `.V` attribute
        (e.g. a fitted `pod.PodBasis`)
        The reduced basis -- NOT assumed mass-orthonormal (see module
        docstring: `M_r` is deliberately kept non-diagonal, whatever
        `V` happens to give).
    M, C : (n_dof, n_dof) ndarray
        Full-order mass / damping matrices, plain arrays (this class,
        like every other rom_engine core module, never imports
        fea_engine -- a caller builds `M`/`C` however it likes, e.g.
        via `fea_engine.FESystem.assemble_mass()`/`assemble_damping()`
        in a script or test, and passes the resulting plain ndarrays
        in here).
    internal_force_fn : callable(u_full) -> (n_dof,) ndarray
        Full-order internal (elastic, in general nonlinear) restoring
        force at a given full-order displacement state.
    load_fn : callable(t) -> (n_dof,) ndarray
        Full-order external force at time `t`.
    tangent_fn : callable(u_full) -> (n_dof, n_dof) ndarray, optional
        Full-order tangent stiffness at a given state. Required to
        compute `K_r` (called once, at `u_full = 0`, right here in
        `__init__` -- see module docstring) and re-used at every
        iteration by `integrate_newton_newmark()` only; `integrate_rk4()`/
        `integrate_solve_ivp()` never call it again after construction.
        Documented as an optional constructor keyword (matching the
        roadmap's own signature, since a caller integrating with
        RK4/solve_ivp only conceptually needs it once, not "for every
        step" the way the Newmark integrator does), but genuinely
        required in practice -- `K_r` cannot be computed without it,
        and `K_r` is needed by every integrator here (it is the linear
        part of `f_int_r`, subtracted out to define `f_nl()`) -- so
        `None` raises immediately with a clear message rather than
        deferring to a confusing failure deep inside a later call.

    Attributes (set in __init__)
    -----------------------------
    M_r, D_r, K_r : (n_modes, n_modes) ndarray
        The reduced mass / damping / (zero-state tangent) stiffness
        matrices -- deliberately kept FULL, never mass-normalized (see
        module docstring).
    """

    def __init__(self, V, M, C, internal_force_fn, load_fn, tangent_fn=None):
        self.V = V.V if hasattr(V, "V") else _as_array(V)
        if self.V.ndim != 2:
            raise ValueError(f"V must be 2-D (n_dof, n_modes), got shape {self.V.shape}")
        self.n_dof, self.n_modes = self.V.shape

        self.M = _as_array(M)
        self.C = _as_array(C)
        self.internal_force_fn = internal_force_fn
        self.load_fn = load_fn
        self.tangent_fn = tangent_fn

        if tangent_fn is None:
            raise ValueError(
                "IntrusiveNonlinearROM requires tangent_fn to compute K_r = "
                "V^T @ tangent_fn(0) @ V at construction time (K_r is the "
                "linear part of every integrator's own reduced force, not "
                "an optional diagnostic) -- pass the full-order tangent "
                "stiffness callback, e.g. fesystem.assemble_tangent_stiffness.")

        K0_full = _as_array(tangent_fn(np.zeros(self.n_dof)))
        self.M_r = self.V.T @ self.M @ self.V
        self.D_r = self.V.T @ self.C @ self.V
        self.K_r = self.V.T @ K0_full @ self.V
        self._M_r_lu = lu_factor(self.M_r)
        # Retained (not just used-and-discarded) for
        # torch_reduced_matrices() below -- an independent torch
        # re-derivation of K_r needs the same full-order zero-state
        # tangent this constructor already computed once, not a second
        # call to tangent_fn(0).
        self._K0_full = K0_full

    # -----------------------------------------------------------------
    # Reduced force evaluation
    # -----------------------------------------------------------------
    def reduced_internal_force(self, q):
        """f_int_r(q) = V^T @ internal_force_fn(V @ q) -- the FULL
        (linear + nonlinear) reduced restoring force, evaluated through
        the real full-order internal-force routine."""
        q = _as_array(q)
        u_full = self.V @ q
        f_full = _as_array(self.internal_force_fn(u_full))
        return self.V.T @ f_full

    def f_nl(self, q):
        """f_nl(q) = f_int_r(q) - K_r @ q -- the pure NONLINEAR part of
        the reduced restoring force (the FE-consistent equivalent of
        the paper's own Gauss-Legendre integral nonlinearity, Eqs.
        29-33), i.e. what remains after subtracting the linear
        (zero-state-tangent) contribution `K_r` already captures."""
        q = _as_array(q)
        return self.reduced_internal_force(q) - self.K_r @ q

    def reduced_tangent(self, q):
        """K_T_r(q) = V^T @ tangent_fn(V @ q) @ V -- the reduced
        tangent stiffness at a general (not necessarily zero) reduced
        state; used only by `integrate_newton_newmark()`."""
        if self.tangent_fn is None:
            raise RuntimeError("reduced_tangent() needs tangent_fn (see __init__).")
        q = _as_array(q)
        u_full = self.V @ q
        K_full = _as_array(self.tangent_fn(u_full))
        return self.V.T @ K_full @ self.V

    def reduced_load(self, t):
        """F_ext_r(t) = V^T @ load_fn(t)."""
        return self.V.T @ _as_array(self.load_fn(t))

    # -----------------------------------------------------------------
    # Reduced initial conditions -- mass-weighted projection through
    # M_r (NOT a naive V^T u0, which is only correct for a
    # mass-orthonormal basis -- see module docstring on M_r being kept
    # general/non-diagonal here).
    # -----------------------------------------------------------------
    def initial_conditions(self, u0_full=None, v0_full=None):
        """Project a full-order initial displacement/velocity into
        reduced coordinates by mass-weighted projection through `M_r`:

            q0    = (V^T M V)^-1 @ V^T @ M @ u0   = M_r^-1 @ V^T @ (M @ u0)
            qdot0 = (V^T M V)^-1 @ V^T @ M @ v0   = M_r^-1 @ V^T @ (M @ v0)

        the mass-orthogonal (Galerkin-consistent) projection -- reduces
        to the familiar `q0 = V^T @ u0` ONLY in the special case where
        `V` is already mass-orthonormal (`V^T M V = I`), which this
        class deliberately does not assume (see module docstring).
        Either argument may be omitted (returns `None` for that one) --
        e.g. call with only `u0_full` for a released-from-rest IC.

        Returns
        -------
        (q0, qdot0) : each an (n_modes,) ndarray, or None if the
        corresponding *_full argument was not given.
        """
        q0 = None
        if u0_full is not None:
            rhs = self.V.T @ (self.M @ _as_array(u0_full))
            q0 = lu_solve(self._M_r_lu, rhs)
        qdot0 = None
        if v0_full is not None:
            rhs = self.V.T @ (self.M @ _as_array(v0_full))
            qdot0 = lu_solve(self._M_r_lu, rhs)
        return q0, qdot0

    def impulse_to_reduced_velocity(self, P_full, tau0):
        """Instance-method convenience wrapper around the module-level
        `impulse_to_reduced_velocity()`, reusing this instance's own
        already-factored `M_r` (no redundant re-factorization)."""
        return impulse_to_reduced_velocity(self.V, self._M_r_lu, P_full, tau0)

    # -----------------------------------------------------------------
    # First-order right-hand side, shared by integrate_rk4() and
    # integrate_solve_ivp() -- ONE function, so the two integrators are
    # genuinely integrating the identical ODE, not two independently
    # (and possibly inconsistently) written ones.
    # -----------------------------------------------------------------
    def _rhs(self, t, z):
        n = self.n_modes
        q, qdot = z[:n], z[n:]
        f_int_r = self.reduced_internal_force(q)
        rhs_force = self.reduced_load(t) - self.D_r @ qdot - f_int_r
        qddot = lu_solve(self._M_r_lu, rhs_force)
        return np.concatenate([qdot, qddot])

    # -----------------------------------------------------------------
    # Integrator 1: fixed-step RK4 (the paper's own choice).
    # -----------------------------------------------------------------
    def integrate_rk4(self, q0, qdot0, dt, n_steps):
        """Fixed-step, classical 4-stage RK4 on `z = [q; qdot]`, built
        on `parameterized_latent_ode.rk4_step()` (Wave 13 item 119's
        already order-of-convergence-validated generic RK4 -- reused,
        not re-derived).

        Returns
        -------
        t : (n_steps+1,) ndarray
        q_hist, qdot_hist : (n_steps+1, n_modes) ndarray
        """
        n = self.n_modes
        z = np.concatenate([_as_array(q0), _as_array(qdot0)])
        t = np.arange(n_steps + 1) * dt
        z_hist = np.zeros((n_steps + 1, 2 * n))
        z_hist[0] = z
        for i in range(n_steps):
            z = rk4_step(self._rhs, t[i], dt, z)
            z_hist[i + 1] = z
        return t, z_hist[:, :n], z_hist[:, n:]

    # -----------------------------------------------------------------
    # Integrator 2: scipy.integrate.solve_ivp on the SAME first-order
    # right-hand side.
    # -----------------------------------------------------------------
    def integrate_solve_ivp(self, q0, qdot0, t_span, t_eval=None, **kwargs):
        """`scipy.integrate.solve_ivp` on the identical `_rhs()` used
        by `integrate_rk4()` -- any `solve_ivp` keyword (`method=`,
        `rtol=`, `atol=`, ...) is forwarded unchanged.

        Returns
        -------
        t : (n_eval,) ndarray
        q_hist, qdot_hist : (n_eval, n_modes) ndarray
        sol : the raw scipy OdeResult, for callers that want it
              (event info, success flag, ...).
        """
        n = self.n_modes
        z0 = np.concatenate([_as_array(q0), _as_array(qdot0)])
        sol = solve_ivp(self._rhs, t_span, z0, t_eval=t_eval, **kwargs)
        return sol.t, sol.y[:n].T, sol.y[n:].T, sol

    # -----------------------------------------------------------------
    # Integrator 3: reduced Newton-Newmark, generalized from Wave 12
    # item 114 (nonlinear_dynamics.integrate_newmark_surrogate's
    # correction="newton" mode) to a non-diagonal M_r/D_r.
    # -----------------------------------------------------------------
    def integrate_newton_newmark(self, q0, qdot0, dt, n_steps,
                                  beta=0.25, gamma=0.5,
                                  tol=1e-9, max_iter=30, max_backtrack=30):
        """Implicit Newmark-beta time integration with a genuine
        Newton-Raphson corrector every step -- for stiff cases where
        `integrate_rk4()`/`integrate_solve_ivp()` would need an
        impractically small `dt`, per this item's own roadmap row.

        DERIVATION (generalizing `nonlinear_dynamics.
        integrate_newmark_surrogate`'s `correction="newton"` mode,
        Wave 12 item 114, to a non-diagonal `M_r`/`D_r` -- that
        function cannot be called here directly because it hardcodes
        the `M_r = I` mass-normalized convention throughout, e.g. its
        `Keff = diag(Lambda) + a0c*I + a1c*C_mat` and its
        `qddot_new = a0c*(q_new-q) - ...` predictor update both assume
        an identity mass matrix; every `I`/diagonal-`Lambda` occurrence
        there is replaced below by the corresponding full matrix
        `M_r`/`K_r`, which is mechanically the SAME generalization, not
        a re-derivation from different first principles):

            a0c = 1/(beta*dt^2),  a1c = gamma/(beta*dt),  a2c = 1/(beta*dt)
            a3c = 1/(2*beta) - 1, a4c = gamma/beta - 1,   a5c = dt/2*(gamma/beta-2)
            a6c = dt*(1-gamma),   a7c = dt*gamma

            rhs_base   = F_ext_r(t+dt) + M_r@(a0c*q + a2c*qdot + a3c*qddot)
                                       + D_r@(a1c*q + a4c*qdot + a5c*qddot)
            Keff       = K_r + a0c*M_r + a1c*D_r            (CONSTANT across steps)
            G(q_trial) = Keff @ q_trial - rhs_base + f_nl(q_trial)
                       = a0c*M_r@q_trial + a1c*D_r@q_trial + f_int_r(q_trial) - rhs_base
            J(q_trial) = a0c*M_r + a1c*D_r + reduced_tangent(q_trial)

        (the two `G(q_trial)` forms are algebraically identical --
        `Keff@q_trial + f_nl(q_trial) = K_r@q_trial + a0c*M_r@q_trial +
        a1c*D_r@q_trial + f_int_r(q_trial) - K_r@q_trial`, the `K_r`
        terms cancel -- the first form is kept in the implementation
        below because it mirrors `integrate_newmark_surrogate`'s own
        `Keff@q + force_model.predict(q)` structure line-for-line,
        making the two functions directly diffable against each other).
        Newton with backtracking line search (halve the step until the
        max-norm residual actually decreases, or `max_backtrack` is
        exhausted) -- the identical globalization
        `integrate_newmark_surrogate`'s own "newton" mode uses.

        Returns
        -------
        t : (n_steps+1,) ndarray
        q_hist, qdot_hist : (n_steps+1, n_modes) ndarray
        """
        n = self.n_modes
        a0c = 1.0 / (beta * dt ** 2); a1c = gamma / (beta * dt); a2c = 1.0 / (beta * dt)
        a3c = 1.0 / (2 * beta) - 1.0; a4c = gamma / beta - 1.0
        a5c = dt / 2.0 * (gamma / beta - 2.0)
        a6c = dt * (1.0 - gamma); a7c = dt * gamma

        Keff = self.K_r + a0c * self.M_r + a1c * self.D_r

        q = _as_array(q0).copy()
        qdot = _as_array(qdot0).copy()
        f0 = self.reduced_internal_force(q)
        qddot = np.linalg.solve(self.M_r, self.reduced_load(0.0) - self.D_r @ qdot - f0)

        t = np.arange(n_steps + 1) * dt
        q_hist = np.zeros((n_steps + 1, n))
        qdot_hist = np.zeros((n_steps + 1, n))
        q_hist[0], qdot_hist[0] = q, qdot

        for step in range(n_steps):
            rhs_base = (self.reduced_load(t[step + 1])
                        + self.M_r @ (a0c * q + a2c * qdot + a3c * qddot)
                        + self.D_r @ (a1c * q + a4c * qdot + a5c * qddot))

            def residual(qv, _rhs_base=rhs_base):
                return Keff @ qv - _rhs_base + self.f_nl(qv)

            q_trial = q.copy()
            G = residual(q_trial)
            ref = max(1.0, float(np.max(np.abs(rhs_base))))
            for _ in range(max_iter):
                if np.max(np.abs(G)) < tol * ref:
                    break
                J = a0c * self.M_r + a1c * self.D_r + self.reduced_tangent(q_trial)
                try:
                    dq = np.linalg.solve(J, G)
                except np.linalg.LinAlgError:
                    break
                step_scale = 1.0
                Gn = np.max(np.abs(G))
                q_next, G_next = q_trial, G
                for _ in range(max_backtrack):
                    cand = q_trial - step_scale * dq
                    G_cand = residual(cand)
                    if np.max(np.abs(G_cand)) < Gn or step_scale < 1e-6:
                        q_next, G_next = cand, G_cand
                        break
                    step_scale *= 0.5
                q_trial, G = q_next, G_next

            qddot_new = a0c * (q_trial - q) - a2c * qdot - a3c * qddot
            qdot_new = qdot + a6c * qddot + a7c * qddot_new
            q, qdot, qddot = q_trial, qdot_new, qddot_new
            q_hist[step + 1], qdot_hist[step + 1] = q, qdot

        return t, q_hist, qdot_hist

    # -----------------------------------------------------------------
    # Diagnostics (used directly by this module's own SPD test, and by
    # item 145's POD-RM diagnostics on top of M_r/D_r/K_r).
    # -----------------------------------------------------------------
    def energy(self, q, qdot, strain_energy_fn=None):
        """Total reduced mechanical energy at one instant:
        `0.5*qdot^T M_r qdot + U(q)`, where `U(q)` (the reduced strain
        energy) is supplied by the caller via `strain_energy_fn(q)` --
        this class has no way to recover a scalar potential from an
        arbitrary `internal_force_fn` in general (only guaranteed to
        exist/be single-valued for a genuinely conservative/hyperelastic
        full-order model), so it is never assumed here, only accepted
        as an optional argument. Returns just the kinetic term if
        `strain_energy_fn` is None."""
        q = _as_array(q); qdot = _as_array(qdot)
        ke = 0.5 * float(qdot @ (self.M_r @ qdot))
        if strain_energy_fn is None:
            return ke
        return ke + float(strain_energy_fn(q))

    # -----------------------------------------------------------------
    # Torch backend (additive addendum to Wave 17 item 144 -- see module
    # docstring "TORCH BACKEND" section for the full rationale/scope).
    # Every method above this point is plain NumPy and untouched;
    # everything below is opt-in, gated by _require_torch().
    # -----------------------------------------------------------------
    def torch_reduced_matrices(self, device="cpu", dtype=None):
        """Independent torch re-derivation of M_r/D_r/K_r -- genuine
        `V_t.T @ A_t @ V_t` torch matmuls on `self.M`/`self.C`/
        `self._K0_full`, NOT `torch.as_tensor(self.M_r)` (which would
        only be a dtype cast of the already-trusted NumPy result, not
        an independent cross-check) -- see module docstring. Kept FULL
        / non-diagonal exactly like the NumPy `M_r`/`D_r`/`K_r`: this is
        the identical projection, just executed with torch ops.

        Parameters
        ----------
        device : str
            "cpu" or "cuda" -- same parameter name/convention as
            fea_engine.torch_sparse_solver.py's own device= parameter.
        dtype : torch.dtype, optional
            Defaults to torch.float64 (matching torch_linalg.py's own
            default).

        Returns
        -------
        (M_r_t, D_r_t, K_r_t) : each a (n_modes, n_modes) torch.Tensor
            on `device`.
        """
        _require_torch()
        dtype = dtype or torch.float64
        V_t = torch.as_tensor(self.V, dtype=dtype, device=device)
        M_t = torch.as_tensor(self.M, dtype=dtype, device=device)
        C_t = torch.as_tensor(self.C, dtype=dtype, device=device)
        K0_t = torch.as_tensor(self._K0_full, dtype=dtype, device=device)
        M_r_t = V_t.T @ M_t @ V_t
        D_r_t = V_t.T @ C_t @ V_t
        K_r_t = V_t.T @ K0_t @ V_t
        return M_r_t, D_r_t, K_r_t

    def integrate_rk4_torch(self, q0, qdot0, dt, n_steps, device="cpu", dtype=None,
                             internal_force_fn_torch=None, load_fn_torch=None):
        """Torch-native mirror of integrate_rk4() -- IDENTICAL step
        logic, built on the SAME `parameterized_latent_ode.rk4_step()`
        (reused verbatim, not reimplemented: that function only ever
        does `+`, `*`, and calls `func` -- no NumPy-specific op anywhere
        in it -- so it already works unchanged on torch.Tensor state).

        `internal_force_fn_torch`/`load_fn_torch` are OPTIONAL
        torch-native overrides of this instance's own plain-NumPy
        `internal_force_fn`/`load_fn`: if omitted, every RK4 stage falls
        back to a NumPy round-trip through `self.reduced_internal_force()`/
        `self.reduced_load()` (correct, but no GPU-residency benefit).
        Passing a torch-native `internal_force_fn_torch(u_full_t) ->
        f_full_t` (called directly on `V_t @ q`, no NumPy round-trip) is
        the actual point of this method -- see module docstring's "TORCH
        BACKEND" section for why this interface is designed to accept a
        future Beam2DReissner torch-autograd internal-force/tangent
        callback directly, even though that callback does not exist yet
        as of this writing (checked directly: fea_engine/autograd_tangent.py
        covers Tet4NeoHookean/Tet10SolidTL/Shell4MITCCorotational only).

        Parameters
        ----------
        q0, qdot0 : (n_modes,) array_like
        dt : float
        n_steps : int
        device : str
            "cpu" or "cuda".
        dtype : torch.dtype, optional
            Defaults to torch.float64.
        internal_force_fn_torch : callable(u_full_t) -> f_full_t, optional
        load_fn_torch : callable(t) -> F_full_t, optional

        Returns
        -------
        t : (n_steps+1,) ndarray (plain NumPy -- only the STATE stays
            torch-native, per this method's own GPU-residency point; the
            time vector is cheap and every caller needs it as plain
            floats regardless).
        q_hist, qdot_hist : (n_steps+1, n_modes) torch.Tensor, on
            `device` -- NOT converted back to NumPy (unlike every other
            method on this class): a caller who wants a NumPy array
            calls `.detach().cpu().numpy()` themselves, the same
            boundary convention torch_linalg.py's own torch_* functions
            use.
        """
        _require_torch()
        dtype = dtype or torch.float64
        n = self.n_modes

        V_t = torch.as_tensor(self.V, dtype=dtype, device=device)
        M_r_t, D_r_t, _K_r_t = self.torch_reduced_matrices(device=device, dtype=dtype)

        def rhs(t_val, z):
            q, qdot = z[:n], z[n:]
            if internal_force_fn_torch is not None:
                u_full_t = V_t @ q
                f_full_t = internal_force_fn_torch(u_full_t)
                f_int_r = V_t.T @ f_full_t
            else:
                q_np = q.detach().cpu().numpy()
                f_int_r = torch.as_tensor(
                    self.reduced_internal_force(q_np), dtype=dtype, device=device)
            if load_fn_torch is not None:
                F_ext_r = load_fn_torch(t_val)
            else:
                F_ext_r = torch.as_tensor(
                    self.reduced_load(float(t_val)), dtype=dtype, device=device)
            rhs_force = F_ext_r - D_r_t @ qdot - f_int_r
            qddot = torch.linalg.solve(M_r_t, rhs_force)
            return torch.cat([qdot, qddot])

        z = torch.cat([
            torch.as_tensor(_as_array(q0), dtype=dtype, device=device),
            torch.as_tensor(_as_array(qdot0), dtype=dtype, device=device),
        ])
        t = np.arange(n_steps + 1) * dt
        z_hist = torch.zeros((n_steps + 1, 2 * n), dtype=dtype, device=device)
        z_hist[0] = z
        for i in range(n_steps):
            z = rk4_step(rhs, float(t[i]), dt, z)
            z_hist[i + 1] = z
        return t, z_hist[:, :n], z_hist[:, n:]

    def integrate_newton_newmark_torch(self, q0, qdot0, dt, n_steps,
                                        beta=0.25, gamma=0.5,
                                        tol=1e-9, max_iter=30, max_backtrack=30,
                                        device="cpu", dtype=None,
                                        internal_force_fn_torch=None,
                                        tangent_fn_torch=None, load_fn_torch=None):
        """Torch-native mirror of integrate_newton_newmark() -- IDENTICAL
        a0c..a7c/Keff/G(q)/J(q) algebra (see that method's own docstring
        for the full derivation, not re-derived here -- this is a
        mechanical port: np.linalg.solve -> torch.linalg.solve,
        np.max(np.abs(.)) -> torch.max(torch.abs(.)), same
        Newton-with-backtracking loop, same convergence/backtracking
        constants and control flow).

        Same optional `internal_force_fn_torch`/`tangent_fn_torch`/
        `load_fn_torch` torch-native-override pattern as
        integrate_rk4_torch() -- see that method's own docstring for the
        full reasoning (round-trips through NumPy each Newton iteration
        when omitted; a future torch-native tangent_fn avoids that).

        Returns
        -------
        t : (n_steps+1,) ndarray (plain NumPy, same convention as
            integrate_rk4_torch()).
        q_hist, qdot_hist : (n_steps+1, n_modes) torch.Tensor on
            `device`.
        """
        _require_torch()
        dtype = dtype or torch.float64
        n = self.n_modes

        V_t = torch.as_tensor(self.V, dtype=dtype, device=device)
        M_r_t, D_r_t, K_r_t = self.torch_reduced_matrices(device=device, dtype=dtype)

        a0c = 1.0 / (beta * dt ** 2); a1c = gamma / (beta * dt); a2c = 1.0 / (beta * dt)
        a3c = 1.0 / (2 * beta) - 1.0; a4c = gamma / beta - 1.0
        a5c = dt / 2.0 * (gamma / beta - 2.0)
        a6c = dt * (1.0 - gamma); a7c = dt * gamma

        Keff = K_r_t + a0c * M_r_t + a1c * D_r_t

        def f_int_r_torch(q):
            if internal_force_fn_torch is not None:
                u_full_t = V_t @ q
                f_full_t = internal_force_fn_torch(u_full_t)
                return V_t.T @ f_full_t
            q_np = q.detach().cpu().numpy()
            return torch.as_tensor(self.reduced_internal_force(q_np), dtype=dtype, device=device)

        def f_nl_torch(q):
            return f_int_r_torch(q) - K_r_t @ q

        def load_r_torch(t_val):
            if load_fn_torch is not None:
                return load_fn_torch(t_val)
            return torch.as_tensor(self.reduced_load(float(t_val)), dtype=dtype, device=device)

        def tangent_r_torch(q):
            if tangent_fn_torch is not None:
                u_full_t = V_t @ q
                K_full_t = tangent_fn_torch(u_full_t)
                return V_t.T @ K_full_t @ V_t
            q_np = q.detach().cpu().numpy()
            return torch.as_tensor(self.reduced_tangent(q_np), dtype=dtype, device=device)

        q = torch.as_tensor(_as_array(q0), dtype=dtype, device=device).clone()
        qdot = torch.as_tensor(_as_array(qdot0), dtype=dtype, device=device).clone()
        f0 = f_int_r_torch(q)
        qddot = torch.linalg.solve(M_r_t, load_r_torch(0.0) - D_r_t @ qdot - f0)

        t = np.arange(n_steps + 1) * dt
        q_hist = torch.zeros((n_steps + 1, n), dtype=dtype, device=device)
        qdot_hist = torch.zeros((n_steps + 1, n), dtype=dtype, device=device)
        q_hist[0], qdot_hist[0] = q, qdot

        for step in range(n_steps):
            rhs_base = (load_r_torch(t[step + 1])
                        + M_r_t @ (a0c * q + a2c * qdot + a3c * qddot)
                        + D_r_t @ (a1c * q + a4c * qdot + a5c * qddot))

            def residual(qv, _rhs_base=rhs_base):
                return Keff @ qv - _rhs_base + f_nl_torch(qv)

            q_trial = q.clone()
            G = residual(q_trial)
            ref = max(1.0, float(torch.max(torch.abs(rhs_base))))
            for _ in range(max_iter):
                if float(torch.max(torch.abs(G))) < tol * ref:
                    break
                J = a0c * M_r_t + a1c * D_r_t + tangent_r_torch(q_trial)
                try:
                    dq = torch.linalg.solve(J, G)
                except Exception:
                    break
                step_scale = 1.0
                Gn = float(torch.max(torch.abs(G)))
                q_next, G_next = q_trial, G
                for _ in range(max_backtrack):
                    cand = q_trial - step_scale * dq
                    G_cand = residual(cand)
                    if float(torch.max(torch.abs(G_cand))) < Gn or step_scale < 1e-6:
                        q_next, G_next = cand, G_cand
                        break
                    step_scale *= 0.5
                q_trial, G = q_next, G_next

            qddot_new = a0c * (q_trial - q) - a2c * qdot - a3c * qddot
            qdot_new = qdot + a6c * qddot + a7c * qddot_new
            q, qdot, qddot = q_trial, qdot_new, qddot_new
            q_hist[step + 1], qdot_hist[step + 1] = q, qdot

        return t, q_hist, qdot_hist


# =========================================================================
# Wave 17 item 145 -- POD-RM modal diagnostics (fea_engine/docs/
# consolidated_future_roadmap.md, same module as item 144 per its own
# roadmap row). Georgiou (2005) Eqs. 41-44, 50-60.
# =========================================================================
class MatrixSPDReport(NamedTuple):
    """Symmetry / positive-definiteness report for one reduced matrix,
    reusing exactly the eigenvalue-margin style item 144's own test
    suite already built (`TestReducedMatricesSPD` in
    `tests/test_intrusive_nonlinear_rom.py`): symmetry via max absolute
    asymmetry against a scale-relative tolerance, positive-definiteness
    via `eigvals.min() / eigvals.max()` on the SYMMETRIC PART (`eigh`
    requires an exactly symmetric input; using `0.5*(A+A.T)` is the
    same margin item 144's test computes when `A` is already symmetric
    to floating-point precision, and degrades gracefully -- rather than
    raising -- when it is not, since a damping matrix `D_r` is not
    guaranteed symmetric in general)."""
    name: str
    symmetric: bool
    asymmetry: float
    eigval_min: float
    eigval_max: float
    margin: float
    positive_definite: bool


def _spd_report(name, A, sym_tol=1e-9, pd_margin_tol=1e-8):
    A = _as_array(A)
    scale = max(float(np.max(np.abs(A))), 1.0)
    asymmetry = float(np.max(np.abs(A - A.T)))
    symmetric = asymmetry < sym_tol * scale
    eigvals = eigh(0.5 * (A + A.T), eigvals_only=True)
    eigval_min, eigval_max = float(eigvals.min()), float(eigvals.max())
    margin = eigval_min / eigval_max if eigval_max != 0.0 else float("nan")
    positive_definite = eigval_min > pd_margin_tol * eigval_max
    return MatrixSPDReport(name, symmetric, asymmetry, eigval_min, eigval_max,
                            margin, positive_definite)


class PodRMAnalysis(NamedTuple):
    """Result of `pod_rm_analysis()` -- see that function's docstring
    for the meaning of every field."""
    freq_sorted: np.ndarray
    freq_pod_order: np.ndarray
    E_hat_sorted: np.ndarray
    E_hat_pod_order: np.ndarray
    pod_order_index: np.ndarray
    diag_E_hat: np.ndarray
    uncoupled_freq: np.ndarray
    spd_report: dict


def _sign_align_columns(E):
    """Sign convention: for each column, force the LARGEST-MAGNITUDE
    entry positive -- the same convention item 143's `MultiFieldPOD`
    already established (its own docstring: "for each retained mode,
    ... find the entry with the LARGEST MAGNITUDE ..., and flip the
    whole column's sign ... if that entry is negative"), reused here
    verbatim (not a reference field, since Ê has no field structure --
    just the whole column) for consistency across this wave's modules."""
    E = E.copy()
    for m in range(E.shape[1]):
        col = E[:, m]
        k = int(np.argmax(np.abs(col)))
        if col[k] < 0:
            E[:, m] = -col
    return E


def pod_rm_analysis(M_r, D_r, K_r, sym_tol=1e-9, pd_margin_tol=1e-8):
    """Wave 17 item 145: POD-RM modal diagnostics from a fitted (or
    plain caller-supplied) reduced mass/damping/stiffness triple
    `(M_r, D_r, K_r)` -- Georgiou (2005) Eqs. 41-44, 50-60.

    Deliberately accepts PLAIN ARRAYS rather than an
    `IntrusiveNonlinearROM` instance (even though item 144 already
    exposes `.M_r`/`.D_r`/`.K_r` on every instance): this keeps the
    function independently testable on synthetic matrices (see this
    item's own validation, below, which never needs to construct a
    full FE-backed `IntrusiveNonlinearROM`) and reusable from any other
    source of a reduced triple, per this item's own roadmap note
    ("prefer accepting plain arrays if that keeps this function more
    reusable/testable in isolation"). A caller with a fitted
    `IntrusiveNonlinearROM` just calls
    `pod_rm_analysis(rom.M_r, rom.D_r, rom.K_r)`.

    THE GENERALIZED EIGENPROBLEM. `scipy.linalg.eigh(K_r, M_r)` solves
    `K_r @ v = lambda * M_r @ v` for a symmetric `K_r` and a symmetric
    POSITIVE DEFINITE `M_r` -- correct for a genuinely non-diagonal,
    non-identity `M_r` (this is exactly why item 144 keeps `M_r`
    un-normalized: `eigh`'s generalized form handles the full mass
    matrix directly, via an internal Cholesky-based congruence
    transform, with no approximation). `eigvals` are returned ASCENDING
    by `eigh` itself; `freq_sorted = sqrt(eigvals)` is therefore already
    frequency-sorted with no extra sort needed (tiny negative
    eigenvalues from floating-point roundoff on a numerically
    positive-semi-definite pencil are clipped to 0 before the sqrt).

    EIGENVECTOR MATRIX E_HAT. Each returned eigenvector column is first
    Euclidean-UNIT-normalized (`eigh`'s own generalized-eigenvector
    convention is `v^T @ M_r @ v = 1`, i.e. MASS-normalized, not unit
    Euclidean length in general -- re-normalizing to Euclidean unit
    length is what makes a perfectly diagonal `(M_r, K_r)` give
    `E_hat = I` EXACTLY, checked below, since only then does each
    eigenvector coincide exactly with a standard basis vector), then
    sign-aligned via `_sign_align_columns()` (positive largest-magnitude
    entry per column, matching item 143's `MultiFieldPOD` convention).

    TWO ORDERINGS, BOTH RETURNED (this item's own roadmap requirement --
    the paper lists RM frequencies in POD-mode order in some places,
    e.g. Eq. 53, and by inspection/sorted order elsewhere):

      - `freq_sorted` / `E_hat_sorted` : `eigh`'s own ascending-eigenvalue
        order, verbatim.
      - `freq_pod_order` / `E_hat_pod_order` : reassigned so that slot
        `m` holds the eigenpair whose eigenvector has the LARGEST overall
        alignment with original POD coordinate `m` -- a genuine one-to-one
        assignment (`scipy.optimize.linear_sum_assignment` on the cost
        matrix `-abs(E_hat_sorted)`, so no two POD coordinates are ever
        assigned the same eigenpair, unlike a naive per-row argmax which
        can collide), not merely "sorted-order relabeled". This is the
        ordering `diag_E_hat` (the near-identity diagnostic) and the
        RM-vs-uncoupled frequency comparison below are meaningful in:
        Eq. 55's uncoupled frequency for POD mode `m` is only a sensible
        contrast against the COUPLED RM frequency that mode `m` actually
        maps to, which need not be the `m`-th smallest eigenvalue.

    NEAR-IDENTITY DIAGNOSTIC. `diag_E_hat = diag(E_hat_pod_order)` --
    the paper's own direct check (Eqs. 51, 54, 60) of whether the POD
    modes are already close to the RM's natural coordinates. Exposed as
    its own field, not buried in a larger structure, since the paper
    checks it directly and finds it near 1 in some cases (weakly
    coupled RM) and clearly not in others (Eqs. 57, 59) -- both outcomes
    are just whatever this array numerically is, no special-casing.

    UNCOUPLED 1-DOF FREQUENCIES (Eq. 55). `sqrt(K_r[m,m] / M_r[m,m])`
    per ORIGINAL POD coordinate `m` (diagonal entries of the INPUT
    `K_r`/`M_r`, untouched by either eigenvector ordering above) -- what
    mode `m`'s frequency would be if the reduced system were (incorrectly)
    treated as diagonal/uncoupled. A diagnostic CONTRAST against the real
    coupled `freq_pod_order`, not a replacement for it.

    SYMMETRY / POSITIVE-DEFINITENESS REPORT. `spd_report` is a dict
    keyed `"M_r"`, `"D_r"`, `"K_r"`, each a `MatrixSPDReport` built by
    `_spd_report()` (see its own docstring) -- the identical
    eigenvalue-margin style item 144's own `TestReducedMatricesSPD`
    test already established, reused rather than reinvented.

    Parameters
    ----------
    M_r, D_r, K_r : (n_modes, n_modes) array_like
        The reduced mass / damping / stiffness matrices (e.g. taken
        directly off a fitted `IntrusiveNonlinearROM`, or any other
        source).
    sym_tol : float
        Relative tolerance (against `max(|A|), 1)`) for the symmetry
        check in `spd_report`.
    pd_margin_tol : float
        `eigval_min/eigval_max` threshold for the positive-definiteness
        flag in `spd_report`.

    Returns
    -------
    PodRMAnalysis
        A `NamedTuple` with fields `freq_sorted`, `freq_pod_order`,
        `E_hat_sorted`, `E_hat_pod_order`, `pod_order_index` (the
        `linear_sum_assignment` permutation itself, for callers who
        want to trace an eigenpair back to its sorted-order index),
        `diag_E_hat`, `uncoupled_freq`, `spd_report`.
    """
    M_r = _as_array(M_r)
    D_r = _as_array(D_r)
    K_r = _as_array(K_r)
    n = M_r.shape[0]

    eigvals, eigvecs = eigh(K_r, M_r)
    eigvals = np.clip(eigvals, 0.0, None)
    freq_sorted = np.sqrt(eigvals)

    # Euclidean-unit-normalize (eigh's own generalized-eigenvector
    # convention is M_r-normalized, not unit length -- see docstring).
    norms = np.linalg.norm(eigvecs, axis=0)
    E_hat_sorted = _sign_align_columns(eigvecs / norms[None, :])

    # POD-mode-order reassignment via optimal one-to-one matching of
    # original coordinates (rows) to eigenpairs (columns), maximizing
    # total |alignment| -- see docstring.
    cost = -np.abs(E_hat_sorted)
    row_index, col_index = linear_sum_assignment(cost)
    pod_order_index = np.empty(n, dtype=int)
    pod_order_index[row_index] = col_index

    freq_pod_order = freq_sorted[pod_order_index]
    E_hat_pod_order = E_hat_sorted[:, pod_order_index]
    diag_E_hat = np.diag(E_hat_pod_order)

    uncoupled_freq = np.sqrt(np.diag(K_r) / np.diag(M_r))

    spd_report = {
        "M_r": _spd_report("M_r", M_r, sym_tol, pd_margin_tol),
        "D_r": _spd_report("D_r", D_r, sym_tol, pd_margin_tol),
        "K_r": _spd_report("K_r", K_r, sym_tol, pd_margin_tol),
    }

    return PodRMAnalysis(
        freq_sorted=freq_sorted,
        freq_pod_order=freq_pod_order,
        E_hat_sorted=E_hat_sorted,
        E_hat_pod_order=E_hat_pod_order,
        pod_order_index=pod_order_index,
        diag_E_hat=diag_E_hat,
        uncoupled_freq=uncoupled_freq,
        spd_report=spd_report,
    )
