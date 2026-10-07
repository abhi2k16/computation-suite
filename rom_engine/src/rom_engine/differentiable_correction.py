"""
differentiable_correction.py -- Wave 10 item 103 (fea_engine/docs/
consolidated_future_roadmap.md): a DESIGN NOTE + prototype, not a full
reimplementation, per that item's own explicit scoping. Applies
fea_engine.differentiable's item-100 explicit residual-minimization
calibration pattern to rom_engine's OWN reduced equilibrium equation,
instead of a full finite-element residual.

Read docs/differentiable_rom_correction_design.md (this package's own
docs folder) first -- it works out the actual interface boundary
between fea_engine's new Wave 10 machinery and this package's own
nonlinear_rom.py training pipeline, and states plainly what is and
isn't built here.

WHY A SEPARATE, INDEPENDENT MODULE (not importing fea_engine.
differentiable directly): rom_engine's own stated design principle
(rom_engine/__init__.py) is that fea_engine is a TEST/EXAMPLE
dependency only, never a LIBRARY dependency -- this package's actual
source code must import and run with zero fea_engine present. So this
module reimplements the SAME "R(w)+f_theta(w)=0, calibrate against a
known reference state" pattern independently, at the reduced-
coordinate level -- exactly the same relationship NeuralSurrogate's
own docstring already established with fea_engine.autograd_tangent
(independently re-implemented there, not imported).

THE REDUCED RESIDUAL this module works against -- PolynomialModalROM's
own equilibrium condition (nonlinear_rom.py): Lambda*q + F_nl(q) =
F_ext, i.e. R(q) = F_ext - Lambda*q - F_nl(q) = 0. A trainable
correction f_theta(q) is inserted the SAME way fea_engine.
differentiable.AdditiveCorrection is (see that module's own "SIGN
CONVENTION" section, transplanted here unchanged):

    F_ext - Lambda*q - F_nl(q) - f_theta(q) = 0

WHEN THIS IS USEFUL (the actual "is this paper relevant to the ROM
projects" answer this item exists to give): today, MultiFidelity
Surrogate/PolynomialModalROM are fit via a SEPARATE offline regression
step against precomputed (q, F_nl(q)) pairs a TrainingStrategy
generates -- this needs CLEAN, already-tabulated force/displacement
data. calibrate_reduced_correction_explicit() below instead calibrates
a correction against a KNOWN REFERENCE EQUILIBRIUM q_m under a KNOWN
(F_ext, Lambda) -- useful precisely when F_nl(q) itself isn't directly
observable/tabulated but an equilibrium q_m is (e.g. a high-fidelity
FE snapshot's own modal projection, or any case where "does this
reduced model, plus a trainable correction, reproduce a known
equilibrium" is the calibration target rather than a raw force-vs-
displacement table).

NOT BUILT HERE (see the design doc's own "interface boundary, not yet
built" section for the full reasoning): the ROM-level counterpart of
fea_engine item 98's implicit-adjoint layer (needed to calibrate a
correction against PARTIAL/indirect reference data -- e.g. a few
sparse sensor dofs of q, not a full q_m -- by differentiating through
an UNCONVERGED-at-calibration-time PolynomialModalROM.predict(F_ext=...,
Lambda=...) Newton solve). This item is explicitly scoped by the
roadmap as "a DESIGN NOTE + prototype ... scoping the actual interface
boundary ... is the first concrete step," not a full port of item 98.
"""
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    # See rom_engine.nonlinear_rom.NeuralSurrogate's own comment on
    # this exact except clause (a CUDA-linked wheel can install but
    # fail to import with OSError/ValueError on a machine with no CUDA
    # runtime) -- the same reasoning applies here.
    _HAS_TORCH = False


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "differentiable_correction.py's torch-dependent functionality "
            "requires the 'torch' package -- reduced_residual() itself does "
            "NOT need torch (pure NumPy); only calibrate_reduced_correction_"
            "explicit() and ScalarModalCorrection do.")


def reduced_residual(q, Lambda, F_nl_fn, F_ext, correction=None):
    """R(q) = F_ext - Lambda*q - F_nl_fn(q) - f_theta(q). Pure NumPy --
    no torch needed, usable and tested unconditionally.

    q : (n_modes,) array-like, the reduced-coordinate state to
        evaluate the residual at.
    Lambda : (n_modes,) array-like, the linear modal stiffness
        (PolynomialModalROM.predict()'s own Lambda convention).
    F_nl_fn : callable, (n_modes,) -> (n_modes,) -- e.g. a fitted
        `nonlinear_rom.PolynomialModalROM.force`, or any plain
        closed-form function (this module's own tests use the latter).
    F_ext : (n_modes,) array-like, external modal force.
    correction : optional, any object exposing `.value(q) ->
        (n_modes,) numpy array` -- the SAME protocol fea_engine.
        differentiable.AdditiveCorrection's `value()` method uses
        (independently satisfied here, not imported -- see this
        module's own docstring)."""
    q = np.asarray(q, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    F_nl = np.asarray(F_nl_fn(q), dtype=float)
    f_corr = correction.value(q) if correction is not None else 0.0
    return np.asarray(F_ext, dtype=float) - Lambda * q - F_nl - f_corr


class ScalarModalCorrection:
    """Simplest trainable correction at reduced-coordinate level: one
    free scalar per mode, f_theta(q) = theta (constant, independent of
    q). Mirrors fea_engine.differentiable.ScalarFieldCorrection's own
    role, independently implemented here per this module's own
    docstring (no fea_engine import)."""

    def __init__(self, n_modes, init=0.0, dtype=None):
        _require_torch()
        self.n_modes = n_modes
        dtype = dtype or torch.float64
        self.theta = torch.full((n_modes,), float(init), dtype=dtype, requires_grad=True)

    def value(self, q):
        return self.theta.detach().numpy().copy()

    def torch_value(self, q_t):
        return self.theta

    def parameters(self):
        return [self.theta]


def calibrate_reduced_correction_explicit(q_reference, Lambda, F_nl_fn, F_ext, correction,
                                           n_epochs=500, lr=1e-2, optimizer_cls=None, verbose=False):
    """ROM-level counterpart of fea_engine.differentiable.calibrate_
    correction_explicit() -- minimizes

        ||F_ext - Lambda*q_m - F_nl_fn(q_m) - f_theta(q_m)||^2

    over correction.parameters(), at a KNOWN reference equilibrium q_m
    (q_reference). No Newton solve needed: F_nl_fn(q_m) is evaluated
    ONCE and held as a constant torch tensor, exactly like fea_engine's
    F_int(w_m) in the analogous fea_engine-level function -- only
    f_theta(q_m) is differentiated w.r.t. theta, every optimizer step.

    correction: any object exposing torch_value(q_t)/parameters() (see
    ScalarModalCorrection above for the minimal example).

    Returns a (n_epochs,) numpy array of the loss history, the same
    convention every other training loop in this package uses
    (NeuralSurrogate.fit()'s loss_history, fea_engine.differentiable.
    calibrate_correction_explicit()'s own return value)."""
    _require_torch()
    q_reference = np.asarray(q_reference, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    F_nl = np.asarray(F_nl_fn(q_reference), dtype=float)
    R_const = np.asarray(F_ext, dtype=float) - Lambda * q_reference - F_nl
    R_const_t = torch.tensor(R_const, dtype=torch.float64)
    q_t = torch.tensor(q_reference, dtype=torch.float64)

    params = list(correction.parameters())
    optimizer = (optimizer_cls or torch.optim.Adam)(params, lr=lr)
    history = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        f_theta_t = correction.torch_value(q_t)
        residual_t = R_const_t - f_theta_t
        loss = torch.sum(residual_t ** 2)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        if verbose and epoch % max(1, n_epochs // 10) == 0:
            print(f"  calibrate_reduced_correction_explicit: epoch {epoch}/{n_epochs} "
                  f"loss={loss.item():.6e}")
    return np.array(history)
