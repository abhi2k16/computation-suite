"""
newton_options.py -- one shared options object for the nonlinear drivers.

Ten ``solve_*`` drivers in :mod:`fea_engine.nonlinear_solver` repeat the same
convergence keywords (``tol``, ``max_iter``, ``du_tol``, ``energy_tol``,
``line_search``, ``verbose``). :class:`NewtonOptions` bundles them so one object
can be reused across drivers::

    opts = NewtonOptions(tol=1e-9, max_iter=40, line_search=True)
    lf, U = solve_nonlinear_static(system, mat, n_steps=20, options=opts)
    lf, U = solve_nonlinear_arc_length(system, mat, delta_L=0.05, options=opts)

Rules (kept simple on purpose):

* Every field defaults to ``None`` meaning "use the driver's own default", because
  the drivers deliberately differ (``tol`` is 1e-8 for most, 1e-10 for the contact
  and displacement-control drivers).
* A non-``None`` field applies to every driver that has a keyword of that name and
  is ignored by drivers that do not (e.g. ``line_search`` for the Koiter-Newton
  tracer). Ignored fields are never an error, so one options object fits all.
* A keyword passed explicitly to the driver always wins over ``options``.
* Old calls without ``options=`` behave exactly as before.
"""
from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, fields, replace
from typing import Optional

__all__ = ["NewtonOptions", "accepts_options"]


@dataclass(frozen=True)
class NewtonOptions:
    """Shared Newton-iteration settings. ``None`` means "driver default".

    Parameters
    ----------
    tol : float, optional
        Residual tolerance.
    max_iter : int, optional
        Maximum Newton iterations per load/time step.
    du_tol : float, optional
        Displacement-increment tolerance (drivers that support it).
    energy_tol : float, optional
        Energy-norm tolerance (drivers that support it).
    line_search : bool, optional
        Enable the Armijo backtracking line search (drivers that support it).
    verbose : bool, optional
        Print per-iteration progress.
    """
    tol: Optional[float] = None
    max_iter: Optional[int] = None
    du_tol: Optional[float] = None
    energy_tol: Optional[float] = None
    line_search: Optional[bool] = None
    verbose: Optional[bool] = None

    def __post_init__(self):
        if self.tol is not None and not self.tol > 0:
            raise ValueError(f"NewtonOptions.tol must be > 0, got {self.tol!r}")
        if self.max_iter is not None and (int(self.max_iter) != self.max_iter or self.max_iter < 1):
            raise ValueError(f"NewtonOptions.max_iter must be a positive integer, got {self.max_iter!r}")
        for name in ("du_tol", "energy_tol"):
            v = getattr(self, name)
            if v is not None and not v > 0:
                raise ValueError(f"NewtonOptions.{name} must be > 0, got {v!r}")

    def updated(self, **changes) -> "NewtonOptions":
        """Return a copy with some fields changed (the object itself is frozen)."""
        return replace(self, **changes)

    def as_kwargs(self, accepted=None) -> dict:
        """Non-``None`` fields as a dict, optionally restricted to the names in ``accepted``."""
        out = {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}
        if accepted is not None:
            out = {k: v for k, v in out.items() if k in accepted}
        return out


def accepts_options(func):
    """Decorator: add a keyword-only ``options=None`` to a nonlinear driver.

    Fields of the :class:`NewtonOptions` fill in keywords the caller did not pass;
    explicit keywords always win. The wrapped function keeps its name, docstring
    and positional signature.
    """
    sig = inspect.signature(func)
    params = sig.parameters
    if "options" in params:                       # already native, nothing to do
        return func
    names = set(params)

    @functools.wraps(func)
    def wrapper(*args, options=None, **kwargs):
        if options is not None:
            if not isinstance(options, NewtonOptions):
                raise TypeError(f"options must be a NewtonOptions, got {type(options).__name__}")
            explicit = set(sig.bind_partial(*args, **{k: v for k, v in kwargs.items() if k in names}).arguments)
            for k, v in options.as_kwargs(accepted=names).items():
                if k not in explicit:
                    kwargs[k] = v
        return func(*args, **kwargs)

    # advertise the new keyword in help()/inspect without changing positional order
    new_params = list(params.values())
    opt = inspect.Parameter("options", inspect.Parameter.KEYWORD_ONLY, default=None)
    var_kw = [i for i, p in enumerate(new_params) if p.kind is inspect.Parameter.VAR_KEYWORD]
    new_params.insert(var_kw[0] if var_kw else len(new_params), opt)
    wrapper.__signature__ = sig.replace(parameters=new_params)
    wrapper.__doc__ = (func.__doc__ or "") + (
        "\n\n    options : NewtonOptions, optional\n"
        "        Shared convergence settings (``tol``, ``max_iter``, ``du_tol``, ``energy_tol``,\n"
        "        ``line_search``, ``verbose``); explicit keywords override it, fields this driver\n"
        "        does not have are ignored. See :class:`fea_engine.newton_options.NewtonOptions`.\n")
    return wrapper
