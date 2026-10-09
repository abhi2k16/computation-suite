# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
backend_dispatch.py -- Wave 11 item 108 (docs/consolidated_future_
roadmap.md, source tensormesh_analysis_report.md Sec 5 "Sparse
Solvers"): a unified `backend="auto"` dispatcher for
`FESystem.solve_static()`, plus the `verbose=True` one-line diagnostic
TensorMesh's own torch-sla layer prints (`[torch-sla] solve: n=...,
nnz=..., dtype=..., device=..., symmetric=..., spd=..., backend=...,
method=...`) -- already flagged as "worth copying" in
tensormesh_comparative_analysis.md Sec.6.1, never actually built until
now.

Integration policy: same as every other Wave 11 module -- no
`tensormesh`/`torch-sla` runtime dependency, native reuse of this
package's own existing pieces (Wave 0 item 5's SPD-aware Cholesky path,
Wave 0 item 3's `backend="torch"` dispatch) behind one new
`backend="auto"` choice on `FESystem`.

Why "auto" is decided at SOLVE time, not construction time
------------------------------------------------------------
Unlike `backend="torch"` (which fails fast at `FESystem.__init__` if
torch isn't importable, see solver.py's own docstring), `backend="auto"`
does NOT require torch at construction -- whether torch actually ends up
being used depends on `Kff`'s own size/SPD-ness, which isn't known until
`solve_static()` has assembled the system and extracted the free-dof
block. `select_backend()` below is called from inside `solve_static()`
for exactly this reason; only if it picks "torch" does the (then
appropriate) `_require_torch()` fail-fast fire.

The size/SPD heuristic
------------------------
TensorMesh's own backend-selection table (tensormesh_analysis_report.md
Sec.5, approximate, DOF-count-based) is the design reference for the
threshold below -- not imported, just mirrored in spirit: a small system
(a few thousand DOF) is cheap enough that a direct SciPy solve wins on
latency alone (no factorization-amortization or GPU-transfer overhead
to justify); a larger, genuinely SPD system is where an iterative/GPU
path can plausibly pay off. Two hard constraints, not heuristics:
  - A non-SPD (or non-symmetric) `Kff` NEVER gets routed to CG --
    `torch_sparse_cg_solve()` (torch_sparse_solver.py) is documented
    SPD-only, so "auto" falls back to the existing SciPy direct/eigen
    chain (Wave 0 items 5/6) unconditionally whenever the SPD probe
    fails, regardless of size.
  - `backend="auto"` therefore can NEVER silently produce a wrong
    answer by using CG where it doesn't apply -- worst case it simply
    doesn't get the GPU/CG speedup and uses the same safe SciPy path
    `backend="scipy"` would have used anyway.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

#: Free-dof count above which "auto" prefers the torch/CG path, IF the
#: system also probes SPD -- a documented rule of thumb (see module
#: docstring), not derived from a benchmark on any particular machine.
#: A caller who knows their own crossover point should just pass an
#: explicit backend="scipy"/"torch" instead of relying on this default.
AUTO_DOF_THRESHOLD = 5000


def _is_symmetric(Kff, tol=1e-8):
    if hasattr(Kff, "toarray"):
        diff = Kff - Kff.T
        if diff.nnz == 0:
            return True
        scale = max(1.0, float(np.abs(Kff).max()))
        return float(np.abs(diff).max()) <= tol * scale
    Kff = np.asarray(Kff)
    scale = max(1.0, float(np.abs(Kff).max()))
    return bool(np.all(np.abs(Kff - Kff.T) <= tol * scale))


def _is_spd(Kff):
    """Attempts a Cholesky factorization -- the SAME cheap SPD test
    `FESystem._dense_spd_solve()` (Wave 0 item 5) already relies on,
    reused here purely as a detection probe (the factorization itself
    is discarded, not reused as an actual solve -- keeping this
    function a pure yes/no check with no side effect on which method
    ultimately performs the real solve)."""
    A = Kff.toarray() if hasattr(Kff, "toarray") else np.asarray(Kff)
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def select_backend(Kff, device="cpu", dof_threshold=AUTO_DOF_THRESHOLD):
    """Decide "scipy" or "torch" for this specific `Kff`, and return a
    diagnostics dict describing why -- the actual content of the
    `verbose=True` one-liner `format_diagnostic_line()` prints.

    Parameters
    ----------
    Kff : ndarray or scipy.sparse matrix
        The free-dof stiffness block about to be solved.
    device : str
        Informational only here (which device a chosen "torch" backend
        would run on) -- this function does no torch import itself.
    dof_threshold : int
        See `AUTO_DOF_THRESHOLD` above.

    Returns
    -------
    (backend, diag) : (str, dict)
        `backend` is `"scipy"` or `"torch"`. `diag` has keys `n`,
        `nnz`, `dtype`, `device`, `symmetric`, `spd`, `backend`,
        `method`.
    """
    n = Kff.shape[0]
    nnz = int(Kff.nnz) if hasattr(Kff, "nnz") else int(np.count_nonzero(Kff))
    symmetric = _is_symmetric(Kff)
    spd = bool(symmetric and _is_spd(Kff))
    if spd and n > dof_threshold:
        backend, method = "torch", "cg"
    else:
        backend, method = "scipy", ("sparse_lu" if hasattr(Kff, "toarray") else "cholesky_or_eigen")
    dtype = str(getattr(Kff, "dtype", np.asarray(Kff).dtype))
    diag = {
        "n": n, "nnz": nnz, "dtype": dtype, "device": device,
        "symmetric": symmetric, "spd": spd, "backend": backend, "method": method,
    }
    return backend, diag


def format_diagnostic_line(diag):
    """The `verbose=True` one-line summary
    (`FESystem.solve_static(verbose=True)`), mirroring TensorMesh's own
    `[torch-sla] solve: n=..., nnz=..., dtype=..., device=...,
    symmetric=..., spd=..., backend=..., method=...` convention
    (tensormesh_comparative_analysis.md Sec.6.1)."""
    return (f"[fea_engine backend={diag['backend']}] solve: n={diag['n']}, "
            f"nnz={diag['nnz']}, dtype={diag['dtype']}, device={diag['device']}, "
            f"symmetric={diag['symmetric']}, spd={diag['spd']}, "
            f"backend={diag['backend']}, method={diag['method']}")
