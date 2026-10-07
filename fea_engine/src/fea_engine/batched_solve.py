"""
batched_solve.py -- Wave 11 item 109 (docs/consolidated_future_roadmap.md,
source tensormesh_analysis_report.md Sec 5 "Sparse Solvers", 'Batched
right-hand sides': "b of shape [n_dof, n_batch] amortizes one
factorization across many back-substitutions -- the workhorse of the
tensormesh.dataset ML workflow").

Factorize an assembled stiffness matrix ONCE and reuse that
factorization to solve against MANY load vectors -- exactly the shape
of a ROM/surrogate training-snapshot sweep (Wave 7's neural-surrogate
strategy, Wave 10 items 103/104's dataset-diagnostics/differentiable-
correction utilities, mfs-nlrom-beam's own static-pressure-sweep
dataset). Today, generating N snapshots via N independent
FESystem.solve_static() calls re-factorizes K from scratch every
single time even though only the load vector changes -- for a direct
solve this is genuinely wasted work (the factorization is the
expensive O(n^3) [dense] / O(n^1.x) [sparse, mesh-dependent] part;
each back-substitution against it is comparatively cheap).

Integration policy: no `tensormesh`/`torch-sla` dependency -- SciPy's
own `lu_factor`/`lu_solve` (dense) and `splu` (sparse), or a single
batched `torch.linalg.solve()` call, both already-standard multi-RHS
solve APIs this module just wires up consistently with this package's
own `backend=` convention (Wave 0 item 3).

Scope: unlike `FESystem.solve_static()`'s own SciPy path (Wave 0 items
5/6), `solve_batched()`'s scipy backend does NOT attempt an SPD-aware
Cholesky path or an automatic near-singular/rigid-body eigen fallback
-- it is a lean multi-RHS LU-factor-once utility for the common,
well-constrained case a training-data sweep already implies (the SAME
BCs held fixed across every sample, only the load changing), matching
the same "no automatic fallback" precedent `backend="torch"`'s own
`solve_static()` path already established (torch_sparse_solver.py's
own docstring). A caller with a genuinely singular/near-singular system
should keep using `FESystem.solve_static()`'s own per-call handling
instead.
"""
import numpy as np


def solve_batched(K, B, backend="scipy", device="cpu"):
    """Solve `K @ X[:, i] = B[:, i]` for every column `i` of `B`,
    factorizing `K` exactly ONCE regardless of how many columns `B`
    has.

    Parameters
    ----------
    K : (n, n) ndarray or scipy.sparse matrix
    B : (n,) or (n, n_batch) array-like
        A 1-D `B` is treated as `n_batch=1` and the result is returned
        as a plain `(n,)` vector (matching `np.linalg.solve`'s own
        1-D-in/1-D-out convention); a 2-D `B` returns a 2-D `X`.
    backend : "scipy" (default) or "torch"
        "scipy": `scipy.linalg.lu_factor`/`lu_solve` (dense `K`) or
        `scipy.sparse.linalg.splu` (sparse `K`) -- both natively accept
        a 2-D right-hand side and solve every column against the SAME
        factorization in one call, so there is no explicit Python loop
        over columns here at all.
        "torch": densifies `K` (if sparse) and calls `torch.linalg.
        solve()` ONCE with the full `(n, n_batch)` right-hand side --
        `torch.linalg.solve` itself batches the back-substitution
        internally; no per-column loop here either.
    device : str
        Passed through to torch when backend="torch"; ignored for
        "scipy" (matches `FESystem`'s own `device=` convention).

    Returns
    -------
    X : (n,) or (n, n_batch) ndarray, matching B's own dimensionality.
    """
    B = np.asarray(B, dtype=float)
    was_1d = B.ndim == 1
    if was_1d:
        B = B[:, None]

    if backend == "scipy":
        if hasattr(K, "toarray"):   # scipy.sparse
            from scipy.sparse.linalg import splu
            lu = splu(K.tocsc())
            X = lu.solve(B)
        else:
            from scipy.linalg import lu_factor, lu_solve
            lu_and_piv = lu_factor(np.asarray(K, dtype=float))
            X = lu_solve(lu_and_piv, B)
    elif backend == "torch":
        from .torch_sparse_solver import _require_torch
        _require_torch()
        import torch
        Kd = K.toarray() if hasattr(K, "toarray") else np.asarray(K, dtype=float)
        Kt = torch.as_tensor(Kd, dtype=torch.float64, device=device)
        Bt = torch.as_tensor(B, dtype=torch.float64, device=device)
        Xt = torch.linalg.solve(Kt, Bt)
        X = Xt.detach().cpu().numpy()
    else:
        raise ValueError(
            f"solve_batched: unknown backend={backend!r} -- expected "
            f"'scipy' (default) or 'torch'.")

    return X[:, 0] if was_1d else X


def solve_static_batched(fesystem, F_batch, backend=None):
    """`FESystem`-level convenience wrapper: solve `fesystem`'s already-
    assembled `K` against MANY external load vectors at once, applying
    the SAME fixed-dof boundary conditions (`fesystem.fixed_dofs`/
    `free_dofs`) to every column, factorizing `Kff` exactly once
    (item 109's own payoff) instead of calling `fesystem.solve_static()`
    once per column.

    Parameters
    ----------
    fesystem : FESystem
        Reads `fesystem.K` and `fesystem.free_dofs` (i.e. the ALREADY-
        assembled stiffness and boundary conditions -- exactly what
        `solve_static()` itself reads); `fesystem.F` is NOT used or
        modified -- the caller supplies every load vector explicitly
        via `F_batch` instead.
    F_batch : (n_dof, n_batch) ndarray
        One external load vector per column, in the SAME global dof
        numbering `fesystem.F` itself uses.
    backend : "scipy", "torch", or None
        None (default) reuses `fesystem.backend` UNLESS it is "auto"
        (item 109 does not implement its own auto-selection -- an
        "auto" FESystem falls back to "scipy" here, the always-safe
        choice; pass backend= explicitly to override).

    Non-homogeneous Dirichlet values (Wave 16 item 133, a follow-up
    closing a gap this function itself created relative to item 131:
    this function predates item 131 and, until now, silently treated
    EVERY fixed DOF as zero regardless of `fix_dofs(..., value=...)`
    -- a batched load-sweep on a mesh with a nonzero prescribed
    displacement got a silently wrong answer at every column. Now
    reads `fesystem.fixed_dof_values` exactly like `solve_static()`
    does: applies the SAME `-Kio*uo` RHS correction to every column
    (the prescribed values, unlike the load, are the SAME for every
    sample in the batch -- that is the whole premise of reusing one
    factorization across the batch) and fills `U_batch[fixed, :]` with
    `u_fixed` broadcast across every column. Zero cost when every
    fixed DOF's value is 0.0 (the pre-item-131, pre-item-133 case) --
    `_dirichlet_rhs_correction()` short-circuits to a zero vector
    immediately, same convention as `solve_static()` itself.

    Returns
    -------
    U_batch : (n_dof, n_batch) ndarray
    """
    F_batch = np.asarray(F_batch, dtype=float)
    if backend is None:
        backend = fesystem.backend if fesystem.backend in ("scipy", "torch") else "scipy"
    free = fesystem.free_dofs
    fixed = fesystem.fixed_dofs_array
    u_fixed = fesystem._fixed_dof_values_array(fixed)
    Kmat = fesystem._as_solve_matrix(fesystem.K)
    Kff = Kmat[np.ix_(free, free)]
    correction = fesystem._dirichlet_rhs_correction(Kmat, free, fixed, u_fixed)
    Ff_batch = F_batch[free, :] - correction[:, None]
    Uf_batch = solve_batched(Kff, Ff_batch, backend=backend, device=fesystem.device)
    U_batch = np.zeros((fesystem.n_dof, F_batch.shape[1]))
    U_batch[fixed, :] = u_fixed[:, None]
    U_batch[free, :] = Uf_batch
    return U_batch
