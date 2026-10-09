"""
iterative_solvers.py -- Wave 8 items 36-40 (docs/consolidated_future_
roadmap.md, source fem_implementation_lessons.md Chapters 10-13):
fill-reducing reordering, preconditioners, conjugate gradients,
classical stationary iterations, and geometric multigrid for
structured Quad4 grids.

Scope and relationship to the rest of the package, stated up front
(the same "narrow the scope, document the gap explicitly" convention
this project has used since Wave 5's Notch/Fillet items): every
solver in this module is a genuinely independent, native NumPy/
SciPy-sparse-graph implementation -- fill_in_count()/reverse_cuthill_
mckee() use only scipy.sparse's plain CSR/CSC data structures and
scipy.sparse.linalg.splu() as a fill-measurement tool (not as a
substitute for hand-written graph code), and every solver (CG,
Jacobi/Gauss-Seidel/SOR, multigrid) is hand-written rather than a
thin wrapper around scipy.sparse.linalg.cg()/etc. None of this
replaces solver.FESystem.solve_static()'s existing direct-solve path
(Wave 0 items 3/5/6's SPD-aware Cholesky / eigen-regularized / sparse
LU dispatch) -- that remains the default and is unaffected. This
module exists for the case fem_implementation_lessons.md's own
Chapter 10-13 summary calls out explicitly: "for large problems,
iterative solvers beat direct solvers because of fill-in" -- a
genuinely different regime from anything else in this package, meant
to be reached for explicitly (see tests/test_iterative_solvers.py and
tests/test_multigrid.py for worked measurements of exactly when/why
each piece here helps), not wired into FESystem automatically.

GPU/torch side-by-side path (Wave 9 addendum item 137, docs/
consolidated_future_roadmap.md)
-------------------------------------------------------------------
`preconditioned_cg(..., backend="scipy"/"torch", device=)`: the scipy
path (default) is completely unchanged. The torch path reuses
`torch_sparse_solver.py`'s own generic `_cg_torch()` loop (item 3's
Jacobi-preconditioned CG was ALREADY written against a generic
`matvec`/`precond_inv` callable pair specifically so a second caller
could reuse it without touching that loop again) -- this item is that
second caller. `M=None` (unpreconditioned) works unconditionally, on
the SciPy sparse `A` this module already accepts. For `M != None`,
`jacobi_preconditioner_torch()`/`ssor_preconditioner_torch()` below are
the torch-native preconditioners this item actually ports; `ssor_
preconditioner_torch()` DENSIFIES the triangular factors and uses
`torch.linalg.solve_triangular` rather than a genuine sparse triangular
solve -- an honest scope limit (torch's sparse-triangular-solve support
is not broadly/stably available across versions the way
`scipy.sparse.linalg.spsolve_triangular` is), acceptable for the
moderate system sizes this module's own SSOR usage already targets
(the same "torch_dense_solve() for moderate size" tradeoff `torch_
sparse_solver.py` already documents for a different function).
`incomplete_cholesky0()`'s IC(0) factorization is NOT ported to torch
by this item -- `preconditioned_cg(backend="torch")` given an IC(0)
preconditioner (or any other numpy-only callable) raises a clear
`TypeError` rather than silently misbehaving when a numpy array meets a
torch tensor mid-iteration; solver.py's own `solve_static(method="pcg",
preconditioner="ic0")` combined with `backend="torch"` raises the same
way. `solve_static(method="cg"/"pcg")` combined with `backend="torch"`
on the `FESystem` itself now routes through this torch path (closing
the gap Wave 16 item 132's own docstring flagged: "method= is only
supported with backend='scipy'... backend='torch'/'auto' already has
its own separate CG dispatch"); `backend="auto"` still raises for
`method=` -- see `solve_static()`'s own updated docstring for why that
one case remains disallowed.
"""
__author__ = "Abhijeet"
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve, spsolve_triangular

try:
    from .torch_sparse_solver import _HAS_TORCH, _require_torch, _cg_torch, _scipy_csr_to_torch_sparse
except Exception:
    # See mesh_transform.py's/vectorized_assembly.py's own identical
    # except clause for why this is a defensive backstop, not the
    # primary path.
    _HAS_TORCH = False

    def _require_torch():
        raise ImportError(
            "iterative_solvers.py: torch_sparse_solver.py (this "
            "project's single _HAS_TORCH/_require_torch source of "
            "truth) could not be imported -- backend='torch' is "
            "unavailable.")


# =====================================================================
# Item 36 -- fill-reducing reordering (min-degree/RCM)
# =====================================================================
def _adjacency_from_sparsity(A):
    """Undirected adjacency lists from A's sparsity PATTERN (i~j iff
    A[i,j] or A[j,i] is structurally nonzero and i != j) -- values are
    discarded, only used for the reordering graph."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    sym = (A + A.T).tocsr()
    adj = []
    for i in range(n):
        row = sym.indices[sym.indptr[i]:sym.indptr[i + 1]]
        adj.append([int(j) for j in row if j != i])
    return adj


def reverse_cuthill_mckee(A):
    """Hand-written Cuthill-McKee ordering (each BFS level visited in
    ascending-degree order, standard George & Liu 1981 heuristic),
    reversed at the end -- the "reverse" in RCM, which empirically
    reduces fill-in further than plain Cuthill-McKee for essentially
    free (see Cuthill & McKee 1969; George 1971 established the
    "reverse" improvement). Handles disconnected graphs by restarting
    the BFS from the lowest-degree unvisited node until every node is
    placed -- a graph-theoretic correctness requirement most textbook
    descriptions gloss over, but a real assembled FE stiffness graph
    for a model with, e.g., 2 physically separate mesh blocks IS
    disconnected.

    Returns perm (ndarray, length n) such that A[np.ix_(perm, perm)]
    is the reordered matrix -- see fill_in_count() for measuring the
    actual fill-in reduction this achieves on a specific A, and
    permuted_solve() for using it directly in a solve."""
    n = A.shape[0]
    adj = _adjacency_from_sparsity(A)
    degree = np.array([len(a) for a in adj])
    visited = np.zeros(n, dtype=bool)
    remaining = set(range(n))
    order = []
    while remaining:
        start = min(remaining, key=lambda i: degree[i])
        visited[start] = True
        remaining.discard(start)
        queue = [start]
        order.append(start)
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            nbrs = sorted((v for v in adj[u] if not visited[v]), key=lambda v: degree[v])
            for v in nbrs:
                visited[v] = True
                remaining.discard(v)
                queue.append(v)
                order.append(v)
    return np.array(order[::-1], dtype=int)


def fill_in_count(A, natural=True):
    """nnz of a sparse LU's combined L+U factors (diagonal counted
    once), via scipy.sparse.linalg.splu() -- the fill-in metric
    fem_implementation_lessons.md cites (minimum-degree reordering
    "cuts factor fill by half" in the book's own numbers).

    natural=True (default) passes permc_spec='NATURAL' and
    diag_pivot_thresh=0.0 to splu(), which disables BOTH SuperLU's own
    internal column-reordering AND its partial-pivoting row search --
    without this, comparing fill_in_count(A) against
    fill_in_count(A[rcm][:,rcm]) would be confounded by SuperLU's own
    (already fairly good) internal reordering silently doing most of
    the work regardless of how the caller pre-permuted A, masking the
    effect this function exists to measure. With natural=True, the
    measured fill is a direct function of the row/column order the
    caller supplies -- exactly what's needed to demonstrate
    reverse_cuthill_mckee()'s effect in isolation (see
    tests/test_iterative_solvers.py). natural=False uses SuperLU's own
    default (COLAMD) reordering on top, appropriate when the goal is
    "how much fill will an actual spsolve() call produce," not
    "how much does MY permutation choice matter."""
    A = sp.csc_matrix(A)
    kwargs = {"permc_spec": "NATURAL", "diag_pivot_thresh": 0.0} if natural else {}
    lu = splu(A, **kwargs)
    return int(lu.L.nnz + lu.U.nnz - A.shape[0])


def permuted_solve(A, b, perm=None):
    """Solve A x = b via an explicit symmetric permutation
    A' = P A P^T, b' = P b, x = P^T x' -- perm=None computes RCM
    internally. Returns (x, perm) so callers can inspect/reuse the
    permutation (e.g. to also call fill_in_count() on the same
    ordering)."""
    if perm is None:
        perm = reverse_cuthill_mckee(A)
    A = sp.csc_matrix(A)
    Ap = A[perm, :][:, perm]
    bp = np.asarray(b)[perm]
    xp = spsolve(Ap, bp)
    x = np.empty_like(xp)
    x[perm] = xp
    return x, perm


# =====================================================================
# Item 37 -- preconditioners (each returns a callable z = M^{-1} r)
# =====================================================================
def jacobi_preconditioner(A):
    """M = diag(A). "Frequently not very helpful" for a constant
    diagonal (fem_implementation_lessons.md's own candid framing of
    the book's assessment) but a real help for variable-coefficient
    problems -- kept as the cheapest, always-available baseline."""
    d = np.asarray(A.diagonal(), dtype=float)
    if np.any(d == 0):
        raise ValueError("jacobi_preconditioner: A has a zero diagonal entry.")
    d_inv = 1.0 / d
    return lambda r: d_inv * r


def ssor_preconditioner(A, omega=1.0):
    """Symmetric SOR preconditioner (Saad, "Iterative Methods for
    Sparse Linear Systems," Ch. 4): for symmetric A = D + L + L^T,
        M = (D/omega + L) D^{-1} (D/omega + L^T)
    applied via one forward and one backward sparse triangular solve.
    (A positive scalar multiple of M gives IDENTICAL PCG iterates in
    exact arithmetic -- the standard "M -> cM leaves x_k unchanged"
    invariance of the preconditioned-CG recurrence -- so the various
    scalar normalization constants different textbooks attach to this
    formula are immaterial here and intentionally omitted.) A must be
    symmetric with a strictly positive diagonal (SPD is assumed, not
    re-checked here -- see solver.FESystem._dense_spd_solve()'s own
    Cholesky-based SPD check for that concern elsewhere in this
    package). "Roughly halves iteration count" in fem_implementation_
    lessons.md's own citation of the book's SSOR example."""
    A = sp.csr_matrix(A)
    d = np.asarray(A.diagonal(), dtype=float)
    if np.any(d <= 0):
        raise ValueError("ssor_preconditioner: requires a strictly positive diagonal (SPD A).")
    L = sp.tril(A, k=-1, format="csr")
    lower = (sp.diags(d / omega) + L).tocsr()
    upper = lower.T.tocsr()

    def apply(r):
        y = spsolve_triangular(lower, r, lower=True)
        y = d * y
        return spsolve_triangular(upper, y, lower=False)
    return apply


def incomplete_cholesky0(A):
    """IC(0): incomplete Cholesky restricted to A's OWN lower-triangle
    sparsity pattern -- no fill-in beyond what A already has (the "(0)"
    level of fill). Hand-written (scipy ships spilu() for the
    general/nonsymmetric ILU case but no symmetric IC(0)) via the
    standard left-looking sparse-Cholesky recursion (Saad Ch. 10,
    Algorithm 10.3, specialized to the symmetric/no-extra-fill case):
    for each row i, for each existing nonzero column k < i (in
    increasing order), subtract the inner product of the ALREADY-
    COMPUTED entries rows i and k share below column k, divide by
    L[k,k]; then the diagonal L[i,i] = sqrt(A[i,i] - sum(L[i,:i]**2)).
    Breaks (raises) if a diagonal update goes non-positive -- IC(0)
    is not guaranteed to exist for every SPD matrix (M-matrices are
    the classical sufficient condition; a general SPD FE stiffness
    matrix usually, but not always, works in practice).

    Cost is O(n * (avg nnz/row)^2) using a dict-of-dicts per row,
    which is genuinely cheap on a narrow-bandwidth matrix -- pairing
    this with reverse_cuthill_mckee() reordering first (item 36) is
    the natural, worth-doing-together combination this module is
    built to make easy (see tests/test_iterative_solvers.py).

    Returns a callable z = M^{-1} r = L^{-T} L^{-1} r via two sparse
    triangular solves."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    # Per-row dict of already-known L entries {col: value}, built in
    # increasing row order so every dependency (k < i) is available.
    L_rows = [dict() for _ in range(n)]
    a_indptr, a_indices, a_data = A.indptr, A.indices, A.data
    for i in range(n):
        row_cols = a_indices[a_indptr[i]:a_indptr[i + 1]]
        row_vals = a_data[a_indptr[i]:a_indptr[i + 1]]
        a_row = dict(zip(row_cols.tolist(), row_vals.tolist()))
        lower_cols = sorted(c for c in row_cols if c < i)
        Li = L_rows[i]
        for k in lower_cols:
            s = a_row[k]
            Lk = L_rows[k]
            # intersect columns < k already present in both row i and row k
            common = (set(Li) & set(Lk))
            common.discard(k)
            for j in common:
                if j < k:
                    s -= Li[j] * Lk[j]
            Li[k] = s / Lk[k]
        diag = a_row.get(i, 0.0)
        for j, v in Li.items():
            diag -= v * v
        if diag <= 0.0:
            raise np.linalg.LinAlgError(
                f"incomplete_cholesky0: breakdown at row {i} (non-positive "
                f"pivot {diag:.3e}) -- IC(0) does not exist for this matrix; "
                "try ssor_preconditioner()/jacobi_preconditioner() instead.")
        Li[i] = diag ** 0.5

    rows, cols, vals = [], [], []
    for i, Li in enumerate(L_rows):
        for j, v in Li.items():
            rows.append(i)
            cols.append(j)
            vals.append(v)
    L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def apply(r):
        y = spsolve_triangular(L, r, lower=True)
        return spsolve_triangular(L.T.tocsr(), y, lower=False)
    return apply


def jacobi_preconditioner_torch(A, device="cpu", dtype=None):
    """Torch-native counterpart of jacobi_preconditioner() above --
    Wave 9 addendum item 137. A pure elementwise scaling, so the torch
    port is a direct, zero-compromise translation (no densification,
    no approximation): M = diag(A), z = r / diag(A). Returns a callable
    z = M^{-1} r taking/returning a 1-D torch.Tensor on `device`, for
    use with preconditioned_cg(..., backend="torch")."""
    _require_torch()
    import torch
    dtype = dtype or torch.float64
    d = np.asarray(sp.csr_matrix(A).diagonal(), dtype=float)
    if np.any(d == 0):
        raise ValueError("jacobi_preconditioner_torch: A has a zero diagonal entry.")
    d_inv_t = torch.as_tensor(1.0 / d, dtype=dtype, device=device)
    apply = lambda r: d_inv_t * r
    apply._fea_engine_torch_preconditioner = True   # see preconditioned_cg()'s own check
    return apply


def ssor_preconditioner_torch(A, omega=1.0, device="cpu", dtype=None):
    """Torch-native counterpart of ssor_preconditioner() above -- Wave 9
    addendum item 137. Same M = (D/omega + L) D^{-1} (D/omega + L^T)
    formula, but DENSIFIES the triangular factors and applies
    torch.linalg.solve_triangular() instead of a genuine sparse
    triangular solve -- see this module's own docstring, 'GPU/torch
    side-by-side path', for why this is an honest, explicitly-scoped
    simplification rather than a full sparse port (torch has no
    broadly-available stable equivalent of scipy.sparse.linalg.
    spsolve_triangular() across versions). Appropriate for the moderate
    system sizes this module's own SSOR usage already targets, not for
    very large systems where densifying the triangular factors would
    itself be the memory bottleneck."""
    _require_torch()
    import torch
    dtype = dtype or torch.float64
    A_csr = sp.csr_matrix(A)
    d = np.asarray(A_csr.diagonal(), dtype=float)
    if np.any(d <= 0):
        raise ValueError("ssor_preconditioner_torch: requires a strictly positive diagonal (SPD A).")
    L = sp.tril(A_csr, k=-1, format="csr")
    lower_np = (sp.diags(d / omega) + L).toarray()
    upper_np = lower_np.T
    lower_t = torch.as_tensor(lower_np, dtype=dtype, device=device)
    upper_t = torch.as_tensor(upper_np, dtype=dtype, device=device)
    d_t = torch.as_tensor(d, dtype=dtype, device=device)

    def apply(r):
        y = torch.linalg.solve_triangular(lower_t, r.unsqueeze(-1), upper=False).squeeze(-1)
        y = d_t * y
        return torch.linalg.solve_triangular(upper_t, y.unsqueeze(-1), upper=True).squeeze(-1)
    apply._fea_engine_torch_preconditioner = True   # see preconditioned_cg()'s own check
    return apply


# =====================================================================
# Item 38 -- (preconditioned) conjugate gradients
# =====================================================================
def preconditioned_cg(A, b, M=None, x0=None, tol=1e-8, maxiter=None, callback=None,
                       backend="scipy", device="cpu"):
    """Hand-written preconditioned CG for SPD A (Hestenes & Stiefel
    1952 recurrence -- see fem_implementation_lessons.md's own summary
    of the book's Chapter 11: convergence tracks sqrt(cond(K)), not
    cond(K), which is CG's headline improvement over plain stationary
    iteration). M: a preconditioner callable z = M^{-1} r (see
    jacobi_preconditioner()/ssor_preconditioner()/incomplete_
    cholesky0() above), or None for unpreconditioned CG.

    backend="scipy" (default) is the ORIGINAL, unchanged behavior --
    every array a plain numpy ndarray, M (if given) a numpy-in-numpy-
    out callable. backend="torch" (Wave 9 addendum item 137, docs/
    consolidated_future_roadmap.md) reuses torch_sparse_solver.py's own
    generic _cg_torch() loop instead of re-deriving the recurrence a
    second time; `callback` is not supported on this path (raises
    TypeError if given, rather than silently never calling it -- this
    module's own "fail loudly" convention). M, if given, must be a
    TORCH-compatible callable (jacobi_preconditioner_torch()/ssor_
    preconditioner_torch() above, or M=None for unpreconditioned) --
    detected via a marker attribute jacobi_preconditioner_torch()/
    ssor_preconditioner_torch() set on their returned callable, checked
    UP FRONT before any iteration starts: passing a numpy-only
    preconditioner (incomplete_cholesky0()'s return value, or
    jacobi_preconditioner()/ssor_preconditioner()'s own numpy versions)
    raises TypeError immediately, rather than silently coercing/
    breaking partway through the iteration.

    Returns (x, n_iter). Raises LinAlgError if a search direction with
    p^T A p <= 0 is encountered (A is not actually SPD), RuntimeError
    if maxiter is reached without meeting tol (relative residual
    norm ||b-Ax||/||b||)."""
    if backend not in ("scipy", "torch"):
        raise ValueError(
            f"preconditioned_cg: unknown backend={backend!r} -- expected "
            f"'scipy' (default) or 'torch'.")
    n = b.shape[0]
    if maxiter is None:
        maxiter = 10 * n

    if backend == "torch":
        if callback is not None:
            raise TypeError(
                "preconditioned_cg(backend='torch'): callback= is not "
                "supported on the torch path -- see this function's own "
                "docstring.")
        _require_torch()
        import torch
        if M is not None and not getattr(M, "_fea_engine_torch_preconditioner", False):
            raise TypeError(
                "preconditioned_cg(backend='torch'): M must be M=None "
                "(unpreconditioned), or a preconditioner built by "
                "jacobi_preconditioner_torch()/ssor_preconditioner_torch() -- "
                "a numpy-only preconditioner (jacobi_preconditioner()/"
                "ssor_preconditioner()/incomplete_cholesky0()'s own return "
                "values, built for the scipy path) is not accepted here, to "
                "avoid a numpy array silently meeting a torch tensor "
                "mid-iteration. See this function's own docstring.")
        A_csr = sp.csr_matrix(A) if sp.issparse(A) else sp.csr_matrix(np.asarray(A, dtype=float))
        A_t = _scipy_csr_to_torch_sparse(A_csr, device, torch.float64)
        matvec = lambda v: A_t @ v
        b_t = torch.as_tensor(np.asarray(b, dtype=float), dtype=torch.float64, device=device)
        x0_t = (torch.zeros(n, dtype=torch.float64, device=device) if x0 is None
                else torch.as_tensor(np.asarray(x0, dtype=float), dtype=torch.float64, device=device))
        precond_inv = (lambda v: v) if M is None else M
        x_t, n_iter, converged = _cg_torch(matvec, b_t, x0_t, precond_inv, tol, maxiter)
        if not converged:
            raise RuntimeError(
                f"preconditioned_cg(backend='torch'): did not converge in "
                f"{maxiter} iterations.")
        return x_t.cpu().numpy(), n_iter

    A_mv = (lambda v: A @ v) if not sp.issparse(A) else (lambda v, A=sp.csr_matrix(A): A @ v)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        bnorm = 1.0
    r = b - A_mv(x)
    if np.linalg.norm(r) / bnorm < tol:
        return x, 0
    Minv = M if M is not None else (lambda v: v)
    z = Minv(r)
    p = z.copy()
    rz_old = float(r @ z)
    rel = np.linalg.norm(r) / bnorm
    for k in range(1, maxiter + 1):
        Ap = A_mv(p)
        pAp = float(p @ Ap)
        if pAp <= 0.0:
            raise np.linalg.LinAlgError(
                f"preconditioned_cg: p^T A p = {pAp:.3e} <= 0 at iteration {k} "
                "-- A is not SPD.")
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        if callback is not None:
            callback(k, x, r)
        rel = np.linalg.norm(r) / bnorm
        if rel < tol:
            return x, k
        z = Minv(r)
        rz_new = float(r @ z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    raise RuntimeError(
        f"preconditioned_cg: did not converge in {maxiter} iterations "
        f"(final relative residual {rel:.3e} >= tol={tol:.1e}).")


# =====================================================================
# Item 39 -- classical stationary iterations (Jacobi, Gauss-Seidel, SOR)
# =====================================================================
# fem_implementation_lessons.md's own framing (Chapter 12): "building
# blocks, not competitive solvers" -- their iteration count tracks
# cond(K) directly (worse than CG's sqrt(cond(K))), so they are not
# meant to be reached for as a standalone large-problem solver. Their
# real role in THIS module is as multigrid smoothers (item 40 below),
# which only need a few sweeps, not convergence to tol -- the
# standalone *_solve() functions below exist to complete the item as
# specified and for direct comparison against CG in
# tests/test_iterative_solvers.py, matching the book's own comparison.
def _jacobi_sweep(A_csr, d_inv, b, x):
    x[:] = x + d_inv * (b - A_csr @ x)
    return x


def _gauss_seidel_sweep(A_csr, d, b, x, omega=1.0):
    """One in-place SOR sweep (omega=1 -> plain Gauss-Seidel). Uses
    the "j<i" entries already updated THIS sweep and the "j>i" entries
    still holding their previous-sweep value -- the defining property
    of Gauss-Seidel (as opposed to Jacobi, which uses only
    previous-sweep values for every j)."""
    indptr, indices, data = A_csr.indptr, A_csr.indices, A_csr.data
    n = len(d)
    for i in range(n):
        s = b[i]
        for idx in range(indptr[i], indptr[i + 1]):
            j = indices[idx]
            if j != i:
                s -= data[idx] * x[j]
        x_new_i = s / d[i]
        x[i] = x[i] + omega * (x_new_i - x[i])
    return x


def jacobi_solve(A, b, x0=None, tol=1e-8, maxiter=10000):
    A = sp.csr_matrix(A)
    d_inv = 1.0 / np.asarray(A.diagonal(), dtype=float)
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    bnorm = np.linalg.norm(b) or 1.0
    for k in range(1, maxiter + 1):
        r = b - A @ x
        if np.linalg.norm(r) / bnorm < tol:
            return x, k - 1
        _jacobi_sweep(A, d_inv, b, x)
    raise RuntimeError(f"jacobi_solve: did not converge in {maxiter} iterations.")


def gauss_seidel_solve(A, b, x0=None, tol=1e-8, maxiter=10000):
    return sor_solve(A, b, omega=1.0, x0=x0, tol=tol, maxiter=maxiter)


def sor_solve(A, b, omega=1.5, x0=None, tol=1e-8, maxiter=10000):
    A = sp.csr_matrix(A)
    d = np.asarray(A.diagonal(), dtype=float)
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    bnorm = np.linalg.norm(b) or 1.0
    for k in range(1, maxiter + 1):
        r = b - A @ x
        if np.linalg.norm(r) / bnorm < tol:
            return x, k - 1
        _gauss_seidel_sweep(A, d, b, x, omega=omega)
    raise RuntimeError(f"sor_solve: did not converge in {maxiter} iterations.")


# =====================================================================
# Item 40 -- geometric multigrid for structured Quad4 rectangle grids
# =====================================================================
# Scope, stated explicitly up front (same "narrow, then document the
# narrowing" convention as grading.Notch/Fillet in Wave 5): this
# multigrid implementation is GEOMETRIC, not algebraic, and applies
# only to the structured Quad4 rectangular grids mesh.rectangle_mesh()
# produces (a plain nx-by-ny grid; no holes/fillets/notches/
# unstructured-Gmsh topology). That is the case where an EXACT nested
# mesh hierarchy is trivial to build: mesh.rectangle_mesh(Lx, Ly,
# k*nx0, k*ny0) for k=1,2,4,... on the SAME (Lx, Ly) reuses the
# identical node_id(i,j) = i*n_ny+j indexing convention, so every
# coarse-level node coordinate is EXACTLY a fine-level node coordinate
# (dx_fine = dx_coarse/2, so fine node (2i,2j) sits at the same (x,y)
# as coarse node (i,j)) -- no approximate nearest-node interpolation
# needed, and the standard bilinear structured-grid prolongation
# stencil below is exact, not approximate, for THIS pair of grids.
# Extending this to unstructured/graded meshes needs a materially
# different (interpolation-based, non-exact) prolongation operator --
# out of scope here, the same "blocked, not attempted" framing Wave 5
# gave items 28/29.
def structured_quad_hierarchy(Lx, Ly, nx0, ny0, n_levels, x0=0.0, y0=0.0):
    """Coarsest-first list of n_levels mesh.Mesh objects; level k has
    (nx0*2**k, ny0*2**k) Quad4 elements per direction."""
    from .mesh import rectangle_mesh
    return [rectangle_mesh(Lx, Ly, nx0 * 2 ** k, ny0 * 2 ** k, x0=x0, y0=y0)
            for k in range(n_levels)]


def node_prolongation_matrix(nx_coarse, ny_coarse):
    """Sparse bilinear prolongation P, shape (n_fine_nodes,
    n_coarse_nodes), between a mesh.rectangle_mesh(nx_coarse,ny_coarse)
    grid and the mesh.rectangle_mesh(2*nx_coarse, 2*ny_coarse) grid
    covering the same (Lx, Ly) -- standard structured-grid multigrid
    stencil: a fine node coincident with a coarse node gets weight 1
    (injection); a fine node at the midpoint of a coarse edge gets
    weight 1/2 from each of that edge's 2 coarse endpoints; a fine
    node at a coarse cell's center gets weight 1/4 from each of the 4
    surrounding coarse corners. This is exact bilinear interpolation
    of the coarse nodal field onto the fine grid, since both grids are
    uniform Cartesian by construction."""
    n_nx_c, n_ny_c = nx_coarse + 1, ny_coarse + 1
    n_nx_f, n_ny_f = 2 * nx_coarse + 1, 2 * ny_coarse + 1

    def cid(i, j):
        return i * n_ny_c + j

    def fid(i, j):
        return i * n_ny_f + j

    rows, cols, vals = [], [], []
    for I in range(n_nx_f):
        ic, i_even = I // 2, (I % 2 == 0)
        for J in range(n_ny_f):
            jc, j_even = J // 2, (J % 2 == 0)
            fr = fid(I, J)
            if i_even and j_even:
                rows.append(fr); cols.append(cid(ic, jc)); vals.append(1.0)
            elif (not i_even) and j_even:
                for di in (0, 1):
                    rows.append(fr); cols.append(cid(ic + di, jc)); vals.append(0.5)
            elif i_even and (not j_even):
                for dj in (0, 1):
                    rows.append(fr); cols.append(cid(ic, jc + dj)); vals.append(0.5)
            else:
                for di in (0, 1):
                    for dj in (0, 1):
                        rows.append(fr); cols.append(cid(ic + di, jc + dj)); vals.append(0.25)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_nx_f * n_ny_f, n_nx_c * n_ny_c))


def dof_prolongation_matrix(P_node, dofs_per_node):
    """Kronecker-expand a node-level prolongation matrix to DOF space
    (node-major numbering dof = dofs_per_node*node + component,
    matching solver.FESystem's own global-DOF convention)."""
    return sp.kron(P_node, sp.identity(dofs_per_node), format="csr")


def restrict_prolongation_to_free_dofs(P, free_fine, free_coarse):
    """Restrict a full-DOF prolongation matrix to the free-DOF
    subspace at each level -- valid whenever "free" is a consistent
    GEOMETRIC predicate applied independently at every level (e.g.
    "not on the fixed edge"), which is the only case this module's
    hierarchy-building helpers are meant to be used with."""
    return P.tocsr()[free_fine, :][:, free_coarse]


def galerkin_coarse_operator(A_fine, P):
    """A_coarse = P^T A_fine P -- the Galerkin coarse-grid operator.
    fem_implementation_lessons.md calls out the matching structural
    fact directly: restricting, applying the fine stiffness matrix,
    and prolonging is EXACTLY the coarse-mesh stiffness matrix
    (I_{h,2h} K_h I_{2h,h} = K_{2h}) for the RIGHT choice of
    restriction/prolongation pair -- verified numerically (not just
    asserted) in tests/test_multigrid.py by comparing this against an
    independently, directly-assembled coarse-mesh K."""
    Pc = sp.csr_matrix(P)
    return (Pc.T @ sp.csr_matrix(A_fine) @ Pc).tocsr()


def _direct_solve(A, b):
    if sp.issparse(A):
        return spsolve(sp.csc_matrix(A), b)
    return np.linalg.solve(A, b)


def _mg_smooth(A_csr, d, b, x, n_sweeps, smoother, omega):
    for _ in range(n_sweeps):
        if smoother == "jacobi":
            _jacobi_sweep(A_csr, 1.0 / d, b, x)
        else:
            _gauss_seidel_sweep(A_csr, d, b, x, omega=omega)
    return x


def v_cycle(A_levels, P_levels, b, n_pre=2, n_post=2, smoother="gauss_seidel", omega=1.0):
    """One recursive V-cycle correction for A_levels[-1] x = b.
    A_levels: coarsest-first list of SPD matrices (already restricted
    to each level's own free DOFs, see restrict_prolongation_to_
    free_dofs()); A_levels[-1] is the finest level, matching b's size.
    P_levels: list of len(A_levels)-1 prolongation matrices,
    P_levels[k] mapping level-k (coarser) DOFs to level-(k+1) (finer)
    DOFs. The coarsest level (level 0) is solved directly.

    Returns the approximate solution x (NOT a correction) to
    A_levels[-1] x = b -- see multigrid_solve() for repeated
    (outer, residual-based) V-cycles used as a standalone iterative
    solver."""
    n_levels = len(A_levels)
    A_csrs = [sp.csr_matrix(A) for A in A_levels]
    diags = [np.asarray(A.diagonal(), dtype=float) for A in A_csrs]

    def recurse(level, b_):
        A = A_csrs[level]
        if level == 0:
            return _direct_solve(A_levels[0], b_)
        x_ = np.zeros_like(b_)
        _mg_smooth(A, diags[level], b_, x_, n_pre, smoother, omega)
        r = b_ - A @ x_
        P = P_levels[level - 1]
        r_coarse = P.T @ r
        e_coarse = recurse(level - 1, r_coarse)
        x_ = x_ + P @ e_coarse
        _mg_smooth(A, diags[level], b_, x_, n_post, smoother, omega)
        return x_

    return recurse(n_levels - 1, b)


def multigrid_solve(A_levels, P_levels, b, tol=1e-8, maxiter=50, n_pre=2, n_post=2,
                     smoother="gauss_seidel", omega=1.0):
    """Standalone iterative solver built from repeated V-cycles: each
    outer iteration solves the RESIDUAL equation A e = r via one
    v_cycle() call and accumulates x += e -- the standard "multigrid
    as a stationary iterative method" usage (as opposed to using a
    single V-cycle as a Krylov preconditioner, not implemented here).
    Returns (x, n_iter)."""
    A_fine = sp.csr_matrix(A_levels[-1])
    x = np.zeros(b.shape[0])
    bnorm = np.linalg.norm(b) or 1.0
    for k in range(1, maxiter + 1):
        r = b - A_fine @ x
        if np.linalg.norm(r) / bnorm < tol:
            return x, k - 1
        e = v_cycle(A_levels, P_levels, r, n_pre=n_pre, n_post=n_post,
                    smoother=smoother, omega=omega)
        x = x + e
    raise RuntimeError(f"multigrid_solve: did not converge in {maxiter} V-cycles.")
