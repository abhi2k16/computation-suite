"""
torch_sparse_solver.py -- Wave 0 item 3 (docs/consolidated_future_
roadmap.md, source tensormesh_comparative_analysis.md Section 6.1): a
PyTorch-based linear-solve path for an already-assembled SciPy sparse
system, opening a GPU-resident solve option for the larger systems
Hex20/Tet10/graded meshes can produce -- WITHOUT touching any
element/assembly code. solver.py's FESystem still does all assembly
(K/F) exactly as before; this module only offers an alternative FINAL
linear-algebra step, via fesystem_solve_static_torch() below, which a
caller opts into explicitly.

Per the user's explicit instruction earlier in this project's
development, this module uses bare `torch` ops only -- no
`tensormesh`/`torch-sla` package import anywhere; TensorMesh's own
docs (Section 6.1's own source) were consulted only as design
inspiration.

TWO solve paths, matching the roadmap item's own "PyTorch's own sparse
linear algebra OR a hand-written preconditioned CG/BiCGSTAB" wording:

  torch_dense_solve()     -- densify + torch.linalg.solve(). The
                              simplest possible GPU-capable path (any
                              Kff, no structural assumption), but O(n^2)
                              memory -- only sensible for small/medium
                              systems, or as an independent correctness
                              reference for the sparse path below.
  torch_sparse_cg_solve() -- a hand-written, Jacobi-preconditioned
                              Conjugate Gradient solver operating on a
                              genuine torch.sparse_csr_tensor (no
                              densification, GPU-resident matvecs).
                              REQUIRES Kff to be SPD -- CG is not a
                              general-purpose solver; BiCGSTAB (the
                              roadmap's other named option) would drop
                              that requirement at the cost of real
                              extra complexity (non-symmetric Krylov
                              recurrences have more numerically
                              delicate breakdown conditions), and this
                              package's own linear-elastic static solve
                              is already SPD in the well-constrained
                              case (see solver.py's Wave 0 item 5,
                              _dense_spd_solve()/_sparse_lu_solve()) --
                              so CG is the right first cut here, with
                              BiCGSTAB left as documented future work
                              (docs/consolidated_future_roadmap.md
                              backlog items 37/38, "Preconditioning"/
                              "Conjugate gradients", note the CG half of
                              that backlog item is effectively
                              superseded by this module).

HONESTY NOTE (development-sandbox limitation, since resolved by real
execution -- see "VALIDATED" below): this module was written and
installed in an environment (this project's own development sandbox)
where the publicly downloadable `torch` PyPI wheel could be downloaded
and pip-installed but NOT actually imported -- every default-index
build tried (current release and a legacy 1.13.1 release) is
CUDA-linked and unconditionally dlopen()s CUDA runtime libraries
(libcudart/libcublas/libcublasLt/...) at import time, which are not
present on a CPU-only machine with no NVIDIA driver; the separate
CPU-only wheel index (download.pytorch.org/whl/cpu) is blocked by this
sandbox's network proxy. The test file skips itself cleanly there
(gated on _HAS_TORCH, see below) rather than reporting a false pass.

VALIDATED (2026-09-08, on the user's own machine -- Windows, NVIDIA
GeForce GTX 1050, PyTorch 2.7.0/CUDA 12.6): all 7 tests in
tests/test_torch_sparse_solver.py pass, on CPU (device="cpu"):
  - torch_dense_solve() vs np.linalg.solve() on a random SPD system:
    2.552e-16 relative.
  - torch_sparse_cg_solve() vs np.linalg.solve() on a random sparse SPD
    system: converged in 13 iterations, 1.184e-11 relative.
  - fesystem_solve_static_torch(method='dense') vs FESystem.
    solve_static() on a real cantilever model: 2.261e-13 relative.
  - fesystem_solve_static_torch(method='cg') vs FESystem.solve_static()
    on the same model (sparse K): 2.119e-14 relative.
  - torch_sparse_cg_solve() correctly raises ValueError on a matrix
    that cannot be SPD (non-positive diagonal).
  - FESystem(..., backend='torch').solve_static() (dense) vs
    FESystem(..., backend='scipy').solve_static() on the same model,
    via the integrated backend= API (solver.py) rather than calling
    fesystem_solve_static_torch() directly: 2.261e-13 relative.
  - FESystem(..., sparse=True, backend='torch').solve_static() (CG) vs
    backend='scipy' on the same model: 8.857e-13 relative.
Run so far on CPU only -- device="cuda" is implemented and available
on this same install but not yet separately exercised.
"""
import numpy as np
import scipy.sparse as sp

try:
    import torch
    _HAS_TORCH = True
except Exception:
    # See autograd_tangent.py's own comment on this exact except clause
    # for why it is deliberately broad, not `except ImportError` --
    # the identical CUDA-linking failure mode applies here too.
    _HAS_TORCH = False


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "torch_sparse_solver.py requires the 'torch' package: pip "
            "install torch -- this, like gmsh (see geometry/gmsh_engine.py) "
            "and autograd_tangent.py, is an OPTIONAL dependency of "
            "fea_engine; every other module (including the default SciPy "
            "solve path in solver.py) works without it. See this module's "
            "own docstring for a known environment where torch installs "
            "but fails to import (a CUDA-linked wheel with no CUDA "
            "runtime present) -- tests/test_torch_sparse_solver.py skips "
            "itself cleanly in that case rather than failing.")


def _to_csr_scipy(K):
    """Accepts a SciPy sparse matrix (any format) or a dense numpy
    array/dense-castable object and returns a SciPy CSR matrix --
    the single normalized input format every function below converts
    from into torch tensors."""
    if sp.issparse(K):
        return K.tocsr()
    return sp.csr_matrix(np.asarray(K, dtype=float))


# =====================================================================
# Dense path (simplest GPU-capable solve; also the correctness
# reference torch_sparse_cg_solve() is validated against)
# =====================================================================
def torch_dense_solve(K, F, device="cpu", dtype=None):
    """Densify K (any SciPy-sparse or dense input) and solve via
    torch.linalg.solve() -- see this module's own docstring for when
    this is the right choice vs. torch_sparse_cg_solve(). device="cuda"
    runs the dense LU on GPU if one is available and torch was built
    with CUDA support; device="cpu" (default) works everywhere torch
    itself works.

    K: SciPy sparse matrix or dense array-like, (n, n). F: (n,)
    array-like. Returns x, a (n,) numpy array."""
    _require_torch()
    dtype = dtype or torch.float64
    K_csr = _to_csr_scipy(K)
    K_dense = np.asarray(K_csr.todense(), dtype=float)
    K_t = torch.tensor(K_dense, dtype=dtype, device=device)
    b_t = torch.tensor(np.asarray(F, dtype=float), dtype=dtype, device=device)
    x_t = torch.linalg.solve(K_t, b_t)
    return x_t.cpu().numpy()


# =====================================================================
# Sparse, Jacobi-preconditioned Conjugate Gradient (SPD only)
# =====================================================================
def _scipy_csr_to_torch_sparse(K_csr, device, dtype):
    """SciPy CSR -> torch.sparse_csr_tensor, sharing the SAME
    crow/col/values layout (no reordering/re-permutation), so this is a
    direct format conversion, not a re-factorization -- the whole point
    of reusing an already-assembled SciPy matrix rather than rebuilding
    it (see this module's own docstring: "without touching any
    element/assembly code")."""
    crow = torch.as_tensor(K_csr.indptr, dtype=torch.int64, device=device)
    col = torch.as_tensor(K_csr.indices, dtype=torch.int64, device=device)
    values = torch.as_tensor(K_csr.data, dtype=dtype, device=device)
    return torch.sparse_csr_tensor(crow, col, values, size=K_csr.shape, device=device)


def _cg_torch(matvec, b, x0, precond_inv, tol, max_iter):
    """Standard preconditioned Conjugate Gradient (Golub & Van Loan
    Algorithm 11.5.1 / Trefethen & Bau Lecture 38), expressed entirely
    in torch ops so every iteration's matvec/dot/axpy can run on GPU
    when the caller passes device="cuda". `matvec` and `precond_inv`
    are plain python callables taking/returning a 1-D torch tensor --
    kept generic (not hardcoded to a specific sparse tensor) so this
    same loop could serve a future non-Jacobi preconditioner without
    changes.

    Returns (x, n_iter, converged) -- converged is True iff the
    relative residual norm ||b - A@x|| / ||b|| dropped below `tol`
    within max_iter iterations; the caller decides what to do with a
    False (this function itself never raises on non-convergence, so it
    stays usable for diagnostics even when it doesn't fully converge)."""
    x = x0.clone()
    r = b - matvec(x)
    b_norm = torch.linalg.norm(b)
    if b_norm == 0:
        return x, 0, True
    z = precond_inv(r)
    p = z.clone()
    rz_old = torch.dot(r, z)
    for k in range(max_iter):
        Ap = matvec(p)
        denom = torch.dot(p, Ap)
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rel_resid = torch.linalg.norm(r) / b_norm
        if rel_resid < tol:
            return x, k + 1, True
        z = precond_inv(r)
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x, max_iter, False


def torch_sparse_cg_solve(K, F, tol=1e-8, max_iter=None, device="cpu", dtype=None):
    """Jacobi-preconditioned Conjugate Gradient solve of K @ x = F on a
    genuine torch.sparse_csr_tensor -- no densification, so this scales
    to the larger systems (Hex20/Tet10/graded meshes, see this module's
    own docstring) a dense path can't afford, and every matvec/dot/axpy
    runs GPU-resident when device="cuda".

    REQUIRES K to be symmetric positive definite -- Conjugate Gradient
    is not a general-purpose Krylov method; passing an indefinite or
    non-symmetric K will silently produce a WRONG or non-converging
    result, not a clean error (checking true SPD-ness up front would
    itself cost an O(n^3) eigendecomposition or an O(n) Cholesky
    attempt, defeating the point of an O(n) sparse iterative solve --
    so this function only sanity-checks that the diagonal is strictly
    positive, a cheap NECESSARY-but-not-sufficient SPD condition, and
    trusts the caller for the rest, exactly the same "well-constrained
    linear-elastic K is SPD" assumption solver.py's own Wave 0 item 5
    (_dense_spd_solve()) already makes for the default SciPy path).

    K: SciPy sparse matrix or dense array-like, (n, n), SPD. F: (n,)
    array-like. max_iter: defaults to 10*n (a generous ceiling; a
    well-preconditioned SPD system converges far faster in practice --
    see tests/test_torch_sparse_solver.py for measured iteration
    counts once torch is actually runnable).

    Returns (x (n,) numpy array, n_iter int, converged bool) -- unlike
    torch_dense_solve(), this does NOT raise on non-convergence (see
    _cg_torch()'s own docstring); fesystem_solve_static_torch() below
    is the caller that turns a non-convergent result into a raised
    error for its own solve_static()-compatible contract."""
    _require_torch()
    dtype = dtype or torch.float64
    K_csr = _to_csr_scipy(K)
    n = K_csr.shape[0]
    if max_iter is None:
        max_iter = 10 * n

    diag = K_csr.diagonal()
    if np.any(diag <= 0):
        raise ValueError(
            "torch_sparse_cg_solve: K's diagonal has a non-positive entry -- "
            "K cannot be SPD (a necessary condition fails), so Conjugate "
            "Gradient is not applicable here. Use torch_dense_solve() "
            "instead for a non-SPD system.")

    K_t = _scipy_csr_to_torch_sparse(K_csr, device, dtype)
    b = torch.as_tensor(np.asarray(F, dtype=float), dtype=dtype, device=device)
    diag_t = torch.as_tensor(diag, dtype=dtype, device=device)
    x0 = torch.zeros(n, dtype=dtype, device=device)

    def matvec(x):
        return K_t @ x

    def jacobi_precond_inv(r):
        return r / diag_t

    x, n_iter, converged = _cg_torch(matvec, b, x0, jacobi_precond_inv, tol, max_iter)
    return x.cpu().numpy(), n_iter, converged


# =====================================================================
# Drop-in FESystem.solve_static() alternative
# =====================================================================
def fesystem_solve_static_torch(sysobj, method="cg", device="cpu", tol=1e-8, max_iter=None):
    """Drop-in alternative to FESystem.solve_static() (solver.py) that
    routes the SAME free-dof linear solve through PyTorch instead of
    SciPy -- the concrete demonstration of "a GPU solve path ...
    without touching any element/assembly code" this module's roadmap
    item calls for: `sysobj`'s K/F are already fully assembled by the
    ordinary assemble_stiffness() (unchanged, SciPy/NumPy throughout)
    by the time this function is called; only the FINAL linear-algebra
    step is replaced.

    method="cg" (default): torch_sparse_cg_solve() -- appropriate for
    the ordinary well-constrained linear-elastic static case
    solve_static() itself already treats as (typically) SPD, see
    solver.py's Wave 0 item 5. Raises RuntimeError if CG fails to
    converge (unlike torch_sparse_cg_solve() itself, which just reports
    converged=False -- this function's contract mirrors
    FESystem.solve_static()'s own "either return a real answer or
    raise" contract).
    method="dense": torch_dense_solve() -- no SPD assumption, but O(n^2)
    memory; only sensible for small/medium models here.

    Returns the SAME shape/convention as FESystem.solve_static(): a
    full (n_dof,) numpy array, zero at every fixed dof."""
    _require_torch()
    free = sysobj.free_dofs
    Kff = sysobj._as_solve_matrix(sysobj.K)[np.ix_(free, free)]
    Ff = sysobj.F[free]

    if method == "cg":
        Uf, n_iter, converged = torch_sparse_cg_solve(
            Kff, Ff, tol=tol, max_iter=max_iter, device=device)
        if not converged:
            raise RuntimeError(
                f"fesystem_solve_static_torch: CG did not converge within "
                f"{n_iter} iterations (tol={tol}) -- Kff may not be SPD, "
                "or needs a tighter tol/more iterations/a better "
                "preconditioner than the Jacobi one used here; try "
                "method='dense' as a fallback or diagnostic.")
    elif method == "dense":
        Uf = torch_dense_solve(Kff, Ff, device=device)
    else:
        raise ValueError(f"fesystem_solve_static_torch: unknown method={method!r}, "
                          "expected 'cg' or 'dense'")

    U = np.zeros(sysobj.n_dof)
    U[free] = Uf
    return U
