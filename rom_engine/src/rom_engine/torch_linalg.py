"""
torch_linalg.py -- Wave 9 addendum item 138 (docs/consolidated_future_
roadmap.md): shared, backend-neutral torch-native dense linear-algebra
primitives for rom_engine's classical state-space MOR family
(balanced_truncation.py, hankel_norm.py -- see those modules' own
docstrings for which of their own functions actually route through
this one). Mirrors fea_engine/torch_sparse_solver.py's role (a single
_HAS_TORCH/_require_torch() source of truth plus a handful of reusable
primitives), independently implemented here since rom_engine does not
depend on fea_engine as a library (rom_engine/__init__.py's own stated
design -- fea_engine is a test/example dependency only; nonlinear_
rom.py's own _HAS_TORCH/_require_torch() block already reimplements
the same defensive except-Exception pattern for the identical reason,
duplicated here rather than imported since the two modules must not
depend on each other).

Scope, stated up front (same "narrow, then document explicitly"
convention Wave 9 item 137 used for IC(0)): this module provides
generic torch equivalents of the SPECIFIC primitives balanced_
truncation.py's own hot path (two Lyapunov solves feeding an SVD)
actually needs -- solve_continuous_lyapunov (no native torch
equivalent; solved here via the classical Kronecker-sum vectorization
of the Lyapunov equation, a genuinely DIFFERENT algorithm from scipy's
dense Bartels-Stewart eigenvalue-based solver, cross-validated against
it in tests/test_torch_linalg.py rather than assumed equivalent), an
SVD, and a Gramian square root (Cholesky-or-eigh fallback, mirroring
balanced_truncation._gramian_square_root() exactly). A general
Sylvester-equation solver (torch_solve_sylvester(), the natural
generalization of the Lyapunov solver above -- the Lyapunov equation is
the special case B=A^T, Q symmetric) is also provided, since deriving
the shared Kronecker-vectorization machinery once serves both, but it
is NOT currently wired into any backend="torch" call path in this
package -- see its own docstring for why.

hankel_norm.py's OWN post-balancing machinery (a real Schur
decomposition with custom stable/antistable eigenvalue sorting, then a
Sylvester-equation decoupling step) is NOT ported here at all -- torch
has no Schur-decomposition-with-arbitrary-sort equivalent, and that
step operates on the REDUCED (r-sized, already small by construction)
system regardless of how large the full-order model is, so there is no
real GPU win to chase there even if it were built. Only hankel_norm.
py's reuse of hankel_singular_values() (the genuinely expensive,
full-order-sized step, called both for the main balancing step and for
OptimalHankelNormROM's own empirical self-check) gets a backend="torch"
option, via balanced_truncation.py's own dispatch.

krylov.py (dense-LU-based block Arnoldi, already cheap for the modest
per-generation systems this package's own state spaces produce) and
loewner.py (a tiny generalized eigenproblem sized by the NUMBER OF
INTERPOLATION POINTS, typically far smaller than a full-order model)
are likewise left NumPy/SciPy-only entirely -- matching roadmap item
138's own reasoning: a GPU win here only materializes for a large
full-order model fed directly into balancing, or a batched sweep
across many parameter points (not attempted in this item), and neither
applies to those two modules' own works-on-something-small paths.
"""
import numpy as np

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except Exception:
    # Bare `except Exception`, not `except ImportError` -- see
    # nonlinear_rom.py's own identical _HAS_TORCH block for why (a
    # CUDA-linked PyPI wheel with no matching CUDA runtime can raise
    # OSError/ValueError at import time, not ImportError).
    pass


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "torch_linalg.py: PyTorch is not importable in this "
            "environment -- install it (`pip install torch`) or use "
            "backend='numpy' (the default) instead.")


def _vec_f(M):
    """Column-major ("Fortran") vectorization of a 2-D torch tensor M,
    via M.T.reshape(-1) -- torch has no native order='F' reshape, but
    flattening the TRANSPOSE in the default row-major order produces
    exactly the same element sequence as column-major-flattening M
    itself. This is the vectorization convention the classical
    identities vec(AX) = (I kron A) vec(X), vec(XB) = (B^T kron I)
    vec(X) assume -- see torch_solve_continuous_lyapunov()/torch_
    solve_sylvester() below for where this actually gets used."""
    return M.T.reshape(-1)


def _unvec_f(y, n_rows, n_cols):
    """Inverse of _vec_f(): recovers an (n_rows, n_cols) tensor from
    its column-major vectorization y (length n_rows*n_cols) -- reshape
    row-major to (n_cols, n_rows) (undoing the transpose-then-flatten
    _vec_f() did) then transpose back."""
    return y.reshape(n_cols, n_rows).T


def torch_solve_continuous_lyapunov(A, Q, device="cpu", dtype=None):
    """Torch-native solve of A X + X A^T = Q for X -- the SAME equation
    scipy.linalg.solve_continuous_lyapunov(A, Q) solves, same sign
    convention, so callers can swap backends with no other change. No
    native torch Lyapunov solver exists, so this uses the classical
    Kronecker-sum vectorization instead of scipy's dense Bartels-
    Stewart algorithm: vec_F(A X + X A^T) = (I kron A + A kron I)
    vec_F(X) = vec_F(Q), one dense (n^2, n^2) linear solve. This is a
    genuinely DIFFERENT numerical method computing the same
    mathematical answer (cross-validated against scipy's own
    solve_continuous_lyapunov() in tests/test_torch_linalg.py, not
    assumed identical) -- and a genuinely more expensive one per call
    (O(n^6) vs. Bartels-Stewart's O(n^3) in principle), an honest
    trade accepted here the same way fea_engine/iterative_solvers.py's
    ssor_preconditioner_torch() accepts densifying its triangular
    factors: no broadly-available, stable torch-native Bartels-Stewart
    equivalent exists, and the (n^2, n^2) dense solve this module falls
    back to is still the practically faster wall-clock choice on a GPU
    for the moderate n this package's own pre-reduced state spaces
    target (see balanced_truncation.py's own "PRACTICAL SCALE NOTE").

    Returns a torch.Tensor (n, n) on `device` -- NOT converted back to
    NumPy here; see balanced_truncation.py's own controllability_
    gramian()/observability_gramian() for where that conversion (or
    lack of it) happens."""
    _require_torch()
    dtype = dtype or torch.float64
    # .contiguous() right after construction, not just where this
    # module's OWN code happens to apply a .T: torch.kron()'s internal
    # implementation uses .view() on its operands (requires
    # contiguity), and torch.as_tensor()/torch.from_numpy() on a
    # NON-contiguous NumPy array (e.g. a caller passing A.T directly,
    # as balanced_truncation.observability_gramian() does) preserves
    # that non-contiguous stride rather than copying -- caught for real
    # on a torch-equipped machine via exactly that caller. Forcing
    # contiguity here, at the one shared entry point, is more robust
    # than requiring every caller to know this.
    A_t = torch.as_tensor(np.asarray(A, dtype=float), dtype=dtype, device=device).contiguous()
    Q_t = torch.as_tensor(np.asarray(Q, dtype=float), dtype=dtype, device=device).contiguous()
    n = A_t.shape[0]
    I_n = torch.eye(n, dtype=dtype, device=device)
    M = torch.kron(I_n, A_t) + torch.kron(A_t, I_n)
    y = torch.linalg.solve(M, _vec_f(Q_t))
    return _unvec_f(y, n, n)


def torch_solve_sylvester(A, B, Q, device="cpu", dtype=None):
    """Torch-native solve of A X + X B = Q for X -- the SAME equation
    scipy.linalg.solve_sylvester(A, B, Q) solves, including its
    generally-RECTANGULAR case (A is (m, m), B is (p, p), X and Q are
    (m, p)) -- same Kronecker-vectorization method as torch_solve_
    continuous_lyapunov() above (a genuine generalization of it: the
    Lyapunov equation is the special case B = A^T, Q symmetric).
    Provided here as a general-purpose reusable primitive alongside
    torch_solve_continuous_lyapunov() -- validated directly against
    scipy.linalg.solve_sylvester() in tests/test_torch_linalg.py -- but
    NOT currently called by any backend="torch" path elsewhere in this
    package: hankel_norm.py's own Schur-plus-Sylvester decoupling step
    operates on the small, r-sized reduced system (see this module's
    own docstring for why that makes it a low-value GPU target
    regardless of the full-order model's size). Kept available for a
    future item that does need it at full-order scale, the same
    "build the honest primitive even if not every caller reaches it
    yet" reasoning fea_engine/torch_sparse_solver.py's own _cg_torch()
    followed (item 3 wrote it generically so a later caller, item 137,
    could reuse it without touching the loop again)."""
    _require_torch()
    dtype = dtype or torch.float64
    # .contiguous() right after construction -- see torch_solve_
    # continuous_lyapunov()'s own comment: torch.kron()'s internal
    # implementation uses .view() on its operands (requires
    # contiguity), and both a caller passing an already-transposed
    # NumPy array AND this function's own B_t.T below can produce a
    # non-contiguous tensor; both are guarded against here rather than
    # relying on every call site to know this.
    A_t = torch.as_tensor(np.asarray(A, dtype=float), dtype=dtype, device=device).contiguous()
    B_t = torch.as_tensor(np.asarray(B, dtype=float), dtype=dtype, device=device).contiguous()
    Q_t = torch.as_tensor(np.asarray(Q, dtype=float), dtype=dtype, device=device).contiguous()
    m = A_t.shape[0]
    p = B_t.shape[0]
    I_m = torch.eye(m, dtype=dtype, device=device)
    I_p = torch.eye(p, dtype=dtype, device=device)
    M = torch.kron(I_p, A_t) + torch.kron(B_t.T.contiguous(), I_m)
    y = torch.linalg.solve(M, _vec_f(Q_t))
    return _unvec_f(y, m, p)


def torch_gramian_square_root(P, device="cpu", dtype=None,
                               jitter_tries=(0.0, 1e-12, 1e-9, 1e-6)):
    """Torch-native counterpart of balanced_truncation._gramian_square_
    root() -- a factor L with P = L @ L.T, via Cholesky first (with the
    SAME small jitter ladder for numerical-noise-induced non-positive-
    definiteness) then an eigh-based fallback (clip negative
    eigenvalues to zero -- expected near-zero-eigenvalue behavior for a
    nearly uncontrollable/unobservable state, not a numerical failure
    to hide, exactly as the NumPy version's own docstring explains) if
    every jittered Cholesky attempt fails. Returns a torch.Tensor
    (n, n) on `device`."""
    _require_torch()
    dtype = dtype or torch.float64
    P_t = torch.as_tensor(np.asarray(P, dtype=float), dtype=dtype, device=device)
    P_t = 0.5 * (P_t + P_t.T)
    n = P_t.shape[0]
    I_n = torch.eye(n, dtype=dtype, device=device)
    for jitter in jitter_tries:
        try:
            return torch.linalg.cholesky(P_t + jitter * I_n)
        except Exception:
            continue
    eigvals, eigvecs = torch.linalg.eigh(P_t)
    eigvals = torch.clamp(eigvals, min=0.0)
    return eigvecs @ torch.diag(torch.sqrt(eigvals))


def torch_balance_from_gramians(P, Q, device="cpu", dtype=None):
    """Torch-native counterpart of balanced_truncation._balance_from_
    gramians() -- the shared square-root balancing step (Gramian
    square roots, then an SVD of Lq^T Lp), same formulas, same return
    convention (sigma descending, T, Tinv) -- but returns plain NumPy
    arrays (NOT torch tensors). Unlike fea_engine's mesh_transform.py/
    vectorized_assembly.py GPU-residency chain (Wave 9 items 135/136,
    where MeshTransformation deliberately keeps its output as live
    torch.Tensors because vectorized_assembly.py's own functions
    consume them directly), nothing downstream of this function in
    rom_engine is torch-native -- BalancedTruncationROM/
    SingularPerturbationROM/FrequencyWeightedBalancedTruncationROM/
    OptimalHankelNormROM's own __init__ methods are all plain NumPy --
    so keeping the result on-device would only add a round trip back
    later, not avoid one now. See this module's own docstring."""
    Lp = torch_gramian_square_root(P, device=device, dtype=dtype)
    Lq = torch_gramian_square_root(Q, device=device, dtype=dtype)
    U, sigma, Vt = torch.linalg.svd(Lq.T @ Lp)
    sigma_safe = torch.where(sigma > 1e-300, sigma, torch.full_like(sigma, 1e-300))
    T = Lp @ Vt.T @ torch.diag(sigma_safe ** -0.5)
    Tinv = torch.diag(sigma_safe ** -0.5) @ U.T @ Lq.T
    return sigma.cpu().numpy(), T.cpu().numpy(), Tinv.cpu().numpy()
