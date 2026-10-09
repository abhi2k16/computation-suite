"""
topopt.py -- Wave 11 item 111 (docs/consolidated_future_roadmap.md,
source tensormesh_analysis_report.md's "Differentiability" section,
`OCOptimizer`: density-per-element, SIMP stiffness interpolation,
volume-fraction constraint via bisection on a Lagrange multiplier,
driven directly by `rho.grad` from the adjoint solve -- demonstrated
there on ~8.6x compliance reduction at an exactly-held volume
constraint).

Hard dependency, now satisfied: Wave 10 item 98 (the whole-solver
implicit-differentiation adjoint layer -- `differentiable.
adjoint_gradient()`). This module's own `compliance_sensitivity_via_
adjoint()` is the concrete "does the adjoint layer actually work
end-to-end, on a problem genuinely different from the one it was built
for" validation this item exists to perform -- the same role
`OCOptimizer` plays for TensorMesh itself -- by reusing `adjoint_
gradient(K_eff, grad_w, vjp_fn)`'s generic signature DIRECTLY, not by
re-deriving a new adjoint mechanism.

Two independent, cross-validated sensitivity paths (the "numpy-shadow-
first" pattern Wave 10 established -- see differentiable.py's own
module docstring): compliance topology optimization is the textbook
SELF-ADJOINT case (the adjoint variable lam turns out to equal -u
exactly, a well-known closed-form fact -- see `compliance_sensitivity_
closed_form()`'s own docstring for the derivation), so this module
gets BOTH a fast, torch-free, hand-derived closed-form sensitivity
(the one the actual optimization LOOP below uses by default) AND a
slower, torch-gated path that computes the exact same quantity through
`adjoint_gradient()` plus a genuine `torch.autograd.grad` vector-
Jacobian product -- the two must agree to floating-point/autograd
precision, which `tests/test_topopt.py` checks directly, not merely
assumed.

Integration policy: no `tensormesh.optimizer` dependency -- native
NumPy/`torch` autograd only, reusing this package's own Wave 11 items
106/107 (`MeshTransformation`/`tensorized_element_stiffness`) to build
the SINGLE "unit-density" element stiffness stack once, then rescaling
it every OC iteration by a cheap per-element SIMP scalar (`vectorized_
assembly.scatter_global_stiffness()` handles the actual global
scatter) -- SIMP's own standard trick, and a natural, concrete use case
for items 106/107's own tensorized-assembly machinery.

Scope: SIMP density-per-element compliance minimization under a single
volume-fraction constraint, Optimality-Criteria updates with a move
limit and bisection on the volume multiplier -- the exact scope
`OCOptimizer` itself covers, nothing more (no multi-load-case
robustness, no stress constraints, no multi-material). Restricted to
the same continuum-elasticity element types `MeshTransformation`/
`tensorized_element_stiffness` already support (see mesh_transform.py's
own Scope docstring) -- in practice, `Quad4PlaneStress`/`Hex8Solid3D`
meshes. An OPTIONAL, simple radius-based sensitivity filter (Sigmund's
classic heuristic, "A 99 line topology optimization code written in
Matlab", 2001) is included to avoid obvious checkerboarding -- density
filtering (a genuinely different, arguably better-founded technique) is
a natural follow-on, not attempted here.
"""
__author__ = "Abhijeet"
import numpy as np

from .solver import FESystem
from .mesh_transform import MeshTransformation
from .vectorized_assembly import tensorized_element_stiffness, scatter_global_stiffness
from .differentiable import _HAS_TORCH, _require_torch, adjoint_gradient

if _HAS_TORCH:
    import torch


# =====================================================================
# SIMP interpolation
# =====================================================================
def simp_scale(rho, p=3.0, rho_min=1e-3):
    """Per-element SIMP stiffness scale factor: `s(rho) = rho_min +
    (1 - rho_min) * rho**p`. `rho_min` (NOT the more common `Emin/E0`
    notation, same idea) keeps every element's stiffness strictly
    positive even at `rho=0` -- the standard fix (Sigmund 2001) for
    the singular-stiffness-matrix problem a literal `rho**p` (allowing
    exactly 0) would otherwise cause in void regions."""
    rho = np.asarray(rho, dtype=float)
    return rho_min + (1.0 - rho_min) * rho ** p


def simp_scale_grad(rho, p=3.0, rho_min=1e-3):
    """d(simp_scale)/d(rho) = p * (1 - rho_min) * rho**(p-1) -- the
    closed-form derivative `compliance_sensitivity_closed_form()` uses
    directly (no autograd needed for this scalar 1-D function)."""
    rho = np.asarray(rho, dtype=float)
    return p * (1.0 - rho_min) * rho ** (p - 1.0)


# =====================================================================
# Global stiffness from a precomputed unit-density element stack
# =====================================================================
def unit_stiffness_stack(fesystem, D, gauss_order=None):
    """The mesh-wide `(n_elements, n_edof, n_edof)` stiffness stack at
    `rho=1` everywhere -- computed ONCE via items 106/107
    (`MeshTransformation` + `tensorized_element_stiffness`), then
    reused (just rescaled per element, every OC iteration) by
    `assemble_K_from_scale()` below instead of ever reassembling from
    element geometry again."""
    mt = MeshTransformation(fesystem.mesh, fesystem.elem, gauss_order=gauss_order)
    ke_unit = tensorized_element_stiffness(mt, D)
    return ke_unit, mt.connectivity


def assemble_K_from_scale(ke_unit, scale, connectivity, dofs_per_node, n_dof):
    """Global K(rho) via item 107's vectorized scatter, given the
    already-rescaled per-element stiffness stack `ke_unit * scale[:,
    None, None]` -- one call, no Python loop over elements."""
    ke_scaled = ke_unit * np.asarray(scale)[:, None, None]
    return scatter_global_stiffness(ke_scaled, connectivity, dofs_per_node, n_dof, sparse=False)


def _elem_dof_table(connectivity, dofs_per_node):
    n_elements, n_basis = connectivity.shape
    return (dofs_per_node * connectivity[:, :, None] +
            np.arange(dofs_per_node)[None, None, :]).reshape(n_elements, n_basis * dofs_per_node)


# =====================================================================
# Forward solve
# =====================================================================
def solve_simp_equilibrium(ke_unit, connectivity, dofs_per_node, n_dof, free_dofs, F_ext, rho, p, rho_min):
    """K(rho) u = F_ext on the free dofs, via the SAME SPD-aware
    Cholesky-then-eigen dispatch `FESystem.solve_static()` itself uses
    (Wave 0 items 5/6) -- reused directly (both are `@staticmethod`),
    not reimplemented, since a SIMP density field genuinely can produce
    a near-singular Kff (a fully-void, disconnected region) that the
    existing fallback chain already knows how to handle safely.

    Returns (u_full, Kff, Ff) -- Kff/Ff are handed back so callers
    (compliance_sensitivity_via_adjoint()) don't need to reassemble/
    re-slice them a second time.
    """
    scale = simp_scale(rho, p, rho_min)
    K = assemble_K_from_scale(ke_unit, scale, connectivity, dofs_per_node, n_dof)
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Ff = F_ext[free_dofs]
    try:
        Uf = FESystem._dense_spd_solve(Kff, Ff)
    except np.linalg.LinAlgError:
        Uf = FESystem._eigen_solve(Kff, Ff)
    u_full = np.zeros(n_dof)
    u_full[free_dofs] = Uf
    return u_full, Kff, Ff


def compliance(F_ext, u_full):
    """C = F_ext . u -- the SIMP objective, a plain dot product (F_ext
    is zero at every free dof the load isn't applied to and u is zero
    at every fixed dof by construction, so this is exactly F^T u over
    the whole vector, no free/fixed split needed)."""
    return float(np.dot(F_ext, u_full))


# =====================================================================
# Sensitivities -- two independent, cross-validated paths
# =====================================================================
def compliance_sensitivity_closed_form(ke_unit, connectivity, dofs_per_node, u_full, rho, p, rho_min):
    """dC/drho_e = -simp_scale_grad(rho_e) * u_e^T @ ke_unit_e @ u_e --
    the textbook self-adjoint compliance-sensitivity formula (e.g.
    Bendsoe & Sigmund 2003, Sec.1.3): for C = F^T u subject to K(rho)u=F
    with F independent of rho, the adjoint variable lam solving
    K^T lam = -F is EXACTLY lam = -u (K symmetric, K u = F => lam =
    -K^-1 F = -u, no extra solve needed), giving dC/drho_e =
    lam^T (dK_e/drho_e) u_e = -u_e^T (dK_e/drho_e) u_e directly. Pure
    NumPy, no torch -- the fast path the actual OC loop below uses by
    default, and the decisive cross-check target for
    compliance_sensitivity_via_adjoint()'s slower, general-purpose
    torch path."""
    g = _elem_dof_table(connectivity, dofs_per_node)
    u_e = u_full[g]                                      # (n_elements, n_edof)
    quad = np.einsum('ei,eij,ej->e', u_e, ke_unit, u_e)   # u_e^T ke_unit_e u_e
    dscale = simp_scale_grad(rho, p, rho_min)
    return -dscale * quad


def compliance_sensitivity_via_adjoint(ke_unit, connectivity, dofs_per_node, n_dof,
                                        free_dofs, u_full, Kff, Ff, rho, p, rho_min):
    """The SAME dC/drho as compliance_sensitivity_closed_form(), computed
    instead through Wave 10 item 98's own general machinery --
    differentiable.adjoint_gradient(K_eff, grad_w, vjp_fn) -- reused
    here EXACTLY as-is, with:
      K_eff = Kff        (this problem is LINEAR, so the "corrected
                           tangent" IS just the plain stiffness -- no
                           AdditiveCorrection/f_theta term needed, and
                           none is constructed)
      grad_w = Ff         (dC/du|_free, since C = F^T u is linear in u
                           for fixed rho)
      vjp_fn(lam) = a genuine torch.autograd.grad vector-Jacobian
                    product of rho -> (K(rho) @ u_full)[free_dofs],
                    with u_full held CONSTANT (matching the implicit-
                    function-theorem derivation adjoint_gradient()'s
                    own docstring spells out: theta enters the root
                    condition only through the "f_theta"-like term,
                    here the whole K(rho)@u itself).

    This is this item's actual reason for existing: a real, independent
    demonstration that item 98's adjoint layer, built for a nonlinear
    structural-correction problem, is genuinely general enough to
    reproduce a textbook closed-form topology-optimization sensitivity
    exactly -- see tests/test_topopt.py for the decisive agreement
    check against compliance_sensitivity_closed_form()."""
    _require_torch()
    g_np = _elem_dof_table(connectivity, dofs_per_node)
    g_t = torch.as_tensor(g_np, dtype=torch.long)
    free_t = torch.as_tensor(free_dofs, dtype=torch.long)
    u_full_t = torch.as_tensor(u_full, dtype=torch.float64)   # CONSTANT (no grad)
    ke_unit_t = torch.as_tensor(ke_unit, dtype=torch.float64)
    rho_t = torch.as_tensor(rho, dtype=torch.float64).requires_grad_(True)

    def _Ku_free(rho_var):
        scale_t = rho_min + (1.0 - rho_min) * rho_var ** p       # (n_elements,)
        u_e_t = u_full_t[g_t]                                     # (n_elements, n_edof)
        Ku_e_t = torch.einsum('e,eij,ej->ei', scale_t, ke_unit_t, u_e_t)
        w_full_t = torch.zeros(n_dof, dtype=torch.float64)
        w_full_t = w_full_t.index_add(0, g_t.reshape(-1), Ku_e_t.reshape(-1))
        return w_full_t[free_t]

    def vjp_fn(lam):
        lam_t = torch.as_tensor(lam, dtype=torch.float64)
        w_free_t = _Ku_free(rho_t)
        scalar = torch.dot(lam_t, w_free_t)
        grad_rho, = torch.autograd.grad(scalar, rho_t)
        return grad_rho.detach().cpu().numpy()

    return adjoint_gradient(Kff, Ff, vjp_fn)


# =====================================================================
# Optional sensitivity filter (Sigmund 2001)
# =====================================================================
def build_filter_weights(centroids, radius):
    """Sparse (dense-array-returned, meshes here are small) weight
    matrix `H[e,j] = max(0, radius - dist(e,j))` for every pair within
    `radius` -- Sigmund's original linear-decay filter kernel."""
    n = centroids.shape[0]
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    H = np.maximum(0.0, radius - dist)
    return H


def apply_sensitivity_filter(H, rho, dc, eps=1e-9):
    """Sigmund's original (1999/2001) heuristic sensitivity filter:
    `dc_tilde_e = (1 / (rho_e * sum_j H_ej)) * sum_j H_ej * rho_j * dc_j`
    -- damps checkerboarding without a separate density-filtering pass.
    """
    denom = np.maximum(rho, eps) * H.sum(axis=1)
    return (H @ (rho * dc)) / np.maximum(denom, eps)


# =====================================================================
# Optimality-Criteria update
# =====================================================================
def oc_update(rho, dc, volume_fraction, move=0.2, rho_min_bound=1e-3, rho_max_bound=1.0,
              bisection_tol=1e-4, max_bisection_iter=100):
    """Classic Optimality-Criteria density update (Bendsoe 1995;
    Sigmund's "99 line" code): for a fixed trial Lagrange multiplier
    `lmid` on the volume constraint,

        rho_new_e = clip(rho_e * sqrt(-dc_e / lmid), rho_e-move, rho_e+move)

    clipped again to `[rho_min_bound, rho_max_bound]`, then `lmid` is
    bisected until `mean(rho_new) == volume_fraction` to within
    `bisection_tol`. `dc` must be <= 0 everywhere (more material always
    helps compliance) -- a positive `dc_e` is clipped to a small
    negative epsilon rather than raising, since a converged-optimum
    element can have `dc_e` numerically indistinguishable from zero.
    """
    dc = np.minimum(dc, -1e-12)   # enforce the sign compliance sensitivity must have
    l1, l2 = 1e-12, 1.0
    # expand the bracket until it's wide enough to contain a root
    # (classic robust-bisection pattern -- the "99 line" code's own
    # fixed [0, 1e9] bracket assumes a particular dc/rho scaling this
    # module doesn't force on the caller, so this expands adaptively
    # instead of assuming a scale).
    for _ in range(60):
        rho_try = np.clip(rho * np.sqrt(-dc / l2), rho - move, rho + move)
        rho_try = np.clip(rho_try, rho_min_bound, rho_max_bound)
        if rho_try.mean() <= volume_fraction:
            break
        l2 *= 2.0
    for _ in range(max_bisection_iter):
        lmid = 0.5 * (l1 + l2)
        rho_new = np.clip(rho * np.sqrt(-dc / lmid), rho - move, rho + move)
        rho_new = np.clip(rho_new, rho_min_bound, rho_max_bound)
        if rho_new.mean() > volume_fraction:
            l1 = lmid
        else:
            l2 = lmid
        if (l2 - l1) < bisection_tol * max(l2, 1.0):
            break
    return rho_new


# =====================================================================
# Orchestrator
# =====================================================================
def topology_optimize_compliance(fesystem, D, F_ext, volume_fraction, p=3.0, rho_min=1e-3,
                                  move=0.2, n_iter=50, filter_radius=None,
                                  sensitivity="closed_form", rho_init=None, tol=1e-3):
    """Full SIMP compliance-minimization loop: solve -> sensitivity ->
    (optional filter) -> Optimality-Criteria update -> repeat.

    Parameters
    ----------
    fesystem : FESystem
        Used ONLY for `.mesh`, `.elem` (the single element formulation
        -- multi-block meshes are out of scope), `.n_dof`, `.free_dofs`
        (i.e. whatever `fix_dofs()` calls were already made). Its own
        `.K` is never assembled/read by this function.
    D : (n_strain, n_strain) ndarray
        Base (rho=1) material matrix -- same object `assemble_stiffness`
        would take, passed straight to `unit_stiffness_stack()`.
    F_ext : (n_dof,) ndarray
        The load case (same convention as `fesystem.F`).
    volume_fraction : float in (0, 1]
        Target `mean(rho)`.
    sensitivity : "closed_form" (default, fast, torch-free) or
        "adjoint" (Wave 10 item 98's general machinery, torch-gated --
        see compliance_sensitivity_via_adjoint()'s own docstring).
        Both are validated to agree in tests/test_topopt.py; "adjoint"
        exists to prove that agreement holds, not because it is faster
        or more accurate for this particular (self-adjoint) problem.

    Returns
    -------
    dict with keys "rho" (final density field), "compliance_history"
    (list, one entry per iteration), "u_final" (equilibrium
    displacement at the final rho).
    """
    if sensitivity not in ("closed_form", "adjoint"):
        raise ValueError(
            f"topology_optimize_compliance: unknown sensitivity={sensitivity!r} "
            f"-- expected 'closed_form' or 'adjoint'.")
    if sensitivity == "adjoint":
        _require_torch()

    ke_unit, connectivity = unit_stiffness_stack(fesystem, D)
    dofs_per_node = fesystem.elem.dofs_per_node
    n_dof = fesystem.n_dof
    free_dofs = fesystem.free_dofs
    F_ext = np.asarray(F_ext, dtype=float)

    n_elements = connectivity.shape[0]
    rho = np.full(n_elements, volume_fraction) if rho_init is None else np.array(rho_init, dtype=float)

    H = None
    if filter_radius is not None and filter_radius > 0:
        centroids = fesystem.mesh.nodes[connectivity].mean(axis=1)
        H = build_filter_weights(centroids, filter_radius)

    history = []
    u_full = None
    for it in range(n_iter):
        u_full, Kff, Ff = solve_simp_equilibrium(
            ke_unit, connectivity, dofs_per_node, n_dof, free_dofs, F_ext, rho, p, rho_min)
        C = compliance(F_ext, u_full)
        history.append(C)

        if sensitivity == "closed_form":
            dc = compliance_sensitivity_closed_form(ke_unit, connectivity, dofs_per_node, u_full, rho, p, rho_min)
        else:
            dc = compliance_sensitivity_via_adjoint(
                ke_unit, connectivity, dofs_per_node, n_dof, free_dofs, u_full, Kff, Ff, rho, p, rho_min)

        if H is not None:
            dc = apply_sensitivity_filter(H, rho, dc)

        rho_new = oc_update(rho, dc, volume_fraction, move=move, rho_min_bound=rho_min)
        change = float(np.max(np.abs(rho_new - rho)))
        rho = rho_new
        if change < tol:
            break

    return {"rho": rho, "compliance_history": history, "u_final": u_full}
