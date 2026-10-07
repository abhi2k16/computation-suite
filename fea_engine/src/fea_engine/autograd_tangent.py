"""
autograd_tangent.py -- Wave 0 item 2 (docs/consolidated_future_
roadmap.md, source tensormesh_comparative_analysis.md Section 6.3): an
independent, PyTorch-autograd-based cross-check of the element tangent
stiffnesses this package's nonlinear solid elements already compute by
other means -- Tet4NeoHookean's central finite difference and
Tet10SolidTL's analytic (hand-derived) tangent (see elements/
nonlinear_solids.py's own module docstring and each class's
tangent_stiffness() docstring for what those existing methods do and
why).

Deviation from the roadmap doc's suggested path (`fea_engine/
materials/autograd_tangent.py`, implying a new `materials/`
subpackage): this codebase's actual convention is a flat module per
concern (material.py, solver.py, mesh.py, ... -- no subpackages except
elements/ and geometry/, both genuinely multi-file families). A single
new file with no siblings doesn't warrant a new subpackage, so this
lives at `fea_engine/autograd_tangent.py` instead, alongside
material.py it cross-checks.

WHY an autograd cross-check adds real value on top of the existing
finite-difference/complex-step tangents: autograd computes the EXACT
derivative of whatever function is expressed in code (no h-dependent
truncation error the way central FD has, and no complex-step-specific
assumptions about analytic continuation) -- so agreement between an
independently re-expressed torch computation and this package's numpy
one is strong evidence BOTH the stress law (neo_hookean_pk2_stress /
the St. Venant-Kirchhoff Green-Lagrange-strain law) and the existing
tangent are correct, not just self-consistent. Disagreement would
point at a real bug rather than at FD's known noise floor (see
Tet10SolidTL's own docstring for a documented case where an FD/
complex-step noise floor was initially, and incorrectly, blamed for an
unrelated Newton-convergence issue -- exactly the kind of ambiguity an
exact third method resolves).

SCOPE: this module ONLY validates existing physics, exactly like the
roadmap item says ("adds no new physics") -- it is not wired into any
solve path, element, or FESystem. `torch` stays an OPTIONAL dependency
(the codebase's established pattern, previously also used for gmsh
before Gmsh-backed geometry support was removed from this package --
see docs/generalized_mesh_grading_roadmap.md -- via
`_HAS_GMSH`/`_require_gmsh()`) mirrored here as
`_HAS_TORCH`/`_require_torch()`): importing this module never requires
torch to be installed, only CALLING its functions does. Per the user's
explicit instruction earlier in this project's development, this
module uses bare `torch` ops only -- no `tensormesh`/`torch-sla`
package import anywhere, TensorMesh's own docs were consulted only as
design inspiration for §6.3's suggested approach.
"""
import numpy as np

from .elements.base import jacobian, tet_quadrature_4pt, gauss_product
from .elements.solids import Tet4Solid3D
from .material import D_solid3d

try:
    import torch
    _HAS_TORCH = True
except Exception:
    # Deliberately a bare `except Exception`, not `except ImportError`:
    # a torch WHEEL can be installed (pip succeeds) yet still fail to
    # actually import for reasons that surface as OSError/ValueError
    # rather than ImportError -- e.g. the default PyPI `torch` wheel is
    # CUDA-linked and raises ValueError/OSError hunting for
    # libcudart.so/libcublasLt.so etc. at import time on a machine
    # with no CUDA runtime installed (confirmed directly: this is
    # exactly what happens in this project's own sandbox, where a
    # full-size torch wheel was successfully downloaded and pip-
    # installed but is NOT importable for this reason -- see this
    # module's own docstring). Any exception here means torch is not
    # usable, full stop, and _HAS_TORCH=False is the correct fallback
    # regardless of the specific cause.
    _HAS_TORCH = False


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "autograd_tangent.py requires the 'torch' package: pip install "
            "torch -- this, like gmsh formerly was (Gmsh-backed geometry "
            "support has since been removed from fea_engine), is an "
            "OPTIONAL dependency of fea_engine; every other module works "
            "without it. If you only need to run this package's test suite "
            "in an environment where torch isn't usable, the tests in "
            "tests/test_autograd_tangent.py skip themselves cleanly (gated "
            "on this module's own _HAS_TORCH, not a bare "
            "pytest.importorskip('torch') -- see that test file's own "
            "docstring for why) rather than failing.")


# =====================================================================
# Tet4NeoHookean cross-check
# =====================================================================
def _neo_hookean_pk2_stress_torch(F, mu, kappa):
    """torch re-expression of material.neo_hookean_pk2_stress(), kept
    op-for-op identical to that function's numpy formula (same variable
    names, same order of operations) so any numerical divergence
    between the two reflects a real discrepancy, not two different
    but-mathematically-equivalent ways of writing the same law."""
    C = F.T @ F
    J = torch.linalg.det(F)
    I1 = torch.trace(C)
    Cinv = torch.linalg.inv(C)
    eye3 = torch.eye(3, dtype=F.dtype)

    S = mu * J ** (-2.0 / 3.0) * (eye3 - (I1 / 3.0) * Cinv) + kappa * J * (J - 1.0) * Cinv
    W = 0.5 * mu * (J ** (-2.0 / 3.0) * I1 - 3.0) + 0.5 * kappa * (J - 1.0) ** 2
    return S, W


def _tet4_neo_hookean_internal_force_torch(dN_dX_t, volume, u_t, mu, kappa):
    """dN_dX_t: (3,4) torch tensor, reference-configuration shape
    gradients -- CONSTANT with respect to u (computed once, in plain
    numpy, from elem_coords alone -- see
    tet4_neo_hookean_tangent_autograd() below, exactly mirroring
    Tet4NeoHookean._deformation_gradient()'s own numpy computation) so
    no gradient needs to (or should) flow through it. volume: python
    float, V0. u_t: (12,) torch tensor with requires_grad=True,
    node-major dof order (u_x0,u_y0,u_z0,u_x1,...), matching
    B_matrix()'s convention used throughout this package. Returns
    f_int (12,) torch tensor, differentiable w.r.t. u_t -- the SAME
    closed-form Tet4NeoHookean.internal_force() computes."""
    u_nodes = u_t.reshape(4, 3)
    F = torch.eye(3, dtype=u_t.dtype) + u_nodes.T @ dN_dX_t.T
    S, _ = _neo_hookean_pk2_stress_torch(F, mu, kappa)
    f_int_nodes = volume * (F @ S @ dN_dX_t)   # (3,4): [dof, node]
    return f_int_nodes.T.reshape(-1)           # node-major (12,)


def tet4_neo_hookean_tangent_autograd(elem_coords, u_elem, mat):
    """Independent torch.autograd.grad tangent stiffness for
    Tet4NeoHookean, cross-checking Tet4NeoHookean.tangent_stiffness()'s
    existing central-finite-difference result (see that method's
    docstring for why FD was chosen there over a hand-derived analytic
    tangent). elem_coords: (4,3) array-like. u_elem: (12,) array-like
    (node-major). mat: material.NeoHookeanMaterial.

    Returns (K (12,12) numpy array, f_int (12,) numpy array) -- K via
    row-by-row torch.autograd.grad of f_int w.r.t. u_elem (exact to
    float64 machine precision, no h-dependent truncation), f_int
    returned too since it is computed as a genuine side effect of
    building K and is itself a free cross-check of Tet4NeoHookean.
    internal_force() (same closed-form expression, re-derived
    independently in torch).

    Does NOT symmetrize K the way Tet4NeoHookean.tangent_stiffness()
    explicitly does for its FD result (see that method's own
    docstring) -- an exact autograd Hessian-of-a-scalar-potential is
    already symmetric to float64 round-off, so an asymmetry here would
    itself be a meaningful diagnostic, not expected noise to paper
    over."""
    _require_torch()
    elem_coords = np.asarray(elem_coords, dtype=float)
    u_elem = np.asarray(u_elem, dtype=float)

    _, dN_nat = Tet4Solid3D().shape_and_derivs((0.25, 0.25, 0.25))
    J_ref, detJ = jacobian(dN_nat, elem_coords)
    dN_dX = np.linalg.solve(J_ref, dN_nat)
    volume = abs(detJ) / 6.0

    dN_dX_t = torch.tensor(dN_dX, dtype=torch.float64)
    u_t = torch.tensor(u_elem, dtype=torch.float64, requires_grad=True)

    f_int_t = _tet4_neo_hookean_internal_force_torch(dN_dX_t, volume, u_t, mat.mu, mat.kappa)

    n = f_int_t.shape[0]
    K = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        (grad_i,) = torch.autograd.grad(f_int_t[i], u_t, retain_graph=True)
        K[i, :] = grad_i

    return K.detach().numpy(), f_int_t.detach().numpy()


# =====================================================================
# Tet10SolidTL cross-check
# =====================================================================
def _green_lagrange_pk2_torch(F, D_t):
    """torch re-expression of Tet10SolidTL._green_lagrange_pk2(), kept
    op-for-op identical to that method's numpy version (same Voigt
    ordering/engineering-shear convention) for the same "identical
    reimplementation, different differentiation mechanism" reason as
    _neo_hookean_pk2_stress_torch() above."""
    eye3 = torch.eye(3, dtype=F.dtype)
    E = 0.5 * (F.T @ F - eye3)
    E_voigt = torch.stack([
        E[0, 0], E[1, 1], E[2, 2],
        2 * E[0, 1], 2 * E[1, 2], 2 * E[0, 2],
    ])
    S_voigt = D_t @ E_voigt
    S = torch.stack([
        torch.stack([S_voigt[0], S_voigt[3], S_voigt[5]]),
        torch.stack([S_voigt[3], S_voigt[1], S_voigt[4]]),
        torch.stack([S_voigt[5], S_voigt[4], S_voigt[2]]),
    ])
    return S


def _tet10_solid_tl_internal_force_torch(dN_dX_list, weights, volume_factors, D_t, u_t):
    """dN_dX_list: list of 4 (3,10) torch tensors, one per Gauss point
    (CONSTANT w.r.t. u -- reference-configuration gradients, computed
    once in plain numpy exactly like Tet10SolidTL.internal_force()'s
    own per-Gauss-point loop). weights, volume_factors: python floats
    (tet_quadrature_4pt() weight and abs(detJ)/6.0 at each point,
    also constant w.r.t. u). D_t: (6,6) torch tensor, material.
    D_solid3d(mat) converted once. u_t: (30,) torch tensor,
    requires_grad=True, node-major. Returns f_int (30,) torch tensor,
    the SAME closed-form sum Tet10SolidTL.internal_force() computes."""
    u_nodes = u_t.reshape(10, 3)
    eye3 = torch.eye(3, dtype=u_t.dtype)
    f_int = torch.zeros(30, dtype=u_t.dtype)
    for dN_dX_t, w, vol_factor in zip(dN_dX_list, weights, volume_factors):
        F = eye3 + u_nodes.T @ dN_dX_t.T
        S = _green_lagrange_pk2_torch(F, D_t)
        f_int_nodes = F @ S @ dN_dX_t   # (3,10)
        f_int = f_int + f_int_nodes.T.reshape(-1) * vol_factor * w
    return f_int


def tet10_solid_tl_tangent_autograd(elem_coords, u_elem, mat, formulation=None):
    """Independent torch.autograd.grad tangent stiffness for
    Tet10SolidTL, cross-checking its ANALYTIC tangent_stiffness()
    (added 2026-09-02, see that method's own docstring for the
    derivation this validates) as well as the KEPT complex-step
    reference (_tangent_stiffness_complex_step()) -- a third,
    independently-implemented method agreeing with both is stronger
    evidence than either alone.

    elem_coords: (10,3) array-like. u_elem: (30,) array-like
    (node-major). mat: a plain material.Material (E, nu) -- St.
    Venant-Kirchhoff uses the SAME linear elastic D as
    Tet10Solid3D's own linear stiffness(), see nonlinear_solids.py's
    module docstring. formulation: an existing Tet10SolidTL instance
    to reuse for its cached shape-function evaluations (Wave 0 item 4,
    see elements/base.py's _cached_shape_and_derivs()) -- optional,
    a fresh instance is created if omitted since the cache is a pure
    performance optimization, not required for correctness.

    Returns (K (30,30) numpy array, f_int (30,) numpy array), same
    conventions as tet4_neo_hookean_tangent_autograd() above."""
    _require_torch()
    from .elements.nonlinear_solids import Tet10SolidTL

    elem_coords = np.asarray(elem_coords, dtype=float)
    u_elem = np.asarray(u_elem, dtype=float)
    if formulation is None:
        formulation = Tet10SolidTL()

    D = D_solid3d(mat)
    points, weights = tet_quadrature_4pt()
    dN_dX_list, volume_factors = [], []
    for p in points:
        _, dN_nat = formulation._cached_shape_and_derivs(p)
        J, detJ = jacobian(dN_nat, elem_coords)
        dN_dX = np.linalg.solve(J, dN_nat)
        dN_dX_list.append(torch.tensor(dN_dX, dtype=torch.float64))
        volume_factors.append(abs(detJ) / 6.0)

    D_t = torch.tensor(D, dtype=torch.float64)
    u_t = torch.tensor(u_elem, dtype=torch.float64, requires_grad=True)

    f_int_t = _tet10_solid_tl_internal_force_torch(dN_dX_list, weights, volume_factors, D_t, u_t)

    n = f_int_t.shape[0]
    K = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        (grad_i,) = torch.autograd.grad(f_int_t[i], u_t, retain_graph=True)
        K[i, :] = grad_i

    return K.detach().numpy(), f_int_t.detach().numpy()


# =====================================================================
# Wave 10 item 102 (docs/consolidated_future_roadmap.md): batched
# (torch.func.vmap/jacrev) multi-element autograd tangent assembly --
# a GPU-throughput item, independent of Wave 10's other items (98-101,
# fea_engine/differentiable.py). tet4_neo_hookean_tangent_autograd()
# above is exact but still a Python-level, one-element-at-a-time loop
# (and within each element, a row-by-row torch.autograd.grad loop) --
# no batching across elements, so backend="torch"'s GPU benefit for AD
# tangents specifically is currently limited to whatever a single
# 12x12 Jacobian's own row-by-row autograd.grad calls can exploit. This
# function evaluates the SAME per-element physics
# (_tet4_neo_hookean_internal_force_torch(), reused verbatim, not
# reimplemented) across a WHOLE BATCH of elements in one vectorized
# torch.func.vmap(jacrev(...)) call instead -- this is where a real,
# measurable GPU speedup would actually show up (today's per-element
# loop underutilizes a GPU badly); genuinely new engineering, not a
# re-validation of existing physics (see this function's own test for
# the numerical-equivalence check that IS the correctness claim here).
# Requires torch>=2.0 (torch.func.vmap/jacrev) -- gated the same way as
# every other torch-dependent function in this module, via
# _require_torch(); an old torch without torch.func raises a plain
# ImportError/AttributeError from the `from torch.func import ...` line
# itself, not silently misbehaving.
# =====================================================================
def tet4_neo_hookean_tangent_autograd_batched(elem_coords_batch, u_elem_batch, mat):
    """Batched counterpart to tet4_neo_hookean_tangent_autograd() above.

    elem_coords_batch: (n_elem, 4, 3) array-like.
    u_elem_batch: (n_elem, 12) array-like (node-major per element).
    mat: a single material.NeoHookeanMaterial, shared by every element
    in the batch (a homogeneous-material batch -- the common case
    within one FESystem block; matches Wave 11 item 107's own
    "homogeneous-element-type mesh region" scoping precedent).

    Returns (K_batch (n_elem,12,12), f_int_batch (n_elem,12)) numpy
    arrays -- validated (tests/test_autograd_tangent.py, torch-gated)
    to match calling tet4_neo_hookean_tangent_autograd() once per
    element in an ordinary Python loop, element by element, to float64
    machine precision: this function changes ONLY how the computation
    is SCHEDULED (one vectorized call vs. n_elem separate ones), never
    the physics or the resulting numbers."""
    _require_torch()
    from torch.func import vmap, jacrev

    elem_coords_batch = np.asarray(elem_coords_batch, dtype=float)
    u_elem_batch = np.asarray(u_elem_batch, dtype=float)
    n_elem = elem_coords_batch.shape[0]

    _, dN_nat = Tet4Solid3D().shape_and_derivs((0.25, 0.25, 0.25))
    dN_dX_batch = np.zeros((n_elem, 3, 4))
    volume_batch = np.zeros(n_elem)
    for e in range(n_elem):
        J_ref, detJ = jacobian(dN_nat, elem_coords_batch[e])
        dN_dX_batch[e] = np.linalg.solve(J_ref, dN_nat)
        volume_batch[e] = abs(detJ) / 6.0

    dN_dX_t = torch.tensor(dN_dX_batch, dtype=torch.float64)
    volume_t = torch.tensor(volume_batch, dtype=torch.float64)
    u_t = torch.tensor(u_elem_batch, dtype=torch.float64)
    mu, kappa = mat.mu, mat.kappa

    def f_int_single(u_e, dN_dX_e, volume_e):
        return _tet4_neo_hookean_internal_force_torch(dN_dX_e, volume_e, u_e, mu, kappa)

    f_int_batch_t = vmap(f_int_single)(u_t, dN_dX_t, volume_t)
    K_batch_t = vmap(jacrev(f_int_single, argnums=0))(u_t, dN_dX_t, volume_t)

    return K_batch_t.detach().numpy(), f_int_batch_t.detach().numpy()


# =====================================================================
# Shell4MITCCorotational cross-check (Wave 9 item 96, docs/consolidated_
# future_roadmap.md): an autograd replacement for the state=... branch's
# production tangent, _tangent_stiffness_committed_complex_step()
# (elements/shells.py) -- NOT a correctness fix (that method's own
# docstring already establishes complex-step recovers the EXACT
# derivative here, same as autograd would -- see this function's own
# docstring below for what this addition actually buys), but the same
# "keep every valid implementation available, explicit opt-in" principle
# every other backend/method choice in this package follows.
#
# SCOPE, DELIBERATELY NARROWED: only the iter_state=None path (the
# overwhelmingly common case -- this package's own 165+ pre-existing
# shell tests never call init_iter_state()) is ported here. The mixed-
# formulation iter_state=... branch changes _bending_membrane_coupling_
# force()'s Nb_offset/mean-subtraction logic (see that method's own
# docstring, "MIXED-FORMULATION MODE" paragraph) and was judged a
# separate, not-yet-attempted extension rather than something to bolt on
# without being able to execute-test it in this sandbox -- calling this
# function with iter_state not None raises NotImplementedError rather
# than silently computing the wrong thing.
#
# VALIDATION METHODOLOGY NOTE (this sandbox has no working torch
# install -- see this module's own _HAS_TORCH comment): before writing
# any torch code, the exact formula sequence below was first transcribed
# into a plain-numpy "shadow" mirror (same variable names, same
# operation order, np. in place of torch.) and run against the REAL
# Shell4MITCCorotational.internal_force(..., state=...) production
# method on a real committed-state trial (tests/test_shell_corotational_
# committed_coupling.py's own RECT fixture, a random post-commit
# rotation + translation state) -- agreement was 1.25e-16 relative
# (float64 machine precision), confirming the derivation itself
# (including the block-diagonal-rotation-as-reshape identity below) is
# correct BEFORE any torch-specific translation risk enters the
# picture. The torch functions below are a mechanical, op-for-op
# translation of that already-validated numpy shadow (torch.stack for
# np.array, torch.cat for np.concatenate, .mean(dim=0) for
# .mean(axis=0), otherwise identical) -- the same low-risk transcription
# strategy every other function in this module already uses relative to
# its own numpy original.
# =====================================================================
def _exact_drill_rotation_torch(theta):
    """torch re-expression of Shell4MITCCorotational._exact_drill_
    rotation() -- op-for-op identical (cos/sin of a differentiable
    scalar, no branching), see that method's own docstring for why a
    single-axis exact rotation is safe to differentiate through."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    zero = torch.zeros((), dtype=theta.dtype)
    one = torch.ones((), dtype=theta.dtype)
    return torch.stack([
        torch.stack([c, s, zero]),
        torch.stack([-s, c, zero]),
        torch.stack([zero, zero, one]),
    ])


def _shell4_mitc_coro_dof_local_torch(u_t, X_ref_t, R0_t, Xref_local_t,
                                       e1_0_t, e2_0_t, e3_0_t):
    """torch re-expression of Shell4MITCCorotational._local_relative_
    dofs() -- X_ref_t/R0_t/Xref_local_t/e1_0_t/e2_0_t/e3_0_t are all
    CONSTANT w.r.t. u (computed once in plain numpy from elem_coords
    alone, exactly mirroring dN_dX_t elsewhere in this module), so no
    gradient needs to flow through them. Returns (dof_local (24,),
    e1, e2, e3 (each (3,), the CURRENT drilling frame -- u-dependent),
    theta_global (4,3), also u-dependent, needed again by the committed
    coupling force below)."""
    u_nodes = u_t.reshape(4, 6)
    u_trans = u_nodes[:, 0:3]
    theta_global = u_nodes[:, 3:6]

    mean_theta = theta_global.mean(dim=0)
    theta_z_mean = mean_theta[2]
    R_drill = R0_t @ _exact_drill_rotation_torch(theta_z_mean)
    e1, e2, e3 = R_drill[0], R_drill[1], R_drill[2]

    x_current = X_ref_t + u_trans
    centroid_current = x_current.mean(dim=0)
    x_current_local = (x_current - centroid_current) @ R_drill.T

    u_local_mem_xy = x_current_local[:, 0:2] - Xref_local_t[:, 0:2]
    thetaz_local = theta_global[:, 2] - theta_z_mean

    node_vecs = []
    for a in range(4):
        node_vecs.append(torch.stack([
            u_local_mem_xy[a, 0],
            u_local_mem_xy[a, 1],
            u_trans[a] @ e3_0_t,
            theta_global[a] @ e1_0_t,
            theta_global[a] @ e2_0_t,
            thetaz_local[a],
        ]))
    dof_local = torch.cat(node_vecs)
    return dof_local, e1, e2, e3, theta_global


def _shell4_mitc_coro_bending_membrane_coupling_force_torch(dof_local, Dm_t, h, gauss_data_t, MEM):
    """torch re-expression of Shell4MITCCorotational._bending_membrane_
    coupling_force()'s N_add=None branch ONLY -- see this module's own
    shell4_mitc_corotational_committed_tangent_autograd() docstring for
    why the N_add-not-None mixed-formulation branch is out of scope
    here. gauss_data_t: list of (N_t (4,), Bm_t (3,8), detJ float, wgt
    float) per Gauss point, all CONSTANT w.r.t. u."""
    dtype = dof_local.dtype
    wx = torch.stack([dof_local[6 * a + 4] for a in range(4)])
    wy = torch.stack([-dof_local[6 * a + 3] for a in range(4)])
    dof_mem = torch.stack([dof_local[i] for i in MEM])

    wx_use = wx - wx.sum() / 4.0
    wy_use = wy - wy.sum() / 4.0
    Nb_offset = -0.25

    contrib = [torch.zeros((), dtype=dtype) for _ in range(24)]
    for N_t, Bm_t, detJ, wgt in gauss_data_t:
        Wx = N_t @ wx_use
        Wy = N_t @ wy_use
        eps_add = torch.stack([0.5 * Wx * Wx, 0.5 * Wy * Wy, Wx * Wy])
        eps_lin = Bm_t @ dof_mem
        sigma_total = Dm_t @ (eps_lin + eps_add)

        f_mem_add = Bm_t.T @ (Dm_t @ eps_add) * detJ * wgt * h
        for i, idx in enumerate(MEM):
            contrib[idx] = contrib[idx] + f_mem_add[i]
        for b in range(4):
            Nb_c = N_t[b] + Nb_offset
            contrib[6 * b + 4] = contrib[6 * b + 4] + h * detJ * wgt * Nb_c * (Wx * sigma_total[0] + Wy * sigma_total[2])
            contrib[6 * b + 3] = contrib[6 * b + 3] - h * detJ * wgt * Nb_c * (Wy * sigma_total[1] + Wx * sigma_total[2])
    return torch.stack(contrib)


def _shell4_mitc_coro_bending_membrane_coupling_force_committed_torch(
        dof_local, theta_global, theta_committed_t, e1_0_t, e2_0_t, Dm_t, h, gauss_data_t, N_add_history_t, MEM):
    """torch re-expression of Shell4MITCCorotational._bending_membrane_
    coupling_force_committed() -- theta_committed_t/N_add_history_t are
    the REAL, u-independent committed-state constants (from `state`),
    exactly like this module's other constant tensors."""
    dtype = dof_local.dtype
    theta_incr = theta_global - theta_committed_t
    wx_incr = theta_incr @ e2_0_t
    wy_incr = -(theta_incr @ e1_0_t)
    dof_mem = torch.stack([dof_local[i] for i in MEM])

    contrib = [torch.zeros((), dtype=dtype) for _ in range(24)]
    for gp_idx, (N_t, Bm_t, detJ, wgt) in enumerate(gauss_data_t):
        Wx = N_t @ wx_incr
        Wy = N_t @ wy_incr
        eps_add_incr = torch.stack([0.5 * Wx * Wx, 0.5 * Wy * Wy, Wx * Wy])
        Nadd_trial_gp = N_add_history_t[gp_idx] + Dm_t @ eps_add_incr
        eps_lin = Bm_t @ dof_mem
        sigma_total = Dm_t @ eps_lin + Nadd_trial_gp

        f_mem_add = Bm_t.T @ Nadd_trial_gp * detJ * wgt * h
        for i, idx in enumerate(MEM):
            contrib[idx] = contrib[idx] + f_mem_add[i]
        for b in range(4):
            Nb = N_t[b]
            contrib[6 * b + 4] = contrib[6 * b + 4] + h * detJ * wgt * Nb * (Wx * sigma_total[0] + Wy * sigma_total[2])
            contrib[6 * b + 3] = contrib[6 * b + 3] - h * detJ * wgt * Nb * (Wy * sigma_total[1] + Wx * sigma_total[2])
    return torch.stack(contrib)


def shell4_mitc_corotational_committed_tangent_autograd(elem_coords, u_elem, D, state,
                                                          iter_state=None, formulation=None):
    """Independent torch.autograd.grad tangent stiffness for
    Shell4MITCCorotational's state=... path (elements/shells.py),
    cross-checking _tangent_stiffness_committed_complex_step()'s
    existing complex-step result.

    WHAT THIS ACTUALLY BUYS, GIVEN COMPLEX-STEP IS ALREADY EXACT: unlike
    Tet4NeoHookean's FD tangent above (genuine truncation error, so
    autograd is a real accuracy upgrade), _tangent_stiffness_committed_
    complex_step() already recovers the exact derivative to machine
    precision (see that method's own docstring) -- so this addition is
    NOT a correctness improvement. Its value is the same "keep every
    valid implementation available, explicit opt-in" principle Wave 0
    item 3 established for backend="torch" generally: (1) a THIRD,
    independently-implemented method agreeing with the existing
    complex-step AND analytic (state=None path) tangents is stronger
    evidence of correctness than either alone (same reasoning this
    module's own docstring gives for the Tet4/Tet10 cross-checks above);
    (2) once wired behind a device= dispatch (left for a future item --
    NOT done here, this function always runs on whatever device its
    input tensors are placed on, matching this module's existing
    functions), this reduces to ONE reverse-mode graph traversal rather
    than complex-step's 24 separate perturbed internal_force() calls,
    which is where an eventual GPU/batched-element win would come from
    for a large multi-element shell mesh -- not exercised by this
    single-element function itself.

    elem_coords: (4,3) array-like. u_elem: (24,) array-like (node-major,
    6 dof/node). D: (Dm, Db, Ds, h) tuple, material.D_shell()'s return.
    state: the dict returned by Shell4MITCCorotational.init_state()/
    commit_state() (keys "theta_committed" (4,3), "N_add_history"
    (n_gauss,3)). iter_state: MUST be None -- see this function's own
    module-level comment above ("SCOPE, DELIBERATELY NARROWED") for why
    the mixed-formulation branch is not yet supported; passing anything
    else raises NotImplementedError rather than silently computing the
    wrong answer. formulation: an existing Shell4MITCCorotational
    instance to reuse (optional, a fresh one is created if omitted).

    Returns (K (24,24) numpy array, f_int (24,) numpy array), same
    conventions as the Tet4/Tet10 functions above. Does NOT symmetrize
    K (same reasoning as tet4_neo_hookean_tangent_autograd() above --
    an exact autograd Hessian-of-a-scalar-potential is already
    symmetric to float64 round-off); note this makes K's raw symmetry
    itself a meaningful diagnostic to check against the production
    method's explicit 0.5*(K+K.T) symmetrization when cross-checking."""
    _require_torch()
    if iter_state is not None:
        raise NotImplementedError(
            "shell4_mitc_corotational_committed_tangent_autograd: the "
            "mixed-formulation iter_state=... branch is not yet ported to "
            "torch -- out of scope for this pass, see this function's own "
            "docstring and this module's 'SCOPE, DELIBERATELY NARROWED' "
            "comment above it. Only iter_state=None (this package's own "
            "overwhelmingly common case) is supported here.")
    from .elements.shells import Shell4MITCCorotational, _MEM

    elem_coords = np.asarray(elem_coords, dtype=float)
    u_elem = np.asarray(u_elem, dtype=float)
    shell = formulation if formulation is not None else Shell4MITCCorotational()
    linear = shell._linear

    X_ref = elem_coords
    e1_0, e2_0, e3_0, local = linear._local_frame_and_coords(X_ref)
    R0 = np.vstack([e1_0, e2_0, e3_0])
    centroid_ref = X_ref.mean(axis=0)
    Xref_local = (X_ref - centroid_ref) @ R0.T

    K_local0 = shell._local_material_stiffness(X_ref, D)
    Dm, Db, Ds, h = D

    pts, wts = gauss_product(shell.gauss_order, shell.dim)
    membrane = linear._membrane
    gauss_data_np = []
    for p, wgt in zip(pts, wts):
        N, _ = membrane.shape_and_derivs(p)
        Bm, detJ = membrane.B_matrix(p, local)
        gauss_data_np.append((N, Bm, detJ, wgt))

    dtype = torch.float64
    X_ref_t = torch.tensor(X_ref, dtype=dtype)
    R0_t = torch.tensor(R0, dtype=dtype)
    Xref_local_t = torch.tensor(Xref_local, dtype=dtype)
    e1_0_t = torch.tensor(e1_0, dtype=dtype)
    e2_0_t = torch.tensor(e2_0, dtype=dtype)
    e3_0_t = torch.tensor(e3_0, dtype=dtype)
    K_local0_t = torch.tensor(K_local0, dtype=dtype)
    Dm_t = torch.tensor(Dm, dtype=dtype)
    theta_committed_t = torch.tensor(state["theta_committed"], dtype=dtype)
    N_add_history_t = torch.tensor(state["N_add_history"], dtype=dtype)
    gauss_data_t = [(torch.tensor(N, dtype=dtype), torch.tensor(Bm, dtype=dtype), float(detJ), float(wgt))
                    for (N, Bm, detJ, wgt) in gauss_data_np]

    u_t = torch.tensor(u_elem, dtype=dtype, requires_grad=True)

    dof_local, e1, e2, e3, theta_global = _shell4_mitc_coro_dof_local_torch(
        u_t, X_ref_t, R0_t, Xref_local_t, e1_0_t, e2_0_t, e3_0_t)
    f_local_orig = K_local0_t @ dof_local + _shell4_mitc_coro_bending_membrane_coupling_force_torch(
        dof_local, Dm_t, h, gauss_data_t, _MEM)
    f_local_committed = _shell4_mitc_coro_bending_membrane_coupling_force_committed_torch(
        dof_local, theta_global, theta_committed_t, e1_0_t, e2_0_t, Dm_t, h, gauss_data_t, N_add_history_t, _MEM)
    f_local = f_local_orig + f_local_committed

    # T.T @ f_local (T = block-diag of R=[e1;e2;e3], 8 blocks) rewritten
    # as a reshape+matmul: for row vector v and orthonormal R,
    # (R.T @ v)_j = sum_i R_{i,j} v_i = (v @ R)_j -- so per-block
    # "rotate local force back to global" reduces to f_blocks @ R,
    # avoiding ever materializing a 24x24 matrix. Verified (not just
    # derived) against the real _rotation_matrix_c()-based production
    # result to 1.25e-16 relative -- see this function's own module-
    # level "VALIDATION METHODOLOGY NOTE" above.
    R_t = torch.stack([e1, e2, e3])
    f_blocks = f_local.reshape(8, 3)
    f_int_t = (f_blocks @ R_t).reshape(24)

    n = 24
    K = torch.zeros((n, n), dtype=dtype)
    for i in range(n):
        (grad_i,) = torch.autograd.grad(f_int_t[i], u_t, retain_graph=True)
        K[i, :] = grad_i

    return K.detach().numpy(), f_int_t.detach().numpy()


# =====================================================================
# Beam2DReissner cross-check (Wave 17 item 140, docs/consolidated_
# future_roadmap.md): an independent torch.autograd.grad cross-check of
# Beam2DReissner's existing hand-derived ANALYTIC tangent_stiffness()
# (elements/beams.py) -- see that class's own docstring, "ANALYTIC
# TANGENT DERIVATION" section, for the chain-rule-through-eps/gam/kap
# derivation this validates, and tests/test_beam2d_reissner.py CHECK
# (a) for the existing central-finite-difference gate that derivation
# already had to pass once, at 1e-6.
#
# WHY THIS IS A GENUINE INDEPENDENT RE-DERIVATION, NOT A RESTATEMENT OF
# THE SAME ALGEBRA: Beam2DReissner._B_and_H() builds grad(eps)/grad(gam)
# /grad(kap) (B) and their Hessians (H_eps/H_gam) BY HAND, then
# internal_force()/tangent_stiffness() assemble f_int/K_T from those
# already-differentiated pieces. Re-using B/H here would just be
# checking that hand algebra against itself. Instead, the function
# below re-expresses ONLY the PRIMAL strain-energy computation --
# eps/gam/kap straight from the strain-measure formulas in
# Beam2DReissner's own class docstring ("STRAIN MEASURES" section),
# with no B or H anywhere -- as a torch scalar strain energy, and lets
# torch.autograd.grad differentiate that from scratch, once for
# f_int (dU/dq) and once more per-row for K_T (d^2U/dq^2). Agreement
# with the analytic B/H-based tangent is therefore real evidence the
# hand-derived chain rule in _B_and_H() is correct, exactly the same
# "third, independently-implemented method" logic this module's other
# cross-checks (Tet4NeoHookean, Tet10SolidTL, Shell4MITCCorotational
# above) already rely on.
#
# QUADRATURE / LENGTH-SCALING NOTE (the one place a beam element's
# reference-domain integral could silently pick up the wrong factor):
# Beam2DReissner's own class docstring ("INTEGRATION" section)
# establishes that eps/gam/kap are each evaluated ONCE, at the element
# midpoint (one-point reduced quadrature, shear-locking-free, and
# coincidentally exact for kap too since theta' is already constant) --
# so the strain-energy integral over the reference domain s in
# [0, L0] collapses to (midpoint strain-energy density) * L0, no
# separate Gauss weight/Jacobian bookkeeping beyond that single L0
# factor (the reference-configuration Jacobian ds/dxi for a 2-node
# element IS L0/2 on the standard xi in [-1,1] domain, but the strain
# measures are already expressed directly in terms of the physical
# arc-length derivative d(.)/ds = d(.)/dL0, per _strain_measures()'s
# own (u1_2-u1_1)/L0 form -- so no extra 2/L0 * L0/2 round trip is
# needed here; multiplying by L0 once, as below, is the exact quadrature
# this class's own internal_force() ("f_local = L0 * (B.T @ [N,Q,M])")
# already performs). Getting this wrong (e.g. an extra factor of 2, or
# 1/2) would show up as a K_T mismatch scaled uniformly across every
# entry, not a subtle isolated error -- worth flagging explicitly since
# this is exactly the failure mode the task that added this function
# was warned about.
#
# VALIDATION STATUS (this sandbox has no working torch install -- see
# this module's own _HAS_TORCH comment at the top): written and
# code-reviewed line-by-line against Beam2DReissner._strain_measures()/
# internal_force()/tangent_stiffness() in elements/beams.py, matching
# every existing torch cross-check function's own careful-transcription
# discipline (same variable names/order as the numpy original: u1p,
# u2p, thp, th_mid, cphi, sphi, eps, gam, kap). NOT YET run for real --
# tests/test_beam2d_reissner_autograd.py skips itself cleanly here
# (_HAS_TORCH-gated, same idiom as tests/test_autograd_tangent.py) and
# is awaiting real execution on the user's own PyTorch-equipped machine
# (Windows, NVIDIA GPU) the same way every other torch addition in this
# module already was (see e.g. this module's own comment on the
# Tet4/Tet10 cross-checks above, and Wave 0 item 2's own roadmap row) --
# no relative-error number is claimed here until that run happens.
# =====================================================================
def _beam2d_reissner_strain_energy_torch(L0, q_t, EA, kappa_s_GA, EI):
    """torch re-expression of Beam2DReissner._strain_measures(), kept
    op-for-op identical to that method's numpy formula (same variable
    names, same order of operations), PLUS the strain-energy density
    -- deliberately NOT reusing _B_and_H()'s hand-derived gradients (see
    this section's own module-level comment above for why). L0: python
    float (reference chord length, constant w.r.t. q). q_t: (6,) torch
    tensor, the LOCAL nodal dof vector (u1_1,u2_1,th1,u1_2,u2_2,th2),
    differentiable. EA, kappa_s_GA, EI: python floats (material
    constants, constant w.r.t. q). Returns (U, eps, gam, kap): U is the
    scalar torch strain energy L0*(0.5*EA*eps^2 + 0.5*kappa_s_GA*gam^2
    + 0.5*EI*kap^2) -- eps/gam/kap returned too since they are a free
    cross-check of Beam2DReissner._strain_measures() itself (same
    closed-form expressions, re-derived independently in torch)."""
    u1_1, u2_1, th1, u1_2, u2_2, th2 = q_t[0], q_t[1], q_t[2], q_t[3], q_t[4], q_t[5]
    u1p = (u1_2 - u1_1) / L0
    u2p = (u2_2 - u2_1) / L0
    thp = (th2 - th1) / L0
    th_mid = 0.5 * (th1 + th2)
    cphi = torch.cos(th_mid)
    sphi = torch.sin(th_mid)
    eps = (1.0 + u1p) * cphi + u2p * sphi - 1.0
    gam = u2p * cphi - (1.0 + u1p) * sphi
    kap = thp

    U = L0 * (0.5 * EA * eps ** 2 + 0.5 * kappa_s_GA * gam ** 2 + 0.5 * EI * kap ** 2)
    return U, eps, gam, kap


def beam2d_reissner_tangent_autograd(elem_coords, u_elem, mat, device="cpu"):
    """Independent torch.autograd.grad tangent stiffness for
    Beam2DReissner, cross-checking its ANALYTIC tangent_stiffness()
    (Wave 17 item 140, see elements/beams.py's own class docstring for
    the hand chain-rule derivation this validates).

    elem_coords: (2,2) array-like (REFERENCE nodal coordinates, same
    convention as internal_force()/tangent_stiffness()). u_elem: (6,)
    array-like, GLOBAL nodal dof vector (u1,u2,theta per node,
    node-major -- same convention as Beam2DReissner.internal_force()).
    mat: (E, G, A, I, kappa_s), same unpacking as
    Beam2DReissner._unpack_mat(). device: "cpu" (default) or "cuda",
    matching torch_sparse_solver.py's own device= convention.

    Method: the GLOBAL -> LOCAL rotation q = T.T @ u_elem (T:
    Beam2DReissner._T(), a CONSTANT 6x6 block-diagonal matrix built
    once from the reference chord angle alone -- no gradient needs to,
    or should, flow through T itself, exactly like dN_dX_t/gc/ga/etc.
    are constant w.r.t. u elsewhere in this module) is applied to a
    torch tensor with requires_grad=True, so grad flows through q
    straight back to the GLOBAL u_elem the caller passed in -- f_int
    and K_T below are therefore already expressed in the SAME global
    dof ordering/frame internal_force()/tangent_stiffness() return,
    with no separate T @ (...) @ T.T step needed afterward (unlike
    Shell4MITCCorotational's cross-check above, which does need that
    because its local-force assembly is built directly on local dofs
    with no torch-differentiable link back to the global ones baked in
    the way q_t = T_t.T @ u_t gives here for free).

    Returns (K (6,6) numpy array, f_int (6,) numpy array), same
    (K, f_int) return-order convention as every other function in this
    module. Does NOT symmetrize K (same reasoning as
    tet4_neo_hookean_tangent_autograd() above -- an exact autograd
    Hessian-of-a-scalar-potential is already symmetric to float64
    round-off, so an asymmetry here would itself be a meaningful
    diagnostic, not expected noise to paper over)."""
    _require_torch()
    from .elements.beams import Beam2DReissner

    elem_coords = np.asarray(elem_coords, dtype=float)
    u_elem = np.asarray(u_elem, dtype=float)
    E, G, A, I, kappa_s = mat

    L0, c0, s0 = Beam2DReissner._ref_frame(elem_coords)
    T = Beam2DReissner._T(c0, s0)

    dtype = torch.float64
    T_t = torch.tensor(T, dtype=dtype, device=device)
    u_t = torch.tensor(u_elem, dtype=dtype, device=device, requires_grad=True)
    q_t = T_t.T @ u_t

    EA = E * A
    kappa_s_GA = kappa_s * G * A
    EI = E * I
    U, _, _, _ = _beam2d_reissner_strain_energy_torch(L0, q_t, EA, kappa_s_GA, EI)

    (f_int_t,) = torch.autograd.grad(U, u_t, create_graph=True)

    n = f_int_t.shape[0]
    K = torch.zeros((n, n), dtype=dtype, device=device)
    for i in range(n):
        (grad_i,) = torch.autograd.grad(f_int_t[i], u_t, retain_graph=True)
        K[i, :] = grad_i

    return K.detach().cpu().numpy(), f_int_t.detach().cpu().numpy()
