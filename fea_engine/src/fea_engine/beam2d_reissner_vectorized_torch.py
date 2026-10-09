# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
beam2d_reissner_vectorized_torch.py -- Wave 17 item 147 torch addendum
(docs/consolidated_future_roadmap.md): a side-by-side PyTorch,
GPU-capable backend for the batched (all-elements-at-once) Beam2DReissner
internal_force()/tangent_stiffness() path item 147 itself built in plain
NumPy (beam2d_reissner_vectorized.py, this module's own direct template).

Why a separate peer module rather than a backend="torch" parameter bolted
onto beam2d_reissner_vectorized.py's existing functions: that module's
internal_force_batched()/tangent_stiffness_batched()/assemble_*_vectorized()
functions currently take no backend/device parameter at all (unlike
vectorized_assembly.py's assemble_stiffness_vectorized(..., backend=,
device=), which was DESIGNED from the start, in item 135/136, around a
MeshTransformation object that already carries a backend/device concept
end to end). Retrofitting a backend= flag onto beam2d_reissner_
vectorized.py's functions would mean branching every internal helper
(_local_dofs_batched, _B_and_H_batched) on backend, duplicating almost all
of their bodies either way -- there is no shared "backend-agnostic
tensor-transformation object" here the way MeshTransformation provides for
the continuum-element path. This project's own established precedent for
exactly this situation is a peer file, not a parameter: torch_sparse_
solver.py sits beside solver.py's SciPy-based solve path (Wave 0 item 3),
offering the SAME solve_static() capability via a completely separate
module rather than a branch inside solver.py itself. This module follows
that same precedent, one level up the call stack (element-level batched
force/stiffness, not the final linear solve) -- matching this project's
standing principle (restated across the roadmap) that "whenever more than
one valid implementation/backend exists for the same capability, keep all
of them available and give the caller an explicit, named choice, never
silently prefer or replace one": nothing in beam2d_reissner_vectorized.py
is modified, nothing in the elements/rom_engine call sites that already
use the numpy path needs to change, and a caller who wants the torch/GPU
path imports this module explicitly by name.

Batching strategy: IDENTICAL to beam2d_reissner_vectorized.py's own --
every array operation below is a straight re-expression, in torch ops, of
that module's _local_dofs_batched()/_B_and_H_batched()/
internal_force_batched()/tangent_stiffness_batched(), with the SAME
leading n_elements axis threaded through instead of a Python loop, and NO
change to which axis means what (re-checked line-by-line against that
module while writing this one -- see each function's own docstring below
for the specific numpy line it mirrors). np.einsum subscript strings carry
over to torch.einsum unchanged; np.zeros(...)/np.stack(...)/[:, None]
broadcasting carry over to torch with an added dtype=/device=; the one
non-literal translation is `.transpose(0, 2, 1)` (a 3-axis numpy
permutation used in _B_and_H_batched() to build ga_gc/gb_gc as the
transpose of gc_ga/gc_gb) -- torch's `Tensor.transpose(dim0, dim1)` takes
exactly two axes to swap, and swapping axes 1 and 2 of a (n, 6, 6) tensor
while leaving axis 0 fixed is exactly what numpy's `.transpose(0, 2, 1)`
does here, so `.transpose(1, 2)` is the correct (not merely similar) torch
equivalent -- confirmed by matching the resulting index pattern
(gc_ga[n, i, j] = gc[n, i] * ga[n, j], so ga_gc[n, i, j] must equal
gc_ga[n, j, i], which is exactly what swapping axes 1 and 2 produces)
rather than assumed from the similar-looking API.

Reuse of existing torch-compatible assembly machinery (per this item's own
instruction to check for it, rather than writing a new one blind):
vectorized_assembly.scatter_global_stiffness() (Wave 9 addendum item 136)
ALREADY dispatches on isinstance(ke, np.ndarray) vs. a live torch.Tensor,
and already returns a numpy/scipy result either way -- it is reused
VERBATIM here for the tangent-stiffness assembly path
(assemble_tangent_stiffness_vectorized_torch() below), with no new code
needed for that half. There is, however, no existing torch-compatible
FORCE-vector scatter anywhere in this codebase: beam2d_reissner_
vectorized.scatter_global_force() (item 147's own addition) is numpy-only
(built on np.add.at, with no torch branch -- vectorized_assembly.py never
needed a force scatter of any kind, numpy or torch, since it is a
stiffness-only module), so this module adds scatter_global_force_torch()
below, mirroring scatter_global_stiffness()'s own dispatch-and-return
convention (isinstance(fe, np.ndarray) dispatch; always returns numpy).

device= parameter: "cpu" (default) or "cuda", the same naming convention
torch_sparse_solver.py's own device= parameter and vectorized_assembly.py's
backend="torch", device= parameter already use throughout this codebase.

Output convention (numpy in / numpy out): internal_force_batched_torch()
and tangent_stiffness_batched_torch() below take the SAME plain-numpy
elem_coords_all (n,2,2) / u_elem_all (n,6) inputs as beam2d_reissner_
vectorized.py's internal_force_batched()/tangent_stiffness_batched(), and
return the SAME plain-numpy f_global (n,6) / K_global (n,6,6) outputs --
the exact numpy boundary that module's own tests/callers already use, so
these are drop-in alternatives (with an added device= choice) wherever
that module's own functions are called today (in particular, tests/
test_beam2d_reissner_vectorized.py's own CHECK (a) element-level
comparison, and rom_engine.attractor's sweep drivers, which the roadmap's
item 146 documents as intending to wrap "item 147's fast vectorized
solve_nonlinear_transient() path" -- this torch backend is usable there
too via the SAME `model(param) -> (t, y[, ydot])` FE-callable contract
attractor.py's own _sweep_driver() dispatch already expects, with no
change needed to attractor.py itself). assemble_internal_force_vectorized_
torch()/assemble_tangent_stiffness_vectorized_torch() below keep the
computation GPU-resident end to end (conversion to torch tensors once, at
the top; conversion back to numpy only at the final scatter step, exactly
matching vectorized_assembly.py's own documented "the one place a copy
back to host is unavoidable is the very end of each chunk" policy) --
these are the functions to use for the actual sweep-scale speedup this
item is for, not the two numpy-in/numpy-out functions above (which exist
primarily for direct, apples-to-apples validation against the numpy
vectorized path and the original looped element, see this module's own
test file).

STATUS (2026-09-30): structurally complete and carefully reasoned through
line-by-line against beam2d_reissner_vectorized.py's own batching
strategy, as above -- but NOT YET EXECUTED. torch is not importable in
this development sandbox (confirmed: `import torch` raises
ModuleNotFoundError here -- a plain "not installed" case, not even
torch_sparse_solver.py's own CUDA-linked-wheel-with-no-CUDA-runtime
failure mode, since no torch wheel is present at all in this sandbox).
This module is gated on the SAME _HAS_TORCH/_require_torch() single
source of truth torch_sparse_solver.py already establishes (imported from
there, not re-derived), so `import fea_engine` and the rest of the test
suite are completely unaffected by torch's absence here. Real numerical
validation -- including this module's own achieved CPU relative error
against both the looped Beam2DReissner element and the numpy vectorized
path, and the device="cuda" path in its entirety -- is pending execution
on the user's own PyTorch-equipped machine (Windows, NVIDIA GeForce GTX
1050, per torch_sparse_solver.py's own VALIDATED note, 2026-09-08, PyTorch
2.7.0/CUDA 12.6) -- see tests/test_beam2d_reissner_vectorized_torch.py,
written to run for real there.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .vectorized_assembly import scatter_global_stiffness

try:
    from .torch_sparse_solver import _HAS_TORCH, _require_torch
except Exception:
    # See vectorized_assembly.py's own identical except clause for why
    # this is a defensive backstop, not the primary path.
    _HAS_TORCH = False

    def _require_torch():
        raise ImportError(
            "beam2d_reissner_vectorized_torch.py: torch_sparse_solver.py "
            "(this project's single _HAS_TORCH/_require_torch source of "
            "truth) could not be imported -- the torch backend is "
            "unavailable.")


# =====================================================================
# Core torch computation (tensors in, tensor out) -- mirrors beam2d_
# reissner_vectorized.py's _local_dofs_batched()/_B_and_H_batched() line
# by line, see module docstring for the translation notes.
# =====================================================================
def _local_dofs_batched_torch(elem_coords_all, u_elem_all):
    """Torch mirror of beam2d_reissner_vectorized._local_dofs_batched():
    elem_coords_all, u_elem_all already torch.Tensors, same device/dtype
    -> L0 (n,), T (n,6,6), q (n,6)."""
    import torch
    X1 = elem_coords_all[:, 0, :]
    X2 = elem_coords_all[:, 1, :]
    dX = X2 - X1
    L0 = torch.linalg.norm(dX, dim=1)
    c0 = dX[:, 0] / L0
    s0 = dX[:, 1] / L0

    n = elem_coords_all.shape[0]
    T = torch.zeros((n, 6, 6), dtype=elem_coords_all.dtype, device=elem_coords_all.device)
    T[:, 0, 0] = c0; T[:, 0, 1] = -s0
    T[:, 1, 0] = s0; T[:, 1, 1] = c0
    T[:, 2, 2] = 1.0
    T[:, 3, 3] = c0; T[:, 3, 4] = -s0
    T[:, 4, 3] = s0; T[:, 4, 4] = c0
    T[:, 5, 5] = 1.0

    # q = T.T @ u_elem, per element -- same einsum subscripts as the
    # numpy version (carry over unchanged).
    q = torch.einsum('nji,nj->ni', T, u_elem_all)
    return L0, T, q


def _B_and_H_batched_torch(L0, q):
    """Torch mirror of beam2d_reissner_vectorized._B_and_H_batched():
    L0 (n,), q (n,6) -> B (n,3,6), H_eps (n,6,6), H_gam (n,6,6),
    eps (n,), gam (n,), kap (n,). See module docstring for the
    .transpose(0, 2, 1) -> .transpose(1, 2) translation note."""
    import torch
    dtype, device = q.dtype, q.device
    n = L0.shape[0]
    u1_1, u2_1, th1 = q[:, 0], q[:, 1], q[:, 2]
    u1_2, u2_2, th2 = q[:, 3], q[:, 4], q[:, 5]
    u1p = (u1_2 - u1_1) / L0
    u2p = (u2_2 - u2_1) / L0
    thp = (th2 - th1) / L0
    th_mid = 0.5 * (th1 + th2)
    cphi, sphi = torch.cos(th_mid), torch.sin(th_mid)
    eps = (1.0 + u1p) * cphi + u2p * sphi - 1.0
    gam = u2p * cphi - (1.0 + u1p) * sphi
    kap = thp

    ga = torch.zeros((n, 6), dtype=dtype, device=device); ga[:, 0] = -1.0 / L0; ga[:, 3] = 1.0 / L0
    gb = torch.zeros((n, 6), dtype=dtype, device=device); gb[:, 1] = -1.0 / L0; gb[:, 4] = 1.0 / L0
    gc = torch.zeros((n, 6), dtype=dtype, device=device); gc[:, 2] = 0.5; gc[:, 5] = 0.5
    gd = torch.zeros((n, 6), dtype=dtype, device=device); gd[:, 2] = -1.0 / L0; gd[:, 5] = 1.0 / L0

    grad_eps = cphi[:, None] * ga + sphi[:, None] * gb + gam[:, None] * gc
    grad_gam = cphi[:, None] * gb - sphi[:, None] * ga - (eps + 1.0)[:, None] * gc
    grad_kap = gd
    B = torch.stack([grad_eps, grad_gam, grad_kap], dim=1)  # (n, 3, 6)

    gc_ga = torch.einsum('ni,nj->nij', gc, ga); ga_gc = gc_ga.transpose(1, 2)
    gc_gb = torch.einsum('ni,nj->nij', gc, gb); gb_gc = gc_gb.transpose(1, 2)
    gc_gc = torch.einsum('ni,nj->nij', gc, gc)

    H_eps = (-sphi[:, None, None] * (gc_ga + ga_gc)
             + cphi[:, None, None] * (gc_gb + gb_gc)
             - (eps + 1.0)[:, None, None] * gc_gc)
    H_gam = (-sphi[:, None, None] * (gc_gb + gb_gc)
             - cphi[:, None, None] * (gc_ga + ga_gc)
             - gam[:, None, None] * gc_gc)

    return B, H_eps, H_gam, eps, gam, kap


def _internal_force_batched_torch(elem_coords_t, u_elem_t, mat):
    """Core torch computation -- tensors in, tensor out (GPU-resident;
    no host copy). Mirrors beam2d_reissner_vectorized.
    internal_force_batched() line by line."""
    import torch
    E, G, A, I, kappa_s = mat
    L0, T, q = _local_dofs_batched_torch(elem_coords_t, u_elem_t)
    B, _, _, eps, gam, kap = _B_and_H_batched_torch(L0, q)
    N = E * A * eps
    Q = kappa_s * G * A * gam
    M = E * I * kap
    resultants = torch.stack([N, Q, M], dim=1)                          # (n, 3)
    f_local = L0[:, None] * torch.einsum('nsm,ns->nm', B, resultants)   # (n, 6)
    f_global = torch.einsum('nij,nj->ni', T, f_local)
    return f_global


def _tangent_stiffness_batched_torch(elem_coords_t, u_elem_t, mat):
    """Core torch computation -- tensors in, tensor out. Mirrors
    beam2d_reissner_vectorized.tangent_stiffness_batched() line by
    line."""
    import torch
    E, G, A, I, kappa_s = mat
    L0, T, q = _local_dofs_batched_torch(elem_coords_t, u_elem_t)
    B, H_eps, H_gam, eps, gam, kap = _B_and_H_batched_torch(L0, q)
    N = E * A * eps
    Q = kappa_s * G * A * gam
    D = torch.diag(torch.tensor([E * A, kappa_s * G * A, E * I],
                                 dtype=elem_coords_t.dtype, device=elem_coords_t.device))
    BtDB = torch.einsum('nsm,st,ntk->nmk', B, D, B)
    K_local = L0[:, None, None] * (BtDB + N[:, None, None] * H_eps + Q[:, None, None] * H_gam)
    K_global = torch.einsum('nij,njk,nlk->nil', T, K_local, T)
    return K_global


# =====================================================================
# Public, numpy-in/numpy-out functions -- drop-in alternatives to
# beam2d_reissner_vectorized.internal_force_batched()/
# tangent_stiffness_batched(), with an added device= choice. See module
# docstring: these exist primarily for direct validation against the
# numpy path and the original looped element; assemble_*_vectorized_
# torch() below (GPU-resident end to end) is the function to use for the
# actual sweep-scale speedup.
# =====================================================================
def internal_force_batched_torch(elem_coords_all, u_elem_all, mat, device="cpu", dtype=None):
    """Torch-vectorized, GPU-capable Beam2DReissner.internal_force(),
    batched across a leading n_elements axis. elem_coords_all (n,2,2),
    u_elem_all (n,6) (plain numpy array-like, matching beam2d_reissner_
    vectorized.internal_force_batched()'s own input convention),
    mat=(E,G,A,I,kappa_s) CONSTANT across all n elements. device="cpu"
    (default) or "cuda".

    Returns f_global (n,6) as a plain numpy array."""
    _require_torch()
    import torch
    dtype = dtype or torch.float64
    elem_coords_t = torch.as_tensor(np.asarray(elem_coords_all, dtype=float), dtype=dtype, device=device)
    u_elem_t = torch.as_tensor(np.asarray(u_elem_all, dtype=float), dtype=dtype, device=device)
    f_global_t = _internal_force_batched_torch(elem_coords_t, u_elem_t, mat)
    return f_global_t.cpu().numpy()


def tangent_stiffness_batched_torch(elem_coords_all, u_elem_all, mat, device="cpu", dtype=None):
    """Torch-vectorized Beam2DReissner.tangent_stiffness() -- same
    conventions as internal_force_batched_torch() above. Returns
    K_global (n,6,6) as a plain numpy array."""
    _require_torch()
    import torch
    dtype = dtype or torch.float64
    elem_coords_t = torch.as_tensor(np.asarray(elem_coords_all, dtype=float), dtype=dtype, device=device)
    u_elem_t = torch.as_tensor(np.asarray(u_elem_all, dtype=float), dtype=dtype, device=device)
    K_global_t = _tangent_stiffness_batched_torch(elem_coords_t, u_elem_t, mat)
    return K_global_t.cpu().numpy()


# =====================================================================
# Force-vector scatter -- torch counterpart of beam2d_reissner_
# vectorized.scatter_global_force() (which is numpy-only, see module
# docstring). Mirrors vectorized_assembly.scatter_global_stiffness()'s
# own isinstance-dispatch / always-numpy-return convention.
# =====================================================================
def scatter_global_force_torch(fe, connectivity, dofs_per_node, n_dof, device="cpu"):
    """Vectorized (no Python loop over elements) scatter-add of a
    stacked local-force array (n_elements, n_edof) into a global force
    vector. fe may be a plain numpy array (dispatches to beam2d_
    reissner_vectorized.scatter_global_force() verbatim) or a live
    torch.Tensor (the GPU-resident path, using torch.Tensor.
    scatter_add_() -- the torch analogue of np.add.at()'s duplicate-
    index accumulation). Always returns a plain numpy (n_dof,) array,
    matching scatter_global_stiffness()'s own "host copy only at the
    very end" convention."""
    if isinstance(fe, np.ndarray):
        from .beam2d_reissner_vectorized import scatter_global_force
        return scatter_global_force(fe, connectivity, dofs_per_node, n_dof)

    _require_torch()
    import torch
    connectivity_t = torch.as_tensor(np.asarray(connectivity), dtype=torch.int64, device=fe.device)
    n_elements, n_edof = fe.shape
    dof_offset = torch.arange(dofs_per_node, dtype=torch.int64, device=fe.device)
    g = (dofs_per_node * connectivity_t[:, :, None] +
         dof_offset[None, None, :]).reshape(n_elements, n_edof)
    F = torch.zeros(n_dof, dtype=fe.dtype, device=fe.device)
    F.scatter_add_(0, g.reshape(-1), fe.reshape(-1))
    return F.cpu().numpy()


def _pick_block(fesystem, block_name):
    for name, formulation, connectivity in fesystem._blocks:
        if block_name is None or name == block_name:
            return name, formulation, connectivity
    raise ValueError(f"no block named {block_name!r} in this fesystem")


# =====================================================================
# GPU-resident-end-to-end assembly path -- torch counterparts of
# beam2d_reissner_vectorized.assemble_internal_force_vectorized()/
# assemble_tangent_stiffness_vectorized(), the functions to actually use
# for the sweep-scale speedup this item is for (see module docstring).
# =====================================================================
def assemble_internal_force_vectorized_torch(fesystem, u_global, mat, block_name=None,
                                               device="cpu", dtype=None):
    """Torch-vectorized, GPU-capable counterpart of beam2d_reissner_
    vectorized.assemble_internal_force_vectorized() -- same additive,
    opt-in, single-homogeneous-material-per-block scope and return-only
    convention (does not mutate fesystem), with a device= parameter.
    Stays GPU-resident from the initial numpy->torch conversion through
    scatter_global_force_torch()'s own final host copy."""
    _require_torch()
    import torch
    dtype = dtype or torch.float64
    name, formulation, connectivity = _pick_block(fesystem, block_name)
    elem_coords_all = fesystem.mesh.nodes[connectivity]           # (n,2,2)
    npn = 3
    g = (npn * connectivity[:, :, None] + np.arange(npn)[None, None, :]).reshape(len(connectivity), -1)
    u_elem_all = u_global[g]

    elem_coords_t = torch.as_tensor(np.asarray(elem_coords_all, dtype=float), dtype=dtype, device=device)
    u_elem_t = torch.as_tensor(np.asarray(u_elem_all, dtype=float), dtype=dtype, device=device)
    fe_t = _internal_force_batched_torch(elem_coords_t, u_elem_t, mat)
    return scatter_global_force_torch(fe_t, connectivity, dofs_per_node=npn,
                                       n_dof=fesystem.n_dof, device=device)


def assemble_tangent_stiffness_vectorized_torch(fesystem, u_global, mat, block_name=None,
                                                  device="cpu", dtype=None, sparse=False):
    """Torch-vectorized, GPU-capable counterpart of beam2d_reissner_
    vectorized.assemble_tangent_stiffness_vectorized() -- reuses
    vectorized_assembly.scatter_global_stiffness() VERBATIM (it already
    dispatches on a live torch.Tensor vs. a plain ndarray, per Wave 9
    addendum item 136, so no new scatter code is needed here, unlike the
    force path above)."""
    _require_torch()
    import torch
    dtype = dtype or torch.float64
    name, formulation, connectivity = _pick_block(fesystem, block_name)
    elem_coords_all = fesystem.mesh.nodes[connectivity]
    npn = 3
    g = (npn * connectivity[:, :, None] + np.arange(npn)[None, None, :]).reshape(len(connectivity), -1)
    u_elem_all = u_global[g]

    elem_coords_t = torch.as_tensor(np.asarray(elem_coords_all, dtype=float), dtype=dtype, device=device)
    u_elem_t = torch.as_tensor(np.asarray(u_elem_all, dtype=float), dtype=dtype, device=device)
    ke_t = _tangent_stiffness_batched_torch(elem_coords_t, u_elem_t, mat)
    return scatter_global_stiffness(ke_t, connectivity, dofs_per_node=npn,
                                     n_dof=fesystem.n_dof, sparse=sparse)
