"""
vectorized_assembly.py -- Wave 11 item 107 (docs/consolidated_future_
roadmap.md): a tensorized (no Python loop over elements) global
stiffness-assembly path, built directly on top of item 106's
`MeshTransformation` (mesh_transform.py).

TensorMesh's own framing for this ("Tensorized assembly", Concepts:
"There is no Python-level loop over elements... assembly is a single
GPU kernel") is the design reference; this module is a native NumPy
reimplementation of the same IDEA (batch the per-element linear algebra
into one array-level contraction instead of a Python `for element in
elements` loop), not a port of any TensorMesh code -- same integration
policy as mesh_transform.py's own docstring restates.

This is an ADDITIVE, opt-in path -- `FESystem.assemble_stiffness()`'s
existing per-element loop is completely unchanged and remains the
default; `assemble_stiffness(..., vectorized=True)` is a second way to
reach the exact same `fesystem.K` result, matching this project's own
"keep every option available, explicit opt-in, never a silent
replacement" convention (Wave 0 item 3's `backend=` parameter, Wave 2
item 12's `method=` choice on `assemble_stiffness()` itself, etc.).

Scope: same continuum-elasticity B-matrix conventions
`mesh_transform.py`'s own Scope section restricts `MeshTransformation`
to -- dofs_per_node in {2, 3} (plane-stress / 3-D solid), using the
exact row-pattern convention every existing Quad4PlaneStress/
Quad8PlaneStress (dofs_per_node=2) and Hex8Solid3D/Hex20Solid3D
(dofs_per_node=3) element's own B_matrix() method already uses
(transcribed directly from those bodies below, not re-derived). A
CONSTANT material matrix D per block (not a per-element or per-Gauss-
point field), matching this item's own "homogeneous-element-type mesh
REGION" framing (tensormesh_analysis_report.md). Only `method="full"`
(the plain `Element.stiffness()` Gauss loop) is supported -- "reduced"/
"hourglass_stabilized" integration is a genuinely different per-element
computation (a second, deflated-eigenproblem stiffness on top of the
plain one) not attempted here. Heterogeneous per-element D is a natural
follow-on (the einsum below still contracts cleanly with an added
element axis on D) but not attempted here either.

GPU/torch side-by-side path (Wave 9 addendum item 136, docs/
consolidated_future_roadmap.md)
-------------------------------------------------------------------
`assemble_stiffness_vectorized(..., backend="numpy"/"torch", device=)`
threads straight through to `MeshTransformation`'s own `backend=`/
`device=` (item 135) for each chunk it builds -- `backend="numpy"`
(default) is byte-for-byte unchanged. `backend="torch"` keeps the
expensive part of this path GPU-resident end to end: `build_B_batched()`
and `tensorized_element_stiffness()` below both DISPATCH on what they
are actually handed (a plain ndarray vs. a live `torch.Tensor`) rather
than taking a separate backend= flag of their own, so a `MeshTransformation`
built with `backend="torch"` flows straight into the B-matrix build and
the einsum contraction as torch tensors on `device`, with NO copy back
to numpy in between -- avoiding exactly the "precompute on GPU, copy to
numpy, convert back to torch for assembly" round trip item 135's own
docstring flags as pointless.

The one place a copy back to host IS unavoidable is the very end of
each chunk: `fesystem.K` is a plain numpy array or `scipy.sparse`
matrix by contract (every other part of `FESystem` -- `fix_dofs()`,
the scipy `solve_static()` path, `assemble_mass()`, ...) -- reads it as
one, regardless of which backend built it, so `scatter_global_stiffness()`
always returns a numpy/scipy result for `assemble_stiffness_vectorized()`
to accumulate into `fesystem.K`, on EITHER backend. On `backend="torch"`
this still needs its own dedicated validation, not an assumed match:
`torch.sparse_coo_tensor(...).coalesce()` sums values at duplicate
indices (the documented behavior of `.coalesce()`), which is the same
duplicate-index accumulation `scipy.sparse.coo_matrix` performs on
`.toarray()`/arithmetic -- confirmed to agree numerically in
tests/test_torch_vectorized_assembly.py, not merely assumed identical
because both are called "COO".
"""
import numpy as np

from .mesh_transform import MeshTransformation

try:
    from .torch_sparse_solver import _HAS_TORCH, _require_torch
except Exception:
    # See mesh_transform.py's own identical except clause for why this
    # is a defensive backstop, not the primary path.
    _HAS_TORCH = False

    def _require_torch():
        raise ImportError(
            "vectorized_assembly.py: torch_sparse_solver.py (this "
            "project's single _HAS_TORCH/_require_torch source of "
            "truth) could not be imported -- backend='torch' is "
            "unavailable.")


def build_B_batched(shape_grad, dofs_per_node):
    """Batched strain-displacement operator B for EVERY element and
    Gauss point in one array call -- the same row-construction
    convention every existing dofs_per_node=2 (Quad4PlaneStress,
    Quad8PlaneStress) or dofs_per_node=3 (Hex8Solid3D, Hex20Solid3D)
    element's own B_matrix() already uses, vectorized over
    (n_elements, n_gauss) instead of computed once per Gauss point
    inside a per-element Python loop. The one remaining Python loop
    here is over n_basis (an element's own node count -- 4/8/20, a
    small, fixed constant), not n_elements (which may be thousands) --
    this is the item's actual scope ("removes the per-ELEMENT loop"),
    not a claim that literally every loop of any kind is gone.

    Parameters
    ----------
    shape_grad : (n_elements, n_gauss, n_basis, dim) ndarray or torch.Tensor
        `MeshTransformation.shape_grad` -- a plain ndarray (backend=
        "numpy", the default) or a live torch.Tensor (backend="torch",
        Wave 9 addendum item 135/136): this function DISPATCHES on
        which one it was actually handed (`isinstance(..., np.ndarray)`)
        rather than taking a separate backend= flag, so it faithfully
        mirrors whatever MeshTransformation it was built from.
    dofs_per_node : int
        2 (plane stress: 3 strain rows [exx, eyy, gxy]) or 3 (3-D
        solid: 6 strain rows [exx, eyy, ezz, gxy, gyz, gzx]).

    Returns
    -------
    B : (n_elements, n_gauss, n_strain, n_basis*dofs_per_node) ndarray or torch.Tensor
        Same array type as `shape_grad` was handed in.
    """
    is_numpy = isinstance(shape_grad, np.ndarray)
    n_elements, n_gauss, n_basis, dim = shape_grad.shape
    if dofs_per_node not in (2, 3):
        raise ValueError(
            f"build_B_batched: unsupported dofs_per_node={dofs_per_node} -- "
            f"only 2 (plane stress) or 3 (3-D solid continuum) are supported; "
            f"see this module's own Scope docstring.")
    if dofs_per_node == 2 and dim != 2:
        raise ValueError("dofs_per_node=2 (plane stress) requires dim=2.")
    if dofs_per_node == 3 and dim != 3:
        raise ValueError("dofs_per_node=3 (3-D solid) requires dim=3.")
    n_strain = 3 if dofs_per_node == 2 else 6

    if is_numpy:
        B = np.zeros((n_elements, n_gauss, n_strain, n_basis * dofs_per_node))
    else:
        import torch
        B = torch.zeros((n_elements, n_gauss, n_strain, n_basis * dofs_per_node),
                         dtype=shape_grad.dtype, device=shape_grad.device)

    if dofs_per_node == 2:
        dNx, dNy = shape_grad[..., 0], shape_grad[..., 1]
        for k in range(n_basis):
            B[:, :, 0, 2 * k] = dNx[:, :, k]
            B[:, :, 1, 2 * k + 1] = dNy[:, :, k]
            B[:, :, 2, 2 * k] = dNy[:, :, k]
            B[:, :, 2, 2 * k + 1] = dNx[:, :, k]
    else:
        dNx, dNy, dNz = shape_grad[..., 0], shape_grad[..., 1], shape_grad[..., 2]
        for k in range(n_basis):
            c = 3 * k
            B[:, :, 0, c] = dNx[:, :, k]
            B[:, :, 1, c + 1] = dNy[:, :, k]
            B[:, :, 2, c + 2] = dNz[:, :, k]
            B[:, :, 3, c] = dNy[:, :, k]; B[:, :, 3, c + 1] = dNx[:, :, k]
            B[:, :, 4, c + 1] = dNz[:, :, k]; B[:, :, 4, c + 2] = dNy[:, :, k]
            B[:, :, 5, c] = dNz[:, :, k]; B[:, :, 5, c + 2] = dNx[:, :, k]
    return B


def tensorized_element_stiffness(mesh_transform, D, thickness=1.0):
    """All `n_elements` local stiffness matrices in one pass (no Python
    loop over elements) via a single einsum contraction over item 106's
    precomputed B/JxW tensors.

    Parameters
    ----------
    mesh_transform : MeshTransformation
        Either `backend="numpy"` (default) or `backend="torch"` (Wave 9
        addendum item 135) -- this function reads `mesh_transform.backend`
        to decide whether to convert `D` to a torch tensor (on
        `mesh_transform.device`) and use `torch.einsum`, or stay on
        plain `np.asarray`/`np.einsum` -- never guesses from array type
        alone, since `mesh_transform.backend` is the single source of
        truth item 135 already established for this.
    D : (n_strain, n_strain) array-like
        Constant material matrix shared by every element in this
        MeshTransformation -- see module docstring Scope.
    thickness : float
        Uniform thickness multiplier (plane-stress only; a harmless
        1.0-default no-op for 3-D solids, matching `Element.stiffness()`'s
        own `thickness=` convention).

    Returns
    -------
    ke : (n_elements, n_dof_per_elem, n_dof_per_elem) ndarray or torch.Tensor
        A torch.Tensor (on mesh_transform.device) when
        mesh_transform.backend == "torch"; a plain ndarray otherwise --
        matching build_B_batched()'s own "same type as what came in"
        convention.
    """
    backend = getattr(mesh_transform, "backend", "numpy")
    B = build_B_batched(mesh_transform.shape_grad, mesh_transform.dofs_per_node)
    if backend == "torch":
        import torch
        D_t = torch.as_tensor(np.asarray(D), dtype=B.dtype, device=mesh_transform.device)
        if D_t.shape[0] != B.shape[2]:
            raise ValueError(
                f"D has shape {tuple(D_t.shape)} but this element's B matrix "
                f"has {B.shape[2]} strain rows -- mismatched material/element pair.")
        # Same contraction as the numpy path below, on torch tensors.
        return torch.einsum('eqsm,st,eqtn,eq->emn', B, D_t, B, mesh_transform.JxW) * thickness

    D = np.asarray(D)
    if D.shape[0] != B.shape[2]:
        raise ValueError(
            f"D has shape {D.shape} but this element's B matrix has "
            f"{B.shape[2]} strain rows -- mismatched material/element pair.")
    # ke[e,m,n] = sum_q sum_s sum_t B[e,q,s,m] * D[s,t] * B[e,q,t,n] * JxW[e,q]
    return np.einsum('eqsm,st,eqtn,eq->emn', B, D, B, mesh_transform.JxW) * thickness


def scatter_global_stiffness(ke, connectivity, dofs_per_node, n_dof, sparse=False):
    """Vectorized (no Python loop over elements) scatter-add of a
    stacked local-stiffness array into a global matrix -- the second
    half of item 107's "no per-element loop anywhere in this path".

    Builds one flat (row, col, data) COO triple across ALL elements at
    once and lets `scipy.sparse.coo_matrix` perform the duplicate-index
    summation (its own well-established behavior on `.tocsr()`/
    `.toarray()` -- the standard vectorized-FEM-assembly idiom), then
    optionally densifies. The node->dof numbering (`dofs_per_node*node
    + local_dof`) exactly mirrors `FESystem._global_dofs()`, so this is
    a drop-in numerical match for what the existing per-element
    `self.K[np.ix_(g, g)] += ke` loop produces.

    `ke` may be a plain ndarray (the original, numpy path -- unchanged)
    or a torch.Tensor (Wave 9 addendum item 136, `ke` built by a
    `backend="torch"` `tensorized_element_stiffness()` call): this
    function DISPATCHES on `isinstance(ke, np.ndarray)` rather than a
    separate flag, mirroring `build_B_batched()`'s own convention.
    `connectivity` is always a plain ndarray regardless of backend (item
    135's own documented contract: connectivity is integer indices, not
    floating-point compute, so it never needs to be a torch tensor).
    Either way, the RETURN is always numpy/scipy (a `coo_matrix` if
    `sparse=True`, a dense ndarray otherwise) -- `fesystem.K` is a plain
    numpy/scipy structure by contract (every other part of `FESystem`
    reads it as one), so this is the one point in the torch path where a
    host copy is unavoidable; see this module's own docstring for why
    that is fine (the expensive per-chunk einsum contraction, and item
    135's own Jacobian/shape-gradient precompute, already ran
    GPU-resident before this point -- only the already-formed local
    stiffness values cross back here, not a second copy of the larger
    per-Gauss-point B/shape_grad/jac tensors).
    """
    connectivity = np.asarray(connectivity)
    n_elements, n_edof, _ = ke.shape

    if isinstance(ke, np.ndarray):
        from scipy.sparse import coo_matrix
        g = (dofs_per_node * connectivity[:, :, None] +
             np.arange(dofs_per_node)[None, None, :]).reshape(n_elements, n_edof)
        row = np.repeat(g, n_edof, axis=1).reshape(-1)
        col = np.tile(g, (1, n_edof)).reshape(-1)
        data = ke.reshape(-1)
        K = coo_matrix((data, (row, col)), shape=(n_dof, n_dof))
        return K if sparse else K.toarray()

    # ------------------------------------------------------------
    # torch path -- Wave 9 addendum item 136. Identical row/col/data
    # construction to the numpy path above, on torch tensors, ending
    # in torch.sparse_coo_tensor(...).coalesce() -- coalesce() sums
    # values at duplicate (row,col) index pairs, the exact duplicate-
    # index-accumulation semantics scipy.sparse.coo_matrix relies on
    # above. This is confirmed numerically against the numpy path in
    # tests/test_torch_vectorized_assembly.py, not merely assumed
    # identical because both call themselves "COO".
    # ------------------------------------------------------------
    import torch
    conn_t = torch.as_tensor(connectivity, dtype=torch.int64, device=ke.device)
    dof_offset = torch.arange(dofs_per_node, dtype=torch.int64, device=ke.device)
    g_t = (dofs_per_node * conn_t[:, :, None] + dof_offset[None, None, :]).reshape(n_elements, n_edof)
    row_t = g_t.repeat_interleave(n_edof, dim=1).reshape(-1)
    col_t = g_t.repeat(1, n_edof).reshape(-1)
    data_t = ke.reshape(-1)
    idx_t = torch.stack([row_t, col_t], dim=0)
    K_sparse_t = torch.sparse_coo_tensor(idx_t, data_t, size=(n_dof, n_dof)).coalesce()

    if sparse:
        from scipy.sparse import coo_matrix
        idx_np = K_sparse_t.indices().cpu().numpy()
        data_np = K_sparse_t.values().cpu().numpy()
        return coo_matrix((data_np, (idx_np[0], idx_np[1])), shape=(n_dof, n_dof))
    return K_sparse_t.to_dense().cpu().numpy()


def assemble_stiffness_vectorized(fesystem, D, gauss_order=None, thickness=1.0, chunk_size=None,
                                   backend="numpy", device="cpu"):
    """Additive, opt-in alternative to `FESystem.assemble_stiffness()`'s
    per-element Python loop -- computes and scatters the stiffness
    contribution of every block in `fesystem._blocks` via item 106/107's
    tensorized path, ADDING into `fesystem.K` exactly like the existing
    per-element loop does (never resetting it -- calling this more than
    once, or mixing it with ordinary `assemble_stiffness()` calls,
    accumulates the same way the existing per-element path always has).
    Mutates `fesystem.K` in place; returns None, matching
    `assemble_stiffness()`'s own convention.

    Only supports the element formulations `MeshTransformation` itself
    supports (see mesh_transform.py's Scope docstring) -- raises the
    same `ValueError` `MeshTransformation.__init__` would for anything
    else, so a mixed-block system with one supported and one
    unsupported block fails loudly rather than silently assembling only
    part of the system.

    backend ("numpy" default, or "torch"), device ("cpu" default, or
    "cuda") -- Wave 9 addendum item 136, docs/consolidated_future_
    roadmap.md: threaded straight through to each chunk's own
    `MeshTransformation(..., backend=backend, device=device)` (item
    135) -- `backend="numpy"` reproduces the ORIGINAL behavior exactly
    (every intermediate array a plain ndarray, unchanged from before
    this item). `backend="torch"` keeps `MeshTransformation`'s own
    Jacobian/shape-gradient precompute, `build_B_batched()`'s B-matrix
    build, and `tensorized_element_stiffness()`'s einsum contraction
    ALL GPU-resident on `device`, chunk by chunk, only converting each
    chunk's already-scattered `K_block` back to numpy/scipy at
    `scatter_global_stiffness()`'s own final step (see that function's
    own docstring for why that one conversion is unavoidable and why
    it is NOT the same wasteful round trip as converting the much
    larger precompute tensors back and forth). `backend="torch"` with
    torch unavailable fails immediately (via `MeshTransformation`'s own
    `_require_torch()` call, at the first chunk's construction), not
    partway through a large mesh's assembly.

    chunk_size (Wave 16 item 129, docs/consolidated_future_roadmap.md,
    source: TensorMesh's `Forms` page `batch_size` argument): optional
    memory-chunking knob. `None` (the default) reproduces the ORIGINAL
    behavior exactly -- one `MeshTransformation`/einsum pass per block,
    covering every element in that block's connectivity at once, same
    as before this item existed. When set to a positive int, each
    block's own connectivity is split into sequential chunks of at most
    `chunk_size` elements (a shorter tail chunk when the count doesn't
    divide evenly -- the same "one extra shorter batch" behavior
    TensorMesh documents for its own `batch_size`), and each chunk gets
    its own `MeshTransformation`/`tensorized_element_stiffness()` call,
    scattered and accumulated into `fesystem.K` before the next chunk is
    built. At most `chunk_size` elements' worth of `shape_grad`/`jac`/
    `JxW` tensors are ever held in memory at once -- the same guarantee
    TensorMesh's `batch_size` provides against its own per-call memory
    ceiling, applied here to the element dimension specifically (not the
    quadrature dimension TensorMesh itself chunks): a per-element
    node/Gauss-point count is small and fixed (4-20 nodes, a handful of
    Gauss points) for every element type this module supports, while
    `n_elements` is exactly the axis that grows unboundedly with mesh
    size, so it's the one worth guarding.

    This produces the SAME `fesystem.K` as `chunk_size=None` (or any
    other chunk_size) for the same inputs, to float64 summation-order
    noise -- NOT bit-for-bit identical in general, checked directly
    (not assumed): a global dof shared by elements that land in
    different chunks gets its contributions summed via a separate
    Python-level `fesystem.K = fesystem.K + K_block` addition per chunk
    boundary it straddles, rather than in one internal scipy
    coo_matrix duplicate-accumulation pass the way the unchunked call
    does it -- floating-point addition isn't associative, so a
    different grouping can round differently (measured: ~1e-4 absolute
    on ~1e11-scale entries for a representative case in
    tests/test_vectorized_assembly.py's own `TestMemoryChunking`,
    matching the ~1e-3 atol/1e-8 rtol tolerance item 107's own
    vectorized-vs-looped comparison already uses for the identical
    reason). Chunk sizes that can't straddle any shared-dof group
    (chunk_size=1, or any chunk_size >= the block's own element count)
    ARE exact, and are checked exactly in that same test file as the
    positive control. This is possible with no new precompute logic at
    all because `MeshTransformation`/`tensorized_element_stiffness()`
    already work correctly on ANY connectivity subset (this is exactly
    why item 106's own `connectivity=` constructor parameter exists in
    the first place -- `FESystem._blocks` already relies on it for
    per-block connectivity slicing); chunking is simply calling that
    same, already-validated machinery on smaller connectivity slices in
    sequence, accumulating into the same `fesystem.K` the unchunked
    single-pass call would have produced in one step.
    """
    if backend not in ("numpy", "torch"):
        raise ValueError(
            f"assemble_stiffness_vectorized: unknown backend={backend!r} -- "
            f"expected 'numpy' (default) or 'torch'.")
    for name, formulation, connectivity in fesystem._blocks:
        block_D = fesystem._per_block_arg(D, name)
        connectivity = np.asarray(connectivity)
        if chunk_size is None:
            conn_chunks = [connectivity]
        else:
            if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
                raise ValueError(
                    f"assemble_stiffness_vectorized: chunk_size must be a "
                    f"positive int (got {chunk_size!r}).")
            conn_chunks = [connectivity[i:i + chunk_size]
                           for i in range(0, len(connectivity), chunk_size)]
        for conn_chunk in conn_chunks:
            if len(conn_chunk) == 0:
                continue
            mt = MeshTransformation(fesystem.mesh, formulation, gauss_order=gauss_order,
                                     connectivity=conn_chunk, backend=backend, device=device)
            ke = tensorized_element_stiffness(mt, block_D, thickness=thickness)
            K_block = scatter_global_stiffness(ke, conn_chunk, formulation.dofs_per_node,
                                                fesystem.n_dof, sparse=fesystem.sparse)
            if fesystem.sparse:
                fesystem.K = (fesystem.K + K_block).tolil()
            else:
                fesystem.K = fesystem.K + K_block
