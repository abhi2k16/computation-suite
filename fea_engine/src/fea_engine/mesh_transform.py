# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
mesh_transform.py -- Wave 11 item 106 (docs/consolidated_future_roadmap.md,
source: tensormesh_analysis_report.md's "Elements and Quadrature" section,
its `Transformation` object): a mesh-wide reference-element/Jacobian
precompute cache.

Integration policy (tensormesh_comparative_analysis.md Sec.0, restated
in the Wave 11 roadmap section): this module takes NO runtime dependency
on the `tensormesh`/`torch-sla` packages -- `MeshTransformation` below is
a native NumPy reimplementation of the *idea* TensorMesh's own
`Transformation` object embodies (cache shape values/gradients/Jacobians
once per mesh instead of recomputing them from closed-form expressions
on every stiffness/internal-force call), using only this package's own
existing quadrature/shape-function machinery (elements/base.py).

Where this sits relative to Wave 0 item 4
------------------------------------------
Wave 0 item 4 (`Element._cached_shape_and_derivs()`) already removed the
O(n*i_d**2) -> O(n*i_d) redundant recomputation the fem_implementation_
lessons.md audit flagged, but that cache is keyed PER ELEMENT TYPE (one
dict living on the formulation instance, shared by every element that
instance is called on) -- it captures "the natural-coordinate shape
values/gradients don't depend on which element we're looking at", but
NOT "the physical gradients/JxW for a whole mesh can be computed once,
outside any per-load-step or per-training-sample loop, and reused".
`MeshTransformation` is that next level: given a Mesh + element
formulation + Gauss order, it computes and stores the STACKED physical
gradients and JxW for every element in the mesh in one pass, so that a
training loop generating many snapshots against the SAME mesh (Wave 7's
neural-surrogate strategy, Wave 10's dataset-diagnostics utility) never
has to reconstitute per-Gauss-point Jacobians from scratch on every
call. It is purely a DATA-LAYOUT addition -- nothing about
`Element.stiffness()`/`FESystem.assemble_stiffness()`'s existing
per-element loop is required to use it, and neither is changed by this
module's existence. It is, however, the direct prerequisite for item
107's tensorized assembly path (vectorized_assembly.py), which consumes
a `MeshTransformation` instead of re-deriving Jacobians itself.

Scope -- read this before reaching for `MeshTransformation` on a new
element type
-------------------------------------------------------------------
`MeshTransformation` supports elements from either of TWO quadrature
families (`Element.quadrature_family`, elements/base.py):

  (a) `"tensor"` (the default) -- elements using the generic,
      tensor-product `gauss_product(order, dim)` Gauss rule, i.e.
      `type(formulation).stiffness is Element.stiffness`, checked
      directly at construction time (see `_check_scope()` below), not
      merely assumed. In this codebase: `Quad4PlaneStress`,
      `Quad8PlaneStress`, `Hex8Solid3D`, `Hex20Solid3D`. Elements that
      override `stiffness()` with something else entirely
      (`Quad4MindlinPlate`'s selective-reduced-integration split, the
      shell formulations) are still explicitly OUT of scope -- each
      would need its own precompute layout matching its own,
      genuinely different, integration convention, not attempted here.

  (b) `"simplex"` (Wave 15 items 125/126, docs/consolidated_future_
      roadmap.md) -- `Tri3PlaneStress`, `Tri6PlaneStress`,
      `Tet4Solid3D`, `Tet10Solid3D`, using `tri_quadrature(order)`/
      `tet_quadrature(order)` (elements/base.py, the order-
      parameterized generalization of the package's original fixed
      `tri_quadrature_3pt()`/`tet_quadrature_4pt()` rules) instead of
      `gauss_product()`. These elements DO override `stiffness()` --
      that override is EXPECTED for this family (a triangle/tet isn't
      mapped from a square/cube, so it was never going to share
      `gauss_product()`), which is exactly why `quadrature_family` is a
      class-level, deliberately-self-declared marker (the same "single
      source of truth on the class" convention `GMSH_NODE_ORDER`
      already established) rather than an implicit `is Element.
      stiffness` identity check -- only the four classes above ever
      set `quadrature_family = "simplex"`, each doing so as a direct,
      reviewed statement "yes, my own stiffness()/mass() do exactly
      the sum-over-{tri,tet}_quadrature(order)-points loop this module
      assumes," not something inferred or guessed at.

      One convention difference to get right at the Jacobian step:
      the simplex element classes' own stiffness()/mass() loops use
      `abs(detJ)` (natural-triangle/tetrahedron "volume" is a fixed
      SIGN convention that can flip with node winding, unlike the
      tensor family's square/cube natural domain, whose existing
      per-element loops use the SIGNED `detJ` directly -- see
      `elements/base.py`'s `Element.stiffness()`). `MeshTransformation`
      reproduces this per-family distinction exactly (signed detJ for
      `"tensor"`, `abs(detJ)` for `"simplex"`) rather than picking one
      convention for both -- tests/test_mesh_transform.py's own
      `TestSimplexFamily` class confirms this numerically against the
      existing per-element path, not just by inspection.

Both families additionally require:
  (c) a SQUARE Jacobian (parametric dim == embedding dim), i.e. not
      embedded (a 1-D truss/beam in 2-D/3-D space, a 2-D shell in 3-D
      space) -- the same restriction `elements/base.py`'s own
      `jacobian()` (as opposed to `jacobian_measure()`) already carries.

GPU/torch side-by-side path (Wave 9 addendum item 135, docs/
consolidated_future_roadmap.md)
-------------------------------------------------------------------
`MeshTransformation.__init__` accepts `backend="numpy"` (default,
UNCHANGED behavior -- every attribute is a plain numpy array exactly as
before this item) or `backend="torch"` (plus `device="cpu"`/`"cuda"`):
the SAME batched `einsum` + batched linear solve this module has always
done, re-expressed on `torch` tensors instead of numpy arrays, since
this whole class was already tensorized/loop-free NumPy (Wave 11 item
106) -- exactly the shape a GPU tensor op wants, and the one stage of
the assembly pipeline Wave 9's original four items (93-96) never
reached (they covered element-tangent autograd and linear-solve
dispatch, not this geometry precompute feeding both). Mirrors
`torch_sparse_solver.py`'s own `_HAS_TORCH`/`_require_torch()`
fail-fast-at-construction convention exactly -- `backend="torch"` with
torch unavailable raises immediately here, not partway through the
precompute.

IMPORTANT: unlike `FESystem(backend="torch")` (which always returns a
plain numpy `U` at the end, since `solve_static()` is a terminal
operation), `backend="torch"` HERE keeps `shape_val`/`shape_grad`/
`jac`/`JxW` as live `torch.Tensor`s on `device`, not converted back to
numpy -- deliberately, so a downstream consumer (item 136,
`vectorized_assembly.py`'s own torch path, not yet built) can keep the
whole assembly pipeline GPU-resident instead of paying a copy-back
this class doesn't need to force. This means the two backends are NOT
drop-in interchangeable for every existing numpy-only consumer of this
class's attributes: `interpolate_point_data()`/
`interpolate_point_data_gradient()` below (Wave 16 item 128) are
explicitly numpy-only and raise a clear `NotImplementedError` if handed
a `backend="torch"` instance, rather than silently mishandling a torch
tensor via `np.einsum`.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .elements.base import (Element, gauss_product, jacobian,  # noqa: F401 (jacobian kept for parity/reference)
                             tri_quadrature, tet_quadrature)

try:
    from .torch_sparse_solver import _HAS_TORCH, _require_torch
except Exception:
    # torch_sparse_solver.py itself already swallows the import-time
    # CUDA-link failure mode into _HAS_TORCH=False (see its own module
    # docstring's HONESTY NOTE) -- this except is only a defensive
    # backstop in case that module's own import path changes shape in
    # the future; _HAS_TORCH=False is always the safe fallback.
    _HAS_TORCH = False

    def _require_torch():
        raise ImportError(
            "mesh_transform.py: torch_sparse_solver.py (this project's "
            "single _HAS_TORCH/_require_torch source of truth) could not "
            "be imported -- backend='torch' is unavailable.")

_SIMPLEX_MEASURE_FACTOR = {2: 0.5, 3: 1.0 / 6.0}   # triangle area / tet volume
                                                    # factor -- see module
                                                    # docstring's (b).


def _check_scope(formulation):
    family = getattr(formulation, "quadrature_family", "tensor")
    if family == "tensor":
        if type(formulation).stiffness is not Element.stiffness:
            raise ValueError(
                f"MeshTransformation (Wave 11 item 106) does not support "
                f"{type(formulation).__name__}: it overrides Element.stiffness() "
                f"with its own quadrature/integration scheme (selective "
                f"reduced integration, or a closed-form formula) while still "
                f"declaring quadrature_family='tensor', so this module's "
                f"generic gauss_product()-based precompute would not match "
                f"what that element's own stiffness()/internal_force() "
                f"actually integrate. See this module's own docstring 'Scope' "
                f"section for the currently-supported element list and why.")
    elif family == "simplex":
        if formulation.dim not in (2, 3):
            raise ValueError(
                f"MeshTransformation: {type(formulation).__name__} declares "
                f"quadrature_family='simplex' with dim={formulation.dim} -- "
                f"only dim=2 (triangle, via tri_quadrature()) or dim=3 "
                f"(tetrahedron, via tet_quadrature()) are supported.")
    else:
        raise ValueError(
            f"MeshTransformation: {type(formulation).__name__} has an "
            f"unrecognized quadrature_family={family!r} -- expected 'tensor' "
            f"or 'simplex'. See this module's own docstring 'Scope' section.")
    if formulation.dim is None:
        raise ValueError("formulation.dim is not set -- not a usable Element subclass.")


def _reference_quadrature(formulation, order):
    """Returns (points, weights) for ONE Gauss point set on the
    formulation's own natural domain, with weights already scaled to
    the convention `JxW = detJ_or_abs_detJ * weight` expects -- i.e.
    weights that, summed with weight 1 over the whole natural domain,
    reproduce that domain's own true measure (1.0 for the tensor
    family's [-1,1]^dim cube/square -- gauss_product()'s own weights
    already have this property; the simplex family's tri_quadrature()/
    tet_quadrature() are barycentric-normalized (sum to 1) instead, so
    THIS function applies the 0.5/1-6 measure factor here, once, so
    every caller downstream (JxW, vectorized_assembly.py) never has to
    know which family produced these weights)."""
    family = getattr(formulation, "quadrature_family", "tensor")
    dim = formulation.dim
    if family == "tensor":
        return gauss_product(order, dim)
    if dim == 2:
        pts, wts = tri_quadrature(order)
    else:
        pts, wts = tet_quadrature(order)
    factor = _SIMPLEX_MEASURE_FACTOR[dim]
    return pts, [w * factor for w in wts]


class MeshTransformation:
    """Precomputed, mesh-wide reference-element/Jacobian data for one
    (mesh, element formulation, Gauss order) combination.

    Parameters
    ----------
    mesh : mesh.Mesh
        Must expose `.nodes` ((n_nodes, dim_embed) array) and, if
        `connectivity` is not given explicitly, `.elements`
        ((n_elements, n_basis) int array) -- a plain single-block
        `Mesh`. For a `MultiBlockMesh`, pass that block's own
        connectivity array explicitly via `connectivity=` (mirroring
        how `FESystem._blocks` already iterates block-by-block, see
        solver.py) rather than trying to make this class multi-block
        aware itself -- one `MeshTransformation` per homogeneous block.
    formulation : elements.base.Element instance
        The element formulation shared by every element referenced by
        `connectivity` -- see the module docstring's Scope section for
        which formulations are currently supported.
    gauss_order : int or None
        Points per direction; defaults to `formulation.gauss_order`.
    connectivity : (n_elements, n_basis) int ndarray or None
        Defaults to `mesh.elements`.
    backend : "numpy" (default) or "torch"
        See this module's own docstring, "GPU/torch side-by-side path"
        section, for the full contract -- in short, `backend="numpy"`
        is byte-for-byte the original (pre-Wave-9-addendum) behavior;
        `backend="torch"` computes the same quantities on `torch`
        tensors and leaves them as tensors (on `device`), not converted
        back to numpy. Unknown values raise `ValueError` immediately.
        `backend="torch"` with torch unavailable raises immediately too
        (at construction, via `_require_torch()`), never partway
        through the precompute.
    device : "cpu" (default) or "cuda"
        Ignored when `backend="numpy"`. Passed straight through to
        `torch.as_tensor(..., device=device)`.

    Attributes
    ----------
    shape_val : (n_gauss, n_basis) ndarray or torch.Tensor
        Shape-function values at each Gauss point -- identical for
        every element (natural-coordinate quantity only), so stored
        once, not stacked per element (matches TensorMesh's own
        `Transformation.shape_val` shape convention exactly).
    shape_grad : (n_elements, n_gauss, n_basis, dim) ndarray or torch.Tensor
        PHYSICAL (post-Jacobian) shape-function gradients at every
        Gauss point of every element -- basis index before the spatial
        dimension, matching TensorMesh's own `[N_e,N_q,N_b,D]` layout.
    jac : (n_elements, n_gauss, dim, dim) ndarray or torch.Tensor
        The Jacobian matrix itself at every Gauss point of every
        element (kept, not just its determinant, since a caller may
        want it directly -- e.g. for a future strain/stress-recovery
        feature; item 107 only consumes `shape_grad`/`JxW`).
    JxW : (n_elements, n_gauss) ndarray or torch.Tensor
        `det(jac) * gauss_weight` -- ready to contract directly against
        a `B^T D B` integrand, exactly the quantity `Element.stiffness()`'s
        own Gauss loop multiplies by (`detJ * w`) at each point.
    connectivity : (n_elements, n_basis) int ndarray
        Copy of the connectivity this was built from (so a caller
        building item 107's global scatter doesn't also need to carry
        the Mesh object around separately). Always a plain numpy array,
        regardless of `backend` -- indices, not floating-point compute.
    backend, device : str
        Copies of the constructor arguments, for a downstream consumer
        (item 136) to introspect without having to be told separately.
    n_elements, n_gauss, n_basis, dim : int
        Convenience shape accessors.
    """

    def __init__(self, mesh, formulation, gauss_order=None, connectivity=None,
                 backend="numpy", device="cpu"):
        if backend not in ("numpy", "torch"):
            raise ValueError(
                f"MeshTransformation: unknown backend={backend!r} -- "
                f"expected 'numpy' (default) or 'torch'. See this "
                f"module's own docstring, 'GPU/torch side-by-side path'.")
        if backend == "torch":
            _require_torch()
            import torch  # noqa: F401 (local import mirrors torch_sparse_solver.py's own convention)
        _check_scope(formulation)
        order = formulation.gauss_order if gauss_order is None else gauss_order
        dim = formulation.dim
        family = getattr(formulation, "quadrature_family", "tensor")
        pts, wts = _reference_quadrature(formulation, order)
        n_gauss = len(pts)

        conn = mesh.elements if connectivity is None else connectivity
        conn = np.asarray(conn)
        n_elements, n_basis = conn.shape
        if n_basis != formulation.n_nodes:
            raise ValueError(
                f"connectivity has {n_basis} nodes/element but "
                f"{type(formulation).__name__}.n_nodes={formulation.n_nodes}.")

        shape_val = np.zeros((n_gauss, n_basis))
        dN_nat_stack = np.zeros((n_gauss, dim, n_basis))
        for q, p in enumerate(pts):
            N, dN_nat = formulation._cached_shape_and_derivs(tuple(p))
            shape_val[q] = N
            dN_nat_stack[q] = dN_nat

        elem_coords = np.asarray(mesh.nodes)[conn]   # (n_elements, n_basis, dim_embed)
        dim_embed = elem_coords.shape[-1]
        if dim_embed != dim:
            raise ValueError(
                f"MeshTransformation is scoped to elements whose parametric "
                f"dim ({dim}) matches their embedding dim ({dim_embed}) -- "
                f"embedded elements (a beam/truss/shell in a higher-dim "
                f"space) are out of scope; see the module docstring.")

        if backend == "numpy":
            # jac[e,q,i,j] = sum_n dN_nat_stack[q,i,n] * elem_coords[e,n,j]
            #              = (dN_natural @ elem_coords) at element e, gauss pt q
            #              -- literally jacobian()'s own formula, batched.
            jac = np.einsum('qin,enj->eqij', dN_nat_stack, elem_coords)

            detJ = np.linalg.det(jac)   # (n_elements, n_gauss)
            if np.any(np.abs(detJ) < 1e-300):
                bad = np.argwhere(np.abs(detJ) < 1e-300)
                raise ValueError(
                    f"degenerate element(s) found (near-zero detJ) at "
                    f"(element,gauss_pt) pairs {bad[:5].tolist()}{'...' if len(bad) > 5 else ''}.")

            # JxW's own detJ convention differs by family, matching each
            # family's existing per-element stiffness()/mass() loop exactly
            # (see the module docstring's (b) note): the tensor family's
            # generic Element.stiffness() loop multiplies by the SIGNED
            # detJ; the simplex classes' own stiffness()/mass() loops all
            # multiply by abs(detJ) instead (their natural-domain "volume"
            # sign can flip with node winding in a way the tensor family's
            # square/cube domain doesn't).
            detJ_for_JxW = np.abs(detJ) if family == "simplex" else detJ

            # Physical gradient: solve J @ dN_phys = dN_nat for each (e,q) --
            # EXACTLY the same np.linalg.solve(J, dN_nat) call every existing
            # B_matrix() makes, just issued once for the whole mesh instead of
            # once per element.
            #
            # dN_nat_stack is explicitly broadcast to jac's full (n_elements,
            # n_gauss) batch shape BEFORE calling solve, rather than relying on
            # np.linalg.solve to broadcast a (n_gauss,)-batch b against jac's
            # (n_elements,n_gauss)-batch a itself. That reliance was a real,
            # version-dependent bug: NumPy < 2.0's linalg.solve requires an
            # EXACT batch-shape match between a and b, with no cross-batch
            # broadcasting at all, and additionally disambiguates "b is a
            # stack of vectors" (b.ndim == a.ndim - 1) from "b is a stack of
            # RHS matrices" (b.ndim == a.ndim) purely by b's ndim -- since
            # dN_nat_stack.ndim (3) is exactly jac.ndim (4) minus 1, older
            # numpy took the VECTOR branch and rejected dN_nat_stack's actual
            # trailing n_basis axis as a malformed vector of the wrong length
            # (numpy>=2.0 added general a/b broadcasting to linalg.solve,
            # which is what let this slip through un-noticed until it was run
            # against an older numpy for the first time). Broadcasting to an
            # explicit (n_elements, n_gauss, dim, n_basis) array first makes
            # a.ndim == b.ndim with identical batch dims on both sides in
            # EVERY numpy version, which every version has always treated
            # identically (the batched-RHS-matrix case, unambiguously).
            dN_nat_b = np.ascontiguousarray(np.broadcast_to(
                dN_nat_stack, (n_elements,) + dN_nat_stack.shape))
            dN_phys = np.linalg.solve(jac, dN_nat_b)   # (n_elements, n_gauss, dim, n_basis)

            self.shape_val = shape_val
            self.shape_grad = np.transpose(dN_phys, (0, 1, 3, 2))   # -> (E,Q,n_basis,dim)
            self.jac = jac
            self.JxW = detJ_for_JxW * np.asarray(wts)[None, :]
        else:
            # ---------------------------------------------------------
            # backend="torch" -- Wave 9 addendum item 135. Identical
            # math to the numpy path above (same einsum contraction
            # string, same signed-vs-abs(detJ) family distinction, same
            # explicit-broadcast-before-solve fix), re-expressed on
            # torch tensors. Kept as live tensors on `device` rather
            # than converted back to numpy -- see this module's own
            # docstring for why (item 136, vectorized_assembly.py's own
            # torch path, consumes these directly).
            # ---------------------------------------------------------
            import torch
            dN_nat_t = torch.as_tensor(dN_nat_stack, dtype=torch.float64, device=device)
            elem_coords_t = torch.as_tensor(elem_coords, dtype=torch.float64, device=device)
            wts_t = torch.as_tensor(np.asarray(wts), dtype=torch.float64, device=device)

            jac_t = torch.einsum('qin,enj->eqij', dN_nat_t, elem_coords_t)

            detJ_t = torch.linalg.det(jac_t)   # (n_elements, n_gauss)
            if torch.any(torch.abs(detJ_t) < 1e-300):
                bad = torch.argwhere(torch.abs(detJ_t) < 1e-300)
                raise ValueError(
                    f"degenerate element(s) found (near-zero detJ) at "
                    f"(element,gauss_pt) pairs {bad[:5].tolist()}{'...' if len(bad) > 5 else ''}.")

            detJ_for_JxW_t = torch.abs(detJ_t) if family == "simplex" else detJ_t

            # Same explicit-broadcast-before-solve discipline as the
            # numpy path (the mesh_transform.py NumPy<2.0 bug this
            # project already found and fixed) -- torch.linalg.solve
            # requires genuinely broadcastable batch dims, not merely
            # "one dim short", so make the match explicit here too
            # rather than relying on torch's own broadcasting rules to
            # interpret dN_nat_t's trailing n_basis axis correctly.
            dN_nat_b_t = dN_nat_t.expand(n_elements, *dN_nat_t.shape).contiguous()
            dN_phys_t = torch.linalg.solve(jac_t, dN_nat_b_t)   # (n_elements, n_gauss, dim, n_basis)

            self.shape_val = torch.as_tensor(shape_val, dtype=torch.float64, device=device)
            self.shape_grad = dN_phys_t.permute(0, 1, 3, 2).contiguous()   # -> (E,Q,n_basis,dim)
            self.jac = jac_t
            self.JxW = detJ_for_JxW_t * wts_t[None, :]

        self.connectivity = conn
        self.gauss_weights = np.asarray(wts)
        self.n_elements = n_elements
        self.n_gauss = n_gauss
        self.n_basis = n_basis
        self.dim = dim
        self.dofs_per_node = formulation.dofs_per_node
        self.backend = backend
        self.device = device


# ---------------------------------------------------------------------
# Wave 16 item 128 (docs/consolidated_future_roadmap.md, source:
# TensorMesh's `Forms` documentation page) -- quadrature-point field
# interpolation.
#
# TensorMesh's Forms page lets any `point_data` key be named directly as
# a `forward(...)` parameter: the base assembler automatically
# interpolates that nodal field to every quadrature point (using the
# SAME basis functions u/v already use), and a `grad<name>` variant is
# available for its physical-space gradient, with zero extra plumbing
# from the user. Adopting that automatic-dispatch-by-parameter-name
# mechanism wholesale is exactly the generic-weak-form-assembler
# architecture change Wave 14/15/16 have each independently declined
# (see this module's own Wave 16 roadmap entry) -- but the underlying
# NEED it serves (get a nodal field's value, and its gradient, at every
# quadrature point of every element, without hand-rolling that
# interpolation loop again for every new coefficient) is genuinely
# useful on its own and does not require that architecture at all: this
# MeshTransformation instance ALREADY has `shape_val` (per-Gauss-point
# basis values, shared across elements) and `shape_grad` (per-element,
# per-Gauss-point PHYSICAL basis gradients) cached from item 106's own
# precompute pass. Interpolating a `point_data` field to quadrature
# points is nothing more than contracting that already-cached basis
# data against the field's own per-element nodal values -- the two
# functions below do exactly that and nothing else.
#
# Wave 14 item 121 gave `Mesh`/`MultiBlockMesh` the `point_data`
# CONTAINER half of TensorMesh's idea (`mesh.point_data[name]`, an
# array indexed by GLOBAL node id, `mesh.point_data[name][i]`
# corresponding to `mesh.nodes[i]`). These two functions are the other
# half: given that same global-node-indexed array and a
# `MeshTransformation` built from a mesh that array came from, produce
# the per-(element, quadrature-point) interpolated value/gradient a
# hand-written `stiffness()`/`mass()` override -- or any future custom
# assembly loop -- can use directly, the same practical capability
# TensorMesh's `kappa`/`WeightedLaplace` example on the Forms page
# demonstrates, without adopting `forward()` dispatch to get it.
# ---------------------------------------------------------------------

def _require_numpy_backend(mesh_transform, caller_name):
    """Wave 9 addendum item 135 guard: interpolate_point_data()/
    interpolate_point_data_gradient() (Wave 16 item 128) are explicitly
    numpy-only consumers of a MeshTransformation's shape_val/shape_grad
    -- they were written and validated before backend="torch" existed,
    and np.einsum() would either raise an opaque error or silently
    misbehave (e.g. via numpy's __array__ coercion) if handed a live
    torch.Tensor. Fail loudly and specifically instead, the same
    principle item 131's own backend="torch" NotImplementedError in
    fix_dofs() already established for this project."""
    backend = getattr(mesh_transform, "backend", "numpy")
    if backend != "numpy":
        raise NotImplementedError(
            f"{caller_name}: mesh_transform was built with "
            f"backend={backend!r}, but this function only supports "
            f"backend='numpy' MeshTransformation instances -- porting "
            f"the Wave 16 item 128 interpolation helpers to torch was "
            f"explicitly out of scope for Wave 9 addendum item 135 (see "
            f"mesh_transform.py's own module docstring). Build a "
            f"second, backend='numpy' MeshTransformation for this call "
            f"if you need both.")


def interpolate_point_data(mesh_transform, field):
    """Interpolate a nodal (`point_data`-style) field to every Gauss
    point of every element in `mesh_transform`.

    Parameters
    ----------
    mesh_transform : MeshTransformation
    field : (n_nodes_total, ...) array-like
        A field indexed by GLOBAL node id -- the same convention
        `Mesh.point_data[name]`/`MultiBlockMesh.point_data[name]`
        already use (Wave 14 item 121): `field[i]` is the value at
        `mesh.nodes[i]`, with any number of trailing "field shape"
        dimensions (`()` for a scalar field, `(dim,)` for a vector
        field, etc.) -- exactly TensorMesh's own "per-node, with one
        trailing field shape" description of `point_data`.

    Returns
    -------
    (n_elements, n_gauss, ...) ndarray
        The field's value at every (element, Gauss point) pair, trailing
        dims preserved unchanged -- e.g. a scalar field returns
        `(n_elements, n_gauss)`, a `(dim,)`-vector field returns
        `(n_elements, n_gauss, dim)`.

    Notes
    -----
    Mathematically this is just `value(x_q) = sum_n N_q[n] * field[node_n]`
    at each element's own Gauss point `q` -- the identical formula every
    existing `shape_and_derivs()`-based interpolation in this codebase
    already uses (e.g. how a displacement field is interpolated from
    nodal DOFs inside `internal_force()`), just issued once for the
    whole mesh via `mesh_transform.shape_val` instead of re-evaluating
    shape functions per element.
    """
    _require_numpy_backend(mesh_transform, "interpolate_point_data")
    field = np.asarray(field)
    conn = mesh_transform.connectivity
    if field.shape[0] <= int(conn.max()):
        raise ValueError(
            f"interpolate_point_data: field has {field.shape[0]} node "
            f"entries but mesh_transform.connectivity references node "
            f"index {int(conn.max())} -- field must be indexed by the "
            f"SAME global node ids as the mesh this MeshTransformation "
            f"was built from (the Mesh.point_data[...] convention, "
            f"Wave 14 item 121).")
    nodal = field[conn]   # (n_elements, n_basis, ...)
    # value[e,q,...] = sum_n shape_val[q,n] * nodal[e,n,...]
    return np.einsum('qn,en...->eq...', mesh_transform.shape_val, nodal)


def interpolate_point_data_gradient(mesh_transform, field):
    """Physical-space GRADIENT of a nodal field at every Gauss point of
    every element in `mesh_transform` -- the `grad<name>` half of
    `interpolate_point_data()` above (TensorMesh's Forms page: "grad +
    key in point_data ... automatic -- request it by name, do not pass
    anything extra"). Same `field`/return-shape convention as
    `interpolate_point_data()`, with one extra trailing axis of size
    `mesh_transform.dim` for the gradient direction: a scalar field
    returns `(n_elements, n_gauss, dim)`; a `(dim,)`-vector field
    returns `(n_elements, n_gauss, dim, dim)` (field component, then
    gradient direction).

    Uses `mesh_transform.shape_grad` -- the ALREADY-cached PHYSICAL
    (post-Jacobian, `J^-T`-chain-rule-applied) gradients item 106's own
    precompute pass computed -- so, exactly like TensorMesh's own
    automatic `gradkappa`, no extra Jacobian work happens here at all;
    this is pure contraction against data already sitting in memory.
    """
    _require_numpy_backend(mesh_transform, "interpolate_point_data_gradient")
    field = np.asarray(field)
    conn = mesh_transform.connectivity
    if field.shape[0] <= int(conn.max()):
        raise ValueError(
            f"interpolate_point_data_gradient: field has {field.shape[0]} "
            f"node entries but mesh_transform.connectivity references "
            f"node index {int(conn.max())} -- field must be indexed by "
            f"the SAME global node ids as the mesh this MeshTransformation "
            f"was built from (the Mesh.point_data[...] convention, "
            f"Wave 14 item 121).")
    nodal = field[conn]   # (n_elements, n_basis, ...)
    # grad[e,q,...,d] = sum_n shape_grad[e,q,n,d] * nodal[e,n,...]
    return np.einsum('eqnd,en...->eq...d', mesh_transform.shape_grad, nodal)
