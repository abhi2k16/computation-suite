# TensorMesh (docs.tensor-mesh.com) — Analysis Report

**Source:** https://docs.tensor-mesh.com/ (User Guide: Concepts, Elements and Quadrature, Mixed Assembly, Sparse Solvers, Differentiability)
**Maintainer:** CAMLab, ETH Zürich (github.com/camlab-ethz/TensorMesh)
**Date reviewed:** 2026-09-13

## 1. What it is

TensorMesh is a finite-element library built natively on top of PyTorch, rather than a legacy FEM engine with a PyTorch wrapper bolted on. Its central design claim is that every stage of the classic FEM pipeline — mesh, weak-form assembly, sparse linear system, boundary-condition handling, solve — is itself a PyTorch object (`nn.Module` or `autograd.Function`), so the pipeline inherits autograd, device placement, and JIT tracing for free. The library separates concerns cleanly: TensorMesh owns the meshing/element/assembly/boundary-condition layer, and delegates all sparse linear algebra (direct and iterative solves, adjoint gradients, GPU/multi-GPU dispatch) to a sibling package, `torch-sla`.

## 2. Architecture

The documented pipeline is:

```
Mesh → Assembler → SparseMatrix → Condenser → Solve
```

- **Mesh** — an `nn.Module` holding point coordinates and connectivity as buffers; built-in generators (`gen_rectangle`, `gen_circle`, `gen_cube`), meshio I/O, and graph/partitioning utilities for distributed runs.
- **Assembler** (`ElementAssembler` / `NodeAssembler` / `FacetAssembler` / `MixedElementAssembler`) — the user-facing extension point. A weak form is written as a plain `forward()` method returning the integrand at a quadrature point; the library handles geometry, quadrature, and the global sparse scatter.
- **SparseMatrix** — a subclass of `torch_sla.SparseTensor`; carries `.solve()` and `.nonlinear_solve()`.
- **Condenser** — applies Dirichlet BCs via static condensation, reducing to the interior-DOF system.
- **Solve** — dispatched through `torch-sla` to one of six backends (below).

This is a fairly conventional isoparametric-FEM architecture (reference element → Jacobian → physical gradients → quadrature-weighted assembly), but the implementation detail that matters is **tensorized, loop-free assembly**: basis functions and quadrature are evaluated once for the whole mesh, and the user's `forward` runs on a tensor with element and quadrature dimensions already broadcast-ready, so assembly compiles down to a single GPU kernel rather than a Python loop over elements.

## 3. Element and quadrature system

Seven reference shapes (`Line`, `Triangle`, `Quadrilateral`, `Tetrahedron`, `Hexahedron`, `Prism`, `Pyramid`) each support linear-through-high-order variants (documented up to `triangle66`, `hexahedron64`+, etc.), selected by an `order=` argument on the mesh generators rather than a different class per order. The reference→physical pipeline is explicitly staged and documented step by step: Lagrange node placement → polynomial space (`P_k` for simplices, `Q_k` for tensor-product shapes) → Vandermonde-inverted hat functions → Gauss quadrature → shape values → cell Jacobian → physical gradients — all cached as buffers inside a `Transformation` object so `.to(device)`/`.double()` propagate automatically. Notably, higher-order fields can be **topological** (edge/vertex DOF maps) rather than tied to mesh order, which is what lets a Taylor-Hood P2 velocity space sit on top of an imported linear Gmsh mesh without remeshing. Documented limits: 3-D topological DOF maps stop at edge DOFs (no face DOFs yet — tetra P2 works, higher does not), and quadrature tables are capped at degree 7 (~P3-P2 in practice).

## 4. Mixed assembly

`MixedElementAssembler` is the multi-field coupling mechanism (Stokes, poroelasticity, Boussinesq convection): fields are declared once (trial/test names, order, vector components), the coupled bilinear form is written as one scalar `forward`, and the library extracts each block by zeroing the other fields — enforced by a runtime check that the integrand really is bilinear. A `BlockLayout` object provides the DOF-vocabulary (masks, node ids, split/cat between the block vector and per-field tensors) so the same `Condenser`/`.solve()`/autograd machinery used for single-field problems works unchanged on the block system. This is a clean generalization, though the docs explicitly flag that mixed assembly is **single-device only** — the distributed-FEM path only covers single-field assemblers, since block DOF numbering across ranks doesn't exist yet ("planned ROADMAP item").

## 5. Sparse solvers (`torch-sla`)

This is the most operationally significant design choice: TensorMesh has no solver code of its own. `tensormesh.sparse` is a hard, import-time dependency on `torch-sla`, and the docs are explicit that legacy in-tree fallbacks (`spsolve`, `nonlinear_solve`) are deprecated and scheduled for removal. `torch-sla` gives:

- Six verified backends — SciPy (CPU default), native PyTorch Krylov (CPU/CUDA/ROCm, the path for >2M DOF), NVIDIA cuDSS (CUDA default direct), STRUMPACK (portable direct, incl. AMD ROCm), PyAMG, NVIDIA AmgX — switchable via one `backend=` keyword, with symmetry/SPD auto-detected per call (no manual `is_spd` hint).
- Batched RHS support (`b` of shape `[n_dof, n_batch]`) that amortizes one factorization across many back-substitutions — described as the workhorse for the `tensormesh.dataset` ML data-generation path.
- `.nonlinear_solve(residual, u0, *params, method="newton"|"picard"|"anderson")` — Newton with Armijo line search by default, Jacobian obtained via autograd, and a backward pass that costs one adjoint linear solve regardless of how many Newton iterations the forward took.

## 6. Differentiability

This is the library's headline feature, and it's documented in unusual depth for a technical-docs page (worked examples, not just an API description):

- **Mechanism.** `SparseMatrix.solve()` is a custom `torch.autograd.Function`. Rather than differentiate through the solver internals, backward solves one adjoint system `AᵀλAᵀλ = ∂L/∂u` and assembles gradients in closed form: `∂L/∂A_ij = -λᵢuⱼ`, `∂L/∂b = λ`. Cost is therefore a flat "2× forward solve," independent of DOF count or assembly complexity.
- **Correctness evidence.** The docs show an actual finite-difference cross-check (not just a claim) on a 35-node mesh, reporting ~11 digits of agreement between autograd and central FD gradients.
- **Worked example 1 (coefficient identification):** recovering a spatially varying Poisson coefficient field from synthetic observations via Adam + the adjoint gradient — 5000 steps, loss drops ~7 orders of magnitude, ~6×10⁻⁵ relative error in the recovered field, with a named, physically sensible failure mode (near-zero-gradient region loses identifiability).
- **Worked example 2 (topology optimization):** SIMP-based thermal compliance minimization via a purpose-built `OCOptimizer` (Optimality Criteria), driven directly by `rho.grad` from the adjoint — ~8.6× compliance reduction at an exactly-held volume constraint.
- **NN coupling patterns.** Three named patterns (NN → coefficient field, NN → Dirichlet value, NN → per-element stiffness modifier) — explicitly requiring no TensorMesh-specific neural layer, just standard `nn.Module`s wired into the same graph. The docs are honest that `tensormesh.nn` currently ships only container utilities (`BufferDict`/`BufferList`), not learnable layers — "higher-level neural-operator... building blocks are planned for a future release."
- **Backend caveat, stated plainly:** gradients are correct through every backend, but only the `pytorch`/`auto` backends keep the forward solve inside the autograd graph; SciPy/cuDSS/CuPy-backed solves are correct-but-opaque (the adjoint is analytic, so it doesn't need to "see into" the forward solver, but you can't inspect intermediate solver state from autograd on those backends).

## 7. Strengths, read critically

1. **The differentiability claims are backed by an actual numerical check, not just an architecture diagram.** The FD-vs-autograd table and the two worked examples (with quantitative convergence numbers and named failure modes) are the kind of evidence that's easy to omit and the authors didn't.
2. **Solver modularity is real, not aspirational.** Six backends "each checked against a reference solution to at/near machine-precision" is a specific, falsifiable claim, and the size-based rule-of-thumb table (direct <2M DOF, iterative Krylov 2M–169M DOF, distributed beyond) reads like it came from an actual benchmark sweep, not a marketing table.
3. **The adjoint-cost argument (2× forward solve, independent of DOF count and Newton iteration count) is the correct theoretical result** for implicit differentiation through a fixed point / linear solve, and matches the framing used in Deep Equilibrium Models and implicit-layer literature more broadly.
4. **Honesty about limitations is a positive signal.** The docs explicitly flag: mixed assembly is single-device only; 3-D topological DOF maps stop short of face DOFs; quadrature tops out around P3-P2; `tensormesh.nn` has no learnable layers yet; legacy solver entry points are being removed. A docs page that surfaces its own gaps this specifically is more trustworthy than one that doesn't.
5. **The weak-form authoring model (`forward()` returning a scalar integrand, dispatched by argument name) is a genuinely nice ergonomic choice** — it reads close to the math (`gradu @ gradv`, `gradu.diagonal().sum()` for divergence) and generalizes uniformly from scalar Laplace forms to multi-field Taylor-Hood Stokes without a different mental model.

## 8. Gaps and open questions

- **No independent verification here.** Everything above is TensorMesh's own documentation; I have not run its code, and the fetched pages don't include third-party benchmarks, a paper, or a citation record — the "Citing TensorMesh" page exists but wasn't fetched. Claims about accuracy and performance should be treated as vendor-reported until cross-checked.
- **Nonlinear solid mechanics coverage is claimed (hyperelastic beam, Hertzian contact, J2 plasticity, Drucker-Prager geomechanics all appear in the example gallery listing) but none of those example pages were fetched**, so depth/correctness of the nonlinear solid-mechanics path — the part most relevant to comparison against `fea_engine` — is unverified from what was read.
- **The `torch-sla` adjoint is for *linear* systems and for `.nonlinear_solve()` wrapped as one opaque call.** It is not documented as a general "differentiate through an arbitrary user-written Newton loop" primitive — closer to what Wave 10 item 98 in `fea_engine`'s roadmap is scoping, but narrower in that TensorMesh owns the Newton loop itself (`A.nonlinear_solve(...)`) rather than exposing an adjoint wrapper around a solver the user already wrote.
- **No trainable-correction-against-data story analogous to the Saverio et al. 2026 RANS paper reviewed earlier in this project.** TensorMesh's worked examples are inverse-parameter-identification and topology optimization, not "combine a physics residual with a learned correction term and calibrate against reference data" — the `f_θ(w)` pattern remains something `fea_engine`'s Wave 10 would still be adding, not something TensorMesh already ships.

## 9. Relevance to the `fea_engine` project

TensorMesh is a close peer to where `fea_engine` is heading via Wave 9/10, and worth treating as a reference implementation rather than mere prior art:

- Its adjoint-wrapped `SparseMatrix.solve()` is essentially Wave 10 item 98 (whole-solver implicit-differentiation layer) already built and shipped, including the exact `AᵀλAᵀλ = ∂L/∂u` formulation `fea_engine`'s roadmap cites from the RANS paper — useful as a design reference and correctness cross-check (the FD-vs-autograd table format is worth imitating directly in `fea_engine`'s own validation).
- Its **tensorized, loop-free assembly** (basis/quadrature evaluated once, broadcast across all elements) is the mature version of Wave 10 item 102 (`vmap` batched multi-element autograd tangent assembly) — `fea_engine`'s current per-element Python loop (items 93/96) is architecturally the "before" picture of what TensorMesh's `Transformation`/assembler pattern already is.
- TensorMesh's `OCOptimizer` (topology optimization via OC + adjoint gradient) is a concrete, narrower example of the "differentiable design-variable calibration" pattern that Wave 10 item 100 (explicit residual-minimization calibration) is reaching toward, though TensorMesh's is compliance-minimization rather than data-fitting.
- Conversely, `fea_engine` has things TensorMesh's fetched pages don't show: an explicit corotational/director shell formulation, contact with augmented Lagrangian, and (per the Saverio et al. review) the specific `f_θ(w)` trainable-correction target is still open ground for both.

## Sources

- [Concepts](https://docs.tensor-mesh.com/user_guide/concepts.html)
- [Elements and Quadrature](https://docs.tensor-mesh.com/user_guide/elements_and_quadrature.html)
- [Mixed Assembly](https://docs.tensor-mesh.com/user_guide/mixed_assembly.html)
- [Sparse Solvers](https://docs.tensor-mesh.com/user_guide/linear_solvers.html)
- [Differentiability](https://docs.tensor-mesh.com/user_guide/differentiability.html)
