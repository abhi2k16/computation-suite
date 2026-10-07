# TensorMesh: Content Analysis, Competitive Positioning, and Integration Opportunities for `computation-suite`

**Prepared:** 2026-09-06
**Scope:** `www.tensor-mesh.com`, `docs.tensor-mesh.com`, and `github.com/camlab-ethz/TensorMesh`, analyzed against the current state of `fea_engine` and `rom_engine` in this workspace.

---

## 1. Executive summary

TensorMesh is an open-source finite element (FEM) library from ETH Zürich's CAMLab, built natively on PyTorch. Its central pitch is that FEM assembly and solving become first-class, GPU-resident, autograd-differentiable PyTorch operations — useful for inverse design, topology optimization, and physics-informed/neural-surrogate training. It is young (v0.2.0, self-rated "immature" on its own comparison table), single-lab in origin, and tied to a specific 2026 paper (*TensorGalerkin*) and a companion sparse-linear-algebra package (`torch-sla`).

Relative to `computation-suite`, TensorMesh is not a competing product so much as a differently-shaped one. `fea_engine` + `rom_engine` form a narrow-but-deep structural-mechanics research stack: real nonlinear solid/shell/beam elements, genuinely advanced nonlinear-stability solvers (a generalized multi-mode Koiter-Newton bifurcation tracker with no counterpart in TensorMesh's documented feature set), and an entire model-order-reduction layer (POD, certified reduced-basis methods, classical MOR, nonlinear ROM, NNM/backbone-curve tracking) that TensorMesh does not attempt at all. TensorMesh is a broad-multiphysics, GPU-first, differentiable *full-order* solver with a comparatively narrow nonlinear-solid-mechanics feature set (no arc-length/buckling solver evident, no volumetric-locking cure, isotropic J2 + Drucker-Prager plasticity only, contact scoped to simple cases).

The practical value of TensorMesh to this project is therefore not "adopt it" but three narrower things: (1) a working reference architecture for autograd-derived tangent stiffness, which is directly relevant to `fea_engine`'s existing mix of analytic and finite-difference/complex-step tangents; (2) a design reference for a GPU sparse-solve path for large systems arising from `fea_engine`'s higher-order/mesh-grading work; and (3) a template for the mixed-field assembly pattern (LBB-stable Taylor-Hood-style formulations) that is the architecturally correct fix for the volumetric-locking gap identified in `fea_engine`'s 3D solid elements, and for the mixed-formulation approach that was previously attempted and rejected on `Shell4MITCCorotational`.

**Integration policy (project decision).** `computation-suite` will not take a runtime dependency on the `tensormesh` or `torch-sla` packages. Any new capability inspired by this analysis will be implemented natively, in-house, using only `torch` itself (a general-purpose library, not TensorMesh's own package) as the computational substrate where PyTorch's autograd or GPU tensor support is genuinely needed. TensorMesh's documentation, module design, and example code are used strictly as a design reference — the same role the Belytschko/Gockenbach textbooks played in earlier audits of this codebase — never as imported code. Section 6 below is written entirely in that frame.

---

## 2. What TensorMesh is

### 2.1 Origin and packaging

TensorMesh is the FEM-solver component of a broader framework called **TensorGalerkin**, documented in an arXiv preprint (Wen, Chi, Yu, Moseley, Michelis, Ren, Sun & Mishra, "Learning, Solving and Optimizing PDEs with TensorGalerkin: an Efficient High-Performance Galerkin Assembly Algorithm," 2026). Its sparse linear algebra is factored into a separate companion package, **`torch-sla`** (Chi & Wen, "torch-sla: Differentiable Sparse Linear Algebra with Adjoint Solvers and Sparse Tensor Parallelism for PyTorch," 2026). It is developed at CAMLab (Computational and Applied Mathematics Lab), ETH Zürich, with support from the ETH AI Center.

It is pip-installable (`pip install tensormesh-fem`), Apache-2.0 licensed, at release v0.2.0 ("Mixed multi-field assembly, distributed FEM & open-domain waves") as of this analysis. On GitHub: 218 stars, 24 forks, 248 commits, 1 open issue, 2 open pull requests — a small, active, but young community rather than an established production dependency.

*Note on provenance:* several citation details (arXiv IDs in the 2601–2602.xxxxx range, a 2026 publication year, hardware described as newly released) suggest this content may be very recent, forward-dated, or a pre-print placeholder. Benchmark numbers and citation metadata should be treated as the project's own self-reported claims, not independently verified facts, until cross-checked.

### 2.2 Core technical claims

- **GPU-native and differentiable.** Moving an entire FEM workflow to GPU is a one-line change (`mesh.cuda()`); PyTorch autograd flows through assembly and solve, enabling end-to-end differentiable PDE pipelines (gradients of outputs with respect to mesh coordinates, material parameters, or boundary data, "for free").
- **Tensorized assembly.** A fully tensorized Map-Reduce algorithm ("TensorGalerkin") fuses per-element operations into monolithic GPU kernels, eliminating Python-level element loops. This is claimed to deliver order-of-magnitude speedups over CPU-based FEM stacks.
- **JIT-free / eager execution.** No compilation step (unlike JAX-FEM's XLA tracing or FEniCS/Firedrake's UFL-to-C form compilation). Dynamic meshes, adaptive refinement, and interactive workflows work without recompilation latency or opaque traces — positioned as the key debugging/dynamic-mesh advantage over JAX-based competitors.
- **Pythonic, DSL-free weak forms.** Custom weak forms are written as plain Python classes (`ElementAssembler`/`NodeAssembler` subclasses whose `forward()` method returns the integrand), rather than a form language like UFL.
- **Element and mesh support.** Triangular, tetrahedral, pyramid, and prismatic elements; automated mesh generation for common geometries; Gmsh and VTK-HDF5 I/O.
- **Mixed multi-field assembly.** `MixedElementAssembler` declares each field's interpolation order independently of the mesh — e.g., quadratic velocity / linear pressure (Taylor-Hood, LBB-stable) spaces generated topologically even on a linear mesh, without remeshing.
- **Flexible solvers.** Linear, nonlinear, and eigenvalue solves across six sparse-solver backends via `torch-sla` (SciPy direct/iterative and a native PyTorch Krylov solver in the base install; cuDSS, PyAMG, STRUMPACK, and NVIDIA AmgX as opt-in extras), with autograd support, batched solves, and multi-GPU scaling.

### 2.3 Module architecture

The stated workflow is **Mesh → Assembler → SparseMatrix → Condenser → Solve**:

| Module | Description |
|---|---|
| `tensormesh.mesh` | Mesh data structure; built-in generators (`gen_rectangle`, `gen_circle`, `gen_cube`, `gen_L`, …); Gmsh/VTK-HDF5 I/O |
| `tensormesh.element` | Shape functions, quadrature rules, element transformations (geometric order 1–4) |
| `tensormesh.assemble` | `ElementAssembler`, `NodeAssembler`, `FacetAssembler`, `FacetBilinearAssembler`; `MixedElementAssembler` for multi-field block systems |
| `tensormesh.sparse` | `SparseMatrix` (subclass of `torch_sla.SparseTensor`); linear/nonlinear sparse solves across six backends |
| `tensormesh.operator` | `Condenser` (Dirichlet BCs via static condensation); `BlochReducer` (Bloch-Floquet periodic BCs); `robin_operator`/`port_source` (wave boundary operators) |
| `tensormesh.ode` | Time integrators: explicit/implicit Euler, midpoint, Runge–Kutta |
| `tensormesh.dataset` | Parametric PDE dataset generation (Poisson, Heat, Wave, linear elasticity) for ML training |
| `tensormesh.visualization` | Matplotlib and PyVista plotting backends |
| `tensormesh.functional` | Tensor utilities for FEM (elasticity, Voigt notation, common operations) |
| `tensormesh.material` | Material property definitions for solid mechanics |
| `tensormesh.optimizer` | Optimization algorithms (e.g., the Optimality Criteria method for topology optimization) |
| `tensormesh.distributed` | Multi-GPU distributed assembly (partitioned meshes, graph coloring) |
| `tensormesh.nn` | Neural-network integration utilities for physics-informed/neural-operator training |

### 2.4 Example gallery and solid-mechanics scope

The gallery spans 11 categories and 50+ runnable examples: basics/visualization, Poisson (2D/3D, h-adaptivity), diffusion (heat equation, Allen-Cahn phase field), wave equation and complex-valued Helmholtz, phononic crystals (Bloch-Floquet band structures, validated against COMSOL), open-domain waves (PML absorbing layers, waveguide ports), modal analysis, solid mechanics, fluid mechanics (Taylor-Hood Stokes, lid-driven cavity, cylinder flow, Rayleigh-Bénard, Taylor-Green vortex), magnetostatics (stabilized nodal curl-curl), inverse design/topology optimization, physics-informed learning, ML dataset generation, and distributed multi-GPU FEM.

The solid-mechanics gallery specifically — described as "a progressive solver ladder" — covers: cantilever beam (linear elasticity baseline), hyperelastic beam (Neo-Hookean, large deformation via energy minimization), Hertzian contact, J2 plasticity with isotropic hardening on a strip, and three geomechanics problems (Drucker-Prager triaxial compression, an elastic strip footing, and a Drucker-Prager strip footing). No arc-length, buckling, or bifurcation-tracking example is documented anywhere in the gallery.

---

## 3. Positioning among open-source FEA packages

Open-source FEM libraries cluster into roughly three lineages; TensorMesh belongs to the newest one.

**Form-compiler / production HPC lineage — FEniCS (DOLFINx), Firedrake, MFEM.** Weak forms are written in a domain-specific language (UFL) compiled to optimized C code at "form-compile" time, run MPI-parallel across CPU clusters, with a decade-plus of production use. None has native autograd (FEniCS requires the external `dolfin-adjoint` package); GPU support is absent or partial. These remain the most mature and most trusted for correctness at scale.

**Minimalist/pedagogical lineage — scikit-fem.** Pure Python, no compiler, easy to read and debug, no GPU, no autograd. Philosophically closest to how `fea_engine` itself is built (direct, readable, no hidden codegen layer), but scoped to general PDE assembly rather than the structural-mechanics-specific nonlinear-solver research `fea_engine` undertakes.

**Differentiable/ML-native lineage — JAX-FEM, torch-fem, TensorMesh.** All three exist specifically because conventional FEM codes cannot backpropagate through a solve. JAX-FEM's differentiability comes from JAX tracing/JIT — fast once compiled, but tracing makes dynamic meshes and adaptive refinement awkward, since a mesh change re-triggers a trace. torch-fem is closer architecturally to TensorMesh (also PyTorch-native) but is rated only ⚠️ on efficiency in TensorMesh's own comparison table. TensorMesh's specific claim within this group is combining GPU-fast tensorized assembly *with* eager (non-JIT) execution *and* autograd — a combination none of the others fully achieve simultaneously.

TensorMesh's own published feature-comparison table (self-reported, included here for reference, not independently verified):

| Feature | FEniCS | scikit-fem | JAX-FEM | torch-fem | TensorMesh |
|---|---|---|---|---|---|
| Custom weak forms (Pythonic) | ⚠️ | ✅ | ❌ | ❌ | ✅ |
| Easy install | ❌ | ✅ | ⚠️ | ✅ | ✅ |
| Easy debug | ❌ | ✅ | ❌ | ✅ | ✅ |
| Easy I/O | ❌ | ❌ | ❌ | ❌ | ✅ |
| Large meshes | ✅ | ✅ | ❌ | ❌ | ✅ |
| GPU support | ✅ | ❌ | ✅ | ✅ | ✅ |
| Efficiency | ✅ | ❌ | ✅ | ⚠️ | ✅ |
| End-to-end autograd | ⚠️ | ❌ | ✅ | ✅ | ✅ |
| DL integration (PyTorch) | ❌ | ❌ | ⚠️ | ✅ | ✅ |
| Maturity | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ |

The one axis TensorMesh concedes against the established tools is maturity — an honest signal, and consistent with its small GitHub footprint and single-paper origin.

---

## 4. Benchmark methodology and results

TensorMesh's performance claims come from a separate repository (`camlab-ethz/tensormesh-bench`), tested on an 8-core AMD EPYC 9005-series (Zen 5) node with one NVIDIA RTX PRO 6000 Blackwell GPU and 64 GB DDR5 ECC — current, workstation-class hardware, so the headline "10x GPU speedup" figure should be read as GPU-vs-CPU on that specific configuration, not a universal constant.

Two forward problems and one inverse problem were benchmarked against FEniCS(DOLFINx), Firedrake, MFEM, scikit-fem, JAX-FEM, and torch-fem:

1. **3D Poisson** on the unit cube, constant source, homogeneous Dirichlet BCs.
2. **3D linear elasticity** on a hollow-cube domain (E=1, ν=0.3, constant body force), introducing geometric complexity via a cubic cavity.
3. **Topology optimization (inverse problem)** — 2D SIMP compliance minimization on a 60×30 Quad4 cantilever mesh (1,891 nodes, 1,800 elements), Method of Moving Asymptotes optimizer, 51 iterations, benchmarked only against JAX-FEM (the only other framework in the comparison with end-to-end autograd).

All frameworks were forced onto the same iterative solver configuration (BiCGSTAB, Jacobi preconditioner, 1e-10 relative/absolute tolerance, 10,000 max iterations) for a controlled comparison — the one documented exception is torch-fem's CUDA path, which uses conjugate gradient instead because its GPU backend does not ship BiCGSTAB.

The most technically notable detail is in the inverse-problem benchmark: TensorMesh's SIMP sensitivity (∂compliance/∂density) is **not hand-derived**. It is obtained purely by backpropagating through the differentiable assembly and sparse solve via PyTorch's reverse-mode autograd; the closed-form adjoint expression is computed only as a consistency check against the autograd result, not used in the optimization loop itself. Both TensorMesh and JAX-FEM converge to the same canonical truss-like topology, with TensorMesh reported as running noticeably faster end-to-end.

---

## 5. Detailed comparison against `computation-suite`

`fea_engine` and `rom_engine`, taken together, occupy a very different shape than TensorMesh: narrow physics domain (structural/solid mechanics only) but unusually deep along two axes — nonlinear-stability solving and model-order reduction — that TensorMesh does not address at all.

| Axis | TensorMesh | `fea_engine` + `rom_engine` |
|---|---|---|
| Core purpose | Full-order differentiable PDE solves (forward + inverse), GPU-scaled | Structural FEA research stack **paired with** a full model-order-reduction layer |
| Physics scope | Broad multiphysics: Poisson, wave/Helmholtz, Maxwell, fluids (Navier-Stokes), phononic crystals, solid mechanics | Solid/structural mechanics only, but deep: nonlinear kinematics, plasticity, buckling/bifurcation |
| Nonlinear solid coverage | Hyperelasticity, J2 plasticity, Drucker-Prager, basic (single-node/rigid-obstacle-class) contact — no arc-length/buckling/bifurcation solver documented | Full and modified Newton, Crisfield arc-length, and a generalized **multi-mode Koiter-Newton bifurcation tracker** — solver research with no documented TensorMesh counterpart |
| Model order reduction | **None** — no POD, no Galerkin projection, no frequency-domain ROM, no certified error bounds, no classical MOR, no nonlinear/NNM ROM | `rom_engine`'s entire purpose: POD, intrusive Galerkin, affine parametric decomposition, certified reduced-basis (greedy sampling + SCM bounds), classical MOR (Krylov moment matching, balanced truncation, Hankel-norm AAK), nonlinear RBF-surrogate ROM, NNM/harmonic-balance backbone curves |
| Tangent stiffness | Autograd-derived by construction | Mixed: analytic closed-form for most elements and J2 plasticity; finite-difference/complex-step for a few (`Tet4NeoHookean`, historically `Tet10SolidTL`, and one contact element by deliberate choice) |
| Element library | General triangular/tetrahedral/pyramid/prismatic PDE elements | Purpose-built structural elements: Tri3/Quad4/Quad8/Tet4/Tet10/Hex8/Hex20 solids, Beam2D/3D (incl. corotational), Shell4MITC/MITCCorotational, with genuine patch-test validation on the solids |
| Locking/hourglass handling | Mixed-field assembly available (architecturally the correct fix) but not documented as applied to a locking cure specifically | Volumetric locking has **no cure** in the 3D solids (no B-bar/SRI/mean-dilatation); reduced integration has a hourglass-mode *detector* but no stabilizer — a known, documented gap |
| GPU / autograd | Native, central to the design | Absent — NumPy/SciPy, CPU-only |
| Sparse solvers | Six backends via `torch-sla` (SciPy, native Krylov, cuDSS, PyAMG, STRUMPACK, AmgX) | SciPy sparse (direct/iterative) only |
| Validation culture | Benchmark suite vs. six other frameworks; SIMP autograd-vs-closed-form cross-check | Extensive hand-derivable benchmarks (elastica comparisons, complex-step cross-checks) and, most recently, a full lesson-by-lesson audit of the codebase against two nonlinear-FEM textbooks (Gockenbach; Belytschko/Liu/Moran/Elkhodary) |
| Maturity | v0.2.0, single-lab, self-rated ⚠️ | Built incrementally with heavy validation discipline at each step; also young/single-developer in the same practical sense |

**Bottom line:** these are not competing products. TensorMesh is a general-purpose differentiable FEM engine with a comparatively narrow nonlinear-solid-mechanics feature set; `computation-suite` is a narrow-domain, unusually deep structural-mechanics-plus-ROM stack. TensorMesh is best treated as a source of specific architectural ideas and validation targets rather than a dependency or a replacement for anything already built here.

---

## 6. Build-native-in-PyTorch plan (TensorMesh used only as a reference)

No module of `tensormesh` is imported anywhere in this plan. Every item below is a `computation-suite`-native script whose only dependency beyond the existing NumPy/SciPy stack is `torch` itself; TensorMesh's public documentation and examples are cited only as the source of the *design idea*, not of any code.

### 6.1 Own PyTorch sparse-solve module (targets: no GPU solve path today)

Write a thin, self-contained module — e.g. `fea_engine/torch_sparse_solver.py` — that takes an already-assembled SciPy sparse matrix (from the existing Phase 2 sparse-assembly path) and converts it to a `torch.sparse_csr_tensor`, then solves via PyTorch's own sparse linear algebra (or a hand-written preconditioned CG/BiCGSTAB loop using plain `torch` tensor ops), optionally GPU-resident via `.to("cuda")`. This needs nothing from `tensormesh.sparse`/`torch-sla` — both are referenced only for two design choices worth copying: (a) a pluggable backend-selection interface (CPU SciPy path vs. GPU `torch` path chosen by a single flag or by device auto-detection) and (b) their solver-diagnostics logging convention (their quickstart prints `n`, `nnz`, `dtype`, `device`, `symmetric`, `backend`, `method` on every solve) as a template for a similar one-line diagnostic print in our own solver. Scope this to the systems that actually get large — higher-order elements (Hex20/Tet10) and graded meshes.

### 6.2 Own PyTorch neural-surrogate training strategy (targets: RBF-only nonlinear ROM)

Extend `rom_engine.nonlinear_rom`'s existing `TrainingStrategy` hierarchy with a new, natively-written `NeuralSurrogateTrainingStrategy` built on plain `torch.nn` (a small MLP or similar), trained on snapshots generated by `fea_engine` itself — no TensorMesh dataset-generation code involved. TensorMesh's `tensormesh.dataset`/`tensormesh.nn` are referenced only for the *shape* of a clean parametric-dataset/train-loop split (how they separate snapshot generation, batching, and training into distinct stages), which is a reasonable pattern to replicate independently in our own module regardless of TensorMesh's existence.

### 6.3 Own PyTorch-autograd tangent cross-check utility (targets: numerical-tangent gap in `Tet4NeoHookean`/`Tet10SolidTL`)

Write a small, self-contained utility — e.g. `fea_engine/materials/autograd_tangent.py` — that re-expresses an existing NumPy stress-update function's math in `torch` ops and calls `torch.autograd.grad` to obtain an exact tangent, used purely to validate the analytic and complex-step tangents already in the codebase against a third, independent method. This idea comes from observing that TensorMesh (and JAX-FEM) get their consistent tangents this way rather than by hand-derivation — it borrows the *technique* (autodiff of a scalar/vector-valued stress-update map), not any TensorMesh code, and depends on nothing but `torch`.

### 6.4 Own PyTorch (or NumPy) mixed-field assembly for the locking gap (targets: no volumetric-locking cure; rejected mixed shell formulation)

Design and implement, natively, a mixed u-p or assumed-strain assembly for the 3D solids' volumetric-locking gap, and/or take a second, better-informed attempt at the Hellinger-Reissner-style mixed shell formulation that was previously abandoned after Newton-divergence failures on `Shell4MITCCorotational`. TensorMesh's `MixedElementAssembler` documentation (its LBB-stability logic — declaring field interpolation order independently of mesh order) is studied only to understand *why* a mixed formulation stays well-posed, then reimplemented from first principles in our own element/assembly code (this doesn't strictly need `torch` — NumPy is sufficient unless autograd-derived tangents from 6.3 are reused here too).

### 6.5 Prioritized order

1. **§6.1, own PyTorch sparse-solve module** — smallest, self-contained, addresses a real capability gap (no GPU solve path today) without touching any existing element/assembly code.
2. **§6.3, own PyTorch-autograd tangent cross-check** — small, isolated utility; validates existing code rather than adding new physics, low risk.
3. **§6.2, own PyTorch neural-surrogate training strategy** — moderate effort, additive to `rom_engine`'s existing `TrainingStrategy` hierarchy rather than replacing anything.
4. **§6.4, own mixed-field assembly for locking** — highest effort and highest risk (a prior related attempt failed), but the only item that fixes a documented gap at its root rather than working around it.

---

## 7. Caveats

- TensorMesh's own comparison table self-rates its maturity as ⚠️ — the same tier it assigns to JAX-FEM and torch-fem, below FEniCS and scikit-fem.
- Several citation and dating details (arXiv IDs, "2026" publication year, newly-released benchmark hardware) suggest the material may be very recent, forward-dated, or pre-print in nature. Benchmark numbers should be treated as self-reported until independently reproduced (e.g., by running `camlab-ethz/tensormesh-bench` directly) or corroborated by a third-party citation.
- No TensorMesh documentation page found in this analysis describes an arc-length, buckling, or bifurcation-tracking solver, nor a volumetric-locking cure, nor a general two-body contact search — these are treated as absent based on the documentation and example gallery, not a source-code audit (unlike the `fea_engine` findings, which were verified directly against the codebase in a prior session).

---

## Sources

- [TensorMesh homepage](https://www.tensor-mesh.com/)
- [TensorMesh documentation](https://docs.tensor-mesh.com/)
- [TensorMesh quickstart](https://docs.tensor-mesh.com/getting_started/quickstart.html)
- [TensorMesh example gallery](https://docs.tensor-mesh.com/example_gallery/)
- [TensorMesh benchmarks](https://docs.tensor-mesh.com/performance/benchmarks.html)
- [TensorMesh GitHub repository](https://github.com/camlab-ethz/TensorMesh)
- Wen, Chi, Yu, Moseley, Michelis, Ren, Sun & Mishra, "Learning, Solving and Optimizing PDEs with TensorGalerkin: an Efficient High-Performance Galerkin Assembly Algorithm," arXiv:2602.05052, 2026.
- Chi & Wen, "torch-sla: Differentiable Sparse Linear Algebra with Adjoint Solvers and Sparse Tensor Parallelism for PyTorch," arXiv:2601.13994, 2026.

See also: this workspace's own `fea_engine/docs/fem_implementation_lessons.md` and `fea_engine/docs/nonlinear_fem_lessons.md`, whose implementation-status appendices were the basis for the `fea_engine` gap analysis in Sections 5–6 above.
