# API reference: `fea_engine` and `rom_engine`

A single-file map of the public API of both packages: what each module is for, every public class, function and method with its signature and a one-line purpose, and runnable examples. Written for `computation-suite/fea_engine` and `computation-suite/rom_engine` as they stand now.

How this file was made: the inventory in Part 3 is generated from the source code (every public, non-underscore class, function and method), so signatures are exact. The one-line purposes are the first sentence of each docstring; the module descriptions and the task table are hand-written. Items marked ★ can also be imported straight from the package (`from fea_engine import X`). The examples in Part 2 were all executed, and the output shown is what they printed. For deeper explanations and many more worked examples, see `USER_GUIDE.md`.

Size: `fea_engine` has 61 public classes, 158 public functions and 227 public methods; `rom_engine` has 36 public classes, 61 public functions and 124 public methods.

Contents: Part 1 task map. Part 2 examples. Part 3 full inventory (`fea_engine`, then `rom_engine`). Part 4 conventions and gotchas.

---

# Part 1 — Which API do I need?

| I want to... | Use | Package.module |
|---|---|---|
| Describe a material | `Material`, `D_plane_stress`, `D_plane_strain`, `EI_beam`, `Section` | `fea_engine.material` |
| Make a mesh (line, rectangle, box, with hole, ...) | `rectangle_mesh`, `generate_mesh`, `Mesh`, `MultiBlockMesh` | `fea_engine.mesh`, `fea_engine.geometry` |
| Build mesh + element + system in one call | `build_system(dim=..., shape=..., physics=...)` | `fea_engine.geometry` |
| Choose how an element behaves | Element classes (`Quad4PlaneStress`, `Hex8Solid3D`, `Beam2DEulerBernoulli`, ...) | `fea_engine.elements` |
| Assemble, apply loads/constraints, solve | `FESystem` (`assemble_stiffness`, `assemble_mass`, `add_nodal_force`, `fix_dofs`, `solve_static`) | `fea_engine.solver` |
| Natural frequencies / mode shapes | `FESystem.solve_modal` | `fea_engine.solver` |
| Buckling load | `FESystem.solve_linear_buckling` | `fea_engine.solver` |
| Frequency response / random vibration | `solve_frequency_sweep`, `solve_random_vibration`, `HarmonicLoad`, `PSDLoad` | `fea_engine.solver`, `fea_engine.loads` |
| Time history (implicit / explicit) | `solve_transient_implicit`, `solve_transient_explicit` | `fea_engine.solver` |
| Large deflection, plasticity, contact, snap-through | `solve_nonlinear_static`, `solve_nonlinear_arc_length`, `solve_nonlinear_displacement_control`, `solve_nonlinear_transient` | `fea_engine.nonlinear_solver` |
| Pressure on an edge or face | facet-load functions | `fea_engine.facet_loads` |
| Refine the mesh automatically | `adaptive_refine_solve` | `fea_engine.adaptivity` |
| Many load cases on one stiffness matrix | `solve_batched`, `solve_static_batched` | `fea_engine.batched_solve` |
| Faster assembly on big meshes | vectorised assembly | `fea_engine.vectorized_assembly` |
| Run on GPU / torch | `FESystem(..., backend="torch", device="cuda")` | `fea_engine.solver`, `fea_engine.torch_sparse_solver` |
| Topology optimisation | SIMP optimiser | `fea_engine.topopt` |
| Reduce a model with POD + Galerkin | `PodBasis`, `GalerkinROM` | `rom_engine.pod`, `rom_engine.galerkin` |
| Fast parameter sweeps | `AffineDecomposition` | `rom_engine.affine` |
| Frequency-domain ROM and certified error bounds | `FrequencyROM`, greedy training, SCM bounds | `rom_engine.frequency`, `.greedy`, `.scm`, `.scm_lp` |
| Systems-and-control MOR | `BalancedTruncationROM`, `KrylovROM`, `SOARROM`, `OptimalHankelNormROM` | `rom_engine.balanced_truncation`, `.krylov`, `.soar`, `.hankel_norm` |
| Identify modes from FRF data only | Loewner identification and screening | `rom_engine.loewner`, `.screening` |
| Nonlinear structural ROM | `IntrusiveNonlinearROM`, RBF/ICE surrogates, NNM backbones | `rom_engine.intrusive_nonlinear_rom`, `.nonlinear_rom`, `.nnm` |
| Choose training parameters | Optimal Latin hypercube sampling | `rom_engine.sampling` |
| Compare full vs reduced results | metrics | `rom_engine.metrics` |

---

# Part 2 — Examples

All of these ran successfully. Paths assume you run from `computation-suite` (the first lines add both `src` folders to `sys.path`; if you installed the packages with pip you can drop that line). The complete runnable script is at the end of this part.

```python
import sys, numpy as np
sys.path[:0] = ["fea_engine/src", "rom_engine/src"]
```

### E1 — FEA: static plane-stress cantilever

```python
from fea_engine import Material, D_plane_stress, Quad4PlaneStress, FESystem
from fea_engine.mesh import rectangle_mesh
mat = Material(E=2.1e11, nu=0.3, rho=7850.0)
mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=24, ny=12)
sys_ = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
sys_.assemble_stiffness(D_plane_stress(mat), thickness=0.02)
tip = mesh.nodes_on_line(axis=0, value=0.4)
sys_.add_nodal_force(tip, dof_index=1, total_force=-20000.0)
sys_.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
U = sys_.solve_static()
print("nodes", len(mesh.nodes), "dofs", sys_.n_dof, "mean tip U_y = %.4e m" % U[2 * tip + 1].mean())
```

Output:

```text
nodes 325 dofs 650 mean tip U_y = -1.7969e-04 m
```

### E2 — FEA: one-call model with geometry.build_system

```python
from fea_engine.geometry import build_system
s2, m2, e2 = build_system(dim=2, Lx=0.4, Ly=0.2, nx=24, ny=12)
print(type(e2).__name__, "nodes", len(m2.nodes))
```

Output:

```text
Quad4PlaneStress nodes 325
```

### E3 — FEA: natural frequencies (Euler-Bernoulli cantilever)

```python
from fea_engine.geometry import generate_mesh
from fea_engine.material import Section, EI_beam
from fea_engine.elements import Beam2DEulerBernoulli
m = Material(E=210e9, nu=0.3, rho=7800.0)
beam = FESystem(generate_mesh(dim=1, L=1.0, n=40), Beam2DEulerBernoulli())
beam.assemble_stiffness(EI_beam(m, Section(A=0.01, I=8.33e-6)))
beam.assemble_mass(m.rho * 0.01)
beam.fix_dofs([0], [0, 1])
f, shapes = beam.solve_modal(n_modes=3)
EI = 210e9 * 8.33e-6; rhoA = 7800 * 0.01
ref = [(1.8751**2) / (2 * np.pi) * np.sqrt(EI / (rhoA * 1.0**4)), (4.6941**2) / (2 * np.pi) * np.sqrt(EI / (rhoA * 1.0**4))]
print("freq_hz =", np.round(f, 2), "| analytic modes 1-2 =", np.round(ref, 2))
```

Output:

```text
freq_hz = [  83.8   525.18 1470.52] | analytic modes 1-2 = [ 83.8  525.18]
```

### E4 — FEA: geometrically nonlinear cantilever (Reissner beam), load ramp

```python
from fea_engine.mesh import Mesh
from fea_engine import elements
from fea_engine.nonlinear_solver import solve_nonlinear_static
E, nu, A, I, L, n = 210e9, 0.3, 1e-3, 8.33e-7, 1.0, 8
G = E / (2 * (1 + nu)); x = np.linspace(0, L, n + 1).reshape(-1, 1)
nb = FESystem(Mesh(nodes=np.hstack([x, 0 * x]), elements=np.array([[i, i + 1] for i in range(n)]), dim=1), elements.Beam2DReissner())
nb.fix_dofs([0], [0, 1, 2]); matb = (E, G, A, I, 1.0); nb.assemble_stiffness(matb)
P = 0.8 * 3 * E * I / L**2          # tip load giving a large deflection
nb.F[:] = 0.0; nb.F[3 * n + 1] = P
lf, Uh = solve_nonlinear_static(nb, matb, n_steps=20)
lin = P * L**3 / (3 * E * I)
print("linear tip deflection %.4f m, nonlinear %.4f m (load factors %d)" % (lin, Uh[-1][3 * n + 1], len(lf)))
```

Output:

```text
linear tip deflection 0.8000 m, nonlinear 0.5472 m (load factors 21)
```

### E5 — ROM: POD + Galerkin on the E1 model (8 snapshots, 6 modes)
Continues from E1 (reuses `sys_`, `mat`, `tip`).

```python
from rom_engine import PodBasis, GalerkinROM
sys_.assemble_mass(mat.rho * np.eye(2))
rng = np.random.default_rng(0); snaps = np.zeros((sys_.n_dof, 8))
for i in range(8):
    sys_.F[:] = 0.0
    sys_.add_nodal_force(tip, dof_index=1, total_force=-20000.0 * rng.uniform(0.5, 1.5))
    snaps[:, i] = sys_.solve_static()
basis = PodBasis().fit(snaps, n_modes=6, M=sys_.M)
rom = GalerkinROM(basis).reduce_system(sys_.K, M=sys_.M, F=sys_.F)
x_full, q = rom.solve_static()
print("modes", basis.V.shape[1], "energy captured %.6f" % basis.energy_captured(), "reduced size", len(q), "full size", sys_.n_dof)
print("ROM vs full-order solve (same load): max abs diff = %.2e" % np.abs(x_full - sys_.solve_static()).max())
fz, _, _ = rom.solve_modal(); print("first ROM frequencies (Hz):", np.round(np.sort(np.real(fz))[:3], 1))
```

Output:

```text
modes 6 energy captured 1.000000 reduced size 6 full size 650
ROM vs full-order solve (same load): max abs diff = 6.91e-06
first ROM frequencies (Hz): [  129.1  6524.1 11039.1]
```

### E6 — ROM: affine decomposition, K(mu) = mu1*K1 + mu2*K2 (two-region beam)

```python
from fea_engine.mesh import MultiBlockMesh
from rom_engine import AffineDecomposition
bm = generate_mesh(dim=1, L=1.0, n=60)
mb = MultiBlockMesh(nodes=bm.nodes, blocks={"r1": bm.elements[:30], "r2": bm.elements[30:]}, dim=bm.dim)
emap = {"r1": Beam2DEulerBernoulli(), "r2": Beam2DEulerBernoulli()}
def assemble(a, b):
    s = FESystem(mb, emap); s.assemble_stiffness({"r1": a, "r2": b}); s.fix_dofs([0], [0, 1]); return s
K1, K2 = assemble(1.0, 0.0).K, assemble(0.0, 1.0).K
free = np.asarray(assemble(1, 1).free_dofs); ix = np.ix_(free, free)
Fv = np.zeros(assemble(1, 1).n_dof); Fv[-2] = 1.0; Ff = Fv[free]
def full(a, b):
    return np.linalg.solve(a * np.array(K1)[ix] + b * np.array(K2)[ix], Ff)
tr = [(1, 1), (2, 1), (1, 2), (0.5, 1), (1, 0.5), (3, 2), (0.7, 2.5), (2.5, 0.8)]
Sn = np.column_stack([full(a, b) for a, b in tr])
V = PodBasis().fit(Sn, n_modes=6).V
aff = AffineDecomposition([np.array(K1)[ix], np.array(K2)[ix]], theta_func=lambda mu: [mu[0], mu[1]]).project(V)
mu = (1.7, 1.3)
Kr = aff.assemble_reduced(mu); xr = V @ np.linalg.solve(Kr, V.T @ Ff); xf = full(*mu)
print("reduced size", Kr.shape[0], "full size", len(Ff), "| rel. error at unseen mu = %.2e" % (np.linalg.norm(xr - xf) / np.linalg.norm(xf)))
```

Output:

```text
reduced size 6 full size 120 | rel. error at unseen mu = 2.00e-09
```

### E7 — ROM: frequency response of a modal ROM
Continues from E3 (reuses `beam`).

```python
from scipy.linalg import eigh
from rom_engine import FrequencyROM
Kb, Mb = np.array(beam.K)[np.ix_(beam.free_dofs, beam.free_dofs)], np.array(beam.M)[np.ix_(beam.free_dofs, beam.free_dofs)]
ev, Vm = eigh(Kb, Mb); wn = np.sqrt(ev)
Fb = np.zeros(beam.n_dof); Fb[-2] = 1000.0; Fbf = Fb[np.asarray(beam.free_dofs)]
fr = FrequencyROM.from_MCK(Mb, Kb, Vm[:, :6], rayleigh=(2.0, 1e-5))
om = np.linspace(0.2, 1.5, 40) * wn[0]
H = fr.frequency_response(om, Fbf, output_dofs=[len(Fbf) - 2])[:, 0]
Cm = 2.0 * Mb + 1e-5 * Kb
Hf = np.array([np.linalg.solve(-w**2 * Mb + 1j * w * Cm + Kb, Fbf)[len(Fbf) - 2] for w in om])
print("6-mode FRF vs full-order sweep: rel. L2 error = %.2e" % (np.linalg.norm(H - Hf) / np.linalg.norm(Hf)))
```

Output:

```text
6-mode FRF vs full-order sweep: rel. L2 error = 1.08e-05
```

### E8 — ROM: balanced truncation (state-space MOR)
Continues from E7.

```python
from rom_engine import BalancedTruncationROM
Bc = Fbf.reshape(-1, 1) / np.linalg.norm(Fbf); Co = np.zeros((1, len(Fbf))); Co[0, len(Fbf) - 2] = 1.0
# modal pre-truncation first (the guide explains why), then balanced truncation
Vr = Vm[:, :10]; Mr, Kr_, Br, Cor = Vr.T @ Mb @ Vr, Vr.T @ Kb @ Vr, Vr.T @ Bc, Co @ Vr
bt = BalancedTruncationROM.from_MCK(Mr, Kr_, Br, Cor, C=Vr.T @ Cm @ Vr, r=4)
print("stable:", bt.is_stable(), "| H-infinity error bound: %.3e" % bt.h_infinity_error_bound())
```

Output:

```text
stable: True | H-infinity error bound: 1.491e-08
```

Notes on the examples:

- E4 uses a load of 0.8 x 3EI/L^2; the linear solution would give `0.8000 m`, the nonlinear one `0.5472 m` because the beam stiffens geometrically and rotates.
- E5's ROM frequencies are for the plane-stress block with the mass assembled in E5; they are not the beam frequencies from E3.
- Measured errors are for these specific models; they are not guarantees for yours.

---

# Part 3 — Full inventory

Signature notation: `name(arg, arg=default, *, keyword_only)`; methods are written `.method(...)` and called on an instance; `Class.method(...)` marks a class/static method. A star means the name is importable from the top-level package.

## `fea_engine` — finite element analysis

### `adaptivity.py`

A-posteriori error estimation, element marking, conforming local mesh refinement and the solve-estimate-mark-refine loop. Use it when a mesh is too coarse near a singularity (re-entrant corner, crack tip) and you want the mesh to refine itself.

- **`element_stresses(mesh, elem_formulation, D, U)`** ★ — Per-element stress [sigma_xx, sigma_yy, sigma_xy], shape (n_elements, 3).
- **`zz_recovery_estimator(mesh, elem_formulation, D, U)`** ★ — Zienkiewicz & Zhu (1987)-style a posteriori error estimator: build a smoother, nodally-continuous recovered stress field sigma* by area-weighted-averaging the discontinuous per-element sigma_h over every element sharing each node ...
- **`jump_residual_estimator(mesh, elem_formulation, D, U)`** ★ — Cheaper, cruder edge-stress-jump indicator -- fem_implementation_ lessons.md's own point 13 ("a cheaper, cruder error indicator can produce a MORE efficient adaptive mesh than an expensive, more accurate one") is the explicit ...
- **`fixed_fraction_marking(eta, fraction=0.3)`** ★ — Mark the top `fraction` of elements BY COUNT, ranked by eta (descending).
- **`threshold_marking(eta, threshold_rel=0.5)`** ★ — Mark every element whose eta_e exceeds threshold_rel * max(eta).
- **`equidistribution_marking(eta, n_target_elements)`** ★ — Mark elements whose error exceeds the level that would equidistribute the total squared (energy-norm) error evenly across n_target_elements -- the standard adaptive-FEM equidistribution prescription eta_e <= TOL/sqrt(N), applied ...
- **`class RefinementRecord(parent_element, level)`** ★ — parent/child tracking across one refinement step. parent_element[i] is the ORIGINAL (pre-refinement) element id new element i descended from; level[i] is its refinement depth relative to whatever `parent_level` was passed into ...
- **`refine_triangle_mesh_longest_edge(mesh, marked, parent_level=None)`** ★ — Rivara (1984) longest-edge conforming bisection: refines every element in `marked` (boolean array, len == n_elements) PLUS whatever additional elements the conformity-preserving propagation step requires -- when triangle T's ...
- **`adaptive_refine_solve(mesh, elem_formulation, D, setup_fn, thickness=1.0, estimator='zz', marking='fixed_fraction', marking_param=0.3, max_refinements=5, tol=None, ...)`** ★ — The capstone loop: solve -> estimate -> mark -> refine -> re-solve.

### `autograd_tangent.py`

PyTorch-autograd tangent stiffnesses used to cross-check (or replace) hand-derived/finite-difference tangents of nonlinear elements. Use it to verify a new nonlinear element or to get exact tangents. Needs PyTorch.

- **`tet4_neo_hookean_tangent_autograd(elem_coords, u_elem, mat)`** — Independent torch.autograd.grad tangent stiffness for Tet4NeoHookean, cross-checking Tet4NeoHookean.tangent_stiffness()'s existing central-finite-difference result (see that method's docstring for why FD was chosen there over a ...
- **`tet10_solid_tl_tangent_autograd(elem_coords, u_elem, mat, formulation=None)`** — Independent torch.autograd.grad tangent stiffness for Tet10SolidTL, cross-checking its ANALYTIC tangent_stiffness() (added 2026-09-02, see that method's own docstring for the derivation this validates) as well as the KEPT ...
- **`tet4_neo_hookean_tangent_autograd_batched(elem_coords_batch, u_elem_batch, mat)`** — Batched counterpart to tet4_neo_hookean_tangent_autograd() above.
- **`shell4_mitc_corotational_committed_tangent_autograd(elem_coords, u_elem, D, state, iter_state=None, formulation=None)`** — Independent torch.autograd.grad tangent stiffness for Shell4MITCCorotational's state=...
- **`beam2d_reissner_tangent_autograd(elem_coords, u_elem, mat, device='cpu')`** — Independent torch.autograd.grad tangent stiffness for Beam2DReissner, cross-checking its ANALYTIC tangent_stiffness().

### `backend_dispatch.py`

Chooses SciPy or torch for a given stiffness matrix (backend="auto") and prints the one-line solve diagnostic behind `solve_static(verbose=True)`.

- **`select_backend(Kff, device='cpu', dof_threshold=AUTO_DOF_THRESHOLD)`** — Decide "scipy" or "torch" for this specific `Kff`, and return a diagnostics dict describing why -- the actual content of the `verbose=True` one-liner `format_diagnostic_line()` prints.
- **`format_diagnostic_line(diag)`** — The `verbose=True` one-line summary (`FESystem.solve_static(verbose=True)`), mirroring TensorMesh's own `[torch-sla] solve: n=..., nnz=..., dtype=..., device=..., symmetric=..., spd=..., backend=..., method=...` convention.

### `batched_solve.py`

Factorise a stiffness matrix once and solve for many load vectors. Use it for load-case sweeps and ROM snapshot generation. (Some functions use PyTorch, optional.)

- **`solve_batched(K, B, backend='scipy', device='cpu')`** — Solve `K @ X[:, i] = B[:, i]` for every column `i` of `B`, factorizing `K` exactly ONCE regardless of how many columns `B` has.
- **`solve_static_batched(fesystem, F_batch, backend=None)`** — `FESystem`-level convenience wrapper: solve `fesystem`'s already- assembled `K` against MANY external load vectors at once, applying the SAME fixed-dof boundary conditions (`fesystem.fixed_dofs`/ `free_dofs`) to every column, ...

### `beam2d_reissner_vectorized.py`

Vectorised (no Python element loop) internal force and tangent for the geometrically exact 2-D Reissner beam. Use it for large beam models in nonlinear solves.

- **`internal_force_batched(elem_coords_all, u_elem_all, mat)`** — Batched Beam2DReissner.internal_force(): elem_coords_all (n,2,2), u_elem_all (n,6), mat=(E,G,A,I,kappa_s) CONSTANT across all n elements (homogeneous material per block, same scope restriction vectorized_assembly.py's own ...
- **`tangent_stiffness_batched(elem_coords_all, u_elem_all, mat)`** — Batched Beam2DReissner.tangent_stiffness() -> K_global (n,6,6).
- **`scatter_global_force(fe, connectivity, dofs_per_node, n_dof)`** — Vectorized (no Python loop over elements) scatter-add of a stacked local-force array (n_elements, n_edof) into a global force vector -- the force-vector counterpart of vectorized_assembly.
- **`assemble_internal_force_vectorized(fesystem, u_global, mat, block_name=None)`** — Additive, opt-in alternative to FESystem.assemble_internal_force() for a Beam2DReissner block -- returns a fresh global force vector (does not mutate fesystem, matching assemble_internal_force()'s own return-only convention).
- **`assemble_tangent_stiffness_vectorized(fesystem, u_global, mat, block_name=None, sparse=False)`** — Additive, opt-in alternative to FESystem.assemble_tangent_ stiffness() for a Beam2DReissner block -- same scope/return/u_global convention as assemble_internal_force_vectorized() above.

### `beam2d_reissner_vectorized_torch.py`

Torch (CPU/GPU) version of the vectorised Reissner-beam force and tangent evaluation. Needs PyTorch.

- **`internal_force_batched_torch(elem_coords_all, u_elem_all, mat, device='cpu', dtype=None)`** — Torch-vectorized, GPU-capable Beam2DReissner.internal_force(), batched across a leading n_elements axis.
- **`tangent_stiffness_batched_torch(elem_coords_all, u_elem_all, mat, device='cpu', dtype=None)`** — Torch-vectorized Beam2DReissner.tangent_stiffness() -- same conventions as internal_force_batched_torch() above.
- **`scatter_global_force_torch(fe, connectivity, dofs_per_node, n_dof, device='cpu')`** — Vectorized (no Python loop over elements) scatter-add of a stacked local-force array (n_elements, n_edof) into a global force vector.
- **`assemble_internal_force_vectorized_torch(fesystem, u_global, mat, block_name=None, device='cpu', dtype=None)`** — Torch-vectorized, GPU-capable counterpart of beam2d_reissner_ vectorized.assemble_internal_force_vectorized() -- same additive, opt-in, single-homogeneous-material-per-block scope and return-only convention (does not mutate ...
- **`assemble_tangent_stiffness_vectorized_torch(fesystem, u_global, mat, block_name=None, device='cpu', dtype=None, sparse=False)`** — Torch-vectorized, GPU-capable counterpart of beam2d_reissner_ vectorized.assemble_tangent_stiffness_vectorized() -- reuses vectorized_assembly.scatter_global_stiffness() VERBATIM.

### `build_mesh.py`

Single entry point that decides from a feature list whether to grade a mesh, picks the generator, and refuses combinations that would give a bad mesh.

- **`class RectangleWithHole(Lx, Ly, hole_center=None, hole_radius=None, n_ring=12)`** ★ — The one geometry this dispatcher currently understands end-to-end on BOTH mesh-generation front ends: a rectangle [0,Lx] x [0,Ly] with at most one circular hole.
  - `.detect_features()`
- **`build_mesh(geometry, target_size, element_formulation, growth_ratio=None, nr=8, n_extend=None)`** ★ — The Phase 4 dispatcher: always calls `geometry.detect_features()` and decides FOR ITSELF whether/how to grade, instead of requiring the caller to notice "this geometry is complex" and reach for a differently-named function.

### `damping.py`

Damping models for dynamics (Rayleigh C = alpha M + beta K, with calibration to a target damping ratio).

- **`class RayleighDamping(alpha: float, beta: float)`** ★ — C = alpha*M + beta*K.
  - `RayleighDamping.calibrate(omega_i, omega_j, zeta)`
  - `.modal_ratio(omega)` — Resulting damping ratio at an arbitrary modal frequency omega (rad/s), for reporting/diagnostics.
- **`class ModalDamping(zeta: float)`** ★ — A damping ratio (or array of ratios, one per retained mode) applied directly to the decoupled modal equations in solver.solve_modal_superposition() -- no global C matrix is ever assembled for this path, since modal superposition ...
- **`class FieldDamping(coefficients: object)`** ★ — a field-wise ("per DOF-type") consistent viscous damping matrix, the damping model Georgiou 2005 actually uses (its Eqs. 1-3, 24, 35 apply ONE scalar D uniformly to every field -- u1, u2, theta -- NOT a Rayleigh alpha*M+beta*K ...
  - `.assemble(fesystem, **kwargs)` — Build and return the global field-damping matrix (same storage format -- dense ndarray or scipy.sparse.lil_matrix -- as fesystem's own K/M) by ...

### `differentiable.py`

Differentiable-solver tools: adjoint (implicit-differentiation) solves and explicit residual-minimisation calibration of correction terms. Needs PyTorch.

- **`class AdditiveCorrection()`** — Duck-typed protocol (not an ABC by inheritance requirement -- matching this package's own established style; see the sibling rom_engine package's `nonlinear_rom.ReducedForceModel` for the same convention) for a trainable additive ...
  - `.value(u_free)`
  - `.jacobian(u_free)`
- **`corrected_residual(fesystem, u, mat, correction, F_ext=None, **kwargs)`** — R_free(u) = F_ext_free - F_int_free(u) - f_theta(u_free) -- the free-standing equivalent of nonlinear_solver.solve_nonlinear_ static()'s own `_residual()` closure, usable outside a Newton loop (calibrate_correction_explicit() ...
- **`corrected_tangent(fesystem, u, mat, correction, **kwargs)`** — K_eff = K_T(u) + d f_theta/du_free -- the free-dof tangent Newton needs to stay consistent with corrected_residual() above: d/du of -f_theta(u) contributes +correction.jacobian(u_free) (two minus signs: R has a -f_theta term, and ...
- **`solve_nonlinear_static_corrected(fesystem, mat, correction=None, n_steps=10, tol=1e-08, max_iter=30, verbose=False, load_factors=None, **kwargs)`** — 's concrete Newton driver: the SAME load-controlled incremental-Newton ladder as nonlinear_solver.solve_nonlinear_ static() (identical load_factors convention, identical commit_all_ states() call after every converged step), ...
- **`calibrate_correction_explicit(fesystem, mat, correction, w_reference, F_ext=None, n_epochs=500, lr=0.01, optimizer_cls=None, verbose=False, **kwargs)`** — the cheap, lower-risk alternative to item 98's implicit adjoint layer: when a full reference state w_m is already available (a finer-mesh solution, a full-field DIC/experimental measurement, or a high-fidelity ROM snapshot -- ...
- **`class ScalarFieldCorrection(n_free, init=0.0, dtype=None)`** (extends `AdditiveCorrection`) — Simplest possible trainable correction: one free trainable scalar per ACTIVE free dof, f_theta(u) = theta (a constant additive term, independent of u -- jacobian() is therefore exactly zero).
  - `.value(u_free)`
  - `.jacobian(u_free)`
  - `.torch_value(u_free_t)`
  - `.parameters()`
- **`class MLPCorrection(n_free, hidden_sizes=(16, 16), feature_fn=None, seed=None, dtype=None)`** (extends `AdditiveCorrection`) — Local, feature-based trainable correction: a small shared- weight torch.nn MLP applied POINTWISE to each free dof's own scalar feature (u_free itself by default, or a caller-supplied feature_fn) -- f_theta(u)_i = MLP(phi(u)_i) -- ...
  - `.torch_value(u_free_t)`
  - `.value(u_free)`
  - `.jacobian(u_free)`
  - `.parameters()`
- **`adjoint_gradient(K_eff, grad_w, vjp_fn)`** — Shared numeric core of the whole-solver implicit-differentiation layer.
- **`implicit_correction_solve(fesystem, mat, correction, F_ext=None, u0=None, tol=1e-08, max_iter=30)`** — Public entry point for item 98: solves the corrected equilibrium F_ext - F_int(w) - f_theta(w) = 0 for w* via the ORDINARY plain- NumPy Newton ladder (_newton_equilibrium(), fully reused from item 99 -- no reimplementation), and ...
- **`mass_cholesky_factor(M_sub)`** — Cholesky factor N of a (symmetric positive definite) mass submatrix M_sub = N @ N.T (Saverio et al. 2026 Appendix C) -- the change-of-variables matrix behind MetricScaledField below.
- **`metric_step_matches_euclidean_reparametrized_step(N, grad_theta, lr=0.1)`** — Decisive, torch-free identity check of MetricScaledField's whole point: ONE ordinary Euclidean gradient-descent step in the reparametrized coordinate theta_tilde = N^-1 @ theta (equivalently theta = N @ theta_tilde) is EXACTLY a ...
- **`class MetricScaledField(n_field, M_sub, init=0.0, dtype=None)`** (extends `AdditiveCorrection`) — Wraps a plain per-dof/per-element scalar-field correction (theta itself, i.e. ScalarFieldCorrection's own shape) so its trainable parameter is theta_tilde = N^-1 @ theta rather than raw theta -- see ...
  - `.value(u_free)`
  - `.jacobian(u_free)`
  - `.torch_value(u_free_t)`
  - `.parameters()`

### `elements/base.py`

Abstract element base class and the contract every element implements (shape functions, B matrix, stiffness, mass, internal force).

- **`gauss_legendre(n)`** — n-point 1-D Gauss-Legendre rule on [-1, 1].
- **`gauss_product(n, dim)`** ★ — Tensor-product Gauss rule for dim in {1, 2, 3}.
- **`tet_quadrature_4pt()`** — 4-point quadrature rule on the natural tetrahedron {r,s,t >= 0, r+s+t <= 1} (Keast/Zienkiewicz-Taylor's standard degree-2-exact rule) -- NOT a tensor-product rule, so it doesn't belong in gauss_product() above (that function ...
- **`tet_quadrature(order)`** — Order-PARAMETERIZED quadrature on the natural tetrahedron {r,s,t >= 0, r+s+t <= 1}, exact for any polynomial of total degree <= order : generalizes tet_quadrature_4pt()'s single fixed (degree-2-exact) rule to a genuine runtime ...
- **`tri_quadrature_3pt()`** — 3-point quadrature rule on the natural triangle {xi,eta >= 0, xi+eta <= 1} -- the standard degree-2-exact rule (same Zienkiewicz-Taylor family as tet_quadrature_4pt() above, one simplex dimension down).
- **`tri_quadrature(order)`** — Order-PARAMETERIZED quadrature on the natural triangle {xi,eta >= 0, xi+eta <= 1}, exact for any polynomial of total degree <= order the 2-D sibling of tet_quadrature() above; see that function's own docstring for the full ...
- **`jacobian(dN_natural, elem_coords)`** — dN_natural: (dim, n_nodes).
- **`jacobian_measure(dN_natural, elem_coords)`** — Like jacobian() above, but returns the length/area/volume SCALE FACTOR appropriate for any embedding, not just the dim==embedding- dimension case jacobian()'s own det(J) assumes: ordinary det(J) when the element's parametric ...
- **`spurious_zero_energy_modes(elem, elem_coords, D, thickness=1.0, tol=1e-08)`** — Detects hourglass (spurious zero-energy) modes introduced by reduced integration, WITHOUT needing to know the theoretical rigid-body-mode count for this element type.
- **`class Element()`** ★
  - `.shape_and_derivs(natural_coords)` — Returns (N, dN_natural): N shape (n_nodes,), dN_natural shape (dim, n_nodes).
  - `.N_matrix(N)` — Generic block-diagonal expansion: scalar shape functions N (n_nodes,) -> (dofs_per_node, n_nodes*dofs_per_node) matrix mapping nodal DOFs to the ...
  - `.B_matrix(natural_coords, elem_coords)` — Physics-specific strain-displacement matrix at a natural coordinate, plus detJ.
  - `.stiffness(elem_coords, D, thickness=1.0, gauss_order=None)` — Generic Gauss-integrated element stiffness: ke = sum_gp Bᵀ D B \|J\| w * thickness.
  - `.full_stiffness(elem_coords, D, thickness=1.0)` — Exact ('full') integration: self.gauss_order points per direction -- exact for this element's stiffness polynomial. The safe default (what ...
  - `.reduced_stiffness(elem_coords, D, thickness=1.0)` — Uniform reduced integration: one Gauss point fewer per direction than full (minimum 1).
  - `.hourglass_stabilized_stiffness(elem_coords, D, thickness=1.0, c_hg=0.1, tol=1e-06)` — adds a Flanagan-Belytschko/Belytschko-Bindeman-style PERTURBATION STIFFNESS to reduced_stiffness(), so reduced integration becomes safe to actually ...
  - `.internal_force(elem_coords, u_elem, D, thickness=1.0, **kwargs)` — Linear default: f_int = ke @ u_elem, using the SAME ke as stiffness().
  - `.tangent_stiffness(elem_coords, u_elem, D, thickness=1.0, **kwargs)` — Linear default: K_T = ke, independent of u_elem (a linear element's tangent IS its stiffness, at every displacement).
  - `.geometric_stiffness(elem_coords, N, thickness=1.0, **kwargs)`
  - `.mass(elem_coords, rho_matrix, thickness=1.0)` — Generic consistent mass matrix: me = sum_gp Nmᵀ rho_matrix Nm \|J\| w * thickness, where rho_matrix (dofs_per_node x dofs_per_node) lets rotational ...
  - `.lumped_mass(elem_coords, rho_matrix, thickness=1.0)` — HRZ (Hinton-Rock-Zienkiewicz) diagonal scaling: take the DIAGONAL of the consistent mass matrix (always positive, since a consistent mass matrix is ...

### `elements/beams.py`

2-D beam elements: Euler-Bernoulli, Timoshenko, and the geometrically exact Reissner beam for large rotations.

- **`class Beam2DEulerBernoulli()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)` — Linear interpolation of x itself -- used only by mesh.check_quality() for a length-positivity sanity check, NOT by stiffness()/mass() (those use the ...
  - `.stiffness(elem_coords, EI, thickness=1.0)`
  - `.full_stiffness(elem_coords, EI, thickness=1.0)` — N/A for this element -- EI is constant per element, so the closed-form Hermite matrix above IS the exact integral; there is no quadrature order to ...
  - `.reduced_stiffness(elem_coords, EI, thickness=1.0)` — Same result as full_stiffness() -- see its docstring.
  - `.mass(elem_coords, rho_A, thickness=1.0)`
  - `.geometric_stiffness(elem_coords, N, thickness=1.0)` — the standard consistent geometric ("stress stiffness") matrix for a 2-node Euler-Bernoulli beam-column under a constant axial force N ...
- **`class Beam2DCorotational()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)` — Linear interpolation along the (reference) bar axis -- interface completeness only (mesh.check_quality()), same as TrussTL2D; ...
  - `.mass(elem_coords, rho_A, thickness=1.0)` — Consistent mass matrix in the REFERENCE (undeformed) configuration -- standard linear elastodynamics mass, like every other element in this package ...
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — mat = (E, A, I).
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — K_T = B^T @ k_local @ B + K_geo -- see class docstring.
  - `.stiffness(elem_coords, mat, thickness=1.0, **kwargs)` — Initial (zero-displacement) tangent -- interface completeness, matching TrussTL2D.stiffness()/TrussPlastic2D.stiffness().
  - `Beam2DCorotational.recover_stress(elem_coords, u_elem, mat, y_fiber, xi=0.0)` — fiber-level axial stress at a deformed state -- `internal_force()` returns nodal forces only; this is the missing "traditional finite element-based ...
- **`class Beam2DReissner()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)` — Linear interpolation along the reference axis for ALL THREE fields (u1, u2, theta) -- used directly by mass()'s generic Gauss loop (unlike ...
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — f_int = T @ (L0 * B^T @ [N,Q,M]) -- see class docstring for the strain measures, the reduced/exact-quadrature reasoning (both collapse to one ...
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — K_T = T @ K_local @ T.T, K_local = L0*(B^T@D@B + N*H_eps + Q*H_gam) -- see class docstring for the full derivation.
  - `.stiffness(elem_coords, mat, thickness=1.0, **kwargs)` — Zero-state (linearized-about-undeformed) tangent -- same interface-completeness convention as TrussTL2D.stiffness()/ Beam2DCorotational.stiffness().
  - `.mass(elem_coords, rho, thickness=1.0)` — rho = (rhoA, rhoI) -- see class docstring for why the generic Gauss-loop mass (Element.mass()) is exact here with no rotation transform needed, ...

### `elements/beams3d.py`

3-D beam / frame elements.

- **`class Beam3DEulerBernoulli(ref_up=None)`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)` — Linear interpolation along the LOCAL chord -- interface completeness only (mesh quality checks), same role as Beam2DEulerBernoulli's; ...
  - `.stiffness(elem_coords, rigidities, thickness=1.0, **kwargs)` — rigidities = (EA, GJ, EIy, EIz), as returned by material.beam3d_rigidities(mat, sec).
  - `.full_stiffness(elem_coords, rigidities, thickness=1.0)` — N/A -- closed-form, no quadrature order to vary (same reasoning as Beam2DEulerBernoulli).
  - `.reduced_stiffness(elem_coords, rigidities, thickness=1.0)`
  - `.geometric_stiffness(elem_coords, N, thickness=1.0)` — reuses Beam2DEulerBernoulli's own geometric_stiffness() for BOTH bending planes -- exactly the same reuse pattern _local_ matrices() above already ...
  - `.mass(elem_coords, mass_props, thickness=1.0)` — mass_props = (rho*A, rho*Ip), as returned by material.beam3d_mass_props(mat, sec) -- Ip = Iy + Iz, the cross-section's polar AREA moment (mass ...
  - `.lumped_mass(elem_coords, mass_props, thickness=1.0)` — HRZ diagonal scaling, computed directly from THIS element's own consistent mass (not the generic base-class version, which assumes a single scalar ...
- **`class Beam3DCorotational(ref_up=None)`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)` — Linear interpolation along the reference chord -- interface completeness only, same role as Beam2DCorotational's/ Beam3DEulerBernoulli's; ...
  - `.internal_force(elem_coords, u_elem, rigidities, thickness=1.0, **kwargs)` — f_int = B^T @ generalized_forces -- the same virtual-work transformation Beam2DCorotational's own internal_force() uses (B^T @ [N, M1, M2]), ...
  - `.tangent_stiffness(elem_coords, u_elem, rigidities, thickness=1.0, h=1e-06, **kwargs)` — Real central-FD differentiation of internal_force() -- see class docstring for why this Phase A implementation uses FD rather than an analytic ...
  - `.stiffness(elem_coords, rigidities, thickness=1.0, **kwargs)` — Initial (zero-displacement) tangent -- interface completeness, matching Beam2DCorotational.stiffness().
  - `.mass(elem_coords, mass_props, thickness=1.0)` — Reference-configuration consistent mass, delegated UNMODIFIED to Beam3DEulerBernoulli.mass() -- same convention every other nonlinear element in this ...

### `elements/contact.py`

Contact elements (penalty and Lagrange-multiplier formulations).

- **`class GapContactPenalty()`** ★ (extends `Element`)
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — Unilateral spring: zero force while delta<=0 (gap open or just touching), f = k_p*delta*n_hat once delta>0 -- the gradient of the one-sided potential ...
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — K_T = k_p * n_hat (x) n_hat while active, else the zero matrix -- literally 'zero stiffness apart, large stiffness on contact,' the status switch the ...
  - `.stiffness(elem_coords, mat, thickness=1.0, **kwargs)` — Initial (zero-displacement) tangent -- interface completeness, matching TrussTL2D.stiffness()/TrussPlastic2D.stiffness().
- **`class GapContactCurvedFriction()`** ★ (extends `Element`)
  - `GapContactCurvedFriction.init_state()` — No prior tangential 'stick' anchor -- the first contact establishes it fresh (zero initial tangential stretch).
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, state=None)`
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, state=None)` — Central-difference tangent of internal_force() -- see the class docstring for why this is computed numerically here.
  - `.commit_state(elem_coords, u_elem, mat, state, thickness=1.0)` — Called once per CONVERGED load step. Resets the stick anchor to None on separation, so a future re-contact starts fresh rather than remembering a ...
  - `.stiffness(elem_coords, mat, thickness=1.0, **kwargs)` — Initial (zero-displacement) tangent -- interface completeness, matching every other nonlinear element's stiffness().
- **`closest_point_on_segment_2d(p, a, b)`** ★ — Closest-point projection of point p onto the line segment a->b (all length-2 array-likes).
- **`find_contact_pairs_2d(slave_node_ids, master_segments, node_coords, search_radius)`** ★ — Broad-phase candidate-pair search.
- **`class NodeToSegmentContact2D()`** ★ (extends `Element`) — frictionless penalty contact between a single SLAVE node and a MASTER SEGMENT made of two ordinary mesh nodes -- the "obstacle" is no longer a fixed plane/circle (GapContactPenalty/GapContactCurvedFriction above), it is itself ...
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, **kwargs)`
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — Central-difference tangent of internal_force() -- see the class docstring for why this is computed numerically here.
  - `.stiffness(elem_coords, mat, thickness=1.0, **kwargs)` — Initial (zero-displacement) tangent -- interface completeness, matching every other nonlinear/contact element's stiffness().
- **`class NodeToSegmentContact2DFriction()`** ★ (extends `NodeToSegmentContact2D`) — adds Coulomb stick-slip friction to NodeToSegmentContact2D -- the direct node-to-segment generalization of GapContactCurvedFriction's stick/slip return map above, extended from "arc-length position along a FIXED circle" to ...
  - `NodeToSegmentContact2DFriction.init_state()`
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs)`
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs)` — Central-difference tangent -- see the class docstring.
  - `.commit_state(elem_coords, u_elem, mat, state, thickness=1.0, **kwargs)` — Called once per CONVERGED load step. Resets the stick anchor to None on separation, matching GapContactCurvedFriction's commit_state() exactly.
  - `.stiffness(elem_coords, mat, thickness=1.0, **kwargs)` — Initial (zero-displacement) tangent -- interface completeness, matching every other nonlinear/contact element's stiffness().

### `elements/nonlinear_solids.py`

Nonlinear continuum elements: J2 plasticity (hexahedron), Neo-Hookean tetrahedron, and a total-Lagrangian quadratic tetrahedron.

- **`class Hex8PlasticJ2()`** ★ (extends `Hex8Solid3D`) — Hex8Solid3D + small-strain J2 (von Mises) plasticity with linear isotropic hardening (material.PlasticMaterialJ2), via the closed- form radial-return algorithm in material.j2_radial_return_3d() -- see that function's docstring ...
  - `.init_state()`
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs)`
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs)`
  - `.commit_state(elem_coords, u_elem, mat, state, thickness=1.0, **kwargs)` — Replays the return map at every Gauss point (same pattern as TrussPlastic2D.commit_state()) and permanently advances eps_p/alpha -- call once per ...
- **`class Quad4PlasticJ2PlaneStress()`** ★ (extends `Quad4PlaneStress`) — Quad4PlaneStress + small-strain J2 (von Mises) plasticity, via material.j2_radial_return_plane_stress() -- the plane-stress- specific local-Newton return map (Simo & Taylor 1986) that closes the gap j2_radial_return_3d()'s own ...
  - `.init_state()`
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs)`
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs)`
  - `.commit_state(elem_coords, u_elem, mat, state, thickness=1.0, **kwargs)` — Replays the return map at every Gauss point (same pattern as Hex8PlasticJ2.commit_state()) and permanently advances eps_p/alpha/eps33 -- call once ...
- **`class Hex8PlasticJ2Kinematic()`** ★ (extends `Hex8PlasticJ2`) — Hex8PlasticJ2 + Armstrong-Frederick nonlinear kinematic hardening (material.PlasticMaterialJ2Kinematic / material. j2_radial_return_3d_kinematic()) -- a SUBCLASS of Hex8PlasticJ2, not a replacement, so isotropic-only plasticity ...
  - `.init_state()`
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs)`
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs)`
  - `.commit_state(elem_coords, u_elem, mat, state, thickness=1.0, **kwargs)` — Replays the return map at every Gauss point (same pattern as Hex8PlasticJ2.commit_state()) and permanently advances eps_p/alpha/beta -- call once per ...
- **`class Tet4NeoHookean()`** ★ (extends `Tet4Solid3D`) — Tet4Solid3D + compressible Neo-Hookean hyperelasticity (material.NeoHookeanMaterial), Total-Lagrangian formulation, via material.neo_hookean_pk2_stress().
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — f_int_a = V0 * F @ S @ (dN_a/dX) for each node a, from virtual work delta_W_int = Integral_V0 S:delta_E dV0 with delta_E_IJ (from varying node a, dof ...
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, h=1e-06, method='fd', **kwargs)` — method="fd" (default, UNCHANGED behavior): the central finite difference described in this class's own docstring above.
  - `.stiffness(elem_coords, mat, thickness=1.0)` — Initial (zero-displacement) tangent -- interface completeness only, matching TrussTL2D's stiffness().
- **`class Tet10SolidTL()`** ★ (extends `Tet10Solid3D`) — Tet10Solid3D + Total-Lagrangian geometric nonlinearity, LINEAR elastic material (material.D_solid3d) applied to the Green-Lagrange strain -- see this module's own comment block above for the full derivation and the deliberate ...
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — f_int = sum_gp w_gp * (V0 weight) * (F @ S @ dN/dX), the same per-Gauss-point virtual-work expression Tet4NeoHookean.
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — Analytic Total-Lagrangian/St.
  - `.stiffness(elem_coords, mat, thickness=1.0)` — Initial (zero-displacement) tangent -- interface completeness only, matching Tet4NeoHookean/TrussTL2D's own stiffness().

### `elements/plates.py`

Plate-bending elements (Mindlin).

- **`class Quad4MindlinPlate()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)`
  - `.stiffness(elem_coords, D, thickness=1.0, integration='sri')` — D here is the (Db, Ds) pair from config.D_mindlin_plate.
  - `.full_stiffness(elem_coords, D, thickness=1.0)`
  - `.reduced_stiffness(elem_coords, D, thickness=1.0)`
  - `.mass(elem_coords, rho_matrix, thickness=1.0)` — rho_matrix = diag([rho*h, rho*h^3/12, rho*h^3/12]) for translational + rotary inertia -- full integration, no locking issue for the mass matrix.

### `elements/shells.py`

Shell elements with membrane-bending coupling (MITC family, corotational variants).

- **`class Shell4MITC(drilling_factor=0.001)`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)` — Same bilinear Quad4 shape functions as Quad4PlaneStress / Quad4MindlinPlate -- delegated, not re-derived, since all three elements share the ...
  - `.stiffness(elem_coords, D, thickness=1.0, **kwargs)` — D = (Dm, Db, Ds, h), as returned by material.D_shell().
  - `.full_stiffness(elem_coords, D, thickness=1.0)` — N/A -- MITC4's anti-locking mechanism is the assumed shear strain FIELD, not a choice of quadrature order (unlike Quad4MindlinPlate's 'sri'); both ...
  - `.reduced_stiffness(elem_coords, D, thickness=1.0)`
  - `.mass(elem_coords, rho_matrix, thickness=1.0)` — rho_matrix = material.shell_rho_matrix(mat, h) (6x6 diag, see its docstring).
  - `.geometric_stiffness(elem_coords, N, thickness=1.0, **kwargs)` — Not implemented -- shell buckling needs a genuinely different (2-D, membrane-stress-dependent) geometric stiffness formulation than the ...
- **`class Shell4MITCCorotational(drilling_factor=0.001)`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)` — Interface completeness only (mesh.check_quality()), same as every other nonlinear element's override of this method -- ...
  - `.init_state()` — building block C: the "nothing committed yet" baseline -- zero accumulated add-on stress, committed rotation equal to the u=0 reference.
  - `.commit_state(elem_coords, u_elem, mat, state, thickness=1.0, **kwargs)` — building block C: advance the committed baseline to u_elem -- call once per CONVERGED load step, never mid-Newton-iteration (same contract as ...
  - `.init_iter_state()` — one (Nxx, Nyy, Nxy) add-on membrane-stress resultant per Gauss point, initialized to zero -- the mixed- formulation internal unknown this element's ...
  - `.update_iter_state(elem_coords, u_elem, delta_u_elem, D, iter_state, **kwargs)` — advance each Gauss point's N_add by ONLY its own share of a joint-linearized Newton step, using the REALIZED correction delta_u_elem this Newton ...
  - `.internal_force(elem_coords, u_elem, D, thickness=1.0, iter_state=None, state=None, **kwargs)` — f_int = T(u_elem).T @ (K_local0 @ dof_local(u_elem) + _bending_membrane_coupling_force(...)) -- see this module's class-level comment block (steps ...
  - `.tangent_stiffness(elem_coords, u_elem, D, thickness=1.0, iter_state=None, state=None, method='complex_step', **kwargs)` — Analytic tangent (added as Phase B of docs/shells.md Section 4.2, replacing complex- step differentiation as the default -- see ...
  - `.stiffness(elem_coords, D, thickness=1.0, **kwargs)` — Initial (zero-displacement) tangent -- interface completeness, matching Tet10SolidTL.stiffness()/Beam2DCorotational.stiffness().
  - `.mass(elem_coords, rho_matrix, thickness=1.0)` — Consistent mass matrix in the REFERENCE (undeformed) configuration -- delegates directly to Shell4MITC.mass(), same convention as every other ...

### `elements/shells_director.py`

Director-based (finite-rotation) shell element.

- **`director_update(t_ref, theta)`** — t' = exp_map(theta) @ t_ref -- the director rotation update.
- **`drilling_angle_from_tangents(X_ref, x_current)`** — The closed-form, ITERATION-FREE in-plane drilling angle, extracted directly from geometry -- no Newton solve, unlike building block A's general 3-vector `_mean_rigid_ rotation()` (which needs iteration precisely because a 3x3 ...
- **`linearized_director_rotation(theta, t_ref)`** — The FIRST-ORDER (linearized) director rotation, `theta x t_ref` -- the exact linearization, in `theta`, of `director_update(t_ref, theta) = exp_map(theta) @ t_ref` at `theta=0` (a standard, general fact about the exponential map: ...
- **`class Shell4Director(drilling_factor=0.001, curvature='green_lagrange')`** ★ (extends `Element`) — the LINEAR (`u=0`) stiffness of a director-based shell element, independently assembled from director kinematics and validated against `Shell4MITC.stiffness()`.
  - `.shape_and_derivs(natural_coords)`
  - `.reference_directors(elem_coords)` — Returns (t_nodes (4,3), e1_0, e2_0, e3_0, local_coords).
  - `.stiffness(elem_coords, D, thickness=1.0, **kwargs)` — D = (Dm, Db, Ds, h), as returned by `material.D_shell()` -- same convention as `Shell4MITC.stiffness()`.
  - `.mass(elem_coords, rho_matrix, thickness=1.0)` — Delegates to `Shell4MITC.mass()` -- the consistent mass matrix has no director/rotation-update content to differ on at this phase (no ...
  - `.strain_energy(elem_coords, u_elem, D, thickness=1.0)` — Total nonlinear strain energy (a scalar), Gauss-integrated from three resultant strain measures -- `internal_force()`/ `tangent_stiffness()` below ...
  - `.internal_force(elem_coords, u_elem, D, thickness=1.0, **kwargs)` — Analytic gradient of `strain_energy()` -- see class-level "TANGENT STRATEGY" comment for the full derivation record and validation numbers.
  - `.tangent_stiffness(elem_coords, u_elem, D, thickness=1.0, **kwargs)` — Fully analytic material-plus-geometric-stiffness tangent 's own "fully analytic tangent" stretch goal, implemented once `Shell4Director`'s ...

### `elements/solids.py`

2-D and 3-D solid elements: quadrilateral, triangle, hexahedron, tetrahedron (linear and higher order), plane stress/strain and axisymmetric variants, plus plasticity for planar problems.

- **`class Quad4PlaneStress()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)`
  - `.B_matrix(natural_coords, elem_coords)`
- **`class Quad8PlaneStress()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)`
  - `.B_matrix(natural_coords, elem_coords)`
- **`class Hex8Solid3D()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)`
  - `.B_matrix(natural_coords, elem_coords)`
- **`class Hex8SolidBbar()`** ★ (extends `Hex8Solid3D`)
  - `.stiffness(elem_coords, D, thickness=1.0, gauss_order=None)` — Ignores gauss_order (B-bar is only meaningful at the full 8-point rule -- there is no reduced/hourglass variant of this element; use Hex8Solid3D + 's ...
  - `.full_stiffness(elem_coords, D, thickness=1.0)`
  - `.reduced_stiffness(elem_coords, D, thickness=1.0)`
- **`class Hex20Solid3D()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)`
  - `.B_matrix(natural_coords, elem_coords)`
- **`class Tri3PlaneStress()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)`
  - `.B_matrix(natural_coords, elem_coords)` — Constant over the element (linear shape functions), so natural_coords doesn't actually matter -- kept as an argument only so this method has the same ...
  - `.stiffness(elem_coords, D, thickness=1.0, **kwargs)` — ke = Bᵀ D B * Area * thickness -- ONE evaluation (any natural_coords works, B is constant), not a Gauss loop. detJ here is twice the physical area ...
  - `.full_stiffness(elem_coords, D, thickness=1.0)` — N/A for this element -- constant strain, nothing for a quadrature order to refine.
  - `.reduced_stiffness(elem_coords, D, thickness=1.0)`
  - `.mass(elem_coords, rho_matrix, thickness=1.0)` — Classic closed-form CST consistent mass: integral_A N_i N_j dA = (Area/12)*(1 + delta_ij) for LINEAR simplex shape functions -- a standard textbook ...
  - `.lumped_mass(elem_coords, rho_matrix, thickness=1.0)` — Same HRZ idea as the base class (rescale the consistent mass's diagonal to preserve exact total mass), but computed from the closed-form Area above ...
- **`class Tet4Solid3D()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)`
  - `.B_matrix(natural_coords, elem_coords)` — Constant over the element -- natural_coords doesn't matter, kept only for interface consistency (see Tri3PlaneStress).
  - `.stiffness(elem_coords, D, thickness=1.0, **kwargs)` — ke = Bᵀ D B * Volume -- ONE evaluation, not a Gauss loop (see Tri3PlaneStress's docstring for why).
  - `.full_stiffness(elem_coords, D, thickness=1.0)`
  - `.reduced_stiffness(elem_coords, D, thickness=1.0)`
  - `.mass(elem_coords, rho_matrix, thickness=1.0)` — Classic closed-form Tet4 consistent mass: integral_V N_i N_j dV = (Volume/20)*(1 + delta_ij) for linear simplex shape functions -- the 3-D analogue ...
  - `.lumped_mass(elem_coords, rho_matrix, thickness=1.0)` — See Tri3PlaneStress.lumped_mass()'s docstring -- same fix, dofs_per_node directions' worth of mass, not just one.
- **`class Tet10Solid3D()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)`
  - `.B_matrix(natural_coords, elem_coords)`
  - `.stiffness(elem_coords, D, thickness=1.0, quad_order=None, **kwargs)` — ke = sum over the tet quadrature of Bᵀ D B * \|J\| * w * (1/6) -- see tet_quadrature_4pt()'s docstring for why this isn't the generic Gauss loop / ...
  - `.full_stiffness(elem_coords, D, thickness=1.0)` — N/A -- this element has one quadrature scheme (the 4-point rule, already exact for its straight-sided integrand), not a full/reduced distinction.
  - `.reduced_stiffness(elem_coords, D, thickness=1.0)`
  - `.mass(elem_coords, rho_matrix, thickness=1.0, quad_order=None)` — Consistent mass via the SAME quadrature stiffness() uses (N is quadratic, so Nᵀ rho N is quartic -- the default 4-point rule is only exact to degree ...
- **`class Tri6PlaneStress()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)`
  - `.B_matrix(natural_coords, elem_coords)`
  - `.stiffness(elem_coords, D, thickness=1.0, quad_order=None, **kwargs)` — ke = sum over the triangle quadrature of Bᵀ D B * \|detJ\| * w * (1/2) -- see tri_quadrature_3pt()'s docstring for why this isn't the generic Gauss ...
  - `.full_stiffness(elem_coords, D, thickness=1.0)` — N/A -- this element has one quadrature scheme (the 3-point rule, already exact for its straight-sided integrand), not a full/reduced distinction.
  - `.reduced_stiffness(elem_coords, D, thickness=1.0)`
  - `.mass(elem_coords, rho_matrix, thickness=1.0, quad_order=None)` — Consistent mass via the SAME quadrature stiffness() uses (N is quadratic, so Nᵀ rho N is quartic -- with quad_order=None the default 3-point rule is ...

### `elements/trusses.py`

Truss and bar elements (linear and geometrically nonlinear).

- **`class TrussTL2D()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)` — Linear interpolation along the bar axis -- provided only for interface completeness (e.g. mesh.check_quality()); the actual ...
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — mat = (E, A).
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, **kwargs)` — K_T = K_material + K_geometric (consistent linearization of internal_force() above, i.e. d(f_int)/d(u) -- verified against a finite-difference check, ...
  - `.stiffness(elem_coords, mat, thickness=1.0)` — Initial (zero-displacement) tangent -- provided so this element still satisfies the linear-element interface (e.g. an initial K for ...
  - `.geometric_stiffness(elem_coords, N, thickness=1.0)` — this is LITERALLY the K_geometric term already inside tangent_stiffness() above -- (S*A/L0)*I -- pulled out on its own so ...
- **`class TrussPlastic2D()`** ★ (extends `Element`)
  - `.shape_and_derivs(natural_coords)` — Linear interpolation along the bar axis -- interface completeness only, as in TrussTL2D.
  - `TrussPlastic2D.init_state()` — Virgin material: zero plastic strain, zero accumulated plastic strain (so the initial yield stress is exactly mat.sigma_y, with no prior hardening).
  - `.internal_force(elem_coords, u_elem, mat, thickness=1.0, state=None)`
  - `.tangent_stiffness(elem_coords, u_elem, mat, thickness=1.0, state=None)`
  - `.commit_state(elem_coords, u_elem, mat, state, thickness=1.0)` — Called once per CONVERGED load step (see solver.FESystem.commit_all_states() / nonlinear_solver.py) -- replays the same return map and permanently ...
  - `.stiffness(elem_coords, mat, thickness=1.0, **kwargs)` — Initial (virgin, zero-displacement) elastic tangent -- see TrussTL2D.stiffness()'s docstring for why this exists.

### `facet_loads.py`

Pressure and traction loads on element edges/faces by quadrature. Use it for distributed pressure instead of lumping forces onto nodes by hand.

- **`list_facets(formulation)`** — Returns the list of (local_node_indices, family, natural_coord_ map) triples for every facet (edge for a 2-D parent, face for a 3-D solid parent) of `formulation`'s own type -- the lookup a caller uses to turn one mesh element's ...
- **`facet_family(parent_dim, n_facet_nodes)`** — Resolves a facet's own shape-function family from (parent_dim, facet-node count) -- e.g. (2, 3) is unambiguous as line3 (a 2-D parent's quadratic edge) even though 3 nodes ALSO describes tri3 (a 3-D parent's linear face); ...
- **`facet_quadrature(family, order=2)`** — Returns (points, weights) on the facet's own natural domain, weights already scaled to the SAME 'final, ready to multiply directly by the physical measure' convention gauss_product()/ tri_quadrature()'s own callers use (see ...
- **`consistent_facet_load_shares(parent_dim, facet_coords, traction, quad_order=2)`** — Returns an (n_facet_nodes,) array of consistent nodal load 'shares' for a uniform scalar traction/pressure MAGNITUDE over one facet, via real Gauss quadrature restricted to that facet -- the generalization of ...

### `geometry/shapes.py`

Parametric shapes (plates with holes, notches, fillets, L-shapes, ...) producing meshes.

- **`generate_mesh(dim, shape=None, mirror=None, **kwargs)`** — dim: 1, 2, or 3.
- **`default_element(dim, shape=None, physics=None)`** — Returns a freshly constructed default Element instance for this dim (+ optional physics, e.g. 'beam' vs 'truss' at dim=1, or 'plane_stress' vs 'plate' at dim=2 -- see ELEMENT_DEFAULTS).
- **`build_system(dim, shape=None, physics=None, thickness=1.0, mirror=None, **mesh_kwargs)`** — The full pipeline in one call: dimension -> mesh -> element -> FESystem, ready for boundary conditions/loads/assembly/solve (see README sections A-K for what comes next, chosen by which `physics` you asked for).

### `grading.py`

Mesh-grading / element-size functions (graded refinement towards features), shared by the structured and unstructured mesh front ends.

- **`class MeshGradingError()`** ★ (extends `Exception`) — Raised when a requested mesh-grading plan cannot be satisfied (an impossible growth ratio, a not-yet-implemented feature type), or by the Phase 4 dispatcher when an element formulation with rotational bending DOFs is asked to ...
- **`graded_partition(start, end, n, grade=1.0, dense_at='start')`** ★ — Monotonic 1-D partition of n+1 points between start and end, via a power law (grade=1.0 -> uniform; grade>1.0 -> points bunched toward the `dense_at` end, since s**grade rises slowly near s=0 for grade>1).
- **`growth_ratio_of_partition(coords)`** ★ — Max ratio between any two ADJACENT segment lengths in a 1-D partition (`coords` need only be strictly monotonic, not uniform).
- **`grade_for_growth_ratio(n, growth_ratio, dense_at='start', grade_max=50.0, tol=0.001, max_iter=60)`** ★ — Solve, by bisection, for the largest power-law exponent `grade` such that `graded_partition(0, 1, n, grade, dense_at)`'s realized growth ratio does not exceed `growth_ratio`.
- **`class Hole(center: tuple, radius: float, n_ring: int=12, h_min: float=None)`** ★ — A circular hole needing local mesh refinement -- the feature `rectangle_with_hole_mesh_quarter_full` already handles, now expressed declaratively instead of as bare positional parameters.
- **`class Fillet(corner: tuple, radius: float, n_ring: int=8, h_min: float=None)`** ★ — A rounded corner -- geometrically a Hole restricted to the single quadrant at `corner` (a full hole needs 4 quadrants around its center; a fillet needs exactly the 1 quadrant that rounds the corner).
- **`class EdgeBias(axis: int, dense_at: str, h_first: float, growth_ratio: float=DEFAULT_GROWTH_RATIO)`** ★ — Boundary-layer-style grading along one direction of a structured block: fine at one end (`h_first`), growing outward at up to `growth_ratio`.
- **`class Notch(path: list, depth: float, n_ring: int=6, h_min: float=None)`** ★ — A rectangular (right-angle) slot cut inward from a straight boundary edge closing the gap this class's own predecessor left open.
- **`class GradingPlan(features: list, h_far: float, growth_ratio: float, warnings: list=field(default_factory=list))`** ★ — The output of `plan()`: a validated feature list plus the derived growth-ratio-respecting grading exponent for each feature.
  - `.grade_for(feature)` — The power-law exponent for `feature`'s own n_ring/count that respects this plan's growth_ratio cap -- Section 1a's grading law, parameterized by ...
- **`plan(features, h_far, growth_ratio=DEFAULT_GROWTH_RATIO)`** — Combine a feature list into a `GradingPlan`.
- **`threshold_field_size(dist, h_min, h_max, dist_min, dist_max)`** ★ — Gmsh's own Threshold-field definition, reproduced exactly (see the Gmsh reference manual's Field module): size is h_min for dist <= dist_min, h_max for dist >= dist_max, and linearly interpolated in between.
- **`dist_max_for_growth_ratio(h_min, h_far, growth_ratio=DEFAULT_GROWTH_RATIO)`** ★ — Distance band width for a Threshold field that grows from h_min at the feature boundary to h_far, without exceeding `growth_ratio` between successive "layers" of that growth -- the unstructured analogue of ...

### `iterative_solvers.py`

Conjugate gradient and preconditioners (Jacobi, SSOR, Gauss-Seidel style) for large SPD systems, in NumPy and torch forms.

- **`reverse_cuthill_mckee(A)`** ★ — Hand-written Cuthill-McKee ordering (each BFS level visited in ascending-degree order, standard George & Liu 1981 heuristic), reversed at the end -- the "reverse" in RCM, which empirically reduces fill-in further than plain ...
- **`fill_in_count(A, natural=True)`** ★ — nnz of a sparse LU's combined L+U factors (diagonal counted once), via scipy.sparse.linalg.splu() -- the fill-in metric fem_implementation_lessons.md cites (minimum-degree reordering "cuts factor fill by half" in the book's own ...
- **`permuted_solve(A, b, perm=None)`** ★ — Solve A x = b via an explicit symmetric permutation A' = P A P^T, b' = P b, x = P^T x' -- perm=None computes RCM internally.
- **`jacobi_preconditioner(A)`** ★ — M = diag(A).
- **`ssor_preconditioner(A, omega=1.0)`** ★ — Symmetric SOR preconditioner (Saad, "Iterative Methods for Sparse Linear Systems," Ch. 4): for symmetric A = D + L + L^T, M = (D/omega + L) D^{-1} (D/omega + L^T) applied via one forward and one backward sparse triangular solve.
- **`incomplete_cholesky0(A)`** ★ — IC(0): incomplete Cholesky restricted to A's OWN lower-triangle sparsity pattern -- no fill-in beyond what A already has (the "(0)" level of fill).
- **`jacobi_preconditioner_torch(A, device='cpu', dtype=None)`** — Torch-native counterpart of jacobi_preconditioner() above.
- **`ssor_preconditioner_torch(A, omega=1.0, device='cpu', dtype=None)`** — Torch-native counterpart of ssor_preconditioner() above.
- **`preconditioned_cg(A, b, M=None, x0=None, tol=1e-08, maxiter=None, callback=None, backend='scipy', device='cpu')`** ★ — Hand-written preconditioned CG for SPD A (Hestenes & Stiefel 1952 recurrence -- see fem_implementation_lessons.md's own summary of the book's Chapter 11: convergence tracks sqrt(cond(K)), not cond(K), which is CG's headline ...
- **`jacobi_solve(A, b, x0=None, tol=1e-08, maxiter=10000)`** ★
- **`gauss_seidel_solve(A, b, x0=None, tol=1e-08, maxiter=10000)`** ★
- **`sor_solve(A, b, omega=1.5, x0=None, tol=1e-08, maxiter=10000)`** ★
- **`structured_quad_hierarchy(Lx, Ly, nx0, ny0, n_levels, x0=0.0, y0=0.0)`** ★ — Coarsest-first list of n_levels mesh.Mesh objects; level k has (nx0*2**k, ny0*2**k) Quad4 elements per direction.
- **`node_prolongation_matrix(nx_coarse, ny_coarse)`** ★ — Sparse bilinear prolongation P, shape (n_fine_nodes, n_coarse_nodes), between a mesh.rectangle_mesh(nx_coarse,ny_coarse) grid and the mesh.rectangle_mesh(2*nx_coarse, 2*ny_coarse) grid covering the same (Lx, Ly) -- standard ...
- **`dof_prolongation_matrix(P_node, dofs_per_node)`** ★ — Kronecker-expand a node-level prolongation matrix to DOF space (node-major numbering dof = dofs_per_node*node + component, matching solver.FESystem's own global-DOF convention).
- **`restrict_prolongation_to_free_dofs(P, free_fine, free_coarse)`** ★ — Restrict a full-DOF prolongation matrix to the free-DOF subspace at each level -- valid whenever "free" is a consistent GEOMETRIC predicate applied independently at every level (e.g. "not on the fixed edge"), which is the only ...
- **`galerkin_coarse_operator(A_fine, P)`** ★ — A_coarse = P^T A_fine P -- the Galerkin coarse-grid operator.
- **`v_cycle(A_levels, P_levels, b, n_pre=2, n_post=2, smoother='gauss_seidel', omega=1.0)`** ★ — One recursive V-cycle correction for A_levels[-1] x = b.
- **`multigrid_solve(A_levels, P_levels, b, tol=1e-08, maxiter=50, n_pre=2, n_post=2, smoother='gauss_seidel', omega=1.0)`** ★ — Standalone iterative solver built from repeated V-cycles: each outer iteration solves the RESIDUAL equation A e = r via one v_cycle() call and accumulates x += e -- the standard "multigrid as a stationary iterative method" usage ...

### `loads.py`

Load descriptions: nodal patterns, harmonic loads, PSD (random-vibration) loads and time histories.

- **`class LoadPattern(node_ids: np.ndarray, dof_index: int)`** — The spatial distribution of a load: `total` is split evenly across `node_ids` at local DOF `dof_index` -- the same simplified lumping convention used throughout this project's earlier scripts.
  - `.vector(n_dof, npn, total=1.0)`
- **`class TimeHistoryLoad(pattern: LoadPattern, time_fn: Callable[[float], float])`** — An arbitrary time-varying load: spatial pattern held fixed, magnitude given by any callable time_fn(t) -> float.
  - `.force_at(t, n_dof, npn)`
- **`class HarmonicLoad(pattern: LoadPattern, F0: float)`** — F(t) = F0 * e^{i*Omega*t}, magnitude/pattern only -- the frequency itself is supplied per-call to solve_harmonic()/ solve_frequency_sweep(), not stored here, since a sweep evaluates many frequencies against the same load.
  - `.force_vector(n_dof, npn)`
- **`class PSDLoad(pattern: LoadPattern, freqs: np.ndarray, psd: np.ndarray)`** — A statistically-defined load: spatial pattern plus a one-sided input power spectral density S_input(f) sampled at `freqs` (Hz).
  - `.force_vector(n_dof, npn)` — Unit-magnitude pattern -- solve_random_vibration() builds the transfer function H(f) from this and multiplies by psd itself, so the load magnitude ...

### `material.py`

Materials, sections and constitutive matrices (plane stress/strain, 3-D, beam rigidities, plasticity and hyperelastic material data).

- **`class Material(E: float, nu: float, rho: float=0.0)`** ★ — Isotropic linear-elastic material. G is derived, not stored, so E and nu can never silently disagree with G.
  - `.G` *(property)*
- **`class Section(A: float, I: float)`** ★ — Cross-section properties for 1-D (planar) beam elements.
- **`class Section3D(A: float, Iy: float, Iz: float, J: float)`** ★ — Cross-section properties for the 3-D frame element (elements.beams3d.Beam3DEulerBernoulli), Module 18 -- a separate dataclass from `Section` rather than extending it, since a 3-D member genuinely needs two bending axes plus ...
- **`class PlasticMaterial1D(E: float, sigma_y: float, H: float=0.0)`** ★ — Uniaxial (1-D) elasto-plastic material: linear elastic up to sigma_y, then linear ISOTROPIC hardening with modulus H (H=0 is perfectly plastic -- flat yield plateau, no further hardening).
- **`class PlasticMaterialJ2(E: float, nu: float, sigma_y: float, H: float=0.0)`** ★ — 3-D (or plane-strain) J2/von Mises elasto-plastic material, Module 19 -- the general-continuum generalization of PlasticMaterial1D above, used with elements.solids.Hex8PlasticJ2.
  - `.mu` *(property)* — Shear modulus.
  - `.kappa` *(property)* — Bulk modulus.
- **`class NeoHookeanMaterial(E: float, nu: float)`** ★ — Compressible Neo-Hookean hyperelastic material, Module 19, used with elements.solids.Tet4NeoHookean.
  - `.mu` *(property)*
  - `.kappa` *(property)*
- **`j2_radial_return_3d(eps_voigt, eps_p_n, alpha_n, mat)`** ★ — One elastic-predictor/radial-return step of small-strain J2 (von Mises) plasticity with linear isotropic hardening -- the 3-D generalization of TrussPlastic2D's 1-D return map. STATELESS: does not mutate eps_p_n/alpha_n, just ...
- **`j2_radial_return_plane_stress(eps_ps_trial, eps_p_n_voigt6, alpha_n, eps33_n, mat, tol=1e-10, max_iter=30)`** ★ — Simo & Taylor (1986), "A return mapping algorithm for plane stress elastoplasticity" -- the plane-stress-specific extension j2_radial_return_3d()'s own docstring names as needing a genuinely different treatment: plane stress ...
- **`class PlasticMaterialJ2Kinematic(C_kin: float=0.0, gamma_AF: float=0.0)`** ★ (extends `PlasticMaterialJ2`) — PlasticMaterialJ2 + Armstrong-Frederick (1966) nonlinear kinematic hardening
- **`j2_radial_return_3d_kinematic(eps_voigt, eps_p_n, alpha_n, beta_n, mat, tol=1e-10, max_iter=30, h_tangent=1e-07)`** ★ — One elastic-predictor/return step of small-strain J2 plasticity with COMBINED linear isotropic (H) + Armstrong-Frederick nonlinear kinematic (C_kin, gamma_AF) hardening -- the general-hardening extension j2_radial_return_3d()'s ...
- **`neo_hookean_pk2_stress(F, mat)`** ★ — 2nd Piola-Kirchhoff stress S(F) for the compressible Neo-Hookean model in NeoHookeanMaterial's docstring, derived by direct differentiation S = 2*dW/dC
- **`D_plane_stress(mat)`** ★
- **`D_plane_strain(mat)`** ★
- **`D_solid3d(mat)`** ★
- **`D_mindlin_plate(mat, h, k_shear=5.0 / 6.0)`** ★ — Returns (Db, Ds): bending and shear constitutive matrices for a Reissner-Mindlin plate of thickness h.
- **`D_shell(mat, h, k_shear=5.0 / 6.0)`** ★ — Returns (Dm, Db, Ds, h): membrane, bending, and shear constitutive matrices for elements.shells.Shell4MITC, plus the thickness itself (needed separately since Dm, unlike Db/Ds, does NOT bake h in -- see below).
- **`shell_rho_matrix(mat, h)`** ★ — Returns the 6x6 diagonal density matrix for Shell4MITC's LOCAL per-node DOF ordering (u, v, w, theta_x, theta_y, theta_z): translational entries rho*h (all three identical -- isotropic mass density, correct for any orientation of ...
- **`EI_beam(mat, sec)`** ★ — Bending rigidity for the 1-D Euler-Bernoulli beam element.
- **`beam3d_rigidities(mat, sec)`** ★ — (EA, GJ, EIy, EIz) -- the four rigidities elements.beams3d.Beam3DEulerBernoulli.stiffness() needs, bundled once so the element itself only handles the FE algebra, not material/section bookkeeping -- mirrors EI_beam()'s role for ...
- **`beam3d_mass_props(mat, sec)`** ★ — (rho*A, rho*Ip) -- the two mass properties Beam3DEulerBernoulli.mass() needs.

### `mesh.py`

Mesh containers and structured mesh generators (lines, rectangles, boxes), node selection helpers (`nodes_on_line`, ...) and multi-block meshes with per-block elements.

- **`class Mesh(nodes: np.ndarray, elements: np.ndarray, dim: int, point_data: dict=field(default_factory=dict), cell_data: dict=field(default_factory=dict), field_data: dict=field(default_factory=dict))`** ★
  - `.register_point_data(name, array)` — Attach a per-node field (any array whose first axis has length == len(self.nodes)) under `name` in self.point_data, mirroring TensorMesh's ...
  - `.register_element_data(name, array)` — Attach a per-element field (any array whose first axis has length == len(self.elements)) under `name` in self.cell_data, mirroring TensorMesh's ...
  - `.nodes_on_line(axis, value, tol=1e-09)` — 1-D/2-D selector: axis in {0:x, 1:y}.
  - `.nodes_on_plane(axis, value, tol=1e-09)` — 3-D selector: axis in {0:x, 1:y, 2:z}.
  - `.compute_boundary_mask(store_as='is_boundary')` — topology-derived boundary-node detection, complementing nodes_on_line()/nodes_on_plane() above -- those only work when the boundary of interest ...
  - `.check_quality(elem_formulation, verbose=True)` — Evaluate the Jacobian determinant at every Gauss point of every element using elem_formulation's OWN shape functions and Gauss order (see ...
  - `.check_grading(growth_ratio_cap=None, verbose=True)` — Realized neighbor-to-neighbor element-size-ratio diagnostic -- complements check_quality()'s detJ-based inversion/degeneracy check with the OTHER ...
- **`class MultiBlockMesh(nodes: np.ndarray, blocks: dict, dim: int, point_data: dict=field(default_factory=dict), cell_data: dict=field(default_factory=dict), field_data: dict=field(default_factory=dict))`** ★ — A mesh with MORE THAN ONE element topology sharing one node array.
  - `.register_point_data(name, array)` — Same contract as Mesh.register_point_data() -- see that method's own docstring.
  - `.register_element_data(block_name, name, array)` — Per-block analogue of Mesh.register_element_data() -- the block whose field is being attached must be named explicitly (block_name), since ...
  - `.nodes_on_line(axis, value, tol=1e-09)`
  - `.nodes_on_plane(axis, value, tol=1e-09)`
  - `.compute_boundary_mask(store_as='is_boundary')` — MultiBlockMesh analogue of Mesh.compute_boundary_mask() -- same edge-/face-adjacency mechanism (an edge/face shared by exactly one element, across ...
  - `.total_elements()`
  - `.elements_of_type(block_name)` — Convenience accessor for code (e.g. plotting) that wants one block's raw connectivity array without reaching into.blocks directly -- mirrors the ...
  - `.check_quality(elem_formulations, verbose=True)` — Generalizes Mesh.check_quality() across blocks: elem_formulations is a dict with the SAME keys as self.blocks, each value the Element instance to ...
- **`line_mesh(L, n, x0=0.0)`** ★ — n 2-node line elements spanning [x0, x0+L] -- for 1-D beam elements.
- **`rectangle_mesh(Lx, Ly, nx, ny, x0=0.0, y0=0.0)`** ★
- **`rectangle_with_hole_mesh_quarter(a, b, R, nr, ntheta, grade_p=2.0)`** ★ — One quadrant of a rectangle with a circular hole at the origin, mapped/transfinite mesh (hole boundary -> outer rectangle edge).
- **`rectangle_mesh_from_partitions(x_coords, y_coords)`** ★ — Structured Quad4 mesh from EXPLICIT (possibly non-uniform) x/y coordinate arrays -- generalizes rectangle_mesh()'s uniform np.linspace to any strictly increasing partition, so callers can grade element size (fine near a feature, ...
- **`weld_meshes(mesh_a, mesh_b, tol=1e-09)`** ★ — Merge two 2-D Quad4 meshes sharing one node array's worth of coincident boundary nodes into a single connected Mesh -- e.g. stitching a near-feature mapped block (rectangle_with_hole_mesh_ quarter) to a plain/graded extension ...
- **`rectangle_with_hole_mesh_quarter_full(a, b, R, nr, ntheta, grade_p=2.0, n_extend=None, extend_grade=2.0)`** ★ — rectangle_with_hole_mesh_quarter(), fixed for a != b (elongated quadrants): that function traces its outer boundary at UNIFORMLY SPACED ANGLES, so for a >> b or b >> a almost every ray lands on the SHORT edge and only a couple ...
- **`mirror_mesh(mesh, mirror_x=False, mirror_y=False, tol=1e-09)`** ★ — Mirror a 2-D Quad4 mesh across x=0 and/or y=0, welding nodes that land on the mirror line so the result is one clean connected mesh.
- **`hole_in_rectangle_mesh(Lx, Ly, hole_center, R, nr, ntheta, grade_p=2.0, n_extend=None, extend_grade=2.0, weld_tol=1e-09)`** ★ — A circular hole ANYWHERE inside a (not necessarily symmetric) rectangle [0,Lx] x [0,Ly] -- the generalization of rectangle_with_hole_mesh_quarter_full() beyond "hole at the origin of an already-quarter-symmetric domain" (that ...
- **`hole_in_rectangle_mesh_graded(Lx, Ly, hole_center, R, nr, ntheta, h_far, growth_ratio=None, n_extend=None, weld_tol=1e-09)`** ★ — hole_in_rectangle_mesh(), but `grade_p`/`extend_grade` are DERIVED from a target growth ratio instead of hand-picked -- the actual "growth-ratio-driven grading" capability the generalized-grading design calls for, not just the ...
- **`fillet_in_rectangle_mesh(Lx, Ly, corner, R, nx, ny, nr, ntheta=8, corner_grade_p=1.0)`** — A rectangle [0,Lx] x [0,Ly] with ONE corner rounded off -- `corner` is one of the rectangle's own 4 corner points (Lx,Ly), (0,Ly), (0,0), or (Lx,0) (checked against the actual rectangle, not guessed from a name/index).
- **`notch_in_rectangle_mesh(Lx, Ly, edge, s0, s1, depth, n_left, n_mid, n_right, n_depth, n_wall)`** — A rectangle [0,Lx] x [0,Ly] with a rectangular slot cut inward from one boundary edge -- the straight-walled notch geometry grading.Notch's own docstring scopes this feature to (see that class for why an arbitrary notch path was ...
- **`extrude_mesh(mesh2d, Lz, nz, z0=0.0)`** ★ — Turn any Quad4 Mesh into a Hex8 Mesh by stacking nz layers along z.
- **`box_mesh(Lx, Ly, Lz, nx, ny, nz)`** ★
- **`box_with_hole_mesh(a, b, R, Lz, nr, ntheta, nz, grade_p=2.0)`** ★
- **`plot_mesh_2d(mesh, ax, color='0.4', lw=0.5)`** ★
- **`plot_mesh_3d(mesh, ax, color='0.4', lw=0.4)`** ★
- **`plot_mesh_annotated(mesh, annotations=None, title='', figsize=(8, 6))`** ★ — 2-D mesh plot (grid via plot_mesh_2d) with a set of highlighted node groups drawn on top -- boundary conditions, loads, symmetry/free edges -- each described by one dict in `annotations`

### `mesh_io.py`

Read/write meshes in external formats through `meshio` (optional dependency).

- **`mesh_from_meshio(raw, dim)`** — Convert an in-memory `meshio.Mesh` into a `Mesh` (single element type) or `MultiBlockMesh` (more than one), the meshio-object analogue of geometry.gmsh_engine.UnifiedGeometryEngine._extract_ mesh() -- deliberately mirroring that ...
- **`read_mesh(filepath, dim, **kwargs)`** — Read any file format `meshio` understands (Abaqus INP, Nastran BDF, VTK/VTU, Gmsh MSH, UNV, MED,...) into a `Mesh`/`MultiBlockMesh`.
- **`mesh_to_meshio(mesh)`** — Convert a `Mesh` or `MultiBlockMesh` into an in-memory `meshio.Mesh`, carrying over `.point_data` and, per-block, `.cell_data` if the mesh has any registered ( 's register_point_data()/register_element_data()) -- the reverse ...
- **`write_mesh(mesh, filepath, **kwargs)`** — Write a `Mesh`/`MultiBlockMesh` to any file format `meshio` can write (inferred from `filepath`'s extension, or forced via `file_format=` in `**kwargs`).

### `mesh_transform.py`

Mesh-wide reference-element/Jacobian precomputation cache used by the vectorised assembly path. (Some functions use PyTorch, optional.)

- **`class MeshTransformation(mesh, formulation, gauss_order=None, connectivity=None, backend='numpy', device='cpu')`** — Precomputed, mesh-wide reference-element/Jacobian data for one (mesh, element formulation, Gauss order) combination.
- **`interpolate_point_data(mesh_transform, field)`** — Interpolate a nodal (`point_data`-style) field to every Gauss point of every element in `mesh_transform`.
- **`interpolate_point_data_gradient(mesh_transform, field)`** — Physical-space GRADIENT of a nodal field at every Gauss point of every element in `mesh_transform` -- the `grad<name>` half of `interpolate_point_data()` above (TensorMesh's Forms page: "grad + key in point_data...

### `mixed_assembly.py`

Mixed-field assembly with independent interpolation order per field (for example displacement-pressure). Use it for incompressible or Stokes-like problems.

- **`class BlockDofLayout(field_sizes)`** — A named-field DOF layout: several fields, each with its own size, concatenated into one flat vector with a fixed field order.
  - `.slice(name)`
  - `.split(vec)`
  - `.cat(parts)`
  - `.zeros()`
  - `.block_slice_pair(name_row, name_col)`
- **`check_bilinearity(form_fn, dim_a, dim_b, rng=None, atol=1e-08, rtol=1e-06)`** — `form_fn(a_vec, b_vec) -> float`, presumed to be a BILINEAR form (i.e. `form_fn(a, b) = a^T @ K_ab @ b` for some fixed matrix `K_ab`).
- **`D_deviatoric_3d(mat)`** — The deviatoric-only part of `material.D_solid3d(mat)`: `D_dev = D_solid3d(mat) - kappa * (m @ m.T)`, `m = [1,1,1,0,0,0]`, `kappa = E/(3*(1-2*nu))`.
- **`tet10_p1_mixed_element_blocks(elem_coords, D_dev, kappa)`** — (Kuu (30,30), Kup (30,4), Kpp (4,4)) for ONE straight-sided Tet10 element, via the SAME 4-point simplex quadrature `Tet10Solid3D.stiffness()` itself uses (`tet_quadrature_4pt()`) -- `Tet10Solid3D.B_matrix()` supplies `Bu`/`detJ` ...
- **`build_pressure_dof_map(connectivity)`** — `connectivity`: `(n_elements, 10)` Tet10 global node-id array (columns 0-3 are corner nodes, per `Tet10Solid3D`'s own convention -- see that class's docstring).
- **`assemble_mixed_tet10_p1(mesh, connectivity, mat, kappa=None, verify_bilinearity=True)`** — Global `(Kuu, Kup, Kpp)` for a Tet10 mesh, plus the `BlockDofLayout` and pressure-dof map a caller needs to apply BCs/interpret `p`.
- **`solve_mixed_static(layout, Kuu, Kup, Kpp, F_u, fixed_u_dofs)`** — Bordered (saddle-point) direct solve of

### `nonlinear_solver.py`

Nonlinear static and transient drivers: load-stepping Newton, displacement control, arc-length (Crisfield), Koiter-Newton, Newmark-Newton, and contact formulations.

- **`solve_contact_lagrange_static(fesystem, mat, contact_node, n_hat, g0, n_steps=10, tol=1e-10, max_iter=50, verbose=False, load_factors=None, **kwargs)`** — Lagrange-multiplier contact: EXACT (not approximate) enforcement of zero penetration at a single contact node, via one extra scalar unknown lambda (the contact reaction magnitude) -- the "extra equation in the global matrix" that ...
- **`solve_contact_augmented_lagrange_static(fesystem, mat, contact_node, n_hat, g0, k_p, n_steps=10, tol=1e-10, max_iter=50, al_tol=1e-09, al_max_iter=30, verbose=False, load_factors=None, ...)`** — augmented- Lagrangian contact enforcement -- the third contact-enforcement option this module now offers, sitting alongside (never replacing) the pure penalty method (element.GapContactPenalty, added via ...
- **`solve_nonlinear_static(fesystem, mat, n_steps=10, tol=1e-08, max_iter=30, verbose=False, load_factors=None, du_tol=None, energy_tol=None, line_search=True, **kwargs)`** — Load-controlled incremental Newton-Raphson.
- **`solve_nonlinear_displacement_control(fesystem, mat, control_dof, u_target_array, tol=1e-10, max_iter=50, verbose=False, du_tol=None, energy_tol=None, line_search=True, **kwargs)`** — Prescribes u[control_dof] = u_target_array[i] at each step i (control_dof is treated as an extra support with a nonzero, changing prescribed value -- it must NOT already be in fesystem.fixed_dofs), Newton-Raphson-solves for every ...
- **`solve_nonlinear_arc_length(fesystem, mat, delta_L, n_steps=50, tol=1e-08, max_iter=30, verbose=False, du_tol=None, energy_tol=None, line_search=True, **kwargs)`** — Crisfield cylindrical arc-length continuation -- see the module docstring above for the method.
- **`solve_nonlinear_koiter_newton(fesystem, mat, delta_L, n_steps=50, tol=1e-08, max_iter=30, predictor_tol=0.1, fd_rel=0.01, growth_factor=2.0, max_growth_iters=6, ...)`** — single-branch predictor/ corrector path-following, replacing solve_nonlinear_arc_length()'s LINEAR tangent predictor with a CUBIC one built from a genuine Koiter asymptotic expansion of the equilibrium path at the current point ...
- **`solve_nonlinear_koiter_newton_generic(fesystem, mat, delta_L, n_steps=50, tol=1e-08, max_iter=30, predictor_tol=0.1, fd_rel=0.01, growth_factor=2.0, max_growth_iters=6, ...)`** — /10 SCOPE NOTE: neither was added to this function's correctors.
- **`solve_nonlinear_static_koiter_newton(fesystem, mat, tol=1e-08, max_iter=30, max_expansions=20, predictor_tol=0.1, fd_rel=0.01, shrink_factor=0.5, max_shrink_iters=10, verbose=False, ...)`** — Koiter-Newton solve to the SINGLE prescribed target load F_ext = 1.0*fesystem.F (same load-scale convention as solve_nonlinear_static()) -- NOT a path tracer like solve_nonlinear_koiter_newton() above.
- **`solve_nonlinear_transient(fesystem, mat, load, T_total, dt, beta=0.25, gamma=0.5, u0=None, v0=None, tol=1e-08, max_iter=30, verbose=False, line_search=True, ...)`** — Newmark-beta implicit time integration for a GEOMETRICALLY (or materially) NONLINEAR structure -- see the module docstring above and docs/nonlinear_transient_dynamics_roadmap.md for the full design rationale.
- **`solve_transient_explicit_nonlinear(fesystem, mat, load, T_total, dt, u0=None, v0=None, **kwargs)`** — nonlinear generalization of FESystem.solve_transient_explicit() -- the SAME central-difference recursion on a LUMPED (diagonal) mass matrix, with the linear Kff @ d term replaced everywhere by the current nonlinear internal-force ...
- **`solve_transient_displacement_control(fesystem, mat, control_dof, u_target_fn, T_total, dt, beta=0.25, gamma=0.5, u0=None, v0=None, v0_control=None, a0_control=None, tol=1e-08, ...)`** — the dynamic (Newmark-implicit) counterpart to solve_nonlinear_ displacement_control() above -- exactly the same generalization solve_nonlinear_transient() already made for load-controlled solve_nonlinear_static(): keep the same ...

### `postprocess.py`

Result post-processing: stresses, strains, reactions and derived field quantities from a displacement solution.

- **`modal_participation_factors(mode_shapes_free, Mff, influence_vector)`** — L_i = phi_i^T M r, where r is the spatial 'influence vector' (e.g. a unit vector in the excited DOF direction for base excitation, or a load pattern for a distributed force).
- **`effective_modal_mass(mode_shapes_free, Mff, influence_vector)`** — Effective modal mass per retained mode: how much of the total excited mass each mode captures.
- **`variance_from_psd(freqs, S)`** — Response variance (= RMS^2) from a one-sided PSD via trapezoidal integration over frequency (Hz).
- **`rms_from_psd(freqs, S)`**

### `solver.py`

The core: `FESystem` assembles K/M/C/F, applies loads and constraints, and runs static, modal, buckling, harmonic, random-vibration and transient solves. Almost every workflow starts here.

- **`class FESystem(mesh, elem_formulation, thickness=1.0, sparse=False, backend='scipy', device='cpu')`** ★
  - `.assemble_stiffness(D, method='full', vectorized=False, **kwargs)` — kwargs are passed straight through to the chosen element method -- e.g. thickness=t for Quad4PlaneStress, or gauss_order= for a one-off ...
  - `.assemble_mass(rho_or_matrix, **kwargs)`
  - `.assemble_lumped_mass(rho_or_matrix, **kwargs)` — Element-by-element HRZ lumping (element.Element.lumped_mass()) -- required before solve_transient_explicit().
  - `.assemble_geometric_stiffness(N, **kwargs)` — global geometric ("stress stiffness") matrix K_sigma, built from a REFERENCE axial force state N (TENSION-POSITIVE, see Element.
  - `.assemble_internal_force(u_global, mat, **kwargs)` — Global internal-force vector F_int(u_global).
  - `.assemble_tangent_stiffness(u_global, mat, **kwargs)` — Global tangent stiffness K_T(u_global) = d(F_int)/d(u), at the current displacement state -- must be reassembled every Newton-Raphson iteration ...
  - `.add_contact_element(elem, node_ids, mat)` — Registers a contact/constraint element (e.g. element.GapContactPenalty) acting on node_ids of the MAIN mesh, with its own mat (e.g. (k_p, g0, n_hat)) ...
  - `.init_state()` — Call once, before the first nonlinear solve, for any system that uses a path-dependent element.
  - `.commit_all_states(u_global, mat, **kwargs)` — Advance self.state to the converged response at u_global -- call this once per load/displacement step, AFTER Newton-Raphson has converged, never ...
  - `.init_iter_state()` — Call once, before the first nonlinear solve, for any system that uses a mixed-formulation element.
  - `.update_iter_states(u_global, delta_u_global, mat, **kwargs)` — Advance self.iter_state using the REALIZED Newton correction delta_u_global a driver's own linear solve just produced -- call this EVERY Newton ...
  - `.assemble_damping(damping, **kwargs)` — damping: a damping.RayleighDamping instance, a damping.FieldDamping instance, or a raw damping matrix (plain ndarray, or scipy.sparse matrix when ...
  - `.fix_dofs(node_ids, dof_indices, value=0.0)` — dof_indices: which local DOF(s) at each node to constrain (0-based, element-formulation-specific, e.g. 0,1 for u,v).
  - `.add_nodal_force(node_ids, dof_index, total_force)` — Splits total_force evenly across node_ids at local DOF dof_index -- the same simplified load-lumping convention used throughout this project's ...
  - `.add_consistent_edge_load(node_pairs, dof_index, traction, thickness=1.0)` — Consistent nodal load for a uniform traction along a chain of 2-node edge segments (node_pairs = [(n0,n1), (n1,n2),...]), weighted by each segment's ...
  - `.add_consistent_facet_load(formulation, facets, dof_index, traction, quad_order=2, thickness=1.0)`
  - `.free_dofs` *(property)*
  - `.fixed_dofs_array` *(property)* — `fixed_dofs` (a set) in a fixed, sorted order, matching `free_dofs`'s own convention -- so a caller pairing this against `_fixed_dof_values_array()` ...
  - `.solve_static(verbose=False, method=None, **method_kwargs)` — Dense: SPD-aware Cholesky solve when Kff is SPD (the common, well-constrained case -- see _dense_spd_solve() ), falling back to the shared ...
  - `.solve_modal(n_modes=4)` — Generalized eigenproblem K*phi = omega^2*M*phi on the free DOFs.
  - `.solve_linear_buckling(n_modes=4)` — the linear (eigenvalue) buckling problem (K + lambda*K_sigma)*phi = 0 on the free DOFs, i.e. K*phi = -lambda*K_sigma*phi -- the exact same ...
  - `.solve_transient_implicit(load, T_total, dt, beta=0.25, gamma=0.5, u0=None, v0=None)` — Newmark-beta direct time integration (default: average- acceleration, unconditionally stable for this linear, time-invariant system).
  - `.critical_timestep()` — dt_crit = 2/omega_max, the central-difference stability limit, from the FULL eigenspectrum of the free system (not just the first few modes retained ...
  - `.critical_timestep_local(wave_speed, alpha=0.9)` — the book's cheap LOCAL per-element estimate dt_crit = alpha * min_e(l_e / c_e), an O(n_elements) alternative to critical_timestep()'s exact but ...
  - `.solve_transient_explicit(load, T_total, dt, u0=None, v0=None)` — Central-difference explicit time integration on a LUMPED mass matrix.
  - `.solve_modal_superposition(load, T_total, dt, n_modes, zeta)` — Reduces the transient problem to n_modes decoupled SDOF equations via the mass-normalized mode shapes from solve_modal() (scipy.linalg.eigh already ...
  - `.solve_harmonic(Omega, F0_vector)` — Direct complex solve of (-Omega^2*M + i*Omega*C + K) U0 = F0 at a single driving frequency Omega (rad/s) -- no time marching.
  - `.solve_frequency_sweep(Omega_array, F0_vector)` — solve_harmonic() at every frequency in Omega_array (rad/s).
  - `.solve_random_vibration(freqs_hz, psd_input, F0_pattern, output_dof)` — PSD (random vibration) formulation: S_out(f) = \|H(f)\|^2 * S_in(f), where H(f) is the transfer function from the unit-magnitude load F0_pattern to ...

### `topopt.py`

Density-based (SIMP) topology optimisation with an adjoint sensitivity and volume constraint. Needs PyTorch.

- **`simp_scale(rho, p=3.0, rho_min=0.001)`** — Per-element SIMP stiffness scale factor: `s(rho) = rho_min + (1 - rho_min) * rho**p`.
- **`simp_scale_grad(rho, p=3.0, rho_min=0.001)`** — d(simp_scale)/d(rho) = p * (1 - rho_min) * rho**(p-1) -- the closed-form derivative `compliance_sensitivity_closed_form()` uses directly (no autograd needed for this scalar 1-D function).
- **`unit_stiffness_stack(fesystem, D, gauss_order=None)`** — The mesh-wide `(n_elements, n_edof, n_edof)` stiffness stack at `rho=1` everywhere -- computed ONCE via items 106/107 (`MeshTransformation` + `tensorized_element_stiffness`), then reused (just rescaled per element, every OC ...
- **`assemble_K_from_scale(ke_unit, scale, connectivity, dofs_per_node, n_dof)`** — Global K(rho) via item 107's vectorized scatter, given the already-rescaled per-element stiffness stack `ke_unit * scale[:, None, None]` -- one call, no Python loop over elements.
- **`solve_simp_equilibrium(ke_unit, connectivity, dofs_per_node, n_dof, free_dofs, F_ext, rho, p, rho_min)`** — K(rho) u = F_ext on the free dofs, via the SAME SPD-aware Cholesky-then-eigen dispatch `FESystem.solve_static()` itself uses -- reused directly (both are `@staticmethod`), not reimplemented, since a SIMP density field genuinely ...
- **`compliance(F_ext, u_full)`** — C = F_ext.
- **`compliance_sensitivity_closed_form(ke_unit, connectivity, dofs_per_node, u_full, rho, p, rho_min)`** — dC/drho_e = -simp_scale_grad(rho_e) * u_e^T @ ke_unit_e @ u_e -- the textbook self-adjoint compliance-sensitivity formula (e.g. Bendsoe & Sigmund 2003, Sec.1.3): for C = F^T u subject to K(rho)u=F with F independent of rho, the ...
- **`compliance_sensitivity_via_adjoint(ke_unit, connectivity, dofs_per_node, n_dof, free_dofs, u_full, Kff, Ff, rho, p, rho_min)`** — The SAME dC/drho as compliance_sensitivity_closed_form(), computed instead through 's own general machinery -- differentiable.adjoint_gradient(K_eff, grad_w, vjp_fn) -- reused here EXACTLY as-is, with: K_eff = Kff (this problem ...
- **`build_filter_weights(centroids, radius)`** — Sparse (dense-array-returned, meshes here are small) weight matrix `H[e,j] = max(0, radius - dist(e,j))` for every pair within `radius` -- Sigmund's original linear-decay filter kernel.
- **`apply_sensitivity_filter(H, rho, dc, eps=1e-09)`** — Sigmund's original (1999/2001) heuristic sensitivity filter: `dc_tilde_e = (1 / (rho_e * sum_j H_ej)) * sum_j H_ej * rho_j * dc_j` -- damps checkerboarding without a separate density-filtering pass.
- **`oc_update(rho, dc, volume_fraction, move=0.2, rho_min_bound=0.001, rho_max_bound=1.0, bisection_tol=0.0001, max_bisection_iter=100)`** — Classic Optimality-Criteria density update (Bendsoe 1995; Sigmund's "99 line" code): for a fixed trial Lagrange multiplier `lmid` on the volume constraint,
- **`topology_optimize_compliance(fesystem, D, F_ext, volume_fraction, p=3.0, rho_min=0.001, move=0.2, n_iter=50, filter_radius=None, sensitivity='closed_form', rho_init=None, ...)`** — Full SIMP compliance-minimization loop: solve -> sensitivity -> (optional filter) -> Optimality-Criteria update -> repeat.

### `torch_sparse_solver.py`

Torch dense and sparse-CG static solves on CPU or GPU behind `FESystem(backend="torch")`.

- **`torch_dense_solve(K, F, device='cpu', dtype=None)`** — Densify K (any SciPy-sparse or dense input) and solve via torch.linalg.solve() -- see this module's own docstring for when this is the right choice vs. torch_sparse_cg_solve().
- **`torch_sparse_cg_solve(K, F, tol=1e-08, max_iter=None, device='cpu', dtype=None)`** — Jacobi-preconditioned Conjugate Gradient solve of K @ x = F on a genuine torch.sparse_csr_tensor -- no densification, so this scales to the larger systems (Hex20/Tet10/graded meshes, see this module's own docstring) a dense path ...
- **`fesystem_solve_static_torch(sysobj, method='cg', device='cpu', tol=1e-08, max_iter=None)`** — Drop-in alternative to FESystem.solve_static() (solver.py) that routes the SAME free-dof linear solve through PyTorch instead of SciPy -- the concrete demonstration of "a GPU solve path...

### `vectorized_assembly.py`

Tensorised global stiffness assembly with no Python loop over elements. Use it for speed on large meshes. (Some functions use PyTorch, optional.)

- **`build_B_batched(shape_grad, dofs_per_node)`** — Batched strain-displacement operator B for EVERY element and Gauss point in one array call -- the same row-construction convention every existing dofs_per_node=2 (Quad4PlaneStress, Quad8PlaneStress) or dofs_per_node=3 ...
- **`tensorized_element_stiffness(mesh_transform, D, thickness=1.0)`** — All `n_elements` local stiffness matrices in one pass (no Python loop over elements) via a single einsum contraction over item 106's precomputed B/JxW tensors.
- **`scatter_global_stiffness(ke, connectivity, dofs_per_node, n_dof, sparse=False)`** — Vectorized (no Python loop over elements) scatter-add of a stacked local-stiffness array into a global matrix -- the second half of item 107's "no per-element loop anywhere in this path".
- **`assemble_stiffness_vectorized(fesystem, D, gauss_order=None, thickness=1.0, chunk_size=None, backend='numpy', device='cpu')`** — Additive, opt-in alternative to `FESystem.assemble_stiffness()`'s per-element Python loop -- computes and scatters the stiffness contribution of every block in `fesystem._blocks` via item 106/107's tensorized path, ADDING into ...



## `rom_engine` — reduced-order modeling

### `affine.py`

Offline/online split for parameter-dependent matrices K(mu) = sum theta_i(mu) K_i: project the components once, reassemble the small reduced matrix for each new parameter. The core of fast parameter sweeps.

- **`class AffineDecomposition(components, theta_func)`** ★ — A parameter-independent affine decomposition of a system matrix (or several -- e.g. stiffness AND mass, if both are affine in the parameters), with an offline projection step for fast reduced online queries.
  - `.assemble(mu)` — theta_1(mu)*K_1 +...
  - `.project(basis)` — Project every component matrix onto a reduced basis ONCE -- the offline step that makes assemble_reduced() cheap.
  - `.assemble_action(mu, q)` — A(mu) @ V @ q, i.e. the full-order EFFECT of applying the full operator to an already-reduced state q -- WITHOUT ever forming the full (n_dof, n_dof) ...
  - `.assemble_reduced(mu)` — theta_1(mu)*K_1,r +...
  - `.solve_reduced(mu, F_r)` — Convenience: assemble_reduced(mu) and directly solve K_r(mu) q = F_r for q, the single most common online query (a parametric static solve).

### `attractor.py`

Poincare/stroboscopic tools and frequency/amplitude sweeps for nonlinear dynamics, with optional multi-process execution (`n_jobs`).

- **`stroboscopic_sample(t, y, Omega, tol=0.001, min_periods=3, norm_ord=np.inf)`** ★ — Stroboscopic sampling of a trajectory `y(t)` at the forcing period `T = 2*pi/Omega`, with an AUTOMATIC settling check -- NOT a fixed arbitrary discard count.
- **`steady_state_amplitude(t, y, Omega, tol=0.001, min_periods=3, norm_ord=np.inf)`** ★ — Steady-state forced-response amplitude(s): `0.5*(max-min)` of the FULL (not stroboscopically decimated) trajectory `y`, evaluated over the LAST fully-settled forcing period (per `stroboscopic_sample()`'s own settling check) -- ...
- **`midspan_transverse_probe(state, n_nodes, dofs_per_node=3, transverse_dof=1, V=None)`** ★ — Transverse-displacement value at (or nearest) the midpoint node of a rod/beam model, generalized to any node count and dof layout.
- **`master_slave_data(t, Q, master=0, slave=1)`** ★ — `Q2` vs `Q1` and `Q2` vs `(Q1, Q1_dot)` extraction from a POD amplitude time series `Q(t)`, for reproducing the paper's own master-slave slow-invariant-manifold plots (Figs. 20-22), where the slaved mode's amplitude is shown as ...
- **`dominant_frequency(t, Q_m, n_peaks=1, rtol=1e-06)`** ★ — FFT-based dominant (angular) frequency of a single mode's own amplitude time series `Q_m(t)`, needed for the paper's "the slaved mode responds at 2x the master frequency" claim (Section 11) and its Section 7 amplitude-frequency ...
- **`class SweepPoint(param: float, t: np.ndarray, y: np.ndarray, ydot: 'np.ndarray | None', strobe_t: np.ndarray, strobe_y: np.ndarray, attractor_t: np.ndarray, attractor_y: np.ndarray, discard_periods: int, settled: bool, probe: 'np.ndarray | float | None', is_reduced: bool)`** ★ — One point of a frequency/amplitude sweep -- IDENTICAL fields regardless of whether it came from a full-order (FS) FE-model run or a reduced-order (RS) `IntrusiveNonlinearROM` run, which is the whole point of this common layout ...
- **`class SweepResult(param_name: str, params: np.ndarray, points: list=field(default_factory=list))`** ★ — A whole frequency- or amplitude-sweep: `param_name`/`params` record what was swept, `points` holds one `SweepPoint` per value, in the SAME order as `params`.
  - `.probes()` — Stacked array of `point.probe` across the sweep, shape `(n_points,)` (scalar probes) or `(n_points,...)` -- convenience for plotting an FRF-style ...
- **`frequency_sweep(model, Omega_values, **kwargs)`** ★ — Frequency sweep: runs one trajectory per `Omega` in `Omega_values` and stroboscopically samples EACH at that same `Omega` (the natural choice -- the forcing period is `2*pi/Omega`).
- **`amplitude_sweep(model, amplitude_values, Omega, **kwargs)`** ★ — Amplitude sweep: runs one trajectory per forcing amplitude in `amplitude_values`, ALL at the SAME forcing frequency `Omega` (needed to define the stroboscopic period, since amplitude alone doesn't).

### `balanced_truncation.py`

Balanced truncation (plain, frequency-weighted, singular-perturbation) for state-space model reduction with an H-infinity error bound. Works on first-order state-space form.

- **`controllability_gramian(A, B, backend='numpy', device='cpu')`** ★ — Solve A P + P A^T + B B^T = 0 for P.
- **`observability_gramian(A, Cout, backend='numpy', device='cpu')`** ★ — Solve A^T Q + Q A + Cout^T Cout = 0 for Q -- the dual of controllability_gramian(), same solver dispatch, same backend=/ device= convention, same A-must-be-stable requirement.
- **`hankel_singular_values(A, B, Cout, return_transform=False, backend='numpy', device='cpu')`** ★ — The system's Hankel singular values, via the square-root method (_balance_from_gramians(), fed the plain, unweighted controllability/ observability Gramians -- see frequency_weighted_hankel_singular_ values() for the Enns ...
- **`lowpass_weight(wc)`** ★ — A first-order low-pass SISO weight W(s) = wc / (s + wc), in controllable canonical form, DC gain 1 -- a convenience for frequency_weighted_gramians()/FrequencyWeightedBalancedTruncationROM's Wi/Wo arguments when the goal is ...
- **`bandpass_weight(omega_n, zeta=0.1)`** ★ — A second-order bandpass SISO weight W(s) = 2*zeta*omega_n*s / (s^2 + 2*zeta*omega_n*s + omega_n^2), peaking at s=i*omega_n, in controllable canonical form -- a convenience for emphasizing balanced-truncation accuracy near a ...
- **`frequency_weighted_gramians(A, B, Cout, Wi=None, Wo=None, backend='numpy', device='cpu')`** ★ — Enns (1984) frequency-weighted controllability/observability Gramians, restricted to the plant's own n states -- see module docstring / docs/classical_mor_roadmap.md Section 10 for the cascade-system derivation.
- **`frequency_weighted_hankel_singular_values(A, B, Cout, Wi=None, Wo=None, return_transform=False, backend='numpy', device='cpu')`** ★ — The Enns frequency-weighted analogue of hankel_singular_values() -- same _balance_from_gramians() step, fed frequency_weighted_ gramians()'s Pw/Qw instead of the plain Gramians.
- **`class BalancedTruncationROM(ss, hsv, T, Tinv, r)`** ★ — A reduced-order model built by balancing then truncating a stable first-order state-space realization.
  - `BalancedTruncationROM.from_MCK(M, K, B, Cout, C=None, r=10, backend='numpy', device='cpu')` — Build a BalancedTruncationROM directly from second-order mass/stiffness (and, optionally, damping) matrices plus an input map B and output map Cout ...
  - `.transfer_function(s)` — H_r(s) = Cout_r (s*I - A_r)^-1 B_r.
  - `.frequency_response(omega_array)` — H_r(i*omega) swept over omega_array -- same convention and return-shape rule as krylov.KrylovROM.frequency_response() (squeezed to a 1-D complex ...
  - `.h_infinity_error_bound()` — 2 * sum(discarded Hankel singular values) -- the classical a priori H-infinity error bound (eq.
  - `.is_stable()` — True iff every pole of the reduced model has negative real part.
- **`class SingularPerturbationROM(ss, hsv, T, Tinv, r)`** ★ — Singular perturbation approximation (SPA; Liu & Anderson 1989, "Singular perturbation approximation of balanced systems" -- also called "residualization", e.g. Antoulas, *Approximation of Large-Scale Dynamical Systems*, Ch. 9) -- ...
  - `SingularPerturbationROM.from_MCK(M, K, B, Cout, C=None, r=10, backend='numpy', device='cpu')` — Build a SingularPerturbationROM directly from second-order mass/stiffness (and, optionally, damping) matrices plus an input map B and output map Cout ...
  - `.transfer_function(s)` — H_r(s) = Cout_r (s*I - A_r)^-1 B_r + D_r.
  - `.frequency_response(omega_array)` — H_r(i*omega) swept over omega_array -- same convention and return-shape rule as BalancedTruncationROM.frequency_response()/ ...
  - `.h_infinity_error_bound()` — 2 * sum(discarded Hankel singular values) -- the SAME formula, and the SAME theorem (Liu & Anderson 1989 prove SPA shares BT's a priori H-infinity ...
  - `.is_stable()` — True iff every pole of the reduced model has negative real part.
- **`class FrequencyWeightedBalancedTruncationROM(ss, hsv, T, Tinv, r, Wi=None, Wo=None)`** ★ — Frequency-weighted balanced truncation (Enns 1984, "Model reduction with balanced realizations: An error bound and a frequency weighted generalization") -- docs/classical_mor_ roadmap.md Section 10 has the full derivation this ...
  - `FrequencyWeightedBalancedTruncationROM.from_MCK(M, K, B, Cout, C=None, r=10, Wi=None, Wo=None, backend='numpy', device='cpu')` — Build a FrequencyWeightedBalancedTruncationROM directly from second-order mass/stiffness (and, optionally, damping) matrices plus an input map B and ...
  - `.transfer_function(s)` — H_r(s) = Cout_r (s*I - A_r)^-1 B_r.
  - `.frequency_response(omega_array)` — H_r(i*omega) swept over omega_array -- same convention and return-shape rule as BalancedTruncationROM.frequency_response().
  - `.is_stable()` — True iff every pole of the reduced model has negative real part.

### `dataset_diagnostics.py`

Checks on a training set before fitting: parameter-space coverage gaps and related diagnostics.

- **`parameter_coverage_report(param_table, n_bins=10, param_names=None)`** ★ — Per-parameter coverage histogram + gap flag for a design-of- experiments parameter table.
- **`pca_dimensionality(X, variance_threshold=0.95)`** ★ — SVD-based PCA dimensionality estimate -- no scikit-learn dependency (this package already avoids adding one for a single small utility; SVD of the centered data matrix is the exact same computation scikit-learn's own PCA performs ...
- **`class GPSurrogate(length_scale=None, noise=1e-06, signal_var=None)`** ★ — A minimal, hand-rolled Gaussian-Process regressor -- isotropic squared-exponential kernel, median-heuristic length scale by default, small noise/nugget regularization, exact (Cholesky) training.
  - `.fit(X, y)`
  - `.predict(X, return_std=False)`
- **`sobol_indices(bounds, model_fn, n_samples=512, rng=None)`** ★ — First-order (Saltelli 2010) and total-order (Jansen 1999) Sobol' sensitivity indices for `model_fn`, a cheap callable (n, d) -> (n,).

### `differentiable_correction.py`

Trainable correction term added to a reduced equilibrium residual, calibrated through an adjoint or by residual minimisation. Needs PyTorch (research-grade).

- **`reduced_residual(q, Lambda, F_nl_fn, F_ext, correction=None)`** ★ — R(q) = F_ext - Lambda*q - F_nl_fn(q) - f_theta(q).
- **`class ScalarModalCorrection(n_modes, init=0.0, dtype=None)`** ★ — Simplest trainable correction at reduced-coordinate level: one free scalar per mode, f_theta(q) = theta (constant, independent of q).
  - `.value(q)`
  - `.torch_value(q_t)`
  - `.parameters()`
- **`calibrate_reduced_correction_explicit(q_reference, Lambda, F_nl_fn, F_ext, correction, n_epochs=500, lr=0.01, optimizer_cls=None, verbose=False)`** ★ — ROM-level counterpart of fea_engine.differentiable.calibrate_ correction_explicit() -- minimizes

### `ensemble_uq.py`

Train a model many times from different seeds and report ensemble mean and spread as a cheap uncertainty indicator. Needs PyTorch.

- **`class EnsembleUQ(model_factory, n_members=5, seeds=None)`** ★ — Trains `n_members` independently-seeded copies of the same model family and reports mean/std across the ensemble at predict time -- a cheap, well-understood epistemic-uncertainty proxy (see this module's own docstring).
  - `.fit(X, y, **fit_kwargs)` — Fits `n_members` fresh model instances, one per seed, each on the SAME (X, y) training data -- the ensemble's spread then reflects only the model ...
  - `.predict_all(X)` — Returns (n_members, n_points, n_out) -- every member's own prediction, un-aggregated (a caller wanting the raw per-member spread rather than just ...
  - `.predict_mean_std(X)` — Returns (mean, std), each (n_points, n_out) -- the ensemble mean prediction and its per-point, per-output-component standard deviation across members ...
- **`neural_surrogate_ensemble(n_modes, q_samples, F_samples, n_members=5, seeds=None, **surrogate_kwargs)`** ★ — Convenience constructor: builds and fits an EnsembleUQ of `NeuralSurrogate` instances -- the concrete, real use case this whole item exists for ("once ANY of items 99/ 100/103 introduces a trainable NN...

### `frequency.py`

Frequency-domain ROM: modal/POD projection of the dynamic stiffness, frequency response, residual norms for error estimation.

- **`class FrequencyROM(components, theta_func, basis)`** ★ — An intrusive reduced-order model of A(omega) = -omega^2*M + i*omega*C + K, built by composing affine.AffineDecomposition (for the {M, C, K} affine-in-omega structure) with galerkin.GalerkinROM (for projecting loads and expanding ...
  - `FrequencyROM.from_MCK(M, K, basis, C=None, rayleigh=None)` — Build a FrequencyROM directly from mass/stiffness (and, optionally, damping) matrices -- the usual entry point.
  - `.solve(omega, F_r)` — Solve the reduced system A_r(omega) q = F_r for q, at one frequency.
  - `.frequency_response(omega_array, F, output_dofs=None)` — Sweep omega_array, returning the FULL-order (expanded) response at each frequency -- the main "many-query" entry point this whole module exists for.
  - `.residual_norm(omega, F)` — The EXACT full-order residual norm \|\|F - A(omega) @ (V @ q)\|\| of the expanded reduced solution at this omega -- computed EFFICIENTLY, in O(Q * n_dof ...
  - `.error_estimate(omega, F)` — Backward-compatible alias for residual_norm() -- kept under its original name since it predates residual_norm() and existing callers/docs refer to ...
  - `.hierarchical_error_indicator(omega, F, comparison_rom)` — \|\|self.frequency_response([omega], F) - comparison_rom.frequency_response([omega], F)\|\| -- the DISAGREEMENT between this ROM and a second ...
- **`build_pod_basis_from_frf_snapshots(training_omegas, M, K, F, C=None, n_modes=None, energy_threshold=None)`** ★ — Full-order-solve A(omega) x = F at each of a handful of TRAINING frequencies and POD the resulting (complex) snapshot matrix -- an alternative to a modal (undamped-eigenmode) basis for FrequencyROM, better suited to capturing the ...

### `galerkin.py`

Galerkin (projection) reduced model: project K, M, F onto a basis and solve static, modal and transient problems in reduced space.

- **`class GalerkinROM(basis)`** ★ — A reduced-order model built by Galerkin-projecting a full-order linear system onto a fixed basis V.
  - `.project_matrix(A)` — V^T A V -- the reduced counterpart of a full (n_dof, n_dof) matrix (stiffness, mass, damping,...).
  - `.project_vector(b)` — V^T b -- the reduced counterpart of a full (n_dof,) or (n_dof, k) load/right-hand-side vector (or batch of vectors).
  - `.expand(q)` — V @ q -- reduced coordinates back to a full-order approximation.
  - `.reduce_system(K, M=None, F=None)` — Project the given full-order matrices/vector once and cache the results as K_r/M_r/F_r, so repeated solve_static()/ solve_modal() calls (e.g. under a ...
  - `.solve_static(F=None)` — Solve K_r q = F_r (F_r from the argument if given, else the cached one from reduce_system()) and expand back to full coordinates.
  - `.solve_modal(n_modes=None)` — Reduced generalized eigenproblem K_r phi = omega^2 M_r phi -- the SAME calculation solve_modal() on a full FESystem would do, just on the tiny ...

### `greedy.py`

Weak-greedy selection of frequency samples to train a frequency-response basis cheaply.

- **`greedy_train_frequency_basis(training_omegas, M, K, F, C=None, rayleigh=None, n_seed=3, tol=0.0001, max_modes=30)`** ★ — Weak-greedy construction of a POD-on-FRF-snapshots basis for FrequencyROM.

### `hankel_norm.py`

Optimal Hankel-norm approximation (best reduced model in the Hankel norm) for stable systems.

- **`class OptimalHankelNormROM(ss, hsv, T, Tinv, r, gap_tol=1e-08, reliability_tol=0.0001, backend='numpy', device='cpu')`** ★ — An order-r reduced model achieving \|\|G - G_r\|\|_Hankel = sigma_(r+1) EXACTLY (the Adamjan-Arov-Krein/Glover optimum) -- see module docstring for the full construction, the SISO restriction, and the numerical self-check this class ...
  - `OptimalHankelNormROM.from_MCK(M, K, B, Cout, C=None, r=10, gap_tol=1e-08, reliability_tol=0.0001, backend='numpy', device='cpu')` — Build an OptimalHankelNormROM directly from second-order mass/stiffness (and, optionally, damping) matrices plus a SINGLE input map B and a SINGLE ...
  - `.transfer_function(s)` — H_r(s) = Cout_r (s*I - A_r)^-1 B_r + D_r.
  - `.frequency_response(omega_array)` — H_r(i*omega) swept over omega_array -- same convention as BalancedTruncationROM.frequency_response(), always squeezed to a 1-D complex array (this ...
  - `.hankel_norm_error_bound()` — sigma_(r+1) -- the AAK-theoretical EXACT Hankel norm of G - G_r (not just a bound: the exact minimum over every possible order-r system, per the AAK ...
  - `.is_stable()` — True iff every pole of the reduced model has negative real part.

### `intrusive_nonlinear_rom.py`

Reduced model of a nonlinear finite element model that keeps the nonlinear internal force (intrusive projection) with time integration in reduced coordinates. (Some functions use PyTorch, optional.)

- **`impulse_to_reduced_velocity(V, M_r, P_full, tau0)`** ★ — Georgiou (2005) Eqs. 47-48, generalized: an impulsive full-order load of magnitude/pattern `P_full` (a plain (n_dof,) force vector) applied over a short duration `tau0` imparts a full-order momentum change `M @ v0 = P_full * ...
- **`class IntrusiveNonlinearROM(V, M, C, internal_force_fn, load_fn, tangent_fn=None)`** ★ — Intrusive nonlinear Galerkin ROM with a full (non-diagonal, non-mass-normalized) reduced mass/damping/stiffness system -- see module docstring for the full derivation and the reasoning behind keeping this deliberately separate ...
  - `.reduced_internal_force(q)` — f_int_r(q) = V^T @ internal_force_fn(V @ q) -- the FULL (linear + nonlinear) reduced restoring force, evaluated through the real full-order ...
  - `.f_nl(q)` — f_nl(q) = f_int_r(q) - K_r @ q -- the pure NONLINEAR part of the reduced restoring force (the FE-consistent equivalent of the paper's own ...
  - `.reduced_tangent(q)` — K_T_r(q) = V^T @ tangent_fn(V @ q) @ V -- the reduced tangent stiffness at a general (not necessarily zero) reduced state; used only by ...
  - `.reduced_load(t)` — F_ext_r(t) = V^T @ load_fn(t).
  - `.initial_conditions(u0_full=None, v0_full=None)` — Project a full-order initial displacement/velocity into reduced coordinates by mass-weighted projection through `M_r`
  - `.impulse_to_reduced_velocity(P_full, tau0)` — Instance-method convenience wrapper around the module-level `impulse_to_reduced_velocity()`, reusing this instance's own already-factored `M_r` (no ...
  - `.integrate_rk4(q0, qdot0, dt, n_steps)` — Fixed-step, classical 4-stage RK4 on `z = [q; qdot]`, built on `parameterized_latent_ode.rk4_step()`.
  - `.integrate_solve_ivp(q0, qdot0, t_span, t_eval=None, **kwargs)` — `scipy.integrate.solve_ivp` on the identical `_rhs()` used by `integrate_rk4()` -- any `solve_ivp` keyword (`method=`, `rtol=`, `atol=`,...) is ...
  - `.integrate_newton_newmark(q0, qdot0, dt, n_steps, beta=0.25, gamma=0.5, tol=1e-09, max_iter=30, max_backtrack=30)` — Implicit Newmark-beta time integration with a genuine Newton-Raphson corrector every step -- for stiff cases where ...
  - `.energy(q, qdot, strain_energy_fn=None)` — Total reduced mechanical energy at one instant: `0.5*qdot^T M_r qdot + U(q)`, where `U(q)` (the reduced strain energy) is supplied by the caller via ...
  - `.torch_reduced_matrices(device='cpu', dtype=None)` — Independent torch re-derivation of M_r/D_r/K_r -- genuine `V_t.T @ A_t @ V_t` torch matmuls on `self.M`/`self.C`/ `self._K0_full`, NOT ...
  - `.integrate_rk4_torch(q0, qdot0, dt, n_steps, device='cpu', dtype=None, internal_force_fn_torch=None, load_fn_torch=None)` — Torch-native mirror of integrate_rk4() -- IDENTICAL step logic, built on the SAME `parameterized_latent_ode.rk4_step()` (reused verbatim, not ...
  - `.integrate_newton_newmark_torch(q0, qdot0, dt, n_steps, beta=0.25, gamma=0.5, tol=1e-09, max_iter=30, max_backtrack=30, device='cpu', dtype=None, internal_force_fn_torch=None, ...)` — Torch-native mirror of integrate_newton_newmark() -- IDENTICAL a0c..a7c/Keff/G(q)/J(q) algebra (see that method's own docstring for the full ...
- **`class MatrixSPDReport(name: str, symmetric: bool, asymmetry: float, eigval_min: float, eigval_max: float, margin: float, positive_definite: bool)`** ★ (extends `NamedTuple`) — Symmetry / positive-definiteness report for one reduced matrix, reusing exactly the eigenvalue-margin style item 144's own test suite already built (`TestReducedMatricesSPD` in `tests/test_intrusive_nonlinear_rom.py`): symmetry ...
- **`class PodRMAnalysis(freq_sorted: np.ndarray, freq_pod_order: np.ndarray, E_hat_sorted: np.ndarray, E_hat_pod_order: np.ndarray, pod_order_index: np.ndarray, diag_E_hat: np.ndarray, uncoupled_freq: np.ndarray, spd_report: dict)`** ★ (extends `NamedTuple`) — Result of `pod_rm_analysis()` -- see that function's docstring for the meaning of every field.
- **`pod_rm_analysis(M_r, D_r, K_r, sym_tol=1e-09, pd_margin_tol=1e-08)`** ★ — POD-RM modal diagnostics from a fitted (or plain caller-supplied) reduced mass/damping/stiffness triple `(M_r, D_r, K_r)` -- Georgiou (2005) Eqs. 41-44, 50-60.

### `krylov.py`

Krylov / moment-matching model reduction (one-sided and two-sided) around an expansion frequency.

- **`arnoldi_basis(A, E, B, k, s0=0.0, tol=1e-12)`** ★ — Orthonormal (modified Gram-Schmidt) basis of the order-k block Krylov subspace K_k((A - s0*E)^-1 E, (A - s0*E)^-1 B), built from ONE dense LU factorization of (A - s0*E), reused for every iteration (each new "generation" of ...
- **`two_sided_arnoldi_bases(A, E, B, Cout, k, s0=0.0, tol=1e-12, cond_tol=1000000000000.0)`** ★ — Build a biorthogonalized pair of Krylov bases (V, W) for TWO-SIDED (Petrov-Galerkin) moment matching
- **`class KrylovROM(ss, V, s0, W=None)`** ★ — A Krylov-subspace moment-matching ROM -- one-sided (Galerkin) by default, or two-sided (Petrov-Galerkin, matching roughly twice as many moments from the same basis size) if built with from_MCK(..., two_sided=True).
  - `.two_sided` *(property)*
  - `KrylovROM.from_MCK(M, K, B, Cout, C=None, s0=0.0, k=10, two_sided=False)` — Build a KrylovROM directly from second-order mass/stiffness (and, optionally, damping) matrices plus an input map B and output map Cout -- the usual ...
  - `.transfer_function(s)` — H_r(s) = Cout_r (s*E_r - A_r)^-1 B_r, the reduced model's own transfer function, at one (possibly complex) frequency s.
  - `.frequency_response(omega_array)` — H_r(i*omega) swept over omega_array -- the reduced model's harmonic transfer function, directly comparable to a full-order model's own ...
  - `.is_stable()` — True iff every pole of the REDUCED model (generalized eigenvalues of (A_r, E_r)) has negative real part.

### `loewner.py`

Non-intrusive identification of modal parameters from frequency-response data (Loewner pencil). No M, C, K needed.

- **`class LoewnerROM(f, eta, Phi_tilde, omega_alpha, omega_beta)`** ★ — A non-intrusive modal model identified from complex frequency- response samples via the Loewner pencil (see module docstring).
  - `LoewnerROM.fit(omega_alpha, omega_beta, x_alpha, x_beta)` — Build a LoewnerROM from single-DOF complex response samples (Eqs. 21-27).
  - `.reconstruct_mode_shapes(X_beta_multi, omega_beta=None)` — Eq. (28): Phi = X_beta_multi @ Phi_tilde -- reconstruct multi-DOF mode shapes from response data sampled at MULTIPLE DOFs, but at the SAME omega_beta ...

### `membrane_expansion.py`

Membrane-basis estimation and expansion step for nonlinear structural ROMs of thin structures (Hollkamp-Gordon).

- **`class MembraneBasis()`** ★ — Eq. 13-16: estimates a membrane basis `T_m` from the SAME static training data already generated for a bending-only ICE-ROM fit (`AppliedLoadStrategy.generate(..., return_snapshots=True)`), then expands a modal bending-amplitude ...
  - `.fit(V, q_nl, W)` — Eq. 15: `T_m ~= [W - Phi_b @ P] @ Q^+`, generalized to any `n_modes` via `_monomial_indices(n_modes, 2)` for Eq. 16's own quadratic-combination ...
  - `.expand(p_hist)` — Eq. 16 + Eq. 11's expansion term: given a time (or sample) history of modal bending amplitudes, returns the physical membrane displacement history ...

### `metrics.py`

Method-independent error and comparison metrics between full and reduced results.

- **`modal_assurance_criterion(phi1, phi2)`** ★ — Modal Assurance Criterion (MAC) between two (possibly complex) mode-shape vectors.
- **`r_squared(y_true, y_pred)`** ★ — Coefficient of determination (Eq. 18 of He et al. 2023)

### `mode_correction.py`

Mode-acceleration and modal-truncation augmentation: cheap closed-form corrections that fix the static contribution lost by truncating modes.

- **`mode_acceleration_correction(K, basis, F)`** ★ — The static correction term
- **`mode_acceleration_response(basis, eta, q_cor)`** ★ — x_MA = V @ eta + q_cor -- the mode-acceleration-corrected response (eq.
- **`augmented_basis(basis, q_cor, M=None, tol=1e-10)`** ★ — Psi = [V, q_cor_orthogonalized] -- fold the static correction INTO the basis (eq.

### `neural_operator.py`

Neural operator mapping excitation to the response of nonlinear reduced dynamics (research-grade). Needs PyTorch.

- **`reduced_eom_residual_trajectory(t, q_hist, Lambda, C_r, F_nl_fn, F_ext_hist)`** ★ — Central-difference residual of the mass-normalized reduced EOM

### `nnm.py`

Nonlinear normal mode backbone curves by multi-harmonic-balance continuation (frequency versus amplitude).

- **`class HarmonicBalanceSystem(Lambda, C_r=None, n_harmonics=5, n_time_samples=None)`** ★ — Assembles the LINEAR (+ damping) harmonic-balance operator `A(omega)` for a given modal stiffness `Lambda`, modal damping `C_r`, and number of retained harmonics -- pure linear-algebra bookkeeping, no nonlinear force involved yet ...
  - `.reconstruct(Z)` — Z, shape (n_coeffs, n) -> q(theta_i), shape (n_time_samples, n).
  - `.project(q_time)` — q(theta_i), shape (n_time_samples, n) -> Z, shape (n_coeffs, n) (exact trigonometric-interpolation coefficients, truncated to n_harmonics -- the AFT ...
  - `.apply_A(Z, omega)` — Z, shape (n_coeffs, n) -> A(omega) @ Z, same shape.
  - `.apply_dA_domega(Z, omega)` — d(A(omega) @ Z)/domega, same shape as Z.
  - `.linear_matrix(omega)` — Dense (n_coeffs*n, n_coeffs*n) matrix form of apply_A(), for Newton-Jacobian assembly.
  - `.dA_domega_matrix(omega)` — Dense (n_coeffs*n, n_coeffs*n) matrix form of apply_dA_domega().
- **`solve_nnm_backbone(hb, force_model, q_amplitude_range, n_points=20, domain='q_nl', master_mode=0, omega0=None, jacobian='auto', tol=1e-08, max_iter=30)`** ★ — Traces an NNM backbone (frequency vs. amplitude) for the master mode `master_mode`, over `n_points` amplitudes linearly spaced across `q_amplitude_range = (a_min, a_max)`.
- **`solve_nnm_backbone_arclength(hb, force_model, a0, n_points=40, direction=1.0, ds=None, domain='q_nl', master_mode=0, omega0=None, jacobian='auto', tol=1e-08, max_iter=30, ...)`** ★ — Traces the SAME kind of NNM backbone as `solve_nnm_backbone` (frequency vs. master-mode amplitude, undamped-autonomous convention -- see module docstring), but via genuine PSEUDO- ARCLENGTH continuation (Keller, "Numerical ...

### `nonlinear_dynamics.py`

Generic reduced nonlinear time integration utilities.

- **`integrate_newmark_surrogate(Lambda, C_r, force_model, F_ext, q0, qdot0, dt, n_steps, domain='q_l', beta=0.25, gamma=0.5, correction='fixed_point', n_fixed_point=4, ...)`** ★ — Reduced nonlinear Newmark-beta time integration, decoupled from any one `ReducedForceModel` implementation.

### `nonlinear_rom.py`

Surrogate nonlinear structural ROMs: RBF surrogate of the nonlinear force (MFS-NLROM) and polynomial (ICE / Nash-form) models with their static and dynamic solvers. (Some functions use PyTorch, optional.)

- **`class ReducedForceModel()`** ★ — Protocol every nonlinear-force model in this family implements.
  - `.fit(q_samples, F_samples, **kwargs)` — Train the model from paired reduced-displacement / reduced nonlinear-force samples.
  - `.predict(q)` — Reduced displacement(s) -> predicted nonlinear force(s).
  - `.jacobian(q)` — dF_nl/dq at a single point q, shape (r, r).
- **`class MultiFidelitySurrogate(kernel='multiquadric', sigma=None, ridge=1e-10)`** ★ (extends `ReducedForceModel`) — RBF interpolant of the nonlinear modal force as a function of the LINEAR modal displacement `q_l` -- ported from the validated `mfs-nlrom-beam` skill's `RBFSurrogate`.
  - `.fit(q_l_samples, F_nl_samples)` — q_l_samples : (N, r) linear modal displacement training points.
  - `.predict(q_l)` — q_l : (r,) or (n, r).
  - `.jacobian(q_l)` — dF_nl/dq_l at a single point q_l, shape (r_out, r_in).
  - `.reconstruct_q_nl(q_l, Lambda)` — Eq. 11 (rearranged): q_nl = q_l - F_nl(q_l)/Lambda, using the trained surrogate's F_nl prediction -- no Newton-Raphson needed, the defining ...
- **`class NeuralSurrogate(n_modes, hidden_sizes=(32, 32), activation='tanh', lr=0.001, n_epochs=2000, weight_decay=0.0, seed=None, device='cpu')`** ★ (extends `ReducedForceModel`) — MLP surrogate of the nonlinear modal force as a function of the LINEAR modal displacement `q_l` -- same protocol and role as `MultiFidelitySurrogate` (Section 3 above), swapping the RBF interpolant for a small feedforward ...
  - `.fit(q_l_samples, F_nl_samples, n_epochs=None, verbose=False)` — q_l_samples, F_nl_samples : (N, n_modes) -- identical contract to `MultiFidelitySurrogate.fit()`.
  - `.predict(q_l)` — q_l : (r,) or (n, r).
  - `.jacobian(q_l)` — dF_nl/dq_l at a single point q_l, shape (r_out, r_in) -- same contract as `MultiFidelitySurrogate.jacobian()`, computed via `torch.autograd.grad` ...
- **`class PolynomialModalROM(n_modes)`** ★ (extends `ReducedForceModel`) — theta_r(q) = sum_{i<=j} B_r(i,j) q_i q_j + sum_{i<=j<=k} A_r(i,j,k) q_i q_j q_k (Nash-form, Eq. 45), fit by ordinary least squares per retained mode -- ported from the validated `mfs-nlrom-beam` skill's `IceRom`.
  - `.fit(q_nl_samples, F_nl_samples)` — q_nl_samples : (N, n_modes) NONLINEAR modal displacement training points (the actual projected full-order solution, not the linear estimate).
  - `.force(q)` — theta(q): direct polynomial evaluation, no Newton-Raphson.
  - `.jacobian(q)` — d(theta_out)/d(q_in) at a single point q, shape (n_modes, n_modes).
  - `.predict(q=None, F_ext=None, Lambda=None, q0=None, tol=1e-10, max_iter=50)` — Either evaluate theta(q) directly (q given, no Newton- Raphson -- just the polynomial force at that displacement), or solve Lambda@q + theta(q) = ...
- **`class TrainingStrategy()`** ★ — Pluggable training-DATA generation, not pluggable regression.
  - `.generate(*args, **kwargs)`
- **`class AppliedLoadStrategy(target_fracs, reference_scale, n_samples, rng)`** ★ (extends `TrainingStrategy`) — Eq. 46-47 / the ICE / Shi & Mei training-load convention: build a per-mode force-scale matrix via `sampling.modal_force_samples()` (OLHS-sampled per-mode target displacement fractions), assemble each training sample's full ...
  - `.generate(V, M_ff, basis_freqs_hz, mode_shape_peaks, fom_solver, return_snapshots=False)` — Parameters ---------- V : ndarray, shape (n_free, n_modes) Mass-normalized retained mode shapes (V.T @ M_ff @ V == I).
- **`class EnforcedDisplacementStrategy(q_range, n_samples, rng)`** ★ (extends `TrainingStrategy`) — STEP's own convention: prescribe q directly (a displacement PATTERN in the shape of one or a combination of retained modes), solve the full-order model in DISPLACEMENT CONTROL, read back the reaction force.
  - `.generate(V, M_ff, basis_freqs_hz, fom_solver)` — Parameters ---------- V : ndarray, shape (n_free, n_modes) M_ff : ndarray, shape (n_free, n_free) basis_freqs_hz : array_like, shape (n_modes,) ...
- **`class TrajectoryPilotedStrategy(n_target, rng)`** ★ (extends `TrainingStrategy`) — training-data generation piloted by states a dynamic simulation actually VISITS, rather than by independently-sampled per-mode static targets.
  - `.generate(V, basis_freqs_hz, trajectories, fom_solver)` — Parameters ---------- V : ndarray, shape (n_free, n_modes) Mass-normalized retained mode shapes.
- **`ICEROM(n_modes)`** ★ — Implicit Condensation and Expansion (Hollkamp & Gordon 2008): `PolynomialModalROM` fit with `AppliedLoadStrategy`-generated training data (the applied-load convention).
- **`ShiMeiROM(n_modes)`** ★ — Same regression as `ICEROM` -- kept as a separate, honestly- labeled name because the literature (Section 1's citations) distinguishes ICE and Shi & Mei by training-data convention details (e.g. membrane augmentation) this module ...
- **`EnforcedDisplacementROM(n_modes)`** ★ — STEP's own convention (Muravyov & Rizzi 2003): `PolynomialModalROM` fit with `EnforcedDisplacementStrategy`-generated training data (no load-scaling formula needed).

### `parameterized_latent_ode.py`

Parameterised latent ODE trained across a sweep of physical parameters (research-grade). Needs PyTorch.

- **`rk4_step(func, t, dt, y, *args)`** ★ — One step of the classical 4-stage, 4th-order Runge-Kutta method for `dy/dt = func(t, y, *args)`.
- **`integrate_rk4(func, y0, t, *args)`** ★ — Integrate `dy/dt = func(t, y, *args)` from `y0` over the time vector `t` (not necessarily uniformly spaced -- each step uses its own local `dt`), via repeated `rk4_step()` calls.
- **`class CurriculumSchedule(n_steps, n_folds=10)`** ★ — Progressively extends the trained integration horizon as the training loss drops below a tolerance -- the `L-NeuralODE` reference's own stabilization technique for a fully black-box `dz/dt=f(...)`.
  - `.is_full_horizon` *(property)*
  - `.advance()` — Unconditionally move to the next stage (a no-op once `is_full_horizon` is already True).
  - `.maybe_advance(loss, tol)` — Advance one stage if `loss < tol` and not already at full horizon.

### `passivity.py`

Passivity (positive-realness) diagnostics for a force-in / velocity-out structural port.

- **`velocity_transfer_function(H_disp, omega_array)`** ★ — H_vel(i*omega) = i*omega * H_disp(i*omega) -- the mobility/ admittance transfer function for a collocated force-in/ displacement-out SISO port, derived from H_disp WITHOUT any new state-space "velocity output" construction, since ...
- **`passivity_margin(H_vel)`** ★ — Re[H_vel(i*omega)] at each swept frequency -- the standard frequency-domain positive-realness/passivity margin for a one-port (SISO, collocated force-in/velocity-out) system: a transfer function is positive real iff this margin ...
- **`is_passive(H_vel, tol=0.0)`** ★ — True iff passivity_margin(H_vel) >= -tol everywhere swept.

### `pod.py`

Proper orthogonal decomposition: `PodBasis` from a snapshot matrix (optionally mass-weighted), energy capture, projection/expansion, and multi-field variants.

- **`class PodBasis()`** ★ — A POD basis extracted from a snapshot matrix, with either a Euclidean or mass-weighted inner product.
  - `.fit(snapshots, n_modes=None, energy_threshold=None, M=None)` — Extract the POD basis from a snapshot matrix.
  - `.energy_captured()` — Fraction of total snapshot 'energy' (sum of squared singular values) captured by the retained n_modes -- a diagnostic for whether the truncation was ...
  - `.orthonormality_error()` — max\|V^T V - I\| (standard) or max\|V^T M V - I\| (mass-weighted) -- should be at machine precision for any basis this class produced itself; a nonzero ...
  - `.project(x)` — Full-order state(s) -> reduced coordinates.
  - `.expand(q)` — Reduced coordinates -> full-order approximation.
  - `.reconstruction_error(snapshots, relative=True)` — \|\|X - V @ project(X)\|\| (Frobenius), optionally normalized by \|\|X\|\| -- the direct, model-agnostic measure of how much of the ORIGINAL snapshot set ...
- **`assemble_field_weight_matrix(n_dof, field_slices, field_grams, scale=1.0)`** ★ — Assemble a full (n_dof, n_dof) block-diagonal weighting/Gram matrix `W` for MultiFieldPOD out of one small per-field Gram matrix per field, scattered into that field's own DOF positions.
- **`trapezoidal_field_gram(s, scale=1.0)`** ★ — A simple, FE-shape-function-free example of a per-field Gram matrix: the LUMPED (diagonal) composite-trapezoidal-rule integration weights for a field sampled at 1-D coordinates `s` -- i.e. the diagonal approximation to ``integral ...
- **`class MultiFieldPOD(field_slices, W, sign_reference_field=0)`** ★ — Multi-field POD diagnostics on top of `PodBasis`, for snapshots whose state vector interleaves several physically DISTINCT fields (Georgiou 2005, "Advanced Proper Orthogonal Decomposition Tools...
  - `.fit(snapshots, n_modes=None, energy_threshold=None, check_block_diagonal=True)` — Fit the underlying mass-weighted PodBasis (M=W) on RAW (not mean-subtracted) snapshots, then compute every multi-field diagnostic on top of it.
  - `.reconstruct_field(field_index, Q=None)` — Field-separated reconstruction: the time history of just ONE field, reconstructed from POD amplitudes -- e.g. pull out just the ...

### `random_vibration.py`

Random-vibration (PSD) response computed on a frequency-domain ROM.

- **`psd_response(rom_freq, freqs_hz, psd_input, F0_pattern, output_dofs)`** ★ — Random-vibration (PSD) response of a FrequencyROM, at one or several output DOFs.
- **`band_limited_gaussian_time_history(n_samples, dt, f_max, rms, rng=None)`** ★ — a band-limited (0 to `f_max` Hz), Gaussian-random, time-domain forcing signal -- the driving input Hollkamp & Gordon (2008)'s own dynamic-response reproduction needs (Sec. 4, a band-limited random pressure driving both the full ...

### `reduced_basis_operator.py`

Encoder/decoder pair that maps fields sampled at any points to a finite-dimensional reduced-basis representation (research-grade). Needs PyTorch.

- **`class RegularizedProjectionEncoder()`** ★ — RONOM's `E := (E_phi o M o P_V^lambda)` encoder / optimal- recovery decoder pair (Section 2.3/2.5 of the paper), specialized to a basis built from an EXISTING discrete POD basis (any `pod.PodBasis.V` column, or a bare ndarray) ...
  - `.fit(coords, V, kernel='thin_plate_spline', smoothing=1e-08)` — Build the continuous basis representation.
  - `.encode(query_coords, values, reg=1e-06)` — Point samples `{x_i, f_i}` -> latent code `z` (RONOM Eq. 2.5's discretized regularized projection): solves
  - `.decode(z, query_coords)` — Latent code `z` -> field value(s) at `query_coords` (RONOM's own "optimal recovery" decoder, Eq. 2.5's dual side) -- again, `query_coords` may be ANY ...
  - `.reconstruct(query_coords, values, reg=1e-06)` — encode() then immediately decode() at the SAME points -- the direct "how well does this basis explain this data" round-trip check, returned alongside ...

### `sampling.py`

Optimal Latin Hypercube design of experiments for choosing training parameters.

- **`optimal_lhs(n_samples, n_dims, criterion='maximin', n_iter=200, rng=None)`** ★ — General-purpose Optimal Latin Hypercube Sampling design in [0, 1]^n_dims.
- **`modal_force_samples(basis_freqs_hz, mode_shape_peaks, target_fracs, reference_scale, n_samples, rng=None)`** ★ — Eq. 16-17 / 46-47 of the reference paper, generalized: OLHS- samples a per-mode target peak displacement within a fraction range of a caller-supplied reference length scale, then converts each sampled displacement target into the ...

### `scm.py`

Successive Constraint Method style certified lower bound on the smallest singular value of the dynamic stiffness, turning a residual into a rigorous error bound.

- **`class SingularValueLowerBound(components, theta_func)`** ★ — A certified, Lipschitz-perturbation-based lower bound on sigma_min(A(omega)) for an affine A(omega) = sum_q theta_q(omega) A_q.
  - `SingularValueLowerBound.from_affine(affine)` — Build directly from an already-constructed affine.AffineDecomposition (e.g. rom.affine on a FrequencyROM) -- reuses its components/theta_func rather ...
  - `.add_reference(omega)` — Full-order-compute the TRUE sigma_min(A(omega)) at this omega (one O(n_dof^3) SVD) and store it as a reference point for lower_bound()/upper_bound() ...
  - `.lower_bound(omega)` — max_j [ sigma_min(A(omega_j)) - sum_q \|theta_q(omega) - theta_q(omega_j)\| * \|\|A_q\|\|_2 ], clipped at 0 (sigma_min is never negative, so 0 is always a ...
  - `.upper_bound(omega)` — min_j [ sigma_min(A(omega_j)) + sum_q \|theta_q(omega) - theta_q(omega_j)\| * \|\|A_q\|\|_2 ] -- the mirror-image bound from the SAME Lipschitz fact, used ...
  - `.greedy_train(candidate_omegas, tol=0.05, max_references=15, seed_omega=None)` — Adaptively add reference points where the CERTIFIED gap (upper_bound - lower_bound) is currently largest, i.e. where the bound is least informative ...
- **`certified_error_bound(rom, scm_bound, omega, F)`** ★ — \|\|x_true(omega) - x_ROM(omega)\|\| <= residual_norm(omega) / sigma_min(A(omega)), evaluated with scm_bound's RIGOROUS lower bound on sigma_min(A(omega)) in place of the true (expensive) value -- the one genuinely CERTIFIED bound in ...

### `scm_lp.py`

Classical SCM with an online linear program for the certified lower bound.

- **`class LPSingularValueLowerBound(components, theta_func)`** ★ — A certified lower bound on sigma_min(A(omega)) for an affine A(omega) = sum_q theta_q(omega) A_q, computed via the genuine classical Successive Constraint Method's online linear program.
  - `LPSingularValueLowerBound.from_affine(affine)` — Build directly from an already-constructed affine.AffineDecomposition (e.g. rom.affine on a FrequencyROM) -- reuses its components/theta_func rather ...
  - `.add_reference(omega)` — Full-order-compute the TRUE lambda_min(A(omega)^H A(omega)) = sigma_min(A(omega))^2 at this omega and store it (plus c(omega)) as a reference/control ...
  - `.lower_bound(omega)` — sqrt(max(0, LP optimum)), where the LP minimizes sum_pq c_pq(omega) y_pq subject to the box bounds (valid for ANY unit vector) and, for every stored ...
  - `.upper_bound(omega)` — A companion upper bound, via a closed-form (NOT LP-based) box-only maximization: sum_pq c_pq(omega) * (y_max_pq if c_pq(omega) >= 0 else y_min_pq).
  - `.greedy_train(candidate_omegas, tol=0.05, max_references=15, seed_omega=None)` — Adaptively add reference points where the certified gap (upper_bound - lower_bound) is currently largest -- see ...
- **`certified_error_bound(rom, scm_bound, omega, F)`** ★ — \|\|x_true(omega) - x_ROM(omega)\|\| <= residual_norm(omega) / sigma_min(A(omega)), evaluated with an LPSingularValueLowerBound's RIGOROUS lower bound on sigma_min(A(omega)) -- the LP-based analogue of scm.certified_error_bound(), ...

### `screening.py`

Cross-ROM stability screening for Loewner identification: keep only modes that recur across many random ROMs.

- **`class ScreenedMode(f: float, eta: float, s_total: float, occurrence_count: int)`** ★ (extends `NamedTuple`) — One physical mode surviving cross-ROM stability screening.
- **`screen_physical_modes(freq_pool, x_pool, fmin, fmax, rng, n_roms=20, n_interp=9, eps_f=0.005, eps_eta=0.1, tau_s=0.95)`** ★ — Eqs. (30)-(32): build `n_roms` LoewnerROMs from random interpolation-frequency subsets of `freq_pool`, then use the cross-ROM stability criterion to separate genuine physical modes from spurious numerical ones.

### `soar.py`

Second-order Arnoldi (SOAR) moment-matching ROM that keeps the second-order (M, C, K) structure.

- **`soar_basis(M, K, B, k, C=None, s0=0.0, tol=1e-12, max_generations=None)`** ★ — Orthonormal (n_dof, k') basis (k' <= k) of the order-k second-order Krylov subspace spanned by r_1=K0^-1 B, r_j = A1 r_{j-1} + A2 r_{j-2} -- see module docstring for the derivation of A1, A2, K0, and for why the raw ...
- **`class SOARROM(M, C, K, B, Cout, V, s0)`** ★ — A SOAR second-order-structure-preserving Krylov moment-matching ROM.
  - `SOARROM.from_MCK(M, K, B, Cout, C=None, s0=0.0, k=10, tol=1e-12)` — Build a SOARROM directly from second-order mass/stiffness (and, optionally, damping) matrices plus an input map B and output map Cout -- mirrors ...
  - `.transfer_function(s)` — H_r(s) = Cout_r (s^2*M_r + s*C_r + K_r)^-1 B_r, the reduced model's own SECOND-ORDER transfer function, at one (possibly complex) frequency s.
  - `.frequency_response(omega_array)` — H_r(i*omega) swept over omega_array -- directly comparable to a full-order model's own solve_harmonic()/solve_frequency_sweep() output at the SAME ...
  - `.is_stable()` — True iff every pole of the REDUCED quadratic eigenvalue problem det(s^2*M_r + s*C_r + K_r)=0 has negative real part, found by linearizing ONLY the ...

### `state_space.py`

Convert a second-order structural system (M, C, K, B, C_out) to first-order state-space form.

- **`class StateSpaceSystem(A, B, Cout, E=None, n_dof=None)`** ★ — A first-order state-space realization of a second-order structural model.
  - `.n_state` *(property)*
  - `.n_in` *(property)*
  - `.n_out` *(property)*
  - `.form` *(property)*
- **`to_state_space(M, K, C=None, B=None, Cout=None, form='E')`** ★ — Build a StateSpaceSystem from second-order (M, K[, C]) matrices and (optional) input/output maps.

### `torch_linalg.py`

Torch-native dense linear algebra helpers (Lyapunov, Sylvester, Gramian square root, balancing) used by the state-space MOR methods. Needs PyTorch.

- **`torch_solve_continuous_lyapunov(A, Q, device='cpu', dtype=None)`** — Torch-native solve of A X + X A^T = Q for X -- the SAME equation scipy.linalg.solve_continuous_lyapunov(A, Q) solves, same sign convention, so callers can swap backends with no other change.
- **`torch_solve_sylvester(A, B, Q, device='cpu', dtype=None)`** — Torch-native solve of A X + X B = Q for X -- the SAME equation scipy.linalg.solve_sylvester(A, B, Q) solves, including its generally-RECTANGULAR case (A is (m, m), B is (p, p), X and Q are (m, p)) -- same Kronecker-vectorization ...
- **`torch_gramian_square_root(P, device='cpu', dtype=None, jitter_tries=(0.0, 1e-12, 1e-09, 1e-06))`** — Torch-native counterpart of balanced_truncation._gramian_square_ root() -- a factor L with P = L @ L.T, via Cholesky first (with the SAME small jitter ladder for numerical-noise-induced non-positive- definiteness) then an ...
- **`torch_balance_from_gramians(P, Q, device='cpu', dtype=None)`** — Torch-native counterpart of balanced_truncation._balance_from_ gramians() -- the shared square-root balancing step (Gramian square roots, then an SVD of Lq^T Lp), same formulas, same return convention (sigma descending, T, Tinv) ...

### `wave13_validation.py`

Validation harness tying together the neural-operator, reduced-basis-operator and latent-ODE research modules. Needs PyTorch.

- **`cross_check_reduced_dynamics(fixture, n_train=40, dt=2e-05, n_steps=2500, F_ext_amplitude=None, seed=0)`** ★ — Line of evidence 1 (see module docstring).
- **`cross_check_basis_generalization(fixture_train, fixture_validate, n_train=30, n_pod_modes=3, n_validate_samples=3, seed_train=0, seed_validate=1)`** ★ — Line of evidence 2 (see module docstring) -- item 118's own decisive check, re-run as part of this harness so the report from `run_cross_configuration_validation()` below covers all three items together.
- **`run_cross_configuration_validation(fixture_train, fixture_validate, **kwargs)`** ★ — Convenience wrapper running both lines of evidence and returning one combined report dict.


---

# Part 4 — Conventions and gotchas

- **Units:** the packages are unit-agnostic. Use one consistent system (SI is used throughout the examples).
- **Degrees of freedom:** `FESystem.n_dof` counts all nodes; `fix_dofs(nodes, [dof indices])` constrains per node; `free_dofs` lists what remains. ROM functions expect plain arrays, usually the free-DOF blocks of K, M and F.
- **Load first or constrain first:** `add_nodal_force` accumulates into `sys.F`. Reset `sys.F[:] = 0.0` between load cases (E5 does this).
- **Backends:** SciPy/NumPy is the default everywhere. `backend="torch"` and `device="cuda"` are opt-in and need PyTorch; asking for torch without it raises `ImportError` immediately. See the "Multi-core, CPU and GPU" section of `USER_GUIDE.md`.
- **Optional dependencies:** PyTorch (autograd, GPU, neural/research modules, topology optimisation), `meshio` (mesh file I/O), `matplotlib` (plotting). A plain `import fea_engine` / `import rom_engine` needs none of them.
- **Aliases and constants at the top level:** `grading_plan` is `grading.plan`, `grading_graded_partition` is `grading.graded_partition`, `certified_error_bound_lp` is `scm_lp.certified_error_bound` (renamed on import to avoid a clash with `scm.certified_error_bound`). Registries and defaults exported from the top level: `ELEMENT_REGISTRY`, `CONSTITUTIVE_REGISTRY`, `ROTATIONAL_DOF_ELEMENTS`, `DEFAULT_GROWTH_RATIO` (`fea_engine`) and `KERNEL_REGISTRY` (`rom_engine`). Submodules (`solver`, `mesh`, `material`, ...) are importable by name too.
- **Private names:** anything starting with an underscore is internal and not listed. Docstrings in the source are the authority for edge cases.
- **Research-grade modules:** the neural-operator, latent-ODE, reduced-basis-operator, ensemble-UQ and differentiable-correction modules are prototypes; treat their results as experimental and validate against a full-order solve.
- **Validation habit:** compare any reduced result with one full-order solve before trusting it (E5, E6 and E7 show how).
