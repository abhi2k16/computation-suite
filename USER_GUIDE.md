<div align="center">

# computation-suite user guide: `fea_engine` and `rom_engine`

**Getting started, concepts, worked examples, multi-core and GPU usage, performance**

![fea_engine](https://img.shields.io/badge/fea__engine-1.0.0-2563eb?style=for-the-badge)
![rom_engine](https://img.shields.io/badge/rom__engine-0.1.0-7c3aed?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.9+-3776ab?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-1.22+-013243?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-1.8+-0054a6?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-optional-ee4c2c?style=for-the-badge)
![license](https://img.shields.io/badge/license-MIT-16a34a?style=for-the-badge)

</div>

## 🧭 Contents

| | |
|---|---|
| 🚀 [Get Started](#s-2) | 📘 [API Reference](#s-32) |
| 💡 [Concepts](#s-6) | ⚡ [Multi-core, CPU and GPU (PyTorch) usage](#s-35) |
| 📖 [User Guide](#s-11) | 📈 [Performance](#s-42) |
| 🖼️ [Example Gallery](#s-26) | 🧭 [Where to go deeper](#s-43) |
| 🔬 [Advanced capabilities (research-grade)](#s-29) |  |

---


A fast, from-scratch, NumPy/SciPy-native finite element and reduced-order
modeling toolkit — with an entirely optional PyTorch layer for GPU
acceleration and automatic differentiation. Everything below was run
against the actual packages in this repository while writing this guide,
not written from memory — every number quoted (deflections, frequencies,
speedups, error percentages) came out of a real Python session, not an
illustrative estimate. Structured the way a documentation site would lay
it out: Get Started, Concepts, User Guide, Example Gallery, API
Reference, Multi-core/CPU/GPU usage, Performance, and pointers for going deeper.

<a id="s-1"></a>

## 💪 Core strengths

- **Two packages, one clean boundary.** `fea_engine` is a full-order
  finite element solver (statics, dynamics, linear and nonlinear
  mechanics). `rom_engine` is a reduced-order-modeling toolkit that
  compresses and accelerates it. `rom_engine`'s library code never
  imports `fea_engine` — every reduction routine takes plain NumPy
  arrays or a plain Python callable, so it works equally well against
  matrices from any other FE source.
- **NumPy/SciPy core, PyTorch opt-in.** A bare `import fea_engine` or
  `import rom_engine` never requires `torch` or `matplotlib`. (Gmsh-backed
  geometry/visualization support has been removed from `fea_engine`
  entirely -- see the note under Installation below -- so `gmsh` is no
  longer a relevant optional dependency at all.) Every GPU/autograd-capable
  code path is a `backend="torch"` (or
  `method="autograd"`) switch layered on top of an already-complete
  NumPy/SciPy implementation — choosing torch never removes or hides
  the default path.
- **Comprehensive element and analysis coverage.** Truss, beam, plate,
  shell, and 3-D solid elements (linear and quadratic); static, modal,
  transient (implicit/explicit), harmonic, and random-vibration solves;
  geometric/material/contact nonlinearity with four different nonlinear
  drivers (monotonic, displacement-control, arc-length, transient).
- **A genuinely broad reduced-order-modeling toolbox.** Linear
  intrusive projection (POD/Galerkin/affine/frequency), classical
  systems-and-control MOR (Krylov, balanced truncation), non-intrusive
  identification straight from frequency-response data (no matrices
  needed), and nonlinear surrogate ROMs (RBF, polynomial, and neural
  network), all sharing one offline/online design.
- **Debugging-friendly, not a black box.** No JIT, no form compiler, no
  DSL — everything is plain, inspectable Python built on `numpy`/`scipy`
  arrays, with a small, explicit set of opt-in switches (`backend=`,
  `device=`, `method=`) for the parts that reach for `torch`.

---

<a id="s-2"></a>

# 🚀 Get Started

<a id="s-3"></a>

## ⚙️ Installation

```bash
cd fea_engine
pip install -e .                 # core: numpy>=1.22, scipy>=1.8 only
pip install -e ".[plot]"         # + matplotlib, for mesh.py's plot_*
pip install -e ".[dev]"          # + pytest/pytest-cov, to run the test suite
```

> [!NOTE]
> **Note:** Gmsh-backed geometry/visualization support (`geometry/gmsh_engine.py`,
> `visualization/gmsh_plot.py`, formerly installed via a `.[gmsh]` extra) has
> been removed from `fea_engine` entirely -- it required a system libGLU
> library that could not be reliably provided, and no verified example or
> result in this package depended on it. `mesh.py`'s structured mesh
> generation and matplotlib `plot_*` functions remain the supported path.

```bash
cd rom_engine
pip install -e .                 # core: numpy, scipy only
pip install -e ".[fea]"          # + fea_engine, needed to run tests/examples (not the library itself)
pip install -e ".[dev]"          # + pytest, to run the test suite
```

`matplotlib` is **not** imported by `fea_engine/__init__.py` — a plain
`import fea_engine` never requires it; it's pulled in explicitly only
when you call a `plot_*` function, so the core package stays light.
(Gmsh was previously in the same category, pulled in only via
`from fea_engine.geometry import gmsh_engine`; that module has since
been removed entirely, see the Installation note above.) The same
is true of `torch` (see "Differentiability & GPU Backends" below) — it
has no generic `[torch]` extra on purpose, because the right wheel
depends on your hardware:

```bash
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU (CUDA 12.6 build shown -- match this to your driver/GPU):
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Two things worth knowing before you pick a CUDA build: (1) PyTorch's
**default** PyPI wheel (a bare `pip install torch`, no `--index-url`) is
CUDA-linked and will fail to `import torch` on a machine with no NVIDIA
driver/CUDA runtime — use the CPU index above for a CPU-only machine.
(2) PyTorch 2.8.0+'s CUDA builds dropped support for Maxwell/Pascal-
architecture GPUs (compute capability 5.0/6.0, e.g. the GTX 9-series/
10-series) — a Pascal card needs PyTorch ≤2.7.x with a matching `cuXXX`
build (2.7.0/`cu126` is the last line confirmed to still support
Pascal). On Windows with an Anaconda/MKL-based install, you may also hit
`OMP: Error #15` / a hard crash the first time NumPy and a pip-installed
PyTorch both load their own OpenMP runtime in the same process — set
`KMP_DUPLICATE_LIB_OK=TRUE` as a quick workaround, or use a dedicated
fresh environment.

<a id="s-4"></a>

## ⚡ Quickstart

Every `fea_engine` problem is the same four objects, wired together:

| Object | Module | Role |
|---|---|---|
| `Material` / `Section` / a `D_xxx(mat)` constitutive matrix | `material.py` | What the structure is made of |
| `Mesh` (or `MultiBlockMesh`) | `mesh.py` / `geometry.py` | Where the nodes and elements are |
| An `Element` subclass instance (e.g. `Quad4PlaneStress()`) | `elements/` | How one element converts displacement to force |
| `FESystem` | `solver.py` | Assembles `K`/`M`/`C`/`F`, applies loads/BCs, solves |

```python
import numpy as np
from fea_engine import Material, D_plane_stress, Quad4PlaneStress, FESystem
from fea_engine.mesh import rectangle_mesh

mat = Material(E=2.1e11, nu=0.3, rho=7850.0)          # steel, SI units
mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=24, ny=12)     # 0.4 x 0.2 m plate, 24x12 elements
elem = Quad4PlaneStress()
sys = FESystem(mesh, elem, thickness=0.02)               # 20 mm thick

sys.assemble_stiffness(D_plane_stress(mat), thickness=0.02)

tip = mesh.nodes_on_line(axis=0, value=0.4)               # nodes at x = 0.4 (the free edge)
sys.add_nodal_force(tip, dof_index=1, total_force=-20000.0)   # 20 kN downward, split across those nodes
sys.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])   # clamp x = 0 edge (both DOFs)

U = sys.solve_static()
```

Run exactly as written above, this produces a 325-node / 650-DOF model
with a mean tip deflection of **`U_y ≈ -1.797e-4 m`** (max
`|U| ≈ 1.814e-4 m`) — confirmed by running it, not estimated.

That five-step pattern — material → mesh → element → `FESystem` →
assemble/load/BC/solve — is the whole package. Everything else is a
variation on it: a different mesh generator, a different element, a
different `solve_*` call.

> [!TIP]
> **Shortcut for the common cases:** `geometry.py` collapses the
> mesh+element+`FESystem` setup into one call:

```python
from fea_engine.geometry import build_system
sys, mesh, elem = build_system(dim=2, Lx=0.4, Ly=0.2, nx=24, ny=12)   # plane_stress rectangle, by default
```

`dim` picks 1-D/2-D/3-D; `shape=` and `physics=` pick a different
default geometry/element on that dimension (e.g. `dim=2, physics='plate'`
for a Mindlin plate on the same rectangle, or `dim=1, physics='truss'`
for a truss instead of a beam on the same line). Unknown combinations
raise a `ValueError` listing what's actually registered.

**And for `rom_engine`**, the pattern in every example is always: build
`K`/`M`/`F` with `fea_engine`, then hand those plain arrays to
`rom_engine`.

```python
from rom_engine import PodBasis, GalerkinROM

basis = PodBasis().fit(snapshot_matrix, n_modes=6, M=sys.M)   # mass-weighted
rom = GalerkinROM(basis).reduce_system(sys.K, M=sys.M, F=sys.F)
x_full, q_reduced = rom.solve_static()
```

The full worked version of this, with real snapshots and real numbers,
is in "ROM: Linear Workflow" under the User Guide below.

<a id="s-5"></a>

## ✅ Verify Install

```bash
python -c "import fea_engine; print(fea_engine.__version__)"
python examples/main.py     # runs 8 problems end to end against closed-form references
```

```bash
python -c "import rom_engine; print(rom_engine.__version__)"
pip install -e ".[fea,dev]" && pytest tests/ -q      # 187 tests, validated against real fea_engine models
```

Minimal install + smoke test for both packages together:

```bash
cd fea_engine  && pip install -e ".[plot,dev]"
cd ../rom_engine && pip install -e ".[fea,dev]"

python -c "import fea_engine, rom_engine; print(fea_engine.__version__, rom_engine.__version__)"
cd ../fea_engine && python examples/main.py
cd ../rom_engine && python examples/two_region_beam_rom.py
```

---

<a id="s-6"></a>

# 💡 Concepts

<a id="s-7"></a>

## `fea_engine`: what it is, and when to reach for it

`fea_engine` is a pure-Python (NumPy/SciPy) finite element package. It
covers:

- **Statics**: linear elastic, and geometrically/materially/contact
  nonlinear.
- **Dynamics**: implicit and explicit time integration, modal
  superposition, harmonic (frequency) response, random vibration (PSD).
- **Elements**: truss, 2-D beam (linear and large-rotation
  corotational), 3-D beam/frame, plane stress/strain (linear and
  quadratic), Mindlin plate, MITC4 shell, 3-D solid brick/tet (linear
  and quadratic), contact/gap elements.
- **Geometry**: a zero-dependency structured-mesh path (rectangles,
  boxes, mapped holes). (A previously-available optional Gmsh-driven
  path for genuinely curved/unstructured geometry and CAD import --
  STEP/IGES/BREP -- has been removed entirely; see `fea_engine/README.md`'s
  "Removed: Gmsh support" section.)

The whole package is organized around one rule: every axis of variation
— a new material, a new element, a new geometry, a new load type, a new
solve strategy — lives behind a registry entry or a subclass, so
`solver.py` never has to know whether it's solving a truss or a 3-D
solid. That's also why it's a good base for `rom_engine` to sit on top
of: `FESystem` always exposes the same `K`, `M`, `C`, `F`
matrices/vectors regardless of the physics underneath.

<a id="s-8"></a>

## `rom_engine`: what it is, and the one design decision that matters most

`rom_engine` compresses and accelerates full-order structural models.
Its core modules (`pod.py`, `galerkin.py`, `affine.py`, `frequency.py`)
take **plain NumPy arrays** — `K`, `M`, `C`, `F`, snapshot matrices — not
`fea_engine` objects. `rom_engine`'s library code never imports
`fea_engine`. That decoupling is deliberate: `rom_engine` works with
matrices from any FE source, and `fea_engine` is simply the most
convenient way to *get* those matrices in this repository (it's the
package used to build every validation fixture and example in
`rom_engine`'s own test suite).

<a id="s-9"></a>

## 🔀 The offline/online split

Every reduction technique in `rom_engine` is organized around the same
idea: do the expensive work (a handful of full-order solves, building a
reduced basis, projecting the full-order matrices onto it) **once,
offline**; then answer every subsequent query — a new load, a new
frequency, a new parameter value — **cheaply, online**, using only the
small reduced system, never touching the full-order model again. The
221.8x speedup measured in "ROM: Linear Workflow" below is what that
split buys you in practice.

<a id="s-10"></a>

## 🔥 The torch switch: NumPy-core, PyTorch-optional

Both packages treat `torch` as strictly opt-in. The rule that shows up
everywhere it's used: any function that needs only ordinary linear
algebra is written once and works identically with or without torch
installed; only the specific pieces that need GPU dispatch or automatic
differentiation are placed behind a `backend=`/`method=`/`device=`
argument, and calling one of those without torch installed raises a
clear error immediately (usually at construction, before any expensive
work has been done), rather than a confusing failure mid-computation.
Both packages check for torch with a broad `except Exception` at import
time, not the narrower `except ImportError` you might expect — a
GPU-linked build of torch that doesn't match the local driver can fail
partway through import with a different kind of error entirely, so the
check is written to catch any way the import can go wrong.

You'll see this switch in four places in this guide: `FESystem`'s
`backend="torch"` static solve and `device="cpu"`/`"cuda"` choice;
`solve_nonlinear_static(..., method="autograd")` for exact automatic-
differentiation tangents; `rom_engine`'s `NeuralSurrogate` as a drop-in
alternative to the RBF/polynomial surrogates; and the differentiable-
correction layer in both packages.

---

<a id="s-11"></a>

# 📖 User Guide

<a id="s-12"></a>

## 🕸️ Meshes: three front ends, one contract

Every generator returns a `Mesh(nodes, elements, dim)` (or
`MultiBlockMesh` if more than one element topology is present) — plain
NumPy arrays. `FESystem`, `solve_static()`, `mesh.check_quality()`, all
of it, work identically no matter which front end produced the mesh.

**`mesh.py` — zero-dependency, structured/mapped:**

```python
from fea_engine.mesh import line_mesh, rectangle_mesh, box_mesh, rectangle_with_hole_mesh_quarter

m1 = line_mesh(L=1.0, n=20)                          # 1-D, for beams
m2 = rectangle_mesh(Lx=0.4, Ly=0.2, nx=24, ny=12)     # 2-D quad4 rectangle
m3 = box_mesh(Lx=1.0, Ly=0.1, Lz=0.1, nx=20, ny=4, nz=4)   # 3-D hex8 block
m4 = rectangle_with_hole_mesh_quarter(a=2.0, b=1.0, R=0.3, nr=8, ntheta=12)  # mapped hole approximation
```

> [!WARNING]
> **Removed:** this section previously also documented a second,
> Gmsh-driven front end (`geometry.gmsh_engine`, for genuinely
> unstructured/curved meshes and `generate_from_step()` CAD import from
> STEP/IGES/BREP files). That front end has been removed from the package
> entirely -- it required `pip install gmsh` plus a system libGLU library
> that could not be reliably provided, and no verified example or result
> in this package depended on it. See `fea_engine/README.md`'s "Removed:
> Gmsh support" section for the fuller note. The structured `mesh.py`
> front end above is now the only supported meshing path.

<a id="s-13"></a>

## 🧱 Elements and quadrature: choosing an element

| Element | Module | Dim | DOF/node | Use for |
|---|---|---|---|---|
| `TrussTL2D`, `TrussPlastic2D` | `elements.trusses` | 1-D→2-D | 2 | Axial-only members, large displacement / plasticity |
| `Beam2DEulerBernoulli`, `Beam2DCorotational` | `elements.beams` | 1-D→2-D | 3 | Frames/beams, linear or large-rotation |
| `Beam3DEulerBernoulli` | `elements.beams3d` | 1-D→3-D | 6 | 3-D space frames |
| `Quad4PlaneStress`, `Quad8PlaneStress` | `elements.solids` | 2-D | 2 | Thin in-plane loaded plates/brackets |
| `Quad4MindlinPlate` | `elements.plates` | 2-D | 3 | Out-of-plane bending plates |
| `Shell4MITC` | `elements.shells` | 2-D→3-D | 6 | Combined membrane + bending (curved shells) |
| `Tet4Solid3D`, `Tet10Solid3D`, `Hex8Solid3D`, `Hex20Solid3D` | `elements.solids` | 3-D | 3 | General 3-D solids |
| `Hex8PlasticJ2`, `Tet4NeoHookean` | `elements.nonlinear_solids` | 3-D | 3 | Elastoplastic / hyperelastic solids |
| `GapContactPenalty`, `GapContactCurvedFriction` | `elements.contact` | — | — | Contact/boundary constraints |

Every registered element is also available by string name via
`fea_engine.elements.ELEMENT_REGISTRY["quad8_plane_stress"]`, etc. —
useful if you're picking an element type at runtime rather than
importing the class directly.

<a id="s-14"></a>

## 📌 Boundary conditions and loads

Boundary conditions and loads attach directly to the `FESystem`/`Mesh`
pair, not to a separate object: `sys.fix_dofs(node_ids, dof_indices)`
constrains degrees of freedom, and `sys.add_nodal_force(node_ids,
dof_index, total_force)` distributes a total force across a set of
nodes (both used already in the Quickstart above). Time-varying and
distributed load types (`TimeHistoryLoad`, `HarmonicLoad`, `PSDLoad`)
are covered in "Time Integration" and "Solvers" below, since which load
type you use is tied to which `solve_*` call you're driving.

<a id="s-15"></a>

## 🧮 Solvers

All of the recipes below assume `sys = FESystem(mesh, elem,
thickness=...)` and `sys.assemble_stiffness(D)` have already run.

**Static:**
```python
U = sys.solve_static()
```

**Choosing a solve backend — SciPy (default) vs PyTorch, CPU vs GPU:**

`solve_static()` doesn't hardcode a linear-algebra engine. `FESystem`
takes two extra, independent constructor arguments that pick one:
`backend="scipy"` (default) or `backend="torch"`, and, only when using
torch, `device="cpu"` (default) or `device="cuda"`. Both engines are
complete, separately validated implementations that stay available side
by side — choosing one never removes or hides the other, and the choice
is made once, explicitly, when you build the `FESystem`:

```python
# default -- SciPy, unchanged from every earlier example in this guide
sys = FESystem(mesh, elem, thickness=0.02)                          # backend="scipy" implicit
U = sys.solve_static()

# PyTorch, on CPU
sys = FESystem(mesh, elem, thickness=0.02, backend="torch")         # device="cpu" implicit
U = sys.solve_static()

# PyTorch, on GPU -- the ONLY thing that changes is device=
sys = FESystem(mesh, elem, thickness=0.02, backend="torch", device="cuda")
U = sys.solve_static()
```

That's the whole CPU/GPU switch: `device="cpu"` or `device="cuda"` on
the same `backend="torch"` construction call — nothing else about the
model (mesh, element, material, loads, BCs) changes. `device="cuda"`
needs a CUDA-enabled PyTorch install and an actual NVIDIA GPU on the
machine running it (see Installation's notes, including the Pascal/
GTX-10-series version pin); on a machine without a working CUDA runtime,
`device="cuda"` fails wherever `torch` itself would.

`sparse=` combines with `backend=` to pick which torch routine actually
runs:

| `sparse` | `backend="torch"` routine |
|---|---|
| `False` | dense `torch.linalg.solve()` |
| `True` | Jacobi-preconditioned sparse Conjugate Gradient on a `torch.sparse_csr_tensor` — requires the assembled system to be SPD, true for the ordinary well-constrained linear-elastic case |

A few things worth knowing: `backend="torch"` needs the optional `torch`
dependency — `FESystem(..., backend="torch")` raises `ImportError`
**immediately at construction** if torch isn't importable in that
environment, rather than failing later inside `solve_static()` after
assembly work is done. And `backend="torch"` currently only affects
`solve_static()` — `solve_modal()`, `solve_linear_buckling()`, and every
transient/nonlinear driver below remain SciPy/NumPy-only regardless of
this setting, since no torch version of those exists yet. See
`fea_engine/src/fea_engine/torch_sparse_solver.py` and
`docs/consolidated_future_roadmap.md` for the full implementation and
validation detail.

**Free vibration (natural frequencies / mode shapes):**
```python
sys.assemble_mass(mat.rho * np.eye(2))   # rho*np.eye(dof_per_node) for 2-D/3-D continuum; rho*A for beams
freq_hz, mode_shapes = sys.solve_modal(n_modes=3)
```
Confirmed by running the cantilever from the Quickstart with mass
assembled: `freq_hz = [126.87, 459.35, 485.05]` Hz for the first three
modes.

**Harmonic / frequency response:**
```python
from fea_engine.loads import HarmonicLoad
sys.assemble_mass(rho_data); sys.assemble_damping(damping)
load = HarmonicLoad(LoadPattern(node_ids, dof_index), F0=1000.0)
F0_vec = load.force_vector(sys.n_dof, sys.npn)
U = sys.solve_frequency_sweep(2*np.pi*freqs_hz, F0_vec)          # Bode/FRF sweep
```

**Random vibration (PSD input):**
```python
from fea_engine.loads import PSDLoad
psd_load = PSDLoad(LoadPattern(node_ids, dof_index), freqs_hz, psd_input)
F0_unit = psd_load.force_vector(sys.n_dof, sys.npn)
S_out, sigma = sys.solve_random_vibration(freqs_hz, psd_input, F0_unit, output_dof=target_dof)
```

<a id="s-16"></a>

## 🕒 Time integration

**Implicit and explicit transients (impact, earthquake, general
transients):**
```python
from fea_engine.damping import RayleighDamping
from fea_engine.loads import TimeHistoryLoad, LoadPattern

sys.assemble_mass(rho_data)
damping = RayleighDamping.calibrate(omega_i, omega_j, zeta=0.02)
sys.assemble_damping(damping)
load = TimeHistoryLoad(LoadPattern(node_ids, dof_index), time_fn=lambda t: ...)

t, U_hist = sys.solve_transient_implicit(load, T_total, dt)     # unconditionally stable, pick dt for accuracy

sys.assemble_lumped_mass(rho_data)
dt = 0.4 * sys.critical_timestep()                               # stability-limited
t, U_hist = sys.solve_transient_explicit(load, T_total, dt)
```

**Modal superposition** (many repeated load cases on a linear system):
```python
sys.assemble_mass(rho_data)
t, U_hist, freq_hz = sys.solve_modal_superposition(load, T_total, dt, n_modes=8, zeta=0.02)
```

**Explicit dynamics for nonlinear models — impact/blast-scale
transients:** §"Time integration" above covers *linear* dynamics. For a
nonlinear element (large deflection, plasticity, contact) under a fast
transient — impact, blast, drop test — reach for
`solve_transient_explicit_nonlinear` instead of the implicit
`solve_nonlinear_transient` covered under Nonlinear Analysis below: it
uses the same central-difference recursion as the linear explicit
driver, but evaluates the *nonlinear* internal-force vector at each step
instead of a fixed `Kff @ d`. Because that internal force is evaluated
at the already-known current state, there's no Newton loop at all — the
cost per step is one `assemble_internal_force()` call, nothing else. The
trade-off, as always with explicit integration, is conditional
stability: `dt` must stay below the mesh's critical timestep or the
solution blows up.

```python
from fea_engine import Material, D_plane_stress, Quad4PlaneStress, FESystem
from fea_engine.mesh import rectangle_mesh
from fea_engine.nonlinear_solver import solve_transient_explicit_nonlinear

mat = Material(E=2.1e11, nu=0.3, rho=7850.0)
D = D_plane_stress(mat)
mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=8, ny=4)
sys = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
sys.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
sys.assemble_lumped_mass(mat.rho * np.eye(2))

# cheap, O(n_elements) local estimate -- no K/M assembly needed
c_e = np.sqrt(mat.E / (mat.rho * (1 - mat.nu**2)))   # plane-stress wave speed
dt_crit = sys.critical_timestep_local(wave_speed=c_e)
dt = 0.5 * dt_crit                                     # extra safety margin

tip = mesh.nodes_on_line(axis=0, value=0.4)
tip_dofs = np.array([sys._global_dofs([n])[1] for n in tip])

class ConstLoad:
    def force_at(self, t, n_dof, npn):
        F = np.zeros(n_dof)
        F[tip_dofs] = -20000.0 / len(tip)
        return F

t_hist, U_hist = solve_transient_explicit_nonlinear(sys, D, ConstLoad(), T_total=200 * dt, dt=dt)
```

Run as written (32 elements, 90 DOF): `dt_crit ≈ 8.2996e-06 s`, so
`dt ≈ 4.1498e-06 s`, 201 recorded steps, `2.16 s` wall time, tip
displacement peaking at `≈ 6.742e-06 m` under the abrupt load (the
dynamic overshoot above the static value — a plain step load excites a
structure's own free-vibration response on top of its eventual static
equilibrium, which is exactly what you'd expect physically).

Two notes on scope: `critical_timestep_local()` needs a per-element wave
speed you supply (material representations vary too much across element
families for one formula to fit all of them) — `sqrt(E/rho)` for a
bar/beam's axial wave speed, `sqrt((lambda + 2*mu)/rho)` for a 3-D
solid's dilatational wave speed, the plane-stress form above for a
membrane. And for a path-dependent material (plasticity), call
`sys.init_state()` first, same convention as the implicit nonlinear
drivers below — `commit_all_states()` is called automatically once per
step internally. See `fea_engine/docs/consolidated_future_roadmap.md`
for the corotational-shell extension and the full derivation.

<a id="s-17"></a>

## 🌀 Nonlinear analysis

Four drivers in `nonlinear_solver.py`, all calling the same
`assemble_internal_force()`/`assemble_tangent_stiffness()` machinery
regardless of *which* nonlinearity (geometric, material, contact) is in
play:

```python
from fea_engine.nonlinear_solver import (
    solve_nonlinear_static, solve_nonlinear_displacement_control,
    solve_nonlinear_arc_length, solve_nonlinear_transient,
)

# monotonic, pre-limit-point (large-deflection cantilever, etc.):
load_factors, U_hist = solve_nonlinear_static(sys, mat, n_steps=20)

# through a snap-through limit point, single meaningful control DOF:
U_hist, reaction_hist = solve_nonlinear_displacement_control(
    sys, mat, control_dof, u_target_array)

# through a limit point, general case (snap-back included) -- Crisfield arc-length:
load_factors, U_hist = solve_nonlinear_arc_length(sys, mat, delta_L=0.01, n_steps=60)

# nonlinear dynamic time integration (Newmark-Newton):
t, U_hist, V_hist, A_hist = solve_nonlinear_transient(
    sys, mat, load, T_total, dt, beta=0.25, gamma=0.5)
```

For a path-dependent element (plasticity), call `sys.init_state()` once
before the first solve, and pass a `load_factors=` sequence (rather than
a single ramp target) to trace a load/unload/reload history:

```python
sys.init_state()
lambda_seq = [0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, 0.0]
load_factors, U_hist = solve_nonlinear_static(sys, mat, load_factors=lambda_seq)
```

> [!TIP]
> **If a step won't converge:** the first three drivers above (plus
> `solve_nonlinear_koiter_newton`/`solve_nonlinear_static_koiter_newton`)
> accept `line_search=True` (the default) to automatically retry a
> non-converging step with Armijo backtracking before raising, and
> `du_tol=`/`energy_tol=` to add displacement-increment/energy-error
> convergence checks on top of the default force-residual one. Both are
> opt-in extras layered onto the existing behavior, not replacements for
> it — see `fea_engine/README.md`'s "Newton-solver robustness knobs" note
> for the full detail.

See `fea_engine/README.md` sections I–L for contact (penalty and exact
Lagrange-multiplier formulations) and the full nonlinear driver
selection guidance.

<a id="s-18"></a>

## 📦 Batched workflows: iterative solvers and adaptive mesh refinement

Two independent large-model capabilities, both opt-in (the direct solve
above remains the default and is unaffected by either).

**Iterative solvers** (`iterative_solvers.py`) — for a system too large
for a dense/sparse-direct factorization to be the best choice,
`preconditioned_cg` runs a Jacobi- or SSOR-preconditioned conjugate
gradient solve directly against `FESystem`'s own assembled `Kff`/`Ff`:

```python
from fea_engine.iterative_solvers import preconditioned_cg, ssor_preconditioner

Kff = sys.K.tocsr()[free, :][:, free]     # or sys.K[np.ix_(free, free)] if dense
Ff = sys.F[free]

x_plain, n_iter = preconditioned_cg(Kff, Ff, tol=1e-8)               # no preconditioner
M_ssor = ssor_preconditioner(Kff, omega=1.0)
x_ssor, n_iter_ssor = preconditioned_cg(Kff, Ff, M=M_ssor, tol=1e-8)  # SSOR-preconditioned
```

Run on the Quickstart cantilever (624 free DOF): unpreconditioned CG
took **116 iterations**; SSOR preconditioning cut that to **58** — both
converged to the same solution as `sys.solve_static()` to within
`2.2e-15` absolute. `jacobi_preconditioner`/`incomplete_cholesky0` are
also available (same `M=` contract); `multigrid_solve` (a V-cycle/
W-cycle geometric multigrid, `structured_quad_hierarchy` builds the mesh
hierarchy) is the next step up for genuinely large structured meshes
where even a preconditioned CG's iteration count grows with mesh size.

`preconditioned_cg` also takes `backend="scipy"` (default, unchanged) /
`backend="torch"` + `device=` (Wave 9 addendum item 137) — with
`jacobi_preconditioner_torch()`/`ssor_preconditioner_torch()` as the
torch-native preconditioners (IC(0) stays SciPy-only; passing it on the
torch path raises `TypeError` rather than silently misbehaving), and
this composes directly with `FESystem.solve_static(method="cg"/"pcg",
backend="torch")` — the same `backend=` this whole section starts from,
no longer restricted to the direct-solve path. Validated for real on a
PyTorch-equipped machine: 9/9 (`preconditioned_cg` itself) then 17/17
(the `solve_static()` wiring) tests passed on the first real run each,
agreeing with the SciPy path to `~1e-6` relative — see
`docs/consolidated_future_roadmap.md` item 137 for the full record.

**Adaptive mesh refinement** (`adaptivity.py`) — the capstone loop:
solve, estimate the per-element error, mark elements, refine locally
(conforming Rivara longest-edge splits, `Tri3PlaneStress` only), and
re-solve, automatically, without you hand-picking a refined mesh:

```python
from fea_engine.mesh import Mesh, rectangle_mesh, weld_meshes
from fea_engine import Tri3PlaneStress, D_plane_stress, Material
from fea_engine.adaptivity import adaptive_refine_solve

def _quad_to_tri3(qmesh):
    tris = [[e[0], e[1], e[2]] for e in qmesh.elements] + \
           [[e[0], e[2], e[3]] for e in qmesh.elements]
    return Mesh(qmesh.nodes.copy(), np.array(tris), dim=2)

# L-shaped domain, re-entrant corner at (1,1) -- the canonical case
# adaptive refinement exists for: the exact solution has a corner
# singularity a uniform mesh converges to only very slowly.
A = rectangle_mesh(2.0, 1.0, 8, 4)
B = rectangle_mesh(1.0, 1.0, 4, 4, x0=0.0, y0=1.0)
mesh0 = _quad_to_tri3(weld_meshes(A, B))

def setup(fs, mesh):
    for n in mesh.nodes_on_line(axis=0, value=0.0):
        fs.fix_dofs([n], [0, 1])
    edge = [n for n in mesh.nodes_on_line(axis=1, value=1.0) if 1.0 <= mesh.nodes[n, 0] <= 2.0]
    for n in edge:
        fs.F[fs._global_dofs([n])[1]] += -5000.0 / len(edge)

history = adaptive_refine_solve(mesh0, Tri3PlaneStress(), D_plane_stress(mat), setup,
                                 estimator="zz", marking="fixed_fraction",
                                 marking_param=0.3, max_refinements=4, verbose=True)
```

Run as written, five ZZ-recovery-driven refinement steps took the mesh
from 96 to 486 elements while the estimated total error dropped
monotonically from `8.818e-03` to `3.970e-03` — concentrating new
elements exactly where the re-entrant corner needs them, not uniformly
everywhere. `estimator="jump"` (inter-element traction jump) is the
alternative estimator; `marking="threshold"`/`"equidistribution"` the
alternative marking strategies; pass `tol=` instead of relying on
`max_refinements` to stop once the estimated error meets a target.

<a id="s-19"></a>

## 🔥 Differentiability and GPU backends

The static-solve `backend="torch"`/`device=` switch above extends to
nonlinear tangents and nonlinear transient drivers — genuinely useful
once a model is large enough, or a tangent expensive enough to
differentiate by finite difference, that GPU compute or exact automatic
differentiation pays for itself:

```python
# exact AD tangent instead of finite-difference/complex-step (accuracy, not speed)
solve_nonlinear_static(sys, mat, method="autograd")          # Tet4NeoHookean
tangent_stiffness(..., method="autograd")                    # Shell4MITCCorotational, iter_state=None only

# GPU dispatch for the Newton solve's linear system, same signature as solve_static's own:
solve_nonlinear_transient(sys, mat, load, T_total, dt, backend="torch", device="cuda")
solve_transient_displacement_control(sys, mat, control_dof, u_target, backend="torch", device="cuda")
```

This sandbox can't install `torch` (its package index is blocked here —
see Installation's own note), so these specific numbers come from the
project's own validation record rather than a session run in this
guide: on the developer's PyTorch-equipped machine, all 30 torch-gated
tests across these items passed (`method="autograd"` agreeing with the
existing finite-difference/complex-step tangents to within `1.25e-16`
relative on the shell case; `backend="torch"` agreeing with the default
`backend="scipy"` transient path to `~1e-9` relative on a linear-limit
case and `~1e-6` relative on a genuinely nonlinear one). Every
`method`/`backend` value not recognized raises `ValueError` immediately,
and `backend="torch"` with torch unavailable fails fast at the call, not
mid-solve — matching every other torch dispatch point in this package.
See `docs/consolidated_future_roadmap.md` for the full per-item
validation detail, and run `pytest
tests/test_torch_autograd_tangent_stiffness.py
tests/test_torch_transient_backend.py
tests/test_torch_shell_autograd_tangent.py -v` yourself once torch is
installed to reproduce that record on your own machine.

**`rom_engine`'s differentiable-correction layer** (torch-gated,
research-grade — see "Advanced Capabilities" below) builds on this same
switch: a trainable additive correction term inserted into an
equilibrium residual, calibrated either through an adjoint
(implicit-differentiation) backward pass or explicit residual
minimization.

<a id="s-20"></a>

## 🧩 ROM: linear workflow (FOM → POD → Galerkin → optional affine sweep)

This is the pattern to learn first for `rom_engine` — everything else
either specializes it (frequency-domain, nonlinear) or complements it
(non-intrusive identification). Three stages:

1. **Offline, once:** run a handful of full-order solves ("snapshots")
   spanning the parameter/load range you care about, extract a POD
   basis that captures how the structure actually deforms.
2. **Offline, once:** project the full-order operators onto that basis
   (Galerkin projection — for a single fixed model; affine
   decomposition — if the model varies with a parameter you want to
   sweep).
3. **Online, many times:** solve the tiny reduced system instead of the
   full one, as many times as a design study needs.

### A minimal single-model example

```python
import numpy as np
from fea_engine import Material, D_plane_stress, Quad4PlaneStress, FESystem
from fea_engine.mesh import rectangle_mesh
from rom_engine import PodBasis, GalerkinROM

# ---- 1. Full-order model (fea_engine) ----
mat = Material(E=2.1e11, nu=0.3, rho=7850.0)
mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=24, ny=12)
sys = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
sys.assemble_stiffness(D_plane_stress(mat), thickness=0.02)
sys.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
sys.assemble_mass(mat.rho * np.eye(2))

# ---- 2. Snapshots: a handful of static solves at varied load patterns ----
tip = mesh.nodes_on_line(axis=0, value=0.4)
rng = np.random.default_rng(0)
snaps = np.zeros((sys.n_dof, 8))
for i in range(8):
    sys.F[:] = 0.0
    sys.add_nodal_force(tip, dof_index=1, total_force=-20000.0 * rng.uniform(0.5, 1.5))
    snaps[:, i] = sys.solve_static()

# ---- 3. POD basis + Galerkin-reduced model ----
basis = PodBasis().fit(snaps, n_modes=6, M=sys.M)     # mass-weighted (physically correct for mixed DOF)
rom = GalerkinROM(basis).reduce_system(sys.K, M=sys.M, F=sys.F)

x_full, q_reduced = rom.solve_static()                 # x_full has the SAME shape as sys.solve_static()
freq_hz, modes_full, modes_reduced = rom.solve_modal()
```

Running this on the same cantilever from the Quickstart, confirmed
live: an 8-snapshot, 6-mode basis already captures 100.0% of the
training energy (`basis.energy_captured()`), and `x_full` from the
reduced solve matches `sys.solve_static()` on the same load to `~7e-6`
absolute (this problem has one dominant load pattern, so a handful of
modes goes a long way — a wider variety of training loads would need
more modes for the same accuracy). Note `n_modes=6` here is less than
the 8 snapshots, so this is a genuine truncation, not an exact
full-rank reproduction; passing `n_modes=8` (or omitting it, which
defaults to full rank) would reproduce every training load to machine
precision, the defining property of a Galerkin-projected basis at full
rank.

### The parametric case: affine decomposition for fast sweeps

When the stiffness (or mass) matrix is a **linear combination of a
parameter**, `K(mu) = theta_1(mu)*K_1 + theta_2(mu)*K_2 + ...`, you can
project the *components* once and reassemble cheaply online for every
new `mu` — this is what makes many-query problems (design studies,
sensitivity sweeps, uncertainty quantification) fast.
`rom_engine/examples/two_region_beam_rom.py` is a complete, runnable
version of this — a cantilever beam split into two regions with
independent bending rigidities `EI_1`, `EI_2`:

```python
from fea_engine.mesh import MultiBlockMesh
from fea_engine.geometry import generate_mesh
from fea_engine.solver import FESystem
from fea_engine import elements
from rom_engine import PodBasis, GalerkinROM, AffineDecomposition

mesh = generate_mesh(dim=1, L=1.0, n=120)
half = 60
blocks = {"region1": mesh.elements[:half], "region2": mesh.elements[half:]}
mb_mesh = MultiBlockMesh(nodes=mesh.nodes, blocks=blocks, dim=mesh.dim)
elem_map = {"region1": elements.Beam2DEulerBernoulli(), "region2": elements.Beam2DEulerBernoulli()}

def assemble(EI1, EI2):
    s = FESystem(mb_mesh, elem_map)
    s.assemble_stiffness({"region1": EI1, "region2": EI2})
    s.fix_dofs([0], [0, 1])
    return s

K1 = assemble(1.0, 0.0).K     # unit-rigidity contribution from region 1
K2 = assemble(0.0, 1.0).K     # unit-rigidity contribution from region 2

# ... build a POD `basis` from static-solve snapshots at varied (EI1, EI2), as above ...

affine = AffineDecomposition([K1, K2], theta_func=lambda mu: [mu[0], mu[1]]).project(basis.V)

# online: as many (EI1, EI2) queries as you like, each a tiny reduced solve
for EI1, EI2 in many_parameter_draws:
    K_r = affine.assemble_reduced((EI1, EI2))
    q = np.linalg.solve(K_r, GalerkinROM(basis).project_vector(F_full))
    x_rom = basis.V @ q
```

Actually running `two_region_beam_rom.py` end to end (120-element beam,
10 training snapshots, 10-mode basis) gives, verified live:

```
POD basis: 10 modes, energy captured = 1.000000
max relative tip-deflection error over 8 held-out (EI1, EI2) points: 1.51e-03
500-point sweep: full-order = 1653.23 ms, reduced-order = 7.45 ms → 221.8x speedup
```

That's the whole point of the affine path: sub-2%-error answers, over
two hundred times faster than reassembling and re-solving the full
model at every query.

<a id="s-21"></a>

## 🧩 ROM: frequency-domain and certified bounds

`A(omega) = -omega^2*M + i*omega*C + K` is exactly an affine-in-`omega`
system, so `FrequencyROM` is `affine.py` + `galerkin.py` composed for
you, with a convenience constructor:

```python
from rom_engine import FrequencyROM

rom_freq = FrequencyROM.from_MCK(sys.M, sys.K, basis, rayleigh=(alpha, beta))   # or C=C_matrix
response = rom_freq.frequency_response(omega_array, sys.F)   # (n_omega, n_dof) complex
```

`basis` can be a `PodBasis` or a plain `(n_dof, n_modes)` array — a
*modal* basis (from `sys.solve_modal()`) or a basis built from **POD on
FRF snapshots** both work; `build_pod_basis_from_frf_snapshots()` is
provided for the latter. `FrequencyROM` is validated against
`fea_engine`'s own `solve_harmonic()`/`solve_frequency_sweep()` both
near resonance (the numerically hard case) and off it.

**Adaptive (greedy) training**, instead of a fixed frequency grid —
spend full-order solves where the current basis is weakest:

```python
from rom_engine import greedy_train_frequency_basis
basis, history = greedy_train_frequency_basis(
    candidate_omegas, sys.M, sys.K, sys.F, rayleigh=(alpha, beta), max_modes=20)
```

**A genuinely certified error bound** (not just an estimate), useful
near a reference frequency of interest:

```python
from rom_engine import SingularValueLowerBound, certified_error_bound
scm = SingularValueLowerBound.from_affine(rom_freq.affine)
scm.add_reference(omega_of_interest)                                   # offline, one SVD
bound = certified_error_bound(rom_freq, scm, omega_of_interest, sys.F)  # online, cheap
```
This bound is honestly weaker near resonance than away from it (a
mathematical property of the underlying method, documented in `scm.py`
— read that module's docstring before leaning on it far from your
reference point).

<a id="s-22"></a>

## 🧩 ROM: classical structural-dynamics / systems-and-control MOR

Four more methods, added to reproduce Besselink et al. (2013)'s own
head-to-head comparison of model-reduction techniques from structural
dynamics, numerical mathematics, and systems and control — see
`rom_engine/docs/classical_mor_roadmap.md` for the full design
rationale. Numbers below are from a live run against a real 20-element
`fea_engine` damped cantilever.

**Mode acceleration and modal truncation augmentation** — cheap,
closed-form fixes for plain mode displacement's static-load blind spot,
using one extra full-order static solve:

```python
from rom_engine import mode_acceleration_correction, mode_acceleration_response, augmented_basis

rom = GalerkinROM(basis).reduce_system(K, F=F)   # basis: e.g. 3 undamped modes
x_md, eta = rom.solve_static()                    # plain mode displacement

q_cor = mode_acceleration_correction(K, basis, F)
x_ma = mode_acceleration_response(basis, eta, q_cor)   # mode acceleration

Psi = augmented_basis(basis, q_cor, M=M)          # fold correction INTO the basis
x_aug, _ = GalerkinROM(Psi).reduce_system(K, F=F).solve_static()
```

With only 3 modes kept, plain mode displacement's relative error against
the exact static solve was `6.18e-03`; mode acceleration and modal
truncation augmentation both reproduced the exact solution to machine
precision (`0.0` and `2.44e-12` respectively) — an algebraic identity
for the load the correction was built from, not an approximation that
happens to be more accurate.

**Krylov-subspace moment matching** — matches the transfer function's
Taylor moments exactly at an expansion point `s0`, for a general
`(M,C,K,B,Cout)` input/output port:

```python
from rom_engine import KrylovROM

rom_mm = KrylovROM.from_MCK(M, K, B, Cout, C=C, s0=0.0, k=5)             # one-sided
rom_mm2 = KrylovROM.from_MCK(M, K, B, Cout, C=C, s0=0.0, k=5, two_sided=True)  # Petrov-Galerkin

H = rom_mm.frequency_response(omega_array)   # complex H(i*omega) at the port
rom_mm.is_stable()                           # NOT guaranteed -- check, don't assume
```

At `k=5` on the cantilever's tip-force/tip-displacement port, far from
`s0=0`, one-sided moment matching had mean relative error `1.53e-01`;
two-sided (matching roughly twice as many moments at the same order) cut
that to `1.15e-02` — a real improvement, though its size varies with `k`
(see `krylov.py`'s own module docstring).

**Balanced truncation** — the one method that stays accurate across the
*whole* frequency range because it uses where the ports are, with a
genuine a priori error bound:

```python
from rom_engine import BalancedTruncationROM

# modally pre-truncate a finely-meshed FE model first (see
# balanced_truncation.py's module docstring for why)
rom_bt = BalancedTruncationROM.from_MCK(M_reduced, K_reduced, B, Cout, C=C_reduced, r=10)

rom_bt.frequency_response(omega_array)
rom_bt.is_stable()                # a THEOREM for balanced truncation
rom_bt.h_infinity_error_bound()   # a real, checked a priori worst-case bound
```

Pre-truncated to 15 modes then balanced down to `r=10` states, this ROM
stayed under `2.35e-03` max relative error across a `1`–`1500` rad/s
sweep, with a computed H-infinity bound of `2.34e-09` (the raw, absolute
worst-case bound — genuinely dominates the observed absolute error,
checked directly in `test_balanced_truncation.py`) and
`is_stable() == True`, as guaranteed.

`from_MCK` also takes `backend="numpy"` (default) / `backend="torch"` +
`device=` (Wave 9 addendum item 138) — accelerates the two dense
Lyapunov solves plus the balancing SVD this method's own compute is
dominated by (no native torch Lyapunov solver exists, so
`torch_linalg.py` uses the classical Kronecker-sum vectorization
instead of SciPy's Bartels-Stewart; cross-validated to machine precision
before any torch code was written). Same `backend=`/`device=` on
`SingularPerturbationROM`/`FrequencyWeightedBalancedTruncationROM`/
`hankel_norm.OptimalHankelNormROM`'s own `from_MCK`. Validated for real
on a PyTorch-equipped machine, 21/21 then 13/13 tests passed (two real
bugs found and fixed along the way — `torch.kron()` requiring
contiguous operands, and an SVD sign-ambiguity in a test's own
assertion, not in the underlying math) — see
`docs/consolidated_future_roadmap.md` item 138 for the full record.
`krylov.py`/`loewner.py` are NOT ported — both already operate on
systems too small for a GPU to help (see `torch_linalg.py`'s own module
docstring).

<a id="s-23"></a>

## 🧩 ROM: non-intrusive identification (no `M`/`C`/`K` required)

`LoewnerROM` identifies modal parameters (frequencies, damping ratios,
mode shapes) from **sampled complex frequency-response data alone** —
useful for experimental data, or for treating a solver as a black box:

```python
from rom_engine import LoewnerROM, screen_physical_modes

rom_id = LoewnerROM.fit(omega_alpha, omega_beta, x_alpha, x_beta)
print(rom_id.f, rom_id.eta)      # identified natural frequencies [Hz], damping ratios
mode_shapes = rom_id.reconstruct_mode_shapes(X_beta_multi, omega_beta=omega_beta)

# cross-check many random ROMs, keep only recurring (physical, not numerical-artifact) modes:
physical = screen_physical_modes(freq_pool, x_pool, fmin, fmax, rng=np.random.default_rng(0))
```

Validated against both a synthetic mass-spring fixture and two real
structural models (a `fea_engine` damped cantilever, and an 867-DOF
plate model matching a published benchmark) — see
`rom_engine/examples/plate_modal_identification.py` and
`mode_shape_vibration_recovery.py`.

<a id="s-24"></a>

## 🧩 ROM: nonlinear structural ROMs

For geometrically nonlinear structures (large-deflection beams/plates),
`rom_engine` combines an intrusive linear modal basis with a
non-intrusive, black-box regression of the nonlinear reduced restoring
force — two families, both fit from sampled training data generated by
running your own `fea_engine` nonlinear solver a modest number of times:

```python
from rom_engine import MultiFidelitySurrogate, AppliedLoadStrategy, PolynomialModalROM

# MFS-NLROM family: RBF surrogate of F_nl(q_l)
strategy = AppliedLoadStrategy(target_fracs=(0.3, 2.0), reference_scale=thickness,
                                n_samples=40, rng=np.random.default_rng(0))
q_l, q_nl, F_nl = strategy.generate(V, M_ff, basis_freqs_hz, mode_shape_peaks, fom_solver)
surrogate = MultiFidelitySurrogate(kernel="cubic").fit(q_l, F_nl)
F_nl_pred = surrogate.predict(q_l_new)         # closed-form, no Newton-Raphson
J = surrogate.jacobian(q_l_new)                # analytic dF_nl/dq_l

# ICE/Shi-Mei family: Nash-form polynomial fit against q_nl, Newton solve
poly_rom = PolynomialModalROM(n_modes=len(basis_freqs_hz)).fit(q_nl, F_nl)
q_solution, converged = poly_rom.predict(F_ext=F_ext, Lambda=Lambda)
```

`fom_solver` is a callable **you** supply — it's typically a thin
wrapper around `fea_engine.nonlinear_solver.solve_nonlinear_static()` or
`solve_nonlinear_transient()` on your model, which is exactly how
`rom_engine`'s own tests build it (see `tests/fea_fixtures.py` and
`tests/test_nonlinear_rom_fea.py` for a complete, real example against a
clamped-clamped `Beam2DCorotational` beam).

**A third option, `NeuralSurrogate`** — same role and
`fit()`/`predict()`/`jacobian()` contract as `MultiFidelitySurrogate`
above (a genuine drop-in: it consumes the identical `(q_l_samples,
F_nl_samples)` data any `TrainingStrategy` already produces), but a
small `torch.nn` MLP in place of the RBF interpolant. Worth reaching for
once your training set is large enough that RBF's `(N, N)` kernel matrix
(revisited and re-solved in full on every `fit()`) becomes the
bottleneck — an MLP's training cost is `O(N)` per epoch instead:

```python
from rom_engine import NeuralSurrogate

surrogate = NeuralSurrogate(n_modes=len(basis_freqs_hz), hidden_sizes=(32, 32),
                             activation="tanh", n_epochs=2000, seed=0)
surrogate.fit(q_l, F_nl)
F_nl_pred = surrogate.predict(q_l_new)
J = surrogate.jacobian(q_l_new)          # torch.autograd.grad, row by row
```

This needs the optional `torch` dependency (Installation's notes apply
here too), which this sandbox can't install — so unlike every other
number in this guide, `NeuralSurrogate`'s own headline numbers aren't
reproducible in this particular session. What *is* confirmed: the
plain-NumPy normalization helpers underneath it
(`_fit_normalization`/`_normalize`/`_denormalize`) pass 3 unconditional
tests with no torch at all, and a downstream consumer —
`neural_surrogate_ensemble()`, an ensemble of several `NeuralSurrogate`
instances trained from different random seeds — is confirmed PASSING on
the developer's own PyTorch-equipped machine (genuine nonzero predictive
spread across independently-initialized members, the actual epistemic-UQ
payoff that utility exists for). The 12+2 tests exercising
`NeuralSurrogate` directly (`tests/test_nonlinear_rom.py`,
`tests/test_nonlinear_rom_fea.py`) are torch-gated and skip cleanly in
this sandbox — run them yourself once torch is installed (`pytest
tests/test_nonlinear_rom.py tests/test_nonlinear_rom_fea.py -v`) for a
full local confirmation.

Once you have a surrogate, two more pieces close the loop:

```python
from rom_engine import integrate_newmark_surrogate, HarmonicBalanceSystem, solve_nnm_backbone

# reduced nonlinear time integration -- entirely Newton-free
t, q_nl_hist, q_l_hist, F_nl_hist = integrate_newmark_surrogate(
    Lambda, C_r, surrogate, F_ext=0.0, q0=q_nl0, qdot0=np.zeros_like(q_nl0),
    dt=dt, n_steps=n_steps, domain="q_l")

# nonlinear normal mode (NNM) backbone: frequency vs. amplitude for the
# unforced, undamped periodic-orbit family of a given mode
hb = HarmonicBalanceSystem(Lambda, n_harmonics=5)
amplitudes, omegas, Z_hist, converged = solve_nnm_backbone(
    hb, surrogate, q_amplitude_range=(a_min, a_max), n_points=20, domain="q_l", master_mode=0)
```

Both were validated against a real `fea_engine` clamped-clamped
`Beam2DCorotational` model — a free-decay run for the time-integration
path, and a backbone measured independently via time-integration + FFT
period for the NNM path (`test_nonlinear_dynamics_fea.py`,
`test_nnm_fea.py`).

`rom_engine` also includes a family of newer, neural-operator-informed
reduced-dynamics routes built the same way — trajectory-level,
torch-gated, and generalizing across excitation shape, mesh resolution,
or physical parameter rather than a single instantaneous state. They are
deliberately out of scope for this everyday-usage guide; see
`fea_engine_rom_engine_coupling.html` in this same folder for the full,
general-purpose explanation of every ROM formulation route in the
package (including these), and `rom_engine/README.md` for the module-
level API.

<a id="s-25"></a>

## 🧭 Choosing a ROM method

| You have... | Reach for... |
|---|---|
| One fixed linear FE model, want a fast reduced solve | `PodBasis` + `GalerkinROM` |
| A linear model that varies with a parameter, want a fast sweep | + `AffineDecomposition` |
| A linear model, want fast harmonic/frequency sweeps | `FrequencyROM` (+ `greedy_train_frequency_basis`, `SingularValueLowerBound` if you need adaptive training or a certified bound) |
| A linear model with a truncated modal basis, want the static response fixed cheaply | `mode_acceleration_correction`/`mode_acceleration_response`, or `augmented_basis` |
| A linear model with explicit inputs/outputs, want a small transfer-function ROM local to one frequency | `KrylovROM` (`two_sided=True` for roughly double the matched moments) |
| A linear model with explicit inputs/outputs, want accuracy across the WHOLE frequency range with a certified bound | `BalancedTruncationROM` |
| Only measured/sampled frequency-response DATA, no matrices | `LoewnerROM` (+ `screen_physical_modes`) |
| A geometrically nonlinear model, want a fast reduced solve or time integration | `MultiFidelitySurrogate`/`PolynomialModalROM` (or `NeuralSurrogate` for a larger training set) + `integrate_newmark_surrogate` |
| A nonlinear model's backbone curve (amplitude-dependent frequency) | `HarmonicBalanceSystem` + `solve_nnm_backbone` |

---

<a id="s-26"></a>

# 🖼️ Example Gallery

Runnable, end-to-end scripts, grouped by workflow.

For a guided, numbered tour instead of this topic-by-topic list, see the
dedicated galleries: `fea_engine/examples/gallery/` (`fea_01_...` through
`fea_06_...`, plus the related `fea_scipy_vs_torch_gpu_benchmark.py`
solver-backend benchmark one level up) and `rom_engine/examples/gallery/`
(`rom_01_...` through `rom_06_...`) — each script is self-contained,
prints its own validation numbers, and saves a plot; see each folder's
own `fea_README.md`/`rom_README.md` for the full script-by-script
breakdown.

<a id="s-27"></a>

## 🏗️ Structural mechanics

- **Cantilever statics/modal/transient** — `fea_engine/examples/main.py`
  runs 8 problems end to end against closed-form references; the
  Quickstart section above walks through the static case by hand.
- **Buckling** — linear buckling eigenproblem examples.
- **Arc-length snap-through** — through-a-limit-point nonlinear statics.
- **Plasticity and hyperelasticity** — path-dependent and large-strain
  material examples.
- **Shells and 3-D frames** — `Shell4MITC` and `Beam3DEulerBernoulli`
  worked models.
- **Sparse vs. dense timing, higher-order elements** — direct
  performance comparisons.

See `fea_engine/examples/` for the full, runnable set.

<a id="s-28"></a>

## 📉 Reduced-order modeling

- **`two_region_beam_rom.py`** — the affine-decomposition parametric
  sweep from "ROM: linear workflow" above; run it yourself for the full
  221.8x-speedup numbers.
- **`frequency_sweep_cantilever.py`** — `FrequencyROM` against
  `fea_engine`'s own harmonic solver.
- **`greedy_frequency_training.py`** — adaptive basis training in
  action.
- **`certified_bound_cantilever.py`** — the Successive Constraint Method
  bound from "ROM: frequency-domain and certified bounds."
- **`plate_modal_identification.py`** and
  **`mode_shape_vibration_recovery.py`** — `LoewnerROM` against both a
  synthetic fixture and an 867-DOF plate benchmark.

The classical-MOR methods (mode acceleration, `KrylovROM`,
`BalancedTruncationROM`) don't yet have a dedicated example script — see
`rom_engine/tests/test_mode_correction.py`/`test_krylov.py`/
`test_balanced_truncation.py` for further worked patterns.

---

<a id="s-29"></a>

# 🔬 Advanced capabilities (research-grade)

Everything above is deliberately the "everyday" path — the capability
set you'd reach for building and analyzing an ordinary structural model.
Two further layers sit on top: differentiable-simulation/data-
calibration infrastructure, and a set of TensorMesh-inspired
performance/architecture upgrades. Both are additive and opt-in —
nothing above changes any default behavior — and both are narrower/more
specialized, so this section is intentionally a map, not a tutorial:
read the module docstrings and `fea_engine/docs/
consolidated_future_roadmap.md` for the full derivations and validation
detail before building on any of them.

<a id="s-30"></a>

## 🎛️ Differentiable correction / data calibration

The idea of wrapping a whole nonlinear solve as a differentiable layer
so a trainable additive correction term can be calibrated end-to-end
against reference data, via either an adjoint (implicit-differentiation)
backward pass or explicit residual minimization.

- **`fea_engine/differentiable.py`** — the full-order version.
  `AdditiveCorrection` is the protocol a correction term implements
  (`value()`/`jacobian()`, plus `torch_value()`/`parameters()` if
  trainable); `corrected_residual()`/`corrected_tangent()` insert it
  into `R(u) = F_ext - F_int(u) - f_θ(u)`;
  `solve_nonlinear_static_corrected()` is the driver.
  `adjoint_gradient()` differentiates *through* that solve without
  unrolling the Newton iteration. `ScalarFieldCorrection`/
  `MLPCorrection` are the two concrete trainable corrections
  (torch-gated); `calibrate_correction_explicit()` is the training loop.
- **`rom_engine/differentiable_correction.py`** — the reduced-order
  analogue, independently reimplemented at reduced-coordinate scale
  (deliberately *not* importing `fea_engine.differentiable` — see
  `fea_engine_rom_engine_coupling.html`'s shared-pattern-coupling
  section for why): `reduced_residual()`, `ScalarModalCorrection`,
  `calibrate_reduced_correction_explicit()`.
- **`rom_engine/dataset_diagnostics.py`** / **`ensemble_uq.py`** —
  PCA/Sobol-based dataset completeness diagnostics, and the ensemble
  epistemic-UQ wrapper mentioned in "ROM: nonlinear structural ROMs"
  above.

Validation: 8 core tests in `fea_engine/tests/test_differentiable.py`
run and pass unconditionally, no torch needed (the adjoint-formula and
Newton-convergence math is checkable against a hand-coded pure-NumPy
correction). The torch-gated pieces — concrete NN corrections, the
training loops, the topology-optimization adjoint-sensitivity path below
— are confirmed passing on the developer's own machine: 23/23 in
`fea_engine`, 11/11 in `rom_engine`, zero failures. Not reproduced in
this guide's own sandbox session for the same reason as elsewhere in
this guide.

<a id="s-31"></a>

## 🏛️ Architecture upgrades (batched assembly, GPU dispatch, topology optimization)

Six additive `fea_engine` modules inspired by TensorMesh's own
GPU-native architecture, each independently opt-in.

| Module | What it adds |
|---|---|
| `mesh_transform.py` | `MeshTransformation` — precomputes shape functions/gradients/Jacobians for a whole homogeneous-element-type mesh in one batched NumPy pass, the shared input every other item below builds on. |
| `vectorized_assembly.py` | `assemble_stiffness_vectorized()` / `FESystem.assemble_stiffness(vectorized=True)` — a loop-free global assembly path (one `einsum` + one vectorized sparse scatter) as an opt-in alternative to the existing per-element Python loop. |
| `backend_dispatch.py` | `FESystem(backend="auto")` + `solve_static(verbose=True)` — inspects the assembled system (size, symmetry, SPD-ness) and picks scipy vs. torch automatically, printing a one-line diagnostic; never routes a non-SPD/non-symmetric system to torch regardless of size. |
| `batched_solve.py` | `solve_batched()` / `solve_static_batched()` — one factorization, many right-hand sides (a parametric load sweep) instead of re-solving from scratch each time. |
| `topopt.py` | `topology_optimize_compliance()` — SIMP-based compliance topology optimization (optimality-criteria update + sensitivity filter), reusing the differentiable-correction layer's adjoint machinery for the compliance sensitivity. |
| `mixed_assembly.py` | `assemble_mixed_tet10_p1()` — Taylor-Hood (P2 displacement / P1 pressure) mixed u-p assembly for 3-D solids, an LBB-stable alternative to B-bar for the near-incompressible limit; scoped to the 3-D solid case only (no shell analogue yet). |

None of these six modules are imported into `fea_engine/__init__.py`
(the same "flat, directly-imported-by-the-caller" convention as
`differentiable.py`/`autograd_tangent.py`) — import them explicitly from
their own module path, as shown above. All 47 unconditional tests across
the six new test files pass in this sandbox; the one torch-gated test
(`topopt.py`'s adjoint-sensitivity cross-check) skips here and awaits a
run on a PyTorch-equipped machine, same pattern as everywhere else
torch-gated in this project.

`mesh_transform.py`/`vectorized_assembly.py` additionally take
`backend="numpy"` (default) / `backend="torch"` + `device=` (Wave 9
addendum items 135/136) — `MeshTransformation(..., backend="torch")`
keeps its shape functions/Jacobians as LIVE `torch.Tensor`s rather than
converting back to NumPy, so `assemble_stiffness_vectorized(...,
backend="torch")` consumes them directly with no GPU→NumPy→GPU round
trip; the one unavoidable host copy happens at the final sparse-scatter
step, since `FESystem.K`'s own contract is always plain NumPy/SciPy
regardless of assembly backend. Validated for real on a PyTorch-equipped
machine: 11/11 tests for each item, agreeing with the NumPy path to
`atol=1e-3` (ordinary float64 summation-order noise at the ~1e11
stiffness scale, the same tolerance the existing NumPy-vs-per-element-
loop regression already needs) — see `docs/consolidated_future_roadmap.md`
items 135/136 for the full record.

---

<a id="s-32"></a>

# 📘 API Reference

A compact pointer table — the full per-function reference lives in each
package's own `README.md`.

<a id="s-33"></a>

## `fea_engine`

| Module | Role |
|---|---|
| `material.py` | `Material`, `Section`, `D_xxx(mat)` constitutive matrices |
| `mesh.py` | `Mesh`, `MultiBlockMesh`, structured mesh generators, quality checks, plotting |
| `geometry.py` | Dimension-driven `build_system()` shortcut (`geometry/gmsh_engine.py`, Gmsh-driven meshing/CAD import, was removed -- see `fea_engine/README.md`'s "Removed: Gmsh support") |
| `elements/` | Every registered `Element` subclass; see "Elements and quadrature" above |
| `solver.py` | `FESystem` — assembly, boundary conditions, `solve_static`/`solve_modal`/`solve_frequency_sweep`/`solve_random_vibration` |
| `nonlinear_solver.py` | The four nonlinear drivers, plus `solve_transient_explicit_nonlinear` |
| `loads.py` | `LoadPattern`, `TimeHistoryLoad`, `HarmonicLoad`, `PSDLoad` |
| `damping.py` | `RayleighDamping` and calibration helpers |
| `iterative_solvers.py` | `preconditioned_cg`, `jacobi_preconditioner`, `ssor_preconditioner`, `incomplete_cholesky0`, `multigrid_solve` |
| `adaptivity.py` | `adaptive_refine_solve` and its estimator/marking strategies |
| `torch_sparse_solver.py`, `autograd_tangent.py` | The `backend="torch"`/`method="autograd"` dispatch points |
| `differentiable.py` | The full-order differentiable-correction layer (see "Advanced capabilities") |
| `mesh_transform.py`, `vectorized_assembly.py`, `backend_dispatch.py`, `batched_solve.py`, `topopt.py`, `mixed_assembly.py` | The architecture-upgrade modules (see "Advanced capabilities") |

<a id="s-34"></a>

## `rom_engine`

| Module | Role |
|---|---|
| `pod.py` | `PodBasis` — Proper Orthogonal Decomposition, standard and mass-weighted |
| `galerkin.py` | `GalerkinROM` — intrusive projection of `K`/`M`/`F` onto a reduced basis |
| `affine.py` | `AffineDecomposition` — offline/online split for parameter-dependent systems |
| `frequency.py` | `FrequencyROM`, `build_pod_basis_from_frf_snapshots` |
| `greedy.py` | `greedy_train_frequency_basis` — adaptive basis training |
| `scm.py` | `SingularValueLowerBound`, `certified_error_bound` |
| `mode_correction.py` | `mode_acceleration_correction`, `mode_acceleration_response`, `augmented_basis` |
| `krylov.py` | `KrylovROM` — one- and two-sided moment matching |
| `balanced_truncation.py` | `BalancedTruncationROM`, `SingularPerturbationROM`, `FrequencyWeightedBalancedTruncationROM` — all three take `backend="numpy"`/`"torch"` (Wave 9 addendum item 138, see `torch_linalg.py`) |
| `hankel_norm.py` | `OptimalHankelNormROM` — same `backend=`/`device=` on its balancing step and its own empirical self-check |
| `torch_linalg.py` | Shared torch-native dense linalg primitives (`torch_solve_continuous_lyapunov`, `torch_solve_sylvester`, `torch_gramian_square_root`) `balanced_truncation.py`/`hankel_norm.py` dispatch to on `backend="torch"` |
| `loewner.py` | `LoewnerROM`, `screen_physical_modes` |
| `nonlinear_rom.py` | `MultiFidelitySurrogate`, `PolynomialModalROM`, `NeuralSurrogate`, `AppliedLoadStrategy` and other training strategies |
| `nonlinear_dynamics.py` | `integrate_newmark_surrogate` |
| `nnm.py` | `HarmonicBalanceSystem`, `solve_nnm_backbone` |
| `differentiable_correction.py` | The reduced-order differentiable-correction layer (see "Advanced capabilities") |
| `dataset_diagnostics.py`, `ensemble_uq.py` | Dataset completeness diagnostics; ensemble epistemic UQ |
| `neural_operator.py`, `reduced_basis_operator.py`, `parameterized_latent_ode.py` | The neural-operator-informed reduced-dynamics routes mentioned at the end of "ROM: nonlinear structural ROMs" — see `fea_engine_rom_engine_coupling.html` for the full explanation |

---

<a id="s-35"></a>

# ⚡ Multi-core, CPU and GPU (PyTorch) usage

Where the time goes decides what to tune. Read this section before you
reach for a GPU: for the model sizes in most examples above, **more CPU
threads or a better solver choice help more than a GPU does**.

<a id="s-36"></a>

## 1. What actually runs in parallel

| Part of the workflow | Parallel by default? | How to control it |
|---|---|---|
| Dense linear algebra (`numpy`/`scipy` LAPACK/BLAS: dense `solve`, `eigh`, SVD for POD, matrix products, Gramians) | Yes, multi-threaded BLAS | Thread environment variables (Section 2) |
| SciPy sparse direct solve (`splu`/`spsolve`) and sparse matrix-vector products | No, single-threaded | Use a dense path, `backend="torch"`, or run independent cases in separate processes |
| Element assembly (Python loops, or the vectorized batch path) | No, single-threaded Python/NumPy | Vectorized assembly (see "Architecture upgrades"); multiple processes for independent models |
| `rom_engine` frequency/amplitude sweeps (`frequency_sweep`, `amplitude_sweep` in `rom_engine.attractor`) | Optional | `n_jobs=N` (process pool). The model and arguments must be picklable, so use module-level functions, not lambdas or local closures. `n_jobs=1` (default) runs serially |
| PyTorch (`backend="torch"`) CPU kernels | Yes, intra-op threads | `torch.set_num_threads(N)` (Section 3) |
| PyTorch on GPU (`device="cuda"`) | Massively parallel on the GPU | Section 4 |

So "use multiple cores" means three different knobs: BLAS threads
(dense math), `n_jobs` (independent sweep points), and separate
processes (independent models or parameter cases that you launch yourself).

<a id="s-37"></a>

## 2. Setting the number of CPU threads (NumPy / SciPy)

BLAS reads its thread count **when NumPy is first imported**, so set it
before importing `numpy`, `fea_engine` or `rom_engine`. Set all four
variables, because different NumPy builds use different ones (OpenBLAS,
MKL/Anaconda, generic OpenMP):

```powershell
# Windows PowerShell -- current session only
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"
python my_script.py
```

```bash
# Linux / macOS
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python my_script.py
```

Or from inside a script, before the first `import numpy`:

```python
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np            # only now
from fea_engine import FESystem
```

Check what NumPy is actually using (needs `pip install threadpoolctl`):

```python
from threadpoolctl import threadpool_info
print(threadpool_info())      # lists the BLAS library and its num_threads
```

Measured example (2-core machine, one 2500 x 2500 dense matrix product):
`1` thread took `0.66 s`, `2` threads took `0.40 s`. Gains depend on matrix
size and BLAS build; very small systems (a few hundred DOF) get no benefit
and can even slow down from thread start-up cost.

Rules of thumb:

- Set the thread count to the number of **physical** cores, not logical (hyper-threaded) cores.
- If you also run several processes at once (`n_jobs`, several scripts, several Streamlit users), set threads per process so that `processes x threads <= physical cores`. Oversubscription slows everything down.
- On a laptop leave one core free so the machine stays responsive.

<a id="s-38"></a>

## 3. PyTorch on the CPU

```python
import torch
print(torch.get_num_threads())          # intra-op threads used by torch kernels
torch.set_num_threads(4)                # set once at the top of your script

sys = FESystem(mesh, elem, thickness=0.02, backend="torch")      # device="cpu" implicit
U = sys.solve_static()
```

- `backend="torch"` changes which routine runs: dense `torch.linalg.solve` for `sparse=False`, Jacobi-preconditioned sparse CG for `sparse=True` (SPD systems only). This is a different algorithm from the SciPy default, not just a different library, so for small or moderately sized models SciPy's direct solve is often as fast or faster. Measure before switching.
- `backend="auto"` picks for you at solve time: SciPy for small or non-SPD systems, torch CG for SPD systems with more than `5000` free DOF (`AUTO_DOF_THRESHOLD` in `fea_engine.backend_dispatch`). Call `solve_static(verbose=True)` to print which backend was chosen and why.
- Threads used by NumPy (Section 2) and by torch (`torch.set_num_threads`) are separate settings. Set both if you mix the two.
- Default precision of the torch routines is float64, the same as NumPy.

<a id="s-39"></a>

## 4. PyTorch on a GPU

### 4.1 Check the installation first

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())                       # must be True
print(torch.cuda.get_device_name(0))                   # e.g. your GPU model
print(torch.cuda.get_device_capability(0))             # compute capability, e.g. (8, 6)
print(torch.version.cuda)                              # CUDA version this torch build uses
```

If `is_available()` is `False`: you installed the CPU wheel (use the
`cuXXX` wheel from Installation), your NVIDIA driver is too old for that
CUDA build, or the machine has no NVIDIA GPU. AMD and Intel GPUs need a
different PyTorch build and are not covered or tested here. The
dashboard's *Environment* page shows the same torch/CUDA status.

### 4.2 Switching a solve to the GPU

Only `device=` changes; mesh, element, material, loads and BCs stay as they are:

```python
sys = FESystem(mesh, elem, thickness=0.02, backend="torch", device="cuda")
U = sys.solve_static()

# nonlinear transient / displacement control: GPU linear solve inside each Newton step
solve_nonlinear_transient(sys, mat, load, T_total, dt, backend="torch", device="cuda")

# many load cases against one stiffness matrix (one factorisation, many right-hand sides)
from fea_engine.batched_solve import solve_batched
X = solve_batched(K, B, backend="torch", device="cuda")

# ROM: balanced truncation / Hankel singular values on the GPU
from rom_engine.balanced_truncation import hankel_singular_values
hsv = hankel_singular_values(A, B, C, backend="torch", device="cuda")
```

Coverage is selective. Every torch dispatch point takes `backend=`
and/or `device=`; the list in the API Reference and the "Differentiability
and GPU backends" section above names them. `solve_modal`,
`solve_linear_buckling` and the explicit transient drivers remain
SciPy/NumPy-only.

### 4.3 When a GPU helps, and when it does not

| Situation | Expect |
|---|---|
| Small model (a few thousand DOF or less) | **Slower** on GPU: data transfer and kernel launch overhead dominate. Stay on the CPU |
| Large sparse SPD system (tens of thousands of DOF and up), iterative CG | Can be faster on GPU; this is the intended use |
| Many repeated solves with the data already on the device (Newton loops, parameter sweeps) | Transfer cost is amortised, so gains are more likely |
| Ill-conditioned systems, iterative solve without a good preconditioner | CG needs many iterations; a direct CPU solve may win |
| Consumer GeForce card in float64 (the default precision) | float64 throughput is a small fraction of float32 on these cards, so the speed-up can be modest or absent. Data-centre GPUs (A100/H100 class) do not have this penalty |
| float32 on GPU | Faster, but accuracy is limited to about 1e-6 relative; check the residual before trusting the result |

The honest workflow: build the same model with `backend="scipy"` and
with `backend="torch", device="cuda"`, compare the results (they should
agree to the tolerance you asked for) and the wall time, and keep the
faster one. A benchmark script is in
`fea_engine/examples/fea_scipy_vs_torch_gpu_benchmark.py`.

When timing a GPU run, call `torch.cuda.synchronize()` before reading the
clock, because CUDA kernels run asynchronously:

```python
import time, torch
torch.cuda.synchronize(); t0 = time.perf_counter()
U = sys.solve_static()
torch.cuda.synchronize(); print(time.perf_counter() - t0, "s")
```

### 4.4 Memory

A dense `n x n` float64 matrix needs `8 n^2` bytes, so 20,000 DOF is
about 3.2 GB for the matrix alone, and a dense solve needs a second
copy. Use `sparse=True` (CG path) for large models. If you see
`CUDA out of memory`, reduce the model, switch to the sparse path, or
use float32. `torch.cuda.memory_allocated()` shows current use.

### 4.5 Choosing the GPU, and several GPUs

```powershell
$env:CUDA_VISIBLE_DEVICES = "1"        # use only the second GPU; it then appears as cuda:0
```

`device="cuda:1"` also works where a function accepts a torch device
string. The packages do not split one solve across several GPUs; use
several GPUs by running independent cases in separate processes, each
with its own `CUDA_VISIBLE_DEVICES`.

### 4.6 Known problems

- **`OMP: Error #15` on Windows with Anaconda:** NumPy (MKL) and pip-installed PyTorch both load an OpenMP runtime. Quick workaround: `$env:KMP_DUPLICATE_LIB_OK = "TRUE"`. Cleaner: use a fresh environment.
- **Pascal / GTX 10-series GPUs:** PyTorch 2.8+ CUDA builds dropped them; use PyTorch 2.7.x with a matching `cuXXX` wheel (see Installation).
- **`import torch` fails on a machine without an NVIDIA driver:** you installed the default CUDA-linked wheel. Install the CPU wheel instead.
- **Results differ in the last digits between SciPy and torch:** expected (different summation order and algorithms). The test suite compares with tolerances, for example about `1e-6` relative for the CG paths.

<a id="s-40"></a>

## 5. Running the dashboard with several cores or a GPU

The dashboard reads the core count before NumPy loads:

```powershell
cd fea_rom_dashboard
$env:FEA_ROM_CORES = "4"               # sets the four BLAS thread variables for the dashboard process (a variable you already set yourself is not overwritten)
streamlit run streamlit_app.py
# or, after `pip install -e .`:   fea-rom-dashboard --cores 4
```

- Workflows that need torch are disabled automatically when torch is not importable, and the CUDA option appears only when `torch.cuda.is_available()` is true (see the *Environment* page). The Model builder and Dual-stage pages use NumPy/SciPy only, so they need no GPU.
- The dashboard's snapshot loop is serial; extra cores help through BLAS threads in the dense solves, not through parallel snapshots.
- All users of one Streamlit server share its CPU cores, so set cores per server to match the number of simultaneous users.
- The in-browser real-time ROM viewer runs on the user's own machine and needs no server cores while sliders move.

<a id="s-41"></a>

## 6. Quick decision guide

| You want to... | Do this |
|---|---|
| Speed up dense-matrix-heavy steps (POD, Gramians, modal, dense solves) | Raise BLAS threads (Section 2) |
| Run a frequency/amplitude sweep faster | `n_jobs=N` on `rom_engine.attractor` sweeps; lower BLAS threads per process |
| Run many independent models/cases | Separate processes with `OMP_NUM_THREADS=1` each |
| Solve a large SPD sparse system | Try `backend="auto"`, then `backend="torch", sparse=True`; add `device="cuda"` if a GPU is available |
| Solve a small or moderate model | Stay on the SciPy default |
| Get exact tangents for a nonlinear solve | `method="autograd"` (needs torch; improves accuracy, not speed) |
| Train a differentiable correction | torch required; GPU optional |
| Diagnose slowness | `solve_static(verbose=True)`, `threadpool_info()`, `torch.cuda.is_available()` |

---

<a id="s-42"></a>

# 📈 Performance

Every number below came from an actual run against a real model in this
repository (see the linked section for the full script/context) —
nothing here is estimated.

| Capability | Result |
|---|---|
| Static solve accuracy, iterative vs. direct | Unpreconditioned CG: 116 iterations; SSOR-preconditioned CG: 58 iterations; both agree with the direct solve to `2.2e-15` absolute ("Batched workflows") |
| Adaptive mesh refinement | 5 refinement steps, 96 → 486 elements, estimated error `8.818e-03` → `3.970e-03`, monotonically ("Batched workflows") |
| Explicit nonlinear transient | 201 steps, `dt ≈ 4.15e-06 s`, `2.16 s` wall time, peak tip displacement `≈ 6.742e-06 m` ("Time integration") |
| ROM: linear affine sweep | 500-point parameter sweep: full-order `1653.23 ms` vs. reduced-order `7.45 ms` → **221.8x speedup**, max relative error `1.51e-03` ("ROM: linear workflow") |
| ROM: POD/Galerkin accuracy | 8 snapshots, 6-mode basis, 100.0% training energy captured, `~7e-6` absolute error vs. full-order static solve ("ROM: linear workflow") |
| ROM: Krylov moment matching | One-sided: `1.53e-01` mean relative error far from the expansion point; two-sided: `1.15e-02` ("ROM: classical MOR") |
| ROM: balanced truncation | `<2.35e-03` max relative error across a 1–1500 rad/s sweep; computed H-infinity bound `2.34e-09`; provably stable ("ROM: classical MOR") |
| ROM: mode acceleration / augmented basis | Plain mode displacement: `6.18e-03` relative error (3 modes); mode acceleration and augmented-basis correction: machine precision ("ROM: classical MOR") |
| GPU/autograd tangents (developer-machine record) | `method="autograd"` vs. finite-difference/complex-step: `1.25e-16` relative (shell case); `backend="torch"` vs. `backend="scipy"` transient: `~1e-9` relative (linear limit), `~1e-6` relative (nonlinear) ("Differentiability and GPU backends") |
| GPU torch side-by-side, Wave 9 addendum (developer-machine record) | `MeshTransformation`/`assemble_stiffness_vectorized` `backend="torch"`: 11/11 tests each, agreeing to `atol=1e-3` (summation-order noise at 1e11 scale); `preconditioned_cg`/`solve_static(method="cg"/"pcg")` `backend="torch"`: 9/9 then 17/17, `~1e-6` relative; `rom_engine` `balanced_truncation`/`hankel_norm` `backend="torch"`: 21/21 then 13/13, `~1e-6` relative ("Batched workflows", "Architecture upgrades", "ROM: classical MOR") |

For raw benchmark scripts and scaling behavior beyond what's captured
here, see `fea_engine/examples/` (sparse-vs-dense timing, higher-order
elements) and each package's own test suite, which is what produced
every number above.

---

<a id="s-43"></a>

# 🧭 Where to go deeper

- `fea_engine/README.md` — the full module reference, every validation,
  every design decision.
- `fea_engine/examples/` — runnable scripts for buckling, arc-length
  snap-through, plasticity, hyperelasticity, shells, 3-D frames,
  sparse-vs-dense timing, higher-order elements.
- `fea_engine/examples/gallery/` — the numbered `fea_01_...`–`fea_06_...`
  gallery (see its own `fea_README.md`) and the sibling
  `fea_scipy_vs_torch_gpu_benchmark.py` solver-backend benchmark.
- `fea_engine/docs/geometry_meshing_alternatives_research.md` — how to
  bring in CAD-authored geometry (CadQuery, FreeCAD, Gmsh's own GUI) and
  quadratic meshing.
- `fea_engine/docs/consolidated_future_roadmap.md` — the full
  development history with full validation detail.
- `rom_engine/README.md` — the full module reference and every
  validation result (all 208 tests summarized).
- `rom_engine/examples/` — the six complete, runnable scripts listed in
  the Example Gallery above.
- `rom_engine/examples/gallery/` — the numbered `rom_01_...`–`rom_06_...`
  gallery mirroring `fea_engine`'s (see its own `rom_README.md`).
- `rom_engine/docs/` — the phased design/roadmap documents for
  frequency-domain ROM, greedy training + error bounds, non-intrusive
  identification, the nonlinear-surrogate-ROM family, and classical
  structural-dynamics/systems-and-control MOR
  (`classical_mor_roadmap.md`).
- `fea_engine_rom_engine_coupling.html` (this folder) — a general-
  purpose, term-by-term explanation of how the two packages couple, the
  torch-optional switch pattern, and every ROM formulation route in the
  package, including the newer neural-operator-informed routes not
  covered in day-to-day detail here.
