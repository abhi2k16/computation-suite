<div align="center">

# fea_engine

**Finite element analysis for structural mechanics: linear, nonlinear, dynamic, contact, plasticity**

![version](https://img.shields.io/badge/version-1.0.1-2563eb?style=for-the-badge)
![status](https://img.shields.io/badge/status-beta-f59e0b?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.9+-3776ab?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-1.22+-013243?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-1.8+-0054a6?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-optional-ee4c2c?style=for-the-badge)
![license](https://img.shields.io/badge/license-MIT-16a34a?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Behaviour changes in the interface (v1.0.1).** Silent mistakes now fail loudly; valid calls are unchanged.
> `assemble_stiffness` / `assemble_mass` / `assemble_lumped_mass` called a second time now **replace** the matrix
> instead of silently doubling it (one-time `UserWarning`; pass `accumulate=True` to add on purpose).
> `fix_dofs`, `add_nodal_force`, `add_consistent_edge_load` and `add_consistent_facet_load` raise `ValueError` for an empty
> node selection, a node id outside the mesh, or a local DOF index outside `0..dofs_per_node-1`.
> `solve_static` raises if `K` was never assembled and warns when the load vector is all zeros; a scalar density on a
> plane or solid element raises an error that says to pass `rho * np.eye(n)`. See `tests/test_api_safety.py`.

> [!NOTE]
> **Named access (v1.0.1, additive).** Integer DOFs and node-id arrays still work; you can now also write
> ```python
> mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=0.4, name="tip")
> s.fix_dofs("root", ["ux", "uy"]); s.add_nodal_force("tip", "uy", -20000.0)
> U = s.solve_static()                      # FEField: an ndarray with names
> U.component("uy", nodes="tip")            # == U[2*tip_nodes + 1]
> U.nodal.shape, U.magnitude()              # (n_nodes, dofs_per_node), per-node |u|
> ```
> `FESystem.dof_names` lists the names for the current element; see `tests/test_named_api.py`.

> [!NOTE]
> **Shared options and batch runs (v1.0.1, additive).**
> ```python
> from fea_engine import NewtonOptions, batch
> opts = NewtonOptions(tol=1e-9, max_iter=40)          # None fields keep each driver's own default
> lf, U = solve_nonlinear_static(system, mat, n_steps=20, options=opts)   # explicit keywords still win
> results = batch.map(build_and_solve, params, n_jobs=4)                  # ordered; backend="thread" for lambdas
> ```

> [!NOTE]
> **Unit labels and plots (v1.0.1, additive).** Labels only, nothing is converted.
> ```python
> system.units = "SI"                       # or fea_engine.units.MM_N_TONNE
> U = system.solve_static()                 # U.units == "m"
> U.plot("uy", deform=True, scale=100)      # contour on the mesh, colour bar "uy [m]"
> ```

> [!NOTE]
> **Multi-step results (v1.0.1, additive).** One access style for load paths, time histories and modes:
> ```python
> lf, hist = solve_nonlinear_static(system, mat, n_steps=20)      # return values unchanged
> path = system.series(hist, steps=lf, step_name="load factor")
> path.history("uy", "tip", reduce="mean")                        # load-displacement curve
> path.at(0.5).plot("uy")                                         # step nearest to 0.5
> system.modal_series(5).at(120.0)                                # mode closest to 120 Hz
> ```

## 🧭 Contents

| | |
|---|---|
| 🗂️ [Package layout](#s-1) | 🪜 [Implementation steps](#s-4) |
| 🔀 [How a problem flows through the modules](#s-2) | 🧰 [General-purpose extensions (Phases 1-7)](#s-5) |
| 📚 [Module reference](#s-3) | ⚠️ [Known limitations (by design, not oversights)](#s-6) |

---


A small, from-scratch finite element package for structural mechanics —
static, dynamic, and (as of Module 8) geometrically nonlinear static —
built around one rule: **every axis of variation
(new material, new geometry, new element, new load, new solution
strategy) lives behind a registry entry or a subclass, so extending the
package means *adding* a function/class, not editing existing ones.**

Every number this package produces has been checked against an
independent reference (closed-form beam/plate theory, a previously
validated script, or a second numerical method) — see `main.py`, which
doubles as a regression test.

<a id="s-1"></a>

## 🗂️ Package layout

> [!NOTE]
> **Restructuring note:** this package started as a flat, single-directory
> development layout (`fea_package/`, one file per module, run via
> `sys.path.insert` from whatever script needed it). It has since been
> reorganized into a standard `src/`-layout, pip-installable package
> (`pip install -e .`) named `fea_engine`, with the one large
> `element.py` split into a small `elements/` subpackage (one file per
> element family) purely for navigability. **No solver logic changed
> during this restructuring** — every class and function was moved
> verbatim and diffed byte-for-byte against the original before and
> after the move; only file organization, import paths, and packaging
> metadata changed. Some code snippets elsewhere in this README predate
> the restructuring and may still show the old flat import style
> (`from fea_engine import element as elmod`) — the mapping is
> mechanical: `element.X` → `elements.X` for any element class,
> `config.Material/Section/PlasticMaterial1D/D_*` → `material.*`, and
> `config.RayleighDamping/ModalDamping` → `damping.*`.

```
fea_engine/                     (project root)
├── pyproject.toml
├── src/fea_engine/
│   ├── __init__.py    flat re-export namespace, same as before
│   ├── material.py    Module 1 (part) — materials, section properties,
│   │                   constitutive (D) matrices
│   ├── damping.py     Module 1 (part) — Rayleigh and modal damping models
│   ├── mesh.py        Module 2 — mesh generation and boundary-node selection
│   ├── grading.py     geometry-agnostic mesh-grading/sizing-function math
│   │                   shared by mesh.py (see
│   │                   docs/generalized_mesh_grading_roadmap.md)
│   ├── build_mesh.py  grading-aware mesh dispatcher (build_mesh()) — decides
│   │                   whether/how to grade from a feature list and refuses
│   │                   rotational-DOF elements on unverified curved/
│   │                   unstructured topology
│   ├── geometry/
│   │   └── shapes.py       Module 11 — dimension-driven entry point (dim ->
│   │                        mesh -> default element -> FESystem, one call)
│   │                        (Module 12, gmsh_engine.py, a Gmsh-driven
│   │                        unstructured mesh generator, was removed --
│   │                        see "Removed: Gmsh support" below)
│   ├── visualization/  (Module 16, gmsh_plot.py, a Gmsh-native rendering
│   │                    path, was removed -- see "Removed: Gmsh support"
│   │                    below; mesh.py's matplotlib plot_mesh_* functions
│   │                    remain the supported rendering path)
│   ├── elements/       Module 3 — Gauss-Legendre integration engine +
│   │   │                element formulations, split by family (no logic
│   │   │                changed, see restructuring note above)
│   │   ├── base.py         Element base class + quadrature engine
│   │   ├── solids.py       Quad4PlaneStress, Hex8Solid3D, Tri3PlaneStress,
│   │   │                    Tet4Solid3D
│   │   ├── plates.py       Quad4MindlinPlate
│   │   ├── beams.py        Beam2DEulerBernoulli, Beam2DCorotational
│   │   ├── trusses.py      TrussTL2D, TrussPlastic2D
│   │   └── contact.py      GapContactPenalty, GapContactCurvedFriction
│   ├── loads.py        Module 6 — load definitions (static, time-history,
│   │                    harmonic, PSD)
│   ├── solver.py       Module 4 — global assembly, boundary conditions,
│   │                    static/modal/dynamic solve, + generic nonlinear
│   │                    assembly (assemble_internal_force/
│   │                    assemble_tangent_stiffness)
│   ├── postprocess.py  Module 7 — derived response quantities (modal
│   │                    participation, PSD statistics)
│   └── nonlinear_solver.py  Modules 8–10 — incremental Newton-Raphson
│                        drivers (geometric nonlinearity: load control +
│                        prescribed-displacement control for snap-through;
│                        material nonlinearity: the same drivers plus
│                        per-element path-dependent state commit;
│                        contact/boundary nonlinearity: penalty contact
│                        reuses the same drivers unmodified, + a dedicated
│                        Lagrange-multiplier augmented-KKT driver)
├── tests/              pytest suite -- one test module per validate_*.py
│                        script from the original layout (same checks,
│                        same asserts, now pytest-collected)
└── examples/           Module 5 — validation driver / example pipelines
                         (main.py, hex8_convergence.py)
```

Numbering follows the order the modules were introduced in, not their
position in the pipeline — `loads.py` and `postprocess.py` were added
later, once the four dynamics solution strategies needed richer load
objects and derived quantities than the original static-only design
required; `nonlinear_solver.py` is the newest, added when the package
was extended past linear elasticity for the first time.

<a id="s-2"></a>

## 🔀 How a problem flows through the modules

```
config          mesh              element             solver
--------        --------          --------            --------
Material   ─┐   geometry mesh ─┐  element formula- ─┐  assemble K (+M,+C)
Section     ├─► (nodes,        ├─► tion (shape       ├─► apply loads/BCs
Damping     │    elements)     │   funcs, B, stiff-  │   solve (static /
D-matrix   ─┘                 ─┘   ness, mass)       │   modal / transient /
                                                       │   harmonic / random)
                                                      ─┘
                                                          │
                                                          ▼
                                                     postprocess / main
                                                     (derived quantities,
                                                      plots, validation)
```

`solver.py` never asks "is this a Quad4 or a Hex8 or a beam?" — it only
calls `element.stiffness()` / `element.mass()` and reads
`element.dofs_per_node`. That's what makes it possible to run four
physically different problems (1-D beam, 2-D plane stress, 2-D plate,
3-D solid) through the *same* `solver.py`, unmodified, and to add four
dynamics solution strategies to `solver.py` without touching
`config.py`, `mesh.py`, or `element.py` at all.

---

<a id="s-3"></a>

## 📚 Module reference

### `config.py` — materials, sections, constitutive matrices, damping

| Object | Purpose |
|---|---|
| `Material(E, nu, rho=0.0)` | Isotropic linear-elastic material. `.G` is derived, never stored separately. |
| `Section(A, I)` | Cross-section area/second-moment for 1-D beam elements. |
| `D_plane_stress(mat)`, `D_plane_strain(mat)`, `D_solid3d(mat)` | Constitutive matrices for 2-D and 3-D continuum elements. |
| `D_mindlin_plate(mat, h, k_shear=5/6)` | Returns `(Db, Ds)` — bending and shear matrices, kept separate because the plate element integrates them with different Gauss orders (selective reduced integration). |
| `EI_beam(mat, sec)` | Bending rigidity for the 1-D beam element. |
| `CONSTITUTIVE_REGISTRY` | Name → builder-function lookup. |
| `RayleighDamping(alpha, beta)` | `C = alpha*M + beta*K`. `.calibrate(omega_i, omega_j, zeta)` solves for `(alpha, beta)` giving a target damping ratio at two modal frequencies (rad/s). `.modal_ratio(omega)` reports the resulting ratio at any frequency. |
| `ModalDamping(zeta)` | A ratio (or array of ratios) applied directly in `solve_modal_superposition()` — no global `C` is ever built for this path. |

**To add a new physics:** write a `D_xxx(material, ...)` function and
add one line to `CONSTITUTIVE_REGISTRY`.

### `mesh.py` — geometry and boundary-node selection

`Mesh(nodes, elements, dim)` is a plain container with four methods:
`.nodes_on_line(axis, value, tol)` (2-D), `.nodes_on_plane(axis, value,
tol)` (3-D), `.check_quality(elem_formulation)`, and `.check_grading
(growth_ratio_cap=None)`. `check_quality` evaluates the Jacobian at
every Gauss point of every element using *that element's own* shape
functions, so it works for any registered element with no per-shape
special-casing. `check_grading` (2-D only for now) is the complementary
diagnostic: the realized max neighbor-to-neighbor element-size ratio —
a mesh can have every element individually well-shaped (`check_quality`
green) and still jump abruptly in size at a boundary, which
`check_quality` alone never flags. With no cap it returns the ratio
itself; with `growth_ratio_cap` given it returns a pass/fail bool, like
`check_quality` does.

| Function | Geometry |
|---|---|
| `line_mesh(L, n)` | 1-D, for beam elements |
| `rectangle_mesh(Lx, Ly, nx, ny)` | 2-D rectangle |
| `rectangle_with_hole_mesh_quarter(a, b, R, nr, ntheta, grade_p=2.0)` | One quadrant of a rectangle with a circular hole, mapped/transfinite mesh, graded toward the hole |
| `rectangle_with_hole_mesh_quarter_full(a, b, R, nr, ntheta, grade_p=2.0, n_extend=None, extend_grade=2.0)` | Fixes the quadrant mesher for `a != b`: near-hole block welded to a graded far-field extension so an elongated (not just square) quadrant is still fully resolved |
| `hole_in_rectangle_mesh(Lx, Ly, hole_center, R, nr, ntheta, grade_p=2.0, n_extend=None, extend_grade=2.0)` | A hole ANYWHERE inside a (not necessarily symmetric) rectangle — 4 quadrant blocks built from the function above and welded together; generalizes hole *placement*, not just quadrant shape (see "Mesh grading" below) |
| `hole_in_rectangle_mesh_graded(Lx, Ly, hole_center, R, nr, ntheta, h_far, growth_ratio=1.2, n_extend=None)` | Same, but `grade_p`/`extend_grade` are DERIVED from a target growth ratio via `grading.grade_for_growth_ratio()` instead of hand-picked |
| `weld_meshes(mesh_a, mesh_b, tol=1e-9)` | Merges two 2-D Quad4 meshes sharing coincident boundary nodes into one connected mesh |
| `mirror_mesh(mesh, mirror_x, mirror_y)` | Mirrors a 2-D mesh and welds nodes on the mirror line (real FE mesh merge, not just a plotting trick) |
| `extrude_mesh(mesh2d, Lz, nz)` | Turns **any** Quad4 mesh into a Hex8 mesh — this one function is what makes `box_mesh` and `box_with_hole_mesh` one-liners |
| `box_mesh(Lx, Ly, Lz, nx, ny, nz)` | 3-D block |
| `box_with_hole_mesh(a, b, R, Lz, nr, ntheta, nz)` | 3-D block with a cylindrical bore |
| `plot_mesh_2d` / `plot_mesh_3d` | Wireframe QA plots (matplotlib `Axes`/`Axes3D` in, nothing returned) |
| `plot_mesh_annotated(mesh, annotations=None, title="", figsize=(8,6))` | 2-D mesh plot with highlighted node groups (BCs, loads, symmetry/free edges) drawn on top — each `annotations` entry is a dict with `nodes` (required) plus any of `label`/`color`/`marker`/`connect`/`linestyle`/`hatch`/`arrow`/`arrow_scale`; returns `(fig, ax)`, not saved/shown (caller's job, same convention as every other plot function here) |

**To add a new geometry:** write one function that returns a `Mesh`.

**`MultiBlockMesh(nodes, blocks, dim)`** (Module 14) is a SEPARATE
container for a mesh with more than one element TOPOLOGY sharing one
node array (`blocks: {block_name: connectivity_array}` — `Mesh.elements`
is a single rectangular array and can't hold rows of different width,
which is why this isn't just a generalization of `Mesh`). Same
`.nodes_on_line`/`.nodes_on_plane` selectors; `.check_quality(elem_
formulations)` takes a dict matching `blocks`' keys instead of one
formulation. Formerly built automatically by the now-removed
`geometry_engine.py`'s `_extract_mesh()` when Gmsh reported more than
one element type (see "Removed: Gmsh support" below); still built
directly from a hand-constructed `blocks` dict for any mixed mesh
today. See `solver.py`'s `FESystem` for how it pairs with a dict of
element formulations to assemble.

### `grading.py` + `build_mesh.py` — geometry-agnostic mesh grading

Added 2026-09-03 (see `docs/generalized_mesh_grading_roadmap.md` for the
full design/research writeup). Two problems, one module each:

**`grading.py` is pure sizing-function math** — no dependency on Gmsh,
mesh topology, or any element formulation. `graded_partition(start, end,
n, grade, dense_at)` is the 1-D power-law grading law `mesh.py` has
always used (moved here, `mesh.py` still re-exports it unchanged).
`growth_ratio_of_partition(coords)` measures the realized max
adjacent-segment ratio; `grade_for_growth_ratio(n, growth_ratio,
dense_at)` solves (by bisection) for the grading exponent that respects
a target growth ratio — 1.2 is the commonly-cited industry target, up to
~1.5 still acceptable — instead of hand-picking `grade_p` and checking
after the fact. `Hole`/`Fillet`/`EdgeBias`/`Notch` are small declarative feature
dataclasses — `Notch` (Wave 5 item 26) is a rectangular slot cut inward
from a straight edge, deliberately narrower than an arbitrary path (see
its own docstring); `mesh.fillet_in_rectangle_mesh()`/`mesh.notch_
in_rectangle_mesh()` (Wave 5 item 27) are the structured meshers that
consume `Fillet`/`Notch`. `plan(features,
h_far, growth_ratio)` combines a feature list into a `GradingPlan`,
warning (not erroring) when `growth_ratio` exceeds ~1.5, and when a
feature's own target size doesn't make sense against `h_far`.
`threshold_field_size`/`dist_max_for_growth_ratio` are the Gmsh-
Threshold-field-matching analogue; they were consumed by
`generate_2d_plate_with_hole_graded` (`geometry/gmsh_engine.py`, now
removed, see "Removed: Gmsh support" below) but remain here as pure,
independently-testable sizing-function math with no Gmsh dependency of
their own.

**`build_mesh.py` is the dispatcher** — the piece that decides
*automatically*, from a feature list, whether/how to grade, instead of
the caller having to notice "this geometry is complex" and reach for a
differently-named function:

```python
from fea_engine.build_mesh import build_mesh, RectangleWithHole
from fea_engine import Shell4MITCCorotational, Tri3PlaneStress

geom = RectangleWithHole(Lx=1.0, Ly=1.0, hole_center=(0.5, 0.5),
                          hole_radius=0.1, n_ring=12)

# Rotational-DOF element -> ALWAYS routed to the structured (Quad4),
# axis-aligned-block path -- the only one verified safe for elements
# with a local bending frame (see the roadmap doc's Section 2).
m_shell = build_mesh(geom, target_size=0.15,
                      element_formulation=Shell4MITCCorotational)

# Translational-DOF element -> used to route to the unstructured Gmsh
# field-graded path (Tri3). That path was removed along with Gmsh
# support entirely (see "Removed: Gmsh support" below); this now raises
# MeshGradingError explaining the removal instead of generating a mesh.
m_solid = build_mesh(geom, target_size=0.15,
                      element_formulation=Tri3PlaneStress)  # raises MeshGradingError
```

A plain `RectangleWithHole(Lx, Ly)` with no `hole_center` has no
features, so `build_mesh()` falls straight through to a plain
`rectangle_mesh` — grading is opt-in-by-geometry, never forced on a
simple case. `ROTATIONAL_DOF_ELEMENTS = (Quad4MindlinPlate, Shell4MITC,
Shell4MITCCorotational)` is the explicit, commented registry the safety
gate checks against — deliberately not inferred from `dofs_per_node`
(beam elements also carry rotational DOFs but have no 2-D
local-frame-misalignment mechanism to worry about). `build_mesh()` the
FUNCTION is intentionally not flattened onto `fea_engine.build_mesh`
(that name is already the module) — call it as `fea_engine.build_mesh.
build_mesh(...)` or `from fea_engine.build_mesh import build_mesh`.

`geometry/gmsh_engine.py` used to have a matching unstructured method,
`generate_2d_plate_with_hole_graded(Lx, Ly, r, lc_far, growth_ratio=1.2,
n_ring=12, use_curvature=True)` — real Gmsh `Distance`+`Threshold`
background fields (plus `CharacteristicLengthFromCurvature`) instead of
the older `generate_2d_plate_with_hole`'s two flat size constants with
no controlled transition between them. That module was removed along
with Gmsh support entirely — see "Removed: Gmsh support" below.

### `geometry.py` — Module 11: one dimension-driven entry point

Every generator above is independently usable; `geometry.py` adds a
single front door on top: say what dimension your problem is (1, 2, or
3) and, optionally, a shape and a physics, and it picks a matching
mesh generator AND a matching default element, wires them into a
ready-to-use `FESystem`, and hands all three back — collapsing the
usual mesh → element → `FESystem` three-liner (README section A) into
one call for the common cases.

| Function | Returns | Notes |
|---|---|---|
| `generate_mesh(dim, shape=None, mirror=None, **kwargs)` | `Mesh` | `shape` defaults to `'line'`/`'rectangle'`/`'box'`; `kwargs` pass straight through to the underlying `mesh.py` generator, unmodified |
| `default_element(dim, shape=None, physics=None)` | An `Element` instance | `physics` picks WHICH behavior on that geometry — e.g. `dim=2` can be `'plane_stress'`/`'plane_strain'` (2 dof/node) or `'plate'` (3 dof/node) on the same rectangle mesh; `dim=1` can be `'beam'`, `'truss'` (`TrussTL2D`), or `'truss_plastic'` (`TrussPlastic2D`) on the same line mesh |
| `build_system(dim, shape=None, physics=None, thickness=1.0, mirror=None, **mesh_kwargs)` | `(fesystem, mesh, elem)` | The full pipeline in one call |

```python
sys, mesh, elem = build_system(dim=2, Lx=0.4, Ly=0.2, nx=24, ny=12)   # plane_stress rectangle
sys.assemble_stiffness(D_plane_stress(mat), thickness=0.02)
tip = mesh.nodes_on_line(axis=0, value=0.4)
sys.add_nodal_force(tip, dof_index=1, total_force=-20000.0)
sys.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
U = sys.solve_static()
```

**dim=1 is a special case worth knowing about:** `mesh.line_mesh()`
itself returns 1-column (x-only) node coordinates — all
`Beam2DEulerBernoulli` needs. `generate_mesh(dim=1, ...)` instead
embeds the line into 2-D (appends a zero-valued y column) so the SAME
mesh also works directly with `TrussTL2D`/`TrussPlastic2D`/
`GapContactPenalty` (2 dof/node) — standard FEA practice (line
elements embedded in a higher-dimensional space), and harmless for the
beam element, which only ever reads the x column. `validate_geometry.py`
CHECK 2 demonstrates this: a `dim=1, physics='truss'` mesh feeding
straight into `nonlinear_solver.solve_nonlinear_static()`, matching
Module 8's already-validated closed form, with no reshaping by the
caller.

Unknown `dim`/`shape`/`physics` raise a `ValueError` listing what IS
registered — checked explicitly (`validate_geometry.py` CHECK 7), not
just assumed to fail gracefully.

**To add a new geometry or default element:** add one entry to
`GEOMETRY_REGISTRY` or `ELEMENT_DEFAULTS` — the same registry
convention as `CONSTITUTIVE_REGISTRY`/`ELEMENT_REGISTRY`.

### Removed: Gmsh support

This package formerly included two Gmsh-backed, optional modules:
`geometry/gmsh_engine.py` (Module 12 — Gmsh-driven unstructured mesh
generation, including `generate_from_step()` STEP/IGES/BREP import and
quadratic-element extraction) and `visualization/gmsh_plot.py`
(Module 16 — Gmsh-native OpenGL geometry/mesh/result screenshot
rendering, a second, non-default rendering path). Both required
`pip install gmsh` plus, on a minimal Linux install, the system libGLU
library. That system dependency could not be reliably provided, and no
verified example, result, or default code path in this package used
either module — every plot `gmsh_plot.py` produced already had a
matplotlib equivalent (`mesh.py`'s `plot_mesh_2d`/`plot_mesh_3d`/
`plot_mesh_annotated`) actually used by every example script. Both
modules, and their gmsh-exclusive tests, were removed entirely; see
`docs/generalized_mesh_grading_roadmap.md` for the fuller removal note.
The structured mesh front end (`geometry/shapes.py`, `mesh.py`) remains
the supported path for all meshing needs in this package.

### `element.py` — the Gauss-Legendre engine and element formulations

`gauss_legendre(n)` wraps `numpy.polynomial.legendre.leggauss` (any
order, not a hardcoded 2-point table); `gauss_product(n, dim)` builds
the tensor-product rule for 1-D/2-D/3-D.

The base `Element` class implements the Gauss loop **once**:

```python
def stiffness(self, elem_coords, D, thickness=1.0):
    ke = 0
    for point, weight in gauss_product(self.gauss_order, self.dim):
        B, detJ = self.B_matrix(point, elem_coords)
        ke += (B.T @ D @ B) * detJ * weight * thickness
    return ke
```

Any element that implements `shape_and_derivs()` and `B_matrix()` gets
`stiffness()`, `mass()`, and `lumped_mass()` for free. Elements that
need something the generic loop can't express **override** the method:

| Element | DOFs/node | Notes |
|---|---|---|
| `Quad4PlaneStress` | 2 (u, v) | Uses the generic loop as-is |
| `Hex8Solid3D` | 3 (u, v, w) | Uses the generic loop as-is |
| `Quad4MindlinPlate` | 3 (w, βx, βy) | Overrides `stiffness()` for selective reduced integration (full 2×2 bending, 1-pt shear — avoids shear locking) |
| `Beam2DEulerBernoulli` | 2 (v, θ) | Overrides `stiffness()`/`mass()` with the closed-form Hermite matrices (EI is constant per element, so there's nothing to numerically integrate) |
| `TrussTL2D` | 2 (u, v) | **Geometrically nonlinear** — see below; overrides `internal_force()`/`tangent_stiffness()` instead of `stiffness()` |
| `Beam2DCorotational` | 3 (u, v, θ) | **Geometrically nonlinear, Module 15** — see below; the bending counterpart to `TrussTL2D` (large displacement/large ROTATION, not just axial stretch); `tangent_stiffness()` is a hand-derived closed form, not finite-difference |
| `TrussPlastic2D` | 2 (u, v) | **Materially nonlinear** (elasto-plastic) — see below; small-displacement kinematics, path-dependent (`init_state()`/`commit_state()`) |
| `GapContactPenalty` | 2 (u, v) | **Contact/boundary nonlinear** — see below; 1-node unilateral gap element vs. a fixed FLAT obstacle, frictionless, stateless (status depends only on current position) |
| `Tri3PlaneStress` | 2 (u, v) | 3-node constant-strain triangle (CST) — see "Simplex elements" below; overrides `stiffness()`/`mass()` directly, does NOT use the generic Gauss loop |
| `Tet4Solid3D` | 3 (u, v, w) | 4-node linear tetrahedron — 3-D counterpart to `Tri3PlaneStress`, same reasoning |
| `GapContactCurvedFriction` | 2 (u, v) | **Contact/boundary nonlinear, Module 13** — see below; 1-node contact vs. a fixed CIRCULAR obstacle, Coulomb friction (stick/slip) + updating normal, path-dependent (reuses `TrussPlastic2D`'s `init_state()`/`commit_state()` pattern); `tangent_stiffness()` is finite-difference, not closed-form |
| `NodeToSegmentContact2D` | 2 (u, v), 3 NODES | **Wave 3 item 16 (`docs/consolidated_future_roadmap.md`)** — `elements/contact.py`; frictionless penalty contact between a slave node and a MASTER SEGMENT of two ordinary (deformable) mesh nodes — see "Node-to-segment contact" below. 3-node element (slave, master_a, master_b); force distributed to master nodes by the closest-point projection's own shape-function weights `(1-t, t)`, so `f_slave+f_master_a+f_master_b=0` identically. `tangent_stiffness()` is finite-difference (same precedent as `Tet4NeoHookean`/`GapContactCurvedFriction`). |
| `NodeToSegmentContact2DFriction` | 2 (u, v), 3 NODES | **Wave 3 item 17 (`docs/consolidated_future_roadmap.md`)** — `elements/contact.py`; subclasses `NodeToSegmentContact2D`, adds Coulomb stick/slip friction along the (possibly deforming) segment — the direct node-to-segment generalization of `GapContactCurvedFriction`'s stick/slip return map. `NodeToSegmentContact2D` itself remains available unchanged as the frictionless sibling. |
| `Quad8PlaneStress` | 2 (u, v) | **Module 17 (general-purpose extensions Phase 1)** — 8-node serendipity quadratic quad; uses the generic loop as-is (`gauss_order=3`). Captures pure bending almost exactly at a coarse element count where `Quad4PlaneStress` shear-locks — see `examples/higher_order_elements_demo.py` |
| `Hex20Solid3D` | 3 (u, v, w) | **Module 17** — 20-node serendipity quadratic hex; uses the generic loop as-is (`gauss_order=3`). **`GMSH_NODE_ORDER`** (found + fixed 2026-08-30) reindexes a raw Gmsh Hexahedron20 (element type 17) connectivity array into this class's own node order — Gmsh's native order does NOT match directly (it interleaves mid-edge nodes per-corner, not bottom/top/vertical); using Gmsh's raw output unreindexed silently produces a non-positive-definite mass matrix. See the class's own "GMSH COMPATIBILITY WARNING" docstring note (the validating `tests/test_gmsh_node_order.py`, built against real Gmsh-generated reference elements, was removed along with Gmsh support -- see "Removed: Gmsh support" above; the `GMSH_NODE_ORDER` table itself is independent, static data and remains). |
| `Tet10Solid3D` | 3 (u, v, w) | **Module 17** — 10-node quadratic tetrahedron; overrides `stiffness()`/`mass()` with a dedicated 4-point SIMPLEX quadrature (`tet_quadrature_4pt()` in `elements/base.py`) — the generic Gauss loop assumes a cube-shaped reference domain, wrong for a tetrahedron; see `Tet10Solid3D`'s docstring. **`GMSH_NODE_ORDER`** (found + fixed 2026-08-30): Gmsh's native 10-node tetrahedron (element type 11) order is CLOSE to but not identical to this class's own — corners and the first four mid-edge nodes match, but two mid-edge nodes are swapped; a subtler bug than Hex20's (doesn't crash the solver, but silently biases stiffness/mass). See the class's own "GMSH COMPATIBILITY WARNING" docstring note (the validating test was removed along with Gmsh support -- see "Removed: Gmsh support" above; the `GMSH_NODE_ORDER` table itself is independent, static data and remains). |
| `Beam3DEulerBernoulli` | 6 (u, v, w, θx, θy, θz) | **Module 18 (general-purpose extensions Phase 3)** — 2-node 3-D frame element (`elements/beams3d.py`); closed-form, no quadrature. Axial + torsional 2×2 blocks plus TWO independent bending blocks reusing `Beam2DEulerBernoulli`'s own Hermite stiffness/mass formula (one directly, one via a sign-flip congruence transform for the out-of-plane axis), rotated to global coordinates via a right-handed local frame (`elements/beams3d._local_axes()`) built from the member axis + a reference "up" vector |
| `Hex8PlasticJ2` | 3 (u, v, w) | **Module 19 (general-purpose extensions Phase 6)** — `elements/nonlinear_solids.py`; subclasses `Hex8Solid3D`, inherits `shape_and_derivs()`/`B_matrix()`/`mass()` unchanged. **Materially nonlinear** (small-strain J2/von-Mises plasticity, linear isotropic hardening) — closed-form radial return per Gauss point, path-dependent (`init_state()`/`commit_state()`, per-GP `{eps_p, alpha}` state, unlike `TrussPlastic2D`'s single scalar); `tangent_stiffness()` uses the closed-form consistent (algorithmic) tangent, not finite-difference |
| `Tet4NeoHookean` | 3 (u, v, w) | **Module 19 (general-purpose extensions Phase 6)** — `elements/nonlinear_solids.py`; subclasses `Tet4Solid3D`. **Geometrically nonlinear** (compressible Neo-Hookean hyperelasticity, large-strain Total Lagrangian, `F = I + du/dX` constant per element); closed-form 2nd Piola-Kirchhoff stress, but `tangent_stiffness()` is finite-difference of its own `internal_force()` (same precedent as `GapContactCurvedFriction`, not a hand-derived analytic tangent) |
| `Tet10SolidTL` | 3 (u, v, w) | **Added 2026-08-30, for the wing-cantilever example in the sibling `Multi_Fidelity_NL_Structural_ROM` project** — `elements/nonlinear_solids.py`; subclasses `Tet10Solid3D`. **Geometrically nonlinear only** (St. Venant-Kirchhoff: `material.D_solid3d` — the SAME linear elastic law `Tet10Solid3D` itself uses — applied to the Green-Lagrange strain, not a new hyperelastic energy function), matching that project's own stated nonlinearity source ("large deflection... stiffness nonlinearization", not material). Unlike `Tet4NeoHookean`'s constant `F` (linear shape functions), Tet10's quadratic shape functions give a genuinely position-varying `F`, so `internal_force()` loops over the same 4-point `tet_quadrature_4pt()` rule `Tet10Solid3D.stiffness()/mass()` already use (a deliberate, documented under-integration of the resulting cubic integrand — same precedent as that mass matrix's own quartic integrand). `tangent_stiffness()` is finite-difference, same reasoning as `Tet4NeoHookean`. Validated in `tests/test_tet10_geometric_nonlinear.py`: exact zero-displacement identities, small-load convergence to the linear `Tet10Solid3D` solution on the same mesh, and a large-deflection cantilever benchmark against a real, sourced reference (SOFiSTiK's "BE7" verification document, citing Bisshopp & Drucker 1945) — lands within ~2% of the reference tip deflection despite a fundamentally different element family (3D solid vs. 1D beam/shell). **Note**: `solve_nonlinear_static()`'s default `tol=1e-8` is too tight for this element's FD tangent (stalls around `1e-7`) — pass `tol=1e-5` or so. |
| `Shell4MITC` | 6 (u, v, w, θx, θy, θz) | **Module 20 (general-purpose extensions Phase 7)** — `elements/shells.py`; general 4-node shell (membrane + bending + transverse shear), unlike `Quad4MindlinPlate`'s bending-only formulation. Membrane part reuses `Quad4PlaneStress.B_matrix()`, bending part reuses `Quad4MindlinPlate._Bb_Bs()`'s `Bb`, both evaluated in a per-ELEMENT flat local frame (`_local_frame_and_coords()`, built from the element's own 4 corners, then rotated to global — the same per-element local-axes + block-rotation pattern `Beam3DEulerBernoulli` already establishes). Transverse shear uses a genuine Dvorkin-Bathe MITC4 assumed-natural-strain tying-point interpolation (NOT selective reduced integration) — see the module docstring for the exact tying-point formula and citation. Small artificial diagonal stiffness on the drilling (θz) DOF (`drilling_factor`, default `1e-3`) to avoid a singular tangent on coplanar meshes — not real physics. |
| `Hex8SolidBbar` | 3 (u, v, w) | **Wave 2 item 11 (`docs/consolidated_future_roadmap.md`)** — `elements/solids.py`; subclasses `Hex8Solid3D`, inherits everything but `stiffness()`. Hughes (1980) mean-dilatation B-bar formulation: splits `B` into volumetric/deviatoric parts at the SAME full 2×2×2 Gauss points and replaces the volumetric part everywhere with its element-volume-weighted average — cures volumetric locking as ν→0.5 with no reduced integration and no hourglass risk. Proven (and checked directly) to reduce EXACTLY to `Hex8Solid3D` on any affine field. `Hex8Solid3D` itself is untouched — use whichever formulation fits. See `tests/test_hex8_bbar.py`. |
| `Quad4PlasticJ2PlaneStress` | 2 (u, v) | **Wave 2 item 13 (`docs/consolidated_future_roadmap.md`)** — `elements/nonlinear_solids.py`; subclasses `Quad4PlaneStress`. **Materially nonlinear**, the plane-stress counterpart to `Hex8PlasticJ2`: `material.j2_radial_return_plane_stress()` treats `eps_33` as an extra local unknown driven by a scalar Newton iteration (using the already-validated `j2_radial_return_3d()` as an inner black box) to enforce `sigma_33=0`, then a Schur-complement condensation of the 6×6 tangent gives the 2-D consistent tangent. Per-GP state adds one array (`eps33`) beyond `Hex8PlasticJ2`'s `{eps_p, alpha}`. See `tests/test_plasticity_plane_stress.py`. |
| `Hex8PlasticJ2Kinematic` | 3 (u, v, w) | **Wave 2 item 14 (`docs/consolidated_future_roadmap.md`)** — `elements/nonlinear_solids.py`; subclasses `Hex8PlasticJ2`. **Materially nonlinear**, combined isotropic + Armstrong-Frederick (1966) nonlinear kinematic hardening via `material.j2_radial_return_3d_kinematic()`/`PlasticMaterialJ2Kinematic` (new `C_kin`/`gamma_AF` fields) — gives the back stress a physically-realistic saturation limit instead of growing unboundedly, reproducing the Bauschinger effect `Hex8PlasticJ2`'s isotropic-only hardening cannot. Per-GP state adds a back-stress array (`beta`, 6 components). Local-Newton slope and consistent tangent are finite-difference past the from-scratch scalar-equation derivation (same choice `Tet4NeoHookean` makes, for an analogous transcription-risk reason). `Hex8PlasticJ2`/`j2_radial_return_3d()` untouched. See `tests/test_plasticity_kinematic.py`. |

`lumped_mass()` uses **HRZ diagonal scaling**, not naive row-sum
lumping — a real bug, found while building the dynamics solvers: for
elements with rotational DOFs (beam, plate), the consistent mass
matrix has *negative* off-diagonal entries, so summing a whole row can
produce a negative "mass" and blow up explicit time integration. HRZ
instead rescales the (always-positive) diagonal of the consistent mass
matrix so the total translational mass is preserved exactly. Each
element declares which of its DOFs are translational via
`translational_dof_mask` (e.g. `[True, False]` for the beam's
`[v, theta]`).

`ELEMENT_REGISTRY` maps names to classes.

**To add a new element:** subclass `Element`, implement
`shape_and_derivs()` (+ `B_matrix()` to use the generic Gauss loop, or
override `stiffness()`/`mass()` directly if you need something the
loop can't express), set `translational_dof_mask` if any DOF isn't a
plain translation, and add one line to `ELEMENT_REGISTRY`.

#### Nonlinear extension point (geometric nonlinearity)

Every element also has `internal_force(elem_coords, u_elem, D, thickness)`
and `tangent_stiffness(elem_coords, u_elem, D, thickness)`. The base
class default is `internal_force = stiffness() @ u_elem` and
`tangent_stiffness = stiffness()` — i.e. every linear element in the
table above already satisfies this interface for free, with no per-
element change, because a linear element's tangent doesn't depend on
`u_elem` at all.

`TrussTL2D` is the first element that overrides both with something
that genuinely depends on `u_elem`: a Total-Lagrangian large-
displacement 2-node truss, built on the Green-Lagrange strain / 2nd
Piola-Kirchhoff stress pair (St Venant-Kirchhoff material — S linear
in E_GL, so *all* the nonlinearity is geometric, none material):

```
L0 = |X2 - X1|                          # reference length
dx = (X2+u2) - (X1+u1)                  # current relative position
E_GL = (dx·dx - L0²) / (2 L0²)          # Green-Lagrange strain
S = E * E_GL                            # 2nd Piola-Kirchhoff stress
f2 = (S*A/L0) * dx ,  f1 = -f2          # internal force (virtual-work consistent)
K_T = (E*A/L0³)(dx⊗dx) + (S*A/L0) I2    # tangent: material + geometric part
```

`K_geometric = (S*A/L0) I2` is the part that has no linear-elasticity
analogue: it says a member already carrying axial force S resists
*transverse* motion even before you ask what the material does — the
reason a taut cable is stiff sideways and a slack one isn't. Verified
against a finite-difference of `internal_force()` (see
`validate_nonlinear.py`, agreement to ~1e-9 relative).

Register a new nonlinear element the same way as a linear one —
`ELEMENT_REGISTRY["my_nl_element"] = MyNLElement` — the only difference
is which two methods you override.

#### Nonlinear extension point, part 1b: large-rotation BENDING (Module 15)

`TrussTL2D` above is large-displacement/small-strain but AXIAL only —
no rotational DOF, so it can't represent a beam bending through a large
angle. `Beam2DCorotational` extends the same geometric-nonlinearity
machinery to bending: 3 dof/node `[u, v, θ]`, a 2-node planar
**corotational** beam-column (small local strain, essentially unlimited
rigid rotation).

The corotational idea: attach a local frame to the current CHORD
between the two end nodes (`φ = atan2(Δy, Δx)` of the deformed chord),
track the frame's own rigid rotation since the reference configuration
(`β = φ − φ0`), and measure three **local** ("natural",
rigid-body-mode-free) deformations relative to that rotating frame:

```
e_bar      = L − L0            # chord stretch
theta1_bar = theta1 − beta     # end-1 rotation relative to the chord
theta2_bar = theta2 − beta     # end-2 rotation relative to the chord
```

These are EXACTLY zero under any rigid translation+rotation of the
whole element, by construction — `L`, `theta1`, `theta2`, and `beta`
all change together under a rigid motion, but the three `_bar`
quantities don't. Local force-deformation is the ordinary linear
Euler-Bernoulli 3×3 "natural stiffness" (`N = EA/L0·e_bar`, `M1, M2`
from the usual `4/2/2/4` bending relation); ALL the nonlinearity is in
how `(e_bar, theta1_bar, theta2_bar)` relate to the 6 global DOFs —
via a 3×6 matrix `B = d(local)/d(global)` — exactly Module 8's
"geometric, not material" pattern, one level up from `TrussTL2D`.

**`tangent_stiffness()` is a hand-derived closed form here**, not a
finite difference like `GapContactCurvedFriction`'s — the geometric
stiffness turned out compact enough to state and verify directly:

```
K_T = B^T @ k_local @ B  +  K_geo
K_geo = (N/L)·(z⊗z rank-1 term)  +  ((M1+M2)/L²)·(chord-rotation term)
```

(`z` is `B`'s second/third row's spatial part — see `element.py` for
the full derivation.) `K_geo`'s two pieces mirror `TrussTL2D`'s single
`K_geometric = (S·A/L)·I` term: the axial-force piece is the same
"string stiffening" effect one dimension up, and the moment piece is
new — it comes from the chord angle itself rotating as the end nodes
move, which has no truss analogue (a truss chord has no bending moment
to rotate against).

Validated in `validate_nonlinear_beam.py` (6 checks): pure axial
stretch matches the closed form to machine precision; a superposed
rigid translation+rotation gives EXACTLY zero internal force (frame
invariance — the single most important sanity check for any
corotational element); the hand-derived tangent matches a
finite-difference tangent of `internal_force()` to ~1e-10 relative
error at generic (rotated, stretched) states; a small-load cantilever
chain matches Euler-Bernoulli `PL³/3EI` and is mesh-independent (the
linearized beam stiffness is nodally exact for a tip load, so even 2
elements already match); and — the strongest check available for a
nonlinear beam — the large-load response of a chain of these elements
converges, as the mesh refines, to the EXACT Euler **elastica**
solution (Bisshopp & Drucker 1945), obtained independently here by
shooting the elastica ODE with `scipy.integrate.solve_bvp` rather than
a hand-typed closed form.

**`mass()`** (added for the flat-beam benchmark below): `Beam2DCorotational`
originally had no `mass()` override, which silently fell back to the
generic `Element.mass()` Gauss loop — WRONG for this element, since that
loop applies the same linear 2-node shape function to all 3 dofs/node
(including `theta`), when transverse displacement/rotation should be
coupled through cubic Hermite shape functions. The added `mass()` builds
the standard linear elastodynamics consistent mass (axial 2×2 +
Hermite-bending 4×4, no separate rotary-inertia term — same convention as
`Beam2DEulerBernoulli.mass()`) in the beam's own local frame, then rotates
it into global coordinates via the reference chord angle so it's correct
at any planar orientation (`M_global = T @ M_local @ T.T`, matching how
`internal_force()`/`tangent_stiffness()` already handle rotation). Verified
against `Beam2DEulerBernoulli.mass()` at zero rotation, exact total-mass
conservation, rotation-invariant eigenvalues, and symmetry/positive-
definiteness — see `../flat_beam_fem.py` (one level up, alongside the
other worked-example scripts), which builds the clamped-clamped "flat
beam" benchmark of He, Yang & Wang (2023, Sect. 4.1) with this element
and cross-checks its modal frequencies against a closed-form
clamped-clamped formula, the paper's own reported values, and an
independently-coded reference beam model, then cross-checks a nonlinear
static pressure sweep against that same reference model.

#### Nonlinear extension point, part 2: path-dependent state (material nonlinearity)

Geometric nonlinearity (`TrussTL2D`) is *state-free* — its
`internal_force()`/`tangent_stiffness()` depend only on the CURRENT
displacement, nothing about history. Elasto-plasticity can't work that
way: whether a point is elastic or yielding right now depends on how
much it has *already* plastically deformed, so the element needs
somewhere to remember that between load steps.

Three more (optional) methods complete the interface:

| Method | Called by | Purpose |
|---|---|---|
| `init_state()` | `FESystem.init_state()`, once, before the first nonlinear solve | Returns this element's initial (virgin) state dict, e.g. `{"eps_p": 0.0, "alpha": 0.0}` |
| `internal_force(..., state=...)` / `tangent_stiffness(..., state=...)` | Every Newton-Raphson iteration | Compute a TRIAL response from the last-*committed* `state` and the current (not-yet-equilibrated) displacement — must NOT mutate `state` |
| `commit_state(elem_coords, u_elem, mat, state)` | `FESystem.commit_all_states()`, once per CONVERGED load step | Replays the same trial computation and returns the new state to persist |

`FESystem` stores one state object per element in `self.state` (`None`
until `init_state()` is called — every element and every existing
script from Module 8 and earlier is completely unaffected, since
`assemble_internal_force()`/`assemble_tangent_stiffness()` only add a
`state=` kwarg when `self.state is not None`, and every element's
signature accepts and ignores unused kwargs). `nonlinear_solver.py`'s
two drivers call `commit_all_states()` automatically after every
converged step — trial-and-discard during iteration, commit only on
convergence, is the standard FE plasticity pattern and this package's
convention now, not something calling code has to manage by hand.

`TrussPlastic2D` is the first element to use this: a small-
displacement (engineering-strain, NOT Green-Lagrange — see its class
docstring for why material nonlinearity is deliberately isolated from
geometric nonlinearity here) 2-node truss with closed-form 1-D J2/von
Mises return mapping and linear ISOTROPIC hardening:

```
eps_e_trial = eps_total - eps_p_n            # trial elastic strain
sigma_trial = E * eps_e_trial
f_trial = |sigma_trial| - (sigma_y + H*alpha_n)   # trial yield check

if f_trial <= 0:  sigma = sigma_trial;  Et = E                    # elastic
else:                                                              # plastic
    dgamma = f_trial / (E + H)             # closed form -- yield fn is LINEAR
    sigma = sigma_trial - E*dgamma*sign(sigma_trial)
    eps_p_new = eps_p_n + dgamma*sign(sigma_trial)
    alpha_new = alpha_n + dgamma
    Et = E*H / (E + H)                     # consistent elastoplastic tangent
```

No local Newton iteration is needed (unlike general 2-D/3-D J2
plasticity) because a bar can only carry uniaxial stress, so the yield
surface is linear in `sigma`. Register a new path-dependent element the
same way: implement the three methods above plus `internal_force()`/
`tangent_stiffness()`, add one line to `ELEMENT_REGISTRY`.

#### Nonlinear extension point, part 2b: mixed-formulation internal state (Module 23)

A THIRD kind of per-element "extra information," added for (and so far
only used, unsuccessfully, by) `Shell4MITCCorotational`'s attempt at a
mixed/Hellinger-Reissner reformulation of its von Karman coupling term
(see that class's "Design history" comment, "DEAD END 7," for the full
story) — deliberately SEPARATE from part 2's `state`/`commit_state()`
above, because the timing is opposite: a mixed formulation's own
internal unknown (e.g. a stress resultant) must be corrected using the
REALIZED Newton step on EVERY iteration, not just once a load step
converges.

| Method | Called by | Purpose |
|---|---|---|
| `init_iter_state()` | `FESystem.init_iter_state()`, once, before the first nonlinear solve | Returns this element's initial internal-unknown value |
| `internal_force(..., iter_state=...)` / `tangent_stiffness(..., iter_state=...)` | Every Newton-Raphson iteration | Use the CURRENT `iter_state` as a parameter |
| `update_iter_state(elem_coords, u_elem, delta_u_elem, mat, iter_state)` | `FESystem.update_iter_states()`, every ACCEPTED Newton correction | Correct `iter_state` using `delta_u_elem`, the correction just realized, and return the new value |

`FESystem` stores this in `self.iter_state`, structurally identical to
`self.state` but tracked and updated on a completely different
schedule; `None` until `init_iter_state()` is called, so — same
guarantee as `self.state` — every existing element and script is
unaffected. Six of `nonlinear_solver.py`'s drivers
(`solve_nonlinear_static`, `solve_nonlinear_displacement_control`,
`solve_nonlinear_arc_length`, and all three Koiter-Newton drivers) call
`update_iter_states()` automatically after every accepted correction,
with a snapshot/restore safeguard around the three Koiter-Newton drivers'
own shrink-and-retry loops so a REJECTED trial's mutations can't leak
into the next, smaller attempt.

**This infrastructure itself is real, general-purpose, and validated**
(`tests/test_iter_state.py`, a toy element with an exact-fingerprint
accumulator) — any future mixed-formulation element could use it
correctly. `Shell4MITCCorotational`'s own specific use of it does
**not** currently work: the only mechanism available without giving the
internal unknown true global-DOF status — correcting it via the
displacement delta realized since the element's own LAST call — is a
linearization valid only for small steps between calls, and the
internal unknown starts at exactly zero and must become substantial on
the very first Newton correction of any real solve. This was confirmed
to diverge (a genuine floating-point overflow, not just slow
convergence) both on a standalone toy problem and on the real element,
with every analytic piece (the Schur-complement tangent correction, the
compatibility residual) independently verified correct via complex-step
— a structural limitation of the lagged-update strategy, not a bug. See
`shells.py`'s "Design history" ("DEAD END 7") for the full account and
the two paths that remain genuinely open.

#### Nonlinear extension point, part 3: contact/boundary nonlinearity

Contact is a third, distinct kind of nonlinearity — not a strain
measure (geometric) or a constitutive law (material), but a
**discontinuous change of boundary condition**: zero stiffness while
apart, a large (penalty) or exactly-enforced (Lagrange) stiffness the
instant two surfaces touch. `GapContactPenalty` implements this as a
1-node unilateral gap element against a FIXED rigid obstacle (not part
of the mesh) — stateless, since whether contact is active depends only
on the current position, not history, unlike `TrussPlastic2D`:

```
delta = n_hat . u_node - g0              # penetration (>0 = touching/through)
f = 0,               K = 0               if delta <= 0   (apart)
f = k_p*delta*n_hat, K = k_p*n_hat(x)n_hat  if delta > 0   (in contact)
```

Because this fits the SAME `internal_force()`/`tangent_stiffness()`
interface as every other nonlinear element, `nonlinear_solver.py`
needed **zero changes** to solve a penalty-contact problem — the
payoff of the additive interface paying off a third time. What *did*
need a small, genuinely new piece: a contact element acts on a node
that belongs to the MAIN structural mesh, but isn't itself part of
that mesh's element list. `FESystem.add_contact_element(elem,
node_ids, mat)` registers it separately (`self.contact_elements`), and
`assemble_internal_force()`/`assemble_tangent_stiffness()` now sum
contributions from both the main mesh AND every registered contact
element into the same global vectors/matrices — letting a contact
pair coexist with an existing structural analysis (mirroring how real
FE codes define contact pairs alongside, not inside, the mesh).

**Lagrange multipliers** are handled differently, and deliberately
NOT as an `Element` subclass: exact (zero-penetration) enforcement
needs one extra scalar unknown (the contact reaction `lambda`) added
to the system, not just a stiffer local term — literally "an extra
equation in the global matrix," as opposed to the penalty method's
approximate, in-place stiffness. `nonlinear_solver.solve_contact_lagrange_static()`
manages this itself (see its module reference below) with a two-phase
active-set Newton-Raphson per load step, rather than trying to force
it through the fixed-DOF-count `Element` interface used everywhere
else in the package.

#### Nonlinear extension point, part 3b: friction + updating contact normal (Module 13)

`GapContactPenalty` above is deliberately the simplest possible contact
element: frictionless, and against a FLAT wall (a fixed `n_hat`, so
"contact geometry nonlinearity" — the idea that the normal direction
itself changes as the node slides along a curved obstacle — isn't
representable). A user-supplied taxonomy of 9 kinds of contact
nonlinearity (normal, frictional, separation, sliding, self-contact,
impact, adhesive, roughness, contact-geometry) was checked against the
package; only normal contact was genuinely implemented. `GapContactCurvedFriction`
closes two of the remaining categories at once — Coulomb friction, and
an updating (deformation-dependent) contact normal — by pairing them in
a single 1-node element against a CIRCULAR obstacle (`mat = (k_p, k_t,
mu, center, R)`):

```
d_vec = x_node - center;  d = |d_vec|;  n_hat = d_vec / d      # updates with position
t_hat = perp(n_hat)                                            # tangent direction
gn = d - R                                                     # penetration
pn = k_p * max(-gn, 0)                                         # normal pressure
```

**Friction as plasticity, reused.** Coulomb stick/slip
(`|tt| <= mu*pn` stick, `|tt| = mu*pn` slip) has the exact same
mathematical shape as 1-D J2 plasticity (Module 9): a linear trial
response capped at a pressure-dependent limit. `GapContactCurvedFriction`
reuses `TrussPlastic2D`'s `init_state()`/`commit_state()` pattern
directly — it remembers a single scalar, the tangential "stick anchor"
arc-length position `s_stick`, and computes a trial tangential force
`tt_trial = k_t*(s - s_stick)` each Newton iteration, clamping to
`mu*pn` and shifting the anchor only on commit. Because contact
elements are added via `add_contact_element()`, not the main mesh, this
session also had to extend `FESystem` itself: `add_contact_element()`
now calls `elem.init_state()` when the contact element defines one
(building a `self.contact_state` list parallel to `self.contact_elements`),
and `assemble_internal_force()`/`assemble_tangent_stiffness()` pass each
contact element its own `state=` kwarg, updated by `commit_all_states()`
exactly like the main mesh's `self.state` — `GapContactPenalty`, being
stateless, never exercised this path, so it was a real gap, not a
redundant addition.

**A sign-convention bug caught before any code ran.** Deriving the
normal force by intuition ("contact should push the node away from the
obstacle") gives `f = k_p*(R-d)*n_hat`, pointing OUTWARD — wrong. Every
other element in this package derives internal force as `F_int = dU/du`
from an explicit potential (`U = 0.5*k_p*(R-d)^2` here), which instead
gives `f = k_p*(d-R)*n_hat = -pn*n_hat`, pointing TOWARD the center (the
*increasing-penetration* direction) when active — matching
`GapContactPenalty`'s own convention and the `F_ext=F_int`-at-
equilibrium convention used everywhere else. The outward formula was
written down first, then caught and corrected purely by insisting on
the potential-energy derivation rather than trusting the intuitive
picture, before any code was run or tested.

**Tangent stiffness is computed numerically** (central finite
difference on `internal_force()`), a deliberate choice rather than a
fragile hand-derived analytical tangent for a combined nonsmooth
(stick/slip) + curved-geometry element — legitimate and used in
production contact codes for exactly this kind of nonsmooth,
path-dependent physics. It's cross-checked, not left unverified: for
the frictionless (`mu=0`) special case, `validate_curved_contact.py`
CHECK 2 derives the closed form independently,

```
K_T = k_p*(1 - R/d)*I + k_p*(R/d)*(n_hat ⊗ n_hat)
```

and confirms it matches the finite-difference tangent to ~1e-10
relative error. This closed form also has a real, expected quirk: its
tangential eigenvalue `k_p*(1-R/d)` is NEGATIVE whenever the node has
penetrated a convex obstacle (`d < R`) — a genuine "curvature effect"
(pushing a node sideways along a convex surface it's pressed into
reduces resistance, it doesn't add it), not a bug. Newton-Raphson only
needs `K_T` invertible at each iterate, not positive-definite, so this
doesn't break convergence.

**Scope, honestly.** This closes categories 2 (friction) and most of 9
(contact geometry nonlinearity) from the user's taxonomy. **Node-to-
segment contact search between two genuinely deformable bodies is now
implemented** (Wave 3 items 16-17, `docs/consolidated_future_roadmap.md`)
via `NodeToSegmentContact2D`/`NodeToSegmentContact2DFriction` and
`find_contact_pairs_2d()` -- see "Node-to-segment contact" below;
`GapContactPenalty`/`GapContactCurvedFriction` above remain untouched
and are still the right, cheaper tool when the obstacle genuinely IS
fixed/analytic. Still NOT implemented, and out of scope unless asked
for explicitly: self-contact, a real spatial-hash/BVH broad phase
(`find_contact_pairs_2d()` is a documented exhaustive distance filter,
adequate for small-to-moderate contact zones, not large-scale contact),
segment-to-segment (vs. node-to-segment) discretization, 3-D
node-to-segment (the new elements are 2-D only), adhesive/cohesive
contact, surface roughness/tribology, and dynamic/explicit impact
integration (the matching penalty formula exists but nothing drives it
through an explicit time-stepping scheme). One contact pair per
element, monotonic engagement within a pair (mirrors `GapContactPenalty`'s
and the Lagrange contact solver's existing scope limits).

#### Mixed-element-type assembly (Module 14)

Every element covered so far assumes `FESystem` has exactly ONE
element formulation for the whole mesh — true for every generator in
`geometry.py`, but not necessarily true of an arbitrary unstructured
mesher in general: a Gmsh "recombine" pass (pairing triangles into
quads), back when this package had Gmsh-backed geometry support (now
removed — see "Removed: Gmsh support" above), often left a handful of
triangles un-paired, so a realistic unstructured 2-D mesh could
genuinely contain BOTH `Tri3PlaneStress` and `Quad4PlaneStress`
elements at once. Before this module, the old `geometry_engine.py`'s
`_extract_mesh()` refused this outright (`RuntimeError: Gmsh produced
MIXED element types`) because `FESystem` had no way to call more than
one element formulation per mesh. This module's `FESystem` support for
mixed element blocks is independent of Gmsh and remains fully
supported; only the Gmsh-based mixed-mesh *producer* described below
was removed along with the rest of Gmsh support.

`FESystem.__init__` now accepts EITHER the original single-formulation
usage (`FESystem(mesh, elem_formulation)`, `mesh` a plain `mesh.Mesh`)
OR a mixed usage (`FESystem(mesh, {block_name: elem_formulation, ...})`,
`mesh` a `mesh.MultiBlockMesh` — see `mesh.py`). Internally, both
normalize to the SAME thing: a list of `(block_name, formulation,
connectivity_array)` triples (`self._blocks`), with the single-
formulation case simply being the one-block special case of that list.
Every `assemble_*`/`init_state`/`commit_all_states` method loops over
`self._blocks` now instead of a single `(self.elem, self.mesh.elements)`
pair — meaning the ORIGINAL usage is not a separate code path anymore,
it's the same loop with `len(self._blocks)==1`, verified byte-for-byte
unchanged in behavior against every validation script written before
this module existed (`validate_contact.py`, `validate_nonlinear.py`,
`validate_plasticity.py`, `validate_geometry.py`,
`validate_simplex_elements.py`, `validate_curved_contact.py`,
`validate_geometry_engine.py` — all still pass unmodified).

`D`/`mat`/`rho` arguments to `assemble_stiffness()`/`assemble_mass()`/
`assemble_internal_force()`/etc. may be a single value (reused for
every block — the common case, one physics spread across more than one
element topology) or a dict keyed the same way as the blocks, for
genuinely per-block materials/physics (`_per_block_arg()` — D/mat/rho
are always tuples/arrays/dataclasses elsewhere in this package, never
plain dicts, so `isinstance(arg, dict)` is an unambiguous switch).

**Hard constraint, by design, not an oversight:** every block sharing
one `FESystem` must have the SAME `dofs_per_node`. Global DOFs are
numbered per NODE (`_global_dofs()`), not per block, so a node touched
by a 2-dof/node block and a 3-dof/node block would need two different
numbers of DOFs at once — `FESystem.__init__` checks this explicitly
and raises a `ValueError` naming the offending `dofs_per_node` values
rather than silently corrupting the DOF numbering. This covers "mixed
element TOPOLOGY, same physics" (Tri3+Quad4 plane stress, Tet4+Hex8
solid — what an unstructured mesher's leftover-element problem actually
looks like); it does NOT cover mixing element FAMILIES with different
`dofs_per_node` (a plate block with a plane-stress block) in one mesh,
which is a genuinely different multi-field problem.

(Historical note, the producer side of this: the old, now-removed
`geometry_engine.py`'s `_extract_mesh()` built one connectivity block
per Gmsh element type it found, and `generate_2d_rectangle_mixed()` was
a producer specifically for exercising this path, using Gmsh's "simple"
recombination algorithm to reliably leave real leftover triangles.
`tests/test_mixed_elements.py`'s CHECK 1-4 still validate the
`FESystem` mixed-block machinery itself end to end on a hand-built
mixed mesh — the CHECK 5 Gmsh-producer variant was removed along with
Gmsh support.)

The pre-existing `contact_elements`/`contact_state` mechanism (Modules
10/13) is untouched and needed no changes — it was already the
package's first working example of "more than one element formulation
per FESystem" (each contact element carries its own formulation/mat),
just for a side-channel of one-node elements rather than the main
mesh; Module 14 generalizes the same idea to the main mesh itself.

#### Simplex elements (Tri3, Tet4) — why they DON'T use the generic Gauss loop

Every other continuum element in this package (`Quad4PlaneStress`,
`Hex8Solid3D`) has natural coordinates that range over `[-1, 1]` in
each direction, exactly matching `gauss_product()`'s tensor-product
quadrature domain — that's what lets the base class's `stiffness()`
work unmodified for both of them. A triangle or tetrahedron's natural
coordinates instead range over a SIMPLEX (`{xi>=0, eta>=0, xi+eta<=1}`
for a triangle) — a genuinely different domain, not a distorted square.
Feeding `gauss_product()`'s sample points into `Tri3PlaneStress`'s
shape functions would evaluate them OUTSIDE the valid simplex for many
points (negative shape function values — not quadrature error, a
domain mismatch).

`Tri3PlaneStress`/`Tet4Solid3D` sidestep this entirely rather than
implementing simplex-specific quadrature: their shape functions are
LINEAR, so strain (and `B`) is CONSTANT over the whole element, and the
"integral" `ke = ∫ Bᵀ D B dV` is exactly one evaluation times the
physical area/volume — closed form, same idea as
`Beam2DEulerBernoulli`'s closed-form Hermite stiffness. `mass()` is
likewise closed-form (the classic `Area/12`-pattern and `Volume/20`-
pattern consistent mass matrices for linear simplices), not a Gauss
sum. **A real bug was caught this way**: the first version of
`lumped_mass()` computed its target total mass using only ONE dof
direction's worth of mass while rescaling a diagonal that spans ALL
directions, silently halving (2-D) or reducing to a third (3-D) every
lumped entry — caught by `validate_simplex_elements.py` CHECK 3
checking the lumped mass's per-direction sum against the exact
`rho*Area*thickness`/`rho*Volume`, not just eyeballing the numbers.

#### Full vs. reduced integration

Every element's `stiffness()` accepts a quadrature override for that
call, plus two named wrappers:

| Method | Meaning |
|---|---|
| `full_stiffness(elem_coords, D)` | `self.gauss_order` points/direction — exact for the element's stiffness polynomial. The default everywhere in this package. |
| `reduced_stiffness(elem_coords, D)` | One point/direction fewer. Relieves locking, at the risk of rank-deficient ("hourglass") spurious modes. |
| `hourglass_stabilized_stiffness(elem_coords, D, c_hg=0.1)` | **Wave 2 item 12** — `reduced_stiffness()` plus a generic deflated-eigenproblem stabilizer that adds back only the spurious/hourglass energy `reduced_stiffness()` is missing (not the literature's tabulated Flanagan-Belytschko/Belytschko-Bindeman shape vectors — see `elements/base.py`'s docstring for why this generic approach was chosen instead). Vanishes exactly on affine fields. Works on any `Element` subclass with no per-element code. |
| `spurious_zero_energy_modes(elem, elem_coords, D)` | Module-level function. Compares `full_stiffness()`'s rank to `reduced_stiffness()`'s to count spurious modes, with no need to know the element's theoretical rigid-body-mode count in advance. |

`FESystem.assemble_stiffness(D, method="full")` accepts the same
three-way choice at the whole-model level — `method="full"` (default,
unchanged prior behavior), `"reduced"`, or `"hourglass_stabilized"` —
so an entire mesh can be reduced-integrated-and-stabilized with one
call rather than looping per-element.

`Quad4MindlinPlate.stiffness()` additionally takes `integration=`:
`'sri'` (default — full bending, reduced shear, the standard
non-hourglass-prone fix), `'full'`, or `'reduced'`.

**This was checked numerically, not assumed** (`main.py` items 9–11)
— the results differ sharply by element and are worth knowing before
reaching for `reduced_stiffness()`:

| Element / problem | Full integration | Reduced integration |
|---|---|---|
| `Hex8Solid3D`, single element | 18 nonzero eigenvalues (24 dof − 6 rigid-body) | Exactly **12 spurious modes** (1-pt rule caps rank at 6) |
| `Hex8Solid3D`, cantilever beam mesh | Locks (ratio 0.71–0.90 vs. Euler-Bernoulli, mesh-dependent) | **Numerically singular** — 82 near-zero eigenvalues survive even after boundary conditions are applied. Do not use. |
| `Quad4MindlinPlate`, cantilever plate | Locks badly (ratio **0.33**, reproducing the historic quadrature bug from `cantilever_plate_fem.py`) | Accurate (ratio 0.98) *for this mesh/load*, but 7 near-zero modes remain — not unconditionally safe. Use `'sri'` (ratio 0.98, 0 near-zero modes) instead. |
| `Quad4PlaneStress`, deep bracket | Good (ratio 0.98) | Also good, slightly better (ratio 0.99), **0 near-zero modes** — the one case here with no caveats |

The takeaway: reduced integration is not a free upgrade. It fixes
locking exactly when the removed integration order was the *cause* of
spurious stiffness (plane stress, and the plate's shear term via SRI),
and it breaks the system when the element doesn't have enough
alternate load paths to constrain the resulting zero-energy modes
(Hex8 in bending, above) — use `hourglass_stabilized_stiffness()` /
`method="hourglass_stabilized"` (Wave 2 item 12, above) in that case
rather than plain `reduced_stiffness()`. Always check
`spurious_zero_energy_modes()` or a near-zero-eigenvalue count on the
constrained system before trusting a reduced-integration result.

### `loads.py` — load definitions

Decouples *what a load is* from *how it's applied* (`solver.py`) and
*how the response is solved for* (`solver.py`'s `solve_*` methods).

| Class | Use with |
|---|---|
| `LoadPattern(node_ids, dof_index)` | The spatial distribution underneath every other load type — `total_force` split evenly across `node_ids`. |
| `TimeHistoryLoad(pattern, time_fn)` | `solve_transient_implicit` / `solve_transient_explicit` |
| `HarmonicLoad(pattern, F0)` | `solve_harmonic` / `solve_frequency_sweep` |
| `PSDLoad(pattern, freqs, psd)` | `solve_random_vibration` — its `.force_vector()` is always unit-magnitude; the PSD magnitude is applied by the solver, not the load object |

**To add a new load type** (e.g. a moving load, base excitation): add
one small class with a `.pattern` and whatever magnitude description
it needs.

### `solver.py` — assembly, boundary conditions, and every solve strategy

`FESystem(mesh, elem_formulation, thickness=1.0)` holds `K`, `M`,
`M_lumped`, `C`, and `F`, and is built once per problem.

**Assembly:**

| Method | Builds |
|---|---|
| `assemble_stiffness(D, **kwargs)` | `K` |
| `assemble_mass(rho_or_matrix, **kwargs)` | `M` (consistent) |
| `assemble_lumped_mass(rho_or_matrix, **kwargs)` | `M_lumped` (HRZ) — required before `solve_transient_explicit()` |
| `assemble_damping(damping)` | `C` from a `RayleighDamping` — required before `solve_transient_implicit()`/`solve_harmonic()` |
| `assemble_internal_force(u_global, mat, **kwargs)` | `F_int(u_global)` — the nonlinear counterpart of `assemble_stiffness()`, calling `element.internal_force()` per element at the CURRENT displacement state; used by `nonlinear_solver.py`, not `self.K` |
| `assemble_tangent_stiffness(u_global, mat, **kwargs)` | `K_T(u_global)` — same idea for `element.tangent_stiffness()`; must be rebuilt every Newton-Raphson iteration, unlike the linear `K` |
| `init_state()` | Allocates `self.state`, one entry per element (`elem.init_state()`, or `None` for stateless elements) — call once before a solve that uses a path-dependent element (e.g. `TrussPlastic2D`) |
| `commit_all_states(u_global, mat, **kwargs)` | Advances every element's state to its converged response at `u_global` — called automatically by `nonlinear_solver.py` after each converged step; a no-op if `init_state()` was never called |
| `init_iter_state()` | Module 23 — allocates `self.iter_state`, DELIBERATELY SEPARATE from `self.state` above (`elem.init_iter_state()`, or `None` for elements that don't define it) — for a mixed-formulation element's own internal unknown, updated every Newton ITERATION rather than every converged step |
| `update_iter_states(u_global, delta_u_global, mat, **kwargs)` | Module 23 — advances every element's `iter_state` using the REALIZED Newton correction `delta_u_global` a driver's own linear solve just produced; called automatically after every accepted correction (not just converged steps) by five of `nonlinear_solver.py`'s drivers; a no-op if `init_iter_state()` was never called. Validated (`tests/test_iter_state.py`) as general-purpose infrastructure; `Shell4MITCCorotational`'s own attempt to use it for its von Karman coupling term is NOT currently working (see that class's "Design history" comment, "DEAD END 7") — a lagged internal-unknown correction is not sufficient on its own, see that comment for what would be needed |
| `add_contact_element(elem, node_ids, mat)` | Registers a contact/constraint element (e.g. `GapContactPenalty`) acting on nodes of the MAIN mesh, with its own `mat` — `assemble_internal_force()`/`assemble_tangent_stiffness()` above automatically sum its contribution alongside the main mesh's |

**Loads and boundary conditions:**

`fix_dofs(node_ids, dof_indices)`, `add_nodal_force(node_ids,
dof_index, total_force)`, `add_consistent_edge_load(node_pairs,
dof_index, traction, thickness)` (length-weighted, for graded meshes).

**Solving — one method per formulation type:**

| Method | Formulation |
|---|---|
| `solve_static()` | `K U = F` |
| `solve_modal(n_modes)` | `K φ = ω² M φ`, returns `(freq_hz, mode_shapes)` |
| `critical_timestep()` | `dt_crit = 2/ω_max` from the full free-system eigenspectrum — use to pick a stable `dt` for explicit integration |
| `solve_transient_implicit(load, T_total, dt, ...)` | Newmark-β; `K_eff` is factored **once** (linear, time-invariant), not every step |
| `solve_transient_explicit(load, T_total, dt)` | Central difference on `M_lumped`; matrix-free per step if `C` is `None` or diagonal, falls back to a per-step solve for a general `C` |
| `solve_modal_superposition(load, T_total, dt, n_modes, zeta)` | Projects onto mass-normalized modes, integrates `n_modes` decoupled scalar equations (no `n_dof × n_dof` solve ever happens) |
| `solve_harmonic(Omega, F0_vector)` / `solve_frequency_sweep(Omega_array, F0_vector)` | `(-Ω²M + iΩC + K) U₀ = F₀`, direct complex solve, no time marching |
| `solve_random_vibration(freqs_hz, psd_input, F0_pattern, output_dof)` | `S_out = \|H(f)\|² S_in(f)` built from `solve_frequency_sweep` — general for any MDOF system, no SDOF closed-form assumed |

### `nonlinear_solver.py` — incremental Newton-Raphson (geometric + material + contact nonlinearity)

Seven drivers — six STATIC (one load-stepping strategy can't trace
every equilibrium path) plus one dynamic (Newmark-Newton time
integration):

| Function | Strategy | Use when |
|---|---|---|
| `solve_nonlinear_static(fesystem, mat, n_steps, tol, max_iter)` | **Load control** — ramps `F_ext = λ·F` from 0 to 1, Newton-Raphson to equilibrium at each `λ` | The load-displacement path is monotonic (e.g. a large-deflection cantilever). Fails at/past a **limit point** (`dP/dδ = 0`) — `K_T` becomes singular and Newton-Raphson can't converge. |
| `solve_nonlinear_displacement_control(fesystem, mat, control_dof, u_target_array, tol, max_iter)` | **Displacement control** — prescribes `u[control_dof]` directly at each step, Newton-Raphson's everything else, records the reaction at `control_dof` | The structure **snaps through** (e.g. a shallow arch/truss) — since `δ` is now the control parameter instead of something equilibrium has to produce, the solver sails straight through the limit point with no special continuation algorithm. Only valid when a single DOF is a meaningful path parameter for the whole structure; **snap-back** (where even the controlled DOF itself reverses) needs arc-length instead. |
| `solve_nonlinear_arc_length(fesystem, mat, delta_L, n_steps, tol, max_iter)` | **Crisfield cylindrical arc-length** (Module 19, general-purpose extensions Phase 5) — solves for BOTH the displacement increment and the load-factor increment together each step, constrained to `‖Δu‖ = delta_L`; neither the load nor any single DOF is prescribed | The general case — works whenever displacement control does, AND for snap-back / any structure where no single DOF is a valid path parameter, with no advance knowledge of where limit points are. |
| `solve_nonlinear_koiter_newton(fesystem, mat, delta_L, n_steps, tol, max_iter, predictor_tol, fd_rel)` | **Koiter-Newton asymptotic PATH CONTINUATION** (Module 21, general-purpose extensions Phase 8) — replaces arc-length's LINEAR tangent predictor with a CUBIC one built from a genuine Koiter asymptotic expansion at each equilibrium point (single-branch/`m=0` case), then corrects with a chord/modified-Newton iteration that reuses ONE frozen-tangent factorization per step (no re-assembly in the corrector) | Same use case as arc-length — tracing an entire equilibrium path with no specific target — but when fewer/larger steps matter. ~11x fewer outer steps than arc-length on the validated benchmark, for the same `delta_L` seed. **NOT for reaching one specific target load** — tried that via a wrapper and it overshot 14x / crashed with overflow on a real model; use `solve_nonlinear_static_koiter_newton` below instead. Not implemented: buckling-mode interaction (`m≥1`) for closely-spaced modes — see its own docstring's "SCOPE" section. |
| `solve_nonlinear_static_koiter_newton(fesystem, mat, tol, max_iter, max_expansions, predictor_tol, fd_rel)` | **Koiter-Newton solve to ONE prescribed target load** (Module 22, general-purpose extensions Phase 9) — `F_ext = 1.0*fesystem.F`, same load convention as `solve_nonlinear_static`. Reuses the same cubic predictor machinery, but aims each expansion's step directly at the remaining gap to the target (never overshoots), then corrects with ordinary fixed-load frozen-tangent Newton (no bordering needed once the target is known) | Generating nonlinear-static ROM-training data cheaply (the actual motivating use case — Yang et al. 2019's own Fig. 1 flowchart), or any other case where you want ONE known load level's equilibrium fast, not a whole path. Validated directly on a real shell model: single-expansion convergence, <5e-7 agreement with plain Newton, ~15-18x faster per solve. Returns just `u` (not a `(load_factors, U_hist)` path) since it solves one equilibrium problem, not a path. |
| `solve_nonlinear_koiter_newton_generic(fesystem, mat, delta_L, n_steps, tol, max_iter, predictor_tol, fd_rel, mode_detect_rel_tol, enable_mode_interaction)` | **GENERIC (m≥1) Koiter-Newton continuation** (Module 24, general-purpose extensions Phase 10) — lifts `solve_nonlinear_koiter_newton()`'s disclosed `m=0` scope reduction: at each step, a cheap partial eigensolve checks `K_T` for a near-critical mode, and when found, adds it as a SECOND perturbation direction in the SAME bordered-matrix machinery, generalized to two directions (2×2 Lagrange-multiplier matrix, polarization-identity cross-tensor extraction, a genuine cubic *equilibrium constraint* for the extra direction with up to 3 real roots — the actual mode-interaction capability, not just a bigger predictor). Falls back to `solve_nonlinear_koiter_newton()`'s own exact `m=0` formulas whenever no critical mode is detected (or `enable_mode_interaction=False`) | Path-following through a structure with closely-spaced or interacting buckling modes, where `solve_nonlinear_koiter_newton()`'s single-direction assumption may miss the real post-buckling behavior. A brand-new function, NOT a modification of `solve_nonlinear_koiter_newton()` — that driver is unchanged and still what `NonLin-HyROM`'s training pipeline depends on. See its own docstring's "SCOPE" section for what's still simplified (at most one extra direction, continuity-based branch selection, no detection hysteresis). |
| `solve_nonlinear_transient(fesystem, mat, load, T_total, dt, beta, gamma, u0, v0, tol, max_iter)` | **Newmark-Newton implicit TIME integration** (`docs/nonlinear_transient_dynamics_roadmap.md`) — the dynamic counterpart to the static drivers above: predicts `d/v/a` exactly as `solve_transient_implicit()` does, then Newton-Raphsons to convergence at EVERY step using the current tangent `K_eff(d) = K_T(d) + a0c*M + a1c*C` (re-factored every iteration, unlike the linear driver's one-time factorization) | Any dynamic (not quasi-static) nonlinear analysis — free vibration, transient forced response, ROM training-data generation — where the STATIC drivers above don't apply because inertia/damping actually matter. |

All seven call `fesystem.assemble_internal_force()`/`assemble_tangent_stiffness()`
each iteration — nothing in this file knows it's a truss, or that it's
geometric (rather than material or contact) nonlinearity; swapping in a
different nonlinear element requires zero changes here.

**Newton-solver robustness knobs (Wave 1, `docs/consolidated_future_
roadmap.md`, 2026-09-07):** `solve_nonlinear_static`,
`solve_nonlinear_displacement_control`, `solve_nonlinear_arc_length`,
`solve_nonlinear_koiter_newton`, and `solve_nonlinear_static_koiter_
newton`'s corrector all additionally accept `du_tol=None`/
`energy_tol=None` (nonlinear_fem_lessons.md §6.3.9's displacement-
increment and energy-error criteria, Belytschko & Schoeberle 1975 —
OPTIONAL, AND-combined ON TOP OF the force-residual check, never a
replacement for it) and, on the first four plus `solve_nonlinear_
static_koiter_newton`, `line_search=True` (Armijo backtracking,
the same algorithm already validated as `solve_nonlinear_transient()`'s
own fallback — tried ONLY if plain Newton exhausts `max_iter`, never
the default path). All three keep every existing caller's results and
cost UNCHANGED (`du_tol=energy_tol=None`, and a step that already
converges never invokes line search at all). Deliberately NOT added to
`solve_nonlinear_koiter_newton_generic`'s two correctors, or as
`line_search` to `solve_nonlinear_koiter_newton`'s own continuation
corrector — see `nonlinear_solver.py`'s own module docstring ("Wave 1"
section) for the full reasoning (both already have a different,
existing globalization strategy of their own, and one has a
previously-debugged sign-error history real enough to make composing
a second mechanism onto it a materially riskier change than this item
asked for). See `nonlinear_solver._extra_convergence_ok()`/
`_armijo_line_search_step()` for the two shared implementations every
driver above calls, and `tests/test_wave1_newton_robustness.py` for
their validation.

```python
mesh = Mesh(nodes=np.array([[-1, 0], [0, 0.1], [1, 0]]), elements=np.array([[0,1],[1,2]]), dim=2)
sys = FESystem(mesh, TrussTL2D())
sys.fix_dofs([0, 2], [0, 1])
mat = (E, A)   # TrussTL2D takes (E, A) directly, like Beam2DEulerBernoulli's EI

# monotonic / pre-limit-point:
sys.add_nodal_force([1], dof_index=1, total_force=-P)
load_factors, U_hist = solve_nonlinear_static(sys, mat, n_steps=20)

# through a limit point (snap-through), single-DOF control:
control_dof = 1 * sys.npn + 1
u_targets = -np.linspace(0, 2*h0, 41)          # prescribed apex y-displacement
U_hist, reaction_hist = solve_nonlinear_displacement_control(sys, mat, control_dof, u_targets)

# through a limit point, general case (arc-length): only a reference load
# DIRECTION is given -- lambda is solved for, not prescribed
sys.add_nodal_force([1], dof_index=1, total_force=-1.0)
load_factors, U_hist = solve_nonlinear_arc_length(sys, mat, delta_L=0.01, n_steps=60)

# same problem, Koiter-Newton continuation -- same delta_L "seed," but the
# cubic asymptotic predictor takes fewer, larger steps to cover the same path
load_factors, U_hist = solve_nonlinear_koiter_newton(sys, mat, delta_L=0.01, n_steps=60)

# a SINGLE prescribed target load (F_ext = 1.0*sys.F) instead of a whole path --
# e.g. generating one nonlinear-static ROM-training point cheaply
u = solve_nonlinear_static_koiter_newton(sys, mat)

# GENERIC (m>=1) continuation -- same call signature as solve_nonlinear_koiter_newton,
# but automatically adds a second perturbation direction whenever a near-critical
# tangent-stiffness mode is detected, instead of being limited to the primary path
load_factors, U_hist = solve_nonlinear_koiter_newton_generic(sys, mat, delta_L=0.01, n_steps=60)
```

`solve_nonlinear_arc_length()` is validated (`tests/test_arc_length.py`)
on the SAME von Mises truss and closed-form `P(δ)` as the other two
drivers: (1) a genuine regression/contrast check that
`solve_nonlinear_static()` really does fail past the limit point, as
its own docstring predicts; (2) arc-length traces `δ` from 0 out past
full inversion (`2·h0`), matching the closed form through BOTH the
ascending peak and the descending trough at ~1e-13 relative error; (3)
an independent cross-check against `solve_nonlinear_displacement_control()`'s
own (separately validated) path, wherever the two overlap. See
`examples/arc_length_snap_through_demo.py` (console) and
`examples/arc_length_snap_through_plot.py` (plots all three solvers'
reach against the closed-form S-curve, needs the optional `matplotlib`
dependency) -> `arc_length_snap_through.png`.

`solve_nonlinear_koiter_newton()` is validated
(`tests/test_koiter_newton.py`) on the SAME von Mises truss/closed-form
benchmark: (1)
traces the full path (through both limit points, past full inversion)
matching the closed form to <1e-6 relative error; (2) cross-checked
against `solve_nonlinear_arc_length()`'s own already-validated path to
<5e-3; (3) the actual claimed efficiency benefit, measured not assumed:
reaches a fixed target displacement in 2 outer steps vs. arc-length's 22
for the same `delta_L` seed (~11x fewer steps); (4) an isolated unit
check of the finite-difference quadratic/cubic force-coefficient
extraction against a hand-derived closed-form cubic force law for a
single Green-Lagrange truss bar loaded along its own axis. Implements
the Koiter-Newton (KN) method (Liang, Abdalla & Gurdal, IJNME 2013),
part of the derivation this package needed to let a sibling project,
`NonLin-HyROM`, reproduce Yang et al.'s 2019 hybrid-ROM paper's own
Section 4.2/Eqs. 16-22 — single-branch (`m=0`, the paper's own
"nonlinear hardening" case) predictor/corrector: a cubic asymptotic
predictor whose reduced quadratic/cubic coefficients come from finite
differences of the existing `assemble_internal_force()` (no per-element
analytic re-derivation needed), and a chord/modified-Newton corrector
that reuses ONE frozen-tangent bordered-matrix factorization for every
correction iteration in a step — see the function's own docstring for
the full derivation, and `docs/general_purpose_extensions_roadmap.md`
Section 9 for the motivation/scope/validation summary. **This is the
PATH CONTINUATION driver, not what that sibling project's ROM-training
loop actually ended up using** — see
`solve_nonlinear_static_koiter_newton()`'s own write-up just below for
why, and Section 10 of that same roadmap doc.

`solve_nonlinear_static_koiter_newton()` is validated
(`tests/test_static_koiter_newton.py`) on the SAME von Mises truss
benchmark (pre-limit-point, where a fixed-load equilibrium is
well-posed for any method), plus a hand-derived cubic-force-law unit
check and a forced-multi-expansion case confirming its shrink-and-
re-expand loop actually engages. It exists as a SEPARATE function from
`solve_nonlinear_koiter_newton()` because that continuation driver was
tried FIRST as `NonLin-HyROM`'s ROM-training `fom_solver` (via a
"run a few steps, interpolate back to lambda=1" wrapper) and failed
directly on the real Shell4MITCCorotational model: >14x overshoot past
the target load on one training sample (185.7s vs. plain Newton's
109.0s — slower, not faster, since the overshoot was pure waste), and
a floating-point overflow (NaN/Inf) on a larger sample. Re-reading the
target paper directly (Fig. 1's flowchart, Section 4.2's text) showed
this was the wrong tool, not a tuning problem — the static-ROM
machinery is meant to be invoked once PER prescribed target load, cubic
root aimed directly at that target, not grown freely. Validated
directly on the real `NonLin-HyROM` Case 1 model afterward (the exact 3
training samples that broke the wrapper): single-expansion convergence,
<5e-7 relative agreement with plain Newton, ~15-18x faster per solve —
and now wired into that project's own `fom_solver()` for real, with the
full pipeline re-run confirming the swap changes nothing about what the
training data converges to (see `docs/general_purpose_extensions_
roadmap.md` Section 10 for the full account, including the "these two
functions are not interchangeable" note).

`solve_nonlinear_koiter_newton_generic()` is validated
(`tests/test_koiter_newton_generic.py`, 4 tests): (1) with mode
interaction disabled, reproduces `solve_nonlinear_koiter_newton()`'s
own results on the same von Mises truss benchmark to ~1e-8 relative — a
direct no-regression check, since both implement identical `m=0`
formulas; (2)-(3) the actual hand-derivable 2-mode bifurcation
benchmark: a small synthetic 2-DOF system with a textbook coupled
symmetric/antisymmetric potential whose equilibrium set is exactly
closed-form (a linear primary path up to a critical load, past which a
secondary branch bifurcates off it) — because the potential is exactly
quartic, both the finite-difference tensor extraction and the Eq.7
algebraic formula can be checked against EXACT closed-form values, and
the resulting mode-interaction cubic's roots are shown to reproduce the
TRUE bifurcated-branch amplitude exactly, not approximately; (4) the
full driver run straight through that same synthetic bifurcation,
confirmed to detect the critical mode, exercise the two-direction code
path, and stay on the (still mathematically valid) primary branch to
near machine precision throughout, with no crash or divergence. See
`docs/general_purpose_extensions_roadmap.md` Section 11 for the full
derivation, the correctness review of Modules 21/22 that motivated
this addition (no bugs found — only the already-disclosed `m=0` scope
was incomplete), and the fea_engine-vs-rom_engine placement reasoning.

Validated (`validate_nonlinear.py`) against a **closed-form von Mises
(two-bar shallow) truss**: total-potential-energy stationarity gives
`P(δ) = (EA/L0³)·δ·(h0−δ)·(2h0−δ)` for the reaction at the apex as a
function of prescribed downward displacement `δ`, including the sign
change past the limit point. `solve_nonlinear_displacement_control()`
matches this to ~1e-14 relative error across the full snap-through path
(rising branch, peak, falling branch, negative branch, and the return
to zero at full inversion); `solve_nonlinear_static()` matches the same
closed form (via `scipy.optimize.brentq`) on the pre-limit-point branch
to ~1e-12. `TrussTL2D.internal_force()` was independently checked
against a hand-derived single-bar closed form (exact to machine
precision) and `tangent_stiffness()` against a finite difference of
`internal_force()` (~1e-9 relative).

**Material nonlinearity (elasto-plasticity, `TrussPlastic2D`)** uses
the SAME two drivers — `solve_nonlinear_static()` now also accepts a
`load_factors=` array instead of a single monotonic ramp, which is
what makes a load-unload-reload HISTORY possible (needed to see
permanent set / hysteresis at all — a single monotonic ramp can't show
it). After every converged step, both drivers call
`fesystem.commit_all_states(u, mat)` (harmless no-op for
`TrussTL2D`/every stateless element):

```python
plas = PlasticMaterial1D(E=200e9, sigma_y=250e6, H=20e9)
mat = (plas, A)
sys = FESystem(mesh, TrussPlastic2D())
sys.fix_dofs(...)
sys.add_nodal_force([tip_node], 0, F_ref)
sys.init_state()                                   # required before ANY nonlinear solve
                                                     # that uses a path-dependent element

lambda_seq = [0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, 0.0]   # load, unload, reverse, unload
load_factors, U_hist = solve_nonlinear_static(sys, mat, load_factors=lambda_seq)
```

Validated (`validate_plasticity.py`) three ways: (1) `internal_force()`
at several strains straddling yield matches a hand return-map to
machine precision; (2) `tangent_stiffness()` matches a finite
difference of `internal_force()` on both the elastic and plastic
branches (~1e-11/1e-10 relative); (3) a fixed-free bar driven through a
15-step load-unload-reverse-unload history, checked against an
INDEPENDENT replay of the resulting strain trajectory through a
separately-written return-map function — confirms equilibrium at every
step (~1e-8 absolute, ~3e-16 relative to `sigma_y`), a genuine nonzero
PERMANENT (plastic) strain remaining after the first full unload
(`eps_total == eps_p` there, to 1e-9), and that isotropic hardening
correctly raises the yield threshold from the virgin `sigma_y` to
`sigma_y + H*alpha` in BOTH tension and the later compressive reversal
— not just in whichever direction was loaded first.

**Contact/boundary nonlinearity** has two paths, matching the two
formulation types named in the module's spec:

| Approach | How | Function |
|---|---|---|
| Penalty | Register a `GapContactPenalty` via `sys.add_contact_element(...)`, then call `solve_nonlinear_static()` exactly as for any other nonlinear problem — no dedicated contact function needed | `solve_nonlinear_static(sys, mat, ...)` |
| Lagrange multiplier | A plain structural `sys` (NO contact element added) plus a dedicated driver that manages one extra scalar unknown itself | `solve_contact_lagrange_static(sys, mat, contact_node, n_hat, g0, ...)` |
| Augmented Lagrangian | **Wave 3 item 15** — same plain-`sys`, dedicated-driver shape as the Lagrange-multiplier row, but an outer Uzawa loop updates a multiplier estimate between ordinary inner (plain-penalty-style) Newton solves, driving penetration to near-zero without a huge `k_p` | `solve_contact_augmented_lagrange_static(sys, mat, contact_node, n_hat, g0, k_p, ...)` |

```python
# penalty:
sys.add_contact_element(GapContactPenalty(), [tip_node], (k_p, g0, n_hat))
load_factors, U_hist = solve_nonlinear_static(sys, mat, n_steps=60)

# Lagrange multiplier (exact):
load_factors, U_hist, lambda_hist, active_hist = solve_contact_lagrange_static(
    sys, mat, contact_node=tip_node, n_hat=n_hat, g0=g0, n_steps=60)

# augmented Lagrangian (near-exact, at a much smaller/better-conditioned k_p):
load_factors, U_hist, lambda_hist, active_hist = solve_contact_augmented_lagrange_static(
    sys, mat, contact_node=tip_node, n_hat=n_hat, g0=g0, k_p=k_p, n_steps=60)
```

**Node-to-segment contact (Wave 3 items 16-17)** — for a genuinely
DEFORMABLE obstacle (the master side is real mesh nodes, not a fixed
plane/circle), use `NodeToSegmentContact2D`/`NodeToSegmentContact2DFriction`
instead, discovering candidate pairs with `find_contact_pairs_2d()`:

```python
pairs = find_contact_pairs_2d(slave_node_ids, master_segments, mesh.nodes,
                               search_radius=0.5)
for slave, (ma, mb) in pairs:
    sys.add_contact_element(NodeToSegmentContact2D(), [slave, ma, mb], (k_p,))
load_factors, U_hist = solve_nonlinear_static(sys, mat, n_steps=60)
```

Same `add_contact_element()`/`solve_nonlinear_static()` pattern as
plain penalty above — no dedicated driver needed, since the "extra
unknowns" here are just two more ordinary mesh nodes per contact pair,
not a Lagrange multiplier. `master_segments` must be consistently
wound (see the class's own docstring for the outward-normal
convention); re-call `find_contact_pairs_2d()` with updated node
positions to refresh the candidate list as a simulation progresses.

Validated (`validate_contact.py`, 5 checks) on a `TrussTL2D` bar (the
already-validated Module 8 element — a deliberate composability
demonstration: geometric nonlinearity + contact, two different modules,
combined with no code changes to either) approaching a rigid wall with
gap `g0`: (1)-(2) `GapContactPenalty`'s force/tangent match a hand
formula and a finite difference exactly, on both the open and active
branches. (3) The full penalty-contact FE solve matches the closed-form
equilibrium `P = N_truss(u1) + k_p·max(u1−g0, 0)` — reusing Module 8's
own already-validated `N_truss` — to ~1e-8 relative error across a
sweep that spends 20 steps with the gap open and 41 in contact. (4) The
Lagrange-multiplier solve gives **exactly** zero penetration while
active (~1e-9, at the limit of the Newton tolerance, vs. the penalty
method's inherent, finite penetration), and its multiplier matches the
closed-form contact reaction `P − N_truss(g0)` to ~1e-10. (5) Sweeping
the penalty stiffness `k_p` from 1e7 to 1e12 shows penetration and the
error in the recovered contact force both shrinking roughly 10× per
decade of `k_p` — the penalty solution converging onto the exact
Lagrange-multiplier one, the classic textbook comparison between the
two formulations, reproduced numerically rather than just asserted.

### `postprocess.py` — derived dynamic quantities

`modal_participation_factors` / `effective_modal_mass` (how much of
the total excited mass each retained mode captures — the standard
diagnostic for "how many modes is enough" in modal superposition), and
`variance_from_psd` / `rms_from_psd` (trapezoidal PSD integration).

### `main.py` — validation driver

Runs 8 problems end to end and checks each against an independent
reference: 4 static (beam, plane-stress bracket, plate, 3-D solid) and
4 dynamic (implicit vs. explicit agreement, modal superposition vs.
direct integration, harmonic response recovering the static limit at
Ω=0, random vibration checked against an independent time-domain
Monte Carlo simulation). Run it directly (`python main.py`) as a smoke
test after any change to the package.

---

<a id="s-4"></a>

## 🪜 Implementation steps

### A. Any static problem

1. **Material/section** — `Material(E, nu, rho)`, `Section(A, I)` if using the beam.
2. **Mesh** — call the matching `mesh.py` generator (or write a new one).
3. **Element** — pick a class from `element.py` (or write a new one).
4. **System** — `sys = FESystem(mesh, element, thickness=...)`.
5. **Assemble** — `sys.assemble_stiffness(D, **kwargs)` with the matching `config.py` D-matrix.
6. **Loads and BCs** — `sys.add_nodal_force(...)` / `add_consistent_edge_load(...)`, then `sys.fix_dofs(...)`.
7. **Solve** — `U = sys.solve_static()`.
8. **Validate** — check against a closed-form/reference result (see any `main.py` item 1–4).

Steps 2–4 collapse into one call for the common cases: `sys, mesh, elem
= geometry.build_system(dim=2, Lx=0.4, Ly=0.2, nx=24, ny=12)` picks the
rectangle generator and `Quad4PlaneStress` automatically — see
`geometry.py`'s module reference above for `dim=1`/`3` and the
`shape=`/`physics=` options that pick something other than the default.

```python
mat = Material(E=2.1e11, nu=0.3, rho=7850.0)
mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=24, ny=12)
elem = Quad4PlaneStress()
sys = FESystem(mesh, elem, thickness=0.02)
sys.assemble_stiffness(D_plane_stress(mat), thickness=0.02)
tip = mesh.nodes_on_line(axis=0, value=0.4)
sys.add_nodal_force(tip, dof_index=1, total_force=-20000.0)
sys.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
U = sys.solve_static()
```

### B. Free vibration (natural frequencies / mode shapes)

Steps 1–4 as above, then:

```python
sys.assemble_mass(mat.rho * sec.A)     # or rho*np.eye(2)/3/etc. for 2-D/3-D/plate
freq_hz, mode_shapes = sys.solve_modal(n_modes=4)
```

### C. Direct time integration (impact, earthquake, general transients)

```python
sys.assemble_mass(rho_data)
damping = RayleighDamping.calibrate(omega_i, omega_j, zeta=0.02)
sys.assemble_damping(damping)          # implicit only

load = TimeHistoryLoad(LoadPattern(node_ids, dof_index), time_fn=lambda t: ...)

# implicit -- unconditionally stable, pick dt for ACCURACY
t, U_hist = sys.solve_transient_implicit(load, T_total, dt)

# explicit -- conditionally stable, pick dt for STABILITY
sys.assemble_lumped_mass(rho_data)
dt = 0.4 * sys.critical_timestep()     # safety margin below the limit
t, U_hist = sys.solve_transient_explicit(load, T_total, dt)
```

Use implicit for long-duration/low-frequency loads (seismic, wind,
waves) where large time steps matter; explicit for short-duration
high-speed events (impact, blast) where per-step cost matters more
than step count.

### D. Modal superposition (linear systems, many repeated load cases)

```python
sys.assemble_mass(rho_data)   # NOT assemble_damping -- zeta is applied per-mode
t, U_hist, freq_hz = sys.solve_modal_superposition(load, T_total, dt, n_modes=8, zeta=0.02)
```

Check `postprocess.effective_modal_mass()` to confirm `n_modes` is
enough — it should capture most of the total mass excited by your load
pattern (typically >90% for the first 5–10 modes of a distributed-mass
structure).

### E. Harmonic / frequency response (rotating machinery, vibration isolation)

```python
sys.assemble_mass(rho_data)
sys.assemble_damping(damping)
load = HarmonicLoad(LoadPattern(node_ids, dof_index), F0=1000.0)
F0_vec = load.force_vector(sys.n_dof, sys.npn)

U0 = sys.solve_harmonic(Omega=2*np.pi*50, F0_vector=F0_vec)        # single frequency
U = sys.solve_frequency_sweep(2*np.pi*freqs_hz, F0_vec)            # a sweep (Bode/FRF)
```

Sanity check: `solve_harmonic(0.0, F0_vec)` must equal `solve_static()`
exactly — always verify this after wiring up a new problem.

### F. Random vibration / PSD (turbulence, road/rail roughness, acoustic loading)

```python
sys.assemble_mass(rho_data)
sys.assemble_damping(damping)
psd_load = PSDLoad(LoadPattern(node_ids, dof_index), freqs_hz, psd_input)
F0_unit = psd_load.force_vector(sys.n_dof, sys.npn)   # always unit magnitude

S_out, sigma = sys.solve_random_vibration(freqs_hz, psd_input, F0_unit, output_dof=target_dof)
```

`sigma` is the response RMS. There's no deterministic peak to read off
— for a peak/fatigue estimate, post-process `S_out` statistically
(e.g. treat the response as narrowband Gaussian near a resonance).

### G. Adding a new element type end to end

1. In `element.py`: subclass `Element`, set `n_nodes`, `dofs_per_node`,
   `dim`, `gauss_order`, `translational_dof_mask`.
2. Implement `shape_and_derivs(natural_coords) -> (N, dN_natural)`.
3. Implement `B_matrix(natural_coords, elem_coords) -> (B, detJ)` to
   use the generic `stiffness()`/`mass()`, **or** override those
   methods directly for a closed-form/non-standard formulation.
4. Register it: `ELEMENT_REGISTRY["my_element"] = MyElement`.
5. In `config.py`: add a `D_xxx()` builder if it needs a new
   constitutive law.
6. In `mesh.py`: reuse an existing generator, or add a new one if the
   element needs a geometry none of the current ones produce.
7. Validate: build the smallest case with a known closed-form answer,
   add it to `main.py`, and run `mesh.check_quality()` before trusting
   any solved result.

### H. Choosing an integration scheme for an existing element

1. Start with the default (`stiffness()`/`assemble_stiffness()` with no
   extra arguments) — full integration for everything except the
   plate, which defaults to `'sri'`.
2. If the result looks too stiff for a bending-dominated problem,
   that's the locking signature (see `main.py` items 4 and 9–10 for
   what it looks like quantitatively). Before switching schemes, ask
   whether a beam/plate element is simply the better tool for this
   geometry (see the `Hex8Solid3D` limitation below) — that's usually
   the real fix.
3. If you still want reduced integration: call
   `sys.assemble_stiffness(D, gauss_order=n)` (Quad4/Hex8) or
   `sys.assemble_stiffness(D, integration='reduced')` (plate) — these
   pass straight through `solver.py`'s `**kwargs`, no code changes
   needed there.
4. **Before trusting the result**, check for spurious modes:
   ```python
   n_spurious, _ = spurious_zero_energy_modes(elem, single_elem_coords, D)
   # and/or, on the assembled+constrained system:
   eigs = np.linalg.eigvalsh(sys.K[np.ix_(sys.free_dofs, sys.free_dofs)])
   n_near_zero = np.sum(np.abs(eigs) < 1e-6 * np.max(np.abs(eigs)))
   ```
   If `n_near_zero` is large (see `main.py` item 9 — 82 for the Hex8
   cantilever), the system is unreliable regardless of how plausible
   the displacement number looks.

### I. Geometric nonlinearity (large displacement, monotonic or snap-through)

1. **Element** — use a nonlinear element (`TrussTL2D` for axial-only
   members, `Beam2DCorotational` for members that also bend through
   large rotations, or a new one built per section G above but
   overriding `internal_force()`/`tangent_stiffness()` instead of
   `stiffness()`).
2. **System** — `sys = FESystem(mesh, elem)` as usual; `mat` replaces
   `D` (e.g. `(E, A)` for `TrussTL2D`, `(E, A, I)` for
   `Beam2DCorotational`) and is passed straight through.
3. **BCs** — `sys.fix_dofs(...)` as usual.
4. **Decide the path parameter:**
   - Monotonic problem → `sys.add_nodal_force(...)`, then
     `solve_nonlinear_static(sys, mat, n_steps=...)`.
   - Possible snap-through → identify the DOF that best parametrizes
     the motion (e.g. the apex of a shallow arch), prescribe it
     directly via `solve_nonlinear_displacement_control(sys, mat,
     control_dof, u_target_array)`, and read the reaction history back
     for the load-displacement curve.
5. **Validate** — for a new geometry, derive (or look up) the
   closed-form load-displacement relation via total-potential-energy
   stationarity (`P = dU/dδ`) the way `validate_nonlinear.py` does for
   the von Mises truss, and check the solver against it at several
   points along the path, not just one.

### J. Material nonlinearity (elasto-plasticity)

1. **Element** — use a path-dependent element (`TrussPlastic2D`, or a
   new one built per section G but adding `init_state()`/
   `commit_state()` and a `state=` kwarg to `internal_force()`/
   `tangent_stiffness()` per the "path-dependent state" extension
   point above).
2. **System** — `sys = FESystem(mesh, elem)`; `mat` is
   `(constitutive_object, A)` (e.g. `(PlasticMaterial1D(E, sigma_y, H), A)`).
3. **BCs and loads** — as usual.
4. **`sys.init_state()`** — required before the first nonlinear solve;
   skip it and every element behaves as if stateless (elastic forever).
5. **Decide the load HISTORY, not just a target load** — a plasticity
   result depends on the whole path, not just the final state. Build a
   `load_factors` array that actually visits the load reversals you
   care about (see the example above), and pass it to
   `solve_nonlinear_static(sys, mat, load_factors=...)`.
6. **Validate** — check `internal_force()` against a hand return-map at
   a few strains straddling yield (cheap, catches sign/formula errors
   immediately), check `tangent_stiffness()` against a finite
   difference on BOTH the elastic and plastic branches (the tangent is
   discontinuous at yield, so don't finite-difference across the kink
   itself), then run a load-unload-reload history and confirm a
   nonzero permanent set remains after unloading to zero load — the
   single clearest sign the plasticity is actually doing something
   (an elastic-only bug looks identical to a correct one until you
   unload and check for residual strain).

### K. Contact / boundary nonlinearity

1. **Structure and obstacle** — build the structural mesh/system as
   usual; the obstacle itself is NOT part of the mesh, just a
   `(g0, n_hat)` pair (gap distance and approach direction) relative to
   whichever node might touch it.
2. **Choose a formulation:**
   - **Penalty** (simpler, approximate) — pick `k_p` large relative to
     the structure's own stiffness (start around 100–1000×, check that
     penetration is small relative to the deformation you care about),
     `sys.add_contact_element(GapContactPenalty(), [node], (k_p, g0, n_hat))`,
     then solve with the ordinary `solve_nonlinear_static()` — nothing
     contact-specific about the call.
   - **Lagrange multiplier** (exact, costs one extra unknown) — do NOT
     register a contact element; call
     `solve_contact_lagrange_static(sys, mat, contact_node, n_hat, g0, ...)`
     instead, which returns the displacement history AND the contact
     reaction (`lambda_hist`) and active/inactive flag
     (`active_hist`) at every step.
3. **Step size matters more here than elsewhere** — the tangent is
   *discontinuous* at the contact transition (not just its derivative,
   unlike the plasticity kink), so use enough load steps that no single
   increment tries to jump clean over the transition.
4. **Validate** — check equilibrium at a few converged steps by hand
   (`P_applied == N_structure(u) + contact_force(u)`, the same pattern
   `validate_contact.py` uses), and if using the penalty method, sweep
   `k_p` and confirm penetration shrinks as it increases — the standard
   sanity check that you haven't just picked an arbitrary stiffness
   that happens to look right at one value.

### L. Friction + updating contact normal (curved obstacle, Module 13)

1. **Obstacle is a circle, not a flat wall** — `mat = (k_p, k_t, mu,
   center, R)` instead of `GapContactPenalty`'s `(k_p, g0, n_hat)`; the
   normal direction `n_hat` is now RECOMPUTED from the node's current
   position every call, not a fixed material parameter.
2. **Register it exactly like any other contact element** —
   `sys.add_contact_element(GapContactCurvedFriction(), [node], mat)`,
   then solve with the ordinary `solve_nonlinear_static()`; nothing
   about the driver call changes.
3. **`sys.init_state()` is still required** if this is the only
   path-dependent element in the system — `add_contact_element()` gives
   the contact element its OWN state slot automatically (a `commit_state()`-
   updated stick-anchor arc-length), independent of whether the main
   mesh has stateful elements too.
4. **Pick a structural test geometry with genuine transverse stiffness
   at `u=0`** — a single straight 2-node truss has ZERO tangent
   stiffness transverse to its own axis before it deflects (a
   mechanism), which makes Newton's very first iteration hit a singular
   `K_T` if the applied load and the obstacle are both off-axis. Use an
   inclined/multi-member structure (e.g. a shallow "V" truss, the same
   fix used in `validate_curved_contact.py` CHECK 4) so the free node
   has real stiffness in every direction from the start.
5. **Validate** — hand-check `pn`/`n_hat`'s sign and direction at a few
   positions straddling the obstacle boundary; for `mu=0`, cross-check
   the element's finite-difference `tangent_stiffness()` against the
   closed form `K_T = k_p*(1-R/d)*I + k_p*(R/d)*(n_hat⊗n_hat)`; sweep a
   node along the obstacle at fixed penetration depth and confirm
   `commit_state()` produces the expected stick-then-slip-plateau
   sequence; and confirm `n_hat` genuinely differs at two different
   tangential positions on the same obstacle (the property a flat-wall
   `GapContactPenalty` cannot have) — the exact sequence
   `validate_curved_contact.py` runs.

### M. Mixed-element-type assembly (Module 14)

1. **Get a `MultiBlockMesh`** — build one by hand (`nodes` array
   plus `blocks = {name: connectivity_array, ...}`, all blocks indexing
   the same shared node list). (The former Gmsh-backed producer,
   `geometry_engine.py`'s `_extract_mesh()`/`generate_2d_rectangle_mixed()`,
   was removed along with Gmsh support -- see "Removed: Gmsh support"
   above.)
2. **Build a matching dict of element formulations** — one entry per
   block name, e.g. `{"tri3": Tri3PlaneStress(), "quad4":
   Quad4PlaneStress()}`. Every formulation in the dict must have the
   SAME `dofs_per_node` — `FESystem` raises a `ValueError` immediately,
   naming the mismatched values, if not.
3. **Construct exactly like the single-element case, just with a dict**
   — `sys = FESystem(mixed_mesh, {"tri3": ..., "quad4": ...})`; every
   downstream call (`assemble_stiffness`, `fix_dofs`, `add_nodal_force`,
   `solve_static`, `solve_modal`, the nonlinear drivers, ...) is
   IDENTICAL to the single-element case — nothing about calling code
   needs to know a mesh has more than one element topology.
4. **Pass per-block materials only if you actually need them** — `D`/
   `mat`/`rho` can be one value shared by every block (the common case:
   same physics, just more than one element topology) or a dict keyed
   the same way as the blocks, for genuinely different
   materials/physics per block.
5. **Validate** — a patch test (impose a uniform strain field across
   ALL blocks' shared nodes, confirm every element type recovers it
   exactly — this also checks inter-block compatibility at shared
   edges, not just each element type in isolation); the usual exact
   equilibrium check; and, for a mesh from an external unstructured
   mesher, confirm
   `mesh.blocks` actually contains more than one key before trusting
   the mixed path was exercised (a coarse/fine `mesh_size` can
   legitimately recombine fully, leaving only one block) — the exact
   sequence `validate_mixed_elements.py` runs.

### N. Visualization (geometry / mesh / result plots)

Gmsh-native visualization (`visualization/gmsh_plot.py`, Module 16 — a
second, non-default OpenGL rendering path) was removed along with Gmsh
support entirely; see "Removed: Gmsh support" above. Use `mesh.py`'s
matplotlib functions instead, which remain the supported path and are
what every example script in this package actually uses:
`mesh.plot_mesh_2d(mesh, filename, ...)`/`mesh.plot_mesh_3d(mesh,
filename, ...)` for geometry/mesh figures, and
`mesh.plot_mesh_annotated(...)` for result overlays (deformed shape,
scalar fields) — see those functions' own docstrings for the full
parameter set.

---

<a id="s-5"></a>

## 🧰 General-purpose extensions (Phases 1-7)

`docs/general_purpose_extensions_roadmap.md` researches and phases
seven general-purpose gaps (higher-order elements, shells, 3-D
beams/frames, 2-D/3-D plasticity + hyperelasticity, arc-length/Riks,
linear buckling, sparse matrices) to build out BEFORE any
problem-specific work (fluid/FSI coupling, modal identification) —
matching this project's research-first, phased-build pattern. Phase 1
(the lowest-effort, highest-value item — the generic Gauss loop needs
zero changes) is done: `Quad8PlaneStress`, `Hex20Solid3D`,
`Tet10Solid3D` (see the module reference table above). Validated in
`tests/test_higher_order_elements.py`: isoparametric-identity +
exact constant-strain patch tests, rigid-body-mode counts, consistent-
mass totals for all three (the same depth `Tri3PlaneStress`/
`Tet4Solid3D` were validated to), plus a genuine two-element
interior-node patch test for `Quad8PlaneStress` (passes at ~1e-19,
literal machine precision) and an independent closed-form strain-
energy check for `Tet10Solid3D`. `Hex20Solid3D`/`Tet10Solid3D`
deliberately do NOT get the same multi-element interior-node patch
test — discovered, while actually attempting it, that a minimal
2-element tet/hex mesh can't provide a genuinely interior node for a
serendipity solid element (every node is still a vertex of some outer
face); documented as a real scoping limit in the test file's module
docstring, not silently dropped.

**Gmsh compatibility (found + fixed 2026-08-30):** the validation above
covers element CORRECTNESS given already-correct node coordinates —
it doesn't by itself guarantee a mesh generated by an external tool
hands nodes to these elements in the right order. Cross-validating
`Hex20Solid3D`/`Tet10Solid3D` against Gmsh (via the wing-cantilever
example in the sibling `Multi_Fidelity_NL_Structural_ROM` project)
surfaced exactly that gap: Gmsh's native Hexahedron20 (element type 17)
and 10-node tetrahedron (element type 11) node orders don't match these
classes' assumed orders exactly (severely for Hex20, subtly — two
swapped mid-edge nodes — for Tet10). Both classes now expose a
`GMSH_NODE_ORDER` permutation (`gmsh_conn[:, GMSH_NODE_ORDER]`) fixing
this, documented in each class's own "GMSH COMPATIBILITY WARNING"
docstring note and originally validated in `tests/test_gmsh_node_order.py`
against real Gmsh-generated reference elements (not hand-typed
assumptions of what Gmsh "should" produce; that test was removed along
with Gmsh support -- see "Removed: Gmsh support" above -- but the
`GMSH_NODE_ORDER` permutation tables themselves are independent, static
data and remain in `elements/solids.py`). `Quad8PlaneStress` got the
same direct check before `geometry/gmsh_engine.py` wired it up
(2026-08-30) and came back CLEAN — Gmsh's native 8-node quadrangle
(type 16) already matches this class's own node order, so its
`GMSH_NODE_ORDER` is the identity, unlike Hex20's severe or Tet10's
subtle mismatch. All three permutations used to be applied
automatically by the now-removed `geometry_engine.py`'s
`_extract_mesh()` when a mesh came from Gmsh
(`docs/geometry_meshing_alternatives_research.md` Section 7 item 2 --
this whole cross-validation effort is retained here for historical/
design-rationale context, though the Gmsh-facing half of it no longer
applies now that Gmsh support has been removed).

`examples/higher_order_elements_demo.py` runs the headline
demonstration the roadmap called for: at an IDENTICAL, coarse element
count (2 elements long, 1 through the height), `Quad4PlaneStress`
shear-locks under pure bending (~33% too stiff — the textbook 2/3
single-element-through-thickness ratio) while `Quad8PlaneStress`
matches the exact elasticity solution (pure bending has an EXACT,
not just approximate, elasticity solution) to within numerical
precision, at the same or fewer DOF.

```bash
python examples/higher_order_elements_demo.py
```

`examples/higher_order_elements_plot.py` (needs the optional `plot`
extra, i.e. `matplotlib`) draws the same comparison as a picture, not
just numbers: a bar chart of the tip-deflection ratio to the exact
solution, and the two elements' actual DEFORMED shapes overlaid on
their undeformed outline (Quad4 barely curves under the applied
moment, Quad8 traces the smooth bent beam) -> `higher_order_elements_
locking.png`.

Phase 2 (sparse matrix support) is also done: `FESystem(mesh, elem,
sparse=True)` stores `K`/`M`/`M_lumped` as `scipy.sparse.lil_matrix`
during assembly, converting to CSR at solve time -- `solve_static()`
uses `scipy.sparse.linalg.spsolve()`, `solve_modal()` uses
`scipy.sparse.linalg.eigsh()` with shift-invert (`sigma=0`) for the
lowest modes. Purely additive: `sparse=False` (the default) is
byte-for-byte the original dense path, unchanged. Validated in
`tests/test_sparse_assembly.py` to agree with the dense path to near
machine precision on both static and modal solves, and measured (not
assumed) to be dramatically faster on a larger model — 24x on a
~2,000-dof cantilever in one test run, growing further with size
(`examples/sparse_vs_dense_cantilever.py` shows the crossover: sparse
has more fixed overhead so it's actually SLOWER below a few hundred
dof, then pulls sharply ahead — reported honestly, not hidden). NOT
yet wired into the nonlinear (`assemble_internal_force`/
`assemble_tangent_stiffness`) or transient-dynamics paths — scoped to
static/modal for this first pass; see the roadmap doc's Section 7 for
the rationale.

```bash
python examples/sparse_vs_dense_cantilever.py
```

`examples/sparse_vs_dense_plot.py` plots the same timing sweep: solve
time vs `n_dof` (log-log, dense vs sparse) alongside the speedup ratio
on its own axis, so the crossover point and the "speedup grows with
size" claim are both visible curves, built from one set of timing
runs shared by both panels -> `sparse_vs_dense_timing.png`.

Phase 3 (3-D beam/frame elements) is also done: `Beam3DEulerBernoulli`
(`elements/beams3d.py`, see the module reference table above) — a
2-node, 6-dof/node closed-form element built from four uncoupled
local blocks (axial, torsional, and two independent Hermite bending
blocks, the second reusing the first's formula via a sign-flip
congruence transform) rotated to global coordinates through a
right-handed local frame. New `Section3D` dataclass and
`beam3d_rigidities()`/`beam3d_mass_props()` helpers in `material.py`
carry the two independent bending rigidities (`Iy`, `Iz`) plus a
torsional constant `J` that is deliberately NOT assumed equal to
`Iy+Iz` (only true for circular sections — see `Section3D`'s
docstring). Validated in `tests/test_beam3d.py` — being fully
closed-form, every check is an EXACT match (not a discretization-error
tolerance): axial/torsional/bending blocks against 1-D cantilever
closed forms (both axis-aligned AND at a skew orientation, so the
general local-frame rotation matrix is actually exercised, not just
its axis-aligned special case); the mass matrix's rigid-translation
and rigid-twist totals against `rho*A*L`/`rho*Ip*L` exactly; the
flat-plane-loading reduction against `Beam2DEulerBernoulli`'s own
stiffness/mass matrices bit-for-bit (0.0 diff — literally the same
closed-form call); and a 2-element L-shaped space frame assembled
through `FESystem` checked against an independently written
reference element-stiffness function + manual scatter-assembly.

```bash
python examples/beam3d_space_frame_demo.py
```

`examples/beam3d_frame_plot.py` draws the same L-frame in 3-D,
undeformed (dashed) against its exaggerated deformed shape under the
tip load, with the load direction and the fixed base both marked --
and prints the tip displacement alongside the plot so the picture and
the number that `tests/test_beam3d.py` validates are shown together
-> `beam3d_frame_deformed.png`.

Phase 4 (linear buckling eigenvalue solver) is also done: a new
`geometric_stiffness(elem_coords, N, ...)` extension point on
`Element` (`elements/base.py`) is a sibling to the existing
`internal_force()`/`tangent_stiffness()` nonlinear extension point,
but for the second-order/geometric ("stress stiffness") effect a
reference axial force `N` (TENSION-POSITIVE, matching `TrussTL2D`'s
existing stress convention) has on transverse stiffness. Implemented
for `Beam2DEulerBernoulli` (the closed-form Przemieniecki geometric
stiffness matrix), `Beam3DEulerBernoulli` (reuses `Beam2DEulerBernoulli`'s
formula for both bending planes, the same reuse pattern Phase 3's
`stiffness()`/`mass()` already use), and `TrussTL2D` (literally the
`K_geometric` term already inside its own `tangent_stiffness()`,
exposed on its own — no new physics, just a new entry point onto
existing, already-validated code). `FESystem` gets
`assemble_geometric_stiffness(N, **kwargs)` (mirrors
`assemble_stiffness()`'s per-block/per-element loop) and
`solve_linear_buckling(n_modes=4)` (mirrors `solve_modal()`'s
structure: `K*phi = -lambda*K_sigma*phi`, `load_factors` are exactly
the multiplier on the reference axial force state used to build
`K_sigma`). Validated in `tests/test_linear_buckling.py` against the
textbook Euler column closed form `P_cr = pi^2*EI/(K*L)^2` for three
boundary conditions (pin-pin, fixed-free, fixed-fixed), with genuine
mesh-refinement convergence (error shrinks monotonically as elements
are added, down to ~1e-7 relative at 32 elements) — not just a single
matched data point — plus the flat-plane-reduction and
`TrussTL2D`-self-consistency checks Phase 3's pattern established.
Also documents a real, honestly-scoped limitation found while writing
the 3-D column test: with both bending planes simultaneously free, the
assembled `-K_sigma` is not always positive definite, so
`solve_linear_buckling()` falls back from `scipy.linalg.eigh()`
(needs positive-definiteness) to the general `scipy.linalg.eig()` on
`LinAlgError` — see that method's docstring.

```bash
python examples/euler_column_buckling_demo.py
```

`examples/euler_column_buckling_plot.py` plots (1) the mesh-refinement
convergence on log-log axes against an `O(n_elem^-2)` reference slope
-- a straight line at the expected slope is the "convincing" signature
of a correctly implemented geometric stiffness, not just a small final
error -- and (2) the actual first buckling mode shape overlaid on the
exact `sin(pi*x/L)` half-sine curve, a qualitative check a table of
numbers alone can't give -> `euler_column_buckling.png`.

Every plotting script above needs the optional `matplotlib` dependency
(`pip install fea_engine[plot]`, or just `pip install matplotlib` in
this dev environment) and is otherwise independent of the corresponding
`_demo.py` script -- run either or both.

Phase 5 (arc-length/Riks solver) is also done: `solve_nonlinear_
arc_length()`, Crisfield's cylindrical arc-length continuation, added
to `nonlinear_solver.py` as a third path-following strategy alongside
`solve_nonlinear_static()`/`solve_nonlinear_displacement_control()` --
see the `nonlinear_solver.py` module reference table above for full
details, validation summary, and a runnable example.

Phase 6 (2-D/3-D J2 plasticity + Neo-Hookean hyperelasticity) is also
done: `Hex8PlasticJ2` and `Tet4NeoHookean` (`elements/nonlinear_
solids.py`, see the module reference table above), plus
`PlasticMaterialJ2`/`NeoHookeanMaterial`/`j2_radial_return_3d()`/
`neo_hookean_pk2_stress()` in `material.py`. J2 plasticity uses the
classical closed-form radial-return mapping (a hypersphere in
deviatoric stress space) with a hand-derived-and-cross-validated
consistent (algorithmic) tangent — matched to `D_elastic -
(6*mu^2/(3*mu+H))*(N⊗N)` in the `dgamma->0` limit and to finite
difference at 1e-10 relative, everywhere else. Neo-Hookean
hyperelasticity uses an isochoric/volumetric strain-energy split
(`W = (mu/2)*(bar_I1-3) + (kappa/2)*(J-1)^2`) with `S=2*dW/dC` giving
a closed-form 2nd Piola-Kirchhoff stress, but a finite-difference
tangent (see the module reference table entry — same established
precedent as `GapContactCurvedFriction`). Both subclass their
linear-elastic parents (`Hex8Solid3D`/`Tet4Solid3D`), reusing
`shape_and_derivs()`/`B_matrix()`/`mass()` unchanged. Validated in
`tests/test_plasticity_j2.py` (5 tests: elastic branch matches
`D_solid3d`, consistent tangent matches finite difference, uniaxial
stress matches `PlasticMaterial1D`'s already-validated 1-D closed
form exactly, monotonic-load-then-unload shows the correct permanent
set, rigid-body modes/zero state) and `tests/test_hyperelastic.py`
(4 tests: small-strain limit matches `D_solid3d` with the correct
O(strain) convergence rate, exact deformation-gradient recovery,
rigid-body modes/zero state, and — the decisive end-to-end check — a
full incremental Newton-Raphson solve where external work equals
independently-computed stored strain energy to 3.7e-7 relative,
confirming the FD tangent is Newton-quality-good, not just
self-consistent in isolation).

```bash
python examples/plasticity_j2_demo.py
python examples/hyperelastic_neo_hookean_demo.py
```

`examples/plasticity_j2_plot.py` plots a single `Hex8PlasticJ2`
element's uniaxial stress-strain loop against `PlasticMaterial1D`'s
closed-form loading curve and the exact elastic-unload line -- the
visual signature of correct plasticity (a sharp kink at yield, a
straight-line unload with permanent set) made directly checkable by
eye -> `plasticity_j2_uniaxial_loop.png`. `examples/hyperelastic_
neo_hookean_plot.py` plots (1) the small-strain convergence on
log-log axes against an `O(strain)` reference slope and (2) external
work vs. stored strain energy tracked along the entire loading path
(not just compared once at the end) -> `hyperelastic_neo_hookean.png`.

Phase 7 (MITC4 general shell elements) is also done: `Shell4MITC`
(`elements/shells.py`, see the module reference table above), plus
`D_shell()`/`shell_rho_matrix()` in `material.py`. Membrane and bending
are standard displacement-based (literal reuse of `Quad4PlaneStress`'s
`B_matrix()` and `Quad4MindlinPlate`'s `_Bb_Bs()`'s `Bb`, in a per-
element flat local frame); transverse shear uses a genuine Dvorkin-
Bathe MITC4 assumed-natural-strain tying-point interpolation (Dvorkin &
Bathe, *Eng. Comput.* 1(1):77-88, 1984), not selective reduced
integration -- the roadmap's headline requirement that this actually
FIX shear locking rather than merely relieve it. Validated in
`tests/test_shell.py` (5 tests): single-element rigid-body/zero-energy
modes; a stringent FRAME-OBJECTIVITY test (solve a cantilever, then
solve the identical physical problem rigidly rotated, and confirm the
rotated-back answer matches exactly) that actually caught a real bug
during development -- `Quad4MindlinPlate`'s own `(w, betax, betay)`
convention defines `betax` as the SLOPE-type rotation producing
x-direction bending curvature, which is physically a rotation ABOUT
THE Y-AXIS, not the x-axis; naively scattering `(betax, betay)`
straight into `(theta_x, theta_y)` true-rotation-vector slots is
invisible on a flat, axis-aligned mesh (where the local-to-global
rotation is the identity) but wrong for any curved or tilted one (see
`elements/shells.py`'s `_BEND` comment for the fix); flat-plate-limit
convergence to `Quad4MindlinPlate`'s own bending answer (0.03%
agreement at a fine mesh -- MITC4's assumed-strain shear and SRI shear
are different, both-valid anti-locking treatments, so they should
converge together, not necessarily match exactly at one coarse mesh);
a multi-element CURVED rigid-body-mode check (confirms the per-element
local-frame + rotation + assembly logic stays energy-consistent when
adjacent elements have genuinely different orientations); and a curved
cylindrical-arc cantilever strip validated by mesh-refinement
convergence against an independent Hex8Solid3D reference mesh of the
same geometry (Hex8Solid3D is separately documented below as locking
significantly in bending, so convergence -- not exact agreement at one
coarse mesh -- is the honest thing to check).

```bash
python examples/shell_demo.py
```

`examples/shell_plot.py` plots (1) the flat-plate-limit ratio
converging to 1 as the mesh refines and (2) the curved cylindrical-arc
cantilever's mid-surface mesh, undeformed (dashed) against its
exaggerated deformed shape under a radial tip load -> `shell_
verification.png`.

Phase 8 (Koiter-Newton continuation, added later than the original
seven-item survey — driven by a real external need, not that survey
itself) is also done: `solve_nonlinear_koiter_newton()`
(`nonlinear_solver.py`, see the module reference table above), a
single-branch (`m=0`) predictor/corrector PATH CONTINUATION driver
generalizing Phase 5's arc-length driver with a CUBIC (Koiter
asymptotic) predictor instead of a linear tangent one, corrected via a
frozen-tangent chord iteration that needs only ONE matrix factorization
per step (predictor AND corrector together). Written because the
sibling `NonLin-HyROM` paper-reproduction project needed it -- Yang et
al.'s 2019 hybrid-ROM paper's own Section 4.2 uses this method (its
Eqs. 16-22) to make its nonlinear-static ROM-training solves cheap, and
neither `fea_engine` nor `rom_engine` had any Koiter/asymptotic-
expansion module before this (confirmed by grep, zero hits). Validated
in `tests/test_koiter_newton.py` (4 tests) on the SAME von Mises truss/
closed-form benchmark Phase 5 uses — see the module reference table
above for the full validation summary, and `docs/general_purpose_
extensions_roadmap.md` Section 9 for the motivation, derivation, and
explicitly-out-of-scope `m≥1` (closely-spaced buckling-mode
interaction) extension.

Phase 9 (Koiter-Newton solve-to-target-load) is also done:
`solve_nonlinear_static_koiter_newton()` (`nonlinear_solver.py`), a
SEPARATE function from Phase 8's continuation driver, written
immediately after Phase 8's driver was actually tried as the
`NonLin-HyROM` project's ROM-training `fom_solver` and failed on the
real target model — an open-ended continuation driver has no concept
of "stop at this one prescribed load," so a naive wrapper around it
overshot the target load by >14x on one training sample (slower
overall than plain Newton) and drove the shell element into a
floating-point overflow on another. This driver instead aims its cubic
predictor's step DIRECTLY at the remaining gap to a single known
target load (`F_ext = 1.0*fesystem.F`, same convention as
`solve_nonlinear_static()`) rather than growing until an open-ended
validity check fails, so it can never overshoot; because the target
load is then externally known, its corrector needs no bordering trick
at all, just ordinary frozen-tangent Newton at that fixed load.
Validated in `tests/test_static_koiter_newton.py` (3 tests) and then
directly on the real `NonLin-HyROM` Case 1 model (3 of the exact
training samples that broke Phase 8's driver): single-expansion
convergence, <5e-7 relative agreement with plain Newton, ~15-18x faster
per solve. Now wired into that project's `case1_flat_plate_with_hole.
py` as its actual `fom_solver()` — see `docs/general_purpose_
extensions_roadmap.md` Section 10 for the full derivation and the
explicit "these two functions are not interchangeable" note.

Phase 10 (generic, `m≥1` Koiter-Newton continuation) is also done:
`solve_nonlinear_koiter_newton_generic()` (`nonlinear_solver.py`),
written after a direct review of Phases 8-9 found no bugs — only the
`m≥1` scope reduction both already disclosed. A brand-new function
(Phase 8's driver is unchanged), it adds automatic near-critical-mode
detection and a second perturbation direction to the same bordered-
matrix machinery, falling back to Phase 8's own exact `m=0` formulas
whenever no critical mode is found. Validated in
`tests/test_koiter_newton_generic.py` (4 tests): a direct regression
check against Phase 8's own results, a hand-derivable synthetic 2-mode
pitchfork-bifurcation benchmark whose exact closed-form bifurcated-
branch amplitude the reduced cubic equation is shown to reproduce
exactly (not approximately), and an end-to-end robustness run of the
full driver through that same bifurcation. See `docs/general_purpose_
extensions_roadmap.md` Section 11 for the full derivation, the
correctness-review finding, and the fea_engine-vs-rom_engine placement
reasoning (fea_engine: this method's "reduced model" is rebuilt from
scratch every step and never has a persistent, queryable existence the
way `rom_engine`'s POD/Galerkin/NNM models do).

Every general-purpose-extensions phase (1-9), plus the later Phase 10
addition, is now implemented and validated; see `docs/general_purpose_
extensions_roadmap.md`'s Section 8 for the original seven-item
build-order plan this delivered against, and its Sections 9-11 for
Phases 8-10's own later additions.

---

<a id="s-6"></a>

## ⚠️ Known limitations (by design, not oversights)

- **Structured/mapped meshes only** — no unstructured/Delaunay
  generation. Fine for the simple geometries this package targets.
  (Gmsh-backed unstructured meshing was previously available as an
  optional dependency but has been removed entirely — see "Removed:
  Gmsh support" above — so an arbitrary/unstructured shape now needs an
  external tool and a hand-off into `mesh.Mesh`/`mesh.MultiBlockMesh`,
  not a built-in generator.)
- **Mixed-element-type assembly (Module 14) covers same-`dofs_per_node`
  topology mixes only.** `mesh.MultiBlockMesh` + `FESystem(mesh,
  {block_name: element_formulation, ...})` lets a mesh with more than
  one element TOPOLOGY (e.g. `Tri3PlaneStress` + `Quad4PlaneStress`,
  the case a Gmsh "recombine" pass with leftover triangles used to
  produce back when this package had Gmsh support) assemble through the
  same `K`/`M`/`F_int`/`K_T` every uniform mesh always has —
  `assemble_stiffness()`/`assemble_mass()`/
  `assemble_internal_force()`/`assemble_tangent_stiffness()`/
  `init_state()`/`commit_all_states()` all loop over blocks internally
  now, with the original single-formulation usage normalizing to
  exactly one block (verified unchanged in behavior against every
  earlier validation script — see `validate_mixed_elements.py`).
  `D`/`mat`/`rho` may be a single value shared by every block (the
  common case: one physics, just more than one topology) or a dict
  keyed the same way as the blocks for per-block materials/physics.
  What this does NOT do: every block sharing one `FESystem` must use
  the SAME `dofs_per_node` — global DOFs are numbered per NODE, not per
  block, so mixing element FAMILIES with different `dofs_per_node`
  (e.g. a plate's `[w,betax,betay]` with a plane-stress element's
  `[u,v]`) in one mesh needs a fundamentally different multi-field DOF
  numbering scheme, not this mechanism, and raises a clear `ValueError`
  rather than silently mishandling it. `contact_elements`/
  `contact_state` (Module 10/13) are unaffected — that side-channel
  mechanism already supported per-element formulations before Module
  14 existed, and composes with mixed-block meshes with no changes.
- **`Tri3PlaneStress`/`Tet4Solid3D` are linear (constant-strain)
  elements** — like any CST/Tet4, they converge more slowly than
  Quad4/Hex8 for the same node count (`validate_geometry_engine.py`
  CHECK 3 shows a coarse mesh over-shooting the classical `Kt=3` stress
  concentration at a hole for exactly this reason) and have no
  quadrature order to trade off (`full_stiffness()`/`reduced_stiffness()`
  are both aliases for the same closed form) — the only lever for
  accuracy is mesh refinement.
- **`geometry.py` only covers the shapes/elements already registered**
  — it's a convenience front end over `mesh.py`/`element.py`, not a new
  meshing capability, so it inherits every limitation below (structured
  meshes, `Hex8Solid3D` locking, etc.). Adding a genuinely new geometry
  or default element still means writing it in `mesh.py`/`element.py`
  first, then registering it here — `geometry.py` never needs to change
  shape-specific logic, only its registries.
- **`Hex8Solid3D` locks in bending** — full-integration trilinear
  solids are a poor choice for slender members (see `main.py` item 4
  and `cantilever_beam_3d_fem.py` for the full discussion). Use
  `Beam2DEulerBernoulli` or a shell/plate element for slender geometry;
  reserve solids for genuinely 3-D/thick parts. `Hex8SolidBbar` (Wave 2
  item 11) relieves the ADDITIONAL volumetric-locking component as
  ν→0.5 specifically, but does not eliminate `Hex8Solid3D`'s baseline
  shear locking in bending — for that, `hourglass_stabilized_stiffness()`
  (below) or a shell/beam element are still the right tools.
- **Hourglass control for reduced integration: now available**
  (Wave 2 item 12) — `hourglass_stabilized_stiffness()` /
  `method="hourglass_stabilized"` stabilizes the zero-energy modes
  plain `reduced_stiffness()` can introduce (quantified in `main.py`
  item 9: a single reduced-integration `Hex8` has exactly 12 spurious
  modes, and the assembled cantilever-beam mesh is numerically
  singular under PLAIN reduced integration). Plain `reduced_stiffness()`
  remains available unstabilized, for callers who want it; the plate's
  `'sri'` mode (bending stays fully integrated; only the shear term is
  reduced, by construction not hourglass-prone) is a separate,
  still-available mechanism.
- **`RayleighDamping` only for the global `C` matrix** — non-classical
  (non-proportional) damping would need a complex/state-space
  eigenproblem, not implemented here.
- **Geometric nonlinearity: two elements (axial + bending), two
  path-following strategies, static only.** `TrussTL2D` (Total
  Lagrangian, Green-Lagrange strain, St Venant-Kirchhoff/2nd-Piola-
  Kirchhoff, `nonlinear_solver.py`) is validated against a closed-form
  snap-through benchmark; `Beam2DCorotational` (Module 15, corotational
  large-rotation beam-column) is validated against the exact Euler
  elastica solution. Both are **large-displacement/small-strain**
  elements only (no large material stretch/volume change — that needs
  a genuinely nonlinear constitutive law, not just nonlinear kinematics
  on a linear one). There is still no nonlinear plate or continuum
  (Quad4/Hex8) element, and no dynamic (time-stepping) nonlinear solve.
  `solve_nonlinear_arc_length()` (Module 19, general-purpose extensions
  Phase 5) now provides general Crisfield arc-length continuation
  alongside the single-DOF `solve_nonlinear_displacement_control()` --
  see `nonlinear_solver.py`'s module reference table above -- but it
  uses a FIXED arc-length radius per step (no automatic adaptation
  based on the previous step's iteration count, a common refinement
  in production arc-length codes, not implemented here).
  `solve_nonlinear_koiter_newton()` (Module 21, general-purpose
  extensions Phase 8) adds a cubic-predictor alternative to arc-length
  with genuine adaptive step sizing, but implements only the published
  method's single-branch (`m=0`) case — the general method's real
  headline strength, tracing THROUGH a bifurcation by adding extra
  perturbation directions for closely-spaced/interacting buckling
  modes (`m≥1`), is NOT implemented here (see that function's own
  docstring "SCOPE" section, and `docs/general_purpose_extensions_
  roadmap.md` Section 9). `solve_nonlinear_static_koiter_newton()`
  (Module 22, Phase 9) shares the same `m=0`-only scope, and is a
  DIFFERENT function, not a drop-in replacement — it solves to one
  prescribed target load rather than tracing a path; see
  `docs/general_purpose_extensions_roadmap.md` Section 10 for why the
  two are not interchangeable via a thin wrapper. The `m≥1` case IS
  now implemented, as a SEPARATE new function,
  `solve_nonlinear_koiter_newton_generic()` (Module 24, Phase 10) —
  automatic near-critical-mode detection plus a second perturbation
  direction in the same bordered-matrix machinery — but still only for
  at most one extra direction at a time, with continuity-based (not
  stability-based) branch selection at a genuine bifurcation; see that
  function's own "SCOPE" docstring section and `docs/general_purpose_
  extensions_roadmap.md` Section 11.
- **Material nonlinearity now covers 1-D, plane-stress, and 3-D/
  continuum J2 plasticity (isotropic AND combined isotropic/kinematic
  hardening) plus compressible hyperelasticity, but no viscoelastic
  support, and still no combined plasticity+large-strain element.**
  `TrussPlastic2D` (1-D J2/von Mises), `Hex8PlasticJ2` (Module 19,
  general-purpose extensions Phase 6 — 3-D continuum J2/von Mises,
  closed-form radial return, per-Gauss-point state), and
  `Quad4PlasticJ2PlaneStress` (Wave 2 item 13 — plane-stress J2 via a
  local-Newton return map, `docs/consolidated_future_roadmap.md`) use
  linear ISOTROPIC hardening only — validated against an independent
  return-map replay through a load-unload-reverse history
  (`TrussPlastic2D`, see `validate_plasticity.py`) and against
  `PlasticMaterial1D`'s closed form under a genuinely uniaxial-stress
  path (`Hex8PlasticJ2`/`Quad4PlasticJ2PlaneStress`, see
  `tests/test_plasticity_j2.py`/`tests/test_plasticity_plane_stress.py`).
  **Kinematic hardening is now available** via `Hex8PlasticJ2Kinematic`
  (Wave 2 item 14): combined isotropic + Armstrong-Frederick nonlinear
  kinematic hardening (`PlasticMaterialJ2Kinematic`/
  `j2_radial_return_3d_kinematic()`), reproducing the Bauschinger
  effect (early re-yielding on load reversal) plain isotropic hardening
  cannot — a back-stress variable added to the same `state` dict
  pattern, exactly as anticipated here previously; plain `Hex8PlasticJ2`
  (isotropic-only) remains available unchanged. See
  `tests/test_plasticity_kinematic.py`.
  Hyperelasticity (`Tet4NeoHookean`, also Phase 6) is COMPRESSIBLE
  Neo-Hookean only — no Mooney-Rivlin/Ogden family, and no
  near-/fully-incompressible formulation (would need a mixed
  displacement-pressure element or an F-bar-style volumetric
  projection; the plain displacement-based element here will
  volumetrically lock as `nu -> 0.5`). No viscoelasticity/creep
  (time-/temperature-dependent flow under constant load). Geometric
  nonlinearity (`TrussTL2D`/`Tet4NeoHookean`) and material
  nonlinearity (`TrussPlastic2D`/`Hex8PlasticJ2`) also remain on
  SEPARATE elements — a single element combining large-strain
  kinematics with plastic return mapping (finite-strain
  elastoplasticity, e.g. multiplicative F=F_e*F_p) is future work, not
  something this package does yet.
- **Nonlinear dynamic (time-stepping) solve: implicit only.**
  `nonlinear_solver.solve_nonlinear_transient()` (Newmark-Newton,
  `docs/nonlinear_transient_dynamics_roadmap.md`) fills the IMPLICIT
  half of this gap — re-factors the effective tangent
  `K_T(d) + a0c*M + a1c*C` every Newton iteration of every step,
  agrees with `solve_transient_implicit()` to machine precision in the
  linear limit, agrees with `solve_nonlinear_static()` in the
  quasi-static limit, and conserves energy to <0.1% over a 5-period
  undamped free-vibration window (`tests/test_nonlinear_transient.py`).
  No EXPLICIT nonlinear scheme exists yet (`solve_transient_explicit`'s
  central-difference update needs no tangent at all for a nonlinear
  internal force, so this would be a smaller addition, deliberately
  deferred — see the roadmap doc's Section 3). (See this project's
  MFS-NLROM/cracked-beam work for a *different* approach to nonlinear
  structural dynamics — reduced-order modeling rather than full FE;
  `solve_nonlinear_transient()` is exactly what that ROM work now uses
  as its full-order dynamic ground truth.)
- **Contact/boundary nonlinearity: four contact elements, three
  enforcement mechanisms, static only, 2-D.** `GapContactPenalty`
  (validated against a closed-form equilibrium check and, via
  `solve_contact_lagrange_static()`, against an exact Lagrange-
  multiplier solution and its own penalty→Lagrange convergence as
  `k_p → ∞` — see `validate_contact.py`) is node-to-FIXED-FLAT-obstacle
  only, frictionless, stateless (contact status is a function of
  current position only), and composes for free with `TrussTL2D`
  (demonstrated in `validate_contact.py`) but has not been tested
  combined with `TrussPlastic2D`. `GapContactCurvedFriction` (Module 13,
  `validate_curved_contact.py`) adds Coulomb friction (stick/slip
  return mapping, reusing `TrussPlastic2D`'s `state` pattern) and an
  updating normal, but only against a single analytic CIRCLE.
  **`NodeToSegmentContact2D`/`NodeToSegmentContact2DFriction`
  (Wave 3 items 16-17, `docs/consolidated_future_roadmap.md`) now cover
  the genuinely-deformable-obstacle case** — node-to-segment contact
  between a slave node and a master segment of two ordinary (moving)
  mesh nodes, with `find_contact_pairs_2d()` doing candidate-pair
  discovery — but real gaps remain: no surface-to-surface (only
  node-to-segment) contact, no self-contact, `find_contact_pairs_2d()`
  is a documented exhaustive O(n·m) distance filter (not a spatial-hash/
  BVH broad phase — adequate for small-to-moderate contact zones only),
  no 3-D node-to-segment (2-D only, same as every other contact element
  in this package), no adhesive/cohesive contact, no surface
  roughness/tribology model, and no dynamic (impact/crash) contact
  analysis — the matching penalty formula exists but nothing drives it
  through an explicit time-stepping scheme. **Enforcement**: penalty
  (all four elements), exact bordered-KKT Lagrange multiplier
  (`solve_contact_lagrange_static()`, `GapContactPenalty`'s flat-wall
  convention only), and augmented Lagrangian (Wave 3 item 15,
  `solve_contact_augmented_lagrange_static()` — same flat-wall scope as
  the exact Lagrange driver, a different enforcement MECHANISM for the
  identical problem, not a generalization of it; at a matched `k_p` its
  relative error vs. the exact reaction is `4.7e-9` vs. plain penalty's
  `2.1e-2`, see `tests/test_augmented_lagrangian_contact.py`) are all
  available side by side. Both single-constraint drivers
  (`solve_contact_lagrange_static()`/`solve_contact_augmented_lagrange_
  static()`) are scoped to ONE contact constraint that engages at most
  once per run (a proper active-set solver that can activate AND
  release multiple simultaneous constraints — needed for something like
  gear teeth meshing/separating repeatedly, or multi-point crash-
  barrier contact — is a natural extension of the same augmented-KKT/
  Uzawa idea, not implemented here; `NodeToSegmentContact2D` itself,
  via `add_contact_element()`, DOES support any number of simultaneous
  contact pairs under plain penalty, since it's an ordinary registered
  element, not a dedicated single-constraint driver). `GapContactCurvedFriction`
  has not been tested against either single-constraint driver (both are
  written specifically for `GapContactPenalty`'s flat-wall `(n_hat, g0)`
  convention, not the curved element's `(center, R)` one).
- **`Shell4MITC` (Module 20, general-purpose extensions Phase 7) fixes
  shear locking only, not membrane locking, and is a flat-facet
  approximation.** Classic MITC4 (Dvorkin & Bathe 1984), as implemented
  here, only applies its assumed-strain treatment to TRANSVERSE SHEAR;
  membrane strains are plain displacement-based (literally
  `Quad4PlaneStress`'s own formulation), which is known in the shell
  literature to lock on curved geometry under distorted meshes — the
  documented fix (MITC4+, Ko/Lee/Bathe 2017, an assumed MEMBRANE strain
  field on top of the same MITC4 shear treatment) is out of scope for
  this phase. Each element also builds and uses exactly ONE local flat
  frame from its own 4 corners (`_local_frame_and_coords()`), so a
  warped (non-planar) quadrilateral is approximated by its best-fit
  flat facet, not exactly represented by a true curved-surface
  (degenerate-shell, per-node director vector) kinematics — fine for a
  shell mesh fine enough that individual elements are nearly flat (the
  standard flat-facet-shell assumption), which is what this element was
  validated against. No shell buckling (`geometric_stiffness()` is not
  implemented — shell buckling needs a genuinely different, membrane-
  stress-dependent geometric stiffness, not the single reference-axial-
  force convention `Beam2DEulerBernoulli`/`Beam3DEulerBernoulli`/
  `TrussTL2D` use) and no nonlinear (large-displacement/large-rotation)
  shell formulation — `Shell4MITC` is linear-static/dynamic only.
  (`Shell4MITCCorotational`, added later, IS the geometrically nonlinear
  large-rotation shell — see the module reference table above and this
  same section's own bullet on its known bending-membrane coupling gap.)
- **`Shell4MITCCorotational` cannot yet reproduce elastica-scale axial
  foreshortening from large bending rotation, and its own mixed-
  formulation fix attempt (Module 23, 2026-09-04) does not work.** The
  shipped element (deviation-from-mean von Karman coupling) is
  correctly-signed but recovers only ~0.05%-0.06% of the true
  foreshortening (`tests/test_shell_corotational_elastica.py`'s own
  canary test) — a deliberate, documented trade-off to preserve exact
  single-element rigid-tilt invariance. Seven documented attempts to
  close this gap for real (absolute rotation, saturated/regularized
  rotation, adaptive arc-length, and a mixed/Hellinger-Reissner
  reformulation with new general-purpose `iter_state` infrastructure)
  have all been tried and rejected — see `shells.py`'s "Design history"
  comment (seven dead ends, in full, with measured numbers) and this
  section's "Nonlinear extension point, part 2b" above for the most
  recent one. Two paths remain genuinely open, not attempted: converging
  the mixed formulation's internal unknown for real within each load
  step (not just one lagged correction), or giving it true joint-DOF
  status.
