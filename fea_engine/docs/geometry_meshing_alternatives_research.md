> **Update (2026-09-24):** Gmsh-backed geometry/visualization support
> (`geometry/gmsh_engine.py`, `visualization/gmsh_plot.py`) was removed
> from the package due to an unresolvable system-library dependency
> (`libGLU.so.1`), with no example or test outside the Gmsh-specific
> ones depending on it. This document is retained for historical/
> design-rationale context but describes/references a feature that no
> longer exists in the codebase.

# Alternative geometry/meshing front ends for `fea_engine` -- research

Status: **research only, no implementation yet.** This document surveys freely
available, Python-scriptable (and in most cases GUI-capable) geometry/meshing
packages that could sit alongside `fea_engine`'s existing `geometry.py`
(structured, zero-dependency) and `geometry_engine.py` (Gmsh-driven,
unstructured) front ends, and proposes concrete integration paths for each.
No code has been written against this document -- it exists to inform a
decision on which path (if any) to turn into an actual implementation phase.

**Direction chosen (Section 7): CadQuery (geometry authoring) + Gmsh
(meshing, already a dependency) -- no new meshing tool needed.** Sections 1-5
are the original survey; Section 6 checks that decision against a specific
target capability set (moderate-complexity geometry, higher-order elements,
problem-appropriate element-type selection) against `fea_engine`'s actual
element inventory, and Section 7 lays out the resulting implementation plan.
**Item 1 (`generate_from_step()`), item 2 (quadratic-element mesh
extraction), and item 3 (`mesh_algorithm` dial) are now DONE and synced
to canonical** -- see Section 7 for validation summaries (item 3's own
summary added 2026-09-08, Wave 5 item 25 of docs/consolidated_future_
roadmap.md). Item 4 (optional `Tri6PlaneStress` element formulation)
was ALSO completed, separately, as Wave 0 item 7 of that same roadmap
-- and Wave 5 item 30 then wired it into this module's own Gmsh
extraction table (see Section 7 item 2's own updated note on what was
"deliberately not extended"). Item 2's
riskiest prerequisite -- whether Gmsh's native 2nd-order node order
actually matches each element class's own -- was checked directly
(not assumed) for all three quadratic classes: `Hex20Solid3D` and
`Tet10Solid3D` each needed a real fix (`GMSH_NODE_ORDER`, found + fixed
2026-08-30 via the wing-cantilever example in the sibling
`Multi_Fidelity_NL_Structural_ROM` project), while `Quad8PlaneStress`
came back clean (identity permutation, checked the same way before
being wired up, not assumed from the other two's result). All three
permutations are now applied automatically inside `gmsh_engine.py`'s
`_extract_mesh()`.

## 1. Why this is a narrow integration problem, not a rewrite

Every mesh generator in `fea_engine`, regardless of front end, hands back one
of exactly two containers (`mesh.py`):

- `Mesh(nodes, elements, dim)` -- `nodes`: `(n_nodes, dim)` float array;
  `elements`: `(n_elements, nodes_per_element)` int connectivity array.
- `MultiBlockMesh(nodes, blocks, dim)` -- same `nodes`, plus one connectivity
  array per element topology (Module 14's mixed-element-type support).

`FESystem`, `assemble_stiffness`/`assemble_internal_force`, `fix_dofs`,
`check_quality`, and every solver downstream only ever read these two arrays
-- they have no idea whether the mesh came from `rectangle_mesh()`, Gmsh, or
anywhere else. **This means any external geometry/meshing tool is a valid
front end the moment its output can be converted into a plain
`(nodes, elements)` pair.** The entire research question below is really
"how much glue code does each candidate need to reach that contract," not
"can it work at all" -- they all can.

## 2. Current state and the specific gaps

- `geometry.py`: structured/mapped meshes (line, rectangle, box, a
  transfinite-mapped hole) -- zero dependencies, but geometry is hand-derived
  analytically. Fine for benchmark shapes, unusable for an arbitrary real part.
- `geometry_engine.py` (`gmsh_engine.py`): Gmsh-driven unstructured meshes,
  but built entirely on Gmsh's **built-in `.geo` kernel**
  (`gmsh.model.geo.addPoint/addLine/addCircleArc/extrude`) -- a simple sketch
  kernel with no booleans, fillets, chamfers, or CAD-file import. Every shape
  is defined by hand-written point-by-point Python code; there is no GUI
  authoring path anywhere in the current pipeline.
- Element support is **linear only**: `_extract_mesh()`'s
  `_GMSH_ELEMENT_NODE_COUNTS = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8}` covers line2/
  tri3/quad4/tet4/hex8 exclusively. This is a real, currently-unused gap:
  `elements/solids.py` already implements `Hex20`/`Tet10`/quadratic element
  formulations, but nothing in `geometry_engine.py` can produce a matching
  quadratic mesh -- Gmsh itself supports this (2nd-order element types), the
  extraction table simply doesn't list them yet.

Three independent gaps, addressed separately below: **(a)** no true CAD
operations/GUI authoring, **(b)** no CAD-file (STEP/IGES) import path, **(c)**
no quadratic/curved element output.

## 3. The key mechanism: Gmsh's own OpenCASCADE kernel closes gaps (a) and (b) almost for free

Gmsh -- already a dependency, already the backend of `geometry_engine.py` --
ships a **second** geometry kernel alongside `.geo`: `gmsh.model.occ`, a full
binding to OpenCASCADE Technology (OCCT), the same B-rep CAD kernel underneath
FreeCAD, Salome, CadQuery, and build123d. Two consequences:

1. `gmsh.model.occ` supports real CAD primitives and booleans
   (`addBox`, `addCylinder`, `fuse`, `cut`, `fillet`, `chamfer`) directly in
   Python -- closes gap (a) without leaving Gmsh at all.
2. `gmsh.model.occ.importShapes(filename)` reads a **STEP, IGES, or BREP
   file** straight into the OCC kernel, after which `synchronize()` +
   `mesh.generate()` proceeds exactly as `geometry_engine.py` already does --
   closes gap (b) for *any* external tool that can export STEP, which is
   every candidate discussed below (STEP is the universal neutral CAD
   exchange format).

```python
gmsh.initialize()
gmsh.model.add("part")
gmsh.model.occ.importShapes("part.step")
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
# ... same node/element extraction _extract_mesh() already does
```

This is the single fact that makes the rest of this document mostly a
**"which GUI do you want to author geometry in"** question rather than a
"which mesh-format parser do we need to write" question: the hand-off point
into `fea_engine` is the same regardless of which tool produced the STEP
file, and `_extract_mesh()`'s existing node-dedup/re-indexing logic needs no
changes to consume it.

## 4. Candidates

| Package | What it is | GUI? | License | Integration path into `fea_engine` |
|---|---|---|---|---|
| **Gmsh (OCC kernel + native app)** | Already a dependency. Its own desktop GUI (separate from the Python API) supports interactive point/curve/surface sketching AND OCC boolean/fillet operations, live meshing preview. | Yes (native Gmsh app) | GPL-2.0-or-later, with a linking exception (permits linking from software under a different license, which is why `fea_engine` calling into it as a runtime dependency doesn't itself impose GPL) | **Zero new dependency.** Either script `gmsh.model.occ` directly (closes gap a), or author interactively in the Gmsh GUI and export `.geo`/`.step`/`.msh`, then feed into the existing `_extract_mesh()` pipeline (already built) or `gmsh.open()`. |
| **CadQuery** (+ **CQ-editor**, actively maintained) | Code-first parametric CAD, real OCCT B-rep kernel (robust booleans/fillets/chamfers), fluent Python API. CQ-editor gives a live-preview GUI over the same scripts (auto-reload on save). | Yes, via CQ-editor (optional; scripting alone needs none) | Apache-2.0 | Build geometry in CadQuery, `.export("part.step")`, hand off via `gmsh.model.occ.importShapes()` above -- **one new method** on `UnifiedGeometryEngine`, no CadQuery-specific parsing needed. Best fit if the project wants to stay in its own "geometry defined by reviewable, testable code" style (matches `geometry.py`'s own stated design principle) while gaining real CAD ops `.geo` lacks. |
| **build123d** | Newer sibling of CadQuery, same OCCT/OCP core, more idiomatic Python (context managers instead of fluent chaining), can even run in-browser (WASM). | Via Jupyter/VS Code viewers, not a dedicated desktop app | Apache-2.0 | Identical hand-off to CadQuery (STEP export -> `importShapes`). Choose over CadQuery mainly on Python-style preference; both wrap the same kernel. |
| **FreeCAD** | Full parametric CAD workbench (Sketcher, Part Design, Assembly), the closest thing here to a conventional commercial CAD GUI. Ships its own FEM workbench wrapping Gmsh/Netgen. Scriptable via an embedded Python console/macros. | Yes, full desktop GUI | LGPL-2.1 | Two options: **(i)** export STEP/BREP from any FreeCAD document and hand off via the same `importShapes()` mechanism (recommended -- reuses all existing extraction code, no new format support needed); **(ii)** use FreeCAD's own FEM workbench mesh export (UNV/MED/VTK/Abaqus INP/Nastran BDF) and read it with `meshio` (below) instead of re-meshing in Gmsh. Best fit if genuinely complex real-world parts (bosses, fillets, multi-body assemblies) need interactive sketching rather than code. |
| **Netgen / NGSolve** | A second, independent unstructured mesher (tet-focused), with its own `OCCGeometry` class for OCCT-based geometry description in Python and a `mesh.Curve(order)` call that produces genuinely curved, high-order meshes matched to the CAD surface -- not just quadratic connectivity on straight edges. | Yes (native Netgen GUI) | LGPL-2.1 | Directly addresses gap (c): use `OCCGeometry(step_file)` + `GenerateMesh()` + `Curve(2)` for a quadratic/curved mesh, then export and read via `meshio` (Netgen `.vol` is supported) into `Mesh`/`MultiBlockMesh`. Worth adding as a **second** engine specifically for the quadratic-element case Gmsh's current extraction table doesn't cover yet -- see Section 5 for the cheaper alternative (extend Gmsh's own table instead). |
| **Salome** | A full, heavyweight open-source CAE platform: parametric CAD module (OCCT-based) + multi-algorithm mesher (can invoke Netgen/Gmsh internally as plugins) + solver coupling, all exposed through both a GUI and an extensive Python API (`smeshpy`/`geompy`). | Yes, full desktop GUI | LGPL-2.1-or-later | Heaviest install of the group. Either export STEP (same hand-off as above) or use Salome's own `smeshpy` Python interface to drive meshing directly and export MED/UNV, read via `meshio`. Best fit only if the project wants **one** unified GUI application covering CAD authoring, meshing, *and* result visualization, and is willing to accept a much larger install/learning-curve cost for that. |
| **meshio** (glue, not a geometry tool) | A pure-Python translator between ~30 mesh file formats (Abaqus INP, Nastran BDF, UNV, MED, VTK/VTU, Netgen VOL, Gmsh MSH, ...). No geometry or meshing capability of its own. | No | MIT | The practical adapter for any candidate above whose *native* mesh export isn't STEP/Gmsh-MSH (e.g. FreeCAD's FEM-workbench UNV export, Salome's MED export, Netgen's VOL format): `meshio.read(file).points` / `.cells` map directly onto `Mesh.nodes` / `Mesh.elements` with one small wrapper function, no per-format parsing needed in `fea_engine` itself. |

All seven are free and open source (no paid tier gates any capability
described above), satisfying the "freely available" requirement directly.

## 5. Recommendation by scenario

- **Want a GUI, minimal new dependency, willing to stay Gmsh-only:** use
  Gmsh's own desktop app (OCC kernel, booleans/fillets included) to author
  and mesh interactively, export `.step`/`.msh`; `geometry_engine.py` already
  has the extraction pipeline, it just needs one new
  `generate_from_step(filepath, dim, mesh_size)` wrapper around
  `importShapes()`. Lowest-effort option; closes gaps (a) and (b) together.
- **Want to keep geometry defined in reviewable/testable Python (this
  project's existing convention), but need real CAD ops `.geo` lacks
  (fillets, chamfers, robust booleans, STEP import of externally-supplied
  parts):** CadQuery (or build123d), STEP export, same `importShapes()`
  hand-off. Also closes gaps (a)/(b), no GUI required at all (CQ-editor
  optional for visual iteration).
- **Need to author a genuinely complex real-world part or assembly
  interactively (not something a few dozen lines of CadQuery would encode
  cleanly):** FreeCAD, STEP export, same hand-off. Full commercial-CAD-like
  GUI experience at LGPL/free cost.
- **Need actual quadratic/curved elements to finally exercise the existing
  `Hex20`/`Tet10` formulations from a Gmsh-produced mesh:** the cheapest fix
  is extending `_GMSH_ELEMENT_NODE_COUNTS`/`_GMSH_TYPE_NAMES` in
  `geometry_engine.py` to Gmsh's own 2nd-order element type codes (Gmsh
  already supports generating them, `_extract_mesh()` just doesn't list them)
  -- no new dependency at all. Netgen/NGSolve's `mesh.Curve(order)` is the
  heavier alternative, worth it only if truly curved (not just quadratic-node)
  geometry fidelity matters, e.g. for a thin curved shell where straight-edge
  quadratic elements would still misrepresent the surface.
- **Want one unified GUI covering CAD + meshing + visualization instead of
  stitching pieces together, and don't mind the larger install:** Salome.

## 6. Target capability check: element library vs. a standard DOF/element-type matrix

Before deciding what to add, it's worth checking what's already there. Cross-
checking `fea_engine`'s actual element inventory (`elements/__init__.py`'s
`ELEMENT_REGISTRY`) against a standard "element selection by dimensionality"
matrix (truss/frame/plane-stress/plate/shell/solid, with DOFs-per-node and
primary solved outputs) shows the element FORMULATION library is already
substantially built -- including higher order, not just linear, in two of the
six rows:

| Row | Standard element type | DOFs/node | Status in `fea_engine` |
|---|---|---|---|
| 1D Truss/Bar | axial, 1-2 DOF `(u, v)` | Axial force | `TrussTL2D`, `TrussPlastic2D` -- 2 DOF/node. Linear (2-node) only, which is standard for a bar element; order doesn't meaningfully apply here the way it does to continuum elements. |
| 1D Frame/Beam | 3 or 6 DOF | Moment/shear/torsion | `Beam2DEulerBernoulli`/`Beam2DCorotational` (3 DOF/node, 2D) and `Beam3DEulerBernoulli` (6 DOF/node, 3D) -- both dimensionalities covered. |
| 2D Plane Stress/Strain | 2 DOF `(u, v)` | In-plane stresses | `Tri3PlaneStress`/`Quad4PlaneStress` (linear) **and `Quad8PlaneStress` (quadratic, already implemented and validated)**. Gap: no quadratic triangle (`Tri6`). |
| 2D Plate Bending | 3 DOF `(w, theta_x, theta_y)` | Out-of-plane moment | `Quad4MindlinPlate` -- linear (4-node) only. No quadratic plate element yet. |
| 2D Shell | 6 DOF, full membrane+bending | Full membrane + bending profile | `Shell4MITC` -- linear (4-node MITC) only. No higher-order shell yet. |
| 3D Volume Brick/Tet | 3 DOF `(u, v, w)` | Full 3D stress tensor, von Mises | `Tet4Solid3D`/`Hex8Solid3D` (linear) **and `Tet10Solid3D`/`Hex20Solid3D` (quadratic, already implemented and validated)** -- the most complete row already. |

Two conclusions follow directly:

- **"Higher-order elements" is largely already solved at the formulation
  level for solids and plane-stress** (`Hex20`/`Tet10`/`Quad8` exist and are
  validated against analytic benchmarks). The bottleneck identified here was
  upstream, at the meshing layer: Section 2's gap (c) -- `geometry_engine.py`'s
  Gmsh extraction originally only ever requested *linear* connectivity, so
  these already-built quadratic formulations had no way to receive a
  matching mesh from anything but a hand-built node array. **DONE as of
  Section 7 item 2** -- the fix was the mesh generator, not a new element
  class, but with one real caveat surfaced along the way: "extend the
  extraction table" was not purely mechanical. Gmsh's own native node order
  for a 2nd-order element does not always match the corresponding
  fea_engine class's assumed order (confirmed false for both
  `Hex20Solid3D` and `Tet10Solid3D`, confirmed TRUE/clean for
  `Quad8PlaneStress`), so each type needed its ordering checked against a
  real Gmsh-built reference element, not assumed from Gmsh's documentation
  table -- `GMSH_NODE_ORDER` on all three classes (identity for `Quad8`) is
  that now-done work, applied automatically by `_extract_mesh()`.**
- **The remaining formulation gaps are narrower than "no higher order
  elements":** specifically `Tri6` (quadratic triangle) and a higher-order
  plate/shell (`Quad8`-plate or an 8-node shell) -- real, but scoped, additions
  if full parity across every row is wanted, not a blocking prerequisite for
  the meshing work.

**On problem-appropriate element-type selection (e.g. preferring hex over
tri/tet where bending dominates):** this is a meshing-*algorithm* choice, not
a tool choice, and it's worth being direct about its real difficulty. Fully
automatic, robust all-hex meshing of *arbitrary* 3-D geometry remains a
genuinely hard, still-unsolved problem industry-wide -- even paid tools
typically fall back to manual block decomposition for it. The stated target
here (moderate complexity, not real-world-arbitrary geometry) is exactly the
regime where it IS tractable, though: `geometry.py`'s existing structured path
(`box_mesh`, `extrude_mesh`) already produces pure hex for anything
block-like or extrudable, and Gmsh's own transfinite/structured meshing
extends that same capability to `geometry_engine.py`'s CAD-driven geometry
wherever the topology supports it (falling back to unstructured tet/tri only
where it doesn't). That control simply isn't exposed as an option in
`geometry_engine.py` yet -- see item 3 below.

## 7. Recommended direction and implementation plan

**Tool choice: CadQuery (geometry authoring) + Gmsh (meshing, already a
dependency) -- no new meshing tool adopted.** Against the "light, GUI +
script, easy to integrate" requirement: CadQuery is pure Python + OCCT
bindings (no multi-gigabyte install the way FreeCAD or Salome are), script-
first (matching `geometry.py`'s own "geometry as reviewable, testable code"
convention), with **CQ-editor** available as an optional live-preview GUI
that isn't required for the scripting path to work. Gmsh needs no new
adoption at all -- it already supports both 2nd-order elements and
transfinite/structured meshing; those capabilities just aren't wired into
`geometry_engine.py` yet. This keeps the integration surface to *one*
external geometry dependency plus small, targeted extensions of the meshing
code already in the package, rather than introducing a second full meshing
library (Netgen) or a heavyweight platform (Salome) for capability Gmsh
already has.

Four small, independent additions follow from Section 6's findings -- three
in `geometry_engine.py`, one optional new element formulation:

1. **DONE.** `generate_from_step(filepath, dim, mesh_size=None,
   target_unit="M")` on `UnifiedGeometryEngine`, via
   `gmsh.model.occ.importShapes()` (Section 3) -- the CadQuery (or any
   STEP-exporting tool) hand-off. Validated against a STEP file actually
   exported from CadQuery (a plate with a genuine boolean-cut hole AND
   fillets -- neither reachable through this module's own `.geo`-kernel
   generators) AND a separately Gmsh-authored STEP file, confirming the
   hand-off is genuinely tool-agnostic, not accidentally Gmsh-specific
   (`tests/test_step_import.py`, 15 tests). Two real findings surfaced by
   this validation, not assumed correct from the API alone: (a) STEP files
   are near-universally declared in millimeters regardless of authoring
   tool -- confirmed by inspecting BOTH files' own headers
   (`SI_UNIT(.MILLI.,.METRE.)` in both), so `target_unit` defaults to
   `"M"` rather than Gmsh's own no-conversion default, avoiding a silent
   1000x-too-large mesh; (b) `Geometry.OCCTargetUnit` turned out to be
   sticky, asymmetrically, across repeated calls in one process --
   switching between two explicit units (`"M"` <-> `"MM"`) is reliable,
   but switching from an explicit unit back to Gmsh's own empty-string
   "no conversion" sentinel is NOT (confirmed directly: it silently kept
   applying the earlier conversion), so `generate_from_step()` requires an
   explicit non-empty `target_unit` on every call rather than exposing
   that unreliable path. A third fix (wrapping the Gmsh session in
   `try`/`finally`) was needed because, unlike every other generator in
   this class, this one ingests externally authored files that can
   genuinely fail partway through, and an exception between
   `_start()`/`_finish()` was found to leave Gmsh "stuck" initialized,
   corrupting the next call in the same process. Synced to
   `computation-suite/fea_engine/` -- see that package's own README
   (`geometry_engine.py` module reference) for the full validation
   summary. **A fourth issue found 2026-08-30, NOT yet fixed:**
   `generate_from_step()` becomes unreliable (hangs/pathologically slow
   at Gmsh's own mesh-generation step) once enough prior calls to it
   have accumulated anywhere in the same process -- see the README's own
   "KNOWN ISSUE" note in this method's section for the direct
   reproduction. Worth investigating alongside a real fix for the
   `OCCTargetUnit` stickiness above, since both point at OCCT-level
   state surviving `gmsh.finalize()`/`initialize()` cycles more broadly
   than just that one option.
2. **DONE.** Extended `_GMSH_ELEMENT_NODE_COUNTS`/`_GMSH_TYPE_NAMES` to
   the three 2nd-order Gmsh element types this package has an element
   formulation for -- tet10 (type 11), quad8 (type 16, serendipity),
   hex20 (type 17, serendipity) -- directly activating the already-
   implemented `Tet10Solid3D`/`Hex20Solid3D`/`Quad8PlaneStress`
   formulations from a Gmsh-produced mesh for the first time.
   **Deliberately NOT extended** (at the time) to hex27 (12)/quad9
   (10)/tri6 (9): no `Hex27Solid3D`/`Quad9PlaneStress`/`Tri6PlaneStress`
   class existed to consume them (a scoped exclusion, matching item 4
   below, not an oversight -- see `gmsh_engine.py`'s own top-of-file
   comment). **UPDATE 2026-09-08 (Wave 5 item 30, docs/consolidated_
   future_roadmap.md): tri6 (9) is now wired in**, now that item 4's
   `Tri6PlaneStress` was built (Wave 0 item 7 of that same roadmap) --
   its `GMSH_NODE_ORDER` was checked directly against a real Gmsh-built
   reference element (identity, like `Quad8PlaneStress`'s own) before
   wiring it up, and end-to-end validated via the same exact Saint-
   Venant pure-bending benchmark `Quad8`'s own extraction test uses
   (`tests/test_quadratic_extraction.py`'s `TestTri6ExactPureBendingViaGmsh`).
   hex27 (12)/quad9 (10) remain excluded -- still no consuming element
   class.

   The node-order risk flagged in the previous version of this
   document (whether Gmsh's native 2nd-order node order actually
   matches each element class's assumed order) was checked directly
   for all three types before wiring any of them up, not assumed:
   `Hex20Solid3D`/`Tet10Solid3D` each needed their existing
   `GMSH_NODE_ORDER` fix (see each class's "GMSH COMPATIBILITY
   WARNING" docstring note, `tests/test_gmsh_node_order.py`);
   `Quad8PlaneStress` was checked the same way (a real Gmsh-built
   reference element, `tests/test_quadratic_extraction.py`'s helper)
   and came back clean -- Gmsh's native 8-node quadrangle already
   matches this class's own order, so `Quad8PlaneStress.GMSH_NODE_ORDER`
   is the identity. `_extract_mesh()` now applies whichever permutation
   applies per element type automatically, sourced directly from each
   class's own attribute (not re-typed in `gmsh_engine.py`, so the two
   can't drift apart).

   `element_order=2` (default remains 1, unchanged behavior) is now a
   parameter on `generate_from_step()`, `generate_3d_cylindrical_pipe()`
   (both produce Tet10), and the new `generate_3d_box()` (Hex20, the
   first hex-topology generator in this module -- transfinite +
   recombine, genuinely useful beyond this validation too);
   `generate_2d_rectangle()` gained a `recombine` flag (needed since it
   never produced quad topology before) alongside `element_order=2` for
   Quad8.

   Validated at two rigor tiers (`tests/test_quadratic_extraction.py`):
   Quad8 gets a machine-precision EXACT check, reproducing
   `tests/test_higher_order_elements.py`'s proven Saint-Venant pure-
   bending comparison (Quad4 locked at ratio≈0.667, Quad8 exact to
   `<1e-6`) through the real Gmsh extraction pipeline instead of a
   hand-typed connectivity array; Tet10/Hex20 get a relative,
   coarse-mesh comparison against the same elementary-beam-theory
   cantilever benchmarks `test_geometry_engine.py` already established
   (Tet10 ratio≈1.01 vs. Tet4's ≈0.87; Hex20 ratio≈0.99 vs. Hex8's
   badly-locked ≈0.70), at identical element counts, each backed by a
   strict global-equilibrium check. The two-tier split is deliberate,
   not a shortcut: an exact 3-D pure-bending match would need face-
   consistent nodal loads accounting for each element's quadratic edge
   weighting across the extra (z) dimension, real additional work for
   uncertain incremental benefit over the relative comparison already
   used.
3. **DONE (2026-09-08, Wave 5 item 25, docs/consolidated_future_
   roadmap.md).** A `mesh_algorithm="unstructured"\|"structured"`
   option, exposing Gmsh's transfinite/structured meshing so a caller
   can request hex-preferential meshing on block-like regions of
   *arbitrary* geometry, with the tet/tri fallback for regions where
   the topology doesn't support it -- documented honestly, including a
   REAL limitation found during validation (not assumed away): on a
   3-D volume where only part of the topology is block-like, Gmsh's
   own automatic detection can need PYRAMID transition elements this
   package has no formulation for, so `mesh_algorithm="structured"`
   correctly raises a clear error there rather than silently falling
   back or corrupting the mesh (see `_apply_mesh_algorithm()`'s own
   "REAL RISK" docstring note in `gmsh_engine.py`, and `tests/
   test_mesh_algorithm_dial.py`). Implemented via a single native Gmsh
   API call (`setTransfiniteAutomatic()`), not hand-rolled per-shape
   corner detection -- `generate_3d_box()`'s own fixed hex-block
   generator, item 2's prefiguring case, was deliberately left
   unchanged rather than rewritten onto this new dial (see that
   method's own updated docstring for why).
4. **DONE, separately (Wave 0 item 7, docs/consolidated_future_
   roadmap.md).** `Tri6PlaneStress`, a quadratic-triangle element
   formulation, for full row-3 parity with `Quad8` -- then wired into
   this module's own Gmsh extraction table as Wave 5 item 30 (see item
   2's own updated note above).

Suggested order was 1 -> 2 -> 3; all of 1-3 are now DONE, and item 4 is
done too (see its own row). Every phase followed the
project's established implement -> test -> sync to canonical -> update
docs -> verify-diff-empty workflow.

## Sources

- [FreeCAD FEM Mesh documentation](https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/FEM_Mesh.md)
- [FreeCAD FEM Workbench documentation](https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/FEM_Workbench.md)
- [FreeCAD gmshtools.py source](https://github.com/FreeCAD/FreeCAD/blob/main/src/Mod/Fem/femmesh/gmshtools.py)
- [From design to mesh generation using FreeCAD and GMSH](https://project.inria.fr/softrobot/documentation/from-design-to-mesh-generation-using-freecad-and-gmsh/)
- [SfePy: Preprocessing with FreeCAD/OpenSCAD + Gmsh](https://sfepy.org/doc-devel/preprocessing.html)
- [build123d vs CadQuery discussion (OCP.wasm)](https://github.com/CadQuery/cadquery/discussions/1876)
- [build123d documentation](https://build123d.readthedocs.io/en/latest/tips.html)
- [OpenSCAD vs CadQuery vs Build123d comparison](https://grandpacad.com/en/blog/openscad-vs-cadquery-vs-build123d)
- [build123d project page, Open CASCADE Technology](https://dev.opencascade.org/project/build123d)
- [CQ-editor on PyPI](https://pypi.org/project/CQ-editor/)
- [CQ-editor usage wiki](https://github.com/CadQuery/CQ-editor/wiki/Usage)
- [Using the Gmsh Python API to generate complex meshes](https://jsdokken.com/src/tutorial_gmsh.html)
- [Gmsh reference manual](https://gmsh.info/doc/texinfo/gmsh.html)
- [Gmsh tutorial t20.py (STEP import via OCC kernel)](https://github.com/live-clones/gmsh/blob/master/tutorials/python/t20.py)
- [pyGIMLi CAD-to-mesh tutorial](https://www.pygimli.org/_examples_auto/1_meshing/plot_cad_tutorial.html)
- [NGSolve: Open Cascade Technology Geometry tutorial](https://docu.ngsolve.org/latest/i-tutorials/unit-4.4-occ/occ.html)
- [SALOME Platform](https://www.salome-platform.org/)
- [SALOME Mesh Python interface documentation](https://docs.salome-platform.org/latest/gui/SMESH/smeshpy_interface.html)
- [SALOME project page, Open CASCADE Technology](https://dev.opencascade.org/project/salome)
- [meshio on PyPI](https://pypi.org/project/meshio/)
- [meshio GitHub README](https://github.com/nschloe/meshio/blob/main/README.md)
