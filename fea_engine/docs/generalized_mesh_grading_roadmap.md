
> **Update (2026-09-24):** Gmsh-backed geometry/visualization support
> (`geometry/gmsh_engine.py`, `visualization/gmsh_plot.py`) was removed
> from the package due to an unresolvable system-library dependency
> (`libGLU.so.1`), with no example or test outside the Gmsh-specific
> ones depending on it. This document is retained for historical/
> design-rationale context but describes/references a feature that no
> longer exists in the codebase. `build_mesh()`'s former "unstructured
> Gmsh path" branch now raises `MeshGradingError` explaining the
> removal instead.

# Generalized mesh grading -- research + design roadmap

Status: **Phases 1-4 implemented and validated** (2026-09-03, same day as
the design); **Fillet/Notch on the structured path (this section's own
originally-deferred item) implemented 2026-09-08, Wave 5 items 26/27 of
docs/consolidated_future_roadmap.md** -- see Section 6's updated note.
This document surveys how mesh grading is done in the FEA/CFD
meshing literature and in production meshers, audits what `fea_engine`
originally had (two independent, narrow implementations), and proposes (and
now implements) a single generalized grading capability that (a) works
across both of `fea_engine`'s existing mesh front ends, (b) works across a
hole placed anywhere in a rectangle rather than only at the origin of an
already-quarter-symmetric domain, and (c) automatically decides *when* and
*how much* to grade from a feature list, instead of requiring the caller to
hand-pick `grade_p`/`extend_grade`/`nr`/`ntheta` by trial and error. Sections
1-5 below are the original research + design (unchanged); **Section 6 is
the new as-built summary** -- what actually got implemented, what was
deliberately deferred, and where the honest gaps are relative to the
original Phase 2/3 wording.

This work was prompted by two things found in the same day's sessions: (1)
the wing-cantilever shell remesh (`geometric_nonlinear_shell_roadmap.md`,
Phase D), whose chordwise element size grades from 0.535mm to 1.25mm purely
as an *incidental* side effect of transfinite-mapping a tapered region at a
fixed division count -- nobody chose that grading, it just happened; and (2)
a concurrent session's `rectangle_with_hole_mesh_quarter_full()` in
`mesh.py`, which grades *intentionally* via explicit `grade_p`/`extend_grade`
parameters, and whose own docstring documents a hard constraint: this kind
of grading is only verified safe for `Shell4MITC`/`Shell4MITCCorotational`
when the mesh stays a structured, axis-aligned grid -- curved or unstructured
topology reintroduced a 40x-1500x stiffening bug (`_BEND_SIGN`, now fixed,
but the underlying geometric constraint the bug exposed is still real: shell
elements' local bending frames must not be mutually misaligned in ways the
element formulation doesn't account for). Those two facts together are the
motivation for this document: `fea_engine` has grading, but it is
one-geometry-specific, 2-D-only, and safe only for a subset of its own
element library. Generalizing it needs to respect that constraint, not
paper over it.

## 1. Research: how mesh grading is actually done

Four largely independent traditions converge on the same underlying idea --
element size should be a smooth, bounded function of position, small where
the geometry or the expected solution is "busy" and large where it isn't --
but they differ in how that function gets built.

**1a. A-priori grading laws (structured/mapped meshes).** When a mesh is
built by mapping a fixed parametric grid onto physical space (transfinite
interpolation, boundary layers, `fea_engine`'s own `rectangle_mesh_from_
partitions`-style approach), grading is a 1-D reparameterization of the
[0,1] parametric coordinate before the map is applied. The two standard
families are a **power law**, `s = t**p` (what `fea_engine` already uses:
`grade_p`/`extend_grade`), and a **geometric progression**, where each
element is a fixed ratio `alpha` larger than its neighbor (the convention
used for boundary-layer meshes, where a wall-normal ratio `alpha` and a
first-layer height `h(0)` fully determine the whole stack: layer *n*'s
thickness is proportional to `alpha**n`). Both are monotonic reparameterizations of a uniform partition and are
interchangeable in principle; a geometric-progression law is more common in
boundary-layer/CFD contexts because it has a direct physical parameter
(constant growth ratio) rather than an indirect exponent.
[A-priori Mesh Grading Techniques](https://www.emergentmind.com/topics/a-priori-mesh-grading)

**1b. Growth-ratio limits.** Independent of which grading law is used, the
literature and every major commercial mesher converge on capping the ratio
between adjacent element edge lengths, because a large jump degrades solution
accuracy at the transition (effectively a local under-resolution) even if
every individual element is well-shaped. The commonly cited target is a
growth ratio of **1.2**, with **up to ~1.5** treated as acceptable in less
sensitive regions; near true geometric singularities (reentrant corners,
crack tips) mesh size is instead required to scale with a *power* of
distance to the singularity to recover the optimal convergence rate that a
uniform mesh would lose there -- a stronger, theoretically-motivated version
of the same idea.
[Meshing Recommendations -- Flow360](https://docs.flexcompute.com/projects/flow360/en/release-25.2/knowledgeBase/PreProcessing/Meshing/Meshing.html),
[A-priori mesh grading for the numerical calculation of the head-related transfer functions](https://arxiv.org/pdf/1606.00278)

**1c. Background size fields (unstructured meshes).** Unstructured meshers
(Gmsh being `fea_engine`'s own dependency) don't reparameterize a grid --
they instead build a scalar target-size field `h(x)` over the whole domain
and hand it to the triangulation/recombination algorithm. Gmsh exposes this
as composable **Fields**: a `Distance` field measures proximity to chosen
points/curves/surfaces, a `Threshold` field turns that distance into a
size that is `h_min` near the feature, `h_max` far from it, and linearly (or
sigmoid-)interpolated in between over a `DistMin`/`DistMax` band, a
`MathEval` field allows an arbitrary analytic `h(x,y,z)`, and setting
`Mesh.CharacteristicLengthFromCurvature` grades size inversely with local
surface curvature so curved regions get more elements automatically. Several
fields combine via a `Min` field (take the tightest constraint anywhere).
This is the direct unstructured analogue of 1a/1b, and it is the piece
`fea_engine`'s own Gmsh front end (`geometry_engine/gmsh_engine.py`) does not
currently use at all -- every generator there sets a single global
`Mesh.CharacteristicLengthMin/Max` and stops.
[GMSH: Mesh Refinement Using Mesh Size Field](https://www.linkedin.com/pulse/gmsh-mesh-refinement-using-size-field-asmaa-hadane),
[Gmsh 4.15 reference manual, Fields](https://gmsh.info/doc/texinfo/gmsh.html)

**1d. Automatic size functions from geometry ("where is the geometry
complex?").** This is the research answer to the question in your message --
how does an algorithm decide grading is needed without a human picking
feature points by hand. The standard construction (Persson & Strang and
follow-on work) builds `h(x)` from three ingredients, then enforces a
**Lipschitz/grading-limit** post-process on top so the result never exceeds
the growth-ratio cap from 1b regardless of how sharply the raw ingredients
vary:
  - **Local feature size**, via the domain's medial axis: `f(x) = 2 *
    distance(x, medial_axis)`, i.e. twice the distance from a boundary point
    to the nearest *other* part of the boundary. This is what actually
    detects "the geometry is complex here" in a shape-agnostic way -- a
    narrow web between a hole and an edge, a thin fin, two nearby fillets,
    all shrink `f(x)` automatically, without anyone having to name those
    features. Curvature alone is provably insufficient in these narrow
    regions -- two boundaries can each be locally flat (zero curvature) and
    still be close enough together to need a fine mesh between them, which
    only a proximity/medial-axis measure catches.
  - **Curvature-based size**: bound size by (a constant times) the local
    radius of curvature, so a tightly curved boundary (small fillet, hole of
    small radius) gets enough elements around its circumference regardless
    of overall domain size.
  - **User/PDE-driven size**: an optional explicit field (e.g. finer where a
    stress concentration or a von Karman nonlinear region is expected).
  - **Gradient limiting**: given the pointwise minimum of the above, solve
    a fast marching / iterative correction so that `|h(x) - h(y)| <= g *
    |x - y|` everywhere (`g` the growth-ratio cap from 1b) -- this is what
    turns a possibly-jagged raw size estimate into a smoothly graded field
    with no sudden jumps, and is the generalization of 1a's single grading
    exponent to an arbitrary 2-D/3-D domain.
[Mesh size functions for implicit geometries and PDE-based gradient limiting (Persson)](https://persson.berkeley.edu/pub/persson05sizefunc.pdf),
[Automatic feature-preserving size field for 3D mesh generation](https://arxiv.org/pdf/2009.03984),
[Automatic Feature Recognition Using the Medial Axis for Structured Meshing](https://www.researchgate.net/publication/340539469_Automatic_Feature_Recognition_Using_the_Medial_Axis_for_Structured_Meshing_of_Automotive_Body_Panels)

The practical takeaway for `fea_engine`: sections 1a/1b are exactly what the
existing `graded_partition`/`rectangle_with_hole_mesh_quarter_full` already
do (correctly, just narrowly scoped); 1c is a real, currently-unused Gmsh
capability sitting right next to code that already calls Gmsh; 1d is the
missing piece that turns "the user manually picks `grade_p`" into "the
algorithm looks at the geometry and grades it," which is what you're asking
for.

## 2. Current state in `fea_engine` -- two front ends, two different gaps

`fea_engine` has had two independent mesh-generation front ends since the
CadQuery/Gmsh research phase (`geometry_meshing_alternatives_research.md`):
a zero-dependency structured/mapped front end (`mesh.py`) and a Gmsh-driven
unstructured front end (`geometry/gmsh_engine.py`). Both hand back the same
`Mesh(nodes, elements, dim)` contract, so `FESystem`/solvers downstream never
know or care which one produced a given mesh -- which is exactly why a
generalized grading capability can be added independently on each side
without touching anything downstream.

**`mesh.py` (structured/mapped): grading exists, but is one-geometry-specific.**
`graded_partition(start, end, n, grade, dense_at)` (a general 1-D power-law
partition, matches 1a directly) and `rectangle_mesh_from_partitions(x_coords,
y_coords)` (a structured Quad4 mesh from *explicit* non-uniform coordinate
arrays) are already fully general -- they'll grade any axis-aligned
rectangular 2-D region, not just the hole case. But the one function that
actually *uses* them for a real feature, `rectangle_with_hole_mesh_quarter_
full(a, b, R, nr, ntheta, grade_p, n_extend, extend_grade)`, is hard-coded to
exactly one geometry: a quarter-symmetric rectangle with a single circular
hole at the origin, radially graded from the hole outward. There is no
version for two holes, a fillet, a notch, an elongated slot, or a
3-D counterpart (`box_with_hole_mesh` still calls the plain, non-welded,
non-`a!=b`-safe `rectangle_with_hole_mesh_quarter`, so even the one existing
feature case hasn't been extended to solids yet).

**`geometry/gmsh_engine.py` (unstructured): no grading at all, anywhere.**
Every generator (`generate_2d_rectangle`, `generate_2d_plate_with_hole`,
`generate_3d_cylindrical_pipe`, `generate_from_step`, ...) sets Gmsh's
`Mesh.CharacteristicLengthMin` and `...Max` to the *same* value (or, in
`generate_2d_plate_with_hole`, two different-but-still-global constants
`lc_outer`/`lc_inner` with no actual spatial field driving the transition)
and calls it done -- none of Gmsh's `Distance`/`Threshold`/`MathEval` fields
or `CharacteristicLengthFromCurvature` (Section 1c) are wired up anywhere in
the file. This is the more surprising gap: `gmsh_engine.py` is exactly where
1d's "automatic, geometry-driven grading" belongs, since Gmsh already
implements the hard part (the field system, the gradient-limiting mesh-size
solver) -- `fea_engine` just isn't calling it.

**The constraint that any generalization must respect: element formulation,
not just mesh topology.** The `_BEND_SIGN`/reference-frame investigations
(this session and the concurrent one) established that `Shell4MITC` and
`Shell4MITCCorotational` -- any element carrying rotational bending DOFs
(`w, theta_x, theta_y` / `betax, betay`) with a local reference frame derived
from element geometry -- develop severe spurious stiffening when neighboring
elements' local frames are mutually misaligned, which happens on any curved
or unstructured mesh (Gmsh's recombine pass included) but not on a
structured axis-aligned grid, however non-uniformly it's spaced. Pure
translational-DOF elements (`Quad4` plane stress, `Tri3`, `Hex8`, `Tet4`/
`Tet10`, `Quad8`/`Hex20`) have no local bending frame and are not subject to
this at all -- grading is unconditionally safe for them regardless of mesh
curvature. **A generalized grading module therefore cannot just be "make the
mesh"; it has to know which element formulation the mesh is for, and refuse
(or fall back to the axis-aligned path) when that formulation is in the
restricted rotational-DOF set.** This is not a hypothetical edge case --
it's the exact bug that blocked the wing pipeline for most of today.

## 3. Proposed design

### 3a. A `grading.py` module, shared by both front ends

Rather than duplicating grading logic per shape (which is how `mesh.py`
ended up with a hole-only implementation), factor out the geometry-agnostic
pieces from Section 1 into one small module that both `mesh.py` and
`gmsh_engine.py` call into:

```
fea_engine/
  grading.py          # NEW -- geometry-agnostic sizing + growth-ratio logic
  mesh.py             # structured front end: feature list -> block decomposition -> grading.py -> weld
  geometry/
    gmsh_engine.py     # unstructured front end: feature list -> grading.py -> Gmsh Field API
```

`grading.py`'s job is exactly Section 1d, minus the actual triangulation
(which each front end already knows how to do): given a description of
"features" (see 3b) and a growth-ratio cap, produce either (i) a set of 1-D
partitions for the structured path, or (ii) a Gmsh field expression for the
unstructured path. It has no dependency on Gmsh and no dependency on any
specific element formulation -- it is pure sizing-function math, which is
also what makes it unit-testable without a mesher at all (check the returned
partition/field obeys the requested growth ratio analytically, the same way
`test_shell_corotational_reference_frame.py` checked the reference-frame fix
analytically before ever running a solve).

### 3b. A minimal, declarative feature description

The reason `rectangle_with_hole_mesh_quarter_full` had to be hand-written
per-shape is that it takes raw geometric parameters (`a, b, R, nr, ntheta`)
with no uniform notion of "a feature that needs local refinement." Replace
that with a small, composable feature list that both front ends can consume:

```python
features = [
    Hole(center=(0, 0), radius=0.02, n_ring=12),         # circular hole
    Fillet(edge_id=3, radius=0.005, n_ring=8),            # rounded corner
    Notch(path=[...], depth=0.01, n_ring=6),              # slot/notch
    EdgeBias(edge_id=0, h_first=0.001, growth=1.2),       # boundary-layer-style edge grading
]
```

Each feature contributes a local target size `h_min` at its own boundary
(from its own radius/geometry, per 1d's curvature term) and a footprint over
which that size applies; `grading.py` combines them via the same `Min`-field
idea Gmsh itself uses (Section 1c) -- take the tightest constraint anywhere
-- then blends out to a caller-supplied `h_far` at a caller-supplied
`growth_ratio` (default 1.2, matching the literature's common target from
1b; a value above ~1.5 should warn, not silently accept, since that is past
what's generally considered acceptable). This is a strict generalization of
what already exists: a single `Hole` feature with `n_ring=ntheta` and the
existing `grade_p` convention reproduces today's
`rectangle_with_hole_mesh_quarter_full` output exactly, so nothing existing
regresses.

### 3c. The "complexity -> grade automatically" dispatcher

This is the specific mechanism your message is asking about. Rather than the
caller deciding "this geometry is complex, let me call the graded mesher,"
the mesh-generation entry point always runs the same analysis and decides
for itself:

```python
def build_mesh(geometry, target_size, element_formulation, growth_ratio=1.2):
    features = geometry.detect_features()          # holes/fillets/notches/
                                                     # curvature extrema, from
                                                     # the geometry description
                                                     # (CAD boolean history if
                                                     # from generate_from_step,
                                                     # or explicit feature list
                                                     # if hand-built)
    if not features:
        return uniform_mesh(geometry, target_size)  # today's plain path,
                                                      # unchanged -- grading
                                                      # is skipped, not forced

    plan = grading.plan(features, h_far=target_size, growth_ratio=growth_ratio)
    if element_formulation in ROTATIONAL_DOF_ELEMENTS:
        if not plan.is_axis_aligned_structured():
            raise MeshGradingError(
                f"{element_formulation.__name__} has verified-safe grading "
                "only on structured axis-aligned topology (see Section 2's "
                "_BEND_SIGN note); this feature set requires curved/welded "
                "blocks. Pass structured_only=True features, or use a "
                "translational-DOF element, or override explicitly.")
        return mesh.build_structured_graded(geometry, plan)   # 3a's structured path
    return gmsh_engine.build_unstructured_graded(geometry, plan)  # 3a's Gmsh-field path
```

The "detect complexity" step doesn't need to be exotic: for the structured
path, "complex" literally means "the feature list is non-empty" (a plain
rectangle/box has none, so it degrades to exactly today's `rectangle_mesh`/
`box_mesh` with zero behavior change); for the Gmsh/CAD path (`generate_from_
step`), it means walking the imported B-rep for small-radius edges/faces
(circles, fillets below some fraction of the bounding-box size) and curved
surfaces, which is a direct, mechanical translation of Section 1d's
feature-size/curvature ingredients into Gmsh `Distance`/`Threshold`/
`CharacteristicLengthFromCurvature` field calls -- no new geometry math
needed, since OpenCASCADE (already `gmsh.model.occ`, per the CadQuery
research doc) exposes edge/face radii and curvature directly.

### 3d. Parameter glossary

| Parameter | Meaning | Default / guidance |
|---|---|---|
| `grade_p` / `extend_grade` | Power-law exponent for a structured 1-D partition, `s = t**p` | 2.0 (existing convention); higher = more aggressive clustering |
| `growth_ratio` | Max allowed ratio between adjacent element edge lengths | 1.2 (literature target); warn above ~1.5 |
| `h_min` | Target size at a feature boundary | derived from feature radius / `n_ring` |
| `h_far` | Target size far from any feature (today's uniform `mesh_size`) | caller-supplied, unchanged from today's API |
| `n_ring` / `nr`, `ntheta` | Element count around/outward from a feature | existing `mesh.py` convention, now per-feature instead of global |
| `n_extend` | Element count in a feature's far-field extension block | existing convention (defaults to `nr`) |
| `dense_at` | Which end of a 1-D partition is clustered | existing convention (`"start"`/`"end"`) |
| `structured_only` | Force the axis-aligned structured path even for translational-DOF elements | `False`; auto-`True` when `element_formulation` is rotational-DOF |

## 4. Validation plan

- **Analytic:** unit-test `grading.py`'s returned partitions/fields directly
  against the requested `growth_ratio` (max adjacent-size ratio <= cap, to
  numerical tolerance) -- no solver involved, mirrors how the reference-frame
  fix was checked analytically first.
- **Regression:** confirm a single-`Hole` feature list reproduces today's
  `rectangle_with_hole_mesh_quarter_full`'s node/element arrays exactly (or
  within a documented, intentional difference); rerun the existing hole-mesh
  and wing-mesh cases through the new dispatcher and confirm identical
  results to what's already validated.
- **Quality:** extend `Mesh.check_quality()` (currently detJ-positivity and
  min/max detJ ratio only) with a `check_grading()` companion that reports
  the actual realized neighbor-to-neighbor size ratio, so a mesh can be
  checked against the growth-ratio cap it claims to satisfy, not just against
  element non-inversion.
- **Safety gate:** a test that intentionally requests a curved/unstructured
  feature mesh for `Shell4MITCCorotational` and asserts `MeshGradingError` is
  raised, plus one confirming the same request succeeds for a translational-
  DOF element (`Quad4` plane stress) -- i.e. the gate itself is exercised in
  both directions, not just assumed to work.

## 5. Phased implementation plan

1. **`grading.py` core** -- `graded_partition`-style 1-D law (already exists,
   move/generalize), a `Feature` dataclass family (`Hole`, `Fillet`, `Notch`,
   `EdgeBias`), and a `plan()` function combining features via a `Min`-style
   footprint into either a partition set (structured) or a size-field
   description (unstructured), enforcing `growth_ratio` via a simple
   iterative Lipschitz correction (1d).
2. **Structured path generalization** -- rewrite `rectangle_with_hole_mesh_
   quarter_full` as a thin wrapper that calls the new generic multi-feature
   block-decomposition-and-weld path with a single `Hole` feature (regression
   target above), then add a genuinely new case (e.g. two holes, or a
   fillet) to prove the generalization isn't just a rename. Extend to 3-D via
   `extrude_mesh` composed with a graded 2-D cross-section (already works
   mechanically; needs a regression test that grading survives extrusion).
3. **Unstructured path (Gmsh fields)** -- wire `Distance`/`Threshold`/
   `CharacteristicLengthFromCurvature` into `gmsh_engine.py`'s generators,
   gated by the `ROTATIONAL_DOF_ELEMENTS` safety check from 3c so it's only
   reachable for translational-DOF element formulations until/unless a
   future fix extends verified-safe territory further.
4. **Dispatcher + safety gate** -- `build_mesh()` entry point from 3c,
   `MeshGradingError`, and the two-directional gate test from Section 4.
5. **Docs + validation sync** -- update this document's status line, run the
   full test suite, sync to canonical + dev workspace, as with every prior
   phase in this project.

Nothing here is a rewrite of the existing meshing code -- `mesh.py`'s
`graded_partition`/`rectangle_mesh_from_partitions` and `gmsh_engine.py`'s
existing generators are reused, not replaced; the new work is the
feature-list abstraction, the automatic-complexity dispatcher, and the
explicit element-formulation safety gate that today's code only enforces by
convention (a docstring warning) rather than by a checked precondition.

## 6. As-built summary (2026-09-03)

Phases 1-4 above are implemented and tested; Phase 5 is this section plus
the sync/test-suite work described at the end. Everything below reflects
what was ACTUALLY built, including two deliberate, documented deviations
from Sections 3-5's original wording -- called out here rather than left
for someone to discover by diffing the code against the plan.

**Phase 1 (`fea_engine/grading.py`, `tests/test_grading.py`, 47 tests).**
Built as designed: `graded_partition` moved here from `mesh.py` (which now
imports and re-exports it, so no existing caller broke);
`growth_ratio_of_partition`/`grade_for_growth_ratio` are the new
growth-ratio-cap solver (bisection on the power-law exponent); `Hole`/
`Fillet`/`EdgeBias` are real dataclasses, `Notch` is declared but raises
`MeshGradingError` on construction (honestly not implemented, not silently
accepted); `plan()` combines a feature list with growth-ratio validation
and warnings; `threshold_field_size`/`dist_max_for_growth_ratio` are the
Gmsh-Threshold-field-matching sizing law Phase 3 consumes directly.

**Phase 2 (`mesh.hole_in_rectangle_mesh`/`hole_in_rectangle_mesh_graded`,
`tests/test_mesh_grading_generalized.py`, 15 tests) -- ONE deviation from
the original wording.** The roadmap's Section 5 said to "rewrite
`rectangle_with_hole_mesh_quarter_full` as a thin wrapper" calling a new
generic path. That rewrite was deliberately NOT done: `rectangle_with_hole_
mesh_quarter_full` already IS the single-hole-at-the-origin special case,
it has an existing external caller (`NonLin-HyROM/case1_flat_plate_with_
hole.py`), and forcing it through a new code path for zero functional gain
carried real regression risk for that caller with no offsetting benefit.
Instead, `hole_in_rectangle_mesh()` is built ON TOP of it by composition
(4 independently-built quadrant blocks, each calling the existing function
with that quadrant's own a/b, welded together) -- the existing function
stays exactly as it was, single source of truth for the actual near-hole
mesh math is preserved, and the new function is what generalizes hole
PLACEMENT (anywhere in a non-symmetric rectangle, not just the origin of
an already-quarter-symmetric domain). Validated by mesh area convergence
to the analytic `Lx*Ly - pi*R^2` target (not array-equality against the
old function, which would be order-dependent and fragile), by a
no-inverted-elements/no-failed-weld check, and by extrusion to 3-D via the
existing `extrude_mesh()` (already generic, no changes needed).
`hole_in_rectangle_mesh_graded()` derives `grade_p`/`extend_grade` from a
`growth_ratio` via `grading.grade_for_growth_ratio()` instead of the
caller hand-picking them -- checked against the realized radial node
spacing, not just trusted.

**Fillet/Notch on the structured path: IMPLEMENTED (Wave 5 items 26/27,
docs/consolidated_future_roadmap.md, 2026-09-08)**, closing the gap this
section originally scoped out in favor of getting the off-center-hole
generalization right first. `grading.Notch` is now a real dataclass (a
rectangular slot, deliberately narrower than this document's original
"arbitrary path" wording -- see the class's own docstring for why);
`mesh.fillet_in_rectangle_mesh()`/`mesh.notch_in_rectangle_mesh()` are
the structured meshers that consume `Fillet`/`Notch`, built via the SAME
block-decomposition-and-weld pattern `hole_in_rectangle_mesh()` already
established (a new `_quarter_disk_sector_mesh()` collapsed-polar block
for the fillet's curved corner; three straight `rectangle_mesh_from_
partitions()` blocks for the notch). Both validated across all 4
corners/edges (`tests/test_mesh_fillet_notch.py`, 18 tests) -- see the
roadmap doc's own item 26/27 rows for the full validation summary. The
fillet's corner block is explicitly NOT claimed verified-safe for
rotational-DOF elements (the same curved/non-axis-aligned-topology
caveat this section's own Section 2 already documents for the hole
mesh) -- extending that verification remains items 28/29's job, still
blocked on Wave 4's own deferred findings.

**Phase 3 (`geometry/gmsh_engine.py`'s new
`generate_2d_plate_with_hole_graded`, `tests/test_gmsh_field_grading.py`,
3 tests).** Wires a Gmsh `Distance` field (measuring distance to the
hole's arcs) into a `Threshold` field (SizeMin/SizeMax/DistMin/DistMax,
with DistMax derived from `grading.dist_max_for_growth_ratio` -- the SAME
growth-ratio law the structured side uses, so both front ends grade "as
aggressively as growth_ratio=1.2 allows" from one shared definition) plus
`Mesh.CharacteristicLengthFromCurvature`, and disables Gmsh's own
point-size extension so only the field controls size. Validated against a
real Gmsh run (this sandbox's `libGLU.so.1` gap was worked around the same
way as earlier sessions -- `apt-get download`+`dpkg-deb -x` extraction,
`LD_LIBRARY_PATH` at test time): near-hole elements average under 60% of
far-field element size, confirming the field is actually taking effect,
not just present in the API calls with no observable result. All 40
pre-existing Gmsh-dependent tests (`test_geometry_engine.py`,
`test_gmsh_node_order.py`, `test_quadratic_extraction.py`,
`test_step_import.py`, `test_mixed_elements.py`,
`test_tet10_geometric_nonlinear.py`) still pass unchanged.

**Phase 4 (`fea_engine/build_mesh.py`, `tests/test_build_mesh_dispatcher.py`,
9 tests) -- ONE deviation from the original pseudocode.** The roadmap's
Section 3c pseudocode has the dispatcher check `plan.is_axis_aligned_
structured()` and RAISE `MeshGradingError` when a rotational-DOF element
meets a non-structured feature list. The as-built `build_mesh()` instead
makes this a routing decision: for the one geometry currently understood
(`RectangleWithHole`, at most one `Hole` feature), a verified-safe
structured path already exists, so a rotational-DOF element is ALWAYS
routed there -- it can never reach the unstructured Gmsh path through this
function, which is the actual safety property that matters, achieved more
usefully (the mesh still gets built) than a hard reject would. The
`MeshGradingError` reject path still exists and is tested, for the case
where no verified-safe representation exists at all (e.g. a rotational-DOF
element against a 2-hole feature list, which nothing implements yet).
`ROTATIONAL_DOF_ELEMENTS = (Quad4MindlinPlate, Shell4MITC,
Shell4MITCCorotational)` is an explicit, commented tuple, deliberately not
inferred from `dofs_per_node` (beam elements also carry rotational DOFs
but have no 2-D local-frame-misalignment mechanism at all, so a blanket
DOF-count rule would mis-classify them). The gate is tested in BOTH
directions per the roadmap's own requirement: a rotational-DOF request
provably never imports the Gmsh module at all (checked by poisoning
`sys.modules` and confirming the rotational path still succeeds while the
translational path then fails, proving the routing decision happens
before any Gmsh-availability check), and a translational-DOF request on
the identical geometry provably DOES reach the Gmsh path (Tri3
connectivity vs. the structured path's Quad4).

`Mesh.check_grading()` was added alongside `check_quality()` (2-D only for
now -- 3-D face-adjacency needs a per-topology face table not yet
written): the realized max neighbor-to-neighbor element-size ratio, either
as a raw number or a pass/fail against a cap, complementing
`check_quality()`'s detJ-positivity check with the OTHER thing grading can
get wrong (individually well-shaped elements that still jump abruptly in
size at a boundary).

**Public API.** `fea_engine/__init__.py` now re-exports `grading` and
`build_mesh` as top-level modules, plus `Hole`/`Fillet`/`EdgeBias`/`Notch`/
`GradingPlan`/`MeshGradingError`/the growth-ratio functions/
`hole_in_rectangle_mesh`/`hole_in_rectangle_mesh_graded`/
`RectangleWithHole`/`ROTATIONAL_DOF_ELEMENTS` as flat names -- EXCEPT
`build_mesh` the function, which is deliberately NOT flattened (it would
collide with and silently shadow the `build_mesh` SUBMODULE name,
depending on import order); call it as `fea_engine.build_mesh.build_mesh(...)`
or `from fea_engine.build_mesh import build_mesh`.

**Test totals.** 74 new tests across the four phases (47 + 15 + 3 + 9 --
Phase 3's count is smaller because it validates one new Gmsh method plus
the shared sizing-law function, not a from-scratch geometry family), all
passing alongside the full pre-existing suite (147 non-Gmsh tests + 40
Gmsh-dependent tests, all green). One pre-existing, unrelated flake
(`test_shell_corotational_elastica.py::test_known_limitation_large_
rotation_regime`, a numerical-noise-sensitivity issue at a 1e-8 tolerance
against a documented zero-foreshortening limitation, ~7.7e-7 observed on
one run) was seen once and confirmed non-reproducible on rerun -- not
caused by, and not fixed by, this phase's changes, noted here rather than
silently ignored.

## References

- [A-priori Mesh Grading Techniques -- Emergent Mind](https://www.emergentmind.com/topics/a-priori-mesh-grading)
- [A-priori mesh grading for the numerical calculation of the head-related transfer functions](https://arxiv.org/pdf/1606.00278)
- [Meshing Recommendations -- Flow360 CFD Solver docs](https://docs.flexcompute.com/projects/flow360/en/release-25.2/knowledgeBase/PreProcessing/Meshing/Meshing.html)
- [GMSH: Mesh Refinement Using Mesh Size Field](https://www.linkedin.com/pulse/gmsh-mesh-refinement-using-size-field-asmaa-hadane)
- [Gmsh 4.15.2 Reference Manual (Fields)](https://gmsh.info/doc/texinfo/gmsh.html)
- [Mesh size functions for implicit geometries and PDE-based gradient limiting (Persson & Strang)](https://persson.berkeley.edu/pub/persson05sizefunc.pdf)
- [Automatic feature-preserving size field for 3D mesh generation](https://arxiv.org/pdf/2009.03984)
- [Automatic Feature Recognition Using the Medial Axis for Structured Meshing of Automotive Body Panels](https://www.researchgate.net/publication/340539469_Automatic_Feature_Recognition_Using_the_Medial_Axis_for_Structured_Meshing_of_Automotive_Body_Panels)
