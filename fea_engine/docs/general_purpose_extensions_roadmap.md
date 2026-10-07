> **Update (2026-09-24):** Gmsh-backed geometry/visualization support
> (`geometry/gmsh_engine.py`, `visualization/gmsh_plot.py`) was removed
> from the package due to an unresolvable system-library dependency
> (`libGLU.so.1`), with no example or test outside the Gmsh-specific
> ones depending on it. This document is retained for historical/
> design-rationale context but describes/references a feature that no
> longer exists in the codebase.

# General-purpose extensions roadmap for `fea_engine`

Status: **Phases 1, 2, 3, 4, 5, 6, and 7 (Sections 3, 7, 2, 6, 5, 4,
and 1) are implemented and validated.** Phase 1: `Quad8PlaneStress`,
`Hex20Solid3D`, `Tet10Solid3D` (`elements/solids.py`), plus
`tet_quadrature_4pt()` (`elements/base.py`). Phase 2: `FESystem(...,
sparse=True)` (`solver.py`'s `_zeros_matrix()`/`_as_solve_matrix()` +
`solve_static()`/`solve_modal()`), additive/opt-in, scoped to the
static/modal paths (not yet the nonlinear or transient-dynamics
paths). Phase 3: `Beam3DEulerBernoulli` (`elements/beams3d.py`),
`Section3D`/`beam3d_rigidities()`/`beam3d_mass_props()`
(`material.py`) — a closed-form 6-dof/node 3-D frame element built
from `Beam2DEulerBernoulli`'s existing bending formula reused for both
principal axes. Phase 4: `geometric_stiffness()` extension point on
`Element` (`elements/base.py`), implemented for
`Beam2DEulerBernoulli`, `Beam3DEulerBernoulli`, and `TrussTL2D`
(`elements/beams.py`/`beams3d.py`/`trusses.py`), plus
`FESystem.assemble_geometric_stiffness()`/`solve_linear_buckling()`
(`solver.py`) — validated against the closed-form Euler column
`P_cr = pi^2*EI/(K*L)^2` for three boundary conditions with genuine
mesh-refinement convergence. Tested in
`tests/test_higher_order_elements.py`, `tests/test_sparse_assembly.py`,
`tests/test_beam3d.py`, and `tests/test_linear_buckling.py`, worked
examples in `examples/higher_order_elements_demo.py`,
`examples/sparse_vs_dense_cantilever.py`,
`examples/beam3d_space_frame_demo.py`, and
`examples/euler_column_buckling_demo.py`. Phase 5:
`solve_nonlinear_arc_length()` (`nonlinear_solver.py`) -- Crisfield
cylindrical arc-length continuation, validated on the same von Mises
truss/closed-form `P(delta)` test_nonlinear.py already uses for
`solve_nonlinear_static()`/`solve_nonlinear_displacement_control()`,
including a genuine regression/contrast check that load control
really does fail past the limit point. Tested in
`tests/test_higher_order_elements.py`, `tests/test_sparse_assembly.py`,
`tests/test_beam3d.py`, `tests/test_linear_buckling.py`, and
`tests/test_arc_length.py`, worked examples also include
`examples/arc_length_snap_through_demo.py`, README updated (each
Phase 1-5 item also has a companion `_plot.py` example script for
visual verification, e.g. `examples/arc_length_snap_through_plot.py`).
Phase 6: `Hex8PlasticJ2`/`Tet4NeoHookean` (`elements/nonlinear_
solids.py`), `PlasticMaterialJ2`/`NeoHookeanMaterial`/
`j2_radial_return_3d()`/`neo_hookean_pk2_stress()` (`material.py`) --
closed-form radial-return J2 plasticity with a hand-derived consistent
tangent (cross-validated against the continuum limit and finite
difference), and compressible Neo-Hookean hyperelasticity (closed-form
2nd Piola-Kirchhoff stress, finite-difference tangent, same established
precedent as `GapContactCurvedFriction`). Both subclass their
linear-elastic parents (`Hex8Solid3D`/`Tet4Solid3D`). Validated in
`tests/test_plasticity_j2.py` (uniaxial stress matches
`PlasticMaterial1D`'s closed form exactly) and `tests/test_hyperelastic.py`
(small-strain limit matches `D_solid3d`; full nonlinear solve satisfies
external-work = stored-strain-energy to 3.7e-7 relative), worked
examples in `examples/plasticity_j2_demo.py`,
`examples/hyperelastic_neo_hookean_demo.py`, README updated, and both
have companion `_plot.py` example scripts
(`examples/plasticity_j2_plot.py`, `examples/hyperelastic_neo_hookean_
plot.py`) for visual verification, following the same pattern as
Phases 1-5.
Phase 7: `Shell4MITC` (`elements/shells.py`), `D_shell()`/
`shell_rho_matrix()` (`material.py`) -- a general 4-node shell
(membrane + bending + MITC4 assumed-strain transverse shear, 6
dof/node), reusing `Quad4PlaneStress`'s membrane `B_matrix()` and
`Quad4MindlinPlate`'s bending `Bb` in a per-element flat local frame,
with a genuine Dvorkin-Bathe MITC4 tying-point shear treatment (not
selective reduced integration -- see the module docstring for the
exact formula/citation). Validated in `tests/test_shell.py`: single-
element rigid-body modes; a frame-objectivity test that caught a real
axis-mislabeling bug (Quad4MindlinPlate's `betax` is physically a
rotation about the Y-axis, not X -- invisible on a flat/axis-aligned
mesh, wrong on any curved or tilted one); flat-plate-limit convergence
to `Quad4MindlinPlate` (0.03% at a fine mesh); a CURVED multi-element
rigid-body-mode check; and a curved cylindrical-arc cantilever
validated by mesh-refinement convergence against an independent
Hex8Solid3D reference (Hex8Solid3D is documented elsewhere in this
package as locking in bending, so convergence rather than exact
coarse-mesh agreement is the honest check). Worked examples in
`examples/shell_demo.py`/`examples/shell_plot.py`, README updated.
Every Phase 1-7 general-purpose-extensions item is now implemented and
validated. Phase 8 and 9 (added later, driven by a real external need
rather than this document's own original seven-item survey):
`solve_nonlinear_koiter_newton()` (Phase 8, Module 21) -- single-branch
Koiter-Newton predictor/corrector PATH CONTINUATION, generalizing Phase
5's arc-length driver with a CUBIC (Koiter asymptotic) predictor
instead of a linear tangent one, corrected via a chord/modified-Newton
iteration (its own frozen-tangent bordered-matrix scheme, not a reuse
of Phase 5's Crisfield corrector -- an earlier draft tried that reuse,
then replaced it after re-reading the target paper's own equations
directly). Validated on the SAME von Mises truss/closed-form benchmark
Phase 5 uses (`tests/test_koiter_newton.py`), including a measured
~11x reduction in outer steps needed to trace the same arc of that
closed-form path (2 steps vs. 22, delta_L=0.01 in both). Then
`solve_nonlinear_static_koiter_newton()` (Phase 9, Module 22) -- a
SEPARATE function for solving to a single prescribed target load rather
than tracing a path, written after the Phase-8 driver was tried (and
failed -- overshot a target load 14x, then crashed with a floating-
point overflow) as a ROM-training-data `fom_solver`; aims its cubic
predictor directly at the remaining gap to the target instead of
growing freely, validated on both a truss benchmark
(`tests/test_static_koiter_newton.py`) and directly on the real
`NonLin-HyROM` Case 1 model (~15-18x faster than plain Newton per
solve, single-expansion convergence). See Sections 9-10 for the full
derivation, scope, and citations of both. Then
`solve_nonlinear_koiter_newton_generic()` (Phase 10, Module 24) -- a
THIRD, brand-new function (Phases 8-9's drivers unchanged) that lifts
the `m=0`-only scope both of those disclose, by adding automatic
near-critical-mode detection and a second perturbation direction to
the same bordered-matrix machinery, falling back to Phase 8's own exact
formulas whenever no critical mode is found. Validated against a
hand-derivable synthetic 2-mode pitchfork-bifurcation benchmark whose
reduced cubic branch equation is shown to reproduce the true
closed-form bifurcated amplitude exactly (`tests/test_koiter_newton_
generic.py`). See Section 11 for the full derivation, the correctness
review of Phases 8-9 that motivated it, and the fea_engine-vs-rom_engine
placement reasoning.
This document is the
Phase-4-style research + roadmap pass for the seven general-purpose
gaps identified when reviewing what's implemented (`ELEMENT_REGISTRY`,
`CONSTITUTIVE_REGISTRY`, `solver.py`, `nonlinear_solver.py`) against
what a general-purpose FE package needs, BEFORE moving on to
problem-specific work (fluid/FSI coupling, modal identification) that
this project's other research threads flagged as deliberately deferred
until a real use case drives them. See
`computation-suite/hpc_translation_roadmap.html` for the companion
performance-scaling roadmap, and `rom_engine/docs/phase4_error_bounds_
greedy_roadmap.md` for the document this one follows in structure and
rigor (research each topic, map it onto the EXISTING architecture's
extension points, phase the build, cite sources, validate against real
models, no big-bang rewrite).

## 0. How this maps onto the existing architecture (read this first)

Every item below is designed to slot into extension points that
already exist and are already documented in the code:

- **New element type** &rarr; subclass `elements.base.Element`, implement
  `shape_and_derivs()` (+ `B_matrix()` for the generic Gauss-loop
  `stiffness()`, or override `stiffness()`/`mass()` directly the way
  `Quad4MindlinPlate` and `Beam2DEulerBernoulli` already do), add one
  line to `ELEMENT_REGISTRY` (`elements/__init__.py`). Nonlinear
  elements additionally override `internal_force()`/
  `tangent_stiffness()` (see `TrussTL2D`'s Total-Lagrangian pattern).
- **New constitutive law** &rarr; write a `D_xxx(material, ...)` function
  in `material.py`, add one line to `CONSTITUTIVE_REGISTRY`. Nothing
  else changes — elements just take whatever `D`/`mat` their
  `stiffness()`/`internal_force()` expects as a plain argument.
- **New solve type** &rarr; add a `solve_xxx()` method to `FESystem`
  (`solver.py`) or a new driver function in `nonlinear_solver.py`,
  reusing the already-assembled `K`/`M`/`C`/`F` and the existing
  `free_dofs`/`fixed_dofs` bookkeeping — this is exactly how
  `solve_harmonic()`, `solve_frequency_sweep()`, and
  `solve_random_vibration()` were added without touching `mesh.py` or
  `elements/`.

None of the seven items below need `FESystem`'s assembly LOOP
(`solver.py`'s `_blocks` iteration) to change — they're new element
formulations, new constitutive laws, and new solve-time drivers, which
is exactly the kind of "additive" extension the package's registry
architecture was built for. The one item that touches the assembly
loop itself is Section 7 (sparse matrices), and even that is a
storage-format change inside the same loop structure, not a new loop.

## 1. Shell elements

**What's missing**: `Quad4MindlinPlate` handles pure bending (plate
theory: transverse deflection + two rotations, no in-plane/membrane
stiffness). A shell additionally carries membrane (in-plane
stretching) stiffness coupled to bending — needed for curved or
folded thin-walled structures (a cylindrical tank, a car body panel,
an aircraft skin) where plate theory alone can't represent the
geometry.

**Recommended formulation**: MITC4 (Mixed Interpolation of Tensorial
Components, 4-node) — Dvorkin & Bathe's 4-node general shell element,
using standard displacement interpolation for membrane/bending strains
and an assumed-strain (collocation) interpolation for transverse shear
to avoid the same locking `Quad4MindlinPlate`'s selective reduced
integration already exists to fix. This is the most widely used,
best-validated low-order shell element in the literature, and it's a
natural generalization of machinery this package already has:

- Membrane part: literally `Quad4PlaneStress`'s `B_matrix()`/`D`
  (already implemented), evaluated in the shell's local in-plane
  frame.
- Bending + transverse shear part: `Quad4MindlinPlate`'s existing
  `D_mindlin_plate()` bending/shear split and selective-reduced-
  integration `stiffness()` pattern, reused as-is for the
  bending/shear block.
- New work: the assumed-NATURAL-strain interpolation for transverse
  shear at the MITC4's four tying points (this is what distinguishes
  MITC4 from a naive membrane+plate superposition, and what actually
  fixes shear locking rather than just relieving it via reduced
  integration), and the local-to-global rotation that orients each
  element's membrane+bending frame to the actual 3-D shell surface
  (needed since, unlike the flat `Quad4MindlinPlate`, shell elements
  in a curved mesh don't share one global in-plane frame).
- `dofs_per_node = 6` (3 translations + 3 rotations, vs. the plate's
  3) — the drilling (in-plane rotation) DOF is the classical extra
  complication of shell elements; the standard fix is a small
  artificial drilling stiffness (Allman-type), not a real physical
  stiffness, added purely to avoid a singular tangent when adjacent
  elements are coplanar.

**Architecture fit**: new `elements/shells.py`, `Shell4MITC` class,
`dofs_per_node=6`, `translational_dof_mask=[T,T,T,F,F,F]` (reusing
`lumped_mass()`'s existing HRZ scaling, which already handles the
translation/rotation split correctly for the plate). Add
`"shell4_mitc": Shell4MITC` to `ELEMENT_REGISTRY`. New `D_shell()` (or
reuse `D_plane_stress()` + `D_mindlin_plate()` composed) in
`material.py`.

**Validation plan**: (1) flat-plate limit — a `Shell4MITC` mesh with
zero curvature should reproduce `Quad4MindlinPlate` bending results to
near machine precision (a direct, checkable regression against an
existing validated element); (2) a curved benchmark with a known
closed-form or heavily-published numerical answer (the classic
pinched/Scordelis-Lo cylindrical shell benchmarks are the standard
choice in the shell literature) validated against fea_engine's own
independent full 3-D solid mesh (`Hex8Solid3D`/`Tet4Solid3D`) of the
same geometry at high refinement, mirroring this package's existing
"validate the reduced/simplified model against an independent
full-fidelity computation" convention.

## 2. 3D beam/frame elements

**What's missing**: `Beam2DEulerBernoulli`/`Beam2DCorotational` are
planar (bending in one plane only, no torsion, no out-of-plane
bending) — correct for 2-D frames, insufficient for a general 3-D
frame/truss structure (a building frame, a space truss, a piping
system) where members bend about two axes and twist.

**Recommended formulation**: a linear 3-D Euler-Bernoulli frame
element (12 DOF: 2 translations + 1 rotation missing per 2-D node
becomes 3 translations + 3 rotations per 3-D node, so 6 DOF/node x 2
nodes), built from FOUR uncoupled 1-D stiffness blocks — axial
(truss-like), torsional (`GJ/L`, St. Venant torsion), and TWO
independent Hermite bending blocks (one per principal bending axis,
each a direct reuse of `Beam2DEulerBernoulli`'s existing closed-form
Hermite-cubic stiffness) — assembled in the element's LOCAL frame,
then rotated to global via a standard 3-D direction-cosine
transformation matrix (needs an explicit "up"/reference-vector
convention to fix the twist-about-axis ambiguity a single axis vector
alone doesn't resolve). For the geometrically nonlinear case, extend
`Beam2DCorotational`'s existing corotational approach (element frame
rotates with the element, strains measured relative to that rotating
frame) to 3-D using an orthogonal local triad tracked via
incremental/spin-tensor rotation update rather than a single rotation
angle (Crisfield's 3-D corotational beam formulation is the standard
reference for this).

**Architecture fit**: new `elements/beams3d.py`,
`Beam3DEulerBernoulli` (`dofs_per_node=6`, `n_nodes=2`), reusing
`Beam2DEulerBernoulli`'s bending-block math internally (literally call
it twice, once per bending axis, then assemble into the 12x12 local
matrix) rather than re-deriving it. `Section` (`material.py`) needs
extending with `J` (torsional constant) and `Iy`/`Iz` (two bending
second moments) alongside the existing `A`/`I`. `Beam3DCorotational`
follows the same "override `internal_force()`/`tangent_stiffness()`"
pattern `Beam2DCorotational` already established.

**Validation plan**: (1) each of the four uncoupled blocks (axial,
torsion, two bending) checked independently against the exact 1-D
textbook solution for a cantilever under that specific loading; (2) a
3-D frame with a known closed-form deflection (e.g. an L-shaped or
grid frame under a point load, standard structural-analysis textbook
problems) checked against the assembled element; (3) the flat/planar
limit (all loading and geometry in one plane) checked against
`Beam2DEulerBernoulli`'s existing validated results, the same
"reduce to an already-validated special case" check used for shells.

## 3. Higher-order elements (Hex20, Tet10, Quad8/9)

**What's missing**: every solid/planar element in the registry is
linear (`Quad4PlaneStress`, `Tri3PlaneStress`, `Hex8Solid3D`,
`Tet4Solid3D`) — constant strain within each element, so curved
boundaries and stress concentrations need heavy mesh refinement to
resolve accurately. Quadratic elements capture curved geometry with
mid-side nodes and represent linearly-varying strain fields with far
fewer elements.

**Recommended formulation**: serendipity Hex20 (8 corner + 12
mid-edge nodes, no interior/face nodes — the standard choice; full
27-node Lagrange Hex27 is more accurate for some cases but rarely
worth the extra DOF in practice) and Tet10 (4 corner + 6 mid-edge
nodes, the standard quadratic tetrahedron every unstructured 3-D
mesher — including Gmsh, which `geometry/gmsh_engine.py` already
wraps — produces natively as its "order 2" element). Quad8
(serendipity) for the 2-D analog. All three are a DIRECT extension of
the EXISTING architecture with zero new machinery: implement
`shape_and_derivs()` for the quadratic shape functions (closed-form,
well-tabulated), and the ALREADY-GENERIC `Element.stiffness()`/
`Element.mass()` Gauss loop (`elements/base.py`) works completely
unmodified — these elements need a HIGHER Gauss order
(`gauss_order=3` typically, vs. the linear elements' 2, since the
integrand `B^T D B` is now higher-degree) but no new integration code.

**Architecture fit**: this is the lowest-effort item in this whole
roadmap given the existing base-class design — add `Hex20Solid3D`,
`Tet10Solid3D`, `Quad8PlaneStress` to `elements/solids.py`, each just
a `shape_and_derivs()` implementation plus `gauss_order` set
appropriately; `B_matrix()`, `stiffness()`, `mass()` are inherited
unchanged from `Element`. Add three lines to `ELEMENT_REGISTRY`. Since
`geometry/gmsh_engine.py` already produces Gmsh meshes, it also needs
a small update to request/read second-order element blocks from Gmsh
(a Gmsh API option, not new geometry logic) when a higher-order
element type is requested.

**Validation plan**: (1) patch test (a single element or small mesh
under a linear displacement field should reproduce that field
EXACTLY, to machine precision — the textbook minimum-correctness check
for any new element); (2) convergence-rate comparison against the
existing linear elements on the SAME curved-boundary or stress-
concentration benchmark (e.g. a plate with a hole under tension, which
`mesh.py` already has `rectangle_with_hole_mesh_quarter()` for) —
quadratic elements should converge to the known stress-concentration
factor with far fewer DOF, a direct, checkable claim rather than an
assumed one.

## 4. 2D/3D J2 plasticity + hyperelasticity

**What's missing**: `PlasticMaterial1D` is uniaxial-only (correct for
`TrussPlastic2D`, where a bar truly can only carry axial stress, so
the general 3-D yield surface collapses to a 1-D closed form — see
that class's docstring). Continuum elements (`Quad4PlaneStress`,
`Hex8Solid3D`, ...) have NO plasticity or large-strain hyperelastic
material option at all — every continuum element today is
linear-elastic only, regardless of load level.

**J2 (von Mises) plasticity — recommended formulation**: the classical
radial-return mapping algorithm (Wilkins; Krieg & Key; formalized in
the now-standard form by Simo & Taylor 1985). For 3-D and plane
strain, the J2 yield surface is a hypersphere in deviatoric-stress
space, so the discrete return map at each Gauss point is literally a
RADIAL rescaling of the elastic trial stress back onto the yield
surface — closed-form, no local Newton iteration needed (the same
"closed-form because the geometry collapses nicely" property that
makes `TrussPlastic2D`'s 1-D case simple, just at one dimension
higher). Plane STRESS is the one case that needs a local Newton
iteration (the return path is constrained to stay in the 2-D
plane-stress subspace, so it's no longer a simple radial scaling) —
Simo & Taylor's 1986 plane-stress paper is the standard reference for
that specific complication.

**Hyperelasticity — recommended formulation**: compressible or
nearly-incompressible Neo-Hookean, via an isochoric/volumetric strain-
energy split (`W = W_vol(J) + W_iso(barF)`) — the standard, numerically
robust starting point for large-strain rubber-like or biological
tissue behavior, and simple enough to implement without a mixed
formulation for the moderately-incompressible case; the fully/nearly-
incompressible limit (`nu -> 0.5`) needs a mixed displacement-pressure
(u/p) formulation to avoid volumetric locking, which is real added
complexity best deferred until a specific use case needs
near-incompressibility (soft tissue, rubber) rather than built
speculatively.

**Architecture fit**: this is the item that most needs
`fesystem.state`/`init_state()`/`commit_all_states()` — the
PATH-DEPENDENT state machinery `TrussPlastic2D` already established
(accumulated plastic strain per Gauss point, not just per element,
since a continuum element has multiple integration points each with
their own potentially-different plastic state, unlike a 1-D truss).
New `material.py` entries: `PlasticMaterialJ2` (E, nu, sigma_y, H) and
`NeoHookeanMaterial` (E or mu/kappa, nu). Continuum elements override
`internal_force()`/`tangent_stiffness()` exactly like `TrussTL2D`
does, looping over Gauss points and calling the return-mapping
function (plasticity) or the hyperelastic stress/tangent function
(Neo-Hookean) at each one, reading/writing per-Gauss-point state via
the same `init_state()`/`commit_all_states()` hooks.

**Validation plan**: (1) uniaxial stress path (a single element loaded
to reproduce pure uniaxial tension) should match `PlasticMaterial1D`'s
already-validated 1-D result exactly — the same "collapse to an
existing validated special case" check used throughout this roadmap;
(2) a monotonic-then-unload load history should show the correct
permanent set / no reverse yielding (elastic unload slope = E) —
directly checkable, not just "looks plastic"; (3) Neo-Hookean: the
small-strain limit should recover the linear-elastic `D_solid3d()`
tangent stiffness (every hyperelastic model must reduce to linear
elasticity as strain -> 0 — a strong, checkable consistency
requirement).

## 5. Arc-length / Riks solver

**What's missing**: `solve_nonlinear_static()`'s own docstring already
flags this — load-controlled Newton-Raphson fails at a limit point
(where the load-displacement curve goes vertical or turns over, i.e.
snap-through/snap-back), and its documented workaround
(`solve_nonlinear_displacement_control()`) only works when the
CONTROLLED dof's response is itself monotonic, which isn't true for
every snap-through problem (e.g. snap-BACK, where even the loaded
point's displacement reverses).

**Recommended formulation**: Crisfield's cylindrical arc-length method
("A fast incremental/iterative solution procedure that handles
'snap-through'", 1981) — instead of prescribing EITHER the load
(load control) OR one displacement component (displacement control),
constrain the COMBINED norm of the displacement increment and a
scaled load-factor increment to a fixed "arc length" radius, and solve
for both the displacement increment AND the load factor increment
together at each Newton iteration (a quadratic equation in the load-
factor increment, from substituting the linearized displacement update
into the arc-length constraint — this quadratic, and which of its two
roots to pick, is the one genuinely new piece of math relative to the
existing Newton-Raphson loop). This traces the FULL equilibrium path
through limit points and snap-back without needing to know in advance
where they are.

**Architecture fit**: a new `solve_nonlinear_arc_length()` function in
`nonlinear_solver.py`, sitting alongside
`solve_nonlinear_static()`/`solve_nonlinear_displacement_control()` as
a third strategy — same `fesystem.assemble_internal_force()`/
`assemble_tangent_stiffness()` calls, same `free_dofs`/
`commit_all_states()` pattern, different increment-control logic
inside the Newton loop. No changes needed to `FESystem` or any element
— this is purely a solver-loop addition, the same kind of addition
`solve_nonlinear_displacement_control()` already was.

**Validation plan**: a benchmark problem with a KNOWN snap-through (a
shallow arch or a von Mises truss — the classic two-bar snap-through
truss is small enough to have a closed-form load-displacement curve)
where `solve_nonlinear_static()` is confirmed to fail exactly as its
own docstring predicts (a genuine regression/contrast test, not just
a new capability tested in isolation), and the arc-length solver is
confirmed to trace the FULL path including the reversal, checked
against the closed-form curve.

## 6. Linear buckling eigenvalue solver

**What's missing**: `solve_modal()` already solves ONE generalized
eigenproblem pattern (`K phi = omega^2 M phi`, vibration). Linear
buckling is the SAME generalized-eigenproblem PATTERN with a different
second matrix: `(K + lambda * K_sigma) phi = 0`, i.e.
`K phi = -lambda * K_sigma phi`, where `K_sigma` (the geometric /
"stress stiffness" matrix) depends on the stress state from a
reference linear static solve, not on mass. The lowest `lambda` is the
critical buckling load FACTOR (multiply by the reference load to get
the actual critical load); `phi` is the buckling mode shape.

**Recommended formulation**: standard linear (eigenvalue) buckling —
(1) run `solve_static()` at a reference load to get the stress state,
(2) build each element's geometric stiffness `k_sigma` from that
stress state (for a continuum element, `k_sigma` comes from the
SECOND-order/nonlinear part of the strain-displacement relation
evaluated at the reference stress — literally the same nonlinear
strain terms `Beam2DCorotational`/`TrussTL2D` already compute for
their Total-Lagrangian tangent stiffness, just evaluated ONCE at a
fixed reference stress rather than updated every Newton iteration, so
this reuses those elements' existing large-deformation kinematics
rather than needing new formulas from scratch for elements that
already have a nonlinear formulation), (3) assemble `K_sigma`
globally exactly like `K`/`M` are assembled today, (4) solve
`eigh(Kff, -K_sigma_ff)` — literally the same `scipy.linalg.eigh` call
`solve_modal()` already makes, with `K_sigma` (negated) in place of
`M`.

**Architecture fit**: needs each element to expose a
`geometric_stiffness(elem_coords, stress_state, ...)` method (default
`None`/not-implemented for purely linear elements, since buckling is
undefined for something with no nonlinear kinematics to build
`K_sigma` from) — a natural sibling to the existing
`internal_force()`/`tangent_stiffness()` nonlinear extension point in
`elements/base.py`. `FESystem` gets `assemble_geometric_stiffness(u,
mat, **kwargs)` (mirrors `assemble_tangent_stiffness()`'s existing
per-block loop) and `solve_linear_buckling(mat, n_modes=4, **kwargs)`
(mirrors `solve_modal()`'s structure almost line-for-line, swapping
`self.M` for the freshly-assembled `K_sigma`).

**Validation plan**: the Euler column is THE textbook closed-form
buckling benchmark (`P_cr = pi^2 EI / (KL)^2` for standard boundary
conditions) — build a column from `Beam2DCorotational` or the new 3-D
beam element (Section 2) and confirm the lowest eigenvalue matches the
closed-form `P_cr` directly, the same "known closed-form answer"
validation style `solve_modal()`'s own tests presumably already use
for vibration frequencies.

## 7. Sparse matrix support

**What's missing**: `FESystem.__init__` allocates `self.K =
np.zeros((n_dof, n_dof))` — DENSE, unconditionally. Every solve
(`np.linalg.solve`, `scipy.linalg.eigh`, `np.linalg.inv` in
`solve_transient_implicit()`) operates on dense arrays. Real FE
stiffness matrices are overwhelmingly sparse (a node only couples to
its few mesh neighbors, not to every other node), so this caps
realistic problem size well under ~10&sup5; dof regardless of element
richness — the single biggest ceiling on problem size this package
currently has, and the Phase-1 item in the companion performance
roadmap (`hpc_translation_roadmap.html`).

**Recommended approach**: `scipy.sparse` throughout, following the
FEM community's standard assembly pattern — accumulate element
contributions into COO (coordinate) format DURING the assembly loop
(cheap to append to, and scipy's COO-&gt;CSR conversion correctly SUMS
duplicate `(i,j)` entries, which is exactly what happens when multiple
elements share a node/DOF — no special-case code needed for that),
then convert once to CSR (compressed sparse row) for solving (fast
matrix-vector products, and CSR is what `scipy.sparse.linalg`'s direct
(`spsolve`, SuperLU-backed) and iterative (`cg`, `gmres`) solvers, and
`scipy.sparse.linalg.eigsh` (Lanczos, for `solve_modal()`/the new
buckling solver) all expect.

**Architecture fit**: this is the one item that touches `solver.py`'s
core assembly loop, but as a STORAGE-FORMAT change, not a structural
one — replace `self.K = np.zeros(...)` + `self.K[dofs, dofs] += ke`
(dense scatter-add) with COO row/col/data list accumulation +
`.tocsr()` at the end of each `assemble_*()` call, and replace
`np.linalg.solve(Kff, F)` with `scipy.sparse.linalg.spsolve(Kff_csr,
F)` (direct) or an iterative solver for very large systems. Given how
much of this package is tested via exact/near-exact numerical
agreement with independent computations, the safest rollout is
ADDITIVE: keep the existing dense path as the default (nothing this
package's 100+ existing tests currently assert would change), and add
a `sparse=True` constructor option to `FESystem` that switches storage
format — new tests then assert the sparse and dense paths give
IDENTICAL results on the same model, rather than replacing the dense
path outright and risking a silent behavior change across the whole
existing test suite.

**Validation plan**: (1) direct equality check — for every existing
validated model in the test suite, `sparse=True` and `sparse=False`
should give the same static/modal/transient result to near machine
precision (this is the strongest possible validation: not a new
physics claim, just "the storage format doesn't change the answer");
(2) a genuinely large model (10&sup4;-10&sup5; dof, per the profiling
plan in `hpc_translation_roadmap.html`) where the dense path becomes
impractically slow/memory-heavy and the sparse path is measured (not
assumed) to be dramatically faster — the actual justification for the
added complexity, checked the same way `rom_engine`'s speedup claims
always are.

## 8. Suggested build order

Not a strict dependency chain (most of these are independent), but a
sequencing that front-loads the highest ratio of value to effort and
defers the pieces that depend on something else being in place first:

1. **Higher-order elements (Section 3)** — lowest effort (the base
   class's generic Gauss loop needs zero changes), immediately useful
   on its own, and a good warm-up validated against the existing
   plate-with-hole mesh utility.
2. **Sparse matrix support (Section 7)** — unblocks every other item
   from hitting a problem-size ceiling, and is additive/opt-in so it
   carries no regression risk to the existing dense-path test suite.
3. **3D beam/frame elements (Section 2)** — reuses
   `Beam2DEulerBernoulli`'s bending math directly, moderate effort,
   immediately useful for general 3-D frame problems.
4. **Linear buckling (Section 6)** — reuses `solve_modal()`'s
   `eigh()` pattern almost directly once ANY element with nonlinear
   kinematics exists (`Beam2DCorotational`/`TrussTL2D` already
   qualify), so it can go in parallel with or right after Section 2.
5. **Arc-length solver (Section 5)** — a solver-loop-only addition,
   independent of new elements, valuable as soon as there's a
   snap-through-capable model to test it on (the existing
   `Beam2DCorotational`/`TrussTL2D` already qualify).
6. **2D/3D plasticity + hyperelasticity (Section 4)** — the most
   involved item (needs the Gauss-point state machinery extended to
   continuum elements), best done once the sparse-matrix and
   higher-order-element groundwork is in place, since large-strain/
   plasticity problems are exactly where both matter most.
7. **Shell elements (Section 1)** — scoped last: it's the most
   complex formulation here (assumed-strain tying, drilling DOF,
   local-to-global frame handling) and benefits from having the
   sparse-matrix and higher-order-element infrastructure already
   proven out first.

## 9. Koiter-Newton continuation (Phase 8, Module 21, implemented)

**What motivated it**: not part of this document's original seven-item
survey -- added because the sibling `NonLin-HyROM` paper-reproduction
project (`ROM_formulations/NonLin-HyROM`, outside this repo) needs it
directly. That project reproduces Yang et al.'s 2019 hybrid ROM paper,
whose own related-work section describes the Koiter-Newton (KN) method
(Liang, Abdalla & Gurdal) as the technique its Section 4.2 uses to make
each static training solve in the hybrid ROM's pipeline cheap. Before
this phase, neither `fea_engine` nor `rom_engine` had ANY Koiter/
asymptotic-expansion module (confirmed by grepping both packages for
"koiter"/"asymptotic" -- zero hits) -- Phase 5's arc-length solver and
plain Newton were the only static continuation options.

**What was implemented**: `solve_nonlinear_koiter_newton()`
(`nonlinear_solver.py`) -- see that function's own extensive docstring
for the full derivation (reproduced from Liang & Sun, ICCM2017, "A
reduced-order modeling technique for nonlinear buckling analysis," Eqs.
1-7, cross-checked against Liang, Abdalla & Gurdal's IJNME 2013 journal
paper, and finalized against the actual target paper's own Section 4.2/
Eqs. 16-22 once that was re-read directly -- see the "one correction"
note below). In one sentence: at each converged point, build a CUBIC
(not linear) asymptotic predictor of the equilibrium path in a single
perturbation parameter, extracting the reduced quadratic/cubic force
coefficients Q(u1,u1)/C(u1,u1,u1) via a symmetric finite-difference
stencil against the EXISTING `assemble_internal_force()` (no per-
element analytic re-derivation needed -- works for every element
already in `elements/`), adaptively size the step from how well that
cubic model predicts the TRUE full-model residual (the paper's own
"unbalanced force" criterion), then correct with a chord/modified-
Newton iteration (Eqs. 21-22) that reuses ONE frozen-tangent bordered-
matrix factorization for every correction iteration -- NOT (an earlier
draft's choice) a re-use of Phase 5's fresh-tangent Crisfield corrector;
re-reading the target paper's own equations directly turned up a
cheaper, more faithful alternative, so that draft was replaced. One
real bug caught along the way: implementing Eq. 21's matrix exactly as
OCR'd from the source PDF (with a sign flip on the load-remainder
unknown) made the corrector's residual grow ~2x every iteration --
fixed by re-deriving the correction directly from the linearized
residual instead of trusting the transcription.

**Scope reduction, stated explicitly (read the function's docstring's
own "SCOPE" section before using this on a buckling-sensitive
problem)**: this is the published method's `m=0` (primary-path-only)
case -- ONE perturbation direction, not the general case with extra
directions added for closely-spaced/interacting buckling modes. That
extension (an eigenvalue check on the tangent stiffness to detect a
bifurcation, then augmenting the same bordered-system machinery with
each near-critical eigenvector) is a real, larger piece of future work,
not attempted here. On any snap-through/limit-point path with no
branching -- which is what the `m=0` case is actually equivalent to
the published method's own single-mode numerical examples running on
-- this is a faithful, unreduced instance of the method, not a toy
approximation of it.

**Validation** (`tests/test_koiter_newton.py`): reuses the exact same
von Mises two-bar snap-through truss and closed-form `P(delta)` Phase
5's own `tests/test_arc_length.py` validates against (same shared
ground truth, not a second independently-derived one) --
(1) traces the full path (through both limit points, past full
inversion) matching the closed form to <1e-6 relative error;
(2) matches `solve_nonlinear_arc_length()`'s own already-validated path
to <5e-3 (cross-solver agreement); (3) confirms the actual claimed
efficiency benefit is real and measured, not assumed: reaches a fixed
target displacement in 2 outer steps vs. arc-length's 22 for the same
delta_L seed (~11x fewer steps); (4) an isolated unit check of the
finite-difference Q/C extraction against a HAND-DERIVED closed-form
cubic force law for a single Green-Lagrange truss bar loaded along its
own axis (`f(u) = E*A*(u + 1.5u^2 + 0.5u^3)`, exact to floating-point
precision since the true force law is itself exactly cubic) -- isolates
correctness of the asymptotic-expansion machinery from the corrector/
step-adaptation logic layered on top of it.

## 10. Koiter-Newton solve-to-target-load (Phase 9, Module 22, implemented)

**What motivated it**: Module 21 (`solve_nonlinear_koiter_newton()`)
is an open-ended PATH CONTINUATION driver -- it was tried, first, as
the `fom_solver` for `NonLin-HyROM`'s Case 1 nonlinear-static ROM-
training loop (generate the equilibrium at ONE specific prescribed
load, repeated per training sample), via a "run a few steps then
interpolate/polish back to lambda=1" wrapper. That failed on the real
model: with nothing telling it where to stop, its own adaptive growth
overshot the target load by >14x on one training sample (making the
wrapper SLOWER overall than plain Newton, 185.7s vs. 109.0s, since all
that extra growth was wasted), and drove the shell element into a
floating-point overflow (NaN/Inf) on a larger sample -- an outright
crash, not just an inefficiency. Root cause: Module 21's step-growth
criterion ("does the cubic model still predict the true residual
well?") has no idea a caller wants to stop at a specific load level --
for a genuinely-cubic-ish force law that criterion alone can be
satisfied "forever."

Re-reading the target paper (Yang et al. 2019) directly -- specifically
Fig. 1's flowchart and Section 4.2's text -- confirmed this mismatch
was real, not a tuning problem: Fig. 1 shows the static-ROM machinery
invoked ONCE PER prescribed training load ("determination of nonlinear
static test cases" -> "construction of static reduced-order model" ->
"calculating static tests"), and Section 4.2 says the reduced cubic
polynomial "can be solved with phi=mu*phi_ext by a path-following
technique" -- i.e. aimed at a KNOWN target multiplier mu, not grown
until an open-ended validity check fails.

**What was implemented**: `solve_nonlinear_static_koiter_newton()`
(`nonlinear_solver.py`) -- a SEPARATE function from Module 21, sharing
the same cubic asymptotic predictor machinery (u1/Lbar1, u11/Qbar11,
Cbar1111, same finite-difference Q/C extraction), but aiming each
expansion's perturbation parameter directly at the real root of the
reduced cubic polynomial that lands on the REMAINING GAP to the target
load (`Cbar1111*a^3 + Qbar11*a^2 + Lbar1*a = 1.0 - lambda_current`,
picking whichever real root is closest to the linear estimate) instead
of growing freely -- so there is nothing left to overshoot with. If
that target-aimed step isn't trustworthy (predictor residual check
fails), it shrinks TOWARD ZERO from that root (never past it), accepts
a partial step, and re-expands from the new point -- the paper's own
multi-step behavior (Fig. 3's star markers), just aimed at a caller's
known target instead of an open path. Because the target load for each
accepted step is then externally KNOWN (not a free unknown the way
continuation's lambda is), the corrector needs no bordering trick at
all -- ordinary frozen-tangent Newton at the fixed load suffices, a
genuine simplification of Eq. 21 for this case, not a deviation from
it.

**Validation** (`tests/test_static_koiter_newton.py`, 3 tests): matches
plain `solve_nonlinear_static()` on the SAME von Mises truss benchmark
before its limit point; an isolated Green-Lagrange-truss-bar unit check
identical in spirit to Module 21's own; and a forced-multi-expansion
case (artificially strict `predictor_tol`) confirming the shrink-and-
re-expand loop engages and still lands exactly on the target. Then
validated directly on the real `NonLin-HyROM` Case 1 model (3 of its
actual training-load samples, the same ones that broke the Module 21
wrapper): all 3 reached the target load in a SINGLE expansion (matching
the paper's own "one perturbation step is enough" claim for hardening
problems), agreed with plain Newton to <5e-7 relative, and ran ~15-18x
FASTER (7.3-7.6s vs. 114-137s per solve) -- a bigger speedup than the
paper's own reported 2-vs-9-step ratio (~4.5x), since the cheap chord
corrector also saves per-iteration cost, not just step count. Swapped
into `case1_flat_plate_with_hole.py`'s `fom_solver()` and the full
pipeline re-run end to end: static training time dropped from ~1200s
to 90.3s for the same 10 samples, and the FOM's own (unrelated, already
documented) transient gap reproduced identically to 5 significant
figures, confirming the swap changed only HOW the training equilibria
are computed, not WHAT they converge to.

**Not interchangeable with Module 21** -- worth stating plainly since
getting this wrong is exactly what motivated writing this function:
use Module 21 (`solve_nonlinear_koiter_newton`) to trace an entire
equilibrium path with no specific target in mind; use Module 22
(`solve_nonlinear_static_koiter_newton`) to reach one known prescribed
load level as cheaply as possible. A thin wrapper turning one into the
other is unsafe, not just inexact.

## 11. Generic (m>=1) Koiter-Newton continuation (Phase 10, Module 24, implemented)

> **Fixed 2026-09-25: two defects found on NonLin-HyROM Case 2 (a thin cylindrical shell).**
> (1) *Mode detection* compared `lambda_min` against `trace(K_ff)/nf`. That is not scale-free:
> a thin shell's membrane stiffness inflates the trace, so m=1 fired at step 1 with no
> instability (ratio 5.6e-6 on Case 2). Detection is now relative to the undeformed
> structure's own `lambda_min(K_ff(0))`, with trace/nf as a fallback only for a singular K(0).
> (2) *The m=1 corrector* reused the predictor's (nf+2) bordered matrix and dropped the
> fictitious-load multiplier. Its `f_1 . du = 0` row froze the second amplitude, so the
> residual along `f_1` was never removed (|R| exactly flat; with nf=2, du = 0). It now
> runs Newton on the real residual, bordered by the real load only, with the tangent
> re-assembled per iteration, as in Liang et al. 2014 Sec. 3 ("Newton arc-length" corrector).
> The m=0 branch is unchanged. New tests: `test_m1_corrector_converges_off_the_predictor`
> and `test_mode_detection_ignores_stiffness_contrast`; both fail on the old code.
> `test_generic_driver_robust_across_synthetic_bifurcation` now asserts that m=1 actually runs
> (previously claimed but unchecked), with settings that hit the narrower detection window.
> Backup: `src/fea_engine/nonlinear_solver.py.pre_m1_fix_2026-09-25.bak`.

**What motivated it**: a direct follow-up request to review whether
Sections 9-10's Koiter-Newton work was "implemented in the correct
order" and, if not, to fix it, then implement the method's GENERIC
(multi-mode) case. The review (below) found no bugs in Module 21/22 --
only the previously-disclosed `m=0` scope reduction (Section 9's own
"Scope reduction" paragraph, and `solve_nonlinear_koiter_newton()`'s
own docstring "SCOPE" section) was incomplete relative to the published
method. This section closes that gap with a new function, and records
the correctness review and the fea_engine-vs-rom_engine placement
decision.

**Correctness review of Module 21/22 (no code changes resulted)**:
re-read both drivers line by line against Liang, Abdalla & Gurdal
(2013), Liang & Sun (ICCM2017), and Yang et al. (2019) Eqs. 16-22
directly, and re-ran all 7 existing tests
(`test_koiter_newton.py`, `test_static_koiter_newton.py`). Findings:
the bordered-matrix construction, the single-factorization reuse across
predictor/corrector, the finite-difference Q(u1,u1)/C(u1,u1,u1)
extraction, the Eq.7 collapse to `Cbar1111 = u1.C(u1,u1,u1) -
2*u11.(K_T@u11)`, and the frozen-tangent chord corrector (including the
Eq.21 sign-transcription fix Section 9 already documents) are all
correct for the disclosed `m=0` case -- no new bug found. The ONLY gap
is the one both drivers already disclose in their own docstrings:
neither implements the published method's `m>=1` case (extra
perturbation directions for closely-spaced/interacting buckling modes).
That gap is what Module 24 below fills, as a NEW function, not a
revision of Module 21/22 (see "why a new function" below).

**What was implemented**: `solve_nonlinear_koiter_newton_generic()`
(`nonlinear_solver.py`), plus three private helpers it shares with no
other function (`_directional_QC`, `_two_direction_QC`,
`_generic_cbar_term`). At each step, after assembling the tangent
stiffness, a cheap partial eigensolve (`scipy.linalg.eigh(K_ff,
subset_by_index=[0,1])`) checks for a near-critical mode (smallest
eigenvalue small relative to a reference stiffness scale fixed at the
path's first step). When none is found, the driver reproduces Module
21's own `m=0` formulas exactly (same bordered matrix, same adaptive
step search, same frozen-tangent corrector) as its own fallback path --
confirmed by regression test to match Module 21's output to
~1e-8 relative on the same benchmark. When a near-critical mode IS
found, its eigenvector seeds a SECOND perturbation direction, and the
whole machinery generalizes to an `(nf+2)`-sized bordered system: two
first-order directions and a 2x2 matrix of Lagrange multipliers, three
second-order (quadratic) solves against the SAME factorization, and the
Eq.7 cubic coefficients via the fully general algebraic formula
`Cbar[p;i,j,k] = u_p.C(u_i,u_j,u_k) - (2/3)*[u_ij.K(u_pk) +
u_jk.K(u_pi) + u_ki.K(u_pj)]` (implemented generically against index
dictionaries, not hand-substituted per case, to avoid an algebra
mistake -- and verified to collapse to Module 21's own single-direction
formula when `p=i=j=k`). The two cross-direction tensor coefficients
(`Q01`, `C001`, `C011`) needed for this are extracted via a
*polarization-identity* construction (`_two_direction_QC`): probing
along the sum and difference of the two directions with the SAME
already-validated 1-D finite-difference stencil Module 21 uses, then
solving simple linear algebra for the cross terms -- reusing validated
machinery rather than deriving a new multivariate stencil. The row
belonging to the real load direction gives `lambda(a0,a1)` as before;
the row belonging to the fictitious extra direction is instead a
genuine EQUILIBRIUM CONSTRAINT -- a CUBIC polynomial in `a1` for fixed
`a0`, solved via `numpy.roots`, with the real root closest to the
previous step's accepted `a1` selected for continuity. This cubic, with
potentially up to 3 real roots, is what actually delivers the
published method's headline mode-interaction capability (multiple
candidate equilibria at the same step), not just a bigger predictor.

**Why a new function, not a revision of Module 21/22**: both existing
drivers are already validated and already in production use by
`NonLin-HyROM`'s training pipeline (Section 10's own ~15-18x measured
speedup on the real Case 1 model). Generalizing them in place would
risk that exact, working code path for a capability most callers don't
need. `solve_nonlinear_koiter_newton_generic()` is additive: existing
call sites are completely unaffected, and its own `m=0` fallback path
gives every caller who doesn't need mode interaction (`enable_mode_
interaction=False`, or simply no near-critical mode ever detected) the
same numerical behavior as Module 21, verified by regression test.

**Documented scope of the generalization itself** (see the function's
own "SCOPE" docstring section for the full statement): at most ONE
extra direction is added per step (`m<=1`, not deeply closely-spaced
multi-mode interaction); branch selection at a genuine bifurcation uses
simple continuity to the previous step, not a stability/energy
criterion, so it follows one physically continuous path through a
simple bifurcation rather than automatically discovering or switching
onto a post-buckling branch; and the near-critical-mode check has no
hysteresis across steps. These are stated simplifications of an already
substantial generalization, following the same "disclose the scope,
don't quietly narrow it" convention Module 21/22 established.

**Validation** (`tests/test_koiter_newton_generic.py`, 4 tests):
(1) `test_generic_m0_fallback_matches_existing_driver` -- the
regression check described above, on the same von Mises truss
benchmark Module 21 itself is validated against.
(2) `test_two_direction_QC_matches_hand_derived_closed_form` and
(3) `test_generic_cbar_reproduces_exact_pitchfork_bifurcation` -- the
actual hand-derivable 2-mode bifurcation benchmark: a small synthetic
2-DOF system (no finite elements at all) built from the textbook
coupled symmetric/antisymmetric potential `V(u1,u2) = 0.5*k1*u1^2 +
0.5*k2*u2^2 - gamma*u1*u2^2 + 0.25*beta*u2^4`, whose exact equilibrium
set is closed-form: a trivial linear primary path up to a critical
`u1_cr = k2/(2*gamma)`, past which a secondary branch `u2^2 =
(2*gamma*u1-k2)/beta` bifurcates off it. Because this potential is
exactly quartic, the module's Q/C tensors have no higher-order
remainder along any direction, so both the finite-difference extraction
and the algebraic Eq.7 formula can be checked against EXACT closed-form
values (not approximate ones) -- and the resulting row-1 cubic's three
roots are shown to reproduce `{0, +sqrt((2*gamma*u1-k2)/beta),
-sqrt(...)}` exactly, i.e. the TRUE bifurcated-branch amplitude, not an
approximation of it. This is the strongest form of validation available
for this machinery short of a full symbolic benchmark, and is possible
specifically because the synthetic model's higher-order terms vanish by
construction.
(4) `test_generic_driver_robust_across_synthetic_bifurcation` -- runs
the FULL driver (predictor, corrector, adaptive stepping, mode
detection, all together, not just the isolated tensor machinery) straight
through the same synthetic bifurcation and confirms it detects the
near-critical mode, exercises the `m=1` code path, never diverges or
crashes, and (since `a1=0` is proven to always be an exact root here,
and continuity-based branch selection is seeded at 0) stays on the
still-valid primary path to near machine precision throughout.

**Where this belongs -- fea_engine, not rom_engine, and not both**:
`solve_nonlinear_koiter_newton_generic()` was added to `fea_engine`
alongside Module 21/22, for the same reason those were: the "reduced-
order model" this method builds is an **internal, per-step
implementation detail** of a full-order-model path-following driver --
a small (`nf+1` or `nf+2`-dimensional) linear system built fresh from
`assemble_tangent_stiffness()`/`assemble_internal_force()` at every
converged point, used once as a predictor, and thrown away the moment
the next step begins. It never has a persistent existence a caller
could inspect, reuse, or query independently of the full-order model --
there is no "trained model" object to hand off. That is exactly the
architectural line this codebase already draws between the two
packages: `fea_engine` owns full-order assembly/solving AND solver
DRIVERS (arc-length, displacement control, transient, and now Koiter-
Newton in all three variants) that happen to use small internal
reductions as a numerical technique; `rom_engine` owns STANDALONE,
persistently-built reduced models (`pod.py`, `galerkin.py`, `nnm.py`,
`PolynomialModalROM` in `nonlinear_rom.py`, `FrequencyROM`, etc.) that
are trained once, against a fixed basis, and then queried repeatedly
*without* returning to the full-order model in between. A Koiter-Newton
step is the opposite of that: its reduced model is rebuilt from scratch
at every single step and is never queried again. Splitting it across
both packages (e.g. a thin `rom_engine` wrapper calling back into
`fea_engine` per step) would add an import-direction dependency for no
functional benefit, since nothing about this method's reduced basis is
reusable across steps the way a POD basis or a Galerkin ROM's basis is.
`rom_engine`'s own nonlinear-ROM modules (`nonlinear_rom.py`,
`nnm.py`) remain the right place for anything that DOES need a
persistent, queryable nonlinear reduced model (e.g. as a fast surrogate
trained FROM Koiter-Newton-generated training data, exactly
`NonLin-HyROM`'s own actual usage pattern for Module 22's output) --
that boundary is unchanged by this addition.

## Sources

- Dvorkin, E.N., Bathe, K.J. ["A continuum mechanics based four-node shell element for general non-linear analysis."](https://web.mit.edu/kjb/www/Principal_Publications/A_new_MITC4+_shell_element.pdf) Engineering Computations, 1, 77-88 (1984) &mdash; the original MITC4 formulation.
- Bathe, K.J., Dvorkin, E.N. "A four-node plate bending element based on Mindlin/Reissner plate theory and a mixed interpolation." IJNME, 21, 367-383 (1985) &mdash; the plate-theory precursor `Quad4MindlinPlate` is already based on.
- ["The MITC4+ shell element and its performance."](https://www.sciencedirect.com/science/article/abs/pii/S0045794916300487) &mdash; a more recent refinement, useful if MITC4's known weak spots (in-layer strain artifacts) matter for a specific target problem.
- Crisfield, M.A. ["A fast incremental/iterative solution procedure that handles 'snap-through'."](https://www.researchgate.net/publication/8618852_Arc-length_technique_for_nonlinear_finite_element_analysis) Computers & Structures, 13, 55-62 (1981) &mdash; the original cylindrical arc-length method.
- Crisfield, M.A. ["An arc-length method including line searches and accelerations."](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620190902) IJNME, 19, 1269-1289 (1983).
- Crisfield, M.A. *Non-linear Finite Element Analysis of Solids and Structures*, Vol. 1-2, Wiley (1991/1997) &mdash; the standard reference for both the arc-length method and the 3-D corotational beam formulation.
- Simo, J.C., Taylor, R.L. "Consistent tangent operators for rate-independent elastoplasticity." Computer Methods in Applied Mechanics and Engineering, 48, 101-118 (1985) &mdash; the standard modern radial-return formulation.
- Simo, J.C., Taylor, R.L. ["A return mapping algorithm for plane stress elastoplasticity."](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620220310) IJNME, 22, 649-670 (1986) &mdash; the plane-stress-specific local-Newton extension.
- ["Analysis of the compressible, isotropic, neo-Hookean hyperelastic model."](https://link.springer.com/article/10.1007/s11012-022-01633-2) Meccanica &mdash; isochoric/volumetric split formulation reference.
- ["Linear Buckling Analysis of a 3D solid"](https://comet-fenics.readthedocs.io/en/latest/demo/buckling_3D/buckling_3d_solid.html), FEniCS numerical tours &mdash; a concrete, code-level worked example of the `[K + lambda*K_sigma]phi=0` eigenproblem this section's implementation follows.
- ["The serendipity family of finite elements."](https://arxiv.org/pdf/1101.0645) &mdash; general treatment of serendipity (Hex20/Quad8-type) vs. full Lagrange (Hex27-type) element families and why serendipity is the standard practical choice.
- SciPy sparse documentation: [`scipy.sparse`](https://docs.scipy.org/doc/scipy/reference/sparse.html) &mdash; COO/CSR format semantics, including the duplicate-entry summation behavior the assembly loop relies on.
- `computation-suite/hpc_translation_roadmap.html` (this project) &mdash; the companion performance-scaling roadmap Section 7 (sparse matrices) directly feeds into.
- `rom_engine/docs/phase4_error_bounds_greedy_roadmap.md` (this project) &mdash; the structural template (research &rarr; architecture mapping &rarr; phased build &rarr; validation plan &rarr; sources) this document follows.
- Liang, K., Abdalla, M.M., Gurdal, Z. ["A Koiter-Newton approach for nonlinear structural analysis."](https://onlinelibrary.wiley.com/doi/10.1002/nme.4581) IJNME, 96(12), 763-786 (2013) &mdash; the original Koiter-Newton (KN) method journal paper.
- Liang, K., Sun, Q. ["A reduced-order modeling technique for nonlinear buckling analysis."](https://www.sci-en-tech.com/ICCM2017/PDFs/2777-7914-1-PB.pdf) ICCM2017 &mdash; the shorter derivation (Eqs. 1-7) `solve_nonlinear_koiter_newton()`'s docstring reproduces directly.
- Liang, K. *A Koiter-Newton Arclength Method for Buckling-Sensitive Structures*, PhD thesis, Delft University of Technology (2013) &mdash; full derivation including mixed-kinematics/higher-order-strain extensions beyond the `m=0` scope implemented here.
- Yang, C., Liang, K., Rong, Y., Sun, Q. ["A hybrid reduced-order modeling technique for nonlinear structural dynamic simulation."](https://doi.org/10.1016/j.ast.2018.11.008) Aerospace Science and Technology, 84, 724-733 (2019) &mdash; the `NonLin-HyROM` sibling project's target paper; its Fig. 1 flowchart and Sections 4.1-4.2 are what motivated implementing both Module 21 and Module 22 here (see Sections 9-10 above).
- Thompson, J.M.T., Hunt, G.W. *A General Theory of Elastic Stability*, Wiley (1973) &mdash; the standard reference for the coupled symmetric/antisymmetric two-mode ("mode interaction") model Section 11's synthetic pitchfork-bifurcation benchmark (`tests/test_koiter_newton_generic.py`) is a minimal instance of.
