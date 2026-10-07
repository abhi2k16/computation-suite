# Shell elements in `fea_engine`

This is the single reference for everything shell-related: what's
implemented, what it's validated against, its known limits, and the
design history behind it. It replaces three earlier documents —
`geometric_nonlinear_shell_roadmap.md`, `shell_rotation_coupling_fix_roadmap.md`,
and `director_based_shell_element_roadmap.md` — which grew, phase by
phase, into ~2000 lines of build-log narration. Section 1 is what a
user or a future contributor actually needs day to day; Section 4 keeps
the design reasoning (including the dead ends) in condensed form,
because several of those dead ends are load-bearing precedent for why
the current architecture looks the way it does.

## 1. Current state

Three shell element classes exist, in `elements/shells.py` and
`elements/shells_director.py`. All are 4-node quads.

| Class | DOF/node | Nonlinear? | Registered? | Status |
|---|---|---|---|---|
| `Shell4MITC` | 6 | No (linear) | `"shell4_mitc"` | Production. |
| `Shell4MITCCorotational` | 6 | Yes (co-rotational) | `"shell4_mitc_corotational"` | Production, with a documented architectural ceiling (below). |
| `Shell4Director` | 6 | Yes (director-based, total-Lagrangian) | `"shell4_director"` | Production (2026-09-10): analytic tangent shipped, mesh-converged at nx=20 scale (below). |

### 1.1 `Shell4MITC` — linear

Membrane (`Quad4PlaneStress`) + Mindlin plate bending
(`Quad4MindlinPlate`) + Dvorkin-Bathe MITC4 assumed-strain transverse
shear, combined block-diagonally, plus a small artificial diagonal
stiffness on the drilling (`theta_z`) DOF (`_drilling_stiffness()`,
its own docstring: "NOT real physics" — needed only to prevent a
singular tangent when neighboring elements are coplanar). This is a
**flat-shell superposition element**: membrane and bending never
exchange energy (`K_local` is exactly block-diagonal between
`{u,v,theta_z}` and `{w,theta_x,theta_y}`), which is the single fact
that shapes everything below. Built as Phase 7 of
`general_purpose_extensions_roadmap.md`; validated against
`Quad4MindlinPlate` in the flat-plate limit, a curved cylindrical-arc
cantilever vs. `Hex8Solid3D` mesh convergence, and a frame-objectivity
test.

**Known limitation**: no membrane-locking fix (only shear locking is
addressed by the MITC4 tying). A working MITC4+ implementation was
built and then found to be an exact no-op for this element's
membrane-DOF-only scope (Section 4.5) — not shipped.

### 1.2 `Shell4MITCCorotational` — geometric nonlinear, flat-shell architecture

A co-rotational wrapper around `Shell4MITC`: at each call, a local
frame `(e1,e2,e3)` is extracted from the current corner geometry, the
membrane+drilling block uses a position-based extraction through an
exact single-axis drilling rotation (`_exact_drill_rotation()`), the
bending+shear block uses the raw nodal DOFs (no frame correction),
and `Shell4MITC`'s own local stiffness is applied to the resulting
local relative DOF vector. Analytic tangent (`tangent_stiffness()`),
cross-validated against complex-step to ~1e-16 relative; ~23x faster
than the complex-step version it replaced.

**Validated regime**: tip-deflection agreement with the exact
Bisshopp-Drucker (1945) cantilever elastica is within ~2% for tip
rotations up to **~13 degrees**, degrading beyond that (~5% at ~21
degrees, ~8% at ~24 degrees) — a load-independent modeling limit, not
a discretization one.

**The architectural ceiling** (the project's most-investigated open
problem — Section 4 has the full account): this element predicts
**exactly zero** bending-induced axial foreshortening, at any tip
rotation, on any mesh. This follows structurally from the flat-shell
block-diagonal decoupling above — there is no membrane-bending
coupling term in the kinematics for any amount of load-step
refinement to expose. The currently-shipping fix
(`_bending_membrane_coupling_force()`, a deviation-from-mean von
Karman coupling term, chosen specifically because it stays exactly
zero for a rigid tilt) recovers foreshortening that is correctly
*signed* but only **~0.05%–0.06%** of the true elastica value — a
correct-direction, not-remotely-sufficient patch. Eight further
attempts at a fix (Section 4.3) were tried and rejected; a 4-source
literature review (Section 4.4) diagnosed *why* — the fix needs a
different **kinematic architecture**, not another energy term — which
directly motivated `Shell4Director` (Section 1.3).

**Other limitation**: single-element rigid TILTS (not just drilling)
are only *near*-exact, not exact — ~1.8% spurious force at 30 degrees,
because the bending block's rotational DOFs are read off a fixed
reference frame rather than an exact current-mean-relative one. An
exact fix exists as opt-in, unwired infrastructure
(`_local_relative_dofs_exact_rotation()`, Section 4.4) but was never
wired into `internal_force()`/`tangent_stiffness()` — see Section 4.4
for why (real, not hypothetical, blockers).

Related: `Beam3DCorotational` (`elements/beams3d.py`), a 3-D
corotational beam built alongside this element's own investigation,
fully implemented and validated (u=0 matches `Beam3DEulerBernoulli`;
rigid rotation about any axis gives exact-zero force).

### 1.3 `Shell4Director` — geometric nonlinear, director-based architecture

A genuinely different element (not a patch on `Shell4MITCCorotational`):
each node carries a **director vector** (the fiber/normal direction),
rotated via the exact Rodrigues exponential map every call —
`t_a' = exp_map(theta_a) @ t_a`. Because rotation *about* a node's own
director leaves it unchanged exactly (not approximately), this
architecture has no drilling-stiffness problem by construction, and
because the exponential map is exact at any magnitude, it needs no
corotational wrapper and no incremental/committed state — `u_elem` is
simply the total displacement/rotation history (`elements/shells_
director.py`, `shells_director.director_update()`).

DOF convention: 6/node (not the textbook 5), for mixed-mesh
compatibility with this project's other 6-DOF elements; the redundant
6th (drilling) DOF is regularized the same way `Shell4MITC` does
(kept, not eliminated — see Section 1.3.3).

**Strain measures** (`Shell4Director.strain_energy()`):

- **Membrane**: full Green-Lagrange strain built from the *current*
  deformed tangent vectors, `eps = 0.5*(g.g - G.G)` — provably exact
  (zero) under a rigid rotation of any magnitude, since `g = R@G` for
  an orthogonal `R`.
- **Curvature**: `Shell4MITC`'s own unmodified `Bb` bending operator,
  applied to a nonlinearly-updated `(w, betax, betay)` triple where
  `betax_a = e1_0 @ t_a`, `betay_a = e2_0 @ t_a` use the *exact*
  current director. Also exact under rigid rotation, because `Bb` is a
  pure spatial-derivative operator and a rigid rotation gives an
  identical (uniform) director at every node.
- **Transverse shear**: NOT the reused linear `Bs` — that was tried
  first and found to give millions of joules of spurious energy under
  a 90-degree rigid rotation (Section 1.3.1). Fixed with a genuinely
  rigid-rotation-exact MITC-tied covariant shear strain
  (`_mitc4_shear_nonlinear()`), reusing `Shell4MITC`'s tying-point
  *locations* but evaluating the full covariant Green-Lagrange strain
  there instead of a linear B-matrix.

Tangent strategy: real (not complex-step) central finite difference,
applied twice — `internal_force()` is the FD gradient of
`strain_energy()`, `tangent_stiffness()` is the FD Jacobian of that.
`director_update()` (`_exp_map()`) is real-valued only (a norm/branch
on its small-angle fallback), so complex-step isn't available; an
analytic B-matrix was not attempted (see 1.3.2).

#### 1.3.1 Validated results

- **Rigid-rotation exactness** (the decisive structural claim): after
  the shear fix, membrane, bending, AND shear energy are all exactly
  zero (~1e-25, machine precision) under a rigid rotation of any
  magnitude (0.5–90 degrees) about any axis (in-plane, out-of-plane,
  general) — not merely small, at every tested case.
- **Linear (`u=0`) consistency**: `tangent_stiffness(u=0)` matches the
  separately-derived linear `stiffness()` (Section 1.3.3) exactly on
  membrane and drilling blocks; the bend+shear block diverges by
  ~2.4% (of the full matrix's max entry), confined to the shear
  block's `w`-`beta` cross-coupling terms — the accepted price of the
  shear fix (below).
- **The headline result — foreshortening recovery**: on a small
  (1- and 3-element, see 1.3.2 for why) cantilever-strip comparison
  against the exact elastica at ~13 degrees tip rotation, axial
  foreshortening recovery goes from **~49%** of the true value (1
  element) to **~94%** (3 elements) — a clear convergence trend, and
  already **three orders of magnitude** past `Shell4MITCCorotational`'s
  own ~0.05%–0.06% ceiling at a comparable load. This is the payoff
  the whole director-based-element effort was undertaken to test for.

#### 1.3.2 Known limitations

- **Performance — RESOLVED (2026-09-10)**: the original double-real-FD
  tangent cost ~50 seconds per element per Newton tangent evaluation
  (measured on this machine: ~2.6s for a single element, still the
  limiting cost at mesh scale). An analytic `internal_force()` now
  replaces the inner FD-of-`strain_energy()` layer — a hand-derived
  gradient built from the same `_dexp_action(v,p) = d(exp_map(v)@p)/dv
  = -R(v)@skew(p)@Jr(v)` SO(3) right-Jacobian identity, chained through
  membrane (standard Total-Lagrangian B-matrix), curvature (`Bb`, fixed,
  times `d(dof_bs)/du`), and shear (the MITC tying-point covariant
  strain's own dependence on each node's director). Validated against
  the original FD version (kept as `_internal_force_fd()`, the
  correctness oracle) to ~1e-8–1e-11 relative at generic states, and
  EXACTLY zero (not just small) at `u=0` — more precise than the FD
  oracle's own ~7e-4 noise floor there. `tangent_stiffness()` is now a
  SINGLE real-FD Jacobian of this analytic force (48 cheap calls, not
  48 calls each internally 48 more) — measured ~26x faster on this
  element/mesh (2.56s → 0.098s per element per iteration). See
  Section 4.6 for the full derivation record.
- **Mesh convergence at scale — RESOLVED (2026-09-10)**: with the
  analytic tangent, a full `nx=20` cantilever mesh (matching
  `Shell4MITCCorotational`'s own elastica-benchmark scale, previously
  unaffordable) now solves in ~54s and gives foreshortening recovery
  of **103.9%** of the true elastica value (tip deflection error
  1.7%) — closing the gap the 1→3-element 49%→94% trend could only
  gesture at. An `nx=10` mesh (27s) gives 102.7%. Both runs use
  `LOAD_FRAC=0.15` (~13° tip rotation), the same regime
  `Shell4MITCCorotational`'s own benchmark uses.
- **Mixed-mesh compatibility with `Shell4MITCCorotational`**: DOF
  count matches (6/node) but `theta_z`'s physical meaning does not
  (closed-form director-derived angle here vs. an independent Euler
  DOF with artificial stiffness there) — mixing the two element types
  at shared nodes needs explicit handling, not yet designed.
- **Flat reference only**: every node's reference director equals the
  same flat-facet normal `e3_0`; a per-node-varying reference director
  (for a genuinely curved or faceted reference shell) is out of scope.

#### 1.3.3 The shear sign divergence (why `stiffness()` and `tangent_stiffness(0)` don't fully agree)

Worth calling out on its own, because it's the single most important
finding of the nonlinear-kinematics phase. The first, simpler design
reused `Shell4MITC`'s own linear `Bs` (transverse shear operator)
directly, feeding it the same nonlinearly-updated `(w,betax,betay)`
triple as `Bb`. That gives an *exact* match to `Shell4MITC.stiffness()`
at `u=0` (checked first, and used as the gate for the element's linear
stiffness, `Shell4Director.stiffness()`) — but it is **not**
rigid-rotation exact: `Bs` mixes a derivative of `w` with the *raw
value* of `beta` (`gamma = dw/dx - beta_interp`), and "uniform value
maps to zero" — the property that makes `Bb` exact — is false for a
value-based term. Direct testing confirmed this: millions of joules of
spurious shear energy under a 90-degree rigid tilt about an in-plane
axis. The fix (the full covariant Green-Lagrange shear, 1.3) is exact
under rigid rotation but linearizes to the *opposite* sign convention
on the `w`-`beta` cross term from `Bs`'s own. Both formulas are
internally consistent; `Bs`'s convention is a standard
Timoshenko-beam-style choice (`gamma = dw/dx - beta`, `kappa =
d(beta)/dx`, no relative flip) that happens not to be objective at
finite rotation — a property that doesn't matter for `Shell4MITC`
(always evaluated at the reference, small-rotation state) but does
matter once shear is evaluated at a large total rotation. Net effect:
`Shell4Director.stiffness()` (Phase 2, still exactly matches
`Shell4MITC`) and `Shell4Director.tangent_stiffness(u=0)` (Phase 3,
rigid-rotation-exact) disagree by ~2.4% on the shear block's
cross-coupling terms specifically — an accepted, documented trade-off,
not a bug in either.

## 2. Three shell formulation categories (why the architecture split exists)

Standard FEA texts distinguish three broad approaches to shell
elements. This project has built two of them.

1. **Flat-shell / superposition** (`Shell4MITC`, `Shell4MITCCorotational`):
   membrane + plate-bending elements combined block-diagonally, with
   an artificial drilling stiffness for the resulting singular 6th
   DOF. Simple, well-understood, but structurally unable to couple
   membrane and bending strain without an added term.
2. **Degenerated / director-based** (`Shell4Director`): Ahmad/Irons-
   Zienkiewicz (1970) / Ramm (1977) / Simo-Fox (1989) lineage — the
   family Abaqus's S4/S4R and the Grange & Bertrand (2025)
   co-rotational shell paper both belong to. Each node carries a
   physical director, rotated exactly every increment; membrane-
   bending coupling and rigid-rotation invariance both follow directly
   from the kinematics, with no added energy term.
3. **Curvilinear / Kirchhoff-Love / isogeometric**: C1-continuous
   NURBS discretization. Not implemented — this project has no IGA
   infrastructure at any layer (mesh, solver, element interface), and
   building it would be a separate, much larger undertaking with no
   current driver.

## 3. Open risks / follow-up work (consolidated)

- **`Shell4Director` fully analytic tangent — DONE (2026-09-10)**.
  `tangent_stiffness()` is now a fully assembled material+geometric
  stiffness (mirroring `Tet10SolidTL`'s `B_L^T D B_L + K_geo` split for
  membrane; a new derivation for curvature/shear), not an FD Jacobian
  of anything — ~0.03s/element/iteration (from ~2.6s originally, ~0.1s
  at the single-level-FD stage). The one non-closed-form piece,
  `_dexp_hessian()` (the director update's second derivative), uses a
  small LOCAL FD of the already-analytic `_dexp_action()` rather than a
  hand-transcribed closed-form SO(3) Hessian — a deliberate lower-risk
  choice, not an oversight; see Section 4.6. This is what made the
  wing-pipeline NNM backbone (item 24, below) newly worth costing out.
- **`Shell4Director` curved/unstructured-mesh safety** — added to
  `build_mesh.ROTATIONAL_DOF_ELEMENTS` on registration (conservatively
  gated the same way `Shell4MITCCorotational` is), but its own
  curved-mesh topology envelope has not been separately checked —
  Section 1.3.2's "flat reference only" limitation is still open.
- **`Shell4MITCCorotational`'s foreshortening ceiling** — Section 4.3's
  8 dead ends and Section 4.4's diagnosis both point toward the
  architecture change `Shell4Director` represents, not a further
  patch on the flat-shell element; `Shell4MITCCorotational` itself is
  not expected to close this gap further.
- **Mixed shell meshes** (`Shell4Director` + `Shell4MITCCorotational`
  sharing nodes) — undefined, needs explicit handling.
- **Membrane locking (MITC4+) for `Shell4MITC`** — investigated,
  found to be a no-op for this element's scope (Section 4.5); not a
  live item unless the scope changes (e.g. w-coupling is added).
- **Explicit dynamics** for `Shell4MITCCorotational` — **validated
  2026-09-10** (Wave 6 item 33, `docs/consolidated_future_roadmap.md`):
  `nonlinear_solver.solve_transient_explicit_nonlinear()` (item 31)
  runs end-to-end on a real multi-element `Shell4MITCCorotational`
  cantilever mesh (`tests/test_explicit_shell_dynamics.py`) — finite,
  bounded output, zero `tangent_stiffness()` calls confirmed directly
  by call-counting (not just inferred), matching the driver's own
  "no tangent needed at all" claim. This exercised (and, along the
  way, fixed a real bug in) `elements/base.py`'s generic `Element.
  lumped_mass()`: it was silently under-lumping every translational
  direction by a factor equal to the element's own translational-
  direction count (2x for Quad4/TrussTL2D, **3x for a shell's x/y/z**)
  whenever more than one translational direction was tracked — the
  shell case this item newly exercised for the first time; see that
  method's own docstring for the full root-cause writeup. `Shell4Director`
  was NOT separately re-validated under this driver (no architectural
  reason it would behave differently — same `mass()`/`internal_force()`
  convention — but not run through the explicit test above); a full
  free-vibration energy-conservation demo (mirroring the 1-D beam
  energy check `tests/test_explicit_nonlinear_transient.py` already
  has) was not attempted either, since this mesh's own exact global
  critical timestep measures ~40,000x smaller than its fundamental
  period — a few periods would need ~10^5 real shell-element internal-
  force assemblies, well outside a practical validation budget here;
  the short-window check above establishes correctness of the
  mass/force wiring, which was this item's actual scope.
- **Meshing items 28/29** (`generalized_mesh_grading_roadmap.md`:
  multi-hole/unstructured grading against rotational-DOF elements) —
  blocked on the underlying rotational-DOF risk `Shell4MITCCorotational`'s
  own dead ends represent; skipped by user decision rather than
  attempted against an open risk.
- **Wing-pipeline Phase D, case (b)** (kinematic IC + NNM backbone,
  `Multi_Fidelity_NL_Structural_ROM`) — the foreshortening-gap blocker
  is closed (`Shell4Director`'s own mesh-converged result, Section
  1.3.2), and the analytic tangent (above) made a real cost estimate
  possible for the first time: on the actual 140-element wing mesh,
  ~20s per Newmark-Newton iteration (measured directly, `solve_
  nonlinear_transient`, not just per-element assembly cost), so a
  single amplitude point (~30 time steps) is on the order of tens of
  minutes to a few hours depending on iterations-to-converge, and a
  multi-point backbone is a multi-hour batch job — feasible in
  absolute terms, not yet run to completion (only probed for early
  convergence behavior; a full run is a session/environment-scale
  undertaking, not a quick check). The probe itself found a real,
  moderate-confidence risk signal worth flagging before committing to
  a full run: with a 3x-thickness kinematic IC and `dt=2ms`, the
  Newton residual decreased for 3 iterations (15500→9210→2515) then
  INCREASED by iteration 7 (7723) — non-monotonic, despite `solve_
  nonlinear_transient`'s own `line_search=True`/`trust_region=True`
  already being on by default. This echoes, without necessarily being
  identical to, the same class of Newton fragility the solid-mesh
  version of this exact case needed algorithmic rescue for. A smaller
  IC amplitude, a smaller `dt`, or both, is the natural next thing to
  try before attempting a full run. Case (a) (released free-decay,
  `Shell4MITCCorotational`) is unaffected and remains done (Section
  4.2) — this finding is specific to `Shell4Director` on a large
  kinematic IC, a combination case (a) never exercised.

## 4. Design history (condensed)

### 4.1 `Shell4MITC` (linear) — built, validated

Phase 7 of `general_purpose_extensions_roadmap.md`. Dvorkin & Bathe
(1984); Bathe & Dvorkin (1985) for the plate-theory precursor. Nothing
further to record here — see Section 1.1 for current status.

### 4.2 `Shell4MITCCorotational` — Phases A–D

- **Phase A**: `internal_force()` + an initial complex-step tangent.
  The as-built kinematics required THREE design iterations before a
  multi-element cantilever mesh would actually converge — the first
  attempt (one uniform frame update applied to all 6 DOF/node) leaked
  a bending state's out-of-plane translation into the much-stiffer
  membrane block, causing catastrophic Newton divergence on an
  ordinary load step despite passing every isolated-element check.
  The shipped design instead uses two different, DOF-group-appropriate
  corrections (drilling-only position-based frame for membrane;
  raw, uncorrected DOFs for bending), licensed by `K_local0`'s exact
  block-diagonal structure.
- **Phase B**: analytic tangent (material term + geometric/"spin"
  term), replacing complex-step as the default. Cross-validated to
  ~1e-16 relative; ~23x faster.
- **Phase C**: the elastica benchmark (Section 1.2) — found the
  foreshortening ceiling, quantified rather than hypothesized.
- **Phase D**: wing-cantilever remesh (in the sibling
  `Multi_Fidelity_NL_Structural_ROM` project). Found and fixed a real
  bug along the way: every mesh through Phase C used axis-aligned CCW
  strips, where the reference local frame trivially equals the global
  frame; the wing's real Gmsh mesh is uniformly CW-wound, exposing
  that the bending block was silently assuming that trivial case
  (60–84% tangent error on the wing mesh). Fixed by projecting the
  bending block onto the fixed reference frame instead of raw global
  components — a strict no-op wherever the reference frame was already
  the identity. Case (a) (released free-decay dynamics) validated
  cleanly; case (b) (kinematic IC + NNM backbone) deferred, blocked on
  the foreshortening gap (Section 3).

### 4.3 The membrane-bending coupling problem — 8 dead ends

Every attempt below modifies the internal-force/energy expression
evaluated once, against a single fixed reference frame — the common
thread the literature review (4.4) eventually identified as the
actual problem.

| # | Approach | Outcome |
|---|---|---|
| 1–3 | Uniform full-tilt frame applied to all 6 DOF/node | Frame-sensitivity leak, or `w`'s absolute magnitude leaking into the membrane block (~1e4x spurious force on a real Newton step) |
| 4 | Von Karman coupling driven by each node's **absolute** bending rotation | Recovers correct foreshortening magnitude, but the tangent goes locally **indefinite** at ordinary Newton states (~1.7°); 3 different solvers (plain Newton, mass-regularized transient, arc-length) all failed to push through |
| 5 | Saturated/tanh-regularized absolute rotation | Mixed: a fine load ramp fails *earlier* than a coarse one — a genuine Newton fixed-point instability |
| 6 | Adaptive arc-length continuation on dead end 4 | Reached 63% of true foreshortening before hitting a **spurious** limit point (the true elastica BVP has none) — conclusively rules out "better solver" |
| 7 | Mixed/Hellinger-Reissner formulation (lagged internal-stress unknown) | The lagged update is a linearization valid only for small steps; the unknown starts at zero and must jump to a substantial value on the first Newton correction — diverges by construction |
| 8 | Element-constant (not bilinear) absolute rotation | Does not fix the indefiniteness — the problem is structural to using absolute rotation in this term, not the field's spatial order |

Currently shipping: the deviation-from-mean coupling term (stable,
~0.05–0.06% foreshortening recovery — Section 1.2).

### 4.4 Literature review (2026-09-08) and the resulting diagnosis

Four sources reviewed for applicability:

- **Grange & Bertrand, "Co-rotational 3D shell element using
  quaternion algebra..."** (*Forces in Mechanics* 20, 2025) —
  directly relevant. The useful piece is not quaternions (the paper
  says so explicitly) but the kinematic decomposition: total nodal
  motion split, via a well-posed Newton-Raphson residual (not a
  heuristic frame), into one element-wide rigid motion plus per-node
  internal, purely strain-generating rotations — with the rigid part
  removed *exactly*.
- **Jun, Yoon, Lee & Bathe, "The MITC3+ shell element enriched..."**
  (*CMAME* 337, 2018) — not relevant (a membrane-locking fix for
  triangles, not a rotation-handling paper); notable only for using no
  drilling DOF at all (5 DOF/node), sidestepping rather than solving
  the problem this project's 6-DOF architecture has.
- **Kim & Kim, "A three-node C0 ANS element..."** (*CMAME* 191, 2002)
  — not relevant (a shear-locking fix, unrelated to rotation).
- **Abaqus Theory Guide, Section 3.6.5** — directly relevant, the most
  actionable source. No drilling DOF: a closed-form 2×2 polar-
  decomposition angle (`tan(Delta_psi) = (f_hat_12-f_hat_21)/(f_hat_11+f_hat_22)`)
  replaces both a redundant unknown and an artificial stiffness.
  Curvature/normal rotation via nodal spin interpolated into a
  quaternion update on the director — the same machinery Grange &
  Bertrand independently arrived at. **The key finding**: Abaqus does
  *not* add a coupling energy term to a fixed-configuration strain.
  Membrane and curvature are each computed independently (Koiter-
  Sanders split, no quadratic cross term), and the coupling/
  foreshortening instead comes from the frame itself (`t_alpha`) being
  re-derived *exactly, every increment* — nonlinearity accumulates
  through many small, well-conditioned re-orientations, not one large
  quadratic-in-rotation term fighting the membrane/bending stiffness
  ratio in a single residual.

**Diagnosis**: `Shell4MITCCorotational` is a single-shot,
total-Lagrangian-style residual — every dead end tried to inject
enough extra physics into *one* evaluation against a fixed frame. Both
external sources instead track the frame as committed, incrementally-
updated state. This directly motivated two further investigations
(building blocks A–C below) and, ultimately, `Shell4Director`
(Section 1.3) as the architecture that has this property natively.

### 4.5 Building blocks A, B, C (investigated before `Shell4Director` was scoped)

- **Building block A — exact rigid-motion extraction**: generalized
  the drilling block's exact single-axis rotation
  (`_exact_drill_rotation()`) to a full 3-vector Frechet-mean rotation
  extraction (`_mean_rigid_rotation()`, Newton/fixed-point iteration
  on SO(3)). Validated in isolation (exact zero residual for any
  common rotation, 0.5–150°, round-trips against `scipy.spatial.
  transform.Rotation` to <1e-10) and wired as an opt-in
  `_local_relative_dofs_exact_rotation()` — but **not** wired into
  `internal_force()`/`tangent_stiffness()` themselves: (1) it's
  real-valued only (`_mean_rigid_rotation()`'s norm/branching), so
  it can't be complex-stepped, the correctness oracle every other
  tangent in this class uses; (2) making `_exp_map()` holomorphic is
  tractable but `_log_map()` is not (`arctan2` has no straightforward
  entire-function substitute) — left genuinely unresolved, the
  concrete reason `Shell4Director` later chose real-FD over
  complex-step from the start (Section 1.3).
- **Building block B — additive Koiter-Sanders strain split**:
  investigated *before* implementation and found to have no
  standalone content. `K_local0` is exactly block-diagonal between
  membrane+drilling and bending+shear, and the drilling
  regularization is *also* purely diagonal within the membrane group
  — so membrane force depends only on `K_local0[uv,uv] @ dof_local[uv]`,
  completely independent of whatever feeds the rotational DOF slots.
  Feeding an exact curvature into the existing `K_local0` (what a
  single-shot Koiter-Sanders split reduces to) is therefore
  mathematically incapable of touching membrane force — proved, then
  confirmed numerically, before any implementation was attempted.
- **Building block C — incremental frame commitment**: two attempts,
  both ruled out by direct evidence.
  - *Attempt 1* (generalize the whole DOF extraction, including
    translation, to a committed baseline): reproduced dead end 3
    immediately — a general 3-vector frame's z-row deviates from
    `[0,0,1]` the moment any within-element bending is present,
    independent of increment size. Reverted; no code shipped.
  - *Attempt 2* (leave translation alone; make only the coupling
    term's rotation input "absolute since last commit"): implemented
    as `init_state()`/`commit_state()` plus a `state=` branch,
    holomorphic (so its tangent reuses complex-step, cross-checked to
    2.6e-11 relative against real-FD) — the plumbing is solid and
    kept, opt-in. The *formula* is not: committing the same final
    rotation via 1/2/4/8/16 sub-steps shows the accumulated add-on
    stress exactly HALVES every time the step count doubles —
    converging to zero, not a stable value. Root cause: the term
    projects onto FIXED axes, so it's a plain state function with no
    genuine path dependence, and a correctly-telescoping accumulation
    of any state function collapses to evaluating it once at the
    current state — i.e. even a bug-fixed version reduces to dead end
    4's already-rejected formula. Confirmed on the elastica benchmark
    directly: 2 coarse steps gave a 30%-looking foreshortening ratio
    that was a discretization artifact — 4 finer steps gave 179%
    (wrong direction), proving non-convergence rather than inferring
    it.
  - **What this ruled in**: a coupling term only benefits from
    incremental commitment if it's measured against a frame that
    *itself* rotates with each commit — not any quantity merely
    chunked into smaller load steps. This is the exact insight that
    motivated `Shell4Director`: a director-based element has that
    rotating frame as its native DOF, not bolted-on state.

### 4.6 `Shell4Director` — Phases 1–3, then the analytic-tangent follow-up

- **Phase 1** (kinematics primitives, isolated): `director_update()`
  (`t' = exp_map(theta) @ t_ref`, reusing `_exp_map()` directly) and
  `drilling_angle_from_tangents()` (closed-form 2×2 polar-decomposition
  angle, `psi = atan2(c-b, a+d)`, verified against an independent SVD
  polar decomposition to machine precision across 200 random trials).
  31 tests.
- **Phase 2** (linear stiffness): derived that the linearized director
  rotation `theta_a x t_a` reduces, on a flat reference, to
  `Quad4MindlinPlate`'s own `(betax,betay)` convention exactly
  (`betax<->+theta_y`, `betay<->-theta_x`) — cross-checked against the
  real, previously-found `_BEND_SIGN` bug fix (a genuine 40x–1500x
  stiffness error on non-axis-aligned meshes), not trusted on algebra
  alone. `Shell4Director.stiffness()` (independently assembled, own
  Gauss loop) matches `Shell4MITC.stiffness()` exactly on an
  axis-aligned mesh and to <1e-10 relative on general orientations and
  a distorted quad. 10 tests. One correction made here: the
  closed-form drilling angle does *not* eliminate the need for
  `Shell4MITC`'s numerical drilling regularization — it's exactly zero
  at `u=0` (Phase 1), so it can't affect the linear stiffness; the
  regularization is kept, mirroring Abaqus's own S4/S4R.
- **Phase 3** (full nonlinear): Section 1.3 above has the complete
  account — finite-rotation strain measures, the shear sign fix, the
  rigid-rotation-exactness result, and the elastica foreshortening
  result. 57 tests.

Effort note, confirmed rather than merely estimated up front: Phase 3
alone — specifically finding and fixing the shear rigid-rotation
defect — was comparable in difficulty to `Shell4MITCCorotational`'s
own original Phase A/B build, consistent with this element having been
scoped from the start as "comparable to or exceeding" that effort.

- **Analytic tangent (2026-09-10)**, the leftover item this element's
  own Phase 3 flagged as its blocking constraint for `ELEMENT_REGISTRY`:
  hand-derived `_dexp_action(v,p) = d(exp_map(v)@p)/dv =
  -R(v)@skew(p)@Jr(v)`, the SO(3) right-Jacobian identity (Chirikjian;
  Barfoot eq. 7.86; Sola et al. eq. 143) — validated FIRST in isolation
  against central-FD of this module's own `director_update()` (worst
  relative error ~7e-10 across rotation magnitudes 0-170°, random axes)
  before being trusted in the full element. Chained through membrane
  (standard Total-Lagrangian B-matrix, no director dependence), curvature
  (`Bb`, unchanged, times the new `d(dof_bs)/du`), and shear (the MITC
  tying-point covariant strain's dependence on each node's director) to
  build a genuinely analytic `internal_force()`, replacing the original
  double-FD version (kept as `_internal_force_fd()`, the oracle this was
  checked against). Validated: ~1e-8–1e-11 relative agreement with the
  oracle at generic states; EXACTLY zero (not just small) at `u=0`,
  cleaner than the oracle's own ~7e-4 FD noise floor there; and,
  re-confirmed directly rather than assumed to carry over, the decisive
  rigid-rotation claim holds for the analytic force too (max |f|
  excluding the 4 (intentionally non-invariant) drilling DOFs is ~9e-8
  across 0.5-90°, all axis types). `tangent_stiffness()` became a
  single-level real-FD Jacobian of this analytic force (dropping the
  nested-FD structure entirely) — measured ~26x faster end to end on
  this element (2.56s → 0.098s per element per Newton iteration). This
  is what unblocked registering `Shell4Director` in `ELEMENT_REGISTRY`
  and running the first real mesh-converged elastica check: `nx=20`
  (54s) gives 103.9% foreshortening recovery and 1.7% tip-deflection
  error against the exact elastica, closing the "mesh convergence not
  established at scale" limitation the original Phase 3 left open. Full
  existing suite (98 Shell4Director-specific tests, plus the broader
  shell/beam suite) re-run and passing unchanged, since every existing
  test calls the public `internal_force()`/`tangent_stiffness()` API and
  needed no test-file changes at all.

## Sources

- `fea_engine/src/fea_engine/elements/shells.py` — `Shell4MITC`,
  `Shell4MITCCorotational`; the "Design history" class-level comment
  (dead ends 1–8) and "BUILDING BLOCK C" comment are the primary
  source for Section 4.3/4.5.
- `fea_engine/src/fea_engine/elements/shells_director.py` —
  `Shell4Director` and its own extensive per-method docstrings
  (Section 1.3/4.6's primary source).
- `fea_engine/src/fea_engine/elements/beams3d.py` — `Beam3DCorotational`.
- `fea_engine/tests/test_shell_corotational*.py`,
  `test_shell4director*.py`, `test_shell_director_kinematics.py`.
- Dvorkin, E.N., Bathe, K.J., "A continuum mechanics based four-node
  shell element for general non-linear analysis," Engineering
  Computations 1, 77-88 (1984).
- Bathe, K.J., Dvorkin, E.N., "A four-node plate bending element based
  on Mindlin/Reissner plate theory and a mixed interpolation," IJNME
  21, 367-383 (1985).
- Grange, S. & Bertrand, D., "Co-rotational 3D shell element using
  quaternion algebra to account for large rotations: Static and
  dynamic applications," Forces in Mechanics 20 (2025) 100315.
- Jun, H., Yoon, K., Lee, P.-S. & Bathe, K.-J., "The MITC3+ shell
  element enriched in membrane displacements by interpolation
  covers," CMAME 337 (2018) 458-480.
- Kim, J.H. & Kim, Y.H., "A three-node C0 ANS element for
  geometrically non-linear structural analysis," CMAME 191 (2002)
  4035-4059.
- Abaqus Theory Guide, Section 3.6.5, "Finite-strain shell element
  formulation" (S4R/S3R/S4) — saved locally as `docs/Finite-strain
  shell element formulation.pdf`.
- Barfoot, T.D., *State Estimation for Robotics*, Cambridge University
  Press (2017), eq. 7.86 — the SO(3) right-Jacobian identity
  `Shell4Director`'s analytic tangent (Section 4.6) is built on.
- Sola, J., Deray, J. & Atchuthan, D., "A micro Lie theory for state
  estimation in robotics," arXiv:1812.01537 (2018), eq. 143 — same
  identity, independent source used to cross-check the formula before
  trusting it.
- Ahmad, S., Irons, B.M., Zienkiewicz, O.C. (1970); Ramm, E. (1977);
  Simo, J.C. & Fox, D.D. (1989) — the degenerated/director-based shell
  lineage `Shell4Director` follows.
- Crisfield, M.A., *Non-linear Finite Element Analysis of Solids and
  Structures*, Vol. 2 (1997) — co-rotational shell formulations.
- Battini, J-M. & Pacoste, C. — co-rotational shell/beam elements with
  consistent tangent stiffness.
