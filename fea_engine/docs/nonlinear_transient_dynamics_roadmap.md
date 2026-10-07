# Nonlinear (Newmark-Newton) transient dynamics for `fea_engine` -- roadmap

Status: **implemented and validated.** `nonlinear_solver.solve_nonlinear_transient()`
exists exactly as designed in Section 2 below, and is tested
(`tests/test_nonlinear_transient.py`: 4 tests) against all three checks
in Section 4's validation plan -- linear-limit regression against
`solve_transient_implicit()` (agreement to ~1e-9 relative, effectively
machine precision), quasi-static-limit agreement against
`solve_nonlinear_static()` (<0.1% relative, on the SAME clamped-free
`Beam2DCorotational` chain `test_nonlinear_beam.py` uses), and
undamped free-vibration energy conservation (<0.1% drift over 5
periods). Synced to the canonical `computation-suite/fea_engine/`
location; full suite passing (48/49 collected tests -- the one
exclusion, `test_geometry_engine.py`, fails only on a missing
`libGLU.so.1` system library in this sandbox, unrelated to this work).
Companion to `rom_engine/docs/nonlinear_surrogate_rom_roadmap.md`
(Section 7 there describes why `rom_engine`'s nonlinear-ROM validation
plan needs this -- this is that roadmap's Step 3/Phase 3, now unblocking
Phases 4-5 there).

**2026-09-10 update (Wave 6, `docs/consolidated_future_roadmap.md`
items 31/32/34):** both Section 3 non-goals below are now implemented
-- `nonlinear_solver.solve_transient_explicit_nonlinear()` (item 31,
central-difference explicit, no Newton loop/tangent stiffness at all,
tested against the linear `FESystem.solve_transient_explicit()` to
near machine precision, `tests/test_explicit_nonlinear_transient.py`)
and `nonlinear_solver.solve_transient_displacement_control()` (item 34
-- displacement control, not literal arc-length; see that function's
own docstring for why the transient case's Newmark mass term regularizes
away the singular-Jacobian failure mode that motivates arc-length
continuation in the STATIC case specifically, `tests/
test_transient_displacement_control.py`, validated against the same
von Mises truss closed-form snap-through curve `test_nonlinear.py`'s
own static displacement-control check uses). Item 32 (a cheap local
per-element CFL estimate, `FESystem.critical_timestep_local()`) and
item 33 (explicit dynamics validated end-to-end on `Shell4MITCCorotational`,
`tests/test_explicit_shell_dynamics.py`) were bundled into the same
wave -- see `docs/consolidated_future_roadmap.md`'s own Wave 6 table
for the full item list and status. The actual flat-beam reference-case
dynamic/NNM run (Section 4's last bullet) remains future work, not
required to unblock `rom_engine`'s own next phases.

**Reference case, not the spec.** Prompted by wanting to reproduce He et
al. (2023)'s flat-beam dynamic/NNM results (Figs. 7, 9 of "A novel
geometric nonlinear reduced order modeling method using multi-fidelity
surrogate for real-time structural analysis"), which need a genuinely
nonlinear "nonlinear FE" dynamic ground-truth curve. But the gap this
fills -- `fea_engine` has no nonlinear transient solver at all, for any
element or geometry -- is general; nothing below is beam-specific.

## 1. Confirmed gap (read directly from `solver.py`, not inferred)

`FESystem` has two transient integrators today:

- `solve_transient_implicit()` -- Newmark-beta. Factors
  `Keff = Kff + a0c*Mff + a1c*Cff` **once**, before the time loop, and
  reuses `Keff_inv` unchanged at every step. Correct for a linear,
  time-invariant system; silently wrong the moment `Kff` should depend
  on the current displacement.
- `solve_transient_explicit()` -- central difference on a lumped mass.
  Also uses a single, fixed `Kff` in its acceleration update, same
  limitation.

Meanwhile `nonlinear_solver.py` has four entry points
(`solve_contact_lagrange_static`, `solve_nonlinear_static`,
`solve_nonlinear_displacement_control`, `solve_nonlinear_arc_length`) --
**all four are static**. There is currently no path in `fea_engine` to
get a time-accurate nonlinear dynamic response for ANY element type
(beam, truss, plate, solid, contact) -- not a beam-specific gap, a
transient-dynamics-module gap.

## 2. Proposed addition

One function, following `solve_nonlinear_static`'s existing calling
convention (same `fesystem, mat` signature style, same docstring
conventions) and reusing `FESystem`'s existing
`assemble_internal_force()`/`assemble_tangent_stiffness()` (already
used by every static nonlinear solver -- no new assembly code needed,
only a new driver loop):

```python
# nonlinear_solver.py

def solve_nonlinear_transient(fesystem, mat, load, T_total, dt,
                               beta=0.25, gamma=0.5,
                               u0=None, v0=None,
                               tol=1e-8, max_iter=30, verbose=False):
    """Newmark-beta implicit time integration for a GEOMETRICALLY (or
    materially) NONLINEAR structure: at every step, predicts
    d/v/a exactly as solve_transient_implicit() does, then runs
    Newton-Raphson to convergence using the CURRENT tangent stiffness
    (assemble_tangent_stiffness(u_current, mat)) and the CURRENT
    internal force (assemble_internal_force(u_current, mat)) --
    i.e. the effective residual at each Newton iteration is

        R = M @ a_pred(d) + C @ v_pred(d) + F_int(d) - F_ext(t+dt)

    with a_pred(d)/v_pred(d) the standard Newmark relations in terms of
    the unknown d, differentiated to build the effective tangent
        K_eff = Kt(d) + a0c*M + a1c*C
    and re-factored EVERY Newton iteration (unlike
    solve_transient_implicit's single factorization) -- this is the
    entire fix; everything else (predictor formulas, a0c..a7c
    constants) is identical to solve_transient_implicit's own, so the
    two solvers should agree to machine precision on a LINEAR mat/element
    (a required regression test, see Section 4).

    `load` takes the same TimeHistoryLoad-like interface
    solve_transient_implicit() already uses (`force_at(t, n_dof, npn)`),
    so existing load definitions in loads.py need no changes.

    Returns (t, U_hist), same shape/convention as
    solve_transient_implicit(), so downstream code (plotting, ROM
    training-data extraction) doesn't need to special-case which
    solver produced a given result."""
```

Design choices, and why they match existing `fea_engine` conventions
rather than inventing new ones:

- **Newton loop reuses `solve_nonlinear_static`'s own convergence
  convention** (relative-to-load-norm tolerance, `max_iter`, `verbose`)
  instead of a new one -- a caller who already knows how to tune
  `solve_nonlinear_static`'s `tol` doesn't need to relearn anything
  here.
- **No new element API.** Every element already exposes
  `internal_force()`/`tangent_stiffness()` (used by the four static
  nonlinear solvers); this addition is pure driver-loop logic in
  `nonlinear_solver.py`, consuming `FESystem.assemble_internal_force`/
  `assemble_tangent_stiffness` exactly as `solve_nonlinear_static`
  does. No element, no matter how exotic its nonlinearity
  (`Hex8PlasticJ2`, `Tet4NeoHookean`, `GapContactCurvedFriction`), needs
  any change to become usable in nonlinear transient analysis.
- **`beta`/`gamma` defaults match `solve_transient_implicit`'s own**
  (0.25/0.5, average-acceleration/unconditionally-stable-for-linear-
  systems) for direct comparability; nonlinear unconditional stability
  is not guaranteed in general (true of every implicit nonlinear Newmark
  scheme, not a defect of this one), so the docstring should say so
  rather than imply a stronger guarantee than the linear case actually
  has.

## 3. Non-goals (kept out of THIS addition, deliberately)

- ~~**No explicit nonlinear scheme.**~~ **Implemented 2026-09-10** as
  `solve_transient_explicit_nonlinear()` (Wave 6 item 31) -- exactly the
  formula anticipated here, `a = (F_ext - F_int(d) - C@v) / m_diag`, no
  Newton loop or tangent stiffness assembly at all. `solve_transient_
  explicit`'s speed advantage (never factoring a matrix) carries over
  unchanged; validated against the linear `FESystem.solve_transient_
  explicit()` to near machine precision on a linear element/material,
  the same cross-check `solve_nonlinear_transient()` itself uses against
  `solve_transient_implicit()`.
- ~~**No arc-length / displacement-control transient variant.**~~
  **Implemented 2026-09-10** as `solve_transient_displacement_control()`
  (Wave 6 item 34) -- displacement control, not literal arc-length; see
  that function's own docstring for why a transient problem's Newmark
  effective stiffness (K_T + a0c*M + a1c*C) stays positive definite
  straight through a STATIC limit point, so the singular-Jacobian
  failure mode that motivates arc-length continuation in the static
  case does not transfer to the dynamic one the same way -- a genuine
  "arc-length in time" is left for a future item if an actual use case
  ever needs it. Validated against the von Mises truss closed-form
  snap-through curve `test_nonlinear.py`'s own static displacement-
  control check already uses, both in the quasi-static limit (heavy-
  but-not-overdamped, near-critical damping tracks the static P(delta)
  curve to <1%) and in a genuinely fast dynamic traverse through full
  inversion.

## 4. Validation plan

- **Linear-limit regression** (the cheapest, highest-value check): run
  the SAME linear problem (e.g. the existing damped-cantilever fixture
  used by `frequency.py`'s `rom_engine` tests) through both
  `solve_transient_implicit` and `solve_nonlinear_transient` with a
  `mat` whose `tangent_stiffness()`/`internal_force()` reduce to the
  linear `K@u` case -- results should agree to machine precision (a few
  Newton iterations converging immediately since the residual is
  already linear). This is the single most important test: it PROVES
  the nonlinear driver's bookkeeping (Newmark constants, effective
  stiffness assembly) is correct independent of any nonlinear element
  behavior.
- **Existing static nonlinear fixtures, driven quasi-statically**:
  reuse `test_nonlinear_beam.py`'s `Beam2DCorotational` benchmark and
  `test_nonlinear.py`'s von Mises truss, applying their same loads as a
  very slow ramp (`T_total` long relative to the structure's fundamental
  period) through `solve_nonlinear_transient` and checking the final
  state converges to the already-validated `solve_nonlinear_static`
  answer -- confirms the two nonlinear solvers agree in the
  quasi-static limit.
- **Energy consistency**: for an undamped, unforced free-vibration
  case, total mechanical energy (kinetic + internal strain energy, the
  latter available for any element with a `strain_energy()`-like
  evaluation or computed as `integral of F_int . du` along the recorded
  history) should stay constant to within time-integration error over a
  reasonably short window -- the same kind of check
  `test_hyperelastic.py` already uses (external-work =
  stored-strain-energy) adapted to a free-vibration, no-external-work
  setting.
- **The actual reference case**: the flat-beam model
  (`Beam2DCorotational`, 40 elements, clamped-clamped), driven at the
  paper's own two initial conditions
  (`F_nl(0)=-2068 Pa`, `x_nl(0)=0.5*th*phi_1/max(phi_1)`), consumed by
  `rom_engine`'s validation plan (Section 9 there) as the "no surrogate"
  ground truth Figs. 7/9 compare against.

## Sources

- `fea_engine/src/fea_engine/solver.py` (`solve_transient_implicit`,
  `solve_transient_explicit`, read directly to confirm Section 1's gap).
- `fea_engine/src/fea_engine/nonlinear_solver.py` (`solve_nonlinear_static`,
  whose convergence/Newton-loop convention this addition reuses).
- `rom_engine/docs/nonlinear_surrogate_rom_roadmap.md` (this addition's
  Section 7 -- the consuming side of this interface).
- He, X. et al. (2023), *Struct Multidisc Optim* 66:233 -- Figs. 7, 9
  (the reference case's dynamic/NNM ground-truth curves this unblocks).
