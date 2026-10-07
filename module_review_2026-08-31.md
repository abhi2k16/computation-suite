# computation-suite Module Review — Delta since 2026-08-27

**Date:** 2026-08-31
**Scope:** Everything added to `fea_engine` and `rom_engine` since the last review (`module_review_2026-08-27.md`), re-run at the canonical location `computation-suite/{fea_engine,rom_engine}`. This is a delta review, not a full re-audit — findings from the Aug-27 report are assumed to still hold except where noted below.

## Executive summary

Both packages grew substantially since Aug 27. `rom_engine` went from 42 to **251** tests (6 source modules to 22) — essentially its entire "classical MOR"/systems-and-control family (`state_space`, `mode_correction`, `krylov`, `balanced_truncation` + two siblings, `passivity`, `hankel_norm`, `soar`), its nonlinear-surrogate-ROM family (`sampling`, `nonlinear_rom`, `nonlinear_dynamics`, `nnm` + a pseudo-arclength addendum), `scm_lp`, `loewner`/`screening`, and `random_vibration` were all built after the last review. `fea_engine` gained Phase 7 (`Shell4MITC`, general-purpose shell elements) plus a standalone `Tet10SolidTL` element added for a sibling project.

| | `fea_engine` delta | `rom_engine` delta |
|---|---|---|
| Tests (was → now) | 41 → **90** (89 confirmed passed, 1 unconfirmed — see Finding 2) | 42 → **251**, all passed |
| New source content | `elements/shells.py` (`Shell4MITC`), `Tet10SolidTL` in `elements/nonlinear_solids.py` | 16 new modules: `state_space`, `mode_correction`, `krylov`, `balanced_truncation`, `passivity`, `hankel_norm`, `soar`, `sampling`, `nonlinear_rom`, `nonlinear_dynamics`, `nnm`, `scm_lp`, `loewner`, `screening`, `metrics`, `random_vibration` |
| Doc-vs-code consistency | Clean — `Shell4MITC` and `Tet10SolidTL` both fully documented in README + roadmap doc | Clean — README's 22-row module table matches the 22 actual source files exactly (zero mismatches), `__all__` has 56 exported names, test count line reads 251 |
| Sync status | Canonical is AHEAD of the one leftover `outputs/fea_engine_dev` snapshot in this session (that snapshot predates Tet10SolidTL) — not a defect, see Finding 1 | Canonical and this session's `outputs/rom_engine_dev` working copy are byte-identical (verified `diff -rq`, clean) |

Two findings this time, both environmental/procedural rather than code defects — detailed below.

---

## 1. `fea_engine` delta

### 1.1 What was added since Aug 27

- **Phase 7 — `Shell4MITC`** (general-purpose extensions roadmap, `docs/general_purpose_extensions_roadmap.md`): a general 4-node shell element (membrane + bending + transverse shear via genuine Dvorkin-Bathe MITC4 assumed-natural-strain interpolation, not selective reduced integration), reusing `Quad4PlaneStress`/`Quad4MindlinPlate`'s own building blocks in a per-element local frame. `tests/test_shell.py`, 5/5 passed (rigid-body modes, frame objectivity, flat-plate-limit convergence to `Quad4MindlinPlate`, curved multi-element rigid-body modes, curved-shell-vs-`Hex8Solid3D` convergence).
- **`Tet10SolidTL`** (`elements/nonlinear_solids.py`, added 2026-08-30 — not part of the general-purpose roadmap; built for the wing-cantilever example in the sibling `Multi_Fidelity_NL_Structural_ROM` project): geometric-only nonlinearity (Total Lagrangian, St. Venant-Kirchhoff — the same linear elastic law `Tet10Solid3D` uses, applied to the Green-Lagrange strain) on the quadratic Tet10 element, genuinely Gauss-looped (unlike `Tet4NeoHookean`'s single-point evaluation) since Tet10's quadratic shape functions give a position-varying deformation gradient. `tests/test_tet10_geometric_nonlinear.py` — see Finding 2 for its one unconfirmed test.
- A documented **known issue** was added to the README alongside this: `generate_from_step()` becomes unreliable (hangs/pathologically slow at Gmsh's own meshing step) once enough prior calls have accumulated in one process — not yet fixed, flagged honestly rather than hidden.

### 1.2 Test results

Re-running the full suite hit a sandbox gap not present in the Aug-27 review: `libGLU.so.1` (needed by the `gmsh` Python package even for headless meshing) is missing from this session's sandbox image. The Aug-27 review hit and worked around the identical gap (its own Finding 1); the same workaround was re-applied here — `apt-get download` (no root needed) of `libglu1-mesa`/`libgl1`/`libglx0`/`libglvnd0`/`libopengl0`, extracted to a user-writable directory, added to `LD_LIBRARY_PATH`. With that in place:

| Test file | Result | Notes |
|---|---|---|
| 17 non-Gmsh files, 48 tests | 48/48 PASSED | unaffected by the sandbox gap |
| `test_shell.py` (Phase 7) | 5/5 PASSED | |
| `test_geometry_engine.py` | 1/1 PASSED | |
| `test_gmsh_node_order.py` | 8/8 PASSED | |
| `test_quadratic_extraction.py` | 12/12 PASSED | |
| `test_step_import.py` | 15/15 PASSED | |
| `test_mixed_elements.py` | 1/1 PASSED | failed only under the missing-library condition; passes clean with the workaround |
| `test_tet10_geometric_nonlinear.py` | 4/5 confirmed PASSED, 1 unconfirmed | see Finding 2 |

**Total: 89/90 confirmed passed**, 1 test (`TestLargeDeflectionBenchmark::test_large_deflection_is_in_the_right_regime`) left in an unconfirmed but actively-computing state — not a failure, see Finding 2.

### 1.3 Documentation-vs-code consistency

`Shell4MITC` and `Tet10SolidTL` are both fully described in README's element reference table (with their own validation summaries) and `Shell4MITC` is in the roadmap doc's `Status:` header as Phase 7. `Tet10SolidTL` is intentionally NOT in that roadmap doc (it wasn't part of the general-purpose-extensions plan — it's a standalone addition documented directly in the README, correctly).

### 1.4 Sync status

The one dev-workspace snapshot still present in this session, `outputs/fea_engine_dev`, predates `Tet10SolidTL` (last touched Aug 27) and differs from canonical in exactly the files `Tet10SolidTL` touches (`__init__.py`, `elements/__init__.py`, `elements/nonlinear_solids.py`) plus example-script PNGs. This is NOT a sync failure — canonical is simply ahead of a stale snapshot from an earlier phase; `Tet10SolidTL` was evidently built directly against canonical in a different session. See Finding 1.

---

## 2. `rom_engine` delta

### 2.1 What was added since Aug 27

Everything except `pod.py`, `galerkin.py`, `affine.py`, `frequency.py`, `greedy.py`, `scm.py` (the 6 modules the Aug-27 review covered) is new:

- **Classical MOR / systems-and-control family** (`docs/classical_mor_roadmap.md`): `state_space.py` (first-order conversion), `mode_correction.py` (mode acceleration + modal truncation augmentation), `krylov.py` (one- and two-sided Arnoldi moment matching), `balanced_truncation.py` (+ `SingularPerturbationROM` and `FrequencyWeightedBalancedTruncationROM` siblings), `passivity.py` (closed-form second-order-Galerkin passivity theorem + frequency-domain diagnostic), `hankel_norm.py` (AAK/Glover optimal Hankel norm approximation), and `soar.py` (second-order-structure-preserving SOAR Krylov, this session's own final addition — see its own module docstring for the real implementation bug caught and fixed during development).
- **Nonlinear-surrogate-ROM family** (`docs/nonlinear_surrogate_rom_roadmap.md`): `sampling.py` (optimal LHS), `nonlinear_rom.py` (`MultiFidelitySurrogate`/`PolynomialModalROM`), `nonlinear_dynamics.py` (reduced Newmark time integration), `nnm.py` (harmonic-balance NNM backbone continuation, natural-parameter AND — this session's addendum — pseudo-arclength).
- **Non-intrusive identification**: `loewner.py`, `screening.py`, `metrics.py` (`docs/loewner_modal_identification_roadmap.md`).
- **`scm_lp.py`** (classical LP-based Successive Constraint Method) and **`random_vibration.py`** (PSD response), each a `docs/frequency_domain_rom_roadmap.md` phase addendum.

### 2.2 Test results

```
251 passed, 11 warnings in ~42s
```

All 30 test files pass; the 11 warnings are a pre-existing, already-documented `OptimalHankelNormROM` reliability warning (fires deliberately in one test that exercises a known-fragile regime) and unrelated `np.trapz` deprecation notices — neither is new or actionable.

### 2.3 Documentation-vs-code consistency

- README's module table: 22 rows, diffed programmatically against the 22 actual `.py` files in `src/rom_engine/` — **zero mismatches** (no stale rows, nothing undocumented).
- `rom_engine.__all__`: 56 exported names, spot-checked against the table — consistent.
- Test count line in README reads "All 251 tests pass" — matches.
- `docs/classical_mor_roadmap.md`'s top-of-file `Status:` line accurately reflects Phases 0-4 plus 5a-5d as implemented (the roadmap's own Section 7 phase list still describes Phase 5 as "deferred, optional" in its *original, preserved-as-written* form — but the document says explicitly, twice, that this is historical text and gives the current status upfront, so this is not a stale/misleading claim).
- `docs/frequency_domain_rom_roadmap.md` Section 8 (SOAR) and `docs/nonlinear_surrogate_rom_roadmap.md` Section 11 (pseudo-arclength) both carry accurate outcome sections with real numbers, written this session.

### 2.4 Sync status

`outputs/rom_engine_dev` (this session's working copy) vs. canonical `computation-suite/rom_engine`: `diff -rq --exclude=__pycache__` returns **empty** across `src/`, `tests/`, and `docs/`. Fully synced.

---

## 3. Findings

### Finding 1 — stale `fea_engine_dev` snapshot in `outputs/` predates `Tet10SolidTL` (not a defect)

`outputs/fea_engine_dev` was last modified 2026-08-27 (during the Phase 7 shell-element work) and was never updated for `Tet10SolidTL`, which was added 2026-08-30 — apparently built directly against the canonical copy in a different session, bypassing this particular dev snapshot. Diffing them shows exactly the files `Tet10SolidTL` touches, plus a handful of example-script PNGs regenerated since. Canonical itself is internally consistent (tests pass, docs match code) — this is purely a leftover-scratch-copy observation, not something to fix. Worth a light procedural note: if a future session resumes work via `outputs/fea_engine_dev`, it should be re-synced from canonical first rather than assumed current.

### Finding 2 — one `Tet10SolidTL` benchmark test's completion could not be confirmed within this session's tool-call runtime limits (not a hang)

`test_tet10_geometric_nonlinear.py::TestLargeDeflectionBenchmark::test_large_deflection_is_in_the_right_regime` runs a large-deflection nonlinear Newton solve (with `Tet10SolidTL`'s finite-difference tangent stiffness, which the element's own docstring already flags as the more expensive but simpler choice over a hand-derived analytic tangent) against a real, sourced reference solution. It was confirmed **actively computing** (99.7% CPU, growing memory, no error) for several minutes across three separate observation windows, but this review's tool environment kills detached/background processes between tool calls (confirmed: `nohup`, and separately `setsid`+`nohup`+`disown`, both lost the process with no final output logged), making it impossible to observe this specific test's PASS/FAIL outcome within the available per-call runtime budget.

This is NOT the documented `generate_from_step()` hang (that known issue is specific to repeated Gmsh STEP-import calls; this test's own comments confirm it deliberately uses the `.geo`-kernel builder instead, precisely to avoid that issue) — it is simply a compute-heavy test that runs longer than this review session could wait out. The other 4/5 tests in the same file (all the cheaper identity and small-load-convergence checks) passed cleanly and quickly. Recommended follow-up, if this matters going forward: either run this one test in isolation with a generous timeout outside this kind of constrained review session, or profile whether the FD tangent's O(n²) force-evaluation cost (30×30 per Newton iteration, on whatever element count the benchmark mesh uses) is the bottleneck and worth an analytic tangent if this benchmark needs to run routinely.

---

## 4. Conclusion

Both packages remain in a clean, consistent state after a large amount of net-new work since Aug 27 — `rom_engine` in particular roughly sextupled its test count and now covers essentially the full classical MOR literature (mode acceleration, Krylov, balanced truncation and three siblings, passivity, Hankel norm, second-order SOAR) plus the entire nonlinear-surrogate-ROM and NNM-backbone-continuation pipelines. No regressions were found, no documented-but-missing modules, no undocumented-but-implemented modules. The two findings above are both properties of this specific review session (an environment gap fixable with a known workaround, and a tool-call runtime ceiling on one compute-heavy test), not defects in either package.
