# computation-suite Module Review — Delta since 2026-08-31

**Date:** 2026-09-04
**Scope:** Everything added to `fea_engine` and `rom_engine` since the last review (`module_review_2026-08-31.md`), re-run at the canonical location `computation-suite/{fea_engine,rom_engine}`. This is a delta review, not a full re-audit — findings from the Aug-31 report are assumed to still hold except where noted below.

## Executive summary

`fea_engine` gained two pieces of work since Aug-31: a generalized mesh-grading system (`grading.py`, `build_mesh.py`, plus `mesh.py`/`gmsh_engine.py` additions) and the generic (`m≥1`) Koiter-Newton continuation driver (`solve_nonlinear_koiter_newton_generic()`). `rom_engine` had zero changes — confirmed by mtime check, not just assumed.

| | `fea_engine` delta | `rom_engine` delta |
|---|---|---|
| Tests (was → now) | 90 (89 confirmed) → **210 collected, 208 passed, 2 pre-existing xfail** | 251 → **251, unchanged** |
| New source content | `grading.py`, `build_mesh.py`, `mesh.py`/`gmsh_engine.py` additions; `solve_nonlinear_koiter_newton_generic()` + 3 private helpers in `nonlinear_solver.py` | None |
| Doc-vs-code consistency | 4 findings, all cosmetic/narrative (stale counts, an arithmetic slip, an indexing gap) — no code defects | 1 pre-existing (not new) cosmetic finding, otherwise clean |
| Sync status | `fea_engine_dev2`: byte-identical for every file this delta touches; diverges only on unrelated concurrent "Dead End 7" shell work. `fea_engine_dev`: unchanged stale Aug-27 snapshot | All three rom_engine mirrors unchanged from their prior (already-stale, already-known) state |

Also of note: the Aug-31 review's own Finding 2 — one `fea_engine` test (`test_large_deflection_is_in_the_right_regime`) that couldn't be confirmed within that review's runtime budget — is now retired. It runs cleanly in 33s and passes.

Total findings this pass: five, all low-severity and documentation-only (four in `fea_engine`, one pre-existing item newly noticed in `rom_engine`). No test failures, no regressions, no sync failures, no missing functionality anywhere in either package.

---

## 1. `fea_engine` delta since 2026-08-31

**Scope:** Two changes since the 2026-08-31 review — both confirmed present and largely, but not perfectly, documented:

1. A generalized mesh-grading system: `grading.py`, `build_mesh.py`, `mesh.py` additions (`hole_in_rectangle_mesh`, `hole_in_rectangle_mesh_graded`, `_reflect_mesh`, `_translate_mesh`, `Mesh.check_grading()`), `geometry/gmsh_engine.py`'s `generate_2d_plate_with_hole_graded`, `docs/generalized_mesh_grading_roadmap.md`, and 4 new test files.
2. A generic (`m≥1`) Koiter-Newton driver: `solve_nonlinear_koiter_newton_generic()` plus `_directional_QC`/`_two_direction_QC`/`_generic_cbar_term` in `nonlinear_solver.py`, `tests/test_koiter_newton_generic.py` (4 tests), `docs/general_purpose_extensions_roadmap.md` Section 11, and matching README updates.

### 1.1 What was added, confirmed present

Both pieces of work are real, complete, and pass every test:

- `src/fea_engine/grading.py` (324 lines): pure sizing-function math — `graded_partition`/`growth_ratio_of_partition`/`grade_for_growth_ratio`, the `Hole`/`Fillet`/`EdgeBias`/`Notch` feature dataclasses (`Notch.__post_init__` correctly raises `MeshGradingError`, verified by reading the source), `plan()`/`GradingPlan`, and the Gmsh-Threshold-field-matching `threshold_field_size`/`dist_max_for_growth_ratio`.
- `src/fea_engine/build_mesh.py` (158 lines): the `build_mesh()` dispatcher, `RectangleWithHole`, `ROTATIONAL_DOF_ELEMENTS = (Quad4MindlinPlate, Shell4MITC, Shell4MITCCorotational)` — verified this is an explicit tuple, not an inferred rule, matching its own comment's rationale (beam elements excluded deliberately).
- `mesh.py`: `hole_in_rectangle_mesh`, `hole_in_rectangle_mesh_graded`, `_reflect_mesh`, `_translate_mesh`, `Mesh.check_grading()` all present and match their documented behavior (2-D-only, raises `NotImplementedError` for 3-D, edge-adjacency-based size-ratio diagnostic).
- `geometry/gmsh_engine.py`: `generate_2d_plate_with_hole_graded` present, matches its docstring's description of Distance/Threshold/CharacteristicLengthFromCurvature field wiring.
- `nonlinear_solver.py`: `solve_nonlinear_koiter_newton_generic()` plus the three private helpers all present. `_generic_cbar_term` was spot-checked line-by-line against the formula its own docstring and the roadmap doc Section 11 both quote (`Cbar[p;i,j,k] = u_p.C(u_i,u_j,u_k) - (2/3)*[...]`) — matches exactly.
- `tests/test_koiter_newton_generic.py`: exactly the 4 tests both README and the roadmap doc describe — the pitchfork-bifurcation synthetic benchmark's closed-form math was read in full and matches the docstring's claims.

### 1.2 Test results

Full suite run in batches (single `pytest tests/` calls exceeded the available per-call runtime, so the run was split further than originally planned):

| Batch | Files | Result |
|---|---|---|
| Grading | `test_grading.py`, `test_mesh_grading_generalized.py`, `test_build_mesh_dispatcher.py`, `test_gmsh_field_grading.py` | **74/74 passed** in 1.29s |
| Shell corotational (core) | `test_shell_corotational.py` | **18/18 passed** in 16.6s |
| Shell corotational (rest) | `test_shell_corotational_reference_frame.py`, `test_shell_corotational_mixed_formulation.py`, `test_shell.py` | **12 passed, 2 xfailed** in 127.7s — pre-existing, `strict=True`, documented (Dead End 7's mixed-formulation approach; unrelated to this delta) |
| Shell corotational (elastica) | `test_shell_corotational_elastica.py` | **4/4 passed** in 88.3s |
| Gmsh-dependent | `test_step_import.py`, `test_gmsh_node_order.py`, `test_quadratic_extraction.py`, `test_geometry_engine.py`, `test_mixed_elements.py`, `test_geometry.py` | **38/38 passed** in 18.0s (with the standard `libGLU.so.1` sandbox workaround) |
| Everything else (19 files) | linear buckling, beam3d, plasticity, transient, both Koiter-Newton test files, iter_state, hyperelastic, higher-order elements, sparse assembly, arc-length, simplex elements, contact, etc. | **57/57 passed** in 20.4s |
| Tet10 (Aug-31's unconfirmed test) | `test_tet10_geometric_nonlinear.py` | **5/5 passed** in 33.2s |

**Total: 208 passed, 2 xfailed (pre-existing/unrelated), 0 failed, 210 collected.** `test_large_deflection_is_in_the_right_regime` — the one test the Aug-31 review couldn't confirm within its own runtime budget — completed cleanly this time (33s for the whole file), retiring that finding.

### 1.3 Documentation-vs-code consistency

Mostly clean, but four genuine textual inconsistencies were found — all narrative, none affecting correctness:

**Finding A — README.md's `nonlinear_solver.py` driver count is stale.** The line just above the driver table still reads "Six drivers — five STATIC ... plus one dynamic," but the table itself now lists **seven** rows, and the line immediately below the table already correctly says "All seven call..." Adding `solve_nonlinear_koiter_newton_generic()` (Module 24) as a sixth static driver pushed the total from six to seven; the downstream line was updated but the header two lines above the table was missed. Should read "Seven drivers — six STATIC ... plus one dynamic."

**Finding B — `docs/generalized_mesh_grading_roadmap.md`'s own test-total arithmetic doesn't add up.** It states "**71** new tests across the four phases (47 + 15 + 3 + 9...)" — but 47+15+3+9 = **74**, matching the actual measured total (the grading batch above ran exactly 74/74). Self-contained arithmetic slip in the doc's own stated sum.

**Finding C (minor, low-confidence) — the same doc's Phase 3 file-list count reads "40" for a list that sums to 42 today.** Plausibly explained by unrelated concurrent test additions to those files after this doc's date stamp, rather than the doc being wrong when written — noted for completeness, not treated as a real defect.

**Finding D (minor) — `docs/general_purpose_extensions_roadmap.md`'s top-of-file Status summary never mentions Phase 10.** The header enumerates Phases 1-9 (including 8 and 9 by name in the explanatory paragraph) but never Phase 10 / Module 24, even though Section 11 later in the same document fully and correctly documents it. Purely an indexing gap at the top of the file — the body itself, and every cross-reference from README/the module docstring into Section 11, is accurate.

Everything else checked was clean: the `nonlinear_solver.py` module docstring's Module 24 entry matches README's table row and the actual function signature exactly; the roadmap doc's correctness-review claims, the Eq.7 general formula, and the "why a new function, not a revision" reasoning were all spot-checked against the actual code and test file and match precisely; the grading roadmap's Phase 1-4 per-file test counts (47/15/3/9) match the actual collected counts exactly; no broken internal doc links were found.

### 1.4 `__init__.py` export consistency

- **`grading.py`**: fully flat-exported (`Hole`, `Fillet`, `EdgeBias`, `Notch`, `GradingPlan`, `MeshGradingError`, `plan`, `graded_partition`, `growth_ratio_of_partition`, `grade_for_growth_ratio`, `threshold_field_size`, `dist_max_for_growth_ratio`, `DEFAULT_GROWTH_RATIO` all present in both the import list and `__all__`). Only two clearly-internal names are left un-exported, both reasonable omissions.
- **`build_mesh.py`**: `RectangleWithHole` and `ROTATIONAL_DOF_ELEMENTS` are flat-exported; `build_mesh()` itself is deliberately not, per an explicit `__init__.py` comment explaining the name would collide with the submodule itself — documented identically in three places (the comment, README, and the roadmap doc), all consistent.
- This is the same "not flat-exported, access via submodule" convention `nonlinear_solver.py`'s driver functions already use, applied correctly and only where an actual collision exists — not an inconsistency.

### 1.5 Sync status

`diff -rq` (excluding caches/egg-info) between canonical and `outputs/fea_engine_dev2`:

```
Files .../docs/geometric_nonlinear_shell_roadmap.md ... differ
Files .../src/fea_engine/elements/shells.py ... differ
Files .../src/fea_engine/solver.py ... differ
Only in .../fea_engine/tests: test_iter_state.py, test_koiter_newton.py,
    test_shell_corotational_mixed_formulation.py, test_static_koiter_newton.py
Only in .../fea_engine: validate_phase_b_coupling.py
```

Every file this delta actually touches (`grading.py`, `build_mesh.py`, `nonlinear_solver.py`, all 5 new/changed test files, `docs/generalized_mesh_grading_roadmap.md`) was individually diffed and confirmed **byte-identical** between canonical and `dev2`. The differences that do exist belong to a different, concurrent piece of work — the "Dead End 7" mixed-formulation shell investigation — and predate that work's own 2026-09-04 timestamp in `dev2`'s snapshot; this is the same benign "stale snapshot from a different session" pattern the Aug-31 review's own Finding 1 already established, not a new defect, and it doesn't touch this delta's scope. `fea_engine_dev` (the older mirror) remains an unchanged Aug-27/28 snapshot, still missing both `grading.py` and `build_mesh.py` entirely — expected, not new.

### 1.6 `docs/geometric_nonlinear_shell_roadmap.md` Status check

Read in full. Its status line correctly reflects the latest state ("Dead end 7, tried and REJECTED 2026-09-04" is the newest entry), and its claims were cross-checked directly against the actual test file and source comments — all confirmed accurate. Nothing in this delta's own scope (grading, generic Koiter-Newton) touches this file or its subject matter.

---

## 2. `rom_engine` delta since 2026-08-31

**Confirmed: nothing in `rom_engine`'s source, tests, or docs changed since the 2026-08-31 review.** No new modules, no edits to any `.py` or `.md` file under `src/rom_engine/`, `tests/`, `docs/`, or `README.md`.

### 2.1 Change check

`find src/ tests/ docs/ README.md -newermt "2026-08-31 23:59:59" -type f` returned only stray `.pyc` bytecode cache files from an earlier pytest run in this session — no source, test, or doc file has an mtime after 2026-08-31. Verified, not assumed.

### 2.2 Test results

```
251 passed, 11 warnings in 40.36s
```

Identical to the Aug-31 figure — same count, same warning set (a deliberate `OptimalHankelNormROM` unreliable-regime warning plus unrelated `np.trapz` deprecation notices). No `gmsh`/`libGLU` workaround needed; this package doesn't depend on Gmsh.

### 2.3 Documentation-vs-code consistency

README's 22-row module table still matches the 22 actual source files; `__all__`'s 56 exported names spot-checked (10 entries) with no mismatches; the "All 251 tests pass" line in README still matches reality.

One pre-existing (not new) item was noticed on this pass: `docs/frequency_domain_rom_roadmap.md`'s top-of-file Status line still reads "design document, no code written yet," while the same document's own body says SOAR is "DONE"/"Implemented" (`soar.py` exists and is fully tested). This predates the Aug-31 review's own cutoff (the file's mtime falls inside that review's window) and simply wasn't flagged then — cosmetic only, doesn't affect correctness.

### 2.4 Sync status

- `outputs/rom_engine_dev`: only difference from canonical is a pre-existing `.coverage` artifact (dated 2026-08-26, predates even the Aug-27 baseline) — not source, benign, same as the Aug-31 finding.
- `outputs/rom_engine_dev2`: last touched 2026-08-26, only 5 modules present (missing even `scm.py`) — a stale pre-baseline snapshot, not mentioned in the Aug-31 review (evidently already stale then too).
- `outputs/rom_engine_dev3`: last touched 2026-08-26, 6 modules present (matches the exact Aug-27 baseline) — also stale, same category.

Neither `dev2` nor `dev3` regressed since Aug-31 — they were already this stale. Worth a light procedural note only if a future session tries to resume work from one of them assuming currency.

### 2.5 Roadmap doc Status-section skim

`classical_mor_roadmap.md`, `nonlinear_surrogate_rom_roadmap.md`, and `loewner_modal_identification_roadmap.md` all have internally consistent Status sections. `frequency_domain_rom_roadmap.md` has the stale header noted in 2.3 above — the only inconsistency found, and it's pre-existing.

---

## Findings summary

| # | Package | Location | Severity | Nature |
|---|---|---|---|---|
| A | fea_engine | `README.md` (driver-count header) | Low (cosmetic) | "Six drivers" header contradicts its own 7-row table and the "All seven" line just below it |
| B | fea_engine | `docs/generalized_mesh_grading_roadmap.md` | Low (cosmetic) | "71 new tests" headline figure; the stated 47+15+3+9 sum is actually 74, matching reality |
| C | fea_engine | `docs/generalized_mesh_grading_roadmap.md` | Very low (unconfirmed) | "40" Gmsh-dependent tests for a file list that sums to 42 today — plausibly a later, unrelated addition rather than an error when written |
| D | fea_engine | `docs/general_purpose_extensions_roadmap.md` (top summary) | Low (indexing gap) | Top-of-file Status summary never mentions Phase 10/Module 24, even though it's fully documented in Section 11 |
| E | rom_engine | `docs/frequency_domain_rom_roadmap.md` (top Status line) | Low (cosmetic, pre-existing) | Says "no code written yet" while the document's own body confirms SOAR is implemented and tested; predates this review window |

None of these five findings are code defects, test failures, missing functionality, or sync failures — all are narrative/summary slips in documentation that is otherwise detailed and accurate at the point of actual technical description. No regressions were found in either package. The Aug-31 review's own Finding 2 (one unconfirmed `fea_engine` test) is resolved — it now passes cleanly and quickly.

## Recommended fixes (all trivial, cosmetic, no code changes needed)

- fea_engine `README.md`: change "Six drivers" to "Seven drivers" in the `nonlinear_solver.py` section header.
- fea_engine `docs/generalized_mesh_grading_roadmap.md`: correct "71 new tests" to "74 new tests" (the parenthetical sum is already right).
- fea_engine `docs/general_purpose_extensions_roadmap.md`: add Phase 10/Module 24 to the top-of-file Status summary alongside Phases 8-9.
- rom_engine `docs/frequency_domain_rom_roadmap.md`: update the top-of-file Status line to reflect that SOAR (Phase 5) is implemented, matching the rest of the document.
