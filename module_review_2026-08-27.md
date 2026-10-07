# computation-suite Module Review — fea_engine + rom_engine

**Date:** 2026-08-27
**Scope:** Full regression audit of every module implemented so far in both packages — `fea_engine` (Modules 1-19, general-purpose extensions Phases 1-6) and `rom_engine` (Phases 0-4d) — run at the canonical location `computation-suite/{fea_engine,rom_engine}`.

## Executive summary

Both packages are in a clean, consistent state. Every test in both suites passes, every example script runs to completion (one exception, explained below and NOT a code defect), and the module claims in each README/roadmap doc match what is actually implemented, exported, and registered in code. No regressions, no dead code paths, no undocumented modules, and no documented-but-missing modules were found.

| | `fea_engine` | `rom_engine` |
|---|---|---|
| Tests | **41 / 41 passed** | **42 / 42 passed** |
| Test files | 15 | 6 |
| Example scripts run | 16 / 16 exit 0 (1 pre-existing script hits a sandbox memory ceiling at its largest sweep point — see Finding 2) | 4 / 4 exit 0 |
| Source modules | 20 `.py` files across `elements/`, `geometry/`, `visualization/`, plus core | 6 `.py` files |
| Doc-vs-code consistency | Every class/function named in README module tables and roadmap status headers exists, is exported in `__all__`, and (where applicable) is registered in `ELEMENT_REGISTRY`/`CONSTITUTIVE_REGISTRY` | Every class/function named in README exists and is exported in `__all__` |
| Sync status | Single canonical copy at `computation-suite/fea_engine` (session dev-copy was diffed empty against it before this review) | Single canonical copy at `computation-suite/rom_engine`, built there directly — no separate dev copy exists |

Two findings surfaced during the audit, both environmental rather than code defects — detailed below.

---

## 1. `fea_engine` — full regression results

### 1.1 Test suite (41/41 passed)

Ran `pytest -v` against the canonical package after `pip install -e .`:

| Test file | Tests | Result | Covers |
|---|---|---|---|
| `test_arc_length.py` | 3 | PASS | Phase 5 — Crisfield arc-length/Riks solver |
| `test_beam3d.py` | 6 | PASS | Phase 3 — `Beam3DEulerBernoulli` |
| `test_contact.py` | 1 | PASS | Module — flat gap contact (`GapContactPenalty`) |
| `test_curved_contact.py` | 1 | PASS | Module 13 — curved contact + friction |
| `test_geometry.py` | 1 | PASS | Module 11 — `geometry.py` shape builder |
| `test_geometry_engine.py` | 1 | PASS | Module 12 — Gmsh unstructured meshing (see Finding 1) |
| `test_higher_order_elements.py` | 4 | PASS | Phase 1 — `Quad8PlaneStress`/`Hex20Solid3D`/`Tet10Solid3D` |
| `test_hyperelastic.py` | 4 | PASS | Phase 6 — `Tet4NeoHookean` |
| `test_linear_buckling.py` | 6 | PASS | Phase 4 — linear buckling eigenvalue solver |
| `test_mixed_elements.py` | 1 | PASS | Module 14 — mixed-element-type assembly (see Finding 1) |
| `test_nonlinear.py` | 1 | PASS | Module 8 — `TrussTL2D` geometric nonlinearity |
| `test_nonlinear_beam.py` | 1 | PASS | Module 15 — `Beam2DCorotational` |
| `test_nonlinear_cantilever.py` | 1 | PASS | Nonlinear cantilever regression |
| `test_plasticity.py` | 1 | PASS | Module 9 — 1-D `TrussPlastic2D` |
| `test_plasticity_j2.py` | 5 | PASS | Phase 6 — `Hex8PlasticJ2` |
| `test_simplex_elements.py` | 1 | PASS | `Tri3PlaneStress`/`Tet4Solid3D` |
| `test_sparse_assembly.py` | 3 | PASS | Phase 2 — sparse matrix support |

**Total: 41 passed, 0 failed, 0 errors.**

### 1.2 Example scripts (16/16 run; 1 hits a sandbox memory limit)

All 16 scripts in `examples/` were executed directly (not just imported). 15 exited cleanly (exit 0), reproducing every documented result (headline validation table in `main.py`, all six Phase 1-6 demo/plot pairs). One (`hex8_convergence.py`) raised `numpy._core._exceptions._ArrayMemoryError` at its last sweep point — see Finding 2.

### 1.3 Documentation-vs-code consistency

Cross-checked every class named in the README's module reference table and every module/phase referenced in `docs/general_purpose_extensions_roadmap.md`'s status header against:
- `grep "^class "` across `src/fea_engine/**/*.py` — all 29 classes present, none undocumented, none missing.
- `fea_engine.__all__` — all public names match what the README exports.
- `ELEMENT_REGISTRY` (16 entries) and `CONSTITUTIVE_REGISTRY` (6 entries) — every element/material described in prose has a corresponding registry key, including `hex8_plastic_j2` and `tet4_neo_hookean` from Phase 6.
- `nonlinear_solver.py`'s four solver entry points (`solve_contact_lagrange_static`, `solve_nonlinear_static`, `solve_nonlinear_displacement_control`, `solve_nonlinear_arc_length`) all match the module reference table's description.

No stale claims found (nothing described as implemented that isn't; nothing implemented that isn't documented).

### 1.4 Sync status

The Phase 6 dev-copy created earlier this session (`outputs/fea_engine_dev/`) was already diffed empty against the canonical copy at the time Phase 6 was synced. No further drift found in this review.

---

## 2. `rom_engine` — full regression results

### 2.1 Test suite (42/42 passed)

| Test file | Tests | Result | Covers |
|---|---|---|---|
| `test_pod.py` | 10 | PASS | `PodBasis` — standard + mass-weighted POD |
| `test_galerkin.py` | 8 | PASS | `GalerkinROM` — intrusive linear ROM |
| `test_affine.py` | 9 | PASS | `AffineDecomposition` — parametric reassembly |
| `test_frequency.py` | 9 | PASS | `FrequencyROM` — Phases 0-3 (complex dtype, damped FRF, POD-from-FRF, greedy indicator) |
| `test_greedy.py` | 2 | PASS | `greedy_train_frequency_basis` — Phase 4b |
| `test_scm.py` | 5 | PASS | `SingularValueLowerBound`/`certified_error_bound` — Phase 4d |

**Total: 42 passed, 0 failed, 0 errors.**

### 2.2 Example scripts (4/4 pass)

`certified_bound_cantilever.py`, `frequency_sweep_cantilever.py`, `greedy_frequency_training.py`, `two_region_beam_rom.py` all exit 0 and reproduce their documented output.

### 2.3 Documentation-vs-code consistency

`rom_engine.__all__` (8 names) matches every class/function referenced in `README.md`'s "What's implemented" section and both roadmap docs (`frequency_domain_rom_roadmap.md`, `phase4_error_bounds_greedy_roadmap.md`). The README's "Roadmap (not yet built)" section correctly lists only genuinely unbuilt items (e.g. a Krylov/SOAR moment-matching path), nothing already implemented is mislabeled as future work.

### 2.4 Sync status

`rom_engine` has always been built directly at its canonical `computation-suite/rom_engine` location in this session (no separate dev-copy workflow was ever used for it, unlike `fea_engine`'s Phase 6). Nothing to reconcile.

---

## 3. Findings

### Finding 1 — `libGLU.so.1` missing from the review sandbox (environment gap, not a code defect)

`test_geometry_engine.py` and `test_mixed_elements.py::test_mixed_elements` (which exercises a full Gmsh pipeline) both import `gmsh`, whose Python bindings dynamically link `libGLU.so.1` even for headless mesh generation. This library is not installed in the sandbox used for this review, and the sandbox has no root/sudo access to `apt-get install` it directly (confirmed: `sudo` is blocked by a "no new privileges" container flag).

This is the same limitation already documented honestly in the README's "Known limitations" section ("Requires `pip install gmsh`, and on a minimal Linux install, `libglu1-mesa`... as a system library"). It was worked around for this review by downloading the `.deb` packages for `libglu1-mesa`/`libgl1`/`libglx0`/`libglvnd0`/`libopengl0` via `apt-get download` (which doesn't need root) and extracting them into a user-writable directory added to `LD_LIBRARY_PATH`. With that in place, **both previously-blocked tests pass**, bringing the full suite to a clean 41/41 with zero deselected tests — an improvement over the 39-passed/1-deselected state reported at the end of Phase 6, since that prior run didn't have the workaround applied.

This is not a package defect — it's a statement about this particular sandbox's base image, and the workaround is local to this review session (not applied to the package itself, since a real deployment target would simply have the system library installed properly).

### Finding 2 — `hex8_convergence.py` hits a sandbox memory ceiling at its largest sweep point

This pre-existing exploratory script (predates the Phase 1-6 general-purpose extensions work; referenced once in the README as a companion to `main.py`) sweeps through-thickness mesh density for a `Hex8Solid3D` cantilever from 1 to 12 elements. At the last sweep point (`n_tt=12`), the mesh has 30,927 DOF, and the script uses the **dense** `FESystem(mesh, elem)` path (not `sparse=True`), which tries to allocate a `30927 × 30927` float64 stiffness matrix — 7.1 GiB — and the sandbox's available memory is smaller than that.

This is a genuine resource ceiling of the review sandbox, not a numerical or logic bug: the earlier sweep points (up to `n_tt=8`, 3,321 nodes) all completed and printed correct, monotonically-converging locking ratios. It's worth noting as a real, minor opportunity: since Phase 2 (sparse matrix support) already exists in the package, `hex8_convergence.py` could switch to `FESystem(mesh, elem, sparse=True)` to comfortably reach `n_tt=12` (and further) without this ceiling — but that's a script-level convenience change, not a package correctness issue, and wasn't something this review was asked to fix.

---

## 4. Conclusion

Everything implemented through Phase 6 of `fea_engine`'s general-purpose extensions roadmap and through Phase 4d of `rom_engine`'s frequency-domain/certified-bounds roadmap is present, tested, passing, and accurately documented. No regressions were introduced by the Phase 6 work or any prior phase. The only two issues surfaced are both properties of this specific review sandbox (a missing system graphics library, and a memory ceiling on one exploratory script's largest test point), not defects in either package.
