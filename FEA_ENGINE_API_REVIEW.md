# fea_engine interface review: lessons from PyMAPDL and PyDPF

Scope: design patterns of the public PyAnsys clients (PyMAPDL 0.74, PyDPF-Core 0.16 user guides) compared with the
current `fea_engine` interface (v1.0.0). No code was changed. Every "observed" item below was reproduced against the
real package (probe scripts, `fea_engine` as shipped); PyAnsys claims come from the official user guides (sources at the end).

## 1. Patterns worth borrowing

| # | Pattern in PyMAPDL / PyDPF | fea_engine today | Suggestion |
|---|---|---|---|
| A | Labelled degrees of freedom: `"UX"`, `"UY"` (PyMAPDL commands take labels) | `fix_dofs(nodes, [0, 1])`: integer local indices, "element-formulation-specific" | Each element declares `dof_names` (`("ux","uy")`, `("ux","uy","rz")`, ...). Accept names or indices: `fix_dofs(nodes, ["ux","uy"])`. Validate against the element. |
| B | Typed, self-describing results: `post_processing.nodal_displacement("X")`; DPF `Field` = data + scoping + location + unit | `solve_static()` returns a flat `ndarray`; callers do `U[2*tip+1]` by hand | `Result` object: `.displacement("y", nodes=tip)`, `.nodal` view of shape `(n_nodes, dof_per_node)`, plus the raw vector. |
| C | One container for multi-step results (DPF `FieldsContainer`: one field per time step / frequency / mode) | Return shapes differ per solver: `(freq, shapes)`, `(load_factors, [U...])`, arrays of snapshots | `ModalResult`, `NonlinearPathResult`, `TransientResult` with the same access style: `.steps`, `.at(step)`, iteration. |
| D | Named selections (`nsel`, components; DPF scoping) | `mesh.nodes_on_line(axis, value)` returns bare index arrays | `mesh.sets["clamped"] = ...`, then `fix_dofs("clamped", ["ux","uy"])`; selectors that fail loudly when empty. |
| E | Fail fast with context: PyMAPDL raises `MapdlRuntimeError` showing the offending command | Several mistakes pass silently (section 2) | Validation at the call site with messages that name the argument and the fix. |
| F | `MapdlPool.map(func, inputs)` for batch / parameter sweeps, with progress | no common batch helper (rom_engine has `n_jobs` in sweeps) | `fea_engine.batch.map(build_and_solve, params, n_jobs=...)`, same convention as rom_engine. |
| G | Every call has a short docstring with Parameters / Examples | Docstrings are long design narratives, many cite internal items ("Wave 9 addendum item 138", "Module 18") | One-line summary + Parameters / Returns / Example; move the design history to a docs page. |
| H | Enriched outputs: `.to_array()`, `.to_dataframe()` | none | `Result.to_dataframe()` (optional pandas). Low priority. |

## 2. Defects found by probing (current behaviour)

| Probe | Observed | Why it matters |
|---|---|---|
| `assemble_stiffness` called twice (notebook re-run) | `K` becomes exactly `2 * K`; no warning | Silent wrong answer. Highest risk. |
| `fix_dofs(nodes, [7])` on a 2-DOF-per-node element | accepted; fixes global DOFs `[7, 9, 11, 13]`, i.e. other nodes' DOFs | Silent wrong constraints. Out-of-range local index is never checked. |
| `solve_static()` before anything is assembled | returns an all-zero vector of the right length | Looks like a valid answer. |
| `solve_static()` with no constraints and no loads | returns zeros, no warning | Same. A free body should raise or warn. |
| `nodes_on_line(axis=0, value=0.41)` matches no node, then `add_nodal_force(...)` | `ZeroDivisionError: float division by zero` | Real cause (empty selection) is hidden. |
| `assemble_mass(rho)` with a scalar on `Quad4PlaneStress` (also with `thickness=`) | `ValueError: matmul: Input operand 1 does not have enough dimensions` | Needs `rho * np.eye(2)`; beams take a scalar. The name `rho_or_matrix` suggests a scalar works everywhere. |
| Misspelled keyword (`thicknes=`) | `TypeError: Element.stiffness() got an unexpected keyword argument` | Good: fails loudly. (Names the inner method, not the user's call.) |

## 2b. Status: P0 implemented (v1.0.1)

All five P0 items are done, each with a regression test in `fea_engine/tests/test_api_safety.py`:

| Item | Result |
|---|---|
| `fix_dofs` validation | out-of-range / negative / non-integer local DOF index and node ids outside the mesh raise `ValueError` naming the element and the valid range; a rejected call leaves no partial constraints |
| Empty selections | `fix_dofs`, `add_nodal_force`, `add_consistent_edge_load`, `add_consistent_facet_load` raise `ValueError("... no nodes selected ...")`; the load methods also validate `dof_index` |
| Solve guards | `solve_static` raises `RuntimeError` if `K` is all zeros; `solve_static` warns when the load vector is all zeros and nothing is prescribed; `solve_modal` / transient / harmonic raise `RuntimeError` (was `assert`) if `M` is missing, and if `K` is empty. Free-body (pure Neumann) solves are deliberately left supported; the existing under-constrained check was already correct. |
| Re-assembly | `assemble_stiffness`, `assemble_mass`, `assemble_lumped_mass` are idempotent: a repeated call REPLACES the matrix with a one-time `UserWarning`; `accumulate=True` adds on purpose |
| Scalar density | a scalar `rho` on a plane/solid element raises a `ValueError` that says to pass `rho * np.eye(n)` |

Behaviour change to be aware of: before, assembling twice silently doubled the matrix; now the second call replaces it.
Everything else that was valid before is unchanged (README example E1 reproduces -1.7969e-4 m).

## 2c. Status: P1 implemented (v1.0.1, additive)

Tests: `fea_engine/tests/test_named_api.py` (58 tests). Old integer DOFs, node-id arrays and plain-array behaviour are unchanged.

| Item | Result |
|---|---|
| Named DOFs | Every element declares `dof_names` (`("ux","uy")`, `("uy","rz")` for the Euler-Bernoulli beam, `("w","betax","betay")` for Mindlin plate, ...) plus `dof_aliases` (`x/y/z`, `u/v/w`, `theta`). `fix_dofs` and the load methods accept names, integers or a mix; `FESystem.dof_names` / `dof_index(name)` expose them. Errors list the valid names. |
| Named node sets | `Mesh` and `MultiBlockMesh` gained `node_sets`, `add_node_set`, `node_set` and `select_nodes(x=, y=, z=, tol=, name=)` (scalar or `(lo, hi)` per axis). Set names (or a list of them) can be passed wherever node ids are accepted. |
| Result object | `solve_static`, `solve_modal` (mode shapes) and `solve_harmonic` return `FEField`, an `ndarray` subclass: `U[2*tip+1]` still works and scalar reductions still return NumPy scalars. New: `U.component("uy", nodes="tip")`, `U.nodal`, `U.at(nodes)`, `U.magnitude()`, `U.to_dataframe()`. `FESystem.field(vec)` wraps other vectors (e.g. nonlinear histories). Pickling keeps names but drops the mesh. |

Decision: the "ndarray subclass" option from item 7 was chosen so no existing caller breaks.

## 2d. Status: P2 implemented (v1.0.1, additive)

Tests: `fea_engine/tests/test_newton_options_and_batch.py` (36 tests).

| Item | Result |
|---|---|
| `NewtonOptions` | Frozen dataclass (`tol`, `max_iter`, `du_tol`, `energy_tol`, `line_search`, `verbose`), all `None` = "driver default" because defaults differ between drivers. All 10 nonlinear/contact/transient drivers gained keyword-only `options=`. Explicit keywords win; fields a driver lacks are ignored; calls without `options` are unchanged. Applied by a decorator at the end of `nonlinear_solver.py`, so no driver body was touched. |
| `fea_engine.batch.map` | `map(fn, items, n_jobs=1, backend="process"\|"thread", on_error="raise"\|"collect", progress=None)`. Order preserved, `n_jobs=-1` = all cores, `BatchError` names the failing item, `Failed` placeholders when collecting. Same `n_jobs` convention as rom_engine. An `FEField` returned from a process worker loses its mesh (re-attach with `system.field(np.asarray(U))`). |
| Docstrings | Short Parameters / Example headers added to `fix_dofs`, `add_nodal_force`, `assemble_mass`, `solve_modal`; the older design notes are kept below them under "Notes". `solve_static` and `assemble_stiffness` still have the long narrative only; moving the design history to a docs page is not done. |
| Warnings | `tests/test_nonlinear_cantilever.py` docstring is now a raw string (no invalid-escape warning); the intended zero-load warning in `test_adaptivity` is filtered. |

## 3. Prioritised plan

**P0: safety fixes, no API change.** Small, testable, backwards compatible.
1. `fix_dofs`: validate each local index against the element's DOFs-per-node; raise `ValueError` naming the element and the valid range.
2. Empty selections: `add_nodal_force` / `fix_dofs` raise `ValueError("no nodes selected ...")` instead of dividing by zero or silently doing nothing.
3. `solve_*`: raise if K was never assembled; warn when there are no loads, or raise a clear singular-system error when no constraints exist.
4. Re-assembly: either replace the matrix by default or warn on a second call. The mixed-formulation path (`mixed_assembly.py`) may rely on accumulation, so check that before changing the default; a keyword like `accumulate=False` keeps it explicit.
5. `assemble_mass`: accept a scalar for solids by expanding to `rho * I`, or raise a message that says "pass rho * np.eye(dof_per_node)".

**P1: additive API, old calls keep working.**
6. `dof_names` on elements + named DOFs in `fix_dofs` / `add_nodal_force`.
7. `Result` classes (items B, C). To avoid breaking callers, `solve_static()` could return an `ndarray` subclass that carries the metadata (so `U[2*tip+1]` still works), or the new object could come from a new method. Both options need a prototype before committing.
8. Named node sets on `Mesh` (item D).

**P2: consistency and ergonomics.**
9. A shared `NewtonOptions` dataclass (`tol`, `max_iter`, `du_tol`, `energy_tol`, `line_search`, `verbose`): today these are repeated across 11 `solve_nonlinear_*` drivers.
10. `fea_engine.batch.map` (item F) and docstring clean-up (item G).

**P3: optional.** A units tag on `Material` / `Result` (labels only, no conversion engine), `to_dataframe`, plotting methods on `Result`.

## 4. What not to copy
- The command-string interface (`mapdl.run("K,1,0,0,0")`): it exists to wrap a legacy solver language; `fea_engine` is already Pythonic.
- Client/server architecture (gRPC, mTLS, remote sessions): needed for a separate proprietary solver, unnecessary for an in-process library.
- Global mutable "database session" state and selection stacks: the source of many PyMAPDL pitfalls. Prefer explicit objects.
- Solver-release version pinning and licence gating.

## 5. Sketch of the target feel (not implemented)

Today (README example E1):
```python
U = sys_.solve_static()
tip_uy = U[2 * tip + 1].mean()
sys_.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
```
Proposed:
```python
mesh.sets["root"] = mesh.nodes_on_line(axis=0, value=0.0)
mesh.sets["tip"]  = mesh.nodes_on_line(axis=0, value=0.4)
sys_.fix_dofs("root", ["ux", "uy"])
res = sys_.solve_static()
tip_uy = res.displacement("y", nodes="tip").mean()
```

## Sources
- PyMAPDL user guide: https://mapdl.docs.pyansys.com/version/stable/user_guide/index.html
- PyMAPDL postprocessing: https://mapdl.docs.pyansys.com/version/stable/user_guide/post.html
- PyMAPDL pool: https://mapdl.docs.pyansys.com/version/stable/user_guide/pool.html
- PyDPF-Core concepts: https://dpf.docs.pyansys.com/version/stable/user_guide/concepts/concepts.html
