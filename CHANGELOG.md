# Changelog

All notable changes to `fea_engine` and `rom_engine`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (`rom_engine` is pre-1.0, so minor releases may change behaviour).

## fea_engine 1.0.1

### Changed (behaviour you may notice)
- **Re-assembly replaces instead of accumulating.** Calling `assemble_stiffness`, `assemble_mass` or
  `assemble_lumped_mass` a second time used to silently add to the existing matrix (doubling it). It now replaces
  the matrix and emits a one-time `UserWarning`. Pass `accumulate=True` to add on purpose. A failed re-assembly
  restores the previous matrix.
- **Invalid input now fails loudly.** `fix_dofs`, `add_nodal_force`, `add_consistent_edge_load` and
  `add_consistent_facet_load` raise `ValueError` for an empty node selection, a node id outside the mesh, or a
  local DOF outside `0..dofs_per_node-1`. A rejected call leaves no partial constraints.
- **Solve guards.** `solve_static` raises `RuntimeError` if the stiffness was never assembled and warns when the
  load vector is all zeros with nothing prescribed. `solve_modal`, the transient solvers and `solve_harmonic`
  raise `RuntimeError` (previously `assert`) when `M` is missing.
- **Scalar density.** A scalar `rho` given to a plane or solid element raises a `ValueError` that says to pass
  `rho * np.eye(n)`.
- **Result type.** `solve_static`, `solve_modal` (mode shapes) and `solve_harmonic` return `FEField`, an
  `ndarray` subclass. Indexing, arithmetic, reductions and saving behave as before.

### Added
- Named DOFs on every element (`dof_names`, aliases) accepted by `fix_dofs` and the load methods;
  `FESystem.dof_names`, `FESystem.dof_index`.
- Named node sets: `Mesh.add_node_set`, `node_set`, `select_nodes(x=, y=, z=, name=)`, usable wherever node ids are.
- `FEField` with `.component`, `.nodal`, `.at`, `.magnitude`, `.to_dataframe`, `.plot`, `.unit_of`; `FESystem.field`.
- `FieldSeries` for load paths, time histories and modes; `FESystem.series`, `FESystem.modal_series`.
- `NewtonOptions`: shared `tol`, `max_iter`, `du_tol`, `energy_tol`, `line_search`, `verbose`; every nonlinear and
  contact driver accepts `options=`. Explicit keywords win.
- `fea_engine.batch.map` for parameter sweeps (serial, thread or process pools, ordered results, error collection).
- `fea_engine.units`: unit labels (`SI`, `MM_N_TONNE`, `UnitSystem`), `Material.units`, `FESystem.units`.
  Labels only; nothing is converted.
- MIT `LICENSE` in each package, author and SPDX headers in every module.

### Fixed
- Invalid-escape `DeprecationWarning` in `tests/test_nonlinear_cantilever.py`; the intended zero-load warning in
  `tests/test_adaptivity.py` is filtered.

## rom_engine 0.2.0

### Added
- `cms` (Craig-Bampton, Guyan, substructure assembly), `hyper_reduction` (ECSW, DEIM/QDEIM, gappy POD, hooked into
  `IntrusiveNonlinearROM`), `linear_dynamics` (reduced Newmark, modal superposition), `validation` (`validate_rom`).
- MIT `LICENSE`, author and SPDX headers in every module.

### Fixed
- Balanced truncation, Hankel-norm and frequency-weighted truncation were sensitive to the BLAS/LAPACK build
  (4 test failures with Intel MKL). The state-space system is now equilibrated with power-of-two diagonal scaling
  (LAPACK `?GEBAL`) before the Gramians are computed.

## Earlier versions
`fea_engine` 1.0.0 and `rom_engine` 0.1.0 were the first tagged states of the packages.
