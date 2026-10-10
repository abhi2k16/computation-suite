# Changelog

All notable changes to `fea_engine` and `rom_engine`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (`rom_engine` is pre-1.0, so minor releases may change behaviour).

## Unreleased

### Added (fea_engine)
- Batched stress recovery: `stress`, `strain` and `von_mises` evaluate each block with array operations
  (`vectorized=True`, default). On a 100x100 Quad4 mesh this took 0.11 s against 20.7 s for the per-element loop
  (8000 Hex8 elements: 0.22 s against 8.0 s). `vectorized=False` keeps the old loop, which is also used
  automatically for blocks with position-dependent coefficients. Results agree to rounding.
- Optional torch paths (torch is not required): `stress/strain/von_mises(..., backend="torch", device=)`,
  differentiable `FESystem.stress_tensor(U)` / `von_mises_tensor(U)` that return torch tensors backpropagating to
  `U`, `ReducedSystem.solve(backend="torch", device=, method="auto"|"dense"|"cg")` (works with constraints) and
  `ReducedSystem.to_torch()`. `expand/recover/restrict/residual` accept torch tensors. Computation is float64.
  Tests: `tests/test_torch_recovery_and_linear_system.py` (torch cases skip when torch is not installed).
- `FESystem.form_linear_system(F=None, K=None)` returns a `ReducedSystem`: the constrained `K`, `F` (prescribed
  values already moved to the right-hand side), `restrict`, `expand`, `recover` (full `FEField`),
  `reduce_matrix(M)`, `solve()` and `residual()`. Use it to bring your own solver, preconditioner or ROM.
- `fea_engine.export.write_vtu` and `write_series`: ParaView/VisIt output (`.vtu`, `.pvd` time series) of meshes
  and `FEField` / `FieldSeries` results (vectors, rotations, mode shapes, complex fields, extra point and
  cell data; ASCII or base64 binary). NumPy only.

- Derived results as `FEField`s: `FESystem.stress`, `strain`, `von_mises` and `reactions`. Stress is computed
  at the integration points with the assembled `D` (coefficients included) and projected to the nodes;
  `at="elements"` gives element averages. Supported: Quad4/Quad8/Tri3/Tri6 plane stress and
  Hex8/Hex20/Tet4/Tet10 solids. `reactions` returns `K U - F` at constrained DOFs, with force and moment units.
- `fea_engine.coefficients`: material, density and thickness that vary in space. `by_position(fn, at=)`,
  `by_element(values)` and `from_material(fn, builder, at=)` are accepted by `assemble_stiffness`,
  `assemble_mass` and `assemble_lumped_mass`. `at="centroid"` (default, any element) is piecewise constant;
  `at="gauss"` evaluates at integration points for the supported plane-stress and solid elements.
  A constant coefficient reproduces the plain assembly. Linear assembly only; not with `vectorized=True`.
- `FEField` accepts per-DOF unit labels (`dof_units`), so stress fields report `Pa` and moments `N*m`.

- Springs, foundations and constraints (`fea_engine.boundary`): `add_spring` (grounded springs, optional
  support displacement), `add_elastic_foundation` (Winkler / Robin term `k (u - u_ref)` on boundary edges or
  faces, `k` constant or a function of position), `add_constraint` (general `sum c_i u_i = value`) and `tie`
  (equal-DOF ties between node sets, with an offset for periodic boundaries). Springs and foundations live in
  `K`, survive re-assembly and work in the nonlinear drivers. Constraints are eliminated exactly (the reduced
  matrix stays symmetric positive definite); `solve_static`, `solve_modal` and `form_linear_system` honour
  them, every other solver raises `NotImplementedError` instead of ignoring them.
- `fea_engine.convergence` (`run_study`, `observed_order`, `pairwise_orders`, `richardson`) for mesh-convergence
  studies, and an independent benchmark suite (`tests/test_benchmarks_convergence.py`): patch tests on all eight
  continuum elements, exact pure-bending reproduction by Quad8/Tri6, observed convergence orders for Quad4, Tri3,
  Euler-Bernoulli and Reissner beams and axial-bar vibration, and the large-deflection cantilever elastica checked
  against a SciPy quadrature.

### Changed (fea_engine)
- `FEField.magnitude()` raises a clear `ValueError` for a field with no translational components
  (previously it returned zeros); `FEField.plot()` of a scalar field draws its single component.
- The vtk export writes one-DOF fields as scalars.

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
