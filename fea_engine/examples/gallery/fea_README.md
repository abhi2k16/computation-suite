# fea_engine Example Gallery

Six runnable, self-contained scripts mirroring the structure of
[TensorMesh's Example Gallery](https://docs.tensor-mesh.com/example_gallery/index.html),
built against this package's own actual capabilities rather than the
PDE domains (Poisson, wave, Helmholtz, fluids, Maxwell) `fea_engine`
doesn't cover. Each script prints its own validation numbers to
stdout and saves one PNG; run any of them directly:

```
cd fea_engine/examples/gallery
python fea_01_cantilever_beam.py
```

Every filename here is prefixed `fea_` to keep it unambiguous which
package a script belongs to -- including this README itself
(`fea_README.md`). `rom_engine` has its own, separately prefixed
(`rom_`) gallery at `rom_engine/examples/gallery/`, covering that
package's reduced-order-modeling tools instead (see that folder's own
`rom_README.md`).

| # | Script | What it shows | Cross-check used |
|---|---|---|---|
| 1 | `fea_01_cantilever_beam.py` | Linear-elastic cantilever (Quad4PlaneStress): deformed shape colored by von Mises stress, mesh-refinement convergence | FE tip deflection converges to the Euler-Bernoulli analytical value (0.29% at nx=128) |
| 2 | `fea_02_hyperelastic_neo_hookean.py` | Large-strain Neo-Hookean hyperelasticity (Tet4NeoHookean): deformed tet, load-deflection curve | Diverges from a linear-elastic reference (same tet/material) as load grows -- the real hyperelastic signature, not asserted |
| 3 | `fea_03_hertzian_contact.py` | An elastic block pressed onto a rigid circular obstacle (GapContactCurvedFriction, one contact element per bottom node) | Contact pressure distribution is checked for the expected bell shape (peaked under the obstacle, zero outside the patch), not a quantitative Hertz-theory match (different formulation) |
| 4 | `fea_04_plasticity_strip.py` | J2 plasticity with isotropic hardening on a plate-with-a-hole (Quad4PlasticJ2PlaneStress + Wave 5's `hole_in_rectangle_mesh`) | Permanent plastic strain localizes at the hole boundary (stress-concentration-driven local yield well below nominal sigma_y) -- shown via the hardening variable `state["alpha"]`, not assumed |
| 5 | `fea_05_modal_analysis.py` | Generalized eigenproblem modal analysis (`FESystem.solve_modal()`) on a steel cantilever: first four mode shapes | First two bending-mode frequencies checked against the closed-form Euler-Bernoulli cantilever formula (agreement within ~2-3%) |
| 6 | `fea_06_topology_optimization.py` | SIMP density-based compliance topology optimization (`topopt.py`, Wave 11 item 111) on the short-cantilever benchmark: density evolution + compliance history | 86.8% compliance reduction at an exactly-held volume fraction -- the same signature `test_topopt.py`'s own end-to-end test validates |

## Related: solver-backend benchmark

`../fea_scipy_vs_torch_gpu_benchmark.py` (one level up, not part of
this numbered gallery since it's an environment-dependent benchmark
rather than a fixed-result example) times `FESystem.solve_static()`
across SciPy's sparse direct solve, PyTorch's sparse CG on CPU, and
the same CG on CUDA, over a sweep of mesh sizes. Needs a working
CUDA-enabled torch install to produce the GPU curve -- see its own
docstring for details, known environment pitfalls (a NumPy-version
compatibility issue in `mesh_transform.py`'s vectorized-assembly path
was found and fixed while developing this script), and how to run it.

## Notes

- Every script uses real, already-tested `fea_engine` code paths
  (`FESystem.solve_static()`/`solve_nonlinear_static()`/`solve_modal()`,
  `nonlinear_solver.py`, `topopt.py`) -- nothing here is standalone
  physics reimplemented for the plot.
- `fea_engine` is a structural-mechanics-only package (no Poisson,
  diffusion, wave, Helmholtz, or fluid-mechanics element types), so
  this gallery does not attempt those TensorMesh categories. See
  `docs/consolidated_future_roadmap.md`'s Wave 16 entries for the
  page-by-page TensorMesh comparison this gallery followed from.
- Material properties are chosen per-example to make the physics
  visually legible (e.g. a softer "rubber-like" modulus for the
  contact and plasticity examples so the deformation/plastic zone is
  visible at a reasonable magnification) -- not tuned to hide or
  exaggerate any result beyond what's needed for a readable plot.
