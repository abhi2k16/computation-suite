# Article experiments

Every number and figure in the article comes from these scripts. Nothing is typed by hand.

```bash
pip install -e fea_engine -e "rom_engine" matplotlib      # torch optional (only e6)
python paper/experiments/run_all.py                       # about 2.5 minutes on 2 CPU cores
python paper/experiments/run_all.py e1 e5                 # a subset (prefix match)
```

| Script | What it measures | Output |
|---|---|---|
| `e1_rom_speedup.py` | POD + Galerkin + affine ROM of a two-region cantilever: held-out error, 500-point sweep time | `results/e1_rom_speedup.json`, figure |
| `e2_recovery_speedup.py` | Batched stress recovery against the per-element loop, Quad4 and Hex8 meshes | `results/e2_recovery_speedup.json`, figure |
| `e3_cms.py` | Craig-Bampton frequency error against modes kept, clamped-clamped beam in two halves | `results/e3_cms.json`, figure |
| `e4_ecsw.py` | ECSW hyper-reduction: elements kept, force error, force and trajectory speed-up (40 and 120 elements) | `results/e4_ecsw.json`, figure |
| `e5_verification.py` | Patch tests (8 elements), pure bending, beam and bar frequencies, Timoshenko, elastica; observed orders | `results/e5_verification.json`, figure |
| `e6_torch_backend.py` | `ReducedSystem.solve` and stress recovery on torch CPU/GPU against SciPy/NumPy (needs torch; skips cleanly) | `results/e6_torch_backend.json`, figure |
| `make_tables.py` | Reads the JSON files, writes `tables/*.tex` and `tables/numbers.tex` (macros the text uses) | `tables/` |

Rules the scripts follow:

- Each result file stores the environment (Python, NumPy, SciPy, torch, CPU, thread variables, git commit).
- Timings are the minimum over repeats after a warm-up, and say so in the JSON (`timing_rule`).
- Seeds are fixed. Reference values are analytic or computed independently (the elastica by SciPy quadrature).
- `e1`, `e3`, `e4`, `e5` reuse the shipped example and test helpers, so the paper and the test suite cannot drift apart.
- Timings depend on the machine. The article states the machine the tables were generated on; rerun on yours to compare.

For GPU numbers run `python paper/experiments/e6_torch_backend.py` on a machine with CUDA torch, then `make_tables.py`.
