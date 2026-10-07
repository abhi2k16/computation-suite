# computation-suite

A from-scratch, NumPy/SciPy-native finite element and reduced-order modeling
toolkit for structural mechanics, with an optional PyTorch layer for GPU
acceleration and automatic differentiation.

## Packages

| Package | Purpose |
|---|---|
| [`fea_engine`](fea_engine/README.md) | Full-order finite element package: static, dynamic and geometrically nonlinear analysis, with extensible materials, elements, loads and solvers. |
| [`rom_engine`](rom_engine/README.md) | Reduced-order modeling built on top of `fea_engine`. |

## Install

```
pip install -e fea_engine
pip install -e rom_engine
```

## Documentation

- [USER_GUIDE.md](USER_GUIDE.md): getting started, concepts, examples, performance
- [API_REFERENCE.md](API_REFERENCE.md): API reference
- `fea_engine/docs/`: roadmaps and implementation notes

## Running scripts

`run.py` runs any `fea_engine` / `rom_engine` script with a chosen CPU core
count for the underlying BLAS libraries, without editing the script. See the
docstring at the top of `run.py` for usage.
