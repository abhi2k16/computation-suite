# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
common.py -- shared helpers for the article experiments.

Every experiment script writes its numbers to ``paper/results/<name>.json`` (with the environment it ran on) and its
figure to ``paper/figures/<name>.pdf`` and ``.png``. The article quotes only values found in those JSON files.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]            # computation-suite/
PAPER = ROOT / "paper"
RESULTS = PAPER / "results"
FIGURES = PAPER / "figures"
for _d in (RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# make the two packages importable when running from a plain checkout (also fine if they are pip-installed)
for _p in (ROOT / "fea_engine" / "src", ROOT / "rom_engine" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def environment():
    """Software and hardware description stored next to every result."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }
    try:
        import scipy
        env["scipy"] = scipy.__version__
    except Exception:                                   # pragma: no cover
        pass
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            try:
                env["nvidia_driver"] = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10).stdout.split()[0]
            except Exception:
                env["nvidia_driver"] = None
    except Exception:
        env["torch"] = None
    try:
        import fea_engine
        import rom_engine
        env["fea_engine"] = fea_engine.__version__
        env["rom_engine"] = rom_engine.__version__
    except Exception:                                   # pragma: no cover
        pass
    try:
        env["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                           capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        env["git_commit"] = None
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = os.environ.get(var)
    return env


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def save_result(name, data):
    """Write ``results/<name>.json`` with the environment attached; returns the path."""
    path = RESULTS / f"{name}.json"
    payload = {"experiment": name, "environment": environment(), "data": _jsonable(data)}
    path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(f"[{name}] results -> {path.relative_to(ROOT)}")
    return path


def load_result(name):
    return json.loads((RESULTS / f"{name}.json").read_text())["data"]


def best_of(fn, repeats=5, warmup=1):
    """Minimum wall time of ``fn()`` over ``repeats`` runs after ``warmup`` untimed runs (seconds)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), float(np.median(times))


def load_module(path, name):
    """Import a Python file by path (used for the test helpers and the shipped examples)."""
    path = Path(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                          "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
    return _plt


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    print(f"[{name}] figure  -> {(FIGURES / (name + '.pdf')).relative_to(ROOT)}")
