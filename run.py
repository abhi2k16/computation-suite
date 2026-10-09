#!/usr/bin/env python3
"""
run.py -- run any fea_engine / rom_engine script with a chosen CPU core
count for the underlying numpy/scipy (OpenBLAS) and, if used, torch
linear algebra -- without editing the target script at all.

Why this exists
----------------
Neither fea_engine nor rom_engine implements its own multi-core
scheduling (no n_jobs / num_threads parameter anywhere in either
package). All multi-core use comes transparently from the OpenBLAS
library numpy/scipy link against, and OpenBLAS decides its thread
count from a handful of environment variables that MUST be set
BEFORE numpy is first imported in a process -- setting them after
`import numpy` has already run has no effect. This wrapper sets them
and then runs your target script inside the same process via
runpy, so the target script's own `import numpy` (whenever it
happens) sees the variables already in place.

Usage
-----
    python run.py --cores 8 path/to/script.py [script args...]
    python run.py --cores 1 fea_engine/examples/main.py
    python run.py fea_engine/examples/main.py            # no --cores: library default (usually all detected cores)

--cores sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
and NUMEXPR_NUM_THREADS all to the same value, which covers every
BLAS/OpenMP backend numpy/scipy might be built against, plus the
handful of packages (e.g. pandas' numexpr) that read the last one.
If you only ever use OpenBLAS (the default for the numpy/scipy
wheels this project uses), OPENBLAS_NUM_THREADS alone would be
enough -- setting all four is just cheap insurance against a
different BLAS backend on a different machine.

Equivalent shell-only alternative (no wrapper, same effect, useful
if you don't want a Python launcher in the loop):

    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python path/to/script.py

On Windows PowerShell:

    $env:OPENBLAS_NUM_THREADS=8; $env:OMP_NUM_THREADS=8; $env:MKL_NUM_THREADS=8; python path\to\script.py
"""
__author__ = "Abhijeet"
import argparse
import os
import runpy
import sys

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def main():
    parser = argparse.ArgumentParser(
        description="Run a Python script with a chosen CPU core count for numpy/scipy (OpenBLAS) and torch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        metavar="N",
        help="Number of CPU cores for BLAS/OpenMP to use. Omit to leave the library default in place "
             "(usually every core the OS reports). Must be a positive integer.",
    )
    parser.add_argument("script", help="Path to the Python script to run.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed through to the target script.")
    args = parser.parse_args()

    if args.cores is not None:
        if args.cores < 1:
            parser.error("--cores must be a positive integer")
        value = str(args.cores)
        for var in _THREAD_ENV_VARS:
            os.environ[var] = value
        print(f"[run.py] Core count set to {value} via {', '.join(_THREAD_ENV_VARS)}", file=sys.stderr)
    else:
        print("[run.py] --cores not given: leaving BLAS/OpenMP thread count at its library default.", file=sys.stderr)

    if not os.path.isfile(args.script):
        parser.error(f"script not found: {args.script}")

    # Make the target script see the same argv it would if invoked directly
    # (sys.argv[0] = its own path, followed by its own arguments).
    sys.argv = [args.script] + args.script_args

    # Also make sure the target script's own directory is importable,
    # matching normal `python script.py` behavior.
    script_dir = os.path.dirname(os.path.abspath(args.script))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
