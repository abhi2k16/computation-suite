# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
run_all.py -- run every article experiment and the table generator.

    python paper/experiments/run_all.py            # all experiments, then tables
    python paper/experiments/run_all.py e1 e5      # selected experiments (prefix match), then tables

Each script runs in its own process, so a failure in one does not stop the others. A summary with exit codes and
wall times is written to ``paper/results/run_summary.json``.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
SCRIPTS = ["e0_inventory", "e1_rom_speedup", "e2_recovery_speedup", "e3_cms", "e4_ecsw", "e5_verification", "e6_torch_backend",
           "e7_quickstart", "e8_workflow", "make_tables"]


def main(argv):
    wanted = [a for a in argv if not a.startswith("-")]
    todo = [s for s in SCRIPTS if not wanted or s == "make_tables" or any(s.startswith(w) for w in wanted)]
    summary = []
    for name in todo:
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(HERE / f"{name}.py")], cwd=HERE)
        summary.append({"script": name, "returncode": proc.returncode, "seconds": round(time.perf_counter() - t0, 1)})
        print(f"== {name}: exit {proc.returncode} in {summary[-1]['seconds']} s\n")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    bad = [s["script"] for s in summary if s["returncode"]]
    print("failed:", bad if bad else "none")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
