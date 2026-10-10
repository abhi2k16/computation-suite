# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
e0_inventory.py -- size and structure of the code base (claim B4): lines, files, tests, registries, public names.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import re
import subprocess
import sys

from common import ROOT, save_result


def loc(path):
    files = [p for p in path.rglob("*.py") if "__pycache__" not in p.parts]
    return {"files": len(files), "lines": sum(sum(1 for _ in open(p, encoding="utf-8")) for p in files)}


def collected(pkg):
    """Number of tests pytest collects (no execution)."""
    out = subprocess.run([sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"],
                         cwd=ROOT / pkg, capture_output=True, text=True, timeout=600).stdout
    m = re.search(r"(\d+) tests? collected", out) or re.search(r"(\d+)/\d+ tests collected", out)
    return int(m.group(1)) if m else None


def main():
    import fea_engine
    import rom_engine
    from fea_engine import CONSTITUTIVE_REGISTRY, ELEMENT_REGISTRY
    data = {
        "fea_src": loc(ROOT / "fea_engine" / "src" / "fea_engine"),
        "rom_src": loc(ROOT / "rom_engine" / "src" / "rom_engine"),
        "fea_tests_code": loc(ROOT / "fea_engine" / "tests"),
        "rom_tests_code": loc(ROOT / "rom_engine" / "tests"),
        "fea_tests_collected": collected("fea_engine"),
        "rom_tests_collected": collected("rom_engine"),
        "elements_registered": len(ELEMENT_REGISTRY),
        "constitutive_registered": len(CONSTITUTIVE_REGISTRY),
        "fea_public_names": len(fea_engine.__all__),
        "rom_public_names": len(rom_engine.__all__),
        "fea_version": fea_engine.__version__,
        "rom_version": rom_engine.__version__,
    }
    save_result("e0_inventory", data)
    print(data)


if __name__ == "__main__":
    main()
