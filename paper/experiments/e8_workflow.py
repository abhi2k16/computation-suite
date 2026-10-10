# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
e8_workflow.py -- run listing 2 (``paper/listings/workflow.py``) after listing 1 and store what it computes: natural
frequencies, maximum von Mises stress of the quick-start load, and the out-of-sample error of the quick-start ROM on
held-out sinusoidal tip tractions (a load shape outside the span of the training loads).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import runpy

from common import PAPER, save_result


def main():
    ns = runpy.run_path(str(PAPER / "listings" / "quickstart.py"))
    ns2 = runpy.run_path(str(PAPER / "listings" / "workflow.py"), init_globals=ns)
    save_result("e8_workflow", ns2["workflow"])
    print(ns2["workflow"])


if __name__ == "__main__":
    main()
