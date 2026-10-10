# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
e7_quickstart.py -- run the code listing printed in the article (``paper/listings/quickstart.py``) and store what it
computes, so the listing is executed code and its quoted output is generated, not typed.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import runpy

from common import PAPER, save_result


def main():
    ns = runpy.run_path(str(PAPER / "listings" / "quickstart.py"))
    save_result("e7_quickstart", ns["result"])
    print(ns["result"])


if __name__ == "__main__":
    main()
