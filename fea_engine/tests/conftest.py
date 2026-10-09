# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
Shared pytest configuration for the fea_engine test suite.

Forces matplotlib's non-interactive Agg backend before any test module
imports pyplot, since several tests (test_nonlinear_beam,
test_nonlinear_cantilever) save validation plots as a side effect and
must not require a display.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import matplotlib
matplotlib.use("Agg")
