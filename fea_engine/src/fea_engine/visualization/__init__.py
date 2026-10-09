# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
visualization -- optional plotting helpers.

Gmsh-native geometry/mesh/result screenshot rendering (formerly
``gmsh_plot.py``, a second, non-default OpenGL rendering path) has been
REMOVED from this package: it required ``pip install gmsh`` plus, on a
minimal Linux install, the system libGLU library, which could not be
reliably provided, and every plot it produced already had a matplotlib
equivalent actually used by every example script in this package. See
docs/generalized_mesh_grading_roadmap.md for the removal note.

fea_engine.mesh's own plot_mesh_2d/plot_mesh_3d/plot_mesh_annotated
(plain matplotlib, no Gmsh dependency) remain in mesh.py, unchanged, and
are the supported rendering path.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"

__all__: list = []
