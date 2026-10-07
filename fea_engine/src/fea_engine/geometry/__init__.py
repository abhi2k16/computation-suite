"""
geometry -- dimension-driven mesh/FESystem front end (shapes.py).

Split out of the original flat geometry.py / geometry_engine.py during
the fea_engine restructuring, grouped into one subpackage since both are
"turn a shape description into a mesh" concerns -- no logic changed,
only file location.

``shapes`` (structured/mapped meshes: line, rectangle, rectangle-with-
hole, box, box-with-hole) has NO external dependency beyond numpy and
is always available.

Gmsh-backed geometry support (arbitrary CAD-like geometry and STEP/IGES/
BREP import via the Gmsh Python API, formerly ``gmsh_engine.py``) has
been REMOVED from this package: it required ``pip install gmsh`` plus,
on a minimal Linux install, the system libGLU library, which could not
be reliably provided, and no verified example or result in this package
depended on it. The supported path for arbitrary/unstructured meshing
needs is now just ``shapes``/``mesh.py``'s structured front end. See
docs/generalized_mesh_grading_roadmap.md for the removal note.
"""
from .shapes import (
    generate_mesh,
    default_element,
    build_system,
    GEOMETRY_REGISTRY,
    DEFAULT_SHAPE,
    ELEMENT_DEFAULTS,
)

__all__ = [
    "generate_mesh", "default_element", "build_system",
    "GEOMETRY_REGISTRY", "DEFAULT_SHAPE", "ELEMENT_DEFAULTS",
]
