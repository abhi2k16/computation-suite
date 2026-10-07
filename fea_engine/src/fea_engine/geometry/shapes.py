"""
geometry.py -- Module 11: a single, dimension-driven entry point for
mesh generation + FEM system setup.

Every mesh generator already lives in mesh.py and is independently
usable; this module doesn't replace any of them, it adds ONE front
door: give it dim (1, 2, or 3) and, optionally, a shape and a physics
choice, and it picks a matching mesh generator AND a matching default
element, wires them into a ready-to-use FESystem, and hands it back --
"further processing for FEM formulation" (boundary conditions, loads,
assembly, solve) can start immediately. It's the same three-line
pattern every script in this project already follows (mesh -> element
-> FESystem, see README section A), just collapsed behind one call for
the common cases.

    dim=1 -> mesh.line_mesh          -> Beam2DEulerBernoulli (default)
                                         or TrussTL2D / TrussPlastic2D
    dim=2 -> mesh.rectangle_mesh     -> Quad4PlaneStress (default)
             (or rectangle_with_hole)   or Quad4MindlinPlate
    dim=3 -> mesh.box_mesh           -> Hex8Solid3D (default)
             (or box_with_hole)

To add a new geometry or a new default element mapping: add one entry
to GEOMETRY_REGISTRY or ELEMENT_DEFAULTS -- nothing else here changes,
the same registry convention as config.CONSTITUTIVE_REGISTRY and
element.ELEMENT_REGISTRY.
"""
import numpy as np
from .. import mesh as meshmod
from .. import elements as elemmod
from ..solver import FESystem


# =====================================================================
# dim -> shape name -> mesh generator (mesh.py functions, untouched)
# =====================================================================
GEOMETRY_REGISTRY = {
    1: {"line": meshmod.line_mesh},
    2: {"rectangle": meshmod.rectangle_mesh,
        "rectangle_with_hole": meshmod.rectangle_with_hole_mesh_quarter},
    3: {"box": meshmod.box_mesh,
        "box_with_hole": meshmod.box_with_hole_mesh},
}
DEFAULT_SHAPE = {1: "line", 2: "rectangle", 3: "box"}

# dim -> physics name -> element class. "physics" picks WHICH field/
# behavior this element models, not the geometry -- e.g. dim=2 can be
# a membrane (plane_stress/plane_strain, 2 dof/node) or plate bending
# (3 dof/node) on the exact same rectangle mesh; dim=1 can be a
# bending beam or an axial truss (linear or geometrically/materially
# nonlinear) on the exact same line mesh.
ELEMENT_DEFAULTS = {
    1: {"beam": elemmod.Beam2DEulerBernoulli,
        "truss": elemmod.TrussTL2D,
        "truss_plastic": elemmod.TrussPlastic2D},
    2: {"plane_stress": elemmod.Quad4PlaneStress,
        "plane_strain": elemmod.Quad4PlaneStress,   # same element; use
                                                      # config.D_plane_strain
                                                      # instead of D_plane_stress
        "plate": elemmod.Quad4MindlinPlate},
    3: {"solid": elemmod.Hex8Solid3D},
}
DEFAULT_PHYSICS = {1: "beam", 2: "plane_stress", 3: "solid"}


def generate_mesh(dim, shape=None, mirror=None, **kwargs):
    """dim: 1, 2, or 3. shape: a key of GEOMETRY_REGISTRY[dim] (defaults
    to DEFAULT_SHAPE[dim] -- the simplest geometry for that dimension:
    'line' / 'rectangle' / 'box'). kwargs are passed straight through,
    UNCHANGED, to the underlying mesh.py generator (e.g. L=1.0, n=20
    for a line; Lx=, Ly=, nx=, ny= for a rectangle) -- see each
    generator's own docstring in mesh.py for its exact parameters;
    this function deliberately doesn't repeat them, so it can never go
    stale if mesh.py's signatures change.

    dim=1 special case: mesh.line_mesh() itself returns 1-column (x
    only) node coordinates, since that's all Beam2DEulerBernoulli
    needs. This function instead embeds the line into 2-D (appends a
    second, zero-valued column) so the SAME 1-D mesh also works
    directly with TrussTL2D / TrussPlastic2D / GapContactPenalty (2
    dof/node) -- standard FEA practice (line elements embedded in a
    higher-dimensional space), and harmless for the beam element,
    which only ever reads the x column.

    mirror: for a quarter-symmetry hole geometry (2-D
    'rectangle_with_hole'), pass mirror=(mirror_x, mirror_y) to reflect
    the quarter into a full cross-section via mesh.mirror_mesh() before
    returning. None (default) leaves it as the quarter mesh -- the
    right choice if you plan to impose symmetry boundary conditions
    yourself (see plate_with_hole_fem.py)."""
    if shape is None:
        if dim not in DEFAULT_SHAPE:
            raise ValueError(f"unknown dim={dim!r}; available dims: {sorted(DEFAULT_SHAPE)}")
        shape = DEFAULT_SHAPE[dim]
    if dim not in GEOMETRY_REGISTRY or shape not in GEOMETRY_REGISTRY[dim]:
        raise ValueError(
            f"no generator registered for dim={dim!r}, shape={shape!r}; "
            f"available: {_available(GEOMETRY_REGISTRY)}")

    m = GEOMETRY_REGISTRY[dim][shape](**kwargs)

    if dim == 1:
        nodes2d = np.zeros((len(m.nodes), 2))
        nodes2d[:, 0] = m.nodes[:, 0]
        m = meshmod.Mesh(nodes2d, m.elements, dim=2)

    if mirror is not None:
        mirror_x, mirror_y = mirror
        m = meshmod.mirror_mesh(m, mirror_x=mirror_x, mirror_y=mirror_y)

    return m


def default_element(dim, shape=None, physics=None):
    """Returns a freshly constructed default Element instance for this
    dim (+ optional physics, e.g. 'beam' vs 'truss' at dim=1, or
    'plane_stress' vs 'plate' at dim=2 -- see ELEMENT_DEFAULTS). shape
    is accepted for a matching signature with generate_mesh()/
    build_system() but doesn't currently affect the choice -- every
    registered shape at a given dim uses the same element family (e.g.
    both 'box' and 'box_with_hole' are Hex8Solid3D)."""
    if physics is None:
        if dim not in DEFAULT_PHYSICS:
            raise ValueError(f"unknown dim={dim!r}; available dims: {sorted(DEFAULT_PHYSICS)}")
        physics = DEFAULT_PHYSICS[dim]
    if dim not in ELEMENT_DEFAULTS or physics not in ELEMENT_DEFAULTS[dim]:
        raise ValueError(
            f"no default element for dim={dim!r}, physics={physics!r}; "
            f"available: {_available(ELEMENT_DEFAULTS)}")
    return ELEMENT_DEFAULTS[dim][physics]()


def build_system(dim, shape=None, physics=None, thickness=1.0, mirror=None,
                  **mesh_kwargs):
    """The full pipeline in one call: dimension -> mesh -> element ->
    FESystem, ready for boundary conditions/loads/assembly/solve (see
    README sections A-K for what comes next, chosen by which `physics`
    you asked for). Returns (fesystem, mesh, elem) -- all three handed
    back explicitly, since the very next steps (fix_dofs,
    add_nodal_force, assemble_stiffness/assemble_internal_force, ...)
    need at least one of them directly, not just the FESystem.

    Example -- shortest path from "I want a 2-D plane-stress problem"
    to a solved static result:
        sys, mesh, elem = build_system(dim=2, Lx=0.4, Ly=0.2, nx=24, ny=12)
        sys.assemble_stiffness(D_plane_stress(mat), thickness=0.02)
        tip = mesh.nodes_on_line(axis=0, value=0.4)
        sys.add_nodal_force(tip, dof_index=1, total_force=-20000.0)
        sys.fix_dofs(mesh.nodes_on_line(axis=0, value=0.0), [0, 1])
        U = sys.solve_static()
    """
    m = generate_mesh(dim, shape=shape, mirror=mirror, **mesh_kwargs)
    elem = default_element(dim, shape=shape, physics=physics)
    sys = FESystem(m, elem, thickness=thickness)
    return sys, m, elem


def _available(registry):
    return {d: sorted(opts) for d, opts in registry.items()}
