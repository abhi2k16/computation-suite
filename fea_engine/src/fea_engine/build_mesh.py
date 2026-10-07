"""
build_mesh.py -- Phase 4 of docs/generalized_mesh_grading_roadmap.md: the
single entry point that decides, from a feature list and the target
element formulation, whether to grade a mesh at all, which mesh-
generation front end to use, and refuses combinations this package
hasn't verified safe -- rather than silently handing back a mesh that
looks fine and is quietly 40x-1500x too stiff, the exact failure mode the
Shell4MITCCorotational reference-frame fix and the concurrent
`_BEND_SIGN` fix (elements/shells.py) both addressed (see the roadmap
doc's Section 2 for the full history).

This module is deliberately the ONLY place that needs to know both "which
mesh-generation functions exist" (mesh.py's structured front end -- the
Gmsh-backed unstructured front end that used to live at
geometry/gmsh_engine.py was removed; see below) AND "which element
formulations carry rotational bending DOFs with a local-frame constraint"
(elements/plates.py, elements/shells.py) -- grading.py itself stays
ignorant of both by design (see that module's own docstring), so the
safety-relevant knowledge lives in exactly one place rather than being
duplicated or, worse, only enforced by convention/docstring warning the
way it was before this phase.

Gmsh-backed unstructured meshing was removed from this package (system
OpenGL/libGLU dependency that could not be reliably provided, and no
verified example or result depended on it -- see
docs/generalized_mesh_grading_roadmap.md for the removal note). The
translational-DOF branch of `build_mesh()` below that used to dispatch to
the Gmsh path now raises `MeshGradingError` with an explanation instead
of silently falling back to a different (unverified-for-that-case) mesh
path.
"""
from .grading import Hole, MeshGradingError, DEFAULT_GROWTH_RATIO
from . import mesh as mesh_mod
from .elements.plates import Quad4MindlinPlate
from .elements.shells import Shell4MITC, Shell4MITCCorotational
from .elements.shells_director import Shell4Director

# Element formulations with rotational bending DOFs (w, theta_x, theta_y
# / betax, betay) whose local reference frame is derived from element
# geometry -- the family shown (this session's reference-frame fix, and
# the concurrent _BEND_SIGN investigation) to develop severe spurious
# stiffening when neighboring elements' local frames are mutually
# misaligned, which happens on any curved or unstructured mesh but not
# on a structured axis-aligned grid.
#
# Deliberately an explicit tuple of classes, NOT a rule like "any element
# with dofs_per_node > 2" or "> 3": beam elements (Beam2DEulerBernoulli,
# Beam3DEulerBernoulli, Beam2DCorotational) also carry rotational DOFs
# but are 1-D elements with no 2-D local-frame-misalignment mechanism at
# all, so a blanket DOF-count rule would both over-restrict (beams) and
# be fragile to get right for future element additions. Extend this
# tuple explicitly, with a comment, when a new rotational-bending-DOF
# 2-D/3-D element formulation is added and its own safe/unsafe topology
# envelope has been checked -- do not infer membership automatically.
# `Shell4Director` (registered 2026-09-10, docs/shells.md Section 1.3)
# added here too: it carries the same class of rotational bending DOFs
# with a geometry-derived local reference director, and its own scope
# to date (docs/shells.md Section 1.3.2, "flat reference only") has
# never been checked against a curved/unstructured mesh -- included
# per this tuple's own stated policy above ("extend... do not infer
# membership automatically"), not because a specific failure was
# found, but because none of the checking this gate exists to require
# has been done for it either.
ROTATIONAL_DOF_ELEMENTS = (Quad4MindlinPlate, Shell4MITC, Shell4MITCCorotational, Shell4Director)


def _is_rotational_dof(element_formulation):
    if isinstance(element_formulation, type):
        return issubclass(element_formulation, ROTATIONAL_DOF_ELEMENTS)
    return isinstance(element_formulation, ROTATIONAL_DOF_ELEMENTS)


class RectangleWithHole:
    """The one geometry this dispatcher currently understands end-to-end
    on BOTH mesh-generation front ends: a rectangle [0,Lx] x [0,Ly] with
    at most one circular hole. This is deliberately narrow, not a stand-
    in for general CAD feature detection -- see the roadmap doc's Phase
    2/3 scope notes for what's not yet implemented (Notch, a Fillet mesh
    path, multiple holes, arbitrary-geometry feature walking). Extending
    `build_mesh()` to a new geometry means adding a new class with the
    same `detect_features()`/`Lx`/`Ly` contract, not editing this one.

    `detect_features()` returns [] for a plain rectangle (hole_center is
    None) -- `build_mesh()` then skips grading entirely and returns
    today's plain uniform mesh, unchanged behavior for the simple case --
    or a single `Hole` feature otherwise."""

    def __init__(self, Lx, Ly, hole_center=None, hole_radius=None, n_ring=12):
        self.Lx, self.Ly = Lx, Ly
        self.hole_center = hole_center
        self.hole_radius = hole_radius
        self.n_ring = n_ring

    def detect_features(self):
        if self.hole_center is None:
            return []
        return [Hole(center=self.hole_center, radius=self.hole_radius,
                      n_ring=self.n_ring)]


def build_mesh(geometry, target_size, element_formulation, growth_ratio=None,
                nr=8, n_extend=None):
    """The Phase 4 dispatcher (roadmap doc Section 3c): always calls
    `geometry.detect_features()` and decides FOR ITSELF whether/how to
    grade, instead of requiring the caller to notice "this geometry is
    complex" and reach for a differently-named function.

    - No features detected -> today's plain uniform mesh
      (`mesh.rectangle_mesh`, sized from `target_size`) -- unchanged
      behavior, no grading machinery invoked for the common simple case.
    - Features present, `element_formulation` has rotational bending
      DOFs (`ROTATIONAL_DOF_ELEMENTS`) -> ALWAYS the structured,
      axis-aligned-block path (`mesh.hole_in_rectangle_mesh_graded`,
      Phase 2) -- this is the safety gate: a rotational-DOF element can
      never reach the curved/unstructured Gmsh path through this
      function, regardless of what would otherwise be the "natural"
      choice, because that combination hasn't been verified safe (see
      the roadmap doc's Section 2). If the feature list ever contains
      something the structured path can't build (not yet possible today
      -- RectangleWithHole only ever produces a single Hole feature, but
      future geometries may add features only the Gmsh path can mesh),
      this raises MeshGradingError instead of silently falling through
      to the unverified path.
    - Features present, `element_formulation` is translational-DOF ->
      previously the Gmsh unstructured field-graded path (Phase 3,
      `generate_2d_plate_with_hole_graded`), which the roadmap doc's
      Section 2 established has no local-frame constraint to violate.
      That path was removed along with Gmsh support; this branch now
      raises `MeshGradingError` explaining the removal instead of
      generating a mesh, rather than silently falling back to the
      structured path (which is only verified-safe for a different
      reason, not as a general substitute).
    """
    if growth_ratio is None:
        growth_ratio = DEFAULT_GROWTH_RATIO
    features = geometry.detect_features()

    if not features:
        nx = max(1, round(geometry.Lx / target_size))
        ny = max(1, round(geometry.Ly / target_size))
        return mesh_mod.rectangle_mesh(geometry.Lx, geometry.Ly, nx, ny)

    single_hole = len(features) == 1 and isinstance(features[0], Hole)

    if _is_rotational_dof(element_formulation):
        if not single_hole:
            raise MeshGradingError(
                f"{element_formulation.__name__ if isinstance(element_formulation, type) else type(element_formulation).__name__} "
                "has rotational bending DOFs and this feature list has no "
                "verified-safe structured (axis-aligned block) mesh "
                "representation yet -- only a single Hole feature is "
                "currently supported on the structured path. Refusing to "
                "fall back to the unstructured Gmsh path, which has not "
                "been verified safe for this element formulation (see "
                "docs/generalized_mesh_grading_roadmap.md, Section 2).")
        hole = features[0]
        return mesh_mod.hole_in_rectangle_mesh_graded(
            geometry.Lx, geometry.Ly, hole.center, hole.radius,
            nr=nr, ntheta=hole.n_ring, h_far=target_size,
            growth_ratio=growth_ratio, n_extend=n_extend)

    if not single_hole:
        raise MeshGradingError(
            "no mesh generator (structured or unstructured) currently "
            "implements this feature combination -- see docs/"
            "generalized_mesh_grading_roadmap.md Phase 2/3 scope notes.")
    raise MeshGradingError(
        "the unstructured (Gmsh) grading path was removed from this "
        "package -- Gmsh-backed geometry support (geometry/gmsh_engine.py) "
        "depended on a system OpenGL library (libGLU) that could not be "
        "reliably provided, and no verified example or result in this "
        "package used it. Only the structured, axis-aligned-block path "
        "(mesh.hole_in_rectangle_mesh_graded) is available for "
        "single-Hole feature lists now, regardless of element "
        "formulation; a rotational-DOF element formulation's structured "
        "path is NOT a general substitute -- that path exists because "
        "it's the only one ever verified safe for THOSE elements, not "
        "because it's a general fallback for translational-DOF elements "
        "on non-rectangular feature lists the structured path cannot "
        "represent. See docs/generalized_mesh_grading_roadmap.md for the "
        "removal note.")
