"""
grading.py -- Module: geometry-agnostic mesh-grading / sizing-function
logic, shared by mesh.py's structured front end and geometry/gmsh_engine.py's
unstructured front end (see docs/generalized_mesh_grading_roadmap.md for the
full design rationale and research background).

This module deliberately knows nothing about Gmsh, nothing about a specific
mesh topology, and nothing about any specific element formulation -- it is
pure sizing-function math (Sections 1a/1b/1c/1d of the roadmap doc):

  - `graded_partition`/`growth_ratio_of_partition`/`grade_for_growth_ratio`:
    the 1-D power-law grading law fea_engine already used (moved here from
    mesh.py, `mesh.py` re-exports it for backward compatibility), plus the
    NEW piece -- solving for the grading exponent that respects a target
    growth ratio, instead of the caller guessing `grade_p` by hand.
  - `Hole`/`Fillet`/`EdgeBias`/`Notch`: a small declarative feature-list
    vocabulary (Section 3b) that both front ends consume. `Notch` (Wave 5
    item 26, docs/consolidated_future_roadmap.md) is scoped to a
    rectangular slot cut inward from a straight edge -- see the class's
    own docstring for why a fully arbitrary notch path was deliberately
    narrowed, and `mesh.notch_in_rectangle_mesh()` for the structured
    mesher that consumes it.
  - `plan()`: combines a feature list into a `GradingPlan`, validating the
    requested growth ratio against the literature's commonly-cited bounds
    and warning (not silently accepting) when a feature's target size
    doesn't make sense relative to the far-field size.
  - `threshold_field_size`/`dist_max_for_growth_ratio`: the unstructured
    (Gmsh Threshold-field) analogue of the same growth-ratio law, using
    Gmsh's own Threshold-field definition exactly (piecewise-linear in
    distance between DistMin/DistMax) so Phase 3 can wire this straight
    into `gmsh.model.mesh.field` calls without re-deriving the geometry.

This module does NOT decide whether a given element formulation is safe to
receive a curved/unstructured mesh (the rotational-bending-DOF constraint
documented in the roadmap doc, Section 2) -- that check belongs to the
dispatcher (Phase 4), which is the layer that actually knows about
`elements/shells.py` and friends. Keeping that knowledge out of this module
is deliberate: sizing-function math should be testable without importing any
element code at all.
"""
from dataclasses import dataclass, field
import numpy as np


class MeshGradingError(Exception):
    """Raised when a requested mesh-grading plan cannot be satisfied (an
    impossible growth ratio, a not-yet-implemented feature type), or by
    the Phase 4 dispatcher when an element formulation with rotational
    bending DOFs is asked to consume a mesh topology that hasn't been
    verified safe for it."""


# Literature-derived defaults (roadmap doc Section 1b): 1.2 is the commonly
# cited target growth ratio between adjacent element edge lengths; ratios
# above ~1.5 are still meshable but are past what's generally considered
# acceptable, so plan() warns rather than silently accepting them.
DEFAULT_GROWTH_RATIO = 1.2
GROWTH_RATIO_WARN_THRESHOLD = 1.5


# =====================================================================
# 1-D grading law (structured path) -- Sections 1a/1b
# =====================================================================
def graded_partition(start, end, n, grade=1.0, dense_at="start"):
    """Monotonic 1-D partition of n+1 points between start and end, via a
    power law (grade=1.0 -> uniform; grade>1.0 -> points bunched toward
    the `dense_at` end, since s**grade rises slowly near s=0 for
    grade>1). `dense_at="end"` flips the same profile to cluster at the
    far end instead. This is the same function `mesh.py` has always
    used for `rectangle_with_hole_mesh_quarter`'s radial grading and
    `rectangle_mesh_from_partitions`'s explicit partitions -- moved here
    so both mesh front ends share one implementation."""
    s = np.linspace(0.0, 1.0, n + 1) ** grade
    if dense_at == "end":
        s = 1.0 - s[::-1]
    elif dense_at != "start":
        raise ValueError(f"dense_at must be 'start' or 'end', got {dense_at!r}")
    return start + s * (end - start)


def growth_ratio_of_partition(coords):
    """Max ratio between any two ADJACENT segment lengths in a 1-D
    partition (`coords` need only be strictly monotonic, not uniform).
    This is the realized-grading diagnostic: the quantity Section 1b's
    growth-ratio cap actually constrains. Returns 1.0 when there are
    fewer than 3 points (no adjacent pair of segments to compare)."""
    coords = np.asarray(coords, dtype=float)
    d = np.abs(np.diff(coords))
    if len(d) < 2:
        return 1.0
    ratios = np.maximum(d[1:] / d[:-1], d[:-1] / d[1:])
    return float(ratios.max())


def grade_for_growth_ratio(n, growth_ratio, dense_at="start", grade_max=50.0,
                            tol=1e-3, max_iter=60):
    """Solve, by bisection, for the largest power-law exponent `grade`
    such that `graded_partition(0, 1, n, grade, dense_at)`'s realized
    growth ratio does not exceed `growth_ratio`. This is what lets a
    caller ask for "grade this feature as aggressively as a 1.2 growth
    ratio allows" instead of hand-picking `grade_p` and checking the
    result after the fact.

    `growth_ratio_of_partition` is monotonically non-decreasing in
    `grade` for a power-law partition (more aggressive clustering only
    ever widens the largest adjacent-segment ratio), which is what makes
    bisection valid here.

    n < 2 has no adjacent segment PAIR to constrain (0 or 1 segments),
    so grading has no effect on the growth ratio either way -- returns
    grade=1.0 (uniform) in that case rather than an arbitrary value.
    growth_ratio <= 1.0 admits only the uniform partition (grade=1.0).
    """
    if n < 2 or growth_ratio <= 1.0 + 1e-12:
        return 1.0

    def ratio(g):
        return growth_ratio_of_partition(graded_partition(0.0, 1.0, n, g, dense_at))

    lo, hi = 1.0, grade_max
    if ratio(hi) <= growth_ratio:
        # Even the most aggressive grading checked stays within the cap --
        # n is small enough (or growth_ratio loose enough) that grading
        # isn't the binding constraint. Return hi rather than searching
        # past it.
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if ratio(mid) <= growth_ratio:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


# =====================================================================
# Declarative feature vocabulary -- Section 3b
# =====================================================================
def _arc_h_min(radius, n_ring):
    """Target element size at a curved feature boundary: a quarter-arc
    of the given radius divided into n_ring elements. Matches the
    existing `rectangle_with_hole_mesh_quarter`'s own ntheta convention
    (that function meshes exactly one quadrant)."""
    return (0.5 * np.pi * radius) / n_ring


@dataclass
class Hole:
    """A circular hole needing local mesh refinement -- the feature
    `rectangle_with_hole_mesh_quarter_full` already handles, now
    expressed declaratively instead of as bare positional parameters.
    `h_min`, if not given, is derived from `radius`/`n_ring` via the
    same quarter-arc convention the existing hole mesher uses."""
    center: tuple
    radius: float
    n_ring: int = 12
    h_min: float = None

    def __post_init__(self):
        if self.h_min is None:
            self.h_min = _arc_h_min(self.radius, self.n_ring)


@dataclass
class Fillet:
    """A rounded corner -- geometrically a Hole restricted to the single
    quadrant at `corner` (a full hole needs 4 quadrants around its
    center; a fillet needs exactly the 1 quadrant that rounds the
    corner). Phase 2 reuses Hole's structured block construction for
    this, restricted to one quadrant, rather than a separate code path."""
    corner: tuple
    radius: float
    n_ring: int = 8
    h_min: float = None

    def __post_init__(self):
        if self.h_min is None:
            self.h_min = _arc_h_min(self.radius, self.n_ring)


@dataclass
class EdgeBias:
    """Boundary-layer-style grading along one direction of a structured
    block: fine at one end (`h_first`), growing outward at up to
    `growth_ratio`. This is a direct, un-disguised wrapper over
    `graded_partition`/`grade_for_growth_ratio` -- it exists so a caller
    can express "bias this edge" as a feature alongside Hole/Fillet
    instead of calling the partition functions directly."""
    axis: int            # which structured-block axis this bias applies to
    dense_at: str         # "start" or "end" -- which end is fine
    h_first: float
    growth_ratio: float = DEFAULT_GROWTH_RATIO

    def __post_init__(self):
        if self.dense_at not in ("start", "end"):
            raise ValueError(
                f"dense_at must be 'start' or 'end', got {self.dense_at!r}")
        self.h_min = self.h_first


@dataclass
class Notch:
    """A rectangular (right-angle) slot cut inward from a straight
    boundary edge -- Wave 5 item 26 (docs/consolidated_future_
    roadmap.md), closing the gap this class's own predecessor left open
    (declared for feature-list API forward-compatibility in Phase 1,
    but instantiating one always raised `MeshGradingError` until now).

    Scoped deliberately narrower than "a general notch/slot along an
    arbitrary path" (the roadmap doc's original Section 3b wording,
    `path=[...]`): a fully arbitrary polyline notch needs bespoke block
    topology per shape (real, unbounded work -- there is no single
    generic decomposition for an arbitrary path the way there is for a
    circular Hole/Fillet). A straight-walled rectangular slot is instead
    the direct structural analogue of Fillet's own "Hole restricted to
    one quadrant" simplification: the simplest well-posed member of the
    notch family, matching common engineering usage (a "notch" or
    "slot" cut squarely into an edge), and -- critically -- one with an
    exact, general block-decomposition-and-weld structured mesh (see
    `mesh.notch_in_rectangle_mesh()`), not a bespoke one-off.

    `path` keeps its original name/type (a list) for API continuity
    with the roadmap doc's declared field, but its MEANING is now
    concrete: exactly 2 points, `[mouth_start, mouth_end]`, both lying
    on the SAME straight boundary edge -- the segment of that edge the
    notch opens into (its "mouth"). `depth` is how far inward
    (perpendicular to that edge) the slot cuts. `h_min`, if not given,
    is derived from the mouth width / n_ring -- the straight-edge
    analogue of Hole/Fillet's own arc-length/n_ring convention (see
    `_arc_h_min`), used the same way by `plan()`'s h_min-vs-h_far sanity
    check."""
    path: list
    depth: float
    n_ring: int = 6
    h_min: float = None

    def __post_init__(self):
        if len(self.path) != 2:
            raise MeshGradingError(
                f"Notch.path must have exactly 2 points (the mouth's two "
                f"endpoints on a single straight boundary edge) -- got "
                f"{len(self.path)}. An arbitrary multi-point notch path is "
                "not implemented (see this class's own docstring for why "
                "the scope was narrowed to a rectangular slot).")
        p0, p1 = np.asarray(self.path[0], dtype=float), np.asarray(self.path[1], dtype=float)
        mouth_width = float(np.linalg.norm(p1 - p0))
        if mouth_width <= 0.0:
            raise MeshGradingError(
                f"Notch.path's two points must be distinct, got {self.path}")
        if self.depth <= 0.0:
            raise MeshGradingError(f"Notch.depth must be > 0, got {self.depth}")
        if self.h_min is None:
            self.h_min = mouth_width / self.n_ring


FEATURE_TYPES = (Hole, Fillet, EdgeBias, Notch)


# =====================================================================
# Combining a feature list into a plan -- Section 3b/3c
# =====================================================================
@dataclass
class GradingPlan:
    """The output of `plan()`: a validated feature list plus the derived
    growth-ratio-respecting grading exponent for each feature. Doesn't
    mesh anything itself -- `mesh.py` (Phase 2) and `gmsh_engine.py`
    (Phase 3) consume this to actually build a Mesh."""
    features: list
    h_far: float
    growth_ratio: float
    warnings: list = field(default_factory=list)

    def grade_for(self, feature):
        """The power-law exponent for `feature`'s own n_ring/count that
        respects this plan's growth_ratio cap -- Section 1a's grading
        law, parameterized by Section 1b's growth-ratio cap instead of
        a caller-chosen `grade_p`."""
        n = getattr(feature, "n_ring", None)
        dense_at = getattr(feature, "dense_at", "start")
        if n is None:
            return 1.0
        return grade_for_growth_ratio(n, self.growth_ratio, dense_at)


def plan(features, h_far, growth_ratio=DEFAULT_GROWTH_RATIO):
    """Combine a feature list into a `GradingPlan`.

    Validates `growth_ratio` (must be > 1.0 for grading to be possible
    at all; anything above `GROWTH_RATIO_WARN_THRESHOLD` gets a warning,
    not a hard error, since it's still meshable -- just outside the
    commonly-cited comfortable range) and sanity-checks each feature's
    `h_min` against `h_far` (a feature whose own target size is already
    at or above the far-field size has nothing to grade toward, which
    usually means a units or radius mistake rather than an intentional
    request).
    """
    if growth_ratio <= 1.0:
        raise MeshGradingError(
            f"growth_ratio must be > 1.0 to allow any grading at all, "
            f"got {growth_ratio}")
    warnings = []
    if growth_ratio > GROWTH_RATIO_WARN_THRESHOLD:
        warnings.append(
            f"growth_ratio={growth_ratio} exceeds the commonly cited upper "
            f"bound (~{GROWTH_RATIO_WARN_THRESHOLD}) from mesh-grading "
            "practice -- large adjacent-element size jumps degrade solution "
            "accuracy at the transition even when every individual element "
            "is well-shaped.")
    for f in features:
        h_min = getattr(f, "h_min", None)
        if h_min is not None and h_min >= h_far:
            warnings.append(
                f"{f!r}: h_min ({h_min:.4g}) is not smaller than h_far "
                f"({h_far:.4g}) -- grading has nothing to do for this "
                "feature; check units/radius/n_ring.")
    return GradingPlan(features=list(features), h_far=h_far,
                        growth_ratio=growth_ratio, warnings=warnings)


# =====================================================================
# Unstructured (Gmsh Threshold-field) sizing law -- Sections 1c/1d,
# consumed by gmsh_engine.py in Phase 3.
# =====================================================================
def threshold_field_size(dist, h_min, h_max, dist_min, dist_max):
    """Gmsh's own Threshold-field definition, reproduced exactly (see
    the Gmsh reference manual's Field module): size is h_min for
    dist <= dist_min, h_max for dist >= dist_max, and linearly
    interpolated in between. Implementing this here (rather than only
    inside a `gmsh.model.mesh.field` call) makes the sizing law itself
    testable without Gmsh installed -- Phase 3 wires the SAME dist_min/
    dist_max derived below into the actual Gmsh Field API calls."""
    dist = np.asarray(dist, dtype=float)
    t = np.clip((dist - dist_min) / (dist_max - dist_min), 0.0, 1.0)
    return h_min + t * (h_max - h_min)


def dist_max_for_growth_ratio(h_min, h_far, growth_ratio=DEFAULT_GROWTH_RATIO):
    """Distance band width for a Threshold field that grows from h_min
    at the feature boundary to h_far, without exceeding `growth_ratio`
    between successive "layers" of that growth -- the unstructured
    analogue of `grade_for_growth_ratio`, grounded the same way (Section
    1a/1b's geometric-progression convention): a geometric series with
    ratio `growth_ratio` starting at h_min reaches h_far after
    n = ceil(log(h_far/h_min) / log(growth_ratio)) steps, and the
    physical distance covered by that series is its partial sum,
    h_min * (growth_ratio**n - 1) / (growth_ratio - 1).

    h_min >= h_far returns dist_max=0.0 (no band needed -- see plan()'s
    warning for this same condition on the structured side)."""
    if h_min >= h_far:
        return 0.0
    if growth_ratio <= 1.0:
        raise MeshGradingError(
            f"growth_ratio must be > 1.0, got {growth_ratio}")
    n = int(np.ceil(np.log(h_far / h_min) / np.log(growth_ratio)))
    n = max(n, 1)
    return h_min * (growth_ratio ** n - 1.0) / (growth_ratio - 1.0)
