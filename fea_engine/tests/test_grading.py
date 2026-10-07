"""
tests/test_grading.py -- Phase 1 validation of fea_engine.grading.

Pure sizing-function math, no mesher and no element formulation involved
(see docs/generalized_mesh_grading_roadmap.md, Section 4's "Analytic"
validation item) -- everything here is checked against closed-form/
geometric-series expectations, not against a solve.
"""
import numpy as np
import pytest

try:
    from fea_engine.grading import (
        graded_partition, growth_ratio_of_partition, grade_for_growth_ratio,
        Hole, Fillet, EdgeBias, Notch, plan, GradingPlan, MeshGradingError,
        threshold_field_size, dist_max_for_growth_ratio,
        DEFAULT_GROWTH_RATIO, GROWTH_RATIO_WARN_THRESHOLD,
    )
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "fea_engine"))
    from grading import (
        graded_partition, growth_ratio_of_partition, grade_for_growth_ratio,
        Hole, Fillet, EdgeBias, Notch, plan, GradingPlan, MeshGradingError,
        threshold_field_size, dist_max_for_growth_ratio,
        DEFAULT_GROWTH_RATIO, GROWTH_RATIO_WARN_THRESHOLD,
    )


# =====================================================================
# graded_partition / growth_ratio_of_partition
# =====================================================================
def test_uniform_partition_has_growth_ratio_one():
    coords = graded_partition(0.0, 1.0, 10, grade=1.0)
    assert coords[0] == pytest.approx(0.0)
    assert coords[-1] == pytest.approx(1.0)
    assert growth_ratio_of_partition(coords) == pytest.approx(1.0, abs=1e-10)


def test_graded_partition_monotonic_and_endpoints():
    for grade in (0.5, 1.0, 2.0, 5.0):
        for dense_at in ("start", "end"):
            coords = graded_partition(-2.0, 3.0, 8, grade=grade, dense_at=dense_at)
            assert coords[0] == pytest.approx(-2.0)
            assert coords[-1] == pytest.approx(3.0)
            assert np.all(np.diff(coords) > 0), "partition must be strictly increasing"


def test_growth_ratio_increases_with_grade():
    """Monotonicity of growth_ratio_of_partition in `grade` is what makes
    grade_for_growth_ratio's bisection valid -- check it directly rather
    than just trusting the solver that depends on it."""
    ratios = [growth_ratio_of_partition(graded_partition(0.0, 1.0, 12, grade=g))
              for g in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)]
    assert np.all(np.diff(ratios) >= -1e-9), f"non-monotonic: {ratios}"


def test_dense_at_end_mirrors_dense_at_start():
    a = graded_partition(0.0, 1.0, 9, grade=2.5, dense_at="start")
    b = graded_partition(0.0, 1.0, 9, grade=2.5, dense_at="end")
    assert np.allclose(1.0 - b[::-1], a, atol=1e-12)


def test_growth_ratio_of_partition_short_partitions():
    assert growth_ratio_of_partition([0.0, 1.0]) == 1.0
    assert growth_ratio_of_partition([0.0]) == 1.0


# =====================================================================
# grade_for_growth_ratio: the actual growth-ratio-cap solver
# =====================================================================
@pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("growth_ratio", [1.1, 1.2, 1.3, 1.5, 2.0])
def test_grade_for_growth_ratio_respects_cap(n, growth_ratio):
    grade = grade_for_growth_ratio(n, growth_ratio)
    realized = growth_ratio_of_partition(graded_partition(0.0, 1.0, n, grade))
    # Bisection lands at or just under the cap (tol=1e-3 default), never over.
    assert realized <= growth_ratio + 1e-2, (
        f"n={n} growth_ratio={growth_ratio}: grade={grade} realized={realized}")


def test_grade_for_growth_ratio_is_near_tight():
    """The solved grade shouldn't be needlessly conservative -- confirm it
    gets reasonably close to the requested cap, not just safely under it."""
    grade = grade_for_growth_ratio(16, 1.2)
    realized = growth_ratio_of_partition(graded_partition(0.0, 1.0, 16, grade))
    assert realized == pytest.approx(1.2, abs=0.05)


def test_grade_for_growth_ratio_trivial_cases():
    assert grade_for_growth_ratio(1, 1.2) == 1.0     # n<2: no adjacent pair
    assert grade_for_growth_ratio(10, 1.0) == 1.0     # ratio<=1: only uniform


# =====================================================================
# Feature dataclasses
# =====================================================================
def test_hole_derives_h_min_from_radius_and_n_ring():
    h = Hole(center=(0.0, 0.0), radius=0.02, n_ring=12)
    expected = (0.5 * np.pi * 0.02) / 12
    assert h.h_min == pytest.approx(expected)


def test_hole_explicit_h_min_not_overridden():
    h = Hole(center=(0.0, 0.0), radius=0.02, n_ring=12, h_min=0.001)
    assert h.h_min == 0.001


def test_fillet_same_arc_convention_as_hole():
    hole = Hole(center=(0, 0), radius=0.01, n_ring=8)
    fillet = Fillet(corner=(0, 0), radius=0.01, n_ring=8)
    assert fillet.h_min == pytest.approx(hole.h_min)


def test_edgebias_rejects_bad_dense_at():
    with pytest.raises(ValueError):
        EdgeBias(axis=0, dense_at="middle", h_first=0.001)


def test_edgebias_h_min_is_h_first():
    eb = EdgeBias(axis=1, dense_at="end", h_first=0.0005, growth_ratio=1.3)
    assert eb.h_min == 0.0005


def test_notch_now_implemented_constructs_and_derives_h_min():
    """Wave 5 item 26 (docs/consolidated_future_roadmap.md): Notch is no
    longer a permanently-raising placeholder -- a REAL rectangular-slot
    implementation now exists (see the class's own docstring for the
    deliberate scope narrowing from an arbitrary path to a 2-point
    mouth, and mesh.notch_in_rectangle_mesh() for the structured mesher
    that consumes it; full mesh-level validation lives in tests/
    test_mesh_fillet_notch.py). This replaces the old placeholder test
    that asserted construction always raised."""
    n = Notch(path=[(0, 0), (1, 0)], depth=0.01, n_ring=4)
    assert n.h_min == pytest.approx(1.0 / 4)


def test_notch_still_rejects_malformed_input():
    with pytest.raises(MeshGradingError):
        Notch(path=[(0, 0), (1, 0), (2, 0)], depth=0.01)   # not exactly 2 points
    with pytest.raises(MeshGradingError):
        Notch(path=[(0, 0), (0, 0)], depth=0.01)           # degenerate mouth
    with pytest.raises(MeshGradingError):
        Notch(path=[(0, 0), (1, 0)], depth=0.0)            # non-positive depth


# =====================================================================
# plan()
# =====================================================================
def test_plan_rejects_growth_ratio_at_or_below_one():
    with pytest.raises(MeshGradingError):
        plan([], h_far=0.01, growth_ratio=1.0)
    with pytest.raises(MeshGradingError):
        plan([], h_far=0.01, growth_ratio=0.9)


def test_plan_warns_above_threshold():
    p = plan([], h_far=0.01, growth_ratio=GROWTH_RATIO_WARN_THRESHOLD + 0.1)
    assert any("exceeds" in w for w in p.warnings)


def test_plan_no_warning_at_default_growth_ratio():
    p = plan([Hole(center=(0, 0), radius=0.005, n_ring=8)], h_far=0.05,
              growth_ratio=DEFAULT_GROWTH_RATIO)
    assert p.warnings == []


def test_plan_warns_when_feature_h_min_not_smaller_than_h_far():
    p = plan([Hole(center=(0, 0), radius=1.0, n_ring=2)], h_far=0.01)
    assert any("h_min" in w for w in p.warnings)


def test_plan_grade_for_matches_solver():
    hole = Hole(center=(0, 0), radius=0.01, n_ring=10)
    p = plan([hole], h_far=0.1, growth_ratio=1.25)
    g = p.grade_for(hole)
    assert g == pytest.approx(grade_for_growth_ratio(10, 1.25, "start"))


# =====================================================================
# Threshold-field sizing law (Gmsh-facing, Phase 3 prerequisite)
# =====================================================================
def test_threshold_field_matches_gmsh_definition_pointwise():
    h_min, h_max, dmin, dmax = 0.001, 0.02, 0.0, 0.05
    assert threshold_field_size(0.0, h_min, h_max, dmin, dmax) == pytest.approx(h_min)
    assert threshold_field_size(0.05, h_min, h_max, dmin, dmax) == pytest.approx(h_max)
    assert threshold_field_size(1.0, h_min, h_max, dmin, dmax) == pytest.approx(h_max)  # clipped
    mid = threshold_field_size(0.025, h_min, h_max, dmin, dmax)
    assert h_min < mid < h_max


def test_dist_max_for_growth_ratio_reaches_h_far_within_cap():
    h_min, h_far, gr = 0.001, 0.05, 1.2
    dmax = dist_max_for_growth_ratio(h_min, h_far, gr)
    # Sample the geometric-series "layers" implied by the derivation and
    # confirm consecutive layer sizes don't exceed the growth ratio.
    sizes = [h_min]
    while sizes[-1] < h_far:
        sizes.append(min(sizes[-1] * gr, h_far))
    ratios = [b / a for a, b in zip(sizes[:-1], sizes[1:])]
    assert all(r <= gr + 1e-9 for r in ratios)
    assert dmax > 0.0


def test_dist_max_zero_when_h_min_not_smaller_than_h_far():
    assert dist_max_for_growth_ratio(0.05, 0.01, 1.2) == 0.0


def test_dist_max_rejects_growth_ratio_at_or_below_one():
    with pytest.raises(MeshGradingError):
        dist_max_for_growth_ratio(0.001, 0.05, 1.0)
