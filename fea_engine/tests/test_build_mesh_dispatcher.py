"""
tests/test_build_mesh_dispatcher.py -- Phase 4 validation: build_mesh()'s
complexity-driven dispatch and the rotational-DOF safety gate (docs/
generalized_mesh_grading_roadmap.md, Section 3c/Section 4's "Safety gate"
validation item).

The gate test is deliberately two-directional (roadmap doc's own
requirement): confirm the restricted path is actually exercised in both
directions, not just assumed to work because the code "looks right".

Gmsh-backed unstructured meshing (the former translational-DOF/"Gmsh
path" direction of this gate) was later removed from the package
entirely (system libGLU dependency; see build_mesh.py's own docstring
and docs/generalized_mesh_grading_roadmap.md) -- that direction's tests
below now confirm build_mesh() raises MeshGradingError instead of
confirming it reached a (now nonexistent) Gmsh code path.
"""
import numpy as np
import pytest

from fea_engine.build_mesh import build_mesh, RectangleWithHole, ROTATIONAL_DOF_ELEMENTS
from fea_engine.grading import MeshGradingError
from fea_engine.elements import Shell4MITCCorotational, Shell4MITC, Tri3PlaneStress, Quad4MindlinPlate
from fea_engine.mesh import Mesh


# =====================================================================
# No features -> plain uniform mesh, unchanged behavior
# =====================================================================
def test_plain_rectangle_no_features_uses_uniform_mesh():
    geom = RectangleWithHole(Lx=1.0, Ly=0.5)  # no hole_center -> no features
    assert geom.detect_features() == []
    m = build_mesh(geom, target_size=0.1, element_formulation=Tri3PlaneStress)
    assert m.dim == 2
    # A plain rectangle_mesh -- every element the same size (uniform grid).
    sizes = []
    for e in m.elements:
        x, y = m.nodes[e, 0], m.nodes[e, 1]
        sizes.append(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    sizes = np.array(sizes)
    assert np.allclose(sizes, sizes[0], rtol=1e-9)


# =====================================================================
# Safety gate (rotational-DOF -> structured path; translational-DOF ->
# was the Gmsh path, now removed -- see build_mesh.py's own docstring
# and docs/generalized_mesh_grading_roadmap.md for the removal note)
# =====================================================================
def test_rotational_dof_element_routed_to_structured_path_only():
    """A Shell4MITCCorotational request on a featured (hole) geometry
    must come back from the STRUCTURED path -- verified by confirming it
    succeeds and produces a Quad4 mesh (the structured path's
    signature)."""
    geom = RectangleWithHole(Lx=1.0, Ly=1.0, hole_center=(0.5, 0.5),
                              hole_radius=0.1, n_ring=8)
    m = build_mesh(geom, target_size=0.15, element_formulation=Shell4MITCCorotational,
                    nr=5)
    assert m.elements.shape[1] == 4, "structured path must return Quad4 connectivity"
    assert m.check_grading(verbose=False) >= 1.0  # sane, no exception


def test_rotational_dof_gate_unaffected_by_gmsh_removal():
    """The rotational-DOF path never touched Gmsh even before its
    removal (see build_mesh.py's own routing comment); confirm it still
    succeeds now that the Gmsh path doesn't exist at all, while the
    TRANSLATIONAL path on the same featured geometry now raises
    MeshGradingError unconditionally (that path was removed along with
    Gmsh support, not merely made conditional on gmsh being
    importable)."""
    geom = RectangleWithHole(Lx=1.0, Ly=1.0, hole_center=(0.5, 0.5),
                              hole_radius=0.1, n_ring=8)
    m = build_mesh(geom, target_size=0.15, element_formulation=Quad4MindlinPlate, nr=5)
    assert m.elements.shape[1] == 4

    with pytest.raises(MeshGradingError):
        build_mesh(geom, target_size=0.15, element_formulation=Tri3PlaneStress)


def test_translational_dof_element_with_features_raises_since_gmsh_removed():
    """A translational-DOF element on a featured (hole) geometry used to
    reach the unstructured Gmsh path (Tri3 connectivity); that path was
    removed along with Gmsh support entirely, so this must now raise
    MeshGradingError with a message pointing at the removal, not attempt
    to mesh anything."""
    geom = RectangleWithHole(Lx=1.0, Ly=1.0, hole_center=(0.5, 0.5),
                              hole_radius=0.1, n_ring=8)
    with pytest.raises(MeshGradingError, match="removed"):
        build_mesh(geom, target_size=0.15, element_formulation=Tri3PlaneStress)


def test_rotational_dof_elements_registry_contents():
    assert Shell4MITC in ROTATIONAL_DOF_ELEMENTS
    assert Shell4MITCCorotational in ROTATIONAL_DOF_ELEMENTS
    assert Quad4MindlinPlate in ROTATIONAL_DOF_ELEMENTS
    assert Tri3PlaneStress not in ROTATIONAL_DOF_ELEMENTS


def test_rotational_dof_gate_rejects_unsupported_feature_combination():
    """A rotational-DOF element combined with a feature list the
    structured path can't handle must raise MeshGradingError, not
    silently attempt the unverified Gmsh path."""
    class _FakeMultiFeatureGeometry:
        Lx, Ly = 1.0, 1.0
        def detect_features(self):
            from fea_engine.grading import Hole
            return [Hole(center=(0.3, 0.5), radius=0.05, n_ring=8),
                    Hole(center=(0.7, 0.5), radius=0.05, n_ring=8)]  # 2 holes -- unsupported

    with pytest.raises(MeshGradingError):
        build_mesh(_FakeMultiFeatureGeometry(), target_size=0.1,
                   element_formulation=Shell4MITC)


# =====================================================================
# check_grading() -- growth-ratio realized-diagnostic companion to
# check_quality()
# =====================================================================
def test_check_grading_uniform_mesh_ratio_near_one():
    from fea_engine.mesh import rectangle_mesh
    m = rectangle_mesh(1.0, 1.0, 10, 10)
    ratio = m.check_grading(verbose=False)
    assert ratio == pytest.approx(1.0, abs=1e-9)


def test_check_grading_detects_growth_ratio_violation():
    from fea_engine.mesh import hole_in_rectangle_mesh_graded
    m = hole_in_rectangle_mesh_graded(1.0, 1.0, (0.5, 0.5), R=0.05, nr=6,
                                       ntheta=8, h_far=0.2, growth_ratio=1.2)
    realized = m.check_grading(verbose=False)
    assert realized > 1.0  # graded mesh is, by construction, not uniform
    # Passing a cap tighter than the realized ratio must fail; looser must pass.
    assert m.check_grading(growth_ratio_cap=realized * 2, verbose=False) is True
    assert m.check_grading(growth_ratio_cap=max(realized - 0.1, 1.001), verbose=False) is False


def test_check_grading_3d_hex_mesh_now_supported():
    """Wave 0 item 8 (docs/consolidated_future_roadmap.md) extended
    check_grading() to 3-D -- this used to unconditionally raise
    NotImplementedError for ANY 3-D mesh; now a Hex8 mesh (extrude_mesh's
    output) is one of the two supported 3-D topologies and gets a real
    answer instead. See tests/test_grading_3d.py for the full Wave 0
    item 8 validation (face-table correctness, Tet4 support, growth-
    ratio detection, and the still-correctly-raising unsupported case)."""
    from fea_engine.mesh import rectangle_mesh, extrude_mesh
    m2d = rectangle_mesh(1.0, 1.0, 3, 3)
    m3d = extrude_mesh(m2d, 1.0, 2)
    ratio = m3d.check_grading(verbose=False)
    assert ratio == pytest.approx(1.0, abs=1e-9)   # uniform box mesh
