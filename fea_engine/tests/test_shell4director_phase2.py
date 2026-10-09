# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_shell4director_phase2.py -- validation for Wave 4 item 47,
Phase 2 (docs/director_based_shell_element_roadmap.md Section 5,
steps 3-6): `elements.shells_director.Shell4Director`'s LINEAR (u=0)
stiffness, independently assembled from director kinematics.

Four lines of evidence:

1. test_linearized_director_rotation_matches_bend_sign_convention --
   the ISOLATED check of Shell4Director's own central derivation
   (theta x t reduces to Quad4MindlinPlate's (betax, betay)
   convention): checks `linearized_director_rotation()` directly
   against the algebraic prediction `theta_y*e1 - theta_x*e2`, for a
   general local frame (not just the axis-aligned identity case, which
   couldn't distinguish a sign error from a correct formula the same
   way the real _BEND_SIGN bug this checks against was invisible on
   axis-aligned meshes -- see shells.py's own comment on that bug).

2. test_stiffness_matches_shell4mitc_axis_aligned -- the basic gate:
   for a simple, axis-aligned reference quad, Shell4Director.stiffness()
   equals Shell4MITC.stiffness() exactly (both reuse the identical
   sub-formulas, so exact equality -- not merely "close" -- is the
   correct expectation here, and is what's checked).

3. test_stiffness_matches_shell4mitc_general_orientation -- THE
   DECISIVE test, mirroring the real _BEND_SIGN bug's own discovery
   method (shells.py's comment: "invisible for any axis-aligned
   element... produces severely wrong stiffness the moment neighboring
   elements have different local frame orientations"): the SAME check,
   run on a quad rotated by a random 3-D rotation (general orientation,
   not axis-aligned) and translated away from the origin. If
   Shell4Director's own index/sign assembly had the same class of bug
   the real _BEND_SIGN fix corrected, THIS is the test that would catch
   it, not test 2.

4. test_stiffness_matches_shell4mitc_distorted_quad -- same check
   again on a non-rectangular (trapezoidal) quad, confirming the match
   isn't an artifact of the SPECIFIC rectangular geometry used in
   tests 2-3.

5. test_reference_directors_are_the_flat_normal -- sanity check that
   `reference_directors()` returns the element's own flat-facet normal
   at every node (Phase 2's flat-reference scope, per the roadmap
   doc), consistent with `Shell4MITC._local_frame_and_coords()`'s own
   e3.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import Material, D_shell
from fea_engine.elements.shells import Shell4MITC
from fea_engine.elements.shells_director import Shell4Director, linearized_director_rotation


def _material():
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    h = 0.005
    return D_shell(mat, h), h


RECT = np.array([[0.875, 0., 0.], [1., 0., 0.], [1., 0.1, 0.], [0.875, 0.1, 0.]])
TRAPEZOID = np.array([[0., 0., 0.], [1.0, 0., 0.], [0.8, 0.6, 0.], [0.1, 0.55, 0.]])


def _random_rigid_transform(seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    t = rng.normal(scale=2.0, size=3)
    return Q, t


@pytest.mark.parametrize("theta", [
    np.array([0.02, -0.015, 0.05]),
    np.array([-0.3, 0.1, 0.0]),
    np.array([0.0, 0.0, 0.4]),
])
def test_linearized_director_rotation_matches_bend_sign_convention(theta):
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    e1, e2, e3 = Q[:, 0], Q[:, 1], Q[:, 2]

    theta_x = theta @ e1
    theta_y = theta @ e2

    lr = linearized_director_rotation(theta, e3)
    expected = theta_y * e1 - theta_x * e2
    assert np.allclose(lr, expected, atol=1e-12), (
        f"linearized_director_rotation disagrees with the betax<->+theta_y, "
        f"betay<->-theta_x convention: got {lr}, expected {expected}"
    )


def test_stiffness_matches_shell4mitc_axis_aligned():
    D, h = _material()
    sd, sm = Shell4Director(), Shell4MITC()
    Kd = sd.stiffness(RECT, D)
    Km = sm.stiffness(RECT, D)
    assert np.array_equal(Kd, Km), (
        f"max abs diff {np.max(np.abs(Kd - Km)):.3e} -- expected EXACT "
        f"equality, since both elements call the identical sub-formulas"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stiffness_matches_shell4mitc_general_orientation(seed):
    """THE decisive test -- see module docstring. A general 3-D
    orientation is exactly the condition under which the real
    _BEND_SIGN bug (shells.py's own account) went from invisible to a
    40x-1500x stiffness error."""
    D, h = _material()
    sd, sm = Shell4Director(), Shell4MITC()
    Q, t = _random_rigid_transform(seed)
    X = RECT @ Q.T + t
    Kd = sd.stiffness(X, D)
    Km = sm.stiffness(X, D)
    rel = np.max(np.abs(Kd - Km)) / max(np.max(np.abs(Km)), 1e-30)
    assert rel < 1e-10, (
        f"seed {seed}: general-orientation stiffness disagrees, "
        f"relative {rel:.3e} -- this is exactly the failure mode an "
        f"axis-aligned-only test would miss"
    )


def test_stiffness_matches_shell4mitc_distorted_quad():
    D, h = _material()
    sd, sm = Shell4Director(), Shell4MITC()
    Kd = sd.stiffness(TRAPEZOID, D)
    Km = sm.stiffness(TRAPEZOID, D)
    rel = np.max(np.abs(Kd - Km)) / max(np.max(np.abs(Km)), 1e-30)
    assert rel < 1e-10, f"distorted quad: relative diff {rel:.3e}"


def test_reference_directors_are_the_flat_normal():
    sd = Shell4Director()
    t_nodes, e1_0, e2_0, e3_0, local = sd.reference_directors(RECT)
    assert t_nodes.shape == (4, 3)
    for a in range(4):
        assert np.allclose(t_nodes[a], e3_0)
    assert np.allclose(e3_0, np.array([0., 0., 1.]))
