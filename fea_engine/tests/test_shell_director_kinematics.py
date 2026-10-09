# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_shell_director_kinematics.py -- validation for Wave 4 item 47,
Phase 1 (docs/director_based_shell_element_roadmap.md Sections 1-2):
`elements/shells_director.py`'s two isolated primitives for the new
director-based shell element --

1. `director_update()` -- the director rotation kinematics.
2. `_polar_decomposition_angle_2x2()` / `drilling_angle_from_tangents()`
   -- the closed-form, iteration-free in-plane drilling angle.

Per the roadmap doc's own build order, this file validates ONLY these
two primitives in isolation -- no full element exists yet (Phase 2/3).

Six lines of evidence:

1. test_director_update_null_direction_is_exact -- the headline
   structural claim: rotating a director about ITS OWN axis leaves it
   EXACTLY unchanged (not merely small), at every angle 0.5-90 degrees,
   for several distinct director orientations. This is the precise
   reason a director-based shell needs no drilling stiffness.

2. test_director_update_small_rotation_matches_cross_product -- for a
   small rotation vector, `director_update()` matches the standard
   first-order rotation formula `t' ~ t + theta x t` to O(theta^2),
   confirming this isn't accidentally correct only at the exact-zero
   case above.

3. test_director_update_general_rotation_matches_known_result -- a
   90-degree rotation about an axis PERPENDICULAR to the director
   sends it to a specific, hand-computable other direction -- a sanity
   check the null-direction test alone can't provide (that test only
   confirms the identity case, not that a genuine rotation is applied
   correctly for a case where it SHOULD change).

4. test_polar_decomposition_angle_matches_svd_ground_truth -- the
   closed-form `psi = atan2(c-b, a+d)` formula is checked, not assumed,
   against `numpy.linalg.svd`'s own independent polar decomposition,
   across 200 random combinations of rotation angle (0.5-170 degrees)
   and anisotropic stretch magnitude -- an actual cross-check against a
   different algorithm, to machine precision.

5. test_drilling_angle_exact_for_pure_inplane_rigid_rotation -- the
   headline claim for the SECOND primitive: for a rigid rotation about
   the element's own out-of-plane axis (e3_0) by a KNOWN angle, at any
   magnitude 0.5-90 degrees (both signs), `drilling_angle_from_tangents()`
   recovers that EXACT angle -- the closed-form analogue of building
   block A's own rigid-tilt exactness standard, but here via a single
   `atan2`, no Newton iteration.

6. test_drilling_angle_zero_at_reference_state -- the trivial but
   necessary sanity check: zero displacement gives exactly zero
   drilling angle.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine.elements.shells_director import (
    director_update,
    _polar_decomposition_angle_2x2,
    drilling_angle_from_tangents,
)

RECT = np.array([[0.875, 0., 0.], [1., 0., 0.], [1., 0.1, 0.], [0.875, 0.1, 0.]])


@pytest.mark.parametrize("angle_deg", [0.5, 6.0, 30.0, 60.0, 90.0])
@pytest.mark.parametrize("t_ref", [
    np.array([0., 0., 1.]),
    np.array([0., 1., 0.]),
    np.array([1., 0., 0.]),
    np.array([1., 1., 1.]) / np.sqrt(3.0),
])
def test_director_update_null_direction_is_exact(angle_deg, t_ref):
    angle = np.radians(angle_deg)
    theta = angle * t_ref   # rotation ABOUT the director's own axis
    t_new = director_update(t_ref, theta)
    max_dev = np.max(np.abs(t_new - t_ref))
    assert max_dev < 1e-12, (
        f"rotation about own axis ({angle_deg}deg) should leave the director "
        f"EXACTLY unchanged, got deviation {max_dev:.3e}"
    )


def test_director_update_small_rotation_matches_cross_product():
    t_ref = np.array([0., 0., 1.])
    rng = np.random.default_rng(5)
    theta = rng.normal(scale=1e-3, size=3)
    t_new = director_update(t_ref, theta)
    approx = t_ref + np.cross(theta, t_ref)
    err = np.linalg.norm(t_new - approx)
    theta_norm = np.linalg.norm(theta)
    assert err < 5 * theta_norm ** 2, (
        f"small-rotation deviation from the first-order cross-product formula "
        f"({err:.3e}) should be O(theta^2)~{theta_norm**2:.3e}"
    )


def test_director_update_general_rotation_matches_known_result():
    # 90deg about the x-axis sends the z-director to -y (right-hand rule:
    # rotating [0,0,1] by +90deg about [1,0,0] gives [0,-1,0]).
    t_ref = np.array([0., 0., 1.])
    theta = (np.pi / 2) * np.array([1., 0., 0.])
    t_new = director_update(t_ref, theta)
    expected = np.array([0., -1., 0.])
    assert np.linalg.norm(t_new - expected) < 1e-12, (
        f"expected {expected}, got {t_new}"
    )


def test_polar_decomposition_angle_matches_svd_ground_truth():
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        angle = np.radians(rng.uniform(0.5, 170.0))
        R = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
        A = rng.normal(size=(2, 2))
        U = A @ A.T + 2.0 * np.eye(2)   # random SPD stretch
        F = R @ U

        psi = _polar_decomposition_angle_2x2(F)

        U_svd, S, Vt = np.linalg.svd(F)
        R_svd = U_svd @ Vt
        if np.linalg.det(R_svd) < 0:
            U_svd[:, -1] *= -1
            R_svd = U_svd @ Vt
        psi_svd = np.arctan2(R_svd[1, 0], R_svd[0, 0])

        # angular difference, wrapped to (-pi, pi]
        err = abs(np.angle(np.exp(1j * (psi - psi_svd))))
        max_err = max(max_err, err)

    assert max_err < 1e-9, (
        f"closed-form polar-decomposition angle disagrees with the "
        f"independent SVD-based ground truth by up to {max_err:.3e} rad "
        f"across 200 random rotation/stretch trials"
    )


@pytest.mark.parametrize("angle_deg", [0.5, 6.0, 30.0, 60.0, 90.0, -30.0, -75.0])
def test_drilling_angle_exact_for_pure_inplane_rigid_rotation(angle_deg):
    angle = np.radians(angle_deg)
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0.],
                    [np.sin(angle), np.cos(angle), 0.],
                    [0., 0., 1.]])
    centroid = RECT.mean(axis=0)
    x_current = (RECT - centroid) @ Rz.T + centroid

    psi, e1_0, e2_0, e3_0 = drilling_angle_from_tangents(RECT, x_current)
    err_deg = abs(np.degrees(psi) - angle_deg)
    assert err_deg < 1e-9, (
        f"in-plane rigid rotation {angle_deg}deg: recovered drilling angle "
        f"{np.degrees(psi):.6f}deg, error {err_deg:.3e}deg"
    )
    # frame sanity: e3_0 should be the flat element's own normal ([0,0,1]
    # for this planar reference mesh), e1_0/e2_0 unit and mutually
    # orthogonal.
    assert np.linalg.norm(e3_0 - np.array([0., 0., 1.])) < 1e-12
    assert abs(np.linalg.norm(e1_0) - 1.0) < 1e-12
    assert abs(np.linalg.norm(e2_0) - 1.0) < 1e-12
    assert abs(e1_0 @ e2_0) < 1e-12


def test_drilling_angle_zero_at_reference_state():
    psi, _, _, _ = drilling_angle_from_tangents(RECT, RECT)
    assert abs(psi) < 1e-12
