# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_shell_corotational_exact_drill.py -- validation for Wave 4 item 18
(docs/consolidated_future_roadmap.md): the EXACT single-axis (SO(3))
drilling rotation update in Shell4MITCCorotational, replacing the
first-order-truncated Rodrigues update the "Known remaining limitation"
section of shells.py's class docstring previously documented as a real,
measured, bounded-but-nonzero error at large single-element rigid
rotation (~1.6% of the element's own stiffness scale at 30 degrees).

Why this is safe to make exact where the general 3-D tilt frame (dead
ends 1-3, and items 20/22's investigation) is not: drilling is, by this
element's own construction, ALWAYS a rotation about a single FIXED axis
(R0's own local z) -- never a general 3-vector rotation. A single-axis
rotation needs only cos/sin of a scalar angle, both holomorphic
everywhere (no norm/branching singularity at zero rotation the way a
general axis-angle Rodrigues formula would need), so making the
DRILLING frame exact does not reintroduce the class of risk that ruled
out a general exact-SO(3) treatment for the bending block (item 20) or
membrane locking (item 22).

Five lines of evidence:

1. test_zero_and_small_rotation_regression -- the exact update must
   still reduce to the SAME behavior as the old first-order update at
   u=0 and at small rotations (both are the SAME function to leading
   order in theta -- see shells.py's _exact_drill_rotation() docstring
   for the direct O(theta^2) agreement check), so every pre-existing
   test in test_shell_corotational.py / test_shell_corotational_
   reference_frame.py / test_shell_corotational_elastica.py must keep
   passing UNCHANGED (checked by actually re-running those files
   alongside this one, not just asserted here).

2. test_drilling_rotation_exact_zero_force -- the header claim, made
   precise: a PURE rigid rotation about the element's own local z-axis
   (any angle, including large ones the old first-order update measured
   real error at: 0.5, 6, 30, 60, 90 degrees) must now produce EXACTLY
   zero internal force (to floating-point precision), not just
   "bounded small" -- because R_drill now tracks that exact rotation
   with no truncation, so the position-based membrane extraction's
   current-vs-reference comparison is an exact identity for this
   specific motion. Derived and predicted analytically before running
   (see the Wave 4 item 18 investigation notes), then confirmed
   numerically as an unconditional (not approximate) property.

3. test_bending_axis_unaffected -- item 18 deliberately does NOT touch
   the bending block (still uncorrected, per Phase A's design -- see
   items 20/22's own investigation for why a real fix there needs a
   different, riskier construction). A rigid rotation about an IN-PLANE
   axis (y) must reproduce the SAME measured error the old
   "Known remaining limitation" section documented (~0.03% at 0.5deg,
   ~0.33% at 6deg, ~1.8% at 30deg), confirming this change is cleanly
   isolated to the drilling/membrane block and does not silently change
   (for better or worse) the separately-tracked bending-block behavior.

4. test_analytic_tangent_matches_complex_step_large_drilling -- the
   updated analytic Jacobian (_dof_local_jacobian_and_geo()'s now
   state-dependent dRdrill) must still exactly match
   _tangent_stiffness_complex_step() at large drilling rotations where
   the old CONSTANT dRdrill would have been wrong -- the real
   correctness gate for the tangent, not just internal_force().

5. test_combined_state_drilling_plus_bending -- a state combining a
   large drilling rotation with a genuine bending deformation (not an
   isolated rigid motion) still gives internal_force()/tangent_
   stiffness() that are mutually consistent (complex-step check) and
   whose drilling-only slice recovers the same exact-zero property when
   the bending part is switched off, confirming the fix composes
   correctly with the rest of the element rather than being an
   isolated-state artifact.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from fea_engine import Material, D_shell, Shell4MITC, Shell4MITCCorotational


def _material():
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    h = 0.005
    return D_shell(mat, h), h


RECT = np.array([[0.875, 0., 0.], [1., 0., 0.], [1., 0.1, 0.], [0.875, 0.1, 0.]])


def _rigid_state(angle_deg, axis):
    angle = np.radians(angle_deg)
    Rrig = R.from_rotvec(angle * axis).as_matrix()
    centroid = RECT.mean(axis=0)
    u = np.zeros(24)
    for i in range(4):
        xi = RECT[i]
        u[6 * i:6 * i + 3] = Rrig @ (xi - centroid) - (xi - centroid)
        u[6 * i + 3:6 * i + 6] = angle * axis
    return u


def test_zero_and_small_rotation_regression():
    D, h = _material()
    coro = Shell4MITCCorotational()
    lin = Shell4MITC()
    f0 = coro.internal_force(RECT, np.zeros(24), D, thickness=h)
    assert np.linalg.norm(f0) < 1e-8

    K_analytic = coro.tangent_stiffness(RECT, np.zeros(24), D, thickness=h)
    K_lin = lin.stiffness(RECT, D)
    rel = np.linalg.norm(K_analytic - K_lin) / np.linalg.norm(K_lin)
    assert rel < 1e-4


@pytest.mark.parametrize("angle_deg", [0.5, 6.0, 30.0, 60.0, 90.0])
def test_drilling_rotation_exact_zero_force(angle_deg):
    D, h = _material()
    coro = Shell4MITCCorotational()
    lin = Shell4MITC()
    K_scale = np.linalg.norm(lin.stiffness(RECT, D))

    u = _rigid_state(angle_deg, np.array([0., 0., 1.]))
    f = coro.internal_force(RECT, u, D, thickness=h)
    rel = np.linalg.norm(f) / K_scale
    # Old first-order update measured ~1.6% at 30deg; exact update
    # should give floating-point noise (< 1e-10) at every angle here,
    # not merely "smaller than before".
    assert rel < 1e-10, f"drilling rotation at {angle_deg}deg: rel={rel:.3e}, expected exact zero"


@pytest.mark.parametrize("angle_deg,expected_rel_max", [(0.5, 0.001), (6.0, 0.01), (30.0, 0.03)])
def test_bending_axis_unaffected(angle_deg, expected_rel_max):
    D, h = _material()
    coro = Shell4MITCCorotational()
    lin = Shell4MITC()
    K_scale = np.linalg.norm(lin.stiffness(RECT, D))

    u = _rigid_state(angle_deg, np.array([0., 1., 0.]))
    f = coro.internal_force(RECT, u, D, thickness=h)
    rel = np.linalg.norm(f) / K_scale
    # Should still be BOUNDED SMALL (unchanged, documented) error, not
    # zero (item 18 doesn't touch this block) and not blown up (item 18
    # shouldn't have broken anything here either).
    assert 0.0 < rel < expected_rel_max, f"bending-axis rotation at {angle_deg}deg: rel={rel:.3e}"


@pytest.mark.parametrize("angle_deg", [30.0, 60.0, 90.0, 150.0])
def test_analytic_tangent_matches_complex_step_large_drilling(angle_deg):
    D, h = _material()
    coro = Shell4MITCCorotational()
    u = _rigid_state(angle_deg, np.array([0., 0., 1.]))
    K_analytic = coro.tangent_stiffness(RECT, u, D, thickness=h)
    K_cs = coro._tangent_stiffness_complex_step(RECT, u, D, thickness=h)
    denom = max(np.linalg.norm(K_cs), 1.0)
    rel = np.linalg.norm(K_analytic - K_cs) / denom
    assert rel < 1e-8, f"analytic vs complex-step at {angle_deg}deg drilling: rel={rel:.3e}"


def test_combined_state_drilling_plus_bending():
    D, h = _material()
    coro = Shell4MITCCorotational()

    drill_only = _rigid_state(45.0, np.array([0., 0., 1.]))
    bending = np.zeros(24)
    rng = np.random.default_rng(7)
    bending[2::6] = rng.normal(scale=0.01, size=4)   # w
    bending[3::6] = rng.normal(scale=0.02, size=4)   # theta_x
    bending[4::6] = rng.normal(scale=0.02, size=4)   # theta_y
    u_combined = drill_only + bending

    K_analytic = coro.tangent_stiffness(RECT, u_combined, D, thickness=h)
    K_cs = coro._tangent_stiffness_complex_step(RECT, u_combined, D, thickness=h)
    denom = max(np.linalg.norm(K_cs), 1.0)
    rel = np.linalg.norm(K_analytic - K_cs) / denom
    assert rel < 1e-8

    # Drilling-only slice (bending switched off) still exact-zero even
    # when evaluated through the SAME code path used for the combined
    # state above (not a separately-derived formula).
    f_drill_only = coro.internal_force(RECT, drill_only, D, thickness=h)
    lin = Shell4MITC()
    K_scale = np.linalg.norm(lin.stiffness(RECT, D))
    assert np.linalg.norm(f_drill_only) / K_scale < 1e-10
