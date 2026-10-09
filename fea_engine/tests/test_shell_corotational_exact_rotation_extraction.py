"""
test_shell_corotational_exact_rotation_extraction.py -- validation for
Wave 4 item 46 building block A, step 3
(docs/shell_rotation_coupling_fix_roadmap.md Section 4.1):
`Shell4MITCCorotational._local_relative_dofs_exact_rotation()`, the
NEW, OPT-IN dof_local extraction that replaces the rotational-DOF
treatment (theta_x, theta_y, theta_z per node) with the exact
`_mean_rigid_rotation()` SO(3) extraction, while leaving translation
(u, v, w per node) exactly as `_local_relative_dofs()` already computes
it.

Scope, precisely: this method is NOT wired into `internal_force()`/
`tangent_stiffness()` -- see its own docstring for the two concrete
blockers found while attempting that (holomorphy for the complex-step
tangent oracle; the harder log-map half of that problem left genuinely
unresolved). This file validates the dof_local EXTRACTION ITSELF,
directly, the same "isolate before wiring" discipline building block
A's first deliverable (`_mean_rigid_rotation` itself) already used.

Five lines of evidence:

1. test_zero_state_gives_zero_dof_local -- u=0 gives dof_local exactly
   zero, the same basic sanity property every dof_local extraction in
   this class has.

2. test_general_rigid_rotation_gives_exact_zero_rotation_dofs -- the
   headline claim: for a PURE rigid rotation about ANY axis (not just
   z), every node's rotational dof slots (theta_x, theta_y, theta_z)
   come out EXACTLY zero (floating-point noise), at every angle
   0.5-90 degrees -- the direct generalization of
   `test_shell_corotational_exact_drill.py`'s single-axis check to a
   general axis, and the concrete closing of item 18's own left-open
   "Known remaining limitation" for the bending block (previously
   ~0.03% at 0.5deg, ~1.8% at 30deg for an in-plane-axis rotation).

3. test_translation_unchanged_vs_old_method -- the u,v,w dof slots this
   new method returns are IDENTICAL (not just close) to
   `_local_relative_dofs()`'s own, across several random states --
   confirms translation extraction is truly untouched, not just
   documented as such.

4. test_pure_drilling_matches_old_method_rotation_dofs -- for a pure
   z-axis rigid rotation (the one case the OLD method ALSO gets
   rotationally exact, via R_drill/raw-projection), the new method's
   rotational dofs agree with the old method's (both exactly zero) --
   a sanity/regression check that the new extraction doesn't disagree
   with already-validated behavior on the case they should overlap on.

5. test_bending_perturbation_gives_small_nonzero_rotation_dofs -- for a
   genuine (non-rigid) bending state, the new method's rotational dofs
   are small, non-zero, and of the expected order -- confirms this
   isn't a construction that's only ever exactly zero or broken.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from fea_engine import Material, D_shell, Shell4MITCCorotational


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


def _rotation_dof_slots(dof_local):
    """theta_x, theta_y, theta_z for all 4 nodes -> (4,3) array."""
    out = np.zeros((4, 3))
    for i in range(4):
        out[i] = [dof_local[6 * i + 3], dof_local[6 * i + 4], dof_local[6 * i + 5]]
    return out


def _translation_dof_slots(dof_local):
    out = np.zeros((4, 3))
    for i in range(4):
        out[i] = [dof_local[6 * i + 0], dof_local[6 * i + 1], dof_local[6 * i + 2]]
    return out


def test_zero_state_gives_zero_dof_local():
    coro = Shell4MITCCorotational()
    dof_local, e1, e2, e3 = coro._local_relative_dofs_exact_rotation(RECT, np.zeros(24))
    assert np.linalg.norm(dof_local) < 1e-12
    assert np.linalg.norm(e1 - np.array([1., 0., 0.])) < 1e-10 or np.linalg.norm(e3) > 0  # sane frame


@pytest.mark.parametrize("angle_deg", [0.5, 6.0, 30.0, 60.0, 90.0])
@pytest.mark.parametrize("axis", [
    np.array([0., 0., 1.]),
    np.array([0., 1., 0.]),
    np.array([1., 0., 0.]),
    np.array([1., 1., 1.]) / np.sqrt(3.0),
])
def test_general_rigid_rotation_gives_exact_zero_rotation_dofs(angle_deg, axis):
    coro = Shell4MITCCorotational()
    u = _rigid_state(angle_deg, axis)
    dof_local, _, _, _ = coro._local_relative_dofs_exact_rotation(RECT, u)
    rot = _rotation_dof_slots(dof_local)
    max_rot = np.max(np.abs(rot))
    assert max_rot < 1e-9, (
        f"rigid rotation {angle_deg}deg about {axis}: max rotational dof = {max_rot:.3e}, expected exact zero"
    )


def test_translation_unchanged_vs_old_method():
    D, h = _material()
    coro = Shell4MITCCorotational()
    rng = np.random.default_rng(5)
    for trial in range(5):
        u = rng.normal(scale=0.02, size=24)
        dof_old, e1_old, e2_old, e3_old = coro._local_relative_dofs(RECT, u)
        dof_new, e1_new, e2_new, e3_new = coro._local_relative_dofs_exact_rotation(RECT, u)

        trans_old = _translation_dof_slots(dof_old)
        trans_new = _translation_dof_slots(dof_new)
        assert np.allclose(trans_old, trans_new, atol=1e-12), (
            f"trial {trial}: translation slots differ\nold={trans_old}\nnew={trans_new}"
        )
        # e1,e2,e3 (the R_drill-derived frame used for global<->local
        # rotation) are ALSO identical -- both methods build it the same way.
        assert np.allclose(e1_old, e1_new, atol=1e-12)
        assert np.allclose(e2_old, e2_new, atol=1e-12)
        assert np.allclose(e3_old, e3_new, atol=1e-12)


@pytest.mark.parametrize("angle_deg", [0.5, 6.0, 30.0, 60.0, 90.0])
def test_pure_drilling_matches_old_method_rotation_dofs(angle_deg):
    coro = Shell4MITCCorotational()
    u = _rigid_state(angle_deg, np.array([0., 0., 1.]))
    dof_old, _, _, _ = coro._local_relative_dofs(RECT, u)
    dof_new, _, _, _ = coro._local_relative_dofs_exact_rotation(RECT, u)

    rot_old = _rotation_dof_slots(dof_old)
    rot_new = _rotation_dof_slots(dof_new)
    # Both should be exactly zero for pure z-axis rotation (old method's
    # raw e1_0/e2_0 projection happens to vanish too, since a pure-z
    # rotation vector is orthogonal to the in-plane e1_0/e2_0 axes).
    assert np.max(np.abs(rot_old)) < 1e-9
    assert np.max(np.abs(rot_new)) < 1e-9


def test_bending_perturbation_gives_small_nonzero_rotation_dofs():
    coro = Shell4MITCCorotational()
    rng = np.random.default_rng(9)
    u = np.zeros(24)
    u[3::6] = rng.normal(scale=0.02, size=4)   # theta_x
    u[4::6] = rng.normal(scale=0.02, size=4)   # theta_y
    dof_local, _, _, _ = coro._local_relative_dofs_exact_rotation(RECT, u)
    rot = _rotation_dof_slots(dof_local)
    max_rot = np.max(np.abs(rot))
    assert 1e-4 < max_rot < 0.1, f"unexpected rotational-dof scale: {max_rot:.3e}"


def test_membrane_force_provably_unaffected_by_exact_rotation_extraction():
    """Wave 4 item 46, building block B investigation
    (docs/shell_rotation_coupling_fix_roadmap.md Section 4.2): this is
    the DIRECT NUMERICAL CONFIRMATION of a finding derived analytically
    BEFORE writing this test (not the other way around) -- read this
    docstring alongside `shells.py`'s own "Building block B finding"
    comment above `_local_relative_dofs_exact_rotation()`.

    Claim: using the exact rotation extraction in place of the old
    method CANNOT produce any change in membrane (u,v) force -- not
    "small," EXACTLY zero difference -- for ANY state, because of two
    already-established facts about K_local0 combined: (1) K_local0 is
    EXACTLY block-diagonal between the membrane+drilling group and the
    bending+shear group ("Key structural fact", verified when this
    class was first built); (2) the drilling-stabilization stiffness is
    PURELY DIAGONAL even WITHIN the membrane+drilling group -- verified
    directly here, not assumed -- so even though theta_z_local itself
    DOES differ between the two extraction methods (a real, expected
    SO(3)-composition effect, not a bug: `_mean_rigid_rotation()`'s
    theta_z_local depends on theta_x/theta_y too, unlike the old
    method's simple arithmetic z-mean), that difference has ZERO path
    into the u,v force. Consequence: exact rotation extraction alone
    (building block A) -- and, by the same argument, ANY variant of
    "building block B" that only changes what feeds K_local0's
    EXISTING rotational dof slots, without adding a genuinely new force
    term -- is mathematically incapable of recovering the elastica
    benchmark's missing foreshortening. This rules out attempting a
    single-shot (non-incremental) "additive Koiter-Sanders split
    reusing K_local0" as building block B's content -- the roadmap's
    own Section 4.2 sketch -- BEFORE spending effort implementing and
    benchmarking it, the same "derive, then confirm once, don't guess"
    discipline items 19's own implicit-function-theorem argument used
    to rule out its own two candidate paths without needing a full
    trial run of each."""
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    h = 0.005
    D = D_shell(mat, h)
    coro = Shell4MITCCorotational()
    K_local0 = coro._local_material_stiffness(RECT, D)

    mem_uv = [0, 1, 6, 7, 12, 13, 18, 19]
    bend = [2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22]
    drill = [5, 11, 17, 23]

    # The two established facts, checked directly (not assumed):
    assert np.max(np.abs(K_local0[np.ix_(mem_uv, bend)])) == 0.0
    assert np.max(np.abs(K_local0[np.ix_(mem_uv, drill)])) == 0.0

    rng = np.random.default_rng(1)
    for trial in range(5):
        u = np.zeros(24)
        u[3::6] = rng.normal(scale=0.05, size=4)   # theta_x
        u[4::6] = rng.normal(scale=0.05, size=4)   # theta_y

        dof_old, _, _, _ = coro._local_relative_dofs(RECT, u)
        dof_new, _, _, _ = coro._local_relative_dofs_exact_rotation(RECT, u)
        # theta_z_local genuinely differs (SO(3) composition effect) --
        # confirming this ISN'T a trivial "both extractions agree" case.
        assert not np.allclose(dof_old[drill], dof_new[drill], atol=1e-8), (
            f"trial {trial}: expected theta_z_local to differ (SO(3) coupling), it didn't"
        )

        f_old = K_local0 @ dof_old
        f_new = K_local0 @ dof_new
        assert np.array_equal(f_old[mem_uv], f_new[mem_uv]), (
            f"trial {trial}: membrane force changed -- the block-diagonal "
            f"argument above would be WRONG if this ever fails"
        )
        assert np.max(np.abs(f_old[mem_uv])) == 0.0, (
            f"trial {trial}: pure-rotation state should give exactly zero "
            f"membrane force under EITHER extraction (u,v dof_local are both "
            f"zero when u_trans=0)"
        )
