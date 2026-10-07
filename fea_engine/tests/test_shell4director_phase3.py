"""
test_shell4director_phase3.py -- validation for Wave 4 item 47, Phase 3
(docs/director_based_shell_element_roadmap.md Section 5, steps 7-8):
`elements.shells_director.Shell4Director`'s full nonlinear
`internal_force()`/`tangent_stiffness()`, isolated from the (separately
tested, slow) multi-element/elastica benchmark in
test_shell4director_elastica.py.

Six lines of evidence, deliberately fast (no Newton solve anywhere in
this file -- every check is a handful of direct internal_force()/
tangent_stiffness()/strain_energy() calls):

1. test_strain_energy_and_force_zero_at_reference -- U(0)=0 exactly;
   f_int(0) is at the FD noise floor (~1e-3, matching the h=1e-6
   central-difference precision measured directly, NOT exactly zero in
   floating point but consistent with the analytic fact that dU/du=0
   at u=0 since every strain measure is itself zero there).

2. test_tangent_matches_linear_stiffness_membrane_and_drilling_blocks
   -- the LINEAR (Phase 2) `stiffness()` and the FD-based nonlinear
   `tangent_stiffness()` at u=0 must agree, block by block, EXCEPT
   where Phase 3's own docstring documents a deliberate, understood
   divergence (see #3 below). Checked here: the membrane block agrees
   to ~1e-10 relative (FD precision), the drilling block agrees
   EXACTLY (unaffected by anything Phase 3 changed).

3. test_tangent_bend_shear_divergence_is_bounded_and_expected -- the
   bend+shear block's divergence from Phase 2's `stiffness()` (found
   necessary to fix a real rigid-rotation-invariance defect in the
   REUSED linear Bs -- see `strain_energy()`'s own "SHEAR -- A REAL,
   FOUND SIGN DIVERGENCE" docstring paragraph) is bounded (a few
   percent of the block's own max entry, not an order-of-magnitude
   blowup) -- a sanity check that the divergence is exactly the KNOWN,
   understood one, not a symptom of a broader bug.

4. test_membrane_and_bending_energy_exact_zero_under_rigid_rotation --
   THE decisive structural claim (roadmap doc step 8), split into its
   two exactly-provable pieces: membrane strain energy and bending
   (curvature) strain energy are EXACTLY zero (~1e-20 or below, machine
   noise) for a rigid rotation of ANY magnitude (0.5-90 degrees) about
   ANY axis (in-plane, out-of-plane, general) -- not merely small,
   proven algebraically in `strain_energy()`'s own docstring (`g = R@G`
   for a rigid rotation, R orthogonal) and confirmed numerically here.

5. test_shear_energy_exact_zero_under_rigid_rotation -- the same claim
   for the (harder -- see docstring history) shear term specifically,
   via `_mitc4_shear_nonlinear()`'s own rigid-rotation-exact
   construction; isolated from membrane/bending so a regression in
   JUST the shear fix would be caught here specifically.

6. test_internal_force_zero_under_rigid_rotation_any_axis -- the same
   claim at the FULL internal_force() level (not just the scalar
   energy), parametrized across magnitude AND axis, checked against the
   SAME FD noise floor #1 established at u=0 (not literally zero, since
   internal_force() is itself a finite-difference derivative, but
   FLAT -- not growing with rotation magnitude the way a genuine
   physics violation would) -- the practical form of the decisive claim
   a caller of internal_force() actually observes.
"""
import numpy as np
import pytest

from fea_engine import Material, D_shell
from fea_engine.elements.shells_director import Shell4Director
from fea_engine.elements.shells import Shell4MITCCorotational, _MEM, _BEND, _DRILL

RECT = np.array([[0.875, 0., 0.], [1., 0., 0.], [1., 0.1, 0.], [0.875, 0.1, 0.]])


def _material():
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    h = 0.005
    return D_shell(mat, h), h


def _rigid_rotation_u(elem_coords, axis, angle_deg):
    """A pure single-element rigid rotation by `angle_deg` about `axis`
    (through the element's own centroid) -- the SAME construction used
    throughout this wave's rigid-rotation exactness checks (e.g.
    building block A's `test_shell_corotational_exact_rotation_
    extraction.py`), reused here rather than re-derived."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta_vec = np.radians(angle_deg) * axis
    R = Shell4MITCCorotational._exp_map(theta_vec)
    centroid = elem_coords.mean(axis=0)
    u = np.zeros(24)
    for a in range(4):
        x_new = centroid + R @ (elem_coords[a] - centroid)
        u[6 * a:6 * a + 3] = x_new - elem_coords[a]
        u[6 * a + 3:6 * a + 6] = theta_vec
    return u


_AXES = {
    "e3 (drilling)": np.array([0., 0., 1.]),
    "e1 (in-plane)": np.array([1., 0., 0.]),
    "e2 (in-plane)": np.array([0., 1., 0.]),
    "general": np.array([0.3, 0.5, 0.8]),
}


def test_strain_energy_and_force_zero_at_reference():
    D, h = _material()
    sd = Shell4Director()
    u0 = np.zeros(24)
    assert sd.strain_energy(RECT, u0, D) == 0.0
    f0 = sd.internal_force(RECT, u0, D)
    # FD noise floor at h=1e-6, measured directly: ~2.6e-3 for this
    # element/material -- NOT exactly zero (internal_force() is itself
    # a finite difference), but far below any real force scale (e.g.
    # ~23000 N at a modest w=1e-4 perturbation, ~1e-7 relative).
    assert np.max(np.abs(f0)) < 0.01


def test_tangent_matches_linear_stiffness_membrane_and_drilling_blocks():
    D, h = _material()
    sd = Shell4Director()
    u0 = np.zeros(24)
    K_fd = sd.tangent_stiffness(RECT, u0, D)
    K_lin = sd.stiffness(RECT, D)
    diff = K_fd - K_lin
    scale = np.max(np.abs(K_lin))

    mem_rel = np.max(np.abs(diff[np.ix_(_MEM, _MEM)])) / np.max(np.abs(K_lin[np.ix_(_MEM, _MEM)]))
    assert mem_rel < 1e-6, f"membrane block diverges from Phase 2 stiffness(): {mem_rel:.3e}"

    drill_diff = np.max(np.abs(diff[np.ix_(_DRILL, _DRILL)]))
    assert drill_diff == 0.0, f"drilling block should be byte-identical (unchanged): diff={drill_diff}"


def test_tangent_bend_shear_divergence_is_bounded_and_expected():
    """See `strain_energy()`'s own docstring -- the bend+shear block IS
    expected to diverge from Phase 2's linear `stiffness()` (the fixed
    shear formula is rigid-rotation-exact; the reused linear `Bs` it
    replaces is not), but the divergence must be a bounded, few-percent
    effect confined to shear cross-coupling -- not an order-of-magnitude
    symptom of an unrelated bug."""
    D, h = _material()
    sd = Shell4Director()
    u0 = np.zeros(24)
    K_fd = sd.tangent_stiffness(RECT, u0, D)
    K_lin = sd.stiffness(RECT, D)
    diff = K_fd - K_lin
    scale = np.max(np.abs(K_lin))
    bend_rel = np.max(np.abs(diff[np.ix_(_BEND, _BEND)])) / scale
    assert 1e-4 < bend_rel < 0.10, (
        f"bend+shear block divergence ({bend_rel:.3e} relative to the "
        f"full matrix's own max entry) is outside the expected 'small, "
        f"understood shear-sign-fix' range -- either the fix regressed "
        f"(near 0, membrane-only-consistency signature) or something "
        f"unrelated broke (order-of-magnitude larger)")


@pytest.mark.parametrize("angle_deg", [0.5, 5.0, 30.0, 60.0, 90.0])
@pytest.mark.parametrize("axis_name", list(_AXES.keys()))
def test_membrane_and_bending_energy_exact_zero_under_rigid_rotation(angle_deg, axis_name):
    D, h = _material()
    sd = Shell4Director()
    u = _rigid_rotation_u(RECT, _AXES[axis_name], angle_deg)

    X_ref = RECT
    u_nodes = u.reshape(4, 6)
    u_trans = u_nodes[:, 0:3]
    theta_nodes = u_nodes[:, 3:6]
    x_current = X_ref + u_trans
    t_nodes0, e1_0, e2_0, e3_0, local = sd.reference_directors(X_ref)
    t_nodes = sd._current_directors(theta_nodes, e3_0)
    w_a = u_trans @ e3_0
    betax_a = t_nodes @ e1_0
    betay_a = t_nodes @ e2_0
    dof_bs = np.zeros(12)
    for a in range(4):
        dof_bs[3 * a + 0] = w_a[a]
        dof_bs[3 * a + 1] = betax_a[a]
        dof_bs[3 * a + 2] = betay_a[a]

    from fea_engine.elements.base import jacobian, gauss_product
    Dm, Db, Ds, hh = D
    pts, wts = gauss_product(sd.gauss_order, sd.dim)
    Um = Ub = 0.0
    for p, wgt in zip(pts, wts):
        _, dN_nat = sd._mitc._membrane.shape_and_derivs(p)
        J, detJ = jacobian(dN_nat, local)
        dN_g = np.linalg.solve(J, dN_nat)
        dNdx, dNdy = dN_g[0], dN_g[1]
        g1 = dNdx @ x_current
        g2 = dNdy @ x_current
        G1 = dNdx @ X_ref
        G2 = dNdy @ X_ref
        eps_mem = np.array([0.5 * (g1 @ g1 - G1 @ G1), 0.5 * (g2 @ g2 - G2 @ G2), g1 @ g2 - G1 @ G2])
        Um += 0.5 * (eps_mem @ Dm @ eps_mem) * detJ * wgt * hh
        Bb, _, detJ_b = sd._mitc._plate._Bb_Bs(p, local)
        kappa = Bb @ dof_bs
        Ub += 0.5 * (kappa @ Db @ kappa) * detJ_b * wgt

    assert Um < 1e-15, f"angle={angle_deg} axis={axis_name}: membrane energy {Um:.3e} should be exactly 0"
    assert Ub < 1e-15, f"angle={angle_deg} axis={axis_name}: bending energy {Ub:.3e} should be exactly 0"


@pytest.mark.parametrize("angle_deg", [0.5, 5.0, 30.0, 60.0, 90.0])
@pytest.mark.parametrize("axis_name", list(_AXES.keys()))
def test_shear_energy_exact_zero_under_rigid_rotation(angle_deg, axis_name):
    D, h = _material()
    Dm, Db, Ds, hh = D
    sd = Shell4Director()
    u = _rigid_rotation_u(RECT, _AXES[axis_name], angle_deg)

    X_ref = RECT
    u_nodes = u.reshape(4, 6)
    u_trans = u_nodes[:, 0:3]
    theta_nodes = u_nodes[:, 3:6]
    x_current = X_ref + u_trans
    t_nodes0, e1_0, e2_0, e3_0, local = sd.reference_directors(X_ref)
    t_nodes = sd._current_directors(theta_nodes, e3_0)

    from fea_engine.elements.base import gauss_product
    pts, wts = gauss_product(sd.gauss_order, sd.dim)
    Us = 0.0
    for p, wgt in zip(pts, wts):
        gamma, detJ_s = sd._mitc4_shear_nonlinear(p, X_ref, x_current, t_nodes0, t_nodes, local)
        Us += 0.5 * (gamma @ Ds @ gamma) * detJ_s * wgt

    assert Us < 1e-15, f"angle={angle_deg} axis={axis_name}: shear energy {Us:.3e} should be exactly 0"


@pytest.mark.parametrize("angle_deg", [0.5, 30.0, 90.0])
@pytest.mark.parametrize("axis_name", list(_AXES.keys()))
def test_internal_force_zero_under_rigid_rotation_any_axis(angle_deg, axis_name):
    """THE decisive claim at the level a real caller sees: internal_
    force() must not grow with rotation magnitude for a pure rigid
    rotation -- checked against the SAME FD noise floor (~0.01)
    test_strain_energy_and_force_zero_at_reference established at u=0,
    across every tested angle 0.5-90 degrees and every axis. Drilling
    DOFs are excluded from the max (the drilling regularization IS a
    real, intentional, nonzero-for-drilling-rotation stiffness -- see
    Phase 2's own 'ON THE DRILLING DOF' docstring -- so a nonzero
    drilling-DOF force under an e3-axis rotation is expected, not a
    violation of this claim)."""
    D, h = _material()
    sd = Shell4Director()
    u = _rigid_rotation_u(RECT, _AXES[axis_name], angle_deg)
    f = sd.internal_force(RECT, u, D)
    for idx in [5, 11, 17, 23]:
        f[idx] = 0.0
    assert np.max(np.abs(f)) < 0.01, (
        f"angle={angle_deg} axis={axis_name}: spurious rigid-rotation force "
        f"{np.max(np.abs(f)):.3e} exceeds the FD noise floor -- indicates a "
        f"genuine (not merely numerical) rigid-rotation-invariance defect"
    )
