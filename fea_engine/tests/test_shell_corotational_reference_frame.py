"""
test_shell_corotational_reference_frame.py -- regression coverage for a
real bug found during the geometric-nonlinear-shell roadmap's Phase D
(wing-cantilever remesh): Shell4MITCCorotational's bending block
({w, theta_x, theta_y} per node) was used RAW in the GLOBAL frame,
silently assuming the element's own reference local frame (e1_0, e2_0,
e3_0) equals the global (x, y, z) frame exactly. Every test in
test_shell_corotational.py and test_shell_corotational_elastica.py used
axis-aligned, CCW-wound flat strips (rectangle_mesh()), where that
assumption happens to hold trivially (e1_0=[1,0,0], e2_0=[0,1,0],
e3_0=[0,0,1]) -- so it was never exercised.

A real Gmsh-generated mesh (the wing-cantilever pentagon planform) has
no reason to respect that convention: element corner winding is a Gmsh
implementation detail, and this particular mesh came out uniformly
CW-wound (e3_0 = -global-z for every element). The bug's actual
symptom, found live: the u=0 tangent stiffness was 60-84% off
Shell4MITC.stiffness() for that mesh (should be near-exact, per
Shell4MITCCorotational's own documented design), and
solve_nonlinear_static() would not converge even at 5-25% of a
genuinely small, physically modest training load -- not a large-
rotation or mesh-quality issue, a wrong tangent.

Fix (shells.py, both _local_relative_dofs() and _dof_local_jacobian_
and_geo()): project the raw global w/theta_x/theta_y onto the FIXED
reference frame (e3_0, e1_0, e2_0) instead of using them unrotated.
Since R0 is evaluated once from the reference geometry (not tracked
with deformation), this is a strict generalization -- a no-op whenever
R0 happens to be the identity (every previously-validated test mesh),
and now also correct when it isn't.

These tests exercise the previously-untested R0 != identity regime
directly: a single CW-wound element, a single in-plane-ROTATED CCW
element, and a multi-element cantilever mesh built with both an
in-plane rotation AND reversed (CW) winding, checked against the SAME
physical answer an axis-aligned CCW mesh gives (physics cannot depend
on how a mesh happens to be authored).
"""
import numpy as np
import pytest

from fea_engine import Material, D_shell, Shell4MITC, Shell4MITCCorotational
from fea_engine import rectangle_mesh, FESystem
from fea_engine.mesh import Mesh
from fea_engine.nonlinear_solver import solve_nonlinear_static


def _material():
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    h = 0.005
    return D_shell(mat, h), h


UNIT_SQUARE = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
CW_SQUARE = UNIT_SQUARE[::-1].copy()   # same physical element, reversed winding

_theta = np.radians(37.0)
_Rz = np.array([[np.cos(_theta), -np.sin(_theta), 0.],
                [np.sin(_theta), np.cos(_theta), 0.],
                [0., 0., 1.]])
ROTATED_SQUARE = UNIT_SQUARE @ _Rz.T   # same physical element, rotated in-plane


@pytest.mark.parametrize("elem_coords", [UNIT_SQUARE, CW_SQUARE, ROTATED_SQUARE])
def test_u0_tangent_matches_linear_shell_any_orientation(elem_coords):
    """The u=0 tangent must reduce to Shell4MITC.stiffness() regardless
    of the element's own winding direction or in-plane orientation --
    before the fix, CW_SQUARE alone was off by ~84% relative (only
    UNIT_SQUARE-style identity-frame elements passed)."""
    D, h = _material()
    coro = Shell4MITCCorotational()
    lin = Shell4MITC()
    K0 = coro.stiffness(elem_coords, D, thickness=h)
    Klin = lin.stiffness(elem_coords, D)
    rel = np.linalg.norm(K0 - Klin) / np.linalg.norm(Klin)
    assert rel < 1e-3


@pytest.mark.parametrize("elem_coords", [CW_SQUARE, ROTATED_SQUARE])
def test_analytic_tangent_matches_complex_step_non_identity_frame(elem_coords):
    """Phase B's analytic-vs-complex-step cross-check, repeated at a
    non-identity reference frame -- confirms the fix's J/K_geo edits
    (not just internal_force()) stay consistent."""
    D, h = _material()
    coro = Shell4MITCCorotational()
    rng = np.random.default_rng(3)
    u = rng.normal(scale=0.05, size=24)
    K_analytic = coro.tangent_stiffness(elem_coords, u, D, thickness=h)
    K_cs = coro._tangent_stiffness_complex_step(elem_coords, u, D, thickness=h)
    rel = np.linalg.norm(K_analytic - K_cs) / max(np.linalg.norm(K_cs), 1.0)
    assert rel < 1e-8


def _cantilever_tip_deflection(mesh, coro, D, h, fixed_nodes, tip_nodes, force_dir, force_mag):
    """Builds+solves a small-load cantilever, returns the mean tip
    displacement in GLOBAL force_dir (a unit 3-vector) -- orientation-
    agnostic readout, since tip_dof_z (index 2) is only meaningful for
    an unrotated mesh."""
    sys_ = FESystem(mesh, coro, thickness=h)
    sys_.fix_dofs(fixed_nodes, [0, 1, 2, 3, 4, 5])
    for d in range(3):
        sys_.add_nodal_force(tip_nodes, d, force_mag * force_dir[d])
    _, U_hist = solve_nonlinear_static(sys_, D, n_steps=3, tol=1e-6, max_iter=30)
    U = U_hist[-1]
    disp = np.array([np.mean([U[6 * int(n) + d] for n in tip_nodes]) for d in range(3)])
    return disp @ force_dir


def test_cantilever_matches_across_orientation():
    """The SAME physical cantilever (1.0 x 0.2m strip, tip force normal
    to the plate), built three ways -- axis-aligned CCW, in-plane-
    rotated 37deg CCW, and in-plane-rotated 37deg with REVERSED (CW)
    connectivity winding -- must give the same tip deflection along the
    physical loading direction. Before the fix, the rotated/CW builds
    would not even converge reliably at this load level (Newton
    diverging from a wrong u=0 tangent), let alone agree numerically."""
    D, h = _material()
    L, W = 1.0, 0.2
    F_mag = -5.0

    mesh2d = rectangle_mesh(L, W, 8, 2)
    nodes_flat = np.hstack([mesh2d.nodes, np.zeros((mesh2d.nodes.shape[0], 1))])

    # Build 1: axis-aligned, CCW (baseline, matches test_shell_corotational.py)
    mesh_base = Mesh(nodes=nodes_flat, elements=mesh2d.elements, dim=2)
    fixed_base = np.where(mesh_base.nodes[:, 0] < 1e-9)[0]
    tip_base = np.where(np.abs(mesh_base.nodes[:, 0] - L) < 1e-9)[0]
    w_base = _cantilever_tip_deflection(
        mesh_base, Shell4MITCCorotational(), D, h, fixed_base, tip_base,
        np.array([0., 0., 1.]), F_mag)

    # Build 2: same physical strip, ROTATED 37deg about z (in-plane) --
    # global force direction must rotate too (it's still "normal to the
    # plate", the plate itself just now sits at an angle in global x-y).
    theta = np.radians(37.0)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0.],
                   [np.sin(theta), np.cos(theta), 0.],
                   [0., 0., 1.]])
    nodes_rot = nodes_flat @ Rz.T
    mesh_rot = Mesh(nodes=nodes_rot, elements=mesh2d.elements, dim=2)
    # locate fixed/tip nodes via the ORIGINAL (unrotated) x-coordinate,
    # since node order/indexing is unchanged by the rotation
    fixed_rot = fixed_base
    tip_rot = tip_base
    w_rot = _cantilever_tip_deflection(
        mesh_rot, Shell4MITCCorotational(), D, h, fixed_rot, tip_rot,
        np.array([0., 0., 1.]), F_mag)

    # Build 3: same rotated strip, but with every element's connectivity
    # REVERSED (CW winding) -- physically identical structure again.
    elements_cw = mesh2d.elements[:, ::-1].copy()
    mesh_cw = Mesh(nodes=nodes_rot, elements=elements_cw, dim=2)
    w_cw = _cantilever_tip_deflection(
        mesh_cw, Shell4MITCCorotational(), D, h, fixed_rot, tip_rot,
        np.array([0., 0., 1.]), F_mag)

    assert abs(w_rot - w_base) / abs(w_base) < 1e-3
    assert abs(w_cw - w_base) / abs(w_base) < 1e-3
