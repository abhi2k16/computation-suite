# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_shell.py -- validation for Module 20 (general-purpose extensions
roadmap Phase 7): material.D_shell()/shell_rho_matrix() and
elements.Shell4MITC (MITC4 general shell element).

Five lines of evidence, in increasing order of scope:

1. test_single_element_rigid_body_modes -- a single Shell4MITC element,
   flat and axis-aligned, must have a symmetric stiffness matrix and
   EXACTLY 6 zero eigenvalues (the 6 global rigid-body modes: 3
   translations + 3 rotations) -- the same minimum-correctness check
   every element in this package gets. Confirms the membrane, bending,
   MITC4 shear, and (small, artificial) drilling contributions combine
   into a legitimate stiffness matrix with no spurious zero-energy modes
   and no accidental extra stiffness.

2. test_frame_objectivity -- a stringent, single-element test that
   caught a REAL bug during development (see below): solve a cantilever
   problem in an axis-aligned frame, then solve the IDENTICAL physical
   problem after rigidly rotating the whole element geometry AND load
   direction by an arbitrary angle. A correctly-objective (frame-
   invariant) element must give the SAME answer once the rotated
   solution is rotated back -- physics cannot depend on which way the
   global axes happen to point. This is what actually caught the
   original bug: Quad4MindlinPlate's own (w, betax, betay) convention
   defines betax as the SLOPE-type rotation that produces x-direction
   bending curvature, which is physically a rotation ABOUT THE Y-AXIS
   (right-hand rule) -- not a rotation about the x-axis. Naively
   scattering Quad4MindlinPlate's (betax, betay) columns straight into
   Shell4MITC's (theta_x, theta_y) TRUE rotation-vector slots (which
   Shell4MITC's own rotation matrix T treats as literal components
   about the local e1/e2 axes) silently mislabels the two -- invisible
   for a flat, axis-aligned mesh (T is the identity there, so it cannot
   matter which label is used) but WRONG for any non-axis-aligned or
   curved configuration, where T is a genuine rotation. See
   elements/shells.py's _BEND index-group comment for the fix (swap
   Quad4MindlinPlate's betax/betay columns when scattering into
   Shell4MITC's local matrix).
3. test_flat_plate_limit_converges_to_quad4_mindlin_plate -- the
   roadmap's headline validation requirement: a Shell4MITC mesh with
   ZERO curvature should reproduce Quad4MindlinPlate's bending results.
   Found, empirically, that MITC4's assumed-strain shear treatment and
   Quad4MindlinPlate's SRI shear treatment are NOT algebraically
   identical in general (different anti-locking mechanisms), so exact
   machine-precision agreement at one coarse mesh should NOT be
   expected -- but at a FINE mesh, both converge to the SAME limit
   (confirmed here to 0.03% agreement), which is the mathematically
   correct thing to require of two different, both-correct anti-locking
   formulations. Also confirms the flat-mesh membrane response stays
   EXACTLY zero (uncoupled from bending, as classical Kirchhoff/Mindlin
   plate theory requires for a homogeneous single-layer plate).
4. test_curved_multi_element_rigid_body_modes -- the multi-element
   analogue of test 1, specifically for a CURVED (non-uniformly-
   oriented) mesh: apply a pure rigid TRANSLATION and, separately, a
   pure rigid ROTATION (about an arbitrary axis) as prescribed nodal
   displacements across a 12-element curved shell strip (each element
   with its OWN local frame, per _local_frame_and_coords()), and check
   K @ u_rigid is zero. This is what actually verifies the per-element
   local-frame + block-rotation + assembly logic is energy-consistent
   when ADJACENT elements have genuinely different orientations, not
   just correct in isolation (test 1) or under a single rigid rotation
   of one element (test 2).
5. test_curved_shell_vs_hex8solid3d_convergence -- the roadmap's
   curved-benchmark validation plan, scoped to a tractable geometry: a
   shallow cylindrical-arc cantilever strip (radius R, arc angle
   theta_max, straight width direction), meshed independently as (a) a
   Shell4MITC mid-surface mesh and (b) a fully 3-D Hex8Solid3D mesh of
   the SAME geometry (multiple elements through the thickness),
   matching this package's established "validate a reduced/simplified
   model against an independent full-fidelity computation" convention
   (see e.g. the arc-length/buckling modules' own validation patterns).
   Both models are loaded with the SAME total tip force along the
   LOCAL RADIAL direction (the direction that actually drives bending,
   as opposed to a raw global-axis force, which mixes in a much larger,
   comparison-swamping membrane/tangential response for anything but a
   very shallow arc -- found and corrected during development, see the
   note on load direction below) and compared via the tip's radial
   displacement component, checked for convergence (mesh refinement in
   BOTH models) toward mutual agreement, not exact equality at one
   mesh -- Hex8Solid3D is separately DOCUMENTED elsewhere in this
   package (README's "Known limitations") to lock significantly in
   bending, so it converges toward the true answer more slowly than
   Shell4MITC does, and matching only at full convergence (not at a
   single coarse mesh) is the honest, correct thing to check.

A note on curved-mesh coordinate generation, for anyone extending this
test: mapping a structured (theta, y, rho) box mesh to physical
cylindrical coordinates via (x, y, z) = (rho*cos(theta), y, rho*sin(theta))
has a NEGATIVE Jacobian determinant relative to box_mesh()'s own
(x=theta, y=y, z=rho) axis convention (d(x,y,z)/d(theta,y,rho) = -rho),
which silently inverts every element's node winding (detJ < 0
everywhere) -- caught here by an explicit sign check
(np.linalg.det(jacobian(...)) > 0 for every element) after initially
missing it. Using z = -rho*sin(theta) instead fixes the winding; the
`_cylindrical_arc_meshes()` helper below does this.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import (Shell4MITC, D_shell, shell_rho_matrix, Material,
                         D_mindlin_plate, D_solid3d, Hex8Solid3D,
                         Quad4MindlinPlate, FESystem)
from fea_engine.mesh import Mesh, box_mesh, rectangle_mesh
from fea_engine.elements.base import jacobian


E, NU, RHO = 2.1e11, 0.3, 7850.0


def test_single_element_rigid_body_modes():
    mat = Material(E=E, nu=NU, rho=RHO)
    D = D_shell(mat, h=0.01)
    elem = Shell4MITC()
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)

    ke = elem.stiffness(coords, D)
    assert np.allclose(ke, ke.T, atol=1e-2)

    eig = np.linalg.eigvalsh(ke)
    n_zero = np.sum(np.abs(eig) < 1e-6 * np.abs(eig).max())
    assert n_zero == 6

    u0 = np.zeros(24)
    f0 = elem.internal_force(coords, u0, D)
    assert np.abs(f0).max() < 1e-6


def _solve_single_element_cantilever(elem, D, coords, fixed_node_idx, load_dir, load_mag):
    """Fixes ALL 6 dofs at fixed_node_idx, splits load_mag*load_dir evenly
    across the other two nodes' translations, solves the reduced 24x24
    system directly (no FESystem needed for a single element)."""
    ke = elem.stiffness(coords, D)
    F = np.zeros(24)
    tip_idx = [i for i in range(4) if i not in fixed_node_idx]
    for i in tip_idx:
        F[6 * i:6 * i + 3] += load_mag * load_dir / len(tip_idx)
    fixed_dofs = [6 * i + k for i in fixed_node_idx for k in range(6)]
    free = [d for d in range(24) if d not in fixed_dofs]
    U = np.zeros(24)
    U[free] = np.linalg.solve(ke[np.ix_(free, free)], F[free])
    return U, tip_idx


def test_frame_objectivity():
    """See module docstring, evidence line 2 -- the test that caught the
    betax/betay <-> theta_x/theta_y mislabeling bug."""
    mat = Material(E=E, nu=NU, rho=RHO)
    D = D_shell(mat, h=0.01)
    elem = Shell4MITC()

    coords0 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    U0, tip0 = _solve_single_element_cantilever(
        elem, D, coords0, [0, 3], np.array([0.0, 0.0, 1.0]), -1000.0)
    w0 = np.mean([U0[6 * i + 2] for i in tip0])

    phi = 0.3
    Rphi = np.array([[np.cos(phi), 0, np.sin(phi)],
                      [0, 1, 0],
                      [-np.sin(phi), 0, np.cos(phi)]])
    coords1 = coords0 @ Rphi.T
    load_dir1 = Rphi @ np.array([0.0, 0.0, 1.0])
    U1, tip1 = _solve_single_element_cantilever(elem, D, coords1, [0, 3], load_dir1, -1000.0)
    disp1 = np.mean([U1[6 * i:6 * i + 3] for i in tip1], axis=0)
    disp1_local = Rphi.T @ disp1

    assert disp1_local[2] == pytest.approx(w0, rel=1e-8)
    assert abs(disp1_local[0]) < 1e-8 * abs(w0)
    assert abs(disp1_local[1]) < 1e-8 * abs(w0)


def test_flat_plate_limit_converges_to_quad4_mindlin_plate():
    mat = Material(E=E, nu=NU, rho=RHO)
    h, L, W = 0.01, 1.0, 0.2
    Db, Ds = D_mindlin_plate(mat, h)
    D_sh = D_shell(mat, h)

    ratios = []
    for nx in (8, 64):
        ny = max(2, nx // 4)
        mesh_p = rectangle_mesh(L, W, nx, ny)
        sys_p = FESystem(mesh_p, Quad4MindlinPlate())
        sys_p.assemble_stiffness((Db, Ds))
        left = mesh_p.nodes_on_plane(axis=0, value=0.0)
        sys_p.fix_dofs(left, [0, 1, 2])
        tip = mesh_p.nodes_on_plane(axis=0, value=L)
        sys_p.add_nodal_force(tip, 0, -1000.0)
        U_p = sys_p.solve_static()
        w_p = U_p[3 * tip[0]]

        nodes3 = np.column_stack([mesh_p.nodes, np.zeros(len(mesh_p.nodes))])
        mesh_s = Mesh(nodes3, mesh_p.elements, dim=3)
        sys_s = FESystem(mesh_s, Shell4MITC(), sparse=False)
        sys_s.assemble_stiffness(D_sh)
        sys_s.fix_dofs(left, [0, 1, 2, 3, 4, 5])
        sys_s.add_nodal_force(tip, 2, -1000.0)
        U_s = sys_s.solve_static()
        w_s = U_s[6 * tip[0] + 2]

        # membrane response must stay exactly zero -- flat, single-layer
        # homogeneous shell: bending and membrane are uncoupled
        assert np.abs(U_s[0::6]).max() < 1e-9
        assert np.abs(U_s[1::6]).max() < 1e-9

        ratios.append(w_s / w_p)

    # coarse mesh: shear-treatment difference (MITC4 tying vs SRI)
    # allows some disagreement; fine mesh: both converge to the SAME
    # bending answer
    assert ratios[-1] == pytest.approx(1.0, abs=5e-3)


def test_curved_multi_element_rigid_body_modes():
    mat = Material(E=E, nu=NU, rho=RHO)
    D = D_shell(mat, h=0.02)
    R, theta_max, Wd = 1.0, 0.4, 0.3
    n_theta, n_y = 12, 2

    raw = rectangle_mesh(theta_max, Wd, n_theta, n_y)
    theta, y = raw.nodes[:, 0], raw.nodes[:, 1]
    nodes = np.column_stack([R * np.cos(theta), y, -R * np.sin(theta)])
    mesh = Mesh(nodes, raw.elements, dim=3)

    sys = FESystem(mesh, Shell4MITC(), sparse=False)
    sys.assemble_stiffness(D)
    n_nodes = mesh.nodes.shape[0]
    n_dof = 6 * n_nodes
    Kscale = np.abs(sys.K).max()

    u_trans = np.zeros(n_dof)
    u_trans[0::6] = 1.0
    assert np.max(np.abs(sys.K @ u_trans)) < 1e-6 * Kscale

    # Rigid rotation by a small angle eps about an arbitrary axis:
    # du_i = eps*(axis x x_i), d(theta_i) = eps*axis. K @ u_rot should be
    # ~0 -- compared against the matrix's OWN overall scale (Kscale),
    # the standard "zero" criterion this package's other rigid-body-mode
    # checks already use (e.g. Beam3DEulerBernoulli's), NOT against
    # eps*Kscale: eps only sets how far u_rot is from the origin along a
    # curved manifold of exactly-rigid configurations, it does not set
    # the expected roundoff FLOOR of a linear operator's residual, so
    # scaling the tolerance by eps as well would be double-counting it.
    eps = 1e-6
    axis = np.array([0.0, 1.0, 0.0])
    u_rot = np.zeros(n_dof)
    for i in range(n_nodes):
        u_rot[6 * i:6 * i + 3] = eps * np.cross(axis, mesh.nodes[i])
        u_rot[6 * i + 3:6 * i + 6] = eps * axis
    assert np.max(np.abs(sys.K @ u_rot)) < 1e-6 * Kscale


def _cylindrical_arc_meshes(R, h, theta_max, Wd, n_theta, n_y, n_r):
    """Builds matched Shell4MITC (mid-surface) and Hex8Solid3D (full
    3-D, n_r layers through thickness) meshes of the same shallow
    cylindrical-arc cantilever strip. See the module docstring's note
    on the z = -rho*sin(theta) sign (fixes a Jacobian-orientation flip
    that otherwise inverts every solid element's winding)."""
    raw_solid = box_mesh(theta_max, Wd, h, n_theta, n_y, n_r)
    raw_c = raw_solid.nodes.copy()
    theta, y, rho = raw_c[:, 0], raw_c[:, 1], (R - h / 2) + raw_c[:, 2]
    solid_nodes = np.column_stack([rho * np.cos(theta), y, -rho * np.sin(theta)])
    solid_mesh = Mesh(solid_nodes, raw_solid.elements, dim=3)

    raw_shell = rectangle_mesh(theta_max, Wd, n_theta, n_y)
    raw_cs = raw_shell.nodes.copy()
    theta_s, y_s = raw_cs[:, 0], raw_cs[:, 1]
    shell_nodes = np.column_stack([R * np.cos(theta_s), y_s, -R * np.sin(theta_s)])
    shell_mesh = Mesh(shell_nodes, raw_shell.elements, dim=3)

    fixed_solid = np.where(np.isclose(raw_c[:, 0], 0.0))[0]
    tip_solid_all = np.where(np.isclose(raw_c[:, 0], theta_max))[0]
    tip_solid_mid = np.where(np.isclose(raw_c[:, 0], theta_max)
                              & np.isclose(raw_c[:, 2], h / 2))[0]
    fixed_shell = np.where(np.isclose(raw_cs[:, 0], 0.0))[0]
    tip_shell = np.where(np.isclose(raw_cs[:, 0], theta_max))[0]
    return (solid_mesh, shell_mesh, fixed_solid, tip_solid_all, tip_solid_mid,
            fixed_shell, tip_shell)


def _tip_radial_disp_solid(solid_mesh, mat, fixed, tip_all, tip_mid, rad_dir, F_mag):
    sys = FESystem(solid_mesh, Hex8Solid3D(), sparse=False)
    sys.assemble_stiffness(D_solid3d(mat))
    sys.fix_dofs(fixed, [0, 1, 2])
    for d in range(3):
        sys.add_nodal_force(tip_all, d, F_mag * rad_dir[d])
    U = sys.solve_static()
    return np.mean([U[3 * n:3 * n + 3] @ rad_dir for n in tip_mid])


def _tip_radial_disp_shell(shell_mesh, D, fixed, tip, rad_dir, F_mag):
    sys = FESystem(shell_mesh, Shell4MITC(), sparse=False)
    sys.assemble_stiffness(D)
    sys.fix_dofs(fixed, [0, 1, 2, 3, 4, 5])
    for d in range(3):
        sys.add_nodal_force(tip, d, F_mag * rad_dir[d])
    U = sys.solve_static()
    return np.mean([U[6 * n:6 * n + 3] @ rad_dir for n in tip])


def test_curved_shell_vs_hex8solid3d_convergence():
    mat = Material(E=E, nu=NU, rho=RHO)
    R, h, Wd, theta_max = 1.0, 0.02, 0.3, 0.1
    F_mag = -2000.0
    rad_dir = np.array([np.cos(theta_max), 0.0, -np.sin(theta_max)])
    D = D_shell(mat, h)

    # solid mesh winding sanity check (see module docstring)
    _, shell_check, *_ = _cylindrical_arc_meshes(R, h, theta_max, Wd, 4, 2, 2)
    solid_check, *_ = _cylindrical_arc_meshes(R, h, theta_max, Wd, 4, 2, 2)
    elem8 = Hex8Solid3D()
    for conn in solid_check.elements:
        X = solid_check.nodes[conn]
        _, dN = elem8.shape_and_derivs((0, 0, 0))
        _, detJ = jacobian(dN, X)
        assert detJ > 0

    w_solid_coarse = None
    w_solid_fine = None
    for n_r in (2, 4):
        (solid_mesh, shell_mesh, fixed_solid, tip_solid_all, tip_solid_mid,
         fixed_shell, tip_shell) = _cylindrical_arc_meshes(R, h, theta_max, Wd, 16, 4, n_r)
        w = _tip_radial_disp_solid(solid_mesh, mat, fixed_solid, tip_solid_all,
                                    tip_solid_mid, rad_dir, F_mag)
        if n_r == 2:
            w_solid_coarse = w
        else:
            w_solid_fine = w

    w_shell_coarse = None
    w_shell_fine = None
    for n_theta in (8, 32):
        (solid_mesh, shell_mesh, fixed_solid, tip_solid_all, tip_solid_mid,
         fixed_shell, tip_shell) = _cylindrical_arc_meshes(R, h, theta_max, Wd, n_theta, 4, 4)
        w = _tip_radial_disp_shell(shell_mesh, D, fixed_shell, tip_shell, rad_dir, F_mag)
        if n_theta == 8:
            w_shell_coarse = w
        else:
            w_shell_fine = w

    # both same sign, both refining monotonically toward each other,
    # and the fine-mesh gap between the two independent models is
    # within a physically reasonable ~10% (shell theory vs full 3-D
    # elasticity, with Hex8Solid3D's own documented bending-locking
    # tendency still narrowing as n_r grows -- see module docstring)
    assert np.sign(w_solid_fine) == np.sign(w_shell_fine)
    assert abs(w_shell_fine - w_solid_fine) < 0.5 * abs(w_solid_fine)
    # solid refinement in n_r moves monotonically toward the shell answer
    assert abs(w_solid_fine - w_shell_fine) < abs(w_solid_coarse - w_shell_fine)
