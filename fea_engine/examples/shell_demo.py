"""
shell_demo.py -- Phase 7 (general-purpose extensions roadmap) worked
example: Shell4MITC (MITC4 general shell element -- membrane + bending
+ MITC4 assumed-strain transverse shear, 6 dof/node).

Two demonstrations, matching the roadmap's own validation plan:

1) Flat-plate limit: a Shell4MITC cantilever strip with ZERO curvature
   should converge, as the mesh refines, to Quad4MindlinPlate's own
   (already-validated) bending answer -- the roadmap's headline
   requirement. Also confirms the membrane response stays exactly zero
   for a flat, homogeneous single-layer shell under a pure transverse
   load (membrane/bending are uncoupled, as classical plate theory
   requires).

2) Curved shell vs. an independent full 3-D solid mesh: a shallow
   cylindrical-arc cantilever strip, meshed both as a Shell4MITC
   mid-surface mesh and (independently) as a Hex8Solid3D mesh with
   several elements through the thickness -- the same "validate a
   reduced/simplified model against full-fidelity" convention this
   package uses elsewhere. Loaded with a tip force along the LOCAL
   RADIAL direction (the direction that actually drives bending -- a
   raw global-axis force mixes in a much larger membrane/tangential
   response for anything but a very shallow arc) and compared via
   mesh-refinement convergence, since MITC4 and a locking-prone linear
   solid element (see this package's own documented Hex8Solid3D
   bending-locking limitation) don't need to match at any one coarse
   mesh, only in their converged limits.
"""
__author__ = "Abhijeet"
import numpy as np

from fea_engine import (Shell4MITC, D_shell, Material, D_mindlin_plate,
                         D_solid3d, Hex8Solid3D, Quad4MindlinPlate, FESystem)
from fea_engine.mesh import Mesh, box_mesh, rectangle_mesh


E, NU, RHO = 2.1e11, 0.3, 7850.0


def flat_plate_limit_demo():
    print("=" * 72)
    print("1) Flat-plate limit: Shell4MITC vs. Quad4MindlinPlate")
    print("=" * 72)
    mat = Material(E=E, nu=NU, rho=RHO)
    h, L, W = 0.01, 1.0, 0.2
    Db, Ds = D_mindlin_plate(mat, h)
    D_sh = D_shell(mat, h)

    print(f"{'nx':>4} {'w_plate (m)':>14} {'w_shell (m)':>14} {'ratio':>8}")
    for nx in (4, 8, 16, 32, 64):
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

        print(f"{nx:4d} {w_p:14.6e} {w_s:14.6e} {w_s / w_p:8.4f}")

    print("(ratio -> 1 as the mesh refines: MITC4's assumed-strain shear\n"
          " and Quad4MindlinPlate's selective-reduced-integration shear\n"
          " are different, both-valid anti-locking treatments -- they\n"
          " converge to the SAME answer, not necessarily agree exactly\n"
          " at one coarse mesh)")
    print(f"max |membrane disp| (should be exactly 0): "
          f"{max(np.abs(U_s[0::6]).max(), np.abs(U_s[1::6]).max()):.2e}")


def _cylindrical_arc_meshes(R, h, theta_max, Wd, n_theta, n_y, n_r):
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


def curved_shell_vs_solid_demo():
    print()
    print("=" * 72)
    print("2) Curved shell strip vs. independent Hex8Solid3D reference")
    print("=" * 72)
    mat = Material(E=E, nu=NU, rho=RHO)
    R, h, Wd, theta_max = 1.0, 0.02, 0.3, 0.1
    F_mag = -2000.0
    rad_dir = np.array([np.cos(theta_max), 0.0, -np.sin(theta_max)])
    D = D_shell(mat, h)

    print(f"R={R} m, h={h} m, arc={np.degrees(theta_max):.1f} deg, "
          f"width={Wd} m, tip force={F_mag} N along local radial direction")
    print(f"\n{'n_r (solid layers)':>20} {'w_solid (radial, m)':>22}")
    for n_r in (2, 4, 8):
        (solid_mesh, _, fixed_solid, tip_solid_all, tip_solid_mid, _, _) = \
            _cylindrical_arc_meshes(R, h, theta_max, Wd, 16, 4, n_r)
        sys_sol = FESystem(solid_mesh, Hex8Solid3D(), sparse=False)
        sys_sol.assemble_stiffness(D_solid3d(mat))
        sys_sol.fix_dofs(fixed_solid, [0, 1, 2])
        for d in range(3):
            sys_sol.add_nodal_force(tip_solid_all, d, F_mag * rad_dir[d])
        U_sol = sys_sol.solve_static()
        w_solid = np.mean([U_sol[3 * n:3 * n + 3] @ rad_dir for n in tip_solid_mid])
        print(f"{n_r:20d} {w_solid:22.6e}")

    print(f"\n{'n_theta (shell elems)':>22} {'w_shell (radial, m)':>22}")
    for n_theta in (8, 16, 32):
        (_, shell_mesh, _, _, _, fixed_shell, tip_shell) = \
            _cylindrical_arc_meshes(R, h, theta_max, Wd, n_theta, 4, 4)
        sys_sh = FESystem(shell_mesh, Shell4MITC(), sparse=False)
        sys_sh.assemble_stiffness(D)
        sys_sh.fix_dofs(fixed_shell, [0, 1, 2, 3, 4, 5])
        for d in range(3):
            sys_sh.add_nodal_force(tip_shell, d, F_mag * rad_dir[d])
        U_sh = sys_sh.solve_static()
        w_shell = np.mean([U_sh[6 * n:6 * n + 3] @ rad_dir for n in tip_shell])
        print(f"{n_theta:22d} {w_shell:22.6e}")

    print("\n(Hex8Solid3D is separately documented in this package's README\n"
          " as locking significantly in bending, so it converges toward\n"
          " the true answer more slowly than Shell4MITC -- both models\n"
          " refining toward mutual agreement, not matching at one coarse\n"
          " mesh, is the correct thing to check here)")


if __name__ == "__main__":
    flat_plate_limit_demo()
    curved_shell_vs_solid_demo()
