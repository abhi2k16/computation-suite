# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"

import numpy as np
from fea_engine import elements as elmod
from fea_engine.mesh import MultiBlockMesh
from fea_engine.solver import FESystem
from fea_engine.material import Material, D_plane_stress



def test_mixed_elements():
    np.set_printoptions(precision=6, suppress=True)


    def equilibrium_check(sysobj, U, local_dof_index, label):
        dofs = np.arange(local_dof_index, sysobj.n_dof, sysobj.npn)
        resid = sysobj.K @ U - sysobj.F
        reaction_sum = resid[np.intersect1d(dofs, list(sysobj.fixed_dofs))].sum()
        applied_sum = sysobj.F[dofs].sum()
        err = abs(reaction_sum + applied_sum)
        scale = max(abs(applied_sum), 1.0)
        print(f"  [{label}] applied={applied_sum:.4f}  reaction={reaction_sum:.4f}  "
              f"|sum|={err:.3e}  (rel {err/scale:.2e})")
        assert err < 1e-8 * scale, f"{label}: global equilibrium violated"


    # A small hand-built mixed mesh: a 1x1 square (nodes 0,1,2,3) split into
    # two Tri3PlaneStress triangles, glued to a 1x1 Quad4PlaneStress square
    # (nodes 1,4,5,2) sharing the edge (node1-node2) -- exactly the "tri3
    # region meets quad4 region" pattern a Gmsh recombine pass with
    # leftover triangles produces.
    #
    #   3---2---5
    #   |  /|   |
    #   | / |   |
    #   |/  |   |
    #   0---1---4
    nodes = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [2.0, 0.0],  # 4
        [2.0, 1.0],  # 5
    ])
    blocks = {
        "tri3": np.array([[0, 1, 2], [0, 2, 3]]),
        "quad4": np.array([[1, 4, 5, 2]]),
    }
    mat = Material(E=210e9, nu=0.3)
    D2 = D_plane_stress(mat)
    tri = elmod.Tri3PlaneStress()
    quad = elmod.Quad4PlaneStress()

    print("=" * 70)
    print("CHECK 1: patch test on a hand-built MIXED Tri3+Quad4 mesh")
    print("=" * 70)
    mesh1 = MultiBlockMesh(nodes=nodes, blocks=blocks, dim=2)
    eps_xx, eps_yy, gamma_xy = 0.0015, -0.0012, 0.0021
    u_all = np.zeros((6, 2))
    for i, (x, y) in enumerate(nodes):
        u_all[i, 0] = eps_xx * x + 0.5 * gamma_xy * y
        u_all[i, 1] = eps_yy * y + 0.5 * gamma_xy * x
    strain_exact = np.array([eps_xx, eps_yy, gamma_xy])

    max_err = 0.0
    for name, connectivity, formulation in [("tri3", blocks["tri3"], tri),
                                             ("quad4", blocks["quad4"], quad)]:
        for elem_conn in connectivity:
            elem_coords = nodes[elem_conn]
            u_elem = u_all[elem_conn].flatten()
            # sample at each element's own natural-coordinate centroid-ish point
            p = (1.0 / 3, 1.0 / 3) if formulation is tri else (0.0, 0.0)
            B, _ = formulation.B_matrix(p, elem_coords)
            strain_fe = B @ u_elem
            err = np.max(np.abs(strain_fe - strain_exact))
            max_err = max(max_err, err)
            print(f"  block={name:5s} elem={list(elem_conn)}  strain_fe={strain_fe}  err={err:.3e}")
    assert max_err < 1e-12
    print("  PASS -- both element types in the SAME mixed mesh reproduce an imposed "
          "uniform strain field exactly, confirming inter-block compatibility (the "
          "shared edge nodes 1,2 carry the same displacement values for both the "
          "triangle and quad formulations, as required for a conforming mesh).")

    print()
    print("=" * 70)
    print("CHECK 2: FESystem assembly + exact global equilibrium on the mixed mesh")
    print("=" * 70)
    sys2 = FESystem(mesh1, {"tri3": elmod.Tri3PlaneStress(), "quad4": elmod.Quad4PlaneStress()})
    thickness = 0.01
    sys2.assemble_stiffness(D2, thickness=thickness)   # same D reused for every block (no dict)
    left_nodes = [0, 3]
    right_nodes = [4, 5]
    sys2.fix_dofs(left_nodes, [0, 1])
    sys2.add_nodal_force(right_nodes, dof_index=0, total_force=20000.0)
    U2 = sys2.solve_static()
    equilibrium_check(sys2, U2, local_dof_index=0, label="mixed Tri3+Quad4 mesh, x-direction")
    print(f"  npn={sys2.npn}  n_dof={sys2.n_dof}  (matches ordinary single-type usage exactly)")
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 3: error handling -- key mismatch, dofs_per_node mismatch, plain-Mesh+dict")
    print("=" * 70)
    try:
        FESystem(mesh1, {"tri3": elmod.Tri3PlaneStress()})   # missing "quad4"
        raise SystemExit("expected ValueError for missing block key, none raised")
    except ValueError as e:
        print(f"  OK, raised ValueError: {e}")

    try:
        FESystem(mesh1, {"tri3": elmod.Tri3PlaneStress(), "quad4": elmod.Quad4MindlinPlate()})
        raise SystemExit("expected ValueError for dofs_per_node mismatch, none raised")
    except ValueError as e:
        print(f"  OK, raised ValueError: {e}")

    from fea_engine.mesh import Mesh as PlainMesh
    plain = PlainMesh(nodes=nodes, elements=blocks["tri3"], dim=2)
    try:
        FESystem(plain, {"tri3": elmod.Tri3PlaneStress()})
        raise SystemExit("expected TypeError for dict formulation + plain Mesh, none raised")
    except TypeError as e:
        print(f"  OK, raised TypeError: {e}")
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 4: per-block materials (D as a dict) vs. an independent hand assembly")
    print("=" * 70)
    mat_soft = Material(E=70e9, nu=0.33)     # e.g. aluminum tri region
    mat_stiff = Material(E=210e9, nu=0.3)    # steel quad region
    D_soft, D_stiff = D_plane_stress(mat_soft), D_plane_stress(mat_stiff)
    sys4 = FESystem(mesh1, {"tri3": elmod.Tri3PlaneStress(), "quad4": elmod.Quad4PlaneStress()})
    sys4.assemble_stiffness({"tri3": D_soft, "quad4": D_stiff}, thickness=thickness)

    K_hand = np.zeros((sys4.n_dof, sys4.n_dof))
    for elem_conn in blocks["tri3"]:
        ke = tri.stiffness(nodes[elem_conn], D_soft, thickness=thickness)
        g = sys4._global_dofs(elem_conn)
        K_hand[np.ix_(g, g)] += ke
    for elem_conn in blocks["quad4"]:
        ke = quad.stiffness(nodes[elem_conn], D_stiff, thickness=thickness)
        g = sys4._global_dofs(elem_conn)
        K_hand[np.ix_(g, g)] += ke
    err4 = np.max(np.abs(sys4.K - K_hand)) / np.max(np.abs(K_hand))
    print(f"  max relative error vs independent per-block hand assembly: {err4:.3e}")
    assert err4 < 1e-12
    print("  PASS -- assemble_stiffness()'s per-block D dict routes D_soft to every "
          "tri3 element and D_stiff to every quad4 element, matching a manual "
          "block-by-block assembly exactly.")

    # CHECK 5 (removed): used to exercise a full Gmsh pipeline
    # (geometry.gmsh_engine.UnifiedGeometryEngine.generate_2d_rectangle_mixed)
    # to build a genuinely mixed tri3+quad4 recombined mesh from Gmsh
    # output, rather than the hand-built mixed mesh CHECK 1-4 above use.
    # Gmsh-backed geometry support was removed from this package (system
    # libGLU dependency; see docs/generalized_mesh_grading_roadmap.md)
    # -- CHECK 1-4 already cover mixed-element assembly/solve on a
    # hand-built mesh, so no coverage is lost for the FESystem mixed-
    # element machinery itself, only for Gmsh's ability to produce such
    # a mesh, which is no longer part of this package.

    print()
    print("ALL CHECKS PASSED")
