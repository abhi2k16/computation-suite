# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"

import numpy as np
from fea_engine import geometry as geo
from fea_engine import elements as elmod
from fea_engine import nonlinear_solver as nls
from fea_engine.material import Material, Section, EI_beam, D_plane_stress, D_mindlin_plate, D_solid3d



def test_geometry():
    np.set_printoptions(precision=6, suppress=True)


    def equilibrium_check(sys, U, local_dof_index, label):
        """Exact (to linear-solve precision), element-agnostic check: for
        ANY conforming displacement-based element, a rigid translation
        produces zero strain, so K @ u_rigid = 0 for the rigid-translation
        vector in any single direction. Since K is symmetric, this forces
        sum_i(applied F)_i + sum_i(reaction)_i = 0 in that direction, for
        the EQUILIBRIUM U specifically (derivation in this script's
        accompanying chat message / README Module 11 section). This holds
        regardless of mesh refinement or element locking, so it validates
        that geometry.build_system()'s mesh+element+FESystem plumbing is
        wired correctly even where a tight closed-form comparison isn't
        practical (plate/solid locking behavior is already characterized
        elsewhere, in main.py)."""
        dofs = np.arange(local_dof_index, sys.n_dof, sys.npn)
        resid = sys.K @ U - sys.F
        reaction_sum = resid[np.intersect1d(dofs, list(sys.fixed_dofs))].sum()
        applied_sum = sys.F[dofs].sum()
        err = abs(reaction_sum + applied_sum)
        scale = max(abs(applied_sum), 1.0)
        print(f"  [{label}] applied={applied_sum:.4f}  reaction={reaction_sum:.4f}  "
              f"|sum|={err:.3e}  (rel {err/scale:.2e})")
        assert err < 1e-6 * scale, f"{label}: global equilibrium violated"


    print("=" * 70)
    print("CHECK 1: dim=1 default (beam) vs closed-form tip deflection")
    print("=" * 70)
    mat = Material(E=210e9, nu=0.3, rho=7850.0)
    sec = Section(A=1e-4, I=8.0e-8)
    L, n = 1.0, 20
    sys1, mesh1, elem1 = geo.build_system(dim=1, L=L, n=n)
    assert isinstance(elem1, elmod.Beam2DEulerBernoulli)
    assert mesh1.nodes.shape == (n + 1, 2), "dim=1 mesh should be embedded in 2 columns"

    EI = EI_beam(mat, sec)
    sys1.assemble_stiffness(EI)
    tip = len(mesh1.nodes) - 1
    P = -1000.0
    sys1.add_nodal_force([tip], 0, P)
    sys1.fix_dofs([0], [0, 1])
    U1 = sys1.solve_static()
    v_tip_fe = U1[sys1.npn * tip + 0]
    v_tip_cf = P * L ** 3 / (3 * EI)
    rel_err1 = abs(v_tip_fe - v_tip_cf) / abs(v_tip_cf)
    print(f"  v_tip_fe={v_tip_fe:.6e}  v_tip_closed_form={v_tip_cf:.6e}  rel_err={rel_err1:.2e}")
    assert rel_err1 < 1e-10
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 2: dim=1, physics='truss' -- composes with Module 8's TrussTL2D")
    print("=" * 70)
    sys2, mesh2, elem2 = geo.build_system(dim=1, physics="truss", L=L, n=1)
    assert isinstance(elem2, elmod.TrussTL2D)
    assert mesh2.nodes.shape == (2, 2), "line mesh must be embedded in 2 columns for the truss"

    E, A = 200e9, 1e-4
    mat_truss = (E, A)


    def N_truss(u1):
        E_GL = ((L + u1) ** 2 - L ** 2) / (2 * L ** 2)
        return E * E_GL * A * (L + u1) / L


    sys2.fix_dofs([0], [0, 1])
    sys2.fix_dofs([1], [1])
    P_apply = 3.0e6
    sys2.add_nodal_force([1], 0, P_apply)
    load_factors, U_hist = nls.solve_nonlinear_static(sys2, mat_truss, n_steps=20, tol=1e-13)
    u1_fe = U_hist[-1, 2]

    from scipy.optimize import brentq
    u1_cf = brentq(lambda u: N_truss(u) - P_apply, -0.5, 0.9)
    rel_err2 = abs(u1_fe - u1_cf) / abs(u1_cf)
    print(f"  u1_fe={u1_fe:.8f}  u1_closed_form={u1_cf:.8f}  rel_err={rel_err2:.2e}")
    assert rel_err2 < 1e-8
    print("  PASS -- geometry.build_system()'s dim=1 mesh feeds TrussTL2D directly, "
          "no reshaping needed by the caller.")

    print()
    print("=" * 70)
    print("CHECK 3: dim=2 default (plane_stress) -- exact global-equilibrium check")
    print("=" * 70)
    sys3, mesh3, elem3 = geo.build_system(dim=2, Lx=0.4, Ly=0.05, nx=40, ny=5)
    assert isinstance(elem3, elmod.Quad4PlaneStress)
    matp = Material(E=210e9, nu=0.3)
    thickness = 0.01
    sys3.assemble_stiffness(D_plane_stress(matp), thickness=thickness)
    tip_nodes = mesh3.nodes_on_line(axis=0, value=0.4)
    sys3.add_nodal_force(tip_nodes, dof_index=1, total_force=-5000.0)
    sys3.fix_dofs(mesh3.nodes_on_line(axis=0, value=0.0), [0, 1])
    U3 = sys3.solve_static()
    equilibrium_check(sys3, U3, local_dof_index=1, label="plane_stress, y-direction")

    # informational cross-check vs elementary beam theory (not a strict pass/
    # fail -- Quad4 locking behavior for this aspect ratio is already
    # characterized in main.py; this just shows the geometry-module result
    # is in a physically sane ballpark)
    I_beam = thickness * 0.05 ** 3 / 12
    EI_beam_equiv = matp.E * I_beam
    v_tip_beam_theory = -5000.0 * 0.4 ** 3 / (3 * EI_beam_equiv)
    tip_node_id = tip_nodes[len(tip_nodes) // 2]
    v_tip_fe3 = U3[sys3.npn * tip_node_id + 1]
    print(f"  (info) v_tip_fe={v_tip_fe3:.6e}  vs elementary-beam-theory estimate "
          f"{v_tip_beam_theory:.6e}  ratio={v_tip_fe3/v_tip_beam_theory:.3f}")
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 4: dim=2, physics='plate' -- exact global-equilibrium check")
    print("=" * 70)
    sys4, mesh4, elem4 = geo.build_system(dim=2, physics="plate", Lx=0.3, Ly=0.2, nx=12, ny=8)
    assert isinstance(elem4, elmod.Quad4MindlinPlate)
    h = 0.01
    Db_Ds = D_mindlin_plate(matp, h)
    sys4.assemble_stiffness(Db_Ds)
    edge_nodes = mesh4.nodes_on_line(axis=0, value=0.0)
    sys4.fix_dofs(edge_nodes, [0, 1, 2])
    tip_nodes4 = mesh4.nodes_on_line(axis=0, value=0.3)
    sys4.add_nodal_force(tip_nodes4, dof_index=0, total_force=-500.0)
    U4 = sys4.solve_static()
    equilibrium_check(sys4, U4, local_dof_index=0, label="plate, w-direction")
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 5: dim=3 default (solid) -- exact global-equilibrium check")
    print("=" * 70)
    sys5, mesh5, elem5 = geo.build_system(dim=3, Lx=0.3, Ly=0.03, Lz=0.03, nx=15, ny=3, nz=3)
    assert isinstance(elem5, elmod.Hex8Solid3D)
    sys5.assemble_stiffness(D_solid3d(matp), thickness=1.0)
    fixed5 = mesh5.nodes_on_plane(axis=0, value=0.0)
    sys5.fix_dofs(fixed5, [0, 1, 2])
    tip5 = mesh5.nodes_on_plane(axis=0, value=0.3)
    sys5.add_nodal_force(tip5, dof_index=1, total_force=-2000.0)
    U5 = sys5.solve_static()
    equilibrium_check(sys5, U5, local_dof_index=1, label="solid, y-direction")
    print("  PASS")

    print()
    print("=" * 70)
    print("CHECK 6: rectangle_with_hole + mirror -> full cross-section")
    print("=" * 70)
    m_quarter = geo.generate_mesh(dim=2, shape="rectangle_with_hole", a=1.0, b=1.0, R=0.2,
                                   nr=6, ntheta=6)
    m_full = geo.generate_mesh(dim=2, shape="rectangle_with_hole", a=1.0, b=1.0, R=0.2,
                                nr=6, ntheta=6, mirror=(True, True))
    print(f"  quarter: {len(m_quarter.nodes)} nodes, {len(m_quarter.elements)} elements")
    print(f"  full:    {len(m_full.nodes)} nodes, {len(m_full.elements)} elements")
    assert len(m_full.elements) == 4 * len(m_quarter.elements)
    ok = m_full.check_quality(elmod.Quad4PlaneStress(), verbose=False)
    assert ok, "mirrored full mesh has inverted/degenerate elements"
    print("  PASS -- full mesh has exactly 4x the quarter's elements, all Jacobians positive.")

    print()
    print("=" * 70)
    print("CHECK 7: error handling for unknown dim/shape/physics")
    print("=" * 70)
    for fn, kwargs in [
        (lambda: geo.generate_mesh(dim=2, shape="triangle"), {}),
        (lambda: geo.default_element(dim=2, physics="thermal"), {}),
        (lambda: geo.build_system(dim=4), {}),
    ]:
        try:
            fn()
            raise AssertionError("expected a ValueError, got none")
        except ValueError as e:
            print(f"  OK, raised ValueError: {e}")
    print("  PASS")

    print()
    print("ALL CHECKS PASSED")
