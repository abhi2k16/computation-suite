"""
test_hyperelastic.py -- validation for Module 19 (general-purpose
extensions roadmap Phase 6): material.NeoHookeanMaterial /
material.neo_hookean_pk2_stress() / elements.Tet4NeoHookean.

Four lines of evidence:

1. test_small_strain_limit_matches_D_solid3d -- the roadmap's headline
   requirement for the hyperelastic model: "every hyperelastic model
   must reduce to linear elasticity as strain -> 0". Checks BOTH the
   stress (which should match D_solid3d()@eps with the expected
   O(strain) RELATIVE error -- i.e. the nonlinear correction vanishes
   proportionally to strain, not just "gets smaller") and the tangent
   (via finite difference of the stress function at F=I, which should
   match D_solid3d() to near machine precision).
2. test_deformation_gradient_recovery -- given nodal displacements
   consistent with a KNOWN uniform deformation gradient F, the element
   must recover that EXACT F (a basic Total-Lagrangian kinematics
   sanity check every subsequent check depends on).
3. test_rigid_body_modes_and_zero_state -- symmetric tangent, 6
   rigid-body modes, zero internal force, all at u=0 -- the same
   minimum-correctness check every element in this package gets.
4. test_energy_conservation_through_full_nonlinear_solve -- the
   decisive, END-TO-END check: run a real incremental Newton-Raphson
   solve (nonlinear_solver.solve_nonlinear_static(), exercising
   internal_force()/tangent_stiffness() together through many
   iterations, not just evaluated once at a hand-picked state) and
   verify total-potential-energy conservation -- the external work
   supplied along the loading path (a path integral of force times
   displacement increment) must equal the strain energy V0*W(F) stored
   in the final deformed state, computed independently via the
   closed-form W(F). This is a strong physical check (conservative
   systems have path-independent work = stored energy) that the
   FINITE-DIFFERENCE tangent_stiffness() (see Tet4NeoHookean's
   docstring for why it's FD rather than a hand-derived closed form)
   is good enough for Newton to actually converge to the correct
   equilibrium, not just "mathematically self-consistent" in isolation.
"""
import numpy as np
import pytest

from fea_engine import Tet4NeoHookean, NeoHookeanMaterial, FESystem, D_solid3d, Material
from fea_engine.mesh import Mesh
from fea_engine.material import neo_hookean_pk2_stress
from fea_engine import nonlinear_solver as nls


E, NU = 1.0e6, 0.4   # rubber-like


def _single_tet():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3]])
    mesh = Mesh(nodes=coords, elements=elements, dim=3)
    return mesh, Tet4NeoHookean()


def test_small_strain_limit_matches_D_solid3d():
    mat = NeoHookeanMaterial(E=E, nu=NU)
    D_el = D_solid3d(Material(E=E, nu=NU))

    rel_errs = []
    for eps_tiny in (1e-2, 1e-3, 1e-4, 1e-5):
        F = np.eye(3) + eps_tiny * np.diag([1.0, -NU, -NU])
        S, _ = neo_hookean_pk2_stress(F, mat)
        S_voigt = np.array([S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[1, 2], S[0, 2]])
        eps_voigt = np.array([eps_tiny, -NU * eps_tiny, -NU * eps_tiny, 0, 0, 0])
        sigma_lin = D_el @ eps_voigt
        rel_errs.append(abs(S_voigt[0] - sigma_lin[0]) / abs(sigma_lin[0]))

    # relative error should shrink roughly linearly with strain (first-order
    # nonlinear correction), not just "be small" -- a genuine convergence check
    for i in range(len(rel_errs) - 1):
        assert rel_errs[i + 1] < rel_errs[i] / 5   # ~10x strain shrink -> >5x error shrink

    # tangent at F=I (finite difference of the closed-form stress) should
    # match D_solid3d to near machine precision
    h = 1e-6
    D_fd = np.zeros((6, 6))
    for j in range(6):
        dv = np.zeros(6)
        dv[j] = h
        dE = np.array([[dv[0], dv[3] / 2, dv[5] / 2],
                        [dv[3] / 2, dv[1], dv[4] / 2],
                        [dv[5] / 2, dv[4] / 2, dv[2]]])
        Sp, _ = neo_hookean_pk2_stress(np.eye(3) + dE, mat)
        Sm, _ = neo_hookean_pk2_stress(np.eye(3) - dE, mat)
        D_fd[:, j] = np.array([Sp[0, 0] - Sm[0, 0], Sp[1, 1] - Sm[1, 1], Sp[2, 2] - Sm[2, 2],
                                Sp[0, 1] - Sm[0, 1], Sp[1, 2] - Sm[1, 2], Sp[0, 2] - Sm[0, 2]]) / (2 * h)
    assert np.max(np.abs(D_fd - D_el)) / np.max(np.abs(D_el)) < 1e-8


def test_deformation_gradient_recovery():
    mesh, elem = _single_tet()
    coords = mesh.nodes
    F_applied = np.diag([1.3, 1.0, 1.0])
    u_applied = np.zeros(12)
    for a in range(4):
        u_applied[3 * a:3 * a + 3] = (F_applied - np.eye(3)) @ coords[a]

    F_recovered, _, _ = elem._deformation_gradient(coords, u_applied)
    assert np.allclose(F_recovered, F_applied, atol=1e-12)


def test_rigid_body_modes_and_zero_state():
    mat = NeoHookeanMaterial(E=E, nu=NU)
    mesh, elem = _single_tet()
    coords = mesh.nodes
    u0 = np.zeros(12)

    ke = elem.tangent_stiffness(coords, u0, mat)
    assert np.allclose(ke, ke.T, atol=1e-2)

    eig = np.linalg.eigvalsh(ke)
    n_zero = np.sum(np.abs(eig) < 1e-3 * np.abs(eig).max())
    assert n_zero == 6

    f0 = elem.internal_force(coords, u0, mat)
    assert np.abs(f0).max() < 1e-8


def test_energy_conservation_through_full_nonlinear_solve():
    mat = NeoHookeanMaterial(E=E, nu=NU)
    mesh, elem = _single_tet()
    fs = FESystem(mesh, elem, sparse=False)

    fs.fix_dofs([0, 1, 2], [0, 1, 2])   # fix the base triangle
    F_applied_total = np.array([5e4, 3e4, -2e4])
    fs.F[9:12] = F_applied_total

    n_steps = 50
    load_factors, U_hist = nls.solve_nonlinear_static(fs, mat, n_steps=n_steps, tol=1e-10, max_iter=60)

    u3_hist = U_hist[:, 9:12]
    work = 0.0
    for i in range(1, len(load_factors)):
        F_avg = 0.5 * (load_factors[i] + load_factors[i - 1]) * F_applied_total
        work += F_avg @ (u3_hist[i] - u3_hist[i - 1])

    u_final = U_hist[-1]
    F_def, _, vol = elem._deformation_gradient(mesh.nodes, u_final)
    _, W = neo_hookean_pk2_stress(F_def, mat)
    energy_stored = vol * W

    assert work == pytest.approx(energy_stored, rel=1e-5)
