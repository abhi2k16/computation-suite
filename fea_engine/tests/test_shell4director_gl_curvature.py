"""
test_shell4director_gl_curvature.py -- the Green-Lagrange curvature measure
that became `Shell4Director`'s default on 2026-09-24 (see the
"GREEN-LAGRANGE CURVATURE" comment block in elements/shells_director.py).

Why this file exists: the Phase 3 "moderate rotation" curvature
(kappa = Bb @ [w, e1.t, e2.t]) measures d(sin theta)/ds instead of
d(theta)/ds. On a cantilever that softening exactly cancels the geometric
stiffening -- w_tip stayed linear to <0.2% up to 37 deg of tip rotation, while
the exact elastica stiffens 13% (NonLin-HyROM/shell4director_elastica_benchmark.py).
test_shell4director_elastica.py could not catch it: it stops at ~13 deg on
1-3 elements, where true stiffening is only ~1%. The last test below checks
transverse STIFFENING directly, at 24 deg, which is the quantity that was wrong.

Checks:
1. linear stiffness unchanged at u=0 (the new measure linearizes to Bb exactly)
2. analytic force/tangent == finite differences at large rotations
3. zero force under a rigid rotation of any size
4. exact curvature for an inextensible circular arc at 40 deg (the old measure
   gives cos(40 deg) = 0.77 of it)
5. cantilever strip: transverse stiffening matches the exact elastica at 24 deg
"""
__author__ = "Abhijeet"
import numpy as np
import pytest
from scipy.integrate import solve_bvp

from fea_engine import D_shell, FESystem, Material, rectangle_mesh
from fea_engine.elements.base import jacobian
from fea_engine.elements.shells import Shell4MITCCorotational
from fea_engine.elements.shells_director import Shell4Director
from fea_engine.mesh import Mesh
from fea_engine.nonlinear_solver import solve_nonlinear_static

D = D_shell(Material(E=7e10, nu=0.33, rho=2700.0), 0.01)
_X_FLAT = np.array([[0, 0, 0], [0.06, 0.005, 0], [0.065, 0.05, 0], [-0.004, 0.047, 0]], float)
# a distorted quad in a general 3D orientation, so the local frame is non-trivial
_R0 = Shell4MITCCorotational._exp_map(np.array([0.3, -0.5, 0.8]))
X_GEN = _X_FLAT @ _R0.T + np.array([0.2, 0.1, -0.3])


def _rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)


def test_linear_stiffness_unchanged_at_zero():
    gl, mo = Shell4Director(), Shell4Director(curvature="moderate")
    z = np.zeros(24)
    assert np.abs(gl.internal_force(X_GEN, z, D)).max() < 1e-20
    assert _rel(gl.tangent_stiffness(X_GEN, z, D), mo.tangent_stiffness(X_GEN, z, D)) < 1e-12


@pytest.mark.parametrize("rot_scale", [0.05, 0.6, 1.2])
def test_force_and_tangent_match_finite_difference(rot_scale):
    gl = Shell4Director()
    rng = np.random.default_rng(7)
    u = np.zeros((4, 6))
    u[:, :3] = rng.normal(0, 0.01, (4, 3))
    u[:, 3:] = rng.normal(0, rot_scale, (4, 3))
    u = u.ravel()
    assert _rel(gl.internal_force(X_GEN, u, D), gl._internal_force_fd(X_GEN, u, D, h=1e-7)) < 1e-6
    assert _rel(gl.tangent_stiffness(X_GEN, u, D), gl._tangent_stiffness_single_fd(X_GEN, u, D, h=1e-7)) < 1e-6


@pytest.mark.parametrize("angle_deg", [10, 45, 90])
def test_zero_force_under_rigid_rotation(angle_deg):
    gl = Shell4Director()
    axis = np.array([0.3, 0.9, 0.2]) / np.linalg.norm([0.3, 0.9, 0.2])
    v = np.radians(angle_deg) * axis
    R = Shell4MITCCorotational._exp_map(v)
    c = X_GEN.mean(axis=0)
    u = np.zeros((4, 6))
    u[:, :3] = (X_GEN - c) @ R.T + c - X_GEN
    u[:, 3:] = v
    f = gl.internal_force(X_GEN, u.ravel(), D)
    f[[5, 11, 17, 23]] = 0.0          # drilling regularization is not rotation-invariant by design
    k_scale = np.abs(gl.tangent_stiffness(X_GEN, np.zeros(24), D)).max()
    assert np.abs(f).max() < 1e-12 * k_scale   # measured ~1e-8 against a ~1e9 stiffness scale


def _center_curvature(elem, X, u):
    """kappa at the element center, by each measure."""
    u_nodes = u.reshape(4, 6)
    x_cur = X + u_nodes[:, :3]
    t0, e1, e2, e3, local = elem.reference_directors(X)
    t = elem._current_directors(u_nodes[:, 3:], e3)
    _, dN_nat = elem._mitc._membrane.shape_and_derivs((0.0, 0.0))
    J, _ = jacobian(dN_nat, local)
    dNdx, dNdy = np.linalg.solve(J, dN_nat)
    if elem.curvature == "green_lagrange":
        return elem._gl_curvature(dNdx, dNdy, x_cur, X, t, t0)[0]
    dof_bs = np.column_stack([u_nodes[:, :3] @ e3, t @ e1, t @ e2]).ravel()
    Bb = elem._mitc._plate._Bb_Bs((0.0, 0.0), local)[0]
    return Bb @ dof_bs


def test_exact_curvature_for_circular_arc_at_40_degrees():
    """Map a small flat element at arc length s0 onto a circle of radius R
    (inextensible, exact director): the true curvature is 1/R. The element
    sits at ~40 deg of rotation, so the old measure should read cos(40)/R."""
    R, s0, Le, b = 1.0, np.radians(40.0), 0.01, 0.01
    X = np.array([[s0, 0, 0], [s0 + Le, 0, 0], [s0 + Le, b, 0], [s0, b, 0]], float)
    u = np.zeros((4, 6))
    for a, (s, y, _) in enumerate(X):
        u[a, :3] = [R * np.sin(s / R) - s, 0.0, R * (1 - np.cos(s / R))]
        u[a, 3:] = [0.0, -s / R, 0.0]       # director stays normal to the arc
    u = u.ravel()
    k_gl = _center_curvature(Shell4Director(), X, u)
    k_mo = _center_curvature(Shell4Director(curvature="moderate"), X, u)
    s_mid = s0 + Le / 2
    assert abs(abs(k_gl[0]) * R - 1.0) < 1e-3
    assert abs(k_gl[1]) < 1e-9 and abs(k_gl[2]) < 1e-9
    assert abs(abs(k_mo[0]) * R - np.cos(s_mid / R)) < 1e-3   # documents the old measure's cos(theta) error


def _elastica_ratio(a):
    """Exact inextensible elastica, fixed-direction tip load, a = P L^2 / EI:
    w_tip(nonlinear) / w_tip(linear)."""
    sol = solve_bvp(lambda s, y: np.vstack([y[1], -a * np.cos(y[0])]), lambda ya, yb: np.array([ya[0], yb[1]]),
                    np.linspace(0, 1, 400), np.zeros((2, 400)), tol=1e-10, max_nodes=50000)
    ss = np.linspace(0, 1, 4000)
    return np.trapz(np.sin(sol.sol(ss)[0]), ss) / (a / 3)


def test_cantilever_stiffening_matches_elastica_at_24_degrees():
    L, b, h, E = 1.0, 0.05, 0.005, 210e9
    Dn = D_shell(Material(E=E, nu=0.0, rho=7800.0), h)
    EI = E * b * h ** 3 / 12
    a = 0.9                                                  # ~24 deg tip rotation
    m2 = rectangle_mesh(L, b, 10, 1)
    mesh = Mesh(nodes=np.hstack([m2.nodes, np.zeros((m2.nodes.shape[0], 1))]), elements=m2.elements, dim=2)
    fes = FESystem(mesh, Shell4Director(), thickness=h)
    fes.fix_dofs(np.where(mesh.nodes[:, 0] < 1e-9)[0], [0, 1, 2, 3, 4, 5])
    tip = np.where(np.abs(mesh.nodes[:, 0] - L) < 1e-9)[0]
    fes.add_nodal_force(tip, 2, a * EI / L ** 2)
    free = fes.free_dofs
    K0 = fes.assemble_tangent_stiffness(np.zeros(fes.n_dof), Dn)[np.ix_(free, free)]
    u_lin = np.zeros(fes.n_dof)
    u_lin[free] = np.linalg.solve(K0, fes.F[free])
    U = solve_nonlinear_static(fes, Dn, n_steps=6, tol=1e-7, max_iter=40)[1][-1]
    ratio_fe = U[tip * fes.npn + 2].mean() / u_lin[tip * fes.npn + 2].mean()
    ratio_el = _elastica_ratio(a)                              # 0.9207
    # The old "moderate" curvature gave ratio_fe = 1.0002 here (zero stiffening).
    assert abs(ratio_fe - ratio_el) < 0.003, f"FE {ratio_fe:.4f} vs elastica {ratio_el:.4f}"
