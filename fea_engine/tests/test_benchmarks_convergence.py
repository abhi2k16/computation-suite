# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_benchmarks_convergence.py -- independent benchmarks and convergence-rate checks.

Every reference value here comes from theory or an independent computation, never from fea_engine itself:

* patch tests          a linear displacement field must be reproduced exactly by every continuum element on
                       distorted meshes, with constant recovered stress;
* pure bending         the exact elasticity solution (quadratic displacement) lies inside the quadratic
                       elements' space, so Quad8/Tri6 must reproduce it to rounding error, while Quad4/Tri3
                       must converge at their theoretical rate (with the known bending-locking offset);
* beam vibration       closed-form cantilever (Euler-Bernoulli) and fixed-free bar frequencies;
* shear-flexible beam  convergence to the Timoshenko tip deflection;
* large deflection     the elastica of a cantilever with an end load, integrated here with SciPy quadrature.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import itertools

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from fea_engine import (FESystem, Material, D_plane_stress, D_solid3d, Quad4PlaneStress, Quad8PlaneStress,
                        Tri3PlaneStress, Tri6PlaneStress, Hex8Solid3D, Hex20Solid3D, Tet4Solid3D,
                        Tet10Solid3D, Beam2DEulerBernoulli, Beam2DReissner, NewtonOptions, convergence as cv)
from fea_engine import facet_loads, nonlinear_solver as ns
from fea_engine.boundary import facets_on
from fea_engine.geometry import generate_mesh
from fea_engine.material import Section, EI_beam
from fea_engine.mesh import Mesh, rectangle_mesh, box_mesh


# ============================================================================ mesh helpers (test only)
def _linear_companion(f):
    simplex = f.quadrature_family == "simplex"
    return {(2, False): Quad4PlaneStress, (3, False): Hex8Solid3D,
            (2, True): Tri3PlaneStress, (3, True): Tet4Solid3D}[(f.dim, simplex)]()


def _quadratic_mesh(mesh, cls):
    """Add the mid-side nodes of `cls` (Quad8/Tri6/Hex20/Tet10) to a linear mesh, at the straight-edge midpoints."""
    f = cls()
    lin = _linear_companion(f)
    grid = [0.0, 0.5, 1.0] if f.quadrature_family == "simplex" else [-1.0, 0.0, 1.0]
    pos = []
    for a in range(f.n_nodes):
        for p in itertools.product(grid, repeat=f.dim):
            N = f.shape_and_derivs(p)[0]
            if abs(N[a] - 1.0) < 1e-12 and abs(np.abs(N).sum() - 1.0) < 1e-12:
                pos.append(p)
                break
    index = {tuple(np.round(x, 9)): i for i, x in enumerate(mesh.nodes)}
    nodes, conn = list(mesh.nodes), []
    for e in mesh.elements:
        X, row = mesh.nodes[e], list(e)
        for a in range(lin.n_nodes, f.n_nodes):
            x = lin.shape_and_derivs(pos[a])[0] @ X
            key = tuple(np.round(x, 9))
            if key not in index:
                index[key] = len(nodes)
                nodes.append(x)
            row.append(index[key])
        conn.append(row)
    return Mesh(nodes=np.array(nodes), elements=np.array(conn), dim=f.dim)


def _perturb(mesh, amp, seed=1):
    rng = np.random.default_rng(seed)
    X = mesh.nodes.copy()
    lo, hi = X.min(0), X.max(0)
    inner = np.all((X > lo + 1e-9) & (X < hi - 1e-9), axis=1)
    h = (hi - lo) / (np.sqrt(len(X)) if X.shape[1] == 2 else np.cbrt(len(X)))
    X[inner] += amp * h * (rng.random((inner.sum(), X.shape[1])) - 0.5)
    return Mesh(nodes=X, elements=mesh.elements, dim=mesh.dim)


_HEX_TO_TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def _to_tets(mesh):
    return Mesh(nodes=mesh.nodes, dim=3,
                elements=np.array([[e[i] for i in t] for e in mesh.elements for t in _HEX_TO_TETS]))


def _to_tris(mesh):
    q = mesh.elements
    return Mesh(nodes=mesh.nodes, elements=np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]), dim=2)


def _patch_mesh(cls):
    f = cls()
    if f.dim == 2:
        base = _perturb(rectangle_mesh(Lx=1.0, Ly=0.7, nx=4, ny=3), 0.6)
        base = _to_tris(base) if f.quadrature_family == "simplex" else base
    else:
        base = _perturb(box_mesh(1.0, 0.8, 0.6, 3, 2, 2), 0.6)
        base = _to_tets(base) if f.quadrature_family == "simplex" else base
    return base if f.n_nodes == _linear_companion(f).n_nodes else _quadratic_mesh(base, cls)


ELEMENTS = [Quad4PlaneStress, Quad8PlaneStress, Tri3PlaneStress, Tri6PlaneStress,
            Hex8Solid3D, Hex20Solid3D, Tet4Solid3D, Tet10Solid3D]


# ============================================================================ convergence helpers
class TestConvergenceModule:
    def test_recovers_known_orders(self):
        h = np.array([0.4, 0.2, 0.1, 0.05])
        for p in (1.0, 2.0, 3.7):
            e = 3.0 * h ** p
            assert cv.observed_order(h, e) == pytest.approx(p, abs=1e-12)
            assert np.allclose(cv.pairwise_orders(h, e), p)

    def test_richardson_extrapolates_exactly_for_a_pure_power_law(self):
        u0, c, p = 5.0, 2.0, 2.0
        u = lambda h: u0 + c * h ** p
        assert cv.richardson(0.2, u(0.2), 0.1, u(0.1), p) == pytest.approx(u0, abs=1e-12)
        with pytest.raises(ValueError, match="different mesh sizes"):
            cv.richardson(0.1, 1.0, 0.1, 1.0, 2.0)

    def test_run_study_with_exact_and_with_reference_level(self):
        st = cv.run_study(lambda n: 1.0 + 0.3 / n ** 2, [4, 8, 16, 32], exact=1.0)
        assert st.order == pytest.approx(2.0, abs=1e-9) and len(st.orders) == 3
        assert "fitted order 2.000" in str(st)
        st2 = cv.run_study(lambda n: 1.0 + 0.3 / n ** 2, [4, 8, 16, 32])         # vs the finest level
        assert 1.7 < st2.order < 2.6 and st2.errors[-1] == 0.0

    def test_validation(self):
        with pytest.raises(ValueError):
            cv.observed_order([0.1], [0.1])
        with pytest.raises(ValueError, match="positive"):
            cv.observed_order([0.1, 0.05], [0.1, 0.0])
        with pytest.raises(ValueError, match="at least two"):
            cv.run_study(lambda n: 1.0, [4])
        with pytest.raises(ValueError, match="distinct"):
            cv.run_study(lambda n: 1.0, [4, 4])
        st = cv.run_study(lambda n: 1.0, [4, 8], exact=1.0)                      # exact answer: no order
        assert np.isnan(st.order)


# ============================================================================ patch tests
def _patch(cls):
    mesh = _patch_mesh(cls)
    f = cls()
    dim = f.dim
    mat = Material(E=2.1e5, nu=0.3, rho=1.0)
    D = D_plane_stress(mat) if dim == 2 else D_solid3d(mat)
    s = FESystem(mesh, f, thickness=1.0) if dim == 2 else FESystem(mesh, f)
    s.assemble_stiffness(D, thickness=1.0) if dim == 2 else s.assemble_stiffness(D)
    G = np.array([[1e-3, 2e-4, 1e-4], [3e-4, -5e-4, 2e-4], [1e-4, 4e-4, 8e-4]])[:dim, :dim]
    b = np.array([1e-3, -2e-3, 5e-4])[:dim]
    X = mesh.nodes
    exact = X @ G.T + b
    lo, hi = X.min(0), X.max(0)
    boundary = np.where(np.any((np.abs(X - lo) < 1e-9) | (np.abs(X - hi) < 1e-9), axis=1))[0]
    for n in boundary:
        for d in range(dim):
            s.fix_dofs([int(n)], [d], value=float(exact[n, d]))
    assert len(boundary) < len(X)                                  # there are interior nodes to test
    U = s.solve_static()
    eps = 0.5 * (G + G.T)
    e = (np.array([eps[0, 0], eps[1, 1], 2 * eps[0, 1]]) if dim == 2 else
         np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[0, 1], 2 * eps[1, 2], 2 * eps[0, 2]]))
    S = s.stress(U)
    return (np.abs(np.asarray(U).reshape(-1, dim) - exact).max() / np.abs(exact).max(),
            np.abs(S.nodal - D @ e).max() / np.abs(D @ e).max())


@pytest.mark.parametrize("cls", ELEMENTS, ids=lambda c: c.__name__)
def test_patch_test_on_distorted_mesh(cls):
    u_err, s_err = _patch(cls)
    assert u_err < 1e-10 and s_err < 1e-10


# ============================================================================ pure bending of a beam
def _bending_ratio(n, cls, L=4.0, H=1.0, E=1.0e3, M=1.0):
    """Tip deflection / exact for a plane-stress beam under an end moment (consistent end loads)."""
    mesh = rectangle_mesh(Lx=L, Ly=H, nx=n, ny=max(1, n // 4))
    f = cls()
    if f.quadrature_family == "simplex":
        mesh = _to_tris(mesh)
    if f.n_nodes > _linear_companion(f).n_nodes:
        mesh = _quadratic_mesh(mesh, cls)
    s = FESystem(mesh, f, thickness=1.0)
    s.assemble_stiffness(D_plane_stress(Material(E=E, nu=0.0, rho=1.0)), thickness=1.0)
    I = H ** 3 / 12.0
    mesh.select_nodes(x=0.0, name="root")
    s.fix_dofs("root", ["ux", "uy"])
    tip = mesh.select_nodes(x=L, name="tip")
    for facet in facets_on(s, tip):                                 # integral of N_a * sigma_xx on the end edge
        xy = mesh.nodes[list(facet)]
        sigma = -M * (xy[:, 1] - H / 2.0) / I
        for node, val in zip(facet, facet_loads.consistent_facet_mass(2, xy, 1.0) @ sigma):
            s.F[2 * node] += val
    U = s.solve_static()
    return U.component("uy", nodes="tip").mean() / (M * L ** 2 / (2.0 * E * I))


@pytest.mark.parametrize("cls", [Quad8PlaneStress, Tri6PlaneStress], ids=lambda c: c.__name__)
@pytest.mark.parametrize("n", [4, 8])
def test_quadratic_elements_reproduce_pure_bending_exactly(cls, n):
    assert _bending_ratio(n, cls) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("cls, min_order", [(Quad4PlaneStress, 1.8), (Tri3PlaneStress, 1.6)],
                         ids=lambda v: getattr(v, "__name__", str(v)))
def test_linear_elements_converge_to_pure_bending_at_second_order(cls, min_order):
    ns_ = [8, 16, 32]
    study = cv.run_study(lambda n: _bending_ratio(n, cls), ns_, exact=1.0)
    assert np.all(study.values < 1.0) and np.all(np.diff(study.errors) < 0)       # locks (too stiff), then converges
    assert cv.observed_order(study.h[1:], study.errors[1:]) > min_order


# ============================================================================ vibration
def test_euler_bernoulli_cantilever_frequencies_converge_at_fourth_order():
    E, rho, A, I, L = 210e9, 7800.0, 0.01, 8.33e-6, 2.0
    lam = np.array([1.8751040687, 4.6940911330, 7.8547574382])           # roots of cos(l) cosh(l) = -1
    exact = lam ** 2 / (2 * np.pi * L ** 2) * np.sqrt(E * I / (rho * A))

    def freqs(n):
        b = FESystem(generate_mesh(dim=1, L=L, n=n), Beam2DEulerBernoulli())
        b.assemble_stiffness(EI_beam(Material(E=E, nu=0.3, rho=rho), Section(A=A, I=I)))
        b.assemble_mass(rho * A)
        b.fix_dofs([0], ["uy", "rz"])
        return b.solve_modal(n_modes=3)[0]

    rel = {n: np.abs(freqs(n) / exact - 1.0) for n in (5, 10, 20)}
    assert rel[20][0] < 1e-6 and rel[20][1] < 5e-5 and rel[20][2] < 5e-4
    study = cv.run_study(lambda n: freqs(n)[0], [5, 10, 20], exact=exact[0])
    assert study.order == pytest.approx(4.0, abs=0.3)


def test_axial_bar_frequencies_converge_at_second_order():
    E, rho, Lb = 1.0e9, 1000.0, 1.0
    exact = np.sqrt(E / rho) / (4.0 * Lb)                                    # fixed-free bar, first mode

    def f1(n):
        mesh = rectangle_mesh(Lx=Lb, Ly=0.1, nx=n, ny=1)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=0.1)
        s.assemble_stiffness(D_plane_stress(Material(E=E, nu=0.0, rho=rho)), thickness=0.1)
        s.assemble_mass(rho * np.eye(2), thickness=0.1)
        mesh.select_nodes(x=0.0, name="root")
        s.fix_dofs("root", ["ux"])
        s.fix_dofs(np.arange(len(mesh.nodes)), ["uy"])                      # axial modes only
        return s.solve_modal(n_modes=1)[0][0]

    study = cv.run_study(f1, [8, 16, 32], exact=exact)
    assert study.order == pytest.approx(2.0, abs=0.1)
    assert study.errors[-1] / exact < 2e-4


# ============================================================================ shear-flexible beam
def test_reissner_beam_converges_to_timoshenko_tip_deflection():
    E, nu, A, I, Lb, P = 210e9, 0.3, 1.0e-3, 8.33e-7, 1.0, 1000.0
    G = E / (2 * (1 + nu))

    def tip(n):
        x = np.linspace(0, Lb, n + 1).reshape(-1, 1)
        s = FESystem(Mesh(nodes=np.hstack([x, 0 * x]), elements=np.array([[i, i + 1] for i in range(n)]), dim=1),
                     Beam2DReissner())
        s.fix_dofs([0], ["ux", "uy", "rz"])
        s.assemble_stiffness((E, G, A, I, 1.0))
        s.add_nodal_force([n], "uy", P)
        return s.solve_static().component("uy", nodes=[n])[0]

    exact = P * Lb ** 3 / (3 * E * I) + P * Lb / (G * A)                       # bending + shear
    study = cv.run_study(tip, [2, 4, 8, 16], exact=exact)
    assert study.order == pytest.approx(2.0, abs=0.15)
    assert study.errors[-1] / exact < 2e-3


# ============================================================================ large deflection
def _elastica_tip(alpha):
    """Tip (vertical deflection, horizontal shortening)/L of a cantilever under end load, alpha = P L^2 / EI."""
    def integrate(theta_t, f):
        g = lambda v: (f(theta_t * (1 - v * v)) * 2 * theta_t * v
                       / np.sqrt(2 * alpha * (np.sin(theta_t) - np.sin(theta_t * (1 - v * v)))))
        return quad(g, 1e-12, 1.0, limit=200)[0]
    theta_t = brentq(lambda t: integrate(t, lambda th: 1.0) - 1.0, 0.05, 1.5)
    return integrate(theta_t, np.sin), 1.0 - integrate(theta_t, np.cos)


def test_cantilever_elastica_matches_independent_quadrature():
    E, nu, A, I, Lb = 210e9, 0.3, 0.05, 8.33e-7, 1.0               # stocky section: shear deformation is negligible
    G = E / (2 * (1 + nu))
    w_ref, shorten_ref = _elastica_tip(1.0)
    assert w_ref == pytest.approx(0.301721, abs=1e-5) and shorten_ref == pytest.approx(0.056433, abs=1e-5)

    def tip(n):
        x = np.linspace(0, Lb, n + 1).reshape(-1, 1)
        s = FESystem(Mesh(nodes=np.hstack([x, 0 * x]), elements=np.array([[i, i + 1] for i in range(n)]), dim=1),
                     Beam2DReissner())
        s.fix_dofs([0], ["ux", "uy", "rz"])
        mat = (E, G, A, I, 1.0)
        s.assemble_stiffness(mat)
        s.add_nodal_force([n], "uy", E * I / Lb ** 2)
        _, hist = ns.solve_nonlinear_static(s, mat, n_steps=10, options=NewtonOptions(tol=1e-8, max_iter=60))
        u = s.field(hist[-1])
        return u.component("uy", nodes=[n])[0] / Lb, -u.component("ux", nodes=[n])[0] / Lb

    w10, sh10 = tip(10)
    w40, sh40 = tip(40)
    assert abs(w40 - w_ref) < abs(w10 - w_ref) and abs(sh40 - shorten_ref) < abs(sh10 - shorten_ref)
    assert w40 == pytest.approx(w_ref, rel=5e-4)
    assert sh40 == pytest.approx(shorten_ref, rel=1e-3)
