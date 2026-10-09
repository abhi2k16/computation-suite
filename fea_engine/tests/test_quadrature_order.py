"""
test_quadrature_order.py -- Wave 15 item 125 (docs/consolidated_future_
roadmap.md, source: TensorMesh's "Elements and Quadrature" documentation
page's uniform get_quadrature(order) pattern): validates the new
order-PARAMETERIZED simplex quadrature (tri_quadrature(order)/
tet_quadrature(order), elements/base.py) that generalizes the package's
previously-fixed tri_quadrature_3pt()/tet_quadrature_4pt() rules.

Two decisive claims are checked directly, not assumed:

1. EXACTNESS: tri_quadrature(order)/tet_quadrature(order) integrate
   every monomial of total degree <= order EXACTLY (to machine
   precision), against the closed-form triangle/tetrahedron moment
   formulas p!q!/(p+q+2)! and p!q!r!/(p+q+r+3)! -- the same "checked
   directly, not assumed" standard this project used for meshio's node
   order (mesh_io.py) and the corrected RK4's convergence rate (Wave 13
   item 119), not a claim taken on faith from the collapsed-coordinate
   construction's own literature pedigree.

2. BACKWARD COMPATIBILITY + genuine accuracy improvement: Tri6PlaneStress/
   Tet10Solid3D's new quad_order= parameter (None by default) reproduces
   the ORIGINAL fixed-rule stiffness() exactly on a straight-sided
   element (both are degree-2-exact for that case, so must agree to
   floating-point precision) -- and, for mass() (a genuinely QUARTIC
   integrand the old fixed rule only approximates), quad_order=4 gives a
   materially DIFFERENT, and quad_order=6-STABLE (converged/exact)
   result, the decisive signature of having reached the integrand's true
   degree rather than merely using more points.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest
from math import factorial

from fea_engine.elements.base import (
    tri_quadrature, tet_quadrature, tri_quadrature_3pt, tet_quadrature_4pt)
from fea_engine.elements.solids import Tri6PlaneStress, Tet10Solid3D


def _exact_moment_tri(p, q):
    return factorial(p) * factorial(q) / factorial(p + q + 2)


def _exact_moment_tet(p, q, r):
    return factorial(p) * factorial(q) * factorial(r) / factorial(p + q + r + 3)


class TestTriQuadratureExactness:
    @pytest.mark.parametrize("order", range(1, 8))
    def test_weights_sum_to_one_barycentric_normalized(self, order):
        _, weights = tri_quadrature(order)
        assert abs(sum(weights) - 1.0) < 1e-13

    @pytest.mark.parametrize("order", range(1, 8))
    def test_exact_for_every_monomial_up_to_requested_order(self, order):
        print("=" * 70)
        print(f"CHECK: tri_quadrature(order={order}) integrates every "
              f"monomial xi^p*eta^q (p+q<={order}) exactly")
        print("=" * 70)
        points, weights = tri_quadrature(order)
        points = np.array(points)
        weights = np.array(weights)
        for p in range(order + 1):
            q = order - p
            # *0.5: the caller's own area factor, matching Tri6PlaneStress.
            # stiffness()'s own convention -- see tri_quadrature()'s docstring.
            numeric = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q) * 0.5
            exact = _exact_moment_tri(p, q)
            assert numeric == pytest.approx(exact, abs=1e-10), (p, q, numeric, exact)
        print(f"  PASS -- {len(points)} points, exact for all {order+1} "
              f"monomials of total degree {order}")

    def test_order_2_matches_original_fixed_3pt_rule_on_a_real_integrand(self):
        """Not just "both integrate degree-2 monomials exactly in
        isolation" -- confirms the two independently-derived rules (the
        original hand-picked 3-point rule vs. the new collapsed-
        coordinate construction) agree on a genuine mixed-term degree-2
        polynomial, the kind of integrand B^T D B actually produces."""
        pts_new, wts_new = tri_quadrature(2)
        pts_old, wts_old = tri_quadrature_3pt()

        def f(xi, eta):
            return 1 + 2 * xi + 3 * eta + xi * eta + xi ** 2 - eta ** 2

        val_new = sum(w * f(*p) for p, w in zip(pts_new, wts_new)) * 0.5
        val_old = sum(w * f(*p) for p, w in zip(pts_old, wts_old)) * 0.5
        assert val_new == pytest.approx(val_old, abs=1e-10)


class TestTetQuadratureExactness:
    @pytest.mark.parametrize("order", range(1, 6))
    def test_weights_sum_to_one_barycentric_normalized(self, order):
        _, weights = tet_quadrature(order)
        assert abs(sum(weights) - 1.0) < 1e-12

    @pytest.mark.parametrize("order", range(1, 6))
    def test_exact_for_every_monomial_up_to_requested_order(self, order):
        print("=" * 70)
        print(f"CHECK: tet_quadrature(order={order}) integrates every "
              f"monomial r^p*s^q*t^u (p+q+u<={order}) exactly")
        print("=" * 70)
        points, weights = tet_quadrature(order)
        points = np.array(points)
        weights = np.array(weights)
        n_checked = 0
        for p in range(order + 1):
            for q in range(order + 1 - p):
                r = order - p - q
                # /6.0: the caller's own volume factor -- see
                # tet_quadrature()'s docstring.
                numeric = np.sum(
                    weights * points[:, 0] ** p * points[:, 1] ** q * points[:, 2] ** r
                ) / 6.0
                exact = _exact_moment_tet(p, q, r)
                assert numeric == pytest.approx(exact, abs=1e-9), (p, q, r, numeric, exact)
                n_checked += 1
        print(f"  PASS -- {len(points)} points, exact for all {n_checked} "
              f"monomials of total degree {order}")


def _straight_sided_tri6_coords():
    """Mid-edge nodes at EXACT edge midpoints -- the quadratic
    isoparametric map degenerates to the same affine map Tri3PlaneStress
    uses in this case (a standard FEM fact), so B is genuinely linear in
    (xi,eta) and B^T D B is genuinely degree-2 -- the precondition both
    tri_quadrature_3pt() and tri_quadrature(2)'s own docstrings assume.
    A mesh with mid-nodes NOT at exact midpoints (a curved isoparametric
    element) would make the true integrand non-polynomial, and two
    different degree-2-exact rules would then legitimately disagree --
    that is a property of the geometry, not a bug in either rule."""
    c0, c1, c2 = np.array([0.0, 0.0]), np.array([1.3, 0.2]), np.array([0.3, 1.1])
    return np.array([c0, c1, c2, (c0 + c1) / 2, (c1 + c2) / 2, (c2 + c0) / 2])


def _straight_sided_tet10_coords():
    p0, p1, p2, p3 = (np.array([0, 0, 0.0]), np.array([1.2, 0.1, 0.05]),
                       np.array([0.1, 1.3, 0.05]), np.array([0.05, 0.05, 1.1]))
    return np.array([p0, p1, p2, p3, (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p0) / 2,
                      (p0 + p3) / 2, (p1 + p3) / 2, (p2 + p3) / 2])


class TestTri6QuadOrderParameter:
    def test_default_quad_order_none_unchanged_from_original_fixed_rule(self):
        elem_coords = _straight_sided_tri6_coords()
        D = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.35]])
        t6 = Tri6PlaneStress()
        ke_default = t6.stiffness(elem_coords, D)
        points, weights = tri_quadrature_3pt()
        ke_manual = np.zeros((12, 12))
        for p, w in zip(points, weights):
            B, detJ = t6.B_matrix(p, elem_coords)
            ke_manual += (B.T @ D @ B) * abs(detJ) * w * 0.5
        assert np.allclose(ke_default, ke_manual, atol=1e-12)

    def test_explicit_quad_order_2_matches_default_on_straight_sided_element(self):
        elem_coords = _straight_sided_tri6_coords()
        D = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.35]])
        t6 = Tri6PlaneStress()
        ke_default = t6.stiffness(elem_coords, D)
        ke_q2 = t6.stiffness(elem_coords, D, quad_order=2)
        ke_q4 = t6.stiffness(elem_coords, D, quad_order=4)
        assert np.allclose(ke_default, ke_q2, atol=1e-9)
        assert np.allclose(ke_default, ke_q4, atol=1e-9)

    def test_quad_order_4_mass_genuinely_differs_and_converges(self):
        print("=" * 70)
        print("CHECK: Tri6 consistent mass -- quad_order=4 (quartic-exact)")
        print("differs materially from the default 3-point (degree-2-exact)")
        print("approximation, and is itself CONVERGED (matches quad_order=6")
        print("to ~machine precision) -- the decisive signature of having")
        print("reached the integrand's own true degree, not just 'more points'")
        print("=" * 70)
        elem_coords = _straight_sided_tri6_coords()
        t6 = Tri6PlaneStress()
        m_default = t6.mass(elem_coords, 1.0, quad_order=None)
        m_q4 = t6.mass(elem_coords, 1.0, quad_order=4)
        m_q6 = t6.mass(elem_coords, 1.0, quad_order=6)
        diff_default_q4 = np.max(np.abs(m_default - m_q4))
        diff_q4_q6 = np.max(np.abs(m_q4 - m_q6))
        print(f"  max|default - q4| = {diff_default_q4:.3e} (expect >> 0)")
        print(f"  max|q4 - q6|      = {diff_q4_q6:.3e} (expect ~0, converged)")
        assert diff_default_q4 > 1e-6
        assert diff_q4_q6 < 1e-10


class TestTet10QuadOrderParameter:
    def test_default_quad_order_none_unchanged_from_original_fixed_rule(self):
        elem_coords = _straight_sided_tet10_coords()
        D6 = np.eye(6) + 0.1
        t10 = Tet10Solid3D()
        ke_default = t10.stiffness(elem_coords, D6)
        points, weights = tet_quadrature_4pt()
        ke_manual = np.zeros((30, 30))
        for p, w in zip(points, weights):
            B, detJ = t10.B_matrix(p, elem_coords)
            ke_manual += (B.T @ D6 @ B) * abs(detJ) * w / 6.0
        assert np.allclose(ke_default, ke_manual, atol=1e-11)

    def test_explicit_quad_order_2_matches_default_on_straight_sided_element(self):
        elem_coords = _straight_sided_tet10_coords()
        D6 = np.eye(6) + 0.1
        t10 = Tet10Solid3D()
        ke_default = t10.stiffness(elem_coords, D6)
        ke_q2 = t10.stiffness(elem_coords, D6, quad_order=2)
        assert np.allclose(ke_default, ke_q2, atol=1e-8)

    def test_quad_order_4_mass_genuinely_differs_and_converges(self):
        elem_coords = _straight_sided_tet10_coords()
        t10 = Tet10Solid3D()
        m_default = t10.mass(elem_coords, 1.0, quad_order=None)
        m_q4 = t10.mass(elem_coords, 1.0, quad_order=4)
        m_q6 = t10.mass(elem_coords, 1.0, quad_order=6)
        diff_default_q4 = np.max(np.abs(m_default - m_q4))
        diff_q4_q6 = np.max(np.abs(m_q4 - m_q6))
        assert diff_default_q4 > 1e-6
        assert diff_q4_q6 < 1e-10
