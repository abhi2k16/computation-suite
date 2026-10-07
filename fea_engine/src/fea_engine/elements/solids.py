"""
solids.py -- isoparametric plane-stress and 3-D solid continuum
elements (Quad4PlaneStress, Hex8Solid3D, Tri3PlaneStress, Tet4Solid3D).

Split out of the original monolithic element.py during the
fea_engine restructuring; no logic changed, only file location.
"""
import numpy as np

from .base import (Element, gauss_product, jacobian, tet_quadrature_4pt,
                    tri_quadrature_3pt, tet_quadrature, tri_quadrature)


class Quad4PlaneStress(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 4, 2, 2, 2

    def shape_and_derivs(self, natural_coords):
        xi, eta = natural_coords
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                              (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        return N, np.vstack([dN_dxi, dN_deta])

    def B_matrix(self, natural_coords, elem_coords):
        _, dN_nat = self._cached_shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN_nat, elem_coords)
        dN_g = np.linalg.solve(J, dN_nat)   # (2,4): [dN/dx; dN/dy]
        B = np.zeros((3, 8))
        for k in range(4):
            B[0, 2 * k] = dN_g[0, k]
            B[1, 2 * k + 1] = dN_g[1, k]
            B[2, 2 * k] = dN_g[1, k]
            B[2, 2 * k + 1] = dN_g[0, k]
        return B, detJ


# =====================================================================
# Quad8 -- serendipity (8-node) plane-stress continuum, Module 15: the
# quadratic counterpart to Quad4PlaneStress above. Corner nodes 0-3 at
# (+-1,+-1) (same ordering/convention as Quad4), mid-edge nodes 4-7 at
# (0,-1),(1,0),(0,1),(-1,0) -- the standard serendipity node layout
# (Cook/Malkus/Plesha; Zienkiewicz & Taylor). "Serendipity" (8 nodes,
# not the full 9-node Lagrange Quad9) is the standard practical choice:
# it reproduces complete quadratic polynomials with fewer DOF than
# Quad9, at the cost of one specific quartic term Quad9's extra
# interior node would capture -- rarely worth the extra DOF in
# practice (see docs/general_purpose_extensions_roadmap.md Section 3).
#
# The GENERIC Gauss loop in Element.stiffness()/mass() (base.py) works
# completely UNMODIFIED for this element -- only shape_and_derivs()
# and B_matrix() are new; gauss_order=3 (vs Quad4's 2) since the
# B^T D B integrand is now higher-degree.
#
# GMSH COMPATIBILITY (checked 2026-08-30, same investigation as
# Hex20Solid3D/Tet10Solid3D's own notes, done for consistency BEFORE
# wiring this element into geometry/gmsh_engine.py's extraction table --
# see docs/geometry_meshing_alternatives_research.md Section 7 item 2,
# which explicitly calls out that Gmsh's documented ordering table
# should not be trusted without checking a real element): unlike
# Hex20/Tet10, this one is CLEAN. A single reference Quad8 built through
# Gmsh (transfinite unit square, SecondOrderIncomplete=1, setOrder(2) --
# element type 16) comes back with corner nodes 0-3 and mid-edge nodes
# 4-7 in EXACTLY this class's own order -- verified by comparing Gmsh's
# returned physical coordinates directly against the natural coordinates
# below (they matched to floating-point precision on the unit square,
# where physical and natural coordinates coincide). GMSH_NODE_ORDER is
# the identity for this element; kept explicit (not omitted) so
# gmsh_engine.py's lookup can treat every quadratic solid/plane-stress
# element uniformly (look up GMSH_NODE_ORDER, apply it) rather than
# special-casing "this one happens to need no reindexing."
# =====================================================================
class Quad8PlaneStress(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 8, 2, 2, 3

    # Permutation mapping Gmsh's native 8-node quadrangle (element type
    # 16, serendipity/incomplete) node order to this class's own order --
    # the identity, verified directly above (not assumed from Gmsh's
    # documentation).
    GMSH_NODE_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]

    # natural coordinates of each of the 8 nodes (corners then mid-edges)
    _xi_i = np.array([-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0])
    _eta_i = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0])

    def shape_and_derivs(self, natural_coords):
        xi, eta = natural_coords
        N = np.zeros(8)
        dN_dxi = np.zeros(8)
        dN_deta = np.zeros(8)
        for i in range(8):
            xii, etai = self._xi_i[i], self._eta_i[i]
            if xii != 0.0 and etai != 0.0:
                # corner node
                a, b = xi * xii, eta * etai
                N[i] = 0.25 * (1 + a) * (1 + b) * (a + b - 1)
                dN_dxi[i] = 0.25 * xii * (1 + b) * (2 * a + b)
                dN_deta[i] = 0.25 * etai * (1 + a) * (a + 2 * b)
            elif xii == 0.0:
                # mid-edge node on a xi=const-varying edge (quadratic in xi)
                N[i] = 0.5 * (1 - xi**2) * (1 + eta * etai)
                dN_dxi[i] = -xi * (1 + eta * etai)
                dN_deta[i] = 0.5 * (1 - xi**2) * etai
            else:
                # mid-edge node on an eta=const-varying edge (quadratic in eta)
                N[i] = 0.5 * (1 + xi * xii) * (1 - eta**2)
                dN_dxi[i] = 0.5 * xii * (1 - eta**2)
                dN_deta[i] = -eta * (1 + xi * xii)
        return N, np.vstack([dN_dxi, dN_deta])

    def B_matrix(self, natural_coords, elem_coords):
        _, dN_nat = self._cached_shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN_nat, elem_coords)
        dN_g = np.linalg.solve(J, dN_nat)   # (2,8): [dN/dx; dN/dy]
        B = np.zeros((3, 16))
        for k in range(8):
            B[0, 2 * k] = dN_g[0, k]
            B[1, 2 * k + 1] = dN_g[1, k]
            B[2, 2 * k] = dN_g[1, k]
            B[2, 2 * k + 1] = dN_g[0, k]
        return B, detJ


# =====================================================================
# Hex8 -- 3-D solid continuum
# =====================================================================
class Hex8Solid3D(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 8, 3, 3, 2

    def shape_and_derivs(self, natural_coords):
        xi, eta, zeta = natural_coords
        xi_i = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        eta_i = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
        zeta_i = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        N = 0.125 * (1 + xi * xi_i) * (1 + eta * eta_i) * (1 + zeta * zeta_i)
        dN_dxi = 0.125 * xi_i * (1 + eta * eta_i) * (1 + zeta * zeta_i)
        dN_deta = 0.125 * eta_i * (1 + xi * xi_i) * (1 + zeta * zeta_i)
        dN_dzeta = 0.125 * zeta_i * (1 + xi * xi_i) * (1 + eta * eta_i)
        return N, np.vstack([dN_dxi, dN_deta, dN_dzeta])

    def B_matrix(self, natural_coords, elem_coords):
        _, dN_nat = self._cached_shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN_nat, elem_coords)
        dN_g = np.linalg.solve(J, dN_nat)   # (3,8)
        B = np.zeros((6, 24))
        for k in range(8):
            dx, dy, dz = dN_g[:, k]
            c = 3 * k
            B[0, c] = dx
            B[1, c + 1] = dy
            B[2, c + 2] = dz
            B[3, c] = dy; B[3, c + 1] = dx
            B[4, c + 1] = dz; B[4, c + 2] = dy
            B[5, c] = dz; B[5, c + 2] = dx
        return B, detJ


# =====================================================================
# Hex8 -- mean-dilatation B-bar (Hughes 1980), Wave 2 item 11
# (docs/consolidated_future_roadmap.md, source general_purpose_
# extensions_roadmap.md / nonlinear_fem_lessons.md / tensormesh_
# comparative_analysis.md Sec.6.4): the volumetric-locking cure for
# near-incompressible 3-D solids -- see nonlinear_fem_lessons.md's own
# quoted description: "assumes the stress field is orthogonal to the
# difference between assumed strain B-bar and actual B(v)... B is
# simply replaced by an assumed-strain matrix B-bar."
#
# THE IDEA: split each Gauss point's strain-displacement matrix B(p)
# into its volumetric part (the 1/3 of trace(strain) distributed across
# the three normal-strain rows) and deviatoric part (everything else):
#     B_vol(p) = (1/3) * m @ m.T @ B(p),   m = [1,1,1,0,0,0]
#     B_dev(p) = B(p) - B_vol(p)
# then replace EVERY Gauss point's volumetric part with the SAME
# element-volume-AVERAGED value:
#     B_vol_bar = (1/V_elem) * sum_p B_vol(p) * detJ(p) * w(p)
#     B_bar(p)  = B_dev(p) + B_vol_bar
# and assemble stiffness with B_bar in place of B, at the SAME full
# 2x2x2 = 8-point rule Hex8Solid3D already uses (NOT a reduced point
# count -- this is deliberately a DIFFERENT cure from Wave 2 item 12's
# hourglass stabilization, which starts from a genuinely under-
# integrated 1-point rule; B-bar keeps full quadrature and just
# constrains the DILATATIONAL part of every point's strain to agree,
# so it never introduces a rank deficiency / hourglass risk at all).
#
# WHY THIS FIXES VOLUMETRIC LOCKING: near-incompressible ADDS an
# effectively infinite penalty (via the bulk modulus, kappa -> infinity
# as nu -> 0.5) on any NONZERO dilatation at a sampled point. Standard
# full integration samples 8 INDEPENDENT dilatation values per element
# -- far more volumetric constraints than an 8-node (24-dof) element's
# displacement field can satisfy everywhere at once, so the penalty
# locks the whole element artificially stiff. Averaging the volumetric
# part down to ONE value per element removes that over-constraint
# (mathematically the same constraint-counting fix "1-point volumetric
# integration" gives -- this is the standard equivalence between mean-
# dilatation B-bar and Q1P0/selective reduced integration on the
# volumetric term -- while the DEVIATORIC part stays fully, safely
# integrated at all 8 points, so shear response and rank are both
# untouched).
#
# WHY THIS CANNOT CORRUPT A PATCH TEST (verified directly in
# tests/test_hex8_bbar.py, not just asserted): for any AFFINE/constant-
# strain displacement field, B(p) is IDENTICAL at every Gauss point by
# construction (that's what "constant strain" means), so B_vol(p) is
# already equal to its own volume average at every point --
# B_bar(p) == B(p) exactly, and this element's stiffness reduces to
# Hex8Solid3D's own, bit-for-bit, for any such field. This is the same
# "limitation principle" (Stolarski & Belytschko 1987, quoted in
# nonlinear_fem_lessons.md) already used to validate Wave 2 item 12's
# hourglass stabilizer: a correct mixed/assumed-strain method can only
# SUPPRESS deleterious strain terms, never change the answer for a
# field the original element already got exactly right.
#
# A SEPARATE class from Hex8Solid3D (not a flag on it) -- both remain
# fully available side by side, the same explicit-choice principle
# established for FESystem's backend= and assemble_stiffness()'s
# method= parameters: B-bar costs a little more (one extra volume-
# average pass per element) and is unnecessary for compressible
# materials (nu well below 0.5, where Hex8Solid3D's plain formulation
# already agrees with it on any real problem, per the exact-reduction
# property above), so it is opt-in, not a silent replacement.
# =====================================================================
class Hex8SolidBbar(Hex8Solid3D):
    _M_VOL = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    def _bbar_matrices(self, elem_coords):
        """Returns (B_bar list, detJ list, weights list) at the SAME
        full 2x2x2 Gauss points Hex8Solid3D.full_stiffness() uses, each
        B_bar(p) = B_dev(p) + B_vol_bar (see class docstring)."""
        pts, wts = gauss_product(self.gauss_order, self.dim)
        Bs, detJs = [], []
        m = self._M_VOL
        mmT = np.outer(m, m) / 3.0
        B_vol_bar = np.zeros((6, 24))
        V_elem = 0.0
        for p, w in zip(pts, wts):
            B, detJ = self.B_matrix(p, elem_coords)
            Bs.append(B)
            detJs.append(detJ)
            B_vol_p = mmT @ B
            B_vol_bar += B_vol_p * detJ * w
            V_elem += detJ * w
        B_vol_bar /= V_elem
        B_bars = [B - (mmT @ B) + B_vol_bar for B in Bs]
        return B_bars, detJs, wts

    def stiffness(self, elem_coords, D, thickness=1.0, gauss_order=None):
        """Ignores gauss_order (B-bar is only meaningful at the full
        8-point rule -- there is no reduced/hourglass variant of this
        element; use Hex8Solid3D + Wave 2 item 12's hourglass_
        stabilized_stiffness() if a reduced-integration Hex8 is what
        you actually want)."""
        B_bars, detJs, wts = self._bbar_matrices(elem_coords)
        n_total = self.n_nodes * self.dofs_per_node
        ke = np.zeros((n_total, n_total))
        for B_bar, detJ, w in zip(B_bars, detJs, wts):
            ke += (B_bar.T @ D @ B_bar) * detJ * w * thickness
        return ke

    def full_stiffness(self, elem_coords, D, thickness=1.0):
        return self.stiffness(elem_coords, D, thickness)

    def reduced_stiffness(self, elem_coords, D, thickness=1.0):
        raise NotImplementedError(
            "Hex8SolidBbar has no reduced-integration variant -- B-bar's "
            "mean-dilatation correction is only meaningful at the full "
            "8-point rule (see this class's own docstring). Use plain "
            "Hex8Solid3D's reduced_stiffness()/hourglass_stabilized_"
            "stiffness() (Wave 2 item 12) if you want a reduced-"
            "integration Hex8 instead.")


# =====================================================================
# Hex20 -- serendipity (20-node) 3-D solid continuum, Module 15: the
# quadratic counterpart to Hex8Solid3D above. 8 corner nodes (same
# ordering/convention as Hex8) + 12 mid-edge nodes -- no face/interior
# nodes (that's the FULL 27-node Lagrange Hex27; serendipity is the
# standard practical choice, same reasoning as Quad8 above).
#
# Mid-edge node natural coordinates: exactly one of (xi,eta,zeta) is 0
# for a mid-edge node (the coordinate that VARIES along that edge);
# the other two match the edge's shared corner value. Node order here
# (8..19): bottom-face edges (0-1,1-2,2-3,3-0), top-face edges
# (4-5,5-6,6-7,7-4), then the four vertical edges (0-4,1-5,2-6,3-7) --
# the standard Zienkiewicz & Taylor / Cook-Malkus-Plesha layout.
#
# GMSH COMPATIBILITY WARNING (found + fixed 2026-08-30, via the wing
# cantilever example in the Multi_Fidelity_NL_Structural_ROM project):
# Tet10Solid3D's node order is CLOSE to Gmsh's native 10-node
# tetrahedron (element type 11) order -- corners and the first four
# mid-edge nodes match, but two mid-edge nodes are swapped (see that
# class's own GMSH COMPATIBILITY WARNING and GMSH_NODE_ORDER, a real,
# separate finding from this one, not identical to it). Hex20's own
# mismatch below is more severe: Gmsh's native 20-node hexahedron
# (element type 17, obtained via gmsh.model.mesh.setOrder(2) with
# Mesh.SecondOrderIncomplete=1) does NOT use the bottom/top/vertical
# grouping above for its own mid-edge nodes 8-19. Verified directly
# (not assumed) by building a single reference unit-cube Hex20 element
# through Gmsh and reading back its actual node coordinates: Gmsh's
# real order interleaves per-corner instead -- 8:(0,1), 9:(0,3),
# 10:(0,4), 11:(1,2), 12:(1,5), 13:(2,3), 14:(2,6), 15:(3,7), 16:(4,5),
# 17:(4,7), 18:(5,6), 19:(6,7). Feeding Gmsh's raw Hexahedron20
# connectivity into this class as-is silently corrupts the isoparametric
# mapping: most elements end up with a Jacobian whose sign flips
# between Gauss points within the SAME element, which surfaces
# downstream as a mass matrix with ~90% zero/negative diagonal entries
# and a non-positive-definite generalized eigenproblem in
# FESystem.solve_modal() -- confirmed to be a pure node-ordering issue,
# not an element-formulation bug, since Hex20Solid3D.mass()/.stiffness()
# are both well-behaved (SPD mass, exactly 6 near-zero rigid-body
# stiffness eigenvalues) on a reference cube AND on an extreme-aspect-
# ratio box when nodes are supplied in the RIGHT order. Any caller
# building a mesh from Gmsh's own Hexahedron20 output MUST reindex its
# connectivity through GMSH_NODE_ORDER below first (corners 0-7 need no
# change; only columns 8-19 differ) -- geometry/gmsh_engine.py does not
# yet generate 2nd-order meshes at all (see
# docs/geometry_meshing_alternatives_research.md Section 7 item 2, not
# yet implemented), so this currently only matters for scripts driving
# Gmsh's API directly, but GMSH_NODE_ORDER is provided here (rather than
# re-derived ad hoc per script) as the single, tested source of truth
# for whoever implements that item.
#
# Same as Quad8: the GENERIC Gauss loop in Element.stiffness()/mass()
# is unmodified -- only shape_and_derivs()/B_matrix() are new.
# gauss_order=3 (27-point full rule) -- see this element's docstring
# note in docs/general_purpose_extensions_roadmap.md Section 3 for why
# a higher order than Hex8's 2 is needed.
# =====================================================================
class Hex20Solid3D(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 20, 3, 3, 3

    # Permutation mapping Gmsh's native Hexahedron20 (element type 17)
    # node order to this class's own order: gmsh_conn[:, GMSH_NODE_ORDER]
    # reindexes a raw Gmsh connectivity array into the order
    # shape_and_derivs()/mass()/stiffness() expect. See the
    # "GMSH COMPATIBILITY WARNING" note above this class for the
    # derivation and verification (a hand-built reference cube element,
    # not an assumption). Corners 0-7 are unchanged (both conventions
    # agree there); only the mid-edge block 8-19 is reordered.
    GMSH_NODE_ORDER = [0, 1, 2, 3, 4, 5, 6, 7,
                        8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15]

    _xi_i = np.array([-1, 1, 1, -1, -1, 1, 1, -1,
                       0, 1, 0, -1, 0, 1, 0, -1, -1, 1, 1, -1], dtype=float)
    _eta_i = np.array([-1, -1, 1, 1, -1, -1, 1, 1,
                        -1, 0, 1, 0, -1, 0, 1, 0, -1, -1, 1, 1], dtype=float)
    _zeta_i = np.array([-1, -1, -1, -1, 1, 1, 1, 1,
                         -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float)

    def shape_and_derivs(self, natural_coords):
        xi, eta, zeta = natural_coords
        N = np.zeros(20)
        dN_dxi = np.zeros(20)
        dN_deta = np.zeros(20)
        dN_dzeta = np.zeros(20)
        for i in range(20):
            xii, etai, zetai = self._xi_i[i], self._eta_i[i], self._zeta_i[i]
            n_zero = (xii == 0.0) + (etai == 0.0) + (zetai == 0.0)
            if n_zero == 0:
                # corner node
                a, b, c = xi * xii, eta * etai, zeta * zetai
                N[i] = 0.125 * (1 + a) * (1 + b) * (1 + c) * (a + b + c - 2)
                dN_dxi[i] = 0.125 * xii * (1 + b) * (1 + c) * (2 * a + b + c - 1)
                dN_deta[i] = 0.125 * etai * (1 + a) * (1 + c) * (a + 2 * b + c - 1)
                dN_dzeta[i] = 0.125 * zetai * (1 + a) * (1 + b) * (a + b + 2 * c - 1)
            elif xii == 0.0:
                N[i] = 0.25 * (1 - xi**2) * (1 + eta * etai) * (1 + zeta * zetai)
                dN_dxi[i] = -0.5 * xi * (1 + eta * etai) * (1 + zeta * zetai)
                dN_deta[i] = 0.25 * (1 - xi**2) * etai * (1 + zeta * zetai)
                dN_dzeta[i] = 0.25 * (1 - xi**2) * (1 + eta * etai) * zetai
            elif etai == 0.0:
                N[i] = 0.25 * (1 + xi * xii) * (1 - eta**2) * (1 + zeta * zetai)
                dN_dxi[i] = 0.25 * xii * (1 - eta**2) * (1 + zeta * zetai)
                dN_deta[i] = -0.5 * eta * (1 + xi * xii) * (1 + zeta * zetai)
                dN_dzeta[i] = 0.25 * (1 + xi * xii) * (1 - eta**2) * zetai
            else:
                N[i] = 0.25 * (1 + xi * xii) * (1 + eta * etai) * (1 - zeta**2)
                dN_dxi[i] = 0.25 * xii * (1 + eta * etai) * (1 - zeta**2)
                dN_deta[i] = 0.25 * (1 + xi * xii) * etai * (1 - zeta**2)
                dN_dzeta[i] = -0.5 * zeta * (1 + xi * xii) * (1 + eta * etai)
        return N, np.vstack([dN_dxi, dN_deta, dN_dzeta])

    def B_matrix(self, natural_coords, elem_coords):
        _, dN_nat = self._cached_shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN_nat, elem_coords)
        dN_g = np.linalg.solve(J, dN_nat)   # (3,20)
        B = np.zeros((6, 60))
        for k in range(20):
            dx, dy, dz = dN_g[:, k]
            c = 3 * k
            B[0, c] = dx
            B[1, c + 1] = dy
            B[2, c + 2] = dz
            B[3, c] = dy; B[3, c + 1] = dx
            B[4, c + 1] = dz; B[4, c + 2] = dy
            B[5, c] = dz; B[5, c + 2] = dx
        return B, detJ


# =====================================================================
# Quad4 -- Reissner-Mindlin plate bending (needs SRI, so it overrides
# stiffness() instead of using the generic single-D Gauss loop)
# =====================================================================

class Tri3PlaneStress(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 3, 2, 2, 1
    quadrature_family = "simplex"   # Wave 15 item 125/126 -- see base.py's
                                     # Element.quadrature_family docstring.

    def shape_and_derivs(self, natural_coords):
        xi, eta = natural_coords
        N = np.array([1 - xi - eta, xi, eta])
        dN_dxi = np.array([-1.0, 1.0, 0.0])
        dN_deta = np.array([-1.0, 0.0, 1.0])
        return N, np.vstack([dN_dxi, dN_deta])

    def B_matrix(self, natural_coords, elem_coords):
        """Constant over the element (linear shape functions), so
        natural_coords doesn't actually matter -- kept as an argument
        only so this method has the same signature as every other
        element's B_matrix()."""
        _, dN_nat = self._cached_shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN_nat, elem_coords)
        dN_g = np.linalg.solve(J, dN_nat)   # (2,3): [dN/dx; dN/dy]
        B = np.zeros((3, 6))
        for k in range(3):
            B[0, 2 * k] = dN_g[0, k]
            B[1, 2 * k + 1] = dN_g[1, k]
            B[2, 2 * k] = dN_g[1, k]
            B[2, 2 * k + 1] = dN_g[0, k]
        return B, detJ

    def stiffness(self, elem_coords, D, thickness=1.0, **kwargs):
        """ke = Bᵀ D B * Area * thickness -- ONE evaluation (any
        natural_coords works, B is constant), not a Gauss loop. detJ
        here is twice the physical area (the natural triangle
        {xi,eta>=0, xi+eta<=1} has area 1/2), matching the standard
        CST convention."""
        B, detJ = self.B_matrix((1.0 / 3, 1.0 / 3), elem_coords)
        area = abs(detJ) * 0.5
        return (B.T @ D @ B) * area * thickness

    def full_stiffness(self, elem_coords, D, thickness=1.0):
        """N/A for this element -- constant strain, nothing for a
        quadrature order to refine. See TrussTL2D/Beam2DEulerBernoulli
        for the same pattern."""
        return self.stiffness(elem_coords, D, thickness)

    def reduced_stiffness(self, elem_coords, D, thickness=1.0):
        return self.stiffness(elem_coords, D, thickness)

    @staticmethod
    def _area(elem_coords):
        (x1, y1), (x2, y2), (x3, y3) = elem_coords
        return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def mass(self, elem_coords, rho_matrix, thickness=1.0):
        """Classic closed-form CST consistent mass: integral_A N_i N_j
        dA = (Area/12)*(1 + delta_ij) for LINEAR simplex shape
        functions -- a standard textbook result, not derived from the
        (inapplicable) generic Gauss loop. rho_matrix may be a scalar
        (isotropic rho) or a (2,2) block, matching the convention used
        elsewhere in this module."""
        rho_matrix = np.asarray(rho_matrix, dtype=float)
        if rho_matrix.ndim == 0:
            rho_matrix = rho_matrix * np.eye(2)
        area = self._area(elem_coords)
        Mscalar = (area * thickness / 12.0) * np.array(
            [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
        M = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                M[2 * i:2 * i + 2, 2 * j:2 * j + 2] = Mscalar[i, j] * rho_matrix
        return M

    def lumped_mass(self, elem_coords, rho_matrix, thickness=1.0):
        """Same HRZ idea as the base class (rescale the consistent
        mass's diagonal to preserve exact total mass), but computed
        from the closed-form Area above instead of the (inapplicable)
        Gauss-integrated total mass the base class's version uses.
        diag.sum() spans BOTH dof directions (x and y), so the target
        total must too -- dofs_per_node * (rho*Area*thickness) is the
        combined mass across both directions, not just one (a real bug
        caught by validate_simplex_elements.py CHECK 3: using a single
        direction's mass here silently halved every lumped entry)."""
        me = self.mass(elem_coords, rho_matrix, thickness)
        diag = np.diag(me).copy()
        rho_scalar = rho_matrix if np.ndim(rho_matrix) == 0 else np.asarray(rho_matrix)[0, 0]
        total_mass_all_directions = self.dofs_per_node * rho_scalar * self._area(elem_coords) * thickness
        c = total_mass_all_directions / diag.sum()
        return np.diag(diag * c)


# =====================================================================
# 4-node linear tetrahedron (Tet4), Module 12: the 3-D counterpart to
# Tri3PlaneStress above -- same reasoning applies (constant strain,
# closed-form stiffness/mass, NOT the tensor-product Gauss loop, since
# the natural tetrahedron {xi,eta,zeta>=0, xi+eta+zeta<=1} is a
# simplex, not a cube).
# =====================================================================
class Tet4Solid3D(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 4, 3, 3, 1
    quadrature_family = "simplex"   # Wave 15 item 125/126 -- see base.py's
                                     # Element.quadrature_family docstring.

    def shape_and_derivs(self, natural_coords):
        xi, eta, zeta = natural_coords
        N = np.array([1 - xi - eta - zeta, xi, eta, zeta])
        dN_dxi = np.array([-1.0, 1.0, 0.0, 0.0])
        dN_deta = np.array([-1.0, 0.0, 1.0, 0.0])
        dN_dzeta = np.array([-1.0, 0.0, 0.0, 1.0])
        return N, np.vstack([dN_dxi, dN_deta, dN_dzeta])

    def B_matrix(self, natural_coords, elem_coords):
        """Constant over the element -- natural_coords doesn't matter,
        kept only for interface consistency (see Tri3PlaneStress)."""
        _, dN_nat = self._cached_shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN_nat, elem_coords)
        dN_g = np.linalg.solve(J, dN_nat)   # (3,4)
        B = np.zeros((6, 12))
        for k in range(4):
            dx, dy, dz = dN_g[:, k]
            c = 3 * k
            B[0, c] = dx
            B[1, c + 1] = dy
            B[2, c + 2] = dz
            B[3, c] = dy; B[3, c + 1] = dx
            B[4, c + 1] = dz; B[4, c + 2] = dy
            B[5, c] = dz; B[5, c + 2] = dx
        return B, detJ

    def stiffness(self, elem_coords, D, thickness=1.0, **kwargs):
        """ke = Bᵀ D B * Volume -- ONE evaluation, not a Gauss loop
        (see Tri3PlaneStress's docstring for why). detJ here is 6x the
        physical volume (the natural tetrahedron {xi,eta,zeta>=0,
        xi+eta+zeta<=1} has volume 1/6), the standard Tet4 convention."""
        B, detJ = self.B_matrix((0.25, 0.25, 0.25), elem_coords)
        volume = abs(detJ) / 6.0
        return (B.T @ D @ B) * volume * thickness

    def full_stiffness(self, elem_coords, D, thickness=1.0):
        return self.stiffness(elem_coords, D, thickness)

    def reduced_stiffness(self, elem_coords, D, thickness=1.0):
        return self.stiffness(elem_coords, D, thickness)

    @staticmethod
    def _volume(elem_coords):
        X = np.asarray(elem_coords, dtype=float)
        return abs(np.linalg.det(X[1:] - X[0])) / 6.0

    def mass(self, elem_coords, rho_matrix, thickness=1.0):
        """Classic closed-form Tet4 consistent mass: integral_V N_i N_j
        dV = (Volume/20)*(1 + delta_ij) for linear simplex shape
        functions -- the 3-D analogue of Tri3PlaneStress's Area/12
        result."""
        rho_matrix = np.asarray(rho_matrix, dtype=float)
        if rho_matrix.ndim == 0:
            rho_matrix = rho_matrix * np.eye(3)
        vol = self._volume(elem_coords)
        Mscalar = (vol / 20.0) * np.array([
            [2.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 2.0]])
        M = np.zeros((12, 12))
        for i in range(4):
            for j in range(4):
                M[3 * i:3 * i + 3, 3 * j:3 * j + 3] = Mscalar[i, j] * rho_matrix
        return M

    def lumped_mass(self, elem_coords, rho_matrix, thickness=1.0):
        """See Tri3PlaneStress.lumped_mass()'s docstring -- same fix,
        dofs_per_node directions' worth of mass, not just one."""
        me = self.mass(elem_coords, rho_matrix, thickness)
        diag = np.diag(me).copy()
        rho_scalar = rho_matrix if np.ndim(rho_matrix) == 0 else np.asarray(rho_matrix)[0, 0]
        total_mass_all_directions = self.dofs_per_node * rho_scalar * self._volume(elem_coords)
        c = total_mass_all_directions / diag.sum()
        return np.diag(diag * c)


# =====================================================================
# Tet10 -- quadratic (10-node) tetrahedron, Module 15: the quadratic
# counterpart to Tet4Solid3D above. 4 corner nodes (same convention as
# Tet4) + 6 mid-edge nodes. Built from BARYCENTRIC shape functions
# (L1=1-r-s-t, L2=r, L3=s, L4=t -- the standard simplex generalization
# of Tri3's area coordinates): corner N_i = L_i(2L_i-1), edge node
# between corners i,j: N = 4*L_i*L_j (Cook/Malkus/Plesha;
# Zienkiewicz & Taylor). Edge-to-node map (0-indexed corners):
# node4=(0,1), node5=(1,2), node6=(2,0), node7=(0,3), node8=(1,3),
# node9=(2,3).
#
# GMSH COMPATIBILITY WARNING (found + fixed 2026-08-30, same
# investigation as Hex20Solid3D's note below): this class's own node
# order above is ALMOST, but not quite, Gmsh's native 10-node
# tetrahedron (element type 11) order -- corners 0-3 and mid-edge nodes
# 4-7 (edges (0,1),(1,2),(2,0),(0,3)) match directly, but Gmsh's real
# nodes 8 and 9 are SWAPPED relative to the (1,3),(2,3) order above:
# Gmsh actually emits node8=(2,3), node9=(1,3). Verified directly (not
# assumed) against 100/100 elements of a real Gmsh-generated Tet10 mesh
# by comparing each mid-edge node's coordinate to the arithmetic
# midpoint its assumed edge implies -- feeding Gmsh's raw connectivity
# in as-is (which several example/analysis scripts in this project's
# ecosystem previously did, having verified only that corners 0-3 and
# the FIRST four mid-edge nodes matched) leaves a small, easy-to-miss
# corruption: a Jacobian that flips sign at one of the four quadrature
# points in every element, and a consistent mass matrix that fails the
# exact partition-of-unity identity dᵀMd=rho*Volume (for a unit rigid
# translation d) by a large, non-noise margin. GMSH_NODE_ORDER below is
# the corrected permutation (identity except for a 8<->9 swap); any
# caller building a mesh from Gmsh's own Tet10 output should reindex
# through it (gmsh_conn[:, GMSH_NODE_ORDER]) rather than assuming
# corners-then-edges-as-listed-above is Gmsh-ready.
#
# UNLIKE Hex20/Quad8, this does NOT reuse the generic Gauss loop --
# gauss_product() builds a tensor-product CUBE grid on [-1,1]^dim,
# which is the wrong reference domain for a SIMPLEX element (see
# tet_quadrature_4pt()'s docstring in base.py). Tet10 overrides
# stiffness()/mass() directly, the same pattern Tri3PlaneStress/
# Tet4Solid3D already established for the linear simplex case, just
# with a genuine 4-point quadrature loop instead of their single-point
# evaluation (Tet10's quadratic shape functions make B vary over the
# element -- unlike Tet4's constant-strain B -- so one point is no
# longer enough; 4 points is the minimum that integrates the resulting
# degree-2 B^T D B integrand exactly for a straight-sided element,
# where the isoparametric map stays affine and detJ stays constant
# despite the quadratic shape functions -- see tet_quadrature_4pt()).
# =====================================================================
class Tet10Solid3D(Element):
    n_nodes, dofs_per_node, dim = 10, 3, 3
    quadrature_family = "simplex"   # Wave 15 item 125/126 -- see base.py's
                                     # Element.quadrature_family docstring.

    # Permutation mapping Gmsh's native 10-node tetrahedron (element
    # type 11) node order to this class's own order:
    # gmsh_conn[:, GMSH_NODE_ORDER] reindexes a raw Gmsh connectivity
    # array into the order shape_and_derivs()/mass()/stiffness() expect.
    # NOT the identity, despite corners 0-3 and mid-edges 4-7 matching
    # directly -- see the "GMSH COMPATIBILITY WARNING" note above this
    # class for the derivation/verification (nodes 8 and 9 are swapped).
    GMSH_NODE_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]

    _edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]

    def shape_and_derivs(self, natural_coords):
        r, s, t = natural_coords
        L = np.array([1 - r - s - t, r, s, t])
        dLdr = np.array([-1.0, 1.0, 0.0, 0.0])
        dLds = np.array([-1.0, 0.0, 1.0, 0.0])
        dLdt = np.array([-1.0, 0.0, 0.0, 1.0])

        N = np.zeros(10)
        dN_dr = np.zeros(10)
        dN_ds = np.zeros(10)
        dN_dt = np.zeros(10)

        for i in range(4):
            N[i] = L[i] * (2 * L[i] - 1)
            coef = 4 * L[i] - 1
            dN_dr[i] = coef * dLdr[i]
            dN_ds[i] = coef * dLds[i]
            dN_dt[i] = coef * dLdt[i]

        for k, (a, b) in enumerate(self._edges):
            i = 4 + k
            N[i] = 4 * L[a] * L[b]
            dN_dr[i] = 4 * (dLdr[a] * L[b] + L[a] * dLdr[b])
            dN_ds[i] = 4 * (dLds[a] * L[b] + L[a] * dLds[b])
            dN_dt[i] = 4 * (dLdt[a] * L[b] + L[a] * dLdt[b])

        return N, np.vstack([dN_dr, dN_ds, dN_dt])

    def B_matrix(self, natural_coords, elem_coords):
        _, dN_nat = self._cached_shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN_nat, elem_coords)
        dN_g = np.linalg.solve(J, dN_nat)   # (3,10)
        B = np.zeros((6, 30))
        for k in range(10):
            dx, dy, dz = dN_g[:, k]
            c = 3 * k
            B[0, c] = dx
            B[1, c + 1] = dy
            B[2, c + 2] = dz
            B[3, c] = dy; B[3, c + 1] = dx
            B[4, c + 1] = dz; B[4, c + 2] = dy
            B[5, c] = dz; B[5, c + 2] = dx
        return B, detJ

    def stiffness(self, elem_coords, D, thickness=1.0, quad_order=None, **kwargs):
        """ke = sum over the tet quadrature of Bᵀ D B * |J| * w * (1/6)
        -- see tet_quadrature_4pt()'s docstring for why this isn't the
        generic Gauss loop / gauss_product(). The 1/6 is the SAME
        "natural tetrahedron {r,s,t>=0, r+s+t<=1} has volume 1/6"
        factor Tet4Solid3D.stiffness() applies explicitly via its
        `volume = abs(detJ) / 6.0` -- both tet_quadrature_4pt() and
        tet_quadrature(order)'s weights are barycentric-normalized (sum
        to 1, the standard way the literature states this rule), so
        this factor has to be applied by the caller, exactly like
        Tet4's single-point rule already does with its own implicit
        weight of 1.

        quad_order: None (default) keeps the ORIGINAL fixed 4-point
        rule (tet_quadrature_4pt(), degree-2-exact) -- no behavior
        change for any existing caller. Pass an explicit int (Wave 15
        item 125, docs/consolidated_future_roadmap.md) to request
        tet_quadrature(quad_order) instead -- e.g. quad_order=4 for a
        genuinely quartic-exact integral, useful for mass() below
        where the N^T*rho*N integrand really is quartic and the
        default 4-point rule is only an approximation to it (see
        mass()'s own docstring)."""
        points, weights = (tet_quadrature_4pt() if quad_order is None
                            else tet_quadrature(quad_order))
        n_total = self.n_nodes * self.dofs_per_node
        ke = np.zeros((n_total, n_total))
        for p, w in zip(points, weights):
            B, detJ = self.B_matrix(p, elem_coords)
            ke += (B.T @ D @ B) * abs(detJ) * w / 6.0
        return ke

    def full_stiffness(self, elem_coords, D, thickness=1.0):
        """N/A -- this element has one quadrature scheme (the 4-point
        rule, already exact for its straight-sided integrand), not a
        full/reduced distinction. Same pattern as Tri3PlaneStress/
        Tet4Solid3D."""
        return self.stiffness(elem_coords, D, thickness)

    def reduced_stiffness(self, elem_coords, D, thickness=1.0):
        return self.stiffness(elem_coords, D, thickness)

    def mass(self, elem_coords, rho_matrix, thickness=1.0, quad_order=None):
        """Consistent mass via the SAME quadrature stiffness() uses (N
        is quadratic, so Nᵀ rho N is quartic -- the default 4-point
        rule is only exact to degree 2, so with quad_order=None this
        mass matrix is a good, standard engineering approximation, not
        machine-precision exact). Pass quad_order=4 (Wave 15 item 125)
        for a genuinely quartic-exact consistent mass matrix instead --
        tests/test_quadrature_order.py confirms directly that this
        actually changes the result (vs. quad_order=None) and that the
        quad_order=4 result stops changing under further refinement
        (quad_order=6), the decisive signature of having reached the
        integrand's own exact degree rather than just "a bigger
        number of points"."""
        rho_matrix = np.asarray(rho_matrix, dtype=float)
        if rho_matrix.ndim == 0:
            rho_matrix = rho_matrix * np.eye(3)
        points, weights = (tet_quadrature_4pt() if quad_order is None
                            else tet_quadrature(quad_order))
        n_total = self.n_nodes * self.dofs_per_node
        me = np.zeros((n_total, n_total))
        for p, w in zip(points, weights):
            N, dN_nat = self._cached_shape_and_derivs(p)
            _, detJ = jacobian(dN_nat, elem_coords)
            Nm = self.N_matrix(N)
            me += (Nm.T @ rho_matrix @ Nm) * abs(detJ) * w / 6.0
        return me


# =====================================================================
# Tri6 -- 6-node quadratic-triangle plane-stress element, Wave 0 item 7
# (docs/consolidated_future_roadmap.md, source geometry_meshing_
# alternatives_research.md item 4): row-3 parity with the existing
# Quad8PlaneStress -- this package had a quadratic quad (Quad8) and a
# quadratic tet/hex (Tet10/Hex20) but no quadratic triangle, the plane-
# stress analogue of Tet10Solid3D one dimension down. Explicitly "not a
# prerequisite" for anything else on the roadmap; touches only this
# file (no gmsh_engine.py extraction-table wiring yet -- that is
# backlog item 30, deliberately deferred until an actual Gmsh-sourced
# Tri6 mesh is needed).
#
# Node convention: corners 0,1,2 (same xi/eta convention as
# Tri3PlaneStress: L0=1-xi-eta, L1=xi, L2=eta), then mid-edge nodes
# 3,4,5 on edges (0,1), (1,2), (2,0) respectively -- the standard
# quadratic-triangle layout (Cook/Malkus/Plesha; Zienkiewicz & Taylor),
# and also the conventional Gmsh 6-node triangle (element type 9) node
# order, though this element is not yet wired into geometry/
# gmsh_engine.py's extraction table (see above) so that correspondence
# is not exercised by anything here.
#
# Like Tri3PlaneStress/Tet4Solid3D/Tet10Solid3D, this does NOT use the
# generic Gauss loop -- gauss_product() builds a tensor-product SQUARE
# grid on [-1,1]^2, the wrong reference domain for a triangle (see
# tri_quadrature_3pt()'s docstring in base.py). Tri6 overrides
# stiffness()/mass() directly with the SAME 3-point simplex quadrature
# pattern Tet10Solid3D established one dimension up: quadratic shape
# functions make B vary over the element (unlike Tri3's constant-strain
# B), so its B^T D B integrand is degree 2 for a straight-sided element
# (constant detJ) -- exactly what tri_quadrature_3pt() integrates
# exactly.
# =====================================================================
# GMSH COMPATIBILITY (checked 2026-09-08, Wave 5 item 30 of docs/
# consolidated_future_roadmap.md, same investigation as Quad8Plane
# Stress/Hex20Solid3D/Tet10Solid3D's own notes, done BEFORE wiring this
# element into geometry/gmsh_engine.py's extraction table -- per this
# project's own established practice of checking a real Gmsh-built
# reference element directly rather than trusting Gmsh's documented
# ordering table, exactly as Hex20/Tet10 turned out to need a real fix
# despite looking "close" on paper). Like Quad8, this one is CLEAN: a
# single reference Tri6 built through Gmsh (a right-triangle domain,
# SecondOrderIncomplete=1, setOrder(2) -- element type 9) comes back
# with corners 0-2 at (0,0)/(1,0)/(0,1) and mid-edge nodes 3-5 at
# (0.5,0)/(0.5,0.5)/(0,0.5) -- i.e. node 3 on edge (0,1), node 4 on edge
# (1,2), node 5 on edge (2,0), EXACTLY this class's own `_edges` order
# above. GMSH_NODE_ORDER is the identity; kept explicit (not omitted)
# for the same reason Quad8PlaneStress's own identity permutation is:
# gmsh_engine.py's lookup treats every quadratic element uniformly
# (look up GMSH_NODE_ORDER, apply it) rather than special-casing "this
# one happens to need no reindexing."
# =====================================================================
class Tri6PlaneStress(Element):
    n_nodes, dofs_per_node, dim = 6, 2, 2
    quadrature_family = "simplex"   # Wave 15 item 125/126 -- see base.py's
                                     # Element.quadrature_family docstring.

    # Permutation mapping Gmsh's native 6-node triangle (element type 9)
    # node order to this class's own order -- the identity, verified
    # directly above (not assumed from Gmsh's documentation).
    GMSH_NODE_ORDER = [0, 1, 2, 3, 4, 5]

    _edges = [(0, 1), (1, 2), (2, 0)]

    def shape_and_derivs(self, natural_coords):
        xi, eta = natural_coords
        L = np.array([1 - xi - eta, xi, eta])
        dLdxi = np.array([-1.0, 1.0, 0.0])
        dLdeta = np.array([-1.0, 0.0, 1.0])

        N = np.zeros(6)
        dN_dxi = np.zeros(6)
        dN_deta = np.zeros(6)

        for i in range(3):
            N[i] = L[i] * (2 * L[i] - 1)
            coef = 4 * L[i] - 1
            dN_dxi[i] = coef * dLdxi[i]
            dN_deta[i] = coef * dLdeta[i]

        for k, (a, b) in enumerate(self._edges):
            i = 3 + k
            N[i] = 4 * L[a] * L[b]
            dN_dxi[i] = 4 * (dLdxi[a] * L[b] + L[a] * dLdxi[b])
            dN_deta[i] = 4 * (dLdeta[a] * L[b] + L[a] * dLdeta[b])

        return N, np.vstack([dN_dxi, dN_deta])

    def B_matrix(self, natural_coords, elem_coords):
        _, dN_nat = self._cached_shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN_nat, elem_coords)
        dN_g = np.linalg.solve(J, dN_nat)   # (2,6): [dN/dx; dN/dy]
        B = np.zeros((3, 12))
        for k in range(6):
            B[0, 2 * k] = dN_g[0, k]
            B[1, 2 * k + 1] = dN_g[1, k]
            B[2, 2 * k] = dN_g[1, k]
            B[2, 2 * k + 1] = dN_g[0, k]
        return B, detJ

    def stiffness(self, elem_coords, D, thickness=1.0, quad_order=None, **kwargs):
        """ke = sum over the triangle quadrature of Bᵀ D B * |detJ| *
        w * (1/2) -- see tri_quadrature_3pt()'s docstring for why this
        isn't the generic Gauss loop / gauss_product(). The 1/2 is the
        SAME "natural triangle {xi,eta>=0, xi+eta<=1} has area 1/2"
        factor Tri3PlaneStress.stiffness() applies explicitly via its
        own `area = abs(detJ) * 0.5` -- both tri_quadrature_3pt() and
        tri_quadrature(order)'s weights are barycentric-normalized (sum
        to 1, the standard way the literature states this rule), so
        this factor has to be applied by the caller, exactly like
        Tet10Solid3D.stiffness() does with its own implicit 1/6
        (natural-tetrahedron-volume) weight.

        quad_order: None (default) keeps the ORIGINAL fixed 3-point
        rule (tri_quadrature_3pt(), degree-2-exact) -- no behavior
        change for any existing caller. Pass an explicit int (Wave 15
        item 125, docs/consolidated_future_roadmap.md) to request
        tri_quadrature(quad_order) instead -- see Tet10Solid3D.
        stiffness()'s own docstring for the mass()-integrand rationale
        this mirrors exactly, one simplex dimension down."""
        points, weights = (tri_quadrature_3pt() if quad_order is None
                            else tri_quadrature(quad_order))
        n_total = self.n_nodes * self.dofs_per_node
        ke = np.zeros((n_total, n_total))
        for p, w in zip(points, weights):
            B, detJ = self.B_matrix(p, elem_coords)
            ke += (B.T @ D @ B) * abs(detJ) * w * 0.5 * thickness
        return ke

    def full_stiffness(self, elem_coords, D, thickness=1.0):
        """N/A -- this element has one quadrature scheme (the 3-point
        rule, already exact for its straight-sided integrand), not a
        full/reduced distinction. Same pattern as Tri3PlaneStress/
        Tet4Solid3D/Tet10Solid3D."""
        return self.stiffness(elem_coords, D, thickness)

    def reduced_stiffness(self, elem_coords, D, thickness=1.0):
        return self.stiffness(elem_coords, D, thickness)

    def mass(self, elem_coords, rho_matrix, thickness=1.0, quad_order=None):
        """Consistent mass via the SAME quadrature stiffness() uses (N
        is quadratic, so Nᵀ rho N is quartic -- with quad_order=None
        the default 3-point rule is only exact to degree 2, so this
        mass matrix is a good, standard engineering approximation, not
        machine-precision exact -- identical reasoning to Tet10Solid3D.
        mass(), one simplex dimension down). Pass quad_order=4 (Wave 15
        item 125) for a genuinely quartic-exact consistent mass matrix
        instead -- see Tet10Solid3D.mass()'s own docstring for the
        decisive verification this mirrors."""
        rho_matrix = np.asarray(rho_matrix, dtype=float)
        if rho_matrix.ndim == 0:
            rho_matrix = rho_matrix * np.eye(2)
        points, weights = (tri_quadrature_3pt() if quad_order is None
                            else tri_quadrature(quad_order))
        n_total = self.n_nodes * self.dofs_per_node
        me = np.zeros((n_total, n_total))
        for p, w in zip(points, weights):
            N, dN_nat = self._cached_shape_and_derivs(p)
            _, detJ = jacobian(dN_nat, elem_coords)
            Nm = self.N_matrix(N)
            me += (Nm.T @ rho_matrix @ Nm) * abs(detJ) * w * 0.5 * thickness
        return me


# =====================================================================
# 1-node contact element vs a FIXED RIGID CIRCLE, Module 13: closes two
# more items from the contact-nonlinearity taxonomy that
# GapContactPenalty (Module 10) deliberately left out --
#
#   (a) an UPDATING contact normal: n = n(u), not a fixed direction.
#       GapContactPenalty's obstacle is a flat wall, whose normal is
#       the same everywhere on it by definition -- nothing to update.
#       A curved obstacle is the simplest geometry where the normal
#       genuinely depends on where on the obstacle the node currently
#       sits, i.e. on u.
#   (b) Coulomb friction (stick/slip): a tangential traction capped at
#       mu*pn. Coulomb friction and J2 plasticity are structurally the
#       SAME problem -- a linear trial response capped at a pressure-
#       dependent limit -- so this reuses Module 9's state/
#       commit_state/return-mapping pattern (TrussPlastic2D) almost
#       unchanged, with mu*pn playing the role sigma_y+H*alpha played
#       there, and a tangential "stick anchor" arc-length playing the
#       role of the committed plastic strain.
#
# Scoped to 2-D (a circular obstacle): the tangential coordinate here
# is arc length around the circle (s = R*theta), which has no
# equally simple analogue for a 3-D sphere/cylinder (geodesics, pole
# singularities) -- a 3-D version is real future work, not attempted
# here.
#
# SIGN CONVENTION, worth getting right explicitly: like every other
# element in this package, internal_force() = dU/du for some one-sided
# potential U -- NOT "the force that pushes the node away from the
# obstacle." Using U = (1/2)*k_p*(R-d)^2 (active only when the node
# has penetrated, d<R) gives f_normal = k_p*(d-R)*n_hat = -pn*n_hat
# (pn = k_p*(R-d) >= 0, the contact PRESSURE magnitude) -- i.e. the
# internal force points TOWARD the center (the "more penetration"
# direction) when active, exactly the same convention GapContactPenalty
# uses (its internal force points further INTO the wall, not away from
# it) and TrussTL2D/TrussPlastic2D use for a stretched bar. Getting this
# backwards (an earlier draft did, caught while re-deriving it against
# this same potential-energy check before writing any code) doesn't
# change whether an ISOLATED call to internal_force() "looks reasonable"
# -- it only shows up as wrong equilibrium once solved through
# FESystem/nonlinear_solver.py, which is exactly why every element in
# this package is derived from a potential/virtual-work argument, not
# assembled from an intuition about "which way the force should point."
#
# tangent_stiffness() is computed NUMERICALLY (central difference of
# internal_force()) rather than hand-derived analytically -- deliberate,
# not a shortcut: the combined stick/slip + rotating-normal kinematics
# makes an analytical consistent tangent intricate and easy to get
# subtly wrong, and a numerical tangent is *exactly* consistent with
# internal_force() by construction. The frictionless (mu=0) special
# case still has closed form -- K_T = k_p*(1-R/d)*I + k_p*(R/d)*(n_hat
# (x) n_hat), also derived from the same potential -- used in
# validate_curved_contact.py CHECK 2 to independently confirm the
# numerical tangent isn't just self-consistent but actually correct.
# Note this closed form has a NEGATIVE eigenvalue along the tangential
# direction while active (k_p*(1-R/d) < 0 since R/d>1 when d<R) -- a
# real, known curvature effect for penalty contact against a CONVEX
# obstacle (as the contact point slides tangentially at fixed
# penetration, the radially-directed force vector rotates with it),
# not a bug; Newton-Raphson only needs K_T invertible at each iterate
# (true except exactly at d=R), not positive-definite.
# =====================================================================
