"""
beams.py -- 2-D beam elements: Beam2DEulerBernoulli (linear) and
Beam2DCorotational (geometrically nonlinear, large rotation).

Split out of the original monolithic element.py during the
fea_engine restructuring; no logic changed, only file location.
"""
import numpy as np

from .base import Element


class Beam2DEulerBernoulli(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 2, 2, 1, 1
    translational_dof_mask = [True, False]   # [v, theta]
    # Note: lumped_mass() is inherited from Element unmodified -- it
    # only needs self.mass() (defined below, closed-form) and
    # self.shape_and_derivs()/gauss_order (defined below too, the
    # linear 1-D interpolation), so the generic HRZ implementation
    # works here without an override.

    def shape_and_derivs(self, natural_coords):
        """Linear interpolation of x itself -- used only by
        mesh.check_quality() for a length-positivity sanity check, NOT
        by stiffness()/mass() (those use the exact closed-form Hermite
        matrices, since EI/rho*A are constant per element)."""
        (xi,) = natural_coords
        N = np.array([(1 - xi) / 2, (1 + xi) / 2])
        dN_dxi = np.array([-0.5, 0.5])
        return N, dN_dxi.reshape(1, 2)

    def stiffness(self, elem_coords, EI, thickness=1.0):
        le = float(elem_coords[1, 0] - elem_coords[0, 0])
        return (EI / le**3) * np.array([
            [12,     6*le,    -12,     6*le],
            [6*le,   4*le**2, -6*le,   2*le**2],
            [-12,   -6*le,     12,    -6*le],
            [6*le,   2*le**2, -6*le,   4*le**2]])

    def full_stiffness(self, elem_coords, EI, thickness=1.0):
        """N/A for this element -- EI is constant per element, so the
        closed-form Hermite matrix above IS the exact integral; there
        is no quadrature order to vary. Provided only so this element
        satisfies the same interface as the others (e.g. for a generic
        loop over ELEMENT_REGISTRY)."""
        return self.stiffness(elem_coords, EI, thickness)

    def reduced_stiffness(self, elem_coords, EI, thickness=1.0):
        """Same result as full_stiffness() -- see its docstring."""
        return self.stiffness(elem_coords, EI, thickness)

    def mass(self, elem_coords, rho_A, thickness=1.0):
        le = float(elem_coords[1, 0] - elem_coords[0, 0])
        return (rho_A * le / 420.0) * np.array([
            [156,     22*le,     54,    -13*le],
            [22*le,   4*le**2,   13*le, -3*le**2],
            [54,      13*le,     156,   -22*le],
            [-13*le, -3*le**2,  -22*le,  4*le**2]])

    def geometric_stiffness(self, elem_coords, N, thickness=1.0):
        """Module 19 (general-purpose extensions Phase 4): the
        standard consistent geometric ("stress stiffness") matrix for
        a 2-node Euler-Bernoulli beam-column under a constant axial
        force N (TENSION-POSITIVE, see Element.geometric_stiffness's
        docstring for the sign convention and why) -- the closed-form
        Przemieniecki/Cook matrix (same dof order as stiffness():
        v1, theta1, v2, theta2), derived from the SAME second-order
        (nonlinear) part of the curvature-displacement relation that
        underlies large-deflection beam theory, just evaluated once
        at a fixed reference N instead of updated every iteration --
        the linear-buckling analogue of TrussTL2D's K_geometric =
        (S*A/L0)*I term for the axial element."""
        le = float(elem_coords[1, 0] - elem_coords[0, 0])
        return (N / (30.0 * le)) * np.array([
            [36,      3*le,     -36,      3*le],
            [3*le,    4*le**2,  -3*le,   -le**2],
            [-36,    -3*le,      36,     -3*le],
            [3*le,   -le**2,    -3*le,    4*le**2]])


# =====================================================================
# 2-node Total-Lagrangian large-displacement truss (Module 8: geometric
# nonlinearity, "large displacement / small-to-moderate strain" case --
# a cable or rod that can swing through large rigid-body rotation while
# the material itself barely stretches, e.g. the "cable swaying" /
# "fishing rod" examples). St Venant-Kirchhoff material: 2nd
# Piola-Kirchhoff stress S is linear in Green-Lagrange strain E_GL, so
# ALL the nonlinearity here is geometric (E_GL itself is a quadratic
# function of displacement), none of it is material -- the cleanest
# possible isolation of Module 8's "Scripting Impact" (Green-Lagrange
# strain + Piola-Kirchhoff stress replacing the linear engineering
# strain / Cauchy stress pair every other element in this package
# uses). This element intentionally overrides internal_force() and
# tangent_stiffness() directly (closed-form, not a Gauss loop over a
# generic B_matrix) -- exact, and the standard textbook derivation
# (e.g. Crisfield, Nonlinear FE of Solids and Structures Vol 1, ch 3),
# so it doubles as an easy hand-derivable check on nonlinear_solver.py.
# =====================================================================

class Beam2DCorotational(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 2, 3, 1, 1
    translational_dof_mask = [True, True, False]   # [u, v, theta]

    def shape_and_derivs(self, natural_coords):
        """Linear interpolation along the (reference) bar axis --
        interface completeness only (mesh.check_quality()), same as
        TrussTL2D; internal_force()/tangent_stiffness() below don't use
        this."""
        (xi,) = natural_coords
        N = np.array([(1 - xi) / 2, (1 + xi) / 2])
        dN_dxi = np.array([-0.5, 0.5])
        return N, dN_dxi.reshape(1, 2)

    def mass(self, elem_coords, rho_A, thickness=1.0):
        """Consistent mass matrix in the REFERENCE (undeformed)
        configuration -- standard linear elastodynamics mass, like
        every other element in this package (no geometric/nonlinear
        correction to the mass itself, only to the stiffness). Added
        for Module 15's flat-beam benchmark (He et al. 2023 Sect.
        4.1), which needs modal frequencies, not just the static
        elastica/cantilever checks validate_nonlinear_beam.py already
        covers -- this element originally had NO mass() override,
        silently falling back to the generic Element.mass() Gauss loop,
        which is WRONG here: that loop uses N_matrix() to apply the
        SAME (linear, 2-node) shape function to all 3 dofs/node,
        including theta -- physically nonsensical for a Hermite beam,
        where transverse displacement and rotation are coupled through
        cubic shape functions, not independently linearly interpolated.

        rho_A: scalar rho*A (same convention as Beam2DEulerBernoulli.mass()
        -- this formulation, like that one, has no separate rotary-
        inertia term in its classic 4x4 bending block).

        Unlike Beam2DEulerBernoulli (which assumes the element lies
        along local x, correct only for a horizontal 1-D mesh), this
        element supports ARBITRARY planar orientation (that's the whole
        point of the corotational formulation), so the LOCAL axial+
        bending mass matrix (built as if the beam were horizontal) is
        rotated into global coordinates via the REFERENCE chord angle
        phi0 = atan2(dY, dX) -- the same rotation internal_force()/
        tangent_stiffness() apply implicitly through c,s at u_elem=0.
        Standard 2-D frame-element mass/stiffness transformation:
        M_global = T @ M_local @ T.T, where T is block-diagonal in the
        two nodes' own 3x3 rotation blocks (u,v rotate; theta, a
        rotation about the shared out-of-plane z-axis, does not).

        Verified (outside this docstring, in flat_beam_fem.py's build
        step / a standalone sanity check): at zero rotation this exactly
        matches Beam2DEulerBernoulli.mass()'s bending sub-block; total
        mass is exactly rho_A*L0 in both the axial and translational-
        bending row sums; eigenvalues of M are identical (to 1e-10) at
        any rotation angle, confirming T@M_local@T.T is mass-preserving;
        M is symmetric and positive-definite."""
        X1, X2 = elem_coords[0], elem_coords[1]
        dX = X2 - X1
        L0 = np.linalg.norm(dX)
        c0, s0 = dX[0] / L0, dX[1] / L0

        m_ax = (rho_A * L0 / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        m_bend = (rho_A * L0 / 420.0) * np.array([
            [156, 22 * L0, 54, -13 * L0],
            [22 * L0, 4 * L0**2, 13 * L0, -3 * L0**2],
            [54, 13 * L0, 156, -22 * L0],
            [-13 * L0, -3 * L0**2, -22 * L0, 4 * L0**2]])
        M_local = np.zeros((6, 6))
        ax_idx = [0, 3]
        bend_idx = [1, 2, 4, 5]
        for a, i in enumerate(ax_idx):
            for b, j in enumerate(ax_idx):
                M_local[i, j] = m_ax[a, b]
        for a, i in enumerate(bend_idx):
            for b, j in enumerate(bend_idx):
                M_local[i, j] = m_bend[a, b]

        T_node = np.array([[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]])
        T = np.zeros((6, 6))
        T[0:3, 0:3] = T_node
        T[3:6, 3:6] = T_node
        return T @ M_local @ T.T

    @staticmethod
    def _wrap(angle):
        """Principal value in (-pi, pi] -- keeps the "relative to
        chord" rotations theta_bar well-defined (a physically periodic
        quantity) regardless of how theta1/theta2/phi individually
        wind during Newton iteration."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def _kinematics(cls, elem_coords, u_elem):
        """Returns (L0, L, c, s, e_bar, th1_bar, th2_bar): L0 the
        reference chord length, L/c/s the CURRENT (deformed) chord's
        length/cos/sin, and the three natural (corotational)
        deformations -- see the class docstring."""
        X1, X2 = elem_coords[0], elem_coords[1]
        dX = X2 - X1
        L0 = np.linalg.norm(dX)
        phi0 = np.arctan2(dX[1], dX[0])

        u1, v1, th1, u2, v2, th2 = u_elem
        dx = (X2 + np.array([u2, v2])) - (X1 + np.array([u1, v1]))
        L = np.linalg.norm(dx)
        phi = np.arctan2(dx[1], dx[0])
        c, s = dx[0] / L, dx[1] / L

        beta = cls._wrap(phi - phi0)
        e_bar = L - L0
        th1_bar = cls._wrap(th1 - beta)
        th2_bar = cls._wrap(th2 - beta)
        return L0, L, c, s, e_bar, th1_bar, th2_bar

    @staticmethod
    def _B_matrix(L, c, s):
        """3x6: d(e_bar, theta1_bar, theta2_bar) / d(u1,v1,th1,u2,v2,th2),
        evaluated at the CURRENT chord (c,s,L) -- see the class
        docstring for the derivation."""
        return np.array([
            [-c,    -s,   0.0,   c,    s,   0.0],
            [-s/L,  c/L,  1.0,  s/L, -c/L,  0.0],
            [-s/L,  c/L,  0.0,  s/L, -c/L,  1.0],
        ])

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """mat = (E, A, I). f_int = B^T @ [N, M1, M2] -- see class
        docstring for N/M1/M2 and the virtual-work argument for why
        this transformation is force-consistent."""
        E, A, I = mat
        L0, L, c, s, e_bar, th1_bar, th2_bar = self._kinematics(elem_coords, u_elem)
        EA_L0, EI_L0 = E * A / L0, E * I / L0
        N = EA_L0 * e_bar
        M1 = EI_L0 * (4 * th1_bar + 2 * th2_bar)
        M2 = EI_L0 * (2 * th1_bar + 4 * th2_bar)
        B = self._B_matrix(L, c, s)
        return B.T @ np.array([N, M1, M2])

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """K_T = B^T @ k_local @ B + K_geo -- see class docstring."""
        E, A, I = mat
        L0, L, c, s, e_bar, th1_bar, th2_bar = self._kinematics(elem_coords, u_elem)
        EA_L0, EI_L0 = E * A / L0, E * I / L0
        N = EA_L0 * e_bar
        M1 = EI_L0 * (4 * th1_bar + 2 * th2_bar)
        M2 = EI_L0 * (2 * th1_bar + 4 * th2_bar)
        M_sum = M1 + M2

        B = self._B_matrix(L, c, s)
        k_local = np.array([[EA_L0, 0.0,     0.0],
                             [0.0,   4*EI_L0, 2*EI_L0],
                             [0.0,   2*EI_L0, 4*EI_L0]])
        K = B.T @ k_local @ B

        # Geometric stiffness: nonzero only in the (u1,v1,u2,v2)
        # sub-block (global indices 0,1,3,4) -- B's dependence on the
        # global DOFs is entirely through c,s,L (functions of the
        # translations only), so d(B)/d(theta1)=d(B)/d(theta2)=0.
        gamma, sigma = c * c - s * s, 2.0 * c * s   # cos(2*phi), sin(2*phi)
        A_N = np.array([[ s*s, -c*s, -s*s,  c*s],
                         [-c*s,  c*c,  c*s, -c*c],
                         [-s*s,  c*s,  s*s, -c*s],
                         [ c*s, -c*c, -c*s,  c*c]])
        A_M = np.array([[-sigma,  gamma,  sigma, -gamma],
                         [ gamma,  sigma, -gamma, -sigma],
                         [ sigma, -gamma, -sigma,  gamma],
                         [-gamma, -sigma,  gamma,  sigma]])
        K_geo_block = (N / L) * A_N + (M_sum / L**2) * A_M
        idx = [0, 1, 3, 4]
        K[np.ix_(idx, idx)] += K_geo_block
        return K

    def stiffness(self, elem_coords, mat, thickness=1.0, **kwargs):
        """Initial (zero-displacement) tangent -- interface completeness,
        matching TrussTL2D.stiffness()/TrussPlastic2D.stiffness()."""
        return self.tangent_stiffness(elem_coords, np.zeros(6), mat, thickness)

    @classmethod
    def recover_stress(cls, elem_coords, u_elem, mat, y_fiber, xi=0.0):
        """Wave 12 item 115 (docs/consolidated_future_roadmap.md,
        `ICE-ROM/GAP_ANALYSIS.md` gap #4): fiber-level axial stress at
        a deformed state -- `internal_force()` returns nodal forces
        only; this is the missing "traditional finite element-based
        stress/strain recovery" piece Hollkamp & Gordon (2008) Figs.
        3-4's membrane/total stress PSDs need, and depends on Wave 12
        item 112's membrane-expansion module for anything upstream of
        this call to actually HAVE a membrane displacement to recover
        stress from (a bending-only ROM prediction has none).

        Standard combined axial + bending beam stress, `sigma = N/A -
        y*M/I`, evaluated from this element's own already-computed
        corotational natural (chord-relative) deformation measures --
        `e_bar` (axial stretch) and the two end-rotation-derived
        moments -- rather than deriving a new stress theory: `E*e_bar/
        L0` is exactly `N/A` (since `N = (EA/L0)*e_bar` is already how
        `internal_force()` computes the axial force). The bending
        moment varies LINEARLY along the element (exact for a 2-node
        beam under no distributed load along its own length, the
        standard assumption this element family already makes), from
        `M(0)=M1` to `M(1)=-M2` -- NOT `M1` to `+M2` -- because this
        element's `M1`/`M2` are the natural-mode ("Przemieniecki-style")
        end moments the `[4,2;2,4]*EI/L0` stiffness block itself is
        built from (see `internal_force()`), whose own sign convention
        makes a UNIFORM-bending state (constant curvature, e.g. a pure
        end moment) satisfy `M1 = -M2`, not `M1 = M2` -- confirmed
        directly (not assumed) by cross-checking a single-element
        cantilever under a pure tip moment against closed-form Euler-
        Bernoulli theory: `M(x) = M1 - (M1+M2)*x/L0` is the constant-
        shear-consistent linear profile satisfying both end conditions,
        and is what is actually implemented below.

        Parameters
        ----------
        elem_coords, u_elem, mat : same convention as `internal_force()`
            (`mat = (E, A, I)`).
        y_fiber : float or array_like
            Fiber distance from the neutral axis (positive on whichever
            side the caller defines as positive -- e.g. `+thickness/2`
            for the top surface of a rectangular section). Broadcasts
            against `xi` if both are arrays.
        xi : float, default 0.0
            Natural coordinate along the (undeformed) element at which
            to evaluate, in `[0, 1]` (`0` = node 1, `1` = node 2,
            `0.5` = midspan). Default `0.0` matches this item's own
            simplest single-station formula; pass an array to recover
            stress at several stations along one element in one call.

        Returns
        -------
        sigma : float or ndarray (broadcast shape of y_fiber, xi)
            Axial fiber stress, `sigma = E*e_bar/L0 - y_fiber*M(xi)/I`.
        """
        E, A, I = mat
        L0, L, c, s, e_bar, th1_bar, th2_bar = cls._kinematics(elem_coords, u_elem)
        EI_L0 = E * I / L0
        M1 = EI_L0 * (4 * th1_bar + 2 * th2_bar)
        M2 = EI_L0 * (2 * th1_bar + 4 * th2_bar)
        xi = np.asarray(xi, dtype=float)
        M_xi = M1 - (M1 + M2) * xi
        sigma_axial = E * e_bar / L0
        return sigma_axial - np.asarray(y_fiber, dtype=float) * M_xi / I


# =====================================================================
# 2-node small-displacement elasto-plastic truss (Module 9: material
# nonlinearity, "Elasto-Plasticity" case -- elastic up to a yield
# point, then permanent/irreversible deformation, e.g. a metal bracket
# bending permanently). Deliberately built on SMALL-displacement
# (engineering-strain) kinematics -- unlike TrussTL2D -- to isolate
# material nonlinearity from geometric nonlinearity, matching the
# user's own taxonomy that treats these as separate axes: the D matrix
# is what's updated every iteration here (via return mapping), not the
# strain measure itself. Combining this constitutive model with
# TrussTL2D's Green-Lagrange kinematics is a natural future extension,
# not a rewrite -- it would only mean writing the return-mapping logic
# below in terms of E_GL instead of engineering strain.
#
# 1-D J2/von Mises plasticity is a textbook special case: a bar can
# only carry uniaxial stress, so the general deviatoric yield surface
# collapses to |sigma| <= sigma_y + H*alpha (alpha = accumulated
# plastic strain, linear ISOTROPIC hardening with modulus H). Because
# the yield function is linear in sigma, the return-mapping equations
# solve in CLOSED FORM (no local Newton iteration, unlike 2-D/3-D J2
# plasticity) -- see _return_map() below.
# =====================================================================


# =====================================================================
# 2-node geometrically EXACT planar (Simo-Reissner) shear-deformable
# rod -- Wave 17 item 140 (docs/consolidated_future_roadmap.md), built
# to reproduce Georgiou (2005), "Advanced Proper Orthogonal
# Decomposition Tools..." (Nonlinear Dynamics 41:69-110), whose planar
# Cosserat-rod equations (its Eqs. 1-3) this element implements
# directly (Appendix A.1-A.7 gives the same strain measures in a
# slightly different notation). UNLIKE Beam2DCorotational above, this
# is a genuine TOTAL-LAGRANGIAN element (not corotational): the local
# frame is fixed to the REFERENCE (undeformed) chord direction once,
# at construction, and never re-updated as the element deforms -- the
# strain measures themselves (not a rotating frame) carry all of the
# large-rotation kinematics. It also carries an INDEPENDENT rotation
# field theta(s) (a genuine Timoshenko/Reissner shear-deformable rod),
# unlike Beam2DCorotational's Euler-Bernoulli (rotation tied to slope,
# no shear strain, no rotary inertia) -- see the roadmap's own gap
# analysis (Wave 17 section) for the frequency-matching evidence this
# distinction matters (Beam2DCorotational is 1.4%/11%/28% off a
# closed-form Timoshenko pinned-pinned rod on modes 1/3/5; this element
# is validated to <0.2%, see tests/test_beam2d_reissner.py CHECK 3).
#
# KINEMATICS. 2 nodes, 3 dofs/node (u1, u2, theta), ALL THREE linearly
# interpolated along the reference arc-length coordinate s in [0, L0]
# (unlike the cubic-Hermite bending field Beam2DEulerBernoulli/
# Beam2DCorotational use -- there is no C1 continuity requirement here
# because curvature kap = theta' does not itself involve second
# derivatives of a transverse-displacement field, theta is its own
# independent DOF). u1, u2 are LOCAL (material-frame) displacement
# components -- u1 along the reference axis, u2 transverse -- and theta
# is the cross-section's rotation angle RELATIVE TO THE REFERENCE AXIS,
# exactly the theta appearing in the paper's Eqs. 1-3. Because linear
# shape functions make u1', u2', theta' CONSTANT along the element,
# only theta(s) itself (linear in s) varies the strain integrands
# below through cos(theta(s))/sin(theta(s)).
#
# STRAIN MEASURES (paper Eqs. 1-3):
#     eps = (1+u1')*cos(theta) + u2'*sin(theta) - 1        (axial)
#     gam = u2'*cos(theta) - (1+u1')*sin(theta)             (shear)
#     kap = theta'                                          (curvature)
# STRESS RESULTANTS: N = EA*eps, Q = kappa_s*GA*gam, M = EI*kap, with
# kappa_s a shear-correction-factor parameter (mat's 5th entry) --
# DEFAULT/paper value 1.0 (Georgiou 2005 itself uses kappa_s=1; the
# roadmap's own frequency-matching derivation confirms this, see the
# Wave 17 section's opening paragraph -- deliberately NOT the ~5/6
# rectangular-section value a general Timoshenko-beam textbook might
# otherwise default to, because this element's job here is reproducing
# THIS paper).
#
# INTEGRATION. theta(s) is linear in s, so kap = theta' is CONSTANT --
# the bending virtual-work integral M*delta(kap)*L0 is exact under ANY
# quadrature (the integrand literally doesn't vary with s), so "exact
# integration on the bending term" reduces to a closed-form evaluation,
# no quadrature loop needed. eps(s)/gam(s), by contrast, are NOT
# polynomial in s (they involve cos(theta(s))/sin(theta(s)), theta(s)
# itself linear but wrapped in a trig function) -- evaluating them with
# TWO Gauss points (what a naive "integrate everything with the
# element's nominal order" choice would do) reproduces the classical
# shear-locking failure mode of a 2-node linear-interpolation
# Timoshenko beam: pure-bending states can't be represented without
# spurious shear strain creeping in away from the locking-free point.
# ONE-POINT (reduced) Gauss quadrature on the eps/gam terms -- i.e.
# evaluating the entire integrand at the element midpoint s=L0/2 and
# multiplying by L0 -- is the standard, textbook fix (uniform reduced
# integration removes shear locking for a 2-node Timoshenko element,
# the same logic Quad4MindlinPlate's SRI default and this base module's
# reduced_stiffness() apply elsewhere in this package). Because kap's
# own integral is ALREADY exact under this same one-point rule (its
# integrand is constant, so the midpoint value times L0 recovers the
# exact integral too), internal_force()/tangent_stiffness() below use
# ONE quadrature evaluation (the midpoint) for the WHOLE strain vector
# [eps, gam, kap] -- there is no separate "exact vs reduced" branch to
# implement, since exact-for-a-constant-integrand and reduced-to-one-
# point coincide here. th_mid = (theta1+theta2)/2 below IS this single
# Gauss point's theta value.
#
# ANALYTIC TANGENT DERIVATION. Let q = (u1_1,u2_1,th1, u1_2,u2_2,th2)
# be the LOCAL nodal dof vector (see _local_dofs() for the global<->
# local rotation) and define the four CONSTANT gradient vectors (L0 =
# reference length):
#     ga = d(u1')/dq = (-1/L0, 0, 0, 1/L0, 0, 0)
#     gb = d(u2')/dq = (0, -1/L0, 0, 0, 1/L0, 0)
#     gc = d(th_mid)/dq = (0, 0, 1/2, 0, 0, 1/2)
#     gd = d(kap)/dq = d(theta')/dq = (0, 0, -1/L0, 0, 0, 1/L0)
# Writing a=1+u1', b=u2', phi=th_mid (so eps=a*cos(phi)+b*sin(phi)-1,
# gam=b*cos(phi)-a*sin(phi)), the chain rule gives the GRADIENTS
# (first derivatives, i.e. the rows of the 3x6 strain-displacement
# matrix B):
#     grad(eps) = cos(phi)*ga + sin(phi)*gb + gam*gc
#     grad(gam) = cos(phi)*gb - sin(phi)*ga - (eps+1)*gc
#     grad(kap) = gd
# (using a*cos(phi)+b*sin(phi) = eps+1 and b*cos(phi)-a*sin(phi) = gam
# to fold the a/b dependence back into eps/gam themselves -- confirmed
# algebraically, not just plausible-looking). Differentiating AGAIN
# (ga, gb, gc, gd are constant vectors, so only phi(q), eps(q), gam(q)
# contribute a second derivative) and using the SAME substitution to
# eliminate a, b in favor of eps, gam gives, after collecting terms
# into manifestly SYMMETRIC outer-product combinations (verified by
# direct index-swap algebra, not merely asserted -- see the derivation
# notes this docstring is transcribed from):
#     H_eps = -sin(phi)*(gc(x)ga + ga(x)gc) + cos(phi)*(gc(x)gb + gb(x)gc)
#             - (eps+1)*(gc(x)gc)
#     H_gam = -sin(phi)*(gc(x)gb + gb(x)gc) - cos(phi)*(gc(x)ga + ga(x)gc)
#             - gam*(gc(x)gc)
# (x) = outer product; H_kap = 0 identically (kap is LINEAR in q, no
# second derivative). The local tangent is then
#     K_local = L0 * (B^T @ diag(EA, kappa_s*GA, EI) @ B
#                      + N*H_eps + Q*H_gam)
# -- B^T@D@B the material part (D diagonal because N/Q/M each depend
# on exactly one strain component, no cross terms), N*H_eps + Q*H_gam
# the GEOMETRIC part (M contributes nothing extra here because H_kap=0
# -- unlike Beam2DCorotational's geometric stiffness, which mixes N and
# M1+M2 together, this element's bending moment M enters the tangent
# ONLY through the (already-linear) B row for kap, never through a
# second derivative). Independently verified against a central finite
# difference of internal_force() at random LARGE-rotation states (not
# just near zero) -- tests/test_beam2d_reissner.py CHECK (a); this is
# the primary correctness gate for the algebra above, per this
# project's own house standard of not trusting a hand-derived tangent
# until it passes that check.
#
# GLOBAL <-> LOCAL TRANSFORM. Unlike Beam2DCorotational (whose local
# frame CO-ROTATES with the current chord), this element's local frame
# is FIXED at construction to the reference chord direction phi0 =
# atan2(dY0, dX0) (dX0/dY0 from elem_coords, the REFERENCE/undeformed
# nodal positions) -- a standard constant rotation, "the usual
# rotation into global axes" every 2-node element in this package with
# arbitrary planar orientation already uses (TrussTL2D needs none,
# being frame-agnostic by construction; Beam2DCorotational's own
# mass()/T_node do exactly this same block-rotation, just re-derived
# every call from the CURRENT chord instead of the fixed reference
# one). u1, u2 (translational dofs) rotate through the per-node 2x2
# block T_node = [[c0,-s0],[s0,c0]] (LOCAL -> GLOBAL, so GLOBAL ->
# LOCAL uses its transpose); theta passes through UNCHANGED -- this is
# not a simplification, it is the CORRECT physical convention this
# package already establishes (see Beam2DCorotational's own theta1/
# theta2, and this element's own objectivity proof below): each theta
# DOF is the cross-section's rotation INCREMENT from its own reference
# orientation (not an absolute angle measured from global x), the same
# way u1/u2 are displacement INCREMENTS from reference position, not
# absolute coordinates -- so theta needs no phi0 shift to already be
# "theta relative to the reference axis," exactly what the strain
# formulas above consume directly.
#
# OBJECTIVITY (rigid-body invariance) -- proven algebraically, not just
# tested: apply a rigid rotation by angle "a" to the whole element
# (both nodes, about node 1's reference position, WLOG with the local
# frame aligned to global x for the algebra -- the fixed local rotation
# doesn't affect this argument). Then u1' = cos(a)-1, u2' = sin(a), and
# BOTH nodal theta DOFs equal "a" (a rigid rotation increments every
# cross-section's orientation by the same angle as the rotation
# itself), so th_mid = a too. Substituting:
#     eps = cos(a)*cos(a) + sin(a)*sin(a) - 1 = cos(a-a) - 1 = 0
#     gam = sin(a)*cos(a) - cos(a)*sin(a) = sin(a-a) = 0
#     kap = (a-a)/L0 = 0
# -- EXACTLY zero for any rotation magnitude "a" (not just small a),
# confirming this strain measure is frame-invariant by construction
# (the entire point of writing eps/gam through cos(theta)/sin(theta)
# rather than a naive linearized engineering strain) -- see
# tests/test_beam2d_reissner.py CHECK (b) for the numerical
# confirmation at generic (not just this special) rigid motions.
#
# MASS. All three fields are linearly interpolated (unlike Beam2D
# Corotational's cubic-Hermite bending field), so -- unlike that
# element, which must override mass() with a closed-form Hermite
# matrix -- this element's mass matrix is exactly the GENERIC
# Element.mass() Gauss loop with dofs_per_node=3 and a diagonal density
# matrix diag(rhoA, rhoA, rhoI): translational kinetic energy 0.5*rhoA*
# (v1^2+v2^2) is a dot product, hence rotation-invariant, so it doesn't
# matter whether v1/v2 are evaluated in the local or global frame --
# the reference (undeformed) elem_coords and plain linear shape
# functions already give the physically correct answer with NO
# rotation transform needed (this is exactly why Beam2DCorotational's
# own mass() docstring needs its explicit T@M_local@T.T step and this
# element's mass() does not: that element's bending block mixes
# rotation and transverse translation through cubic Hermite functions,
# genuinely anisotropic between the local axial/transverse directions;
# this element's mass block does not, each field is independently
# linearly interpolated and rho is the SAME scalar in both
# translational directions). rhoI (rotary inertia) is the specific
# capability Beam2DCorotational lacks entirely (see that element's own
# docstring: "no separate rotary-inertia term") -- the whole reason
# this element needs its own mass(), not a reuse of an existing one.
# =====================================================================

class Beam2DReissner(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 2, 3, 1, 2
    translational_dof_mask = [True, True, False]   # [u1, u2, theta]

    def shape_and_derivs(self, natural_coords):
        """Linear interpolation along the reference axis for ALL THREE
        fields (u1, u2, theta) -- used directly by mass()'s generic
        Gauss loop (unlike Beam2DEulerBernoulli/Beam2DCorotational,
        which override mass() with a closed-form Hermite matrix and
        only use this for mesh.check_quality() interface completeness)."""
        (xi,) = natural_coords
        N = np.array([(1 - xi) / 2, (1 + xi) / 2])
        dN_dxi = np.array([-0.5, 0.5])
        return N, dN_dxi.reshape(1, 2)

    @staticmethod
    def _unpack_mat(mat):
        """mat = (E, G, A, I, kappa_s) -- kappa_s explicit (not
        defaulted here) per the roadmap's own convention; pass 1.0 to
        match Georgiou (2005)'s own value (see class docstring)."""
        E, G, A, I, kappa_s = mat
        return E, G, A, I, kappa_s

    @staticmethod
    def _ref_frame(elem_coords):
        X1, X2 = elem_coords[0], elem_coords[1]
        dX = X2 - X1
        L0 = np.linalg.norm(dX)
        c0, s0 = dX[0] / L0, dX[1] / L0
        return L0, c0, s0

    @staticmethod
    def _T(c0, s0):
        """6x6 block-diagonal LOCAL -> GLOBAL rotation: u1,u2 rotate
        through the per-node 2x2 chord-angle block, theta passes
        through unchanged (see class docstring for why)."""
        T_node = np.array([[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]])
        T = np.zeros((6, 6))
        T[0:3, 0:3] = T_node
        T[3:6, 3:6] = T_node
        return T

    @classmethod
    def _local_dofs(cls, elem_coords, u_elem):
        """Returns (L0, T, q): q = T.T @ u_elem is the LOCAL nodal dof
        vector (u1,u2 rotated into the fixed reference-chord frame,
        theta unchanged -- see class docstring)."""
        L0, c0, s0 = cls._ref_frame(elem_coords)
        T = cls._T(c0, s0)
        q = T.T @ np.asarray(u_elem, dtype=float)
        return L0, T, q

    @staticmethod
    def _strain_measures(L0, q):
        u1_1, u2_1, th1, u1_2, u2_2, th2 = q
        u1p = (u1_2 - u1_1) / L0
        u2p = (u2_2 - u2_1) / L0
        thp = (th2 - th1) / L0
        th_mid = 0.5 * (th1 + th2)
        cphi, sphi = np.cos(th_mid), np.sin(th_mid)
        eps = (1.0 + u1p) * cphi + u2p * sphi - 1.0
        gam = u2p * cphi - (1.0 + u1p) * sphi
        kap = thp
        return cphi, sphi, eps, gam, kap

    @staticmethod
    def _B_and_H(L0, q):
        """Returns (B, H_eps, H_gam, eps, gam, kap) -- see class
        docstring for the derivation. B: 3x6 strain-displacement
        matrix (rows grad(eps), grad(gam), grad(kap)). H_eps/H_gam:
        6x6 symmetric Hessians (d^2(eps)/dq^2, d^2(gam)/dq^2) --
        H_kap is identically zero (kap is linear in q) and simply
        omitted from tangent_stiffness()'s assembly below."""
        cphi, sphi, eps, gam, kap = Beam2DReissner._strain_measures(L0, q)
        ga = np.array([-1.0 / L0, 0.0, 0.0, 1.0 / L0, 0.0, 0.0])
        gb = np.array([0.0, -1.0 / L0, 0.0, 0.0, 1.0 / L0, 0.0])
        gc = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.5])
        gd = np.array([0.0, 0.0, -1.0 / L0, 0.0, 0.0, 1.0 / L0])

        grad_eps = cphi * ga + sphi * gb + gam * gc
        grad_gam = cphi * gb - sphi * ga - (eps + 1.0) * gc
        grad_kap = gd
        B = np.vstack([grad_eps, grad_gam, grad_kap])

        gc_ga = np.outer(gc, ga); ga_gc = gc_ga.T
        gc_gb = np.outer(gc, gb); gb_gc = gc_gb.T
        gc_gc = np.outer(gc, gc)

        H_eps = -sphi * (gc_ga + ga_gc) + cphi * (gc_gb + gb_gc) - (eps + 1.0) * gc_gc
        H_gam = -sphi * (gc_gb + gb_gc) - cphi * (gc_ga + ga_gc) - gam * gc_gc

        return B, H_eps, H_gam, eps, gam, kap

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """f_int = T @ (L0 * B^T @ [N,Q,M]) -- see class docstring for
        the strain measures, the reduced/exact-quadrature reasoning
        (both collapse to one midpoint evaluation here), and the
        local/global transform T."""
        E, G, A, I, kappa_s = self._unpack_mat(mat)
        L0, T, q = self._local_dofs(elem_coords, u_elem)
        B, _, _, eps, gam, kap = self._B_and_H(L0, q)
        N = E * A * eps
        Q = kappa_s * G * A * gam
        M = E * I * kap
        f_local = L0 * (B.T @ np.array([N, Q, M]))
        return T @ f_local

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """K_T = T @ K_local @ T.T, K_local = L0*(B^T@D@B + N*H_eps +
        Q*H_gam) -- see class docstring for the full derivation."""
        E, G, A, I, kappa_s = self._unpack_mat(mat)
        L0, T, q = self._local_dofs(elem_coords, u_elem)
        B, H_eps, H_gam, eps, gam, kap = self._B_and_H(L0, q)
        N = E * A * eps
        Q = kappa_s * G * A * gam
        D = np.diag([E * A, kappa_s * G * A, E * I])
        K_local = L0 * (B.T @ D @ B + N * H_eps + Q * H_gam)
        return T @ K_local @ T.T

    def stiffness(self, elem_coords, mat, thickness=1.0, **kwargs):
        """Zero-state (linearized-about-undeformed) tangent -- same
        interface-completeness convention as TrussTL2D.stiffness()/
        Beam2DCorotational.stiffness(). At u_elem=0, N=Q=M=0 so the
        geometric (H_eps/H_gam) terms vanish and this reduces to the
        ordinary linear shear-deformable (Timoshenko) beam stiffness,
        still evaluated with the SAME one-point-reduced B (so it is
        also automatically shear-locking-free)."""
        return self.tangent_stiffness(elem_coords, np.zeros(6), mat, thickness)

    def mass(self, elem_coords, rho, thickness=1.0):
        """rho = (rhoA, rhoI) -- see class docstring for why the
        generic Gauss-loop mass (Element.mass()) is exact here with no
        rotation transform needed, unlike Beam2DCorotational."""
        rhoA, rhoI = rho
        rho_matrix = np.diag([rhoA, rhoA, rhoI])
        return super().mass(elem_coords, rho_matrix, thickness)
