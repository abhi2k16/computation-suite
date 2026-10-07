"""
shells.py -- Module 20 (general-purpose extensions roadmap Phase 7):
Shell4MITC, a 4-node general shell element carrying MEMBRANE (in-plane
stretching) stiffness coupled to bending, unlike Quad4MindlinPlate
(plates.py) which is bending-only. Needed for curved or folded
thin-walled structures (a cylindrical tank, a car body panel) where
plate theory alone can't represent the geometry.

Formulation: MITC4 (Mixed Interpolation of Tensorial Components,
4-node), Dvorkin & Bathe's 1984 element ("A continuum mechanics based
four-node shell element for general non-linear analysis", Eng. Comput.
1(1):77-88) -- the most widely used, best-validated low-order shell
element in the literature. Built as a genuine generalization of
machinery this package already has, by DIRECT REUSE (composition, not
re-derivation):

  - Membrane part: Quad4PlaneStress's own B_matrix()/D_plane_stress(),
    evaluated in the element's local in-plane frame -- standard
    displacement-based membrane strains, no special treatment (classic
    MITC4 does not address MEMBRANE locking -- see the "Known
    limitations" note in this module's Shell4MITC docstring; that is a
    later refinement, MITC4+, out of this phase's scope).
  - Bending part: Quad4MindlinPlate's own _Bb_Bs() bending curvature
    matrix Bb and D_mindlin_plate()'s Db -- also standard displacement-
    based, no special treatment (classic MITC4 does not modify the
    bending strain field either).
  - Transverse shear part: THIS is what actually distinguishes MITC4
    from a naive membrane+plate superposition, and what genuinely FIXES
    shear locking (rather than merely relieving it via reduced
    integration the way Quad4MindlinPlate's 'sri' does): the assumed
    NATURAL-STRAIN interpolation at four edge-midpoint tying points,
    below.
  - Drilling DOF (in-plane rotation about the local normal, theta_z):
    classical shell-element complication -- degenerate-shell theory has
    no real physical stiffness for this DOF, but a 6-dof/node element
    needs SOME (even if small/artificial) stiffness there to avoid a
    singular tangent when adjacent elements are locally coplanar (a
    flat mesh, or any patch of elements sharing one normal direction).
    Fixed with a small diagonal penalty (Allman-type regularization),
    NOT real physics -- see _drilling_stiffness()'s docstring.
  - Local-to-global rotation: shells in a curved mesh don't share one
    global in-plane frame the way Quad4MindlinPlate's flat plate mesh
    does, so every element builds its OWN local (e1, e2, e3) frame from
    its own 4 corner nodes (see _local_frame_and_coords()), exactly
    mirroring elements.beams3d.Beam3DEulerBernoulli's per-element local-
    axes + block-rotation pattern -- just with a 3-vector "up" reference
    replaced by TWO independent in-plane geometry vectors a quad
    naturally provides (no ambiguous-axis problem a 1-D beam has).

Assumed transverse shear strain field (Dvorkin & Bathe 1984; formulas
as restated in Ko, Lee & Bathe, "A new MITC4+ shell element", Computers
& Structures 182:404-418, 2017, Eq. 6):

    e~_rt(r,s) = (1/2)(1+s) e_rt^(A) + (1/2)(1-s) e_rt^(B)
    e~_st(r,s) = (1/2)(1+r) e_st^(C) + (1/2)(1-r) e_st^(D)

with tying points A=(r=0,s=+1), B=(r=0,s=-1) (midpoints of the s=+-1
edges) and C=(r=+1,s=0), D=(r=-1,s=0) (midpoints of the r=+-1 edges).
e_rt^(A) etc. denotes the COVARIANT (natural-coordinate) transverse
shear strain, computed with the ordinary displacement-based formula but
evaluated AT that specific tying point rather than at a Gauss point.
The key effect: e~_rt is interpolated LINEARLY in s but is CONSTANT in
r (it only ever samples r=0), and symmetrically for e~_st -- this
removes the spurious r-dependent (resp. s-dependent) parasitic shear
term that a naive displacement-based Bs picks up under pure bending,
without discarding so much of the shear field that the element becomes
rank-deficient the way uniform reduced integration can.

This implementation evaluates the covariant strain at each tying point
by taking Quad4MindlinPlate's own CARTESIAN shear B-matrix (the exact
same Bs used by 'sri' integration) at that point and transforming it to
covariant (natural) components via the local Jacobian:
    [e_rt; e_st] = J @ [gamma_xz; gamma_yz],  J[i,j] = dx_j/dxi_i
(the same J = jacobian(dN_natural, elem_coords) already used everywhere
in this package; this transform is the standard covariant push-forward
e_i = g_i . u,i = sum_j (dx_j/dxi_i) gamma_j, not a new convention).
After interpolating [e~_rt; e~_st] via the tying-point formula above,
the assumed strain is transformed BACK to Cartesian at the actual
integration point via J^-1, exactly mirroring how B_matrix() elsewhere
in this package turns natural derivatives into Cartesian ones.

Known limitations (by design, matching classic MITC4, not this
package's oversight):
  - Flat-facet approximation: each element uses ONE local frame (built
    from its own 4 corners' geometry, see _local_frame_and_coords()),
    projecting nodal coordinates onto that single flat plane. A warped
    (non-planar) quadrilateral is approximated by its best-fit flat
    facet, not exactly represented -- fine for a shell mesh fine enough
    that individual elements are nearly flat (the standard assumption
    for a flat-facet shell), not a genuinely curved-surface (degenerate-
    shell, per-node director vector) formulation. See the roadmap doc's
    Section 1 for why the full per-node-director formulation was scoped
    out of this phase.
  - Membrane locking (distinct from transverse SHEAR locking) is NOT
    addressed -- classic MITC4, as implemented here, only fixes shear
    locking. Membrane locking shows up specifically in curved geometries
    under coarse/distorted meshes; MITC4+ (Ko, Lee & Bathe 2017, the
    same paper this module's formula citation is drawn from) is the
    documented fix, out of scope for this phase.

  Wave 4 investigation note (2026-09-08, item 22): a MITC4+ membrane
  fix was investigated and a working implementation was built and
  numerically verified against the paper's own flat-geometry
  equivalence guarantee -- but on closer analysis it was found to be
  PROVABLY INERT for warped (non-flat) elements too, not just flat
  ones, as long as displacement is kept purely in-plane (u, v only):
  the paper's membrane strain is built from dot products of reference
  geometry vectors with displacement vectors, and every quantity that
  could carry a warped element's out-of-plane geometric content (x_d's
  n-component, reached only via dot products against in-plane-only
  m_r/m_s or displacement) gets annihilated when displacement has no
  out-of-plane (w) component -- confirmed numerically: assembled
  stiffness was unchanged to floating-point precision even at a warp
  1.0 units on a ~1.5-unit element. A REAL fix needs w to enter the
  membrane strain calculation, which reintroduces membrane-bending DOF
  coupling -- the same class of risk that produced 3 of
  Shell4MITCCorotational's own documented dead ends (see that class's
  "Design history" below) -- so this was deferred rather than shipped
  as an always-inert "fix". Not attempted further this wave.
"""
import numpy as np

from .base import Element, gauss_product, jacobian
from .solids import Quad4PlaneStress
from .plates import Quad4MindlinPlate


# Tying points (Dvorkin & Bathe 1984 convention): A, B sample the
# covariant e_rt component (used for edges of constant s); C, D sample
# e_st (used for edges of constant r). See the module docstring for the
# interpolation formula that combines them.
_TYING_A = (0.0, 1.0)
_TYING_B = (0.0, -1.0)
_TYING_C = (1.0, 0.0)
_TYING_D = (-1.0, 0.0)

# Local per-node DOF layout: (u, v, w, theta_x, theta_y, theta_z), where
# theta_x/theta_y/theta_z are TRUE rotation-VECTOR components about the
# local e1/e2/e3 axes -- this is what _rotation_matrix()'s block-diagonal
# R treats them as, since it rotates translation and rotation triplets
# by the exact SAME 3x3 R (the standard small-rotation-as-a-vector
# convention, e.g. Beam3DEulerBernoulli's (thx,thy,thz) triplet).
#
# Quad4MindlinPlate's own (w, betax, betay) convention is DIFFERENT:
# its _Bb_Bs() defines gamma_xz = dw/dx - betax and gamma_yz = dw/dy -
# betay (see plates.py) -- the standard Reissner-Mindlin SLOPE
# convention, where "betax" means "the rotation that produces bending
# curvature in x" (i.e. curvature kappa_xx = d(betax)/dx). By the
# right-hand rule that rotation is physically a rotation ABOUT THE
# Y-AXIS (tilting the normal within the x-z plane), not about the
# x-axis -- so Quad4MindlinPlate's betax is Shell4MITC's theta_y, and
# betay is theta_x. Index groups below scatter Quad4MindlinPlate's
# (w, betax, betay) columns (in that exact order, coming straight out
# of _Bb_Bs()/_mitc4_shear_B()) into (w, theta_y, theta_x) local slots
# accordingly -- get this swap wrong and the FLAT case still looks
# correct (T is the identity there, so any x/y-label swap is invisible),
# but any CURVED mesh assembles the wrong rotation transform at shared
# nodes between differently-oriented elements, which is exactly the
# failure mode a flat-only test would never catch.
#
# BUG FOUND AND FIXED (NonLin-HyROM Case-1 session, 2026-09-03): the
# swap above got the INDEX right but not the SIGN. Quad4MindlinPlate's
# _Bb_Bs() defines BOTH shear rows with the same-sign convention
# (gamma_xz = dw/dx - betax, gamma_yz = dw/dy - betay -- see plates.py),
# but a genuine right-handed rotation VECTOR (theta_x, theta_y, theta_z)
# -- which is what _rotation_matrix() treats this triplet as, rotating
# it with the exact same 3x3 R used for translations -- needs the
# standard Reissner-Mindlin kinematics u = u0 - z*theta_y, v = v0 +
# z*theta_x, i.e. OPPOSITE relative signs between the two shear rows.
# betax matches theta_y with no sign correction; betay matches -theta_x,
# not +theta_x, so it needs one. Missing that sign is invisible for any
# axis-aligned element (T reduces to the identity there, so a wrong-sign
# but self-consistent local scatter never gets exercised) but produces
# severely wrong (found: 40x-1500x too stiff) assembled stiffness the
# moment neighboring elements have different local frame orientations --
# i.e. every curved/hole/unstructured mesh. Diagnosed via a minimal
# reproduction: cyclically relabeling one element's local node order
# (0,1,2,3 -> 1,2,3,0 -- same physical element, same connectivity, only
# rotates which corner is "local node 0" by 90 degrees) must leave a
# correctly-implemented element's assembled stiffness invariant up to
# the matching DOF permutation; before this fix it was not (relative
# error ~1500x on a simple cantilever tip deflection), isolated
# block-by-block to the _BEND slots specifically (membrane and drilling
# both already passed this test), and resolved by this sign array.
# _BEND_SIGN's -1 entries hit exactly the betay-into-theta_x slot (the
# 3rd of every (w, betax, betay) triple in Kbs's own layout) that needed
# the flip; verified to restore relabeling-invariance to ~1e-9 relative
# (floating-point level) and to fix the >100x static/modal errors this
# bug caused on fea_engine's own rectangle_with_hole_mesh_quarter() and
# on Gmsh-generated meshes alike.
_MEM = [0, 1, 6, 7, 12, 13, 18, 19]                  # u0,v0,u1,v1,u2,v2,u3,v3
_BEND = [2, 4, 3, 8, 10, 9, 14, 16, 15, 20, 22, 21]  # (w,thy,thx) x4 nodes
_DRILL = [5, 11, 17, 23]                              # theta_z x4 nodes
_BEND_SIGN = np.array([1.0, 1.0, -1.0] * 4)  # negate betay (-> theta_x) in Kbs's own (w,betax,betay)x4 layout
_BEND_SIGN_OUTER = np.outer(_BEND_SIGN, _BEND_SIGN)  # S @ Kbs @ S via elementwise product


class Shell4MITC(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 4, 6, 2, 2
    translational_dof_mask = [True, True, True, False, False, False]

    def __init__(self, drilling_factor=1e-3):
        """drilling_factor: dimensionless scale for the artificial
        drilling (theta_z) stiffness, as a fraction of the element's
        transverse shear stiffness scale (Ds[0,0]) times its area -- see
        _drilling_stiffness()'s docstring. NOT real physics; the default
        1e-3 is a small, standard regularization value (the drilling DOF
        should carry negligible strain energy compared to the real
        membrane/bending/shear response for any reasonable mesh)."""
        self.drilling_factor = drilling_factor
        self._membrane = Quad4PlaneStress()
        self._plate = Quad4MindlinPlate()

    def shape_and_derivs(self, natural_coords):
        """Same bilinear Quad4 shape functions as Quad4PlaneStress /
        Quad4MindlinPlate -- delegated, not re-derived, since all three
        elements share the identical isoparametric map."""
        return self._membrane.shape_and_derivs(natural_coords)

    def _local_frame_and_coords(self, elem_coords):
        """Builds ONE flat local (e1, e2, e3) frame per element from its
        own 4 corner nodes, following the "characteristic geometry
        vectors" construction of Ko, Lee & Bathe (2017) Eq. 9-10: x_r,
        x_s are the bilinear derivatives of the nodal position field at
        the element center (r=s=0), and their cross product gives the
        ONE flat plane that equally accounts for every nodal point's
        geometry -- well-defined even for a warped (non-planar)
        quadrilateral, where it serves as the flat-facet APPROXIMATION
        (see the module docstring's "Known limitations").

        Returns (e1, e2, e3, local_coords): local_coords is (4, 2), each
        node's (x, y) position in the local frame relative to the
        element centroid -- the SAME shape Quad4PlaneStress/
        Quad4MindlinPlate's elem_coords already expect, so their
        B_matrix()/_Bb_Bs() can be called on it directly without any
        further translation."""
        X = np.asarray(elem_coords, dtype=float)
        x_r = 0.25 * (-X[0] + X[1] + X[2] - X[3])
        x_s = 0.25 * (-X[0] - X[1] + X[2] + X[3])
        e3 = np.cross(x_r, x_s)
        e3 = e3 / np.linalg.norm(e3)
        e1 = x_r / np.linalg.norm(x_r)
        e2 = np.cross(e3, e1)

        centroid = X.mean(axis=0)
        local = np.zeros((4, 2))
        for a in range(4):
            d = X[a] - centroid
            local[a, 0] = d @ e1
            local[a, 1] = d @ e2
        return e1, e2, e3, local

    def _rotation_matrix(self, e1, e2, e3):
        """24x24 block-diagonal rotation: 8 independent 3-vectors per
        element (translation + rotation triplet at each of 4 nodes),
        each rotated by the SAME 3x3 R = [e1; e2; e3] -- identical
        pattern to Beam3DEulerBernoulli's T (there: 4 triplets for 2
        nodes x translation/rotation; here: 8 triplets for 4 nodes x
        translation/rotation)."""
        R = np.vstack([e1, e2, e3])
        T = np.zeros((24, 24))
        for i in range(8):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return T

    def _mitc4_shear_B(self, natural_coords, local_coords):
        """Assumed transverse shear B-matrix at natural_coords=(r,s),
        via the Dvorkin-Bathe tying scheme described in the module
        docstring. Returns (Bs_assumed, detJ): Bs_assumed is (2, 12) in
        the SAME (w, theta_x, theta_y) x4-node local ordering as
        Quad4MindlinPlate's own Bs, so it plugs directly into the same
        Db/Ds-based bending+shear energy integral."""
        r, s = natural_coords

        def covariant_rows(tying_pt):
            _, Bs_t, _ = self._plate._Bb_Bs(tying_pt, local_coords)
            _, dN_nat_t = self._plate.shape_and_derivs(tying_pt)
            J_t, _ = jacobian(dN_nat_t, local_coords)
            row_r = J_t[0, 0] * Bs_t[0, :] + J_t[0, 1] * Bs_t[1, :]
            row_s = J_t[1, 0] * Bs_t[0, :] + J_t[1, 1] * Bs_t[1, :]
            return row_r, row_s

        row_r_A, _ = covariant_rows(_TYING_A)
        row_r_B, _ = covariant_rows(_TYING_B)
        _, row_s_C = covariant_rows(_TYING_C)
        _, row_s_D = covariant_rows(_TYING_D)

        gamma_r_tilde = 0.5 * (1 + s) * row_r_A + 0.5 * (1 - s) * row_r_B
        gamma_s_tilde = 0.5 * (1 + r) * row_s_C + 0.5 * (1 - r) * row_s_D

        _, dN_nat = self._plate.shape_and_derivs((r, s))
        J, detJ = jacobian(dN_nat, local_coords)
        Jinv = np.linalg.inv(J)
        Bs_assumed = Jinv @ np.vstack([gamma_r_tilde, gamma_s_tilde])
        return Bs_assumed, detJ

    def _drilling_stiffness(self, Ds, area):
        """Small artificial diagonal stiffness for the drilling
        (theta_z) DOF -- NOT real shell physics (degenerate-shell theory
        has no genuine stiffness conjugate to in-plane normal rotation),
        added purely to prevent a singular tangent stiffness when
        adjacent elements are coplanar (every real DOF's stiffness
        contribution from such elements is then blind to a uniform
        theta_z rotation of the shared node).

        Scaled from Ds[0,0] = k_shear * G * h (the transverse shear
        stiffness-per-unit-width already computed for THIS element,
        available with no extra material lookups) times the element's
        own flat-facet area, giving units of N.m (torque per unit
        rotation) -- dimensionally a rotational stiffness, and scaled to
        the SAME element's real stiffness magnitude so drilling_factor
        is a meaningful, mesh-independent dimensionless knob rather than
        a magic absolute number. drilling_factor defaults to 1e-3, a
        standard small-regularization value (see __init__'s docstring);
        large enough to lift the singularity, small enough that the
        artificial energy it stores is negligible next to the real
        membrane/bending/shear response for any reasonable load."""
        return self.drilling_factor * Ds[0, 0] * area

    def stiffness(self, elem_coords, D, thickness=1.0, **kwargs):
        """D = (Dm, Db, Ds, h), as returned by material.D_shell(). The
        'thickness' kwarg is accepted for interface consistency with
        every other element's stiffness() signature (see
        Element.stiffness()'s docstring) but unused -- h is already
        carried inside D, mirroring Quad4MindlinPlate's own convention
        of not taking a separate thickness multiplier once D already
        encodes it."""
        Dm, Db, Ds, h = D
        e1, e2, e3, local = self._local_frame_and_coords(elem_coords)

        K_local = np.zeros((24, 24))
        pts, wts = gauss_product(self.gauss_order, self.dim)

        # Membrane: standard displacement-based Quad4PlaneStress B,
        # scaled by thickness h explicitly (Dm is per-unit-thickness).
        Km = np.zeros((8, 8))
        area = 0.0
        for p, w in zip(pts, wts):
            Bm, detJ = self._membrane.B_matrix(p, local)
            Km += (Bm.T @ Dm @ Bm) * detJ * w * h
            area += detJ * w
        K_local[np.ix_(_MEM, _MEM)] += Km

        # Bending (standard displacement-based Bb) + transverse shear
        # (MITC4 assumed-strain Bs) -- both integrated at the SAME full
        # 2x2 Gauss order; unlike Quad4MindlinPlate's 'sri', there is no
        # reduced-integration step here, since the assumed shear field
        # -- not a lower quadrature order -- is what avoids locking.
        Kbs = np.zeros((12, 12))
        for p, w in zip(pts, wts):
            Bb, _, detJ = self._plate._Bb_Bs(p, local)
            Kbs += (Bb.T @ Db @ Bb) * detJ * w
            Bs_assumed, detJ_s = self._mitc4_shear_B(p, local)
            Kbs += (Bs_assumed.T @ Ds @ Bs_assumed) * detJ_s * w
        K_local[np.ix_(_BEND, _BEND)] += Kbs * _BEND_SIGN_OUTER

        # Drilling: diagonal-only artificial regularization.
        k_drill = self._drilling_stiffness(Ds, area) / 4.0
        for idx in _DRILL:
            K_local[idx, idx] += k_drill

        T = self._rotation_matrix(e1, e2, e3)
        return T.T @ K_local @ T

    def full_stiffness(self, elem_coords, D, thickness=1.0):
        """N/A -- MITC4's anti-locking mechanism is the assumed shear
        strain FIELD, not a choice of quadrature order (unlike
        Quad4MindlinPlate's 'sri'); both full_stiffness() and
        reduced_stiffness() return the same (standard 2x2 Gauss)
        result. There is no reduced-integration variant of this
        element."""
        return self.stiffness(elem_coords, D, thickness)

    def reduced_stiffness(self, elem_coords, D, thickness=1.0):
        return self.stiffness(elem_coords, D, thickness)

    def mass(self, elem_coords, rho_matrix, thickness=1.0):
        """rho_matrix = material.shell_rho_matrix(mat, h) (6x6 diag, see
        its docstring). Builds the LOCAL consistent mass matrix via the
        inherited GENERIC Gauss loop (Element.mass() -- valid here
        because Shell4MITC defines shape_and_derivs() and has
        dofs_per_node=6, n_nodes=4, matching what that generic loop
        assumes), on the same projected local_coords stiffness() uses,
        then rotates to global with the same T as stiffness()."""
        e1, e2, e3, local = self._local_frame_and_coords(elem_coords)
        M_local = super().mass(local, rho_matrix, thickness)
        T = self._rotation_matrix(e1, e2, e3)
        return T.T @ M_local @ T

    def geometric_stiffness(self, elem_coords, N, thickness=1.0, **kwargs):
        """Not implemented -- shell buckling needs a genuinely different
        (2-D, membrane-stress-dependent) geometric stiffness formulation
        than the reference-axial-force convention Element.
        geometric_stiffness()'s scalar N expects; out of this phase's
        scope (see docs/general_purpose_extensions_roadmap.md Section 1
        -- shell elements were scoped for static/dynamic linear analysis
        only). Inherits Element's NotImplementedError, which already
        describes this correctly ("this element only has linear
        kinematics" -- true of Shell4MITC's IN-PLANE strain measures;
        the transverse-shear ASSUMED strain doesn't change that)."""
        return super().geometric_stiffness(elem_coords, N, thickness, **kwargs)


# =====================================================================
# Shell4MITCCorotational -- Phase A of
# docs/shells.md Section 4.2: geometric nonlinearity
# (large displacement / large rotation, small local strain) for
# Shell4MITC, via a co-rotational wrapper -- the same category of
# extension Beam2DCorotational already adds on top of
# Beam2DEulerBernoulli's linear machinery (see beams.py), generalized
# from a 2-D scalar chord rotation to a 3-D per-element flat-facet
# frame. See docs/shells.md Section 4.2 for the full derivation this
# implementation follows; this docstring also records the real,
# numerically-traced dead ends this class's own validation ran into
# before landing on the construction actually used below -- see "Design
# history" -- since the roadmap's own Section 2.2 explicitly flagged
# this exact difficulty as likely, and the dead ends are the concrete
# version of that warning, not a hypothetical one.
#
# Key structural fact this construction exploits (verified directly,
# not assumed): K_local0 -- Shell4MITC's local membrane+bending+shear+
# drilling stiffness -- is EXACTLY block-diagonal between the membrane+
# drilling DOF group {u, v, theta_z} and the bending+shear DOF group
# {w, theta_x, theta_y} for a flat facet (checked numerically: the
# cross-block entries are 0.0, not merely small). Classic MITC4 simply
# never builds a membrane-bending coupling term -- membrane uses
# Quad4PlaneStress's B, bending+shear uses Quad4MindlinPlate/MITC4's
# Bb/Bs, and they are summed into disjoint DOF slots (see _MEM/_BEND/
# _DRILL above and stiffness()'s own K_local[np.ix_(_MEM,_MEM)] /
# K_local[np.ix_(_BEND,_BEND)] assembly). This licenses two DIFFERENT,
# group-appropriate co-rotational corrections instead of one uniform
# 3-D frame rotation applied indiscriminately to all 6 DOFs/node --
# which is exactly the trap the rejected attempts below fell into (a
# uniform frame rotation applied to a RAW out-of-plane translation like
# a genuine bending deflection injects that translation's ABSOLUTE
# magnitude, not its within-element VARIATION, into the membrane block
# through the frame's own tilt; a leak of relative size ~1e-4 there
# gets amplified by K_local0's ~1e6-1e7 membrane/bending stiffness
# RATIO into a spurious force thousands of times larger than the real
# bending force the state should have produced -- see dead end 2/3).
#
# Kinematics (roadmap Section 2.1/2.2): at every call, elem_coords is
# the REFERENCE (undeformed) nodal geometry and u_elem is the TOTAL
# global displacement/rotation history to date -- the same convention
# every other nonlinear element in this package uses (TrussTL2D,
# Tet10SolidTL, Beam2DCorotational).
#   1. Reference frame (e1_0,e2_0,e3_0) and local material stiffness
#      K_local0: Shell4MITC's own machinery, unchanged (see
#      _local_material_stiffness()).
#   2. Membrane+drilling block ({u,v,theta_z} per node): a DRILLING-
#      ONLY current frame R_drill = R0 @ (I - skew([0,0,theta_z_mean]))
#      -- the small-rotation update of R0 by ONLY the mean of the 4
#      nodes' theta_z DOFs, deliberately ignoring theta_x/theta_y (the
#      out-of-plane tilt content) entirely. Local in-plane translation
#      is POSITION-based (exact to all orders in the drilling angle,
#      not just linearized): rotate the CURRENT nodal position
#      (relative to the current centroid) into R_drill and subtract the
#      REFERENCE local position (relative to the reference centroid);
#      only the (x,y) components are used. Local drilling rotation is
#      theta_z(node) - theta_z_mean. Because R_drill only ever rotates
#      about the LOCAL z-axis, it structurally CANNOT mix a node's z
#      (out-of-plane, potentially large-magnitude bending) coordinate
#      into the x/y outputs -- this is what makes a POSITION-based (not
#      just theta-based) extraction safe here, without re-creating dead
#      end 1's translation-sensitive-frame problem.
#   3. Bending+shear block ({w,theta_x,theta_y} per node): used RAW,
#      with NO frame correction at all -- w(node), theta_x(node),
#      theta_y(node) go directly into dof_local unmodified. Deliberately
#      the simplest possible choice, licensed by two things verified
#      together: (a) K_local0's bending block is a near-exact null
#      space for the correct small-rotation rigid-tilt kinematic
#      pattern (uniform theta, position-proportional w), so a genuine
#      small rigid tilt costs almost no spurious energy with no
#      correction at all; and (b) "correcting" this block by rotating
#      it through a tilt-tracking frame reintroduces exactly the
#      position-times-tilt leak described above -- worse than the
#      uncorrected error it would fix (see dead end 3). The cost is the
#      same kind of bounded, honestly-scoped drift as the drilling
#      block: large single-element rigid TILTS are not exactly
#      invariant -- see "Known remaining limitation" below.
#   4. Local internal force: K_local0 @ dof_local(u) -- K_local0 itself
#      is evaluated ONCE per call at the REFERENCE geometry (like the
#      beam's EA/L0, EI/L0 in Beam2DCorotational), never updated with
#      deformation.
#   5. Global internal force: rotate the local force back to global via
#      T(u).T, where T(u) is the SAME block-diagonal-of-R_drill
#      transform used to build the membrane block in step 2 (applied
#      uniformly to all 24 local dofs for simplicity -- since R_drill's
#      third row/column is exactly [0,0,1], this leaves the w
#      component of the bending block completely unrotated, and only
#      mixes theta_x/theta_y with each other by the drilling angle,
#      which is the physically-correct in-plane-frame transform for
#      those vector components regardless of block).
#
# Why this reduces EXACTLY to Shell4MITC.stiffness() at u=0 (verified
# numerically -- tests/test_shell_corotational.py): at u=0, theta_z_
# mean=0 so R_drill=R0 exactly, and dof_local(u) is linear in u to
# leading order with Jacobian exactly T0 in every DOF slot (both the
# position-based membrane extraction and the raw bending extraction
# reduce to the identity-like linear map T0 at u=0) -- so the u=0
# tangent is T0.T @ K_local0 @ T0, exactly Shell4MITC.stiffness()'s own
# construction. Measured mismatch: ~5e-7 relative (floating-point-
# level, not a modeling approximation).
#
# Design history (three dead ends plus one abandoned variant, kept here
# because the roadmap's own Section 2.2 anticipated this class of
# difficulty and this is the concrete account of it, not a
# hypothetical):
#   1. First attempt tracked the CURRENT frame from corner POSITIONS
#      (Shell4MITC._local_frame_and_coords()'s own bilinear
#      x_r-cross-x_s construction, evaluated at the deformed corners)
#      and extracted dof_local via "current local position minus
#      reference local position" per node, uniformly for all 6 DOFs.
#      internal_force(0)=0 held, but the u=0 TANGENT was 32% off
#      Shell4MITC.stiffness() -- traced to the position-tracked frame's
#      orientation being sensitive to EVERY node's translation (moving
#      one corner out of plane tilts the best-fit normal for all four
#      nodes), feeding into the local rotation measure as a spurious
#      transverse-SHEAR-producing term.
#   2. Second attempt kept a rotation(theta)-tracked frame (all 3
#      components of mean_theta, not just drilling) and extracted
#      dof_local uniformly via "raw dof rotated by that frame, then
#      projected off the frame's own 6-dimensional near-null
#      eigenspace" (an eigenvector-based rigid-mode projector P, built
#      once per call from K_local0's own eigendecomposition -- eigh()
#      on K_local0 rather than a hand-derived formula, since an earlier
#      variant using a hand-derived "textbook" rigid-mode formula
#      scored even worse, 12.96% at u=0: K_local0's own near-null
#      eigenspace does not exactly coincide with the textbook formula's
#      rigid-mode SHAPES on a general element geometry). This
#      reproduced Shell4MITC.stiffness() at u=0 to ~1e-13 relative --
#      excellent -- and passed complex-step-vs-FD self-consistency on
#      GENERIC small random states. But it failed catastrophically on
#      the realistic, highly-CORRELATED state a Newton solve actually
#      produces (a bending-dominated cantilever step: translations at
#      machine-epsilon scale, w and theta_y both a few percent of the
#      element size): the frame rotation, built from mean_theta and
#      applied uniformly to the RAW w translation, leaked w's ABSOLUTE
#      magnitude (not its within-element VARIATION) into the membrane
#      block by an amount proportional to mean_theta*w -- small in an
#      absolute sense (~1e-4) but, hitting K_local0's ~1e6-1e7 stiffer
#      membrane block, produced a spurious force of order 1e4 for a
#      state where the correct (linear-stiffness) answer was ~0. On a
#      real multi-element mesh this made solve_nonlinear_static()
#      diverge catastrophically from the very first Newton correction
#      (residual growing from ~6 to ~1e19 over 12 iterations) -- a
#      genuinely small, physically-unremarkable first Newton step, not
#      a contrived adversarial input.
#   3. An intermediate variant of attempt 2 additionally tried
#      subtracting a "rigid-tilt-induced" position term from the
#      membrane translations (a step toward the fix that ultimately
#      worked, but applied using the FULL 3-D tilt frame rather than a
#      drilling-only one) -- this made the spurious membrane force
#      WORSE, not better (a full-tilt frame's z-row is not [0,0,1], so
#      it re-couples w into x/y from a different direction). This is
#      what motivated isolating the exactly-decoupled membrane/bending
#      block structure (see "Key structural fact" above) and using a
#      DIFFERENT, group-appropriate frame for each block, instead of
#      continuing to look for one uniform 3-D correction that avoids
#      the leak in both directions at once -- attempts 1-3 all apply a
#      single frame to all 6 DOFs/node, and all three leak precisely
#      because of that.
#
# Known remaining limitation (Phase A's honestly-scoped gap; UPDATED
# 2026-09-08, Wave 4 item 18): originally, neither block's frame
# correction was exact for a LARGE single-element rigid motion -- the
# drilling frame was a first-order (not exact Rodrigues) update, and
# the bending block has no frame correction at all. The DRILLING half
# of this is now CLOSED (item 18): _local_relative_dofs()'s R_drill
# uses _exact_drill_rotation(theta_z_mean), the EXACT single-axis
# rotation, not a first-order truncation -- safe to make exact
# specifically BECAUSE drilling is always a rotation about the single
# FIXED axis R0's own local z (never a general 3-vector rotation), so
# only cos/sin of a scalar angle are needed, both holomorphic
# everywhere with no norm/branching singularity at zero rotation (the
# exact issue that ruled out a general exact-SO(3) treatment for a
# general 3-vector tilt frame -- see "Design history" dead ends 1-3,
# and items 20/22's own investigation, both of which needed a general
# 3-vector or out-of-plane-coupled treatment and hit real structural
# obstacles this narrower, single-axis case does not). Measured
# single-element rigid-motion error (internal-force norm relative to
# the element's own stiffness scale) is now: drilling (membrane block)
# EXACTLY zero (~1e-17, floating-point noise) at every angle tested, 0.5
# through 90 degrees -- not just improved, an exact identity, verified
# in tests/test_shell_corotational_exact_drill.py -- versus the
# ORIGINAL first-order update's ~4e-6 at 0.5 degrees, ~6e-4 at 6
# degrees, ~1.6% at 30 degrees. The BENDING block's error is UNCHANGED
# by this fix (still uncorrected, deliberately -- item 18 only touches
# the drilling/membrane frame): out-of-plane tilt ~0.03% at 0.5 degrees,
# ~0.33% at 6 degrees, ~1.8% at 30 degrees, still notably better than
# dead end 2's already-accepted 12-25%-at-30-degrees figure. Closing the
# BENDING block's residual large-rotation drift needs a position-tracked
# extraction for that block too (item 20) -- investigated 2026-09-08 and
# found to require a genuinely novel construction (a literal reading
# reusing R_drill's own frame produces zero new nonlinearity, since
# R_drill's z-row is fixed at [0,0,1] regardless of drilling angle);
# deferred, not attempted, given the dead-end risk documented for
# adjacent frame-coupling attempts (see docs/consolidated_future_
# roadmap.md Wave 4 item 20 for the finding).
#
# Phase C finding (bigger consequence of the same limitation, EXACT not
# approximate -- tests/test_shell_corotational_elastica.py): R_drill
# only ever rotates about the local z-axis, so its own z-row/column is
# [0,0,1] IDENTICALLY -- meaning the membrane (u,v) block's position-
# based extraction (step 2) was, by construction, EXACTLY, STRUCTURALLY
# blind to w/theta_x/theta_y (the bending block) for ANY element, ANY
# mesh density, ANY load: no amount of bending produced membrane strain
# in that construction alone (not "small," exactly zero -- confirmed
# down to ~1e-18 in tests/test_shell_corotational_elastica.py's
# assembled-mesh checks, and unchanged by mesh refinement, which is
# itself the signature of a MODELING limit rather than a discretization
# one). The practical consequence: a bent cantilever strip predicted
# ZERO axial (in-plane) foreshortening at ANY tip rotation, when the
# true (elastica) large-rotation solution foreshortens increasingly
# with rotation -- so transverse tip deflection itself, not just
# foreshortening, drifts from the exact elastica solution once rotation
# stops being small. Measured (a 20x2-element, L=1m/b=0.05m/h=5mm
# narrow strip, matching the classic Bisshopp-Drucker cantilever setup):
# even at near-zero load this strip's tip deflection differs from
# Euler-Bernoulli's P*L^3/(3EI) by a mesh-converged ~0.7% baseline (the
# strip's finite width and MITC4's own transverse-shear flexibility,
# NOT a rotation effect -- confirmed unchanged by refining across the
# strip's width, so this is a fixed offset, not something the elastica
# comparison below needs to correct for at each load level separately).
# On TOP of that fixed baseline, agreement with the exact elastica
# stays within ~1.7% out to ~13 degrees of tip rotation, crosses ~5%
# around ~21 degrees, and reaches ~8% by ~24 degrees, GROWING (not
# shrinking) with further mesh refinement at a fixed rotation -- i.e.
# this element is quantitatively trustworthy for SMALL-TO-MODERATE tip
# rotation (order 10-15 degrees or less), not the fully general large-
# rotation regime "co-rotational" might suggest without this caveat.
# This is exactly the risk the roadmap's own Section 2.2
# flagged in advance ("a real possibility... should be budgeted as a
# likely... requirement") for the small-rotation approximation Phase A
# adopted.
#
# PARTIAL FIX (2026-09-03, _bending_membrane_coupling_force() below):
# added a von Karman-type membrane strain correction driven by each
# node's bending-rotation DEVIATION FROM ITS OWN ELEMENT'S MEAN (not
# the raw/absolute rotation) -- exactly zero under a single-element
# rigid tilt (verified: max spurious force ~0.0 at 17 degrees of
# uniform tilt), avoiding a repeat of dead ends 1-3 above. Verified
# DIRECTLY that this was NOT sufficient to close the gap: recovered
# foreshortening was only ~0.05%-0.06% of the true elastica value.
#
# DEAD END 4, ABSOLUTE-ROTATION ATTEMPT (tried and REJECTED 2026-09-03,
# same session as the partial fix above -- code reverted to the
# deviation-based version; this account is kept, like dead ends 1-3,
# because it is a real, informative negative result, not a hypothetical
# one). Reasoning that motivated the attempt (still believed correct,
# see below): von Karman membrane strain is a CONFIGURATIONAL quantity
# (a function of current-vs-original shape alone, not of load history),
# so the true physical bending slope of a smoothly-curving structure is
# carried by each element's ABSOLUTE rotation, not by how much its own 4
# nodes disagree with each other (small for any individual element in a
# fine mesh even when the STRUCTURE's overall curvature is large) --
# deviation-from-mean discards exactly this signal by construction. wx/
# wy were changed from "deviation from element mean" to raw absolute
# (betax, betay) to test this. Confirmed directly: this DOES recover
# elastica-scale foreshortening magnitude.
#
# But it also makes the element's own tangent stiffness go locally
# INDEFINITE (a negative eigenvalue, e.g. -3729 measured on a real
# assembled mesh element) at ORDINARY trial states -- ~1.7 degrees of
# local rotation, ~2cm tip deflection on a 1m strip, the kind of
# completely unremarkable intermediate state that arises routinely
# during Newton iteration on any real load step, not an exotic edge
# case. Verified this is NOT a derivative/coding bug: the analytic
# tangent matched full complex-step differentiation of the SAME (now
# indefinite) internal_force() to ~1e-16 relative at the offending
# state -- the indefiniteness is a genuine property of the formulation.
# Root cause: the coupling term's stress scales as Dm*wx^2, and Dm
# (membrane stiffness) is ~1e6-1e7x the bending stiffness this element
# is normally governed by (the same ratio documented throughout this
# file's other sections), so even small-looking absolute rotations
# during an UNCONVERGED Newton trial (not just a converged large-
# rotation equilibrium state) already inject forces wildly
# disproportionate to any physically reasonable applied load.
#
# Tried three different solver strategies specifically to push through
# this, rather than assuming it was unsolvable: (1) plain Newton
# (solve_nonlinear_static) -- diverges catastrophically (residual
# overflow to ~1e32) regardless of load-step fineness, failing even at
# the FIRST, smallest step. (2) solve_nonlinear_transient's default
# line-search + trust-region Newton with mass-matrix regularization
# (K_eff = K_T + a0c*M + a1c*C) -- DID converge, but only on a mesh far
# too coarse to trust (95% transverse-deflection error vs. the exact
# elastica), raising the separate concern that the mass term merely
# numerically papers over an ill-posed static problem rather than
# resolving it, rather than being genuine evidence of correctness.
# (3) solve_nonlinear_arc_length (Crisfield continuation, purpose-built
# to trace through indefinite/limit-point regions) -- failed early
# (step 6 of 40, small cumulative arc length, discriminant < 0 -- no
# real root) on the SAME properly-resolved (nx=20) mesh the original
# elastica benchmark uses. A solver designed exactly for this situation
# still could not push through, which is strong evidence this is a
# genuine local ill-posedness of the added term's scaling, not a
# solver-tuning inconvenience.
#
# This is a DIFFERENT failure mode from dead ends 1-3 (which broke on
# an ISOLATED single-element rigid tilt, arguably an edge case outside
# an assembled mesh's real operating conditions) -- this one breaks on
# a REAL, BC-constrained, assembled mesh at completely ordinary states.
# Closing it needs a redesigned coupling term (e.g. a saturating or
# reduced-stiffness dependence on rotation instead of raw quadratic-in-
# absolute-rotation, so the effective membrane stiffness felt by this
# term doesn't carry the full 1e6-1e7x ratio) -- not attempted, a real
# open problem for a future session, not this one's partial fix.
#
# One correction to an EARLIER version of this comment, made in the
# course of this investigation: it proposed closing the (dead-end-1-3-
# style) isolated-element caveat with a STATEFUL, incrementally-updated
# per-element reference frame, by analogy to "Beam2DCorotational tracks
# its own chord angle as evolving state via init_state()/
# commit_all_states()". That analogy is WRONG -- Beam2DCorotational
# does not use init_state()/commit_all_states() at all; it recomputes
# its chord angle via atan2() on CURRENT total nodal positions every
# call, exact at any rotation with no history needed (beams.py:189-210),
# a trick that works because 2 points exactly determine a direction,
# unlike a 4-node facet's best-fit orientation (see dead end 1). Beyond
# the factual error, the proposal would not have helped even if built
# correctly: since von Karman strain is configurational, not path-
# dependent, an incremental/committed-state formulation would still
# expose only each step's small increment relative to its own last-
# committed baseline -- the same order of quantity as "deviation from
# element mean," just re-baselined every step instead of every element,
# and subject to the identical magnitude shortfall this whole section
# documents. State-tracking is the right tool for genuinely path-
# dependent physics (plasticity, Hex8PlasticJ2); this isn't that.
#
# DEAD END 5, SATURATED-ROTATION ATTEMPT (tried and REJECTED 2026-09-03,
# same investigation as dead end 4 -- code reverted again to the
# deviation-based version). Tried regularizing dead end 4 rather than
# abandoning absolute rotation outright: wx, wy = ref*tanh(wx_raw/ref)
# (ref=0.1 rad, untuned), capping the term's growth for large trial
# excursions while staying near-identity for wx_raw << ref. Genuinely
# mixed result, not a clean rejection: a single-load-level Newton trace
# that diverged under dead end 4 now converged cleanly (residual
# 3.8 -> 3e-7 over 6-9 iterations, tangent briefly dips indefinite at
# iteration 1 but recovers); a coarse 4-step load ramp got further than
# dead end 4 ever did. But a PROPER fine 30-step ramp from zero -- the
# ordinary way this would actually be used -- failed EARLIER (step 6 of
# 30, ~6% of reference load) and WORSE than the coarse version, which is
# backwards for plain step-size sensitivity. Traced iteration-by-
# iteration: every step's first Newton correction overshoots and the
# tangent briefly goes indefinite (true at every step 1-6, self-corrects
# within 3-4 iterations for steps 1-5), but at step 6 the residual, after
# dipping promisingly to 1.7e-4, GROWS GEOMETRICALLY on every subsequent
# iteration while the tangent's smallest eigenvalue stays frozen near
# +0.10 against a largest eigenvalue ~7e9 (condition number ~1e10-1e11,
# matching the ALREADY-documented ill-conditioning of the original
# deviation-based coupling, ~1.7e10 -- but there it merely bounces at a
# noise floor; here it diverges without bound, pointing to a genuine
# instability in the Newton fixed-point map itself, plausibly tied to
# the tanh term's own inflection region, not yet understood). Saturation
# is a real, measurable improvement over raw absolute rotation but does
# NOT make the term robust for ordinary incremental use.
#
# A further check this same investigation ruled OUT as a next idea:
# "the coupling term is missing the geometric-stiffness contribution of
# membrane stress back onto the bending DOFs (the classic plate-
# buckling cross term) -- completing that might fix the non-convexity."
# FALSE on inspection, not just unhelpful: _coupling_tangent_local()
# already complex-steps the FULL 24-DOF Jacobian of the coupling force
# (not a bending-only sub-block), so d(f_add_bending)/d(dof_membrane) is
# already nonzero and present (measured ~7.26e6 on a random test state),
# verified correct to 1e-16 against complex-step of the whole internal_
# force(). K_local0's membrane block plus U_add together already
# EXACTLY reconstruct the standard von Karman total membrane potential
# 0.5*integral(eps_full . Dm . eps_full) with eps_full = eps_lin +
# eps_add -- a complete, exact Hessian of a legitimate (if simplified)
# quartic energy, nothing missing to add. A quartic energy generically
# HAS non-convex regions away from its minimum; that is normal, not a
# modeling gap. The real difficulty is Newton robustness on a genuinely
# non-convex, badly-conditioned energy landscape -- a known-hard problem
# class for von Karman-type formulations, not fixable by "completing
# missing physics."
#
# DEAD END 6, ADAPTIVE ARC-LENGTH RETRY (tried and REJECTED 2026-09-03,
# same investigation, closing the "(a) properly globalized solver" idea
# dead end 5 left open -- code reverted again to the deviation-based
# version). solve_nonlinear_arc_length() does not auto-adapt its radius
# (see that function's own docstring); wrote a custom driver around its
# same predictor/corrector algorithm with real step-size adaptation
# (grow delta_L after a fast-converging step, shrink and retry after a
# failed one, standard practice) and reran dead end 4's exact absolute-
# rotation formulation on the SAME properly-resolved (nx=20) elastica
# mesh. Genuinely different result from dead ends 4-5, not another flat
# rejection: delta_L grew smoothly from 0.003 to its cap over 20 steps
# as convergence stayed fast (1-4 iterations/step), reaching lambda=
# 0.061 (~5.2 degrees tip rotation) with recovered foreshortening at
# 63% of the true elastica value (u_fe=-1.40e-3 vs u_elastica=-2.22e-3)
# -- a genuine order-of-magnitude improvement over dead end 4/5's
# ~0.05%-0.06%. But it then hit a wall: the arc-length quadratic's
# discriminant went negative and STAYED negative even as delta_L was
# shrunk six more orders of magnitude (down to ~1e-9) -- the signature
# of a genuine limit/bifurcation point in the equilibrium path itself,
# not step-size-fixable ill-conditioning. Plain incremental Newton
# (solve_nonlinear_static, no arc-length) independently failed at
# almost the identical load level (lambda=0.057) -- two different
# solvers agreeing is strong evidence the wall is real, not one
# solver's artifact.
#
# Checked directly, and this is what makes the result CONCLUSIVE rather
# than merely disappointing: (1) at this same load level, the
# transverse tip deflection w itself -- not just the coupling-term's own
# foreshortening -- disagrees with the exact elastica by ~20%, versus
# 0.36% for the deviation-based baseline at the IDENTICAL load level
# (confirmed by running the deviation-based version through the same
# test, not assumed) -- so absolute rotation is not cleanly adding a
# small secondary foreshortening correction on top of an otherwise-
# unchanged primary bending response, it is substantially distorting
# that primary response too. (2) The Bisshopp-Drucker elastica BVP for
# a cantilever under a monotonically increasing tip point load is
# smooth for any load level -- the EXACT physical problem has NO limit
# point at all. So the wall both solvers hit at lambda~0.06 is a
# SPURIOUS artifact of this ad hoc quartic energy's own structure, not
# a real physical instability being (correctly) discovered. Even a
# perfect, exotic continuation method that somehow tunnelled past this
# wall would be tracing a mathematically well-defined but PHYSICALLY
# WRONG path -- there is nothing real on the other side to reach.
#
# This closes off direction (a) from the previous version of this
# comment ("a properly globalized solver suited to non-convex Newton")
# conclusively, not just empirically: better solvers can and do make
# more PROGRESS against this formulation, but the formulation itself
# predicts a fictitious singularity and a distorted primary response
# well before reaching it, so no solver improvement fixes the actual
# problem. Only direction (b) remains: a fundamentally different
# (better-conditioned) discretization of the coupling itself, e.g. a
# reduced/selective or assumed-natural-strain treatment specific to
# this term (mirroring how MITC already relieves TRANSVERSE SHEAR
# locking in this same element) rather than the raw quartic energy used
# today, which carries the full ~1e6-1e7 membrane/bending stiffness
# ratio directly into its own conditioning AND (newly confirmed here)
# into a spurious non-physical limit point.
#
# DEAD END 7, MIXED (HELLINGER-REISSNER-STYLE) FORMULATION (tried and
# REJECTED 2026-09-04 -- code LEFT IN PLACE, since it is a genuine,
# separately-useful, OPT-IN capability that simply doesn't yet make
# this element's own coupling term robust; see "What was kept" below).
# Direction (b) above pointed at "a fundamentally different
# discretization"; the literature on exactly this failure class
# (Magisano, Leonetti & Garcea, "Advantages of the mixed format in
# geometrically nonlinear analysis of beams and shells," IJNME 2013,
# 10.1002/nme.4577 -- confirmed via direct research, not assumed. NOTE:
# this is the SAME paper cited as reference [30] in Yang et al. 2019,
# the actual paper `NonLin-HyROM` reproduces) states plainly: "Newton's
# method performance deteriorates in displacement formulations when
# membrane/flexural stiffness ratios increase... mixed formulations
# avoid this problem by using stress unknowns as independent
# variables" -- exactly the ~1e6-1e7 ratio and exactly the Newton
# indefiniteness dead ends 4-6 measured.
#
# PILOT FIRST (NonLin-HyROM/pilot_mixed_formulation/, standalone, no
# fea_engine changes): a discrete "extensible elastica" chain of
# von-Karman-coupled segments, EA/Kb~1e7, reproduced dead end 4's exact
# failure signature in miniature (plain Newton stalls/goes indefinite
# as chain length or single-jump load grows) -- and a GENUINE joint
# (simultaneous) mixed Newton solve of that SAME toy fixed it completely
# (2-3 iterations, every case, vs. stalling/indefinite for the
# displacement-only version). This confirmed the METHOD works before
# any real element was touched.
#
# ARCHITECTURE ADDED (Module 23, `solver.py`/`nonlinear_solver.py`,
# KEPT, working, validated -- see FESystem.init_iter_state()/
# update_iter_states()'s own docstrings and tests/test_iter_state.py):
# a genuinely new, general-purpose per-element internal-unknown
# mechanism, deliberately SEPARATE from init_state()/commit_all_states()
# because it updates every Newton ITERATION (not just every converged
# step) -- needed because a mixed formulation's own internal stress
# unknown must be corrected using the REALIZED Newton step, information
# no earlier extension point exposed. Fully opt-in (None by default,
# zero effect on any of the 165+ pre-existing tests), and this part
# works correctly and generally -- any future element could use it.
#
# WHAT ACTUALLY FAILED, specific to THIS element's coupling term: tried
# to apply the SAME "joint" idea by (a) switching wx/wy to ABSOLUTE
# rotation (dead end 4's own correct physics) and (b) tracking the
# add-on stress N_add as per-Gauss-point iter_state, corrected each
# call via the REALIZED displacement delta since the element's own last
# call (the only architecturally available option without literally
# adding N_add as extra GLOBAL DOFs). Two real bugs were found and
# fixed along the way -- first, the outer Newton tangent was missing
# the Schur-complement correction term entirely (K_dd alone, not
# K_dd - K_dN@K_NN^-1@K_Nd), confirmed via complex-step to be a genuine
# omission, not a simplification; second, a follow-up derivation showed
# the CONDENSED RESIDUAL must always evaluate the coupling force from
# the exact, fresh Dm@eps_add (never N_add's own tracked value) for
# correctness -- so the real force is now IDENTICAL, in mixed mode, to
# a pure absolute-rotation formula, and N_add's only remaining role is
# to build a deliberately SMOOTHED (lagged) Newton tangent via a
# separate function, _bending_membrane_coupling_force_for_tangent().
# Every analytic piece (K_dd, K_dN, K_Nd, K_NN) was verified to match
# complex-step of the true joint residual system to floating-point
# precision -- this is NOT a sign error or coding bug.
#
# Even so, the assembled elastica benchmark mesh overflows within the
# FIRST load step. Traced directly (not guessed): the first Newton
# correction is IDENTICAL between mixed and non-mixed modes (both use
# N_add=0, confirmed byte-for-byte), so the divergence starts exactly
# at the SECOND iteration -- the first point where update_iter_state()
# has corrected N_add away from its zero initial value. Compared side
# by side at the same load step: non-mixed shows a normal, benign
# Newton overshoot-then-converge (|R| = 1.89 -> 181.6 -> 2.2e-7);
# mixed diverges geometrically from the same starting point (|R| =
# 1.89 -> 1.56e4 -> 2.04e4 -> 3.5e4 -> ... -> 5.6e8 over 15 iterations).
# Root cause: N_add's lagged update is itself a LINEARIZATION of its
# own governing equation around the PREVIOUS call's state, valid only
# for a small step between calls -- but N_add starts at EXACTLY zero
# and must jump to something substantial in the very first correction,
# which is not a small step by construction. This is not a tunable
# threshold or a fixable sign: it EXACTLY reproduces (now at the real
# element, not just the toy) a failure already found and confirmed in
# NonLin-HyROM/pilot_mixed_formulation/pilot_lagged_condensation.py --
# a truss/chain toy whose OWN lagged-N-update scheme worked perfectly
# when the true equilibrium N was zero, then failed identically the
# moment a genuine (nonzero-at-equilibrium) axial force was introduced.
# Two toy fixes (heavy under-relaxation of the N update, damping
# factors from 1.0 down to 0.01) were tried on that toy and BOTH still
# failed to converge -- ruling out "just damp it more" as a fix here
# too, without needing to re-run that experiment on the real element.
#
# What this closes off, and what remains open: a LAGGED (one-call-
# behind) internal-unknown update is NOT a viable way to get the mixed
# formulation's benefit for a coupling term whose true value is
# generically nonzero (as this one's is, unlike a toy problem
# specifically built with a zero-at-equilibrium internal force) --
# confirmed at both toy and real-element scale, with every analytic
# piece independently verified correct. The two paths that remain
# genuinely open (NOT attempted): (a) sub-iterate N_add to actual
# convergence WITHIN each load step, before trusting it for the next
# real displacement correction (a properly-converged staggered/block
# Gauss-Seidel scheme, not the single-lagged-correction version tried
# here); (b) give N_add true joint-DOF status so it is solved
# SIMULTANEOUSLY with displacement in one linear system, eliminating
# the lag entirely -- the mechanism the pilot actually validated works,
# at the cost of being the heaviest option (global DOF-count/assembly
# changes, not just a per-element addition).
#
# What was kept, deliberately, not reverted: the Module 23
# infrastructure itself (`solver.py`'s iter_state mechanism,
# `nonlinear_solver.py`'s wiring into five drivers) is real, tested,
# working, general-purpose capability, useful for the next attempt at
# this problem (or any future mixed-formulation element) regardless of
# this specific coupling-term application not yet working --
# `_bending_membrane_coupling_force_for_tangent()`/
# `_coupling_schur_correction()`/`init_iter_state()`/
# `update_iter_state()` below are ALSO kept, correctly implemented and
# individually verified, as the documented, checkable starting point
# for whoever attempts (a) or (b) above next, not because they
# currently produce a working element. iter_state remains fully
# opt-in (default None) -- do NOT call `fesystem.init_iter_state()` on
# a `Shell4MITCCorotational` model expecting a working nonlinear solve
# today; the default (no iter_state) path is what actually ships and is
# unaffected by any of this.
#
# DEAD END 8, ELEMENT-CONSTANT ABSOLUTE ROTATION (tried and REJECTED
# 2026-09-08, Wave 4 item 21 -- code NOT kept, this was a standalone
# experiment, never merged into this method). Direction (b) from dead
# end 4's own docstring ("a fundamentally different, better-conditioned
# discretization of the coupling term itself... mirroring how MITC
# already relieves transverse shear locking... rather than the raw
# quartic energy used today") suggested REDUCED INTEGRATION as the
# untried fix. Hypothesis: keep dead end 4's ABSOLUTE rotation (needed
# to recover elastica-scale foreshortening magnitude -- the deviation
# mode this method actually ships only recovers ~0.05-0.06% of it, see
# _bending_membrane_coupling_force()'s own docstring), but represent
# Wx, Wy as a SINGLE ELEMENT-CONSTANT value (mean of the 4 nodal
# betax/betay, not N(p)@wx bilinearly interpolated at each Gauss point)
# -- i.e. apply the reduction to the coupling term's own STRAIN FIELD,
# not just the quadrature rule, on the reasoning that a lower-order
# representation of Wx/Wy might be what keeps the added quartic energy
# from creating the local indefiniteness dead end 4 hit.
#
# Tested the ONLY way that setting found dead ends 4-6's own failure
# (isolated single-element complex-step eigenvalue checks are NOT
# sufficient -- run independently here first and came back
# inconclusive, tiny normalized eigenvalues at every tested scale,
# because the real failure only shows up as an intermediate Newton
# iterate on an assembled, boundary-constrained mesh): a real multi-
# element `solve_nonlinear_static()` run on the same cantilever-strip
# geometry `test_shell_corotational_elastica.py` uses (nx=6 instead of
# that file's nx=20, to fit this sandbox's ~170s command budget,
# ny=2 unchanged). Result: the deviation-mode (shipped) element
# converges cleanly at 3 coarse load steps up to P/P_ref=0.15 in 2.7s.
# The element-constant-absolute variant FAILS to converge -- plain
# Newton stalls at |R|~9.4 after 40 iterations at just P/P_ref=0.10 (3-
# step ramp), and STILL fails, now at |R|~1.9 after 40 iterations at
# P/P_ref=0.08, even with a much finer 15-step ramp from 0.01 to 0.15 --
# i.e. finer load stepping does not rescue it, the signature of a
# genuine limit point / local indefiniteness, not merely an
# under-resolved load increment (exactly how dead end 4 was itself
# originally diagnosed).
#
# Conclusion: reducing the bilinear Wx/Wy field to element-constant
# does NOT fix dead end 4's indefiniteness -- the problem is
# structural to using ABSOLUTE (not deviation-from-mean) rotation in
# this quadratic-in-rotation coupling term on an assembled multi-
# element mesh, not an artifact of the field's spatial order. This
# closes off "just reduce the field order" as a fix; whatever
# "different, better-conditioned discretization" direction (b)
# originally had in mind, a naive reduced-integration-style
# simplification of Wx/Wy alone is not it. The two paths dead end 7
# left open (properly-converged staggered N_add sub-iteration, or true
# joint-DOF status for N_add) remain the only genuinely untried
# directions; MITC's own transverse-shear fix (which this direction
# was named after) works by changing which STRAIN COMPONENT is sampled
# and where, not by coarsening a field that was already being sampled
# correctly -- a hint that a real fix here would need a comparably
# specific, not-yet-identified reformulation of the coupling term
# itself, not a generic "coarsen it" heuristic.
#
# Read this whole "Design history" (8 dead ends now) before proposing
# a 9th.
#
# ITEM 19 ANALYSIS (2026-09-08, Wave 4 -- an analytical argument, not a
# new experiment; no code changed here): item 19 names the two paths
# dead end 7 left open -- (a) sub-iterate N_add to real convergence
# within a load step (a properly-converged staggered/block Gauss-Seidel
# scheme), or (b) give N_add true joint-DOF status, solved
# SIMULTANEOUSLY with displacement in one linear system. Before
# attempting either (both are real implementation effort -- (a) needs a
# new staggered outer loop in nonlinear_solver.py, (b) needs global
# DOF-count/assembly changes), it is worth noting what the class's own
# already-established "IMPORTANT CORRECTION" fact (in
# _bending_membrane_coupling_force()'s docstring above) implies for
# each: the coupling FORCE returned to the outer Newton solve is,
# UNCONDITIONALLY, the fresh Dm@eps_add(dof_local) -- N_add is proven
# (both algebraically and by the direct floating-point-overflow
# experiment that established this correction) to never enter the
# force itself, only the TANGENT. This means the EQUILIBRIUM PROBLEM
# ("find dof_local with force=0") being solved is, regardless of how
# N_add is managed, EXACTLY dead end 4's own original absolute-rotation
# problem -- lagged, staggered, and joint-DOF approaches can only ever
# change how Newton SEARCHES for a root of that one fixed residual, not
# change what the root is or whether one exists at a given load level.
# By the implicit function theorem, this also means path (a) taken
# literally -- sub-iterating N_add to EXACT convergence at a FIXED
# dof_local (trivial here: R_N=eps_add-Dm^-1@N_add=0 is affine in
# N_add, so its exact solution is N_add=Dm@eps_add(dof_local) in one
# algebraic step, not an iterative process) and then using that exact
# N_add to build the Schur-corrected condensed tangent, reproduces dead
# end 4's own tangent EXACTLY (this is what "exact static condensation"
# means) -- zero benefit, by construction, not by experiment. A
# staggered scheme that instead OMITS the Schur correction while N_add
# is held fixed within an inner solve (a genuinely different linear
# system at each inner step, not condensation) is not ruled out by this
# argument, and is the only reading of (a) that could plausibly differ
# from dead end 4 -- but dead end 4's own account already reports that
# arc-length continuation (path-following built specifically to trace
# through indefinite/limit-point regions, exactly what a bad-Newton-
# path explanation would need to be rescuable by) ALSO failed to find a
# trustworthy, mesh-resolved result -- evidence pointing toward a
# genuine absence/instability of the equilibrium branch itself at that
# load level for THIS residual, not merely a Newton-path artifact any
# alternative search strategy (staggered included) could route around.
# Path (b) (true joint DOFs) is the one direction this argument does
# NOT rule out, because a monolithic block-saddle-point linear solve
# changes which MATRIX gets inverted at each Newton step (not just the
# search direction for the same matrix) -- but it is also the
# "heaviest option" dead end 7 already flagged (global DOF-count/
# assembly changes across FESystem/solver.py, affecting every element
# type, not a per-element addition), disproportionate effort for a
# single documented, non-load-bearing coupling term when the shipped
# deviation-mode default already works correctly through the small-to-
# moderate-rotation regime this element is validated for. Deferred on
# this basis rather than attempted -- if revisited, (b) is the only
# remaining candidate worth the effort; (a) is not, per the argument
# above.
#
# Tangent (Phase B, DONE -- roadmap Section 2.2's recommended
# ordering: "implement internal_force() first... get a numerically
# CONSISTENT tangent for free via complex-step differentiation... THEN
# hand-derive the analytic geometric tangent as a follow-up speed
# optimization"): tangent_stiffness() is now the analytic material-
# plus-geometric split T.T @ K_local0 @ J + K_geo (see
# _dof_local_jacobian_and_geo()'s own docstring for the full
# derivation), matching Tet10SolidTL's own Phase A -> Phase B path
# exactly -- the ORIGINAL Phase A complex-step implementation is kept,
# renamed _tangent_stiffness_complex_step(), as the independent
# reference the analytic version was validated against (~1e-16
# relative at u=0, random small/large states, the bending-dominated
# state that broke the rejected Phase A dof-extraction designs, and
# rigid rotations up to 30 degrees -- tests/test_shell_corotational.py)
# and as a debugging fallback if this class's kinematics ever change
# again. f_int(u) = T(u).T @ K_local0 @ dof_local(u) is not exactly the
# gradient of a scalar energy in this construction (T(u) and dof_local
# (u) draw on the same drilling angle but are not tied together tightly
# enough to guarantee an exact symmetric Hessian), so the raw analytic
# Jacobian carries the same small asymmetric part the complex-step
# version does (confirmed: the two match to ~1e-16 BEFORE either is
# symmetrized, so this is not a Phase-A-vs-Phase-B discrepancy, it is
# a genuine property of f_int itself); tangent_stiffness() symmetrizes
# it (0.5*(K+K.T)) before returning, same as Phase A -- Newton-Raphson
# only needs a good-enough linearization to converge, not an exact
# Hessian.
# =====================================================================
class Shell4MITCCorotational(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 4, 6, 2, 2
    translational_dof_mask = [True, True, True, False, False, False]

    def __init__(self, drilling_factor=1e-3):
        """Wraps a plain Shell4MITC (same drilling_factor convention,
        see that class's own __init__ docstring) as the source of the
        local material stiffness -- composition, not re-derivation,
        exactly as Shell4MITC itself composes Quad4PlaneStress/
        Quad4MindlinPlate."""
        self._linear = Shell4MITC(drilling_factor)

    def shape_and_derivs(self, natural_coords):
        """Interface completeness only (mesh.check_quality()), same as
        every other nonlinear element's override of this method --
        internal_force()/tangent_stiffness() below don't call it."""
        return self._linear.shape_and_derivs(natural_coords)

    @staticmethod
    def _skew(v):
        """3x3 skew-symmetric (cross-product) matrix of a 3-vector v --
        polynomial in v's own components, no norm/abs/branching, so
        this stays holomorphic for COMPLEX v (needed since v here is
        mean_theta, which is complex during tangent_stiffness()'s
        complex-step evaluation of internal_force()). Used for the
        SMALL-rotation frame update R0 @ (I - skew(mean_theta)) -- the
        first-order truncation of the exact Rodrigues rotation formula,
        deliberately NOT the exact (sin/cos-based) version: the exact
        version needs a norm(mean_theta) that is exactly zero at
        mean_theta=0 (the very common case of a pure-translation
        complex-step perturbation direction), and 0/0-style branching
        on that norm would itself break complex-step holomorphy (the
        same class of problem _c_norm-style helpers exist to avoid
        elsewhere in this package -- see Tet10SolidTL's own
        internal_force() docstring on avoiding abs()/branching). The
        small-rotation truncation has no such singularity and matches
        this class's already-declared MVP scope (roadmap Section 2.2)."""
        z = 0.0 * v[0]
        return np.array([
            [z, -v[2], v[1]],
            [v[2], z, -v[0]],
            [-v[1], v[0], z],
        ])

    @staticmethod
    def _exact_drill_rotation(theta):
        """Wave 4 item 18 (docs/consolidated_future_roadmap.md): EXACT
        replacement for _skew()'s first-order-truncated drilling update,
        used by _local_relative_dofs()/_dof_local_jacobian_and_geo() as
        R_drill = R0 @ _exact_drill_rotation(theta_z_mean).

        Unlike a GENERAL 3-vector Rodrigues update (which _skew()'s own
        docstring rules out for the reason above -- it needs
        norm(mean_theta), singular/non-holomorphic at mean_theta=0),
        this is safe to make exact: drilling is, by this element's own
        construction (see class-level "Design history"), ALWAYS a
        rotation about a single FIXED axis -- R0's own local z. A
        single-axis rotation by scalar angle theta needs only cos(theta)
        and sin(theta), both ENTIRE (holomorphic everywhere, no
        norm/branching) functions of a COMPLEX theta -- so this stays
        complex-step-safe with no singularity, without needing the
        general 3-vector axis-angle machinery _skew()'s docstring
        correctly avoided for the bending block.

        Verified (docs/consolidated_future_roadmap.md Wave 4 item 18
        investigation): matches _skew()-based (I - skew([0,0,theta]))
        to O(theta^2) as theta->0 (confirmed numerically: the SIGN
        convention here is Rz(-theta), i.e. this element's original
        first-order code was linearizing a CLOCKWISE (about local +z)
        small rotation, not Rz(+theta) -- direct comparison at
        theta=1e-3 gives 5e-7 agreement with Rz(-theta) vs. 2e-3 with
        Rz(+theta), an order-of-magnitude-clean discriminator, not a
        coin flip)."""
        c = np.cos(theta)
        s = np.sin(theta)
        one = 1.0 + 0.0 * theta
        zero = 0.0 * theta
        return np.array([
            [c, s, zero],
            [-s, c, zero],
            [zero, zero, one],
        ])

    @staticmethod
    def _exact_drill_rotation_deriv(theta):
        """d(_exact_drill_rotation(theta))/d(theta), real-valued only
        (used by _dof_local_jacobian_and_geo()'s ANALYTIC tangent, which
        is never itself complex-stepped -- unlike _exact_drill_rotation()
        above, this does not need to stay holomorphic). Closed form,
        cross-checked against central finite-difference to ~1e-11
        relative at theta=0.37 (see Wave 4 item 18 investigation)."""
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [-s, c, 0.0],
            [-c, -s, 0.0],
            [0.0, 0.0, 0.0],
        ])

    # =================================================================
    # Wave 4 item 46, building block A (docs/shell_rotation_coupling_
    # fix_roadmap.md): general exact SO(3) rigid-rotation extraction,
    # generalizing item 18's exact single-axis drilling update
    # (_exact_drill_rotation above) to a genuine 3-vector rotation. This
    # is intentionally scoped NARROWLY and is NOT yet wired into
    # _local_relative_dofs()/internal_force() -- see the "NOT wired in"
    # note on _mean_rigid_rotation()'s own docstring for why, and read
    # this whole comment before changing that.
    #
    # WHY THIS IS SCOPED THE WAY IT IS (re-derived 2026-09-08, before
    # writing any of this code, specifically to avoid reproducing a 9th
    # dead end): the roadmap's own "building block A" description
    # sketched extracting ONE rigid rotation R_e and using it uniformly
    # to re-derive BOTH the membrane block (translation, via the same
    # position-rotate-and-subtract trick R_drill already uses) AND the
    # bending block. Re-deriving this against `shells.py`'s own "Design
    # history" BEFORE implementing shows that doing so naively would
    # exactly reproduce DEAD END 3: "an intermediate variant... tried
    # subtracting a rigid-tilt-induced position term... using the FULL
    # 3-D tilt frame rather than a drilling-only one -- this made the
    # spurious membrane force WORSE, not better (a full-tilt frame's
    # z-row is not [0,0,1], so it re-couples w into x/y from a different
    # direction)". R_drill's z-row is EXACTLY [0,0,1] identically (a
    # single-axis, local-z-only rotation cannot tilt out of plane by
    # construction) -- which is exactly why R_drill's position-based
    # membrane extraction is safe (see item 20's own "found to require a
    # genuinely novel construction" conclusion) and exactly why a
    # GENERAL 3-vector R_e (whose z-row is generally NOT [0,0,1] the
    # moment any real bending is present) is NOT safe to use the same
    # way for translation.
    #
    # What IS safe, and what this block actually implements: using a
    # general exact R_e to extract the BENDING BLOCK'S OWN ROTATIONAL
    # dofs (theta_x_local, theta_y_local) -- NOT translation (w stays
    # exactly as `_local_relative_dofs()` already computes it, projected
    # onto the FIXED reference frame, untouched). This closes a
    # DIFFERENT, smaller, already-quantified gap than the membrane-
    # bending coupling/foreshortening problem: item 18's own "Known
    # remaining limitation" note left the BENDING block's single-element
    # large-RIGID-TILT error explicitly unclosed after fixing drilling
    # ("out-of-plane tilt ~0.03% at 0.5 degrees,... ~1.8% at 30
    # degrees"), precisely because the bending rotations were extracted
    # via a raw, u-independent projection onto the FIXED reference frame
    # (e1_0,e2_0,e3_0) rather than relative to the element's own CURRENT
    # mean rotation. Using this block's `_mean_rigid_rotation()` to
    # extract theta_x_local/theta_y_local as an EXACT (any-magnitude,
    # not small-angle-truncated) relative rotation should close that
    # residual to exact zero, the same way item 18 closed drilling's
    # analogous gap -- but does NOT, by itself, touch the membrane-
    # bending coupling/foreshortening problem (items 19/21, dead ends
    # 4-8), which needs building block B's additive Koiter-Sanders
    # strain redesign, not this rotation-only extraction. Wiring even
    # this narrower rotation-only use into `_local_relative_dofs()` is
    # left for a follow-up validation pass (needs its own single-element
    # rigid-tilt regression test, analogous to
    # `test_shell_corotational_exact_drill.py`, PLUS a full-suite
    # regression run before it can be trusted) -- not done in this pass.
    # =================================================================

    @staticmethod
    def _exp_map(theta_vec):
        """Exact exponential map: rotation matrix for a general 3-vector
        rotation `theta_vec` (axis = direction, angle = norm). Real-
        valued only (no complex-step safety attempted or needed -- this
        block is not called from `internal_force()`/`tangent_stiffness()`
        , see the comment above). Same exact Rodrigues construction, and
        the same norm-based (not naive) small-angle branch, that
        `beams3d.py`'s `Beam3DCorotational._axis_rotation()`/`_slerp()`
        already use and validated (item 23) -- reused here rather than
        re-derived from scratch."""
        v = np.asarray(theta_vec, dtype=float)
        angle = float(np.linalg.norm(v))
        if angle < 1e-10:
            K = Shell4MITCCorotational._skew(v)
            return np.eye(3) + K + 0.5 * (K @ K)
        axis = v / angle
        K = Shell4MITCCorotational._skew(axis)
        return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    @staticmethod
    def _log_map(Rmat):
        """Exact logarithmic map (inverse of `_exp_map`): the rotation
        VECTOR of a rotation matrix. Identical construction to
        `Beam3DCorotational._rotation_vector()` (arctan2-based, NOT
        arccos-based, for the same catastrophic-cancellation-near-
        theta=0 reason documented there) -- reused, not re-derived.
        Same documented degeneracy near theta=pi (axis ill-conditioned);
        not expected to matter for one element's own small internal
        rotations relative to its own mean."""
        R = np.asarray(Rmat, dtype=float)
        v = 0.5 * np.array([R[2, 1] - R[1, 2],
                             R[0, 2] - R[2, 0],
                             R[1, 0] - R[0, 1]])
        sin_theta = float(np.linalg.norm(v))
        if sin_theta < 1e-10:
            return v   # theta/sin_theta -> 1 in this limit
        cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        theta = np.arctan2(sin_theta, cos_theta)
        return v * (theta / sin_theta)

    @staticmethod
    def _mean_rigid_rotation(theta_nodes, max_iter=20, tol=1e-13):
        """General SO(3) Frechet/Karcher mean of the 4 nodes' current
        TOTAL rotation vectors `theta_nodes` (shape (4,3)), generalizing
        `_exact_drill_rotation`'s single-axis exactness to a genuine
        3-vector rotation -- the quad-element analogue of Grange &
        Bertrand's Section 3 rigid-body-motion extraction (a well-posed
        Newton-Raphson residual, not a heuristic best-fit-normal frame,
        which is exactly what made "Design history" dead end 1 fail).

        Returns (R_e, r_nodes): R_e is the 3x3 mean rotation matrix;
        r_nodes is (4,3), each row the EXACT (any-magnitude, not small-
        angle-truncated) relative rotation vector of that node's own
        total rotation with respect to R_e -- `_log_map(R_e.T @ R_i)`.

        Algorithm: Newton/fixed-point iteration on SO(3), NOT a naive
        arithmetic mean of rotation VECTORS (which is only correct to
        first order and is exactly the kind of small-rotation
        approximation item 18 replaced for drilling). Initialize R_e
        from the arithmetic mean rotation vector (a good starting guess
        for nodes whose rotations are all close together, the normal
        case for one element), then repeatedly: (1) map every node's
        rotation into R_e's own tangent space, r_i = log_map(R_e.T @
        R_i); (2) the Frechet-mean stationarity condition is
        mean_i(r_i) = 0 (the tangent-space residuals cancel); correct
        R_e by composing R_e <- R_e @ exp_map(mean_i(r_i)) and repeat
        until mean_i(r_i) is below `tol`. Converges in a handful of
        iterations for any physically realistic single-element rotation
        spread (checked directly, not assumed -- see
        `test_shell_corotational_mean_rotation.py`), since 4 nodes of
        one element are never far apart on SO(3) even at large GLOBAL
        structure rotation.

        Exactness property (the actual acceptance test for this
        function, mirroring `test_shell_corotational_exact_drill.py`'s
        own single-axis check but for a GENERAL, not z-only, rigid
        rotation): if all 4 nodes carry the IDENTICAL rotation vector
        (a pure single-element rigid rotation about ANY axis, any
        magnitude 0.5-90 degrees), R_e must equal that common rotation
        EXACTLY and every r_i must be EXACTLY zero (to floating-point
        noise) -- not merely small. This is what a Frechet mean
        guarantees by construction (zero variance implies the mean
        equals every sample), unlike a hand-derived heuristic frame."""
        theta_nodes = np.asarray(theta_nodes, dtype=float)
        R_nodes = [Shell4MITCCorotational._exp_map(theta_nodes[i]) for i in range(4)]
        R_e = Shell4MITCCorotational._exp_map(theta_nodes.mean(axis=0))
        r_nodes = np.zeros((4, 3))
        for _ in range(max_iter):
            for i in range(4):
                r_nodes[i] = Shell4MITCCorotational._log_map(R_e.T @ R_nodes[i])
            correction = r_nodes.mean(axis=0)
            if np.linalg.norm(correction) < tol:
                break
            R_e = R_e @ Shell4MITCCorotational._exp_map(correction)
        for i in range(4):
            r_nodes[i] = Shell4MITCCorotational._log_map(R_e.T @ R_nodes[i])
        return R_e, r_nodes

    @staticmethod
    def _rotation_matrix_c(e1, e2, e3):
        """Complex-step-safe reimplementation of Shell4MITC.
        _rotation_matrix(): identical block-diagonal-of-R construction,
        but allocates T with e1/e2/e3's own dtype rather than
        np.zeros()'s hardcoded float64 default. Shell4MITC._rotation_
        matrix() is only ever called with a REAL (e1,e2,e3) (the
        reference frame, fixed once per element), so it never hits
        this; this class's CURRENT frame is exactly the thing
        tangent_stiffness()'s complex-step perturbs, so calling the
        original here would silently truncate the imaginary part on
        assignment into a real-dtype array (a numpy ComplexWarning,
        easy to miss) and corrupt every derivative that passes through
        the frame rotation -- caught by inspection before it became a
        numerical bug, not found via a failing test."""
        R = np.vstack([e1, e2, e3])
        T = np.zeros((24, 24), dtype=R.dtype)
        for i in range(8):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return T

    def _local_material_stiffness(self, elem_coords, D):
        """K_local0 (24x24): Shell4MITC's local membrane+bending+shear+
        drilling stiffness, evaluated at the REFERENCE local geometry --
        mirrors Beam2DCorotational's k_local = EA/L0, EI/L0 (a
        REFERENCE-configuration quantity, not updated with deformation;
        see this module's class-level comment block above). Recovered
        from Shell4MITC.stiffness() (which already builds this
        internally, then rotates it to global via that SAME reference
        frame) by undoing the rotation: T0 is orthonormal, so
        K_local0 = T0 @ K_global0 @ T0.T. elem_coords here is always
        real (the reference geometry never varies with u_elem), so
        Shell4MITC's own (non-complex-safe) private helpers are fine to
        call directly."""
        X_ref = np.asarray(elem_coords, dtype=float)
        K_global0 = self._linear.stiffness(X_ref, D)
        e1_0, e2_0, e3_0, _ = self._linear._local_frame_and_coords(X_ref)
        T0 = self._linear._rotation_matrix(e1_0, e2_0, e3_0)
        return T0 @ K_global0 @ T0.T

    def _local_relative_dofs(self, elem_coords, u_elem):
        """Returns (dof_local, e1, e2, e3): the local relative DOF
        vector built from two group-appropriate extractions (this
        module's class-level comment block, steps 2-3) plus the CURRENT
        drilling frame (e1,e2,e3), needed by internal_force() to rotate
        the resulting local force back to global (step 5).

        Membrane+drilling ({u,v,theta_z} per node): position-based, via
        the drilling-only frame R_drill -- rotate the CURRENT nodal
        position (relative to the current centroid) into R_drill and
        subtract the REFERENCE local position (relative to the
        reference centroid); local drilling rotation is theta_z(node)
        minus the frame's own mean theta_z. Bending+shear ({w,theta_x,
        theta_y} per node): used RAW, no frame correction. See the
        class-level comment block for why each choice is safe for its
        own DOF group (in particular, why R_drill -- which only ever
        rotates about local z -- cannot leak an out-of-plane
        translation into the membrane block the way a full 3-D tilt
        frame would).

        dof_local(u_elem=0) is exactly zero (at u=0, theta_z_mean=0 so
        R_drill=R0, current position equals reference position, and the
        raw bending DOFs are all zero), so internal_force(elem_coords,
        0, D) == 0, as required of any nonlinear element with no
        prestress -- verified directly in
        tests/test_shell_corotational.py."""
        X_ref = np.asarray(elem_coords, dtype=float)
        u = np.asarray(u_elem)
        u_nodes = u.reshape(4, 6)
        u_trans = u_nodes[:, 0:3]
        theta_global = u_nodes[:, 3:6]

        e1_0, e2_0, e3_0, _ = self._linear._local_frame_and_coords(X_ref)
        R0 = np.vstack([e1_0, e2_0, e3_0])
        centroid_ref = X_ref.mean(axis=0)
        Xref_local = (X_ref - centroid_ref) @ R0.T   # (4, 3); z ~ 0 (flat facet)

        mean_theta = theta_global.mean(axis=0)
        theta_z_mean = mean_theta[2]
        R_drill = R0 @ self._exact_drill_rotation(theta_z_mean)
        e1, e2, e3 = R_drill[0], R_drill[1], R_drill[2]

        x_current = X_ref + u_trans
        centroid_current = x_current.mean(axis=0)
        x_current_local = (x_current - centroid_current) @ R_drill.T   # (4, 3)

        u_local_mem_xy = x_current_local[:, 0:2] - Xref_local[:, 0:2]   # (4, 2)
        thetaz_local = theta_global[:, 2] - theta_z_mean                # (4,)

        # Bending block, projected onto the FIXED reference frame
        # (e1_0, e2_0, e3_0), NOT used raw in the global frame: w_local
        # is the component of the node's global translation along the
        # reference normal e3_0, theta_x_local/theta_y_local are the
        # components of the global rotation VECTOR along e1_0/e2_0. This
        # is a CONSTANT (u-independent) linear map -- R0 is evaluated
        # once from the reference geometry, so this introduces none of
        # the deformation-tracking-frame leaks "Design history" dead
        # ends 1-3 found; it only matters when R0 != identity, i.e.
        # whenever the element's own local frame is not trivially
        # aligned with the global axes (a rotated-in-plane element, or
        # one whose corner winding gives e3_0 pointing along -global-z
        # rather than +global-z). Every test mesh through Phase C used
        # axis-aligned CCW rectangle_mesh() strips, where R0 IS the
        # identity, so raw global components and this projection are
        # numerically identical there -- this generalization is a
        # strict no-op on everything already validated, and only changes
        # behavior for meshes with a non-identity reference frame (see
        # tests/test_shell_corotational_reference_frame.py, added when
        # this was found via a real gmsh-generated wing mesh whose
        # element winding gave e3_0 = -global-z: the u=0 tangent was off
        # by 60-84% relative before this fix, exactly 0 after).
        dof_local = np.zeros(24, dtype=u.dtype)
        for i in range(4):
            dof_local[6 * i + 0] = u_local_mem_xy[i, 0]
            dof_local[6 * i + 1] = u_local_mem_xy[i, 1]
            dof_local[6 * i + 5] = thetaz_local[i]
            dof_local[6 * i + 2] = u_trans[i] @ e3_0
            dof_local[6 * i + 3] = theta_global[i] @ e1_0
            dof_local[6 * i + 4] = theta_global[i] @ e2_0
        return dof_local, e1, e2, e3

    def _local_relative_dofs_exact_rotation(self, elem_coords, u_elem):
        """Wave 4 item 46, building block A, step 3 (docs/shell_rotation_
        coupling_fix_roadmap.md Section 4.1): a NEW, OPT-IN dof_local
        extraction that replaces `_local_relative_dofs()`'s rotational
        DOF treatment with `_mean_rigid_rotation()`'s exact SO(3)
        extraction, closing item 18's own left-open "Known remaining
        limitation" for the bending block's single-element large-rigid-
        TILT error (~1.8% at 30 degrees).

        NOT wired into `internal_force()`/`tangent_stiffness()` -- see
        two real blockers found while attempting that in this same pass,
        neither a hypothetical risk:

        1. HOLOMORPHY. `internal_force()`'s own docstring requires the
           WHOLE call chain to stay holomorphic in `u_elem`, because
           `_tangent_stiffness_complex_step()` complex-steps it directly
           and is the correctness oracle this class's own analytic
           tangent (and every dead-end investigation) is checked
           against. `_mean_rigid_rotation()`/`_exp_map()`/`_log_map()`
           are REAL-valued only (`np.linalg.norm`, `float()` casts, real
           branching) -- exactly the kind of norm/branching singularity
           `_skew()`'s own docstring already flagged as the reason a
           general 3-vector exact update was ruled out for the
           complex-step path when item 18 was scoped to drilling only.
           Making `_exp_map` holomorphic is tractable (sin(theta)/theta
           and (1-cos(theta))/theta^2 are entire functions of
           theta^2=v.v, a polynomial in v, so they can be evaluated
           without ever taking sqrt/norm of a complex value in a way
           that isn't itself entire) -- NOT attempted here. Making
           `_log_map` holomorphic is materially harder (arctan2 is not
           holomorphic and has no straightforward entire-function
           substitute the way the exponential map's trig ratios do) --
           genuinely unresolved, not just unattempted.
        2. Consequently, this method is REAL-valued only and is NOT
           safe to call from the existing complex-step-differentiated
           `internal_force()`. Using it there would require either (a)
           the holomorphic redesign above (item 1's harder half still
           open), or (b) switching this element's tangent strategy to
           REAL central finite-difference, following the EXACT precedent
           `Beam3DCorotational` (item 23) already established for a
           closely related reason -- not done here, left as the
           concrete next step.

        Design of what IS implemented: translation ({u,v,w} per node)
        is EXACTLY UNCHANGED from `_local_relative_dofs()` -- still
        R_drill-based position-subtraction for u,v and fixed-e3_0
        projection for w -- deliberately, because re-deriving
        translation through a general (non-z-row-[0,0,1]) frame is
        exactly what "Design history" dead end 3 already found breaks.
        Only the ROTATIONAL dofs ({theta_x,theta_y,theta_z} per node)
        change: instead of raw projection onto the fixed (e1_0,e2_0)
        for bending and a SEPARATE exact z-only diff (R_drill) for
        drilling, ALL THREE come from ONE consistent extraction --
        `r_nodes = _mean_rigid_rotation(theta_global)[1]`, so
        theta_x_local=r_nodes[:,0], theta_y_local=r_nodes[:,1],
        theta_z_local=r_nodes[:,2]. This does not reintroduce the
        R_drill-vs-R_e inconsistency risk it might look like at first:
        R_drill (still used for translation only) and R_e (used for
        rotation only) never interact or need to agree with each other
        -- each is independently exact for its own, disjoint DOF group,
        the same "two different, group-appropriate corrections" strategy
        the class-level comment block's "Key structural fact" already
        established, just with a different (better) rotation extraction
        for the rotation group this time.

        Returns (dof_local, e1, e2, e3) with the SAME shape/convention
        as `_local_relative_dofs()`, so a future caller can swap them
        once the holomorphy question above is resolved."""
        X_ref = np.asarray(elem_coords, dtype=float)
        u = np.asarray(u_elem, dtype=float)
        u_nodes = u.reshape(4, 6)
        u_trans = u_nodes[:, 0:3]
        theta_global = u_nodes[:, 3:6]

        e1_0, e2_0, e3_0, _ = self._linear._local_frame_and_coords(X_ref)
        R0 = np.vstack([e1_0, e2_0, e3_0])
        centroid_ref = X_ref.mean(axis=0)
        Xref_local = (X_ref - centroid_ref) @ R0.T

        mean_theta = theta_global.mean(axis=0)
        theta_z_mean = mean_theta[2]
        R_drill = R0 @ self._exact_drill_rotation(theta_z_mean)
        e1, e2, e3 = R_drill[0], R_drill[1], R_drill[2]

        x_current = X_ref + u_trans
        centroid_current = x_current.mean(axis=0)
        x_current_local = (x_current - centroid_current) @ R_drill.T

        u_local_mem_xy = x_current_local[:, 0:2] - Xref_local[:, 0:2]

        _, r_nodes = self._mean_rigid_rotation(theta_global)

        dof_local = np.zeros(24)
        for i in range(4):
            dof_local[6 * i + 0] = u_local_mem_xy[i, 0]
            dof_local[6 * i + 1] = u_local_mem_xy[i, 1]
            dof_local[6 * i + 5] = r_nodes[i, 2]
            dof_local[6 * i + 2] = u_trans[i] @ e3_0
            dof_local[6 * i + 3] = r_nodes[i, 0]
            dof_local[6 * i + 4] = r_nodes[i, 1]
        return dof_local, e1, e2, e3

    # =================================================================
    # Wave 4 item 46, BUILDING BLOCK B FINDING (2026-09-08,
    # docs/shells.md Section 4.5, "Building block B"): before
    # implementing "re-derive strain as an additive Koiter-Sanders split
    # reusing K_local0" as originally sketched, derived (then confirmed
    # numerically, `tests/test_shell_corotational_exact_rotation_
    # extraction.py::test_membrane_force_provably_unaffected_by_exact_
    # rotation_extraction`) that this specific content for building
    # block B has NO EFFECT WHATSOEVER on membrane force, for a clean,
    # already-established structural reason -- not by trial and error.
    #
    # K_local0 is exactly block-diagonal between the membrane+drilling
    # DOF group and the bending+shear group ("Key structural fact",
    # established when this class was first built). What's newly
    # confirmed here: the drilling-stabilization stiffness is ALSO
    # purely DIAGONAL even WITHIN the membrane+drilling group (checked
    # directly: K_local0[u/v-slots, theta_z-slots] is exactly 0.0, not
    # merely small). Consequence: membrane (u,v) FORCE = K_local0[uv,uv]
    # @ dof_local[uv] ONLY -- completely independent of whatever value
    # feeds the rotational dof slots (theta_x, theta_y, theta_z),
    # regardless of how those values are computed. `_local_relative_
    # dofs_exact_rotation()` above changes theta_z_local's VALUE (a
    # real, expected SO(3)-composition effect: the exact mean rotation's
    # z-component depends on theta_x/theta_y too, unlike the old
    # method's simple arithmetic z-mean -- confirmed non-trivial, not a
    # no-op, in the same test) but this has PROVABLY zero path into u,v
    # force, hence zero effect on foreshortening.
    #
    # This closes off "building block B = feed a better/exact curvature
    # into the EXISTING K_local0, no new coupling term" as having any
    # standalone content, BEFORE spending effort building and
    # benchmarking it against the elastica case -- the same "derive,
    # confirm once analytically-then-numerically, don't guess"
    # discipline item 19's own implicit-function-theorem argument used.
    # It does NOT mean building block B's underlying IDEA (an additive,
    # non-quadratic strain split, avoiding dead ends 4-8's indefinite
    # coupling FORCE) is wrong -- it means that idea's actual content,
    # per Abaqus's own derivation (docs/shell_rotation_coupling_fix_
    # roadmap.md Section 2.4), is inseparable from building block C:
    # Abaqus's own membrane strain (f_ab = t_a . dx/ds_b, a GRADIENT
    # measure) is evaluated using a TANGENT FRAME that is itself
    # committed, incrementally-updated state, re-derived EXACTLY every
    # load/time step -- not a fixed-reference, single-shot quantity the
    # way `_local_relative_dofs()`'s u_local_mem_xy is. A single-shot
    # (Total-Lagrangian-style) evaluation of ANY curvature-aware
    # membrane strain measure against the ORIGINAL reference
    # configuration would need a frame with genuine out-of-plane tilt to
    # do so -- exactly what dead end 3 already found breaks the
    # position-based extraction (a tilted frame's z-row is not
    # [0,0,1], so it re-couples w into x/y). So the two remaining paths
    # are the same two Section 3's diagnosis already named: (a) a
    # genuinely new coupling FORCE term (dead ends 4-8's family,
    # exhausted for "quadratic in absolute rotation"; an UNTRIED
    # non-quadratic form is not ruled out by this argument, but has no
    # obvious candidate construction yet), or (b) building block C's
    # incremental frame commitment, which is now understood to be the
    # ONLY path that can make a GRADIENT-based (not position-difference-
    # based) membrane strain measure well-posed without dead end 3's
    # tilt-leak, since each committed step's frame update is a SMALL,
    # well-conditioned correction relative to its own last-committed
    # state rather than one large tilt measured against a fixed origin.
    # =================================================================

    # =================================================================
    # Wave 4 item 46, BUILDING BLOCK C (2026-09-08, docs/shell_rotation_
    # coupling_fix_roadmap.md Section 4.3): committed, incrementally-
    # updated bending-membrane coupling, via this project's EXISTING
    # init_state()/commit_state()/commit_all_states() extension point
    # (solver.py, "Path-dependent element state" -- the same mechanism
    # Hex8PlasticJ2/Quad4PlasticJ2PlaneStress already use for plasticity
    # history). No solver-side change needed: solve_nonlinear_static()
    # already calls fesystem.commit_all_states() once per CONVERGED load
    # step and is a no-op unless fesystem.init_state() was called first
    # -- this class only supplies init_state()/commit_state() plus a
    # state= branch in internal_force()/tangent_stiffness().
    #
    # FIRST ATTEMPT, REVISED IN PLACE (found and fixed within this same
    # implementation pass, not shipped then reverted): the first design
    # tried generalizing `_local_relative_dofs_exact_rotation()`'s WHOLE
    # dof_local extraction (translation AND rotation) to be relative to
    # a committed baseline, using `_mean_rigid_rotation()`'s general
    # 3-vector frame for TRANSLATION too. Direct testing (before this
    # was wired into any committed test file) immediately reproduced
    # "Design history" dead end 3: even a SMALL, purely-rotational
    # increment (u_trans=0, only theta perturbed) produced spurious
    # nonzero w -- because a general 3-vector frame's z-row deviates
    # from [0,0,1] the moment ANY within-element bending gradient is
    # present, independent of increment size. This is a DIFFERENT
    # mechanism from dead end 3's original large-single-shot-tilt
    # failure, and confirms building block B's own finding from a new
    # angle: `_mean_rigid_rotation()`'s general frame is fundamentally
    # unsafe for TRANSLATION extraction, incremental or not -- it is
    # only safe for the ROTATIONAL dof slots (building block A's own,
    # correctly narrower, scoping). Consequence: this element's
    # TRANSLATION extraction does not need, and must not use, ANY
    # incremental/committed machinery -- `_local_relative_dofs()`'s
    # existing single-shot, drilling-only-frame formula already is
    # exact/well-conditioned at any magnitude (that was never the
    # broken part; building block B already proved swapping the
    # rotation extraction alone has zero effect on membrane force via
    # K_local0's own block-diagonal structure).
    #
    # REVISED DESIGN (what is actually implemented below): the ONLY
    # thing that needs to become committed state is the input to
    # `_bending_membrane_coupling_force()`'s own von Karman coupling --
    # specifically, REPLACING its two existing choices (deviation-from-
    # mean, safe-but-weak; absolute-since-REFERENCE, dead end 4's
    # correct-but-indefinite) with a THIRD: absolute-since-COMMIT. This
    # is dead end 4's own correct physics (the coupling needs the
    # CONFIGURATIONAL, non-deviation slope, not a within-element-only
    # proxy) made safe by measuring it relative to the last-converged
    # state instead of the original reference -- exactly Section 3's
    # diagnosis, now applied surgically to the one piece that actually
    # needed it, rather than the whole dof_local extraction. Each
    # commit's own contribution (a small quadratic strain increment,
    # since the rotation it's built from is small) is accumulated into
    # a per-Gauss-point ADD-ON STRESS RESULTANT, `N_add_history` -- the
    # SAME (Nxx,Nyy,Nxy) DATA STRUCTURE Module 23's `iter_state`
    # mechanism already uses for a different (iteration-level, not
    # step-level) purpose, confirming the "Open risk" section of this
    # project's own roadmap doc ("that existing, validated
    # infrastructure may be directly reusable for STEP-level
    # commitment here").
    #
    # HOLOMORPHY: unlike the reverted first attempt (and unlike building
    # block A's own `_local_relative_dofs_exact_rotation()`), this
    # design uses NO `_mean_rigid_rotation()`/`_exp_map()`/`_log_map()`
    # at all -- `theta_incr = theta - theta_committed` is a plain
    # subtraction (theta complex during complex-step, theta_committed a
    # REAL constant from state), and every downstream quantity
    # (wx_incr, wy_incr, eps_add_incr, N_add_trial, sigma_total, the
    # bending/membrane force contributions) is built from products and
    # sums of that -- a polynomial, hence entire/holomorphic, exactly
    # mirroring why `_bending_membrane_coupling_force()` itself already
    # tolerates complex-step (see that method's own docstring). This
    # means the state=... path CAN reuse complex-step differentiation
    # for its tangent (see `_tangent_stiffness_committed_complex_step()`
    # below) -- no need for `Beam3DCorotational`'s real-FD fallback, and
    # no unresolved holomorphic-log-map blocker to carry forward.
    #
    # SECOND FINDING (2026-09-08, same day, found via direct numerical
    # testing AFTER the plumbing above was implemented and unit-tested,
    # BEFORE claiming this building block "works"): the REVISED DESIGN's
    # accumulation rule is ITSELF mathematically flawed, in a precise,
    # provable way -- not merely "needs a finer load ramp."
    #
    # `_bending_membrane_coupling_force_committed()` accumulates
    # `0.5*(Wx_incr)^2` (a fresh eps_add built from EACH commit's own
    # since-commit increment) into N_add_history at every commit. Direct
    # testing -- committing the SAME final rotation state via 1, 2, 4, 8,
    # then 16 EQUAL sub-steps and comparing the resulting N_add_history
    # -- shows it exactly HALVES every time the step count doubles (5.03e6
    # -> 2.52e6 -> 1.26e6 -> 6.29e5 -> 3.15e5 for a fixed final state),
    # i.e. it converges to ZERO as steps -> infinity, regardless of the
    # (fixed, non-infinitesimal) final rotation. This is the textbook
    # signature of the wrong discretization of an integral: summing
    # `0.5*(dWx)^2` over n equal sub-steps of a linear ramp is
    # `O(Wx_final^2/n)`, NOT an approximation of `0.5*Wx_final^2`. The
    # CORRECT incremental update for that integral is the standard
    # trapezoidal/telescoping one, `d(0.5*Wx^2) = Wx*dWx`, i.e. each
    # commit should add `Wx_committed_before * (Wx_new - Wx_committed_
    # before)` (to leading order) -- NOT `0.5*(Wx_new-Wx_committed_
    # before)^2` alone, which discards exactly that leading, dominant
    # term.
    #
    # Fixing the discretization (using the correct telescoping update)
    # was checked analytically, not just patched and re-tested: because
    # `Wx = theta @ e2_0` uses the FIXED (non-committed, non-rotating)
    # e1_0/e2_0 basis, it is a plain STATE FUNCTION of the CURRENT total
    # theta alone -- `Wx(theta)`, with no path/history dependence. A
    # correctly-telescoping incremental update of ANY state function
    # collapses EXACTLY to evaluating that function once at the current
    # state, independent of how many commits subdivided the path to get
    # there (a basic calculus identity, `integral of d(g) from a to b`
    # = `g(b)-g(a)` regardless of the partition). Concretely: a corrected
    # version of this method would give N_add_trial = Dm @ [0.5*Wx_total^2,
    # 0.5*Wy_total^2, Wx_total*Wy_total] at ANY trial state, EXACTLY
    # `_bending_membrane_coupling_force()`'s own dead-end-4 "ABSOLUTE
    # rotation" branch (class-level "Design history," dead end 4) --
    # already documented there as recovering full elastica-scale
    # foreshortening but making the tangent go locally INDEFINITE at
    # ordinary Newton-iterate states (~1.7 degrees), independent of load-
    # step count, since it depends only on the CURRENT state.
    #
    # CONSEQUENCE: "commit more often" cannot buy safety for a coupling
    # term built from projections onto the FIXED e1_0/e2_0 axes -- being
    # a state function, it has no notion of "small since the last
    # commit" to exploit; commit granularity only changes how badly (not
    # whether) an INCORRECT discretization of it under/over-shoots, which
    # is exactly the erratic, non-convergent behavor observed on the
    # elastica benchmark (this specific, buggy, non-telescoping version:
    # foreshortening ratio 30% at 2 coarse load steps to 0.05, jumping to
    # 179% -- overshooting, wrong direction -- at 4 finer steps to the
    # SAME final load, both against the state=None baseline's ~1.3%).
    # This is a NEW, sharper version of Section 3's own diagnosis: the
    # "committed, incrementally-updated frame" mechanism only supplies
    # genuine conditioning benefit when the quantity being tracked is
    # measured against a frame that ITSELF rotates with each commit (true
    # path/history dependence) -- not merely any quantity chunked into
    # smaller load steps. Building block C's FIRST attempt (reverted
    # above) tried exactly that (a genuinely rotating frame) for
    # TRANSLATION and reproduced dead end 3; this SECOND attempt avoided
    # that by deliberately using the FIXED e1_0/e2_0 axes for the
    # coupling term instead, which is exactly why it cannot escape dead
    # end 4's own conclusion.
    #
    # STATUS: the state machinery, holomorphy, and complex-step tangent
    # below are all independently correct and tested (tests/test_shell_
    # corotational_committed_coupling.py) and are SAFE to keep (state=
    # None -- the only path every pre-existing test exercises -- is
    # completely untouched, verified byte-identical). The COUPLING
    # FORMULA itself is NOT validated as a fix and should not be treated
    # as one -- kept, opt-in and dormant, as documented infrastructure
    # for whoever attempts the one path this now clearly rules IN: a
    # coupling term measured against a frame that is ITSELF committed/
    # incrementally rotated (not the fixed e1_0/e2_0 used here), reusing
    # this state machinery's plumbing but replacing `_bending_membrane_
    # coupling_force_committed()`'s projection axes with a genuinely
    # rotating pair -- the same "keep, opt-in, dormant, documented
    # starting point" precedent `init_iter_state()`'s own docstring
    # already established for dead end 7's own unresolved paths.
    # =================================================================

    def init_state(self):
        """Wave 4 item 46, building block C: the "nothing committed
        yet" baseline -- zero accumulated add-on stress, committed
        rotation equal to the u=0 reference. Passing this state into
        internal_force()/tangent_stiffness() at u_elem=0 gives exactly
        zero force/the u=0 tangent (checked directly,
        tests/test_shell_corotational_committed_coupling.py), the same
        "state=None and state=init_state() agree at the reference
        point" sanity property every stateful element in this package
        satisfies trivially."""
        n_gauss = len(gauss_product(self.gauss_order, self.dim)[0])
        return {
            "N_add_history": np.zeros((n_gauss, 3)),
            "theta_committed": np.zeros((4, 3)),
        }

    def _bending_membrane_coupling_force_committed(self, elem_coords, dof_local, D,
                                                     theta, theta_committed, N_add_history):
        """Wave 4 item 46, building block C: NOT VALIDATED AS A FIX --
        see the class-level "BUILDING BLOCK C" comment's "SECOND
        FINDING" paragraph before using or extending this. `wx_incr`/
        `wy_incr` are the SINCE-COMMIT increment of a quantity (`theta @
        e1_0`/`e2_0`) projected onto FIXED, non-rotating axes, so it is
        a plain state function of the current total theta with no true
        path dependence -- accumulating `0.5*(wx_incr)^2` per commit is
        a specific, PROVEN-WRONG discretization of the resulting
        integral (converges to ZERO as commit count -> infinity for any
        fixed final state, confirmed directly: N_add_history exactly
        halves every time step count doubles at a fixed target rotation
        -- not an approximation error that shrinks with finer stepping,
        the opposite). Kept, opt-in and dormant, as documented
        infrastructure only -- see the class-level comment for what a
        corrected design would need (a genuinely rotating, not fixed,
        projection axis).

        an "absolute-since-commit" variant of
        `_bending_membrane_coupling_force()`'s von Karman coupling,
        with the resulting add-on membrane stress ACCUMULATED across
        commits (via N_add_history, updated by `commit_state()` below)
        rather than recomputed fresh from a fixed reference every call.

        wx_incr(a)/wy_incr(a) = the SAME betax/betay proxy
        `_bending_membrane_coupling_force()` already uses
        (theta_y_raw/-theta_x_raw, i.e. theta[a]@e2_0 / -theta[a]@e1_0),
        but of `theta - theta_committed` (rotation SINCE the last
        commit) instead of the total theta -- dead end 4's own
        ABSOLUTE (not deviation-from-mean) rotation, made safe because
        "absolute since commit" stays small whenever commits happen
        often enough, unlike "absolute since reference" which grows
        with the whole load history and was dead end 4's actual failure
        mode (tangent indefiniteness at ordinary Newton-iterate states,
        not an isolated edge case -- see class-level "Design history").
        No mean subtraction (Nb_offset=0 throughout, matching
        `_bending_membrane_coupling_force()`'s own N_add-is-not-None
        branch) -- deviation-from-mean is deliberately NOT reused here,
        since the whole point is to recover the CONFIGURATIONAL
        (non-deviation) coupling dead end 4 correctly identified as
        missing.

        At each Gauss point, N_add_trial = N_add_history (every PRIOR
        commit's own accumulated add-on stress) + Dm @ eps_add_incr
        (THIS call's own fresh since-commit contribution) -- i.e. the
        TOTAL add-on membrane stress if this trial state were committed
        right now. sigma_total = Dm@eps_lin + N_add_trial (linear
        membrane stress, from the UNCHANGED/existing dof_local[_MEM]
        extraction, plus the total add-on) drives the bending-side
        reaction force (Wx*sigma_total[0]+Wy*sigma_total[2] etc, the
        same von Karman energy-conjugate structure
        `_bending_membrane_coupling_force()` already uses). The
        MEMBRANE force contribution is Bm.T @ N_add_trial (the TOTAL
        add-on stress, not just this call's fresh increment -- unlike
        the non-accumulating original method, which only ever needs its
        own fresh eps_add since it has nothing to accumulate against);
        eps_lin's own linear contribution is already carried by
        K_local0 @ dof_local, so only the add-on part belongs here,
        exactly the same "avoid double-counting" reasoning
        `_bending_membrane_coupling_force()`'s own docstring uses.

        Returns (f_local_add (24,), N_add_trial (n_gauss,3)) -- the
        second return value is what `commit_state()` below permanently
        writes into N_add_history when this call's u_elem is the
        converged state of a load step."""
        Dm, Db, Ds, h = D
        X_ref = np.asarray(elem_coords, dtype=float)
        _, _, _, local = self._linear._local_frame_and_coords(X_ref)
        e1_0, e2_0, e3_0, _ = self._linear._local_frame_and_coords(X_ref)

        dof_local = np.asarray(dof_local)
        dtype = dof_local.dtype
        theta = np.asarray(theta)
        theta_committed = np.asarray(theta_committed, dtype=float)
        theta_incr = theta - theta_committed          # (4,3), holomorphic in theta
        wx_incr = theta_incr @ e2_0                   # betax_incr(a) = theta_y_incr(a)
        wy_incr = -(theta_incr @ e1_0)                 # betay_incr(a) = -theta_x_incr(a)
        dof_mem = dof_local[_MEM]
        pts, wts = gauss_product(self.gauss_order, self.dim)
        membrane = self._linear._membrane
        f_local_add = np.zeros(24, dtype=dtype)
        N_add_trial = np.zeros((len(wts), 3), dtype=dtype)

        for gp_idx, (p, wgt) in enumerate(zip(pts, wts)):
            N, _ = membrane.shape_and_derivs(p)
            Bm, detJ = membrane.B_matrix(p, local)
            Wx = N @ wx_incr
            Wy = N @ wy_incr
            eps_add_incr = np.array([0.5 * Wx * Wx, 0.5 * Wy * Wy, Wx * Wy], dtype=dtype)
            Nadd_trial_gp = N_add_history[gp_idx] + Dm @ eps_add_incr
            N_add_trial[gp_idx] = Nadd_trial_gp
            eps_lin = Bm @ dof_mem
            sigma_total = Dm @ eps_lin + Nadd_trial_gp

            f_mem_add = Bm.T @ Nadd_trial_gp * detJ * wgt * h
            for i, idx in enumerate(_MEM):
                f_local_add[idx] += f_mem_add[i]
            for b in range(4):
                Nb = N[b]
                f_local_add[6 * b + 4] += h * detJ * wgt * Nb * (Wx * sigma_total[0] + Wy * sigma_total[2])
                f_local_add[6 * b + 3] += -h * detJ * wgt * Nb * (Wy * sigma_total[1] + Wx * sigma_total[2])
        return f_local_add, N_add_trial

    def _internal_force_committed(self, elem_coords, u_elem, D, state, thickness=1.0, iter_state=None):
        """Wave 4 item 46, building block C: the state-aware trial
        internal force. dof_local (translation AND rotation) comes from
        the EXISTING, unmodified, single-shot `_local_relative_dofs()`
        -- see the class-level comment block above for why translation
        must NOT be re-derived through any committed/incremental frame.
        f_local = K_local0 @ dof_local + the ORIGINAL deviation-based
        coupling force (kept, additive, unaffected by any of this) + the
        NEW committed coupling force (this building block's own
        contribution). "Trial" because this does NOT mutate state --
        the established convention `solver.py`'s "Path-dependent
        element state" comment documents (a trial response from the
        last-COMMITTED state plus the current, possibly not-yet-
        equilibrated u_elem; only commit_state() permanently advances
        anything)."""
        u = np.asarray(u_elem)
        u_nodes = u.reshape(4, 6)
        theta = u_nodes[:, 3:6]
        K_local0 = self._local_material_stiffness(elem_coords, D)
        dof_local, e1, e2, e3 = self._local_relative_dofs(elem_coords, u_elem)
        f_local_orig = K_local0 @ dof_local + self._bending_membrane_coupling_force(
            elem_coords, dof_local, D, N_add=iter_state)
        f_local_committed, _ = self._bending_membrane_coupling_force_committed(
            elem_coords, dof_local, D, theta, state["theta_committed"], state["N_add_history"])
        T = self._rotation_matrix_c(e1, e2, e3)
        return T.T @ (f_local_orig + f_local_committed)

    def commit_state(self, elem_coords, u_elem, mat, state, thickness=1.0, **kwargs):
        """Wave 4 item 46, building block C: advance the committed
        baseline to u_elem -- call once per CONVERGED load step, never
        mid-Newton-iteration (same contract as Hex8PlasticJ2.commit_
        state() et al.). `mat` plays the same role `D` plays in
        internal_force()/tangent_stiffness() (solver.py's
        commit_all_states() calls every formulation positionally the
        same way regardless of element type).

        Computes this step's own trial N_add (via
        `_bending_membrane_coupling_force_committed()`, at the just-
        converged u_elem, against the OLD committed baseline) and writes
        it PERMANENTLY into N_add_history, then advances
        theta_committed to u_elem's own rotation -- so the NEXT call's
        "since commit" increment starts back at zero, small by
        construction, and this step's own contribution is now baked
        into N_add_history for every future call to build on."""
        u = np.asarray(u_elem, dtype=float)
        u_nodes = u.reshape(4, 6)
        theta = u_nodes[:, 3:6]
        dof_local, _, _, _ = self._local_relative_dofs(elem_coords, u_elem)
        _, N_add_trial = self._bending_membrane_coupling_force_committed(
            elem_coords, dof_local, mat, theta, state["theta_committed"], state["N_add_history"])
        return {
            "N_add_history": np.asarray(N_add_trial, dtype=float),
            "theta_committed": theta.copy(),
        }

    def _bending_membrane_coupling_force(self, elem_coords, dof_local, D, N_add=None):
        """Phase B (docs/shells.md Section 4.2, "closing
        this residual... giving the bending block its own position-
        tracked extraction, symmetric to what the membrane block
        already has"): the ADDITIONAL local force, on top of
        K_local0 @ dof_local, from genuine bending-induced membrane
        strain -- the physics this module's "Phase C finding" comment
        documents as EXACTLY missing (K_local0's membrane and bending
        blocks are exactly decoupled; a bent cantilever predicts zero
        axial foreshortening at any rotation).

        Standard von Karman moderate-rotation strain-displacement
        relation (e.g. Reddy, "Mechanics of Laminated Composite Plates
        and Shells," Ch. 3): membrane strain gets a quadratic
        correction from the transverse slope,
            eps_xx += 0.5*wx^2, eps_yy += 0.5*wy^2, gamma_xy += wx*wy,
        with (wx, wy) = (dw/dx, dw/dy) approximated by this element's
        own (betax, betay) rotation DOFs -- the same Kirchhoff-limit
        slope proxy Db/Bb already use to relate betax/betay to bending
        curvature (gamma_xz = dw/dx - betax ~ 0 etc., see the module's
        _BEND_SIGN comment for the betax=theta_y, betay=-theta_x
        convention this reuses directly).

        Critical departure from a naive von Karman implementation: wx,
        wy are evaluated as each node's DEVIATION from the element's
        own MEAN (betax, betay) -- e.g. wx_dev(a) = betax(a) -
        mean(betax) -- mirroring EXACTLY how the membrane block's own
        drilling extraction already subtracts theta_z_mean (see
        _local_relative_dofs()'s docstring). A uniform RIGID TILT of
        the whole flat facet gives the SAME betax/betay at all 4 nodes,
        so the deviation -- and hence this entire coupling force -- is
        EXACTLY zero for a rigid tilt, matching K_local0's own bending
        block (already a near-exact null space for rigid tilt, see the
        class-level comment's item 3(a)); only genuine WITHIN-ELEMENT
        curvature (nodes disagreeing on betax/betay) triggers the
        coupling. This is what avoids reproducing dead ends 1-3's
        failure mode (a uniform frame update leaking a node's ABSOLUTE
        translation/rotation into the membrane block) -- here the
        membrane coupling is driven by a quantity that is analytically
        zero whenever the bending state is a pure rigid tilt, not merely
        small.

        REJECTED ALTERNATIVE, dead end 4 (2026-09-03, see class-level
        "ABSOLUTE-ROTATION ATTEMPT" comment below for the full account):
        tried wx, wy = each node's ABSOLUTE (betax, betay) instead of the
        deviation above, on the correct reasoning that von Karman strain
        is a CONFIGURATIONAL quantity (depends on current-vs-original
        shape alone, not load history) that deviation-from-mean under-
        counts for a smoothly-curving structure. This DID recover
        elastica-scale foreshortening magnitude, but made the element's
        own tangent stiffness go locally INDEFINITE (negative eigenvalue,
        confirmed via exact complex-step match -- not a derivative bug)
        at ordinary, non-exotic trial states (~1.7 degrees local
        rotation) that arise routinely as intermediate Newton iterates on
        a real assembled, boundary-constrained mesh -- not just the
        isolated-single-element edge case dead ends 1-3 already
        documented. Confirmed across three different solver strategies
        (plain Newton, transient with line search + trust region + mass
        regularization, arc-length continuation designed specifically to
        trace through indefinite/limit-point regions) -- none recovered
        a trustworthy, converged, properly-mesh-resolved result. This is
        a genuine structural property of adding Dm*wx^2-scale coupling
        (Dm being ~1e6-1e7x the bending stiffness this element is
        normally governed by) directly on top of K_local0, not a solver-
        tuning problem -- closing it would need a redesigned coupling
        term (e.g. a saturating/reduced-stiffness dependence on rotation
        instead of raw quadratic-in-absolute-rotation), not attempted.

        Energy-consistent (conservative) construction: this equals
        d(U_add)/d(dof_local), where U_add = integral(sigma_lin . eps_add
        + 0.5*eps_add . Dm . eps_add) h dA is the incremental membrane
        strain energy from eps_add alone, given the ALREADY-computed
        linear membrane strain eps_lin = Bm @ dof_local[_MEM]. Since
        U_add's dependence on dof_local is at least quadratic (eps_add
        itself is quadratic in the bending DOFs), this force -- and its
        own derivative -- are EXACTLY zero at dof_local=0, so
        Shell4MITCCorotational.stiffness() (the u=0 tangent) is
        unaffected by this addition (still matches Shell4MITC.stiffness()
        to the same ~1e-13 relative precision as before).

        No abs()/branching/norm on any dof_local-derived quantity (only
        means, differences, and products), so this stays holomorphic in
        dof_local -- safe for complex-step differentiation (see
        _coupling_tangent_local() below).

        MIXED-FORMULATION MODE (Module 23, added to actually close the
        D5 gap this docstring's "REJECTED ALTERNATIVE, dead end 4"
        paragraph documents): pass N_add (the (n_gauss, 3) per-Gauss-
        point add-on stress resultant tracked in this element's
        iter_state -- see init_iter_state()/update_iter_state() below)
        to switch from the default deviation-from-mean rotation to the
        physically-correct ABSOLUTE rotation (dead end 4's own
        reasoning was right: von Karman strain is CONFIGURATIONAL, and
        deviation-from-mean under-counts it for a smoothly-curving
        multi-element structure), while keeping Newton ROBUST by
        computing the coupling force from N_add directly (a tracked,
        incrementally-corrected quantity) rather than from Dm@eps_add
        recomputed fresh from the current displacement every call --
        exactly the mechanism validated in NonLin-HyROM/pilot_mixed_
        formulation/pilot_lagged_condensation.py before implementing it
        here for real. N_add=None (the default) reproduces the ORIGINAL
        formula bit-for-bit -- this is a strict backward-compatible
        addition, not a replacement; the 165+ existing tests that never
        call fesystem.init_iter_state() are completely unaffected.

        IMPORTANT CORRECTION (2026-09-04, found while validating against
        the elastica benchmark): the ACTUAL FORCE returned here, even in
        mixed mode, always uses the EXACT, freshly-computed Dm@eps_add
        (never N_add's own tracked value) -- only the rotation measure
        (absolute vs. deviation-from-mean) changes with mode. This is
        not an oversight; it is REQUIRED for correctness. A full,
        properly-derived static condensation of the joint
        (dof_local, N_add) Newton system (see update_iter_state()'s own
        docstring) shows the CONDENSED RESIDUAL is algebraically
        IDENTICAL to evaluating the force with N_add replaced by
        Dm@eps_add directly, regardless of N_add's current (generally
        inconsistent, mid-iteration) value -- i.e. correctness of the
        force/residual cannot depend on how well-converged N_add
        happens to be. The actual benefit of tracking N_add lives
        entirely in the TANGENT: _coupling_tangent_local() below
        evaluates K_dd against a SEPARATE function,
        _bending_membrane_coupling_force_for_tangent(), which DOES use
        N_add's own (lagged, deliberately smoothed) value in place of
        Dm@eps_add -- a modified/quasi-Newton Jacobian for this exact
        residual, not a different residual. Confirmed necessary by
        direct experiment: using N_add in the FORCE itself (an earlier
        version of this method) produced a floating-point overflow
        within the first load step on the real elastica benchmark mesh."""
        Dm, Db, Ds, h = D
        X_ref = np.asarray(elem_coords, dtype=float)
        _, _, _, local = self._linear._local_frame_and_coords(X_ref)

        dof_local = np.asarray(dof_local)
        dtype = dof_local.dtype
        wx = np.array([dof_local[6 * a + 4] for a in range(4)])       # betax(a) = theta_y_raw(a)
        wy = np.array([-dof_local[6 * a + 3] for a in range(4)])      # betay(a) = -theta_x_raw(a)
        dof_mem = dof_local[_MEM]
        pts, wts = gauss_product(self.gauss_order, self.dim)
        membrane = self._linear._membrane
        f_local_add = np.zeros(24, dtype=dtype)

        if N_add is None:
            wx_use = wx - wx.sum() / 4.0
            wy_use = wy - wy.sum() / 4.0
            Nb_offset = -0.25
        else:
            wx_use = wx           # ABSOLUTE rotation: no mean subtraction
            wy_use = wy
            Nb_offset = 0.0

        for p, wgt in zip(pts, wts):
            N, _ = membrane.shape_and_derivs(p)
            Bm, detJ = membrane.B_matrix(p, local)
            Wx = N @ wx_use
            Wy = N @ wy_use
            eps_add = np.array([0.5 * Wx * Wx, 0.5 * Wy * Wy, Wx * Wy], dtype=dtype)
            eps_lin = Bm @ dof_mem
            sigma_total = Dm @ (eps_lin + eps_add)

            f_mem_add = Bm.T @ (Dm @ eps_add) * detJ * wgt * h
            for i, idx in enumerate(_MEM):
                f_local_add[idx] += f_mem_add[i]
            for b in range(4):
                Nb_c = N[b] + Nb_offset
                f_local_add[6 * b + 4] += h * detJ * wgt * Nb_c * (Wx * sigma_total[0] + Wy * sigma_total[2])
                f_local_add[6 * b + 3] += -h * detJ * wgt * Nb_c * (Wy * sigma_total[1] + Wx * sigma_total[2])
        return f_local_add

    def _bending_membrane_coupling_force_for_tangent(self, elem_coords, dof_local, D, N_add):
        """Module 23: NOT the real force (see _bending_membrane_coupling_
        force()'s own "IMPORTANT CORRECTION" note) -- a separate function
        used ONLY to build the mixed formulation's TANGENT, evaluating
        the bending-side stress as sigma_lin + N_add (N_add's OWN
        tracked, lagged value) instead of the exact sigma_lin+Dm@eps_add,
        and the membrane force as Bm.T@N_add instead of Bm.T@Dm@eps_add.
        This is exactly the Hellinger-Reissner-style "add-on stress as
        an independent unknown" construction, complex-stepped by
        _coupling_tangent_local() to give K_dd -- the deliberately
        SMOOTHED Jacobian _coupling_schur_correction() then completes
        into the full condensed tangent. Always uses ABSOLUTE rotation
        (this function is only ever called when N_add is provided, i.e.
        mixed mode is active)."""
        Dm, Db, Ds, h = D
        X_ref = np.asarray(elem_coords, dtype=float)
        _, _, _, local = self._linear._local_frame_and_coords(X_ref)
        dof_local = np.asarray(dof_local)
        dtype = dof_local.dtype
        wx = np.array([dof_local[6 * a + 4] for a in range(4)])
        wy = np.array([-dof_local[6 * a + 3] for a in range(4)])
        dof_mem = dof_local[_MEM]
        pts, wts = gauss_product(self.gauss_order, self.dim)
        membrane = self._linear._membrane
        f_local_add = np.zeros(24, dtype=dtype)

        for p_idx, (p, wgt) in enumerate(zip(pts, wts)):
            N, _ = membrane.shape_and_derivs(p)
            Bm, detJ = membrane.B_matrix(p, local)
            Wx = N @ wx
            Wy = N @ wy
            eps_lin = Bm @ dof_mem
            sigma_lin = Dm @ eps_lin
            Nadd_p = N_add[p_idx]
            sigma_total_mixed = sigma_lin + Nadd_p

            f_mem_add = Bm.T @ Nadd_p * detJ * wgt * h
            for i, idx in enumerate(_MEM):
                f_local_add[idx] += f_mem_add[i]
            for b in range(4):
                Nb_c = N[b]   # ABSOLUTE rotation: no "-0.25" mean subtraction
                f_local_add[6 * b + 4] += h * detJ * wgt * Nb_c * (Wx * sigma_total_mixed[0] + Wy * sigma_total_mixed[2])
                f_local_add[6 * b + 3] += -h * detJ * wgt * Nb_c * (Wy * sigma_total_mixed[1] + Wx * sigma_total_mixed[2])
        return f_local_add

    def init_iter_state(self):
        """Module 23: one (Nxx, Nyy, Nxy) add-on membrane-stress
        resultant per Gauss point, initialized to zero -- the mixed-
        formulation internal unknown this element's own coupling
        machinery uses when iter_state is provided. Opt-in only: a
        solve that never calls fesystem.init_iter_state() never sees
        this (default behavior, what actually ships, is unaffected).

        DO NOT call this expecting a working nonlinear solve today.
        This class's own "Design history" comment, "DEAD END 7,"
        documents why: the lagged (one-call-behind) N_add update this
        mechanism relies on is confirmed -- at both a standalone toy
        and this real element -- to diverge (a real, geometric
        residual blowup, not slow convergence) once N_add must move
        substantially from its zero starting value, which happens on
        essentially the very first Newton correction of any real solve.
        Every analytic piece involved (the Schur-complement tangent
        correction, the compatibility residual) has been independently
        verified correct via complex-step -- this is a genuine
        limitation of the LAGGED update strategy itself, not a bug
        waiting to be found. Kept, opt-in and dormant, as the
        documented starting point for whoever attempts the two open
        paths that same comment block describes (sub-iterating N_add to
        real convergence within a step, or giving it true joint-DOF
        status) -- not because it currently works."""
        n_gauss = len(gauss_product(self.gauss_order, self.dim)[0])
        return np.zeros((n_gauss, 3))

    def update_iter_state(self, elem_coords, u_elem, delta_u_elem, D, iter_state, **kwargs):
        """Module 23: advance each Gauss point's N_add by ONLY its own
        share of a joint-linearized Newton step, using the REALIZED
        correction delta_u_elem this Newton iteration actually took --
        NOT by re-solving N_add exactly from the current displacement
        (verified, both algebraically and by direct experiment in
        NonLin-HyROM/pilot_mixed_formulation/, to collapse back to the
        original ill-conditioned formula with zero benefit whenever
        tried). The compatibility residual this drives toward zero is

            R_N(dof_local, N_add) = eps_add(dof_local) - Dm^-1 @ N_add

        (stationary at N_add = Dm @ eps_add, exactly reconstructing the
        non-mixed formula's own coupling stress at self-consistency).
        Linearizing R_N=0 around the state THIS element was last called
        with (dof_local_old, N_add) and solving for the correction
        consistent with the REALIZED delta_dof_local gives

            dN = Dm @ (R_N_old + K_Nd_old @ delta_dof_local)

        (K_NN = d(R_N)/d(N_add) = -Dm^-1, so -K_NN^-1 = Dm). K_Nd_old =
        d(eps_add)/d(dof_local) is available in closed form (eps_add is
        an explicit quadratic in the bending-rotation dof slots alone,
        exactly the same Wx/Wy this class's own coupling-FORCE formula
        already differentiates by hand) -- no finite difference needed,
        and it is exactly zero outside those 8 columns (eps_add does
        not depend on the other 16 dof_local entries at all).

        delta_dof_local is NOT delta_u_elem transformed by a FIXED
        matrix -- dof_local is a genuinely nonlinear function of
        u_elem (the corotational frame itself moves), so this
        recomputes dof_local at u_elem AND at u_elem-delta_u_elem
        (the "old" state) and takes their exact difference, avoiding a
        second, unnecessary linearization on top of the one already
        inherent in lagging N_add by one call."""
        Dm, Db, Ds, h = D
        if iter_state is None:
            iter_state = self.init_iter_state()
        u_elem = np.asarray(u_elem, dtype=float)
        delta_u_elem = np.asarray(delta_u_elem, dtype=float)
        u_old = u_elem - delta_u_elem

        dof_local_new, _, _, _ = self._local_relative_dofs(elem_coords, u_elem)
        dof_local_old, _, _, _ = self._local_relative_dofs(elem_coords, u_old)
        delta_dof_local = dof_local_new - dof_local_old

        wx_old = np.array([dof_local_old[6 * a + 4] for a in range(4)])
        wy_old = np.array([-dof_local_old[6 * a + 3] for a in range(4)])

        X_ref = np.asarray(elem_coords, dtype=float)
        _, _, _, local = self._linear._local_frame_and_coords(X_ref)
        pts, wts = gauss_product(self.gauss_order, self.dim)
        membrane = self._linear._membrane
        Dm_inv = np.linalg.inv(Dm)

        N_add_new = iter_state.copy()
        for p_idx, p in enumerate(pts):
            N, _ = membrane.shape_and_derivs(p)
            Wx_old = N @ wx_old
            Wy_old = N @ wy_old
            eps_add_old = np.array([0.5 * Wx_old * Wx_old, 0.5 * Wy_old * Wy_old, Wx_old * Wy_old])
            R_N_old = eps_add_old - Dm_inv @ iter_state[p_idx]

            K_Nd_old = np.zeros((3, 24))
            for a in range(4):
                # d(eps_add)/d(wx(a)) = [Wx*N[a], 0, Wy*N[a]], wx(a) lives
                # at dof_local[6a+4]; d(eps_add)/d(wy(a)) = [0, Wy*N[a],
                # Wx*N[a]], but wy(a) = -dof_local[6a+3] (sign flip) --
                # exactly the same chain rule this class's own coupling-
                # FORCE formula already uses, just read off directly
                # instead of complex-stepped, since it is this simple.
                K_Nd_old[:, 6 * a + 4] = [Wx_old * N[a], 0.0, Wy_old * N[a]]
                K_Nd_old[:, 6 * a + 3] = [0.0, -Wy_old * N[a], -Wx_old * N[a]]

            dN = Dm @ (R_N_old + K_Nd_old @ delta_dof_local)
            N_add_new[p_idx] = iter_state[p_idx] + dN

        return N_add_new

    def _coupling_schur_correction(self, elem_coords, dof_local, D):
        """Module 23: K_dN @ K_NN^-1 @ K_Nd, the piece of the mixed
        formulation's condensed tangent that _coupling_tangent_local()
        ALONE (a PARTIAL derivative of the coupling force at FIXED
        N_add) does not include -- N_add's own governing equation
        R_N = eps_add(dof_local) - Dm^-1 @ N_add = 0 implicitly makes
        N_add a function of dof_local too, and this term is exactly
        that implicit sensitivity's contribution to d(f_add)/d(dof_local),
        via the standard exact static-condensation (Schur complement)
        formula for a linearized (dof_local, N_add) system -- NOT an
        approximation, algebraically identical to solving that joint
        system directly (the mechanism validated in NonLin-HyROM/
        pilot_mixed_formulation/pilot_mixed_chain.py before implementing
        it here). Omitting this term is not a smaller/cheaper
        approximation -- confirmed directly (2026-09-04): without it,
        the outer Newton solve's own displacement correction is
        computed from an incomplete tangent while N_add is corrected
        separately in update_iter_state(), and the two disagree enough
        to overflow within the first load step on a real multi-element
        mesh.

        K_NN = -Dm^-1 (POINTWISE/unweighted, matching the constitutive-
        law convention update_iter_state() also uses -- no detJ*wgt*h
        quadrature weight belongs on a stress-strain relation itself),
        so K_NN^-1 = -Dm. K_dN, K_Nd are read off the same explicit
        eps_add derivatives update_iter_state() already uses (no
        complex-step needed, eps_add is an explicit quadratic in the
        bending-rotation dof slots) -- K_dN carries the SAME detJ*wgt*h
        quadrature weight the coupling FORCE formula does (K_dN is a
        force-per-N_add sensitivity, i.e. itself an integrated
        quantity); NOT symmetric to K_Nd in the membrane block (the
        membrane force depends on N_add linearly, K_dN[mem,:]=Bm.T*
        detJ*wgt*h, but N_add's own residual does not depend on the
        membrane dofs at all, K_Nd[:,mem]=0) -- a genuine, checked
        property of this specific (non-conservative) mixed
        reformulation, not a bug: Newton's tangent does not need to be
        symmetric to be correct, only to be the actual Jacobian of the
        residuals actually being driven to zero."""
        Dm, Db, Ds, h = D
        X_ref = np.asarray(elem_coords, dtype=float)
        _, _, _, local = self._linear._local_frame_and_coords(X_ref)
        dof_local = np.asarray(dof_local, dtype=float)
        wx = np.array([dof_local[6 * a + 4] for a in range(4)])
        wy = np.array([-dof_local[6 * a + 3] for a in range(4)])
        pts, wts = gauss_product(self.gauss_order, self.dim)
        membrane = self._linear._membrane
        K_NN_inv = -Dm   # per Gauss point, pointwise/unweighted, constant

        K_corr = np.zeros((24, 24))
        for p, wgt in zip(pts, wts):
            N, _ = membrane.shape_and_derivs(p)
            Bm, detJ = membrane.B_matrix(p, local)
            Wx = N @ wx
            Wy = N @ wy

            K_dN = np.zeros((24, 3))
            for i, idx in enumerate(_MEM):
                K_dN[idx, :] = Bm[:, i]   # Bm is (3, n_mem); Bm.T's i-th row = Bm[:, i]
            K_Nd = np.zeros((3, 24))
            for a in range(4):
                K_dN[6 * a + 4, :] = [Wx * N[a], 0.0, Wy * N[a]]
                K_dN[6 * a + 3, :] = [0.0, -Wy * N[a], -Wx * N[a]]
                K_Nd[:, 6 * a + 4] = [Wx * N[a], 0.0, Wy * N[a]]
                K_Nd[:, 6 * a + 3] = [0.0, -Wy * N[a], -Wx * N[a]]

            K_corr += (K_dN @ K_NN_inv @ K_Nd) * detJ * wgt * h
        return K_corr

    def _coupling_tangent_local(self, elem_coords, dof_local, D, N_add=None, h=1e-20):
        """Complex-step differentiation w.r.t. dof_local ALONE (not the
        full internal_force() chain) -- both correct (the underlying
        force function is holomorphic) and cheaper than a naive full
        complex-step of internal_force(): K_local0 and J are already
        exact/analytic and completely unchanged by this new term (see
        tangent_stiffness()), so only this genuinely NEW, nonlinear
        piece needs a numerical derivative, and it is a much cheaper
        function (a single 2x2 Gauss loop over the membrane/bending
        DOFs only) than the full local material stiffness this avoids
        recomputing 24 times.

        N_add=None (default mode): differentiates the REAL force,
        _bending_membrane_coupling_force() itself (deviation-from-mean
        rotation, unchanged from before Module 23).

        N_add provided (mixed mode): differentiates
        _bending_membrane_coupling_force_for_tangent() instead -- NOT
        the real force (see that method's and the real force method's
        own docstrings for why the real force must always use the
        exact, fresh Dm@eps_add, never N_add's own value, for
        correctness) -- N_add is held FIXED across every dof_local
        perturbation, giving the "K_dd" partial derivative
        _coupling_schur_correction() then completes into the full
        condensed/smoothed tangent."""
        n = len(dof_local)
        dof_c = np.asarray(dof_local, dtype=complex)
        K = np.zeros((n, n))
        force_fn = ((lambda dl: self._bending_membrane_coupling_force_for_tangent(elem_coords, dl, D, N_add))
                    if N_add is not None else
                    (lambda dl: self._bending_membrane_coupling_force(elem_coords, dl, D)))
        for j in range(n):
            dd = np.zeros(n, dtype=complex)
            dd[j] = 1j * h
            f_pert = force_fn(dof_c + dd)
            K[:, j] = f_pert.imag / h
        return K

    def internal_force(self, elem_coords, u_elem, D, thickness=1.0, iter_state=None, state=None, **kwargs):
        """f_int = T(u_elem).T @ (K_local0 @ dof_local(u_elem) +
        _bending_membrane_coupling_force(...)) -- see this module's
        class-level comment block (steps 2-5), _local_relative_dofs()'s
        own docstring, and _bending_membrane_coupling_force()'s own
        docstring for the added bending-induced membrane term. Every
        operation here is algebraic (matrix products, the polynomial
        _skew(), a mean, and a difference of positions) with no
        abs()/branching/norm on any u_elem-derived quantity, so this
        stays holomorphic in u_elem end to end -- required for
        _tangent_stiffness_complex_step()'s complex-step differentiation
        below, mirroring Tet10SolidTL.internal_force()'s own requirement
        (see that method's docstring). iter_state (Module 23), when
        FESystem.init_iter_state() has been called, is this element's
        current (Nxx,Nyy,Nxy)-per-Gauss-point mixed-formulation internal
        state -- passed straight through to _bending_membrane_coupling_
        force() as N_add; None (the default) reproduces the original
        deviation-from-mean formula exactly.

        state (Wave 4 item 46, building block C -- docs/shell_rotation_
        coupling_fix_roadmap.md Section 4.3), when FESystem.init_state()
        has been called, is this element's last-COMMITTED frame/
        accumulated-local-dof state (see init_state()/commit_state()).
        None (the default -- the ONLY path every one of the 165+
        existing tests exercises) reproduces the original single-shot,
        total-Lagrangian-style formula EXACTLY, unchanged, byte-for-
        byte: this branch is purely ADDITIVE, not a modification of the
        default path. When state is provided, this delegates to
        `_internal_force_committed()` instead, which keeps translation/
        rotation extraction on the SAME holomorphic `_local_relative_
        dofs()` this default path uses and adds a committed-state
        coupling force built entirely from polynomial operations on
        theta and real state constants (see `_bending_membrane_
        coupling_force_committed()`'s own docstring) -- so, unlike an
        earlier reverted design, this state=... path stays holomorphic
        too and `tangent_stiffness()`'s own state branch reuses complex-
        step (`_tangent_stiffness_committed_complex_step()`), not real
        finite-difference."""
        if state is not None:
            return self._internal_force_committed(elem_coords, u_elem, D, state,
                                                    thickness=thickness, iter_state=iter_state)
        K_local0 = self._local_material_stiffness(elem_coords, D)
        dof_local, e1, e2, e3 = self._local_relative_dofs(elem_coords, u_elem)
        f_local = K_local0 @ dof_local + self._bending_membrane_coupling_force(
            elem_coords, dof_local, D, N_add=iter_state)
        T = self._rotation_matrix_c(e1, e2, e3)
        return T.T @ f_local

    def _dof_local_jacobian_and_geo(self, elem_coords, u_elem, K_local0, D, iter_state=None):
        """Returns (J, K_geo, R_drill, f_local): the two pieces Phase B's
        analytic tangent_stiffness() needs beyond what internal_force()
        already computes. f_local here is the TOTAL local force
        (K_local0 @ dof_local PLUS _bending_membrane_coupling_force()),
        matching internal_force()'s own total -- K_geo's own derivation
        (below) depends on whatever the CURRENT total local force is,
        regardless of which term produced it.

        J (24x24) = d(dof_local)/d(u_elem), hand-derived block-by-block
        from _local_relative_dofs()'s own construction (this module's
        class-level comment block, steps 2-3):
          - Bending+shear rows (w, theta_x, theta_y): dof_local uses
            these RAW, so J is just the identity on those 12 rows/cols
            (each node's own w/theta_x/theta_y column only).
          - Drilling row (theta_z_local = theta_z(node) - theta_z_mean):
            linear in the four theta_z DOFs alone, so J's entry is
            (delta_ij - 1/4) for every (node i row, node j theta_z col).
          - Membrane rows (u,v): u_local_mem_xy[i] = (R_drill @ d_i)[:2]
            - Xref_local[i][:2], where d_i = x_current[i] - centroid_
            current. Two independent dependencies, both linear given
            R_drill and d_i are evaluated at the CURRENT u (this is a
            derivative of a product of two u-dependent factors, R_drill
            and d_i, each separately linear in u):
              * w.r.t. u_trans[j] (any of the 3 translation DOFs of node
                j): d(d_i)/d(u_trans[j]) = (delta_ij - 1/4) * I3 (same
                centroid-subtraction structure as the position-based
                extraction itself), so this contributes
                (delta_ij - 1/4) * R_drill[0:2, :] to J's (i, j)
                translation sub-block.
              * w.r.t. theta_z[k] (any of the 4 drilling DOFs): only
                R_drill itself depends on theta_z_mean, so this
                contributes d(R_drill)/d(theta_z_mean) * (1/4) @ d_i,
                taking the first two rows -- SAME value for every k
                (theta_z_mean is a plain average, so its derivative
                w.r.t. any one of the four theta_z DOFs is 1/4
                regardless of which one), hence J's (i, *) theta_z
                columns are all identical 2-vectors, one per node i.
              d(R_drill)/d(theta_z_mean) = R0 @ _exact_drill_rotation_
              deriv(theta_z_mean) (Wave 4 item 18 -- EXACT single-axis
              rotation, replacing the first-order truncation this
              docstring originally described; no longer a fixed
              constant, since the exact update is nonlinear in
              theta_z_mean, so dRdrill must be recomputed from the
              CURRENT state every call) -- dRdrill := (1/4) * R0 @
              _exact_drill_rotation_deriv(theta_z_mean) supplies every
              one of those 16 theta_z-column entries via
              (dRdrill @ d_i)[0:2].

        K_geo (24x24) is the OTHER half of the product rule that a naive
        "differentiate dof_local only" tangent would miss: internal_
        force()'s own T(u).T factor also depends on u (through the same
        theta_z_mean), so d(f_global)/du has a second term d(T.T)/du @
        f_local, exactly mirroring Beam2DCorotational's K_geo (a term
        proportional to the CURRENT local force, not the material
        stiffness) and Tet10SolidTL's initial-stress/geometric term.
        Since T is block-diagonal-of-R_drill and d(R_drill)/d(theta_z_k)
        = dRdrill for every k (the same reasoning as above), this
        collapses to: build T_dRdrill, the block-diagonal-of-dRdrill
        matrix (using _rotation_matrix_c()'s exact stacking, even though
        dRdrill is not itself a rotation -- the stacking is just how
        T's block structure is built, it does not require its argument
        to be orthonormal), then vec_g = T_dRdrill.T @ f_local is the
        (identical) contribution to EVERY one of the four theta_z
        columns of K_geo; every other column of K_geo is exactly zero
        (T does not depend on any other DOF).

        Verified (tests/test_shell_corotational.py): T.T @ K_local0 @ J
        + K_geo matches _tangent_stiffness_complex_step()'s output to
        ~1e-16 relative at u=0, at random small/large states, at the
        bending-dominated state that broke the rejected Phase A designs
        (see "Design history"), and at rigid rotations up to 30 degrees
        -- i.e. this is a genuine closed-form derivative of
        internal_force(), not merely a plausible-looking approximation
        of it."""
        X_ref = np.asarray(elem_coords, dtype=float)
        u = np.asarray(u_elem, dtype=float)
        u_nodes = u.reshape(4, 6)
        u_trans = u_nodes[:, 0:3]
        theta = u_nodes[:, 3:6]

        # Same local-frame/dof_local construction as _local_relative_
        # dofs() (duplicated here, not called, since this method needs
        # several of the intermediate pieces -- R_drill, dRdrill, d --
        # individually, not just dof_local itself).
        e1_0, e2_0, e3_0, _ = self._linear._local_frame_and_coords(X_ref)
        R0 = np.vstack([e1_0, e2_0, e3_0])
        centroid_ref = X_ref.mean(axis=0)
        Xref_local = (X_ref - centroid_ref) @ R0.T

        theta_z_mean = theta[:, 2].mean()
        R_drill = R0 @ self._exact_drill_rotation(theta_z_mean)
        # Wave 4 item 18: d(R_drill)/d(theta_z_mean) is no longer the
        # CONSTANT -R0@S_z the first-order update gave -- it's now
        # R0 @ _exact_drill_rotation_deriv(theta_z_mean), evaluated at
        # the CURRENT state, since the exact rotation is nonlinear in
        # theta_z_mean. dRdrill still carries the 1/4 (d(theta_z_mean)/
        # d(any one theta_z DOF), same chain-rule reasoning as before).
        dRdrill = 0.25 * R0 @ self._exact_drill_rotation_deriv(theta_z_mean)

        x_current = X_ref + u_trans
        centroid_current = x_current.mean(axis=0)
        d = x_current - centroid_current   # (4, 3)

        x_current_local = d @ R_drill.T
        u_local_mem_xy = x_current_local[:, 0:2] - Xref_local[:, 0:2]
        thetaz_local = theta[:, 2] - theta_z_mean

        dof_local = np.zeros(24)
        J = np.zeros((24, 24))
        for i in range(4):
            dof_local[6 * i + 0] = u_local_mem_xy[i, 0]
            dof_local[6 * i + 1] = u_local_mem_xy[i, 1]
            dof_local[6 * i + 5] = thetaz_local[i]
            dof_local[6 * i + 2] = u_trans[i] @ e3_0
            dof_local[6 * i + 3] = theta[i] @ e1_0
            dof_local[6 * i + 4] = theta[i] @ e2_0

            mem_theta_z_contrib = (dRdrill @ d[i])[0:2]
            for j in range(4):
                delta = 1.0 if i == j else 0.0
                J[6 * i:6 * i + 2, 6 * j:6 * j + 3] = (delta - 0.25) * R_drill[0:2, :]
                J[6 * i:6 * i + 2, 6 * j + 5] = mem_theta_z_contrib
                J[6 * i + 5, 6 * j + 5] = delta - 0.25
            # Bending block Jacobian: w_local(i) = u_trans[i] @ e3_0, so
            # d(w_local(i))/d(u_trans[i]) = e3_0 (a FIXED row vector, R0
            # doesn't depend on u) -- same pattern for theta_x/theta_y
            # against e1_0/e2_0. Reduces to the old identity block
            # exactly when R0 = I (e1_0=[1,0,0] etc.), matching every
            # previously-validated test mesh; see _local_relative_dofs()'s
            # own comment for why this generalization was needed.
            J[6 * i + 2, 6 * i:6 * i + 3] = e3_0
            J[6 * i + 3, 6 * i + 3:6 * i + 6] = e1_0
            J[6 * i + 4, 6 * i + 3:6 * i + 6] = e2_0

        f_local = K_local0 @ dof_local + self._bending_membrane_coupling_force(
            elem_coords, dof_local, D, N_add=iter_state)

        T_dRdrill = self._rotation_matrix_c(dRdrill[0], dRdrill[1], dRdrill[2])
        vec_g = T_dRdrill.T @ f_local
        K_geo = np.zeros((24, 24))
        for j in range(4):
            K_geo[:, 6 * j + 5] = vec_g

        return J, K_geo, R_drill, f_local

    def _tangent_stiffness_committed_complex_step(self, elem_coords, u_elem, D, state, thickness=1.0,
                                                    iter_state=None, h=1e-20):
        """Wave 4 item 46, building block C: complex-step tangent for
        the state=... path -- NOT real finite-difference, unlike the
        reverted first attempt at this building block (see the class-
        level "BUILDING BLOCK C" comment's "FIRST ATTEMPT, REVISED IN
        PLACE" paragraph). That first attempt needed
        `_mean_rigid_rotation()` (real-valued only, via `_log_map()`'s
        arctan2), which is why it would have been stuck with
        `Beam3DCorotational`'s real-FD precedent. The REVISED design
        `_internal_force_committed()` now uses has NO such dependency
        -- `_bending_membrane_coupling_force_committed()` is built
        entirely from subtraction/products/sums of `theta` (complex
        during complex-step) and REAL state constants, exactly the same
        polynomial structure `_bending_membrane_coupling_force()` itself
        already tolerates -- so this can, and does, reuse
        `_tangent_stiffness_complex_step()`'s own exact method, just
        with state=... threaded through `internal_force()`. h=1e-20
        (not FD's 1e-6) since complex-step has no cancellation error to
        balance against truncation error -- same reasoning
        `_tangent_stiffness_complex_step()`'s own docstring gives."""
        n = len(u_elem)
        u_c = np.asarray(u_elem, dtype=complex)
        K = np.zeros((n, n))
        for j in range(n):
            du = np.zeros(n, dtype=complex)
            du[j] = 1j * h
            f_pert = self.internal_force(elem_coords, u_c + du, D, thickness,
                                          iter_state=iter_state, state=state)
            K[:, j] = f_pert.imag / h
        return 0.5 * (K + K.T)

    def tangent_stiffness(self, elem_coords, u_elem, D, thickness=1.0, iter_state=None, state=None,
                           method="complex_step", **kwargs):
        """Analytic tangent (added as Phase B of
        docs/shells.md Section 4.2, replacing complex-
        step differentiation as the default -- see
        _tangent_stiffness_complex_step() below, KEPT as the validation
        reference this was checked against, not dead code, exactly
        Tet10SolidTL's own Phase A -> Phase B naming convention).

        d(f_global)/du = T.T @ (K_local0 + K_add_local) @ J + K_geo, a
        standard material-plus-geometric split (Bathe, "Finite Element
        Procedures," Sec. 6.3.3; Belytschko/Liu/Moran, "Nonlinear Finite
        Elements," Ch. 4) -- T.T @ K_local0 @ J is the ORIGINAL material
        term (how K_local0 @ dof_local responds to a change in local
        relative DOFs, at FIXED frame; K_local0 is constant, so this is
        exact); K_add_local = _coupling_tangent_local(...) is the
        material term for the NEW bending-membrane coupling force
        (_bending_membrane_coupling_force() is nonlinear in dof_local,
        so unlike K_local0 this piece is state-dependent, hence
        evaluated by complex-step rather than a constant matrix -- see
        that method's own docstring for why this is both correct and
        cheap); K_geo is the "geometric" term (how the CURRENT TOTAL
        local force gets rotated differently as the frame itself
        changes, at FIXED local relative DOFs) -- see
        _dof_local_jacobian_and_geo()'s own docstring for the full
        derivation. Symmetrized (0.5*(K+K.T)) before returning, matching
        _tangent_stiffness_complex_step()'s own convention -- see this
        module's class-level "Tangent" comment for why f_int(u) is not
        exactly the gradient of a scalar energy in this construction, so
        the raw Jacobian is not exactly symmetric.

        state (Wave 4 item 46, building block C), when not None,
        bypasses this whole analytic derivation (which is hand-derived
        specifically for the state=None extraction's own force/Jacobian
        structure -- J, K_geo above are not what `_internal_force_
        committed()` needs) and uses `_tangent_stiffness_committed_
        complex_step()` instead -- see that method's own docstring for
        why complex-step (not real finite-difference) is safe and
        correct here.

        method (only consulted when state is not None -- Wave 9 item 96,
        docs/consolidated_future_roadmap.md, PyTorch side-by-side
        extension): "complex_step" (default, UNCHANGED behavior) keeps
        using _tangent_stiffness_committed_complex_step() above.
        "autograd" (opt-in) routes through autograd_tangent.
        shell4_mitc_corotational_committed_tangent_autograd() instead --
        an INDEPENDENTLY reimplemented (in torch) version of the exact
        same state=... force computation, differentiated via
        torch.autograd.grad rather than complex-step. NOTE: unlike
        Tet4NeoHookean's method="autograd" (elements/nonlinear_solids.py,
        Wave 9 item 93), this is NOT a truncation-error fix -- complex-
        step already recovers the exact derivative here (see that
        method's own docstring) -- so this option exists for the same
        "keep every valid implementation available, an independent
        third method is stronger evidence than two that already agree"
        reason autograd_tangent.py's own module docstring gives for its
        other cross-checks, plus a documented future path to a GPU/
        batched-element win once wired behind a device= dispatch (not
        done here). Requires torch (raises ImportError otherwise, via
        that module's own _require_torch()); only supports
        iter_state=None (raises NotImplementedError for the mixed-
        formulation branch -- see that function's own docstring for
        why). method="autograd" with state=None is a no-op (the
        analytic branch below never looks at method at all)."""
        if state is not None:
            if method == "autograd":
                from ..autograd_tangent import shell4_mitc_corotational_committed_tangent_autograd
                K, _ = shell4_mitc_corotational_committed_tangent_autograd(
                    elem_coords, u_elem, D, state, iter_state=iter_state, formulation=self)
                return K
            elif method != "complex_step":
                raise ValueError(
                    f"Shell4MITCCorotational.tangent_stiffness: unknown method={method!r} "
                    "-- expected 'complex_step' (default) or 'autograd'.")
            return self._tangent_stiffness_committed_complex_step(elem_coords, u_elem, D, state,
                                                                    thickness=thickness, iter_state=iter_state)
        K_local0 = self._local_material_stiffness(elem_coords, D)
        J, K_geo, R_drill, f_local = self._dof_local_jacobian_and_geo(
            elem_coords, u_elem, K_local0, D, iter_state=iter_state)
        dof_local, _, _, _ = self._local_relative_dofs(elem_coords, u_elem)
        K_add_local = self._coupling_tangent_local(elem_coords, dof_local, D, N_add=iter_state)
        if iter_state is not None:
            K_add_local = K_add_local + self._coupling_schur_correction(elem_coords, dof_local, D)
        T = self._rotation_matrix_c(R_drill[0], R_drill[1], R_drill[2])
        K = T.T @ (K_local0 + K_add_local) @ J + K_geo
        return 0.5 * (K + K.T)

    def _tangent_stiffness_complex_step(self, elem_coords, u_elem, D, thickness=1.0, h=1e-20, **kwargs):
        """Complex-step differentiation of internal_force() -- the
        ORIGINAL tangent_stiffness() implementation (Phase A), RENAMED
        (not removed) when the analytic tangent above replaced it as
        the default: kept as the independent validation reference the
        analytic tangent was checked against (tests/
        test_shell_corotational.py) and as a debugging fallback, exactly
        mirroring Tet10SolidTL._tangent_stiffness_complex_step()'s own
        role (see that method's docstring for why complex-step recovers
        the exact derivative, not merely an improved approximation of
        it, for any internal_force() that is genuinely holomorphic in
        u_elem)."""
        n = len(u_elem)
        u_c = np.asarray(u_elem, dtype=complex)
        K = np.zeros((n, n))
        for j in range(n):
            du = np.zeros(n, dtype=complex)
            du[j] = 1j * h
            f_pert = self.internal_force(elem_coords, u_c + du, D, thickness, **kwargs)
            K[:, j] = f_pert.imag / h
        return 0.5 * (K + K.T)

    def stiffness(self, elem_coords, D, thickness=1.0, **kwargs):
        """Initial (zero-displacement) tangent -- interface completeness,
        matching Tet10SolidTL.stiffness()/Beam2DCorotational.stiffness().
        Reduces to Shell4MITC.stiffness(elem_coords, D) to ~1e-13
        relative precision -- see this module's class-level comment
        block ("Why this reduces EXACTLY...") and
        tests/test_shell_corotational.py for the measured number."""
        return self.tangent_stiffness(elem_coords, np.zeros(24), D, thickness, **kwargs)

    def mass(self, elem_coords, rho_matrix, thickness=1.0):
        """Consistent mass matrix in the REFERENCE (undeformed)
        configuration -- delegates directly to Shell4MITC.mass(), same
        convention as every other nonlinear element in this package
        (Beam2DCorotational.mass(), Tet10SolidTL inheriting Tet10Solid3D
        .mass() unmodified): no geometric/nonlinear correction to the
        mass itself, only to the stiffness/internal force."""
        return self._linear.mass(elem_coords, rho_matrix, thickness)
