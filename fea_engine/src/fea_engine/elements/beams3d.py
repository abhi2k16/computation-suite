"""
beams3d.py -- Module 18 (general-purpose extensions roadmap Phase 3):
a linear 3-D frame element, Beam3DEulerBernoulli. The 2-D beam
(Beam2DEulerBernoulli, beams.py) only carries bending in one plane
(2 dof/node: v, theta) -- correct for a 2-D frame, but a general 3-D
frame member (a building frame, a space truss/piping run) bends about
TWO principal axes, twists, and stretches axially: 6 dof/node
(u, v, w, theta_x, theta_y, theta_z).

Deliberately built from FOUR uncoupled 1-D blocks in the element's
LOCAL frame -- axial (truss-like), torsional (St. Venant, GJ/L), and
TWO independent Hermite bending blocks (one per principal axis) --
then rotated to global coordinates. The two bending blocks are not
re-derived from scratch: they REUSE Beam2DEulerBernoulli's existing,
already-validated closed-form Hermite stiffness/mass matrices, called
once for the (v, theta_z) in-plane bending block directly, and once
more (with a sign-flip transform, see _FLIP below) for the (w,
theta_y) out-of-plane bending block -- see docs/general_purpose_
extensions_roadmap.md Section 2 for why this reuse is possible (both
blocks are the SAME Euler-Bernoulli bending physics, just oriented
differently, and a fixed local coordinate SIGN convention is the only
thing that differs between them).
"""
import numpy as np

from .base import Element
from .beams import Beam2DEulerBernoulli


# The (w, theta_y) bending block uses the SAME Hermite stiffness/mass
# form as the (v, theta_z) block, but with the local rotation's sign
# flipped: for a right-handed local frame (x along the beam, y and z
# the two principal bending axes), the slope-rotation relation is
# dv/dx = +theta_z in the x-y plane but dw/dx = -theta_y in the x-z
# plane (the sign difference is a property of the right-handed frame,
# not a modeling choice). Substituting phi_y = -theta_y turns the
# (w, phi_y) block into EXACTLY Beam2DEulerBernoulli's formula (with
# EIy in place of EI); transforming back to theta_y is the congruence
# transform M_w = _FLIP @ M_v_formula(EIy) @ _FLIP, valid for both
# stiffness and (consistent) mass since both are energy-conjugate
# quadratic forms. _FLIP is its own inverse (diag(+-1) is orthogonal),
# so this single 4x4 matrix does the whole job.
_FLIP = np.diag([1.0, -1.0, 1.0, -1.0])

# Local DOF layout per node: (u, v, w, theta_x, theta_y, theta_z).
# Index groups used to scatter each 1-D/bending block into the full
# 12x12 local matrix.
_AX = [0, 6]                    # axial: u1, u2
_TW = [3, 9]                    # torsion: theta_x1, theta_x2
_VB = [1, 5, 7, 11]             # in-plane bending: v1, theta_z1, v2, theta_z2
_WB = [2, 4, 8, 10]             # out-of-plane bending: w1, theta_y1, w2, theta_y2


def _local_axes(X1, X2, ref_up=None):
    """Right-handed local frame (e1, e2, e3) for a 3-D beam from node
    X1 to X2: e1 along the beam; e2, e3 the two principal bending
    directions, fixed by a reference "up" vector (default global Z,
    falling back to global Y for a member that's itself vertical/
    parallel to Z, where Z can't serve as an independent reference) --
    a single axis vector alone leaves rotation ABOUT that axis
    undetermined, so a second, non-parallel reference vector is always
    needed; this is the standard convention (e.g. SAP2000, most
    matrix-structural-analysis texts) rather than an ad hoc choice."""
    dX = np.asarray(X2, dtype=float) - np.asarray(X1, dtype=float)
    L = np.linalg.norm(dX)
    e1 = dX / L
    if ref_up is None:
        ref_up = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(np.cross(e1, ref_up)) < 1e-6:
            ref_up = np.array([0.0, 1.0, 0.0])
    e3 = np.cross(e1, ref_up)
    e3 = e3 / np.linalg.norm(e3)
    e2 = np.cross(e3, e1)   # already unit: e3 perp e1, both unit
    return L, e1, e2, e3


class Beam3DEulerBernoulli(Element):
    n_nodes, dofs_per_node, dim = 2, 6, 1
    translational_dof_mask = [True, True, True, False, False, False]

    def __init__(self, ref_up=None):
        """ref_up: optional reference "up" vector fixing the local
        y/z orientation (see _local_axes()) -- defaults to global Z
        (falling back to Y for vertical members), the standard
        convention; pass an explicit vector to orient a member's
        principal axes differently (e.g. for a rotated/rolled section)."""
        self.ref_up = None if ref_up is None else np.asarray(ref_up, dtype=float)

    def shape_and_derivs(self, natural_coords):
        """Linear interpolation along the LOCAL chord -- interface
        completeness only (mesh quality checks), same role as
        Beam2DEulerBernoulli's; stiffness()/mass() below use the
        closed-form blocks directly, not this."""
        (xi,) = natural_coords
        N = np.array([(1 - xi) / 2, (1 + xi) / 2])
        dN_dxi = np.array([-0.5, 0.5])
        return N, dN_dxi.reshape(1, 2)

    def _local_matrices(self, elem_coords, EA, GJ, EIy, EIz, rho_A=None, rho_Ip=None):
        """Builds the 12x12 LOCAL stiffness matrix (rho_A/rho_Ip=None)
        or LOCAL mass matrix (rho_A/rho_Ip given) -- shared scaffolding
        so stiffness()/mass() don't duplicate the block-assembly logic,
        only which four blocks they ask for."""
        L, e1, e2, e3 = _local_axes(elem_coords[0], elem_coords[1], self.ref_up)
        want_mass = rho_A is not None
        Mloc = np.zeros((12, 12))

        if want_mass:
            m_ax = (rho_A * L / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
            m_tw = (rho_Ip * L / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
            beam2d = Beam2DEulerBernoulli()
            coords_1d = np.array([[0.0], [L]])
            m_v = beam2d.mass(coords_1d, rho_A)
            m_w = _FLIP @ m_v @ _FLIP
            for idx, block in [(_AX, m_ax), (_TW, m_tw), (_VB, m_v), (_WB, m_w)]:
                Mloc[np.ix_(idx, idx)] = block
        else:
            k_ax = (EA / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])
            k_tw = (GJ / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])
            beam2d = Beam2DEulerBernoulli()
            coords_1d = np.array([[0.0], [L]])
            k_v = beam2d.stiffness(coords_1d, EIz)
            k_w = _FLIP @ beam2d.stiffness(coords_1d, EIy) @ _FLIP
            for idx, block in [(_AX, k_ax), (_TW, k_tw), (_VB, k_v), (_WB, k_w)]:
                Mloc[np.ix_(idx, idx)] = block

        R = np.vstack([e1, e2, e3])          # 3x3: global -> local
        T = np.zeros((12, 12))
        for i in range(4):                   # 4 vector triplets: (u,v,w) x2 nodes, (thx,thy,thz) x2 nodes
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return T.T @ Mloc @ T                # local -> global

    def stiffness(self, elem_coords, rigidities, thickness=1.0, **kwargs):
        """rigidities = (EA, GJ, EIy, EIz), as returned by
        material.beam3d_rigidities(mat, sec)."""
        EA, GJ, EIy, EIz = rigidities
        return self._local_matrices(elem_coords, EA, GJ, EIy, EIz)

    def full_stiffness(self, elem_coords, rigidities, thickness=1.0):
        """N/A -- closed-form, no quadrature order to vary (same
        reasoning as Beam2DEulerBernoulli)."""
        return self.stiffness(elem_coords, rigidities, thickness)

    def reduced_stiffness(self, elem_coords, rigidities, thickness=1.0):
        return self.stiffness(elem_coords, rigidities, thickness)

    def geometric_stiffness(self, elem_coords, N, thickness=1.0):
        """Module 19 (general-purpose extensions Phase 4): reuses
        Beam2DEulerBernoulli's own geometric_stiffness() for BOTH
        bending planes -- exactly the same reuse pattern _local_
        matrices() above already uses for stiffness()/mass() (call it
        once directly for the (v, theta_z) block, once more through
        the _FLIP congruence transform for the (w, theta_y) block),
        then rotate to global with the same local frame. N is the
        reference axial force (TENSION-POSITIVE, see Element.
        geometric_stiffness's docstring) -- assumed uniform along the
        member, the standard linear-buckling idealization (the
        reference STATIC state that sets N is a separate, prior
        solve; this element has no notion of how N varies along its
        own length beyond that one number).

        Scope note: this is transverse-bending geometric stiffness
        only (Przemieniecki's classic Kg) -- it does NOT include the
        torsion-bending (Wagner) coupling term that governs LATERAL-
        TORSIONAL buckling of open thin-walled sections; that needs
        the section's warping/monosymmetry properties, which
        Section3D does not carry (same honestly-documented scope
        limit as Section3D.J's docstring). Fine for flexural (Euler
        column) buckling of doubly-symmetric sections -- the case
        this module is validated against."""
        L, e1, e2, e3 = _local_axes(elem_coords[0], elem_coords[1], self.ref_up)
        beam2d = Beam2DEulerBernoulli()
        coords_1d = np.array([[0.0], [L]])
        kg_v = beam2d.geometric_stiffness(coords_1d, N)
        kg_w = _FLIP @ kg_v @ _FLIP

        Kgloc = np.zeros((12, 12))
        Kgloc[np.ix_(_VB, _VB)] = kg_v
        Kgloc[np.ix_(_WB, _WB)] = kg_w
        # _AX, _TW (axial, torsion) get no geometric-stiffness contribution
        # here -- Przemieniecki's Kg is purely a transverse-bending effect;
        # see the scope note above for what this deliberately omits.

        R = np.vstack([e1, e2, e3])
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return T.T @ Kgloc @ T

    def mass(self, elem_coords, mass_props, thickness=1.0):
        """mass_props = (rho*A, rho*Ip), as returned by
        material.beam3d_mass_props(mat, sec) -- Ip = Iy + Iz, the
        cross-section's polar AREA moment (mass property), NOT the
        torsional STIFFNESS constant J used by stiffness() (see
        Section3D's docstring for why these differ in general)."""
        rho_A, rho_Ip = mass_props
        return self._local_matrices(elem_coords, None, None, None, None,
                                     rho_A=rho_A, rho_Ip=rho_Ip)

    def lumped_mass(self, elem_coords, mass_props, thickness=1.0):
        """HRZ diagonal scaling, computed directly from THIS element's
        own consistent mass (not the generic base-class version, which
        assumes a single scalar density and a genuinely isoparametric
        Jacobian -- neither holds here: mass_props is a 2-tuple, and
        this element's natural 1-D parametrization is embedded in a
        3-D physical space, so a square-Jacobian assumption doesn't
        apply). Rescales translational and rotational-about-beam-axis
        entries separately by their own exact block total (translation:
        rho*A*L; torsion: rho*Ip*L), matching Tri3PlaneStress/
        Tet4Solid3D's pattern of a fully self-contained, non-generic
        HRZ for elements whose physics doesn't fit the generic Gauss
        loop -- BENDING-rotation entries (theta_y, theta_z) keep their
        consistent-mass diagonal value directly (HRZ conventionally
        does not attempt to rescale those independently)."""
        me = self.mass(elem_coords, mass_props, thickness)
        rho_A, rho_Ip = mass_props
        X1, X2 = elem_coords[0], elem_coords[1]
        L = np.linalg.norm(np.asarray(X2, dtype=float) - np.asarray(X1, dtype=float))
        diag = np.diag(me).copy()

        trans_idx = [0, 1, 2, 6, 7, 8]
        c_trans = (rho_A * L) / diag[trans_idx].sum()
        diag[trans_idx] *= c_trans

        tw_idx = _TW
        c_tw = (rho_Ip * L) / diag[tw_idx].sum()
        diag[tw_idx] *= c_tw

        return np.diag(diag)


def _skew3(v):
    """3x3 skew-symmetric (cross-product) matrix of a 3-vector v."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


# =====================================================================
# Beam3DCorotational -- Wave 4 item 23 (docs/consolidated_future_
# roadmap.md): geometric nonlinearity (large displacement/rotation,
# small local strain) for Beam3DEulerBernoulli, generalizing
# Beam2DCorotational's 2-D scalar-chord-angle co-rotational pattern to
# a full 3-D beam (axial + torsion + two independent bending planes).
# The same category of extension Shell4MITCCorotational adds on top of
# Shell4MITC -- built independently of that element's own dead-end
# history (a 1-D beam's kinematics don't have a shell's membrane/
# bending DOF-coupling problem: a beam's "membrane" (axial) and
# "bending" natural strains are ALREADY exactly decoupled scalars by
# construction, not a discretization choice that can leak).
#
# Kinematics, mirroring Beam2DCorotational's own (L0, L, beta, e_bar,
# th1_bar, th2_bar) pattern, generalized from one scalar chord angle to
# a full 3-D corotated frame (e1, e2, e3):
#   1. e1 = the CURRENT (deformed) unit chord direction -- exact, same
#      role as Beam2DCorotational's (c, s).
#   2. e2, e3 = the reference frame's own (e2_0, e3_0) (Beam3DEulerBernoulli's
#      _local_axes(), unchanged), TRANSPORTED to the current chord via
#      the "smallest rotation" (Rodrigues, about e1_0 x e1) that takes
#      e1_0 to e1 -- EXACT for any bending angle except the single
#      singular case e1 = -e1_0 (axis undefined; see _minimal_rotation()
#      for the documented, still-reasonable fallback), then further
#      TWISTED about the (now-current) e1 by the mean of the two nodes'
#      own rotation-about-e1 -- an EXACT single-axis rotation (the same
#      trick Wave 4 item 18 validated for the shell's drilling frame:
#      rotation about one FIXED, already-known axis needs only a scalar
#      angle's cos/sin, no general 3-vector Rodrigues singularity).
#   3. Six natural (rigid-body-free) strains: e_bar = L - L0 (axial,
#      exact); twist_bar = (theta2 - theta1).e1 (relative twist about
#      the CURRENT beam axis -- exact given e1); and, for EACH bending
#      plane independently, th1/th2 = the node's own global rotation
#      VECTOR projected onto the corotated e2 or e3 axis (theta_i.e2,
#      theta_i.e3) -- the beam-axis analogue of Beam2DCorotational's
#      own th_bar = wrap(theta - beta): since the frame's own e2/e3
#      construction (step 2) already absorbed the "beta"-equivalent
#      chord-tilt correction via the minimal-rotation transport, no
#      further subtraction is needed for the bending naturals, exactly
#      mirroring how beta is subtracted ONCE (into the frame itself,
#      via phi0) rather than separately from each node's theta_bar.
#
# Deliberately built WITHOUT hand-deriving an analytic B-matrix
# (natural-dof-to-global Jacobian) or analytic tangent -- unlike
# Beam2DCorotational's own hand-derived 3x6 B/K_geo (tractable for one
# scalar rotation), a full 3-D corotated-frame Jacobian (Battini &
# Pacoste's own consistent-tangent derivation, Crisfield Vol 2 Ch 17)
# is a substantial, error-prone undertaking on its own (see this
# module's own investigation notes and Shell4MITCCorotational's "Design
# history" for how much transcription/derivation risk this class of
# problem carries even for domain experts). Following this package's
# OWN established staged-build precedent (Tet10SolidTL,
# Shell4MITCCorotational Phase A: "implement internal_force() first...
# get a numerically CONSISTENT tangent for free via [numerical]
# differentiation... THEN hand-derive the analytic ... tangent as a
# follow-up speed optimization"), this class instead builds BOTH the
# natural-dof Jacobian (B, 6x12) AND tangent_stiffness() via REAL
# CENTRAL FINITE DIFFERENCE, not complex-step: _kinematics()'s reliance
# on np.linalg.norm() (for L and for the minimal-rotation angle) is not
# holomorphic, so complex-step (the technique Shell4MITCCorotational's
# own _tangent_stiffness_complex_step() uses) is not safe here --
# exactly the scenario docs/shells.md Section 1.3 (`Shell4Director`'s
# own "TANGENT STRATEGY") anticipates ("complex-step (or, if internal_force()
# isn't complex-safe throughout, real central-FD)"), so real FD is used
# throughout instead, deliberately, not as an oversight.
#
# Cost/correctness tradeoff, honestly scoped: this makes tangent_
# stiffness() noticeably slower (a nested FD -- differentiating
# internal_force(), itself internally differentiating _natural_dofs())
# than a closed-form analytic tangent would be -- 24 internal_force()
# evaluations per tangent, each internally doing 24 _natural_dofs()
# evaluations for its own B matrix. This is a genuine Phase A, not a
# finished optimization -- an analytic Phase B tangent (the real
# Battini-Pacoste consistent-tangent derivation) is explicitly deferred
# future work, exactly as docs/shells.md Section 4.2 (Phase A -> Phase B)
# phased the analogous shell derivation. Validated (tests/
# test_beam3d_corotational.py): u=0 regression against Beam3DEulerBernoulli.
# stiffness() to near machine precision; FD-tangent self-consistency
# (tangent_stiffness() matches an INDEPENDENT, higher-order-accurate FD
# of internal_force(), confirming the nested-FD construction is at
# least internally consistent, not merely "the same FD twice");
# rigid-motion invariance (translation exactly zero force by
# construction; rotation about the beam's own axis exactly zero by the
# SAME exact-single-axis-rotation argument item 18 established;
# transverse rigid rotation bounded-small, analogous to the shell's own
# bending-block limitation, not claimed exact); and a multi-element
# cantilever actually converging under solve_nonlinear_static() and
# landing near the linear Euler-Bernoulli tip deflection at small load.
# =====================================================================
class Beam3DCorotational(Element):
    n_nodes, dofs_per_node, dim = 2, 6, 1
    translational_dof_mask = [True, True, True, False, False, False]

    def __init__(self, ref_up=None):
        """ref_up: same reference "up" vector convention as
        Beam3DEulerBernoulli (fixes the REFERENCE local y/z
        orientation, i.e. e2_0/e3_0 -- see _local_axes())."""
        self.ref_up = None if ref_up is None else np.asarray(ref_up, dtype=float)
        self._linear = Beam3DEulerBernoulli(ref_up=ref_up)

    def shape_and_derivs(self, natural_coords):
        """Linear interpolation along the reference chord -- interface
        completeness only, same role as Beam2DCorotational's/
        Beam3DEulerBernoulli's; internal_force()/tangent_stiffness()
        below don't use this."""
        (xi,) = natural_coords
        N = np.array([(1 - xi) / 2, (1 + xi) / 2])
        dN_dxi = np.array([-0.5, 0.5])
        return N, dN_dxi.reshape(1, 2)

    @staticmethod
    def _minimal_rotation(a, b):
        """3x3 rotation matrix R such that R @ a = b, for unit vectors
        a, b, via the "smallest rotation" (Rodrigues) construction:
        rotate about axis = a x b by the angle between a and b. This is
        EXACT for any angle except a = -b, where a x b ~ 0 leaves the
        axis undefined -- a genuine, singular edge case (an element
        whose chord has flipped to point exactly opposite its own
        reference direction), handled by falling back to an arbitrary
        perpendicular flip axis (a valid, if not uniquely-determined,
        180-degree rotation) rather than dividing by ~0.

        The "already aligned" shortcut is gated on s = |a x b| (which
        scales LINEARLY with the rotation angle theta for small theta),
        NOT on c = a.b (which scales as 1 - theta^2/2 -- a MUCH worse-
        conditioned quantity to threshold against a small-but-genuine
        rotation: at theta ~ 1e-6, 1-c ~ 5e-13, already below a naive
        "c > 1 - 1e-12" cutoff, which would then silently return
        IDENTITY for a real, FD-perturbation-scale rotation. Caught by
        this class's own u=0 tangent regression test going through
        exactly this size of perturbation (h=1e-6) and finding zero
        bending-translation coupling that a larger, non-FD-scale
        perturbation correctly showed as nonzero -- fixed by
        thresholding s (< 1e-13) instead, which correctly distinguishes
        "no rotation at all" from "a small but real rotation this
        class's own finite-difference machinery needs to resolve"."""
        c = float(np.dot(a, b))
        axis_raw = np.cross(a, b)
        s = np.linalg.norm(axis_raw)
        if s < 1e-13:
            if c > 0.0:
                return np.eye(3)
            trial = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = trial - np.dot(trial, a) * a
            axis = axis / np.linalg.norm(axis)
            K = _skew3(axis)
            return np.eye(3) + 2.0 * (K @ K)
        axis = axis_raw / s
        theta = np.arctan2(s, c)
        K = _skew3(axis)
        return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    @staticmethod
    def _axis_rotation(axis, theta):
        """Exact rotation matrix for angle theta about a KNOWN unit
        axis (Rodrigues formula) -- used only for the twist correction
        about e1, the same "single fixed axis, so an exact update is
        safe/simple" reasoning Wave 4 item 18 established for the
        shell's drilling frame."""
        K = _skew3(axis)
        return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    @staticmethod
    def _rotation_vector(Rmat):
        """Inverse Rodrigues (log-map): the axis*angle rotation VECTOR
        of a rotation matrix Rmat. Computes theta via arctan2(sin_theta,
        cos_theta), NOT arccos(cos_theta) alone -- arccos is numerically
        UNSTABLE near theta=0 (catastrophic cancellation: cos_theta is
        computed as (trace-1)/2, which sits right at ~1.0 for a small
        rotation, and arccos's derivative diverges there, amplifying a
        tiny floating-point error in cos_theta into a badly wrong theta;
        arctan2(sin_theta, cos_theta) has no such instability, since
        sin_theta -- computed directly from the matrix's antisymmetric
        part, no cancellation -- dominates the ratio smoothly as
        theta->0). Caught via this element's own FD-tangent Jacobian at
        h=1e-6 returning exactly zero coupling that a LARGER perturbation
        (1e-3) correctly showed as nonzero -- a clean sign of exactly
        this kind of small-angle precision loss, not a conceptual bug in
        the log-map formula itself. Near theta=pi the axis becomes
        ill-conditioned (a well-known degeneracy of ANY axis-angle
        log-map) -- left as a documented, narrow edge case (this
        element's own frame-to-frame rotation is not expected to
        approach a full half-turn within its MVP small-relative-
        rotation scope)."""
        v = 0.5 * np.array([Rmat[2, 1] - Rmat[1, 2],
                             Rmat[0, 2] - Rmat[2, 0],
                             Rmat[1, 0] - Rmat[0, 1]])
        sin_theta = np.linalg.norm(v)
        if sin_theta < 1e-10:
            return v   # theta/sin_theta -> 1 in this limit
        cos_theta = np.clip((np.trace(Rmat) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arctan2(sin_theta, cos_theta)
        return v * (theta / sin_theta)

    def _frame_and_naturals(self, elem_coords, u_elem):
        """Returns (e1, e2, e3, naturals): naturals is the 6-vector
        [e_bar, twist_bar, th1_z, th2_z, th1_y, th2_y] -- see class
        docstring for the derivation. th*_z drives e1-e2-plane bending
        (EIz); th*_y drives e1-e3-plane bending (EIy) -- matching
        Beam3DEulerBernoulli's own (_VB, _WB) plane convention."""
        X1 = np.asarray(elem_coords[0], dtype=float)
        X2 = np.asarray(elem_coords[1], dtype=float)
        L0, e1_0, e2_0, e3_0 = _local_axes(X1, X2, self.ref_up)

        u = np.asarray(u_elem, dtype=float)
        u1, th1 = u[0:3], u[3:6]
        u2, th2 = u[6:9], u[9:12]

        dx = (X2 + u2) - (X1 + u1)
        L = np.linalg.norm(dx)
        e1 = dx / L

        R_transport = self._minimal_rotation(e1_0, e1)
        e2_t = R_transport @ e2_0
        e3_t = R_transport @ e3_0

        mean_twist = 0.5 * (float(th1 @ e1) + float(th2 @ e1))
        R_twist = self._axis_rotation(e1, mean_twist)
        e2 = R_twist @ e2_t
        e3 = R_twist @ e3_t

        # beta_vec: the rotation VECTOR (axis*angle) of the frame's own
        # reference-to-current rotation (R_total = R_twist @ R_transport)
        # -- the 3-D generalization of Beam2DCorotational's scalar beta
        # (= wrap(phi-phi0)). Each node's rotation is SUBTRACTED by
        # beta_vec (vector subtraction, mirroring 2D's th_bar=th-beta)
        # BEFORE projecting onto e2/e3 -- projecting theta_i onto e2/e3
        # directly (without this subtraction) would miss the frame's own
        # translation-dependence entirely (theta_i=0 forces the raw dot
        # product to 0 regardless of how e2/e3 themselves move with
        # translation, which is NOT how 2D's subtractive beta behaves at
        # theta=0) -- caught via the u=0 regression test against
        # Beam3DEulerBernoulli.stiffness() before this was corrected.
        R_total = R_twist @ R_transport
        beta_vec = self._rotation_vector(R_total)
        th1_rel = th1 - beta_vec
        th2_rel = th2 - beta_vec

        e_bar = L - L0
        # beta_vec's own component along e1 cancels in this difference
        # (both nodes subtract the SAME beta_vec), so twist_bar reduces
        # to the raw relative-twist difference either way.
        twist_bar = float(th2_rel @ e1) - float(th1_rel @ e1)
        th1_z = float(th1_rel @ e3)
        th2_z = float(th2_rel @ e3)
        th1_y = float(th1_rel @ e2)
        th2_y = float(th2_rel @ e2)

        naturals = np.array([e_bar, twist_bar, th1_z, th2_z, th1_y, th2_y])
        return e1, e2, e3, naturals

    def _natural_dofs(self, elem_coords, u_elem):
        _, _, _, naturals = self._frame_and_naturals(elem_coords, u_elem)
        return naturals

    def _B_matrix_fd(self, elem_coords, u_elem, h=1e-6):
        """6x12: d(naturals)/d(u_elem), via real central finite
        difference (see class docstring for why complex-step isn't
        used) -- built ONCE per internal_force()/tangent_stiffness()
        call, not hand-derived symbolically."""
        u = np.asarray(u_elem, dtype=float)
        B = np.zeros((6, 12))
        for j in range(12):
            du = np.zeros(12)
            du[j] = h
            n_plus = self._natural_dofs(elem_coords, u + du)
            n_minus = self._natural_dofs(elem_coords, u - du)
            B[:, j] = (n_plus - n_minus) / (2.0 * h)
        return B

    @staticmethod
    def _generalized_forces(elem_coords, rigidities, naturals):
        X1 = np.asarray(elem_coords[0], dtype=float)
        X2 = np.asarray(elem_coords[1], dtype=float)
        L0 = np.linalg.norm(X2 - X1)
        EA, GJ, EIy, EIz = rigidities
        e_bar, twist_bar, th1_z, th2_z, th1_y, th2_y = naturals
        N = (EA / L0) * e_bar
        T = (GJ / L0) * twist_bar
        M1z = (EIz / L0) * (4.0 * th1_z + 2.0 * th2_z)
        M2z = (EIz / L0) * (2.0 * th1_z + 4.0 * th2_z)
        M1y = (EIy / L0) * (4.0 * th1_y + 2.0 * th2_y)
        M2y = (EIy / L0) * (2.0 * th1_y + 4.0 * th2_y)
        return np.array([N, T, M1z, M2z, M1y, M2y])

    def internal_force(self, elem_coords, u_elem, rigidities, thickness=1.0, **kwargs):
        """f_int = B^T @ generalized_forces -- the same virtual-work
        transformation Beam2DCorotational's own internal_force() uses
        (B^T @ [N, M1, M2]), generalized to 6 natural dofs and built
        via _B_matrix_fd() rather than a hand-derived analytic B."""
        naturals = self._natural_dofs(elem_coords, u_elem)
        generalized = self._generalized_forces(elem_coords, rigidities, naturals)
        B = self._B_matrix_fd(elem_coords, u_elem)
        return B.T @ generalized

    def tangent_stiffness(self, elem_coords, u_elem, rigidities, thickness=1.0, h=1e-6, **kwargs):
        """Real central-FD differentiation of internal_force() -- see
        class docstring for why this Phase A implementation uses FD
        rather than an analytic (material + geometric) tangent."""
        u = np.asarray(u_elem, dtype=float)
        K = np.zeros((12, 12))
        for j in range(12):
            du = np.zeros(12)
            du[j] = h
            f_plus = self.internal_force(elem_coords, u + du, rigidities, thickness)
            f_minus = self.internal_force(elem_coords, u - du, rigidities, thickness)
            K[:, j] = (f_plus - f_minus) / (2.0 * h)
        return K

    def stiffness(self, elem_coords, rigidities, thickness=1.0, **kwargs):
        """Initial (zero-displacement) tangent -- interface completeness,
        matching Beam2DCorotational.stiffness(). Regression-tested to
        match Beam3DEulerBernoulli.stiffness() on the same geometry."""
        return self.tangent_stiffness(elem_coords, np.zeros(12), rigidities, thickness)

    def mass(self, elem_coords, mass_props, thickness=1.0):
        """Reference-configuration consistent mass, delegated UNMODIFIED
        to Beam3DEulerBernoulli.mass() -- same convention every other
        nonlinear element in this package uses (no geometric correction
        to the mass itself), matching Beam2DCorotational's own choice."""
        return self._linear.mass(elem_coords, mass_props, thickness)
