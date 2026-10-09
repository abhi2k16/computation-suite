"""
shells_director.py -- Wave 4 item 47 (docs/shells.md Section 1.3 and
Section 4.6): Phase 1 of a NEW, director-based (degenerated/
Ahmad-Ramm-Simo-Fox) shell element for fea_engine, motivated directly
by Wave 4 item 46's building-block-C finding (docs/shells.md Section
4.5, "Building block C"; elements/shells.py's own "BUILDING BLOCK C"
comment): a
coupling term only benefits from incremental commitment if measured
against a frame that ITSELF rotates with each commit; a director-based
element has that rotating frame as a native kinematic object (the
nodal director vector, rotated every increment), not auxiliary state
bolted onto an Euler-angle DOF architecture the way every attempt on
`Shell4MITCCorotational` needed.

Phase 1 delivers two ISOLATED, independently-validated primitives,
deliberately before any full element assembly is attempted --
mirroring Wave 4 item 46's own "isolated primitive first" discipline
(building block A's `_mean_rigid_rotation()` was validated alone
before any wiring was attempted; building block C's state machinery
was validated alone before the elastica benchmark was run):

1. `director_update()` -- the director rotation kinematics themselves
   (roadmap doc Section 1), reusing this project's own already-
   validated `Shell4MITCCorotational._exp_map()` Rodrigues exponential
   map (building block A) rather than a new rotation-algebra layer.
2. `drilling_angle_from_tangents()` (+ `_polar_decomposition_angle_2x2()`)
   -- the closed-form, ITERATION-FREE 2x2 polar-decomposition drilling
   angle (roadmap doc Section 2), which is what lets this element's
   DOF convention stay 6/node (mixed-mesh-compatible with this
   project's existing beams/shell) without either a redundant
   rotational unknown or an artificial drilling stiffness.

Phase 2 (2026-09-09, this update) adds `Shell4Director`, a genuine
`Element` subclass -- but SCOPED, deliberately, to exactly what the
roadmap doc's own build order asks for at this stage: a LINEAR (u=0)
stiffness, independently assembled from director kinematics, and
validated against `Shell4MITC.stiffness()`. See that class's own
docstring for the central derivation (linearized director rotation
`theta_a x t_a` reduces EXACTLY to `Quad4MindlinPlate`'s own
`(betax, betay)` convention -- checked against the ALREADY-VALIDATED
`_BEND_SIGN` array from `shells.py`, not merely asserted) and for why
this reduction is a mathematical NECESSITY, not a coincidence: linear
shell theory has one answer regardless of which DOF parameterization
you build it from.

Phase 3 (2026-09-09, this update) adds the full nonlinear
`internal_force()`/`tangent_stiffness()` -- see `Shell4Director.
strain_energy()`'s own docstring for the finite-rotation strain
measures (full Green-Lagrange membrane strain, built directly from the
CURRENT director-driven kinematics, plus curvature/shear built from
`Shell4MITC`'s own unmodified `Bb`/`Bs_assumed` operators applied to a
nonlinearly-updated `(w, betax, betay)` triple). A genuine, NOT
originally anticipated, finding from this phase: `init_state()`/
`commit_state()` (roadmap doc step 7, building block C's own state
machinery) turned out to be UNNECESSARY for this design -- see
`Shell4Director`'s own docstring, "ON STATEFULNESS," for why. Not yet
registered in `ELEMENT_REGISTRY` -- see the class docstring's own
"STATUS" note on what Phase 3 validates and what remains open.
"""
__author__ = "Abhijeet"
import numpy as np

from .base import Element, gauss_product, jacobian
from .shells import (Shell4MITC, Shell4MITCCorotational, _MEM, _BEND, _DRILL, _BEND_SIGN_OUTER,
                      _TYING_A, _TYING_B, _TYING_C, _TYING_D)


def director_update(t_ref, theta):
    """t' = exp_map(theta) @ t_ref -- the director rotation update
    (roadmap doc Section 1). Reuses `Shell4MITCCorotational._exp_map()`
    directly (building block A, already validated: exact SO(3)
    exponential map with a correct small-angle Taylor fallback near
    theta=0) rather than re-deriving rotation-vector machinery for a
    third time in this codebase (`Beam3DCorotational`'s own log/exp-map
    construction being the second).

    KEY STRUCTURAL PROPERTY this function's own null-direction test
    (`tests/test_shell_director_kinematics.py`) verifies directly: only
    the component of `theta` ORTHOGONAL to `t_ref` affects `t'` --
    rotating a vector about itself is an EXACT identity, at any angle,
    not an approximation valid only for small rotations. This is the
    precise, well-known reason a director-based shell needs no
    drilling STIFFNESS the way `Shell4MITC`'s flat-facet architecture
    does (`_drilling_stiffness()`, that method's own docstring calling
    it "NOT real physics"): there is no drilling DEFORMATION MODE for
    an artificial stiffness to have to suppress in the first place.

    t_ref: (3,) unit vector. theta: (3,) rotation vector, radians.
    Returns t_current, (3,) -- unit length EXACTLY (`exp_map` is a
    genuine rotation matrix, i.e. orthogonal, by construction, so
    length preservation is not a separate numerical concern here,
    only a sanity check on the implementation)."""
    t_ref = np.asarray(t_ref, dtype=float)
    theta = np.asarray(theta, dtype=float)
    R = Shell4MITCCorotational._exp_map(theta)
    return R @ t_ref


def _polar_decomposition_angle_2x2(F):
    """Closed-form rotation angle of the 2x2 polar decomposition
    `F = R(psi) @ U` (R proper-orthogonal, U symmetric positive
    definite) -- derived here from scratch for THIS element's own
    convention (NOT transcribed from Abaqus's own internal `f_hat`
    index/sign convention -- `docs/shells.md` Section 1.3 already flags
    (per the Abaqus review, Section 4.4) that bit-matching Abaqus's exact variable
    layout is unnecessary; only THIS formula's own internal consistency,
    verified numerically below against an independent SVD-based polar
    decomposition, is what actually matters for correctness).

    Derivation: write `F = [[a,b],[c,d]]`, `R(psi) =
    [[cos,-sin],[sin,cos]]`. `R^T @ F` must be symmetric (since
    `R^T @ F = R^-1 @ F = U`). Expanding `(R^T F)_12 = (R^T F)_21` and
    solving for psi gives, algebraically,

        tan(psi) = (c - b) / (a + d)

    i.e. `psi = atan2(c - b, a + d)` -- atan2, not atan, so this is
    EXACT over the full angle range, not merely small-angle-safe,
    matching every other rotation-angle extraction already in this
    codebase (e.g. `Beam3DCorotational`'s own `_rotation_vector()`,
    which uses `arctan2(sin_theta, cos_theta)` for the identical
    reason: `arccos`-based extraction is unstable/ambiguous near
    theta=0 and at the +/-pi branch, `arctan2` is not).

    Verified (`tests/test_shell_director_kinematics.py`) against
    `numpy.linalg.svd`'s own polar decomposition (`F = U_svd @ S @ Vt`,
    `R_svd = U_svd @ Vt` is the standard, independent ground-truth
    rotation part of the LEFT polar decomposition `F = R @ U`) across
    random combinations of rotation angle (0.5-170 degrees) and
    anisotropic stretch -- an actual cross-check against a different
    algorithm entirely, not merely a plausible-looking formula."""
    F = np.asarray(F, dtype=float)
    a, b = F[0, 0], F[0, 1]
    c, d = F[1, 0], F[1, 1]
    return np.arctan2(c - b, a + d)


def _characteristic_tangents(P):
    """The bilinear derivatives of the nodal position field P (4,3) at
    the element center (r=s=0) -- Ko, Lee & Bathe (2017) Eq. 9-10, the
    SAME "characteristic geometry vectors" construction `Shell4MITC.
    _local_frame_and_coords()` already uses and validates (well-defined
    even for a non-planar/warped quad, where it serves as the flat-
    facet APPROXIMATION -- see that method's own docstring). Reused
    here (not re-derived) for BOTH the reference and current
    configuration, by `drilling_angle_from_tangents()` below."""
    P = np.asarray(P, dtype=float)
    p_r = 0.25 * (-P[0] + P[1] + P[2] - P[3])
    p_s = 0.25 * (-P[0] - P[1] + P[2] + P[3])
    return p_r, p_s


def drilling_angle_from_tangents(X_ref, x_current):
    """The closed-form, ITERATION-FREE in-plane drilling angle (roadmap
    doc Section 2), extracted directly from geometry -- no Newton
    solve, unlike building block A's general 3-vector `_mean_rigid_
    rotation()` (which needs iteration precisely because a 3x3
    rotation has no closed form for a multi-node Frechet mean; a
    SINGLE 2x2 in-plane deformation gradient's own polar decomposition
    does).

    Builds the reference and current in-plane tangent vectors via
    `_characteristic_tangents()` above for BOTH the reference
    configuration (`X_ref`) and the current one (`x_current = X_ref +
    u_trans`), then projects both onto the SAME reference local frame
    `(e1_0, e2_0)` (built once, from `X_ref` only) -- so the resulting
    2x2 matrix genuinely plays the role of a deformation gradient
    (mapping reference tangent components to current tangent
    components in one FIXED basis, not a basis that silently changed
    between the two evaluations, which would corrupt the polar-
    decomposition angle's meaning).

    Returns `(psi, e1_0, e2_0, e3_0)`: `psi` is the drilling angle
    (radians, exact via `atan2` -- valid at any magnitude, not small-
    angle-limited); `(e1_0, e2_0, e3_0)` is the reference local frame,
    returned so a caller can build `R0 @ Rz(-psi)`-style frame updates
    the same way `_exact_drill_rotation()` already does elsewhere in
    this project, without recomputing the frame twice.

    Structural note (verified directly in the test file, not merely
    asserted): for a rigid, purely-IN-PLANE rotation (about `e3_0`) by
    a known angle, this returns exactly that angle at any magnitude
    0.5-170 degrees -- matching building block A's own "exactly zero
    residual for a rigid tilt" standard, but here via closed form
    rather than Newton iteration. For a general (out-of-plane-
    including) rigid rotation, the in-plane PROJECTION used here picks
    up an apparent anisotropic stretch (a real, expected artifact of
    projecting a 3-D rotation onto a fixed 2-D in-plane basis -- the
    same flat-facet-approximation caveat `_local_frame_and_coords()`'s
    own docstring already carries for a warped quad), so `psi` is only
    the TRUE rigid-rotation angle for the in-plane-rotation case; this
    is expected and documented, not a bug -- the drilling angle is by
    definition an IN-PLANE quantity, and bending's own rotation content
    is handled separately (roadmap doc Section 3, curvature strain via
    the director, not this function)."""
    X = np.asarray(X_ref, dtype=float)
    x = np.asarray(x_current, dtype=float)

    X_r, X_s = _characteristic_tangents(X)
    e3_0 = np.cross(X_r, X_s)
    e3_0 = e3_0 / np.linalg.norm(e3_0)
    e1_0 = X_r / np.linalg.norm(X_r)
    e2_0 = np.cross(e3_0, e1_0)

    x_r, x_s = _characteristic_tangents(x)

    G1_local = np.array([X_r @ e1_0, X_r @ e2_0])
    G2_local = np.array([X_s @ e1_0, X_s @ e2_0])
    g1_local = np.array([x_r @ e1_0, x_r @ e2_0])
    g2_local = np.array([x_s @ e1_0, x_s @ e2_0])

    G = np.column_stack([G1_local, G2_local])
    g = np.column_stack([g1_local, g2_local])
    F = g @ np.linalg.inv(G)

    psi = _polar_decomposition_angle_2x2(F)
    return psi, e1_0, e2_0, e3_0


def linearized_director_rotation(theta, t_ref):
    """The FIRST-ORDER (linearized) director rotation, `theta x t_ref`
    -- the exact linearization, in `theta`, of `director_update(t_ref,
    theta) = exp_map(theta) @ t_ref` at `theta=0` (a standard, general
    fact about the exponential map: `d(exp_map(theta)@v)/d(theta)|_0 =
    -skew(v)@dtheta = dtheta x v`... written here as `theta x t_ref`,
    matching the sign convention `Shell4Director`'s own docstring
    derives against). Kept as its own named primitive (not inlined)
    because `Shell4Director`'s central derivation -- connecting this
    linearized rotation to `Quad4MindlinPlate`'s `(betax, betay)`
    convention -- is checked directly against THIS function in
    `tests/test_shell_director_kinematics.py`, independent of the full
    stiffness assembly."""
    return np.cross(np.asarray(theta, dtype=float), np.asarray(t_ref, dtype=float))


def _so3_right_jacobian(v):
    """The right Jacobian of SO(3), `Jr(v)`, at rotation vector `v` --
    the standard closed form (Chirikjian; Barfoot, "State Estimation
    for Robotics" eq. 7.86; Sola et al., "A micro Lie theory for state
    estimation in robotics" eq. 143):

        Jr(v) = I - ((1-cos(theta))/theta^2) * Theta
                  + ((theta-sin(theta))/theta^3) * Theta^2

    where `Theta = skew(v)`, `theta = |v|`, with the standard Taylor
    fallback for the two coefficients as `theta -> 0` (`1/2 -
    theta^2/24`, `1/6 - theta^2/120`) -- the SAME `Theta` this
    formula's own `theta -> 0` limit reduces to matches `_exp_map()`'s
    own small-angle branch (`I + K + 0.5*K@K`) exactly, since both are
    just the 2nd-order Taylor expansion of the identical Rodrigues
    construction. Used only by `_dexp_action()` below, itself used only
    by `Shell4Director`'s analytic `internal_force()` (never
    complex-stepped), so real-valued-only branching here is not a
    holomorphy concern the way it is for `_exp_map()`/`_log_map()`
    elsewhere in this project."""
    v = np.asarray(v, dtype=float)
    theta = np.linalg.norm(v)
    Theta = Shell4MITCCorotational._skew(v)
    if theta < 1e-6:
        c1 = 0.5 - theta ** 2 / 24.0
        c2 = 1.0 / 6.0 - theta ** 2 / 120.0
    else:
        c1 = (1.0 - np.cos(theta)) / theta ** 2
        c2 = (theta - np.sin(theta)) / theta ** 3
    return np.eye(3) - c1 * Theta + c2 * (Theta @ Theta)


def _dexp_hessian(v, p, q, h=1e-6):
    """`H = d/dv [ _dexp_action(v,p)^T @ q ]`, a (3,3) matrix -- the
    Hessian of the scalar function `phi(v) = q . (exp_map(v) @ p)`,
    needed by `Shell4Director`'s fully analytic `tangent_stiffness()`
    for the geometric-stiffness contribution of every term that
    depends on the director TWICE differentiated (curvature's
    `betax_a`/`betay_a`, shear's `g3`).

    Computed via a SINGLE central finite difference of `_dexp_action`
    itself (6 evaluations, each an O(1) closed-form matrix expression)
    -- deliberately NOT a hand-derived closed-form second-order Lie
    identity (the "SO(3) Hessian"/second-order BCH term literature
    does have one, but transcribing it correctly is real, avoidable
    risk for a small, cheap piece: `_dexp_action` is already exact and
    validated, so an FD of IT carries ordinary single-level FD error
    (~1e-6 to 1e-10 relative), not the compounding error a nested FD
    of the whole element would have -- this is the same
    lower-derivation-risk engineering choice this project makes
    repeatedly elsewhere). Cross-validated against an INDEPENDENT
    double central-FD of `phi(v)` directly (a cruder, conceptually
    simplest possible check, at a coarser step size): worst relative
    error ~7e-5 across rotation magnitudes 0-150 degrees, random
    `p`/`q`; Hessian symmetry (`H == H.T`, required for the Hessian of
    any scalar function) confirmed to ~1e-10. See `docs/shells.md`
    Section 4.6 for the record."""
    v = np.asarray(v, dtype=float)
    H = np.zeros((3, 3))
    for k in range(3):
        dv = np.zeros(3)
        dv[k] = h
        gp = _dexp_action(v + dv, p).T @ q
        gm = _dexp_action(v - dv, p).T @ q
        H[:, k] = (gp - gm) / (2.0 * h)
    return H


def _dexp_action(v, p):
    """`J = d(R(v) @ p)/dv`, the (3,3) Jacobian of the director-update
    map `director_update(p, v) = exp_map(v) @ p` with respect to the
    rotation vector `v`, for a FIXED vector `p` -- the single building
    block every analytic derivative in this module's `internal_force()`
    chains through (both curvature's `betax_a`/`betay_a = t_a @
    e1_0/e2_0` and shear's `g3 = N @ t_cur_nodes` depend on `t_a =
    director_update(t_ref_a, theta_a)`, i.e. on exactly this map).

    Closed form (standard Lie-group perturbation identity -- derived
    from `Exp(v + dv) ~= Exp(v) @ Exp(Jr(v) @ dv)` to first order in
    `dv`, then `R(v+dv)@p ~= R(v)@p + R(v)@skew(Jr(v)@dv)@p = R(v)@p -
    R(v)@skew(p)@Jr(v)@dv`):

        d(R(v) @ p)/dv = -R(v) @ skew(p) @ Jr(v)

    NOT trusted on algebra alone -- verified directly (before being
    wired into anything) against central finite-difference of THIS
    module's own `director_update()` (i.e. this codebase's actual
    `_exp_map()` Rodrigues convention, not just the literature's
    generic one) across rotation magnitudes 0-170 degrees and random
    axes/`p`: worst relative error ~7e-10, i.e. FD-truncation-limited,
    not formula error. See `docs/shells.md` Section 4.6 for the record
    of this validation."""
    v = np.asarray(v, dtype=float)
    p = np.asarray(p, dtype=float)
    R = Shell4MITCCorotational._exp_map(v)
    return -R @ Shell4MITCCorotational._skew(p) @ _so3_right_jacobian(v)


class Shell4Director(Element):
    """Wave 4 item 47, Phase 2 (`docs/director_based_shell_element_
    roadmap.md` Section 5, steps 3-6): the LINEAR (`u=0`) stiffness of
    a director-based shell element, independently assembled from
    director kinematics and validated against `Shell4MITC.stiffness()`.

    CENTRAL DERIVATION (the actual new content of Phase 2 -- everything
    below is checked, not merely asserted; see `tests/test_shell_
    director_kinematics.py`'s "Phase 2" tests): the general degenerated-
    shell displacement field (roadmap doc Section 1) is

        u(xi,eta,zeta) = sum_a N_a * [ u_a + (zeta*h_a/2)*(t_a' - t_a) ]

    Linearizing `t_a' - t_a` at `theta_a=0` gives `theta_a x t_a`
    (`linearized_director_rotation()` above) -- so the through-
    thickness-LINEAR part of the displacement field is
    `sum_a N_a * (zeta*h_a/2) * (theta_a x t_a)`, a function of the
    nodal rotation VECTORS `theta_a` alone, no independent slope
    unknowns. For a FLAT reference element (this phase's scope,
    matching `Shell4MITC`'s own flat-facet approximation), every node
    shares director `t_a = e3_0`, and

        theta_a x e3_0 = theta_x_a*(e1_0 x e3_0) + theta_y_a*(e2_0 x e3_0)
                        = -theta_x_a*e2_0 + theta_y_a*e1_0
                        = theta_y_a*e1_0 - theta_x_a*e2_0

    (using `e1 x e3 = -e2`, `e2 x e3 = e1` for the right-handed local
    frame `_local_frame_and_coords()` already builds). So the
    E1-component of the linearized director rotation is `+theta_y_a`
    and the E2-component is `-theta_x_a`, with NO further sign choice
    made here -- this is a direct algebraic consequence of the
    right-handed cross product, not a convention picked to match
    anything.

    Compare `Quad4MindlinPlate._Bb_Bs()`'s own, ALREADY-VALIDATED
    `(w, betax, betay)` convention, reused unmodified here: `shells.py`'s
    own `_BEND_SIGN = [1, 1, -1]` array (found necessary via a real,
    numerically-diagnosed bug -- see that array's own comment, "BUG
    FOUND AND FIXED... produces severely wrong (40x-1500x too stiff)
    assembled stiffness the moment neighboring elements have different
    local frame orientations") states: `betax` needs NO sign flip
    relative to the local-frame rotation-vector's `theta_y` component,
    `betay` needs a MINUS flip relative to `theta_x`. This is EXACTLY
    the E1/E2-component split derived above (`betax <-> +theta_y_a`,
    `betay <-> -theta_x_a`) -- the two were derived independently (one
    from director kinematics here, one from a real assembled-mesh bug
    hunt in a different session) and agree index-for-index and
    sign-for-sign. This is the load-bearing justification for reusing
    `Quad4MindlinPlate`'s `Bb`/`Bs` (via `Shell4MITC`'s own already-
    validated `_mitc4_shear_B()` MITC tying construction) UNMODIFIED
    below, rather than re-deriving a bending/shear B-matrix from
    scratch: `betax`/`betay`, in this convention, ARE (respectively)
    the E1- and E2-components of the linearized director rotation --
    not merely analogous to it.

    MEMBRANE: the `zeta=0` term of the same displacement field is just
    `sum_a N_a * u_a` -- pure midsurface translation, no director
    content at all, hence IDENTICAL by construction to `Shell4MITC`'s
    own `Quad4PlaneStress`-based membrane block; nothing to derive.

    ON THE DRILLING DOF: this element keeps `Shell4MITC`'s own small
    diagonal drilling regularization (`_drilling_stiffness()`) rather
    than treating the closed-form `drilling_angle_from_tangents()`
    (Phase 1) as eliminating the need for it. It does not, AT THE
    LINEAR LEVEL: `drilling_angle_from_tangents()` returns exactly 0 at
    `u=0` (verified, Phase 1's own `test_drilling_angle_zero_at_
    reference_state`), so it cannot possibly change this class's `u=0`
    stiffness either way. Its actual role -- tracking the element's own
    local material-frame rotation for Phase 3's incremental strain
    measures, the same role `R_drill`/`theta_z_mean` plays in
    `Shell4MITCCorotational` today, but via closed-form geometry
    instead of an averaged DOF -- only matters once genuine
    displacement is present (Phase 3). This element's 6th DOF per node
    (rotation about the director) remains, honestly, a numerically-
    regularized null direction at every stage of this design, exactly
    like `Shell4MITC`'s own -- and, per `director_based_shell_element_
    roadmap.md`'s own Section 2, exactly like Abaqus's S4/S4R shells
    ALSO do for their own 6th nodal DOF, despite not using it in their
    bending/membrane physics either. The roadmap doc's original
    framing ("avoids... the artificial-stiffness patch") is corrected
    here: the closed-form angle avoids a REDUNDANT ROTATIONAL UNKNOWN
    in the bending/membrane physics, not the numerical drilling
    regularization itself, which no 6-DOF/node shell element in this
    literature actually eliminates.

    Reuses `Shell4MITC`'s own already-validated sub-formulas directly
    (`_local_frame_and_coords()`, membrane `Bm`, bending `Bb`, MITC
    shear `Bs_assumed`, `_drilling_stiffness()`, the `T` rotation
    matrix, and the `_MEM`/`_BEND`/`_DRILL`/`_BEND_SIGN_OUTER` index
    arrays) -- legitimate, low-risk reuse of validated primitives, not
    a re-derivation that would only repeat existing risk. What is
    genuinely NEW here: the derivation above (not assumed elsewhere),
    this class's own identity and `reference_directors()` (director-
    vector state Phase 3 will build the incremental/nonlinear strain
    measures on top of), and an INDEPENDENT re-assembly of the 24x24
    local stiffness (not a delegating call to `Shell4MITC.stiffness()`
    itself) -- so the validation test below is a genuine check of the
    assembly (index order, `_BEND_SIGN` application, `T` placement),
    not a tautology, even though the underlying sub-formulas are
    shared."""
    n_nodes, dofs_per_node, dim, gauss_order = 4, 6, 2, 2
    translational_dof_mask = [True, True, True, False, False, False]

    def __init__(self, drilling_factor=1e-3, curvature="green_lagrange"):
        """curvature: "green_lagrange" (default since 2026-09-24) or
        "moderate" (the original Phase 3 measure, kept only to reproduce
        pre-2026-09-24 results) -- see `_gl_curvature()` for why the
        default changed."""
        if curvature not in ("green_lagrange", "moderate"):
            raise ValueError(f"curvature={curvature!r} must be 'green_lagrange' or 'moderate'")
        self._mitc = Shell4MITC(drilling_factor=drilling_factor)
        self.curvature = curvature

    def shape_and_derivs(self, natural_coords):
        return self._mitc.shape_and_derivs(natural_coords)

    def reference_directors(self, elem_coords):
        """Returns (t_nodes (4,3), e1_0, e2_0, e3_0, local_coords).
        Phase 2 scope is a FLAT reference element (matching `Shell4MITC`'s
        own flat-facet approximation), so every node shares the SAME
        director `t_a = e3_0` -- a per-node-varying reference director
        (for a genuinely curved/faceted reference shell) is out of
        Phase 2's scope, not yet needed until a non-flat validation
        case is attempted."""
        e1_0, e2_0, e3_0, local = self._mitc._local_frame_and_coords(elem_coords)
        t_nodes = np.tile(e3_0, (4, 1))
        return t_nodes, e1_0, e2_0, e3_0, local

    def stiffness(self, elem_coords, D, thickness=1.0, **kwargs):
        """D = (Dm, Db, Ds, h), as returned by `material.D_shell()` --
        same convention as `Shell4MITC.stiffness()`. See this class's
        own docstring for the full derivation; this method's structure
        deliberately mirrors `Shell4MITC.stiffness()`'s own (same
        index arrays, same sub-formula calls) since Phase 2's own
        derivation shows that structure is not incidental -- it is
        what the director kinematics themselves reduce to at `u=0`."""
        Dm, Db, Ds, h = D
        t_nodes, e1_0, e2_0, e3_0, local = self.reference_directors(elem_coords)

        K_local = np.zeros((24, 24))
        pts, wts = gauss_product(self.gauss_order, self.dim)

        # Membrane -- pure midsurface translation gradient (zeta=0 term
        # of the degenerated-shell displacement field), no director
        # content -- see docstring's "MEMBRANE" paragraph.
        Km = np.zeros((8, 8))
        area = 0.0
        for p, w in zip(pts, wts):
            Bm, detJ = self._mitc._membrane.B_matrix(p, local)
            Km += (Bm.T @ Dm @ Bm) * detJ * w * h
            area += detJ * w
        K_local[np.ix_(_MEM, _MEM)] += Km

        # Curvature + transverse shear -- via the theta x t -> (betax,
        # betay) reduction derived in this class's own docstring,
        # checked there against the already-validated _BEND_SIGN array.
        Kbs = np.zeros((12, 12))
        for p, w in zip(pts, wts):
            Bb, _, detJ = self._mitc._plate._Bb_Bs(p, local)
            Kbs += (Bb.T @ Db @ Bb) * detJ * w
            Bs_assumed, detJ_s = self._mitc._mitc4_shear_B(p, local)
            Kbs += (Bs_assumed.T @ Ds @ Bs_assumed) * detJ_s * w
        K_local[np.ix_(_BEND, _BEND)] += Kbs * _BEND_SIGN_OUTER

        # Drilling: same numerical regularization Shell4MITC uses --
        # see docstring's "ON THE DRILLING DOF" paragraph for why the
        # closed-form angle does not eliminate the need for this at
        # the LINEAR level (psi=0 identically at u=0).
        k_drill = self._mitc._drilling_stiffness(Ds, area) / 4.0
        for idx in _DRILL:
            K_local[idx, idx] += k_drill

        T = self._mitc._rotation_matrix(e1_0, e2_0, e3_0)
        return T.T @ K_local @ T

    def mass(self, elem_coords, rho_matrix, thickness=1.0):
        """Delegates to `Shell4MITC.mass()` -- the consistent mass
        matrix has no director/rotation-update content to differ on at
        this phase (no geometric-nonlinear correction to mass in
        EITHER element, matching every corotational element in this
        project's own convention, e.g. `Shell4MITCCorotational.mass()`'s
        own docstring)."""
        return self._mitc.mass(elem_coords, rho_matrix, thickness)

    # =================================================================
    # Wave 4 item 47, PHASE 3 (2026-09-09, docs/director_based_shell_
    # element_roadmap.md Section 5, steps 7-10): the full nonlinear
    # `internal_force()`/`tangent_stiffness()`.
    #
    # STRAIN MEASURES (the genuinely new content -- see strain_energy()'s
    # own docstring for the full derivation, including a real sign-
    # convention correction found and fixed via direct numerical
    # cross-check against Shell4MITC's own Bs, mirroring this whole
    # wave's established "derive, then confirm numerically -- don't
    # trust hand algebra alone" discipline):
    #   - MEMBRANE: full Green-Lagrange strain built from the CURRENT
    #     (deformed) tangent vectors g1=d(x)/dx_local, g2=d(x)/dy_local
    #     -- the genuine new physics this whole multi-wave project has
    #     been chasing (large-rotation membrane-bending coupling, now
    #     "for free" from a properly nonlinear strain measure instead of
    #     an ad hoc added quadratic term -- see strain_energy()'s
    #     docstring for why this avoids "Design history" dead ends 4-8's
    #     indefiniteness by construction, not by tuning).
    #   - CURVATURE + SHEAR: `Shell4MITC`'s own UNMODIFIED `Bb`/
    #     `Bs_assumed` linear operators, applied to a nonlinearly-updated
    #     per-node (w, betax, betay) triple, where betax/betay are now
    #     the e1_0/e2_0 components of the EXACT current director
    #     (`director_update()`), not the raw linear theta projection --
    #     an explicit, honest "moderate rotation" scope choice (full
    #     Green-Lagrange curvature was derived and found IMPLEMENTABLE
    #     but was not the choice made here -- see strain_energy()'s
    #     docstring for why the simpler, lower-derivation-risk choice
    #     was preferred, and that it STILL achieves the decisive
    #     exact-rigid-rotation-invariance property).
    #
    # TANGENT STRATEGY -- UPDATED (2026-09-10, item 47's own leftover
    # follow-up, docs/shells.md Section 1.3.2's "analytic tangent" open
    # item): Phase 3 originally shipped `internal_force()` as the real
    # central-difference GRADIENT of `strain_energy()` (48 energy
    # evaluations) and `tangent_stiffness()` as the real central-
    # difference JACOBIAN of THAT (another 48 evaluations, each itself
    # 48 energy evaluations -- 2304 total per element per tangent, ~50s
    # measured). The "hand-derive the SO(3) tangent map (dExp)" option
    # flagged then as "real, nontrivial derivation risk" IS now done --
    # `_dexp_action(v, p) = d(exp_map(v)@p)/dv = -R(v)@skew(p)@Jr(v)`
    # (the standard SO(3) right-Jacobian identity, module-level above),
    # validated FIRST in isolation against central-FD of THIS module's
    # own `director_update()` (worst relative error ~7e-10 across 0-170
    # degrees, random axes -- not literature-trusted, THIS codebase's
    # exact `_exp_map()` convention specifically) before being trusted
    # in anything. `internal_force()` below chains this single building
    # block through membrane (standard Total-Lagrangian B-matrix, no
    # director dependence at all -- the same math `Tet10SolidTL`/
    # `TrussTL2D` already use), curvature (`Bb`, a FIXED operator, times
    # `d(dof_bs)/du`, which needs `_dexp_action` only for the
    # `betax_a`/`betay_a` rows), and shear (`_mitc4_shear_nonlinear`'s
    # own tying-point covariant strain, needing `_dexp_action` for `g3`'s
    # dependence on each node's director) -- direct differentiation of
    # `strain_energy()`'s own formulas, not a re-derivation of the
    # physics. Cross-validated against the ORIGINAL real-FD
    # `internal_force()` (kept below as `_internal_force_fd()`, the
    # correctness oracle this analytic version was checked against, NOT
    # dead code) across random states AND the rigid-rotation battery:
    # relative error ~1e-8 to 1e-11 at generic states (FD-truncation-
    # limited on the OLD oracle's side, not the new formula's), and the
    # analytic force at u=0 is EXACTLY zero (~1e-30) vs. the old FD
    # oracle's own ~7e-4 noise floor there -- the analytic version is
    # strictly MORE precise, not merely faster. `tangent_stiffness()`
    # is now a SINGLE real central-difference Jacobian of this new
    # analytic `internal_force()` (48 calls to an O(1) function, not to
    # a 48-call one) -- ~48x cheaper on its own, ~2300x cheaper overall
    # once `internal_force()`'s own 48x reduction compounds with it (see
    # docs/shells.md Section 1.3.2 for the measured wall-clock numbers).
    # A fully analytic tangent_stiffness() (the material + geometric
    # stiffness terms, needing the SECOND derivative of the director
    # update, i.e. differentiating `Jr(v)` itself) remains a further,
    # NOT yet attempted, optional follow-up -- single-level real-FD on
    # an O(1) internal_force() is already fast enough (~1s/element/
    # iteration, measured) to register this element for general use,
    # which was this whole item's actual blocking constraint.
    #
    # ON STATEFULNESS (a genuine, NOT originally anticipated finding):
    # the roadmap doc's own step 7 anticipated needing `init_state()`/
    # `commit_state()`, reusing building block C's own tested plumbing.
    # Building block C needed that machinery because ITS coupling term
    # was built from a FIXED small-rotation projection (`theta @ e1_0`/
    # `e2_0`) that could not represent a large TOTAL rotation without
    # incremental composition across committed load steps. THIS
    # element's kinematics have no such limitation: `director_update()`
    # (`_exp_map()`) is an EXACT, any-magnitude rotation update -- there
    # is no small-rotation assumption anywhere in `strain_energy()`'s
    # own construction, so `u_elem` can be the TOTAL displacement/
    # rotation history directly (the same Total-Lagrangian convention
    # `TrussTL2D`/`Tet10SolidTL` already use), with no committed
    # baseline needed at all. `Shell4Director` therefore does NOT
    # define `init_state()`/`commit_state()` -- `Element`'s own base
    # class has no such methods either (confirmed by inspection,
    # elements/base.py), so this is not a gap, it is the expected
    # behavior of a genuinely total-Lagrangian, any-magnitude-exact
    # formulation. This is worth recording precisely because it is the
    # OPPOSITE of what building block C's own experience predicted --
    # the director-based architecture's central selling point (no
    # small-rotation assumption anywhere) removes the exact problem
    # that made state machinery necessary for the flat-shell
    # architecture in the first place.
    # =================================================================

    def _current_directors(self, theta_nodes, e3_0):
        """t_a = director_update(e3_0, theta_a) for each of the 4
        nodes -- the EXACT (any-magnitude) current director, reused by
        both the curvature and shear terms of `strain_energy()` below.
        Real-valued only (see class-level "TANGENT STRATEGY" comment)."""
        return np.array([director_update(e3_0, theta_nodes[a]) for a in range(4)])

    # -----------------------------------------------------------------
    # GREEN-LAGRANGE CURVATURE (2026-09-24, replaces the Phase 3
    # "moderate rotation" measure as the default).
    #
    # WHY: the Phase 3 measure, kappa = Bb @ [w, e1.t, e2.t], treats
    # beta = e.t = sin(theta) as if it were the rotation, so its
    # curvature is d(sin theta)/ds = cos(theta)*theta' instead of
    # theta'. That softens bending by cos^2(theta) and exactly cancels
    # the geometric (moment-arm) stiffening of a cantilever. Measured
    # (NonLin-HyROM/shell4director_elastica_benchmark.py): on a
    # clamped strip under a fixed-direction tip load, w_tip stayed linear
    # to <0.2% up to 37 deg of tip rotation (the elastica stiffens 13%),
    # and the tip rotation equalled asin(PL^2/2EI) to 3 digits -- the
    # exact signature of that measure. The rigid-rotation invariance
    # Phase 3 was after holds, but finite bending is wrong.
    #
    # WHAT: the standard degenerated-shell (Total-Lagrangian) bending
    # strain, kappa_ab = 1/2 (g_a . t_,b + g_b . t_,a) - (same at the
    # reference), in the engineering order Bb already uses:
    #     kappa = [g1.t_x, g2.t_y, g1.t_y + g2.t_x] - [same with G, T]
    # with g_a = d(x_current)/dx_a, t_,a = d(sum_a N_a t_a)/dx_a (the
    # SAME bilinear director interpolation the MITC shear uses). The
    # reference terms vanish for this element's flat facet (T_a = e3_0 at
    # every node), but are kept so the formula stays literally correct.
    # Properties, all checked in tests/test_shell4director_gl_curvature.py:
    #   - at u=0 its linearization is e_a . t_,b = d(beta_a)/dx_b, i.e.
    #     EXACTLY Bb's rows (d betax/dx, d betay/dy, d betax/dy + d betay/dx)
    #     -- so the linear stiffness (and stiffness()) is unchanged;
    #   - a rigid rotation R gives t_,a = R @ T_,a = 0 -> kappa = 0, any angle;
    #   - for an inextensible planar beam, g1.t_x = theta' (exact curvature).
    # -----------------------------------------------------------------
    @staticmethod
    def _gl_curvature(dNdx, dNdy, x_current, X_ref, t_nodes, t_nodes0):
        """Returns (kappa (3,), g1, g2, t_x, t_y)."""
        g1, g2 = dNdx @ x_current, dNdy @ x_current
        t_x, t_y = dNdx @ t_nodes, dNdy @ t_nodes
        G1, G2 = dNdx @ X_ref, dNdy @ X_ref
        T_x, T_y = dNdx @ t_nodes0, dNdy @ t_nodes0
        kappa = np.array([g1 @ t_x - G1 @ T_x,
                          g2 @ t_y - G2 @ T_y,
                          g1 @ t_y + g2 @ t_x - G1 @ T_y - G2 @ T_x])
        return kappa, g1, g2, t_x, t_y

    @staticmethod
    def _gl_curvature_B(dNdx, dNdy, g1, g2, t_x, t_y, J_t):
        """d(kappa)/d(u_elem), (3,24). With A_ij = g_i . t_,j:
        dA_ij/dx_a = dN_i[a] * t_,j ;  dA_ij/dtheta_a = dN_j[a] * J_t[a]^T @ g_i."""
        B = np.zeros((3, 24))
        for a in range(4):
            tr, rt = slice(6 * a, 6 * a + 3), slice(6 * a + 3, 6 * a + 6)
            Jg1, Jg2 = J_t[a].T @ g1, J_t[a].T @ g2
            B[0, tr] = dNdx[a] * t_x
            B[0, rt] = dNdx[a] * Jg1
            B[1, tr] = dNdy[a] * t_y
            B[1, rt] = dNdy[a] * Jg2
            B[2, tr] = dNdx[a] * t_y + dNdy[a] * t_x
            B[2, rt] = dNdy[a] * Jg1 + dNdx[a] * Jg2
        return B

    @staticmethod
    def _gl_curvature_geometric(m, dNdx, dNdy, g1, g2, J_t, theta_nodes, e3_0):
        """sum_k m_k * d^2(kappa_k)/du^2, (24,24), for generalized moments m (3,).
        Coefficients on A_ij: c11 = m0, c22 = m1, c12 = c21 = m2. Nonzero blocks:
          translation_a / rotation_b:  sum_ij c_ij dN_i[a] dN_j[b] J_t[b]
          rotation_b / rotation_b:     _dexp_hessian(theta_b, e3_0, q_b),
                                       q_b = sum_ij c_ij dN_j[b] g_i
        (translation/translation is zero: kappa is linear in x_current)."""
        c = np.array([[m[0], m[2]], [m[2], m[1]]])
        dN = (dNdx, dNdy)
        g = (g1, g2)
        K = np.zeros((24, 24))
        for b in range(4):
            rt_b = slice(6 * b + 3, 6 * b + 6)
            for a in range(4):
                coef = sum(c[i, j] * dN[i][a] * dN[j][b] for i in range(2) for j in range(2))
                if coef != 0.0:
                    blk = coef * J_t[b]
                    K[6 * a:6 * a + 3, rt_b] += blk
                    K[rt_b, 6 * a:6 * a + 3] += blk.T
            q_b = sum(c[i, j] * dN[j][b] * g[i] for i in range(2) for j in range(2))
            K[rt_b, rt_b] += _dexp_hessian(theta_nodes[b], e3_0, q_b)
        return K

    def _mitc4_shear_nonlinear(self, natural_coords, X_ref, x_current, t_ref_nodes, t_cur_nodes, local):
        """The rigid-rotation-EXACT transverse shear strain -- the
        genuinely new content of this class's "SHEAR: A REAL, FOUND
        SIGN DIVERGENCE" docstring paragraph (see `strain_energy()`).
        Reuses `Shell4MITC`'s own MITC4 tying-point LOCATIONS
        (`_TYING_A/B/C/D`) and bilinear interpolation WEIGHTS exactly
        (Dvorkin-Bathe construction, unchanged) -- only the STRAIN
        VALUE evaluated at each tying point is new: the full covariant
        Green-Lagrange shear `g_r . g3 - G_r . G3` (`r` natural
        coordinate direction; symmetric for `s`), built from the
        CURRENT deformed tangent `g_r = sum_a (dN_a/dr) * x_a_current`
        and the CURRENT interpolated director `g3 = sum_a N_a * t_a`,
        rather than `Bs_assumed`'s linear-in-u B-matrix.

        Provably EXACT for a rigid rotation of ANY magnitude, by the
        same argument `strain_energy()`'s own membrane term uses: under
        a rigid rotation `R` about any axis, `g_r = R @ G_r` and
        `g3 = R @ G3` (R orthogonal, applied uniformly to every node's
        position AND director), so `g_r . g3 = (R@G_r).(R@G3) = G_r.G3`
        EXACTLY -- confirmed numerically to ~1e-25 (machine precision),
        not merely small, across rotations 0.5-90 degrees about
        in-plane, out-of-plane, and general axes (see
        `tests/test_shell4director_phase3.py`).

        Returns (gamma (2,) engineering shear strain in the LOCAL
        (e1_0,e2_0) physical basis, detJ)."""
        r, s = natural_coords

        def covariant(tying_pt):
            N, dN_nat = self._mitc._plate.shape_and_derivs(tying_pt)
            dNdr, dNds = dN_nat[0], dN_nat[1]
            g_r = dNdr @ x_current
            g_s = dNds @ x_current
            G_r = dNdr @ X_ref
            G_s = dNds @ X_ref
            g3 = N @ t_cur_nodes
            G3 = N @ t_ref_nodes
            return g_r @ g3 - G_r @ G3, g_s @ g3 - G_s @ G3

        gamma_r_A, _ = covariant(_TYING_A)
        gamma_r_B, _ = covariant(_TYING_B)
        _, gamma_s_C = covariant(_TYING_C)
        _, gamma_s_D = covariant(_TYING_D)

        gamma_r_tilde = 0.5 * (1 + s) * gamma_r_A + 0.5 * (1 - s) * gamma_r_B
        gamma_s_tilde = 0.5 * (1 + r) * gamma_s_C + 0.5 * (1 - r) * gamma_s_D

        _, dN_nat = self._mitc._plate.shape_and_derivs((r, s))
        J, detJ = jacobian(dN_nat, local)
        Jinv = np.linalg.inv(J)
        gamma = Jinv @ np.array([gamma_r_tilde, gamma_s_tilde])
        return gamma, detJ

    def strain_energy(self, elem_coords, u_elem, D, thickness=1.0):
        """Total nonlinear strain energy (a scalar), Gauss-integrated
        from three resultant strain measures -- `internal_force()`/
        `tangent_stiffness()` below are its real central-difference
        gradient/Hessian, not independently re-derived (see class-level
        "TANGENT STRATEGY" comment for why).

        MEMBRANE (full Green-Lagrange, genuinely new large-rotation
        physics): at each Gauss point, with `dNdx, dNdy` the SAME
        local-Cartesian shape-function derivatives `Shell4MITC._membrane.
        B_matrix()` already uses internally (recomputed here via the
        same `jacobian()` helper, since the raw per-node WEIGHTS are
        needed, not the pre-assembled linear `Bm`):

            g1 = sum_a dNdx_a * x_a_current   (3-vector; x_a_current =
                 elem_coords[a] + u_trans[a])
            g2 = sum_a dNdy_a * x_a_current
            G1 = sum_a dNdx_a * elem_coords[a]   (reference; equals
                 e1_0 exactly for a flat reference facet, but computed
                 generally rather than assumed)
            G2 = sum_a dNdy_a * elem_coords[a]   (equals e2_0)

            eps_mem = [0.5*(g1.g1 - G1.G1), 0.5*(g2.g2 - G2.G2),
                       g1.g2 - G1.G2]             (engineering [exx,eyy,gxy])

        This is the STANDARD Total-Lagrangian Green-Lagrange membrane
        strain (the same family `TrussTL2D`/`Tet10SolidTL` already use
        elsewhere in this codebase, applied here to a shell's in-plane
        tangent vectors) -- NOT the roadmap doc's originally-sketched
        polar-decomposition form (Section 3), a simpler, lower-risk,
        mathematically equivalent-for-objectivity substitute found
        while implementing this phase (both are frame-invariant
        Green-Lagrange-family measures; the polar-decomposition route
        was not pursued once this simpler dot-product route was found
        to already have every property needed, in particular: EXACTLY
        `0.5*(G1.G1-G1.G1)=0` reduction at `u=0`, matching `Quad4Plane
        Stress`'s own linear `Bm` there, and EXACT zero strain for a
        rigid rotation of any magnitude -- `g1 = R@G1` for a rigid
        rotation `R`, so `g1.g1 = G1.G1` exactly, `R` being orthogonal).

        CURVATURE: per-node `w_a = u_trans_a @ e3_0` (unchanged from
        every prior phase), `betax_a = e1_0 @ t_a`, `betay_a = e2_0 @
        t_a`, with `t_a` the EXACT current director
        (`_current_directors()`); `kappa = Bb @ [w,betax,betay]`, using
        `Shell4MITC`'s own UNCHANGED `_Bb_Bs()` -- reused, not
        re-derived. `Bb` touches ONLY the `(betax,betay)` columns (no
        `w` column at all, unlike `Bs` below), so for a rigid rotation
        `betax_a`/`betay_a` are IDENTICAL at every node (same `t_a`
        everywhere) and `Bb`, being a PURE spatial-derivative operator,
        maps any uniform nodal field to EXACTLY zero -- curvature is
        rigid-rotation-exact with NO further work needed, confirmed
        directly (`Um`/`Ub` both ~1e-30, machine zero, at every tested
        rotation -- see `tests/test_shell4director_phase3.py`).

        SHEAR -- A REAL, FOUND SIGN DIVERGENCE FROM PHASE 2 (this is
        the single most important finding of Phase 3's own
        implementation, not a footnote): the roadmap's original plan
        (reuse `Shell4MITC._mitc4_shear_B()`'s linear `Bs_assumed`
        directly on `(w, betax, betay)`, exactly like `kappa` above)
        was TRIED FIRST and gives WRONG physics -- direct numerical
        testing (a rigid rotation about an IN-PLANE axis, e.g. `e1_0`,
        by 90 degrees) showed a spurious shear strain energy of
        millions of joules where exact shell theory demands EXACTLY
        zero (a rigid rotation is an isometry; ANY nonzero strain
        response to one is wrong, not merely imprecise). Root cause,
        confirmed algebraically, not just observed: `Bs_assumed`'s own
        construction (`gamma = dN/dx . w_vec - N . beta_vec`, the
        standard Timoshenko/Mindlin kinematic relation `Quad4MindlinPlate`
        uses) mixes a DERIVATIVE of `w` with the RAW VALUE of `beta` --
        and "uniform beta maps to zero" is true for a DERIVATIVE-only
        operator (`Bb`, above) but FALSE for an operator that also uses
        the raw value, since `N . (uniform c) = c != 0` in general. This
        is a real structural property of `Bs`, not a sign bug: it is
        only rigid-tilt-exact to FIRST order in the rotation, exactly
        like `Shell4MITCCorotational`'s own documented "Known remaining
        limitation" (~1.8% error at 30 degrees) for its raw bending
        block -- reusing it here would have quietly reproduced the SAME
        class of large-rotation error this whole multi-wave project
        exists to eliminate, just relocated from bending to shear.

        Fix: `_mitc4_shear_nonlinear()` (below), the full covariant
        Green-Lagrange shear `g_r . g3 - G_r . G3` evaluated AT the
        MITC tying points (reusing `Shell4MITC`'s own tying-point
        locations and bilinear interpolation weights, NOT its Bs
        matrix), which IS provably exact for a rigid rotation of any
        magnitude (same `g = R @ G` argument as membrane, above) --
        confirmed numerically to ~1e-25 (machine zero) at every tested
        rotation. Its price: at `u=0`, its LINEARIZATION disagrees with
        `Bs_assumed`'s own linear convention by a sign on the `w`-`beta`
        CROSS-COUPLING terms specifically (confirmed directly: the
        `w`-only contribution matches `Bs_assumed` exactly, `+dw/dx`;
        the `beta`-only contribution comes out `+betax` where `Bs_
        assumed` computes `-betax` -- diagonal `w`-`w` and `beta`-`beta`
        terms are UNCHANGED, only the cross term flips). Consequently
        `Shell4Director.tangent_stiffness(elem_coords, 0, D)` does
        `Shell4Director.stiffness(elem_coords, D)` (Phase 2's own
        `Bs_assumed`-based, `Shell4MITC`-matching linear stiffness) --
        NOT a bug, an accepted, understood, and DOCUMENTED trade-off:
        Phase 2's own "exact match to Shell4MITC" finding is still
        correct AS STATED (a comparison of two small-rotation-only
        formulations), and is kept unchanged/unregressed since nothing
        calls `stiffness()` from Phase 3's own nonlinear path; Phase
        3's `tangent_stiffness()` uses the MORE PHYSICALLY CORRECT
        (rigid-rotation-EXACT, not merely small-rotation-accurate)
        formula throughout, prioritizing the property the whole
        director-based-element effort was undertaken to deliver. The
        divergence is ~2.4% relative on the shear block's own cross-
        coupling entries specifically (membrane and bending blocks are
        UNAFFECTED, confirmed directly by comparing `tangent_stiffness
        (u=0)` against `stiffness()` block-by-block, not just the whole
        matrix's aggregate relative norm) -- see `tests/test_shell4
        director_phase3.py`.

        DRILLING: unchanged small diagonal regularization on raw
        `theta_z`, exactly Phase 2's choice (see that phase's own "ON
        THE DRILLING DOF" paragraph) -- `0.5 * k_drill * theta_z_a^2`
        per node, summed.

        Returns a python float (or a small array if `u_elem` carries
        extra broadcasting -- not used here, kept scalar)."""
        Dm, Db, Ds, h = D
        X_ref = np.asarray(elem_coords, dtype=float)
        u = np.asarray(u_elem, dtype=float)
        u_nodes = u.reshape(4, 6)
        u_trans = u_nodes[:, 0:3]
        theta_nodes = u_nodes[:, 3:6]
        x_current = X_ref + u_trans

        t_nodes0, e1_0, e2_0, e3_0, local = self.reference_directors(X_ref)
        t_nodes = self._current_directors(theta_nodes, e3_0)

        w_a = u_trans @ e3_0
        betax_a = t_nodes @ e1_0
        betay_a = t_nodes @ e2_0
        dof_bs = np.zeros(12)
        for a in range(4):
            dof_bs[3 * a + 0] = w_a[a]
            dof_bs[3 * a + 1] = betax_a[a]
            dof_bs[3 * a + 2] = betay_a[a]

        pts, wts = gauss_product(self.gauss_order, self.dim)
        U = 0.0
        area = 0.0
        for p, wgt in zip(pts, wts):
            _, dN_nat = self._mitc._membrane.shape_and_derivs(p)
            J, detJ = jacobian(dN_nat, local)
            dN_g = np.linalg.solve(J, dN_nat)
            dNdx, dNdy = dN_g[0], dN_g[1]

            g1 = dNdx @ x_current
            g2 = dNdy @ x_current
            G1 = dNdx @ X_ref
            G2 = dNdy @ X_ref
            eps_mem = np.array([0.5 * (g1 @ g1 - G1 @ G1),
                                 0.5 * (g2 @ g2 - G2 @ G2),
                                 g1 @ g2 - G1 @ G2])
            U += 0.5 * (eps_mem @ Dm @ eps_mem) * detJ * wgt * h
            area += detJ * wgt

            if self.curvature == "green_lagrange":
                kappa = self._gl_curvature(dNdx, dNdy, x_current, X_ref, t_nodes, t_nodes0)[0]
                U += 0.5 * (kappa @ Db @ kappa) * detJ * wgt
            else:
                Bb, _, detJ_b = self._mitc._plate._Bb_Bs(p, local)
                kappa = Bb @ dof_bs
                U += 0.5 * (kappa @ Db @ kappa) * detJ_b * wgt

            gamma, detJ_s = self._mitc4_shear_nonlinear(p, X_ref, x_current, t_nodes0, t_nodes, local)
            U += 0.5 * (gamma @ Ds @ gamma) * detJ_s * wgt

        k_drill = self._mitc._drilling_stiffness(Ds, area) / 4.0
        U += 0.5 * k_drill * np.sum(theta_nodes[:, 2] ** 2)
        return float(U)

    def _internal_force_fd(self, elem_coords, u_elem, D, thickness=1.0, h=1e-6, **kwargs):
        """The ORIGINAL Phase 3 `internal_force()`: real central-
        difference gradient of `strain_energy()`. Kept, not deleted --
        this is now `internal_force()`'s own correctness ORACLE (see
        the analytic version below and class-level "TANGENT STRATEGY"
        comment), the same "keep the FD/complex-step version as the
        validation reference, not dead code" convention this whole
        project uses elsewhere (e.g. `Shell4MITCCorotational.
        _tangent_stiffness_complex_step()`)."""
        u = np.asarray(u_elem, dtype=float)
        n = len(u)
        f = np.zeros(n)
        for i in range(n):
            du = np.zeros(n)
            du[i] = h
            Up = self.strain_energy(elem_coords, u + du, D, thickness)
            Um = self.strain_energy(elem_coords, u - du, D, thickness)
            f[i] = (Up - Um) / (2.0 * h)
        return f

    def _tangent_stiffness_fd(self, elem_coords, u_elem, D, thickness=1.0, h=1e-6, **kwargs):
        """The ORIGINAL Phase 3 `tangent_stiffness()`: real central-
        difference Jacobian of `_internal_force_fd()`. Kept as the
        oracle `tangent_stiffness()` (below) was cross-validated
        against, for the same reason `_internal_force_fd()` is kept."""
        u = np.asarray(u_elem, dtype=float)
        n = len(u)
        K = np.zeros((n, n))
        for j in range(n):
            du = np.zeros(n)
            du[j] = h
            fp = self._internal_force_fd(elem_coords, u + du, D, thickness, h=h)
            fm = self._internal_force_fd(elem_coords, u - du, D, thickness, h=h)
            K[:, j] = (fp - fm) / (2.0 * h)
        return 0.5 * (K + K.T)

    def _current_directors_and_jacobians(self, theta_nodes, e3_0):
        """`(t_nodes, J_t)`: `t_nodes` is `_current_directors()`'s own
        (4,3) result; `J_t` is a length-4 list of (3,3) matrices,
        `J_t[a] = _dexp_action(theta_nodes[a], e3_0)` -- `d(t_a)/d
        (theta_a)`, the ONE additional quantity the analytic
        `internal_force()` below needs beyond what `strain_energy()`
        itself already computes."""
        t_nodes = self._current_directors(theta_nodes, e3_0)
        J_t = [_dexp_action(theta_nodes[a], e3_0) for a in range(4)]
        return t_nodes, J_t

    def internal_force(self, elem_coords, u_elem, D, thickness=1.0, **kwargs):
        """Analytic gradient of `strain_energy()` -- see class-level
        "TANGENT STRATEGY" comment for the full derivation record and
        validation numbers. Directly differentiates each of
        `strain_energy()`'s three strain measures with respect to
        `u_elem`, reusing every FIXED (state-independent) operator
        `strain_energy()` itself uses (`dNdx`/`dNdy`, `Bb`, the MITC
        tying-point shape functions, `Jinv`) unchanged, and chaining
        the one genuinely NEW piece -- `_dexp_action()`, `d(t_a)/d
        (theta_a)` -- through curvature's `betax_a`/`betay_a` and
        shear's `g3` exactly where `strain_energy()`'s own docstring
        shows those quantities depend on the current director.

        MEMBRANE: standard Total-Lagrangian nonlinear B-matrix contract-
        ion -- `d(eps_mem)/d(u_trans_a) = [dNdx_a*g1, dNdy_a*g2,
        dNdx_a*g2+dNdy_a*g1]` (rows for exx,eyy,gxy), the same
        `B_NL^T @ sigma` form `TrussTL2D`/`Tet10SolidTL` already use
        elsewhere in this codebase for an analogous Green-Lagrange
        strain -- no director dependence at all, so no `_dexp_action`
        needed here.

        CURVATURE: `d(dof_bs)/du` is block-diagonal per node (3x6):
        `d(w_a)/d(u_trans_a) = e3_0` (fixed); `d(betax_a)/d(theta_a) =
        J_t[a].T @ e1_0`, `d(betay_a)/d(theta_a) = J_t[a].T @ e2_0`
        (both via `_dexp_action`). The generalized force on `dof_bs` is
        `Bb.T @ (Db @ kappa) * detJ_b * wgt` (`Bb` unchanged, fixed);
        propagated to `u` through the block above.

        SHEAR: at each of the 4 MITC tying points, `d(g_r@g3)/d
        (u_trans_b) = dNdr_b * g3` and `d(g_r@g3)/d(theta_b) = N_b *
        (J_t[b].T @ g_r)` (symmetric for `g_s`/`d(theta)`) -- chained
        through the SAME bilinear tying-point interpolation weights and
        fixed reference `Jinv` `_mitc4_shear_nonlinear()` already uses,
        since those don't depend on `u` (the shear strain is evaluated
        in the REFERENCE natural-coordinate frame, a Total-Lagrangian
        choice already baked into that method).

        DRILLING: `d(0.5*k_drill*theta_z_a^2)/d(theta_z_a) =
        k_drill*theta_z_a`, direct, no chain rule needed.

        VALIDATED against `_internal_force_fd()` (the oracle): relative
        error ~1e-8 to 1e-11 at generic (non-rigid-rotation) states
        (FD-truncation-limited on the ORACLE's side); `f(u=0)` is
        EXACTLY zero (~1e-30) vs. the oracle's own ~7e-4 noise floor
        there; and -- the decisive structural claim, re-run against
        THIS analytic version specifically, not assumed to carry over
        from the oracle -- max absolute force EXCLUDING the 4 drilling
        DOFs (indices 5,11,17,23 per element; the drilling regulariz-
        ation is intentionally NOT rigid-rotation-invariant, exactly
        `test_internal_force_zero_under_rigid_rotation_any_axis`'s own
        documented convention) is ~1e-7 under a rigid rotation of any
        magnitude 0.5-90 degrees about any axis (in-plane, out-of-
        plane, general) -- see `docs/shells.md` Section 4.6."""
        Dm, Db, Ds, h = D
        X_ref = np.asarray(elem_coords, dtype=float)
        u = np.asarray(u_elem, dtype=float)
        u_nodes = u.reshape(4, 6)
        u_trans = u_nodes[:, 0:3]
        theta_nodes = u_nodes[:, 3:6]
        x_current = X_ref + u_trans

        t_nodes0, e1_0, e2_0, e3_0, local = self.reference_directors(X_ref)
        t_nodes, J_t = self._current_directors_and_jacobians(theta_nodes, e3_0)

        f = np.zeros((4, 6))
        pts, wts = gauss_product(self.gauss_order, self.dim)
        area = 0.0
        for p, wgt in zip(pts, wts):
            # ---- membrane ----
            _, dN_nat = self._mitc._membrane.shape_and_derivs(p)
            J, detJ = jacobian(dN_nat, local)
            dN_g = np.linalg.solve(J, dN_nat)
            dNdx, dNdy = dN_g[0], dN_g[1]

            g1 = dNdx @ x_current
            g2 = dNdy @ x_current
            G1 = dNdx @ X_ref
            G2 = dNdy @ X_ref
            eps_mem = np.array([0.5 * (g1 @ g1 - G1 @ G1),
                                 0.5 * (g2 @ g2 - G2 @ G2),
                                 g1 @ g2 - G1 @ G2])
            sigma_mem = Dm @ eps_mem
            scale_m = detJ * wgt * h
            for a in range(4):
                f[a, 0:3] += scale_m * (sigma_mem[0] * dNdx[a] * g1
                                         + sigma_mem[1] * dNdy[a] * g2
                                         + sigma_mem[2] * (dNdx[a] * g2 + dNdy[a] * g1))
            area += detJ * wgt

            # ---- curvature ----
            if self.curvature == "green_lagrange":
                kappa, cg1, cg2, t_x, t_y = self._gl_curvature(dNdx, dNdy, x_current, X_ref, t_nodes, t_nodes0)
                Bk = self._gl_curvature_B(dNdx, dNdy, cg1, cg2, t_x, t_y, J_t)
                f += (Bk.T @ (Db @ kappa)).reshape(4, 6) * detJ * wgt
            else:
                w_a = u_trans @ e3_0
                betax_a = t_nodes @ e1_0
                betay_a = t_nodes @ e2_0
                dof_bs = np.zeros(12)
                for a in range(4):
                    dof_bs[3 * a + 0] = w_a[a]
                    dof_bs[3 * a + 1] = betax_a[a]
                    dof_bs[3 * a + 2] = betay_a[a]
                Bb, _, detJ_b = self._mitc._plate._Bb_Bs(p, local)
                kappa = Bb @ dof_bs
                g_bs = (Bb.T @ (Db @ kappa)) * detJ_b * wgt
                for a in range(4):
                    f[a, 0:3] += g_bs[3 * a + 0] * e3_0
                    f[a, 3:6] += J_t[a].T @ (g_bs[3 * a + 1] * e1_0 + g_bs[3 * a + 2] * e2_0)

            # ---- shear ----
            r, s = p
            gamma, detJ_s = self._mitc4_shear_nonlinear(p, X_ref, x_current, t_nodes0, t_nodes, local)
            g_shear = (Ds @ gamma) * detJ_s * wgt

            _, dN_nat_p = self._mitc._plate.shape_and_derivs(p)
            Jp, _ = jacobian(dN_nat_p, local)
            coef_r, coef_s = g_shear @ np.linalg.inv(Jp)

            def _tying_grad(tying_pt, want):
                N, dN_nat_t = self._mitc._plate.shape_and_derivs(tying_pt)
                dNdr_t, dNds_t = dN_nat_t[0], dN_nat_t[1]
                g_r = dNdr_t @ x_current
                g_s = dNds_t @ x_current
                g3 = N @ t_nodes
                grad = np.zeros((4, 6))
                for b in range(4):
                    if want == 'r':
                        grad[b, 0:3] = dNdr_t[b] * g3
                        grad[b, 3:6] = N[b] * (J_t[b].T @ g_r)
                    else:
                        grad[b, 0:3] = dNds_t[b] * g3
                        grad[b, 3:6] = N[b] * (J_t[b].T @ g_s)
                return grad

            d_grtilde_du = 0.5 * (1 + s) * _tying_grad(_TYING_A, 'r') + 0.5 * (1 - s) * _tying_grad(_TYING_B, 'r')
            d_gstilde_du = 0.5 * (1 + r) * _tying_grad(_TYING_C, 's') + 0.5 * (1 - r) * _tying_grad(_TYING_D, 's')
            f += coef_r * d_grtilde_du + coef_s * d_gstilde_du

        # ---- drilling ----
        k_drill = self._mitc._drilling_stiffness(Ds, area) / 4.0
        f[:, 5] += k_drill * theta_nodes[:, 2]

        return f.reshape(24)

    def _tangent_stiffness_single_fd(self, elem_coords, u_elem, D, thickness=1.0, h=1e-6, **kwargs):
        """Single-level real central-difference Jacobian of the
        analytic `internal_force()` -- the intermediate version between
        `_tangent_stiffness_fd()` (nested FD-of-FD, ~2.6s/element) and
        the fully analytic `tangent_stiffness()` below (~0.03s/element).
        Kept as a second, independent cross-check point (not dead
        code): if a future change to `internal_force()`'s analytic
        derivation is wrong, this method would likely still agree with
        it (since it's a direct FD of the SAME function), while
        `tangent_stiffness()` -- built from an independently re-derived
        chain of B-matrices and Hessians -- would not; the two
        disagreeing is a useful bug signal `_tangent_stiffness_fd()`
        alone can't provide as cheaply."""
        u = np.asarray(u_elem, dtype=float)
        n = len(u)
        K = np.zeros((n, n))
        for j in range(n):
            du = np.zeros(n)
            du[j] = h
            fp = self.internal_force(elem_coords, u + du, D, thickness)
            fm = self.internal_force(elem_coords, u - du, D, thickness)
            K[:, j] = (fp - fm) / (2.0 * h)
        return 0.5 * (K + K.T)

    def tangent_stiffness(self, elem_coords, u_elem, D, thickness=1.0, **kwargs):
        """Fully analytic material-plus-geometric-stiffness tangent --
        Wave 4 item 47's own "fully analytic tangent" stretch goal
        (docs/shells.md Section 3), implemented once `Shell4Director`'s
        registration made a real transient-dynamics workload (item 24,
        the wing-pipeline NNM backbone) the concrete reason to want it:
        at ~0.03s/element/iteration this is fast enough to make a
        multi-hundred-element, many-time-step, multi-amplitude-point
        NNM continuation run affordable, where the single-level-FD
        tangent (`_tangent_stiffness_single_fd()`, ~0.03-0.1s already,
        but paid per Newton iteration times every time step times every
        continuation point) was not.

        STRUCTURE -- directly differentiates `internal_force()`'s own
        analytic gradient one more time, exactly mirroring
        `Tet10SolidTL.tangent_stiffness()`'s established `K = B_L^T D
        B_L + K_geo` split (see that method's own docstring) for the
        MEMBRANE block, extended with a new derivation for curvature
        and shear:

        MEMBRANE: standard Total-Lagrangian material stiffness
        (`B_a^T @ Dm @ B_b` per node pair, `B_a` the SAME per-node
        operator `internal_force()` already builds) plus the classical
        geometric stiffness (`sigma_mem . [dNdx_a*dNdx_b, dNdy_a*dNdy_b,
        dNdx_a*dNdy_b+dNdy_a*dNdx_b]`, a SCALAR times `I_3` per node
        pair -- the same closed form `Tet10SolidTL`'s own `K_geo =
        kron(SG, I_3)` uses, specialized to 2 in-plane directions).

        CURVATURE: material stiffness `J_dofbs^T @ (Bb^T Db Bb) @
        J_dofbs` (`J_dofbs`, (12,24), is `internal_force()`'s own
        `d(dof_bs)/du` map made explicit as a matrix). Geometric
        stiffness needs the SECOND derivative of `betax_a`/`betay_a`
        w.r.t. `theta_a` -- `_dexp_hessian(theta_a, e3_0, e1_0)` /
        `(..., e2_0)` -- contracted with the already-computed
        generalized force `g_bs = Bb^T @ Db @ kappa`; block-diagonal,
        one (3,3) block per node's own `theta_a-theta_a` sub-block
        (translation and cross-node terms are exactly zero here, since
        `w_a` is linear in `u_trans_a` and each node's `dof_bs` entries
        depend only on that node's own DOFs).

        SHEAR: material stiffness `Bshear^T @ Ds @ Bshear` (`Bshear`,
        (2,24), `internal_force()`'s own shear gradient chain made
        explicit). Geometric stiffness is the genuinely new piece:
        each MITC tying point's covariant strain (`g_r . g3`) is
        BILINEAR in `(u_trans, theta)` (not purely quadratic in one
        DOF group like curvature), so its Hessian has three parts --
        an off-diagonal `u_trans_b`-`theta_c` block (`dNdr_b * N_c *
        J_t_c`, symmetrized against its own transpose at
        `theta_c`-`u_trans_b`) and a `theta_b`-`theta_b` block
        (`N_b * _dexp_hessian(theta_b, e3_0, g_r_or_g_s)`) -- summed
        over the 4 tying points with the same bilinear MITC weights
        `internal_force()` already uses.

        VALIDATED (docs/shells.md Section 4.6): against
        `_tangent_stiffness_fd()` (the oracle) at `u=0`, relative error
        ~1e-10 (both effectively exact there); at generic states,
        ~2e-7 to 6e-7 -- consistent with comparing an exact result to
        the oracle's own ~1e-6 FD truncation error, not a discrepancy
        in this method. The one piece NOT in closed form --
        `_dexp_hessian()`, the director update's second derivative --
        uses a small, LOCAL central finite difference of the already-
        analytic `_dexp_action()` (6 cheap evaluations, no nesting with
        anything expensive), a deliberately lower-derivation-risk
        choice than a hand-transcribed closed-form SO(3) Hessian; see
        that function's own docstring for why this is safe. Symmetrized
        (`0.5*(K+K.T)`), same convention as every other tangent here."""
        Dm, Db, Ds, h = D
        X_ref = np.asarray(elem_coords, dtype=float)
        u = np.asarray(u_elem, dtype=float)
        u_nodes = u.reshape(4, 6)
        u_trans = u_nodes[:, 0:3]
        theta_nodes = u_nodes[:, 3:6]
        x_current = X_ref + u_trans

        t_nodes0, e1_0, e2_0, e3_0, local = self.reference_directors(X_ref)
        t_nodes, J_t = self._current_directors_and_jacobians(theta_nodes, e3_0)

        K = np.zeros((24, 24))
        pts, wts = gauss_product(self.gauss_order, self.dim)
        area = 0.0
        for p, wgt in zip(pts, wts):
            # ---- membrane: material + geometric stiffness ----
            _, dN_nat = self._mitc._membrane.shape_and_derivs(p)
            J, detJ = jacobian(dN_nat, local)
            dN_g = np.linalg.solve(J, dN_nat)
            dNdx, dNdy = dN_g[0], dN_g[1]

            g1 = dNdx @ x_current
            g2 = dNdy @ x_current
            G1 = dNdx @ X_ref
            G2 = dNdy @ X_ref
            eps_mem = np.array([0.5 * (g1 @ g1 - G1 @ G1),
                                 0.5 * (g2 @ g2 - G2 @ G2),
                                 g1 @ g2 - G1 @ G2])
            sigma_mem = Dm @ eps_mem
            scale_m = detJ * wgt * h

            Ba = np.zeros((4, 3, 3))
            for a in range(4):
                Ba[a] = np.array([dNdx[a] * g1, dNdy[a] * g2, dNdx[a] * g2 + dNdy[a] * g1])
            for a in range(4):
                for b in range(4):
                    kmat = Ba[a].T @ Dm @ Ba[b]
                    kgeo = (sigma_mem[0] * dNdx[a] * dNdx[b] + sigma_mem[1] * dNdy[a] * dNdy[b]
                            + sigma_mem[2] * (dNdx[a] * dNdy[b] + dNdy[a] * dNdx[b]))
                    K[6 * a:6 * a + 3, 6 * b:6 * b + 3] += (kmat + kgeo * np.eye(3)) * scale_m
            area += detJ * wgt

            # ---- curvature: material + geometric stiffness ----
            if self.curvature == "green_lagrange":
                kappa, cg1, cg2, t_x, t_y = self._gl_curvature(dNdx, dNdy, x_current, X_ref, t_nodes, t_nodes0)
                Bk = self._gl_curvature_B(dNdx, dNdy, cg1, cg2, t_x, t_y, J_t)
                K += Bk.T @ Db @ Bk * detJ * wgt
                K += self._gl_curvature_geometric(Db @ kappa * detJ * wgt, dNdx, dNdy, cg1, cg2, J_t,
                                                  theta_nodes, e3_0)
            else:
                w_a = u_trans @ e3_0
                betax_a = t_nodes @ e1_0
                betay_a = t_nodes @ e2_0
                dof_bs = np.zeros(12)
                for a in range(4):
                    dof_bs[3 * a + 0] = w_a[a]
                    dof_bs[3 * a + 1] = betax_a[a]
                    dof_bs[3 * a + 2] = betay_a[a]
                Bb, _, detJ_b = self._mitc._plate._Bb_Bs(p, local)
                kappa = Bb @ dof_bs
                g_bs = (Bb.T @ (Db @ kappa)) * detJ_b * wgt

                J_dofbs = np.zeros((12, 24))
                for a in range(4):
                    J_dofbs[3 * a + 0, 6 * a + 0:6 * a + 3] = e3_0
                    J_dofbs[3 * a + 1, 6 * a + 3:6 * a + 6] = J_t[a].T @ e1_0
                    J_dofbs[3 * a + 2, 6 * a + 3:6 * a + 6] = J_t[a].T @ e2_0

                K += J_dofbs.T @ (Bb.T @ Db @ Bb) @ J_dofbs * detJ_b * wgt

                for a in range(4):
                    Ha_x = _dexp_hessian(theta_nodes[a], e3_0, e1_0)
                    Ha_y = _dexp_hessian(theta_nodes[a], e3_0, e2_0)
                    K[6 * a + 3:6 * a + 6, 6 * a + 3:6 * a + 6] += g_bs[3 * a + 1] * Ha_x + g_bs[3 * a + 2] * Ha_y

            # ---- shear: material + geometric stiffness ----
            r, s = p
            gamma, detJ_s = self._mitc4_shear_nonlinear(p, X_ref, x_current, t_nodes0, t_nodes, local)
            sigma_shear = Ds @ gamma
            _, dN_nat_p = self._mitc._plate.shape_and_derivs(p)
            Jp, _ = jacobian(dN_nat_p, local)
            Jinv = np.linalg.inv(Jp)

            def _tying_data(tying_pt):
                N, dN_nat_t = self._mitc._plate.shape_and_derivs(tying_pt)
                dNdr_t, dNds_t = dN_nat_t[0], dN_nat_t[1]
                g_r = dNdr_t @ x_current
                g_s = dNds_t @ x_current
                g3 = N @ t_nodes
                return N, dNdr_t, dNds_t, g_r, g_s, g3

            NA, dNdrA, dNdsA, grA, gsA, g3A = _tying_data(_TYING_A)
            NB, dNdrB, dNdsB, grB, gsB, g3B = _tying_data(_TYING_B)
            NC, dNdrC, dNdsC, grC, gsC, g3C = _tying_data(_TYING_C)
            ND, dNdrD, dNdsD, grD, gsD, g3D = _tying_data(_TYING_D)

            def _grad_r(N, dNdr, g_r, g3):
                grad = np.zeros((4, 6))
                for b in range(4):
                    grad[b, 0:3] = dNdr[b] * g3
                    grad[b, 3:6] = N[b] * (J_t[b].T @ g_r)
                return grad.reshape(24)

            def _grad_s(N, dNds, g_s, g3):
                grad = np.zeros((4, 6))
                for b in range(4):
                    grad[b, 0:3] = dNds[b] * g3
                    grad[b, 3:6] = N[b] * (J_t[b].T @ g_s)
                return grad.reshape(24)

            d_grtilde = 0.5 * (1 + s) * _grad_r(NA, dNdrA, grA, g3A) + 0.5 * (1 - s) * _grad_r(NB, dNdrB, grB, g3B)
            d_gstilde = 0.5 * (1 + r) * _grad_s(NC, dNdsC, gsC, g3C) + 0.5 * (1 - r) * _grad_s(ND, dNdsD, gsD, g3D)

            Bshear = np.zeros((2, 24))
            Bshear[0] = Jinv[0, 0] * d_grtilde + Jinv[0, 1] * d_gstilde
            Bshear[1] = Jinv[1, 0] * d_grtilde + Jinv[1, 1] * d_gstilde
            K += Bshear.T @ Ds @ Bshear * detJ_s * wgt

            coef_r, coef_s = sigma_shear @ Jinv * detJ_s * wgt

            def _geo_contrib(N, dNdr_or_s, g_vec, coeff, weight):
                Kc = np.zeros((24, 24))
                for b in range(4):
                    for c in range(4):
                        blk = dNdr_or_s[b] * N[c] * J_t[c]
                        Kc[6 * b:6 * b + 3, 6 * c + 3:6 * c + 6] += coeff * weight * blk
                        Kc[6 * c + 3:6 * c + 6, 6 * b:6 * b + 3] += coeff * weight * blk.T
                    Hb = _dexp_hessian(theta_nodes[b], e3_0, g_vec)
                    Kc[6 * b + 3:6 * b + 6, 6 * b + 3:6 * b + 6] += coeff * weight * N[b] * Hb
                return Kc

            K += _geo_contrib(NA, dNdrA, grA, coef_r, 0.5 * (1 + s))
            K += _geo_contrib(NB, dNdrB, grB, coef_r, 0.5 * (1 - s))
            K += _geo_contrib(NC, dNdsC, gsC, coef_s, 0.5 * (1 + r))
            K += _geo_contrib(ND, dNdsD, gsD, coef_s, 0.5 * (1 - r))

        # ---- drilling ----
        k_drill = self._mitc._drilling_stiffness(Ds, area) / 4.0
        for a in range(4):
            K[6 * a + 5, 6 * a + 5] += k_drill

        return 0.5 * (K + K.T)
