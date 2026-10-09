"""
contact.py -- gap/contact elements: GapContactPenalty (flat-wall,
fixed normal), GapContactCurvedFriction (updating normal + Coulomb
friction, curved obstacle), and (Wave 3 items 16-17, docs/
consolidated_future_roadmap.md) NodeToSegmentContact2D /
NodeToSegmentContact2DFriction -- real node-to-segment contact between
a slave node and a MASTER SEGMENT made of two ordinary mesh nodes, so
both sides of a contact pair can be genuinely deformable (unlike the
two elements above, whose obstacle is always an analytically-
prescribed fixed plane/circle), plus find_contact_pairs_2d(), a simple
broad-phase helper for discovering candidate slave-node/master-segment
pairs.

Split out of the original monolithic element.py during the
fea_engine restructuring; no logic changed, only file location (this
docstring paragraph and everything below the historical GapContact*
classes is new, Wave 3).
"""
__author__ = "Abhijeet"
import numpy as np

from .base import Element


class GapContactPenalty(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 1, 2, 1, 1

    @staticmethod
    def _penetration(u_elem, mat):
        k_p, g0, n_hat = mat
        n_hat = np.asarray(n_hat, dtype=float)
        delta = n_hat @ u_elem - g0   # >0 means penetrating the obstacle
        return delta, n_hat, k_p

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """Unilateral spring: zero force while delta<=0 (gap open or
        just touching), f = k_p*delta*n_hat once delta>0 -- the
        gradient of the one-sided potential U = (1/2) k_p delta^2 for
        delta>0 (else 0), so this is virtual-work consistent with
        tangent_stiffness() below by construction, the same way
        TrussTL2D's/TrussPlastic2D's force/tangent pairs are."""
        delta, n_hat, k_p = self._penetration(u_elem, mat)
        if delta <= 0.0:
            return np.zeros(2)
        return (k_p * delta) * n_hat

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """K_T = k_p * n_hat (x) n_hat while active, else the zero
        matrix -- literally 'zero stiffness apart, large stiffness on
        contact,' the status switch the whole module is named for.
        NOTE: K_T is discontinuous (not just its derivative) at
        delta=0, so this element's response is only PIECEWISE smooth;
        Newton-Raphson still converges fine as long as load steps
        don't try to jump clean over the transition in one step (see
        validate_contact.py, which uses enough steps to avoid this)."""
        delta, n_hat, k_p = self._penetration(u_elem, mat)
        if delta <= 0.0:
            return np.zeros((2, 2))
        return k_p * np.outer(n_hat, n_hat)

    def stiffness(self, elem_coords, mat, thickness=1.0, **kwargs):
        """Initial (zero-displacement) tangent -- interface completeness,
        matching TrussTL2D.stiffness()/TrussPlastic2D.stiffness()."""
        return self.tangent_stiffness(elem_coords, np.zeros(2), mat, thickness)


# =====================================================================
# 3-node constant-strain triangle (CST), Module 12: the linear-simplex
# counterpart to Quad4PlaneStress -- needed because a general (e.g.
# Gmsh-generated, unstructured) mesh of an arbitrary 2-D shape is
# triangulated, not quadrilateral. IMPORTANT: this element deliberately
# does NOT use the base class's generic stiffness()/mass() Gauss loop.
# gauss_product()/the module's whole quadrature engine is a TENSOR-
# PRODUCT rule over the square [-1,1]^dim -- correct for Quad4/Hex8,
# but the WRONG domain for a triangle (whose natural coordinates live
# on the simplex {xi>=0, eta>=0, xi+eta<=1}). Sampling a tensor-product
# rule's points against these shape functions would evaluate N outside
# the valid simplex region for many points (negative shape function
# values -- not a real quadrature error, a domain mismatch). Instead:
# since the shape functions are LINEAR, the strain (and hence B) is
# CONSTANT over the whole element, so the "integral" is exactly one
# evaluation times the physical area -- no quadrature loop needed at
# all, closed form, same idea as Beam2DEulerBernoulli's closed-form
# Hermite stiffness.
# =====================================================================

class GapContactCurvedFriction(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 1, 2, 1, 1

    @staticmethod
    def init_state():
        """No prior tangential 'stick' anchor -- the first contact
        establishes it fresh (zero initial tangential stretch)."""
        return {"s_stick": None}

    @staticmethod
    def _trial(elem_coords, u_elem, mat, state):
        """mat = (k_p, k_t, mu, center, R). Returns a dict describing
        the TRIAL response at u_elem given the last-committed state --
        does not mutate state (see commit_state())."""
        k_p, k_t, mu, center, R = mat
        center = np.asarray(center, dtype=float)
        d_vec = elem_coords[0] + u_elem - center
        d = np.linalg.norm(d_vec)
        n_hat = d_vec / d
        t_hat = np.array([-n_hat[1], n_hat[0]])   # +90 deg from n_hat (increasing theta)
        gn = d - R

        if gn >= 0.0:
            return dict(pn=0.0, tt=0.0, n_hat=n_hat, t_hat=t_hat,
                        s_stick_new=state["s_stick"], active=False)

        pn = k_p * (-gn)                          # contact pressure magnitude, >= 0
        theta = np.arctan2(d_vec[1], d_vec[0])
        s = R * theta                              # tangential arc-length coordinate
        s_stick_n = state["s_stick"]
        if s_stick_n is None:
            s_stick_n = s                          # fresh contact: zero initial stretch
        e_t_trial = s - s_stick_n
        tt_trial = k_t * e_t_trial
        limit = mu * pn
        if abs(tt_trial) <= limit:
            tt, s_stick_new = tt_trial, s_stick_n                   # stick
        else:
            tt = limit * np.sign(tt_trial)
            s_stick_new = s - tt / k_t                              # slip (return-mapped)
        return dict(pn=pn, tt=tt, n_hat=n_hat, t_hat=t_hat,
                    s_stick_new=s_stick_new, active=True)

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, state=None):
        if state is None:
            state = self.init_state()
        r = self._trial(elem_coords, u_elem, mat, state)
        return -r["pn"] * r["n_hat"] + r["tt"] * r["t_hat"]

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, state=None):
        """Central-difference tangent of internal_force() -- see the
        class docstring for why this is computed numerically here."""
        if state is None:
            state = self.init_state()
        h = 1e-7 * max(1.0, float(np.linalg.norm(u_elem)))
        K = np.zeros((2, 2))
        for j in range(2):
            du = np.zeros(2); du[j] = h
            fp = self.internal_force(elem_coords, u_elem + du, mat, thickness, state=state)
            fm = self.internal_force(elem_coords, u_elem - du, mat, thickness, state=state)
            K[:, j] = (fp - fm) / (2 * h)
        return K

    def commit_state(self, elem_coords, u_elem, mat, state, thickness=1.0):
        """Called once per CONVERGED load step. Resets the stick anchor
        to None on separation, so a future re-contact starts fresh
        rather than remembering a stale tangential position."""
        r = self._trial(elem_coords, u_elem, mat, state)
        return {"s_stick": r["s_stick_new"] if r["active"] else None}

    def stiffness(self, elem_coords, mat, thickness=1.0, **kwargs):
        """Initial (zero-displacement) tangent -- interface completeness,
        matching every other nonlinear element's stiffness()."""
        return self.tangent_stiffness(elem_coords, np.zeros(2), mat, thickness,
                                       state=self.init_state())


# =====================================================================
# Node-to-segment contact, Wave 3 items 16-17 (docs/consolidated_
# future_roadmap.md): TWO deformable bodies in contact, discovered by
# real geometric search, rather than one node against an analytically-
# prescribed rigid obstacle. GapContactPenalty/GapContactCurvedFriction
# above are untouched and remain the right, cheaper tool whenever the
# obstacle genuinely IS fixed/analytic (a wall, a pin) -- this section
# is additive, for the case where the "obstacle" is itself a moving,
# deforming edge of another (or the same) mesh.
# =====================================================================
def closest_point_on_segment_2d(p, a, b):
    """Closest-point projection of point p onto the line segment a->b
    (all length-2 array-likes). Returns (t, proj): t in [0,1] is the
    CLAMPED parametric coordinate (t=0 at a, t=1 at b -- t is clamped
    to the segment's actual extent, so a point beyond either end
    projects onto that end, the standard simplified node-to-segment
    behavior -- see NodeToSegmentContact2D's own docstring for the
    real limitation this has at a multi-segment master curve's
    vertices/corners), and proj is the projected point itself.

    Degenerate (zero-length) segments return t=0, proj=a rather than
    dividing by zero -- a caller building master_segments from real
    mesh connectivity should never hit this in practice, but a defined,
    non-crashing answer is safer than an uncaught ZeroDivisionError."""
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    L2 = ab @ ab
    if L2 < 1e-30:
        return 0.0, a.copy()
    t = ((p - a) @ ab) / L2
    t = min(max(t, 0.0), 1.0)
    return t, a + t * ab


def find_contact_pairs_2d(slave_node_ids, master_segments, node_coords, search_radius):
    """Broad-phase candidate-pair search, Wave 3 item 16. For every
    (slave_node, master_segment) combination, computes the actual
    closest-point distance (via closest_point_on_segment_2d()) and
    keeps the pair if that distance is <= search_radius -- i.e. this is
    an exhaustive O(n_slave * n_segments) DISTANCE FILTER, not a real
    spatial-hash/tree broad phase. Documented, deliberate scope: this
    is adequate for the small-to-moderate contact zones this package's
    own examples/tests use (tens to low hundreds of candidate nodes/
    segments), not large-scale contact with thousands of bodies -- a
    genuine spatial-hash or bounding-volume-hierarchy broad phase would
    be the natural next step if that scale is ever needed, and would
    slot in here as a DROP-IN replacement (same return format), not a
    rewrite of anything downstream.

    slave_node_ids: iterable of node indices (into node_coords) treated
    as potential contact points.
    master_segments: iterable of (node_a, node_b) index pairs (into
    node_coords) forming the candidate master surface -- ORDER
    MATTERS: NodeToSegmentContact2D's outward normal is defined from
    a->b by a fixed right-hand convention (see that class's own
    docstring), so master_segments must be consistently wound (the
    master body on the same side of every segment) for contact to
    engage in the intended direction.
    node_coords: (n_nodes, 2) array -- the CURRENT (deformed) or
    reference configuration, caller's choice; re-call this function
    with updated node_coords to refresh the candidate list as a
    simulation progresses (this function does not track state itself).

    Returns a list of (slave_node_id, (master_a_id, master_b_id))
    tuples, one per candidate pair found -- feed each directly to
    fesystem.add_contact_element(NodeToSegmentContact2D(), [slave_node_id,
    master_a_id, master_b_id], mat)."""
    pairs = []
    coords = np.asarray(node_coords, dtype=float)
    for s in slave_node_ids:
        p = coords[s]
        for (ma, mb) in master_segments:
            _, proj = closest_point_on_segment_2d(p, coords[ma], coords[mb])
            if np.linalg.norm(p - proj) <= search_radius:
                pairs.append((s, (ma, mb)))
    return pairs


class NodeToSegmentContact2D(Element):
    """Wave 3 item 16 (docs/consolidated_future_roadmap.md): frictionless
    penalty contact between a single SLAVE node and a MASTER SEGMENT
    made of two ordinary mesh nodes -- the "obstacle" is no longer a
    fixed plane/circle (GapContactPenalty/GapContactCurvedFriction
    above), it is itself part of a deformable body's boundary, so a
    real two-deformable-body contact problem is now expressible: mesh
    BOTH bodies normally (Quad4PlaneStress, Tri3PlaneStress, ...),
    discover candidate slave-node/master-segment pairs with
    find_contact_pairs_2d() (or supply them directly if already known),
    and register one NodeToSegmentContact2D per candidate pair via
    fesystem.add_contact_element() -- no other solver-side machinery is
    needed; assemble_internal_force()/assemble_tangent_stiffness()
    already sum every registered contact element automatically (see
    FESystem.add_contact_element()'s own docstring), exactly the same
    additive mechanism GapContactPenalty already uses.

    3 NODES per element -- [slave, master_a, master_b], node-major DOF
    order [slave_x, slave_y, master_a_x, master_a_y, master_b_x,
    master_b_y] (elem_coords/u_elem both follow this same 3-row/6-entry
    layout). mat = (k_p,) -- just the penalty stiffness; unlike
    GapContactPenalty, g0 and n_hat are no longer fixed inputs, they
    are DERIVED every call from the segment's own (possibly deformed)
    current geometry via closest_point_on_segment_2d().

    GEOMETRY / SIGN CONVENTION: current slave position p = elem_coords[0]
    + u_slave, current master endpoints a, b similarly. n_hat is the
    RIGHT-HAND-ROTATED unit normal of the CURRENT segment direction
    (b-a) -- i.e. n_hat = normalize([dy, -dx]) for (dx,dy) = b-a -- by
    convention this must point AWAY from the master body's interior
    (the caller is responsible for winding master_segments consistently
    so this holds, exactly the same "caller supplies a correct outward
    normal" responsibility GapContactPenalty already places on its own
    n_hat argument). Penetration delta = -dot(p-proj, n_hat): delta>0
    means the slave node has crossed to the WRONG (inward) side of the
    segment -- same delta>0-means-penetrating sign convention as
    GapContactPenalty (chosen deliberately, so an existing k_p intuition
    carries over directly).

    FORCE DISTRIBUTION: virtual-work-consistent node-to-segment
    discretization (Wriggers, "Computational Contact Mechanics", Ch. 3)
    -- the penalty force f_c = k_p*delta (magnitude, delta>0 only) is
    applied to the slave node along -n_hat (pointing INTO the master
    body, i.e. in the direction of INCREASING penetration -- matching
    this package's F_int = dU/du convention, the SAME "points toward
    the obstacle, not away from it" sign choice GapContactCurvedFriction's
    own docstring already flags and explains for exactly this reason),
    and the exactly equal-and-opposite reaction is split across the two
    master nodes by the projection's own shape-function weights
    (1-t, t) -- so f_slave + f_master_a + f_master_b = 0 identically,
    REGARDLESS of delta or t (a genuine Newton's-third-law self-
    equilibration, checked directly in tests/test_node_to_segment_
    contact.py rather than merely asserted).

    tangent_stiffness() is FINITE DIFFERENCE of internal_force() --
    deliberately, the same choice Tet4NeoHookean/GapContactCurvedFriction
    already make in this package: the exact consistent tangent would
    need t's own dependence on all 3 nodes' positions (the closest-
    point projection is itself a function of a, b, AND p) plus n_hat's
    dependence on the segment's rotation, both genuinely intricate to
    hand-differentiate correctly -- a numerically consistent tangent
    from the already-validated closed-form force law above is safer.

    KNOWN LIMITATION (honestly scoped, not silently glossed over): a
    single master segment has no knowledge of its neighbors, so a slave
    node sliding past a segment's clamped endpoint does not smoothly
    hand off to the next segment in a multi-segment master curve --
    the classic node-to-segment "vertex/corner" issue that motivates
    segment-to-segment methods in the contact-mechanics literature.
    Adequate for the single- or few-segment contact zones this
    package's own tests use; a genuinely smooth multi-segment master
    curve is future work, not implemented here."""
    n_nodes, dofs_per_node, dim, gauss_order = 3, 2, 1, 1

    @staticmethod
    def _geometry(elem_coords, u_elem):
        p = elem_coords[0] + u_elem[0:2]
        a = elem_coords[1] + u_elem[2:4]
        b = elem_coords[2] + u_elem[4:6]
        t, proj = closest_point_on_segment_2d(p, a, b)
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-30:
            n_hat = np.array([0.0, 1.0])
            t_hat = np.array([1.0, 0.0])
        else:
            n_hat = np.array([ab[1], -ab[0]]) / L
            t_hat = ab / L
        delta = -((p - proj) @ n_hat)
        return dict(p=p, a=a, b=b, t=t, proj=proj, n_hat=n_hat, t_hat=t_hat,
                    delta=delta, seg_len=L)

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        (k_p,) = mat
        g = self._geometry(elem_coords, u_elem)
        if g["delta"] <= 0.0:
            return np.zeros(6)
        f_c = k_p * g["delta"]
        n_hat, t = g["n_hat"], g["t"]
        f = np.zeros(6)
        f[0:2] = -f_c * n_hat
        f[2:4] = f_c * n_hat * (1.0 - t)
        f[4:6] = f_c * n_hat * t
        return f

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """Central-difference tangent of internal_force() -- see the
        class docstring for why this is computed numerically here."""
        h = 1e-7 * max(1.0, float(np.linalg.norm(u_elem)))
        K = np.zeros((6, 6))
        for j in range(6):
            du = np.zeros(6); du[j] = h
            fp = self.internal_force(elem_coords, u_elem + du, mat, thickness, **kwargs)
            fm = self.internal_force(elem_coords, u_elem - du, mat, thickness, **kwargs)
            K[:, j] = (fp - fm) / (2 * h)
        return K

    def stiffness(self, elem_coords, mat, thickness=1.0, **kwargs):
        """Initial (zero-displacement) tangent -- interface completeness,
        matching every other nonlinear/contact element's stiffness()."""
        return self.tangent_stiffness(elem_coords, np.zeros(6), mat, thickness)


class NodeToSegmentContact2DFriction(NodeToSegmentContact2D):
    """Wave 3 item 17 (docs/consolidated_future_roadmap.md): adds
    Coulomb stick-slip friction to NodeToSegmentContact2D -- the direct
    node-to-segment generalization of GapContactCurvedFriction's
    stick/slip return map above, extended from "arc-length position
    along a FIXED circle" to "position along a (possibly deforming)
    master SEGMENT," reusing the identical stick/slip decision
    structure (see GapContactCurvedFriction's own _trial() docstring
    for the algorithm this mirrors) rather than deriving a new friction
    law from scratch.

    Tangential coordinate: s = t * seg_len, t/seg_len from the SAME
    closest-point projection the normal-contact geometry already
    computes (using one consistent projection point for both directions,
    not two independent ones). state = {"s_stick": float or None},
    None meaning "no prior anchor" (matching GapContactCurvedFriction's
    own convention exactly) -- reset to None on separation by
    commit_state(), so a future re-contact (possibly against a
    different part of the segment, or after the segment has moved)
    starts fresh rather than remembering a stale tangential position.

    mat = (k_p, k_t, mu) -- k_p/mu match NodeToSegmentContact2D's/
    GapContactCurvedFriction's own meaning; k_t is the tangential
    ("stick") penalty stiffness, same role as GapContactCurvedFriction's
    k_t. NodeToSegmentContact2D itself (frictionless) remains available
    unchanged as a separate class -- this is an ADDITIVE sibling, not a
    replacement, the same "keep every option available" principle
    every other Wave 2/3 addition in this package follows.

    DOCUMENTED SIMPLIFICATION: because the master segment can itself
    stretch (its two endpoints are ordinary, independently-moving mesh
    DOFs), "stuck at s=s_stick" is tracked as a material point's
    position measured in CURRENT-configuration arc-length units at each
    evaluation, not re-mapped through the segment's own stretch history
    -- adequate for the small-sliding, low-stretch contact regime this
    package's own tests exercise; a fully rate-consistent treatment of
    friction on a stretching master surface is a further refinement,
    not implemented here."""

    @staticmethod
    def init_state():
        return {"s_stick": None}

    def _trial(self, elem_coords, u_elem, mat, state):
        k_p, k_t, mu = mat
        g = self._geometry(elem_coords, u_elem)
        if g["delta"] <= 0.0:
            return dict(pn=0.0, tt=0.0, n_hat=g["n_hat"], t_hat=g["t_hat"],
                        t=g["t"], s_stick_new=state["s_stick"], active=False)

        pn = k_p * g["delta"]
        s = g["t"] * g["seg_len"]
        s_stick_n = state["s_stick"]
        if s_stick_n is None:
            s_stick_n = s
        e_t_trial = s - s_stick_n
        tt_trial = k_t * e_t_trial
        limit = mu * pn
        if abs(tt_trial) <= limit:
            tt, s_stick_new = tt_trial, s_stick_n
        else:
            tt = limit * np.sign(tt_trial)
            s_stick_new = s - tt / k_t
        return dict(pn=pn, tt=tt, n_hat=g["n_hat"], t_hat=g["t_hat"],
                    t=g["t"], s_stick_new=s_stick_new, active=True)

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs):
        if state is None:
            state = self.init_state()
        r = self._trial(elem_coords, u_elem, mat, state)
        if not r["active"]:
            return np.zeros(6)
        t = r["t"]
        f = np.zeros(6)
        f[0:2] = -r["pn"] * r["n_hat"] + r["tt"] * r["t_hat"]
        f[2:4] = (r["pn"] * r["n_hat"] - r["tt"] * r["t_hat"]) * (1.0 - t)
        f[4:6] = (r["pn"] * r["n_hat"] - r["tt"] * r["t_hat"]) * t
        return f

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs):
        """Central-difference tangent -- see the class docstring."""
        if state is None:
            state = self.init_state()
        h = 1e-7 * max(1.0, float(np.linalg.norm(u_elem)))
        K = np.zeros((6, 6))
        for j in range(6):
            du = np.zeros(6); du[j] = h
            fp = self.internal_force(elem_coords, u_elem + du, mat, thickness, state=state)
            fm = self.internal_force(elem_coords, u_elem - du, mat, thickness, state=state)
            K[:, j] = (fp - fm) / (2 * h)
        return K

    def commit_state(self, elem_coords, u_elem, mat, state, thickness=1.0, **kwargs):
        """Called once per CONVERGED load step. Resets the stick anchor
        to None on separation, matching GapContactCurvedFriction's
        commit_state() exactly."""
        r = self._trial(elem_coords, u_elem, mat, state)
        return {"s_stick": r["s_stick_new"] if r["active"] else None}

    def stiffness(self, elem_coords, mat, thickness=1.0, **kwargs):
        """Initial (zero-displacement) tangent -- interface completeness,
        matching every other nonlinear/contact element's stiffness()."""
        return self.tangent_stiffness(elem_coords, np.zeros(6), mat, thickness,
                                       state=self.init_state())

