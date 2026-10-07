"""
adaptivity.py -- Wave 8 items 41-45 (docs/consolidated_future_roadmap.md,
source fem_implementation_lessons.md Chapters 14-15): a posteriori
error estimation, marking strategies, conforming local mesh
refinement, mesh-hierarchy tracking, and the solution-driven adaptive
refinement loop (solve -> estimate -> mark -> refine -> re-solve).

Scope, stated up front (same "narrow, then document the narrowing"
convention as grading.Notch/Fillet in Wave 5, and iterative_solvers.
py's own multigrid section): every function in this module is scoped
to Tri3PlaneStress -- a deliberate choice, not an oversight, so items
41-45 form one coherent, end-to-end-tested pipeline on a single
element family rather than five independently-scoped fragments.
Tri3's constant-per-element stress field also makes the error
estimator in item 41 exact-integral rather than Gauss-quadrature-
approximate, which keeps the whole pipeline simple to verify.
Extending any piece here to Quad4/higher-order elements is real,
undone future work -- not attempted here.

fem_implementation_lessons.md's own Appendix note is worth repeating
here explicitly: fea_engine's actual design generates a fresh mesh
every time (mesh.py/build_mesh.py), rather than iteratively editing
an existing one -- so local refinement (item 43) and the resulting
mesh hierarchy (item 44) are a genuinely NEW capability added by this
wave, not a gap in something that already existed. The refinement
algorithm chosen (Rivara 1984 longest-edge bisection with
conformity-preserving propagation) is the specific one
fem_implementation_lessons.md names as "provably cannot degenerate,"
as opposed to repeated green-refinement closure, which the book
flags as able to produce arbitrarily thin triangles.
"""
import numpy as np


# =====================================================================
# Item 41 -- a posteriori error estimators
# =====================================================================
def element_stresses(mesh, elem_formulation, D, U):
    """Per-element stress [sigma_xx, sigma_yy, sigma_xy], shape
    (n_elements, 3). Tri3's shape functions are linear, so strain
    (and hence stress, for a linear-elastic D) is CONSTANT over the
    element -- evaluated once at the centroid (1/3, 1/3), which is
    exact, not an approximation, for this element."""
    npn = elem_formulation.dofs_per_node
    centroid = (1.0 / 3.0, 1.0 / 3.0)   # plain tuple: Element._cached_shape_and_derivs()
                                         # keys its cache on natural_coords directly, which
                                         # must be hashable -- an ndarray is not.
    stresses = np.zeros((len(mesh.elements), 3))
    for e, elem in enumerate(mesh.elements):
        g = np.array([npn * n + k for n in elem for k in range(npn)])
        u_elem = U[g]
        B, _ = elem_formulation.B_matrix(centroid, mesh.nodes[elem])
        stresses[e] = D @ (B @ u_elem)
    return stresses


def _node_to_element_map(mesh):
    node_elems = [[] for _ in range(len(mesh.nodes))]
    for e, elem in enumerate(mesh.elements):
        for n in elem:
            node_elems[n].append(e)
    return node_elems


def _element_areas(mesh):
    areas = np.zeros(len(mesh.elements))
    for e, elem in enumerate(mesh.elements):
        x = mesh.nodes[elem]
        areas[e] = 0.5 * abs((x[1, 0] - x[0, 0]) * (x[2, 1] - x[0, 1])
                              - (x[2, 0] - x[0, 0]) * (x[1, 1] - x[0, 1]))
    return areas


def zz_recovery_estimator(mesh, elem_formulation, D, U):
    """Zienkiewicz & Zhu (1987)-style a posteriori error estimator:
    build a smoother, nodally-continuous recovered stress field
    sigma* by area-weighted-averaging the discontinuous per-element
    sigma_h over every element sharing each node (a simpler averaging
    recovery than the full polynomial superconvergent-patch-recovery
    fit ZZ's original paper uses, but the same underlying idea --
    "recover a better field, measure distance to it"), then estimate
    each element's error in the ENERGY norm:
        eta_e = sqrt(area_e * (sigma*_e - sigma_h,e)^T D^{-1} (sigma*_e - sigma_h,e))
    where sigma*_e is the average of the 3 corner-recovered values on
    element e. Returns (eta, sigma_h, sigma_star)."""
    sigma_h = element_stresses(mesh, elem_formulation, D, U)
    areas = _element_areas(mesh)
    node_elems = _node_to_element_map(mesh)
    n_nodes = len(mesh.nodes)
    sigma_star = np.zeros((n_nodes, 3))
    for n in range(n_nodes):
        elems = node_elems[n]
        if not elems:
            continue
        w = areas[elems]
        sigma_star[n] = (w[:, None] * sigma_h[elems]).sum(axis=0) / w.sum()

    Dinv = np.linalg.inv(D)
    eta = np.zeros(len(mesh.elements))
    for e, elem in enumerate(mesh.elements):
        diff = sigma_star[elem].mean(axis=0) - sigma_h[e]
        eta[e] = np.sqrt(max(float(diff @ Dinv @ diff), 0.0) * areas[e])
    return eta, sigma_h, sigma_star


def jump_residual_estimator(mesh, elem_formulation, D, U):
    """Cheaper, cruder edge-stress-jump indicator -- fem_implementation_
    lessons.md's own point 13 ("a cheaper, cruder error indicator can
    produce a MORE efficient adaptive mesh than an expensive, more
    accurate one") is the explicit motivation for keeping this
    alongside zz_recovery_estimator() rather than treating ZZ as
    strictly superior. For each INTERIOR edge shared by two elements,
    computes the jump in traction (stress dotted with the edge
    normal) between the two constant per-element stress states, times
    edge length, split evenly onto the two sharing elements.

    Documented limitation, not a bug: boundary edges contribute
    nothing (no neighbor to compare against), so this indicator
    structurally undercounts error driven by a boundary condition or
    boundary load right at the edge of the domain -- zz_recovery_
    estimator() does not have this blind spot. See tests/
    test_adaptivity.py for a direct side-by-side comparison."""
    sigma_h = element_stresses(mesh, elem_formulation, D, U)
    edge_to_elems = {}
    for e, elem in enumerate(mesh.elements):
        for a, b in ((elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edge_to_elems.setdefault(key, []).append(e)

    def traction(sig, n):
        sxx, syy, sxy = sig
        return np.array([sxx * n[0] + sxy * n[1], sxy * n[0] + syy * n[1]])

    eta = np.zeros(len(mesh.elements))
    for (a, b), elems in edge_to_elems.items():
        if len(elems) != 2:
            continue
        p0, p1 = mesh.nodes[a], mesh.nodes[b]
        edge_vec = p1 - p0
        length = np.linalg.norm(edge_vec)
        if length == 0.0:
            continue
        normal = np.array([edge_vec[1], -edge_vec[0]]) / length
        e0, e1 = elems
        jump = np.linalg.norm(traction(sigma_h[e0], normal) - traction(sigma_h[e1], normal))
        contrib = 0.5 * jump * length
        eta[e0] += contrib
        eta[e1] += contrib
    return eta


# =====================================================================
# Item 42 -- marking strategies
# =====================================================================
def fixed_fraction_marking(eta, fraction=0.3):
    """Mark the top `fraction` of elements BY COUNT, ranked by eta
    (descending). fraction=0.3 marks the worst 30% of elements."""
    n = len(eta)
    if n == 0:
        return np.zeros(0, dtype=bool)
    n_mark = max(1, int(np.ceil(fraction * n)))
    order = np.argsort(eta)[::-1]
    marked = np.zeros(n, dtype=bool)
    marked[order[:n_mark]] = True
    return marked


def threshold_marking(eta, threshold_rel=0.5):
    """Mark every element whose eta_e exceeds threshold_rel * max(eta)."""
    eta = np.asarray(eta)
    if eta.size == 0 or np.max(eta) == 0:
        return np.zeros(eta.shape[0], dtype=bool)
    return eta > threshold_rel * np.max(eta)


def equidistribution_marking(eta, n_target_elements):
    """Mark elements whose error exceeds the level that would
    equidistribute the total squared (energy-norm) error evenly
    across n_target_elements -- the standard adaptive-FEM
    equidistribution prescription eta_e <= TOL/sqrt(N), applied here
    as a marking rule (mark whatever currently exceeds that per-
    element share) rather than a stopping rule."""
    eta = np.asarray(eta)
    total_sq = np.sum(eta ** 2)
    if total_sq == 0 or n_target_elements <= 0:
        return np.zeros(eta.shape[0], dtype=bool)
    target_level = np.sqrt(total_sq / n_target_elements)
    return eta > target_level


# =====================================================================
# Items 43-44 -- conforming local refinement (Rivara longest-edge
# bisection) + mesh-hierarchy (parent/child, level) tracking
# =====================================================================
class RefinementRecord:
    """Item 44: parent/child tracking across one refinement step.
    parent_element[i] is the ORIGINAL (pre-refinement) element id new
    element i descended from; level[i] is its refinement depth
    relative to whatever `parent_level` was passed into
    refine_triangle_mesh_longest_edge() (0 if this was the first
    refinement of the original mesh)."""
    def __init__(self, parent_element, level):
        self.parent_element = np.asarray(parent_element, dtype=int)
        self.level = np.asarray(level, dtype=int)


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _longest_local_edge(coords):
    """coords: (3,2) triangle vertex coordinates. Returns local edge
    index ei in {0,1,2}: 0 -> (v0,v1) opposite v2, 1 -> (v1,v2)
    opposite v0, 2 -> (v2,v0) opposite v1."""
    d01 = np.sum((coords[1] - coords[0]) ** 2)
    d12 = np.sum((coords[2] - coords[1]) ** 2)
    d20 = np.sum((coords[0] - coords[2]) ** 2)
    return int(np.argmax([d01, d12, d20]))


def refine_triangle_mesh_longest_edge(mesh, marked, parent_level=None):
    """Rivara (1984) longest-edge conforming bisection: refines every
    element in `marked` (boolean array, len == n_elements) PLUS
    whatever additional elements the conformity-preserving
    propagation step requires -- when triangle T's longest edge is
    shared with a neighbor T2 for whom that edge is NOT also T2's own
    longest edge, T2 is bisected first (along ITS longest edge), and
    this repeats outward until reaching either a boundary edge or a
    pair of triangles that agree on their shared longest edge. This
    is what guarantees no hanging nodes are ever created (a shared
    edge always gets exactly one midpoint, used by both sides) and
    is the specific property that makes this algorithm provably
    terminate without degenerating into arbitrarily thin triangles
    (Rivara's own termination argument: the longest edge in the
    propagation CHAIN is strictly decreasing, so it cannot cycle).

    Returns (new_mesh, record) -- see RefinementRecord above."""
    from .mesh import Mesh
    n0 = len(mesh.nodes)
    n_elem0 = len(mesh.elements)
    node_coords = [mesh.nodes[i].copy() for i in range(n0)]
    tris = {t: tuple(int(v) for v in mesh.elements[t]) for t in range(n_elem0)}
    if parent_level is None:
        parent_level = np.zeros(n_elem0, dtype=int)

    edge_to_tris = {}
    for t, verts in tris.items():
        for a, b in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            edge_to_tris.setdefault(_edge_key(a, b), set()).add(t)

    def longest_edge_key(t):
        verts = tris[t]
        coords = np.array([node_coords[v] for v in verts])
        ei = _longest_local_edge(coords)
        a, b = verts[ei], verts[(ei + 1) % 3]
        return ei, _edge_key(a, b)

    midpoints = {}          # edge_key -> new node id
    parent_of = {t: t for t in tris}       # current-tri-id -> ORIGINAL id
    level_of = {t: int(parent_level[t]) for t in tris}
    next_tid = n_elem0

    def get_or_make_midpoint(ek):
        if ek in midpoints:
            return midpoints[ek]
        a, b = ek
        m = len(node_coords)
        node_coords.append(0.5 * (node_coords[a] + node_coords[b]))
        midpoints[ek] = m
        return m

    def remove_tri_from_edges(t):
        verts = tris[t]
        for a, b in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            edge_to_tris.get(_edge_key(a, b), set()).discard(t)

    def add_tri_edges(t):
        verts = tris[t]
        for a, b in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            edge_to_tris.setdefault(_edge_key(a, b), set()).add(t)

    def split_one(t, ek):
        """Split triangle t along the (already-agreed) edge ek,
        replacing it in `tris` with 2 children sharing ek's midpoint."""
        nonlocal next_tid
        verts = tris[t]
        ei, key_check = longest_edge_key(t)
        assert key_check == ek, "split_one: edge mismatch -- internal bug"
        m = get_or_make_midpoint(ek)
        oi = (ei + 2) % 3
        p_opp, p_a, p_b = verts[oi], verts[ei], verts[(ei + 1) % 3]
        remove_tri_from_edges(t)
        orig = parent_of[t]
        lvl = level_of[t] + 1
        del tris[t]
        del parent_of[t]
        del level_of[t]
        c1, c2 = next_tid, next_tid + 1
        next_tid += 2
        tris[c1] = (p_opp, p_a, m)
        tris[c2] = (p_opp, m, p_b)
        parent_of[c1] = orig
        parent_of[c2] = orig
        level_of[c1] = lvl
        level_of[c2] = lvl
        add_tri_edges(c1)
        add_tri_edges(c2)
        return c1, c2

    def ensure_refined(start_t):
        stack = [start_t]
        while stack:
            t = stack[-1]
            if t not in tris:
                stack.pop()
                continue
            ei, ek = longest_edge_key(t)
            if ek in midpoints:
                # Midpoint already exists (created by a neighbor's
                # own propagation) -- safe to split t now using it.
                split_one(t, ek)
                stack.pop()
                continue
            neighbors = [t2 for t2 in edge_to_tris.get(ek, ()) if t2 != t and t2 in tris]
            if neighbors:
                t2 = neighbors[0]
                ei2, ek2 = longest_edge_key(t2)
                if ek2 != ek:
                    stack.append(t2)
                    continue
                # Neighbor agrees this IS its longest edge too --
                # split both, sharing the one new midpoint.
                split_one(t2, ek)
                split_one(t, ek)
                stack.pop()
                continue
            # boundary edge (no neighbor) -- safe to split unilaterally.
            split_one(t, ek)
            stack.pop()

    marked_ids = [t for t in range(n_elem0) if marked[t]]
    for t in marked_ids:
        ensure_refined(t)

    final_ids = sorted(tris.keys())
    remap = {old: new for new, old in enumerate(final_ids)}
    new_elements = np.array([tris[t] for t in final_ids], dtype=int)
    new_nodes = np.array(node_coords)
    record = RefinementRecord(
        parent_element=[parent_of[t] for t in final_ids],
        level=[level_of[t] for t in final_ids],
    )
    new_mesh = Mesh(new_nodes, new_elements, dim=2)
    return new_mesh, record


# =====================================================================
# Item 45 -- solution-driven adaptive refinement loop
# =====================================================================
def adaptive_refine_solve(mesh, elem_formulation, D, setup_fn, thickness=1.0,
                           estimator="zz", marking="fixed_fraction",
                           marking_param=0.3, max_refinements=5, tol=None,
                           verbose=False):
    """The capstone loop: solve -> estimate -> mark -> refine -> re-solve
    (fem_implementation_lessons.md Ch. 14-15). `setup_fn(fesystem,
    mesh)` is called on EVERY mesh in the sequence (initial and every
    refined one) to apply boundary conditions and loads -- it must be
    supplied by the caller (not hard-coded here) because node ids
    change after every refinement step, so BCs/loads have to be
    re-derived geometrically each time (e.g. via mesh.nodes_on_line()),
    exactly the same pattern every other geometry-driven part of this
    package already uses.

    estimator: "zz" (zz_recovery_estimator, default) or "jump"
    (jump_residual_estimator). marking: "fixed_fraction" (default),
    "threshold", or "equidistribution" (marking_param is the
    corresponding fraction/threshold_rel/n_target_elements).

    Stops after max_refinements steps, or as soon as the TOTAL
    estimated error (sqrt(sum(eta**2))) drops below `tol` (if given).

    Returns a list of dicts, one per mesh in the sequence: {'mesh':
    Mesh, 'n_elements': int, 'n_dofs': int, 'U': displacement vector,
    'eta': per-element error array, 'total_error': float, 'record':
    RefinementRecord or None (None for the first, un-refined mesh)}."""
    from .solver import FESystem

    history = []
    current_mesh = mesh
    record = None
    for step in range(max_refinements + 1):
        fs = FESystem(current_mesh, elem_formulation, thickness=thickness, sparse=False)
        fs.assemble_stiffness(D)
        setup_fn(fs, current_mesh)
        U = fs.solve_static()

        if estimator == "zz":
            eta, _, _ = zz_recovery_estimator(current_mesh, elem_formulation, D, U)
        elif estimator == "jump":
            eta = jump_residual_estimator(current_mesh, elem_formulation, D, U)
        else:
            raise ValueError(f"adaptive_refine_solve: unknown estimator={estimator!r}")

        total_error = float(np.sqrt(np.sum(eta ** 2)))
        history.append({
            "mesh": current_mesh, "n_elements": len(current_mesh.elements),
            "n_dofs": fs.n_dof, "U": U, "eta": eta,
            "total_error": total_error, "record": record,
        })
        if verbose:
            print(f"  step {step}: n_elements={len(current_mesh.elements)} "
                  f"n_dofs={fs.n_dof} total_error={total_error:.6e}")

        if tol is not None and total_error < tol:
            break
        if step == max_refinements:
            break

        if marking == "fixed_fraction":
            marked = fixed_fraction_marking(eta, fraction=marking_param)
        elif marking == "threshold":
            marked = threshold_marking(eta, threshold_rel=marking_param)
        elif marking == "equidistribution":
            marked = equidistribution_marking(eta, n_target_elements=marking_param)
        else:
            raise ValueError(f"adaptive_refine_solve: unknown marking={marking!r}")

        if not marked.any():
            break
        current_mesh, record = refine_triangle_mesh_longest_edge(current_mesh, marked)

    return history
