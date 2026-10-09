"""
element.py -- Module 3: Gauss-Legendre integration core engine +
element formulations.

The Gauss-quadrature machinery (gauss_legendre / gauss_product) is
general-purpose: any order, 1-D/2-D/3-D. The generic Element.stiffness()
/ Element.mass() Gauss loops live ONCE on the base class and work for
any isoparametric element that supplies shape_and_derivs() and
B_matrix() -- Quad4PlaneStress and Hex8Solid3D use this default as-is.

Elements that need something the generic loop can't express (selective
reduced integration for the Mindlin plate; a closed-form Hermite
formula for the Euler-Bernoulli beam) simply OVERRIDE stiffness()
and/or mass() in their subclass while still reusing shape_and_derivs()
and the module-level quadrature helpers. Nothing about the base class
or the other elements has to change either way.

To add a new element type: subclass Element, implement
shape_and_derivs() (+ B_matrix() if you want the generic Gauss loop,
or override stiffness()/mass() directly if not), and add one line to
ELEMENT_REGISTRY.

Full vs. reduced integration
-----------------------------
Every element exposes stiffness() with an optional gauss_order= (or,
for the plate, integration=) override, plus two named convenience
wrappers:
    full_stiffness()     -- self.gauss_order points/direction (exact
                             for this element's polynomial; the default
                             used everywhere in this package unless you
                             ask for something else)
    reduced_stiffness()  -- one point/direction fewer; relieves the
                             locking that full integration causes in
                             bending-dominated problems, at the risk of
                             rank-deficient ('hourglass') spurious modes
Use spurious_zero_energy_modes() to check how many spurious modes a
given reduced-integration element has, by comparing its rank against
the same element's full-integration stiffness (assumed reference-
correct). Quad4MindlinPlate additionally supports a 'sri' mode
(selective reduced integration: full for bending, reduced for shear --
its actual default, and the standard, non-hourglass-prone fix for
plate/shell locking) alongside plain 'full' and 'reduced'.

hourglass_stabilized_stiffness() -- Wave 2 item 12 (docs/consolidated_
future_roadmap.md): reduced_stiffness() plus a Flanagan-Belytschko/
Belytschko-Bindeman-style perturbation stiffness restoring just the
spurious (hourglass) directions' rank, leaving every genuinely-well-
integrated mode (including every affine/constant-strain field --
patch-test-safe by construction) untouched. See that method's own
docstring for the full derivation. All three of full_stiffness()/
reduced_stiffness()/hourglass_stabilized_stiffness() remain available
side by side on every element that defines B_matrix() -- this is an
explicit, opt-in CHOICE (matching the same principle established for
FESystem's SciPy-vs-PyTorch backend= choice), never a silent
replacement of the existing (honestly-labeled-unsafe) reduced_
stiffness() default.
"""
__author__ = "Abhijeet"
import numpy as np
from scipy.special import roots_jacobi   # required dep (scipy>=1.8) -- see
                                          # tri_quadrature()/tet_quadrature()
                                          # below for why this specific,
                                          # well-tested library routine (not
                                          # a hand-transcribed coefficient
                                          # table) was deliberately chosen.


# =====================================================================
# Gauss-Legendre quadrature engine
# =====================================================================
def gauss_legendre(n):
    """n-point 1-D Gauss-Legendre rule on [-1, 1]. Exact for polynomials
    up to degree 2n-1. Any n >= 1 works (not limited to a hardcoded
    2-point table)."""
    pts, wts = np.polynomial.legendre.leggauss(n)
    return pts, wts


def gauss_product(n, dim):
    """Tensor-product Gauss rule for dim in {1, 2, 3}. Returns a list of
    natural-coordinate tuples and a matching list of combined weights."""
    pts1d, wts1d = gauss_legendre(n)
    if dim == 1:
        return [(p,) for p in pts1d], list(wts1d)
    if dim == 2:
        pts = [(xi, eta) for xi in pts1d for eta in pts1d]
        wts = [wx * wy for wx in wts1d for wy in wts1d]
        return pts, wts
    if dim == 3:
        pts = [(xi, eta, zeta) for xi in pts1d for eta in pts1d for zeta in pts1d]
        wts = [wx * wy * wz for wx in wts1d for wy in wts1d for wz in wts1d]
        return pts, wts
    raise ValueError(f"unsupported dim={dim}")


def tet_quadrature_4pt():
    """4-point quadrature rule on the natural tetrahedron
    {r,s,t >= 0, r+s+t <= 1} (Keast/Zienkiewicz-Taylor's standard
    degree-2-exact rule) -- NOT a tensor-product rule, so it doesn't
    belong in gauss_product() above (that function builds a cube-
    shaped grid on [-1,1]^dim, appropriate for Hex8/Hex20/Quad4/Quad8,
    which are mapped FROM a cube/square; a simplex element like Tet10
    is mapped from a triangle/tetrahedron, a genuinely different
    reference domain with its own quadrature theory). Exact for any
    polynomial of total degree <= 2 over the tetrahedron -- exactly
    what Tet10Solid3D's B^T D B integrand needs (quadratic shape
    functions -> linear B -> quadratic B^T D B, for a straight-sided
    element with constant detJ; see Tet10Solid3D's docstring).

    Returns (points, weights): points is a list of 4 (r, s, t) tuples,
    weights is a list of 4 floats summing to 1 (barycentric-normalized
    -- the CALLER multiplies by the element's physical volume, exactly
    like Tet4Solid3D's single-point rule already does with its 1/6
    natural-tetrahedron-volume factor)."""
    a = 0.5854101966249685   # (5 + 3*sqrt(5)) / 20
    b = 0.1381966011250105   # (5 - sqrt(5)) / 20
    points = [(b, b, b), (a, b, b), (b, a, b), (b, b, a)]
    weights = [0.25, 0.25, 0.25, 0.25]
    return points, weights


def tet_quadrature(order):
    """Order-PARAMETERIZED quadrature on the natural tetrahedron
    {r,s,t >= 0, r+s+t <= 1}, exact for any polynomial of total degree
    <= order -- Wave 15 item 125 (docs/consolidated_future_roadmap.md):
    generalizes tet_quadrature_4pt()'s single fixed (degree-2-exact)
    rule to a genuine runtime order parameter, the uniform
    `get_quadrature(order)` pattern TensorMesh's own "Elements and
    Quadrature" documentation page gives every one of its seven
    reference shapes, and gauss_product() already gives this package's
    own quad/hex family.

    CONSTRUCTION: the collapsed-coordinate (Duffy) transform standard
    in spectral/hp element methods (Karniadakis & Sherwin, "Spectral/hp
    Element Methods for CFD"; Hesthaven & Warburton, "Nodal
    Discontinuous Galerkin Methods" Ch.6) -- maps the cube [-1,1]^3
    (a,b,c) onto the tetrahedron via
        r = (1+a)/2 * (1-b)/2 * (1-c)/2
        s = (1+b)/2 * (1-c)/2
        t = (1+c)/2
    whose Jacobian dr*ds*dt = (1-b)*(1-c)^2/64 da db dc. The (1-b) and
    (1-c)^2 weight factors are absorbed EXACTLY into an n-point
    Gauss-JACOBI(alpha=1,beta=0) rule in b and an n-point Gauss-
    JACOBI(alpha=2,beta=0) rule in c (scipy.special.roots_jacobi -- a
    well-tested SciPy library routine, deliberately chosen over
    hand-transcribing a symmetric Keast/Dunavant-style coefficient
    table from the literature: this package's own history shows how
    easy that kind of transcription is to get subtly wrong -- see
    GMSH_NODE_ORDER's own "checked directly against a real Gmsh build,
    not assumed from documentation" precedent in elements/solids.py --
    and a wrong library call is far less likely than a wrong hand-typed
    digit), crossed with a plain n-point Gauss-LEGENDRE rule in a.

    n = ceil((order+1)/2) points per direction (n^3 total) -- CHECKED
    DIRECTLY, not assumed: tests/test_quadrature_order.py integrates
    every monomial r^p * s^q * t^u with p+q+u <= order against the
    exact closed-form tetrahedron moment p!q!u!/(p+q+u+3)!, for order
    1 through 7, and confirms machine-precision agreement -- the same
    decisive, reproducible-computation standard this project used to
    confirm meshio's node order (mesh_io.py) and the corrected RK4's
    convergence rate (Wave 13 item 119).

    tet_quadrature_4pt() (order=2, the fixed rule used everywhere in
    this package before this function existed, and still Tet10Solid3D's
    default) is UNCHANGED -- this function is purely additive, reached
    only via an explicit order= argument (Tet10Solid3D.stiffness()/
    .mass()'s new quad_order= parameter).

    Returns (points, weights): points is a list of n^3 (r, s, t)
    tuples, weights is a list of n^3 floats summing to 1 (barycentric-
    normalized, matching tet_quadrature_4pt()'s own convention EXACTLY
    -- the CALLER still multiplies by the element's physical volume,
    e.g. Tet10Solid3D.stiffness()'s own `* abs(detJ) * w / 6.0`)."""
    import math
    n = max(1, math.ceil((order + 1) / 2))
    a_pts, a_wts = gauss_legendre(n)
    b_pts, b_wts = roots_jacobi(n, 1, 0)
    c_pts, c_wts = roots_jacobi(n, 2, 0)
    points = []
    weights = []
    for ai, wa in zip(a_pts, a_wts):
        for bi, wb in zip(b_pts, b_wts):
            for ci, wc in zip(c_pts, c_wts):
                r = (1 + ai) / 2 * (1 - bi) / 2 * (1 - ci) / 2
                s = (1 + bi) / 2 * (1 - ci) / 2
                t = (1 + ci) / 2
                points.append((r, s, t))
                weights.append(wa * wb * wc * 3.0 / 32.0)
    return points, weights


def tri_quadrature_3pt():
    """3-point quadrature rule on the natural triangle
    {xi,eta >= 0, xi+eta <= 1} -- the standard degree-2-exact rule
    (same Zienkiewicz-Taylor family as tet_quadrature_4pt() above, one
    simplex dimension down). NOT a tensor-product rule, so -- for
    exactly the reasoning in tet_quadrature_4pt()'s own docstring --
    it doesn't belong in gauss_product() either: this is for a
    triangle-shaped reference domain, gauss_product() only knows
    square/cube ones. Wave 0 item 7 (docs/consolidated_future_roadmap.
    md, source geometry_meshing_alternatives_research.md item 4):
    Tri6PlaneStress (elements/solids.py) is the first CALLER, mirroring
    Tet10Solid3D's use of tet_quadrature_4pt() for the same reason --
    Tri6's quadratic shape functions make B vary over the element
    (unlike Tri3's constant-strain B), so its B^T D B integrand is
    degree 2 for a straight-sided element (constant detJ), and this
    3-point rule integrates that exactly.

    Returns (points, weights): points is a list of 3 (xi, eta) tuples,
    weights is a list of 3 floats summing to 1 (barycentric-normalized
    -- the CALLER multiplies by the element's physical area, exactly
    like Tri3PlaneStress's single-point rule already does with its own
    explicit `area = abs(detJ) * 0.5` factor)."""
    a = 2.0 / 3.0
    b = 1.0 / 6.0
    points = [(b, b), (a, b), (b, a)]
    weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return points, weights


def tri_quadrature(order):
    """Order-PARAMETERIZED quadrature on the natural triangle
    {xi,eta >= 0, xi+eta <= 1}, exact for any polynomial of total
    degree <= order -- Wave 15 item 125 (docs/consolidated_future_
    roadmap.md), the 2-D sibling of tet_quadrature() above; see that
    function's own docstring for the full construction rationale
    (collapsed-coordinate/Duffy transform, scipy.special.roots_jacobi
    over a hand-transcribed literature table, the "checked directly"
    verification standard) -- identical reasoning, one simplex
    dimension down.

    CONSTRUCTION: maps the square [-1,1]^2 (a,b) onto the triangle via
        xi  = (1+a)/2 * (1-b)/2
        eta = (1+b)/2
    whose Jacobian dxi*deta = (1-b)/8 da db. The (1-b) weight is
    absorbed exactly into an n-point Gauss-JACOBI(alpha=1,beta=0) rule
    in b, crossed with a plain n-point Gauss-LEGENDRE rule in a.
    n = ceil((order+1)/2) points per direction (n^2 total).

    tri_quadrature_3pt() (order=2, the fixed rule used everywhere in
    this package before this function existed, and still Tri6Plane
    Stress's default) is UNCHANGED -- purely additive, reached only via
    an explicit order= argument (Tri6PlaneStress.stiffness()/.mass()'s
    new quad_order= parameter).

    Returns (points, weights): points is a list of n^2 (xi, eta)
    tuples, weights is a list of n^2 floats summing to 1 (barycentric-
    normalized, matching tri_quadrature_3pt()'s own convention EXACTLY
    -- the CALLER still multiplies by the element's physical area,
    e.g. Tri6PlaneStress.stiffness()'s own `* abs(detJ) * w * 0.5`)."""
    import math
    n = max(1, math.ceil((order + 1) / 2))
    a_pts, a_wts = gauss_legendre(n)
    b_pts, b_wts = roots_jacobi(n, 1, 0)
    points = []
    weights = []
    for ai, wa in zip(a_pts, a_wts):
        for bi, wb in zip(b_pts, b_wts):
            xi = (1 + ai) / 2 * (1 - bi) / 2
            eta = (1 + bi) / 2
            points.append((xi, eta))
            weights.append(wa * wb / 4.0)
    return points, weights


def jacobian(dN_natural, elem_coords):
    """dN_natural: (dim, n_nodes). elem_coords: (n_nodes, dim). Returns
    (J, detJ). General for dim=1,2,3 -- valid for any (non-degenerate)
    element shape since it's built from the element's ACTUAL nodal
    coordinates, not an assumed regular geometry. Assumes elem_coords'
    own column count equals dim (J square) -- see jacobian_measure()
    below for the embedded case (dim < elem_coords.shape[1])."""
    J = dN_natural @ elem_coords
    detJ = np.linalg.det(J)
    return J, detJ


def jacobian_measure(dN_natural, elem_coords):
    """Like jacobian() above, but returns the length/area/volume SCALE
    FACTOR appropriate for any embedding, not just the dim==embedding-
    dimension case jacobian()'s own det(J) assumes: ordinary det(J)
    when the element's parametric dimension matches its embedding-
    space dimension (dim=2 in 2-D, dim=3 in 3-D -- every isoparametric
    continuum element in this package), or the Gram determinant
    sqrt(det(J @ J.T)) when it doesn't (dim=1 with 2-D/3-D nodal
    coordinates -- a beam or truss; dim=2 with 3-D nodal coordinates --
    a shell) -- the standard parametric curve-length/surface-area
    element formula. IDENTICAL to abs(det(J)) in the square case, so
    this is a strict generalization of jacobian()'s own detJ, not a
    different formula with a seam between the two cases.

    Used by Element.mass()/lumped_mass() below so any element relying
    on the GENERIC Gauss-quadrature mass loop (i.e. one that does not
    override mass() itself) integrates correctly regardless of its
    embedding -- found and fixed doing the Wave 6 (docs/consolidated_
    future_roadmap.md items 31-33) explicit-dynamics work: TrussTL2D
    (dim=1, 2-D nodal coordinates) and every shell formulation (dim=2,
    3-D nodal coordinates) hit the plain square-J jacobian() the moment
    assemble_mass()/assemble_lumped_mass() was actually exercised on
    them for the first time -- previously invisible because no
    existing caller in this package had assembled a mass matrix for a
    1-D-in-N-D or 2-D-in-3-D element before explicit dynamics needed
    one; beams.py's own Beam2D* classes were already unaffected only
    because they override mass() with a closed-form Hermite formula
    that never calls jacobian() at all."""
    J = dN_natural @ elem_coords
    if J.shape[0] == J.shape[1]:
        return np.linalg.det(J)
    return np.sqrt(max(np.linalg.det(J @ J.T), 0.0))


def spurious_zero_energy_modes(elem, elem_coords, D, thickness=1.0, tol=1e-8):
    """Detects hourglass (spurious zero-energy) modes introduced by
    reduced integration, WITHOUT needing to know the theoretical
    rigid-body-mode count for this element type. Full integration is
    taken as the reference (it never introduces spurious zero-energy
    modes for the formulations in this package); any eigenvalue of
    ke_reduced that full integration says should be nonzero, but that
    reduced integration reports as ~zero, is a genuine spurious mode --
    deformation the reduced quadrature literally can't see because it
    lands exactly on strain-free sampling points.

    Returns (n_spurious, eigenvalues_of_ke_reduced). For a single
    unconnected Hex8 under 1-point reduced integration, this returns
    12 -- a textbook, independently-verifiable number (rank(ke) <= 6
    from a single-point Bᵀ D B product, vs. 18 nonzero eigenvalues
    under full integration, i.e. 24 - 6 rigid-body modes)."""
    ke_full = elem.full_stiffness(elem_coords, D, thickness)
    ke_reduced = elem.reduced_stiffness(elem_coords, D, thickness)
    eig_full = np.linalg.eigvalsh(ke_full)
    eig_reduced = np.linalg.eigvalsh(ke_reduced)
    scale = max(np.max(np.abs(eig_full)), 1e-30)
    rank_full = int(np.sum(np.abs(eig_full) > tol * scale))
    rank_reduced = int(np.sum(np.abs(eig_reduced) > tol * scale))
    return rank_full - rank_reduced, eig_reduced


# Default per-node DOF names by DOFs-per-node count (see Element.dof_names).
_DEFAULT_DOF_NAMES = {
    1: ("u",),
    2: ("ux", "uy"),
    3: ("ux", "uy", "uz"),
    6: ("ux", "uy", "uz", "rx", "ry", "rz"),
}
# Shorthand accepted everywhere a DOF name is (applied only where the canonical name exists).
_COMMON_DOF_ALIASES = {"u": "ux", "v": "uy", "w": "uz"}


# =====================================================================
# Base element
# =====================================================================
class Element:
    n_nodes = None
    dofs_per_node = None
    dim = None            # dimension of the natural-coordinate space
    gauss_order = 2        # points per direction for the default Gauss loop
    quadrature_family = "tensor"   # "tensor" (gauss_product(), the default
                                    # -- Quad4/Quad8/Hex8/Hex20/plate/shell)
                                    # or "simplex" (tri_quadrature()/
                                    # tet_quadrature(), Wave 15 item 125/126
                                    # -- Tri3/Tri6/Tet4/Tet10 override this).
                                    # A single source-of-truth marker so
                                    # mesh_transform.py's MeshTransformation
                                    # (Wave 11 item 106) knows which
                                    # quadrature engine to precompute with,
                                    # rather than re-deriving that fact from
                                    # which stiffness() method got overridden.
    translational_dof_mask = None   # None -> every DOF is translational
                                     # (correct default for Quad4/Hex8);
                                     # override with e.g. [True, False, False]
                                     # for elements that mix translation and
                                     # rotation DOFs (plate, beam) -- see
                                     # lumped_mass() for why this matters.

    # Named per-node DOFs (v1.0.1), so BCs / loads / results can say "uy" instead of a bare index.
    # dof_names: tuple in LOCAL DOF order, or None -> a default is derived from dofs_per_node
    # (1: u | 2: ux,uy | 3: ux,uy,uz | 6: ux,uy,uz,rx,ry,rz). Elements whose DOFs are NOT those
    # (2-D beams, the Mindlin plate) override it. dof_aliases: extra accepted spellings -> canonical name.
    dof_names = None
    dof_aliases = {}

    def local_dof_names(self):
        """Per-node DOF names in local order (what fix_dofs / add_nodal_force accept instead of an
        integer index), or None if this element's DOFs have no standard names."""
        if self.dof_names is not None:
            return tuple(self.dof_names)
        return _DEFAULT_DOF_NAMES.get(self.dofs_per_node)

    def local_dof_aliases(self):
        """{alternative spelling: canonical name}: the element's own aliases plus the common
        u/v/w -> ux/uy/uz shorthand (only where the canonical name exists)."""
        names = self.local_dof_names() or ()
        out = {a: c for a, c in _COMMON_DOF_ALIASES.items() if c in names}
        out.update({a.lower(): c for a, c in self.dof_aliases.items()})
        return out

    def dof_index(self, name):
        """Local index of a DOF given by name (case-insensitive, aliases and 'x'/'y'/'z' shorthand
        accepted) or integer; ValueError listing the valid names otherwise."""
        names = self.local_dof_names()
        if isinstance(name, (int,)) or hasattr(name, "__index__"):
            i = int(name)
            if not 0 <= i < self.dofs_per_node:
                raise ValueError(f"DOF index {i} out of range 0..{self.dofs_per_node - 1} for "
                                 f"{type(self).__name__}.")
            return i
        if names is None:
            raise ValueError(f"{type(self).__name__} has {self.dofs_per_node} DOF(s) per node with no "
                             f"standard names; use integer indices 0..{self.dofs_per_node - 1}.")
        key = str(name).strip().lower()
        low = [n.lower() for n in names]
        if key in low:
            return low.index(key)
        if key in ("x", "y", "z") and f"u{key}" in low:
            return low.index(f"u{key}")
        canon = self.local_dof_aliases().get(key)
        if canon is not None and canon.lower() in low:
            return low.index(canon.lower())
        raise ValueError(f"unknown DOF name {name!r} for {type(self).__name__}; valid names: "
                         f"{list(names)}" + (f" (aliases: {sorted(self.local_dof_aliases())})"
                                              if self.local_dof_aliases() else "") + ".")

    def shape_and_derivs(self, natural_coords):
        """Returns (N, dN_natural): N shape (n_nodes,), dN_natural shape
        (dim, n_nodes). Must be implemented by every subclass."""
        raise NotImplementedError

    def _cached_shape_and_derivs(self, natural_coords):
        """Memoizing wrapper around shape_and_derivs() -- Wave 0 item 4
        of docs/consolidated_future_roadmap.md, closing the gap flagged
        in docs/fem_implementation_lessons.md's appendix: shape_and_derivs()
        recomputes N/dN from closed-form expressions on every call, with
        the book's reference-element O(n*i_d^2) -> O(n*i_d) payoff for
        higher-order elements unrealized.

        Every element in a mesh assembled through one formulation
        instance (solver.assemble_stiffness()/assemble_mass() reuse the
        SAME formulation object across the whole per-block element loop
        -- see solver.py) evaluates shape_and_derivs() at the SAME
        small, fixed set of Gauss-point natural coordinates (determined
        by self.gauss_order/self.dim, both class-level constants), so
        (N, dN_natural) at a given natural_coords is identical across
        every element in the mesh -- safe to compute once per reference-
        element Gauss point and reuse, rather than re-evaluating shape
        functions from scratch (polynomial/trig closed forms) on every
        single element. The cache lives on the instance (self.__dict__,
        not the class), so it never leaks across different formulation
        objects and needs no explicit invalidation -- a formulation's
        shape functions are a fixed property of its natural coordinates,
        never mutated after construction.

        Cache key: the natural_coords argument itself, which every call
        site in this package already passes as a tuple of floats (from
        gauss_product()/tet_quadrature_4pt(), or a literal tuple) --
        hashable by construction, no extra key-building needed.

        Returned (N, dN) arrays are marked read-only (setflags(write=
        False)) so an accidental in-place mutation by a caller raises
        immediately instead of silently corrupting the shared cache for
        every other element that reuses this same natural_coords entry."""
        cache = self.__dict__.setdefault('_shape_cache', {})
        hit = cache.get(natural_coords)
        if hit is None:
            N, dN = self.shape_and_derivs(natural_coords)
            N = np.asarray(N)
            dN = np.asarray(dN)
            N.setflags(write=False)
            dN.setflags(write=False)
            hit = (N, dN)
            cache[natural_coords] = hit
        return hit

    def N_matrix(self, N):
        """Generic block-diagonal expansion: scalar shape functions N
        (n_nodes,) -> (dofs_per_node, n_nodes*dofs_per_node) matrix
        mapping nodal DOFs to the physical field. Valid whenever every
        DOF at a node is interpolated by the SAME shape function (true
        for Quad4/Hex8/plate translations+rotations -- not true for the
        Hermite beam, which overrides mass() directly instead)."""
        npn = self.dofs_per_node
        Nm = np.zeros((npn, self.n_nodes * npn))
        for a in range(self.n_nodes):
            for k in range(npn):
                Nm[k, npn * a + k] = N[a]
        return Nm

    def B_matrix(self, natural_coords, elem_coords):
        """Physics-specific strain-displacement matrix at a natural
        coordinate, plus detJ. Implemented per element subclass."""
        raise NotImplementedError

    def stiffness(self, elem_coords, D, thickness=1.0, gauss_order=None):
        """Generic Gauss-integrated element stiffness: ke = sum_gp
        Bᵀ D B |J| w * thickness. Shared by every element that defines
        B_matrix() -- Quad4PlaneStress and Hex8Solid3D use this
        unmodified; Quad4MindlinPlate and Beam2DEulerBernoulli override
        it (see their docstrings for why).

        gauss_order: number of Gauss points per direction for THIS
        call, overriding self.gauss_order (the element's default/full
        order). None uses self.gauss_order. Pass a lower order for
        reduced integration -- see full_stiffness()/reduced_stiffness()
        for named convenience wrappers, and the module docstring's
        "Full vs. reduced integration" section for what each buys you."""
        order = self.gauss_order if gauss_order is None else gauss_order
        pts, wts = gauss_product(order, self.dim)
        n_total = self.n_nodes * self.dofs_per_node
        ke = np.zeros((n_total, n_total))
        for p, w in zip(pts, wts):
            B, detJ = self.B_matrix(p, elem_coords)
            ke += (B.T @ D @ B) * detJ * w * thickness
        return ke

    def full_stiffness(self, elem_coords, D, thickness=1.0):
        """Exact ('full') integration: self.gauss_order points per
        direction -- exact for this element's stiffness polynomial. The
        safe default (what solver.assemble_stiffness() uses when no
        integration scheme is specified), but KNOWN TO LOCK (be
        artificially stiff) for bending-dominated problems with
        elements that can't represent pure bending curvature without
        spurious shear strain -- Quad4 and Hex8 in particular, see
        cantilever_beam_3d_fem.py and main.py's integration-scheme demo."""
        return self.stiffness(elem_coords, D, thickness, gauss_order=self.gauss_order)

    def reduced_stiffness(self, elem_coords, D, thickness=1.0):
        """Uniform reduced integration: one Gauss point fewer per
        direction than full (minimum 1). Relieves locking -- the
        element becomes more flexible, closer to the true bending
        stiffness -- but can make the element RANK DEFICIENT (spurious
        'hourglass' deformation modes that carry zero strain energy at
        every sampling point the reduced rule uses). Check
        spurious_zero_energy_modes() before trusting a reduced-
        integration result, especially on a coarse or loosely
        constrained mesh. See hourglass_stabilized_stiffness() below for
        a version of this that adds back exactly enough stiffness to
        make reduced integration safe to actually deploy."""
        order = max(1, self.gauss_order - 1)
        return self.stiffness(elem_coords, D, thickness, gauss_order=order)

    def hourglass_stabilized_stiffness(self, elem_coords, D, thickness=1.0,
                                        c_hg=0.1, tol=1e-6):
        """Wave 2 item 12 (docs/consolidated_future_roadmap.md, source
        nonlinear_fem_lessons.md's appendix, Sec.8.7.3-8.7.6): adds a
        Flanagan-Belytschko/Belytschko-Bindeman-style PERTURBATION
        STIFFNESS to reduced_stiffness(), so reduced integration becomes
        safe to actually deploy instead of purely diagnostic (see that
        method's own docstring, and spurious_zero_energy_modes(), which
        this function reuses the same eigenvalue-comparison IDEA from --
        "any eigenvalue reduced integration reports as ~zero that full
        integration says should be nonzero is a spurious mode" -- but
        here CORRECTS it instead of just detecting it).

        THE APPROACH (a generic, numerically-derived equivalent of the
        literature's closed-form shape-vector formulas, not a
        transcription of them): the classic Flanagan-Belytschko/
        Belytschko-Bindeman formulas are tabulated PER ELEMENT TYPE
        (specific hourglass base vectors for an 8-node hex, a different
        set for other topologies) and derived from that element's own
        assumed-strain field -- hand-transcribing one from memory risks
        exactly the kind of sign/ordering/convention error this
        project's own history has hit before (e.g. the Koiter-Newton
        corrector's sign-error war story, nonlinear_solver.py's own
        docstring). Instead, this function derives the SAME kind of
        correction directly from the two matrices every element already
        provides (full_stiffness()/reduced_stiffness()), via a
        DEFLATED EIGENPROBLEM:

        1. Eigendecompose ke_reduced. Its near-zero eigenvectors (below
           `tol` relative to ke_full's own eigenvalue scale) span EXACTLY
           ker(B_reduced) -- every displacement mode invisible to the
           reduced quadrature's single sampling point, rigid-body modes
           and hourglass modes both (see spurious_zero_energy_modes()'s
           own docstring for why full integration is trusted as the
           "this SHOULD be nonzero" reference).
        2. Project ke_full onto exactly that null space (N.T @ ke_full
           @ N, N = the null eigenvectors) and eigendecompose THAT small
           matrix too. This "deflated" eigenproblem cleanly separates
           the null space into its two physically distinct parts: modes
           with ~zero energy in ke_full TOO (genuine rigid-body motion --
           correctly invisible to both full and reduced integration,
           left alone) vs. modes with REAL, nonzero ke_full energy
           (spurious/hourglass -- invisible to reduced integration only
           because the single sample point happens to sit exactly where
           their strain vanishes).
        3. Add back c_hg (default 0.1, matching Belytschko-Bindeman's
           own "~0.1 of the true element stiffness eigenvalue" scaling,
           quoted in nonlinear_fem_lessons.md) times each hourglass
           mode's OWN true (full-integration) energy, via that exact
           mode shape: K_stab = sum_k c_hg * lambda_k * phi_k (x) phi_k.

        WHY THIS IS SAFE (proven, not just hoped -- see
        tests/test_hourglass_stabilization.py for the numerical
        confirmation): K_stab's range is, by construction, a subset of
        ker(B_reduced) (every phi_k is a linear combination of ke_
        reduced's own null eigenvectors), so K_stab never touches any
        mode reduced integration was already computing correctly --
        this is purely an ADDITIVE correction in the blind spot, not a
        modification of the existing reduced-integration response. And
        because ker(B_reduced) splits exactly into {rigid body} (excluded
        above) + {hourglass} (stabilized), and any AFFINE/constant-strain
        displacement field decomposes into a rigid part (which lies IN
        ker(B_reduced), hence is exactly one of the excluded directions)
        plus a pure-strain part (which lies in range(B_reduced.T), the
        Euclidean-orthogonal COMPLEMENT of ker(B_reduced) for any matrix
        -- a basic linear-algebra fact, not element-specific), K_stab
        vanishes EXACTLY on every affine field: this stabilizer cannot
        corrupt a patch test, by construction, the same "limitation
        principle" (Stolarski & Belytschko 1987) nonlinear_fem_lessons.md
        itself names as the theoretical requirement any correct locking/
        hourglass fix must satisfy.

        c_hg: the stabilization fraction -- 0 gives back plain
        reduced_stiffness() (still spurious), 1 would fully restore
        each hourglass mode's true full-integration stiffness (safe,
        but that itself is one way to just re-lock the volumetric
        response for a near-incompressible material -- "you will cure
        hourglassing by locking the mesh," per nonlinear_fem_lessons.md
        -- so this is deliberately a caller-tunable knob, not a fixed
        internal constant). tol: relative eigenvalue threshold (against
        ke_full's own largest |eigenvalue|) for deciding "near zero" at
        both stages above.

        Falls back to reduced_stiffness() itself, unchanged, whenever
        reduced integration already has full rank for this element (no
        null space, or every null direction turns out to be rigid-body)
        -- e.g. a single-point element like Tet4Solid3D/Tri3PlaneStress
        where full_stiffness()==reduced_stiffness() already; calling
        this method on those is harmless, just a no-op."""
        ke_full = self.full_stiffness(elem_coords, D, thickness)
        ke_reduced = self.reduced_stiffness(elem_coords, D, thickness)
        scale = max(np.max(np.abs(np.linalg.eigvalsh(ke_full))), 1e-30)

        eigvals_r, eigvecs_r = np.linalg.eigh(ke_reduced)
        null_mask = np.abs(eigvals_r) < tol * scale
        if not np.any(null_mask):
            return ke_reduced   # already full rank -- nothing to stabilize
        N = eigvecs_r[:, null_mask]

        M = N.T @ ke_full @ N
        eigvals_m, eigvecs_m = np.linalg.eigh(M)
        hg_mask = np.abs(eigvals_m) >= tol * scale   # real energy in ke_full -> spurious here
        if not np.any(hg_mask):
            return ke_reduced   # every null direction is genuine rigid-body motion

        Phi_hg = N @ eigvecs_m[:, hg_mask]           # (n_dof, n_hg), still orthonormal columns
        hg_energy = eigvals_m[hg_mask]
        K_stab = (Phi_hg * (c_hg * hg_energy)) @ Phi_hg.T
        return ke_reduced + K_stab

    # -------------------------------------------------------------
    # Nonlinear extension point (geometric nonlinearity, Module 8).
    # Every element in this package is LINEAR by default: internal
    # force is just ke @ u and the tangent is just ke, so nothing
    # about Quad4/Hex8/plate/beam has to change to support the new
    # nonlinear_solver.py driver. An element that IS geometrically
    # nonlinear (e.g. TrussTL2D below) overrides both methods with a
    # Total-Lagrangian formulation instead -- the driver itself never
    # knows or cares which case it's calling.
    # -------------------------------------------------------------
    def internal_force(self, elem_coords, u_elem, D, thickness=1.0, **kwargs):
        """Linear default: f_int = ke @ u_elem, using the SAME ke as
        stiffness(). Correct for any linear element with no further
        work; nonlinear elements override this with the actual
        (generally nonlinear-in-u) internal force."""
        ke = self.stiffness(elem_coords, D, thickness, **kwargs)
        return ke @ u_elem

    def tangent_stiffness(self, elem_coords, u_elem, D, thickness=1.0, **kwargs):
        """Linear default: K_T = ke, independent of u_elem (a linear
        element's tangent IS its stiffness, at every displacement).
        Nonlinear elements override this with d(internal_force)/du."""
        return self.stiffness(elem_coords, D, thickness, **kwargs)

    # -------------------------------------------------------------
    # Linear buckling extension point (Module 19, general-purpose
    # extensions roadmap Phase 4). A sibling to internal_force()/
    # tangent_stiffness() above, but for a DIFFERENT kind of
    # nonlinearity question: not "what is K_T at displacement u_elem"
    # but "how much does a given axial force N reduce (or increase)
    # this element's transverse stiffness". This is exactly the
    # SECOND-order/geometric part of tangent_stiffness() -- e.g.
    # TrussTL2D.tangent_stiffness()'s own K_geometric = (S*A/L0)*I
    # term -- evaluated ONCE at a fixed reference axial force rather
    # than recomputed every Newton iteration from a full nonlinear
    # displacement state, which is why this takes N directly (a
    # scalar reference force) instead of u_elem.
    #
    # Sign convention: N is TENSION-POSITIVE, matching TrussTL2D's
    # existing S (2nd Piola-Kirchhoff stress) convention -- N>0
    # stiffens (a taut cable resists transverse motion more), N<0
    # (compression) softens, and can drive the total stiffness
    # K_elastic + geometric_stiffness(N) singular at a large enough
    # |N| -- that's buckling. Undefined (raises) for the generic
    # linear default: an element with no nonlinear strain-displacement
    # kinematics to build K_sigma from has no notion of buckling under
    # its own stiffness() alone (a Quad4PlaneStress panel or a
    # Beam2DMindlinPlate needs a genuinely different, 2-D formulation
    # not implemented here -- see docs/general_purpose_extensions_
    # roadmap.md Section 6 for scope). Elements that DO support this
    # (Beam2DEulerBernoulli, Beam3DEulerBernoulli, TrussTL2D) override
    # it with their own closed-form K_sigma."""
    def geometric_stiffness(self, elem_coords, N, thickness=1.0, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} has no geometric_stiffness() -- linear "
            f"buckling needs an element with known second-order/geometric "
            f"strain-displacement kinematics (see Element.geometric_stiffness's "
            f"docstring); this element only has linear kinematics.")

    def mass(self, elem_coords, rho_matrix, thickness=1.0):
        """Generic consistent mass matrix: me = sum_gp Nmᵀ rho_matrix Nm
        |J| w * thickness, where rho_matrix (dofs_per_node x
        dofs_per_node) lets rotational DOFs carry a different inertia
        than translational ones (e.g. the plate's rotary inertia).
        Uses jacobian_measure() (not jacobian() directly) so this
        generic loop also integrates correctly for an element whose
        parametric dim is LOWER than its nodal-coordinate embedding
        dimension (e.g. a would-be 1-D-in-2-D truss/beam or 2-D-in-3-D
        shell that relies on this default rather than overriding
        mass() itself) -- see jacobian_measure()'s own docstring for
        why plain jacobian()'s square-J det() is not enough here."""
        pts, wts = gauss_product(self.gauss_order, self.dim)
        n_total = self.n_nodes * self.dofs_per_node
        me = np.zeros((n_total, n_total))
        for p, w in zip(pts, wts):
            N, dN = self._cached_shape_and_derivs(p)
            detJ = jacobian_measure(dN, elem_coords)
            Nm = self.N_matrix(N)
            me += (Nm.T @ rho_matrix @ Nm) * detJ * w * thickness
        return me

    def lumped_mass(self, elem_coords, rho_matrix, thickness=1.0):
        """HRZ (Hinton-Rock-Zienkiewicz) diagonal scaling: take the
        DIAGONAL of the consistent mass matrix (always positive, since
        a consistent mass matrix is positive definite) and rescale it
        so that, independently for EACH translational direction, that
        direction's own diagonal entries sum to the exact physical
        total mass -- so a rigid-body translation in ANY one direction
        alone carries the correct total mass and kinetic energy, while
        every lumped entry -- including rotational ones -- stays
        positive.

        PER-DIRECTION, not one factor shared across every translational
        direction at once (Wave 6 item 31/32 fix, docs/consolidated_
        future_roadmap.md, found directly validating solve_transient_
        explicit_nonlinear()/critical_timestep_local() against an exact
        bar-wave-speed ground truth on TrussTL2D): an element with N
        translational directions per node (N=2 for any planar element
        with both in-plane translations tracked -- Quad4/TrussTL2D/
        Beam2DCorotational; N=3 for a shell's x,y,z) has its raw
        consistent-mass diagonal entries, summed ACROSS ALL N
        directions together, equal to N times a single direction's own
        total mass (each direction integrates to ~the same physical
        total mass independently, by symmetry, so summing N of them
        gives N times that). The single combined correction factor c =
        total_mass / (sum over ALL directions combined) this used to
        compute is therefore ~1/N of what any ONE direction actually
        needs -- silently under-lumping every translational entry by a
        factor of N (2x for Quad4/TrussTL2D, 3x for a shell) the moment
        more than one translational direction is tracked. Invisible
        until Wave 6 actually exercised assemble_lumped_mass() on a
        multi-directional element for the first time (previously only
        ever exercised on Beam2DEulerBernoulli, which tracks exactly
        ONE translational direction -- w -- so N=1 there and the bug
        has no effect, which is why it went unnoticed). Fixed by
        computing the physical total_mass ONCE (direction-independent,
        same rho_trans/measure integral as before) but the SCALE
        FACTOR separately per translational local index k, applied only
        to that direction's own diagonal entries.

        This deliberately does NOT use naive row-sum lumping. A
        consistent mass matrix for elements with rotational DOFs (beam,
        plate) has NEGATIVE off-diagonal entries in those rows, so
        summing a whole row can and does produce a negative "mass" --
        catastrophic for explicit time integration (found and fixed
        while validating solve_transient_explicit(), see main.py)."""
        me = self.mass(elem_coords, rho_matrix, thickness)
        diag = np.diag(me).copy()
        npn = self.dofs_per_node
        mask = self.translational_dof_mask
        if mask is None:
            mask = [True] * npn
        trans_local = [k for k in range(npn) if mask[k]]

        rho_trans = rho_matrix[trans_local[0], trans_local[0]] \
            if np.ndim(rho_matrix) == 2 else rho_matrix
        pts, wts = gauss_product(self.gauss_order, self.dim)
        total_mass = 0.0
        for p, w in zip(pts, wts):
            _, dN = self._cached_shape_and_derivs(p)
            # jacobian_measure(), not jacobian() directly -- same
            # embedded-element reasoning as mass() above (see that
            # method's own docstring and jacobian_measure()'s).
            measure = abs(jacobian_measure(dN, elem_coords))
            total_mass += rho_trans * measure * w * thickness

        out = diag.copy()
        for k in trans_local:
            idx_k = [npn * a + k for a in range(self.n_nodes)]
            S_k = diag[idx_k].sum()
            out[idx_k] = diag[idx_k] * (total_mass / S_k)
        return np.diag(out)


# =====================================================================
# Quad4 -- plane stress / plane strain continuum
# =====================================================================
