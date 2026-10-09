"""
mixed_assembly.py -- Wave 11 item 110 (docs/consolidated_future_
roadmap.md, source tensormesh_analysis_report.md's "Mixed Assembly"
section: `MixedElementAssembler`/`Field`/`BlockLayout` -- independent
trial/test polynomial order per field, each `(field_i, field_j)` block
extracted by zeroing the other fields with a runtime bilinearity guard,
topological (not mesh-order-tied) DOF maps).

Wave 16 item 130 addendum (2026-09-18, source: a later, live comparison
against TensorMesh's actual `Mixed Assembly` documentation page,
https://docs.tensor-mesh.com/user_guide/mixed_assembly.html): that
comparison noted `check_bilinearity()` below was opt-in/standalone,
while TensorMesh runs its own analogous guard AUTOMATICALLY inside
every mixed-assembler call. `assemble_mixed_tet10_p1()` now does the
same by default (`verify_bilinearity=True`) -- see that function's own
docstring and `_verify_mixed_blocks()`/`_tet10_p1_direct_forms()` below
for what the automatic guard does and, honestly, doesn't protect
against here (there's no user-supplied `forward()` in this codebase's
version of mixed assembly, so the guard is a regression check against
a future code edit, not a check on end-user-supplied math).

Second attempt, scoped down deliberately
------------------------------------------
`tensormesh_comparative_analysis.md` Sec.6.4 already named true mixed
(LBB-stable, block) assembly as "the architecturally correct fix" for
BOTH the 3-D solids' volumetric-locking gap (Wave 2 item 11 instead
solved that with B-bar -- a real, validated fix, but not this
architecturally-different alternative) and the corotational shell's own
mixed-formulation internal unknown (`docs/shells.md` item 19, dead ends
4/7 -- that attempt sub-iterated the internal unknown rather than
promoting it to a genuinely independent field, and produced Newton
indefiniteness no variant fixed). Per this item's own roadmap row, the
SHELL case is explicitly NOT reattempted here -- only the lower-risk
3-D SOLID case is, and only in its LINEAR (not corotational/nonlinear)
form. The two lightweight, genuinely reusable building blocks
(`BlockDofLayout`, `check_bilinearity`) are useful on their own
regardless of whether the solid case below is ever extended further --
see the roadmap row's own framing.

The concrete formulation: Taylor-Hood P2-P1 on Tet10/Tet4
------------------------------------------------------------
Rather than inventing a new element/shape-function pair (the riskiest
possible way to attempt this, and the closest analogue to what the
shell attempt's own dead ends ran into), this reuses TWO ALREADY-
VALIDATED element formulations this package has -- `Tet10Solid3D`
(quadratic, 10-node, already used throughout this codebase) for
displacement, and `Tet4Solid3D`'s own shape functions (evaluated at
Tet10's own first 4 "corner" nodes -- see Tet10Solid3D's own docstring:
its node-numbering convention places the 4 corners first, node-for-
node identical to Tet4Solid3D's own corner convention) for a
PIECEWISE-LINEAR pressure field. This is the classical Taylor-Hood
P2-P1 pair (Taylor & Hood 1973) -- the single most standard, textbook-
proven inf-sup-stable mixed pairing in the FEM literature -- built here
with ZERO new shape-function derivation, only new BLOCK-ASSEMBLY
machinery (`BlockDofLayout`, the mixed weak form below, the bordered
saddle-point solve). This is what makes this attempt lower-risk than
the shell case: no new interpolation theory, only new bookkeeping.

Weak form (linear, isotropic, near-incompressible elasticity; see
`tests/test_mixed_assembly.py`'s own patch-test docstring for the
decisive correctness check):

    find (u, p) such that for all (v, q):
        integral( eps(v) : D_dev : eps(u) ) dV + integral( p * div(v) ) dV = integral( f . v ) dV
        integral( q * p / kappa )           dV - integral( q * div(u) ) dV = 0

`D_dev` is `D_solid3d(mat)`'s own DEVIATORIC part (see
`D_deviatoric_3d()` below) -- the volumetric part of the stress is
carried entirely by `p` instead, which is exactly what relieves
volumetric locking as `nu -> 0.5` (`kappa = E/(3*(1-2*nu)) -> inf`,
and `1/kappa -> 0`, recovering the EXACT incompressible saddle-point
system in the limit -- this module uses a large-but-FINITE `kappa`, the
"near-incompressible" case, so the resulting system stays nonsingular
without needing a separate pressure-pinning/null-space treatment).

Block system, per element and assembled globally:

    [ Kuu   Kup ] [u]   [F]
    [ Kup^T -Kpp] [p] = [0]

`Kuu = integral(Bu^T D_dev Bu) dV`, `Kup[i,a] = integral((m^T Bu)_i * Np_a) dV`
(`m = [1,1,1,0,0,0]`, the trace/divergence operator on Voigt strain),
`Kpp = integral(Np^T Np) dV / kappa`.
"""
__author__ = "Abhijeet"
import numpy as np

from .elements.base import jacobian
from .elements.solids import Tet4Solid3D, Tet10Solid3D
from .elements.base import tet_quadrature_4pt


# =====================================================================
# BlockDofLayout -- topological (not mesh-order-tied) named-field DOF
# bookkeeping, TensorMesh's own `BlockLayout` idea, reimplemented
# natively (plain NumPy index bookkeeping, no tensormesh dependency).
# =====================================================================
class BlockDofLayout:
    """A named-field DOF layout: several fields, each with its own
    size, concatenated into one flat vector with a fixed field order.
    `split()`/`cat()` are exact inverses of each other -- the whole
    point is that a caller solving a coupled block system (e.g. this
    module's own mixed (u,p) saddle-point system) can work with named
    sub-vectors/sub-matrices instead of hand-tracking numeric offsets.
    """

    def __init__(self, field_sizes):
        self.names = list(field_sizes)
        self.sizes = dict(field_sizes)
        self.offsets = {}
        off = 0
        for name in self.names:
            self.offsets[name] = off
            off += self.sizes[name]
        self.total = off

    def slice(self, name):
        o = self.offsets[name]
        return slice(o, o + self.sizes[name])

    def split(self, vec):
        vec = np.asarray(vec)
        return {name: vec[self.slice(name)] for name in self.names}

    def cat(self, parts):
        return np.concatenate([np.asarray(parts[name]) for name in self.names])

    def zeros(self):
        return np.zeros(self.total)

    def block_slice_pair(self, name_row, name_col):
        return self.slice(name_row), self.slice(name_col)


# =====================================================================
# Bilinearity self-check -- the cheap, worth-copying correctness guard
# tensormesh_analysis_report.md's own MixedElementAssembler describes:
# evaluate the coupled integrand with one field's contribution zeroed
# (i.e. probed via additivity/homogeneity on random vectors) and
# confirm the result is genuinely LINEAR in what remains.
# =====================================================================
def check_bilinearity(form_fn, dim_a, dim_b, rng=None, atol=1e-8, rtol=1e-6):
    """`form_fn(a_vec, b_vec) -> float`, presumed to be a BILINEAR form
    (i.e. `form_fn(a, b) = a^T @ K_ab @ b` for some fixed matrix
    `K_ab`). Confirms this holds by testing additivity AND homogeneity
    in EACH argument separately, on random vectors -- independent of
    whether the caller can even construct `K_ab` explicitly (useful as
    a standalone guard on any coupled-block assembly routine, not just
    this module's own Tet10-P1 blocks below).

    Returns True/False (does not raise) -- callers that want a hard
    assertion should wrap this themselves, matching every other pure
    "check_*"-style helper in this package (e.g. Mesh.check_quality())."""
    rng = rng if rng is not None else np.random.default_rng(0)
    a1, a2 = rng.standard_normal(dim_a), rng.standard_normal(dim_a)
    b1, b2 = rng.standard_normal(dim_b), rng.standard_normal(dim_b)
    c = float(rng.uniform(0.5, 2.0))

    checks = [
        np.isclose(form_fn(a1 + a2, b1), form_fn(a1, b1) + form_fn(a2, b1), atol=atol, rtol=rtol),
        np.isclose(form_fn(c * a1, b1), c * form_fn(a1, b1), atol=atol, rtol=rtol),
        np.isclose(form_fn(a1, b1 + b2), form_fn(a1, b1) + form_fn(a1, b2), atol=atol, rtol=rtol),
        np.isclose(form_fn(a1, c * b1), c * form_fn(a1, b1), atol=atol, rtol=rtol),
    ]
    return bool(all(checks))


# =====================================================================
# Deviatoric-only material matrix
# =====================================================================
def D_deviatoric_3d(mat):
    """The deviatoric-only part of `material.D_solid3d(mat)`:
    `D_dev = D_solid3d(mat) - kappa * (m @ m.T)`, `m = [1,1,1,0,0,0]`,
    `kappa = E/(3*(1-2*nu))`. Derived (and checked directly in
    tests/test_mixed_assembly.py) from `D_solid3d`'s own closed-form
    entries: writing `c = E/((1+nu)(1-2nu))`, its normal-normal block
    has diagonal `c*(1-nu)` and off-diagonal `c*nu`; their difference
    `c*(1-2nu)` equals `2*mu` (`mu = E/(2*(1+nu))`, `D_solid3d`'s own
    shear-diagonal value) regardless of `kappa` -- i.e. subtracting a
    pure `kappa * m m^T` term from `D_solid3d` leaves EXACTLY the
    standard deviatoric projection `2*mu*(I - (1/3) m m^T)` on the
    normal-normal block and leaves the shear block (where `m` is zero)
    completely untouched, matching the physical fact that shear stress
    has no volumetric component."""
    E, nu = mat.E, mat.nu
    from .material import D_solid3d
    kappa = E / (3.0 * (1.0 - 2.0 * nu))
    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    return D_solid3d(mat) - kappa * np.outer(m, m)


# =====================================================================
# Per-element Taylor-Hood P2-P1 mixed blocks
# =====================================================================
_M_TRACE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _tet10_p1_direct_forms(elem_coords, D_dev, kappa):
    """Wave 16 item 130 (docs/consolidated_future_roadmap.md, source:
    TensorMesh's `Mixed Assembly` documentation page): a SECOND,
    independently-structured evaluation of the SAME per-element weak
    form `tet10_p1_mixed_element_blocks()` computes -- used only to
    automatically guard that function's own output (see
    `assemble_mixed_tet10_p1(verify_bilinearity=True)` below).

    TensorMesh's `MixedElementAssembler` runs an automatic check on
    every user-supplied `forward(...)`, at the point where it is still
    just a black-box Python function of raw trial/test values (BEFORE
    it has been reduced to a block matrix): evaluate it with the other
    fields zeroed and confirm the result is bilinear, catching a
    mis-written weak form before it silently produces a wrong matrix.
    `fea_engine` has no user-supplied `forward()` -- the weak form here
    is a project-authored constant (Taylor & Hood 1973's classical
    pairing), not re-derived by end users on every call -- so there is
    no "user math error" for an automatic guard to catch the way
    TensorMesh's does. What an automatic guard here CAN still usefully
    catch is a REGRESSION: a future edit to `tet10_p1_mixed_element_
    blocks()` that accidentally introduces non-bilinear structure (a
    stray constant term, a branch depending on the DATA rather than the
    fixed geometry/material, etc.), or a future edit to just ONE of two
    independently-written expressions of the same physics that leaves
    them disagreeing. This function is that second, independent
    expression -- it computes each block's own scalar pairing directly
    from raw nodal DOF vectors via a fresh per-Gauss-point loop, rather
    than by multiplying against the already-assembled `Kuu`/`Kup`/`Kpp`
    matrices `tet10_p1_mixed_element_blocks()` returns.

    Returns (form_uu, form_up, form_pp), three `(a, b) -> float`
    closures:
      form_uu(u_a, u_b)  ~ u_a . Kuu . u_b   (u_a, u_b: (30,))
      form_up(u_a, p_b)  ~ u_a . Kup . p_b   (u_a: (30,), p_b: (4,))
      form_pp(p_a, p_b)  ~ p_a . Kpp . p_b   (p_a, p_b: (4,))
    """
    u_elem = Tet10Solid3D()
    points, weights = tet_quadrature_4pt()

    def form_uu(u_a, u_b):
        u_a = np.asarray(u_a); u_b = np.asarray(u_b)
        total = 0.0
        for pt, w in zip(points, weights):
            Bu, detJ = u_elem.B_matrix(pt, elem_coords)
            dV = abs(detJ) * w / 6.0
            strain_a = Bu @ u_a
            strain_b = Bu @ u_b
            total += float(strain_a @ D_dev @ strain_b) * dV
        return total

    def form_up(u_a, p_b):
        u_a = np.asarray(u_a); p_b = np.asarray(p_b)
        total = 0.0
        for pt, w in zip(points, weights):
            Bu, detJ = u_elem.B_matrix(pt, elem_coords)
            Np, _ = Tet4Solid3D().shape_and_derivs(pt)
            dV = abs(detJ) * w / 6.0
            div_a = float(_M_TRACE @ (Bu @ u_a))
            total += div_a * float(Np @ p_b) * dV
        return total

    def form_pp(p_a, p_b):
        p_a = np.asarray(p_a); p_b = np.asarray(p_b)
        total = 0.0
        for pt, w in zip(points, weights):
            _, detJ = u_elem.B_matrix(pt, elem_coords)
            Np, _ = Tet4Solid3D().shape_and_derivs(pt)
            dV = abs(detJ) * w / 6.0
            total += float(Np @ p_a) * float(Np @ p_b) * (dV / kappa)
        return total

    return form_uu, form_up, form_pp


def _verify_mixed_blocks(elem_coords, D_dev, kappa, Kuu, Kup, Kpp, rng=None):
    """Runs automatically from `assemble_mixed_tet10_p1(verify_bilinearity=
    True)` (the default) -- see `_tet10_p1_direct_forms()`'s own
    docstring for what this does and doesn't protect against. Cheap: a
    handful of random-vector evaluations against ONE representative
    element's own geometry (the SAME weak form is used for every
    straight-sided element in the mesh, so checking it once per
    assembly call, not once per element, matches TensorMesh's own
    "cheap... check" framing rather than adding a real per-element
    cost). Raises RuntimeError with a specific, actionable message on
    either kind of failure (non-bilinear structure, or direct-vs-matrix
    disagreement) rather than silently returning.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    form_uu, form_up, form_pp = _tet10_p1_direct_forms(elem_coords, D_dev, kappa)

    if not check_bilinearity(form_uu, 30, 30, rng=np.random.default_rng(rng.integers(1 << 30))):
        raise RuntimeError(
            "assemble_mixed_tet10_p1: the direct (u,u) weak-form evaluation "
            "failed the bilinearity self-check -- this indicates a regression "
            "in the Kuu physics (a non-bilinear term was introduced). See "
            "mixed_assembly.py's _tet10_p1_direct_forms() docstring.")
    if not check_bilinearity(form_up, 30, 4, rng=np.random.default_rng(rng.integers(1 << 30))):
        raise RuntimeError(
            "assemble_mixed_tet10_p1: the direct (u,p) weak-form evaluation "
            "failed the bilinearity self-check -- this indicates a regression "
            "in the Kup physics (a non-bilinear term was introduced). See "
            "mixed_assembly.py's _tet10_p1_direct_forms() docstring.")
    if not check_bilinearity(form_pp, 4, 4, rng=np.random.default_rng(rng.integers(1 << 30))):
        raise RuntimeError(
            "assemble_mixed_tet10_p1: the direct (p,p) weak-form evaluation "
            "failed the bilinearity self-check -- this indicates a regression "
            "in the Kpp physics (a non-bilinear term was introduced). See "
            "mixed_assembly.py's _tet10_p1_direct_forms() docstring.")

    u_a = rng.standard_normal(30); u_b = rng.standard_normal(30)
    p_a = rng.standard_normal(4); p_b = rng.standard_normal(4)
    checks = [
        (float(u_a @ Kuu @ u_b), form_uu(u_a, u_b), "Kuu"),
        (float(u_a @ Kup @ p_b), form_up(u_a, p_b), "Kup"),
        (float(p_a @ Kpp @ p_b), form_pp(p_a, p_b), "Kpp"),
    ]
    for matrix_val, direct_val, block_name in checks:
        if not np.isclose(matrix_val, direct_val, atol=1e-6, rtol=1e-6):
            raise RuntimeError(
                f"assemble_mixed_tet10_p1: the assembled {block_name} block "
                f"disagrees with the independent direct weak-form evaluation "
                f"({matrix_val!r} vs {direct_val!r}) -- this indicates the two "
                f"code paths (tet10_p1_mixed_element_blocks() and "
                f"_tet10_p1_direct_forms()) have drifted out of sync, most "
                f"likely from an edit to one without the other. See "
                f"mixed_assembly.py's _tet10_p1_direct_forms() docstring.")


def tet10_p1_mixed_element_blocks(elem_coords, D_dev, kappa):
    """(Kuu (30,30), Kup (30,4), Kpp (4,4)) for ONE straight-sided
    Tet10 element, via the SAME 4-point simplex quadrature
    `Tet10Solid3D.stiffness()` itself uses (`tet_quadrature_4pt()`) --
    `Tet10Solid3D.B_matrix()` supplies `Bu`/`detJ` at each point; the
    pressure shape values `Np` are `Tet4Solid3D.shape_and_derivs()`'s
    own `N` output at the SAME natural coordinates (a pure function of
    natural coordinates only, so no separate geometry/Jacobian call is
    needed for the pressure field -- its Jacobian is shared with the
    displacement field's own, both fields living on the identical
    physical tetrahedron)."""
    u_elem = Tet10Solid3D()
    points, weights = tet_quadrature_4pt()
    Kuu = np.zeros((30, 30))
    Kup = np.zeros((30, 4))
    Kpp = np.zeros((4, 4))
    for pt, w in zip(points, weights):
        Bu, detJ = u_elem.B_matrix(pt, elem_coords)          # (6,30)
        Np, _ = Tet4Solid3D().shape_and_derivs(pt)             # (4,)
        dV = abs(detJ) * w / 6.0
        Kuu += (Bu.T @ D_dev @ Bu) * dV
        div_row = _M_TRACE @ Bu                                # (30,)
        Kup += np.outer(div_row, Np) * dV
        Kpp += np.outer(Np, Np) * (dV / kappa)
    return Kuu, Kup, Kpp


# =====================================================================
# Global DOF maps and assembly
# =====================================================================
def build_pressure_dof_map(connectivity):
    """`connectivity`: `(n_elements, 10)` Tet10 global node-id array
    (columns 0-3 are corner nodes, per `Tet10Solid3D`'s own convention
    -- see that class's docstring). Pressure DOFs live only on corner
    nodes, so this builds a COMPACT map from "global node id" to
    "pressure dof index" (0..n_p-1) covering only the nodes that
    actually appear as a corner somewhere -- exactly the "topological,
    not mesh-order-tied" DOF map `tensormesh_analysis_report.md`'s own
    `BlockLayout` describes."""
    corner_ids = np.unique(connectivity[:, :4].reshape(-1))
    p_dof_of_node = {int(nid): i for i, nid in enumerate(corner_ids)}
    return corner_ids, p_dof_of_node


def assemble_mixed_tet10_p1(mesh, connectivity, mat, kappa=None, verify_bilinearity=True):
    """Global `(Kuu, Kup, Kpp)` for a Tet10 mesh, plus the `BlockDofLayout`
    and pressure-dof map a caller needs to apply BCs/interpret `p`.

    Parameters
    ----------
    mesh : mesh.Mesh
        `.nodes` gives every node's coordinates (corner AND mid-edge --
        `Mesh` itself doesn't distinguish them, `Tet10Solid3D`'s own
        node-ordering convention is what makes columns 0-3 of
        `connectivity` the corners).
    connectivity : (n_elements, 10) int ndarray
        Defaults are NOT read from `mesh.elements` automatically (unlike
        `MeshTransformation`) since a caller may be assembling only a
        SUBSET of a larger mixed mesh; pass `mesh.elements` explicitly
        for the common single-block case.
    mat : material.Material
    kappa : float or None
        Bulk modulus; defaults to `mat.E/(3*(1-2*mat.nu))` (the EXACT
        isotropic-elasticity value, i.e. this reduces to ordinary,
        fully-consistent linear elasticity by default -- pass a larger
        `kappa` explicitly to probe the near-incompressible limit
        beyond what `mat.nu` alone implies).
    verify_bilinearity : bool
        Wave 16 item 130 (docs/consolidated_future_roadmap.md, source:
        TensorMesh's `Mixed Assembly` page's automatic per-forward()
        zero-evaluation guard). Default True: before assembling the
        mesh, runs `_verify_mixed_blocks()` ONCE against the first
        element's own geometry -- confirms `tet10_p1_mixed_element_
        blocks()`'s three blocks (a) are genuinely bilinear and (b)
        agree with an independently-written direct evaluation of the
        same weak form (`_tet10_p1_direct_forms()`) -- and raises a
        specific RuntimeError naming which block disagreed if either
        check fails, rather than silently assembling a wrong system.
        This is automatic (unlike `check_bilinearity()` itself, which
        remains available as a standalone, opt-in helper -- see the
        module's own top-level docstring for that history) to match
        TensorMesh's own choice of running its guard inside the
        assembly path rather than leaving it for the caller to
        remember. Set False to skip it (e.g. inside a tight loop over
        many small meshes with the same material, where the check has
        already passed once and its modest constant cost isn't worth
        repeating) -- this never changes the assembled result, only
        whether it's checked before returning.

    Returns
    -------
    (layout, Kuu, Kup, Kpp, p_dof_of_node)
    """
    connectivity = np.asarray(connectivity)
    n_u_dof = mesh.nodes.shape[0] * 3
    corner_ids, p_dof_of_node = build_pressure_dof_map(connectivity)
    n_p_dof = len(corner_ids)
    layout = BlockDofLayout({"u": n_u_dof, "p": n_p_dof})

    D_dev = D_deviatoric_3d(mat)
    if kappa is None:
        kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu))

    Kuu = np.zeros((n_u_dof, n_u_dof))
    Kup = np.zeros((n_u_dof, n_p_dof))
    Kpp = np.zeros((n_p_dof, n_p_dof))
    checked = not verify_bilinearity
    for conn in connectivity:
        elem_coords = mesh.nodes[conn]
        Kuu_e, Kup_e, Kpp_e = tet10_p1_mixed_element_blocks(elem_coords, D_dev, kappa)
        if not checked:
            _verify_mixed_blocks(elem_coords, D_dev, kappa, Kuu_e, Kup_e, Kpp_e)
            checked = True
        gu = np.array([3 * n + k for n in conn for k in range(3)])
        gp = np.array([p_dof_of_node[int(n)] for n in conn[:4]])
        Kuu[np.ix_(gu, gu)] += Kuu_e
        Kup[np.ix_(gu, gp)] += Kup_e
        Kpp[np.ix_(gp, gp)] += Kpp_e
    return layout, Kuu, Kup, Kpp, p_dof_of_node


def solve_mixed_static(layout, Kuu, Kup, Kpp, F_u, fixed_u_dofs):
    """Bordered (saddle-point) direct solve of

        [ Kuu   Kup ] [u_free]   [F_u[free]]
        [ Kup^T -Kpp] [p     ] = [0        ]

    on the FREE u-dofs (every p-dof is always "free" -- this
    formulation, unlike plain displacement elasticity, has no Dirichlet
    condition on pressure; the near-incompressible finite-kappa `Kpp`
    block is what keeps the system nonsingular even with pressure
    completely unconstrained -- see this module's own docstring).
    Dense `np.linalg.solve` (the block system here is symmetric
    INDEFINITE, not SPD, so `FESystem`'s own Cholesky-first dispatch
    does not apply) -- adequate for the small patch-test-scale meshes
    this first pass targets; a sparse/iterative saddle-point solver
    (e.g. MINRES with a block preconditioner) is a natural follow-on
    for mesh-scale use, not attempted here.

    Returns (u_full (n_u,), p (n_p,))."""
    n_u = layout.sizes["u"]
    n_p = layout.sizes["p"]
    free_u = np.array([d for d in range(n_u) if d not in set(fixed_u_dofs)])
    nf = len(free_u)

    A = np.zeros((nf + n_p, nf + n_p))
    A[:nf, :nf] = Kuu[np.ix_(free_u, free_u)]
    A[:nf, nf:] = Kup[free_u, :]
    A[nf:, :nf] = Kup[free_u, :].T
    A[nf:, nf:] = -Kpp
    rhs = np.concatenate([np.asarray(F_u)[free_u], np.zeros(n_p)])

    x = np.linalg.solve(A, rhs)
    u_full = np.zeros(n_u)
    u_full[free_u] = x[:nf]
    p = x[nf:]
    return u_full, p
