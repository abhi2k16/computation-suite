# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_tet10_geometric_nonlinear.py -- validates Tet10SolidTL
(elements/nonlinear_solids.py), the geometric-only (Total Lagrangian,
linear elastic material) nonlinear counterpart to Tet10Solid3D, added
2026-08-30 for the wing-cantilever example in the sibling
Multi_Fidelity_NL_Structural_ROM project (see that class's own module-
level docstring in nonlinear_solids.py for the full derivation).

Three tiers, increasing in how much they actually exercise:

1. Exact zero-displacement identities -- no tolerance needed, catch any
   sign/index error in the F/E/S chain immediately.
2. Small-load convergence to the ALREADY-VALIDATED linear Tet10Solid3D
   solution on the SAME mesh -- confirms the nonlinear path reduces
   correctly to the trusted linear one, without a mesh-vs-beam-theory
   discrepancy confounding the comparison.
3. A large-deflection cantilever benchmark against a REAL, sourced
   reference solution -- SOFiSTiK's own "BE7: Large Deflection of
   Cantilever Beams I" verification document, itself citing Bisshopp &
   Drucker (1945)'s classical elastica solution. Fetched directly
   (not recalled from memory) rather than risk citing a misremembered
   historical table: L=10m, E=100 MPa, equivalent rectangular cross-
   section w=0.3m / t=0.10261m (I=2.701e-5 m^4, matching a D=0.2m/
   t=0.01m circular pipe's own I, used there for a beam-vs-shell
   comparison), P=269.35N total tip load, reference tip vertical
   deflection delta=8.10-8.11m (beam vs. quad-shell elements) -- a
   dramatically LARGE deflection (81% of L) precisely because
   elementary (linear) beam theory badly fails here: PL^3/(3EI) with
   these numbers gives ~33.2m, more than 3x the beam's own length, the
   exact "unphysical" failure mode the reference document itself
   states linear theory exhibits for large loads. Reproduced here with
   a 3D solid Tet10 mesh (a fundamentally different discretization from
   the reference's 1D beam/shell elements) -- NOT expected to match the
   reference to several significant figures (different element family
   entirely, and deliberately coarse here), but lands within 1% in
   practice (found directly, not assumed).

KNOWN ISSUE found while developing this file, unrelated to Tet10SolidTL
itself and NOT exercised by anything in this file (worked around by
avoiding it entirely, see below): UnifiedGeometryEngine.
generate_from_step() (fea_engine.geometry.gmsh_engine -- the same
pipeline wing_beam_fem.py's Tet10 mesh uses) becomes unreliable --
hangs/becomes pathologically slow at Gmsh's mesh-generation step --
once enough PRIOR generate_from_step() calls have accumulated anywhere
in the same process, not merely within one test file: confirmed
directly that running this file's tests together with other test
files that also call generate_from_step() repeatedly (test_step_
import.py, test_quadratic_extraction.py) reproduces the hang on a
call that works fine in isolation. Most likely OpenCASCADE-level state
not fully released across gmsh.finalize()/initialize() cycles -- a
more severe symptom than (and apparently separate from) that method's
own already-documented OCCTargetUnit stickiness. This file's meshes are
therefore built via a plain `.geo`-kernel helper
(_build_rect_cantilever_tet10_geo(), which never touches
gmsh.model.occ.importShapes()) instead, not because the STEP-import
path is wrong for THIS use case, but to keep this file's own results
reproducible regardless of what else has run earlier in the same
pytest session -- flagged here for whoever investigates the actual fix
in gmsh_engine.py itself.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

# Tiers 2/3 below build their mesh fixture via the raw `gmsh` package
# directly (NOT fea_engine.geometry.gmsh_engine, which was removed from
# this package -- see build_mesh.py/docs/generalized_mesh_grading_
# roadmap.md for that removal). `pytest.importorskip("gmsh")` alone is
# not sufficient here: on a `pip install gmsh` with the system libGLU
# missing, `import gmsh` raises OSError, not ImportError, so
# importorskip doesn't catch it and collection hard-errors instead of
# skipping. Caught explicitly below so Tier 1 (below, gmsh-independent)
# still collects and runs, and Tiers 2/3 skip cleanly instead of
# erroring in this environment.
try:
    import gmsh
except (ImportError, OSError):
    gmsh = None

from fea_engine import FESystem, Material, D_solid3d, Tet10Solid3D, Tet10SolidTL, Mesh
from fea_engine.nonlinear_solver import solve_nonlinear_static


def _build_rect_cantilever_tet10_geo(L, w, t, mesh_size, n_layers=2):
    """Solid rectangular cantilever via Gmsh's plain `.geo` kernel
    (extrude + setOrder(2) + manual Tet10Solid3D.GMSH_NODE_ORDER
    reindex) -- deliberately NOT generate_from_step(), see this file's
    own module docstring ("KNOWN ISSUE") for why."""
    if gmsh is None:
        pytest.skip("gmsh not importable in this environment (see module docstring / "
                     "top-of-file note): needed only for this raw-gmsh mesh fixture, "
                     "not for Tier 1 (gmsh-independent) tests above.")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("small_cantilever_geo")
    geo = gmsh.model.geo
    p1 = geo.addPoint(0, 0, 0, mesh_size)
    p2 = geo.addPoint(0, w, 0, mesh_size)
    p3 = geo.addPoint(0, w, t, mesh_size)
    p4 = geo.addPoint(0, 0, t, mesh_size)
    l1, l2, l3, l4 = (geo.addLine(p1, p2), geo.addLine(p2, p3),
                      geo.addLine(p3, p4), geo.addLine(p4, p1))
    loop = geo.addCurveLoop([l1, l2, l3, l4])
    surf = geo.addPlaneSurface([loop])
    geo.synchronize()
    geo.extrude([(2, surf)], L, 0.0, 0.0, numElements=[n_layers], recombine=False)
    geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    id_map = {tag: idx for idx, tag in enumerate(node_tags)}
    nodes_all = coords.reshape(-1, 3).copy()
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    assert list(elem_types) == [11], f"expected only Tet10 (type 11), got {elem_types}"
    conn_all = np.array([id_map[t_] for t_ in elem_node_tags[0]], dtype=int).reshape(-1, 10)
    conn_all = conn_all[:, Tet10Solid3D.GMSH_NODE_ORDER]
    gmsh.finalize()

    used = np.unique(conn_all.ravel())
    remap = -np.ones(len(nodes_all), dtype=int)
    remap[used] = np.arange(len(used))
    return Mesh(nodes_all[used], remap[conn_all], dim=3)


# =====================================================================
# Tier 1: exact zero-displacement identities
# =====================================================================
class TestZeroDisplacementIdentities:
    """A single, hand-built, well-shaped reference Tet10 element (exact
    arithmetic midpoints, matching Tet10Solid3D's own node order
    directly -- no Gmsh reindexing needed for this isolated check)."""

    @pytest.fixture
    def elem_coords(self):
        corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
        mid = np.array([(corners[a] + corners[b]) / 2 for a, b in edges])
        return np.vstack([corners, mid])

    @pytest.fixture
    def mat(self):
        return Material(E=70.3e9, nu=0.33, rho=2830.0)

    def test_internal_force_is_exactly_zero_at_zero_displacement(self, elem_coords, mat):
        elem = Tet10SolidTL()
        f0 = elem.internal_force(elem_coords, np.zeros(30), mat)
        assert np.abs(f0).max() == 0.0

    def test_tangent_at_zero_displacement_matches_linear_stiffness(self, elem_coords, mat):
        """The St. Venant-Kirchhoff tangent must reduce EXACTLY to the
        linear elastic stiffness at u=0 (F=I, E=0) -- checked to the
        precision the finite-difference tangent (h=1e-6) allows, not
        exact machine precision, since tangent_stiffness() is FD by
        design (see Tet10SolidTL's own docstring)."""
        elem = Tet10SolidTL()
        K_nl0 = elem.tangent_stiffness(elem_coords, np.zeros(30), mat)
        D = D_solid3d(mat)
        K_lin = Tet10Solid3D().stiffness(elem_coords, D, 1.0)
        rel_err = np.abs(K_nl0 - K_lin).max() / np.abs(K_lin).max()
        assert rel_err < 1e-8, f"relative diff {rel_err:.3e} too large for FD truncation at h=1e-6"

    def test_stiffness_method_matches_tangent_at_zero(self, elem_coords, mat):
        elem = Tet10SolidTL()
        assert np.allclose(elem.stiffness(elem_coords, mat),
                            elem.tangent_stiffness(elem_coords, np.zeros(30), mat))


# =====================================================================
# Tier 2 + 3: full nonlinear static solves on a real mesh
# =====================================================================
@pytest.fixture(scope="module")
def small_cantilever_mesh():
    """A modest Tet10 mesh of a small cantilever -- shared by the
    small-load convergence check (Tier 2), which needs a small,
    self-consistent mesh, not the large SOFiSTiK-benchmark one. Uses
    the `.geo`-kernel builder, not generate_from_step() -- see that
    builder's own docstring for why (avoids a real repeated-STEP-
    import defect found while developing this file)."""
    return _build_rect_cantilever_tet10_geo(L=1.0, w=0.05, t=0.05, mesh_size=0.15, n_layers=2)


class TestSmallLoadConvergesToLinear:
    def test_small_load_matches_linear_static_solution(self, small_cantilever_mesh):
        """Solve the SAME mesh two ways: Tet10Solid3D linear static
        (FESystem.solve_static(), already validated elsewhere in this
        package) vs. Tet10SolidTL through solve_nonlinear_static() at a
        deliberately tiny load -- geometric nonlinearity should be
        negligible, so the two must agree closely. This isolates
        "does Tet10SolidTL reduce to the trusted linear element" from
        any 3D-solid-vs-beam-theory modeling discrepancy."""
        mesh = small_cantilever_mesh
        mat = Material(E=70.3e9, nu=0.33, rho=2830.0)
        D = D_solid3d(mat)
        root_nodes = mesh.nodes_on_plane(axis=0, value=0.0, tol=1e-6)
        tip_nodes = mesh.nodes_on_plane(axis=0, value=1.0, tol=1e-6)
        assert len(root_nodes) > 0 and len(tip_nodes) > 0

        # Linear reference
        lin_sys = FESystem(mesh, Tet10Solid3D(), sparse=False)
        lin_sys.assemble_stiffness(D)
        lin_sys.fix_dofs(root_nodes, [0, 1, 2])
        P_small = 1.0   # N -- small enough that geometric nonlinearity is negligible
        lin_sys.add_nodal_force(tip_nodes, 2, -P_small)
        U_lin = lin_sys.solve_static()
        tip_dof_z = 3 * tip_nodes + 2   # every tip-face node's z-DOF
        w_lin = U_lin[tip_dof_z].mean()   # averaged over the tip face, not one arbitrary node

        # Nonlinear, same mesh, same load, single step. tol relaxed from
        # solve_nonlinear_static()'s default 1e-8: Tet10SolidTL's tangent
        # is FINITE-DIFFERENCE (h=1e-6, by design -- see that class's own
        # docstring), which caps how tightly Newton can converge at a
        # residual floor set by the FD truncation itself, not true
        # machine precision -- observed directly (a 1e-8 tol run stalled
        # at |R|~1.5e-7, just above tolerance, not a real non-convergence).
        nl_sys = FESystem(mesh, Tet10SolidTL(), sparse=False)
        nl_sys.fix_dofs(root_nodes, [0, 1, 2])
        nl_sys.add_nodal_force(tip_nodes, 2, -P_small)
        _, U_hist = solve_nonlinear_static(nl_sys, mat, n_steps=1, tol=1e-5)
        w_nl = U_hist[-1][tip_dof_z].mean()

        rel_err = abs(w_nl - w_lin) / abs(w_lin)
        assert rel_err < 0.02, f"small-load nonlinear ({w_nl:.6e}) vs linear ({w_lin:.6e}) diverge by {rel_err:.2%}"


class TestLargeDeflectionBenchmark:
    """SOFiSTiK BE7 / Bisshopp-Drucker (1945): L=10m, E=100 MPa,
    equivalent rectangular cross-section w=0.3m/t=0.10261m
    (I=2.701e-5 m^4), P=269.35N -> reference tip deflection
    delta=8.10-8.11m. See this file's own module docstring for the
    source and why linear theory (~33.2m) is not a valid comparison."""

    def test_large_deflection_is_in_the_right_regime(self):
        L, w, t = 10.0, 0.3, 0.10261
        E, nu = 100e6, 0.3   # nu not given by the reference (bending response is
                              # insensitive to it); a plausible generic value
        P_full = 269.35
        mat = Material(E=E, nu=nu, rho=1.0)
        D = D_solid3d(mat)

        # `.geo`-kernel builder, NOT generate_from_step() -- see this
        # file's own module docstring ("KNOWN ISSUE") and
        # _build_rect_cantilever_tet10_geo()'s docstring for why:
        # found directly that generate_from_step() becomes unreliable
        # after enough PRIOR calls have accumulated anywhere in the
        # same process (not just within this file), so this benchmark
        # -- the one test in this file that most needs to be reliably
        # reproducible -- deliberately avoids that pipeline entirely.
        mesh = _build_rect_cantilever_tet10_geo(L, w, t, mesh_size=0.15, n_layers=14)
        root_nodes = mesh.nodes_on_plane(axis=0, value=0.0, tol=1e-6)
        tip_nodes = mesh.nodes_on_plane(axis=0, value=L, tol=1e-6)
        assert len(root_nodes) > 0 and len(tip_nodes) > 0
        n_dof = 3 * len(mesh.nodes)
        print(f"\n  mesh: {len(mesh.elements)} Tet10 elements, {n_dof} DOF")

        sys = FESystem(mesh, Tet10SolidTL(), sparse=False)
        sys.fix_dofs(root_nodes, [0, 1, 2])
        sys.add_nodal_force(tip_nodes, 2, -P_full)   # tip load in -z (depth) direction

        # n_steps=4, max_iter=80: each Newton iteration costs several
        # seconds here (assemble_tangent_stiffness's finite-difference
        # cost dominates), and convergence per step is somewhat erratic
        # (the FD tangent's own truncation noise) -- found directly
        # (not assumed) that Gmsh's unstructured tetrahedralization for
        # this exact geometry/mesh_size is not perfectly reproducible
        # run-to-run (a known property of unstructured meshers), so the
        # resulting element shapes -- and therefore how many Newton
        # iterations this erratic-convergence problem needs -- can vary
        # between runs. A gentler ramp (4 steps, not 2 big ones) and a
        # generous max_iter make this reliably convergent regardless of
        # that mesh-to-mesh variability, rather than tuned to whichever
        # mesh one particular run happened to produce.
        load_factors, U_hist = solve_nonlinear_static(sys, mat, n_steps=4, max_iter=80, tol=1e-5)

        tip_dof_x = 3 * tip_nodes + 0
        tip_dof_z = 3 * tip_nodes + 2
        w_tip = -U_hist[-1][tip_dof_z].mean()   # vertical deflection, sign-positive, averaged over tip face
        u_tip = U_hist[-1][tip_dof_x].mean()     # horizontal (axial) tip displacement (negative = shortening)

        delta_lin = P_full * L ** 3 / (3 * E * (w * t ** 3 / 12))
        print(f"  linear-theory delta (invalid here): {delta_lin:.2f} m (> L={L}m -- the reference "
              f"document's own stated failure mode for elementary beam theory at this load)")
        print(f"  FE (Tet10SolidTL) tip deflection: {w_tip:.3f} m, axial tip motion: {u_tip:.3f} m")
        print(f"  reference (SOFiSTiK BE7, beam/quad elements): delta=8.10-8.11 m")

        # Qualitative, robust checks first (not sensitive to mesh coarseness):
        assert w_tip < L, "large-deflection solution must stay physically bounded by the beam length"
        assert w_tip < 0.6 * delta_lin, "must show strong geometric stiffening vs. (invalid) linear theory"

        # Quantitative comparison to the sourced reference. A real 3D
        # solid Tet10 mesh (coarse, deliberately) vs. a specialized 1D
        # beam/shell reference wasn't expected to match tightly -- but
        # profiled directly during development, this configuration lands
        # at w_tip=8.046m vs. the reference's 8.10-8.11m, under 1% --
        # asserting <5% here (generous margin over the observed ~0.7%)
        # makes this a real regression guard, not just a printed number.
        rel_err = abs(w_tip - 8.105) / 8.105
        print(f"  relative error vs. reference: {rel_err:.1%}")
        assert rel_err < 0.05, f"tip deflection {w_tip:.3f}m vs. reference 8.10-8.11m: {rel_err:.1%} off"
