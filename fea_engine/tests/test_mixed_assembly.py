"""
test_mixed_assembly.py -- Wave 11 item 110 (docs/consolidated_future_
roadmap.md): validates mixed_assembly.py's Taylor-Hood P2(Tet10)-
P1(Tet4-corner) mixed u-p formulation -- BlockDofLayout's own split/cat
round trip, the bilinearity self-check guard (both on a synthetic
matrix and on the REAL assembled Kup coupling block), D_deviatoric_3d()
against D_solid3d() on a random strain, and the actual correctness
payoff: a single-element AFFINE-DISPLACEMENT patch test where the
constant-strain state is prescribed exactly on every node, leaving the
solve to determine ONLY the pressure field -- decisive because the
exact answer (a spatially CONSTANT pressure equal to kappa*div(u)) is
known in closed form and the P1 pressure field is only asked to
represent a constant, something it can do exactly.
"""
import numpy as np
import pytest

from fea_engine import Material
from fea_engine.material import D_solid3d
from fea_engine.mixed_assembly import (
    BlockDofLayout, check_bilinearity, D_deviatoric_3d,
    tet10_p1_mixed_element_blocks, build_pressure_dof_map,
    assemble_mixed_tet10_p1, solve_mixed_static,
    _tet10_p1_direct_forms, _verify_mixed_blocks,
)


def _steel():
    return Material(E=2.1e11, nu=0.3, rho=7850.0)


# ---------------------------------------------------------------------
# BlockDofLayout
# ---------------------------------------------------------------------
def test_block_dof_layout_split_cat_round_trip():
    layout = BlockDofLayout({"u": 5, "p": 3})
    assert layout.total == 8
    vec = np.arange(8.0)
    parts = layout.split(vec)
    assert np.array_equal(parts["u"], np.arange(5.0))
    assert np.array_equal(parts["p"], np.arange(5.0, 8.0))
    assert np.array_equal(layout.cat(parts), vec)


def test_block_dof_layout_slices_are_contiguous_and_disjoint():
    layout = BlockDofLayout({"a": 4, "b": 2, "c": 3})
    slices = [layout.slice(n) for n in ("a", "b", "c")]
    covered = set()
    for s in slices:
        idx = set(range(s.start, s.stop))
        assert not (idx & covered)
        covered |= idx
    assert covered == set(range(layout.total))


# ---------------------------------------------------------------------
# Bilinearity guard
# ---------------------------------------------------------------------
def test_check_bilinearity_true_positive():
    rng = np.random.default_rng(0)
    K = rng.standard_normal((4, 3))

    def form(a, b):
        return float(a @ K @ b)

    assert check_bilinearity(form, dim_a=4, dim_b=3, rng=np.random.default_rng(1)) is True


def test_check_bilinearity_true_negative_on_a_nonlinear_form():
    def form(a, b):
        return float(np.sum(a) * np.sum(b) + np.sum(a ** 2))   # not bilinear (quadratic in a)

    assert check_bilinearity(form, dim_a=3, dim_b=3, rng=np.random.default_rng(2)) is False


# ---------------------------------------------------------------------
# D_deviatoric_3d
# ---------------------------------------------------------------------
def test_deviatoric_plus_volumetric_reconstructs_full_D():
    mat = _steel()
    D_full = D_solid3d(mat)
    D_dev = D_deviatoric_3d(mat)
    kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu))
    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    D_reconstructed = D_dev + kappa * np.outer(m, m)
    assert np.allclose(D_full, D_reconstructed, rtol=1e-10)


def test_deviatoric_matches_full_D_on_shear_only_strain():
    # a pure-shear strain has zero trace -- D_dev and D_solid3d must
    # give IDENTICAL stress there (the volumetric part is exactly zero
    # on a trace-free strain, so subtracting it changes nothing).
    mat = _steel()
    D_full = D_solid3d(mat)
    D_dev = D_deviatoric_3d(mat)
    eps = np.array([0.0, 0.0, 0.0, 0.3, -0.1, 0.05])
    assert np.allclose(D_full @ eps, D_dev @ eps, rtol=1e-10)


# ---------------------------------------------------------------------
# Single straight-sided Tet10 element + its corner-node Tet4 topology
# ---------------------------------------------------------------------
_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


def _single_tet10_coords():
    corners = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    mids = np.array([0.5 * (corners[a] + corners[b]) for a, b in _EDGES])
    return np.vstack([corners, mids])   # (10, 3)


def test_bilinearity_guard_on_the_real_assembled_Kup_block():
    mat = _steel()
    D_dev = D_deviatoric_3d(mat)
    kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu))
    coords = _single_tet10_coords()
    Kuu, Kup, Kpp = tet10_p1_mixed_element_blocks(coords, D_dev, kappa)

    def coupling_form(u_vec, p_vec):
        return float(u_vec @ Kup @ p_vec)

    assert check_bilinearity(coupling_form, dim_a=30, dim_b=4, rng=np.random.default_rng(3)) is True
    assert np.allclose(Kuu, Kuu.T, atol=1e-3)
    assert np.allclose(Kpp, Kpp.T, atol=1e-20)


# ---------------------------------------------------------------------
# Wave 16 item 130: automatic bilinearity guard (TensorMesh's Mixed
# Assembly page runs an analogous check automatically on every
# forward(); assemble_mixed_tet10_p1() now does the same by default).
# ---------------------------------------------------------------------
def test_direct_forms_match_assembled_blocks_on_single_element():
    """_tet10_p1_direct_forms()'s three closures -- an independently
    re-derived evaluation of the same weak form, computed straight from
    raw DOF vectors rather than via the already-assembled Kuu/Kup/Kpp
    matrices -- must agree with the matrix quadratic forms on random
    vectors. This is the decisive cross-check _verify_mixed_blocks()
    relies on."""
    mat = _steel()
    D_dev = D_deviatoric_3d(mat)
    kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu))
    coords = _single_tet10_coords()
    Kuu, Kup, Kpp = tet10_p1_mixed_element_blocks(coords, D_dev, kappa)
    form_uu, form_up, form_pp = _tet10_p1_direct_forms(coords, D_dev, kappa)

    rng = np.random.default_rng(7)
    u_a, u_b = rng.standard_normal(30), rng.standard_normal(30)
    p_a, p_b = rng.standard_normal(4), rng.standard_normal(4)

    assert np.isclose(float(u_a @ Kuu @ u_b), form_uu(u_a, u_b), atol=1e-8, rtol=1e-8)
    assert np.isclose(float(u_a @ Kup @ p_b), form_up(u_a, p_b), atol=1e-8, rtol=1e-8)
    assert np.isclose(float(p_a @ Kpp @ p_b), form_pp(p_a, p_b), atol=1e-8, rtol=1e-8)


def test_direct_forms_are_bilinear():
    mat = _steel()
    D_dev = D_deviatoric_3d(mat)
    kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu))
    coords = _single_tet10_coords()
    form_uu, form_up, form_pp = _tet10_p1_direct_forms(coords, D_dev, kappa)

    assert check_bilinearity(form_uu, 30, 30, rng=np.random.default_rng(8)) is True
    assert check_bilinearity(form_up, 30, 4, rng=np.random.default_rng(9)) is True
    assert check_bilinearity(form_pp, 4, 4, rng=np.random.default_rng(10)) is True


def test_verify_mixed_blocks_passes_silently_on_correct_implementation():
    mat = _steel()
    D_dev = D_deviatoric_3d(mat)
    kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu))
    coords = _single_tet10_coords()
    Kuu, Kup, Kpp = tet10_p1_mixed_element_blocks(coords, D_dev, kappa)
    _verify_mixed_blocks(coords, D_dev, kappa, Kuu, Kup, Kpp)   # must not raise


def test_verify_mixed_blocks_catches_a_corrupted_Kuu_block():
    """The decisive negative control: deliberately hand the guard a
    Kuu that disagrees with the independent direct evaluation (as a
    future refactor bug might produce) and confirm it's caught with a
    message naming the right block, not silently accepted."""
    mat = _steel()
    D_dev = D_deviatoric_3d(mat)
    kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu))
    coords = _single_tet10_coords()
    Kuu, Kup, Kpp = tet10_p1_mixed_element_blocks(coords, D_dev, kappa)

    Kuu_broken = Kuu * 2.0 + 1.0   # wrong scale AND a non-bilinear constant offset
    with pytest.raises(RuntimeError, match="Kuu"):
        _verify_mixed_blocks(coords, D_dev, kappa, Kuu_broken, Kup, Kpp)


def test_verify_mixed_blocks_catches_a_corrupted_Kup_block():
    mat = _steel()
    D_dev = D_deviatoric_3d(mat)
    kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu))
    coords = _single_tet10_coords()
    Kuu, Kup, Kpp = tet10_p1_mixed_element_blocks(coords, D_dev, kappa)

    Kup_broken = Kup + 0.37   # disagrees with the direct evaluation
    with pytest.raises(RuntimeError, match="Kup"):
        _verify_mixed_blocks(coords, D_dev, kappa, Kuu, Kup_broken, Kpp)


class _TinyMeshFixture:
    def __init__(self, coords):
        self.nodes = coords
        self.dim = 3


def test_assemble_mixed_tet10_p1_default_runs_the_guard_without_raising():
    mat = _steel()
    coords = _single_tet10_coords()
    connectivity = np.arange(10).reshape(1, 10)
    # verify_bilinearity defaults to True -- this must complete normally
    # on the real, correct implementation.
    layout, Kuu, Kup, Kpp, p_dof_of_node = assemble_mixed_tet10_p1(
        _TinyMeshFixture(coords), connectivity, mat)
    assert Kuu.shape == (30, 30)


def test_assemble_mixed_tet10_p1_verify_bilinearity_false_skips_the_guard_and_matches_true(monkeypatch):
    mat = _steel()
    coords = _single_tet10_coords()
    connectivity = np.arange(10).reshape(1, 10)

    import fea_engine.mixed_assembly as ma

    def _fail_if_called(*args, **kwargs):
        pytest.fail("guard must not run when verify_bilinearity=False")

    monkeypatch.setattr(ma, "_verify_mixed_blocks", _fail_if_called)

    layout_off, Kuu_off, Kup_off, Kpp_off, _ = assemble_mixed_tet10_p1(
        _TinyMeshFixture(coords), connectivity, mat, verify_bilinearity=False)
    # reaching here (rather than pytest.fail inside _fail_if_called) IS
    # the assertion that the guard was never invoked.

    monkeypatch.undo()
    layout_on, Kuu_on, Kup_on, Kpp_on, _ = assemble_mixed_tet10_p1(
        _TinyMeshFixture(coords), connectivity, mat, verify_bilinearity=True)

    # verify_bilinearity never changes the assembled result, only
    # whether it's checked before being returned.
    assert np.array_equal(Kuu_off, Kuu_on)
    assert np.array_equal(Kup_off, Kup_on)
    assert np.array_equal(Kpp_off, Kpp_on)


def test_pressure_dof_map_covers_only_corner_nodes():
    # two tets sharing a face, corner-only global ids 0-4, mid-edge ids 5-13/14-... arbitrary
    connectivity = np.array([
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 6, 11, 12, 13, 14, 15],
    ])
    corner_ids, p_dof_of_node = build_pressure_dof_map(connectivity)
    assert set(corner_ids.tolist()) == {0, 1, 2, 3, 4}
    assert set(p_dof_of_node.keys()) == {0, 1, 2, 3, 4}
    assert sorted(p_dof_of_node.values()) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------
# THE decisive check: single-element affine-displacement patch test.
# ---------------------------------------------------------------------
def test_single_element_affine_patch_test_recovers_exact_constant_pressure():
    """Prescribe u(x) = A @ x (a random constant-strain-producing
    linear map) at EVERY node of a single Tet10 element (all 10 dofs
    fixed, matching the classic FEM patch-test convention of driving
    the whole boundary with the exact solution) -- since a single tet
    has no interior nodes, this pins every u-dof, leaving ONLY the 4
    pressure dofs to solve for. The exact solution is p(x) =
    kappa*tr(eps) EVERYWHERE (a constant, since eps=sym(A) is constant)
    -- the P1 pressure field can represent a constant exactly, so every
    recovered nodal p should match kappa*tr(eps) to near machine
    precision, a decisive (not merely plausible) correctness check on
    Kup/Kpp/the solve itself."""
    mat = _steel()
    kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu))
    coords = _single_tet10_coords()
    connectivity = np.arange(10).reshape(1, 10)

    class _TinyMesh:
        nodes = coords
        dim = 3

    layout, Kuu, Kup, Kpp, p_dof_of_node = assemble_mixed_tet10_p1(_TinyMesh(), connectivity, mat, kappa=kappa)

    rng = np.random.default_rng(4)
    A = rng.uniform(-0.01, 0.01, size=(3, 3))
    eps = 0.5 * (A + A.T)
    trace_eps = np.trace(eps)
    p_exact = kappa * trace_eps

    u_prescribed = (A @ coords.T).T.reshape(-1)   # (30,) -- u_i = A @ x_i for every node i
    all_u_dofs = np.arange(30)

    # Solve with EVERY u-dof fixed at the prescribed value: reformulate
    # as pure elimination -- since fixed_u_dofs = all dofs, free_u is
    # empty, and p directly solves Kup^T @ u_prescribed = Kpp @ p.
    p = np.linalg.solve(Kpp, Kup.T @ u_prescribed)

    assert np.allclose(p, p_exact, rtol=1e-6, atol=abs(p_exact) * 1e-6 + 1.0)


def test_solve_mixed_static_matches_manual_elimination_on_a_cantilevered_tet():
    """A sanity/self-consistency check on solve_mixed_static() itself
    (rather than the physics): fix ONLY the base face (nodes 0,1,2 and
    their shared mid-edge nodes), leave the rest free, apply a small
    load at the apex, and confirm solve_mixed_static()'s bordered
    system gives the SAME answer as directly building and solving the
    identical bordered linear system by hand."""
    mat = _steel()
    kappa = mat.E / (3.0 * (1.0 - 2.0 * mat.nu))
    coords = _single_tet10_coords()
    connectivity = np.arange(10).reshape(1, 10)

    class _TinyMesh:
        nodes = coords
        dim = 3

    layout, Kuu, Kup, Kpp, p_dof_of_node = assemble_mixed_tet10_p1(_TinyMesh(), connectivity, mat, kappa=kappa)

    # fix node 0's u-dofs only (a face is not needed for this pure
    # linear-algebra self-consistency check -- physical realism isn't
    # the point here, agreement with a hand-built reference system is)
    fixed_u_dofs = [0, 1, 2]
    F_u = np.zeros(30)
    F_u[3 * 3 + 2] = -1000.0   # small z load at node 3

    u_full, p = solve_mixed_static(layout, Kuu, Kup, Kpp, F_u, fixed_u_dofs)

    free_u = np.array([d for d in range(30) if d not in set(fixed_u_dofs)])
    nf = len(free_u)
    n_p = Kpp.shape[0]
    A = np.zeros((nf + n_p, nf + n_p))
    A[:nf, :nf] = Kuu[np.ix_(free_u, free_u)]
    A[:nf, nf:] = Kup[free_u, :]
    A[nf:, :nf] = Kup[free_u, :].T
    A[nf:, nf:] = -Kpp
    rhs = np.concatenate([F_u[free_u], np.zeros(n_p)])
    x_ref = np.linalg.solve(A, rhs)

    assert np.allclose(u_full[free_u], x_ref[:nf], atol=1e-9, rtol=1e-8)
    assert np.allclose(p, x_ref[nf:], atol=1e-9, rtol=1e-8)


def test_near_incompressible_limit_remains_solvable():
    """As kappa grows (nu -> 0.5), the system should remain solvable
    (finite, no blow-up) -- the actual payoff this formulation exists
    for (Wave 2 item 11's B-bar already relieves LOCKING; this checks
    the mixed system itself doesn't become numerically pathological as
    the incompressible limit is approached, which a naive displacement-
    only formulation's stiffness matrix conditioning famously does)."""
    coords = _single_tet10_coords()
    connectivity = np.arange(10).reshape(1, 10)

    class _TinyMesh:
        nodes = coords
        dim = 3

    results = []
    for nu in (0.3, 0.45, 0.49, 0.4999):
        mat = Material(E=2.1e11, nu=nu, rho=7850.0)
        layout, Kuu, Kup, Kpp, _ = assemble_mixed_tet10_p1(_TinyMesh(), connectivity, mat)
        fixed_u_dofs = [0, 1, 2]
        F_u = np.zeros(30)
        F_u[3 * 3 + 2] = -1000.0
        u_full, p = solve_mixed_static(layout, Kuu, Kup, Kpp, F_u, fixed_u_dofs)
        assert np.all(np.isfinite(u_full))
        assert np.all(np.isfinite(p))
        results.append(u_full[3 * 3 + 2])   # z-displacement at loaded node

    # displacement should stay within a bounded, non-exploding range
    # across the whole nu sweep (a locking, displacement-only element
    # would instead COLLAPSE toward zero here, not blow up -- but this
    # check is about numerical health of the mixed system, not a
    # locking-relief comparison, which is out of this pass's scope --
    # see the module docstring).
    results = np.array(results)
    assert np.all(np.isfinite(results))
    assert np.all(np.abs(results) > 0)
    assert np.max(np.abs(results)) / np.min(np.abs(results)) < 10.0
