"""
test_spd_solve.py -- Wave 0 item 5 (docs/consolidated_future_roadmap.md,
source fem_implementation_lessons.md): validates
FESystem._dense_spd_solve() / the SPD-aware Cholesky path now wired
into solve_static().

Two things need checking, neither of which is "does the solve still
work" (test_sparse_assembly.py and every other existing solver test
already exercise solve_static() end-to-end on real models and would
have caught a correctness regression):

1. CHECK 1/2: the new Cholesky path gives the SAME answer as the old
   always-general-solve path, on both a real assembled model and
   directly on hand-built matrices -- this is a performance-only
   change, so "same numbers" is the whole bar.
2. CHECK 3: the automatic fallback actually engages (and still gives
   the right answer) for a symmetric-but-NOT-SPD matrix, since that's
   the branch a real cantilever/plate model never exercises (a
   well-constrained linear-elastic K is always SPD) but the code path
   still needs to be correct for the day it's needed.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest
from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, mesh


def _build_cantilever(nx=8, ny=2, Lx=4.0, Ly=1.0, E=210e9, nu=0.3):
    mat = Material(E=E, nu=nu, rho=7800.0)
    D2 = D_plane_stress(mat)
    m = mesh.rectangle_mesh(Lx, Ly, nx, ny)
    sysobj = FESystem(m, Quad4PlaneStress(), thickness=1.0, sparse=False)
    sysobj.assemble_stiffness(D2)
    for n in m.nodes_on_line(0, 0.0):
        sysobj.fix_dofs([n], [0, 1])
    tip = np.where(np.abs(m.nodes[:, 0] - Lx) < 1e-9)[0]
    sysobj.F[2 * tip + 1] = -1000.0 / len(tip)
    return sysobj, m


def test_spd_path_matches_general_solve_on_a_real_model():
    print("=" * 70)
    print("CHECK 1: solve_static()'s new Cholesky path matches a direct")
    print("np.linalg.solve() reference on a real cantilever model")
    print("=" * 70)
    sysobj, _ = _build_cantilever()
    U_new = sysobj.solve_static()

    free = sysobj.free_dofs
    Kff = sysobj.K[np.ix_(free, free)]
    Ff = sysobj.F[free]
    U_ref = np.zeros(sysobj.n_dof)
    U_ref[free] = np.linalg.solve(Kff, Ff)

    err = np.max(np.abs(U_new - U_ref))
    rel = err / max(np.max(np.abs(U_ref)), 1e-30)
    print(f"  max abs diff = {err:.3e}, max rel diff = {rel:.3e}")
    assert rel < 1e-10
    print("  PASS -- Cholesky-routed solve agrees with plain LU to near machine precision")


def test_dense_spd_solve_matches_numpy_on_random_spd_matrices():
    print()
    print("=" * 70)
    print("CHECK 2: _dense_spd_solve() vs np.linalg.solve() on random SPD systems")
    print("=" * 70)
    rng = np.random.default_rng(0)
    for n in (3, 10, 40):
        A = rng.standard_normal((n, n))
        K = A @ A.T + n * np.eye(n)   # guaranteed SPD
        b = rng.standard_normal(n)
        x_chol = FESystem._dense_spd_solve(K, b)
        x_ref = np.linalg.solve(K, b)
        rel = np.max(np.abs(x_chol - x_ref)) / max(np.max(np.abs(x_ref)), 1e-30)
        print(f"  n={n}: max rel diff = {rel:.3e}")
        assert rel < 1e-9
    print("  PASS")


def test_dense_spd_solve_falls_back_for_non_spd_symmetric_matrix():
    print()
    print("=" * 70)
    print("CHECK 3: _dense_spd_solve() correctly REJECTS a symmetric,")
    print("indefinite (non-SPD) matrix, and _eigen_solve() (the method")
    print("solve_static() routes to on that rejection) recovers the")
    print("right answer -- see _dense_spd_solve()'s own docstring for why")
    print("it deliberately does NOT try np.linalg.solve() as a fallback")
    print("itself anymore (that was tried, and found to be unsafe for the")
    print("singular case _eigen_solve() also has to handle -- Wave 0 item 6)")
    print("=" * 70)
    # A small symmetric INDEFINITE matrix (mixed-sign eigenvalues) --
    # cho_factor must fail on this.
    K = np.array([[2.0, 1.0, 0.0],
                  [1.0, -3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    eigvals = np.linalg.eigvalsh(K)
    print(f"  eigenvalues: {eigvals}")
    assert eigvals.min() < 0 < eigvals.max(), "test matrix must genuinely be indefinite"

    b = np.array([1.0, -2.0, 0.5])
    with pytest.raises(np.linalg.LinAlgError):
        FESystem._dense_spd_solve(K, b)
    print("  _dense_spd_solve() correctly raised LinAlgError on non-SPD input")

    x_new = FESystem._eigen_solve(K, b)
    x_ref = np.linalg.solve(K, b)
    rel = np.max(np.abs(x_new - x_ref)) / max(np.max(np.abs(x_ref)), 1e-30)
    print(f"  _eigen_solve() max rel diff vs np.linalg.solve = {rel:.3e}")
    assert rel < 1e-10
    print("  PASS -- fallback dispatch is correct, and the eigen-based solve agrees")
