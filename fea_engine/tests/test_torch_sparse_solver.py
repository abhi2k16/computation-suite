"""
test_torch_sparse_solver.py -- Wave 0 item 3 (docs/consolidated_future_
roadmap.md, source tensormesh_comparative_analysis.md Section 6.1):
validates fea_engine.torch_sparse_solver's dense and sparse-CG PyTorch
solve paths against this package's own existing SciPy-based
FESystem.solve_static().

torch is an OPTIONAL dependency -- see test_autograd_tangent.py's own
docstring for why this file gates on
fea_engine.torch_sparse_solver._HAS_TORCH directly rather than a bare
`pytest.importorskip("torch")` (which does not skip gracefully for a
torch wheel that installs but fails to import, e.g. a CUDA-linked
build with no CUDA runtime present -- exactly the state torch_sparse_
solver.py's own docstring documents for this project's development
sandbox). These tests have not been executed end-to-end in that
sandbox for the same reason; they are written to run for real, and
SHOULD be run at least once, in any environment with a working torch
install.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest
from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, mesh
from fea_engine.torch_sparse_solver import _HAS_TORCH

pytestmark = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not usable in this environment (not installed, or installed "
           "but failing to import -- see torch_sparse_solver.py's docstring); "
           "these tests validate torch-based solve paths and have nothing to "
           "run without it.")

if _HAS_TORCH:
    from fea_engine.torch_sparse_solver import (
        torch_dense_solve, torch_sparse_cg_solve, fesystem_solve_static_torch,
    )


def _build_cantilever(sparse, nx=10, ny=3, Lx=4.0, Ly=1.0, E=210e9, nu=0.3):
    mat = Material(E=E, nu=nu, rho=7800.0)
    D2 = D_plane_stress(mat)
    m = mesh.rectangle_mesh(Lx, Ly, nx, ny)
    sysobj = FESystem(m, Quad4PlaneStress(), thickness=1.0, sparse=sparse)
    sysobj.assemble_stiffness(D2)
    for n in m.nodes_on_line(0, 0.0):
        sysobj.fix_dofs([n], [0, 1])
    tip = np.where(np.abs(m.nodes[:, 0] - Lx) < 1e-9)[0]
    sysobj.F[2 * tip + 1] = -1000.0 / len(tip)
    return sysobj, m


def test_torch_dense_solve_random_spd_system():
    print("=" * 70)
    print("CHECK 1: torch_dense_solve() vs np.linalg.solve() on a random")
    print("SPD system")
    print("=" * 70)
    rng = np.random.default_rng(0)
    n = 20
    A = rng.standard_normal((n, n))
    K = A @ A.T + n * np.eye(n)
    b = rng.standard_normal(n)

    x_ref = np.linalg.solve(K, b)
    x_torch = torch_dense_solve(K, b)
    rel = np.max(np.abs(x_torch - x_ref)) / max(np.max(np.abs(x_ref)), 1e-30)
    print(f"  max rel diff = {rel:.3e}")
    assert rel < 1e-8
    print("  PASS")


def test_torch_sparse_cg_solve_matches_reference_on_random_spd_system():
    print()
    print("=" * 70)
    print("CHECK 2: torch_sparse_cg_solve() vs np.linalg.solve() on a")
    print("random SPARSE SPD system, and it actually converges")
    print("=" * 70)
    import scipy.sparse as sp
    rng = np.random.default_rng(1)
    n = 200
    density = 0.05
    A = sp.random(n, n, density=density, random_state=rng, format="csr")
    K = (A + A.T).tocsr()
    K = K + sp.eye(n) * (abs(K).sum(axis=1).max() + 1.0)   # diagonally dominant -> SPD
    b = rng.standard_normal(n)

    x_ref = np.linalg.solve(K.toarray(), b)
    x_cg, n_iter, converged = torch_sparse_cg_solve(K, b, tol=1e-10)
    print(f"  converged={converged} in {n_iter} iterations")
    assert converged
    rel = np.max(np.abs(x_cg - x_ref)) / max(np.max(np.abs(x_ref)), 1e-30)
    print(f"  max rel diff vs reference = {rel:.3e}")
    assert rel < 1e-6
    print("  PASS")


def test_fesystem_solve_static_torch_dense_matches_scipy_dense():
    print()
    print("=" * 70)
    print("CHECK 3: fesystem_solve_static_torch(method='dense') matches")
    print("FESystem.solve_static() on a real cantilever model (dense K)")
    print("=" * 70)
    sysobj, _ = _build_cantilever(sparse=False)
    U_ref = sysobj.solve_static()
    U_torch = fesystem_solve_static_torch(sysobj, method="dense")
    rel = np.max(np.abs(U_torch - U_ref)) / max(np.max(np.abs(U_ref)), 1e-30)
    print(f"  max rel diff = {rel:.3e}")
    assert rel < 1e-8
    print("  PASS")


def test_fesystem_solve_static_torch_cg_matches_scipy_sparse():
    print()
    print("=" * 70)
    print("CHECK 4: fesystem_solve_static_torch(method='cg') matches")
    print("FESystem.solve_static() on a real cantilever model (sparse K,")
    print("genuinely SPD -- the well-constrained linear-elastic case)")
    print("=" * 70)
    sysobj, _ = _build_cantilever(sparse=True)
    U_ref = sysobj.solve_static()
    U_torch = fesystem_solve_static_torch(sysobj, method="cg", tol=1e-10)
    rel = np.max(np.abs(U_torch - U_ref)) / max(np.max(np.abs(U_ref)), 1e-30)
    print(f"  max rel diff = {rel:.3e}")
    assert rel < 1e-6
    print("  PASS -- SciPy direct solve and PyTorch sparse CG agree on the"
          " SAME assembled system, confirming the GPU-capable path is a"
          " correct drop-in alternative for solve_static()")


def test_torch_sparse_cg_solve_rejects_non_positive_diagonal():
    print()
    print("=" * 70)
    print("CHECK 5: torch_sparse_cg_solve() raises ValueError on a matrix")
    print("that CANNOT be SPD (non-positive diagonal entry), rather than")
    print("silently running CG on a system it isn't valid for")
    print("=" * 70)
    import scipy.sparse as sp
    K = sp.csr_matrix(np.array([[1.0, 0.5], [0.5, -1.0]]))
    b = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        torch_sparse_cg_solve(K, b)
    print("  PASS")


def test_fesystem_backend_torch_dense_matches_backend_scipy():
    print()
    print("=" * 70)
    print("CHECK 6: FESystem(..., backend='torch').solve_static() (dense)")
    print("matches FESystem(..., backend='scipy').solve_static() -- the")
    print("integrated single-point-of-choice API, not the standalone")
    print("fesystem_solve_static_torch() call CHECK 3 already validated")
    print("=" * 70)
    sys_scipy, _ = _build_cantilever(sparse=False)
    U_ref = sys_scipy.solve_static()

    sys_torch, m = _build_cantilever(sparse=False)
    # _build_cantilever() doesn't take a backend kwarg -- rebuild
    # identically with backend="torch" instead.
    from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, mesh as mesh_mod
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    D2 = D_plane_stress(mat)
    mm = mesh_mod.rectangle_mesh(4.0, 1.0, 10, 3)
    sys_torch = FESystem(mm, Quad4PlaneStress(), thickness=1.0, sparse=False, backend="torch")
    sys_torch.assemble_stiffness(D2)
    for n in mm.nodes_on_line(0, 0.0):
        sys_torch.fix_dofs([n], [0, 1])
    tip = np.where(np.abs(mm.nodes[:, 0] - 4.0) < 1e-9)[0]
    sys_torch.F[2 * tip + 1] = -1000.0 / len(tip)

    U_torch = sys_torch.solve_static()
    rel = np.max(np.abs(U_torch - U_ref)) / max(np.max(np.abs(U_ref)), 1e-30)
    print(f"  max rel diff = {rel:.3e}")
    assert rel < 1e-8
    print("  PASS -- backend='torch' is a correct drop-in for backend='scipy'"
          " (default) via the single solve_static() call, no separate"
          " fesystem_solve_static_torch() import needed")


def test_fesystem_backend_torch_sparse_cg_matches_backend_scipy():
    print()
    print("=" * 70)
    print("CHECK 7: FESystem(..., sparse=True, backend='torch')")
    print(".solve_static() (CG) matches backend='scipy' on the same model")
    print("=" * 70)
    sys_scipy, _ = _build_cantilever(sparse=True)
    U_ref = sys_scipy.solve_static()

    from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, mesh as mesh_mod
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    D2 = D_plane_stress(mat)
    mm = mesh_mod.rectangle_mesh(4.0, 1.0, 10, 3)
    sys_torch = FESystem(mm, Quad4PlaneStress(), thickness=1.0, sparse=True, backend="torch")
    sys_torch.assemble_stiffness(D2)
    for n in mm.nodes_on_line(0, 0.0):
        sys_torch.fix_dofs([n], [0, 1])
    tip = np.where(np.abs(mm.nodes[:, 0] - 4.0) < 1e-9)[0]
    sys_torch.F[2 * tip + 1] = -1000.0 / len(tip)

    U_torch = sys_torch.solve_static()
    rel = np.max(np.abs(U_torch - U_ref)) / max(np.max(np.abs(U_ref)), 1e-30)
    print(f"  max rel diff = {rel:.3e}")
    assert rel < 1e-6
    print("  PASS")
