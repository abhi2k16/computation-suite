"""
test_sparse_assembly.py -- validates FESystem(..., sparse=True) (Phase
2, docs/general_purpose_extensions_roadmap.md Section 7): a storage-
format change (scipy.sparse.lil_matrix -> CSR at solve time, instead
of a dense numpy array), opt-in via sparse=True, additive to the
existing dense default.

The validation strategy is the strongest kind available here: not a
new physics claim, just "does the storage format change the answer" --
solve the SAME model both ways and require near-machine-precision
agreement (CHECK 1, 2). CHECK 3 is the actual justification for the
added complexity: measured (not assumed) that sparse pulls ahead of
dense as problem size grows, on a large-enough model that dense
assembly/storage genuinely becomes the bottleneck the roadmap doc
identifies it as.
"""
import time
import numpy as np
from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, mesh


def _build_system(sparse, nx, ny, Lx=4.0, Ly=1.0, E=210e9, nu=0.3, rho=7800.0):
    mat = Material(E=E, nu=nu, rho=rho)
    D2 = D_plane_stress(mat)
    m = mesh.rectangle_mesh(Lx, Ly, nx, ny)
    sysobj = FESystem(m, Quad4PlaneStress(), thickness=1.0, sparse=sparse)
    sysobj.assemble_stiffness(D2)
    sysobj.assemble_mass(rho * np.eye(2))
    for n in m.nodes_on_line(0, 0.0):
        sysobj.fix_dofs([n], [0, 1])
    tip = np.where(np.abs(m.nodes[:, 0] - Lx) < 1e-9)[0]
    sysobj.F[2 * tip + 1] = -1000.0 / len(tip)
    return sysobj, m


def test_sparse_static_matches_dense():
    print("=" * 70)
    print("CHECK 1: sparse=True solve_static() matches sparse=False exactly")
    print("=" * 70)
    sys_d, _ = _build_system(sparse=False, nx=8, ny=2)
    sys_s, _ = _build_system(sparse=True, nx=8, ny=2)
    Ud = sys_d.solve_static()
    Us = sys_s.solve_static()
    err = np.max(np.abs(Ud - Us))
    rel = err / max(np.max(np.abs(Ud)), 1e-30)
    print(f"  max abs diff = {err:.3e}, max rel diff = {rel:.3e}")
    assert rel < 1e-8
    print("  PASS")


def test_sparse_modal_matches_dense():
    print()
    print("=" * 70)
    print("CHECK 2: sparse=True solve_modal() matches sparse=False (freq + mode shapes)")
    print("=" * 70)
    sys_d, _ = _build_system(sparse=False, nx=8, ny=2)
    sys_s, _ = _build_system(sparse=True, nx=8, ny=2)
    freq_d, modes_d = sys_d.solve_modal(n_modes=4)
    freq_s, modes_s = sys_s.solve_modal(n_modes=4)
    freq_rel = np.max(np.abs(freq_d - freq_s) / np.maximum(freq_d, 1e-30))
    print(f"  dense freqs (Hz):  {freq_d}")
    print(f"  sparse freqs (Hz): {freq_s}")
    print(f"  max relative frequency diff = {freq_rel:.3e}")
    assert freq_rel < 1e-6

    # mode shapes are only defined up to sign -- compare via |cosine similarity|
    for i in range(4):
        md, ms = modes_d[:, i], modes_s[:, i]
        cos_sim = abs(md @ ms) / (np.linalg.norm(md) * np.linalg.norm(ms))
        print(f"  mode {i}: |cosine similarity| = {cos_sim:.10f}")
        assert cos_sim > 1 - 1e-8
    print("  PASS -- frequencies and mode shapes agree to near machine precision")


def test_sparse_pulls_ahead_on_a_larger_model():
    print()
    print("=" * 70)
    print("CHECK 3: sparse measurably faster than dense on a larger model")
    print("=" * 70)
    # large enough that dense O(n_dof^2) storage / O(n_dof^3) solve cost
    # is real, small enough to still run quickly in a test
    nx, ny = 60, 15
    sys_d, m = _build_system(sparse=False, nx=nx, ny=ny)
    sys_s, _ = _build_system(sparse=True, nx=nx, ny=ny)
    print(f"  model: {sys_d.n_dof} dof ({nx}x{ny} elements)")

    t0 = time.perf_counter()
    Ud = sys_d.solve_static()
    t_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    Us = sys_s.solve_static()
    t_sparse = time.perf_counter() - t0

    rel = np.max(np.abs(Ud - Us)) / max(np.max(np.abs(Ud)), 1e-30)
    print(f"  dense solve_static():  {t_dense*1000:.2f} ms")
    print(f"  sparse solve_static(): {t_sparse*1000:.2f} ms")
    print(f"  speedup: {t_dense/max(t_sparse,1e-12):.1f}x, agreement rel diff = {rel:.3e}")
    assert rel < 1e-6
    assert t_sparse < t_dense, (
        "expected the sparse solve to be faster on this large a model -- if this "
        "ever fails, either the model isn't large enough anymore for this machine "
        "or something regressed in the sparse path")
    print("  PASS -- sparse solve is both correct AND measurably faster here, "
          "not just theoretically justified")
