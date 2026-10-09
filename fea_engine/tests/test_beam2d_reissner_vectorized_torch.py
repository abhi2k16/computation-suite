"""
test_beam2d_reissner_vectorized_torch.py -- validates the Wave 17 item
147 torch addendum (docs/consolidated_future_roadmap.md,
beam2d_reissner_vectorized_torch.py): a side-by-side PyTorch, GPU-capable
backend for the batched Beam2DReissner internal_force()/
tangent_stiffness() path item 147 itself built in NumPy
(beam2d_reissner_vectorized.py).

torch is an OPTIONAL dependency -- see test_torch_sparse_solver.py's own
docstring for why this file gates on fea_engine.torch_sparse_solver.
_HAS_TORCH directly rather than a bare pytest.importorskip("torch")
(which does not skip gracefully for a torch wheel that installs but
fails to import, e.g. a CUDA-linked build with no CUDA runtime present).
These tests have NOT been executed end-to-end in this development
sandbox -- torch is not importable here at all (confirmed: plain
ModuleNotFoundError, no torch wheel present) -- so they are written to
run for real, and SHOULD be run at least once on a torch-equipped
machine (the user's own Windows/NVIDIA GTX 1050 install, per torch_
sparse_solver.py's own VALIDATED note).

Three checks, matching this item's own stated validation criteria:

(a) Element level, torch-vectorized vs. the existing per-element looped
    Beam2DReissner.internal_force()/tangent_stiffness(): the SAME 1e-12
    relative-error bar test_beam2d_reissner_vectorized.py's own CHECK
    (a) uses for the numpy-vectorized-vs-looped comparison, on the SAME
    style of random large-rotation states test_beam2d_reissner.py's own
    CHECK (a) and that file's CHECK (a) both use. NOTE ON TOLERANCE:
    torch's own reduction/einsum implementation may use a different
    internal summation order and/or fused-multiply-add strategy than
    NumPy's for the same einsum contractions -- this is a plausible
    source of MORE floating-point noise than the numpy-vectorized-vs-
    looped comparison sees (which is NumPy compared against NumPy,
    written in the same library with the same reduction conventions
    throughout). The 1e-12 bar is kept here because it is this item's
    own stated criterion and because every operation involved (a small,
    fixed 6-dof contraction, not a large reduction) is unlikely to
    accumulate meaningfully more roundoff than the numpy path already
    does -- but this has NOT been confirmed by actually running torch,
    and if the real run on the user's machine finds torch's error is
    reliably above 1e-12 but still small (e.g. 1e-10 to 1e-11), that
    would be a genuine, reportable finding about torch's reduction
    order, not evidence of a bug in this module, and the tolerance
    should be loosened with that reasoning documented, not silently.
(b) Torch-vectorized vs. NumPy-vectorized (beam2d_reissner_vectorized.py)
    directly, on the SAME batch of elements -- expected to agree even
    MORE tightly than either agrees with the looped baseline, since both
    are the identical batching strategy, just in two different tensor
    libraries, evaluated on the exact same floating-point inputs.
(c) device="cpu" vs. device="cuda" consistency -- the CUDA half is
    skipped cleanly (not run, not assumed to pass) when torch.cuda.
    is_available() is False, mirroring torch_sparse_solver.py's own
    "implemented and available on the user's GPU-enabled install but not
    yet separately exercised" honest framing for its own untested CUDA
    path.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from fea_engine import elements as elmod
from fea_engine.torch_sparse_solver import _HAS_TORCH
from fea_engine.beam2d_reissner_vectorized import (
    internal_force_batched, tangent_stiffness_batched,
)

pytestmark = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not usable in this environment (not installed, or installed "
           "but failing to import -- see torch_sparse_solver.py's docstring); "
           "these tests validate the torch-vectorized Beam2DReissner backend "
           "and have nothing to run without it.")

if _HAS_TORCH:
    import torch
    from fea_engine.beam2d_reissner_vectorized_torch import (
        internal_force_batched_torch, tangent_stiffness_batched_torch,
    )


def _random_states(n, seed=42):
    """Same random large-rotation state construction as test_beam2d_
    reissner_vectorized.py's own CHECK (a) -- kept in sync deliberately
    (not re-derived) so both files exercise the identical states."""
    rng = np.random.default_rng(seed)
    elem_coords_all = np.zeros((n, 2, 2))
    u_elem_all = np.zeros((n, 6))
    for i in range(n):
        ec = np.array([[0.0, 0.0], [1.5, 0.9]]) + 0.1 * rng.standard_normal((2, 2))
        u = rng.standard_normal(6)
        u[0:2] *= 0.05; u[3:5] *= 0.05
        u[2] = rng.uniform(-3.0, 3.0)
        u[5] = rng.uniform(-3.0, 3.0)
        elem_coords_all[i] = ec
        u_elem_all[i] = u
    return elem_coords_all, u_elem_all


def _mat():
    E, G, A, I, kappa_s = 210e9, 80e9, 1.0e-3, 1.0e-7, 1.0
    return (E, G, A, I, kappa_s)


def test_check_a_torch_vectorized_element_matches_looped_to_1e12():
    print("=" * 70)
    print("CHECK (a): internal_force_batched_torch()/")
    print("tangent_stiffness_batched_torch() vs. the existing per-element")
    print("looped Beam2DReissner calls, at random large-rotation states")
    print("=" * 70)
    beam = elmod.Beam2DReissner()
    mat = _mat()
    n = 12
    elem_coords_all, u_elem_all = _random_states(n)

    f_loop = np.zeros((n, 6))
    K_loop = np.zeros((n, 6, 6))
    for i in range(n):
        f_loop[i] = beam.internal_force(elem_coords_all[i], u_elem_all[i], mat)
        K_loop[i] = beam.tangent_stiffness(elem_coords_all[i], u_elem_all[i], mat)

    f_torch = internal_force_batched_torch(elem_coords_all, u_elem_all, mat, device="cpu")
    K_torch = tangent_stiffness_batched_torch(elem_coords_all, u_elem_all, mat, device="cpu")

    f_err = np.max(np.abs(f_torch - f_loop)) / max(np.max(np.abs(f_loop)), 1e-30)
    K_err = np.max(np.abs(K_torch - K_loop)) / max(np.max(np.abs(K_loop)), 1e-30)
    print(f"  internal_force max relative error (torch vs looped): {f_err:.3e}")
    print(f"  tangent_stiffness max relative error (torch vs looped): {K_err:.3e}")
    assert f_err < 1e-12, (
        "internal_force_batched_torch does not match the looped version to "
        "1e-12 -- see this test's own docstring note on torch reduction-order "
        "noise as a plausible (but unconfirmed) explanation before assuming "
        "a bug.")
    assert K_err < 1e-12, (
        "tangent_stiffness_batched_torch does not match the looped version "
        "to 1e-12 -- see this test's own docstring note on torch "
        "reduction-order noise.")
    print("  PASS")


def test_check_b_torch_vectorized_matches_numpy_vectorized_tightly():
    print("=" * 70)
    print("CHECK (b): internal_force_batched_torch()/")
    print("tangent_stiffness_batched_torch() vs. the numpy-vectorized")
    print("beam2d_reissner_vectorized.py path, same batch of elements --")
    print("expected to agree even more tightly than either agrees with the")
    print("looped baseline (same batching strategy, two tensor libraries)")
    print("=" * 70)
    mat = _mat()
    n = 12
    elem_coords_all, u_elem_all = _random_states(n)

    f_np = internal_force_batched(elem_coords_all, u_elem_all, mat)
    K_np = tangent_stiffness_batched(elem_coords_all, u_elem_all, mat)

    f_torch = internal_force_batched_torch(elem_coords_all, u_elem_all, mat, device="cpu")
    K_torch = tangent_stiffness_batched_torch(elem_coords_all, u_elem_all, mat, device="cpu")

    f_err = np.max(np.abs(f_torch - f_np)) / max(np.max(np.abs(f_np)), 1e-30)
    K_err = np.max(np.abs(K_torch - K_np)) / max(np.max(np.abs(K_np)), 1e-30)
    print(f"  internal_force max relative error (torch vs numpy): {f_err:.3e}")
    print(f"  tangent_stiffness max relative error (torch vs numpy): {K_err:.3e}")
    assert f_err < 1e-12, "torch-vectorized internal_force does not match numpy-vectorized to 1e-12"
    assert K_err < 1e-12, "torch-vectorized tangent_stiffness does not match numpy-vectorized to 1e-12"
    print("  PASS")


def test_check_c_device_cpu_vs_cuda_consistency_note():
    print("=" * 70)
    print("CHECK (c): device='cpu' vs device='cuda' consistency -- the")
    print("cuda half is skipped cleanly (not assumed to pass) unless a CUDA")
    print("device is actually available in this run, mirroring torch_")
    print("sparse_solver.py's own honest 'implemented but not yet")
    print("separately exercised' framing for its own untested CUDA path")
    print("=" * 70)
    mat = _mat()
    n = 6
    elem_coords_all, u_elem_all = _random_states(n, seed=7)

    f_cpu = internal_force_batched_torch(elem_coords_all, u_elem_all, mat, device="cpu")
    K_cpu = tangent_stiffness_batched_torch(elem_coords_all, u_elem_all, mat, device="cpu")
    assert np.all(np.isfinite(f_cpu)) and np.all(np.isfinite(K_cpu))

    if not torch.cuda.is_available():
        pytest.skip(
            "no CUDA device available in this environment -- device='cuda' "
            "is implemented (beam2d_reissner_vectorized_torch.py's "
            "internal_force_batched_torch()/tangent_stiffness_batched_"
            "torch()/assemble_*_vectorized_torch() all accept device='cuda' "
            "and thread it straight through to torch.as_tensor(...)) but not "
            "yet separately exercised -- see this module's own docstring, "
            "matching torch_sparse_solver.py's identical honest framing for "
            "its own not-yet-exercised CUDA path.")

    f_cuda = internal_force_batched_torch(elem_coords_all, u_elem_all, mat, device="cuda")
    K_cuda = tangent_stiffness_batched_torch(elem_coords_all, u_elem_all, mat, device="cuda")
    f_err = np.max(np.abs(f_cuda - f_cpu)) / max(np.max(np.abs(f_cpu)), 1e-30)
    K_err = np.max(np.abs(K_cuda - K_cpu)) / max(np.max(np.abs(K_cpu)), 1e-30)
    print(f"  internal_force max relative error (cuda vs cpu): {f_err:.3e}")
    print(f"  tangent_stiffness max relative error (cuda vs cpu): {K_err:.3e}")
    assert f_err < 1e-10, "device='cuda' internal_force disagrees with device='cpu'"
    assert K_err < 1e-10, "device='cuda' tangent_stiffness disagrees with device='cpu'"
    print("  PASS")
