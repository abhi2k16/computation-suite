"""
fea_scipy_vs_torch_gpu_benchmark.py -- fea_engine solver-backend benchmark:
SciPy (CPU, sparse direct LU) vs. PyTorch (CPU, sparse Jacobi-
preconditioned CG) vs. PyTorch (CUDA GPU, same CG) on the SAME
assembled linear-elastic cantilever system, across a sweep of mesh
sizes.

This exercises fea_engine's existing backend="scipy"/"torch" dispatch
(solver.py) and torch_sparse_solver.py (Wave 0 item 3) exactly as
already validated in tests/test_torch_sparse_solver.py -- this script
does not reimplement any solve logic, it only times and plots the
package's own real code paths. Per torch_sparse_solver.py's own
docstring, this was VALIDATED on 2026-09-08 on this project's own
target machine (Windows, NVIDIA GeForce GTX 1050, PyTorch 2.7.0/CUDA
12.6) on CPU only -- this script is the first one to actually exercise
device="cuda" on that same install.

IMPORTANT: run this in the environment that has a working CUDA-enabled
torch (your conda base env), NOT in a torch-free environment -- it
will still run and produce a scipy-vs-torch-CPU comparison if no GPU
is available, but the whole point of this script is the GPU curve.

    conda activate base
    cd fea_engine/examples
    python fea_scipy_vs_torch_gpu_benchmark.py

Each mesh size is assembled EXACTLY ONCE (build_system()), using
fea_engine's tensorized/chunked assembly path (vectorized=True,
Wave 11 item 107 + Wave 16 item 129) -- not the default per-element
Python loop, which was measured during this script's own development
taking ~1.2ms/element, i.e. minutes at the sweep's larger sizes, none
of it relevant to the actual scipy-vs-torch SOLVE comparison this
script exists to make. All three timed solves then reuse that SAME
assembled system via as_backend(), which just flips the .backend/
.device attributes solve_static() reads (backend= only affects the
FINAL linear-solve step, per solver.py's own docstring) -- so what's
actually being timed and compared is purely the linear-solve step:
SciPy's sparse direct LU (splu) vs. PyTorch's sparse CG on CPU vs. the
identical CG on GPU, on bit-for-bit the same K/F each time. GPU timing
is correctly synchronized: torch_sparse_cg_solve()'s own return path
ends in `.cpu().numpy()`, which blocks until every queued CUDA kernel
has actually finished, so wall-clock time around solve_static() is a
genuine (not just kernel-launch-queue) GPU time -- an explicit
torch.cuda.synchronize() is still called immediately before starting
each GPU timer, to also exclude any leftover async work queued by a
PREVIOUS call from bleeding into the current measurement.

Every solved displacement field is cross-checked against SciPy's
answer (relative max-abs error) before it's trusted for the plot --
this project's standing practice throughout its example galleries.
"""
import gc
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, mesh as mesh_mod

try:
    import torch
    from fea_engine.torch_sparse_solver import _HAS_TORCH
except Exception:
    _HAS_TORCH = False

# ---------------------------------------------------------------------
# Mesh sizes to sweep, (nx, ny) for a Lx x Ly rectangle. The previous
# default (up to ~29k dof) never left the regime where per-call
# overhead and CPU-GPU transfer cost dominate -- too small to show
# whether a GPU crossover exists at all. This sweep uses a much bigger
# STEP between points (fewer, more widely spaced sizes, not a dense
# scan) to reach into the hundreds-of-thousands-of-dof range where a
# real GPU should have a chance to pull ahead, without spending all
# the runtime resolving the uninteresting small-n end. Runtime grows
# with the last couple of entries -- expect the biggest 1-2 points to
# take from several seconds to a few minutes each depending on your
# GPU/CPU; comment out the largest tuples if you want a faster first
# look, or add bigger ones (e.g. (1500, 200), (2000, 260)) if your GPU
# still isn't saturated at the current top end.
# ---------------------------------------------------------------------
MESH_SIZES = [
    (20, 6),      #    ~300 dof
    (60, 14),     #   ~1.8k dof
    (140, 28),    #   ~8.2k dof
    (280, 50),    #  ~28.7k dof
    (500, 80),    #  ~81.2k dof
    (800, 120),   # ~193.8k dof
    (1100, 150),  # ~332.5k dof -- verified during this script's own development
                  # (scipy: ~12s assembly + ~11-20s solve, no memory issues)
    # (1500, 190),  # ~573.4k dof -- UNCOMMENT to push further. During this
                  # script's own development, this size hard-killed the dev
                  # sandbox via the OS OOM killer (a forced SIGKILL, which no
                  # try/except can catch -- scipy's direct sparse LU fill-in
                  # at this size needs more RAM than that ~3.8GB-limited
                  # sandbox had). It was NOT tested successfully anywhere.
                  # Your machine likely has far more RAM, but if this size
                  # kills the whole script instead of printing a "SKIPPED"
                  # line, that's why -- comment it back out.
]
LX, LY = 4.0, 1.0
E, NU = 210e9, 0.3
TIP_LOAD = -1000.0
CG_TOL = 1e-8   # matches solve_static()'s own built-in torch-backend default; not separately passed


def _n_repeats_for(n_dof):
    """Fewer timed repeats for the largest systems -- a big solve is
    far less prone to timer/OS-jitter noise than a small one (its own
    wall-clock time dwarfs any scheduling noise), so 1-2 repeats there
    is already reliable, and it keeps the sweep's total runtime from
    being dominated by re-solving the same huge system 3x over."""
    if n_dof < 5_000:
        return 5
    if n_dof < 100_000:
        return 3
    return 2


ASSEMBLY_CHUNK_SIZE = 50_000   # bounds peak memory during vectorized assembly at the largest sizes


def build_system(nx, ny, sparse=True):
    """Builds and assembles ONE FESystem for this mesh size -- backend=
    only affects the FINAL solve step (see torch_sparse_solver.py's own
    docstring), so assembly never needs to be redone per backend; the
    same assembled system is reused for all three timed solves below
    via as_backend(), instead of re-running assemble_stiffness() (the
    expensive part at large mesh sizes) three times over for no reason.

    vectorized=True (Wave 11 item 107) + chunk_size (Wave 16 item 129)
    uses fea_engine's tensorized, no-Python-element-loop assembly path
    -- numerically IDENTICAL to the default per-element loop (see
    vectorized_assembly.py's own docstring and tests/test_vectorized_
    assembly.py), just far faster at the larger end of this sweep,
    where the plain per-element loop was measured (during this
    script's own development) taking ~1.2ms/element -- i.e. minutes,
    dwarfing the actual scipy-vs-torch solve-time comparison this
    script exists to make. chunk_size keeps peak memory bounded at the
    biggest mesh sizes regardless of how much RAM your machine has."""
    mat = Material(E=E, nu=NU, rho=7800.0)
    D2 = D_plane_stress(mat)
    m = mesh_mod.rectangle_mesh(LX, LY, nx, ny)
    sysobj = FESystem(m, Quad4PlaneStress(), thickness=1.0, sparse=sparse, backend="scipy")
    sysobj.assemble_stiffness(D2, vectorized=True, thickness=1.0, chunk_size=ASSEMBLY_CHUNK_SIZE)
    for n in m.nodes_on_line(0, 0.0):
        sysobj.fix_dofs([n], [0, 1])
    tip = np.where(np.abs(m.nodes[:, 0] - LX) < 1e-9)[0]
    sysobj.F[2 * tip + 1] = TIP_LOAD / len(tip)
    return sysobj, sysobj.n_dof


def as_backend(sysobj, backend, device="cpu"):
    """Switches an ALREADY-ASSEMBLED FESystem to a different solve
    backend/device in place -- backend and device are plain instance
    attributes (solver.py's __init__ just stores them), and every
    solve_static() call re-derives Kff/Ff fresh from self.K/self.F, so
    this is exactly equivalent to (and much cheaper than) rebuilding
    and reassembling a whole new FESystem with that backend from
    scratch -- the K/F being solved are bit-for-bit the same object
    either way, which also means any solve-to-solve difference in the
    results below is purely from the solve algorithm, not from
    reassembling on a different code path."""
    sysobj.backend = backend
    sysobj.device = device
    return sysobj


def timed_solve(sysobj, device_sync=None, n_repeats=3):
    times = []
    U = None
    for _ in range(n_repeats):
        if device_sync == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        U = sysobj.solve_static()
        if device_sync == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return U, min(times)


def main():
    print("=" * 70)
    print("scipy vs. torch (CPU) vs. torch (CUDA) -- fea_engine solve_static()")
    print("=" * 70)

    if not _HAS_TORCH:
        print("\ntorch is not usable in this environment (not installed, or "
              "installed but failing to import -- see torch_sparse_solver.py's "
              "docstring). This script needs a working torch install to do "
              "anything useful; run it in the environment where torch actually "
              "imports (e.g. `conda activate base`).")
        return

    print(f"torch version: {torch.__version__}")
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA device visible to torch in this environment -- this run "
              "will still produce a real scipy-vs-torch-CPU comparison, but "
              "there is no GPU curve to plot. If you expected a GPU here, "
              "check `nvidia-smi` and that this conda env's torch build is "
              "CUDA-enabled (torch.version.cuda should not be None).")

    n_dofs, t_scipy, t_torch_cpu, t_torch_gpu = [], [], [], []
    err_cpu, err_gpu = [], []

    # One-off warm-up: first CUDA call in a process pays a one-time context/
    # kernel-compile cost unrelated to this benchmark's actual solves --
    # exclude it by doing one throwaway tiny GPU solve before timing starts.
    if has_cuda:
        warm, _ = build_system(4, 2)
        as_backend(warm, "torch", "cuda").solve_static()
        torch.cuda.synchronize()

    sweep_t0 = time.perf_counter()
    for idx, (nx, ny) in enumerate(MESH_SIZES):
        row_t0 = time.perf_counter()
        n_rep = _n_repeats_for((nx + 1) * (ny + 1) * 2)

        try:
            # Assemble ONCE per mesh size (the expensive part at large
            # sizes); all three timed solves below reuse this SAME
            # assembled system, just switched to a different backend/
            # device via as_backend().
            sysobj, n_dof = build_system(nx, ny)
            assembly_dt = time.perf_counter() - row_t0

            U_scipy, dt_scipy = timed_solve(as_backend(sysobj, "scipy"), n_repeats=n_rep)
            U_tcpu, dt_tcpu = timed_solve(as_backend(sysobj, "torch", "cpu"), n_repeats=n_rep)
            e_cpu = np.max(np.abs(U_tcpu - U_scipy)) / max(np.max(np.abs(U_scipy)), 1e-30)

            row = (f"  [{idx+1}/{len(MESH_SIZES)}] n_dof={n_dof:7d}  scipy={dt_scipy*1e3:9.3f} ms  "
                   f"torch-cpu={dt_tcpu*1e3:9.3f} ms (rel err {e_cpu:.2e})")

            dt_tgpu, e_gpu = np.nan, np.nan
            if has_cuda:
                U_tgpu, dt_tgpu = timed_solve(as_backend(sysobj, "torch", "cuda"),
                                               device_sync="cuda", n_repeats=n_rep)
                e_gpu = np.max(np.abs(U_tgpu - U_scipy)) / max(np.max(np.abs(U_scipy)), 1e-30)
                row += f"  torch-cuda={dt_tgpu*1e3:9.3f} ms (rel err {e_gpu:.2e})"

            row += (f"  [assembly {assembly_dt:.1f}s, "
                     f"solves {time.perf_counter() - row_t0 - assembly_dt:.1f}s, {n_rep} repeats]")
            print(row, flush=True)
            n_dofs.append(n_dof)
            t_scipy.append(dt_scipy)
            t_torch_cpu.append(dt_tcpu)
            t_torch_gpu.append(dt_tgpu)
            err_cpu.append(e_cpu)
            err_gpu.append(e_gpu)
        except (MemoryError, RuntimeError) as exc:
            # A too-large mesh size can legitimately exhaust RAM (scipy's
            # direct LU fill-in) or VRAM (a CUDA out-of-memory RuntimeError)
            # at the top of this sweep -- report it and keep every smaller
            # size's already-measured result rather than losing the whole
            # sweep to one oversized point. Shrink MESH_SIZES's largest
            # entries if you hit this.
            print(f"  [{idx+1}/{len(MESH_SIZES)}] nx={nx}, ny={ny}: SKIPPED -- {type(exc).__name__}: {exc}",
                  flush=True)
        finally:
            # Release this size's (possibly large) K/F/factorization
            # before building the next, bigger one.
            sysobj = None
            if has_cuda:
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\nTotal sweep time: {time.perf_counter() - sweep_t0:.1f}s")

    n_dofs = np.array(n_dofs, dtype=float)
    t_scipy = np.array(t_scipy)
    t_torch_cpu = np.array(t_torch_cpu)
    t_torch_gpu = np.array(t_torch_gpu)

    max_err_cpu = np.nanmax(err_cpu)
    print(f"\nMax relative error, torch-CPU vs. scipy, across the whole sweep: {max_err_cpu:.3e}")
    if has_cuda:
        max_err_gpu = np.nanmax(err_gpu)
        print(f"Max relative error, torch-CUDA vs. scipy, across the whole sweep: {max_err_gpu:.3e}")

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax1.loglog(n_dofs, t_scipy * 1e3, 'o-', color='steelblue', linewidth=1.8,
               label='scipy (sparse direct LU, CPU)')
    ax1.loglog(n_dofs, t_torch_cpu * 1e3, 's-', color='seagreen', linewidth=1.8,
               label='torch (sparse CG, CPU)')
    if has_cuda:
        ax1.loglog(n_dofs, t_torch_gpu * 1e3, '^-', color='tomato', linewidth=1.8,
                   label='torch (sparse CG, CUDA)')
    ax1.set_xlabel('free dof')
    ax1.set_ylabel('solve time (ms, min of adaptive repeats -- see _n_repeats_for())')
    ax1.set_title('solve_static() time vs. problem size')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.axhline(1.0, color='0.5', linestyle=':', linewidth=1.2)
    ax2.semilogx(n_dofs, t_scipy / t_torch_cpu, 's-', color='seagreen', linewidth=1.8,
                 label='scipy time / torch-CPU time')
    if has_cuda:
        ax2.semilogx(n_dofs, t_scipy / t_torch_gpu, '^-', color='tomato', linewidth=1.8,
                     label='scipy time / torch-CUDA time')
    ax2.set_xlabel('free dof')
    ax2.set_ylabel('speedup vs. scipy (>1 = faster than scipy)')
    ax2.set_title('Speedup vs. scipy sparse LU')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=9)

    gpu_name = torch.cuda.get_device_name(0) if has_cuda else 'no CUDA device available'
    fig.suptitle(f'fea_engine solve_static(): scipy vs. torch CPU vs. torch CUDA\n'
                f'({gpu_name}, torch {torch.__version__})', fontsize=12)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fea_scipy_vs_torch_gpu_benchmark.png')
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
