"""
sparse_vs_dense_plot.py -- Phase 2 (general-purpose extensions
roadmap) worked example: visual verification companion to
sparse_vs_dense_cantilever.py.

Two panels: (1) log-log solve time vs n_dof for the dense and sparse
paths, making the crossover point (sparse has more fixed overhead, so
it's actually SLOWER on small problems, then pulls sharply ahead) a
visible feature of the curve rather than a sentence in a README;
(2) the measured speedup ratio vs n_dof on its own axis, so the
"grows with problem size" claim is a single, unmistakably increasing
line. Both panels are built from the SAME timing runs, so the numbers
in the two panels are internally consistent by construction, not two
separately-run experiments that happen to agree.
"""
__author__ = "Abhijeet"
import time
import numpy as np
import matplotlib.pyplot as plt

from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, mesh


def build_and_time(sparse, nx, ny, Lx=4.0, Ly=1.0, E=210e9, nu=0.3, rho=7800.0):
    mat = Material(E=E, nu=nu, rho=rho)
    D2 = D_plane_stress(mat)
    m = mesh.rectangle_mesh(Lx, Ly, nx, ny)
    sysobj = FESystem(m, Quad4PlaneStress(), thickness=1.0, sparse=sparse)
    sysobj.assemble_stiffness(D2)
    for n in m.nodes_on_line(0, 0.0):
        sysobj.fix_dofs([n], [0, 1])
    tip = np.where(np.abs(m.nodes[:, 0] - Lx) < 1e-9)[0]
    sysobj.F[2 * tip + 1] = -1000.0 / len(tip)

    t0 = time.perf_counter()
    sysobj.solve_static()
    t_solve = time.perf_counter() - t0
    return sysobj.n_dof, t_solve


def main():
    sizes = [(8, 2), (16, 4), (24, 6), (32, 8), (50, 12), (70, 17), (90, 22)]
    n_dofs, t_dense, t_sparse = [], [], []
    for nx, ny in sizes:
        n_dof_d, t_d = build_and_time(False, nx, ny)
        n_dof_s, t_s = build_and_time(True, nx, ny)
        assert n_dof_d == n_dof_s
        n_dofs.append(n_dof_d)
        t_dense.append(t_d)
        t_sparse.append(t_s)
        print(f"n_dof={n_dof_d:6d}  dense={t_d*1000:8.2f} ms  "
              f"sparse={t_s*1000:8.2f} ms  speedup={t_d/max(t_s,1e-9):6.2f}x")

    n_dofs = np.array(n_dofs)
    t_dense = np.array(t_dense)
    t_sparse = np.array(t_sparse)
    speedup = t_dense / t_sparse

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax1.loglog(n_dofs, t_dense * 1000, 'o-', color='tomato', label='dense (default)')
    ax1.loglog(n_dofs, t_sparse * 1000, 's-', color='steelblue', label='sparse=True')
    ax1.set_xlabel('n_dof')
    ax1.set_ylabel('static solve time (ms)')
    ax1.set_title('solve_static() cost: dense vs sparse')
    ax1.grid(True, which='both', alpha=0.4)
    ax1.legend()

    ax2.semilogx(n_dofs, speedup, 'D-', color='seagreen')
    ax2.axhline(1.0, color='k', linestyle='--', linewidth=1, label='break-even (1x)')
    crossover_idx = np.searchsorted(speedup, 1.0)
    if 0 < crossover_idx < len(n_dofs):
        ax2.axvline(n_dofs[crossover_idx - 1], color='0.6', linestyle=':', linewidth=1)
    ax2.set_xlabel('n_dof')
    ax2.set_ylabel('speedup (dense time / sparse time)')
    ax2.set_title('sparse speedup grows with problem size')
    ax2.grid(True, which='both', alpha=0.4)
    ax2.legend()

    fig.suptitle('FESystem(sparse=True) vs dense: measured, not assumed')
    fig.savefig('sparse_vs_dense_timing.png', dpi=150)
    print("\nSaved sparse_vs_dense_timing.png")


if __name__ == "__main__":
    main()
