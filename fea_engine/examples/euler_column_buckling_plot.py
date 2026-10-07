"""
euler_column_buckling_plot.py -- Phase 4 (general-purpose extensions
roadmap) worked example: visual verification companion to
euler_column_buckling_demo.py.

Two panels, following this project's established plotting convention
(see examples/hex8_convergence.py): (1) log-log mesh-refinement
convergence of the FEM critical load to the closed-form Euler value --
a straight line on log-log axes with a slope matching the expected
O(h^2) convergence rate is the "convincing" signature of a correctly
implemented consistent geometric-stiffness formulation, not just a
low final error; (2) the actual buckling mode shape (first eigenvector
of solve_linear_buckling()) overlaid on the undeformed column, which
should look exactly like the textbook half-sine-wave pin-pin buckling
shape -- a qualitative sanity check a table of numbers can't give you.
"""
import numpy as np
import matplotlib.pyplot as plt

from fea_engine import Beam2DEulerBernoulli, FESystem
from fea_engine.mesh import Mesh


def euler_column(n_elem, L, EI):
    n_nodes = n_elem + 1
    nodes = np.linspace(0, L, n_nodes).reshape(-1, 1)
    elements = np.array([[i, i + 1] for i in range(n_elem)])
    mesh = Mesh(nodes=nodes, elements=elements, dim=1)
    beam = Beam2DEulerBernoulli()
    fs = FESystem(mesh, beam, sparse=False)
    fs.assemble_stiffness(EI)
    fs.assemble_geometric_stiffness(-1.0)
    fs.fix_dofs([0], [0])
    fs.fix_dofs([n_nodes - 1], [0])
    return fs, nodes


def main():
    E, I, L = 210e9, 8e-6, 3.0
    EI = E * I
    P_exact = np.pi**2 * EI / L**2

    # ---- Panel 1 data: mesh-refinement convergence ----
    n_elem_list = np.array([2, 4, 8, 16, 32, 64])
    rel_errs = []
    for n_elem in n_elem_list:
        fs, _ = euler_column(int(n_elem), L, EI)
        load_factors, _ = fs.solve_linear_buckling(n_modes=1)
        rel_errs.append(abs(load_factors[0] - P_exact) / P_exact)
    rel_errs = np.array(rel_errs)

    # ---- Panel 2 data: buckling mode shape at a moderate mesh ----
    n_elem_shape = 20
    fs, nodes = euler_column(n_elem_shape, L, EI)
    load_factors, mode_shapes = fs.solve_linear_buckling(n_modes=1)
    v = mode_shapes[0::2, 0]              # transverse (v) dof at every node
    v = v / np.max(np.abs(v))             # normalize to unit amplitude
    x = nodes[:, 0]
    x_exact = np.linspace(0, L, 200)
    v_exact = np.sin(np.pi * x_exact / L)
    # match sign convention to the FEM mode (eigenvectors have arbitrary sign)
    if np.dot(v, np.sin(np.pi * x / L)) < 0:
        v = -v

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: log-log convergence
    ax1.loglog(n_elem_list, rel_errs, 'o-', color='blue', label='FEM (consistent $K_\\sigma$)')
    ref = rel_errs[0] * (n_elem_list[0] / n_elem_list)**2
    ax1.loglog(n_elem_list, ref, 'k--', linewidth=1, label=r'$O(n_{elem}^{-2})$ reference slope')
    ax1.set_xlabel('number of elements')
    ax1.set_ylabel(r'relative error in $P_{cr}$')
    ax1.set_title('Euler column buckling: mesh convergence')
    ax1.grid(True, which='both', alpha=0.4)
    ax1.legend()

    # Panel 2: buckling mode shape
    ax2.plot(x_exact, v_exact, '-', color='k', linewidth=1.5, label=r'exact: $\sin(\pi x/L)$')
    ax2.plot(x, v, 'o', color='crimson', markersize=5, label=f'FEM mode ({n_elem_shape} elements)')
    ax2.axhline(0.0, color='0.6', linewidth=0.8)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('normalized transverse deflection v')
    ax2.set_title('First buckling mode shape (pin-pin column)')
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    fig.suptitle(f'Euler column buckling verification  '
                 f'(exact $P_{{cr}}$ = {P_exact:.1f} N, FEM @ {n_elem_shape} elem = '
                 f'{load_factors[0]:.1f} N)')
    fig.tight_layout()
    fig.savefig('euler_column_buckling.png', dpi=150)
    print("Saved euler_column_buckling.png")


if __name__ == "__main__":
    main()
