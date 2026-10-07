"""
hyperelastic_neo_hookean_plot.py -- Phase 6 (general-purpose
extensions roadmap) worked example: visual verification companion to
hyperelastic_neo_hookean_demo.py.

Two panels: (1) log-log convergence of the small-strain relative error
against a slope-1 (O(strain)) reference line -- the correct small-
strain-limit signature, made visible as a straight line at the right
slope rather than just "small numbers"; (2) the external-work-vs-
stored-energy check from the full nonlinear solve, shown as a running
comparison along the loading path (not just the final numbers) --
work and energy should track each other at every step, not just agree
once at the end.
"""
import numpy as np
import matplotlib.pyplot as plt

from fea_engine import Tet4NeoHookean, NeoHookeanMaterial, FESystem, D_solid3d, Material
from fea_engine.mesh import Mesh
from fea_engine.material import neo_hookean_pk2_stress
from fea_engine import nonlinear_solver as nls


E, NU = 1.0e6, 0.4


def main():
    mat = NeoHookeanMaterial(E=E, nu=NU)
    D_el = D_solid3d(Material(E=E, nu=NU))

    # ---- Panel 1: small-strain convergence ----
    eps_list = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    rel_errs = []
    for eps_tiny in eps_list:
        F = np.eye(3) + eps_tiny * np.diag([1.0, -NU, -NU])
        S, _ = neo_hookean_pk2_stress(F, mat)
        sigma_lin = D_el[0] @ np.array([eps_tiny, -NU * eps_tiny, -NU * eps_tiny, 0, 0, 0])
        rel_errs.append(abs(S[0, 0] - sigma_lin) / abs(sigma_lin))
    rel_errs = np.array(rel_errs)

    # ---- Panel 2: work vs. stored energy along the loading path ----
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3]])
    mesh = Mesh(nodes=coords, elements=elements, dim=3)
    elem = Tet4NeoHookean()
    fs = FESystem(mesh, elem, sparse=False)
    fs.fix_dofs([0, 1, 2], [0, 1, 2])
    F_applied_total = np.array([5e4, 3e4, -2e4])
    fs.F[9:12] = F_applied_total

    n_steps = 50
    load_factors, U_hist = nls.solve_nonlinear_static(fs, mat, n_steps=n_steps, tol=1e-10, max_iter=60)
    u3_hist = U_hist[:, 9:12]

    work_running = [0.0]
    energy_running = [0.0]
    for i in range(1, len(load_factors)):
        F_avg = 0.5 * (load_factors[i] + load_factors[i - 1]) * F_applied_total
        work_running.append(work_running[-1] + F_avg @ (u3_hist[i] - u3_hist[i - 1]))
        F_def, _, vol = elem._deformation_gradient(mesh.nodes, U_hist[i])
        _, W = neo_hookean_pk2_stress(F_def, mat)
        energy_running.append(vol * W)
    work_running = np.array(work_running)
    energy_running = np.array(energy_running)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    ax1.loglog(eps_list, rel_errs, 'o-', color='seagreen', label='Neo-Hookean vs. linear elastic')
    ref = rel_errs[0] * (eps_list / eps_list[0])
    ax1.loglog(eps_list, ref, 'k--', linewidth=1, label=r'$O(\varepsilon)$ reference slope')
    ax1.set_xlabel(r'strain $\varepsilon$')
    ax1.set_ylabel('relative error in $S_{11}$ vs. linear elastic')
    ax1.set_title('Small-strain limit convergence')
    ax1.invert_xaxis()
    ax1.grid(True, which='both', alpha=0.4)
    ax1.legend()

    ax2.plot(load_factors, work_running, 'o-', color='tomato', markersize=4,
              label='external work (path integral)')
    ax2.plot(load_factors, energy_running, '-', color='steelblue', linewidth=2.5,
              alpha=0.6, label='stored strain energy $V_0 W(F)$')
    ax2.set_xlabel(r'load factor $\lambda$')
    ax2.set_ylabel('energy (J)')
    ax2.set_title('Work-energy balance along the loading path')
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc='upper left')

    fig.suptitle('Tet4NeoHookean verification')
    fig.savefig('hyperelastic_neo_hookean.png', dpi=150)
    print("Saved hyperelastic_neo_hookean.png")


if __name__ == "__main__":
    main()
