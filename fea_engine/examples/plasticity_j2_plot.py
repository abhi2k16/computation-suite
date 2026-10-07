"""
plasticity_j2_plot.py -- Phase 6 (general-purpose extensions roadmap)
worked example: visual verification companion to plasticity_j2_demo.py.

The classic elasto-plastic stress-strain loop: a single Hex8PlasticJ2
element's sigma-eps response under uniaxial stress, loaded past yield
then unloaded, plotted against PlasticMaterial1D's closed-form loading
curve and the exact elastic-unload line (slope E) -- the visual
signature of correct plasticity (a sharp kink at yield, a straight-line
unload with permanent set) made directly checkable by eye.
"""
import numpy as np
import matplotlib.pyplot as plt

from fea_engine import Hex8PlasticJ2, PlasticMaterialJ2, FESystem
from fea_engine.mesh import Mesh
from fea_engine import nonlinear_solver as nls


E, NU, SIGMA_Y, H = 200e9, 0.3, 250e6, 1e9


def build_cube():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mesh = Mesh(nodes=nodes, elements=elements, dim=3)
    fs = FESystem(mesh, Hex8PlasticJ2(), sparse=False)
    x0_nodes = [0, 3, 4, 7]
    fs.fix_dofs(x0_nodes, [0])
    fs.fix_dofs([0], [1, 2])
    fs.fix_dofs([3], [2])
    fs.fix_dofs([4], [1])
    return fs, [1, 2, 5, 6]


def uniaxial_1d(sigma_applied, E, sigma_y, H):
    if abs(sigma_applied) <= sigma_y:
        return sigma_applied / E
    alpha = (sigma_applied - sigma_y) / H
    return sigma_applied / E + alpha


def main():
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    fs, x1_nodes = build_cube()
    sigma_peak = 1.8 * SIGMA_Y
    fs.add_nodal_force(x1_nodes, 0, sigma_peak)
    fs.init_state()

    lf_up = np.linspace(0.0, 1.0, 41)
    lf_down = np.linspace(1.0, 0.001, 40)
    load_factors = np.concatenate([lf_up, lf_down])
    _, U_hist = nls.solve_nonlinear_static(fs, mat, load_factors=load_factors,
                                            tol=1e-9, max_iter=50)

    eps_fe = U_hist[:, 3 * np.array(x1_nodes)].mean(axis=1)
    sigma_hist = load_factors * sigma_peak
    n_up = len(lf_up)

    eps_axis = np.linspace(0, eps_fe.max() * 1.05, 300)
    sigma_cf_loading = np.array([
        E * e if E * e <= SIGMA_Y else SIGMA_Y + H * (e - SIGMA_Y / E) for e in eps_axis])

    eps_p = (sigma_peak - SIGMA_Y) / H
    eps_unload_line = np.linspace(eps_p, eps_fe[n_up - 1], 50)
    sigma_unload_line = E * (eps_unload_line - eps_p)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot(eps_axis / 1e-2, sigma_cf_loading / 1e6, '-', color='k', linewidth=1.2,
             alpha=0.6, label='closed form (PlasticMaterial1D): loading')
    ax.plot(eps_unload_line / 1e-2, sigma_unload_line / 1e6, '--', color='0.4', linewidth=1.2,
             label='exact elastic unload (slope E)')
    ax.plot(eps_fe[:n_up] / 1e-2, sigma_hist[:n_up] / 1e6, 'o', color='tomato',
             markersize=4, label='Hex8PlasticJ2 FE (loading)', zorder=3)
    ax.plot(eps_fe[n_up:] / 1e-2, sigma_hist[n_up:] / 1e6, 's', color='steelblue',
             markersize=4, label='Hex8PlasticJ2 FE (unloading)', zorder=3)

    ax.axhline(SIGMA_Y / 1e6, color='0.8', linewidth=0.8, linestyle=':')
    ax.annotate('yield', xy=(0.05, SIGMA_Y / 1e6), xytext=(3, SIGMA_Y / 1e6 * 0.7),
                fontsize=9, color='0.3', arrowprops=dict(arrowstyle='->', color='0.5'))
    ax.axvline(eps_p / 1e-2, color='0.85', linewidth=0.8, linestyle=':')
    ax.annotate('permanent set', xy=(eps_p / 1e-2, 230), xytext=(eps_p / 1e-2 - 9, 280),
                fontsize=9, color='0.3', arrowprops=dict(arrowstyle='->', color='0.5'))

    ax.set_xlabel(r'axial strain $\varepsilon$ (%)')
    ax.set_ylabel(r'axial stress $\sigma$ (MPa)')
    ax.set_title('Hex8PlasticJ2 uniaxial stress: elastic-plastic loop vs. closed form')
    ax.grid(True, alpha=0.35)
    ax.legend(loc='upper left', fontsize=9)

    fig.tight_layout()
    fig.savefig('plasticity_j2_uniaxial_loop.png', dpi=150)
    print("Saved plasticity_j2_uniaxial_loop.png")


if __name__ == "__main__":
    main()
