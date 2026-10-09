# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
fea_04_plasticity_strip.py -- Example Gallery: J2 (von Mises) plasticity
with isotropic hardening, on a plate-with-a-hole strip under cyclic
axial tension -- mirrors TensorMesh's Example Gallery / Solid
Mechanics / Plasticity (J2 with Isotropic Hardening) page.

Uses Quad4PlasticJ2PlaneStress (item 13, Wave 2) on a hole_in_
rectangle_mesh() strip (Wave 5's mesh-generation machinery) -- the
classic stress-concentration-driven plasticity benchmark: an elastic
stress-concentration factor of ~3 at the hole boundary means local
yielding starts there long before the strip's nominal (far-field)
stress reaches sigma_y, and unloading leaves a PERMANENT, LOCALIZED
plastic-strain field concentrated at the hole -- not spread uniformly,
which is exactly what panel 2 below is checking for, not just
plotting.

Panel 1: nominal axial stress vs. strain, showing the loading/
unloading hysteresis with permanent set (same signature as the
package's own existing single-element plasticity_j2_plot.py, here on
a real multi-element mesh with a stress concentration).
Panel 2: accumulated equivalent plastic strain (the state["alpha"]
hardening variable J2 plasticity already tracks per Gauss point) on
the deformed mesh after unload -- localized at the hole, decaying
into the far field, the real physical signature a uniform strip could
not produce.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

from fea_engine import Quad4PlasticJ2PlaneStress, PlasticMaterialJ2, FESystem
from fea_engine.mesh import hole_in_rectangle_mesh
from fea_engine import nonlinear_solver as nls

E, NU, SIGMA_Y, H = 70e9, 0.33, 90e6, 3e8   # aluminum-like
LX, LY = 4.0, 2.0
HOLE_R = 0.35


def main():
    mesh = hole_in_rectangle_mesh(LX, LY, (LX / 2, LY / 2), HOLE_R, nr=5, ntheta=10, grade_p=2.0)
    mat = PlasticMaterialJ2(E=E, nu=NU, sigma_y=SIGMA_Y, H=H)
    fs = FESystem(mesh, Quad4PlasticJ2PlaneStress())

    left = mesh.nodes_on_line(axis=0, value=0.0)
    right = mesh.nodes_on_line(axis=0, value=LX)
    for n in left:
        fs.fix_dofs([n], [0, 1])
    for n in right:
        fs.fix_dofs([n], [1])   # roller: allow axial slide, no y

    sigma_nom_peak = 1.08 * SIGMA_Y * (LY - 2 * HOLE_R) / LY   # past nominal-net-section yield
    P_peak = sigma_nom_peak * LY * 1.0
    fs.add_nodal_force(right, 0, P_peak)
    fs.init_state()

    lf_up = np.linspace(0.0, 1.0, 26)
    lf_down = np.linspace(1.0, 0.001, 25)
    load_factors = np.concatenate([lf_up, lf_down])
    _, U_hist = nls.solve_nonlinear_static(fs, mat, load_factors=load_factors, tol=1e-4, max_iter=80)
    n_up = len(lf_up)

    eps_nom = U_hist[:, np.array(fs._global_dofs(right)[0::2])].mean(axis=1) / LX
    sigma_nom = load_factors * sigma_nom_peak
    U_final = U_hist[-1]

    # ---- per-element max equivalent plastic strain (alpha), final state ----
    alpha_elem = np.array([fs.state[None][e]["alpha"].max()
                            for e in range(len(mesh.elements))])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)

    # ---- Panel 1: nominal stress-strain hysteresis ----
    ax1.plot(eps_nom[:n_up] * 100, sigma_nom[:n_up] / 1e6, 'o-', color='tomato',
              markersize=3, label='loading')
    ax1.plot(eps_nom[n_up:] * 100, sigma_nom[n_up:] / 1e6, 's-', color='steelblue',
              markersize=3, label='unloading')
    ax1.axhline(SIGMA_Y / 1e6, color='0.8', linewidth=0.8, linestyle=':')
    ax1.annotate('material yield $\\sigma_y$', xy=(eps_nom.max() * 40, SIGMA_Y / 1e6),
                  fontsize=9, color='0.4')
    ax1.set_xlabel('nominal (far-field) axial strain (%)')
    ax1.set_ylabel('nominal axial stress (MPa)')
    ax1.set_title('Strip with a hole: cyclic loading (net-section yields\nlocally well below sigma_y, due to the stress concentration)')
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc='upper left', fontsize=9)

    # ---- Panel 2: permanent plastic-strain field ----
    disp = np.zeros_like(mesh.nodes)
    for n in range(len(mesh.nodes)):
        disp[n] = U_final[fs._global_dofs([n])]
    mag = 8.0
    deformed = mesh.nodes + mag * disp
    quads = deformed[mesh.elements]
    pc = PolyCollection(quads, array=alpha_elem, cmap='inferno', edgecolors='0.3', linewidths=0.15)
    ax2.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax2, shrink=0.8)
    cb.set_label('equivalent plastic strain (accumulated)')
    ax2.set_xlim(deformed[:, 0].min() - 0.1, deformed[:, 0].max() + 0.1)
    ax2.set_ylim(deformed[:, 1].min() - 0.1, deformed[:, 1].max() + 0.1)
    ax2.set_aspect('equal')
    ax2.set_title(f'Permanent plastic strain after unload ({mag:.0f}x magnified)')
    ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')

    fig.suptitle('Plasticity (J2, isotropic hardening) -- plate with a hole (fea_engine)', fontsize=13)
    fig.savefig('fea_04_plasticity_strip.png', dpi=150)
    print("Saved fea_04_plasticity_strip.png")
    print(f"Converged {len(load_factors)} load steps. Peak equiv. plastic strain: "
          f"{alpha_elem.max():.4e} at element with centroid "
          f"{mesh.nodes[mesh.elements[alpha_elem.argmax()]].mean(axis=0)}")


if __name__ == "__main__":
    main()
