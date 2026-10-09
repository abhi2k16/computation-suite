"""
fea_05_modal_analysis.py -- Example Gallery: generalized eigenproblem
modal analysis (FESystem.solve_modal()), mirroring TensorMesh's
Example Gallery / Modal Analysis page ("cantilever vibration" is
literally named there as one of its two worked cases).

A steel cantilever (Quad4PlaneStress), K*phi = omega^2*M*phi solved on
the free DOFs. Plots the first four mode shapes (deformed mesh,
magnified) with their natural frequencies, plus a cross-check against
the classical Euler-Bernoulli cantilever bending-frequency formula
for the two bending-dominated modes -- an independent, closed-form
reference, not just "the eigensolver ran."
"""
__author__ = "Abhijeet"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

from fea_engine import Quad4PlaneStress, FESystem, Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh

STEEL = Material(E=2.1e11, nu=0.3, rho=7850.0)
L, H = 4.0, 0.4
NX, NY = 30, 4
BEAM_BETA_L = [1.8751, 4.6941, 7.8548, 10.9955]   # Euler-Bernoulli cantilever roots


def main():
    mesh = rectangle_mesh(Lx=L, Ly=H, nx=NX, ny=NY)
    D = D_plane_stress(STEEL)
    fs = FESystem(mesh, Quad4PlaneStress())
    fs.assemble_stiffness(D)
    fs.assemble_mass(STEEL.rho * np.eye(2))
    left = mesh.nodes_on_line(axis=0, value=0.0)
    for n in left:
        fs.fix_dofs([n], [0, 1])

    n_modes = 4
    freq_hz, mode_shapes = fs.solve_modal(n_modes=n_modes)

    # Euler-Bernoulli reference for the two OUT-OF-PLANE-bending-like
    # modes among these four (a 2-D plane-stress cantilever also has
    # axial/other modes the 1-D beam formula doesn't cover -- reported
    # for context, not asserted to match every mode).
    I = H**3 / 12.0
    A = H * 1.0
    f_eb = [(bl**2 / (2 * np.pi * L**2)) * np.sqrt(STEEL.E * I / (STEEL.rho * A))
            for bl in BEAM_BETA_L[:2]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
    for m, ax in enumerate(axes.flat):
        disp = np.zeros_like(mesh.nodes)
        for n in range(len(mesh.nodes)):
            disp[n] = mode_shapes[fs._global_dofs([n]), m]
        norm = np.max(np.linalg.norm(disp, axis=1))
        mag = 0.6 * H / max(norm, 1e-12)
        deformed = mesh.nodes + mag * disp

        undeformed = mesh.nodes[mesh.elements]
        ax.add_collection(LineCollection([np.vstack([q, q[0]]) for q in undeformed],
                                          colors='0.85', linewidths=0.5, linestyles='dashed'))
        quads = deformed[mesh.elements]
        vals = np.linalg.norm(disp[mesh.elements].mean(axis=1), axis=1)
        pc = PolyCollection(quads, array=vals, cmap='coolwarm', edgecolors='0.3', linewidths=0.2)
        ax.add_collection(pc)
        ax.set_xlim(-0.3, L + 0.3)
        ax.set_ylim(-H * 2, H * 3)
        ax.set_aspect('equal')
        title = f'Mode {m+1}: {freq_hz[m]:.2f} Hz'
        if m < 2:
            title += f'  (E-B bending ref: {f_eb[m]:.2f} Hz)'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x (m)')

    fig.suptitle('Modal Analysis -- steel cantilever, Quad4PlaneStress (fea_engine)', fontsize=13)
    fig.savefig('fea_05_modal_analysis.png', dpi=150)
    print("Saved fea_05_modal_analysis.png")
    print(f"FE frequencies (Hz): {np.round(freq_hz, 3)}")
    print(f"Euler-Bernoulli bending reference (first 2 modes, Hz): {np.round(f_eb, 3)}")


if __name__ == "__main__":
    main()
