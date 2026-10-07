"""
fea_06_topology_optimization.py -- Example Gallery: SIMP density-based
compliance topology optimization (topopt.py, Wave 11 item 111),
mirroring TensorMesh's Example Gallery / Inverse Design &
Identification page (its own worked example is thermal-compliance
topology optimization via an Optimality-Criteria optimizer -- this is
the structural-compliance analogue, same OC-update mechanism, native
to fea_engine, not a port of TensorMesh code).

Short-cantilever benchmark (test_topopt.py's own validated setup):
fixed left edge, point load at the bottom-right corner. Snapshots the
density field at several iterations by calling topology_optimize_
compliance() repeatedly, chaining each call's final rho as the next
call's rho_init (the function itself only returns the FINAL density,
so this is how intermediate snapshots are captured without touching
topopt.py's own internals).

Panel 1: density-field evolution (a small multi-panel strip).
Panel 2: compliance history, full run -- the "large reduction, volume
held exactly" signature this driver's own test suite already
validates (test_topology_optimize_compliance_reduces_compliance_and_
holds_volume), shown as a convergence curve here.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from fea_engine import Quad4PlaneStress, FESystem, Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh
from fea_engine.topopt import topology_optimize_compliance

STEEL = Material(E=2.1e11, nu=0.3, rho=7850.0)
NX, NY = 60, 30
VOL_FRAC = 0.4


def short_cantilever():
    mesh = rectangle_mesh(Lx=float(NX), Ly=float(NY), nx=NX, ny=NY)
    fs = FESystem(mesh, Quad4PlaneStress())
    left = mesh.nodes_on_line(axis=0, value=0.0)
    for n in left:
        fs.fix_dofs([n], [0, 1])
    F_ext = np.zeros(fs.n_dof)
    br = [n for n in mesh.nodes_on_line(axis=0, value=float(NX))
          if abs(mesh.nodes[n, 1] - 0.0) < 1e-9]
    F_ext[fs._global_dofs(br)[1]] = -1.0e6
    return fs, F_ext


def main():
    fs, F_ext = short_cantilever()
    D = D_plane_stress(STEEL)
    connectivity = fs.mesh.elements

    snapshot_its = [1, 3, 8, 20, 60]
    rho_snapshots = []
    compliance_all = []
    rho_current = None
    done = 0
    for target in snapshot_its:
        n_iter = target - done
        result = topology_optimize_compliance(
            fs, D, F_ext, volume_fraction=VOL_FRAC, p=3.0, move=0.2,
            n_iter=n_iter, filter_radius=1.5, rho_init=rho_current, tol=0.0)
        rho_current = result["rho"]
        compliance_all.extend(result["compliance_history"])
        rho_snapshots.append(rho_current.copy())
        done = target

    fig = plt.figure(figsize=(13, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, len(snapshot_its))

    for i, (it, rho) in enumerate(zip(snapshot_its, rho_snapshots)):
        ax = fig.add_subplot(gs[0, i])
        quads = fs.mesh.nodes[connectivity]
        pc = PolyCollection(quads, array=rho, cmap='Greys', edgecolors='none')
        pc.set_clim(0, 1)
        ax.add_collection(pc)
        ax.set_xlim(0, NX); ax.set_ylim(0, NY)
        ax.set_aspect('equal')
        ax.set_title(f'iter {it}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(np.arange(1, len(compliance_all) + 1), compliance_all, 'o-',
              color='steelblue', markersize=3)
    ax2.set_xlabel('OC iteration')
    ax2.set_ylabel('compliance $C = F^T u$ (J)')
    ax2.set_title(f'Compliance history -- reduced by '
                   f'{(1 - compliance_all[-1]/compliance_all[0])*100:.1f}% '
                   f'at fixed volume fraction {VOL_FRAC}')
    ax2.grid(True, alpha=0.35)

    fig.suptitle('Topology Optimization -- SIMP compliance minimization (fea_engine)', fontsize=13)
    fig.savefig('fea_06_topology_optimization.png', dpi=150)
    print("Saved fea_06_topology_optimization.png")
    print(f"Compliance: {compliance_all[0]:.4e} -> {compliance_all[-1]:.4e} "
          f"({(1 - compliance_all[-1]/compliance_all[0])*100:.1f}% reduction), "
          f"final mean(rho)={rho_current.mean():.4f} (target {VOL_FRAC})")


if __name__ == "__main__":
    main()
