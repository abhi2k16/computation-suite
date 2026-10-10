# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
e6_torch_backend.py -- optional PyTorch paths: ReducedSystem.solve and stress recovery, CPU and (if present) GPU
(claim E6).

Needs torch. Without it the script records that it was skipped and exits with status 0, so ``run_all.py`` still
completes. Run it on the GPU machine to fill the GPU columns; the article quotes only what is stored in
``results/e6_torch_backend.json``.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from common import best_of, plt, save_fig, save_result
from fea_engine import D_plane_stress, FESystem, Material, Quad4PlaneStress
from fea_engine.mesh import rectangle_mesh

MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)
SIZES = [(32, 16), (64, 32), (128, 64), (256, 128)]          # (nx, ny) Quad4 plates
DENSE_MAX_DOF = 12000


def plate(nx, ny):
    mesh = rectangle_mesh(Lx=2.0, Ly=1.0, nx=nx, ny=ny)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02, sparse=True)
    s.assemble_stiffness(D_plane_stress(MAT), thickness=0.02)
    mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=2.0, name="tip")
    s.fix_dofs("root", ["ux", "uy"])
    s.add_nodal_force("tip", "uy", -1.0e4)
    return s


def main():
    try:
        import torch
    except Exception as exc:                                   # missing, or a CUDA wheel without CUDA runtime
        save_result("e6_torch_backend", {"skipped": True, "reason": f"torch not importable: {exc}"})
        print("torch not available; recorded as skipped")
        return

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def sync(dev):
        if dev == "cuda":
            torch.cuda.synchronize()

    rows = []
    for nx, ny in SIZES:
        s = plate(nx, ny)
        rs = s.form_linear_system()
        row = {"nx": nx, "ny": ny, "n_free": int(rs.n_free), "n_elements": int(nx * ny)}
        ref = np.asarray(rs.solve())
        row["t_scipy_s"] = best_of(lambda: rs.solve(), repeats=3)[0]
        for dev in devices:
            for method in ("cg", "dense"):
                if method == "dense" and rs.n_free > DENSE_MAX_DOF:
                    continue
                def run():
                    out = rs.solve(backend="torch", device=dev, method=method, tol=1e-10)
                    sync(dev)
                    return out
                got = np.asarray(run())
                tmin = best_of(run, repeats=3, warmup=0)[0]
                row[f"t_torch_{dev}_{method}_s"] = tmin
                row[f"rel_diff_{dev}_{method}"] = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        U = rs.solve()
        row["t_stress_numpy_s"] = best_of(lambda: s.stress(U), repeats=3)[0]
        for dev in devices:
            def srun():
                out = s.stress(U, backend="torch", device=dev)
                sync(dev)
                return out
            got = np.asarray(srun())
            row[f"t_stress_torch_{dev}_s"] = best_of(srun, repeats=3, warmup=0)[0]
            row[f"stress_rel_diff_{dev}"] = float(np.abs(got - np.asarray(s.stress(U))).max()
                                                 / np.abs(np.asarray(s.stress(U))).max())
        rows.append(row)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})

    save_result("e6_torch_backend", {"skipped": False, "devices": devices, "rows": rows,
                                     "note": "CG uses a Jacobi preconditioner, tol 1e-10; float64 throughout"})

    P = plt()
    fig, ax = P.subplots(figsize=(4.2, 3.0))
    x = [r["n_free"] for r in rows]
    ax.loglog(x, [r["t_scipy_s"] for r in rows], "o-", label="SciPy (rs.solve)")
    for dev in devices:
        ax.loglog(x, [r[f"t_torch_{dev}_cg_s"] for r in rows], "s--", label=f"torch CG ({dev})")
    ax.set_xlabel("free DOF"); ax.set_ylabel("solve time (s)"); ax.legend(fontsize=7)
    save_fig(fig, "e6_torch_backend")


if __name__ == "__main__":
    main()
