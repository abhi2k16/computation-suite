# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
e2_recovery_speedup.py -- batched stress recovery against the per-element reference loop (claims C12, E2).

For each mesh the loop and the batched path are timed on the same displacement vector and their results compared.
The loop is only run up to ``LOOP_MAX`` elements because it costs about 2 ms per element.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from common import best_of, plt, save_fig, save_result
from fea_engine import D_plane_stress, D_solid3d, FESystem, Hex8Solid3D, Material, Quad4PlaneStress
from fea_engine.mesh import box_mesh, rectangle_mesh

MAT = Material(E=2.1e11, nu=0.3, rho=7850.0)
LOOP_MAX = 4000
QUAD_SIZES = [10, 20, 40, 60, 100]          # n x n Quad4
HEX_SIZES = [6, 10, 14, 20]                 # n^3 Hex8


def make(kind, n):
    if kind == "quad4":
        mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=n, ny=n)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=1.0, sparse=True)
        s.assemble_stiffness(D_plane_stress(MAT), thickness=1.0)
    else:
        mesh = box_mesh(1.0, 1.0, 1.0, n, n, n)
        s = FESystem(mesh, Hex8Solid3D(), sparse=True)
        s.assemble_stiffness(D_solid3d(MAT))
    U = np.random.default_rng(0).normal(size=s.n_dof) * 1e-4
    return s, U, len(mesh.elements)


def main():
    rows = []
    for kind, sizes in (("quad4", QUAD_SIZES), ("hex8", HEX_SIZES)):
        for n in sizes:
            s, U, ne = make(kind, n)
            tb, _ = best_of(lambda: s.stress(U), repeats=5, warmup=1)
            row = {"element": kind, "n": n, "n_elements": ne, "n_dof": int(s.n_dof), "t_batched_s": tb}
            if ne <= LOOP_MAX:
                tl, _ = best_of(lambda: s.stress(U, vectorized=False), repeats=2, warmup=0)
                a, b = np.asarray(s.stress(U)), np.asarray(s.stress(U, vectorized=False))
                row.update(t_loop_s=tl, speedup=tl / tb,
                           max_rel_diff=float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)))
            rows.append(row)
            print(row)
    save_result("e2_recovery_speedup", {"rows": rows, "loop_max_elements": LOOP_MAX,
                                        "timing_rule": "minimum over repeats (batched 5, loop 2)"})

    P = plt()
    fig, ax = P.subplots(figsize=(4.2, 3.0))
    for kind, c in (("quad4", "C0"), ("hex8", "C1")):
        r = [x for x in rows if x["element"] == kind]
        ax.loglog([x["n_elements"] for x in r], [x["t_batched_s"] for x in r], "o-", color=c, label=f"{kind} batched")
        rl = [x for x in r if "t_loop_s" in x]
        ax.loglog([x["n_elements"] for x in rl], [x["t_loop_s"] for x in rl], "s--", color=c, label=f"{kind} loop")
    ax.set_xlabel("number of elements"); ax.set_ylabel("time (s)"); ax.legend(fontsize=7)
    save_fig(fig, "e2_recovery_speedup")


if __name__ == "__main__":
    main()
