# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
e3_cms.py -- Craig-Bampton component mode synthesis of a clamped-clamped beam split in two halves (claim E3).

The reference is the generalised eigenproblem of the unreduced full beam. Error is max over the first four
frequencies of |f_cms - f_full| / f_full. Same model as ``rom_engine/tests/test_cms.py``.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import eigh

from common import plt, save_fig, save_result
from fea_engine import elements
from fea_engine.geometry import generate_mesh
from fea_engine.material import EI_beam, Material, Section
from fea_engine.solver import FESystem
from rom_engine.cms import couple, craig_bampton

E, RHO, A, I, N = 210e9, 7800.0, 0.01, 8.33e-6, 20
MAT = Material(E=E, nu=0.3, rho=RHO)


def beam(L, n, fixed):
    s = FESystem(generate_mesh(dim=1, L=L, n=n), elements.Beam2DEulerBernoulli())
    s.assemble_stiffness(EI_beam(MAT, Section(A=A, I=I)))
    s.assemble_mass(RHO * A)
    s.fix_dofs(fixed, [0, 1])
    fr = np.asarray(s.free_dofs)
    return fr, np.array(s.K)[np.ix_(fr, fr)], np.array(s.M)[np.ix_(fr, fr)]


def main():
    fr, K, M = beam(1.0, 2 * N, [0, 2 * N])
    f_full = np.sqrt(np.clip(eigh(K, M, eigvals_only=True), 0, None)) / (2 * np.pi)
    frL, KL, ML = beam(0.5, N, [0])
    frR, KR, MR = beam(0.5, N, [N])
    pos = lambda f, d: int(np.where(f == d)[0][0])
    ibL = [pos(frL, 2 * N), pos(frL, 2 * N + 1)]
    ibR = [pos(frR, 0), pos(frR, 1)]

    rows = []
    for nm in (1, 2, 3, 5, 8, 10, 15, 20, None):
        cl = craig_bampton(KL, ML, ibL, nm)
        cr = craig_bampton(KR, MR, ibR, nm)
        cp = couple([cl, cr], [[0, 1], [0, 1]])
        f, _ = cp.solve_modal(4)
        err = np.abs(f - f_full[:4]) / f_full[:4]
        rows.append({"modes_per_half": "all" if nm is None else nm, "reduced_size": int(cp.K.shape[0]),
                     "max_rel_error_first4": float(err.max()), "rel_error_per_mode": err})
        print(rows[-1]["modes_per_half"], rows[-1]["reduced_size"], f"{err.max():.2e}")

    save_result("e3_cms", {"n_elements_full": 2 * N, "n_dof_full_free": int(len(fr)), "f_full_hz": f_full[:4],
                           "rows": rows})

    P = plt()
    fig, ax = P.subplots(figsize=(4.2, 3.0))
    r = [x for x in rows if x["modes_per_half"] != "all"]
    ax.semilogy([x["modes_per_half"] for x in r], [x["max_rel_error_first4"] for x in r], "o-",
                label="Craig-Bampton")
    ax.set_xlabel("fixed-interface modes per substructure"); ax.set_ylabel("max relative frequency error")
    save_fig(fig, "e3_cms")


if __name__ == "__main__":
    main()
