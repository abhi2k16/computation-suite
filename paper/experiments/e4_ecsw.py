# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
e4_ecsw.py -- ECSW hyper-reduction of a geometrically nonlinear clamped-clamped beam (claim E4).

Same model and training/test draws as ``rom_engine/tests/test_hyper_reduction.py``. Reports the number of elements
ECSW keeps, the reduced-force error on held-out states, the time of one reduced-force evaluation (full assembly
against ECSW) and the error and wall time of a 300-step RK4 trajectory.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from common import ROOT, best_of, load_module, plt, save_fig, save_result
from rom_engine.hyper_reduction import ECSW, HyperReducedNonlinearROM, reduced_force_error
from rom_engine.intrusive_nonlinear_rom import IntrusiveNonlinearROM

TOLS = [1e-2, 1e-3, 1e-4, 1e-5]
N_ELEM_LIST = [40, 120]


def run(n_elem, tol):
    fx = load_module(ROOT / "rom_engine" / "tests" / "fea_fixtures.py", "fea_fixtures")
    d = fx.clamped_clamped_nonlinear_beam_system(n_elem=n_elem, n_modes=3, damping_alpha=2.0)
    fes, mat = d["sys"], d["mat"]
    free = np.asarray(d["free_dofs"]); n = d["n_dof"]
    V = np.zeros((n, 3)); V[free, :] = d["V"]
    conn = fes.mesh.elements
    elem = fes._blocks[0][1]
    dofs = [np.asarray(fes._global_dofs(c)) for c in conn]
    force = lambda e, u: elem.internal_force(fes.mesh.nodes[conn[e]], u, mat)
    tang = lambda e, u: elem.tangent_stiffness(fes.mesh.nodes[conn[e]], u, mat)
    h = np.sqrt(12 * 8.333e-9 / 1e-4)
    peaks = d["mode_shape_peaks"]
    rng = np.random.default_rng(0)
    Q = rng.uniform(-1.5, 1.5, (3, 60)) * h / peaks[:, None]
    ec = ECSW.fit(V, V @ Q, dofs, force, tang, tol=tol)
    full = lambda q: V.T @ fes.assemble_internal_force(V @ q, mat)
    Qt = (np.random.default_rng(5).uniform(-1.5, 1.5, (3, 20)) * h / peaks[:, None]).T
    ferr = float(reduced_force_error(ec, full, Qt).max())

    q1 = np.array([0.5, -0.3, 0.2]) * h / peaks
    t_full, _ = best_of(lambda: full(q1), repeats=7)
    t_hyp, _ = best_of(lambda: ec.reduced_force(q1), repeats=7)

    Mf, Cf = np.array(fes.M), np.array(fes.C)
    K0 = fes.assemble_tangent_stiffness(np.zeros(n), mat)
    zero = lambda t: np.zeros(n)
    rom_full = IntrusiveNonlinearROM(V, Mf, Cf, lambda u: fes.assemble_internal_force(u, mat), zero,
                                     tangent_fn=lambda u: fes.assemble_tangent_stiffness(u, mat))
    rom_hyp = HyperReducedNonlinearROM(V, Mf, Cf, zero, K0, ec)
    q0 = np.array([1.0, 0.0, 0.0]) * h / peaks
    dt, steps = 2e-5, 300
    run_full = lambda: rom_full.integrate_rk4(q0, np.zeros(3), dt, steps)
    run_hyp = lambda: rom_hyp.integrate_rk4(q0, np.zeros(3), dt, steps)
    tf, _ = best_of(run_full, repeats=3, warmup=0)
    th, _ = best_of(run_hyp, repeats=3, warmup=0)
    qf, qh = np.asarray(run_full()[1]), np.asarray(run_hyp()[1])
    return {"n_elements": n_elem, "tol": tol, "n_selected": int(ec.n_selected), "n_total": int(ec.n_total),
            "heldout_force_error_max": ferr, "t_force_full_s": t_full, "t_force_ecsw_s": t_hyp,
            "force_speedup": t_full / t_hyp, "t_traj_full_s": tf, "t_traj_hyper_s": th,
            "traj_speedup": tf / th, "traj_rel_error": float(np.linalg.norm(qh - qf) / np.linalg.norm(qf))}


def main():
    rows = []
    for n_elem in N_ELEM_LIST:
        for tol in (TOLS if n_elem == N_ELEM_LIST[0] else [1e-4]):
            r = run(n_elem, tol)
            rows.append(r)
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    save_result("e4_ecsw", {"rows": rows, "n_modes": 3, "rk4_dt": 2e-5, "rk4_steps": 300})

    P = plt()
    fig, ax = P.subplots(figsize=(4.2, 3.0))
    r = [x for x in rows if x["n_elements"] == N_ELEM_LIST[0]]
    ax.loglog([x["n_selected"] for x in r], [x["heldout_force_error_max"] for x in r], "o-")
    for x in r:
        ax.annotate(f"tol={x['tol']:g}", (x["n_selected"], x["heldout_force_error_max"]), fontsize=6,
                    textcoords="offset points", xytext=(4, 4))
    from matplotlib.ticker import NullFormatter, FixedLocator
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_major_locator(FixedLocator([x["n_selected"] for x in r]))
    ax.xaxis.set_major_formatter(__import__("matplotlib").ticker.ScalarFormatter())
    ax.set_xlabel(f"elements kept (of {r[0]['n_total']})"); ax.set_ylabel("held-out reduced-force error")
    save_fig(fig, "e4_ecsw")


if __name__ == "__main__":
    main()
