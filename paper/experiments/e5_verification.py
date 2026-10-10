# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
e5_verification.py -- independent benchmarks and observed convergence orders (claim E5).

Re-runs the cases of ``fea_engine/tests/test_benchmarks_convergence.py`` (importing its helpers, so the paper uses
exactly what the tests check) and records the numbers: patch tests for all eight continuum elements, pure bending of
a plane-stress beam, Euler-Bernoulli and axial-bar frequencies, a Timoshenko tip deflection, and the elastica.
Every reference value is analytic or computed independently (scipy quadrature for the elastica).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from common import ROOT, load_module, plt, save_fig, save_result
from fea_engine import (Beam2DEulerBernoulli, Beam2DReissner, D_plane_stress, FESystem, Material, NewtonOptions,
                        Quad4PlaneStress, convergence as cv, nonlinear_solver as ns)
from fea_engine.geometry import generate_mesh
from fea_engine.material import EI_beam, Section
from fea_engine.mesh import Mesh, rectangle_mesh

T = load_module(ROOT / "fea_engine" / "tests" / "test_benchmarks_convergence.py", "bench_helpers")


def study_dict(study, **extra):
    d = {"h": study.h, "values": study.values, "errors": study.errors, "pairwise_orders": study.orders,
         "fitted_order": float(study.order)}
    d.update(extra)
    return d


def main():
    out = {}

    # ---- patch tests on distorted meshes -------------------------------------------------------
    out["patch"] = []
    for cls in T.ELEMENTS:
        u_err, s_err = T._patch(cls)
        out["patch"].append({"element": cls.__name__, "displacement_error": float(u_err),
                             "stress_error": float(s_err)})
        print("patch", cls.__name__, f"{u_err:.1e} {s_err:.1e}")

    # ---- pure bending ------------------------------------------------------------------------
    out["pure_bending_quadratic"] = [{"element": c.__name__, "n": n, "ratio_minus_1": float(T._bending_ratio(n, c) - 1.0)}
                                     for c in (T.Quad8PlaneStress, T.Tri6PlaneStress) for n in (4, 8)]
    bend = {}
    for cls in (T.Quad4PlaneStress, T.Tri3PlaneStress):
        st = cv.run_study(lambda n, c=cls: T._bending_ratio(n, c), [8, 16, 32], exact=1.0)
        bend[cls.__name__] = study_dict(st, order_last_two=float(cv.observed_order(st.h[1:], st.errors[1:])))
        print("bending", cls.__name__, bend[cls.__name__]["values"], bend[cls.__name__]["order_last_two"])
    out["pure_bending_linear"] = bend

    # ---- vibration -----------------------------------------------------------------------------
    E, rho, A, I, L = 210e9, 7800.0, 0.01, 8.33e-6, 2.0
    lam = np.array([1.8751040687, 4.6940911330, 7.8547574382])
    exact = lam ** 2 / (2 * np.pi * L ** 2) * np.sqrt(E * I / (rho * A))

    def freqs(n):
        b = FESystem(generate_mesh(dim=1, L=L, n=n), Beam2DEulerBernoulli())
        b.assemble_stiffness(EI_beam(Material(E=E, nu=0.3, rho=rho), Section(A=A, I=I)))
        b.assemble_mass(rho * A)
        b.fix_dofs([0], ["uy", "rz"])
        return b.solve_modal(n_modes=3)[0]

    st = cv.run_study(lambda n: freqs(n)[0], [5, 10, 20], exact=exact[0])
    out["eb_cantilever_frequency"] = study_dict(
        st, exact_hz=exact, rel_error_n20=np.abs(freqs(20) / exact - 1.0))
    print("EB freq order", st.order)

    Eb, rhob, Lb = 1.0e9, 1000.0, 1.0
    exact_bar = np.sqrt(Eb / rhob) / (4.0 * Lb)

    def f_bar(n):
        mesh = rectangle_mesh(Lx=Lb, Ly=0.1, nx=n, ny=1)
        s = FESystem(mesh, Quad4PlaneStress(), thickness=0.1)
        s.assemble_stiffness(D_plane_stress(Material(E=Eb, nu=0.0, rho=rhob)), thickness=0.1)
        s.assemble_mass(rhob * np.eye(2), thickness=0.1)
        mesh.select_nodes(x=0.0, name="root")
        s.fix_dofs("root", ["ux"])
        s.fix_dofs(np.arange(len(mesh.nodes)), ["uy"])
        return s.solve_modal(n_modes=1)[0][0]

    st = cv.run_study(f_bar, [8, 16, 32], exact=exact_bar)
    out["axial_bar_frequency"] = study_dict(st, exact_hz=exact_bar)
    print("bar order", st.order)

    # ---- Timoshenko tip deflection -------------------------------------------------------------
    Et, nu, At, It, Lt, P = 210e9, 0.3, 1.0e-3, 8.33e-7, 1.0, 1000.0
    Gt = Et / (2 * (1 + nu))

    def tip(n):
        x = np.linspace(0, Lt, n + 1).reshape(-1, 1)
        s = FESystem(Mesh(nodes=np.hstack([x, 0 * x]), elements=np.array([[i, i + 1] for i in range(n)]), dim=1),
                     Beam2DReissner())
        s.fix_dofs([0], ["ux", "uy", "rz"])
        s.assemble_stiffness((Et, Gt, At, It, 1.0))
        s.add_nodal_force([n], "uy", P)
        return s.solve_static().component("uy", nodes=[n])[0]

    exact_t = P * Lt ** 3 / (3 * Et * It) + P * Lt / (Gt * At)
    st = cv.run_study(tip, [2, 4, 8, 16], exact=exact_t)
    out["timoshenko_tip"] = study_dict(st, exact=exact_t)
    print("Timoshenko order", st.order)

    # ---- elastica -------------------------------------------------------------------------------
    Ee, Ae, Ie, Le = 210e9, 0.05, 8.33e-7, 1.0
    Ge = Ee / (2 * (1 + nu))
    w_ref, sh_ref = T._elastica_tip(1.0)

    def elastica(n):
        x = np.linspace(0, Le, n + 1).reshape(-1, 1)
        s = FESystem(Mesh(nodes=np.hstack([x, 0 * x]), elements=np.array([[i, i + 1] for i in range(n)]), dim=1),
                     Beam2DReissner())
        s.fix_dofs([0], ["ux", "uy", "rz"])
        mat = (Ee, Ge, Ae, Ie, 1.0)
        s.assemble_stiffness(mat)
        s.add_nodal_force([n], "uy", Ee * Ie / Le ** 2)
        _, hist = ns.solve_nonlinear_static(s, mat, n_steps=10, options=NewtonOptions(tol=1e-8, max_iter=60))
        u = s.field(hist[-1])
        return u.component("uy", nodes=[n])[0] / Le, -u.component("ux", nodes=[n])[0] / Le

    rows = []
    for n in (10, 20, 40):
        w, sh = elastica(n)
        rows.append({"n": n, "w_over_L": w, "shortening_over_L": sh, "w_rel_error": abs(w - w_ref) / w_ref,
                     "shortening_rel_error": abs(sh - sh_ref) / sh_ref})
        print("elastica", rows[-1])
    out["elastica"] = {"load_parameter_PL2_over_EI": 1.0, "w_ref": w_ref, "shortening_ref": sh_ref, "rows": rows,
                       "reference": "independent scipy quadrature of the elastica integral"}

    save_result("e5_verification", out)

    P_ = plt()
    fig, axes = P_.subplots(1, 3, figsize=(9.0, 2.8))
    for name, c in (("Quad4PlaneStress", "C0"), ("Tri3PlaneStress", "C1")):
        b = bend[name]
        axes[0].loglog(b["h"], b["errors"], "o-", color=c, label=name.replace("PlaneStress", ""))
    axes[0].set_title("pure bending (plane stress)")
    f = out["eb_cantilever_frequency"]; axes[1].loglog(f["h"], f["errors"], "o-", label="Euler-Bernoulli $f_1$")
    g = out["axial_bar_frequency"]; axes[1].loglog(g["h"], g["errors"], "s-", label="axial bar $f_1$")
    axes[1].set_title("frequencies")
    t = out["timoshenko_tip"]; axes[2].loglog(t["h"], t["errors"], "o-", label="Reissner beam")
    axes[2].set_title("Timoshenko tip deflection")
    from matplotlib.ticker import NullFormatter
    for ax, ref, (h, e) in zip(axes, (2, 4, 2), ((bend["Quad4PlaneStress"]["h"], bend["Quad4PlaneStress"]["errors"]),
                                                  (f["h"], f["errors"]), (t["h"], t["errors"]))):
        h, e = np.asarray(h, float), np.asarray(e, float)
        ax.loglog(h, e[-1] * (h / h[-1]) ** ref * 0.5, "k:", lw=1, label=f"slope {ref}")      # reference slope
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("element size $h$"); ax.set_ylabel("absolute error"); ax.legend(fontsize=7)
    save_fig(fig, "e5_verification")


if __name__ == "__main__":
    main()
