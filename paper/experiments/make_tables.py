# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
make_tables.py -- turn ``paper/results/*.json`` into LaTeX.

Writes ``paper/tables/*.tex`` (booktabs tables) and ``paper/tables/numbers.tex``, a file of ``\\newcommand`` macros
holding every number the text quotes, so the prose never contains a hand-typed measurement.
Run after the experiments: ``python paper/experiments/make_tables.py``.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import json

from common import PAPER, RESULTS

TABLES = PAPER / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def sci(x, digits=1):
    if x == 0:
        return "$0$"
    m, e = f"{x:.{digits}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def num(x, digits=2):
    return f"{x:.{digits}f}"


def write(name, text):
    (TABLES / name).write_text(text)
    print("wrote", (TABLES / name).relative_to(PAPER.parent))


def tabular(cols, header, rows, caption_note=None):
    lines = ["\\begin{tabular}{" + cols + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    macros = []

    def macro(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # ---------------------------------------------------------------- verification (e5)
    e5 = load("e5_verification")
    if e5:
        d = e5["data"]
        rows = [[f"\\texttt{{{r['element']}}}", sci(r["displacement_error"]), sci(r["stress_error"])]
                for r in d["patch"]]
        write("tab_patch.tex", tabular("lrr", ["Element", "displacement error", "stress error"], rows))
        orders = [
            ("Plane-stress pure bending, Quad4", d["pure_bending_linear"]["Quad4PlaneStress"]["order_last_two"], 2),
            ("Plane-stress pure bending, Tri3", d["pure_bending_linear"]["Tri3PlaneStress"]["order_last_two"], 2),
            ("Euler--Bernoulli beam, first frequency", d["eb_cantilever_frequency"]["fitted_order"], 4),
            ("Fixed-free bar, first frequency (Quad4)", d["axial_bar_frequency"]["fitted_order"], 2),
            ("Reissner beam, Timoshenko tip deflection", d["timoshenko_tip"]["fitted_order"], 2)]
        write("tab_orders.tex", tabular("lrr", ["Benchmark", "observed order", "theory"],
                                        [[n, num(o, 2), str(t)] for n, o, t in orders]))
        rows = [[str(r["n"]), num(r["w_over_L"], 6), num(r["shortening_over_L"], 6), sci(r["w_rel_error"], 1),
                 sci(r["shortening_rel_error"], 1)] for r in d["elastica"]["rows"]]
        rows.append(["reference", num(d["elastica"]["w_ref"], 6), num(d["elastica"]["shortening_ref"], 6), "--", "--"])
        write("tab_elastica.tex", tabular("rrrrr", ["elements", "$w/L$", "shortening$/L$", "error in $w$", "error in shortening"], rows))
        el = d["elastica"]["rows"][-1]
        macro("elasticaN", el["n"])
        macro("elasticaWerr", sci(el["w_rel_error"], 1))
        macro("elasticaShortErr", sci(el["shortening_rel_error"], 1))
        macro("elasticaWref", num(d["elastica"]["w_ref"], 6))
        macro("elasticaShortRef", num(d["elastica"]["shortening_ref"], 6))
        macro("patchMaxErr", sci(max(max(r["displacement_error"], r["stress_error"]) for r in d["patch"]), 0))
        for r in d["pure_bending_quadratic"]:
            pass
        macro("quadBendingMaxErr", sci(max(abs(r["ratio_minus_1"]) for r in d["pure_bending_quadratic"]), 0))

    # ---------------------------------------------------------------- ROM (e1, e3, e4)
    e1, e3, e4 = load("e1_rom_speedup"), load("e3_cms"), load("e4_ecsw")
    if e1:
        d = e1["data"]
        macro("eOneElements", d["n_elements"]); macro("eOneFull", d["n_dof_full"]); macro("eOneReduced", d["n_dof_reduced"])
        macro("eOneErr", sci(d["max_rel_tip_error"], 1)); macro("eOneSpeedup", num(d["speedup"], 0))
        macro("eOneSweep", d["n_sweep"]); macro("eOneTfom", num(d["t_fom_s"] * 1e3, 0)); macro("eOneTrom", num(d["t_rom_s"] * 1e3, 1))
    if e3:
        rows = [[str(r["modes_per_half"]), str(r["reduced_size"]), sci(r["max_rel_error_first4"])]
                for r in e3["data"]["rows"]]
        write("tab_cms.tex", tabular("rrr", ["modes per half", "reduced size", "max rel. freq. error"], rows))
        macro("cmsFullSize", e3["data"]["n_dof_full_free"])
    if e4:
        rows = [[str(r["n_elements"]), sci(r["tol"], 0), f"{r['n_selected']} / {r['n_total']}",
                 sci(r["heldout_force_error_max"]), num(r["force_speedup"], 1), sci(r["traj_rel_error"]),
                 num(r["traj_speedup"], 1)] for r in e4["data"]["rows"]]
        write("tab_ecsw.tex", tabular("rrrrrrr", ["elements", "tol", "kept", "force err.", "force speed-up",
                                                   "trajectory err.", "trajectory speed-up"], rows))

    # ---------------------------------------------------------------- recovery (e2)
    e2 = load("e2_recovery_speedup")
    if e2:
        rows = []
        for r in e2["data"]["rows"]:
            rows.append([r["element"], str(r["n_elements"]), str(r["n_dof"]), num(r["t_batched_s"] * 1e3, 1),
                         num(r["t_loop_s"] * 1e3, 0) if "t_loop_s" in r else "--",
                         num(r["speedup"], 0) if "speedup" in r else "--",
                         sci(r["max_rel_diff"], 0) if "max_rel_diff" in r else "--"])
        write("tab_recovery.tex", tabular("lrrrrrr", ["element", "elements", "DOF", "batched (ms)", "loop (ms)",
                                                      "speed-up", "max rel. diff."], rows))
        sp = [r["speedup"] for r in e2["data"]["rows"] if "speedup" in r]
        macro("recoveryMinSpeedup", num(min(sp), 0)); macro("recoveryMaxSpeedup", num(max(sp), 0))
        per = [r["t_loop_s"] * 1e3 / r["n_elements"] for r in e2["data"]["rows"] if "t_loop_s" in r]
        macro("recoveryLoopMsMin", num(min(per), 2)); macro("recoveryLoopMsMax", num(max(per), 2))
        macro("recoveryLoopMax", e2["data"]["loop_max_elements"])
        big = max(e2["data"]["rows"], key=lambda r: r["n_elements"])
        macro("recoveryBigElements", big["n_elements"]); macro("recoveryBigMs", num(big["t_batched_s"] * 1e3, 0))

    # ---------------------------------------------------------------- torch (e6)
    e6 = load("e6_torch_backend")
    if e6 and not e6["data"].get("skipped"):
        d6, env6 = e6["data"], e6.get("environment", {})
        devs, rows6 = d6["devices"], d6["rows"]
        header = ["free DOF", "SciPy"]
        for dv in devs:
            header += [f"torch CG {dv}", f"torch dense {dv}"]
        rows = []
        for r in rows6:
            row = [str(r["n_free"]), num(r["t_scipy_s"] * 1e3, 1)]
            for dv in devs:
                for m in ("cg", "dense"):
                    k = f"t_torch_{dv}_{m}_s"
                    row.append(num(r[k] * 1e3, 1) if k in r else "--")
            rows.append(row)
        write("tab_torch.tex", tabular("r" * len(header), header, rows))
        header = ["elements", "NumPy"] + [f"torch {dv}" for dv in devs]
        rows = [[str(r["n_elements"]), num(r["t_stress_numpy_s"] * 1e3, 1)] +
                [num(r[f"t_stress_torch_{dv}_s"] * 1e3, 1) for dv in devs] for r in rows6]
        write("tab_torch_stress.tex", tabular("r" * len(header), header, rows))
        macro("torchAvailable", 1); macro("torchVersion", env6.get("torch", "?"))
        macro("torchGpu", env6.get("gpu", "none")); macro("torchCuda", env6.get("torch_cuda_build") or "none")
        macro("torchDriver", env6.get("nvidia_driver") or "n/a"); macro("torchHasGpu", 1 if "cuda" in devs else 0)
        diffs = [v for r in rows6 for k, v in r.items() if k.startswith("rel_diff_") or k.startswith("stress_rel_diff_")]
        macro("torchMaxDiff", sci(max(diffs), 0))
        macro("torchOs", env6.get("platform", "?").split("-")[0]); macro("torchCpus", env6.get("cpu_count", "?"))
        macro("torchPython", env6.get("python", "?")); macro("torchNumpy", env6.get("numpy", "?"))
        macro("torchScipy", env6.get("scipy", "?")); macro("torchCommit", env6.get("git_commit", "?"))

        def cross(key_t, key_ref, dev_key):
            hit = [r["n_free"] if "n_free" in r else r["n_elements"] for r in rows6
                   if dev_key in r and r[dev_key] < r[key_ref]]
            return str(hit[0]) if hit else "none"
        macro("torchCrossCpu", cross(None, "t_scipy_s", "t_torch_cpu_cg_s"))
        macro("torchCrossGpu", cross(None, "t_scipy_s", "t_torch_cuda_cg_s") if "cuda" in devs else "none")
        macro("torchStressCrossCpuElems", next((str(r["n_elements"]) for r in rows6 if r["t_stress_torch_cpu_s"] < r["t_stress_numpy_s"]), "none"))
        macro("torchStressCrossGpuElems", next((str(r["n_elements"]) for r in rows6 if "t_stress_torch_cuda_s" in r and r["t_stress_torch_cuda_s"] < r["t_stress_numpy_s"]), "none") if "cuda" in devs else "none")
        big = rows6[-1]
        macro("torchBigDof", big["n_free"]); macro("torchBigScipyMs", num(big["t_scipy_s"] * 1e3, 1))
        macro("torchBigCpuMs", num(big["t_torch_cpu_cg_s"] * 1e3, 1))
        macro("torchBigCpuRatio", num(big["t_scipy_s"] / big["t_torch_cpu_cg_s"], 2))
        macro("torchBigStressNumpyMs", num(big["t_stress_numpy_s"] * 1e3, 1))
        macro("torchBigStressCpuMs", num(big["t_stress_torch_cpu_s"] * 1e3, 1))
        if "cuda" in devs:
            macro("torchBigGpuMs", num(big["t_torch_cuda_cg_s"] * 1e3, 1))
            macro("torchBigGpuRatio", num(big["t_scipy_s"] / big["t_torch_cuda_cg_s"], 2))
            macro("torchBigStressGpuMs", num(big["t_stress_torch_cuda_s"] * 1e3, 1))
        else:
            for k in ("torchBigGpuMs", "torchBigGpuRatio", "torchBigStressGpuMs"):
                macro(k, "n/a")
    else:
        write("tab_torch.tex", "% torch experiment was skipped on the machine that generated these tables\n")
        write("tab_torch_stress.tex", "% torch experiment was skipped\n")
        macro("torchAvailable", 0)

    # ---------------------------------------------------------------- inventory (e0) and quick start (e7)
    e0, e7 = load("e0_inventory"), load("e7_quickstart")
    if e0:
        d = e0["data"]
        macro("feaLines", f"{d['fea_src']['lines']:,}".replace(",", "\\,")); macro("romLines", f"{d['rom_src']['lines']:,}".replace(",", "\\,"))
        macro("totalLines", f"{d['fea_src']['lines'] + d['rom_src']['lines']:,}".replace(",", "\\,"))
        macro("feaFiles", d["fea_src"]["files"]); macro("romFiles", d["rom_src"]["files"])
        macro("feaTestLines", f"{d['fea_tests_code']['lines']:,}".replace(",", "\\,")); macro("romTestLines", f"{d['rom_tests_code']['lines']:,}".replace(",", "\\,"))
        macro("feaTests", f"{d['fea_tests_collected']:,}".replace(",", "\\,")); macro("romTests", d["rom_tests_collected"])
        macro("nElements", d["elements_registered"]); macro("nConstitutive", d["constitutive_registered"])
        macro("feaPublic", d["fea_public_names"]); macro("romPublic", d["rom_public_names"])
        macro("feaVersion", d["fea_version"]); macro("romVersion", d["rom_version"])
    if e7:
        d = e7["data"]
        macro("qsDof", d["n_dof"]); macro("qsFree", d["n_free"]); macro("qsModes", d["n_modes"])
        macro("qsTip", sci(abs(d["tip_fom"]), 3)); macro("qsErr", sci(d["rel_error"], 0))


    # ---------------------------------------------------------------- extra macros for Sections 10-11
    if e1:
        env = e1.get("environment", {}); d = e1["data"]
        macro("envPython", env.get("python", "?")); macro("envCpus", env.get("cpu_count", "?"))
        macro("envNumpy", env.get("numpy", "?")); macro("envScipy", env.get("scipy", "?"))
        macro("envCommit", env.get("git_commit", "?"))
        macro("eOneMeanErr", sci(d["mean_rel_tip_error"], 1)); macro("eOneNtest", d["n_test"]); macro("eOneNtrain", d["n_train"])
    if e5:
        d = e5["data"]
        macro("ebFirstHz", num(d["eb_cantilever_frequency"]["exact_hz"][0], 3))
        macro("ebRelErrTwenty", sci(d["eb_cantilever_frequency"]["rel_error_n20"][0], 1))
        t = d["timoshenko_tip"]; macro("timoRelErr", sci(t["errors"][-1] / t["exact"], 1)); macro("timoN", len(t["h"]))
        macro("quadBendFine", num(d["pure_bending_linear"]["Quad4PlaneStress"]["values"][-1], 4))
        macro("triBendFine", num(d["pure_bending_linear"]["Tri3PlaneStress"]["values"][-1], 4))
    if e3:
        rows = {str(r["modes_per_half"]): r for r in e3["data"]["rows"]}
        macro("cmsFirstHz", num(e3["data"]["f_full_hz"][0], 1))
        for k, name in (("2", "Two"), ("5", "Five"), ("all", "All")):
            macro(f"cmsErr{name}", sci(rows[k]["max_rel_error_first4"], 1)); macro(f"cmsSize{name}", rows[k]["reduced_size"])
    if e4:
        big = [r for r in e4["data"]["rows"] if r["n_elements"] == 120][-1]
        macro("ecswBigKept", big["n_selected"]); macro("ecswBigTotal", big["n_total"])
        macro("ecswBigForceSpeed", num(big["force_speedup"], 1)); macro("ecswBigTrajSpeed", num(big["traj_speedup"], 1))
        macro("ecswBigTrajErr", sci(big["traj_rel_error"], 1)); macro("ecswBigForceErr", sci(big["heldout_force_error_max"], 1))
        macro("ecswSteps", e4["data"]["rk4_steps"]); macro("ecswModes", e4["data"]["n_modes"])

    e8 = load("e8_workflow")
    if e8:
        d = e8["data"]
        macro("wfFirstHz", num(d["f_hz"][0], 0)); macro("wfVmMax", num(d["vm_max"] / 1e6, 0)); macro("wfErr", sci(d["max_error"], 1))
        macro("wfPassed", "passed" if d["passed"] else "did not pass")

    macro("resultsGenerated", "yes")
    write("numbers.tex", "% generated by make_tables.py -- do not edit\n" + "\n".join(macros) + "\n")


if __name__ == "__main__":
    main()
