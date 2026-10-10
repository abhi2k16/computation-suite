# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
make_arxiv.py -- build a flat, self-contained arXiv source package from ``paper/``.

    python paper/make_arxiv.py            # writes paper/arxiv_submission/ and paper/arxiv_submission.tar.gz

What it does
  1. copies main.tex, macros.tex, sec*.tex, the tables, the PDF figures and the code listings into ONE flat directory;
  2. rewrites ``tables/``, ``figures/`` and ``listings/`` paths so every file is found in that directory;
  3. compiles (pdflatex, bibtex, pdflatex x2) and keeps the generated ``main.bbl`` (arXiv does not run BibTeX on its own
     .bib files reliably, so the .bbl is shipped and refs.bib is not);
  4. runs the checks of ARXIV_CHECKLIST.md that can be automated and prints a report;
  5. archives the sources (no PDF, no aux files) as a .tar.gz.

Run it on a machine with a full TeX Live (the ``algorithm`` and ``algpseudocode`` packages) for the final build.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
FINAL = PAPER / "arxiv_submission"
OUT = Path(tempfile.mkdtemp(prefix="arxiv_build_"))   # built outside the project: a PDF open in a viewer cannot block it
MAX_MB = 50.0


def sh(cmd, cwd, env=None):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def flat_copy():
    for f in ["main.tex", "macros.tex"] + sorted(p.name for p in PAPER.glob("sec*.tex")):
        text = (PAPER / f).read_text(encoding="utf-8")
        text = re.sub(r"(\\input\{)tables/", r"\1", text)
        text = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{)figures/", r"\1", text)
        text = re.sub(r"(\\lstinputlisting(?:\[[^\]]*\])?\{)listings/", r"\1", text)
        # \lstinputlisting[caption={...}]{...} keeps its optional argument; the path regex above only matches
        # when no ']' appears before '{'.  Handle the caption form explicitly:
        text = re.sub(r"\}\{listings/", "}{", text)
        (OUT / f).write_text(text, encoding="utf-8")
    used = set()
    for f in OUT.glob("*.tex"):
        used |= set(re.findall(r"\\input\{([^}]+)\}", f.read_text()))
    for name in sorted(used):
        src = PAPER / "tables" / (name + ".tex")
        if src.exists():
            shutil.copy(src, OUT / src.name)
    for pdf in sorted((PAPER / "figures").glob("*.pdf")):
        text = " ".join(f.read_text() for f in OUT.glob("*.tex"))
        if pdf.name in text:
            shutil.copy(pdf, OUT / pdf.name)
    shutil.copy(PAPER / "refs.bib", OUT / "refs.bib")      # needed to build main.bbl; not archived
    for py in sorted((PAPER / "listings").glob("*.py")):
        shutil.copy(py, OUT / py.name)


def compile_all():
    env = dict(os.environ)
    stubs = Path("/tmp/texstubs")
    if stubs.exists():       # local preview only: stand-ins when algorithm.sty / algpseudocode.sty are missing
        env["TEXINPUTS"] = f"{stubs}:" + env.get("TEXINPUTS", "")
    log = []
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main"], ["pdflatex", "-interaction=nonstopmode", "main"]):
        code, out = sh(cmd, OUT, env)
        log.append((cmd[0], code))
    return log


def checks():
    rep = []
    tex = {f.name: f.read_text() for f in OUT.glob("*.tex")}
    logt = (OUT / "main.log").read_text(errors="replace")
    aux_bad = [l for l in logt.splitlines() if l.startswith("!")]
    rep.append(("LaTeX errors", len(aux_bad), 0))
    rep.append(("undefined references/citations", len(re.findall(r"undefined", logt)), 0))
    rep.append(("multiply-defined labels", len(re.findall(r"multiply defined", logt)), 0))
    rep.append(("overfull boxes > 10pt", len([m for m in re.findall(r"Overfull \\hbox \(([\d.]+)pt", logt) if float(m) > 10]), 0))
    rep.append(("TODO markers in the compiled PDF", subprocess.run(["pdftotext", "main.pdf", "-"], cwd=OUT, capture_output=True, text=True).stdout.count("TODO:"), 0))
    rep.append(("placeholders [..to be supplied]", sum(t.count("to be supplied") for t in tex.values()), 0))
    rep.append((".bbl present", int((OUT / "main.bbl").exists()), 1))
    rep.append(("absolute paths in .tex", sum(len(re.findall(r"\b[A-Za-z]:\\(?:Users|Windows|Program)|/home/|/sessions/|/Users/", t)) for t in tex.values()), 0))
    names = [p.name for p in OUT.iterdir()]
    rep.append(("non-ASCII or spaced file names", len([n for n in names if not re.fullmatch(r"[A-Za-z0-9._-]+", n)]), 0))
    size = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
    rep.append((f"source size {size:.2f} MB (limit {MAX_MB:g})", int(size > MAX_MB), 0))
    return rep


def archive():
    keep = [p for p in sorted(OUT.iterdir()) if p.suffix in {".tex", ".bbl", ".pdf", ".py", ".sty", ".bst"}
            and p.name != "main.pdf"]
    tgz = PAPER / "arxiv_submission.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        for p in keep:
            t.add(p, arcname=p.name)
    return tgz, [p.name for p in keep]


def main():
    flat_copy()
    log = compile_all()
    print("compile steps:", log)
    rep = checks()
    ok = True
    for name, got, want in rep:
        good = got == want
        ok &= good
        print(f"[{'ok' if good else 'FIX'}] {name}: {got} (want {want})")
    tgz, files = archive()
    FINAL.mkdir(exist_ok=True)
    for f in sorted(OUT.iterdir()):
        try:
            shutil.copy(f, FINAL / f.name)
        except OSError as exc:
            print(f"could not update {f.name} in arxiv_submission/ ({exc}); close it if it is open in a viewer")
    print(f"archive {tgz.name}: {len(files)} files")
    print("files:", " ".join(files))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
