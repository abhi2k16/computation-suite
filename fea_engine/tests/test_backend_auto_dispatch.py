# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_backend_auto_dispatch.py -- Wave 11 item 108 (docs/consolidated_
future_roadmap.md): validates backend="auto" (backend_dispatch.py) --
the selection rule itself (SPD/symmetry probes, DOF-count threshold),
the verbose=True diagnostic line, and an end-to-end FESystem round trip
confirming backend="auto" produces the SAME answer backend="scipy"
would (the only backend actually runnable in this sandbox, since torch
isn't installed here -- see backend_dispatch.py's own docstring for why
"auto" never needs torch at construction time).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import Quad4PlaneStress, FESystem, Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh
from fea_engine.backend_dispatch import select_backend, format_diagnostic_line, AUTO_DOF_THRESHOLD


def _steel():
    return Material(E=2.1e11, nu=0.3, rho=7850.0)


def test_select_backend_small_spd_picks_scipy():
    K = np.array([[4.0, 1.0], [1.0, 3.0]])   # SPD, tiny
    backend, diag = select_backend(K)
    assert backend == "scipy"
    assert diag["symmetric"] is True
    assert diag["spd"] is True
    assert diag["n"] == 2


def test_select_backend_large_spd_picks_torch_with_low_threshold():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((10, 10))
    K = A @ A.T + 10 * np.eye(10)   # guaranteed SPD
    backend, diag = select_backend(K, dof_threshold=5)   # force "large" relative to threshold
    assert backend == "torch"
    assert diag["spd"] is True


def test_select_backend_nonsymmetric_never_picks_torch_regardless_of_size():
    rng = np.random.default_rng(1)
    K = rng.standard_normal((10, 10))   # generic, not symmetric
    backend, diag = select_backend(K, dof_threshold=0)   # threshold forced to "always big enough"
    assert backend == "scipy"
    assert diag["symmetric"] is False
    assert diag["spd"] is False


def test_select_backend_symmetric_but_indefinite_never_picks_torch():
    K = np.array([[1.0, 2.0], [2.0, 1.0]])   # symmetric, indefinite (eigs +3,-1)
    backend, diag = select_backend(K, dof_threshold=0)
    assert backend == "scipy"
    assert diag["symmetric"] is True
    assert diag["spd"] is False


def test_select_backend_sparse_input():
    from scipy.sparse import csr_matrix
    K = csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    backend, diag = select_backend(K)
    assert backend == "scipy"
    assert diag["nnz"] == K.nnz


def test_format_diagnostic_line_contains_all_fields():
    _, diag = select_backend(np.array([[4.0, 1.0], [1.0, 3.0]]))
    line = format_diagnostic_line(diag)
    for key in ("n=", "nnz=", "dtype=", "device=", "symmetric=", "spd=", "backend=", "method="):
        assert key in line


def test_fesystem_backend_auto_construction_does_not_require_torch():
    # Should not raise even though torch isn't importable in this
    # sandbox -- backend="auto" is only resolved inside solve_static().
    mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
    fs = FESystem(mesh, Quad4PlaneStress(), backend="auto")
    assert fs.backend == "auto"


def test_fesystem_backend_auto_matches_scipy_end_to_end(capsys):
    mesh = rectangle_mesh(Lx=3.0, Ly=1.0, nx=6, ny=3)
    D = D_plane_stress(_steel())
    left = mesh.nodes_on_line(axis=0, value=0.0)
    tip = mesh.nodes_on_line(axis=0, value=3.0)

    def _build(backend):
        fs = FESystem(mesh, Quad4PlaneStress(), backend=backend)
        fs.assemble_stiffness(D)
        for n in left:
            fs.fix_dofs([n], [0, 1])
        for n in tip:
            fs.F[fs._global_dofs([n])[1]] += -500.0 / len(tip)
        return fs

    fs_scipy = _build("scipy")
    U_scipy = fs_scipy.solve_static()

    fs_auto = _build("auto")
    U_auto = fs_auto.solve_static(verbose=True)
    captured = capsys.readouterr()
    assert "backend=scipy" in captured.out   # small system -> below AUTO_DOF_THRESHOLD -> scipy
    assert np.allclose(U_scipy, U_auto, atol=1e-9, rtol=1e-8)


def test_unknown_backend_still_rejected():
    mesh = rectangle_mesh(Lx=1.0, Ly=1.0, nx=1, ny=1)
    with pytest.raises(ValueError, match="unknown backend"):
        FESystem(mesh, Quad4PlaneStress(), backend="bogus")
