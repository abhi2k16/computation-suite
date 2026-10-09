# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_backend_selection.py -- validates the backend= construction
parameter on FESystem (solver.py) added on top of Wave 0 item 3
(torch_sparse_solver.py): the explicit, user-facing choice between the
SciPy solve path (backend="scipy", default) and the PyTorch solve path
(backend="torch"), requested so a caller never has to separately import
and call fesystem_solve_static_torch() -- solve_static() itself
dispatches based on what was chosen at construction time.

This file only covers what's runnable WITHOUT torch installed (the
argument-validation/fail-fast contract) -- the actual numerical
agreement between backend="scipy" and backend="torch" on a real model
is covered by test_torch_sparse_solver.py's CHECK 6/CHECK 7, which are
torch-gated the same way every other torch test in this package is
(see that file's own docstring for why).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import pytest
from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, mesh
from fea_engine.torch_sparse_solver import _HAS_TORCH


def _build_system(**kwargs):
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    D2 = D_plane_stress(mat)
    m = mesh.rectangle_mesh(2.0, 1.0, 4, 2)
    return FESystem(m, Quad4PlaneStress(), thickness=1.0, **kwargs)


def test_default_backend_is_scipy():
    print("=" * 70)
    print("CHECK 1: FESystem() with no backend= argument defaults to")
    print("backend='scipy' -- zero behavior change for every existing")
    print("caller in this package")
    print("=" * 70)
    sysobj = _build_system()
    assert sysobj.backend == "scipy"
    print("  PASS")


def test_unknown_backend_raises_value_error():
    print()
    print("=" * 70)
    print("CHECK 2: an unrecognized backend= value raises ValueError at")
    print("construction time, not a confusing failure later inside")
    print("solve_static()")
    print("=" * 70)
    with pytest.raises(ValueError, match="backend"):
        _build_system(backend="numpy")
    print("  PASS")


@pytest.mark.skipif(
    _HAS_TORCH,
    reason="this checks the FAIL-FAST behavior when torch is unavailable; "
           "not meaningful in an environment where torch actually imports.")
def test_backend_torch_without_torch_installed_raises_at_construction():
    print()
    print("=" * 70)
    print("CHECK 3: backend='torch' fails FAST at FESystem construction")
    print("(not deferred to solve_static()) when torch isn't usable in")
    print("this environment -- same convention as UnifiedGeometryEngine's")
    print("_require_gmsh() in geometry/gmsh_engine.py")
    print("=" * 70)
    with pytest.raises(ImportError):
        _build_system(backend="torch")
    print("  PASS")


def test_backend_and_device_are_stored_on_the_instance():
    print()
    print("=" * 70)
    print("CHECK 4: backend= and device= are stored as plain attributes")
    print("(no hidden state), so a caller/test can introspect what a")
    print("given FESystem instance will actually do on solve_static()")
    print("=" * 70)
    sysobj = _build_system(backend="scipy", device="cpu")
    assert sysobj.backend == "scipy"
    assert sysobj.device == "cpu"
    print("  PASS")
