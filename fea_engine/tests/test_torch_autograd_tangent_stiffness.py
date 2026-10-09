"""
test_torch_autograd_tangent_stiffness.py -- validation for item 93 of the
PyTorch side-by-side extension (docs/consolidated_future_roadmap.md): the
opt-in method="autograd" path wired into Tet4NeoHookean.tangent_stiffness()
(elements/nonlinear_solids.py), which routes through the already-validated
autograd_tangent.tet4_neo_hookean_tangent_autograd() instead of the default
central-finite-difference tangent.

This is a PURE WIRING check, not a re-derivation of autograd_tangent.py's
own correctness -- that module's own test_autograd_tangent.py already
validates tet4_neo_hookean_tangent_autograd() against internal_force() and
finite difference in isolation. What's new here is only:

1. test_autograd_method_matches_direct_call -- method="autograd" on the
   ELEMENT returns EXACTLY (bit-identical) what calling
   tet4_neo_hookean_tangent_autograd() directly returns -- confirms the
   plumbing (deferred import, arg forwarding) introduces no transformation
   of its own.
2. test_fd_default_unchanged -- method="fd" (the default, both explicit
   and implicit) is completely unaffected by this edit -- a regression
   check that adding the method= parameter didn't perturb the existing
   finite-difference path's numerics AT ALL.
3. test_unknown_method_raises -- an unrecognized method value fails loudly
   (ValueError) rather than silently falling through to one of the two
   known paths.
4. test_autograd_matches_fd_tangent -- the two methods agree with each
   other on a genuinely deformed (non-trivial-F) state, to FD's own
   truncation-error floor (~1e-6 relative) -- the actual "these two
   independent derivations of the same physics agree" cross-check.
5. test_autograd_vs_fd_newton_convergence -- END TO END: run
   nonlinear_solver.solve_nonlinear_static() to the SAME target load with
   method="fd" vs method="autograd" (forwarded through FESystem's
   assemble_tangent_stiffness()'s **kwargs plumbing -- see solver.py) and
   confirm both converge to the same equilibrium displacement field. This
   is the test that matters most: it's not enough for the two tangents to
   numerically agree in isolation, Newton's iteration itself must actually
   converge using the autograd tangent on a real (if small) mesh.

SANDBOX NOTE (2026-09-11): torch is not installed/importable in the sandbox
this file was authored in (pip install times out / is blocked here -- see
torch_sparse_solver.py's own prior documented finding). Every test in this
file is skipped cleanly via _HAS_TORCH (mirroring test_autograd_tangent.py's
own gating, NOT pytest.importorskip, since a CUDA-linked wheel can raise
ValueError/OSError at import time rather than ImportError) rather than
pytest.importorskip.

VALIDATED (2026-09-11, on the user's own machine -- Windows, conda base
env, Python 3.10.18, PyTorch installed): all 5 tests in this file PASSED
for real (`pytest tests/test_torch_autograd_tangent_stiffness.py -v`),
alongside test_torch_transient_backend.py, test_torch_shell_autograd_
tangent.py, test_autograd_tangent.py, and test_torch_sparse_solver.py in
the same run -- 30 passed, 1 benign warning, 0 failed. This is the same
sandbox-then-user-machine validation path items 2/3 (autograd_tangent.py/
torch_sparse_solver.py) already established.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from fea_engine import Tet4NeoHookean, NeoHookeanMaterial, FESystem
from fea_engine.mesh import Mesh
from fea_engine import nonlinear_solver as nls

try:
    import torch  # noqa: F401
    from fea_engine.autograd_tangent import tet4_neo_hookean_tangent_autograd
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")

E, NU = 1.0e6, 0.4   # rubber-like, matching test_hyperelastic.py's convention


def _single_tet():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3]])
    mesh = Mesh(nodes=coords, elements=elements, dim=3)
    return mesh, Tet4NeoHookean()


def _random_displacement(scale=0.15, seed=0):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal(12)


def test_autograd_method_matches_direct_call():
    mesh, elem = _single_tet()
    mat = NeoHookeanMaterial(E=E, nu=NU)
    u = _random_displacement()

    K_via_element = elem.tangent_stiffness(mesh.nodes, u, mat, method="autograd")
    K_direct, _ = tet4_neo_hookean_tangent_autograd(mesh.nodes, u, mat)

    assert np.array_equal(K_via_element, K_direct)


def test_fd_default_unchanged():
    mesh, elem = _single_tet()
    mat = NeoHookeanMaterial(E=E, nu=NU)
    u = _random_displacement()

    K_implicit = elem.tangent_stiffness(mesh.nodes, u, mat)
    K_explicit_fd = elem.tangent_stiffness(mesh.nodes, u, mat, method="fd")

    assert np.array_equal(K_implicit, K_explicit_fd)
    assert np.allclose(K_implicit, K_implicit.T)


def test_unknown_method_raises():
    mesh, elem = _single_tet()
    mat = NeoHookeanMaterial(E=E, nu=NU)
    u = _random_displacement()

    with pytest.raises(ValueError):
        elem.tangent_stiffness(mesh.nodes, u, mat, method="bogus")


def test_autograd_matches_fd_tangent():
    mesh, elem = _single_tet()
    mat = NeoHookeanMaterial(E=E, nu=NU)
    u = _random_displacement(scale=0.2, seed=1)

    K_fd = elem.tangent_stiffness(mesh.nodes, u, mat, method="fd")
    K_ad = elem.tangent_stiffness(mesh.nodes, u, mat, method="autograd")

    assert np.allclose(K_ad, K_ad.T, atol=1e-6 * np.abs(K_ad).max())
    rel_err = np.max(np.abs(K_ad - K_fd)) / np.max(np.abs(K_fd))
    assert rel_err < 1e-5


def test_autograd_vs_fd_newton_convergence():
    mat = NeoHookeanMaterial(E=E, nu=NU)
    F_applied_total = np.array([5e4, 3e4, -2e4])
    n_steps = 50

    results = {}
    for method in ("fd", "autograd"):
        mesh, elem = _single_tet()
        fs = FESystem(mesh, elem, sparse=False)
        fs.fix_dofs([0, 1, 2], [0, 1, 2])
        fs.F[9:12] = F_applied_total
        kwargs = {} if method == "fd" else {"method": "autograd"}
        _, U_hist = nls.solve_nonlinear_static(fs, mat, n_steps=n_steps, tol=1e-10,
                                                max_iter=60, **kwargs)
        results[method] = U_hist[-1]

    assert np.allclose(results["fd"], results["autograd"], rtol=1e-5, atol=1e-9)
