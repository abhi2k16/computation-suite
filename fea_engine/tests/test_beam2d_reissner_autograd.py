"""
test_beam2d_reissner_autograd.py -- validates
fea_engine.autograd_tangent.beam2d_reissner_tangent_autograd() (Wave 17
item 140's torch-autograd side-by-side backend, added alongside
Beam2DReissner's own analytic tangent_stiffness()) against that
existing analytic implementation (elements/beams.py). See
autograd_tangent.py's own "Beam2DReissner cross-check" section
docstring for the full derivation/quadrature-scaling reasoning this
file exercises.

torch is an OPTIONAL dependency (see autograd_tangent.py's own
docstring for the _HAS_TORCH/_require_torch() pattern). This file
therefore gates on fea_engine.autograd_tangent._HAS_TORCH itself,
exactly like tests/test_autograd_tangent.py does, so it SKIPS cleanly
(not fails) in an environment where torch isn't usable -- including a
torch wheel that installs but can't actually import (see that file's
own docstring for why a bare pytest.importorskip("torch") does not
handle that case).

Two checks, mirroring test_beam2d_reissner.py's own CHECK (a)/(b)
(reusing the SAME random-large-rotation state generation as CHECK (a)
so both checks validate the same state space):

1. torch-autograd internal_force()/tangent_stiffness() vs. the
   existing analytic implementation, at several random large-rotation
   states. Autograd vs. an EXACT analytic derivation (not vs. finite
   differences), so this should be much tighter than CHECK (a)'s own
   1e-6 FD bar -- target machine precision (~1e-10 to 1e-14), matching
   test_autograd_tangent.py's own Tet10SolidTL-vs-analytic agreement
   (~1e-16).
2. Objectivity: the torch-autograd path also gives zero internal force
   under a pure rigid-body rotation (mirrors CHECK (b) -- a sanity
   check that the torch strain-energy expression above is genuinely
   frame-invariant, not just numerically close by coincidence).
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from fea_engine import elements as elmod
from fea_engine.autograd_tangent import _HAS_TORCH

pytestmark = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not usable in this environment (not installed, or installed "
           "but failing to import -- see autograd_tangent.py's docstring); "
           "these tests validate a torch-based cross-check and have nothing "
           "to run without it.")

if _HAS_TORCH:
    from fea_engine.autograd_tangent import beam2d_reissner_tangent_autograd


def test_beam2d_reissner_autograd_matches_analytic_at_large_rotations():
    print("=" * 70)
    print("CHECK 1: torch-autograd internal_force()/tangent_stiffness()")
    print("vs. Beam2DReissner's own analytic implementation, at RANDOM")
    print("LARGE-ROTATION states (same state generation as")
    print("test_beam2d_reissner.py's own CHECK (a))")
    print("=" * 70)
    beam = elmod.Beam2DReissner()
    E, G, A, I, kappa_s = 210e9, 80e9, 1.0e-3, 1.0e-7, 1.0
    mat = (E, G, A, I, kappa_s)
    rng = np.random.default_rng(42)

    max_f_rel = 0.0
    max_k_rel = 0.0
    for trial in range(12):
        ec = np.array([[0.0, 0.0], [1.5, 0.9]]) + 0.1 * rng.standard_normal((2, 2))
        u = rng.standard_normal(6)
        u[0:2] *= 0.05; u[3:5] *= 0.05
        u[2] = rng.uniform(-3.0, 3.0)
        u[5] = rng.uniform(-3.0, 3.0)

        f_ref = beam.internal_force(ec, u, mat)
        K_ref = beam.tangent_stiffness(ec, u, mat)

        K_ad, f_ad = beam2d_reissner_tangent_autograd(ec, u, mat)

        f_err = np.max(np.abs(f_ad - f_ref))
        f_rel = f_err / max(np.max(np.abs(f_ref)), 1e-30)
        k_err = np.max(np.abs(K_ad - K_ref))
        k_rel = k_err / max(np.max(np.abs(K_ref)), 1e-30)
        max_f_rel = max(max_f_rel, f_rel)
        max_k_rel = max(max_k_rel, k_rel)
        print(f"  trial {trial:2d}: theta1={u[2]:+.3f} theta2={u[5]:+.3f}  "
              f"f_rel={f_rel:.3e}  K_rel={k_rel:.3e}")

    print(f"  max relative error (f_int): {max_f_rel:.3e}")
    print(f"  max relative error (K_T):   {max_k_rel:.3e}")
    assert max_f_rel < 1e-10, "torch-autograd internal_force does not match analytic"
    assert max_k_rel < 1e-10, "torch-autograd tangent does not match analytic"
    print("  PASS -- autograd (exact) tangent/force agree with the existing"
          " analytic implementation to near machine precision")


def test_beam2d_reissner_autograd_objectivity_rigid_rotation_gives_zero_force():
    print("=" * 70)
    print("CHECK 2: objectivity -- the torch-autograd path also gives")
    print("EXACTLY zero internal force under a pure rigid-body rotation")
    print("(mirrors test_beam2d_reissner.py's own CHECK (b))")
    print("=" * 70)
    E, G, A, I, kappa_s = 210e9, 80e9, 1.0e-3, 1.0e-7, 1.0
    mat = (E, G, A, I, kappa_s)
    rng = np.random.default_rng(7)

    max_f = 0.0
    for trial in range(10):
        ec = np.array([[0.0, 0.0], [1.7, 0.0]]) + 0.1 * rng.standard_normal((2, 2))
        X1, X2 = ec
        theta_rigid = rng.uniform(-3.0, 3.0)
        trans = rng.standard_normal(2)
        c, s = np.cos(theta_rigid), np.sin(theta_rigid)
        R = np.array([[c, -s], [s, c]])
        X1r = X1 + trans
        X2r = X1r + R @ (X2 - X1)
        u_rigid = np.concatenate([X1r - X1, [theta_rigid], X2r - X2, [theta_rigid]])

        _, f_ad = beam2d_reissner_tangent_autograd(ec, u_rigid, mat)
        max_f = max(max_f, np.max(np.abs(f_ad)))
        print(f"  trial {trial}: theta_rigid={theta_rigid:+.3f} rad  "
              f"max|internal_force|={np.max(np.abs(f_ad)):.3e}")

    # Same threshold/reasoning as test_beam2d_reissner.py's own CHECK (b) --
    # floating-point roundoff at this element's EA ~ 2e8 N stiffness scale,
    # not residual strain (CHECK 1 above already isolates strain-measure
    # correctness independently).
    assert max_f < 1e-6, "rigid rotation produces spurious internal force (torch path)"
    print(f"  PASS -- max|f| = {max_f:.3e} (floating-point roundoff only)")
