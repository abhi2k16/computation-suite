# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_autograd_tangent.py -- Wave 0 item 2 (docs/consolidated_future_
roadmap.md, source tensormesh_comparative_analysis.md Section 6.3):
validates fea_engine.autograd_tangent's independent torch.autograd
cross-checks against the existing Tet4NeoHookean (finite-difference)
and Tet10SolidTL (analytic) element tangents.

torch is an OPTIONAL dependency (see autograd_tangent.py's own
docstring for the _HAS_TORCH/_require_torch() pattern, mirroring
gmsh's). This file must therefore SKIP cleanly, not fail, wherever
torch isn't usable -- which, empirically (see autograd_tangent.py's
comment on this), includes an environment where a torch WHEEL
installed successfully but can't actually be imported (e.g. a CUDA-
linked wheel with no CUDA runtime present), a case plain
`pytest.importorskip("torch")` does NOT handle gracefully (it only
catches ImportError, and the CUDA-linking failure surfaces as
ValueError/OSError instead -- confirmed directly while building this
module). So this file gates on fea_engine.autograd_tangent._HAS_TORCH
itself (the SAME robust any-exception detection the module's own
_require_torch() guard relies on) rather than a bare importorskip.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import (
    Material, NeoHookeanMaterial, Tet4NeoHookean, Tet10SolidTL, mesh,
)
from fea_engine.autograd_tangent import _HAS_TORCH

pytestmark = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch not usable in this environment (not installed, or installed "
           "but failing to import -- see autograd_tangent.py's docstring); "
           "these tests validate torch-based cross-checks and have nothing "
           "to run without it.")

if _HAS_TORCH:
    from fea_engine.autograd_tangent import (
        tet4_neo_hookean_tangent_autograd, tet10_solid_tl_tangent_autograd,
        tet4_neo_hookean_tangent_autograd_batched,
    )


# A single non-degenerate reference tet, reused across checks.
_TET4_COORDS = np.array([
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])

_TET10_COORDS = np.array([
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (0.5, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)])


def _random_displacement(n_dof, scale, seed):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal(n_dof)


def test_tet4_neo_hookean_autograd_matches_internal_force():
    print("=" * 70)
    print("CHECK 1: torch-autograd internal_force() matches Tet4NeoHookean's")
    print("own closed-form internal_force() at a nonzero displacement")
    print("=" * 70)
    mat = NeoHookeanMaterial(E=1e6, nu=0.3)
    elem = Tet4NeoHookean()
    u = _random_displacement(12, 0.05, seed=0)

    f_ref = elem.internal_force(_TET4_COORDS, u, mat)
    K_ad, f_ad = tet4_neo_hookean_tangent_autograd(_TET4_COORDS, u, mat)

    err = np.max(np.abs(f_ad - f_ref))
    rel = err / max(np.max(np.abs(f_ref)), 1e-30)
    print(f"  max abs diff = {err:.3e}, max rel diff = {rel:.3e}")
    assert rel < 1e-10
    print("  PASS -- the torch re-expression of neo_hookean_pk2_stress -> f_int"
          " reproduces the numpy closed form to near machine precision")


def test_tet4_neo_hookean_autograd_tangent_matches_finite_difference():
    print()
    print("=" * 70)
    print("CHECK 2: torch-autograd tangent matches Tet4NeoHookean's own")
    print("central-finite-difference tangent_stiffness()")
    print("=" * 70)
    mat = NeoHookeanMaterial(E=1e6, nu=0.3)
    elem = Tet4NeoHookean()
    u = _random_displacement(12, 0.05, seed=1)

    K_fd = elem.tangent_stiffness(_TET4_COORDS, u, mat)
    K_ad, _ = tet4_neo_hookean_tangent_autograd(_TET4_COORDS, u, mat)

    err = np.max(np.abs(K_ad - K_fd))
    rel = err / max(np.max(np.abs(K_fd)), 1e-30)
    print(f"  max abs diff = {err:.3e}, max rel diff = {rel:.3e}")
    # FD tangent has real O(h^2) truncation error (h=1e-6 default) --
    # a looser tolerance than the machine-precision internal_force()
    # check above is expected and correct, not a red flag.
    assert rel < 1e-5
    print("  PASS -- autograd (exact) tangent agrees with the existing FD"
          " tangent well within FD's own expected truncation error")

    print()
    print("  Sub-check: the autograd tangent is symmetric to near machine")
    print("  precision (it is the exact Hessian of a scalar strain-energy")
    print("  potential -- unlike the FD tangent, it needs no explicit")
    print("  symmetrization, so checking this IS a meaningful diagnostic).")
    asym = np.max(np.abs(K_ad - K_ad.T))
    scale = np.max(np.abs(K_ad))
    print(f"  max asymmetry = {asym:.3e} (relative to scale {scale:.3e})")
    assert asym < 1e-8 * scale


def test_tet4_neo_hookean_autograd_tangent_zero_displacement():
    print()
    print("=" * 70)
    print("CHECK 3: at zero displacement, the autograd tangent matches")
    print("Tet4NeoHookean.stiffness() (the initial tangent)")
    print("=" * 70)
    mat = NeoHookeanMaterial(E=1e6, nu=0.3)
    elem = Tet4NeoHookean()
    u0 = np.zeros(12)

    K0_ref = elem.stiffness(_TET4_COORDS, mat)
    K0_ad, f0_ad = tet4_neo_hookean_tangent_autograd(_TET4_COORDS, u0, mat)

    print(f"  max |f_int at u=0| = {np.max(np.abs(f0_ad)):.3e} (expect ~0)")
    assert np.max(np.abs(f0_ad)) < 1e-8

    rel = np.max(np.abs(K0_ad - K0_ref)) / max(np.max(np.abs(K0_ref)), 1e-30)
    print(f"  max rel diff (K0) = {rel:.3e}")
    assert rel < 1e-5
    print("  PASS")


def test_tet10_solid_tl_autograd_matches_internal_force():
    print()
    print("=" * 70)
    print("CHECK 4: torch-autograd internal_force() matches Tet10SolidTL's")
    print("own closed-form internal_force()")
    print("=" * 70)
    mat = Material(E=1e6, nu=0.3)
    elem = Tet10SolidTL()
    u = _random_displacement(30, 0.02, seed=2)

    f_ref = elem.internal_force(_TET10_COORDS, u, mat)
    K_ad, f_ad = tet10_solid_tl_tangent_autograd(_TET10_COORDS, u, mat)

    err = np.max(np.abs(f_ad - f_ref))
    rel = err / max(np.max(np.abs(f_ref)), 1e-30)
    print(f"  max abs diff = {err:.3e}, max rel diff = {rel:.3e}")
    assert rel < 1e-10
    print("  PASS")


def test_tet10_solid_tl_autograd_tangent_matches_analytic():
    print()
    print("=" * 70)
    print("CHECK 5: torch-autograd tangent matches Tet10SolidTL's ANALYTIC")
    print("tangent_stiffness() (the current default, added 2026-09-02)")
    print("=" * 70)
    mat = Material(E=1e6, nu=0.3)
    elem = Tet10SolidTL()
    u = _random_displacement(30, 0.02, seed=3)

    K_analytic = elem.tangent_stiffness(_TET10_COORDS, u, mat)
    K_ad, _ = tet10_solid_tl_tangent_autograd(_TET10_COORDS, u, mat)

    err = np.max(np.abs(K_ad - K_analytic))
    rel = err / max(np.max(np.abs(K_analytic)), 1e-30)
    print(f"  max abs diff = {err:.3e}, max rel diff = {rel:.3e}")
    assert rel < 1e-8
    print("  PASS -- analytic and autograd tangents agree to near machine"
          " precision (both are exact, unlike the FD case above)")


def test_tet10_solid_tl_autograd_tangent_matches_complex_step_reference():
    print()
    print("=" * 70)
    print("CHECK 6: torch-autograd tangent ALSO matches the complex-step")
    print("reference tangent Tet10SolidTL keeps around for exactly this")
    print("kind of independent validation -- three independently")
    print("implemented methods (analytic, complex-step, autograd) all")
    print("agreeing is strong evidence, not a coincidence")
    print("=" * 70)
    mat = Material(E=1e6, nu=0.3)
    elem = Tet10SolidTL()
    u = _random_displacement(30, 0.02, seed=4)

    K_cs = elem._tangent_stiffness_complex_step(_TET10_COORDS, u, mat)
    K_ad, _ = tet10_solid_tl_tangent_autograd(_TET10_COORDS, u, mat)

    err = np.max(np.abs(K_ad - K_cs))
    rel = err / max(np.max(np.abs(K_cs)), 1e-30)
    print(f"  max abs diff = {err:.3e}, max rel diff = {rel:.3e}")
    assert rel < 1e-8
    print("  PASS")


# =====================================================================
# Wave 10 item 102: batched (vmap) multi-element autograd tangent
# assembly -- validates tet4_neo_hookean_tangent_autograd_batched()
# against calling the per-element tet4_neo_hookean_tangent_autograd()
# once per element in a plain Python loop (this file's own existing
# function, unmodified) on a random multi-element batch. This is the
# whole correctness claim item 102 makes: batching only changes HOW
# the computation is scheduled, never the physics or the numbers.
# =====================================================================
class TestTet4NeoHookeanBatchedAutograd:
    def _random_batch(self, n_elem, seed):
        rng = np.random.default_rng(seed)
        base = _TET4_COORDS
        coords_batch = np.zeros((n_elem, 4, 3))
        u_batch = np.zeros((n_elem, 12))
        for e in range(n_elem):
            # small random perturbation of the reference tet's own
            # nodes so every element in the batch is a genuinely
            # DIFFERENT (still non-degenerate) geometry, not n_elem
            # copies of the same one -- a batching bug that only shows
            # up with heterogeneous per-element geometry (e.g.
            # accidentally broadcasting element 0's Jacobian to every
            # element) would otherwise go undetected.
            coords_batch[e] = base + 0.05 * rng.standard_normal((4, 3))
            u_batch[e] = _random_displacement(12, 0.05, seed=100 + e + seed)
        return coords_batch, u_batch

    def test_batched_matches_looped_per_element_forces_and_tangents(self):
        mat = NeoHookeanMaterial(E=1e6, nu=0.3)
        n_elem = 6
        coords_batch, u_batch = self._random_batch(n_elem, seed=42)

        K_batch, f_batch = tet4_neo_hookean_tangent_autograd_batched(coords_batch, u_batch, mat)
        assert K_batch.shape == (n_elem, 12, 12)
        assert f_batch.shape == (n_elem, 12)

        for e in range(n_elem):
            K_loop, f_loop = tet4_neo_hookean_tangent_autograd(coords_batch[e], u_batch[e], mat)
            f_err = np.max(np.abs(f_batch[e] - f_loop))
            f_rel = f_err / max(np.max(np.abs(f_loop)), 1e-30)
            K_err = np.max(np.abs(K_batch[e] - K_loop))
            K_rel = K_err / max(np.max(np.abs(K_loop)), 1e-30)
            assert f_rel < 1e-10, f"element {e}: force mismatch rel={f_rel:.3e}"
            assert K_rel < 1e-10, f"element {e}: tangent mismatch rel={K_rel:.3e}"

    def test_batched_single_element_matches_unbatched(self):
        # n_elem=1 edge case -- vmap's own batching machinery still
        # applies, just over a trivial batch dimension.
        mat = NeoHookeanMaterial(E=1e6, nu=0.3)
        coords_batch, u_batch = self._random_batch(1, seed=7)
        K_batch, f_batch = tet4_neo_hookean_tangent_autograd_batched(coords_batch, u_batch, mat)
        K_loop, f_loop = tet4_neo_hookean_tangent_autograd(coords_batch[0], u_batch[0], mat)
        assert np.max(np.abs(K_batch[0] - K_loop)) / np.max(np.abs(K_loop)) < 1e-10
        assert np.max(np.abs(f_batch[0] - f_loop)) / np.max(np.abs(f_loop)) < 1e-10
