# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_singular_system.py -- Wave 0 item 6 (docs/consolidated_future_
roadmap.md, source fem_implementation_lessons.md): validates
FESystem._singular_regularized_solve() / the automatic fallback
solve_static() now takes for a genuinely SINGULAR free-dof stiffness
matrix, i.e. a pure-Neumann / floating-structure model with no
Dirichlet boundary condition removing rigid-body motion.

Two models, both built the same way as test_sparse_assembly.py's
_build_system() (a plain rectangle_mesh() cantilever) but with ZERO
fixed dofs:

1. A physically well-posed one: equal-and-opposite tension pulling the
   left and right edges (net force AND net moment both exactly zero by
   construction -- see the docstring below for why). This is the
   textbook "free-floating body under self-equilibrated load" case the
   roadmap item is about; solve_static() must not raise, and the
   resulting displacement field must actually satisfy the ORIGINAL
   (unconstrained, full) K @ U = F to good accuracy -- proof that the
   regularization recovered the real physics rather than a numerically
   stable but meaningless answer.

2. A genuinely ill-posed one: load applied on only one edge, nothing
   resisting it anywhere -- not self-equilibrated. This must raise
   LinAlgError with a clear message rather than silently returning
   nonsense (inf/nan or an arbitrary rigid-body drift), since accepting
   it would hide what is actually a missing-boundary-condition modeling
   error.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest
from fea_engine import FESystem, Material, D_plane_stress, Quad4PlaneStress, mesh


def _build_free_floating(nx=6, ny=3, Lx=4.0, Ly=1.0, E=210e9, nu=0.3, sparse=False):
    mat = Material(E=E, nu=nu, rho=7800.0)
    D2 = D_plane_stress(mat)
    m = mesh.rectangle_mesh(Lx, Ly, nx, ny)
    sysobj = FESystem(m, Quad4PlaneStress(), thickness=1.0, sparse=sparse)
    sysobj.assemble_stiffness(D2)
    # deliberately NO fix_dofs() calls -- every dof is free, K is exactly
    # singular (3 rigid-body modes in 2-D: x-translation, y-translation,
    # in-plane rotation)
    return sysobj, m


def _apply_self_equilibrated_tension(sysobj, m, Lx, total_force=1000.0):
    """Equal-and-opposite point loads on the left (x=0) and right (x=Lx)
    edges. Net force is zero by construction (+F on the right, -F on
    the left, same total magnitude). Net moment about any point is also
    zero: left and right edges have IDENTICAL y-coordinates (structured
    rectangle_mesh), and a force pair (+f_i, 0) at (Lx, y_i) / (-f_i, 0)
    at (0, y_i) contributes moment -y_i*f_i and +y_i*f_i respectively,
    which cancel term-by-term -- this is a clean, exactly
    self-equilibrated uniaxial tension load, not an approximately
    balanced one."""
    left = m.nodes_on_line(0, 0.0)
    right = m.nodes_on_line(0, Lx)
    assert len(left) == len(right)
    f_each = total_force / len(right)
    sysobj.F[2 * right] += f_each
    sysobj.F[2 * left] -= f_each


def test_self_equilibrated_free_floating_solve_does_not_raise():
    print("=" * 70)
    print("CHECK 1: free-floating (zero fixed dofs) model with a")
    print("self-equilibrated tension load solves without raising")
    print("=" * 70)
    Lx = 4.0
    sysobj, m = _build_free_floating(Lx=Lx)
    _apply_self_equilibrated_tension(sysobj, m, Lx)

    U = sysobj.solve_static()
    assert np.all(np.isfinite(U))
    print(f"  solve_static() succeeded, max |U| = {np.max(np.abs(U)):.3e}")

    # The real correctness check: does the ORIGINAL (full, unconstrained)
    # system actually balance? K @ U should reproduce F to good accuracy
    # -- this is what would fail if the regularization had corrupted the
    # physics rather than just resolved the singularity.
    residual = sysobj.K @ U - sysobj.F
    resid_norm = np.max(np.abs(residual))
    load_scale = np.max(np.abs(sysobj.F))
    rel_resid = resid_norm / load_scale
    print(f"  max |K@U - F| = {resid_norm:.3e}  (relative to load scale: {rel_resid:.3e})")
    assert rel_resid < 1e-6
    print("  PASS -- regularized solution satisfies the original equilibrium equations")


def test_self_equilibrated_free_floating_solve_no_rigid_body_drift():
    print()
    print("=" * 70)
    print("CHECK 2: the regularized solution carries no arbitrary rigid-body")
    print("drift -- it is the minimum-norm (pure straining) solution")
    print("=" * 70)
    Lx = 4.0
    sysobj, m = _build_free_floating(Lx=Lx)
    _apply_self_equilibrated_tension(sysobj, m, Lx)
    U = sysobj.solve_static()

    ux = U[0::2]
    uy = U[1::2]
    mean_ux, mean_uy = np.mean(ux), np.mean(uy)
    print(f"  mean(ux) = {mean_ux:.3e}, mean(uy) = {mean_uy:.3e}  (both ~0 => no net translation drift)")
    scale = np.max(np.abs(U))
    assert abs(mean_ux) < 1e-6 * scale
    assert abs(mean_uy) < 1e-6 * scale

    # And physically: pulled from both ends in tension, the right edge
    # should have moved net-positive in x relative to the left edge.
    right = m.nodes_on_line(0, Lx)
    left = m.nodes_on_line(0, 0.0)
    elongation = np.mean(U[2 * right]) - np.mean(U[2 * left])
    print(f"  mean right-edge ux - mean left-edge ux = {elongation:.3e}  (expect > 0, tension)")
    assert elongation > 0
    print("  PASS -- minimum-norm solution: zero net drift, physically-sensible elongation")


def test_sparse_free_floating_matches_dense():
    print()
    print("=" * 70)
    print("CHECK 3: sparse=True takes the same regularized-fallback path")
    print("and agrees with the dense path")
    print("=" * 70)
    Lx = 4.0
    sys_d, m_d = _build_free_floating(Lx=Lx, sparse=False)
    _apply_self_equilibrated_tension(sys_d, m_d, Lx)
    U_d = sys_d.solve_static()

    sys_s, m_s = _build_free_floating(Lx=Lx, sparse=True)
    _apply_self_equilibrated_tension(sys_s, m_s, Lx)
    U_s = sys_s.solve_static()

    rel = np.max(np.abs(U_d - U_s)) / max(np.max(np.abs(U_d)), 1e-30)
    print(f"  max rel diff (dense vs sparse) = {rel:.3e}")
    assert rel < 1e-6
    print("  PASS")


def test_non_self_equilibrated_free_floating_raises_clear_error():
    print()
    print("=" * 70)
    print("CHECK 4: a genuinely under-constrained model (load NOT")
    print("self-equilibrated, nothing resists it) raises LinAlgError,")
    print("not a silent garbage/inf/nan 'solution'")
    print("=" * 70)
    Lx = 4.0
    sysobj, m = _build_free_floating(Lx=Lx)
    # load on the right edge only -- no counter-force anywhere, no
    # supports anywhere: this is not in equilibrium
    right = m.nodes_on_line(0, Lx)
    sysobj.F[2 * right] += 1000.0 / len(right)

    with pytest.raises(np.linalg.LinAlgError, match="(?i)not self-equilibrated"):
        sysobj.solve_static()
    print("  PASS -- raised LinAlgError with an actionable message instead of a silent bad answer")
