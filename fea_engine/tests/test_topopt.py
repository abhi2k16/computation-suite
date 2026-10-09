# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_topopt.py -- Wave 11 item 111 (docs/consolidated_future_roadmap.md):
validates topopt.py -- the SIMP interpolation, the closed-form
compliance sensitivity against a direct finite-difference perturbation
of the actual forward solve (decisive, unconditional -- no torch
needed), the torch-gated adjoint-layer sensitivity path against that
SAME closed-form reference (the actual "does item 98's adjoint layer
work end-to-end on a new problem" claim this item exists to validate),
the OC update's own volume-constraint/move-limit contract, and a real
end-to-end compliance-minimization run on a short-cantilever domain.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from fea_engine import Quad4PlaneStress, FESystem, Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh
from fea_engine.differentiable import _HAS_TORCH
from fea_engine.topopt import (
    simp_scale, simp_scale_grad, unit_stiffness_stack, solve_simp_equilibrium,
    compliance, compliance_sensitivity_closed_form, compliance_sensitivity_via_adjoint,
    oc_update, build_filter_weights, apply_sensitivity_filter, topology_optimize_compliance,
)


def _steel():
    return Material(E=2.1e11, nu=0.3, rho=7850.0)


def _short_cantilever(nx=8, ny=4):
    mesh = rectangle_mesh(Lx=float(nx), Ly=float(ny), nx=nx, ny=ny)
    fs = FESystem(mesh, Quad4PlaneStress())
    left = mesh.nodes_on_line(axis=0, value=0.0)
    for n in left:
        fs.fix_dofs([n], [0, 1])
    F_ext = np.zeros(fs.n_dof)
    # point load at bottom-right corner, pulling down -- classic short-
    # cantilever topopt benchmark shape
    br = mesh.nodes_on_line(axis=0, value=float(nx))
    br = [n for n in br if abs(mesh.nodes[n, 1] - 0.0) < 1e-9]
    assert len(br) == 1
    F_ext[fs._global_dofs(br)[1]] = -1.0e5
    return fs, F_ext


# ---------------------------------------------------------------------
# SIMP interpolation
# ---------------------------------------------------------------------
def test_simp_scale_bounds():
    rho = np.array([0.0, 0.5, 1.0])
    s = simp_scale(rho, p=3.0, rho_min=1e-3)
    assert np.isclose(s[0], 1e-3)
    assert np.isclose(s[-1], 1.0)
    assert s[0] < s[1] < s[2]


def test_simp_scale_grad_matches_finite_difference():
    rho = np.array([0.2, 0.5, 0.8])
    p, rho_min = 3.0, 1e-3
    eps = 1e-6
    g_analytic = simp_scale_grad(rho, p, rho_min)
    g_fd = (simp_scale(rho + eps, p, rho_min) - simp_scale(rho - eps, p, rho_min)) / (2 * eps)
    assert np.allclose(g_analytic, g_fd, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------
# Closed-form sensitivity vs. direct finite difference of the REAL
# forward solve -- the decisive, unconditional check.
# ---------------------------------------------------------------------
def test_closed_form_sensitivity_matches_finite_difference_of_forward_solve():
    fs, F_ext = _short_cantilever(nx=6, ny=3)
    D = D_plane_stress(_steel())
    ke_unit, connectivity = unit_stiffness_stack(fs, D)
    dofs_per_node = fs.elem.dofs_per_node
    n_dof = fs.n_dof
    free = fs.free_dofs
    p, rho_min = 3.0, 1e-2

    rng = np.random.default_rng(0)
    rho = rng.uniform(0.3, 0.9, size=connectivity.shape[0])

    u_full, Kff, Ff = solve_simp_equilibrium(ke_unit, connectivity, dofs_per_node, n_dof, free, F_ext, rho, p, rho_min)
    dc_analytic = compliance_sensitivity_closed_form(ke_unit, connectivity, dofs_per_node, u_full, rho, p, rho_min)

    # perturb a handful of elements individually and re-solve
    eps = 1e-4
    check_elems = [0, 3, 7, 12]
    for e in check_elems:
        rho_p = rho.copy(); rho_p[e] += eps
        rho_m = rho.copy(); rho_m[e] -= eps
        u_p, _, _ = solve_simp_equilibrium(ke_unit, connectivity, dofs_per_node, n_dof, free, F_ext, rho_p, p, rho_min)
        u_m, _, _ = solve_simp_equilibrium(ke_unit, connectivity, dofs_per_node, n_dof, free, F_ext, rho_m, p, rho_min)
        C_p = compliance(F_ext, u_p)
        C_m = compliance(F_ext, u_m)
        dc_fd = (C_p - C_m) / (2 * eps)
        assert np.isclose(dc_analytic[e], dc_fd, rtol=1e-3, atol=1.0), (e, dc_analytic[e], dc_fd)


# ---------------------------------------------------------------------
# Torch-gated: adjoint-layer sensitivity vs. the closed-form reference
# ---------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed in this environment")
def test_adjoint_sensitivity_matches_closed_form():
    fs, F_ext = _short_cantilever(nx=6, ny=3)
    D = D_plane_stress(_steel())
    ke_unit, connectivity = unit_stiffness_stack(fs, D)
    dofs_per_node = fs.elem.dofs_per_node
    n_dof = fs.n_dof
    free = fs.free_dofs
    p, rho_min = 3.0, 1e-2

    rng = np.random.default_rng(1)
    rho = rng.uniform(0.3, 0.9, size=connectivity.shape[0])

    u_full, Kff, Ff = solve_simp_equilibrium(ke_unit, connectivity, dofs_per_node, n_dof, free, F_ext, rho, p, rho_min)
    dc_closed = compliance_sensitivity_closed_form(ke_unit, connectivity, dofs_per_node, u_full, rho, p, rho_min)
    dc_adjoint = compliance_sensitivity_via_adjoint(
        ke_unit, connectivity, dofs_per_node, n_dof, free, u_full, Kff, Ff, rho, p, rho_min)

    assert np.allclose(dc_closed, dc_adjoint, rtol=1e-6, atol=1e-3)


# ---------------------------------------------------------------------
# OC update contract
# ---------------------------------------------------------------------
def test_oc_update_respects_move_limit_and_volume_target():
    rho = np.full(20, 0.4)
    dc = -np.linspace(1.0, 5.0, 20)   # all negative, varying magnitude
    rho_new = oc_update(rho, dc, volume_fraction=0.4, move=0.2, rho_min_bound=1e-3)
    assert np.all(rho_new >= rho - 0.2 - 1e-9)
    assert np.all(rho_new <= rho + 0.2 + 1e-9)
    assert np.all(rho_new >= 1e-3 - 1e-12)
    assert np.all(rho_new <= 1.0 + 1e-12)
    assert np.isclose(rho_new.mean(), 0.4, atol=0.02)


def test_oc_update_positive_dc_is_handled_not_raised():
    rho = np.full(5, 0.3)
    dc = np.array([1.0, -1.0, 0.0, -2.0, 1e-15])   # a couple of non-negative entries
    rho_new = oc_update(rho, dc, volume_fraction=0.3, move=0.2)
    assert np.all(np.isfinite(rho_new))


# ---------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------
def test_filter_weights_symmetric_and_self_weighted():
    centroids = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    H = build_filter_weights(centroids, radius=1.5)
    assert np.allclose(H, H.T)
    assert np.all(np.diag(H) > 0)


def test_apply_sensitivity_filter_constant_field_is_unchanged():
    centroids = np.array([[float(i), float(j)] for i in range(4) for j in range(4)])
    H = build_filter_weights(centroids, radius=1.5)
    rho = np.full(16, 0.5)
    dc = np.full(16, -3.0)
    dc_filtered = apply_sensitivity_filter(H, rho, dc)
    assert np.allclose(dc_filtered, -3.0, atol=1e-8)


# ---------------------------------------------------------------------
# End-to-end optimization
# ---------------------------------------------------------------------
def test_topology_optimize_compliance_reduces_compliance_and_holds_volume():
    fs, F_ext = _short_cantilever(nx=10, ny=5)
    D = D_plane_stress(_steel())
    result = topology_optimize_compliance(
        fs, D, F_ext, volume_fraction=0.4, p=3.0, rho_min=1e-2,
        move=0.2, n_iter=40, filter_radius=1.5, sensitivity="closed_form")

    hist = result["compliance_history"]
    assert len(hist) >= 2
    # headline claim (mirrors OCOptimizer's own "large compliance
    # reduction at an exactly-held volume constraint"): final
    # compliance is substantially below the uniform-density starting
    # point, and the volume constraint is respected.
    assert hist[-1] < 0.5 * hist[0]
    assert np.isclose(result["rho"].mean(), 0.4, atol=0.03)
    assert np.all(result["rho"] >= 1e-2 - 1e-9)
    assert np.all(result["rho"] <= 1.0 + 1e-9)


def test_unknown_sensitivity_mode_rejected():
    fs, F_ext = _short_cantilever(nx=4, ny=2)
    D = D_plane_stress(_steel())
    with pytest.raises(ValueError, match="unknown sensitivity"):
        topology_optimize_compliance(fs, D, F_ext, volume_fraction=0.4, sensitivity="bogus", n_iter=1)
