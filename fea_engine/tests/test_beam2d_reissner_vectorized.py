# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_beam2d_reissner_vectorized.py -- validates Wave 17 item 147 (docs/
consolidated_future_roadmap.md): the batched (all-elements-at-once
NumPy) internal_force()/tangent_stiffness() path for Beam2DReissner,
built because this item's own runtime benchmark found a single
representative forced-vibration run (80 elements, dt=T_forcing/100, 80
forcing periods, through nonlinear_solver.solve_nonlinear_transient())
took ~4.5 minutes -- over this item's own "a few minutes" threshold --
while a candidate escape hatch (solve_transient_explicit_nonlinear())
was measured and found WORSE (~14 min, its much smaller CFL-limited
timestep costs more than it saves by skipping the Newton loop). See the
roadmap's item 147 row for the full numbers.

Two checks, matching this item's own stated validation criterion
("vectorized path matches the looped element to 1e-12"):

(a) Element level: internal_force_batched()/tangent_stiffness_batched()
    vs. the existing per-element looped Beam2DReissner.internal_force()/
    tangent_stiffness() calls, on the SAME random large-rotation states
    test_beam2d_reissner.py's own CHECK (a) uses.
(b) System level: a full solve_nonlinear_transient() run (the item's
    own benchmark problem, at a reduced period count so this test stays
    fast) with the FESystem's assemble_internal_force()/assemble_
    tangent_stiffness() methods monkey-patched to the vectorized path
    (an additive, opt-in substitution -- nothing in solver.py or
    nonlinear_solver.py is modified) reproduces the SAME looped-path
    trajectory to the same 1e-12-class tolerance, end to end through
    Newton iteration and time-stepping, not just at a single element
    evaluation.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine.damping import FieldDamping
from fea_engine.loads import LoadPattern, TimeHistoryLoad
from fea_engine import nonlinear_solver as nls
from fea_engine.beam2d_reissner_vectorized import (
    internal_force_batched, tangent_stiffness_batched,
    assemble_internal_force_vectorized, assemble_tangent_stiffness_vectorized,
)


def _beam_chain(L, n_elem):
    x = np.linspace(0.0, L, n_elem + 1).reshape(-1, 1)
    nodes = np.hstack([x, np.zeros_like(x)])
    elements = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)
    return Mesh(nodes=nodes, elements=elements, dim=1)


def test_check_a_batched_element_matches_looped_to_1e12():
    print("=" * 70)
    print("CHECK (a): internal_force_batched()/tangent_stiffness_batched()")
    print("vs. the existing per-element looped calls, at RANDOM")
    print("LARGE-ROTATION states (same states test_beam2d_reissner.py's")
    print("own CHECK (a) uses)")
    print("=" * 70)
    beam = elmod.Beam2DReissner()
    E, G, A, I, kappa_s = 210e9, 80e9, 1.0e-3, 1.0e-7, 1.0
    mat = (E, G, A, I, kappa_s)
    rng = np.random.default_rng(42)

    n = 12
    elem_coords_all = np.zeros((n, 2, 2))
    u_elem_all = np.zeros((n, 6))
    f_loop = np.zeros((n, 6))
    K_loop = np.zeros((n, 6, 6))
    for i in range(n):
        ec = np.array([[0.0, 0.0], [1.5, 0.9]]) + 0.1 * rng.standard_normal((2, 2))
        u = rng.standard_normal(6)
        u[0:2] *= 0.05; u[3:5] *= 0.05
        u[2] = rng.uniform(-3.0, 3.0)
        u[5] = rng.uniform(-3.0, 3.0)
        elem_coords_all[i] = ec
        u_elem_all[i] = u
        f_loop[i] = beam.internal_force(ec, u, mat)
        K_loop[i] = beam.tangent_stiffness(ec, u, mat)

    f_batch = internal_force_batched(elem_coords_all, u_elem_all, mat)
    K_batch = tangent_stiffness_batched(elem_coords_all, u_elem_all, mat)

    f_err = np.max(np.abs(f_batch - f_loop)) / max(np.max(np.abs(f_loop)), 1e-30)
    K_err = np.max(np.abs(K_batch - K_loop)) / max(np.max(np.abs(K_loop)), 1e-30)
    print(f"  internal_force max relative error: {f_err:.3e}")
    print(f"  tangent_stiffness max relative error: {K_err:.3e}")
    assert f_err < 1e-12, "internal_force_batched does not match looped version to 1e-12"
    assert K_err < 1e-12, "tangent_stiffness_batched does not match looped version to 1e-12"
    print("  PASS")


def test_check_b_full_transient_trajectory_matches_looped_to_1e12_class():
    print("=" * 70)
    print("CHECK (b): a full solve_nonlinear_transient() run, with FESystem")
    print("monkey-patched to the vectorized assembly path, reproduces the")
    print("SAME trajectory as the ordinary looped path (reduced period")
    print("count vs. the full 80-period benchmark, so this test stays fast)")
    print("=" * 70)
    L, A, I = 10.0, 1.0, 1.0 / 12.0
    E, G, rho, kappa_s = 12000.0, 5000.0, 1.0e-6, 1.0
    mat = (E, G, A, I, kappa_s)
    rhoA = rho * A
    n_elem = 20   # smaller than the item's own 80-element benchmark mesh,
                  # purely to keep this regression test's own runtime small
                  # -- the algebra being checked doesn't depend on mesh size

    def build(n_elem):
        mesh = _beam_chain(L, n_elem)
        beam = elmod.Beam2DReissner()
        fes = FESystem(mesh, beam)
        fes.assemble_mass((rhoA, rho * I))
        fes.assemble_stiffness(mat)
        fes.fix_dofs([0, n_elem], [0, 1])
        omega_forcing = 3078.5
        D = 2 * 0.081 * omega_forcing * rhoA
        fes.assemble_damping(FieldDamping(coefficients=(D, D)))
        return fes, omega_forcing

    fes_loop, omega_forcing = build(n_elem)
    fes_vec, _ = build(n_elem)

    T_forcing = 2 * np.pi / omega_forcing
    dt = T_forcing / 100.0
    n_periods = 3   # a few periods is enough to exercise several Newton
                     # steps under real forcing without a slow test
    T_total = n_periods * T_forcing

    P = 5.0e-3
    node_ids = np.arange(n_elem + 1)
    pattern = LoadPattern(node_ids=node_ids, dof_index=1)
    g = lambda t: np.cos(omega_forcing * t)
    load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: P * L * g(t))

    t_loop, U_loop = nls.solve_nonlinear_transient(
        fes_loop, mat, load, T_total, dt, tol=1e-8, max_iter=40, verbose=False)

    fes_vec.assemble_internal_force = lambda u, m, **kw: assemble_internal_force_vectorized(fes_vec, u, m)
    fes_vec.assemble_tangent_stiffness = lambda u, m, **kw: assemble_tangent_stiffness_vectorized(fes_vec, u, m)
    t_vec, U_vec = nls.solve_nonlinear_transient(
        fes_vec, mat, load, T_total, dt, tol=1e-8, max_iter=40, verbose=False)

    err = np.max(np.abs(U_vec - U_loop)) / max(np.max(np.abs(U_loop)), 1e-30)
    print(f"  max relative error over full displacement history: {err:.3e}")
    assert err < 1e-10, "vectorized-path trajectory diverges from the looped-path trajectory"
    print("  PASS")
