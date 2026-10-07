"""
test_explicit_shell_dynamics.py -- validates Wave 6 item 33 (docs/
consolidated_future_roadmap.md: "Explicit dynamics for the corotational
shell"), i.e. that nonlinear_solver.solve_transient_explicit_nonlinear()
(item 31) actually works end-to-end on Shell4MITCCorotational, not just
on the 1-D beam/truss fixtures item 31's own tests use.

Item 33 was framed in the roadmap as mostly a VALIDATION task, not new
element code: Shell4MITCCorotational already has a working mass()
(delegates to the underlying Shell4MITC's) and internal_force(), and
the explicit driver needs neither a Newton loop nor tangent_stiffness()
at all -- so this element should "just work" once item 31 exists. This
file confirms that directly, and along the way exercises the item 32
mass-lumping fix (elements/base.py's Element.lumped_mass()) on a
GENUINELY multi-directional element (Shell4MITCCorotational tracks 3
translational directions per node, x/y/z -- the exact case that fix
was for, distinct from item 31/32's own truss/beam fixtures which only
ever had 1 or 2).

Scope note on timestep size: a full free-vibration run for enough
periods to show a clean energy-conservation trend (item 31/32's own
1-D beam energy test) is NOT attempted here -- directly measured, this
shell mesh's exact global critical timestep (FESystem.critical_
timestep()) is ~40,000x smaller than its fundamental period, so even a
few periods would need on the order of 10^5 explicit steps, each
costing a real Shell4MITCCorotational internal_force() assembly --
far outside this sandbox's per-call wall-clock budget. Instead this
file runs a SHORT window (hundreds of steps, seconds of wall time) and
checks what that window CAN honestly establish: the driver produces a
finite, bounded, physically continuous trajectory (not a crash, not a
NaN, not an unphysical jump), and never calls tangent_stiffness() --
the actual content of item 33's "does this element work under the
explicit driver" question. A full-period energy-conservation demo
would be a natural follow-up if a real use case needs it (mirroring
Wave 4 item 24's own "scoped-down probe, not a full run" precedent for
the exact same sandbox constraint).
"""
import numpy as np
import pytest

from fea_engine import Material, D_shell, Shell4MITCCorotational, rectangle_mesh, FESystem
from fea_engine.mesh import Mesh
from fea_engine.nonlinear_solver import solve_nonlinear_static, solve_transient_explicit_nonlinear
from fea_engine.loads import LoadPattern, TimeHistoryLoad


def _small_cantilever_shell(tip_force=-2.0):
    """A small (4-element) Shell4MITCCorotational cantilever, released
    from nonlinear static equilibrium under a tip load -- same
    construction test_shell_corotational.py's own acceptance test uses
    (rectangle_mesh + a zero-z-coordinate extrusion into 3-D), just a
    coarser mesh (4 elements instead of 16) to keep the explicit
    driver's per-step cost low enough for this sandbox."""
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    h = 0.005
    D = D_shell(mat, h)
    rho = 7800.0
    coro = Shell4MITCCorotational()

    mesh2d = rectangle_mesh(1.0, 0.2, 4, 1)
    nodes3d = np.hstack([mesh2d.nodes, np.zeros((mesh2d.nodes.shape[0], 1))])
    mesh = Mesh(nodes=nodes3d, elements=mesh2d.elements, dim=2)

    fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    tip_nodes = np.where(np.abs(mesh.nodes[:, 0] - 1.0) < 1e-9)[0]

    fes = FESystem(mesh, coro)
    fes.fix_dofs(fixed_nodes, [0, 1, 2, 3, 4, 5])
    fes.add_nodal_force(tip_nodes, 2, tip_force)
    _, U_hist = solve_nonlinear_static(fes, D, n_steps=3, tol=1e-6, max_iter=30, thickness=h)
    u0 = U_hist[-1].copy()

    # rho_matrix = rho*I(6) (isotropic across all 6 dofs/node): the
    # physically load-bearing translational block (indices 0,1,2 local)
    # gets the correct rho*Area*thickness total mass via lumped_mass()'s
    # PER-DIRECTION correction (item 32's fix); the rotational block
    # (3,4,5) just needs a positive placeholder value here (this test
    # never excites/measures rotary inertia specifically), which
    # rho*I(6) trivially provides.
    fes.assemble_lumped_mass(rho * np.eye(6), thickness=h)
    fes.assemble_mass(rho * np.eye(6), thickness=h)     # for critical_timestep()
    fes.assemble_stiffness(D, thickness=h)              # for critical_timestep()

    return fes, D, h, u0, tip_nodes


class TestExplicitShellRuns:
    def test_short_window_stays_finite_and_bounded(self):
        fes, D, h, u0, tip_nodes = _small_cantilever_shell()
        dt_crit = fes.critical_timestep()
        dt = 0.3 * dt_crit
        n_steps = 500

        zero_pattern = LoadPattern(node_ids=tip_nodes, dof_index=2)
        zero_load = TimeHistoryLoad(pattern=zero_pattern, time_fn=lambda t: 0.0)

        t, U_hist = solve_transient_explicit_nonlinear(
            fes, D, zero_load, n_steps * dt, dt, u0=u0, thickness=h)

        assert U_hist.shape == (n_steps + 1, fes.n_dof)
        assert np.all(np.isfinite(U_hist))
        # bounded relative to the release-point static deflection --
        # free vibration around that state should stay the same order
        # of magnitude, not run away, over this short window.
        w0 = np.mean([u0[6 * int(n) + 2] for n in tip_nodes])
        w_hist = np.array([np.mean([U_hist[i][6 * int(n) + 2] for n in tip_nodes])
                            for i in range(len(t))])
        assert np.max(np.abs(w_hist)) < 5.0 * abs(w0)
        # fixed dofs stay exactly zero at every recorded step
        fixed_idx = sorted(fes.fixed_dofs)
        assert np.all(U_hist[:, fixed_idx] == 0.0)

    def test_never_calls_tangent_stiffness(self):
        """The actual architectural claim item 31/33 make: an explicit
        driver needs NO tangent stiffness at all. Verified directly by
        counting calls, not just inferred from reading the driver's
        code -- the same "trust but verify" discipline this project
        applies to every other claim about its own machinery."""
        fes, D, h, u0, tip_nodes = _small_cantilever_shell()
        dt_crit = fes.critical_timestep()
        dt = 0.3 * dt_crit
        n_steps = 200

        zero_pattern = LoadPattern(node_ids=tip_nodes, dof_index=2)
        zero_load = TimeHistoryLoad(pattern=zero_pattern, time_fn=lambda t: 0.0)

        calls = {"n": 0}
        orig = Shell4MITCCorotational.tangent_stiffness

        def _counted(self, *a, **kw):
            calls["n"] += 1
            return orig(self, *a, **kw)

        Shell4MITCCorotational.tangent_stiffness = _counted
        try:
            solve_transient_explicit_nonlinear(
                fes, D, zero_load, n_steps * dt, dt, u0=u0, thickness=h)
        finally:
            Shell4MITCCorotational.tangent_stiffness = orig

        assert calls["n"] == 0
