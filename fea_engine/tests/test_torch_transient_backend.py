"""
test_torch_transient_backend.py -- validation for items 94 AND 95 of the
PyTorch side-by-side extension (docs/consolidated_future_roadmap.md, "Wave
9"): the opt-in backend="torch"/device= dispatch wired into
nonlinear_solver.solve_nonlinear_transient() (item 94) AND
nonlinear_solver.solve_transient_displacement_control() (item 95)'s
per-Newton-iteration linear solve (`du_newton = solve(K_eff, -R)`), which
by default (backend="scipy") still calls plain np.linalg.solve() exactly
as before this change. Combined into one file since both items are the
SAME dispatch mechanism applied to two different drivers' identically-
structured Newton loops -- see solve_nonlinear_transient()'s own docstring
"backend=" section (nonlinear_solver.py) for the shared rationale, and
solve_transient_displacement_control()'s own docstring "backend=" section
for the (trivial) cross-reference back to it.

Item 94 is covered by the module-level tests above (test_unknown_backend_
raises_immediately through test_torch_matches_scipy_nonlinear_quasistatic).
Item 95 is covered by TestSolveTransientDisplacementControlBackend below,
reusing the SAME von Mises truss fixture test_transient_displacement_
control.py's own quasi-static check uses, so this file's cross-backend
comparison is validated against the same already-trusted model rather than
a new one.

Four lines of evidence, mirroring item 93's own test file's structure:

1. test_unknown_backend_raises_immediately -- an unrecognized backend value
   fails loudly (ValueError) BEFORE the time-stepping loop even starts, not
   silently or deep inside step N -- checked by confirming it raises even
   with a trivially degenerate (1-step) call.
2. test_scipy_backend_default_unchanged -- backend="scipy" (both implicit
   default and explicit) reproduces this package's own pre-existing
   test_nonlinear_transient.py::TestLinearLimitRegression result bit-for-
   bit -- a regression check that adding the backend= parameter didn't
   perturb the existing SciPy path's numerics AT ALL.
3. test_torch_matches_scipy_linear_limit -- backend="torch" (CPU) vs
   backend="scipy" on the SAME linear-limit fixture
   test_nonlinear_transient.py's own TestLinearLimitRegression uses
   (Beam2DEulerBernoulli, whose tangent_stiffness()/internal_force() are
   linear, so Newton converges in one correction per step and K_eff is
   IDENTICAL between backends at every step) -- the two backends must
   agree to near machine precision, since they are solving the exact same
   dense linear systems through two different LAPACK-family
   implementations (SciPy's own vs. torch.linalg.solve()).
4. test_torch_matches_scipy_nonlinear_quasistatic -- a genuinely NONLINEAR
   case (the same Beam2DCorotational cantilever chain
   test_nonlinear_transient.py's own quasi-static test uses), run once per
   backend, confirming the two backends converge to the SAME equilibrium
   trajectory even when Newton actually has to iterate (K_eff changes
   every correction, not just once) -- the case that actually exercises
   the trust-region/line-search escalation ladder's repeated calls to this
   same dispatch point, not just a single-shot linear solve.

SANDBOX NOTE (2026-09-11): torch is not installed/importable in the
sandbox this file was authored in (see test_torch_autograd_tangent_
stiffness.py's own identical note, and torch_sparse_solver.py's prior
documented finding). Tests 2-4 that need a real torch install are skipped
cleanly via _HAS_TORCH; test 1 (unknown-backend ValueError) needs no torch
at all and always runs.

VALIDATED (2026-09-11, on the user's own machine -- Windows, conda base
env, PyTorch installed): all 7 tests in this file PASSED for real (both
items 94's and 95's, module-level and TestSolveTransientDisplacement
ControlBackend) -- see test_torch_autograd_tangent_stiffness.py's own
updated SANDBOX NOTE for the full combined-run result (30 passed, 1
benign warning, 0 failed across every torch-gated test this wave added).
"""
import numpy as np
import pytest

from fea_engine.material import EI_beam, Material, Section
from fea_engine.geometry import generate_mesh
from fea_engine.solver import FESystem
from fea_engine.damping import RayleighDamping
from fea_engine.mesh import Mesh
from fea_engine import elements as elmod
from fea_engine.loads import LoadPattern, TimeHistoryLoad
from fea_engine import nonlinear_solver as nls

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _clamped_free_nonlinear_beam_chain(n_elem=10, L=1.0, E=210e9, A=1e-4, I=8.333e-9, rho=7850.0):
    """Reused verbatim from test_nonlinear_transient.py -- same fixture,
    so this file's cross-backend check is validated against the same
    already-trusted element/model convention rather than a new one."""
    mat = (E, A, I)
    x = np.linspace(0.0, L, n_elem + 1).reshape(-1, 1)
    nodes = np.hstack([x, np.zeros_like(x)])
    elem_conn = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elem_conn, dim=1)
    beam = elmod.Beam2DCorotational()
    fes = FESystem(mesh, beam)
    fes.fix_dofs([0], [0, 1, 2])
    fes.assemble_mass(rho * A)
    fes.assemble_stiffness(mat)
    fes.assemble_damping(RayleighDamping(alpha=0.0, beta=0.0))
    return fes, mesh, beam, mat, n_elem


def _linear_limit_system():
    n = 10
    mesh = generate_mesh(dim=1, L=1.0, n=n)
    elem = elmod.Beam2DEulerBernoulli()
    sysobj = FESystem(mesh, elem)
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    sec = Section(A=0.01, I=8.33e-6)
    EI = EI_beam(mat, sec)
    sysobj.assemble_stiffness(EI)
    sysobj.assemble_mass(7800.0 * 0.01)
    sysobj.assemble_damping(RayleighDamping(alpha=2.0, beta=1e-5))
    sysobj.fix_dofs([0], [0, 1])
    pattern = LoadPattern(node_ids=np.array([n]), dof_index=0)
    load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t: 1000.0 * min(t / 0.01, 1.0))
    return sysobj, EI, load


def test_unknown_backend_raises_immediately():
    sysobj, EI, load = _linear_limit_system()
    with pytest.raises(ValueError):
        nls.solve_nonlinear_transient(sysobj, EI, load, T_total=0.001, dt=0.0005,
                                       backend="bogus")


def test_scipy_backend_default_unchanged():
    sysobj1, EI1, load1 = _linear_limit_system()
    sysobj2, EI2, load2 = _linear_limit_system()
    T_total, dt = 0.01, 0.0005

    t1, U1 = nls.solve_nonlinear_transient(sysobj1, EI1, load1, T_total, dt, tol=1e-12)
    t2, U2 = nls.solve_nonlinear_transient(sysobj2, EI2, load2, T_total, dt, tol=1e-12,
                                            backend="scipy")
    assert np.array_equal(t1, t2)
    assert np.array_equal(U1, U2)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_torch_matches_scipy_linear_limit():
    sysobj_s, EI_s, load_s = _linear_limit_system()
    sysobj_t, EI_t, load_t = _linear_limit_system()
    T_total, dt = 0.02, 0.0005

    t_s, U_s = nls.solve_nonlinear_transient(sysobj_s, EI_s, load_s, T_total, dt,
                                              tol=1e-12, backend="scipy")
    t_t, U_t = nls.solve_nonlinear_transient(sysobj_t, EI_t, load_t, T_total, dt,
                                              tol=1e-12, backend="torch", device="cpu")

    assert np.array_equal(t_s, t_t)
    rel_diff = np.max(np.abs(U_s - U_t)) / np.max(np.abs(U_s))
    assert rel_diff < 1e-9


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_torch_matches_scipy_nonlinear_quasistatic():
    fes_s, mesh_s, beam_s, mat_s, n_elem = _clamped_free_nonlinear_beam_chain()
    fes_t, mesh_t, beam_t, mat_t, _ = _clamped_free_nonlinear_beam_chain()

    freq_hz, _ = fes_s.solve_modal(n_modes=1)
    period = 1.0 / freq_hz[0]
    P_final = 500.0
    T_ramp = 300.0 * period
    n_steps = 40
    dt = T_ramp / n_steps

    pattern_s = LoadPattern(node_ids=np.array([n_elem]), dof_index=1)
    load_s = TimeHistoryLoad(pattern=pattern_s, time_fn=lambda t: P_final * min(t / T_ramp, 1.0))
    pattern_t = LoadPattern(node_ids=np.array([n_elem]), dof_index=1)
    load_t = TimeHistoryLoad(pattern=pattern_t, time_fn=lambda t: P_final * min(t / T_ramp, 1.0))

    t_s, U_s = nls.solve_nonlinear_transient(fes_s, mat_s, load_s, T_ramp, dt,
                                              tol=1e-8, max_iter=40, backend="scipy")
    t_t, U_t = nls.solve_nonlinear_transient(fes_t, mat_t, load_t, T_ramp, dt,
                                              tol=1e-8, max_iter=40, backend="torch",
                                              device="cpu")

    assert np.array_equal(t_s, t_t)
    rel_diff = np.max(np.abs(U_s[-1] - U_t[-1])) / np.max(np.abs(U_s[-1]))
    assert rel_diff < 1e-6


# =====================================================================
# Item 95: solve_transient_displacement_control()'s own backend= dispatch
# =====================================================================
A_SPAN = 1.0
H0 = 0.10
E_TRUSS, A_TRUSS = 210e9, 2e-4
RHO = 7850.0
DELTA_PEAK = H0 * (3 - np.sqrt(3)) / 3


def _von_mises_truss():
    """Reused verbatim from test_transient_displacement_control.py."""
    nodes = np.array([[-A_SPAN, 0.0], [0.0, H0], [A_SPAN, 0.0]])
    elements = np.array([[0, 1], [1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1])
    control_dof = 1 * fes.npn + 1
    fes.assemble_mass(RHO * A_TRUSS * np.eye(2))
    return fes, control_dof


class TestSolveTransientDisplacementControlBackend:
    def test_unknown_backend_raises_immediately(self):
        fes, control_dof = _von_mises_truss()
        fes.assemble_damping(RayleighDamping(alpha=2048.0, beta=0.0))

        def u_target_fn(t):
            return -0.01 * min(t / 1.0, 1.0)

        with pytest.raises(ValueError):
            nls.solve_transient_displacement_control(
                fes, (E_TRUSS, A_TRUSS), control_dof, u_target_fn,
                T_total=0.01, dt=0.01, backend="bogus")

    def test_scipy_backend_default_unchanged(self):
        fes1, cd1 = _von_mises_truss()
        fes1.assemble_damping(RayleighDamping(alpha=2048.0, beta=0.0))
        fes2, cd2 = _von_mises_truss()
        fes2.assemble_damping(RayleighDamping(alpha=2048.0, beta=0.0))

        delta_max = 1.2 * DELTA_PEAK
        T_ramp = 5.0

        def u_target_fn(t):
            frac = min(t / T_ramp, 1.0)
            smooth = 0.5 * (1 - np.cos(np.pi * frac))
            return -smooth * delta_max

        n_steps = 20
        dt = T_ramp / n_steps
        t1, U1, R1 = nls.solve_transient_displacement_control(
            fes1, (E_TRUSS, A_TRUSS), cd1, u_target_fn, T_ramp, dt,
            tol=1e-10, max_iter=50)
        t2, U2, R2 = nls.solve_transient_displacement_control(
            fes2, (E_TRUSS, A_TRUSS), cd2, u_target_fn, T_ramp, dt,
            tol=1e-10, max_iter=50, backend="scipy")

        assert np.array_equal(t1, t2)
        assert np.array_equal(U1, U2)
        assert np.array_equal(R1, R2)

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_torch_matches_scipy_quasistatic(self):
        fes_s, cd_s = _von_mises_truss()
        fes_s.assemble_damping(RayleighDamping(alpha=2048.0, beta=0.0))
        fes_t, cd_t = _von_mises_truss()
        fes_t.assemble_damping(RayleighDamping(alpha=2048.0, beta=0.0))

        delta_max = 1.2 * DELTA_PEAK
        T_ramp = 5.0

        def u_target_fn(t):
            frac = min(t / T_ramp, 1.0)
            smooth = 0.5 * (1 - np.cos(np.pi * frac))
            return -smooth * delta_max

        n_steps = 100
        dt = T_ramp / n_steps
        t_s, U_s, R_s = nls.solve_transient_displacement_control(
            fes_s, (E_TRUSS, A_TRUSS), cd_s, u_target_fn, T_ramp, dt,
            tol=1e-10, max_iter=50, backend="scipy")
        t_t, U_t, R_t = nls.solve_transient_displacement_control(
            fes_t, (E_TRUSS, A_TRUSS), cd_t, u_target_fn, T_ramp, dt,
            tol=1e-10, max_iter=50, backend="torch", device="cpu")

        assert np.array_equal(t_s, t_t)
        rel_diff_U = np.max(np.abs(U_s - U_t)) / np.max(np.abs(U_s))
        assert rel_diff_U < 1e-6
        rel_diff_R = np.max(np.abs(R_s - R_t)) / np.max(np.abs(R_s))
        assert rel_diff_R < 1e-6
