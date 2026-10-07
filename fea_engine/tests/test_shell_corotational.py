"""
test_shell_corotational.py -- validation for Phases A and B of
docs/geometric_nonlinear_shell_roadmap.md: elements.Shell4MITCCorotational,
the geometric-nonlinear (co-rotational) wrapper around Shell4MITC.

Six lines of evidence, matching the roadmap's Section 4 validation plan
plus the checks Phase A's and Phase B's own build orders call out
explicitly (Section 5: Phase A "validated against Section 4 items 1-2";
Phase B "analytic tangent_stiffness()... cross-validated against Phase
A's complex-step version"):

1. test_zero_displacement_gives_zero_force -- internal_force(elem_coords,
   0, D) must be exactly zero (no prestress in a fresh element), for
   both a plain axis-aligned unit square and a non-trivial rectangular
   element -- the most basic sanity check for any nonlinear element.

2. test_u0_tangent_matches_linear_shell -- roadmap Section 4 item 1: at
   u=0, tangent_stiffness() (equivalently stiffness()) must reduce to
   Shell4MITC.stiffness() on the SAME geometry, since a co-rotational
   element with zero displacement is, by definition, indistinguishable
   from the underlying linear element. Verified to ~1e-4 relative or
   better (see this class's own module-level "Why this reduces
   EXACTLY..." comment for why the small residual is floating-point-
   level, not a modeling approximation).

3. test_complex_step_matches_finite_difference -- roadmap Section 4
   item 2: tangent_stiffness()'s derivative must agree with an
   independent real central-finite-difference derivative of
   internal_force(), confirming tangent_stiffness() genuinely IS the
   derivative of internal_force() and not some other, inconsistent
   computation -- checked at a generic (non-equilibrium, non-trivial)
   displacement state. Runs against whichever implementation
   tangent_stiffness() currently is (Phase B's analytic version by
   default), so this test alone would have caught either Phase's tangent
   being wrong.

4. test_analytic_matches_complex_step -- Phase B's own explicit
   cross-validation requirement: tangent_stiffness() (now the analytic
   material+geometric split, see shells.py's "Tangent" comment) must
   match _tangent_stiffness_complex_step() (the ORIGINAL Phase A
   implementation, kept as the independent reference) to near-machine
   precision, at u=0, at small and larger random states, at the specific
   bending-dominated state that broke the rejected Phase A dof-
   extraction designs (see "Design history"), and at rigid rotations --
   i.e. everywhere the earlier tests already probed internal_force()
   itself, now probing its DERIVATIVE the same way.

5. test_rigid_rotation_invariance -- translating and rotating an entire
   element rigidly (zero local strain by construction) must produce
   near-zero internal force, at several rotation magnitudes (both an
   in-plane "drilling" rotation and an out-of-plane "tilt" rotation),
   confirming the residual error stays small and grows the way this
   class's own docstring describes (see "Known remaining limitation").

6. test_cantilever_newton_convergence -- the real acceptance test for
   this element: a multi-element Shell4MITCCorotational cantilever mesh,
   loaded incrementally via solve_nonlinear_static(), must actually
   CONVERGE using whichever tangent_stiffness() is current (this is what
   caught two earlier, rejected Phase A dof-extraction designs -- see
   shells.py's own "Design history" comment -- neither of which could
   solve even this unremarkable problem), and the small-load limit must
   be close to the linear Euler-Bernoulli tip deflection while a
   100x-larger load shows the expected geometric SOFTENING-of-the-
   linear-scaling (smaller deflection than naive linear scaling would
   predict) characteristic of large-deflection bending.
"""
import numpy as np
import pytest

from fea_engine import Material, D_shell, Shell4MITC, Shell4MITCCorotational
from fea_engine import rectangle_mesh, FESystem
from fea_engine.mesh import Mesh
from fea_engine.nonlinear_solver import solve_nonlinear_static


def _material():
    mat = Material(E=210e9, nu=0.3, rho=7800.0)
    h = 0.005
    return D_shell(mat, h), h


UNIT_SQUARE = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
RECT = np.array([[0.875, 0., 0.], [1., 0., 0.], [1., 0.1, 0.], [0.875, 0.1, 0.]])


@pytest.mark.parametrize("elem_coords", [UNIT_SQUARE, RECT])
def test_zero_displacement_gives_zero_force(elem_coords):
    D, h = _material()
    coro = Shell4MITCCorotational()
    f0 = coro.internal_force(elem_coords, np.zeros(24), D, thickness=h)
    assert np.linalg.norm(f0) == 0.0


@pytest.mark.parametrize("elem_coords", [UNIT_SQUARE, RECT])
def test_u0_tangent_matches_linear_shell(elem_coords):
    D, h = _material()
    coro = Shell4MITCCorotational()
    lin = Shell4MITC()
    K0 = coro.stiffness(elem_coords, D, thickness=h)
    Klin = lin.stiffness(elem_coords, D)
    rel = np.linalg.norm(K0 - Klin) / np.linalg.norm(Klin)
    assert rel < 1e-4


def test_complex_step_matches_finite_difference():
    D, h = _material()
    coro = Shell4MITCCorotational()
    rng = np.random.default_rng(0)
    u = rng.normal(scale=0.05, size=24)

    K_cs = coro.tangent_stiffness(RECT, u, D, thickness=h)

    eps = 1e-6
    K_fd = np.zeros((24, 24))
    for j in range(24):
        up = u.copy(); up[j] += eps
        um = u.copy(); um[j] -= eps
        K_fd[:, j] = (coro.internal_force(RECT, up, D, thickness=h)
                      - coro.internal_force(RECT, um, D, thickness=h)) / (2 * eps)

    rel = np.linalg.norm(K_cs - K_fd) / np.linalg.norm(K_fd)
    # tangent_stiffness() symmetrizes an internal_force() Jacobian that is
    # not exactly symmetric by construction (see shells.py's own "Tangent"
    # comment) -- a few percent, not the ~1e-10 a truly symmetric analytic
    # Jacobian would give, but this is Newton-Raphson's linearization, not
    # a claim of an exact Hessian.
    assert rel < 0.05


def _rigid_state(angle_deg, axis):
    from scipy.spatial.transform import Rotation as R
    angle = np.radians(angle_deg)
    Rrig = R.from_rotvec(angle * axis).as_matrix()
    centroid = RECT.mean(axis=0)
    u_rigid = np.zeros(24)
    for i in range(4):
        xi = RECT[i]
        u_rigid[6 * i:6 * i + 3] = Rrig @ (xi - centroid) - (xi - centroid)
        u_rigid[6 * i + 3:6 * i + 6] = angle * axis
    return u_rigid


# The bending-dominated state that broke the two rejected Phase A
# dof-extraction designs (see shells.py's "Design history") -- near-zero
# in-plane translation, moderate w and theta_y, exactly the shape a real
# Newton-Raphson correction on a real cantilever mesh produces (this is
# where the earlier, rejected constructions diverged catastrophically
# even though every OTHER check here passed for them).
BENDING_STATE = np.array([
    -9.45496627e-19, -6.60226464e-18, 6.01493623e-03, 8.85492033e-05,
    1.10707016e-02, 1.90659248e-15, -7.28417505e-19, -8.15580899e-18,
    7.41009246e-03, 2.51317002e-05, 1.12513581e-02, -1.40270019e-18,
    1.27651772e-18, -7.67937185e-18, 7.41134521e-03, -1.02329006e-12,
    1.12292501e-02, -1.43540089e-19, 7.02402439e-19, -6.20884405e-18,
    6.01936519e-03, -9.63461344e-13, 1.10422762e-02, 6.98332856e-15,
])


@pytest.mark.parametrize("u", [
    np.zeros(24),
    np.random.default_rng(1).normal(scale=0.05, size=24),
    np.random.default_rng(2).normal(scale=0.2, size=24),
    BENDING_STATE,
    _rigid_state(6.0, np.array([0., 0., 1.])),
    _rigid_state(6.0, np.array([0., 1., 0.])),
])
def test_analytic_matches_complex_step(u):
    D, h = _material()
    coro = Shell4MITCCorotational()
    K_analytic = coro.tangent_stiffness(RECT, u, D, thickness=h)
    K_cs = coro._tangent_stiffness_complex_step(RECT, u, D, thickness=h)
    denom = max(np.linalg.norm(K_cs), 1.0)
    rel = np.linalg.norm(K_analytic - K_cs) / denom
    assert rel < 1e-8


@pytest.mark.parametrize("angle_deg", [0.5, 6.0, 30.0])
@pytest.mark.parametrize("axis", [np.array([0., 0., 1.]), np.array([0., 1., 0.])])
def test_rigid_rotation_invariance(angle_deg, axis):
    D, h = _material()
    coro = Shell4MITCCorotational()
    lin = Shell4MITC()
    K_scale = np.linalg.norm(lin.stiffness(RECT, D))

    u_rigid = _rigid_state(angle_deg, axis)
    f_rigid = coro.internal_force(RECT, u_rigid, D, thickness=h)
    rel = np.linalg.norm(f_rigid) / K_scale
    # Known remaining limitation (shells.py docstring): bounded but
    # non-zero for large single-element rigid motion -- generous bound,
    # this test is about the error staying SMALL and BOUNDED, not exact.
    assert rel < 0.05


def test_cantilever_newton_convergence():
    D, h = _material()
    coro = Shell4MITCCorotational()

    mesh2d = rectangle_mesh(1.0, 0.2, 8, 2)
    nodes3d = np.hstack([mesh2d.nodes, np.zeros((mesh2d.nodes.shape[0], 1))])
    mesh = Mesh(nodes=nodes3d, elements=mesh2d.elements, dim=2)

    fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    tip_nodes = np.where(np.abs(mesh.nodes[:, 0] - 1.0) < 1e-9)[0]

    # Small load: must converge and land close to the linear
    # Euler-Bernoulli tip deflection P L^3/(3EI).
    #
    # tol=1e-6, not 1e-8 (2026-09-03, added alongside the bending-
    # membrane coupling term -- see shells.py's _bending_membrane_
    # coupling_force()): coupling the membrane block into the bending
    # block's response (needed to produce real bending-induced axial
    # foreshortening, see that method's own docstring) raises this
    # system's condition number from block-diagonal-well-conditioned to
    # ~1.7e10 (measured directly) -- the two blocks' native stiffness
    # scales differ by the same ~1e6-1e7 ratio this file's other
    # comments already document for K_local0 itself. Verified directly
    # (not assumed) that this is a genuine double-precision noise FLOOR,
    # not a bug: plain Newton's residual plateaus at ~1.5e-7-5e-7
    # (bouncing, not trending down) even after 80 iterations, while
    # |du| shrinks to ~1e-12 -- i.e. Newton has already found the best
    # answer representable at this system's conditioning; disabling the
    # new coupling term reproduces the OLD clean one-iteration
    # convergence to ~1e-10 exactly. tol=1e-6 sits comfortably above the
    # measured floor with margin.
    sys_small = FESystem(mesh, coro, thickness=h)
    sys_small.fix_dofs(fixed_nodes, [0, 1, 2, 3, 4, 5])
    sys_small.add_nodal_force(tip_nodes, 2, -5.0)
    _, U_hist_small = solve_nonlinear_static(
        sys_small, D, n_steps=3, tol=1e-6, max_iter=30)
    w_small = np.mean([U_hist_small[-1][6 * int(n) + 2] for n in tip_nodes])

    E, L, b = 210e9, 1.0, 0.2
    I = b * h ** 3 / 12
    w_eb = -5.0 * L ** 3 / (3 * E * I)
    assert abs(w_small - w_eb) / abs(w_eb) < 0.10

    # Large load (100x): must still converge, and the large-deflection
    # response must show geometric SOFTENING relative to naive linear
    # scaling of the small-load result (smaller |w| than 100x w_small).
    sys_large = FESystem(mesh, coro, thickness=h)
    sys_large.fix_dofs(fixed_nodes, [0, 1, 2, 3, 4, 5])
    sys_large.add_nodal_force(tip_nodes, 2, -500.0)
    _, U_hist_large = solve_nonlinear_static(
        sys_large, D, n_steps=10, tol=1e-6, max_iter=30)
    w_large = np.mean([U_hist_large[-1][6 * int(n) + 2] for n in tip_nodes])

    assert abs(w_large) < 100 * abs(w_small)
