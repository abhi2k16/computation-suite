"""
test_beam3d_corotational.py -- validation for Wave 4 item 23
(docs/consolidated_future_roadmap.md): elements.Beam3DCorotational, the
geometric-nonlinear (co-rotational) generalization of
Beam3DEulerBernoulli, mirroring Beam2DCorotational's own 2-D
scalar-chord-angle pattern extended to a full 3-D beam (axial +
torsion + two bending planes).

This element is deliberately built via numerical (real central finite
difference) differentiation throughout -- both the natural-dof Jacobian
and the tangent stiffness -- rather than a hand-derived analytic
Battini-Pacoste tangent (see beams3d.py's class docstring for why: that
derivation is a substantial undertaking on its own, and this package's
own established precedent for exactly this situation is "implement via
FD/complex-step first, defer the analytic tangent as a follow-up").

Six lines of evidence, mirroring test_shell_corotational.py's own
structure for the analogous shell element:

1. test_zero_displacement_gives_zero_force -- no prestress in a fresh
   element, for both an axis-aligned and an arbitrarily-oriented beam.

2. test_u0_tangent_matches_linear_beam -- at u=0, tangent_stiffness()
   must reduce to Beam3DEulerBernoulli.stiffness() on the SAME
   geometry. This is the check that caught two real numerical bugs
   during development (see beams3d.py's own class docstring / this
   file's own investigation notes): (a) the natural rotations must be
   extracted via VECTOR SUBTRACTION from a frame-rotation vector,
   mirroring Beam2DCorotational's th_bar=theta-beta, not a raw
   projection (a raw projection is identically blind to translation-
   induced frame rotation at theta=0, silently dropping the exact
   coupling term that gives the correct u=0 bending-translation
   stiffness); and (b) the log-map's angle extraction must avoid
   arccos(trace-based cosine) at small angles (catastrophic
   cancellation near cos=1) in favor of arctan2(sin_theta, cos_theta),
   and the "already aligned" shortcut in the minimal-rotation transport
   must threshold on the SINE of the angle (which scales linearly with
   theta), not the cosine (which scales as 1-theta^2/2 and can
   incorrectly trigger for a genuine but small, FD-relevant rotation).

3. test_tangent_fd_self_consistency -- tangent_stiffness() (itself an
   FD differentiation of internal_force() at h=1e-6) must agree with an
   INDEPENDENT finite difference of internal_force() at a different
   step size, confirming the nested-FD construction is genuinely
   consistent, not merely "the same computation checked against
   itself".

4. test_rigid_motion_invariance -- pure translation gives exactly zero
   force (by construction); rigid rotation about ANY axis (not just the
   beam's own axis) gives force at the floating-point noise floor, a
   stronger result than Shell4MITCCorotational's own drilling-only
   exactness (item 18) since this element's bending naturals ALSO use
   the frame-subtraction construction, not a raw/uncorrected DOF group.

5. test_mass_matches_linear_beam -- mass() is delegated unmodified to
   Beam3DEulerBernoulli.mass() (same convention every nonlinear element
   in this package uses), so it must match exactly.

6. test_cantilever_newton_convergence -- a multi-element cantilever
   under solve_nonlinear_static() must converge and match the linear
   Euler-Bernoulli tip deflection at small load, then show geometric
   SOFTENING (smaller deflection than naive linear scaling) at 100x
   load -- the same qualitative acceptance criterion
   test_nonlinear_beam.py's own CHECK 6 uses for Beam2DCorotational.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from fea_engine.elements.beams3d import Beam3DCorotational, Beam3DEulerBernoulli
from fea_engine.mesh import Mesh
from fea_engine import FESystem
from fea_engine.nonlinear_solver import solve_nonlinear_static


def _rigidities():
    E, G = 210e9, 80e9
    A, J, Iy, Iz = 1e-3, 2e-6, 1e-6, 1.2e-6
    return E * A, G * J, E * Iy, E * Iz


AXIS_ALIGNED = np.array([[0., 0., 0.], [1.5, 0., 0.]])
SKEW = np.array([[0., 0., 0.], [1.5, 0.3, 0.2]])


@pytest.mark.parametrize("elem_coords", [AXIS_ALIGNED, SKEW])
def test_zero_displacement_gives_zero_force(elem_coords):
    rig = _rigidities()
    coro = Beam3DCorotational()
    f0 = coro.internal_force(elem_coords, np.zeros(12), rig)
    assert np.linalg.norm(f0) < 1e-10


@pytest.mark.parametrize("elem_coords", [AXIS_ALIGNED, SKEW])
def test_u0_tangent_matches_linear_beam(elem_coords):
    rig = _rigidities()
    coro = Beam3DCorotational()
    lin = Beam3DEulerBernoulli()
    K_coro = coro.tangent_stiffness(elem_coords, np.zeros(12), rig)
    K_lin = lin.stiffness(elem_coords, rig)
    rel = np.linalg.norm(K_coro - K_lin) / np.linalg.norm(K_lin)
    assert rel < 1e-6

    K_stiff = coro.stiffness(elem_coords, rig)
    assert np.allclose(K_stiff, K_coro)


def test_tangent_fd_self_consistency():
    rig = _rigidities()
    coro = Beam3DCorotational()
    rng = np.random.default_rng(3)
    u = rng.normal(scale=0.02, size=12)

    K_analytic = coro.tangent_stiffness(SKEW, u, rig)

    h = 1e-5
    K_indep = np.zeros((12, 12))
    for j in range(12):
        du = np.zeros(12)
        du[j] = h
        fp = coro.internal_force(SKEW, u + du, rig)
        fm = coro.internal_force(SKEW, u - du, rig)
        K_indep[:, j] = (fp - fm) / (2.0 * h)

    denom = max(np.linalg.norm(K_indep), 1.0)
    rel = np.linalg.norm(K_analytic - K_indep) / denom
    assert rel < 1e-4
    assert np.allclose(K_analytic, K_analytic.T, atol=1e-2 * np.max(np.abs(K_analytic)))


def _rigid_state(elem_coords, angle_deg, axis, translation=np.zeros(3)):
    angle = np.radians(angle_deg)
    Rrig = R.from_rotvec(angle * axis).as_matrix()
    centroid = elem_coords.mean(axis=0)
    u = np.zeros(12)
    for i, X in enumerate(elem_coords):
        u[6 * i:6 * i + 3] = Rrig @ (X - centroid) - (X - centroid) + translation
        u[6 * i + 3:6 * i + 6] = angle * axis
    return u


def test_rigid_motion_invariance():
    rig = _rigidities()
    coro = Beam3DCorotational()
    lin = Beam3DEulerBernoulli()
    K_scale = np.linalg.norm(lin.stiffness(SKEW, rig))

    u_trans = np.zeros(12)
    u_trans[0:3] = [1.0, -0.5, 0.3]
    u_trans[6:9] = [1.0, -0.5, 0.3]
    f_trans = coro.internal_force(SKEW, u_trans, rig)
    assert np.linalg.norm(f_trans) / K_scale < 1e-12

    e1 = (SKEW[1] - SKEW[0]) / np.linalg.norm(SKEW[1] - SKEW[0])
    for angle_deg in [0.5, 6.0, 30.0, 60.0]:
        u = _rigid_state(SKEW, angle_deg, e1)
        f = coro.internal_force(SKEW, u, rig)
        assert np.linalg.norm(f) / K_scale < 1e-10

    perp = np.array([0., 0., 1.]) - np.dot([0., 0., 1.], e1) * e1
    perp = perp / np.linalg.norm(perp)
    for angle_deg in [0.5, 6.0, 30.0]:
        u = _rigid_state(SKEW, angle_deg, perp)
        f = coro.internal_force(SKEW, u, rig)
        assert np.linalg.norm(f) / K_scale < 1e-10


def test_mass_matches_linear_beam():
    coro = Beam3DCorotational()
    lin = Beam3DEulerBernoulli()
    mass_props = (7800.0 * 1e-3, 7800.0 * (1e-6 + 1.2e-6))
    M_coro = coro.mass(SKEW, mass_props)
    M_lin = lin.mass(SKEW, mass_props)
    assert np.allclose(M_coro, M_lin)


def test_cantilever_newton_convergence():
    E, G, A, J, Iy, Iz = 210e9, 80e9, 1e-3, 2e-6, 1e-6, 1.2e-6
    rig = (E * A, G * J, E * Iy, E * Iz)
    n_elem = 6
    L = 2.0
    nodes = np.zeros((n_elem + 1, 3))
    nodes[:, 0] = np.linspace(0, L, n_elem + 1)
    elements = np.array([[i, i + 1] for i in range(n_elem)])
    mesh = Mesh(nodes=nodes, elements=elements, dim=3)
    coro = Beam3DCorotational()
    tip_node = n_elem
    tip_dof_z = tip_node * 6 + 2

    # Small load: converges, near-linear Euler-Bernoulli tip deflection.
    fes = FESystem(mesh, coro, sparse=False)
    fes.fix_dofs([0], [0, 1, 2, 3, 4, 5])
    P_small = 500.0
    fes.add_nodal_force([tip_node], 2, -P_small)
    _, U_hist = solve_nonlinear_static(fes, rig, n_steps=5, tol=1e-8, max_iter=30)
    w_tip = U_hist[-1][tip_dof_z]
    w_lin = P_small * L ** 3 / (3 * E * Iz)
    assert abs(w_tip - (-w_lin)) / w_lin < 0.001

    # Large load (100x): converges, shows geometric SOFTENING (smaller
    # deflection than naive linear scaling), the qualitative signature
    # of large-deflection cantilever bending.
    fes2 = FESystem(mesh, coro, sparse=False)
    fes2.fix_dofs([0], [0, 1, 2, 3, 4, 5])
    P_big = P_small * 100
    fes2.add_nodal_force([tip_node], 2, -P_big)
    _, U_hist2 = solve_nonlinear_static(fes2, rig, n_steps=20, tol=1e-8, max_iter=50)
    w_tip2 = U_hist2[-1][tip_dof_z]
    w_lin2 = P_big * L ** 3 / (3 * E * Iz)
    assert abs(w_tip2) < abs(w_lin2)   # softening: |actual| < |naive linear|
    assert abs(w_tip2) / abs(w_lin2) > 0.5   # not wildly, implausibly off either
