"""
test_shell_corotational_mean_rotation.py -- validation for Wave 4 item 46
building block A (docs/shell_rotation_coupling_fix_roadmap.md,
docs/consolidated_future_roadmap.md): the general, exact SO(3)
rigid-rotation extraction `Shell4MITCCorotational._mean_rigid_rotation()`
(plus its `_exp_map`/`_log_map` primitives), generalizing item 18's
exact single-axis drilling update to a genuine 3-vector rotation.

Scope, precisely: this is an ISOLATED validation of the extraction
PRIMITIVE alone. It is NOT yet wired into `_local_relative_dofs()`/
`internal_force()`/`tangent_stiffness()` -- see the large comment block
in shells.py directly above `_exp_map` for why (using this naively to
replace the membrane block's translation extraction the way `R_drill`
does would reproduce "Design history" dead end 3: a general 3-vector
frame's z-row is not [0,0,1], so it re-couples out-of-plane translation
into the membrane block). This file only tests whether the extraction
itself is correct, exact, and well-posed -- the necessary first gate
before any wiring-in is attempted.

Four lines of evidence:

1. test_exp_log_map_roundtrip -- the two primitives `_exp_map`/
   `_log_map` are correct, mutual inverses, and match an independent
   reference (`scipy.spatial.transform.Rotation`) across small and
   large angles and several axes -- the correctness gate for
   everything built on top of them.

2. test_identical_rotations_gives_exact_mean_and_zero_residual -- the
   header claim, made precise and parametrized exactly like
   `test_shell_corotational_exact_drill.py`'s own drilling-exactness
   check but for a GENERAL (not z-only) axis: if all 4 nodes carry the
   IDENTICAL rotation vector (any axis, any angle 0.5-150 degrees), the
   extracted mean R_e must equal that rotation EXACTLY and every node's
   residual rotation must be EXACTLY zero (floating-point noise, not
   "small") -- what a genuine Frechet mean guarantees by construction,
   unlike a heuristic hand-derived frame (which is exactly what made
   dead end 1 fail).

3. test_small_bending_perturbation_captured -- for a NON-rigid state
   (nodes disagreeing on rotation, a genuine bending signal), the
   residuals are non-zero, sum to (near) zero (the Frechet-mean
   stationarity condition the iteration solves for), and scale with the
   size of the injected perturbation -- confirms the extraction responds
   to real bending content, not just passing the trivial rigid case.

4. test_convergence_is_fast -- the Newton/fixed-point iteration
   converges in a small, bounded number of steps for any physically
   realistic single-element rotation spread (checked directly by
   instrumenting the same iteration inline, not assumed) -- relevant
   because a future wiring-in would call this once per
   internal_force()/tangent_stiffness() evaluation, so slow or
   non-converging behavior would be a real practical blocker.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from fea_engine import Shell4MITCCorotational


def _exp(v):
    return Shell4MITCCorotational._exp_map(v)


def _log(Rmat):
    return Shell4MITCCorotational._log_map(Rmat)


def _mean(theta_nodes, **kw):
    return Shell4MITCCorotational._mean_rigid_rotation(np.asarray(theta_nodes), **kw)


@pytest.mark.parametrize("angle_deg", [0.0, 1e-6, 0.5, 6.0, 30.0, 90.0, 150.0])
@pytest.mark.parametrize("axis", [
    np.array([0., 0., 1.]),
    np.array([0., 1., 0.]),
    np.array([1., 0., 0.]),
    np.array([1., 1., 1.]) / np.sqrt(3.0),
])
def test_exp_log_map_roundtrip(angle_deg, axis):
    angle = np.radians(angle_deg)
    v = angle * axis
    Rmat = _exp(v)

    # Cross-check against an independent reference implementation.
    R_ref = R.from_rotvec(v).as_matrix()
    assert np.linalg.norm(Rmat - R_ref) < 1e-10, f"_exp_map vs scipy at {angle_deg}deg"

    # Round-trip: log_map(exp_map(v)) == v (away from the documented
    # theta~pi degeneracy -- 150deg is comfortably clear of it).
    v_back = _log(Rmat)
    assert np.linalg.norm(v_back - v) < 1e-9, f"round-trip at {angle_deg}deg: {v} vs {v_back}"


@pytest.mark.parametrize("angle_deg", [0.5, 6.0, 30.0, 60.0, 90.0, 150.0])
@pytest.mark.parametrize("axis", [
    np.array([0., 0., 1.]),
    np.array([0., 1., 0.]),
    np.array([1., 0., 0.]),
    np.array([1., 2., -1.]) / np.linalg.norm([1., 2., -1.]),
])
def test_identical_rotations_gives_exact_mean_and_zero_residual(angle_deg, axis):
    angle = np.radians(angle_deg)
    v = angle * axis
    theta_nodes = np.tile(v, (4, 1))   # all 4 nodes: identical rotation

    R_e, r_nodes = _mean(theta_nodes)

    R_expected = _exp(v)
    rel_R = np.linalg.norm(R_e - R_expected) / max(np.linalg.norm(R_expected), 1.0)
    assert rel_R < 1e-10, f"R_e vs exact rotation at {angle_deg}deg about {axis}: rel={rel_R:.3e}"

    max_residual = np.max(np.linalg.norm(r_nodes, axis=1))
    assert max_residual < 1e-9, f"residual rotation at {angle_deg}deg: max|r|={max_residual:.3e}"


def test_zero_rotation_gives_identity():
    R_e, r_nodes = _mean(np.zeros((4, 3)))
    assert np.linalg.norm(R_e - np.eye(3)) < 1e-12
    assert np.linalg.norm(r_nodes) < 1e-12


def test_small_bending_perturbation_captured():
    rng = np.random.default_rng(11)
    base = np.radians(20.0) * np.array([0.3, -0.2, 0.9]) / np.linalg.norm([0.3, -0.2, 0.9])
    perturb = rng.normal(scale=0.01, size=(4, 3))   # genuine within-element disagreement
    theta_nodes = base[None, :] + perturb

    R_e, r_nodes = _mean(theta_nodes)

    # Stationarity condition the iteration solves for: mean residual ~ 0.
    assert np.linalg.norm(r_nodes.mean(axis=0)) < 1e-10

    # Residuals are non-trivial (this is a real bending signal, not the
    # degenerate all-identical case above)...
    assert np.max(np.linalg.norm(r_nodes, axis=1)) > 1e-4
    # ...but stay commensurate with the injected perturbation's own
    # scale (order 0.01 rad), not blown up by the extraction.
    assert np.max(np.linalg.norm(r_nodes, axis=1)) < 0.05

    # Each node's residual should be close to its own perturbation minus
    # the mean perturbation (first-order consistency check -- exact SO(3)
    # composition isn't identical to vector subtraction, so this can only
    # hold to O(base_angle * perturb_scale), not floating-point precision;
    # measured discrepancy here (~2.5e-3) is the right ORDER for that
    # (base_angle~0.35 rad * perturb_scale~0.01 rad ~ 3.5e-3), not a bug
    # in the extraction -- confirmed directly by rerunning with the base
    # angle scaled down 10x (20deg -> 2deg), where the max discrepancy
    # dropped ~8.6x (2.5e-3 -> 2.9e-4), consistent with roughly-linear
    # scaling in the base angle, not a fixed extraction error).
    expected = perturb - perturb.mean(axis=0)
    for i in range(4):
        assert np.linalg.norm(r_nodes[i] - expected[i]) < 5e-3, (
            f"node {i}: r={r_nodes[i]}, expected~{expected[i]}"
        )


@pytest.mark.parametrize("scale_deg", [1.0, 20.0, 60.0])
def test_convergence_is_fast(scale_deg):
    rng = np.random.default_rng(3)
    base = np.radians(scale_deg) * np.array([0.5, 0.1, -0.3])
    perturb = rng.normal(scale=np.radians(5.0), size=(4, 3))
    theta_nodes = base[None, :] + perturb

    theta_nodes = np.asarray(theta_nodes, dtype=float)
    R_nodes = [Shell4MITCCorotational._exp_map(theta_nodes[i]) for i in range(4)]
    R_e = Shell4MITCCorotational._exp_map(theta_nodes.mean(axis=0))
    n_iter = 0
    for n_iter in range(1, 21):
        r_nodes = np.array([
            Shell4MITCCorotational._log_map(R_e.T @ R_nodes[i]) for i in range(4)
        ])
        correction = r_nodes.mean(axis=0)
        if np.linalg.norm(correction) < 1e-13:
            break
        R_e = R_e @ Shell4MITCCorotational._exp_map(correction)

    assert n_iter <= 6, f"mean-rotation iteration took {n_iter} steps at scale {scale_deg}deg"


def test_returns_orthonormal_rotation_matrix():
    rng = np.random.default_rng(42)
    theta_nodes = rng.normal(scale=0.3, size=(4, 3))
    R_e, _ = _mean(theta_nodes)
    assert np.linalg.norm(R_e @ R_e.T - np.eye(3)) < 1e-10
    assert abs(np.linalg.det(R_e) - 1.0) < 1e-10
