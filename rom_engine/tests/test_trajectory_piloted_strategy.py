"""
test_trajectory_piloted_strategy.py -- Wave 12 item 113 (docs/
consolidated_future_roadmap.md, ICE-ROM/GAP_ANALYSIS.md gap #2):
validates `_farthest_point_subsample()` and `TrajectoryPilotedStrategy`.

Uses the SAME hand-built cubic-spring synthetic "full order model" as
`test_nonlinear_rom.py`'s own `AppliedLoadStrategy`/
`EnforcedDisplacementStrategy` tests (`_SyntheticCubicSpringFOM`),
re-declared locally to keep this file self-contained -- a known,
closed-form nonlinear FOM lets every check here be exact, not just
plausible, with no FE package involved.

Real-fea_engine validation (a genuine flat clamped-clamped beam,
exploratory dynamic trajectory generated via
`fea_engine.nonlinear_solver.solve_nonlinear_transient`, training data
built by this strategy from the visited states) was run by hand during
development against the real fixture used elsewhere in this suite
(`fea_fixtures.clamped_clamped_nonlinear_beam_system`) -- mechanically
correct and substantially cheaper per training point than
`AppliedLoadStrategy` (one internal-force evaluation, no Newton solve,
per selected point). The SPECIFIC divergence failure mode
`GAP_ANALYSIS.md` gap #2 documents (a 4-retained-mode fit trained on
`AppliedLoadStrategy` data blowing up under dynamic driving despite an
excellent static fit) did not reproduce cleanly at a sandbox-tractable
2-3-mode/short-trajectory scale in that exploration -- both training
strategies produced comparably stable fits there under the Newton-free
correction modes, and both diverged under the SAME Newton-free modes
in the same run, an effect that tracked the correction SCHEME more than
the training-data SOURCE at this reduced scale. That specific dynamic-
stability comparison is therefore not asserted here as an automated,
deterministic test (it did not reproduce reliably at reduced scale, so
baking it in would be a flaky, not a decisive, check) -- what IS tested
below, and is fully decisive, is that this class does exactly what it
claims to do: correctly subsample a visited trajectory and correctly
reuse EnforcedDisplacementStrategy's own already-validated static
solve-and-project mechanism at the selected points.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from rom_engine.nonlinear_rom import (
    _farthest_point_subsample, TrajectoryPilotedStrategy, EnforcedDisplacementStrategy,
)


class _SyntheticCubicSpringFOM:
    """theta(u) = k3 * u^3, mass-normalized (Lambda already diagonal in
    this basis) -- SAME synthetic FOM test_nonlinear_rom.py's own
    AppliedLoadStrategy/EnforcedDisplacementStrategy tests use."""

    def __init__(self, Lambda, k3):
        self.Lambda = Lambda
        self.k3 = k3

    def solve_enforced_displacement(self, u_target):
        return self.Lambda * u_target + self.k3 * u_target ** 3


# =====================================================================
# _farthest_point_subsample
# =====================================================================
class TestFarthestPointSubsample:
    def test_returns_all_indices_when_n_target_exceeds_pool(self):
        pts = np.random.default_rng(0).standard_normal((10, 3))
        idx = _farthest_point_subsample(pts, 50, np.random.default_rng(1))
        assert sorted(idx.tolist()) == list(range(10))

    def test_returns_exactly_n_target_unique_indices(self):
        pts = np.random.default_rng(0).standard_normal((300, 2))
        idx = _farthest_point_subsample(pts, 25, np.random.default_rng(1))
        assert len(idx) == 25
        assert len(set(idx.tolist())) == 25
        assert np.all((idx >= 0) & (idx < 300))

    def test_selected_points_are_well_spread_vs_random_subset(self):
        # a well-designed maximin subset should have a LARGER minimum
        # pairwise distance than a plain random subset of the same size
        # -- the actual space-filling property this function exists for.
        rng = np.random.default_rng(0)
        pts = rng.uniform(-1, 1, size=(500, 2))
        idx_fps = _farthest_point_subsample(pts, 30, np.random.default_rng(1))
        idx_rand = np.random.default_rng(2).choice(500, size=30, replace=False)

        from scipy.spatial.distance import pdist
        min_dist_fps = pdist(pts[idx_fps]).min()
        min_dist_rand = pdist(pts[idx_rand]).min()
        assert min_dist_fps > min_dist_rand

    def test_deterministic_given_same_rng_state(self):
        pts = np.random.default_rng(0).standard_normal((100, 2))
        idx1 = _farthest_point_subsample(pts, 10, np.random.default_rng(42))
        idx2 = _farthest_point_subsample(pts, 10, np.random.default_rng(42))
        assert np.array_equal(idx1, idx2)


# =====================================================================
# TrajectoryPilotedStrategy
# =====================================================================
class TestTrajectoryPilotedStrategy:
    def _setup(self):
        n_modes = 2
        Lambda = np.array([100.0, 64.0])
        k3 = np.array([5.0, 8.0])
        fom = _SyntheticCubicSpringFOM(Lambda, k3)
        V = np.eye(n_modes)
        basis_freqs_hz = np.sqrt(Lambda) / (2 * np.pi)
        return fom, V, basis_freqs_hz, Lambda, k3

    def test_generate_returns_correctly_shaped_triple(self):
        fom, V, freqs, Lambda, k3 = self._setup()
        rng = np.random.default_rng(0)
        trajectory = rng.uniform(-0.5, 0.5, size=(200, 2))
        strat = TrajectoryPilotedStrategy(n_target=20, rng=np.random.default_rng(1))
        q_l, q_nl, F_nl = strat.generate(V, freqs, trajectory, fom.solve_enforced_displacement)
        assert q_l.shape == (20, 2)
        assert q_nl.shape == (20, 2)
        assert F_nl.shape == (20, 2)

    def test_matches_enforced_displacement_strategy_at_the_same_target_points(self):
        # decisive correctness check: TrajectoryPilotedStrategy reuses
        # EnforcedDisplacementStrategy's own per-sample math exactly --
        # feeding it the SAME target q_nl points EnforcedDisplacementStrategy
        # would have prescribed (rather than a trajectory to subsample
        # from) must give BIT-FOR-BIT identical q_l/q_nl/F_nl.
        fom, V, freqs, Lambda, k3 = self._setup()
        targets = np.array([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.5], [0.0, 0.2]])

        strat = TrajectoryPilotedStrategy(n_target=4, rng=np.random.default_rng(0))
        q_l, q_nl, F_nl = strat.generate(V, freqs, targets, fom.solve_enforced_displacement)

        # n_target == n_visited -> _farthest_point_subsample returns ALL
        # indices in original order (no thinning needed), so q_nl should
        # equal `targets` exactly, in the same order.
        assert np.array_equal(q_nl, targets)

        # cross-check against EnforcedDisplacementStrategy's own formula,
        # evaluated directly (not through OLHS sampling -- computed by
        # hand from the same Lambda/solve_enforced_displacement contract)
        F_reaction = np.array([fom.solve_enforced_displacement(t) for t in targets])
        modal_force = F_reaction   # V=I here, so V.T @ F_reaction == F_reaction
        q_l_expected = modal_force / Lambda
        F_nl_expected = modal_force - Lambda * targets
        assert np.allclose(q_l, q_l_expected, atol=1e-12)
        assert np.allclose(F_nl, F_nl_expected, atol=1e-12)

    def test_accepts_list_of_trajectories(self):
        fom, V, freqs, Lambda, k3 = self._setup()
        rng = np.random.default_rng(0)
        traj1 = rng.uniform(-0.3, 0.3, size=(50, 2))
        traj2 = rng.uniform(-0.3, 0.3, size=(80, 2))
        strat = TrajectoryPilotedStrategy(n_target=15, rng=np.random.default_rng(1))
        q_l, q_nl, F_nl = strat.generate(V, freqs, [traj1, traj2], fom.solve_enforced_displacement)
        assert q_nl.shape == (15, 2)
        # every selected point genuinely came from the concatenated pool
        pool = np.concatenate([traj1, traj2], axis=0)
        for row in q_nl:
            assert np.any(np.all(np.isclose(pool, row), axis=1))

    def test_downstream_polynomial_rom_fits_trajectory_piloted_data(self):
        from rom_engine.nonlinear_rom import PolynomialModalROM
        fom, V, freqs, Lambda, k3 = self._setup()
        rng = np.random.default_rng(0)
        trajectory = rng.uniform(-0.6, 0.6, size=(300, 2))
        strat = TrajectoryPilotedStrategy(n_target=40, rng=np.random.default_rng(1))
        q_l, q_nl, F_nl = strat.generate(V, freqs, trajectory, fom.solve_enforced_displacement)
        model = PolynomialModalROM(n_modes=2).fit(q_nl, F_nl)
        pred = model.force(q_nl)
        r2 = 1 - np.sum((F_nl - pred) ** 2) / np.sum((F_nl - F_nl.mean(axis=0)) ** 2)
        assert r2 > 0.999

    def test_rejects_wrong_n_modes(self):
        fom, V, freqs, Lambda, k3 = self._setup()
        bad_trajectory = np.zeros((10, 3))   # V has 2 modes, not 3
        strat = TrajectoryPilotedStrategy(n_target=5, rng=np.random.default_rng(0))
        with pytest.raises(ValueError):
            strat.generate(V, freqs, bad_trajectory, fom.solve_enforced_displacement)

    def test_fewer_visited_points_than_n_target_returns_all_of_them(self):
        fom, V, freqs, Lambda, k3 = self._setup()
        trajectory = np.array([[0.1, 0.1], [0.2, -0.1], [-0.1, 0.3]])
        strat = TrajectoryPilotedStrategy(n_target=100, rng=np.random.default_rng(0))
        q_l, q_nl, F_nl = strat.generate(V, freqs, trajectory, fom.solve_enforced_displacement)
        assert q_nl.shape == (3, 2)
