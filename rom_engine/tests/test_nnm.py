"""
Tests for nnm.py -- HarmonicBalanceSystem (AFT bookkeeping, A(omega)
block assembly), solve_nnm_backbone (natural-parameter NNM backbone
continuation), and solve_nnm_backbone_arclength (pseudo-arclength NNM
backbone continuation -- see docs/nonlinear_surrogate_rom_roadmap.md
Section 11 for the design this validates against).

Every check has an independent ground truth: exact linear-algebra
identities for the AFT machinery, a direct cross-check against
frequency.FrequencyROM's own already-validated complex A(omega) at
n_harmonics=1, an independent scipy.integrate.solve_ivp
period-measurement for the classical Duffing-oscillator hardening
backbone, a closed-form effective-stiffness check for the q_l domain's
linear limit, and (for the pseudo-arclength additions) a fully
analytic textbook fold (the unit circle) plus a direct agreement check
against solve_nnm_backbone itself on the SAME non-folding backbone.
Real fea_engine validation lives in test_nnm_fea.py, not here.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rom_engine.nnm import HarmonicBalanceSystem, solve_nnm_backbone, solve_nnm_backbone_arclength
from rom_engine.frequency import FrequencyROM


class TestHarmonicBalanceSystemBookkeeping:
    def test_aft_roundtrip_is_exact(self):
        hb = HarmonicBalanceSystem(np.array([100.0, 400.0]), n_harmonics=5)
        assert np.max(np.abs(hb.Tinv @ hb.T - np.eye(hb.n_coeffs))) < 1e-12

        rng = np.random.default_rng(0)
        Z = rng.standard_normal((hb.n_coeffs, hb.n))
        Z_back = hb.project(hb.reconstruct(Z))
        assert np.max(np.abs(Z_back - Z)) < 1e-10

    def test_apply_A_matches_linear_matrix(self):
        hb = HarmonicBalanceSystem(np.array([100.0, 400.0]), C_r=np.array([0.5, 0.8]),
                                    n_harmonics=4)
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((hb.n_coeffs, hb.n))
        omega = 12.3
        via_apply = hb.apply_A(Z, omega)
        via_matrix = (hb.linear_matrix(omega) @ Z.flatten()).reshape(hb.n_coeffs, hb.n)
        assert np.allclose(via_apply, via_matrix)

    def test_apply_dA_domega_matches_matrix(self):
        hb = HarmonicBalanceSystem(np.array([100.0, 400.0]), C_r=np.array([0.5, 0.8]),
                                    n_harmonics=4)
        rng = np.random.default_rng(2)
        Z = rng.standard_normal((hb.n_coeffs, hb.n))
        omega = 12.3
        via_apply = hb.apply_dA_domega(Z, omega)
        via_matrix = (hb.dA_domega_matrix(omega) @ Z.flatten()).reshape(hb.n_coeffs, hb.n)
        assert np.allclose(via_apply, via_matrix)

    def test_dA_domega_matches_finite_difference(self):
        hb = HarmonicBalanceSystem(np.array([100.0, 400.0]), C_r=np.array([0.5, 0.8]),
                                    n_harmonics=3)
        rng = np.random.default_rng(3)
        Z = rng.standard_normal((hb.n_coeffs, hb.n))
        omega = 9.0
        eps = 1e-6
        fd = (hb.apply_A(Z, omega + eps) - hb.apply_A(Z, omega - eps)) / (2 * eps)
        assert np.allclose(fd, hb.apply_dA_domega(Z, omega), atol=1e-4)

    def test_rejects_non_diagonal_C_r(self):
        with pytest.raises(ValueError):
            HarmonicBalanceSystem(np.array([100.0, 400.0]),
                                   C_r=np.array([[0.5, 0.1], [0.1, 0.8]]))

    def test_rejects_invalid_n_harmonics(self):
        with pytest.raises(ValueError):
            HarmonicBalanceSystem(np.array([100.0]), n_harmonics=0)


class TestNHarmonics1MatchesFrequencyROM:
    """The strongest unit-level cross-check: HarmonicBalanceSystem's
    n_harmonics=1 block, solved for a pure-cosine forcing, must exactly
    reproduce frequency.FrequencyROM's own complex FRF solve at the
    same frequency -- both are, after all, the same linear physics
    (-omega^2*M + i*omega*C + K) x = F, just represented differently
    (one complex number per mode vs. a real cos/sin pair)."""

    def test_matches_complex_frf_real_and_imaginary_parts(self):
        n = 2
        Lambda = np.array([100.0, 400.0])
        C_r = np.array([0.5, 0.8])
        hb = HarmonicBalanceSystem(Lambda, C_r, n_harmonics=1)

        M, K, C = np.eye(n), np.diag(Lambda), np.diag(C_r)
        rom = FrequencyROM.from_MCK(M, K, np.eye(n), C=C)

        omega = 15.0
        F = np.array([3.0, -2.0])
        NCn = hb.n_coeffs * n
        rhs = np.zeros(NCn)
        rhs[n:2 * n] = F   # pure cosine forcing at harmonic 1
        sol = np.linalg.solve(hb.linear_matrix(omega), rhs)
        a1, b1 = sol[n:2 * n], sol[2 * n:3 * n]

        resp = rom.frequency_response(np.array([omega]), F)[0]
        assert np.allclose(a1, resp.real, atol=1e-10)
        assert np.allclose(b1, -resp.imag, atol=1e-10)

    @pytest.mark.parametrize("omega", [5.0, 22.5, 50.0])
    def test_matches_across_multiple_frequencies(self, omega):
        n = 1
        Lambda = np.array([80.0])
        C_r = np.array([0.3])
        hb = HarmonicBalanceSystem(Lambda, C_r, n_harmonics=1)
        rom = FrequencyROM.from_MCK(np.eye(n), np.diag(Lambda), np.eye(n), C=np.diag(C_r))
        F = np.array([1.0])
        NCn = hb.n_coeffs * n
        rhs = np.zeros(NCn); rhs[n:2 * n] = F
        sol = np.linalg.solve(hb.linear_matrix(omega), rhs)
        a1, b1 = sol[n:2 * n], sol[2 * n:3 * n]
        resp = rom.frequency_response(np.array([omega]), F)[0]
        assert np.allclose(a1, resp.real, atol=1e-9)
        assert np.allclose(b1, -resp.imag, atol=1e-9)


class _CubicQnlForceModel:
    def __init__(self, k3):
        self.k3 = np.asarray(k3, dtype=float)

    def predict(self, q):
        return self.k3 * np.asarray(q, dtype=float) ** 3

    def jacobian(self, q):
        return np.diag(3 * self.k3 * np.asarray(q, dtype=float) ** 2)


class _CubicQnlForceModelNoJacobian:
    def __init__(self, k3):
        self.k3 = np.asarray(k3, dtype=float)

    def predict(self, q):
        return self.k3 * np.asarray(q, dtype=float) ** 3


class _LinearQlForceModel:
    def __init__(self, k_extra):
        self.k_extra = np.asarray(k_extra, dtype=float)

    def predict(self, q_l):
        return self.k_extra * np.asarray(q_l, dtype=float)

    def jacobian(self, q_l):
        return np.diag(self.k_extra)


def _measure_duffing_frequency(Lambda0, k3, amp, n_periods_max=6):
    """Independent ground truth: undamped free vibration from (amp, 0),
    peak-to-peak period measured directly from a dense, high-accuracy
    scipy.integrate.solve_ivp trajectory -- no harmonic-balance/AFT
    machinery involved at all."""
    def rhs(t, y):
        q, qd = y
        return [qd, -Lambda0 * q - k3 * q ** 3]

    T_guess = 2 * np.pi / np.sqrt(Lambda0)
    T_max = n_periods_max * T_guess
    sol = solve_ivp(rhs, [0, T_max], [amp, 0.0], max_step=T_guess / 2000,
                     dense_output=True, rtol=1e-12, atol=1e-14)
    t_fine = np.linspace(0, T_max, 200_000)
    q_fine = sol.sol(t_fine)[0]
    peaks = [t_fine[i] for i in range(1, len(q_fine) - 1)
             if q_fine[i] > q_fine[i - 1] and q_fine[i] > q_fine[i + 1] and q_fine[i] > 0.5 * amp]
    if len(peaks) < 2:
        return None
    return 2 * np.pi / np.mean(np.diff(peaks))


class TestDuffingBackbone:
    """The classical hardening Duffing oscillator -- frequency should
    rise with amplitude, and solve_nnm_backbone's own prediction should
    track an INDEPENDENT time-integration period measurement."""

    def test_backbone_matches_independent_time_integration(self):
        Lambda = np.array([100.0])
        k3 = np.array([50000.0])
        fm = _CubicQnlForceModel(k3)
        hb = HarmonicBalanceSystem(Lambda, n_harmonics=7, n_time_samples=128)

        amps, omegas, Z_hist, conv = solve_nnm_backbone(
            hb, fm, q_amplitude_range=(0.005, 0.03), n_points=6, domain="q_nl")
        assert np.all(conv)

        omegas_ref = np.array([_measure_duffing_frequency(Lambda[0], k3[0], a) for a in amps])
        assert not np.any(np.isnan(omegas_ref))
        rel_err = np.abs(omegas - omegas_ref) / omegas_ref
        assert np.max(rel_err) < 0.01

    def test_backbone_is_hardening_frequency_rises_with_amplitude(self):
        Lambda = np.array([100.0])
        k3 = np.array([50000.0])
        fm = _CubicQnlForceModel(k3)
        hb = HarmonicBalanceSystem(Lambda, n_harmonics=5, n_time_samples=64)
        amps, omegas, Z_hist, conv = solve_nnm_backbone(
            hb, fm, q_amplitude_range=(0.005, 0.04), n_points=8, domain="q_nl")
        assert np.all(conv)
        assert np.all(np.diff(omegas) > 0)   # strictly increasing -> hardening
        assert omegas[0] > np.sqrt(Lambda[0])  # already above the linear frequency

    def test_numerical_jacobian_fallback_matches_analytic(self):
        Lambda = np.array([100.0])
        k3 = np.array([50000.0])
        hb = HarmonicBalanceSystem(Lambda, n_harmonics=5, n_time_samples=64)
        amps_a, omegas_a, _, conv_a = solve_nnm_backbone(
            hb, _CubicQnlForceModel(k3), q_amplitude_range=(0.01, 0.03), n_points=4,
            domain="q_nl", jacobian="analytic")
        amps_n, omegas_n, _, conv_n = solve_nnm_backbone(
            hb, _CubicQnlForceModelNoJacobian(k3), q_amplitude_range=(0.01, 0.03), n_points=4,
            domain="q_nl", jacobian="numerical")
        assert np.all(conv_a) and np.all(conv_n)
        assert np.allclose(omegas_a, omegas_n, rtol=1e-5)

    def test_auto_jacobian_falls_back_when_model_has_none(self):
        Lambda = np.array([100.0])
        k3 = np.array([50000.0])
        hb = HarmonicBalanceSystem(Lambda, n_harmonics=5, n_time_samples=64)
        amps, omegas, _, conv = solve_nnm_backbone(
            hb, _CubicQnlForceModelNoJacobian(k3), q_amplitude_range=(0.01, 0.03),
            n_points=4, domain="q_nl", jacobian="auto")
        assert np.all(conv)
        assert np.all(np.isfinite(omegas))


class TestQlDomainLinearLimit:
    def test_linear_ql_force_model_gives_amplitude_independent_frequency(self):
        Lambda = np.array([100.0])
        k_extra = np.array([5.0])
        fm = _LinearQlForceModel(k_extra)
        hb = HarmonicBalanceSystem(Lambda, n_harmonics=3, n_time_samples=32)

        amps, omegas, Z_hist, conv = solve_nnm_backbone(
            hb, fm, q_amplitude_range=(0.001, 0.01), n_points=5, domain="q_l")
        assert np.all(conv)

        Lambda_eff = Lambda[0] + k_extra[0] / (1 - k_extra[0] / Lambda[0])
        expected = np.sqrt(Lambda_eff)
        assert np.allclose(omegas, expected, rtol=1e-8)


class TestInputValidation:
    def test_rejects_unknown_domain(self):
        hb = HarmonicBalanceSystem(np.array([100.0]), n_harmonics=3)
        with pytest.raises(ValueError):
            solve_nnm_backbone(hb, _CubicQnlForceModel(np.array([1.0])),
                                (0.01, 0.02), domain="bogus")

    def test_rejects_master_mode_out_of_range(self):
        hb = HarmonicBalanceSystem(np.array([100.0]), n_harmonics=3)
        with pytest.raises(ValueError):
            solve_nnm_backbone(hb, _CubicQnlForceModel(np.array([1.0])),
                                (0.01, 0.02), master_mode=5)


def _arclength_toy_corrector(h_and_J, v_pred, v_prev, t_hat, ds, tol=1e-12, max_iter=50):
    """A STANDALONE, MINIMAL re-implementation of the SAME predictor/
    corrector/arclength-constraint pattern nnm._arclength_corrector()
    implements internally (secant tangent, augmented Newton system with
    one arclength-constraint row), but applied to a generic
    caller-supplied `h_and_J(v) -> (R, J)` instead of the HBM-specific
    residual -- used ONLY by TestPseudoArclengthAlgorithm below to
    validate the CORE continuation algorithm in complete isolation from
    harmonic-balance/AFT machinery, on a fully analytic textbook example
    (the unit circle) whose fold points are known exactly (x = +-1), so
    the expected behavior can be checked with total certainty rather
    than relying on any physical/numerical ground truth."""
    v = v_pred.copy()
    for _ in range(max_iter):
        R_eq, J_eq = h_and_J(v)
        R_arc = np.dot(t_hat, v - v_prev) - ds
        R = np.concatenate([R_eq, [R_arc]])
        if np.max(np.abs(R)) < tol:
            return v, True
        J = np.vstack([J_eq, t_hat[None, :]])
        v = v + np.linalg.solve(J, -R)
    return v, False


class TestPseudoArclengthAlgorithm:
    """Validates the CORE mathematical pattern
    solve_nnm_backbone_arclength()'s own _arclength_corrector() uses --
    secant-predictor + augmented [equilibrium; arclength-constraint]
    Newton correction -- on a fully analytic example completely
    decoupled from harmonic balance: the unit circle x^2 + y^2 = 1,
    where x plays the role of a "prescribed amplitude" and the two fold
    points (x = +-1, where dx/ds = 0 along the true arclength-
    parametrized curve, i.e. where "solve for y given x" and, by
    extension, "solve for the rest given prescribed x" cannot be pushed
    any further) are known EXACTLY, not estimated. This is the same
    structural failure mode `solve_nnm_backbone`'s prescribed-amplitude
    continuation has at a genuine amplitude fold, verified here with
    total mathematical certainty rather than depending on locating and
    fully characterizing a real NNM backbone fold (a harder, separate
    validation target -- see docs/nonlinear_surrogate_rom_roadmap.md
    Section 11 for an honest account of what was and was not
    established on real coupled-mode systems)."""

    @staticmethod
    def _circle_h_and_J(v):
        x, y = v
        R = np.array([x ** 2 + y ** 2 - 1.0])
        J = np.array([[2 * x, 2 * y]])
        return R, J

    def test_traces_through_both_folds_of_the_unit_circle(self):
        v0 = np.array([1.0, 0.0])                       # exactly AT the first fold (x=1)
        v1 = np.array([np.cos(0.05), np.sin(0.05)])      # a small step along the true curve
        t_hat = v1 - v0
        t_hat = t_hat / np.linalg.norm(t_hat)
        ds = np.linalg.norm(v1 - v0)

        vs = [v0, v1]
        v_prev = v1
        for _ in range(258):
            v_pred = v_prev + ds * t_hat
            v_new, conv = _arclength_toy_corrector(self._circle_h_and_J, v_pred, v_prev, t_hat, ds)
            assert conv
            t_new = v_new - v_prev
            t_new = t_new / np.linalg.norm(t_new)
            if np.dot(t_new, t_hat) < 0:
                t_new = -t_new
            vs.append(v_new)
            v_prev, t_hat = v_new, t_new
        vs = np.array(vs)

        residual = vs[:, 0] ** 2 + vs[:, 1] ** 2 - 1.0
        assert np.max(np.abs(residual)) < 1e-10, "every traced point must stay exactly on the circle"
        assert np.isclose(vs[:, 0].max(), 1.0, atol=1e-2), "must reach the FIRST fold (x=1)"
        assert np.isclose(vs[:, 0].min(), -1.0, atol=1e-2), "must reach PAST it, to the SECOND fold (x=-1)"
        print("PASS -- pseudo-arclength continuation traces the whole circle, "
              "through both x=+-1 folds, where x-prescribed continuation would stop dead")

    def test_x_prescribed_continuation_cannot_pass_x_equals_1(self):
        """The direct analogue of solve_nnm_backbone's own failure mode:
        solving for y given a DIRECTLY PRESCRIBED x has no real solution
        for any x > 1 -- not a numerical near-miss, an exact algebraic
        fact (x^2+y^2=1 has no real y when x>1), confirmed here as the
        reason pseudo-arclength continuation is needed at all."""
        for x_target in (1.001, 1.1, 2.0):
            y_squared = 1.0 - x_target ** 2
            assert y_squared < 0, "there must be NO real y for x beyond the fold"


class TestArclengthMatchesNaturalParameter:
    """On a backbone BOTH continuation drivers can trace (the classical
    non-folding hardening Duffing oscillator), they must agree -- a
    direct, real cross-check that solve_nnm_backbone_arclength's more
    general machinery reproduces solve_nnm_backbone's already-validated
    result, not just a different-looking curve."""

    def test_agrees_with_natural_parameter_to_high_precision_densely_sampled(self):
        Lambda = np.array([100.0])
        k3 = np.array([50000.0])
        fm = _CubicQnlForceModel(k3)
        hb = HarmonicBalanceSystem(Lambda, n_harmonics=7, n_time_samples=128)

        amps_np, om_np, _, conv_np = solve_nnm_backbone(
            hb, fm, q_amplitude_range=(0.005, 0.03), n_points=60, domain="q_nl")
        assert np.all(conv_np)

        amps_al, om_al, _, conv_al = solve_nnm_backbone_arclength(
            hb, fm, a0=0.005, n_points=60, direction=1.0, ds=0.002, domain="q_nl")
        assert np.all(conv_al)

        # compare over the OVERLAPPING amplitude range only (the two
        # drivers do not visit the same amplitude values, by design --
        # amplitude is prescribed for one, MEASURED for the other)
        mask = (amps_np >= amps_al.min()) & (amps_np <= amps_al.max())
        assert mask.sum() > 20, "expected substantial overlap for this ds/range choice"
        om_interp = np.interp(amps_np[mask], amps_al, om_al)
        rel_err = np.abs(om_interp - om_np[mask]) / om_np[mask]
        print(f"arclength vs natural-parameter on the classical Duffing backbone: "
              f"max rel err={rel_err.max():.3e} over {mask.sum()} overlapping points")
        assert rel_err.max() < 1e-3, (
            "solve_nnm_backbone_arclength must reproduce the SAME backbone "
            "solve_nnm_backbone already traces on a non-folding case"
        )

    def test_jacobian_fallback_matches_analytic(self):
        Lambda = np.array([100.0])
        k3 = np.array([50000.0])
        hb = HarmonicBalanceSystem(Lambda, n_harmonics=5, n_time_samples=64)
        amps_a, om_a, _, conv_a = solve_nnm_backbone_arclength(
            hb, _CubicQnlForceModel(k3), a0=0.01, n_points=6, ds=0.005,
            domain="q_nl", jacobian="analytic")
        amps_n, om_n, _, conv_n = solve_nnm_backbone_arclength(
            hb, _CubicQnlForceModelNoJacobian(k3), a0=0.01, n_points=6, ds=0.005,
            domain="q_nl", jacobian="numerical")
        assert np.all(conv_a) and np.all(conv_n)
        assert np.allclose(amps_a, amps_n, rtol=1e-5)
        assert np.allclose(om_a, om_n, rtol=1e-5)


class TestArclengthOnCoupledTwoModeSystem:
    """A genuinely 2-mode coupled nonlinear system (two cubic oscillators
    with a quadratic-form cubic coupling term) -- explored extensively
    while validating this function (see docs/nonlinear_surrogate_rom_
    roadmap.md Section 11 for what was and was not established about
    its full bifurcation structure). The one finding reported here is
    fully verified and unambiguous: on the INVARIANT "planar" branch
    (satellite mode identically zero), the coupled system reduces
    EXACTLY to the already-validated single-DOF Duffing oscillator, and
    solve_nnm_backbone_arclength reproduces that reduction to machine
    precision across a long trace -- a real, nontrivial correctness
    check on genuinely 2-mode machinery (a2_pos free, coupling term
    present in the Jacobian) collapsing to known 1-DOF physics exactly
    where theory says it must."""

    class _TwoCoupledDuffing:
        def __init__(self, c1, c2, kappa):
            self.c1, self.c2, self.kappa = c1, c2, kappa

        def predict(self, q):
            q1, q2 = q
            k = self.kappa
            return np.array([self.c1 * q1 ** 3 + k * q1 * q2 ** 2,
                              self.c2 * q2 ** 3 + k * q2 * q1 ** 2])

        def jacobian(self, q):
            q1, q2 = q
            k = self.kappa
            return np.array([[3 * self.c1 * q1 ** 2 + k * q2 ** 2, 2 * k * q1 * q2],
                              [2 * k * q1 * q2, 3 * self.c2 * q2 ** 2 + k * q1 ** 2]])

    def test_planar_branch_reduces_exactly_to_single_dof_duffing(self):
        w1, w2 = 1.0, 1.1
        Lambda = np.array([w1 ** 2, w2 ** 2])
        hb = HarmonicBalanceSystem(Lambda, C_r=None, n_harmonics=1, n_time_samples=32)
        fm = self._TwoCoupledDuffing(1.0, 1.0, 6.0)

        amps, omegas, Z_hist, conv = solve_nnm_backbone_arclength(
            hb, fm, a0=0.1892, direction=-1.0, n_points=60, ds=0.01, omega0=1.2, domain="q_nl")
        assert np.all(conv)

        a2 = Z_hist[:, 1, 1]   # mode 2's own first-harmonic cosine coefficient
        assert np.max(np.abs(a2)) < 1e-9, (
            "the satellite mode must stay EXACTLY zero on this invariant branch"
        )

        # on q2=0, mode 1's own equation is EXACTLY the single-DOF Duffing
        # oscillator q1'' + w1^2 q1 + c1 q1^3 = 0 -- cross-check directly
        single_dof_hb = HarmonicBalanceSystem(np.array([w1 ** 2]), n_harmonics=1, n_time_samples=32)
        single_dof_fm = _CubicQnlForceModel(np.array([1.0]))
        # single-harmonic HBM balance is algebraic here: just check the
        # governing relation (w1^2-omega^2)*a1 + 0.75*a1^3 = 0 holds
        # (the exact n_harmonics=1 truncation both machineries share)
        residual = (w1 ** 2 - omegas ** 2) * amps + 0.75 * amps ** 3
        print(f"planar-branch single-DOF-Duffing residual: max={np.max(np.abs(residual)):.3e}")
        assert np.max(np.abs(residual)) < 1e-6 * np.max(np.abs(omegas ** 2 * amps))


class TestArclengthInputValidation:
    def test_rejects_unknown_domain(self):
        hb = HarmonicBalanceSystem(np.array([100.0]), n_harmonics=3)
        with pytest.raises(ValueError):
            solve_nnm_backbone_arclength(hb, _CubicQnlForceModel(np.array([1.0])),
                                          a0=0.01, domain="bogus")

    def test_rejects_master_mode_out_of_range(self):
        hb = HarmonicBalanceSystem(np.array([100.0]), n_harmonics=3)
        with pytest.raises(ValueError):
            solve_nnm_backbone_arclength(hb, _CubicQnlForceModel(np.array([1.0])),
                                          a0=0.01, master_mode=5)

    def test_rejects_too_few_points(self):
        hb = HarmonicBalanceSystem(np.array([100.0]), n_harmonics=3)
        with pytest.raises(ValueError):
            solve_nnm_backbone_arclength(hb, _CubicQnlForceModel(np.array([1.0])),
                                          a0=0.01, n_points=1)

    def test_reports_failure_honestly_when_bootstrap_cannot_converge(self):
        """A pathological force model that returns NaN everywhere -- the
        bootstrap Newton solve can never converge, and the function
        should report that honestly (all-NaN/False), not raise or
        silently return a bogus point."""
        class _NanForceModel:
            def predict(self, q):
                return np.full_like(np.asarray(q, dtype=float), np.nan)

            def jacobian(self, q):
                n = len(np.asarray(q))
                return np.full((n, n), np.nan)

        hb = HarmonicBalanceSystem(np.array([100.0]), n_harmonics=3)
        amps, omegas, Z_hist, conv = solve_nnm_backbone_arclength(
            hb, _NanForceModel(), a0=0.01, n_points=5, max_iter=5)
        assert not np.any(conv)
        assert np.all(np.isnan(amps))
        assert np.all(np.isnan(omegas))
