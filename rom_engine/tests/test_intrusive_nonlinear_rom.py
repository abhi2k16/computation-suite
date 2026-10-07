"""
test_intrusive_nonlinear_rom.py -- validates rom_engine.intrusive_
nonlinear_rom.IntrusiveNonlinearROM (Wave 17 item 144) against real
fea_engine models, per the roadmap's own four-part validation criteria:

  1. Identity-basis reduction reproduces fea_engine.nonlinear_solver.
     solve_nonlinear_transient()'s own trajectory to integrator
     tolerance -- the strongest check available: it directly validates
     that f_int_r(q) = V^T @ internal_force_fn(V@q) is mathematically
     consistent with the full-order system when there's no actual
     reduction happening. Built on a real, genuinely nonlinear
     fea_engine.elements.Beam2DReissner cantilever (item 140), not a
     toy linear system.
  2. In the linear regime (an artificially-linearized internal_force_fn),
     this class's own K_r/M_r and the resulting modal frequencies match
     rom_engine.galerkin.GalerkinROM's -- confirms the new intrusive
     machinery doesn't silently break agreement with the package's
     existing linear intrusive ROM -- plus a genuine dynamic check: a
     single GalerkinROM eigenmode's own analytic linear-oscillator
     trajectory is reproduced by IntrusiveNonlinearROM's RK4 integrator.
  3. Undamped RK4 integration conserves total reduced energy
     (0.5*qdot^T M_r qdot + strain energy, both computed on the SAME
     genuinely nonlinear Beam2DReissner model) to O(dt^4) -- measured as
     a real convergence ratio across three step sizes, not just "small
     at one dt".
  4. M_r/D_r/K_r come out symmetric positive definite (an eigenvalue
     check, not just a symmetry check) for a genuinely SPD/PSD
     full-order system.
"""
import numpy as np
import pytest
from scipy.linalg import qr, eigh

from rom_engine import GalerkinROM, IntrusiveNonlinearROM
from fea_engine import nonlinear_solver as nls
from fea_engine.damping import RayleighDamping
from fea_engine.elements.beams import Beam2DReissner

import fea_fixtures as ff


class _ZeroLoad:
    """force_at(t, n_dof, npn) -> zeros -- a free-decay (F_ext=0)
    load, same convention as test_nonlinear_dynamics_fea.py's own
    _ZeroLoad."""
    def force_at(self, t, n_dof, npn):
        return np.zeros(n_dof)


def _embed(u_free, free, n_dof):
    u = np.zeros(n_dof)
    u[free] = u_free
    return u


def _static_tip_deflection(fes, mat, free, n_elem, force, n_steps=6):
    """Ramp a transverse tip force up via fea_engine's own already-
    validated solve_nonlinear_static(), returning the free-dof
    displacement at the final (real, genuinely nonlinear) equilibrium
    -- used as a non-trivial initial condition for the free-decay
    checks below (a released-from-rest linear IC would not exercise
    the nonlinearity at all)."""
    fes.F[:] = 0.0
    tip_dof = 3 * n_elem + 1   # u2 (transverse) dof at the tip node
    fes.F[tip_dof] = force
    _, U_hist = nls.solve_nonlinear_static(fes, mat, n_steps=n_steps, tol=1e-10, max_iter=40)
    fes.F[:] = 0.0
    return U_hist[-1][free]


def _strain_energy_full(fes, mat, u_full):
    """Exact elastic strain energy of the (genuinely nonlinear)
    Beam2DReissner assembly at a given full-order state --
    0.5*L0*(EA*eps^2 + kappa_s*GA*gam^2 + EI*kap^2) per element, summed
    -- the scalar potential internal_force() is the analytic gradient
    of (N=EA*eps, Q=kappa_s*GA*gam, M=EI*kap are exactly linear-elastic
    stress resultants in these strain measures, and the local->global
    transform T is a fixed rotation, hence energy-preserving). Used
    ONLY by the energy-conservation check below (the class itself never
    assumes a potential exists for an arbitrary internal_force_fn --
    see IntrusiveNonlinearROM.energy()'s own docstring)."""
    E, G, A, I, kappa_s = mat
    total = 0.0
    for _name, _formulation, connectivity in fes._blocks:
        for elem_conn in connectivity:
            elem_coords = fes.mesh.nodes[elem_conn]
            g = fes._global_dofs(elem_conn)
            u_elem = u_full[g]
            L0, _T, q_local = Beam2DReissner._local_dofs(elem_coords, u_elem)
            _cphi, _sphi, eps, gam, kap = Beam2DReissner._strain_measures(L0, q_local)
            total += 0.5 * L0 * (E * A * eps ** 2 + kappa_s * G * A * gam ** 2 + E * I * kap ** 2)
    return total


# =====================================================================
# CHECK 1 -- identity-basis reduction reproduces the full-order
# fea_engine nonlinear transient trajectory.
# =====================================================================
class TestIdentityBasisReproducesFullOrder:
    @pytest.fixture(scope="class")
    def setup(self):
        # A small element count: Beam2DReissner's shear/rotary-inertia
        # stiffness ratio puts its HIGHEST natural frequency around
        # 15-18 kHz regardless of mesh refinement (a material-property
        # effect, not a discretization one -- confirmed directly by
        # comparing n_elem=2 vs n_elem=4 spectra), which sets the
        # explicit RK4 stability limit (dt < ~2.83/omega_max) far below
        # what the LOWEST/fundamental mode alone would suggest. n_elem=2
        # keeps the free-dof count (and hence the full-order Newton
        # cost at that small dt) small while still giving a genuinely
        # nonlinear identity-basis system to compare.
        fx = ff.reissner_cantilever_system(n_elem=2, L=1.0)
        fes, mat, free = fx["sys"], fx["mat"], fx["free_dofs"]
        n_dof, n_free = fx["n_dof"], len(free)
        fes.assemble_damping(RayleighDamping(alpha=0.0, beta=0.0))  # C=0, but defined
        M_ff = fx["M"][np.ix_(free, free)]
        C_ff = fes.C[np.ix_(free, free)]

        def internal_force_fn(u_free):
            return fes.assemble_internal_force(_embed(u_free, free, n_dof), mat)[free]

        def tangent_fn(u_free):
            return fes.assemble_tangent_stiffness(_embed(u_free, free, n_dof), mat)[np.ix_(free, free)]

        def load_fn(t):
            return np.zeros(n_free)

        V = np.eye(n_free)
        rom = IntrusiveNonlinearROM(V, M_ff, C_ff, internal_force_fn, load_fn, tangent_fn=tangent_fn)

        u0_free = _static_tip_deflection(fes, mat, free, fx["n_elem"], force=4.0e3)

        freq_hz_all, _ = fes.solve_modal(n_modes=n_free)
        freq_max = freq_hz_all[-1]
        # dt at 1/40 of the STIFFEST mode's own period -- a comfortable
        # margin inside RK4's imaginary-axis stability bound
        # (dt*omega_max <~ 2.83; here dt*omega_max = 2*pi/40 = 0.157).
        dt = (1.0 / freq_max) / 40.0
        T1 = 1.0 / freq_hz_all[0]
        n_steps = int(round(0.5 * T1 / dt))   # half a fundamental period
        T_total = n_steps * dt

        q0, qdot0 = rom.initial_conditions(u0_full=u0_free, v0_full=np.zeros(n_free))
        t_r, q_hist, qdot_hist = rom.integrate_rk4(q0, qdot0, dt, n_steps)

        t_fe, U_hist_fe = nls.solve_nonlinear_transient(
            fes, mat, _ZeroLoad(), T_total, dt, u0=_embed(u0_free, free, n_dof),
            tol=1e-7, max_iter=40)

        return {"q_hist": q_hist, "U_free_fe": U_hist_fe[:, free], "u0_free": u0_free}

    def test_q0_equals_u0_exactly(self, setup):
        # V = I => M_r = M_ff => q0 = M_r^-1 V^T M u0 = u0 exactly.
        assert np.allclose(setup["q_hist"][0], setup["u0_free"], atol=1e-12)

    def test_trajectory_matches_full_order(self, setup):
        q_hist, U_fe = setup["q_hist"], setup["U_free_fe"]
        scale = np.max(np.abs(U_fe))
        err = np.max(np.abs(q_hist - U_fe)) / scale
        print(f"identity-basis vs full-order (RK4 vs implicit Newmark-Newton) "
              f"max relative error: {err:.3e}")
        assert np.all(np.isfinite(q_hist))
        assert err < 5e-3


# =====================================================================
# CHECK 2 -- linear limit matches GalerkinROM.
# =====================================================================
class TestLinearLimitMatchesGalerkinROM:
    @pytest.fixture(scope="class")
    def setup(self):
        fx = ff.reissner_cantilever_system(n_elem=4, L=1.0)
        free = fx["free_dofs"]
        n_free = len(free)
        K0_ff = fx["K"][np.ix_(free, free)]
        M_ff = fx["M"][np.ix_(free, free)]

        rng = np.random.default_rng(3)
        r = min(4, n_free)
        Q, _ = qr(rng.standard_normal((n_free, r)))
        V = Q[:, :r]   # orthonormal but NOT mass-orthonormal -> M_r genuinely non-diagonal

        galerkin = GalerkinROM(V).reduce_system(K0_ff, M=M_ff)
        freq_hz_g, _mode_shapes_full_g, eigvecs_r_g = galerkin.solve_modal()

        rom = IntrusiveNonlinearROM(
            V, M_ff, np.zeros_like(M_ff),
            internal_force_fn=lambda u: K0_ff @ u,
            load_fn=lambda t: np.zeros(n_free),
            tangent_fn=lambda u: K0_ff)

        return {"galerkin": galerkin, "rom": rom, "freq_hz_g": freq_hz_g, "eigvecs_r_g": eigvecs_r_g}

    def test_Kr_Mr_match_galerkin_exactly(self, setup):
        # Both classes compute V^T K V / V^T M V the same way -- should
        # agree to floating-point precision, not just approximately.
        assert np.allclose(setup["rom"].K_r, setup["galerkin"].K_r, atol=1e-9)
        assert np.allclose(setup["rom"].M_r, setup["galerkin"].M_r, atol=1e-9)

    def test_modal_frequencies_match_galerkin(self, setup):
        rom = setup["rom"]
        eigvals = np.clip(eigh(rom.K_r, rom.M_r, eigvals_only=True), 0, None)
        freq_hz_rom = np.sqrt(eigvals) / (2 * np.pi)
        err = np.max(np.abs(freq_hz_rom - setup["freq_hz_g"]) / setup["freq_hz_g"])
        print(f"IntrusiveNonlinearROM vs GalerkinROM modal frequency max relative error: {err:.3e}")
        assert err < 1e-8

    def test_single_mode_rk4_trajectory_matches_analytic_linear_oscillator(self, setup):
        rom, eigvecs_r_g, freq_hz_g = setup["rom"], setup["eigvecs_r_g"], setup["freq_hz_g"]
        omega1 = 2 * np.pi * freq_hz_g[0]
        phi1 = eigvecs_r_g[:, 0]   # M_r-orthonormal (scipy.linalg.eigh generalized-eigenvector convention)

        amp = 1e-4
        q0 = amp * phi1
        qdot0 = np.zeros_like(q0)
        dt = (2 * np.pi / omega1) / 200.0
        n_steps = 400

        t, q_hist, _qdot_hist = rom.integrate_rk4(q0, qdot0, dt, n_steps)
        analytic_q = amp * np.cos(omega1 * t)[:, None] * phi1[None, :]
        err = np.max(np.abs(q_hist - analytic_q)) / amp
        print(f"single-mode RK4 trajectory vs closed-form cos(omega1 t) max abs error "
              f"(normalized by amplitude): {err:.3e}")
        assert err < 1e-4


# =====================================================================
# CHECK 3 -- undamped RK4 conserves total reduced energy to O(dt^4).
# =====================================================================
class TestEnergyConservationRK4FourthOrder:
    @pytest.fixture(scope="class")
    def setup(self):
        # Same n_elem=2 / CFL-based dt reasoning as
        # TestIdentityBasisReproducesFullOrder above (Beam2DReissner's
        # stiffest mode sets RK4's stability limit, not the fundamental
        # one).
        fx = ff.reissner_cantilever_system(n_elem=2, L=1.0)
        fes, mat, free = fx["sys"], fx["mat"], fx["free_dofs"]
        n_dof, n_free = fx["n_dof"], len(free)
        M_ff = fx["M"][np.ix_(free, free)]
        C_ff = np.zeros_like(M_ff)   # undamped, per the roadmap's own criterion

        def internal_force_fn(u_free):
            return fes.assemble_internal_force(_embed(u_free, free, n_dof), mat)[free]

        def tangent_fn(u_free):
            return fes.assemble_tangent_stiffness(_embed(u_free, free, n_dof), mat)[np.ix_(free, free)]

        V = np.eye(n_free)
        rom = IntrusiveNonlinearROM(V, M_ff, C_ff, internal_force_fn,
                                     load_fn=lambda t: np.zeros(n_free), tangent_fn=tangent_fn)
        strain_energy_fn = lambda q: _strain_energy_full(fes, mat, _embed(V @ q, free, n_dof))

        u0_free = _static_tip_deflection(fes, mat, free, fx["n_elem"], force=2.0e3)
        q0, qdot0 = rom.initial_conditions(u0_full=u0_free, v0_full=np.zeros(n_free))

        freq_hz_all, _ = fes.solve_modal(n_modes=n_free)
        dt_finest = (1.0 / freq_hz_all[-1]) / 40.0   # safely inside RK4's stability bound
        T_total = 800.0 * dt_finest   # fixed window; only the STEP COUNT varies below

        return {"rom": rom, "q0": q0, "qdot0": qdot0,
                "strain_energy_fn": strain_energy_fn, "T_total": T_total}

    def test_rk4_energy_drift_scales_as_dt4(self, setup):
        rom, q0, qdot0 = setup["rom"], setup["q0"], setup["qdot0"]
        sef, T_total = setup["strain_energy_fn"], setup["T_total"]

        n_steps_list = [200, 400, 800]
        drifts = []
        for n_steps in n_steps_list:
            dt = T_total / n_steps
            t, q_hist, qdot_hist = rom.integrate_rk4(q0, qdot0, dt, n_steps)
            E_hist = np.array([rom.energy(q_hist[i], qdot_hist[i], sef) for i in range(len(t))])
            E0 = E_hist[0]
            drift = np.max(np.abs(E_hist - E0)) / abs(E0)
            drifts.append(drift)

        r1 = drifts[0] / drifts[1]
        r2 = drifts[1] / drifts[2]
        print(f"RK4 energy drifts at n_steps={n_steps_list}: {drifts}")
        print(f"halving-dt drift ratios: {r1:.3f}, {r2:.3f} (expect ~16 for a genuine O(dt^4) method)")
        # a real 4th-order method halves dt -> drift shrinks by ~2^4=16;
        # generous bracket (order in [3, 5]) to absorb the fact that the
        # dominant error term isn't purely quartic at finite dt, while
        # still decisively distinguishing 4th order from 1st/2nd/3rd.
        assert 8.0 < r1 < 40.0
        assert 8.0 < r2 < 40.0


# =====================================================================
# CHECK 4 -- M_r/D_r/K_r symmetric positive definite.
# =====================================================================
class TestReducedMatricesSPD:
    def test_Mr_Dr_Kr_symmetric_positive_definite(self):
        fx = ff.reissner_cantilever_system(n_elem=4, L=1.0)
        fes, mat, free = fx["sys"], fx["mat"], fx["free_dofs"]
        n_dof, n_free = fx["n_dof"], len(free)
        fes.assemble_damping(RayleighDamping(alpha=2.0, beta=1e-6))
        M_ff = fx["M"][np.ix_(free, free)]
        C_ff = fes.C[np.ix_(free, free)]

        rng = np.random.default_rng(5)
        r = min(5, n_free)
        Q, _ = qr(rng.standard_normal((n_free, r)))
        V = Q[:, :r]

        def internal_force_fn(u_free):
            return fes.assemble_internal_force(_embed(u_free, free, n_dof), mat)[free]

        def tangent_fn(u_free):
            return fes.assemble_tangent_stiffness(_embed(u_free, free, n_dof), mat)[np.ix_(free, free)]

        rom = IntrusiveNonlinearROM(V, M_ff, C_ff, internal_force_fn,
                                     load_fn=lambda t: np.zeros(n_free), tangent_fn=tangent_fn)

        for name, A in [("M_r", rom.M_r), ("D_r", rom.D_r), ("K_r", rom.K_r)]:
            asym = np.max(np.abs(A - A.T))
            assert asym < 1e-9 * max(np.max(np.abs(A)), 1.0), f"{name} not symmetric (max asymmetry {asym:.3e})"
            eigvals = eigh(A, eigvals_only=True)
            margin = eigvals.min() / eigvals.max()
            print(f"{name} eigenvalues: min={eigvals.min():.6e}, max={eigvals.max():.6e}, "
                  f"min/max margin={margin:.3e}")
            assert eigvals.min() > 1e-8 * eigvals.max(), \
                f"{name} not positive definite with adequate margin (min/max={margin:.3e})"


# =====================================================================
# Extra sanity checks -- integrate_solve_ivp(), integrate_newton_newmark(),
# and impulse_to_reduced_velocity(), not one of the roadmap's own four
# named validation criteria but required so every integrator this item
# promises ("genuinely available, not just mentioned") is actually
# exercised, not merely defined.
# =====================================================================
class TestSolveIvpAgreesWithRK4:
    def test_solve_ivp_matches_rk4_on_linear_single_mode_case(self):
        fx = ff.reissner_cantilever_system(n_elem=4, L=1.0)
        free = fx["free_dofs"]
        n_free = len(free)
        K0_ff = fx["K"][np.ix_(free, free)]
        M_ff = fx["M"][np.ix_(free, free)]

        rng = np.random.default_rng(9)
        r = min(3, n_free)
        Q, _ = qr(rng.standard_normal((n_free, r)))
        V = Q[:, :r]

        rom = IntrusiveNonlinearROM(
            V, M_ff, np.zeros_like(M_ff),
            internal_force_fn=lambda u: K0_ff @ u,
            load_fn=lambda t: np.zeros(n_free),
            tangent_fn=lambda u: K0_ff)

        eigvals = np.clip(eigh(rom.K_r, rom.M_r, eigvals_only=True), 0, None)
        omega1 = np.sqrt(eigvals[0])
        q0 = np.full(rom.n_modes, 1e-5)
        qdot0 = np.zeros(rom.n_modes)
        dt = (2 * np.pi / omega1) / 200.0
        n_steps = 300

        t_rk4, q_rk4, _ = rom.integrate_rk4(q0, qdot0, dt, n_steps)
        t_ivp, q_ivp, _qdot_ivp, sol = rom.integrate_solve_ivp(
            q0, qdot0, (t_rk4[0], t_rk4[-1]), t_eval=t_rk4, rtol=1e-10, atol=1e-12)

        assert sol.success
        err = np.max(np.abs(q_rk4 - q_ivp)) / np.max(np.abs(q_rk4))
        print(f"integrate_rk4 vs integrate_solve_ivp max relative error: {err:.3e}")
        assert err < 1e-5


class TestNewtonNewmarkIntegrator:
    def test_newton_newmark_agrees_with_rk4_on_identity_basis_free_decay(self):
        fx = ff.reissner_cantilever_system(n_elem=2, L=1.0)
        fes, mat, free = fx["sys"], fx["mat"], fx["free_dofs"]
        n_dof, n_free = fx["n_dof"], len(free)
        fes.assemble_damping(RayleighDamping(alpha=0.0, beta=0.0))
        M_ff = fx["M"][np.ix_(free, free)]
        C_ff = fes.C[np.ix_(free, free)]

        def internal_force_fn(u_free):
            return fes.assemble_internal_force(_embed(u_free, free, n_dof), mat)[free]

        def tangent_fn(u_free):
            return fes.assemble_tangent_stiffness(_embed(u_free, free, n_dof), mat)[np.ix_(free, free)]

        V = np.eye(n_free)
        rom = IntrusiveNonlinearROM(V, M_ff, C_ff, internal_force_fn,
                                     load_fn=lambda t: np.zeros(n_free), tangent_fn=tangent_fn)

        u0_free = _static_tip_deflection(fes, mat, free, fx["n_elem"], force=3.0e3)
        q0, qdot0 = rom.initial_conditions(u0_full=u0_free, v0_full=np.zeros(n_free))

        freq_hz_all, _ = fes.solve_modal(n_modes=n_free)
        dt_rk4 = (1.0 / freq_hz_all[-1]) / 40.0
        n_steps_rk4 = 400
        t_rk4, q_rk4, _ = rom.integrate_rk4(q0, qdot0, dt_rk4, n_steps_rk4)
        T_total = n_steps_rk4 * dt_rk4

        # Newton-Newmark is unconditionally stable, so it can take a MUCH
        # larger dt than RK4's own CFL-restricted one and still reach the
        # same final time -- exactly the stiff-case use case this
        # integrator exists for.
        n_steps_nm = 40
        dt_nm = T_total / n_steps_nm
        t_nm, q_nm, _ = rom.integrate_newton_newmark(q0, qdot0, dt_nm, n_steps_nm)

        err = np.max(np.abs(q_nm[-1] - q_rk4[-1])) / np.max(np.abs(q_rk4))
        print(f"Newton-Newmark (dt={dt_nm:.3e}, {n_steps_nm} steps) vs RK4 "
              f"(dt={dt_rk4:.3e}, {n_steps_rk4} steps) final-state relative error: {err:.3e}")
        assert np.all(np.isfinite(q_nm))
        assert err < 0.05


class TestImpulseToReducedVelocity:
    def test_impulse_reduces_consistently_through_Mr(self):
        fx = ff.reissner_cantilever_system(n_elem=4, L=1.0)
        free = fx["free_dofs"]
        n_free = len(free)
        n_dof = fx["n_dof"]
        K0_ff = fx["K"][np.ix_(free, free)]
        M_ff = fx["M"][np.ix_(free, free)]

        rng = np.random.default_rng(11)
        r = min(3, n_free)
        Q, _ = qr(rng.standard_normal((n_free, r)))
        V = Q[:, :r]

        rom = IntrusiveNonlinearROM(
            V, M_ff, np.zeros_like(M_ff),
            internal_force_fn=lambda u: K0_ff @ u,
            load_fn=lambda t: np.zeros(n_free),
            tangent_fn=lambda u: K0_ff)

        P_free = rng.standard_normal(n_free)
        tau0 = 2.5e-5
        qdot0 = rom.impulse_to_reduced_velocity(P_free, tau0)

        # cross-check against the full-order relation M v0 = P*tau0,
        # v0 = M^-1 @ (P*tau0), reduced the SAME way
        # initial_conditions() reduces an ordinary velocity.
        v0_free = np.linalg.solve(M_ff, P_free * tau0)
        _q0, qdot0_via_ic = rom.initial_conditions(v0_full=v0_free)

        err = np.max(np.abs(qdot0 - qdot0_via_ic))
        scale = max(np.max(np.abs(qdot0_via_ic)), 1e-30)
        print(f"impulse_to_reduced_velocity vs initial_conditions(v0_full=M^-1 P tau0) "
              f"max abs error (normalized): {err / scale:.3e}")
        assert err / scale < 1e-9
