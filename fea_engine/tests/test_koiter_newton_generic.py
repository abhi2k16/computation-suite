"""
test_koiter_newton_generic.py -- validation for Module 24:
nonlinear_solver.solve_nonlinear_koiter_newton_generic() (the GENERIC,
m>=1 Koiter-Newton continuation driver that lifts
solve_nonlinear_koiter_newton()'s disclosed m=0 scope reduction).

Three independent lines of evidence, matching the "hand-derivable
benchmark, then regression, then end-to-end robustness" plan in
docs/general_purpose_extensions_roadmap.md Section 11:

1. test_generic_m0_fallback_matches_existing_driver -- with
   enable_mode_interaction=False, the generic driver reproduces the
   ALREADY-VALIDATED solve_nonlinear_koiter_newton()'s own results on
   the same von Mises truss snap-through benchmark test_koiter_newton.py
   uses, to near machine precision. Both implement the identical m=0
   formulas (the generic driver's m=0 branch was written by copying
   them, not re-deriving), so this is a direct no-regression check.

2. test_two_direction_QC_matches_hand_derived_closed_form and
   test_generic_cbar_reproduces_exact_pitchfork_bifurcation -- the
   actual "hand-derivable 2-mode bifurcation benchmark" (task #195):
   a small synthetic 2-DOF system (_PitchforkStub below) with a
   textbook coupled symmetric/antisymmetric potential
       V(u1,u2) = 0.5*k1*u1^2 + 0.5*k2*u2^2 - gamma*u1*u2^2 + 0.25*beta*u2^4
   whose exact equilibrium set is closed-form: a trivial linear primary
   path u2=0 (lambda = k1*u1/F1) up to u1_cr = k2/(2*gamma), where a
   secondary branch u2^2 = (2*gamma*u1-k2)/beta bifurcates off it. Since
   this potential is exactly quartic, the module's quadratic/cubic
   Taylor-coefficient tensors Q/C have NO higher-order remainder along
   any direction -- so both the finite-difference extraction
   (_two_direction_QC) and the algebraic Eq.7 formula
   (_generic_cbar_term) can be checked against EXACT, hand-derived
   closed-form values (not just "roughly consistent" ones), and the
   resulting reduced mode-interaction (row-1) cubic equation can be
   checked to reproduce the TRUE bifurcated-branch amplitude exactly,
   not approximately.

3. test_generic_driver_robust_across_synthetic_bifurcation -- runs the
   full solve_nonlinear_koiter_newton_generic() driver (predictor,
   corrector, adaptive stepping, mode detection, all together) straight
   through the same synthetic bifurcation, checking it (a) actually
   detects the near-critical mode and exercises the m=1 code path
   in some steps, (b) never crashes or diverges, and (c) -- since no
   branch-switching perturbation is applied, and a1=0 is proven below
   to always be an exact root of the row-1 cubic here -- stays on the
   (still mathematically valid) primary path to near machine precision
   the whole way, exactly reproducing lambda = k1*u1/F1.
"""
__author__ = "Abhijeet"
import numpy as np
import pytest

from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine import nonlinear_solver as nls
from fea_engine.nonlinear_solver import _directional_QC, _two_direction_QC, _generic_cbar_term
from scipy.linalg import lu_factor, lu_solve


A_HALFSPAN, H0 = 1.0, 0.10
E, A = 210e9, 2e-4
L0 = np.sqrt(A_HALFSPAN ** 2 + H0 ** 2)


def _von_mises_truss():
    nodes = np.array([[-A_HALFSPAN, 0.0], [0.0, H0], [A_HALFSPAN, 0.0]])
    elements = np.array([[0, 1], [1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, dim=2)
    fes = FESystem(mesh, elmod.TrussTL2D())
    fes.fix_dofs([0, 2], [0, 1])
    control_dof = 1 * fes.npn + 1
    return fes, control_dof


def test_generic_m0_fallback_matches_existing_driver():
    fes_ref, control_dof = _von_mises_truss()
    fes_ref.add_nodal_force([1], 1, -1.0)
    lf_ref, U_ref = nls.solve_nonlinear_koiter_newton(
        fes_ref, (E, A), delta_L=0.01, n_steps=40, tol=1e-10, max_iter=60)

    fes_gen, control_dof2 = _von_mises_truss()
    assert control_dof2 == control_dof
    fes_gen.add_nodal_force([1], 1, -1.0)
    lf_gen, U_gen = nls.solve_nonlinear_koiter_newton_generic(
        fes_gen, (E, A), delta_L=0.01, n_steps=40, tol=1e-10, max_iter=60,
        enable_mode_interaction=False)

    assert np.allclose(lf_gen, lf_ref, rtol=1e-8, atol=1e-10)
    assert np.allclose(U_gen, U_ref, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Synthetic 2-DOF pitchfork-bifurcation model (no finite elements at all --
# a hand-derivable closed-form nonlinear force law, duck-typed as a minimal
# FESystem so it can be fed directly to the real driver).
# ---------------------------------------------------------------------------
K1, K2, GAMMA, BETA, F1_EXT = 100.0, 10.0, 2.0, 3.0, 1.0
U1_CR = K2 / (2 * GAMMA)   # = 2.5, exact critical primary-path displacement


def _V_force(u):
    u1, u2 = u
    f1 = K1 * u1 - GAMMA * u2 ** 2
    f2 = K2 * u2 - 2 * GAMMA * u1 * u2 + BETA * u2 ** 3
    return np.array([f1, f2])


def _V_tangent(u):
    u1, u2 = u
    k12 = -2 * GAMMA * u2
    k22 = K2 - 2 * GAMMA * u1 + 3 * BETA * u2 ** 2
    return np.array([[K1, k12], [k12, k22]])


class _PitchforkStub:
    """Minimal duck-typed FESystem for the coupled symmetric/antisymmetric
    two-mode potential V(u1,u2) = 0.5*K1*u1^2 + 0.5*K2*u2^2 -
    GAMMA*u1*u2^2 + 0.25*BETA*u2^4. Loading only dof1 traces the EXACT
    linear primary path u2=0, lambda=K1*u1/F1_EXT, until the
    antisymmetric tangent K22(u1)=K2-2*GAMMA*u1 crosses zero at
    u1=U1_CR and the secondary branch u2^2=(2*GAMMA*u1-K2)/BETA
    bifurcates off it -- see module docstring."""

    def __init__(self):
        self.n_dof = 2
        self.free_dofs = [0, 1]
        self.F = np.array([F1_EXT, 0.0])
        self.iter_state = {}

    def assemble_internal_force(self, u, mat, **kwargs):
        return _V_force(u)

    def assemble_tangent_stiffness(self, u, mat, **kwargs):
        return _V_tangent(u)

    def commit_all_states(self, u, mat, **kwargs):
        pass

    def update_iter_states(self, u, du, mat, **kwargs):
        pass


def test_two_direction_QC_matches_hand_derived_closed_form():
    """Direct unit check of _two_direction_QC's polarization-identity
    FD extraction against CLOSED-FORM Taylor coefficients hand-derived
    from _V_force's own exact polynomial expansion (see module
    docstring): for direction d=(d1,d2),
        Q(d,d) = (-GAMMA*d2^2,  -2*GAMMA*d1*d2 + 3*BETA*u2n*d2^2)
        C(d,d,d) = (0, BETA*d2^3)
    and, by symmetric polarization, the bilinear/trilinear forms
        Q(a,b) = (-GAMMA*a2*b2,  -GAMMA*(a1*b2+a2*b1) + 3*BETA*u2n*a2*b2)
        C(a,b,c) = (0, BETA*a2*b2*c2)
    A large fd_rel (0.05) is used deliberately: since this potential's
    F_int is exactly quadratic/cubic (no quartic-or-higher remainder)
    along any direction, the stencil has NO truncation error at all
    here, so a LARGER h reduces floating-point cancellation error
    instead of trading it against truncation error -- confirmed below
    by the near-machine-precision (~1e-10) tolerance, far tighter than
    would be meaningful on a genuinely higher-order (finite-element)
    force law."""
    u_n = np.array([0.1, 0.05])
    u0 = np.array([1.0, 0.3])
    u1 = np.array([0.0, 1.0])
    K_ff = _V_tangent(u_n)
    F0 = _V_force(u_n)
    stub = _PitchforkStub()

    qc = _two_direction_QC(stub, None, u_n, [0, 1], u0, u1, K_ff, 0.05, F0)

    def Q_exact(a, b):
        return np.array([-GAMMA * a[1] * b[1],
                          -GAMMA * (a[0] * b[1] + a[1] * b[0]) + 3 * BETA * u_n[1] * a[1] * b[1]])

    def C_exact(a, b, c):
        return np.array([0.0, BETA * a[1] * b[1] * c[1]])

    assert np.allclose(qc['Q00'], Q_exact(u0, u0), atol=1e-8)
    assert np.allclose(qc['Q11'], Q_exact(u1, u1), atol=1e-8)
    assert np.allclose(qc['Q01'], Q_exact(u0, u1), atol=1e-8)
    assert np.allclose(qc['C000'], C_exact(u0, u0, u0), atol=1e-8)
    assert np.allclose(qc['C111'], C_exact(u1, u1, u1), atol=1e-8)
    assert np.allclose(qc['C001'], C_exact(u0, u0, u1), atol=1e-8)
    assert np.allclose(qc['C011'], C_exact(u0, u1, u1), atol=1e-8)
    assert qc['Q01_consistency'] < 1e-8


def test_generic_cbar_reproduces_exact_pitchfork_bifurcation():
    """The decisive hand-derivable mode-interaction check: builds the
    SAME (nf+2)-sized bordered system, second-order solves, and
    Eq.7 cubic coefficients solve_nonlinear_koiter_newton_generic()
    itself builds internally (reusing the actual private helpers, not
    a re-implementation), at a point on the primary path just past the
    bifurcation (u1 = U1_CR + 0.3, u2 = 0), and shows the reduced
    row-1 "equilibrium constraint" cubic in a1 (at a0=0, i.e. evaluated
    AT the current point) has roots EXACTLY {0, +sqrt((2*GAMMA*u1-K2)/BETA),
    -sqrt(...)} -- the true bifurcated-branch amplitude from the exact
    closed-form solution, not an approximation of it. This is possible
    only because the synthetic potential is exactly quartic (see module
    docstring); it is the strongest form of validation available for
    this machinery without a full symbolic/PDE-constrained benchmark."""
    u1n = U1_CR + 0.3
    u_n = np.array([u1n, 0.0])
    free = [0, 1]
    K_ff = _V_tangent(u_n)
    F0 = _V_force(u_n)
    stub = _PitchforkStub()

    f0 = np.array([1.0, 0.0])
    phi_crit = np.array([0.0, 1.0])   # exact eigenvector of K_ff here (K_ff is diagonal on u2=0)
    f1_raw = K_ff @ phi_crit
    ref = 1.0
    f1 = f1_raw * (ref / np.linalg.norm(f1_raw))

    nf = 2
    A_aug2 = np.zeros((nf + 2, nf + 2))
    A_aug2[:nf, :nf] = K_ff
    A_aug2[:nf, nf] = -f0
    A_aug2[:nf, nf + 1] = -f1
    A_aug2[nf, :nf] = -f0
    A_aug2[nf + 1, :nf] = -f1
    lu2 = lu_factor(A_aug2)

    u_dir = {}
    Lbar = np.zeros((2, 2))
    for i in range(2):
        rhs = np.zeros(nf + 2); rhs[nf + i] = -1.0
        x = lu_solve(lu2, rhs)
        u_dir[i] = x[:nf]
        Lbar[:, i] = x[nf:nf + 2]

    qc = _two_direction_QC(stub, None, u_n, free, u_dir[0], u_dir[1], K_ff, 0.05, F0)
    Q_lookup = {(0, 0): qc['Q00'], (0, 1): qc['Q01'], (1, 1): qc['Q11']}
    u_quad, Qbar = {}, {}
    for pair in [(0, 0), (0, 1), (1, 1)]:
        rhs = np.zeros(nf + 2); rhs[:nf] = -Q_lookup[pair]
        x = lu_solve(lu2, rhs)
        u_quad[pair] = x[:nf]
        Qbar[pair] = x[nf:nf + 2]

    C_lin = {(0, 0, 0): qc['C000'], (0, 0, 1): qc['C001'],
             (0, 1, 1): qc['C011'], (1, 1, 1): qc['C111']}
    triples = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
    Cbar = {(p, t): _generic_cbar_term(p, t[0], t[1], t[2], u_dir, u_quad, C_lin, K_ff)
            for p in (0, 1) for t in triples}

    a0 = 0.0
    c3 = Cbar[(1, (1, 1, 1))]
    c2 = 3.0 * Cbar[(1, (0, 1, 1))] * a0
    c1 = Lbar[1, 1] + 2.0 * Qbar[(0, 1)][1] * a0 + 3.0 * Cbar[(1, (0, 0, 1))] * a0 ** 2
    c0 = Lbar[1, 0] * a0 + Qbar[(0, 0)][1] * a0 ** 2 + Cbar[(1, (0, 0, 0))] * a0 ** 3
    roots = np.roots([c3, c2, c1, c0])
    real_roots = sorted(r.real for r in roots if abs(r.imag) < 1e-6)

    true_u2 = np.sqrt((2 * GAMMA * u1n - K2) / BETA)
    expected = sorted([-true_u2, 0.0, true_u2])

    assert len(real_roots) == 3
    assert np.allclose(real_roots, expected, atol=1e-6)


def _run_verbose(fes, **kw):
    """Run the generic driver with verbose=True, returning (lf, U, log)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lf, U = nls.solve_nonlinear_koiter_newton_generic(fes, mat=None, verbose=True, **kw)
    return lf, U, buf.getvalue()


def test_generic_driver_robust_across_synthetic_bifurcation():
    # 2026-09-25: mode detection is now relative to the UNDEFORMED smallest
    # eigenvalue (K2 = 10 here), a narrower window than the old trace/n scale.
    # delta_L/tol below were chosen so that window is actually hit, and property
    # (a) of the module docstring (the m=1 path runs) is now ASSERTED; it was
    # previously claimed but not checked.
    fes = _PitchforkStub()

    lf, U, log = _run_verbose(fes, delta_L=0.05, n_steps=40, tol=1e-10, max_iter=60,
                              mode_detect_rel_tol=0.3, enable_mode_interaction=True)
    assert "[m=1]" in log   # (a) the m=1 branch was actually exercised

    u1, u2 = U[:, 0], U[:, 1]
    assert u1.max() > 1.5 * U1_CR   # actually crossed the bifurcation

    lam_cf = K1 * u1 / F1_EXT
    scale = max(np.max(np.abs(lf)), 1e-30)
    assert np.max(np.abs(lf - lam_cf)) / scale < 1e-6
    assert np.max(np.abs(u2)) < 1e-6


# ---------------------------------------------------------------------------
# 2026-09-25 regression tests for two defects found on NonLin-HyROM Case 2
# (a thin cylindrical shell): see solve_nonlinear_koiter_newton_generic()'s
# docstring, "AUTOMATIC MODE DETECTION" and "CORRECTOR".
# ---------------------------------------------------------------------------
SEXTIC = 0.5   # V += SEXTIC * u2^6 / 6: makes the cubic Koiter predictor inexact


class _ImperfectSexticStub(_PitchforkStub):
    """The pitchfork potential plus a sextic term (so the cubic predictor is
    NOT exact) and an imperfection load EPS on dof 2 (so the residual has a
    component along the extra direction f_1 = K phi_crit)."""

    def __init__(self, eps):
        super().__init__()
        self.F = np.array([F1_EXT, eps])

    def assemble_internal_force(self, u, mat, **kwargs):
        f = _V_force(u)
        f[1] += SEXTIC * u[1] ** 5
        return f

    def assemble_tangent_stiffness(self, u, mat, **kwargs):
        K = _V_tangent(u)
        K[1, 1] += 5 * SEXTIC * u[1] ** 4
        return K


def test_m1_corrector_converges_off_the_predictor():
    """The m=1 corrector used to solve the (nf+2) bordered system with the
    f_1 row (f_1 . du = 0) and drop the f_1 multiplier: with nf=2 that forces
    du = 0, so it could never correct a residual. It "converged" only when the
    predictor alone met tol, and failed at step 1 with "chord corrector failed
    to converge ... after 8 predictor shrinks" on this exact problem (and on
    Case 2). Forcing m=1 on every step (mode_detect_rel_tol=5, i.e. |lam_min| <
    5 x the undeformed lam_min), the fixed Newton corrector must converge every
    step without retries, to states that are genuine equilibria."""
    fes = _ImperfectSexticStub(eps=0.02)
    n_steps = 40
    lf, U, log = _run_verbose(fes, delta_L=0.2, n_steps=n_steps, tol=1e-10, max_iter=60,
                              mode_detect_rel_tol=5.0)
    assert log.count("[m=1]") == n_steps          # m=1 really active on every step
    assert "did not converge" not in log          # no predictor-shrink retries needed
    for lam, u in zip(lf, U):                     # every committed state is in equilibrium
        R = lam * fes.F - fes.assemble_internal_force(u, None)
        assert np.linalg.norm(R) < 1e-9
    assert U[-1, 1] > 0.5                         # followed the imperfection (+u2) branch
    assert np.all(np.diff(lf) > 0)                # monotone load path, no spurious jumps


class _StiffContrastStub(_PitchforkStub):
    """A STABLE system with a 1e6 stiffness contrast, like a thin shell's
    membrane vs bending stiffness: V = 0.5*1e6*u1^2 + 0.5*u2^2 + 0.25*u2^4,
    loaded on the soft dof. No instability anywhere."""

    def __init__(self):
        super().__init__()
        self.F = np.array([0.0, 1.0])

    def assemble_internal_force(self, u, mat, **kwargs):
        return np.array([1e6 * u[0], u[1] + u[1] ** 3])

    def assemble_tangent_stiffness(self, u, mat, **kwargs):
        return np.diag([1e6, 1.0 + 3 * u[1] ** 2])


def test_mode_detection_ignores_stiffness_contrast():
    """The old detection compared lam_min against 0.05 * trace(K)/n, which a
    thin shell's membrane stiffness inflates by orders of magnitude: here
    1 < 0.05 * 5e5, so m=1 fired at step 1 with no instability (on NonLin-HyROM
    Case 2 the ratio was 5.6e-6). Relative to the undeformed lam_min it must
    never fire on this hardening, stable system, and the path must be exact."""
    fes = _StiffContrastStub()
    lf, U, log = _run_verbose(fes, delta_L=0.2, n_steps=15, tol=1e-12, max_iter=60)
    assert "[m=1]" not in log
    u2 = U[:, 1]
    assert np.allclose(lf, u2 + u2 ** 3, rtol=1e-9, atol=1e-12)
    assert u2.max() > 1.0
