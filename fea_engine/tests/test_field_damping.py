"""
test_field_damping.py -- Wave 17 item 141 (docs/consolidated_future_
roadmap.md, source: Georgiou 2005 "Advanced Proper Orthogonal
Decomposition Tools...", Eqs. 1-3, 24, 35): validates
damping.FieldDamping, a field-wise ("per DOF-type") consistent viscous
damping matrix built with the SAME shape-function/quadrature structure
element.Element.mass() already uses, with a damping coefficient D_i
substituted for the density-like quantity rho_i -- structurally
parallel to how the mass matrix is assembled from rho, NOT a Rayleigh
(mass/stiffness-proportional) model, which is what fea_engine.damping
already had (RayleighDamping) before this item.

Two decisive checks, matching the roadmap's own stated validation
criteria for item 141:

1. test_field_damping_matches_mass_with_density_replaced -- C from
   FieldDamping(coefficients=D) must be produced by literally the same
   Element.mass() call as assemble_mass(D): checked bit-identical
   (np.array_equal), not approximately equal, since both code paths
   call the exact same function with the exact same arguments.

2. test_damping_ratio_matches_closed_form -- the per-mode damping
   ratio of a damped linear structure matches the closed-form
   zeta = D / (2*rhoA*omega) for bending-dominated modes, checked TWO
   independent ways: (a) modal projection of the assembled C/M/K
   (exact to near machine precision here, since this element's mass()
   has no separate rotational weighting, making the resulting C
   exactly mass-proportional -- D/rho_A times M), and (b) an actual
   free-vibration transient log-decrement measurement from
   solve_transient_implicit(), independent numerical evidence rather
   than a restatement of (a).

Deviation from the literal roadmap text, honestly noted: the roadmap's
item 141 row names Beam2DReissner (Wave 17 item 140, a 3-field
geometrically-exact rod element with genuine rotary inertia) as the
target for check 2, with target ratios ~0.080 (mode 1) and ~0.0099
(mode 3). Beam2DReissner is a SEPARATE, parallel roadmap item and was
NOT present in this codebase when this file was written and run (see
this module's own report). Per this item's own explicit fallback
instruction ("construct an equivalent check with whatever damped
element/system is available... and derive the matching closed-form
ratio yourself"), Beam2DEulerBernoulli (already in the package, 2
dofs/node: v, theta, a single rho_A weighting the WHOLE consistent
mass matrix -- no separate translational/rotational split) is used
instead. It is a genuine single-field case for check 1 exactly as
specified, and its bending-dominated modes satisfy the same
zeta = D/(2*rhoA*omega) relation used for check 2.
"""
import numpy as np
import pytest
from scipy.linalg import eigh

from fea_engine import Beam2DEulerBernoulli, FESystem
from fea_engine.mesh import line_mesh
from fea_engine.damping import FieldDamping, RayleighDamping

L = 10.0
E = 2.0e11
I = 8.333e-6
EI = E * I
RHO_A = 7.85
N_ELEM = 10


def _pinned_beam():
    """A pinned-pinned (v=0 at both ends, rotation free) Euler-Bernoulli
    beam -- transverse-bending-dominated modes, matching the roadmap's
    own "translational/bending-dominated" qualifier for the closed-form
    check."""
    mesh = line_mesh(L, N_ELEM)
    fs = FESystem(mesh, Beam2DEulerBernoulli())
    fs.assemble_stiffness(EI)
    fs.assemble_mass(RHO_A)
    fs.fix_dofs([0], [0])
    fs.fix_dofs([N_ELEM], [0])
    return fs


# ---------------------------------------------------------------------
# Check 1: C == mass() with density replaced by D, bit-identical.
# ---------------------------------------------------------------------
def test_field_damping_matches_mass_with_density_replaced():
    D = 3.7

    fs = _pinned_beam()
    fs.assemble_damping(FieldDamping(coefficients=D))

    fs_D_mass = FESystem(line_mesh(L, N_ELEM), Beam2DEulerBernoulli())
    fs_D_mass.assemble_mass(D)

    assert fs.C.shape == fs_D_mass.M.shape
    assert np.array_equal(fs.C, fs_D_mass.M), (
        "FieldDamping's assembled C must be bit-identical to a mass "
        "matrix assembled with density replaced by D -- both call the "
        "SAME Element.mass() code path.")


def test_field_damping_scales_linearly_in_D():
    """Sanity check on top of the exact-equality check above: since
    Element.mass() is linear in its density argument (a Gauss-loop sum
    of Nm^T * scalar * Nm terms), C(2D) must equal 2*C(D) exactly."""
    D = 1.3
    fs1 = _pinned_beam()
    fs1.assemble_damping(FieldDamping(coefficients=D))
    fs2 = _pinned_beam()
    fs2.assemble_damping(FieldDamping(coefficients=2 * D))
    assert np.allclose(fs2.C, 2 * fs1.C, atol=0, rtol=1e-13)


# ---------------------------------------------------------------------
# Check 2: per-mode damping ratio matches closed form D/(2*rhoA*omega).
# ---------------------------------------------------------------------
def test_damping_ratio_matches_closed_form_modal_projection():
    """(a) Modal-projection check: build undamped modes (omega_n, phi_n)
    from the generalized eigenproblem K phi = omega^2 M phi, mass-
    normalize each mode, then project the assembled FieldDamping matrix
    C onto each mode: zeta_n = phi_n^T C phi_n / (2*omega_n). Compare
    against the closed-form D/(2*rhoA*omega_n) for the first three
    bending modes."""
    D = 0.6 * RHO_A   # arbitrary, comparable-scale damping coefficient

    fs = _pinned_beam()
    fs.assemble_damping(FieldDamping(coefficients=D))

    free = fs.free_dofs
    Kff = fs.K[np.ix_(free, free)]
    Mff = fs.M[np.ix_(free, free)]
    Cff = fs.C[np.ix_(free, free)]

    omega2, phi = eigh(Kff, Mff)
    omega = np.sqrt(np.clip(omega2, 0, None))

    for n in range(3):
        phin = phi[:, n]
        # eigh(K, M) already returns M-orthonormal eigenvectors
        # (phi_n^T M phi_n == 1) -- verify that directly rather than
        # assuming it.
        mass_norm = phin @ Mff @ phin
        assert np.isclose(mass_norm, 1.0, atol=1e-8), (
            "expected eigh(K, M) to return M-normalized eigenvectors")
        zeta_n = (phin @ Cff @ phin) / (2 * omega[n])
        zeta_closed_form = D / (2 * RHO_A * omega[n])
        assert np.isclose(zeta_n, zeta_closed_form, rtol=1e-10), (
            f"mode {n+1}: modal zeta={zeta_n!r} vs "
            f"closed-form D/(2*rhoA*omega)={zeta_closed_form!r}")


def test_damping_ratio_matches_closed_form_free_decay():
    """(b) Independent check via an actual free-vibration transient:
    give the beam an initial shape close to mode 1, integrate the
    damped system with no external load through
    solve_transient_implicit(), and extract the damping ratio from the
    log-decrement of the resulting envelope. Cross-checked against the
    SAME closed-form D/(2*rhoA*omega_1) used in the modal-projection
    check above, but via a completely different numerical procedure
    (time integration + peak-picking, not eigen-projection) -- a
    plausible-but-wrong C matrix that happened to satisfy the modal
    projection check by construction would not necessarily also
    satisfy this one."""
    D = 0.6 * RHO_A

    fs = _pinned_beam()
    fs.assemble_damping(FieldDamping(coefficients=D))

    free = fs.free_dofs
    Kff = fs.K[np.ix_(free, free)]
    Mff = fs.M[np.ix_(free, free)]
    omega2, phi = eigh(Kff, Mff)
    omega1 = np.sqrt(omega2[0])
    zeta1_closed_form = D / (2 * RHO_A * omega1)

    # Initial condition: mode-1 shape, scaled to a small transverse tip
    # displacement, zero velocity.
    n_dof = fs.n_dof
    u0 = np.zeros(n_dof)
    u0[free] = phi[:, 0] * (1e-3 / np.max(np.abs(phi[:, 0])))
    v0 = np.zeros(n_dof)

    class _ZeroLoad:
        def force_at(self, t, n_dof, npn):
            return np.zeros(n_dof)

    T_period1 = 2 * np.pi / omega1
    dt = T_period1 / 200.0
    n_periods = 40
    T_total = n_periods * T_period1

    t, U = fs.solve_transient_implicit(_ZeroLoad(), T_total, dt, u0=u0, v0=v0)

    # Track the transverse displacement at the midspan node (index
    # N_ELEM//2, dof 0 = v) -- dominated by mode 1 for this initial
    # condition/mesh.
    mid_node = N_ELEM // 2
    mid_dof = mid_node * 2  # 2 dofs/node: v, theta
    trace = U[:, mid_dof]

    # Peak-picking: find local maxima of |trace| well past the
    # transient's very first quarter-period (to avoid start-up
    # artifacts), then fit log(|peak|) vs peak time -- slope = -zeta*omega.
    abs_trace = np.abs(trace)
    peak_idx = [i for i in range(2, len(abs_trace) - 2)
                if abs_trace[i] > abs_trace[i - 1] and abs_trace[i] >= abs_trace[i + 1]
                and abs_trace[i] > 1e-8]
    assert len(peak_idx) >= 5, "not enough resolved peaks to measure decay"

    peak_t = t[peak_idx]
    peak_amp = abs_trace[peak_idx]
    # Linear fit of ln(amplitude) vs time -> slope = -zeta*omega
    slope, intercept = np.polyfit(peak_t, np.log(peak_amp), 1)
    zeta1_measured = -slope / omega1

    assert np.isclose(zeta1_measured, zeta1_closed_form, rtol=0.03), (
        f"free-decay-measured zeta1={zeta1_measured!r} vs "
        f"closed-form D/(2*rhoA*omega)={zeta1_closed_form!r}")


# ---------------------------------------------------------------------
# assemble_damping() wiring: additive combination + raw-matrix escape
# hatch, both explicitly required by the roadmap ("Accept it in
# assemble_damping() next to RayleighDamping... and also accept a raw
# matrix for full generality").
# ---------------------------------------------------------------------
def test_assemble_damping_is_additive_across_damping_types():
    D = 2.0
    ray = RayleighDamping(alpha=0.5, beta=1e-6)

    fs_combined = _pinned_beam()
    fs_combined.assemble_damping(ray)
    fs_combined.assemble_damping(FieldDamping(coefficients=D))

    fs_ray = _pinned_beam()
    fs_ray.assemble_damping(ray)

    fs_field = _pinned_beam()
    fs_field.assemble_damping(FieldDamping(coefficients=D))

    assert np.allclose(fs_combined.C, fs_ray.C + fs_field.C, atol=0, rtol=1e-13)


def test_assemble_damping_accepts_raw_matrix():
    """fesystem.C can still be built (additively) from a raw ndarray,
    the pre-existing 'set it by hand' escape hatch the roadmap says
    must remain available."""
    fs = _pinned_beam()
    raw = np.eye(fs.n_dof) * 5.0
    fs.assemble_damping(raw)
    assert np.array_equal(fs.C, raw)

    # A second call accumulates rather than replacing.
    fs.assemble_damping(raw)
    assert np.array_equal(fs.C, 2 * raw)
