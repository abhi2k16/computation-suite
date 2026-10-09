# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
material.py -- material/section data and constitutive (stress-strain)
matrix builders.

Split out of the original config.py during the fea_engine restructuring
(damping models moved to damping.py, since damping is conceptually a
separate system property, not a constitutive law) -- no logic changed,
only file location.

To support a NEW physics (e.g. plane strain thermal, orthotropic
composites, ...):
    1. write a D_xxx(material, ...) function that returns the matrix
    2. add one line to CONSTITUTIVE_REGISTRY
No other module needs to change -- fea_engine.elements element classes
just take whatever D their stiffness() expects as a plain array/tuple
argument.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Material:
    """Isotropic linear-elastic material. G is derived, not stored, so
    E and nu can never silently disagree with G."""
    E: float
    nu: float
    rho: float = 0.0
    # label only (see fea_engine.units): which unit system E and rho are expressed in
    units: object = field(default=None, compare=False, repr=False)

    @property
    def G(self):
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass
class Section:
    """Cross-section properties for 1-D (planar) beam elements."""
    A: float   # area (m^2)
    I: float   # second moment of area about the bending axis (m^4)


@dataclass
class Section3D:
    """Cross-section properties for the 3-D frame element
    (elements.beams3d.Beam3DEulerBernoulli), Module 18 (general-purpose
    extensions Phase 3) -- a separate dataclass from `Section` rather
    than extending it, since a 3-D member genuinely needs two bending
    axes plus torsion, not a superset of the 2-D beam's single `I`.

    A: cross-sectional area (m^2)
    Iy: second moment of area about the LOCAL y axis -- governs
        bending in the local x-z plane (m^4)
    Iz: second moment of area about the LOCAL z axis -- governs
        bending in the local x-y plane (m^4)
    J: St. Venant TORSIONAL constant (m^4) -- NOT the same as Iy+Iz
       except for circular (solid or thin-walled tube) cross-sections;
       for a general open thin-walled section (I-beams, channels, ...)
       J is much smaller than Iy+Iz and requires its own warping-aware
       calculation this package does not perform -- the caller must
       supply the correct J for their section, this dataclass just
       carries it through."""
    A: float
    Iy: float
    Iz: float
    J: float


@dataclass
class PlasticMaterial1D:
    """Uniaxial (1-D) elasto-plastic material: linear elastic up to
    sigma_y, then linear ISOTROPIC hardening with modulus H (H=0 is
    perfectly plastic -- flat yield plateau, no further hardening).
    This is J2/von Mises plasticity specialized to 1-D (a bar can only
    be in uniaxial tension/compression, so the general 3-D deviatoric
    yield surface collapses to the textbook |sigma| <= sigma_y +
    H*alpha, alpha = accumulated plastic strain), which is why the
    return-mapping algorithm in fea_engine.elements.TrussPlastic2D is
    closed-form (no local Newton iteration needed) -- see that class's
    docstring for the derivation. Used together with an element's own
    area A, e.g. mat=(PlasticMaterial1D(...), A) for TrussPlastic2D,
    mirroring the (E, A) convention TrussTL2D already uses."""
    E: float        # elastic (Young's) modulus
    sigma_y: float  # initial (virgin) yield stress
    H: float = 0.0  # isotropic hardening modulus (stress per unit plastic strain)


@dataclass
class PlasticMaterialJ2:
    """3-D (or plane-strain) J2/von Mises elasto-plastic material,
    Module 19 (general-purpose extensions roadmap Phase 6) -- the
    general-continuum generalization of PlasticMaterial1D above, used
    with elements.solids.Hex8PlasticJ2. Linear isotropic hardening
    (H=0 is perfectly plastic, same convention as PlasticMaterial1D).
    See j2_radial_return_3d()'s docstring for the return-mapping
    algorithm this drives (a closed-form RADIAL rescaling in deviatoric
    stress space -- no local Newton needed, the 3-D generalization of
    the same "geometry collapses nicely" property that makes
    PlasticMaterial1D/TrussPlastic2D's 1-D case simple)."""
    E: float
    nu: float
    sigma_y: float
    H: float = 0.0

    @property
    def mu(self):
        """Shear modulus."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def kappa(self):
        """Bulk modulus."""
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))


@dataclass
class NeoHookeanMaterial:
    """Compressible Neo-Hookean hyperelastic material, Module 19
    (general-purpose extensions roadmap Phase 6), used with
    elements.solids.Tet4NeoHookean. Isochoric/volumetric strain-energy
    split W(F) = W_iso(bar_F) + W_vol(J):

        W = (mu/2)*(bar_I1 - 3) + (kappa/2)*(J-1)^2

    bar_I1 = J^(-2/3)*I1, I1 = tr(C), C = F^T F, J = det(F) -- the
    standard, numerically robust starting point for large-strain
    rubber-like behavior (Bonet & Wood, "Nonlinear Continuum Mechanics
    for Finite Element Analysis"). mu, kappa are the SAME small-strain
    shear/bulk moduli as PlasticMaterialJ2 (both, and D_solid3d(),
    reduce to the identical isotropic elastic tensor as strain -> 0 --
    see Tet4NeoHookean's docstring for the check). Does NOT attempt
    the fully/nearly-incompressible limit (nu -> 0.5, which needs a
    mixed u/p formulation to avoid volumetric locking) -- see
    docs/general_purpose_extensions_roadmap.md Section 4 for why that
    is deliberately out of scope here."""
    E: float
    nu: float

    @property
    def mu(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def kappa(self):
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))


# =====================================================================
# J2 (von Mises) radial-return plasticity -- 3-D / plane-strain
# =====================================================================
# Voigt convention used throughout this package (see elements/solids.py
# B_matrix()): strain/stress 6-vectors are ordered [11, 22, 33, 12, 23,
# 13], strain using ENGINEERING shear (gamma_ij = 2*eps_ij); stress
# components are plain tensor components (no factor of 2 -- shear
# stress IS tau_ij, not doubled). The helpers below convert between
# this Voigt convention and true 3x3 tensors so the return-mapping math
# can be written directly in tensor form (safer than hand-tracking
# Voigt factor-of-2 bookkeeping through the return map itself).
def _voigt_strain_to_tensor(eps_voigt):
    e11, e22, e33, g12, g23, g13 = eps_voigt
    return np.array([[e11, g12 / 2, g13 / 2],
                      [g12 / 2, e22, g23 / 2],
                      [g13 / 2, g23 / 2, e33]])


def _tensor_to_voigt_stress(sigma_t):
    return np.array([sigma_t[0, 0], sigma_t[1, 1], sigma_t[2, 2],
                      sigma_t[0, 1], sigma_t[1, 2], sigma_t[0, 2]])


def _tensor_to_voigt_strain(eps_t):
    """Inverse of _voigt_strain_to_tensor -- used to store the plastic
    strain TENSOR back into the same engineering-shear Voigt convention
    the rest of this module uses (so eps_p can be subtracted directly
    from a total-strain Voigt vector next call, with no separate
    bookkeeping about which convention it's in)."""
    return np.array([eps_t[0, 0], eps_t[1, 1], eps_t[2, 2],
                      2 * eps_t[0, 1], 2 * eps_t[1, 2], 2 * eps_t[0, 2]])


def j2_radial_return_3d(eps_voigt, eps_p_n, alpha_n, mat: PlasticMaterialJ2):
    """One elastic-predictor/radial-return step of small-strain J2
    (von Mises) plasticity with linear isotropic hardening -- the 3-D
    generalization of TrussPlastic2D's 1-D return map. STATELESS: does
    not mutate eps_p_n/alpha_n, just returns the trial response at the
    given total strain (see elements.solids.Hex8PlasticJ2 for how
    internal_force()/tangent_stiffness() call this fresh every Newton
    iteration, and commit_state() calls it once more to permanently
    advance state -- exactly TrussPlastic2D's pattern, one level up in
    stress-state dimensionality).

    Algorithm (Simo & Taylor 1985 / de Souza Neto et al., "Computational
    Methods for Plasticity", Box 7.3-7.4): decompose the elastic trial
    stress into volumetric (unaffected by J2 plasticity, which is
    purely deviatoric/incompressible flow) and deviatoric parts. The J2
    yield surface is a HYPERSPHERE of radius sqrt(2/3)*(sigma_y+H*alpha)
    in deviatoric-stress space, so if the trial deviatoric stress
    exceeds that radius, the return to the surface is a RADIAL rescaling
    -- closed-form, no local Newton iteration (unlike plane-stress J2,
    where the return path is constrained to a 2-D subspace and genuinely
    needs one -- see docs/general_purpose_extensions_roadmap.md Section
    4 for why that case is out of scope here).

    The consistent (algorithmic) tangent modulus below was derived by
    directly differentiating this exact discrete algorithm (not copied
    from a possibly-misremembered textbook formula) -- see git history /
    development notes for the derivation; cross-checked against the
    well-known continuum (rate) tangent modulus D_elastic -
    (6*mu^2/(3*mu+H))*(N tensor N) in the dgamma->0 limit, and further
    validated in tests/test_plasticity_j2.py against a finite-difference
    of this same function's own stress output.

    Returns (sigma_voigt (6,), D_tangent_voigt (6,6), eps_p_new_voigt
    (6,), alpha_new, yielded (bool))."""
    mu, kappa = mat.mu, mat.kappa

    eps_t = _voigt_strain_to_tensor(eps_voigt)
    eps_p_t = _voigt_strain_to_tensor(eps_p_n)
    tr_eps = np.trace(eps_t)
    eps_dev = eps_t - (tr_eps / 3.0) * np.eye(3)
    eps_e_dev_trial = eps_dev - eps_p_t
    s_trial = 2.0 * mu * eps_e_dev_trial
    p = kappa * tr_eps
    s_trial_norm = np.sqrt(np.sum(s_trial * s_trial))

    f_trial = s_trial_norm - np.sqrt(2.0 / 3.0) * (mat.sigma_y + mat.H * alpha_n)

    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    I_dev_voigt = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - (1.0 / 3.0) * np.outer(m, m)

    if f_trial <= 0.0:
        sigma_t = s_trial + p * np.eye(3)
        sigma_voigt = _tensor_to_voigt_stress(sigma_t)
        D_elastic = 2.0 * mu * I_dev_voigt + kappa * np.outer(m, m)
        return sigma_voigt, D_elastic, eps_p_n.copy(), alpha_n, False

    h = 2.0 * mu + (2.0 / 3.0) * mat.H
    dgamma = f_trial / h
    N = s_trial / s_trial_norm
    s_new = s_trial - 2.0 * mu * dgamma * N
    eps_p_t_new = eps_p_t + dgamma * N
    alpha_new = alpha_n + np.sqrt(2.0 / 3.0) * dgamma

    sigma_t = s_new + p * np.eye(3)
    sigma_voigt = _tensor_to_voigt_stress(sigma_t)
    eps_p_voigt_new = _tensor_to_voigt_strain(eps_p_t_new)

    theta = 1.0 - 2.0 * mu * dgamma / s_trial_norm
    theta_bar = (1.0 - theta) - 1.0 / (1.0 + mat.H / (3.0 * mu))
    N_voigt = np.array([N[0, 0], N[1, 1], N[2, 2], N[0, 1], N[1, 2], N[0, 2]])
    D_ep = (2.0 * mu * theta * I_dev_voigt
            + 2.0 * mu * theta_bar * np.outer(N_voigt, N_voigt)
            + kappa * np.outer(m, m))

    return sigma_voigt, D_ep, eps_p_voigt_new, alpha_new, True


# =====================================================================
# J2 (von Mises) radial-return plasticity -- PLANE STRESS, Wave 2 item
# 13 (docs/consolidated_future_roadmap.md, source general_purpose_
# extensions_roadmap.md / nonlinear_fem_lessons.md)
# =====================================================================
def j2_radial_return_plane_stress(eps_ps_trial, eps_p_n_voigt6, alpha_n,
                                   eps33_n, mat: PlasticMaterialJ2,
                                   tol=1e-10, max_iter=30):
    """Simo & Taylor (1986), "A return mapping algorithm for plane
    stress elastoplasticity" -- the plane-stress-specific extension
    j2_radial_return_3d()'s own docstring names as needing a genuinely
    different treatment: plane stress constrains the return path to a
    2-D subspace (sigma_33 == 0 identically), so the deviatoric radial
    rescaling that makes the 3-D/plane-strain case closed-form no
    longer applies on its own -- a local Newton iteration is needed.

    THE APPROACH TAKEN HERE (condensation via the already-validated 3-D
    routine, rather than re-deriving Simo & Taylor's own closed-form
    P-matrix formulation from scratch): plane stress is EXACTLY the
    statement "add eps_33 as an extra local unknown, chosen so the
    already-correct 3-D return map's own sigma_33 output comes out to
    zero." Concretely, at a fixed in-plane trial strain (e11, e22, g12)
    -- known, from the element's B_matrix() -- eps_33 is solved for by
    a SCALAR Newton iteration on

        R(eps33) = sigma_33(eps11, eps22, eps33, g12, 0, 0)   [via
                   j2_radial_return_3d() itself, treated as a black box]

    using that SAME function's own consistent tangent entry D6[2,2] =
    d(sigma_33)/d(eps33) (exact, since D6 IS the full consistent
    Jacobian of the 3-D map -- no separate derivative needed) as the
    Newton slope. This reuses j2_radial_return_3d()'s entire, already-
    validated deviatoric return-map algorithm UNCHANGED -- the only new
    code here is the outer scalar root-find and the final tangent
    condensation below -- deliberately minimizing the amount of new,
    untested plasticity math versus re-implementing Simo & Taylor's
    formulation independently (which would duplicate the whole
    deviatoric algorithm a second time with new opportunities for a
    transcription error, exactly the kind of risk this project's own
    history warns about -- see nonlinear_solver.py's Koiter-Newton
    corrector sign-error story).

    eps33_n: the PREVIOUS converged step's eps_33 (0.0 for the very
    first call at a given material point) -- used as the Newton seed,
    a good warm start since eps_33 evolves smoothly step to step. Also
    returned (as eps33_new) so the caller's state can carry it forward,
    exactly like eps_p_n_voigt6/alpha_n.

    Consistent (algorithmic) tangent for the 2-D (plane-stress) problem:
    a standard Schur-complement/STATIC CONDENSATION of the FULL 6x6
    tangent D6, eliminating the sigma_33 row/column analytically (valid
    because sigma_33==0 is being enforced as an exact CONSTRAINT, not
    an independent equation) --

        D_ps[i,j] = D6[i,j] - D6[i,2]*D6[2,j] / D6[2,2]   for i,j in {0,1,3}

    (indices 0,1,3 = the 6-Voigt slots for sigma_11, sigma_22,
    sigma_12). Validated in tests/test_plasticity_plane_stress.py by
    finite-differencing THIS function's own (eps33-Newton-condensed)
    stress output directly -- i.e. checking this implementation's
    internal self-consistency, not trusting the condensation formula's
    textbook correctness on faith.

    Returns (sigma_ps (3,) = [s11,s22,s12], D_ps (3,3), eps_p_new_voigt6
    (6,), alpha_new, eps33_new, yielded (bool)) -- eps_p_new_voigt6
    keeps the FULL 6-component plastic strain (including eps_p_33, kept
    for continuity/generality even though nothing here reads it back
    out in-plane) so it can be fed straight back into
    j2_radial_return_3d() unchanged on the next call, exactly the same
    "self-consistent state vector" convention j2_radial_return_3d()
    itself uses for eps_p_n."""
    e11, e22, g12 = eps_ps_trial
    eps33 = eps33_n
    tol_abs = tol * max(mat.sigma_y, 1.0)

    sigma6 = D6 = eps_p_new6 = alpha_new = yielded = None
    for _ in range(max_iter):
        eps_full = np.array([e11, e22, eps33, g12, 0.0, 0.0])
        sigma6, D6, eps_p_new6, alpha_new, yielded = j2_radial_return_3d(
            eps_full, eps_p_n_voigt6, alpha_n, mat)
        R = sigma6[2]
        if abs(R) < tol_abs:
            break
        eps33 -= R / D6[2, 2]
    else:
        raise RuntimeError(
            f"j2_radial_return_plane_stress: local Newton on eps_33 did not "
            f"converge within {max_iter} iterations (|sigma_33|={abs(R):.3e}, "
            f"tol={tol_abs:.3e}) -- unexpected for a smooth J2 return map; "
            f"check eps_ps_trial/mat for something pathological (e.g. "
            f"nu >= 0.5).")

    idx = [0, 1, 3]
    D_ps = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            D_ps[a, b] = D6[idx[a], idx[b]] - D6[idx[a], 2] * D6[2, idx[b]] / D6[2, 2]
    sigma_ps = np.array([sigma6[0], sigma6[1], sigma6[3]])

    return sigma_ps, D_ps, eps_p_new6, alpha_new, eps33, yielded


@dataclass
class PlasticMaterialJ2Kinematic(PlasticMaterialJ2):
    """PlasticMaterialJ2 + Armstrong-Frederick (1966) nonlinear kinematic
    hardening, Wave 2 item 14 (docs/consolidated_future_roadmap.md,
    source nonlinear_fem_lessons.md appendix's own status-table line:
    "Non-associative / kinematic hardening -- Not Implemented: Isotropic
    hardening only, associative (radial) flow only").

    A separate dataclass SUBCLASSING PlasticMaterialJ2 -- not new fields
    bolted onto PlasticMaterialJ2 itself -- so every existing call site
    that constructs a plain PlasticMaterialJ2 (Hex8PlasticJ2,
    Quad4PlasticJ2PlaneStress, every isotropic-only test in this
    package) keeps working completely unchanged; this is the SAME
    "keep every existing capability reachable, add the new one
    alongside" principle as Hex8SolidBbar sitting next to Hex8Solid3D
    and hourglass_stabilized_stiffness() sitting next to
    reduced_stiffness() elsewhere in Wave 2.

    Two new fields, both defaulting to 0.0 (pure isotropic hardening,
    i.e. mathematically identical to plain PlasticMaterialJ2 -- see
    j2_radial_return_3d_kinematic()'s docstring for why C_kin=0 makes
    the back stress beta stay exactly zero for all time, reproducing
    j2_radial_return_3d()'s own stress/tangent output bit-for-bit):

    C_kin:     linear kinematic hardening modulus (Prager's rule) --
               the back stress beta grows at rate (2/3)*C_kin per unit
               accumulated plastic strain in the flow direction, before
               any dynamic-recovery correction.
    gamma_AF:  Armstrong-Frederick dynamic-recovery ("recall")
               parameter. gamma_AF=0 with C_kin>0 reduces to ordinary
               LINEAR (Prager) kinematic hardening -- unbounded back-
               stress growth under continued straining, a closed-form
               special case used as an independent cross-check in
               tests/test_plasticity_kinematic.py (see that file). A
               nonzero gamma_AF adds a term proportional to -gamma_AF*
               beta*dgamma that makes the hardening evolution rule
               NON-ASSOCIATIVE (it no longer derives from the same
               single hardening potential the flow rule uses -- see
               Simo & Hughes, "Computational Inelasticity", or de Souza
               Neto/Peric/Owen Ch.8, for this exact terminology) and
               gives beta a physically-realistic SATURATION limit as
               accumulated plastic strain grows without bound (matching
               the classic experimentally observed Bauschinger-effect
               saturation AF hardening was built to capture, unlike
               plain Prager hardening, whose back stress grows linearly
               forever) -- taking dgamma -> infinity in a single step of
               j2_radial_return_3d_kinematic()'s own beta update formula
               (beta_new = (beta_old + (2/3)*C_kin*dgamma*N) /
               (1+gamma_AF*dgamma)) shows ||beta|| -> (2/3)*C_kin/
               gamma_AF regardless of beta_old -- exercised directly in
               tests/test_plasticity_kinematic.py."""
    C_kin: float = 0.0
    gamma_AF: float = 0.0


def _voigt_stress_to_tensor(sigma_voigt):
    """Inverse-convention counterpart of _voigt_strain_to_tensor: sigma_
    voigt uses the KINETIC Voigt rule (no factor of 2 on the off-
    diagonal terms, see this module's own convention note above
    j2_radial_return_3d()) -- used here because the back stress beta is
    a STRESS-like internal variable, not a strain, so it must NOT be
    divided by 2 the way _voigt_strain_to_tensor divides its off-
    diagonal entries."""
    s11, s22, s33, s12, s23, s13 = sigma_voigt
    return np.array([[s11, s12, s13],
                      [s12, s22, s23],
                      [s13, s23, s33]])


def j2_radial_return_3d_kinematic(eps_voigt, eps_p_n, alpha_n, beta_n,
                                   mat: PlasticMaterialJ2Kinematic,
                                   tol=1e-10, max_iter=30, h_tangent=1e-7):
    """One elastic-predictor/return step of small-strain J2 plasticity
    with COMBINED linear isotropic (H) + Armstrong-Frederick nonlinear
    kinematic (C_kin, gamma_AF) hardening -- the general-hardening
    extension j2_radial_return_3d()'s own docstring and nonlinear_fem_
    lessons.md both flag as needing "more general treatment" than plain
    radial return, because "combined isotropic-kinematic hardening
    breaks the fixed-flow-direction property."

    DERIVATION (worked from scratch below, not copied from a textbook
    formula, precisely because this combined-hardening case is genuinely
    non-standard enough that a misremembered formula is a real risk --
    see the Koiter-Newton corrector sign-error precedent this project's
    own docstrings keep citing as the cautionary tale):

    Flow is associative w.r.t. the SHIFTED (relative) stress xi = s -
    beta (s = deviatoric stress, beta = back stress): Delta(eps_p) =
    dgamma * N, N = xi_{n+1}/||xi_{n+1}||. Backward-Euler hardening
    evolution: beta_{n+1} = beta_n + dgamma*[(2/3)*C_kin*N -
    gamma_AF*beta_{n+1}] (recall term evaluated IMPLICITLY at n+1, the
    standard/stable choice -- de Souza Neto et al. Box 8.1). Solving
    that equation for beta_{n+1} (linear once N is fixed):

        beta_{n+1} = [beta_n + (2/3)*C_kin*dgamma*N] / (1+gamma_AF*dgamma)

    Substituting into xi_{n+1} = s_trial - 2*mu*dgamma*N - beta_{n+1}
    and demanding N = xi_{n+1}/||xi_{n+1}|| (self-consistency) gives,
    after collecting terms (full algebra in the module's development
    notes):

        xi_{n+1} = A(dgamma) - k(dgamma)*N,   where
        A(dgamma) = s_trial - beta_n/(1+gamma_AF*dgamma)
        k(dgamma) = dgamma*[2*mu + (2/3)*C_kin/(1+gamma_AF*dgamma)]

    Since A(dgamma) is a FIXED vector for any given scalar dgamma, and
    xi_{n+1}+k(dgamma)*N = A(dgamma) with xi_{n+1} parallel to N by
    construction, N must equal A(dgamma)/||A(dgamma)|| -- i.e. the
    return direction is a well-defined (if dgamma-dependent, unlike
    plain isotropic or pure-Prager hardening where it's fixed) function
    of the single scalar unknown dgamma, and the yield condition
    collapses to ONE scalar equation:

        R(dgamma) = ||A(dgamma)|| - k(dgamma)
                     - sqrt(2/3)*(sigma_y + H*(alpha_n+sqrt(2/3)*dgamma)) = 0

    Sanity checks on this formula (also exercised directly in
    tests/test_plasticity_kinematic.py, not just asserted here):
    - gamma_AF=0: A(dgamma)=s_trial-beta_n=xi_trial (dgamma-independent,
      confirming pure-Prager kinematic hardening keeps a FIXED return
      direction, same as plain isotropic hardening) and R(dgamma)
      reduces to the classic closed form dgamma = f_trial/(2*mu +
      (2/3)*(H+C_kin)) -- an "effective combined hardening modulus"
      result, independently checked in the test file.
    - C_kin=0: beta_n=0 stays 0 forever (each update is beta_n scaled by
      1/(1+gamma_AF*dgamma), and 0 scaled by anything is 0), so this
      reduces EXACTLY to j2_radial_return_3d()'s own math.

    R(dgamma) is solved by a scalar local Newton iteration using a
    FINITE-DIFFERENCE slope (not a hand-differentiated dR/ddgamma) --
    deliberately: R(dgamma) above is already a from-scratch derivation,
    and differentiating it again by hand would stack a second layer of
    fresh, unvalidated algebra on top of the first for no real benefit
    (the local iteration only needs *a* slope that gets the Newton loop
    to converge, not an exact one). The RETURNED consistent tangent
    D_ep is, similarly, a finite difference of this function's own
    converged stress output at the given (eps_p_n, alpha_n, beta_n) --
    exactly the choice elements.nonlinear_solids.Tet4NeoHookean.
    tangent_stiffness() already documents making for the same reason
    (hand-deriving the full combined isotropic-kinematic consistent
    tangent has real transcription risk; a numerically consistent
    tangent from the already-validated closed-form stress update is
    safer and still gives Newton-quality convergence). Independent
    (non-circular) validation of this FD tangent lives in
    tests/test_plasticity_kinematic.py: at gamma_AF=0 it is checked
    against the SEPARATELY-derivable closed form D_ep = 2*mu*theta*
    I_dev + 2*mu*theta_bar*outer(N,N) + kappa*outer(m,m) (identical in
    FORM to j2_radial_return_3d()'s own tangent, just with hardening
    modulus H replaced by H+C_kin) -- an algebraic cross-check, not
    merely FD-against-itself.

    Returns (sigma_voigt (6,), D_tangent_voigt (6,6), eps_p_new_voigt
    (6,), alpha_new, beta_new_voigt (6,), yielded (bool)). beta uses the
    KINETIC Voigt convention (stress-like, no engineering-shear factor
    of 2 -- see _voigt_stress_to_tensor())."""
    mu, kappa = mat.mu, mat.kappa
    C_kin, gamma_AF = mat.C_kin, mat.gamma_AF
    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    I_dev_voigt = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - (1.0 / 3.0) * np.outer(m, m)

    def _response(eps_voigt_, eps_p_n_, alpha_n_, beta_n_):
        """Stateless trial evaluation (no tangent) -- used both for the
        function's real output and, via central differences, to build
        the FD consistent tangent below."""
        eps_t = _voigt_strain_to_tensor(eps_voigt_)
        eps_p_t = _voigt_strain_to_tensor(eps_p_n_)
        beta_t = _voigt_stress_to_tensor(beta_n_)
        tr_eps = np.trace(eps_t)
        eps_dev = eps_t - (tr_eps / 3.0) * np.eye(3)
        eps_e_dev_trial = eps_dev - eps_p_t
        s_trial = 2.0 * mu * eps_e_dev_trial
        p = kappa * tr_eps
        xi_trial = s_trial - beta_t
        xi_trial_norm = np.sqrt(np.sum(xi_trial * xi_trial))

        f_trial = xi_trial_norm - np.sqrt(2.0 / 3.0) * (mat.sigma_y + mat.H * alpha_n_)
        if f_trial <= 0.0:
            sigma_t = s_trial + p * np.eye(3)
            return (_tensor_to_voigt_stress(sigma_t), eps_p_n_.copy(), alpha_n_,
                    beta_n_.copy(), False)

        def R_of(dgamma):
            denom = 1.0 + gamma_AF * dgamma
            A_t = s_trial - beta_t / denom
            A_norm = np.sqrt(np.sum(A_t * A_t))
            k = dgamma * (2.0 * mu + (2.0 / 3.0) * C_kin / denom)
            alpha_trial = alpha_n_ + np.sqrt(2.0 / 3.0) * dgamma
            R = A_norm - k - np.sqrt(2.0 / 3.0) * (mat.sigma_y + mat.H * alpha_trial)
            return R, A_t, A_norm

        # closed-form gamma_AF=0 combined-modulus formula as the Newton seed
        dgamma = f_trial / (2.0 * mu + (2.0 / 3.0) * (mat.H + C_kin))
        tol_abs = tol * max(mat.sigma_y, 1.0)
        R, A_t, A_norm = R_of(dgamma)
        for _ in range(max_iter):
            if abs(R) < tol_abs:
                break
            hh = max(1e-9, 1e-6 * abs(dgamma))
            R_p, _, _ = R_of(dgamma + hh)
            R_m, _, _ = R_of(max(dgamma - hh, 0.0))
            slope = (R_p - R_m) / (dgamma + hh - max(dgamma - hh, 0.0))
            dgamma = max(dgamma - R / slope, 0.0)
            R, A_t, A_norm = R_of(dgamma)
        else:
            raise RuntimeError(
                f"j2_radial_return_3d_kinematic: local Newton on dgamma did "
                f"not converge within {max_iter} iterations (|R|={abs(R):.3e}, "
                f"tol={tol_abs:.3e}).")

        denom = 1.0 + gamma_AF * dgamma
        N_t = A_t / A_norm
        s_new = s_trial - 2.0 * mu * dgamma * N_t
        beta_new_t = (beta_t + (2.0 / 3.0) * C_kin * dgamma * N_t) / denom
        eps_p_t_new = eps_p_t + dgamma * N_t
        alpha_new_ = alpha_n_ + np.sqrt(2.0 / 3.0) * dgamma
        sigma_t = s_new + p * np.eye(3)
        return (_tensor_to_voigt_stress(sigma_t), _tensor_to_voigt_strain(eps_p_t_new),
                alpha_new_, _tensor_to_voigt_stress(beta_new_t), True)

    sigma_voigt, eps_p_new, alpha_new, beta_new, yielded = _response(
        eps_voigt, eps_p_n, alpha_n, beta_n)

    D_ep = np.zeros((6, 6))
    for j in range(6):
        dE = np.zeros(6)
        dE[j] = h_tangent
        sp, *_ = _response(eps_voigt + dE, eps_p_n, alpha_n, beta_n)
        sm, *_ = _response(eps_voigt - dE, eps_p_n, alpha_n, beta_n)
        D_ep[:, j] = (sp - sm) / (2.0 * h_tangent)
    D_ep = 0.5 * (D_ep + D_ep.T)   # FD roundoff can break exact symmetry by
    # ~1e-8 relative even though the true tangent is symmetric for this
    # associative-flow model -- same symmetrization Tet4NeoHookean.
    # tangent_stiffness() already performs for the same reason.

    return sigma_voigt, D_ep, eps_p_new, alpha_new, beta_new, yielded


# =====================================================================
# Compressible Neo-Hookean hyperelasticity -- 2nd Piola-Kirchhoff stress
# =====================================================================
def neo_hookean_pk2_stress(F, mat: NeoHookeanMaterial):
    """2nd Piola-Kirchhoff stress S(F) for the compressible Neo-Hookean
    model in NeoHookeanMaterial's docstring, derived by direct
    differentiation S = 2*dW/dC:

        S = mu*J^(-2/3)*(I - (I1/3)*C^-1) + kappa*J*(J-1)*C^-1

    where C = F^T F, J = det(F), I1 = tr(C) -- see
    elements.solids.Tet4NeoHookean for how this drives internal_force()
    (a single evaluation per element, since Tet4's linear shape
    functions give a CONSTANT deformation gradient, exactly like
    Tet4Solid3D's linear-elastic stiffness() needs only one B_matrix()
    evaluation). Returns (S (3x3), W (scalar strain energy density) --
    W is not needed by the element itself, only useful for diagnostics/
    energy-conservation checks)."""
    mu, kappa = mat.mu, mat.kappa
    C = F.T @ F
    J = np.linalg.det(F)
    I1 = np.trace(C)
    Cinv = np.linalg.inv(C)

    S = mu * J ** (-2.0 / 3.0) * (np.eye(3) - (I1 / 3.0) * Cinv) + kappa * J * (J - 1.0) * Cinv
    W = 0.5 * mu * (J ** (-2.0 / 3.0) * I1 - 3.0) + 0.5 * kappa * (J - 1.0) ** 2
    return S, W


# =====================================================================
# Constitutive matrix builders
# =====================================================================
def D_plane_stress(mat: Material) -> np.ndarray:
    E, nu = mat.E, mat.nu
    return E / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]])


def D_plane_strain(mat: Material) -> np.ndarray:
    E, nu = mat.E, mat.nu
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([
        [1 - nu, nu, 0],
        [nu, 1 - nu, 0],
        [0, 0, (1 - 2 * nu) / 2]])


def D_solid3d(mat: Material) -> np.ndarray:
    E, nu = mat.E, mat.nu
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([
        [1 - nu, nu,     nu,     0,                0,                0],
        [nu,     1 - nu, nu,     0,                0,                0],
        [nu,     nu,     1 - nu, 0,                0,                0],
        [0,      0,      0,      (1 - 2 * nu) / 2, 0,                0],
        [0,      0,      0,      0,                (1 - 2 * nu) / 2, 0],
        [0,      0,      0,      0,                0,                (1 - 2 * nu) / 2]])


def D_mindlin_plate(mat: Material, h: float, k_shear: float = 5.0 / 6.0):
    """Returns (Db, Ds): bending and shear constitutive matrices for a
    Reissner-Mindlin plate of thickness h. Kept as a pair (not stacked
    into one matrix) because bending and shear are integrated with
    different Gauss orders (selective reduced integration) -- see
    fea_engine.elements.Quad4MindlinPlate.stiffness()."""
    E, nu = mat.E, mat.nu
    Dp = E * h**3 / (12 * (1 - nu**2))
    Db = Dp * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    Ds = k_shear * mat.G * h * np.eye(2)
    return Db, Ds


def D_shell(mat: Material, h: float, k_shear: float = 5.0 / 6.0):
    """Returns (Dm, Db, Ds, h): membrane, bending, and shear
    constitutive matrices for elements.shells.Shell4MITC (Module 20,
    general-purpose extensions roadmap Phase 7), plus the thickness
    itself (needed separately since Dm, unlike Db/Ds, does NOT bake h
    in -- see below).

    Dm = D_plane_stress(mat): the membrane part is literally
    Quad4PlaneStress's constitutive matrix, PER UNIT THICKNESS, since
    Shell4MITC.stiffness() multiplies it by h explicitly at the
    integration step -- the same "thickness is a separate multiplicative
    factor" convention Quad4PlaneStress's own stiffness() (the generic
    Element.stiffness() Gauss loop) already uses.

    (Db, Ds) = D_mindlin_plate(mat, h, k_shear): the bending/shear part
    is reused UNCHANGED from the existing Reissner-Mindlin plate
    formulation (see that function's docstring) -- h is already baked
    into Db (~h^3) and Ds (~h) there, since plate bending/shear rigidity
    isn't a simple linear-in-thickness scaling the way membrane/solid
    stiffness is, so Shell4MITC's bending+shear integration does NOT
    multiply by h again (matching Quad4MindlinPlate.stiffness()'s own
    convention of not taking a separate thickness multiplier either).

    h is returned as the 4th tuple element purely so Shell4MITC.stiffness()
    has it on hand for the membrane multiply and the drilling-stiffness
    scale, without the caller needing to pass it as a second, easy-to-
    desync argument alongside D."""
    Dm = D_plane_stress(mat)
    Db, Ds = D_mindlin_plate(mat, h, k_shear)
    return Dm, Db, Ds, h


def shell_rho_matrix(mat: Material, h: float):
    """Returns the 6x6 diagonal density matrix for Shell4MITC's LOCAL
    per-node DOF ordering (u, v, w, theta_x, theta_y, theta_z):
    translational entries rho*h (all three identical -- isotropic mass
    density, correct for any orientation of the local frame), rotary
    inertia rho*h^3/12 for theta_x/theta_y (matching Quad4MindlinPlate's
    own rho*h^3/12 convention for its bending rotations), and the SAME
    rho*h^3/12 for the artificial drilling DOF theta_z.

    Giving theta_z the same rotary inertia as theta_x/theta_y (rather
    than zero, or some other ad hoc value) is not just a convenience
    default: it's what keeps the local 3x3 rotational sub-block of the
    per-node mass matrix ISOTROPIC (a scalar multiple of the identity),
    which is what makes lumped_mass()'s inherited HRZ diagonal-scaling
    logic (see Element.lumped_mass()) remain correct after
    Shell4MITC.mass() rotates the local mass matrix into global
    coordinates -- diag(R^T (c*I) R) = c for any orthonormal R only
    when the pre-rotation block is already isotropic. A non-isotropic
    choice here would silently break that inherited logic instead of
    just being physically arbitrary."""
    rho = mat.rho
    return np.diag([rho * h, rho * h, rho * h,
                     rho * h**3 / 12, rho * h**3 / 12, rho * h**3 / 12])


def EI_beam(mat: Material, sec: Section) -> float:
    """Bending rigidity for the 1-D Euler-Bernoulli beam element."""
    return mat.E * sec.I


def beam3d_rigidities(mat: Material, sec: Section3D):
    """(EA, GJ, EIy, EIz) -- the four rigidities
    elements.beams3d.Beam3DEulerBernoulli.stiffness() needs, bundled
    once so the element itself only handles the FE algebra, not
    material/section bookkeeping -- mirrors EI_beam()'s role for the
    2-D beam, generalized to three independent stiffnesses (axial,
    torsional, two bending) instead of one."""
    return mat.E * sec.A, mat.G * sec.J, mat.E * sec.Iy, mat.E * sec.Iz


def beam3d_mass_props(mat: Material, sec: Section3D):
    """(rho*A, rho*Ip) -- the two mass properties
    Beam3DEulerBernoulli.mass() needs. Ip = Iy + Iz is the cross-
    section's AREA polar moment (about the beam axis) -- the physically
    correct quantity for the torsional/rotational MASS matrix, and
    deliberately NOT the same as sec.J (the torsional STIFFNESS
    constant): the two coincide only for circular sections, same
    caveat as Section3D.J's docstring."""
    return mat.rho * sec.A, mat.rho * (sec.Iy + sec.Iz)


CONSTITUTIVE_REGISTRY = {
    "plane_stress": D_plane_stress,
    "plane_strain": D_plane_strain,
    "solid3d": D_solid3d,
    "mindlin_plate": D_mindlin_plate,
    "beam_ei": EI_beam,
    "beam3d_rigidities": beam3d_rigidities,
}
