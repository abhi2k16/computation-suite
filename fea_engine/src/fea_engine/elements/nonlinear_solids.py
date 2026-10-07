"""
nonlinear_solids.py -- Module 19 (general-purpose extensions roadmap
Phase 6): nonlinear continuum elements, Hex8PlasticJ2 (small-strain J2
plasticity) and Tet4NeoHookean (large-strain compressible hyperelasticity).
Plus Tet10SolidTL (added 2026-08-30, see that class's own docstring):
geometric-only nonlinearity (Total Lagrangian, linear elastic material) on
the quadratic Tet10 element -- a different axis from the first two, which
are both material nonlinearity on LINEAR-order elements.

Hex8PlasticJ2/Tet4NeoHookean SUBCLASS their existing linear-elastic
counterparts (Hex8Solid3D, Tet4Solid3D respectively) rather than
reimplementing shape functions/B_matrix/mass from scratch -- exactly the
"additive, reuse via subclass" pattern Beam2DCorotational (beams.py) and
TrussPlastic2D (trusses.py) already established: geometry/interpolation is
identical to the linear parent, only internal_force()/tangent_stiffness()
(and, for the path-dependent plasticity case, init_state()/commit_state())
differ. Tet10SolidTL follows the same pattern, subclassing Tet10Solid3D.

Split into its own file (rather than added to solids.py, already the
largest element module) since this is a genuinely separate axis of
extension -- material/geometric NONLINEARITY on TOP of existing
elements -- mirroring how beams3d.py was split out for Phase 3 rather
than folded into beams.py.
"""
import numpy as np

from .base import gauss_product, jacobian, tet_quadrature_4pt
from .solids import Hex8Solid3D, Tet4Solid3D, Tet10Solid3D, Quad4PlaneStress
from ..material import (j2_radial_return_3d, j2_radial_return_plane_stress,
                         j2_radial_return_3d_kinematic,
                         neo_hookean_pk2_stress, D_solid3d)


# =====================================================================
# Hex8PlasticJ2 -- small-strain J2 (von Mises) plasticity, 3-D solid
# =====================================================================
class Hex8PlasticJ2(Hex8Solid3D):
    """Hex8Solid3D + small-strain J2 (von Mises) plasticity with linear
    isotropic hardening (material.PlasticMaterialJ2), via the closed-
    form radial-return algorithm in material.j2_radial_return_3d() --
    see that function's docstring for the algorithm and its consistent
    tangent's derivation.

    State is PER-GAUSS-POINT (unlike TrussPlastic2D's single scalar
    eps_p/alpha -- a continuum element genuinely has 2x2x2=8 independent
    stress points, each with its own potentially-different plastic
    history, e.g. a Gauss point near a stress concentration yields
    while one near the neutral axis stays elastic): init_state()
    returns {"eps_p": (n_gp, 6), "alpha": (n_gp,)}, both zero (virgin
    material, same convention as TrussPlastic2D.init_state()).

    mat = a single PlasticMaterialJ2 instance (unlike TrussPlastic2D's
    (PlasticMaterial1D, A) tuple -- there's no separate cross-section
    area here, B_matrix()/detJ already carry all the geometry a
    continuum element needs)."""

    def init_state(self):
        pts, wts = gauss_product(self.gauss_order, self.dim)
        n_gp = len(wts)
        return {"eps_p": np.zeros((n_gp, 6)), "alpha": np.zeros(n_gp)}

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs):
        if state is None:
            state = self.init_state()
        pts, wts = gauss_product(self.gauss_order, self.dim)
        f_int = np.zeros(24)
        for gp, (p, w) in enumerate(zip(pts, wts)):
            B, detJ = self.B_matrix(p, elem_coords)
            eps_voigt = B @ u_elem
            sigma, _, _, _, _ = j2_radial_return_3d(
                eps_voigt, state["eps_p"][gp], state["alpha"][gp], mat)
            f_int += (B.T @ sigma) * detJ * w
        return f_int

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs):
        if state is None:
            state = self.init_state()
        pts, wts = gauss_product(self.gauss_order, self.dim)
        K_T = np.zeros((24, 24))
        for gp, (p, w) in enumerate(zip(pts, wts)):
            B, detJ = self.B_matrix(p, elem_coords)
            eps_voigt = B @ u_elem
            _, D_ep, _, _, _ = j2_radial_return_3d(
                eps_voigt, state["eps_p"][gp], state["alpha"][gp], mat)
            K_T += (B.T @ D_ep @ B) * detJ * w
        return K_T

    def commit_state(self, elem_coords, u_elem, mat, state, thickness=1.0, **kwargs):
        """Replays the return map at every Gauss point (same pattern
        as TrussPlastic2D.commit_state()) and permanently advances
        eps_p/alpha -- call once per CONVERGED load step, never
        mid-Newton-iteration."""
        pts, wts = gauss_product(self.gauss_order, self.dim)
        n_gp = len(wts)
        new_eps_p = np.zeros((n_gp, 6))
        new_alpha = np.zeros(n_gp)
        for gp, (p, w) in enumerate(zip(pts, wts)):
            B, _ = self.B_matrix(p, elem_coords)
            eps_voigt = B @ u_elem
            _, _, eps_p_new, alpha_new, _ = j2_radial_return_3d(
                eps_voigt, state["eps_p"][gp], state["alpha"][gp], mat)
            new_eps_p[gp] = eps_p_new
            new_alpha[gp] = alpha_new
        return {"eps_p": new_eps_p, "alpha": new_alpha}


# =====================================================================
# Quad4PlasticJ2PlaneStress -- small-strain J2 plasticity, PLANE STRESS,
# Wave 2 item 13 (docs/consolidated_future_roadmap.md)
# =====================================================================
class Quad4PlasticJ2PlaneStress(Quad4PlaneStress):
    """Quad4PlaneStress + small-strain J2 (von Mises) plasticity, via
    material.j2_radial_return_plane_stress() -- the plane-stress-
    specific local-Newton return map (Simo & Taylor 1986) that closes
    the gap j2_radial_return_3d()'s own docstring names: Hex8PlasticJ2
    above is 3-D/plane-strain only (closed-form radial return); a
    genuinely plane-stress element (free out-of-plane, sigma_33==0
    enforced rather than eps_33==0) needs the local-Newton-condensed
    version this class drives instead. See j2_radial_return_plane_
    stress()'s own docstring for the full algorithm and why it's built
    as a wrapper around the already-validated 3-D routine rather than
    a fresh derivation.

    Same PER-GAUSS-POINT state pattern as Hex8PlasticJ2, with one
    addition: eps_33 (n_gp,) -- the out-of-plane strain the local
    Newton solves for at each Gauss point, carried forward step to
    step as a warm start (see j2_radial_return_plane_stress()'s own
    docstring for why). mat = a single PlasticMaterialJ2 instance,
    same convention as Hex8PlasticJ2 (Quad4PlaneStress's own B_matrix()/
    detJ already carry the in-plane geometry; thickness is applied the
    same way Quad4PlaneStress's linear stiffness() already does)."""

    def init_state(self):
        pts, wts = gauss_product(self.gauss_order, self.dim)
        n_gp = len(wts)
        return {"eps_p": np.zeros((n_gp, 6)), "alpha": np.zeros(n_gp),
                "eps33": np.zeros(n_gp)}

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs):
        if state is None:
            state = self.init_state()
        pts, wts = gauss_product(self.gauss_order, self.dim)
        n_dof = self.n_nodes * self.dofs_per_node
        f_int = np.zeros(n_dof)
        for gp, (p, w) in enumerate(zip(pts, wts)):
            B, detJ = self.B_matrix(p, elem_coords)
            eps_ps = B @ u_elem
            sigma_ps, _, _, _, _, _ = j2_radial_return_plane_stress(
                eps_ps, state["eps_p"][gp], state["alpha"][gp], state["eps33"][gp], mat)
            f_int += (B.T @ sigma_ps) * detJ * w * thickness
        return f_int

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs):
        if state is None:
            state = self.init_state()
        pts, wts = gauss_product(self.gauss_order, self.dim)
        n_dof = self.n_nodes * self.dofs_per_node
        K_T = np.zeros((n_dof, n_dof))
        for gp, (p, w) in enumerate(zip(pts, wts)):
            B, detJ = self.B_matrix(p, elem_coords)
            eps_ps = B @ u_elem
            _, D_ps, _, _, _, _ = j2_radial_return_plane_stress(
                eps_ps, state["eps_p"][gp], state["alpha"][gp], state["eps33"][gp], mat)
            K_T += (B.T @ D_ps @ B) * detJ * w * thickness
        return K_T

    def commit_state(self, elem_coords, u_elem, mat, state, thickness=1.0, **kwargs):
        """Replays the return map at every Gauss point (same pattern as
        Hex8PlasticJ2.commit_state()) and permanently advances
        eps_p/alpha/eps33 -- call once per CONVERGED load step, never
        mid-Newton-iteration."""
        pts, wts = gauss_product(self.gauss_order, self.dim)
        n_gp = len(wts)
        new_eps_p = np.zeros((n_gp, 6))
        new_alpha = np.zeros(n_gp)
        new_eps33 = np.zeros(n_gp)
        for gp, (p, w) in enumerate(zip(pts, wts)):
            B, _ = self.B_matrix(p, elem_coords)
            eps_ps = B @ u_elem
            _, _, eps_p_new, alpha_new, eps33_new, _ = j2_radial_return_plane_stress(
                eps_ps, state["eps_p"][gp], state["alpha"][gp], state["eps33"][gp], mat)
            new_eps_p[gp] = eps_p_new
            new_alpha[gp] = alpha_new
            new_eps33[gp] = eps33_new
        return {"eps_p": new_eps_p, "alpha": new_alpha, "eps33": new_eps33}


# =====================================================================
# Hex8PlasticJ2Kinematic -- small-strain J2 plasticity, COMBINED
# isotropic + Armstrong-Frederick kinematic hardening, Wave 2 item 14
# (docs/consolidated_future_roadmap.md)
# =====================================================================
class Hex8PlasticJ2Kinematic(Hex8PlasticJ2):
    """Hex8PlasticJ2 + Armstrong-Frederick nonlinear kinematic hardening
    (material.PlasticMaterialJ2Kinematic / material.
    j2_radial_return_3d_kinematic()) -- a SUBCLASS of Hex8PlasticJ2, not
    a replacement, so isotropic-only plasticity keeps working through
    the plain Hex8PlasticJ2 class exactly as before (same "add, don't
    replace" principle as every other Wave 2 item). Only
    j2_radial_return_3d() is swapped for j2_radial_return_3d_kinematic()
    and a per-Gauss-point back-stress "beta" (6,) is added to the state
    dict -- everything else (B_matrix, node/DOF layout, gauss loop
    structure) is inherited unchanged from Hex8Solid3D via Hex8PlasticJ2.

    mat = a PlasticMaterialJ2Kinematic instance (a plain PlasticMaterialJ2
    also works here, since its missing C_kin/gamma_AF simply take
    PlasticMaterialJ2Kinematic's own defaults of 0.0 if constructed as
    that subclass -- but passing a plain PlasticMaterialJ2 object
    directly would fail on the .C_kin/.gamma_AF attribute lookups, so
    callers wanting pure-isotropic behavior through THIS class should
    still construct PlasticMaterialJ2Kinematic(..., C_kin=0, gamma_AF=0)
    explicitly, or simply use plain Hex8PlasticJ2 instead -- which is
    the whole point of keeping both classes available)."""

    def init_state(self):
        pts, wts = gauss_product(self.gauss_order, self.dim)
        n_gp = len(wts)
        return {"eps_p": np.zeros((n_gp, 6)), "alpha": np.zeros(n_gp),
                "beta": np.zeros((n_gp, 6))}

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs):
        if state is None:
            state = self.init_state()
        pts, wts = gauss_product(self.gauss_order, self.dim)
        f_int = np.zeros(24)
        for gp, (p, w) in enumerate(zip(pts, wts)):
            B, detJ = self.B_matrix(p, elem_coords)
            eps_voigt = B @ u_elem
            sigma, _, _, _, _, _ = j2_radial_return_3d_kinematic(
                eps_voigt, state["eps_p"][gp], state["alpha"][gp], state["beta"][gp], mat)
            f_int += (B.T @ sigma) * detJ * w
        return f_int

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, state=None, **kwargs):
        if state is None:
            state = self.init_state()
        pts, wts = gauss_product(self.gauss_order, self.dim)
        K_T = np.zeros((24, 24))
        for gp, (p, w) in enumerate(zip(pts, wts)):
            B, detJ = self.B_matrix(p, elem_coords)
            eps_voigt = B @ u_elem
            _, D_ep, _, _, _, _ = j2_radial_return_3d_kinematic(
                eps_voigt, state["eps_p"][gp], state["alpha"][gp], state["beta"][gp], mat)
            K_T += (B.T @ D_ep @ B) * detJ * w
        return K_T

    def commit_state(self, elem_coords, u_elem, mat, state, thickness=1.0, **kwargs):
        """Replays the return map at every Gauss point (same pattern as
        Hex8PlasticJ2.commit_state()) and permanently advances
        eps_p/alpha/beta -- call once per CONVERGED load step, never
        mid-Newton-iteration."""
        pts, wts = gauss_product(self.gauss_order, self.dim)
        n_gp = len(wts)
        new_eps_p = np.zeros((n_gp, 6))
        new_alpha = np.zeros(n_gp)
        new_beta = np.zeros((n_gp, 6))
        for gp, (p, w) in enumerate(zip(pts, wts)):
            B, _ = self.B_matrix(p, elem_coords)
            eps_voigt = B @ u_elem
            _, _, eps_p_new, alpha_new, beta_new, _ = j2_radial_return_3d_kinematic(
                eps_voigt, state["eps_p"][gp], state["alpha"][gp], state["beta"][gp], mat)
            new_eps_p[gp] = eps_p_new
            new_alpha[gp] = alpha_new
            new_beta[gp] = beta_new
        return {"eps_p": new_eps_p, "alpha": new_alpha, "beta": new_beta}


# =====================================================================
# Tet4NeoHookean -- large-strain compressible hyperelasticity, 3-D solid
# =====================================================================
class Tet4NeoHookean(Tet4Solid3D):
    """Tet4Solid3D + compressible Neo-Hookean hyperelasticity
    (material.NeoHookeanMaterial), Total-Lagrangian formulation, via
    material.neo_hookean_pk2_stress(). Tet4's linear shape functions
    give a CONSTANT reference-configuration gradient dN/dX over the
    element, hence a CONSTANT deformation gradient F -- one evaluation
    per element, exactly like Tet4Solid3D's own linear stiffness()
    needs only one B_matrix() call, no Gauss loop.

    internal_force() is closed-form (f_int_a = V0 * F @ S @ (dN_a/dX),
    the standard Total-Lagrangian virtual-work result -- see this
    class's internal_force() docstring for the derivation).
    tangent_stiffness() is FINITE-DIFFERENCE of that closed-form
    internal_force(), NOT a hand-derived analytic material+geometric
    tangent -- the same choice elements.contact.GapContactCurvedFriction
    (Module 13) already made for a different reason (friction's return-
    map derivative), made here because the full consistent tangent for
    compressible Neo-Hookean needs the 4th-order elasticity tensor
    C_IJKL = 4*d2W/dCdC dotted through both the isochoric AND
    volumetric terms PLUS the geometric (initial-stress) contribution
    -- correct but intricate to hand-derive without a subtle sign/
    index error; a numerically consistent tangent from the already-
    validated closed-form S(F) is safer, still gives Newton-quality
    (quadratic-near-convergence) behavior, and is honestly documented
    as such rather than presented as a hand-derived closed form it
    isn't -- see docs/general_purpose_extensions_roadmap.md Section 4."""

    def _deformation_gradient(self, elem_coords, u_elem):
        (xi, eta, zeta) = (0.25, 0.25, 0.25)   # constant strain -> any point works
        _, dN_nat = self._cached_shape_and_derivs((xi, eta, zeta))
        J_ref, detJ = jacobian(dN_nat, elem_coords)
        dN_dX = np.linalg.solve(J_ref, dN_nat)   # (3,4): reference-config gradients
        u_nodes = u_elem.reshape(4, 3)           # (node, dof)
        F = np.eye(3) + u_nodes.T @ dN_dX.T      # F_iJ = delta_iJ + sum_a u_a_i * dN_a/dX_J
        volume = abs(detJ) / 6.0
        return F, dN_dX, volume

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """f_int_a = V0 * F @ S @ (dN_a/dX) for each node a, from
        virtual work delta_W_int = Integral_V0 S:delta_E dV0 with
        delta_E_IJ (from varying node a, dof i) = sym(dN_a/dX_I * F_iJ)
        and S symmetric -- see nonlinear_solids.py's module docstring."""
        F, dN_dX, volume = self._deformation_gradient(elem_coords, u_elem)
        S, _ = neo_hookean_pk2_stress(F, mat)
        f_int_nodes = volume * (F @ S @ dN_dX)   # (3,4): [dof, node]
        return f_int_nodes.T.reshape(-1)         # node-major (12,), matching B_matrix's dof order

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, h=1e-6,
                           method="fd", **kwargs):
        """method="fd" (default, UNCHANGED behavior): the central finite
        difference described in this class's own docstring above.

        method="autograd" (opt-in, torch side-by-side addition,
        docs/consolidated_future_roadmap.md "PyTorch AD" item): routes
        through autograd_tangent.tet4_neo_hookean_tangent_autograd()
        instead -- the EXACT derivative of the same closed-form
        internal_force() this method's own FD approximates, computed
        via torch.autograd.grad (no h-dependent truncation error, no
        extra internal_force() evaluations beyond building the
        computational graph once). Requires torch installed (raises
        ImportError with a clear message otherwise, via that module's
        own _require_torch()) -- this package's other two solve/
        element paths that touch torch (FESystem's backend="torch" and
        this same autograd_tangent module's Tet10SolidTL cross-check)
        make the SAME "optional dependency, opt-in per-call, never a
        silent default" choice, not a new pattern introduced here.
        Validated (tests/test_torch_autograd_tangent_stiffness.py)
        against this method's own FD result to ~1e-6 relative (FD's
        own truncation-error floor, not autograd's -- autograd itself
        is machine-precision exact) and, independently, against a
        real Newton-Raphson run (both methods converge to the same
        equilibrium state) -- the SAME "cross-check before trusting"
        discipline autograd_tangent.py's own module docstring already
        establishes for Tet10SolidTL's analytic tangent."""
        if method == "autograd":
            from ..autograd_tangent import tet4_neo_hookean_tangent_autograd
            K, _ = tet4_neo_hookean_tangent_autograd(elem_coords, u_elem, mat)
            return K
        elif method != "fd":
            raise ValueError(
                f"Tet4NeoHookean.tangent_stiffness: unknown method={method!r} "
                "-- expected 'fd' (default) or 'autograd'.")

        n = len(u_elem)
        K = np.zeros((n, n))
        for j in range(n):
            du = np.zeros(n)
            du[j] = h
            fp = self.internal_force(elem_coords, u_elem + du, mat, thickness)
            fm = self.internal_force(elem_coords, u_elem - du, mat, thickness)
            K[:, j] = (fp - fm) / (2 * h)
        return 0.5 * (K + K.T)   # symmetrize -- FD roundoff can break exact
        # symmetry by ~1e-9 relative even though the true tangent (a second
        # derivative of a scalar energy) is exactly symmetric; downstream
        # solves (np.linalg.solve on K_T[free,free]) don't need this, but
        # keeping K_T visibly symmetric matches every other element in
        # this package and avoids surprising a caller who checks for it.
        # (method="autograd" above needs no such cleanup -- an exact
        # Hessian-of-a-scalar-potential is already symmetric to float64
        # round-off, see autograd_tangent.py's own docstring.)

    def stiffness(self, elem_coords, mat, thickness=1.0):
        """Initial (zero-displacement) tangent -- interface completeness
        only, matching TrussTL2D's stiffness()."""
        return self.tangent_stiffness(elem_coords, np.zeros(12), mat, thickness)


# =====================================================================
# Tet10SolidTL -- geometric-only nonlinearity (Total Lagrangian, LINEAR
# elastic material) on the quadratic Tet10 element. Added 2026-08-30 for
# the wing-cantilever example in the sibling Multi_Fidelity_NL_
# Structural_ROM project: that paper's own nonlinearity is explicitly
# geometric ("caused by the large deflection phenomenon... stiffness
# nonlinearization"), NOT material -- unlike Hex8PlasticJ2/Tet4NeoHookean
# above, which are both material nonlinearity (J2 plasticity, Neo-Hookean
# hyperelasticity) on LINEAR-order elements. Neither existing class
# matches: this one does, on the SAME quadratic element (Tet10Solid3D)
# already used for that example's linear modal validation.
#
# Formulation: St. Venant-Kirchhoff -- material.D_solid3d(mat) (the SAME
# linear elastic constitutive matrix Tet10Solid3D's own linear stiffness()
# uses) applied to the nonlinear Green-Lagrange strain E = 0.5*(F^T F - I),
# rather than a new hyperelastic energy function. This is the standard
# "geometric nonlinearity with a linear material law" formulation -- the
# Total-Lagrangian continuum-solid counterpart of what TrussTL2D/
# Beam2DCorotational already do for truss/beam elements in this package,
# named "...TL" to match TrussTL2D's own convention.
#
# Per-Gauss-point loop (UNLIKE Tet4NeoHookean, which evaluates its
# constant deformation gradient F just ONCE): Tet4's LINEAR shape
# functions give a constant dN/dX (hence constant F) over the element,
# the exact property Tet4NeoHookean's own docstring leans on for its
# single-evaluation shortcut. Tet10's QUADRATIC shape functions give a
# dN/dX (and hence F) that genuinely varies with position -- confirmed
# directly (not assumed): for a straight-sided element the reference-
# configuration Jacobian/detJ area constant across Gauss points (the
# geometry map stays affine when mid-edge nodes sit at exact edge
# midpoints), but dN/dX itself is still LINEAR in natural coordinates
# (derivative of a quadratic shape function), so F, E, and S all
# genuinely vary point-to-point -- a real Gauss loop is required, reusing
# the SAME tet_quadrature_4pt() 4-point rule Tet10Solid3D.stiffness()/
# mass() already use.
#
# QUADRATURE ACCURACY (a deliberate, documented approximation, not an
# oversight): this element's internal-force integrand
# (dN/dX [linear] contracted through F [linear in u] and S [quadratic in
# u]) is CUBIC overall -- higher degree than tet_quadrature_4pt()'s
# degree-2 exactness. Reusing that same 4-point rule anyway (rather than
# adding a new, higher-order simplex quadrature) mirrors the EXACT same
# choice Tet10Solid3D.mass() already makes for its own quartic N^T*rho*N
# integrand (see that method's own docstring: "a good, standard
# engineering approximation, not machine-precision exact") -- consistent
# with existing precedent in this package, and not worth a new
# quadrature rule's added scope/risk given the wing geometry this was
# built for is itself only an approximate reconstruction.
#
# TANGENT STIFFNESS is an ANALYTIC material+geometric (initial-stress)
# tangent (added 2026-09-02, replacing complex-step differentiation --
# see tangent_stiffness()'s own docstring for the full derivation and
# the validation it was checked against before being trusted as the
# default). Went through two prior tangent implementations, in order:
# a real central finite difference (h=1e-6, the original choice) had a
# noise floor that caused real Newton-Raphson robustness problems in
# downstream use; complex-step differentiation (2026-09-01) fixed that
# noise floor exactly but was still ~10x more expensive per call than
# the analytic tangent now is (measured directly, not estimated -- see
# Multi_Fidelity_NL_Structural_ROM/diag_analytic_tangent_validation.py)
# because it re-evaluates the full internal_force() Gauss loop once per
# DOF (30 times) rather than assembling the tangent in a single pass.
# That cost was the STATED reason the sibling Multi_Fidelity_NL_
# Structural_ROM project's wing-cantilever mesh had to stay coarse
# (~78:1 element aspect ratio) -- removing it is what makes a finer,
# lower-aspect-ratio mesh computationally feasible there. The complex-
# step tangent is KEPT (as `_tangent_stiffness_complex_step()`, not
# deleted) as the independent reference the analytic tangent was
# validated against and as a debugging fallback.
# =====================================================================
class Tet10SolidTL(Tet10Solid3D):
    """Tet10Solid3D + Total-Lagrangian geometric nonlinearity, LINEAR
    elastic material (material.D_solid3d) applied to the Green-Lagrange
    strain -- see this module's own comment block above for the full
    derivation and the deliberate quadrature/tangent choices. No new
    material class needed (unlike Tet4NeoHookean's NeoHookeanMaterial):
    `mat` here is a plain Material (E, nu), exactly what Tet10Solid3D's
    own linear stiffness()/mass() already take.

    No path-dependent state (unlike Hex8PlasticJ2): internal_force()/
    tangent_stiffness() are pure functions of (elem_coords, u_elem),
    since geometric nonlinearity alone has no history/loading-path
    dependence the way plasticity does.

    nonlinear_solver.solve_nonlinear_static()'s default tol=1e-8 is
    STILL TOO TIGHT for this element even after tangent_stiffness()
    was switched to complex-step differentiation (2026-09-01, see that
    method's own docstring) -- re-checked directly, not assumed: the
    residual plateaus around 1e-7 REGARDLESS of tangent accuracy (an
    exact-to-1e-16 complex-step tangent hits the identical plateau a
    real h=1e-6 FD tangent did), so this floor was mis-attributed to
    the old FD tangent's truncation; it is actually floating-point
    cancellation inherent to internal_force()'s own F@S@dN_dX
    evaluation at this element's stiffness/displacement scale, and the
    tangent choice doesn't touch it. Pass tol=1e-5 or so explicitly
    when driving this element through solve_nonlinear_static() -- see
    tests/test_tet10_geometric_nonlinear.py for a working example.

    Also checked directly, and worth flagging for whoever next drives
    this element through a LARGE-amplitude/coarse-dt transient (e.g.
    solve_nonlinear_transient() on a kinematic, non-equilibrium IC):
    the complex-step tangent measurably reduces cost and removes
    non-monotonic per-step behavior WHERE Newton already converged
    (e.g. one wing-cantilever case went from 5.65-10.15s/step,
    non-monotonically in dt, down to a consistent ~2.3s/step), but it
    does NOT extend the stable dt range for large-deformation
    non-equilibrium ICs -- a case that failed at dt=2ms with the old
    FD tangent still fails at dt=2ms (and, at one tested amplitude, now
    also fails at dt=1ms where the noisier FD tangent had happened to
    converge). An imprecise Jacobian can accidentally act like implicit
    regularization on a full Newton step; a genuinely exact one does
    not add that accidental forgiveness back. Fixing THAT failure mode
    needs a globalization strategy solve_nonlinear_transient() doesn't
    have yet (backtracking line search, or automatic dt bisection on
    Newton failure) -- see Multi_Fidelity_NL_Structural_ROM/
    paper_notes.md's 2026-09-01 wing-dynamic-pipeline write-up for the
    original failures this was investigated from.

    **Update (2026-09-02): the globalization strategy flagged above as
    missing now exists** (solve_nonlinear_transient() gained trust-
    region-capped Newton as an escalation, plus line search as a final
    fallback -- see that function's own docstring), AND
    tangent_stiffness() itself is no longer complex-step (see this
    class's own tangent_stiffness()/`_tangent_stiffness_complex_step()`
    docstrings) -- both changes were driven by the same downstream wing-
    cantilever work, now root-caused to this element's own extreme (up
    to ~78:1) aspect ratio at the mesh sizes that project's Newton-cost
    budget forced it into: geometric strain is quadratic in displacement
    GRADIENT, and gradients through a single thin element layer get
    amplified by dividing by that tiny thickness, so nonlinear terms can
    dominate the linear ones at physically tiny (micron-scale)
    displacements. The analytic tangent doesn't fix that aspect-ratio
    sensitivity itself, but removes the cost penalty that made a finer,
    lower-aspect-ratio mesh infeasible to even try."""

    def _green_lagrange_pk2(self, F, D):
        """E (Green-Lagrange, tensor) -> E_voigt (engineering-shear
        convention) -> S_voigt = D @ E_voigt -> S (2nd Piola-Kirchhoff,
        tensor). Shared by internal_force() (needs S as a 3x3 tensor to
        contract with F) -- kept as its own method rather than inlined
        so the strain/stress conversion logic isn't duplicated if this
        class ever needs S elsewhere (e.g. a future stress-recovery
        diagnostic)."""
        E = 0.5 * (F.T @ F - np.eye(3))
        E_voigt = np.array([
            E[0, 0], E[1, 1], E[2, 2],
            2 * E[0, 1], 2 * E[1, 2], 2 * E[0, 2],
        ])
        S_voigt = D @ E_voigt
        S = np.array([
            [S_voigt[0], S_voigt[3], S_voigt[5]],
            [S_voigt[3], S_voigt[1], S_voigt[4]],
            [S_voigt[5], S_voigt[4], S_voigt[2]],
        ])
        return S

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """f_int = sum_gp w_gp * (V0 weight) * (F @ S @ dN/dX), the same
        per-Gauss-point virtual-work expression Tet4NeoHookean.
        internal_force() uses for its single point -- summed over all 4
        quadrature points here since F/S genuinely vary (see this
        module's own comment block above)."""
        D = D_solid3d(mat)
        u_nodes = u_elem.reshape(10, 3)   # (node, dof), node-major like Tet4NeoHookean
        points, weights = tet_quadrature_4pt()
        # dtype follows u_elem, not hardcoded float64 -- tangent_stiffness()
        # calls this with a COMPLEX u_elem (complex-step differentiation,
        # see that method's own docstring), and a fixed-float64 accumulator
        # would reject the complex result being added into it.
        f_int = np.zeros(30, dtype=u_elem.dtype)
        for p, w in zip(points, weights):
            _, dN_nat = self._cached_shape_and_derivs(p)
            J, detJ = jacobian(dN_nat, elem_coords)
            dN_dX = np.linalg.solve(J, dN_nat)          # (3,10): reference-config gradients
            F = np.eye(3) + u_nodes.T @ dN_dX.T          # F_iJ = delta_iJ + sum_a u_a_i * dN_a/dX_J
            S = self._green_lagrange_pk2(F, D)
            f_int_nodes = F @ S @ dN_dX                  # (3,10): [dof, node]
            f_int += f_int_nodes.T.reshape(-1) * abs(detJ) * w / 6.0
        return f_int

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """Analytic Total-Lagrangian/St. Venant-Kirchhoff tangent
        (added 2026-09-02, replacing complex-step differentiation --
        see `_tangent_stiffness_complex_step()` below, KEPT as the
        validation reference this was checked against, not dead code).
        Standard material + geometric (initial-stress) split (e.g.
        Bathe, "Finite Element Procedures," Sec. 6.3.3; Belytschko/Liu/
        Moran, "Nonlinear Finite Elements," Ch. 4) -- not a novel
        derivation, but implemented and validated here for the first
        time for this element:

        K_T = Integral_V0 (B_L^T D B_L + K_geo) dV0

        B_L (6x30) is the NONLINEAR (displacement-dependent)
        strain-displacement matrix, dE_voigt = B_L @ du, derived from
        dE_IJ/d(u_b_k) = 0.5*(F_kI * dNb/dXJ + F_kJ * dNb/dXI) (the
        standard result: E is linear in F, F is affine in u, so dE/du
        is linear in the CURRENT F). At F=I (u=0) this reduces exactly
        to Tet10Solid3D's own linear B_matrix() (checked directly, e.g.
        row 3 (2*E_xy): F=I gives dNa[1]*(1,0,0)+dNa[0]*(0,1,0), i.e.
        coefficients (dNa/dy, dNa/dx) on the node's (x,y) DOFs -- the
        exact B[3,c]=dy, B[3,c+1]=dx convention B_matrix() already
        uses, confirming both the derivation and the Voigt ordering
        match).

        K_geo (30x30) is the geometric/initial-stress contribution from
        differentiating F itself (not just E) through the f_int = F @ S
        @ dN/dX product: block (a,b) = (dNa/dX . S . dNb/dX) * I_3,
        i.e. `K_geo = kron(dN_dX.T @ S @ dN_dX, I_3)` -- this is why an
        analytic tangent needs BOTH terms and can't just reuse a linear
        B^T D B: the geometric term has no small-strain analogue (it
        vanishes at S=0, e.g. u=0) and is exactly what makes this
        tangent capture stress-stiffening/softening under large
        rotation.

        Same per-Gauss-point loop and abs(detJ)*w/6.0 quadrature
        weighting as internal_force() (see this module's own comment
        block above for why a real Gauss loop is required here, unlike
        Tet4NeoHookean's single-point shortcut).

        Why this replaces complex-step: validated (see
        diag_analytic_tangent_validation.py in the sibling
        Multi_Fidelity_NL_Structural_ROM project, run 2026-09-02) to
        agree with the complex-step tangent to ~1e-10 relative error
        at both u=0 and large, non-trivial displacements, while costing
        ~30-60x less per call (no more looping internal_force() once
        per DOF) -- the complex-step tangent's ~1000x-vs-analytic-beam-
        element cost (see this module's earlier comment block) was the
        stated reason the sibling project's wing-cantilever mesh had to
        stay coarse (30 elements, ~78:1 aspect ratio) in the first
        place; removing that penalty is what makes a finer, more
        isotropic mesh computationally feasible."""
        D = D_solid3d(mat)
        u_nodes = np.asarray(u_elem, dtype=float).reshape(10, 3)
        points, weights = tet_quadrature_4pt()
        K = np.zeros((30, 30))
        for p, w in zip(points, weights):
            _, dN_nat = self._cached_shape_and_derivs(p)
            J, detJ = jacobian(dN_nat, elem_coords)
            dN_dX = np.linalg.solve(J, dN_nat)           # (3,10)
            F = np.eye(3) + u_nodes.T @ dN_dX.T           # (3,3)
            S = self._green_lagrange_pk2(F, D)            # (3,3)
            scale = abs(detJ) * w / 6.0

            B_L = np.zeros((6, 30))
            F0, F1, F2 = F[:, 0], F[:, 1], F[:, 2]
            for a in range(10):
                dNa = dN_dX[:, a]
                cols = slice(3 * a, 3 * a + 3)
                B_L[0, cols] = dNa[0] * F0
                B_L[1, cols] = dNa[1] * F1
                B_L[2, cols] = dNa[2] * F2
                B_L[3, cols] = dNa[1] * F0 + dNa[0] * F1
                B_L[4, cols] = dNa[2] * F1 + dNa[1] * F2
                B_L[5, cols] = dNa[2] * F0 + dNa[0] * F2

            SG = dN_dX.T @ S @ dN_dX                      # (10,10)
            K_geo = np.kron(SG, np.eye(3))                # (30,30)

            K += (B_L.T @ D @ B_L + K_geo) * scale
        return 0.5 * (K + K.T)   # symmetrize -- true tangent (a second
        # derivative of a scalar energy) is exactly symmetric; this only
        # cleans up floating-point roundoff from summation order, kept
        # for interface consistency with the complex-step reference below.

    def _tangent_stiffness_complex_step(self, elem_coords, u_elem, mat, thickness=1.0, h=1e-20, **kwargs):
        """Complex-step differentiation of internal_force() -- the
        ORIGINAL tangent_stiffness() implementation (2026-09-01),
        RENAMED (not removed) on 2026-09-02 when the analytic tangent
        above replaced it as the default: kept as the independent
        validation reference the analytic tangent was checked against,
        and as a debugging fallback if the analytic tangent is ever
        suspected of a derivation/implementation bug on a new use case.

        Why complex-step applies cleanly here: internal_force() is an
        EXACT polynomial in u_elem (F is affine in u, E=0.5*(F^T F - I)
        is quadratic, S=D@E is linear in E, f_int=F@S@dN_dX is cubic
        overall), and every operation inside it is holomorphic in
        u_elem -- the one abs() call (on detJ) is a function of
        elem_coords alone, never of u_elem, so it never breaks
        differentiability of the perturbed evaluation. That makes
        Im(f_int(u + i*h*e_j)) / h exact (no subtractive-cancellation
        error, unlike a real difference of two nearby real evaluations)
        up to an O(h^2) term from the internal force's own nonzero
        third derivative (it's cubic, not quadratic) -- utterly
        negligible at h=1e-20 (h^2=1e-40), so this recovers the tangent
        to floating-point precision, not just an improved approximation
        of it.

        Requires internal_force() to stay complex-safe (no abs/max/
        branching on any u_elem-derived quantity) -- true today, and
        worth re-checking if internal_force() is ever modified."""
        n = len(u_elem)
        u_c = np.asarray(u_elem, dtype=complex)
        K = np.zeros((n, n))
        for j in range(n):
            du = np.zeros(n, dtype=complex)
            du[j] = 1j * h
            f_pert = self.internal_force(elem_coords, u_c + du, mat, thickness)
            K[:, j] = f_pert.imag / h
        return 0.5 * (K + K.T)

    def stiffness(self, elem_coords, mat, thickness=1.0):
        """Initial (zero-displacement) tangent -- interface completeness
        only, matching Tet4NeoHookean/TrussTL2D's own stiffness(). Exactly
        recovers Tet10Solid3D.stiffness(elem_coords, D_solid3d(mat),
        thickness) at u=0 (verified directly in
        tests/test_tet10_geometric_nonlinear.py), since at zero
        displacement F=I, E=0, and the St. Venant-Kirchhoff tangent
        reduces exactly to the linear elastic one."""
        return self.tangent_stiffness(elem_coords, np.zeros(30), mat, thickness)
