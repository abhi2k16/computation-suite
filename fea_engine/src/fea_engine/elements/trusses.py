# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
trusses.py -- 2-D truss elements: TrussTL2D (geometrically
nonlinear, total-Lagrangian) and TrussPlastic2D (small-
displacement, 1-D J2 return-mapping plasticity).

Split out of the original monolithic element.py during the
fea_engine restructuring; no logic changed, only file location.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .base import Element


class TrussTL2D(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 2, 2, 1, 1

    def shape_and_derivs(self, natural_coords):
        """Linear interpolation along the bar axis -- provided only for
        interface completeness (e.g. mesh.check_quality()); the actual
        internal_force()/tangent_stiffness() below don't use it."""
        (xi,) = natural_coords
        N = np.array([(1 - xi) / 2, (1 + xi) / 2])
        dN_dxi = np.array([-0.5, 0.5])
        return N, dN_dxi.reshape(1, 2)

    @staticmethod
    def _kinematics(elem_coords, u_elem):
        X1, X2 = elem_coords[0], elem_coords[1]
        u1, u2 = u_elem[0:2], u_elem[2:4]
        L0 = np.linalg.norm(X2 - X1)
        dx = (X2 + u2) - (X1 + u1)          # current relative position vector
        l2 = dx @ dx
        E_GL = (l2 - L0**2) / (2.0 * L0**2)  # Green-Lagrange strain (1-D)
        return L0, dx, E_GL

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """mat = (E, A). f_int conjugate to (u1,u2): f2 = (S*A/L0)*dx,
        f1 = -f2, where S = E*E_GL is the 2nd Piola-Kirchhoff stress
        (St Venant-Kirchhoff: S linear in E_GL) -- this is exactly the
        virtual-work-consistent TL truss force (Crisfield eq. 3.x), NOT
        a naive "resolve a physical force along the current bar axis"
        shortcut; the two coincide only because of the special 1-D
        geometry of a truss, and this is the form that generalizes to
        beams/continua."""
        E, A = mat
        L0, dx, E_GL = self._kinematics(elem_coords, u_elem)
        S = E * E_GL
        f2 = (S * A / L0) * dx
        return np.concatenate([-f2, f2])

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, **kwargs):
        """K_T = K_material + K_geometric (consistent linearization of
        internal_force() above, i.e. d(f_int)/d(u) -- verified against
        a finite-difference check, see the validation script):
            K_material  = (E*A/L0^3) * dx (x) dx     -- from d(S)/du
            K_geometric = (S*A/L0)   * I2            -- from d(dx)/du
        K_geometric is the hallmark of geometric nonlinearity: even a
        ZERO-stiffness material would still resist transverse motion
        once it's carrying axial force S (this is literally what keeps
        a guitar string / cable taut and is why it's absent from every
        linear element in this package)."""
        E, A = mat
        L0, dx, E_GL = self._kinematics(elem_coords, u_elem)
        S = E * E_GL
        Ksub = (E * A / L0**3) * np.outer(dx, dx) + (S * A / L0) * np.eye(2)
        K = np.zeros((4, 4))
        K[0:2, 0:2] = Ksub;  K[0:2, 2:4] = -Ksub
        K[2:4, 0:2] = -Ksub; K[2:4, 2:4] = Ksub
        return K

    def stiffness(self, elem_coords, mat, thickness=1.0):
        """Initial (zero-displacement) tangent -- provided so this
        element still satisfies the linear-element interface (e.g. an
        initial K for eigenvalue/pre-stress checks), NOT used by
        nonlinear_solver.py, which always calls tangent_stiffness()
        with the current, generally nonzero, displacement state."""
        return self.tangent_stiffness(elem_coords, np.zeros(4), mat, thickness)

    def geometric_stiffness(self, elem_coords, N, thickness=1.0):
        """Module 19 (general-purpose extensions Phase 4): this is
        LITERALLY the K_geometric term already inside
        tangent_stiffness() above -- (S*A/L0)*I -- pulled out on its
        own so FESystem.assemble_geometric_stiffness() can build a
        buckling K_sigma from a reference axial force N (=S*A,
        TENSION-POSITIVE, matching this element's own S convention)
        without needing a full nonlinear displacement state u_elem.
        No new physics here, just exposing an already-existing,
        already-validated piece of this element's own tangent through
        the Module 19 extension point."""
        X1, X2 = elem_coords[0], elem_coords[1]
        L0 = np.linalg.norm(X2 - X1)
        Ksub = (N / L0) * np.eye(2)
        K = np.zeros((4, 4))
        K[0:2, 0:2] = Ksub;  K[0:2, 2:4] = -Ksub
        K[2:4, 0:2] = -Ksub; K[2:4, 2:4] = Ksub
        return K


# =====================================================================
# 2-node planar corotational beam-column (Module 15: large-displacement/
# large-ROTATION geometric nonlinearity for BENDING members, extending
# Module 8's geometric-nonlinearity machinery -- TrussTL2D -- from
# axial-only to bending). 3 dof/node [u, v, theta] -- unlike TrussTL2D/TrussPlastic2D,
# this element carries bending, so it needs a rotational DOF and a
# THIRD "natural" (rigid-body-mode-free) local deformation beyond the
# truss's single axial stretch.
#
# Corotational idea: attach a local frame to the current CHORD between
# the two end nodes (angle phi = atan2(dy,dx) of the deformed chord),
# track how far the beam has rotated as a rigid body since the
# reference configuration (beta = phi - phi0), and measure LOCAL
# ("natural") deformations relative to that rotating frame:
#     e_bar      = L - L0                (chord stretch)
#     theta1_bar = theta1 - beta          (end-1 rotation relative to chord)
#     theta2_bar = theta2 - beta          (end-2 rotation relative to chord)
# By construction these three are EXACTLY zero under any rigid
# translation+rotation of the whole element (validate_nonlinear_beam.py
# CHECK 2) -- L, theta1, theta2, and beta all change together under a
# rigid motion, but e_bar/theta1_bar/theta2_bar don't, which is the
# entire point of "co-rotating" the frame with the element instead of
# using a FIXED reference frame (as TrussPlastic2D does -- fine for a
# 2-dof/node axial-only bar with small rotations, wrong here).
#
# Local force-deformation is the ORDINARY linear Euler-Bernoulli 3x3
# "natural stiffness" (the same physics as Beam2DEulerBernoulli's 4x4
# matrix, with the 2 rigid-body modes -- axial rigid translation isn't
# even a mode here since e_bar already removes it, and overall rigid
# rotation -- projected out):
#     N  = (EA/L0) * e_bar
#     M1 = (EI/L0) * (4*theta1_bar + 2*theta2_bar)
#     M2 = (EI/L0) * (2*theta1_bar + 4*theta2_bar)
# ALL the nonlinearity is in how (e_bar, theta1_bar, theta2_bar) relate
# to the 6 global DOFs -- exactly Module 8's "geometric, not material"
# nonlinearity, same as TrussTL2D.
#
# Global internal force: f_global = B^T @ [N, M1, M2], where B (3x6) is
# d(local deformations)/d(global dofs) -- see _B_matrix(). This is
# virtual-work consistent by construction (internal virtual work
# f_local . delta(local defs) = f_local . B . delta(global dofs) =
# (B^T f_local) . delta(global dofs)), the same pattern TrussTL2D uses.
#
# tangent_stiffness() below is HAND-DERIVED (unlike
# GapContactCurvedFriction's numerical tangent) because the resulting
# closed form is compact enough to state and verify directly:
#     K_T = B^T @ k_local @ B  +  K_geo
# k_local = d(local force)/d(local deformations) -- the "material"
# part, constant, same 3x3 form as Beam2DEulerBernoulli's bending
# sub-block. K_geo is the GEOMETRIC stiffness -- d(B^T)/d(global dofs)
# contracted with the CURRENT [N,M1,M2] -- obtained by direct
# differentiation of B's entries (c=cos(phi), s=sin(phi), L all depend
# on the global dofs through dx = (X2+u2)-(X1+u1)): the axial force N
# contributes N*L*(z (x) z) (a rank-1 "string-stiffening" term, the
# beam analogue of TrussTL2D's K_geometric = (S*A/L)*I), and the end
# moments contribute an (M1+M2)-weighted term from how the chord angle
# itself rotates with the end displacements -- both act ONLY on the
# translational (u1,v1,u2,v2) sub-block, since B's spatial dependence
# (on c,s,L) doesn't involve theta1/theta2 at all. Independently
# verified against a finite-difference tangent of internal_force() in
# validate_nonlinear_beam.py CHECK 3 (max rel error < 1e-5 across
# random rotated/stretched states) -- not merely self-consistent by
# construction, since it was derived separately from internal_force().
# =====================================================================

class TrussPlastic2D(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 2, 2, 1, 1

    def shape_and_derivs(self, natural_coords):
        """Linear interpolation along the bar axis -- interface
        completeness only, as in TrussTL2D."""
        (xi,) = natural_coords
        N = np.array([(1 - xi) / 2, (1 + xi) / 2])
        dN_dxi = np.array([-0.5, 0.5])
        return N, dN_dxi.reshape(1, 2)

    @staticmethod
    def init_state():
        """Virgin material: zero plastic strain, zero accumulated
        plastic strain (so the initial yield stress is exactly
        mat.sigma_y, with no prior hardening)."""
        return {"eps_p": 0.0, "alpha": 0.0}

    @staticmethod
    def _return_map(elem_coords, u_elem, mat, state):
        """Elastic-predictor / radial-return step, using the FIXED
        (undeformed) bar orientation c -- valid because this element's
        kinematics are small-displacement, so c doesn't rotate with
        the deformation (that's TrussTL2D's job). Returns (c, L0,
        sigma, Et, eps_p_trial, alpha_trial) -- a TRIAL response only;
        does not touch `state`, which the caller may or may not choose
        to persist (internal_force()/tangent_stiffness() don't;
        commit_state() does)."""
        plas_mat, A = mat
        E, sigma_y, H = plas_mat.E, plas_mat.sigma_y, plas_mat.H
        X1, X2 = elem_coords[0], elem_coords[1]
        L0 = np.linalg.norm(X2 - X1)
        c = (X2 - X1) / L0                      # fixed reference direction
        u1, u2 = u_elem[0:2], u_elem[2:4]
        e = c @ (u2 - u1)                       # axial elongation (linear)
        eps_total = e / L0

        eps_p_n, alpha_n = state["eps_p"], state["alpha"]
        eps_e_trial = eps_total - eps_p_n
        sigma_trial = E * eps_e_trial
        f_trial = abs(sigma_trial) - (sigma_y + H * alpha_n)

        if f_trial <= 0.0:
            return c, L0, sigma_trial, E, eps_p_n, alpha_n   # elastic step

        dgamma = f_trial / (E + H)              # closed-form (linear yield fn)
        sign = np.sign(sigma_trial)
        sigma = sigma_trial - E * dgamma * sign
        eps_p_new = eps_p_n + dgamma * sign
        alpha_new = alpha_n + dgamma            # alpha always increases
        Et = E * H / (E + H)                    # consistent elastoplastic tangent
        return c, L0, sigma, Et, eps_p_new, alpha_new

    def internal_force(self, elem_coords, u_elem, mat, thickness=1.0, state=None):
        if state is None:
            state = self.init_state()   # allows direct/stateless calls (tests)
        _, A = mat
        c, L0, sigma, Et, _, _ = self._return_map(elem_coords, u_elem, mat, state)
        f2 = (sigma * A) * c
        return np.concatenate([-f2, f2])

    def tangent_stiffness(self, elem_coords, u_elem, mat, thickness=1.0, state=None):
        if state is None:
            state = self.init_state()
        _, A = mat
        c, L0, sigma, Et, _, _ = self._return_map(elem_coords, u_elem, mat, state)
        Ksub = (Et * A / L0) * np.outer(c, c)
        K = np.zeros((4, 4))
        K[0:2, 0:2] = Ksub;  K[0:2, 2:4] = -Ksub
        K[2:4, 0:2] = -Ksub; K[2:4, 2:4] = Ksub
        return K

    def commit_state(self, elem_coords, u_elem, mat, state, thickness=1.0):
        """Called once per CONVERGED load step (see
        solver.FESystem.commit_all_states() / nonlinear_solver.py) --
        replays the same return map and permanently advances eps_p/alpha.
        Never call this mid-Newton-iteration; internal_force()/
        tangent_stiffness() above are what iterations use instead."""
        _, _, _, _, eps_p_new, alpha_new = self._return_map(elem_coords, u_elem, mat, state)
        return {"eps_p": eps_p_new, "alpha": alpha_new}

    def stiffness(self, elem_coords, mat, thickness=1.0, **kwargs):
        """Initial (virgin, zero-displacement) elastic tangent -- see
        TrussTL2D.stiffness()'s docstring for why this exists."""
        return self.tangent_stiffness(elem_coords, np.zeros(4), mat, thickness,
                                       state=self.init_state())


# =====================================================================
# 1-node unilateral gap/contact element, PENALTY formulation (Module 10:
# boundary/contact nonlinearity, "the status switch" -- zero stiffness
# while apart, an immense compressive stiffness the instant contact
# occurs, e.g. a column hitting a safety stopper, a crash barrier). A
# single structural node approaches a FIXED rigid obstacle (not part of
# the mesh, not a DOF) along a prescribed direction n_hat, with initial
# gap g0. This is the classic textbook "gap element": stateless (unlike
# TrussPlastic2D -- whether contact is active depends only on the
# CURRENT position, not history, so there's no init_state()/
# commit_state() here), and plugs into the EXISTING
# internal_force()/tangent_stiffness() extension point + FESystem's
# generic nonlinear assembly with zero changes to nonlinear_solver.py
# -- solve_nonlinear_static() already solves a penalty-contact problem
# correctly the moment a contact element is registered via
# FESystem.add_contact_element(), the same payoff the additive design
# has delivered for every module so far.
#
# mat = (k_p, g0, n_hat): k_p = penalty stiffness (a numerical
# parameter, not a physical property -- large enough that penetration
# is small relative to the structure's own deformation, but not so
# large that K_T becomes ill-conditioned; see validate_contact.py for
# how penetration shrinks, and the contact force converges to the
# EXACT Lagrange-multiplier solution, as k_p increases). g0 = initial
# gap (>=0). n_hat = unit vector, the direction of approach (positive
# = penetrating).
# =====================================================================
