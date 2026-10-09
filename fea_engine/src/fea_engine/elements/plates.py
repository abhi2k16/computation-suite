# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
plates.py -- Quad4MindlinPlate (selective reduced integration).

Split out of the original monolithic element.py during the
fea_engine restructuring; no logic changed, only file location.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .base import Element, gauss_product, jacobian


class Quad4MindlinPlate(Element):
    n_nodes, dofs_per_node, dim, gauss_order = 4, 3, 2, 2
    translational_dof_mask = [True, False, False]   # [w, betax, betay]
    # element-native names, kept as documented (betax/betay are Mindlin section rotations, not
    # simply rotations about x/y, so they are deliberately NOT renamed rx/ry)
    dof_names = ("w", "betax", "betay")
    dof_aliases = {"uz": "w"}

    def shape_and_derivs(self, natural_coords):
        xi, eta = natural_coords
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                              (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        return N, np.vstack([dN_dxi, dN_deta])

    def _Bb_Bs(self, natural_coords, elem_coords):
        N, dN_nat = self._cached_shape_and_derivs(natural_coords)
        J, detJ = jacobian(dN_nat, elem_coords)
        dN_g = np.linalg.solve(J, dN_nat)
        Bb = np.zeros((3, 12))
        Bs = np.zeros((2, 12))
        for a in range(4):
            c = 3 * a
            Bb[0, c + 1] = dN_g[0, a]
            Bb[1, c + 2] = dN_g[1, a]
            Bb[2, c + 1] = dN_g[1, a]; Bb[2, c + 2] = dN_g[0, a]
            Bs[0, c] = dN_g[0, a]; Bs[0, c + 1] = -N[a]
            Bs[1, c] = dN_g[1, a]; Bs[1, c + 2] = -N[a]
        return Bb, Bs, detJ

    def stiffness(self, elem_coords, D, thickness=1.0, integration='sri'):
        """D here is the (Db, Ds) pair from config.D_mindlin_plate.

        integration:
          'sri'     (default, CORRECT) -- full 2x2 Gauss for bending,
                    1x1 (element-center) reduced for shear. The
                    standard fix for shear locking in this element --
                    same as in cantilever_plate_fem.py, where a
                    quadrature-WEIGHT bug that accidentally
                    under-integrated both terms by the same factor was
                    found and fixed (see that script's history). This
                    is what solver.assemble_stiffness() uses by default.
          'full'    -- full 2x2 Gauss for BOTH bending and shear.
                    Reproduces shear locking on purpose -- useful for
                    demonstrating why 'sri' is necessary (see main.py),
                    not for a result you'd actually use.
          'reduced' -- 1x1 reduced Gauss for BOTH bending and shear.
                    Avoids locking but is rank-deficient for a single
                    element (hourglass modes) -- see
                    spurious_zero_energy_modes()."""
        Db, Ds = D
        if integration == 'sri':
            bending_order, shear_reduced = self.gauss_order, True
        elif integration == 'full':
            bending_order, shear_reduced = self.gauss_order, False
        elif integration == 'reduced':
            bending_order, shear_reduced = max(1, self.gauss_order - 1), True
        else:
            raise ValueError(f"unknown integration mode: {integration!r}")

        ke = np.zeros((12, 12))
        pts, wts = gauss_product(bending_order, self.dim)
        for p, w in zip(pts, wts):
            Bb, _, detJ = self._Bb_Bs(p, elem_coords)
            ke += (Bb.T @ Db @ Bb) * detJ * w

        if shear_reduced:
            Bb0, Bs0, detJ0 = self._Bb_Bs((0.0, 0.0), elem_coords)
            ke += (Bs0.T @ Ds @ Bs0) * detJ0 * 4.0   # 1-pt rule, weight = 2*2
        else:
            pts_s, wts_s = gauss_product(self.gauss_order, self.dim)
            for p, w in zip(pts_s, wts_s):
                _, Bs, detJ = self._Bb_Bs(p, elem_coords)
                ke += (Bs.T @ Ds @ Bs) * detJ * w
        return ke

    def full_stiffness(self, elem_coords, D, thickness=1.0):
        return self.stiffness(elem_coords, D, thickness, integration='full')

    def reduced_stiffness(self, elem_coords, D, thickness=1.0):
        return self.stiffness(elem_coords, D, thickness, integration='reduced')

    def mass(self, elem_coords, rho_matrix, thickness=1.0):
        """rho_matrix = diag([rho*h, rho*h^3/12, rho*h^3/12]) for
        translational + rotary inertia -- full integration, no locking
        issue for the mass matrix."""
        return super().mass(elem_coords, rho_matrix, thickness)


# =====================================================================
# 2-node Euler-Bernoulli beam (closed-form Hermite element -- not
# isoparametric in the same sense as the elements above, so it
# overrides both stiffness() and mass() with the standard textbook
# 4x4 matrices instead of using the generic Gauss loop)
# =====================================================================
