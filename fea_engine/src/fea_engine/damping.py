"""
damping.py -- damping models needed by solver.py's dynamics solve
methods (solve_transient_implicit/explicit, solve_harmonic, ...).

Split out of the original config.py during the fea_engine restructuring
(kept separate from material.py because damping is a system-level
property specification, not a constitutive/material law) -- no logic
changed, only file location.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class RayleighDamping:
    """C = alpha*M + beta*K. Use .calibrate() to solve for (alpha,beta)
    that gives a target damping ratio at two chosen modal frequencies
    (rad/s) -- the same calibration used throughout this project's
    earlier beam scripts."""
    alpha: float
    beta: float

    @classmethod
    def calibrate(cls, omega_i: float, omega_j: float, zeta: float):
        A = np.array([[1 / (2 * omega_i), omega_i / 2],
                       [1 / (2 * omega_j), omega_j / 2]])
        alpha, beta = np.linalg.solve(A, [zeta, zeta])
        return cls(alpha, beta)

    def modal_ratio(self, omega: float) -> float:
        """Resulting damping ratio at an arbitrary modal frequency
        omega (rad/s), for reporting/diagnostics."""
        return (self.alpha + self.beta * omega**2) / (2 * omega)


@dataclass
class ModalDamping:
    """A damping ratio (or array of ratios, one per retained mode)
    applied directly to the decoupled modal equations in
    solver.solve_modal_superposition() -- no global C matrix is ever
    assembled for this path, since modal superposition doesn't need one."""
    zeta: float


@dataclass
class FieldDamping:
    """Wave 17 item 141 (docs/consolidated_future_roadmap.md): a
    field-wise ("per DOF-type") consistent viscous damping matrix,
    the damping model Georgiou 2005 actually uses (its Eqs. 1-3, 24,
    35 apply ONE scalar D uniformly to every field -- u1, u2, theta --
    NOT a Rayleigh alpha*M+beta*K model). Structurally, C is built
    exactly like a consistent mass matrix, with the density-like
    quantity replaced field-by-field by a damping coefficient:

        M = sum_gp  Nm^T rho_matrix Nm |J| w        (rho_matrix has
                                                       rhoA, rhoA, rhoI
                                                       on its diagonal
                                                       for a 3-field
                                                       beam, say)
        C = sum_gp  Nm^T  D_matrix  Nm |J| w        (D_matrix has
                                                       D_1, D_2, D_3
                                                       in the SAME
                                                       slots instead)

    This is exactly the SAME shape-function/quadrature structure
    element.Element.mass() (and every element-specific mass()
    override) already builds -- so rather than duplicate any
    per-element interpolation logic, FieldDamping literally calls the
    target element's own mass(elem_coords, coefficients, **kwargs)
    with `coefficients` substituted for that element's usual density
    argument. No element in this package factors its mass-matrix
    shape-function structure out into a separate reusable hook (each
    override -- e.g. Beam2DEulerBernoulli's closed-form Hermite
    formula, Beam2DCorotational's rotated local mass -- is a
    self-contained closed-form or Gauss-loop expression), so reusing
    mass() itself, verbatim, is the only way to guarantee IDENTICAL
    interpolation between M and C without a second implementation that
    could silently drift out of sync with the first.

    coefficients: whatever shape/type the TARGET element's own mass()
    density argument expects, with D_i substituted for rho_i --
    a plain scalar for a single-field element (e.g. Beam2DCorotational's
    rho_A -> D), or a tuple/array/matrix for a multi-field element
    (e.g. a future Beam2DReissner's (rhoA, rhoA, rhoI) -> (D1, D2, D3)).
    May also be a dict keyed by block name for per-block coefficients,
    exactly like assemble_mass()'s rho_or_matrix -- see
    FESystem._per_block_arg()."""
    coefficients: object

    def assemble(self, fesystem, **kwargs):
        """Build and return the global field-damping matrix (same
        storage format -- dense ndarray or scipy.sparse.lil_matrix --
        as fesystem's own K/M) by looping over fesystem._blocks and
        calling each block's own formulation.mass(elem_coords,
        block_coefficients, **kwargs), exactly like FESystem.
        assemble_mass() does for the actual mass matrix. Called by
        FESystem.assemble_damping() -- not normally called directly."""
        C = fesystem._zeros_matrix()
        for name, formulation, connectivity in fesystem._blocks:
            block_coeff = fesystem._per_block_arg(self.coefficients, name)
            for elem_conn in connectivity:
                elem_coords = fesystem.mesh.nodes[elem_conn]
                ce = formulation.mass(elem_coords, block_coeff, **kwargs)
                g = fesystem._global_dofs(elem_conn)
                C[np.ix_(g, g)] += ce
        return C
