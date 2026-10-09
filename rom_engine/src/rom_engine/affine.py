"""
affine.py -- affine parametric decomposition, the "offline-online"
trick that makes reduced-basis methods fast for many-query problems
(parameter sweeps, optimization loops, Monte Carlo / uncertainty
quantification -- anything that needs the SAME structure re-solved at
many different parameter values).

The idea: many structural stiffness matrices depend on their material/
geometric parameters mu AFFINELY -- i.e. they can be written as a sum

    K(mu) = theta_1(mu)*K_1 + theta_2(mu)*K_2 + ... + theta_Q(mu)*K_Q

where the K_q are FIXED, PARAMETER-INDEPENDENT matrices (assembled
ONCE, offline) and only the scalar coefficients theta_q(mu) change with
the query parameter mu. The simplest example: a homogeneous linear-
elastic structure with Young's modulus E as the only parameter has
K(E) = E * K_unit (Q=1, theta_1(E)=E, K_unit = K assembled with E=1) --
stiffness is always linear in E for linear elasticity, so this is
always available essentially for free. A richer, still-common case:
a structure with two (or more) independently-parametrized material
regions has K(E_1, E_2) = E_1*K_1 + E_2*K_2, where K_1/K_2 are the
stiffness contributions from EACH region assembled with unit modulus
(zero everywhere else) -- still Q=2 terms, still exact, no
approximation involved (this is NOT a surrogate/interpolation method;
affine decomposition is an EXACT algebraic re-expression of K(mu)+, valid
whenever the underlying physics really is affine in the parameters,
which ordinary linear elasticity always is with respect to modulus
scaling).

Why this matters for a ROM: combined with galerkin.py's projection,
each K_q only needs to be projected onto the reduced basis ONCE
(K_q,r = V^T K_q V, an O(n_dof^2 * n_modes) cost paid offline, a single
time). After that, assembling and solving the REDUCED system at a NEW
parameter value mu costs only O(Q * n_modes^2) -- assembling Q small
matrices and adding them -- completely independent of the full model's
size n_dof. A parameter sweep of a million (E_1, E_2) pairs over a
100,000-DOF FE model becomes a million tiny (n_modes x n_modes) linear
solves, not a million full re-assemblies + full solves.
"""
__author__ = "Abhijeet"
import numpy as np


def _as_array(x):
    """np.asarray(x), preserving complex dtype -- see pod._as_array for
    the full rationale. This one matters MOST in this particular
    module: theta_func(mu) for a frequency-domain problem returns
    coefficients like i*omega, which is PURELY imaginary. The old
    ``np.asarray(theta_func(mu), dtype=float)`` would silently discard
    the entire imaginary part -- not raise, not warn, just quietly
    return 0 for that term -- which for a damping coefficient means
    "silently solve the UNDAMPED problem" no matter what C is. That is
    exactly the kind of loud-should-be-quiet failure this project's
    testing culture exists to catch (see
    docs/frequency_domain_rom_roadmap.md Section 4); test_affine.py
    includes a regression test asserting a purely-imaginary theta_q
    actually affects assemble_reduced()'s output."""
    x = np.asarray(x)
    return x if np.iscomplexobj(x) else x.astype(float, copy=False)


class AffineDecomposition:
    """A parameter-independent affine decomposition of a system matrix
    (or several -- e.g. stiffness AND mass, if both are affine in the
    parameters), with an offline projection step for fast reduced
    online queries.

    Parameters
    ----------
    components : sequence of ndarray, each (n_dof, n_dof)
        The Q parameter-independent component matrices K_1, ..., K_Q.
    theta_func : callable
        mu -> array-like of length Q, the parameter-dependent scalar
        coefficients theta_1(mu), ..., theta_Q(mu). Called once per
        assemble()/assemble_reduced() query; kept as a caller-supplied
        function rather than a fixed form so ANY affine parametrization
        (linear in E, linear in a thickness, a sum of several such
        terms, ...) can be expressed without this class needing to
        know the physics.
    """

    def __init__(self, components, theta_func):
        self.components = [_as_array(Kq) for Kq in components]
        if len(self.components) == 0:
            raise ValueError("need at least one component matrix")
        shape0 = self.components[0].shape
        for Kq in self.components:
            if Kq.shape != shape0:
                raise ValueError("all component matrices must have the same shape")
        self.theta_func = theta_func
        self.Q = len(self.components)
        self.n_dof = shape0[0]
        self.reduced_components = None   # set by project()
        self.component_action = None     # set by project() -- see assemble_action()
        self.n_modes = None

    def _theta(self, mu):
        theta = _as_array(self.theta_func(mu)).ravel()
        if len(theta) != self.Q:
            raise ValueError(f"theta_func(mu) returned {len(theta)} coefficients, expected {self.Q}")
        return theta

    # -----------------------------------------------------------------
    # Full-order (offline-cost-equivalent) assembly -- provided mainly
    # for VALIDATION: this should exactly match a direct from-scratch
    # reassembly of the full-order model at the same mu, to machine
    # precision, since it's algebraically the same matrix, just built
    # as a weighted sum instead of an element loop.
    # -----------------------------------------------------------------
    def assemble(self, mu):
        """theta_1(mu)*K_1 + ... + theta_Q(mu)*K_Q, full (n_dof, n_dof)."""
        theta = self._theta(mu)
        K = theta[0] * self.components[0]
        for q in range(1, self.Q):
            K = K + theta[q] * self.components[q]
        return K

    # -----------------------------------------------------------------
    # Offline projection step
    # -----------------------------------------------------------------
    def project(self, basis):
        """Project every component matrix onto a reduced basis ONCE --
        the offline step that makes assemble_reduced() cheap.

        Parameters
        ----------
        basis : ndarray (n_dof, n_modes), or an object with a .V
            attribute (a fitted pod.PodBasis) -- same duck-typing
            convention as galerkin.GalerkinROM.

        Returns self, so this can be chained:
        ``affine = AffineDecomposition(comps, theta).project(basis)``.
        """
        V = basis.V if hasattr(basis, "V") else _as_array(basis)
        if V.shape[0] != self.n_dof:
            raise ValueError(
                f"basis has {V.shape[0]} rows, components have {self.n_dof} -- "
                "basis and components must be on the same (full) dof numbering")
        self.reduced_components = [V.T @ Kq @ V for Kq in self.components]
        # One projection short of reduced_components above -- Kq @ V,
        # (n_dof, n_modes), not yet hit with V.T on the left -- cached
        # at the SAME offline cost as reduced_components (same V, same
        # matrix products, just kept before the final contraction).
        # This is what makes assemble_action() below possible without
        # ever forming the full (n_dof, n_dof) A(mu): see
        # docs/phase4_error_bounds_greedy_roadmap.md Section 2.
        self.component_action = [Kq @ V for Kq in self.components]
        self.n_modes = V.shape[1]
        return self

    def assemble_action(self, mu, q):
        """A(mu) @ V @ q, i.e. the full-order EFFECT of applying the
        full operator to an already-reduced state q -- WITHOUT ever
        forming the full (n_dof, n_dof) A(mu) matrix explicitly.

        Computed as theta_1(mu)*(K_1 @ V)@q + ... , using the
        component_action cached by project() -- O(Q * n_dof * n_modes)
        (dominated by the Q mat-vecs), instead of assemble(mu)'s
        O(n_dof^2) (building and holding the full dense matrix). Since
        n_modes << n_dof for any basis worth reducing to, this is a
        real, substantial saving, not a micro-optimization -- and it
        computes the algebraically EXACT same full-order vector
        assemble(mu) @ (V @ q) would, just without the expensive
        intermediate. This is the building block for a cheap, EXACT
        residual norm (frequency.FrequencyROM.residual_norm()),
        callable many times inside a greedy training loop's inner
        sweep over candidate parameters -- see
        docs/phase4_error_bounds_greedy_roadmap.md Section 2.

        Requires project(basis) to have been called first. q must be
        in the SAME reduced coordinates that basis (V) defines --
        typically the solution of assemble_reduced(mu) @ q = F_r.
        """
        if self.component_action is None:
            raise RuntimeError("call project(basis) before assemble_action()")
        theta = self._theta(mu)
        q = np.asarray(q)
        result = theta[0] * (self.component_action[0] @ q)
        for i in range(1, self.Q):
            result = result + theta[i] * (self.component_action[i] @ q)
        return result

    def assemble_reduced(self, mu):
        """theta_1(mu)*K_1,r + ... + theta_Q(mu)*K_Q,r, the tiny
        (n_modes, n_modes) reduced matrix -- the fast ONLINE query.
        Requires project(basis) to have been called first."""
        if self.reduced_components is None:
            raise RuntimeError("call project(basis) before assemble_reduced()")
        theta = self._theta(mu)
        Kr = theta[0] * self.reduced_components[0]
        for q in range(1, self.Q):
            Kr = Kr + theta[q] * self.reduced_components[q]
        return Kr

    def solve_reduced(self, mu, F_r):
        """Convenience: assemble_reduced(mu) and directly solve
        K_r(mu) q = F_r for q, the single most common online query
        (a parametric static solve). F_r must already be in reduced
        coordinates (e.g. from GalerkinROM.project_vector()) -- this
        class deliberately doesn't own a basis object to project F
        with, keeping it decoupled from galerkin.py; a thin wrapper
        combining the two is a natural place for a caller-specific
        convenience function, not something this generic class should
        hardcode."""
        Kr = self.assemble_reduced(mu)
        return np.linalg.solve(Kr, F_r)
