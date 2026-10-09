"""
state_space.py -- convert a second-order structural system (M, C, K),
plus an explicit INPUT map B and OUTPUT map Cout, into first-order
state-space form.

Every module in this package before this one (pod, galerkin, affine,
frequency) works with a full load vector F and a full-order response x --
there has never been a need for a compressed "a few inputs go in, a few
outputs come out" view of the model. krylov.py and balanced_truncation.py
(see docs/classical_mor_roadmap.md) both need exactly that view, in two
different first-order forms, so this module builds it once and both of
those modules import from here rather than reimplementing the same
linear algebra twice.

The physics: with state x = [q; q_dot] (the n_dof generalized
displacements stacked on the n_dof generalized velocities), the equation
of motion M*q'' + C*q' + K*q = F becomes, in first-order form,

    q'      = q_dot
    M*q_dot' = -K*q - C*q_dot + F

Writing F = B @ u for a small number of inputs u (B is n_dof x n_in,
selecting which physical DOF(s) each input load enters at) and
y = Cout @ q for a small number of outputs y (Cout is n_out x n_dof,
selecting which physical DOF(s) are read out -- e.g. a single tip
displacement, not the full state), this gives two algebraically
equivalent state-space representations of the SAME system:

  form="E" (generalized / descriptor form, eqs. 68-70 of Besselink et al.
  2013):
      E x' = A x + B_full u,   y = Cout_full x
      E = [[I, 0], [0, M]],  A = [[0, I], [-K, -C]]
      B_full = [[0], [B]],   Cout_full = [Cout, 0]
  No M-inversion anywhere -- preferred for krylov.py, where avoiding a
  dense M^-1 (and preserving the original K/M pencil) matters for
  numerical conditioning of the Krylov recursion.

  form="A" (explicit / ODE form, eqs. 65-67):
      x' = A x + B_full u,   y = Cout_full x
      A = [[0, I], [-M^-1 K, -M^-1 C]]
      B_full = [[0], [M^-1 B]]
  One dense M-solve, needed by balanced_truncation.py because scipy's
  Lyapunov solver (solve_continuous_lyapunov) only solves the ORDINARY
  (non-generalized) Lyapunov equation -- there's no separate numerical-
  conditioning reason to avoid the M-inversion for BT specifically, since
  BT already pays for an equivalent-cost dense Lyapunov solve regardless.

Both forms are (2*n_dof, 2*n_dof) -- this doubling (not n_dof) is exactly
why the paper reduces to k=20 states, not 10, when comparing its
first-order-form methods (MM, BT) against its purely second-order method
(MD) at "the same order".

If B/Cout aren't given, this defaults to every DOF being both an input
and an (position-only) output -- a reasonable default for exploring a
model before deciding on a specific port, though krylov.py/
balanced_truncation.py are only really useful with a genuinely small
number of inputs/outputs (that's the whole point of the "port" view).
"""
__author__ = "Abhijeet"
import numpy as np


def _as_array(x):
    """np.asarray(x), preserving complex dtype -- see pod._as_array for
    the full rationale (identical helper, duplicated rather than
    imported to keep this module's only dependency scipy/numpy, per the
    project's "no cross-module rom_engine imports in the core library"
    convention)."""
    x = np.asarray(x)
    return x if np.iscomplexobj(x) else x.astype(float, copy=False)


def _as_2d_map(x, n, axis):
    """Coerce a 1-D selection vector to a proper 2-D (n, 1) or (1, n)
    map, or pass a genuine 2-D map straight through. axis="in" means x
    selects rows (an input map, n_dof x n_in); axis="out" means x
    selects columns (an output map, n_out x n_dof)."""
    x = _as_array(x)
    if x.ndim == 1:
        return x.reshape(-1, 1) if axis == "in" else x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {x.shape}")
    return x


class StateSpaceSystem:
    """A first-order state-space realization of a second-order
    structural model. Plain container -- krylov.py/balanced_truncation.py
    read A/B/Cout (and E, when present) directly off this.

    Attributes
    ----------
    A : ndarray (2*n_dof, 2*n_dof)
    B : ndarray (2*n_dof, n_in)
    Cout : ndarray (n_out, 2*n_dof)
    E : ndarray (2*n_dof, 2*n_dof), or None
        None exactly when this was built with form="A" (ordinary,
        non-descriptor state-space -- E is implicitly the identity and
        callers should not assume otherwise).
    n_dof : int
        The size of the ORIGINAL second-order system -- half the state
        dimension. Kept around because both krylov.py and
        balanced_truncation.py need to map reduced states back to
        "how many original second-order modes does this correspond to"
        for reporting/plotting.
    """

    def __init__(self, A, B, Cout, E=None, n_dof=None):
        self.A = A
        self.B = B
        self.Cout = Cout
        self.E = E
        self.n_dof = n_dof

    @property
    def n_state(self):
        return self.A.shape[0]

    @property
    def n_in(self):
        return self.B.shape[1]

    @property
    def n_out(self):
        return self.Cout.shape[0]

    @property
    def form(self):
        return "A" if self.E is None else "E"


def to_state_space(M, K, C=None, B=None, Cout=None, form="E"):
    """Build a StateSpaceSystem from second-order (M, K[, C]) matrices
    and (optional) input/output maps.

    Parameters
    ----------
    M, K : ndarray (n_dof, n_dof)
        Mass and stiffness.
    C : ndarray (n_dof, n_dof), optional
        Damping. Defaults to zero (undamped) if omitted -- note an
        undamped model's state-space poles sit exactly ON the imaginary
        axis, which is a degenerate case for balanced_truncation.py (its
        Lyapunov equations assume A is Hurwitz-stable); pass a real
        (even lightly) damped C for that module.
    B : ndarray (n_dof, n_in) or (n_dof,), optional
        Input map: column j says which physical DOF(s) input u_j enters
        at (and with what weight). Defaults to the identity (every DOF
        is its own input) if omitted.
    Cout : ndarray (n_out, n_dof) or (n_dof,), optional
        Output map: row i says which physical DOF(s) output y_i reads
        (and with what weight). Position-only (velocities are never
        observed by this convention, matching the paper's own
        displacement-output examples). Defaults to the identity
        (every DOF's position is its own output) if omitted.
    form : {"E", "A"}
        See module docstring -- "E" (default) is the descriptor form
        krylov.py wants (no M-inversion); "A" is the explicit ODE form
        balanced_truncation.py wants (one dense M-solve, needed because
        scipy's Lyapunov solver has no generalized-E variant).

    Returns
    -------
    StateSpaceSystem
    """
    M = _as_array(M)
    K = _as_array(K)
    n = M.shape[0]
    if K.shape != (n, n):
        raise ValueError(f"M and K must be the same (n_dof, n_dof) shape, got {M.shape} vs {K.shape}")
    C = np.zeros((n, n)) if C is None else _as_array(C)

    B = np.eye(n) if B is None else _as_2d_map(B, n, axis="in")
    Cout = np.eye(n) if Cout is None else _as_2d_map(Cout, n, axis="out")
    if B.shape[0] != n:
        raise ValueError(f"B must have {n} rows (n_dof), got shape {B.shape}")
    if Cout.shape[1] != n:
        raise ValueError(f"Cout must have {n} columns (n_dof), got shape {Cout.shape}")

    I = np.eye(n)
    Z = np.zeros((n, n))
    A = np.block([[Z, I], [-K, -C]])
    Cout_full = np.hstack([Cout, np.zeros_like(Cout)])

    if form == "E":
        E = np.block([[I, Z], [Z, M]])
        B_full = np.vstack([np.zeros_like(B), B])
        return StateSpaceSystem(A=A, B=B_full, Cout=Cout_full, E=E, n_dof=n)
    elif form == "A":
        Minv_K = np.linalg.solve(M, K)
        Minv_C = np.linalg.solve(M, C)
        Minv_B = np.linalg.solve(M, B)
        A = np.block([[Z, I], [-Minv_K, -Minv_C]])
        B_full = np.vstack([np.zeros_like(B), Minv_B])
        return StateSpaceSystem(A=A, B=B_full, Cout=Cout_full, E=None, n_dof=n)
    else:
        raise ValueError(f"form must be 'E' or 'A', got {form!r}")
