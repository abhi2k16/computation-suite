# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
boundary.py -- springs, elastic foundations (Robin boundary terms) and general linear constraints.

Reached through ``FESystem.add_spring``, ``add_elastic_foundation``, ``add_constraint``, ``tie`` ::

    system.add_spring("tip", "uy", k=5.0e6)                       # grounded spring on every node of the set
    system.add_elastic_foundation("bottom", "uy", k=2.0e8)        # Winkler support: k u per unit area (length)
    system.add_elastic_foundation("wall", ["ux", "uy"], k=1e9, u_ref=0.001)   # Robin: k (u - u_ref) = traction
    system.add_constraint([(n1, "ux", 1.0), (n2, "ux", -1.0)], value=0.0)      # u1 - u2 = 0
    system.tie("slave_nodes", "master_nodes", ["ux", "uy"])                      # equal-DOF ties, node by node

Springs and foundations are added to the stiffness matrix ``K`` and are re-applied whenever
``assemble_stiffness`` replaces it, so the order of the calls does not matter. They are part of ``K``, hence
every linear solver (static, modal, transient, harmonic, buckling) sees them. The nonlinear drivers do not
include them yet and raise ``NotImplementedError``.

Constraints ``sum_i c_i u_i = value`` are eliminated exactly: for each independent constraint one DOF is chosen
as dependent (largest coefficient, full pivoting) and expressed through the others, then the system is
reduced to ``T^T K T``. This keeps the reduced matrix symmetric positive definite (no multipliers, no penalty
error) and handles chains of constraints and constraints on prescribed DOFs. ``solve_static`` and
``solve_modal`` support them; the other solvers raise ``NotImplementedError`` rather than ignoring them.
Constraint forces are not reported by ``reactions`` (it covers ``fix_dofs`` supports only).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from . import facet_loads


# ---------------------------------------------------------------------------------------------- facets
def facets_on(system, nodes):
    """Element facets (edges in 2-D, faces in 3-D) all of whose nodes are in ``nodes``.

    Each facet is a tuple of global node ids in the corner-then-mid-side order of ``facet_loads.list_facets``.
    Needs a plane-stress or solid element with a facet table; returns each facet once."""
    nodeset = set(int(n) for n in nodes)
    seen, out = set(), []
    for name, f, conn in system._blocks:
        try:
            table = facet_loads.list_facets(f)
        except ValueError as err:
            raise ValueError(f"cannot find boundary facets for {type(f).__name__}: {err}") from None
        for e in conn:
            for idx, _family, _ in table:
                tup = tuple(int(e[i]) for i in idx)
                if all(n in nodeset for n in tup):
                    key = tuple(sorted(tup))
                    if key not in seen:
                        seen.add(key)
                        out.append(tup)
    return out


def _is_facet_list(items):
    if isinstance(items, (str, np.ndarray)) and np.ndim(items) < 2:
        return False
    try:
        first = items[0]
    except (TypeError, IndexError, KeyError):
        return False
    return not isinstance(first, (str, int, np.integer)) and np.ndim(first) >= 1


# ---------------------------------------------------------------------------------------------- springs
def spring_blocks(system, nodes, dofs, k):
    """[(global dof array, 1x1 matrix)] for grounded springs; ``k`` scalar or one value per selected node."""
    nodes = np.asarray(nodes, dtype=int)
    kk = np.broadcast_to(np.asarray(k, dtype=float), nodes.shape)
    if np.any(kk < 0):
        raise ValueError("spring stiffness must be non-negative.")
    out = []
    for n, kv in zip(nodes, kk):
        for d in dofs:
            out.append((np.array([system.npn * int(n) + int(d)]), np.array([[float(kv)]])))
    return out


def foundation_blocks(system, facets, dofs, k, thickness=1.0, quad_order=2):
    """[(global dofs, matrix)] for the Robin term ``integral k u v`` on each facet, one block per DOF."""
    if not callable(k) and np.any(np.asarray(k, dtype=float) < 0):
        raise ValueError("foundation stiffness must be non-negative.")
    kfun = k if callable(k) else float(k)
    out = []
    dim = system.mesh.nodes.shape[1]
    for facet in facets:
        facet = tuple(int(n) for n in facet)
        coords = system.mesh.nodes[list(facet)]
        m = facet_loads.consistent_facet_mass(dim, coords, kfun, quad_order) * thickness
        for d in dofs:
            g = np.array([system.npn * n + int(d) for n in facet])
            out.append((g, m))
    return out


# ---------------------------------------------------------------------------------------------- constraints
def reduce_constraints(rows, values, fixed_values, tol=1e-10):
    """Eliminate linear constraints ``sum_j rows[i][j] u_j = values[i]``.

    Parameters
    ----------
    rows : list of dict {global dof: coefficient}
    values : list of float
    fixed_values : dict {global dof: prescribed value} -- terms on these DOFs move to the right-hand side.

    Returns
    -------
    slave : int ndarray -- the dependent DOFs
    mat : dict {slave dof: {master dof: coefficient}} -- ``u_slave = g_slave - sum coeff * u_master``
    g : dict {slave dof: float}
    """
    clean, rhs = [], []
    for r, v in zip(rows, values):
        row = {}
        v = float(v)
        for d, c in r.items():
            if d in fixed_values:
                v -= c * fixed_values[d]
            else:
                row[d] = row.get(d, 0.0) + c
        clean.append(row)
        rhs.append(v)
    cols = sorted({d for r in clean for d in r})
    m = len(clean)
    if not cols:
        for v in rhs:
            if abs(v) > tol * max(1.0, abs(v)):
                raise ValueError("inconsistent constraint: all its DOFs are prescribed and the equation is not satisfied.")
        return np.zeros(0, dtype=int), {}, {}
    A = np.zeros((m, len(cols) + 1))
    cidx = {d: j for j, d in enumerate(cols)}
    for i, r in enumerate(clean):
        for d, c in r.items():
            A[i, cidx[d]] = c
        A[i, -1] = rhs[i]
    scale = max(1.0, np.abs(A).max())
    pivots = []                                  # (row, col)
    used_rows = []
    work = A.copy()
    for _ in range(min(m, len(cols))):
        sub = np.abs(work[:, :-1]).copy()
        sub[used_rows, :] = 0.0
        for _r, c in pivots:
            sub[:, c] = 0.0
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        if sub[i, j] <= tol * scale:
            break
        work[i] /= work[i, j]
        for r in range(m):
            if r != i and work[r, j] != 0.0:
                work[r] -= work[r, j] * work[i]
        pivots.append((i, j))
        used_rows.append(i)
    for r in range(m):                           # rows left over are 0 = rhs: must be consistent
        if r not in used_rows and abs(work[r, -1]) > 1e-8 * scale:
            raise ValueError("inconsistent linear constraints (conflicting equations).")
    pivot_cols = {j for _, j in pivots}
    slave, mat, g = [], {}, {}
    for i, j in pivots:
        s = cols[j]
        slave.append(s)
        mat[s] = {cols[k]: work[i, k] for k in range(len(cols))
                  if k not in pivot_cols and abs(work[i, k]) > 1e-14}
        g[s] = float(work[i, -1])
    return np.array(slave, dtype=int), mat, g
