# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
beam2d_reissner_vectorized.py -- Wave 17 item 147 (docs/consolidated_
future_roadmap.md): a batched (all-elements-at-once NumPy) internal_
force()/tangent_stiffness() path for elements.beams.Beam2DReissner,
built ONLY because item 147's own measurement step found a single
representative forced-vibration run (80 elements, dt=T_forcing/100, 80
forcing periods, through nonlinear_solver.solve_nonlinear_transient())
takes ~4.5 minutes wall clock -- over the item's own "a few minutes"
threshold for needing either this or a switch to explicit dynamics.
Explicit dynamics was measured and rejected (its critical timestep,
h/sqrt(E/rho) ~ 1e-6 s here, needs ~18x more steps than the implicit
dt used above, and even with no per-step tangent assembly/Newton loop,
the internal_force()-bound explicit cost estimate came out to ~14
minutes -- WORSE, not better, confirming the roadmap's own warning that
this is "a genuinely different cost tradeoff, not a free win"). See
this item's own row in the roadmap for the full numbers.

Convention: follows vectorized_assembly.py's own "batch the per-element
linear algebra into one array-level contraction instead of a Python
`for element in elements` loop" idea -- the SAME idea, but a from-scratch
implementation, because vectorized_assembly.py's own machinery
(MeshTransformation's shape_grad/JxW precompute, build_B_batched()) is
explicitly scoped (see that module's own docstring) to isoparametric
continuum elements with dofs_per_node in {2, 3} and a CONSTANT small
material matrix contracted via multi-Gauss-point quadrature --
Beam2DReissner is a 2-node, dofs_per_node=3, ONE-POINT-quadrature
(already collapsed to a single midpoint evaluation, see that class's
own docstring) NONLINEAR element whose strain measures are closed-form
trig expressions of the nodal state, not a B-matrix contraction against
a constant D over several Gauss points -- a genuinely different batched
computation, so it is its own small module rather than a forced fit
into vectorized_assembly.py's scope. It DOES reuse
vectorized_assembly.scatter_global_stiffness() verbatim for scattering
the batched (n_elements, 6, 6) tangent array into a global matrix (that
function is already fully generic in dofs_per_node/n_edof, no
continuum-only assumption baked into IT specifically), and adds the
one missing piece that module doesn't have: a matching vectorized
scatter-add for a FORCE vector (needed for internal_force(), which
vectorized_assembly.py -- a stiffness-only module -- never had to
provide).

Every array operation below is the exact same closed-form algebra
elements.beams.Beam2DReissner._local_dofs()/_B_and_H()/internal_force()/
tangent_stiffness() already use, just with a leading `n_elements` axis
threaded through instead of a Python loop -- see that class's own
docstring for the strain-measure/tangent DERIVATION (not repeated here);
this module only re-shapes the same formulas.

Validation (tests/test_beam2d_reissner_vectorized.py): the batched
internal_force_batched()/tangent_stiffness_batched() outputs match the
existing per-element looped Beam2DReissner.internal_force()/
tangent_stiffness() calls to < 1e-12 relative error, on the same random
large-rotation states test_beam2d_reissner.py's own CHECK (a) uses --
this item's own stated validation criterion.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from .vectorized_assembly import scatter_global_stiffness


def _local_dofs_batched(elem_coords_all, u_elem_all):
    """Batched elements.beams.Beam2DReissner._local_dofs(): elem_coords_all
    (n,2,2), u_elem_all (n,6) -> L0 (n,), T (n,6,6), q (n,6)."""
    X1 = elem_coords_all[:, 0, :]
    X2 = elem_coords_all[:, 1, :]
    dX = X2 - X1
    L0 = np.linalg.norm(dX, axis=1)
    c0 = dX[:, 0] / L0
    s0 = dX[:, 1] / L0

    n = elem_coords_all.shape[0]
    T = np.zeros((n, 6, 6))
    T[:, 0, 0] = c0; T[:, 0, 1] = -s0
    T[:, 1, 0] = s0; T[:, 1, 1] = c0
    T[:, 2, 2] = 1.0
    T[:, 3, 3] = c0; T[:, 3, 4] = -s0
    T[:, 4, 3] = s0; T[:, 4, 4] = c0
    T[:, 5, 5] = 1.0

    # q = T.T @ u_elem, per element
    q = np.einsum('nji,nj->ni', T, u_elem_all)
    return L0, T, q


def _B_and_H_batched(L0, q):
    """Batched elements.beams.Beam2DReissner._B_and_H(): L0 (n,), q (n,6)
    -> B (n,3,6), H_eps (n,6,6), H_gam (n,6,6), eps (n,), gam (n,), kap (n,)."""
    n = L0.shape[0]
    u1_1, u2_1, th1 = q[:, 0], q[:, 1], q[:, 2]
    u1_2, u2_2, th2 = q[:, 3], q[:, 4], q[:, 5]
    u1p = (u1_2 - u1_1) / L0
    u2p = (u2_2 - u2_1) / L0
    thp = (th2 - th1) / L0
    th_mid = 0.5 * (th1 + th2)
    cphi, sphi = np.cos(th_mid), np.sin(th_mid)
    eps = (1.0 + u1p) * cphi + u2p * sphi - 1.0
    gam = u2p * cphi - (1.0 + u1p) * sphi
    kap = thp

    ga = np.zeros((n, 6)); ga[:, 0] = -1.0 / L0; ga[:, 3] = 1.0 / L0
    gb = np.zeros((n, 6)); gb[:, 1] = -1.0 / L0; gb[:, 4] = 1.0 / L0
    gc = np.zeros((n, 6)); gc[:, 2] = 0.5; gc[:, 5] = 0.5
    gd = np.zeros((n, 6)); gd[:, 2] = -1.0 / L0; gd[:, 5] = 1.0 / L0

    grad_eps = cphi[:, None] * ga + sphi[:, None] * gb + gam[:, None] * gc
    grad_gam = cphi[:, None] * gb - sphi[:, None] * ga - (eps + 1.0)[:, None] * gc
    grad_kap = gd
    B = np.stack([grad_eps, grad_gam, grad_kap], axis=1)  # (n, 3, 6)

    gc_ga = np.einsum('ni,nj->nij', gc, ga); ga_gc = gc_ga.transpose(0, 2, 1)
    gc_gb = np.einsum('ni,nj->nij', gc, gb); gb_gc = gc_gb.transpose(0, 2, 1)
    gc_gc = np.einsum('ni,nj->nij', gc, gc)

    H_eps = (-sphi[:, None, None] * (gc_ga + ga_gc)
             + cphi[:, None, None] * (gc_gb + gb_gc)
             - (eps + 1.0)[:, None, None] * gc_gc)
    H_gam = (-sphi[:, None, None] * (gc_gb + gb_gc)
             - cphi[:, None, None] * (gc_ga + ga_gc)
             - gam[:, None, None] * gc_gc)

    return B, H_eps, H_gam, eps, gam, kap


def internal_force_batched(elem_coords_all, u_elem_all, mat):
    """Batched Beam2DReissner.internal_force(): elem_coords_all (n,2,2),
    u_elem_all (n,6), mat=(E,G,A,I,kappa_s) CONSTANT across all n
    elements (homogeneous material per block, same scope restriction
    vectorized_assembly.py's own tensorized_element_stiffness() already
    imposes) -> f_global (n,6)."""
    E, G, A, I, kappa_s = mat
    L0, T, q = _local_dofs_batched(elem_coords_all, u_elem_all)
    B, _, _, eps, gam, kap = _B_and_H_batched(L0, q)
    N = E * A * eps
    Q = kappa_s * G * A * gam
    M = E * I * kap
    resultants = np.stack([N, Q, M], axis=1)          # (n, 3)
    f_local = L0[:, None] * np.einsum('nsm,ns->nm', B, resultants)  # (n, 6)
    f_global = np.einsum('nij,nj->ni', T, f_local)
    return f_global


def tangent_stiffness_batched(elem_coords_all, u_elem_all, mat):
    """Batched Beam2DReissner.tangent_stiffness() -> K_global (n,6,6)."""
    E, G, A, I, kappa_s = mat
    L0, T, q = _local_dofs_batched(elem_coords_all, u_elem_all)
    B, H_eps, H_gam, eps, gam, kap = _B_and_H_batched(L0, q)
    N = E * A * eps
    Q = kappa_s * G * A * gam
    D = np.diag([E * A, kappa_s * G * A, E * I])
    BtDB = np.einsum('nsm,st,ntk->nmk', B, D, B)
    K_local = L0[:, None, None] * (BtDB + N[:, None, None] * H_eps + Q[:, None, None] * H_gam)
    K_global = np.einsum('nij,njk,nlk->nil', T, K_local, T)
    return K_global


def scatter_global_force(fe, connectivity, dofs_per_node, n_dof):
    """Vectorized (no Python loop over elements) scatter-add of a
    stacked local-force array (n_elements, n_edof) into a global force
    vector -- the force-vector counterpart of vectorized_assembly.
    scatter_global_stiffness() (which only handles matrices); built
    here because internal_force() assembly needs it and vectorized_
    assembly.py, a stiffness-only module, never had to provide one.
    Uses np.add.at for the duplicate-index accumulation (the vector
    analogue of that function's scipy.sparse.coo_matrix duplicate-sum
    trick)."""
    connectivity = np.asarray(connectivity)
    n_elements, n_edof = fe.shape
    g = (dofs_per_node * connectivity[:, :, None] +
         np.arange(dofs_per_node)[None, None, :]).reshape(n_elements, n_edof)
    F = np.zeros(n_dof)
    np.add.at(F, g.reshape(-1), fe.reshape(-1))
    return F


def _pick_block(fesystem, block_name):
    for name, formulation, connectivity in fesystem._blocks:
        if block_name is None or name == block_name:
            return name, formulation, connectivity
    raise ValueError(f"no block named {block_name!r} in this fesystem")


def assemble_internal_force_vectorized(fesystem, u_global, mat, block_name=None):
    """Additive, opt-in alternative to FESystem.assemble_internal_force()
    for a Beam2DReissner block -- returns a fresh global force vector
    (does not mutate fesystem, matching assemble_internal_force()'s own
    return-only convention). Only supports a single homogeneous mat
    across the block (see module docstring's "CONSTANT material per
    block" scope, same restriction vectorized_assembly.py's own
    tensorized_element_stiffness() already imposes); block_name picks
    which fesystem._blocks entry to use (defaults to the first, the
    common single-block-mesh case this item's own benchmark uses).
    Takes u_global explicitly since a Newton loop calls this fresh every
    iteration with the current trial displacement."""
    name, formulation, connectivity = _pick_block(fesystem, block_name)
    elem_coords_all = fesystem.mesh.nodes[connectivity]           # (n,2,2)
    npn = 3
    g = (npn * connectivity[:, :, None] + np.arange(npn)[None, None, :]).reshape(len(connectivity), -1)
    u_elem_all = u_global[g]
    fe = internal_force_batched(elem_coords_all, u_elem_all, mat)
    return scatter_global_force(fe, connectivity, dofs_per_node=npn, n_dof=fesystem.n_dof)


def assemble_tangent_stiffness_vectorized(fesystem, u_global, mat, block_name=None, sparse=False):
    """Additive, opt-in alternative to FESystem.assemble_tangent_
    stiffness() for a Beam2DReissner block -- same scope/return/u_global
    convention as assemble_internal_force_vectorized() above."""
    name, formulation, connectivity = _pick_block(fesystem, block_name)
    elem_coords_all = fesystem.mesh.nodes[connectivity]
    npn = 3
    g = (npn * connectivity[:, :, None] + np.arange(npn)[None, None, :]).reshape(len(connectivity), -1)
    u_elem_all = u_global[g]
    ke = tangent_stiffness_batched(elem_coords_all, u_elem_all, mat)
    return scatter_global_stiffness(ke, connectivity, dofs_per_node=npn,
                                     n_dof=fesystem.n_dof, sparse=sparse)
