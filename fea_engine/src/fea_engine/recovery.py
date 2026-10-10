# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
recovery.py -- derived results as FEFields: strain, stress, von Mises and support reactions.

Reached through ``FESystem.stress(U)``, ``.strain(U)``, ``.von_mises(U)`` and ``.reactions(U)``::

    U = system.solve_static()
    S = system.stress(U)                      # FEField, components sxx, syy, sxy (2-D) or sxx..sxz (3-D)
    S.component("sxx", nodes="root")          # same named access as displacements
    S.plot("syy")                             # contour, colour bar "syy [Pa]"
    vm = system.von_mises(U)                  # scalar FEField "von_mises"
    R = system.reactions(U)                   # FEField of support forces (zero at free DOFs)
    R.component("uy", nodes="root").sum()     # total vertical reaction at the root

How nodal stress is obtained
----------------------------
Stress is computed at the integration points (``sigma = D(x) B u``) and moved to the corner nodes with a
lumped L2 projection using the linear shape functions: ``sigma_a = sum(N_a sigma |J| w) / sum(N_a |J| w)``
over the elements sharing node ``a``. It is positive-weighted, so it cannot overshoot the integration-point
values, and for constant-strain triangles/tetrahedra it reduces to the usual volume-weighted average.
Mid-side nodes of quadratic elements are interpolated from their corners (a quadratic shape function
integrates to zero or less at a corner, so projecting with it directly would be ill-posed). Values on a
boundary are smoothed estimates, not exact boundary stresses. ``at="elements"`` returns the volume-averaged element
values instead (an ndarray, one row per element in block order, convenient for ``cell_data`` export).

Supported elements: Quad4/Quad8/Tri3/Tri6 plane stress and Hex8/Hex20/Tet4/Tet10 solids (full integration).
D comes from the last ``assemble_stiffness`` call (coefficients included); pass ``D=`` to override.
Voigt order: 2-D ``[xx, yy, xy]``; 3-D ``[xx, yy, zz, xy, yz, xz]``; strains use engineering shear.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np

from . import coefficients as cf
from .fields import FEField

_NAMES = {
    ("stress", 3): ("sxx", "syy", "sxy"),
    ("stress", 6): ("sxx", "syy", "szz", "sxy", "syz", "sxz"),
    ("strain", 3): ("exx", "eyy", "gxy"),
    ("strain", 6): ("exx", "eyy", "ezz", "gxy", "gyz", "gxz"),
}


def _resolve_D(system, D):
    if D is not None:
        return D
    args = getattr(system, "_stiffness_args", None)
    if args is None:
        raise RuntimeError("stress/strain recovery needs the constitutive matrix: call assemble_stiffness() "
                           "first (it is remembered) or pass D=.")
    return args[0]


def _check_vector(system, U):
    U = np.asarray(U)
    if U.ndim != 1 or U.shape[0] != system.n_dof:
        raise ValueError(f"expected a displacement vector of length {system.n_dof}, got shape {U.shape}.")
    if np.iscomplexobj(U):
        raise NotImplementedError("recovery of complex (harmonic) fields: pass U.real or U.imag.")
    return U


_LAYOUT_CACHE = {}


def _layout(f):
    """(linear element, n_corner, T): the linear companion element, how many leading nodes are corners, and
    T (n_nodes - n_corner, n_corner) interpolating corner values to the remaining (mid-side) nodes."""
    key = type(f)
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    from .elements.solids import Quad4PlaneStress, Hex8Solid3D, Tri3PlaneStress, Tet4Solid3D
    simplex = f.quadrature_family == "simplex"
    lin = {(2, False): Quad4PlaneStress, (3, False): Hex8Solid3D,
           (2, True): Tri3PlaneStress, (3, True): Tet4Solid3D}[(f.dim, simplex)]()
    nc = lin.n_nodes
    T = None
    if f.n_nodes != nc:
        import itertools
        grid = [0.0, 0.5, 1.0] if simplex else [-1.0, 0.0, 1.0]
        pos = []
        for a in range(f.n_nodes):
            for p in itertools.product(grid, repeat=f.dim):
                N = f.shape_and_derivs(p)[0]
                if abs(N[a] - 1.0) < 1e-12 and abs(np.abs(N).sum() - 1.0) < 1e-12:
                    pos.append(p)
                    break
            else:
                raise NotImplementedError(f"recovery: cannot locate node {a} of {type(f).__name__}.")
        for a in range(nc):          # the first nc nodes must be the corners, in the linear element's order
            if not np.allclose(lin.shape_and_derivs(pos[a])[0], np.eye(nc)[a]):
                raise NotImplementedError(f"recovery: {type(f).__name__} corner nodes are not its first {nc}.")
        T = np.array([lin.shape_and_derivs(pos[a])[0] for a in range(nc, f.n_nodes)])
    _LAYOUT_CACHE[key] = (lin, nc, T)
    return _LAYOUT_CACHE[key]


class _NumpyOps:
    """Array shims so the batched recovery is written once for NumPy and torch."""
    name = "numpy"

    def arr(self, x):
        return np.asarray(x, dtype=float)

    def zeros(self, *shape):
        return np.zeros(shape)

    einsum = staticmethod(np.einsum)
    solve = staticmethod(np.linalg.solve)
    det = staticmethod(np.linalg.det)
    absv = staticmethod(np.abs)

    def add_at(self, target, idx, src):
        np.add.at(target, idx, src)

    def out(self, x):
        return x


class _TorchOps:
    name = "torch"

    def __init__(self, device):
        import torch
        self.t = torch
        self.device = device

    def arr(self, x):
        t = self.t
        if isinstance(x, t.Tensor):
            return x.to(device=self.device, dtype=t.float64)
        return t.as_tensor(np.array(x, dtype=float), dtype=t.float64, device=self.device)   # copy: may be read-only

    def zeros(self, *shape):
        return self.t.zeros(shape, dtype=self.t.float64, device=self.device)

    def einsum(self, spec, *ops):
        return self.t.einsum(spec, *ops)

    def solve(self, A, b):
        return self.t.linalg.solve(A, b)

    def det(self, A):
        return self.t.linalg.det(A)

    def absv(self, x):
        return self.t.abs(x)

    def add_at(self, target, idx, src):
        idx = self.t.as_tensor(np.asarray(idx), dtype=self.t.int64, device=self.device)
        target.index_add_(0, idx.reshape(-1), src.reshape((-1,) + tuple(src.shape[idx.ndim:])))

    def out(self, x):
        return x.detach().cpu().numpy()


def _ops(backend, device):
    if backend == "numpy":
        return _NumpyOps()
    if backend == "torch":
        from . import torch_sparse_solver as tss
        tss._require_torch()
        return _TorchOps(device)
    raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}.")


def _B_batched(xp, dN_g, dim):
    """(E, dim, n) physical shape-function gradients -> B (E, nstrain, n*dim); same row layout as the
    elements' own ``B_matrix`` (engineering shear; 3-D order xx, yy, zz, xy, yz, xz)."""
    E, _, n = dN_g.shape
    ns = 3 if dim == 2 else 6
    B = xp.zeros(E, ns, n * dim)
    for k in range(n):
        c = dim * k
        if dim == 2:
            B[:, 0, c] = dN_g[:, 0, k]
            B[:, 1, c + 1] = dN_g[:, 1, k]
            B[:, 2, c] = dN_g[:, 1, k]
            B[:, 2, c + 1] = dN_g[:, 0, k]
        else:
            dx, dy, dz = dN_g[:, 0, k], dN_g[:, 1, k], dN_g[:, 2, k]
            B[:, 0, c] = dx
            B[:, 1, c + 1] = dy
            B[:, 2, c + 2] = dz
            B[:, 3, c] = dy
            B[:, 3, c + 1] = dx
            B[:, 4, c + 1] = dz
            B[:, 4, c + 2] = dy
            B[:, 5, c] = dz
            B[:, 5, c + 2] = dx
    return B


def _block_batched(xp, system, f, conn, U, rule, lin, nc, Dmat):
    """Whole block at once. -> (numE (E, nc, ncomp), denE (E, nc), elem (E, ncomp))."""
    dim = f.dim
    npn = system.npn
    conn = np.asarray(conn)
    X = xp.arr(system.mesh.nodes[conn])                                   # (E, n, dim)
    dofs = (conn[:, :, None] * npn + np.arange(npn)[None, None, :]).reshape(len(conn), -1)
    ue = U[xp.t.as_tensor(dofs, device=xp.device)] if xp.name == "torch" else U[dofs]   # (E, n*npn)
    ncomp = 3 if dim == 2 else 6
    E = len(conn)
    numE = xp.zeros(E, nc, ncomp)
    denE = xp.zeros(E, nc)
    acc = xp.zeros(E, ncomp)
    vol = xp.zeros(E)
    Dm = xp.arr(Dmat) if Dmat is not None else None
    for p, w in rule:
        dN = xp.arr(f._cached_shape_and_derivs(p)[1])                      # (dim, n)
        Nl = xp.arr(lin.shape_and_derivs(p)[0])                            # (nc,)
        J = xp.einsum("dn,enk->edk", dN, X)
        detJ = xp.det(J)
        dN_g = xp.solve(J, dN.expand(J.shape[0], -1, -1) if xp.name == "torch"
                        else np.broadcast_to(dN, (E,) + dN.shape))
        B = _B_batched(xp, dN_g, dim)
        val = xp.einsum("esm,em->es", B, ue)
        if Dm is not None:
            val = xp.einsum("st,et->es", Dm, val)
        wt = xp.absv(detJ) * w
        numE = numE + xp.einsum("e,c,es->ecs", wt, Nl, val)
        denE = denE + xp.einsum("e,c->ec", wt, Nl)
        acc = acc + wt[:, None] * val
        vol = vol + wt
    return numE, denE, acc / vol[:, None]


def _block_loop(system, f, conn, U, rule, lin, nc, block_D, kind):
    """Reference per-element loop; the only path that supports coefficient (position-dependent) D."""
    ncomp = 3 if f.dim == 2 else 6
    E = len(conn)
    numE = np.zeros((E, nc, ncomp))
    denE = np.zeros((E, nc))
    elem = np.zeros((E, ncomp))
    for i, e in enumerate(conn):
        coords = system.mesh.nodes[e]
        ue = U[system._global_dofs(e)]
        acc, vol = np.zeros(ncomp), 0.0
        for p, w in rule:
            B, detJ = f.B_matrix(p, coords)
            N = f._cached_shape_and_derivs(p)[0]
            Nl = lin.shape_and_derivs(p)[0]
            val = B @ ue
            if kind == "stress":
                Dg = cf._val(block_D, i, coords, N @ coords)
                val = np.asarray(Dg) @ val
            wt = abs(detJ) * w
            numE[i] += wt * Nl[:, None] * val[None, :]
            denE[i] += wt * Nl
            acc += wt * val
            vol += wt
        elem[i] = acc / vol
    return numE, denE, elem


def _recover(system, U, D, kind, vectorized=True, backend="numpy", device="cpu", raw=False):
    """-> (nodal (n_nodes, ncomp), element-average (n_elements, ncomp)).

    Integration-point values are projected to the corner nodes with the (positive) linear shape functions;
    mid-side nodes of quadratic elements are interpolated from their corners. With ``vectorized=True`` (default)
    each block is evaluated with batched array operations; blocks with position-dependent coefficients fall
    back to the per-element loop. ``backend="torch"`` requires the batched path. ``raw=True`` returns the
    backend's own arrays (torch tensors stay attached to the autograd graph)."""
    xp = _ops(backend, device)
    if xp.name == "torch" and not vectorized:
        raise ValueError("backend='torch' requires vectorized=True.")
    if xp.name == "numpy":
        U = _check_vector(system, U)
    else:
        if tuple(U.shape) != (system.n_dof,):
            raise ValueError(f"expected a displacement vector of length {system.n_dof}, got shape {tuple(U.shape)}.")
        U = xp.arr(U)
    D = _resolve_D(system, D) if kind == "stress" else None
    n_nodes = len(system.mesh.nodes)
    ncomp = None
    num = den = None
    elem_parts = []
    mid_info = []                                     # (conn, T, nc) per block with mid-side nodes
    for name, f, conn in system._blocks:
        cf.check_supported(f, f"{kind} recovery")
        nb = 3 if f.dim == 2 else 6
        if ncomp is None:
            ncomp = nb
            num = xp.zeros(n_nodes, ncomp)
            den = xp.zeros(n_nodes)
        elif nb != ncomp:
            raise NotImplementedError("recovery needs all blocks to be 2-D or all 3-D.")
        block_D = system._per_block_arg(D, name) if D is not None else None
        rule = cf.quadrature(f, "recovery")
        lin, nc, T = _layout(f)
        constant_D = block_D is None or not cf.has_coefficient(block_D)
        conn_arr = np.asarray(conn)
        if vectorized and constant_D:
            numE, denE, elem = _block_batched(xp, system, f, conn_arr, U, rule, lin, nc, block_D)
        else:
            if xp.name == "torch":
                raise NotImplementedError("backend='torch' does not support position-dependent coefficients; "
                                          "use backend='numpy'.")
            numE, denE, elem = _block_loop(system, f, conn_arr, U, rule, lin, nc, block_D, kind)
        xp.add_at(num, conn_arr[:, :nc], numE)
        xp.add_at(den, conn_arr[:, :nc], denE)
        elem_parts.append(elem)
        if T is not None:
            mid_info.append((conn_arr, T, nc))
    if xp.name == "numpy":
        nodal = np.divide(num, den[:, None], out=np.zeros_like(num), where=den[:, None] > 0)
    else:
        t = xp.t
        safe = t.where(den > 0, den, t.ones_like(den))
        nodal = t.where((den > 0)[:, None], num / safe[:, None], t.zeros_like(num))
    if mid_info:
        acc = xp.zeros(n_nodes, ncomp)
        cnt = xp.zeros(n_nodes)
        for conn_arr, T, nc in mid_info:
            Tm = xp.arr(T)
            idx_c = conn_arr[:, :nc]
            vals = nodal[xp.t.as_tensor(idx_c, device=xp.device)] if xp.name == "torch" else nodal[idx_c]
            midv = xp.einsum("mc,ecj->emj", Tm, vals)
            xp.add_at(acc, conn_arr[:, nc:], midv)
            xp.add_at(cnt, conn_arr[:, nc:], xp.zeros(*conn_arr[:, nc:].shape) + 1.0)
        if xp.name == "numpy":
            only_mid = (den == 0) & (cnt > 0)
            nodal[only_mid] = acc[only_mid] / cnt[only_mid, None]
        else:
            t = xp.t
            only_mid = (den == 0) & (cnt > 0)
            safe = t.where(cnt > 0, cnt, t.ones_like(cnt))
            nodal = t.where(only_mid[:, None], acc / safe[:, None], nodal)
    elem_all = np.concatenate(elem_parts) if xp.name == "numpy" else xp.t.cat(elem_parts)
    if raw:
        return nodal, elem_all
    return xp.out(nodal), xp.out(elem_all)


def _field(system, data, names, label, unit, translational=None):
    n = len(system.mesh.nodes)
    k = len(names)
    return FEField(np.ascontiguousarray(data).reshape(n * k), n_nodes=n, dofs_per_node=k, dof_names=names,
                   aliases={}, translational=translational or (False,) * k, mesh=system.mesh, label=label,
                   units=unit, dof_units=(unit or "",) * k if unit else None)


def _stress_unit(system):
    return (system.units.stress or None) if system.units is not None else None


def stress_tensor(system, U, D=None, device="cpu", at="nodes"):
    """Differentiable stress as a torch tensor ``(n_nodes, ncomp)`` (``at="elements"``: ``(n_elements, ncomp)``).

    ``U`` may be a torch tensor with ``requires_grad=True``; gradients flow back to it. Constant ``D`` only."""
    nodal, elem = _recover(system, U, D, "stress", backend="torch", device=device, raw=True)
    return elem if at == "elements" else nodal


def von_mises_tensor(system, U, D=None, device="cpu", plane="stress", nu=None, at="nodes"):
    """Differentiable von Mises stress as a torch tensor; see :func:`stress_tensor`."""
    if plane == "strain" and nu is None:
        raise ValueError("von_mises(plane='strain') needs the Poisson ratio nu= to form sigma_zz.")
    s = stress_tensor(system, U, D, device=device, at=at)
    return _mises(s, plane, nu)


def stress(system, U, D=None, at="nodes", vectorized=True, backend="numpy", device="cpu"):
    """Stress components; see the module docstring. ``at``: ``"nodes"`` (FEField) or ``"elements"``.
    ``vectorized=False`` forces the per-element reference loop; ``backend="torch"`` evaluates on ``device``."""
    nodal, elem = _recover(system, U, D, "stress", vectorized, backend, device)
    if at == "elements":
        return elem
    if at != "nodes":
        raise ValueError("at must be 'nodes' or 'elements'.")
    return _field(system, nodal, _NAMES[("stress", nodal.shape[1])], "stress", _stress_unit(system))


def strain(system, U, at="nodes", vectorized=True, backend="numpy", device="cpu"):
    """Strain components (engineering shear), same layout as :func:`stress`."""
    nodal, elem = _recover(system, U, None, "strain", vectorized, backend, device)
    if at == "elements":
        return elem
    if at != "nodes":
        raise ValueError("at must be 'nodes' or 'elements'.")
    return _field(system, nodal, _NAMES[("strain", nodal.shape[1])], "strain", None)


def _mises(s, plane, nu):
    if hasattr(s, "detach"):                         # torch tensor: stay on the autograd graph
        import torch
        sqrt = torch.sqrt
    else:
        sqrt = np.sqrt
    if s.shape[-1] == 6:
        sxx, syy, szz, sxy, syz, sxz = (s[..., j] for j in range(6))
        return sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                    + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))
    sxx, syy, sxy = (s[..., j] for j in range(3))
    if plane == "stress":
        return sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)
    szz = nu * (sxx + syy)
    return sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * sxy ** 2)


def von_mises(system, U, D=None, plane="stress", nu=None, at="nodes", vectorized=True, backend="numpy",
              device="cpu"):
    """Von Mises equivalent stress. 2-D: ``plane="stress"`` (default) or ``"strain"`` (needs ``nu``)."""
    if plane not in ("stress", "strain"):
        raise ValueError("plane must be 'stress' or 'strain'.")
    if plane == "strain" and nu is None:
        raise ValueError("von_mises(plane='strain') needs the Poisson ratio nu= to form sigma_zz.")
    nodal, elem = _recover(system, U, D, "stress", vectorized, backend, device)
    if at == "elements":
        return _mises(elem, plane, nu)
    if at != "nodes":
        raise ValueError("at must be 'nodes' or 'elements'.")
    unit = _stress_unit(system)
    f = _field(system, _mises(nodal, plane, nu), ("von_mises",), "von Mises stress", unit, translational=(True,))
    f._aliases = {"vm": "von_mises", "mises": "von_mises"}
    f.units = unit
    return f


def reactions(system, U=None):
    """Support reactions ``R = K U - F`` at the constrained DOFs (zero elsewhere) as an FEField.

    Forces for translational DOFs, moments for rotations. Sum a component over a node set to get the
    total reaction; with the applied loads in ``system.F`` it balances to rounding error.
    """
    if U is None:
        U = system.solve_static()
    U = _check_vector(system, U)
    system._require_stiffness("reactions")
    K = system._as_solve_matrix(system.K)
    r = np.asarray(K @ U).ravel() - np.asarray(system.F).ravel()
    out = np.zeros(system.n_dof)
    fixed = system.fixed_dofs_array
    out[fixed] = r[fixed]
    units = system.units
    elem = system._blocks[0][1]
    mask = elem.translational_dof_mask
    mask = mask if mask is not None else (True,) * system.npn
    dof_units = None
    if units is not None:
        dof_units = tuple(units.force if t else f"{units.force}*{units.length}" for t in mask)
    f = system.field(out, label="reaction")
    if isinstance(f, FEField):
        f._dof_units = dof_units
        f.units = units.force if units is not None else None
    return f
