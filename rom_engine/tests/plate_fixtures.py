"""
plate_fixtures.py -- the "Example 1" simply-supported rectangular steel
plate benchmark from:

    Liu, J. & Li, S. (2026). "A Reduced-Order Model for Modal Parameter
    Identification of Fluid-Structure Interaction Systems Based on
    Vibration Responses." J. Vib. Eng. Technol., 14:335.

This is a materially harder, materially more realistic validation
target for `rom_engine.loewner`/`rom_engine.screening` than the
mass-spring-chain fixture in `loewner_fixtures.py`: a real (if
hand-rolled, fea_engine-independent) plate finite element model, built
to the paper's own stated geometry/material/mesh, cross-checked against
the closed-form Navier solution.

Ported from the working prototype `plate_fe_model.py` (originally a
standalone script that also saved an .npz and printed a Table-2
comparison) into an importable, parameterized fixture function with NO
script-level side effects (no file I/O, no printing at import time) --
matching this package's `tests/fea_fixtures.py` convention of plain
functions returning plain dicts of numpy arrays.

Geometry / material / mesh (as stated in the paper)
-----------------------------------------------------
  Lx = 0.455 m, Ly = 0.379 m, h = 0.003 m
  rho_s = 7850 kg/m^3, E = 2.1e11 Pa, nu = 0.3
  Rayleigh damping: alpha = 7.362 (mass-proportional),
                     beta  = 1.177e-5 (stiffness-proportional)
  16x16 structured mesh
  Air:   rho_a = 1.21 kg/m^3,  c_a = 340 m/s   (not used directly --
         only the low-ka baffled-piston added-mass magnitude is)
  Water: rho_w = 1000 kg/m^3, c_w = 1500 m/s   (same)

Scope note on fluid loading (unchanged from the prototype): the paper
couples the plate to the surrounding fluid through a frequency-
dependent acoustic impedance matrix computed with the boundary element
method in commercial software, using a much finer (ND=32321),
proprietary discretization that cannot be regenerated here. This
fixture instead adds an approximate, frequency-INDEPENDENT fluid-
loading model (a uniformly distributed added mass, from the classical
low-frequency added mass of a baffled circular piston of equivalent
area), applied only to translational inertia -- captures the right
order of magnitude and direction of the fluid effect (water shifts
frequencies down substantially, air barely shifts them) but does NOT
include fluid radiation damping (the paper's dominant damping
mechanism). This fixture is a legitimate, independent "truth" system
for benchmarking non-intrusive identification against -- not a
reproduction of the paper's exact Table 2 numbers.
"""
__author__ = "Abhijeet"
import numpy as np
from scipy.linalg import eigh


def _shape_derivs(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                          (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dN_dxi, dN_deta


def plate_mesh(Lx, Ly, nex, ney):
    """Regular structured mesh, origin at plate center (matches the
    coordinate convention used for the paper's Table 1 node locations)."""
    nnx, nny = nex + 1, ney + 1
    xs = np.linspace(-Lx / 2, Lx / 2, nnx)
    ys = np.linspace(-Ly / 2, Ly / 2, nny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i, j):
        return i * nny + j

    elems = []
    for i in range(nex):
        for j in range(ney):
            elems.append([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)])
    return coords, np.array(elems)


def element_matrices(xy, E, nu, rho, h, ks=5 / 6, rho_add=0.0):
    """4-node Reissner-Mindlin plate element. DOF order per node:
    [w, theta_x, theta_y]. Selective reduced integration: 2x2 Gauss
    for bending+mass, 1-pt for shear (avoids shear locking). rho_add:
    extra uniform surface density [kg/m^2] added to translational (w)
    inertia only -- the approximate fluid added-mass model."""
    Db = E * h ** 3 / (12 * (1 - nu ** 2)) * np.array(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    G = E / (2 * (1 + nu))
    Ds = ks * G * h * np.eye(2)
    Dm = np.diag([rho * h + rho_add, rho * h ** 3 / 12, rho * h ** 3 / 12])

    Ke = np.zeros((12, 12)); Me = np.zeros((12, 12))
    gp = 1 / np.sqrt(3)
    for xi, eta in [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]:
        N, dN_dxi, dN_deta = _shape_derivs(xi, eta)
        J = np.array([[dN_dxi @ xy[:, 0], dN_dxi @ xy[:, 1]],
                      [dN_deta @ xy[:, 0], dN_deta @ xy[:, 1]]])
        detJ = np.linalg.det(J)
        dN_dxy = np.linalg.inv(J) @ np.vstack([dN_dxi, dN_deta])
        dNdx, dNdy = dN_dxy[0], dN_dxy[1]
        Bb = np.zeros((3, 12)); Nmat = np.zeros((3, 12))
        for a in range(4):
            Bb[0, 3 * a + 2] = dNdx[a]
            Bb[1, 3 * a + 1] = -dNdy[a]
            Bb[2, 3 * a + 1] = -dNdx[a]
            Bb[2, 3 * a + 2] = dNdy[a]
            Nmat[0, 3 * a + 0] = N[a]
            Nmat[1, 3 * a + 1] = N[a]
            Nmat[2, 3 * a + 2] = N[a]
        Ke += (Bb.T @ Db @ Bb) * detJ
        Me += (Nmat.T @ Dm @ Nmat) * detJ

    N, dN_dxi, dN_deta = _shape_derivs(0.0, 0.0)
    J = np.array([[dN_dxi @ xy[:, 0], dN_dxi @ xy[:, 1]],
                  [dN_deta @ xy[:, 0], dN_deta @ xy[:, 1]]])
    detJ = np.linalg.det(J)
    dN_dxy = np.linalg.inv(J) @ np.vstack([dN_dxi, dN_deta])
    dNdx, dNdy = dN_dxy[0], dN_dxy[1]
    Bs = np.zeros((2, 12))
    for a in range(4):
        Bs[0, 3 * a + 0] = dNdx[a]; Bs[0, 3 * a + 2] = N[a]
        Bs[1, 3 * a + 0] = dNdy[a]; Bs[1, 3 * a + 1] = -N[a]
    Ke += (Bs.T @ Ds @ Bs) * detJ * 4.0
    return Ke, Me


def assemble(coords, elems, E, nu, rho, h, rho_add=0.0):
    ndof = 3 * coords.shape[0]
    K = np.zeros((ndof, ndof)); M = np.zeros((ndof, ndof))
    for el in elems:
        xy = coords[el]
        Ke, Me = element_matrices(xy, E, nu, rho, h, rho_add=rho_add)
        dofs = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in el]).ravel()
        K[np.ix_(dofs, dofs)] += Ke
        M[np.ix_(dofs, dofs)] += Me
    return K, M


def simply_supported_free_dofs(coords, Lx, Ly, ndof):
    tol = 1e-9
    boundary = (np.abs(coords[:, 0] - Lx / 2) < tol) | (np.abs(coords[:, 0] + Lx / 2) < tol) | \
               (np.abs(coords[:, 1] - Ly / 2) < tol) | (np.abs(coords[:, 1] + Ly / 2) < tol)
    fixed = 3 * np.where(boundary)[0]  # constrain w only ("hard" SS: Mnn=0 natural BC)
    return np.setdiff1d(np.arange(ndof), fixed), fixed


def modal_solve(K, M, free, n_modes=12):
    vals, vecs = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
    vals = np.clip(vals, 0, None)
    f = np.sqrt(vals) / (2 * np.pi)
    order = np.argsort(f)
    return f[order][:n_modes], vecs[:, order][:, :n_modes]


def added_surface_density(rho_fluid, Lx, Ly):
    """Low-ka added mass of a baffled circular piston of equivalent
    area, smeared uniformly over the plate as an added surface density
    [kg/m^2]."""
    a_eq = np.sqrt(Lx * Ly / np.pi)
    M_add = (8.0 / 3.0) * rho_fluid * a_eq ** 3
    return M_add / (Lx * Ly)


def navier_frequencies(Lx, Ly, E, nu, rho, h, n_modes=9):
    """Closed-form natural frequencies of a simply-supported rectangular
    plate (Navier solution) -- used ONLY to validate the FE model's dry
    (in-vacuo) frequencies, never consumed by any identification method."""
    D = E * h ** 3 / (12 * (1 - nu ** 2))
    navier_modes = [(m, n) for m in range(1, 5) for n in range(1, 5)]
    f = np.array(sorted((np.pi / 2) * np.sqrt(D / (rho * h)) *
                         ((m / Lx) ** 2 + (n / Ly) ** 2) for m, n in navier_modes))
    return f[:n_modes]


# Table 1 excitation/measurement node coordinates (mm, plate-center origin)
TABLE1_MM = {
    32: (0.0, 23.7), 50: (85.3, 94.8), 58: (113.8, 118.4), 136: (142.2, -94.8),
    150: (199.1, -94.8), 193: (-142.2, 71.1), 265: (-113.8, -94.8),
}


def build_paper_plate_system(nex=16, ney=16,
                              Lx=0.455, Ly=0.379, h=0.003,
                              E=2.1e11, nu=0.3, rho_s=7850.0,
                              alpha_R=7.362, beta_R=1.177e-5,
                              rho_air=1.21, rho_water=1000.0):
    """Build the paper's Example-1 plate FE model: dry, air-loaded, and
    water-loaded mass matrices sharing one stiffness matrix, plus
    Rayleigh damping and the Table-1 sensor-node map. All defaults
    match the paper's stated values.

    Returns a dict: coords, elems, K, M_dry, M_air, M_water, C_dry
    (alpha_R*M_dry + beta_R*K), alpha_R, beta_R, free, fixed, n_dof,
    table1_fe_node (dict: paper_node_id -> nearest FE node index),
    f_dry/f_air/f_water (first 9 modal frequencies, Hz, of each case).
    """
    coords, elems = plate_mesh(Lx, Ly, nex, ney)
    n_nodes = coords.shape[0]
    ndof = 3 * n_nodes

    K, M_dry = assemble(coords, elems, E, nu, rho_s, h)
    free, fixed = simply_supported_free_dofs(coords, Lx, Ly, ndof)

    rho_add_air = added_surface_density(rho_air, Lx, Ly)
    rho_add_water = added_surface_density(rho_water, Lx, Ly)
    _, M_air = assemble(coords, elems, E, nu, rho_s, h, rho_add=rho_add_air)
    _, M_water = assemble(coords, elems, E, nu, rho_s, h, rho_add=rho_add_water)

    f_dry, _ = modal_solve(K, M_dry, free, n_modes=9)
    f_air, _ = modal_solve(K, M_air, free, n_modes=9)
    f_water, _ = modal_solve(K, M_water, free, n_modes=9)

    table1_fe_node = {}
    for pid, (xmm, ymm) in TABLE1_MM.items():
        xy = np.array([xmm, ymm]) / 1000.0
        d = np.linalg.norm(coords - xy, axis=1)
        table1_fe_node[pid] = int(np.argmin(d))

    C_dry = alpha_R * M_dry + beta_R * K
    return dict(
        coords=coords, elems=elems, K=K, M_dry=M_dry, M_air=M_air, M_water=M_water,
        C_dry=C_dry, alpha_R=alpha_R, beta_R=beta_R, free=free, fixed=fixed,
        n_dof=ndof, table1_fe_node=table1_fe_node,
        f_dry=f_dry, f_air=f_air, f_water=f_water,
        Lx=Lx, Ly=Ly, h=h, E=E, nu=nu, rho_s=rho_s,
    )
