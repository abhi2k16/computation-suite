"""
plate_modal_identification.py -- end-to-end rom_engine.loewner /
rom_engine.screening example: non-intrusive modal identification on
the "Example 1" simply-supported rectangular steel plate from

    Liu, J. & Li, S. (2026). "A Reduced-Order Model for Modal Parameter
    Identification of Fluid-Structure Interaction Systems Based on
    Vibration Responses." J. Vib. Eng. Technol., 14:335.

Pipeline demonstrated (self-contained -- this script builds its own
plate FE model, the same way every other example in this package
builds its own fea_engine model, rather than importing tests/ fixture
code):

  1. Build the paper's plate FE model (867 structural DOF, 16x16 mesh,
     from-scratch Reissner-Mindlin plate element) at both the paper's
     stated Table-1 excitation/measurement node locations, for two
     fluid-loading cases (air, water -- an approximate added-mass
     model; see the module-level caveat below).
  2. Generate SYNTHETIC vibration response data x(omega) at those
     sensor locations via a direct complex FRF solve -- standing in for
     "measured/simulated vibration responses," exactly as an
     experimentalist would hand data to this method.
  3. Feed ONLY that response data (never M, C, K again) to
     rom_engine.loewner.LoewnerROM.fit() and
     rom_engine.screening.screen_physical_modes() to identify natural
     frequencies, damping ratios, and (via cross-ROM stability
     screening) which of the raw identified eigenpairs are genuinely
     physical.
  4. Reconstruct multi-DOF mode shapes (Eq. 28) and compare against the
     FE model's own eigen-decomposition via
     rom_engine.metrics.modal_assurance_criterion.
  5. Report and plot identified vs. true frequency/damping for both
     loading cases.

Caveat on fluid loading (same as the paper-benchmark test fixture):
the paper couples the plate to the surrounding fluid via a frequency-
dependent BEM-computed acoustic impedance matrix from a proprietary,
much finer (ND=32321) commercial model that cannot be regenerated
here. This script uses an approximate, frequency-INDEPENDENT added-
mass model instead (captures the right order of magnitude/direction of
the fluid effect, but omits fluid radiation damping -- the paper's
dominant damping mechanism). The point of this script is to
demonstrate and validate the IDENTIFICATION METHOD against a real,
paper-matched structural model's own honest ground truth -- not to
reproduce the paper's exact Table 2 numbers.
"""
__author__ = "Abhijeet"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from rom_engine.loewner import LoewnerROM
from rom_engine.screening import screen_physical_modes
from rom_engine.metrics import modal_assurance_criterion


# =====================================================================
# 1. Plate mesh + from-scratch Reissner-Mindlin plate element
# =====================================================================
def _shape_derivs(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                          (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dN_dxi, dN_deta


def plate_mesh(Lx, Ly, nex, ney):
    nnx, nny = nex + 1, ney + 1
    xs = np.linspace(-Lx / 2, Lx / 2, nnx)
    ys = np.linspace(-Ly / 2, Ly / 2, nny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i, j):
        return i * nny + j

    elems = [[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
             for i in range(nex) for j in range(ney)]
    return coords, np.array(elems)


def element_matrices(xy, E, nu, rho, h, ks=5 / 6, rho_add=0.0):
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
    fixed = 3 * np.where(boundary)[0]
    return np.setdiff1d(np.arange(ndof), fixed), fixed


def modal_solve(K, M, free, n_modes=9):
    vals, vecs = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
    vals = np.clip(vals, 0, None)
    f = np.sqrt(vals) / (2 * np.pi)
    order = np.argsort(f)
    return f[order][:n_modes], vecs[:, order][:, :n_modes]


def added_surface_density(rho_fluid, Lx, Ly):
    a_eq = np.sqrt(Lx * Ly / np.pi)
    return (8.0 / 3.0) * rho_fluid * a_eq ** 3 / (Lx * Ly)


def frf(omega, F, M, C, K):
    """Direct complex FRF solve -- the ONLY function in this script
    that touches M, C, K downstream of assembly. Everything after its
    return value is response-data-only, exactly as LoewnerROM.fit()
    requires."""
    omega = np.atleast_1d(omega).astype(complex)
    n = M.shape[0]
    X = np.zeros((n, len(omega)), dtype=complex)
    for k, w in enumerate(omega):
        D = K + 1j * w * C - w ** 2 * M
        X[:, k] = np.linalg.solve(D, F)
    return X


def ground_truth_modes(M, C, K, alpha, beta, fmin, fmax):
    wn2, Phi = eigh(K, M)
    wn = np.sqrt(np.abs(wn2))
    zeta = (alpha + beta * wn ** 2) / (2 * wn)
    f = wn * np.sqrt(np.clip(1 - zeta ** 2, 0, None)) / (2 * np.pi)
    eta = zeta / np.sqrt(np.clip(1 - zeta ** 2, 1e-12, None))
    mask = (f >= fmin) & (f <= fmax)
    order = np.argsort(f[mask])
    return f[mask][order], eta[mask][order], Phi[:, mask][:, order]


# =====================================================================
# 2. Build the paper's plate model
# =====================================================================
Lx, Ly, h = 0.455, 0.379, 0.003
E, nu, rho_s = 2.1e11, 0.3, 7850.0
alpha_R, beta_R = 7.362, 1.177e-5
rho_air, rho_water = 1.21, 1000.0
nex = ney = 16

coords, elems = plate_mesh(Lx, Ly, nex, ney)
n_nodes = coords.shape[0]
ndof = 3 * n_nodes
K, M_dry = assemble(coords, elems, E, nu, rho_s, h)
free, fixed = simply_supported_free_dofs(coords, Lx, Ly, ndof)

rho_add_air = added_surface_density(rho_air, Lx, Ly)
rho_add_water = added_surface_density(rho_water, Lx, Ly)
_, M_air = assemble(coords, elems, E, nu, rho_s, h, rho_add=rho_add_air)
_, M_water = assemble(coords, elems, E, nu, rho_s, h, rho_add=rho_add_water)
f_air, _ = modal_solve(K, M_air, free)
f_water, _ = modal_solve(K, M_water, free)

table1_mm = {32: (0.0, 23.7), 50: (85.3, 94.8), 58: (113.8, 118.4), 136: (142.2, -94.8),
             150: (199.1, -94.8), 193: (-142.2, 71.1), 265: (-113.8, -94.8)}
table1_fe_node = {}
for pid, (xmm, ymm) in table1_mm.items():
    d = np.linalg.norm(coords - np.array([xmm, ymm]) / 1000.0, axis=1)
    table1_fe_node[pid] = int(np.argmin(d))

print(f"Plate model: {nex}x{ney} elements, {n_nodes} nodes, {ndof} structural DOF "
      f"(paper's ANSYS+BEM model uses ND=32321, a much finer proprietary "
      f"discretization -- not reproduced here, see module docstring)")


# =====================================================================
# 3. Non-intrusive identification, per fluid-loading case
# =====================================================================
def run_case(case_name, M_wet, f_wet_undamped, excite_pid, meas_pids,
             rng, n_target_modes=7):
    print(f"\n{'='*70}\n{case_name.upper()} LOADING\n{'='*70}")
    free_index_of = -np.ones(ndof, dtype=int)
    free_index_of[free] = np.arange(len(free))

    def w_dof(fe_node):
        return free_index_of[3 * fe_node]

    K_ff = K[np.ix_(free, free)]
    M_ff = M_wet[np.ix_(free, free)]
    C_ff = alpha_R * M_ff + beta_R * K_ff

    fmin, fmax = 0.7 * f_wet_undamped[0], 1.2 * f_wet_undamped[n_target_modes - 1]
    f_true, eta_true, Phi_true = ground_truth_modes(M_ff, C_ff, K_ff, alpha_R, beta_R, fmin, fmax)
    n_modes = len(f_true)
    print(f"Target band: {fmin:.2f}-{fmax:.2f} Hz, {n_modes} true modes")

    excite_dof = w_dof(table1_fe_node[excite_pid])
    meas_dofs = [w_dof(table1_fe_node[p]) for p in meas_pids]
    F_ff = np.zeros(len(free)); F_ff[excite_dof] = 1.0

    n_pool = 60
    f_pool_hz = np.linspace(fmin, fmax, n_pool) * (1 + 1e-3 * rng.standard_normal(n_pool))
    omega_pool = 2 * np.pi * f_pool_hz
    x_pool_all = frf(omega_pool, F_ff, M_ff, C_ff, K_ff)
    x_pool_ref = x_pool_all[meas_dofs[0], :]   # node 58 -- avoids node 32's nodal line

    n_interp = n_modes + 3
    idx = rng.choice(n_pool, size=2 * n_interp, replace=False)
    idx_a, idx_b = idx[:n_interp], idx[n_interp:]
    rom = LoewnerROM.fit(omega_pool[idx_a], omega_pool[idx_b],
                          x_pool_ref[idx_a], x_pool_ref[idx_b])
    in_band = (rom.f >= fmin) & (rom.f <= fmax) & (rom.eta >= 0) & (rom.eta <= 1)
    print(f"Single LoewnerROM (order {n_interp}): {in_band.sum()} raw eigenpairs in "
          f"band (incl. possible spurious) vs {n_modes} true modes")

    physical = screen_physical_modes(omega_pool, x_pool_ref, fmin, fmax,
                                      rng=rng, n_roms=25, n_interp=n_interp)
    print(f"After cross-ROM stability screening: {len(physical)} physical modes "
          f"retained (target {n_modes})")

    X_beta_multi = frf(omega_pool[idx_b], F_ff, M_ff, C_ff, K_ff)[meas_dofs, :]
    Phi_ident = rom.reconstruct_mode_shapes(X_beta_multi, omega_beta=omega_pool[idx_b])

    print(f"\n{'Mode':>4} {'f_true':>9} {'f_ROM':>9} {'err%':>7} "
          f"{'eta_true':>10} {'eta_ROM':>10} {'err%':>7} {'MAC':>7}")
    rows = []
    phys_f = np.array([m.f for m in physical]) if physical else np.array([])
    matched_true = set()
    for i in range(n_modes):
        if len(phys_f) == 0:
            break
        k = np.argmin(np.abs(phys_f - f_true[i]))
        if k in matched_true or abs(phys_f[k] - f_true[i]) / f_true[i] > 0.05:
            print(f"{i+1:>4} {f_true[i]:>9.3f}   -- not identified "
                  f"(no matching screened mode within 5%) --")
            continue
        matched_true.add(k)
        f_id, e_id, s_tot, cnt = physical[k]
        j = np.argmin(np.abs(rom.f - f_true[i]))
        mac = modal_assurance_criterion(Phi_ident[:, j], Phi_true[meas_dofs, i])
        f_err = 100 * (f_id - f_true[i]) / f_true[i]
        e_err = 100 * (e_id - eta_true[i]) / eta_true[i]
        print(f"{i+1:>4} {f_true[i]:>9.3f} {f_id:>9.3f} {f_err:>7.3f} "
              f"{eta_true[i]:>10.5f} {e_id:>10.5f} {e_err:>7.3f} {mac:>7.4f}")
        rows.append((f_true[i], f_id, eta_true[i], e_id, mac))

    return dict(fmin=fmin, fmax=fmax, f_true=f_true, eta_true=eta_true,
                physical=physical, rows=np.array(rows))


results = {}
results["air"] = run_case("air", M_air, f_air, excite_pid=50,
                           meas_pids=[58, 32, 136, 150, 193, 265], rng=np.random.default_rng(1))
results["water"] = run_case("water", M_water, f_water, excite_pid=50,
                             meas_pids=[58, 32, 136, 150, 193, 265], rng=np.random.default_rng(2))

# =====================================================================
# 4. Summary plot
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for col, case in enumerate(["air", "water"]):
    r = results[case]["rows"]
    ax_f = axes[0, col]
    ax_f.plot(r[:, 0], r[:, 0], "k--", lw=1, label="perfect ID")
    ax_f.plot(r[:, 0], r[:, 1], "ro", ms=6, label="ROM identified")
    ax_f.set_title(f"{case}-loaded: frequency"); ax_f.set_xlabel("f_true [Hz]")
    ax_f.set_ylabel("f_ROM [Hz]"); ax_f.legend(fontsize=8)

    ax_e = axes[1, col]
    ax_e.plot(r[:, 2], r[:, 2], "k--", lw=1, label="perfect ID")
    ax_e.plot(r[:, 2], r[:, 3], "bo", ms=6, label="ROM identified")
    ax_e.set_title(f"{case}-loaded: damping ratio"); ax_e.set_xlabel("eta_true")
    ax_e.set_ylabel("eta_ROM"); ax_e.legend(fontsize=8)

fig.suptitle("Plate FE model (ground truth) vs. non-intrusive LoewnerROM "
             "identification -- Example 1", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("plate_modal_identification.png", dpi=150)
print("\nSaved plot: plate_modal_identification.png")
