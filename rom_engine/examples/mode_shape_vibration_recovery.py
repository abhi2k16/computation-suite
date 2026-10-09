"""
mode_shape_vibration_recovery.py -- end-to-end rom_engine.loewner
example: full-field mode-shape recovery and time-domain vibration
recovery from a purely non-intrusive identified modal model.

This goes one step further than frequency/damping identification
(shown in plate_modal_identification.py): it demonstrates that the
identified modal model is actually USABLE, not just numerically close
on paper, via two additional steps built on top of
rom_engine.loewner.LoewnerROM:

  1. FULL-FIELD MODE SHAPE RECOVERY: LoewnerROM.reconstruct_mode_shapes()
     (Eq. 28) applied using response data at EVERY degree of freedom of
     the structure (not just a handful of measurement points),
     recovering the full spatial deflection pattern of each identified
     mode purely from vibration response data + the ROM eigenvectors.
  2. TIME-DOMAIN VIBRATION RECOVERY: given ONLY the identified modal
     model (frequencies, damping ratios, recovered mode shapes -- no M,
     C, K from this point on) and an observed initial-displacement
     snapshot, the modal coordinates are recovered by least-squares
     projection (pinv of the identified mode-shape matrix) and
     propagated forward in time analytically. The recovered response is
     compared against directly time-integrating the FULL system.

Structure: a synthetic proportionally-damped mass-spring-damper chain
(self-contained -- this script builds its own system, the same way
every other example in this package builds its own model, rather than
importing tests/ fixture code), with known closed-form ground-truth
modal parameters.
"""
__author__ = "Abhijeet"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import solve_ivp

from rom_engine.loewner import LoewnerROM
from rom_engine.metrics import modal_assurance_criterion


# =====================================================================
# 1. Synthetic structure + FRF sampling (test-data generation only --
#    nothing downstream of frf()'s output ever sees M, C, or K again)
# =====================================================================
def build_system(n=15, k0=8.0e5, m0=1.0, alpha=0.8, beta=2.0e-5, rng=None):
    M = m0 * np.eye(n)
    k = k0 * (1.0 + 0.15 * rng.standard_normal(n + 1))
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k[i] + k[i + 1]
        if i > 0:
            K[i, i - 1] -= k[i]
            K[i - 1, i] -= k[i]
    C = alpha * M + beta * K
    return M, C, K, alpha, beta


def frf(omega, F, M, C, K):
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


def align_to_real(v_complex, v_ref_real):
    """Rotate a complex mode-shape vector by the phase that best aligns
    it with a real reference, then return its (now ~real) real part,
    unit-normalized -- handles the arbitrary complex scale/phase of ROM
    eigenvectors."""
    phase = np.angle(np.vdot(v_ref_real, v_complex))
    v = (v_complex * np.exp(-1j * phase)).real
    return v / np.max(np.abs(v))


rng = np.random.default_rng(0)
M, C, K, alpha, beta = build_system(n=15, rng=rng)
n_dof = M.shape[0]

f_all, eta_all, _ = ground_truth_modes(M, C, K, alpha, beta, 0, 1e6)
fmin, fmax = 0.7 * f_all[0], 1.3 * f_all[5]
f_true, eta_true, Phi_true = ground_truth_modes(M, C, K, alpha, beta, fmin, fmax)
n_modes = len(f_true)
print(f"Target band: {fmin:.2f}-{fmax:.2f} Hz, {n_modes} true modes")

ref_dof = 4
F = rng.standard_normal(n_dof)   # arbitrary excitation, used only to synthesize data
n_pool = 60
f_pool_hz = np.linspace(fmin, fmax, n_pool) * (1 + 1e-3 * rng.standard_normal(n_pool))
omega_pool = 2 * np.pi * f_pool_hz
x_pool = frf(omega_pool, F, M, C, K)[ref_dof, :]

n_interp = n_modes + 3
idx = rng.choice(n_pool, size=2 * n_interp, replace=False)
idx_a, idx_b = idx[:n_interp], idx[n_interp:]
rom = LoewnerROM.fit(omega_pool[idx_a], omega_pool[idx_b], x_pool[idx_a], x_pool[idx_b])


# =====================================================================
# 2. Full-field mode shape recovery (Eq. 28 applied across ALL DOFs)
# =====================================================================
X_beta_full = frf(omega_pool[idx_b], F, M, C, K)     # (n_dof, n_interp)
Phi_ident_full = rom.reconstruct_mode_shapes(X_beta_full, omega_beta=omega_pool[idx_b])

in_band = (rom.f >= fmin) & (rom.f <= fmax) & (rom.eta >= 0) & (rom.eta <= 1)
rom_idx_in_band = np.where(in_band)[0]

recovered_shapes = np.zeros((n_dof, n_modes))
matched_rom_col = np.zeros(n_modes, dtype=int)
mac_full = np.zeros(n_modes)
for i in range(n_modes):
    j = rom_idx_in_band[np.argmin(np.abs(rom.f[rom_idx_in_band] - f_true[i]))]
    matched_rom_col[i] = j
    phi_true_i = Phi_true[:, i] / np.max(np.abs(Phi_true[:, i]))
    phi_rec_i = align_to_real(Phi_ident_full[:, j], phi_true_i)
    recovered_shapes[:, i] = phi_rec_i
    mac_full[i] = modal_assurance_criterion(phi_rec_i, phi_true_i)

print(f"\n{'Mode':>4} {'f_true[Hz]':>11} {'f_ROM[Hz]':>11} {'MAC(full field)':>16}")
for i in range(n_modes):
    j = matched_rom_col[i]
    print(f"{i+1:>4} {f_true[i]:>11.3f} {rom.f[j]:>11.3f} {mac_full[i]:>16.4f}")


# =====================================================================
# 3. Time-domain vibration recovery from the identified modal model
# =====================================================================
x0 = np.zeros(n_dof)
x0[3:7] = np.array([0.4, 1.0, 0.8, 0.3])   # localized initial deflection ("pluck")

Minv = np.linalg.inv(M)
A_state = np.block([[np.zeros((n_dof, n_dof)), np.eye(n_dof)],
                     [-Minv @ K, -Minv @ C]])
z0 = np.concatenate([x0, np.zeros(n_dof)])
t_eval = np.linspace(0, 0.5, 800)
sol = solve_ivp(lambda t, z: A_state @ z, [t_eval[0], t_eval[-1]], z0,
                 t_eval=t_eval, rtol=1e-9, atol=1e-12)
x_true_t = sol.y[:n_dof, :]   # ground truth: direct full-order time integration

# Recovered response uses ONLY the identified modal parameters
# (f/eta -> decay rate + damped frequency) and the recovered mode
# shapes from Step 2 -- no M, C, K used from here on.
Phi_model = recovered_shapes
decay_rate = np.array([eta_true_i * 2 * np.pi * f_id
                        for eta_true_i, f_id in zip(
                            rom.eta[matched_rom_col], rom.f[matched_rom_col])])
damped_omega = np.array([2 * np.pi * rom.f[matched_rom_col[i]] for i in range(n_modes)])

q0 = np.linalg.pinv(Phi_model) @ x0   # modal-coordinate initial conditions, least-squares
q_t = np.zeros((n_modes, len(t_eval)))
for i in range(n_modes):
    q_t[i, :] = q0[i] * np.exp(-decay_rate[i] * t_eval) * np.cos(damped_omega[i] * t_eval)
x_recovered_t = Phi_model @ q_t

print("\nTime-domain vibration recovery (recovered vs. true), normalized RMS error:")
for dof_check in [2, 4, 8, 12]:
    err = x_recovered_t[dof_check] - x_true_t[dof_check]
    rms = np.sqrt(np.mean(err ** 2)) / (np.max(np.abs(x_true_t[dof_check])) + 1e-12)
    print(f"  DOF {dof_check:2d}: {100*rms:.2f}%")


# =====================================================================
# 4. Plots
# =====================================================================
fig = plt.figure(figsize=(13, 8))
n_plot_modes = min(6, n_modes)
for i in range(n_plot_modes):
    ax = fig.add_subplot(3, 4, i + 1)
    dof_axis = np.arange(1, n_dof + 1)
    ax.plot(dof_axis, Phi_true[:, i] / np.max(np.abs(Phi_true[:, i])),
            "k-o", ms=3, label="true")
    ax.plot(dof_axis, recovered_shapes[:, i], "r--x", ms=4, label="recovered")
    ax.set_title(f"Mode {i+1}: {f_true[i]:.1f} Hz (MAC={mac_full[i]:.3f})", fontsize=9)
    ax.set_xlabel("DOF", fontsize=8)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)

ax_t1 = fig.add_subplot(3, 2, 5)
ax_t1.plot(t_eval, x_true_t[4], "k-", lw=1.2, label="true (DOF 4)")
ax_t1.plot(t_eval, x_recovered_t[4], "r--", lw=1.2, label="recovered (DOF 4)")
ax_t1.set_xlabel("time [s]"); ax_t1.set_ylabel("displacement")
ax_t1.set_title("Vibration recovery at DOF 4 (excited)", fontsize=9)
ax_t1.legend(fontsize=8)

ax_t2 = fig.add_subplot(3, 2, 6)
ax_t2.plot(t_eval, x_true_t[12], "k-", lw=1.2, label="true (DOF 12)")
ax_t2.plot(t_eval, x_recovered_t[12], "r--", lw=1.2, label="recovered (DOF 12)")
ax_t2.set_xlabel("time [s]"); ax_t2.set_ylabel("displacement")
ax_t2.set_title("Vibration recovery at DOF 12 (unexcited)", fontsize=9)
ax_t2.legend(fontsize=8)

fig.suptitle("Mode shape recovery (top) and time-domain vibration recovery (bottom)\n"
             "using only the matrix-free LoewnerROM identified modal parameters", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("mode_shape_and_vibration_recovery.png", dpi=150)
print("\nSaved plot: mode_shape_and_vibration_recovery.png")
