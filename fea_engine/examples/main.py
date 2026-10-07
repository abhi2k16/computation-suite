"""
main.py -- Module 5: post-processing, visualization, and execution
pipeline.

Part A runs four physically distinct STATIC problems through the same
four modules (config, mesh, element, solver) to prove the package is
genuinely additive: none of config.py/mesh.py/element.py/solver.py were
written per-problem, and adding the plate or the 3-D solid case
required zero changes to solver.py. Each result is checked against a
value already independently validated earlier in this project.

    1) 1-D cantilever beam, static      (Beam2DEulerBernoulli)
    2) 2-D plane-stress deep bracket    (Quad4PlaneStress)
    3) 2-D Mindlin cantilever plate     (Quad4MindlinPlate)
    4) 3-D solid cantilever beam        (Hex8Solid3D)

Part B exercises the four DYNAMICS formulations added to solver.py
(direct time integration, modal superposition, harmonic response,
random vibration), all on the same 1-D beam from (1) so the results
can be checked against each other:

    5) Direct time integration: implicit (Newmark-beta) vs explicit
       (central difference) agreement on a step-load transient
    6) Modal superposition vs direct implicit integration agreement
    7) Harmonic response: Omega=0 recovers the static solution exactly;
       resonance amplification matches the calibrated damping ratio
    8) Random vibration: frequency-domain PSD integration vs an
       independent time-domain Monte Carlo simulation (random-phase
       synthesis of a matching input PSD, run through the SAME
       implicit transient solver from item 5)

Part C exercises the full-vs-reduced integration options added to
element.py, on the SAME three static problems from Part A so the
effect of the integration scheme is isolated from everything else:

    9) Hex8: a single element's rank deficiency under reduced
       integration (an exact, independently-checkable number), then
       the mesh-level consequence -- full integration locks, uniform
       reduced integration is UNSTABLE here (no hourglass control is
       implemented in this package)
   10) Quad4MindlinPlate: 'full' integration deliberately reproduces
       the historic shear-locking bug found earlier in this project;
       'sri' (the default) is correct; plain 'reduced' happens to
       remain usable for this mesh/load but is NOT unconditionally safe
   11) Quad4PlaneStress: full vs. reduced on the bracket -- the one
       case here where reduced integration is both stable and slightly
       more accurate, with no caveats
"""
import numpy as np
import matplotlib.pyplot as plt

from fea_engine.material import (Material, Section, D_plane_stress, D_mindlin_plate,
                                 D_solid3d, EI_beam)
from fea_engine.damping import RayleighDamping
from fea_engine.mesh import (Mesh, line_mesh, rectangle_mesh, box_mesh,
                             plot_mesh_2d, plot_mesh_3d)
from fea_engine.elements import (Quad4PlaneStress, Quad4MindlinPlate, Hex8Solid3D,
                                 Beam2DEulerBernoulli, spurious_zero_energy_modes)
from fea_engine.solver import FESystem
from fea_engine.loads import LoadPattern, TimeHistoryLoad, HarmonicLoad, PSDLoad

steel = Material(E=2.1e11, nu=0.3, rho=7850.0)

# =====================================================================
# 1) 1-D cantilever beam, static (Beam2DEulerBernoulli)
# =====================================================================
print("=" * 70)
print("1) 1-D cantilever beam (Beam2DEulerBernoulli)")
L, F_tip = 1.0, -1000.0
sec = Section(A=8.0e-4, I=1.667e-7)
mesh1 = line_mesh(L, n=10)
elem1 = Beam2DEulerBernoulli()
sys1 = FESystem(mesh1, elem1)
sys1.assemble_stiffness(EI_beam(steel, sec))
tip_node = len(mesh1.nodes) - 1
sys1.add_nodal_force([tip_node], dof_index=0, total_force=F_tip)
sys1.fix_dofs([0], [0, 1])
U1 = sys1.solve_static()
w_tip_fem = U1[2 * tip_node]
w_tip_EB = F_tip * L**3 / (3 * steel.E * sec.I)
print(f"  FEM tip deflection:  {w_tip_fem:.6f} m")
print(f"  Euler-Bernoulli:     {w_tip_EB:.6f} m")
print(f"  match: {np.isclose(w_tip_fem, w_tip_EB, rtol=1e-6)}  "
      "(cubic Hermite beam elements are exact for this loading)")

# =====================================================================
# 2) 2-D plane-stress deep bracket (Quad4PlaneStress)
# =====================================================================
print("\n" + "=" * 70)
print("2) 2-D plane-stress deep cantilever bracket (Quad4PlaneStress)")
Lx, Ly, t = 0.4, 0.2, 0.02
F_total = -20000.0
nx, ny = 24, 12
mesh2 = rectangle_mesh(Lx, Ly, nx, ny)
elem2 = Quad4PlaneStress()
mesh2.check_quality(elem2)
sys2 = FESystem(mesh2, elem2, thickness=t)
sys2.assemble_stiffness(D_plane_stress(steel), thickness=t)
tip_nodes = mesh2.nodes_on_line(axis=0, value=Lx)
sys2.add_nodal_force(tip_nodes, dof_index=1, total_force=F_total)
fixed_nodes = mesh2.nodes_on_line(axis=0, value=0.0)
sys2.fix_dofs(fixed_nodes, [0, 1])
U2 = sys2.solve_static()
tip_center = tip_nodes[np.argmin(np.abs(mesh2.nodes[tip_nodes, 1] - Ly / 2))]
v_tip_fem = U2[2 * tip_center + 1]

I_bracket = t * Ly**3 / 12
G = steel.G
k_shear = 5.0 / 6.0
v_timoshenko = (F_total * Lx**3) / (3 * steel.E * I_bracket) + \
               (F_total * Lx) / (G * k_shear * Ly * t)
print(f"  FEM tip deflection:     {v_tip_fem:.6e} m")
print(f"  Timoshenko reference:   {v_timoshenko:.6e} m")
print(f"  ratio: {v_tip_fem / v_timoshenko:.4f}  "
      "(expected ~0.98, matching deep_bracket_plane_stress_fem.py)")

# =====================================================================
# 3) 2-D Mindlin cantilever plate (Quad4MindlinPlate)
# =====================================================================
print("\n" + "=" * 70)
print("3) 2-D Mindlin cantilever plate (Quad4MindlinPlate)")
Lx_p, Ly_p, h_p = 1.0, 0.2, 0.02
F_total_p = -1000.0
nx_p, ny_p = 20, 4
mesh3 = rectangle_mesh(Lx_p, Ly_p, nx_p, ny_p)
elem3 = Quad4MindlinPlate()
sys3 = FESystem(mesh3, elem3)
Db_Ds = D_mindlin_plate(steel, h_p)
sys3.assemble_stiffness(Db_Ds)
tip_nodes_p = mesh3.nodes_on_line(axis=0, value=Lx_p)
sys3.add_nodal_force(tip_nodes_p, dof_index=0, total_force=F_total_p)
fixed_nodes_p = mesh3.nodes_on_line(axis=0, value=0.0)
sys3.fix_dofs(fixed_nodes_p, [0, 1, 2])
U3 = sys3.solve_static()
tip_center_p = tip_nodes_p[np.argmin(np.abs(mesh3.nodes[tip_nodes_p, 1] - Ly_p / 2))]
w_tip_plate_fem = U3[3 * tip_center_p]

I_equiv = Ly_p * h_p**3 / 12
w_beam_equiv = F_total_p * Lx_p**3 / (3 * steel.E * I_equiv)
print(f"  FEM tip-center deflection: {w_tip_plate_fem:.6e} m")
print(f"  Equivalent-beam estimate:  {w_beam_equiv:.6e} m")
print(f"  ratio: {w_tip_plate_fem / w_beam_equiv:.4f}  "
      "(expected ~0.98, matching cantilever_plate_fem.py)")

# =====================================================================
# 4) 3-D solid cantilever beam (Hex8Solid3D)
# =====================================================================
print("\n" + "=" * 70)
print("4) 3-D solid cantilever beam (Hex8Solid3D)")
L3, h_cs, w_cs = 1.0, 0.05, 0.05
F_tip3 = -1000.0
nx3, ny3, nz3 = 20, 4, 4
mesh4 = box_mesh(L3, w_cs, h_cs, nx3, ny3, nz3)
elem4 = Hex8Solid3D()
mesh4.check_quality(elem4)
sys4 = FESystem(mesh4, elem4)
sys4.assemble_stiffness(D_solid3d(steel))
tip_nodes3 = mesh4.nodes_on_plane(axis=0, value=L3)
sys4.add_nodal_force(tip_nodes3, dof_index=2, total_force=F_tip3)
fixed_nodes3 = mesh4.nodes_on_plane(axis=0, value=0.0)
sys4.fix_dofs(fixed_nodes3, [0, 1, 2])
U4 = sys4.solve_static()
tip_center3 = tip_nodes3[np.argmin(np.abs(mesh4.nodes[tip_nodes3, 1] - w_cs / 2))]
w_tip3_fem = U4[3 * tip_center3 + 2]

I3 = w_cs * h_cs**3 / 12
w_tip3_EB = F_tip3 * L3**3 / (3 * steel.E * I3)
print(f"  FEM tip deflection:  {w_tip3_fem:.6e} m")
print(f"  Euler-Bernoulli:     {w_tip3_EB:.6e} m")
print(f"  ratio: {w_tip3_fem / w_tip3_EB:.4f}  (expected ~0.71 at this mesh -- "
      "Hex8 shear locking, see cantilever_beam_3d_fem.py for the full discussion)")

# =====================================================================
# 5)-8) Dynamics formulations, all on the 1-D beam from Part 1
# =====================================================================
def build_beam_system():
    s = FESystem(mesh1, elem1)
    s.assemble_stiffness(EI_beam(steel, sec))
    s.assemble_mass(steel.rho * sec.A)
    s.fix_dofs([0], [0, 1])
    return s


sys_modal = build_beam_system()
freq_hz_dyn, mode_shapes_dyn = sys_modal.solve_modal(n_modes=4)
omega1, omega4 = 2 * np.pi * freq_hz_dyn[0], 2 * np.pi * freq_hz_dyn[3]
damping = RayleighDamping.calibrate(omega1, omega4, zeta=0.02)
dt_crit = sys_modal.critical_timestep()

pattern = LoadPattern(node_ids=np.array([tip_node]), dof_index=0)

# --- 5) implicit vs explicit, step load -------------------------------
print("\n" + "=" * 70)
print("5) Direct time integration: implicit vs explicit (step load)")
step_load = TimeHistoryLoad(pattern, time_fn=lambda tt: F_tip)
dt_dyn = 0.4 * dt_crit
T_dyn = 0.05

sys_imp = build_beam_system(); sys_imp.assemble_damping(damping)
t_imp, U_imp = sys_imp.solve_transient_implicit(step_load, T_dyn, dt_dyn)

sys_exp = build_beam_system()
sys_exp.assemble_damping(damping)
sys_exp.assemble_lumped_mass(steel.rho * sec.A)
t_exp, U_exp = sys_exp.solve_transient_explicit(step_load, T_dyn, dt_dyn)

tip_imp, tip_exp = U_imp[:, 2 * tip_node], U_exp[:, 2 * tip_node]
rel_diff_dyn = np.max(np.abs(tip_imp - tip_exp)) / np.max(np.abs(tip_imp))
print(f"  critical timestep (explicit stability limit): {dt_crit:.3e} s, using dt={dt_dyn:.3e} s")
print(f"  max relative difference between methods: {rel_diff_dyn:.4f}  (expected: a few percent,")
print("  from consistent-vs-lumped mass -- not exact agreement, but close)")

# --- 6) modal superposition vs direct implicit -------------------------
print("\n" + "=" * 70)
print("6) Modal superposition vs direct implicit integration")
T_ms, dt_ms = 0.3, 3e-5
sys_imp2 = build_beam_system(); sys_imp2.assemble_damping(damping)
t_imp2, U_imp2 = sys_imp2.solve_transient_implicit(step_load, T_ms, dt_ms)
sys_ms = build_beam_system()
t_ms, U_ms, _ = sys_ms.solve_modal_superposition(step_load, T_ms, dt_ms, n_modes=8, zeta=0.02)
tip_imp2, tip_ms = U_imp2[:, 2 * tip_node], U_ms[:, 2 * tip_node]
rel_diff_ms = np.max(np.abs(tip_imp2 - tip_ms)) / np.max(np.abs(tip_imp2))
print(f"  8 of {len(sys_ms.free_dofs)} modes retained")
print(f"  max relative difference vs direct integration: {rel_diff_ms:.4f}  (expected: well under 1%)")

# --- 7) harmonic response: static limit + resonance --------------------
print("\n" + "=" * 70)
print("7) Harmonic response (frequency domain)")
sys_h = build_beam_system(); sys_h.assemble_damping(damping)
hload = HarmonicLoad(pattern, F0=F_tip)
F0_vec = hload.force_vector(sys_h.n_dof, sys_h.npn)
sys_h.F = F0_vec.copy()
U_static_h = sys_h.solve_static()
U0_zero = sys_h.solve_harmonic(0.0, F0_vec)
print(f"  static solve tip deflection:      {U_static_h[2*tip_node]:.6e} m")
print(f"  harmonic solve at Omega=0:         {U0_zero[2*tip_node].real:.6e} m  "
      f"(match: {np.isclose(U_static_h[2*tip_node], U0_zero[2*tip_node].real, rtol=1e-6)})")

freqs_hz_sweep = np.linspace(0.1, 1.3 * freq_hz_dyn[3], 6000)
U_sweep = sys_h.solve_frequency_sweep(2 * np.pi * freqs_hz_sweep, F0_vec)
DAF_sweep = np.abs(U_sweep[:, 2 * tip_node]) / np.abs(U_static_h[2 * tip_node])
peak_idx = np.argmax(DAF_sweep)
zeta1 = damping.modal_ratio(omega1)
print(f"  peak DAF: {DAF_sweep[peak_idx]:.2f} at {freqs_hz_sweep[peak_idx]:.2f} Hz  "
      f"(SDOF estimate 1/(2*zeta1) = {1/(2*zeta1):.2f}, small downward shift is expected damped-resonance behavior)")

# --- 8) random vibration: frequency-domain vs time-domain Monte Carlo --
print("\n" + "=" * 70)
print("8) Random vibration (PSD): frequency-domain vs time-domain cross-check")
freqs_psd = np.linspace(0.5, 200, 800)
S0 = 1.0e4
psd_in = np.full_like(freqs_psd, S0)
psd_load = PSDLoad(pattern, freqs_psd, psd_in)
F0_unit = psd_load.force_vector(sys_h.n_dof, sys_h.npn)
S_out, sigma_freq = sys_h.solve_random_vibration(freqs_psd, psd_in, F0_unit, output_dof=2 * tip_node)
print(f"  frequency-domain sigma (response RMS): {sigma_freq:.5f} m")

df = freqs_psd[1] - freqs_psd[0]
amps = np.sqrt(2 * S0 * df)
dt_td = 1.0 / (20 * freqs_psd[-1])
T_td = 4.0
sigmas_mc = []
for seed in range(4):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, size=len(freqs_psd))

    def F_of_t(tt, phases=phases):
        return np.sum(amps * np.cos(2 * np.pi * freqs_psd * tt + phases))

    rand_load = TimeHistoryLoad(pattern, time_fn=F_of_t)
    sys_mc = build_beam_system(); sys_mc.assemble_damping(damping)
    _, U_mc = sys_mc.solve_transient_implicit(rand_load, T_td, dt_td)
    tip_mc = U_mc[int(0.2 * len(U_mc)):, 2 * tip_node]   # discard initial transient settling
    sigmas_mc.append(np.std(tip_mc))
sigmas_mc = np.array(sigmas_mc)
print(f"  time-domain Monte Carlo sigma (4 independent realizations): "
      f"{np.round(sigmas_mc, 5)}")
print(f"  mean ratio (MC/frequency-domain): {np.mean(sigmas_mc)/sigma_freq:.3f}  "
      "(expected close to 1.0, with realization-to-realization scatter --")
print("  this is a statistical cross-check, not an exact-match one, unlike items 5-7)")

# =====================================================================
# 9)-11) Full vs. reduced integration
# =====================================================================
def near_zero_modes(K_free, tol=1e-6):
    eigs = np.linalg.eigvalsh(K_free)
    return int(np.sum(np.abs(eigs) < tol * np.max(np.abs(eigs)))), eigs

print("\n" + "=" * 70)
print("9) Hex8: reduced-integration rank deficiency (single element + mesh)")
elem4_coords = mesh4.nodes[mesh4.elements[0]]
n_spurious, _ = spurious_zero_energy_modes(elem4, elem4_coords, D_solid3d(steel))
print(f"  single element: {n_spurious} spurious modes under 1-pt reduced integration")
print("  (exact, checkable by hand: 24 dof - 6 rigid-body = 18 nonzero under full;")
print("   a single Gauss point gives Bt.D.B rank <= 6, so reduced keeps at most 6 -> 12 spurious)")

sys4_red = FESystem(mesh4, elem4)
sys4_red.assemble_stiffness(D_solid3d(steel), gauss_order=1)
sys4_red.fix_dofs(fixed_nodes3, [0, 1, 2])
free4 = sys4_red.free_dofs
n_nz4, eigs4 = near_zero_modes(sys4_red.K[np.ix_(free4, free4)])
print(f"  mesh level ({nx3}x{ny3}x{nz3} elements): {n_nz4} near-zero eigenvalues remain "
      f"even AFTER applying the fixed-end BC")
print(f"  min|eig|={np.min(np.abs(eigs4)):.2e}, max|eig|={np.max(np.abs(eigs4)):.2e} "
      "-- the system is numerically singular.")
print("  CONCLUSION: uniform reduced integration is UNSAFE for Hex8Solid3D in this")
print("  package (no hourglass stabilization is implemented) -- use full_stiffness()")
print("  (the default) and accept the locking, don't reach for reduced_stiffness() here.")

print("\n" + "=" * 70)
print("10) Quad4MindlinPlate: 'sri' vs 'full' vs 'reduced'")


def solve_plate_integration(integration):
    sys = FESystem(mesh3, elem3)
    sys.assemble_stiffness(Db_Ds, integration=integration)
    sys.add_nodal_force(tip_nodes_p, dof_index=0, total_force=F_total_p)
    sys.fix_dofs(fixed_nodes_p, [0, 1, 2])
    U = sys.solve_static()
    return U, sys


U_sri, _ = solve_plate_integration('sri')
U_full_plate, _ = solve_plate_integration('full')
U_red_plate, sys_red_plate = solve_plate_integration('reduced')
w_sri = U_sri[3 * tip_center_p]
w_full_plate = U_full_plate[3 * tip_center_p]
w_red_plate = U_red_plate[3 * tip_center_p]
free_p = sys_red_plate.free_dofs
n_nz_p, _ = near_zero_modes(sys_red_plate.K[np.ix_(free_p, free_p)])
print(f"  equivalent-beam reference: {w_beam_equiv:.6e} m")
print(f"  'sri'     (default): {w_sri:.6e} m, ratio={w_sri/w_beam_equiv:.4f}  -- correct")
print(f"  'full'    (both terms): {w_full_plate:.6e} m, ratio={w_full_plate/w_beam_equiv:.4f}  "
      "-- reproduces the historic shear-locking bug on purpose")
print(f"  'reduced' (both terms): {w_red_plate:.6e} m, ratio={w_red_plate/w_beam_equiv:.4f}  "
      f"-- {n_nz_p} near-zero modes remain in the constrained system;")
print("            happens to stay usable for THIS mesh/load, but is not unconditionally")
print("            safe the way 'sri' is -- 'sri' is the recommended reduced-integration")
print("            option for this element, not plain 'reduced'.")

print("\n" + "=" * 70)
print("11) Quad4PlaneStress bracket: full vs. reduced")


def solve_bracket_integration(gauss_order):
    sys = FESystem(mesh2, elem2, thickness=t)
    sys.assemble_stiffness(D_plane_stress(steel), thickness=t, gauss_order=gauss_order)
    sys.add_nodal_force(tip_nodes, dof_index=1, total_force=F_total)
    sys.fix_dofs(fixed_nodes, [0, 1])
    U = sys.solve_static()
    return U, sys


U_full_br, _ = solve_bracket_integration(2)
U_red_br, sys_red_br = solve_bracket_integration(1)
v_full_br = U_full_br[2 * tip_center + 1]
v_red_br = U_red_br[2 * tip_center + 1]
free_br = sys_red_br.free_dofs
n_nz_br, _ = near_zero_modes(sys_red_br.K[np.ix_(free_br, free_br)])
print(f"  Timoshenko reference: {v_timoshenko:.6e} m")
print(f"  full (2x2):    {v_full_br:.6e} m, ratio={v_full_br/v_timoshenko:.4f}")
print(f"  reduced (1x1): {v_red_br:.6e} m, ratio={v_red_br/v_timoshenko:.4f}, "
      f"{n_nz_br} near-zero modes -- clean and slightly MORE accurate here, no caveats.")

# =====================================================================
# Post-processing graphics
# =====================================================================
fig = plt.figure(figsize=(13, 10))

ax1 = fig.add_subplot(2, 2, 1)
scale2 = 0.15 * Lx / np.max(np.abs(U2))
nodes2_def = mesh2.nodes + scale2 * U2.reshape(-1, 2)
plot_mesh_2d(mesh2, ax1, color='0.8')
plot_mesh_2d(Mesh(nodes2_def, mesh2.elements, dim=2), ax1, color='b')
ax1.set_title(f"Bracket (Quad4PlaneStress): deformed, {scale2:.0f}x scale")
ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
scale4 = 0.1 * L3 / np.max(np.abs(U4))
nodes4_def = mesh4.nodes + scale4 * U4.reshape(-1, 3)
plot_mesh_3d(mesh4, ax2, color='0.75', lw=0.3)
plot_mesh_3d(Mesh(nodes4_def, mesh4.elements, dim=3), ax2, color='b', lw=0.5)
combined = np.vstack([mesh4.nodes, nodes4_def])
cmin, cmax = combined.min(axis=0), combined.max(axis=0)
ax2.set_xlim(cmin[0], cmax[0]); ax2.set_ylim(cmin[1], cmax[1]); ax2.set_zlim(cmin[2], cmax[2])
ax2.set_title(f"3-D beam (Hex8Solid3D): deformed, {scale4:.0f}x scale")

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(t_imp2 * 1000, tip_imp2 * 1000, 'b-', lw=1, label='direct implicit')
ax3.plot(t_ms * 1000, tip_ms * 1000, 'r--', lw=1, label='modal superposition (8 modes)')
ax3.set_xlabel('time (ms)'); ax3.set_ylabel('tip deflection (mm)')
ax3.set_title('Item 6: modal superposition vs direct integration')
ax3.legend(fontsize=8); ax3.grid(True)

ax4 = fig.add_subplot(2, 2, 4)
ax4.semilogy(freqs_hz_sweep, DAF_sweep, 'b-', lw=1)
for f in freq_hz_dyn:
    ax4.axvline(f, color='gray', ls=':', lw=0.8)
ax4.set_xlabel('driving frequency (Hz)'); ax4.set_ylabel('dynamic amplification factor')
ax4.set_title('Item 7: harmonic response (FRF)')
ax4.grid(True, which='both')

fig.tight_layout()
fig.savefig("fea_package_validation.png", dpi=150)
plt.show()

print("\n" + "=" * 70)
print("All eleven items ran through the same config/mesh/element/solver/loads")
print("modules with no per-item changes to solver.py's assembly logic.")
