"""Standalone validation of the new bending-membrane coupling term in
Shell4MITCCorotational (added 2026-09-03, see elements/shells.py's
_bending_membrane_coupling_force/_coupling_tangent_local and
tangent_stiffness()). Checks, in order:

  1. internal_force(elem_coords, 0, D) == 0 (required invariant).
  2. Rigid single-element tilt (uniform betax across all 4 nodes)
     produces ~zero coupling force (the key design property that
     distinguishes this from the rejected "dead end" approaches).
  3. Genuine within-element curvature (nodes disagreeing on betax)
     produces a NONZERO coupling force -- confirms the fix actually
     does something.
  4. Analytic (hybrid) tangent_stiffness() matches full complex-step
     differentiation of internal_force() (_tangent_stiffness_complex_
     step()) at several random/large states -- the correctness gate
     before trusting the analytic tangent.
  5. stiffness() (u=0 tangent) still reduces to Shell4MITC.stiffness()
     as before (no regression).
"""
__author__ = "Abhijeet"
import numpy as np
from fea_engine import Material, D_shell, Shell4MITC, Shell4MITCCorotational

np.random.seed(0)
mat = Material(E=210e9, nu=0.3, rho=7800.0)
H = 0.01
D = D_shell(mat, H)
elem = Shell4MITCCorotational()

# A generic (non-axis-aligned, mildly irregular) flat quad, z=0
coords = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.1, 0.0],
    [1.05, 1.0, 0.0],
    [-0.05, 0.95, 0.0],
])

print("=== 1. internal_force(0) == 0 ===")
f0 = elem.internal_force(coords, np.zeros(24), D)
print("max|f_int(0)| =", np.max(np.abs(f0)))
assert np.allclose(f0, 0.0, atol=1e-8)

print("\n=== 2. rigid tilt -> ~zero coupling force ===")
# uniform betax at all 4 nodes (theta_y_raw = alpha for every node),
# zero everything else -- a pure rigid tilt about the local y-axis.
alpha = 0.3  # rad, ~17 degrees, well into "large" territory
u_tilt = np.zeros(24)
for a in range(4):
    u_tilt[6 * a + 4] = alpha  # theta_y_raw(a) = betax(a) = alpha for all a
dof_local_tilt = u_tilt.copy()  # for this state _local_relative_dofs' raw theta assembly == u_tilt on these slots
f_add_tilt = elem._bending_membrane_coupling_force(coords, dof_local_tilt, D)
print("max|f_add(rigid tilt)| =", np.max(np.abs(f_add_tilt)))
assert np.max(np.abs(f_add_tilt)) < 1e-10, "rigid tilt should give exactly zero coupling force"

print("\n=== 3. genuine curvature -> nonzero coupling force ===")
u_curve = np.zeros(24)
x_local_approx = np.array([0.0, 1.0, 1.0, 0.0])  # rough per-node x used only to vary betax
for a in range(4):
    u_curve[6 * a + 4] = 0.3 * x_local_approx[a]  # betax varies node-to-node -> real curvature
f_add_curve = elem._bending_membrane_coupling_force(coords, u_curve, D)
print("max|f_add(curvature)| =", np.max(np.abs(f_add_curve)))
assert np.max(np.abs(f_add_curve)) > 1e-6, "genuine curvature should produce a nonzero coupling force"
print("nonzero membrane-row force present:", np.max(np.abs(f_add_curve[[0, 1, 6, 7, 12, 13, 18, 19]])) > 1e-6)

print("\n=== 4. analytic tangent vs complex-step, several states ===")
states = []
states.append(("zero", np.zeros(24)))
states.append(("small random", 1e-4 * np.random.randn(24)))
states.append(("large random", 0.2 * np.random.randn(24)))
states.append(("large bending-dominated",
                np.array([0, 0, 0, 0, 0, 0,
                           0, 0, 0.05, 0.3, 0.1, 0,
                           0, 0, 0.08, -0.2, 0.25, 0,
                           0, 0, 0.02, 0.1, -0.15, 0], dtype=float)))
u_rigid = np.zeros(24)
for a in range(4):
    u_rigid[6 * a + 3] = 0.5   # theta_x_raw
    u_rigid[6 * a + 4] = -0.3  # theta_y_raw
states.append(("rigid 3D-ish tilt (~30 deg)", u_rigid))

max_rel_err = 0.0
for name, u in states:
    K_analytic = elem.tangent_stiffness(coords, u, D)
    K_cs = elem._tangent_stiffness_complex_step(coords, u, D)
    denom = max(np.linalg.norm(K_cs), 1e-30)
    rel_err = np.linalg.norm(K_analytic - K_cs) / denom
    max_rel_err = max(max_rel_err, rel_err)
    print(f"  {name:32s}: rel_err = {rel_err:.3e}")
    assert rel_err < 1e-6, f"analytic tangent mismatch at state '{name}': {rel_err:.3e}"

print(f"\nmax rel err across all states: {max_rel_err:.3e}")

print("\n=== 5. u=0 tangent still matches Shell4MITC.stiffness() ===")
linear = Shell4MITC()
K_lin = linear.stiffness(coords, D)
K_coro0 = elem.stiffness(coords, D)
rel = np.linalg.norm(K_coro0 - K_lin) / np.linalg.norm(K_lin)
print(f"rel diff at u=0: {rel:.3e}")
assert rel < 1e-4  # matches tests/test_shell_corotational.py's own established tolerance

print("\nALL CHECKS PASSED")
