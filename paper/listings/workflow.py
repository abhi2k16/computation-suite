# workflow.py -- continues listing 1 (names fe, mesh, tn, y, basis, rom, U, tip come from quickstart.py).
from rom_engine import validate_rom

f_hz, modes = fe.solve_modal(n_modes=3)               # natural frequencies [Hz]
vm = fe.von_mises(U)                                  # scalar FEField of nodal von Mises stress
vm_max = float(np.max(vm))

def traction(a):                                      # held-out load: a * sin(pi*y) on the tip edge, y-direction
    F = np.zeros(fe.n_dof); F[2 * tn + 1] = a * np.sin(np.pi * y)
    return F

fom = lambda a: solve_static_batched(fe, traction(a)[:, None])[:, 0]       # full-order solve
red = lambda a: basis.expand(rom.solve_static(traction(a))[1])             # reduced solve, expanded
report = validate_rom(red, fom, [1.0e4, -2.0e4, 5.0e4], tol=1e-2)          # errors, timing, pass/fail
workflow = {"f_hz": [float(f) for f in f_hz], "vm_max": vm_max, "max_error": float(report.max),
            "passed": bool(report.passed)}
