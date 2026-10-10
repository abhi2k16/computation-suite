# quickstart.py -- the listing shown in the article (executed by experiments/e7_quickstart.py).
import numpy as np
from fea_engine import Material, D_plane_stress, Quad4PlaneStress, FESystem
from fea_engine.batched_solve import solve_static_batched
from fea_engine.mesh import rectangle_mesh
from rom_engine import PodBasis, GalerkinROM

mat = Material(E=2.1e11, nu=0.3, rho=7850.0)
mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=24, ny=12)
fe = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
fe.assemble_stiffness(D_plane_stress(mat), thickness=0.02)
fe.assemble_mass(mat.rho * np.eye(2), thickness=0.02)
mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=0.4, name="tip")
fe.fix_dofs("root", ["ux", "uy"])
fe.add_nodal_force("tip", "uy", -2.0e4)
U = fe.solve_static()                               # FEField with named components
tip = U.component("uy", nodes="tip").mean()

# snapshot loads: polynomial traction shapes on the tip edge, solved with one factorisation
tn = mesh.select_nodes(x=0.4)
y = mesh.nodes[tn, 1] / 0.2
loads = np.zeros((fe.n_dof, 8))
for j, (d, k) in enumerate([(d, k) for d in (0, 1) for k in range(4)]):
    loads[2 * tn + d, j] = y ** k
snaps = solve_static_batched(fe, loads)
basis = PodBasis().fit(snaps, n_modes=8, M=fe.M)    # mass-weighted POD
rom = GalerkinROM(basis).reduce_system(fe.K, M=fe.M, F=fe.F)
x_rom, q = rom.solve_static()                       # 8 unknowns instead of 624 free DOF
tip_rom = np.asarray(x_rom).reshape(-1, 2)[tn, 1].mean()
result = {"n_dof": int(fe.n_dof), "n_free": len(fe.free_dofs), "n_modes": int(basis.n_modes),
          "tip_fom": float(tip), "tip_rom": float(tip_rom), "rel_error": float(abs(tip_rom - tip) / abs(tip))}
