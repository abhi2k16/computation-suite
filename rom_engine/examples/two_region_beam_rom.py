"""
two_region_beam_rom.py -- end-to-end rom_engine example.

Pipeline demonstrated:
    fea_engine full-order model  -->  POD  -->  Galerkin ROM
                                  -->  affine parametric decomposition
                                  -->  fast many-query parameter sweep

Physical setup: a clamped-free (cantilever) beam, split into two
material regions along its length with INDEPENDENT bending
rigidities EI_1 (first half) and EI_2 (second half) -- e.g. a beam
that's been locally reinforced or damaged over part of its span. This
is a genuinely two-parameter problem: for a design study, a damage
assessment, or an uncertainty-quantification run, you need the tip
deflection under a fixed load at MANY (EI_1, EI_2) combinations.

This is exactly the situation rom_engine's linear core is built for:

  1. Run a handful of full-order solves offline to build a POD basis
     that captures how the beam actually deforms (pod.py).
  2. Galerkin-project the two unit-rigidity stiffness contributions
     K_1, K_2 onto that basis ONCE (galerkin.py + affine.py's
     project() step).
  3. Answer the ENTIRE parameter sweep -- thousands of (EI_1, EI_2)
     queries -- using only tiny reduced solves (affine.py's
     assemble_reduced() + a reduced linear solve), each orders of
     magnitude cheaper than a fresh full finite element solve.

Every ROM answer in the sweep is cross-checked against a real
fea_engine full-order solve so the printed accuracy numbers are
genuine, not illustrative.
"""
import time
import numpy as np

from fea_engine.material import EI_beam, Material, Section
from fea_engine.mesh import MultiBlockMesh
from fea_engine.geometry import generate_mesh
from fea_engine.solver import FESystem
from fea_engine import elements

from rom_engine import PodBasis, GalerkinROM, AffineDecomposition


# ---------------------------------------------------------------------
# 1. Build the two-region full-order model (fea_engine)
# ---------------------------------------------------------------------
def build_two_region_beam(n=120, L=1.0):
    """A cantilever beam of n elements, split into two equal-length
    regions, each independently parametrized by its own bending
    rigidity. Returns the unit-rigidity component matrices K1, K2 (so
    that K(EI1, EI2) = EI1*K1 + EI2*K2 exactly -- ordinary linear
    elasticity is always affine in a scaling modulus/rigidity), the
    mass matrix (rigidity-independent, so no affine decomposition
    needed for it), and a K_direct(EI1, EI2) helper that reassembles
    the model from scratch via fea_engine's own per-block assembly --
    used below purely to VALIDATE the ROM against, not by the ROM
    itself.
    """
    mesh = generate_mesh(dim=1, L=L, n=n)
    half = n // 2
    blocks = {"region1": mesh.elements[:half], "region2": mesh.elements[half:]}
    mb_mesh = MultiBlockMesh(nodes=mesh.nodes, blocks=blocks, dim=mesh.dim)
    elem = elements.Beam2DEulerBernoulli()
    elem_map = {"region1": elem, "region2": elem}

    def _assemble(EI1, EI2):
        sysobj = FESystem(mb_mesh, elem_map)
        sysobj.assemble_stiffness({"region1": EI1, "region2": EI2})
        sysobj.fix_dofs([0], [0, 1])
        return sysobj

    sys1 = _assemble(1.0, 0.0)   # K1: region-1 unit-rigidity contribution
    sys2 = _assemble(0.0, 1.0)   # K2: region-2 unit-rigidity contribution

    mat = Material(E=1.0, nu=0.3, rho=7800.0)   # E folded into EI1/EI2 directly below
    sec = Section(A=0.01, I=1.0)
    rho_A = 7800.0 * 0.01
    sys_mass = FESystem(mb_mesh, elem_map)
    sys_mass.assemble_mass(rho_A)
    sys_mass.fix_dofs([0], [0, 1])

    free_dofs = np.asarray(sys1.free_dofs)

    def K_direct(EI1, EI2):
        return _assemble(EI1, EI2).K

    return {
        "K1": sys1.K, "K2": sys2.K, "M": sys_mass.M,
        "free_dofs": free_dofs, "n_dof": sys1.n_dof,
        "K_direct": K_direct, "mesh": mb_mesh,
    }


# ---------------------------------------------------------------------
# 2. Offline stage: training snapshots -> POD basis -> Galerkin/affine
#    projection
# ---------------------------------------------------------------------
def build_rom(model, n_train=10, n_modes=10, tip_load=-1000.0, seed=0):
    """Collect static-solve snapshots at TRAINING (EI1, EI2) draws
    covering the parameter range of interest, each solved with the
    SAME representative tip load (the load pattern that matters for
    this study) plus a touch of random load variation so the basis
    isn't overfit to one exact load shape. Build a POD basis from
    those snapshots, then project the affine components onto it.
    """
    free = model["free_dofs"]
    n_dof = model["n_dof"]
    K1, K2 = model["K1"], model["K2"]
    K1_ff = K1[np.ix_(free, free)]
    K2_ff = K2[np.ix_(free, free)]

    rng = np.random.default_rng(seed)
    snaps = np.zeros((n_dof, n_train))
    # tip transverse dof is the last free dof's translational component;
    # free_dofs alternate [v, theta] per node after node 0, so the tip
    # deflection dof is free[-2]
    tip_dof_local = len(free) - 2
    for i in range(n_train):
        EI1 = rng.uniform(0.5, 5.0)
        EI2 = rng.uniform(0.5, 5.0)
        K_ff = EI1 * K1_ff + EI2 * K2_ff
        F_free = np.zeros(len(free))
        F_free[tip_dof_local] = tip_load
        F_free += rng.normal(scale=abs(tip_load) * 0.02, size=len(free))  # small variety
        snaps[free, i] = np.linalg.solve(K_ff, F_free)

    basis = PodBasis().fit(snaps, n_modes=min(n_modes, n_train))
    print(f"POD basis: {basis.n_modes} modes, "
          f"energy captured = {basis.energy_captured():.6f}")

    galerkin = GalerkinROM(basis)
    affine = AffineDecomposition([K1, K2], lambda mu: [mu[0], mu[1]]).project(basis.V)

    return basis, galerkin, affine, tip_dof_local


# ---------------------------------------------------------------------
# 3. Online stage: fast parameter sweep + accuracy/timing check
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("rom_engine end-to-end example: two-region cantilever beam")
    print("=" * 70)

    model = build_two_region_beam(n=120)
    free = model["free_dofs"]
    n_dof = model["n_dof"]
    tip_load = -1000.0

    basis, galerkin, affine, tip_dof_local = build_rom(
        model, n_train=10, n_modes=10, tip_load=tip_load, seed=0)

    F_free = np.zeros(len(free))
    F_free[tip_dof_local] = tip_load
    F_full = np.zeros(n_dof)
    F_full[free] = F_free
    F_r = galerkin.project_vector(F_full)

    # ---- accuracy check across held-out (EI1, EI2) test points -------
    rng = np.random.default_rng(99)
    n_test = 8
    print(f"\nAccuracy check ({n_test} held-out (EI1, EI2) points, "
          f"tip load = {tip_load:.0f} N):")
    print(f"{'EI1':>7} {'EI2':>7} {'ROM tip (m)':>14} {'FOM tip (m)':>14} {'rel. err':>10}")
    rel_errs = []
    for _ in range(n_test):
        EI1, EI2 = rng.uniform(0.5, 5.0, size=2)

        K_r = affine.assemble_reduced((EI1, EI2))
        q = np.linalg.solve(K_r, F_r)
        x_rom_free = (basis.V @ q)[free]

        K_fom_ff = model["K_direct"](EI1, EI2)[np.ix_(free, free)]
        x_fom_free = np.linalg.solve(K_fom_ff, F_free)

        tip_rom = x_rom_free[tip_dof_local]
        tip_fom = x_fom_free[tip_dof_local]
        rel_err = abs(tip_rom - tip_fom) / abs(tip_fom)
        rel_errs.append(rel_err)
        print(f"{EI1:7.3f} {EI2:7.3f} {tip_rom:14.6e} {tip_fom:14.6e} {rel_err:10.2e}")

    print(f"\nmax relative tip-deflection error over {n_test} held-out points: "
          f"{max(rel_errs):.2e}")

    # ---- timing: full parameter sweep, ROM vs full re-assembly -------
    n_sweep = 500
    mus = [tuple(rng.uniform(0.5, 5.0, size=2)) for _ in range(n_sweep)]

    t0 = time.perf_counter()
    for EI1, EI2 in mus:
        K_r = affine.assemble_reduced((EI1, EI2))
        np.linalg.solve(K_r, F_r)
    t_rom = time.perf_counter() - t0

    t0 = time.perf_counter()
    for EI1, EI2 in mus:
        K_fom_ff = model["K_direct"](EI1, EI2)[np.ix_(free, free)]
        np.linalg.solve(K_fom_ff, F_free)
    t_fom = time.perf_counter() - t0

    print(f"\nTiming, {n_sweep}-point sweep over (EI1, EI2):")
    print(f"  full-order (fea_engine reassembly + solve): {t_fom*1e3:8.2f} ms "
          f"({t_fom/n_sweep*1e3:.4f} ms/query)")
    print(f"  reduced-order (affine + Galerkin solve):     {t_rom*1e3:8.2f} ms "
          f"({t_rom/n_sweep*1e3:.4f} ms/query)")
    print(f"  speedup: {t_fom / t_rom:.1f}x")

    print("\n" + "=" * 70)
    print("Done. This is the pattern for any linear, affinely-parametrized")
    print("structural model: build once (POD + affine projection), then")
    print("query as many times as a design study, sweep, or UQ run needs.")
    print("=" * 70)

    # Plot FEA and ROM tip deflection against the rigidity ratio.
    import matplotlib.pyplot as plt
    rigidity_ratios = np.linspace(0.1, 10.0, 20)
    fixed_EI2s = (0.5, 1.0, 2.0)
    fig, ax = plt.subplots()
    for EI2 in fixed_EI2s:
        tip_roms = []
        tip_foms = []
        for ratio in rigidity_ratios:
            EI1 = ratio * EI2
            K_r = affine.assemble_reduced((EI1, EI2))
            q = np.linalg.solve(K_r, F_r)
            x_rom_free = (basis.V @ q)[free]
            tip_roms.append(x_rom_free[tip_dof_local])

            K_fom_ff = model["K_direct"](EI1, EI2)[np.ix_(free, free)]
            x_fom_free = np.linalg.solve(K_fom_ff, F_free)
            tip_foms.append(x_fom_free[tip_dof_local])

        ax.plot(rigidity_ratios, tip_foms, "o-", label=f"FEA, EI2={EI2:g}")
        ax.plot(rigidity_ratios, tip_roms, "--", label=f"ROM, EI2={EI2:g}")

    ax.set_xlabel("EI1 / EI2")
    ax.set_ylabel("Tip deflection (m)")
    ax.set_title("FEA versus ROM tip deflection")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
