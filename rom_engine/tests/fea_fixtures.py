# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
fea_fixtures.py -- shared fea_engine-based test fixtures for
rom_engine's validation suite.

This is the ONLY place in rom_engine's test suite that imports
fea_engine directly -- rom_engine's own library code (pod.py,
galerkin.py, affine.py) never does (see package docstring), but
validating a ROM package against a made-up random matrix only proves
the linear algebra is self-consistent, not that it behaves correctly
on a REAL structural model. Every fixture here returns plain numpy
arrays (K, M, free-dof indices, ...), keeping the actual test files
just as FE-package-agnostic as rom_engine's library code -- fea_engine
is confined to this one file.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from fea_engine.material import EI_beam, Material, Section
from fea_engine.mesh import Mesh, MultiBlockMesh
from fea_engine.geometry import generate_mesh
from fea_engine.solver import FESystem
from fea_engine.damping import RayleighDamping
from fea_engine import elements
from fea_engine import nonlinear_solver as nls


def cantilever_beam_system(n=20, L=1.0, E=210e9, rho=7800.0, A=0.01, I=8.33e-6):
    """A fixed-free (cantilever) Euler-Bernoulli beam: n elements, n+1
    nodes, 2 dof/node [v, theta], node 0 fully fixed. Returns a dict
    with K, M (full (n_dof, n_dof), BEFORE any boundary condition is
    applied -- rom_engine's own modules never assume a particular BC
    convention, so tests apply free_dofs themselves), free_dofs (the
    indices of the 2n free dof after fixing node 0), and n_dof."""
    mesh = generate_mesh(dim=1, L=L, n=n)
    elem = elements.Beam2DEulerBernoulli()
    sysobj = FESystem(mesh, elem)
    mat = Material(E=E, nu=0.3, rho=rho)
    sec = Section(A=A, I=I)
    EI = EI_beam(mat, sec)
    sysobj.assemble_stiffness(EI)
    sysobj.assemble_mass(rho * A)
    sysobj.fix_dofs([0], [0, 1])
    return {
        "K": sysobj.K, "M": sysobj.M, "n_dof": sysobj.n_dof,
        "free_dofs": np.asarray(sysobj.free_dofs), "mesh": mesh,
        "sys": sysobj, "EI": EI, "rho_A": rho * A, "n_elem": n,
    }


def cantilever_static_snapshots(n=20, n_loads=6, L=1.0, seed=0, **kwargs):
    """n_loads independent static-load solutions of the SAME cantilever
    from cantilever_beam_system(), each a random nodal force pattern on
    the free dofs -- a snapshot set with a KNOWN generating mechanism
    (n_loads independent linear solves of the same K), useful for
    testing that POD recovers (at most) rank n_loads and that a
    Galerkin ROM built from these snapshots reproduces the FULL
    solution exactly for any load that is a linear combination of the
    ones used to build it.

    Returns (fixture_dict, snapshots (n_dof, n_loads), loads (n_dof,
    n_loads)) -- snapshots/loads are FULL (n_dof,) vectors, zero on the
    fixed dofs, so they can be used directly with rom_engine's basis
    machinery without the caller re-deriving the free/fixed split.
    """
    fx = cantilever_beam_system(n=n, L=L, **kwargs)
    rng = np.random.default_rng(seed)
    n_dof = fx["n_dof"]
    free = fx["free_dofs"]
    K_free = fx["K"][np.ix_(free, free)]

    loads = np.zeros((n_dof, n_loads))
    snaps = np.zeros((n_dof, n_loads))
    for i in range(n_loads):
        F_free = rng.standard_normal(len(free))
        loads[free, i] = F_free
        snaps[free, i] = np.linalg.solve(K_free, F_free)
    return fx, snaps, loads


def damped_cantilever_beam_system(n=20, L=1.0, E=210e9, rho=7800.0, A=0.01, I=8.33e-6,
                                   alpha=2.0, beta=1e-5):
    """The SAME cantilever as cantilever_beam_system(), plus a Rayleigh
    damping matrix C = alpha*M + beta*K built via fea_engine's own
    RayleighDamping + FESystem.assemble_damping() -- not hand-rolled
    here, so the ground truth this fixture is used to validate against
    (fx["sys"].solve_harmonic()/.solve_frequency_sweep(), fea_engine's
    own already-existing complex direct solvers, see solver.py) comes
    from the SAME code path a real fea_engine user would call, matching
    this project's standing practice of validating against a real
    model's own real solve, not a hand-built stand-in for one.

    Returns the same keys as cantilever_beam_system(), plus "C" (full
    (n_dof, n_dof) damping matrix), "damping" (the RayleighDamping
    instance), and "alpha"/"beta" (for building the algebraically-
    equivalent 2-term rom_engine.frequency proportional-damping
    decomposition and checking it matches the general 3-term one).
    """
    mesh = generate_mesh(dim=1, L=L, n=n)
    elem = elements.Beam2DEulerBernoulli()
    sysobj = FESystem(mesh, elem)
    mat = Material(E=E, nu=0.3, rho=rho)
    sec = Section(A=A, I=I)
    EI = EI_beam(mat, sec)
    sysobj.assemble_stiffness(EI)
    sysobj.assemble_mass(rho * A)
    damping = RayleighDamping(alpha=alpha, beta=beta)
    sysobj.assemble_damping(damping)
    sysobj.fix_dofs([0], [0, 1])
    return {
        "K": sysobj.K, "M": sysobj.M, "C": sysobj.C, "n_dof": sysobj.n_dof,
        "free_dofs": np.asarray(sysobj.free_dofs), "mesh": mesh,
        "sys": sysobj, "EI": EI, "rho_A": rho * A, "n_elem": n,
        "damping": damping, "alpha": alpha, "beta": beta,
    }


def reissner_cantilever_system(n_elem=6, L=1.0, E=210e9, nu=0.3, rho=7800.0,
                                A=1e-3, I=8.33e-7, kappa_s=1.0):
    """A fixed-free (cantilever) chain of the real, geometrically EXACT
    `fea_engine.elements.Beam2DReissner` rod (Wave 17 item 140), node 0
    fully clamped (u1=u2=theta=0). Used by
    rom_engine/tests/test_intrusive_nonlinear_rom.py as the real,
    genuinely nonlinear full-order model `IntrusiveNonlinearROM`
    (item 144) is validated against -- unlike
    `clamped_clamped_nonlinear_beam_system()` above (built on
    `Beam2DCorotational`, no independent rotation field / rotary
    inertia), this fixture exists specifically so item 144's own
    `internal_force_fn`/`tangent_fn` callbacks can be exercised through
    a SHEAR-DEFORMABLE element with a real rotary-inertia mass term, the
    element this whole Wave 17 track was built around.

    Returns a dict with the assembled linear stiffness/mass ("K", "M"),
    "mat" (the (E, G, A, I, kappa_s) tuple `internal_force()`/
    `tangent_stiffness()` expect), "free_dofs", "n_dof", "sys" (the
    live `FESystem`, so a caller can call
    `sys.assemble_internal_force(u, mat)`/`sys.assemble_tangent_stiffness(u, mat)`
    directly to build `internal_force_fn`/`tangent_fn` closures), and
    "n_elem"/"L" for building a matching `fea_engine.nonlinear_solver.
    solve_nonlinear_transient()` ground-truth run.
    """
    G = E / (2.0 * (1.0 + nu))
    x = np.linspace(0.0, L, n_elem + 1).reshape(-1, 1)
    nodes = np.hstack([x, np.zeros_like(x)])
    elem_conn = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elem_conn, dim=1)
    beam = elements.Beam2DReissner()
    fes = FESystem(mesh, beam)
    fes.fix_dofs([0], [0, 1, 2])   # clamp node 0: u1=u2=theta=0

    mat = (E, G, A, I, kappa_s)
    rho_A, rho_I = rho * A, rho * I
    fes.assemble_stiffness(mat)
    fes.assemble_mass((rho_A, rho_I))

    return {
        "K": fes.K, "M": fes.M, "n_dof": fes.n_dof,
        "free_dofs": np.asarray(fes.free_dofs), "mesh": mesh,
        "sys": fes, "mat": mat, "n_elem": n_elem, "L": L,
        "rho_A": rho_A, "rho_I": rho_I,
    }


def two_region_beam_components(n=20, L=1.0):
    """A cantilever beam split into two contiguous regions (elements
    [0:n//2) and [n//2:n)), each its own MultiBlockMesh block sharing
    the SAME element formulation (beam2d_euler_bernoulli) -- built
    specifically to exercise affine.py's two-region material-parameter
    use case via fea_engine's own, already-validated Module 14
    (mixed-element / per-block-material) assembly path.

    Returns (K1, K2, K_direct_func, free_dofs, n_dof): K1/K2 are the
    UNIT-EI (EI=1) stiffness contributions of region 1 / region 2 alone
    (the affine components); K_direct_func(EI1, EI2) independently
    reassembles the combined stiffness via fea_engine's own per-block
    assemble_stiffness(), for cross-checking affine.assemble(mu)
    against a real, independent reassembly rather than just against
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
        return sysobj

    sys_K1 = _assemble(1.0, 0.0)
    sys_K2 = _assemble(0.0, 1.0)
    sys_K1.fix_dofs([0], [0, 1])
    sys_K2.fix_dofs([0], [0, 1])

    def K_direct(EI1, EI2):
        s = _assemble(EI1, EI2)
        s.fix_dofs([0], [0, 1])
        return s.K

    return {
        "K1": sys_K1.K, "K2": sys_K2.K, "K_direct": K_direct,
        "free_dofs": np.asarray(sys_K1.free_dofs), "n_dof": sys_K1.n_dof,
        "mesh": mb_mesh,
    }


def clamped_clamped_nonlinear_beam_system(n_elem=10, L=1.0, E=210e9, rho=7850.0,
                                           A=1e-4, I=8.333e-9, n_modes=2,
                                           damping_alpha=0.0, damping_beta=0.0):
    """A clamped-clamped chain of the real, geometrically nonlinear
    fea_engine.elements.Beam2DCorotational element (fixed u=v=theta=0
    at BOTH end nodes, not just one -- the standard benchmark geometry
    for the nonlinear-surrogate-ROM literature this module targets,
    e.g. He et al. 2023's flat-beam benchmark), used by
    test_nonlinear_rom_fea.py as the real-model validation for
    nonlinear_rom.py's MultiFidelitySurrogate/PolynomialModalROM --
    unlike test_nonlinear_beam.py's own fixture (a CANTILEVER, fixed
    at node 0 only), which exists to validate the ELEMENT against the
    Euler elastica, not to train a reduced nonlinear force model.

    Returns a dict with the assembled linear stiffness/mass ("K", "M"),
    the retained-mode natural frequencies/mass-normalized mode shapes
    on the free dofs ("freq_hz", "V"), each mode's own peak (max-abs)
    free-dof shape component ("mode_shape_peaks" -- the generic,
    modes-may-be-of-any-type quantity `sampling.modal_force_samples`
    needs, not assumed to be a specific transverse-displacement dof),
    "free_dofs", "n_dof", "mat" (the (E, A, I) tuple
    internal_force()/tangent_stiffness() expect), and "fom_solver" --
    a ready-to-use callable(F_free) -> u_free wrapping fea_engine's
    OWN already-validated nonlinear_solver.solve_nonlinear_static(),
    exactly the `AppliedLoadStrategy.generate(..., fom_solver=...)`
    contract from nonlinear_rom.py's module docstring.

    `damping_alpha`/`damping_beta`: Rayleigh damping coefficients,
    default 0 (undamped) for Step 2's static-only use -- ALWAYS
    assembled (even at 0) via FESystem.assemble_damping() so `fes.C` is
    never None, satisfying nonlinear_solver.solve_nonlinear_transient()'s
    own requirement (Step 4's dynamic validation, test_nonlinear_dynamics_fea.py,
    passes nonzero values here; Step 2's static tests are unaffected by
    a zero damping matrix that nothing there reads).
    """
    x = np.linspace(0.0, L, n_elem + 1).reshape(-1, 1)
    nodes = np.hstack([x, np.zeros_like(x)])
    elem_conn = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elem_conn, dim=1)
    beam = elements.Beam2DCorotational()
    fes = FESystem(mesh, beam)
    fes.fix_dofs([0, n_elem], [0, 1, 2])   # BOTH ends clamped

    mat = (E, A, I)
    rho_A = rho * A
    fes.assemble_stiffness(mat)
    fes.assemble_mass(rho_A)
    fes.assemble_damping(RayleighDamping(alpha=damping_alpha, beta=damping_beta))

    freq_hz, mode_shapes = fes.solve_modal(n_modes=n_modes)
    free = fes.free_dofs
    V = mode_shapes[free, :]
    M_ff = fes.M[np.ix_(free, free)]
    C_ff = fes.C[np.ix_(free, free)]
    mode_shape_peaks = np.max(np.abs(V), axis=0)

    def fom_solver(F_free, n_steps=8, tol=1e-8, max_iter=40):
        """callable(F_free) -> u_free, per AppliedLoadStrategy's
        contract. Ramps the given free-dof force vector up over
        n_steps load increments (not a single Newton step) via
        fea_engine's own solve_nonlinear_static -- more robust
        convergence at the larger training loads than a single-shot
        load_factors=[1.0] call."""
        fes.F[:] = 0.0
        fes.F[free] = F_free
        _, U_hist = nls.solve_nonlinear_static(fes, mat, n_steps=n_steps, tol=tol, max_iter=max_iter)
        return U_hist[-1, free]

    return {
        "K": fes.K, "M": fes.M, "C": fes.C, "M_ff": M_ff, "C_ff": C_ff,
        "freq_hz": freq_hz, "V": V,
        "mode_shape_peaks": mode_shape_peaks, "free_dofs": np.asarray(free),
        "n_dof": fes.n_dof, "mat": mat, "fom_solver": fom_solver, "sys": fes,
        "n_elem": n_elem, "L": L,
    }
