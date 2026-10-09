# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_intrusive_nonlinear_rom_torch.py -- torch-backend addendum to Wave
17 item 144 (fea_engine/docs/consolidated_future_roadmap.md), validates
the `torch_reduced_matrices()`/`integrate_rk4_torch()`/
`integrate_newton_newmark_torch()` methods added to
`rom_engine.intrusive_nonlinear_rom.IntrusiveNonlinearROM` against this
same module's own existing, already-validated plain-NumPy methods.

A separate peer file, not appended to test_intrusive_nonlinear_rom.py --
matching this project's OWN established convention for torch additions
sitting beside an existing NumPy-only test file (see
rom_engine/tests/test_torch_linalg.py beside balanced_truncation.py's
own tests, and fea_engine/tests/test_torch_sparse_solver.py /
test_torch_autograd_tangent_stiffness.py beside their own NumPy
counterparts). Torch-gated on
`rom_engine.intrusive_nonlinear_rom._HAS_TORCH`, not a bare
`pytest.importorskip('torch')` -- same reasoning
fea_engine/autograd_tangent.py's own test file gives: importing this
module never requires torch, only actually calling a `torch_*`/
`*_torch` method does, and this project's sandbox has a torch PyPI
wheel that is installable but not importable (CUDA-linked, no CUDA
runtime -- see intrusive_nonlinear_rom.py's own "TORCH BACKEND" module
docstring section), so these tests skip themselves cleanly here and are
written to run for real on a torch-equipped machine.

Reuses the SAME `reissner_cantilever_system()` fixture
(`fea_fixtures.py`) test_intrusive_nonlinear_rom.py's own checks are
built on, rather than a new synthetic system -- so any agreement here
is on the real, genuinely nonlinear Beam2DReissner cantilever this
whole Wave 17 track targets, not a toy linear system.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import pytest

from rom_engine import IntrusiveNonlinearROM
from rom_engine.intrusive_nonlinear_rom import _HAS_TORCH
from fea_engine.damping import RayleighDamping

import fea_fixtures as ff


def _embed(u_free, free, n_dof):
    u = np.zeros(n_dof)
    u[free] = u_free
    return u


def _build_rom(n_elem=2, L=1.0):
    """A real, genuinely nonlinear IntrusiveNonlinearROM on an identity
    basis (V=eye(n_free)) -- same construction pattern
    test_intrusive_nonlinear_rom.py's own
    TestIdentityBasisReproducesFullOrder/TestEnergyConservationRK4FourthOrder
    fixtures use, reused here rather than re-derived."""
    fx = ff.reissner_cantilever_system(n_elem=n_elem, L=L)
    fes, mat, free = fx["sys"], fx["mat"], fx["free_dofs"]
    n_dof, n_free = fx["n_dof"], len(free)
    fes.assemble_damping(RayleighDamping(alpha=0.5, beta=1e-6))
    M_ff = fx["M"][np.ix_(free, free)]
    C_ff = fes.C[np.ix_(free, free)]

    def internal_force_fn(u_free):
        return fes.assemble_internal_force(_embed(u_free, free, n_dof), mat)[free]

    def tangent_fn(u_free):
        return fes.assemble_tangent_stiffness(_embed(u_free, free, n_dof), mat)[np.ix_(free, free)]

    def load_fn(t):
        return np.zeros(n_free)

    V = np.eye(n_free)
    rom = IntrusiveNonlinearROM(V, M_ff, C_ff, internal_force_fn, load_fn, tangent_fn=tangent_fn)
    return rom, fx, free, n_dof


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not importable in this environment")
class TestTorchReducedMatricesMatchNumpy:
    def test_Mr_Dr_Kr_match_numpy_to_near_machine_precision(self):
        import torch

        rom, _fx, _free, _n_dof = _build_rom(n_elem=4)
        M_r_t, D_r_t, K_r_t = rom.torch_reduced_matrices(device="cpu", dtype=torch.float64)

        for name, A_np, A_t in [("M_r", rom.M_r, M_r_t), ("D_r", rom.D_r, D_r_t), ("K_r", rom.K_r, K_r_t)]:
            A_t_np = A_t.detach().cpu().numpy()
            scale = max(np.max(np.abs(A_np)), 1.0)
            err = np.max(np.abs(A_t_np - A_np)) / scale
            print(f"torch_reduced_matrices() vs numpy {name}: max relative error {err:.3e}")
            assert err < 1e-12, f"{name} torch/numpy mismatch: {err:.3e}"

    def test_Mr_Dr_Kr_remain_non_diagonal(self):
        """The whole point of this class (see module docstring) is that
        M_r/D_r/K_r are NEVER mass-normalized/diagonalized -- confirm
        the torch path preserves this explicitly, on a basis that is
        genuinely non-mass-orthonormal (V=eye is trivially diagonal in
        M_r only if M itself happens to be diagonal, which a real FE
        mass matrix is not -- checked directly here rather than
        assumed)."""
        import torch

        rom, _fx, _free, _n_dof = _build_rom(n_elem=4)
        M_r_t, _D_r_t, K_r_t = rom.torch_reduced_matrices(device="cpu", dtype=torch.float64)

        M_r_np = M_r_t.detach().cpu().numpy()
        K_r_np = K_r_t.detach().cpu().numpy()
        off_diag_M = M_r_np - np.diag(np.diag(M_r_np))
        off_diag_K = K_r_np - np.diag(np.diag(K_r_np))
        assert np.max(np.abs(off_diag_M)) > 1e-8 * np.max(np.abs(M_r_np)), \
            "torch M_r came out (near-)diagonal -- basis was silently mass-normalized"
        assert np.max(np.abs(off_diag_K)) > 1e-8 * np.max(np.abs(K_r_np)), \
            "torch K_r came out (near-)diagonal -- unexpected for this fixture"


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not importable in this environment")
class TestTorchRK4MatchesNumpyRK4:
    def test_trajectory_matches_numpy_rk4_to_near_machine_precision(self):
        """Same integrator (classical RK4, both built on
        parameterized_latent_ode.rk4_step()), same math, only the
        tensor backend differs -- this should agree to near machine
        precision, not merely an approximate tolerance."""
        import torch

        rom, fx, free, n_dof = _build_rom(n_elem=2)
        fes, mat = fx["sys"], fx["mat"]

        tip_dof = 3 * fx["n_elem"] + 1
        fes.F[:] = 0.0
        fes.F[tip_dof] = 2.0e3
        from fea_engine import nonlinear_solver as nls
        _, U_hist = nls.solve_nonlinear_static(fes, mat, n_steps=6, tol=1e-10, max_iter=40)
        fes.F[:] = 0.0
        u0_free = U_hist[-1][free]

        q0, qdot0 = rom.initial_conditions(u0_full=u0_free, v0_full=np.zeros(len(free)))

        freq_hz_all, _ = fes.solve_modal(n_modes=len(free))
        dt = (1.0 / freq_hz_all[-1]) / 40.0
        n_steps = 50

        t_np, q_np, qdot_np = rom.integrate_rk4(q0, qdot0, dt, n_steps)
        t_t, q_t, qdot_t = rom.integrate_rk4_torch(q0, qdot0, dt, n_steps, device="cpu", dtype=torch.float64)

        assert np.allclose(t_np, t_t, atol=0.0, rtol=0.0)
        q_t_np = q_t.detach().cpu().numpy()
        qdot_t_np = qdot_t.detach().cpu().numpy()

        scale_q = max(np.max(np.abs(q_np)), 1e-30)
        scale_qdot = max(np.max(np.abs(qdot_np)), 1e-30)
        err_q = np.max(np.abs(q_t_np - q_np)) / scale_q
        err_qdot = np.max(np.abs(qdot_t_np - qdot_np)) / scale_qdot
        print(f"integrate_rk4_torch vs integrate_rk4: q max relative error {err_q:.3e}, "
              f"qdot max relative error {err_qdot:.3e}")
        assert err_q < 1e-10
        assert err_qdot < 1e-10


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not importable in this environment")
class TestTorchNewtonNewmarkMatchesNumpy:
    def test_final_state_matches_numpy_newton_newmark_to_near_machine_precision(self):
        """Same Newton-Newmark algebra (a0c..a7c/Keff/G/J, identical
        derivation), same backtracking control flow -- torch path vs
        numpy path on the IDENTICAL step sequence should agree tightly
        (not the looser RK4-vs-Newton-Newmark cross-integrator
        tolerance test_intrusive_nonlinear_rom.py's own
        TestNewtonNewmarkIntegrator uses -- that one compares two
        DIFFERENT integrators/step counts; this one compares the SAME
        integrator on two tensor backends)."""
        import torch

        rom, fx, free, n_dof = _build_rom(n_elem=2)
        fes, mat = fx["sys"], fx["mat"]

        tip_dof = 3 * fx["n_elem"] + 1
        fes.F[:] = 0.0
        fes.F[tip_dof] = 1.5e3
        from fea_engine import nonlinear_solver as nls
        # tol=1e-10 is tighter than fea_engine's own solve_nonlinear_static()
        # default (1e-8) and was found, on this specific load/mesh, to stall
        # just past the double-precision noise floor (|R| plateaus at
        # ~2.7e-08 after 40 iterations) rather than genuinely fail to
        # converge -- confirmed by the OTHER test in this file (RK4 vs
        # torch RK4, same tol=1e-10, different tip load) converging fine,
        # so this is a per-load-case numerical-noise-floor issue in the
        # TEST's own setup, not a solver or torch-backend defect. 1e-8
        # matches the library's own default and converges cleanly here.
        _, U_hist = nls.solve_nonlinear_static(fes, mat, n_steps=6, tol=1e-8, max_iter=40)
        fes.F[:] = 0.0
        u0_free = U_hist[-1][free]

        q0, qdot0 = rom.initial_conditions(u0_full=u0_free, v0_full=np.zeros(len(free)))

        freq_hz_all, _ = fes.solve_modal(n_modes=len(free))
        dt = (1.0 / freq_hz_all[0]) / 20.0
        n_steps = 20

        t_np, q_np, qdot_np = rom.integrate_newton_newmark(q0, qdot0, dt, n_steps)
        t_t, q_t, qdot_t = rom.integrate_newton_newmark_torch(
            q0, qdot0, dt, n_steps, device="cpu", dtype=torch.float64)

        q_t_np = q_t.detach().cpu().numpy()
        qdot_t_np = qdot_t.detach().cpu().numpy()

        scale_q = max(np.max(np.abs(q_np)), 1e-30)
        scale_qdot = max(np.max(np.abs(qdot_np)), 1e-30)
        err_q = np.max(np.abs(q_t_np - q_np)) / scale_q
        err_qdot = np.max(np.abs(qdot_t_np - qdot_np)) / scale_qdot
        print(f"integrate_newton_newmark_torch vs integrate_newton_newmark: "
              f"q max relative error {err_q:.3e}, qdot max relative error {err_qdot:.3e}")
        assert np.all(np.isfinite(q_t_np))
        assert err_q < 1e-8
        assert err_qdot < 1e-8


class TestTorchGateDoesNotBreakImport:
    """Runs UNCONDITIONALLY (no torch needed): confirms this whole test
    module, and rom_engine.intrusive_nonlinear_rom itself, import
    cleanly with no torch installed -- the actual regression risk this
    addition must not introduce, per the task's own "gated so `import
    rom_engine` and the rest of the test suite work fine without torch
    installed" requirement."""

    def test_module_imports_without_torch(self):
        from rom_engine.intrusive_nonlinear_rom import IntrusiveNonlinearROM, _HAS_TORCH
        assert hasattr(IntrusiveNonlinearROM, "integrate_rk4_torch")
        assert hasattr(IntrusiveNonlinearROM, "integrate_newton_newmark_torch")
        assert hasattr(IntrusiveNonlinearROM, "torch_reduced_matrices")
        assert isinstance(_HAS_TORCH, bool)

    def test_torch_methods_raise_clean_importerror_without_torch(self):
        from rom_engine.intrusive_nonlinear_rom import _HAS_TORCH
        if _HAS_TORCH:
            pytest.skip("torch IS importable in this environment -- this checks the no-torch error path only")
        rom, _fx, _free, _n_dof = _build_rom(n_elem=2)
        with pytest.raises(ImportError):
            rom.torch_reduced_matrices()
        with pytest.raises(ImportError):
            rom.integrate_rk4_torch(np.zeros(rom.n_modes), np.zeros(rom.n_modes), 1e-4, 2)
        with pytest.raises(ImportError):
            rom.integrate_newton_newmark_torch(np.zeros(rom.n_modes), np.zeros(rom.n_modes), 1e-4, 2)
