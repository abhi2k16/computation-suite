# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_random_vibration.py -- validates rom_engine.random_vibration.psd_response()
against a REAL, damped fea_engine cantilever beam
(fea_fixtures.damped_cantilever_beam_system()), using fea_engine's own
FESystem.solve_random_vibration() (already-validated, already-shipped
full-order functionality) as ground truth throughout -- the same
fixture and ground-truth convention test_frequency.py already uses for
FrequencyROM itself, since psd_response() is built directly on top of
it (see docs/frequency_domain_rom_roadmap.md Phase 6).

Checks:
  1. Off resonance: S_out/sigma_out match fea_engine's own full-order
     solve_random_vibration() closely (the easy case).
  2. Near/through a resonance: same check, on a PSD band that straddles
     the first natural frequency -- the hard case, where |H(f)|^2 peaks
     sharply and the trapezoidal frequency-grid integration is most
     sensitive.
  3. Multi-output-DOF generalization (beyond fea_engine's own
     single-output_dof signature): a 2-DOF call reproduces the SAME
     per-DOF numbers a separate single-DOF call would give, checked
     both against each other and against fea_engine's own full-order
     result for each DOF individually.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import eigh
from rom_engine import FrequencyROM
from rom_engine.random_vibration import psd_response
import fea_fixtures as ff


def _modal_basis(Kff, Mff, n_modes):
    eigvals, eigvecs = eigh(Kff, Mff)
    return eigvecs[:, :n_modes]


def _build_rom_and_fixture(n=20, n_modes=12):
    fx = ff.damped_cantilever_beam_system(n=n)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    basis = _modal_basis(Kff, Mff, n_modes)
    rom = FrequencyROM.from_MCK(Mff, Kff, basis, rayleigh=(fx["alpha"], fx["beta"]))

    eigvals, _ = eigh(Kff, Mff)
    omega1_hz = float(np.sqrt(eigvals[0])) / (2 * np.pi)
    return fx, rom, free, omega1_hz


def _full_order_psd(fx, free, freqs_hz, psd_input, F0_full, output_dof):
    sysobj = fx["sys"]
    return sysobj.solve_random_vibration(freqs_hz, psd_input, F0_full, output_dof)


def test_off_resonance_matches_full_order():
    fx, rom, free, omega1_hz = _build_rom_and_fixture()
    n_dof = fx["n_dof"]
    tip_dof = n_dof - 2
    F0_full = np.zeros(n_dof); F0_full[tip_dof] = 1000.0
    F0_free = F0_full[free]

    freqs_hz = np.linspace(0.05 * omega1_hz, 0.5 * omega1_hz, 40)
    psd_input = np.ones_like(freqs_hz)   # flat (white) input PSD

    tip_local = np.where(free == tip_dof)[0][0]
    S_out_true, sigma_true = _full_order_psd(fx, free, freqs_hz, psd_input, F0_full, tip_dof)
    S_out_rom, sigma_rom = psd_response(rom, freqs_hz, psd_input, F0_free, tip_local)

    rel_err_S = np.max(np.abs(S_out_rom - S_out_true) / np.max(S_out_true))
    rel_err_sigma = abs(sigma_rom - sigma_true) / sigma_true
    print(f"off-resonance: max relative S_out error={rel_err_S:.3e}, sigma relative error={rel_err_sigma:.3e}")
    assert rel_err_S < 1e-3
    assert rel_err_sigma < 1e-3


def test_near_resonance_matches_full_order():
    fx, rom, free, omega1_hz = _build_rom_and_fixture()
    n_dof = fx["n_dof"]
    tip_dof = n_dof - 2
    F0_full = np.zeros(n_dof); F0_full[tip_dof] = 1000.0
    F0_free = F0_full[free]
    tip_local = np.where(free == tip_dof)[0][0]

    # a band straddling the first natural frequency -- the hard case
    freqs_hz = np.linspace(0.5 * omega1_hz, 1.5 * omega1_hz, 60)
    psd_input = np.ones_like(freqs_hz)

    S_out_true, sigma_true = _full_order_psd(fx, free, freqs_hz, psd_input, F0_full, tip_dof)
    S_out_rom, sigma_rom = psd_response(rom, freqs_hz, psd_input, F0_free, tip_local)

    rel_err_S = np.max(np.abs(S_out_rom - S_out_true) / np.max(S_out_true))
    rel_err_sigma = abs(sigma_rom - sigma_true) / sigma_true
    print(f"near resonance: max relative S_out error={rel_err_S:.3e}, sigma relative error={rel_err_sigma:.3e}")
    assert rel_err_S < 1e-2
    assert rel_err_sigma < 1e-2


def test_multi_output_dofs_generalization():
    fx, rom, free, omega1_hz = _build_rom_and_fixture()
    n_dof = fx["n_dof"]
    tip_dof = n_dof - 2
    mid_dof = free[len(free) // 2]
    F0_full = np.zeros(n_dof); F0_full[tip_dof] = 1000.0
    F0_free = F0_full[free]
    tip_local = np.where(free == tip_dof)[0][0]
    mid_local = np.where(free == mid_dof)[0][0]

    freqs_hz = np.linspace(0.05 * omega1_hz, 1.5 * omega1_hz, 50)
    psd_input = np.ones_like(freqs_hz)

    # single-DOF calls
    S_tip, sigma_tip = psd_response(rom, freqs_hz, psd_input, F0_free, tip_local)
    S_mid, sigma_mid = psd_response(rom, freqs_hz, psd_input, F0_free, mid_local)

    # one multi-DOF call
    S_multi, sigma_multi = psd_response(rom, freqs_hz, psd_input, F0_free, [tip_local, mid_local])

    assert S_multi.shape == (len(freqs_hz), 2)
    assert sigma_multi.shape == (2,)
    np.testing.assert_allclose(S_multi[:, 0], S_tip, rtol=1e-10)
    np.testing.assert_allclose(S_multi[:, 1], S_mid, rtol=1e-10)
    np.testing.assert_allclose(sigma_multi, [sigma_tip, sigma_mid], rtol=1e-10)

    # and each column still matches fea_engine's own full-order result
    S_tip_true, sigma_tip_true = _full_order_psd(fx, free, freqs_hz, psd_input, F0_full, tip_dof)
    S_mid_true, sigma_mid_true = _full_order_psd(fx, free, freqs_hz, psd_input, F0_full, mid_dof)
    assert abs(sigma_multi[0] - sigma_tip_true) / sigma_tip_true < 1e-2
    assert abs(sigma_multi[1] - sigma_mid_true) / sigma_mid_true < 1e-2
    print("PASS -- multi-output-DOF call matches both individual single-DOF ROM calls "
          "and fea_engine's own full-order result per DOF")
