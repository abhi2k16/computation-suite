# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_state_space.py -- validates rom_engine.state_space.to_state_space()
against (1) a hand-computed single-DOF mass-spring-damper, where the
first-order poles have a known closed form, and (2) a REAL, Rayleigh-
damped fea_engine cantilever beam (fea_fixtures.damped_cantilever_beam_system()),
where the poles must reproduce the fixture's own known undamped natural
frequencies and Rayleigh damping ratios.

Checks (mirroring docs/classical_mor_roadmap.md Section 6):
  1. form="E" and form="A" give ALGEBRAICALLY EQUIVALENT poles (E-form:
     generalized eig(A, E); A-form: ordinary eig(A)) for the same model
     -- the two representations of the module docstring are the same
     system, not two different ones.
  2. Those poles match the single-DOF analytical solution exactly.
  3. Those poles match the multi-DOF fixture's own known
     omega_n/zeta_n (from Rayleigh damping) to numerical-solve precision.
  4. B/Cout selection actually behaves like a selection (picking the
     tip DOF out of a bigger model returns just that DOF's row/column).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
from scipy.linalg import eig, eigh
from rom_engine.state_space import to_state_space
import fea_fixtures as ff


def _sorted_poles(eigvals):
    """Keep only the finite poles (E-form generalized eig can return
    inf for a singular-pencil direction, none expected here since M is
    full rank, but guard anyway), sorted by imaginary part ascending so
    complex-conjugate pairs line up predictably for comparison."""
    eigvals = eigvals[np.isfinite(eigvals)]
    return eigvals[np.argsort(eigvals.imag)]


def test_single_dof_analytical_poles():
    m, k, c = 1.0, 100.0, 0.4   # underdamped: omega_n=10, zeta=0.02
    M = np.array([[m]])
    K = np.array([[k]])
    C = np.array([[c]])

    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))
    s_expected = np.array([
        -zeta * omega_n - 1j * omega_n * np.sqrt(1 - zeta**2),
        -zeta * omega_n + 1j * omega_n * np.sqrt(1 - zeta**2),
    ])
    s_expected = _sorted_poles(s_expected)

    sys_E = to_state_space(M, K, C=C, form="E")
    poles_E = _sorted_poles(eig(sys_E.A, sys_E.E, right=False))
    assert sys_E.E is not None
    np.testing.assert_allclose(poles_E, s_expected, rtol=1e-10)

    sys_A = to_state_space(M, K, C=C, form="A")
    poles_A = _sorted_poles(eig(sys_A.A, right=False))
    assert sys_A.E is None
    np.testing.assert_allclose(poles_A, s_expected, rtol=1e-10)

    # form="E" and form="A" are two representations of the same system
    np.testing.assert_allclose(poles_E, poles_A, rtol=1e-10)


def test_fixture_poles_match_known_modal_frequencies_and_rayleigh_damping():
    fx = ff.damped_cantilever_beam_system(n=15)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    Cff = fx["C"][np.ix_(free, free)]
    alpha, beta = fx["alpha"], fx["beta"]

    n_free = len(free)
    omega_undamped, _ = eigh(Kff, Mff)
    omega_undamped = np.sqrt(np.clip(omega_undamped, 0, None))

    n_check = 6   # lowest modes -- least sensitive to mesh/ordering noise
    zeta = 0.5 * (alpha / omega_undamped[:n_check] + beta * omega_undamped[:n_check])
    s_expected = -zeta * omega_undamped[:n_check] + 1j * omega_undamped[:n_check] * np.sqrt(1 - zeta**2)
    s_expected_conj = np.conj(s_expected)
    expected_set = np.concatenate([s_expected, s_expected_conj])

    sys_E = to_state_space(Mff, Kff, C=Cff, form="E")
    poles_E = eig(sys_E.A, sys_E.E, right=False)
    poles_E = poles_E[np.isfinite(poles_E)]

    # match each expected pole to its nearest computed pole (order isn't
    # guaranteed to line up with the undamped-frequency ordering once
    # damping mixes things slightly) and check the residual is small
    for s in expected_set:
        closest = poles_E[np.argmin(np.abs(poles_E - s))]
        assert abs(closest - s) < 1e-6 * (1 + abs(s)), \
            f"no computed pole near expected {s}, closest was {closest}"

    sys_A = to_state_space(Mff, Kff, C=Cff, form="A")
    poles_A = eig(sys_A.A, right=False)
    for s in expected_set:
        closest = poles_A[np.argmin(np.abs(poles_A - s))]
        assert abs(closest - s) < 1e-6 * (1 + abs(s)), \
            f"[form=A] no computed pole near expected {s}, closest was {closest}"


def test_input_output_maps_select_the_right_dofs():
    fx = ff.damped_cantilever_beam_system(n=10)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    n_free = len(free)

    tip_local = n_free - 2   # tip transverse dof, free-dof-local numbering
    B = np.zeros((n_free, 1)); B[tip_local, 0] = 1.0
    Cout = np.zeros((1, n_free)); Cout[0, tip_local] = 1.0

    sys = to_state_space(Mff, Kff, B=B, Cout=Cout, form="E")
    assert sys.n_in == 1
    assert sys.n_out == 1
    assert sys.n_dof == n_free
    assert sys.n_state == 2 * n_free
    # B_full is [0; B]: nonzero only in the velocity-equation block, at
    # the tip row
    assert np.count_nonzero(sys.B) == 1
    assert sys.B[n_free + tip_local, 0] == 1.0
    # Cout_full is [Cout, 0]: nonzero only in the position block
    assert np.count_nonzero(sys.Cout) == 1
    assert sys.Cout[0, tip_local] == 1.0

    # a 1-D B/Cout (bare selection vectors) should behave the same as
    # their explicit 2-D column/row form
    B_1d = np.zeros(n_free); B_1d[tip_local] = 1.0
    Cout_1d = np.zeros(n_free); Cout_1d[tip_local] = 1.0
    sys_1d = to_state_space(Mff, Kff, B=B_1d, Cout=Cout_1d, form="E")
    np.testing.assert_array_equal(sys_1d.B, sys.B)
    np.testing.assert_array_equal(sys_1d.Cout, sys.Cout)


def test_default_B_Cout_is_identity_every_dof():
    fx = ff.damped_cantilever_beam_system(n=6)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    n_free = len(free)

    sys = to_state_space(Mff, Kff, form="E")
    assert sys.n_in == n_free
    assert sys.n_out == n_free
