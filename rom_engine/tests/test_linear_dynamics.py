"""linear_dynamics.py: closed-form SDOF checks, method cross-checks, and a real fea_engine beam."""
__author__ = "Abhijeet"
import numpy as np
import pytest

from rom_engine.linear_dynamics import newmark_linear, modal_superposition, piecewise_linear_exact, galerkin_transient
from rom_engine import PodBasis, GalerkinROM
from fea_fixtures import damped_cantilever_beam_system


def test_newmark_undamped_free_vibration_matches_cosine():
    w = 2 * np.pi * 5.0
    t, q, v, a = newmark_linear([[1.0]], None, [[w**2]], np.zeros(1), 1e-3, 1000, q0=[1.0], v0=[0.0])
    assert np.max(np.abs(q[:, 0] - np.cos(w * t))) < 5e-3          # phase error accumulated over 5 periods at dt = T/200
    t2, q2, *_ = newmark_linear([[1.0]], None, [[w**2]], np.zeros(1), 5e-4, 2000, q0=[1.0], v0=[0.0])
    e1 = np.max(np.abs(q[:, 0] - np.cos(w * t))); e2 = np.max(np.abs(q2[:, 0] - np.cos(w * t2)))
    assert 3.0 < e1 / e2 < 5.0                                      # halving dt divides the error by ~4


def test_newmark_step_load_static_limit_and_overshoot():
    k, m, c = 100.0, 1.0, 0.0
    t, q, *_ = newmark_linear([[m]], [[c]], [[k]], np.array([10.0]), 1e-3, 2000)
    assert q[:, 0].max() == pytest.approx(2 * 10.0 / k, rel=2e-3)   # undamped step response peaks at 2*F/k


def test_exact_propagator_has_no_integration_error_for_linear_load():
    # x'' + 2 z w x' + w^2 x = u(t) with u = ramp: compare against a very fine Newmark reference
    w, z = 3.0, 0.1
    A = np.array([[0, 1], [-w**2, -2 * z * w]]); B = np.array([[0.0], [1.0]])
    dt, n = 0.2, 50
    t = np.arange(n + 1) * dt
    U = t[:, None] * 0.5
    X = piecewise_linear_exact(A, B, U, dt, np.zeros(2))
    ref_t, ref_q, *_ = newmark_linear([[1.0]], [[2 * z * w]], [[w**2]], lambda s: np.array([0.5 * s]), dt / 200, n * 200)
    assert np.max(np.abs(X[:, 0] - ref_q[::200, 0])) < 1e-6


def test_modal_superposition_matches_sdof_closed_form_free_decay():
    w, z = 2 * np.pi * 4.0, 0.05
    t, u, info = modal_superposition([[w**2]], [[1.0]], np.zeros(1), 0.01, 400, 1, zeta=z, u0=[1.0], v0=[0.0])
    wd = w * np.sqrt(1 - z**2)
    ref = np.exp(-z * w * t) * (np.cos(wd * t) + z * w / wd * np.sin(wd * t))
    assert np.max(np.abs(u[:, 0] - ref)) < 1e-9                    # exact, despite the coarse dt


@pytest.fixture(scope="module")
def beam():
    d = damped_cantilever_beam_system(n=20, alpha=2.0, beta=1e-5)
    # The fixture's K/M/C are the UNCONSTRAINED beam: K is singular (rigid-body modes whose
    # eigenvalues are pure round-off and differ between LAPACK builds). Use the clamped block.
    fd = np.asarray(d["free_dofs"])
    ix = np.ix_(fd, fd)
    return {"K": d["K"][ix], "M": d["M"][ix], "C": d["C"][ix]}


def test_methods_agree_on_real_beam(beam):
    K, M, C = beam["K"], beam["M"], beam["C"]
    n = K.shape[0]
    f = np.zeros(n); f[-2] = 1000.0
    dt, steps = 2e-4, 400
    load = lambda t: f * np.sin(2 * np.pi * 60 * t)
    t, qn, *_ = newmark_linear(M, C, K, load, dt, steps)
    tm, qm, info = modal_superposition(K, M, load, dt, steps, n_modes=n, rayleigh=(2.0, 1e-5))
    e1 = np.linalg.norm(qn - qm) / np.linalg.norm(qm)
    assert e1 < 2e-2                                                # all modes: only the Newmark time error remains
    # halving dt must cut that error ~4x (second order); the exact modal reference does not depend on dt
    t2, qn2, *_ = newmark_linear(M, C, K, load, dt / 2, 2 * steps)
    _, qm2f, _ = modal_superposition(K, M, load, dt / 2, 2 * steps, n_modes=n, rayleigh=(2.0, 1e-5))
    e2 = np.linalg.norm(qn2 - qm2f) / np.linalg.norm(qm2f)
    assert 3.0 < e1 / e2 < 5.0
    tm2, qm2, _ = modal_superposition(K, M, load, dt, steps, n_modes=6, rayleigh=(2.0, 1e-5))
    assert np.linalg.norm(qm2 - qm) / np.linalg.norm(qm) < 5e-2   # truncation to 6 modes stays close


def test_galerkin_transient_matches_full_order(beam):
    K, M, C = beam["K"], beam["M"], beam["C"]
    n = K.shape[0]
    f = np.zeros(n); f[-2] = 1000.0
    dt, steps = 2e-4, 400
    load = lambda t: f * np.sin(2 * np.pi * 60 * t)
    from scipy.linalg import eigh
    w2, V = eigh(K, M)
    rom = GalerkinROM(V[:, :8]).reduce_system(K, M=M, F=f)
    t, Ur = galerkin_transient(rom, load, dt, steps, C=C)
    t, Uf, *_ = newmark_linear(M, C, K, load, dt, steps)
    assert np.linalg.norm(Ur - Uf) / np.linalg.norm(Uf) < 5e-2


def test_input_validation():
    with pytest.raises(ValueError):
        newmark_linear([[1.0]], None, [[1.0]], np.zeros((3, 1)), 0.1, 10)


def test_galerkin_transient_initial_condition_needs_mass(beam):
    K, M = beam["K"], beam["M"]
    from scipy.linalg import eigh
    _, V = eigh(K, M)
    rom = GalerkinROM(V[:, :4]).reduce_system(K, M=M, F=np.zeros(K.shape[0]))
    u0 = V[:, 0] * 1e-3
    with pytest.raises(ValueError):
        galerkin_transient(rom, np.zeros(K.shape[0]), 1e-3, 10, u0=u0)
    t, U = galerkin_transient(rom, np.zeros(K.shape[0]), 1e-3, 10, M=M, u0=u0)
    assert np.allclose(U[0], u0, atol=1e-12)
