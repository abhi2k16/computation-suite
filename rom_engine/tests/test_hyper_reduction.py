"""Hyper-reduction (hyper_reduction.py): ECSW on a real fea_engine nonlinear beam, DEIM/QDEIM on synthetic data."""
import numpy as np
import pytest

from fea_fixtures import clamped_clamped_nonlinear_beam_system
from rom_engine.hyper_reduction import (deim_indices, qdeim_indices, DEIM, gappy_reconstruct, ecsw_weights,
                                        ECSW, reduced_force_error, HyperReducedNonlinearROM)
from rom_engine.intrusive_nonlinear_rom import IntrusiveNonlinearROM


# ------------------------------------------------------------------ DEIM family
def _smooth_snapshots(n=200, m=60, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)[:, None]
    mu = rng.uniform(1, 3, (1, m))
    return np.exp(-mu * x) * np.sin(4 * x * mu) + 0.0 * x


@pytest.mark.parametrize("method", ["deim", "qdeim"])
def test_deim_reconstructs_unseen_snapshot(method):
    S = _smooth_snapshots()
    d = DEIM.from_snapshots(S[:, :50], n_points=10, method=method)
    f = S[:, 55]
    rec = d.approximate(f[d.indices])
    assert np.linalg.norm(rec - f) / np.linalg.norm(f) < 1e-3
    assert len(set(d.indices.tolist())) == 10


def test_deim_exact_for_vector_inside_the_basis():
    S = _smooth_snapshots()
    U, _, _ = np.linalg.svd(S, full_matrices=False)
    U = U[:, :8]
    d = DEIM(U)
    f = U @ np.arange(1, 9)
    assert np.allclose(d.approximate(f[d.indices]), f)
    # the reduced form equals projecting the approximation
    V = np.linalg.qr(np.random.default_rng(1).normal(size=(200, 4)))[0]
    assert np.allclose(d.approximate_reduced(V, f[d.indices]), V.T @ f)


def test_qdeim_not_worse_conditioned_than_greedy_much():
    U = np.linalg.svd(_smooth_snapshots(), full_matrices=False)[0][:, :12]
    ci = lambda idx: np.linalg.cond(U[idx, :])
    assert ci(qdeim_indices(U)) <= 10 * ci(deim_indices(U))


def test_gappy_least_squares_with_extra_points():
    U = np.linalg.svd(_smooth_snapshots(), full_matrices=False)[0][:, :6]
    f = U @ np.ones(6)
    idx = np.linspace(0, 199, 30).astype(int)
    assert np.allclose(gappy_reconstruct(U, idx, f[idx]), f)


# ------------------------------------------------------------------ ECSW: the weights solver
def test_ecsw_weights_nonnegative_sparse_and_accurate():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(30, 80)) ** 2          # positive contributions
    b = G.sum(axis=1)
    w, info = ecsw_weights(G, b, tol=1e-6)
    assert np.all(w >= 0)
    assert info["residual"] <= 1e-6 + 1e-12
    assert np.linalg.norm(G @ w - b) <= 1e-6 * np.linalg.norm(b) * (1 + 1e-9)
    assert info["n_selected"] <= 30 + 1          # at most about the number of rows


def test_ecsw_weights_respects_cap_and_zero_target():
    G = np.abs(np.random.default_rng(1).normal(size=(20, 40)))
    w, info = ecsw_weights(G, G.sum(axis=1), tol=1e-12, max_elements=5)
    assert info["n_selected"] <= 5
    w0, i0 = ecsw_weights(G, np.zeros(20))
    assert not w0.any() and i0["n_selected"] == 0


# ------------------------------------------------------------------ ECSW on a real nonlinear beam
@pytest.fixture(scope="module")
def beam():
    d = clamped_clamped_nonlinear_beam_system(n_elem=40, n_modes=3, damping_alpha=2.0)
    fes, mat = d["sys"], d["mat"]
    free = np.asarray(d["free_dofs"]); n = d["n_dof"]
    V = np.zeros((n, 3)); V[free, :] = d["V"]
    conn = fes.mesh.elements
    elem = fes._blocks[0][1]
    dofs = [np.asarray(fes._global_dofs(c)) for c in conn]
    force = lambda e, u: elem.internal_force(fes.mesh.nodes[conn[e]], u, mat)
    tang = lambda e, u: elem.tangent_stiffness(fes.mesh.nodes[conn[e]], u, mat)
    h = np.sqrt(12 * 8.333e-9 / 1e-4)
    peaks = d["mode_shape_peaks"]
    rng = np.random.default_rng(0)
    Q = rng.uniform(-1.5, 1.5, (3, 60)) * h / peaks[:, None]
    ec = ECSW.fit(V, V @ Q, dofs, force, tang, tol=1e-4)
    full = lambda q: V.T @ fes.assemble_internal_force(V @ q, mat)
    return dict(d=d, fes=fes, mat=mat, V=V, ec=ec, full=full, h=h, peaks=peaks, n=n, free=free)


def test_ecsw_selects_fewer_elements_and_matches_force_on_heldout_states(beam):
    ec = beam["ec"]
    assert ec.n_selected < ec.n_total
    rng = np.random.default_rng(5)
    Qt = (rng.uniform(-1.5, 1.5, (3, 20)) * beam["h"] / beam["peaks"][:, None]).T
    assert reduced_force_error(ec, beam["full"], Qt).max() < 1e-3


def test_ecsw_tangent_matches_full_reduced_tangent(beam):
    ec, V, fes, mat = beam["ec"], beam["V"], beam["fes"], beam["mat"]
    q = np.array([0.5, -0.3, 0.2]) * beam["h"] / beam["peaks"]
    Kfull = V.T @ fes.assemble_tangent_stiffness(V @ q, mat) @ V
    Kh = ec.reduced_tangent(q)
    assert np.linalg.norm(Kh - Kfull) / np.linalg.norm(Kfull) < 1e-2


def test_hyper_reduced_rom_trajectory_matches_full_intrusive_rom(beam):
    d, V, fes, mat, ec = beam["d"], beam["V"], beam["fes"], beam["mat"], beam["ec"]
    n = beam["n"]
    Mf = np.array(fes.M); Cf = np.array(fes.C)
    K0 = fes.assemble_tangent_stiffness(np.zeros(n), mat)
    zero = lambda t: np.zeros(n)
    full = IntrusiveNonlinearROM(V, Mf, Cf, lambda u: fes.assemble_internal_force(u, mat), zero,
                                 tangent_fn=lambda u: fes.assemble_tangent_stiffness(u, mat))
    hyper = HyperReducedNonlinearROM(V, Mf, Cf, zero, K0, ec)
    assert np.allclose(full.K_r, hyper.K_r)
    q0 = np.array([1.0, 0.0, 0.0]) * beam["h"] / beam["peaks"]
    dt, steps = 2e-5, 300
    _, qf, _ = full.integrate_rk4(q0, np.zeros(3), dt, steps)
    _, qh, _ = hyper.integrate_rk4(q0, np.zeros(3), dt, steps)
    qf, qh = np.asarray(qf), np.asarray(qh)
    assert qf.shape == qh.shape
    assert np.linalg.norm(qh - qf) / np.linalg.norm(qf) < 1e-2
