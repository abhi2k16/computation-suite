"""Component mode synthesis (cms.py), validated against a real fea_engine clamped-clamped beam split in two."""
import numpy as np
import pytest
from scipy.linalg import eigh

from fea_engine import elements
from fea_engine.geometry import generate_mesh
from fea_engine.material import Material, Section, EI_beam
from fea_engine.solver import FESystem

from rom_engine.cms import guyan, craig_bampton, couple

E, RHO, A, I = 210e9, 7800.0, 0.01, 8.33e-6
MAT = Material(E=E, nu=0.3, rho=RHO)
N = 20


def _beam(L, n, fixed):
    s = FESystem(generate_mesh(dim=1, L=L, n=n), elements.Beam2DEulerBernoulli())
    s.assemble_stiffness(EI_beam(MAT, Section(A=A, I=I)))
    s.assemble_mass(RHO * A)
    s.fix_dofs(fixed, [0, 1])
    fr = np.asarray(s.free_dofs)
    return fr, np.array(s.K)[np.ix_(fr, fr)], np.array(s.M)[np.ix_(fr, fr)]


@pytest.fixture(scope="module")
def parts():
    fr, K, M = _beam(1.0, 2 * N, [0, 2 * N])
    f_full = np.sqrt(np.clip(eigh(K, M, eigvals_only=True), 0, None)) / (2 * np.pi)
    frL, KL, ML = _beam(0.5, N, [0])
    frR, KR, MR = _beam(0.5, N, [N])
    pos = lambda f, d: int(np.where(f == d)[0][0])
    ibL = [pos(frL, 2 * N), pos(frL, 2 * N + 1)]
    ibR = [pos(frR, 0), pos(frR, 1)]
    return dict(f_full=f_full, KL=KL, ML=ML, KR=KR, MR=MR, ibL=ibL, ibR=ibR)


def _coupled(p, nm):
    cl = craig_bampton(p["KL"], p["ML"], p["ibL"], nm)
    cr = craig_bampton(p["KR"], p["MR"], p["ibR"], nm)
    return couple([cl, cr], [[0, 1], [0, 1]]), cl, cr


def test_all_modes_reproduce_full_model(parts):
    cp, *_ = _coupled(parts, None)
    f, _ = cp.solve_modal(6)
    assert np.max(np.abs(f - parts["f_full"][:6]) / parts["f_full"][:6]) < 1e-7


def test_error_decreases_with_more_modes(parts):
    errs = []
    for nm in (2, 5, 10, 20):
        f, _ = _coupled(parts, nm)[0].solve_modal(4)
        errs.append(np.max(np.abs(f - parts["f_full"][:4]) / parts["f_full"][:4]))
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-3


def test_reduced_size_and_structure(parts):
    _, cl, _ = _coupled(parts, 5)
    assert cl.K.shape == (2 + 5, 2 + 5)
    # modal block: K = diag(omega^2), M = identity, no coupling to interface in K
    assert np.allclose(cl.K[2:, 2:], np.diag(cl.omega_fixed**2), rtol=1e-8, atol=1e-3)
    assert np.allclose(cl.M[2:, 2:], np.eye(5), atol=1e-8)
    assert np.all(cl.K[:2, 2:] == 0)


def test_fixed_interface_frequencies_match_clamped_substructure(parts):
    # fixing the interface of the left half makes it clamped-clamped
    fr, K, M = _beam(0.5, N, [0, N])
    f_cc = np.sqrt(eigh(K, M, eigvals_only=True)[:3]) / (2 * np.pi)
    cl = craig_bampton(parts["KL"], parts["ML"], parts["ibL"], 3)
    assert np.allclose(cl.freq_fixed_hz, f_cc, rtol=1e-8)


def test_expand_gives_continuous_displacement_at_interface(parts):
    cp, cl, cr = _coupled(parts, 8)
    _, X = cp.solve_modal(1)
    uL = cp.expand(0, X[:, 0]); uR = cp.expand(1, X[:, 0])
    assert np.allclose(uL[parts["ibL"]], uR[parts["ibR"]])


def test_guyan_is_exact_statically_but_approximate_dynamically(parts):
    K, M, ib = parts["KL"], parts["ML"], parts["ibL"]
    g = guyan(K, M, ib)
    # static: a displacement imposed on the masters is reproduced exactly by the condensed stiffness
    um = np.array([1e-3, 2e-4])
    u = g.expand(um)
    f_cond = g.K @ um
    f_full = K @ u
    assert np.allclose(g.T.T @ f_full, f_cond)
    assert np.allclose(f_full[g.slave], 0.0, atol=1e-6 * np.abs(f_full).max())   # slaves force-free
    # dynamic: with masters = interface only, Guyan misses internal modes; CB with modes beats it
    f_g = np.sqrt(np.clip(eigh(g.K, g.M, eigvals_only=True), 0, None))[:1] / (2 * np.pi)
    f_ref = np.sqrt(eigh(K, M, eigvals_only=True)[:1]) / (2 * np.pi)
    cb = craig_bampton(K, M, ib, 6)
    f_cb = np.sqrt(np.clip(eigh(cb.K, cb.M, eigvals_only=True), 0, None))[:1] / (2 * np.pi)
    assert abs(f_cb[0] - f_ref[0]) < abs(f_g[0] - f_ref[0]) + 1e-12


def test_input_validation(parts):
    with pytest.raises(ValueError):
        craig_bampton(parts["KL"], parts["ML"], [0, 0])
    with pytest.raises(ValueError):
        couple([craig_bampton(parts["KL"], parts["ML"], parts["ibL"], 2)], [[0, 1], [0, 1]])
