"""validation.py: the report on real fea_engine cantilever POD-Galerkin ROMs."""
import numpy as np
import pytest

from fea_fixtures import cantilever_beam_system, cantilever_static_snapshots
from rom_engine import PodBasis, GalerkinROM
from rom_engine.validation import validate_rom, convergence_study, select_basis_size


@pytest.fixture(scope="module")
def setup():
    sysd = cantilever_beam_system(n=30)
    fr = sysd["free_dofs"]; K = np.asarray(sysd["K"])[np.ix_(fr, fr)]; n = K.shape[0]
    rng = np.random.default_rng(0)
    # load patterns: random combinations of a few smooth shapes (so a small basis can work)
    x = np.linspace(0, 1, n)
    shapes = np.array([np.sin((k + 1) * np.pi * x / 2) for k in range(5)])
    mk = lambda c: c @ shapes
    train = [mk(rng.normal(size=5)) for _ in range(12)]
    test = [mk(rng.normal(size=5)) for _ in range(8)]
    S = np.column_stack([np.linalg.solve(K, f) for f in train])
    return dict(K=K, S=S, test=test)


def _build(setup):
    def b(r):
        basis = PodBasis().fit(setup["S"], n_modes=r)
        rom = GalerkinROM(basis).reduce_system(setup["K"])
        return lambda f: rom.solve_static(F=f)[0]
    return b


def test_report_fields_and_pass_fail(setup):
    fom = lambda f: np.linalg.solve(setup["K"], f)
    rep = validate_rom(_build(setup)(12), fom, setup["test"], tol=1e-6)
    assert len(rep.errors) == 8 and rep.max < 1e-6 and rep.passed is True
    assert rep.speedup > 0 and "PASS" in rep.summary()
    bad = validate_rom(_build(setup)(1), fom, setup["test"], tol=1e-9)
    assert bad.passed is False and "FAIL" in bad.summary()
    assert bad.worst_index == int(np.argmax(bad.errors))
    assert set(rep.as_table()) == {"case", "error"}


def test_convergence_is_monotone_and_selection_works(setup):
    fom = lambda f: np.linalg.solve(setup["K"], f)
    st = convergence_study(_build(setup), [1, 2, 3, 5, 8], fom, setup["test"])
    assert all(a >= b - 1e-12 for a, b in zip(st["max"], st["max"][1:]))
    r = select_basis_size(st, 1e-3)
    assert r is not None and st["max"][st["sizes"].index(r)] <= 1e-3
    assert select_basis_size(st, 1e-30) is None


def test_other_norms_and_complex_outputs():
    fom = lambda x: np.array([1.0 + 1j, 2.0, 3.0]) * x
    rom = lambda x: np.array([1.0 + 1j, 2.0, 3.1]) * x
    r1 = validate_rom(rom, fom, [1.0, 2.0], norm="rel_l2")
    r2 = validate_rom(rom, fom, [1.0, 2.0], norm="rel_max")
    r3 = validate_rom(rom, fom, [1.0], norm="abs_l2")
    assert 0 < r1.max < 0.1 and 0 < r2.max < 0.1 and r3.max == pytest.approx(0.1)


def test_input_errors():
    f = lambda x: np.ones(3)
    with pytest.raises(ValueError):
        validate_rom(f, f, [])
    with pytest.raises(ValueError):
        validate_rom(lambda x: np.ones(2), f, [0])
    with pytest.raises(ValueError):
        validate_rom(f, f, [0], norm="bogus")
    nan = validate_rom(lambda x: np.full(3, np.nan), f, [0])
    assert "warning" in nan.extra
