"""
test_newton_options_and_batch.py -- v1.0.1 P2: shared NewtonOptions and batch.map.
Both are additive; drivers called without `options=` behave exactly as before.
"""
__author__ = "Abhijeet"
import inspect
import os

import numpy as np
import pytest

import fea_engine
from fea_engine import NewtonOptions, FESystem, Beam2DReissner
from fea_engine import nonlinear_solver as ns
from fea_engine.mesh import Mesh
from fea_engine.newton_options import accepts_options
from fea_engine import batch

DRIVERS = ["solve_contact_lagrange_static", "solve_contact_augmented_lagrange_static",
           "solve_nonlinear_static", "solve_nonlinear_displacement_control",
           "solve_nonlinear_arc_length", "solve_nonlinear_koiter_newton",
           "solve_nonlinear_koiter_newton_generic", "solve_nonlinear_static_koiter_newton",
           "solve_nonlinear_transient", "solve_transient_displacement_control"]


# ============================================================== NewtonOptions object
class TestNewtonOptionsObject:
    def test_defaults_are_all_none_and_frozen(self):
        o = NewtonOptions()
        assert o.as_kwargs() == {}
        with pytest.raises(Exception):
            o.tol = 1.0                       # frozen dataclass

    def test_validation(self):
        for bad in (dict(tol=0), dict(tol=-1e-8), dict(max_iter=0), dict(max_iter=2.5),
                    dict(du_tol=-1.0), dict(energy_tol=0.0)):
            with pytest.raises(ValueError):
                NewtonOptions(**bad)

    def test_updated_and_as_kwargs(self):
        o = NewtonOptions(tol=1e-9, line_search=False)
        assert o.updated(max_iter=5).as_kwargs() == dict(tol=1e-9, line_search=False, max_iter=5)
        assert o.as_kwargs(accepted={"tol"}) == dict(tol=1e-9)
        assert o.tol == 1e-9                   # original untouched


# ============================================================== decorator semantics
class TestAcceptsOptions:
    @staticmethod
    def _make():
        seen = {}

        @accepts_options
        def drv(a, b=1, tol=1e-8, max_iter=30, line_search=True, **kwargs):
            """Doc."""
            seen.update(a=a, b=b, tol=tol, max_iter=max_iter, line_search=line_search, kwargs=kwargs)
            return "ok"
        return drv, seen

    def test_options_fill_unspecified_keywords(self):
        drv, seen = self._make()
        drv(1, options=NewtonOptions(tol=1e-4, max_iter=7))
        assert (seen["tol"], seen["max_iter"], seen["line_search"]) == (1e-4, 7, True)

    def test_explicit_keyword_wins_and_positional_wins(self):
        drv, seen = self._make()
        drv(1, tol=1e-2, options=NewtonOptions(tol=1e-4, max_iter=7))
        assert seen["tol"] == 1e-2 and seen["max_iter"] == 7
        drv(1, 5, 1e-3, options=NewtonOptions(tol=1e-4))          # tol passed positionally
        assert seen["tol"] == 1e-3

    def test_fields_the_driver_lacks_are_ignored_not_forwarded(self):
        drv, seen = self._make()
        drv(1, options=NewtonOptions(du_tol=1e-6, energy_tol=1e-6, verbose=True))
        assert seen["kwargs"] == {}                                # nothing leaked into **kwargs

    def test_no_options_is_a_pure_passthrough(self):
        drv, seen = self._make()
        assert drv(3) == "ok" and seen["tol"] == 1e-8 and seen["a"] == 3

    def test_type_check_and_signature_and_doc(self):
        drv, _ = self._make()
        with pytest.raises(TypeError, match="NewtonOptions"):
            drv(1, options={"tol": 1e-3})
        p = inspect.signature(drv).parameters
        assert p["options"].kind is inspect.Parameter.KEYWORD_ONLY and p["options"].default is None
        assert list(p)[-1] == "kwargs"
        assert "options : NewtonOptions" in drv.__doc__ and drv.__name__ == "drv"

    def test_idempotent_on_native_options_parameter(self):
        def f(a, options=None):
            return options
        assert accepts_options(f) is f


# ============================================================== wired into the real drivers
@pytest.mark.parametrize("name", DRIVERS)
def test_every_driver_advertises_options(name):
    fn = getattr(ns, name)
    assert "options" in inspect.signature(fn).parameters
    assert "NewtonOptions" in fn.__doc__


def _cantilever(n=8):
    E, nu, A, I, L = 210e9, 0.3, 1e-3, 8.33e-7, 1.0
    G = E / (2 * (1 + nu))
    x = np.linspace(0, L, n + 1).reshape(-1, 1)
    s = FESystem(Mesh(nodes=np.hstack([x, 0 * x]),
                      elements=np.array([[i, i + 1] for i in range(n)]), dim=1), Beam2DReissner())
    s.fix_dofs([0], ["ux", "uy", "rz"])
    mat = (E, G, A, I, 1.0)
    s.assemble_stiffness(mat)
    s.add_nodal_force([n], "uy", 0.8 * 3 * E * I / L ** 2)
    return s, mat


def test_real_driver_options_equal_explicit_keywords():
    s1, m = _cantilever()
    lf1, U1 = ns.solve_nonlinear_static(s1, m, n_steps=5, tol=1e-9, max_iter=40, line_search=False)
    s2, m = _cantilever()
    lf2, U2 = ns.solve_nonlinear_static(
        s2, m, n_steps=5, options=NewtonOptions(tol=1e-9, max_iter=40, line_search=False))
    assert np.array_equal(np.asarray(lf1), np.asarray(lf2))
    assert np.array_equal(np.asarray(U1[-1]), np.asarray(U2[-1]))


def test_real_driver_options_actually_reach_the_driver(monkeypatch):
    """The wrapped function must receive the values (spy on the undecorated function)."""
    s, m = _cantilever()
    captured = {}
    inner = ns.solve_nonlinear_static.__wrapped__

    def stand_in(fesystem, mat, n_steps=10, tol=1e-8, max_iter=30, verbose=False, load_factors=None,
                 du_tol=None, energy_tol=None, line_search=True, **kw):
        captured.update(tol=tol, max_iter=max_iter, du_tol=du_tol, line_search=line_search)
        return inner(fesystem, mat, n_steps=n_steps, tol=tol, max_iter=max_iter, line_search=line_search)
    accepts_options(stand_in)(s, m, n_steps=2, options=NewtonOptions(tol=1e-7, max_iter=11, line_search=False))
    assert captured == dict(tol=1e-7, max_iter=11, du_tol=None, line_search=False)


def test_one_options_object_works_across_different_drivers():
    opts = NewtonOptions(tol=1e-9, max_iter=40, line_search=True, verbose=False)
    s, m = _cantilever()
    lf, U = ns.solve_nonlinear_static(s, m, n_steps=4, options=opts)
    assert len(lf) >= 1
    s, m = _cantilever()
    ns.solve_nonlinear_arc_length(s, m, delta_L=0.05, n_steps=2, options=opts)
    s, m = _cantilever()
    # koiter-newton has no line_search keyword: the same options object must still be accepted
    ns.solve_nonlinear_koiter_newton(s, m, delta_L=0.05, n_steps=1, options=opts)


# ============================================================== batch.map
def _square(x):
    return x * x


def _boom_on_three(x):
    if x == 3:
        raise ValueError("three is bad")
    return x


def _pid(_):
    return os.getpid()


class TestBatchMap:
    def test_serial_preserves_order(self):
        assert batch.map(_square, range(6)) == [0, 1, 4, 9, 16, 25]

    def test_empty_and_single_item(self):
        assert batch.map(_square, []) == []
        assert batch.map(_square, [4], n_jobs=4) == [16]

    @pytest.mark.parametrize("backend", ["thread", "process"])
    def test_parallel_matches_serial_and_keeps_order(self, backend):
        items = list(range(12))
        assert batch.map(_square, items, n_jobs=3, backend=backend) == [i * i for i in items]

    def test_process_backend_really_uses_other_processes(self):
        pids = set(batch.map(_pid, range(6), n_jobs=2, backend="process"))
        assert os.getpid() not in pids

    def test_thread_backend_accepts_closures(self):
        k = 10
        assert batch.map(lambda x: x + k, [1, 2, 3], n_jobs=2, backend="thread") == [11, 12, 13]

    def test_n_jobs_minus_one_means_all_cores(self):
        assert batch.map(_square, range(4), n_jobs=-1, backend="thread") == [0, 1, 4, 9]

    def test_error_raise_names_the_item(self):
        with pytest.raises(batch.BatchError, match=r"batch item 3 .*three is bad") as ei:
            batch.map(_boom_on_three, range(6))
        assert ei.value.index == 3 and isinstance(ei.value.cause, ValueError)
        with pytest.raises(batch.BatchError, match="item 3"):
            batch.map(_boom_on_three, range(6), n_jobs=2, backend="thread")

    def test_error_collect_keeps_going(self):
        out = batch.map(_boom_on_three, range(6), on_error="collect")
        assert [o for i, o in enumerate(out) if i != 3] == [0, 1, 2, 4, 5]
        assert isinstance(out[3], batch.Failed) and out[3].index == 3
        assert isinstance(out[3].exception, ValueError)
        out = batch.map(_boom_on_three, range(6), n_jobs=2, backend="process", on_error="collect")
        assert isinstance(out[3], batch.Failed) and out[5] == 5

    def test_progress_callback(self):
        calls = []
        batch.map(_square, range(5), progress=lambda d, t: calls.append((d, t)))
        assert calls == [(i, 5) for i in range(1, 6)]
        calls.clear()
        batch.map(_square, range(5), n_jobs=2, backend="thread", progress=lambda d, t: calls.append((d, t)))
        assert [d for d, _ in calls] == [1, 2, 3, 4, 5] and all(t == 5 for _, t in calls)

    def test_argument_validation(self):
        for kw in (dict(n_jobs=0), dict(n_jobs=-2), dict(n_jobs=1.5), dict(backend="gpu"), dict(on_error="ignore")):
            with pytest.raises(ValueError):
                batch.map(_square, [1, 2], **kw)

    def test_does_not_shadow_builtin_map_globally(self):
        assert list(map(str, [1, 2])) == ["1", "2"]          # builtin untouched in this module


def _solve_case(load):
    from fea_engine import Material, D_plane_stress, Quad4PlaneStress
    from fea_engine.mesh import rectangle_mesh
    mesh = rectangle_mesh(Lx=0.4, Ly=0.2, nx=8, ny=4)
    s = FESystem(mesh, Quad4PlaneStress(), thickness=0.02)
    s.assemble_stiffness(D_plane_stress(Material(E=2.1e11, nu=0.3, rho=7850.0)), thickness=0.02)
    mesh.select_nodes(x=0.0, name="root"); mesh.select_nodes(x=0.4, name="tip")
    s.fix_dofs("root", ["ux", "uy"]); s.add_nodal_force("tip", "uy", load)
    return float(s.solve_static().component("uy", nodes="tip").mean())


def test_batch_runs_real_fe_sweep_and_result_is_linear_in_load():
    loads = [-1e3, -2e3, -4e3]
    serial = batch.map(_solve_case, loads)
    par = batch.map(_solve_case, loads, n_jobs=2, backend="process")
    assert par == pytest.approx(serial, rel=1e-12)
    assert serial[1] == pytest.approx(2 * serial[0], rel=1e-9) and serial[2] == pytest.approx(4 * serial[0], rel=1e-9)


def test_top_level_exports():
    assert fea_engine.NewtonOptions is NewtonOptions and fea_engine.batch is batch
