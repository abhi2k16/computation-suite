"""
batch.py -- run many independent FE jobs (parameter sweeps, load cases, datasets).

    from fea_engine.batch import map as fe_map
    results = fe_map(build_and_solve, params, n_jobs=4)

The convention matches rom_engine's sweeps: ``n_jobs=1`` (default) runs serially in
the calling process; ``n_jobs>1`` uses a pool. Results keep the input order.

Process pools (the default for ``n_jobs>1``) need a module-level, picklable function
and, on Windows/macOS, the usual ``if __name__ == "__main__":`` guard in the calling
script. Pass ``backend="thread"`` to run closures or lambdas; threads help when the
work is dominated by SciPy/BLAS calls that release the GIL.

Note: an :class:`~fea_engine.fields.FEField` returned from a process worker arrives
without its mesh (names are kept, node-set access is not). Re-attach with
``system.field(np.asarray(U))`` in the parent if you need ``nodes="tip"``.
"""
from __future__ import annotations
__author__ = "Abhijeet"

import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

__all__ = ["map", "BatchError"]

_builtin_map = map


class BatchError(RuntimeError):
    """Raised by :func:`map` (``on_error="raise"``) naming the failing item."""

    def __init__(self, index, item, cause):
        super().__init__(f"batch item {index} ({item!r:.80}) failed: {type(cause).__name__}: {cause}")
        self.index, self.item, self.cause = index, item, cause


class _Failed:
    """Placeholder for a failed item when ``on_error="collect"``."""

    def __init__(self, index, item, exception):
        self.index, self.item, self.exception = index, item, exception

    def __repr__(self):
        return f"<Failed item {self.index}: {type(self.exception).__name__}: {self.exception}>"


Failed = _Failed


def _call(fn, index, item):
    try:
        return index, True, fn(item)
    except Exception as exc:          # returned, not raised, so the pool never loses the index
        return index, False, exc


def map(fn, items, n_jobs=1, backend="process", on_error="raise", progress=None):
    """Apply ``fn`` to every element of ``items`` and return the results in order.

    Parameters
    ----------
    fn : callable
        ``fn(item) -> result``. Must be picklable for ``backend="process"``.
    items : iterable
        Inputs (parameter dicts, load cases, ...).
    n_jobs : int
        1 = serial (default); ``-1`` = all CPU cores; otherwise the worker count.
    backend : {"process", "thread"}
        Pool type when ``n_jobs != 1``.
    on_error : {"raise", "collect"}
        ``"raise"`` stops with :class:`BatchError` naming the failing item.
        ``"collect"`` puts a ``Failed`` object (``.index``, ``.item``, ``.exception``)
        in that slot and carries on.
    progress : callable, optional
        ``progress(done, total)`` called after each finished item.

    Returns
    -------
    list
        One result per item, in input order.
    """
    items = list(items)
    if on_error not in ("raise", "collect"):
        raise ValueError(f"on_error must be 'raise' or 'collect', got {on_error!r}")
    if backend not in ("process", "thread"):
        raise ValueError(f"backend must be 'process' or 'thread', got {backend!r}")
    if not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be a positive integer or -1, got {n_jobs!r}")
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 1
    total = len(items)
    results = [None] * total

    def _store(index, ok, value):
        if ok:
            results[index] = value
        elif on_error == "raise":
            raise BatchError(index, items[index], value) from value
        else:
            results[index] = _Failed(index, items[index], value)

    done = 0
    if n_jobs == 1 or total <= 1:
        for i, item in enumerate(items):
            _store(*_call(fn, i, item))
            done += 1
            if progress:
                progress(done, total)
        return results

    Pool = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    with Pool(max_workers=min(n_jobs, total)) as ex:
        futures = [ex.submit(_call, fn, i, item) for i, item in enumerate(items)]
        try:
            for fut in as_completed(futures):
                _store(*fut.result())
                done += 1
                if progress:
                    progress(done, total)
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return results
