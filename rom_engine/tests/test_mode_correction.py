"""
test_mode_correction.py -- validates rom_engine.mode_correction against
a real fea_engine cantilever beam (fea_fixtures.cantilever_beam_system()).

Checks (mirroring docs/classical_mor_roadmap.md Section 6):
  1. Plain mode displacement (galerkin.GalerkinROM.solve_static() on a
     truncated modal basis) has REAL, nonzero truncation error against
     the true K^-1 F solve -- establishing the baseline this module
     exists to fix.
  2. mode_acceleration_response() reproduces K^-1 F to MACHINE PRECISION,
     independent of how few modes are kept -- this is not an
     approximation the module happens to make more accurate, it is an
     algebraic identity (x_MA = V*eta + q_cor telescopes exactly back to
     K^-1 F by construction, see the module's own eq. 13-16 derivation),
     and the test is written to actually prove that rather than just
     show "smaller error".
  3. augmented_basis()'s Psi = [V, q_cor] also reproduces K^-1 F to
     machine precision for the SAME load q_cor was built from -- the
     same algebraic identity, reached via the basis-augmentation route
     (eq. 20) instead of the post-hoc-correction route (eq. 13), and
     confirmed to hold with both plain and M-orthogonalization.
  4. augmented_basis() refuses (raises) when q_cor is already in the
     span of V, rather than silently appending an ill-conditioning
     near-zero column.
  5. An M-orthogonalized augmented basis still supports solve_modal()
     (i.e. its reduced mass matrix stays invertible/SPD) -- augmented_
     basis() is useful for more than just the one static load it was
     built from.
"""
import numpy as np
import pytest
from scipy.linalg import eigh, qr
from rom_engine import GalerkinROM
from rom_engine.mode_correction import (
    mode_acceleration_correction,
    mode_acceleration_response,
    augmented_basis,
)
import fea_fixtures as ff


def _modal_basis(Kff, Mff, n_modes):
    eigvals, eigvecs = eigh(Kff, Mff)
    return eigvecs[:, :n_modes]


def test_plain_mode_displacement_has_real_truncation_error():
    fx = ff.cantilever_beam_system(n=24)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    n_free = len(free)

    rng = np.random.default_rng(3)
    F = rng.standard_normal(n_free)   # a generic distributed load, not
                                        # concentrated at the tip, so
                                        # higher modes genuinely matter
    x_exact = np.linalg.solve(Kff, F)

    errs = []
    for n_modes in (2, 4, 8):
        basis = _modal_basis(Kff, Mff, n_modes)
        rom = GalerkinROM(basis).reduce_system(Kff, F=F)
        x_md, _ = rom.solve_static()
        err = np.linalg.norm(x_md - x_exact) / np.linalg.norm(x_exact)
        errs.append(err)
        assert err > 1e-6, (
            f"expected REAL truncation error at n_modes={n_modes} to set "
            f"up the comparison mode acceleration is supposed to fix, got {err:.3e}"
        )
    # error should shrink as more modes are kept (sanity on the baseline)
    assert errs[0] > errs[1] > errs[2]
    print(f"plain mode-displacement relative errors vs n_modes=(2,4,8): {errs}")


def test_mode_acceleration_is_exact_regardless_of_truncation():
    fx = ff.cantilever_beam_system(n=24)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    n_free = len(free)

    rng = np.random.default_rng(3)
    F = rng.standard_normal(n_free)
    x_exact = np.linalg.solve(Kff, F)

    for n_modes in (2, 4, 8):
        basis = _modal_basis(Kff, Mff, n_modes)
        rom = GalerkinROM(basis).reduce_system(Kff, F=F)
        _, eta = rom.solve_static()

        q_cor = mode_acceleration_correction(Kff, basis, F)
        x_ma = mode_acceleration_response(basis, eta, q_cor)

        err = np.linalg.norm(x_ma - x_exact) / np.linalg.norm(x_exact)
        assert err < 1e-9, (
            f"mode acceleration should reproduce the exact static solve "
            f"to machine precision at ANY n_modes (it's an algebraic "
            f"identity, not an approximation) -- got relative error "
            f"{err:.3e} at n_modes={n_modes}"
        )
    print("PASS -- mode acceleration exactly reproduces K^-1 F at every tested n_modes")


@pytest.mark.parametrize("use_M_orthogonalization", [False, True])
def test_augmented_basis_is_exact_for_its_own_load(use_M_orthogonalization):
    fx = ff.cantilever_beam_system(n=24)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    n_free = len(free)

    rng = np.random.default_rng(3)
    F = rng.standard_normal(n_free)
    x_exact = np.linalg.solve(Kff, F)

    n_modes = 3
    basis = _modal_basis(Kff, Mff, n_modes)
    q_cor = mode_acceleration_correction(Kff, basis, F)

    M_arg = Mff if use_M_orthogonalization else None
    Psi = augmented_basis(basis, q_cor, M=M_arg)
    assert Psi.shape == (n_free, n_modes + 1)

    rom = GalerkinROM(Psi).reduce_system(Kff, F=F)
    x_aug, _ = rom.solve_static()
    err = np.linalg.norm(x_aug - x_exact) / np.linalg.norm(x_exact)
    assert err < 1e-9, (
        f"modal truncation augmentation should reproduce the exact "
        f"static solve to machine precision for the load q_cor was "
        f"built from -- got relative error {err:.3e} "
        f"(M-orthogonalized={use_M_orthogonalization})"
    )

    # also strictly better than the plain (non-augmented) basis at the
    # same n_modes, for the SAME load
    rom_plain = GalerkinROM(basis).reduce_system(Kff, F=F)
    x_plain, _ = rom_plain.solve_static()
    err_plain = np.linalg.norm(x_plain - x_exact) / np.linalg.norm(x_exact)
    assert err < err_plain
    print(f"PASS (M-orthogonalized={use_M_orthogonalization}) -- augmented "
          f"basis error {err:.3e} vs plain-basis error {err_plain:.3e}")


def test_augmented_basis_refuses_when_qcor_already_in_span():
    fx = ff.cantilever_beam_system(n=10)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    n_free = len(free)

    rng = np.random.default_rng(5)
    Q, _ = qr(rng.standard_normal((n_free, n_free)))   # full-rank basis
    # q_cor computed against a FULL-RANK basis must be (numerically) zero
    # -- there is no static contribution left outside a full-rank basis
    F = rng.standard_normal(n_free)
    q_cor = mode_acceleration_correction(Kff, Q, F)
    assert np.linalg.norm(q_cor) < 1e-8

    with pytest.raises(ValueError, match="span"):
        augmented_basis(Q, q_cor)


def test_augmented_basis_still_supports_solve_modal():
    fx = ff.cantilever_beam_system(n=24)
    free = fx["free_dofs"]
    Kff = fx["K"][np.ix_(free, free)]
    Mff = fx["M"][np.ix_(free, free)]
    n_free = len(free)

    rng = np.random.default_rng(3)
    F = rng.standard_normal(n_free)
    n_modes = 4
    basis = _modal_basis(Kff, Mff, n_modes)
    q_cor = mode_acceleration_correction(Kff, basis, F)
    Psi = augmented_basis(basis, q_cor, M=Mff)

    rom = GalerkinROM(Psi).reduce_system(Kff, M=Mff)
    freq_hz, mode_shapes_full, _ = rom.solve_modal()
    assert np.all(np.isfinite(freq_hz))
    assert np.all(freq_hz >= 0)
    assert mode_shapes_full.shape == (n_free, n_modes + 1)
