"""
test_multi_field_pod.py -- validates rom_engine.pod.MultiFieldPOD.

Checks (per the Wave 17 item 143 roadmap entry):
  1. Synthetic separable-field recovery: snapshots built from a KNOWN
     rank-2 separable field Q(t) * Phi(s), with a KNOWN per-mode
     component split C_km and known normalized shapes phi_km(s), are
     recovered to machine precision by MultiFieldPOD.fit() (noiseless,
     exact rank-2-per-mode construction -- no reason for any recovery
     error above floating-point roundoff).
  2. Energy fractions sum to 1 across all retained modes when every
     mode is retained (the full, untruncated basis).
  3. Convergence stability: refining the time-sampling density of the
     same underlying continuous two-tone signal, once past the
     resolution needed to resolve both tones, leaves C_km / shapes /
     energy fractions essentially unchanged.

Also covers the supporting pieces MultiFieldPOD depends on:
  - assemble_field_weight_matrix / trapezoidal_field_gram (the
    dependency-free example W-builder).
  - the block-diagonal-W guard in fit().
  - the sign convention (stable across a sign-flipped snapshot set).
  - reconstruct_field().
"""
import numpy as np
import pytest
from rom_engine import MultiFieldPOD, assemble_field_weight_matrix, trapezoidal_field_gram


def _make_two_field_problem(n_nodes=5, L=4.0, seed=0):
    """Build a 2-field (n_nodes each), block-diagonal weight matrix W
    (Georgiou's own (2/L) * integral convention, via the trapezoidal
    lumped Gram as a dependency-free stand-in for a real FE shape-
    function Gram), plus two W-orthonormal "ground truth" mode shapes
    with a KNOWN, chosen component split.

    Field layout is block-major: dofs [0:n_nodes) = field A ("axial"),
    dofs [n_nodes:2*n_nodes) = field B ("transverse").
    """
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, L, n_nodes)
    G = trapezoidal_field_gram(s)             # bare integral Gram, one field
    n_dof = 2 * n_nodes
    idx_A = np.arange(0, n_nodes)
    idx_B = np.arange(n_nodes, 2 * n_nodes)
    W = assemble_field_weight_matrix(n_dof, [idx_A, idx_B], [G, G], scale=2.0 / L)
    W_AA = W[np.ix_(idx_A, idx_A)]
    W_BB = W[np.ix_(idx_B, idx_B)]

    def weighted_orthonormal_pair(Wk):
        # Same trick PodBasis._fit_mass_weighted uses: Cholesky-transform,
        # orthonormalize in that frame, transform back -> exactly
        # Wk-orthonormal columns (to machine precision).
        Lk = np.linalg.cholesky(Wk)
        R = rng.standard_normal((Wk.shape[0], 2))
        U, _ = np.linalg.qr(Lk.T @ R)
        return np.linalg.solve(Lk.T, U)

    gA = weighted_orthonormal_pair(W_AA)   # (n_nodes, 2), W_AA-orthonormal
    gB = weighted_orthonormal_pair(W_BB)   # (n_nodes, 2), W_BB-orthonormal

    C_true = np.array([[0.6, 0.8],
                        [0.8, 0.6]])        # C_true[k, m], sum_k C_true[k,m]**2 == 1

    Phi_true = np.zeros((n_dof, 2))
    Phi_true[idx_A, 0] = C_true[0, 0] * gA[:, 0]
    Phi_true[idx_B, 0] = C_true[1, 0] * gB[:, 0]
    Phi_true[idx_A, 1] = C_true[0, 1] * gA[:, 1]
    Phi_true[idx_B, 1] = C_true[1, 1] * gB[:, 1]

    return {
        "s": s, "W": W, "idx_A": idx_A, "idx_B": idx_B,
        "Phi_true": Phi_true, "C_true": C_true,
        "gA": gA, "gB": gB,
    }


def _sign_align(Phi_true, ref_idx):
    """Apply MultiFieldPOD's own sign convention (largest-magnitude entry
    of the reference field positive) to a ground-truth basis, so it is
    directly comparable to MultiFieldPOD's recovered basis.V."""
    Phi = Phi_true.copy()
    for m in range(Phi.shape[1]):
        col = Phi[ref_idx, m]
        k = int(np.argmax(np.abs(col)))
        if col[k] < 0:
            Phi[:, m] *= -1
    return Phi


def test_synthetic_separable_field_recovery():
    prob = _make_two_field_problem(n_nodes=5, L=4.0, seed=0)
    W, Phi_true, C_true = prob["W"], prob["Phi_true"], prob["C_true"]
    idx_A, idx_B = prob["idx_A"], prob["idx_B"]

    # Known, mutually-orthogonal (exact, via QR) time amplitudes with
    # distinct norms 5 and 2, so the true singular values are unambiguous
    # and mode ordering (largest singular value first) is guaranteed.
    rng = np.random.default_rng(1)
    n_snap = 60
    raw = rng.standard_normal((n_snap, 2))
    Qo, _ = np.linalg.qr(raw)              # (n_snap, 2), orthonormal columns
    Q_true = (Qo * np.array([5.0, 2.0])).T  # (2, n_snap), orthogonal rows, norms 5 and 2

    X = Phi_true @ Q_true                  # (n_dof, n_snap) raw, un-centered snapshots

    mfpod = MultiFieldPOD([idx_A, idx_B], W, sign_reference_field=0)
    mfpod.fit(X, n_modes=2)

    Phi_true_aligned = _sign_align(Phi_true, idx_A)

    max_shape_err = np.max(np.abs(mfpod.basis.V - Phi_true_aligned))
    print(f"max |recovered V - sign-aligned ground truth Phi_true| = {max_shape_err:.3e}")
    assert max_shape_err < 1e-9, "exact rank-2 separable snapshots should recover the true modes to machine precision"

    max_C_err = np.max(np.abs(mfpod.C - C_true))
    print(f"recovered C_km:\n{mfpod.C}\nknown C_true:\n{C_true}\nmax error = {max_C_err:.3e}")
    assert max_C_err < 1e-9, "component norms C_km should match the known ground truth to machine precision"

    # sum_k C_km^2 == 1 for every mode, the real mathematical identity.
    sums = np.sum(mfpod.C**2, axis=0)
    print(f"sum_k C_km^2 per mode: {sums}")
    assert np.allclose(sums, 1.0, atol=1e-10), "sum_k C_km^2 must equal 1 for every retained mode"

    # normalized shapes: phi_km^T W_kk phi_km == 1
    W_AA = W[np.ix_(idx_A, idx_A)]
    W_BB = W[np.ix_(idx_B, idx_B)]
    for m in range(2):
        nA = mfpod.component_shapes[0][:, m] @ (W_AA @ mfpod.component_shapes[0][:, m])
        nB = mfpod.component_shapes[1][:, m] @ (W_BB @ mfpod.component_shapes[1][:, m])
        print(f"mode {m}: (2/L)-weighted shape norms field A={nA:.10f}, field B={nB:.10f}")
        assert abs(nA - 1.0) < 1e-9
        assert abs(nB - 1.0) < 1e-9

    # field-separated reconstruction reproduces the known field time
    # histories exactly (same sign convention applied consistently to
    # both V and Q, so reconstruction itself is sign-independent).
    X_A_hat = mfpod.reconstruct_field(0)
    X_B_hat = mfpod.reconstruct_field(1)
    err_A = np.max(np.abs(X_A_hat - X[idx_A, :]))
    err_B = np.max(np.abs(X_B_hat - X[idx_B, :]))
    print(f"field-separated reconstruction max errors: A={err_A:.3e}, B={err_B:.3e}")
    assert err_A < 1e-9 and err_B < 1e-9
    print("PASS -- synthetic separable-field recovery exact to machine precision")


def test_energy_fractions_sum_to_one_at_full_rank():
    prob = _make_two_field_problem(n_nodes=6, L=3.0, seed=2)
    W, Phi_true = prob["W"], prob["Phi_true"]
    idx_A, idx_B = prob["idx_A"], prob["idx_B"]

    rng = np.random.default_rng(3)
    n_snap = 40
    raw = rng.standard_normal((n_snap, 2))
    Qo, _ = np.linalg.qr(raw)
    Q_true = (Qo * np.array([5.0, 2.0])).T
    X = Phi_true @ Q_true

    mfpod = MultiFieldPOD([idx_A, idx_B], W).fit(X, n_modes=2)   # full rank (2) retained
    total = np.sum(mfpod.energy_fractions)
    print(f"energy fractions (full rank retained): {mfpod.energy_fractions}, sum={total:.12f}")
    assert abs(total - 1.0) < 1e-10, "energy fractions must sum to 1 when every mode is retained"

    expected = np.array([25.0, 4.0]) / 29.0   # sigma1=5, sigma2=2 -> energies 25, 4
    print(f"expected fractions from known singular values 5, 2: {expected}")
    assert np.allclose(mfpod.energy_fractions, expected, atol=1e-8)

    # truncating to 1 mode: fraction no longer sums to 1, but is still the
    # correct fraction of the FULL spectrum's energy.
    mfpod_trunc = MultiFieldPOD([idx_A, idx_B], W).fit(X, n_modes=1)
    print(f"truncated (n_modes=1) energy fraction: {mfpod_trunc.energy_fractions}")
    assert abs(float(mfpod_trunc.energy_fractions[0]) - expected[0]) < 1e-8
    print("PASS -- energy fractions sum to 1 at full rank, and truncated fractions match the known spectrum")


def test_convergence_stability_under_time_refinement():
    prob = _make_two_field_problem(n_nodes=5, L=4.0, seed=4)
    W, Phi_true = prob["W"], prob["Phi_true"]
    idx_A, idx_B = prob["idx_A"], prob["idx_B"]

    # Two-tone continuous signal at NON-integer cycle counts (4.3 and 1.7
    # cycles over the window) so the two rows of Q_true are NOT exactly
    # discretely orthogonal at any finite sample count (unlike
    # integer-multiple harmonics, whose discrete inner product vanishes
    # identically regardless of resolution) -- their discrete overlap is
    # O(1/n_time) and only vanishes as the sampling is refined, so this
    # genuinely exercises "converges as the snapshot interval is refined"
    # rather than being exact from the coarsest resolution onward.
    T = 1.0

    def sample(n_time):
        t = np.linspace(0.0, T, n_time, endpoint=False)
        Q_true = np.vstack([5.0 * np.cos(2 * np.pi * 4.3 * t / T),
                             2.0 * np.sin(2 * np.pi * 1.7 * t / T)])
        X = Phi_true @ Q_true
        return MultiFieldPOD([idx_A, idx_B], W).fit(X, n_modes=2)

    coarse = sample(40)
    fine = sample(800)

    dC = np.max(np.abs(coarse.C - fine.C))
    dE = np.max(np.abs(coarse.energy_fractions - fine.energy_fractions))
    dPhiA = np.max(np.abs(coarse.component_shapes[0] - fine.component_shapes[0]))
    dPhiB = np.max(np.abs(coarse.component_shapes[1] - fine.component_shapes[1]))
    print(f"coarse (n=40) vs fine (n=800) time sampling:")
    print(f"  max |dC_km| = {dC:.3e}")
    print(f"  max |d energy_fraction| = {dE:.3e}")
    print(f"  max |d phi_A| = {dPhiA:.3e}, max |d phi_B| = {dPhiB:.3e}")
    assert dC < 5e-3
    assert dE < 1e-2
    assert dPhiA < 5e-3
    assert dPhiB < 5e-3

    # and refining further (800 -> 3200) should move things much LESS than
    # the coarse -> fine step did -- confirms this is genuine convergence
    # (the fine resolution is already past what's needed), not
    # coincidence at these two particular sample counts.
    finer = sample(3200)
    dC2 = np.max(np.abs(fine.C - finer.C))
    print(f"fine (n=800) vs finer (n=3200): max |dC_km| = {dC2:.3e}")
    assert dC2 < dC / 10
    print("PASS -- C_km / shapes / energy fractions converge under time-sampling refinement")


def test_block_diagonal_guard_rejects_coupled_weight_matrix():
    prob = _make_two_field_problem(n_nodes=4, L=2.0, seed=5)
    W = prob["W"].copy()
    idx_A, idx_B = prob["idx_A"], prob["idx_B"]
    # inject real cross-field coupling -- small enough relative to the
    # diagonal to keep W symmetric positive definite (needed for the
    # check_block_diagonal=False path, which still Cholesky-factors W
    # inside PodBasis), but far above the 1e-9-relative tolerance the
    # block-diagonal guard itself checks against.
    diag_scale = np.min(np.diag(W)[np.r_[idx_A, idx_B]])
    coupling = 1e-3 * diag_scale
    W[idx_A[0], idx_B[0]] = W[idx_B[0], idx_A[0]] = coupling

    X = np.random.default_rng(6).standard_normal((W.shape[0], 10))
    mfpod = MultiFieldPOD([idx_A, idx_B], W)
    with pytest.raises(ValueError, match="block-diagonal"):
        mfpod.fit(X, n_modes=2)
    print("PASS -- fit() rejects a W with real cross-field coupling by default")

    # explicitly opting out of the check should proceed without raising
    mfpod2 = MultiFieldPOD([idx_A, idx_B], W)
    mfpod2.fit(X, n_modes=2, check_block_diagonal=False)
    assert mfpod2.basis is not None
    print("PASS -- check_block_diagonal=False lets a coupled W proceed")


def test_sign_convention_is_stable_across_globally_flipped_snapshots():
    prob = _make_two_field_problem(n_nodes=5, L=4.0, seed=7)
    W, Phi_true = prob["W"], prob["Phi_true"]
    idx_A, idx_B = prob["idx_A"], prob["idx_B"]

    rng = np.random.default_rng(8)
    n_snap = 30
    raw = rng.standard_normal((n_snap, 2))
    Qo, _ = np.linalg.qr(raw)
    Q_true = (Qo * np.array([5.0, 2.0])).T
    X = Phi_true @ Q_true

    basis1 = MultiFieldPOD([idx_A, idx_B], W).fit(X, n_modes=2).basis.V
    basis2 = MultiFieldPOD([idx_A, idx_B], W).fit(-X, n_modes=2).basis.V  # globally sign-flipped data

    max_diff = np.max(np.abs(basis1 - basis2))
    print(f"max |V(X) - V(-X)| under the sign convention: {max_diff:.3e}")
    assert max_diff < 1e-9, "the sign convention should make recovered mode shapes identical regardless of a global sign flip of the input data"
    print("PASS -- sign convention is stable under a global sign flip of the snapshot data")
