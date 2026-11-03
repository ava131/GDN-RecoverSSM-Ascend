# SPDX-License-Identifier: Apache-2.0
"""Numerical oracle for the GDN RecoverSSM record/commit math.

Runs on plain NumPy -- no NPU, no torch, no Triton -- so it can be executed on a
laptop CI box. It pins down the *semantics* the Ascend kernels must implement:

* verify advances the state from a read-only checkpoint and stores, per token,
  ``[correction(V), normalised_key(K), decay(1)]`` (decay in log space is
  exponentiated once);
* commit folds the accepted prefix back with
  ``h <- h * decay + correction (x) key`` and must reproduce the sequential
  state exactly (in float64, bit-for-bit up to fp error).

If this test fails, the kernel is wrong regardless of what the NPU does.
"""

from __future__ import annotations

import numpy as np

RTOL = 1e-9
ATOL = 1e-9


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def gdn_gate(a_log: float, dt_bias: float, a: np.ndarray) -> np.ndarray:
    """``g = -exp(A_log) * softplus(a + dt_bias)`` (log-space decay)."""
    return -np.exp(a_log) * _softplus(a + dt_bias)


def sequential_reference(state0, q, k, v, g, beta, scale):
    """Return (outputs, final_state, per_token_mid_states).

    ``q``/``k`` are L2-normalised inside, matching ``l2norm_fwd`` semantics.
    Shapes: state0 (V, K); q/k (T, K); v (T, V); g/beta (T,).
    """
    T = q.shape[0]
    k_hat = k / np.linalg.norm(k, axis=-1, keepdims=True).clip(1e-6)
    q_hat = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-6)
    state = state0.astype(np.float64).copy()
    outs = np.empty((T, v.shape[-1]), dtype=np.float64)
    mids = []
    for t in range(T):
        decay = np.exp(g[t])
        state = decay * state
        corr = beta[t] * (v[t] - state @ k_hat[t])
        state = state + np.outer(corr, k_hat[t])
        mids.append(state.copy())
        outs[t] = (state @ q_hat[t]) * scale
    return outs, state, mids


def verify_records(state0, q, k, v, g, beta, scale):
    """Simulate the Ascend verify kernel; returns (outputs, records)."""
    T = q.shape[0]
    k_hat = k / np.linalg.norm(k, axis=-1, keepdims=True).clip(1e-6)
    q_hat = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-6)
    state = state0.astype(np.float64).copy()
    outs = np.empty((T, v.shape[-1]), dtype=np.float64)
    records = np.empty((T, v.shape[-1] + k.shape[-1] + 1), dtype=np.float64)
    V = v.shape[-1]
    K = k.shape[-1]
    for t in range(T):
        decay = np.exp(g[t])
        state = decay * state
        corr = beta[t] * (v[t] - state @ k_hat[t])
        state = state + np.outer(corr, k_hat[t])
        outs[t] = (state @ q_hat[t]) * scale
        records[t, :V] = corr
        records[t, V : V + K] = k_hat[t]
        records[t, V + K] = decay
    return outs, records


def commit(state0, records, accepted: int):
    """Simulate the Ascend commit kernel (one checkpoint write)."""
    h = state0.astype(np.float64).copy()
    # Record layout is [correction (V), key (K), decay] and ``h`` is [V, K],
    # so the dimensions come straight from the state, not from the record width.
    Vdim, K = h.shape[0], h.shape[1]
    assert records.shape[1] == Vdim + K + 1, (
        f"record width {records.shape[1]} != V({Vdim}) + K({K}) + 1"
    )
    for t in range(accepted):
        corr = records[t, :Vdim]
        key = records[t, Vdim : Vdim + K]
        decay = records[t, Vdim + K]
        h = h * decay + np.outer(corr, key)
    return h


def _make_case(seed=0, T=4, V=8, K=8):
    rng = np.random.default_rng(seed)
    state0 = rng.standard_normal((V, K)) * 0.5
    q = rng.standard_normal((T, K))
    k = rng.standard_normal((T, K))
    v = rng.standard_normal((T, V))
    a = rng.standard_normal(T) * 0.5
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal(T)))
    g = gdn_gate(0.3, -0.1, a)
    return state0, q, k, v, g, beta


def test_verify_matches_sequential():
    state0, q, k, v, g, beta = _make_case(T=5)
    scale = k.shape[-1] ** -0.5
    ref_out, ref_final, _ = sequential_reference(state0, q, k, v, g, beta, scale)
    out, _ = verify_records(state0, q, k, v, g, beta, scale)
    np.testing.assert_allclose(out, ref_out, rtol=RTOL, atol=ATOL)
    # verify must not modify the checkpoint: it is a pure function of state0
    state0_copy = state0.copy()
    verify_records(state0, q, k, v, g, beta, scale)
    np.testing.assert_array_equal(state0, state0_copy)


def test_commit_every_accept_length():
    state0, q, k, v, g, beta = _make_case(T=5)
    scale = k.shape[-1] ** -0.5
    _, _, mids = sequential_reference(state0, q, k, v, g, beta, scale)
    _, records = verify_records(state0, q, k, v, g, beta, scale)

    # a = 0 must leave the checkpoint untouched (bit-for-bit).
    np.testing.assert_array_equal(commit(state0, records, 0), state0)

    for a in range(1, records.shape[0] + 1):
        got = commit(state0, records, a)
        np.testing.assert_allclose(
            got,
            mids[a - 1],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"commit mismatch at accepted={a}",
        )


def test_record_layout_roundtrip():
    """One record is exactly [corr(V), key_hat(K), decay(1)] and replayable."""
    state0, q, k, v, g, beta = _make_case(T=3, V=5, K=7)
    scale = k.shape[-1] ** -0.5
    _, records = verify_records(state0, q, k, v, g, beta, scale)
    V, K = v.shape[-1], k.shape[-1]
    assert records.shape[1] == V + K + 1
    # decay stored must equal exp(g)
    np.testing.assert_allclose(records[:, V + K], np.exp(g), rtol=RTOL, atol=ATOL)
    # key stored must be the L2-normalised one
    k_hat = k / np.linalg.norm(k, axis=-1, keepdims=True).clip(1e-6)
    np.testing.assert_allclose(records[:, V : V + K], k_hat, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# align mode: commit-plan boundary arithmetic
# --------------------------------------------------------------------------- #
def commit_plan(num_computed: int, mamba_block_size: int, accepted: int, block_table_width: int):
    """Mirror of the boundary computation in ``_prepare_commit_plan_kernel``.

    Returns (final_block_col, boundary_block_col_or_None, boundary_len).
    """
    final_num_computed = num_computed + accepted
    # Must match the kernel: the destination is the block that holds the last
    # committed token, which is always allocated ((C+A)//B can point one column
    # past the allocated range when the window ends exactly on a boundary).
    final_col = min(
        max(final_num_computed - 1, 0) // mamba_block_size, block_table_width - 1
    )
    next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
    crosses = final_num_computed >= next_boundary
    boundary_col = None
    boundary_len = 0
    if crosses:
        boundary_col = next_boundary // mamba_block_size - 1
        boundary_len = next_boundary - num_computed
    return final_col, boundary_col, boundary_len


def test_commit_destinations_are_allocated():
    """The destination column must be one the request has actually allocated.

    P1 from the review: final_col = (C + A) // B picks column 1 at
    C=124, A=4, B=128 while the manager only guarantees ceil(128/128) = 1 block,
    so the write can land on a null page.
    """
    block = 128
    for computed, accepted in ((124, 4), (125, 3), (126, 2), (127, 1), (128, 1), (255, 1)):
        col, _, _ = commit_plan(computed, block, accepted, 8)
        needed_blocks = -(-(computed + accepted) // block)  # ceil
        assert col <= max(needed_blocks - 1, 0), (computed, accepted, col, needed_blocks)


def test_boundary_state_starts_from_same_checkpoint():
    """P1 from the review: folding the boundary in a second pass resurrects state.

    C=126, A=3, boundary_len=2, with h <- 0.5*h + 1 from h0=10:
      final  = 3   (10 -> 6 -> 4 -> 3)
      boundary = 4 (10 -> 6 -> 4)
    A second pass over the page the final fold already overwrote produced 2.25.
    """
    h0, decay, corr, key = 10.0, 0.5, 1.0, 1.0

    def fold(steps, start):
        h = start
        for _ in range(steps):
            h = h * decay + corr * key
        return h

    records = [corr] * 3
    assert abs(fold(3, h0) - 3.0) < 1e-12                     # final
    assert abs(fold(2, h0) - 4.0) < 1e-12                     # boundary, one pass

    # single pass: snapshot after 2 steps, continue to 3
    h, boundary = h0, None
    for t, c in enumerate(records):
        h = h * decay + c * key
        if t + 1 == 2:
            boundary = h
    assert abs(h - 3.0) < 1e-12 and abs(boundary - 4.0) < 1e-12

    # the old two-pass shape: boundary folded again starting from the final state
    assert abs(fold(2, 3.0) - 2.25) < 1e-12


def test_commit_plan_boundary_cases():
    block = 128
    # exactly on a boundary: nothing crosses, final lands on the new column
    assert commit_plan(128, block, 3, 8) == (1, None, 0)
    # One token before the boundary, one accepted: the committed token is #127,
    # which lives in column 0 (the only allocated column).
    assert commit_plan(127, block, 1, 8) == (0, 0, 1)
    # one token before the boundary, all rejected -> stays in column 0
    assert commit_plan(127, block, 0, 8) == (0, None, 0)
    # a full window crossing the boundary
    assert commit_plan(126, block, 4, 8) == (1, 0, 2)
    # clamp at the last block table column
    assert commit_plan(128 * 8, block, 4, 8) == (7, None, 0)


def test_commit_non_square():
    """V != K must work: the record layout is [corr(V), key(K), decay]."""
    rng = np.random.default_rng(7)
    V, K = 5, 7
    h0 = rng.standard_normal((V, K))
    records = rng.standard_normal((4, V + K + 1))
    records[:, V + K] = rng.random(4) * 0.5
    h = h0.copy()
    for a in range(5):
        assert commit(h0, records[:a], a).shape == (V, K)
    ref = h0.copy()
    for t in range(3):
        ref = ref * records[t, V + K] + np.outer(records[t, :V], records[t, V : V + K])
    np.testing.assert_allclose(commit(h0, records, 3), ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} oracle checks passed")
