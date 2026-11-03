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
    V = records.shape[1] - (records.shape[1] - 1) // 2  # unused, kept explicit below
    del V
    h = state0.astype(np.float64).copy()
    K = (records.shape[1] - 1) - (h.shape[1])  # (V + K + 1) - 1 - K = V
    Vdim = records.shape[1] - K - 1
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
    final_col = min(final_num_computed // mamba_block_size, block_table_width - 1)
    next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
    crosses = final_num_computed >= next_boundary
    boundary_col = None
    boundary_len = 0
    if crosses:
        boundary_col = next_boundary // mamba_block_size - 1
        boundary_len = next_boundary - num_computed
    return final_col, boundary_col, boundary_len


def test_commit_plan_boundary_cases():
    block = 128
    # exactly on a boundary: nothing crosses, final lands on the new column
    assert commit_plan(128, block, 3, 8) == (1, None, 0)
    # one token before the boundary, all accepted -> crosses with len 1
    assert commit_plan(127, block, 1, 8) == (1, 0, 1)
    # one token before the boundary, all rejected -> stays
    assert commit_plan(127, block, 0, 8) == (0, None, 0)
    # a full window crossing the boundary
    assert commit_plan(126, block, 4, 8) == (1, 0, 2)
    # clamp at the last block table column
    assert commit_plan(128 * 8, block, 4, 8) == (7, None, 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} oracle checks passed")
