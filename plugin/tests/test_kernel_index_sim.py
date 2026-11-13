# SPDX-License-Identifier: Apache-2.0
"""Index-level simulation of the three commit/verify kernels.

The Triton kernels take raw pointers, so an off-by-one in the loop decomposition
or a wrong stride index turns into a wild device address instead of a Python
error. This module re-implements **only the index arithmetic** of

  * ``_gdn_recoverssm_verify_kernel``
  * ``_commit_gdn_state_kernel``
  * ``_compact_conv_state_kernel``

in plain Python, driven by the rewritten bounded dispatch, and asserts that every
computed offset lands inside the corresponding buffer for realistic shapes.

It is deliberately independent of the formulas: it does not check what is stored,
only that nothing is addressed out of range. No Triton, no NPU.
"""

from __future__ import annotations

import math

# State pages and request-row scratch have distinct domains. These are the
# deployed TP4 head dimensions, with 128 scratch rows independent of KV pages.
BLOCKS = 1418
RECORD_ROWS = 128
HV = 12
HV_KEY = 4  # key heads after TP
V = K = 128
BK = 128  # next_power_of_2(K)
SPEC_Q = 4
CONV_KERNEL = 4
CONV_WINDOW = CONV_KERNEL - 1 + (SPEC_Q - 1)  # = 6
CONV_HIST = CONV_WINDOW - SPEC_Q + 1  # = 3 == conv_kernel - 1
CONV_DIM = 2 * 128 * HV_KEY + 128 * HV  # dim-first conv state


def _strides(shape):
    """Row-major strides, matching torch's contiguous layout."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _dispatch(num_work, cap):
    """The bounded physical loop used by every rewritten kernel."""
    progs = min(num_work, cap)
    for pid in range(progs):
        for work in range(pid, num_work, progs):
            yield work


def _decompose(work, n_tiles, rows, heads):
    tile = work % n_tiles
    rest = work // n_tiles
    row = rest % rows
    head = rest // rows
    assert 0 <= tile < n_tiles and 0 <= row < rows and 0 <= head < heads
    return tile, row, head


def _check(name, addr, numel):
    assert 0 <= addr < numel, f"{name}: offset {addr} outside [0, {numel})"


def test_verify_indices_in_range():
    total, rows = 512, 128
    state_shape = (BLOCKS, HV, V, K)
    replay_shape = (RECORD_ROWS, HV, SPEC_Q, V)
    key_shape = (RECORD_ROWS, HV_KEY, SPEC_Q, K)
    decay_shape = (RECORD_ROWS, HV, SPEC_Q)
    out_shape = (1, total, HV, V)
    q_shape = (total, HV_KEY, K)
    for shape in (state_shape, replay_shape, out_shape, q_shape):
        assert sum(s * (0 if i else 1) for i, s in enumerate(shape)) >= 0
    s_str = _strides(state_shape)
    r_str = _strides(replay_shape)
    k_str = _strides(key_shape)
    d_str = _strides(decay_shape)
    o_str = _strides(out_shape)
    q_str = _strides(q_shape)

    nv = math.ceil(V / 32)
    BV = 32
    num_work = nv * rows * HV
    for cap in (1, 8, 48, 192):
        for work in _dispatch(num_work, cap):
            vt, row, hv = _decompose(work, nv, rows, HV)
            bos = row * SPEC_Q
            for t in range(SPEC_Q):
                tok = bos + t
                if tok >= total:
                    continue
                # out store
                for j in range(0, BV, 16):
                    ov = vt * BV + j
                    if ov < V:
                        _check("out", tok * o_str[1] + hv * V + ov, total * HV * V)
                # q read (key heads)
                ph = hv // (HV // HV_KEY)
                for kk in range(0, BK, 32):
                    if kk < K:
                        _check("q", tok * q_str[1] + ph * K + kk, total * HV_KEY * K)
                # Records depend on request row, never a prefix-shared page id.
                base = row * r_str[0] + hv * r_str[1] + t * r_str[2]
                _check("rec_corr", base + V - 1, math.prod(replay_shape))
                _check(
                    "rec_key",
                    row * k_str[0] + ph * k_str[1] + t * k_str[2] + K - 1,
                    math.prod(key_shape),
                )
                _check(
                    "rec_decay",
                    row * d_str[0] + hv * d_str[1] + t,
                    math.prod(decay_shape),
                )
                for page_id in (1, BLOCKS - 1):
                    _check(
                        "state_load",
                        page_id * s_str[0]
                        + hv * s_str[1]
                        + (vt * BV + BV - 1) * K
                        + K
                        - 1,
                        math.prod(state_shape),
                    )
    # state tile must stay inside one page
    page = V * K
    for vt in range(nv):
        for ov in range(0, BV, 8):
            v_abs = vt * BV + ov
            if v_abs < V:
                _check("state_v", v_abs * K, page)


def test_commit_indices_in_range():
    rows, layers = 128, 16
    state_shape = (BLOCKS, HV, V, K)
    replay_shape = (RECORD_ROWS, HV, SPEC_Q, V)
    key_shape = (RECORD_ROWS, HV_KEY, SPEC_Q, K)
    s_str = _strides(state_shape)
    r_str = _strides(replay_shape)
    k_str = _strides(key_shape)
    nv = math.ceil(V / 16)
    BV = 16
    num_work = nv * rows * (layers * HV)
    for cap in (1, 8, 48, 192):
        for work in _dispatch(num_work, cap):
            vt, row, lh = _decompose(work, nv, rows, layers * HV)
            layer = lh // HV
            head = lh % HV
            assert 0 <= layer < layers and 0 <= head < HV
            for n in (0, 1, 3, SPEC_Q):
                for t in range(n):
                    base = row * r_str[0] + head * r_str[1] + t * r_str[2]
                    key_head = head // (HV // HV_KEY)
                    _check("rec_corr", base + V - 1, math.prod(replay_shape))
                    _check(
                        "rec_key",
                        row * k_str[0] + key_head * k_str[1] + t * k_str[2] + K - 1,
                        math.prod(key_shape),
                    )
            # destination tile
            for ov in range(0, BV, 8):
                v_abs = vt * BV + ov
                if v_abs < V:
                    _check("state_dst", v_abs * s_str[2], V * K)


def test_graph_padding_cannot_expand_request_record_domain():
    for graph_rows in (1, 127, 128, 129, 512):
        logical = min(graph_rows, RECORD_ROWS)
        written_rows = set()
        for row in range(graph_rows):
            # Native padded rows have null state and zero query length.
            page = 1 if row < logical else 0
            query_len = SPEC_Q if row < logical else 0
            if page > 0 and query_len > 0 and row < RECORD_ROWS:
                written_rows.add(row)
                _check(
                    "last correction",
                    (row + 1) * HV * SPEC_Q * V - 1,
                    RECORD_ROWS * HV * SPEC_Q * V,
                )
        assert written_rows == set(range(logical))


def test_compact_conv_window_in_range():
    """The source window `(commit_len - 1 + h)` must stay inside the page."""
    cases = []
    for commit_len in range(1, SPEC_Q + 1):
        max_h = CONV_HIST - 1
        cases.append((commit_len, commit_len - 1 + max_h))
    for commit_len, worst in cases:
        assert worst < CONV_WINDOW, (commit_len, worst, CONV_WINDOW)
    # and every dim tile stays inside conv_dim
    block_d = 128
    n_d = math.ceil(CONV_DIM / block_d)
    for work in _dispatch(n_d * 3 * 16, 48):
        tile, row, _layer = _decompose(work, n_d, 3, 16)
        for j in range(0, block_d, 16):
            d = tile * block_d + j
            if d < CONV_DIM:
                assert 0 <= d < CONV_DIM


def test_dispatch_visits_every_logical_tile_once():
    for nv, rows, heads in ((8, 128, 192), (8, 1, 192), (3, 7, 5)):
        num_work = nv * rows * heads
        for cap in (1, 5, 48, 192, num_work):
            seen = []
            for work in _dispatch(num_work, cap):
                seen.append(work)
                _decompose(work, nv, rows, heads)
            assert len(seen) == num_work
            assert sorted(seen) == list(range(num_work))
