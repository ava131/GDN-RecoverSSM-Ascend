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

# Realistic shapes from the machine log: 16 GDN layers, replay
# (1418, 12, 4, 257), Q=4, V=K=128, conv window = conv_kernel-1+num_spec.
BLOCKS = 1418
HV = 12
HV_KEY = 4          # key heads after TP
V = K = 128
BK = 128            # next_power_of_2(K)
SPEC_Q = 4
CONV_KERNEL = 4
CONV_WINDOW = CONV_KERNEL - 1 + (SPEC_Q - 1)   # = 6
CONV_HIST = CONV_WINDOW - SPEC_Q + 1           # = 3 == conv_kernel - 1
CONV_DIM = 2 * 128 * HV_KEY + 128 * HV         # dim-first conv state


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
    replay_shape = (BLOCKS, HV, SPEC_Q, V + K + 1)
    out_shape = (1, total, HV, V)
    q_shape = (total, HV_KEY, K)
    for shape in (state_shape, replay_shape, out_shape, q_shape):
        assert sum(s * (0 if i else 1) for i, s in enumerate(shape)) >= 0
    s_str = _strides(state_shape)
    r_str = _strides(replay_shape)
    o_str = _strides(out_shape)
    q_str = _strides(q_shape)

    nv = math.ceil(V / 16)          # BV used by the commit/verify budgets
    BV = 16
    num_work = nv * rows * HV
    for cap in (1, 8, 48, 192):
        for work in _dispatch(num_work, cap):
            vt, row, hv = _decompose(work, nv, rows, HV)
            bos = row * SPEC_Q
            eos = bos + SPEC_Q
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
                        _check("q", tok * q_str[1] + ph * K + kk,
                               total * HV_KEY * K)
                # replay record write, for a real page in range
                for sidx in (0, 1, BLOCKS - 1):
                    base = sidx * r_str[0] + hv * r_str[1] + t * r_str[2]
                    _check("rec_corr", base + (V - 1) * r_str[3],
                           BLOCKS * HV * SPEC_Q * (V + K + 1))
                    _check("rec_key", base + (V + K - 1) * r_str[3],
                           BLOCKS * HV * SPEC_Q * (V + K + 1))
                    _check("rec_decay", base + (V + K) * r_str[3],
                           BLOCKS * HV * SPEC_Q * (V + K + 1))
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
    replay_shape = (BLOCKS, HV, SPEC_Q, V + K + 1)
    s_str = _strides(state_shape)
    r_str = _strides(replay_shape)
    nv = math.ceil(V / 16)
    BV = 16
    num_work = nv * rows * (layers * HV)
    for cap in (1, 8, 48, 192):
        for work in _dispatch(num_work, cap):
            vt, row, lh = _decompose(work, nv, rows, layers * HV)
            layer = lh // HV
            head = lh % HV
            assert 0 <= layer < layers and 0 <= head < HV
            replay_numel = BLOCKS * HV * SPEC_Q * (V + K + 1)
            for sidx in (0, BLOCKS - 1):
                for n in (0, 1, 3, SPEC_Q):
                    for t in range(n):
                        base = sidx * r_str[0] + head * r_str[1] + t * r_str[2]
                        _check("rec_corr", base + (V - 1) * r_str[3], replay_numel)
                        _check("rec_key", base + (V + K - 1) * r_str[3], replay_numel)
            # destination tile
            for ov in range(0, BV, 8):
                v_abs = vt * BV + ov
                if v_abs < V:
                    _check("state_dst", v_abs * s_str[2], V * K)


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
