# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Ported for Ascend A2 (910B) from upstream vLLM PR #61309 (`recoverssm_gdn.py`
# @ `bf0219d8`) plus the model-independent shared ops of
# `vllm/model_executor/layers/mamba/ops/recoverssm.py`
# (`prepare_commit_plan_kernel`, `compact_conv_state_kernel`), and the small
# helpers of `.../mamba/recoverssm_utils.py` (`recoverssm_require`,
# `recoverssm_ptr_table`) inlined as `_require` / `_ptr_table`.
#
# Ascend adaptations, and nothing else:
#   * every UB-resident tile is derived at runtime from `get_ub_size_bytes()`
#     with a conservative FP32 budget and an explicit fit assertion;
#   * no `is_950()` branches and no grouped-head fast path (A2 always runs the
#     generic kernels);
#   * `spec_query_len` is bounded by `MAX_SPEC_QUERY_LEN` and the per-token loop
#     is a `tl.static_range`;
#   * the GDN gate/beta are precomputed by the layer
#     (`DeviceOperator.fused_gdn_gating`) and enter the verify kernel as
#     `g` (log-space decay) and `beta` (= sigmoid(b)).
# The math, the `[corr(V), k_hat(K), decay(1)]` record layout, the null-row
# behaviour, the commit fold and the ALIGN/BOUNDARY handling are upstream's.

"""RecoverSSM for Gated DeltaNet (Qwen-style GDN) on Ascend NPUs: speculative
verify from one checkpoint without per-draft state snapshots, and accepted-state
recovery after sampling.

GDN port of the Kimi-K3 KDA RecoverSSM path. Differences from KDA: the gate is a
scalar per value head (g = -exp(A_log) * softplus(a + dt_bias)), q/k heads are
grouped over value heads (HV // H), and q/k are L2-normalised inside the kernel
when requested. The verify step caches, per token, the delta-rule correction
c_t = beta_t * (v_t - S'_t k_t) (V floats), the normalised key (K floats) and the
decay exp(g_t) (1 float) in one fp32 "replay" record. Recovery folds the accepted
prefix: S_n = exp(sum_t g_t) S_0 + sum_t exp(sum_{s>t} g_s) c_t k_t^T and writes
the checkpoint once (plus the block-boundary state in align mode).

Ascend A2 (910B, Triton-Ascend 3.6.0) specifics:
  * The gate is not recomputed here. The caller runs the fused GDN gating once
    per step and passes `g` (log-space decay, shape [1, T, HV] or [T, HV]) and
    `beta` (already sigmoid(b)). The kernel only does `decay = exp(g)`.
  * q/k may arrive pre-normalised (`qk_already_normalized=True`, the default,
    matching FLA `l2norm_fwd` with eps 1e-6); the in-kernel L2 path is kept for
    the `False` case with the same epsilon.
  * Tiles (`BV`, the conv `BLOCK_D`) are selected at runtime from the Unified
    Buffer size with an explicit budget assertion; see `_ub_budget_floats`.
  * No A5 grouped-head path: A2 always uses the generic commit kernel.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes

try:  # natural bound for a per-tile loop on this device
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
except ImportError:  # pragma: no cover - older drops
    get_vectorcore_num = None


def _launch_cap() -> int:
    """Upper bound on physical programs for one launch.

    Triton-Ascend maps a logical grid onto cores itself, but relying on that for
    very large grids is a version-specific assumption; its own migration guide
    recommends a bounded 1-D grid with an in-kernel loop. Cap at the vector-core
    count, which is what vllm-ascend does for its tiled kernels.
    """
    if get_vectorcore_num is None:
        return 48
    try:
        return max(1, int(get_vectorcore_num()))
    except Exception:
        return 48

# ---------------------------------------------------------------------------
# Configuration constants (Ascend A2)
# ---------------------------------------------------------------------------

# Speculative-window capacity this module is compiled and budgeted for. Q is a
# `tl.constexpr` (the verify loop is a `tl.static_range`), so every extra token
# multiplies the compiled kernel size; the UB budget below was sized for this
# bound and `gdn_recoverssm_verify` refuses larger windows loudly.
MAX_SPEC_QUERY_LEN = 8

# Value-tile caps, kept at the upstream values (verify 32, commit 16) and then
# clamped by the UB budget. Upstream's comment for the 16-row commit tile: the
# commit streams each state once, and smaller tiles keep more of them in flight.
RECOVERSSM_VERIFY_VALUE_TILE = 32
RECOVERSSM_COMMIT_VALUE_TILE = 16

# Upstream `compact_conv_state_kernel` launches with BLOCK_D=256. We never grow
# past it on A2 (growth is a performance knob to revisit on hardware) and shrink
# it when one history tile would not fit the UB.
RECOVERSSM_CONV_TILE = 256

# Everything that lives in the UB is FP32 here, so budget in 4-byte floats.
RECOVERSSM_FP32_BYTES = 4

# Simultaneously live BV x BK FP32 matrices per program. These are conservative
# hand counts of the peak live set, not compiler-reported buffer counts (the KDA
# RecoverSSM Ascend port budgets the same way).
#   verify: `state`, `new_state` (state * decay, then + corr * k) and the
#           `tl.sum(new_state * k[None, :], 1)` / `tl.sum(state * q[None, :], 1)`
#           operand, which the backend materialises as a full tile.
#   commit: `h`, the `h * decay` product and the `corr[:, None] * k[None, :]`
#           outer product consumed by `tl.fma`.
RECOVERSSM_VERIFY_UB_MATRICES = 3
RECOVERSSM_COMMIT_UB_MATRICES = 3

# Simultaneously live [BLOCK_D, BLOCK_HISTORY] conv tiles in align mode: the
# final-history tile and the boundary tile are both live across the boundary
# store. Budgeted at 4 bytes/element although conv state is usually BF16/FP16.
RECOVERSSM_CONV_UB_TILES = 2

# FP32 vector rows reserved on top of the matrices for q/k/v/corr/o, the gate
# and beta scalars, and their broadcast temporaries.
RECOVERSSM_UB_VECTOR_ROWS = 4


# ---------------------------------------------------------------------------
# Inlined upstream helpers (recoverssm_utils)
# ---------------------------------------------------------------------------


def _require(cond: bool, msg: str) -> None:
    """Argument validation that raises ``ValueError`` before any kernel launch.

    Inlined from ``recoverssm_utils.recoverssm_require(cond, msg, "GDN
    RecoverSSM")`` so this module does not depend on the upstream RecoverSSM
    helper modules. Same message format as upstream.
    """
    if not cond:
        raise ValueError(f"GDN RecoverSSM: {msg}")


def _ptr_table(tensors: Sequence[torch.Tensor], device) -> torch.Tensor:
    """int64 table of the tensors' data pointers, keeping each uint64 bit pattern.

    Inlined from ``recoverssm_utils.recoverssm_ptr_table``. Device pointers with
    the top bit set overflow a plain int64 conversion; the kernels only cast the
    value back to a pointer.
    """
    ptrs = [t.data_ptr() for t in tensors]
    ptrs = [p if p < (1 << 63) else p - (1 << 64) for p in ptrs]
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


# ---------------------------------------------------------------------------
# UB budget (Ascend A2)
# ---------------------------------------------------------------------------


def _ub_budget_floats() -> int:
    """FP32 float capacity of the current device's Unified Buffer.

    Deliberately a function, never a module constant: ``get_ub_size_bytes()``
    asserts that the device-properties cache was initialized (the Ascend worker
    calls ``init_device_properties_triton()`` during device init) and reports a
    per-device value. Calling it lazily keeps this module importable on hosts
    without an NPU (import-time device queries would abort) and makes the tile
    choice follow the real UB of the device that runs the kernel.
    """
    return get_ub_size_bytes() // RECOVERSSM_FP32_BYTES


def _verify_ub_floats(block_v: int, block_k: int) -> int:
    """Conservative FP32 UB footprint of one verify program, in floats.

    ``BV * BK * live_matrices`` FP32 elements for the state/recurrence tiles,
    plus the per-token vector reserve, plus the record tile (corr BV + key BK +
    decay 1) and the output tile (BV) that the kernel stores each token. The
    caller multiplies by ``RECOVERSSM_FP32_BYTES`` to compare against the UB.
    """
    matrices = block_v * block_k * RECOVERSSM_VERIFY_UB_MATRICES
    vectors = RECOVERSSM_UB_VECTOR_ROWS * (block_v + block_k)
    record = block_v + block_k + 1
    output = block_v
    return matrices + vectors + record + output


def _assert_verify_ub_fit(block_v: int, block_k: int) -> None:
    """Fail loudly if one verify program cannot fit the UB.

    A tile that silently overflows the Unified Buffer corrupts state instead of
    crashing, so the launch is refused before it happens. The expression is the
    documented budget: BV*BK*4*live_matrices + record tile + output tile (plus
    the FP32 vector reserve) <= UB.
    """
    ub_bytes = get_ub_size_bytes()
    need = _verify_ub_floats(block_v, block_k) * RECOVERSSM_FP32_BYTES
    assert need <= ub_bytes, (
        "GDN RecoverSSM verify tile does not fit the UB: "
        f"BV*BK*4*{RECOVERSSM_VERIFY_UB_MATRICES} + record + output + vectors "
        f"= {need} B > UB {ub_bytes} B (BV={block_v}, BK={block_k})"
    )


def _verify_value_tile(block_k: int, value_dim: int) -> int:
    """Select BV for the verify kernel: upstream's 32-row cap, UB-clamped.

    Starts at ``min(next_power_of_2(V), 32)`` exactly as upstream and halves
    (keeping it a power of two) until one program fits the UB budget. ``BK`` is
    fixed by the key width, so a key tile that does not fit even with BV=1 trips
    the assertion in ``_assert_verify_ub_fit`` instead of running.
    """
    tile = min(triton.next_power_of_2(value_dim), RECOVERSSM_VERIFY_VALUE_TILE)
    budget = _ub_budget_floats()
    while tile > 1 and _verify_ub_floats(tile, block_k) > budget:
        tile //= 2
    _assert_verify_ub_fit(tile, block_k)
    return tile


def _commit_ub_floats(block_v: int, block_k: int) -> int:
    """Conservative FP32 UB footprint of one commit program, in floats.

    The commit keeps one [BV, BK] accumulator (`h`) live across the accepted
    prefix plus the two fma operands, and per-token corr/key vectors. It writes
    the state tile once, so there is no separate output tile beyond `h`.
    """
    matrices = block_v * block_k * RECOVERSSM_COMMIT_UB_MATRICES
    vectors = RECOVERSSM_UB_VECTOR_ROWS * (block_v + block_k)
    return matrices + vectors


def _assert_commit_ub_fit(block_v: int, block_k: int) -> None:
    """Fail loudly if one commit program cannot fit the UB (see verify)."""
    ub_bytes = get_ub_size_bytes()
    need = _commit_ub_floats(block_v, block_k) * RECOVERSSM_FP32_BYTES
    assert need <= ub_bytes, (
        "GDN RecoverSSM commit tile does not fit the UB: "
        f"BV*BK*4*{RECOVERSSM_COMMIT_UB_MATRICES} + vectors "
        f"= {need} B > UB {ub_bytes} B (BV={block_v}, BK={block_k})"
    )


def _commit_value_tile(block_k: int, value_dim: int) -> int:
    """Select BV for the commit kernel: upstream's 16-row cap, UB-clamped."""
    tile = min(triton.next_power_of_2(value_dim), RECOVERSSM_COMMIT_VALUE_TILE)
    budget = _ub_budget_floats()
    while tile > 1 and _commit_ub_floats(tile, block_k) > budget:
        tile //= 2
    _assert_commit_ub_fit(tile, block_k)
    return tile


def _conv_ub_floats(block_d: int, block_history: int) -> int:
    """Conservative FP32 UB footprint of one conv-compaction program, in floats.

    One [BLOCK_D, BLOCK_HISTORY] source tile plus, in align mode, the boundary
    tile that is loaded before the final store. Budgeted at 4 B/element even
    though the conv state is normally BF16/FP16.
    """
    return block_d * block_history * RECOVERSSM_CONV_UB_TILES


def _assert_conv_ub_fit(block_d: int, block_history: int) -> None:
    """Fail loudly if one conv-compaction program cannot fit the UB."""
    ub_bytes = get_ub_size_bytes()
    need = _conv_ub_floats(block_d, block_history) * RECOVERSSM_FP32_BYTES
    assert need <= ub_bytes, (
        "GDN RecoverSSM conv tile does not fit the UB: "
        f"BLOCK_D*BLOCK_HISTORY*4*{RECOVERSSM_CONV_UB_TILES} "
        f"= {need} B > UB {ub_bytes} B "
        f"(BLOCK_D={block_d}, BLOCK_HISTORY={block_history})"
    )


def _conv_dim_tile(block_history: int) -> int:
    """Select BLOCK_D for conv compaction: upstream's 256, UB-clamped.

    ``BLOCK_HISTORY`` is not tunable here: the kernel covers the whole conv
    history in one tile and its mask is written against that tile, so the budget
    is spent on BLOCK_D only.
    """
    tile = RECOVERSSM_CONV_TILE
    budget = _ub_budget_floats()
    while tile > 1 and _conv_ub_floats(tile, block_history) > budget:
        tile //= 2
    _assert_conv_ub_fit(tile, block_history)
    return tile


# ---------------------------------------------------------------------------
# Verify kernel
# ---------------------------------------------------------------------------


@triton.jit
def _gdn_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    replay_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    total_tokens,
    scale,
    num_rows,
    num_work,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_replay_block,
    stride_replay_head,
    stride_replay_pos,
    stride_replay_dim,
    stride_out_token,
    stride_qsl,
    stride_si,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_VALUE_TILES: tl.constexpr,
):
    # Bounded physical dispatch. Logical work is
    #   NUM_VALUE_TILES x num_rows x HV
    # and the launch uses at most min(logical, cores) programs, each looping over
    # its share. A 3-D grid would ask the runtime for the whole logical product;
    # this keeps the physical grid bounded for every supported batch while each
    # valid tile still runs exactly once.
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    for work in range(pid, num_work, grid):
        pid_v = work % NUM_VALUE_TILES
        rest = work // NUM_VALUE_TILES
        pid_b = rest % num_rows
        pid_hv = rest // num_rows
        pid_h = pid_hv // (HV // H)
        bos = tl.load(query_start_loc_ptr + pid_b * stride_qsl).to(tl.int64)
        eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_qsl).to(tl.int64)
        query_len = eos - bos
        state_idx = tl.load(state_indices_ptr + pid_b * stride_si).to(tl.int64)
        offs_k = tl.arange(0, BK)
        offs_v = pid_v * BV + tl.arange(0, BV)
        mask_k = offs_k < K
        mask_v = offs_v < V
        mask_state = mask_v[:, None] & mask_k[None, :]
        if state_idx <= null_block_id or state_idx >= NUM_BLOCKS:
            # Null rows (and out-of-range page ids from a stale block table) emit
            # zeros and never touch the replay records. This stays a conditional
            # block rather than an early return: one physical program owns
            # several logical tiles.
            for t in tl.static_range(SPEC_QUERY_LEN):
                tok_null = bos + t
                tl.store(
                    out_ptr + tok_null * stride_out_token + pid_hv * V + offs_v,
                    tl.zeros([BV], dtype=tl.float32),
                    mask=(t < query_len) & (tok_null < total_tokens) & mask_v,
                )
        else:
            state = tl.load(
                state_ptr
                + state_idx * stride_state_block
                + pid_hv * stride_state_head
                + offs_v[:, None] * stride_state_v
                + offs_k[None, :] * stride_state_k,
                mask=mask_state,
                other=0.0,
            ).to(tl.float32)
            rec_base = (
                replay_ptr
                + state_idx * stride_replay_block
                + pid_hv * stride_replay_head
            )
            for t in tl.static_range(SPEC_QUERY_LEN):
                tok = bos + t
                # The window may be shorter than the token tensor, never longer:
                # ``spec_query_start_loc`` can describe graph-padded rows.
                valid = (t < query_len) & (tok < total_tokens)
                q = tl.load(
                    q_ptr + tok * stride_q_token + pid_h * K + offs_k,
                    mask=valid & mask_k,
                    other=0.0,
                ).to(tl.float32)
                k = tl.load(
                    k_ptr + tok * stride_k_token + pid_h * K + offs_k,
                    mask=valid & mask_k,
                    other=0.0,
                ).to(tl.float32)
                v = tl.load(
                    v_ptr + tok * stride_v_token + pid_hv * V + offs_v,
                    mask=valid & mask_v,
                    other=0.0,
                ).to(tl.float32)
                g = tl.load(
                    g_ptr + tok * stride_g_token + pid_hv, mask=valid, other=0.0
                ).to(tl.float32)
                beta = tl.load(
                    beta_ptr + tok * stride_beta_token + pid_hv,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                if USE_QK_L2NORM:
                    # Same epsilon as FLA l2norm_fwd so both paths agree.
                    q = q * tl.rsqrt(tl.sum(q * q) + 1e-6)
                    k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
                q = q * scale
                decay = tl.exp(g)
                new_state = state * decay
                corr = (v - tl.sum(new_state * k[None, :], 1)) * beta
                new_state = new_state + corr[:, None] * k[None, :]
                state = tl.where(valid, new_state, state)
                o = tl.sum(state * q[None, :], 1)
                tl.store(
                    out_ptr + tok * stride_out_token + pid_hv * V + offs_v,
                    o.to(out_ptr.dtype.element_ty),
                    mask=valid & mask_v,
                )
                rec = rec_base + t * stride_replay_pos
                tl.store(rec + offs_v * stride_replay_dim, corr, mask=valid & mask_v)
                if pid_v == 0:
                    tl.store(
                        rec + (V + offs_k) * stride_replay_dim, k, mask=valid & mask_k
                    )
                    # the decay exactly as multiplied here, so the commit replays
                    # this float
                    tl.store(rec + (V + K) * stride_replay_dim, decay, mask=valid)


def gdn_recoverssm_verify(
    g: torch.Tensor,
    beta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    replay_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    scale: float | None = None,
    qk_already_normalized: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a GDN speculative window against its checkpoint without modifying the
    checkpoint.

    Ascend signatures: ``g`` (log-space decay) and ``beta`` (already sigmoid) come
    from the layer's ``fused_gdn_gating`` in shape [1, T, HV] or [T, HV]; q, k:
    [1, T, H, K]; v: [1, T, HV, V]; checkpoint_state: [blocks, HV, V, K];
    replay_cache: [blocks, HV, spec_query_len, V + K + 1] fp32. ``qk_already_normalized``
    selects the pre-normalised (FLA ``l2norm_fwd``, eps 1e-6) path, which is the
    default; pass ``False`` to normalise q/k inside the kernel with the same
    epsilon. Returns out [1, T, HV, V].
    """
    _require(
        q.ndim == 4 and q.shape[0] == 1, "q must have shape [1, tokens, heads, dim]"
    )
    _, total, H, K = q.shape
    _require(k.shape == q.shape, "q and k shapes differ")
    _require(
        v.ndim == 4 and tuple(v.shape[:2]) == (1, total),
        "v must have shape [1, tokens, value heads, dim]",
    )
    HV, V = v.shape[-2], v.shape[-1]
    _require(
        H > 0 and HV >= H and HV % H == 0,
        "value heads must be a positive multiple of the q/k heads",
    )
    _require(
        g.numel() == total * HV and beta.numel() == total * HV,
        "gate or beta shape is incompatible",
    )
    g2 = g.reshape(total, HV)
    beta2 = beta.reshape(total, HV)
    _require(
        g2.stride(1) == 1 and beta2.stride(1) == 1,
        "gate and beta heads must be contiguous",
    )
    # Ascend: the window is a compile-time constant and every kernel in this
    # module is budgeted for at most MAX_SPEC_QUERY_LEN tokens.
    _require(
        0 < spec_query_len <= MAX_SPEC_QUERY_LEN,
        f"spec_query_len must be in [1, {MAX_SPEC_QUERY_LEN}]",
    )
    _require(
        all(t_.stride(-1) == 1 and t_.stride(-2) == K for t_ in (q, k)),
        "q and k heads must be contiguous",
    )
    _require(v.stride(-1) == 1 and v.stride(-2) == V, "v heads must be contiguous")
    _require(checkpoint_state.ndim == 4, "checkpoint must be four-dimensional")
    nb = checkpoint_state.shape[0]
    _require(
        tuple(checkpoint_state.shape[1:]) == (HV, V, K),
        "checkpoint shape is incompatible",
    )
    _rec = (nb, HV, spec_query_len, V + K + 1)
    _require(tuple(replay_cache.shape) == _rec, f"replay buffer needs shape {_rec}")
    _require(replay_cache.dtype == torch.float32, "replay buffer must use float32")
    _require(state_indices.ndim == 1, "state indices must be one-dimensional")
    _require(
        query_start_loc.ndim == 1
        and query_start_loc.shape[0] == state_indices.shape[0] + 1,
        "query metadata is incompatible",
    )
    _require(
        total <= state_indices.shape[0] * spec_query_len,
        "speculative decode input exceeds its activation capacity",
    )
    _dev = q.device
    _require(
        all(
            t_.device == _dev
            for t_ in (
                k,
                v,
                g,
                beta,
                checkpoint_state,
                replay_cache,
                query_start_loc,
                state_indices,
            )
            if t_ is not None
        )
        and (out is None or out.device == _dev),
        "inputs must be on the same device",
    )
    if scale is None:
        scale = K**-0.5
    if out is None:
        out = torch.empty(1, total, HV, V, dtype=v.dtype, device=v.device)
    _require(
        tuple(out.shape) == (1, total, HV, V) and out.stride()[2:] == (V, 1),
        "output shape or layout is incompatible",
    )
    batch = state_indices.shape[0]
    if total == 0 or batch == 0:
        return out
    BK = triton.next_power_of_2(K)
    # Ascend: upstream hardcodes BV = min(next_power_of_2(V), 32). The cap is
    # kept, but the tile is additionally clamped so one program's FP32 tiles fit
    # the UB (see _verify_value_tile / _assert_verify_ub_fit).
    BV = _verify_value_tile(BK, V)
    # Bounded physical dispatch: at most one program per vector core, each
    # walking several logical tiles.
    num_value_tiles = triton.cdiv(V, BV)
    num_work = num_value_tiles * batch * HV
    grid = (min(num_work, _launch_cap()),)
    _gdn_recoverssm_verify_kernel[grid](
        q,
        k,
        v,
        g2,
        beta2,
        checkpoint_state,
        replay_cache,
        out,
        query_start_loc,
        state_indices,
        total,
        scale,
        batch,
        num_work,
        NULL_BLOCK_ID,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        g2.stride(0),
        beta2.stride(0),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        replay_cache.stride(0),
        replay_cache.stride(1),
        replay_cache.stride(2),
        replay_cache.stride(3),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SPEC_QUERY_LEN=spec_query_len,
        USE_QK_L2NORM=not qk_already_normalized,
        NUM_BLOCKS=nb,
        NUM_VALUE_TILES=num_value_tiles,
        num_warps=4,
        num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# Commit plan + conv compaction (shared ops, upstream verbatim)
# ---------------------------------------------------------------------------


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    # Upstream `prepare_commit_plan_kernel`, renamed only. Per speculative row it
    # resolves the commit length and the destination state blocks (final and, in
    # align mode, the block-boundary block) from the accepted counts and the
    # block table; null rows and zero accepted lengths become no-op plan entries.
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(
        tl.int64
    )
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(
            request_indices_ptr + spec_idx * stride_request_indices
        ).to(tl.int64)
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(
        tl.int32
    )
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(
            tl.int32
        )
        final_num_computed = num_computed + commit_len
        # The committed state belongs to the block holding the last committed
        # token. A window ending exactly on a boundary stays in that block; the
        # next block may still be unallocated.
        # The destination must be a column the request actually owns. The
        # manager only guarantees ceil(tokens / block) blocks (align mode ignores
        # the lookahead and num_speculative_blocks is 0 here), so
        # ``(computed + accepted) // block`` can point one column past the
        # allocated range when the window ends exactly on a boundary
        # (C=124, A=4, B=128 -> column 1, which may still be a null page).
        # The block holding the last committed token always exists.
        final_state_col = tl.minimum(
            tl.maximum(final_num_computed - 1, 0) // mamba_block_size,
            block_table_width - 1,
        )
        final_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + final_state_col * stride_block_table_col
        ).to(tl.int64)
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=crosses_boundary,
            other=null_block_id,
        ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
    )


@triton.jit
def _compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    num_rows,
    num_work,
    conv_dim,
    conv_history_len,
    stride_state_indices,
    BLOCK_D: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_D_TILES: tl.constexpr,
):
    # Bounded physical dispatch over (dim tile, row, layer). Each program walks
    # several logical tiles, so every guard below is a per-tile no-op rather than
    # a function return.
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    for work in range(pid, num_work, grid):
        pid_d = work % NUM_D_TILES
        rest = work // NUM_D_TILES
        pid_b = rest % num_rows
        pid_l = rest // num_rows
        source_state_idx = tl.load(
            state_indices_ptr + pid_b * stride_state_indices
        ).to(tl.int64)
        commit_len = tl.load(commit_lens_ptr + pid_b)
        final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
        row_ok = (
            (source_state_idx > null_block_id)
            & (commit_len != 0)
            & (final_state_idx > null_block_id)
            & (final_state_idx < NUM_BLOCKS)
        )
        if row_ok:
            boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(
                tl.int64
            )
            boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

            base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
            block_stride = tl.load(conv_state_block_strides_ptr + pid_l)
            dim_stride = tl.load(conv_state_dim_strides_ptr + pid_l)
            token_stride = tl.load(conv_state_token_strides_ptr + pid_l)
            conv_state_ptr = base_addr.to(
                tl.pointer_type(conv_state_ref_ptr.dtype.element_ty)
            )
            source_ptr = conv_state_ptr + source_state_idx * block_stride
            final_ptr = conv_state_ptr + final_state_idx * block_stride

            offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
            offs_h = tl.arange(0, BLOCK_HISTORY)
            mask = (offs_d[:, None] < conv_dim) & (
                offs_h[None, :] < conv_history_len
            )
            final_values = tl.load(
                source_ptr
                + offs_d[:, None] * dim_stride
                + (commit_len - 1 + offs_h[None, :]) * token_stride,
                mask=mask,
            )
            if ALIGN_MODE:
                boundary_ok = (boundary_state_idx > null_block_id) & (
                    boundary_state_idx < NUM_BLOCKS
                )
                boundary_values = tl.load(
                    source_ptr
                    + offs_d[:, None] * dim_stride
                    + (boundary_recovery_len - 1 + offs_h[None, :]) * token_stride,
                    mask=mask & boundary_ok,
                )
                boundary_ptr = conv_state_ptr + boundary_state_idx * block_stride
                tl.store(
                    boundary_ptr
                    + offs_d[:, None] * dim_stride
                    + offs_h[None, :] * token_stride,
                    boundary_values,
                    mask=mask & boundary_ok,
                )
            tl.store(
                final_ptr
                + offs_d[:, None] * dim_stride
                + offs_h[None, :] * token_stride,
                final_values,
                mask=mask,
            )


@triton.jit
def _commit_gdn_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    replay_ref_ptr,
    replay_base_addrs_ptr,
    replay_block_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    num_rows,
    num_work,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_replay_head,
    stride_replay_pos,
    stride_replay_dim,
    stride_si,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_VALUE_TILES: tl.constexpr,
):
    # One physical program handles several logical (value tile, row, layer*head)
    # tiles. Every early exit below is a per-tile no-op, never a function return:
    # a padding tile must not cancel the remaining tiles of the same program.
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    for work in range(pid, num_work, grid):
        pid_v = work % NUM_VALUE_TILES
        rest = work // NUM_VALUE_TILES
        pid_b = rest % num_rows
        pid_lh = rest // num_rows
        pid_l = pid_lh // NUM_HEADS
        pid_h = pid_lh % NUM_HEADS
        src = tl.load(state_indices_ptr + pid_b * stride_si).to(tl.int64)
        n = tl.load(commit_lens_ptr + pid_b).to(tl.int32)
        final_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
        row_ok = (
            (src > null_block_id)
            & (src < NUM_BLOCKS)
            & (n > 0)
            & (final_idx > null_block_id)
        )
        if row_ok:
            dst_idx = final_idx
            # Boundary state is captured DURING the same fold. It must never be a
            # second pass over the source page: when the final state lands in the
            # window's source column, a second launch would read the overwritten
            # checkpoint (C=126, A=3, boundary_len=2 gave 2.25 instead of 4).
            b_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
            b_len = tl.load(boundary_recovery_lens_ptr + pid_b).to(tl.int32)
            # Chained ``and`` is rejected by triton-ascend: fold into one term.
            store_boundary = (
                (b_idx > null_block_id) & (b_idx != final_idx) & (b_len > 0)
            )
            state_ptr = tl.load(state_base_addrs_ptr + pid_l).to(
                tl.pointer_type(state_ref_ptr.dtype.element_ty)
            )
            sbs = tl.load(state_block_strides_ptr + pid_l)
            replay_ptr = tl.load(replay_base_addrs_ptr + pid_l).to(
                tl.pointer_type(replay_ref_ptr.dtype.element_ty)
            )
            rbs = tl.load(replay_block_strides_ptr + pid_l)
            rec_base = replay_ptr + src * rbs + pid_h * stride_replay_head
            offs_k = tl.arange(0, BK)
            offs_v = pid_v * BV + tl.arange(0, BV)
            mask_k = offs_k < K
            mask_v = offs_v < V
            mask_state = mask_v[:, None] & mask_k[None, :]
            sp = offs_v[:, None] * stride_state_v + offs_k[None, :] * stride_state_k
            s0 = tl.load(
                state_ptr + src * sbs + pid_h * stride_state_head + sp,
                mask=mask_state,
                other=0.0,
            ).to(tl.float32)
            # Start from the ORIGINAL checkpoint and fold the accepted tokens with
            # the same arithmetic the verify step used, so the committed state is
            # bit-for-bit the per-token state. The boundary snapshot and the final
            # state therefore share one starting point.
            h = s0
            b_dst = state_ptr + b_idx * sbs + pid_h * stride_state_head + sp
            for t in range(n):
                rec = rec_base + t * stride_replay_pos
                c = tl.load(rec + offs_v * stride_replay_dim, mask=mask_v, other=0.0)
                kk = tl.load(
                    rec + (V + offs_k) * stride_replay_dim, mask=mask_k, other=0.0
                )
                decay = tl.load(rec + (V + K) * stride_replay_dim)
                h = tl.fma(c[:, None], kk[None, :], h * decay)
                if store_boundary:
                    if t + 1 == b_len:
                        tl.store(
                            b_dst,
                            h.to(state_ref_ptr.dtype.element_ty),
                            mask=mask_state,
                        )
            tl.store(
                state_ptr + dst_idx * sbs + pid_h * stride_state_head + sp,
                h.to(state_ref_ptr.dtype.element_ty),
                mask=mask_state,
            )


@dataclass
class GDNRecoverSSMCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    replays: tuple[torch.Tensor, ...]
    replay_base_addrs: torch.Tensor
    replay_block_strides: torch.Tensor
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    spec_query_len: int
    num_blocks: int

    @classmethod
    def from_tensors(
        cls,
        conv_states: Sequence[torch.Tensor],
        checkpoints: Sequence[torch.Tensor],
        replays: Sequence[torch.Tensor],
        *,
        spec_query_len: int,
        max_num_reqs: int,
        conv_dim_first: bool = True,
    ) -> "GDNRecoverSSMCommitContext":
        """Bind layer-owned page views and allocate reusable commit-plan buffers.

        Validates every layer's page layout up front (the kernels take raw
        pointers), then records the int64 base-address and block-stride tables
        the batched commit needs.
        """
        _require(len(checkpoints) > 0, "commit requires at least one layer")
        _require(
            len(conv_states) == len(checkpoints) == len(replays),
            "conv, state and replay lists differ",
        )
        # Ascend: the window is a compile-time constant in every kernel below.
        _require(
            0 < spec_query_len <= MAX_SPEC_QUERY_LEN,
            f"spec_query_len must be in [1, {MAX_SPEC_QUERY_LEN}]",
        )
        if not conv_dim_first:
            conv_states = [s.transpose(-1, -2) for s in conv_states]
        ref = checkpoints[0]
        _require(ref.ndim == 4, "checkpoint must be four-dimensional")
        nb, HV, V, K = ref.shape
        _dev = ref.device
        for s in checkpoints:
            _require(
                s.shape == ref.shape
                and s.dtype == ref.dtype
                and s.stride()[1:] == ref.stride()[1:]
                and s.device == _dev,
                "layers need matching checkpoints",
            )
        _rec = (nb, HV, spec_query_len, V + K + 1)
        for r in replays:
            _require(
                tuple(r.shape) == _rec
                and r.dtype == torch.float32
                and r.stride()[1:] == replays[0].stride()[1:]
                and r.device == _dev,
                f"layers need matching float32 replay buffers of shape {_rec}",
            )
        conv_ref = conv_states[0]
        _require(
            conv_ref.ndim == 3 and conv_ref.shape[0] == nb,
            "conv state must be [blocks, dim, window] with the checkpoint's "
            "block count",
        )
        for c in conv_states:
            _require(
                c.shape == conv_ref.shape
                and c.dtype == conv_ref.dtype
                and c.stride()[1:] == conv_ref.stride()[1:]
                and c.device == _dev,
                "layers need matching conv states",
            )
        conv_dim, conv_len = conv_ref.shape[1:]
        hist = conv_len - spec_query_len + 1
        _require(hist > 0, "conv state is shorter than its window")
        dev = ref.device
        addr = lambda ts: _ptr_table(ts, dev)  # noqa: E731
        bstr = lambda ts: torch.tensor(  # noqa: E731
            [t.stride(0) for t in ts], dtype=torch.int64, device=dev
        )
        z = lambda: torch.empty(max_num_reqs, dtype=torch.int32, device=dev)  # noqa: E731
        return cls(
            tuple(conv_states),
            addr(conv_states),
            bstr(conv_states),
            torch.tensor(
                [s.stride(1) for s in conv_states], dtype=torch.int64, device=dev
            ),
            torch.tensor(
                [s.stride(2) for s in conv_states], dtype=torch.int64, device=dev
            ),
            hist,
            tuple(checkpoints),
            addr(checkpoints),
            bstr(checkpoints),
            tuple(replays),
            addr(replays),
            bstr(replays),
            z(),
            z(),
            z(),
            z(),
            spec_query_len,
            nb,
        )

    @classmethod
    def create(
        cls, layers: Sequence[Any], *, spec_query_len: int, max_num_reqs: int
    ) -> "GDNRecoverSSMCommitContext":
        """Build the context from live layers' ``kv_cache`` pages.

        Each layer must expose ``kv_cache`` with exactly three entries: conv
        state (``[blocks, conv_dim, window]`` in the layout selected by
        ``is_conv_state_dim_first()``), recurrent checkpoint
        (``[blocks, HV, V, K]``) and the fp32 replay records
        (``[blocks, HV, Q, V + K + 1]``).

        Ascend: no gate parameters are read here. The commit kernel replays the
        decay stored in the record, so ``A_log``/``dt_bias`` never enter the
        context (the KDA port stacks them because its commit re-runs the gate).
        """
        # Deferred import: same call as upstream, kept out of the module-level
        # import list this port is restricted to. mamba_utils is a vLLM core
        # module that exists in both the 0.30.0 baseline and main.
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        if any(len(layer.kv_cache) != 3 for layer in layers):
            raise ValueError("GDN RecoverSSM pages must contain conv, state and replay")
        return cls.from_tensors(
            [layer.kv_cache[0] for layer in layers],
            [layer.kv_cache[1] for layer in layers],
            [layer.kv_cache[2] for layer in layers],
            spec_query_len=spec_query_len,
            max_num_reqs=max_num_reqs,
            conv_dim_first=is_conv_state_dim_first(),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
        commit_conv: bool = True,
    ) -> None:
        """Fold accepted GDN and convolution inputs into every layer's checkpoint."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        _require(state_indices.ndim == 1, "state indices must be one-dimensional")
        _require(
            batch <= self.commit_lens.shape[0], "commit batch exceeds its plan capacity"
        )
        _require(
            query_start_loc.ndim == 1 and query_start_loc.shape[0] == batch + 1,
            "commit metadata is incompatible",
        )
        _require(
            request_indices is None or request_indices.shape[0] >= batch,
            "request mapping is too short",
        )
        _require(
            num_accepted_tokens.ndim == 1
            and (request_indices is not None or num_accepted_tokens.shape[0] >= batch),
            "accepted-token counts are too short",
        )
        _align = (block_table, num_computed_tokens, mamba_block_size)
        _require(
            all(x is None for x in _align) or all(x is not None for x in _align),
            "align metadata is incomplete",
        )
        _require(
            mamba_block_size is None or mamba_block_size >= self.spec_query_len,
            "align block size must cover one speculative window",
        )
        _require(
            block_table is None or block_table.ndim == 2,
            "block table must be two-dimensional",
        )
        _dev = self.checkpoints[0].device
        _require(
            all(
                t_ is None or t_.device == _dev
                for t_ in (
                    num_accepted_tokens,
                    state_indices,
                    query_start_loc,
                    request_indices,
                    block_table,
                    num_computed_tokens,
                )
            ),
            "commit inputs must be on the same device",
        )
        bt_stride = (0, 0) if block_table is None else block_table.stride()
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            bt_stride[0],
            bt_stride[1],
            0 if num_computed_tokens is None else num_computed_tokens.stride(0),
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        num_layers = len(self.checkpoints)
        if commit_conv:
            conv_ref = self.conv_states[0]
            conv_dim = conv_ref.shape[1]
            # BLOCK_HISTORY must cover the whole window in one tile (the kernel's
            # mask is written against it), so only BLOCK_D is UB-sized.
            block_history = triton.next_power_of_2(self.conv_history_len)
            conv_block_d = _conv_dim_tile(block_history)
            num_d_tiles = triton.cdiv(conv_dim, conv_block_d)
            conv_work = num_d_tiles * batch * num_layers
            _compact_conv_state_kernel[(min(conv_work, _launch_cap()),)](
                conv_ref,
                self.conv_state_base_addrs,
                self.conv_state_block_strides,
                self.conv_state_dim_strides,
                self.conv_state_token_strides,
                state_indices,
                self.commit_lens,
                self.final_state_indices,
                self.boundary_state_indices,
                self.boundary_recovery_lens,
                NULL_BLOCK_ID,
                batch,
                conv_work,
                conv_dim,
                self.conv_history_len,
                state_indices.stride(0),
                BLOCK_D=conv_block_d,
                BLOCK_HISTORY=block_history,
                ALIGN_MODE=block_table is not None,
                NUM_BLOCKS=self.num_blocks,
                NUM_D_TILES=num_d_tiles,
                num_warps=4,
            )
        ref = self.checkpoints[0]
        _, HV, V, K = ref.shape
        BK = triton.next_power_of_2(K)
        # Upstream caps the commit tile at 16 rows (each state is streamed once
        # and smaller tiles keep more of them in flight); the cap is kept and the
        # tile is additionally clamped to the UB budget of this device.
        BV = _commit_value_tile(BK, V)
        rr = self.replays[0]
        align = block_table is not None
        args = (
            ref,
            self.state_base_addrs,
            self.state_block_strides,
            rr,
            self.replay_base_addrs,
            self.replay_block_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            batch,
            num_work,
            ref.stride(1),
            ref.stride(2),
            ref.stride(3),
            rr.stride(1),
            rr.stride(2),
            rr.stride(3),
            state_indices.stride(0),
        )
        num_value_tiles = triton.cdiv(V, BV)
        num_work = num_value_tiles * batch * (num_layers * HV)
        # One launch per layer set: the final and the boundary state come out of a
        # single fold over one checkpoint load. The physical grid stays bounded;
        # each program walks several logical tiles.
        _commit_gdn_state_kernel[(min(num_work, _launch_cap()),)](
            *args,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            NUM_HEADS=HV,
            NUM_BLOCKS=self.num_blocks,
            NUM_VALUE_TILES=num_value_tiles,
            num_warps=4,
            num_stages=2,
        )


__all__ = ["GDNRecoverSSMCommitContext", "gdn_recoverssm_verify"]
