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
# Ascend adaptations:
#   * every UB-resident tile is derived at runtime from `get_ub_size_bytes()`
#     with a conservative FP32 budget and an explicit fit assertion;
#   * no `is_950()` branches and no grouped-head fast path (A2 always runs the
#     generic kernels);
#   * `spec_query_len` is bounded by `MAX_SPEC_QUERY_LEN` and the per-token loop
#     is a `tl.static_range`;
#   * the GDN gate/beta are precomputed by the layer
#     (`DeviceOperator.fused_gdn_gating`) and enter the verify kernel as
#     `g` (log-space decay) and `beta` (= sigmoid(b)).
# The forward recovery arithmetic is retained. Serving records use typed SoA
# planes with one key per GQA key head; packed FP32 buffers remain a constructor
# compatibility path. Boundary and final states are captured during one fold.

"""RecoverSSM for Gated DeltaNet (Qwen-style GDN) on Ascend NPUs: speculative
verify from one checkpoint without per-draft state snapshots, and accepted-state
recovery after sampling.

GDN port of the Kimi-K3 KDA RecoverSSM path. Differences from KDA: the gate is a
scalar per value head (g = -exp(A_log) * softplus(a + dt_bias)), q/k heads are
grouped over value heads (HV // H), and q/k are L2-normalised inside the kernel
when requested. The verify step caches, per token, the delta-rule correction
c_t = beta_t * (v_t - S'_t k_t) (V floats), the normalised key (K floats) and the
decay exp(g_t) (1 float). Typed records retain FP32 correction/decay and
store each normalized GQA key once in its exact input dtype. Recovery folds the accepted
prefix: S_n = exp(sum_t g_t) S_0 + sum_t exp(sum_{s>t} g_s) c_t k_t^T and writes
the checkpoint once (plus the block-boundary state in align mode).

Ascend A2 (910B, Triton-Ascend 3.2.2 deployment) specifics:
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
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes

try:  # natural bound for a per-tile loop on this device
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
except ImportError:  # pragma: no cover - older drops
    get_vectorcore_num = None


logger = init_logger("vllm.gdn_recoverssm_ascend.kernels")

_LAUNCH_CAP: int | None = None


def _launch_cap() -> int:
    """Upper bound on physical programs for one launch.

    Triton-Ascend maps a logical grid onto cores itself, but relying on that for
    very large grids is a version-specific assumption; its own migration guide
    recommends a bounded 1-D grid with an in-kernel loop. Cap at the vector-core
    count, which is what vllm-ascend does for its tiled kernels.
    """
    global _LAUNCH_CAP
    if _LAUNCH_CAP is not None:
        return _LAUNCH_CAP
    if get_vectorcore_num is None:
        _LAUNCH_CAP = 48
        return _LAUNCH_CAP
    try:
        _LAUNCH_CAP = max(1, int(get_vectorcore_num()))
    except Exception:
        _LAUNCH_CAP = 48
    return _LAUNCH_CAP


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


@dataclass(frozen=True)
class ReplayRecordBuffers:
    """Step-owned typed records; keys are shared by grouped value heads.

    Keys loaded from an already-normalized BF16/FP16 tensor retain those exact
    input bits. Correction and decay always remain FP32. A future kernel doing
    normalization internally must allocate FP32 keys instead.
    """

    correction: torch.Tensor
    key: torch.Tensor
    decay: torch.Tensor

    @property
    def shape(self) -> tuple[int, int, int, int]:
        rows, heads, window, value_dim = self.correction.shape
        return (rows, heads, window, value_dim + self.key.shape[-1] + 1)

    @property
    def dtype(self):
        # Logical record arithmetic dtype, retained for diagnostic consumers.
        return torch.float32

    @property
    def device(self):
        return self.correction.device

    @property
    def num_key_heads(self) -> int:
        return int(self.key.shape[1])

    @property
    def key_dtype(self):
        return self.key.dtype

    @property
    def record_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.correction, self.key, self.decay)
        )

    @classmethod
    def allocate(
        cls,
        rows: int,
        num_value_heads: int,
        num_key_heads: int,
        window: int,
        value_dim: int,
        key_dim: int,
        *,
        device,
        key_dtype=torch.float32,
    ):
        _require(
            rows > 0
            and num_key_heads > 0
            and 0 < window <= MAX_SPEC_QUERY_LEN
            and value_dim > 0
            and key_dim > 0
            and num_value_heads > 0
            and num_value_heads % num_key_heads == 0,
            "record head/row geometry is invalid",
        )
        _require(
            key_dtype in (torch.float16, torch.bfloat16, torch.float32),
            "record keys require FP16, BF16 or FP32",
        )
        return cls(
            torch.zeros(
                (rows, num_value_heads, window, value_dim),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (rows, num_key_heads, window, key_dim), dtype=key_dtype, device=device
            ),
            torch.zeros(
                (rows, num_value_heads, window), dtype=torch.float32, device=device
            ),
        )


def record_nbytes(record: torch.Tensor | ReplayRecordBuffers) -> int:
    return (
        record.record_bytes
        if isinstance(record, ReplayRecordBuffers)
        else record.numel() * record.element_size()
    )


def record_geometry_bytes(
    rows: int,
    num_layers: int,
    num_value_heads: int,
    num_key_heads: int,
    window: int,
    key_dim: int,
    value_dim: int,
    key_dtype=torch.float32,
) -> int:
    """Physical bytes, including deduplicated keys and their actual dtype."""
    _require(
        min(
            rows, num_layers, num_value_heads, num_key_heads, window, key_dim, value_dim
        )
        > 0,
        "record geometry must be positive",
    )
    _require(
        num_value_heads % num_key_heads == 0,
        "value heads must be a multiple of key heads",
    )
    _require(
        key_dtype in (torch.float16, torch.bfloat16, torch.float32),
        "record keys require FP16, BF16 or FP32",
    )
    key_bytes = 4 if key_dtype == torch.float32 else 2
    return (
        rows
        * num_layers
        * window
        * (
            num_value_heads * value_dim * 4
            + num_key_heads * key_dim * key_bytes
            + num_value_heads * 4
        )
    )


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
    staged = MAX_SPEC_QUERY_LEN * (2 * block_k + block_v + 2)
    return matrices + vectors + record + output + staged


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
def _load_verify_token(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    bos,
    query_len,
    total_tokens,
    pid_h,
    pid_hv,
    offs_k,
    offs_v,
    mask_k,
    mask_v,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    K: tl.constexpr,
    V: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    T: tl.constexpr,
):
    tok = bos + T
    valid = (T < SPEC_QUERY_LEN) & (T < query_len) & (tok >= 0) & (tok < total_tokens)
    q = tl.load(
        q_ptr + tok * stride_q_token + pid_h * K + offs_k,
        mask=valid & mask_k,
        other=0.0,
    )
    k = tl.load(
        k_ptr + tok * stride_k_token + pid_h * K + offs_k,
        mask=valid & mask_k,
        other=0.0,
    )
    v = tl.load(
        v_ptr + tok * stride_v_token + pid_hv * V + offs_v,
        mask=valid & mask_v,
        other=0.0,
    )
    g = tl.load(g_ptr + tok * stride_g_token + pid_hv, mask=valid, other=0.0).to(
        tl.float32
    )
    beta = tl.load(
        beta_ptr + tok * stride_beta_token + pid_hv, mask=valid, other=0.0
    ).to(tl.float32)
    return q, k, v, tl.exp(g), beta


@triton.jit(do_not_specialize=["total_tokens", "num_rows", "num_work"])
def _gdn_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    replay_ptr,
    key_record_ptr,
    decay_record_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    total_tokens,
    scale,
    num_rows,
    num_work,
    num_record_rows,
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
    stride_key_row,
    stride_key_head,
    stride_key_pos,
    stride_key_dim,
    stride_decay_row,
    stride_decay_head,
    stride_decay_pos,
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
    RECORD_SOA: tl.constexpr,
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
                    mask=(t < query_len)
                    & (tok_null >= 0)
                    & (tok_null < total_tokens)
                    & mask_v,
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
            # Records are indexed by REQUEST ROW, not by cache page: they are
            # per (request, speculative step) temporaries (RFC #19987 s3), so a
            # page shared through prefix caching can never carry another
            # request's records.
            #
            # ``num_record_rows`` is the scratch capacity, which is sized by
            # max_num_seqs rather than by the (padded) graph batch: graph padding
            # rows have a null state index and never reach this branch, so the
            # extra rows would be pure waste. The guard keeps an unexpected
            # non-null padding row from writing past the buffer.
            record_row_ok = pid_b < num_record_rows
            rec_base = (
                replay_ptr + pid_b * stride_replay_block + pid_hv * stride_replay_head
            )
            for t in tl.static_range(SPEC_QUERY_LEN):
                tok = bos + t
                # The window may be shorter than the token tensor, never longer:
                # ``spec_query_start_loc`` can describe graph-padded rows.
                valid = (t < query_len) & (tok >= 0) & (tok < total_tokens)
                # Loaded per token inside the unrolled loop: Triton-Ascend 3.2.2
                # rejects a staged window held in a tuple (nested tuples are not
                # convertible, and a tuple of tensors cannot be indexed by a
                # constexpr -- "unsupported tensor index: constexpr[0]").
                q, raw_k, v, decay, beta = _load_verify_token(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    g_ptr,
                    beta_ptr,
                    bos,
                    query_len,
                    total_tokens,
                    pid_h,
                    pid_hv,
                    offs_k,
                    offs_v,
                    mask_k,
                    mask_v,
                    stride_q_token,
                    stride_k_token,
                    stride_v_token,
                    stride_g_token,
                    stride_beta_token,
                    K,
                    V,
                    SPEC_QUERY_LEN,
                    t,
                )
                q = q.to(tl.float32)
                k = raw_k.to(tl.float32)
                v = v.to(tl.float32)
                if USE_QK_L2NORM:
                    # Same epsilon as FLA l2norm_fwd so both paths agree.
                    q = q * tl.rsqrt(tl.sum(q * q) + 1e-6)
                    k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
                q = q * scale
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
                tl.store(
                    rec + offs_v * stride_replay_dim,
                    corr,
                    mask=valid & mask_v & record_row_ok,
                )
                if RECORD_SOA:
                    if pid_v == 0:
                        tl.store(
                            decay_record_ptr
                            + pid_b * stride_decay_row
                            + pid_hv * stride_decay_head
                            + t * stride_decay_pos,
                            decay,
                            mask=valid & record_row_ok,
                        )
                        if pid_hv % (HV // H) == 0:
                            key_value = k if USE_QK_L2NORM else raw_k
                            tl.store(
                                key_record_ptr
                                + pid_b * stride_key_row
                                + pid_h * stride_key_head
                                + t * stride_key_pos
                                + offs_k * stride_key_dim,
                                key_value,
                                mask=valid & mask_k & record_row_ok,
                            )
                elif pid_v == 0:
                    tl.store(
                        rec + (V + offs_k) * stride_replay_dim,
                        k,
                        mask=valid & mask_k & record_row_ok,
                    )
                    tl.store(
                        rec + (V + K) * stride_replay_dim,
                        decay,
                        mask=valid & record_row_ok,
                    )


def gdn_recoverssm_verify(
    g: torch.Tensor,
    beta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    replay_cache: torch.Tensor | ReplayRecordBuffers,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    scale: float | None = None,
    qk_already_normalized: bool = True,
    out: torch.Tensor | None = None,
    _collect: list | None = None,
) -> torch.Tensor:
    """Verify a GDN speculative window against its checkpoint without modifying the
    checkpoint.

    Ascend signatures: ``g`` (log-space decay) and ``beta`` (already sigmoid) come
    from the layer's ``fused_gdn_gating`` in shape [1, T, HV] or [T, HV]; q, k:
    [1, T, H, K]; v: [1, T, HV, V]; checkpoint_state: [blocks, HV, V, K];
    replay_cache: [rows, HV, spec_query_len, V + K + 1] fp32 (row-indexed scratch). ``qk_already_normalized``
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
    # Records are row-indexed scratch, not cache pages: shape is
    # [rows, HV, spec_query_len, V + K + 1] and only the row count has to cover
    # the batch (the checkpoint keeps using ``nb`` pages).
    soa = isinstance(replay_cache, ReplayRecordBuffers)
    if soa:
        corr, key_record, decay_record = (
            replay_cache.correction,
            replay_cache.key,
            replay_cache.decay,
        )
        _require(
            corr.ndim == 4
            and tuple(corr.shape[1:]) == (HV, spec_query_len, V)
            and tuple(key_record.shape) == (corr.shape[0], H, spec_query_len, K)
            and tuple(decay_record.shape) == (corr.shape[0], HV, spec_query_len),
            "typed record geometry is incompatible",
        )
        _require(
            corr.dtype == decay_record.dtype == torch.float32,
            "correction and decay records must remain FP32",
        )
        _require(
            key_record.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and (
                key_record.dtype == torch.float32
                or (qk_already_normalized and key_record.dtype == k.dtype)
            ),
            "low-precision key records require already-normalized input dtype",
        )
        _require(
            all(t.stride(-1) == 1 for t in (corr, key_record, decay_record)),
            "typed record innermost dimensions must be contiguous",
        )
        key_strides = key_record.stride()
        decay_strides = decay_record.stride()
    else:
        corr = key_record = decay_record = replay_cache
        _rec_tail = (HV, spec_query_len, V + K + 1)
        _require(
            corr.ndim == 4 and tuple(corr.shape[1:]) == _rec_tail,
            "packed replay record shape is incompatible",
        )
        _require(corr.dtype == torch.float32, "packed records must remain FP32")
        key_strides = (0, 0, 0, 0)
        decay_strides = (0, 0, 0)
    _require(corr.shape[0] >= 1, "replay buffer has no rows")
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
                corr,
                key_record,
                decay_record,
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
    spec = _LaunchSpec(
        kernel=_gdn_recoverssm_verify_kernel,
        args=(
            q,
            k,
            v,
            g2,
            beta2,
            checkpoint_state,
            corr,
            key_record,
            decay_record,
            out,
            query_start_loc,
            state_indices,
            total,
            scale,
            batch,
            num_work,
            corr.shape[0],
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
            corr.stride(0),
            corr.stride(1),
            corr.stride(2),
            corr.stride(3),
            *key_strides,
            *decay_strides,
            out.stride(1),
            query_start_loc.stride(0),
            state_indices.stride(0),
        ),
        kwargs={
            "H": H,
            "HV": HV,
            "K": K,
            "V": V,
            "BK": BK,
            "BV": BV,
            "SPEC_QUERY_LEN": spec_query_len,
            "USE_QK_L2NORM": not qk_already_normalized,
            "NUM_BLOCKS": nb,
            "NUM_VALUE_TILES": num_value_tiles,
            "RECORD_SOA": soa,
            "num_warps": 4,
            "num_stages": 2,
        },
        grid=grid,
    )
    if _collect is None:
        spec.launch()
    else:
        _collect.append(spec)
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
    align_valid_ptr,
    align_col_ptr,
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
    NUM_BLOCKS: tl.constexpr,
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
    final_state_col = 0
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
        boundary_col = next_boundary // mamba_block_size - 1
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + boundary_col * stride_block_table_col,
            mask=crosses_boundary
            & (boundary_col >= 0)
            & (boundary_col < block_table_width),
            other=null_block_id,
        ).to(tl.int64)
    # Unified validity for this plan row. Everything downstream -- conv
    # compaction, SSM fold and the bookkeeping -- consumes exactly this, instead
    # of each kernel re-deciding on its own. Padding and zero-length rows are a
    # defined no-op; a row that claims work but has an impossible geometry is
    # dropped rather than partially executed.
    geometry_ok = (bos >= 0) & (bos <= eos) & (query_len <= SPEC_QUERY_LEN)
    source_ok = (source_state_idx > null_block_id) & (source_state_idx < NUM_BLOCKS)
    final_ok = (final_state_idx > null_block_id) & (final_state_idx < NUM_BLOCKS)
    valid = (
        source_ok
        & final_ok
        & geometry_ok
        & (commit_len > 0)
        & (commit_len <= query_len)
    )
    if ALIGN_MODE:
        valid = valid & (num_computed >= 0)
        boundary_ok = (boundary_state_idx > null_block_id) & (
            boundary_state_idx < NUM_BLOCKS
        )
        boundary_recovery_len = tl.where(boundary_ok, boundary_recovery_len, 0)
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

    tl.store(align_valid_ptr + spec_idx, valid.to(tl.int32))
    tl.store(align_col_ptr + spec_idx, final_state_col)


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
        source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
            tl.int64
        )
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
            mask = (offs_d[:, None] < conv_dim) & (offs_h[None, :] < conv_history_len)
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
    key_ref_ptr,
    key_base_addrs_ptr,
    key_block_strides_ptr,
    decay_ref_ptr,
    decay_base_addrs_ptr,
    decay_block_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    num_rows,
    num_work,
    num_record_rows,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_replay_head,
    stride_replay_pos,
    stride_replay_dim,
    stride_key_head,
    stride_key_pos,
    stride_key_dim,
    stride_decay_head,
    stride_decay_pos,
    stride_si,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_VALUE_TILES: tl.constexpr,
    NUM_KEY_HEADS: tl.constexpr,
    RECORD_SOA: tl.constexpr,
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
            & (final_idx < NUM_BLOCKS)
            # Records live in a scratch sized by max_num_seqs; padded graph rows
            # never reach here, but a non-null padding row must not read past it.
            & (pid_b < num_record_rows)
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
                (b_idx > null_block_id)
                & (b_idx < NUM_BLOCKS)
                & (b_idx != final_idx)
                & (b_len > 0)
            )
            state_ptr = tl.load(state_base_addrs_ptr + pid_l).to(
                tl.pointer_type(state_ref_ptr.dtype.element_ty)
            )
            sbs = tl.load(state_block_strides_ptr + pid_l)
            replay_ptr = tl.load(replay_base_addrs_ptr + pid_l).to(
                tl.pointer_type(replay_ref_ptr.dtype.element_ty)
            )
            rbs = tl.load(replay_block_strides_ptr + pid_l)
            # Row-indexed records (see the verify kernel): the checkpoint is read
            # from the request's page (``src``), the records from its row.
            rec_base = replay_ptr + pid_b * rbs + pid_h * stride_replay_head
            if RECORD_SOA:
                key_ptr = tl.load(key_base_addrs_ptr + pid_l).to(
                    tl.pointer_type(key_ref_ptr.dtype.element_ty)
                )
                key_bs = tl.load(key_block_strides_ptr + pid_l)
                key_head = pid_h // (NUM_HEADS // NUM_KEY_HEADS)
                key_base = key_ptr + pid_b * key_bs + key_head * stride_key_head
                decay_ptr = tl.load(decay_base_addrs_ptr + pid_l).to(
                    tl.pointer_type(decay_ref_ptr.dtype.element_ty)
                )
                decay_bs = tl.load(decay_block_strides_ptr + pid_l)
                decay_base = decay_ptr + pid_b * decay_bs + pid_h * stride_decay_head
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
                if RECORD_SOA:
                    kk = tl.load(
                        key_base + t * stride_key_pos + offs_k * stride_key_dim,
                        mask=mask_k,
                        other=0.0,
                    ).to(tl.float32)
                    decay = tl.load(decay_base + t * stride_decay_pos)
                else:
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
class _LaunchSpec:
    """One kernel launch, shared by the serving path and startup preparation.

    Both paths must use the identical kernel object, argument tuple, constexpr
    inputs and options: the startup preparation exists to compile exactly what a
    real step will launch, so any divergence would prepare a different variant.
    """

    kernel: Any
    args: tuple
    kwargs: dict
    grid: tuple

    def launch(self) -> None:
        self.kernel[self.grid](*self.args, **self.kwargs)

    def compile_only(self) -> Any:
        """Compile + load this variant without launching it.

        ``warmup=True`` builds and caches the CompiledKernel but skips the device
        launch; ``_init_handles`` then performs the per-process binary/function
        registration. A disk cache hit does not mean this worker has registered
        the binary, and a failed registration can surface as zero handles rather
        than an exception, so the handles are checked explicitly.
        """
        compiled = self.kernel.run(
            *self.args, grid=self.grid, warmup=True, **self.kwargs
        )
        resolve = getattr(compiled, "result", None)
        if callable(resolve):  # Future-like
            compiled = resolve()
        _init_handles(compiled)
        _require_nonzero_handles(compiled, self.kernel)
        return compiled


def _init_handles(compiled: Any) -> None:
    init = getattr(compiled, "_init_handles", None)
    if init is None:
        raise RuntimeError(
            "This Triton build does not expose CompiledKernel._init_handles; "
            "cannot verify that the commit kernel was registered on the device. "
            "Refusing to report the RecoverSSM runtime as ready."
        )
    init()


def _require_nonzero_handles(compiled: Any, kernel: Any) -> None:
    # rtDevBinaryRegister / rtFunctionRegister can fail and hand back a null
    # handle instead of raising, so "no exception" is not proof of success.
    for name in ("module", "function"):
        value = getattr(compiled, name, None)
        if value is None or value == 0:
            raise RuntimeError(
                f"Triton kernel {getattr(kernel, '__name__', kernel)} did not "
                f"register on the device ({name}={value!r}); refusing to report "
                "the RecoverSSM runtime as ready"
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
    replays: tuple[Any, ...]
    replay_base_addrs: torch.Tensor
    replay_block_strides: torch.Tensor
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    spec_query_len: int
    """Speculative window length (constexpr in every kernel)."""
    num_blocks: int

    @classmethod
    def from_tensors(
        cls,
        conv_states: Sequence[torch.Tensor],
        checkpoints: Sequence[torch.Tensor],
        replays: Sequence[torch.Tensor | ReplayRecordBuffers],
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
        soa = isinstance(replays[0], ReplayRecordBuffers)
        _require(
            all(isinstance(r, ReplayRecordBuffers) == soa for r in replays),
            "packed and typed records cannot share a commit context",
        )
        if soa:
            corrs = [r.correction for r in replays]
            keys = [r.key for r in replays]
            decays = [r.decay for r in replays]
            H = keys[0].shape[1]
            _require(H > 0 and HV % H == 0, "record key heads are incompatible")
            for c, k, d in zip(corrs, keys, decays):
                _require(
                    c.shape == corrs[0].shape
                    and tuple(c.shape[1:]) == (HV, spec_query_len, V)
                    and tuple(k.shape) == (c.shape[0], H, spec_query_len, K)
                    and tuple(d.shape) == (c.shape[0], HV, spec_query_len)
                    and c.dtype == d.dtype == torch.float32
                    and k.dtype in (torch.float16, torch.bfloat16, torch.float32)
                    and k.dtype == keys[0].dtype
                    and all(t.device == _dev for t in (c, k, d))
                    and c.stride()[1:] == corrs[0].stride()[1:]
                    and k.stride()[1:] == keys[0].stride()[1:]
                    and d.stride()[1:] == decays[0].stride()[1:],
                    "layers need matching typed replay geometry/dtypes/strides",
                )
        else:
            corrs = keys = decays = list(replays)
            H = HV
            for r in replays:
                _require(
                    r.ndim == 4
                    and tuple(r.shape[1:]) == (HV, spec_query_len, V + K + 1)
                    and r.shape == replays[0].shape
                    and r.dtype == torch.float32
                    and r.stride()[1:] == replays[0].stride()[1:]
                    and r.device == _dev,
                    "layers need matching packed FP32 replay buffers",
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
        context = cls(
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
            addr(corrs),
            bstr(corrs),
            z(),
            z(),
            z(),
            z(),
            spec_query_len,
            nb,
        )
        # Expose the records on every construction path, not only in ``create``:
        # the start-up self-check builds a context straight from ``from_tensors``
        # and must use the same attribute the serving path reads.
        context.records = list(replays)
        context.record_rows = int(corrs[0].shape[0])
        context.record_layout = "soa" if soa else "packed"
        context.num_key_heads = int(H)
        context.key_dtype = keys[0].dtype
        context.correction_base_addrs = context.replay_base_addrs
        context.correction_block_strides = context.replay_block_strides
        context.key_base_addrs = addr(keys)
        context.key_block_strides = bstr(keys)
        context.decay_base_addrs = addr(decays)
        context.decay_block_strides = bstr(decays)
        context.record_bytes = sum(record_nbytes(r) for r in replays)
        context.align_valid = z()
        context.align_col = z()
        return context

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
        record_rows: int | None = None,
        num_key_heads: int | None = None,
        key_dtype: torch.dtype | None = None,
    ) -> "GDNRecoverSSMCommitContext":
        """Build the context from live layers' ``kv_cache`` pages.

        Each layer must expose ``kv_cache`` with exactly two entries: conv state
        (``[blocks, conv_dim, window]`` in the layout selected by
        ``is_conv_state_dim_first()``) and the recurrent checkpoint
        (``[blocks, HV, V, K]``). The fp32 records are allocated here as scratch
        of shape ``[rows, HV, Q, V + K + 1]`` -- they are per (request, step)
        temporaries, so they must not live in the cache pages.


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

        if any(len(layer.kv_cache) != 2 for layer in layers):
            raise ValueError(
                "GDN RecoverSSM expects conv and checkpoint pages; the replay "
                "records live outside the cache"
            )
        state0 = layers[0].kv_cache[1]
        num_heads = state0.shape[1]
        head_v = state0.shape[2]
        head_k = state0.shape[3]
        device = state0.device
        # Only real requests write records: padded graph rows carry a null state
        # index and the kernels skip them, so sizing by max_num_seqs (instead of
        # the padded graph batch) removes ~3/4 of this buffer.
        rows = int(record_rows or max_num_reqs)
        key_heads = num_heads if num_key_heads is None else int(num_key_heads)
        record_key_dtype = torch.float32 if key_dtype is None else key_dtype
        records = [
            ReplayRecordBuffers.allocate(
                rows,
                num_heads,
                key_heads,
                spec_query_len,
                head_v,
                head_k,
                device=device,
                key_dtype=record_key_dtype,
            )
            for _ in layers
        ]
        context = cls.from_tensors(
            [layer.kv_cache[0] for layer in layers],
            [layer.kv_cache[1] for layer in layers],
            records,
            spec_query_len=spec_query_len,
            max_num_reqs=max_num_reqs,
            conv_dim_first=is_conv_state_dim_first(),
        )
        # Keep strong references: the address tables below only store raw
        # pointers, and a collected scratch would leave the kernels writing into
        # freed device memory.
        context.records = records
        # Hand each layer its own record buffer. The layer's forward runs the
        # verify kernel and must write to the same scratch the commit context
        # reads; binding it here means a context rebuild (KV re-bind, re-profile)
        # refreshes the layer automatically instead of leaving a stale tensor.
        for layer, record in zip(layers, records):
            layer._gdn_recoverssm_records = record
        return context

    def make_commit_specs(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
        commit_conv: bool = True,
    ) -> tuple:
        """Build the launch specs for one step, without launching them.

        The serving path and the startup preparation both call this, so the
        prepared variants are exactly the ones a real step will use.
        """
        specs: list[_LaunchSpec] = []
        self.commit(
            num_accepted_tokens,
            state_indices,
            query_start_loc,
            request_indices=request_indices,
            block_table=block_table,
            num_computed_tokens=num_computed_tokens,
            mamba_block_size=mamba_block_size,
            commit_conv=commit_conv,
            _collect=specs,
        )
        return tuple(specs)

    def prepare_verify(
        self,
        *,
        activation_dtype: torch.dtype,
        state_table_width: int | None = None,
        device: torch.device | None = None,
    ) -> list[dict[str, Any]]:
        """Compile serving-pool verify signatures without reading/writing state.

        Native gating implementations return either activation-typed or FP32
        beta. The value view may be contiguous or retain the combined-QKV token
        stride. Both legal signatures are registered before readiness. Runtime
        row/token/work counts are deliberately not specialized, so this single
        logical row covers every serving batch size of the same layer geometry.
        """
        device = self.checkpoints[0].device if device is None else device
        record = self.records[0]
        _, hv, value_dim, key_dim = self.checkpoints[0].shape
        key_heads = self.num_key_heads
        window = self.spec_query_len
        width = int(state_table_width or window)
        _require(width > 0, "verify preparation requires a positive table stride")
        q = torch.empty(
            (1, window, key_heads, key_dim), dtype=activation_dtype, device=device
        )
        k = torch.empty_like(q)
        g = torch.empty((1, window, hv), dtype=torch.float32, device=device)
        output = torch.empty(
            (1, window, hv, value_dim), dtype=activation_dtype, device=device
        )
        state_table = torch.empty((1, width), dtype=torch.int32, device=device)
        offsets = torch.empty(2, dtype=torch.int32, device=device)
        beta_dtypes = tuple(dict.fromkeys((torch.float32, activation_dtype)))
        report = []
        for packed_value in (False, True):
            if packed_value:
                features = 2 * key_heads * key_dim + hv * value_dim
                combined = torch.empty(
                    (1, window, features), dtype=activation_dtype, device=device
                )
                value = combined[:, :, 2 * key_heads * key_dim :].view(
                    1, window, hv, value_dim
                )
            else:
                value = torch.empty_like(output)
            for beta_dtype in beta_dtypes:
                beta = torch.empty((1, window, hv), dtype=beta_dtype, device=device)
                specs = []
                gdn_recoverssm_verify(
                    g,
                    beta,
                    q,
                    k,
                    value,
                    checkpoint_state=self.checkpoints[0],
                    replay_cache=record,
                    query_start_loc=offsets,
                    state_indices=state_table[:, 0],
                    spec_query_len=window,
                    qk_already_normalized=True,
                    out=output,
                    _collect=specs,
                )
                for spec in specs:
                    spec.compile_only()
                block_k = triton.next_power_of_2(key_dim)
                block_v = _verify_value_tile(block_k, value_dim)
                description = dict(
                    activation_dtype=str(activation_dtype),
                    beta_dtype=str(beta_dtype),
                    value_token_stride=value.stride(1),
                    state_index_stride=width,
                    state_blocks=self.num_blocks,
                    record_rows=self.record_rows,
                    num_value_heads=hv,
                    num_key_heads=key_heads,
                    window=window,
                    value_tile=block_v,
                    key_tile=block_k,
                    ub_estimate_bytes=_verify_ub_floats(block_v, block_k) * 4,
                )
                report.append(description)
                logger.info("GDN_VERIFY_PREPARED %s", description)
        self.verify_prepare_report = tuple(report)
        return report

    def prepare(
        self,
        *,
        capacity: int,
        state_table_width: int,
        request_mapping: bool,
        device: torch.device,
        align_mode: bool = False,
        mamba_block_size: int | None = None,
        block_table_width: int | None = None,
        request_index_dtype=torch.int64,
        accepted_dtype=torch.int32,
        state_index_dtype=torch.int32,
        query_index_dtype=torch.int32,
        computed_dtype=torch.int32,
    ) -> list[int]:
        """Compile exact none/align input signatures before requests arrive.

        Plans may have graph-padded capacity, but only real record rows commit.
        Dummies preserve runtime dtype, table-column stride and optional inputs.
        """
        _require(
            not align_mode
            or (
                mamba_block_size is not None and mamba_block_size >= self.spec_query_len
            ),
            "align preparation requires the runtime block size",
        )
        compiled_rows = []
        for rows in range(1, min(capacity, self.record_rows) + 1):
            table = torch.empty(
                (rows, state_table_width), dtype=state_index_dtype, device=device
            )
            state_indices = table[:, 0]
            qsl = torch.empty(rows + 1, dtype=query_index_dtype, device=device)
            accepted = torch.empty(rows, dtype=accepted_dtype, device=device)
            mapping = (
                torch.empty(rows, dtype=request_index_dtype, device=device)
                if request_mapping
                else None
            )
            bt = computed = None
            if align_mode:
                width = int(block_table_width or state_table_width)
                bt = torch.empty((rows, width), dtype=state_index_dtype, device=device)
                computed = torch.empty(rows, dtype=computed_dtype, device=device)
            for spec in self.make_commit_specs(
                accepted,
                state_indices,
                qsl,
                request_indices=mapping,
                block_table=bt,
                num_computed_tokens=computed,
                mamba_block_size=mamba_block_size if align_mode else None,
            ):
                spec.compile_only()
            compiled_rows.append(rows)
        return compiled_rows

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
        _collect: list | None = None,
    ) -> None:
        """Fold accepted GDN and convolution inputs into every layer's checkpoint.

        ``_collect`` is startup-preparation only: when a list is passed, the exact
        same launch specs are appended to it instead of being launched. That keeps
        one construction path for serving and for compilation, which is the whole
        point -- a separately written "prepare" path would compile a different
        variant as soon as the two drift apart.
        """
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
            block_table is None or (block_table.ndim == 2 and block_table.shape[1] > 0),
            "block table must be two-dimensional with a positive width",
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
        plan_spec = _LaunchSpec(
            kernel=_prepare_commit_plan_kernel,
            args=(
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
                self.align_valid,
                self.align_col,
                NULL_BLOCK_ID,
                mamba_block_size or 1,
                block_table.shape[1] if block_table is not None else 1,
                num_accepted_tokens.stride(0),
                request_indices.stride(0) if request_indices is not None else 0,
                state_indices.stride(0),
                query_start_loc.stride(0),
                bt_stride[0],
                bt_stride[1],
                (0 if num_computed_tokens is None else num_computed_tokens.stride(0)),
            ),
            kwargs={
                "SPEC_QUERY_LEN": self.spec_query_len,
                "NUM_BLOCKS": self.num_blocks,
                "num_warps": 1,
            },
            grid=(batch,),
        )
        if _collect is not None:
            _collect.append(plan_spec)
        else:
            plan_spec.launch()
        num_layers = len(self.checkpoints)
        conv_spec = None
        if commit_conv:
            conv_ref = self.conv_states[0]
            conv_dim = conv_ref.shape[1]
            # BLOCK_HISTORY must cover the whole window in one tile (the kernel's
            # mask is written against it), so only BLOCK_D is UB-sized.
            block_history = triton.next_power_of_2(self.conv_history_len)
            conv_block_d = _conv_dim_tile(block_history)
            num_d_tiles = triton.cdiv(conv_dim, conv_block_d)
            conv_work = num_d_tiles * batch * num_layers
            conv_spec = _LaunchSpec(
                kernel=_compact_conv_state_kernel,
                grid=(min(conv_work, _launch_cap()),),
                args=(
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
                ),
                kwargs={
                    "BLOCK_D": conv_block_d,
                    "BLOCK_HISTORY": block_history,
                    "ALIGN_MODE": block_table is not None,
                    "NUM_BLOCKS": self.num_blocks,
                    "NUM_D_TILES": num_d_tiles,
                    "num_warps": 4,
                },
            )
            if _collect is not None:
                _collect.append(conv_spec)
            else:
                conv_spec.launch()
        ref = self.checkpoints[0]
        _, HV, V, K = ref.shape
        BK = triton.next_power_of_2(K)
        # Upstream caps the commit tile at 16 rows (each state is streamed once
        # and smaller tiles keep more of them in flight); the cap is kept and the
        # tile is additionally clamped to the UB budget of this device.
        BV = _commit_value_tile(BK, V)
        record = self.replays[0]
        soa = self.record_layout == "soa"
        rr = record.correction if soa else record
        kr = record.key if soa else record
        dr = record.decay if soa else record
        key_strides = kr.stride()[1:] if soa else (0, 0, 0)
        decay_strides = dr.stride()[1:] if soa else (0, 0)
        # Must be computed before the argument tuple that consumes them.
        num_value_tiles = triton.cdiv(V, BV)
        num_work = num_value_tiles * batch * (num_layers * HV)
        args = (
            ref,
            self.state_base_addrs,
            self.state_block_strides,
            rr,
            self.replay_base_addrs,
            self.replay_block_strides,
            kr,
            self.key_base_addrs,
            self.key_block_strides,
            dr,
            self.decay_base_addrs,
            self.decay_block_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            batch,
            num_work,
            rr.shape[0],
            ref.stride(1),
            ref.stride(2),
            ref.stride(3),
            rr.stride(1),
            rr.stride(2),
            rr.stride(3),
            *key_strides,
            *decay_strides,
            state_indices.stride(0),
        )
        # One launch per layer set: the final and the boundary state come out of a
        # single fold over one checkpoint load. The physical grid stays bounded;
        # each program walks several logical tiles.
        commit_spec = _LaunchSpec(
            kernel=_commit_gdn_state_kernel,
            grid=(min(num_work, _launch_cap()),),
            args=args,
            kwargs={
                "K": K,
                "V": V,
                "BK": BK,
                "BV": BV,
                "NUM_HEADS": HV,
                "NUM_BLOCKS": self.num_blocks,
                "NUM_VALUE_TILES": num_value_tiles,
                "NUM_KEY_HEADS": self.num_key_heads,
                "RECORD_SOA": soa,
                "num_warps": 4,
                "num_stages": 2,
            },
        )
        if _collect is not None:
            _collect.append(commit_spec)
        else:
            commit_spec.launch()
        self._last_specs = (
            (plan_spec,)
            + ((conv_spec,) if conv_spec is not None else ())
            + (commit_spec,)
        )


__all__ = [
    "GDNRecoverSSMCommitContext",
    "ReplayRecordBuffers",
    "gdn_recoverssm_verify",
    "record_geometry_bytes",
    "record_nbytes",
]
