# SPDX-License-Identifier: Apache-2.0
"""Step-owned recovery: one plan, one convolution pass, one SSM pass.

Layer storage and physical page tables remain group-specific. The flat three-slot
kernel ABI supports Triton-Ascend 3.2.2; larger group sets use multiple buckets.
Borrowing native metadata adds no per-step Host pointer-table upload or device
pack. Rare multi-bucket publication uses one persistent GPU descriptor table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from .kernels import (
    _commit_value_tile,
    _conv_dim_tile,
    _launch_cap,
    _LaunchSpec,
    _ptr_table,
)

logger = init_logger("vllm.gdn_recoverssm_ascend.grouped_commit")


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "plan_capacity",
        "table_width0",
        "table_width1",
        "table_width2",
        "table_row_stride0",
        "table_row_stride1",
        "table_row_stride2",
    ]
)
def _grouped_plan_kernel(
    sampled_ptr,
    state0,
    state1,
    state2,
    qsl0,
    qsl1,
    qsl2,
    mapping0,
    mapping1,
    mapping2,
    table0,
    table1,
    table2,
    computed0,
    computed1,
    computed2,
    table_width0,
    table_width1,
    table_width2,
    table_row_stride0,
    table_row_stride1,
    table_row_stride2,
    source_plan_ptr,
    lens_plan_ptr,
    final_plan_ptr,
    boundary_plan_ptr,
    boundary_lens_ptr,
    valid_plan_ptr,
    col_plan_ptr,
    idx_mapping_ptr,
    persistent_cols_ptr,
    persistent_accepted_ptr,
    num_rows,
    plan_capacity,
    null_block_id,
    SI_STRIDES0: tl.constexpr,
    SI_STRIDES1: tl.constexpr,
    SI_STRIDES2: tl.constexpr,
    QSL_STRIDES0: tl.constexpr,
    QSL_STRIDES1: tl.constexpr,
    QSL_STRIDES2: tl.constexpr,
    MAP_STRIDES0: tl.constexpr,
    MAP_STRIDES1: tl.constexpr,
    MAP_STRIDES2: tl.constexpr,
    TABLE_COL_STRIDES0: tl.constexpr,
    TABLE_COL_STRIDES1: tl.constexpr,
    TABLE_COL_STRIDES2: tl.constexpr,
    COMPUTED_STRIDES0: tl.constexpr,
    COMPUTED_STRIDES1: tl.constexpr,
    COMPUTED_STRIDES2: tl.constexpr,
    NUM_BLOCKS0: tl.constexpr,
    NUM_BLOCKS1: tl.constexpr,
    NUM_BLOCKS2: tl.constexpr,
    BLOCK_SIZES0: tl.constexpr,
    BLOCK_SIZES1: tl.constexpr,
    BLOCK_SIZES2: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    PUBLISH_ALIGN: tl.constexpr,
):
    # Triton-Ascend 3.2.2 supports neither nested tuples nor indexing a tuple of
    # runtime values by a constexpr ("unsupported tensor index: constexpr[0]"), so
    # the per-group values are selected with a compile-time branch instead of a
    # tuple lookup.
    row = tl.program_id(0)
    all_valid = tl.full((), True, tl.int1)
    shared_col = tl.full((), -1, tl.int32)
    shared_request = tl.full((), -1, tl.int64)
    for group in tl.static_range(NUM_GROUPS):
        if group == 0:
            state_p, qsl_p, mapping_p, table_p = state0, qsl0, mapping0, table0
            computed_p, table_width, table_row_stride = (
                computed0,
                table_width0,
                table_row_stride0,
            )
            num_blocks, block_size = NUM_BLOCKS0, BLOCK_SIZES0
            si_stride, qsl_stride = SI_STRIDES0, QSL_STRIDES0
            map_stride, col_stride = MAP_STRIDES0, TABLE_COL_STRIDES0
            computed_stride = COMPUTED_STRIDES0
        elif group == 1:
            state_p, qsl_p, mapping_p, table_p = state1, qsl1, mapping1, table1
            computed_p, table_width, table_row_stride = (
                computed1,
                table_width1,
                table_row_stride1,
            )
            num_blocks, block_size = NUM_BLOCKS1, BLOCK_SIZES1
            si_stride, qsl_stride = SI_STRIDES1, QSL_STRIDES1
            map_stride, col_stride = MAP_STRIDES1, TABLE_COL_STRIDES1
            computed_stride = COMPUTED_STRIDES1
        else:
            state_p, qsl_p, mapping_p, table_p = state2, qsl2, mapping2, table2
            computed_p, table_width, table_row_stride = (
                computed2,
                table_width2,
                table_row_stride2,
            )
            num_blocks, block_size = NUM_BLOCKS2, BLOCK_SIZES2
            si_stride, qsl_stride = SI_STRIDES2, QSL_STRIDES2
            map_stride, col_stride = MAP_STRIDES2, TABLE_COL_STRIDES2
            computed_stride = COMPUTED_STRIDES2
        request = tl.load(mapping_p + row * map_stride).to(tl.int64)
        source = tl.load(state_p + row * si_stride).to(tl.int64)
        bos = tl.load(qsl_p + row * qsl_stride).to(tl.int64)
        eos = tl.load(qsl_p + (row + 1) * qsl_stride).to(tl.int64)
        qlen = eos - bos
        accepted = tl.load(sampled_ptr + request).to(tl.int32)
        count = tl.minimum(tl.maximum(accepted, 0), qlen)
        count = tl.minimum(count, SPEC_QUERY_LEN)
        final = source
        boundary = tl.full((), null_block_id, tl.int64)
        boundary_len = tl.full((), 0, tl.int32)
        col = tl.full((), 0, tl.int32)
        computed_ok = tl.full((), True, tl.int1)
        if ALIGN_MODE:
            computed = tl.load(
                computed_p + request * computed_stride
            ).to(tl.int64)
            computed_ok = computed >= 0
            position = computed + count
            col = tl.minimum(
                tl.maximum(position - 1, 0) // block_size,
                table_width - 1,
            ).to(tl.int32)
            final = tl.load(
                table_p
                + request * table_row_stride
                + col * col_stride
            ).to(tl.int64)
            next_boundary = (computed // block_size + 1) * block_size
            crossed = position >= next_boundary
            boundary_len = (next_boundary - computed).to(tl.int32)
            boundary_col = next_boundary // block_size - 1
            boundary = tl.load(
                table_p
                + request * table_row_stride
                + boundary_col * col_stride,
                mask=crossed
                & computed_ok
                & (boundary_col >= 0)
                & (boundary_col < table_width),
                other=null_block_id,
            ).to(tl.int64)
        geometry = (bos >= 0) & (bos <= eos) & (qlen <= SPEC_QUERY_LEN)
        valid = geometry & (source > null_block_id) & (source < num_blocks)
        valid = valid & (final > null_block_id) & (final < num_blocks)
        valid = valid & (count > 0) & (count <= qlen)
        valid = valid & computed_ok
        boundary_ok = (boundary > null_block_id) & (boundary < num_blocks)
        boundary_len = tl.where(boundary_ok, boundary_len, 0)
        # Later kernels use the maximum pool bound across this homogeneous
        # bucket. Canonicalize a group-local invalid boundary now so a larger
        # neighbouring pool can never make that address appear valid.
        boundary = tl.where(boundary_ok, boundary, null_block_id)
        offset = group * plan_capacity + row
        tl.store(source_plan_ptr + offset, tl.where(valid, source, null_block_id))
        tl.store(lens_plan_ptr + offset, tl.where(valid, count, 0))
        tl.store(final_plan_ptr + offset, tl.where(valid, final, null_block_id))
        tl.store(boundary_plan_ptr + offset, tl.where(valid, boundary, null_block_id))
        tl.store(boundary_lens_ptr + offset, tl.where(valid, boundary_len, 0))
        tl.store(valid_plan_ptr + offset, valid.to(tl.int32))
        tl.store(col_plan_ptr + offset, col)
        if group == 0:
            shared_col = col
            shared_request = request
        else:
            all_valid = all_valid & (col == shared_col) & (request == shared_request)
        all_valid = all_valid & valid
    if PUBLISH_ALIGN:
        # The batch sampled counts above and the persistent accepted counts below
        # are distinct buffers. All physical consumers use the plan snapshots,
        # not this persistent column. Same-stream ordering ensures conv/fold
        # finish before native postcopy or the next forward can consume it.
        slot = tl.load(idx_mapping_ptr + shared_request).to(tl.int64)
        if all_valid & (slot >= 0):
            tl.store(persistent_cols_ptr + slot, shared_col)
            tl.store(persistent_accepted_ptr + slot, 1)


@triton.jit(do_not_specialize=["num_rows", "num_work", "plan_capacity"])
def _grouped_compact_conv_state_kernel(
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
    layer_groups_ptr,
    plan_capacity,
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
        plan_row = tl.load(layer_groups_ptr + pid_l) * plan_capacity + pid_b
        source_state_idx = tl.load(
            state_indices_ptr + plan_row * stride_state_indices
        ).to(tl.int64)
        commit_len = tl.load(commit_lens_ptr + plan_row)
        final_state_idx = tl.load(final_state_indices_ptr + plan_row).to(tl.int64)
        row_ok = (
            (source_state_idx > null_block_id)
            & (commit_len != 0)
            & (final_state_idx > null_block_id)
            & (final_state_idx < NUM_BLOCKS)
        )
        if row_ok:
            boundary_state_idx = tl.load(boundary_state_indices_ptr + plan_row).to(
                tl.int64
            )
            boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + plan_row)

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


@triton.jit(do_not_specialize=["num_rows", "num_work", "plan_capacity"])
def _grouped_commit_gdn_state_kernel(
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
    layer_groups_ptr,
    plan_capacity,
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
        plan_row = tl.load(layer_groups_ptr + pid_l) * plan_capacity + pid_b
        src = tl.load(state_indices_ptr + plan_row * stride_si).to(tl.int64)
        n = tl.load(commit_lens_ptr + plan_row).to(tl.int32)
        final_idx = tl.load(final_state_indices_ptr + plan_row).to(tl.int64)
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
            b_idx = tl.load(boundary_state_indices_ptr + plan_row).to(tl.int64)
            b_len = tl.load(boundary_recovery_lens_ptr + plan_row).to(tl.int32)
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


@dataclass(frozen=True)
class CommitEntry:
    metadata: Any
    payload: Any
    context: Any


def _layout_key(context):
    state = context.checkpoints[0]
    conv = context.conv_states[0]
    record = context.records[0]
    soa = context.record_layout == "soa"
    rr = record.correction if soa else record
    kr = record.key if soa else record
    dr = record.decay if soa else record
    return (
        state.device,
        state.dtype,
        tuple(state.shape[1:]),
        tuple(state.stride()[1:]),
        conv.dtype,
        tuple(conv.shape[1:]),
        tuple(conv.stride()[1:]),
        context.conv_history_len,
        context.spec_query_len,
        context.record_layout,
        context.num_key_heads,
        rr.dtype,
        tuple(rr.stride()[1:]),
        kr.dtype,
        tuple(kr.stride()[1:]),
        dr.dtype,
        tuple(dr.stride()[1:]),
    )


def _logical_key(entry):
    p = entry.payload
    rows = int(p.state_indices.shape[0])
    host_rows = tuple(getattr(p, "request_rows", ()))
    host_lens = tuple(getattr(p, "query_lens", ()))
    if len(host_rows) != rows or len(host_lens) != rows:
        # A compatible older metadata producer is still supported, but without
        # host row evidence it cannot share a physical launch with another group.
        return (id(entry.metadata),)
    return (host_rows, host_lens, p.block_size, p.block_table is not None)


@triton.jit(do_not_specialize=["num_rows", "num_buckets"])
def _publish_grouped_align_kernel(
    descriptors_ptr,
    request_indices_ptr,
    idx_mapping_ptr,
    persistent_cols_ptr,
    persistent_accepted_ptr,
    num_rows,
    num_buckets,
    MAP_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    valid = tl.full((), True, tl.int1)
    first_cols = tl.load(descriptors_ptr + 1).to(tl.pointer_type(tl.int32))
    shared_col = tl.load(first_cols + row)
    for bucket in range(num_buckets):
        descriptor = descriptors_ptr + bucket * 4
        valid_ptr = tl.load(descriptor).to(tl.pointer_type(tl.int32))
        col_ptr = tl.load(descriptor + 1).to(tl.pointer_type(tl.int32))
        capacity = tl.load(descriptor + 2)
        group_count = tl.load(descriptor + 3).to(tl.int32)
        for group in range(group_count):
            offset = group * capacity + row
            valid = valid & (tl.load(valid_ptr + offset) != 0)
            valid = valid & (tl.load(col_ptr + offset) == shared_col)
    request = tl.load(request_indices_ptr + row * MAP_STRIDE).to(tl.int64)
    slot = tl.load(idx_mapping_ptr + request).to(tl.int64)
    if valid & (slot >= 0):
        tl.store(persistent_cols_ptr + slot, shared_col)
        tl.store(persistent_accepted_ptr + slot, 1)


def _flat_three(values):
    """Flat launch ABI; duplicate an unused slot instead of passing a tuple."""
    values = tuple(values)
    if not 0 < len(values) <= 3:
        raise ValueError("one physical recovery bucket supports one to three groups")
    return values + (values[0],) * (3 - len(values))


def _publisher_descriptors(buckets):
    owner = buckets[0]
    key = tuple((id(b.valid), id(b.col), b.capacity, len(b.contexts)) for b in buckets)
    if getattr(owner, "_publisher_key", None) != key:
        values = []
        for bucket in buckets:
            for value in (
                bucket.valid.data_ptr(),
                bucket.col.data_ptr(),
                bucket.capacity,
                len(bucket.contexts),
            ):
                values.append(value if value < (1 << 63) else value - (1 << 64))
        owner._publisher_descriptors = torch.tensor(
            values, dtype=torch.int64, device=owner.device
        )
        # Keep the descriptor's referenced plan allocations alive, without
        # retaining the other buckets' full model KV tensors in a cache cycle.
        owner._publisher_plan_owners = tuple((b.valid, b.col) for b in buckets)
        owner._publisher_key = key
    return owner._publisher_descriptors


def _publish_spec(
    buckets, request_indices, idx_mapping, persistent_cols, persistent_accepted, rows
):
    return _LaunchSpec(
        kernel=_publish_grouped_align_kernel,
        args=(
            _publisher_descriptors(buckets),
            request_indices,
            idx_mapping,
            persistent_cols,
            persistent_accepted,
            rows,
            len(buckets),
        ),
        kwargs={"MAP_STRIDE": request_indices.stride(0), "num_warps": 1},
        grid=(rows,),
    )


class GroupedCommitBucket:
    """Stable per-layer address tables and per-group plan storage for one epoch."""

    def __init__(self, contexts):
        self.contexts = tuple(contexts)
        if not 0 < len(contexts) <= 3:
            raise ValueError("one recovery bucket requires one to three contexts")
        key = _layout_key(contexts[0])
        if any(_layout_key(c) != key for c in contexts[1:]):
            raise ValueError("grouped recovery requires homogeneous layer layouts")
        self.capacity = min(int(c.record_rows) for c in contexts)
        if self.capacity <= 0:
            raise ValueError("grouped recovery has no record rows")
        self.device = contexts[0].checkpoints[0].device
        self.checkpoints = tuple(t for c in contexts for t in c.checkpoints)
        self.conv_states = tuple(t for c in contexts for t in c.conv_states)
        self.records = tuple(t for c in contexts for t in c.records)
        self.layer_groups = torch.tensor(
            [g for g, c in enumerate(contexts) for _ in c.checkpoints],
            dtype=torch.int32,
            device=self.device,
        )
        self.max_blocks = max(int(c.num_blocks) for c in contexts)
        self.spec_query_len = int(contexts[0].spec_query_len)
        self.record_layout = contexts[0].record_layout
        self.num_key_heads = int(contexts[0].num_key_heads)
        self.conv_history_len = int(contexts[0].conv_history_len)
        self.state_addrs = _ptr_table(self.checkpoints, self.device)
        self.conv_addrs = _ptr_table(self.conv_states, self.device)
        self.state_block_strides = self._strides(self.checkpoints, 0)
        self.conv_block_strides = self._strides(self.conv_states, 0)
        self.conv_dim_strides = self._strides(self.conv_states, 1)
        self.conv_token_strides = self._strides(self.conv_states, 2)
        soa = self.record_layout == "soa"
        self.corrections = tuple(r.correction if soa else r for r in self.records)
        self.keys = tuple(r.key if soa else r for r in self.records)
        self.decays = tuple(r.decay if soa else r for r in self.records)
        self.correction_addrs = _ptr_table(self.corrections, self.device)
        self.key_addrs = _ptr_table(self.keys, self.device)
        self.decay_addrs = _ptr_table(self.decays, self.device)
        self.correction_block_strides = self._strides(self.corrections, 0)
        self.key_block_strides = self._strides(self.keys, 0)
        self.decay_block_strides = self._strides(self.decays, 0)
        size = self.capacity * len(contexts)
        self.source = torch.empty(size, dtype=torch.int32, device=self.device)
        self.length = torch.empty_like(self.source)
        self.final = torch.empty_like(self.source)
        self.boundary = torch.empty_like(self.source)
        self.boundary_len = torch.empty_like(self.source)
        self.valid = torch.empty_like(self.source)
        self.col = torch.empty_like(self.source)

    def _strides(self, tensors, dim):
        return torch.tensor(
            [t.stride(dim) for t in tensors], dtype=torch.int64, device=self.device
        )

    def make_specs(
        self,
        entries,
        sampled,
        idx_mapping,
        persistent_cols,
        persistent_accepted,
        *,
        publish_align=True,
    ):
        contexts = tuple(e.context for e in entries)
        if len(contexts) != len(self.contexts) or any(
            a is not b for a, b in zip(contexts, self.contexts)
        ):
            raise ValueError("grouped recovery bucket belongs to a different KV epoch")
        p = tuple(e.payload for e in entries)
        rows = int(p[0].state_indices.shape[0])
        if not 0 < rows <= self.capacity:
            raise ValueError(
                "grouped recovery rows out of range: "
                f"rows={rows}, record capacity={self.capacity}, "
                f"groups={len(p)}, state_indices shapes="
                f"{[tuple(x.state_indices.shape) for x in p]}"
            )
        if any(int(x.state_indices.shape[0]) != rows for x in p):
            raise ValueError("grouped recovery row counts differ")
        align = p[0].block_table is not None
        if any((x.block_table is not None) != align for x in p):
            raise ValueError("grouped recovery cannot mix align and none plans")
        if any(x.request_indices is None for x in p):
            raise ValueError("grouped recovery requires an explicit native row mapping")
        required = [sampled, idx_mapping, persistent_accepted]
        for x in p:
            required.extend((x.state_indices, x.query_start_loc, x.request_indices))
            if align:
                required.extend((x.block_table, x.num_computed_tokens))
        if align:
            required.append(persistent_cols)
        if any(t is None or t.device != self.device for t in required):
            raise ValueError("grouped recovery inputs must share the checkpoint device")
        if sampled.data_ptr() == persistent_accepted.data_ptr():
            raise ValueError(
                "sampled counts must not alias persistent acceptance output"
            )
        for x in p:
            if x.query_start_loc.numel() < rows + 1:
                raise ValueError("grouped recovery offsets are shorter than their rows")
            if align and (x.block_size is None or x.block_size < self.spec_query_len):
                raise ValueError("align blocks must cover the speculative window")
            if align and x.block_table.shape[1] <= 0:
                raise ValueError("align block table has no columns")
            host_rows = tuple(getattr(x, "request_rows", ()))
            if len(host_rows) == rows and any(
                r < 0 or r >= sampled.numel() or r >= idx_mapping.numel()
                for r in host_rows
            ):
                raise ValueError("native request mapping exceeds the sampled batch")
        def _slots(values, fill=0):
            out = list(values)
            while len(out) < 3:
                out.append(out[-1] if out else fill)
            return out[:3]

        si = _slots([x.state_indices.stride(0) for x in p])
        qsl = _slots([x.query_start_loc.stride(0) for x in p])
        mp = _slots([x.request_indices.stride(0) for x in p])
        tc = _slots([x.block_table.stride(1) if align else 0 for x in p])
        cs = _slots([x.num_computed_tokens.stride(0) if align else 0 for x in p])
        nb = _slots([c.num_blocks for c in contexts])
        bs = _slots([int(x.block_size) if align else 1 for x in p])

        plan = _LaunchSpec(
            kernel=_grouped_plan_kernel,
            args=(
                sampled,
                *_flat_three(x.state_indices for x in p),
                *_flat_three(x.query_start_loc for x in p),
                *_flat_three(x.request_indices for x in p),
                *_flat_three(x.block_table for x in p),
                *_flat_three(x.num_computed_tokens for x in p),
                *_flat_three(int(x.block_table.shape[1]) if align else 1 for x in p),
                *_flat_three(int(x.block_table.stride(0)) if align else 0 for x in p),
                self.source,
                self.length,
                self.final,
                self.boundary,
                self.boundary_len,
                self.valid,
                self.col,
                idx_mapping,
                persistent_cols,
                persistent_accepted,
                rows,
                self.capacity,
                NULL_BLOCK_ID,
            ),
            kwargs={
                # Triton-Ascend's constexpr is not subscriptable, so the
                # per-group strides, pool bound and block size are passed as
                # three explicit scalars and chosen by a compile-time branch
                # inside the kernel.
                "SI_STRIDES0": si[0],
                "SI_STRIDES1": si[1],
                "SI_STRIDES2": si[2],
                "QSL_STRIDES0": qsl[0],
                "QSL_STRIDES1": qsl[1],
                "QSL_STRIDES2": qsl[2],
                "MAP_STRIDES0": mp[0],
                "MAP_STRIDES1": mp[1],
                "MAP_STRIDES2": mp[2],
                "TABLE_COL_STRIDES0": tc[0],
                "TABLE_COL_STRIDES1": tc[1],
                "TABLE_COL_STRIDES2": tc[2],
                "COMPUTED_STRIDES0": cs[0],
                "COMPUTED_STRIDES1": cs[1],
                "COMPUTED_STRIDES2": cs[2],
                "NUM_BLOCKS0": nb[0],
                "NUM_BLOCKS1": nb[1],
                "NUM_BLOCKS2": nb[2],
                "BLOCK_SIZES0": bs[0],
                "BLOCK_SIZES1": bs[1],
                "BLOCK_SIZES2": bs[2],
                "SPEC_QUERY_LEN": self.spec_query_len,
                "NUM_GROUPS": len(p),
                "ALIGN_MODE": align,
                "PUBLISH_ALIGN": align and publish_align,
                "num_warps": 1,
            },
            grid=(rows,),
        )
        conv = self.conv_states[0]
        block_history = triton.next_power_of_2(self.conv_history_len)
        block_d = _conv_dim_tile(block_history)
        d_tiles = triton.cdiv(conv.shape[1], block_d)
        conv_work = d_tiles * rows * len(self.conv_states)
        compact = _LaunchSpec(
            kernel=_grouped_compact_conv_state_kernel,
            args=(
                conv,
                self.conv_addrs,
                self.conv_block_strides,
                self.conv_dim_strides,
                self.conv_token_strides,
                self.source,
                self.length,
                self.final,
                self.boundary,
                self.boundary_len,
                self.layer_groups,
                self.capacity,
                NULL_BLOCK_ID,
                rows,
                conv_work,
                int(conv.shape[1]),
                self.conv_history_len,
                1,
            ),
            kwargs={
                "BLOCK_D": block_d,
                "BLOCK_HISTORY": block_history,
                "ALIGN_MODE": align,
                "NUM_BLOCKS": self.max_blocks,
                "NUM_D_TILES": d_tiles,
                "num_warps": 4,
            },
            grid=(min(conv_work, _launch_cap()),),
        )
        state = self.checkpoints[0]
        _, hv, v, k = state.shape
        bk = triton.next_power_of_2(k)
        bv = _commit_value_tile(bk, v)
        v_tiles = triton.cdiv(v, bv)
        work = v_tiles * rows * len(self.checkpoints) * hv
        rr, kr, dr = self.corrections[0], self.keys[0], self.decays[0]
        soa = self.record_layout == "soa"
        key_strides = kr.stride()[1:] if soa else (0, 0, 0)
        decay_strides = dr.stride()[1:] if soa else (0, 0)
        fold = _LaunchSpec(
            kernel=_grouped_commit_gdn_state_kernel,
            args=(
                state,
                self.state_addrs,
                self.state_block_strides,
                rr,
                self.correction_addrs,
                self.correction_block_strides,
                kr,
                self.key_addrs,
                self.key_block_strides,
                dr,
                self.decay_addrs,
                self.decay_block_strides,
                self.source,
                self.length,
                self.final,
                self.boundary,
                self.boundary_len,
                self.layer_groups,
                self.capacity,
                NULL_BLOCK_ID,
                rows,
                work,
                self.capacity,
                *state.stride()[1:],
                *rr.stride()[1:],
                *key_strides,
                *decay_strides,
                1,
            ),
            kwargs={
                "K": k,
                "V": v,
                "BK": bk,
                "BV": bv,
                "NUM_HEADS": hv,
                "NUM_BLOCKS": self.max_blocks,
                "NUM_VALUE_TILES": v_tiles,
                "NUM_KEY_HEADS": self.num_key_heads,
                "RECORD_SOA": soa,
                "num_warps": 4,
            },
            grid=(min(work, _launch_cap()),),
        )
        return plan, compact, fold


class GroupedCommitCoordinator:
    """One coordinator belongs to one RecoverSSMState, never a module global."""

    def __init__(self):
        self._buckets = {}
        self._seen = set()
        self.steps = 0
        self.last_launch_count = 0
        self.last_group_count = 0
        self.last_rows = 0

    def commit(
        self, metadata, sampled, idx_mapping, persistent_cols, persistent_accepted
    ):
        entries = []
        for m in metadata:
            p = getattr(m, "recoverssm_commit", None)
            builder = getattr(m, "recoverssm_builder", None)
            if p is None or builder is None:
                continue
            context = getattr(m, "recoverssm_context", None)
            if context is None:
                raise RuntimeError(
                    "a serving RecoverSSM commit has no producer binding epoch"
                )
            entries.append(CommitEntry(m, p, context))
        if not entries:
            self.last_launch_count = self.last_group_count = self.last_rows = 0
            return
        aligned = [e for e in entries if e.payload.block_table is not None]
        if aligned:
            if len(aligned) != len(entries):
                raise ValueError(
                    "one recovery step cannot mix align and none ownership"
                )
            logical = _logical_key(aligned[0])
            if len(logical) == 1 or any(
                _logical_key(e) != logical for e in aligned[1:]
            ):
                raise ValueError(
                    "GDN groups must agree on logical rows, query lengths and align block size"
                )
            if any(
                e.context.spec_query_len != entries[0].context.spec_query_len
                for e in entries
            ):
                raise ValueError("GDN groups must agree on the speculative window")
        logical_groups = {}
        for entry in entries:
            key = (_layout_key(entry.context), _logical_key(entry))
            logical_groups.setdefault(key, []).append(entry)
        # Three flat pointer slots cover the primary Qwen geometry. Larger
        # homogeneous group sets remain supported through multiple buckets and
        # one AND-valid publication after all buckets have physically committed.
        grouped = [
            items[start : start + 3]
            for items in logical_groups.values()
            for start in range(0, len(items), 3)
        ]
        live = {}
        launches = 0
        for subset in grouped:
            contexts = tuple(e.context for e in subset)
            epoch = tuple(id(c) for c in contexts)
            bucket = self._buckets.get(epoch)
            if bucket is None:
                prepared = getattr(
                    subset[0].metadata.recoverssm_builder,
                    "_prepared_group_bucket",
                    None,
                )
                if (
                    prepared is not None
                    and len(prepared.contexts) == len(contexts)
                    and all(a is b for a, b in zip(prepared.contexts, contexts))
                ):
                    bucket = prepared
                else:
                    bucket = GroupedCommitBucket(contexts)
            live[epoch] = bucket
            specs = bucket.make_specs(
                subset,
                sampled,
                idx_mapping,
                persistent_cols,
                persistent_accepted,
                publish_align=len(grouped) == 1,
            )
            for spec in specs:
                spec.launch()
            launches += len(specs)
        if aligned and len(grouped) > 1:
            buckets = tuple(live.values())
            request_indices = entries[0].payload.request_indices
            publish = _publish_spec(
                buckets,
                request_indices,
                idx_mapping,
                persistent_cols,
                persistent_accepted,
                int(entries[0].payload.state_indices.shape[0]),
            )
            publish.launch()
            launches += 1
        # Retain only current epochs. A sleep/wake must not keep old KV/records
        # alive indefinitely through a coordinator cache.
        self._buckets = live
        self.steps += 1
        self.last_launch_count = launches
        self.last_group_count = len(entries)
        self.last_rows = int(entries[0].payload.state_indices.shape[0])
        reason = (len(entries), len(grouped))
        if reason not in self._seen:
            self._seen.add(reason)
            logger.info(
                "GDN recovery coordinator: groups=%d buckets=%d launches=%d "
                "(separate buckets preserve heterogeneous layouts or row geometry)",
                len(entries),
                len(grouped),
                launches,
            )


def prepare_grouped_commit(builders, block_tables=None, *, num_computed_tokens=None):
    """Register serving-shaped grouped variants before requests can arrive.

    These are compile-only launches. Layer pointers refer to the real contexts;
    dummy inputs cover optional align pointers and native int64 row mappings.
    The running coordinator adopts the prepared bucket without reallocating it.
    """
    from types import SimpleNamespace

    groups = {}
    for builder in builders:
        context = builder._get_context()
        if context is not None:
            align = builder.vllm_config.cache_config.mamba_cache_mode == "align"
            key = (_layout_key(context), align, int(builder.kv_cache_spec.block_size))
            groups.setdefault(key, []).append((builder, context))
    total = 0
    prepared_buckets = []
    prepared_entries = []
    chunks = [
        items[start : start + 3]
        for items in groups.values()
        for start in range(0, len(items), 3)
    ]
    for items in chunks:
        contexts = tuple(c for _, c in items)
        bucket = GroupedCommitBucket(contexts)
        entries = []
        for builder, context in items:
            dev = bucket.device
            q = context.spec_query_len
            native_state_table = getattr(builder, "spec_state_indices_tensor", None)
            # The native table may not be materialised yet at startup (notably on
            # the eager path, where it is only filled once a batch is built), so a
            # zero-row buffer is not a usable signature source: the preparation
            # needs exactly one row to compile against.
            if (
                native_state_table is not None
                and getattr(native_state_table, "shape", (0,))[0] >= 1
            ):
                table = native_state_table[:1]
            else:
                table = torch.empty((1, q), dtype=torch.int32, device=dev)
            align = builder.vllm_config.cache_config.mamba_cache_mode == "align"
            block_size = int(builder.kv_cache_spec.block_size)
            if align:
                group_id = getattr(builder, "_gdn_kv_cache_group_id", None)
                bound = getattr(block_tables, "input_block_tables", None)
                if bound is None or group_id is None or not 0 <= group_id < len(bound):
                    raise RuntimeError(
                        "grouped align preparation requires the bound native group table"
                    )
                # Preserve the allocator's exact width, dtype and row/column
                # strides, including lookahead/padding. max_model_len alone
                # cannot reconstruct this serving signature.
                bound_table = bound[group_id][:1]
            else:
                bound_table = None
            # Every buffer the preparation borrows must actually hold the rows it
            # is sliced by: on the eager path these native buffers can still be
            # empty at startup, and a zero-row slice compiles a different (and
            # unusable) signature.
            def _at_least(tensor, rows, *, dtype=torch.int32):
                if tensor is not None and getattr(tensor, "shape", (0,))[0] >= rows:
                    return tensor[:rows]
                return torch.empty(
                    (rows,) if getattr(tensor, "dim", lambda: 1)() == 1 else (rows, q),
                    dtype=dtype,
                    device=dev,
                )

            qsl = _at_least(
                getattr(builder, "spec_query_start_loc", None), 2,
                dtype=getattr(
                    getattr(builder, "spec_query_start_loc", None), "dtype", torch.int32
                ),
            )
            payload = SimpleNamespace(
                state_indices=table[:, 0],
                query_start_loc=qsl,
                request_indices=_at_least(
                    getattr(builder, "spec_sequence_indices", None), 1,
                    dtype=getattr(
                        getattr(builder, "spec_sequence_indices", None),
                        "dtype",
                        torch.int32,
                    ),
                ),
                block_table=bound_table,
                num_computed_tokens=(
                    num_computed_tokens[:1]
                    if num_computed_tokens is not None
                    else torch.empty(1, dtype=torch.int32, device=dev)
                )
                if align
                else None,
                block_size=block_size if align else None,
                request_rows=(0,),
                query_lens=(q,),
            )
            entries.append(CommitEntry(None, payload, context))
        dev = bucket.device
        sampled = torch.empty(1, dtype=torch.int32, device=dev)
        mapping = torch.empty(1, dtype=torch.int32, device=dev)
        cols = torch.empty(1, dtype=torch.int32, device=dev)
        accepted = torch.empty(1, dtype=torch.int32, device=dev)
        for spec in bucket.make_specs(
            entries,
            sampled,
            mapping,
            cols,
            accepted,
            publish_align=len(chunks) == 1,
        ):
            spec.compile_only()
            total += 1
        prepared_buckets.append(bucket)
        prepared_entries.extend(entries)
        for builder, _ in items:
            builder._prepared_group_bucket = bucket
    if (
        len(prepared_buckets) > 1
        and prepared_entries[0].payload.block_table is not None
    ):
        block_sizes = {e.payload.block_size for e in prepared_entries}
        if len(block_sizes) != 1:
            raise ValueError("shared GDN align ownership requires one block size")
        spec = _publish_spec(
            tuple(prepared_buckets),
            prepared_entries[0].payload.request_indices,
            mapping,
            cols,
            accepted,
            1,
        )
        spec.compile_only()
        total += 1
    return total


def _run_grouped_selftest(
    *,
    device,
    state_dtype=torch.bfloat16,
    key_dtype=torch.bfloat16,
    num_value_heads=6,
    num_key_heads=2,
    key_dim=32,
    value_dim=32,
    spec_query_len=4,
):
    """Execute all three grouped kernels on owned deterministic synthetic pages.

    Two groups have different physical tables and two layers each. References
    cover every page, including source/boundary aliasing, untouched/null pages,
    reordered request rows and zero/partial/full acceptance. No random generator
    or live model state is read or changed.
    """
    from types import SimpleNamespace

    from .kernels import GDNRecoverSSMCommitContext, ReplayRecordBuffers

    if spec_query_len < 1:
        raise ValueError("grouped lifecycle selftest requires a positive window")
    q = int(spec_query_len)
    hv, hk, v, k = num_value_heads, num_key_heads, value_dim, key_dim
    blocks, rows, history, block_size = 8, 3, 3, max(q, 4)
    conv_dim = 16
    cpu_tables = (
        torch.tensor(
            [[1, 2, 7], [4, 5, 6], [0, 0, 0]], dtype=torch.int32, device="cpu"
        ),
        torch.tensor(
            [[3, 4, 7], [1, 2, 6], [0, 0, 0]], dtype=torch.int32, device="cpu"
        ),
    )
    computed = torch.tensor([block_size - 1, 1, 0], dtype=torch.int32, device="cpu")
    host_rows = (1, 0, 2)
    qsl_cpu = torch.tensor([0, q, 2 * q, 2 * q], dtype=torch.int32, device="cpu")
    cases = ((q, max(1, q - 1), 0), (0, 1, 0))
    for case in cases:
        entries, expected_states, expected_convs = [], [], []
        for group in range(2):
            states, convs, records = [], [], []
            for layer in range(2):
                initial = (
                    torch.arange(
                        blocks * hv * v * k, dtype=torch.float32, device="cpu"
                    ).reshape(blocks, hv, v, k)
                    % 31
                    - 15
                ) / 1000
                initial = (initial + (group * 2 + layer) / 100).to(state_dtype)
                conv_initial = (
                    torch.arange(
                        blocks * conv_dim * (history + q - 1),
                        dtype=torch.float32,
                        device="cpu",
                    ).reshape(blocks, conv_dim, history + q - 1)
                    % 97
                )
                conv_initial = (conv_initial + group * 100 + layer * 10).to(state_dtype)
                corr = (
                    torch.arange(
                        rows * hv * q * v, dtype=torch.float32, device="cpu"
                    ).reshape(rows, hv, q, v)
                    % 7
                    - 3
                ) / 100
                keys = (
                    torch.arange(
                        rows * hk * q * k, dtype=torch.float32, device="cpu"
                    ).reshape(rows, hk, q, k)
                    % 11
                    - 5
                ) / 100
                keys = keys.to(key_dtype)
                decay = torch.full(
                    (rows, hv, q), 0.875, dtype=torch.float32, device="cpu"
                )
                states.append(initial.to(device))
                convs.append(conv_initial.to(device))
                records.append(
                    ReplayRecordBuffers(
                        corr.to(device), keys.to(device), decay.to(device)
                    )
                )
                want_state = initial.clone()
                want_conv = conv_initial.clone()
                for row, request in enumerate(host_rows):
                    n = min(
                        max(int(case[request]), 0),
                        int(qsl_cpu[row + 1] - qsl_cpu[row]),
                        q,
                    )
                    if n == 0:
                        continue
                    c = int(computed[request])
                    src = int(cpu_tables[group][request, 0])
                    final_col = max(c + n - 1, 0) // block_size
                    final = int(cpu_tables[group][request, final_col])
                    boundary_at = (c // block_size + 1) * block_size
                    crosses = c + n >= boundary_at
                    boundary_len = boundary_at - c
                    boundary = (
                        int(cpu_tables[group][request, boundary_at // block_size - 1])
                        if crosses
                        else 0
                    )
                    h = initial[src].double()
                    for token in range(n):
                        key = (
                            keys[row, :, token]
                            .double()
                            .repeat_interleave(hv // hk, dim=0)
                        )
                        correction = corr[row, :, token].double()
                        h = h * decay[row, :, token].double()[:, None, None]
                        h = h + correction[:, :, None] * key[:, None, :]
                        if crosses and token + 1 == boundary_len and boundary != final:
                            want_state[boundary] = h.to(state_dtype)
                    want_state[final] = h.to(state_dtype)
                    # Both reads use the original source before either write.
                    if crosses:
                        want_conv[boundary, :, :history] = conv_initial[
                            src, :, boundary_len - 1 : boundary_len - 1 + history
                        ]
                    want_conv[final, :, :history] = conv_initial[
                        src, :, n - 1 : n - 1 + history
                    ]
                expected_states.append(want_state)
                expected_convs.append(want_conv)
            context = GDNRecoverSSMCommitContext.from_tensors(
                convs,
                states,
                records,
                spec_query_len=q,
                max_num_reqs=rows,
                conv_dim_first=True,
            )
            state_table = torch.zeros((rows, q), dtype=torch.int32, device=device)
            state_table[:, 0].copy_(
                torch.tensor(
                    [int(cpu_tables[group][request, 0]) for request in host_rows],
                    dtype=torch.int32,
                    device=device,
                )
            )
            payload = SimpleNamespace(
                state_indices=state_table[:, 0],
                query_start_loc=qsl_cpu.to(device),
                request_indices=torch.tensor(
                    host_rows, dtype=torch.int64, device=device
                ),
                block_table=cpu_tables[group].to(device),
                num_computed_tokens=computed.to(device),
                block_size=block_size,
                request_rows=host_rows,
                query_lens=(q, q, 0),
            )
            entries.append(CommitEntry(None, payload, context))
        bucket = GroupedCommitBucket(tuple(e.context for e in entries))
        sampled = torch.tensor(case, dtype=torch.int32, device=device)
        mapping = torch.tensor([2, 0, -1], dtype=torch.int32, device=device)
        cols = torch.full((4,), -9, dtype=torch.int32, device=device)
        accepted = torch.full((4,), 7, dtype=torch.int32, device=device)
        for spec in bucket.make_specs(entries, sampled, mapping, cols, accepted):
            spec.launch()
        actual_cols = cols.cpu()
        actual_accepted = accepted.cpu()
        for request, slot in ((0, 2), (1, 0)):
            n = int(case[request])
            expected_col = (int(computed[request]) + n - 1) // block_size if n else -9
            if int(actual_cols[slot]) != expected_col or int(actual_accepted[slot]) != (
                1 if n else 7
            ):
                raise RuntimeError(
                    "GDN grouped selftest: persistent owner publication mismatch"
                )
        if (
            int(actual_cols[1]) != -9
            or int(actual_cols[3]) != -9
            or int(actual_accepted[1]) != 7
            or int(actual_accepted[3]) != 7
        ):
            raise RuntimeError(
                "GDN grouped selftest: inactive persistent slot was changed"
            )
        tolerance = 1e-5 if state_dtype == torch.float32 else 5e-3
        for index, (state, conv) in enumerate(
            zip(bucket.checkpoints, bucket.conv_states)
        ):
            torch.testing.assert_close(
                state.cpu().float(),
                expected_states[index].float(),
                atol=tolerance,
                rtol=tolerance,
            )
            if not torch.equal(conv.cpu(), expected_convs[index]):
                raise RuntimeError(
                    "GDN grouped selftest: convolution boundary/final mismatch"
                )
    logger.info(
        "GDN_GROUPED_SELFTEST PASS groups=2 layers_per_group=2 cases=%d "
        "(actual plan/conv/fold; reordered rows, null/no-work, boundary and full acceptance)",
        len(cases),
    )
    return True


def run_grouped_selftest(**kwargs):
    """Fail readiness explicitly if the grouped device lifecycle check fails."""
    try:
        return _run_grouped_selftest(**kwargs)
    except Exception:
        logger.exception("GDN_GROUPED_SELFTEST FAIL")
        raise
