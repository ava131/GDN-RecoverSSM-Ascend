# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend GDN attention metadata for RecoverSSM speculative decode.

Port of the upstream GDN RecoverSSM builder (vllm-project/vllm PR #61309,
``vllm/v1/attention/backends/gdn_recoverssm.py`` @ ``bf0219d8``) onto the
Ascend GDN builder (:class:`AscendGDNAttentionMetadataBuilder`).

What RecoverSSM changes, relative to the stock Ascend spec-decode path:

* The recurrent checkpoint is **read-only** during verify. The spec window's
  state is never written per draft position, so the recurrent kernel can no
  longer be ``recurrent_gated_delta_rule`` (which walks the per-position state
  slots); the layer calls :func:`gdn_recoverssm_verify` instead.
* Verify writes one compact FP32 record per token,
  ``[correction(V), normalised_key(K), decay(1)]``, into the layer's replay
  cache. After sampling, :class:`GDNRecoverSSMCommitContext` folds the accepted
  prefix back into the checkpoint exactly once.
* ``num_accepted_tokens`` for spec rows is pinned to 1: the checkpoint already
  holds the accepted state, so the next step starts from a compacted window.

Ascend-specific differences from upstream (see ``docs/FEASIBILITY.md``):

* Upstream forces **every** post-prefill row onto the spec path and rewrites
  ``num_decode_draft_tokens_cpu``. This port keeps Ascend's own spec/non-spec
  split (:attr:`AscendGDNAttentionMetadataBuilder` already routes all rows with
  drafts onto the spec path under MTP), and only rewrites the metadata that the
  commit needs. A row without drafts keeps the stock non-spec path, which
  writes its own state -- consistent, and never committed twice.
* Upstream sizes the spec state indices to ``[:, :1]``; Ascend keeps the
  builder's ``[:, :num_spec + 1]`` table but the layer and the commit only ever
  read column 0. The extra columns are unused because the Mamba page count is
  already reduced to ``num_speculative_blocks = 0`` by ``--use-replayssm``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)

logger = init_logger("vllm.gdn_recoverssm_ascend.builder")

# Pending align bookkeeping for the step being postprocessed. The commit runs
# inside ``RecoverSSMState.commit_step`` and the native align postcopy runs right
# after it, so the update has to be applied in between -- the wrapper around
# ``postprocess_state`` consumes this and clears it.
_PENDING_ALIGN: dict | None = None


def take_pending_align():
    """Pop the pending align update (consumed once per step)."""
    global _PENDING_ALIGN
    pending, _PENDING_ALIGN = _PENDING_ALIGN, None
    return pending


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip() not in ("", "0", "false", "False")


@dataclass
class AscendGDNRecoverSSMCommitMetadata:
    """Everything the post-sampling commit needs, captured at build time."""

    # [num_spec_decodes] -- the checkpoint (column 0) page of each spec row.
    state_indices: torch.Tensor
    # [num_spec_decodes + 1]
    query_start_loc: torch.Tensor
    # Batch rows of the spec requests; None when they are exactly 0..n-1.
    request_indices: torch.Tensor | None
    # align mode only:
    block_table: torch.Tensor | None
    num_computed_tokens: torch.Tensor | None
    block_size: int | None


# Host-only counter: the commit runs in Python after sampling (outside the graph),
# so it is the one reliable signal that the RecoverSSM path is live during
# serving. FULL graph replay does not re-run Python, so verify_calls stops
# growing after capture and cannot be used for this.
_COMMIT_STEPS = 0


@dataclass
class AscendGDNRecoverSSMMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    """GDN metadata that also carries the RecoverSSM commit payload."""

    recoverssm_commit: AscendGDNRecoverSSMCommitMetadata | None = None
    # The builder resolves (and re-resolves) the commit context, so a metadata
    # object created before the KV cache was bound still commits correctly.
    recoverssm_builder: Any = field(default=None, repr=False, compare=False)

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        commit = self.recoverssm_commit
        if commit is None or self.recoverssm_builder is None:
            return None
        context = self.recoverssm_builder._get_context()
        if context is None:
            logger.warning(
                "Ascend GDN RecoverSSM: commit skipped, the KV cache is not bound yet"
            )
            return None
        n = int(commit.state_indices.shape[0])
        if n == 0:
            return None
        global _COMMIT_STEPS
        _COMMIT_STEPS += 1
        if _COMMIT_STEPS <= 3 or _COMMIT_STEPS % 200 == 0:
            logger.info(
                "Ascend GDN RecoverSSM commit: step=%d rows=%d mode=%s "
                "(checkpoint advanced by the GDN commit kernel)",
                _COMMIT_STEPS,
                n,
                "align" if commit.block_table is not None else "none",
            )
        context.commit(
            num_accepted_tokens,
            commit.state_indices,
            commit.query_start_loc,
            request_indices=commit.request_indices,
            block_table=commit.block_table,
            num_computed_tokens=commit.num_computed_tokens,
            mamba_block_size=commit.block_size,
        )
        # Deliberately no RecoverSSMPostprocessMetadata: the generic kernel would
        # advance the running column with (computed + sampled) // block, which is
        # not the column this commit wrote. The GDN-specific bookkeeping (running
        # column = commit target, persistent accepted = 1) is staged here and
        # applied by the postprocess hook after this commit and before the native
        # align postcopy.
        if commit.block_table is not None:
            # The speculative window length lives on the BUILDER; ``self`` here is
            # the metadata object, which has no such attribute (this line only
            # runs in align mode, so the none-mode runs never reached it).
            builder = self.recoverssm_builder
            num_spec = getattr(builder, "num_spec", None)
            window = None if num_spec is None else int(num_spec) + 1
            _stage_align_update(commit, num_accepted_tokens, n, window)
        return None


def _stage_align_update(
    commit, num_accepted_tokens, n: int, spec_query_len: int | None
) -> None:
    """Record ``(rows, final_col)`` so postprocess can fix the running column.

    RecoverSSM restores the accepted state itself, so the persistent running
    column must become the column the commit actually wrote
    (``floor(max(C + A - 1, 0) / block)``) and the persistent accepted count must
    go back to the neutral 1. The generic Recover kernel uses a different column
    formula, so returning its metadata would be wrong; this is the GDN-specific
    bookkeeping instead.
    """
    global _PENDING_ALIGN
    try:
        import torch

        rows = commit.request_indices
        if rows is None:
            # Without the batch-row mapping we cannot address the persistent
            # request slots; guessing an identity mapping would write another
            # request's column. Skip and say so.
            logger.warning(
                "Ascend GDN RecoverSSM: align bookkeeping skipped, the spec rows "
                "have no batch-row mapping"
            )
            return
        computed = commit.num_computed_tokens[rows.long()]
        qsl = commit.query_start_loc
        qlen = (qsl[1 : n + 1] - qsl[:n]).clamp(min=0)
        # Same clamp chain as the plan kernel: max(0) -> query_len -> spec window.
        accepted = num_accepted_tokens[rows.long()].clamp(min=0)
        accepted = torch.minimum(accepted, qlen)
        if spec_query_len is not None:
            # Same bound as the plan kernel's SPEC_QUERY_LEN clamp; without it the
            # column could point past the window the commit actually folded.
            accepted = accepted.clamp(max=int(spec_query_len))
        width = int(commit.block_table.shape[1])
        col = ((computed + accepted - 1).clamp(min=0) // int(commit.block_size)).clamp(
            max=width - 1
        )
        _PENDING_ALIGN = {"rows": rows, "cols": col}
    except Exception as exc:  # never break the step on bookkeeping
        logger.warning("Ascend GDN RecoverSSM: align bookkeeping skipped (%s)", exc)


class AscendGDNRecoverSSMMetadataBuilder(AscendGDNAttentionMetadataBuilder):
    """Ascend GDN builder with the RecoverSSM commit payload attached."""

    # Uniform spec-decode batches replay as full graphs. The commit runs after
    # sampling, outside the graph, on the staged tensors.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._context: Any = None
        self._ctx_key: tuple | None = None
        self._ctx_unbound_logged = False
        self._mapping_logged = False
        self._capturing = False
        # A full-graph replay re-uses the addresses captured at capture time, so
        # the accepted-token input must live in one **persistent** buffer that is
        # rewritten in place. Allocating a clone per step (the previous
        # behaviour) hands the graph a fresh address every step while the
        # captured kernel keeps reading the old one; the old storage can even be
        # freed and reused. Neutral value is 1: the checkpoint already holds the
        # accepted state, so the next convolution starts a one-token window.
        max_rows = max(
            vllm_config.scheduler_config.max_num_seqs,
            int(getattr(self, "decode_cudagraph_max_bs", 0) or 0),
        )
        self._accepted_buffer = torch.ones(
            max_rows, dtype=torch.int32, device=device
        )

    # ------------------------------------------------------------------ context
    def _get_context(self):
        """Create (or refresh) the per-layer commit context.

        The context caches the **data pointers** of every layer's state pages, so
        it must be rebuilt whenever the KV cache is re-bound: vLLM allocates the
        real cache after the profiling run, and a sleep/wake or re-profile
        replaces the tensors. Stale pointers turn the commit kernel into a write
        to freed device memory, which surfaces much later as "MTE accesses an
        invalid GM address" or "SUSPECT REMOTE ERROR". Upstream main hit the same
        class of bug (vllm#63045: "the commit context is rebuilt when the KV
        cache is re-bound").
        """
        from .kernels import GDNRecoverSSMCommitContext

        fc = self.vllm_config.compilation_config.static_forward_context
        layers = [fc[name] for name in self.layer_names]

        kv = getattr(layers[0], "kv_cache", None)
        if kv is None or len(kv) < 2:
            # KV cache not bound yet (profiling / early build).
            if not self._ctx_unbound_logged:
                self._ctx_unbound_logged = True
                logger.warning(
                    "Ascend GDN RecoverSSM: layers expose %s KV-cache entries; "
                    "commit context deferred until the KV cache is bound.",
                    "no" if kv is None else len(kv),
                )
            return None

        # Signature of EVERY participating layer's bound state tensors. Keying on
        # the first layer alone misses a partial re-bind (or a layout change) on
        # any other layer. Host-only: shape/stride/dtype/data_ptr, no device read.
        key = tuple(
            (
                layer_name,
                tuple(
                    (
                        t.data_ptr(),
                        tuple(t.shape),
                        tuple(t.stride()),
                        str(t.dtype),
                    )
                    for t in layer.kv_cache
                ),
            )
            for layer_name, layer in zip(self.layer_names, layers)
        )
        if self._context is not None and key == self._ctx_key:
            return self._context
        if self._context is not None:
            logger.warning(
                "Ascend GDN RecoverSSM: KV cache re-bound, rebuilding the commit "
                "context (stale pointers would write to freed device memory)."
            )

        # Plan buffers must cover the graph batch too: a full-graph step can
        # carry more rows than max_num_seqs.
        max_rows = max(
            self.vllm_config.scheduler_config.max_num_seqs,
            int(getattr(self, "decode_cudagraph_max_bs", 0) or 0),
        )
        self._context = GDNRecoverSSMCommitContext.create(
            layers,
            spec_query_len=1 + self.num_spec,
            max_num_reqs=max_rows,
            # Records are only written by real requests; the padded graph rows are
            # skipped by the kernels, so cap the scratch at max_num_seqs.
            record_rows=min(
                max_rows, int(self.vllm_config.scheduler_config.max_num_seqs or max_rows)
            ),
        )
        self._ctx_key = key
        logger.info(
            "Ascend GDN RecoverSSM commit context: %d layers, spec_query_len=%d, "
            "state tensors per layer=%d, replay shape=%s, plan rows=%d",
            len(layers),
            1 + self.num_spec,
            len(kv),
            tuple(self._context.records[0].shape),
            max_rows,
        )
        return self._context

    # -------------------------------------------------------------------- build
    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
    ) -> GDNAttentionMetadata:
        meta = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
            num_actual_reqs=num_actual_reqs,
        )

        if meta.spec_state_indices_tensor is None or meta.spec_query_start_loc is None:
            # Nothing to commit (no speculative rows this step).
            return meta
        # Authoritative row count: the offset tensor's length. ``num_spec_decodes``
        # may include graph-padded rows, and the stock Ascend builder derives the
        # same count as ``spec_query_start_loc.size(0) - 1``.
        qsl_rows = int(meta.spec_query_start_loc.shape[0]) - 1
        # Full graph materialises the offsets to graph_request_count + 1 with
        # zero-length dummy segments; the logical count is the smaller one.
        n = min(int(meta.num_spec_decodes), qsl_rows)
        if n <= 0:
            return meta

        # Only the checkpoint column is meaningful under RecoverSSM.
        state_indices = meta.spec_state_indices_tensor[:n, 0]
        query_start_loc = meta.spec_query_start_loc[: n + 1]

        # The checkpoint already holds the accepted state, so the next step's
        # conv update must start from the compacted window: one accepted token.
        # Allocate from the actual tensor shape: the spec index table can be
        # sized by the graph batch, which may exceed max_num_seqs.
        # Stable address + in-place neutral value (see __init__). Every row gets 1:
        # real spec rows because the checkpoint is compacted, padding rows because
        # the framework's dummy value is 1 as well.
        #
        # The capacity is fixed before capture. Growing the buffer afterwards would
        # hand new addresses to the Python-side launch while the captured graph
        # keeps replaying the old ones, so an oversized batch is a lifecycle event
        # (recapture), not something this path may paper over.
        need = int(meta.spec_state_indices_tensor.shape[0])
        capacity = self._accepted_buffer.numel()
        if need > capacity:
            raise RuntimeError(
                "Ascend GDN RecoverSSM: speculative batch of "
                f"{need} rows exceeds the {capacity}-row accepted buffer that was "
                "sized before graph capture; recapture with a larger "
                "decode_cudagraph_max_bs instead of growing it here"
            )
        num_accepted = self._accepted_buffer[:need]
        num_accepted.fill_(1)

        # The layer's convolution reads the **nested** descriptor, which the base
        # builder created before we rewrote the top-level tensor. Rebinding both
        # is required: patching only the top level left the conv window using the
        # previous step's accepted count (recorded evidence in
        # docs/reference/... and in the review on issue #1, item 3).
        spec_decode = getattr(meta, "spec_decode_metadata", None)
        conv_spec = getattr(spec_decode, "spec_causal_conv1d", None)
        if conv_spec is not None and hasattr(conv_spec, "num_accepted_tokens"):
            conv_spec.num_accepted_tokens = num_accepted

        m = common_attn_metadata
        align = self.vllm_config.cache_config.mamba_cache_mode == "align"
        block_table = None
        num_computed = None
        block_size = None
        if align:
            block_table = m.block_table_tensor
            num_computed = m.compute_num_computed_tokens()
            block_size = self.kv_cache_spec.block_size

        # ``spec_sequence_masks`` is indexed by **batch row** and
        # ``spec_state_indices_tensor`` was gathered with
        # ``nonzero(spec_sequence_masks)``, so spec row i corresponds to batch row
        # ``nonzero(masks)[i]``. The commit plan needs exactly that mapping:
        # identity here is only correct when the spec rows happen to be the first
        # n rows, which is false as soon as chunked prefill puts a prefill row in
        # front of a decode row.
        request_indices = None
        masks = meta.spec_sequence_masks
        if masks is not None and masks.numel() > 0:
            rows = masks.nonzero(as_tuple=False).flatten()
            if rows.numel() < n:
                # Never guess: a wrong mapping folds another request's records
                # into this request's checkpoint (silent corruption).
                logger.warning(
                    "Ascend GDN RecoverSSM: %d spec rows but only %d carry the "
                    "spec mask; skipping the commit for this step.",
                    n,
                    rows.numel(),
                )
                return meta
            request_indices = rows[:n].to(torch.int32)
            if not self._mapping_logged:
                self._mapping_logged = True
                identity = bool(
                    torch.equal(
                        request_indices,
                        torch.arange(n, dtype=torch.int32, device=request_indices.device),
                    )
                )
                if not identity:
                    logger.info(
                        "Ascend GDN RecoverSSM: spec rows are not the batch "
                        "prefix; commit maps rows %s", request_indices.tolist()
                    )
        elif n > 0:
            logger.warning(
                "Ascend GDN RecoverSSM: no spec mask on the metadata; skipping the "
                "commit for this step."
            )
            return meta

        commit = AscendGDNRecoverSSMCommitMetadata(
            state_indices=state_indices,
            query_start_loc=query_start_loc,
            request_indices=request_indices,
            block_table=block_table,
            num_computed_tokens=num_computed,
            block_size=block_size,
        )

        base = {f.name: getattr(meta, f.name) for f in fields(meta)}
        base["num_accepted_tokens"] = num_accepted
        out = AscendGDNRecoverSSMMetadata(
            **base,
            recoverssm_commit=commit,
            recoverssm_builder=self,
        )
        # The Ascend builder attaches descriptors that are *not* dataclass
        # fields (``spec_decode_metadata``, ``non_spec_prefill_metadata``,
        # ``non_spec_decode_metadata``). Rebuilding from ``fields(meta)`` alone
        # dropped them, and the layer immediately reads
        # ``attn_metadata.spec_decode_metadata.spec_causal_conv1d``. Copy every
        # instance attribute the new object does not already carry.
        for key, value in vars(meta).items():
            if key not in base:
                setattr(out, key, value)
        return out

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata):
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False


class AscendGDNRecoverSSMBackend(AscendGDNAttentionBackend):
    """GDN backend that builds RecoverSSM-aware metadata."""

    @staticmethod
    def get_name() -> str:
        return "ASCEND_GDN_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[AscendGDNRecoverSSMMetadataBuilder]:
        return AscendGDNRecoverSSMMetadataBuilder


__all__ = [
    "AscendGDNRecoverSSMBackend",
    "AscendGDNRecoverSSMCommitMetadata",
    "AscendGDNRecoverSSMMetadata",
    "AscendGDNRecoverSSMMetadataBuilder",
]
