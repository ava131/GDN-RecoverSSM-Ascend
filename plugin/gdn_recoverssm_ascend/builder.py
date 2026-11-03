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

logger = init_logger(__name__)


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


@dataclass
class AscendGDNRecoverSSMMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    """GDN metadata that also carries the RecoverSSM commit payload."""

    recoverssm_commit: AscendGDNRecoverSSMCommitMetadata | None = None
    recoverssm_context: Any = field(default=None, repr=False, compare=False)

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        commit = self.recoverssm_commit
        if commit is None or self.recoverssm_context is None:
            return None
        n = int(commit.state_indices.shape[0])
        if n == 0:
            return None
        self.recoverssm_context.commit(
            num_accepted_tokens,
            commit.state_indices,
            commit.query_start_loc,
            request_indices=commit.request_indices,
            block_table=commit.block_table,
            num_computed_tokens=commit.num_computed_tokens,
            mamba_block_size=commit.block_size,
        )
        if commit.block_table is None:
            return None
        assert commit.block_size is not None and commit.num_computed_tokens is not None
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=n,
            request_indices=commit.request_indices,
            block_table=commit.block_table,
            num_computed_tokens=commit.num_computed_tokens,
            block_size=commit.block_size,
        )


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
        self._capturing = False

    # ------------------------------------------------------------------ context
    def _get_context(self):
        """Create the per-layer commit context lazily.

        Must be lazy: at builder-construction time the layers are not bound to
        their KV caches yet.
        """
        if self._context is None:
            from .kernels import GDNRecoverSSMCommitContext

            fc = self.vllm_config.compilation_config.static_forward_context
            layers = [fc[name] for name in self.layer_names]
            self._context = GDNRecoverSSMCommitContext.create(
                layers,
                spec_query_len=1 + self.num_spec,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
            first = layers[0]
            kv = getattr(first, "kv_cache", ())

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
        n = int(meta.spec_query_start_loc.shape[0]) - 1
        if n <= 0:
            return meta

        # Only the checkpoint column is meaningful under RecoverSSM.
        state_indices = meta.spec_state_indices_tensor[:n, 0]
        query_start_loc = meta.spec_query_start_loc[: n + 1]

        # The checkpoint already holds the accepted state, so the next step's
        # conv update must start from the compacted window: one accepted token.
        # Allocate from the actual tensor shape: the spec index table can be
        # sized by the graph batch, which may exceed max_num_seqs.
        if meta.num_accepted_tokens is not None:
            num_accepted = meta.num_accepted_tokens.clone()
            num_accepted[:n] = 1
        else:
            num_accepted = torch.ones(
                meta.spec_state_indices_tensor.shape[0],
                dtype=torch.int32,
                device=state_indices.device,
            )

        m = common_attn_metadata
        align = self.vllm_config.cache_config.mamba_cache_mode == "align"
        block_table = None
        num_computed = None
        block_size = None
        if align:
            block_table = m.block_table_tensor
            num_computed = m.compute_num_computed_tokens()
            block_size = self.kv_cache_spec.block_size

        # Ascend's spec rows may be a non-contiguous subset of the batch rows, so
        # the commit plan gets the batch-row mapping. It is always passed (instead
        # of only when it differs from the identity prefix) because deciding that
        # would need a device->host sync on the hot path; one extra int32 load per
        # program in the plan kernel is cheaper.
        request_indices = None
        masks = meta.spec_sequence_masks
        if masks is not None and masks.numel() == meta.spec_state_indices_tensor.shape[0]:
            request_indices = masks.nonzero(as_tuple=False).flatten()[:n].to(torch.int32)
            if request_indices.numel() != n:
                # Padding rows were counted; fall back to the identity mapping and
                # let the plan kernel treat row i as batch row i.
                logger.warning_once(
                    "GDN RecoverSSM: %d spec rows but %d masked rows; the commit "
                    "plan falls back to the identity row mapping.",
                    n,
                    request_indices.numel(),
                )
                request_indices = None
        elif masks is not None:
            # Graph-padded state buffers: the builder materialises graph-sized
            # tensors after building on the logical batch, so the mask and the
            # index table have different lengths. Identity is the documented
            # layout for the padded case.
            logger.warning_once(
                "GDN RecoverSSM: spec mask length %d != state index rows %d; "
                "using the identity row mapping.",
                masks.numel(),
                meta.spec_state_indices_tensor.shape[0],
            )

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
            recoverssm_context=self._get_context(),
        )
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
