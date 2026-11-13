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

logger = init_logger("vllm.gdn_recoverssm_ascend.builder")


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
    # CPU-only identities for deciding whether groups share logical rows. The
    # physical page tables are deliberately NOT shared between groups.
    request_rows: tuple[int, ...] = ()
    query_lens: tuple[int, ...] = ()


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
    # The producer and consumer share one immutable binding epoch. Resolving a
    # fresh context only after verify would lose the records just produced.
    recoverssm_context: Any = field(default=None, repr=False, compare=False)

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        commit = self.recoverssm_commit
        if commit is None or self.recoverssm_builder is None:
            return None
        if commit.block_table is not None:
            raise RuntimeError(
                "GDN align recovery must run through the step coordinator so "
                "persistent ownership is published from the same valid plan"
            )
        context = self.recoverssm_context
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
        # The model-state coordinator normally handles this method's entire
        # step. A direct caller still gets the same physical recovery; align
        # publication requires the model-state persistent slot mapping.
        # Deliberately no RecoverSSMPostprocessMetadata: the generic kernel would
        # advance the running column with (computed + sampled) // block, which is
        # not the column this commit wrote. The GDN-specific bookkeeping (running
        # column = commit target, persistent accepted = 1) is staged here and
        # applied by the postprocess hook after this commit and before the native
        # align postcopy.
        return None


def _native_spec_rows(builder, n: int, device):
    """Borrow the exact map the native builder has already uploaded this step."""
    rows = getattr(builder, "spec_sequence_indices", None)
    host_rows = getattr(builder, "spec_sequence_indices_cpu", None)
    if rows is None or host_rows is None or rows.numel() < n or host_rows.numel() < n:
        raise RuntimeError(
            "Ascend GDN RecoverSSM requires the native speculative row buffers; "
            "refusing to guess a mapping or add a synchronizing device nonzero"
        )
    if rows.device != device:
        raise RuntimeError("native speculative row mapping is on the wrong device")
    return rows[:n], tuple(int(x) for x in host_rows[:n].tolist())


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
        self._accepted_buffer = torch.ones(max_rows, dtype=torch.int32, device=device)

    def _record_key_heads(self, layer) -> int:
        config = self.vllm_config.model_config.hf_text_config
        heads = getattr(config, "linear_num_key_heads", None)
        if heads is None:
            attn = getattr(config, "linear_attn_config", None)
            if isinstance(attn, dict):
                heads = attn.get("num_key_heads")
        if heads is None:
            # No grouping is safer than inventing a model's head mapping.
            return int(layer.kv_cache[1].shape[1])
        tp = int(self.vllm_config.parallel_config.tensor_parallel_size)
        if int(heads) % tp:
            raise ValueError("GDN key heads must be divisible by tensor parallel size")
        return int(heads) // tp

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
                        id(t),
                        t.data_ptr(),
                        tuple(t.shape),
                        tuple(t.stride()),
                        t.dtype,
                        t.device,
                    )
                    for t in layer.kv_cache
                ),
            )
            for layer_name, layer in zip(self.layer_names, layers)
        )
        if self._context is not None and key == self._ctx_key:
            return self._context
        if self._context is not None:
            if (
                getattr(self, "_gdn_serving_ready", False)
                and getattr(self, "use_full_cuda_graph", False)
                and getattr(self, "_captured_context", None) is not None
                and not self._capturing
            ):
                raise RuntimeError(
                    "GDN KV binding changed after graph capture; the native "
                    "model graph must be invalidated and recaptured before serving"
                )
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
                max_rows,
                int(self.vllm_config.scheduler_config.max_num_seqs or max_rows),
            ),
            num_key_heads=self._record_key_heads(layers[0]),
            key_dtype=getattr(self.vllm_config.model_config, "dtype", torch.float32),
        )
        self._ctx_key = key
        self._prepared_group_bucket = None
        # Keep the binding objects alive; neither container identity nor a raw
        # pointer alone is a reliable storage/layout invalidation protocol.
        self._ctx_bindings = tuple(tuple(layer.kv_cache) for layer in layers)
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

        context = self._get_context()
        if context is None:
            raise RuntimeError(
                "speculative verify requires a bound GDN recovery context"
            )

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

        request_indices, request_rows = _native_spec_rows(self, n, state_indices.device)
        # All groups see the same scheduler batch, but own distinct physical
        # tables. Keep a CPU geometry identity so the coordinator can validate
        # row agreement without reading any device value.
        host_qsl = m.query_start_loc_cpu.tolist()
        query_lens = tuple(
            int(host_qsl[row + 1]) - int(host_qsl[row]) for row in request_rows
        )

        commit = AscendGDNRecoverSSMCommitMetadata(
            state_indices=state_indices,
            query_start_loc=query_start_loc,
            request_indices=request_indices,
            block_table=block_table,
            num_computed_tokens=num_computed,
            block_size=block_size,
            request_rows=request_rows,
            query_lens=query_lens,
        )

        base = {f.name: getattr(meta, f.name) for f in fields(meta)}
        base["num_accepted_tokens"] = num_accepted
        out = AscendGDNRecoverSSMMetadata(
            **base,
            recoverssm_commit=commit,
            recoverssm_builder=self,
            recoverssm_context=context,
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

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        self._capturing = True
        try:
            result = super().build_for_cudagraph_capture(common_attn_metadata)
            self._captured_context = self._context
            return result
        finally:
            self._capturing = False


class AscendGDNRecoverSSMBackend(AscendGDNAttentionBackend):
    """GDN backend that builds RecoverSSM-aware metadata."""

    @staticmethod
    def get_name() -> str:
        return "ASCEND_GDN_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[AscendGDNRecoverSSMMetadataBuilder]:
        # Preserve the native backend *factory* decision. Calling super().build
        # does not restore the HostMetadataBuilder class attribute.
        from vllm_ascend.ops import gdn_attn_builder as native

        factory = getattr(native, "AscendGDNFusedAttentionBackend", None)
        if factory is not None:
            selected = factory.get_builder_cls()
            if not selected.build_device_chunk_metadata:
                return AscendGDNRecoverSSMHostMetadataBuilder
        return AscendGDNRecoverSSMMetadataBuilder


class AscendGDNRecoverSSMHostMetadataBuilder(AscendGDNRecoverSSMMetadataBuilder):
    """RecoverSSM plus native fused-prefill host-only descriptors."""

    build_device_chunk_metadata = False


__all__ = [
    "AscendGDNRecoverSSMBackend",
    "AscendGDNRecoverSSMCommitMetadata",
    "AscendGDNRecoverSSMMetadata",
    "AscendGDNRecoverSSMMetadataBuilder",
    "AscendGDNRecoverSSMHostMetadataBuilder",
]
