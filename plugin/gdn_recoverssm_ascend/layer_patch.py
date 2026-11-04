"""Wrap the Ascend GDN layer without touching any source file.

Patched attributes
------------------
``AscendGatedDeltaNetAttention.__init__``
    Record ``self.use_gdn_recoverssm`` (from ``cache_config.use_kda_recoverssm``).
``get_attn_backend``
    Return the RecoverSSM backend (which builds the commit metadata).
``_forward_core``
    Push the current layer onto a thread-local stack so the
    ``recurrent_gated_delta_rule`` shim knows whether to redirect.
``vllm_ascend.ops.gdn.recurrent_gated_delta_rule``
    Redirect **only** the speculative-decode call (the one that passes
    ``num_accepted_tokens``) to :func:`kernels.gdn_recoverssm_verify`.
``AscendMambaHybridModelState.prepare_attn``
    Ascend overrides ``prepare_attn`` without calling ``super()``, so the
    upstream hook that feeds ``RecoverSSMState.record_step`` never runs. Without
    this wrapper the commit would see an empty step list and the checkpoint
    would silently never advance.

The state shape/dtype are **not** patched here: the Qwen model classes call
``MambaStateShapeCalculator`` / ``MambaStateDtypeCalculator`` directly, so those
are wrapped in ``state_patch.py`` instead (one place, covers every caller).

Everything else -- the convolution update, the projections, sampling, the
post-sampling commit lifecycle -- stays exactly as shipped.
"""

from __future__ import annotations

import functools
import os
import threading

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.gdn_recoverssm_ascend." + __name__.rsplit(".", 1)[-1])

_APPLIED = False

# The layer whose ``_forward_core`` is currently executing. The shim needs it to
# reach ``layer.kv_cache[2]`` (the replay record) and ``layer.num_spec``.
_local = threading.local()


def _current_layer():
    stack = getattr(_local, "stack", None)
    return stack[-1] if stack else None


def _push_layer(layer):
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    stack.append(layer)
    return len(stack)


def _pop_layer(depth):
    stack = getattr(_local, "stack", None)
    if stack is not None:
        while len(stack) >= depth:
            stack.pop()


_GLOBAL_ENABLED = False


def enable_runtime() -> None:
    """Feature switch driven by the plugin, not by each layer's ``__init__``.

    The worker replaces the real layer's ``__init__`` with
    ``_gdn_init_with_packed_weight`` (which calls the *original* vLLM init), so an
    ``__init__``-based flag is never set on real layers. Relying on it made the
    whole feature silently inert.
    """
    global _GLOBAL_ENABLED
    _GLOBAL_ENABLED = True


def _enabled(layer) -> bool:
    if not _GLOBAL_ENABLED or _env_flag("GDN_RECOVERSSM_DISABLE"):
        return False
    cache_config = getattr(layer, "cache_config", None)
    if cache_config is not None and not getattr(
        cache_config, "use_kda_recoverssm", False
    ):
        # ``--use-replayssm`` is what shrinks the mamba page to one state slot and
        # appends the replay record; without it there is nowhere to write.
        return False
    # An explicit per-instance opt-out still wins.
    return getattr(layer, "use_gdn_recoverssm", True) is not False


class _RecoverCommitProxy:
    """Wraps RecoverSSMState so the GDN align bookkeeping lands at the right time."""

    __slots__ = ("_inner",)

    def __init__(self, inner):
        self._inner = inner

    def record_step(self, *args, **kwargs):
        return self._inner.record_step(*args, **kwargs)

    def commit_step(self, *args, **kwargs):
        result = self._inner.commit_step(*args, **kwargs)
        try:
            _apply_align_bookkeeping(kwargs.get("state_indices"), kwargs.get("num_accepted_tokens"), kwargs.get("idx_mapping") or (args[1] if len(args) > 1 else None))
        except Exception:
            pass
        return result


def _apply_align_bookkeeping(state_indices, num_accepted_tokens, idx_mapping):
    if state_indices is None or num_accepted_tokens is None or idx_mapping is None:
        return
    from . import builder as builder_mod

    pending = builder_mod.take_pending_align()
    if not pending:
        return
    rows = pending["rows"].long()
    slots = idx_mapping[rows].long()
    keep = slots >= 0
    if bool(keep.all()) is False:
        slots = slots[keep]
        cols = pending["cols"][keep]
    else:
        cols = pending["cols"]
    # One request slot per committed batch row: point it at the column the commit
    # actually wrote and reset the accepted count to the neutral value.
    state_indices[slots] = cols.to(state_indices.dtype)
    num_accepted_tokens[slots] = 1


def install() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from vllm_ascend.ops import gdn as gdn_mod

    # ``AscendGatedDeltaNetAttention`` is only a method donor: the worker
    # (vllm_ascend/patch/worker/patch_qwen3_5.py) copies its methods onto the
    # **vLLM** class and the real layers are instances of that one:
    #
    #   _GDN_PATCH_TARGET = QwenGatedDeltaNetAttention
    #   _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention._forward_core
    #   _GDN_PATCH_TARGET.__init__ = _gdn_init_with_packed_weight   # calls the ORIGINAL init
    #
    # Patching only the Ascend subclass therefore patched dead code: the real
    # layers never got ``use_gdn_recoverssm`` and never entered verify.
    donor_cls = gdn_mod.AscendGatedDeltaNetAttention
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as real_cls,
    )

    # Patching both classes makes the result independent of import order: if the
    # worker patch runs first we override it, if it runs later it copies our
    # wrappers from the donor class.
    for cls in (donor_cls, real_cls):
        # ------------------------------------------------------------ backend
        original_backend = cls.get_attn_backend

        @functools.wraps(original_backend)
        def get_attn_backend(self, _original=original_backend):
            if _enabled(self):
                from .builder import AscendGDNRecoverSSMBackend

                return AscendGDNRecoverSSMBackend
            return _original(self)

        cls.get_attn_backend = get_attn_backend

        # -------------------------------------------------------- forward hook
        original_core = cls._forward_core

        @functools.wraps(original_core)
        def _forward_core(self, *args, _original=original_core, **kwargs):
            depth = _push_layer(self)
            try:
                return _original(self, *args, **kwargs)
            finally:
                _pop_layer(depth)

        cls._forward_core = _forward_core

    real_cls.__name__ = real_cls.__name__  # keep linters quiet about the alias

    # ------------------------------------------------- recurrent call redirect
    original_recurrent = gdn_mod.recurrent_gated_delta_rule

    def recurrent_gated_delta_rule(*args, **kwargs):
        layer = _current_layer()
        # Only the speculative path passes ``num_accepted_tokens``; the plain
        # decode path must keep the stock kernel so single-token steps write
        # their state normally.
        if (
            layer is not None
            and _enabled(layer)
            and kwargs.get("num_accepted_tokens") is not None
            and len(args) >= 4
        ):
            return _verify(layer, args[0], args[1], args[2], args[3], kwargs)
        return original_recurrent(*args, **kwargs)

    gdn_mod.recurrent_gated_delta_rule = recurrent_gated_delta_rule

    # ---------------------------------------------------- model-state record hook
    # ``RecoverSSMState.record_step`` is called from the base
    # ``MambaHybridModelState.prepare_attn``. Ascend replaces that method without
    # calling super(), so restore the hook here.
    from vllm_ascend.worker.v2.model_states import mamba_hybrid as mamba_state_mod

    state_cls = mamba_state_mod.AscendMambaHybridModelState
    original_prepare_attn = state_cls.prepare_attn

    @functools.wraps(original_prepare_attn)
    def prepare_attn(self, *args, **kwargs):
        attn_metadata = original_prepare_attn(self, *args, **kwargs)
        recoverssm = getattr(self, "recoverssm", None)
        if recoverssm is None:
            return attn_metadata
        attn_groups = kwargs.get("attn_groups")
        if attn_groups is None:
            # (input_batch, cudagraph_mode, block_tables, slot_mappings,
            #  attn_groups, kv_cache_config, for_capture=False, ubatch_idx=0)
            attn_groups = args[4] if len(args) > 4 else []
        for_capture = bool(kwargs.get("for_capture", False))
        recoverssm.record_step(attn_metadata, attn_groups, for_capture=for_capture)
        return attn_metadata

    state_cls.prepare_attn = prepare_attn

    # ------------------------------------------------- align bookkeeping hook
    # ``postprocess_state`` runs, in order: scatter accepted -> RecoverSSM
    # commit -> native align postcopy. The running column and the persistent
    # accepted count must be corrected after the commit but before the postcopy,
    # and the only seam available from Python is inside ``commit_step`` itself,
    # so ``self.recoverssm`` is swapped for a proxy during the call.
    original_postprocess = state_cls.postprocess_state

    @functools.wraps(original_postprocess)
    def postprocess_state(self, *args, **kwargs):
        rec = getattr(self, "recoverssm", None)
        if rec is None:
            return original_postprocess(self, *args, **kwargs)
        proxy = _RecoverCommitProxy(rec)
        self.recoverssm = proxy
        try:
            return original_postprocess(self, *args, **kwargs)
        finally:
            self.recoverssm = rec

    state_cls.postprocess_state = postprocess_state

    enable_runtime()
    _APPLIED = True
    logger.debug(
        "gdn_recoverssm_ascend: AscendGatedDeltaNetAttention + model-state "
        "record hook patched"
    )


_TRACE = {"steps": 0}
# Host-only counters: how much work our path contributes to the stream.
_COUNTERS = {"verify_calls": 0, "last_rows": 0, "last_tokens": 0}


def counters() -> dict:
    """Snapshot of the host-side counters (no device access)."""
    return dict(_COUNTERS)


def _is_capturing() -> bool:
    """Whether the current NPU stream is capturing a graph."""
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "is_current_stream_capturing"):
        return False
    try:
        return bool(npu.is_current_stream_capturing())
    except Exception:
        return False


def _env_flag(name: str) -> bool:
    """Read a diagnostic switch every call: bisecting needs no restart of code."""
    value = os.getenv(name)
    return value is not None and value.strip() not in ("", "0", "false", "False")


def _trace_step(layer, attn_metadata, rows, total, query_start_loc, checkpoint_indices):
    """One line per speculative step for the first few steps of the process.

    This is the "is RecoverSSM live on this request?" signal. Every field below is
    a Python integer already sitting in the metadata, so the line never touches
    device data and can never force a host synchronisation.

    Why that matters: vLLM runs one **eager warmup forward per capture size**
    before it starts recording (``create_forward_fn(desc, warmup=True)`` then
    ``with torch.cuda.graph(...)``), and the Ascend wrapper documents that
    ``capturing`` is False during that warmup. So these first steps can legitimately
    print while the "Capturing ... 0%" bar is on screen; ``is_current_stream_capturing``
    still returns the truth for the recording phase itself. Keeping the trace free of
    device reads means it is safe in either phase without needing to tell them apart.
    """
    if _is_capturing():
        return
    limit = int(os.getenv("GDN_RECOVERSSM_TRACE_STEPS", "3") or "0")
    if limit <= 0 or _TRACE["steps"] >= limit:
        return
    _TRACE["steps"] += 1
    num_prefills = int(getattr(attn_metadata, "num_prefills", 0))
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
    logger.info(
        "Ascend GDN RecoverSSM step %d: prefill=%d decode=%d spec_rows=%d "
        "window=%d tokens=%d cache_mode=%s",
        _TRACE["steps"],
        num_prefills,
        num_decodes,
        rows,
        layer.num_spec + 1,
        total,
        getattr(getattr(layer, "cache_config", None), "mamba_cache_mode", "?"),
    )


def _verify(layer, q, k, v, ssm_state, kwargs):
    """Run the RecoverSSM verify for one speculative window.

    ``q``/``k`` are ``[T, H, K]`` and already L2-normalised by the caller
    (``l2norm_fwd``); ``v`` is ``[T, HV, V]``; ``g``/``beta`` are ``[T, HV]``.

    The window geometry is taken from the layer's attention metadata (the same
    object the stock path reads), not from ``actual_seq_lengths``: on Ascend that
    argument carries **per-sequence lengths**, not cumulative offsets.
    """
    from vllm.forward_context import get_forward_context

    from .kernels import gdn_recoverssm_verify

    attn_metadata = get_forward_context().attn_metadata[layer.prefix]
    if attn_metadata.spec_query_start_loc is None:
        raise RuntimeError(
            "Ascend GDN RecoverSSM: spec_query_start_loc is required"
        )

    # Authoritative spec-row count: the length of ``spec_query_start_loc``.
    # ``num_spec_decodes`` can count graph-padded rows, and the stock Ascend code
    # derives the same number the same way
    # (``spec_num_rows = spec_query_start_loc.size(0) - 1``). Using the padded
    # count here reads past both the offset tensor and q/k/v, which shows up as
    # "MTE accesses an invalid GM address" on a later synchronize.
    query_start_loc = attn_metadata.spec_query_start_loc
    # Two counts exist and neither is right on its own:
    #   * eager: len(qsl) - 1 == num_spec_decodes;
    #   * full graph: qsl is materialised to graph_request_count + 1 with
    #     zero-length dummy segments at the back, so len(qsl) - 1 > the logical
    #     count.
    # The logical count is the authoritative one whenever it is smaller.
    qsl_rows = int(query_start_loc.shape[0]) - 1
    rows = min(int(getattr(attn_metadata, "num_spec_decodes", qsl_rows)), qsl_rows)
    if rows <= 0:
        raise RuntimeError(
            "Ascend GDN RecoverSSM: speculative call without spec-decode rows"
        )

    total = int(q.shape[0])
    # Pass the metadata's own buffer (a view starting at offset 0) and let the
    # kernel mask ``tok < total_tokens``. A clamped copy cost one extra device
    # task per layer inside the captured graph and, more importantly, pinned the
    # input to a buffer created at capture time. (Captured operators re-execute at
    # replay, so a fixed address does not mean a fixed value -- the point here is
    # which buffer the task reads, not that the value would be frozen.)
    query_start_loc = query_start_loc[: rows + 1]
    # Column 0 is the checkpoint; the per-draft-token slots are unused under
    # RecoverSSM (num_speculative_blocks == 0).
    checkpoint_indices = attn_metadata.spec_state_indices_tensor[:rows, 0]

    _trace_step(layer, attn_metadata, rows, total, query_start_loc, checkpoint_indices)

    kv_cache = layer.kv_cache
    if len(kv_cache) < 3:
        raise RuntimeError(
            "Ascend GDN RecoverSSM: expected kv_cache = (conv, checkpoint, "
            f"replay), got {len(kv_cache)} entries. Is --use-replayssm set and "
            "GDN_RECOVERSSM_DISABLE unset?"
        )

    _COUNTERS["verify_calls"] += 1
    _COUNTERS["last_rows"] = rows
    _COUNTERS["last_tokens"] = total

    out = gdn_recoverssm_verify(
        kwargs["g"],
        kwargs["beta"],
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        checkpoint_state=kv_cache[1],
        replay_cache=kv_cache[2],
        query_start_loc=query_start_loc,
        state_indices=checkpoint_indices,
        spec_query_len=layer.num_spec + 1,
        scale=kwargs.get("scale"),
        qk_already_normalized=True,
    )
    # The stock kernel returns [T, HV, V] and the layer adds the batch dim.
    return out.squeeze(0)


