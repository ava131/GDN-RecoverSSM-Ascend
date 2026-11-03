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
import logging
import threading

import torch

logger = logging.getLogger(__name__)

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


def _enabled(layer) -> bool:
    return bool(getattr(layer, "use_gdn_recoverssm", False))


def install() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from vllm_ascend.ops import gdn as gdn_mod

    cls = gdn_mod.AscendGatedDeltaNetAttention

    # ---------------------------------------------------------------- __init__
    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        cache_config = getattr(self, "cache_config", None)
        self.use_gdn_recoverssm = bool(
            getattr(cache_config, "use_kda_recoverssm", False)
        ) and getattr(self, "num_spec", 0) > 0

    cls.__init__ = __init__

    # ------------------------------------------------------------- backend
    original_backend = cls.get_attn_backend

    @functools.wraps(original_backend)
    def get_attn_backend(self):
        if _enabled(self):
            from .builder import AscendGDNRecoverSSMBackend

            return AscendGDNRecoverSSMBackend
        return original_backend(self)

    cls.get_attn_backend = get_attn_backend

    # ------------------------------------------------------------- forward hook
    original_core = cls._forward_core

    @functools.wraps(original_core)
    def _forward_core(self, *args, **kwargs):
        depth = _push_layer(self)
        try:
            return original_core(self, *args, **kwargs)
        finally:
            _pop_layer(depth)

    cls._forward_core = _forward_core

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

    _APPLIED = True
    logger.debug(
        "gdn_recoverssm_ascend: AscendGatedDeltaNetAttention + model-state "
        "record hook patched"
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
    rows = int(query_start_loc.shape[0]) - 1
    if rows <= 0:
        raise RuntimeError(
            "Ascend GDN RecoverSSM: speculative call without spec-decode rows"
        )

    total = int(q.shape[0])
    # Never let a padded/stale offset describe tokens past the activation tensors.
    query_start_loc = query_start_loc[: rows + 1].clamp(min=0, max=total)
    # Column 0 is the checkpoint; the per-draft-token slots are unused under
    # RecoverSSM (num_speculative_blocks == 0).
    checkpoint_indices = attn_metadata.spec_state_indices_tensor[:rows, 0]

    kv_cache = layer.kv_cache
    if len(kv_cache) < 3:
        raise RuntimeError(
            "Ascend GDN RecoverSSM: expected kv_cache = (conv, checkpoint, "
            f"replay), got {len(kv_cache)} entries. Is --use-replayssm set and "
            "GDN_RECOVERSSM_DISABLE unset?"
        )

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


