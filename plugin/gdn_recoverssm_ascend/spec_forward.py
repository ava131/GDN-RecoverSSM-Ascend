# SPDX-License-Identifier: Apache-2.0
"""Pure-speculative core path using the pinned Ascend 0.30rc1 helper ABI.

The native core allocates a recurrence output and copies it into the custom
op's output.  For a pure speculative batch RecoverSSM can write that final
buffer directly.  Mixed batches retain the native split/merge path.  In
particular this adapter does not change gate activation, normalization,
convolution, capture geometry, or the mathematical recurrence.
"""

from __future__ import annotations


def try_spec_forward(layer, mixed_qkv, b, a, core_attn_out, **kwargs) -> bool:
    """Fill the core output and return True, or decline before any side effect.

    This matches ``AscendGatedDeltaNetAttention._forward_core`` at the supported
    0.30rc1 source.  Never catch an exception and fall back after the convolution
    has begun: it has already updated the speculative convolution history.
    """
    if kwargs:
        # A changed native ABI needs an explicit adapter, not discarded options.
        return False
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ops import gdn as native

    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        # Profiling without attention metadata must preserve the native no-op.
        return False
    meta = metadata.get(layer.prefix)
    if meta is None:
        return False
    if (
        getattr(meta, "num_prefills", 0) != 0
        or getattr(meta, "num_decodes", 0) != 0
        or int(getattr(meta, "num_spec_decodes", 0)) <= 0
        or getattr(meta, "spec_sequence_masks", None) is None
    ):
        return False
    descriptor = getattr(meta, "spec_decode_metadata", None)
    conv = getattr(descriptor, "spec_causal_conv1d", None)
    if conv is None:
        return False
    required = (
        "_get_packed_conv_weights",
        "_normalize_causal_cache_indices",
        "causal_conv1d_update",
        "wait_for_kv_layer_from_connector",
        "record_attention_compute_start",
        "maybe_save_kv_layer_to_connector",
        "l2norm_fwd",
        "DeviceOperator",
    )
    if any(not hasattr(native, name) for name in required):
        return False
    if any(
        getattr(conv, name, None) is None
        for name in ("query_start_loc", "cache_indices", "num_accepted_tokens")
    ):
        return False
    actual = int(meta.num_actual_tokens)
    if actual <= 0:
        return False
    if any(actual > tensor.shape[0] for tensor in (mixed_qkv, b, a, core_attn_out)):
        raise ValueError("RecoverSSM pure-spec token count exceeds an input buffer")

    import torch

    from .layer_patch import _verify

    # Preserve native layerwise transfer fences inside the custom op body.
    native.wait_for_kv_layer_from_connector(layer.prefix)
    native.record_attention_compute_start()
    inputs = mixed_qkv[:actual]
    conv_output = torch.empty_like(inputs)
    conv_output = native.causal_conv1d_update(
        inputs,
        layer.kv_cache[0],
        native._get_packed_conv_weights(layer),
        bias=layer.conv1d.bias,
        activation="silu" if layer.activation else None,
        conv_state_indices=native._normalize_causal_cache_indices(conv.cache_indices),
        num_accepted_tokens=conv.num_accepted_tokens,
        query_start_loc=conv.query_start_loc,
        max_query_len=layer.num_spec + 1,
        null_block_id=0,
        out=conv_output,
    )
    q, k, v = layer.rearrange_mixed_qkv(conv_output)
    g, beta = native.DeviceOperator.fused_gdn_gating(
        layer.A_log, a[:actual], b[:actual], layer.dt_bias
    )
    # Identical normalization, storage rounding and scale to native rc1.
    q = native.l2norm_fwd(q)
    k = native.l2norm_fwd(k)
    destination = core_attn_out[:actual].unsqueeze(0)
    _verify(
        layer,
        q.squeeze(0),
        k.squeeze(0),
        v.squeeze(0),
        layer.kv_cache[1],
        {"g": g.squeeze(0), "beta": beta.squeeze(0), "scale": k.shape[-1] ** -0.5},
        out=destination,
    )
    native.maybe_save_kv_layer_to_connector("", [])
    return True
