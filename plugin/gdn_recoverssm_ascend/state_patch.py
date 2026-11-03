# SPDX-License-Identifier: Apache-2.0
"""Append the ReplaySSM record to the GDN mamba state shape/dtype.

Why not patch the layer? ``Qwen3_5ForConditionalGeneration`` exposes the state
layout through **classmethods on the model** that call the calculators directly::

    get_mamba_state_shape_from_config()  -> MambaStateShapeCalculator.gated_delta_net_state_shape(...)
    get_mamba_state_dtype_from_config()  -> MambaStateDtypeCalculator.gated_delta_net_state_dtype(...)

so a layer-level override would be bypassed and the shape/dtype tuples would end
up with different lengths. Wrapping the two calculators is the single place that
covers every caller (model classmethod, layer method, and anything else).

The wrap is gated by a process-local flag that ``config_patch`` sets only after
it has accepted a Qwen GDN RecoverSSM configuration.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

_FLAG = {"enabled": False, "spec_query_len": 0, "head_k": 0, "head_v": 0}

_APPLIED = False


def enable(spec_query_len: int) -> None:
    _FLAG["enabled"] = True
    _FLAG["spec_query_len"] = int(spec_query_len)


def disable() -> None:
    _FLAG["enabled"] = False


def is_enabled() -> bool:
    return bool(_FLAG["enabled"])


def _replay_shape(tp_world_size: int, num_v_heads: int, head_k_dim: int, head_v_dim: int, spec_query_len: int):
    """One FP32 record per token: [corr(V), key_hat(K), decay(1)]."""
    local_value_heads = num_v_heads // tp_world_size
    return (
        local_value_heads,
        spec_query_len,
        head_v_dim + head_k_dim + 1,
    )


def install() -> None:
    global _APPLIED
    if _APPLIED:
        return

    import torch
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )

    # ---------------------------------------------------------------- shape
    orig_shape = MambaStateShapeCalculator.gated_delta_net_state_shape.__func__

    @functools.wraps(orig_shape)
    def gated_delta_net_state_shape(
        cls,
        tp_world_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size,
        num_spec=0,
    ):
        shapes = orig_shape(
            cls,
            tp_world_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            num_spec,
        )
        if not _FLAG["enabled"]:
            return shapes
        spec_query_len = _FLAG["spec_query_len"] or (num_spec + 1)
        return (
            *shapes,
            _replay_shape(
                tp_world_size, num_v_heads, head_k_dim, head_v_dim, spec_query_len
            ),
        )

    MambaStateShapeCalculator.gated_delta_net_state_shape = classmethod(
        gated_delta_net_state_shape
    )

    # ---------------------------------------------------------------- dtype
    orig_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype.__func__

    @functools.wraps(orig_dtype)
    def gated_delta_net_state_dtype(cls, *args, **kwargs):
        dtypes = orig_dtype(cls, *args, **kwargs)
        if not _FLAG["enabled"]:
            return dtypes
        return (*dtypes, torch.float32)

    MambaStateDtypeCalculator.gated_delta_net_state_dtype = classmethod(
        gated_delta_net_state_dtype
    )

    _APPLIED = True
    logger.debug(
        "gdn_recoverssm_ascend: GDN state calculators patched "
        "(replay record appended when enabled)"
    )


__all__ = ["enable", "disable", "install", "is_enabled"]
