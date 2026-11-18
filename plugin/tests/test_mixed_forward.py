# SPDX-License-Identifier: Apache-2.0
# Native core fixture: Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Execute the patched native rc1 core with real CPU tensors and owned KV.

The frozen source fixture below is the actual v0.30.0rc1 method, not a second
implementation of the patch. Native device helpers are replaced with independent
CPU arithmetic. This proves dispatch, precision boundaries, ordering and output
mapping; it does not establish NPU compilation or hardware timing.
"""

from __future__ import annotations

import importlib.util
import linecache
import logging
import sys
import types
from itertools import accumulate, pairwise
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend"

# https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/vllm_ascend/ops/gdn.py
_NATIVE_CORE = r'''def _forward_core(
    self,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
):
    """
    Core attention computation (called by custom op).
    """
    forward_context = get_forward_context()
    attn_metadata: AttentionMetadata = forward_context.attn_metadata

    if attn_metadata is None:
        # V1 profile run
        return

    # Layerwise KV pool hooks must stay inside the custom op body: the
    # forward() caller region is traced by Dynamo in fullgraph mode, and
    # these side effects (thread locks, connector waits) would break the
    # graph. Waiting here still orders the deferred mamba state copy and
    # the layer load before conv/attention kernels touch mamba state.
    wait_for_kv_layer_from_connector(self.prefix)
    record_attention_compute_start()

    assert isinstance(attn_metadata, dict)
    attn_metadata = attn_metadata[self.prefix]
    assert isinstance(attn_metadata, GDNAttentionMetadata)
    spec_sequence_masks = attn_metadata.spec_sequence_masks
    spec_token_indx = attn_metadata.spec_token_indx
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
    self_kv_cache = self.kv_cache
    ssm_state = self_kv_cache[1]
    num_actual_tokens = attn_metadata.num_actual_tokens

    mixed_qkv = mixed_qkv[:num_actual_tokens]
    b = b[:num_actual_tokens]
    a = a[:num_actual_tokens]

    # 1. Convolution sequence transformation
    conv_weights_T = _get_packed_conv_weights(self)
    if spec_sequence_masks is not None:
        if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            mixed_qkv_spec = mixed_qkv
            mixed_qkv_non_spec = None
        else:
            mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
            mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
    else:
        mixed_qkv_spec = None
        mixed_qkv_non_spec = mixed_qkv

    # 1.1: Process the multi-query part
    if spec_sequence_masks is not None:
        spec_causal_conv1d_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
        spec_query_start_loc_device = spec_causal_conv1d_meta.query_start_loc
        output_spec = torch.empty_like(mixed_qkv_spec)
        output_spec = causal_conv1d_update(
            mixed_qkv_spec,
            self_kv_cache[0],
            conv_weights_T,
            bias=self.conv1d.bias,
            activation="silu" if self.activation else None,
            conv_state_indices=_normalize_causal_cache_indices(spec_causal_conv1d_meta.cache_indices),
            num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens,
            query_start_loc=spec_query_start_loc_device,
            max_query_len=self.num_spec + 1,
            null_block_id=0,
            out=output_spec,
        )
        mixed_qkv_spec = output_spec

    # 1.2: Process the remaining part
    if attn_metadata.num_prefills > 0:
        if mixed_qkv_non_spec is not None:
            non_spec_causal_conv1d_meta = attn_metadata.non_spec_prefill_metadata.causal_conv1d
            query_start_loc_opt = non_spec_causal_conv1d_meta.query_start_loc
            cache_indices_opt = non_spec_causal_conv1d_meta.cache_indices
            initial_state_mode_opt = non_spec_causal_conv1d_meta.initial_state_mode
            if get_pcp_group().world_size > 1:
                non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
                assert non_spec_query_start_loc is not None
                non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
                width = self.conv_kernel_size
                state_len = width - 1
                num_seqs = non_spec_query_start_loc.shape[0] - 1
                prefill_seq_offset = max(0, num_seqs - attn_metadata.num_prefills)
                prefill_cache_indices = non_spec_state_indices_tensor[prefill_seq_offset:]
                mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                last_width_prefill_x = extract_last_width(
                    mixed_qkv_non_spec_T, non_spec_query_start_loc[prefill_seq_offset:], state_len
                )
                pcp_rank = get_pcp_group().rank_in_group
                all_last_width_prefill_x = get_pcp_group().all_gather(
                    last_width_prefill_x.unsqueeze(0).contiguous(), 0
                )
                if pcp_rank > 0 and prefill_cache_indices.shape[0] > 0:
                    self_kv_cache[0][prefill_cache_indices, :state_len, :] = all_last_width_prefill_x[
                        pcp_rank - 1, ...
                    ].transpose(-1, -2)
                mixed_qkv_non_spec_output = causal_conv1d_fn(
                    mixed_qkv_non_spec,
                    conv_weights_T,
                    self.conv1d.bias,
                    conv_states=self_kv_cache[0],
                    query_start_loc=query_start_loc_opt,
                    cache_indices=_normalize_causal_cache_indices(cache_indices_opt),
                    has_initial_state=initial_state_mode_opt,
                    activation="silu" if self.activation else None,
                    pad_slot_id=PAD_SLOT_ID,
                    null_block_id=0,
                )
                mixed_qkv_non_spec = mixed_qkv_non_spec_output
                if prefill_cache_indices.shape[0] > 0:
                    self_kv_cache[0][prefill_cache_indices, :state_len, :] = all_last_width_prefill_x[
                        -1, ...
                    ].transpose(-1, -2)
            else:
                mixed_qkv_non_spec_output = causal_conv1d_fn(
                    mixed_qkv_non_spec,
                    conv_weights_T,
                    self.conv1d.bias,
                    conv_states=self_kv_cache[0],
                    query_start_loc=query_start_loc_opt,
                    cache_indices=_normalize_causal_cache_indices(cache_indices_opt),
                    has_initial_state=initial_state_mode_opt,
                    activation="silu" if self.activation else None,
                    pad_slot_id=PAD_SLOT_ID,
                    null_block_id=0,
                )
                mixed_qkv_non_spec = mixed_qkv_non_spec_output
    elif attn_metadata.num_decodes > 0:
        non_spec_causal_conv1d_meta = attn_metadata.non_spec_decode_metadata.causal_conv1d
        non_spec_query_start_loc_device = non_spec_causal_conv1d_meta.query_start_loc
        output_non_spec = torch.empty_like(mixed_qkv_non_spec)
        output_non_spec = causal_conv1d_update(
            mixed_qkv_non_spec,
            self_kv_cache[0],
            conv_weights_T,
            bias=self.conv1d.bias,
            activation="silu" if self.activation else None,
            conv_state_indices=_normalize_causal_cache_indices(non_spec_causal_conv1d_meta.cache_indices),
            query_start_loc=non_spec_query_start_loc_device,
            max_query_len=1,
            null_block_id=0,
            out=output_non_spec,
        )
        mixed_qkv_non_spec = output_non_spec
    else:
        mixed_qkv_non_spec = None

    query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
    query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

    # 2. Recurrent attention
    g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
    if spec_sequence_masks is not None:
        if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            g_spec = g
            beta_spec = beta
            g_non_spec = None
            beta_non_spec = None
        else:
            g_spec = g.index_select(1, spec_token_indx)
            beta_spec = beta.index_select(1, spec_token_indx)
            g_non_spec = g.index_select(1, non_spec_token_indx)
            beta_non_spec = beta.index_select(1, non_spec_token_indx)
    else:
        g_spec = None
        beta_spec = None
        g_non_spec = g
        beta_non_spec = beta

    split_non_spec = (
        spec_sequence_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
    )
    num_decode_tokens = attn_metadata.num_decode_tokens

    # 2.1: Process the multi-query part
    if spec_sequence_masks is not None:
        actual_seq_lengths = attn_metadata.spec_decode_metadata.actual_seq_lengths
        query_spec = l2norm_fwd(query_spec)
        key_spec = l2norm_fwd(key_spec)
        core_attn_out_spec = recurrent_gated_delta_rule(
            query_spec.squeeze(0),
            key_spec.squeeze(0),
            value_spec.squeeze(0),
            ssm_state,
            g=g_spec.squeeze(0),
            beta=beta_spec.squeeze(0),
            scale=key_spec.shape[-1] ** -0.5,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=spec_state_indices_tensor.flatten(),
            num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens.to(torch.int32),
        ).unsqueeze(0)
    else:
        core_attn_out_spec, last_recurrent_state = None, None

    # 2.2: Process non-spec-decode part in mixed non-spec batches
    if split_non_spec:
        assert g_non_spec is not None
        assert beta_non_spec is not None
        query_decode = query_non_spec[:, :num_decode_tokens]
        key_decode = key_non_spec[:, :num_decode_tokens]
        value_decode = value_non_spec[:, :num_decode_tokens]
        actual_seq_lengths = attn_metadata.non_spec_decode_metadata.actual_seq_lengths
        query_decode = l2norm_fwd(query_decode)
        key_decode = l2norm_fwd(key_decode)
        core_attn_out_decode = recurrent_gated_delta_rule(
            query_decode.squeeze(0),
            key_decode.squeeze(0),
            value_decode.squeeze(0),
            ssm_state,
            g=g_non_spec[:, :num_decode_tokens].squeeze(0),
            beta=beta_non_spec[:, :num_decode_tokens].squeeze(0),
            scale=key_decode.shape[-1] ** -0.5,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_decodes],
        ).unsqueeze(0)
    else:
        core_attn_out_decode = None

    # 2.3: Process the remaining part
    if attn_metadata.num_prefills > 0:
        prefill_query_start_loc = attn_metadata.prefill_query_start_loc
        prefill_state_indices = attn_metadata.prefill_state_indices
        prefill_has_initial_state = attn_metadata.prefill_has_initial_state
        assert prefill_query_start_loc is not None
        assert prefill_state_indices is not None
        assert prefill_has_initial_state is not None
        assert g_non_spec is not None
        assert beta_non_spec is not None
        if split_non_spec:
            query_non_spec = query_non_spec[:, num_decode_tokens:]
            key_non_spec = key_non_spec[:, num_decode_tokens:]
            value_non_spec = value_non_spec[:, num_decode_tokens:]
            g_non_spec = g_non_spec[:, num_decode_tokens:]
            beta_non_spec = beta_non_spec[:, num_decode_tokens:]

        use_fla_gdn_prefill = get_pcp_group().world_size == 1 and get_current_hardware_profile().supports(
            HardwareCapability.FLA_GDN_PREFILL
        )
        fla_gdn_prefill_op = _get_fla_gdn_prefill_op() if use_fla_gdn_prefill else None
        # The built-in fused operator is NPU-only. Keep CPU/unit-test
        # dispatch on the Triton-compatible path even when its availability
        # probe was cached globally.
        use_fused_chunk = (
            fla_gdn_prefill_op is None
            and query_non_spec.device.type != "cpu"
            and AscendGatedDeltaNetAttention._probe_fused_chunk()
            and get_pcp_group().world_size == 1
        )
        if fla_gdn_prefill_op is not None:
            initial_state = gather_ssm_states(
                ssm_state,
                prefill_state_indices,
                prefill_has_initial_state,
            )
            (core_attn_out_non_spec, last_recurrent_state) = DeviceOperator.fla_gdn_prefill(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                scale=key_non_spec.shape[-1] ** -0.5,
                prebuilt_meta=attn_metadata.non_spec_prefill_metadata.chunk,
                fused_fwd=fla_gdn_prefill_op,
            )
            scatter_ssm_states_(
                ssm_state,
                prefill_state_indices,
                last_recurrent_state.to(ssm_state.dtype).contiguous(),
            )
        elif use_fused_chunk:
            # Gather only the selected rows. Generic advanced indexing first
            # materializes the complete cache when ssm_state has a padded
            # batch stride under the hybrid KV-cache manager.
            initial_state = gather_ssm_states(
                ssm_state,
                prefill_state_indices,
                prefill_has_initial_state,
                output_dtype=torch.bfloat16,
            )
            core_attn_out_non_spec, last_recurrent_state = (
                AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    cu_seqlens=prefill_query_start_loc,
                    scale=key_non_spec.shape[-1] ** -0.5,
                )
            )
            scatter_ssm_states_(
                ssm_state,
                prefill_state_indices,
                last_recurrent_state,
            )
        else:
            initial_state = ssm_state[prefill_state_indices].transpose(-1, -2).contiguous()
            clear_ssm_states(initial_state, prefill_has_initial_state)
            (core_attn_out_non_spec, last_recurrent_state) = chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=prefill_query_start_loc,
                prebuilt_meta=attn_metadata.non_spec_prefill_metadata.chunk,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state[prefill_state_indices] = (
                last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
            )
        if split_non_spec:
            core_attn_out_non_spec = torch.cat(
                [core_attn_out_decode, core_attn_out_non_spec],
                dim=1,
            )
    elif attn_metadata.num_decodes > 0:
        actual_seq_lengths = attn_metadata.non_spec_decode_metadata.actual_seq_lengths
        query_non_spec = l2norm_fwd(query_non_spec)
        key_non_spec = l2norm_fwd(key_non_spec)
        core_attn_out_non_spec = recurrent_gated_delta_rule(
            query_non_spec.squeeze(0),
            key_non_spec.squeeze(0),
            value_non_spec.squeeze(0),
            ssm_state,
            g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
            beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
            scale=key_non_spec.shape[-1] ** -0.5,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=non_spec_state_indices_tensor,
        ).unsqueeze(0)
    else:
        core_attn_out_non_spec, last_recurrent_state = None, None

    # 3. Merge core attention output
    if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
        merged_out = torch.empty(
            (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
            dtype=core_attn_out_non_spec.dtype,
            device=core_attn_out_non_spec.device,
        )
        merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
        merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
    elif spec_sequence_masks is not None:
        core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
    else:
        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
    maybe_save_kv_layer_to_connector("", [])
'''


def _values(shape, *, offset=0, scale=0.03, dtype=torch.float32):
    count = 1
    for length in shape:
        count *= length
    data = ((torch.arange(count) + offset) % 37 - 18).float() * scale
    return data.reshape(shape).to(dtype)


def _normalise(x):
    return (x.float() / (x.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(
        x.dtype
    )


def _gate(A_log, a, b, bias):
    x = a.float() + bias.float()
    softplus = torch.where(x <= 20, torch.log1p(x.exp()), x)
    return -A_log.float().exp() * softplus, b.float().sigmoid().to(b.dtype)


def _sequence(checkpoint, q, k, v, g, beta, *, normalise, scale):
    if normalise:
        q, k = _normalise(q), _normalise(k)
    ratio = v.shape[1] // k.shape[1]
    keys = k.float().repeat_interleave(ratio, dim=1)
    queries = q.float().repeat_interleave(ratio, dim=1)
    state = checkpoint.float().clone()
    outputs, corrections, decays = [], [], []
    for token in range(q.shape[0]):
        decay = g[token].float().exp()
        decayed = state * decay[:, None, None]
        correction = v[token].float() - (decayed * keys[token, :, None, :]).sum(-1)
        correction *= beta[token].float()[:, None]
        state = decayed + correction[:, :, None] * keys[token, :, None, :]
        outputs.append((state * queries[token, :, None, :]).sum(-1) * scale)
        corrections.append(correction)
        decays.append(decay)
    if not outputs:
        return torch.empty_like(v), state, None, None, k
    return (
        torch.stack(outputs).to(v.dtype),
        state,
        torch.stack(corrections),
        torch.stack(decays),
        k,
    )


class _Copies(TorchDispatchMode):
    def __init__(self, actual, value_heads):
        super().__init__()
        self.shape = (actual, value_heads, 128)
        self.full_output_copies = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.copy_.default and tuple(args[0].shape) == self.shape:
            self.full_output_copies += 1
        return func(*args, **(kwargs or {}))


@pytest.fixture
def runtime(monkeypatch):
    package_name = "_gdn_mixed_forward_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, package_name, package)

    def load(name):
        fullname = package_name + "." + name
        spec = importlib.util.spec_from_file_location(fullname, ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
        return module

    backend = load("ascendc_backend")
    backend._ENABLED, backend._OPS = True, object()
    backend._log = lambda: logging.getLogger("gdn-mixed-cpu")
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda _: logging.getLogger("gdn-mixed-cpu")
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    module = load("mixed_forward")
    events = []
    rt = types.SimpleNamespace(
        module=module,
        backend=backend,
        events=events,
        monkeypatch=monkeypatch,
        context=types.SimpleNamespace(attn_metadata={}),
        active=None,
        fail_spec_conv=False,
    )

    def conv_update(inputs, state, weights, **kwargs):
        events.append(("conv_spec_or_decode", inputs.shape[0]))
        state.add_(1)
        if rt.fail_spec_conv:
            raise RuntimeError("owned convolution failed after state update")
        kwargs["out"].copy_((inputs.float() * 0.5 + 0.01).to(inputs.dtype))
        return kwargs["out"]

    def conv_prefill(inputs, weights, bias, **kwargs):
        events.append(("conv_non_spec", inputs.shape[0]))
        kwargs["conv_states"].add_(1)
        return (inputs.float() * 0.5 + 0.01).to(inputs.dtype)

    def gating(A_log, a, b, bias):
        events.append(("gate", a.shape[0], a.clone(), b.clone()))
        g, beta = _gate(A_log, a, b, bias)
        return g.unsqueeze(0), beta.unsqueeze(0)

    def norm(value):
        events.append(("norm", value.shape[1]))
        return _normalise(value)

    def verify(layer, q, k, v, state, kwargs, *, out=None):
        events.append(("fused_verify", q.shape[0], kwargs))
        assert kwargs["raw_fused"] == (layer.A_log, layer.dt_bias)
        assert kwargs["raw_fused_scope"] == "mixed"
        assert state is layer.kv_cache[1]
        g, beta = _gate(layer.A_log, kwargs["g"], kwargs["beta"], layer.dt_bias)
        result = verify_reference(layer, q, k, v, g, beta, normalise=True)
        if out is not None:
            out.copy_(result.unsqueeze(0))
            return out.squeeze(0)
        return result

    def verify_reference(layer, q, k, v, g, beta, *, normalise):
        meta = rt.active.meta
        result = torch.zeros_like(v)
        records = layer._gdn_recoverssm_records
        starts = meta.spec_query_start_loc.tolist()
        for row, (begin, end) in enumerate(pairwise(starts)):
            block = int(meta.spec_state_indices_tensor[row, 0])
            if not 0 < block < layer.kv_cache[1].shape[0] or begin == end:
                continue
            output, _, corr, decay, norm_k = _sequence(
                layer.kv_cache[1][block],
                q[begin:end],
                k[begin:end],
                v[begin:end],
                g[begin:end],
                beta[begin:end],
                normalise=normalise,
                scale=128**-0.5,
            )
            result[begin:end] = output
            length = end - begin
            records.correction[row, :, :length] = corr.transpose(0, 1)
            records.key[row, :, :length] = norm_k.transpose(0, 1)
            records.decay[row, :, :length] = decay.transpose(0, 1)
        return result

    def recurrent(q, k, v, state, *, g, beta, scale, **kwargs):
        events.append(
            ("ordinary_recurrent", q.shape[0], "num_accepted_tokens" in kwargs)
        )
        layer = rt.active.layer
        if "num_accepted_tokens" in kwargs:
            return verify_reference(layer, q, k, v, g, beta, normalise=False)
        indices = kwargs["ssm_state_indices"].flatten().tolist()
        result = torch.zeros_like(v)
        for token, block in enumerate(indices):
            output, last, _, _, _ = _sequence(
                state[block],
                q[token : token + 1],
                k[token : token + 1],
                v[token : token + 1],
                g[token : token + 1],
                beta[token : token + 1],
                normalise=False,
                scale=scale,
            )
            result[token] = output[0]
            state[block] = last.to(state.dtype)
        return result

    def chunk(
        *, q, k, v, g, beta, initial_state, cu_seqlens, scale=128**-0.5, **kwargs
    ):
        events.append(("chunk_prefill", q.shape[1]))
        out = torch.zeros_like(v)
        last = []
        starts = cu_seqlens.tolist()
        for row, (begin, end) in enumerate(pairwise(starts)):
            output, final, _, _, _ = _sequence(
                initial_state[row].transpose(-1, -2),
                q[0, begin:end],
                k[0, begin:end],
                v[0, begin:end],
                g[0, begin:end],
                beta[0, begin:end],
                normalise=True,
                scale=scale,
            )
            out[0, begin:end] = output
            last.append(final.transpose(-1, -2))
        return out, torch.stack(last)

    def clear(initial, has_initial):
        for row, enabled in enumerate(has_initial.tolist()):
            if not enabled:
                initial[row].zero_()

    class Metadata(types.SimpleNamespace):
        pass

    namespace = {
        "torch": torch,
        "AttentionMetadata": dict,
        "GDNAttentionMetadata": Metadata,
        "get_forward_context": lambda: rt.context,
        "wait_for_kv_layer_from_connector": lambda prefix: events.append(
            ("wait", prefix)
        ),
        "record_attention_compute_start": lambda: events.append(("fence",)),
        "maybe_save_kv_layer_to_connector": lambda *args: events.append(("save",)),
        "_get_packed_conv_weights": lambda layer: layer.conv1d.weight,
        "_normalize_causal_cache_indices": lambda value: value,
        "causal_conv1d_update": conv_update,
        "causal_conv1d_fn": conv_prefill,
        "get_pcp_group": lambda: types.SimpleNamespace(world_size=1),
        "PAD_SLOT_ID": 0,
        "get_current_hardware_profile": lambda: types.SimpleNamespace(
            supports=lambda _: False
        ),
        "HardwareCapability": types.SimpleNamespace(FLA_GDN_PREFILL=object()),
        "AscendGatedDeltaNetAttention": types.SimpleNamespace(
            _probe_fused_chunk=lambda: False
        ),
        "DeviceOperator": types.SimpleNamespace(fused_gdn_gating=gating),
        "l2norm_fwd": norm,
        "recurrent_gated_delta_rule": recurrent,
        "clear_ssm_states": clear,
        "chunk_gated_delta_rule": chunk,
    }
    relative = types.ModuleType(package_name + ".layer_patch")
    relative._verify = verify
    monkeypatch.setitem(sys.modules, relative.__name__, relative)
    rt.Metadata, rt.namespace = Metadata, namespace

    def compile_core(source=_NATIVE_CORE):
        filename = "<owned-native-rc1-core-" + str(id(source)) + ">"
        monkeypatch.setitem(
            linecache.cache,
            filename,
            (len(source), None, source.splitlines(True), filename),
        )
        exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - frozen, owned native fixture.
        return namespace["_forward_core"]

    rt.compile_core = compile_core
    rt.original = compile_core()

    def case(
        *,
        dtype=torch.bfloat16,
        bias_dtype=torch.bfloat16,
        mode="prefill",
        lengths=(4, 4),
        null=False,
    ):
        spec = mode not in ("only_prefill", "only_decode")
        n_spec = sum(lengths) if spec else 0
        if mode == "pure_spec":
            non_lengths = []
        elif mode in ("one_token", "only_decode"):
            non_lengths = [1] if mode == "one_token" else [1, 1]
        elif mode == "prefill_plus_one":
            non_lengths = [1, 3]
        else:
            non_lengths = [3]
        n_non = sum(non_lengths)
        actual = n_spec + n_non
        # Deterministic irregular/interleaved gather order. Source ranges are
        # not assumed to be contiguous prefixes or sorted by token position.
        order = torch.tensor(
            list(range(0, actual, 2))[::-1] + list(range(1, actual, 2)),
            dtype=torch.int64,
        )
        spec_idx, non_idx = order[:n_spec], order[n_spec:]
        if mode == "pure_spec":
            spec_idx = torch.arange(actual, dtype=torch.int64)
        qsl = torch.tensor(
            [0] + list(accumulate(lengths if spec else [])), dtype=torch.int32
        )
        non_qsl = torch.tensor([0] + list(accumulate(non_lengths)), dtype=torch.int32)
        non_blocks = torch.tensor([2, 4][: len(non_lengths)], dtype=torch.int32)
        state = _values((6, 3, 128, 128), scale=0.0003)
        state_before = state.clone()
        record = types.SimpleNamespace(
            correction=torch.full((2, 3, 4, 128), -17.0),
            key=torch.full((2, 1, 4, 128), -16.0, dtype=dtype),
            decay=torch.full((2, 3, 4), -19.0),
        )
        layer = types.SimpleNamespace(
            prefix="layer.0",
            head_k_dim=128,
            head_v_dim=128,
            num_k_heads=4,
            num_v_heads=12,
            key_dim=512,
            value_dim=1536,
            tp_size=4,
            num_spec=3,
            conv_kernel_size=4,
            activation=True,
            A_log=torch.tensor([-1.0, -0.5, 0.0]),
            dt_bias=torch.nn.Parameter(
                torch.tensor([-0.2, 0.1, 0.3], dtype=bias_dtype)
            ),
            kv_cache=(torch.zeros(6, 4), state),
            conv1d=types.SimpleNamespace(weight=torch.empty(640, 4), bias=None),
            _gdn_recoverssm_records=record,
        )

        def rearrange(packed):
            if packed is None:
                return None, None, None
            events.append(("native_rearrange", packed.shape[0]))
            q, k, v = packed.split((128, 128, 384), dim=-1)
            return (
                q.contiguous().view(1, -1, 1, 128),
                k.contiguous().view(1, -1, 1, 128),
                v.contiguous().view(1, -1, 3, 128),
            )

        layer.rearrange_mixed_qkv = rearrange
        conv_spec = types.SimpleNamespace(
            query_start_loc=qsl,
            cache_indices=torch.tensor([3, 1], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([1, 3], dtype=torch.int32),
        )
        conv_non = types.SimpleNamespace(
            query_start_loc=non_qsl,
            cache_indices=non_blocks,
            initial_state_mode=torch.tensor([True] * len(non_lengths)),
        )
        meta = Metadata(
            num_actual_tokens=actual,
            num_spec_decodes=len(lengths) if spec else 0,
            num_prefills=len(non_lengths) if mode != "only_decode" else 0,
            num_decodes=len(non_lengths) if mode == "only_decode" else 0,
            num_decode_tokens=n_non if mode == "only_decode" else 0,
            spec_sequence_masks=object() if spec else None,
            spec_token_indx=spec_idx if spec else None,
            non_spec_token_indx=non_idx if spec else None,
            spec_query_start_loc=qsl,
            spec_state_indices_tensor=torch.tensor(
                [[0 if null else 3], [1]], dtype=torch.int32
            ),
            non_spec_state_indices_tensor=non_blocks,
            non_spec_query_start_loc=non_qsl,
            prefill_query_start_loc=non_qsl,
            prefill_state_indices=non_blocks,
            prefill_has_initial_state=torch.tensor([True] * len(non_lengths)),
            spec_decode_metadata=types.SimpleNamespace(
                spec_causal_conv1d=conv_spec, actual_seq_lengths=qsl
            ),
            non_spec_prefill_metadata=types.SimpleNamespace(
                causal_conv1d=conv_non, chunk=object()
            ),
            non_spec_decode_metadata=types.SimpleNamespace(
                causal_conv1d=conv_non, actual_seq_lengths=non_qsl
            ),
        )
        # Native ordinary-decode classification has no spec mask. With a spec
        # subset, one-token non-spec rows are classified as prefill by rc1.
        if mode == "pure_spec":
            meta.num_prefills = 0
        mixed = _values((actual + 3, 640), offset=3, scale=0.02, dtype=dtype)
        a = _values((actual + 3, 3), offset=5, scale=0.1, dtype=dtype)
        b = _values((actual + 3, 3), offset=9, scale=0.1, dtype=dtype)
        destination = torch.full((actual + 3, 3, 128), 91, dtype=dtype)
        return types.SimpleNamespace(
            layer=layer,
            meta=meta,
            mixed=mixed,
            a=a,
            b=b,
            out=destination,
            actual=actual,
            spec_tokens=n_spec,
            non_tokens=n_non,
            original_state=state_before,
        )

    rt.case = case

    def run(core, owned):
        rt.active = owned
        rt.context.attn_metadata = {owned.layer.prefix: owned.meta}
        events.clear()
        copies = _Copies(owned.actual, 3)
        with torch.no_grad(), copies:
            core(owned.layer, owned.mixed, owned.b, owned.a, owned.out)
        return list(events), copies.full_output_copies

    rt.run = run
    return rt


@pytest.mark.parametrize("mode", ["prefill", "one_token", "prefill_plus_one"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
def test_transformed_native_core_matches_original_interleaved_mixed(
    runtime, mode, dtype, bias_dtype
):
    original_case = runtime.case(mode=mode, dtype=dtype, bias_dtype=bias_dtype)
    original_events, original_copies = runtime.run(runtime.original, original_case)
    fused = runtime.module.build_mixed_core(runtime.original)
    owned = runtime.case(mode=mode, dtype=dtype, bias_dtype=bias_dtype)
    events, copies = runtime.run(fused, owned)
    assert original_copies == 1 and copies == 0
    for got, expected in (
        (owned.out, original_case.out),
        (owned.layer.kv_cache[1], original_case.layer.kv_cache[1]),
        (owned.layer.kv_cache[0], original_case.layer.kv_cache[0]),
    ):
        torch.testing.assert_close(
            got, expected, rtol=5e-3 if got.dtype == torch.bfloat16 else 1e-5, atol=1e-7
        )
    for name in ("correction", "key", "decay"):
        torch.testing.assert_close(
            getattr(owned.layer._gdn_recoverssm_records, name),
            getattr(original_case.layer._gdn_recoverssm_records, name),
            rtol=5e-3 if name == "key" and dtype == torch.bfloat16 else 1e-5,
            atol=1e-7,
        )
    assert len([e for e in events if e[0] == "fused_verify"]) == 1
    gates = [e for e in events if e[0] == "gate"]
    assert len(gates) == 1 and gates[0][1] == owned.non_tokens
    assert [e[1] for e in original_events if e[0] == "gate"] == [owned.actual]
    assert not [e for e in events if e[0] == "norm"]
    assert [e[1] for e in events if e[0] == "native_rearrange"] == [owned.non_tokens]
    assert events[0][0] == "wait" and events[1][0] == "fence"
    assert events[-1][0] == "save"
    assert torch.equal(
        owned.out[owned.actual :], torch.full_like(owned.out[owned.actual :], 91)
    )
    assert torch.equal(owned.layer.kv_cache[1][1], owned.original_state[1])
    assert torch.equal(owned.layer.kv_cache[1][3], owned.original_state[3])


@pytest.mark.parametrize("lengths", [(1, 4), (0, 4), (3, 2)])
@pytest.mark.parametrize("null", [False, True])
def test_mixed_partial_windows_null_and_record_rows_keep_native_mapping(
    runtime, lengths, null
):
    original_case = runtime.case(lengths=lengths, null=null)
    runtime.run(runtime.original, original_case)
    owned = runtime.case(lengths=lengths, null=null)
    qsl, indices = owned.meta.spec_query_start_loc, owned.meta.spec_state_indices_tensor
    runtime.run(runtime.module.build_mixed_core(runtime.original), owned)
    torch.testing.assert_close(owned.out, original_case.out)
    assert owned.meta.spec_query_start_loc is qsl
    assert owned.meta.spec_state_indices_tensor is indices
    for name in ("correction", "key", "decay"):
        torch.testing.assert_close(
            getattr(owned.layer._gdn_recoverssm_records, name),
            getattr(original_case.layer._gdn_recoverssm_records, name),
        )


@pytest.mark.parametrize("mode", ["only_prefill", "only_decode", "pure_spec"])
def test_non_mixed_modes_keep_original_core_path(runtime, mode):
    baseline = runtime.case(mode=mode)
    baseline_events, baseline_copies = runtime.run(runtime.original, baseline)
    owned = runtime.case(mode=mode)
    events, copies = runtime.run(
        runtime.module.build_mixed_core(runtime.original), owned
    )
    assert not [e for e in events if e[0] == "fused_verify"]
    assert [e[0] for e in events] == [e[0] for e in baseline_events]
    assert copies == baseline_copies
    torch.testing.assert_close(owned.out, baseline.out)
    torch.testing.assert_close(owned.layer.kv_cache[1], baseline.layer.kv_cache[1])


def test_preflight_uses_spec_token_count_and_real_bf16_bias(runtime):
    owned = runtime.case()
    assert owned.actual > owned.spec_tokens == 8
    assert owned.layer.dt_bias.dtype == torch.bfloat16
    assert not runtime.backend.can_fuse_verify(
        owned.layer, owned.mixed, owned.a, owned.b, meta=owned.meta, out=owned.out
    )
    plan = runtime.module.mixed_preflight(
        owned.layer, owned.meta, owned.mixed, owned.a, owned.b, owned.out
    )
    assert plan is not None
    assert plan.spec_tokens == owned.spec_tokens
    assert plan.non_spec_tokens == owned.non_tokens
    assert plan.spec_indices is owned.meta.spec_token_indx
    assert plan.non_spec_indices is owned.meta.non_spec_token_indx
    assert owned.meta.spec_query_start_loc.dtype == torch.int32
    assert owned.meta.spec_state_indices_tensor.dtype == torch.int32


@pytest.mark.parametrize(
    "bad",
    [
        "empty_spec",
        "empty_non_spec",
        "index_dtype",
        "int32_indices",
        "strided_indices",
        "gate_layout",
        "output_layout",
        "fp16_bias",
        "ordinary_with_spec",
    ],
)
def test_invalid_mixed_signature_is_declined_before_state_update(runtime, bad):
    owned = runtime.case()
    if bad == "empty_spec":
        owned.meta.spec_token_indx = torch.empty(0, dtype=torch.int64)
        owned.meta.non_spec_token_indx = torch.arange(owned.actual, dtype=torch.int64)
    elif bad == "empty_non_spec":
        owned.meta.non_spec_token_indx = torch.empty(0, dtype=torch.int64)
        owned.meta.spec_token_indx = torch.arange(owned.actual, dtype=torch.int64)
    elif bad == "index_dtype":
        owned.meta.spec_token_indx = owned.meta.spec_token_indx.float()
    elif bad == "int32_indices":
        owned.meta.spec_token_indx = owned.meta.spec_token_indx.int()
    elif bad == "strided_indices":
        source = torch.repeat_interleave(owned.meta.spec_token_indx, 2)
        owned.meta.spec_token_indx = source[::2]
    elif bad == "gate_layout":
        owned.a = torch.repeat_interleave(owned.a, 2, dim=1)[:, ::2]
    elif bad == "output_layout":
        owned.out = torch.empty(owned.actual + 3, 3, 256)[..., ::2]
    elif bad == "fp16_bias":
        owned.layer.dt_bias = torch.nn.Parameter(owned.layer.dt_bias.half())
    else:
        owned.meta.num_decodes = 1
    before = owned.layer.kv_cache[0].clone()
    assert (
        runtime.module.mixed_preflight(
            owned.layer, owned.meta, owned.mixed, owned.a, owned.b, owned.out
        )
        is None
    )
    assert torch.equal(owned.layer.kv_cache[0], before)
    assert not runtime.events


@pytest.mark.parametrize("bad", ["partition", "capacity"])
def test_invalid_total_geometry_raises_before_convolution(runtime, bad):
    owned = runtime.case()
    if bad == "partition":
        owned.meta.non_spec_token_indx = owned.meta.non_spec_token_indx[:1]
    else:
        owned.meta.num_actual_tokens = owned.out.shape[0] + 1
    before = owned.layer.kv_cache[0].clone()
    with pytest.raises(ValueError):
        runtime.module.mixed_preflight(
            owned.layer, owned.meta, owned.mixed, owned.a, owned.b, owned.out
        )
    assert torch.equal(owned.layer.kv_cache[0], before)
    assert not runtime.events


def test_mixed_preflight_does_not_read_device_values(runtime):
    owned = runtime.case()

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must inspect tensor metadata only")

    with runtime.monkeypatch.context() as patch:
        for name in ("cpu", "item", "tolist", "numpy"):
            patch.setattr(torch.Tensor, name, forbidden)
        plan = runtime.module.mixed_preflight(
            owned.layer, owned.meta, owned.mixed, owned.a, owned.b, owned.out
        )
    assert plan is not None


def test_declined_fusion_keeps_all_original_helpers(runtime):
    owned = runtime.case()
    runtime.backend._ENABLED = False
    events, copies = runtime.run(
        runtime.module.build_mixed_core(runtime.original), owned
    )
    assert not [e for e in events if e[0] == "fused_verify"]
    assert [e[1] for e in events if e[0] == "gate"] == [owned.actual]
    assert len([e for e in events if e[0] == "norm"]) == 2
    assert copies == 1


def test_partial_convolution_failure_is_propagated_without_retry(runtime):
    owned = runtime.case()
    runtime.fail_spec_conv = True
    core = runtime.module.build_mixed_core(runtime.original)
    with pytest.raises(RuntimeError, match="after state update"):
        runtime.run(core, owned)
    assert [e[0] for e in runtime.events] == ["wait", "fence", "conv_spec_or_decode"]
    assert torch.equal(
        owned.layer.kv_cache[0], torch.ones_like(owned.layer.kv_cache[0])
    )


def test_later_native_global_redirect_is_visible_to_transformed_core(runtime):
    core = runtime.module.build_mixed_core(runtime.original)
    original_recurrent = runtime.namespace["recurrent_gated_delta_rule"]
    calls = []

    def later(*args, **kwargs):
        calls.append(True)
        return original_recurrent(*args, **kwargs)

    runtime.namespace["recurrent_gated_delta_rule"] = later
    runtime.run(core, runtime.case(mode="only_decode"))
    assert calls == [True]
    assert core.__globals__ is runtime.original.__globals__


@pytest.mark.parametrize("change", ["missing_gate", "duplicate_qkv"])
def test_source_abi_mismatch_fails_before_runtime(runtime, change):
    source = _NATIVE_CORE
    if change == "missing_gate":
        source = source.replace(
            "DeviceOperator.fused_gdn_gating", "DeviceOperator.changed_gate"
        )
    else:
        line = "    query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)"
        assert source.count(line) == 1
        source = source.replace(line, line + "\n" + line)
    original = runtime.compile_core(source)
    with pytest.raises(runtime.module.UnsupportedCoreError):
        runtime.module.build_mixed_core(original)
    assert not runtime.events
