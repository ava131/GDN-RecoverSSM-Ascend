# SPDX-License-Identifier: Apache-2.0
"""Mandatory startup-only AscendC probes on owned buffers.

No model KV, request data or RNG is used here. Pure CPU helpers are also usable
by the test suite; only run() imports the backend and touches the NPU.
"""

from __future__ import annotations

import torch

VERIFY_FIELDS = (
    "rows",
    "tokens",
    "heads",
    "value_heads",
    "query_len",
    "blocks",
    "record_rows",
    "null_block",
    "q_token_stride",
    "k_token_stride",
    "v_token_stride",
    "g_token_stride",
    "beta_token_stride",
    "state_block_stride",
    "state_head_stride",
    "state_v_stride",
    "corr_row_stride",
    "corr_head_stride",
    "corr_token_stride",
    "key_row_stride",
    "key_head_stride",
    "key_token_stride",
    "decay_row_stride",
    "decay_head_stride",
    "decay_token_stride",
    "out_token_stride",
    "qsl_stride",
    "index_stride",
)
FOLD_FIELDS = (
    "rows",
    "layers",
    "heads",
    "key_heads",
    "query_len",
    "blocks",
    "record_rows",
    "plan_capacity",
    "null_block",
    "state_head_stride",
    "state_v_stride",
    "corr_head_stride",
    "corr_token_stride",
    "key_head_stride",
    "key_token_stride",
    "decay_head_stride",
    "decay_token_stride",
)
TOL_FP32 = 1e-5
TOL_BF16 = 5e-3


def _values(shape, *, offset=0, amplitude=0.1):
    count = 1
    for dim in shape:
        count *= dim
    # Integral construction and no random APIs: startup preserves all RNG state.
    values = (torch.arange(count, dtype=torch.float64) + offset).remainder(37)
    return ((values - 18) * (amplitude / 18)).reshape(shape)


def make_case(query_len, activation_dtype, state_dtype, beta_dtype):
    """Build CPU inputs covering every prefix, null and zero-length padding."""
    if query_len not in (1, 4, 8):
        raise ValueError("probe query length must be 1, 4, or 8")
    if any(
        dtype not in (torch.float32, torch.bfloat16)
        for dtype in (activation_dtype, state_dtype, beta_dtype)
    ):
        raise ValueError("AscendC probes cover BF16/FP32 only")
    if beta_dtype not in (activation_dtype, torch.float32):
        raise ValueError("beta must match activation or be FP32")
    real_rows = query_len + 1
    rows = real_rows + 2
    tokens = (real_rows + 1) * query_len
    heads, value_heads, dim = 1, 3, 128
    q = _values((1, tokens, heads, dim), offset=1)
    k = _values((1, tokens, heads, dim), offset=9)
    q = (q / q.square().sum(-1, keepdim=True).sqrt()).to(activation_dtype)
    k = (k / k.square().sum(-1, keepdim=True).sqrt()).to(activation_dtype)
    v = _values((1, tokens, value_heads, dim), offset=13).to(activation_dtype)
    g = (-0.08 + _values((tokens, value_heads), amplitude=0.01)).float()
    beta = (0.3 + _values((tokens, value_heads), offset=7, amplitude=0.04)).to(
        beta_dtype
    )
    qsl = torch.arange(rows + 1, dtype=torch.int32) * query_len
    qsl[-1] = qsl[-2]  # Padding has no token segment.
    indices = torch.arange(1, rows + 1, dtype=torch.int32)
    indices[-2:] = 0
    blocks = 3 * real_rows + 1
    state = _values((blocks, value_heads, dim, dim), offset=5, amplitude=0.01).to(
        state_dtype
    )
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        qsl=qsl,
        indices=indices,
        state=state,
        rows=rows,
        real_rows=real_rows,
        tokens=tokens,
        query_len=query_len,
        scale=dim**-0.5,
    )


def sequential_reference(checkpoint, q, k, v, g, beta, scale):
    """Independent FP64 GDN oracle: output, all prefix states, correction, decay."""
    convert = lambda x: x.detach().to(device="cpu", dtype=torch.float64)
    state = convert(checkpoint).clone()
    q, k, v, g, beta = map(convert, (q, k, v, g, beta))
    ratio = state.shape[0] // k.shape[1]
    q = q.repeat_interleave(ratio, dim=1)
    expanded_k = k.repeat_interleave(ratio, dim=1)
    states, outputs, corrections, decays = [state.clone()], [], [], []
    for token in range(q.shape[0]):
        decay = g[token].exp()
        decayed = state * decay[:, None, None]
        correction = (
            v[token] - (decayed * expanded_k[token, :, None, :]).sum(-1)
        ) * beta[token, :, None]
        state = decayed + correction[:, :, None] * expanded_k[token, :, None, :]
        outputs.append((state * q[token, :, None, :]).sum(-1) * scale)
        corrections.append(correction)
        decays.append(decay)
        states.append(state.clone())
    return (torch.stack(outputs), states, torch.stack(corrections), torch.stack(decays))


def fold_reference(checkpoint, keys, corrections, decays, accepted):
    """FP64 record fold, separate from the verify recurrence implementation."""
    if not 0 <= accepted <= corrections.shape[1]:
        raise ValueError("invalid accepted prefix")
    state = checkpoint.detach().to(device="cpu", dtype=torch.float64).clone()
    keys = keys.detach().to(device="cpu", dtype=torch.float64)
    corrections = corrections.detach().to(device="cpu", dtype=torch.float64)
    decays = decays.detach().to(device="cpu", dtype=torch.float64)
    ratio = state.shape[0] // keys.shape[0]
    keys = keys.repeat_interleave(ratio, dim=0)
    states = [state.clone()]
    for token in range(accepted):
        state = (
            state * decays[:, token, None, None]
            + corrections[:, token, :, None] * keys[:, token, None, :]
        )
        states.append(state.clone())
    return states


def verify_config(case, tensors):
    q, k, v, g, beta, state, corr, key, decay, out, qsl, indices = tensors
    values = (
        case["rows"],
        case["tokens"],
        q.shape[2],
        v.shape[2],
        case["query_len"],
        state.shape[0],
        corr.shape[0],
        0,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        g.stride(0),
        beta.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        corr.stride(0),
        corr.stride(1),
        corr.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        decay.stride(0),
        decay.stride(1),
        decay.stride(2),
        out.stride(1),
        qsl.stride(0),
        indices.stride(0),
    )
    assert len(values) == len(VERIFY_FIELDS)
    return tuple(int(v) for v in values)


def fold_config(rows, layers, capacity, state, corr, key, decay):
    values = (
        rows,
        layers,
        state.shape[1],
        key.shape[1],
        corr.shape[2],
        state.shape[0],
        corr.shape[0],
        capacity,
        0,
        state.stride(1),
        state.stride(2),
        corr.stride(1),
        corr.stride(2),
        key.stride(1),
        key.stride(2),
        decay.stride(1),
        decay.stride(2),
    )
    assert len(values) == len(FOLD_FIELDS)
    return tuple(int(v) for v in values)


def _assert_close(got, expected, *, label, dtype):
    got = got.detach().to(device="cpu", dtype=torch.float64)
    expected = expected.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(got).all() or not torch.isfinite(expected).all():
        raise RuntimeError(f"GDN_ASCENDC_SELFTEST {label}: nonfinite value")
    error = float((got - expected).norm())
    denominator = max(float(expected.norm()), 1e-12)
    tolerance = TOL_BF16 if dtype == torch.bfloat16 else TOL_FP32
    if error / denominator > tolerance and float((got - expected).abs().max()) > 1e-7:
        raise RuntimeError(
            f"GDN_ASCENDC_SELFTEST {label}: relative_error={error / denominator:.3g} "
            f"tolerance={tolerance}"
        )


def _probe(case, device, *, fused=False):
    from .ascendc_backend import AscendCSpec

    cpu = case
    q, k, v, g, beta, state, qsl, indices = (
        cpu[name].to(device)
        for name in ("q", "k", "v", "g", "beta", "state", "qsl", "indices")
    )
    rows, query_len = case["rows"], case["query_len"]
    corr = torch.full(
        (rows, 3, query_len, 128), -17.0, dtype=torch.float32, device=device
    )
    key = torch.full((rows, 1, query_len, 128), -16.0, dtype=k.dtype, device=device)
    decay = torch.full((rows, 3, query_len), -19.0, dtype=torch.float32, device=device)
    out = torch.full_like(v, -23.0)
    tensors = (q, k, v, g, beta, state, corr, key, decay, out, qsl, indices)
    if fused:
        q, k, g, beta = (
            cpu[name].to(device) for name in ("raw_q", "raw_k", "raw_a", "raw_b")
        )
        # The model path consumes views of packed convolution output. Probe
        # that actual token stride instead of testing only compact Q/K inputs.
        tokens, heads, value_heads = q.shape[1], q.shape[2], v.shape[2]
        packed = torch.cat(
            (q.reshape(tokens, -1), k.reshape(tokens, -1), v.reshape(tokens, -1)),
            dim=-1,
        )
        qw, vw = heads * 128, value_heads * 128
        q, k, v = packed.split((qw, qw, vw), dim=-1)
        q = q.view(1, tokens, heads, 128)
        k = k.view(1, tokens, heads, 128)
        v = v.view(1, tokens, value_heads, 128)
        tensors = (q, k, v, g, beta, state, corr, key, decay, out, qsl, indices)
        fused_tensors = (*tensors, cpu["A_log"].to(device), cpu["dt_bias"].to(device))
        AscendCSpec(
            "verify_fused", fused_tensors, verify_config(case, tensors), case["scale"]
        ).launch()
    else:
        AscendCSpec(
            "verify", tensors, verify_config(case, tensors), case["scale"]
        ).launch()
    # Startup-owned copies synchronize naturally; no request path is involved.
    state_cpu, out_cpu = state.cpu(), out.cpu()
    corr_cpu, key_cpu, decay_cpu = corr.cpu(), key.cpu(), decay.cpu()
    if not torch.equal(state_cpu, cpu["state"]):
        raise RuntimeError(
            "GDN_ASCENDC_SELFTEST verify changed the resident checkpoint"
        )
    for row in range(case["real_rows"]):
        start, end = row * query_len, (row + 1) * query_len
        outputs, _, corrections, decays = sequential_reference(
            cpu["state"][row + 1],
            cpu["q"][0, start:end],
            cpu["k"][0, start:end],
            cpu["v"][0, start:end],
            cpu["g"][start:end],
            cpu["beta"][start:end],
            case["scale"],
        )
        _assert_close(
            out_cpu[0, start:end], outputs, label="verify output", dtype=q.dtype
        )
        # Verify records are consumed by a separate operator: check their actual
        # values, not just the combined verify/fold final state.
        _assert_close(
            corr_cpu[row].transpose(0, 1),
            corrections,
            label="correction",
            dtype=torch.float32,
        )
        _assert_close(
            decay_cpu[row].transpose(0, 1), decays, label="decay", dtype=torch.float32
        )
        if fused:
            _assert_close(
                key_cpu[row, 0],
                cpu["k"][0, start:end, 0],
                label="fused normalized key",
                dtype=k.dtype,
            )
        elif not torch.equal(key_cpu[row, 0], cpu["k"][0, start:end, 0]):
            raise RuntimeError(
                "GDN_ASCENDC_SELFTEST key record did not preserve input bits"
            )
    null_start = case["real_rows"] * query_len
    if not torch.equal(
        out_cpu[:, null_start:], torch.zeros_like(out_cpu[:, null_start:])
    ):
        raise RuntimeError("GDN_ASCENDC_SELFTEST null output was not zero")
    for name, values, sentinel in (
        ("correction", corr_cpu, -17),
        ("key", key_cpu, -16),
        ("decay", decay_cpu, -19),
    ):
        if not torch.equal(values[-2:], torch.full_like(values[-2:], sentinel)):
            raise RuntimeError(f"GDN_ASCENDC_SELFTEST invalid rows wrote {name}")

    # Two independent layer owners and two distinct plan groups. Layer 1 uses
    # different state/records and reversed accepted prefixes to catch routing
    # errors that homogeneous identical test tensors would hide.
    states_cpu = [cpu["state"].clone(), (cpu["state"].float() * 0.7).to(state.dtype)]
    corrs_cpu = [corr_cpu, corr_cpu * 1.25]
    keys_cpu = [key_cpu, key_cpu.clone()]
    decays_cpu = [decay_cpu, decay_cpu * 0.95]
    states = [x.to(device) for x in states_cpu]
    corrs = [x.to(device) for x in corrs_cpu]
    keys = [x.to(device) for x in keys_cpu]
    decays = [x.to(device) for x in decays_cpu]
    plan = {
        name: torch.zeros((2, rows), dtype=torch.int32)
        for name in ("source", "length", "final", "boundary", "boundary_length")
    }
    expected = [x.clone() for x in states_cpu]
    changed = [set(), set()]
    for layer in range(2):
        for row in range(case["real_rows"]):
            source = row + 1
            accepted = row if layer == 0 else query_len - row
            final = source if row % 3 == 0 else case["real_rows"] + source
            boundary = (
                source
                if row % 3 == 0
                else (final if row % 3 == 1 else 2 * case["real_rows"] + source)
            )
            boundary_length = accepted if row % 3 == 1 else max(1, accepted // 2)
            for name, value in (
                ("source", source),
                ("length", accepted),
                ("final", final),
                ("boundary", boundary),
                ("boundary_length", boundary_length),
            ):
                plan[name][layer, row] = value
            prefixes = fold_reference(
                states_cpu[layer][source],
                keys_cpu[layer][row],
                corrs_cpu[layer][row],
                decays_cpu[layer][row],
                accepted,
            )
            if accepted:
                expected[layer][final] = prefixes[-1].to(state.dtype)
                changed[layer].add(final)
                if boundary != final and 0 < boundary_length <= accepted:
                    expected[layer][boundary] = prefixes[boundary_length].to(
                        state.dtype
                    )
                    changed[layer].add(boundary)
    # Padding/null plan rows are deliberately nonempty acceptance with null pages
    # in one group: this must still do no work and never consume invalid records.
    plan["length"][0, -2] = query_len
    pointer_table = lambda xs: torch.tensor(
        [x.data_ptr() for x in xs], dtype=torch.int64, device=device
    )
    stride_table = lambda xs: torch.tensor(
        [x.stride(0) for x in xs], dtype=torch.int64, device=device
    )
    fold_tensors = (
        states[0],
        corrs[0],
        keys[0],
        decays[0],
        pointer_table(states),
        stride_table(states),
        pointer_table(corrs),
        stride_table(corrs),
        pointer_table(keys),
        stride_table(keys),
        pointer_table(decays),
        stride_table(decays),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        *(
            plan[name].flatten().to(device)
            for name in ("source", "length", "final", "boundary", "boundary_length")
        ),
    )
    config = fold_config(rows, 2, rows, states[0], corrs[0], keys[0], decays[0])
    AscendCSpec("fold", fold_tensors, config).launch()
    for layer in range(2):
        actual = states[layer].cpu()
        for block in range(actual.shape[0]):
            if block in changed[layer]:
                _assert_close(
                    actual[block],
                    expected[layer][block],
                    label="fold page",
                    dtype=state.dtype,
                )
            elif not torch.equal(actual[block], states_cpu[layer][block]):
                raise RuntimeError(
                    "GDN_ASCENDC_SELFTEST fold modified an untouched page"
                )


def make_fused_case(
    query_len, activation_dtype, state_dtype, beta_dtype, bias_dtype=torch.float32
):
    """Raw inputs and independently rounded rc1 preprocessing reference."""
    if bias_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("fused bias probes cover BF16/FP32 only")
    case = make_case(query_len, activation_dtype, state_dtype, beta_dtype)
    case["raw_q"] = _values(case["q"].shape, offset=3).to(activation_dtype)
    case["raw_k"] = _values(case["k"].shape, offset=11).to(activation_dtype)
    # Exercise epsilon, zero reductions and a single live component in both
    # the smallest window and the multi-token specializations.
    case["raw_q"][0, 0] = 0
    case["raw_k"][0, 1] = 0
    case["raw_q"][0, 2] = 0
    case["raw_q"][0, 2, 0, 0] = 1e-5
    for name in ("q", "k"):
        raw = case["raw_" + name].float()
        case[name] = (raw / (raw.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(
            activation_dtype
        )
    shape = case["g"].shape
    case["raw_a"] = _values(shape, offset=5, amplitude=2.0).to(activation_dtype)
    case["raw_b"] = _values(shape, offset=7, amplitude=2.0).to(beta_dtype)
    case["raw_a"].flatten()[::7] = 25.0  # softplus linear branch
    case["raw_a"].flatten()[1::7] = -25.0
    case["raw_b"].flatten()[::7] = 20.0
    case["raw_b"].flatten()[1::7] = -20.0
    case["A_log"] = torch.tensor([-1.0, -0.5, 0.0], dtype=torch.float32)
    case["dt_bias"] = torch.tensor([-0.2, 0.1, 0.3], dtype=bias_dtype)
    # Native gating promotes the parameter's stored value before adding it;
    # BF16 bias must not be replaced with its unquantized FP32 construction.
    x = case["raw_a"].float() + case["dt_bias"].float()
    case["g"] = -case["A_log"].exp() * torch.where(
        x <= 20.0, torch.log(1.0 + x.exp()), x
    )
    case["beta"] = case["raw_b"].float().sigmoid().to(beta_dtype)
    return case


def _mixed_staging_probe(device):
    """Validate mixed gathers/gate/merge on owned buffers, never model state."""
    from types import SimpleNamespace

    from vllm.logger import init_logger
    from vllm_ascend.ops import gdn as native

    from .mixed_forward import MixedPlan, merge_mixed_output, prepare_mixed_gates

    actual, heads = 11, 3
    spec_cpu = torch.tensor([6, 1, 9, 3], dtype=torch.int64)
    other_cpu = torch.tensor([0, 2, 4, 5, 7, 8, 10], dtype=torch.int64)
    cases = 0
    for dtype in (torch.float32, torch.bfloat16):
        for bias_dtype in (torch.float32, torch.bfloat16):
            a_cpu = _values((actual, heads), offset=4, amplitude=2).to(dtype)
            b_cpu = _values((actual, heads), offset=11, amplitude=2).to(dtype)
            logA = torch.tensor([-1.0, -0.5, 0.0], dtype=torch.float32)
            bias = torch.tensor([-0.2, 0.1, 0.3], dtype=bias_dtype)
            layer = SimpleNamespace(A_log=logA.to(device), dt_bias=bias.to(device))
            plan = MixedPlan(spec_cpu.to(device), other_cpu.to(device), 4, 7)
            raw_a, raw_b, g, beta = prepare_mixed_gates(
                layer, a_cpu.to(device), b_cpu.to(device), plan,
                native.DeviceOperator.fused_gdn_gating, record=False,
            )
            if not torch.equal(raw_a.cpu()[0], a_cpu[spec_cpu]) or not torch.equal(
                raw_b.cpu()[0], b_cpu[spec_cpu]
            ):
                raise RuntimeError("GDN_MIXED_SELFTEST raw partition mismatch")
            x = a_cpu[other_cpu].float() + bias.float()
            expected_g = -logA.exp() * torch.where(x <= 20, (1 + x.exp()).log(), x)
            expected_beta = b_cpu[other_cpu].float().sigmoid().to(dtype)
            _assert_close(g[0], expected_g, label="mixed native gate", dtype=torch.float32)
            _assert_close(beta[0], expected_beta, label="mixed native beta", dtype=dtype)
            # Distinct values make accidental subset swaps and padded-tail
            # writes observable without running a synthetic prefill kernel.
            spec_out = torch.stack((raw_a[0], raw_b[0]), dim=-1).unsqueeze(0)
            other_out = torch.stack((g[0], beta[0].float()), dim=-1).to(dtype).unsqueeze(0)
            out = torch.full((actual + 2, heads, 2), -7.0, dtype=dtype, device=device)
            merge_mixed_output(
                out, actual, plan.spec_indices, plan.non_spec_indices,
                spec_out, other_out, record=False,
            )
            expected = torch.full((actual + 2, heads, 2), -7.0, dtype=dtype)
            expected[spec_cpu] = torch.stack((a_cpu[spec_cpu], b_cpu[spec_cpu]), dim=-1)
            expected[other_cpu] = torch.stack((expected_g, expected_beta.float()), dim=-1).to(dtype)
            _assert_close(out, expected, label="mixed output scatter", dtype=dtype)
            if not torch.equal(out.cpu()[actual:], expected[actual:]):
                raise RuntimeError("GDN_MIXED_SELFTEST output padding overwritten")
            cases += 1
    init_logger("vllm.gdn_recoverssm_ascend.ascendc").info(
        "GDN_MIXED_SELFTEST PASS cases=%d raw_partition=PASS native_non_spec_gate=PASS "
        "direct_output_merge=PASS padding=PASS model_buffers_touched=0 "
        "scope=OWNED_STAGING_NOT_FULL_PREFILL", cases,
    )


def run(device):
    """Execute every BF16/FP32 entry-point specialization before graph capture."""
    from vllm.logger import init_logger

    calls = 0
    for query_len in (1, 4, 8):
        for activation in (torch.float32, torch.bfloat16):
            for state in (torch.float32, torch.bfloat16):
                for beta in dict.fromkeys((activation, torch.float32)):
                    _probe(make_case(query_len, activation, state, beta), device)
                    calls += 1
    fused_calls = 0
    for query_len in (1, 4, 8):
        for activation in (torch.float32, torch.bfloat16):
            for state in (torch.float32, torch.bfloat16):
                for beta in dict.fromkeys((activation, torch.float32)):
                    for bias in (torch.float32, torch.bfloat16):
                        _probe(
                            make_fused_case(query_len, activation, state, beta, bias),
                            device,
                            fused=True,
                        )
                        fused_calls += 1
    _mixed_staging_probe(device)
    init_logger("vllm.gdn_recoverssm_ascend.ascendc").info(
        "GDN_ASCENDC_SELFTEST PASS cases=%d verify_calls=%d fold_calls=%d "
        "fused_verify_calls=%d norm_gate_roundtrip=PASS fused_key_tolerance=PASS "
        "Q=1,4,8 activation=fp32,bf16 state=fp32,bf16 beta=fp32,activation "
        "dt_bias=fp32,bf16 bias_storage_promotion=PASS packed_qkv=PASS "
        "GQA=3 prefix=0..Q null_padding=PASS readonly=PASS prenormalized_key_bits=PASS "
        "boundary_source_final_alias=PASS tolerance_fp32=%g tolerance_bf16=%g",
        calls + fused_calls,
        calls,
        calls + fused_calls,
        fused_calls,
        TOL_FP32,
        TOL_BF16,
    )
    return True
