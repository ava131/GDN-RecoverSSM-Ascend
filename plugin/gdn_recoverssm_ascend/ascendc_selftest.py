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


def _probe(case, device):
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
    AscendCSpec("verify", tensors, verify_config(case, tensors), case["scale"]).launch()
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
        if not torch.equal(key_cpu[row, 0], cpu["k"][0, start:end, 0]):
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
    init_logger("vllm.gdn_recoverssm_ascend.ascendc").info(
        "GDN_ASCENDC_SELFTEST PASS cases=%d verify_calls=%d fold_calls=%d "
        "Q=1,4,8 activation=fp32,bf16 state=fp32,bf16 beta=fp32,activation "
        "GQA=3 prefix=0..Q null_padding=PASS readonly=PASS key_bits=PASS "
        "boundary_source_final_alias=PASS tolerance_fp32=%g tolerance_bf16=%g",
        calls,
        calls,
        calls,
        TOL_FP32,
        TOL_BF16,
    )
    return True
