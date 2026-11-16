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


def make_case(
    query_len,
    activation_dtype,
    state_dtype,
    beta_dtype,
    *,
    key_heads=1,
    repeats=1,
    partial=False,
):
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
    if key_heads < 1 or repeats < 1:
        raise ValueError("probe key_heads and repeats must be positive")
    rows_per_repeat = query_len + 3
    rows = repeats * rows_per_repeat
    valid_rows = tuple(r for r in range(rows) if r % rows_per_repeat <= query_len)
    invalid_rows = tuple(r for r in range(rows) if r not in valid_rows)
    real_rows = len(valid_rows)
    lengths = []
    for row in range(rows):
        position = row % rows_per_repeat
        if position == rows_per_repeat - 1:
            length = 0
        elif partial and position < query_len:
            length = position + 1
        else:
            length = query_len
        lengths.append(length)
    tokens = sum(lengths)
    heads, value_heads, dim = key_heads, 3 * key_heads, 128
    q = _values((1, tokens, heads, dim), offset=1)
    k = _values((1, tokens, heads, dim), offset=9)
    q = (q / q.square().sum(-1, keepdim=True).sqrt()).to(activation_dtype)
    k = (k / k.square().sum(-1, keepdim=True).sqrt()).to(activation_dtype)
    v = _values((1, tokens, value_heads, dim), offset=13).to(activation_dtype)
    g = (-0.08 + _values((tokens, value_heads), amplitude=0.01)).float()
    beta = (0.3 + _values((tokens, value_heads), offset=7, amplitude=0.04)).to(
        beta_dtype
    )
    qsl = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).to(torch.int32)
    indices = torch.arange(1, rows + 1, dtype=torch.int32)
    indices[list(invalid_rows)] = 0
    blocks = 3 * rows + 1
    state = _values((blocks, value_heads, dim, dim), offset=5, amplitude=0.01).to(
        state_dtype
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "qsl": qsl,
        "indices": indices,
        "state": state,
        "rows": rows,
        "real_rows": real_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "rows_per_repeat": rows_per_repeat,
        "repeats": repeats,
        "heads": heads,
        "value_heads": value_heads,
        "tokens": tokens,
        "query_len": query_len,
        "scale": dim**-0.5,
    }


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


def fold_targets(row, rows, accepted):
    """Disjoint pages plus each supported alias, including source overwrite."""
    source = row + 1
    alias = row % 4
    final = source if alias == 0 else rows + source
    boundary = (
        source if alias in (0, 3) else (final if alias == 1 else 2 * rows + source)
    )
    boundary_length = accepted if alias == 1 else max(1, accepted // 2)
    return source, final, boundary, boundary_length


def _probe(case, device):
    from .ascendc_backend import AscendCSpec

    cpu = case
    q, k, v, g, beta, state, qsl, indices = (
        cpu[name].to(device)
        for name in ("q", "k", "v", "g", "beta", "state", "qsl", "indices")
    )
    rows, query_len = case["rows"], case["query_len"]
    heads, value_heads = case["heads"], case["value_heads"]
    corr = torch.full(
        (rows, value_heads, query_len, 128), -17.0, dtype=torch.float32, device=device
    )
    key = torch.full((rows, heads, query_len, 128), -16.0, dtype=k.dtype, device=device)
    decay = torch.full(
        (rows, value_heads, query_len), -19.0, dtype=torch.float32, device=device
    )
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
    for row in case["valid_rows"]:
        start, end = map(int, cpu["qsl"][row : row + 2])
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
            corr_cpu[row, :, : end - start].transpose(0, 1),
            corrections,
            label="correction",
            dtype=torch.float32,
        )
        _assert_close(
            decay_cpu[row, :, : end - start].transpose(0, 1),
            decays,
            label="decay",
            dtype=torch.float32,
        )
        if not torch.equal(
            key_cpu[row, :, : end - start], cpu["k"][0, start:end].transpose(0, 1)
        ):
            raise RuntimeError(
                "GDN_ASCENDC_SELFTEST key record did not preserve input bits"
            )
        for name, values, sentinel in (
            ("correction", corr_cpu, -17),
            ("key", key_cpu, -16),
            ("decay", decay_cpu, -19),
        ):
            unused = values[row, :, end - start :]
            if not torch.equal(unused, torch.full_like(unused, sentinel)):
                raise RuntimeError(
                    f"GDN_ASCENDC_SELFTEST partial row wrote unused {name}"
                )
    for row in case["invalid_rows"]:
        start, end = map(int, cpu["qsl"][row : row + 2])
        if not torch.equal(
            out_cpu[:, start:end], torch.zeros_like(out_cpu[:, start:end])
        ):
            raise RuntimeError("GDN_ASCENDC_SELFTEST null output was not zero")
    for name, values, sentinel in (
        ("correction", corr_cpu, -17),
        ("key", key_cpu, -16),
        ("decay", decay_cpu, -19),
    ):
        invalid = values[list(case["invalid_rows"])]
        if not torch.equal(invalid, torch.full_like(invalid, sentinel)):
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
        for row in case["valid_rows"]:
            prefix = row % case["rows_per_repeat"]
            available = int(cpu["qsl"][row + 1] - cpu["qsl"][row])
            accepted = min(prefix if layer == 0 else query_len - prefix, available)
            source, final, boundary, boundary_length = fold_targets(row, rows, accepted)
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
    # More than one grid-stride wave on the supported A2 devices, with null
    # rows interleaved between valid groups. Exercise both packed-output and
    # FP32-output bank reuse, using the deployed GQA head geometry (4/12).
    for activation, state in (
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ):
        _probe(
            make_case(
                4,
                activation,
                state,
                torch.float32,
                key_heads=4,
                repeats=4,
                partial=True,
            ),
            device,
        )
        calls += 1
    init_logger("vllm.gdn_recoverssm_ascend.ascendc").info(
        "GDN_ASCENDC_SELFTEST PASS cases=%d verify_calls=%d fold_calls=%d "
        "Q=1,4,8 activation=fp32,bf16 state=fp32,bf16 beta=fp32,activation "
        "GQA=3 prefix=0..Q null_padding=PASS readonly=PASS key_bits=PASS "
        "reuse_stress_tasks=336 interleaved_null=PASS partial_rows=PASS "
        "boundary_source_final_alias=PASS tolerance_fp32=%g tolerance_bf16=%g",
        calls,
        calls,
        calls,
        TOL_FP32,
        TOL_BF16,
    )
    return True
