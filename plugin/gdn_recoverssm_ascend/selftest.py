# SPDX-License-Identifier: Apache-2.0
"""Startup self-check: does verify+commit reproduce the sequential GDN state?

RFC #19987 asks for the reconstructed recurrent state to match the sequential
GDN update within a numerical tolerance. The CPU oracle in ``tests`` proves the
arithmetic; this check proves the *deployed* Triton kernels agree with a plain
per-token recurrence on the real device, using the real layouts and strides.

It is a startup check, not a per-step one: comparing every decode step would
double the forward work and -- worse -- change the timing of the very path it is
supposed to observe.

Enabled with ``GDN_RECOVERSSM_SELFTEST=1``. The whole matrix costs one short
buffer per case and runs before serving, so it cannot perturb a request.

What it compares
----------------
For a synthetic token sequence and a chosen number of accepted tokens ``m``:

    sequential:  S_{n+m}  -- apply the GDN recurrence one token at a time
    recover:     commit(checkpoint, records from verify of Q tokens)

Both start from the same checkpoint. ``m = 0`` is a zero-commit/no-work
case; rejection of the first draft still commits the target token (``m = 1``).

Reported per case: ``max_abs``, ``max_rel`` and ``cos`` over every (head, v, k),
plus a one-line verdict. A summary line closes the block so a single grep is
enough.
"""

from __future__ import annotations

import os
from functools import wraps

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.gdn_recoverssm_ascend.selftest")

# fp32 kernels: the reference is fp64, so the floor is the kernel's own rounding.
TOL_FP32 = 1e-5
# bf16 state pages round the WHOLE checkpoint on every fold (the record itself is
# fp32), so the reconstruction error grows with the number of folded tokens and is
# dominated by the storage precision, not by the kernel. This leg is a
# "does it still track" check; the fp32 leg is the strict one.
TOL_BF16 = 5e-2

_ENV = "GDN_RECOVERSSM_SELFTEST"


def enabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Reference: the GDN recurrence, one token at a time
# ---------------------------------------------------------------------------


def sequential_state(
    checkpoint: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Apply ``S_t = a S_{t-1} + b (v_t - a S_{t-1} k_t) k_t^T`` token by token.

    ``checkpoint`` is ``[HV, V, K]``; ``k``/``v``/``g``/``beta`` are ``[T, HV, *]``.
    Computed in float64 so the reference itself contributes no error.
    """
    # The NPU has no float64: run the reference on the host so it is a genuine
    # high-precision comparison rather than a silent fp32 down-cast.
    ref = lambda t: t.detach().float().cpu().double()  # noqa: E731
    out = ref(checkpoint)
    for t in range(k.shape[0]):
        k_t = ref(k[t])  # [HV, K]
        v_t = ref(v[t])  # [HV, V]
        a = ref(g[t]).exp()  # [HV]
        b = ref(beta[t])  # [HV]
        prev = out
        # (S k) is a vector per head, shape [HV, V]
        sk = torch.einsum("hvk,hk->hv", prev, k_t)
        out = (
            a[:, None, None] * prev
            + b[:, None, None] * (v_t - a[:, None] * sk)[:, :, None] * k_t[:, None, :]
        )
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _metrics(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float, float]:
    """Absolute error, NORM-wise relative error, cosine similarity.

    An element-wise ``|diff| / |want|`` is not usable here: a recurrent state has
    entries that are legitimately near zero, so a 6e-8 absolute error on such an
    entry reports a large relative error and fails a correct implementation. The
    fp32 leg hit exactly that (max_abs 5.96e-08, "max_rel" 5.7e-3). The norm-wise
    ratio ``||diff|| / ||want||`` is the standard measure and is what the
    tolerance below is written against.
    """
    got = got.detach().float().cpu().double()
    want = want.detach().float().cpu().double()
    diff = (got - want).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    denom = float(want.norm())
    max_rel = float(diff.norm()) / denom if denom > 0 else 0.0
    flat_got = got.reshape(-1)
    flat_want = want.reshape(-1)
    denom = float(flat_got.norm()) * float(flat_want.norm())
    cos = float(torch.dot(flat_got, flat_want)) / denom if denom > 0 else 1.0
    return max_abs, max_rel, cos


def _sequential_outputs(checkpoint, q, k, v, g, beta):
    """Independent host FP64 output and every accepted-prefix state oracle."""
    ref = lambda t: t.detach().float().cpu().double()
    state = ref(checkpoint)
    ratio = state.shape[0] // k.shape[1]
    qh = ref(q).repeat_interleave(ratio, dim=1)
    kh = ref(k).repeat_interleave(ratio, dim=1)
    vh, gh, bh = ref(v), ref(g), ref(beta)
    states = [state.clone()]
    outputs = []
    for t in range(k.shape[0]):
        decayed = state * gh[t].exp()[:, None, None]
        correction = (vh[t] - torch.einsum("hvk,hk->hv", decayed, kh[t])) * bh[
            t, :, None
        ]
        state = decayed + correction[:, :, None] * kh[t, :, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, qh[t]) * k.shape[-1] ** -0.5)
        states.append(state.clone())
    return torch.stack(outputs), states


def _preserve_rng(fn):
    """Startup diagnostics must not change the user's generation seed."""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        device = torch.device(kwargs.get("device", "cpu"))
        cpu_state = torch.random.get_rng_state()
        npu = getattr(torch, "npu", None)
        device_state = None
        if device.type == "npu" and npu is not None:
            device_state = npu.get_rng_state(device)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.random.set_rng_state(cpu_state)
            if device_state is not None:
                npu.set_rng_state(device_state, device)

    return wrapped


@_preserve_rng
def run(
    *,
    num_layers: int,
    num_heads: int,
    head_k: int,
    head_v: int,
    spec_query_len: int,
    device: torch.device,
    dtypes: tuple | None = None,
    seed: int = 0,
    num_key_heads: int | None = None,
    key_dtype: torch.dtype | None = None,
) -> bool:
    """Check every layer's GQA output, prefix state, null and no-work behavior."""
    from .kernels import (
        GDNRecoverSSMCommitContext,
        ReplayRecordBuffers,
        gdn_recoverssm_verify,
    )

    if dtypes is None:
        dtypes = (torch.float32, getattr(torch, "bfloat16", torch.float32))
    key_heads = (
        num_key_heads
        if num_key_heads is not None
        else (num_heads // 3 if num_heads % 3 == 0 else num_heads)
    )
    if key_heads <= 0 or num_heads % key_heads:
        raise ValueError("selftest GQA key heads must divide value heads")
    torch.random.default_generator.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed(seed)
    qlen = spec_query_len
    rows, blocks = 3, 4
    cases = passed = 0
    worst = {}
    for dtype in dtypes:
        # Exercise the strict FP32 key path as well as the serving activation
        # dtype, even when the state cache and the activations differ.
        input_dtype = (
            torch.float32
            if dtype == torch.float32
            else (key_dtype if key_dtype is not None else dtype)
        )
        checkpoints = [
            (torch.randn(blocks, num_heads, head_v, head_k, device=device) * 0.1).to(
                dtype
            )
            for _ in range(num_layers)
        ]
        conv_states = [
            torch.zeros(blocks, 4, qlen + 2, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        records = [
            ReplayRecordBuffers.allocate(
                rows,
                num_heads,
                key_heads,
                qlen,
                head_v,
                head_k,
                device=device,
                key_dtype=input_dtype,
            )
            for _ in range(num_layers)
        ]
        context = GDNRecoverSSMCommitContext.from_tensors(
            conv_states,
            checkpoints,
            records,
            spec_query_len=qlen,
            max_num_reqs=rows,
            conv_dim_first=True,
        )
        # fold=0 is a real zero-commit case; fold=Q is full MTP acceptance.
        acceptance_cases = [(fold, max(0, qlen - fold)) for fold in range(qlen + 1)] + [
            (0, 0)
        ]
        for fold, other_fold in acceptance_cases:
            padding = fold == 0
            total = 2 * qlen if padding else rows * qlen
            qsl = torch.tensor(
                [0, qlen, 2 * qlen, total], dtype=torch.int32, device=device
            )
            indices = torch.tensor([1, 2, 0], dtype=torch.int32, device=device)
            mapping = torch.tensor([1, 0, 2], dtype=torch.int64, device=device)
            accepted = torch.tensor(
                [other_fold, fold, 0], dtype=torch.int32, device=device
            )
            expected, snapshots = [], []
            outputs, output_refs = [], []
            for layer in range(num_layers):
                checkpoints[layer].copy_(
                    (
                        torch.randn(
                            checkpoints[layer].shape, dtype=torch.float32, device=device
                        )
                        * 0.1
                    ).to(dtype)
                )
                snapshots.append(checkpoints[layer].clone())
                raw_q = torch.randn(total, key_heads, head_k, device=device)
                raw_k = torch.randn(total, key_heads, head_k, device=device)
                q = (raw_q / (raw_q.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(
                    input_dtype
                )
                k = (raw_k / (raw_k.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(
                    input_dtype
                )
                v = torch.randn(total, num_heads, head_v, device=device).to(input_dtype)
                g = -torch.rand(total, num_heads, dtype=torch.float32, device=device)
                beta = torch.rand(total, num_heads, device=device).to(input_dtype)
                out = torch.full(
                    (1, total, num_heads, head_v),
                    float("nan"),
                    dtype=input_dtype,
                    device=device,
                )
                for plane in (
                    records[layer].correction,
                    records[layer].key,
                    records[layer].decay,
                ):
                    plane.fill_(float("nan"))
                want_out = torch.zeros(
                    total, num_heads, head_v, dtype=torch.float64, device="cpu"
                )
                want_states = []
                for row, count in enumerate((fold, other_fold)):
                    start = row * qlen
                    reference_out, prefixes = _sequential_outputs(
                        snapshots[layer][row + 1],
                        q[start : start + qlen],
                        k[start : start + qlen],
                        v[start : start + qlen],
                        g[start : start + qlen],
                        beta[start : start + qlen],
                    )
                    want_out[start : start + qlen] = reference_out
                    want_states.append(prefixes[count])
                gdn_recoverssm_verify(
                    g,
                    beta,
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    checkpoint_state=checkpoints[layer],
                    replay_cache=records[layer],
                    query_start_loc=qsl,
                    state_indices=indices,
                    spec_query_len=qlen,
                    qk_already_normalized=True,
                    out=out,
                )
                # The unique key plane must preserve the actual normalized input.
                copied = records[layer].key[:2].detach().cpu()
                exact_keys = (
                    k[: 2 * qlen]
                    .reshape(2, qlen, key_heads, head_k)
                    .permute(0, 2, 1, 3)
                    .cpu()
                )
                if not torch.equal(copied, exact_keys):
                    raise RuntimeError(
                        "GDN selftest: key record changed normalized input bits"
                    )
                expected.append(want_states)
                outputs.append(out.squeeze(0))
                output_refs.append(want_out)
            context.commit(
                accepted, indices, qsl, request_indices=mapping, commit_conv=False
            )
            max_rel = 0.0
            ok = True
            tol = TOL_FP32 if dtype == torch.float32 else TOL_BF16
            for layer in range(num_layers):
                for row in range(2):
                    _, rel, cosine = _metrics(
                        checkpoints[layer][row + 1], expected[layer][row]
                    )
                    max_rel = max(max_rel, rel)
                    ok &= rel <= tol and cosine >= 0.999
                _, rel, cosine = _metrics(outputs[layer], output_refs[layer])
                max_rel = max(max_rel, rel)
                ok &= rel <= tol and cosine >= 0.999
                ok &= torch.equal(checkpoints[layer][0], snapshots[layer][0])
                ok &= torch.equal(checkpoints[layer][3], snapshots[layer][3])
                ok &= bool(torch.isnan(records[layer].correction[2]).all().item())
                ok &= bool(torch.isnan(records[layer].key[2]).all().item())
                ok &= bool(torch.isnan(records[layer].decay[2]).all().item())
            cases += 1
            passed += int(ok)
            name = str(dtype).split(".")[-1]
            worst[name] = max(worst.get(name, 0.0), max_rel)
            logger.info(
                "Ascend GDN RecoverSSM selftest: %s q=%d fold=%d layers=%d "
                "heads=%d key_heads=%d output+all-layer-state+null+padding "
                "max_rel=%.3e %s",
                name,
                qlen,
                fold,
                num_layers,
                num_heads,
                key_heads,
                max_rel,
                "PASS" if ok else "FAIL",
            )
        align_ok = _run_align(
            context, checkpoints, records, dtype, key_heads, head_k, head_v, device
        )
        cases += 1
        passed += int(align_ok)
    logger.info(
        "Ascend GDN RecoverSSM selftest SUMMARY: cases=%d pass=%d fail=%d "
        "worst_max_rel=%s tolerance fp32=%.1e bf16=%.1e %s",
        cases,
        passed,
        cases - passed,
        {k: f"{v:.3e}" for k, v in worst.items()},
        TOL_FP32,
        TOL_BF16,
        "OK" if passed == cases else "MISMATCH",
    )
    return passed == cases


def _run_align(context, checkpoints, records, dtype, key_heads, head_k, head_v, device):
    """One crossing window checks both boundary/final states in every layer."""
    from .kernels import gdn_recoverssm_verify

    window = context.spec_query_len
    if window < 2:
        return True
    fold = window
    boundary_len = max(1, window - 1)
    block_size = max(128, window)
    computed = torch.tensor(
        [block_size - boundary_len], dtype=torch.int32, device=device
    )
    qsl = torch.tensor([0, window], dtype=torch.int32, device=device)
    indices = torch.tensor([1], dtype=torch.int32, device=device)
    accepted = torch.tensor([fold], dtype=torch.int32, device=device)
    mapping = torch.zeros(1, dtype=torch.int64, device=device)
    table = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
    expected = []
    for layer in range(len(checkpoints)):
        raw = torch.randn(window, key_heads, head_k, device=device)
        input_dtype = records[layer].key_dtype
        k = (raw / (raw.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(input_dtype)
        q = torch.flip(k, dims=(-1,))
        hv = checkpoints[layer].shape[1]
        v = torch.randn(window, hv, head_v, device=device).to(input_dtype)
        g = -torch.rand(window, hv, device=device)
        beta = torch.rand(window, hv, device=device).to(input_dtype)
        _, prefixes = _sequential_outputs(checkpoints[layer][1], q, k, v, g, beta)
        expected.append((prefixes[boundary_len], prefixes[fold]))
        gdn_recoverssm_verify(
            g,
            beta,
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            checkpoint_state=checkpoints[layer],
            replay_cache=records[layer],
            query_start_loc=qsl,
            state_indices=indices,
            spec_query_len=window,
        )
    context.commit(
        accepted,
        indices,
        qsl,
        request_indices=mapping,
        block_table=table,
        num_computed_tokens=computed,
        mamba_block_size=block_size,
        commit_conv=False,
    )
    tolerance = TOL_FP32 if dtype == torch.float32 else TOL_BF16
    ok = True
    for layer in range(len(checkpoints)):
        for page, want in zip((2, 3), expected[layer]):
            _, rel, cosine = _metrics(checkpoints[layer][page], want)
            ok &= rel <= tolerance and cosine >= 0.999
    logger.info(
        "Ascend GDN RecoverSSM selftest ALIGN: layers=%d q=%d boundary=%d %s",
        len(checkpoints),
        window,
        boundary_len,
        "PASS" if ok else "FAIL",
    )
    return ok


@_preserve_rng
def run_conv(
    *,
    num_layers: int,
    num_heads: int,
    head_k: int,
    head_v: int,
    spec_query_len: int,
    device: torch.device,
    dtypes=None,
    seed: int = 0,
) -> bool:
    """Check the convolution-history compaction against its definition.

    The layer's conv update leaves one row holding ``[history | drafts]``:
    ``conv_kernel_size - 1`` committed inputs followed by the ``num_spec`` draft
    inputs. Committing ``A`` accepted drafts must keep the window that ends at the
    committed token:

        new[h] = old[A + h]     for h in [0, history)

    which is "compact the convolution history after sampling" (RFC #19987 s1/s2)
    and is what the commit kernel does with ``commit_len - 1 = accepted``.
    """
    from .kernels import GDNRecoverSSMCommitContext

    if dtypes is None:
        dtypes = (torch.float32, getattr(torch, "bfloat16", torch.float32))
    torch.random.default_generator.manual_seed(seed + 17)
    if device.type == "npu":
        torch.npu.manual_seed(seed + 17)

    blocks = 2
    num_spec = spec_query_len - 1
    history = 3  # conv_kernel_size - 1 for the Qwen3.8 kernel size of 4
    conv_len = history + num_spec
    conv_dim = 16 * num_heads  # multiple of 16 keeps the tile path exact
    cases = passed = 0
    worst = 0.0

    for dtype in dtypes:
        conv = [
            torch.randn(blocks, conv_dim, conv_len, device=device).to(dtype)
            for _ in range(num_layers)
        ]
        zero_state = [
            torch.zeros(blocks, num_heads, head_v, head_k, device=device).to(dtype)
            for _ in range(num_layers)
        ]
        records = [
            torch.zeros(
                blocks,
                num_heads,
                spec_query_len,
                head_v + head_k + 1,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(num_layers)
        ]
        context = GDNRecoverSSMCommitContext.from_tensors(
            conv,
            zero_state,
            records,
            spec_query_len=spec_query_len,
            max_num_reqs=blocks,
            conv_dim_first=True,
        )
        block = 1
        for accepted in range(spec_query_len):
            # Fresh history per case so the cases cannot influence each other.
            conv[0][block].copy_(
                torch.randn(conv_dim, conv_len, device=device).to(dtype)
            )
            before = conv[0][block].clone()
            context.commit(
                torch.tensor([accepted + 1], dtype=torch.int32, device=device),
                torch.full((1,), block, dtype=torch.int32, device=device),
                torch.tensor([0, spec_query_len], dtype=torch.int32, device=device),
                request_indices=torch.zeros(1, dtype=torch.int32, device=device),
                commit_conv=True,
            )
            got = context.conv_states[0][block][:, :history]
            want = before[:, accepted : accepted + history]
            max_abs, max_rel, _cos = _metrics(got, want)
            tol = TOL_FP32 if dtype == torch.float32 else TOL_BF16
            ok = max_rel <= tol
            cases += 1
            passed += int(ok)
            worst = max(worst, max_rel)
            logger.info(
                "Ascend GDN RecoverSSM selftest: conv %s accepted=%d history=%d "
                "max_abs=%.3e max_rel=%.3e %s",
                str(dtype).split(".")[-1],
                accepted,
                history,
                max_abs,
                max_rel,
                "PASS" if ok else "FAIL",
            )
    logger.info(
        "Ascend GDN RecoverSSM selftest CONV SUMMARY: cases=%d pass=%d fail=%d "
        "worst_max_rel=%.3e tolerance fp32=%.1e bf16=%.1e %s",
        cases,
        passed,
        cases - passed,
        worst,
        TOL_FP32,
        TOL_BF16,
        "OK" if passed == cases else "MISMATCH",
    )
    return passed == cases


__all__ = ["enabled", "run", "sequential_state"]
