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

Both start from the same checkpoint. ``m = 0`` is the "first draft rejected"
case: the checkpoint must come back unchanged.

Reported per case: ``max_abs``, ``max_rel`` and ``cos`` over every (head, v, k),
plus a one-line verdict. A summary line closes the block so a single grep is
enough.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.gdn_recoverssm_ascend.selftest")

# fp32 kernels: the reference is fp64, so the floor is the kernel's own rounding.
TOL_FP32 = 1e-5
# bf16 state pages round the checkpoint on every step; the record itself is fp32.
TOL_BF16 = 2e-2

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
    state = checkpoint.to(torch.float64)
    out = state
    for t in range(k.shape[0]):
        k_t = k[t].to(torch.float64)          # [HV, K]
        v_t = v[t].to(torch.float64)          # [HV, V]
        a = g[t].to(torch.float64).exp()      # [HV]
        b = beta[t].to(torch.float64)         # [HV]
        prev = out
        # (S k) is a vector per head, shape [HV, V]
        sk = torch.einsum("hvk,hk->hv", prev, k_t)
        out = a[:, None, None] * prev + b[:, None, None] * (
            v_t - a[:, None] * sk
        )[:, :, None] * k_t[:, None, :]
        del state
        state = out
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _metrics(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float, float]:
    got = got.to(torch.float64)
    want = want.to(torch.float64)
    diff = (got - want).abs()
    scale = want.abs().clamp_min(1e-6)
    max_abs = float(diff.max()) if diff.numel() else 0.0
    max_rel = float((diff / scale).max()) if diff.numel() else 0.0
    flat_got = got.reshape(-1)
    flat_want = want.reshape(-1)
    denom = float(flat_got.norm()) * float(flat_want.norm())
    cos = (
        float(torch.dot(flat_got, flat_want)) / denom if denom > 0 else 1.0
    )
    return max_abs, max_rel, cos


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
) -> bool:
    """Run the whole matrix and log one block. Returns True if every case passed."""
    from .kernels import GDNRecoverSSMCommitContext

    if dtypes is None:
        dtypes = (torch.float32, getattr(torch, "bfloat16", torch.float32))

    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)

    width = head_v + head_k + 1
    blocks = 2
    cases = 0
    passed = 0
    worst = {}

    for dtype in dtypes:
        conv_states = [
            torch.randn(blocks, 4, 4, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        checkpoints = [
            torch.randn(blocks, num_heads, head_v, head_k, dtype=dtype, device=device)
            * 0.1
            for _ in range(num_layers)
        ]
        records = [
            torch.zeros(
                blocks, num_heads, spec_query_len, width,
                dtype=torch.float32, device=device,
            )
            for _ in range(num_layers)
        ]
        context = GDNRecoverSSMCommitContext.from_tensors(
            conv_states,
            checkpoints,
            records,
            spec_query_len=spec_query_len,
            max_num_reqs=blocks,
            conv_dim_first=True,
        )

        for fold in range(1, spec_query_len + 1):
            # One fresh window per case, and a fresh checkpoint for each dtype run
            # so cases cannot leak state into each other.
            for i in range(num_layers):
                checkpoints[i].normal_(0.0, 0.1)
            kb = torch.randn(spec_query_len, num_heads, head_k, dtype=dtype, device=device)
            kb = kb / kb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            vb = torch.randn(spec_query_len, num_heads, head_v, dtype=dtype, device=device)
            gb = -torch.rand(spec_query_len, num_heads, dtype=dtype, device=device)
            bb = torch.rand(spec_query_len, num_heads, dtype=dtype, device=device)

            start = checkpoints[0][0].clone()
            want = sequential_state(
                start, kb[:fold], vb[:fold], gb[:fold], bb[:fold]
            )
            got = _recover(context, kb, vb, gb, bb, fold)
            max_abs, max_rel, cos = _metrics(got, want)
            tol = TOL_FP32 if dtype == torch.float32 else TOL_BF16
            ok = max_rel <= tol and cos >= 0.999
            cases += 1
            passed += int(ok)
            worst[str(dtype).split(".")[-1]] = max(
                worst.get(str(dtype).split(".")[-1], 0.0), max_rel
            )
            logger.info(
                "Ascend GDN RecoverSSM selftest: %s q=%d accepted=%d (fold=%d) "
                "layers=%d heads=%d max_abs=%.3e max_rel=%.3e cos=%.6f %s",
                str(dtype).split(".")[-1],
                spec_query_len,
                fold - 1,
                fold,
                num_layers,
                num_heads,
                max_abs,
                max_rel,
                cos,
                "PASS" if ok else "FAIL",
            )

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


def _recover(context, kb, vb, gb, bb, fold: int) -> torch.Tensor:
    """Run verify for the whole window, then commit ``fold`` tokens, in place.

    Uses the deployed launchers, the deployed page layout and the deployed record
    layout: the only thing synthetic is the state and the tokens.
    """
    from .kernels import gdn_recoverssm_verify

    num_heads = kb.shape[1]
    head_k = kb.shape[2]
    head_v = vb.shape[2]
    window = context.spec_query_len
    tokens = kb.shape[0]

    q = kb.unsqueeze(0)                       # [1, T, H, K]
    k = kb.unsqueeze(0)
    v = vb.unsqueeze(0)                       # [1, T, HV, V]
    g = gb.unsqueeze(0)                       # [1, T, HV]
    beta = bb.unsqueeze(0)
    out = torch.zeros(1, tokens, num_heads, head_v, dtype=v.dtype, device=v.device)

    state_indices = torch.zeros(1, dtype=torch.int32, device=v.device)
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device=v.device)

    records = context.records[0]
    # The window starts at token ``tokens - window``; the reference checkpoint is
    # the pre-verify state, so verify all ``window`` tokens from index 0.
    gdn_recoverssm_verify(
        g,
        beta,
        q,
        k,
        v,
        checkpoint_state=context.checkpoints[0],
        replay_cache=records,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        spec_query_len=window,
        qk_already_normalized=True,
        out=out,
    )

    accepted = torch.tensor([fold], dtype=torch.int32, device=v.device)
    request_indices = torch.zeros(1, dtype=torch.int32, device=v.device)
    context.commit(
        accepted,
        state_indices,
        query_start_loc,
        request_indices=request_indices,
        commit_conv=False,
    )
    # none mode writes the folded state back into the source page.
    return context.checkpoints[0][0].clone()


__all__ = ["enabled", "run", "sequential_state"]
