# SPDX-License-Identifier: Apache-2.0
"""Bound the existing Ascend sampling filter without changing its distribution.

The rows are independent until rejection sampling. Splitting this deterministic
filter does not split requests or change RNG counters. In particular we retain
the deployed sort/threshold semantics (including ties and mixed top-k/top-p),
rather than substituting a different fused sampler.
"""

from __future__ import annotations

import importlib
import math

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.gdn_recoverssm_ascend.sampling")

# Internal workspace accounting, not a new user option. The explicit live
# tensors are bounded; CANN operator-private workspace is additionally protected
# by the existing 512 MiB reservation. This is not a promise about allocator RSS.
FILTER_WORKING_BYTES = 128 << 20
ESTIMATED_BYTES_PER_ELEMENT = 32
_SORT_VALUES = None
_INSTALLED = False
_PREPARED = False
_FIRST_CALL = True


def chunk_rows(vocab_size: int, rows: int) -> int:
    if vocab_size <= 0 or rows < 0:
        raise ValueError("invalid sampling shape")
    return max(
        1,
        min(
            max(rows, 1),
            FILTER_WORKING_BYTES // (vocab_size * ESTIMATED_BYTES_PER_ELEMENT),
        ),
    )


def filter_logits(logits, k, p, *, sort_values=None):
    """Apply exactly the rc1 probability-threshold filter, in bounded row tiles."""
    if k is None and p is None:
        return logits
    rows, vocab = logits.shape
    limit = chunk_rows(vocab, rows)
    for start in range(0, rows, limit):
        end = min(start + limit, rows)
        target = logits[start:end]
        probs = target.softmax(dim=-1)
        if sort_values is None or probs.dtype != torch.float32:
            # The fallback still materializes indices, but only for this tile.
            # Selecting [0] releases them immediately after the call.
            sorted_probs = probs.sort(dim=-1, descending=False)[0]
        else:
            sorted_probs = sort_values(probs, dim=-1, descending=False)
        if k is not None:
            counts = (vocab - k[start:end].to(torch.long)).unsqueeze(1)
            cutoff = sorted_probs.gather(-1, counts)
            cutoff.masked_fill_((k[start:end] == vocab).unsqueeze(1), -math.inf)
            target.masked_fill_(probs < cutoff, -math.inf)
        if p is not None:
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            discard = cumulative <= 1 - p[start:end].unsqueeze(1)
            discard[:, -1] = False
            cutoff = sorted_probs.gather(-1, discard.sum(dim=-1).unsqueeze(1))
            target.masked_fill_(probs < cutoff, -math.inf)
        # Do not keep the previous tile's buffers alive while allocating the next.
        del target, probs, sorted_probs
        if k is not None or p is not None:
            del cutoff
        if p is not None:
            del cumulative, discard
    return logits


def _enabled():
    from .state_patch import is_enabled

    return is_enabled()


def _adapter(original):
    def apply(logits, k, p):
        global _FIRST_CALL
        if not _enabled():
            return original(logits, k, p)
        if _FIRST_CALL and (k is not None or p is not None):
            _FIRST_CALL = False
            logger.info(
                "GDN_SAMPLER active rows=%d vocab=%d row_tile=%d "
                "sort=%s workspace_model_bytes=%d semantics=rc1-threshold",
                logits.shape[0],
                logits.shape[1],
                chunk_rows(logits.shape[1], logits.shape[0]),
                "values-only" if _SORT_VALUES is not None else "bounded-sort",
                FILTER_WORKING_BYTES,
            )
        return filter_logits(logits, k, p, sort_values=_SORT_VALUES)

    return apply


def prepare(worker):
    """Install aliases and validate the optional values-only primitive at startup."""
    global _INSTALLED, _SORT_VALUES, _PREPARED
    if not _enabled() or _PREPARED:
        return
    base = importlib.import_module("vllm_ascend.sample.sampler")
    v2 = importlib.import_module("vllm_ascend.worker.v2.sample.apply_top_k_top_p")
    original = base._apply_top_k_top_p_pytorch
    if not _INSTALLED:
        wrapped = _adapter(original)
        base._apply_top_k_top_p_pytorch = wrapped
        # v2 imported the helper by value; changing only base misses live calls.
        v2._apply_top_k_top_p_pytorch = wrapped
        _INSTALLED = True
    import torch_npu

    values_sort = getattr(torch_npu, "npu_sort_v2", None)
    reason = "API unavailable"
    if values_sort is not None:
        # An inexpensive startup-only correctness probe; no request-path sync.
        device = getattr(worker, "device", "npu")
        probe = torch.tensor(
            [[0.0, 1.0, 1.0, -1.0, -math.inf, math.inf]],
            dtype=torch.float32,
            device=device,
        )
        try:
            got = values_sort(probe, dim=-1, descending=False)
            want = probe.sort(dim=-1, descending=False)[0]
            if torch.equal(got, want):
                _SORT_VALUES = values_sort
                reason = "startup probe PASS"
            else:
                reason = "startup probe mismatch; retaining bounded native sort"
        except (NotImplementedError, TypeError, AttributeError) as exc:
            # A missing primitive must not corrupt state or trigger live retries.
            reason = (
                f"unsupported primitive ({type(exc).__name__}); bounded native sort"
            )
        del probe
    _PREPARED = True
    logger.info(
        "GDN_SAMPLER ready sort=%s working_bytes=%d sampling_params=unchanged "
        "reason=%s",
        "values-only" if _SORT_VALUES else "bounded-sort",
        FILTER_WORKING_BYTES,
        reason,
    )
