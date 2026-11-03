# SPDX-License-Identifier: Apache-2.0
"""Runtime proof that RecoverSSM is actually doing the work.

One-shot INFO lines prove the *configuration* was accepted; they cannot prove
that every decode step runs verify and that the post-sampling commit really
writes the checkpoint back. This module keeps cheap per-process counters and
prints a periodic summary, plus a first-step banner that includes the concrete
tensor layout (the replay record's shape/dtype and the checkpoint page count).

Design rules:

* **No device sync on the hot path.** ``bump_commit`` accumulates the accepted
  token count into a 0-dim device tensor; the single ``.item()`` happens only
  when the periodic report fires.
* **No allocation on the hot path** after the first call (the accumulator is
  created once and reused).
* Everything is best-effort: telemetry must never break a request.

Environment knobs::

    GDN_RECOVERSSM_LOG_EVERY=200     # report every N commits (0 disables)
    GDN_RECOVERSSM_LOG_VERIFY=1      # log the first verify of each step at INFO
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_STATE: dict = {
    "banner_done": False,
    "verify_steps": 0,
    "verify_rows": 0,
    "verify_tokens": 0,
    "commit_steps": 0,
    "commit_rows": 0,
    "empty_commits": 0,
    "accepted_acc": None,  # 0-dim int64 tensor, device-side
    "accepted_total": 0,
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def report_every() -> int:
    return max(0, _env_int("GDN_RECOVERSSM_LOG_EVERY", 200))


def reset() -> None:
    """Test helper."""
    _STATE.update(
        banner_done=False,
        verify_steps=0,
        verify_rows=0,
        verify_tokens=0,
        commit_steps=0,
        commit_rows=0,
        empty_commits=0,
        accepted_acc=None,
        accepted_total=0,
    )


def banner_once(*, kv_cache, num_spec: int, enabled: bool) -> None:
    """First-layer banner: proves the record tensor was allocated."""
    if _STATE["banner_done"] or not enabled:
        return
    _STATE["banner_done"] = True
    shapes = [tuple(getattr(t, "shape", ())) for t in kv_cache]
    dtypes = [str(getattr(t, "dtype", "?")) for t in kv_cache]
    logger.info(
        "Ascend GDN RecoverSSM ACTIVE: spec_query_len=%d, mamba states per layer=%d "
        "(conv, checkpoint, replay), shapes=%s, dtypes=%s",
        1 + num_spec,
        len(shapes),
        shapes,
        dtypes,
    )


def first_verify(rows: int, spec_query_len: int, query_start_loc) -> None:
    """INFO on the first verify of the process; DEBUG afterwards."""
    if _STATE["verify_steps"] == 1:
        logger.info(
            "Ascend GDN RecoverSSM verify #1: %d spec rows, window=%d, "
            "query_start_loc[:4]=%s",
            rows,
            spec_query_len,
            _peek(query_start_loc, 4),
        )
    elif logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "RecoverSSM verify: rows=%d window=%d", rows, spec_query_len
        )


def bump_verify(rows: int, tokens: int, spec_query_len: int, query_start_loc) -> None:
    _STATE["verify_steps"] += 1
    _STATE["verify_rows"] += int(rows)
    _STATE["verify_tokens"] += int(tokens)
    if _STATE["verify_steps"] == 1:
        first_verify(rows, spec_query_len, query_start_loc)


def bump_commit(num_accepted, rows: int) -> None:
    """Accumulate accepted tokens on device; no sync until the report."""
    _STATE["commit_steps"] += 1
    _STATE["commit_rows"] += int(rows)
    if rows == 0:
        _STATE["empty_commits"] += 1
    try:
        import torch

        acc = _STATE["accepted_acc"]
        if acc is None or acc.device != num_accepted.device:
            acc = _STATE["accepted_acc"] = torch.zeros(
                (), dtype=torch.int64, device=num_accepted.device
            )
        acc.add_(num_accepted[:rows].sum().to(torch.int64))
    except Exception:  # telemetry is best-effort
        pass

    every = report_every()
    if every and _STATE["commit_steps"] % every == 0:
        report()


def report() -> None:
    accepted = _STATE["accepted_total"]
    try:
        acc = _STATE["accepted_acc"]
        if acc is not None:
            # The only device sync in this module, once per reporting interval.
            accepted = _STATE["accepted_total"] = int(acc.item())
    except Exception:
        pass
    commits = max(_STATE["commit_steps"], 1)
    mean_accepted = accepted / commits
    logger.info(
        "Ascend GDN RecoverSSM stats: verify_steps=%d commit_steps=%d "
        "spec_rows=%d commit_rows=%d empty_commits=%d "
        "accepted_tokens=%d mean_accepted=%.2f",
        _STATE["verify_steps"],
        _STATE["commit_steps"],
        _STATE["verify_rows"],
        _STATE["commit_rows"],
        _STATE["empty_commits"],
        accepted,
        mean_accepted,
    )
    if _STATE["verify_steps"] > 0 and _STATE["commit_steps"] == 0:
        logger.warning(
            "Ascend GDN RecoverSSM: verify ran %d times but commit never ran. "
            "The record_step/commit_step hook is not wired -- the checkpoint will "
            "never advance and output will silently drift.",
            _STATE["verify_steps"],
        )
    elif (
        _STATE["commit_steps"] > 0
        and _STATE["verify_steps"] > 0
        and _STATE["commit_steps"] < 0.9 * _STATE["verify_steps"]
    ):
        logger.warning(
            "Ascend GDN RecoverSSM: commit_steps(%d) < verify_steps(%d); some "
            "verify steps did not reach the commit hook.",
            _STATE["commit_steps"],
            _STATE["verify_steps"],
        )
    if _STATE["commit_steps"] > 0 and mean_accepted < 1.0:
        logger.warning(
            "Ascend GDN RecoverSSM: mean accepted tokens per commit is %.2f (< 1). "
            "Either the drafter is rejecting everything or the commit hook is not "
            "running -- check for '%s' in the log.",
            mean_accepted,
            "commit context",
        )


def _peek(tensor, n: int):
    try:
        return tensor[:n].tolist()
    except Exception:
        return "?"


__all__ = [
    "banner_once",
    "bump_commit",
    "bump_verify",
    "report",
    "report_every",
    "reset",
]
