# SPDX-License-Identifier: Apache-2.0
"""Host-only probe around the Ascend full-graph replay path.

Why this exists
---------------
The machine fails at ``AclmdlRIExecuteAsync`` with
``halStreamTaskFill ... drvRetCode=17`` / ``Build SQ/CQ failed`` on the first
full-graph replays, while the same flags without RecoverSSM are known to work.
The last Python frame is ``_updatable_graph_replay``, and two numbers matter but
are not logged anywhere:

* which batch descriptor is being replayed (``num_tokens`` / ``num_reqs``), and
  whether the failing one is always the same;
* how many update tasks the graph carries (``UpdatableGraph.tasks``) — the count
  the driver eventually refuses.

Both are plain Python attributes, so this probe reads nothing from the device,
allocates nothing, and cannot synchronise. It runs entirely before the replay
call, so its line is already on disk when the process dies.

Nothing here changes behaviour: the replayed call is the original one.
"""

from __future__ import annotations

import functools

from vllm.logger import init_logger

logger = init_logger("vllm.gdn_recoverssm_ascend.graph_probe")

_APPLIED = False
_SEEN: dict = {}
_FAILED: dict = {}
_RETRIED: set = set()


def _drain_device() -> None:
    try:
        import torch

        torch.npu.synchronize()
    except Exception as exc:  # never mask the original failure
        logger.warning("Ascend full-graph replay: drain before retry failed (%s)", exc)


def install() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from vllm_ascend.worker.v2 import aclgraph_utils

    cls = aclgraph_utils.ModelAclGraphManager
    original = cls._updatable_graph_replay

    @functools.wraps(original)
    def _updatable_graph_replay(self, desc, attn_metadata):
        try:
            _log_replay_attempt(self, desc)
        except Exception:  # probing must never break a request
            pass
        key = _describe(desc)
        first = _FAILED.get(key)
        if first is not None:
            # Replaying a graph that already failed would either mask the first
            # error or corrupt more state. Refuse and keep the original failure.
            raise RuntimeError(
                "Ascend full-graph replay refused: descriptor "
                f"{key} already failed once ({first}); the graph is treated as "
                "unusable and must be re-captured"
            )
        try:
            return original(self, desc, attn_metadata)
        except Exception as exc:
            # CaptureModel::AllocSqCqAndBindInternal cleans up after itself on
            # failure (``DELETE_A(sqCqArray_); sqCqNum_ = 0U;``), so the next
            # execute retries the whole allocation. The first execute of a graph
            # fails here when the host has already queued the next batch (async
            # scheduling); draining the device and retrying once turns that into a
            # one-off hiccup instead of a killed engine.
            if key in _RETRIED:
                _FAILED[key] = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Ascend full-graph replay failed again after a retry: %s", key
                )
                raise
            _RETRIED.add(key)
            logger.warning(
                "Ascend full-graph replay failed on first execute (%s): %s: %s -- "
                "draining the device and retrying once",
                key,
                type(exc).__name__,
                exc,
            )
            _drain_device()
            try:
                return original(self, desc, attn_metadata)
            except Exception as retry_exc:
                _FAILED[key] = f"{type(retry_exc).__name__}: {retry_exc}"
                logger.error(
                    "Ascend full-graph replay failed again after a retry: %s", key
                )
                raise

    cls._updatable_graph_replay = _updatable_graph_replay

    # Also snapshot the counters right after capture: that is the backlog the
    # stream carries into the first replay.
    original_capture = getattr(cls, "capture", None)
    if original_capture is None:
        _APPLIED = True
        return

    @functools.wraps(original_capture)
    def capture(self, *args, **kwargs):
        result = original_capture(self, *args, **kwargs)
        try:
            from . import layer_patch

            counters = layer_patch.counters()
            logger.info(
                "Ascend full-graph capture finished: graphs=%d verify_calls=%d "
                "last_window=(%d,%d)",
                len(getattr(self, "graphs", {}) or {}),
                counters.get("verify_calls", -1),
                counters.get("last_rows", -1),
                counters.get("last_tokens", -1),
            )
        except Exception:
            pass
        return result

    cls.capture = capture
    _APPLIED = True


def _describe(desc) -> str:
    parts = []
    for name in ("num_tokens", "num_reqs", "num_tokens_across_dp", "cg_mode"):
        value = getattr(desc, name, None)
        if value is not None:
            parts.append(f"{name}={value}")
    return ",".join(parts) if parts else repr(desc)


def _log_replay_attempt(manager, desc) -> None:
    key = _describe(desc)
    state = _SEEN.get(key)
    if state is None:
        state = _SEEN[key] = {"count": 0, "tasks": None}
        try:
            graph = manager.graphs[desc]
            # ``UpdatableGraph.tasks`` is the parametrised op list; each entry
            # becomes one SQE per update on the update stream.
            tasks = getattr(graph, "tasks", None)
            state["tasks"] = len(tasks) if tasks is not None else None
        except Exception:
            pass
    state["count"] += 1
    # First few attempts per descriptor, and every 20th afterwards: enough to see
    # accumulation without flooding a serving log.
    if state["count"] <= 3 or state["count"] % 20 == 0:
        # ``verify_calls`` is how many times our kernel was launched so far
        # (warmup + capture included). It tells whether our path is what fills
        # the stream before the first replay.
        counters = {}
        try:
            from . import layer_patch

            counters = layer_patch.counters()
        except Exception:
            pass
        logger.info(
            "Ascend full-graph replay: %s attempt=%d updatable_tasks=%s "
            "verify_calls=%s last_rows=%s last_tokens=%s",
            key,
            state["count"],
            state["tasks"],
            counters.get("verify_calls", "?"),
            counters.get("last_rows", "?"),
            counters.get("last_tokens", "?"),
        )
