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
        return original(self, desc, attn_metadata)

    cls._updatable_graph_replay = _updatable_graph_replay
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
        logger.info(
            "Ascend full-graph replay: %s attempt=%d updatable_tasks=%s",
            key,
            state["count"],
            state["tasks"],
        )
