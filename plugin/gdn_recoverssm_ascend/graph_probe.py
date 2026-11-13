# SPDX-License-Identifier: Apache-2.0
"""Host-only graph observability and startup first-execution preparation.

Capture metadata is retained only while capture is in progress, used once to
warm each graph through the native update path, then released.  Serving never
retries a failed graph: an arbitrary replay exception may follow partial state
writes and cannot safely be treated as a resource-allocation-only failure.
"""

from __future__ import annotations

import functools
import os

from vllm.logger import init_logger

logger = init_logger("vllm.gdn_recoverssm_ascend.graph_probe")

_APPLIED = False
_ENABLED = False
_COLLECTING = False
_CAPTURE_META: dict = {}
_SEEN: dict = {}
_FAILED: dict = {}
_METRICS = {"replays": 0, "capture_graphs": 0, "warmed_graphs": 0}


def enable() -> None:
    global _ENABLED
    _ENABLED = True


def metrics() -> dict:
    """Return CPU counters; this function never reads a device tensor."""
    return dict(_METRICS)


def remember_metadata(state, args, kwargs, attn_metadata) -> None:
    """Keep startup capture metadata, never runtime eager metadata."""
    if not _COLLECTING:
        return
    batch = args[0] if args else kwargs.get("input_batch")
    if batch is None or attn_metadata is None:
        return
    tokens = getattr(batch, "num_tokens", None)
    reqs = getattr(batch, "num_reqs", None)
    if tokens is not None and reqs is not None:
        _CAPTURE_META[(int(tokens), int(reqs))] = attn_metadata


def _key(desc) -> tuple:
    # The descriptor fields are host values, not tensor contents.
    return (
        getattr(desc, "num_tokens", None),
        getattr(desc, "num_reqs", None),
        str(getattr(desc, "cg_mode", None)),
    )


def install() -> None:
    global _APPLIED
    if _APPLIED:
        return
    if os.getenv("GDN_RECOVERSSM_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        return
    from vllm_ascend.worker.v2 import aclgraph_utils

    cls = aclgraph_utils.ModelAclGraphManager
    original = cls._updatable_graph_replay

    @functools.wraps(original)
    def _updatable_graph_replay(self, desc, attn_metadata):
        if not _ENABLED:
            return original(self, desc, attn_metadata)
        key = _key(desc)
        if key in _FAILED:
            raise RuntimeError(
                "RecoverSSM graph is unusable after an earlier replay failure: "
                f"{_describe(desc)}; original={_FAILED[key]}"
            )
        _log_replay_attempt(self, desc, attn_metadata)
        try:
            result = original(self, desc, attn_metadata)
            from . import diagnostics

            diagnostics.observe_graph(desc, attn_metadata)
            return result
        except Exception as exc:
            _FAILED[key] = f"{type(exc).__name__}: {exc}"
            logger.error(
                "RecoverSSM graph replay failed: %s; no automatic replay of "
                "possibly partial state writes is attempted",
                _describe(desc),
            )
            raise

    cls._updatable_graph_replay = _updatable_graph_replay
    original_capture = getattr(cls, "capture", None)
    if original_capture is not None:

        @functools.wraps(original_capture)
        def capture(self, *args, **kwargs):
            global _COLLECTING
            if not _ENABLED:
                return original_capture(self, *args, **kwargs)
            _CAPTURE_META.clear()
            _COLLECTING = True
            try:
                result = original_capture(self, *args, **kwargs)
                _COLLECTING = False
                _METRICS["capture_graphs"] = len(getattr(self, "graphs", {}) or {})
                _warm_captured_graphs(self)
                return result
            finally:
                _COLLECTING = False
                # Metadata can retain entire profiling caches and context owners.
                # Native graphs retain their own required inputs.
                _CAPTURE_META.clear()

        cls.capture = capture
    _APPLIED = True


def _warm_captured_graphs(manager) -> None:
    """Run each captured graph once before readiness via native run_fullgraph.

    This preserves the proven startup workaround for lazy first-execution
    queue setup.  A raw graph.replay bypasses update/external-event resolution,
    so it must not be used here.  A failure stops startup rather than deferring
    an unprepared graph or retrying partially executed work during serving.
    """
    graphs = getattr(manager, "graphs", None)
    model_state = getattr(getattr(manager, "model_runner", None), "model_state", None)
    if not graphs:
        return
    if model_state is None:
        raise RuntimeError("RecoverSSM graph warmup needs the model-state owner")
    missing = [
        desc
        for desc in graphs
        if (int(getattr(desc, "num_tokens", -1)), int(getattr(desc, "num_reqs", -1)))
        not in _CAPTURE_META
    ]
    if missing:
        raise RuntimeError(
            "RecoverSSM capture metadata missing for "
            f"{len(missing)} graphs; refusing to defer first execution to serving"
        )
    done = 0
    for desc in graphs:
        key = (int(desc.num_tokens), int(desc.num_reqs))
        saved = getattr(model_state, "attn_metadata", None)
        model_state.attn_metadata = _CAPTURE_META[key]
        try:
            manager.run_fullgraph(desc)
        finally:
            model_state.attn_metadata = saved
        done += 1
    import torch

    # Startup only. Surface asynchronous warmup failures before server readiness.
    torch.npu.synchronize()
    _METRICS["warmed_graphs"] += done
    logger.info(
        "Ascend full-graph warmup: executed %d captured graphs before serving "
        "(0 without recorded metadata); runtime_retry=disabled",
        done,
    )


def _describe(desc) -> str:
    parts = []
    for name in ("num_tokens", "num_reqs", "num_tokens_across_dp", "cg_mode"):
        value = getattr(desc, name, None)
        if value is not None:
            parts.append(f"{name}={value}")
    return ", ".join(parts) if parts else repr(desc)


def _logical_rows(attn_metadata):
    if not isinstance(attn_metadata, dict):
        return None
    for meta in attn_metadata.values():
        if getattr(meta, "spec_sequence_masks", None) is not None:
            return int(getattr(meta, "num_spec_decodes", 0))
    return None


def _log_replay_attempt(manager, desc, attn_metadata=None) -> None:
    key = _key(desc)
    _METRICS["replays"] += 1
    state = _SEEN.get(key)
    if state is not None:
        state["count"] += 1
        return
    tasks = None
    graph = (getattr(manager, "graphs", {}) or {}).get(desc)
    if graph is not None:
        values = getattr(graph, "tasks", None)
        tasks = len(values) if values is not None else None
    _SEEN[key] = {"count": 1, "tasks": tasks}
    logical = _logical_rows(attn_metadata)
    capacity = getattr(desc, "num_reqs", None)
    padded = (
        None if logical is None or capacity is None else max(int(capacity) - logical, 0)
    )
    logger.info(
        "RecoverSSM graph dispatch: %s updatable_tasks=%s "
        "logical_spec_rows=%s graph_request_capacity=%s padded_rows=%s",
        _describe(desc),
        tasks,
        logical,
        capacity,
        padded,
    )
