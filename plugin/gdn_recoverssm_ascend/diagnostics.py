# SPDX-License-Identifier: Apache-2.0
"""Server-side evidence, consuming the runner's existing completed CPU outputs.

There are no device scalar reads, device events or device synchronizations here.
Host-submit timings must never be presented as device execution timings.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import json
import os
import threading
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vllm.logger import init_logger

from .metrics_math import padding_fraction

logger = init_logger("vllm.gdn_recoverssm_ascend.perf")
PATCH_ID = "ascendc-op-dev-v2-clean-evidence"
REPORT_STEPS = 200
TIMING_SAMPLE_BLOCK = 64
TIMING_FIRST_STEPS = 3
_CONTEXT_BUCKETS = (0, 1024, 4096, 8192, 16384, 32768, 65536)
_CONTEXT_LABELS = (*map(str, _CONTEXT_BUCKETS), ">65536")
_TIMING_SCOPES = {
    "prepare_attn_host": "native metadata and recovery step preparation on the worker host",
    "mixed_preflight_host": "mixed layer eligibility checks on the worker host",
    "mixed_verify_submit_host": "mixed verify binding checks and submission on the worker host",
    "prefill_state_load_host": "native prefill state load call",
    "prefill_compute_submit_host": "native prefill compute call",
    "prefill_state_store_host": "native prefill state store call",
    "fullgraph_submit_host": "native graph update and replay submission call",
    "sampler_filter_host": "plugin top-k/top-p filter call",
    "output_get_wall": "native output get, including its existing copy wait and serialization",
    "grouped_commit_host": "grouped recovery preparation and submission on the worker host",
}
_INSTALLED = False
_READY = False
_RANK = 0
_LATEST = None
_LOCK = threading.Lock()
_REPORT_LOCK = threading.Lock()
_COUNTS = Counter()
_GRAPHS = Counter()
_BATCHES = Counter()
_PHASES = defaultdict(Counter)
_PHASE_ACCEPTED = defaultdict(Counter)
_PHASE_CONTEXT = defaultdict(Counter)
_PHASE_GRAPHS = defaultdict(Counter)
_PHASE_COMMITS = defaultdict(Counter)
_HOST_TIMINGS = {}
_PREFILL_DISPATCH = {}
_SUBMITTED_STEPS = 0
_START = None
_FIRST_SUBMIT_UNIX = None
_LAST_COMPLETED_UNIX = None
_LAST_CONFIG = None
_SOURCE_FINGERPRINT = None
_IDLE_THREAD = None
_LAST_OUTPUT_AT = None
_LAST_REPORTED_STEPS = 0
_LAST_REPORTED_COMMITS = 0
_PATHS_AT_READY = {}
_STOP_REPORTING = threading.Event()


@dataclass(frozen=True)
class StepSnapshot:
    spec_rows: tuple | None
    num_tokens: int
    num_reqs: int
    phase: str = "unknown"
    context_sum: int = 0
    context_max: int = 0
    context_rows: int = 0
    prefill_rows: int | None = None
    prefill_tokens: int | None = None
    spec_tokens: int | None = None
    attention_work: int | None = None
    prefill_attention_work: int | None = None
    scheduled_drafts: int | None = None
    step_id: int = 0
    timing_sampled: bool = False


def _cpu_array(value):
    # Never convert an unknown Tensor: numpy/cpu/item can synchronize the NPU.
    return value if isinstance(value, np.ndarray) and value.ndim == 1 else None


def _enabled():
    from .state_patch import is_enabled

    return is_enabled()


def launch_manifest(worker):
    config = worker.vllm_config
    scheduler = config.scheduler_config
    compilation = config.compilation_config
    cache = config.cache_config
    parallel = config.parallel_config
    transfer = getattr(config, "kv_transfer_config", None)
    return {
        "max_num_seqs": scheduler.max_num_seqs,
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "async_scheduling": scheduler.async_scheduling,
        "capture_sizes": list(
            getattr(compilation, "cudagraph_capture_sizes", ()) or ()
        ),
        "graph_mode": str(getattr(compilation, "cudagraph_mode", None)),
        "enforce_eager": config.model_config.enforce_eager,
        "max_model_len": config.model_config.max_model_len,
        "dtype": str(config.model_config.dtype),
        "tp": parallel.tensor_parallel_size,
        "dp": getattr(parallel, "data_parallel_size", 1),
        "dp_rank": getattr(parallel, "data_parallel_rank", 0),
        "pp": getattr(parallel, "pipeline_parallel_size", 1),
        "pcp": getattr(parallel, "prefill_context_parallel_size", 1),
        "dcp": getattr(parallel, "decode_context_parallel_size", 1),
        "chunked_prefill": getattr(scheduler, "enable_chunked_prefill", None),
        "long_prefill_token_threshold": getattr(scheduler, "long_prefill_token_threshold", None),
        "kv_transfer_configured": transfer is not None,
        "kv_transfer_role": getattr(transfer, "kv_role", None),
        "kv_connector": getattr(transfer, "kv_connector", None),
        "num_speculative_tokens": config.num_speculative_tokens,
        "prefix_caching": cache.enable_prefix_caching,
        "mamba_cache_mode": cache.mamba_cache_mode,
        "mamba_block_size": cache.mamba_block_size,
        "mamba_ssm_cache_dtype": str(getattr(cache, "mamba_ssm_cache_dtype", "unavailable")),
        "recovery_algorithm": "per_step_recover",
    }


def _fingerprint(manifest):
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def configure(worker):
    global _READY, _RANK, _LAST_CONFIG, _LATEST, _SOURCE_FINGERPRINT
    _READY = False
    _LATEST = None
    _RANK = int(getattr(worker, "rank", 0))
    manifest = launch_manifest(worker)
    _LAST_CONFIG = manifest
    _SOURCE_FINGERPRINT = source_fingerprint()


def source_fingerprint():
    """Identify installed source contents, including editable installations."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    paths = sorted(
        [*root.glob("*.py")] + [p for p in (root / "csrc").rglob("*") if p.is_file()]
    )
    if not paths:
        return "source-unavailable"
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _idle_reports():
    """One daemon on rank zero; it never touches the device or serving stream."""
    while not _STOP_REPORTING.wait(5.0):
        with _LOCK:
            idle = (
                _LAST_OUTPUT_AT is not None
                and time.monotonic() - _LAST_OUTPUT_AT >= 5.0
                and _COUNTS.get("output_steps", 0) != _LAST_REPORTED_STEPS
            )
        if idle:
            report()


def _shutdown_reporting():
    _STOP_REPORTING.set()
    if (
        _IDLE_THREAD is not None
        and _IDLE_THREAD.is_alive()
        and _IDLE_THREAD is not threading.current_thread()
    ):
        _IDLE_THREAD.join(timeout=1.0)
    report()


def _sample_timing(step_id):
    if step_id <= TIMING_FIRST_STEPS:
        return step_id > 0
    block, offset = divmod(step_id - 1, TIMING_SAMPLE_BLOCK)
    # A fixed offset in each block could always miss a phase in a periodic
    # workload. Vary it deterministically without a random-number generator.
    return offset == ((block * 2654435761) >> 26) % TIMING_SAMPLE_BLOCK


def timing_active(step=None):
    if not _READY or _RANK != 0:
        return False
    snapshot = _LATEST if step is None else step
    return snapshot is not None and snapshot.timing_sampled


def timing_start(stage, phase=None, *, step=None, next_step=False):
    """Read a host clock on sampled submission steps only."""
    if next_step:
        if not _READY or _RANK != 0 or not _sample_timing(_SUBMITTED_STEPS + 1):
            return None
        return stage, phase or "unknown", _SUBMITTED_STEPS + 1, time.perf_counter_ns()
    snapshot = _LATEST if step is None else step
    if not timing_active(snapshot):
        return None
    return stage, phase or snapshot.phase, snapshot.step_id, time.perf_counter_ns()


def timing_end(token, *, step=None):
    if token is None:
        return
    stage, phase, step_id, start_ns = token
    duration_ns = time.perf_counter_ns() - start_ns
    if step is not None:
        if step.step_id != step_id:
            return
        phase = step.phase
    with _LOCK:
        stats = _HOST_TIMINGS.get((stage, phase))
        if stats is None:
            stats = _HOST_TIMINGS[stage, phase] = {
                "samples": 0,
                "sampled_steps": 0,
                "sum_ns": 0,
                "max_ns": 0,
                "last_step_id": 0,
            }
        if stats["last_step_id"] != step_id:
            stats["sampled_steps"] += 1
            stats["last_step_id"] = step_id
        stats["samples"] += 1
        stats["sum_ns"] += duration_ns
        stats["max_ns"] = max(stats["max_ns"], duration_ns)


def prefill_dispatch(path, tokens, rows, state):
    """Count layer calls and capture examples of their host cache layout."""
    if not _READY or _RANK != 0 or _LATEST is None:
        return
    key = _LATEST.phase, path
    with _LOCK:
        counts = _PREFILL_DISPATCH.get(key)
        if counts is None:
            counts = _PREFILL_DISPATCH[key] = {
                "calls": 0, "layer_target_tokens": 0, "layer_request_rows": 0,
                "token_known_calls": 0, "row_known_calls": 0, "state_layout_examples": [],
            }
        counts["calls"] += 1
        if isinstance(tokens, (int, np.integer)):
            counts["layer_target_tokens"] += int(tokens)
            counts["token_known_calls"] += 1
        if isinstance(rows, (int, np.integer)):
            counts["layer_request_rows"] += int(rows)
            counts["row_known_calls"] += 1
        layouts = counts["state_layout_examples"]
        if state is not None and len(layouts) < 4 and (
            not layouts or _LATEST.timing_sampled
        ):
            layout = {
                "shape": [int(x) for x in state.shape],
                "stride": [int(x) for x in state.stride()],
                "dtype": str(state.dtype),
            }
            if layout not in layouts:
                layouts.append(layout)


def note_step(spec_rows, num_tokens, num_reqs, *, batch=None):
    """Snapshot CPU-known batch-row indices before output-copy construction."""
    global _LATEST, _START, _FIRST_SUBMIT_UNIX, _SUBMITTED_STEPS
    num_tokens = int(num_tokens)
    num_reqs = int(num_reqs)
    phase, context_sum, context_max, context_rows = "unknown", 0, 0, 0
    prefill_rows = prefill_tokens = spec_tokens = None
    attention_work = prefill_attention_work = scheduled_drafts = None
    flags = _cpu_array(getattr(batch, "is_prefilling_np", None))
    if flags is not None and flags.dtype.kind == "b" and len(flags) >= num_reqs:
        flags = flags[:num_reqs]
        prefill_rows = int(flags.sum())
        phase = "decode" if prefill_rows == 0 else (
            "prefill" if prefill_rows == num_reqs else "mixed"
        )
    else:
        flags = None
    computed = _cpu_array(getattr(batch, "num_computed_tokens_np", None))
    if computed is not None and computed.dtype.kind in "iu" and len(computed) >= num_reqs and num_reqs:
        values = computed[:num_reqs]
        context_sum = int(values.sum())
        context_max = int(values.max())
        context_rows = num_reqs
    rows = None if spec_rows is None else tuple(spec_rows)
    if _READY and _RANK == 0:
        draft_count = getattr(batch, "num_draft_tokens", None)
        if isinstance(draft_count, (int, np.integer)) and draft_count >= 0:
            scheduled_drafts = int(draft_count)
        scheduled = _cpu_array(getattr(batch, "num_scheduled_tokens", None))
        if (
            scheduled is not None and len(scheduled) == num_reqs
            and scheduled.dtype.kind in "iu" and int(scheduled.sum()) == num_tokens
        ):
            if flags is not None:
                prefill_tokens = int(scheduled.sum(where=flags))
            if rows and min(rows) >= 0 and max(rows) < num_reqs:
                spec_tokens = int(scheduled[list(rows)].sum())
            if context_rows and num_tokens:
                query = scheduled.astype(np.int64, copy=False)
                # Dense causal query/key pair estimate from the CPU mirror,
                # excluding graph padding. Async scheduling may overestimate.
                pairs = query * (2 * computed[:num_reqs].astype(np.int64, copy=False) + query + 1) // 2
                attention_work = int(pairs.sum())
                if flags is not None:
                    prefill_attention_work = int(pairs.sum(where=flags))
        _SUBMITTED_STEPS += 1
    step_id = _SUBMITTED_STEPS
    _LATEST = StepSnapshot(
        rows, num_tokens,
        num_reqs, phase, context_sum, context_max, context_rows,
        prefill_rows, prefill_tokens, spec_tokens, attention_work,
        prefill_attention_work, scheduled_drafts, step_id, _sample_timing(step_id),
    )
    if _READY:
        with _LOCK:
            if _START is None:
                _START = time.monotonic()
            if _FIRST_SUBMIT_UNIX is None:
                _FIRST_SUBMIT_UNIX = time.time()
    return _LATEST


def host_commit(duration_us, group_count, launch_count, rows):
    if not _READY:
        return
    with _LOCK:
        _COUNTS["commits"] += 1
        _COUNTS["host_submit_us"] += duration_us
        _COUNTS["kernel_submissions"] += launch_count
        _COUNTS["commit_rows"] += rows
        _COUNTS["groups"] += group_count
        if _RANK == 0:
            stats = _PHASE_COMMITS[_LATEST.phase if _LATEST is not None else "unknown"]
            stats["samples"] += 1
            stats["sum_us"] += duration_us
            stats["max_us"] = max(stats["max_us"], duration_us)
            stats["kernel_submissions"] += launch_count
            stats["rows"] += rows
            stats["groups"] += group_count


def observe_graph(desc, attn_metadata):
    if not _READY:
        return
    captured = int(getattr(desc, "num_tokens", 0) or 0)
    values = (
        attn_metadata.values() if isinstance(attn_metadata, dict) else (attn_metadata,)
    )
    actual = next(
        (
            getattr(m, "num_actual_tokens", None)
            for m in values
            if getattr(m, "num_actual_tokens", None) is not None
        ),
        None,
    )
    if actual is None:
        return
    actual = int(actual)
    with _LOCK:
        _GRAPHS[(actual, captured)] += 1
        _COUNTS["graph_actual_tokens"] += actual
        _COUNTS["graph_padded_tokens"] += max(captured - actual, 0)
        _COUNTS["graph_observed_steps"] += 1
        if _RANK == 0:
            graph = _PHASE_GRAPHS[_LATEST.phase if _LATEST is not None else "unknown"]
            graph["observed_steps"] += 1
            graph["actual_tokens"] += actual
            graph["padded_tokens"] += max(captured - actual, 0)


def _record_phase(step, emitted):
    stats = _PHASES[step.phase]
    stats["steps"] += 1
    stats["emitted_tokens"] += emitted
    stats["target_tokens"] += step.num_tokens
    stats["request_rows"] += step.num_reqs
    if step.scheduled_drafts is not None:
        stats["scheduled_draft_tokens"] += step.scheduled_drafts
        stats["scheduled_draft_known_steps"] += 1
    if step.prefill_rows is not None:
        stats["prefill_rows"] += step.prefill_rows
        stats["prefill_rows_known_steps"] += 1
    if step.prefill_tokens is not None:
        stats["prefill_target_tokens"] += step.prefill_tokens
        stats["decode_target_tokens"] += step.num_tokens - step.prefill_tokens
        stats["target_split_known_steps"] += 1
    if step.context_rows:
        stats["context_known_rows"] += step.context_rows
        stats["context_tokens_sum"] += step.context_sum
        stats["context_tokens_max"] = max(stats["context_tokens_max"], step.context_max)
        label = _CONTEXT_LABELS[bisect_left(_CONTEXT_BUCKETS, step.context_max)]
        _PHASE_CONTEXT[step.phase][label] += 1
    if step.attention_work is not None:
        stats["attention_work_proxy"] += step.attention_work
        stats["q_context_known_rows"] += step.context_rows
    if step.prefill_attention_work is not None:
        stats["prefill_attention_work_proxy"] += step.prefill_attention_work
        stats["decode_attention_work_proxy"] += step.attention_work - step.prefill_attention_work
        stats["q_context_split_known_steps"] += 1


def record_completed(counts, step):
    """Called only after AsyncOutput.get_output's pre-existing copy-event wait."""
    global _START, _LAST_OUTPUT_AT, _LAST_COMPLETED_UNIX
    if not _READY or _cpu_array(counts) is None:
        return
    # counts is the runner-owned NumPy array, not its device Tensor counterpart.
    with _LOCK:
        if _START is None:
            _START = time.monotonic()
        _LAST_OUTPUT_AT = time.monotonic()
        _COUNTS["output_steps"] += 1
        emitted = int(counts.sum())
        _COUNTS["emitted_tokens"] += emitted
        if _RANK == 0:
            _LAST_COMPLETED_UNIX = time.time()
        if step is not None:
            if not isinstance(step, StepSnapshot):
                step = StepSnapshot(*step)  # Legacy tuple/mock compatibility.
            spec_rows = step.spec_rows
            _COUNTS["batch_snapshot_steps"] += 1
            _COUNTS["actual_target_tokens"] += step.num_tokens
            _COUNTS["request_rows"] += step.num_reqs
            _COUNTS["peak_request_rows"] = max(_COUNTS["peak_request_rows"], step.num_reqs)
            _COUNTS["context_tokens_sum"] += step.context_sum
            _COUNTS["context_known_rows"] += step.context_rows
            _COUNTS["context_tokens_max"] = max(_COUNTS["context_tokens_max"], step.context_max)
            _BATCHES[(step.phase, step.num_reqs)] += 1
            if _RANK == 0:
                _record_phase(step, emitted)
            if spec_rows is not None:
                # These are CPU-owned completed outputs and CPU row identities.
                # Reduce in NumPy rather than taking a Python scalar per row.
                rows = np.asarray(spec_rows, dtype=np.int64)
                rows = rows[(rows >= 0) & (rows < len(counts))]
                values = counts[rows]
                spec_emitted = int(values.sum())
                _COUNTS["spec_rows"] += int(values.size)
                _COUNTS["spec_emitted_tokens"] += spec_emitted
                unique, frequencies = np.unique(values, return_counts=True)
                histogram = dict(zip(map(int, unique), map(int, frequencies)))
                if _RANK == 0:
                    stats = _PHASES[step.phase]
                    stats["spec_rows"] += int(values.size)
                    stats["spec_emitted_tokens"] += spec_emitted
                    if step.spec_tokens is not None and len(spec_rows) == values.size:
                        stats["spec_target_tokens"] += step.spec_tokens
                        stats["spec_target_known_rows"] += int(values.size)
                    _PHASE_ACCEPTED[step.phase].update(histogram)
        emit = (
            _COUNTS["output_steps"] == 1
            or _COUNTS["output_steps"] % REPORT_STEPS == 0
            or (_LAST_REPORTED_COMMITS == 0 and _COUNTS["commits"] > 0)
        )
    if emit:
        report()


def _phase_payload(elapsed, counts):
    """Build the cumulative JSON while holding the snapshot lock."""
    phases = {}
    for phase, stats in sorted(_PHASES.items()):
        phases[phase] = {
            **stats,
            "accepted_output_hist": dict(_PHASE_ACCEPTED.get(phase, {})),
            "context_max_steps": dict(_PHASE_CONTEXT.get(phase, {})),
        }
    timings = defaultdict(dict)
    for (stage, phase), stats in sorted(_HOST_TIMINGS.items()):
        total_us = stats["sum_ns"] / 1000.0
        timings[stage][phase] = {
            "samples": stats["samples"],
            "sampled_steps": stats["sampled_steps"],
            "sample_total_us": round(total_us, 3),
            "mean_us": round(total_us / stats["samples"], 3),
            "max_us": stats["max_ns"] / 1000.0,
        }
    for phase, stats in sorted(_PHASE_COMMITS.items()):
        timings["grouped_commit_host"][phase] = {
            "samples": stats["samples"],
            "sample_total_us": round(stats["sum_us"], 3),
            "mean_us": round(stats["sum_us"] / stats["samples"], 3),
            "max_us": round(stats["max_us"], 3),
            "kernel_submissions": stats["kernel_submissions"],
            "rows": stats["rows"],
            "groups": stats["groups"],
        }
    dispatch = defaultdict(dict)
    for (phase, path), stats in sorted(_PREFILL_DISPATCH.items()):
        dispatch[phase][path] = {
            **stats, "state_layout_examples": list(stats["state_layout_examples"]),
        }
    return {
        "schema": 2,
        "pid": os.getpid(),
        "rank": _RANK,
        "scope": "WORKER_CUMULATIVE_CPU_COMPLETED",
        "host_timing_scope": "HOST_WALL_MAY_INCLUDE_NATIVE_WAIT_OR_IN_FLIGHT_STEPS",
        "prefill_dispatch_scope": "HOST_LAYER_CALLS_NOT_DEVICE_GRAPH_EXECUTIONS",
        "context_source": "CPU_INPUT_MIRROR_MAY_BE_OPTIMISTIC_WITH_ASYNC",
        "first_submit_unix_s": _FIRST_SUBMIT_UNIX,
        "last_completed_unix_s": _LAST_COMPLETED_UNIX,
        "worker_window_s": round(elapsed, 6),
        "steps": counts.get("output_steps", 0),
        "emitted_tokens": counts.get("emitted_tokens", 0),
        "unattributed_steps": counts.get("output_steps", 0) - counts.get("batch_snapshot_steps", 0),
        "phases": phases,
        "graph_submissions": {phase: dict(stats) for phase, stats in _PHASE_GRAPHS.items()},
        "host_timings": dict(timings),
        "prefill_paths": dict(dispatch),
    }


def _path_counters():
    from . import ascendc_backend, mixed_forward

    return {
        "counts": ascendc_backend.counts(),
        "mixed": mixed_forward.counters(),
        "ascendc_fallbacks": ascendc_backend.fallback_counts(),
        "mixed_fallbacks": mixed_forward.fallback_counts(),
    }


def _report_totals(counts):
    spec_rows = counts.get("spec_rows", 0)
    commits = counts.get("commits", 0)
    actual = counts.get("graph_actual_tokens", 0)
    padded = counts.get("graph_padded_tokens", 0)
    graph_steps = counts.get("graph_observed_steps", 0)
    snapshot_steps = counts.get("batch_snapshot_steps", 0)
    context_rows = counts.get("context_known_rows", 0)
    return {
        "spec_rows": spec_rows,
        "spec_emitted_per_row": counts.get("spec_emitted_tokens", 0) / spec_rows if spec_rows else None,
        "commits": commits,
        "host_submit_us_avg": counts.get("host_submit_us", 0) / commits if commits else None,
        "kernel_submissions": counts.get("kernel_submissions", 0),
        "launches_per_commit": counts.get("kernel_submissions", 0) / commits if commits else None,
        "groups_per_commit": counts.get("groups", 0) / commits if commits else None,
        "graph_observed_steps": graph_steps,
        "graph_padding_tokens": padded,
        "graph_padding_fraction": padding_fraction(actual, actual + padded) if graph_steps else None,
        "graph_hist": [{"actual": a, "captured": b, "steps": n} for (a, b), n in _GRAPHS.most_common(8)],
        "batch_snapshot_steps": snapshot_steps,
        "target_tokens": counts.get("actual_target_tokens", 0) if snapshot_steps else None,
        "mean_rows": counts.get("request_rows", 0) / snapshot_steps if snapshot_steps else None,
        "peak_rows": counts.get("peak_request_rows", 0) if snapshot_steps else None,
        "mean_commit_rows": counts.get("commit_rows", 0) / commits if commits else None,
        "mean_computed_context": counts.get("context_tokens_sum", 0) / context_rows if context_rows else None,
        "batch_hist": [
            {"phase": phase, "rows": rows, "steps": steps}
            for (phase, rows), steps in sorted(_BATCHES.items())
        ],
    }


def report():
    global _LAST_REPORTED_STEPS, _LAST_REPORTED_COMMITS
    if _RANK != 0:
        return
    from . import sampling_patch

    # Serialize periodic and idle reports without holding the worker's stats
    # lock while writing to the logging handler.
    with _REPORT_LOCK:
        with _LOCK:
            counts = dict(_COUNTS)
            steps = counts.get("output_steps", 0)
            if steps <= _LAST_REPORTED_STEPS:
                return
            elapsed = (
                0.0
                if _START is None or _LAST_OUTPUT_AT is None
                else _LAST_OUTPUT_AT - _START
            )
            payload = _phase_payload(elapsed, counts)
            payload["totals"] = _report_totals(counts)
            _LAST_REPORTED_STEPS = steps
            _LAST_REPORTED_COMMITS = counts.get("commits", 0)
        paths = {}
        for metric, values in _path_counters().items():
            baseline = _PATHS_AT_READY.get(metric, {})
            paths[metric] = {
                name: count - baseline.get(name, 0)
                for name, count in values.items()
                if count > baseline.get(name, 0)
            }
        paths["scope"] = "HOST_DISPATCH_COUNTS_SINCE_READY_NOT_GRAPH_EXECUTIONS"
        payload["paths"] = paths
        payload["sampler"] = sampling_patch.perf_configuration()
        logger.info("GDN_PHASE_PERF json=%s", json.dumps(payload, separators=(",", ":")))


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm_ascend.worker.worker import NPUWorker

    original_init = AsyncOutput.__init__
    original_get = AsyncOutput.get_output
    original_warmup = NPUWorker.compile_or_warm_up_model

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if _READY and _enabled():
            self._gdn_perf_step = _LATEST
            self._gdn_perf_reported = False

    @functools.wraps(original_get)
    def get_output(self, *args, **kwargs):
        tracked = _READY and not getattr(self, "_gdn_perf_reported", True)
        step = getattr(self, "_gdn_perf_step", None) if tracked else None
        timer = timing_start("output_get_wall", step=step) if step is not None else None
        result = original_get(self, *args, **kwargs)
        if tracked:
            self._gdn_perf_reported = True
            timing_end(timer)
            record_completed(getattr(self, "num_sampled_tokens_np", None), step)
        return result

    @functools.wraps(original_warmup)
    def warmup(self, *args, **kwargs):
        global _READY, _IDLE_THREAD, _LATEST, _PATHS_AT_READY
        result = original_warmup(self, *args, **kwargs)
        if _enabled():
            if not getattr(self, "_gdn_patch_prepared", False):
                raise RuntimeError(
                    "RecoverSSM readiness refused: startup preparation and "
                    "numerical checks did not complete"
                )
            final = launch_manifest(self)
            # The patch never changes these fields. Native normalization, if any,
            # is visible rather than silently rewriting a user's capture policy.
            if _LAST_CONFIG is not None and final != _LAST_CONFIG:
                logger.warning(
                    "GDN_CONFIG native normalization before=%s after=%s",
                    _LAST_CONFIG,
                    final,
                )
            from . import graph_probe

            _PATHS_AT_READY = _path_counters()
            _LATEST = None
            _READY = True
            if _RANK == 0 and _IDLE_THREAD is None:
                _STOP_REPORTING.clear()
                _IDLE_THREAD = threading.Thread(
                    target=_idle_reports, name="gdn-perf-log", daemon=True
                )
                _IDLE_THREAD.start()
                atexit.register(_shutdown_reporting)
            logger.info(
                "GDN_READY json=%s",
                json.dumps({
                    "schema": 2, "pid": os.getpid(), "rank": _RANK,
                    "local_rank": getattr(self, "local_rank", None),
                    "patch": PATCH_ID, "source_fingerprint": _SOURCE_FINGERPRINT,
                    "launch_fingerprint": _fingerprint(final),
                    "ready_unix_s": time.time(), "parameters": final,
                    "replay_enabled": True,
                    "validation": "STARTUP_COMPLETE",
                    "startup": getattr(self, "_gdn_perf_startup", {}),
                    "startup_graphs": graph_probe.metrics(),
                    "startup_host_dispatches": _PATHS_AT_READY,
                    "workload_report_rank": 0,
                    "timing_sample_block_steps": TIMING_SAMPLE_BLOCK,
                    "timing_first_steps": TIMING_FIRST_STEPS,
                    "timing_scopes": _TIMING_SCOPES,
                    "timing_full_counter": "grouped_commit_host",
                    "report_interval_completed_steps": REPORT_STEPS,
                }, separators=(",", ":"), default=str),
            )
        return result

    AsyncOutput.__init__ = init
    AsyncOutput.get_output = get_output
    NPUWorker.compile_or_warm_up_model = warmup
    _INSTALLED = True
