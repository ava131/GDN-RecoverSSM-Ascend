# SPDX-License-Identifier: Apache-2.0
"""Server-side evidence, consuming the runner's existing completed CPU outputs.

Completed-output counters never read device tensors. Sparse phase intervals
measure host wall time, never device execution. In particular torch_npu event
elapsed_time may flush/synchronize queues, so no timing events are introduced.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import json
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vllm.logger import init_logger

from .metrics_math import padding_fraction

logger = init_logger("vllm.gdn_recoverssm_ascend.perf")
PATCH_ID = "ascendc-op-dev-pipeline-v2"
REPORT_STEPS = 200
_INSTALLED = False
_READY = False
_RANK = 0
_LATEST = None
_LOCK = threading.Lock()
_COUNTS = Counter()
_GRAPHS = Counter()
_BATCHES = Counter()
_PHASES = Counter()
_ACCEPTED = Counter()
_START = None
_LAST_CONFIG = None
_IDLE_THREAD = None
_LAST_OUTPUT_AT = None
_LAST_REPORTED_STEPS = 0
_LAST_REPORTED_REVISION = 0
_STOP_REPORTING = threading.Event()
_HOST_STAGE_CALLS = Counter()
_HOST_STAGE_HIST = {}
_OBSERVABILITY_FAILED = False
_BATCH_SCHEMAS = set()
_LAST_LAYOUT_ID = None


def _safe_observation(function):
    """Optional evidence must never mask a serving result or original error."""
    @functools.wraps(function)
    def observe(*args, **kwargs):
        global _OBSERVABILITY_FAILED
        if _OBSERVABILITY_FAILED:
            return None
        try:
            return function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - observability is non-critical.
            _OBSERVABILITY_FAILED = True
            logger.warning(
                "GDN_DIAGNOSTICS disabled reason=%s:%s serving=UNCHANGED "
                "cumulative_metrics_may_be_partial=1", type(exc).__name__, exc,
            )
            return None
    return observe


@dataclass(frozen=True)
class StepSnapshot:
    spec_rows: tuple | None
    num_tokens: int
    num_reqs: int
    phase: str = "unknown"
    prefill_rows: int = 0
    prefill_target_tokens: int = 0
    spec_draft_slots: tuple | None = None
    context_sum: int = 0
    context_max: int = 0
    context_rows: int = 0
    prefill_token_known: bool = False


def _cpu_array(value):
    # Torch Tensor does not implement this NumPy array interface. In particular,
    # do not call numpy(), cpu(), item() or tolist() on an unknown object.
    return value if hasattr(value, "__array_interface__") else None


def _enabled():
    from .state_patch import is_enabled

    return is_enabled()


def launch_manifest(worker):
    config = worker.vllm_config
    scheduler = config.scheduler_config
    compilation = config.compilation_config
    cache = config.cache_config
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
        "tp": config.parallel_config.tensor_parallel_size,
        "num_speculative_tokens": config.num_speculative_tokens,
        "prefix_caching": cache.enable_prefix_caching,
        "mamba_cache_mode": cache.mamba_cache_mode,
        "mamba_block_size": cache.mamba_block_size,
        "mamba_ssm_cache_dtype": str(getattr(cache, "mamba_ssm_cache_dtype", "unavailable")),
        "hf_mamba_ssm_dtype": str(getattr(
            getattr(config.model_config, "hf_text_config", None), "mamba_ssm_dtype", "unavailable"
        )),
        "recovery_algorithm": "per_step_recover",
    }


def _fingerprint(manifest):
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def configure(worker):
    global _READY, _RANK, _LAST_CONFIG
    _READY = False
    _RANK = int(getattr(worker, "rank", 0))
    manifest = launch_manifest(worker)
    _LAST_CONFIG = manifest
    logger.info(
        "GDN_CONFIG patch=%s rank=%d source_fingerprint=%s fingerprint=%s parameters=%s",
        PATCH_ID,
        _RANK,
        source_fingerprint(),
        _fingerprint(manifest),
        json.dumps(manifest, sort_keys=True, default=str),
    )
    if _RANK == 0:
        logger.info(
            "GDN_TIMING device_timing=UNAVAILABLE "
            "reason=framework_elapsed_time_may_flush_streams "
            "phase_timing=HOST_WALL_NOT_DEVICE sample_stride=128 "
            "timing_events=0 additional_host_synchronize=0",
        )


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
                and _COUNTS.get("revision", 0) != _LAST_REPORTED_REVISION
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


@_safe_observation
def note_step(spec_rows, num_tokens, num_reqs, *, batch=None):
    """Snapshot CPU-known batch-row indices before output-copy construction."""
    global _LATEST, _START
    rows = None if spec_rows is None else tuple(spec_rows)
    num_reqs = int(num_reqs)
    flags = _cpu_array(getattr(batch, "is_prefilling_np", None))
    scheduled = _cpu_array(getattr(batch, "num_scheduled_tokens", None))
    starts = _cpu_array(getattr(batch, "query_start_loc_np", None))
    drafts = _cpu_array(getattr(batch, "num_draft_tokens_per_req", None))
    computed = _cpu_array(getattr(batch, "num_computed_tokens_np", None))
    schema = tuple(value is not None for value in (flags, scheduled, starts, drafts, computed))
    if _READY and schema not in _BATCH_SCHEMAS:
        _BATCH_SCHEMAS.add(schema)
        logger.info(
            "GDN_BATCH_SCHEMA cpu_arrays=%s missing_fields_report_unknown=1 "
            "tensor_contents_read=0",
            json.dumps(dict(zip(
                ("is_prefilling_np", "num_scheduled_tokens", "query_start_loc_np",
                 "num_draft_tokens_per_req", "num_computed_tokens_np"), schema
            )), separators=(",", ":")),
        )
    prefill_rows, prefill_tokens, phase, prefill_known = 0, 0, "unknown", False
    if flags is not None and len(flags) >= num_reqs:
        prefill_rows = sum(bool(value) for value in flags[:num_reqs])
        phase = "decode" if not prefill_rows else (
            "prefill" if prefill_rows == num_reqs else "mixed"
        )
        if scheduled is not None and len(scheduled) >= num_reqs:
            prefill_known = True
            prefill_tokens = sum(
                int(scheduled[row]) for row in range(num_reqs) if flags[row]
            )
    draft_slots = None
    if (rows is not None and starts is not None and len(starts) >= num_reqs + 1
            and drafts is not None and len(drafts) >= num_reqs):
        draft_slots = tuple(
            min(max(int(starts[row + 1]) - int(starts[row]) - 1, 0),
                max(int(drafts[row]), 0))
            if 0 <= row < num_reqs else 0 for row in rows
        )
    context_sum, context_max, context_rows = 0, 0, 0
    if computed is not None and len(computed) >= num_reqs:
        context_rows = num_reqs
        context_sum = sum(int(value) for value in computed[:num_reqs])
        context_max = max((int(value) for value in computed[:num_reqs]), default=0)
    _LATEST = StepSnapshot(
        rows, int(num_tokens), num_reqs, phase, prefill_rows,
        prefill_tokens, draft_slots, context_sum, context_max, context_rows, prefill_known,
    )
    if _READY:
        with _LOCK:
            if _START is None:
                _START = time.monotonic()
    return _LATEST


@_safe_observation
def bind_step(metadata, step):
    """The producer's immutable CPU snapshot travels with its commit metadata."""
    if step is None:
        return
    for meta in metadata.values():
        # Only our extended GDN metadata owns this additional attribute. Other
        # backends may use slots/frozen metadata and must remain untouched.
        if getattr(meta, "recoverssm_commit", None) is not None:
            meta.recoverssm_perf_step = step


@_safe_observation
def step_for_metadata(metadata):
    if metadata is None:
        return None
    values = metadata.values() if isinstance(metadata, dict) else (
        metadata if isinstance(metadata, (tuple, list)) else (metadata,)
    )
    for meta in values:
        step = getattr(meta, "recoverssm_perf_step", None)
        if isinstance(step, StepSnapshot):
            return step
    return None


@_safe_observation
def note_layout(context):
    """Actual tensor metadata only; no owner references or device values kept."""
    global _LAST_LAYOUT_ID
    if not _READY or context is None or id(context) == _LAST_LAYOUT_ID:
        return
    checkpoints = getattr(context, "checkpoints", ())
    if not checkpoints:
        return
    state = checkpoints[0]
    records = getattr(context, "records", ())
    record = records[0] if records else None
    logger.info(
        "GDN_RUNTIME_LAYOUT state_dtype=%s state_shape=%s state_stride=%s "
        "record_key_dtype=%s record_correction_dtype=%s layers_in_context=%d "
        "record_rows=%s record_bytes=%s recovery_algorithm=per_step_recover "
        "prepared_backend=ascendc_verify_fold plan_conv_backend=triton "
        "runtime_dispatch=signature_guarded tensor_contents_read=0",
        state.dtype, tuple(state.shape), tuple(state.stride()),
        getattr(record, "key_dtype", "legacy-fp32"),
        getattr(getattr(record, "correction", None), "dtype", "legacy-fp32"),
        len(checkpoints), getattr(context, "record_rows", "unknown"),
        getattr(context, "record_bytes", "unknown"),
    )
    _LAST_LAYOUT_ID = id(context)


@_safe_observation
def phase_begin(scope, *, step=None):
    if not _READY or _RANK != 0:
        return None
    with _LOCK:
        _HOST_STAGE_CALLS[scope] += 1
        call = _HOST_STAGE_CALLS[scope]
    if call != 1 and call % 128:
        return None
    context = ("unknown", -1, -1) if step is None else (
        step.phase, step.num_reqs, step.num_tokens
    )
    return (time.perf_counter_ns(), str(scope), *context)


@_safe_observation
def phase_end(sample, *, failed=False):
    if sample is None:
        return
    start, scope, phase, rows, tokens = sample
    duration_us = (time.perf_counter_ns() - start) / 1000.0
    with _LOCK:
        if failed:
            _COUNTS["host_stage_failed_samples"] += 1
            _COUNTS["revision"] += 1
            return
        key = (scope, phase, rows, tokens)
        if key not in _HOST_STAGE_HIST and len(_HOST_STAGE_HIST) >= 127:
            key = ("overflow", "unknown", -1, -1)
        stat = _HOST_STAGE_HIST.setdefault(key, [0, 0.0, duration_us, duration_us])
        stat[0] += 1
        stat[1] += duration_us
        stat[2] = min(stat[2], duration_us)
        stat[3] = max(stat[3], duration_us)
        _COUNTS["revision"] += 1


@_safe_observation
def host_commit(duration_us, group_count, launch_count, rows):
    if not _READY:
        return
    with _LOCK:
        _COUNTS["commits"] += 1
        _COUNTS["host_submit_us"] += duration_us
        _COUNTS["kernel_submissions"] += launch_count
        _COUNTS["commit_rows"] += rows
        _COUNTS["groups"] += group_count
        _COUNTS["revision"] += 1


@_safe_observation
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
        _COUNTS["revision"] += 1


@_safe_observation
def record_completed(counts, step):
    """Called only after AsyncOutput.get_output's pre-existing copy-event wait."""
    global _START, _LAST_OUTPUT_AT
    if not _READY or counts is None or not hasattr(counts, "__array_interface__"):
        return
    # counts is the runner-owned NumPy array, not its device Tensor counterpart.
    emitted = int(counts.sum())
    with _LOCK:
        if _START is None:
            _START = time.monotonic()
        _LAST_OUTPUT_AT = time.monotonic()
        _COUNTS["output_steps"] += 1
        _COUNTS["revision"] += 1
        _COUNTS["emitted_tokens"] += emitted
        if step is not None:
            _COUNTS["batch_snapshot_steps"] += 1
            if not isinstance(step, StepSnapshot):
                step = StepSnapshot(*step)  # Previous/mock three-field ABI.
            spec_rows = step.spec_rows
            _COUNTS["actual_target_tokens"] += step.num_tokens
            _COUNTS["request_rows"] += step.num_reqs
            _COUNTS["peak_request_rows"] = max(
                _COUNTS["peak_request_rows"], step.num_reqs
            )
            _COUNTS["prefill_rows"] += step.prefill_rows
            _COUNTS["prefill_target_tokens"] += step.prefill_target_tokens
            _COUNTS["prefill_tokens_known_steps"] += int(step.prefill_token_known)
            _COUNTS["context_tokens_sum"] += step.context_sum
            _COUNTS["context_known_rows"] += step.context_rows
            _COUNTS["context_tokens_max"] = max(
                _COUNTS["context_tokens_max"], step.context_max
            )
            _BATCHES[(step.phase, step.num_reqs)] += 1
            _PHASES[(step.phase, "steps")] += 1
            _PHASES[(step.phase, "target_tokens")] += step.num_tokens
            _PHASES[(step.phase, "emitted_tokens")] += emitted
            if spec_rows is not None:
                for index, row in enumerate(spec_rows):
                    if 0 <= row < len(counts):
                        value = int(counts[row])
                        _COUNTS["spec_rows"] += 1
                        _COUNTS["spec_emitted_tokens"] += value
                        _ACCEPTED[value] += 1
                        if step.spec_draft_slots is not None:
                            slots = step.spec_draft_slots[index]
                            _COUNTS["spec_draft_slots"] += slots
                            _COUNTS["spec_draft_accepted"] += min(max(value - 1, 0), slots)
        emit = (
            _COUNTS["output_steps"] <= 3 or _COUNTS["output_steps"] % REPORT_STEPS == 0
        )
    if emit:
        report()


@_safe_observation
def report():
    global _LAST_REPORTED_STEPS, _LAST_REPORTED_REVISION
    if _RANK != 0:
        return
    with _LOCK:
        c = dict(_COUNTS)
        hist = [
            {"actual": a, "captured": b, "steps": n}
            for (a, b), n in sorted(_GRAPHS.items())
        ]
        batches = [{"phase": phase, "rows": rows, "steps": n}
                   for (phase, rows), n in sorted(_BATCHES.items())]
        phases = {phase: {name: _PHASES[(phase, name)]
                          for name in ("steps", "target_tokens", "emitted_tokens")}
                  for phase in sorted({key[0] for key in _PHASES})}
        accepted = dict(sorted(_ACCEPTED.items()))
        host_samples = [
            {"scope": key[0], "phase": key[1], "batch_rows": key[2], "target_tokens": key[3],
             "samples": stat[0], "mean_us": stat[1] / stat[0],
             "min_us": stat[2], "max_us": stat[3]}
            for key, stat in sorted(_HOST_STAGE_HIST.items())
        ]
        elapsed = (
            0.0
            if _START is None or _LAST_OUTPUT_AT is None
            else _LAST_OUTPUT_AT - _START
        )
        _LAST_REPORTED_STEPS = c.get("output_steps", 0)
        _LAST_REPORTED_REVISION = c.get("revision", 0)
    spec_rows = c.get("spec_rows", 0)
    commits = c.get("commits", 0)
    actual = c.get("graph_actual_tokens", 0)
    padded = c.get("graph_padded_tokens", 0)
    logger.info(
        "GDN_PERF steps=%d emitted_tokens=%d spec_rows=%d "
        "spec_emitted_per_row=%.4f commits=%d host_submit_us_avg=%.1f "
        "kernel_submissions=%d launches_per_commit=%.3f groups_per_commit=%.3f "
        "graph_observed_steps=%d graph_padding_tokens=%d graph_padding_fraction=%.6f "
        "worker_observed_s=%.3f graph_hist=%s timing=HOST_ONLY",
        c.get("output_steps", 0),
        c.get("emitted_tokens", 0),
        spec_rows,
        c.get("spec_emitted_tokens", 0) / spec_rows if spec_rows else 0.0,
        commits,
        c.get("host_submit_us", 0) / commits if commits else 0.0,
        c.get("kernel_submissions", 0),
        c.get("kernel_submissions", 0) / commits if commits else 0.0,
        c.get("groups", 0) / commits if commits else 0.0,
        c.get("graph_observed_steps", 0), padded,
        padding_fraction(actual, actual + padded) if c.get("graph_observed_steps", 0) else -1.0,
        elapsed,
        json.dumps(hist, separators=(",", ":")),
    )
    steps, rows = c.get("output_steps", 0), c.get("request_rows", 0)
    batch_steps = c.get("batch_snapshot_steps", 0)
    draft_slots = c.get("spec_draft_slots", 0)
    logger.info(
        "GDN_WORKLOAD steps=%d mean_rows=%.4f peak_rows=%d mean_gdn_rows=%.4f "
        "mean_commit_rows=%.4f target_tokens=%d "
        "batch_snapshot_steps=%d unknown_phase_steps=%d prefill_row_target_tokens=%d "
        "prefill_tokens_known_steps=%d context_known_rows=%d "
        "mean_computed_context=%.1f max_computed_context=%d "
        "spec_draft_slots=%d spec_draft_accepted=%d draft_acceptance=%.6f "
        "accepted_output_hist=%s phases=%s batch_hist=%s "
        "scope=WORKER_CUMULATIVE_CPU_COMPLETED not_client_duration=1",
        steps, rows / batch_steps if batch_steps else -1.0,
        c.get("peak_request_rows", 0) if batch_steps else -1,
        spec_rows / batch_steps if batch_steps else -1.0,
        c.get("commit_rows", 0) / commits if commits else 0.0,
        c.get("actual_target_tokens", 0) if batch_steps else -1,
        batch_steps, phases.get("unknown", {}).get("steps", 0),
        c.get("prefill_target_tokens", 0) if c.get("prefill_tokens_known_steps", 0) else -1,
        c.get("prefill_tokens_known_steps", 0), c.get("context_known_rows", 0),
        c.get("context_tokens_sum", 0) / c["context_known_rows"]
        if c.get("context_known_rows", 0) else -1.0,
        c.get("context_tokens_max", 0) if c.get("context_known_rows", 0) else -1,
        draft_slots, c.get("spec_draft_accepted", 0),
        c.get("spec_draft_accepted", 0) / draft_slots if draft_slots else -1.0,
        json.dumps(accepted, separators=(",", ":")),
        json.dumps(phases, separators=(",", ":")),
        json.dumps(batches, separators=(",", ":")),
    )
    logger.info(
        "GDN_STAGE_HOST samples=%s failed_samples=%d sample_stride=128 "
        "timing=HOST_WALL_NOT_DEVICE not_additive_critical_path=1 "
        "scope_row_semantics=PRODUCER_BATCH_NOT_GDN_SUBSET "
        "device_timing=UNAVAILABLE timing_events=0",
        json.dumps(host_samples, separators=(",", ":")),
        c.get("host_stage_failed_samples", 0),
    )


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
        if _enabled():
            self._gdn_perf_step = _LATEST
            self._gdn_perf_reported = False

    @functools.wraps(original_get)
    def get_output(self, *args, **kwargs):
        result = original_get(self, *args, **kwargs)
        if _enabled() and not getattr(self, "_gdn_perf_reported", True):
            self._gdn_perf_reported = True
            record_completed(
                getattr(self, "num_sampled_tokens_np", None),
                getattr(self, "_gdn_perf_step", None),
            )
        return result

    @functools.wraps(original_warmup)
    def warmup(self, *args, **kwargs):
        global _READY, _IDLE_THREAD
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
            _READY = True
            if _RANK == 0 and _IDLE_THREAD is None:
                _STOP_REPORTING.clear()
                _IDLE_THREAD = threading.Thread(
                    target=_idle_reports, name="gdn-perf-log", daemon=True
                )
                _IDLE_THREAD.start()
                atexit.register(_shutdown_reporting)
            logger.info(
                "GDN_READY patch=%s rank=%d launch_fingerprint=%s "
                "capture_policy=UNCHANGED validation=STARTUP_COMPLETE",
                PATCH_ID,
                _RANK,
                _fingerprint(final),
            )
        return result

    AsyncOutput.__init__ = init
    AsyncOutput.get_output = get_output
    NPUWorker.compile_or_warm_up_model = warmup
    _INSTALLED = True
