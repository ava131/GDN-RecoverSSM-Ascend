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
import threading
import time
from collections import Counter
from pathlib import Path

from vllm.logger import init_logger

from .metrics_math import padding_fraction

logger = init_logger("vllm.gdn_recoverssm_ascend.perf")
PATCH_ID = "ascendc-op-dev-v1"
REPORT_STEPS = 200
_INSTALLED = False
_READY = False
_RANK = 0
_LATEST = None
_LOCK = threading.Lock()
_COUNTS = Counter()
_GRAPHS = Counter()
_START = None
_LAST_CONFIG = None
_IDLE_THREAD = None
_LAST_OUTPUT_AT = None
_LAST_REPORTED_STEPS = 0
_STOP_REPORTING = threading.Event()


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


def note_step(spec_rows, num_tokens, num_reqs):
    """Snapshot CPU-known batch-row indices before output-copy construction."""
    global _LATEST, _START
    _LATEST = (
        None if spec_rows is None else tuple(spec_rows),
        int(num_tokens),
        int(num_reqs),
    )
    if _READY:
        with _LOCK:
            if _START is None:
                _START = time.monotonic()


def host_commit(duration_us, group_count, launch_count, rows):
    if not _READY:
        return
    with _LOCK:
        _COUNTS["commits"] += 1
        _COUNTS["host_submit_us"] += duration_us
        _COUNTS["kernel_submissions"] += launch_count
        _COUNTS["commit_rows"] += rows
        _COUNTS["groups"] += group_count


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


def record_completed(counts, step):
    """Called only after AsyncOutput.get_output's pre-existing copy-event wait."""
    global _START, _LAST_OUTPUT_AT
    if not _READY or counts is None or not hasattr(counts, "__array_interface__"):
        return
    # counts is the runner-owned NumPy array, not its device Tensor counterpart.
    with _LOCK:
        if _START is None:
            _START = time.monotonic()
        _LAST_OUTPUT_AT = time.monotonic()
        _COUNTS["output_steps"] += 1
        _COUNTS["emitted_tokens"] += int(counts.sum())
        if step is not None:
            spec_rows, num_tokens, num_reqs = step
            _COUNTS["actual_target_tokens"] += num_tokens
            _COUNTS["request_rows"] += num_reqs
            if spec_rows is not None:
                for row in spec_rows:
                    if 0 <= row < len(counts):
                        _COUNTS["spec_rows"] += 1
                        _COUNTS["spec_emitted_tokens"] += int(counts[row])
        emit = (
            _COUNTS["output_steps"] <= 3 or _COUNTS["output_steps"] % REPORT_STEPS == 0
        )
    if emit:
        report()


def report():
    global _LAST_REPORTED_STEPS
    if _RANK != 0:
        return
    with _LOCK:
        c = dict(_COUNTS)
        hist = [
            {"actual": a, "captured": b, "steps": n}
            for (a, b), n in _GRAPHS.most_common(8)
        ]
        elapsed = (
            0.0
            if _START is None or _LAST_OUTPUT_AT is None
            else _LAST_OUTPUT_AT - _START
        )
        _LAST_REPORTED_STEPS = c.get("output_steps", 0)
    spec_rows = c.get("spec_rows", 0)
    commits = c.get("commits", 0)
    actual = c.get("graph_actual_tokens", 0)
    padded = c.get("graph_padded_tokens", 0)
    logger.info(
        "GDN_PERF steps=%d emitted_tokens=%d spec_rows=%d "
        "spec_emitted_per_row=%.4f commits=%d host_submit_us_avg=%.1f "
        "kernel_submissions=%d launches_per_commit=%.3f groups_per_commit=%.3f "
        "graph_padding_tokens=%d graph_padding_fraction=%.6f "
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
        padded,
        padding_fraction(actual, actual + padded),
        elapsed,
        json.dumps(hist, separators=(",", ":")),
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
