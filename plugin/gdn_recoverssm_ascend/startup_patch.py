# SPDX-License-Identifier: Apache-2.0
"""Move the RecoverSSM commit's first-time resource preparation into startup.

Today the commit context, the first JIT compilation of the three commit kernels
and their device-side binary registration all happen on the **first real
sampling step**, i.e. while the server is already serving. A disk cache hit does
not help there: ``CompiledKernel._init_handles`` performs the per-process
``rtDevBinaryRegister`` / ``rtFunctionRegister``, which only runs when the kernel
is actually reached.

This module prepares that work right after the real KV cache is bound and before
vLLM's own warmup / graph capture:

    initialize_from_config   (KV bound)
      -> NPUWorker.compile_or_warm_up_model
           -> prepare()        # create contexts, compile+register, verify handles
           -> _dummy_run / kernel_warmup / capture_model   (unchanged)

Nothing is launched here: the kernels are compiled and registered only
(``warmup=True`` + ``_init_handles``), so no GDN state is touched.

Scope: the three commit kernels, for every speculative row count the plan buffers
can hold. The verify kernel is compiled by graph capture on its own path and is
not part of this promise.
"""

from __future__ import annotations

import functools
import os

from vllm.logger import init_logger

from . import selftest

logger = init_logger("vllm.gdn_recoverssm_ascend.startup")

_APPLIED = False
_SELFTEST_DONE = False


def install() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from vllm_ascend.worker.worker import NPUWorker

    original = NPUWorker.compile_or_warm_up_model

    @functools.wraps(original)
    def compile_or_warm_up_model(self, *args, **kwargs):
        # KV cache is bound by initialize_from_config, which runs before this
        # method; if it is not bound the preparation must fail the startup
        # instead of silently deferring to the first request.
        _prepare(self)
        return original(self, *args, **kwargs)

    NPUWorker.compile_or_warm_up_model = compile_or_warm_up_model

    # Reserve the scratch memory the records will occupy BEFORE vLLM sizes the
    # KV cache. ``determine_available_memory`` decides how many blocks the pool
    # gets from (utilization x total - weights - activations); a buffer allocated
    # afterwards competes with that pool and can push the run past
    # gpu_memory_utilization. Subtracting the reservation here means the pool is
    # sized with the scratch already accounted for, so nothing over-subscribes.
    original_avail = NPUWorker.determine_available_memory

    @functools.wraps(original_avail)
    def determine_available_memory(self):
        available = original_avail(self)
        config = getattr(self, "vllm_config", None)
        if not feature_enabled(config):
            return available
        reserve = _record_scratch_bytes(config)
        margin = _graph_margin_bytes(config)
        total = reserve + margin
        if not total:
            return available
        logger.info(
            "Ascend GDN RecoverSSM KV budget: records %.1f MiB (%s) + graph "
            "headroom %.0f MiB = %.1f MiB deducted from available KV memory "
            "(%.1f GiB -> %.1f GiB). The graph headroom exists because vLLM sizes "
            "the pool to fill the budget and would otherwise leave capture short; "
            "set %s to tune it (0 disables).",
            reserve / (1024 * 1024),
            _record_scratch_desc(config),
            margin / (1024 * 1024),
            total / (1024 * 1024),
            available / (1024**3),
            max(available - total, 0) / (1024**3),
            _MARGIN_ENV,
        )
        return max(available - total, 0)

    NPUWorker.determine_available_memory = determine_available_memory
    _APPLIED = True


_MARGIN_ENV = "GDN_RECOVERSSM_KV_MARGIN_MB"


def feature_enabled(config) -> bool:
    """True only when the RecoverSSM path will actually run.

    The plugin is installed in every process as soon as its entry point loads, and
    the cache patch only sets ``use_kda_recoverssm`` when ``--use-replayssm`` was
    accepted. Everything below must key off that resolved flag: a run without the
    feature is a baseline and must keep its full KV budget and its logs.
    """
    cache = getattr(config, "cache_config", None)
    return bool(getattr(cache, "use_kda_recoverssm", False))


def _graph_margin_bytes(config) -> int:
    """Headroom kept out of the KV pool for the graph pool.

    vLLM sizes the KV pool to fill the budget: ``available = requested -
    non_kv_cache_memory``, where non_kv_cache_memory deliberately excludes graph
    memory. That works as long as the utilisation slack is larger than the graph
    pool, and moving the records out of the pages grew the pool by ~3 GiB, which
    ate that slack: capture then fails asking for 770 MiB with 545 MiB free.

    So keep an explicit margin. Eager has no graph pool, so the default is 0
    there; graph mode defaults to 2 GiB, which covers the ~1.9 GiB the 34 captured
    graphs need. ``GDN_RECOVERSSM_KV_MARGIN_MB`` overrides it.
    """
    compilation = getattr(config, "compilation_config", None)
    mode = getattr(compilation, "cudagraph_mode", None)
    # CUDAGraphMode is a plain Enum, but the value can also arrive as a string.
    # Only an explicit NONE means "no graph pool"; anything unknown (including a
    # config that has not resolved the mode yet) is treated as graph mode, because
    # reserving too little fails the start while reserving too much only costs
    # capacity.
    name = (getattr(mode, "name", None) or str(mode or "")).upper()
    eager = name == "NONE"
    default_mb = 0 if eager else 2048
    raw = os.environ.get(_MARGIN_ENV)
    try:
        mb = int(raw) if raw is not None else default_mb
    except ValueError:
        logger.warning(
            "Ascend GDN RecoverSSM: %s=%r is not an integer; using %d",
            _MARGIN_ENV,
            raw,
            default_mb,
        )
        mb = default_mb
    return max(mb, 0) << 20


def _record_geometry(config) -> dict | None:
    """Rows / layers / heads / window / width of the record scratch, or None."""
    if config is None:
        return None
    try:
        hf = config.model_config.hf_text_config
        heads = getattr(hf, "linear_num_value_heads", None)
        dim_k = getattr(hf, "linear_key_head_dim", None)
        dim_v = getattr(hf, "linear_value_head_dim", None)
        tp = config.parallel_config.tensor_parallel_size
        window = 1 + int(config.num_speculative_tokens or 0)
        if not (heads and dim_k and dim_v and heads % tp == 0):
            return None
        layer_types = getattr(hf, "layer_types", None)
        if layer_types:
            layers = sum(1 for t in layer_types if t != "full_attention")
        else:
            layers = int(getattr(hf, "num_hidden_layers", 0) or 0)
        # The records must cover a full-graph batch, which can be wider than
        # max_num_seqs: the commit context sizes its scratch as
        # max(max_num_seqs, decode_cudagraph_max_bs). At KV-sizing time the latter
        # is not resolved yet, so derive it from the capture sizes (tokens /
        # speculative window) the same way. Under-reserving here would silently
        # eat the graph headroom later.
        rows = int(config.scheduler_config.max_num_seqs or 0)
        compilation = getattr(config, "compilation_config", None)
        sizes = getattr(compilation, "cudagraph_capture_sizes", None) or ()
        try:
            capture_rows = max(int(x) for x in sizes) // max(window, 1) if sizes else 0
        except (TypeError, ValueError):
            capture_rows = 0
        rows = max(rows, capture_rows)
        if layers <= 0 or rows <= 0:
            return None
        return {
            "rows": rows,
            "layers": layers,
            "heads": heads // tp,
            "window": window,
            "width": dim_v + dim_k + 1,
        }
    except Exception:
        return None


def _record_scratch_bytes(config) -> int:
    geo = _record_geometry(config)
    if geo is None:
        return 0
    return (
        geo["rows"]
        * geo["layers"]
        * geo["heads"]
        * geo["window"]
        * geo["width"]
        * 4  # fp32
    )


def _record_scratch_desc(config) -> str:
    geo = _record_geometry(config)
    if geo is None:
        return "geometry unavailable"
    return (
        "rows={rows}, layers={layers}, heads={heads}, window={window}, "
        "width={width}, fp32, per-request temporaries reused every step"
    ).format(**geo)


def _iter_builders(model_runner):
    """Every distinct RecoverSSM metadata builder the runner will use."""
    seen: set[int] = set()
    for groups in getattr(model_runner, "attn_groups", ()) or ():
        for group in groups:
            builder = None
            try:
                builder = group.get_metadata_builder()
            except Exception:
                builder = getattr(group, "metadata_builder", None)
            if builder is None or id(builder) in seen:
                continue
            # Duck typing: only our builder owns these.
            if not (hasattr(builder, "_accepted_buffer") and hasattr(builder, "_get_context")):
                continue
            seen.add(id(builder))
            yield builder


def _log_runtime_flags(worker) -> None:
    """Record the FINAL config values, not what the command line looked like.

    ``--async-scheduling`` is a BooleanOptionalAction: omitting it leaves the
    value at ``None``, which VllmConfig may resolve to ``True``. Deleting the flag
    is therefore not a way to disable it, and ``--async-scheduling false`` is
    rejected by argparse. Logging the resolved value (and the scheduler class it
    selected) is the only reliable record.
    """
    config = getattr(worker, "vllm_config", None)
    scheduler = getattr(config, "scheduler_config", None)
    if scheduler is None:
        return
    try:
        cls_name = scheduler.get_scheduler_cls().__name__
    except Exception as exc:  # pragma: no cover - defensive
        cls_name = f"<unavailable: {exc}>"
    enabled = bool(scheduler.async_scheduling)
    logger.info(
        "Ascend GDN RecoverSSM: async_scheduling=%s (raw=%s) scheduler_cls=%s",
        enabled,
        scheduler.async_scheduling,
        cls_name,
    )
    compilation = getattr(config, "compilation_config", None)
    if compilation is not None:
        mode = getattr(compilation, "cudagraph_mode", None)
        logger.info(
            "Ascend GDN RecoverSSM: cudagraph_mode=%s%s",
            mode,
            " (eager: no graph capture, so no first-execute queue setup is needed)"
            if str(mode).endswith("NONE") or mode is None
            else "",
        )
    cache = getattr(config, "cache_config", None)
    if cache is not None:
        logger.info(
            "Ascend GDN RecoverSSM: prefix_caching=%s mamba_cache_mode=%s "
            "records=row-indexed (a prefix-cache-shared page cannot carry another "
            "request's records)",
            getattr(cache, "enable_prefix_caching", "?"),
            getattr(cache, "mamba_cache_mode", "?"),
        )
    if enabled:
        # Asynchronous scheduling is supported: the plugin executes every captured
        # graph once right after capture (see graph_probe._warm_captured_graphs),
        # which is where the lazy queue construction used to fail.
        logger.info(
            "Ascend GDN RecoverSSM: async scheduling enabled; a graph's first "
            "execute is retried once after a device drain if it fails"
        )


def _prepare(worker) -> None:
    global _SELFTEST_DONE
    if not feature_enabled(getattr(worker, "vllm_config", None)):
        return
    _log_runtime_flags(worker)
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        return
    builders = list(_iter_builders(model_runner))
    if not builders:
        return

    total_rows = 0
    for builder in builders:
        context = builder._get_context()
        if context is None:
            raise RuntimeError(
                "Ascend GDN RecoverSSM: the KV cache is not bound before "
                "warmup, so the commit context cannot be prepared. Refusing to "
                "start and defer this to the first request."
            )
        rows = builder._accepted_buffer.numel()
        capacity = min(rows, context.commit_lens.numel())
        prepared = context.prepare(
            capacity=capacity,
            # ``state_indices`` is a column view of the [rows, num_spec+1] table,
            # so its stride is the table width; the prepared variant has to match
            # that or it compiles a different one.
            state_table_width=int(getattr(builder, "num_spec", 0)) + 1,
            request_mapping=True,
            device=context.checkpoints[0].device,
        )
        total_rows += len(prepared)
        logger.info(
            "Ascend GDN RecoverSSM: prepared %d commit variants for %d layers "
            "(rows 1..%d)", len(prepared), len(context.checkpoints), capacity
        )
    logger.info(
        "Ascend GDN RecoverSSM: startup preparation complete (%d variants)",
        total_rows,
    )

    if not _SELFTEST_DONE and selftest.enabled():
        _SELFTEST_DONE = True
        context = builders[0]._get_context()
        ckpt = context.checkpoints[0]
        ok = selftest.run(
            num_layers=len(context.checkpoints),
            num_heads=int(ckpt.shape[1]),
            head_k=int(ckpt.shape[3]),
            head_v=int(ckpt.shape[2]),
            spec_query_len=int(context.spec_query_len),
            device=ckpt.device,
            # Both precisions on purpose: the check builds its own tensors, so the
            # deployed cache dtype must not shrink the coverage. An fp32 run is the
            # one that proves the kernel math to 1e-5; a bf16 run only shows the
            # page rounding stays inside its own tolerance.
            dtypes=None,
        )
        conv_ok = selftest.run_conv(
            num_layers=len(context.checkpoints),
            num_heads=int(ckpt.shape[1]),
            head_k=int(ckpt.shape[3]),
            head_v=int(ckpt.shape[2]),
            spec_query_len=int(context.spec_query_len),
            device=ckpt.device,
            dtypes=None,
        )
        if not ok:
            logger.error(
                "Ascend GDN RecoverSSM selftest reported a MISMATCH; the committed "
                "state does not reproduce the sequential GDN recurrence on this "
                "build. The server keeps running -- this is a report, not a guard."
            )
        if not conv_ok:
            logger.error(
                "Ascend GDN RecoverSSM selftest reported a CONV MISMATCH; the "
                "convolution-history compaction does not keep the window that ends "
                "at the committed token. The server keeps running -- report only."
            )
