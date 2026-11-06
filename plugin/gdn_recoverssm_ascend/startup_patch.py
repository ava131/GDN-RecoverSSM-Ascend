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
        reserve = _record_scratch_bytes(getattr(self, "vllm_config", None))
        if not reserve:
            return available
        logger.info(
            "Ascend GDN RecoverSSM scratch reservation: %.1f MiB -- %s; "
            "deducted from available KV memory (%.1f GiB -> %.1f GiB) so the "
            "records cannot over-subscribe gpu_memory_utilization",
            reserve / (1024 * 1024),
            _record_scratch_desc(getattr(self, "vllm_config", None)),
            available / (1024**3),
            max(available - reserve, 0) / (1024**3),
        )
        return max(available - reserve, 0)

    NPUWorker.determine_available_memory = determine_available_memory
    _APPLIED = True


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
        rows = int(config.scheduler_config.max_num_seqs or 0)
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
            dtypes=(ckpt.dtype,),
        )
        if not ok:
            logger.error(
                "Ascend GDN RecoverSSM selftest reported a MISMATCH; the committed "
                "state does not reproduce the sequential GDN recurrence on this "
                "build. The server keeps running -- this is a report, not a guard."
            )
