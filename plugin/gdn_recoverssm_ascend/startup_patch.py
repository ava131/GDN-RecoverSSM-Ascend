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

Commit preparation compiles and registers only (``warmup=True`` plus binary
handle initialization), so it does not touch request state. Mandatory numerical
self-checks then execute on separately owned synthetic buffers and restore RNG.

Preparation covers commit signatures for actual schedulable record rows, with
the deployment's none/align mode and index dtypes. Graph-padded plan capacity is
not treated as additional real request capacity. Verify preparation also compiles
the actual checkpoint/record geometry and supported activation-layout signatures
without executing a model forward.
"""

from __future__ import annotations

import functools
import math
import os
from importlib.metadata import PackageNotFoundError, version

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
        result = original(self, *args, **kwargs)
        if feature_enabled(getattr(self, "vllm_config", None)):
            for builder in _iter_builders(getattr(self, "model_runner", None)):
                builder._gdn_serving_ready = True
        return result

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
            "(%.1f GiB -> %.1f GiB). The existing transient headroom defaults to "
            "512 MiB; it is a budget reserve, not measured graph-pool bytes. "
            "Override remains %s.",
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
    from .config_patch import is_gdn_architecture

    cache = getattr(config, "cache_config", None)
    return bool(
        getattr(cache, "use_kda_recoverssm", False)
        and is_gdn_architecture(getattr(config, "model_config", None))
    )


def _graph_margin_bytes(config) -> int:
    """Preserve the existing 512 MiB transient reserve and override contract.

    Sampling and graph allocations both need room beyond the KV pool.  This
    fixed reserve is not evidence that the records or the graph pool are that
    size; actual record bytes are deducted separately below.  Do not silently
    change a user's memory policy while optimizing kernels.
    """
    # The last working deployment uses this default.  A sort allocation failed
    # when it was zero, but that does not identify a measured 512 MiB peak or
    # justify charging the already reserved records twice.
    default_mb = 512
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
        key_heads = getattr(hf, "linear_num_key_heads", None)
        dim_k = getattr(hf, "linear_key_head_dim", None)
        dim_v = getattr(hf, "linear_value_head_dim", None)
        tp = config.parallel_config.tensor_parallel_size
        window = 1 + int(config.num_speculative_tokens or 0)
        if not (
            heads
            and key_heads
            and dim_k
            and dim_v
            and heads % tp == 0
            and key_heads % tp == 0
        ):
            return None
        layer_types = getattr(hf, "layer_types", None)
        if layer_types:
            layers = sum(1 for t in layer_types if t != "full_attention")
        else:
            layers = int(getattr(hf, "num_hidden_layers", 0) or 0)
        # Same row count the commit context allocates: only real requests write
        # records (padded graph rows carry a null state index and are skipped by
        # the kernels), so max_num_seqs is the true bound. Sizing this by the
        # padded graph batch made the reservation ~4x larger than the buffer.
        rows = int(config.scheduler_config.max_num_seqs or 0)
        if layers <= 0 or rows <= 0:
            return None
        return {
            "rows": rows,
            "layers": layers,
            "heads": heads // tp,
            "key_heads": key_heads // tp,
            "window": window,
            "width": dim_v + dim_k + 1,
            "key_dim": dim_k,
            "value_dim": dim_v,
            "key_dtype": str(config.model_config.dtype),
            "key_bytes": 2
            if str(config.model_config.dtype)
            in ("torch.float16", "torch.bfloat16", "float16", "bfloat16")
            else 4,
        }
    except Exception:
        return None


def _record_scratch_bytes(config) -> int:
    geo = _record_geometry(config)
    if geo is None:
        return 0
    # One FP32 correction and decay per value head, but only one key per
    # grouped-query key head.  Keys retain the already normalized input dtype.
    return (
        geo["rows"]
        * geo["layers"]
        * geo["window"]
        * (
            geo["heads"] * geo["value_dim"] * 4
            + geo["key_heads"] * geo["key_dim"] * geo["key_bytes"]
            + geo["heads"] * 4
        )
    )


def _record_scratch_desc(config) -> str:
    geo = _record_geometry(config)
    if geo is None:
        return "geometry unavailable"
    return (
        "rows={rows}, layers={layers}, value_heads={heads}, key_heads={key_heads}, "
        "window={window}, typed-SoA: correction/decay=fp32 key={key_dtype}, "
        "per-request temporaries reused every step"
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
            if not (
                hasattr(builder, "_accepted_buffer")
                and hasattr(builder, "_get_context")
            ):
                continue
            group_id = getattr(group, "kv_cache_group_id", None)
            if group_id is not None:
                builder._gdn_kv_cache_group_id = int(group_id)
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
            "Ascend GDN RecoverSSM: async scheduling enabled; graph first "
            "execution is prepared before readiness; serving replay failures "
            "are not automatically retried"
        )


def _prepare(worker) -> None:
    global _SELFTEST_DONE
    if not feature_enabled(getattr(worker, "vllm_config", None)):
        return
    worker._gdn_patch_prepared = False
    from . import diagnostics, sampling_patch

    diagnostics.configure(worker)
    sampling_patch.prepare(worker)
    _log_versions()
    _log_runtime_flags(worker)
    compilation = getattr(worker.vllm_config, "compilation_config", None)
    mode = getattr(compilation, "cudagraph_mode", None)
    if mode is not None and not str(mode).endswith("NONE"):
        from . import graph_probe

        if not graph_probe._APPLIED:
            raise RuntimeError(
                "RecoverSSM graph startup adapter was not installed; refusing "
                "to defer graph first execution to serving"
            )
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        raise RuntimeError("RecoverSSM enabled GDN has no model runner to prepare")
    builders = list(_iter_builders(model_runner))
    if not builders:
        raise RuntimeError(
            "RecoverSSM enabled GDN has no recovery metadata builders; "
            "refusing startup readiness without verify/commit preparation"
        )

    # op-dev uses an ahead-of-time AscendC library for verify/grouped fold.
    # Build/load before capture. A compiler/registration failure must not turn
    # this deployment silently back into the old Triton implementation.
    from . import ascendc_backend

    ascendc_backend.initialize()

    total_rows = 0
    verify_variants = 0
    actual_record_bytes = 0
    req_states = getattr(model_runner, "req_states", None)
    computed_owner = getattr(req_states, "num_computed_tokens", None)
    computed_tokens = getattr(computed_owner, "gpu", None)
    for builder in builders:
        context = builder._get_context()
        if context is None:
            raise RuntimeError(
                "Ascend GDN RecoverSSM: the KV cache is not bound before "
                "warmup, so the commit context cannot be prepared. Refusing to "
                "start and defer this to the first request."
            )
        rows = builder._accepted_buffer.numel()
        capacity = min(rows, context.commit_lens.numel(), context.record_rows)
        config = worker.vllm_config
        align = config.cache_config.mamba_cache_mode == "align"
        block_size = int(builder.kv_cache_spec.block_size) if align else None
        block_width = (
            max(1, math.ceil(config.model_config.max_model_len / block_size))
            if align
            else None
        )
        block_tables = getattr(model_runner, "block_tables", None)
        input_tables = getattr(block_tables, "input_block_tables", None)
        group_id = getattr(builder, "_gdn_kv_cache_group_id", None)
        table = None
        if input_tables is not None and group_id is not None:
            table = input_tables[group_id]
            if align:
                block_width = int(table.shape[1])
        import torch

        state_index_dtype = table.dtype if table is not None else torch.int32
        query_buffer = getattr(builder, "spec_query_start_loc", None)
        query_dtype = getattr(query_buffer, "dtype", torch.int32)
        verify_prepared = context.prepare_verify(
            activation_dtype=config.model_config.dtype,
            state_table_width=int(getattr(builder, "num_spec", 0)) + 1,
            device=context.checkpoints[0].device,
        )
        verify_variants += len(verify_prepared)
        prepared = context.prepare(
            capacity=capacity,
            # ``state_indices`` is a column view of the [rows, num_spec+1] table,
            # so its stride is the table width; the prepared variant has to match
            # that or it compiles a different one.
            state_table_width=int(getattr(builder, "num_spec", 0)) + 1,
            request_mapping=True,
            device=context.checkpoints[0].device,
            align_mode=align,
            mamba_block_size=block_size,
            block_table_width=block_width,
            request_index_dtype=builder.spec_sequence_indices.dtype,
            state_index_dtype=state_index_dtype,
            query_index_dtype=query_dtype,
            computed_dtype=getattr(computed_tokens, "dtype", torch.int32),
        )
        total_rows += len(prepared)
        logger.info(
            "Ascend GDN RecoverSSM: prepared %d commit variants for %d layers "
            "(rows 1..%d)",
            len(prepared),
            len(context.checkpoints),
            capacity,
        )
        actual_record_bytes += _log_context_geometry(builder, context, capacity)
    reserved_record_bytes = _record_scratch_bytes(worker.vllm_config)
    if actual_record_bytes > reserved_record_bytes:
        raise RuntimeError(
            "RecoverSSM record allocation exceeds its KV-budget reservation: "
            f"actual={actual_record_bytes}, reserved={reserved_record_bytes}"
        )
    logger.info(
        "GDN_RECORD_BUDGET actual_bytes=%d reserved_bytes=%d transient_reserve_bytes=%d PASS",
        actual_record_bytes,
        reserved_record_bytes,
        _graph_margin_bytes(worker.vllm_config),
    )
    from .grouped_commit import prepare_grouped_commit

    grouped_variants = prepare_grouped_commit(
        builders, model_runner.block_tables, num_computed_tokens=computed_tokens
    )
    logger.info(
        "Ascend GDN RecoverSSM: startup preparation complete "
        "(%d verify signatures, %d per-group commit variants, "
        "%d grouped variants; capture sizes unchanged)",
        verify_variants,
        total_rows,
        grouped_variants,
    )

    if not _SELFTEST_DONE:
        from . import ascendc_selftest

        ascendc_selftest.run(device=builders[0]._get_context().checkpoints[0].device)
        context = builders[0]._get_context()
        ckpt = context.checkpoints[0]
        ok = selftest.run(
            num_layers=len(context.checkpoints),
            num_heads=int(ckpt.shape[1]),
            num_key_heads=int(context.num_key_heads),
            key_dtype=context.key_dtype,
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
                "build. Startup readiness is refused."
            )
        if not conv_ok:
            logger.error(
                "Ascend GDN RecoverSSM selftest reported a CONV MISMATCH; the "
                "convolution-history compaction does not keep the window that ends "
                "at the committed token. Startup readiness is refused."
            )
        if not ok or not conv_ok:
            raise RuntimeError("RecoverSSM startup numerical self-check failed")
        from .grouped_commit import run_grouped_selftest

        run_grouped_selftest(
            device=ckpt.device,
            state_dtype=ckpt.dtype,
            key_dtype=context.key_dtype,
            num_value_heads=int(ckpt.shape[1]),
            num_key_heads=int(context.num_key_heads),
            key_dim=int(ckpt.shape[3]),
            value_dim=int(ckpt.shape[2]),
            spec_query_len=int(context.spec_query_len),
        )
    ascendc_backend.assert_ready()
    _SELFTEST_DONE = True
    worker._gdn_patch_prepared = True


def _log_context_geometry(builder, context, capacity: int) -> int:
    """Describe actual bound tensors without reading any device contents."""
    state = context.checkpoints[0]
    records = context.records
    key_dtype = getattr(records[0], "key_dtype", "legacy-fp32")
    actual_bytes = int(getattr(context, "record_bytes", 0))
    if not actual_bytes:
        from .kernels import record_nbytes

        actual_bytes = sum(record_nbytes(record) for record in records)
    logger.info(
        "GDN_PREPARED_CONTEXT layers=%d state_shape=%s state_stride=%s "
        "state_dtype=%s record_key_dtype=%s actual_record_bytes=%d "
        "record_rows=%d plan_rows=%d commit_rows_prepared=%d "
        "cache_mode=%s available_strategies=typed-SoA/direct-output/grouped-commit "
        "capture_geometry=unchanged",
        len(context.checkpoints),
        tuple(state.shape),
        tuple(state.stride()),
        state.dtype,
        key_dtype,
        actual_bytes,
        context.record_rows,
        context.commit_lens.numel(),
        capacity,
        builder.vllm_config.cache_config.mamba_cache_mode,
    )
    return actual_bytes


def _log_versions() -> None:
    """Record installed package versions without importing extra device code."""
    packages = {}
    for name in (
        "gdn-recoverssm-ascend",
        "vllm",
        "vllm-ascend",
        "torch",
        "torch-npu",
        "triton-ascend",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "distribution-metadata-unavailable"
    logger.info(
        "GDN_RUNTIME packages=%s adapter_ABI=Ascend-0.30rc1 "
        "source_pair=ascend:5f828ad/vllm:ced6857 "
        "source_pair_status=adapter-reference-not-installed-sha",
        packages,
    )
