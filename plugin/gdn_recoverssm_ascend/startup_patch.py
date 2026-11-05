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

logger = init_logger("vllm.gdn_recoverssm_ascend.startup")

_APPLIED = False


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
    _APPLIED = True


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
    if enabled:
        # Observed on A2 / CANN 9.1: with async scheduling the host runs ahead and
        # a graph that is executed for the FIRST time fails inside
        # CaptureModel::BuildSqCq -> SendSqe -> halStreamTaskFill (drvRetCode=17,
        # surfaced as 507899 by the next synchronize). The same workload with
        # --no-async-scheduling replays that very graph without error and at the
        # same throughput, so the asynchronous host run-ahead is the trigger.
        # Failing at startup is much better than failing mid-serving.
        raise RuntimeError(
            "Ascend GDN RecoverSSM requires synchronous scheduling: run with "
            "--no-async-scheduling. With async scheduling enabled a first-use "
            "full-graph replay fails in BuildSqCq (507899), and deleting the "
            "--async-scheduling flag does NOT disable it (the value stays None "
            "and vLLM resolves it to True)."
        )


def _prepare(worker) -> None:
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
