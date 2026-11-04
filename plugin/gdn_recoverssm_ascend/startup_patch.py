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


def _prepare(worker) -> None:
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
