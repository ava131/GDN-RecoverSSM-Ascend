# SPDX-License-Identifier: Apache-2.0
"""RecoverSSM for Gated DeltaNet on Ascend -- installed as a vLLM plugin.

This package changes **no** vLLM or vllm-ascend source file. It is loaded through
the ``vllm.general_plugins`` entry point (see ``pyproject.toml``), which vLLM
resolves in every process before the engine is built, and it monkey-patches:

1. ``vllm.config.VllmConfig.validate_mamba_cached_kernel`` -- allow the Qwen GDN
   (``qwen3_5``) architectures to use RecoverSSM and set
   ``cache_config.use_kda_recoverssm`` (config_patch.py).
2. ``vllm_ascend.ops.gdn.AscendGatedDeltaNetAttention`` -- the RecoverSSM state
   the attention backend and a forward hook (layer_patch.py), and the state
   shape/dtype calculators (state_patch.py).
3. ``vllm_ascend.ops.gdn.recurrent_gated_delta_rule`` -- redirects only the
   speculative-decode call to the RecoverSSM verify kernel (layer_patch.py).

The commit after sampling needs no patching: vLLM v0.30.0 already calls
``RecoverSSMState.commit_step`` on the metadata object our backend builds
(see ``vllm/v1/worker/gpu/model_states/recoverssm.py``).
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger("vllm.gdn_recoverssm_ascend." + __name__.rsplit(".", 1)[-1])

_INSTALLED = False


def install() -> None:
    """Entry point: called by vLLM's plugin loader in every process."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    if os.getenv("GDN_RECOVERSSM_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        # Silent by design: this runs in every process (API server, engine core and
        # each worker), so an INFO line here turns a baseline run into six lines of
        # RecoverSSM output. The plugin stays fully inert -- nothing below runs.
        logger.debug("gdn_recoverssm_ascend: disabled by GDN_RECOVERSSM_DISABLE")
        return

    try:
        import torch_npu  # noqa: F401
    except Exception:  # pragma: no cover - non-Ascend machines
        logger.debug("gdn_recoverssm_ascend: torch_npu unavailable, plugin inactive")
        return

    # Feature/version guard: this plugin only makes sense on a baseline that
    # already ships the generic RecoverSSM lifecycle (vLLM >= v0.30). On any
    # other tree it must stay inert instead of breaking engine startup.
    try:
        from vllm.config import VllmConfig
        from vllm.v1.attention.backends import recoverssm_metadata  # noqa: F401
        from vllm.v1.worker.gpu.model_states.recoverssm import (  # noqa: F401
            RecoverSSMState,
        )
        import vllm_ascend.ops.gdn as _ascend_gdn

        if not hasattr(VllmConfig, "validate_mamba_cached_kernel"):
            raise AttributeError("VllmConfig.validate_mamba_cached_kernel")
        if not hasattr(_ascend_gdn, "AscendGatedDeltaNetAttention"):
            raise AttributeError("vllm_ascend.ops.gdn.AscendGatedDeltaNetAttention")
        if not hasattr(_ascend_gdn, "recurrent_gated_delta_rule"):
            raise AttributeError("vllm_ascend.ops.gdn.recurrent_gated_delta_rule")
    except Exception as exc:  # noqa: BLE001 - any mismatch means "stay inert"
        logger.debug(
            "gdn_recoverssm_ascend: baseline lacks RecoverSSM support (%s: %s); "
            "plugin stays inert.",
            type(exc).__name__,
            exc,
        )
        return

    from . import config_patch, graph_probe, layer_patch, state_patch, startup_patch

    state_patch.install()
    config_patch.install()
    layer_patch.install()
    # The preparation is part of the feature contract, not an optional extra: if
    # it cannot be installed, the commit path would silently fall back to being
    # cold-initialised on the first request. Fail loudly instead.
    startup_patch.install()
    try:
        graph_probe.install()
    except Exception as exc:  # probing is optional
        logger.debug("gdn_recoverssm_ascend: graph probe unavailable (%s)", exc)
    logger.debug("gdn_recoverssm_ascend: installed")


__all__ = ["install"]
