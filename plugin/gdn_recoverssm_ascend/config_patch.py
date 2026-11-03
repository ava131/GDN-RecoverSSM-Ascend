# SPDX-License-Identifier: Apache-2.0
"""Widen vLLM's RecoverSSM config gate for Qwen GDN on Ascend.

Upstream vLLM v0.30.0 rejects everything except Kimi KDA::

    self.cache_config.use_kda_recoverssm = self.num_speculative_tokens > 0
    if architecture not in ("KimiLinearForCausalLM", "KimiK3ForConditionalGeneration"):
        raise ValueError("RecoverSSM is only supported for Kimi-K3 KDA")

It also requires the model class to declare ``supports_replayssm``, which the
``Qwen3_5*`` classes do not. Ascend ships its own GDN layer, so we validate the
shapes the Ascend kernel supports and then enable the same downstream machinery
(``use_kda_recoverssm`` drives the generic ``RecoverSSMState`` lifecycle).
"""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

_GDN_ARCHITECTURES = (
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
)

SUPPORTED_HEAD_DIM = 128
MAX_SPEC_QUERY_LEN = 8

_APPLIED = False


def is_gdn_architecture(model_config) -> bool:
    return (
        model_config is not None
        and getattr(model_config, "architecture", None) in _GDN_ARCHITECTURES
    )


def _escape_hatch() -> str | None:
    if os.getenv("GDN_RECOVERSSM_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        return "GDN_RECOVERSSM_DISABLE"
    return None


def validate_gdn_recoverssm(config) -> None:
    """Fail fast on shapes this kernel family cannot serve."""
    text_config = config.model_config.hf_text_config
    k_dim = getattr(text_config, "linear_key_head_dim", None)
    v_dim = getattr(text_config, "linear_value_head_dim", None)
    if k_dim != SUPPORTED_HEAD_DIM or v_dim != SUPPORTED_HEAD_DIM:
        raise ValueError(
            "Ascend GDN RecoverSSM requires linear head dims K=V="
            f"{SUPPORTED_HEAD_DIM}, got K={k_dim} V={v_dim}"
        )

    if config.speculative_config is None:
        raise ValueError("Ascend GDN RecoverSSM requires speculative decoding")
    query_len = 1 + config.num_speculative_tokens
    if query_len > MAX_SPEC_QUERY_LEN:
        raise ValueError(
            "Ascend GDN RecoverSSM supports at most "
            f"{MAX_SPEC_QUERY_LEN - 1} speculative tokens, got "
            f"{config.num_speculative_tokens}"
        )

    if config.mamba_config.enable_stochastic_rounding:
        raise ValueError(
            "Ascend GDN RecoverSSM supports bfloat16/float32 SSM caches, not "
            "--enable-mamba-cache-stochastic-rounding"
        )
    from vllm.config.mamba import MambaBackendEnum

    if config.mamba_config.backend != MambaBackendEnum.TRITON:
        raise ValueError("Ascend GDN RecoverSSM requires --mamba-backend triton")

    cache = config.cache_config
    if cache.mamba_cache_mode not in ("none", "align"):
        raise ValueError(
            "Ascend GDN RecoverSSM supports only none and align Mamba cache "
            f"modes, got {cache.mamba_cache_mode}"
        )
    # RecoverSSM's post-sampling commit lives in the V2 model-state lifecycle
    # (``MambaHybridModelState.postprocess_state`` -> ``RecoverSSMState.commit_step``).
    # The V1 runner has no RecoverSSM hooks at all: verify would run and write
    # records, but nothing would ever commit, so the checkpoint would freeze and
    # the output would silently drift. This is required for **both** cache modes.
    if not config.use_v2_model_runner:
        raise ValueError(
            "Ascend GDN RecoverSSM requires the V2 model runner "
            "(export VLLM_USE_V2_MODEL_RUNNER=1); the V1 runner has no "
            "post-sampling commit hook, so the checkpoint would never advance"
        )
    block = cache.mamba_block_size
    if block is not None and block % query_len != 0:
        raise ValueError(
            "Ascend GDN RecoverSSM requires --mamba-block-size to be a multiple "
            f"of the speculative window ({query_len}), got {block}"
        )

    parallel = config.parallel_config
    if parallel.pipeline_parallel_size > 1:
        raise ValueError("Ascend GDN RecoverSSM currently requires PP=1")
    for name in (
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) > 1:
            raise ValueError(
                f"Ascend GDN RecoverSSM currently requires {name}=1"
            )
    if config.kv_transfer_config is not None:
        raise ValueError(
            "Ascend GDN RecoverSSM does not support KV transfer yet "
            "(P/D disaggregation, KV offloading)"
        )


def install() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from pydantic.dataclasses import rebuild_dataclass
    from vllm.config import VllmConfig

    from . import state_patch

    state_patch.install()
    original = VllmConfig.validate_mamba_cached_kernel

    @functools.wraps(original)
    def validate_mamba_cached_kernel(self):
        if not self.cache_config.use_replayssm:
            self.cache_config.use_kda_recoverssm = False
            return self

        if not is_gdn_architecture(self.model_config):
            # Kimi / GLM / Mamba2 keep the upstream behaviour.
            return original(self)

        reason = _escape_hatch()
        if reason is not None:
            logger.warning(
                "Ascend GDN RecoverSSM disabled (%s); per-draft state snapshots "
                "stay in use.",
                reason,
            )
            self.cache_config.use_kda_recoverssm = False
            from . import state_patch

            state_patch.disable()
            return self

        validate_gdn_recoverssm(self)
        self.cache_config.use_kda_recoverssm = self.num_speculative_tokens > 0
        # The state layout only gains the replay record when the feature is on.
        from . import state_patch

        state_patch.enable(1 + self.num_speculative_tokens)
        logger.info(
            "Ascend GDN RecoverSSM enabled: spec_query_len=%d, "
            "mamba_cache_mode=%s, mamba_block_size=%s",
            1 + self.num_speculative_tokens,
            self.cache_config.mamba_cache_mode,
            self.cache_config.mamba_block_size,
        )
        return self

    VllmConfig.validate_mamba_cached_kernel = validate_mamba_cached_kernel
    decorators = getattr(VllmConfig, "__pydantic_decorators__", None)
    if decorators is not None:
        decorators.model_validators["validate_mamba_cached_kernel"].func = (
            validate_mamba_cached_kernel
        )
    rebuild_dataclass(VllmConfig, force=True)
    _APPLIED = True


__all__ = ["install", "is_gdn_architecture", "validate_gdn_recoverssm"]
