# gdn-recoverssm-ascend

**RecoverSSM for Gated DeltaNet on Ascend NPU — shipped as a vLLM plugin. No source changes.**

`--use-replayssm` makes a speculative GDN model verify a window of `Q = 1 + MTP`
tokens from **one read-only checkpoint** and rebuild only the accepted state
after sampling, instead of storing one full recurrent state per draft position.

Upstream vLLM already has the generic RecoverSSM lifecycle and the Kimi-K3 KDA
implementation; vllm-ascend already runs KDA RecoverSSM on Ascend. This package
supplies what is missing for **Qwen GDN on Ascend**, without editing either tree.

---

## What it patches

| Target | Why |
|---|---|
| `vllm.config.VllmConfig.validate_mamba_cached_kernel` | upstream only allows Kimi KDA; widen to `Qwen3_5*` and set `cache_config.use_kda_recoverssm` |
| `AscendGatedDeltaNetAttention.__init__` | record `self.use_gdn_recoverssm` |
| `…get_state_shape` / `…get_state_dtype` | append the FP32 replay record `[HV_tp, Q, V+K+1]` as `kv_cache[2]` |
| `…get_attn_backend` | return the RecoverSSM metadata backend |
| `…_forward_core` | push the current layer onto a thread-local stack |
| `vllm_ascend.ops.gdn.recurrent_gated_delta_rule` | redirect **only** the speculative call (the one passing `num_accepted_tokens`) to `gdn_recoverssm_verify` |

The post-sampling commit needs **no** patch: vLLM v0.30.0 already calls
`RecoverSSMState.commit_step()` on the metadata object our backend builds, and
`MambaHybridModelState.postprocess_state()` drives it.

---

## Install

```bash
# on the Ascend host, inside the SAME environment as vllm-ascend v0.30.0
cd plugin
python -m pip install -e .

# sanity check: the entry point must be visible
python -c "from importlib.metadata import entry_points; \
print([e.name for e in entry_points(group='vllm.general_plugins')])"
# -> ['gdn_recoverssm_ascend', ...]
```

vLLM loads `vllm.general_plugins` in every process (engine core + workers) before
the engine is built, so no import site or `sitecustomize` tricks are needed.

## Run (A2 / 910B, MTP3, prefix cache on)

```bash
export VLLM_USE_V2_MODEL_RUNNER=1          # required for mamba_cache_mode=align

vllm serve /path/to/Qwen3.8-27B \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mamba-backend triton \
  --mamba-block-size 128 \
  --use-replayssm \
  --additional-config '{"mamba_cache_mode":"align"}'
```

Notes:

* `--mamba-block-size` must be a multiple of `Q = 4`.
* Start with `--mamba-cache-mode none` (drop `--additional-config`) to prove the
  verify/commit path end to end, **then** switch on align.
* The MTP head must be the model's native one (`mtp_num_hidden_layers: 1`).

## Kill switch

```bash
export GDN_RECOVERSSM_DISABLE=1   # plugin loads but does nothing
```
or simply omit `--use-replayssm`.

## What must be true for the feature to engage

* `--use-replayssm` **and** `num_speculative_tokens > 0`;
* architecture in `Qwen3_5{ForCausalLM,MoeForCausalLM,ForConditionalGeneration,MoeForConditionalGeneration}`;
* `linear_key_head_dim == linear_value_head_dim == 128`;
* `mamba_cache_mode ∈ {none, align}`; `align` requires the V2 model runner;
* `PP = DCP = PCP = 1`; no KV transfer;
* `--mamba-backend triton`; no `--enable-mamba-cache-stochastic-rounding`.

Any violation raises at config time with an explicit message (fail fast, no
silent fallback).

## Tests that run without an NPU

```bash
python tests/test_math_oracle.py      # record/commit semantics vs a sequential oracle
python tests/test_layer_patch_mock.py # plugin wiring against a stubbed vllm_ascend.ops.gdn
```

`test_math_oracle.py` pins the kernel semantics: verify must not modify the
checkpoint, commit must reproduce the sequential state for **every** accepted
length, and the align-mode boundary arithmetic must handle the
"exactly on a boundary" case.

## Verification on the NPU (after install)

1. `Q=1` (no drafts): RecoverSSM path must be bit-identical to the stock path.
2. `Q=4`, each accepted length `a ∈ [0, 4]`: reconstructed state vs a FP32
   sequential oracle, relative error < 1e-5.
3. `a = 0`: the checkpoint must be untouched (bit-for-bit).
4. align mode: requests that cross a block boundary must commit the boundary
   state correctly (test `num_computed % block_size == 0` and `== block_size-1`).
5. `FULL_DECODE_ONLY`: graph replay must match eager.
6. End to end: greedy output of Qwen3.8-27B + MTP3 must pass
   `check_logprobs_close` against the run without `--use-replayssm`.

## Known limits of this first version

* Supported head dims: `K = V = 128` only.
* `Q ≤ 8` speculative query length (`num_speculative_tokens ≤ 7`).
* The replay record lives **inside** the Mamba page (v0.30.0 design). Upstream
  main moved it into a per-request scratch buffer (vllm#63045); that is a
  follow-up.
* A decode row **without** drafts stays on the stock path (it writes its own
  state), which is consistent but not committed.
* A2 is not part of the KDA test matrix; treat the first NPU run as bring-up.
