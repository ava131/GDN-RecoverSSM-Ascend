# 现场是 v0.30.0rc1，我们对着 v0.30.0 正式 tag 分析

结论：**我们依赖的接口在 rc1 与正式 tag 上完全一致**，之前基于 tag 的分析对 rc1 同样成立；
差异集中在与本特性无关的地方。另有一处 rc1 特有、会直接影响 RecoverSSM 的差异（见第 3 节）。

## 1. 逐文件对比（rc1 vs v0.30.0）

| 文件 | 结果 |
|---|---|
| `vllm/v1/worker/gpu/model_states/recoverssm.py` | **完全一致** |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | **完全一致** |
| `vllm/v1/attention/backends/recoverssm_metadata.py` | **完全一致** |
| `vllm/model_executor/layers/mamba/mamba_utils.py`（状态 shape/dtype 计算器） | **完全一致** |
| `vllm/v1/attention/backends/gdn_attn.py` | **完全一致** |
| `vllm_ascend/ops/gdn_attn_builder.py`（我们继承的 builder、所有 spec 元数据语义） | **完全一致** |
| `vllm_ascend/worker/v2/aclgraph_utils.py`（图重放路径） | **完全一致** |
| `vllm_ascend/ops/triton/triton_utils.py`（`get_ub_size_bytes`） | **完全一致** |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | 差异 87 行，但 **GDN 接线结构相同**（`_GDN_PATCH_TARGET`、按值复制、`_gdn_init_with_packed_weight`） |
| `vllm_ascend/worker/v2/model_states/mamba_hybrid.py` | 差异 32 行，全部是 **KV transfer / bulk copy**（我们不支持 P/D，无关） |
| `vllm_ascend/ops/gdn.py` | 差异较大（738 → 897 行），但 `_forward_core` 签名、类继承、`num_accepted_tokens` 传参方式一致 |

所以：`model_runner.py:517`（现场）vs `:638`（正式 tag）**只是 rc1 的 `model_runner.py` 行号差异**，
不是异常信号。插件 `install()` 的依赖检查在 rc1 上全部通过。

## 2. rc1 与正式 tag 的 `gdn.py` 差异（不影响我们）

`_forward_core` 签名相同，`self_kv_cache[1]` 取状态相同。rc1 也有
`_supports_host_metadata_prefill` / `AscendGDNHostMetadataBuilder` / `npu_chunk_gated_delta_rule` 融合路径。

注意：我们的 backend 会**无条件**返回自己的 builder，因此即使 rc1 本来会选 fused host-prefill 的
metadata builder，我们也会走通用路径。这是**性能**层面的差异，不是正确性问题。

## 3. ⚠️ rc1 特有：stock 投机路径与 `num_speculative_blocks=0` 不兼容

rc1 的 spec 调用：

```python
# v0.30.0rc1, vllm_ascend/ops/gdn.py:573
ssm_state_indices=spec_state_indices_tensor.flatten(),
```

正式 tag 则是：

```python
ssm_state_indices=_flatten_recurrent_state_indices(
    spec_state_indices_tensor, query_spec.shape[1], spec_causal_conv1d_meta.query_start_loc,
),
```

在 `none` 模式下 `max_num_blocks_per_req = 1 + num_speculative_blocks = 1`，block table 只有 **1 列**，
`spec_state_indices_tensor` 形状是 `[rows, 1]`，`.flatten()` 得到 **rows** 个索引；
而 kernel 需要 **每 token 一个**索引（`rows * Q`）。

**后果**：在 rc1 上，`--use-replayssm` 打开后 stock 投机路径拿到的索引数不足 →
只有我们的 verify 接管那次调用才是正确的。这也从侧面证明：现场那次运行里 verify 必须是活的
（日志确实显示 `spec_rows=128` 在跑）。

**推论**：rc1 上不存在"关掉插件还能带 `--use-replayssm` 跑"的组合，A/B 对照必须去掉该 flag。
