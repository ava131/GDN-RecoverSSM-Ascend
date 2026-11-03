# 上机手册（A2 / 910B，MTP3，prefix cache）

> 目标：把 `plugin/` 装到 Ascend 机器上，用 `--use-replayssm` 跑通 Qwen3.8-27B + MTP3 + align。
> 全程**不改任何源码**；出问题可以用一个环境变量回退。

---

## 0. 容器/启动参数（最容易踩的三个坑）

### 0.1 `/dev/shm` 必须放大，否则引擎起不来

```
RuntimeError: Insufficient space in /dev/shm: 160 MiB required, 64 MiB free.
```

这是 Docker 默认 `--shm-size=64m` 导致的，**发生在模型加载之前**，与本插件无关。

```bash
docker run --shm-size=16g ...          # 或 --ipc=host
# K8s:
#   volumes: [{name: dshm, emptyDir: {medium: Memory, sizeLimit: 16Gi}}]
#   volumeMounts: [{name: dshm, mountPath: /dev/shm}]
```

### 0.2 `VLLM_USE_V2_MODEL_RUNNER=1` 是硬要求

RecoverSSM 的 **commit 挂在 V2 model-state 生命周期里**
（`MambaHybridModelState.postprocess_state` → `RecoverSSMState.commit_step`）。
V1 runner（`vllm/v1/worker/gpu_model_runner.py`）里 **没有任何 recoverssm 代码**：

```bash
grep -c recoverssm vllm/v1/worker/gpu_model_runner.py    # -> 0
```

所以在 V1 上会是这样：verify 照跑、记录照写，**但永远没有 commit** → 检查点冻结 →
输出越跑越偏，而且**不报错**。插件的配置校验现在会直接拒绝这种组合（fail fast）。

### 0.3 投机长度按 `Q=4` 配（`num_speculative_tokens=3`）

`num_speculative_tokens=1` ⇒ Q=2。按量化模型，**Q=2 时 RecoverSSM 的状态访存是 0.98×（反而更亏）**，
交叉点在 Q≈2.06。要看到收益必须 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`。

## 0. 前置检查（5 分钟）

```bash
# 1) 版本必须匹配
python -c "import vllm, vllm_ascend; print(vllm.__version__, vllm_ascend.__version__)"
#   期望：0.30.0 0.30.0

# 2) 芯片必须是 910B（A2）
python -c "import torch_npu; print(torch_npu.npu.get_soc_version())"

# 3) triton-ascend 版本
python -c "import triton; print(triton.__version__)"   # 期望 3.6.0.post2

# 4) 源码必须是干净的（本方案不改源码）
cd $VLLM_ASCEND_DIR && git status --short    # 期望为空
cd $VLLM_DIR        && git status --short    # 期望为空
```

任何一条不匹配，先解决再往下走。

---

## 1. 安装插件

```bash
cd /path/to/GDN-RecoverSSM-Ascend/plugin
python -m pip install -e .

# 确认 entry point 被 vLLM 看到
python - <<'PY'
from importlib.metadata import entry_points
eps = entry_points(group="vllm.general_plugins")
print([e.name for e in eps])
assert any(e.name == "gdn_recoverssm_ascend" for e in eps), "plugin not registered"
print("ok")
PY
```

---

## 2. 先用**关闭**状态跑通 baseline（关键一步）

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export GDN_RECOVERSSM_DISABLE=1          # 先关闭插件，确认基线本身是好的

vllm serve /models/Qwen3.8-27B \
  --tensor-parallel-size 4 --max-model-len 8192 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mamba-backend triton --mamba-block-size 128 \
  --additional-config '{"mamba_cache_mode":"align"}' \
  --use-replayssm
```

> 注意：`--use-replayssm` 在插件关闭时会被上游 validator 拒绝（只支持 Kimi）。
> 所以这一步**先不带 `--use-replayssm`**，只验证 baseline + align + MTP3 能跑：
>
> ```bash
> vllm serve ... --mamba-block-size 128 --additional-config '{"mamba_cache_mode":"align"}'
> ```
>
> 记录：`nvidia-smi` 等价物（`npu-smi info`）里的显存占用、TTFT、TPOT、接受长度。

---

## 3. 打开 RecoverSSM（先 none 模式，再 align）

### 3.1 先 `mamba_cache_mode=none`（排除 boundary 变量）

```bash
unset GDN_RECOVERSSM_DISABLE

vllm serve /models/Qwen3.8-27B \
  --tensor-parallel-size 4 --max-model-len 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mamba-backend triton \
  --use-replayssm
```

期望日志：

```
Ascend GDN RecoverSSM enabled: spec_query_len=4, mamba_cache_mode=none, mamba_block_size=None
Ascend GDN RecoverSSM commit context: 48 layers, spec_query_len 4
Ascend GDN RecoverSSM path taken: 8 spec rows, align=False, request_indices=False
```

**验收 1**：输出与第 2 步（无 RecoverSSM）在 `check_logprobs_close` 容差内一致。

### 3.2 再开 align（prefix cache）

```bash
vllm serve /models/Qwen3.8-27B \
  --tensor-parallel-size 4 --max-model-len 32768 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mamba-backend triton --mamba-block-size 128 \
  --additional-config '{"mamba_cache_mode":"align"}' \
  --use-replayssm
```

**验收 2**：日志出现 `align=True`；重复前缀命中后，第二轮的输出与第一轮一致。

---

## 3.3 怎么证明 RecoverSSM 真的生效（日志证据链）

插件在四个时机打日志，全部走 vLLM 的 `init_logger`，默认 INFO 就会出现在 server 的 stdout。

### 3.3 启动日志里就一行

插件只在**启动时**打一条 INFO，跟着 vLLM 自己的启动日志走（同一路 stdout）：

```bash
grep -i "RecoverSSM" server.log
```

```
INFO ... [patch_recoverssm_config.py] Ascend GDN RecoverSSM enabled: window=4 (num_spec=3), mamba_cache_mode=none, mamba_block_size=None, tp=4, mamba states=3 per layer (conv, checkpoint, replay), replay=[12, 4, 257] fp32, num_speculative_blocks=0
```

这一行同时说明五件事：

| 字段 | 含义 |
|---|---|
| `window=4 (num_spec=3)` | Q=4，投机窗口 |
| `mamba_cache_mode=none` | 当前缓存模式（`align` = 开了 prefix cache） |
| `mamba states=3 per layer` | **记录张量被追加了**（原来只有 conv + checkpoint 两个） |
| `replay=[12, 4, 257] fp32` | 每个 token 的恢复记录形状与精度（`[HV_tp, Q, V+K+1]`） |
| `num_speculative_blocks=0` | 每请求的状态槽从 Q 降到 1，容量收益生效 |

**没有这一行 = 没生效。** 排查顺序：

1. `--use-replayssm` 没带？→ 上游要求 `num_speculative_tokens > 0`；
2. 设了 `GDN_RECOVERSSM_DISABLE=1`？→ 会打 `disabled by GDN_RECOVERSSM_DISABLE`；
3. 插件没装进 vLLM 的环境？→ 完全没有任何输出，`pip show gdn-recoverssm-ascend` 确认；
4. 版本不匹配？→ 会打 `stays inert (原因)`。

### 3.3b 请求进来时的那一行（前 3 步）

第一个投机步会打一行，默认前 **3 步**，之后不再打（不会刷屏）：

```
Ascend GDN RecoverSSM step 1: prefill=0 decode=1 spec_rows=1 window=4 tokens=4 qsl_tail=4 state_idx=[12..12] cache_mode=none
```

| 字段 | 含义 |
|---|---|
| `prefill` / `decode` | 本步的 prefill / decode 行数；**prefill>0 就是 chunked-prefill 混合批** |
| `spec_rows` | 走投机路径的行数 |
| `tokens` | 本步送入 verify 的 token 数 |
| **`qsl_tail`** | `spec_query_start_loc` 的末值，**必须等于 `tokens`**；不等就会打 WARNING（这正是越界访程的前置条件） |
| `state_idx=[lo..hi]` | 本步用到的状态页号范围，应在 `[0, num_blocks)` 内 |

调这个行为：

```bash
export GDN_RECOVERSSM_TRACE_STEPS=3    # 默认 3；设 0 关闭；排查时可设 50
```

> 更早的版本还有"commit context / ACTIVE / 每 200 步 stats"等日志，已全部删掉。

### 3.4 想进一步确认运行时行为（可选，不改代码）

启动那一行只能证明"配置和内存布局对了"，不能证明"每步真的在 commit"。要确认这一点，
**不需要额外日志**，看两个外部指标即可：

```bash
# 1) 同一命令、开/关 --use-replayssm，比较启动日志里的可用 KV 容量
grep -iE "Available KV cache memory|GPU KV cache size|Maximum concurrency" server.log

# 2) 对比时延与接受长度
vllm bench serve --model /models/Qwen3.8-27B --num-prompts 200 --temperature 0
```

`--use-replayssm` 打开后，`num_speculative_blocks` 从 Q 变 0，每请求的状态槽少 Q−1 个，
可用 KV 容量应随之变大；接受长度应与关闭时基本一致（差 >5% 说明语义有问题）。

## 4. 正确性验收（必须在真机上做）## 4. 正确性验收（必须在真机上做）

| # | 用例 | 判据 | 怎么做 |
|---|---|---|---|
| 1 | 同 greedy 输入，开/关 RecoverSSM | `check_logprobs_close` 通过；接受长度不显著下降 | 相同 prompt、`temperature=0`、固定 seed，跑 200 条 |
| 1b | **不要**用 `num_speculative_tokens=0` 做"逐位一致"验证 | 说明 | Q=1 时配置校验会把 `use_kda_recoverssm` 置 False，路径根本不激活，测了等于没测 |
| 2 | 接受长度 a=0 的请求 | 检查点逐位不变 | 构造"必然被拒"的 draft（例如强 temperature + 故意错 token） |
| 3 | 全部接受 / 部分接受 | 与 FP32 顺序 oracle 相对误差 < 1e-5 | 用 `plugin/tests/test_math_oracle.py` 的公式在 NPU 上重算 |
| 4 | align 跨块边界 | `num_computed % 128 == 0` 与 `== 127` 两种都要覆盖 | 让 prompt 长度落在块边界前后 |
| 5 | `FULL_DECODE_ONLY` | 与 eager 一致 | `-O3 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` |
| 6 | 混合 batch（有 draft 的行 + 无 draft 的行） | 无 draft 的行走 stock path，不出错 | 并发两个请求，一个强制 0 draft |

---

## 5. 出问题怎么查、怎么退

| 现象 | 先看什么 | 处理 |
|---|---|---|
| 配置阶段直接报错 | 报错文案（我们全部 fail-fast） | 按文案调整：head dim、block size 倍数、PP/DCP/PCP、KV transfer |
| kernel 编译失败 | triton 的报错行 | 按 `docs/KERNEL_NOTES.md` §2 的 10 条 fallback 逐项替换 |
| UB 断言触发（"UB budget exceeded"） | 报错里的 tile 数值 | 换更小 `BV`（代码里已按 UB 自动降；若仍触发说明器件 UB 异常小） |
| 数值不对（输出乱） | 先关 align（`mamba_cache_mode=none`） | 若 none 正常 → boundary 折叠有问题；若 none 也乱 → 记录布局/索引有问题 |
| 一开就崩 | `npu-smi info` + CANN 日志 | `export GDN_RECOVERSSM_DISABLE=1` 回退，确认 baseline 仍正常 |

**一键回退**：

```bash
export GDN_RECOVERSSM_DISABLE=1     # 插件仍加载，但什么都不做
# 或彻底卸载
python -m pip uninstall gdn-recoverssm-ascend
```

---

## 6. 性能采样（确认收益方向）

在 B=32 / 128 / 256 三档下，对比"开/不开 RecoverSSM"：

| 指标 | 怎么取 |
|---|---|
| 持久 SSM cache 占用 | `npu-smi info` 显存 + vLLM 启动日志里的 KV cache 大小 |
| verify 延迟 | profiler 抓 `_gdn_recoverssm_verify_kernel` |
| commit 延迟 | profiler 抓 `_commit_gdn_state_kernel`（注意有两次 launch：final / boundary） |
| TPOT / 吞吐 | `vllm bench serve` |
| 接受长度 | 日志里的 acceptance length |

预期（按 `docs/GDN-RecoverSSM-实现方案.md` §5 的模型）：**Q=4 时状态访存约 1.6× 改善、容量释放约 (Q−1) 份**；A2 带宽更低，容量收益更明显，但 commit 那 288 MiB/请求/步的访存是真实成本——**第一版只要求"不退化"，不承诺加速**。
