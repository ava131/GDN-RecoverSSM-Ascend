# GDN RecoverSSM on Ascend — 实施计划（v0.30.0 基线）

> 目标：仿照 vllm-ascend 的 **KDA RecoverSSM**，在 Ascend 上实现 **GDN RecoverSSM**，对齐 issue [#19987](https://github.com/vllm-project/vllm-ascend/issues/19987)。
> 本文件是**开工前的规划**，不包含实现代码。等确认后再动手。
> 规划日期：2026-11-03　|　基线：vLLM `v0.30.0` + vllm-ascend `v0.30.0`

---

## 0. 版本现实（先说清，因为它改变了整个计划）

我核对了两个 tag 的实际内容：

| 事实 | 证据 |
|---|---|
| **vllm-ascend `v0.30.0`（2026-10-29 发布）里没有 KDA RecoverSSM** | `vllm_ascend/ops/triton/kda/recoverssm.py`、`models/glm5next/recoverssm_config.py`、`patch/platform/patch_replayssm_config.py` 全部 **404**；`ops/triton/kda/` 下只有基础算子（`chunk_delta_h.py`、`conv_state.py`、`cumsum.py`、`fused_recurrent_kda.py`、`gate.py`、`kda.py`、`l2norm.py`、`output_writeback.py`、`solve_tril.py`） |
| **KDA RecoverSSM 是 0.31 线合入的** | `[2/N] Add GLM KDA RecoverSSM framework integration`（#20131）merged **2026-10-29 01:34**，而 v0.30.0 的分支早于此 |
| **vLLM `v0.30.0`（2026-09-22）里有完整的 KDA RecoverSSM Triton 实现** | 已 clone 到本地并核对：`vllm/models/kimi_k3/nvidia/ops/recoverssm.py` **1067 行 / 6 个 `@triton.jit` kernel**（`_kda_gate`、`_kda_recurrent_step`、`_kda_recoverssm_verify_kernel`、`_prepare_commit_plan_kernel`、`_compact_conv_state_kernel`、`_commit_kda_state_kernel`）；`vllm/config/cache.py` 已有 `use_replayssm` / `use_kda_recoverssm` |
| **vLLM 至今没有 GDN RecoverSSM** | #61309 / #58863 仍 open；vLLM 已发到 `v0.33.0` |

### 0.2 main 与 v0.30.0 到底差在哪（实测）

| 维度 | v0.30.0 | main（2026-11-03） | 对我们的影响 |
|---|---|---|---|
| 落后程度 | — | **642 个 commit / 300 个文件** | 不小，但不是"另一个项目" |
| **依赖版本** | torch-npu **2.10.0.post7**、triton-ascend **3.6.0.post2** | **完全相同** | ✅ **环境零差异**（我之前把 v0.31.0rc1 的 post4/3.2.2 记混了） |
| **通用 RecoverSSM 生命周期** | ✅ **已有**：`vllm/v1/worker/gpu/model_states/recoverssm.py`（`RecoverSSMState.record_step` / `commit_step`），`MambaHybridModelState` 里由 `use_kda_recoverssm` 控制 | 同 | ✅ **最关键的底座已经在 0.30.0 里** |
| **Ascend 侧的消费点** | ✅ **已有**：`patch/worker/patch_v2/patch_model_runner.py` 传 `use_replayssm`；`_310p/worker/v2/model_state.py` 已在调 `self.recoverssm.commit_step(...)`；`patch_bind_kv_cache.py` 有 `share_replayssm_ring_trackers` | 同 | ✅ 说明"谁来调 commit"这件事 0.30.0 已经接好了 |
| **KDA 具体算子** | ❌ 缺（`ops/triton/kda/recoverssm.py`） | ✅ 1451 行 | 缺的**只是算子 + 模型接线** |
| `ops/gdn.py` | 897 行 | 929 行（+46/−14，主要是 `vllm_version_is("0.32.0")` 兼容） | 差异小 |
| `ops/gdn_attn_builder.py` | 1022 行 | 1336 行（+376/−62） | **差异大，不能直接 cp** |
| `worker/v2/model_states/mamba_hybrid.py` | 357 行 | 498 行（+154/−13） | 中等 |
| 模型支持（Qwen3.8） | ✅ `qwen3_5` 路径 | ✅ 同 | 无差异 |

**结论：0.30.0 缺的是"算子 + 接线"，不缺"机制"。** 而算子是可以照着 main 写的——它只 import 6 个符号，**这 6 个在 0.30.0 里全部存在**（已本地核实）：

```python
is_conv_state_dim_first()          # vllm/model_executor/layers/mamba/mamba_utils.py:48  ✅
NULL_BLOCK_ID                      # vllm/v1/attention/backends/utils.py:47           ✅
is_950()                           # vllm_ascend/device/device_config.py:61           ✅
get_ub_size_bytes()                # vllm_ascend/ops/triton/triton_utils.py:112       ✅
init_device_properties_triton()    # vllm_ascend/ops/triton/triton_utils.py:71        ✅
triton / tl                        # vllm.triton_utils                                  ✅
```
而且 main 的那个 1451 行文件里**没有硬编码 glm5next**（grep 无命中），模型耦合全在 `models/glm5next/recoverssm_metadata.py` 里 ⇒ **算子可以近乎原样落到 0.30.0，只要重接模型层**。

### 三条可选路线

| 路线 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A. 严格 v0.30.0** | 在 0.30.0 上做，参考 **vLLM 0.30.0 自带的 KDA Triton 实现（1067 行，在 clone 里）** | 版本由你指定；**算法层参考完整**（就是 Triton，不是 CuTe） | **Ascend 侧的 UB/tiling 实现没有可抄的**（那部分在 0.31 线），多花 3–5 人天 |
| **B. 用 v0.31.0rc1 / main** | 直接站在现成的 Ascend KDA 实现上 | 有 1451 行现成实现可抄 | 偏离你指定的版本 |
| **C. v0.30.0 + vendor 参考**（**推荐**） | 主仓仍用 v0.30.0，但把 **main 上的 KDA 实现**和**上游 GDN 实现**作为参考材料放进 `docs/reference/` | 版本满足要求，参考材料齐全 | 参考代码与基线不完全一致，移植时要小心 API 差异 |

**本文按 C 走**（参考 main 的实现，落在 v0.30.0）。依据见 §0.2：0.30.0 已经具备 RecoverSSM 的**生命周期与消费点**，缺的只是算子实现，而算子只依赖 6 个符号且全部存在。

> 路线 B（上 main）的唯一好处是"KDA 现成可跑"，但我们要交的是 **GDN**，KDA 只是模板——**KDA 能不能在 0.30.0 上跑起来，其实无关紧要**。为它把整个基线迁到 main，反而引入 642 个 commit 的移动靶风险。

---

## 1. 目标与验收标准（Definition of Done）

### 1.1 功能目标

| # | 目标 | 验收 |
|---|---|---|
| F1 | GDN 的 **verify**：一次前向验证 `Q = 1 + num_spec` 个 token，写输出 + 写记录，**检查点只读** | 输出与原生路径在容差内一致 |
| F2 | GDN 的 **commit**：采样后按被接受前缀重建检查点，**只写一次** | 与 FP32 顺序执行 oracle 相对误差 < 1e-5 |
| F3 | **卷积状态压缩**（conv history 左对齐） | 压缩后 conv 状态与顺序执行一致 |
| F4 | **记录移出 KV page**（per-layer scratch，按 `max_num_reqs` 行） | Mamba page 大小回到"stock conv + SSM" |
| F5 | **MRV2 集成**（eager 先通） | Qwen3.8-27B + MTP3 端到端跑通 |
| F6 | **NPU Graph** 可跑 | `FULL_DECODE_ONLY` 下结果一致 |
| F7 | 复用 `--use-replayssm` 开关，不新增用户配置 | 只加 `--additional-config` 里的内部开关 |

### 1.2 正确性验收矩阵（来自 RFC #19987）

| 维度 | 用例 | 判据 |
|---|---|---|
| 数值 | 独立 FP32 递推 oracle，逐前缀 / 逐窗口 | 相对误差 < 1e-5 |
| 接受长度 | `a = 0 / 1 / … / Q` | 与 oracle 一致 |
| 立即拒绝 | `a = 0` | 检查点**逐位不变** |
| 连续多轮 | ≥ 3 轮 verify/commit | 游标 / 状态不漂移 |
| 多请求 | 不同 accepted 数、乱序、slot 复用 | 无串扰 |
| 空行 / 填充行 | inactive 请求、graph padding | 只输出 0，不写记录、不写状态 |
| 卷积 | conv history 压缩 | 与 oracle 一致 |
| 精度 | FP32 / BF16 状态 | 分别给容差 |
| 图 | eager vs `FULL_DECODE_ONLY` | 输出一致 |
| align 边界 | `steps % block == 0` 与 `== block-1` | boundary 状态正确（**最容易错的地方**） |

### 1.3 性能验收（不许只看 TPOT）

| 指标 | 目标 |
|---|---|
| 状态占用 | Q 份 → 1 份（**每请求每层 3 MiB → 3.05 MiB，但只占 1 个槽**） |
| verify 延迟 | 记录写入开销 < 该层状态一次完整写的 20% |
| commit 延迟 | < 该层状态一次完整写的 2 倍 |
| 端到端 TPOT | **不退化**（第一版目标），后续再追收益 |
| 接受长度 | 无回归（否则说明语义错了） |

> 注意：**第一版的目标是"正确 + 不退化"，不是"变快"**。按 §5 的量化，Q=4 时带宽账只有 1.6×，而 commit 本身要花 288 MiB 访存；能不能转正取决于 `f`（状态访存占比）和 UB 效率。把它当"先拿容量、再谈速度"的项目更现实。

---

## 2. 参考物地图（写代码前必须先读透的 6 份材料）

| # | 材料 | 位置 | 行数 | 用途 |
|---|---|---|---|---|
| **R0** | **vLLM 0.30.0 自带的 KDA RecoverSSM（Triton）** | clone 里：`vllm/vllm/models/kimi_k3/nvidia/ops/recoverssm.py`（副本见 `docs/reference/`） | **1067** | ⭐ **主要抄写对象**：4 个 kernel 的完整骨架、commit plan、conv 压缩。**就在基线里，不用跨版本移植** |
| R1 | Ascend KDA RecoverSSM 算子（main） | `docs/reference/kda_recoverssm_ascend.py` | 1451 | **Ascend 化要点**：UB 预算常量、tile 选择、地址表、null 行、boundary 两段式 |
| R2 | KDA RecoverSSM 设计文档（main） | `docs/reference/kda_recoverssm_design.md` | 355 | 公式、参数表、支持范围、tile 说明 |
| R3 | 上游 GDN RecoverSSM（#61309） | `docs/reference/recoverssm_gdn_upstream.py` | 702 | **GDN 的算法与 kernel 切分**：标量 gate、`HV//H` 分组、`V+K+1` 记录布局 |
| R4 | 上游共享算子（#61309） | `docs/reference/recoverssm_ops_upstream.py` | 200 | commit plan + conv 压缩的参考实现 |
| R5 | Ascend MRV1 GDN 原型（#18765） | `gdn_recover_ascend_mrv1.py` + `gdn_recover_batch_ascend_mrv1.py` | 399 + 177 | **已验证过的 GDN 数学**（3.188 MiB vs 12 MiB）+ conv 压缩 |
| R6 | GLM 的元数据 / 配置 / patch（main） | `glm5next_recoverssm_metadata.py`、`recoverssm_config.py`、`patch_replayssm_config.py`、`patch_kda_recoverssm.py` | 161 / 38 / 26 / 15 | MRV2 接线范式 |

**另外必读的基线代码**（不在 docs 里，在 clone 的源码里）：

以下行号已在本地 clone 中核对（v0.30.0）：

```
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py   2062 行  # GDN 层本体
vllm/model_executor/models/qwen3_5.py                            750 行  # 模型接线
vllm/models/kimi_k3/nvidia/ops/recoverssm.py                    1067 行  # ★ KDA RecoverSSM（Triton）
vllm_ascend/ops/gdn.py                                           897 行  # Ascend 的 GDN 接入
vllm_ascend/ops/gdn_attn_builder.py                             1022 行  # 元数据构建
vllm_ascend/worker/v2/model_states/mamba_hybrid.py               357 行  # MRV2 钩子
vllm_ascend/ops/triton/kda/kda.py                               1534 行  # KDA 的 chunk 实现（对照）
vllm_ascend/ops/triton/kda/solve_tril.py                         402 行  # 三角求逆（可选复用）
```

---

## 3. 逐文件改动清单（这是"改哪里"的答案）

### 3.1 新增文件（5 个）

| 文件 | 行数估算 | 内容 | 抄谁 |
|---|---|---|---|
| `vllm_ascend/ops/triton/mamba/gdn_recoverssm.py` | **800–1050** | 4 个 kernel + 2 个公共接口 + 1 个 CommitContext | R1 结构 + R3 数学 |
| `vllm_ascend/models/qwen3_5/recoverssm_metadata.py` | **170–220** | 层参数 / 页 / stride 组织，`create()` / `commit()` | R6 的 glm5next 版 |
| `vllm_ascend/models/qwen3_5/recoverssm_config.py` | **40–50** | `is_qwen_recoverssm_model` / `validate_qwen_recoverssm` | R6 的 config 版 |
| `vllm_ascend/patch/worker/patch_gdn_recoverssm.py` | **20–30** | worker 侧绑定 | R6 的 patch 版 |
| `tests/e2e/pull_request/one_card/test_gdn_recoverssm.py` | **500–700** | UT + NPU 算子测试 + 端到端 | R5 的测试结构 |

### 3.2 修改文件（8 个）

| 文件 | 改动 | 行数估算 | 说明 |
|---|---|---|---|
| `vllm_ascend/ops/gdn_attn_builder.py` | 加恢复元数据分支 | +25 / −2 | 对齐 KDA 的 +15/−2 |
| `vllm_ascend/ops/gdn.py` | 记录张量布局、cache spec、dtype | +60 / −10 | 对齐 #18765 的 +73/−15 |
| `vllm_ascend/patch/platform/patch_replayssm_config.py` | 扩展到 qwen3_5 架构判定 | +15 / −2 | 现有 26 行是 GLM 专用 |
| `vllm_ascend/worker/v2/model_states/mamba_hybrid.py` | MRV2 钩子 | +5 | KDA 是 +2 |
| `vllm_ascend/ascend_config.py` | 内部开关 `enable_gdn_recover` | +12 | 对齐 #18765 |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | 模型 patch 注册 | +5 | — |
| `vllm_ascend/platform.py` / `worker/worker.py` | 注册 | +8 | — |
| `docs/source/user_guide/feature_guide/*.md` | 用户文档 | +60 | 对齐 KDA 的 30 行 + #18765 的 67 行 |

**合计：新增约 1530–2010 行，修改约 210 行。**

### 3.3 明确不做（第一版）

| 排除项 | 理由 |
|---|---|
| PLE（`ple_recoverssm.py`，344 行） | Qwen4Exp 专用，与本目标无关 |
| grouped head 快路径（A5） | 先要正确性，性能优化后置 |
| KV transfer / P-D 分离 | RFC 要求"单独验证" |
| prefix caching（align） | 第一版先 `mamba_cache_mode=none`，第二版再加 boundary |
| PP > 1 / DCP > 1 | RFC 当前限制 |

---

## 4. 批次与里程碑（含每个批次的产出与验收）

### M0 — 准备（0.5–1 周）

| 项 | 内容 |
|---|---|
| 产出 | ① vendor 好的 `docs/reference/`；② 基线性能数据；③ **FP32 顺序执行 oracle** |
| 动作 | 跑通 v0.30.0 上 Qwen3.8-27B + MTP3（不开 RecoverSSM），记录：state 占用、TPOT、acceptance length；用纯 torch 写 oracle（`for t in range(Q): S = decay*S + ...`） |
| 验收 | oracle 与原生路径在 Q=1 时逐位一致 |
| 风险 | 0.30.0 上 Qwen3.8-27B 能不能跑通（`qwen3_5` 架构支持情况要确认） |

### M1 — verify 算子（1.5–2 周）

| 项 | 内容 |
|---|---|
| 产出 | `_gdn_verify_kernel` + `gdn_recoverssm_verify()` |
| 关键点 | ① 标量 gate：`g = -exp(A_log[h]) * softplus(a + dt_bias[h])`，`A_log` 是 `[HV]`、`dt_bias` 是 `[HV]`（**不是 KDA 的 `H*K`**）；② `pid_h = pid_hv // (HV // H)` 的 GQA 分组；③ 记录写 `[delta(V), k̂(K), decay(1)]`；④ null 行只输出 0；⑤ Q 用 `tl.static_range` |
| 验收 | 单 head / 单请求 / Q=1..8，输出与 oracle 一致（相对误差 < 1e-5） |
| 风险 | Triton-Ascend 的 `softplus` 与 `exp` 精度；`HV//H` 导致的 program 数变化 |

### M2 — commit + plan + conv（1.5–2 周）

| 项 | 内容 |
|---|---|
| 产出 | `_prepare_commit_plan_kernel`、`_compact_conv_state_kernel`、`_commit_gdn_state_kernel`、`GDNRecoverSSMCommitContext` |
| 关键点 | ① plan 与 conv 内核**基本照抄 KDA**（与算子无关）；② commit 的反向折叠把 KDA 的 `decay[None, :]` 换成标量；③ `a=0` 直接 return；④ 跨层一次 launch（地址表） |
| 验收 | 状态重建与 oracle 一致；`a=0` 时逐位不变；conv 压缩正确 |
| 风险 | **boundary 逻辑**（第一版先不做，第二版再加）|

### M3 — 缓存与元数据（~1 周）

| 项 | 内容 |
|---|---|
| 产出 | `recoverssm_metadata.py`（qwen3_5 版）、cache spec、记录 scratch buffer |
| 关键点 | 记录**不放进 Mamba page**（对齐 #63045）：每层一块 `[max_num_reqs, Q, HV, V+K+1]` FP32；conv 状态保持"stock + 加宽 Q" |
| 验收 | page 大小回到 stock；记录行数 = `max_num_reqs`；rebind 后上下文可重建 |
| 风险 | v0.30.0 的 cache config 结构可能与 main 不同（0.31 才重构过） |

### M4 — MRV2 集成（~1 周）

| 项 | 内容 |
|---|---|
| 产出 | config 校验、patch、模型接线、`--use-replayssm` 分支 |
| 关键点 | ① `is_qwen_recoverssm_model` 判 `qwen3_5`；② 复用 `use_kda_recoverssm` 这个字段（或新增 `use_gdn_recoverssm` 但**不改用户接口**）；③ MRV2 的 `model_states/mamba_hybrid.py` 钩子 |
| 验收 | Qwen3.8-27B + MTP3 eager 端到端跑通，输出正确 |
| 风险 | 0.30.0 的 MRV2 接口与 main 差异；`mamba_hybrid.py` 可能还没有 KDA 的钩子 |

### M5 — 测试与图（1.5–2 周）

| 项 | 内容 |
|---|---|
| 产出 | UT（oracle 对比 + 边界矩阵）、NPU 算子测试、`FULL_DECODE_ONLY` 图测试 |
| 验收 | §1.2 的 10 条全过 |
| 风险 | 图捕获与"记录 scratch"的地址稳定性；padding 行 |

### M6 — 性能与调优（1–1.5 周）

| 项 | 内容 |
|---|---|
| 产出 | UB tile 调参、commit 跨层合并、延迟提交（可选）、性能报告 |
| 验收 | TPOT 不退化；发 PR |

**总计：7–9.5 周（单人串行）**，两人并行约 4.5–5.5 周。

---

## 5. 人天估算（细化到批次）

| 批次 | 内容 | 人天 | 依据 |
|---|---|---|---|
| M0 | 环境 + baseline + oracle | **3–5** | oracle 不难，但 0.30.0 上跑通 Qwen3.8 可能要修 |
| M1 | verify 算子 | **5–8** | R0 的 verify 约 160 行（含 2 个 helper），GDN 改门控 + GQA |
| M2 | commit + plan + conv | **6–9** | R0 的 commit 约 190 行 + plan/conv 约 160 行，可大段照抄 |
| M3 | 缓存与元数据 | **3–5** | 生命周期已在 0.30.0，只需对齐 161 行 metadata + cache spec |
| M4 | MRV2 集成 | **4–6** | `self.recoverssm` 消费点已在 0.30.0（`_310p/model_state.py` 已在调 `commit_step`），只需接 GDN |
| M5 | 测试与图 | **6–9** | 对齐 #18765 的 384 UT + 45 NPU 测 |
| M6 | 性能 | **4–6** | UB 调参是不确定性最大的部分 |
| **合计** | | **30–45 人天** | 取中位 **≈ 37 人天（7.4 周）** |

**与"仿照 KDA"的差距解释**：`vLLM 0.30.0` 自带完整 Triton KDA（1067 行），所以**算法层几乎不用摸索**；真正多花时间的是 **Ascend 化**（UB/tiling/地址表），这部分只能参考 main 上的 1451 行实现。若走路线 B（直接站在 0.31 的 Ascend KDA 上），M1/M2/M3 各能再省 1–2 人天，约 **28–40 人天**。

---

## 6. Token 预算估算（你专门问的）

> 这里的"token"= 我（或任何 agent）完成这项工作要消耗的模型 token，**含反复读上下文的开销**，不只是输出代码。

### 6.1 单位成本假设

| 项 | 假设 |
|---|---|
| 代码 token 密度 | **15–18 token/行**（Triton + Python，带注释） |
| 测试密度 | 12–15 token/行 |
| 每次"精读一份文件"的成本 | 行数 × 1.3（含注释/上下文） |
| 每轮调试的上下文增长 | 25–40k token |
| 返工系数 | 2.5–3.0（写 → 测 → 改） |

### 6.2 分批次预算

| 批次 | 新增行 | 输出 token | 输入/上下文 token | 小计 |
|---|---|---|---|---|
| M0 准备 | ~150（oracle） | 2.5k | 90k（读 baseline + 跑通） | **~95k** |
| M1 verify | 400 | 7k | 180k（读 R1/R3/R4 + 调试） | **~190k** |
| M2 commit | 550 | 9k | 260k（读 R1 的 commit + plan/conv + 调试） | **~270k** |
| M3 缓存/元数据 | 230 | 4k | 150k（读 R6 + cache 结构） | **~155k** |
| M4 MRV2 接线 | 190 | 3.5k | 220k（读 worker / patch / 模型） | **~225k** |
| M5 测试与图 | 600 | 9k | 320k（写测试 + 跑 + 修） | **~330k** |
| M6 性能 | 100 | 2k | 180k（profile + 调 tile） | **~180k** |
| **首次实现小计** | **~2220** | **~37k** | **~1400k** | **≈ 1.45M** |
| 返工 / 二次迭代（×1.4） | — | — | — | **≈ 2.0M** |

### 6.3 结论

| 口径 | 估算 |
|---|---|
| **顺利（一次过）** | **≈ 1.0M–1.3M token** |
| **正常（含 1 轮返工）** | **≈ 1.6M–2.0M token** |
| **不顺利（图/UB 反复）** | **≈ 2.2M–3.0M token** |
| 只做 M0–M2（能跑通 verify/commit，不做集成） | **≈ 0.55M token** |

**加速手段**（能省 30–40%）：

1. **并行子 agent 精读**：3 个 agent 同时读 R1/R3/R6，各出一份"接口清单"，我只读清单 → 省 300–400k；
2. **先写 oracle 再写 kernel**：减少"改完不知道对不对"的往返 → 省 15–20%；
3. **把 KDA 的 kernel 逐行对照着改**（而不是凭记忆重写）→ 省最多；
4. **先只支持 Q=4**（跳过 Q=1..8 的参数化）→ 省 5–10%，但会限制后续。

---

## 7. 执行方式（我会怎么干）

```
阶段 A（并行，~0.2M token）
  SA1: 精读 R1（1451 行 KDA 算子）→ 输出「4 个 kernel 的输入输出契约 + UB 常量表」
  SA2: 精读 R3+R4（上游 GDN 702 行 + 共享算子 200 行）→ 输出「GDN 与 KDA 的差异清单 + 记录布局」
  SA3: 精读 R5+R6（MRV1 原型 + GLM 接线）→ 输出「MRV2 接线点清单 + 配置校验清单」
  我：只读三份清单（~20k），不读原始代码

阶段 B（串行，我写，~0.4M token）
  M0 → M1 → M2，每步：写 → 用 oracle 验 → 记录偏差

阶段 C（并行，~0.3M token）
  SA4: 写 UT + 边界矩阵
  SA5: 写图测试 + 性能脚本
  我：集成 + 跑 + 修

阶段 D（串行，~0.2M token）
  M4 → M5 → M6
```

**关键原则**：**每写完一个 kernel 立刻用 oracle 对一次**，绝不允许"写完一大坨再一起调"——那是 token 爆炸的主因。

---

## 8. 风险清单（按 概率×影响）

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | **0.30.0 上 Qwen3.8-27B 跑不通**（架构支持在 0.31 才进） | 中高 | 致命（M0 就卡） | **M0 第一件事就是验证**；不行就换 Qwen3.5-35B 或升版本 |
| R2 | **UB 溢出 / 静默算错** | 高 | 致命 | 照 #11236 的教训：改 Q 必须同步改 UB 核算并加断言 |
| R3 | Triton-Ascend 语义缺失（`tf32x3`、动态循环） | 中高 | 中 | 用 `('ieee','hf32')`；循环用 `static_range` |
| R4 | 记录 scratch 与图捕获冲突 | 中 | 中 | 地址表固定 + 每层固定行；先 eager 后图 |
| R5 | 0.30.0 的 MRV2 接口与 main 不同 | 中高 | 中 | M3 先确认 `mamba_hybrid.py` 在 0.30.0 的形态 |
| R6 | conv 压缩与 boundary 的边界 off-by-one | 中 | 高 | 单测覆盖 `steps % block == 0 / -1`（参考 vLLM #59933） |
| R7 | 性能不达预期（commit 288 MiB 访存） | 中高 | 中 | 第一版只求"不退化"；后续上延迟提交 / 混合精度 |
| R8 | 上游接口变动（#61309 未合） | 中 | 低 | 做成 patch，少耦合内部接口 |

---

## 9. 开工前需要你确认的 5 件事

| # | 问题 | 我的建议 |
|---|---|---|
| **Q1** | ~~版本：0.30.0 还是 0.31/main~~ | ✅ **已定：留在 v0.30.0，参考 main 的算子实现**（见 §0.2 的依据） |
| Q2 | **目标硬件**：A2 / A3 / A5？ | KDA 文档明确 **A2 未验证**；优先 A3 或 A5。若必须 A2，要先补 A2 验证（+3–5 人天） |
| Q3 | **目标模型**：Qwen3.8-27B（48 GDN 层 / 48 heads）？还是更小的 Qwen3.5-4B 做原型？ | 风险已排除（Qwen3.8 走 `qwen3_5` 路径，v0.30.0 就支持）。仍建议**双轨**：先用 4B 打通链路，再用 27B 验收 |
| Q4 | **MTP 深度范围**：只支持 Q=4，还是 2–8？ | 先 Q=4（覆盖主流 MTP3），再参数化 |
| Q5 | **是否要 prefix caching（align 模式）** | 第一版**不要**（RFC 也说单独验证），但设计时要给 boundary 留接口 |

---

## 10. 附：目录结构（已初始化）

```
/Users/Admin/Workspace/GDN-RecoverSSM-Ascend/
├── vllm/                          # vLLM v0.30.0（浅克隆）
├── vllm-ascend/                   # vllm-ascend v0.30.0（浅克隆）
├── docs/
│   ├── PLAN.md                    # ← 本文件
│   ├── ReplaySSM-2610.13778.pdf   # 论文
│   ├── GDN-RecoverSSM-实现方案.md  # 原理 + RFC + 现状 + 工作量 + 量化
│   ├── 七个算法汇总.md             # 上下文（7 个方案对比）
│   └── reference/                 # vendor 的参考实现（0.30.0 里没有的部分）
│       ├── kda_recoverssm_ascend.py
│       ├── kda_recoverssm_design.md
│       ├── recoverssm_gdn_upstream.py
│       ├── recoverssm_ops_upstream.py
│       ├── gdn_recover_ascend_mrv1.py
│       ├── gdn_recover_batch_ascend_mrv1.py
│       ├── glm5next_recoverssm_metadata.py
│       ├── recoverssm_config.py
│       ├── patch_replayssm_config.py
│       └── patch_kda_recoverssm.py
└── README.md                      # 一句话说明 + 怎么开始
```

---

## 11. 一句话总结

> **改动量：新增约 1500–2000 行（1 个内核文件占一半）+ 修改约 210 行（8 个文件）；时间 30–45 人天（中位 37，约 7.5 周）；token 约 0.9–2.1M（正常 1.5–1.9M）。**
> **版本不是障碍**：0.30.0 已有 RecoverSSM 生命周期与 Ascend 消费点，依赖版本与 main 完全一致，Qwen3.8 走既有 `qwen3_5` 路径；缺的只是算子，而算子只依赖 6 个在 0.30.0 里都存在的符号。
> **真正的风险转移到**：① UB 预算 ② align 模式的 boundary 折叠 ③ 图捕获。
