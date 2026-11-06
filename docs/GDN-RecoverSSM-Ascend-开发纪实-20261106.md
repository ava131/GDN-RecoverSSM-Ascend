# GDN RecoverSSM on Ascend — 开发纪实

**文档时间戳：2026-11-06 13:10 (+08:00)**
**对应代码：`a054916`（main）**
**目标 RFC：[vllm-project/vllm-ascend#19987](https://github.com/vllm-project/vllm-ascend/issues/19987) — *Support GDN RecoverSSM for Speculative Decoding on Ascend***

---

## 0. 一句话总结

在 Ascend MRV2 上为 Qwen3.8 的 **Gated DeltaNet** 实现了 RecoverSSM 的 verify/commit 生命周期：**投机解码时只保存一份循环状态检查点**，拒绝采样后按接受前缀一次性重建状态，取代"每个投机位置各存一份完整 SSM 状态"的传统做法；并在 full-graph + 异步调度下跑通长时服务。

---

## 1. 背景与目标

### 1.1 问题

投机解码时，target model 一次要验证 `K+1` 个候选 token，而 GDN 的循环状态 `S` 是逐 token 递推的。传统做法为**每个投机位置保留一份完整 SSM 状态**，于是状态缓存随投机窗口线性增长：

```
state cache = #requests × (1 + num_speculative_blocks)
```

`num_speculative_blocks = num_speculative_tokens`（MTP3 时为 3），即每请求 **4 份**完整状态。

### 1.2 RecoverSSM 的思路

只维护**一份检查点** `S_{n-1}`。verify 阶段：

- **不修改检查点**；
- 把重建"如果第 m 个 token 被接受，状态应该是什么"所需的**中间量**记下来。

拒绝采样知道接受了 `m` 个 token 后，commit 阶段把 `m` 条记录**一次折叠**进检查点，得到 `S_{n-1+m}`。

```
state cache = #requests × 1  +  临时记录
```

`num_speculative_blocks` 从 3 降到 0 —— 这是容量收益的来源。

### 1.3 RFC 的里程碑划分

RFC #19987 第 4 节明确：

> **Initially, focus on the MRV2 eager execution path. NPU Graph replay, prefix caching, KV transfer, and additional parallel configurations can be enabled after independent correctness validation.**

以及：

> **The first milestone focuses on functional correctness.**

---

## 2. 交付物

### 2.1 仓库

`https://github.com/ava131/GDN-RecoverSSM-Ascend`

以 **vLLM 插件**形式交付（`vllm.general_plugins` entry point），**不修改 vLLM / vllm-ascend 任何源码**——两棵上游树的 `git status` 全程保持干净。

### 2.2 代码结构

```
plugin/gdn_recoverssm_ascend/           3210 行
├── __init__.py         install() 入口，惰性守卫，装配各 patch
├── config_patch.py     配置校验、启停开关、启动横幅
├── state_patch.py      MambaStateShapeCalculator：新增记录张量、改 dtype
├── layer_patch.py      层内 verify 重定向、record_step、commit 后处理
├── builder.py          AscendGDNRecoverSSMMetadata + Builder + Backend
├── kernels.py          4 个 Triton kernel + 提交上下文 + 启动期编译准备
├── startup_patch.py    真实 KV 绑定之后、原生 warmup 之前完成首次资源准备
└── graph_probe.py      图捕获/重放探针、启动期全图预热、首错保护

plugin/tests/                           1560 行
├── test_math_oracle.py            11 项：verify/commit 数学对照
├── test_layer_patch_mock.py       12 项：装配与生命周期
├── test_kernel_launch_contract.py     启动契约 + 特化 opt-out + 静态检查
└── test_kernel_index_sim.py        4 项：寻址越界与 tile 覆盖
```

### 2.3 记录格式

每个 (token, head) 记 **257 个 float32**：

```
[ corr(V) : 128 | key_hat(K) : 128 | decay : 1 ]
```

- `corr(V)`：该 token 注入到状态里的"修正项"方向；
- `key_hat(K)`：归一化后的 key；
- `decay`：该 token 的衰减因子。

足以把任意"接受前缀长度 m"的状态一次性折叠出来，无需重放整段递推。

### 2.4 关键提交

| 提交 | 内容 |
|---|---|
| `16f4c2c` | 每个 verify/commit 索引做边界检查（解决 AI Core MTE 非法地址） |
| `9864fb0` | 修正 spec row → batch row 映射（此前会写错请求的状态） |
| `ffd2ff6` | 补丁打到模型真正实例化的类上（此前功能是死的） |
| `20fbd5e` | 有界 NPU dispatch、冻结捕获输入、GDN align 簿记 |
| `7c1c3cf` | 边界态与最终态由**同一次**折叠产出（此前两趟互相覆盖） |
| `cb11897` | 索引仿真、缓存 launch cap、无同步的 align 应用 |
| `5d682e2` | 提交上下文与 kernel 首次编译提前到原生 warmup 之前 |
| `a054916` | **捕获后通过正常路径对全部图各执行一次**（507899 的根治） |

---

## 3. 实现架构

### 3.1 数据流

```
                    ┌──────────────── 每次 decode step ────────────────┐
prefill 结束         │                                                   │
   │                 │  prepare_attn                                     │
   ▼                 │    └─ record_step(metadata)   ← 记住这一步的上下文 │
持久检查点 S_{n-1} ──►│                                                   │
                     │  forward（可能被 FULL graph 捕获）                 │
                     │    └─ GDN verify kernel                          │
                     │         ├─ 读 S_{n-1}（只读，绝不修改）             │
                     │         ├─ 写记录 rec[block][head][q][257]        │
                     │         └─ 写本步输出                             │
                     │                                                   │
                     │  sample_tokens                                    │
                     │    └─ postprocess_state                           │
                     │         └─ commit_recoverssm_state(accepted)      │
                     │              ├─ plan   : 每行的 commit_len = acc+1 │
                     │              ├─ compact: 卷积历史压缩              │
                     │              └─ commit : 单趟折叠 → 最终态 + 边界态 │
                     └───────────────────────────────────────────────────┘
```

### 3.2 四个 kernel

| kernel | 职责 |
|---|---|
| `_gdn_recoverssm_verify_kernel` | 4 个投机 token 在一个 tile 里递推，写记录、写输出；**检查点只读** |
| `_prepare_commit_plan_kernel` | 由 `accepted`/`qsl`/block_table 算出每行的 `commit_len`、最终态列、边界态列 |
| `_compact_conv_state_kernel` | 按 `commit_len` 压缩卷积历史到最终列与边界列 |
| `_commit_gdn_state_kernel` | 从**一次**检查点载入出发，单趟折叠出最终态与边界态 |

**为什么 verify 不能跨层融合**：第 i 层的输出是第 i+1 层的输入，必须逐层串行。
**为什么 commit 可以融合**：它只访问状态页，不依赖前向数据流。

### 3.3 有界 dispatch

Ascend 的物理核数有限，逻辑工作量（`num_work`）可能远大于可用核数。所有 kernel 采用：

```python
for work in range(pid, num_work, grid_size):
    if work >= num_work: break
    ...
```

物理 grid 取 `min(num_work, _launch_cap())`，`_launch_cap()` 缓存 `get_vectorcore_num()`。
这样同一份 kernel 在任何 grid 下都安全，且物理 grid 不会爆炸。

### 3.4 全图捕获下的约束

FULL graph 捕获后 **Python 不再运行**，因此：

- 所有张量地址在捕获时固定 → 提交上下文按"全部层张量的 `(name, data_ptr, shape, stride, dtype)`"做 key，变化才重建；
- 不能在捕获区内做任何设备同步或动态分配（曾因 `torch.empty` 与 `.item()` 触发问题）；
- `num_accepted_tokens` 等设备值不能读回 host。

---

## 4. 踩过的坑（完整清单）

> 每一条都标注：现象 → 证据 → 根因 → 修复。

### 坑 1：补丁打在了永远不会被实例化的类上

- **现象**：所有测试通过，服务正常，但 `verify_calls=0`，功能实际是死的。
- **证据**：vllm-ascend 里 `AscendGatedDeltaNetAttention` 只是**方法供体**，真实层是 `QwenGatedDeltaNetAttention`，被 `patch_qwen3_5.py` 在运行时改写。
- **根因**：只按类名找"看起来对"的类。
- **修复**：同时补两者，并在 `_forward_core` 里用全局开关判定（`ffd2ff6`）。

### 坑 2：插件日志在服务日志里看不见

- **现象**：插件打了日志，但 `vllm serve` 控制台里一条都没有。
- **根因**：vLLM 只配置 `vllm` 这个 logger，且 `propagate=False`；用模块名直接 `logging.getLogger(__name__)` 会落到无 handler 的 root。
- **修复**：全部改用 `init_logger("vllm.gdn_recoverssm_ascend.xxx")`（`875cd85`）。

### 坑 3：spec row → batch row 映射错误，写错了请求的状态

- **现象**：能跑，但多请求时状态逐渐错乱。
- **根因**：把"第 i 个投机行"直接当成"第 i 个 batch 行"。投机行是按请求顺序排列的，而 batch 行包含 prefill/非投机请求。
- **修复**：用 `spec_sequence_masks` 取 `nonzero` 得到映射；无法确定时**跳过并告警**，绝不猜（`9864fb0`）。

### 坑 4：AI Core MTE 非法 GM 地址

- **现象**：`507035 MTE invalid GM address`，随机出现在边界 batch。
- **根因**：verify/commit 里多个索引没有边界保护（空行、padding 行、未分配列）。
- **修复**：为每个索引加 `row_ok / source_ok / geometry_ok` 判定，非法行整体 no-op；再加一个**索引仿真测试**（`test_kernel_index_sim.py`）验证 tile 覆盖与越界（`16f4c2c`、`cb11897`）。

### 坑 5：重建 metadata 时丢了 Ascend 的动态字段

- **现象**：commit 时 `AttributeError`，或拿到 `None` 的 `spec_decode_metadata`。
- **根因**：用 `GDNAttentionMetadata(**{...})` 手工重建，丢掉了 vllm-ascend 动态挂上去的字段。
- **修复**：改为 `vars(meta)` 复制后替换个别字段。

### 坑 6：嵌套的卷积描述符没拿到 `num_accepted_tokens`

- **现象**：卷积历史压缩用了错误长度。
- **根因**：`spec_decode_metadata.spec_causal_conv1d` 是另一个对象，改外层不会传导。
- **修复**：显式把重写后的张量写回嵌套描述符（`c821a53`）。

### 坑 7：align 模式的列簿记与框架约定冲突

- **现象**：align 模式下 commit 写到了错误的列，状态偏移一格。
- **根因**：我们按 `(C+A-1)//B` 计算目标列，框架的通用后处理按 `(C+A)//B`；两者在整除边界上不同。
- **修复**：为 GDN 增加专用 staging，采用框架的**运行列**约定，并把"精确边界"路径删掉（`a14b93f`、`cb11897`）。

### 坑 8：边界态与最终态两趟折叠互相覆盖

- **现象**：单测里恢复出的状态是 `2.25`，期望 `4`。
- **根因**：第一趟折叠产出最终态，第二趟又从**已被改写的**检查点出发产出边界态。
- **修复**：改为**单趟**折叠，一次载入同时产出最终态与边界态（`7c1c3cf`）。

### 坑 9：在捕获区内做同步与动态分配

- **现象**：捕获期异常 / 卡死。
- **根因**：`_forward_core` 在捕获时也被执行，里面的 `.item()`、`torch.empty`、`synchronize` 都不合法。
- **修复**：探针全部改为 host-only 计数；`is_current_stream_capturing()` 时静默；分配改为捕获前一次性建好（`086a7b2`、`9a01005`）。

### 坑 10：kernel 缺少 constexpr / 变量先用后赋值

- **现象**：`CompilationError: NUM_BLOCKS is not defined`；以及 `UnboundLocalError: num_work`。
- **根因**：重写 dispatch 时漏传 constexpr；`args` 元组里引用了后面才赋值的变量。
- **修复**：修复后增加**启动契约测试**，静态检查"每个 constexpr 都有传"和"局部变量不得先用后赋值"，并**用注入错误的方式证明该检查有效**（`f6e4854`、`3aea6af`）。

### 坑 11：关闭 Triton 整型特化 → 编译直接失败

- **现象**：启动即失败，`hivm-plan-memory` 阶段报错。
- **证据**：

```
triton.compiler.errors.MLIRCompilationError:
[ConvertLinalgRToBinary] encounters error:
// -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
func.func @_gdn_recoverssm_verify_kernel(...) { mix_mode = "aiv", ... }
```

IR 里所有整型参数变成裸 `i32`。
- **根因**：Triton-Ascend 的 UB 内存规划**依赖整型特化**提供对齐/常量信息。
- **修复**：**回退**（`73d0388`）。结论：不能用"关特化"消除运行期编译。

### 坑 12：提交路径的冷初始化发生在第一个真实请求

- **现象**：commit 上下文、三个 kernel 的首次 JIT、设备二进制注册，全部发生在服务已开始跑之后。
- **根因**：`_get_context()` 是惰性的，第一次 `commit_step` 才创建。
- **修复**：包装 `NPUWorker.compile_or_warm_up_model`，在**真实 KV 已绑定、原生 warmup 之前**创建全部上下文并 compile+load 三个 kernel；显式检查 `module`/`function` 句柄非零（`rtDevBinaryRegister` 失败可能返回空句柄而不抛异常）（`5d682e2`、`e6f10e2`）。

### 坑 13：异步调度下，图首次执行失败（507899）

见第 5 章（主体）。

### 坑 14：裸 `graph.replay()` 预热导致启动死锁

- **现象**：capture 完成后卡住，60 秒后引擎报 `No available shared memory broadcast block found in 60 seconds`。
- **根因**：捕获图里的跨卡任务由 **ExternalEvent** 驱动，只有正常路径（`resolve_tasks → replay → update`）会 signal；裸重放会永久等待。
- **修复**：改为走 `run_fullgraph`（`5de78a9` → `a054916`）。

---

## 5. 主战役：异步调度下的 507899

### 5.1 现象

```
ERROR Ascend full-graph replay failed: num_tokens=8,num_reqs=2,cg_mode=FULL
ERROR halStreamTaskFill failed, drvRetCode=17, drvDevId=5, streamId=146, taskCnt=2595
ERROR error code is 507899
```

特征：

| 观察 | 值 |
|---|---|
| 触发时机 | 服务跑了几分钟后，批从 16 个请求缩到 2 个 |
| 固定 descriptor | `num_tokens=8, num_reqs=2` |
| 固定 model | `model_id=17` |
| 固定任务数 | `taskCnt=2595`（多次运行、多张卡完全一致） |
| 出错卡号 | 出现过 `drvDevId=4 / 5 / 6 / 7`（不固定） |
| 调用链 | `graph.replay → BuildSqCq → BindSqCqAndSendSqe → SendSqe → halStreamTaskFill` |
| 出现条件 | **async scheduling 开** |
| 消失条件 | async 关 / host 被拖慢（INFO 日志） |

### 5.2 源码依据

`CaptureModel::BuildSqCq` 的第一条判定就是早退：

```cpp
COND_PROC_RETURN_WARN((sqCqArray_ != nullptr) && (sqCqNum_ != 0U),
    RT_ERROR_NONE, refCount_++, ...);
```

**建过一次队列，之后永远不再走这条分支**，因此：

> 失败只可能发生在**每张图的第一次执行**。

这解释了为什么"跑很久才炸"（前面没换过图）、"只有某个形状炸"（那张图第一次被用）、以及为什么复用的图从不失败。

失败路径本身会自清理：

```cpp
error = BindSqCqAndSendSqe();
ERROR_PROC_RETURN_MSG_INNER(error, DELETE_A(sqCqArray_); sqCqNum_ = 0U;, ...);
```

### 5.3 被实验否定的四个假设

| # | 假设 | 为什么被否定 |
|---|---|---|
| 1 | 捕获后重建 context 让图持有野指针 | commit 在**图外**执行，它的地址表从未被捕获进图 |
| 2 | 设备队列积压导致灌不进任务 | 在首次执行前显式 `synchronize()`（4 个 rank 都打了日志），**照样失败** |
| 3 | 异步产生两条模型执行流，模型被跨流执行 | worker 是 FIFO 单 runner；PyTorch `NPUGraph` 明写 *model RI may be replayed in any stream* |
| 4 | 芯片级 AICPU 故障（两 die 共享 SO 竞态） | 后来**两张不同的卡**（dev 4 和 6）报出**同一个** `model_id=17`、**同一个** `taskCnt=2595` → 确定性对象，不是随机竞态 |

另外两条也被否掉：

- **"关掉整型特化消除运行期编译"** → `hivm-plan-memory` 编译失败（坑 11）；
- **"裸 replay 预热"** → 死锁（坑 14）。

### 5.4 最终修法

**把每张图的"第一次执行"提前到启动阶段、且必须走正常路径。**

```
capture 结束
  → prepare_attn 已记录每张图的 (num_tokens, num_reqs) → attn_metadata
  → 对 34 张图逐个：
        model_state.attn_metadata = 该图的那份
        manager.run_fullgraph(desc)        ← 含 resolve_tasks + update + replay
        恢复 attn_metadata
  → torch.npu.synchronize()
  → 开始服务（此后只有复用）
```

关键点：

1. **必须走 `run_fullgraph`**，不能裸 `replay()`（坑 14）；
2. **每个形状用它自己的 metadata**，缺记录就跳过、不猜；
3. 服务期仍保留"首次执行失败 → 排空 → 重试一次"作为第二道防线（依据是运行时自清理）。

### 5.5 验证证据（决定性）

```
03:34:32  (8,2) attempt=1     ← 启动预热完成首次执行（建队列）
03:34:34  warmup: executed 34 captured graphs (0 without recorded metadata)
03:40:56  (8,2) attempt=2     ← 6 分钟后，服务期真实复用
          last_rows=13 last_tokens=52
全程无 "replaying ... retrying once"
```

| 想排除的可能 | 证据 |
|---|---|
| 只是没走到（skip） | `attempt=2` 在服务期出现，是真活 |
| 靠重试兜底 | 没有任何 `retrying once` |
| 预热空转 | `0 without recorded metadata`，且 `attempt=1` 恰在预热窗口 |
| 预热后服务期不敢用 | `attempt=2` 就是服务期用的 |

**结论：根治，不是绕过。**

### 5.6 为什么这是"修"而不是"绕"

- 它没有关闭任何功能（async scheduling 照开）；
- 它把**惰性初始化**提前到语义上本来就该发生的时刻（服务开始前）；
- 它符合 `BuildSqCq` 自身的生命周期设计（建一次、永久复用）。

---

## 6. 测试与验证

### 6.1 离线测试（每次提交必跑）

| 测试 | 覆盖 | 结果 |
|---|---|---|
| `test_math_oracle.py` | verify/commit 数学：全接受、部分接受、立即拒绝、连续步、多请求不同接受数、卷积一致性 | 11 项通过 |
| `test_layer_patch_mock.py` | 装配顺序、metadata 复制、启动准备、图预热、重试语义 | 12 项通过 |
| `test_kernel_launch_contract.py` | 参数个数/顺序、constexpr 齐全、局部变量先用后赋值、特化 opt-out | 通过（且注入错误可复现失败） |
| `test_kernel_index_sim.py` | 寻址越界、tile 覆盖、空行/空批 | 4 项通过 |

### 6.2 上机验证记录

| 时间 | 配置 | 结果 |
|---|---|---|
| 11-03 | TP4 + MTP3 + FULL，功能初版 | 6.5 分钟、16 并发、312–356 tok/s、接受长度 3.15–3.19；尾部 `(8,2)` 报 507899 |
| 11-04 | + 有界 dispatch / 冻结捕获输入 | 同上，稳定性提升，尾部仍失败 |
| 11-04 | + 提交路径提前准备（1536 个变体） | 失败点不变 → 证明与 commit 冷初始化无关 |
| 11-05 | `--no-async-scheduling` | **不崩**，速度几乎不变 → 锁定 async 为触发条件 |
| 11-05 | 首次执行前排空设备 | **仍崩** → 否定"队列积压" |
| 11-05 | 裸 replay 预热 | **死锁** → 否定该实现方式 |
| 11-06 | 走 `run_fullgraph` 的全图预热 | **34 张图预热完成，服务期零失败**；`(8,2)` 在服务期被正常复用 |

### 6.3 规模数字

| 项 | 值 |
|---|---|
| 捕获图数量 | 34（FULL_DECODE_ONLY） |
| 捕获耗时 | 约 22 秒 |
| 图占用显存 | 约 1.9 GiB |
| GDN 层数 | 48（3 组 × 16） |
| 每 rank 头数 | 12 |
| 每图任务数 | 约 2595 |
| KV cache | 721,786 tokens / 1418 个状态页 |
| 并发上限 | 51（`plan rows`） |

---

## 7. 当前状态与未完成项

### 7.1 已完成

- ✅ verify/commit 数学正确（离线 oracle 11 项）
- ✅ 与 MRV2 `RecoverSSMMetadata` 生命周期对接
- ✅ `num_speculative_blocks` 3 → 0（容量收益的一半）
- ✅ 全图捕获 + 异步调度下单机长跑稳定
- ✅ 启动期资源准备（commit kernel 编译 + 全图首次执行）
- ✅ 首错保护、失败重试、边界安全
- ✅ 4 个离线测试文件（27 项检查）

### 7.2 未完成（按 RFC 优先级）

| 项 | 说明 |
|---|---|
| **NPU 正确性对照** | 目前只有 CPU oracle。需要在 NPU 上把"顺序 GDN 递推"与"verify+commit 重建"逐层逐 head 比矩阵（RFC 第一优先级） |
| **§3 记录移出 KV cache** | 现在记录是 page 定义里的第三个字段，按 **1418 个 block** 分配；RFC 要求改成按**请求行（51）**索引的 scratch，预计 3.2 GiB → 120 MB |
| 性能测量 | RFC 的性能表：持久 cache 占用、总 cache、verify/commit 延迟、TPOT、吞吐、接受长度 |
| eager 路径验证 | RFC 第一里程碑要求；目前一直跑 FULL graph |
| 无 align / prefix cache 组合 | 当前只验证 none + prefix cache |

### 7.3 已知限制

- 必须 **MRV2**（`VLLM_USE_V2_MODEL_RUNNER=1`）：RecoverSSM 的 commit 生命周期只存在于 V2；
- 必须 `cache_config.mamba_cache_mode ∈ {none, align}`；
- 必须 `num_speculative_tokens > 0`（否则不启用 RecoverSSM）；
- 记录当前占用**持久 cache**（§3 未做），因此容量收益只兑现了一半。

---

## 8. 附录

### 8.1 环境

| 项 | 版本 |
|---|---|
| vLLM | 0.30.0 `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vllm-ascend | v0.30.0rc1 `5f828ad638ee4f9b12db87f9d5481f29076d37f9` |
| CANN | 9.1.0 |
| torch-npu | 2.10.0.post4 |
| triton-ascend | 3.2.2 |
| 硬件 | A2（Ascend910B4），TP4 |

### 8.2 启动参数（关键项）

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve /models_directory/Qwen3.8-27B \
  --tensor-parallel-size 4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --use-replayssm \
  --enable-prefix-caching \
  --mamba-cache-mode none \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

### 8.3 容量计算（估算）

```
每层每 block 检查点 = HV × K × V × 4B
                    = 12 × 128 × 128 × 4 = 786 KB
48 层              → 37.7 MB / block

原生（4 份状态）   ：51 请求并发 → 51 × 4 × 37.7 MB ≈ 7.7 GB
RecoverSSM（1 份） ：51 × 1 × 37.7 MB           ≈ 1.9 GB

记录（当前，按 block）：
  每层每 block = 12 heads × 4 spec × 257 × 4B ≈ 48 KB
  × 1418 blocks = 66.7 MiB / 层
  × 48 层       ≈ 3.2 GiB / 每卡

记录（§3 目标，按请求行）：
  51 × 48 KB × 48 层 ≈ 120 MB / 每卡   → 约省 27 倍
```

### 8.4 关键日志

```
Ascend GDN RecoverSSM: async_scheduling=True (raw=True) scheduler_cls=AsyncScheduler
Ascend GDN RecoverSSM: prepared 512 commit variants for 16 layers (rows 1..512)
Ascend GDN RecoverSSM commit context: 16 layers, spec_query_len=4,
    state tensors per layer=3, replay shape=(1418, 12, 4, 257), plan rows=51
Ascend full-graph capture finished: graphs=34 verify_calls=3264 last_window=(1,4)
Ascend full-graph warmup: executed 34 captured graphs before serving
    (0 without recorded metadata); queue construction is done
Ascend full-graph replay: num_tokens=8,num_reqs=2,cg_mode=FULL attempt=2 ...
```

### 8.5 参考资料

- RFC：vllm-project/vllm-ascend#19987
- KDA RecoverSSM 参考实现：vLLM #51855、vllm-ascend #20128 / #20131 / #20134
- 记录外置的设计来源：vLLM #63045
- CANN `CaptureModel::BuildSqCq` / `AllocSqCqAndBindInternal` / `SendSqe`
