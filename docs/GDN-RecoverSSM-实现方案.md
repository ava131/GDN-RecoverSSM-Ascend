# GDN RecoverSSM 实现方案（对齐 vllm-ascend issue #19987）

面向"在昇腾上把 GDN 的 n+1 份状态问题解决掉"这一目标，把 ReplaySSM 原理、RFC 要求、KDA 现有实现、以及 GDN 的工作量拆解放在一起。

- 上游 RFC：<https://github.com/vllm-project/vllm-ascend/issues/19987>（`[RFC]: Support GDN RecoverSSM for Speculative Decoding on Ascend`，2026-10-27 开）
- 上位 RFC：<https://github.com/vllm-project/vllm-ascend/issues/19733>（`RecoverSSM state recovery for speculative decoding on Ascend`，2026-10-25 开）—— 明确写了 **"GDN recovery is follow-up work; it is not implemented by #19133"**
- 本文所有代码位置、行数、PR 状态均按 **2026-11-02** 的 `main` 核对过，来源见 §6。

---

## 先分清三个名字

| 名字 | 是什么 | 出处 |
|---|---|---|
| **ReplaySSM** | **论文/算法名**（Princeton + NVIDIA，Tri Dao 等） | arXiv 2610.13778；<https://github.com/Johnny-Liou/ReplaySSM> |
| **RecoverSSM** | vLLM 社区给"**投机解码版**"起的名字。机制同源，但**去掉环形缓冲**，改成"每步 verify 写记录、每步 commit 重建" | 类 `RecoverSSMState`（`vllm/v1/worker/gpu/model_states/recoverssm.py`）、`vllm/v1/attention/backends/recoverssm_metadata.py` |
| **`--use-replayssm`** | **唯一的用户开关**（名字仍叫 replay）。打开后内部会派生 `use_kda_recoverssm` | `vllm/config/cache.py` |

它们怎么串起来（`vllm/config/vllm.py` 的校验器，原文）：

```python
if not self.cache_config.use_replayssm:
    self.cache_config.use_kda_recoverssm = False      # 没开 replay，就没有 recover
    return
...
use_kda_recoverssm = (self.num_speculative_tokens > 0 and is_kda_model)
```

Ascend 侧（`vllm_ascend/patch/platform/patch_replayssm_config.py`）：

```python
if config.cache_config.use_replayssm and is_glm_recoverssm_model(config):
    validate_glm_recoverssm(config)
    config.cache_config.use_kda_recoverssm = True     # KDA 走 RecoverSSM 分支
```

⇒ **用户只看到 `--use-replayssm` 一个开关，内部把它分成两条路：**

| 开关组合 | 走哪条 |
|---|---|
| `--use-replayssm`（无投机 / Mamba2） | **ReplaySSM**：环形缓冲（`replayssm_buffer_len=16`）+ 每 `L` 步 flush |
| `--use-replayssm` + 投机解码 + KDA/GDN 模型 | **RecoverSSM**：每步 verify 写记录、每步 commit 重建，`num_speculative_blocks = 0` |

**本文的分工**：§1 讲 **ReplaySSM 论文**的原始原理（含环形缓冲、双精度、output-only 验证）；§3/§4 讲 **RecoverSSM 这个变体**（因为 #19987 要落的就是它）。两者的逐项差异见 §1.7。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| 要做什么 | 把 vLLM 上游的 **GDN RecoverSSM**（#61309，702 行内核 + 198 行后端 + 接线）在 vllm-ascend 上用 Triton-Ascend 重写一份，并对齐 #19987 的四项要求（MRV2、记录移出 KV page、GDN 专用内核、conv 压缩） |
| KDA 那边有多现成 | **已经在 main 里跑起来了**：`vllm_ascend/ops/triton/kda/recoverssm.py`（1451 行）+ `recoverssm_metadata.py`（161 行）+ 4 个 patch 文件，支持 A3 / 950DT，eager + NPU graph |
| GDN 能复用多少 | **约 55–65%**：commit plan 内核、conv 压缩内核、反向折叠框架、paged 地址表机制、UB 预算经验、MRV2 接线骨架、测试脚手架 |
| 必须重写多少 | **约 35–45%**：gate 计算（标量 vs 逐通道）、衰减广播方式、q/k 的 GQA 分组、记录布局、模型接线（qwen3_5/next）、配置判定 |
| 估时 | **30–50 人天**（单人 6–10 周），其中算子 10–16 人天、框架接线 8–12 人天、正确性 5–8 人天、图与性能 5–10 人天 |
| **带宽收益** | Q=4 时状态访存 **1.60×**（每步 commit）／**2.35×**（+延迟提交）／**4.21×**（论文环形缓冲）；`num_spec ≤ 1` 时是 **0.98×，不划算** |
| **容量收益** | 每请求状态槽 576 MiB → 153 MiB；B=256 时整模型 **144 GiB → 38 GiB（释放 106 GiB）**，约等于 **173 万 token 的 KV** |
| **重算开销** | 算力只占前向 **0.14–1.26%**（可忽略）；但访存 288 MiB/请求/步，**约 5.7 ms/步（B=256, TP=8）**，是真实成本 |
| 最大风险 | ① UB 预算（记录是"每 token 一份"，tile 数量和 Q 直接耦合）② align 模式的 boundary state 二次折叠 ③ MRV2 的 ragged/padding 行语义 ④ 上游 #61309 未合入，接口可能还会变 |

---

# 1. ReplaySSM 原理与公式

## 1.1 记号

| 符号 | 含义 |
|---|---|
| `S ∈ R^{V×K}` | 一个 value head 的循环状态（GDN 里 `K=V=128`） |
| `H / HV` | key 头数 / value 头数（GDN 是 GQA：`HV/H = 组大小`，如 16→32 时组大小 2） |
| `Q` | 投机窗口长度 = `num_speculative_tokens + 1`（1 个已确认 + K 个候选） |
| `P` | 缓存页数（paged cache 的 block 数） |
| `n` | 被接受的长度（`1 ≤ n ≤ Q`），即要提交多少步 |
| `A_log_h, dt_bias_h` | 每 head 的衰减参数（`a_t` 是投影出的标量，`b_t` 是写入门） |

## 1.2 GDN 的单步递推（确切形式）

```text
g_t      = -exp(A_log_h) * softplus(a_t + dt_bias_h)      # 标量，per value head
decay_t  = exp(g_t) ∈ (0, 1)
beta_t   = sigmoid(b_t)
k̂_t      = k_t / ||k_t||,   q̂_t = q_t / ||q_t||        # head 内 L2 归一化（可选开关）
S_t      = decay_t * S_{t-1} + delta_t * k̂_tᵀ
delta_t  = beta_t * ( v_t − decay_t * (S_{t-1} k̂_t) )     # V 维向量，"写什么"
o_t      = S_t q̂_t   (再乘 scale = K^-0.5)
```

**关键点**：`decay_t` 是一个**标量**（整 head 共用一个），这是 GDN 和 KDA 最本质的区别（KDA 是长度 K 的向量，逐 key 通道不同）。

## 1.3 把状态"展开"：为什么可以不存中间状态

把递推从窗口起点的检查点 `S_0` 一路展开，记 `γ_t = ∏_{s≤t} decay_s`：

```text
S_n = γ_n · S_0 + Σ_{t=1..n} (γ_n / γ_t) · delta_t · k̂_tᵀ            (式 1)
```

**这就是全部秘密**：只要手上还有 `S_0` 和每一个 `(delta_t, k̂_t, decay_t)`，**任意一步的 `S_n` 都能直接乘出来**，不需要它当时被存下来。

- `delta_t`（V 个数）+ `k̂_t`（K 个数）+ `decay_t`（1 个数）= **每条记录 `V+K+1 = 257` 个数**；
- 而一份状态是 `V×K = 16384` 个数；
- **差 64 倍**（128/2，正好是 K/2）。

## 1.4 验证（verify）：只写输出和记录，绝不碰检查点

对窗口内 `Q` 个 token 依次做 1.2 的递推，但状态只活在寄存器里：

```text
读：checkpoint_state ← 页 src，1 次
循环 t = 0..Q-1：
    读 q/k/v/a/b（这几行很小）
    算 g、decay、beta、delta、新状态、输出 o_t
    写：o_t → 输出张量
    写：记录[t] = [delta_t(V), k̂_t(K), decay_t(1)] ← FP32
结束：**检查点不动**
```

对比现状（写 Q 份完整状态）：**省掉了 `Q-1` 次完整状态写**，代价是写了 `Q×(V+K+1)` 个 FP32 —— 比例是

```
记录开销 / 一份状态 = (V+K+1) / (V·K) ≈ 257 / 16384 ≈ 1.57%   （Q 份记录 ≈ 6.3%）
```

实测印证（vllm-ascend #18765，48 value heads、K=V=128、MTP3）：**SSM + 记录 ≈ 3.188 MiB，而 4 份快照要 12 MiB**。

## 1.5 提交（commit）：只重建被接受的那一份

采样拿到接受长度 `n` 之后，反向折叠：

```text
S_n = exp(Σ_{t≤n} g_t) · S_0 + Σ_{t≤n} exp(Σ_{t<s≤n} g_s) · delta_t · k̂_tᵀ    (式 2)
```

实现上是"从第 n 条记录往回走，一边累乘衰减、一边把 `delta k̂ᵀ` 加权累加"：

```text
acc = S_0
for t = n-1 down to 0:
    acc = decay_{t+1} * acc + delta_{t+1} k̂_{t+1}ᵀ
写：checkpoint ← acc     （1 次）
```

> 式 2 与式 1 是同一个东西的两种写法。代码注释里写的是式 2 的形式（"recovery folds the accepted prefix"）。

**align 模式还有第二次折叠**：当窗口跨过一个 `mamba_block_size` 边界时，除了"最终状态"，还要额外写一份**边界状态**（`BOUNDARY` 那次 launch），因为前缀缓存需要块边界上的检查点。

## 1.6 指针回滚

被拒绝的记录**什么都不用做**：`n` 是他们唯一的下标依据，下一轮直接覆盖写在第 `n` 条之后。**零字节拷贝。**

## 1.7 论文里的另一种配方（环形缓冲 + flush），别和实现对不上

ReplaySSM 论文（arXiv 2610.13778）描述的是**通用形态**：checkpoint + 容量 `L+T` 的环形缓冲，**每 L 步才 flush 一次**，平时连记录都只 append。

而 vLLM 落地的是**更简单的形态**（上游叫 RecoverSSM）：**每一步 verify 都写记录、每一步 commit 都重建**。区别：

| | 论文 ReplaySSM | 落地 RecoverSSM |
|---|---|---|
| 记录生命周期 | 环形缓冲，容量 `L+T`，指针回退 | 每步重写，长度 `Q` |
| 状态写频率 | 每 `L` 步一次 | 每步一次（但只 1 份） |
| 适用 | 标准 decode（`replayssm_buffer_len=16`）+ 投机 | 投机（`num_speculative_blocks=0`） |

**在 Ascend 上要做的，是后者（RecoverSSM）。** 论文的环形缓冲版本对应的是 vLLM 里 Mamba2 标准解码那条路，和本 RFC 无关。

## 1.8 双精度检查点（论文 §6，可选增强）

论文把检查点拆成两份：**HP 只负责递推、LP 只负责算输出**，让低精度舍入误差不回流。实测（四模型平均，相对 FP32）：

| 配置 | MMLU-Pro | GPQA-D | Math-L5 |
|---|---|---|---|
| 共享 NVFP4 | −5.88 | −13.05 | −10.92 |
| **分离 NVFP4/FP16** | −0.15 | −2.36 | −0.10 |

⇒ 如果要压精度，**必须做成"递推/输出分离的两份"**，不能只降唯一那份。
（当前 Ascend KDA 路径用 **FP32 状态**、BF16 激活/记录，尚未启用这条。）

---

# 2. issue #19987 的要求与名词解释

## 2.1 RFC 的目标（原文要点）

> 把 RecoverSSM 扩展到 GDN，在 **Ascend Model Runner V2 (MRV2)** 上：
> - 投机解码时只维护**一个检查点**，验证期间不被候选状态覆盖；
> - 验证后按**实际接受的 token 前缀**重建状态（含首 token）；
> - 减少状态缓存占用，**不牺牲正确性**。

## 2.2 RFC 要求实现的能力（逐条）

| # | 要求 | 具体含义 |
|---|---|---|
| R1 | **GDN 的 verify/commit 生命周期** | 复用 KDA 的 RecoverSSM 架构和 MRV2 的"采样后提交"时机；验证期间**检查点只读** |
| R2 | **GDN 专用内核** | GDN 的状态更新方程与 KDA 不同，需要专门算子；支持 GDN 的状态布局与数值精度 |
| R3 | **记录移出 KV cache** | 按上游 [#63045](https://github.com/vllm-project/vllm/pull/63045) 的做法，把恢复记录放到 **per-layer scratch buffer**（按投机 batch 的行索引），不再占 SSM cache 的页 |
| R4 | **MRV2 集成** | 恢复上下文初始化、GDN 验证/恢复元数据、accepted-token 计数、采样后提交、**卷积状态压缩**、NPU 算子 |
| R5 | **复用 `--use-replayssm` 开关** | 不新增用户配置项 |
| R6 | **先跑通 eager，图后置** | "Initially, focus on the MRV2 eager execution path"；NPU Graph replay、prefix caching、KV transfer 各自单独验证 |

RFC 明确列出的**必须支持**的边界情况：

- 部分接受 / 全部接受；
- **第一个 draft token 就被拒**（`n=0` 也要正确，即检查点不变）；
- 连续多轮 verify/commit；
- 同一 batch 内**不同请求接受长度不同**；
- **卷积状态正确恢复**（conv history compaction）。

## 2.3 正确性验证清单（RFC 给的）

| 维度 | 用例 |
|---|---|
| 数值 | 与**顺序执行的 GDN 状态更新**逐项对比，误差在给定容差内 |
| 边界 | 不同投机长度；部分/全部/立即拒绝；连续多步 |
| 多请求 | 不同 accepted 数、**请求重排/重映射**、inactive 请求、**graph padding 行** |
| 状态一致性 | recurrent state 与 convolution state 都要对 |
| 精度 | FP32 / BF16 |
| 图 | NPU Graph replay 下结果一致 |
| 端到端 | 生成准确率、接受长度不退化 |

## 2.4 性能验证（RFC 给的）

在 **A3 / A5** 上对比"普通 GDN 投机解码"与"GDN RecoverSSM"：

| 指标 | 目标 |
|---|---|
| 持久 SSM cache 占用 | 下降 |
| 实际分配的 cache 总量 | 下降 |
| verify 延迟 | 记录开销可控 |
| commit 延迟 | 重建开销可控 |
| TPOT / 吞吐 | 端到端改善 |
| 接受长度 | 无回归（否则说明语义错了） |

## 2.5 名词表

| 名词 | 解释 |
|---|---|
| **MRV1 / MRV2** | vLLM 的 ModelRunner V1 / V2。V2 是把"模型状态"从 runner 里拆出来（`vllm/v1/worker/gpu/model_states/`）的新架构。**#18765 是 MRV1 的 GDN 实现，#19987 要的是 MRV2** |
| **ReplaySSM / RecoverSSM** | 同一套思想的两个名字。上游把"投机解码版"叫 **RecoverSSM**，对外开关沿用 `--use-replayssm` |
| **checkpoint（检查点）** | 窗口起点那份完整状态 `S_0`；验证期间**只读** |
| **recovery record（恢复记录）** | 每个 token 一条：`[delta(V), k̂(K), decay(1)]`，FP32 |
| **verify** | 一次前向验证 `Q` 个候选：写输出 + 写记录，**不写检查点** |
| **commit** | 采样后，用被接受前缀的记录把检查点重建成 `S_n`，**只写一次** |
| **rejection sampling / accepted prefix** | 投机解码的接受规则；结果是"接受前 `n` 个" |
| **num_accepted_tokens** | 每个采样行的接受数（含目标模型自己那一步），commit 的驱动参数 |
| **conv state / conv history** | GDN 前面的短卷积（kernel=4）的状态，长度 `conv_kernel-1+Q`；commit 时要**把被接受部分的窗口压缩到左侧** |
| **scratch buffer（暂存区）** | 记录不放进 KV page，而是按 `[layer, request_row]` 单独分配，长度 `Q` |
| **graph padding rows（图填充分行）** | NPU Graph 是固定形状，批次不满时补的空行（`state_indices ≤ NULL_BLOCK_ID`）。必须保证这些行**不写脏数据**、输出为 0 |
| **request remapping（请求重映射）** | 投机 batch 的行序和采样 batch 的行序不一定一致，commit 要能按映射找回 |
| **align mode / boundary state** | `mamba_cache_mode=align`：按 KV block 边界存检查点以支持前缀缓存。跨边界窗口除了最终状态，还要额外折出**边界状态** |
| **KV cache rebind（重绑定）** | 缓存在 P/D 分离、KV 迁移、重分配后地址会变，恢复上下文要能重建（代码里是"地址表 + stride 表"的做法） |
| **UB** | 昇腾 AI Core 的片上缓冲（约 192 KB）。Triton-Ascend kernel 的 tile 选择必须按 UB 算，超了会静默出错 |
| **NULL_BLOCK_ID** | 表示"这一行没有实际页"的哨兵值；null 行必须只产出 0 输出、不写记录 |
| **PLE** | per-layer embedding（Qwen4Exp 用）。上游有独立的 `ple_recoverssm.py`，**不在本次范围** |

---

# 3. KDA RecoverSSM 在 vllm-ascend 的现状

## 3.1 版本与依赖（2026-11-02 核对）

| 项 | 值 |
|---|---|
| vllm-ascend 分支 | `main` @ `31fa24405a`（2026-11-02） |
| 最新 tag | `v0.31.0rc1`（此前 `v0.30.0`） |
| torch / torch-npu | `torch==2.10.0` / `torch-npu==2.10.0.post7` |
| **Triton-Ascend** | **`triton-ascend==3.6.0.post2`** |
| 上游 vLLM 参考基线 | 代码注释写明 "Adapted from vLLM `c4df37dfcc`"（与 #18765 同一基线） |

## 3.2 已合入 main 的文件清单

| 文件 | 行数 | 作用 |
|---|---|---|
| `vllm_ascend/ops/triton/kda/recoverssm.py` | **1451** | 全部 NPU 算子：verify（3 个 helper + 1 个主 kernel）、commit plan、conv 压缩、commit（普通 + grouped）、commit context |
| `vllm_ascend/ops/triton/docs/kda_recoverssm.md` | **355** | 设计文档：公式、参数表、支持范围 |
| `vllm_ascend/models/glm5next/recoverssm_metadata.py` | **161** | GLM 的恢复元数据（把层参数、页、stride 组织起来） |
| `vllm_ascend/models/glm5next/recoverssm_config.py` | **38** | 判定 + 配置校验（`is_glm_recoverssm_model` / `validate_glm_recoverssm`） |
| `vllm_ascend/patch/platform/patch_replayssm_config.py` | **26** | 改写上游 `VllmConfig.validate_mamba_cached_kernel`，把 GLM 接进 `use_kda_recoverssm` |
| `vllm_ascend/patch/worker/patch_kda_recoverssm.py` | **15** | worker 侧补丁 |
| `vllm_ascend/models/glm5next/kda.py`（改） | +56/−21 | KDA 层接入 |
| `vllm_ascend/ops/gdn_attn_builder.py`（改） | +15/−2 | 元数据构建 |
| `vllm_ascend/worker/v2/model_states/mamba_hybrid.py`（改） | +2 | **MRV2 钩子（只有 2 行）** |
| `vllm_ascend/platform.py` / `worker/worker.py`（改） | +4/+4 | 注册 |

> **注意**：`vllm_ascend/ops/gdn_recover.py` 在 main 里**不存在（404）** —— 那是 PR #18765 的文件，还没合。

## 3.3 算子执行流（4 个 kernel）

```
① _kda_recoverssm_verify_kernel
   每个 (value tile, 请求, head) 一个 program
   读 checkpoint（只读）→ 寄存器里连推 Q 步 → 写 o_t 和记录
   null 页行：输出 0，不写记录

② _prepare_commit_plan_kernel
   投机行 → 采样行 的映射；把 accepted 长度 clamp 到 Q；
   算出 final / boundary 目的地；null 页和 0 长度变成 no-op

③ _compact_conv_state_kernel（+ _copy_conv_state_tile）
   卷积历史搬到左对齐：源区间 [c-1, c-1+W) → 目标 [0, W)

④ _commit_kda_state_kernel / _commit_kda_state_grouped_kernel
   反向遍历被接受记录，累乘衰减 + 累加外积 → 写回 checkpoint
   grouped 版：两个相邻 head 一个 program（A5 快路径）
```

**复用的三个工程要点**（这些和算子无关，GDN 能直接抄）：

1. **paged 地址表**：`state_base_addrs_ptr` / `state_block_strides_ptr` 按层组织，用 `recoverssm_ptr_table` 生成 —— 解决 KV rebind。
2. **两个不相交的 device 侧 request filter**：把"要写 boundary"和"不写 boundary"的请求分成两批，避免一个 kernel 里分支。
3. **UB 预算常量**：`RECOVERSSM_COMMIT_UB_MATRICES = 2`、`..._BOUNDARY_... = 6`、`RECOVERSSM_VALUE_TILE = 64/128`、`RECOVERSSM_CONV_TILE = 4096`、`..._STRIDED_TILE = 256`、`RECOVERSSM_COMMIT_HEAD_TILE = 2` —— **这些就是"改宽 Q 就要同步改"的那本账**。

## 3.4 支持的硬件与执行模式（文档原文）

- ✅ **Atlas A3**、**Ascend 950DT（A5）**
- ❗ **Atlas A2、950PR 未验证**（文档明确写 "not validated by these tests"）
- ✅ eager 与 **NPU graph**
- A5 快路径：FP32、`H32/H64`、`K=V=128` 用 grouped head 内核；其他布局走普通内核 + UB 受限的 value tile

## 3.5 配置侧的限制（`validate_glm_recoverssm` 实测代码）

```
必须有投机解码
不支持 --enable-mamba-cache-stochastic-rounding
mamba_cache_mode 必须是 align（开 prefix caching）或 none（不开）
PP=1 且 PCP=1，DCP=1
KV transfer 只允许 MooncakeConnectorV2 / MooncakePullConnector
```

## 3.6 上游与 ascend 的 PR 关系

| 侧 | PR | 状态 | 内容 |
|---|---|---|---|
| vLLM | [#51855](https://github.com/vllm-project/vllm/pull/51855) | ✅ merged | KDA RecoverSSM 的 checkpoint/record 接口 + MRV2 commit 生命周期 |
| vLLM | [#63045](https://github.com/vllm-project/vllm/pull/63045) | ✅ merged 2026-10-24 | 记录移出 KV cache + CuTe verify/commit 内核 |
| vLLM | [#61566](https://github.com/vllm-project/vllm/pull/61566) | 🔄 open | GLM-5.3-Flash KDA 集成 |
| vllm-ascend | [#19087](https://github.com/vllm-project/vllm-ascend/pull/19087) | 🔄 open | MRV1 原型（已归档，仅供参照） |
| vllm-ascend | [#19133](https://github.com/vllm-project/vllm-ascend/pull/19133) | ❌ closed 未合并 | MRV2 实验版（paged 记录，需按 #63045 对齐） |
| vllm-ascend | [#20131](https://github.com/vllm-project/vllm-ascend/pull/20131) | ✅ merged 2026-10-29 | **`[2/N]` 框架集成** —— KDA 的能力就是这样分批落进 main 的 |

> **对 GDN 的启示**：KDA 是"先算子 PR、再框架 PR 分批"N 次落地的，**GDN 大概率也走同一条路子**，#19987 就是那个"要不要做、怎么做"的总纲。

## 3.7 上游 KDA 参考实现（供 Ascend 对照）

```
vllm/models/kimi_k3/nvidia/ops/recoverssm.py        ← CUDA（CuTe）verify/commit
vllm/models/kimi_k3/nvidia/recoverssm/cutedsl/verify.py
vllm/models/kimi_k3/nvidia/recoverssm/cutedsl/commit.py
vllm/v1/worker/gpu/model_states/recoverssm.py       ← 通用 RecoverSSMState（含 for_capture 跳过记录）
vllm/v1/attention/backends/recoverssm_metadata.py    ← 通用元数据
```
**上游 KDA 内核是 CuTeDSL 写的、目标是 NVIDIA SM10x**（RFC #19733 原文），所以 Ascend 必须用 Triton-Ascend / AscendC 重写 —— 这就是那 1451 行的由来。

---

# 4. GDN 的实现工作量评估

## 4.1 上游已经有 GDN 的参考实现（关键前提）

vLLM 的 GDN RecoverSSM 在 **[#61309](https://github.com/vllm-project/vllm/pull/61309)（open，Qwen3.5/Qwen3-Next）** 和 **[#58863](https://github.com/vllm-project/vllm/pull/58863)（open，Qwen4Exp）**，文件构成：

| 文件 | 行数 | 作用 |
|---|---|---|
| `vllm/model_executor/layers/mamba/gdn/recoverssm_gdn.py` | **702** | GDN verify 内核 + commit 上下文 |
| `vllm/model_executor/layers/mamba/ops/recoverssm.py` | 200 | 公共的 commit plan / conv 压缩内核 |
| `vllm/model_executor/layers/mamba/recoverssm_utils.py` | 37 | 地址表工具 |
| `vllm/v1/attention/backends/gdn_recoverssm.py` | 198 | GDN 的恢复后端 |
| `qwen_gdn_linear_attn.py`（改） | +122/−10 | 层内接入 |
| `qwen3_next.py` / `qwen3_5.py`（改） | +68/−24 / +19/−53 | 模型接线 |
| `config/cache.py` / `config/vllm.py`（改） | +6/−2 / +40/−8 | 配置与校验 |

**它自己的文件头就写清了和 KDA 的差异**（这段值得整段保留）：

> *"GDN port of the Kimi-K3 KDA RecoverSSM path. Differences from KDA: the gate is a **scalar per value head** (`g = -exp(A_log) * softplus(a + dt_bias)`), **q/k heads are grouped over value heads** (`HV // H`), and q/k are L2-normalised inside the kernel when requested. The verify step caches, per token, the delta-rule correction `c_t = beta_t*(v_t − S'_t k_t)` (V floats), the normalised key (K floats) and the decay `exp(g_t)` (1 float) in one fp32 replay record."*

⇒ **算法、记录布局、kernel 切分都有现成参考，Ascend 侧的工作是"照着重写 + 换布局 + 接 MRV2"，不是从零设计。**

## 4.2 可直接复用（估 55–65%）

| 资产 | 位置 | 复用方式 |
|---|---|---|
| **commit plan 内核** | KDA `_prepare_commit_plan_kernel`（343 行处） | 与算子完全无关：行映射、clamp、boundary 目的地。**改常量即可** |
| **conv 状态压缩内核** | KDA `_compact_conv_state_kernel` + `_copy_conv_state_tile` | GDN 也是短卷积，窗口搬移逻辑相同 |
| **commit 折叠框架** | KDA `_commit_kda_state_kernel` | "反向遍历 + 衰减累乘 + 外积累加"结构相同，只改衰减的广播方式（标量 vs 向量） |
| **paged 地址表 / rebind** | `recoverssm_ptr_table`、`state_base_addrs_ptr` + stride 表 | 直接照抄 |
| **NULL 页 / 空行语义** | `state_indices ≤ NULL_BLOCK_ID` 分支 | 直接照抄 |
| **UB 预算与 tile 选择** | `RECOVERSSM_*` 常量 + `get_ub_size_bytes()` | 照抄，按 GDN 的记录宽度重算 |
| **MRV2 接线骨架** | `worker/v2/model_states/mamba_hybrid.py`（+2 行）、`patch/worker/patch_v2/patch_model_runner.py` | 照抄，换判定条件 |
| **配置补丁范式** | `patch/platform/patch_replayssm_config.py`（26 行） | 照抄，改成 GDN 的架构判定 |
| **测试脚手架** | `tests/e2e/nightly/.../test_kda_recoverssm_npu.py`（882 行）、`test_glm5next_recoverssm_metadata.py`（272 行） | 用例结构（边界、ragged、空行、slot 复用）可直接改编 |

> 还有一个**现成的 MRV1 版 GDN 实现**：[vllm-ascend#18765](https://github.com/vllm-project/vllm-ascend/pull/18765)（21 文件 / +2191−17），含 `ops/triton/mamba/gdn_recover.py`（399 行）、`gdn_recover_batch.py`（177 行）、`ops/gdn_recover.py`（154 行）。它的**验证数学和 conv 压缩逻辑可以直接搬**，但它的记录是 paged 的、且是 MRV1 —— 正对应 #19987 要改的两处。

## 4.3 必须重新实现（估 35–45%）

| 项 | 为什么不能直接用 KDA 的 |
|---|---|
| **gate 计算** | KDA：`g = lower_bound * sigmoid(a*x)` 或 `-a*softplus(x)`，**逐 key 通道**（`dt_bias` 是 `H*K` 个）；GDN：`g = -exp(A_log)*softplus(a+dt_bias)`，**标量 per head**（`A_log[H]`、`dt_bias[H]`）。计算量小很多，但布局和参数形状全变 |
| **衰减广播** | KDA：`state * decay[None, :]`（按 key 通道）；GDN：`state * decay`（标量）。**看起来是简化，但会影响 tile 选择和寄存器压力** |
| **q/k 的 GQA 分组** | GDN 有 `HV // H` 的组映射（`pid_h = pid_hv // (HV//H)`），KDA 是 1:1。**多一次整数除法 + 索引** |
| **记录布局** | KDA：`kg_cache [P,H,Q,2K]`（原始 k + 原始 g）；GDN：`[P,H,Q,V+K+1]`（correction + 归一化 k + decay）。**尺寸和语义都不同** |
| **softplus 分支** | GDN 需要 `softplus`（含 `beta/threshold` 线性分支）；KDA 用 sigmoid-with-lower-bound。Triton-Ascend 上 `exp/log` 的精度与分支写法要重验 |
| **align 的 boundary 折叠** | 逻辑相同但 GDN 的"边界状态"要用**标量衰减**重算，且上游 #61309 里 boundary 路径的写法与 KDA 不同 |
| **模型接线** | KDA 接 `glm5next`；GDN 要接 `qwen3_5` / `qwen3_next` 及其 MTP 层 |
| **配置判定** | 上游 `use_kda_recoverssm` 是按 `KimiLinearForCausalLM` / `KimiK3ForConditionalGeneration` 硬编码的；GDN 要在 `cache.py` / `vllm.py` 里加自己的分支（上游 #61309 为此改了 +40/−8） |

**不需要做的**（明确排除）：PLE / Qwen4Exp 的 `ple_recoverssm.py`（344 行）、KDA 的 grouped head 快路径（除非 GDN 也想要）、KV transfer（RFC 说单独验证）。

## 4.4 工作量拆解（人天，单人）

| 阶段 | 内容 | 估时 | 依据 |
|---|---|---|---|
| **P0 预研对齐** | 读上游 #61309 + 现有 KDA 1451 行；搭 FP32 顺序执行 oracle；跑通 KDA 现状做基线 | **3–5** | 代码量 700 + 1450 行要读透 |
| **P1 verify 内核** | `_gdn_recoverssm_verify_kernel`：gate/softplus、GQA 分组、记录写、null 行、L2 norm | **5–8** | 上游 175 行 kernel，但 Triton-Ascend 要重做 tile/UB |
| **P2 commit 内核** | `_commit_gdn_state_kernel`（含 boundary 二次 launch、对齐/非对齐两条路） | **5–8** | 上游 commit 约 90 行；boundary 逻辑最绕 |
| **P3 缓存与元数据** | 4 个 `kv_cache[i]` 的布局、`recoverssm_metadata`、cache spec、dtype | **3–5** | 对齐 `recoverssm_metadata.py` 161 行 + #63045 的"记录出页" |
| **P4 MRV2 集成** | `mamba_hybrid.py` 钩子、patch_v2、config 校验、qwen3_5/next 接线、`--use-replayssm` 分支 | **5–7** | 上游 config 改 40/8 + 模型改 87 行；ascend 侧还要 patch |
| **P5 正确性** | 边界矩阵（0/部分/全接受、连续多轮、重排、空行、graph padding、slot 复用）、FP32/BF16 | **5–8** | 对齐 #18765 的 384 单测 + 45 NPU 测的口径 |
| **P6 图与性能** | NPU Graph 路径、align 模式、profiling（cache 占用/verify/commit/TPOT） | **5–10** | RFC 说图后置，但"图下正确"是必须项 |
| **合计** | | **31–51 人天** | ≈ 单人 6–10 周，两人并行 4–5 周 |

### 与已有工作量的对照（用来校准估计）

| 参考物 | 规模 |
|---|---|
| 上游 GDN RecoverSSM（#61309，CUDA） | **41 文件 / +3150 −313**；其中核心新增约 1140 行（`recoverssm_gdn.py` 702 + 公共算子 200 + 工具 37 + 后端 198），改动约 240 行，其余是测试（643 + 281 + 129…） |
| Ascend KDA RecoverSSM（main，Triton-Ascend） | 核心 1451 行 + 161 行元数据 + 4 个 patch；#19133 整体 39 文件 / +3807−98（含 mooncake 等无关改动） |
| Ascend GDN MRV1（#18765，未合） | 21 文件 / +2191−17 |

⇒ **GDN 在 Ascend 上的净新增量预计 1200–1800 行**（内核 + 元数据 + patch + 接线），比 KDA 略少（GDN 的 gate 更简单），比 #18765 略多（因为要多做"记录出页 + MRV2 + 图"）。

## 4.5 风险清单（按 概率×影响）

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | **UB 溢出 / 静默算错** | 高 | 致命 | 记录是"每 token 一份"，`Q` 与 tile 数直接耦合。**照 #11236 的教训：改 Q 必须同步改 UB 核算并加断言** |
| R2 | **align 模式 boundary 折叠写错** | 中高 | 高 | 上游 commit 注释里专门处理了"窗口正好落在边界"的退化情况；单测必须覆盖 `steps % block == 0 / -1` |
| R3 | **MRV2 的 ragged / padding 行语义** | 中高 | 高 | RFC 专门点名；参考 KDA 的 null 行处理 + `gdn_attn_builder` 的 padding 逻辑 |
| R4 | **上游 #61309 未合、接口会变** | 中 | 中 | 把 Ascend 侧做成 patch/plugin（学 `patch_kda_recoverssm.py` 的做法），减少对上游内部接口的耦合 |
| R5 | **Triton-Ascend 语义差异** | 中 | 中 | 已知：`tf32x3` 不支持（只接受 `('ieee','hf32')`）；`softplus` 要手写线性分支 |
| R6 | **A2 未验证** | 高 | 中 | KDA 文档明确 A2/950PR 未验证；若目标机器是 A2，**先补 A2 验证再谈 GDN** |
| R7 | **精度** | 中 | 中 | 记录用 FP32；state 用 FP32；BF16 只用于激活。**记录里的 `decay` 必须存"实际乘上去的那个 float"**（上游代码注释特意强调） |
| R8 | **性能不达预期** | 中 | 中 | 收益取决于 cache 压力；把 `cache 占用 / verify / commit / TPOT` 四个指标分开测，别只看 TPOT |

## 4.6 建议的实施顺序

```
第 1 步（1 周）  P0：读透 + 建 oracle。先证明"你能用 FP32 顺序执行算出正确答案"，
                 这是后面所有对比的基准（#18765 就是这么做的）
第 2 步（2 周）  P1+P2：verify + commit 算子，先不做 boundary、不做 grouped，
                 单请求单 head 先跑通，再上 batch / 多 tile
第 3 步（1.5 周）P3+P4：缓存布局 + 记录出页 + MRV2 接线，跑通 eager 端到端
第 4 步（1.5 周）P5：边界矩阵测试（这是 RFC 里写得最细的部分，别省）
第 5 步（1–2 周）P6：NPU Graph、align 模式、性能profiling
```

**Go / No-Go 判据**（第 2 步结束时）：

- commit 重建出的状态与 FP32 顺序执行的相对误差 < 1e-5（FP32 路径）；
- `Q=1`（无投机）时，RecoverSSM 路径与原生路径**输出逐位一致**；
- 单层 commit 延迟 < 该层状态一次完整写的 2 倍。

---

# 5. 收益与代价量化（算例：Qwen3.8-27B）

## 5.1 算例模型参数

取自 HF `Qwen/Qwen3.8-27B` 的 `config.json`（`model_type: qwen3_5`，`architectures: Qwen3_5ForConditionalGeneration`）：

| 参数 | 值 | 说明 |
|---|---|---|
| `num_hidden_layers` | **64** | 其中 `linear_attention` **48 层**、`full_attention` **16 层**（`full_attention_interval=4`） |
| `linear_num_value_heads` | **48** | 每个 value head 一份状态 |
| `linear_num_key_heads` | 16 | GQA：3 个 value head 共 1 个 key head |
| `linear_key_head_dim` / `linear_value_head_dim` | **128 / 128** | 状态是 128×128 |
| `mamba_ssm_dtype` | **float32** | **状态默认 FP32**（4 字节的来源） |
| `linear_conv_kernel_dim` | 4 | 短卷积，状态长度 `3+Q` |
| `num_key_value_heads` / `head_dim` | 4 / 256 | 普通注意力层，决定 KV 大小 |
| `mtp_num_hidden_layers` | 1 | 原生 MTP |
| `hidden_size` | 5120 | — |

投机长度取 `num_spec=3` ⇒ **Q = 4**（1 个已确认 + 3 个候选），与 #18765 的 "MTP3" 口径一致。

## 5.2 三个基础量

```
一份状态（一层一序列）   S₁ = 48 heads × 128 × 128 × 4 B     = 3.00 MiB
一份状态（48 层一序列）  S  = 48 × 3 MiB                     = 144 MiB
一个 token 的记录（48 层）R  = 48 × 48 × (128+128+1) × 4 B   = 2.26 MiB
普通注意力的 KV          = 16 层 × 2 × 4 head × 256 × 2 B    = 64 KiB/token
卷积状态（一层，Q=4）    = 10240 通道 × (3+4) × 2 B          = 140 KiB
```

> **记录开销比例**：`R / S = 2.26 / 144 = 1.57%` 每 token，Q=4 时 **6.3%**。

## 5.3 收益与代价的来源（每请求每步，Q=4）

| 类别 | 项 | 量 |
|---|---|---|
| **收益 ①** | 状态写：Q 份 → 1 份 | **省 432 MiB** |
| 代价 C1 | commit 要多读一次状态 | 花 144 MiB |
| 代价 C2 | commit 要多写一次状态 | 花 144 MiB |
| 代价 C3 | 记录写 + 读回 | 花 2 × 9.0 = 18 MiB |
| 代价 C4 | 折叠计算 | 301 MFLOP（见 5.6） |
| **收益 ②** | 容量：Q 个状态槽 → 1 个 | **省 423 MiB/请求**（持久占用） |

⇒ **带宽净收益 = 432 − (144 + 144 + 18) = 126 MiB/步/请求（1.60×）**。

**关键判断：带宽账只是小头（1.6×），容量账才是大头（见 5.5）。**

## 5.4 交通账（三种形态对比）

单位：MiB / 请求 / 单步；Q = 1 + `num_spec`

| Q | 现状（快照） | **RecoverSSM**<br>每步 commit | 收益 | **+ 延迟提交** | 收益 | 环形缓冲 L=16 | 收益 |
|---|---|---|---|---|---|---|---|
| 2 | 432 | 441 | **0.98×（亏）** | 297 | 1.45× | 167 | 2.59× |
| 3 | 576 | 446 | 1.29× | 302 | 1.91× | 169 | 3.41× |
| **4** | **720** | **450** | **1.60×** | **306** | **2.35×** | **171** | **4.21×** |
| 6 | 1008 | 459 | 2.20× | 315 | 3.20× | 176 | 5.74× |
| 8 | 1296 | 468 | 2.77× | 324 | 4.00× | 180 | 7.20× |

**三个结论：**

1. **交叉点在 Q ≈ 2.06** ⇒ `num_spec ≤ 1` 时 **RecoverSSM 反而是亏的**。这解释了上游 #63045 的原话："*With its Triton kernels it was **slower per decode step** than speculative decoding without it*" —— 小 Q + kernel 未优化叠加的结果。
2. **"延迟提交"（把 commit 折进下一次 verify）把 1.60× 拉到 2.35×**，省掉的正是 C1 那 144 MiB 重复读。
3. **论文的环形缓冲版能到 4.21×**，因为它把 C2 也摊到 16 步一次。

> ⚠️ **align 模式（开前缀缓存）会打折**：跨块边界的请求还要多写一份 boundary 状态。Q=4 下若所有请求都跨边界，收益从 1.60× 掉到 **1.21×**；实际介于两者之间 ⇒ **上线前必须先测"跨边界请求占比"**。

## 5.5 容量账（真正的收益来源）

状态槽是**持久占用**，不是每步付一次：

| | 每请求 | B=256（整模型） | 每卡（TP=8） |
|---|---|---|---|
| 现状（Q=4） | 576 MiB | **144 GiB** | 18.0 GiB |
| RecoverSSM | 153 MiB | **38 GiB** | 4.8 GiB |
| **释放** | **423 MiB** | **106 GiB** | **13.2 GiB** |

106 GiB 能换成什么（本模型 KV = 64 KiB/token）：

| 换算 | 结果 |
|---|---|
| KV token 总量 | **≈ 173 万 token** |
| B=256 时每请求的上下文 | **+6.8K token** |
| 换成并发（4K 上下文） | **+423 个请求** |
| 换成并发（8K 上下文） | **+211 个请求** |
| 换成并发（32K 上下文） | **+53 个请求** |

> 单卡 64 GB 的 A2 上，**13.2 GiB/卡 ≈ 20% 的 HBM**。这正是 ReplaySSM 论文能报 1.8×、TreeWY 能报 2.0–2.45× KV 余量 / 5.6× p99 TTFT 改善的机制：**不是每一步变快，而是能同时服务更多请求。**

## 5.6 "重算"到底花多少

| 口径 | 数值（每请求每步，Q=4） | 评价 |
|---|---|---|
| **算力** | 4 步 × 48 层 × 48 head × 128×128 = **1.51 亿 MAC ≈ 301 MFLOP** | 对 27B 稠密前向（4 token ≈ 216 GFLOP）是 **0.14%**；按 3B 激活估也只有 **1.26%** ⇒ **算力可忽略** |
| **访存** | 读 144 + 写 144 = **288 MiB** | **这才是真实成本**：commit 是 memory-bound kernel |
| **时间**（B=256、TP=8、1.6 TB/s 假设） | 288 MiB × 256 ÷ 8 = 9.2 GiB/卡 ⇒ **约 5.7 ms/步** | 占单步 20–30% 量级 ⇒ **不可忽略** |

⇒ **"重算"只能按访存算，不能按算力算。** 它之所以划算，是因为替代了 432 MiB 的写。

**还能再压的四招（按收益排序）**：

1. **延迟提交**：省 144 MiB 读 ⇒ 1.60× → 2.35×；
2. **commit 跨层合并成一次 launch**（上游 #63045 已做；Ascend KDA 的 `_commit_kda_state_kernel` 也按 `state_base_addrs_ptr` 走层表）；
3. **value 维 tile 化**：只把 tile 拉进 UB 再写回，避免整份状态进出；
4. **混合精度**（DAMP 式）：`S` 从 FP32 降到 9.9 bit ⇒ 288 MiB 打三折，commit 的读一起降。

## 5.7 端到端能拿多少（Amdahl）

设 `f` = 状态访存占单步时间的比例，`r` = 5.4 表中的交通收益：

```
加速比 = 1 / ( (1 − f) + f / r )
```

| f | r=1.60（每步 commit） | r=2.35（+延迟提交） | r=4.21（环形缓冲） |
|---|---|---|---|
| 30% | 1.10× | 1.16× | 1.24× |
| 50% | 1.23× | 1.41× | 1.61× |
| 60% | 1.30× | 1.57× | **1.88×** |

**与已公开实测对照**（用于校准模型，非本报告实测）：

| 来源 | 实测 | 对应形态 | 反推 f |
|---|---|---|---|
| ReplaySSM 论文（投机，120B） | **≈1.8×** | 环形缓冲（4.21×） | ≈0.58 |
| ReplaySSM 论文（标准解码） | 1.20–1.47× | 环形缓冲（~1.9×） | ≈0.5–0.8 |
| DeltaLog（只减写回，单 token） | 1.05–1.20× | 减写（1.6×） | ≈0.2–0.35 |
| #63045（Triton 版 RecoverSSM） | **比不做还慢** | 每步 commit，Q 小 | — |

**结论：纯带宽账落在 1.1–1.6×；要拿 1.8× 必须靠容量（释放 HBM 换并发）。**

## 5.8 Ascend 上的修正

| 因素 | 影响 |
|---|---|
| **带宽更低** | 910B 约 1.6 TB/s vs H200 约 4.8 TB/s。同样省 1 GiB，**在 Ascend 上省下的时间约是 GPU 的 3 倍** ⇒ 带宽账的相对收益更好 |
| **机器平衡更偏带宽** | Ascend 约 235 FLOP/byte（bf16）vs H100 约 295 ⇒ **更"饿"带宽**，访存优化更值钱 |
| **HBM 更小** | 单卡 64 GB，而本例释放 13.2 GiB/卡（≈20%）⇒ **容量账更容易直接转成并发** |
| **UB 约束** | commit 的 tile 必须按 `RECOVERSSM_COMMIT_UB_MATRICES` / `RECOVERSSM_VALUE_TILE` 之类常量算；放不下就要多趟，把 C1/C2 的 288 MiB 放大 ⇒ **UB 预算就是这个方案的性能上限** |

## 5.9 什么时候不值得做

| 条件 | 判断 |
|---|---|
| `num_spec ≤ 1`（Q ≤ 2） | **不值得**：交通 0.98×，白付复杂度 |
| 显存不紧张、f < 20% | 带宽收益 < 6%，只剩容量收益（本就不缺并发时等于白做） |
| 必须开 align 且跨边界请求占比高 | 收益 1.60× → 1.21×，**先测占比** |
| 部署需要 PP > 1 / DCP > 1 | RFC 当前限制 PP=1、DCP=1、PCP=1，**先确认形态** |

## 5.10 通用工作表（换模型直接套）

```
S₁ = value_heads × K × V × 4                          # 一份状态，一层（字节）
S  = linear_layers × S₁                               # 一份状态，全层，一序列
R  = linear_layers × value_heads × (V + K + 1) × 4    # 一个 token 的记录，全层
KV = full_layers × 2 × kv_heads × head_dim × kv_bytes  # 每 token 的 KV

现状        = S × (1 + Q)      # 读 1 份、写 Q 份
每步 commit = 3S + 2RQ         # verify 读 + commit 读 + commit 写 + 两趟记录
延迟提交     = 2S + 2RQ
环形缓冲(L)  = S + RQ + 2S/L

持久容量（现状）    = Q × S      # 注意是 Q 份，不是 Q+1
持久容量（Recover） = S + RQ     # 1 份 + 记录
释放                = S × (Q − 1) − RQ
端到端加速比        = 1 / ((1 − f) + f / (现状 ÷ 形态))
```

**交叉校验（用 #18765 的公开数字验这个模型）**：

| 项 | 本模型算出 | #18765 实测 |
|---|---|---|
| 每层每请求：状态 + 记录（MTP3） | **3.047 MiB** | **3.188 MiB**（差额是 conv / 历史 / 页填充） |
| 4 份快照 | **12.00 MiB** | **12 MiB** |

⇒ 模型与实测吻合，可直接用于其他模型 / 卡型的预估。

---

# 6. 出处与核对方式

| 内容 | 来源（2026-11-02 核对） |
|---|---|
| RFC 要求、验证清单、性能指标 | <https://github.com/vllm-project/vllm-ascend/issues/19987>（issue body 全文） |
| 上位 RFC、PR 依赖关系 | <https://github.com/vllm-project/vllm-ascend/issues/19733> |
| KDA 设计文档（公式、参数、支持范围） | `vllm_ascend/ops/triton/docs/kda_recoverssm.md`（355 行） |
| KDA 算子实现 | `vllm_ascend/ops/triton/kda/recoverssm.py`（1451 行） |
| KDA 配置与限制 | `vllm_ascend/models/glm5next/recoverssm_config.py`、`patch/platform/patch_replayssm_config.py` |
| 版本 / 依赖 | `pyproject.toml`、`requirements.txt`、`main@31fa24405a`、tag `v0.31.0rc1` |
| 上游 GDN 参考实现 | `vllm/model_executor/layers/mamba/gdn/recoverssm_gdn.py` @ `Yuchi017/vllm:bf0219d8`（PR #61309 head） |
| 上游 GDN 文件构成 | `gh api repos/vllm-project/vllm/pulls/61309/files` |
| Ascend MRV1 GDN 原型 | `gh api repos/vllm-project/vllm-ascend/pulls/18765/files`（21 文件 / +2191−17） |
| 论文原理、精度表 | ReplaySSM，<https://arxiv.org/abs/2610.13778> |
| **Qwen3.8-27B 模型结构**（层数/头数/dtype/KV） | `https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json`（2026-11-03 取） |
| 带宽 / 机器平衡等硬件数字 | **假设值**（910B ≈1.6 TB/s、H200 ≈4.8 TB/s），请按实际卡型替换 |

**未验证 / 需自己确认的事项**

- 人天估算是**基于代码量与结构差异的推算**，未做真实排期；
- A2 / 950PR 上 KDA 是否可用：官方文档说"未验证"，需实测；
- 上游 #61309 / #58863 仍在 open，**接口（`use_kda_recoverssm` 的判定方式、记录张量名）可能变化**；
- `ple_recoverssm.py`（PLE）与本次 GDN 目标无关，但若目标模型是 Qwen4Exp 系，需要单独评估；
- §5 的量化是**解析模型**（按字节数和 Amdahl 推的），不是实测；`f`（状态访存占比）必须在自己机器上 profile；
- §5 假定 TP=8、B=256、1.6 TB/s，**换卡型/换并行度要重算**，工作表见 §5.10。
