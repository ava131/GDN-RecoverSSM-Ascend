# 可行性验证报告：GDN RecoverSSM on Ascend A2（MTP3 + align）

> 目标配置：**A2（910B）+ MTP3（Q=4）+ prefix cache（`mamba_cache_mode=align`）**
> 基线：vLLM `v0.30.0` + vllm-ascend `v0.30.0`
> 交付形态：**vLLM 插件（`vllm.general_plugins`），不改 vLLM / vllm-ascend 任何源码**，见 `plugin/`
> 结论：**可行。**

---

## 1. 逐项验证（每条都对着 clone 的代码核过）

### V1 ✅ RecoverSSM 的生命周期在 0.30.0 已存在

`vllm/v1/worker/gpu/model_states/recoverssm.py`（101 行）：

```python
class RecoverSSMState:
    def record_step(attn_metadata, attn_groups, *, for_capture): ...   # 收集 RecoverSSMMetadata；捕获时跳过
    def commit_step(num_sampled, idx_mapping, *, state_indices, num_accepted_tokens):
        for metadata in step:
            postprocess_meta = metadata.commit_recoverssm_state(num_sampled)   # ← 具体实现由模型/后端提供
            ...  _postprocess_recoverssm_align_kernel[...]                      # 更新 align 模式的 running column
```

`MambaHybridModelState`（`vllm/v1/worker/gpu/model_states/mamba_hybrid.py`）：
- L97：`self.recoverssm = RecoverSSMState() if self.cache_config.use_kda_recoverssm else None`
- L332：`preprocess` 里 `record_step(...)`
- L366：`postprocess_state` 里 `commit_step(...)`，align 时传 `state_indices=self._mamba_state_idx_gpu`

⇒ **"谁在什么时候提交"这件事，0.30.0 已经接好了。我们只要提供一个实现 `RecoverSSMMetadata` 的对象。**

### V2 ✅ align 模式（prefix cache）在 0.30.0 已支持，boundary 逻辑有现成参考

`vllm/config/vllm.py` L3234：

```python
if self.cache_config.mamba_cache_mode not in ("none", "align"):
    raise ValueError("RecoverSSM supports only none and align Mamba cache modes")
if mamba_cache_mode == "align" and not self.use_v2_model_runner:
    raise ValueError("RecoverSSM with align mode requires VLLM_USE_V2_MODEL_RUNNER=1")
```

两套参考实现的 boundary 做法不同，都可抄：

| 实现 | boundary 做法 |
|---|---|
| 0.30.0 的 KDA（`_commit_kda_state_kernel`） | **单次 launch**，`ALIGN_MODE: tl.constexpr`，同一 kernel 里既写 final 又写 boundary |
| 上游 GDN（#61309，本插件移植的版本） | **两次 launch**，`BOUNDARY: tl.constexpr`，先 final 后 boundary |

本插件跟随上游 GDN 的两次 launch（`for boundary in (False, True) if align else (False,)`）。

关键边界：`next_boundary == final_num_computed`（正好落在块边界）时，boundary 与 final 是同一个块，**不能重复写**——已用 oracle 单测覆盖（`test_commit_plan_boundary_cases`）。

### V3 ✅ A2（910B）被显式覆盖

- `vllm_ascend/ops/triton/triton_utils.py::get_ub_size_bytes()` 文档原文：
  *"auto-detected from device properties …, with a safe fallback of **192 KB (covering Ascend 910B/A3)**"*
- `is_950()` 只判 A5；**移植的 kernel 里没有任何 `is_950` 分支**（grep 只在注释里出现），A2 走通用路径。
- KDA 文档说"A2 未验证"是指**测试矩阵**没覆盖 A2，不是代码不支持。

### V4 ✅ 只需一处配置补丁（插件内）

`vllm/config/vllm.py` L3215–3227：
```python
self.cache_config.use_kda_recoverssm = self.num_speculative_tokens > 0
...
if architecture not in ("KimiLinearForCausalLM", "KimiK3ForConditionalGeneration"):
    raise ValueError("RecoverSSM is only supported for Kimi-K3 KDA")
```
插件用 `config_patch.py` 包装这个 validator：GDN 架构走自己的校验，其余架构仍走原逻辑。

### V5 ✅ **Ascend 有自己的 GDN 层 ⇒ 全部改动都能挂在 Python 层**

- `vllm_ascend/ops/gdn.py:220` `class AscendGatedDeltaNetAttention(GatedDeltaNetAttention)`
  - `get_state_shape()` / `get_state_dtype()`：**在 Ascend 层上**，可包装
  - `get_attn_backend()` → `AscendGDNFusedAttentionBackend`，可包装
  - `_forward_core()` 里调用模块级函数 `recurrent_gated_delta_rule`，**模块属性可替换**
- 该层由 `vllm_ascend/utils.py:946` 注册为自定义算子，替换 vLLM 的实现

⇒ **不需要改源码**：所有落点都是类属性 / 模块全局，插件在 import 时替换即可。

### V5.1 ⚠️ 一个致命集成陷阱（已修）

`RecoverSSMState.record_step` 只在上游基类的 **`MambaHybridModelState.prepare_attn`** 里被调用；
而 Ascend 的 `AscendMambaHybridModelState` **把 `prepare_attn` 整个覆盖了，且没有调 `super()`**：

```
vllm/v1/worker/gpu/model_states/mamba_hybrid.py:232  def prepare_attn   ← 里面有 record_step（L332）
vllm-ascend/.../v2/model_states/mamba_hybrid.py:280  def prepare_attn   ← 没有任何 recoverssm 引用
```

后果：`_step` 永远是 `None` → `commit_step` 收到空列表 → **检查点永不前进，输出静默错误**（不报错，只是越跑越偏）。

修法：插件额外包一层 `AscendMambaHybridModelState.prepare_attn`，在返回前补调
`recoverssm.record_step(attn_metadata, attn_groups, for_capture=...)`。
`postprocess_state` 没有被 Ascend 覆盖，`commit_step` 那边是正常的。

### V6 ✅ 记录"放在 KV page 里"（0.30.0 的设计）

- `vllm/models/kimi_k3/nvidia/kda.py` L566：`MambaStateShapeCalculator.append_kda_recoverssm_record(...)`
- L918：`conv_state, recurrent_state, *recoverssm_records = self.kv_cache`

⇒ 0.30.0 是**记录塞进 Mamba page** 的老设计；RFC #19987 要求的"记录移出 page"是 main 上 #63045 的新设计。
本实现跟随 0.30.0 的 in-page 设计，并在文档里注明。

### V7 ✅ 内核是纯 Triton，依赖只有 4 个

移植后的 `kernels.py` 顶层 import：
```python
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID     # == 0
from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes
```
（另有一个函数内延迟 import：`is_conv_state_dim_first`。）

### V8 ⚠️ Qwen3.8-27B 的 GDN 参数（用于定 tile）

`config.json`：48 个 GDN 层、`linear_num_value_heads=48`、`linear_num_key_heads=16`（GQA 3:1）、
`head_k_dim=head_v_dim=128`、`linear_conv_kernel_dim=4`、`mamba_ssm_dtype=float32`、`mtp_num_hidden_layers=1`。

⇒ 每 head 状态 `128×128` FP32；记录 `V+K+1 = 257` float/token/head。

---

## 2. 架构：插件方式（零源码改动）

```
plugin/
├── pyproject.toml                     # entry point: vllm.general_plugins → install()
├── README.md                          # 安装/运行/验收
├── gdn_recoverssm_ascend/
│   ├── __init__.py                    # install()：torch_npu 存在才激活
│   ├── config_patch.py                # 包 VllmConfig.validate_mamba_cached_kernel
│   ├── layer_patch.py                 # 包 AscendGatedDeltaNetAttention + 替换 recurrent_gated_delta_rule
│   ├── builder.py                     # RecoverSSM 的 metadata builder + backend
│   └── kernels.py                     # 4 个 kernel + CommitContext（1175 行）
└── tests/
    ├── test_math_oracle.py            # 无需 NPU，验证记录/提交语义
    └── test_layer_patch_mock.py       # 无需 NPU，验证插件接线
```

| 被包装的对象 | 做什么 |
|---|---|
| `VllmConfig.validate_mamba_cached_kernel` | 允许 `Qwen3_5*`；置 `use_kda_recoverssm`；形状/并行/缓存模式的快速失败校验 |
| `AscendGatedDeltaNetAttention.__init__` | 记录 `self.use_gdn_recoverssm` |
| `…get_state_shape` / `…get_state_dtype` | 追加 FP32 记录 `[HV_tp, Q, V+K+1]`（成为 `kv_cache[2]`） |
| `…get_attn_backend` | 返回 RecoverSSM 的 metadata backend |
| `…_forward_core` | 压栈"当前层"，供下面的 shim 判断 |
| `vllm_ascend.ops.gdn.recurrent_gated_delta_rule` | **只**把投机那次调用（带 `num_accepted_tokens` 的）改道到 verify；普通 decode 原样 |

**commit 完全不用 patch**：vLLM 0.30.0 会在 `postprocess_state` 里调到我们 backend 产出的 metadata 的 `commit_recoverssm_state()`。

**没有改一行 vLLM / vllm-ascend 源码**——`git status` 在两边都是干净的。

---

## 3. 目标配置带来的三个专门设计

### 3.1 A2（910B）：只用通用路径，tile 按 UB 现算

- 无 `is_950` 分支、无 grouped-head 快路径；
- `BV` / `BLOCK_D` 由 `get_ub_size_bytes()` 运行时算，并带**显式 UB 预算与断言**（192 KiB 下 verify BV=32、commit BV=16、conv 256，与上游默认一致；16 KiB 时 BV 降到 8；4 KiB 直接报错）。

### 3.2 Q=4：编译期常量

- `SPEC_QUERY_LEN = 4` 作为 `tl.constexpr`，per-token 循环用 `tl.static_range`；
- `MAX_SPEC_QUERY_LEN = 8` 上限，超过直接报错；
- Q 与 tile 的乘积进 UB 预算函数。

### 3.3 align 模式

- commit 分两次 launch（final / boundary），`BOUNDARY: tl.constexpr`；
- 边界算术已用 oracle 单测锁定（含"正好落在边界"的退化情形）。

---

## 4. 已知风险

| # | 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|---|
| R1 | Triton-Ascend 不支持某些写法 | 中高 | 高 | 移植时已逐项列出 10 个"语义不变的 fallback"（见 `docs/KERNEL_NOTES.md`） |
| R2 | UB 超限 | 中 | 高 | 运行时按 UB 算 tile + 断言；小 UB 自动降 tile |
| R3 | align 的 boundary off-by-one | 中 | 高 | oracle 单测 + 上机验收清单 |
| R4 | A2 未经验证 | 中高 | 中 | `GDN_RECOVERSSM_DISABLE=1` 一键回退；先 `mamba_cache_mode=none` |
| R5 | MRV2 在 A2 上的 spec decode 路径 | 中 | 中 | 先跑 baseline 再开开关 |
| R6 | 记录在 page 内（与 RFC 的"移出"不同） | 确定 | 低 | 文档注明，后续再迁 |

---

## 5. 上机验收清单

1. `Q=1`（无投机）时，RecoverSSM 路径与原生路径**输出逐位一致**；
2. `Q=4` 任意接受长度 `a∈[0,4]`，重建状态与 FP32 顺序 oracle 相对误差 < 1e-5；
3. `a=0` 时检查点**逐位不变**；
4. align：跨块边界请求的状态正确（`num_computed % block == 0` 与 `== block-1` 两种）；
5. `FULL_DECODE_ONLY` 图捕获后与 eager 一致；
6. 端到端：Qwen3.8-27B + MTP3 的 greedy 输出与不开 `--use-replayssm` 时 `check_logprobs_close` 通过。
