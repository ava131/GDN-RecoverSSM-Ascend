# GDN-RecoverSSM-Ascend

当前 `op-dev` 为 AscendC 算子开发分支（`0.3.0.dev0`）。verify/grouped fold 已接入 AC，尚待 CANN 编译和 NPU 验证；部署及验收见 [AscendC 分支说明](docs/ASCENDC_OP_DEV.md)。main 保持原稳定基线。

在 Ascend A2（910B）上为 **GDN** 实现 **RecoverSSM**（投机解码只保留 1 份检查点 + 紧凑记录，
采样后重建被接受的状态），目标配置 **MTP3（Q=4）+ prefix cache（align）**。

**交付形态：vLLM 插件。不改 vLLM / vllm-ascend 任何源码**（两棵树的 `git status` 都是干净的）。

## 目录

| 路径 | 内容 |
|---|---|
| **`plugin/`** | ⭐ 实现本体：`pip install -e plugin` 即生效，见 [`plugin/README.md`](plugin/README.md) |
| `docs/FEASIBILITY.md` | 可行性验证（8 条逐项核过源码）+ 插件架构 + 风险 |
| `docs/RUNBOOK.md` | 上机手册：装插件、先基线后开关、6 条验收、回退方案 |
| `docs/KERNEL_NOTES.md` | 内核移植说明：与上游的逐函数对应 + 10 条 Triton-Ascend fallback + UB 预算 |
| `docs/GDN-RecoverSSM-实现方案.md` | 原理与公式、issue #19987 要求、KDA 现状、工作量、Qwen3.8-27B 收益量化 |
| `docs/PLAN.md` | 早期规划（按里程碑），已被插件方案取代，保留作历史 |
| `docs/七个算法汇总.md` | 背景：7 个方案横向对比 |
| `docs/ReplaySSM-2610.13778.pdf` | 论文原文 |
| `docs/reference/` | vendor 的参考实现（含 0.30.0 缺失的 Ascend KDA 与上游 GDN 实现） |
| `vllm/` `vllm-ascend/` | 基线源码，v0.30.0，**未改动** |

## 它做了什么

```
投机验证：读 1 份 checkpoint（只读）→ 寄存器里连推 Q 步
         → 写输出 + 每 token 一条 FP32 记录 [corr(V), k̂(K), decay(1)]
采样之后：commit 只折叠被接受前缀 → 写回 1 份 checkpoint（+ align 模式下的块边界状态）
```

状态槽从 **Q 份降到 1 份**（`num_speculative_blocks = 0` 由上游 `--use-replayssm` 自动生效）。

## 快速开始

```bash
cd plugin && python -m pip install -e .
# 上机步骤见 docs/RUNBOOK.md
```

## 本地可跑的验证（无需 NPU）

```bash
cd plugin
python tests/test_math_oracle.py       # 4 项：记录布局、任意接受长度的提交、align 边界算术
PYTHONPATH=. python tests/test_layer_patch_mock.py   # 3 项：插件接线
```

两套都已通过（`4 oracle checks passed` / `3 wiring checks passed`）。

## 状态

- ✅ 可行性验证、内核移植（1175 行）、插件接线、本地数值/接线测试
- ⏳ 待上机：kernel 编译、Q=1 逐位一致、align 边界、图捕获、端到端精度与性能
