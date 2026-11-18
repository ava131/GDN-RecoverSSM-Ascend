# GDN-RecoverSSM-Ascend

以 vLLM 插件形式在 Ascend A2（910B）上实现 GDN RecoverSSM：投机验证只读 checkpoint，保存紧凑记录，采样后重建被接受的状态。目标是降低状态缓存需求并缩短端到端时间。

当前分支关系：

- `main`：`9c5ce34`，现阶段已有运行结果的基线。
- `op-dev-v2`：从上述 main 新建，本轮 AscendC 优化；CPU 检查已通过，设备编译、数值、图运行和性能仍待部署验证。
- `op-dev`：保留原开发历史，本轮未在该分支继续叠加。

本轮说明见 [设计与交接](docs/OPDEV_V2_DESIGN.md) 和 [正常启动及日志验收](docs/OPDEV_V2_SERVER_VALIDATION.md)。沿用现有完整启动命令，无新增用户开关。

11月18日已修复首版 v2 错误拒绝 BF16 dt_bias 的融合接入问题，并删除融合路径的 QKV 整段副本。原因、证据与收益边界见 [v2 无明显收益复查](docs/OPDEV_V2_NO_GAIN_REVIEW_20261118.md)。

## 执行路径

```text
投机验证：checkpoint 只读 → UB 中递推 Q 步 → 输出及 typed-SoA records
采样之后：按接受长度折叠 records → final checkpoint 及需要的 align 边界
```

correction、decay 使用 FP32；key 保留 activation dtype，并在 GQA value heads 间共享。resident state 页数由原生缓存模式和物理页规则决定，不能把 align 模式的整个状态池简单视为每请求固定一页。实际 dtype、shape、stride 和 records 字节数见 `GDN_PREPARED_CONTEXT`。

AscendC 实现 verify 和 grouped fold；plan、conv compaction、align publisher 继续使用已有 Triton 实现。代码只在 `plugin/` 内接入，不修改 vLLM、vllm-ascend 或 FLA 源码。

## 安装和检查

```bash
python -m pip install -e ./plugin --no-deps
# 随后使用保存的原 Replay server 启动命令。
```

AscendC 库在 server 启动时按源码和工具链指纹自动构建或命中缓存；构建、自检错误阻止 ready。开发机没有 NPU/CANN，本地测试不能替代设备验证。

```bash
.venv/bin/python -m pytest plugin/tests -q
```

## 文档

| 路径 | 内容 |
|---|---|
| `plugin/` | 插件和测试 |
| [OPDEV_V2_DESIGN.md](docs/OPDEV_V2_DESIGN.md) | 本轮改动、数学、同步合同和交接 |
| [OPDEV_V2_SERVER_VALIDATION.md](docs/OPDEV_V2_SERVER_VALIDATION.md) | 原配置运行时的日志检查与验收 |
| [ASCENDC_OP_DEV.md](docs/ASCENDC_OP_DEV.md) | 初版 AC 设计历史，版本和资源数字以 v2 文档为准 |
| [GDN-RecoverSSM-实现方案.md](docs/GDN-RecoverSSM-实现方案.md) | 原理、RFC、KDA 和收益建模背景 |
| `docs/reference/` | 参考实现 |

较早的 runbook、规划及性能审查保留作历史记录；本分支的部署步骤以 v2 验收文档为准。
