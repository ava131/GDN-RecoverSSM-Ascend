# op-dev-v2：正常启动时看什么

本分支从 main 的 `9c5ce344d33f94ac0416e2ba1c12ea2ae587a467` 开始；旧 op-dev 保留。用户启动参数、并发、graph、async、MTP和prefix cache沿用原来的配置，不增加参数，也不要求关闭功能测试。

**最新实现已扩展mixed batch，详细合同和日志见 [PD融合/mixed接入说明](OPDEV_V2_MIXED_BATCH.md)。这与是否采用PD分离部署无关。**

## 代码身份与启动检查

拉取本分支并按现有方式安装插件后，继续使用原启动命令。AC库按源码指纹在启动时构建/加载；仅git pull而没有切换分支不会拿到本分支。安装路径与运行路径必须一致。

在服务器的项目仓库、原 Python 环境中执行：

```bash
git fetch origin op-dev-v2
git switch op-dev-v2
git pull --ff-only origin op-dev-v2
python -m pip install -e ./plugin --no-deps
```

然后原样执行已保存的 Replay server 启动命令及完整压测。不需要手动清构建缓存；源码变化会产生新 fingerprint。首次构建和启动自检有开销，后续启动可命中编译缓存；记录整批时间时应保留其具体计时范围，不能混算。

正常 INFO 日志保留三类：

- `GDN_READY json=...`。每个 TP rank 一条。实际模型绑定、强制自检和原生 warmup 完成后才打印，包含生效参数、state/records 布局、各组件版本、源码及 AscendC 构建指纹。
- `GDN_MEMORY json=...`。每个 rank 三个阶段的 KV 页配置、预算及 PyTorch allocator 快照。
- `GDN_PHASE_PERF json=...`。rank 0 汇总完成步骤，包含分阶段工作量、选中的 prefill 分支、融合 verify 和 fold、Host 时间、graph bucket/padding。看 `totals.commits` 是否随请求增长。看 `paths.counts.verify_fused_mixed_model_calls` 确认混批进入融合。图重放不重复执行 Python，因此可结合 READY 的捕获计数和运行时 `graph_submissions` 核对纯 decode。

失败阶段、回退和配置变化仍可定位。成功自检及逐图等历史细节降到 DEBUG。source fingerprint 识别安装内容，build fingerprint 识别实际编译输入。 [字段说明与提取脚本](GDN_PERF_EVIDENCE_AND_SCENARIOS.md)

## 本轮实现目标

1. 窗口计算结束后集中写出结果和records，避免同一个输出事件在多个token中重复发布。保持typed-SoA、GQA共享和合法stride能力。
2. 把norm/gate放进AC verify，保留现有输入/输出dtype舍入。已处理的中间结果无需再单独经过显存；实际选择由内部签名检查决定。
3. 最后token直接算输出，不生成后续无人读取的完整临时state。correction/key/decay记录仍完整，采样后的恢复不删除。
4. fold按窗口加载decay，避免递推中逐个读取GM标量。

这些改变都需要数值验证。尤其最后token的公式调整了浮点计算顺序，不能因为数学恒等就声称逐bit相同。本地CPU检查不能证明CANN编译、设备数值或性能已经通过。

融合签名修复后须正常重启 server，让图重新捕获；仅对运行中的进程执行 git pull 不会替换已有图。仍使用原命令，不需要清缓存或新增参数。

## 压测结束后看什么

保持完整负载，用脚本同时提取服务端统计和客户端成绩。把文件名换成实际保存的日志。只有服务端日志时省略 `--client`。客户端的 `Benchmark Duration` 衡量整批端到端时间，worker 时钟测量自己的观察窗口。

```bash
python3 tools/extract_gdn_perf.py \
  --server server.log --client ais.log --json-out replay-evidence.json
```

重点核对成功请求数、输入与输出量、TTFT/TPOT、接受长度、平均批次、graph 大小和实际 prefill 分支。Host 时间只覆盖提交或原有等待，路径计数只统计 Python 分派。Base 20 到 Replay 25 的历史 eager 容量收益已有实测，后续比较关心这项收益如何转成整批吞吐。 [历史容量结果与模型](GDN_PERF_EVIDENCE_AND_SCENARIOS.md)

无需增加启动参数或单独安排关闭功能的实验。
