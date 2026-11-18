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

在server日志中保留这些行：

- `GDN_CONFIG patch=ascendc-op-dev-v2-perf-evidence` 带配置快照、源码与启动参数指纹。该版沿用 `ascendc-op-dev-v2-mixed-fused` 的混批路径，并增加 [性能证据采集与脚本](GDN_PERF_EVIDENCE_AND_SCENARIOS.md)。再早的 `ascendc-op-dev-v2-bf16-bias-qkv-views` 只覆盖 pure-spec。
- `GDN_ASCENDC loaded`：实际加载的库、build fingerprint、是否命中缓存及算子入口。
- `GDN_PREPARED_CONTEXT`：实际state dtype、shape、stride、record key dtype、records字节数。这里的数字比配置摘要更直接。
- `GDN_ASCENDC_SELFTEST`及现有grouped/conv自检结果：启动时在自有缓冲中验证，失败应阻止ready，不能当成运行中fallback理由。修正版应包含 `dt_bias=fp32,bf16 bias_storage_promotion=PASS`。
- `GDN_ASCENDC_FUSION model_dispatch=PASS`：模型路径首次提交融合 verify；与专项自检分开计数，并打印真实 `A_log`、`dt_bias` dtype 与 Q/V token stride。TP4、HK4/HV12/K128 的 packed view token stride 应为 2560。若出现 `eligible=0`，同一行说明保留独立 helper 的原因，verify 仍按已有后端选择。
- `GDN_READY`：只有准备和原生warmup完成后才打印。新增 `fused_model_dispatches` 和 `fused_total_dispatches`，明确模型调用与包含专项自检的调用总数。eager warmup若未构造pure-spec行，模型计数可以暂为0；普通模型首次进入时仍应看到 `model_dispatch=PASS`，不能只拿自检PASS证明模型融合。
- `GDN_MIXED_ADAPTER READY`确认实际模型方法的五处patch，`GDN_MIXED_SELFTEST PASS`检查自有缓冲的分区/gate/merge。实际混批应看到 `model_dispatch=PASS phase=mixed`。

source fingerprint识别安装后的真实文件内容，包括editable install。build fingerprint识别实际编译输入。它们与git提交号属于不同口径，不能用单个旧日志证明新提交已执行。

## 本轮实现目标

1. 窗口计算结束后集中写出结果和records，避免同一个输出事件在多个token中重复发布。保持typed-SoA、GQA共享和合法stride能力。
2. 把norm/gate放进AC verify，保留现有输入/输出dtype舍入。已处理的中间结果无需再单独经过显存；实际选择由内部签名检查决定。
3. 最后token直接算输出，不生成后续无人读取的完整临时state。correction/key/decay记录仍完整，采样后的恢复不删除。
4. fold按窗口加载decay，避免递推中逐个读取GM标量。

这些改变都需要数值验证。尤其最后token的公式调整了浮点计算顺序，不能因为数学恒等就声称逐bit相同。本地CPU检查不能证明CANN编译、设备数值或性能已经通过。

融合签名修复后须正常重启 server，让图重新捕获；仅对运行中的进程执行 git pull 不会替换已有图。仍使用原命令，不需要清缓存或新增参数。

## 压测结束后看哪些统计

`GDN_PERF`：

- `spec_emitted_per_row`：每个spec请求行最终采样产出的平均token数，不等于仅draft接受数。
- `launches_per_commit`、`groups_per_commit`：提交结构是否保持预期；不能只看某个kernel变快，却增加大量提交。
- `graph_observed_steps`、`graph_hist`、`graph_padding_fraction`：覆盖legacy和updatable两种原生fullgraph分派。未观测到图时padding fraction为-1，表示未知，不表示没有padding。
- `host_submit_us_avg`：CPU提交时间，**不是设备kernel时间**。
- `worker_observed_s`：worker已观察窗口，不是客户端整批Duration；不包含最后输出后的空闲等待。

`GDN_WORKLOAD`：

- `mean_rows` / `peak_rows`：平均与峰值请求行。
- `mean_gdn_rows` / `mean_commit_rows`：实际GDN/恢复工作量口径。
- `accepted_output_hist`：产出token数分布，辅助识别接受行为是否变化。
- `batch_hist`：decode、prefill、mixed以及请求行数分布。字段不存在时phase为unknown，不读取设备Tensor补齐。
- `spec_phase_hist`：每个phase的spec请求行和产出，直接核对mixed覆盖率。
- `mean_computed_context` / `max_computed_context`：已有CPU输入镜像的上下文长度；async下可能是乐观镜像，日志明确标注，不能当成精确设备状态。

`GDN_PATHS`：

- `verify_fused`包含专项启动自检，单独大于零不能证明模型使用融合路径。
- `verify_fused_model_calls`只统计模型入口，配合 `model_dispatch=PASS` 确认 warmup/捕获/普通模型调用进入了融合路径。
- `verify_fused_mixed_model_calls`与`verify_fused_pure_spec_model_calls`区分模型路径。`verify_fused_saved_helpers`在mixed每次加2、pure-spec加3；mixed仍需non-spec gate，不能算整个gate被删除。
- 行尾`mixed={...}`汇总分区、gate、verify、直接merge的Host调用，startup staging probe不计入。
- fullgraph replay不会重复执行Python，**不能把这些计数当成全程设备kernel次数**。日志有 `HOST_DISPATCH_COUNTS_NOT_GRAPH_EXECUTIONS` 明确提示。

统计只使用runner已经完成的CPU输出和CPU batch信息，不增加NPU scalar读取、计时event或热路径synchronize。每200步、首几步、空闲和退出时输出汇总，避免每层每token刷日志。

## 性能判断

保持原有完整压测，保留客户端结果与本次server完整日志。首先看总请求成功数、总输入/输出token和整批Duration，然后看TTFT、TPOT、接受长度、平均batch、graph分布。容量增加和每轮执行成本会同时影响最终时间，不能用峰值batch替代吞吐成绩。

主基线仍是9c5ce34。相同完整负载的Duration下降才算端到端改善。CPU测试通过、启动自检通过、进入新算子、单个host提交时间下降，各自都不能单独证明性能已达标。

应保留四个 TP worker 的日志，确认它们都完成自检、ready，并正常处理到最后一个请求。最终汇总从三类行取数：

```bash
rg 'GDN_CONFIG|GDN_ASCENDC|GDN_PREPARED_CONTEXT|GDN_READY' server.log
rg 'GDN_PERF|GDN_WORKLOAD|GDN_PATHS' server.log
```

无需增加 debug 日志级别、改变 async/graph/batch/prefix 参数或单独安排关闭功能的实验。
