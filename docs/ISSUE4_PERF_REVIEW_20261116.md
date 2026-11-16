# #4 新结果复审：历史回退成立，图观测有缺口

2026-11-16。审查基线为 `op-dev acc4e08240e60286811dd87bf949b2da97eaf0e3`。本报告采用用户纠正后的两条新评论：[压测结果](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6911913013)、[服务日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6911925863)。此前误贴的830.38秒结果按用户要求排除，不作为本轮Base对照，不据此宣称新Replay获益。

**结论：新Replay相对旧AC的端到端时间确实回退约9.03%，本轮没有达到性能目标。现有证据不能把回退唯一归因于AC核，也不能把它解释为算法已经到头。日志还证实图观测只覆盖一条原生路径，必须先纠正这处明确缺陷。**

## 1. 实测变化

[历史表](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6906418603)与新Replay结果：

| 指标 | 旧AC | 新Replay | 变化 |
|---|---:|---:|---:|
| Benchmark Duration | 747.6488 s | 815.1825 s | +67.5337 s，+9.0328% |
| 平均TTFT | 24.5311 s | 25.1100 s | +0.5789 s，+2.3599% |
| 平均TPOT | 118.3 ms | 124.2 ms | +5.9 ms，+4.9873% |

新结果明确是120成功、0失败，输入2,827,697 token，输出122,880 token，平均输入23,564.14，固定每请求1024输出。若历史表对应相同输出工作量，则输出吞吐由约164.36降至150.74 token/s，下降约8.28%。

没有新的有效Base结果可用于同轮比较。即使启动参数未变，单次跨轮结果也不能分离kernel、缓存、请求执行顺序及运行环境的贡献；不能只挑一种原因。

### 主要差额在首token之后

固定1024输出时，平均延迟满足近似关系：

\[
\overline{E2EL}=\overline{TTFT}+1023\times\overline{TPOT}.
\]

新TPOT只保留0.1ms精度，因而会有四舍五入误差。旧均值按此推算约145.552秒；新实测152.1931秒，相差约6.6411秒/请求：其中首字前增加0.5789秒，首字后增加约6.0622秒。

因此本次没有证据说TTFT出现了先前误贴结果中的大幅跳升。当前主要表现是持续生成时间增加。TTFT增加与TPOT增加也不是两个互相独立的原因：生成阶段变慢会影响后续排队和请求准入。

Benchmark Duration是整批makespan，平均E2EL是各请求驻留时间，不能把平均每请求差额直接乘120当作总耗时差额。

## 2. 新日志确实补上了一些关键事实

- 当前实际checkpoint为 **FP32**，record key为 **BF16**，correction为 **FP32**；不再需要只根据HF默认精度猜测。
- 首个context有16层、record_rows128、record_bytes59,113,472；结合三组恢复结构，不能把这个单context字节数当成整模型总量。
- 恢复仍为每轮RecoverSSM，prepare的verify/fold backend是AC，plan/conv为Triton。
- 到step800，commits784、launches_per_commit3、groups_per_commit3，主机恢复提交平均811.2微秒；这是主机提交时间，不能据此说设备fold只花0.8ms。
- 该时点累计接受输出约3.1746/投机行，draft接受率约72.49%；没有同配置同区间的旧接受分布，不能把其差异确定归为回退原因。

截至该片段的累计主机恢复提交约0.636秒，但它不包含设备执行，也不足以排除设备核时间或其他主机路径。

## 3. 这份日志不是全程收尾统计

压测结果标记主请求批次约在08:29:45开始，08:43:21结束。服务日志更早在08:28:15就开始记录单请求，Running1持续了一段时间；这部分属于主批次之前的预先请求/预热，混入worker累计统计。

已贴日志止于 **08:32:42、step800**，此时累计输出23,492 token，远不是本批122,880 token的完整收尾。所以：

- `mean_rows=9.8025`不是本轮全程平均batch。
- `worker_observed_s=266.788`不是815.182秒的客户端Benchmark Duration。
- `peak_rows=24`是该片段确实观察到的峰值，可以确认至少有24请求进入执行；它仍不能替代平均batch。
- AISbench的`Concurrency=22.4038`是客户端在途请求均值；按Little's law也可以由120/815.182494×152.1931重建。它包含等待请求，不能当作服务器有效batch。

局部仍可获得有用证据。例如step400→800的区间内，400步中有340步mixed、60步decode，约85%步骤含prefill，且mean_rows的累计差分约18.23。这只是早段区间，不是全程或时间比例。

FULL_DECODE_ONLY不会把mixed步骤整体捕获成纯decode图；同时AC的GDN投机核也会在mixed步骤内被调用。不能由“很多mixed”推出AC优化完全没有作用，也不能按token数量推算各阶段耗时。

## 4. 确定缺陷：graph计数漏掉原生另一条full-graph路径

这段日志里`graph_observed_steps=0`、`graph_hist=[]`，而且没有`target_fullgraph_host_submit`样本。

插件原先只包了`ModelAclGraphManager._updatable_graph_replay`。[Ascend v0.30.0rc1原生代码](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/vllm_ascend/worker/v2/aclgraph_utils.py#L174)明确有两条路线：

```text
run_fullgraph
  ├─ _updatable_graph_replay
  └─ _graph_relay
```

所以**现有计数为零不能证明正在跑eager**。也不能凭空认定本轮一定走了另一条graph路线：在缺少启动配置和完整运行证据时，它仍有多种解释。

本次修复将graph计数和主机提交范围移动到共同入口`run_fullgraph`，覆盖两条路线，避免updatable路径双计数。原来的图选择、参数更新、首次预热和失败保护不变；没有强制使用某种图模式，没有新增stream、事件或设备同步。

这是一项观测修复，**不能声称已经修复67.5秒的性能差额**。新增测试覆盖两条路径各只计一次、原生图失败不计为成功；原有生命周期测试保留。

## 5. 当前如何判断、下一步做什么

1. 承认相对旧AC的性能回退，本轮不能按目标完成验收。
2. 保留被证实的观测修复；不靠关闭参数、改启动命令或二分实验来掩盖问题。
3. 用已有完整服务日志的启动身份及主压测后半段/收尾统计，对齐实际图路径、接受分布、mixed/decode比例及有效batch。当前帖子没有这些完整信息，不能编造全程结论。
4. 原始累计日志可以按相邻报告做差，剔除预先单请求；不要直接使用进程生命周期平均值比较120请求的主批次。
5. 在设备计时缺失时，不把HOST范围替代核耗时。源码显示新的AC等待减少了，但这并不构成设备更快的证据；也没有证据要求立即回退内核或直接升级更复杂的跨轮Replay。

这次已明确的代码改动只补齐观测范围，性能原因仍需完整证据来归属。不会把不确定归因包装成确定修复。

本次观测补丁全量CPU回归 **206 passed**，改动的graph hook与新测试Ruff通过，diff空白检查通过。没有CANN/NPU设备证据，不将它称为性能修复或设备验收通过。
