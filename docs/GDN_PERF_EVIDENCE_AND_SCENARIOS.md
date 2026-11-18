# ReplaySSM 下一轮要测什么

当前最值得确认的是，**同一批请求里，时间花在长输入处理、解码恢复，还是两者之间的调度和等待**。同为 BS20 图模式，Base 为 **739.37 s**，旧 main Replay 为 **768.51 s**，本轮 mixed Replay 为 **759.52 s**。混批路径已接入，本轮首 token 更快，但平均每个输出 token 仍比 Base 多 **4.4 ms**，整批多 **20.15 s**。这是单轮历史对照，20.15 s 是待解释的差距。 [结果审查](OPDEV_V2_MIXED_RESULTS_20261118.md)

**容量收益已经有实测。** 早期 eager 的实际并发从 Base 20 到 Replay 25，增加 25%，两边整批都约 14 分钟。另一组图模式在客户端并发 24 时，Replay C 跑满 24 且未抢占，Base 到 20 且有抢占。客户端并发 32 的图模式又观察到 Base 21、Replay 25，同时 TPOT 从 100.2 增到 123.5 ms。后者表示每个输出 token 的间隔变长了 23.3%。BS20 对照用来衡量同并发成本，后续要解释的是已验证的额外容量如何更好地换成整批吞吐。 [#3 eager 记录](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/3#issuecomment-6811604488) · [#4 对照表](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6940508676)

## 一次完整压测里需要留下的证据

继续使用当前启动命令和完整请求集。已有 Base/Replay 日志也可以先提取，同步保留启动配置、模型版本、客户端并发和请求集。两次运行缺失的字段记为未知。只要能把下面几项对上，就能决定下一刀改哪里。

| 要看什么 | 用来判断什么 |
|---|---|
| 客户端整批时间、120 条是否全成功、处理的输入和输出量、TTFT 和 TPOT 的均值与尾部 | 先确认两轮工作量接近，再看差距集中在首 token、持续生成还是尾部收尾。整体耗时以客户端 `Benchmark Duration` 为准。 |
| 服务端每个阶段的 CPU 调度目标 token、请求行、上下文长度、spec 提案和接受量、graph 桶与 padding | 看 mixed、纯 decode、纯 prefill 分别处理了多少工作。若平均接受长度下降，每生成同量 token 需要更多轮，有可能抵消单轮提速。 |
| 各阶段 Host 提交和等待的样本，mixed 实际命中的 prefill backend，verify/fold 路径与次数 | Host 调用很重就先清重复检查或提交；mixed 命中率正常时，转看原生 prefill 的状态 gather/scatter、拷贝和算子。Host 样本只能定位 CPU 时间和等待线索。 |
| KV 页数和使用率、等待队列、抢占、state dtype、records/图的显存 | 解释已测得的容量增益来自哪些页和状态变化，判断当前负载有多少持续活跃请求、是否用足已释放的容量。 |
| 仍需拆算子的那一次设备时间线，限定同一完整负载中的少量代表步 | 若 Host 证据解释不了差距，才从设备侧核对原生 prefill/state 搬运、verify、采样、fold 及其空隙。看各段实际阻塞后续工作的时间和跨 stream 重叠。普通服务日志无法给出这些 kernel 耗时。 |

这轮已经知道，**69.4% 的 spec 窗口在 mixed，mixed 里 spec 却只占 token 的 3.70%**。大量 prefill 工作仍在执行。两者都可以从计数读出，阶段耗时和关键路径要分别测。各 TP rank 往往观察的是同一批请求，按 rank 比较异常和不均衡，不能把输出量相加。单轮尾日志和客户端报告也各有自己的时间窗口。 [本轮服务端结果](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6974380561)

## 怎么收集

拉取 `op-dev-v2`，在服务环境执行 `python -m pip install -e ./plugin --no-deps`，然后继续用原来的启动命令和完整负载。保留从启动到最后一个请求结束的服务端日志和 AIS 客户端日志，每轮压测分别保存。已有的 Base 日志可直接用，脚本会保留其原生指标并标明缺少的插件数据。

```bash
python3 tools/extract_gdn_perf.py \
  --server server.log --client ais.log \
  --json-out replay-evidence.json
```

把 `server.log` 和 `ais.log` 换成你的实际文件名。AIS 是发请求的压测客户端，日志提供整批耗时、TTFT、TPOT、吞吐和成功请求数；只定位服务端路径时可省略 `--client`。给我 **`replay-evidence.json`** 就可以。`replay=`、`base=` 都是可选标签，只在一起提取多份日志时方便区分。各文件和 worker 独立输出。支持普通文本、`.gz`、Docker JSON 日志。脚本只读本地文件，可选写 JSON。

| 日志 | 读哪一项 |
|---|---|
| `GDN_READY` | 每个 TP rank 启动准备、强制自检和原生 warmup 通过后各一条，包含生效参数、版本指纹、READY 时间、实际 state/records 布局和采样范围。 |
| `GDN_PHASE_PERF` | 看 `phases` 的阶段 token 和接受量、`totals` 的批次和恢复提交，以及 `paths` 从 READY 之后的模型分派与回退。若 `prefill_paths` 命中 Triton 且 state 行间有填充，优先核对原生 gather 是否使整块 cache 连续化。 [rc1 prefill 分支源码](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/vllm_ascend/ops/gdn.py#L655) |
| 同一行的 `host_timings`、`graph_submissions` | 观察 metadata、预检、混批 verify 提交、prefill state 读/算/写提交、采样过滤、图提交、恢复提交，以及原生获取输出的等待与处理。多数计时取前三步及每 64 步约一个样本，commit 延续原来每步都记录的 Host 计时。看 `sampled_steps` 和 `samples` 确认覆盖。同一阶段若命中多种 prefill 分支，计时是这些分支的混合样本。 |
| `GDN_MEMORY` 和原生引擎日志 | 对照 KV block、页布局和内存预算。分 rank 采集 KV 已绑定、插件准备完成、原生 warmup 完成三个时点的 PyTorch allocator 数据。服务期容量另看引擎的 KV 占用、等待和抢占。 |

阶段统计由 rank 0 累计。第一个完成步骤、首次恢复提交对应的完成步骤及每 200 步输出，空闲后补发尚未输出的尾部。worker 窗口可能包含 READY 后的预热请求。token 分项来自 CPU 调度数组，`known_steps` 或 `known_rows` 表示覆盖量，异步调度的上下文也可能偏高。`attention_work_proxy` 按每行新增 token 所能看到的历史长度累计，用来比较上下文工作量。 [rc1 InputBatch 来源](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/vllm_ascend/worker/v2/model_runner.py#L790)

请求期间沿用原有输出等待，不增加设备 Event、同步或 Tensor 回读。Host 分段可能重叠，输出等待也可能很短，因为设备在此前已经完成部分工作。若它们解释不了差距，设备时间线只需回答一件事：真实的 prefill/state 搬运、verify、采样、fold 在完整负载下各阻塞了后续工作多久。可复用已有 trace，仅在确需区分时另作采集。

## 输入变短之后

假设 Qwen3.8-27B 继续用 TP4、BF16 KV、MTP3 和同样的 1,024 输出。主模型 16 层 attention 加 1 层 MTP，每 token 的 KV 逻辑容量约 **17 KiB/卡**。一份 FP32 GDN state 在 48 层共 **36 MiB/请求/卡**，BF16 则是 18 MiB。改变长度主要改变 KV 和 prefill，state 的这项大小相同。 [固定模型配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/366a23fd246985933964e9d21ac63eb3a93190ab/config.json) · [MTP 层源码](https://github.com/vllm-project/vllm/blob/v0.30.0rc1/vllm/model_executor/models/qwen3_5_mtp.py#L119)

| 输入长度 | 输入相当于多少个 2,048-token 预算 | 完成 1,024 输出后的 KV/请求/卡 | FP32 state 的理想容量比 Replay/Base |
|---:|---:|---:|---:|
| 23,564 | 11.51 | 408.2 MiB | 1.22× |
| 8,192 | 4.00 | 153 MiB | 1.48× |
| 2,048 | 1.00 | 51 MiB | 1.88× |
| 512 | 0.25 | 25.5 MiB | 2.11× |

右列只对比每请求逻辑 KV + state，假设 align 下 Base 5 槽、Replay 2 槽且无额外 prefill checkpoint。按这组形状少用三份 state，毛节省 **108 MiB/请求/卡**，BF16 时是 54 MiB。实际能开多少并发还取决于 records、页填充与共享前缀、图、预算和调度上限。比如 BF16 激活、128 行 records 的预分配为 **169.125 MiB/卡**，另有独立的 512 MiB transient 预算预留。每步 token budget 是所有请求共享的，表中的预算比例只描述工作量。 [align 槽位模型](https://github.com/vllm-project/vllm/blob/v0.30.0rc1/vllm/v1/kv_cache_interface.py#L1040) · [记录布局](RUNBOOK.md)

- **短输入、长输出**。2K 输入只有 23.5K 的约 8.7%，无前缀命中时，按 token 计的前向工作减少，长上下文 attention 也缩短。如果接受长度维持约 3.2，1,024 输出仍对应约 **320 个 spec 窗口/请求**。verify/fold 在剩余时间里的权重可能变大，适合判断它们是否真能加速，省下的状态空间也更容易占到请求内存的大头。
- **短输入、短输出**。128 输出按同样接受长度只有约 **40 个窗口/请求**。每请求调度、首 token、连接和尾部等待的影响往往更明显，单个算子的节省更难直接反映为整批收益。若 KV 本来就有余量，容量收益也可能用不上。
- **长输入、共享前缀多**。缓存命中会减少新算的 prefill token，但 decode 仍面对完整上下文。估 prefill 看实际处理 token，估 decode 的 attention 看上下文长度，估容量再结合实际页共享。

在纯 decode 饱和的情况下，可以用 `输出吞吐 ≈ 每步有效请求数 × 每请求平均产出 / 每步耗时` 判断容量能否换成速度。举例，若持续运行的请求从 20 增至 24，接受长度不变，每步耗时增 10%，理论输出吞吐约为 `1.2/1.1 = 1.09×`。如果每步也变慢 20%，吞吐约持平。这里的 20 和 24 是举例，现有 BS20 负载下两端的实际峰值都是 20。整批还有 prefill 和尾部，实际收益需由客户端结果确认。

对当前 759.52 s 要追回的时间做一笔账。假设某段只做局部加速，且它占的是整批上实际不能被其他工作覆盖的时间，则整批降幅约为 `这段占比 × (1 − 1 / 局部速度倍率)`。

| 某段原来占整批时间 | 这段提到 1.1× | 这段提到 1.2× |
|---:|---:|---:|
| 10% | 整批约省 0.91% | 约省 1.67% |
| 25% | 约省 2.27% | 约省 4.17% |
| 50% | 约省 4.55% | 约省 8.33% |

追平历史 Base 要从本轮再省 **2.65%**。所以局部 1.2× 需要覆盖约 16% 的整批关键路径，局部 1.1× 则要约 29%。先量级定位，再决定值不值得重写 fold 或重排 verify 的 GQA norm。

## 如果以后改成 PD 分离

**当前插件遇到 KV transfer 配置会主动拒绝启动。** PD 分离先要验证选定 connector 能正确交接 attention KV、卷积状态和 GDN checkpoint，确认 P/D 两边的 cache group/slot 及 MTP 处理方式一致。D 侧必须在使用收到的 checkpoint 前等传输完成，后续由 D 按采样结果更新状态。首个验证目标是和融合部署的输出对齐。 [现有配置校验](https://github.com/ava131/GDN-RecoverSSM-Ascend/blob/6ca94c5/plugin/gdn_recoverssm_ascend/config_patch.py#L117) · [vLLM PD 机制](https://docs.vllm.ai/en/stable/features/disagg_prefill/)

- **先看哪端在排队。** P 做输入，D 做输出，传输在中间。整条流水线的可持续请求吞吐取决于 P、D、传输三者谁先饱和。若 D 排队且 P 有余量，减少 D 的 verify/fold 时间或增加 D 的有效请求容量才可能提高总吞吐。若 P 饱和，D 更快首先改善当前能进入 D 的请求。两端分别量 TTFT 或阶段时长，加代理端的客户端 TTFT/TPOT 与 P99 ITL 观察尾部。
- **算传输有多大。** 按冷请求交接主干 16 KiB/token/卡的 KV 加一份 FP32 state，23.5K 约 **404 MiB/卡**（TP4 合计 1.58 GiB），2K 约 **68 MiB/卡**（0.27 GiB）。若假设 TP4 聚合有效带宽为 20 GiB/s 且整包串行，搬运本身分别约 **79 ms** 和 **13 ms**。线上还可能有排队、分块重叠、重传、缓存命中或额外 MTP KV。应分别记录 bytes、完成时间和 D 端实际等待，未确认前只按示例估量。
- **按同一总资源比较。** 两端都 TP4 且各独占一套卡，需要 8 张卡。若拿它跟今天 4 卡的融合部署比，只能说明新部署的总效果。要评价 PD 或 Replay 本身，应匹配总卡数和负载，同时看每卡吞吐和延迟目标。短输入时传输字节更小，但固定交接成本仍在；长输入时隔离 prefill 对尾部输出间隔可能更有价值。

当前优先用原场景的完整日志分清 Host、prefill 与恢复的工作分布，并保留两个已确定的优化点：verify 重复读和每层重复静态检查。若随后要验证新场景，先用短输入、固定长输出来放大 decode 的比重。PD 分离另按新部署验证正确性、传输和容量，现有融合压测可继续作为对照。
