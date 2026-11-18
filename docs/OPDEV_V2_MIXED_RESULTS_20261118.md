comment from gpt

# 11 月 18 日 #4 混批结果审查

混批融合已在本次 PD 融合压测中接入。按同为 BS20 graph 的历史对照，整批从 main 的 **768.51 秒降到 759.52 秒，少 8.99 秒（1.17%）**。平均 TTFT 少 1.57 秒，TPOT 基本持平。与 Base 相比，这轮仍多 **20.15 秒（2.73%）**，平均 TPOT 高 **4.4 ms**。这组结果支持保留混批路径继续优化。当前还没有同并发下追平 Base 的结果。 [客户端结果](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6974349077) · [历史对照](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6940508676)

| BS20 graph | 整批 Duration | 平均 TTFT | 平均 TPOT |
|---|---:|---:|---:|
| Base | 739.373 s | 18.5991 s | 99.4 ms |
| main `9c5ce34` 的历史 Replay C | 768.506 s | 19.0159 s | 103.6 ms |
| 本轮 mixed | **759.521 s** | **17.4416 s** | **103.8 ms** |

这轮 LongBenchv2 120 条请求全部成功，每条输出 1024 tokens，总输入 2,827,697 tokens，平均输入 23,564 tokens。实际平均并发 19.53，最高 20，客户端输出吞吐 161.79 tokens/s。当前压测最多发 20 个并发，可以对照这一负载下的速度。容量上限和更高并发下的吞吐仍需其他已有数据判断。历史表只给出约 23k/1k 和单次结果，因此 8.99 秒是观测差值，尚无法剥离请求顺序、批次组成和运行波动的贡献。表中统一使用 AIS 的 `Benchmark Duration`。`Api infer task time elapsed=821.65s` 和 `worker_observed_s=792.391` 各有自己的统计窗口。 [客户端结果](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6974349077) · [服务器日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6974380561)

## 日志与源码对上了什么

- 2,708 步包含 1,604 个 mixed、1,088 个纯 decode 和 16 个纯 prefill。`verify_fused_mixed_model_calls=76992` 正好等于 `1604 × 48` 层。mixed 预检通过、non-spec 门控、融合 verify 和直接写输出的计数也都为 76,992。尾段多次快照中普通 `verify` 一直为 242，而 mixed 计数持续上升。该运行的路径与 `b4e3061` 一致。片段缺少启动源码指纹和完整自检日志，提交身份以启动记录为准。 [服务器日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6974380561) · [混批实现](https://github.com/ava131/GDN-RecoverSSM-Ascend/blob/b4e306174012750fdab0b60102309cbd4b8d2bf3/plugin/gdn_recoverssm_ascend/mixed_forward.py)
- 38,821 个 spec 请求窗口中，**69.39%** 位于 mixed。但累加每层实际输入并除以 48 层后，mixed 的 spec 是 **107,752 tokens**，non-spec 是 **2,803,081 tokens**。所以本轮从原生门控移入融合核的输入只占 mixed 的 **3.70%**。剩下 96.30% 仍走原生门控和既有 non-spec 路径。请求窗口的覆盖率与算子的 token 工作量差得很大，且 token 比例本身也无法换算成耗时比例。 [服务器日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6974380561)
- 每个 spec 窗口平均产出 3.1997 tokens，上一版 worker 日志为 3.2025，差约 0.09%。两段 worker 日志按 `target_tokens − 4 × spec_rows` 核算，non-spec 目标 token 都是 2,828,121。工作构成仍有变化，步数从 2,731 变为 2,708，worker 时间从 800.320 秒变为 792.391 秒。可以排除这两段日志之间明显的接受率崩塌，无法用总时长差分单独测出 mixed 核省了多少时间。 [上一版日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6969969441) · [本轮日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6974380561)

按 BF16 激活、每 TP rank 的 `Hk=4, Hv=12, K=V=128`，本轮删除整块 `merged_out → core_attn_out` 的一读一写。用日志里已经累加 48 层的 token 数计算，每 rank 的逻辑张量读写少了 `2 × (5,172,096 + 134,547,888) × 12 × 128 × 2` 字节，约 **799.48 GiB**。spec QKV 连续副本约 49.32 GiB，两个独立 norm 的输入和输出约 19.73 GiB。这里统计整段运行中反复读写的累计字节。实际 HBM 流量、缓存命中和设备耗时需要对应的设备计量。融合核还要自行完成 norm 和门控，且两路结果仍按索引写入最终输出。 [服务器日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6974380561) · [实现合同](https://github.com/ava131/GDN-RecoverSSM-Ascend/blob/b4e306174012750fdab0b60102309cbd4b8d2bf3/docs/OPDEV_V2_MIXED_BATCH.md)

## 下一步开发判断

1. **先去掉融合 verify 的重复读。** `LoadFusedMeta` 第一次从 GM 读 raw `a`，加 bias 得到 `x`。处理 `softplus` 的线性分支时又读取同一项并重算 `x`。Q4 下每个有效 request/value-head 任务多 4 次 GM 标量读取。可将第一次的 `x` 留在 verify 已预留的独立 32B UB 区，用现有 Scalar/Vector 等待顺序取回，保持 `x > 20`、舍入、records 和总 UB 预算。按 mixed 的有效 token 数和 12 个本地 value heads，整轮约有 6,207 万次这样的重复读可消除。它们对关键路径的影响要看设备执行时间。 [融合核](https://github.com/ava131/GDN-RecoverSSM-Ascend/blob/b4e306174012750fdab0b60102309cbd4b8d2bf3/plugin/gdn_recoverssm_ascend/csrc/fusion.h#L77) · [verify UB 布局](https://github.com/ava131/GDN-RecoverSSM-Ascend/blob/b4e306174012750fdab0b60102309cbd4b8d2bf3/plugin/gdn_recoverssm_ascend/csrc/verify.h#L24)
2. **把可证明稳定的预检前移到绑定阶段。** `can_fuse_verify` 对 76,992 次 mixed 层调用重复检查门控参数、state 和 records 的形状、dtype、stride 等。可随准备好的 context 绑定这些静态检查结果，在对象替换、布局变化或绑定 epoch 改变时重建。每步继续验证动态 `qsl`、索引、token 数、实际输入和输出布局、行容量及 producer 对应关系。当前 `host_submit_us_avg=836.3` 只覆盖采样后 CPU 提交段，乘 2,692 次约 2.25 秒，无法用于估算预检或设备 fold 的耗时。建议在现有服务端日志低频汇总预检 Host 耗时与实际 prefill 分支命中，保持原启动方式和完整负载。 [资格检查](https://github.com/ava131/GDN-RecoverSSM-Ascend/blob/b4e306174012750fdab0b60102309cbd4b8d2bf3/plugin/gdn_recoverssm_ascend/ascendc_backend.py#L250) · [服务器日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6974380561)
3. **更大的算子改动看实际命中路径。** fused verify 按 value head 做 q/k norm，本配置同一 key head 的 norm 会重复 3 次。三个完整 FP32 state 就占 192 KiB UB，若要在核内复用 norm，需要重新划分 state tile。另开预处理核会增加中间写读和提交，需要一起计入。fold 还要等采样给出接受长度，不能和采样前的 verify 一次完成。先核对真实 prefill backend 的 state gather/scatter 和是否发生大张量连续化，判断回收这段耗时是否比重写 fold 更可行。当前三次 commit 提交和约 5.25% 的 decode graph padding 只表明路径结构和形状，无法单独解释 20.15 秒差距。 [verify 任务布局](https://github.com/ava131/GDN-RecoverSSM-Ascend/blob/b4e306174012750fdab0b60102309cbd4b8d2bf3/plugin/gdn_recoverssm_ascend/csrc/verify.h#L358) · [服务器日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6974380561)

按本轮负载，追平历史 Base 需要从 759.52 秒再缩短约 **2.65%**。即使某段可以加速 20%（耗时降为原来的 1/1.2），它也要占当前总时间约 16%，才够补上这 20.15 秒。两项确定的重复工作适合先清理。预期收益应按其实际耗时衡量，优先关注混批还保留的 prefill 和恢复路径。此次提交只记录审查，下一轮仍沿用原启动命令、完整负载和服务器日志。
