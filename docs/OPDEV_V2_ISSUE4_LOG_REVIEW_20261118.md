comment from gpt

# 11月18日新增日志复查：融合已接入，但多数请求窗口落在未覆盖的 mixed 路径

本次通过 Git 远端和 GitHub CLI 读取 [#4 新评论](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6969969441)，并对照本地 `c831247a38794fc4fc7d8434251337c45fc9265c` 及精确 tag `v0.30.0rc1` 的原生 GDN 源码。启动参数保持原样。本次只增加审查文档，没有修改算子或服务代码，没有安排关闭功能或改变参数的实验。

## 1. 结论先说清楚

**本轮不能再解释为“融合没有进入”。更明确的问题是优化覆盖范围不足：当前 norm/gate 融合及 packed-QKV views 只覆盖纯投机 decode；真实负载的大部分请求窗口夹在 prefill/decode 混合批次中，这部分继续走独立 helper。**

全程58.44%的步骤是 mixed；其中累计请求行占69.62%。按Q4纯decode几何估算，约68.31%的 spec 请求窗口发生于 mixed，因此没有得到这轮 norm/gate 融合的收益。窗口集中写回、最后token输出公式、fold decay加载等共同AC改动仍会作用于普通verify/fold，不能说mixed完全没有v2改动。

这是一处优化范围的缺口，不表示mixed计算结果错误。此前我对只优化pure-decode的端到端覆盖预期偏高。下一优先级应转向mixed中的spec子集接入，先复用已经写好的AC核，不应继续假定把pure-decode单核再改快一点就能明显改变整批时间。

日志没有逐phase设备计时，因此**58.44%的步骤不等于58.44%的时间，也不能直接承诺覆盖mixed后会加速多少**。同时，新的客户端Duration/TTFT/TPOT没有贴在该评论中，不能据此给出本轮精确回退或加速百分比。

## 2. 从日志直接得到的量

最后一组统计在server日志05:18:56：

| 项目 | 数值 | 口径 |
|---|---:|---|
| 完成输出步骤 | 2731 | worker CPU完成输出累计 |
| 纯decode步骤 | 1110，40.64% | CPU batch快照 |
| mixed步骤 | 1596，58.44% | CPU batch快照 |
| 纯prefill步骤 | 25，0.92% | 其余prefill夹在mixed中 |
| graph观测步骤 | 1110 | 与纯decode步数一致 |
| decode累计请求行 | 12290 | 每步行数之和 |
| mixed累计请求行 | 28222 | 占全部40539行的69.62% |
| 平均 / 峰值请求行 | 14.844 / 20 | 含首部和尾部 |
| spec请求行 | 38782 | 每请求每投机轮累计 |
| spec平均产出 | 3.202542 token/行 | 含bonus/首token口径，不是单纯draft接受率 |
| commit数 | 2706 | Host观测 |
| 每commit提交 | 3个kernel，3个group | 保持既有跨组协调 |
| graph padding | 5.3232% | padding token / 捕获token |
| worker观察窗口 | 800.320秒 | 不是客户端Duration |

接受直方图为：`1:4980, 2:5388, 3:5211, 4:23203`。加权spec产出为124201，除以38782得到3.202542。没有看到接受长度退化到接近1的迹象；但历史结果的同口径全负载接受行为仍不能由此证明严格相同。

去掉部分开头和结尾，取step400→2600这个主体窗口：2200步中1559步mixed，占70.86%，worker时间差为738.478秒。mixed和decode在该窗口的平均请求行都约18，说明全程40.64%的decode步数被开头单请求阶段和尾部小batch明显影响。仍不能用这个窗口求出每phase真实设备耗时。

## 3. 为什么fused计数不增长是正常的

第一组和最后一组 `GDN_PATHS`：

| 计数 | 第一组 | 最后一组 |
|---|---:|---:|
| ordinary verify | 242 | 76850 |
| verify_fused | 3300 | 3300 |
| verify_fused_model_calls | 3264 | 3264 |
| fold | 56 | 2762 |

这些是Python/Host dispatch计数。warmup/capture运行模型时会增加；已经捕获的full graph replay不会重新执行Python，所以不会随着每次请求增加fused计数。

更有力的是三条严格对应关系：

```text
76850 - 242 = 76608 = 1596 mixed steps × 48 GDN layers
graph_observed_steps = 1110 = decode steps
2762 - 56 = 2706 = commits
```

前两条在全部17个汇总点都成立，不是最终恰好对上。这与“pure decode使用图内已捕获融合入口、mixed在图外逐层调用普通verify”的源码高度一致；但该片段没有逐bucket设备trace，不能升级为逐一验证了全部图的入口。`3264=68×48` 表示启动阶段确有多次完整模型Host调用，不应把68再解释成确定的图bucket个数。

`verify_fused - verify_fused_model_calls = 36` 与修正版新增两种bias的36次专项probe相符，支持运行了修正版的判断。但该片段没有 `GDN_CONFIG`、库指纹、`GDN_ASCENDC_FUSION` 首条和 `GDN_READY`，不能仅靠计数锁定完整提交号、参数dtype和输入stride。

## 4. 源码为什么漏掉mixed中的融合

插件 [spec_forward.py:66](https://github.com/ava131/GDN-RecoverSSM-Ascend/blob/c831247a38794fc4fc7d8434251337c45fc9265c/plugin/gdn_recoverssm_ascend/spec_forward.py#L66) 明确要求 `num_prefills==0` 且 `num_decodes==0`，否则在副作用之前返回False；[layer_patch.py:211](https://github.com/ava131/GDN-RecoverSSM-Ascend/blob/c831247a38794fc4fc7d8434251337c45fc9265c/plugin/gdn_recoverssm_ascend/layer_patch.py#L211) 随即调用原生core。

精确rc1原生路径接着会：

1. 用 `spec_token_indx/non_spec_token_indx` gather出spec与non-spec QKV，分别执行卷积。
2. 调 `rearrange_mixed_qkv`，物化Q/K/V布局。
3. 对全批次执行gating，然后再gather出spec/non-spec的g/beta。
4. 对spec的q/k执行两个独立l2norm。
5. 通过插件的recurrent重定向进入pre-normalized ordinary verify。
6. 分别产出spec/non-spec结果，先merge到临时buffer，再复制到core输出。

来源：[rc1 QKV拆分](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/vllm_ascend/ops/gdn.py#L427)、[布局与gate](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/vllm_ascend/ops/gdn.py#L532)、[spec norm及recurrent](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/vllm_ascend/ops/gdn.py#L562)、[输出merge](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/vllm_ascend/ops/gdn.py#L723)。

当前 `raw_fused` 只由pure-spec适配器传入。等到普通recurrent重定向拿到q/k/g/beta时，norm/gate已经执行，不能在这个位置再调用raw融合核来省掉先前工作，更不能重复归一化。

若采用纯decode请求行均为Q4 spec行的几何，则mixed的spec窗口约为：

```text
38782 - 12290 = 26492
26492 / 38782 = 68.31%
```

这是窗口数量覆盖估算，不是耗时比例。无条件事实是69.62%的全部请求行在mixed；后续日志可直接按phase累计spec行，省去这个推算。

## 5. 另外三种误判应排除

### 不能用800.320秒直接宣称新版本回退

新评论有121条POST 200；开头先出现单请求，前338个decode graph步骤均为rows=1，随后才出现批量连接。这与“额外首请求加正式压测”相容，但评论未标记客户端边界。

因此800.320秒不能直接和历史 [BS20 main的768.506秒](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6940508676) 或BS24的747.649秒比较，不能说精确回退4.14%或7%。worker累计还可能包含客户端最终裁剪前的sampled tokens，和客户端结果不是同一统计口径。

### 峰值batch20不是容量上限20

82条Engine快照中，最高Running为20，最高KV usage为84.7%，并有大量Running20/Waiting0的时段。当前日志无法证明请求供给曾持续超过20，所以peak20不能当作OOM容量或scheduler硬上限。

v2与main没有改变resident state/records的容量规则，本来也不应期待main→v2再获得一次显存压缩。此前Base→Recover的容量收益与本轮算子优化需要分开判断。

### 830.9微秒不是设备恢复耗时

`2706 × 830.9µs ≈ 2.248秒` 仅是已测Host提交包络的累计。它没有给出设备fold/conv/plan在关键路径上的执行时间，也不涵盖所有Host准备工作。不能据此说恢复全程只花2秒，或删除这段Host代码就能追回几十秒。

同理，5.3232%的图token padding是工作量指标，不是5.3232%的整批耗时；没有证据显示图bucket爆炸或恢复提交由3个膨胀成大量kernel。

## 6. 下一改动方向：mixed中的spec子集

优先做插件框架适配，复用现有 `verify_fused` 与packed-QKV views。完整覆盖同一启动方式，不能靠关闭chunked prefill、async、prefix或更改batch/capture参数改变工作负载。

建议明确实现一个mixed适配器，避免全局替换norm/gate为no-op：

- 在spec卷积开始之前检查整个mixed签名。spec/non-spec token索引、各自metadata和缓冲容量都要明确；执行开始后不允许fallback重试。
- 按原索引拆出原始a/b与QKV。spec部分卷积后保留packed views，直接调用现有raw-fused verify；A_log FP32和bias BF16/FP32合同沿用。
- **预检使用spec实际token数**。当前pure适配使用 `meta.num_actual_tokens`，在mixed中它代表整个batch，直接照搬会超出spec窗口或误拒绝。spec的qsl、checkpoint索引、request-row records及producer epoch保持原映射。
- non-spec的普通decode/prefill保留原生gate、norm、卷积、chunk kernel和状态写回语义；只对所需non-spec tokens计算这些中间结果。
- 在最终core输出上按原索引合并结果，研究删除整块 `merged_out→core_attn_out` 的多余copy；不在第一步引入新的AC散写ABI或改变物理页规则。
- 保持connector wait/fence、commit、align boundary、异步采样及请求复用的生命周期。补齐乱序/交错索引、空子集、Q1/4、padding/null、FP32/BF16、bias两种精度、接受0..Q及所有原输出位置恰好写一次的检查。

**mixed里不能简单声称“每层再少一个gate kernel”**：prefill仍需要gating，通常仍会保留一次non-spec gate调用。也不能先对全批次做gate，再让spec raw核重复做同一份gate。可明确争取的是spec的两个独立norm、spec QKV布局副本、spec中间结果，以及多余输出copy；最终净收益还要扣除融合中GQA重复norm和gate读取等成本。

其中整块输出copy覆盖的是整个mixed batch。以1792个actual tokens、HV12、V128、BF16为例，单层输出5.25MiB，额外完整copy的逻辑读写是10.5MiB，48层约504MiB/步/卡。实际mixed token数不同，不能把这个示例套到每一步；逻辑字节也不是设备时间。但它说明输出合并值得和spec融合一起审查，而不是只盯几十个spec token的局部算术。

后续正常server日志应直接按phase记录spec窗口数及入口选择，使“实际覆盖多少请求工作”可核对；计数继续使用CPU已知信息，不增加设备scalar回读或计时同步。交付仍使用同一完整负载，先看整批Duration，再看TTFT/TPOT和接受行为。

## 7. 本次工作范围

新增日志已完整读取并用脚本解析17个汇总点，交叉核对phase、graph、verify、fold及接受直方图。没有执行新的NPU实验，也没有修改现有服务/算子代码；因此没有重复运行与本次文档变更无关的测试。

**当前能确定的开发优先级是：先补mixed中spec子集的优化覆盖，再评估单核内部进一步微调。现有证据不支持“已经优化到头”，也不支持“只要把计数涨起来就能快”。**
