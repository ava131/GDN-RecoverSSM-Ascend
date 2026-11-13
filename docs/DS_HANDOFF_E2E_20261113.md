comment from gpt

# DS 开发交接：GDN RecoverSSM 端到端优化补丁

日期：2026-11-13。后续开发、NPU验证和性能调优交由DS接手。

**代码实现提交：[`0220ca4c7c41db002862c0cab2793d453f837e8d`](https://github.com/ava131/GDN-RecoverSSM-Ascend/commit/0220ca4c7c41db002862c0cab2793d453f837e8d)。稳定快照：`release/2026-11-12-stable-before-e2e`，指向`a0a7f3911ac9352024fc09a25e9043bbc046ee77`。后续工作继续在main。** 本次交接提交另外移除手敲练习文字，`padding_fraction`已有完整实现，不存在需要用户补写的函数体。

**当前状态：实现已提交，114项CPU测试通过；没有Ascend NPU实测，也没有新的端到端提速数字。** 代码中的强制预编译、自检及READY保护，是部署时执行的验证机制，不是本机已经取得的设备证据。请先按原命令完成一次全功能验证，再依据结果继续开发。

## 1. 必须遵守的任务边界

- 目标是缩短整批请求从开始到最后完成的端到端时间。容量、内核微基准和Host提交开销均为解释指标。
- 用户的原vLLM启动命令、全部环境及参数保持不变，包括TP/MTP、async、prefix caching、align、max batch tokens、max model len、max num seqs、图模式和capture sizes。
- 继续以本仓外部plugin/patch形式开发，不编辑原生vLLM、vllm-ascend或FLA文件。
- 不新增用户开关，不设计关闭verify、commit、async、prefix等功能的二分实验，不要求用户扫描batch/TP/模型矩阵。
- 需要确认的接口、实际路径和自检结果应在server日志中明确记录。精确设备时间需要开发侧设备证据，不能将Host计时冒充GPU/NPU内核耗时。

插件版本为0.2.0。部署参考接口是vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、Ascend rc1 `5f828ad638ee4f9b12db87f9d5481f29076d37f9`和Triton-Ascend3.2.2；这不是强制替换用户wheel的指令。现场实际包版本由`GDN_RUNTIME`打印，source_pair仅表示适配审查依据，不能当作已安装wheel的构建SHA。

## 2. 原来的问题与本轮的改动对应

此前#3同负载记录为Base graph约742.382秒、Recover graph约859.644秒，容量约21→25；Recover eager约857.213秒。增加容量尚未带来整场收益。没有证据能从这些跨运行的max bs和TPOT拟合出一个精确的“同并发固有14ms开销”。

| 原问题 | 本轮实现 | 为什么这样改 |
|---|---|---|
| 每个value head重复保存同一个GQA key | typed-SoA records，key按key head保存一次 | 减少scratch及record写，保持原计算精度 |
| verify逐token即时取输入，新增record路径不同于优化过的原生AscendC | 短窗口输入及gate预载，保留原forward recurrence | 尝试减少数据依赖等待，同时不改变数值次序 |
| 纯spec先生成临时core output再逐层复制 | 插件纯spec分支直接写最终core_attn_out | 删除这一路临时分配和输出copy节点 |
| 三组分别plan→conv→fold，伴随重复align公式和bool compact | step-owned跨组协调器，plan唯一决定有效性与列 | 当前三组同构恢复9次提交变3次，消除重复簿记链 |
| 重新nonzero/cast/阻塞H2D上传spec行映射 | 复用原生spec_sequence_indices及CPU行证据 | 去掉插件重复映射构造，保留真实重排关系 |
| context到采样后才刷新可能丢掉本轮record | forward前检查binding，metadata保存producer snapshot | verify与commit使用同一代checkpoint/record/pointer表 |
| prepare只覆盖none或猜测页表尺寸 | 使用真实dtype/stride/页表，补verify和grouped签名 | 将首次注册/编译尽可能移到请求前 |
| record预算、采样峰值与图内存混算 | 按SoA真实bytes预留，保留512MiB旧margin，采样有界分块 | 降低峰值风险并明确每项账目，不靠改启动参数 |
| 图metadata在服务期滞留，异常后盲目重放 | 仅capture期间收集，用后清空；失败图不重放 | 避免引用滞留及可能重复的部分state写 |
| 原日志不足以证明路径与接受行为 | 增加版本/源码/config指纹、准备结果和完成输出统计 | 为后续归因提供可读证据，不增加服务期device scalar读取 |

## 3. 文件地图及调用关系

| 文件 | DS应关注的入口/职责 |
|---|---|
| `plugin/gdn_recoverssm_ascend/__init__.py` | plugin入口和patch安装顺序；diagnostics最后包装原生warmup |
| `config_patch.py`、`state_patch.py` | 既有开关与GDN适配状态，本轮未重写其数学/原生page合同 |
| `builder.py` | `_native_spec_rows`、`_get_context`、`build`、HostMetadataBuilder选择、producer snapshot |
| `kernels.py` | `ReplayRecordBuffers`、verify、单context恢复、`_LaunchSpec`、`prepare_verify`及prepare |
| `layer_patch.py` | 真实GDN类/方法路由、纯spec入口、`_verify`、`_RecoverCommitProxy` |
| `spec_forward.py` | `try_spec_forward`，使用固定rc1 helper ABI直写最终输出 |
| `grouped_commit.py` | grouped plan/conv/fold、bucket、跨桶publisher、预编译和设备启动自检 |
| `sampling_patch.py` | 原top-k/top-p阈值语义、有界行分块和值排序探针 |
| `startup_patch.py` | GDN范围判定、预算、真实签名准备、强制自检、prepared token |
| `graph_probe.py` | capture窗口metadata、正常首次图执行、失败传播 |
| `diagnostics.py`、`metrics_math.py` | CPU完成输出统计、路径/配置证据与padding比例 |
| `selftest.py`、`plugin/tests/` | 部署启动自检与本地CPU合同测试 |

主流程为：原生构建batch → Recover builder在forward前绑定context并构建metadata → target执行verify并写request-row records → 原生sampling产生num_sampled → 原生MRV2 postprocess调用proxy → coordinator统一plan/conv/fold与align发布 → 原生postcopy/draft继续。FULL target graph和采样后提交仍是不同执行阶段。

## 4. typed-SoA records：精度、布局和显存

服务期`ReplayRecordBuffers`是一个对象，不再是单张packed Tensor：

```text
correction : [R, HV, Q, V]  FP32
key        : [R, H,  Q, K]  normalized key原输入dtype
decay      : [R, HV, Q]     FP32
```

R为请求scratch容量，独立于KV物理block数和图padding容量。HV为每卡value heads，H为每卡key heads，Q=1+num_speculative_tokens。Qwen本次TP4为HV12/H4，因此一个key被三个value heads共享。

唯一key writer由对应key head的指定value head和V tile负责；commit按`head // (HV/H)`取key。correction/decay保持FP32。输入已经归一化且存储为BF16/FP16时，原实现load后转FP32；将原始key bits存入相同dtype再读回FP32可以精确往返。若在verify内部做FP32 normalization，key也必须FP32；代码会拒绝不成立的低精度合同。

**不得将这项描述为“把所有FP32 records降成BF16”。** 同样，不得把临时normalized key输入指针保留到commit：前向结束后输入可能被回收，必须保留context拥有的专属record缓冲。

按R128、48层、HV12/H4、Q4、K=V128：

| 每卡 | 原packed FP32 | GQA去重、FP32 key | GQA去重、原BF16 key |
|---|---:|---:|---:|
| correction | 144MiB | 144MiB | 144MiB |
| key | 144MiB | 48MiB | 24MiB |
| decay | 1.125MiB | 1.125MiB | 1.125MiB |
| 合计 | 289.125MiB | 193.125MiB | 169.125MiB |

本轮额外减少的是scratch，BF16 key情景少120MiB/卡；之前已完成的resident state减少仍是既有功能，不能重复认领为本轮新增收益。减少key写和更紧凑布局可能帮助时间，但每个value head/V tile仍需读取key，不能声称读取次数自动少三倍。

兼容注意：对象`.shape/.dtype`保留legacy逻辑诊断形式，**不能用shape×element_size推算实际SoA显存**。应使用`record_nbytes`或`.record_bytes`。`from_tensors`仍支持packed FP32兼容路径，但同一context不允许混用两种layout。

## 5. verify与direct output究竟实现到哪一步

`_load_verify_token`与verify中的静态tuple提前加载短窗口q/k/v/exp(g)/beta，然后按原次序推进state，checkpoint在verify期间只读。窗口最大值仍按现有8个token约束，默认verify BV32、commit BV16及vector-core cap保留；没有把回退过的BV64直接恢复。

预载粒度仍是每个逻辑(row、value head、V tile)工作。跨head/V tile重复加载、持久循环和部分metadata开销仍存在。**没有实现新AscendC算子、跨next-V-tile显式双缓冲/事件流水，也没有融合原来的gate/norm。** 因此不能把当前staging说成已经达到原生AscendC流水质量。

`try_spec_forward`只在原生rc1 ABI及纯spec条件满足时接管，保持原生卷积、gate、L2 norm和connector顺序，将verify直接写入最终`core_attn_out`。混合prefill、普通decode、profile或不匹配的接口在任何state副作用前保留原生分流；其中spec调用仍由RecoverSSM拦截和提交。卷积开始后不捕获异常重走原路径，避免重复推进。

当前未删除混排分支的全部merge/index_copy，也未声称每个服务期batch都使用direct output。`available_strategies`仅是能力说明；不能仅看这一个字段就断言实际所有步都命中纯spec。

## 6. 跨组提交和align发布合同

`_RecoverCommitProxy`消费原生`RecoverSSMState._step`一次，将GDN metadata交给step coordinator，其他metadata继续走原逻辑。删除了全局`_PENDING_ALIGN`、独立重算列及动态bool compact的应用链。

每个bucket持有每层conv/checkpoint/record的地址和stride，以及layer→group映射；每组保留自己的物理state index、block table、qsl、request map、block数与block size。逻辑行相同不代表物理page id相同。

同构同逻辑行的当前三组：

1. 一次grouped plan，产生source/commit_len/final/boundary/valid/col。
2. 一次跨层conv compaction。
3. 一次跨层SSM forward fold，同时生成必要的boundary和final state。

这里的9→3只指专用recovery提交，不是说整个step只有三个算子。对于N个物理bucket，none模式为3N；align且N>1时为3N+1，最后增加一次统一publisher，对全部bucket的valid/col做一致性检查后发布。单bucket的发布在plan内完成；同流顺序和不可变plan快照保证后续postcopy不会抢先消费未完成fold。

align跨组必须在CPU已知的request rows、query lengths和block size上达成一致；不兼容时fail fast，不能默默“以最后一组为准”。各组无效boundary必须在plan中规范为null，不能因为fold使用bucket最大pool bound而让小pool的越界地址变合法。

### Triton-Ascend 3.2.2为何必须flat ABI

最终检查确认该版本JIT不支持运行时Python tuple参数。plan改为显式三槽Tensor/标量参数，未用槽复制首槽，`NUM_GROUPS`控制实际处理数量；local tuple和constexpr tuple仍可用于静态索引。更多组按相同规则在prepare和serving中切块。

跨bucket publisher读取启动期/epoch期创建并缓存的GPU描述表，表中是valid/col地址、capacity和group_count，并持有相应plan owner。**不要重新引入runtime Tensor/整数tuple，也不要为拼接groups在每步添加H2D指针表上传。** 新CPU回归覆盖7组切块、统一发布、描述表复用及flat ABI。

## 7. context、绑定代次和图地址

`builder.build()`在verify前执行`_get_context()`；metadata保存`recoverssm_context`生产者快照。commit只消费这份快照，不允许采样后再调用_get_context换成新空record。`_verify`还核对checkpoint和record身份，防止producer与consumer跨代。

当前仍保守检查每层Tensor身份、data_ptr、shape、stride、dtype及device，**并未实现真正的原生binding epoch廉价替代**。本轮消除的是错误刷新时点和commit重复扫描，不应宣传为所有层扫描已删除。

FULL图READY之后检测到binding改变会拒绝继续，要求原生正确invalidate/recapture。**没有实现任意sleep/wake后的自动无缝重捕获。** DS不能只删guard；真正支持重绑定时，records、地址表、预编译bucket、target捕获的record/checkpoint地址及publisher描述表必须一起换代。源码中持有引用是为了让异步使用中的owner活着，不是仅靠容器id猜测合法性。

## 8. prefill能力和图生命周期

恢复了原生backend factory对HostMetadataBuilder的能力选择。支持host-only fused-prefill元数据时，Recover builder也设置`build_device_chunk_metadata=False`；fallback所需device metadata保留。仅调用普通builder的super().build不会恢复已经绕过的factory决策，因此必须在get_builder_cls处处理。

`graph_probe`只在capture窗口收集metadata，使用正常`run_fullgraph`路径完成首次执行，以保留updatable task和external-event依赖；预热后清空插件额外引用。服务期eager不再收集。

原先对任意图异常drain后重试的行为被移除。异常可能发生在部分state写之后，盲目重放不具备幂等性。第一次图执行若在启动预热失败，也会阻止READY；后续应分析初始化/资源原因，不能为了跑完恢复任意异常重试。

本轮没有增加独立commit graph，也没有调整capture目录。图padding只观测。用户固定参数约束同样约束后续DS开发，不能借“内部优化”偷偷改capture sizes。

## 9. 预编译、预算和READY顺序

启动准备限定为支持的GDN架构且feature已启用，原生KDA不应被本插件新增预算或GDN自检干扰。缺少model runner、builder、KV绑定或真实页表时应拒绝GDN READY。

准备包括：

- `prepare_verify`使用实际context的NUM_BLOCKS/R/H/HV、activation dtype、两种value token stride（连续与combined QKV）、FP32/activation beta dtype和normalized输入合同。不是任意dtype/stride全面覆盖。
- 单context commit准备实际none/align、native request-index dtype及真实record行域。
- `prepare_grouped_commit`使用原生input_block_tables对应group的真实dtype、width、stride，以及native索引与computed布局；不能用ceil(max_model_len/block_size)猜签名。
- `_LaunchSpec`同时用于服务发射和compile-only，统一参数构造并检查句柄注册。
- 部署启动时自动执行数值检查，只有全部完成才设置`worker._gdn_patch_prepared=True`。
- 原生warmup和图首次执行成功之后，最外层diagnostics再次检查prepared token，才输出最终`GDN_READY ... STARTUP_COMPLETE`。

`GDN_PREPARED_CONTEXT`只表示几何/预分配，不是READY。UB日志为保守手算估计，不是BiSheng实际live-set或spill报告；实际编译仍须设备完成。

SoA预算在决定KV pool前扣除，创建后的实际record bytes与预留比较；超出则拒绝启动。`GDN_RECORD_BUDGET`打印actual/reserved和transient reserve。原有512MiB默认margin与其既有覆盖方式保留，没有改为0或再升到2GiB。它不是已测得的graph pool或完整sampler峰值。

## 10. sampling filter：做了什么，精度与峰值边界

`sampling_patch.prepare`同时替换base helper和MRV2按值导入的alias；只替换一个模块属性会漏掉真实调用。适配器在GDN feature关闭时保持原调用。

过滤保留rc1原来的softmax→排序→top-k/top-p阈值及tie/mixed-k/p语义。它在拒绝采样前按独立行分块，不切换随机采样算法，不改变行顺序或RNG counter。不能为了优化而改为greedy，也不能将不同语义的新fused sampler当作等价替换。

内部`FILTER_WORKING_BYTES=128MiB`和每元素32B的模型用于限制显式活跃张量的行tile；它们是代码内部常量，没有新增CLI/env开关。CANN私有workspace不由这两个数严格封顶，仍需保留margin和设备验证。

若现场有`torch_npu.npu_sort_v2`且FP32启动探针通过，使用只返回排序值的接口；否则保留有界原排序。非FP32输入走已覆盖的原排序分块路径。仅对明确的接口/未实现类异常做适配；未知RuntimeError不在服务期吞掉重试。

好处是避免无用int64索引输出及整批大临时矩阵，代价是可能增加过滤kernel提交数。**没有测出这一项一定更快，也不能说此前480MiB分配一定就是排序索引。** 这是底座通用工程优化，报告收益时应说明，不应全部归因于RecoverSSM数学。

## 11. server日志：该怎么看，哪些不能证明

无需新增参数或自检环境变量。安装更新后的plugin，执行保存的原始启动命令，然后查看：

```bash
grep -E 'GDN_CONFIG|GDN_RUNTIME|GDN_SAMPLER|GDN_VERIFY_PREPARED|GDN_PREPARED_CONTEXT|GDN_RECORD_BUDGET|GDN_GROUPED_SELFTEST|GDN_READY|selftest.*SUMMARY|full-graph warmup' server.log
grep -E 'GDN_PERF|GDN recovery coordinator|RecoverSSM graph dispatch' server.log
grep -E 'Traceback|OutOfMemory|MISMATCH|MTE accesses|SUSPECT REMOTE|507899|561002|replay failed|SELFTEST FAIL' server.log
```

| 日志 | 证明/用途 | 不能据此宣称 |
|---|---|---|
| GDN_CONFIG source_fingerprint / fingerprint | 实际插件源码和解析参数分别有指纹 | 仅PATCH_ID一致就说明所有代码完全一致 |
| GDN_RUNTIME | 安装distribution版本、参考接口 | 现场wheel的精确源码SHA已核实 |
| GDN_VERIFY_PREPARED / prepared variants | 真实几何的编译/注册已完成到日志位置 | 所有未来模型/布局自动受支持 |
| GDN_PREPARED_CONTEXT / GDN_RECORD_BUDGET | 实际record布局、bytes、预留和margin | 这就是全部显存或图池大小 |
| selftest SUMMARY / GDN_GROUPED_SELFTEST PASS | 本机部署启动时实际执行了相应数值检查 | 已验证真实48层模型/完整请求/接受率 |
| 最终GDN_READY ... STARTUP_COMPLETE | 所有启动保护与原生warmup成功 | 整场端到端已经快于Base |
| coordinator groups=3 buckets=1 launches=3 | 当前恢复确实合并为3次 | 整个step只有3个算子 |
| GDN_PERF host_submit_us_avg | Host观察到的提交成本 | 纯device时间或TPOT |
| spec_emitted_per_row | 从已完成CPU输出按spec行次加权的推进量，含target token | draft接受概率已按同一口径测出 |
| graph_hist / graph_padding_fraction | 实际token到既有图bucket的分布；分母为captured tokens | 图bucket已被调整 |
| worker_observed_s | 首次服务工作到最后观察输出的worker区间，空闲汇总不继续加idle时间 | 客户端整批Duration、排队/网络延迟 |

统计复用原生AsyncOutput已经完成的CPU副本，不增加服务期device scalar读取/D2H/synchronize。每个输出对象保存自己的step快照，避免异步下一步覆盖归属；只计一次。rank0定期和空闲尾部汇总，退出时停止CPU日志线程。

**一个容易误读的边界：AsyncOutput在采样后commit之前发起复制。** CPU输出已可读、甚至客户端最后token返回，并不等于后续NPU commit/draft全部完成。GDN_PERF不是device-drain证明。请保留最后请求后的自然server尾日志，出现延迟MTE/worker异常仍判失败。

## 12. 验证清单与证据状态

### 已执行的本地CPU检查

完整命令：`.venv/bin/python -m pytest plugin/tests -q`；本次交接前再次全量复跑，结果为114 passed（5.54秒）。另通过修改Python文件的关键Ruff检查、格式检查及git diff --check。CPU开发环境的Torch与服务器Torch-NPU是不同环境，不要照搬CPU测试环境覆盖服务器依赖。

| 测试文件 | 项数 | 重点 |
|---|---:|---|
| test_math_oracle.py | 11 | 顺序递推、接受前缀及边界公式 |
| test_kernel_index_sim.py | 5 | SoA按请求行索引、GQA key head、record/padding边界 |
| test_kernel_launch_contract.py | 3 | 所有LaunchSpec、缺失constexpr负例、dataclass guard接入 |
| test_layer_patch_mock.py | 24 | 原有接线与新合同回归 |
| test_record_buffers.py | 13 | 物理bytes、dtype精确往返、布局/ABI、全层与错误输出负例 |
| test_grouped_commit.py | 16 | 真实plan/publisher逻辑、不同物理页、flat ABI、7组、descriptor复用、snapshot |
| test_spec_forward_and_graph_lifecycle.py | 13 | direct output别名、写前判定、图生命周期、真实准备与READY保护 |
| test_sampling_patch.py | 16 | 对照rc1阈值过滤、不同dtype、ties、mixed k/p、分块及禁用路径 |
| test_diagnostics.py | 6 | CPU副本快照、去重、不读device对象、参数不变、READY token和空闲时长 |
| test_metrics_math.py | 7 | padding比例、零长度和非法输入 |

测试数不等于硬件覆盖：grouped CPU测试没有编译BiSheng，也没有在NPU运行fold/conv。`selftest.run`覆盖所建合成context的不同q/k、GQA、所有被测层output/state及0..Q；传统run_conv仍主要比较第0层，另有grouped设备自检覆盖2组×2层conv。不能将这些日志写成真实整模型全层ST已经完成。

### 设备上由原命令自动执行、当前尚未取得结果

- Triton-Ascend3.2.2的实际编译、binary/function注册和UB资源是否满足。
- FP32/BF16 verify/commit、真实输出布局与采样适配的NPU数值。
- FULL graph capture、updatable参数和首次执行、混排及最后请求边界。
- 真实模型输出、全场推进/接受行为及120请求无错完成。
- 相比同负载Base的Duration、吞吐、TTFT/TPOT和容量。

## 13. DS后续工作安排

1. 先确认main实现基线与source_fingerprint，沿用原始软件环境和启动命令。检查四个worker的最终READY、grouped PASS、record预算、实际coordinator提交数。
2. 用既有完整请求集运行，保留server从启动到尾部的日志和客户端原始完整结果。不减少输出、不关功能、不改变B/TP来归因。
3. 若失败，按日志所处阶段修patch：先区分API/签名/注册、数值、图初始化、服务期异步错误。不能跳过失败检查或恢复任意图异常重放来掩盖。
4. 若功能完整但仍慢，用同配置的开发侧全路径trace定位占比，覆盖prefill/混排/decode/尾部和全部TP rank。再决定是否继续投入verify调度/流水、公共采样过滤或Host协调；不要把max bs增量或Host时间直接当根因。
5. 新代码继续维护原生行映射、producer snapshot、group-specific物理表、forward边界恢复、精度前提和3.2.2 flat ABI。任何layout变更必须同时更新分配、预算、指针表、预编译、自检和日志。

可继续优化但本轮尚未交付的方向包括：原生绑定generation替代全扫描、真正的AscendC/next-tile流水、受控gate/norm融合、混排直接输出，以及在内存/图生命周期证明充分时的独立commit graph。capture几何修改受用户固定参数约束，本轮与后续均不能擅自调整。

后续开发由DS接手。本交接不要求用户再补写函数，也不把CPU通过或READY保护机制等同于已经缩短端到端时间。

