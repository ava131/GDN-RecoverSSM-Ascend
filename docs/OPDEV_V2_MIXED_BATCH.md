comment from gpt

# op-dev-v2：PD融合部署中的mixed batch接入

## 1. 修正的限制

PD分离/融合是部署方式，pure decode/mixed是单个执行步骤的批次组成，两者不是同一个概念。此前RecoverSSM本身已经支持PD融合运行，但norm/gate融合及packed-QKV views只在纯spec批次启用；一旦同一步含prefill，优化就退回原生helper路径。

11月18日[#4日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6969969441)中1596/2731步为mixed，累计请求行约69.6%在mixed，因此这是当前配置的重要覆盖缺口。本轮在`op-dev-v2`继续开发，main与旧op-dev不变，启动参数和缓存策略不变。

## 2. 当前执行路径

```text
原生 metadata / token indices
    ↓ 卷积之前完成mixed签名检查
原生spec与non-spec卷积、connector顺序
    ├─ spec：packed QKV views + 原始a/b → 已有AC verify_fused
    └─ non-spec：原生gate / prefill / 普通decode及state写回
    ↓
按原token indices直接index-copy到最终core输出
    ↓
原生sampling → 既有RecoverSSM commit / align
```

没有开发新的AscendC数学核。本轮复用已存在的`verify_fused`，没有修改csrc或原生vLLM/vllm-ascend/FLA文件。

mixed中的non-spec仍需要门控，不能宣称每层把整个gate kernel删掉。该helper保留一次，但只计算non-spec输入；spec的a/b保持原始值，融合核中计算，避免重复gating。

每个符合签名的mixed层调用可删除spec的两个独立l2norm、spec QKV连续布局副本，以及`merged_out→core_attn_out`的整块copy。后者覆盖整个mixed batch。举例：1792 tokens、HV12、V128、BF16时，单层完整copy逻辑读写10.5MiB，48层约504MiB/步/卡；不能将字节量直接当成耗时收益。

## 3. 怎样保持原生prefill分支

新增`mixed_forward.py`在安装阶段读取已安装原生`_forward_core`，对函数AST做五处内存替换：卷积前检查、spec QKV视图、gate分区、spec验证、输出合并。其余原生语句保留，包括PCP、FLA/fused-chunk、prefix相关状态读取/写回、connector调用及异常行为。

这比维护另一份数百行原生prefill逻辑更容易核对。只处理已审查的Ascend`v0.30.0rc1`结构：五个位置均须唯一匹配；新增/改变关键语句、装饰器、closure、参数ABI或插入位置不符合时明确拒绝构建。生成函数带原生source digest、生成代码digest和五处替换计数，traceback使用缓存的生成源码。

生成函数共享原生globals，所以插件稍后安装的recurrent重定向仍可见；没有冻结旧kernel引用。donor类和真实vLLM类都安装wrapper，解决原生worker复制方法的导入时序。

启动时再检查`static_forward_context`里每个实际模型层绑定的方法，而不是只看donor类。任何实际层缺少完整mixed adapter都会拒绝READY，避免再次出现“代码已安装、优化没有接入”的静默成功。原生wheel仍需要提供可检查的Python源码；本分支不宣称兼容其他版本任意改写后的core。

## 4. 索引、精度和生命周期合同

- preflight在任何状态卷积之前完成；执行开始后不捕获异常并重试另一条路径。
- spec实际token数来自`spec_token_indx.numel()`，不能使用包含prefill的`meta.num_actual_tokens`作为spec窗口长度。
- token selectors遵循原生连续INT64合同；qsl/checkpoint indices继续INT32。所有检查只读Tensor元信息，不读设备值。
- spec/non-spec原索引、qsl局部偏移、checkpoint列0、request-row records及producer epoch原样保留；调用既有`_verify`继续执行绑定一致性检查。
- spec存在时原生builder把non-spec单token行归入prefill分类。本轮保留该分类，不自行创造`num_decodes>0`与`num_spec_decodes>0`共存的metadata。
- A_log FP32、bias BF16/FP32、activation/state的已有精度合同保留；没有每步参数转换或模型Parameter改写。
- 结果按原索引写入最终buffer，actual tokens之外的padding不写；输出没有整块临时merge后再复制。
- 图捕获策略、sampling后恢复、align边界、原生cache页分配、地址表及异步生命周期保持原合同。

非目标路径（pure prefill、普通decode、profile或不支持的mixed签名）在写前选择原生分支；纯spec路径继续已有适配。源码结构不支持导致实际层无法安装mixed适配，与运行时某个签名不支持是两回事：前者拒绝READY，后者在卷积前保留原生计算并记录原因。

## 5. 已做的开发侧验证

完整CPU套件：**350 passed**。

39项新mixed core测试嵌入并实际执行完整的精确rc1原方法及转换后方法。冻结fixture的AST已独立核对与官方rc1方法一致。测试使用真实CPU Tensor，对比输出、conv/SSM状态和records，覆盖：

- FP32/BF16 activation与两种bias精度。
- 交错token indices、单token non-spec行、partial/空窗口、null、尾部padding。
- 仅non-spec执行原生gate，spec norm/QKV copy和整块输出copy被删除。
- 非法token索引dtype/stride、布局、容量与分区在卷积前拒绝。
- 执行失败不重试，没有item/cpu/tolist/numpy设备回读。
- 共享globals的后续重定向可见；源码ABI变更明确拒绝。

另有真实层READY检查、spec子集长度、mixed helper计数、诊断phase统计和owned-staging自检的检查。FP32容差1e-5、BF16容差5e-3，不调宽现有容差。

Python语法、diff检查通过；对修改前基线比较没有新增Ruff诊断，原有19条告警未在本轮混入修改。wheel离线打包成功，已核对包含mixed适配器、启动检查/自检/分派及全部原有AC源文件。

本机没有NPU/CANN，未声称设备数值或端到端性能通过。PCP/FLA分支源码被保留，未独立做这些分支的真机验证。

## 6. 正常启动时自动验证

保留原有AC、state、conv和grouped自检。36个raw-fused核probe现在实际使用packed QKV token stride，避免只测试独立连续q/k。

额外4个owned-staging probe在自有小缓冲中检查：交错raw a/b分区、原生non-spec gating、直接输出merge和padding；覆盖FP32/BF16 activation及两种bias。这些probe不接触模型KV，也不计入模型调用数。它们不代表整个原生prefill或PCP已被独立设备验证，日志明确写`OWNED_STAGING_NOT_FULL_PREFILL`。

任一自检失败继续阻止READY。请求期间没有新增NPU回读、Host synchronize或计时event。

## 7. 如何部署和看日志

在原服务器仓库和原Python环境中：

```bash
git fetch origin op-dev-v2
git switch op-dev-v2
git pull --ff-only origin op-dev-v2
python -m pip install -e ./plugin --no-deps
```

正常重启，使用原启动命令和原完整负载。没有新开关，不用更改PD部署、batch、graph、async、prefix或capture sizes，不需要手动清编译缓存。本轮没有改AC源码，已有库可以命中缓存；Python代码与图需要随server重启更新。

保留四个worker的以下日志：

| 日志 | 应看什么 |
|---|---|
| `GDN_CONFIG` | `patch=ascendc-op-dev-v2-mixed-fused`和实际source fingerprint |
| `GDN_MIXED_ADAPTER READY` | 实际层数、native source digests、五处patch；是方法绑定证明 |
| `GDN_ASCENDC_SELFTEST PASS` | `packed_qkv=PASS`及原有精度/record检查 |
| `GDN_MIXED_SELFTEST PASS` | 4个owned staging案例与明确scope |
| `GDN_ASCENDC_FUSION model_dispatch=PASS phase=mixed` | 正常混批时实际进入模型入口，`external_helpers_removed=2` |
| `GDN_READY` | 原有准备、原生warmup/capture完成；mixed调用此时可以为0 |

`FULL_DECODE_ONLY`通常只在启动捕获纯decode，所以不能因warmup时mixed模型计数为0就拒绝READY；实际层绑定检查与正常混批的入口日志共同确认接入。

运行中/结束时：

- `GDN_WORKLOAD spec_phase_hist`直接打印每phase的spec请求行和产出，不再用total rows估算覆盖率。
- `GDN_PATHS counts`增加`verify_fused_mixed_model_calls`、`verify_fused_pure_spec_model_calls`。mixed逐层分派应出现增长；图重放不跑Python，pure-spec计数可能保持不变。
- `GDN_PATHS mixed`含preflight、non-spec gate、model verify和merge次数，以及spec门控token节省量。它们都是Host调用结构，不是设备kernel计时；per-layer token累计也不等于唯一请求token数。
- `verify_fused_saved_helpers`对mixed每次只加2，对pure-spec加3；没有把仍需保留的non-spec gate算成已经删除。
- 若混批没有进入，保留`GDN_MIXED eligibility=0`或`GDN_ASCENDC_FUSION eligible=0`的首条原因，无需改开关定位。

最终仍以相同完整负载的客户端Duration、TTFT/TPOT和请求/token完成情况判断。server的worker累计包含首请求/预热等可能范围，不能直接替代客户端Duration。

## 8. 性能边界与后续交接

本轮扩大了既有优化的覆盖范围，并删除一项覆盖整个mixed输出的copy。它没有减少模型权重计算、长上下文attention、prefill主体或必要恢复，也没有再压缩main已有的缓存布局。

mixed每步也新增Host签名检查；融合核仍有GQA重复norm、gate读取和同步成本。没有真实设备计时之前，不能保证这些节省一定转化成明显端到端收益。当前不缓存未经证明稳定的绑定/metadata来换取表面更少检查。

后续交给DS时先核对source digest、mixed实际入口和phase覆盖，再判断整批时间。若数值或绑定失败，保留首错并修复具体合同，不能绕过自检、放宽容差或关闭功能来宣布完成。
