# v2 与 main 接近：已找到的融合遗漏及修正

comment from gpt

2026-11-18。用户反馈已在 op-dev-v2 运行，但与 main 几乎没有差异。这里的 main 是已有 AC 的 `9c5ce34`，首版 v2 是 `e132137`，不能将两者误当成“没有 Replay 的 Base”和“Replay”。

## 结论

首版 v2 没有证明达到端到端提速目标。复查发现一个确定的接入缺陷：**融合分支错误要求 dt_bias 必须 FP32，而标准 BF16 Qwen 的 dt_bias 是 BF16，因此模型会继续运行独立 gate/norm helper。** CPU 接口复现已经确认同一组 TP4/HK4/HV12/Q4 参数在 `e132137` 被拒绝，修正后被接受。

这能解释为何部分预期工作根本没被删除，但没有本次设备日志和精确性能数，不能把全部耗时差额唯一归因于它。剩余几项优化规模有限，也可能被其他开销抵消。容量方面，v2 没有改变 main 的 state/records 分配规则，所以并发上限相近符合实现。

本次读取 [#4](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4)，共22条评论，最后更新时间仍为11月17日，没有找到本轮 v2 的设备日志或完整结果表。因此不引用旧 acc4/b480 日志证明 v2 实际路径，也不编造本轮加速百分比。

## 1. 为什么 BF16 bias 是原生合同

精确 tag `v0.30.0rc1`，以及本地 v0.30.0 源码均可核对：

1. Qwen GDN 的 `A_log` 创建时显式指定 FP32。
2. `dt_bias = nn.Parameter(torch.ones(...))` 没有指定 dtype。
3. model loader 在 `set_default_torch_dtype(model_config.dtype)` 内创建模型；BF16 模型的默认 dtype 因此是 BF16。
4. sharded/default weight loader 使用 `param.data.copy_`，保留已有 Parameter 的 dtype；即使 checkpoint 中参数原本是 FP32，也不会将目标 Parameter 自动改成 FP32。
5. 原生 gating 读取 bias 后转 FP32 计算，支持这种组合。

来源：[Qwen GDN 参数定义](https://github.com/vllm-project/vllm/blob/v0.30.0rc1/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L467)、[模型加载默认精度](https://github.com/vllm-project/vllm/blob/v0.30.0rc1/vllm/model_executor/model_loader/base_loader.py#L66)、[参数复制](https://github.com/vllm-project/vllm/blob/v0.30.0rc1/vllm/model_executor/model_loader/weight_utils.py#L1244)。

首版 v2 的 Python preflight、C++ Host 校验、Device 指针类型三处都限定 bias 为 FP32。这是实现对原生合同支持不完整。启动自检只构造 FP32 bias，且融合调用次数包含专项自检，所以“READY + verify_fused计数非零”也可能同时成立。上轮验收覆盖不够，应修正，不能归因于用户分支不正确。

## 2. 本次代码修正

- Python preflight 接受 FP32 A_log + BF16/FP32 dt_bias；尺寸、连续性、设备检查保留。
- C++ 根据实际 bias dtype 选择 Device 模板；Device 直接按 BF16/FP32 读取，随后转 FP32 计算。没有修改模型 Parameter，没有每步 `.float()` 或参数副本。
- CPU fixture 改用真实 BF16 Parameter bias；测试合法/非法 dtype、原对象传递和存储精度舍入。数值参考使用量化后的 bias 再转 FP32，不用原始未舍入 FP32 常量替代。
- mandatory startup probes 增加两种 bias 的全部已有组合，保留数值容差和 checkpoint/record/边界检查。
- `model_dispatch=PASS` 增加参数 dtype 与 Q/V token stride；READY 增加模型入口和自检总调用的独立计数。

另一个可以从源码直接删除的工作是 `rearrange_mixed_qkv` 的 QKV 副本。首版融合仍调用它；原生实现通过 `torch.cat` 物化连续 Q/K/V。AC `LoadWindow` 已支持 packed token stride，因此修正版的 fused 分支直接 split + view，保留原 conv_output storage。独立 helper 分支保持原行为。

TP4 的 HK4/HV12/K=V128，packed width=(2×4+12)×128=2560。Q4、BF16 时原 cat 每请求每层处理20KiB，读写约40KiB；48层约1.875MiB，请求行24时约45MiB/轮/卡的逻辑读写。删除副本还减少一个显式 copy helper。这些是结构量，不是实测总线流量或时延保证。

## 3. 为什么不能期待容量再涨、时间大幅下降

v2 与 main 都已有相同 RecoverSSM 缓存布局、只读 verify checkpoint、采样后 fold。此次新增 UB 是片上临时存储，不会额外释放每请求 HBM state。因此 main→v2 的容量倍率应接近1。

最后 token 的改写只省去一次完整 rank-one 更新。Q4 时粗略约是 verify 主矩阵算术的7%；不等于 GDN 全路径快7%，更不等于端到端快7%。举例而言，如果 verify 只占整批时间10%，即使其耗时真下降7%，整批也只下降约0.7%；这里10%是假设，当前未测。

窗口集中写回减少搬运指令和依赖阶段，但有效数据字节不变。fold 的 decay 窗口加载减少 Scalar GM 读取，却没有消除完整 state 恢复。融合又会引入按 value head 重复的 q/k norm（GQA3），以及 gate 的 Scalar/Vector handoff，节省和新增成本可能互相抵消。不能仅凭接口选中就认定净加速。

当前更实质的下一步是：先确保原本设计的融合确实进入模型，并删除其输入整段副本；随后基于现有正常完整结果判断是否值得继续投入整矩阵归约、重复 GQA norm、gate 参数读取和 fold 的计算结构。主权重、长上下文 attention、prefill 和已有恢复依赖没有被本轮一并加速。

## 4. 如何确认修正实际进入，不改变压测方式

同一个 op-dev-v2 分支更新后，正常重启 server，使用原启动命令与原完整负载。已有 graph 内的分支是在捕获时确定的，运行中 git pull 不能替换它。

- PATCH_ID 应为 `ascendc-op-dev-v2-bf16-bias-qkv-views`。
- 自检包含 `dt_bias=fp32,bf16 bias_storage_promotion=PASS`。
- 模型进入时查看 `GDN_ASCENDC_FUSION model_dispatch=PASS`。标准 BF16 模型应显示 `A_log=torch.float32 dt_bias=torch.bfloat16`；packed view 的 Q/V token stride 应为2560。
- `verify_fused_model_calls` 只统计模型 Host dispatch；专项自检不计入。graph replay不重复执行Python，不能将其当成全程设备调用数。
- 保留四个 worker 的 ready/尾日志及相同完整负载的 Duration、TTFT、TPOT。无需增加开关、关闭 async/graph/prefix 或安排二分实验。

源码缺陷和 CPU 选择差异已确定；CANN 编译、NPU 数值与净性能仍以设备证据为准。本次不会以CPU测试通过代替端到端成功，也不会将修正后的开发版直接提升 main。

## 5. 本次开发侧验证

完整 CPU 套件 **302 passed**。新增覆盖真实 BF16 Parameter bias、TP4/HK4/HV12/Q4 的修正前后选择、BF16存储值参与gate计算、QKV views的共同storage/offset/stride、单token与padding、GQA几何，以及融合与普通路径的完整fence/conv/verify/save顺序。

修改后的 Torch Host 函数、注册和 Meta 入口在本地模拟 NPU 接口的编译检查中通过；FP32/BF16 bias 的设备模板也通过普通 C++17 stub 语法检查。Python语法、相关新增模块Ruff和diff检查通过。NPU API由stub代替，所以这些检查不能证明CANN设备编译或实际性能。
