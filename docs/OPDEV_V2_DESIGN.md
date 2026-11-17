# op-dev-v2：从 9c5ce34 开始的 AscendC 优化交接

comment from gpt

## 1. 范围与当前结论

本分支从 main `9c5ce344d33f94ac0416e2ba1c12ea2ae587a467` 新建。main 和旧 op-dev 不变；本轮仅修改插件、插件测试和文档。不修改 vLLM、vllm-ascend、FLA 源码，不增加用户开关，不改变原启动参数。

目标是缩短同一完整负载的端到端时间。实现集中在四个可直接减少执行工作的点：集中写回 verify 窗口、融合 norm/gate、最后 token 只计算输出、fold 一次加载 decay。没有把旧 op-dev 的输入/输出流水重排整体搬过来，也没有改恢复协调器、请求生命周期、地址表更新或 async 的依赖协议。

这是待设备验证的开发版本。CPU 测试和本地 Host 接口检查通过，尚无本分支的 NPU 编译、自检、full graph 稳定性或端到端收益数据。不能把减少 helper 数量直接换算成已经实现的加速。

## 2. 稳定版保留了什么

- checkpoint 在 verify 中只读；采样后仍完整执行恢复。
- request-row 索引、producer epoch/context、每组物理页表和稳定设备地址表保留。
- source、final、align boundary 可能别名；fold 从原 checkpoint 只读入一次，在同一递推中生成边界与最终状态。
- correction/decay FP32，key 保留 activation dtype，并按 key head 共享。没有恢复成全 FP32、按 value head 重复存 key 的布局。
- plan、conv compaction、align publisher 仍是原有 Triton 核；verify/fold 继续用 AscendC。
- 算子仍发射到当前 NPU stream，没有新增 stream、请求期编译、设备标量回读或 host synchronize。
- 不支持的融合签名在状态卷积之前选择现有 helper 路径。卷积或 AC 执行开始后，错误直接传播，不能重试其他后端。

## 3. verify 窗口写回

文件：`csrc/verify.h`、`csrc/common.h`、`csrc/ops.cpp`。

原版每 token 的 correction/output 使用同一 UB 槽，计算后立即写 GM 并等待完成。v2 为窗口内每 token 分配独立 correction/output 槽，递推结束后才发射写回。

- `StageToken` 仅完成 activation 输出转换，不发射 DMA。
- `SaveWindow` 发布一次 Vector→MTE3 依赖，批量写 output/correction/key，再写实际长度的 decay。
- UB 内行紧密存储，GM token 间隔使用字节 gap；超出多块指令表示范围的合法写布局保留逐行 store。
- decay 仅写 `count * 4` 字节。Q4 为 16 字节，不能为了 32 字节对齐而覆盖相邻 head/row。
- key 仅由该 GQA 组的首个 value head 写出。
- 写回完成同时等待 MTE3→Vector 和 MTE3→Scalar。后者保护下一 task 对 gates 的 Scalar `SetValue`，只等 Vector 不够。

verify 固定 UB 从 156800 增至 **163968 字节**，Host 与 Device 共用常量，并预留 2048 字节检查余量；小于 A2 的 192 KiB。窗口容量仍为 Q≤8，没有 batch 规模的新增常驻矩阵。

Q4 下，每个 request/value-head task 从四次 token 写回阶段降为一次窗口写回阶段；写出的有效业务数据量没有减少。收益来自 DMA 指令和依赖组织，不能按四倍估算性能。

## 4. 最后 token 的输出计算

文件：`csrc/verify.h` 中 `LastOutput`。

令 state 形状为 V×K，归一化后的 q/k 为 K 维，v/u 为 V 维，q 已含模型 scale：

\[
d_t=\exp(g_t),\quad D_t=d_t S_{t-1},\quad
u_t=\beta_t(v_t-D_t k_t),\quad
S_t=D_t+u_tk_t^T,\quad o_t=S_tq_t.
\]

窗口中间的 state 会被下一 token 使用，必须构造。最后一个临时 state 没有消费者：持久 checkpoint 没被 verify 修改，commit 消费的是 `(u,k,d)`。

因此最后 token 改为：

\[
o_Q=D_Qq_Q+u_Q(k_Q^Tq_Q).
\]

correction/key/decay 仍完整生成，接受任意前缀的恢复公式不变。这省去最后一次完整 V×K rank-one 更新，额外付出 K 维点积及 V 维乘加。K=V=128 时，粗略省下 `2*128*128-4*128=32256` 次标量运算；以原 verify 每 token 约 `7KV` 的矩阵主体运算估算，Q4 约是其 7% 的算术工作。归约、搬运、同步和其他模型层都未计入，**这不是端到端 7% 提速预测**。

公式改变了 FP32 计算结合顺序；不能声称逐 bit 等价。启动自检对照独立 FP64 递推检查输出和每个接受前缀。

## 5. norm/gate 融入 verify

文件：新增 `csrc/fusion.h`；`verify.h`、`ops.cpp`、`ascendc_backend.py`、`spec_forward.py`、`layer_patch.py`。

新增 `verify_fused` Torch-NPU 入口，输入原始 q/k/a/b 与 A_log/dt_bias。原 `verify` 接口保留，mixed batch 等原路径继续可用。融合只在已有 pure-spec core 适配中启用，prefill/mixed 不被强行改写。

计算遵循原 helper 的存储精度边界：

\[
\hat q=\mathrm{cast}_{act}\!\left(q/\sqrt{\sum q_i^2+10^{-6}}\right),\quad
\hat k=\mathrm{cast}_{act}\!\left(k/\sqrt{\sum k_i^2+10^{-6}}\right),
\]

\[
x=a+dt\_bias,\quad
\mathrm{softplus}(x)=\begin{cases}\log(1+\exp x)&x\le20\\x&x>20\end{cases},
\]

\[
g=-\exp(A\_log)\,\mathrm{softplus}(x),\quad
d=\exp(g),\quad \beta=\mathrm{cast}_{b.dtype}(\sigma(b)).
\]

BF16 的 q/k、beta 先 roundtrip 再参与 FP32 recurrence；key records 保存这个归一化后的 activation 值。norm 使用 Sqrt+Div，归约顺序与 Triton 也可能不同，dtype 边界相同不等于数值逐 bit 相同。

归一化和 gating 复用递推前空闲的第二个 state bank 作为 scratch，不再新增 UB bank。归一化对整个窗口使用 Vector 行归约和 Brcb 广播，不逐 token 回到 Scalar 读取 norm。softplus 的线性分支避免先计算溢出的 exp(x)。gate 参数及标量的 Scalar↔Vector handoff 明确发布依赖。

符合签名时，每个 GDN 层的模型调用移除两个外部 l2norm helper 和一个 gating helper，以及它们的中间结果往返。full graph 中这体现为图内任务减少，不是每次 replay 减少 Python 调用。

**成本与边界：** verify 按 value head 分任务，GQA 下 q/k 归一化会重复；本核也新增 Vector 归约和 gate 的 Scalar↔Vector handoff。融合节省的外部开销必须大于这些代价才能缩短时间。没有硬件数据之前不能认定一定胜过独立 helper。

写前 eligibility 校验 activation/state/record dtype、局部 head 几何、TP 整除、Q≤8、INT32 metadata、容量、布局、设备和 DMA 范围。校验在有副作用的卷积之前。融合不支持时仍使用原 helper 加现有 verify 后端，而不是执行半途回退。

## 6. fold 的 decay 读取

文件：`csrc/fold.h`。

原实现每次递推通过 Scalar 从 GM 读一个 decay。v2 在加载 records 时将被接受前缀一次搬入 32 字节 UB bank；循环从 UB 取值。只有接受长度的有效字节被读取，输入合同要求 decay token stride=1。

新增显式 MTE2→Scalar 依赖；已有 MTE2→Vector 不能代替它。state/correction/key 原有等待及顺序递推保留。fold UB 为 **112672 字节**。

没有实施反向 fold。它涉及额外矩阵、精度及资源/并行度取舍，不能只因减少递推步骤就判断更快；本分支先消除已有前向路径的明显重复工作。

## 7. 日志与建模口径

`PATCH_ID=ascendc-op-dev-v2-window-fused`。源码指纹包含 csrc；构建指纹包含源码和实际工具链，新头文件会触发新库，不需手动清缓存。

- `GDN_ASCENDC_FUSION model_dispatch=PASS`：模型首次使用融合入口；`eligible=0 reason=...`：为何保留独立 helper。
- `GDN_PATHS verify_fused`：包含专项自检；`verify_fused_model_calls` 和 `verify_fused_saved_helpers` 只来自模型入口。都是 Host dispatch 计数，不能当成 graph replay 的设备执行次数。
- `GDN_WORKLOAD`：完成输出的产出 token 直方图、mean/peak rows、prefill/decode/mixed 分布、CPU 镜像 context 长度。
- context 来自已有 NumPy 输入镜像，async 下可能乐观，不代表精确设备接受长度。
- graph 观测移动到原生 `run_fullgraph` 外层，legacy/updatable 两条成功路径各计一次。无观测时 padding fraction=-1，不误报成零 padding。
- 保留周期/空闲/退出汇总，不添加每层每 token 刷屏、设备计时 event 或设备 scalar 回读。

容量、吞吐和时间应分开判断。内存节省只有转化为更高实际有效 batch 或更少调度等待，才可能抵消恢复开销；峰值 BS 增大不自动意味着 Duration 下降。输入越长，其他 attention/prefill 工作越重，局部 decode 算子节省占整批时间的比例可能越低。

设稳定版端到端时间中可被本轮优化的部分比例为 f，该部分加速倍数为 s，其余条件相同，则理想时间比约为 `(1-f)+f/s`。本轮没有可信的设备时间 f，不能从 Host submit 时间或 helper 数量反推出它。日志用于约束 batch、接受行为、graph padding 和是否选中路径，最终仍以相同完整负载的 Duration/TTFT/TPOT 为准。

## 8. 验证、交接和下一步

本地已执行完整 CPU 套件：**278 passed**。覆盖原生命周期/协调器/接线，以及新 raw 入口、写前拒绝、错误不重试、布局/容量、Q1..8 地址边界、任意接受前缀和最后 token 数学恒等式。测试不模拟出硬件性能结论。

另外使用普通 C++ 编译器保留完整 Host 函数体、真实 Torch 校验和注册，只替换 NPU 头及设备 launch 机制，成功编译并加载，执行 ABI/verify/verify_fused/fold 的 Meta 接口。本机 Torch 2.15/C++20；这不证明部署工具链、Torch 版本或设备 API 编译通过。

Python 语法、改动 diff 和新增测试/诊断模块 Ruff 检查通过；wheel 离线构建成功，已检查其中包含新 `fusion.h` 及其余全部 csrc 文件。普通 C++17 的设备模板语法检查也通过，但 NPU API 使用的是 stub，不能替代 CANN 检查。部分原有文件仍有基线 lint 告警，本轮不宣称仓库全量 lint 清零。

服务器正常启动时，既有强制自检新增 raw 融合入口：Q1/4/8，BF16/FP32 activation/state/beta 组合，GQA3，零向量/极小范数、softplus 线性分支与 sigmoid 边界、checkpoint 只读、null/padding、接受0..Q、source/boundary/final alias。保留原 grouped/conv 检查，不调宽原有 FP32/BF16 容差。任一失败阻止 ready。

部署后沿用原参数和完整压测，保留四个 worker 的日志、客户端总请求/token、Duration/TTFT/TPOT。具体操作见 [验收文档](OPDEV_V2_SERVER_VALIDATION.md)。不需要关闭功能或改变 batch 的二分实验。

交接给 DS 时，应首先确认实际源码/二进制指纹、模型融合入口和四 worker ready。若数值或构建失败，利用首错定位到本轮具体入口及 dtype，不能放宽容差或绕过自检。若完整运行通过但时间不降，应结合已增加的 batch/接受长度/graph 分布排除工作量变化，再评估重复 GQA norm、Scalar 读取及归约 handoff 等本核成本。不能将开发版本直接替换 main；主线晋升应由实际完整结果支持。
