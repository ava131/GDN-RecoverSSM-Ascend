# AscendC 算子开发分支：op-dev

基线：main `406b7b014fb5bd0273d4ac2adfbc073221ad796f`。本分支插件版本为 `0.3.0.dev0`，PATCH_ID 为 `ascendc-op-dev-v1`。main 与稳定快照不改动。

## 当前交付与证据边界

已经实现 AscendC verify 与 grouped fold、Torch-NPU 注册、启动构建缓存、真实 LaunchSpec 分派和强制数值自检。这是可供部署验证的开发实现；开发机没有 CANN/NPU，尚未证明 BiSheng 编译通过、NPU 数值通过、FULL graph 稳定或端到端提速。不要将本分支直接称为已验证稳定版。

plan、conv compaction、align publisher 继续使用现有 Triton，实现与物理页规则不变。原生 vLLM、vllm-ascend、FLA 源码未编辑。没有新增用户开关、改变原启动参数、限制 batch 或调整 capture sizes。

## 1. 为什么先实现这两个核

verify 原先按 V tile 重复准备 Q/K/V。新核按请求×value head 持有完整128×128 FP32 state，最多8 token的 QKV 窗口只加载一次，使用两个 state bank：新 bank 保存递推结果，已消费的旧 bank 用作归约 scratch。state GM 在 verify 期间只读；输出、FP32 correction/decay 与原始 normalized key bits写入独立 buffers。

fold 原先按 V tile 多次加载同一 key。新核按层×请求×value head加载完整 checkpoint 及接受前缀的 records，在 UB 中顺序递推，再保存 final；boundary 在同一递推途中保存，不重新读取已覆盖的 source。boundary==final不重复写。

这是任务划分和数据复用的改变，不只是将 Triton 语法翻译为 C++。也没有改变恢复数学或省略恢复步骤。

## 2. 文件与执行路径

| 文件 | 职责 |
|---|---|
| `csrc/common.h` | 共享 POD ABI；28个verify整数字段+scale，17个fold字段 |
| `csrc/verify.h` | AscendC 向量 verify |
| `csrc/fold.h` | AscendC 向量 grouped fold |
| `csrc/ops.cpp` | Torch schema、Meta、PrivateUse1、输入检查及当前NPU stream发射 |
| `csrc/CMakeLists.txt` | BiSheng mixed host/device shared library |
| `ascendc_build.py` | 工具链发现、内容缓存、跨进程锁、构建/加载 |
| `ascendc_backend.py` | 真实 LaunchSpec 到 AC ABI 的适配、写前选择、执行计数 |
| `ascendc_selftest.py` | 自有缓冲的启动数值检查 |
| `kernels.py` | LaunchSpec 的 launch/compile_only共用同一分派 |
| `startup_patch.py` | GDN guard后构建，真实签名准备，自检，READY保护 |

原生 batch/metadata → 原插件 verify LaunchSpec → AC verify → 原生 sampling → 原协调器 plan(Triton) → conv(Triton) → fold(AC) → 原align/后续流程。

保留 request-row record 索引、producer context snapshot、每组独立物理表、稳定地址表及原有同流顺序。没有每步 H2D 指针表、Host synchronize、额外stream或请求期编译。

## 3. 支持范围与写前选择

本轮 AC 快路径支持：

- Ascend A2 / 910B，BiSheng目标 `dav-2201`。
- K=V=128，窗口Q为1..8，GQA head整除。
- BF16/FP32 activation、state与key；beta为activation dtype或FP32，g为FP32。
- 已外部归一化的 typed-SoA records，correction/decay始终FP32。
- state最内层K连续，V/head/block可有合法间隔；QKV可保留combined-QKV token stride。
- INT32 verify行元数据；grouped plan INT32，地址和stride表INT64。
- DMA行间gap非负且字节数可由UINT32表示。

FP16转换的溢出/饱和语义尚未验证，首版不实例化FP16核；原有FP16、packed record、内部归一化或其他布局保留Triton兼容路径，选择发生在提交前并输出原因。执行AC后发生异常不会再重试Triton。

实际部署的verify和grouped fold必须都成功选择并prepare AC，否则本分支拒绝最终READY，避免“专项自检在跑AC，模型却全走Triton”的假成功。已有启动参数无需增加backend选项。

Torch schema声明verify的checkpoint只读、四个输出可修改。fold通过epoch-owned地址表写多个state，当前只在采样后的图外执行；不要将它随意交给自动函数化或torch.compile重排。若以后将fold纳入编译图，必须补齐完整TensorList mutation/owner合同。

## 4. 设备资源与同步

- verify固定UB布局156800字节，另留2048字节检查余量。
- fold固定UB布局112640字节，另留2048字节检查余量。
- 使用向量乘、归约、广播及乘加；不是标量矩阵循环。
- 明确保护MTE2→Vector、Scalar↔Vector、Vector→MTE3及缓冲复用。
- 首版verify每token等待本核的输出搬运完成；这是设备内依赖，不是Host同步。后续可研究双输出bank重叠，但本轮不提前认领流水收益。
- 跨编译器不承诺位级相同，使用CPU FP64数值参考及现有严格/低精度容差检查；MulAddDst名称本身不是FMA舍入等价的证明。

## 5. 构建与缓存

服务器原环境需要已有CANN开发工具链（BiSheng、AscendC headers）、CMake、Torch和Torch-NPU。保持既有环境设置，不新增构建开关。

启动自动：

1. 发现实际CANN、Torch、Torch-NPU与compiler。
2. 将源码、编译器/库版本及ABI合成fingerprint。
3. 多TP worker通过文件锁共享构建；从源码快照构建并原子发布产物。
4. `torch.ops.load_library`加载，核对ABI，再执行自检。
5. 任何构建/加载/数值失败都阻止READY，不静默换回旧实现。

默认缓存位于系统既有XDG缓存目录或 `~/.cache/gdn_recoverssm_ascend/ascendc/<fingerprint>`；失败日志给出具体路径。不会向原生源码仓或全局OPP目录安装文件。第一次启动有构建成本，后续命中缓存仍需每进程加载和启动检查。

GDN_CONFIG的source_fingerprint也包含本插件的csrc源码，避免只更新头文件而Python指纹不变。GDN_ASCENDC的构建fingerprint另外包含工具链信息，用于确认实际加载的二进制。

## 6. 自动验证与日志

沿用原来的Replay启动命令；最新issue里展示的disable环境变量及注释掉use-replayssm属于Base示例，不应把Base启动成功当作AC启用成功。

部署更新仅需切本分支并在原Python环境重新安装插件：

```bash
git fetch origin op-dev
git switch op-dev
python -m pip install -e ./plugin
```

随后用保存的原Replay命令启动。不要安装开发机的CPU Torch环境到服务器。

```bash
grep -E 'GDN_ASCENDC|GDN_READY|GDN_VERIFY_PREPARED|GDN_GROUPED_SELFTEST|selftest.*SUMMARY' server.log
grep -E 'Traceback|AscendCBuildError|MISMATCH|SELFTEST FAIL|MTE accesses|507899|561002' server.log
```

证据顺序：

- `GDN_ASCENDC loaded`：库已加载，记录fingerprint、位置和cache_hit；不是数值PASS。
- AC专项自检：固定36次小型设备调用，覆盖BF16/FP32全部编译实例、Q1/4/8、GQA3、accept0..Q、null/padding、两层两组地址路由及source/boundary/final alias。
- 原有完整窗口verify/state、conv和grouped检查继续执行。
- `GDN_ASCENDC_SELFTEST PASS`：AC核实际启动过，实际模型签名也已prepare；不是端到端加速证明。
- 四个worker最终 `GDN_READY ... STARTUP_COMPLETE`：全部保护及原生warmup/capture完成。

所有专项数值检查用自有缓冲及确定性CPU构造，不读写模型KV、不改变RNG。启动检查中CPU复制会等待结果；请求期没有新增等待。

然后沿用完整原负载完成至最后一个请求，保留客户端Duration、TTFT/TPOT、输出/接受行为及server尾日志。第一目标是数值和生命周期完整，第二目标才是相对当前Triton实现缩短端到端时间。

## 7. 开发侧检查

```bash
.venv/bin/python -m pytest plugin/tests -q
```

新增CPU检查覆盖构建缓存/失败传播、真实Triton参数表到AC ABI的映射、写前选择与错误不重试、grouped地址表所有权、实际prepare/执行的READY保护、独立数值oracle及schema mutation。

交付前完整CPU套件145项通过。修改Python文件的关键Ruff检查与diff检查通过；CPU wheel打包成功，实际检查包内包含全部5个AscendC构建源文件。

另外通过临时普通C++ API stub检查设备头的模板语法；保留ops.cpp完整Host函数体、真实Torch校验和注册，仅模拟NPU接口后编译共享库，加载并执行ABI/verify Meta/fold Meta均成功。该probe使用本机Torch2.15/C++20；部署Torch2.10/C++17仍待验证。这些检查均不能替代CANN前端、设备API、UB编译资源和真机图捕获检查。

## 8. 后续性能工作

若本分支设备验证通过，再使用原配置的阶段事件或开发侧微基准比较verify/fold，确认整head任务的复用收益是否超过并行度/同步代价。可继续研究：

- 双输出bank避免verify每token等待写回。
- state读取与下一task输入预取重叠。
- 特定Q4下的调度与vector repeat优化。
- 在精度舍入合同明确后融合gating/norm。

本轮不调整capture sizes、内部限batch、不关闭async/prefix/align、不恢复异常后的盲目图重放，也不宣称“AC一定快于Triton”。

