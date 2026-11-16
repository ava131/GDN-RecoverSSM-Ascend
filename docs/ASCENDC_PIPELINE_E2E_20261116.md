# AscendC 第二轮：降低等待，补齐容量与吞吐建模日志

日期：2026-11-16。开发分支：`op-dev`；改动基线：`9c5ce344d33f94ac0416e2ba1c12ea2ae587a467`。PATCH_ID：`ascendc-op-dev-pipeline-v2`。本文供后续开发者和部署方交接。

## 1. 目标与验收口径

目标是保持现有启动参数、采样配置、图桶和数值语义，缩短相同工作负载的端到端时间。阶段目标是先消除上一版 AC 相对 Base 约6.16秒的差距，而不是提高某个微基准的倍数。

已知上一组 graph、客户端24并发、约23k输入/1k输出的数据：Base741.49秒、Triton Replay754.06秒、AC Replay747.65秒。该表是外部整批测量，不能由本地 CPU 测试重建。当前改动尚无新的设备性能结果，不宣称阶段目标已经达成。

这轮保持每轮 RecoverSSM 算法形态：verify写短记录，采样后fold重建接受前缀。跨轮ring、output-only验证、周期flush属于下一层结构改造，本轮没有混入其状态协议。

## 2. 核函数改动及原因

### verify：输入等待4→1，输出完成等待Q→1

之前按顺序加载state、q、k、v，每次load后都做一次MTE2→Vector等待。现在为三个raw窗口分配互不重叠的UB区域，先发四组搬运，再统一等待；转换和递推的算术顺序不变。

之前每token写correction、key、decay、output后立即等MTE3完成，才开始下一个token。现在为每个token保留独立的correction/output/packed-output/decay源槽，后续token可以继续计算，直到整窗口结束才统一等待输出完成。每token的Vector→MTE3发布仍保留。

安全条件：

- raw q只有在整个输入窗口转换完成后才复用为packed-output。
- raw key及FP32 key在窗口期间不覆写。
- 各token DMA源槽不重叠，decay每token独占32字节槽。
- 每个RunTask及ZeroOutput返回前均drain输出；下一任务的Vector→MTE2依赖保护UB再次用于输入。
- 不改变state递推、DotRows归约顺序、cast舍入或GQA key写入规则。

### fold：输入等待2→1

state、correction、key的输入区域原本已经不重叠，现在统一发起加载，再等待和转换。boundary/final写回后的等待保持，尤其保留boundary写回source后继续计算的保护。

### UB预算同步覆盖Host和Device

| 项目 | verify | fold |
|---|---:|---:|
| UB布局 | 168,288 B | 112,640 B |
| 加2,048 B guard后 | 170,336 B | 114,688 B |

verify增加了11,488 B，仍在192KiB预算内。`common.h`集中定义Host/Device资源合同；两个header的static_assert检查实际布局与合同相同，`ops.cpp`检查真实设备UB不少于170,336 B。不能只扩Device布局却保留旧Host门槛。

代码位置：`plugin/gdn_recoverssm_ascend/csrc/{common.h,verify.h,fold.h,ops.cpp}`。ABI字段和用户配置没有改变。

## 3. 删除每轮重复的中性buffer填充

`builder.py`中的`_accepted_buffer`现在初始化一次后只读使用。它供下一轮卷积读取固定中性值1，并不是采样实际接受长度。

依据是Ascend rc1 native builder在`super().build()`内操作自己的padding buffer；插件随后才把独立buffer绑定到顶层和嵌套卷积descriptor。FLA的卷积只读取该接受长度，不修改它。真实采样长度仍由原模型状态流入恢复plan。

因此移除每组每步的`fill_(1)`提交，保留稳定地址、全部行初值1、容量检查和双层descriptor绑定。实际收益取决于组数及执行路径，不提前认领毫秒数。

## 4. 自动日志：记录了什么

不新增用户开关。保留原`GDN_CONFIG/GDN_READY/GDN_PERF`，新增或完善以下证据：

| 标记 | 用途与口径 |
|---|---|
| `GDN_CONFIG` | PATCH_ID、源指纹、原启动配置与恢复策略；HF配置精度不冒充实际cache精度 |
| `GDN_RUNTIME_LAYOUT` | 首个实际context的state dtype/shape/stride、record dtype/bytes及已准备backend；不是整模型所有组的内存总和 |
| `GDN_BATCH_SCHEMA` | 所需字段是否确实存在为CPU NumPy数组；缺失字段不读取设备Tensor |
| `GDN_WORKLOAD` | 已完成输出对应的batch均值/峰值、GDN行数、commit行数、接受分布、draft接受率、context以及prefill/mixed/decode分布 |
| `GDN_PERF` | 原累计提交计数、主机提交时间；graph_hist改为全桶，无图观测时padding fraction为-1 |
| `GDN_STAGE_HOST` | 低频主机墙钟范围，不是设备核耗时 |
| `GDN_TIMING` | 明确设备计时不可用及没有引入计时event |

`GDN_WORKLOAD`重点字段：

- `mean_rows/peak_rows`是所有已知batch快照的整批请求数；`mean_gdn_rows`是GDN投机子集；`mean_commit_rows`是恢复协调器计数。三者不可混用。
- `spec_emitted_per_row`包括每行的一枚正常输出；`draft_acceptance`才是被接受draft数/真实draft槽位数。槽位同时受QSL和实际draft数量约束。
- `accepted_output_hist`保留完整有效输出数量分布；`batch_hist`按phase保留完整batch分布。
- `prefill_row_target_tokens`是prefill行的全部target输入量，不称为去重后的纯prompt token数量。
- `mean_computed_context/max_computed_context`来自已有CPU数组，能帮助检查实际长度。
- `batch_snapshot_steps/unknown_phase_steps/prefill_tokens_known_steps/context_known_rows`报告覆盖程度；未知数据用-1，不能当0。

统计在原有`AsyncOutput.get_output`完成输出拷贝等待后消费CPU数组，不新增D2H。生产者不可变StepSnapshot跟随metadata传到graph/commit计时，避免异步下一轮的LATEST覆盖上一轮标签。

周期汇总仍为前三步及每200个完成输出步骤；空闲约5秒补报最终累计状态，退出也尽力汇总。空闲补报依据revision，避免最后一个周期点输出后才完成commit而漏报。

### 主机阶段时间如何理解

每个scope首次以及每128次采样，记录均值/最小/最大：

- `target_fullgraph_host_submit`：原生updatable graph提交范围，包含整张target图的相关主机工作，不是纯GDN verify。
- `target_to_recovery_host`：prepare结束到postprocess入口的主机区间，包含target和sampling等路径，不包含随后drafter。
- `recovery_host_submit`：原协调器提交恢复工作的主机区间。

三个范围有重叠，不可相加，不可直接代入设备roofline或称为GPU耗时。`worker_observed_s`是worker累计观察区间，不是客户端Duration，也不会排除进程生命周期中的所有空闲。

### 为什么没有加NPU event计时

核实[官方Torch-NPU v2.9.0 NPUEvent.cpp](https://github.com/Ascend/pytorch/blob/v2.9.0/torch_npu/csrc/core/npu/NPUEvent.cpp#L159)后，发现`elapsed_time()`内部会flush队列并同步event，提前query完成也不能保证不改变异步时序。当前不引入这条路径，也不使用未经验证的ctypes绕过。

日志明确：`device_timing=UNAVAILABLE reason=framework_elapsed_time_may_flush_streams phase_timing=HOST_WALL_NOT_DEVICE timing_events=0`。当前日志能改善工作负载和调度模型，不能分离出真实的verify/fold设备耗时占比，也不能据此确定设备是否已饱和。

## 5. 自动启动数值保护和本地检查

原18组Q1/4/8、BF16/FP32 activation/state/beta专项验证保留，新增两组部署几何HK4/HV12、部分长度窗口、交错null/零长度行的336任务自检，覆盖多次grid-stride复用。总计20组verify+fold探针。

补充了此前没有真正覆盖的alias：boundary写回source，随后继续更新，final写不同页。对无效行、部分行未使用记录位置和未修改页都检查sentinel；校验全部key heads的原始bits。

新增CPU检查包括UB资源/延迟DMA源生命周期、Host/Device预算合同、中性buffer只读稳定地址、异步统计快照、完整直方图/unknown口径/最后汇总等。

本地结果：**203项CPU测试通过**；修改过的Python文件Ruff通过；`git diff --check`通过。开发机没有CANN/NPU，本轮尚未完成BiSheng编译、设备数值、FULL graph运行或性能验收。启动自检是部署时自动执行的保护，不是本地已经执行的设备证据。

## 6. 保持原命令，怎样查看一次正常全量运行

更新`op-dev`，在原服务环境按原方式安装插件。已有editable安装会直接使用新源码；如非editable，沿用：

```bash
git switch op-dev
git pull --ff-only
python -m pip install -e ./plugin --no-deps
```

沿用现有Replay启动命令和完整压测，保存服务日志即可。第一次启动按新指纹自动构建AC，后续缓存命中；不需要手动删除缓存，不增加参数，也不要求关闭async/graph或二分实验。

```bash
rg 'GDN_CONFIG|GDN_ASCENDC|GDN_READY|GDN_BATCH_SCHEMA|GDN_RUNTIME_LAYOUT|GDN_TIMING' server.log
rg 'GDN_PERF|GDN_WORKLOAD|GDN_STAGE_HOST|GDN_DIAGNOSTICS' server.log
```

核对顺序：

1. PATCH_ID为`ascendc-op-dev-pipeline-v2`，源指纹变化；AC加载、自检20组及READY成功。
2. 以原压测客户端的Duration和真实token总量判断端到端收益。首次构建属于启动成本，不能与只覆盖请求阶段的旧Duration混计。
3. 看`batch_hist/phases/mean_gdn_rows`，区分纯decode、mixed及尾部的有效batch；不要用峰值代替全程平均。
4. 看接受分布和draft接受率，检查产出工作量是否可比；检查schema及coverage，未知项不要强行建模。
5. 看完整graph_hist及padding，而不是沿用旧B25→108推测；看HOST阶段是否存在提交侧变化，保留其测量边界。

Base在feature disabled时不会生成这套Replay诊断。仅靠Replay日志不能倒推出Base相同batch的开销曲线；这一限制保留。下一步判断性能瓶颈时，不把主机计时包装为缺失的设备计时。

## 7. 后续开发建议

先根据相同负载的端到端结果和新增工作量日志评估本轮。需要进一步结构升级时，单独设计request-owned ring、周期flush、output-only求解、prefix-cache边界和异步epoch，重新核算新增record容量。不要直接跳过fold、减少保护或将所有小核换语言后认领论文收益。
