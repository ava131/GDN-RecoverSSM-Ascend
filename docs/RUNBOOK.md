# 原启动命令不变的完整验收

本轮补丁以整批请求完成时间为目标。保持服务器已经使用的完整启动命令和环境不变：TP、MTP、async scheduling、prefix caching、align、max-num-seqs、max-num-batched-tokens、max-model-len、图模式及全部 capture sizes 都沿用原值。插件不修改这些参数，也不新增试验开关。

先更新并安装本仓库的 plugin，然后用原命令启动。不要把本工作区的 v0.23.0 参考源码安装到服务器；适配参考仍是现场 vLLM 0.30 系列 / vllm-ascend 0.30rc1 的 MRV2 接口。

## 安装与版本记录

在服务器原来的 Python 环境中执行：

```bash
git rev-parse HEAD
python -m pip install -e ./plugin
python -c "from importlib.metadata import entry_points; print([e.name for e in entry_points(group='vllm.general_plugins')])"
```

应包含 gdn_recoverssm_ascend。随后执行保存好的原始完整启动命令，保持所有选项、环境变量及重定向位置不变。本手册故意不重新拼出另一条 serve 命令，以免遗漏现场量化、draft 或 capture 参数。

GDN_RUNTIME 日志记录实际已安装的 plugin、vllm、vllm-ascend、torch、torch-npu 和 triton-ascend distribution 版本。GDN_CONFIG 的 source_fingerprint 标识实际插件源码，fingerprint 标识启动参数。日志中的 source_pair 是本适配器审查所依据的源码 SHA，并不是声称现场 wheel 必然来自同一 SHA。保留原有软件/驱动版本记录。

## 启动阶段自动完成的检查

无需添加自检环境变量。启用 RecoverSSM 后，启动会：

1. 输出 GDN_CONFIG，列出解析后的原始参数及 fingerprint。
2. 准备 bounded sampling filter，打印 GDN_SAMPLER ready；有值排序接口且启动探针一致时使用 values-only，否则仍用有界分块的原排序。没有替换拒绝采样或随机数语义。
3. 在 KV 已绑定后创建 typed-SoA records，按真实 GQA key heads 和 key dtype 计算实际 bytes；compile-only 准备真实页/record几何、activation布局的verify签名，以及实际none/align、索引dtype及行容量的commit变体，不执行完整模型dummy forward。
4. 准备多组提交的完整签名；不能把不同组的物理 block ID 当成相同。
5. 在自有小型合成缓冲中检查顺序 GDN 输出、接受前缀和卷积；覆盖完整 Q 窗口、GQA、padding/null 与 align 边界。自检失败不会发布 ready。
6. 保留正常 run_fullgraph 的启动首次执行，完成 native update/event 依赖后释放插件额外保留的 capture metadata。服务期不会自动重试可能已经部分执行的图。
7. 完成本机原生预热和 capture 后，输出每个 worker 的 GDN_READY patch=... validation=STARTUP_COMPLETE。

自检保护并不等于整模型验证已经完成。它不改动模型真实 KV，也会恢复随机数状态。

```bash
grep -E 'GDN_CONFIG|GDN_RUNTIME|GDN_SAMPLER|GDN_VERIFY_PREPARED|GDN_PREPARED_CONTEXT|GDN_RECORD_BUDGET|GDN_READY|GDN_GROUPED_SELFTEST|startup preparation complete|selftest.*SUMMARY|full-graph warmup' server.log
```

检查四个 rank 都出现最终的 `GDN_READY patch=... validation=STARTUP_COMPLETE`；较早的 `GDN_PREPARED_CONTEXT` 只表示几何与预分配，不能代替 ready。确认 `GDN_GROUPED_SELFTEST PASS`，且原先图模式的 worker 都完成 captured graphs 的首次执行。eager 没有图预热行是正常的。

检查 source 中实际状态 dtype。Qwen 本轮 BF16 激活、TP4、48 GDN 层、HV=12/HK=4、Q=4、records128行的 typed-SoA 总量为：

```text
128 × 48 × 4 × (12×128×4 + 4×128×2 + 12×4)
= 177340416 bytes = 169.125 MiB / rank
```

其中 correction/decay 保持 FP32，key 保存原本已归一化输入的精确 BF16 值；GQA 共享 key 仅保存一份。此数值不能套给其他 dtype/head 几何。原有 512 MiB transient headroom 保留并独立列账，不能当作 records 实际大小或图池测量。

## 请求阶段直接读取服务端证据

照原来的完整 aisbench 工作负载运行一次，不减少请求数/输出长度，不关闭任何功能。GDN_PERF 使用原有 AsyncOutput 完成后的 CPU 输出统计；不新增设备标量读取、D2H 或服务期 synchronize。

```bash
grep 'GDN_PERF' server.log
grep -E 'RecoverSSM graph dispatch|GDN_SAMPLER active|RecoverSSM commit' server.log
grep -E 'Traceback|OutOfMemory|MISMATCH|MTE accesses|SUSPECT REMOTE|507899|561002|replay failed|startup.*failed' server.log
```

| 日志字段 | 正确读法 |
|---|---|
| emitted_tokens | worker 已观察到的真实输出量；不能拿所有 rank 相加当客户端总输出 |
| spec_emitted_per_row | 按有效 spec 行次加权的推进长度，包含目标 token；检查数值/接受行为是否退化 |
| host_submit_us_avg | Python/Host 提交成本，明确不是 kernel 或 TPOT 时间 |
| kernel_submissions | 累计提交数，结合 commits/每提交平均数判断是否走合并路径 |
| graph_padding_fraction | 运行时 padding token / 捕获 token；只观测，不改 capture sizes |
| graph_hist | 实际 token 数到捕获桶 token 数的主要分布 |
| worker_observed_s | worker 观察区间，不能替代客户端整批开始到最后完成的 Duration |

前几步、每200次完成输出及空闲后的尾部汇总会自动打印。不需要提高 ASCEND_GLOBAL_LOG_LEVEL，也不要求 profiler 才能确认插件进入了对应路径。若需要精确 device 耗时，由开发侧在不变配置下单独取得 profiler trace，不能把 Host 日志数字写成 device 时间。

## 一次完整验收的通过条件

- 四个 worker 成功 ready，无 startup selftest MISMATCH，图模式完成原有 capture 桶的首次执行。
- 同一120请求、相同23k输入集、1024输出和原有参数完成到最后一个请求，不出现退出阶段错误、OOM 或失败后自动重放。
- 客户端整批 Duration、输出吞吐、TTFT/TPOT、接受长度及输出正确性一起保留。功能通过和端到端提速分别报告。
- GDN_CONFIG 对应的启动参数与原运行一致；若原生配置规范化改变 fingerprint，日志会列 before/after，不能把不同 capture 策略混进性能比较。
- GDN_PREPARED_CONTEXT 的实际 record bytes 不超 KV budget 预留；记录独立于图池和固定 transient margin。
- 只有原生纯投机 batch 进入 direct-output；混合 prefill 和普通 decode 保留 native 分流，spec 行仍由 RecoverSSM 验证和提交，不因性能优化跳过 state 更新。

不要由 max bs 快照反推“必然算力受限”或一个固定毫秒回退。最终是否缩短端到端时间，以这一完整工作负载的客户端 Duration 为准。

## 本地开发检查与限制

```bash
.venv/bin/python -m pytest plugin/tests -q
```

CPU 测试覆盖接口、所有权、分组索引、布局/公式及错误传播。开发环境无 Ascend NPU时，测试通过不能宣传为 NPU 数值、CANN 图稳定性或端到端加速已经通过。上面的自动启动检查和完整原负载用于在部署侧一次性完成实际验证；不要求反复关功能或更换批量二分。

## 想亲手实现一个小函数

可以手敲 `plugin/gdn_recoverssm_ascend/metrics_math.py` 的 `padding_fraction(actual_tokens, captured_tokens)`。它只有参数校验、零长度处理和一个除法；例如实际100、捕获108，应得到8/108。它只用于日志比例，不选择图、不调整capture sizes、不影响采样或state。

```bash
.venv/bin/python -m pytest plugin/tests/test_metrics_math.py -q
```

手敲后只运行这个CPU测试即可确认该函数；服务器启动命令仍保持原样。
