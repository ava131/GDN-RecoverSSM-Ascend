# 内核移植说明（GDN RecoverSSM → Ascend A2）

源：`gdn_recoverssm_ascend/kernels.py`（1175 行）
上游参考：vLLM PR #61309 `vllm/model_executor/layers/mamba/gdn/recoverssm_gdn.py`（702 行）
+ `.../mamba/ops/recoverssm.py`（200 行）+ `.../mamba/recoverssm_utils.py`（37 行）

---

## 1. 与上游的对应关系

| 上游 | 本移植 | 行为差异 |
|---|---|---|
| `_gdn_recoverssm_verify_kernel` | 同名 | **仅门控几行**：`a/b/A_log/dt_bias` + softplus + sigmoid → 调用方预计算的 `g`/`beta`。状态数学、记录布局 `[corr(V), k̂(K), decay(1)]`、`pid_v==0` 写记录、null 行写 0、`tl.static_range(SPEC_QUERY_LEN)` —— 与上游逐字一致 |
| `gdn_recoverssm_verify` | 同名 | 新签名；`BV` 由 UB 推导（保留 32 上限）；加 `MAX_SPEC_QUERY_LEN` 检查；其余 `_require` 校验顺序不变 |
| `prepare_commit_plan_kernel` | `_prepare_commit_plan_kernel` | 仅改名 |
| `compact_conv_state_kernel` | `_compact_conv_state_kernel` | 仅改名 + `BLOCK_D` 由 UB 推导（保留 256 上限） |
| `_commit_gdn_state_kernel` | 同名 | **AST 完全一致**（只有注释差异） |
| `GDNRecoverSSMCommitContext`（`from_tensors` / `create` / `commit`） | 同名 | `_ptr_table` 内联；`spec_query_len` 范围检查；tile 由 UB 推导；align 的两次 launch 顺序不变 |

校验方式：对每个函数做 `ast.unparse()` 后逐函数 diff（注释/格式对结果不可见）。

---

## 2. Triton-Ascend 风险点与"语义不变"的 fallback

> 上机若编译失败，按此表逐项替换；每一行的 fallback 都与原实现**逐位等价或仅差 FP 结合律**。

| # | 风险写法 | fallback |
|---|---|---|
| 1 | 指针运算跨 `tl.static_range`（`rec_base` 在循环外构造，循环内 `rec_base + t*stride`） | 每个 token 从 `replay_ptr + state_idx*stride_block + pid_hv*stride_head + t*stride_pos` 重新算地址 |
| 2 | commit 折叠里的运行期循环 `for t in range(n)` | `for t in tl.static_range(SPEC_QUERY_LEN): if t < n:`（`n ≤ spec_query_len`，等价） |
| 3 | `tl.fma(c[:,None], kk[None,:], h*decay)` 二维外积 | `h = h*decay + c[:,None]*kk[None,:]` |
| 4 | `triton.heuristics` 从 `None` 指针推导 constexpr | 从 Python 侧显式传 `HAS_REQUEST_INDICES` / `ALIGN_MODE` |
| 5 | conv kernel 的 masked load 会算出越界地址（虽然被 mask 掉） | clamp 索引，或改用 KDA 的 `USE_1D_CHANNEL_GATHER` 逐历史位置 gather |
| 6 | `int64 → tl.pointer_type` 转换 + 每层地址表 | 每层一次 launch，tensor 作为普通指针参数（结果相同，N 次 launch） |
| 7 | null 行把 FP32 零写进 bf16 输出（隐式转换） | 显式 `.to(out_ptr.dtype.element_ty)` |
| 8 | `num_stages=2` / `num_warps=4` | 去掉 `num_stages`，`num_warps=1` |
| 9 | `tl.where(valid, new_state, state)` 二维 tile 的标量谓词 | 可整句删除：`t ≥ query_len` 时 `k=v=beta=0, decay=1`，`new_state == state` 精确成立 |
| 10 | `assert` 在 `python -O` 下被剥离 | 改用 `_require` 抛异常 |

---

## 3. UB 预算

`_ub_budget_floats()` 从 `get_ub_size_bytes()`（A2 回退 192 KiB）推每个 tile 的上限，并带显式断言：

| kernel | 192 KiB 下的 tile | 说明 |
|---|---|---|
| verify | `BV=32`（保留上游默认） | 约 52 KB 活跃 |
| commit | `BV=16` | 约 26 KB |
| conv | `BLOCK_D=256` | 约 8 KB |

小 UB 器件（16 KiB）会把 `BV` 降到 8；4 KiB 时断言直接报错（宁可失败也不要静默算错）。

> 活跃 tile 数是**人工估算**，不是编译器上报值；在 192 KiB 上从不成为约束。

---

## 4. 待上机确认的猜测（不是默默选择，而是明确记录）

1. **`BOUNDARY` 还是 `ALIGN_MODE`**：上游 kernel 的 constexpr 名是 `BOUNDARY`（`ALIGN_MODE` 只存在于 Python 层）。本移植保留 `BOUNDARY` 以保持 kernel AST 一致；如需改名是纯重命名。
2. **masked 通道的 `beta`**：本移植为 `0.0`，上游为 `sigmoid(0)=0.5`。可证明无影响（这些通道既不写状态也不写记录）。
3. **conv tile 不再增长**（不超过上游的 256），即使 UB 允许——这是留给硬件上调的性能旋钮。
4. **无 UB 回退**：`init_device_properties_triton()` 未初始化时 `get_ub_size_bytes()` 会抛断言。Ascend worker 在 `init_device` 里调用它，serving 路径没问题；**算子级单测必须在 fixture 里先初始化**。
5. 保留了 `commit_conv=True` 与 `create(*, spec_query_len, max_num_reqs)` 的关键字参数形式（上游形式）。

---

## 5. 本地已跑的检查（无 NPU）

- `ast.parse` 全部通过；
- stub-import 测试（伪造 `torch` / `tl` / `triton`）能建出 dataclass 与常量；
- 4 个 kernel 的 launch 参数个数与定义一致（28 / 20 / 14 / 19）；
- UB 决策表：192 KiB → 上游默认；16 KiB → BV=8；4 KiB → 断言；
- 数值语义由 `plugin/tests/test_math_oracle.py` 用 NumPy 独立验证（记录布局、任意接受长度的提交、align 边界算术）。
