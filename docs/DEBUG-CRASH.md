# 崩溃排查：先分三档，别再猜

> 这份文档只做一件事：用**最少的实验**把故障范围缩小到一段确定的代码。
> 每一档的结论都是"机器给的"，不是推理出来的。

---

## 第 0 步：10 秒确认"插件到底有没有在跑"（不做完这一步，后面都白做）

在**装了 vllm 的那个 python 环境**里执行：

```bash
python - <<'EOF'
from importlib.metadata import entry_points
print("entry points:",
      [e.name for e in entry_points(group="vllm.general_plugins")])

import gdn_recoverssm_ascend.config_patch as c
print("module  :", c.__file__)
print("logger  :", c.logger.name)
EOF
```

判读：

| 输出 | 结论 |
|---|---|
| `['gdn_recoverssm_ascend']` + `logger.name` 以 `vllm.` 开头 | ✅ 插件已装、且是修复后的版本 |
| 列表里没有它 | ❌ 装到了别的环境 → `pip install -e plugin` 重装 |
| `logger.name` 是 `gdn_recoverssm_ascend.config_patch` | ❌ 还是旧代码 → `git pull` 后重装 |

启动后必须能在 server 日志里看到：

```bash
grep -E "RecoverSSM" server.log
```

```
INFO ... [config_patch.py] Ascend GDN RecoverSSM enabled: window=4 ...
INFO ... [layer_patch.py]  Ascend GDN RecoverSSM commit context: 48 layers, ... state tensors per layer=3, replay shape=[12, 4, 257] ...
INFO ... [layer_patch.py]  Ascend GDN RecoverSSM step 1: prefill=0 decode=1 spec_rows=1 ...
```

**这三行一个都没有 → 插件没生效，先解决这个，不要看崩溃。**

---

## 第 1 档：单请求、无并发（区分"几何错"与"batch 相关"）

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export GDN_RECOVERSSM_TRACE_STEPS=20        # 想看更多步就调大

vllm serve <model> --tensor-parallel-size 4 \
  --max-model-len 4096 --max-num-seqs 1 \
  --mamba-backend triton --enforce-eager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --use-replayssm
```

发 1 条短请求。

| 结果 | 结论 | 下一步 |
|---|---|---|
| 不崩 | 基础几何是对的 | 进第 2 档 |
| 崩，且崩之前有 `step 1:` | 几何/内核问题 | 把 `step 1:` 那行 + 第一条 ERROR 贴出来 |
| 崩，且**没有** `step 1:` | 说明根本没进 verify | 回第 0 步 |

---

## 第 2 档：放开并发（区分"单请求"与"多请求"）

只改一个参数：去掉 `--max-num-seqs 1`。同样 4096 上下文、单条请求。

| 结果 | 结论 |
|---|---|
| 第 1 档不崩、第 2 档崩 | 与 batch 容量/plan 缓冲区相关 |
| 两档都崩 | 与并发无关 |

---

## 第 3 档：chunked prefill（用户现场）

长 prompt（> 2048 tokens） + 多个并发请求 + 默认 `--max-num-batched-tokens`。

这一档会触发 Ascend GDN builder 的这条特殊逻辑：

```python
# vllm_ascend/ops/gdn_attn_builder.py:782
if num_decodes > 0 and num_spec_decodes > 0:
    num_prefills += num_decodes      # 普通 decode 行被并进 prefill 桶
    num_prefill_tokens += num_decode_tokens
    num_decodes = 0
```

也就是说：**同一批里既有投机行又有普通 decode 行时，普通 decode 行会走 chunked-prefill 代码路径**。
这与"chunked prefill 才崩"的现象吻合，也是当前最大的未验证风险（见下）。

---

## 已知的两个结构性风险（尚未验证）

### R1 路径不统一（高）

上游 RecoverSSM 会**强制把所有活跃 decode 行都放到投机路径**：

```python
# recoverssm_utils.py（上游）
active_decode = (~is_prefilling_cpu[:num_reqs]) & (query_lens > 0)
spec_mask = (num_decode_draft_tokens_cpu[:num_reqs] >= 0) | active_decode
drafts[:num_reqs] = torch.where(spec_mask, 1, -1)
```

理由是：RecoverSSM 把 conv 窗口从 `conv_kernel-1` 扩展成 `conv_kernel-1+Q`，
**所有 decode 行必须使用同一套窗口布局**，否则一行在不同步之间切换路径，conv 状态就对不上。

Ascend 的 builder 用 `spec_decode_mask = (drafts > 0) & is_decode` 判断，
**drafts==0 的 decode 行无法被强制进投机路径**。所以一个请求在
"有 draft → 无 draft"之间切换时，conv 窗口布局会切换 → **conv 状态错乱**
（表现为输出漂移，不一定是崩溃）。

### R2 之前的行数/映射 bug（已修，`16f4c2c` + `9864fb0`）

- 行数：`min(num_spec_decodes, len(qsl)-1)`（图模式 qsl 会被材料化到图大小）；
- 映射：spec 行 i ↔ 批行 `nonzero(spec_sequence_masks)[i]`。
  之前把"掩码长度 == 状态表行数"当判据，两者天生不等，导致**总是**回退到恒等映射 →
  chunked prefill 下 spec 行不在批首时，**把 A 请求的记录提交进 B 请求的状态**。
  这正是"commit 对不上"的来源。

---

## 如果第 3 档仍崩，按这个顺序取信息

```bash
# 1) 崩溃前最后 20 行插件日志（几何信息）
grep -E "RecoverSSM" server.log | tail -20

# 2) 最早的 ERROR（不是最后的级联错误）
grep -n "ERROR" server.log | head -3

# 3) 是不是远端错误级联
grep -n "SUSPECT REMOTE ERROR\|507035\|507057" server.log | head

# 4) 卡上真实的 fault（CANN 侧）
ls -d plog-* 2>/dev/null && grep -il "aicore error\|exception" plog-*/device-*/*.log | head
```

`507057 SUSPECT REMOTE ERROR` 表示**别的 rank 先挂了**，当前 rank 只是看到远端错误；
真正的根因在**时间最早**那条 ERROR 所在的 rank 上。
