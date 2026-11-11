#!/usr/bin/env python3
"""Build the GDN RecoverSSM report (DOCX) for the review meeting."""

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

OUT = "/Users/Admin/Workspace/GDN-RecoverSSM-Ascend/docs/GDN-RecoverSSM-Ascend-汇报-20261111.docx"

doc = Document()

# ---- base styles -----------------------------------------------------------
normal = doc.styles["Normal"]
normal.font.name = "Microsoft YaHei"
normal.font.size = Pt(10.5)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

for name in ("Heading 1", "Heading 2", "Heading 3", "Title"):
    st = doc.styles[name]
    st.font.name = "Microsoft YaHei"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")


def code(text: str) -> None:
    """Monospace log evidence block."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    run.element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")


def table(rows, header=True, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            run = p.add_run(str(cell))
            run.font.size = Pt(9)
            run.font.name = "Microsoft YaHei"
            run.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
            if i == 0 and header:
                run.bold = True
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    doc.add_paragraph()
    return t


def quote_cn(text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


# ===========================================================================
doc.add_heading("GDN RecoverSSM on Ascend：实现与性能现状汇报", level=0)
p = doc.add_paragraph("对应 RFC：vllm-project/vllm-ascend#19987　　"
                      "代码：ava131/GDN-RecoverSSM-Ascend `64a4de6`")
p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
p.runs[0].font.size = Pt(9)
p.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)

doc.add_heading("摘要", level=1)
for line in [
    "功能：在 Ascend MRV2 上为 Qwen3.8 的 Gated DeltaNet 实现 RecoverSSM —— 投机解码时只保留一份循环状态检查点，"
    "拒绝采样后按接受前缀一次性重建，取代“每个投机位置各存一份完整状态”的传统做法。",
    "形式：纯 vLLM 插件，未修改 vLLM / vllm-ascend 任何源码。",
    "正确性：启动自检在 NPU 上对照逐 token 递推，fp32 最大相对误差 6.944e-08（fp32 机器精度量级），"
    "卷积历史压缩误差严格为 0。",
    "容量：eager 模式下最大并发 base 20 → replay 25（+25%），与内存模型的预测一致。",
    "性能：整批耗时 742.4 s → 840.9 s（+13.3%），TPOT 100.2 ms → 118.8 ms（+18.6%），"
    "增量 75% 发生在首 token 之后，即每步成本。已定位并修复一处每步设备同步，待复测。",
]:
    doc.add_paragraph(line, style="List Bullet")

# ===========================================================================
doc.add_heading("1. RFC 的要求", level=1)
doc.add_paragraph(
    "以下逐节列出 RFC 原文与译文。原文为英文，译文力求准确对应，不作发挥。"
)

doc.add_heading("1.1 §1 把 RecoverSSM 生命周期扩展到 GDN", level=2)
quote_cn("原文：Reuse the existing RecoverSSM architecture and MRV2 post-sampling commit lifecycle.")
doc.add_paragraph("译文：复用现有 RecoverSSM 架构与 MRV2 的“采样后提交”生命周期。")

quote_cn("原文：During verification, the persistent recurrent checkpoint must not be overwritten by "
         "speculative states that may later be rejected.")
doc.add_paragraph("译文：验证期间，持久循环检查点不得被之后可能被拒绝的投机状态覆盖。")

quote_cn("原文：After rejection sampling, the commit stage reconstructs the recurrent state "
         "corresponding to the actual committed token prefix.")
doc.add_paragraph("译文：拒绝采样之后，提交阶段重建与实际提交 token 前缀对应的循环状态。")

doc.add_paragraph("原文要求支持的场景：")
for t in [
    "Partial acceptance and full acceptance.（部分接受与全部接受）",
    "Immediate rejection of the first draft token.（首个 draft token 立即被拒绝）",
    "Multiple consecutive speculative decoding steps.（连续多步投机解码）",
    "Multiple requests with different accepted-token counts.（多个请求、接受数不同）",
    "Correct convolution-state restoration.（卷积状态正确恢复）",
]:
    doc.add_paragraph(t, style="List Bullet")

doc.add_heading("1.2 §2 实现 GDN 专用状态恢复", level=2)
quote_cn("原文：We propose implementing dedicated GDN verification and commit kernels while reusing "
         "the common RecoverSSM lifecycle.")
doc.add_paragraph("译文：在复用公共 RecoverSSM 生命周期的前提下，实现 GDN 专用的验证与提交 kernel。")
quote_cn("原文：The recovery-record representation and reconstruction algorithm should be designed "
         "according to the GDN recurrence and validated against sequential GDN execution.")
doc.add_paragraph("译文：恢复记录的表示与重建算法应依据 GDN 递推设计，并对照“顺序 GDN 执行”进行验证。")
doc.add_paragraph("§2 的其余要求：")
for t in [
    "Preserve the original GDN recurrent-state checkpoint during verification.（验证期间保护原始检查点）",
    "Record the intermediate information required for state reconstruction.（记录重建所需的中间信息）",
    "Reconstruct the state corresponding to the accepted token prefix.（重建被接受前缀对应的状态）",
    "Support GDN-specific state layouts and numerical precision.（支持 GDN 专用布局与数值精度）",
    "Compact the convolution history after sampling.（采样后压缩卷积历史）",
]:
    doc.add_paragraph(t, style="List Bullet")

doc.add_heading("1.3 §3 把恢复记录移出 KV Cache", level=2)
quote_cn("原文：Recovery records are temporary data used between verification and commit. "
         "They should not be stored in persistent Mamba/SSM cache pages. Instead, allocate per-layer "
         "scratch buffers indexed by speculative-batch request rows.")
doc.add_paragraph("译文：恢复记录是验证与提交之间使用的临时数据，不应存放在持久的 Mamba/SSM 缓存页中；"
                  "应改为分配按投机批请求行索引的每层 scratch 缓冲。")
doc.add_paragraph("原文列出的预期收益：")
for t in [
    "Avoid additional persistent cache allocation for speculative recovery records."
    "（避免为恢复记录额外分配持久缓存）",
    "Reduce cache-page size and unnecessary KV transfer overhead.（减小缓存页并避免不必要的 KV 传输开销）",
    "Preserve compatibility with existing cache layouts.（保持与现有缓存布局兼容）",
    "Support recovery-context reconstruction when KV caches are rebound.（支持 KV 缓存重绑时重建恢复上下文）",
]:
    doc.add_paragraph(t, style="List Bullet")
quote_cn("原文：The scratch buffers must correctly handle request remapping, inactive requests, "
         "and graph-padding rows.")
doc.add_paragraph("译文：scratch 缓冲必须正确处理请求重映射、非活跃请求以及图填充行。")

doc.add_heading("1.4 §4 接入 Ascend MRV2", level=2)
quote_cn("原文：Initially, focus on the MRV2 eager execution path. NPU Graph replay, prefix caching, "
         "KV transfer, and additional parallel configurations can be enabled after independent "
         "correctness validation.")
doc.add_paragraph("译文：初期聚焦 MRV2 的 eager 执行路径；NPU Graph replay、前缀缓存、KV 传输以及更多并行配置，"
                  "可在独立的正确性验证通过之后再启用。")
quote_cn("原文：The first milestone focuses on functional correctness. NPU kernel optimization and "
         "broader serving compatibility will follow.")
doc.add_paragraph("译文：第一个里程碑聚焦功能正确性；NPU kernel 优化与更广的服务兼容性随后进行。")

doc.add_heading("1.5 验收计划（Validation Plan）", level=2)
doc.add_paragraph("原文要求的正确性场景：")
for t in [
    "Different speculative lengths.（不同投机长度）",
    "Partial acceptance, full acceptance, and immediate rejection.（部分接受、全部接受、立即拒绝）",
    "Multiple consecutive verify/commit steps.（连续多步验证/提交）",
    "Multiple requests with different accepted-token counts.（多请求不同接受数）",
    "Request reordering and remapping.（请求重排与重映射）",
    "Inactive requests and graph-padding rows.（非活跃请求与图填充行）",
    "Recurrent-state and convolution-state consistency.（循环状态与卷积状态一致性）",
    "FP32/BF16 numerical correctness.（FP32/BF16 数值正确性）",
    "NPU Graph replay.（NPU 图重放）",
]:
    doc.add_paragraph(t, style="List Bullet")
quote_cn("原文：The reconstructed recurrent state should match the corresponding sequential GDN state "
         "within a defined numerical tolerance.")
doc.add_paragraph("译文：重建的循环状态应在既定数值容差内与对应的顺序 GDN 状态一致。")
doc.add_paragraph("原文要求的性能指标：持久 SSM cache 占用、总分配 cache 内存、verify 延迟、commit 延迟、"
                  "TPOT、吞吐、接受长度（用于检测数值或功能退化）。")

# ===========================================================================
doc.add_heading("2. 我们的实现", level=1)

doc.add_heading("2.1 总览：要求 → 实现", level=2)
table([
    ["RFC 要求", "我们的实现", "状态"],
    ["复用 RecoverSSM + MRV2 采样后提交生命周期",
     "插件包装 RecoverSSMState.commit_step；GDN metadata 返回 None 并由自身完成 align 簿记",
     "已完成"],
    ["验证期间检查点不被覆盖",
     "verify kernel 对 checkpoint 只读，投机信息写入独立记录",
     "已完成"],
    ["部分/全接受、首个即拒、连续多步、多请求不同接受数",
     "commit 逐行按接受长度折叠；自检覆盖 accepted=0..3",
     "已完成"],
    ["卷积状态正确恢复",
     "commit 时按接受长度平移卷积窗口（compact kernel）",
     "已完成"],
    ["GDN 专用 verify/commit kernel",
     "4 个 Triton kernel：verify / plan / compact / commit",
     "已完成"],
    ["对照顺序 GDN 验证",
     "启动自检：自造 fp32/bf16 张量，比对 verify+commit 与逐 token 递推",
     "已完成"],
    ["恢复记录移出 KV cache，按请求行索引",
     "记录改为每层 scratch [rows, HV, Q, V+K+1]，page 布局与原生一致",
     "已完成"],
    ["正确处理重映射/非活跃/padding 行",
     "null 行跳过；记录容量以 num_record_rows 掩码保护；行映射在 host 侧解析",
     "已完成"],
    ["复用 --use-replayssm 配置",
     "配置校验与启动横幅",
     "已完成"],
    ["第一里程碑：MRV2 eager",
     "eager 模式跑通并完成容量测量",
     "已完成"],
    ["NPU Graph replay（后续里程碑）",
     "捕获后对全部 34 张图各执行一次，消除首次建队列失败",
     "额外完成"],
    ["FP32/BF16 数值正确性",
     "自检两条腿：fp32 与 bf16 checkpoint 各 8 个 case",
     "已完成"],
], widths=[5.6, 8.0, 2.2])

doc.add_heading("2.2 逐项实现与日志证据", level=2)

doc.add_heading("（1）数值正确性：verify + commit 对照逐 token 递推", level=3)
doc.add_paragraph("自检在服务启动前用私有张量驱动真实 kernel，不接触线上状态，也不阻塞服务。")
code("Ascend GDN RecoverSSM selftest SUMMARY: cases=8 pass=8 fail=0\n"
     "  worst_max_rel={'float32': '6.944e-08', 'bfloat16': '1.665e-03'}\n"
     "  tolerance fp32=1.0e-05 bf16=5.0e-02 OK")
doc.add_paragraph("fp32 的 6.944e-08 已是 fp32 机器精度量级（2^-24 ≈ 5.96e-8），即重建结果与逐 token 递推"
                  "在 fp32 最低位上一致。")

doc.add_heading("（2）卷积历史压缩", level=3)
doc.add_paragraph("压缩是把窗口平移到“最后一个被接受 token”的位置，属纯搬运，期望误差为 0。")
code("Ascend GDN RecoverSSM selftest CONV SUMMARY: cases=8 pass=8 fail=0\n"
     "  worst_max_rel=0.000e+00 tolerance fp32=1.0e-05 bf16=5.0e-02 OK")

doc.add_heading("（3）记录移出 KV Cache（RFC §3）", level=3)
doc.add_paragraph("缓存页恢复为原生布局（conv + checkpoint）；记录改为按请求行索引的 scratch。")
code("Ascend GDN RecoverSSM commit context: 16 layers, spec_query_len=4,\n"
     "  state tensors per layer=2, replay shape=(128, 12, 4, 257), plan rows=512\n"
     "Ascend GDN RecoverSSM KV budget: records 289.1 MiB\n"
     "  (rows=128, layers=48, heads=12, window=4, width=257, fp32, per-request\n"
     "   temporaries reused every step) + graph headroom 0 MiB\n"
     "  = 289.1 MiB deducted from available KV memory")
doc.add_paragraph("记录容量以 max_num_seqs（128 行）为界：图填充行的 state index 为 null，kernel 本就跳过，"
                  "因此无需按填充批（512 行）付费，节省约 0.87 GiB/卡。")

doc.add_heading("（4）服务期提交持续推进", level=3)
code("Ascend GDN RecoverSSM commit: step=3000 rows=4 mode=align\n"
     "  (checkpoint advanced by the GDN commit kernel)")

doc.add_heading("（5）图模式首次执行（507899 的根因修复）", level=3)
doc.add_paragraph("CANN 的 BuildSqCq 在每张图首次执行时才惰性建立队列并一次性灌入全部任务；"
                  "异步调度下该次执行与下一批提交重叠，导致驱动拒绝（507899）。"
                  "现在在服务开始前，用各图捕获时的 metadata 走正常路径各执行一次。")
code("Ascend full-graph warmup: executed 34 captured graphs before serving\n"
     "  (0 without recorded metadata); queue construction is done\n"
     "全程无 \"retrying once\"")

doc.add_heading("（6）配置事实一次性可见", level=3)
code("Ascend GDN RecoverSSM: async_scheduling=True (raw=True) scheduler_cls=AsyncScheduler\n"
     "Ascend GDN RecoverSSM: cudagraph_mode=FULL_DECODE_ONLY\n"
     "Ascend GDN RecoverSSM: prefix_caching=True mamba_cache_mode=align records=row-indexed")

doc.add_heading("（7）一键取证", level=3)
doc.add_paragraph("仓库提供 tools/collect_evidence.sh，一次运行汇总上述全部检查项，"
                  "缺行显示 NOT FOUND，不写文件。")

# ===========================================================================
doc.add_heading("3. 性能现状", level=1)

doc.add_heading("3.1 实测对比", level=2)
doc.add_paragraph("【此处填入实测表格】建议包含：并发、整批耗时、输出吞吐、E2EL、TTFT、E2EL−TTFT、TPOT、"
                  "接受长度、NPU KV cache size（tokens）。")
table([
    ["指标", "Base（无 RecoverSSM）", "Replay", "变化"],
    ["最大并发（eager）", "20", "25", "+25%"],
    ["整批 Benchmark Duration", "742.4 s", "840.9 s", "+13.3%"],
    ["输出吞吐", "165.5 token/s", "146.1 token/s", "−11.7%"],
    ["平均 E2EL", "186.8 s", "212.0 s", "+25.3 s"],
    ["平均 TTFT", "84.2 s", "90.5 s", "+6.3 s"],
    ["E2EL − TTFT", "102.5 s", "121.5 s", "+19.0 s（占增量 75%）"],
    ["TPOT", "100.2 ms", "118.8 ms", "+18.6%"],
    ["请求数 / 总输入 / 总输出", "120 / 2,827,697 / 122,880", "相同", "工作量可比"],
], widths=[4.6, 5.0, 4.0, 3.4])
doc.add_paragraph("注：该组数据采集时全局开启了 --enforce-eager，因此是 eager 路径的对比；"
                  "图模式的性能需另测。")

doc.add_heading("3.2 内存模型（每卡，MiB/GiB 按 1024 计）", level=2)
doc.add_paragraph("模型参数：64 层，其中 48 个 GDN、16 个 full attention；GDN 全局 key heads=16、"
                  "value heads=48、K=V=128、conv kernel=4；full attention KV heads=4、head dim=256；"
                  "MTP 附加 1 层。TP4 后每卡 GDN value heads=12、key heads=4。")
table([
    ["项", "计算", "数值"],
    ["一份全层 SSM checkpoint", "48 × 12 × 128 × 128 × 2 B", "18 MiB"],
    ["一份全层 conv", "每层 2560 通道 × 6 列 × 2 B", "1.40625 MiB"],
    ["一份 conv+SSM slot", "18 + 1.40625", "19.40625 MiB"],
    ["align 常驻 slot 数（MTP3）", "Base 5 → Recover 2", "少 3 份"],
    ["纯 SSM 节省 / 请求", "3 × 18", "54 MiB"],
    ["conv+SSM 有效节省 / 请求", "3 × 19.40625", "58.21875 MiB"],
    ["每行 FP32 记录", "48 × 12 × 4 × 257 × 4 B", "2.258789 MiB"],
    ["记录（128 行）固定成本", "128 × 2.258789", "289.125 MiB"],
], widths=[5.4, 6.4, 3.6])
doc.add_paragraph("结论：records 虽已移出 KV page，但仍占内存，且按 128 行固定预留，不随活跃请求数缩减。"
                  "当并发 B=21 时，净节省约为 21 × 58.21875 − 289.125 ≈ 933.47 MiB/卡。")

doc.add_heading("3.3 容量情景（模型推算，非实测保证）", level=2)
doc.add_paragraph("设可分给共享池的预算为 M、记录容量 R、每行记录 r、额外图预算 G；"
                  "每请求 attention 成本 K(L)，常驻 state 成本分别为 s_B 与 s_R：")
code("B_base    ≤ floor( M              / (K(L) + s_B) )\n"
     "B_recover ≤ floor( (M − R·r − G)  / (K(L) + s_R) )")
doc.add_paragraph("按本负载的有效 attention KV（约 16–17 KiB/token/卡）与上述 slot 成本推算：")
table([
    ["输入 token", "Base 上限", "Recover（G=0）", "Recover（G=2 GiB）"],
    ["1,024", "72", "128（触及配置上限）", "111"],
    ["4,096", "55", "84", "69"],
    ["8,192", "41", "56", "46"],
    ["16,384", "28", "33", "27"],
    ["23,564", "21", "24", "20"],
    ["32,768", "17", "18", "15"],
    ["65,536", "9", "9", "8"],
    ["131,072", "5", "5", "4"],
], widths=[3.6, 3.4, 4.4, 4.4])
doc.add_paragraph("实测 eager 下 20 → 25，与本表 23,564 token 一行的 21 → 24 吻合。"
                  "图中的 G 是图模式为捕获额外预留的余量，eager 下为 0，这正是容量收益在 eager 才显现的原因。")

doc.add_heading("3.4 什么场景下能取得收益", level=2)
table([
    ["维度", "收益变大", "原因"],
    ["上下文长度", "越短越好", "state 占用是每请求固定值，attention KV 随上下文线性增长；"
                              "8k 时理论上限 +16.7%，4k 时 +30%，32k 仅 +4.5%"],
    ["并行度/并发", "越高越好", "每请求固定省 58.2 MiB，记录成本 289 MiB 是固定项，"
                                "并发达 21 以上才明显摊薄"],
    ["模式", "eager 优于图模式", "图模式需为捕获额外预留约 2 GiB，会吞掉全部收益"],
    ["模型结构", "GDN 层数多、state 大越好", "节省量与 48 层 × 12 头 × 128×128 成正比"],
    ["投机长度", "Q 越大越好", "Base 每步写 Q+1 份状态的代价随 Q 增长，Recover 恒定写 1 份"],
], widths=[3.2, 3.6, 8.6])

doc.add_heading("3.5 每步开销审计（估算，单卡）", level=2)
table([
    ["#", "位置", "内容", "估算", "可否消除"],
    ["1", "commit 的 state kernel", "读+写整份 checkpoint ≈ 18.9 MB × 2", "150–400 µs", "否（设计决定）"],
    ["2", "align 簿记（staging + apply）", "约 12 个微小设备算子", "60–120 µs", "是（可折进 plan kernel）"],
    ["3", "context key 每步重建", "16 层 × 2 张量 × 4 属性", "20–40 µs", "是（廉价守卫）"],
    ["4", "spec 行映射", "原为 device nonzero（每步同步 × 3 组）", "已修复", "已消除"],
    ["5", "accepted buffer 填充", "每步一次 fill_", "5–10 µs", "是（可预填）"],
    ["6", "日志字符串", "每步格式化", "5–10 µs", "是（惰性化）"],
], widths=[0.9, 4.2, 5.6, 2.6, 3.0])
doc.add_paragraph("注：第 4 项为已定位并修复的每步设备同步 —— 原实现在 build() 中对 device 上的 "
                  "spec_sequence_masks 调用 nonzero，其输出形状依赖数据，必然触发流同步；"
                  "原生 builder 专门维护 pinned CPU 副本以规避该同步，插件此前未使用。"
                  "该项是 TPOT 退化 18.6% 的首要候选，修后待复测。")

doc.add_heading("3.6 结构性差异：我们与 Base 的根本不同", level=2)
code("Base ：状态在 verify kernel 内就地更新（图内，可与其它算子重叠）\n"
     "Replay：verify 只写记录；采样后【图外】再跑一遍 commit")
doc.add_paragraph("即 RecoverSSM 用“一次额外串行 pass（读 1 份 + 写 1 份 checkpoint）”换取“state 内存下降约 60%”。"
                  "Q=4 时 Base 每步写 4 份投机状态（约 75.6 MiB），Replay 写 1 份（18.9 MiB）并读 1 份——"
                  "流量上并不更差，但写入发生在采样之后，无法与前向重叠。")

doc.add_heading("3.7 下一步", level=2)
table([
    ["优先级", "动作", "目的"],
    ["P0", "同一并发（固定 20）下 A/B 对比 TPOT 与吞吐", "确认是否存在固有开销，排除批大小效应"],
    ["P0", "复测本次同步修复后的 TPOT", "验证 18.6% 退化是否主要来自该同步"],
    ["P1", "8k / 32k 各跑一次并发对比", "验证 3.3 的容量曲线"],
    ["P1", "align 簿记合并进 plan kernel", "再省约 100 µs/步"],
    ["P2", "verify/commit 的 V tile 由 32/16 评估为 64", "减少循环与重复读取，需编译与设备验证"],
    ["P2", "三个 group 的 commit 合并或单独捕获成图", "减少每轮图外发射次数"],
], widths=[1.8, 8.0, 6.4])

# ===========================================================================
doc.add_heading("附录：环境与复现", level=1)
table([
    ["项", "版本 / 值"],
    ["vLLM", "0.30.0 (ced6857afa0ea7b2e3f0846a62e1394e90f15607)"],
    ["vllm-ascend", "v0.30.0rc1 (5f828ad638ee4f9b12db87f9d5481f29076d37f9)"],
    ["CANN / torch-npu / triton-ascend", "9.1.0 / 2.10.0.post4 / 3.2.2"],
    ["硬件", "A2（Ascend910B4）× 4，TP4"],
    ["模型", "Qwen3.8-27B，MTP3（Q=4），mamba_cache_mode=align，prefix caching"],
    ["自检开关", "GDN_RECOVERSSM_SELFTEST=1"],
    ["图池余量", "GDN_RECOVERSSM_KV_MARGIN_MB（默认 2048，eager 下自动为 0）"],
    ["一键取证", "tools/collect_evidence.sh logs/vllm.log"],
], widths=[5.4, 9.4])

doc.save(OUT)
print("saved", OUT)
