#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a vLLM server log to prove RecoverSSM is actually running.

Usage::

    python plugin/tools/check_startup.py server.log
    python plugin/tools/check_startup.py -            # read stdin
    grep -E "RecoverSSM|gdn_recoverssm" server.log | python plugin/tools/check_startup.py -

Stdlib only, runs anywhere. Exit code 0 = startup checks passed.
"""

from __future__ import annotations

import re
import sys

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[2m",
    "\033[0m",
)

# (id, 说明, 缺失时的动作)
CHECKS = [
    (
        "plugin_loaded",
        "插件被 vLLM 加载",
        "确认 pip install -e 装的是 vLLM 所在的同一个 python 环境；"
        "python -c \"from importlib.metadata import entry_points as e;"
        "print([x.name for x in e(group='vllm.general_plugins')])\"",
    ),
    (
        "config_accepted",
        "配置校验通过（--use-replayssm 对 Qwen GDN 生效）",
        "检查是否设了 GDN_RECOVERSSM_DISABLE=1；或架构不在 Qwen3_5* 列表里",
    ),
    (
        "commit_context",
        "commit 上下文建立，且记录张量已分配（3 个 state/layer）",
        "说明还没走投机路径：先发一个请求；若请求后仍没有，检查 num_speculative_tokens>0",
    ),
    (
        "verify_running",
        "verify 真的在跑",
        "同上：需要一个带 draft 的 decode 请求",
    ),
    (
        "stats",
        "运行时统计（verify/commit 同步推进）",
        "默认每 200 次 commit 打一行；或设 GDN_RECOVERSSM_LOG_EVERY=20 加速",
    ),
]


def find(lines, pattern):
    rx = re.compile(pattern)
    for line in lines:
        m = rx.search(line)
        if m:
            return line.strip(), m
    return None, None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    text = sys.stdin.read() if path == "-" else open(path, encoding="utf-8", errors="ignore").read()
    lines = text.splitlines()
    if not lines:
        print("空日志？")
        return 2

    results = {}

    # ---------------------------------------------------------------- 1. 加载
    line, _ = find(lines, r"gdn_recoverssm_ascend: installed")
    inert, m_inert = find(lines, r"gdn_recoverssm_ascend: stays inert(?: \((.*?)\))?")
    disabled, _ = find(lines, r"gdn_recoverssm_ascend: disabled by")
    if line:
        results["plugin_loaded"] = (True, "installed")
    elif inert:
        results["plugin_loaded"] = (
            False,
            f"stays inert —— {m_inert.group(1) if m_inert and m_inert.group(1) else '原因见日志'}",
        )
    elif disabled:
        results["plugin_loaded"] = (False, "GDN_RECOVERSSM_DISABLE 已设置")
    else:
        results["plugin_loaded"] = (False, "没找到插件加载日志")

    # ------------------------------------------------------------ 2. 配置生效
    line, m = find(lines, r"Ascend GDN RecoverSSM enabled: (.*)")
    if line:
        detail = m.group(1).strip()
        results["config_accepted"] = (True, detail)
    else:
        results["config_accepted"] = (False, "没有 'enabled' 日志")

    # ------------------------------------------------------- 3. commit context
    line, m = find(lines, r"commit context: (\d+) layers, spec_query_len=(\d+), state tensors per layer=(\d+), replay shape=(\([^)]*\))")
    if line:
        layers, sql, tensors, shape = m.groups()
        ok = int(tensors) >= 3
        results["commit_context"] = (
            ok,
            f"layers={layers} window={sql} states/layer={tensors} replay={shape}"
            + ("" if ok else "  ← 只有 2 个 state，说明记录没被追加"),
        )
    else:
        results["commit_context"] = (False, "还没建立（未进入投机路径）")

    # ------------------------------------------------------------ 4. verify
    line, m = find(lines, r"verify #1: (\d+) spec rows, window=(\d+)")
    if line:
        results["verify_running"] = (True, f"rows={m.group(1)} window={m.group(2)}")
    else:
        results["verify_running"] = (False, "还没跑到 verify（先发请求）")

    # ------------------------------------------------------------- 5. stats
    line, m = find(lines, r"stats: verify_steps=(\d+) commit_steps=(\d+).*?mean_accepted=([\d.]+)")
    if line:
        vs, cs, acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if cs == 0:
            results["stats"] = (False, f"verify={vs} commit=0 ← commit 钩子没接上，检查点不会前进")
        elif cs < 0.9 * vs:
            results["stats"] = (False, f"verify={vs} commit={cs} ← 部分 verify 没有 commit")
        else:
            results["stats"] = (True, f"verify={vs} commit={cs} mean_accepted={acc}")
    else:
        results["stats"] = (None, "还没到汇报周期")

    # ---------------------------------------------------------- 负向指标
    bad_kimi, _ = find(lines, r"RecoverSSM is only supported for Kimi-K3 KDA")
    bad_v1, _ = find(lines, r"V1 runner has no post-sampling commit hook")
    ns, m_ns = find(lines, r"num_speculative_tokens['\"]?\s*:\s*(\d+)")
    num_spec = int(m_ns.group(1)) if m_ns else None
    mode, m_mode = find(lines, r"mamba_cache_mode=(\w+)")

    # ----------------------------------------------------------------- 打印
    print()
    print("  Ascend GDN RecoverSSM — 启动自检")
    print("  " + "-" * 68)
    failed = 0
    for cid, label, hint in CHECKS:
        status, detail = results[cid]
        if status is True:
            mark = f"{GREEN}[ OK ]{RESET}"
        elif status is False:
            mark = f"{RED}[FAIL]{RESET}"
            failed += 1
        else:
            mark = f"{YELLOW}[ -- ]{RESET}"
        print(f"  {mark} {label}")
        print(f"         {DIM}{detail}{RESET}")
        if status is False:
            print(f"         {YELLOW}→ {hint}{RESET}")
    print("  " + "-" * 68)

    if bad_kimi:
        print(f"  {RED}[FAIL]{RESET} 上游把 --use-replayssm 拒了（插件没生效）")
        failed += 1
    if bad_v1:
        print(f"  {RED}[FAIL]{RESET} 少了 VLLM_USE_V2_MODEL_RUNNER=1（commit 不会发生）")
        failed += 1
    if num_spec is not None:
        q = num_spec + 1
        note = ""
        if q <= 2:
            note = f"  {YELLOW}← Q={q}：状态访存收益 0.98×，验证没有意义，建议 num_speculative_tokens=3{RESET}"
        print(f"  {DIM}num_speculative_tokens={num_spec} → Q={q}{RESET}{note}")
    if m_mode:
        print(f"  {DIM}mamba_cache_mode={m_mode.group(1)}{RESET}")

    print()
    if failed == 0:
        print(f"  {GREEN}启动期全部通过{RESET}；有流量后确认 commit_steps ≈ verify_steps。")
    else:
        print(f"  {RED}{failed} 项未通过{RESET}，按上面的 → 处理。")
    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
