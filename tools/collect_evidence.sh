#!/usr/bin/env bash
# One-shot collector for a GDN RecoverSSM run.
#
#   tools/collect_evidence.sh [LOG]        (default: logs/vllm.log)
#
# Prints every fact the acceptance checklist needs, grouped, with NOT FOUND where a
# line is missing. No file is written and nothing is sent anywhere: paste the
# output into the issue.
set -uo pipefail

LOG="${1:-logs/vllm.log}"
if [[ ! -f "$LOG" ]]; then
  echo "log not found: $LOG" >&2
  exit 1
fi

have() { grep -c "$1" "$LOG" 2>/dev/null || true; }
sec()  { printf '\n===== %s =====\n' "$1"; }
show() {  # show <description> <grep-pattern> [head-count]
  local desc="$1" pat="$2" n="${3:-3}"
  local out
  out="$(grep -E "$pat" "$LOG" 2>/dev/null | head -"$n")"
  if [[ -z "$out" ]]; then
    printf '%-46s NOT FOUND\n' "$desc:"
  else
    printf '%s\n' "$out"
  fi
}

sec "0. 文件"
printf 'log=%s\nlines=%s\n' "$LOG" "$(wc -l < "$LOG" | tr -d ' ')"

sec "1. 生效配置（每个 rank 各一条，取前 4 条）"
show "banner"        "Ascend GDN RecoverSSM: enabled:" 2
show "async/scheduler" "async_scheduling=" 2
show "cudagraph"     "cudagraph_mode=" 2
show "prefix/align"  "prefix_caching=" 2

sec "2. 内存与容量"
show "KV budget"     "KV budget: records" 1
show "KV cache size" "NPU KV cache size" 1
show "commit context" "commit context:" 1
show "commit prep"   "startup preparation complete" 1

sec "3. 数值自检（SSM）"
show "SUMMARY"       "selftest SUMMARY" 1
show "失败用例"      "selftest: .* FAIL" 8

sec "4. 数值自检（conv）"
show "CONV SUMMARY"  "selftest CONV SUMMARY" 1
show "失败用例"      "selftest: conv .* FAIL" 8

sec "5. 图捕获与首次执行"
show "capture"       "full-graph capture finished" 1
show "warmup"        "full-graph warmup" 1
show "首次即失败重试" "retrying once" 3

sec "6. 服务期"
show "commit 步数(尾)" "RecoverSSM commit: step=" 3
show "接受长度(尾)"    "Mean acceptance length" 2
# 同一行还带 Per-position acceptance rate（前 3 个 draft 位置各自的命中率），
# 这是判断"精度是否让 draft 命中下降"的关键，不用另外抓。
show "吞吐(尾)"        "Engine 000: Avg prompt" 2

sec "7. 错误扫描"
for pat in 507899 "MTE accesses" "SUSPECT REMOTE" 561002 MISMATCH OutOfMemory \
           UnboundLocalError IndexError AttributeError "Traceback"; do
  printf '%-24s %s\n' "$pat" "$(have "$pat")"
done

sec "8. 结论"
ssm_ok=0; conv_ok=0
grep -q "selftest SUMMARY.*OK" "$LOG" && ssm_ok=1
grep -q "selftest CONV SUMMARY.*OK" "$LOG" && conv_ok=1
errs=$(grep -cE "507899|MTE accesses|SUSPECT REMOTE|561002|OutOfMemory" "$LOG" 2>/dev/null || true)
warm=$(grep -c "full-graph warmup: executed" "$LOG" 2>/dev/null || true)
retry=$(grep -c "retrying once" "$LOG" 2>/dev/null || true)
steps=$(grep -oE "RecoverSSM commit: step=[0-9]+" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
printf 'SSM selftest      : %s\n' "$([[ $ssm_ok == 1 ]] && echo OK || echo MISSING/FAIL)"
printf 'conv selftest     : %s\n' "$([[ $conv_ok == 1 ]] && echo OK || echo MISSING/FAIL)"
printf 'graph warmup      : %s\n' "$( [[ ${warm:-0} -gt 0 ]] && echo "$warm 条" || echo "无（eager 下正常）")"
printf 'first-exec retry  : %s\n' "${retry:-0}"
printf 'hard errors       : %s\n' "${errs:-0}"
printf 'last commit step  : %s\n' "${steps:-none}"
