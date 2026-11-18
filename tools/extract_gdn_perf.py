#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract GDN performance evidence from local vLLM and AIS Bench logs."""

import argparse
import gzip
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROCESS = re.compile(r"\(([-\w]+)\s+pid\s*=\s*(\d+)\)")
TIMESTAMP = re.compile(r"\b(?:\d{4}-)?\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?")
DECIMAL = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FIELD = re.compile(r"(?<!\S)([a-zA-Z_][\w-]*)=")
STATUS = re.compile(r"\b(PASS|FAIL|READY)\b")
PERIODIC = {"GDN_PERF", "GDN_WORKLOAD", "GDN_PATHS", "GDN_PHASE_PERF"}
STARTUP = {
    "GDN_CONFIG",
    "GDN_PHASE_PERF_CONFIG",
    "GDN_MEMORY",
    "GDN_READY",
    "GDN_RUNTIME",
    "GDN_PREPARED_CONTEXT",
    "GDN_RECORD_BUDGET",
    "GDN_ASCENDC",
    "GDN_ASCENDC_READY",
    "GDN_ASCENDC_SELFTEST",
    "GDN_GROUPED_SELFTEST",
    "GDN_MIXED_SELFTEST",
    "GDN_MIXED_ADAPTER",
}
DISPATCH = {"GDN_ASCENDC_FUSION", "GDN_MIXED", "GDN_SAMPLER"}
MARKER = re.compile(r"\b(" + "|".join(sorted(PERIODIC | STARTUP | DISPATCH)) + r")\b")
DECODER = json.JSONDecoder()
ERROR = re.compile(
    r"\b(?:507899|561002|Traceback|OutOfMemory(?:Error)?|RuntimeError|"
    r"UnboundLocalError|IndexError|AttributeError|GDN_ASCENDC_BUILD_FAILURE|"
    r"GDN_ASCENDC_LOAD_FAILURE)\b|MTE accesses|SUSPECT REMOTE|\bERROR\b"
)
ENGINE = re.compile(r"\bEngine\s+(\d+):\s+Avg prompt throughput:")
ENGINE_FIELDS = {
    "prompt_tokens_s": "Avg prompt throughput",
    "generation_tokens_s": "Avg generation throughput",
    "running": "Running",
    "waiting": "Waiting",
    "deferred": "Deferred",
    "kv_usage_pct": r"(?:NPU|GPU) KV cache usage",
    "prefix_hit_pct": "Prefix cache hit rate",
    "external_prefix_hit_pct": "External prefix cache hit rate",
    "preemptions": "Preemptions",
}
CAPACITY = {
    "available_kv_gib": r"Available KV cache memory:\s*([\d.]+) GiB",
    "kv_cache_tokens": r"(?:GPU|NPU) KV cache size:\s*([\d,]+) tokens",
    "model_load_gib": r"Model loading took\s*([\d.]+) GiB",
    "graph_capture_mib": r"Graph captur(?:e|ing) finished[^\n]*?took\s*([\d.]+) MiB",
    "graph_capture_gib": r"Graph captur(?:e|ing) finished[^\n]*?took\s*([\d.]+) GiB",
}
CLIENT_METRICS = {
    "TTFT": "ttft",
    "TPOT": "tpot",
    "ITL": "itl",
    "E2EL": "e2el",
    "InputTokens": "input_tokens",
    "OutputTokens": "output_tokens",
    "OutputTokenThroughput": "per_request_output_tokens_s",
    "Benchmark Duration": "benchmark_duration",
    "Total Requests": "total_requests",
    "Failed Requests": "failed_requests",
    "Success Requests": "success_requests",
    "Concurrency": "mean_concurrency",
    "Max Concurrency": "max_concurrency",
    "Request Throughput": "request_throughput",
    "Total Input Tokens": "total_input_tokens",
    "Total Generated Tokens": "total_generated_tokens",
    "Input Token Throughput": "input_tokens_s",
    "Output Token Throughput": "output_tokens_s",
    "Prefill Token Throughput": "prefill_tokens_s_ais",
    "Total Token Throughput": "total_tokens_s",
}


def scalar(value):
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(DECIMAL, value):
        return float(value)
    return {"true": True, "false": False, "null": None}.get(value.lower(), value)


def parse_fields(message):
    fields, end = {}, 0
    while match := FIELD.search(message, end):
        name, start = match.group(1), match.end()
        if message[start : start + 1] in ("{", "["):
            try:
                fields[name], width = DECODER.raw_decode(message[start:])
                end = start + width
                continue
            except json.JSONDecodeError:
                pass  # Legacy messages can contain Python dicts and tuples.
        next_field = FIELD.search(message, start)
        end = next_field.start() if next_field else len(message)
        value = message[start:end].strip()
        value = re.sub(r"\s+(?:PASS|FAIL|READY)$", "", value)
        fields[name] = scalar(value)
    return fields


def log_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for index, raw in enumerate(stream, 1):
            raw = raw.strip()
            wrapped_time = None
            if raw.startswith("{"):
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict) and isinstance(
                        payload.get("log"), str
                    ):
                        raw = payload["log"]
                        wrapped_time = payload.get("time")
                except ValueError:
                    pass
            line = ANSI.sub("", raw).strip()
            timestamp = TIMESTAMP.search(line)
            yield (
                line,
                {
                    "line": index,
                    "time": wrapped_time or (timestamp[0] if timestamp else None),
                },
            )


def process_identity(line):
    if match := PROCESS.search(line):
        return match.group(1), int(match.group(2))
    return "unattributed", None


def remember(tables, key, record):
    table = tables.setdefault(key, {"count": 0, "first": record, "last": record})
    table["count"] += 1
    table["last"] = record
    return table


def new_process(name, pid, rank, session):
    return {
        "name": name,
        "pid": pid,
        "rank": rank,
        "session": session,
        "startup": {},
        "periodic": {},
        "dispatch": {},
        "native": {},
        "errors": {},
    }


def collect_gdn(line, where, active, processes):
    marker = MARKER.search(line)
    if not marker:
        return None
    key = marker[1]
    tail = line[marker.end() :].lstrip()
    fields = parse_fields(tail)
    name, pid = process_identity(line)
    # The JSON metrics carry their own identity even if log forwarding drops the prefix.
    payload = fields.get("json")
    if isinstance(payload, dict):
        pid = payload.get("pid", pid)
    rank = (payload if isinstance(payload, dict) else fields).get("rank")
    identity = (pid, None) if pid is not None else (name, str(rank))
    process = active.get(identity)
    reset = False
    if process:
        earlier = process["periodic"].get(key, {}).get("last", {}).get("fields", {})
        earlier = earlier.get("json", earlier)
        current = payload if isinstance(payload, dict) else fields
        old_steps, steps = earlier.get("steps"), current.get("steps")
        reset = (
            isinstance(steps, int) and isinstance(old_steps, int) and steps < old_steps
        )
        if process["periodic"]:
            ready = last_fields(process, "GDN_READY", "startup")
            if key == "GDN_CONFIG" and "patch" in fields:
                reset = True
            elif key == "GDN_MEMORY" and current.get("phase") == "kv_bound_before_prepare":
                # A new worker's memory snapshots are emitted before READY.
                reset = True
            elif key == "GDN_READY" and current.get("ready_unix_s") is not None:
                reset = current["ready_unix_s"] != ready.get("ready_unix_s")
    if process is None or reset:
        process = new_process(name, pid, rank, process["session"] + 1 if process else 1)
        active[identity] = process
        processes.append(process)
    elif rank is not None:
        process["rank"] = rank
    if name != "unattributed":
        process["name"] = name
    if tail.startswith("json=") and not isinstance(payload, dict):
        remember(process["errors"], "invalid_json:" + key, where)
        return process
    # Paths to local libraries add no evidence and can contain usernames.
    fields.pop("library", None)
    status = STATUS.search(tail.split("=", 1)[0]) or re.search(
        r"\s(PASS|FAIL|READY)$", tail
    )
    if status:
        fields["status"] = status[1]
    if fields.get("status") == "FAIL":
        remember(process["errors"], key + "_FAIL", where)
    if key == "GDN_ASCENDC" and tail.startswith("loaded "):
        fields["status"] = "loaded"
    if (
        key in STARTUP
        and key != "GDN_CONFIG"
        and re.search(r"\b(?:failed|failure):", tail, re.IGNORECASE)
    ):
        remember(process["errors"], key + "_FAILURE", where)
    table = process[
        "periodic" if key in PERIODIC else "startup" if key in STARTUP else "dispatch"
    ]
    if key == "GDN_CONFIG" and tail.startswith("native normalization "):
        key = "GDN_CONFIG_NATIVE_NORMALIZATION"
    if key == "GDN_SAMPLER" and tail.startswith("ready "):
        table = process["startup"]
    record = {**where, "fields": fields}
    series = remember(table, key, record)
    if (
        key == "GDN_MEMORY"
        and isinstance(payload, dict)
        and isinstance(payload.get("phase"), str)
    ):
        series.setdefault("by_phase", {})[payload["phase"]] = record
    return process


def scan_native(line, where, native):
    for key, pattern in CAPACITY.items():
        if match := re.search(pattern, line):
            remember(native, key, {**where, "value": scalar(match[1].replace(",", ""))})
    if match := re.search(
        r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", line
    ):
        remember(
            native,
            "native_concurrency_estimate",
            {
                **where,
                "tokens_per_request": int(match[1].replace(",", "")),
                "max_concurrency": float(match[2]),
            },
        )
    if match := re.search(
        r"total_num_cumulative_preemption\s*[=:]\s*(\d+)", line, re.IGNORECASE
    ):
        remember(
            native, "cumulative_preemption_report", {**where, "value": int(match[1])}
        )
    if match := re.search(
        r"KV budget: records ([\d.]+) MiB .*?headroom ([\d.]+) MiB", line
    ):
        remember(
            native,
            "gdn_kv_reservation",
            {**where, "records_mib": float(match[1]), "headroom_mib": float(match[2])},
        )
    if engine := ENGINE.search(line):
        values = {}
        tail = line[engine.start() :]
        for key, label in ENGINE_FIELDS.items():
            if match := re.search(r"(?:^|[:,])\s*" + label + rf":\s*({DECIMAL})", tail):
                values[key] = scalar(match[1])
        table = remember(native, "engine_" + engine[1], {**where, **values})
        peak = table.setdefault("peak", {})
        for key, value in values.items():
            peak[key] = max(peak.get(key, value), value)
        # These native log entries are interval counts when emitted.
        if "preemptions" in values:
            table["preemptions_reported_sum"] = (
                table.get("preemptions_reported_sum", 0) + values["preemptions"]
            )
            table["preemptions_reported_intervals"] = (
                table.get("preemptions_reported_intervals", 0) + 1
            )
    if "Mean acceptance length:" in line:
        pairs = {
            "mean_acceptance_length": "Mean acceptance length",
            "accepted": "Accepted",
            "drafted": "Drafted",
            "draft_acceptance_pct": "Avg Draft acceptance rate",
        }
        values = {}
        for key, label in pairs.items():
            if match := re.search(label + rf":\s*({DECIMAL})", line):
                values[key] = scalar(match[1])
        if match := re.search(r"Per-position acceptance rate: ([\d.,\s]+)", line):
            values["per_position"] = [float(n) for n in re.findall(DECIMAL, match[1])]
        table = remember(native, "spec_decoding_intervals", {**where, **values})
        for key in ("accepted", "drafted"):
            if key in values:
                total = key + "_reported_sum"
                table[total] = table.get(total, 0) + values[key]
    if "KV Transfer metrics:" in line:
        tail = line.split("KV Transfer metrics:", 1)[1].strip()
        # Connector-specific field labels can contain spaces and units.
        entries = re.split(r",\s*(?=[^,=]+=)", tail)
        values = {
            k.strip(): scalar(v.strip())
            for k, _, v in (s.partition("=") for s in entries)
            if k and v
        }
        remember(native, "kv_transfer_intervals", {**where, "fields": values})
    if "Application startup complete." in line:
        remember(native, "api_startup_complete", where)
    if match := re.search(
        r"Initializing a V1 LLM engine \(([^)]+)\) with config:", line
    ):
        tail = line[match.end() :]
        fields = {"vllm_version": match[1]}
        for key in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "dtype",
            "max_seq_len",
            "enforce_eager",
            "enable_prefix_caching",
            "enable_chunked_prefill",
        ):
            if value := re.search(rf"\b{key}=([^, ]+)", tail):
                fields[key] = scalar(value[1])
        remember(native, "engine_config", {**where, "fields": fields})


def read_server(label, path):
    active, processes = {}, []
    for line, where in log_lines(path):
        gdn = collect_gdn(line, where, active, processes)
        name, pid = process_identity(line)
        identity = (pid, None) if pid is not None else (name, "None")
        process = gdn or active.get(identity)
        native = process["native"] if process else {}
        scan_native(line, where, native)
        error = ERROR.search(line)
        if not process and (native or error):
            process = new_process(name, pid, None, 1)
            active[identity] = process
            processes.append(process)
            process["native"] = native
        if error:
            remember(process["errors"], error[0], where)
    return {"label": label, "file": path.name, "processes": processes}


def number_and_unit(cell):
    if match := re.fullmatch(rf"({DECIMAL})\s*(.*)", cell):
        return scalar(match[1]), match[2] or None
    return None, None


def read_client(label, path):
    runs, run = [], None
    for line, where in log_lines(path):
        begins = "run with concurrency:" in line or "Starting progress bar" in line
        if begins and run and (run["metrics"] or "api_task_end" in run):
            run = None
        if run is None:
            run = {"metrics": {}}
            runs.append(run)
        if match := re.search(r"run with concurrency:\s*(\d+)", line):
            run["offered_concurrency"] = int(match[1])
        if "Starting progress bar" in line:
            run["progress_start"] = where
        if match := re.search(rf"Api infer task time elapsed:\s*({DECIMAL})s", line):
            run["api_task_elapsed_s"] = float(match[1])
            run["api_task_end"] = where
        if "Inference tasks completed." in line:
            run["inference_tasks_completed"] = where
        edge = "│" if "│" in line else "|"
        cells = [cell.strip() for cell in line.strip(edge).split(edge)]
        if len(cells) < 3 or cells[0] not in CLIENT_METRICS:
            continue
        metric, stage = CLIENT_METRICS[cells[0]], cells[1].lower()
        value, unit = number_and_unit(cells[2])
        if value is None:
            continue
        # A concatenated bare AIS table can lack lifecycle markers. Keep the tables
        # separate instead of taking metrics from different benchmark runs.
        if metric in run["metrics"] and stage in run["metrics"][metric]:
            run = {"metrics": {}}
            runs.append(run)
        record = {**where, "value": value, "unit": unit}
        if len(cells) >= 10:
            for key, cell in zip(
                ("min", "max", "median", "p75", "p90", "p99", "samples"), cells[3:]
            ):
                record[key] = number_and_unit(cell)[0]
        run["metrics"].setdefault(metric, {})[stage] = record
    return {
        "label": label,
        "file": path.name,
        "runs": [r for r in runs if r != {"metrics": {}}],
    }


def last_fields(process, marker, table="periodic"):
    fields = process[table].get(marker, {}).get("last", {}).get("fields", {})
    return fields.get("json", fields)


def display_fields(values, keys):
    if not isinstance(values, dict):
        return ""
    return " ".join(f"{key}={values[key]}" for key in keys if key in values)


def print_phase(phase):
    print(
        "    阶段（worker 累计）",
        display_fields(
            phase, ("scope", "first_submit_unix_s", "last_completed_unix_s")
        ),
    )
    for key, values in phase.get("phases", {}).items():
        known_context = values.get("context_known_rows", 0)
        mean = (
            values.get("context_tokens_sum", 0) / known_context
            if known_context
            else None
        )
        print(
            "      " + key,
            display_fields(
                values,
                (
                    "steps",
                    "target_tokens",
                    "prefill_target_tokens",
                    "decode_target_tokens",
                    "target_split_known_steps",
                    "spec_rows",
                    "spec_target_tokens",
                    "spec_target_known_rows",
                    "spec_emitted_tokens",
                    "context_tokens_max",
                    "context_known_rows",
                    "attention_work_proxy",
                ),
            ),
            f"mean_context={mean:.1f}" if mean is not None else "",
        )
    timings = phase.get("host_timings", {})
    if timings:
        print("    Host 分段（累计，完整字段见 JSON）")
        for stage, phases in timings.items():
            for name, values in phases.items():
                print(
                    f"      {stage}/{name}",
                    display_fields(
                        values,
                        (
                            "samples",
                            "sampled_steps",
                            "mean_us",
                            "max_us",
                            "kernel_submissions",
                        ),
                    ),
                )
    if paths := phase.get("prefill_paths", {}):
        print("    Native prefill 实际分支（逐层 Host 调用）")
        for name, branches in paths.items():
            for branch, values in branches.items():
                print(
                    f"      {name}/{branch}",
                    display_fields(
                        values, ("calls", "layer_target_tokens", "token_known_calls")
                    ),
                )
    if graph := phase.get("graph_submissions", {}):
        print("    Graph 提交", json.dumps(graph, ensure_ascii=False))


def print_server(work):
    print(f"\n服务端 [{work['label']}] {work['file']}")
    if not work["processes"]:
        print("  未识别到启动、运行或错误指标。")
    for p in work["processes"]:
        label = f"{p['name']} pid={p['pid']} rank={p['rank']} 会话={p['session']}"
        print("  " + label)
        ready_record = last_fields(p, "GDN_READY", "startup")
        config = (
            (ready_record if "parameters" in ready_record else None)
            or last_fields(p, "GDN_PHASE_PERF_CONFIG", "startup")
            or last_fields(p, "GDN_CONFIG", "startup")
            or ready_record
        )
        ready = bool(ready_record)
        if config or ready:
            manifest = config.get("parameters", {})
            print(
                "    启动",
                display_fields(
                    config,
                    (
                        "patch",
                        "source_fingerprint",
                        "launch_fingerprint",
                        "fingerprint",
                    ),
                ),
                f"GDN_READY={'有' if ready else '未见'}",
            )
            if isinstance(manifest, dict):
                print(
                    "    配置",
                    display_fields(
                        manifest,
                        (
                            "tp",
                            "num_speculative_tokens",
                            "graph_mode",
                            "async_scheduling",
                            "max_num_seqs",
                            "max_num_batched_tokens",
                            "max_model_len",
                            "dtype",
                            "mamba_ssm_cache_dtype",
                            "prefix_caching",
                            "kv_transfer_role",
                            "kv_connector",
                        ),
                    ),
                )
        startup = ready_record.get("startup", {})
        legacy_context = last_fields(p, "GDN_PREPARED_CONTEXT", "startup")
        contexts = startup.get("contexts") or ([legacy_context] if legacy_context else [])
        for ctx in contexts:
            print(
                "    State",
                display_fields(
                    ctx,
                    (
                        "kv_group",
                        "layers",
                        "state_shape",
                        "state_stride",
                        "state_dtype",
                        "actual_record_bytes",
                        "record_rows",
                    ),
                ),
            )
        memories = p["startup"].get("GDN_MEMORY", {}).get("by_phase", {})
        for stage, record in memories.items():
            memory = record["fields"]["json"]
            allocator = memory.get("allocator_bytes", {})
            print(
                f"    启动显存/{stage}",
                display_fields(memory, ("kv_blocks", "kv_plan_tensor_bytes")),
                display_fields(
                    allocator,
                    (
                        "allocated_bytes.all.current",
                        "allocated_bytes.all.peak",
                        "reserved_bytes.all.current",
                        "reserved_bytes.all.peak",
                    ),
                ),
            )
        phase = last_fields(p, "GDN_PHASE_PERF")
        totals = phase.get("totals")
        perf = (
            {"steps": phase.get("steps"), "emitted_tokens": phase.get("emitted_tokens"),
             "worker_observed_s": phase.get("worker_window_s"), **totals}
            if isinstance(totals, dict)
            else last_fields(p, "GDN_PERF")
        )
        if perf:
            print(
                "    Worker 累计",
                display_fields(
                    perf,
                    (
                        "steps",
                        "emitted_tokens",
                        "spec_rows",
                        "spec_emitted_per_row",
                        "worker_observed_s",
                        "commits",
                        "host_submit_us_avg",
                        "launches_per_commit",
                        "graph_padding_fraction",
                        "target_tokens",
                        "mean_rows",
                        "peak_rows",
                    ),
                ),
            )
        if phase:
            print_phase(phase)
        elif perf:
            batch = last_fields(p, "GDN_WORKLOAD")
            if batch:
                steps = {}
                histogram = batch.get("batch_hist")
                if isinstance(histogram, list):
                    for row in histogram:
                        name = row["phase"]
                        steps[name] = steps.get(name, 0) + row["steps"]
                print(
                    "    旧版负载",
                    display_fields(
                        batch,
                        (
                            "target_tokens",
                            "mean_rows",
                            "peak_rows",
                            "mean_computed_context",
                        ),
                    ),
                    f"phase_steps={json.dumps(steps)}",
                )
        current_paths = phase.get("paths")
        if paths := current_paths or last_fields(p, "GDN_PATHS"):
            print(
                "    verify Host 分派（READY 后，图 replay 不走 Python）"
                if current_paths
                else "    verify Host 分派（旧计数包括启动，图 replay 不走 Python）",
                display_fields(
                    paths.get("counts", {}),
                    (
                        "verify",
                        "verify_fused_mixed_model_calls",
                        "verify_fused_pure_spec_model_calls",
                    ),
                ),
            )
        if sampler := (
            phase.get("sampler")
            or startup.get("sampler")
            or last_fields(p, "GDN_SAMPLER", "dispatch")
        ):
            print(
                "    Sampling",
                display_fields(
                    sampler,
                    ("rows", "vocab", "row_tile", "sort", "workspace_model_bytes"),
                ),
            )
        for key, table in p["native"].items():
            if key.startswith("engine_") and key[len("engine_"):].isdigit():
                print(
                    "    " + key,
                    "已见区间峰值",
                    display_fields(
                        table["peak"], ("running", "waiting", "kv_usage_pct")
                    ),
                    display_fields(table, ("preemptions_reported_sum", "preemptions_reported_intervals")),
                    f"区间数={table['count']}",
                )
            elif key == "engine_config":
                print("    原生启动", display_fields(
                    table["last"]["fields"], ("vllm_version", "tensor_parallel_size", "dtype",
                                               "max_seq_len", "enforce_eager", "enable_prefix_caching"),
                ))
            elif key == "cumulative_preemption_report":
                print(f"    最后一条 cumulative_preemption={table['last']['value']}")
            elif key in CAPACITY:
                print(f"    {key}={table['last']['value']}")
            elif key in ("kv_transfer_intervals", "spec_decoding_intervals"):
                print(f"    {key} 已见区间={table['count']}")
        if p["errors"]:
            counts = {key: table["count"] for key, table in p["errors"].items()}
            print("    错误信号", json.dumps(counts, ensure_ascii=False))
    if not any("GDN_PHASE_PERF" in p["periodic"] for p in work["processes"]):
        print("  未见 GDN_PHASE_PERF。旧版、Base 或截取日志可以缺少此项。")
    print("  每个进程/会话各取最近快照。TP rank 与 P/D 日志分别保留，累计值未相加。")


def print_client(work):
    print(f"\n客户端 [{work['label']}] {work['file']}")
    if not work["runs"]:
        print("  未识别到 AIS Bench 文本指标。")
    for i, run in enumerate(work["runs"], 1):
        print(
            f"  基准 #{i}",
            display_fields(run, ("offered_concurrency", "api_task_elapsed_s")),
        )
        for key in (
            "benchmark_duration",
            "ttft",
            "tpot",
            "e2el",
            "success_requests",
            "failed_requests",
            "total_input_tokens",
            "total_generated_tokens",
            "output_tokens_s",
        ):
            table = run["metrics"].get(key, {})
            if not table:
                continue
            for stage, row in table.items():
                p99 = f" P99={row['p99']}" if row.get("p99") is not None else ""
                print(
                    f"    {key} ({stage}): {row['value']} {row['unit'] or ''}{p99}".rstrip()
                )
    print(
        "  整批耗时以 Benchmark Duration 为准。服务端 worker 窗口和 AIS 外层耗时单独保留。"
    )


def input_logs(entries, parser, kind):
    result, labels = [], set()
    for entry in entries:
        label, sep, value = entry.partition("=")
        path = Path(value if sep else label)
        label = label if sep else path.name
        if not label or label in labels:
            parser.error(f"{kind} 日志标签为空或重复: {label!r}")
        if not path.is_file():
            parser.error(f"日志文件不存在: {path}")
        result.append((label, path))
        labels.add(label)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: %(prog)s --server server.log --client ais.log "
            "--json-out evidence.json. File labels are optional: "
            "--server replay=server.log gives this input the label 'replay'."
        ),
    )
    parser.add_argument(
        "--server",
        action="append",
        required=True,
        metavar="PATH",
        help="vLLM server log file; repeat for separate P/D, Base or Replay logs",
    )
    parser.add_argument(
        "--client",
        action="append",
        default=[],
        metavar="PATH",
        help="optional AIS Bench client log for whole-run duration, TTFT/TPOT and throughput",
    )
    parser.add_argument(
        "--json-out", type=Path, metavar="PATH", help="save structured evidence JSON"
    )
    args = parser.parse_args()
    servers = input_logs(args.server, parser, "server")
    clients = input_logs(args.client, parser, "client")
    if args.json_out and args.json_out.resolve() in {
        path.resolve() for _, path in servers + clients
    }:
        parser.error("JSON 输出不能覆盖输入日志")
    try:
        result = {
            "schema": 1,
            "server": [read_server(*entry) for entry in servers],
            "client": [read_client(*entry) for entry in clients],
        }
        if args.json_out:
            args.json_out.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except (OSError, EOFError) as exc:
        parser.exit(1, f"读取或写入失败: {exc}\n")
    for work in result["server"]:
        print_server(work)
    for work in result["client"]:
        print_client(work)
    if args.json_out:
        print(f"\n完整 JSON 已写入 {args.json_out}")


if __name__ == "__main__":
    main()
