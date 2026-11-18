# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "extract_gdn_perf.py"


def extract(tmp_path, *options):
    output = tmp_path / "evidence.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *options, "--json-out", str(output)],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout, json.loads(output.read_text(encoding="utf-8"))


def test_phase_snapshots_remain_separate_by_pid_session_and_file(tmp_path):
    def phase(pid, rank, steps):
        return "GDN_PHASE_PERF json=" + json.dumps(
            {
                "schema": 1,
                "pid": pid,
                "rank": rank,
                "steps": steps,
                "scope": "WORKER_CUMULATIVE_CPU_COMPLETED",
                "phases": {"mixed": {"steps": steps, "target_tokens": 10 * steps}},
                "host_timings": {
                    "prepare_attn_host": {
                        "mixed": {"samples": steps, "sample_total_us": 21}
                    }
                },
            }
        )

    lines = [
        '(Worker_TP0 pid=10) GDN_CONFIG patch=v2 rank=0 parameters={"tp": 2}',
        "(Worker_TP0 pid=10) GDN_CONFIG native normalization before={} after={}",
        "(Worker_TP0 pid=10) GDN_READY patch=v2 rank=0 validation=STARTUP_COMPLETE",
        '(Worker_TP0 pid=10) GDN_MEMORY json={"rank":0,"phase":"before_prepare","kv_blocks":20}',
        '(Worker_TP0 pid=10) GDN_MEMORY json={"rank":0,"phase":"after_prepare","kv_blocks":20}',
        '(Worker_TP0 pid=10) GDN_MEMORY json={"rank":0,"phase":"after_warmup","kv_blocks":20}',
        phase(10, 0, 1),
        phase(11, 1, 5),
        phase(10, 0, 5),
        "(Worker_TP0 pid=10) GDN_PHASE_PERF json={",  # Truncated log tail.
        phase(10, 0, 2),  # New cumulative epoch even when the log reused the pid.
    ]
    p = tmp_path / "p.log.gz"
    with gzip.open(p, "wt", encoding="utf-8") as out:
        for line in lines:
            out.write(
                json.dumps({"time": "2026-11-18T10:00:00.000Z", "log": line}) + "\n"
            )
    d = tmp_path / "d.log"
    d.write_text(phase(10, 0, 2), encoding="utf-8")

    _, data = extract(tmp_path, "--server", f"p={p}", "--server", f"d={d}")
    prefiller, decoder = data["server"]
    assert [work["label"] for work in data["server"]] == ["p", "d"]
    a, b, c = prefiller["processes"]
    assert [(r["pid"], r["session"], r["rank"]) for r in (a, b, c)] == [
        (10, 1, 0),
        (11, 1, 1),
        (10, 2, 0),
    ]
    assert a["startup"]["GDN_CONFIG"]["last"]["fields"]["parameters"] == {"tp": 2}
    memory = a["startup"]["GDN_MEMORY"]["by_phase"]
    assert set(memory) == {"before_prepare", "after_prepare", "after_warmup"}
    assert memory["after_prepare"]["fields"]["json"]["kv_blocks"] == 20
    series = a["periodic"]["GDN_PHASE_PERF"]
    assert series["count"] == 2
    assert series["last"]["time"] == "2026-11-18T10:00:00.000Z"
    assert series["last"]["fields"]["json"]["phases"]["mixed"]["target_tokens"] == 50
    assert a["errors"]["invalid_json:GDN_PHASE_PERF"]["count"] == 1
    assert decoder["processes"][0]["session"] == 1


def test_native_log_and_ais_have_independent_windows(tmp_path):
    server = tmp_path / "base.log"
    server.write_text(
        "(EngineCore pid=2) Initializing a V1 LLM engine (v0.30.0rc1) with config: tensor_parallel_size=4, dtype=torch.bfloat16\n"
        "(EngineCore pid=2) NPU KV cache size: 184,320 tokens\n"
        "(APIServer pid=1) Engine 000: Avg prompt throughput: 42.1 tokens/s, "
        "Avg generation throughput: 11.3 tokens/s, Running: 3 reqs, Waiting: 2 reqs, "
        "Preemptions: 2, NPU KV cache usage: 82.6%, Prefix cache hit rate: 0.9%\n"
        "(APIServer pid=1) Engine 000: Avg prompt throughput: 0.0 tokens/s, "
        "Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, "
        "NPU KV cache usage: 0.0%, Prefix cache hit rate: 0.9%\n"
        "(APIServer pid=1) Engine 001: Avg prompt throughput: 0.0 tokens/s, Running: 0 reqs\n"
        "(APIServer pid=1) SpecDecoding metrics: Mean acceptance length: 3.04, "
        "Accepted: 1356 tokens, Drafted: 2200 tokens\n"
        "(APIServer pid=1) KV Transfer metrics: Median latency (ms)=1.21, bytes=500\n"
        '(APIServer pid=1) 123.4.5.6:101 - "POST /private-content HTTP/1.1" 200 OK\n',
        encoding="utf-8",
    )
    client = tmp_path / "ais.log"
    client.write_text(
        "[2026-11-18 09:01:00,100] run with concurrency: 20\n"
        "[2026-11-18 09:02:00,100] Starting progress bar\n"
        "[2026-11-18 09:15:00,100] Api infer task time elapsed: 821.65s\n"
        "│ TTFT │ total │ 17441.6 ms │ 4.0 ms │ 123 ms │ 11 ms │ 20 ms │ 21 ms │ 29 ms │ 120 │\n"
        "│ Benchmark Duration │ total │ 759520.6659 ms │\n"
        "│ Success Requests │ total │ 120 │\n"
        "[2026-11-18 10:01:00,100] run with concurrency: 4\n"
        "│ Benchmark Duration │ total │ 24000.0 ms │\n",
        encoding="utf-8",
    )
    text, data = extract(
        tmp_path, "--server", f"base={server}", "--client", str(client)
    )
    native = data["server"][0]["processes"][1]["native"]
    assert native["engine_000"]["peak"]["kv_usage_pct"] == 82.6
    assert native["engine_000"]["preemptions_reported_sum"] == 2
    assert native["engine_000"]["preemptions_reported_intervals"] == 1
    assert "preemptions_reported_sum" not in native["engine_001"]
    assert "原生启动" in text
    assert (
        native["kv_transfer_intervals"]["last"]["fields"]["Median latency (ms)"] == 1.21
    )
    assert "123.4.5.6" not in json.dumps(data)
    a, b = data["client"][0]["runs"]
    assert a["progress_start"]["time"] == "2026-11-18 09:02:00,100"
    assert a["api_task_elapsed_s"] == 821.65
    assert a["metrics"]["benchmark_duration"]["total"]["value"] == 759520.6659
    assert a["metrics"]["ttft"]["total"]["p99"] == 29
    assert b["offered_concurrency"] == 4
    assert "success_requests" not in b["metrics"]
    assert "未见 GDN_PHASE_PERF" in text


def test_consolidated_logs_keep_ready_evidence_with_each_epoch(tmp_path):
    def line(marker, **payload):
        return f"{marker} json=" + json.dumps({"schema": 2, "pid": 10, "rank": 0, **payload})

    def ready(timestamp):
        return line(
            "GDN_READY",
            ready_unix_s=timestamp,
            patch="quiet",
            launch_fingerprint="f00d",
            parameters={"tp": 4, "num_speculative_tokens": 3},
            startup={
                "contexts": [
                    {"kv_group": 1, "layers": 48, "state_dtype": "torch.bfloat16",
                     "actual_record_bytes": 4194304}
                ],
                "sampler": {"sort": "native", "workspace_model_bytes": 134217728},
            },
        )

    def phase(steps):
        return line(
            "GDN_PHASE_PERF", steps=steps, worker_window_s=3.14,
            totals={"peak_rows": 25, "graph_padding_fraction": 0.05, "commits": steps},
            paths={"counts": {"verify_fused_mixed_model_calls": 23}},
        )

    path = tmp_path / "server.log"
    path.write_text(
        "\n".join([
            ready(1),
            phase(20),
            ready(1),  # A duplicate record still belongs to the first startup.
            line("GDN_MEMORY", phase="kv_bound_before_prepare", kv_blocks=25),
            line("GDN_MEMORY", phase="after_native_warmup", kv_blocks=25),
            ready(2),
            phase(24),  # An increasing step number still belongs to a new worker.
            ready(3),  # Recognize restart when the first memory line was omitted.
            phase(30),
        ]),
        encoding="utf-8",
    )

    shown, data = extract(tmp_path, "--server", str(path))
    assert data["client"] == []
    server = data["server"][0]
    assert server["label"] == "server.log"
    first, second, third = server["processes"]
    assert [p["session"] for p in (first, second, third)] == [1, 2, 3]
    assert first["startup"]["GDN_READY"]["count"] == 2
    assert set(second["startup"]["GDN_MEMORY"]["by_phase"]) == {
        "kv_bound_before_prepare", "after_native_warmup"
    }
    assert second["startup"]["GDN_READY"]["last"]["fields"]["json"]["ready_unix_s"] == 2
    assert third["periodic"]["GDN_PHASE_PERF"]["last"]["fields"]["json"]["steps"] == 30
    assert "launch_fingerprint=f00d" in shown
    assert "kv_group=1" in shown and "actual_record_bytes=4194304" in shown
    assert "peak_rows=25" in shown and "worker_observed_s=3.14" in shown
    assert "verify_fused_mixed_model_calls=23" in shown
    assert "sort=native" in shown
