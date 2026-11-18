"""Completed-output ownership and logging must never require device reads."""

import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def perf(monkeypatch):
    root = Path(__file__).parents[1] / "gdn_recoverssm_ascend"
    package = types.ModuleType("_gdn_perf_test")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "_gdn_perf_test", package)
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    spec = importlib.util.spec_from_file_location(
        "_gdn_perf_test.diagnostics", root / "diagnostics.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_completed_output_uses_its_own_step_snapshot(perf, monkeypatch, caplog):
    events = []

    class AsyncOutput:
        def __init__(self, counts):
            self.num_sampled_tokens_np = counts

        def get_output(self):
            events.append("native_copy_event_wait")
            return "result"

    class Worker:
        def compile_or_warm_up_model(self):
            return None

    async_module = types.ModuleType("vllm.v1.worker.gpu.async_utils")
    async_module.AsyncOutput = AsyncOutput
    worker_module = types.ModuleType("vllm_ascend.worker.worker")
    worker_module.NPUWorker = Worker
    monkeypatch.setitem(sys.modules, async_module.__name__, async_module)
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setattr(perf, "_enabled", lambda: True)
    perf.install()
    perf._READY = True
    perf.note_step(
        (1,), 8, 2, batch=types.SimpleNamespace(
            is_prefilling_np=np.array([True, False]),
        ),
    )
    first = AsyncOutput(np.array([1, 4], dtype=np.int32))
    perf.note_step(
        (0,), 4, 1, batch=types.SimpleNamespace(
            is_prefilling_np=np.array([False]),
        ),
    )
    second = AsyncOutput(np.array([2], dtype=np.int32))
    with caplog.at_level(logging.INFO):
        assert first.get_output() == "result"
        assert second.get_output() == "result"
        first.get_output()  # an accidental repeated read must not double count
    assert len(events) == 3
    assert perf._COUNTS["output_steps"] == 2
    assert perf._COUNTS["spec_rows"] == 2
    assert perf._COUNTS["spec_emitted_tokens"] == 6
    assert perf._COUNTS["emitted_tokens"] == 7
    assert perf._HOST_TIMINGS["output_get_wall", "mixed"]["samples"] == 1
    assert perf._HOST_TIMINGS["output_get_wall", "decode"]["samples"] == 1
    assert len(_phase_reports(caplog)) == 1
    assert _phase_reports(caplog)[0]["host_timing_scope"].startswith("HOST_WALL")


def test_device_tensor_like_input_is_not_read(perf):
    class ForbiddenDeviceValue:
        def sum(self):
            raise AssertionError("device read")

    perf._READY = True
    perf.record_completed(ForbiddenDeviceValue(), None)
    assert not perf._COUNTS


def test_padding_and_launch_accounting(perf, caplog):
    perf._READY = True
    meta = types.SimpleNamespace(num_actual_tokens=100)
    perf.observe_graph(types.SimpleNamespace(num_tokens=108), {"gdn": meta})
    perf.host_commit(90, group_count=3, launch_count=3, rows=25)
    with caplog.at_level(logging.INFO):
        perf.record_completed(np.array([4], dtype=np.int32), ((0,), 100, 25))
    assert perf._COUNTS["graph_padded_tokens"] == 8
    totals = _phase_reports(caplog)[-1]["totals"]
    assert totals["graph_padding_fraction"] == pytest.approx(8 / 108)
    assert totals["launches_per_commit"] == totals["groups_per_commit"] == 3
    assert totals["graph_hist"] == [{"actual": 100, "captured": 108, "steps": 1}]


def test_launch_manifest_does_not_modify_user_configuration(perf):
    from types import SimpleNamespace as NS

    sizes = [1, 84, 96, 108, 512]
    config = NS(
        scheduler_config=NS(
            max_num_seqs=128, max_num_batched_tokens=2048, async_scheduling=True
        ),
        compilation_config=NS(
            cudagraph_capture_sizes=sizes, cudagraph_mode="FULL_DECODE_ONLY"
        ),
        cache_config=NS(
            enable_prefix_caching=True, mamba_cache_mode="align", mamba_block_size=512
        ),
        model_config=NS(enforce_eager=False, max_model_len=32768, dtype="bfloat16"),
        parallel_config=NS(tensor_parallel_size=4),
        num_speculative_tokens=3,
    )
    before = list(sizes)
    manifest = perf.launch_manifest(NS(vllm_config=config))
    manifest["capture_sizes"].append(100)
    assert sizes == before
    assert config.compilation_config.cudagraph_capture_sizes is sizes


def test_ready_requires_completed_prepare_token(perf, monkeypatch, caplog):
    class AsyncOutput:
        def __init__(self):
            pass

        def get_output(self):
            return None

    class Worker:
        def compile_or_warm_up_model(self):
            return None

    async_module = types.ModuleType("vllm.v1.worker.gpu.async_utils")
    async_module.AsyncOutput = AsyncOutput
    worker_module = types.ModuleType("vllm_ascend.worker.worker")
    worker_module.NPUWorker = Worker
    monkeypatch.setitem(sys.modules, async_module.__name__, async_module)
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setattr(perf, "_enabled", lambda: True)
    monkeypatch.setattr(perf, "launch_manifest", lambda worker: {})
    perf._RANK = 1  # No background logger is needed for this readiness test.
    perf.install()
    worker = Worker()
    with pytest.raises(RuntimeError, match="readiness refused"):
        worker.compile_or_warm_up_model()
    assert not perf._READY
    worker._gdn_patch_prepared = True
    worker._gdn_perf_startup = {"contexts": [{"state_dtype": "torch.float32"}]}
    with caplog.at_level(logging.INFO):
        worker.compile_or_warm_up_model()
    assert perf._READY
    ready = [
        json.loads(r.message.split("json=", 1)[1])
        for r in caplog.records
        if r.message.startswith("GDN_READY json=")
    ]
    assert len(ready) == 1
    assert ready[0]["rank"] == 1 and ready[0]["validation"] == "STARTUP_COMPLETE"
    assert ready[0]["replay_enabled"] is True
    assert ready[0]["startup"] == worker._gdn_perf_startup
    assert "GDN_PHASE_PERF_CONFIG" not in caplog.text


def test_idle_summary_does_not_count_idle_time(perf, caplog, monkeypatch):
    perf._START = 10.0
    perf._LAST_OUTPUT_AT = 20.0
    perf._COUNTS["output_steps"] = 1
    monkeypatch.setattr(perf.time, "monotonic", lambda: 1000.0)
    with caplog.at_level(logging.INFO):
        perf.report()
        perf.report()  # Shutdown after an idle report need not repeat it.
    assert len(_phase_reports(caplog)) == 1
    assert _phase_reports(caplog)[0]["worker_window_s"] == 10.0


def test_workload_uses_completed_snapshot_and_cpu_context(perf, caplog):
    first = types.SimpleNamespace(
        is_prefilling_np=np.array([False, True]),
        num_computed_tokens_np=np.array([23000, 1000]),
    )
    perf._READY = True
    perf.note_step((0,), 7, 2, batch=first)
    snapshot = perf._LATEST
    # The producer is allowed to advance before this output becomes available.
    first.num_computed_tokens_np[:] = 0
    perf.note_step((0,), 4, 1)
    with caplog.at_level(logging.INFO):
        perf.record_completed(np.array([4, 1], dtype=np.int32), snapshot)
    assert perf._BATCHES[("mixed", 2)] == 1
    assert perf._COUNTS["context_tokens_sum"] == 24000
    assert perf._COUNTS["context_tokens_max"] == 23000
    assert perf._COUNTS["peak_request_rows"] == 2
    report = _phase_reports(caplog)[-1]
    mixed = report["phases"]["mixed"]
    assert mixed["spec_rows"] == 1 and mixed["spec_emitted_tokens"] == 4
    assert mixed["accepted_output_hist"] == {"4": 1}
    assert report["totals"]["mean_computed_context"] == 12000.0
    assert report["paths"]["scope"] == "HOST_DISPATCH_COUNTS_SINCE_READY_NOT_GRAPH_EXECUTIONS"
    assert not any(
        record.message.startswith(("GDN_PERF ", "GDN_WORKLOAD ", "GDN_PATHS "))
        for record in caplog.records
    )


def test_workload_does_not_convert_tensor_fields(perf):
    class ForbiddenTensor:
        def __array__(self):
            raise AssertionError("must not convert device tensors")

        def item(self):
            raise AssertionError("must not read device tensors")

    batch = types.SimpleNamespace(
        is_prefilling_np=ForbiddenTensor(),
        num_computed_tokens_np=ForbiddenTensor(),
    )
    perf.note_step((0,), 4, 1, batch=batch)
    assert perf._LATEST.phase == "unknown"
    assert perf._LATEST.context_rows == 0


def test_no_graph_observations_are_reported_as_unknown(perf, caplog):
    perf._COUNTS["output_steps"] = 1
    with caplog.at_level(logging.INFO):
        perf.report()
    totals = _phase_reports(caplog)[-1]["totals"]
    assert totals["graph_observed_steps"] == 0
    assert totals["graph_padding_fraction"] is None


def _phase_reports(caplog):
    return [
        json.loads(record.message.split("json=", 1)[1])
        for record in caplog.records
        if record.message.startswith("GDN_PHASE_PERF json=")
    ]


def test_first_serving_commit_reports_deltas_from_ready(perf, caplog, monkeypatch):
    perf._READY = True
    counts = {"fold": 56, "verify_fused": 36}
    perf._PATHS_AT_READY = {"counts": dict(counts)}
    monkeypatch.setattr(perf, "_path_counters", lambda: {"counts": dict(counts)})
    with caplog.at_level(logging.INFO):
        perf.record_completed(np.array([0]), None)
        perf.host_commit(85, group_count=1, launch_count=3, rows=1)
        counts.update(fold=57, verify_fused=37, verify_fused_mixed_model_calls=1)
        perf.record_completed(np.array([4]), None)
        perf.record_completed(np.array([4]), None)
        perf.report()
    first, active, tail = _phase_reports(caplog)
    assert first["paths"]["counts"] == {}
    assert active["steps"] == 2 and active["totals"]["commits"] == 1
    assert active["paths"]["counts"] == {
        "fold": 1, "verify_fused": 1, "verify_fused_mixed_model_calls": 1,
    }
    assert tail["steps"] == 3


def test_phase_tokens_use_actual_per_row_schedule_and_owned_snapshot(perf, caplog):
    perf._READY = True
    batch = types.SimpleNamespace(
        is_prefilling_np=np.array([False, True, False]),
        num_computed_tokens_np=np.array([3200, 100, 9], dtype=np.int32),
        num_scheduled_tokens=np.array([3, 5, 1], dtype=np.int32),
        num_draft_tokens=2,
    )
    first = perf.note_step((0,), 9, 3, batch=batch)
    batch.num_scheduled_tokens.fill(0)  # Advanced producer cannot affect first.
    batch.num_computed_tokens_np.fill(0)
    second = perf.note_step(None, 9, 3, batch=batch)  # Inconsistent token sum.
    with caplog.at_level(logging.INFO):
        perf.record_completed(np.array([2, 0, 1]), first)
        perf.record_completed(np.array([0, 0, 0]), second)
        perf.report()
    report = _phase_reports(caplog)[-1]
    stats = report["phases"]["mixed"]
    assert report["steps"] == stats["steps"] == 2
    assert report["first_submit_unix_s"] <= report["last_completed_unix_s"]
    assert stats["target_tokens"] == 18
    assert stats["emitted_tokens"] == 3
    assert stats["target_split_known_steps"] == 1
    assert stats["prefill_target_tokens"] == 5
    assert stats["decode_target_tokens"] == 4
    assert stats["scheduled_draft_tokens"] == 4
    assert stats["spec_target_tokens"] == 3  # Partial window, not MTP3 x 4.
    assert stats["spec_target_known_rows"] == 1
    assert stats["accepted_output_hist"] == {"2": 1}
    assert stats["context_max_steps"] == {"0": 1, "4096": 1}
    assert stats["q_context_known_rows"] == 3
    assert stats["prefill_attention_work_proxy"] == 515
    assert stats["attention_work_proxy"] == 10131


def test_prefill_paths_and_timings_distinguish_dispatches_from_samples(perf, caplog, monkeypatch):
    perf._READY = True
    batch = types.SimpleNamespace(is_prefilling_np=np.array([True]))
    token = perf.timing_start("prepare_attn_host", next_step=True)
    first = perf.note_step(None, 8, 1, batch=batch)
    perf.timing_end(token, step=first)
    state = types.SimpleNamespace(
        shape=(128, 12, 128, 128),
        dtype="torch.float32",
        stride=lambda: (196608, 16384, 128, 1),
    )
    perf.prefill_dispatch("fused_chunk", 8, 1, state)
    token = perf.timing_start("prefill_compute_submit_host")
    perf.timing_end(token)
    monkeypatch.setattr(perf, "_SUBMITTED_STEPS", 6)
    second = perf.note_step(None, 10, 1, batch=batch)
    assert not perf.timing_active()
    assert perf.timing_start("prefill_compute_submit_host") is None

    class NoSampleInspection:
        @property
        def shape(self):
            raise AssertionError("unsampled state metadata should not be read")

    perf.prefill_dispatch("fused_chunk", 10, 1, NoSampleInspection())
    with caplog.at_level(logging.INFO):
        perf.record_completed(np.array([0]), second)
    report = _phase_reports(caplog)[-1]
    branch = report["prefill_paths"]["prefill"]["fused_chunk"]
    assert branch["calls"] == 2
    assert branch["layer_target_tokens"] == 18
    assert branch["layer_request_rows"] == 2
    assert branch["state_layout_examples"] == [{
        "shape": [128, 12, 128, 128],
        "stride": [196608, 16384, 128, 1],
        "dtype": "torch.float32",
    }]
    assert report["host_timings"]["prefill_compute_submit_host"]["prefill"]["samples"] == 1
    assert report["host_timings"]["prepare_attn_host"]["prefill"]["sampled_steps"] == 1
