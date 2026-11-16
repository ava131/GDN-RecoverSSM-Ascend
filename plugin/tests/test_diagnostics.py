"""Completed-output ownership and logging must never require device reads."""

import ast
import importlib.util
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
    perf.note_step((1,), 8, 2)
    first = AsyncOutput(np.array([1, 4], dtype=np.int32))
    perf.note_step((0,), 4, 1)
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
    assert "timing=HOST_ONLY" in caplog.text


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
    assert "graph_padding_fraction=0.074074" in caplog.text
    assert "launches_per_commit=3.000" in caplog.text
    assert "groups_per_commit=3.000" in caplog.text


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


def test_ready_requires_completed_prepare_token(perf, monkeypatch):
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
    worker.compile_or_warm_up_model()
    assert perf._READY


def test_idle_summary_does_not_count_idle_time(perf, caplog, monkeypatch):
    perf._START = 10.0
    perf._LAST_OUTPUT_AT = 20.0
    monkeypatch.setattr(perf.time, "monotonic", lambda: 1000.0)
    with caplog.at_level(logging.INFO):
        perf.report()
    assert "worker_observed_s=10.000" in caplog.text


def test_cpu_batch_geometry_distinguishes_mixed_work_and_draft_acceptance(perf, caplog):
    perf._READY = True
    batch = types.SimpleNamespace(
        is_prefilling_np=np.array([True, False, False]),
        num_scheduled_tokens=np.array([32, 4, 4]),
        query_start_loc_np=np.array([0, 32, 36, 40]),
        # A routed small-prefill row must not count 31 prompt tokens as drafts.
        num_draft_tokens_per_req=np.array([0, 3, 3]),
        num_computed_tokens_np=np.array([0, 23000, 24000]),
    )
    perf.note_step((0, 1, 2), 40, 3, batch=batch)
    step = perf._LATEST
    # A subsequent prepare must not change the completed output's snapshot.
    perf.note_step((), 100, 1)
    with caplog.at_level(logging.INFO):
        perf.record_completed(np.array([0, 4, 2]), step)
    assert perf._COUNTS["spec_draft_slots"] == 6
    assert perf._COUNTS["spec_draft_accepted"] == 4
    assert perf._COUNTS["prefill_target_tokens"] == 32
    assert perf._COUNTS["context_tokens_sum"] == 47000
    assert perf._BATCHES[("mixed", 3)] == 1
    assert perf._ACCEPTED == {0: 1, 2: 1, 4: 1}
    assert "draft_acceptance=0.666667" in caplog.text
    assert '"mixed":{"steps":1,"target_tokens":40,"emitted_tokens":6}' in caplog.text


def test_full_graph_histogram_keeps_bins_beyond_the_previous_top_eight(perf, caplog):
    perf._READY = True
    for value in range(1, 12):
        perf.observe_graph(
            types.SimpleNamespace(num_tokens=value + 1),
            types.SimpleNamespace(num_actual_tokens=value),
        )
    with caplog.at_level(logging.INFO):
        perf.report()
    assert '"actual":1,"captured":2' in caplog.text
    assert '"actual":11,"captured":12' in caplog.text


def test_unknown_batch_objects_never_trigger_device_conversion(perf):
    class Forbidden:
        def __getattr__(self, name):
            if name == "__array_interface__":
                raise AttributeError(name)
            raise AssertionError(f"unexpected device access {name}")

    batch = types.SimpleNamespace(
        is_prefilling_np=Forbidden(), num_scheduled_tokens=Forbidden(),
        query_start_loc_np=Forbidden(), num_computed_tokens_np=Forbidden(),
        num_draft_tokens_per_req=Forbidden(),
    )
    perf.note_step((0,), 4, 1, batch=batch)
    assert perf._LATEST.phase == "unknown"
    assert perf._LATEST.spec_draft_slots is None


def test_host_phase_sampling_skips_warmup_and_is_sparse(perf, monkeypatch):
    clock = iter([1000, 6000, 10000, 17000])
    monkeypatch.setattr(perf.time, "perf_counter_ns", lambda: next(clock))
    assert perf.phase_begin("recovery_host_submit") is None
    assert not perf._HOST_STAGE_CALLS
    perf._READY = True
    step = perf.StepSnapshot((0,), 4, 1, "decode")
    for _ in range(128):
        sample = perf.phase_begin("recovery_host_submit", step=step)
        perf.phase_end(sample)
    assert perf._HOST_STAGE_HIST[("recovery_host_submit", "decode", 1, 4)] == [
        2, 12.0, 5.0, 7.0
    ]


def test_host_phase_snapshot_belongs_to_producer_not_latest_prepare(perf):
    perf._READY = True
    old = perf.StepSnapshot((0,), 4, 1, "decode")
    # Native attention metadata may be slot-based/read-only. Only our GDN
    # extension receives the producer snapshot.
    metadata = {
        "native": object(),
        "gdn": types.SimpleNamespace(recoverssm_commit=object()),
    }
    perf.bind_step(metadata, old)
    perf.note_step((), 2048, 32)  # A later prepare cannot retag old recovery.
    sample = perf.phase_begin(
        "recovery_host_submit", step=perf.step_for_metadata(tuple(metadata.values()))
    )
    perf.phase_end(sample)
    assert ("recovery_host_submit", "decode", 1, 4) in perf._HOST_STAGE_HIST
    assert perf.step_for_metadata(None) is None
    assert not perf._OBSERVABILITY_FAILED


def test_failed_model_scope_never_becomes_successful_host_sample(perf):
    perf._READY = True
    sample = perf.phase_begin("target_fullgraph_host_submit")
    perf.phase_end(sample, failed=True)
    assert not perf._HOST_STAGE_HIST
    assert perf._COUNTS["host_stage_failed_samples"] == 1


def test_observability_failure_disables_metrics_without_raising(perf, caplog):
    perf._READY = True
    with caplog.at_level(logging.WARNING):
        perf.record_completed(np.array(["not an integer"]), None)
        perf.record_completed(np.array(["still not an integer"]), None)
    assert perf._OBSERVABILITY_FAILED
    assert caplog.text.count("GDN_DIAGNOSTICS disabled") == 1


def test_missing_cpu_schema_is_explicit_and_host_timing_is_not_device(perf, caplog):
    perf._READY = True
    with caplog.at_level(logging.INFO):
        perf.note_step((0,), 4, 1)
        sample = perf.phase_begin("recovery_host_submit", step=perf._LATEST)
        perf.phase_end(sample)
        perf.record_completed(np.array([2]), perf._LATEST)
    assert '"is_prefilling_np":false' in caplog.text
    assert "missing_fields_report_unknown=1" in caplog.text
    assert "timing=HOST_WALL_NOT_DEVICE" in caplog.text
    assert "device_timing=UNAVAILABLE timing_events=0" in caplog.text
    assert "draft_acceptance=-1.000000" in caplog.text


def test_phase_shape_cardinality_is_bounded_without_losing_sample_total(perf):
    perf._READY = True
    for rows in range(1, 300):
        # Distinct scopes force a first sample; overflow remains one fixed bin.
        sample = perf.phase_begin(
            f"test_scope_{rows}", step=perf.StepSnapshot((), rows, rows)
        )
        perf.phase_end(sample)
    assert len(perf._HOST_STAGE_HIST) == 128
    assert sum(stat[0] for stat in perf._HOST_STAGE_HIST.values()) == 299


def test_idle_summary_includes_commit_arriving_after_output_was_reported(perf, monkeypatch, caplog):
    perf._READY = True
    monkeypatch.setattr(perf.time, "monotonic", lambda: 20.0)
    perf.record_completed(np.array([1]), ((0,), 4, 1))
    assert perf._LAST_REPORTED_STEPS == perf._COUNTS["output_steps"]
    perf.host_commit(100.0, 3, 3, 1)
    assert perf._LAST_REPORTED_REVISION < perf._COUNTS["revision"]

    class OneWake:
        def __init__(self):
            self.calls = 0

        def wait(self, timeout):
            self.calls += 1
            return self.calls > 1

    monkeypatch.setattr(perf, "_STOP_REPORTING", OneWake())
    monkeypatch.setattr(perf.time, "monotonic", lambda: 30.0)
    with caplog.at_level(logging.INFO):
        perf._idle_reports()
    assert "commits=1" in caplog.text
    assert perf._LAST_REPORTED_REVISION == perf._COUNTS["revision"]


def test_diagnostics_never_calls_device_timing_or_conversion_apis(perf):
    tree = ast.parse(Path(perf.__file__).read_text())
    forbidden = {"Event", "synchronize", "elapsed_time", "recorded_time", "query",
                 "item", "cpu", "numpy"}
    calls = [node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and not (isinstance(node.func.value, ast.Name)
                      and node.func.value.id == "threading")]
    assert not forbidden.intersection(calls)
