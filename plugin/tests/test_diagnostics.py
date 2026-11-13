"""Completed-output ownership and logging must never require device reads."""

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
