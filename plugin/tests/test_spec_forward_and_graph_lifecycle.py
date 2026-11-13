# SPDX-License-Identifier: Apache-2.0
"""Behavioral host tests for direct output and graph/startup ownership.

No NPU execution is claimed. Helper spies model the pinned native ABI and assert
that fallback happens before stateful convolution, output aliases are retained,
failed replays are not repeated, and startup prepares the actual align contract.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend"


class Array:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape

    def __getitem__(self, key):
        return Array(self.data[key])

    def unsqueeze(self, dim):
        return Array(np.expand_dims(self.data, dim))

    def squeeze(self, dim):
        return Array(np.squeeze(self.data, axis=dim))


@pytest.fixture
def runtime(monkeypatch):
    package_name = "gdn_perf_runtime_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, package_name, package)
    events = []
    logger = types.SimpleNamespace(
        info=lambda *a: events.append(("info", a)),
        warning=lambda *a: events.append(("warning", a)),
        error=lambda *a: events.append(("error", a)),
    )
    log_module = types.ModuleType("vllm.logger")
    log_module.init_logger = lambda _: logger
    monkeypatch.setitem(sys.modules, "vllm.logger", log_module)
    torch = types.ModuleType("torch")
    torch.int32 = "int32"
    torch.empty_like = lambda value: Array(np.empty_like(value.data))
    torch.npu = types.SimpleNamespace(synchronize=lambda: events.append(("sync",)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    forward = types.ModuleType("vllm.forward_context")
    context = types.SimpleNamespace(attn_metadata={})
    forward.get_forward_context = lambda: context
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward)

    def load(name):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(full_name, ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        return module

    def relative(name, **values):
        module = types.ModuleType(f"{package_name}.{name}")
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        return module

    return types.SimpleNamespace(
        load=load,
        relative=relative,
        context=context,
        events=events,
        monkeypatch=monkeypatch,
    )


def spec_setup(runtime):
    events = runtime.events
    conv = types.SimpleNamespace(
        query_start_loc=object(), cache_indices=object(), num_accepted_tokens=object()
    )
    meta = types.SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=2,
        spec_sequence_masks=object(),
        num_actual_tokens=8,
        spec_decode_metadata=types.SimpleNamespace(spec_causal_conv1d=conv),
    )
    runtime.context.attn_metadata = {"layer.0": meta}
    native = types.ModuleType("vllm_ascend.ops.gdn")
    native._get_packed_conv_weights = lambda _: "packed"
    native._normalize_causal_cache_indices = lambda value: value
    native.wait_for_kv_layer_from_connector = lambda name: events.append(("wait", name))
    native.record_attention_compute_start = lambda: events.append(("fence",))
    native.maybe_save_kv_layer_to_connector = lambda *args: events.append(
        ("save", args)
    )

    def convolution(inputs, state, weights, **kwargs):
        events.append(("conv", inputs.shape, weights, kwargs))
        kwargs["out"].data[:] = inputs.data
        return kwargs["out"]

    native.causal_conv1d_update = convolution

    def norm(value):
        events.append(("norm", value.shape))
        return value

    native.l2norm_fwd = norm

    def gating(A_log, a, b, dt_bias):
        events.append(("gate", A_log, a.data.copy(), b.data.copy(), dt_bias))
        return Array(np.ones((1, 8, 1))), Array(np.ones((1, 8, 1)))

    native.DeviceOperator = types.SimpleNamespace(fused_gdn_gating=gating)
    ascend_ops = types.ModuleType("vllm_ascend.ops")
    ascend_ops.gdn = native
    runtime.monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ascend_ops)
    runtime.monkeypatch.setitem(sys.modules, "vllm_ascend.ops.gdn", native)

    def verify(layer, q, k, v, state, kwargs, *, out=None):
        events.append(("verify", q.shape, kwargs, out))
        out.data[:] = 7

    runtime.relative("layer_patch", _verify=verify)
    state = Array(np.zeros((2, 1, 2, 2)))
    layer = types.SimpleNamespace(
        prefix="layer.0",
        kv_cache=(object(), state),
        conv1d=types.SimpleNamespace(bias="bias"),
        activation=True,
        num_spec=3,
        A_log="A_log",
        dt_bias="dt_bias",
        rearrange_mixed_qkv=lambda value: (
            Array(np.zeros((1, value.shape[0], 1, 2))),
            Array(np.zeros((1, value.shape[0], 1, 2))),
            Array(np.zeros((1, value.shape[0], 1, 2))),
        ),
    )
    inputs = (
        Array(np.arange(12 * 6).reshape(12, 6)),
        Array(np.full((12, 1), 2)),
        Array(np.full((12, 1), 3)),
        Array(np.zeros((12, 1, 2))),
    )
    return runtime.load("spec_forward"), layer, meta, inputs, native


def test_pure_spec_writes_actual_output_and_preserves_padding(runtime):
    module, layer, meta, inputs, native = spec_setup(runtime)
    assert module.try_spec_forward(layer, *inputs)
    output = inputs[-1].data
    np.testing.assert_array_equal(output[:8], 7)
    np.testing.assert_array_equal(output[8:], 0)
    np.testing.assert_array_equal(layer.kv_cache[1].data, 0)
    operations = [event[0] for event in runtime.events]
    assert operations == [
        "wait",
        "fence",
        "conv",
        "gate",
        "norm",
        "norm",
        "verify",
        "save",
    ]
    gate = next(event for event in runtime.events if event[0] == "gate")
    np.testing.assert_array_equal(gate[2], 3)  # raw a before raw b
    np.testing.assert_array_equal(gate[3], 2)
    convolution = next(event for event in runtime.events if event[0] == "conv")
    assert (
        convolution[3]["num_accepted_tokens"]
        is meta.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens
    )
    assert convolution[3]["max_query_len"] == 4


@pytest.mark.parametrize("kind", ["prefill", "ordinary", "profile", "changed_abi"])
def test_fallback_happens_before_any_stateful_helper(runtime, kind):
    module, layer, meta, inputs, native = spec_setup(runtime)
    extra = {}
    if kind == "prefill":
        meta.num_prefills = 1
    elif kind == "ordinary":
        meta.num_spec_decodes = 0
    elif kind == "profile":
        runtime.context.attn_metadata = None
    else:
        extra["unknown_option"] = True
    assert not module.try_spec_forward(layer, *inputs, **extra)
    assert runtime.events == []


def test_partial_stateful_failure_is_propagated_without_fallback(runtime):
    module, layer, meta, inputs, native = spec_setup(runtime)
    calls = []

    def broken(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("convolution failed after submission")

    native.causal_conv1d_update = broken
    with pytest.raises(RuntimeError, match="after submission"):
        module.try_spec_forward(layer, *inputs)
    assert len(calls) == 1


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int
    cg_mode: str = "FULL"


def graph_setup(runtime, *, fail=False, capture_missing=False):
    graph_module = None
    saved_meta = object()

    class Manager:
        def __init__(self):
            self.graphs = {Descriptor(8, 2): types.SimpleNamespace(tasks=[1, 2])}
            self.model_runner = types.SimpleNamespace(
                model_state=types.SimpleNamespace(attn_metadata=saved_meta)
            )
            self.calls = 0
            self.fail = fail

        def capture(self):
            if not capture_missing:
                graph_module.remember_metadata(
                    None,
                    (types.SimpleNamespace(num_tokens=8, num_reqs=2),),
                    {},
                    {"layer": "capture-owned-metadata"},
                )
            return "captured"

        def run_fullgraph(self, desc):
            return self._updatable_graph_replay(
                desc, self.model_runner.model_state.attn_metadata
            )

        def _updatable_graph_replay(self, desc, metadata):
            self.calls += 1
            if self.fail:
                raise RuntimeError("arbitrary failure after partial work")
            return "ok"

    native = types.ModuleType("vllm_ascend.worker.v2.aclgraph_utils")
    native.ModelAclGraphManager = Manager
    worker_v2 = types.ModuleType("vllm_ascend.worker.v2")
    worker_v2.aclgraph_utils = native
    runtime.monkeypatch.setitem(sys.modules, "vllm_ascend.worker.v2", worker_v2)
    runtime.monkeypatch.setitem(sys.modules, native.__name__, native)
    runtime.relative("diagnostics", observe_graph=lambda *args: None)
    graph_module = runtime.load("graph_probe")
    graph_module.enable()
    graph_module.install()
    return graph_module, Manager(), saved_meta


def test_capture_metadata_is_startup_owned_and_released(runtime):
    module, manager, saved = graph_setup(runtime)
    module.remember_metadata(
        None, (types.SimpleNamespace(num_tokens=12, num_reqs=3),), {}, object()
    )
    assert module._CAPTURE_META == {}  # runtime eager does not retain metadata
    assert manager.capture() == "captured"
    assert manager.calls == 1
    assert manager.model_runner.model_state.attn_metadata is saved
    assert module._CAPTURE_META == {}
    assert not module._COLLECTING
    assert module.metrics()["warmed_graphs"] == 1


def test_failed_graph_is_never_automatically_repeated(runtime):
    module, manager, saved = graph_setup(runtime, fail=True)
    desc = next(iter(manager.graphs))
    with pytest.raises(RuntimeError, match="partial work"):
        manager._updatable_graph_replay(desc, {})
    assert manager.calls == 1
    manager.fail = False
    with pytest.raises(RuntimeError, match="unusable"):
        manager._updatable_graph_replay(desc, {})
    assert manager.calls == 1


def test_incomplete_capture_cannot_report_server_ready(runtime):
    module, manager, saved = graph_setup(runtime, capture_missing=True)
    with pytest.raises(RuntimeError, match="metadata missing"):
        manager.capture()
    assert manager.calls == 0
    assert module._CAPTURE_META == {}
    assert not module._COLLECTING
    assert manager.model_runner.model_state.attn_metadata is saved


def test_record_reservation_matches_soa_gqa_and_keeps_margin(runtime):
    runtime.relative("selftest")
    module = runtime.load("startup_patch")
    config = make_config()
    # rows128, layers48, Q4: HV12 corr/decay FP32, HK4 normalized BF16 key.
    expected = 128 * 48 * 4 * (12 * 128 * 4 + 4 * 128 * 2 + 12 * 4)
    assert module._record_scratch_bytes(config) == expected
    assert expected == int(169.125 * 1024**2)
    assert module._graph_margin_bytes(config) == 512 * 1024**2


def test_serving_pointer_guard_arms_only_after_native_warmup(runtime):
    runtime.relative("selftest")
    module = runtime.load("startup_patch")
    builder = types.SimpleNamespace(_gdn_serving_ready=False)
    events = []

    class Worker:
        def __init__(self):
            self.vllm_config = make_config()
            self.model_runner = object()
            self.fail = False

        def determine_available_memory(self):
            return 1 << 30

        def compile_or_warm_up_model(self):
            assert not builder._gdn_serving_ready
            events.append("native-capture-and-warmup")
            if self.fail:
                raise RuntimeError("native warmup failed")
            return "ready"

    worker_module = types.ModuleType("vllm_ascend.worker.worker")
    worker_module.NPUWorker = Worker
    runtime.monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    module._iter_builders = lambda runner: iter([builder])
    module._prepare = lambda worker: events.append("plugin-guards")
    module.install()
    worker = Worker()
    assert worker.compile_or_warm_up_model() == "ready"
    assert events == ["plugin-guards", "native-capture-and-warmup"]
    assert builder._gdn_serving_ready
    builder._gdn_serving_ready = False
    worker.fail = True
    with pytest.raises(RuntimeError, match="native warmup failed"):
        worker.compile_or_warm_up_model()
    assert not builder._gdn_serving_ready


def make_config():
    hf = types.SimpleNamespace(
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["linear_attention"] * 48 + ["full_attention"] * 16,
    )
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_text_config=hf,
            dtype="torch.bfloat16",
            max_model_len=32768,
            architecture="Qwen3_5ForCausalLM",
        ),
        scheduler_config=types.SimpleNamespace(max_num_seqs=128),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
        cache_config=types.SimpleNamespace(
            use_kda_recoverssm=True, mamba_cache_mode="align"
        ),
        num_speculative_tokens=3,
    )


def test_startup_prepares_align_actual_rows_dtype_and_mandatory_selftest(runtime):
    calls = []
    runtime.relative(
        "ascendc_backend",
        initialize=lambda: calls.append(("ascendc_load",)),
        assert_ready=lambda: calls.append(("ascendc_ready",)),
    )
    runtime.relative(
        "ascendc_selftest",
        run=lambda **kwargs: calls.append(("ascendc_selftest", kwargs)),
    )
    runtime.relative(
        "diagnostics", configure=lambda worker: calls.append(("configure",))
    )
    runtime.relative(
        "sampling_patch", prepare=lambda worker: calls.append(("sampling",))
    )
    runtime.relative(
        "grouped_commit",
        prepare_grouped_commit=lambda builders, tables, **kwargs: (
            calls.append(("grouped", tables)) or 5
        ),
        run_grouped_selftest=lambda **kwargs: calls.append(
            ("grouped_selftest", kwargs)
        ),
    )
    selftest = runtime.relative(
        "selftest",
        enabled=lambda: False,
        run=lambda **kwargs: calls.append(("selftest", kwargs)) or True,
        run_conv=lambda **kwargs: calls.append(("convtest", kwargs)) or True,
    )
    module = runtime.load("startup_patch")
    config = make_config()
    state = types.SimpleNamespace(
        shape=(100, 12, 128, 128),
        dtype="bf16",
        device="npu:0",
        stride=lambda: (196608, 16384, 128, 1),
    )
    record = types.SimpleNamespace(key_dtype="bf16")
    context = types.SimpleNamespace(
        checkpoints=[state] * 48,
        records=[record] * 48,
        record_bytes=module._record_scratch_bytes(config),
        record_rows=128,
        num_key_heads=4,
        key_dtype="bf16",
        spec_query_len=4,
        commit_lens=types.SimpleNamespace(numel=lambda: 512),
        prepare_verify=lambda **kwargs: (
            calls.append(("verify_prepare", kwargs)) or [{"ready": True}]
        ),
        prepare=lambda **kwargs: (
            calls.append(("prepare", kwargs)) or list(range(1, 129))
        ),
    )
    builder = types.SimpleNamespace(
        _get_context=lambda: context,
        _accepted_buffer=types.SimpleNamespace(numel=lambda: 512),
        num_spec=3,
        kv_cache_spec=types.SimpleNamespace(block_size=512),
        spec_sequence_indices=types.SimpleNamespace(dtype="int64"),
        vllm_config=config,
    )
    worker = types.SimpleNamespace(
        vllm_config=config,
        model_runner=types.SimpleNamespace(block_tables="actual-block-tables"),
    )
    module._iter_builders = lambda runner: iter([builder])
    module._log_runtime_flags = lambda worker: None
    module._prepare(worker)
    prepared = next(entry[1] for entry in calls if entry[0] == "prepare")
    assert prepared["capacity"] == 128
    assert prepared["align_mode"] is True
    assert prepared["mamba_block_size"] == 512
    assert prepared["block_table_width"] == 64
    assert prepared["request_index_dtype"] == "int64"
    verify_prepared = next(entry[1] for entry in calls if entry[0] == "verify_prepare")
    assert verify_prepared["activation_dtype"] == "torch.bfloat16"
    assert verify_prepared["state_table_width"] == 4
    tested = next(entry[1] for entry in calls if entry[0] == "selftest")
    assert tested["num_key_heads"] == 4
    assert tested["key_dtype"] == "bf16"
    grouped_test = next(entry[1] for entry in calls if entry[0] == "grouped_selftest")
    assert grouped_test["num_value_heads"] == 12
    assert grouped_test["num_key_heads"] == 4
    assert grouped_test["spec_query_len"] == 4
    assert module._SELFTEST_DONE
    assert worker._gdn_patch_prepared

    # Numerical success alone cannot prove the actual model selected AC.
    module._SELFTEST_DONE = False
    backend = sys.modules[module.__package__ + ".ascendc_backend"]

    def missing_route():
        raise RuntimeError("actual model AC route missing")

    backend.assert_ready = missing_route
    with pytest.raises(RuntimeError, match="AC route missing"):
        module._prepare(worker)
    assert not module._SELFTEST_DONE
    assert not worker._gdn_patch_prepared
    backend.assert_ready = lambda: None

    module._SELFTEST_DONE = False
    selftest.run = lambda **kwargs: False
    with pytest.raises(RuntimeError, match="self-check failed"):
        module._prepare(worker)
    assert not module._SELFTEST_DONE
    assert not worker._gdn_patch_prepared


def test_native_kda_is_inert_and_missing_gdn_builders_refuse_ready(runtime):
    runtime.relative("selftest")
    module = runtime.load("startup_patch")
    native_config = make_config()
    native_config.model_config.architecture = "KimiK3ForConditionalGeneration"
    assert not module.feature_enabled(native_config)
    native_worker = types.SimpleNamespace(vllm_config=native_config)
    module._prepare(native_worker)
    assert not hasattr(native_worker, "_gdn_patch_prepared")

    runtime.relative("diagnostics", configure=lambda worker: None)
    runtime.relative("sampling_patch", prepare=lambda worker: None)
    module._log_versions = lambda: None
    module._log_runtime_flags = lambda worker: None
    worker = types.SimpleNamespace(vllm_config=make_config())
    with pytest.raises(RuntimeError, match="no model runner"):
        module._prepare(worker)
    assert worker._gdn_patch_prepared is False
    worker.model_runner = types.SimpleNamespace(attn_groups=[])
    with pytest.raises(RuntimeError, match="no recovery metadata builders"):
        module._prepare(worker)
    assert worker._gdn_patch_prepared is False
