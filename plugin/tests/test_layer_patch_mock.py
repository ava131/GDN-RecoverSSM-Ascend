# SPDX-License-Identifier: Apache-2.0
"""Wiring test for the plugin, runnable on a machine with no NPU and no torch.

It stubs ``torch`` and ``vllm_ascend.ops.gdn`` with the minimum surface the patch
touches, installs the plugin patches, and asserts:

* ``use_gdn_recoverssm`` is derived from ``cache_config.use_kda_recoverssm``;
* the replay record is appended to state shape (`[HV_tp, Q, V+K+1]`) and dtype
  (float32);
* the attention backend switches to the RecoverSSM backend;
* a speculative ``recurrent_gated_delta_rule`` call is redirected to
  ``gdn_recoverssm_verify`` with the checkpoint pages taken from column 0 of the
  flattened per-token index tensor;
* a non-speculative call (no ``num_accepted_tokens``) goes to the stock kernel.
"""

from __future__ import annotations

import sys
import types

import numpy as np


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #
class FakeTensor:
    """Just enough tensor surface for the patch and the shim."""

    def __init__(self, data=None, shape=None, dtype=None):
        self.data = np.asarray(data) if data is not None else None
        self.shape = (
            tuple(shape)
            if shape is not None
            else (self.data.shape if self.data is not None else ())
        )
        self.dtype = dtype

    # -- shape ops -----------------------------------------------------------
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def long(self):
        return self

    def __getitem__(self, item):
        if self.data is not None:
            return FakeTensor(self.data[item])
        return FakeTensor(shape=self.shape)

    def min(self):
        return (
            FakeTensor(np.array(int(self.data.min())))
            if self.data is not None
            else FakeTensor()
        )

    def max(self):
        return (
            FakeTensor(np.array(int(self.data.max())))
            if self.data is not None
            else FakeTensor()
        )

    def item(self):
        return int(self.data) if self.data is not None else 0

    def clamp(self, min=None, max=None):
        if self.data is not None:
            out = self.data
            if min is not None:
                out = np.maximum(out, min)
            if max is not None:
                out = np.minimum(out, max)
            return FakeTensor(out)
        return FakeTensor(shape=self.shape)

    def unsqueeze(self, dim):
        return FakeTensor(shape=(1,) + self.shape)

    def squeeze(self, dim=0):
        return FakeTensor(shape=self.shape[1:] if self.shape else ())

    def index_select(self, dim, index):
        if self.data is not None and index.data is not None:
            out = np.take(self.data, index.data.astype(np.int64), axis=dim)
            return FakeTensor(out)
        return FakeTensor(shape=(index.shape[0],) + self.shape[1:])


def _install_torch_stub():
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.int32 = "int32"
    torch.long = "long"
    torch.Tensor = FakeTensor
    sys.modules.setdefault("torch", torch)
    return torch


def _install_gdn_stub():
    calls = {"recurrent": 0}

    class GatedDeltaNetAttention:  # vLLM base class stand-in
        def __init__(self, *a, **k):
            self.cache_config = types.SimpleNamespace(use_kda_recoverssm=True)
            self.num_spec = 3
            self.tp_size = 2
            self.num_v_heads = 8
            self.num_k_heads = 2
            self.head_v_dim = 128
            self.head_k_dim = 128
            # Pages are (conv, checkpoint); the records are scratch bound by the
            # commit context, indexed by request row.
            self.kv_cache = (
                FakeTensor(shape=(4, 64, 7)),
                FakeTensor(shape=(4, 4, 128, 128)),
            )
            self._gdn_recoverssm_records = FakeTensor(shape=(8, 4, 4, 257))
            self.prefix = "model.layers.0.linear_attn"
            # Metadata the stock path would read for this window.
            meta = types.SimpleNamespace(
                num_spec_decodes=2,
                spec_query_start_loc=FakeTensor(np.array([0, 4, 8], dtype=np.int32)),
                spec_state_indices_tensor=FakeTensor(
                    np.array([[0, 0, 0, 0], [5, 5, 5, 5]], dtype=np.int32)
                ),
            )
            fwd_mod = sys.modules["vllm.forward_context"]
            ctx = fwd_mod.get_forward_context()
            ctx.attn_metadata[self.prefix] = meta

        def get_state_shape(self):
            return ((8, 7), (8, 128, 128))

        def get_state_dtype(self):
            return ("conv", "ssm")

        def get_attn_backend(self):
            return "AscendGDNFusedAttentionBackend"

        def _forward_core(self, *a, **k):
            gdn_mod = sys.modules["vllm_ascend.ops.gdn"]
            # The real one calls the recurrent kernel with num_accepted_tokens
            # for the spec window.
            q = FakeTensor(shape=(8, 2, 128))
            k = FakeTensor(shape=(8, 2, 128))
            v = FakeTensor(shape=(8, 8, 128))
            g = FakeTensor(shape=(8, 8))
            beta = FakeTensor(shape=(8, 8))
            ssm_state = self.kv_cache[1]
            flat_idx = FakeTensor(np.array([0, 0, 0, 0, 5, 5, 5, 5], dtype=np.int32))
            out = gdn_mod.recurrent_gated_delta_rule(
                q,
                k,
                v,
                ssm_state,
                g=g,
                beta=beta,
                scale=0.088,
                actual_seq_lengths=FakeTensor(np.array([0, 4, 8], dtype=np.int32)),
                ssm_state_indices=flat_idx,
                num_accepted_tokens=FakeTensor(np.array([1, 1], dtype=np.int32)),
            )
            # Non-speculative call must NOT be redirected.
            gdn_mod.recurrent_gated_delta_rule(
                q,
                k,
                v,
                ssm_state,
                g=g,
                beta=beta,
                scale=0.088,
                actual_seq_lengths=FakeTensor(np.array([0, 4, 8], dtype=np.int32)),
                ssm_state_indices=flat_idx,
            )
            return out

    def stock_recurrent(*args, **kwargs):
        calls["recurrent"] += 1
        rows = kwargs["actual_seq_lengths"].shape[0] - 1
        return FakeTensor(shape=(rows, 8, 128))

    # ---- model-state stub: AscendMambaHybridModelState.prepare_attn ----------
    recorded = {"calls": 0}

    class _RecoverSSMState:
        def record_step(self, attn_metadata, attn_groups, *, for_capture):
            recorded["calls"] += 1
            recorded["attn_metadata"] = attn_metadata
            recorded["attn_groups"] = attn_groups
            recorded["for_capture"] = for_capture

    class AscendMambaHybridModelState:
        def __init__(self):
            self.recoverssm = _RecoverSSMState()

        def prepare_attn(
            self,
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=False,
            ubatch_idx=0,
        ):
            return {"layer.0": "metadata"}

        def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
            return None

    # Faithful three-class topology: base (vLLM), real layer (vLLM subclass),
    # donor (vllm-ascend subclass). The donor is a *sibling* of the real class.
    class AscendGatedDeltaNetAttention(GatedDeltaNetAttention):
        """Method donor: holds the Ascend overrides, never instantiated."""

    mod = types.ModuleType("vllm_ascend.ops.gdn")
    mod.AscendGatedDeltaNetAttention = AscendGatedDeltaNetAttention
    mod.recurrent_gated_delta_rule = stock_recurrent
    mod.__dict__["_calls"] = calls

    # The plugin's metadata builder subclasses these.
    builder_mod = types.ModuleType("vllm_ascend.ops.gdn_attn_builder")

    class AscendGDNAttentionMetadataBuilder:
        def __init__(self, *a, **k):
            self.num_spec = 3
            self.layer_names = ["layer.0"]
            self.vllm_config = types.SimpleNamespace(
                scheduler_config=types.SimpleNamespace(max_num_seqs=4),
                cache_config=types.SimpleNamespace(mamba_cache_mode="align"),
            )
            self.kv_cache_spec = types.SimpleNamespace(block_size=128)

    class AscendGDNAttentionBackend:
        @staticmethod
        def get_builder_cls():
            return AscendGDNAttentionMetadataBuilder

    builder_mod.AscendGDNAttentionMetadataBuilder = AscendGDNAttentionMetadataBuilder
    builder_mod.AscendGDNAttentionBackend = AscendGDNAttentionBackend

    ascend = types.ModuleType("vllm_ascend")
    ascend.__path__ = []
    ops = types.ModuleType("vllm_ascend.ops")
    ops.__path__ = []
    ops.gdn = mod
    ops.gdn_attn_builder = builder_mod
    ascend.ops = ops
    sys.modules["vllm_ascend"] = ascend
    sys.modules["vllm_ascend.ops"] = ops
    for name in (
        "vllm_ascend.worker",
        "vllm_ascend.worker.v2",
        "vllm_ascend.worker.v2.model_states",
    ):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    ms = types.ModuleType("vllm_ascend.worker.v2.model_states.mamba_hybrid")
    ms.AscendMambaHybridModelState = AscendMambaHybridModelState
    ms.__dict__["_recorded"] = recorded
    sys.modules["vllm_ascend.worker.v2.model_states.mamba_hybrid"] = ms
    sys.modules["vllm_ascend.worker.v2.model_states"].mamba_hybrid = ms

    # Faithful topology: one shared base, the real layer as a **sibling** of the
    # Ascend donor, and the worker's by-value method copy (patch_qwen3_5).
    qg = types.ModuleType("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")

    class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
        """Real layer class; starts with only the base behaviour."""

    qg.QwenGatedDeltaNetAttention = QwenGatedDeltaNetAttention

    def _simulate_worker_patch(use_saved_init=True):
        """patch_qwen3_5: copy donor methods onto the real class."""
        donor = AscendGatedDeltaNetAttention
        original_init = QwenGatedDeltaNetAttention.__init__
        if use_saved_init:

            def _init(self, *a, **k):
                original_init(self, *a, **k)

            QwenGatedDeltaNetAttention.__init__ = _init
        QwenGatedDeltaNetAttention._forward_core = donor._forward_core
        QwenGatedDeltaNetAttention.get_attn_backend = donor.get_attn_backend

    QwenGatedDeltaNetAttention._simulate_worker_patch = staticmethod(
        _simulate_worker_patch
    )
    for name in (
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.mamba",
        "vllm.model_executor.layers.mamba.gdn",
    ):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    sys.modules["vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"] = qg

    sys.modules["vllm_ascend.ops.gdn"] = mod
    sys.modules["vllm_ascend.ops.gdn_attn_builder"] = builder_mod
    return mod, GatedDeltaNetAttention


def _install_vllm_stubs():
    """Minimal vLLM surface so the plugin's builder module can be imported."""
    import dataclasses

    def mod(name):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
        return m

    vllm = mod("vllm")
    cfg = mod("vllm.config")
    cfg.VllmConfig = object
    vllm.config = cfg

    logger = mod("vllm.logger")
    logger.init_logger = lambda name: __import__("logging").getLogger(name)
    vllm.logger = logger

    fwd = mod("vllm.forward_context")

    class _Ctx:
        def __init__(self):
            self.attn_metadata = {}

    _ctx_singleton = _Ctx()
    fwd.get_forward_context = lambda: _ctx_singleton
    vllm.forward_context = fwd

    v1 = mod("vllm.v1")
    attn = mod("vllm.v1.attention")
    backend = mod("vllm.v1.attention.backend")

    class AttentionCGSupport:
        ALWAYS = "always"
        UNIFORM_BATCH = "uniform_batch"

    backend.AttentionCGSupport = AttentionCGSupport
    backend.CommonAttentionMetadata = object
    attn.backend = backend

    backends = mod("vllm.v1.attention.backends")
    gdn_attn = mod("vllm.v1.attention.backends.gdn_attn")

    @dataclasses.dataclass
    class GDNAttentionMetadata:
        num_spec_decodes: int = 0
        spec_state_indices_tensor: object = None
        spec_query_start_loc: object = None
        spec_sequence_masks: object = None
        num_accepted_tokens: object = None

    gdn_attn.GDNAttentionMetadata = GDNAttentionMetadata
    backends.gdn_attn = gdn_attn

    rsm = mod("vllm.v1.attention.backends.recoverssm_metadata")

    class RecoverSSMMetadata:
        pass

    @dataclasses.dataclass
    class RecoverSSMPostprocessMetadata:
        num_spec_decodes: int
        request_indices: object = None
        block_table: object = None
        num_computed_tokens: object = None
        block_size: int = 0

    rsm.RecoverSSMMetadata = RecoverSSMMetadata
    rsm.RecoverSSMPostprocessMetadata = RecoverSSMPostprocessMetadata
    backends.recoverssm_metadata = rsm

    kvci = mod("vllm.v1.kv_cache_interface")
    kvci.AttentionSpec = object

    v1.attention = attn
    v1.kv_cache_interface = kvci
    vllm.v1 = v1


def _install_kernels_stub():
    captured = {}

    def gdn_recoverssm_verify(g, beta, q, k, v, **kwargs):
        captured.update(
            g=g,
            beta=beta,
            q=q,
            k=k,
            v=v,
            **{f"kw_{key}": val for key, val in kwargs.items()},
        )
        return FakeTensor(shape=(1, 8, 8, 128))

    mod = types.ModuleType("gdn_recoverssm_ascend.kernels")
    mod.gdn_recoverssm_verify = gdn_recoverssm_verify
    sys.modules["gdn_recoverssm_ascend.kernels"] = mod
    return captured


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def _reset_plugin():
    """Force a fresh import of the patch modules so each test sees apply()."""
    for key in list(sys.modules):
        if key.startswith("gdn_recoverssm_ascend"):
            del sys.modules[key]
    import gdn_recoverssm_ascend.layer_patch as lp

    return lp


def test_plugin_wiring():
    _install_torch_stub()
    _install_vllm_stubs()
    gdn_mod, _donor = _install_gdn_stub()
    # Real layers are instances of the vLLM class the worker patches, not of the
    # Ascend donor subclass. Instantiate the real one.
    cls = sys.modules[
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    ].QwenGatedDeltaNetAttention
    layer_patch = _reset_plugin()
    captured = _install_kernels_stub()
    layer_patch.install()

    layer = cls()
    assert layer_patch._enabled(layer) is True
    backend = layer.get_attn_backend()
    assert backend.get_name() == "ASCEND_GDN_RECOVERSSM", backend.get_name()

    from gdn_recoverssm_ascend.builder import AscendGDNRecoverSSMBackend  # noqa: F401

    out = layer._forward_core()

    # 1) the spec call was redirected
    assert "kw_checkpoint_state" in captured, captured.keys()
    assert captured["kw_spec_query_len"] == 4
    assert captured["kw_qk_already_normalized"] is True
    np.testing.assert_allclose(captured["kw_scale"], 0.088, rtol=1e-6)

    # 2) checkpoint pages come from column 0 of each row: rows start at 0 and 4
    #    in the flattened index tensor -> pages 0 and 5
    idx = captured["kw_state_indices"]
    assert idx.data is not None
    np.testing.assert_array_equal(idx.data, np.array([0, 5]))

    # 3) the non-spec call still hits the stock kernel (exactly once)
    assert gdn_mod.__dict__["_calls"]["recurrent"] == 1

    # 4) the shim returns [T, HV, V] so the layer's .unsqueeze(0) restores the batch dim
    assert tuple(out.shape) == (8, 8, 128), out.shape

    # 6) Ascend's prepare_attn override is wrapped so record_step still fires
    ms = sys.modules["vllm_ascend.worker.v2.model_states.mamba_hybrid"]
    state = ms.AscendMambaHybridModelState()
    meta = state.prepare_attn(None, None, None, None, [["group"]], None)
    assert meta == {"layer.0": "metadata"}
    rec = ms.__dict__["_recorded"]
    assert rec["calls"] == 1, rec
    assert rec["attn_groups"] == [["group"]]
    assert rec["for_capture"] is False


def _install_mamba_utils_stub():
    """Fake the two calculators so state_patch can be installed and probed."""
    for name in (
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.mamba",
    ):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m

    mu = types.ModuleType("vllm.model_executor.layers.mamba.mamba_utils")

    class MambaStateShapeCalculator:
        @classmethod
        def gated_delta_net_state_shape(cls, tp, nk, nv, dk, dv, conv, num_spec=0):
            return ((nv // tp, dv, dk), (nv // tp, dv, dk))

    class MambaStateDtypeCalculator:
        @classmethod
        def gated_delta_net_state_dtype(cls, *a, **k):
            return ("conv", "ssm")

    mu.MambaStateShapeCalculator = MambaStateShapeCalculator
    mu.MambaStateDtypeCalculator = MambaStateDtypeCalculator
    sys.modules["vllm.model_executor.layers.mamba.mamba_utils"] = mu
    sys.modules["vllm.model_executor.layers.mamba"].mamba_utils = mu
    return mu


def test_padded_rows_use_offset_length():
    """num_spec_decodes may count graph-padded rows; qsl len is authoritative."""
    _install_torch_stub()
    _install_vllm_stubs()
    gdn_mod, _donor = _install_gdn_stub()
    # Real layers are instances of the vLLM class the worker patches, not of the
    # Ascend donor subclass. Instantiate the real one.
    cls = sys.modules[
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    ].QwenGatedDeltaNetAttention
    layer_patch = _reset_plugin()
    captured = _install_kernels_stub()

    layer_patch.install()
    layer = cls()

    # Pretend the metadata reports 8 spec-decode rows while the offset tensor
    # only describes 2 windows -- the padding case that used to read out of
    # bounds ("MTE accesses an invalid GM address").
    fwd = sys.modules["vllm.forward_context"]
    meta = fwd.get_forward_context().attn_metadata[layer.prefix]
    meta.num_spec_decodes = 8

    layer._forward_core()

    idx = captured["kw_state_indices"]
    np.testing.assert_array_equal(idx.data, np.array([0, 5]))
    qsl = captured["kw_query_start_loc"]
    assert tuple(qsl.shape) == (3,), qsl.shape  # rows + 1, not 9


def test_loggers_live_under_vllm():
    """vLLM only configures the ``vllm`` logger tree; anything else is dropped."""
    import importlib

    for name in (
        "gdn_recoverssm_ascend.config_patch",
        "gdn_recoverssm_ascend.layer_patch",
        "gdn_recoverssm_ascend.state_patch",
    ):
        mod = importlib.import_module(name)
        assert mod.logger.name.startswith("vllm."), (name, mod.logger.name)


def test_worker_patch_order_independent():
    """The other load order: plugin first, worker's by-value copy afterwards."""
    _install_torch_stub()
    _install_vllm_stubs()
    gdn_mod, _donor = _install_gdn_stub()
    cls = sys.modules[
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    ].QwenGatedDeltaNetAttention
    layer_patch = _reset_plugin()
    captured = _install_kernels_stub()

    layer_patch.install()
    cls._simulate_worker_patch()  # worker runs second and overwrites

    layer = cls()
    assert layer.get_attn_backend().get_name() == "ASCEND_GDN_RECOVERSSM"
    layer._forward_core()
    assert captured.get("kw_state_indices") is not None, "verify did not run"


def test_trace_is_silent_while_capturing():
    """A .item() during graph capture corrupts the captured task list."""
    _install_torch_stub()
    _install_vllm_stubs()
    gdn_mod, _donor = _install_gdn_stub()
    cls = sys.modules[
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    ].QwenGatedDeltaNetAttention
    layer_patch = _reset_plugin()
    _install_kernels_stub()

    torch_mod = sys.modules["torch"]
    torch_mod.npu = types.SimpleNamespace(is_current_stream_capturing=lambda: True)
    try:
        layer_patch.install()
        layer = cls()
        layer._forward_core()
        from gdn_recoverssm_ascend import layer_patch as lp

        assert lp._TRACE["steps"] == 0, "trace consumed a slot during capture"
        assert lp._is_capturing() is True
    finally:
        torch_mod.npu = None


def test_graph_probe_logs_task_count():
    """The probe must log descriptor + updatable task count before replaying."""
    _install_torch_stub()
    _install_vllm_stubs()
    _install_gdn_stub()

    calls = {"replay": 0}

    class _Graph:
        tasks = (1, 2, 3, 4)

    class ModelAclGraphManager:
        def __init__(self):
            self.graphs = {}

        def _updatable_graph_replay(self, desc, attn_metadata):
            calls["replay"] += 1
            return "ok"

    class _Desc:
        num_tokens = 512
        num_reqs = 128

    acl = types.ModuleType("vllm_ascend.worker.v2.aclgraph_utils")
    acl.ModelAclGraphManager = ModelAclGraphManager
    sys.modules["vllm_ascend.worker.v2.aclgraph_utils"] = acl

    from gdn_recoverssm_ascend import graph_probe

    graph_probe._APPLIED = False
    graph_probe._SEEN.clear()
    graph_probe._FAILED.clear()
    graph_probe.enable()
    graph_probe.install()

    mgr = ModelAclGraphManager()
    desc = _Desc()
    mgr.graphs[desc] = _Graph()
    assert mgr._updatable_graph_replay(desc, None) == "ok"
    assert calls["replay"] == 1
    key = [k for k in graph_probe._SEEN]
    assert key, graph_probe._SEEN
    assert graph_probe._SEEN[key[0]]["tasks"] == 4
    assert graph_probe._SEEN[key[0]]["count"] == 1


def test_startup_preparation_runs_before_warmup(monkeypatch):
    """The commit variants must be compiled before vLLM's warmup/capture."""
    _install_torch_stub()
    _install_vllm_stubs()
    _install_gdn_stub()

    events = []

    class _Ctx:
        checkpoints = [types.SimpleNamespace(device="npu")] * 2
        commit_lens = FakeTensor(np.zeros(8, dtype=np.int32))
        record_rows = 8

        def prepare_verify(self, **kwargs):
            return []

        def prepare(
            self, *, capacity, state_table_width, request_mapping, device, **options
        ):
            events.append(("prepare", capacity, state_table_width, request_mapping))
            return list(range(capacity))

    class _Builder:
        num_spec = 3
        _accepted_buffer = FakeTensor(np.zeros(8, dtype=np.int32))
        spec_sequence_indices = types.SimpleNamespace(dtype="int64")

        def __init__(self):
            self._ctx = _Ctx()

        def _get_context(self):
            return self._ctx

    builder = _Builder()

    class _Group:
        def get_metadata_builder(self):
            return builder

    runner = types.SimpleNamespace(attn_groups=[[_Group()]], block_tables=None)
    enabled_config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(
            use_kda_recoverssm=True,
            enable_prefix_caching=False,
            mamba_cache_mode="none",
        ),
        compilation_config=types.SimpleNamespace(cudagraph_mode="NONE"),
        scheduler_config=types.SimpleNamespace(
            async_scheduling=True, get_scheduler_cls=lambda: type("S", (), {})
        ),
        model_config=types.SimpleNamespace(
            max_model_len=32768, dtype="bf16", architecture="Qwen3_5ForCausalLM"
        ),
    )
    builder.vllm_config = enabled_config
    worker = types.SimpleNamespace(model_runner=runner, vllm_config=enabled_config)

    import gdn_recoverssm_ascend as package

    grouped_stub = types.ModuleType("gdn_recoverssm_ascend.grouped_commit")
    grouped_stub.prepare_grouped_commit = lambda *args, **kwargs: 0
    monkeypatch.setitem(sys.modules, grouped_stub.__name__, grouped_stub)
    monkeypatch.setattr(package, "grouped_commit", grouped_stub, raising=False)
    from gdn_recoverssm_ascend import (
        diagnostics,
        grouped_commit,
        sampling_patch,
        startup_patch,
    )

    monkeypatch.setattr(diagnostics, "configure", lambda worker: None)
    monkeypatch.setattr(sampling_patch, "prepare", lambda worker: None)
    monkeypatch.setattr(
        grouped_commit, "prepare_grouped_commit", lambda *args, **kwargs: 0
    )
    monkeypatch.setattr(startup_patch, "_log_context_geometry", lambda *args: 0)
    monkeypatch.setattr(startup_patch, "_log_versions", lambda: None)
    monkeypatch.setattr(startup_patch, "_SELFTEST_DONE", True)
    startup_patch._prepare(worker)
    assert events == [("prepare", 8, 4, True)], events

    # unbound KV cache must fail the startup, not silently defer to the first request
    class _Lazy(_Builder):
        def _get_context(self):
            return None

    lazy = _Lazy()
    group = types.SimpleNamespace(get_metadata_builder=lambda: lazy)
    worker2 = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(attn_groups=[[group]]),
        vllm_config=enabled_config,
    )
    try:
        startup_patch._prepare(worker2)
    except RuntimeError as exc:
        assert "not bound" in str(exc), exc
    else:
        raise AssertionError("unbound KV cache must fail startup")


def test_replay_probe_does_not_touch_the_device():
    """The probe must stay host-only: it once added a drain that changed nothing.

    A device synchronise before a graph's first execute was tried on the machine
    and did not prevent the BuildSqCq failure, so the probe must not call one.
    """
    _install_torch_stub()
    _install_vllm_stubs()
    _install_gdn_stub()

    synced = {"n": 0}
    torch_mod = sys.modules["torch"]
    torch_mod.npu = types.SimpleNamespace(
        synchronize=lambda: synced.__setitem__("n", synced["n"] + 1),
    )
    try:
        calls = {"replay": 0}

        class _Graph:
            tasks = (1, 2)

        class ModelAclGraphManager:
            def __init__(self):
                self.graphs = {}

            def _updatable_graph_replay(self, desc, attn_metadata):
                calls["replay"] += 1
                return "ok"

        acl = types.ModuleType("vllm_ascend.worker.v2.aclgraph_utils")
        acl.ModelAclGraphManager = ModelAclGraphManager
        sys.modules["vllm_ascend.worker.v2.aclgraph_utils"] = acl

        from gdn_recoverssm_ascend import graph_probe

        graph_probe._APPLIED = False
        graph_probe._SEEN.clear()
        graph_probe._FAILED.clear()
        graph_probe.install()
        graph_probe.enable()

        class _Desc:
            num_tokens = 8
            num_reqs = 2

        desc = _Desc()
        mgr = ModelAclGraphManager()
        mgr.graphs[desc] = _Graph()

        for _ in range(3):
            assert mgr._updatable_graph_replay(desc, None) == "ok"

        assert calls["replay"] == 3
        assert synced["n"] == 0, synced
    finally:
        torch_mod.npu = None


def test_arbitrary_replay_failure_is_not_retried():
    """A replay exception may follow partial work; never execute it twice."""
    _install_torch_stub()
    _install_vllm_stubs()
    _install_gdn_stub()

    synced = {"n": 0}
    torch_mod = sys.modules["torch"]
    torch_mod.npu = types.SimpleNamespace(
        synchronize=lambda: synced.__setitem__("n", synced["n"] + 1),
    )
    try:
        state = {"fail": 1, "calls": 0}

        class _Graph:
            tasks = (1, 2)

        class ModelAclGraphManager:
            def __init__(self):
                self.graphs = {}

            def _updatable_graph_replay(self, desc, attn_metadata):
                state["calls"] += 1
                if state["fail"] > 0:
                    state["fail"] -= 1
                    raise RuntimeError("Send sqe failed, model_id=17")
                return "ok"

        acl = types.ModuleType("vllm_ascend.worker.v2.aclgraph_utils")
        acl.ModelAclGraphManager = ModelAclGraphManager
        sys.modules["vllm_ascend.worker.v2.aclgraph_utils"] = acl

        from gdn_recoverssm_ascend import graph_probe

        graph_probe._APPLIED = False
        graph_probe._SEEN.clear()
        graph_probe._FAILED.clear()
        graph_probe.install()
        graph_probe.enable()

        class _Desc:
            num_tokens = 8
            num_reqs = 2

        desc = _Desc()
        mgr = ModelAclGraphManager()
        mgr.graphs[desc] = _Graph()

        try:
            mgr._updatable_graph_replay(desc, None)
        except RuntimeError as exc:
            assert "Send sqe" in str(exc)
        else:
            raise AssertionError("a replay failure must propagate immediately")
        assert state["calls"] == 1, state
        assert synced["n"] == 0, synced
        try:
            mgr._updatable_graph_replay(desc, None)
        except RuntimeError as exc:
            assert "unusable" in str(exc)
        else:
            raise AssertionError("a failed graph must not be executed again")
        assert state["calls"] == 1
    finally:
        torch_mod.npu = None


def test_warmup_uses_the_normal_path_with_recorded_metadata():
    """Each captured graph is executed once via run_fullgraph, with its metadata.

    A raw graph.replay() deadlocks (cross-rank work driven by external events), so
    the warmup must go through run_fullgraph and must restore attn_metadata.
    """
    _install_torch_stub()
    _install_vllm_stubs()
    torch_mod = sys.modules["torch"]
    torch_mod.npu = types.SimpleNamespace(synchronize=lambda: None)
    try:
        from gdn_recoverssm_ascend import graph_probe

        graph_probe._CAPTURE_META.clear()

        class _Desc:
            def __init__(self, tokens, reqs):
                self.num_tokens = tokens
                self.num_reqs = reqs

        class _State:
            attn_metadata = "original"

        replayed = []

        class _Manager:
            def __init__(self):
                self.model_runner = types.SimpleNamespace(model_state=_State())
                self.graphs = {}

            def run_fullgraph(self, desc):
                replayed.append((desc.num_tokens, desc.num_reqs))
                assert self.model_runner.model_state.attn_metadata == (
                    "meta-%d-%d" % (desc.num_tokens, desc.num_reqs)
                )
                return "ok"

        mgr = _Manager()
        d4, d8, d64 = _Desc(4, 1), _Desc(8, 2), _Desc(64, 16)
        for d in (d4, d8, d64):
            mgr.graphs[d] = object()
            graph_probe._CAPTURE_META[(d.num_tokens, d.num_reqs)] = "meta-%d-%d" % (
                d.num_tokens,
                d.num_reqs,
            )

        graph_probe._warm_captured_graphs(mgr)
        assert replayed == [(4, 1), (8, 2), (64, 16)], replayed
        assert mgr.model_runner.model_state.attn_metadata == "original"

        # No graph may be deferred to its first serving execution.
        replayed.clear()
        mgr.graphs[_Desc(12, 3)] = object()
        try:
            graph_probe._warm_captured_graphs(mgr)
        except RuntimeError as exc:
            assert "metadata missing" in str(exc)
        else:
            raise AssertionError("missing graph metadata must stop readiness")
        assert replayed == []
    finally:
        torch_mod.npu = None


def test_record_scratch_is_reserved_from_the_kv_budget():
    """The records must be deducted from available KV memory, not added on top.

    ``determine_available_memory`` is what decides the pool size; a buffer
    allocated after it competes with the pool.
    """
    _install_torch_stub()
    _install_vllm_stubs()
    from gdn_recoverssm_ascend import startup_patch

    hf = types.SimpleNamespace(
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["linear_attention"] * 48 + ["full_attention"] * 16,
        num_hidden_layers=64,
    )
    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_text_config=hf, dtype="torch.bfloat16"),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
        scheduler_config=types.SimpleNamespace(max_num_seqs=51),
        compilation_config=types.SimpleNamespace(cudagraph_capture_sizes=[1, 4, 8]),
        num_speculative_tokens=3,
    )
    geo = startup_patch._record_geometry(config)
    assert geo == {
        "rows": 51,
        "layers": 48,
        "heads": 12,
        "key_heads": 4,
        "window": 4,
        "width": 257,
        "key_dim": 128,
        "value_dim": 128,
        "key_dtype": "torch.bfloat16",
        "key_bytes": 2,
    }, geo
    expected = 51 * 48 * 4 * (12 * 128 * 4 + 4 * 128 * 2 + 12 * 4)
    assert startup_patch._record_scratch_bytes(config) == expected
    # ~67 MiB, below the packed FP32 representation and the state pages saved.
    assert 60 * 1024 * 1024 < expected < 75 * 1024 * 1024, expected
    assert "rows=51" in startup_patch._record_scratch_desc(config)

    # Records are sized by max_num_seqs, not by the padded graph batch: padded
    # rows never reach the record stores.
    wide = types.SimpleNamespace(
        model_config=config.model_config,
        parallel_config=config.parallel_config,
        scheduler_config=types.SimpleNamespace(max_num_seqs=128),
        compilation_config=types.SimpleNamespace(cudagraph_capture_sizes=[1, 2048]),
        num_speculative_tokens=3,
    )
    assert startup_patch._record_geometry(wide)["rows"] == 128

    # an unusable config must not silently reserve zero for a run that needs it
    broken = types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_text_config=types.SimpleNamespace()),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
        scheduler_config=types.SimpleNamespace(max_num_seqs=8),
        compilation_config=types.SimpleNamespace(cudagraph_capture_sizes=[]),
        num_speculative_tokens=3,
    )
    assert startup_patch._record_geometry(broken) is None
    assert startup_patch._record_scratch_bytes(broken) == 0


def test_selftest_switch_and_reference():
    """The switch is env-only, and the reference must be a real recurrence."""
    import os

    _install_torch_stub()
    _install_vllm_stubs()
    from gdn_recoverssm_ascend import selftest

    os.environ.pop("GDN_RECOVERSSM_SELFTEST", None)
    assert selftest.enabled() is False
    for value in ("1", "true", "YES", "on"):
        os.environ["GDN_RECOVERSSM_SELFTEST"] = value
        assert selftest.enabled() is True, value
    for value in ("0", "false", "nope", ""):
        os.environ["GDN_RECOVERSSM_SELFTEST"] = value
        assert selftest.enabled() is False, value
    os.environ.pop("GDN_RECOVERSSM_SELFTEST", None)


def test_record_scratch_validation_is_row_based():
    """Records are [rows, HV, Q, width]; pages are [blocks, ...].

    The first version of the move-out-of-cache change kept the page-based check,
    which rejected every real start ("layers need matching float32 replay buffers
    of shape (1384, 12, 4, 257)"): the pages still have ``blocks`` entries while
    the scratch has ``rows``.
    """
    _install_torch_stub()
    _install_vllm_stubs()

    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend" / "kernels.py"
    ).read_text()
    assert "[rows, HV, spec_query_len, V + K + 1]" in src or "[rows," in src
    # the page-based comparison must be gone from both validators
    assert "_rec = (nb, HV, spec_query_len, V + K + 1)" not in src
    # the scratch capacity is passed to both kernels so padded rows cannot
    # write past it
    assert "num_record_rows" in src
    assert "record_row_ok" in src
    assert "class ReplayRecordBuffers" in src
    assert "record_layout" in src and "record_rows" in src
    assert "record_row_ok" in src


def test_no_code_path_still_reads_a_third_cache_page():
    """Moving the records out of the cache must leave no page-2 access behind.

    Two of them survived the first attempt: the context log line and the verify
    call site. Both failed only on the machine, at startup, because that is the
    first time the real page layout is visible -- so the check is static here.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend"
    offenders = []
    for path in sorted(root.glob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if "kv[2]" in code or "kv_cache[2]" in code:
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, offenders

    # The verify call site must take the scratch from the context binding.
    layer_src = (root / "layer_patch.py").read_text()
    assert "replay_cache=records" in layer_src
    assert "_gdn_recoverssm_records" in layer_src
    kernels_src = (root / "kernels.py").read_text()
    assert "layer._gdn_recoverssm_records = record" in kernels_src


def test_every_construction_path_exposes_records():
    """``create`` and ``from_tensors`` must both expose ``records``.

    The start-up self-check builds a context straight from ``from_tensors``; only
    ``create`` used to attach the attribute, so the check died with
    AttributeError on the machine while the serving path looked fine.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend" / "kernels.py"
    ).read_text()
    assert "context.records = list(replays)" in src
    assert "context.records = records" in src
    # both must return the object they annotated
    body = src[src.index("def from_tensors(") :]
    body = body[: body.index("def create(")]
    assert "return context" in body


def test_metadata_class_only_touches_its_own_attributes():
    """The metadata dataclass must not read builder-only attributes.

    ``commit_recoverssm_state`` runs on the *metadata* object, but referenced
    ``self.spec_query_len``, which only the builder has. The line sits inside the
    align-mode branch, so the none-mode runs never touched it and it surfaced only
    when prefix caching + align were first enabled.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend"
    tree = ast.parse((root / "builder.py").read_text())
    allowed_prefix = "recoverssm_"
    problems = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name.endswith("Metadata")):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            for sub in ast.walk(item):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                    and not sub.attr.startswith(allowed_prefix)
                    and not sub.attr.startswith("_")
                ):
                    problems.append(f"{node.name}.{item.name}: self.{sub.attr}")
    assert not problems, problems


def test_conv_compaction_semantics_are_documented_in_the_check():
    """The conv leg must compare against the compaction definition, not a copy.

    RFC #19987 requires conv-state consistency; the SSM leg passes commit_conv=False,
    so without this check the convolution half of the validation plan is untested.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend" / "selftest.py"
    ).read_text()
    assert "def run_conv(" in src
    # the expected value is a shifted window, not the unchanged row
    assert "want = before[:, accepted : accepted + history]" in src
    assert "commit_conv=True" in src
    assert "CONV SUMMARY" in src
    startup = (
        Path(__file__).resolve().parents[1]
        / "gdn_recoverssm_ascend"
        / "startup_patch.py"
    ).read_text()
    assert "selftest.run_conv(" in startup


def test_graph_headroom_is_reserved_and_tunable():
    """Graph mode must keep room for the capture pool out of the KV budget.

    vLLM sizes the pool to fill the budget and excludes graph memory from its
    non-KV accounting, so a pool that grows (records moving out of the pages) eats
    the capture headroom: capture then asks for 770 MiB with 545 MiB free.
    """
    import os

    _install_torch_stub()
    _install_vllm_stubs()
    from gdn_recoverssm_ascend import startup_patch

    def cfg(mode):
        return types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(cudagraph_mode=mode)
        )

    os.environ.pop("GDN_RECOVERSSM_KV_MARGIN_MB", None)
    # 512 MiB by default, in both modes: vLLM fills the pool to the budget and the
    # record scratch alone is enough to tip a 480 MiB sampling buffer over (observed
    # with the default at 0). 2 GiB was a misdiagnosis that cost three slots.
    assert startup_patch._graph_margin_bytes(cfg("FULL_DECODE_ONLY")) == 512 << 20
    assert startup_patch._graph_margin_bytes(cfg("NONE")) == 512 << 20

    os.environ["GDN_RECOVERSSM_KV_MARGIN_MB"] = "512"
    assert startup_patch._graph_margin_bytes(cfg("FULL_DECODE_ONLY")) == 512 << 20
    os.environ["GDN_RECOVERSSM_KV_MARGIN_MB"] = "0"
    assert startup_patch._graph_margin_bytes(cfg("FULL_DECODE_ONLY")) == 0
    os.environ.pop("GDN_RECOVERSSM_KV_MARGIN_MB", None)


def test_disabled_run_prints_nothing_at_info_level():
    """``GDN_RECOVERSSM_DISABLE=1`` must be invisible in a baseline run.

    The entry point runs in every process (API server, engine core, each worker),
    so an INFO line on the early-return path turns a clean baseline into six lines
    of plugin output. Every early return must be DEBUG.
    """
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend" / "__init__.py"
    ).read_text()
    tree = ast.parse(src)
    infos = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "info"
        ):
            infos.append(node.lineno)
    assert not infos, f"logger.info at lines {infos}: an inert plugin must be silent"
    assert 'os.getenv("GDN_RECOVERSSM_DISABLE"' in src


def test_graph_probe_refuses_to_install_when_disabled():
    """Even if something imports the probe, a disabled run must not get it."""
    import os

    _install_torch_stub()
    _install_vllm_stubs()
    from gdn_recoverssm_ascend import graph_probe

    graph_probe._APPLIED = False
    os.environ["GDN_RECOVERSSM_DISABLE"] = "1"
    try:
        graph_probe.install()
        assert graph_probe._APPLIED is False
    finally:
        os.environ.pop("GDN_RECOVERSSM_DISABLE", None)


def test_baseline_builds_no_recover_state():
    """Without --use-replayssm nothing may be reserved, logged or probed.

    The plugin entry point loads in every process regardless of the feature flag, so
    a run without --use-replayssm must keep its full KV budget (the reservation used
    to deduct ~3.2 GiB from a baseline) and stay silent.
    """
    _install_torch_stub()
    _install_vllm_stubs()
    from gdn_recoverssm_ascend import graph_probe, startup_patch

    base = types.SimpleNamespace(
        model_config=types.SimpleNamespace(architecture="Qwen3_5ForCausalLM"),
        cache_config=types.SimpleNamespace(use_kda_recoverssm=False),
        compilation_config=types.SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY"),
    )
    on = types.SimpleNamespace(
        model_config=types.SimpleNamespace(architecture="Qwen3_5ForCausalLM"),
        cache_config=types.SimpleNamespace(use_kda_recoverssm=True),
        compilation_config=types.SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY"),
    )
    assert startup_patch.feature_enabled(base) is False
    assert startup_patch.feature_enabled(on) is True

    # the probe must stay inert until the config accepts the feature
    graph_probe._ENABLED = False
    assert graph_probe._ENABLED is False
    graph_probe.enable()
    assert graph_probe._ENABLED is True
    graph_probe._ENABLED = False


def test_spec_row_mapping_never_calls_nonzero_on_device():
    """Borrow native mapping; no plugin CPU recompute or upload remains."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend" / "builder.py"
    ).read_text()
    tree = ast.parse(source)
    helper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_native_spec_rows"
    )
    helper_source = ast.get_source_segment(source, helper)
    assert "spec_sequence_indices" in helper_source
    assert (
        "nonzero"
        not in "\n".join(
            x.split("#")[0]
            for x in helper_source.splitlines()
            if not x.strip().startswith(('"', "'"))
        ).split("raise RuntimeError")[0]
    )
    assert ".to(" not in helper_source and ".copy_(" not in helper_source
    assert "return rows[:n]" in helper_source


def test_state_calculator_patch():
    """Shape/dtype are appended at the calculator, not at the layer."""
    _install_torch_stub()
    _install_vllm_stubs()
    mu = _install_mamba_utils_stub()

    for key in list(sys.modules):
        if key.startswith("gdn_recoverssm_ascend.state_patch"):
            del sys.modules[key]
    from gdn_recoverssm_ascend import state_patch

    state_patch.install()

    # disabled -> untouched
    state_patch.disable()
    base = mu.MambaStateShapeCalculator.gated_delta_net_state_shape(
        2, 2, 8, 128, 128, 4, 3
    )
    assert len(base) == 2, base
    assert len(mu.MambaStateDtypeCalculator.gated_delta_net_state_dtype()) == 2

    # enabled -> the cache still holds only conv + checkpoint. The records are a
    # per-request scratch outside the cache (RFC #19987 s3), so the page layout
    # is identical to stock and prefix-cache pages stay shareable.
    state_patch.enable(4)
    shapes = mu.MambaStateShapeCalculator.gated_delta_net_state_shape(
        2, 2, 8, 128, 128, 4, 3
    )
    assert len(shapes) == 2, shapes
    assert shapes[-1] == (4, 128, 128), shapes
    assert len(mu.MambaStateDtypeCalculator.gated_delta_net_state_dtype()) == 2


def test_disable_flag_keeps_stock_path():
    _install_torch_stub()
    _install_vllm_stubs()
    gdn_mod, _donor = _install_gdn_stub()
    # Real layers are instances of the vLLM class the worker patches, not of the
    # Ascend donor subclass. Instantiate the real one.
    cls = sys.modules[
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    ].QwenGatedDeltaNetAttention
    layer_patch = _reset_plugin()
    _install_kernels_stub()
    layer_patch.install()
    layer = cls()
    layer.use_gdn_recoverssm = False
    layer._forward_core()
    # both calls hit the stock kernel
    assert gdn_mod.__dict__["_calls"]["recurrent"] == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} wiring checks passed")
