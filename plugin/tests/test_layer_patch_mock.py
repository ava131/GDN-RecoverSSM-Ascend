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
        self.shape = tuple(shape) if shape is not None else (
            self.data.shape if self.data is not None else ()
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
            self.kv_cache = (
                FakeTensor(shape=(4, 64, 7)),
                FakeTensor(shape=(4, 4, 128, 128)),
                FakeTensor(shape=(4, 4, 4, 257)),
            )
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
            q = FakeTensor(shape=(12, 2, 128))
            k = FakeTensor(shape=(12, 2, 128))
            v = FakeTensor(shape=(12, 8, 128))
            g = FakeTensor(shape=(12, 8))
            beta = FakeTensor(shape=(12, 8))
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

        def prepare_attn(self, input_batch, cudagraph_mode, block_tables,
                         slot_mappings, attn_groups, kv_cache_config,
                         for_capture=False, ubatch_idx=0):
            return {"layer.0": "metadata"}

    mod = types.ModuleType("vllm_ascend.ops.gdn")
    mod.AscendGatedDeltaNetAttention = GatedDeltaNetAttention
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
    for name in ("vllm_ascend.worker", "vllm_ascend.worker.v2",
                 "vllm_ascend.worker.v2.model_states"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    ms = types.ModuleType("vllm_ascend.worker.v2.model_states.mamba_hybrid")
    ms.AscendMambaHybridModelState = AscendMambaHybridModelState
    ms.__dict__["_recorded"] = recorded
    sys.modules["vllm_ascend.worker.v2.model_states.mamba_hybrid"] = ms
    sys.modules["vllm_ascend.worker.v2.model_states"].mamba_hybrid = ms

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
        return FakeTensor(shape=(1, 12, 8, 128))

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
    gdn_mod, cls = _install_gdn_stub()
    layer_patch = _reset_plugin()
    captured = _install_kernels_stub()
    layer_patch.install()

    layer = cls()
    assert layer.use_gdn_recoverssm is True

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
    assert tuple(out.shape) == (12, 8, 128), out.shape

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
    gdn_mod, cls = _install_gdn_stub()
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
    base = mu.MambaStateShapeCalculator.gated_delta_net_state_shape(2, 2, 8, 128, 128, 4, 3)
    assert len(base) == 2, base
    assert len(mu.MambaStateDtypeCalculator.gated_delta_net_state_dtype()) == 2

    # enabled -> one extra fp32 record of [HV_tp, Q, V+K+1]
    state_patch.enable(4)
    shapes = mu.MambaStateShapeCalculator.gated_delta_net_state_shape(2, 2, 8, 128, 128, 4, 3)
    assert shapes[-1] == (4, 4, 257), shapes
    assert mu.MambaStateDtypeCalculator.gated_delta_net_state_dtype()[-1] == "float32"


def test_disable_flag_keeps_stock_path():
    _install_torch_stub()
    _install_vllm_stubs()
    gdn_mod, cls = _install_gdn_stub()
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
