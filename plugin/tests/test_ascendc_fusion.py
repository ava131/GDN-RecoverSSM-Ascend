"""Raw-input dispatch and independent preprocessing references; no NPU claim."""
# ruff: noqa: F811  # pytest intentionally injects imported fixtures by name.

import importlib
import math
import types

import pytest
import torch
from test_ascendc_backend import runtime, verify_spec  # noqa: F401
from test_record_buffers import kernels  # noqa: F401
from test_spec_forward_and_graph_lifecycle import runtime as spec_runtime  # noqa: F401
from test_spec_forward_and_graph_lifecycle import spec_setup


def layer_and_inputs():
    layer = types.SimpleNamespace(
        head_k_dim=128,
        head_v_dim=128,
        A_log=torch.zeros(3),
        dt_bias=torch.zeros(3),
        num_spec=3,
        kv_cache=(None, torch.zeros(4, 3, 128, 128)),
        _gdn_recoverssm_records=types.SimpleNamespace(
            correction=torch.zeros(2, 3, 4, 128),
            key=torch.zeros(2, 1, 4, 128, dtype=torch.bfloat16),
            decay=torch.zeros(2, 3, 4),
        ),
    )
    return (
        layer,
        torch.zeros(8, 640, dtype=torch.bfloat16),
        torch.zeros(8, 3, dtype=torch.bfloat16),
        torch.zeros(8, 3, dtype=torch.bfloat16),
    )


def test_fused_preflight_is_host_only_and_rejects_other_contracts(runtime):
    _, ac, calls = runtime
    layer, mixed, a, b = layer_and_inputs()
    meta = types.SimpleNamespace(
        spec_query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        spec_state_indices_tensor=torch.ones(2, 1, dtype=torch.int32),
        num_spec_decodes=2,
        num_actual_tokens=8,
    )
    out = torch.empty(8, 3, 128, dtype=torch.bfloat16)
    eligible = lambda: ac.can_fuse_verify(layer, mixed, a, b, meta=meta, out=out)
    assert eligible()
    layer.A_log = layer.A_log.bfloat16()
    assert not eligible()
    layer.A_log = torch.zeros(3)
    layer.head_k_dim = 64
    assert not eligible()
    layer.head_k_dim = 128
    assert not ac.can_fuse_verify(layer, mixed, a.float(), b, meta=meta, out=out)
    ac._ENABLED = False
    assert not eligible()
    assert calls == []


def test_fused_dispatch_preserves_output_record_and_index_strides(runtime, monkeypatch):
    kernels, ac, calls = runtime
    monkeypatch.setattr(
        ac._OPS,
        "verify_fused",
        lambda *args: calls.append(("verify_fused", args)),
        raising=False,
    )
    base = verify_spec(kernels)
    q, k, v, _, beta, state, corr, key, decay, out, qsl, indices = base.args[:12]
    raw_a = torch.zeros((8, 3), dtype=q.dtype)
    raw_b = torch.zeros((8, 3), dtype=beta.dtype)
    records = types.SimpleNamespace(correction=corr, key=key, decay=decay)
    result = ac.launch_fused_verify(
        q,
        k,
        v,
        raw_a,
        raw_b,
        torch.zeros(3),
        torch.zeros(3),
        state,
        records,
        qsl,
        indices,
        spec_query_len=4,
        out=out,
    )
    assert result is out
    assert [x[0] for x in calls] == ["verify_fused"]
    args = calls[0][1]
    assert args[3] is raw_a and args[4] is raw_b
    assert args[6] is corr and args[7] is key and args[8] is decay
    assert args[-2] == list(ac.select(base).config)
    assert args[-2][-1] == 4
    assert ac.counts()["verify_fused"] == 1
    assert ac.counts()["verify_fused_model_calls"] == 1
    assert ac.counts()["verify_fused_saved_helpers"] == 3


def test_fused_execution_failure_is_not_retried(runtime, monkeypatch):
    kernels, ac, calls = runtime

    def fail(*args):
        raise RuntimeError("fused device failure")

    monkeypatch.setattr(ac._OPS, "verify_fused", fail, raising=False)
    base = verify_spec(kernels)
    q, k, v, _, beta, state, corr, key, decay, out, qsl, indices = base.args[:12]
    records = types.SimpleNamespace(correction=corr, key=key, decay=decay)
    with pytest.raises(RuntimeError, match="fused device failure"):
        ac.launch_fused_verify(
            q,
            k,
            v,
            torch.zeros(8, 3, dtype=q.dtype),
            beta,
            torch.zeros(3),
            torch.zeros(3),
            state,
            records,
            qsl,
            indices,
            spec_query_len=4,
            out=out,
        )
    assert calls == []
    assert ac.counts().get("verify_fused", 0) == 0


@pytest.mark.parametrize("activation", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("query_len", [1, 4, 8])
def test_owned_raw_case_rounding_and_gate_reference(runtime, activation, query_len):
    kernels, _, _ = runtime
    probe = importlib.import_module(kernels._contract_package + ".ascendc_selftest")
    case = probe.make_fused_case(query_len, activation, torch.float32, activation)
    assert torch.count_nonzero(case["q"][0, 0]) == 0
    assert torch.count_nonzero(case["k"][0, 1]) == 0
    assert torch.count_nonzero(case["q"][0, 2]) == 1
    for name in ("q", "k"):
        raw = case["raw_" + name].double()
        expected = (raw / (raw.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(
            activation
        )
        torch.testing.assert_close(
            case[name],
            expected,
            rtol=1e-6 if activation == torch.float32 else 8e-3,
            atol=1e-7,
        )
        assert case[name].dtype == activation
    for token in range(case["tokens"]):
        for head in range(3):
            x = float(case["raw_a"][token, head]) + float(case["dt_bias"][head])
            softplus = math.log1p(math.exp(x)) if x <= 20 else x
            expected_g = -math.exp(float(case["A_log"][head])) * softplus
            assert float(case["g"][token, head]) == pytest.approx(
                expected_g, abs=1e-7, rel=2e-6
            )
    assert case["beta"].dtype == activation
    assert (case["raw_a"] > 20).any()
    assert (case["raw_a"] < -20).any()


def test_pure_spec_fused_route_skips_all_three_external_helpers(spec_runtime):
    module, layer, _, inputs, _ = spec_setup(spec_runtime)

    def preflight(*args, **kwargs):
        spec_runtime.events.append(("preflight",))
        return True

    spec_runtime.relative("ascendc_backend", can_fuse_verify=preflight)
    assert module.try_spec_forward(layer, *inputs)
    names = [event[0] for event in spec_runtime.events]
    assert names == ["preflight", "wait", "fence", "conv", "verify", "save"]
    verified = next(event for event in spec_runtime.events if event[0] == "verify")
    assert verified[2]["raw_fused"] == (layer.A_log, layer.dt_bias)
    assert verified[2]["g"].shape == (8, 1)
    assert verified[2]["beta"].shape == (8, 1)


@pytest.mark.parametrize(
    "change",
    [
        "fp16_state",
        "strided_state_k",
        "int64_offsets",
        "int64_indices",
        "fp32_key",
        "record_capacity",
        "output_layout",
        "large_window",
        "packed_width",
        "key_head_geometry",
        "record_rank",
        "input_rank",
        "output_rank",
        "tp_geometry",
    ],
)
def test_raw_eligibility_rejects_before_convolution_for_all_unsupported_abi(
    runtime, change
):
    _, ac, calls = runtime
    layer, mixed, a, b = layer_and_inputs()
    meta = types.SimpleNamespace(
        spec_query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        spec_state_indices_tensor=torch.ones(2, 1, dtype=torch.int32),
        num_spec_decodes=2,
        num_actual_tokens=8,
    )
    out = torch.empty(8, 3, 128, dtype=torch.bfloat16)
    if change == "fp16_state":
        layer.kv_cache = (None, layer.kv_cache[1].half())
    elif change == "strided_state_k":
        layer.kv_cache = (None, torch.empty(4, 3, 128, 256)[..., ::2])
    elif change == "int64_offsets":
        meta.spec_query_start_loc = meta.spec_query_start_loc.long()
    elif change == "int64_indices":
        meta.spec_state_indices_tensor = meta.spec_state_indices_tensor.long()
    elif change == "fp32_key":
        layer._gdn_recoverssm_records.key = layer._gdn_recoverssm_records.key.float()
    elif change == "record_capacity":
        layer._gdn_recoverssm_records.correction = (
            layer._gdn_recoverssm_records.correction[:1]
        )
    elif change == "output_layout":
        out = out.transpose(1, 2)
    elif change == "large_window":
        layer.num_spec = 8
    elif change == "packed_width":
        mixed = mixed[:, :512]
    elif change == "key_head_geometry":
        layer.num_k_heads = 2
    elif change == "record_rank":
        layer._gdn_recoverssm_records.key = torch.empty(0)
    elif change == "input_rank":
        mixed = mixed.flatten()
    elif change == "output_rank":
        out = out.flatten()
    elif change == "tp_geometry":
        layer.tp_size = 2
        layer.num_k_heads = 3
    assert not ac.can_fuse_verify(layer, mixed, a, b, meta=meta, out=out)
    assert calls == []
