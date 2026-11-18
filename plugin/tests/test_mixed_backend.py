# SPDX-License-Identifier: Apache-2.0
"""The mixed subset keeps its own token geometry and honest helper counts."""
# ruff: noqa: F811  # pytest injects the imported fixture by name.

import importlib
import sys
import types

import pytest
import torch
from test_ascendc_backend import runtime, verify_spec  # noqa: F401
from test_ascendc_fusion import layer_and_inputs
from test_record_buffers import kernels  # noqa: F401


def test_mixed_preflight_uses_spec_tokens_instead_of_total_batch(runtime):
    _, ac, calls = runtime
    layer, mixed, a, b = layer_and_inputs()
    mixed = torch.empty(100, mixed.shape[1], dtype=mixed.dtype)
    a, b = torch.empty(100, 3, dtype=a.dtype), torch.empty(100, 3, dtype=b.dtype)
    out = torch.empty(100, 3, 128, dtype=mixed.dtype)
    meta = types.SimpleNamespace(
        spec_query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        spec_state_indices_tensor=torch.ones(2, 1, dtype=torch.int32),
        num_spec_decodes=2, num_actual_tokens=100,
    )
    assert not ac.can_fuse_verify(layer, mixed, a, b, meta=meta, out=out)
    assert ac.can_fuse_verify(layer, mixed, a, b, meta=meta, out=out, num_tokens=8)
    assert not ac.can_fuse_verify(layer, mixed, a, b, meta=meta, out=out, num_tokens=9)
    assert calls == []


@pytest.mark.parametrize("scope", ["mixed", "invalid"])
def test_mixed_dispatch_counts_only_two_removed_norm_helpers(runtime, monkeypatch, scope):
    kernels, ac, calls = runtime
    monkeypatch.setattr(
        ac._OPS, "verify_fused", lambda *args: calls.append(args), raising=False
    )
    q, k, v, _, beta, state, corr, key, decay, out, qsl, indices = verify_spec(kernels).args[:12]
    args = (
        q, k, v, torch.zeros(8, 3, dtype=q.dtype), beta,
        torch.zeros(3), torch.zeros(3), state,
        types.SimpleNamespace(correction=corr, key=key, decay=decay), qsl, indices,
    )
    if scope == "invalid":
        with pytest.raises(ValueError, match="scope"):
            ac.launch_fused_verify(*args, spec_query_len=4, out=out, scope=scope)
        assert calls == []
        return
    assert ac.launch_fused_verify(*args, spec_query_len=4, out=out, scope=scope) is out
    assert len(calls) == 1
    counts = ac.counts()
    assert counts["verify_fused_mixed_model_calls"] == 1
    assert counts["verify_fused_saved_helpers"] == 2
    assert counts["verify_fused_mixed_tokens_host_layer_sum"] == 8


def test_owned_mixed_staging_probe_checks_partition_gate_and_merge(runtime, monkeypatch):
    kernels, _, _ = runtime
    probe = importlib.import_module(kernels._contract_package + ".ascendc_selftest")
    mixed = importlib.import_module(kernels._contract_package + ".mixed_forward")
    parent = types.ModuleType("vllm_ascend")
    parent.__path__ = []
    ops = types.ModuleType("vllm_ascend.ops")
    ops.__path__ = []
    gdn = types.ModuleType("vllm_ascend.ops.gdn")

    def gate(logA, a, b, bias):
        x = a.float() + bias.float()
        g = -logA.exp() * torch.where(x <= 20, (1 + x.exp()).log(), x)
        return g.unsqueeze(0), b.float().sigmoid().to(b.dtype).unsqueeze(0)

    gdn.DeviceOperator = types.SimpleNamespace(fused_gdn_gating=gate)
    ops.gdn = gdn
    parent.ops = ops
    for module in (parent, ops, gdn):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    before = mixed.counters()
    probe._mixed_staging_probe("cpu")
    assert mixed.counters() == before  # probes do not impersonate model calls

    def corrupt_merge(out, *args, **kwargs):
        out.zero_()

    monkeypatch.setattr(mixed, "merge_mixed_output", corrupt_merge)
    with pytest.raises(RuntimeError, match="mixed output"):
        probe._mixed_staging_probe("cpu")
