# SPDX-License-Identifier: Apache-2.0
"""Real CPU Tensor ownership/stride checks for the raw packed-QKV route."""
# ruff: noqa: F811  # pytest injects the imported fixture by name.

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from test_spec_forward_and_graph_lifecycle import runtime, spec_setup  # noqa: F401


def _module():
    path = Path(__file__).parents[1] / "gdn_recoverssm_ascend" / "spec_forward.py"
    spec = importlib.util.spec_from_file_location("_packed_qkv_view_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 8, 96])
@pytest.mark.parametrize("record_geometry", [False, True])
def test_packed_qkv_views_retain_storage_offsets_stride_and_gqa(
    dtype, tokens, record_geometry
):
    module = _module()
    tp, key_heads, value_heads, dim = 4, 4, 12, 128
    key_width, value_width = key_heads * dim, value_heads * dim
    width = key_width * 2 + value_width
    # Include both a nonzero storage offset and token padding. All views must
    # preserve these, rather than silently materializing a contiguous buffer.
    storage = torch.arange((tokens + 2) * (width + 16)).to(dtype)
    packed = storage.view(tokens + 2, width + 16)[1 : tokens + 1, :width]
    layer = types.SimpleNamespace(
        tp_size=tp,
        num_k_heads=key_heads * tp,
        num_v_heads=value_heads * tp,
        head_k_dim=dim,
        head_v_dim=dim,
        _gdn_recoverssm_records=types.SimpleNamespace(
            key=types.SimpleNamespace(shape=(2, key_heads, 4, dim))
        ),
        kv_cache=(None, types.SimpleNamespace(shape=(2, value_heads, dim, dim))),
    )
    if record_geometry:
        del layer.num_k_heads
        del layer.num_v_heads
    q, k, v = module._packed_qkv_views(layer, packed)
    expected = (
        packed[:, :key_width],
        packed[:, key_width : key_width * 2],
        packed[:, key_width * 2 :],
    )
    for value, reference, heads, offset in zip(
        (q, k, v),
        expected,
        (key_heads, key_heads, value_heads),
        (0, key_width, key_width * 2),
    ):
        assert value.shape == (1, tokens, heads, dim)
        assert value.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
        assert value.storage_offset() == packed.storage_offset() + offset
        # Tensor.view may canonicalize a singleton token dimension. No second
        # token is addressable in that case; both shapes retain the same source
        # byte offsets. Multi-token windows must retain the packed row stride.
        assert value.stride(1) == (packed.stride(0) if tokens > 1 else heads * dim)
        assert value.stride(2) == dim
        assert value.stride(3) == 1
        torch.testing.assert_close(value.squeeze(0).flatten(1), reference)
    # A write through the view must affect the same packed source location.
    k[0, 0, 0, 0] = -7
    assert packed[0, key_width] == -7


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_model_route_skips_native_rearrange_without_changing_fences(
    runtime, dtype
):
    module, layer, _, _, native = spec_setup(runtime)
    runtime.monkeypatch.setitem(sys.modules, "torch", torch)
    layer.tp_size, layer.num_k_heads, layer.num_v_heads = 4, 16, 48
    layer.head_k_dim = layer.head_v_dim = 128
    layer.kv_cache = (object(), torch.zeros(4, 12, 128, 128))
    tokens, actual, key_width, value_width = 12, 8, 512, 1536
    width = key_width * 2 + value_width
    mixed = torch.arange(tokens * width).view(tokens, width).to(dtype)
    b = torch.full((tokens, 12), 2, dtype=dtype)
    a = torch.full((tokens, 12), 3, dtype=dtype)
    destination = torch.zeros(tokens, 12, 128, dtype=dtype)
    convolutions, verified = [], []

    def preflight(*args, **kwargs):
        runtime.events.append(("preflight",))
        return True

    def convolution(inputs, state, weights, **kwargs):
        runtime.events.append(("conv",))
        kwargs["out"].copy_(inputs)
        convolutions.append(kwargs["out"])
        return kwargs["out"]

    def native_rearrange(_):
        raise AssertionError("the fused route must not call native cat/rearrange")

    def verify(layer, q, k, v, state, kwargs, *, out):
        runtime.events.append(("verify",))
        verified.append((q, k, v, kwargs, out))
        out.fill_(7)

    runtime.relative("ascendc_backend", can_fuse_verify=preflight)
    runtime.relative("layer_patch", _verify=verify)
    native.causal_conv1d_update = convolution
    layer.rearrange_mixed_qkv = native_rearrange
    assert module.try_spec_forward(layer, mixed, b, a, destination)
    assert [event[0] for event in runtime.events] == [
        "preflight",
        "wait",
        "fence",
        "conv",
        "verify",
        "save",
    ]
    assert len(convolutions) == len(verified) == 1
    q, k, v, kwargs, out = verified[0]
    packed = convolutions[0]
    for value, heads, offset in zip(
        (q, k, v), (4, 4, 12), (0, key_width, key_width * 2)
    ):
        assert value.shape == (actual, heads, 128)
        assert value.stride() == (width, 128, 1)
        assert value.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
        assert value.storage_offset() == packed.storage_offset() + offset
    assert kwargs["raw_fused"] == (layer.A_log, layer.dt_bias)
    assert kwargs["g"].untyped_storage().data_ptr() == a.untyped_storage().data_ptr()
    assert kwargs["beta"].untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    assert out.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr()
    torch.testing.assert_close(
        destination[:actual], torch.full_like(destination[:actual], 7)
    )
    torch.testing.assert_close(
        destination[actual:], torch.zeros_like(destination[actual:])
    )


def test_separate_helper_route_still_calls_native_rearrange(runtime):
    module, layer, _, inputs, _ = spec_setup(runtime)
    native_rearrange = layer.rearrange_mixed_qkv
    rearranged = []

    def rearrange(packed):
        rearranged.append(packed)
        return native_rearrange(packed)

    layer.rearrange_mixed_qkv = rearrange
    runtime.relative("ascendc_backend", can_fuse_verify=lambda *args, **kwargs: False)
    assert module.try_spec_forward(layer, *inputs)
    assert len(rearranged) == 1
    assert [event[0] for event in runtime.events] == [
        "wait", "fence", "conv", "gate", "norm", "norm", "verify", "save"
    ]
