# SPDX-License-Identifier: Apache-2.0
"""CPU storage/launch contracts; these tests do not compile NPU Triton code."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
import types
from pathlib import Path

import pytest
import torch


class _Kernel:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


@pytest.fixture
def kernels(monkeypatch):
    """Load the real Python ABI against CPU Torch and recording launchers."""

    def module(name, **attrs):
        result = types.ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm_ascend",
        "vllm_ascend.ops",
        "vllm_ascend.ops.triton",
    ):
        module(name)
    triton = types.SimpleNamespace(
        jit=lambda fn=None, **options: (
            _Kernel(fn) if fn is not None else lambda actual: _Kernel(actual)
        ),
        heuristics=lambda values: lambda fn: fn,
        cdiv=lambda a, b: (a + b - 1) // b,
        next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
    )
    module(
        "vllm.triton_utils", triton=triton, tl=types.SimpleNamespace(constexpr=object())
    )
    module("vllm.v1.attention.backends.utils", NULL_BLOCK_ID=0)
    module(
        "vllm_ascend.ops.triton.triton_utils",
        get_ub_size_bytes=lambda: 192 * 1024,
        get_vectorcore_num=lambda: 48,
    )
    module("vllm.logger", init_logger=logging.getLogger)
    root = Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend"
    package_name = "_gdn_record_contract"
    package = module(package_name)
    package.__path__ = [str(root)]
    name = package_name + ".kernels"
    spec = importlib.util.spec_from_file_location(name, root / "kernels.py")
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, result)
    spec.loader.exec_module(result)
    result._contract_package = package_name
    return result


def _context(kernels, *, soa=True, key_dtype=torch.bfloat16):
    records = [
        kernels.ReplayRecordBuffers.allocate(
            3, 6, 2, 4, 8, 8, device="cpu", key_dtype=key_dtype
        )
        for _ in range(2)
    ]
    if not soa:
        records = [torch.empty(3, 6, 4, 17) for _ in range(2)]
    return kernels.GDNRecoverSSMCommitContext.from_tensors(
        [torch.zeros(5, 10, 6) for _ in range(2)],
        [torch.zeros(5, 6, 8, 8) for _ in range(2)],
        records,
        spec_query_len=4,
        max_num_reqs=3,
    )


def _inputs(key_dtype=torch.bfloat16):
    return dict(
        g=torch.zeros(1, 8, 6),
        beta=torch.ones(1, 8, 6),
        q=torch.zeros(1, 8, 2, 8, dtype=key_dtype),
        k=torch.zeros(1, 8, 2, 8, dtype=key_dtype),
        v=torch.zeros(1, 8, 6, 8, dtype=key_dtype),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        state_indices=torch.tensor([1, 2], dtype=torch.int32),
        spec_query_len=4,
    )


def test_exact_geometry_and_physical_bytes(kernels):
    kwargs = dict(
        rows=128,
        num_layers=48,
        num_value_heads=12,
        num_key_heads=4,
        window=4,
        key_dim=128,
        value_dim=128,
    )
    assert (
        kernels.record_geometry_bytes(**kwargs, key_dtype=torch.float32)
        == 193.125 * 1024**2
    )
    assert (
        kernels.record_geometry_bytes(**kwargs, key_dtype=torch.bfloat16)
        == 169.125 * 1024**2
    )
    context = _context(kernels)
    expected = kernels.record_geometry_bytes(3, 2, 6, 2, 4, 8, 8, torch.bfloat16)
    assert context.record_bytes == expected
    assert sum(kernels.record_nbytes(r) for r in context.records) == expected
    assert context.num_key_heads == 2
    assert context.record_layout == "soa"
    assert context.key_dtype == torch.bfloat16
    assert context.records[0].shape == (3, 6, 4, 17)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_key_input_dtype_roundtrip_is_exact(kernels, dtype):
    record = kernels.ReplayRecordBuffers.allocate(
        2, 6, 2, 4, 8, 8, device="cpu", key_dtype=dtype
    )
    generator = torch.Generator().manual_seed(231)
    values = torch.randn(record.key.shape, generator=generator).to(dtype)
    record.key.copy_(values)
    assert torch.equal(record.key.float(), values.float())
    assert record.correction.dtype == record.decay.dtype == torch.float32
    assert record.key.shape[1] == 2  # shared by three value heads


def test_invalid_record_head_and_dtype_rejected(kernels):
    with pytest.raises(ValueError, match="geometry"):
        kernels.ReplayRecordBuffers.allocate(2, 6, 4, 4, 8, 8, device="cpu")
    with pytest.raises(ValueError, match="FP16, BF16 or FP32"):
        kernels.ReplayRecordBuffers.allocate(
            2, 6, 2, 4, 8, 8, device="cpu", key_dtype=torch.float64
        )


def test_all_generated_commit_launch_abis_are_checked(kernels):
    for soa in (False, True):
        context = _context(kernels, soa=soa)
        for align in (False, True):
            table = torch.ones(2, 2, dtype=torch.int32) if align else None
            specs = context.make_commit_specs(
                torch.ones(2, dtype=torch.int32),
                torch.tensor([1, 2], dtype=torch.int32),
                torch.tensor([0, 4, 8], dtype=torch.int32),
                request_indices=torch.arange(2, dtype=torch.int64),
                block_table=table,
                num_computed_tokens=torch.zeros(2, dtype=torch.int32)
                if align
                else None,
                mamba_block_size=128 if align else None,
            )
            assert len(specs) == 3
            for spec in specs:
                signature = inspect.signature(spec.kernel.fn)
                non_constexpr = [
                    p
                    for p in signature.parameters.values()
                    if p.annotation is not kernels.tl.constexpr
                ]
                assert len(spec.args) == len(non_constexpr), spec.kernel.__name__
                formal = set(signature.parameters)
                assert set(spec.kwargs) - {"num_warps", "num_stages"} <= formal
            assert specs[-1].kwargs["RECORD_SOA"] == soa
            assert specs[-1].kwargs["NUM_KEY_HEADS"] == (2 if soa else 6)


def test_verify_launch_has_typed_planes_and_rejects_lossy_keys(kernels):
    context = _context(kernels)
    kernels.gdn_recoverssm_verify(
        **_inputs(),
        checkpoint_state=context.checkpoints[0],
        replay_cache=context.records[0],
    )
    _, args, kwargs = kernels._gdn_recoverssm_verify_kernel.calls[-1]
    params = list(
        inspect.signature(kernels._gdn_recoverssm_verify_kernel.fn).parameters
    )
    bound = dict(zip(params, args))
    assert bound["replay_ptr"] is context.records[0].correction
    assert bound["key_record_ptr"] is context.records[0].key
    assert bound["decay_record_ptr"] is context.records[0].decay
    assert kwargs["RECORD_SOA"] is True
    with pytest.raises(ValueError, match="already-normalized"):
        kernels.gdn_recoverssm_verify(
            **_inputs(),
            checkpoint_state=context.checkpoints[0],
            replay_cache=context.records[0],
            qk_already_normalized=False,
        )
    with pytest.raises(ValueError, match="already-normalized"):
        kernels.gdn_recoverssm_verify(
            **_inputs(torch.float16),
            checkpoint_state=context.checkpoints[0],
            replay_cache=context.records[0],
        )


def test_prepare_compiles_actual_align_signatures_and_only_real_rows(
    kernels, monkeypatch
):
    context = _context(kernels)
    seen = []
    monkeypatch.setattr(
        kernels._LaunchSpec, "compile_only", lambda self: seen.append(self)
    )
    rows = context.prepare(
        capacity=512,
        state_table_width=4,
        request_mapping=True,
        device=torch.device("cpu"),
        align_mode=True,
        mamba_block_size=512,
        block_table_width=19,
        request_index_dtype=torch.int64,
    )
    assert rows == [1, 2, 3]
    assert len(seen) == 9
    for spec in seen[::3]:
        names = list(inspect.signature(spec.kernel.fn).parameters)
        arguments = dict(zip(names, spec.args))
        assert arguments["block_table_ptr"].shape[1] == 19
        assert arguments["num_computed_ptr"] is not None
        assert arguments["request_indices_ptr"].dtype == torch.int64
        assert arguments["stride_state_indices"] == 4
        assert arguments["mamba_block_size"] == 512


def _load_selftest(kernels, monkeypatch):
    name = kernels._contract_package + ".selftest"
    path = Path(kernels.__file__).with_name("selftest.py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, result)
    spec.loader.exec_module(result)
    return result


def _cpu_verify(
    g,
    beta,
    q,
    k,
    v,
    *,
    checkpoint_state,
    replay_cache,
    query_start_loc,
    state_indices,
    spec_query_len,
    qk_already_normalized=True,
    out=None,
    **unused,
):
    """Independent CPU executor only for testing the startup check itself."""
    q, k, v = q[0], k[0], v[0]
    heads = checkpoint_state.shape[1]
    ratio = heads // k.shape[1]
    if out is None:
        out = torch.empty(1, q.shape[0], heads, v.shape[-1], dtype=v.dtype)
    for row, page in enumerate(state_indices.tolist()):
        start, end = query_start_loc[row : row + 2].tolist()
        if page == 0:
            out[0, start:end].zero_()
            continue
        state = checkpoint_state[page].float().clone()
        for t, token in enumerate(range(start, end)):
            kk = k[token].float().repeat_interleave(ratio, 0)
            qq = q[token].float().repeat_interleave(ratio, 0)
            decay = g[token].float().exp()
            state = state * decay[:, None, None]
            corr = (v[token].float() - (state * kk[:, None, :]).sum(-1)) * beta[
                token, :, None
            ].float()
            state = state + corr[:, :, None] * kk[:, None, :]
            out[0, token] = (state * qq[:, None, :]).sum(-1) * k.shape[-1] ** -0.5
            replay_cache.correction[row, :, t] = corr
            replay_cache.key[row, :, t] = k[token]
            replay_cache.decay[row, :, t] = decay
    return out


def _cpu_commit(
    self,
    accepted,
    state_indices,
    qsl,
    request_indices=None,
    block_table=None,
    num_computed_tokens=None,
    mamba_block_size=None,
    **unused,
):
    for row, source in enumerate(state_indices.tolist()):
        request = row if request_indices is None else int(request_indices[row])
        count = min(
            int(accepted[request]), int(qsl[row + 1] - qsl[row]), self.spec_query_len
        )
        if source == 0 or count <= 0:
            continue
        destination = source
        boundary = None
        if block_table is not None:
            computed = int(num_computed_tokens[request])
            destination = int(
                block_table[request, max(computed + count - 1, 0) // mamba_block_size]
            )
            cross = (computed // mamba_block_size + 1) * mamba_block_size - computed
            if count >= cross:
                boundary = (
                    cross,
                    int(
                        block_table[request, (computed + cross) // mamba_block_size - 1]
                    ),
                )
        for state, record in zip(self.checkpoints, self.records):
            current = state[source].float().clone()
            ratio = current.shape[0] // record.key.shape[1]
            for token in range(count):
                kk = record.key[row, :, token].float().repeat_interleave(ratio, 0)
                current = current * record.decay[row, :, token, None, None]
                current = (
                    current + record.correction[row, :, token, :, None] * kk[:, None, :]
                )
                if boundary is not None and token + 1 == boundary[0]:
                    state[boundary[1]].copy_(current)
            state[destination].copy_(current)


def test_startup_selftest_checks_all_outputs_gqa_and_preserves_rng(
    kernels, monkeypatch
):
    check = _load_selftest(kernels, monkeypatch)
    monkeypatch.setattr(kernels, "gdn_recoverssm_verify", _cpu_verify)
    monkeypatch.setattr(kernels.GDNRecoverSSMCommitContext, "commit", _cpu_commit)
    before = torch.random.get_rng_state().clone()
    assert check.run(
        num_layers=2,
        num_heads=6,
        num_key_heads=2,
        head_k=8,
        head_v=8,
        spec_query_len=4,
        device=torch.device("cpu"),
        key_dtype=torch.bfloat16,
    )
    assert torch.equal(before, torch.random.get_rng_state())


def test_startup_check_fails_when_attention_output_is_wrong(kernels, monkeypatch):
    check = _load_selftest(kernels, monkeypatch)

    def broken(*args, **kwargs):
        out = _cpu_verify(*args, **kwargs)
        out[0, 0].add_(1)
        return out

    monkeypatch.setattr(kernels, "gdn_recoverssm_verify", broken)
    monkeypatch.setattr(kernels.GDNRecoverSSMCommitContext, "commit", _cpu_commit)
    assert not check.run(
        num_layers=2,
        num_heads=6,
        num_key_heads=2,
        head_k=8,
        head_v=8,
        spec_query_len=4,
        device=torch.device("cpu"),
        dtypes=(torch.float32,),
    )


def test_prepare_verify_covers_serving_pool_and_native_value_layout(
    kernels, monkeypatch
):
    context = _context(kernels)
    seen = []
    monkeypatch.setattr(
        kernels._LaunchSpec, "compile_only", lambda self: seen.append(self)
    )
    report = context.prepare_verify(
        activation_dtype=torch.bfloat16, state_table_width=4
    )
    assert len(report) == len(seen) == 4
    assert {x["value_token_stride"] for x in report} == {6 * 8, (2 * 2 * 8 + 6 * 8)}
    assert {x["beta_dtype"] for x in report} == {
        str(torch.float32),
        str(torch.bfloat16),
    }
    assert all(x["record_rows"] == context.record_rows for x in report)
    assert all(x["state_blocks"] == context.num_blocks for x in report)
    assert all(x["state_index_stride"] == 4 for x in report)
    assert all(x["num_key_heads"] == 2 for x in report)
    for spec in seen:
        params = inspect.signature(spec.kernel.fn).parameters
        non_constexpr = [
            p for p in params.values() if p.annotation is not kernels.tl.constexpr
        ]
        assert len(spec.args) == len(non_constexpr)
        assert spec.kwargs["NUM_BLOCKS"] == context.num_blocks
        assert spec.kwargs["RECORD_SOA"] is True


def test_startup_check_fails_when_only_layer_zero_is_committed(kernels, monkeypatch):
    check = _load_selftest(kernels, monkeypatch)
    monkeypatch.setattr(kernels, "gdn_recoverssm_verify", _cpu_verify)

    def first_layer_only(self, *args, **kwargs):
        checkpoints = self.checkpoints
        try:
            self.checkpoints = checkpoints[:1]
            return _cpu_commit(self, *args, **kwargs)
        finally:
            self.checkpoints = checkpoints

    monkeypatch.setattr(kernels.GDNRecoverSSMCommitContext, "commit", first_layer_only)
    assert not check.run(
        num_layers=2,
        num_heads=6,
        num_key_heads=2,
        head_k=8,
        head_v=8,
        spec_query_len=4,
        device=torch.device("cpu"),
        dtypes=(torch.float32,),
    )


def test_startup_rng_is_restored_even_when_key_check_raises(kernels, monkeypatch):
    check = _load_selftest(kernels, monkeypatch)

    def corrupt_key(*args, **kwargs):
        out = _cpu_verify(*args, **kwargs)
        kwargs["replay_cache"].key[0, 0, 0, 0].add_(1)
        return out

    monkeypatch.setattr(kernels, "gdn_recoverssm_verify", corrupt_key)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="key record"):
        check.run(
            num_layers=2,
            num_heads=6,
            num_key_heads=2,
            head_k=8,
            head_v=8,
            spec_query_len=4,
            device=torch.device("cpu"),
            dtypes=(torch.float32,),
        )
    assert torch.equal(before, torch.random.get_rng_state())
