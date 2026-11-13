# SPDX-License-Identifier: Apache-2.0
"""Exercise the real launch producers through the AscendC selector on CPU.

The device operator is replaced with a recorder; no NPU execution is claimed.
"""

import importlib
import sys
import types
from collections import Counter

import pytest
import torch
from test_record_buffers import kernels  # noqa: F401


@pytest.fixture
def runtime(kernels, monkeypatch):
    for name in ("ascendc_backend", "grouped_commit"):
        monkeypatch.delitem(
            sys.modules, kernels._contract_package + "." + name, raising=False
        )
    ac = importlib.import_module(kernels._contract_package + ".ascendc_backend")
    monkeypatch.setattr(ac, "_COUNTS", Counter())
    monkeypatch.setattr(ac, "_FALLBACKS", set())
    calls = []
    monkeypatch.setattr(ac, "_ENABLED", True)
    monkeypatch.setattr(
        ac,
        "_OPS",
        types.SimpleNamespace(
            verify=lambda *args: calls.append(("verify", args)),
            fold=lambda *args: calls.append(("fold", args)),
        ),
    )
    return kernels, ac, calls


def verify_spec(kernels, dtype=torch.bfloat16, beta_dtype=torch.float32):
    q = torch.zeros((1, 8, 1, 128), dtype=dtype)
    value = torch.zeros((1, 8, 3, 128), dtype=dtype)
    state = torch.zeros((4, 3, 128, 128), dtype=torch.float32)
    record = kernels.ReplayRecordBuffers.allocate(
        2, 3, 1, 4, 128, 128, device="cpu", key_dtype=dtype
    )
    specs = []
    kernels.gdn_recoverssm_verify(
        torch.zeros((1, 8, 3), dtype=torch.float32),
        torch.zeros((1, 8, 3), dtype=beta_dtype),
        q,
        q.clone(),
        value,
        checkpoint_state=state,
        replay_cache=record,
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        state_indices=torch.ones((2, 4), dtype=torch.int32)[:, 0],
        spec_query_len=4,
        _collect=specs,
    )
    return specs[0]


def test_real_verify_launch_enters_ac_and_preparation_never_executes(runtime):
    kernels, ac, calls = runtime
    spec = verify_spec(kernels)
    spec.compile_only()
    assert calls == []
    spec.launch()
    assert calls[0][0] == "verify"
    args = calls[0][1]
    assert len(args) == 14
    config = args[-2]
    assert len(config) == 28
    assert config[:8] == [2, 8, 1, 3, 4, 4, 2, 0]
    assert config[-1] == 4  # column-view index stride, not contiguous
    assert args[-1] == 128**-0.5
    assert spec.kernel.calls == []


def test_execution_error_is_never_retried_on_triton(runtime, monkeypatch):
    kernels, ac, _ = runtime

    def fail(*args):
        raise RuntimeError("partially executed state write")

    monkeypatch.setattr(ac._OPS, "verify", fail)
    spec = verify_spec(kernels)
    with pytest.raises(RuntimeError, match="partially executed"):
        spec.launch()
    assert spec.kernel.calls == []


@pytest.mark.parametrize("change", ["half", "inner_norm", "state_layout", "dma_stride"])
def test_unsupported_layout_declines_before_any_submission(runtime, change):
    kernels, ac, calls = runtime
    spec = (
        verify_spec(kernels, torch.float16)
        if change == "half"
        else verify_spec(kernels)
    )
    if change == "inner_norm":
        spec.kwargs["USE_QK_L2NORM"] = True
    elif change == "state_layout":
        a = list(spec.args)
        a[26] = 128
        spec.args = tuple(a)
    elif change == "dma_stride":
        a = list(spec.args)
        a[18] = 127
        spec.args = tuple(a)
    assert ac.select(spec) is None
    assert calls == []
    spec.launch()
    assert len(spec.kernel.calls) == 1
    assert calls == []


def test_actual_grouped_fold_preserves_all_owner_tables(runtime):
    kernels, ac, calls = runtime
    group = importlib.import_module(kernels._contract_package + ".grouped_commit")
    records = [
        kernels.ReplayRecordBuffers.allocate(
            2, 3, 1, 4, 128, 128, device="cpu", key_dtype=torch.bfloat16
        )
        for _ in range(2)
    ]
    states = [torch.zeros((4, 3, 128, 128), dtype=torch.float32) for _ in range(2)]
    convs = [torch.zeros((4, 8, 6), dtype=torch.float32) for _ in range(2)]
    contexts = [
        kernels.GDNRecoverSSMCommitContext.from_tensors(
            [c], [s], [r], spec_query_len=4, max_num_reqs=2
        )
        for c, s, r in zip(convs, states, records)
    ]
    entries = []
    for context in contexts:
        p = types.SimpleNamespace(
            state_indices=torch.tensor([1, 2], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
            request_indices=torch.tensor([0, 1], dtype=torch.int64),
            block_table=None,
            num_computed_tokens=None,
            block_size=None,
            request_rows=(0, 1),
            query_lens=(4, 4),
        )
        entries.append(group.CommitEntry(None, p, context))
    bucket = group.GroupedCommitBucket(contexts)
    persistent = torch.zeros(2, dtype=torch.int32)
    fold = bucket.make_specs(
        entries,
        torch.ones(2, dtype=torch.int32),
        persistent.clone(),
        persistent.clone(),
        persistent.clone(),
    )[-1]
    fold.compile_only()
    assert calls == []
    fold.launch()
    assert calls[0][0] == "fold"
    args = calls[0][1]
    assert len(args) == 19 and len(args[-1]) == 17
    assert args[-1][:9] == [2, 2, 3, 1, 4, 4, 2, 2, 0]
    assert args[4] is bucket.state_addrs
    assert args[12] is bucket.layer_groups
    assert args[13] is bucket.source
    assert args[17] is bucket.boundary_len
    assert fold.kernel.calls == []


def test_ready_requires_actual_preparation_and_execution(runtime):
    _, ac, _ = runtime
    with pytest.raises(RuntimeError, match="actual model"):
        ac.assert_ready()
    ac._COUNTS.update(prepared_verify=1, prepared_fold=1, verify=1, fold=1)
    ac.assert_ready()


def test_dma_gap_bounds(runtime):
    _, ac, _ = runtime
    assert ac._dma_gap(128, 4)
    assert ac._dma_gap(128 + 0xFFFFFFFF // 4, 4)
    assert not ac._dma_gap(127, 4)
    assert not ac._dma_gap(129 + 0xFFFFFFFF // 4, 4)
