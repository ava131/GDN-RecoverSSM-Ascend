"""Compare the live deterministic filter with the pinned rc1 threshold oracle."""

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def sampling(monkeypatch):
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    path = Path(__file__).parents[1] / "gdn_recoverssm_ascend" / "sampling_patch.py"
    spec = importlib.util.spec_from_file_location("_test_sampling_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_filter(logits, k, p):
    if p is None and k is None:
        return logits
    probs = logits.softmax(dim=-1)
    sorted_probs, _ = probs.sort(dim=-1, descending=False)
    if k is not None:
        cutoff = sorted_probs.gather(-1, (sorted_probs.size(1) - k.long()).unsqueeze(1))
        cutoff.masked_fill_((k == logits.shape[1]).unsqueeze(1), -float("inf"))
        logits.masked_fill_(probs < cutoff, -float("inf"))
    if p is not None:
        remove = sorted_probs.cumsum(-1) <= 1 - p.unsqueeze(1)
        remove[:, -1] = False
        cutoff = sorted_probs.gather(-1, remove.sum(-1).unsqueeze(1))
        logits.masked_fill_(probs < cutoff, -float("inf"))
    return logits


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["k", "p", "both", "none"])
def test_chunked_filter_exact_distribution_mask(sampling, dtype, mode):
    torch.manual_seed(12)
    values = torch.randn(9, 128, dtype=dtype)
    values[0].zero_()  # exact ties must preserve the deployed threshold semantics
    values[1, 0] = -float("inf")
    k = (
        torch.tensor([128, 1, 3, 16, 64, 128, 10, 10, 127])
        if mode in ("k", "both")
        else None
    )
    p = (
        torch.tensor([1.0, 0.01, 0.5, 0.8, 0.95, 1.0, 0.3, 0.9, 0.7])
        if mode in ("p", "both")
        else None
    )
    sampling.FILTER_WORKING_BYTES = 128 * sampling.ESTIMATED_BYTES_PER_ELEMENT * 2
    expected = native_filter(values.clone(), k, p)
    actual = values.clone()
    result = sampling.filter_logits(actual, k, p)
    assert result is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_values_only_primitive_is_used_without_indices(sampling):
    calls = []

    def sort_values(x, **kwargs):
        calls.append(tuple(x.shape))
        return x.sort(**kwargs).values

    x = torch.randn(7, 31)
    sampling.FILTER_WORKING_BYTES = 31 * 32 * 2
    p = torch.full((7,), 0.8)
    got = sampling.filter_logits(x.clone(), None, p, sort_values=sort_values)
    want = native_filter(x.clone(), None, p)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert calls == [(2, 31), (2, 31), (2, 31), (1, 31)]


def test_original_dtype_fallback_and_empty_rows(sampling):
    def must_not_call(*a, **kw):
        raise AssertionError("only FP32 was validated by the startup probe")

    x = torch.randn(3, 11, dtype=torch.bfloat16)
    p = torch.full((3,), 0.8)
    got = sampling.filter_logits(x.clone(), None, p, sort_values=must_not_call)
    torch.testing.assert_close(got, native_filter(x.clone(), None, p), rtol=0, atol=0)
    empty = torch.empty((0, 11))
    assert sampling.filter_logits(empty, None, torch.empty(0)) is empty


def test_baseline_is_inert_and_shape_budget_is_bounded(sampling, monkeypatch):
    monkeypatch.setattr(sampling, "_enabled", lambda: False)
    sentinel = object()
    fn = sampling._adapter(lambda *args: sentinel)
    assert fn(object(), None, None) is sentinel
    rows = sampling.chunk_rows(248320, 100)
    assert rows * 248320 * 32 <= sampling.FILTER_WORKING_BYTES
    with pytest.raises(ValueError):
        sampling.chunk_rows(0, 1)


def test_padding_fraction_helper():
    path = Path(__file__).parents[1] / "gdn_recoverssm_ascend" / "metrics_math.py"
    spec = importlib.util.spec_from_file_location("_test_metrics_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    f = module.padding_fraction
    assert f(84, 84) == 0
    assert f(0, 0) == 0
    assert f(100, 108) == pytest.approx(8 / 108)
    assert f(0, 108) == 1
    for actual, captured in [(-1, 1), (2, 1), (0, -1)]:
        with pytest.raises(ValueError):
            f(actual, captured)
