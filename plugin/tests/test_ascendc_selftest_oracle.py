"""CPU-only checks for the independent AscendC startup oracle and ABI."""

import importlib.util
import re
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).parents[1] / "gdn_recoverssm_ascend" / "ascendc_selftest.py"
spec = importlib.util.spec_from_file_location("_ascendc_probe_oracle", SOURCE)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_sequential_reference_has_independent_hand_computed_prefixes():
    initial = torch.zeros((3, 2, 2))
    q = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    k = q.clone()
    v = torch.tensor([[[1.0, 2.0]] * 3, [[3.0, 4.0]] * 3])
    gate = torch.zeros((2, 3))
    beta = torch.full((2, 3), 0.5)
    output, prefixes, correction, decay = oracle.sequential_reference(
        initial, q, k, v, gate, beta, 1.0
    )
    expected = torch.tensor([[[0.5, 1.0]] * 3, [[1.5, 2.0]] * 3]).double()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert len(prefixes) == 3
    assert torch.count_nonzero(prefixes[0]) == 0
    torch.testing.assert_close(prefixes[1][:, :, 0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(
        prefixes[1][:, :, 1], torch.zeros((3, 2)).double(), rtol=0, atol=0
    )
    torch.testing.assert_close(prefixes[2][:, :, 1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(correction, expected, rtol=0, atol=0)
    torch.testing.assert_close(decay, torch.ones((2, 3)).double(), rtol=0, atol=0)


def test_fold_reference_zero_prefix_and_gqa_records():
    checkpoint = torch.ones((3, 2, 2))
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    corrections = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]] * 3)
    decays = torch.full((3, 2), 0.5)
    original = checkpoint.clone()
    zero = oracle.fold_reference(checkpoint, keys, corrections, decays, 0)
    assert len(zero) == 1
    torch.testing.assert_close(zero[0], original.double(), rtol=0, atol=0)
    prefixes = oracle.fold_reference(checkpoint, keys, corrections, decays, 2)
    expected = torch.tensor([[[1.25, 6.25], [2.25, 8.25]]] * 3).double()
    torch.testing.assert_close(prefixes[-1], expected, rtol=0, atol=0)
    assert torch.equal(checkpoint, original)
    with pytest.raises(ValueError, match="invalid accepted"):
        oracle.fold_reference(checkpoint, keys, corrections, decays, 3)


@pytest.mark.parametrize("query_len", [1, 4, 8])
def test_case_geometry_is_deterministic_and_preserves_rng(query_len):
    before = torch.random.get_rng_state().clone()
    case = oracle.make_case(query_len, torch.bfloat16, torch.float32, torch.float32)
    duplicate = oracle.make_case(
        query_len, torch.bfloat16, torch.float32, torch.float32
    )
    assert torch.equal(before, torch.random.get_rng_state())
    for name in ("q", "k", "v", "g", "beta", "state", "qsl", "indices"):
        assert torch.equal(case[name], duplicate[name])
    assert case["rows"] == query_len + 3
    assert case["real_rows"] == query_len + 1
    assert case["qsl"][-1] == case["tokens"]
    assert case["qsl"][-1] == case["qsl"][-2]
    assert torch.equal(case["indices"][-2:], torch.zeros(2, dtype=torch.int32))
    assert case["q"].dtype == case["k"].dtype == torch.bfloat16
    assert case["state"].dtype == torch.float32
    assert torch.isfinite(case["q"]).all()


def _header_fields(macro):
    text = (SOURCE.parent / "csrc" / "common.h").read_text()
    section = text.split("#define " + macro + "(X)", 1)[1].split("#define", 1)[0]
    return tuple(re.findall(r"X\((\w+)\)", section))


def test_python_config_order_exactly_matches_host_device_header():
    assert oracle.VERIFY_FIELDS == _header_fields("GDN_VERIFY_FIELDS")
    assert oracle.FOLD_FIELDS == _header_fields("GDN_FOLD_FIELDS")
    case = oracle.make_case(4, torch.bfloat16, torch.float32, torch.bfloat16)
    rows = case["rows"]
    corr = torch.zeros(rows, 3, 4, 128)
    key = torch.zeros(rows, 1, 4, 128, dtype=torch.bfloat16)
    decay = torch.zeros(rows, 3, 4)
    out = torch.zeros_like(case["v"])
    tensors = tuple(case[n] for n in ("q", "k", "v", "g", "beta", "state"))
    tensors += (corr, key, decay, out, case["qsl"], case["indices"])
    values = dict(zip(oracle.VERIFY_FIELDS, oracle.verify_config(case, tensors)))
    assert values["heads"] == 1
    assert values["value_heads"] == 3
    assert values["rows"] == rows
    assert values["state_block_stride"] == case["state"].stride(0)
    assert values["key_token_stride"] == 128
    assert values["index_stride"] == 1
    fold = dict(
        zip(
            oracle.FOLD_FIELDS,
            oracle.fold_config(rows, 2, rows, case["state"], corr, key, decay),
        )
    )
    assert fold["layers"] == 2
    assert fold["plan_capacity"] == rows
    assert fold["key_heads"] == 1


def test_numerical_guard_rejects_real_errors_and_nonfinite_values():
    oracle._assert_close(
        torch.ones(4), torch.ones(4), label="test", dtype=torch.float32
    )
    with pytest.raises(RuntimeError, match="relative_error"):
        oracle._assert_close(
            torch.zeros(4), torch.ones(4), label="test", dtype=torch.float32
        )
    with pytest.raises(RuntimeError, match="nonfinite"):
        oracle._assert_close(
            torch.tensor([float("nan")]),
            torch.ones(1),
            label="test",
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="BF16/FP32"):
        oracle.make_case(4, torch.float16, torch.float32, torch.float32)
