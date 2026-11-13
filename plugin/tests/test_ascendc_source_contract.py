# SPDX-License-Identifier: Apache-2.0
"""Check the ABI crossing native Triton specs, Python dispatch and C++ schemas.

These are interface checks, not claims of AscendC compilation or arithmetic.
The spec arguments come from the actual Triton function signatures so a future
argument insertion cannot silently shift a positional AscendC adapter.
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PACKAGE = Path(__file__).parents[1] / "gdn_recoverssm_ascend"


@pytest.fixture
def backend(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_ac_source_backend", PACKAGE / "ascendc_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _fields(name):
    text = (PACKAGE / "csrc/common.h").read_text()
    macro = re.search(rf"#define GDN_{name}_FIELDS\(X\)(.*?)(?=\n#define)", text, re.S)
    assert macro is not None
    return re.findall(r"X\((\w+)\)", macro.group(1))


def _native_spec(filename, function):
    tree = ast.parse((PACKAGE / filename).read_text())
    fn = next(
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == function
    )
    names = [x.arg for x in fn.args.args if x.annotation is None]
    # Distinct CPU-known scalar sentinels detect shifted config positions.
    arguments = {name: 1000 + index for index, name in enumerate(names)}
    return names, arguments


def _tensor(dtype=torch.float32, shape=(128, 12, 4, 128), count=48):
    return SimpleNamespace(
        dtype=dtype,
        shape=shape,
        numel=lambda: count,
        element_size=lambda: torch.empty((), dtype=dtype).element_size(),
    )


def test_verify_native_arguments_match_cpp_tiling_field_order(backend):
    names, a = _native_spec("kernels.py", "_gdn_recoverssm_verify_kernel")
    for name in names[:12]:
        a[name] = _tensor()
    a["query_start_loc_ptr"] = _tensor(torch.int32)
    a["state_indices_ptr"] = _tensor(torch.int32)
    for name in (
        "stride_state_k",
        "stride_replay_dim",
        "stride_key_dim",
        "stride_decay_pos",
    ):
        a[name] = 1
    kw = dict(
        RECORD_SOA=True,
        USE_QK_L2NORM=False,
        K=128,
        V=128,
        H=4,
        HV=12,
        SPEC_QUERY_LEN=4,
        NUM_BLOCKS=8192,
    )
    spec = SimpleNamespace(args=tuple(a[name] for name in names), kwargs=kw)
    result = backend._verify_spec(spec)
    assert result is not None
    expected = {
        "rows": a["num_rows"],
        "tokens": a["total_tokens"],
        "heads": kw["H"],
        "value_heads": kw["HV"],
        "query_len": kw["SPEC_QUERY_LEN"],
        "blocks": kw["NUM_BLOCKS"],
        "record_rows": a["num_record_rows"],
        "null_block": a["null_block_id"],
        "q_token_stride": a["stride_q_token"],
        "k_token_stride": a["stride_k_token"],
        "v_token_stride": a["stride_v_token"],
        "g_token_stride": a["stride_g_token"],
        "beta_token_stride": a["stride_beta_token"],
        "state_block_stride": a["stride_state_block"],
        "state_head_stride": a["stride_state_head"],
        "state_v_stride": a["stride_state_v"],
        "corr_row_stride": a["stride_replay_block"],
        "corr_head_stride": a["stride_replay_head"],
        "corr_token_stride": a["stride_replay_pos"],
        "key_row_stride": a["stride_key_row"],
        "key_head_stride": a["stride_key_head"],
        "key_token_stride": a["stride_key_pos"],
        "decay_row_stride": a["stride_decay_row"],
        "decay_head_stride": a["stride_decay_head"],
        "decay_token_stride": a["stride_decay_pos"],
        "out_token_stride": a["stride_out_token"],
        "qsl_stride": a["stride_qsl"],
        "index_stride": a["stride_si"],
    }
    assert dict(zip(_fields("VERIFY"), result.config, strict=True)) == expected


def test_fold_native_arguments_match_cpp_tiling_and_tensor_order(backend):
    names, a = _native_spec("grouped_commit.py", "_grouped_commit_gdn_state_kernel")
    for name in names[:18]:
        a[name] = _tensor()
    for name in (
        "stride_state_k",
        "stride_replay_dim",
        "stride_key_dim",
        "stride_decay_pos",
        "stride_si",
    ):
        a[name] = 1
    kw = dict(
        RECORD_SOA=True, K=128, V=128, NUM_HEADS=12, NUM_KEY_HEADS=4, NUM_BLOCKS=8192
    )
    spec = SimpleNamespace(args=tuple(a[name] for name in names), kwargs=kw)
    result = backend._fold_spec(spec)
    assert result is not None
    expected = {
        "rows": a["num_rows"],
        "layers": a["state_base_addrs_ptr"].numel(),
        "heads": kw["NUM_HEADS"],
        "key_heads": kw["NUM_KEY_HEADS"],
        "query_len": a["replay_ref_ptr"].shape[2],
        "blocks": kw["NUM_BLOCKS"],
        "record_rows": a["num_record_rows"],
        "plan_capacity": a["plan_capacity"],
        "null_block": a["null_block_id"],
        "state_head_stride": a["stride_state_head"],
        "state_v_stride": a["stride_state_v"],
        "corr_head_stride": a["stride_replay_head"],
        "corr_token_stride": a["stride_replay_pos"],
        "key_head_stride": a["stride_key_head"],
        "key_token_stride": a["stride_key_pos"],
        "decay_head_stride": a["stride_decay_head"],
        "decay_token_stride": a["stride_decay_pos"],
    }
    assert dict(zip(_fields("FOLD"), result.config, strict=True)) == expected
    schemas = _schemas()
    schema_names = [
        x.name for x in schemas["fold"].arguments if str(x.type) == "Tensor"
    ]
    tensor_mapping = {
        "state": "state_ref_ptr",
        "correction": "replay_ref_ptr",
        "key": "key_ref_ptr",
        "decay": "decay_ref_ptr",
        "state_addrs": "state_base_addrs_ptr",
        "state_block_strides": "state_block_strides_ptr",
        "corr_addrs": "replay_base_addrs_ptr",
        "corr_block_strides": "replay_block_strides_ptr",
        "key_addrs": "key_base_addrs_ptr",
        "key_block_strides": "key_block_strides_ptr",
        "decay_addrs": "decay_base_addrs_ptr",
        "decay_block_strides": "decay_block_strides_ptr",
        "layer_groups": "layer_groups_ptr",
        "source": "state_indices_ptr",
        "lengths": "commit_lens_ptr",
        "final_indices": "final_state_indices_ptr",
        "boundary_indices": "boundary_state_indices_ptr",
        "boundary_lengths": "boundary_recovery_lens_ptr",
    }
    assert len(result.tensors) == len(schema_names)
    assert all(
        tensor is a[tensor_mapping[name]]
        for tensor, name in zip(result.tensors, schema_names)
    )


def _schemas():
    text = (PACKAGE / "csrc/ops.cpp").read_text()
    literals = re.findall(r'm\.def\("([^"\n]+)"', text)
    schemas = [torch._C.parse_schema(item) for item in literals]
    return {schema.name: schema for schema in schemas}


def test_dispatcher_schema_marks_verify_outputs_and_keeps_checkpoint_readonly():
    schema = _schemas()["verify"]
    args = {arg.name: arg for arg in schema.arguments}
    assert not schema.returns
    assert args["state"].alias_info is None
    assert args["q"].alias_info is None
    assert {
        name for name, arg in args.items() if arg.alias_info and arg.alias_info.is_write
    } == {"correction", "key", "decay", "out"}
    assert str(args["config"].type) == "List[int]"
    assert str(args["scale"].type) == "float"


def test_dispatcher_schema_marks_fold_state_as_mutable_and_plan_readonly():
    schema = _schemas()["fold"]
    args = {arg.name: arg for arg in schema.arguments}
    assert not schema.returns
    assert args["state"].alias_info.is_write
    assert all(
        args[name].alias_info is None
        for name in (
            "source",
            "lengths",
            "final_indices",
            "boundary_indices",
            "boundary_lengths",
        )
    )
