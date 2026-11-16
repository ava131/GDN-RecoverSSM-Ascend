# SPDX-License-Identifier: Apache-2.0
"""Exercise the production builder against the pinned native buffer contract.

The native builder owns and rewrites its padding buffer. The plugin's neutral
convolution descriptor is a separate, stable, read-only tensor. CPU tensors
make version/storage checks meaningful without importing an NPU-only package.
"""

import ast
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@dataclass
class NativeMetadata:
    num_spec_decodes: int = 0
    spec_state_indices_tensor: object = None
    spec_query_start_loc: object = None
    num_accepted_tokens: object = None


class NativeBuilder:
    def __init__(self, spec, names, config, device):
        self.vllm_config = config
        self.num_spec = 3
        self.decode_cudagraph_max_bs = 16
        self.num_accepted_tokens = torch.empty(16, dtype=torch.int32)
        self.spec_sequence_indices = torch.arange(16, dtype=torch.int32)
        self.spec_sequence_indices_cpu = self.spec_sequence_indices.clone()

    def build(self, common_prefix_len, common, **kwargs):
        # rc1 native _pad_spec_decode_metadata writes the native-owned buffer;
        # it does not write through a descriptor returned on a previous step.
        self.num_accepted_tokens.fill_(3)
        n = common.rows
        meta = NativeMetadata(
            num_spec_decodes=n,
            spec_state_indices_tensor=torch.ones((n, 4), dtype=torch.int32),
            spec_query_start_loc=torch.arange(n + 1, dtype=torch.int32) * 4,
            num_accepted_tokens=self.num_accepted_tokens[:n],
        )
        meta.spec_decode_metadata = SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(
                num_accepted_tokens=self.num_accepted_tokens[:n]
            )
        )
        return meta


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / "gdn_recoverssm_ascend/builder.py"
    tree = ast.parse(path.read_text())
    names = {
        "AscendGDNRecoverSSMCommitMetadata",
        "AscendGDNRecoverSSMMetadata",
        "_native_spec_rows",
        "AscendGDNRecoverSSMMetadataBuilder",
    }
    # Compile the production definitions, with only their native dependencies
    # substituted. The actual build/__init__ bodies execute below.
    nodes = ast.parse("from __future__ import annotations").body
    nodes += [n for n in tree.body if getattr(n, "name", None) in names]
    namespace = {
        "torch": torch,
        "dataclass": dataclass,
        "field": field,
        "fields": fields,
        "GDNAttentionMetadata": NativeMetadata,
        "RecoverSSMMetadata": type("RecoverSSMMetadata", (), {}),
        "AscendGDNAttentionMetadataBuilder": NativeBuilder,
        "AttentionCGSupport": SimpleNamespace(UNIFORM_BATCH="uniform"),
    }
    # The only executable input is the checked-in production source above.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    obj = namespace["AscendGDNRecoverSSMMetadataBuilder"](
        None, [], config, torch.device("cpu")
    )
    context = object()
    obj._get_context = lambda: context
    return obj


def test_neutral_descriptor_survives_native_padding_and_batch_changes(builder):
    address = builder._accepted_buffer.data_ptr()
    version = builder._accepted_buffer._version
    for n in (1, 8, 3, 16, 2):
        common = SimpleNamespace(rows=n, query_start_loc_cpu=torch.arange(n + 1) * 4)
        meta = builder.build(0, common)
        top = meta.num_accepted_tokens
        nested = meta.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens
        assert top is nested
        assert top.data_ptr() == address
        assert top.data_ptr() != builder.num_accepted_tokens.data_ptr()
        torch.testing.assert_close(top, torch.ones(n, dtype=torch.int32))
        torch.testing.assert_close(
            builder.num_accepted_tokens, torch.full((16,), 3, dtype=torch.int32)
        )
        # No in-place refresh and no replacement of the graph-owned storage.
        assert builder._accepted_buffer._version == version
        assert meta.recoverssm_context is builder._get_context()


def test_oversize_does_not_replace_captured_neutral_storage(builder):
    address = builder._accepted_buffer.data_ptr()
    common = SimpleNamespace(rows=17, query_start_loc_cpu=torch.arange(18) * 4)
    with pytest.raises(RuntimeError, match="exceeds the 16-row accepted buffer"):
        builder.build(0, common)
    assert builder._accepted_buffer.data_ptr() == address
    torch.testing.assert_close(
        builder._accepted_buffer, torch.ones(16, dtype=torch.int32)
    )
