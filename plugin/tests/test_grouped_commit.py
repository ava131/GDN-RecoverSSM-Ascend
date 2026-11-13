# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the real plan and binding checks for all grouped launches."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend"


class _Ptr:
    registry = {}

    def __init__(self, tensor, offset=0):
        size = tensor.untyped_storage().nbytes() // tensor.element_size()
        self.flat = torch.as_strided(tensor, (size,), (1,), storage_offset=0)
        self.offset = tensor.storage_offset() + int(offset)
        self.registry[tensor.data_ptr()] = self

    def __add__(self, offset):
        result = object.__new__(_Ptr)
        result.flat = self.flat
        result.offset = self.offset + int(offset)
        return result

    __radd__ = __add__


class _TL:
    constexpr = object()
    int1 = torch.bool
    int32 = torch.int32
    int64 = torch.int64
    float32 = torch.float32
    row = 0

    def program_id(self, dim):
        return self.row

    @staticmethod
    def full(shape, value, dtype):
        return torch.full(shape, value, dtype=dtype)

    @staticmethod
    def static_range(n):
        return range(n)

    @staticmethod
    def pointer_type(dtype):
        return ("pointer", dtype)

    @staticmethod
    def load(ptr, mask=True, other=0):
        value = ptr.flat[ptr.offset] if bool(mask) else torch.tensor(other)
        if (
            value.ndim == 0
            and value.dtype == torch.int64
            and int(value) in _Ptr.registry
        ):

            class Address:
                def to(self, dtype):
                    if isinstance(dtype, tuple) and dtype[0] == "pointer":
                        return _Ptr.registry[int(value)]
                    return value.to(dtype)

            return Address()
        return value

    @staticmethod
    def store(ptr, value, mask=True):
        if bool(mask):
            ptr.flat[ptr.offset] = value

    @staticmethod
    def minimum(a, b):
        return torch.minimum(torch.as_tensor(a), torch.as_tensor(b))

    @staticmethod
    def maximum(a, b):
        return torch.maximum(torch.as_tensor(a), torch.as_tensor(b))

    where = staticmethod(torch.where)


def _load(monkeypatch):
    tl = _TL()
    triton = types.SimpleNamespace(
        jit=lambda *a, **kw: a[0] if a else lambda fn: fn,
        cdiv=lambda a, b: (a + b - 1) // b,
        next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length(),
    )
    parent = types.ModuleType("gdn_recoverssm_ascend")
    parent.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    for name in ("vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    log = types.ModuleType("vllm.logger")
    log.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, log.__name__, log)
    utils = types.ModuleType("vllm.triton_utils")
    utils.tl, utils.triton = tl, triton
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    null = types.ModuleType("vllm.v1.attention.backends.utils")
    null.NULL_BLOCK_ID = 0
    monkeypatch.setitem(sys.modules, null.__name__, null)
    kernels = types.ModuleType("gdn_recoverssm_ascend.kernels")
    calls = []

    class Launch:
        def __init__(self, *, kernel, args, kwargs, grid):
            self.kernel, self.args, self.kwargs, self.grid = kernel, args, kwargs, grid
            params = {k: v for k, v in kwargs.items() if k != "num_warps"}
            inspect.signature(kernel).bind(*args, **params)

        def launch(self):
            calls.append(self)

        def compile_only(self):
            calls.append(self)

    kernels._LaunchSpec = Launch
    kernels._commit_value_tile = lambda k, v: min(16, v)
    kernels._conv_dim_tile = lambda h: 256
    kernels._launch_cap = lambda: 48
    kernels._ptr_table = lambda ts, dev: torch.tensor(
        [t.data_ptr() for t in ts], dtype=torch.int64
    )
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    spec = importlib.util.spec_from_file_location(
        "gdn_recoverssm_ascend.grouped_commit", ROOT / "grouped_commit.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, tl, calls


def _context(layers=2, dtype=torch.bfloat16, blocks=64):
    states = tuple(torch.empty(blocks, 6, 8, 8, dtype=dtype) for _ in range(layers))
    conv = tuple(torch.empty(blocks, 32, 6, dtype=dtype) for _ in range(layers))
    records = tuple(
        types.SimpleNamespace(
            correction=torch.empty(8, 6, 4, 8),
            key=torch.empty(8, 2, 4, 8, dtype=dtype),
            decay=torch.empty(8, 6, 4),
        )
        for _ in range(layers)
    )
    return types.SimpleNamespace(
        checkpoints=states,
        conv_states=conv,
        records=records,
        record_rows=8,
        conv_history_len=3,
        spec_query_len=4,
        record_layout="soa",
        num_key_heads=2,
        num_blocks=blocks,
    )


def _entry(module, context, group=0, align=True, sources=(8, 4)):
    table = torch.zeros(2, 4, dtype=torch.int32)
    table[:, 0] = torch.tensor(sources, dtype=torch.int32)
    bt = (
        torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]], dtype=torch.int32)
        + group * 20
    )
    p = types.SimpleNamespace(
        state_indices=table[:, 0],
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        request_indices=torch.tensor([2, 0], dtype=torch.int64),
        block_table=bt if align else None,
        num_computed_tokens=torch.tensor([3, 0, 7], dtype=torch.int32)
        if align
        else None,
        block_size=4 if align else None,
        request_rows=(2, 0),
        query_lens=(4, 4),
    )
    builder = types.SimpleNamespace(_get_context=lambda: context)
    meta = types.SimpleNamespace(
        recoverssm_commit=p, recoverssm_builder=builder, recoverssm_context=context
    )
    return module.CommitEntry(meta, p, context)


def _inputs():
    return (
        torch.tensor([1, 0, 3], dtype=torch.int32),
        torch.tensor([5, -1, 1], dtype=torch.int32),
        torch.full((6,), -9, dtype=torch.int32),
        torch.full((6,), 7, dtype=torch.int32),
    )


def _execute_plan(tl, spec):
    def ptr(x):
        if isinstance(x, torch.Tensor):
            return _Ptr(x)
        if isinstance(x, tuple):
            return tuple(ptr(y) for y in x)
        return x

    args = tuple(ptr(x) for x in spec.args)
    kwargs = {k: v for k, v in spec.kwargs.items() if k != "num_warps"}
    for row in range(spec.grid[0]):
        tl.row = row
        spec.kernel(*args, **kwargs)


def test_three_launches_bind_all_arguments_and_keep_physical_tables(monkeypatch):
    module, tl, calls = _load(monkeypatch)
    c0, c1, c2 = _context(), _context(), _context()
    entries = [_entry(module, c, group=g) for g, c in enumerate((c0, c1, c2))]
    # Group2's physical IDs need a larger group-local pool, independent of others.
    bucket = module.GroupedCommitBucket((c0, c1, c2))
    sampled, mapping, cols, accepted = _inputs()
    specs = bucket.make_specs(entries, sampled, mapping, cols, accepted)
    assert len(specs) == 3
    assert [x.kernel.__name__ for x in specs] == [
        "_grouped_plan_kernel",
        "_grouped_compact_conv_state_kernel",
        "_grouped_commit_gdn_state_kernel",
    ]
    _execute_plan(tl, specs[0])
    for group in range(3):
        start = group * bucket.capacity
        assert bucket.final[start : start + 2].tolist() == [
            12 + 20 * group,
            2 + 20 * group,
        ]
        assert bucket.boundary[start].item() == 11 + 20 * group
        assert bucket.boundary_len[start].item() == 1
        assert bucket.length[start : start + 2].tolist() == [3, 1]
    assert cols.tolist() == [-9, 2, -9, -9, -9, 0]
    assert accepted.tolist() == [7, 1, 7, 7, 7, 1]
    assert bucket.layer_groups.tolist() == [0, 0, 1, 1, 2, 2]
    assert bucket.key_addrs.numel() == 6
    for spec in specs:
        spec.launch()
    assert len(calls) == 3


def test_invalid_group_cannot_publish_shared_running_column(monkeypatch):
    module, tl, _ = _load(monkeypatch)
    c0, c1 = _context(), _context()
    entries = [_entry(module, c0), _entry(module, c1, group=1, sources=(0, 9))]
    bucket = module.GroupedCommitBucket((c0, c1))
    sampled, mapping, cols, accepted = _inputs()
    _execute_plan(tl, bucket.make_specs(entries, sampled, mapping, cols, accepted)[0])
    assert bucket.length[bucket.capacity].item() == 0
    assert cols[1].item() == -9 and accepted[1].item() == 7
    assert cols[5].item() == 0 and accepted[5].item() == 1


def test_none_plan_never_publishes_or_touches_absent_tables(monkeypatch):
    module, tl, _ = _load(monkeypatch)
    ctx = _context()
    entry = _entry(module, ctx, align=False)
    bucket = module.GroupedCommitBucket((ctx,))
    sampled, mapping, cols, accepted = _inputs()
    specs = bucket.make_specs((entry,), sampled, mapping, None, accepted)
    _execute_plan(tl, specs[0])
    assert bucket.final[:2].tolist() == [8, 4]
    assert bucket.length[:2].tolist() == [3, 1]
    assert accepted.tolist() == [7] * 6


def test_zero_acceptance_and_finished_slots_are_noops(monkeypatch):
    module, tl, _ = _load(monkeypatch)
    ctx = _context()
    entry = _entry(module, ctx)
    bucket = module.GroupedCommitBucket((ctx,))
    sampled, mapping, cols, accepted = _inputs()
    sampled.zero_()
    mapping.fill_(-1)
    _execute_plan(tl, bucket.make_specs((entry,), sampled, mapping, cols, accepted)[0])
    assert bucket.length[:2].tolist() == [0, 0]
    assert cols.tolist() == [-9] * 6
    assert accepted.tolist() == [7] * 6


def test_coordinator_reuses_epoch_and_buckets_heterogeneous_dtypes(monkeypatch):
    module, _, calls = _load(monkeypatch)
    contexts = (_context(), _context(), _context())
    entries = [_entry(module, c, g) for g, c in enumerate(contexts)]
    coordinator = module.GroupedCommitCoordinator()
    args = _inputs()
    coordinator.commit([e.metadata for e in entries], *args)
    assert coordinator.last_group_count == 3 and coordinator.last_launch_count == 3
    first = next(iter(coordinator._buckets.values()))
    coordinator.commit([e.metadata for e in entries], *args)
    assert next(iter(coordinator._buckets.values())) is first
    replacement = _entry(module, _context(dtype=torch.float32), group=1)
    coordinator.commit([entries[0].metadata, replacement.metadata], *args)
    assert coordinator.last_launch_count == 7
    assert len(coordinator._buckets) == 2
    assert first not in coordinator._buckets.values()
    assert len(calls) == 13


def test_aliasing_sampled_output_and_wrong_epoch_fail_before_launch(monkeypatch):
    module, _, calls = _load(monkeypatch)
    ctx = _context()
    entry = _entry(module, ctx)
    bucket = module.GroupedCommitBucket((ctx,))
    sampled, mapping, cols, accepted = _inputs()
    with pytest.raises(ValueError, match="must not alias"):
        bucket.make_specs((entry,), sampled, mapping, cols, sampled)
    other = _entry(module, _context())
    with pytest.raises(ValueError, match="different KV epoch"):
        bucket.make_specs((other,), sampled, mapping, cols, accepted)
    assert not calls


def test_heterogeneous_publication_waits_for_all_layout_buckets(monkeypatch):
    module, tl, calls = _load(monkeypatch)
    entries = [
        _entry(module, _context()),
        _entry(module, _context(dtype=torch.float32), 1, sources=(0, 9)),
    ]
    coordinator = module.GroupedCommitCoordinator()
    sampled, mapping, cols, accepted = _inputs()
    coordinator.commit([e.metadata for e in entries], sampled, mapping, cols, accepted)
    assert len(calls) == 7
    for spec in calls[:-1]:
        if spec.kernel.__name__ == "_grouped_plan_kernel":
            assert spec.kwargs["PUBLISH_ALIGN"] is False
            _execute_plan(tl, spec)
    _execute_plan(tl, calls[-1])
    assert cols[1].item() == -9 and accepted[1].item() == 7
    assert cols[5].item() == 0 and accepted[5].item() == 1


def test_conflicting_logical_align_owners_are_rejected_before_launch(monkeypatch):
    module, _, calls = _load(monkeypatch)
    entries = [_entry(module, _context()), _entry(module, _context(), 1)]
    entries[1].payload.block_size = 8
    with pytest.raises(ValueError, match="must agree"):
        module.GroupedCommitCoordinator().commit(
            [e.metadata for e in entries], *_inputs()
        )
    assert not calls


def test_commit_uses_producer_snapshot_without_late_context_resolution(monkeypatch):
    module, _, _ = _load(monkeypatch)
    old = _context()
    entry = _entry(module, old)

    def resolve_late():
        raise AssertionError("commit resolved a different epoch after verify")

    entry.metadata.recoverssm_builder._get_context = resolve_late
    coordinator = module.GroupedCommitCoordinator()
    coordinator.commit([entry.metadata], *_inputs())
    assert next(iter(coordinator._buckets.values())).contexts[0] is old


def test_prepare_borrows_exact_bound_native_table_layout(monkeypatch):
    module, _, calls = _load(monkeypatch)
    ctx = _context()
    table_storage = torch.empty((8, 40), dtype=torch.int32)
    native_table = table_storage[:, :37]
    config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(mamba_cache_mode="align"),
        model_config=types.SimpleNamespace(max_model_len=262144),
    )
    builder = types.SimpleNamespace(
        _get_context=lambda: ctx,
        _gdn_kv_cache_group_id=0,
        vllm_config=config,
        kv_cache_spec=types.SimpleNamespace(block_size=4),
        spec_sequence_indices=torch.empty(8, dtype=torch.int64),
    )
    assert (
        module.prepare_grouped_commit(
            [builder],
            types.SimpleNamespace(input_block_tables=[native_table]),
            num_computed_tokens=torch.empty(8, dtype=torch.int32),
        )
        == 3
    )
    plan = calls[0]
    bound = (
        inspect.signature(plan.kernel)
        .bind(*plan.args, **{k: v for k, v in plan.kwargs.items() if k != "num_warps"})
        .arguments
    )
    assert bound["table0"].data_ptr() == native_table.data_ptr()
    assert bound["table_width0"] == 37 and bound["table_row_stride0"] == 40
    assert plan.kwargs["ALIGN_MODE"] is True
    assert bound["mapping0"].dtype == torch.int64
    assert builder._prepared_group_bucket.contexts[0] is ctx


def _builder_methods(*names):
    import ast

    source = (ROOT / "builder.py").read_text()
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "AscendGDNRecoverSSMMetadataBuilder"
    )
    return {
        name: ast.get_source_segment(
            source,
            next(
                n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
            ),
        )
        for name in names
    }


def test_native_mapping_is_a_view_without_any_upload(monkeypatch):
    import ast

    src = (ROOT / "builder.py").read_text()
    helper = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_native_spec_rows"
    )
    namespace = {}
    exec(ast.get_source_segment(src, helper), namespace)
    native = torch.tensor([3, 1, 4, 0], dtype=torch.int64)
    builder = types.SimpleNamespace(
        spec_sequence_indices=native, spec_sequence_indices_cpu=native.clone()
    )
    rows, host = namespace["_native_spec_rows"](builder, 3, native.device)
    assert rows.data_ptr() == native.data_ptr()
    assert rows.dtype == torch.int64 and host == (3, 1, 4)
    with pytest.raises(RuntimeError, match="requires the native"):
        namespace["_native_spec_rows"](types.SimpleNamespace(), 2, native.device)


def test_context_guard_detects_inplace_container_rebind_and_layout(monkeypatch):
    module, _, _ = _load(monkeypatch)
    kernels = sys.modules["gdn_recoverssm_ascend.kernels"]
    made = []

    def create(layers, **kwargs):
        context = types.SimpleNamespace(
            records=[types.SimpleNamespace(shape=(8, 6, 4, 17))],
            checkpoints=tuple(x.kv_cache[1] for x in layers),
        )
        made.append(context)
        return context

    kernels.GDNRecoverSSMCommitContext = types.SimpleNamespace(create=create)
    methods = _builder_methods("_get_context", "_record_key_heads")
    namespace = {
        "logger": logging.getLogger("test"),
        "torch": torch,
        "__name__": "gdn_recoverssm_ascend.builder",
        "__package__": "gdn_recoverssm_ascend",
    }
    for source in methods.values():
        exec("from __future__ import annotations\n" + source, namespace)
    cls = type("Builder", (), {name: namespace[name] for name in methods})
    builder = cls()
    layer = types.SimpleNamespace(
        kv_cache=[torch.empty(8, 32, 6), torch.empty(8, 6, 8, 8)]
    )
    builder.vllm_config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"layer": layer}
        ),
        scheduler_config=types.SimpleNamespace(max_num_seqs=8),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
        model_config=types.SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=types.SimpleNamespace(linear_num_key_heads=2),
        ),
    )
    builder.layer_names, builder.num_spec = ["layer"], 3
    builder._context = builder._ctx_key = None
    builder._ctx_unbound_logged, builder._capturing = False, False
    first = builder._get_context()
    assert builder._get_context() is first
    container = layer.kv_cache
    layer.kv_cache[1] = torch.empty_like(layer.kv_cache[1])
    second = builder._get_context()
    assert second is not first and layer.kv_cache is container
    layer.kv_cache[1].transpose_(2, 3)
    third = builder._get_context()
    assert third is not second
    builder._gdn_serving_ready = builder.use_full_cuda_graph = True
    builder._captured_context = third
    layer.kv_cache[1] = torch.empty_like(layer.kv_cache[1])
    with pytest.raises(RuntimeError, match="invalidated and recaptured"):
        builder._get_context()
    assert len(made) == 3


def test_build_resolves_producer_context_before_returning_verify_metadata(monkeypatch):
    import ast
    from dataclasses import dataclass, fields

    @dataclass
    class NativeMeta:
        num_spec_decodes: int
        spec_state_indices_tensor: object
        spec_query_start_loc: object
        num_accepted_tokens: object = None
        spec_decode_metadata: object = None

    events = []
    native_meta = NativeMeta(
        2,
        torch.tensor([[2, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.int32),
        torch.tensor([0, 4, 8], dtype=torch.int32),
    )

    class Native:
        def build(self, *args, **kwargs):
            events.append("native_metadata")
            return native_meta

    source = (ROOT / "builder.py").read_text()
    tree = ast.parse(source)
    helper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_native_spec_rows"
    )
    namespace = {
        "torch": torch,
        "Native": Native,
        "fields": fields,
        "AscendGDNRecoverSSMCommitMetadata": types.SimpleNamespace,
        "AscendGDNRecoverSSMMetadata": types.SimpleNamespace,
    }
    exec(ast.get_source_segment(source, helper), namespace)
    method = _builder_methods("build")["build"]
    generated = "from __future__ import annotations\nclass Builder(Native):\n"
    generated += "    " + method.replace("\n", "\n    ") + "\n"
    exec(generated, namespace)
    builder = namespace["Builder"]()
    old_context = object()

    def resolve():
        events.append("resolve_context")
        return old_context

    builder._get_context = resolve
    builder._accepted_buffer = torch.ones(4, dtype=torch.int32)
    builder.spec_sequence_indices = torch.tensor([1, 0], dtype=torch.int64)
    builder.spec_sequence_indices_cpu = builder.spec_sequence_indices.clone()
    builder.vllm_config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(mamba_cache_mode="align")
    )
    builder.kv_cache_spec = types.SimpleNamespace(block_size=4)
    common = types.SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        block_table_tensor=torch.zeros(2, 3, dtype=torch.int32),
        compute_num_computed_tokens=lambda: torch.zeros(2, dtype=torch.int32),
    )
    result = builder.build(0, common)
    assert result.recoverssm_context is old_context
    assert (
        result.recoverssm_commit.request_indices.data_ptr()
        == builder.spec_sequence_indices.data_ptr()
    )
    assert result.recoverssm_commit.request_rows == (1, 0)
    assert result.recoverssm_commit.query_lens == (4, 4)
    assert events == ["native_metadata", "resolve_context"]


def test_factory_retains_native_host_metadata_capability(monkeypatch):
    import ast

    for name in ("vllm_ascend", "vllm_ascend.ops"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    native = types.ModuleType("vllm_ascend.ops.gdn_attn_builder")
    selected = types.SimpleNamespace(build_device_chunk_metadata=False)
    native.AscendGDNFusedAttentionBackend = types.SimpleNamespace(
        get_builder_cls=lambda: selected
    )
    monkeypatch.setitem(sys.modules, native.__name__, native)
    sys.modules["vllm_ascend.ops"].gdn_attn_builder = native
    source = (ROOT / "builder.py").read_text()
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendGDNRecoverSSMBackend"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_builder_cls"
    )
    host, fallback = type("Host", (), {}), type("Fallback", (), {})
    namespace = {
        "AscendGDNRecoverSSMHostMetadataBuilder": host,
        "AscendGDNRecoverSSMMetadataBuilder": fallback,
    }
    exec(
        "from __future__ import annotations\n" + ast.get_source_segment(source, method),
        namespace,
    )
    factory = namespace["get_builder_cls"]
    assert factory() is host
    selected.build_device_chunk_metadata = True
    assert factory() is fallback


def test_group_local_invalid_boundary_is_not_legalized_by_larger_pool(monkeypatch):
    module, tl, _ = _load(monkeypatch)
    small, large = _context(blocks=16), _context(blocks=64)
    entries = [_entry(module, small), _entry(module, large, 1)]
    entries[0].payload.block_table[2, 1] = 32  # invalid only in the small pool
    bucket = module.GroupedCommitBucket((small, large))
    _execute_plan(tl, bucket.make_specs(entries, *_inputs())[0])
    assert bucket.final[0].item() == 12
    assert bucket.boundary[0].item() == 0
    assert bucket.boundary_len[0].item() == 0
    assert bucket.boundary[bucket.capacity].item() == 31


def test_triton_322_flat_runtime_abi_and_more_than_three_groups(monkeypatch):
    module, tl, calls = _load(monkeypatch)
    entries = [_entry(module, _context(), i % 2) for i in range(7)]
    coordinator = module.GroupedCommitCoordinator()
    sampled, mapping, cols, accepted = _inputs()
    coordinator.commit([e.metadata for e in entries], sampled, mapping, cols, accepted)
    assert coordinator.last_group_count == 7
    assert coordinator.last_launch_count == 10  # three chunks plus one publication
    assert [len(b.contexts) for b in coordinator._buckets.values()] == [3, 3, 1]
    first = next(iter(coordinator._buckets.values()))
    descriptor = first._publisher_descriptors
    for spec in calls:
        signature = inspect.signature(spec.kernel)
        bound = signature.bind(
            *spec.args, **{k: v for k, v in spec.kwargs.items() if k != "num_warps"}
        )
        for name, value in bound.arguments.items():
            parameter = signature.parameters[name]
            if "constexpr" not in str(parameter.annotation):
                # Exact 3.2.2 mangle_type supports tensor/scalar/None, not tuple.
                assert not isinstance(value, (tuple, list)), (name, value)
        if spec.kernel.__name__ == "_grouped_plan_kernel":
            assert spec.kwargs["PUBLISH_ALIGN"] is False
            _execute_plan(tl, spec)
    _execute_plan(tl, calls[-1])
    assert cols.tolist() == [-9, 2, -9, -9, -9, 0]
    coordinator.commit([e.metadata for e in entries], sampled, mapping, cols, accepted)
    assert first._publisher_descriptors is descriptor
    assert calls[-1].args[0].data_ptr() == descriptor.data_ptr()
