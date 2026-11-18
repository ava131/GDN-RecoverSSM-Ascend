# SPDX-License-Identifier: Apache-2.0
"""Targeted in-memory adaptation of the installed Ascend 0.30rc1 GDN core.

Only the speculative subset of a mixed batch changes. The installed native
non-speculative convolution, PCP, prefill kernels and state writes are retained
as AST nodes. Structural checks run before constructing a callable; unsupported
source must never become a partially rewritten forward method.
"""

from __future__ import annotations

import ast
import copy
import functools
import hashlib
import inspect
import linecache
import textwrap
from collections import Counter
from dataclasses import dataclass

_COUNTS = Counter()
_DECLINED = set()
_FALLBACK_COUNTS = Counter()
_PLAN_NAME = "_gdn_recoverssm_mixed_plan"


class UnsupportedCoreError(RuntimeError):
    """The installed native method does not match the reviewed core contract."""


@dataclass(frozen=True)
class MixedPlan:
    spec_indices: object
    non_spec_indices: object
    spec_tokens: int
    non_spec_tokens: int


def counters():
    """Host dispatch counts, not graph execution counts."""
    return dict(_COUNTS)


def fallback_counts():
    """Reasons genuine mixed candidates kept the original helper path."""
    return dict(_FALLBACK_COUNTS)


def _decline(reason):
    _FALLBACK_COUNTS[reason] += 1
    if reason not in _DECLINED:
        _DECLINED.add(reason)
        from vllm.logger import init_logger

        init_logger("vllm.gdn_recoverssm_ascend.mixed").debug(
            "GDN_MIXED eligibility=0 reason=%s decision=BEFORE_CONV", reason
        )


def mixed_preflight(layer, meta, mixed_qkv, a, b, core_out):
    """Select a mixed spec/prefill path using CPU-known tensor metadata only.

    A native metadata builder classifies the non-spec one-token rows as prefill
    when speculative rows are present. We preserve that classification instead
    of inventing a num_decodes + num_spec_decodes combination.
    """
    if (
        meta is None
        or getattr(meta, "spec_sequence_masks", None) is None
        or int(getattr(meta, "num_spec_decodes", 0)) <= 0
        or int(getattr(meta, "num_prefills", 0)) <= 0
    ):
        return None
    if int(getattr(meta, "num_decodes", 0)) != 0:
        _decline("native mixed spec metadata requires num_decodes=0")
        return None

    from . import diagnostics

    timer = diagnostics.timing_start("mixed_preflight_host")
    result = _check_mixed_inputs(layer, meta, mixed_qkv, a, b, core_out)
    diagnostics.timing_end(timer)
    return result


def _check_mixed_inputs(layer, meta, mixed_qkv, a, b, core_out):
    import torch

    spec = getattr(meta, "spec_token_indx", None)
    non_spec = getattr(meta, "non_spec_token_indx", None)
    if not all(isinstance(x, torch.Tensor) for x in (mixed_qkv, a, b, core_out)):
        _decline("inputs are not tensors")
        return None
    for indices in (spec, non_spec):
        if (
            not isinstance(indices, torch.Tensor)
            or indices.ndim != 1
            or indices.dtype != torch.int64
            or not indices.is_contiguous()
            or indices.device != mixed_qkv.device
        ):
            # Native nonzero() token selectors are INT64; unlike index_select,
            # index_copy_ does not generally accept an INT32 selector.
            _decline("token indices must be contiguous INT64 vectors on the input device")
            return None
    actual = int(getattr(meta, "num_actual_tokens", 0))
    if actual <= 0 or mixed_qkv.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        _decline("empty tokens or unsupported input rank")
        return None
    if core_out.ndim != 3:
        _decline("unsupported core output rank")
        return None
    if actual > min(x.shape[0] for x in (mixed_qkv, a, b, core_out)):
        raise ValueError("GDN mixed token count exceeds an input/output buffer")
    spec_tokens, non_spec_tokens = spec.numel(), non_spec.numel()
    if spec_tokens + non_spec_tokens != actual:
        raise ValueError("GDN mixed token indices do not cover num_actual_tokens")
    if not spec_tokens or not non_spec_tokens:
        _decline("empty spec or non-spec subset")
        return None
    conv = getattr(getattr(meta, "spec_decode_metadata", None), "spec_causal_conv1d", None)
    if conv is None or any(
        getattr(conv, name, None) is None
        for name in ("query_start_loc", "cache_indices", "num_accepted_tokens")
    ):
        _decline("missing speculative convolution metadata")
        return None

    from .ascendc_backend import can_fuse_verify

    # The inputs still cover the complete batch at this point. Index-select
    # below creates compact spec inputs; no device read is needed to know their
    # length. Using meta.num_actual_tokens here would include prefill tokens.
    if not can_fuse_verify(
        layer, mixed_qkv, a, b, meta=meta, out=core_out, num_tokens=spec_tokens
    ):
        return None
    _COUNTS["preflight_accepted"] += 1
    return MixedPlan(spec, non_spec, int(spec_tokens), int(non_spec_tokens))


def prepare_mixed_gates(layer, a, b, plan, native_gate, *, record=True):
    """Keep raw spec values while computing native gates for non-spec only."""
    raw_a = a.index_select(0, plan.spec_indices)
    raw_b = b.index_select(0, plan.spec_indices)
    non_a = a.index_select(0, plan.non_spec_indices)
    non_b = b.index_select(0, plan.non_spec_indices)
    g_non, beta_non = native_gate(layer.A_log, non_a, non_b, layer.dt_bias)
    if record:
        _COUNTS["non_spec_gate_calls"] += 1
        _COUNTS["spec_gate_tokens_avoided"] += plan.spec_tokens
        _COUNTS["non_spec_gate_tokens"] += plan.non_spec_tokens
    return raw_a.unsqueeze(0), raw_b.unsqueeze(0), g_non, beta_non


def verify_mixed_spec(layer, q, k, v, raw_a, raw_b, ssm_state):
    """Retain _verify's epoch/records checks and only replace preprocessing."""
    from . import diagnostics
    from .layer_patch import _verify

    timer = diagnostics.timing_start("mixed_verify_submit_host")
    output = _verify(
        layer,
        q.squeeze(0),
        k.squeeze(0),
        v.squeeze(0),
        ssm_state,
        {
            "g": raw_a.squeeze(0),
            "beta": raw_b.squeeze(0),
            "scale": k.shape[-1] ** -0.5,
            "raw_fused": (layer.A_log, layer.dt_bias),
            "raw_fused_scope": "mixed",
        },
    )
    diagnostics.timing_end(timer)
    _COUNTS["verify_model_calls"] += 1
    return output.unsqueeze(0)


def merge_mixed_output(
    core_out, actual, spec_indices, non_spec_indices, spec_out, non_spec_out, *, record=True
):
    """Write the original partition directly into the caller-owned output."""
    target = core_out[:actual].unsqueeze(0)
    target.index_copy_(1, spec_indices, spec_out)
    target.index_copy_(1, non_spec_indices, non_spec_out)
    if record:
        _COUNTS["merged_output_copies_avoided"] += 1


def _nodes(source):
    return ast.parse(textwrap.dedent(source)).body


def _same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _find_block(body, expected, label):
    matches = [
        index
        for index in range(len(body) - len(expected) + 1)
        if all(_same(body[index + offset], node) for offset, node in enumerate(expected))
    ]
    if len(matches) != 1:
        raise UnsupportedCoreError(
            f"GDN mixed core source contract {label}: expected one block, found {len(matches)}"
        )
    return matches[0]


def _instrument_native_prefill(function, prefix):
    """Bracket the original rc1 stages in place, with a safe skip on other source."""
    pattern = _nodes("if fla_gdn_prefill_op is not None: pass")[0]
    candidates = [
        node for node in ast.walk(function)
        if isinstance(node, ast.If) and _same(node.test, pattern.test)
    ]
    if len(candidates) != 1:
        return False
    choice = candidates[0]
    if len(choice.orelse) != 1 or not isinstance(choice.orelse[0], ast.If):
        return False
    chunk = choice.orelse[0]
    if not _same(chunk.test, ast.Name(id="use_fused_chunk", ctx=ast.Load())):
        return False
    stages = ("prefill_state_load_host", "prefill_compute_submit_host", "prefill_state_store_host")
    branches = (
        ("fla", choice.body, 1, "gather_ssm_states", "fla_gdn_prefill", "scatter_ssm_states_"),
        ("fused_chunk", chunk.body, 1, "gather_ssm_states", "_chunk_gated_delta_rule_fused", "scatter_ssm_states_"),
        ("triton_chunk", chunk.orelse, 2, "clear_ssm_states", "chunk_gated_delta_rule", None),
    )

    def calls(node, name):
        return any(
            isinstance(child, ast.Call)
            and (getattr(child.func, "id", None) == name or getattr(child.func, "attr", None) == name)
            for child in ast.walk(node)
        )

    for _, body, load_end, load_call, compute_call, store_call in branches:
        if (
            len(body) != load_end + 2
            or not calls(body[load_end - 1], load_call)
            or not calls(body[load_end], compute_call)
            or (
                not calls(body[-1], store_call) if store_call is not None
                else not (
                    isinstance(body[-1], ast.Assign)
                    and _same(body[-1].targets[0], _nodes(
                        "ssm_state[prefill_state_indices] = None"
                    )[0].targets[0])
                )
            )
        ):
            return False
    sampled = prefix + "_perf_sampled"
    timer = prefix + "_perf_timer"
    for path, body, load_end, *_ in branches:
        instrumented = _nodes(f"{sampled} = {prefix}_perf_active()")
        groups = (body[:load_end], body[load_end:load_end + 1], body[load_end + 1:])
        for stage, original in zip(stages, groups):
            instrumented.extend(_nodes(f"if {sampled}: {timer} = {prefix}_perf_start({stage!r})"))
            instrumented.extend(original)
            instrumented.extend(_nodes(f"if {sampled}: {prefix}_perf_end({timer})"))
        instrumented.extend(_nodes(f"""
            {prefix}_perf_prefill({path!r}, query_non_spec.shape[1],
                                 prefill_state_indices.numel(), ssm_state)
        """))
        body[:] = instrumented
    return True


_PRE_CONV = "conv_weights_T = _get_packed_conv_weights(self)"
_SPEC_QKV = "query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)"
_GATE_BLOCK = """
g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
if spec_sequence_masks is not None:
    if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
        g_spec = g
        beta_spec = beta
        g_non_spec = None
        beta_non_spec = None
    else:
        g_spec = g.index_select(1, spec_token_indx)
        beta_spec = beta.index_select(1, spec_token_indx)
        g_non_spec = g.index_select(1, non_spec_token_indx)
        beta_non_spec = beta.index_select(1, non_spec_token_indx)
else:
    g_spec = None
    beta_spec = None
    g_non_spec = g
    beta_non_spec = beta
"""
_SPEC_VERIFY = """
if spec_sequence_masks is not None:
    actual_seq_lengths = attn_metadata.spec_decode_metadata.actual_seq_lengths
    query_spec = l2norm_fwd(query_spec)
    key_spec = l2norm_fwd(key_spec)
    core_attn_out_spec = recurrent_gated_delta_rule(
        query_spec.squeeze(0), key_spec.squeeze(0), value_spec.squeeze(0), ssm_state,
        g=g_spec.squeeze(0), beta=beta_spec.squeeze(0),
        scale=key_spec.shape[-1] ** -0.5,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=spec_state_indices_tensor.flatten(),
        num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens.to(torch.int32),
    ).unsqueeze(0)
else:
    core_attn_out_spec, last_recurrent_state = None, None
"""
_MERGE = """
merged_out = torch.empty(
    (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
    dtype=core_attn_out_non_spec.dtype, device=core_attn_out_non_spec.device,
)
merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
"""


def build_mixed_core(original_core):
    """Build the five reviewed substitutions or fail without changing the core.

    The returned callable shares the native function's globals. In particular,
    a later recurrent_gated_delta_rule plugin redirect remains visible. The
    source is obtained solely from the installed native Python function.
    """
    if getattr(original_core, "_gdn_mixed_source_digest", None):
        return original_core
    if not inspect.isfunction(original_core):
        raise UnsupportedCoreError("GDN mixed core requires a native Python function")
    if getattr(original_core, "__wrapped__", None) is not None or original_core.__closure__:
        raise UnsupportedCoreError("GDN mixed core requires an undecorated function without closure")
    try:
        source = textwrap.dedent(inspect.getsource(original_core))
    except (OSError, TypeError) as exc:
        raise UnsupportedCoreError("GDN mixed core installed source is unavailable") from exc
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise UnsupportedCoreError("GDN mixed core source must contain one function")
    function = tree.body[0]
    if function.decorator_list or function.args.defaults or function.args.kw_defaults:
        raise UnsupportedCoreError("GDN mixed core decorators/defaults do not match the native ABI")
    if (
        [arg.arg for arg in function.args.args] != ["self", "mixed_qkv", "b", "a", "core_attn_out"]
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
        or function.args.posonlyargs
    ):
        raise UnsupportedCoreError("GDN mixed core tensor argument ABI changed")
    if any(isinstance(node, ast.Name) and node.id == _PLAN_NAME for node in ast.walk(function)):
        raise UnsupportedCoreError("GDN mixed core local name collision")

    # Validate every anchor before editing any native AST or namespace.
    body = function.body
    pre = _find_block(body, _nodes(_PRE_CONV), "pre-convolution")
    qkv = _find_block(body, _nodes(_SPEC_QKV), "spec QKV")
    gates = _find_block(body, _nodes(_GATE_BLOCK), "gate split")
    verify = _find_block(body, _nodes(_SPEC_VERIFY), "spec verify")
    merge_candidates = [
        (index, node)
        for index, node in enumerate(body)
        if isinstance(node, ast.If)
        and _same(node.test, _nodes(
            "if spec_sequence_masks is not None and core_attn_out_non_spec is not None: pass"
        )[0].test)
    ]
    if len(merge_candidates) != 1:
        raise UnsupportedCoreError("GDN mixed core source contract merge guard changed")
    merge_index, merge_if = merge_candidates[0]
    merge_body = _find_block(merge_if.body, _nodes(_MERGE), "mixed output merge")
    if merge_body != 0 or len(merge_if.body) != len(_nodes(_MERGE)):
        raise UnsupportedCoreError("GDN mixed core merge has unreviewed additional statements")
    if not pre < qkv < gates < verify < merge_index:
        raise UnsupportedCoreError("GDN mixed core native operation order changed")
    sliced = _nodes("""
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]
    """)
    slicing = _find_block(body, sliced, "actual token slices")
    if slicing + len(sliced) != pre:
        raise UnsupportedCoreError("GDN mixed core insertion point is not before convolution")

    digest = hashlib.sha256(source.encode()).hexdigest()
    prefix = "_gdn_recoverssm_mixed_" + digest[:16]
    prefill_diagnostics = _instrument_native_prefill(function, prefix)
    bindings = {
        prefix + "_preflight": mixed_preflight,
        prefix + "_qkv": None,  # resolved only after source validation below
        prefix + "_gates": prepare_mixed_gates,
        prefix + "_verify": verify_mixed_spec,
        prefix + "_merge": merge_mixed_output,
    }
    if prefill_diagnostics:
        from . import diagnostics

        bindings.update({
            prefix + "_perf_active": diagnostics.timing_active,
            prefix + "_perf_start": diagnostics.timing_start,
            prefix + "_perf_end": diagnostics.timing_end,
            prefix + "_perf_prefill": diagnostics.prefill_dispatch,
        })
    from .spec_forward import _packed_qkv_views

    bindings[prefix + "_qkv"] = _packed_qkv_views
    replacements = {
        qkv: (1, _nodes(f"""
            if {_PLAN_NAME} is not None:
                query_spec, key_spec, value_spec = {prefix}_qkv(self, mixed_qkv_spec)
            else:
                {_SPEC_QKV}
        """)),
        gates: (len(_nodes(_GATE_BLOCK)), [ast.If(
            test=ast.parse(f"{_PLAN_NAME} is not None", mode="eval").body,
            body=_nodes(f"""
                g_spec, beta_spec, g_non_spec, beta_non_spec = {prefix}_gates(
                    self, a, b, {_PLAN_NAME}, DeviceOperator.fused_gdn_gating)
            """),
            orelse=copy.deepcopy(_nodes(_GATE_BLOCK)),
        )]),
        verify: (1, [ast.If(
            test=ast.parse(f"{_PLAN_NAME} is not None", mode="eval").body,
            body=_nodes(f"""
                core_attn_out_spec = {prefix}_verify(
                    self, query_spec, key_spec, value_spec, g_spec, beta_spec, ssm_state)
            """),
            orelse=copy.deepcopy(_nodes(_SPEC_VERIFY)),
        )]),
    }
    merge_if.body = [ast.If(
        test=ast.parse(f"{_PLAN_NAME} is not None", mode="eval").body,
        body=_nodes(f"""
            {prefix}_merge(core_attn_out, num_actual_tokens, spec_token_indx,
                          non_spec_token_indx, core_attn_out_spec, core_attn_out_non_spec)
        """),
        orelse=copy.deepcopy(merge_if.body),
    )]
    for index in sorted(replacements, reverse=True):
        count, replacement = replacements[index]
        body[index:index + count] = replacement
    body[pre:pre] = _nodes(f"""
        {_PLAN_NAME} = {prefix}_preflight(self, attn_metadata, mixed_qkv, a, b, core_attn_out)
    """)
    function.name = prefix + "_core"
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<gdn-recoverssm-mixed-{digest[:16]}>"
    code = compile(generated, filename, "exec")
    namespace = original_core.__globals__
    if any(name in namespace and namespace[name] is not value for name, value in bindings.items()):
        raise UnsupportedCoreError("GDN mixed core private helper name collision")
    namespace.update(bindings)
    linecache.cache[filename] = (len(generated), None, generated.splitlines(True), filename)
    definitions = {}
    exec(code, namespace, definitions)  # noqa: S102 - reviewed installed source; same native globals.
    adapted = definitions[function.name]
    functools.update_wrapper(adapted, original_core)
    adapted._gdn_mixed_source_digest = digest
    adapted._gdn_mixed_generated_digest = hashlib.sha256(generated.encode()).hexdigest()
    adapted._gdn_prefill_diagnostics = prefill_diagnostics
    adapted._gdn_mixed_patch_counts = {
        "preflight": 1, "spec_qkv": 1, "gates": 1, "spec_verify": 1, "merge": 1,
    }
    return adapted
