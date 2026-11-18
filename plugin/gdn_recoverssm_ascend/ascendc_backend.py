# SPDX-License-Identifier: Apache-2.0
"""AscendC dispatch for the two expensive replay kernels.

Selection happens before any kernel is submitted. Execution errors propagate:
retrying another backend after a partial state write is never safe.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

_ENABLED = False
_OPS = None
_COUNTS = Counter()
_FALLBACKS = set()


def _log():
    from vllm.logger import init_logger

    return init_logger("vllm.gdn_recoverssm_ascend.ascendc")


def initialize():
    global _ENABLED, _OPS
    from .ascendc_build import ensure_loaded

    artifact = ensure_loaded()
    import torch

    _OPS = torch.ops.gdn_recoverssm_ac
    if _OPS.abi_version() != 1:
        raise RuntimeError("GDN AscendC library ABI mismatch")
    _ENABLED = True
    _log().info(
        "GDN_ASCENDC loaded fingerprint=%s library=%s cache_hit=%s "
        "kernels=verify,verify_fused,grouped-fold plan_conv_publisher=triton",
        artifact.fingerprint,
        artifact.library_path,
        artifact.cache_hit,
    )
    return artifact


@dataclass(frozen=True)
class AscendCSpec:
    name: str
    tensors: tuple
    config: tuple
    scale: float | None = None

    def compile_only(self):
        # The ahead-of-time library is already loaded. Mandatory owned-buffer
        # device tests execute the entry points before graph capture / readiness.
        _COUNTS["prepared_" + self.name] += 1
        return self

    def launch(self):
        if not _ENABLED or _OPS is None:
            raise RuntimeError("GDN AscendC was not initialized before launch")
        op = getattr(_OPS, self.name)
        if self.scale is None:
            op(*self.tensors, list(self.config))
        else:
            op(*self.tensors, list(self.config), self.scale)
        _COUNTS[self.name] += 1


def counts():
    return dict(_COUNTS)


def _no_fusion(reason):
    key = ("verify_fused", reason)
    if key not in _FALLBACKS:
        _FALLBACKS.add(key)
        _log().info(
            "GDN_ASCENDC_FUSION eligible=0 preprocessing=separate-native-helpers "
            "verify_backend=existing-dispatch reason=%s", reason
        )
    return False


def _decline(name, reason):
    key = (name, reason)
    if key not in _FALLBACKS:
        _FALLBACKS.add(key)
        _log().info(
            "GDN_ASCENDC compatibility backend=triton kernel=%s reason=%s", name, reason
        )
    return None


def select(spec):
    if not _ENABLED:
        return None
    kernel = spec.kernel
    name = getattr(kernel, "__name__", None) or getattr(
        getattr(kernel, "fn", None), "__name__", ""
    )
    if name == "_gdn_recoverssm_verify_kernel":
        return _verify_spec(spec)
    if name == "_grouped_commit_gdn_state_kernel":
        return _fold_spec(spec)
    return None


def _verify_spec(spec):
    import torch

    a, kw = spec.args, spec.kwargs
    if not kw.get("RECORD_SOA") or kw.get("USE_QK_L2NORM"):
        return _decline("verify", "requires pre-normalized typed-SoA")
    if kw["K"] != 128 or kw["V"] != 128 or a[26] != 1:
        return _decline("verify", "requires K=V=128 and contiguous state K")
    q, k, v, g, beta, state, corr, key, decay, out, qsl, indices = a[:12]
    supported = (torch.float32, torch.bfloat16)
    if (
        q.dtype not in supported
        or state.dtype not in supported
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or key.dtype != q.dtype
        or out.dtype != q.dtype
        or beta.dtype not in (q.dtype, torch.float32)
        or g.dtype != torch.float32
        or corr.dtype != torch.float32
        or decay.dtype != torch.float32
        or qsl.dtype != torch.int32
        or indices.dtype != torch.int32
    ):
        return _decline("verify", "dtype outside the ahead-of-time specialization set")
    if a[30] != 1 or a[34] != 1 or a[37] != 1:
        return _decline("verify", "record innermost stride must be one")
    if not all(
        _dma_gap(stride, tensor.element_size())
        for stride, tensor in ((a[18], q), (a[19], k), (a[20], v), (a[25], state))
    ):
        return _decline("verify", "DMA row stride outside fast-path range")
    config = (
        a[14],
        a[12],
        kw["H"],
        kw["HV"],
        kw["SPEC_QUERY_LEN"],
        kw["NUM_BLOCKS"],
        a[16],
        a[17],
        *a[18:26],
        *a[27:30],
        *a[31:34],
        *a[35:38],
        a[38],
        a[39],
        a[40],
    )
    return AscendCSpec(
        "verify", tuple(a[:12]), tuple(int(x) for x in config), float(a[13])
    )


def _fold_spec(spec):
    import torch

    a, kw = spec.args, spec.kwargs
    if not kw.get("RECORD_SOA") or kw["K"] != 128 or kw["V"] != 128 or a[25] != 1:
        return _decline("fold", "requires typed-SoA K=V=128 and contiguous state K")
    state, _, _, corr, _, _, key, _, _, decay = a[:10]
    supported = (torch.float32, torch.bfloat16)
    if (
        state.dtype not in supported
        or key.dtype not in supported
        or corr.dtype != torch.float32
        or decay.dtype != torch.float32
    ):
        return _decline("fold", "unsupported state/record dtype")
    if a[28] != 1 or a[31] != 1 or a[33] != 1 or a[34] != 1:
        return _decline("fold", "noncontiguous record or plan elements")
    if not all(
        _dma_gap(stride, tensor.element_size())
        for stride, tensor in ((a[24], state), (a[27], corr), (a[30], key))
    ):
        return _decline("fold", "DMA row stride outside fast-path range")
    config = (
        a[20],
        a[1].numel(),
        kw["NUM_HEADS"],
        kw["NUM_KEY_HEADS"],
        corr.shape[2],
        kw["NUM_BLOCKS"],
        a[22],
        a[18],
        a[19],
        *a[23:25],
        *a[26:28],
        *a[29:31],
        *a[32:34],
    )
    # Device address tables are borrowed from the bucket, never uploaded here.
    tensors = (
        a[0],
        a[3],
        a[6],
        a[9],
        a[1],
        a[2],
        a[4],
        a[5],
        a[7],
        a[8],
        a[10],
        a[11],
        a[17],
        a[12],
        a[13],
        a[14],
        a[15],
        a[16],
    )
    return AscendCSpec("fold", tensors, tuple(int(x) for x in config))


def _dma_gap(stride, element_bytes):
    return 0 <= (int(stride) - 128) * element_bytes <= 0xFFFFFFFF


def assert_ready():
    if (
        not _ENABLED
        or not _COUNTS["verify"]
        or not _COUNTS["fold"]
        or not _COUNTS["verify_fused"]
        or not _COUNTS["prepared_verify"]
        or not _COUNTS["prepared_fold"]
    ):
        raise RuntimeError(
            "GDN AscendC verify/fold must be prepared for the actual model; verify_fused "
            "and ordinary entry points must execute startup self-tests before READY"
        )
    _log().info(
        "GDN_ASCENDC_READY PASS verify_launches=%d fold_launches=%d fused_launches=%d "
        "request_parameters=unchanged",
        _COUNTS["verify"],
        _COUNTS["fold"],
        _COUNTS["verify_fused"],
    )


def can_fuse_verify(layer, mixed_qkv, a, b, *, meta=None, out=None):
    """Host-only eligibility, evaluated before convolution changes its history."""
    if not _ENABLED or _OPS is None:
        return _no_fusion("AscendC not initialized")
    import torch

    activation = mixed_qkv.dtype
    supported = (torch.float32, torch.bfloat16)
    params = (getattr(layer, "A_log", None), getattr(layer, "dt_bias", None))
    if (
        mixed_qkv.ndim != 2
        or activation not in supported
        or a.dtype != activation
        or b.dtype not in (activation, torch.float32)
        or getattr(layer, "head_k_dim", None) != 128
        or getattr(layer, "head_v_dim", None) != 128
        or a.ndim != 2
        or b.shape != a.shape
        or a.stride(1) != 1
        or b.stride(1) != 1
        or any(not isinstance(x, torch.Tensor) for x in params)
    ):
        return _no_fusion("raw activation/gate dtype, layout, or head dimension")
    for param in params:
        if (
            param.ndim != 1
            or param.numel() != a.shape[1]
            or not param.is_contiguous()
            or param.device != mixed_qkv.device
        ):
            return _no_fusion("gate parameters must be contiguous local heads on the input device")
    if params[0].dtype != torch.float32 or params[1].dtype not in supported:
        return _no_fusion(
            f"gate parameter dtype: A_log={params[0].dtype} dt_bias={params[1].dtype}; "
            "requires FP32 A_log and BF16/FP32 dt_bias"
        )
    if a.device != b.device or a.device != mixed_qkv.device:
        return _no_fusion("raw inputs must share a device")
    state = getattr(layer, "kv_cache", ())
    records = getattr(layer, "_gdn_recoverssm_records", None)
    qsl = getattr(meta, "spec_query_start_loc", None)
    indices = getattr(meta, "spec_state_indices_tensor", None)
    window = int(getattr(layer, "num_spec", -1)) + 1
    if len(state) < 2 or meta is None or not 1 <= window <= 8:
        return _no_fusion("missing metadata/state or unsupported window")
    state = state[1]
    if (
        not isinstance(state, torch.Tensor)
        or state.dtype not in supported
        or state.ndim != 4
        or tuple(state.shape[2:]) != (128, 128)
        or state.shape[1] != a.shape[1]
        or state.stride(3) != 1
        or not _dma_gap(state.stride(2), state.element_size())
    ):
        return _no_fusion("state dtype, shape, or DMA stride")
    if not all(isinstance(x, torch.Tensor) for x in (qsl, indices, out)):
        return _no_fusion("missing output or row metadata tensors")
    if out.ndim != 3:
        return _no_fusion("output rank")
    rows = min(int(getattr(meta, "num_spec_decodes", 0)), qsl.numel() - 1)
    actual = int(getattr(meta, "num_actual_tokens", -1))
    if (
        qsl.dtype != torch.int32
        or indices.dtype != torch.int32
        or qsl.ndim != 1
        or indices.ndim != 2
        or indices.shape[0] < rows
        or indices.shape[1] < 1
        or rows <= 0
        or actual <= 0
        or actual > rows * window
        or actual > min(mixed_qkv.shape[0], a.shape[0], b.shape[0], out.shape[0])
        or mixed_qkv.ndim != 2
        or mixed_qkv.stride(1) != 1
        or out.dtype != activation
        or out.ndim != 3
        or tuple(out.shape[1:]) != (a.shape[1], 128)
        or out.stride(2) != 1
        or out.stride(1) != 128
    ):
        return _no_fusion("row metadata, token capacity, or output layout")
    corr, key, decay = (
        getattr(records, name, None) for name in ("correction", "key", "decay")
    )
    if not all(isinstance(x, torch.Tensor) for x in (corr, key, decay)):
        return _no_fusion("missing typed records")
    if corr.ndim != 4 or key.ndim != 4 or decay.ndim != 3:
        return _no_fusion("record rank")
    tp = int(getattr(layer, "tp_size", 1))
    if tp <= 0:
        return _no_fusion("invalid TP size")
    global_key_heads = int(getattr(layer, "num_k_heads", key.shape[1] * tp))
    global_value_heads = int(getattr(layer, "num_v_heads", a.shape[1] * tp))
    if global_key_heads % tp or global_value_heads % tp:
        return _no_fusion("heads must divide TP size")
    key_heads = global_key_heads // tp
    value_heads = global_value_heads // tp
    if (
        corr.dtype != torch.float32
        or decay.dtype != torch.float32
        or key.dtype != activation
        or corr.ndim != 4
        or key.ndim != 4
        or decay.ndim != 3
        or corr.shape[0] < rows
        or tuple(corr.shape[1:]) != (a.shape[1], window, 128)
        or key.shape[0] != corr.shape[0]
        or key.shape[1] < 1
        or key.shape[1] != key_heads
        or value_heads != a.shape[1]
        or a.shape[1] % key.shape[1]
        or tuple(key.shape[2:]) != (window, 128)
        or tuple(decay.shape) != (corr.shape[0], a.shape[1], window)
        or any(x.stride(-1) != 1 for x in (corr, key, decay))
    ):
        return _no_fusion("record dtype, capacity, stride, or head geometry")
    # The pinned rearrange_mixed_qkv ABI splits this packed Q/K/V width.
    # It materializes contiguous outputs; the packed width also bounds their
    # token DMA gaps. Check geometry before convolution changes its history.
    packed_width = (2 * key_heads + value_heads) * 128
    if mixed_qkv.shape[1] != packed_width or not _dma_gap(
        packed_width, mixed_qkv.element_size()
    ):
        return _no_fusion("packed QKV width or DMA range")
    if not all(
        x.device == mixed_qkv.device
        for x in (state, corr, key, decay, qsl, indices, out)
    ):
        return _no_fusion("state/records/metadata/output must share a device")
    return True


def launch_fused_verify(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
    records,
    qsl,
    indices,
    *,
    spec_query_len,
    scale=None,
    out=None,
):
    """Raw-input AC entry point. A failure is fatal; never retry after conv."""
    import torch

    if not _ENABLED or _OPS is None:
        raise RuntimeError("GDN fused verify was not initialized")
    if out is None:
        out = torch.empty_like(v)
    corr, key, decay = records.correction, records.key, records.decay
    rows, tokens = indices.numel(), q.shape[1]
    config = (
        rows,
        tokens,
        q.shape[2],
        v.shape[2],
        spec_query_len,
        state.shape[0],
        corr.shape[0],
        0,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        a.stride(0),
        b.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        corr.stride(0),
        corr.stride(1),
        corr.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        decay.stride(0),
        decay.stride(1),
        decay.stride(2),
        out.stride(1),
        qsl.stride(0),
        indices.stride(0),
    )
    tensors = (
        q,
        k,
        v,
        a,
        b,
        state,
        corr,
        key,
        decay,
        out,
        qsl,
        indices,
        A_log,
        dt_bias,
    )
    AscendCSpec(
        "verify_fused",
        tensors,
        config,
        float(scale if scale is not None else 128**-0.5),
    ).launch()
    _COUNTS["verify_fused_model_calls"] += 1
    _COUNTS["verify_fused_saved_helpers"] += 3
    if _COUNTS["verify_fused_model_calls"] == 1:
        _log().info(
            "GDN_ASCENDC_FUSION model_dispatch=PASS activation=%s state=%s "
            "A_log=%s dt_bias=%s "
            "query_len=%d key_heads=%d value_heads=%d external_helpers_removed=3 "
            "q_token_stride=%d v_token_stride=%d "
            "scope=HOST_DISPATCH_NOT_GRAPH_REPLAY",
            q.dtype, state.dtype, A_log.dtype, dt_bias.dtype,
            spec_query_len, q.shape[2], v.shape[2],
            q.stride(1), v.stride(1),
        )
    return out
