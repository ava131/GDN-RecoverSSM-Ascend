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
        "kernels=verify,grouped-fold plan_conv_publisher=triton",
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
        or not _COUNTS["prepared_verify"]
        or not _COUNTS["prepared_fold"]
    ):
        raise RuntimeError(
            "GDN AscendC verify/fold must be prepared for the actual model "
            "and executed by startup self-tests before READY"
        )
    _log().info(
        "GDN_ASCENDC_SELFTEST PASS verify_launches=%d fold_launches=%d "
        "request_parameters=unchanged",
        _COUNTS["verify"],
        _COUNTS["fold"],
    )
