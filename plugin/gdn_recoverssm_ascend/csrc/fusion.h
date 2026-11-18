// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <type_traits>
#include "kernel_operator.h"

namespace gdn_ac {
namespace verify_detail {
using namespace AscendC;

__aicore__ inline void FusedVectorToScalar(TEventID event)
{
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
}
__aicore__ inline void FusedScalarToVector(TEventID event)
{
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
}

template<class T>
__aicore__ inline float FusedFloat(T value)
{
    if constexpr (std::is_same<T, float>::value) return value;
    else return ToFloat(value);
}

// scratch belongs to the second state bank, before any recurrence uses it.
// The normalized vectors deliberately round through the activation dtype, as
// rc1's separate l2norm output does. Key records then contain these exact keys.
template<class Activation>
__aicore__ inline void NormalizeFusedWindow(
    LocalTensor<float> q, LocalTensor<float> k,
    LocalTensor<Activation> rawq, LocalTensor<Activation> rawk,
    LocalTensor<float> scratch, uint32_t count)
{
    // Up to eight 128-wide rows. Keep reductions and broadcast entirely on
    // Vector: a Scalar GetValue + S/V round trip per token defeats fusion.
    auto norms = scratch[1024];
    auto denominator = scratch[1056];
    for (uint32_t which = 0; which < 2; ++which) {
        auto values = which == 0 ? q : k;
        Duplicate(norms, 0.0f, 8);  // Brcb also reads the inactive tail.
        Mul(scratch, values, values, count * 128);
        PipeBarrier<PIPE_V>();
        Add(scratch, scratch, scratch[64], 64, count, {1, 1, 1, 16, 16, 16});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(norms, scratch, 64, count, 1, 1, 16);
        PipeBarrier<PIPE_V>();
        Adds(norms, norms, 1e-6f, 8);
        PipeBarrier<PIPE_V>();
        // Sqrt+Div avoids depending on the low-accuracy raw Rsqrt intrinsic.
        Sqrt(norms, norms, 8);
        PipeBarrier<PIPE_V>();
        Brcb(denominator, norms, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < 128; column += 64) {
            Div(values[column], values[column], denominator, 64, count,
                {1, 1, 0, 16, 16, 1});
        }
        PipeBarrier<PIPE_V>();
    }
    if constexpr (!std::is_same<Activation, float>::value) {
        Cast(rawq, q, RoundMode::CAST_RINT, count * 128);
        Cast(rawk, k, RoundMode::CAST_RINT, count * 128);
        PipeBarrier<PIPE_V>();
        Cast(q, rawq, RoundMode::CAST_NONE, count * 128);
        Cast(k, rawk, RoundMode::CAST_NONE, count * 128);
        PipeBarrier<PIPE_V>();
    } else {
        Muls(rawk, k, 1.0f, count * 128);
        PipeBarrier<PIPE_V>();
    }
}

template<class Activation, class Beta, class Bias>
__aicore__ inline void LoadFusedMeta(
    GlobalTensor<Activation> a, GlobalTensor<Beta> b,
    GlobalTensor<float> A_log, GlobalTensor<Bias> dt_bias,
    LocalTensor<float> gates, LocalTensor<float> beta_values,
    LocalTensor<Beta> raw_beta, LocalTensor<float> scratch,
    int64_t head, int64_t begin, uint32_t count,
    int64_t a_stride, int64_t b_stride,
    TEventID scalarReady, TEventID scalarResult)
{
    // Native Qwen creates dt_bias in the model dtype, while A_log is FP32.
    // Convert the loaded scalar, preserving the stored BF16 parameter value.
    const float bias = FusedFloat(dt_bias.GetValue(head));
    const float logA = A_log.GetValue(head);
    auto expA = scratch;
    auto ones = scratch[32];
    for (uint32_t t = 0; t < 8; ++t) {
        const float x = t < count ? FusedFloat(a.GetValue((begin + t) * a_stride + head)) + bias : 0.0f;
        // Avoid evaluating exp(x) for the softplus linear branch.
        gates.SetValue(t, x <= 20.0f ? x : 0.0f);
        beta_values.SetValue(t, t < count ? -FusedFloat(b.GetValue((begin + t) * b_stride + head)) : 0.0f);
    }
    FusedScalarToVector(scalarReady);
    Duplicate(expA, logA, 8);
    Duplicate(ones, 1.0f, 8);
    PipeBarrier<PIPE_V>();
    Exp(expA, expA, 8);
    Exp(gates, gates, 8);
    Exp(beta_values, beta_values, 8);
    PipeBarrier<PIPE_V>();
    Adds(gates, gates, 1.0f, 8);
    Adds(beta_values, beta_values, 1.0f, 8);
    PipeBarrier<PIPE_V>();
    Ln(gates, gates, 8);
    Div(beta_values, ones, beta_values, 8);
    PipeBarrier<PIPE_V>();
    FusedVectorToScalar(scalarResult);
    for (uint32_t t = 0; t < count; ++t) {
        const float x = FusedFloat(a.GetValue((begin + t) * a_stride + head)) + bias;
        if (x > 20.0f) gates.SetValue(t, x);
    }
    const float negA = -expA.GetValue(0);
    FusedScalarToVector(scalarReady);
    Muls(gates, gates, negA, 8);
    PipeBarrier<PIPE_V>();
    Exp(gates, gates, 8);
    if constexpr (!std::is_same<Beta, float>::value) {
        Cast(raw_beta, beta_values, RoundMode::CAST_RINT, 8);
        PipeBarrier<PIPE_V>();
        Cast(beta_values, raw_beta, RoundMode::CAST_NONE, 8);
    }
    PipeBarrier<PIPE_V>();
    FusedVectorToScalar(scalarResult);
}
} // namespace verify_detail
} // namespace gdn_ac
