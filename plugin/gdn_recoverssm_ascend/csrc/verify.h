// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <type_traits>
#include "common.h"
#include "kernel_operator.h"

namespace gdn_ac {
namespace verify_detail {
using namespace AscendC;

// One task owns an entire request/value-head matrix. The two FP32 banks
// alternate as recurrence state and reduction scratch. QKV are loaded once per
// short window; checkpoint GM is read-only throughout this kernel.
template <typename Activation, typename State, typename Key, typename Beta,
          typename Index, typename Qsl>
class Verify {
    static constexpr uint32_t kDim = 128;
    static constexpr uint32_t kElements = kDim * kDim;
    static constexpr uint32_t kWindow = 8;
    static constexpr uint32_t kMatrixBytes = kElements * sizeof(float);
    static constexpr uint32_t kWindowElements = kWindow * kDim;
    static constexpr uint32_t kWindowBytes = kWindowElements * sizeof(float);
    static constexpr uint32_t kQOffset = 2 * kMatrixBytes;
    static constexpr uint32_t kKOffset = kQOffset + kWindowBytes;
    static constexpr uint32_t kVOffset = kKOffset + kWindowBytes;
    static constexpr uint32_t kRawOffset = kVOffset + kWindowBytes;
    static constexpr uint32_t kRawKeyOffset = kRawOffset + kWindowBytes;
    static constexpr uint32_t kCorrectionOffset = kRawKeyOffset + kWindowBytes;
    static constexpr uint32_t kOutputOffset = kCorrectionOffset + kDim * sizeof(float);
    static constexpr uint32_t kBroadcastOffset = kOutputOffset + kDim * sizeof(float);
    static constexpr uint32_t kBetaRawOffset = kBroadcastOffset + kDim * 32;
    static constexpr uint32_t kMetaOffset = kBetaRawOffset + 32;
    static constexpr uint32_t kDecayOffset = kMetaOffset + 2 * kWindow * sizeof(float);
    static constexpr uint32_t kUbBytes = kDecayOffset + 32;
    static_assert(kUbBytes + 2048 <= 192 * 1024, "verify UB exceeds A2 capacity");
    static_assert(std::is_same<Key, Activation>::value || std::is_same<Key, float>::value,
                  "records must preserve normalized key precision");

public:
    __aicore__ inline explicit Verify(const VerifyTiling &tiling) : td_(tiling) {}

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
        GM_ADDR state, GM_ADDR correction, GM_ADDR key_record,
        GM_ADDR decay_record, GM_ADDR out, GM_ADDR qsl, GM_ADDR indices)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ Activation *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ Activation *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ Activation *>(v));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ Beta *>(beta));
        state_.SetGlobalBuffer(reinterpret_cast<__gm__ State *>(state));
        correction_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(correction));
        keys_.SetGlobalBuffer(reinterpret_cast<__gm__ Key *>(key_record));
        decays_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(decay_record));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ Activation *>(out));
        qsl_.SetGlobalBuffer(reinterpret_cast<__gm__ Qsl *>(qsl));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ Index *>(indices));
        pipe_.InitBuffer(ub_, kUbBytes);
        inputReady_ = pipe_.AllocEventID<HardEvent::MTE2_V>();
        inputFree_ = pipe_.AllocEventID<HardEvent::V_MTE2>();
        outputReady_ = pipe_.AllocEventID<HardEvent::V_MTE3>();
        outputFree_ = pipe_.AllocEventID<HardEvent::MTE3_V>();
        scalarReady_ = pipe_.AllocEventID<HardEvent::S_V>();
        scalarResult_ = pipe_.AllocEventID<HardEvent::V_S>();
    }

    __aicore__ inline void Process()
    {
        const int64_t tasks = td_.rows * td_.value_heads;
        for (int64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
            const int64_t row = task / td_.value_heads;
            const int64_t head = task % td_.value_heads;
            const int64_t begin = qsl_.GetValue(row * td_.qsl_stride);
            const int64_t end = qsl_.GetValue((row + 1) * td_.qsl_stride);
            const int64_t length = end - begin;
            const int64_t slot = indices_.GetValue(row * td_.index_stride);
            const bool valid = slot > td_.null_block && slot < td_.blocks &&
                row < td_.record_rows && begin >= 0 && end >= begin &&
                end <= td_.tokens && length <= td_.query_len;
            if (!valid) {
                ZeroOutput(head, begin, length);
            } else if (length > 0) {
                RunTask(row, head, slot, begin, static_cast<uint32_t>(length));
            }
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(inputFree_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputFree_);
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarReady_);
        pipe_.ReleaseEventID<HardEvent::V_S>(scalarResult_);
    }

private:
    __aicore__ inline void InputReady()
    {
        SetFlag<HardEvent::MTE2_V>(inputReady_);
        WaitFlag<HardEvent::MTE2_V>(inputReady_);
    }

    __aicore__ inline void ReuseInput()
    {
        SetFlag<HardEvent::V_MTE2>(inputFree_);
        WaitFlag<HardEvent::V_MTE2>(inputFree_);
    }

    __aicore__ inline void BeforeOutput()
    {
        SetFlag<HardEvent::V_MTE3>(outputReady_);
        WaitFlag<HardEvent::V_MTE3>(outputReady_);
    }

    __aicore__ inline void AfterOutput()
    {
        SetFlag<HardEvent::MTE3_V>(outputFree_);
        WaitFlag<HardEvent::MTE3_V>(outputFree_);
    }

    __aicore__ inline void VectorToScalar()
    {
        SetFlag<HardEvent::V_S>(scalarResult_);
        WaitFlag<HardEvent::V_S>(scalarResult_);
    }

    template <typename T>
    __aicore__ inline void StoreRow(GlobalTensor<T> target, int64_t offset,
                                    LocalTensor<T> values)
    {
        DataCopyExtParams copy{1, kDim * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(target[offset], values, copy);
    }

    __aicore__ inline void ZeroOutput(int64_t head, int64_t begin, int64_t length)
    {
        auto zeros = ub_.GetWithOffset<float>(kDim, kOutputOffset);
        Duplicate(zeros, 0.0f, kDim);
        PipeBarrier<PIPE_V>();
        auto raw = ub_.GetWithOffset<Activation>(kDim, kRawOffset);
        if constexpr (!std::is_same<Activation, float>::value) {
            Cast(raw, zeros, RoundMode::CAST_RINT, kDim);
            PipeBarrier<PIPE_V>();
        }
        BeforeOutput();
        for (int64_t t = 0; t < td_.query_len && t < length; ++t) {
            const int64_t token = begin + t;
            if (token < 0 || token >= td_.tokens) continue;
            const int64_t offset = token * td_.out_token_stride + head * kDim;
            if constexpr (std::is_same<Activation, float>::value) {
                StoreRow(out_, offset, zeros);
            } else {
                StoreRow(out_, offset, raw);
            }
        }
        AfterOutput();
    }

    __aicore__ inline void LoadMatrix(int64_t slot, int64_t head)
    {
        ReuseInput();
        const int64_t offset = slot * td_.state_block_stride + head * td_.state_head_stride;
        DataCopyExtParams copy{
            kDim, kDim * static_cast<uint32_t>(sizeof(State)),
            static_cast<uint32_t>((td_.state_v_stride - kDim) * sizeof(State)), 0, 0};
        DataCopyPadExtParams<State> padding{false, 0, 0, 0};
        auto current = ub_.GetWithOffset<float>(kElements, 0);
        if constexpr (std::is_same<State, float>::value) {
            DataCopyPad(current, state_[offset], copy, padding);
        } else {
            auto packed = ub_.GetWithOffset<State>(kElements, kMatrixBytes);
            DataCopyPad(packed, state_[offset], copy, padding);
        }
        InputReady();
        if constexpr (!std::is_same<State, float>::value) {
            Cast(current, ub_.GetWithOffset<State>(kElements, kMatrixBytes),
                 RoundMode::CAST_NONE, kElements);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void LoadWindow(GlobalTensor<Activation> source,
                                      int64_t offset, int64_t stride,
                                      uint32_t count, uint32_t raw_offset,
                                      uint32_t float_offset)
    {
        ReuseInput();
        auto raw = ub_.GetWithOffset<Activation>(kWindowElements, raw_offset);
        DataCopyExtParams copy{
            static_cast<uint16_t>(count), kDim * static_cast<uint32_t>(sizeof(Activation)),
            static_cast<uint32_t>((stride - kDim) * sizeof(Activation)), 0, 0};
        DataCopyPadExtParams<Activation> padding{false, 0, 0, 0};
        DataCopyPad(raw, source[offset], copy, padding);
        InputReady();
        auto values = ub_.GetWithOffset<float>(kWindowElements, float_offset);
        if constexpr (std::is_same<Activation, float>::value) {
            Muls(values, raw, 1.0f, count * kDim);
        } else {
            Cast(values, raw, RoundMode::CAST_NONE, count * kDim);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadMeta(int64_t head, int64_t begin, uint32_t count)
    {
        auto gates = ub_.GetWithOffset<float>(kWindow, kMetaOffset);
        auto beta_values = ub_.GetWithOffset<float>(kWindow, kMetaOffset + 32);
        auto raw = ub_.GetWithOffset<Beta>(kWindow, kBetaRawOffset);
        for (uint32_t t = 0; t < kWindow; ++t) {
            gates.SetValue(t, t < count ? g_.GetValue((begin + t) * td_.g_token_stride + head) : 0.0f);
            if constexpr (std::is_same<Beta, float>::value) {
                beta_values.SetValue(t, t < count ? beta_.GetValue((begin + t) * td_.beta_token_stride + head) : 0.0f);
            } else {
                raw.SetValue(t, t < count ? beta_.GetValue((begin + t) * td_.beta_token_stride + head) : static_cast<Beta>(0.0f));
            }
        }
        SetFlag<HardEvent::S_V>(scalarReady_);
        WaitFlag<HardEvent::S_V>(scalarReady_);
        if constexpr (!std::is_same<Beta, float>::value) {
            Cast(beta_values, raw, RoundMode::CAST_NONE, kWindow);
        }
        Exp(gates, gates, kWindow);
        PipeBarrier<PIPE_V>();
        VectorToScalar();
    }

    // Reduce 128-column rows with vector instructions. The old state bank is
    // dead at this point and supplies a disjoint full matrix of scratch.
    __aicore__ inline void DotRows(LocalTensor<float> result,
                                   LocalTensor<float> matrix,
                                   LocalTensor<float> vector,
                                   LocalTensor<float> scratch)
    {
        for (uint32_t column = 0; column != kDim; column += 64) {
            Mul(scratch[column], matrix[column], vector[column], 64, kDim,
                {1, 1, 1, 16, 16, 0});
        }
        PipeBarrier<PIPE_V>();
        Add(scratch, scratch, scratch[64], 64, kDim, {1, 1, 1, 16, 16, 16});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(result, scratch, 64, kDim, 1, 1, 16);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SaveToken(int64_t row, int64_t head,
                                     int64_t key_head, int64_t token,
                                     uint32_t t, float decay)
    {
        auto correction = ub_.GetWithOffset<float>(kDim, kCorrectionOffset);
        auto output = ub_.GetWithOffset<float>(kDim, kOutputOffset);
        auto raw = ub_.GetWithOffset<Activation>(kDim, kRawOffset);
        auto decay_value = ub_.GetWithOffset<float>(8, kDecayOffset);
        Duplicate(decay_value, decay, 8);
        if constexpr (!std::is_same<Activation, float>::value) {
            Cast(raw, output, RoundMode::CAST_RINT, kDim);
        }
        PipeBarrier<PIPE_V>();
        BeforeOutput();
        const int64_t out_offset = token * td_.out_token_stride + head * kDim;
        if constexpr (std::is_same<Activation, float>::value) {
            StoreRow(out_, out_offset, output);
        } else {
            StoreRow(out_, out_offset, raw);
        }
        StoreRow(correction_, row * td_.corr_row_stride + head * td_.corr_head_stride +
                 t * td_.corr_token_stride, correction);
        DataCopyExtParams scalar_copy{1, sizeof(float), 0, 0, 0};
        DataCopyPad(decays_[row * td_.decay_row_stride + head * td_.decay_head_stride +
                           t * td_.decay_token_stride], decay_value, scalar_copy);
        if (head % (td_.value_heads / td_.heads) == 0) {
            const int64_t key_offset = row * td_.key_row_stride + key_head * td_.key_head_stride +
                t * td_.key_token_stride;
            if constexpr (std::is_same<Key, Activation>::value) {
                StoreRow(keys_, key_offset,
                    ub_.GetWithOffset<Activation>(kWindowElements, kRawKeyOffset)[t * kDim]);
            } else {
                StoreRow(keys_, key_offset,
                    ub_.GetWithOffset<float>(kWindowElements, kKOffset)[t * kDim]);
            }
        }
        AfterOutput();
    }

    __aicore__ inline void RunTask(int64_t row, int64_t head, int64_t slot,
                                   int64_t begin, uint32_t count)
    {
        const int64_t key_head = head / (td_.value_heads / td_.heads);
        LoadMatrix(slot, head);
        LoadWindow(q_, begin * td_.q_token_stride + key_head * kDim,
                   td_.q_token_stride, count, kRawOffset, kQOffset);
        LoadWindow(k_, begin * td_.k_token_stride + key_head * kDim,
                   td_.k_token_stride, count, kRawKeyOffset, kKOffset);
        LoadWindow(v_, begin * td_.v_token_stride + head * kDim,
                   td_.v_token_stride, count, kRawOffset, kVOffset);
        LoadMeta(head, begin, count);
        auto query = ub_.GetWithOffset<float>(kWindowElements, kQOffset);
        Muls(query, query, td_.scale, count * kDim);
        PipeBarrier<PIPE_V>();
        auto corr = ub_.GetWithOffset<float>(kDim, kCorrectionOffset);
        auto result = ub_.GetWithOffset<float>(kDim, kOutputOffset);
        auto broadcast = ub_.GetWithOffset<float>(kDim * 8, kBroadcastOffset);
        auto gates = ub_.GetWithOffset<float>(kWindow, kMetaOffset);
        auto betas = ub_.GetWithOffset<float>(kWindow, kMetaOffset + 32);
        uint32_t bank = 0;
        for (uint32_t t = 0; t < count; ++t) {
            auto previous = ub_.GetWithOffset<float>(kElements, bank * kMatrixBytes);
            auto next = ub_.GetWithOffset<float>(kElements, (1 - bank) * kMatrixBytes);
            auto key = ub_.GetWithOffset<float>(kWindowElements, kKOffset)[t * kDim];
            const float decay = gates.GetValue(t);
            const float beta = betas.GetValue(t);
            // Publish scalar metadata reads before dependent vector commands.
            SetFlag<HardEvent::S_V>(scalarReady_);
            WaitFlag<HardEvent::S_V>(scalarReady_);
            Muls(next, previous, decay, kElements);
            PipeBarrier<PIPE_V>();
            DotRows(corr, next, key, previous);
            Sub(corr, ub_.GetWithOffset<float>(kWindowElements, kVOffset)[t * kDim], corr, kDim);
            PipeBarrier<PIPE_V>();
            Muls(corr, corr, beta, kDim);
            PipeBarrier<PIPE_V>();
            Brcb(broadcast, corr, 16, {1, 8});
            PipeBarrier<PIPE_V>();
            for (uint32_t column = 0; column != kDim; column += 64) {
                MulAddDst(next[column], key[column], broadcast, 64, kDim,
                          {1, 1, 0, 16, 0, 1});
            }
            PipeBarrier<PIPE_V>();
            DotRows(result, next, query[t * kDim], previous);
            SaveToken(row, head, key_head, begin + t, t, decay);
            bank = 1 - bank;
        }
    }

    VerifyTiling td_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<Activation> q_, k_, v_, out_;
    GlobalTensor<float> g_, correction_, decays_;
    GlobalTensor<State> state_;
    GlobalTensor<Key> keys_;
    GlobalTensor<Beta> beta_;
    GlobalTensor<Index> indices_;
    GlobalTensor<Qsl> qsl_;
    TEventID inputReady_, inputFree_, outputReady_, outputFree_, scalarReady_, scalarResult_;
};
} // namespace verify_detail

template <typename Activation, typename State, typename Key, typename Beta,
          typename Index, typename Qsl>
__global__ __aicore__ void VerifyKernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR state,
    GM_ADDR correction, GM_ADDR key_record, GM_ADDR decay_record,
    GM_ADDR out, GM_ADDR qsl, GM_ADDR indices, VerifyTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    verify_detail::Verify<Activation, State, Key, Beta, Index, Qsl> op(tiling);
    op.Init(q, k, v, g, beta, state, correction, key_record, decay_record, out, qsl, indices);
    op.Process();
}
} // namespace gdn_ac

