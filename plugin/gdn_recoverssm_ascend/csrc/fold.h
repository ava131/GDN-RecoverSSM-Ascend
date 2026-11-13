// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <type_traits>
#include "kernel_operator.h"
#include "common.h"

namespace gdn_ac {

// A task owns one complete (layer, request, value-head) state. Loading the
// original checkpoint once is essential when source/final/boundary pages alias.
// FP32 recurrence remains live in UB; low precision is applied to snapshots
// only, never fed back into subsequent tokens.
template <typename State, typename Key>
class FoldEngine {
    static constexpr uint32_t kDim = 128;
    static constexpr uint32_t kElements = kDim * kDim;
    static constexpr uint32_t kWindow = 8;
    static constexpr uint32_t kStateBytes = kElements * sizeof(float);
    static constexpr uint32_t kIoOffset = kStateBytes;
    static constexpr uint32_t kIoBytes = kElements * sizeof(half);
    static constexpr uint32_t kKeysOffset = kIoOffset + kIoBytes;
    static constexpr uint32_t kWindowBytes = kWindow * kDim * sizeof(float);
    static constexpr uint32_t kCorrectionsOffset = kKeysOffset + kWindowBytes;
    static constexpr uint32_t kRawKeysOffset = kCorrectionsOffset + kWindowBytes;
    static constexpr uint32_t kRawKeyBytes = kWindow * kDim * sizeof(half);
    static constexpr uint32_t kBroadcastOffset = kRawKeysOffset + kRawKeyBytes;
    // Brcb expands each scalar into a 32-byte block, one block per state row.
    static constexpr uint32_t kBroadcastBytes = kDim * 32;
    static constexpr uint32_t kUbBytes = kBroadcastOffset + kBroadcastBytes;
    static_assert(kUbBytes + 2048 <= 192 * 1024, "fold UB exceeds the A2 limit");

public:
    __aicore__ inline FoldEngine(const FoldTiling &td) : td_(td) {}

    __aicore__ inline void Init(
        GM_ADDR state_addrs, GM_ADDR state_block_strides,
        GM_ADDR corr_addrs, GM_ADDR corr_block_strides,
        GM_ADDR key_addrs, GM_ADDR key_block_strides,
        GM_ADDR decay_addrs, GM_ADDR decay_block_strides,
        GM_ADDR layer_groups, GM_ADDR source, GM_ADDR lengths,
        GM_ADDR final_indices, GM_ADDR boundary_indices, GM_ADDR boundary_lengths)
    {
        stateAddrs_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(state_addrs));
        stateBlockStrides_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(state_block_strides));
        corrAddrs_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(corr_addrs));
        corrBlockStrides_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(corr_block_strides));
        keyAddrs_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(key_addrs));
        keyBlockStrides_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(key_block_strides));
        decayAddrs_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(decay_addrs));
        decayBlockStrides_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(decay_block_strides));
        layerGroups_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(layer_groups));
        sources_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(source));
        lengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lengths));
        finals_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(final_indices));
        boundaries_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(boundary_indices));
        boundaryLengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(boundary_lengths));
        pipe_.InitBuffer(ub_, kUbBytes);
        loadReady_ = pipe_.AllocEventID<AscendC::HardEvent::MTE2_V>();
        inputFree_ = pipe_.AllocEventID<AscendC::HardEvent::V_MTE2>();
        outputReady_ = pipe_.AllocEventID<AscendC::HardEvent::V_MTE3>();
        outputDone_ = pipe_.AllocEventID<AscendC::HardEvent::MTE3_V>();
        scalarReady_ = pipe_.AllocEventID<AscendC::HardEvent::S_V>();
    }

    __aicore__ inline void Process()
    {
        const int64_t work = td_.layers * td_.rows * td_.heads;
        for (int64_t task = AscendC::GetBlockIdx(); task < work; task += AscendC::GetBlockNum()) {
            const int64_t head = task % td_.heads;
            const int64_t row = (task / td_.heads) % td_.rows;
            const int64_t layer = task / (td_.heads * td_.rows);
            if (row >= td_.record_rows) {
                continue;
            }
            const int64_t plan = layerGroups_.GetValue(layer) * td_.plan_capacity + row;
            const int64_t source = sources_.GetValue(plan);
            const int32_t accepted = lengths_.GetValue(plan);
            const int64_t final = finals_.GetValue(plan);
            if (source <= td_.null_block || source >= td_.blocks ||
                final <= td_.null_block || final >= td_.blocks ||
                accepted <= 0 || accepted > td_.query_len) {
                continue;
            }
            RunTask(layer, row, head, source, final, accepted, plan);
        }
        pipe_.ReleaseEventID<AscendC::HardEvent::MTE2_V>(loadReady_);
        pipe_.ReleaseEventID<AscendC::HardEvent::V_MTE2>(inputFree_);
        pipe_.ReleaseEventID<AscendC::HardEvent::V_MTE3>(outputReady_);
        pipe_.ReleaseEventID<AscendC::HardEvent::MTE3_V>(outputDone_);
        pipe_.ReleaseEventID<AscendC::HardEvent::S_V>(scalarReady_);
    }

private:
    __aicore__ inline void AwaitInput()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loadReady_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loadReady_);
    }

    __aicore__ inline void InputFree()
    {
        // No prior Vector access to a reusable bank may race the next DMA.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_);
    }

    __aicore__ inline void LoadState(int64_t slot, int64_t head, int64_t blockStride)
    {
        InputFree();
        AscendC::DataCopyExtParams params{
            kDim, static_cast<uint32_t>(kDim * sizeof(State)),
            static_cast<uint32_t>((td_.state_v_stride - kDim) * sizeof(State)), 0, 0};
        AscendC::DataCopyPadExtParams<State> pad{false, 0, 0, 0};
        const int64_t offset = slot * blockStride + head * td_.state_head_stride;
        if constexpr (std::is_same<State, float>::value) {
            AscendC::DataCopyPad(ub_.template GetWithOffset<float>(kElements, 0),
                                state_[offset], params, pad);
        } else {
            AscendC::DataCopyPad(ub_.template GetWithOffset<State>(kElements, kIoOffset),
                                state_[offset], params, pad);
        }
        AwaitInput();
        if constexpr (!std::is_same<State, float>::value) {
            AscendC::Cast(ub_.template GetWithOffset<float>(kElements, 0),
                          ub_.template GetWithOffset<State>(kElements, kIoOffset),
                          AscendC::RoundMode::CAST_NONE, kElements);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void LoadRecords(
        int64_t row, int64_t head, int32_t accepted,
        int64_t corrBlockStride, int64_t keyBlockStride)
    {
        InputFree();
        const int64_t corrOffset = row * corrBlockStride + head * td_.corr_head_stride;
        const int64_t keyHead = head / (td_.heads / td_.key_heads);
        const int64_t keyOffset = row * keyBlockStride + keyHead * td_.key_head_stride;
        AscendC::DataCopyExtParams corrParams{
            static_cast<uint16_t>(accepted), kDim * sizeof(float),
            static_cast<uint32_t>((td_.corr_token_stride - kDim) * sizeof(float)), 0, 0};
        AscendC::DataCopyPadExtParams<float> corrPad{false, 0, 0, 0};
        AscendC::DataCopyPad(ub_.template GetWithOffset<float>(kWindow * kDim, kCorrectionsOffset),
                            corrections_[corrOffset], corrParams, corrPad);
        AscendC::DataCopyExtParams keyParams{
            static_cast<uint16_t>(accepted), static_cast<uint32_t>(kDim * sizeof(Key)),
            static_cast<uint32_t>((td_.key_token_stride - kDim) * sizeof(Key)), 0, 0};
        AscendC::DataCopyPadExtParams<Key> keyPad{false, 0, 0, 0};
        if constexpr (std::is_same<Key, float>::value) {
            AscendC::DataCopyPad(ub_.template GetWithOffset<float>(kWindow * kDim, kKeysOffset),
                                keys_[keyOffset], keyParams, keyPad);
        } else {
            AscendC::DataCopyPad(ub_.template GetWithOffset<Key>(kWindow * kDim, kRawKeysOffset),
                                keys_[keyOffset], keyParams, keyPad);
        }
        AwaitInput();
        if constexpr (!std::is_same<Key, float>::value) {
            AscendC::Cast(ub_.template GetWithOffset<float>(kWindow * kDim, kKeysOffset),
                          ub_.template GetWithOffset<Key>(kWindow * kDim, kRawKeysOffset),
                          AscendC::RoundMode::CAST_NONE, accepted * kDim);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreState(int64_t slot, int64_t head, int64_t blockStride)
    {
        auto state = ub_.template GetWithOffset<float>(kElements, 0);
        if constexpr (!std::is_same<State, float>::value) {
            AscendC::Cast(ub_.template GetWithOffset<State>(kElements, kIoOffset),
                          state, AscendC::RoundMode::CAST_RINT, kElements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_);
        AscendC::DataCopyExtParams params{
            kDim, static_cast<uint32_t>(kDim * sizeof(State)), 0,
            static_cast<uint32_t>((td_.state_v_stride - kDim) * sizeof(State)), 0};
        const int64_t offset = slot * blockStride + head * td_.state_head_stride;
        if constexpr (std::is_same<State, float>::value) {
            AscendC::DataCopyPad(state_[offset], state, params);
        } else {
            AscendC::DataCopyPad(state_[offset],
                                ub_.template GetWithOffset<State>(kElements, kIoOffset), params);
        }
        // A boundary snapshot can be followed by another update, or this UB can
        // be reused by another task. Complete MTE3 before allowing either.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputDone_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputDone_);
    }

    __aicore__ inline void RunTask(
        int64_t layer, int64_t row, int64_t head, int64_t source,
        int64_t final, int32_t accepted, int64_t plan)
    {
        const int64_t stateBlockStride = stateBlockStrides_.GetValue(layer);
        const int64_t corrBlockStride = corrBlockStrides_.GetValue(layer);
        const int64_t keyBlockStride = keyBlockStrides_.GetValue(layer);
        const int64_t decayBlockStride = decayBlockStrides_.GetValue(layer);
        state_.SetGlobalBuffer(reinterpret_cast<__gm__ State *>(stateAddrs_.GetValue(layer)));
        corrections_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(corrAddrs_.GetValue(layer)));
        keys_.SetGlobalBuffer(reinterpret_cast<__gm__ Key *>(keyAddrs_.GetValue(layer)));
        decays_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(decayAddrs_.GetValue(layer)));
        const int64_t boundary = boundaries_.GetValue(plan);
        const int32_t boundaryLength = boundaryLengths_.GetValue(plan);
        const bool saveBoundary = boundary > td_.null_block && boundary < td_.blocks &&
                                  boundary != final && boundaryLength > 0 && boundaryLength <= accepted;
        LoadState(source, head, stateBlockStride);
        LoadRecords(row, head, accepted, corrBlockStride, keyBlockStride);
        auto state = ub_.template GetWithOffset<float>(kElements, 0);
        auto broadcast = ub_.template GetWithOffset<float>(kDim * 8, kBroadcastOffset);
        for (int32_t token = 0; token < accepted; ++token) {
            const float decay = decays_.GetValue(row * decayBlockStride +
                head * td_.decay_head_stride + token * td_.decay_token_stride);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarReady_);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarReady_);
            auto key = ub_.template GetWithOffset<float>(kDim, kKeysOffset + token * kDim * sizeof(float));
            auto corr = ub_.template GetWithOffset<float>(kDim, kCorrectionsOffset + token * kDim * sizeof(float));
            AscendC::Muls(state, state, decay, kElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(broadcast, corr, kDim / 8, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            // Preserve the multiplication-then-multiply/add operation order.
            // MulAddDst avoids explicitly materializing a rounded c*k tensor;
            // equality of device rounding with Triton's tl.fma is established
            // by deployment numerical self-tests, not assumed from this API.
            for (uint32_t half = 0; half < kDim; half += 64) {
                AscendC::MulAddDst(state[half], key[half], broadcast, 64, kDim,
                                  {1, 1, 0, 16, 0, 1});
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (saveBoundary && token + 1 == boundaryLength) {
                StoreState(boundary, head, stateBlockStride);
            }
        }
        StoreState(final, head, stateBlockStride);
    }

    FoldTiling td_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub_;
    AscendC::GlobalTensor<int64_t> stateAddrs_, stateBlockStrides_;
    AscendC::GlobalTensor<int64_t> corrAddrs_, corrBlockStrides_;
    AscendC::GlobalTensor<int64_t> keyAddrs_, keyBlockStrides_;
    AscendC::GlobalTensor<int64_t> decayAddrs_, decayBlockStrides_;
    AscendC::GlobalTensor<int32_t> layerGroups_, sources_, lengths_, finals_, boundaries_, boundaryLengths_;
    AscendC::GlobalTensor<State> state_;
    AscendC::GlobalTensor<float> corrections_, decays_;
    AscendC::GlobalTensor<Key> keys_;
    AscendC::TEventID loadReady_, inputFree_, outputReady_, outputDone_, scalarReady_;
};

template <typename State, typename Key>
__global__ __aicore__ void FoldKernel(
    GM_ADDR state_ref, GM_ADDR correction_ref, GM_ADDR key_ref, GM_ADDR decay_ref,
    GM_ADDR state_addrs, GM_ADDR state_block_strides,
    GM_ADDR corr_addrs, GM_ADDR corr_block_strides,
    GM_ADDR key_addrs, GM_ADDR key_block_strides,
    GM_ADDR decay_addrs, GM_ADDR decay_block_strides,
    GM_ADDR layer_groups, GM_ADDR source, GM_ADDR lengths, GM_ADDR final_indices,
    GM_ADDR boundary_indices, GM_ADDR boundary_lengths, FoldTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)state_ref;
    (void)correction_ref;
    (void)key_ref;
    (void)decay_ref;
    FoldEngine<State, Key> kernel(tiling);
    kernel.Init(state_addrs, state_block_strides, corr_addrs, corr_block_strides,
                key_addrs, key_block_strides, decay_addrs, decay_block_strides,
                layer_groups, source, lengths, final_indices, boundary_indices, boundary_lengths);
    kernel.Process();
}

} // namespace gdn_ac

