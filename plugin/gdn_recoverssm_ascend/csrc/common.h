// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace gdn_ac {
// Host and device share by-value POD tiling; strides are measured in elements.
// Both kernels specialize K=V=128. No device allocation or raw host pointer tables.
#define GDN_VERIFY_FIELDS(X) \
    X(rows) \
    X(tokens) \
    X(heads) \
    X(value_heads) \
    X(query_len) \
    X(blocks) \
    X(record_rows) \
    X(null_block) \
    X(q_token_stride) \
    X(k_token_stride) \
    X(v_token_stride) \
    X(g_token_stride) \
    X(beta_token_stride) \
    X(state_block_stride) \
    X(state_head_stride) \
    X(state_v_stride) \
    X(corr_row_stride) \
    X(corr_head_stride) \
    X(corr_token_stride) \
    X(key_row_stride) \
    X(key_head_stride) \
    X(key_token_stride) \
    X(decay_row_stride) \
    X(decay_head_stride) \
    X(decay_token_stride) \
    X(out_token_stride) \
    X(qsl_stride) \
    X(index_stride)
#define GDN_FOLD_FIELDS(X) \
    X(rows) \
    X(layers) \
    X(heads) \
    X(key_heads) \
    X(query_len) \
    X(blocks) \
    X(record_rows) \
    X(plan_capacity) \
    X(null_block) \
    X(state_head_stride) \
    X(state_v_stride) \
    X(corr_head_stride) \
    X(corr_token_stride) \
    X(key_head_stride) \
    X(key_token_stride) \
    X(decay_head_stride) \
    X(decay_token_stride)
#define GDN_INT_FIELD(name) int64_t name;
struct VerifyTiling {
    GDN_VERIFY_FIELDS(GDN_INT_FIELD)
    float scale;
};
struct FoldTiling {
    GDN_FOLD_FIELDS(GDN_INT_FIELD)
};
#undef GDN_INT_FIELD
constexpr int64_t kAbiVersion = 1;
} // namespace gdn_ac

