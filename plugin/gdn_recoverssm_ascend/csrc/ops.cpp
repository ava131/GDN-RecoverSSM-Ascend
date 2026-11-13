// SPDX-License-Identifier: Apache-2.0
// Direct AscendC launch on the existing Torch-NPU stream. No workspace,
// synchronization, model-state allocation, or extra stream on the request path.
#include <algorithm>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "platform/platform_ascendc.h"
#include "common.h"
#include "verify.h"
#include "fold.h"

namespace gdn_ac {
#define COUNT_FIELD(name) + 1
constexpr size_t kVerifyFields = 0 GDN_VERIFY_FIELDS(COUNT_FIELD);
constexpr size_t kFoldFields = 0 GDN_FOLD_FIELDS(COUNT_FIELD);
#undef COUNT_FIELD
using at::Tensor;

void CheckNpu(const Tensor& tensor, const Tensor& reference) {
    TORCH_CHECK(tensor.defined() && tensor.device() == reference.device(),
                "GDN AscendC: all tensors must be on the same NPU");
}
void CheckFloat(at::ScalarType type) {
    TORCH_CHECK(type == at::kFloat || type == at::kBFloat16,
                "GDN AscendC: only FP32/BF16 supported; FP16 uses Triton until cast semantics are verified");
}
void CheckDmaGap(int64_t stride, int64_t bytes) {
    TORCH_CHECK(stride >= 128 && (stride-128) <= 0xFFFFFFFFLL/bytes,
                "GDN AscendC DMA stride byte gap exceeds UINT32 range");
}
uint32_t Blocks(int64_t work) {
    // A worker owns one NPU. Resolve the platform only once after device init.
    static const auto cores = [] {
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        TORCH_CHECK(platform != nullptr, "GDN AscendC platform unavailable");
        uint64_t ubBytes=0;
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);
        TORCH_CHECK(ubBytes>=156800+2048, "GDN AscendC device UB is too small");
        return static_cast<int64_t>(platform->GetCoreNumAiv());
    }();
    TORCH_CHECK(cores > 0 && work > 0, "GDN AscendC invalid work/core count");
    return static_cast<uint32_t>(std::min(cores, work));
}
#define PTR(tensor) reinterpret_cast<GM_ADDR>((tensor).data_ptr())

template<class A, class S, class B>
void LaunchVerify(uint32_t blocks, aclrtStream stream,
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
    const Tensor& beta, const Tensor& state, const Tensor& correction,
    const Tensor& key, const Tensor& decay, const Tensor& out,
    const Tensor& qsl, const Tensor& indices, VerifyTiling t) {
    VerifyKernel<A,S,A,B,int32_t,int32_t><<<blocks,nullptr,stream>>>(
        PTR(q),PTR(k),PTR(v),PTR(g),PTR(beta),PTR(state),PTR(correction),
        PTR(key),PTR(decay),PTR(out),PTR(qsl),PTR(indices),t);
}
template<class A, class S>
void VerifyBeta(uint32_t blocks, aclrtStream stream,
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
    const Tensor& beta, const Tensor& state, const Tensor& correction,
    const Tensor& key, const Tensor& decay, const Tensor& out,
    const Tensor& qsl, const Tensor& indices, VerifyTiling t) {
    if (beta.scalar_type() == at::kFloat) {
        LaunchVerify<A,S,float>(blocks,stream,q,k,v,g,beta,state,correction,key,decay,out,qsl,indices,t);
    } else {
        LaunchVerify<A,S,A>(blocks,stream,q,k,v,g,beta,state,correction,key,decay,out,qsl,indices,t);
    }
}
template<class A>
void VerifyState(uint32_t blocks, aclrtStream stream,
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
    const Tensor& beta, const Tensor& state, const Tensor& correction,
    const Tensor& key, const Tensor& decay, const Tensor& out,
    const Tensor& qsl, const Tensor& indices, VerifyTiling t) {
#define RUN_STATE(S) VerifyBeta<A,S>(blocks,stream,q,k,v,g,beta,state,correction,key,decay,out,qsl,indices,t)
    if (state.scalar_type() == at::kFloat) { RUN_STATE(float); }
    
    else { RUN_STATE(bfloat16_t); }
#undef RUN_STATE
}

void Verify(const Tensor& q,const Tensor& k,const Tensor& v,const Tensor& g,
    const Tensor& beta,const Tensor& state,const Tensor& correction,const Tensor& key,
    const Tensor& decay,const Tensor& out,const Tensor& qsl,const Tensor& indices,
    at::IntArrayRef config,double scale) {
    TORCH_CHECK(config.size() == kVerifyFields, "GDN AscendC verify tiling ABI");
    VerifyTiling t{}; size_t field=0;
#define ASSIGN(name) t.name = config[field++];
    GDN_VERIFY_FIELDS(ASSIGN)
#undef ASSIGN
    t.scale = static_cast<float>(scale);
    TORCH_CHECK(t.null_block==0, "GDN AscendC reserves physical page zero");
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "GDN AscendC requires NPU tensors");
    for (const Tensor* x : {&k,&v,&g,&beta,&state,&correction,&key,&decay,&out,&qsl,&indices}) CheckNpu(*x,q);
    CheckFloat(q.scalar_type()); CheckFloat(state.scalar_type());
    TORCH_CHECK(k.scalar_type()==q.scalar_type() && v.scalar_type()==q.scalar_type()
        && key.scalar_type()==q.scalar_type() && out.scalar_type()==q.scalar_type(), "GDN AscendC activation/key dtype");
    TORCH_CHECK(beta.scalar_type()==q.scalar_type() || beta.scalar_type()==at::kFloat, "GDN AscendC beta dtype");
    TORCH_CHECK(g.scalar_type()==at::kFloat && correction.scalar_type()==at::kFloat && decay.scalar_type()==at::kFloat,
                "GDN AscendC FP32 gate/correction/decay required");
    TORCH_CHECK(qsl.scalar_type()==at::kInt && indices.scalar_type()==at::kInt, "GDN AscendC INT32 row metadata required");
    TORCH_CHECK(q.dim()==4 && k.sizes()==q.sizes() && v.dim()==4 && out.sizes()==v.sizes(), "GDN AscendC activation shape");
    TORCH_CHECK(q.size(0)==1 && q.size(3)==128 && v.size(3)==128 && q.size(1)==v.size(1), "GDN AscendC K=V=128");
    TORCH_CHECK(t.rows>0 && t.tokens>=0 && t.tokens==q.size(1) && t.heads==q.size(2)
        && t.value_heads==v.size(2) && t.heads>0 && t.value_heads%t.heads==0
        && t.query_len>0 && t.query_len<=8 && t.tokens<=t.rows*t.query_len, "GDN AscendC verify geometry");
    TORCH_CHECK(state.dim()==4 && state.size(0)==t.blocks && state.size(1)==t.value_heads
        && state.size(2)==128 && state.size(3)==128 && state.stride(3)==1 && state.stride(2)>=128,
        "GDN AscendC state geometry");
    TORCH_CHECK(qsl.dim()==1 && qsl.numel()>=t.rows+1 && indices.dim()==1 && indices.numel()>=t.rows,
                "GDN AscendC row capacity");
    TORCH_CHECK(correction.dim()==4 && key.dim()==4 && decay.dim()==3 && t.record_rows==correction.size(0)
        && correction.size(1)==t.value_heads && correction.size(2)==t.query_len && correction.size(3)==128
        && key.size(0)==t.record_rows && key.size(1)==t.heads && key.size(2)==t.query_len && key.size(3)==128
        && decay.size(0)==t.record_rows && decay.size(1)==t.value_heads && decay.size(2)==t.query_len,
        "GDN AscendC record geometry");
    TORCH_CHECK(correction.stride(3)==1 && key.stride(3)==1 && decay.stride(2)==1,
                "GDN AscendC contiguous record elements required");
    TORCH_CHECK(g.dim()==2 && beta.dim()==2 && g.size(0)==t.tokens && beta.size(0)==t.tokens
        && g.size(1)==t.value_heads && beta.size(1)==t.value_heads && g.stride(1)==1 && beta.stride(1)==1,
        "GDN AscendC gate geometry");
    TORCH_CHECK(q.stride(3)==1 && q.stride(2)==128 && k.stride(3)==1 && k.stride(2)==128
        && v.stride(3)==1 && v.stride(2)==128 && out.stride(3)==1 && out.stride(2)==128,
        "GDN AscendC activation elements");
    CheckDmaGap(state.stride(2),state.element_size());
    CheckDmaGap(q.stride(1),q.element_size());
    CheckDmaGap(k.stride(1),k.element_size());
    CheckDmaGap(v.stride(1),v.element_size());
    TORCH_CHECK(t.q_token_stride==q.stride(1) && t.k_token_stride==k.stride(1) && t.v_token_stride==v.stride(1)
        && t.g_token_stride==g.stride(0) && t.beta_token_stride==beta.stride(0)
        && t.state_block_stride==state.stride(0) && t.state_head_stride==state.stride(1) && t.state_v_stride==state.stride(2)
        && t.corr_row_stride==correction.stride(0) && t.corr_head_stride==correction.stride(1) && t.corr_token_stride==correction.stride(2)
        && t.key_row_stride==key.stride(0) && t.key_head_stride==key.stride(1) && t.key_token_stride==key.stride(2)
        && t.decay_row_stride==decay.stride(0) && t.decay_head_stride==decay.stride(1) && t.decay_token_stride==decay.stride(2)
        && t.out_token_stride==out.stride(1) && t.qsl_stride==qsl.stride(0) && t.index_stride==indices.stride(0),
        "GDN AscendC verify config does not match tensor strides");
    const c10::OptionalDeviceGuard guard(q.device());
    const auto stream=c10_npu::getCurrentNPUStream().stream(false);
    const auto blocks=Blocks(t.rows*t.value_heads);
    auto launch=[=]() -> int {
#define RUN_ACT(A) VerifyState<A>(blocks,stream,q,k,v,g,beta,state,correction,key,decay,out,qsl,indices,t)
        if(q.scalar_type()==at::kFloat) { RUN_ACT(float); }
        
        else { RUN_ACT(bfloat16_t); }
#undef RUN_ACT
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("GDNReplayVerifyAC",launch);
}

template<class S,class K>
void LaunchFold(uint32_t blocks,aclrtStream stream,
    const Tensor& state,const Tensor& correction,const Tensor& key,const Tensor& decay,const Tensor& state_addrs,const Tensor& state_block_strides,const Tensor& corr_addrs,const Tensor& corr_block_strides,const Tensor& key_addrs,const Tensor& key_block_strides,const Tensor& decay_addrs,const Tensor& decay_block_strides,const Tensor& layer_groups,const Tensor& source,const Tensor& lengths,const Tensor& final_indices,const Tensor& boundary_indices,const Tensor& boundary_lengths,FoldTiling t) {
    FoldKernel<S,K><<<blocks,nullptr,stream>>>(PTR(state),PTR(correction),PTR(key),PTR(decay),PTR(state_addrs),PTR(state_block_strides),PTR(corr_addrs),PTR(corr_block_strides),PTR(key_addrs),PTR(key_block_strides),PTR(decay_addrs),PTR(decay_block_strides),PTR(layer_groups),PTR(source),PTR(lengths),PTR(final_indices),PTR(boundary_indices),PTR(boundary_lengths),t);
}
template<class S>
void FoldKey(uint32_t blocks,aclrtStream stream,
    const Tensor& state,const Tensor& correction,const Tensor& key,const Tensor& decay,const Tensor& state_addrs,const Tensor& state_block_strides,const Tensor& corr_addrs,const Tensor& corr_block_strides,const Tensor& key_addrs,const Tensor& key_block_strides,const Tensor& decay_addrs,const Tensor& decay_block_strides,const Tensor& layer_groups,const Tensor& source,const Tensor& lengths,const Tensor& final_indices,const Tensor& boundary_indices,const Tensor& boundary_lengths,FoldTiling t) {
#define RUN_KEY(K) LaunchFold<S,K>(blocks,stream,state,correction,key,decay,state_addrs,state_block_strides,corr_addrs,corr_block_strides,key_addrs,key_block_strides,decay_addrs,decay_block_strides,layer_groups,source,lengths,final_indices,boundary_indices,boundary_lengths,t)
    if(key.scalar_type()==at::kFloat) { RUN_KEY(float); }
    
    else { RUN_KEY(bfloat16_t); }
#undef RUN_KEY
}
void Fold(const Tensor& state,const Tensor& correction,const Tensor& key,const Tensor& decay,const Tensor& state_addrs,const Tensor& state_block_strides,const Tensor& corr_addrs,const Tensor& corr_block_strides,const Tensor& key_addrs,const Tensor& key_block_strides,const Tensor& decay_addrs,const Tensor& decay_block_strides,const Tensor& layer_groups,const Tensor& source,const Tensor& lengths,const Tensor& final_indices,const Tensor& boundary_indices,const Tensor& boundary_lengths,at::IntArrayRef config) {
    TORCH_CHECK(config.size()==kFoldFields,"GDN AscendC fold tiling ABI");
    FoldTiling t{}; size_t field=0;
#define ASSIGN(name) t.name=config[field++];
    GDN_FOLD_FIELDS(ASSIGN)
#undef ASSIGN
    TORCH_CHECK(t.null_block==0, "GDN AscendC reserves physical page zero");
    TORCH_CHECK(state.device().type()==c10::DeviceType::PrivateUse1,"GDN AscendC requires NPU");
    CheckNpu(correction,state);
    CheckNpu(key,state);
    CheckNpu(decay,state);
    CheckNpu(state_addrs,state);
    CheckNpu(state_block_strides,state);
    CheckNpu(corr_addrs,state);
    CheckNpu(corr_block_strides,state);
    CheckNpu(key_addrs,state);
    CheckNpu(key_block_strides,state);
    CheckNpu(decay_addrs,state);
    CheckNpu(decay_block_strides,state);
    CheckNpu(layer_groups,state);
    CheckNpu(source,state);
    CheckNpu(lengths,state);
    CheckNpu(final_indices,state);
    CheckNpu(boundary_indices,state);
    CheckNpu(boundary_lengths,state);
    CheckFloat(state.scalar_type()); CheckFloat(key.scalar_type());
    TORCH_CHECK(correction.scalar_type()==at::kFloat && decay.scalar_type()==at::kFloat, "GDN AscendC FP32 records required");
    for(const Tensor* x : {&state_addrs,&state_block_strides,&corr_addrs,&corr_block_strides,&key_addrs,&key_block_strides,&decay_addrs,&decay_block_strides}) {
        TORCH_CHECK(x->scalar_type()==at::kLong && x->dim()==1 && x->is_contiguous() && x->numel()==t.layers,
                    "GDN AscendC invalid layer address/stride table");
    }
    TORCH_CHECK(layer_groups.scalar_type()==at::kInt && layer_groups.dim()==1 && layer_groups.is_contiguous()
        && layer_groups.numel()==t.layers,"GDN AscendC invalid layer group table");
    for(const Tensor* x : {&source,&lengths,&final_indices,&boundary_indices,&boundary_lengths}) {
        TORCH_CHECK(x->scalar_type()==at::kInt && x->dim()==1 && x->is_contiguous()
            && x->numel()==source.numel(),"GDN AscendC invalid plan tensor");
    }
    TORCH_CHECK(t.layers>0 && t.rows>0 && t.rows<=t.record_rows && t.rows<=t.plan_capacity
        && t.plan_capacity>0 && source.numel()%t.plan_capacity==0
        && source.numel()/t.plan_capacity>0 && t.heads>0 && t.key_heads>0 && t.heads%t.key_heads==0
        && t.query_len>0 && t.query_len<=8, "GDN AscendC fold geometry");
    TORCH_CHECK(state.dim()==4 && state.size(1)==t.heads && state.size(2)==128 && state.size(3)==128
        && state.stride(3)==1 && state.stride(2)>=128,"GDN AscendC fold state geometry");
    TORCH_CHECK(correction.dim()==4 && correction.size(0)>=t.record_rows && correction.size(1)==t.heads
        && correction.size(2)==t.query_len && correction.size(3)==128 && correction.stride(3)==1
        && key.dim()==4 && key.size(0)>=t.record_rows && key.size(1)==t.key_heads
        && key.size(2)==t.query_len && key.size(3)==128 && key.stride(3)==1
        && decay.dim()==3 && decay.size(0)>=t.record_rows && decay.size(1)==t.heads
        && decay.size(2)==t.query_len && decay.stride(2)==1,"GDN AscendC fold record geometry");
    TORCH_CHECK(t.state_head_stride==state.stride(1) && t.state_v_stride==state.stride(2)
        && t.corr_head_stride==correction.stride(1) && t.corr_token_stride==correction.stride(2)
        && t.key_head_stride==key.stride(1) && t.key_token_stride==key.stride(2)
        && t.decay_head_stride==decay.stride(1) && t.decay_token_stride==decay.stride(2),
        "GDN AscendC fold config does not match reference strides");
    CheckDmaGap(state.stride(2),state.element_size());
    CheckDmaGap(correction.stride(2),correction.element_size());
    CheckDmaGap(key.stride(2),key.element_size());
    const c10::OptionalDeviceGuard guard(state.device());
    const auto stream=c10_npu::getCurrentNPUStream().stream(false);
    const auto blocks=Blocks(t.layers*t.rows*t.heads);
    auto launch=[=]() -> int {
#define RUN_STATE(S) FoldKey<S>(blocks,stream,state,correction,key,decay,state_addrs,state_block_strides,corr_addrs,corr_block_strides,key_addrs,key_block_strides,decay_addrs,decay_block_strides,layer_groups,source,lengths,final_indices,boundary_indices,boundary_lengths,t)
        if(state.scalar_type()==at::kFloat) { RUN_STATE(float); }
        
        else { RUN_STATE(bfloat16_t); }
#undef RUN_STATE
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("GDNReplayFoldAC",launch);
}
#undef PTR
} // namespace gdn_ac

TORCH_LIBRARY(gdn_recoverssm_ac,m) {
    m.def("abi_version() -> int", []() -> int64_t { return gdn_ac::kAbiVersion; });
    m.def("verify(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state, Tensor(a!) correction, Tensor(b!) key, Tensor(c!) decay, Tensor(d!) out, Tensor qsl, Tensor indices, int[] config, float scale) -> ()");
    m.def("fold(Tensor(a!) state, Tensor correction, Tensor key, Tensor decay, Tensor state_addrs, Tensor state_block_strides, Tensor corr_addrs, Tensor corr_block_strides, Tensor key_addrs, Tensor key_block_strides, Tensor decay_addrs, Tensor decay_block_strides, Tensor layer_groups, Tensor source, Tensor lengths, Tensor final_indices, Tensor boundary_indices, Tensor boundary_lengths, int[] config) -> ()");
}
TORCH_LIBRARY_IMPL(gdn_recoverssm_ac,PrivateUse1,m) {
    m.impl("verify",TORCH_FN(gdn_ac::Verify));
    m.impl("fold",TORCH_FN(gdn_ac::Fold));
}
TORCH_LIBRARY_IMPL(gdn_recoverssm_ac,Meta,m) {
    m.impl("verify", [](const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
        const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
        const at::Tensor&,const at::Tensor&,const at::Tensor&,at::IntArrayRef,double) {});
    m.impl("fold", [](const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::IntArrayRef) {});
}

