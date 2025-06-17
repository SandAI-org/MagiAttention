/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <torch/version.h>  // For TORCH_VERSION* macros
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>
#include <cute/numeric/arithmetic_tuple.hpp>

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "cuda_check.h"

// Copied from https://github.com/pytorch/pytorch/commit/7931eee5c5ebcdf468bff4d308510b03355cd909
// This is so that we can pass in torch.dtype as a parameter to the function.
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 4)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace pybind11::detail {

    template <>
    struct type_caster<at::ScalarType> {
    public:
        // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
        PYBIND11_TYPE_CASTER(at::ScalarType, _("torch.dtype"));
        // PYBIND11_TYPE_CASTER defines a member field called value. at::ScalarType
        // cannot be default-initialized, we provide this constructor to explicitly
        // initialize that field. The value doesn't matter as it will be overwritten
        // after a successful call to load.
        type_caster() : value(at::kFloat) {}
        bool load(handle src, bool) {
            PyObject* obj = src.ptr();
            if (THPDtype_Check(obj)) {
                value = reinterpret_cast<THPDtype*>(obj)->scalar_type;
                return true;
            }
            return false;
        }
        static handle cast(
                           const at::ScalarType& src,
                           return_value_policy /* policy */,
                           handle /* parent */) {
            return Py_NewRef(torch::getTHPDtype(src));
        }
    };

} // namespace pybind11::detail

#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t max_seqlen_q,
                      const size_t max_seqlen_k,
                      const size_t max_seqlen_q_rounded,
                      const size_t max_seqlen_k_rounded,
                      const size_t total_q,
                      const size_t total_k,
                      const size_t h_qo,
                      const size_t h_kv,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor kernel_out,
                      void *q_ranges_d,
                      void *k_ranges_d,
                      void *range_locks_d,
                      void *attn_type_map_d,
                      int merge_batch_size,
                      void *merge_q_ranges_d,
                      void *qk_map_d,
                      void *softmax_lse_d,
                      float softmax_scale,
                      void *tile_count_semaphore_d,
                      float const softcap=0.f,
                      int const sm_margin=0,
                      bool const disable_fwd_atomic_reduction=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.is_fp32_out = kernel_out.dtype() == torch::kFloat32;
    params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;
    params.disable_fwd_atomic_reduction = disable_fwd_atomic_reduction;

    // Set the pointers of Q, K, V
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // Set the strides of Q, K, V
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);

    // Set the pointer of O
    params.o_ptr = kernel_out.data_ptr();
    // Set the strides of O
    // All stride are in elements, not bytes.
    params.o_row_stride = kernel_out.stride(-3);
    params.o_head_stride = kernel_out.stride(-2);

    // Set other pointers
    params.q_ranges = static_cast<int *>(q_ranges_d);
    params.k_ranges = static_cast<int *>(k_ranges_d);
    params.attn_type_map = static_cast<int *>(attn_type_map_d);
    params.merge_q_ranges = static_cast<int *>(merge_q_ranges_d);
    params.qk_map = static_cast<int *>(qk_map_d);

    // Set kernel utility pointers
    params.range_locks = static_cast<int *>(range_locks_d);
    params.tile_count_semaphore = static_cast<int *>(tile_count_semaphore_d);


    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.merge_batch_size = merge_batch_size;
    params.h_qo = h_qo;
    params.h_kv = h_kv;
    params.max_seqlen_q = max_seqlen_q;
    params.max_seqlen_k = max_seqlen_k;
    params.max_seqlen_q_rounded = max_seqlen_q_rounded;
    params.max_seqlen_k_rounded = max_seqlen_k_rounded;
    params.total_q = total_q;
    params.total_k = total_k;
    params.d = d;
    params.d_rounded = d_rounded;
    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.softcap = softcap;

    // Set the architecture and number of SMs to used in the kernel.
    params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
    params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t max_seqlen_q,
                      const size_t max_seqlen_k,
                      const size_t max_seqlen_q_rounded,
                      const size_t max_seqlen_k_rounded,
                      const size_t total_q,
                      const size_t total_k,
                      const size_t h_qo,
                      const size_t h_kv,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *q_ranges_d,
                      void *k_ranges_d,
                      void *attn_type_map_d,
                      int merge_batch_size,
                      void *merge_k_ranges_d,
                      void *bwd_kq_map_d,
                      void *softmax_lse_d,
                      void *softmax_lse_log2_d,
                      void *dsoftmax_sum_d,
                      float softmax_scale,
                      void *tile_count_semaphore_d,
                      const float softcap=0.f,
                      bool deterministic=false,
                      int const sm_margin=0) {

    set_params_fprop(params,
                     b, max_seqlen_q, max_seqlen_k, 
                     max_seqlen_q_rounded, max_seqlen_k_rounded, 
                     total_q, total_k, 
                     h_qo, h_kv, 
                     d, d_rounded,
                     q, k, v, out,
                     /*q_ranges_d*/q_ranges_d, 
                     /*k_ranges_d*/k_ranges_d,
                     /*range_locks_d*/nullptr,
                     /*attn_type_map_d*/attn_type_map_d,
                     /*merge_batch_size*/merge_batch_size,
                     /*merge_q_ranges_d*/nullptr,
                     /*qk_map_d*/nullptr,
                     /*softmax_lse_d*/softmax_lse_d,
                     /*softmax_scale*/softmax_scale,
                     /*tile_count_semaphore_d*/tile_count_semaphore_d,
                     /*softcap*/softcap,
                     /*sm_margin*/sm_margin,
                     /*disable_fwd_atomic_reduction*/false);

    params.merge_k_ranges = static_cast<int *>(merge_k_ranges_d);
    params.bwd_kq_map = static_cast<int *>(bwd_kq_map_d);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);
    
    // Set softmax_lse_log2_ptr and dsoftmax_sum
    params.softmax_lse_log2_ptr = softmax_lse_log2_d;
    params.dsoftmax_sum = dsoftmax_sum_d;
    
    // Set the deterministic flag
    params.deterministic = deterministic;
}

void run_fast_zero_fill(Flash_fwd_params &params, cudaStream_t stream) {
    if (params.is_fp32_out) {
        #ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.d <= 64) { return run_fast_zero_fill_<float, 64>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.d <= 128) { return run_fast_zero_fill_<float, 128>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.d <= 192) { return run_fast_zero_fill_<float, 192>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.d <= 256) { return run_fast_zero_fill_<float, 256>(params, stream); }
        #endif
    } else if (params.is_bf16) {
        #ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.d <= 64) { return run_fast_zero_fill_<cutlass::bfloat16_t, 64>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.d <= 128) { return run_fast_zero_fill_<cutlass::bfloat16_t, 128>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.d <= 192) { return run_fast_zero_fill_<cutlass::bfloat16_t, 192>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.d <= 256) { return run_fast_zero_fill_<cutlass::bfloat16_t, 256>(params, stream); }
        #endif
    } else {
        #ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.d <= 64) { return run_fast_zero_fill_<cutlass::half_t, 64>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.d <= 128) { return run_fast_zero_fill_<cutlass::half_t, 128>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.d <= 192) { return run_fast_zero_fill_<cutlass::half_t, 192>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.d <= 256) { return run_fast_zero_fill_<cutlass::half_t, 256>(params, stream); }
        #endif
    }
}

inline int get_max_headdim() {
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    return 256;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    return 192;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    return 128;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    return 96;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    return 64;
    #endif
    return 0;
}

inline int round_up_headdim(int head_size) {
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    if (head_size <= 64) { return 64; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    if (head_size <= 96) { return 96; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    if (head_size <= 128) { return 128; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    if (head_size <= 192) { return 192; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    if (head_size <= 256) { return 256; }
    #endif
    return 256;
}


void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    ARCH_SWITCH(params.arch, Arch, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
            BOOL_SWITCH(params.disable_fwd_atomic_reduction, DisableFwdAtomicReduction, [&] {
                if (params.is_bf16) {
                    if (params.is_fp32_out) {
                        #ifndef FLASHATTENTION_DISABLE_HDIM64
                        if (params.d <= 64) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 64, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM96
                        if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 96, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM128
                        if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 128, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM192
                        if (params.d <= 192) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 192, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM256
                        if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 256, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                    } else {
                        #ifndef FLASHATTENTION_DISABLE_HDIM64
                        if (params.d <= 64) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, 64, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM96
                        if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, 96, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM128
                        if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, 128, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM192
                        if (params.d <= 192) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, 192, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM256
                        if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, 256, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                    }
                } else {
                    #ifndef FLASHATTENTION_DISABLE_FP16
                    if (params.is_fp32_out) {
                        #ifndef FLASHATTENTION_DISABLE_HDIM64
                        if (params.d <= 64) { return run_mha_fwd_<Arch, cutlass::half_t, float, 64, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM96
                        if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, float, 96, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM128
                        if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, float, 128, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM192
                        if (params.d <= 192) { return run_mha_fwd_<Arch, cutlass::half_t, float, 192, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM256
                        if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, float, 256, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                    }
                    else{
                        #ifndef FLASHATTENTION_DISABLE_HDIM64
                        if (params.d <= 64) { return run_mha_fwd_<Arch, cutlass::half_t, cutlass::half_t, 64, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM96
                        if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, cutlass::half_t, 96, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM128
                        if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, cutlass::half_t, 128, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM192
                        if (params.d <= 192) { return run_mha_fwd_<Arch, cutlass::half_t, cutlass::half_t, 192, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                        #ifndef FLASHATTENTION_DISABLE_HDIM256
                        if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, cutlass::half_t, 256, Has_softcap, DisableFwdAtomicReduction>(params, stream); }
                        #endif
                    }
                    #else
                    TORCH_CHECK(false, "This flash attention build does not support FP16.");
                    #endif
                }
            });
        });
    });
}

// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// h_q: num_heads_qo
// h_k: num_heads_kv
// d: head_size
std::vector<at::Tensor>
mha_fwd(const at::Tensor &q, // (total_q, h_q, d)
        const at::Tensor &k, // (total_k, h_k, d)
        const at::Tensor &v, // (total_k, h_k, d)
        const at::Tensor &q_ranges,  // (b, 2)
        const at::Tensor &k_ranges,  // (b, 2)
        int max_seqlen_q,
        int max_seqlen_k,
        std::optional<const at::Tensor> &attn_type_map_, // (b, )
        std::optional<const at::Tensor> &merge_q_ranges_,
        std::optional<const at::Tensor> &qk_map_,
        float const softmax_scale,
        float const softcap,
        int const sm_margin,
        // performance tuning arguments
        bool const disable_fwd_atomic_reduction,
        std::optional<at::ScalarType> out_type_,
        bool const deterministic_enable
) {
    // Check compute capability
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm9x = dprops->major >= 9;
    TORCH_CHECK(is_sm9x, "Flexible Flash Attention only supports Hopper GPUs or newer.");


    int const batch_size = q_ranges.size(0);
    int const total_q = q.size(0);
    int const total_k = k.size(0);
    int const num_heads_qo = q.size(1);
    int const num_heads_kv = k.size(1);
    int const head_size = q.size(2);

    // Check q, k, v (dtype, device, layout)
    auto q_type = q.scalar_type();
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16, "Flexible Flash Attention only supports fp16 and bf16 data type");
    TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    TORCH_CHECK(q.dim() == 3, "query tensor must be a 3D tensor(total_q, num_heads_qo, head_size)");
    TORCH_CHECK(k.dim() == 3, "key tensor must be a 3D tensor(total_k, num_heads_kv, head_size)");
    TORCH_CHECK(v.dim() == 3, "value tensor must be a 3D tensor(total_k, num_heads_kv, head_size)");
    CHECK_SHAPE(q, total_q, num_heads_qo, head_size);
    CHECK_SHAPE(k, total_k, num_heads_kv, head_size);
    CHECK_SHAPE(v, total_k, num_heads_kv, head_size);
    TORCH_CHECK(q.stride(-1) == 1, "query tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "key tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "value tensor must have contiguous last dimension"); 

    // Check q_ranges (dtype, device, layout)
    TORCH_CHECK(q_ranges.dtype() == torch::kInt32, "q_ranges must have dtype torch.int32");
    CHECK_DEVICE(q_ranges);
    TORCH_CHECK(q_ranges.dim() == 2, "q_ranges must be a 2D tensor");
    TORCH_CHECK(q_ranges.size(1) == 2, "q_ranges must have 2 columns");
    CHECK_SHAPE(q_ranges, batch_size, 2);
    CHECK_CONTIGUOUS(q_ranges);

    // Check k_ranges (dtype, device, layout)
    CHECK_DEVICE(k_ranges); 
    TORCH_CHECK(k_ranges.dtype() == torch::kInt32, "k_ranges must have dtype torch.int32");
    TORCH_CHECK(k_ranges.dim() == 2, "k_ranges must be a 2D tensor");
    TORCH_CHECK(k_ranges.size(1) == 2, "k_ranges must have 2 columns");
    CHECK_SHAPE(k_ranges, batch_size, 2);
    CHECK_CONTIGUOUS(k_ranges);

    // attn_type_map may not given, in this case, we will calculate all attn_slice in with full attention
    at::Tensor attn_type_map;
    bool const has_attn_type_map = attn_type_map_.has_value();
    if (has_attn_type_map) {
        // Check attn_type_map (dtype, device, layout)
        attn_type_map = attn_type_map_.value();
        CHECK_DEVICE(attn_type_map); 
        TORCH_CHECK(attn_type_map.dtype() == torch::kInt32, "attn_type_map must have dtype torch.int32");
        TORCH_CHECK(attn_type_map.dim() == 1, "attn_type_map must be a 1D tensor");
        CHECK_SHAPE(attn_type_map, batch_size);
        CHECK_CONTIGUOUS(attn_type_map);
    }

    int merge_batch_size = batch_size;
    at::Tensor merge_q_ranges;
    bool const has_merge_q_ranges = merge_q_ranges_.has_value();
    if (has_merge_q_ranges) {
        merge_q_ranges = merge_q_ranges_.value();
        // Check merge_q_ranges (dtype, device, layout)
        TORCH_CHECK(merge_q_ranges.dtype() == torch::kInt32, "merge_q_ranges must have dtype torch.int32");
        CHECK_DEVICE(merge_q_ranges);
        merge_batch_size = merge_q_ranges.size(0);
        CHECK_CONTIGUOUS(merge_q_ranges);
    }

    at::Tensor qk_map;
    bool const has_qk_map = qk_map_.has_value();
    if (has_qk_map) {
        qk_map = qk_map_.value();
        // Check qk_map (dtype, device, layout)
        TORCH_CHECK(qk_map.dtype() == torch::kInt32, "qk_map must have dtype torch.int32");
        CHECK_DEVICE(qk_map);
        CHECK_SHAPE(qk_map, merge_batch_size);
        CHECK_CONTIGUOUS(qk_map);
    }
    
    // Check head_size is within the supported range
    int const max_headdim = get_max_headdim();
    TORCH_CHECK(head_size <= max_headdim, "Flexible Flash Attention forward only supports head dimension at most " + std::to_string(max_headdim));
    // Check head_size is a multiple of 8
    int const head_alignment = 8;
    TORCH_CHECK(head_size % head_alignment == 0, "head_size should be a multiple of " + std::to_string(head_alignment));
    // Check num_heads_qo is a multiple of num_heads_kv
    TORCH_CHECK(num_heads_qo % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");

    auto opts = q.options();

    // Determine output dtype
    at::ScalarType out_type;
    if (out_type_.has_value()) {
        TORCH_CHECK(out_type_.value() == at::ScalarType::Half || out_type_.value() == at::ScalarType::BFloat16 || out_type_.value() == at::ScalarType::Float, "Flexible Flash Attention only supports fp16, bf16 and float output dtype");
        out_type = out_type_.value();
    } else {
        out_type = q_type;
    }

    // Define a helper function to round up to multiple of m
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

    // Round head_size to multiple of 8
    int const head_size_rounded = round_up_headdim(head_size);

    // Round max seqlen to multiple of 128
    int const max_seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    int const max_seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // Create softmax_lse tensor, need to satisfy two conditions
    // 1. initialize with -infinity
    // 2. use float32 to ensure numerical stability
    auto softmax_lse = torch::full({num_heads_qo, total_q}, -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));
    
    // Use float32 to ensure numerical stability when enable atomic reduction
    at::ScalarType kernel_out_type = !disable_fwd_atomic_reduction ? at::kFloat : out_type;
    // Create Kernel output tensor
    auto kernel_out = torch::empty({total_q, num_heads_qo, head_size}, opts.dtype(kernel_out_type));  
    // Get element size
    int element_size = (q_type == at::ScalarType::BFloat16) ? sizeof(cutlass::bfloat16_t) : sizeof(cutlass::half_t);
    // Get q block size, used to initialize range_locks
    // FIXME: hack way to get the block size
    int const kBlockM = std::get<0>(tile_size_fwd_sm90(head_size, element_size, softcap > 0.0));
    // Initialize range_locks, ceil_div(total_q, kBlockM) + 1 rows, num_heads_qo columns
    at::Tensor range_locks = torch::empty({(total_q + kBlockM - 1) / kBlockM + 1, num_heads_qo}, opts.dtype(torch::kInt32));
    // Initialize is_first_store_map tensor, used to store whether the first store to the global memory
    // The shape is same as range_locks

    // Create tile_count_semaphore tensor, used to count the number of tiles
    at::Tensor tile_count_semaphore;  
    tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));

    // If atomic reduction is enabled, we need to zero out the out_accum tensor
    if (!disable_fwd_atomic_reduction) {
        range_locks.zero_();
    }

    Flash_fwd_params params;
    
    // Initialize determin_range_locks tensor, the shape is same as range_locks
    at::Tensor determin_range_locks = torch::empty({(total_q + kBlockM - 1) / kBlockM + 1, num_heads_qo}, opts.dtype(torch::kInt32));
    // Initialize determin_conflict_state, num_sm_max rows, ceil_div(total_q, kBlockM) + 1 columns
    int const num_sm_max = 132; // max sm number for H100
    at::Tensor determin_conflict_state = torch::empty({num_sm_max, (total_q + kBlockM - 1) / kBlockM + 1}, opts.dtype(torch::kInt32));

    // If deterministic is enabled, we need to zero out the out_accum tensor and conflict state
    if (deterministic_enable) {
        determin_range_locks.zero_();
        determin_conflict_state.zero_();
    }

    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     max_seqlen_q_rounded, max_seqlen_k_rounded,
                     total_q, total_k,
                     num_heads_qo, num_heads_kv,
                     head_size, head_size_rounded,
                     q, k, v, kernel_out,
                     /*q_ranges*/ q_ranges.data_ptr(),
                     /*k_ranges*/ k_ranges.data_ptr(),
                     /*range_locks*/ range_locks.data_ptr(),
                     /*attn_type_map*/ has_attn_type_map ? attn_type_map.data_ptr() : nullptr,
                     /*merge_batch_size*/ merge_batch_size,
                     /*merge_q_ranges*/ has_merge_q_ranges ? merge_q_ranges.data_ptr() : nullptr,
                     /*qk_map*/ has_qk_map ? qk_map.data_ptr() : nullptr,
                     /*softmax_lse*/ softmax_lse.data_ptr(),
                     /*softmax_scale*/ softmax_scale,
                     /*tile_count_semaphore*/ tile_count_semaphore.data_ptr(),
                     /*softcap*/ softcap,
                     /*sm_margin*/ sm_margin,
                     /*disable_fwd_atomic_reduction*/ disable_fwd_atomic_reduction);

    params.determin_range_locks = deterministic_enable ? determin_range_locks.data_ptr<int>() : nullptr;
    params.determin_conflict_state = deterministic_enable ? determin_conflict_state.data_ptr<int>() : nullptr;
    
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
    run_fast_zero_fill(params, stream);

    // Cast kernel_out to user specified output type
    auto out = kernel_out.to(out_type);
    return {out, softmax_lse};
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    #ifndef FLASHATTENTION_DISABLE_BACKWARD
    ARCH_SWITCH(params.arch, Arch, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
            if (!params.is_bf16) {
                #ifndef FLASHATTENTION_DISABLE_FP16
                #ifndef FLASHATTENTION_DISABLE_HDIM64
                if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::half_t, float, 64, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM96
                if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::half_t, float, 96, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM128
                if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::half_t, float, 128, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM192
                if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::half_t, float, 192, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM256
                if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::half_t, float, 256, Has_softcap>(params, stream); }
                #endif
                #else
                TORCH_CHECK(false, "This flash attention build does not support FP16.");
                #endif
            } else {
                #ifndef FLASHATTENTION_DISABLE_HDIM64
                if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, float, 64, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM96
                if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, float, 96, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM128
                if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, float, 128, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM192
                if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, float, 192, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM256
                if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, float, 256, Has_softcap>(params, stream); }
                #endif
            }
        });
    });
    #endif
}

// b: batch_size
// s_q: seqlen_q
// s_k: seqlen_k
// h_qo: num_heads
// h_kv: num_heads_kv
// d: head_size
std::vector<at::Tensor> mha_bwd(
    const at::Tensor &dout,  // (total_q, h_qo, d)
    const at::Tensor &q,     // (total_q, h_qo, d)
    const at::Tensor &k,     // (total_k, h_kv, d)
    const at::Tensor &v,     // (total_k, h_kv, d)
    const at::Tensor &out,   // (total_q, h_qo, d)
    std::optional<const at::Tensor> &dq_,   // (total_q, h_qo, d)
    std::optional<const at::Tensor> &dk_,   // (total_k, h_kv, d)
    std::optional<const at::Tensor> &dv_,   // (total_k, h_kv, d)
    const at::Tensor &softmax_lse,    // (h_qo, total_q)
    const at::Tensor &q_ranges,  // (b, 2)
    const at::Tensor &k_ranges,  // (b, 2)
    int max_seqlen_q,
    int max_seqlen_k,
    std::optional<const at::Tensor> &attn_type_map_, // (b, )
    std::optional<const at::Tensor> &merge_k_ranges_,
    std::optional<const at::Tensor> &bwd_kq_map_,
    float const softmax_scale,
    float const softcap,
    std::optional<at::ScalarType> out_type_,
    bool const deterministic,
    int const sm_margin) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif

    // Check compute capability
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm9x = dprops->major >= 9;
    TORCH_CHECK(is_sm9x, "Flexible Flash Attention only supports Hopper GPUs or newer.");

    // Get dims of input tensors
    int batch_size = q_ranges.size(0);
    int const total_q = q.size(0);
    int const total_k = k.size(0);
    int const num_heads_qo = q.size(1);
    int const num_heads_kv = k.size(1);
    int const head_size = q.size(2);

    // Check q, k, v, out, dout (dtype, device, layout)
    // dtype
    auto q_type = q.scalar_type();
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16, "Flexible Flash Attention only supports fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_type, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_type, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_type, "query and dout must have the same dtype");
    // device
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); 
    // layout
    CHECK_SHAPE(q, total_q, num_heads_qo, head_size);
    CHECK_SHAPE(out, total_q, num_heads_qo, head_size);
    CHECK_SHAPE(dout, total_q, num_heads_qo, head_size);
    CHECK_SHAPE(k, total_k, num_heads_kv, head_size);
    CHECK_SHAPE(v, total_k, num_heads_kv, head_size);
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    // check softmax_lse (dtype, device, layout)
    TORCH_CHECK(softmax_lse.dtype() == at::kFloat, "softmax_lse must have dtype torch.float32");
    CHECK_DEVICE(softmax_lse);
    CHECK_SHAPE(softmax_lse, num_heads_qo, total_q);
    TORCH_CHECK(softmax_lse.stride(-1) == 1, "softmax_lse tensor must have contiguous last dimension");

    // check q_ranges, k_ranges (dtype, device, layout)
    TORCH_CHECK(q_ranges.dtype() == torch::kInt32, "q_ranges must have dtype torch.int32");
    TORCH_CHECK(k_ranges.dtype() == torch::kInt32, "k_ranges must have dtype torch.int32");
    CHECK_DEVICE(q_ranges); CHECK_DEVICE(k_ranges);
    CHECK_SHAPE(q_ranges, batch_size, 2);
    CHECK_SHAPE(k_ranges, batch_size, 2);
    CHECK_CONTIGUOUS(q_ranges); CHECK_CONTIGUOUS(k_ranges);
    
    // attn_type_map may not given, in this case, we will calculate all attn_slice in with full attention
    at::Tensor attn_type_map;
    bool const has_attn_type_map = attn_type_map_.has_value();
    if (has_attn_type_map) {
        // If attn_type_map is given, check dtype, device and layout.
        attn_type_map = attn_type_map_.value();
        TORCH_CHECK(attn_type_map.dtype() == torch::kInt32, "attn_type_map must have dtype torch.int32");
        CHECK_DEVICE(attn_type_map);
        CHECK_SHAPE(attn_type_map, batch_size);
        CHECK_CONTIGUOUS(attn_type_map);
    }

    // check merge_k_ranges, bwd_kq_map (dtype, device, layout) if given
    at::Tensor merge_k_ranges;
    at::Tensor bwd_kq_map;
    bool const has_merge_k_ranges = merge_k_ranges_.has_value();
    bool const has_bwd_kq_map = bwd_kq_map_.has_value();
    int merge_batch_size = batch_size;
    if (has_merge_k_ranges) {
        merge_k_ranges = merge_k_ranges_.value();
        // HACK
        merge_batch_size = merge_k_ranges.size(0);
        // Check dtype, device and layout
        TORCH_CHECK(merge_k_ranges.dtype() == torch::kInt32, "merge_k_ranges must have dtype torch.int32");
        CHECK_DEVICE(merge_k_ranges);
        CHECK_SHAPE(merge_k_ranges, merge_batch_size, 2);
        CHECK_CONTIGUOUS(merge_k_ranges);
    }
    if (has_bwd_kq_map) {
        bwd_kq_map = bwd_kq_map_.value();
        // Check dtype, device and layout
        TORCH_CHECK(bwd_kq_map.dtype() == torch::kInt32, "bwd_kq_map must have dtype torch.int32");
        CHECK_DEVICE(bwd_kq_map);
        CHECK_SHAPE(bwd_kq_map, merge_batch_size);
    }
    TORCH_CHECK(has_merge_k_ranges == has_bwd_kq_map, "merge_k_ranges and bwd_kq_map must be both given or both not given");
    
    // Check head_size
    int const max_headdim = get_max_headdim();
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= max_headdim, "FlashAttention forward only supports head dimension at most " + std::to_string(max_headdim));
    // Check num_heads_qo and num_heads_kv
    TORCH_CHECK(num_heads_qo % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");
    // Get rounded head_size
    int const head_size_rounded = round_up_headdim(head_size);
    
    // Get block size for tiling
    int element_size = (q_type == at::ScalarType::BFloat16) ? sizeof(cutlass::bfloat16_t) : sizeof(cutlass::half_t);
    int const kBlockM = std::get<0>(tile_size_bwd_sm90(head_size, element_size, softcap > 0.0));
    int const kBlockN = std::get<1>(tile_size_bwd_sm90(head_size, element_size, softcap > 0.0));
    
    // Get rounded max_seqlen
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int const max_seqlen_q_rounded = round_multiple(max_seqlen_q, kBlockM);
    int const max_seqlen_k_rounded = round_multiple(max_seqlen_k, kBlockN);

    // Determine output dtype for dq, dk, dv
    at::ScalarType out_type;
    if (out_type_.has_value()) {
        TORCH_CHECK(out_type_.value() == at::ScalarType::Half || out_type_.value() == at::ScalarType::BFloat16 || out_type_.value() == at::ScalarType::Float, "Flexible Flash Attention only supports fp16, bf16 and float output dtype");
        out_type = out_type_.value();
    } else {
        out_type = q_type;
    }

    // Get tensor options including dtype, device and layout
    auto opts = q.options();

    // dq, dk, dv are output tensors, if they are not given, we will create them.
    // If they are given, we will check dtype, device and layout.
    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.scalar_type() == out_type, "dq must have the same dtype as out_type (or same as query if out_type is not specified)");
        CHECK_DEVICE(dq);
        CHECK_SHAPE(dq, total_q, num_heads_qo, head_size);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    } else {
        dq = torch::zeros_like(q, opts.dtype(out_type));
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == out_type, "dk must have the same dtype as out_type (or same as key if out_type is not specified)");
        CHECK_DEVICE(dk);
        CHECK_SHAPE(dk, total_k, num_heads_kv, head_size);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    } else {
        dk = torch::zeros_like(k, opts.dtype(out_type));
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == out_type, "dv must have the same dtype as out_type (or same as value if out_type is not specified)");
        CHECK_DEVICE(dv);
        CHECK_SHAPE(dv, total_k, num_heads_kv, head_size);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    } else {
        dv = torch::zeros_like(v, opts.dtype(out_type));
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    at::Tensor softmax_d, softmax_lse_log2;
    // Need softmax_d and softmax_lse_log2 to have max_seqlen_q_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
    softmax_d = torch::empty({batch_size, num_heads_qo, max_seqlen_q_rounded}, opts.dtype(at::kFloat));
    softmax_lse_log2 = torch::empty({batch_size, num_heads_qo, max_seqlen_q_rounded}, opts.dtype(at::kFloat));

    // Create tile_count_semaphore tensor, used to count the number of tiles
    at::Tensor tile_count_semaphore;  
    tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));

    Flash_bwd_params params;
    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     max_seqlen_q_rounded, max_seqlen_k_rounded,
                     total_q, total_k,
                     num_heads_qo, num_heads_kv,
                     head_size, head_size_rounded,
                     q, k, v, out, dout,  // input tensors
                     dq, dk, dv,  // output tensors
                     q_ranges.data_ptr(),
                     k_ranges.data_ptr(),
                     has_attn_type_map ? attn_type_map.data_ptr() : nullptr,
                     merge_batch_size,
                     has_merge_k_ranges ? merge_k_ranges.data_ptr() : nullptr,
                     has_bwd_kq_map ? bwd_kq_map.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     softmax_lse_log2.data_ptr(),
                     softmax_d.data_ptr(),
                     softmax_scale,
                     tile_count_semaphore.data_ptr(),
                     softcap,
                     deterministic,
                     sm_margin);

    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
    TORCH_CHECK(params.softcap == 0.0, "This flash attention build does not support tanh softcapping.");
    #endif

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_bwd(params, stream);
    // TODO: using fast zero fill if dq, dk, dv are not given

    return { dq, dk, dv, softmax_d, softmax_lse_log2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlexibleFlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
}
