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
#include "heuristics.h"
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

    // Set kernel utility pointers
    params.range_locks = static_cast<int *>(range_locks_d);
    params.tile_count_semaphore = static_cast<int *>(tile_count_semaphore_d);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
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

// void set_params_dgrad(Flash_bwd_params &params,
//                       // sizes
//                       const size_t b,
//                       const size_t seqlen_q,
//                       const size_t seqlen_k,
//                       const size_t seqlen_q_rounded,
//                       const size_t seqlen_k_rounded,
//                       const size_t h,
//                       const size_t h_k,
//                       const size_t d,
//                       const size_t d_rounded,
//                       // device pointers
//                       const at::Tensor q,
//                       const at::Tensor k,
//                       const at::Tensor v,
//                       const at::Tensor out,
//                       const at::Tensor dout,
//                       at::Tensor dq,
//                       at::Tensor dk,
//                       at::Tensor dv,
//                       void *q_ranges_d,
//                       void *k_ranges_d,
//                       void *cu_seqlens_q_d,
//                       void *cu_seqlens_k_d,
//                       void *seqused_q,
//                       void *seqused_k,
//                       void *attn_type_map,
//                       void *dq_accum_d,
//                       void *dk_accum_d,
//                       void *dv_accum_d,
//                       void *softmax_lse_d,
//                       void *dsoftmax_sum_d,
//                       float p_dropout,
//                       float softmax_scale,
//                       int window_size_left,
//                       int window_size_right,
//                       const float softcap=0.f,
//                       bool deterministic=false,
//                       int const sm_margin=0) {

//     set_params_fprop(params,
//                      b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
//                      q, k, v, out,
//                      q_ranges_d, k_ranges_d,
//                      cu_seqlens_q_d,
//                      cu_seqlens_k_d,
//                      seqused_q,
//                      seqused_k,
//                      attn_type_map,
//                      softmax_lse_d,
//                      p_dropout,
//                      softmax_scale,
//                      window_size_left,
//                      window_size_right,
//                      softcap,
//                      sm_margin);

//     // Set the pointers and strides.
//     params.do_ptr = dout.data_ptr();
//     params.do_row_stride = dout.stride(-3);
//     params.do_head_stride = dout.stride(-2);
//     params.dq_ptr = dq.data_ptr();
//     params.dk_ptr = dk.data_ptr();
//     params.dv_ptr = dv.data_ptr();
//     params.dq_row_stride = dq.stride(-3);
//     params.dk_row_stride = dk.stride(-3);
//     params.dv_row_stride = dv.stride(-3);
//     params.dq_head_stride = dq.stride(-2);
//     params.dk_head_stride = dk.stride(-2);
//     params.dv_head_stride = dv.stride(-2);

//     if (cu_seqlens_q_d == nullptr) {
//         params.do_batch_stride = dout.stride(0);
//         params.dq_batch_stride = dq.stride(0);
//         params.dk_batch_stride = dk.stride(0);
//         params.dv_batch_stride = dv.stride(0);
//     }

//     params.dq_accum_ptr = dq_accum_d;
//     params.dk_accum_ptr = dk_accum_d;
//     params.dv_accum_ptr = dv_accum_d;

//     // Softmax sum
//     params.dsoftmax_sum = dsoftmax_sum_d;

//     params.deterministic = deterministic;
// }

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // HEADDIM_SWITCH(params.d, [&] {
    //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
    // });
    ARCH_SWITCH(params.arch, Arch, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
            if (params.is_bf16) {
                if (params.is_fp32_out) {
                    #ifndef FLASHATTENTION_DISABLE_HDIM64
                    if (params.d <= 64) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 64, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM96
                    if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 96, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM128
                    if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 128, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM192
                    if (params.d <= 192) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 192, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM256
                    if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 256, Has_softcap>(params, stream); }
                    #endif
                } else {
                    #ifndef FLASHATTENTION_DISABLE_HDIM64
                    if (params.d <= 64) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, 64, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM96
                    if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, 96, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM128
                    if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, 128, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM192
                    if (params.d <= 192) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, 192, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM256
                    if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, 256, Has_softcap>(params, stream); }
                    #endif
                }
            } else {
                #ifndef FLASHATTENTION_DISABLE_FP16
                if (params.is_fp32_out) {
                    #ifndef FLASHATTENTION_DISABLE_HDIM64
                    if (params.d <= 64) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 64, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM96
                    if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 96, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM128
                    if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 128, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM192
                    if (params.d <= 192) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 192, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM256
                    if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, float, 256, Has_softcap>(params, stream); }
                    #endif
                }
                else{
                    #ifndef FLASHATTENTION_DISABLE_HDIM64
                    if (params.d <= 64) { return run_mha_fwd_<Arch, cutlass::half_t, cutlass::half_t, 64, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM96
                    if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, cutlass::half_t, 96, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM128
                    if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, cutlass::half_t, 128, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM192
                    if (params.d <= 192) { return run_mha_fwd_<Arch, cutlass::half_t, cutlass::half_t, 192, Has_softcap>(params, stream); }
                    #endif
                    #ifndef FLASHATTENTION_DISABLE_HDIM256
                    if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, cutlass::half_t, 256, Has_softcap>(params, stream); }
                    #endif
                }
                #else
                TORCH_CHECK(false, "This flash attention build does not support FP16.");
                #endif
            }
        });
    });
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

inline int round_up_headdimv(int head_size) {
    if (head_size <= 64) { return 64; }
    if (head_size <= 96) { return 96; }
    if (head_size <= 128) { return 128; }
    if (head_size <= 192) { return 192; }
    if (head_size <= 256) { return 256; }
    return 512;
}

// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
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
        float const softmax_scale,
        float const softcap,
        int const sm_margin,
        // performance tuning arguments
        bool const disable_fwd_atomic_reduction,
        std::optional<at::ScalarType> out_type_
) {
    // Check compute capability
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm9x = dprops->major >= 9;
    TORCH_CHECK(is_sm9x, "Flexible Flash Attention only supports Hopper GPUs or newer.");

    // Check dtype
    auto q_type = q.scalar_type();
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16, "Flexible Flash Attention only supports fp16 and bf16 data type");
    TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
    TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");

    // Check device and contiguity
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    // Check q_ranges
    CHECK_DEVICE(q_ranges); CHECK_CONTIGUOUS(q_ranges);
    TORCH_CHECK(q_ranges.dtype() == torch::kInt32, "q_ranges must have dtype torch.int32");
    TORCH_CHECK(q_ranges.dim() == 2, "q_ranges must be a 2D tensor");
    TORCH_CHECK(q_ranges.size(1) == 2, "q_ranges must have 2 columns");

    // Check k_ranges
    CHECK_DEVICE(k_ranges); CHECK_CONTIGUOUS(k_ranges);
    TORCH_CHECK(k_ranges.dtype() == torch::kInt32, "k_ranges must have dtype torch.int32");
    TORCH_CHECK(k_ranges.dim() == 2, "k_ranges must be a 2D tensor");
    TORCH_CHECK(k_ranges.size(1) == 2, "k_ranges must have 2 columns");

    // attn_type_map may not given, in this case, we will calculate all attn_slice in with full attention
    at::Tensor attn_type_map;
    bool const has_attn_type_map = attn_type_map_.has_value();
    if (has_attn_type_map) {
        // Check attn_type_map
        attn_type_map = attn_type_map_.value();
        CHECK_DEVICE(attn_type_map); CHECK_CONTIGUOUS(attn_type_map);
        TORCH_CHECK(attn_type_map.dtype() == torch::kInt32, "attn_type_map must have dtype torch.int32");
        TORCH_CHECK(attn_type_map.dim() == 1, "attn_type_map must be a 1D tensor");
    }

    const int batch_size = q_ranges.size(0);
    int const total_q = q.size(0);
    int const total_k = k.size(0);
    int const num_heads_qo = q.size(1);
    int const num_heads_kv = k.size(1);
    int const head_size = q.size(2);
    
    // Check head_size is within the supported range
    int const max_headdim = get_max_headdim();
    TORCH_CHECK(head_size <= max_headdim, "Flexible Flash Attention forward only supports head dimension at most " + std::to_string(max_headdim));
    // Check head_size is a multiple of 8
    int const head_alignment = 8;
    TORCH_CHECK(head_size % head_alignment == 0, "head_size should be a multiple of " + std::to_string(head_alignment));
    // Check num_heads_qo is a multiple of num_heads_kv
    TORCH_CHECK(num_heads_qo % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");

    // Check shape q
    CHECK_SHAPE(q, total_q, num_heads_qo, head_size);

    // Check shape k
    CHECK_SHAPE(k, total_k, num_heads_kv, head_size);

    // Check shape v
    CHECK_SHAPE(v, total_k, num_heads_kv, head_size);

    // Check shape q_ranges
    CHECK_SHAPE(q_ranges, batch_size, 2);

    // Check shape k_ranges
    CHECK_SHAPE(k_ranges, batch_size, 2);

    // Check shape attn_type_map
    if (has_attn_type_map) {
        CHECK_SHAPE(attn_type_map, batch_size);
    }

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
    at::ScalarType kernel_out_type = !disable_fwd_atomic_reduction ? at::kFloat : q_type;
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
                     /*softmax_lse*/ softmax_lse.data_ptr(),
                     /*softmax_scale*/ softmax_scale,
                     /*tile_count_semaphore*/ tile_count_semaphore.data_ptr(),
                     /*softcap*/ softcap,
                     /*sm_margin*/ sm_margin,
                     /*disable_fwd_atomic_reduction*/ disable_fwd_atomic_reduction);
        
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
    // TODO: ADD kernel for fast zeros filling

    // Cast kernel_out to user specified output type
    auto out = kernel_out.to(out_type);

    return {out, softmax_lse};
}

// void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
//     #ifndef FLASHATTENTION_DISABLE_BACKWARD
//         // FP16_SWITCH(!params.is_bf16, [&] {
//         //     HEADDIM_SWITCH(params.d, [&] {
//         //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
//         //     });
//         // });
//     ARCH_SWITCH(params.arch, Arch, [&] {
//         SOFTCAP_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
//             if (!params.is_bf16) {
//                 #ifndef FLASHATTENTION_DISABLE_FP16
//                 #ifndef FLASHATTENTION_DISABLE_HDIM64
//                 if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::half_t, 64, Has_softcap>(params, stream); }
//                 #endif
//                 #ifndef FLASHATTENTION_DISABLE_HDIM96
//                 if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::half_t, 96, Has_softcap>(params, stream); }
//                 #endif
//                 #ifndef FLASHATTENTION_DISABLE_HDIM128
//                 if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::half_t, 128, Has_softcap>(params, stream); }
//                 #endif
//                 #ifndef FLASHATTENTION_DISABLE_HDIM192
//                 if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::half_t, 192, Has_softcap>(params, stream); }
//                 #endif
//                 #ifndef FLASHATTENTION_DISABLE_HDIM256
//                 if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::half_t, 256, Has_softcap>(params, stream); }
//                 #endif
//                 #else
//                 TORCH_CHECK(false, "This flash attention build does not support FP16.");
//                 #endif
//             } else {
//                 #ifndef FLASHATTENTION_DISABLE_HDIM64
//                 if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 64, Has_softcap>(params, stream); }
//                 #endif
//                 #ifndef FLASHATTENTION_DISABLE_HDIM96
//                 if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 96, Has_softcap>(params, stream); }
//                 #endif
//                 #ifndef FLASHATTENTION_DISABLE_HDIM128
//                 if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 128, Has_softcap>(params, stream); }
//                 #endif
//                 #ifndef FLASHATTENTION_DISABLE_HDIM192
//                 if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 192, Has_softcap>(params, stream); }
//                 #endif
//                 #ifndef FLASHATTENTION_DISABLE_HDIM256
//                 if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 256, Has_softcap>(params, stream); }
//                 #endif
//             }
//         });
//     });
//     #endif
// }


// b: batch_size
// s_q: seqlen_q
// s_k: seqlen_k
// h: num_heads
// h_k: num_heads_k
// d: head_size
// std::vector<at::Tensor> mha_bwd(
//     const at::Tensor &dout,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
//     const at::Tensor &q,     // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
//     const at::Tensor &k,     // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
//     const at::Tensor &v,     // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
//     const at::Tensor &out,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
//     const at::Tensor &softmax_lse,    // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
//     std::optional<at::Tensor> &dq_,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
//     std::optional<at::Tensor> &dk_,   // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
//     std::optional<at::Tensor> &dv_,   // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
//     std::optional<const at::Tensor> &q_ranges_,  // (b, 2)
//     std::optional<const at::Tensor> &k_ranges_,  // (b, 2)
//     std::optional<const at::Tensor> &cu_seqlens_q_,   // b+1
//     std::optional<const at::Tensor> &cu_seqlens_k_,   // b+1
//     std::optional<const at::Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
//     std::optional<const at::Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
//     std::optional<int> max_seqlen_q_,
//     std::optional<int> max_seqlen_k_,
//     std::optional<const at::Tensor> &attn_type_map_, // b. If given, only this many elements of each batch element's queries and outputs are used.
//     float const softmax_scale,
//     bool is_causal,
//     int window_size_left,
//     int window_size_right,
//     float const softcap,
//     bool const deterministic,
//     int const sm_margin) {

//     #ifdef FLASHATTENTION_DISABLE_BACKWARD
//         TORCH_CHECK(false, "This flash attention build does not support backward.");
//     #endif

//     auto dprops = at::cuda::getCurrentDeviceProperties();
//     bool is_sm8x = dprops->major >= 8;
//     TORCH_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

//     auto q_type = q.dtype();
//     TORCH_CHECK(q_type == torch::kFloat16 || q_type == torch::kBFloat16,
//                 "FlashAttention only support fp16 and bf16 data type");
//     TORCH_CHECK(k.dtype() == q_type, "query and key must have the same dtype");
//     TORCH_CHECK(v.dtype() == q_type, "query and value must have the same dtype");
//     TORCH_CHECK(out.dtype() == q_type, "query and out must have the same dtype");
//     TORCH_CHECK(dout.dtype() == q_type, "query and dout must have the same dtype");

//     CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
//     CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

//     TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
//     TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

//     at::Tensor cu_seqlens_q;
//     bool const is_varlen_q = cu_seqlens_q_.has_value();
//     if (is_varlen_q) {
//         cu_seqlens_q = cu_seqlens_q_.value();
//         CHECK_DEVICE(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_q);
//         TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
//         TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
//     }
//     at::Tensor cu_seqlens_k;
//     bool const is_varlen_k = cu_seqlens_k_.has_value();
//     if (is_varlen_k) {
//         cu_seqlens_k = cu_seqlens_k_.value();
//         CHECK_DEVICE(cu_seqlens_k); CHECK_CONTIGUOUS(cu_seqlens_k);
//         TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");
//         TORCH_CHECK(max_seqlen_k_.has_value(), "max_seqlen_k must be provided if cu_seqlens_k is provided");
//     }

//     at::Tensor q_ranges;
//     bool const is_flex_q = q_ranges_.has_value();
//     if (is_flex_q) {
//         q_ranges = q_ranges_.value();
//         CHECK_DEVICE(q_ranges); CHECK_CONTIGUOUS(q_ranges);
//         TORCH_CHECK(q_ranges.dtype() == torch::kInt32, "q_ranges must have dtype torch.int32");
//         TORCH_CHECK(q_ranges.dim() == 2, "q_ranges must be a 2D tensor");
//         TORCH_CHECK(q_ranges.size(1) == 2, "q_ranges must have 2 columns");
//         TORCH_CHECK(!is_varlen_q, "q_ranges is conflict with varlen queries");
//     }
//     at::Tensor k_ranges;
//     bool const is_flex_k = k_ranges_.has_value();
//     if (is_flex_k) {
//         k_ranges = k_ranges_.value();
//         CHECK_DEVICE(k_ranges); CHECK_CONTIGUOUS(k_ranges);
//         TORCH_CHECK(k_ranges.dtype() == torch::kInt32, "k_ranges must have dtype torch.int32");
//         TORCH_CHECK(k_ranges.dim() == 2, "k_ranges must be a 2D tensor");
//         TORCH_CHECK(k_ranges.size(1) == 2, "k_ranges must have 2 columns");
//         TORCH_CHECK(!is_varlen_k, "k_ranges is conflict with varlen keys");
//     }

//     at::Tensor attn_type_map;
//     bool const has_attn_type_map = attn_type_map_.has_value();
//     if (has_attn_type_map) {
//         attn_type_map = attn_type_map_.value();
//         CHECK_DEVICE(attn_type_map); CHECK_CONTIGUOUS(attn_type_map);
//         TORCH_CHECK(attn_type_map.dtype() == torch::kInt32, "attn_type_map must have dtype torch.int32");
//         TORCH_CHECK(attn_type_map.dim() == 1, "attn_type_map must be a 1D tensor");
//     }

//     // This is what we will template on
//     bool const is_varlen = is_varlen_q || is_varlen_k || is_flex_q || is_flex_k || seqused_q_.has_value() || seqused_k_.has_value();
//     #ifdef FLASHATTENTION_DISABLE_VARLEN
//         TORCH_CHECK(!is_varlen, "This flash attention build does not support varlen.");
//     #endif

//     auto const sizes = q.sizes();
//     int const batch_size = !(is_varlen_q || is_flex_q) ? sizes[0] : (!is_flex_q ? cu_seqlens_q.size(0) - 1 : q_ranges.size(0));
//     int const seqlen_q = !(is_varlen_q || is_flex_q) ? sizes[1] : max_seqlen_q_.value();
//     int const total_q = !(is_varlen_q || is_flex_q) ? batch_size * sizes[1] : sizes[0];
//     int const num_heads = q.size(-2);
//     int const head_size = q.size(-1);
//     int const seqlen_k = !(is_varlen_k || is_flex_k) ? k.size(1) : max_seqlen_k_.value();
//     int const total_k = !(is_varlen_k || is_flex_k) ? batch_size * k.size(1) : k.size(0);
//     int const num_heads_k = k.size(-2);
//     TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
//     int const max_headdim = get_max_headdim();
//     TORCH_CHECK(head_size <= max_headdim, "FlashAttention forward only supports head dimension at most " + std::to_string(max_headdim));
//     TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

//     // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and is_causal to set kBlockM
//     if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
//     if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
//     if (is_causal) { window_size_right = 0; }
//     // There's a case where is_causal=false, window_size=(-1, 0). Then set_params_bprop will set params.is_causal=true.
//     // If we don't have is_causal here matching params.is_causal, we might get the wrong kBlockM (and cause IMA).
//     is_causal = window_size_left < 0 && window_size_right == 0;

//     int const arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
//     int const head_size_rounded = round_up_headdim(head_size);
//     // Very important that these match the kernel configs
//     bool const is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
//     int const kBlockM_sm90 = head_size_rounded <= 64 ? (is_causal && softcap > 0.0 ? 96 : 128)
//         : (head_size_rounded <= 96 ? 64
//            : (head_size_rounded <= 128 ? (is_causal || is_local || softcap > 0.0 ? 64 : 80)
//               : 64));
//     int const kBlockM_sm80 = head_size_rounded <= 64 ? 128 : 64;
//     int const kBlockM_sm86 = head_size_rounded <= 192 ? 64 : 32;
//     int const kBlockM = arch >= 90 ? kBlockM_sm90 : (arch == 86 || arch == 89 ? kBlockM_sm86 : kBlockM_sm80);
//     int const kBlockN_sm90 = head_size_rounded <= 128
//         ? 128
//         : (head_size_rounded <= 192 ? 96 : 80);
//     int const kBlockN_sm80 = head_size_rounded <= 128
//         ? 128
//         : (head_size_rounded <= 192 ? 80 : 64);
//     int const kBlockN_sm86 = head_size_rounded <= 64 ? 128
//         : (head_size_rounded <= 96 ? 128
//            : (head_size_rounded <= 128 ? 96
//               : (head_size_rounded <= 192 ? 64 : 64)));
//     int const kBlockN = arch >= 90 ? kBlockN_sm90 : (arch == 86 || arch == 89 ? kBlockN_sm86 : kBlockN_sm80);
//     auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
//     int const seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
//     int const seqlen_k_rounded = round_multiple(seqlen_k, kBlockN);
//     int const total_q_padded_rounded = round_multiple(total_q + batch_size * kBlockM, kBlockM);
//     int const total_k_padded_rounded = round_multiple(total_k + batch_size * kBlockN, kBlockN);
//     int const total_q_rounded = round_multiple(total_q + kBlockM, kBlockM);
//     int const total_k_rounded = round_multiple(total_k + kBlockN, kBlockN);

//     if (!(is_varlen_q || is_flex_q)) {
//         // common case
//         CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
//         CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
//         CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);
//     } else {
//         // varlen or flex case
//         CHECK_SHAPE(q, total_q, num_heads, head_size);
//         CHECK_SHAPE(out, total_q, num_heads, head_size);
//         CHECK_SHAPE(dout, total_q, num_heads, head_size);
//         if (is_flex_q) {
//             CHECK_SHAPE(q_ranges, batch_size, 2);
//         }
//         else {
//             CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
//         }
//     }
//     if (!(is_varlen_k || is_flex_k)) {
//         // common case
//         CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
//         CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
//     } else {
//         // varlen or flex case
//         CHECK_SHAPE(k, total_k, num_heads_k, head_size);
//         CHECK_SHAPE(v, total_k, num_heads_k, head_size);
//         if (is_flex_k) {
//             CHECK_SHAPE(k_ranges, batch_size, 2);
//         }
//         else {
//             CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
//         }
//     }
//     if (has_attn_type_map) {
//         CHECK_SHAPE(attn_type_map, batch_size);
//     }

//     if (seqused_q_.has_value()){
//         auto seqused_q = seqused_q_.value();
//         TORCH_CHECK(seqused_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
//         CHECK_DEVICE(seqused_q); CHECK_CONTIGUOUS(seqused_q);
//         CHECK_SHAPE(seqused_q, batch_size);
//     }
//     if (seqused_k_.has_value()){
//         auto seqused_k = seqused_k_.value();
//         TORCH_CHECK(seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
//         CHECK_DEVICE(seqused_k); CHECK_CONTIGUOUS(seqused_k);
//         CHECK_SHAPE(seqused_k, batch_size);
//     }

//     at::Tensor dq, dk, dv;
//     if (dq_.has_value()) {
//         dq = dq_.value();
//         TORCH_CHECK(dq.dtype() == q_type, "dq must have the same dtype as q");
//         CHECK_DEVICE(dq);
//         TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
//         if (!(is_varlen_q || is_flex_q)) {
//             // common case
//             CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
//         } else {
//             // varlen or flex case
//             CHECK_SHAPE(dq, total_q, num_heads, head_size);
//         }
//     } else {
//         dq = torch::empty_like(q);
//     }
//     if (dk_.has_value()) {
//         dk = dk_.value();
//         TORCH_CHECK(dk.dtype() == q_type, "dk must have the same dtype as q");
//         CHECK_DEVICE(dk);
//         TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
//         if (!(is_varlen_k || is_flex_k)) {
//             // common case
//             CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
//         } else {
//             // varlen or flex case
//             CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
//         }
//     } else {
//         dk = torch::empty_like(k);
//     }
//     if (dv_.has_value()) {
//         dv = dv_.value();
//         TORCH_CHECK(dv.dtype() == q_type, "dv must have the same dtype as q");
//         CHECK_DEVICE(dv);
//         TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
//         if (!(is_varlen_k || is_flex_k)) {
//             // common case
//             CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
//         } else {
//             // varlen or flex case
//             CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
//         }
//     } else {
//         dv = torch::empty_like(v);
//     }

//     // Otherwise the kernel will be launched from cuda:0 device
//     // Cast to char to avoid compiler warning about narrowing
//     at::cuda::CUDAGuard device_guard{(char)q.get_device()};

//     auto opts = q.options();
//     // Need softmax_d to have total_q_padded_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
//     at::Tensor softmax_d, softmax_lse_log2;
//     if (!is_varlen || is_flex_q) {
//         // common case
//         // Need softmax_d to have seqlen_q_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
//         softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
//         softmax_lse_log2 = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
//     } else {
//         // varlen or flex case
//         softmax_d = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
//         softmax_lse_log2 = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
//     }
//     at::Tensor dq_accum, dk_accum, dv_accum;
//     if (!is_varlen) {
//         // common case
//         dq_accum = torch::empty({batch_size, num_heads, seqlen_q_rounded, head_size_rounded}, opts.dtype(at::kFloat));
//     } else {
//         // varlen or flex case
//         dq_accum = torch::zeros_like(dq, opts.dtype(at::kFloat));
//     }
//     // REVIEW(littsk): 扩展到mha的flex setting
//     if (!is_varlen) {
//         // common case
//         dk_accum = torch::zeros({batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}, opts.dtype(at::kFloat));
//         dv_accum = torch::zeros({batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}, opts.dtype(at::kFloat));
//     } else {
//         // varlen or flex case
//         dk_accum = torch::zeros_like(dk, opts.dtype(at::kFloat));
//         dv_accum = torch::zeros_like(dv, opts.dtype(at::kFloat));
//     }

//     Flash_bwd_params params;
//     set_params_dgrad(params,
//                      batch_size,
//                      seqlen_q, seqlen_k,
//                      seqlen_q_rounded, seqlen_k_rounded,
//                      num_heads, num_heads_k,
//                      head_size, head_size_rounded,
//                      q, k, v, out,
//                      dout, dq, dk, dv,
//                      !is_flex_q ? nullptr : q_ranges.data_ptr(),
//                      !is_flex_k ? nullptr : k_ranges.data_ptr(),
//                      !is_varlen_q ? nullptr : cu_seqlens_q.data_ptr(),
//                      !is_varlen_k ? nullptr : cu_seqlens_k.data_ptr(),
//                      seqused_q_.has_value() ? seqused_q_.value().data_ptr() : nullptr,
//                      seqused_k_.has_value() ? seqused_k_.value().data_ptr() : nullptr,
//                      has_attn_type_map ? attn_type_map.data_ptr() : nullptr,
//                      dq_accum.data_ptr(),
//                      dk_accum.data_ptr(),
//                      dv_accum.data_ptr(),
//                      softmax_lse.data_ptr(),
//                      softmax_d.data_ptr(),
//                      /*p_dropout=*/0.f,
//                      softmax_scale,
//                      window_size_left,
//                      window_size_right,
//                      softcap,
//                      deterministic,
//                      sm_margin);
//     params.total_q = total_q;
//     params.total_k = total_k;
//     params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();
//     params.dv = head_size;  // We don't support hdim_v being different from hdim_qk for now

//     // auto tile_count_semaphore = (params.is_causal || params.is_local) ? torch::zeros({1}, opts.dtype(torch::kInt32)) : torch::empty({1}, opts.dtype(torch::kInt32));
//     // params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
//     // Will be zero'ed out in the backward preprocess kernel
//     at::Tensor dq_semaphore = torch::empty({(seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, opts.dtype(torch::kInt32));
//     params.dq_semaphore = dq_semaphore.data_ptr<int>();
//     if (num_heads_k != num_heads && params.deterministic) {
//         // TODO: do we need to zero them out?
//         at::Tensor dk_semaphore = torch::empty({(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, opts.dtype(torch::kInt32));
//         at::Tensor dv_semaphore = torch::empty({(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, opts.dtype(torch::kInt32));
//         params.dk_semaphore = dk_semaphore.data_ptr<int>();
//         params.dv_semaphore = dv_semaphore.data_ptr<int>();
//     }

//     #ifdef FLASHATTENTION_DISABLE_LOCAL
//     TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
//     #endif
//     #ifdef FLASHATTENTION_DISABLE_SOFTCAP
//     TORCH_CHECK(params.softcap == 0.0, "This flash attention build does not support tanh softcapping.");
//     #endif

//     // REVIEW(littsk): 怎么设置q_ranges没覆盖的地方到zeros
//     if (total_q > 0 && total_k > 0 && num_heads_k > 0) {
//         auto stream = at::cuda::getCurrentCUDAStream().stream();
//         run_mha_bwd(params, stream);
//     } else if (total_k > 0 && num_heads_k > 0) {
//         // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
//         dk.zero_();
//         dv.zero_();
//         softmax_d.zero_();
//     } else if (total_q > 0 && num_heads_k > 0) {
//         dq.zero_();
//         softmax_d.zero_();
//     }

//     return { dq, dk, dv, softmax_d, softmax_lse_log2, dq_accum, dk_accum, dv_accum };
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlexibleFlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    // m.def("bwd", &mha_bwd, "Backward pass");
}
