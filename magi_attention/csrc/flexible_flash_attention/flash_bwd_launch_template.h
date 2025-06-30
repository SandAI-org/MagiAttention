/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/device_kernel.h"  // For device_kernel
#include "cutlass/kernel_launch.h"  // For kernel_launch
#include "cutlass/cluster_launch.hpp"  // For ClusterLauncher

#include "../common/static_switch.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "tile_size.h"
#include "tile_scheduler.hpp"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_bwd.hpp"
#include "flash_bwd_kernel_sm90.h"

using namespace cute;

template <int Arch, int kHeadDim, int kBlockM, int kBlockN, bool Has_softcap,
          typename Element, typename ElementOut, bool Deterministic, 
          int Stages=2, int Stages_dO=2, int Stages_dS=2,
          bool SdP_swapAB=true, bool dKV_swapAB=false, bool dQ_swapAB=false,
          int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
          bool V_in_regs=false, bool RangeMerge=false>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr bool Is_causal = false;
    constexpr bool Is_local = false;
    constexpr bool Varlen = true;
    int batch_q = 1;
    int batch_k = 1;

    using ElementAccum = float;
    using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;

    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using PreprocessKernel = flash::FlashAttnBwdPreprocess<TileShape_MK, Element, ElementAccum, ArchTag, /*Clear_dQ=*/false, /*Clear_dK=*/false, /*Clear_dV=*/false>;
    typename PreprocessKernel::Arguments preprocess_args {
        static_cast<Element const*>(params.o_ptr),
        {params.total_q, params.d, params.h_qo},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride},  // stride_O
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride},  // stride_dO
        static_cast<float*>(params.dsoftmax_sum),
        {params.max_seqlen_q_rounded, params.h_qo, params.b},  // shape_dPsum
        {_1{}, params.max_seqlen_q_rounded, params.h_qo * params.max_seqlen_q_rounded},  // stride_dPsum
        {params.total_q, params.h_qo},  // shape_LSE
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, params.total_q},  // stride_LSE
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {_1{}, params.max_seqlen_q_rounded, params.h_qo * params.max_seqlen_q_rounded},  // stride_LSE_log2
        params.q_ranges,
        params.k_ranges
    };
    typename PreprocessKernel::Params preprocess_params = PreprocessKernel::to_underlying_arguments(preprocess_args);
    int num_m_block = cute::ceil_div(params.max_seqlen_q, kBlockM);
    dim3 grid_m(params.b, num_m_block, params.h_qo);
    cutlass::kernel_launch<PreprocessKernel>(grid_m, PreprocessKernel::MaxThreadsPerBlock, PreprocessKernel::SharedStorageSize, stream, preprocess_params, false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;  // Currently doesn't not support cluster
    // Stages_dS_or_QSm80 is Stages_dS if Sm90 and Stages if Sm80
    using CollectiveMainloop = flash::CollectiveMainloopBwdSm90<Stages, Stages_dO, Stages_dS, ClusterShape, TileShape_MNK, 
            Element, ElementAccum, cutlass::arch::Sm90, 
            Has_softcap, Deterministic, SdP_swapAB, dKV_swapAB, dQ_swapAB, 
            NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>;
    using CollectiveEpilogue = flash::CollectiveEpilogueBwd<TileShape_MNK, ElementAccum, ArchTag, CollectiveMainloop::NumMmaThreads, dKV_swapAB, NumMmaWarpGroups * (Arch >= 90 ? 1 : cutlass::NumWarpsPerWarpGroup) / AtomLayoutNdKV>;
    using Scheduler = flash::VarlenDynamicPersistentTileScheduler<kBlockN, CollectiveMainloop::NumMmaThreads, CollectiveMainloop::NumProducerThreads, false /*Split*/, false /*PackGQA*/, Arch >= 90 /*WarpSpecialized*/>;
    using AttnKernel = flash::enable_sm90_or_later<flash::FlashAttnBwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler, RangeMerge>>;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {params.total_q, params.d, params.h_qo},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride},  // stride_Q
        static_cast<Element const*>(params.k_ptr),
        {params.total_k, params.d, params.h_kv},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride},  // stride_K
        static_cast<Element const*>(params.v_ptr),
        {params.v_row_stride, _1{}, params.v_head_stride},  // stride_V
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride},  // stride_dO
        static_cast<ElementAccum*>(params.dq_ptr),
        {params.total_q, params.d, params.h_qo},  // shape_dQ
        {params.dq_row_stride, _1{}, params.dq_head_stride}, // stride_dQ
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {params.max_seqlen_q_rounded, params.h_qo, params.b},   // shape_LSE
        {_1{}, params.max_seqlen_q_rounded, params.h_qo * params.max_seqlen_q_rounded},  // stride_LSE_log2
        static_cast<float*>(params.dsoftmax_sum),
        {_1{}, params.max_seqlen_q_rounded, params.h_qo * params.max_seqlen_q_rounded},  // stride_dPsum
        params.scale_softmax,
        params.softcap,
        params.q_ranges, params.k_ranges,
        params.attn_type_map
    };
    // The case work with GQA is ugly but idk how to fix it.
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<typename CollectiveEpilogue::Element*>(params.dk_ptr),
        {params.total_k, params.d, params.h_kv},  // shape_dK
        {params.dk_row_stride, _1{}, params.dk_head_stride},  // stride_dK
        static_cast<typename CollectiveEpilogue::Element*>(params.dv_ptr),
        {params.dv_row_stride, _1{}, params.dv_head_stride},  // stride_dV
        params.h_qo,
        params.q_ranges,
        params.k_ranges
    };

    int num_blocks_n = cutlass::ceil_div(params.max_seqlen_k, get<1>(TileShape_MNK{}));
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_n, params.h_qo, params.merge_batch_size, 1 /*num_splits*/,
        params.h_qo / params.h_kv,
        params.max_seqlen_k,
        params.max_seqlen_q, params.d, params.d, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_k, params.k_ranges, params.seqused_k, nullptr,
        params.merge_k_ranges,
        params.bwd_kq_map
    };

    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    // int smem_size_q = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_q));
    // int smem_size_do = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_do));
    // int smem_size_ds = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_ds));
    // int smem_size_dqacc = [&] {
    //     if constexpr (Arch >= 90) {
    //         return sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_dqacc));
    //     } else {
    //         return 0;
    //     }
    // }();
    // int smem_size_k = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v));
    // int smem_size_lse = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_lse));
    // int smem_size_dpsum = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_dpsum));
    // printf("smem_size = %d, q = %d, k = %d, v = %d, do = %d, ds = %d, dqacc = %d, lse = %d, dpsum = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v, smem_size_do, smem_size_ds, smem_size_dqacc, smem_size_lse, smem_size_dpsum);
    if constexpr (size(ClusterShape{}) > 1) {
        void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLauncher::launch(
            grid_dims, cluster_dims, block_dims, smem_size, stream, kernel, kernel_params, false /*launch_with_pdl*/);
    } else {
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<AttnKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/);
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<int Arch, typename T, typename T_out, int kHeadDim, bool Has_softcap>
void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream) {
    static_assert(sizeof(T) == 2, "Only 16bit computation are supported");
    static constexpr int kBlockM = std::get<0>(tile_size_bwd_sm90(kHeadDim, sizeof(T) /*element_size*/, Has_softcap));
    static constexpr int kBlockN = std::get<1>(tile_size_bwd_sm90(kHeadDim, sizeof(T) /*element_size*/, Has_softcap));
    
    // TODO: Add a specific tuning function for different kHeadDim
    static constexpr int Stages = 2;
    static constexpr int Stages_dO = kHeadDim <= 128 ? 2 : 1;
    static constexpr int Stages_dS = kHeadDim <= 128 ? 2 : 1;

    static constexpr bool SdP_swapAB = kHeadDim <= 128 ? true : false;
    static constexpr bool dKV_swapAB = kHeadDim <= 128 ? false : true;
    static constexpr bool dQ_swapAB = kHeadDim <= 64 ? false : true;

    static constexpr int NumMmaWarpGroups = kHeadDim == 192 ? 3 : 2;
    static constexpr int AtomLayoutMSdP = 1;
    static constexpr int AtomLayoutNdKV = kHeadDim <= 128 ? 2 : 1;
    static constexpr int AtomLayoutMdQ = kHeadDim <= 64 ? 2 : 1;
    static constexpr bool V_in_regs = false;
    
    BOOL_SWITCH(params.deterministic, Deterministic, [&] {
        BOOL_SWITCH(params.merge_k_ranges != nullptr, RangeMerge, [&] {
            run_flash_bwd<
                /*Arch=*/Arch,
                /*kHeadDim=*/kHeadDim,
                /*kBlockM=*/kBlockM,
                /*kBlockN=*/kBlockN,
                /*Has_softcap=*/Has_softcap,
                /*Element=*/T,
                /*ElementOut=*/T_out,
                /*Deterministic=*/Deterministic,
                /*Stages=*/Stages,
                /*Stages_dO=*/Stages_dO,
                /*Stages_dS=*/Stages_dS,
                /*SdP_swapAB=*/SdP_swapAB,
                /*dKV_swapAB=*/dKV_swapAB,
                /*dQ_swapAB=*/dQ_swapAB,
                /*NumMmaWarpGroups=*/NumMmaWarpGroups,
                /*AtomLayoutMSdP=*/AtomLayoutMSdP,
                /*AtomLayoutNdKV=*/AtomLayoutNdKV,
                /*AtomLayoutMdQ=*/AtomLayoutMdQ,
                /*V_in_regs=*/V_in_regs,
                /*RangeMerge=*/RangeMerge
            >(params, stream);
        });
    });
}
