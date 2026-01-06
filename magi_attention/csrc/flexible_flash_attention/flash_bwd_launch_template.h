/**********************************************************************************
 * Copyright (c) 2025-2026 SandAI. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *********************************************************************************/

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <stdexcept>

#include <cute/tensor.hpp>

#include <cutlass/cluster_launch.hpp> // For ClusterLauncher
#include <cutlass/device_kernel.h> // For device_kernel
#include <cutlass/kernel_launch.h> // For kernel_launch

#include "epilogue_bwd.hpp"
#include "flash.h"
#include "flash_bwd_kernel_sm90.h"
#include "flash_bwd_preprocess_kernel.h"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "static_switch.h"
#include "tile_scheduler.hpp"
#include "tile_size.h"
#include "utils/profile_utils.h"

using namespace cute;

template <typename TileShape_MK, typename Element, typename ElementAccum, typename ArchTag, bool Has_sink, flash::SinkLayout kSinkLayout, bool ProfileMode = false>
void run_flash_bwd_pre_process(Flash_bwd_params& params, cudaStream_t stream) {
  if constexpr (ProfileMode)
    MagiEvents::start("bwd_preprocess");

  using PreprocessKernel = flash::FlashAttnBwdPreprocess<
      /*TileShape_MK_=*/TileShape_MK,
      /*Element=*/Element,
      /*ElementAccum=*/ElementAccum,
      /*ArchTag_=*/ArchTag,
      /*Clear_dQ=*/false,
      /*Clear_dK=*/false,
      /*Clear_dV=*/false,
      /*Has_sink=*/Has_sink,
      /*kSinkLayout=*/kSinkLayout>;

  typename PreprocessKernel::Arguments preprocess_args{
      // O
      static_cast<Element const*>(params.o_ptr),
      {params.total_q, params.d, params.h_qo}, // shape_O: [sq, hd, nhq]
      {params.o_row_stride, _1{}, params.o_head_stride}, // stride_O: [nhq*hd, 1, hd]
      // dO
      static_cast<Element const*>(params.do_ptr),
      {params.do_row_stride, _1{}, params.do_head_stride}, // stride_dO: [nhq*hd, 1, hd]
      // dPsum
      static_cast<float*>(params.dsoftmax_sum),
      {_4{}, params.total_q_rounded, params.h_qo}, // shape_dPsum: [4, sq_rounded, nhq]
      {_1{}, _4{}, params.total_q_rounded * 4}, // stride_dPsum: [1, 4, sq_rounded*4]
      // LSE
      static_cast<float*>(params.softmax_lse_ptr),
      {params.total_q, params.h_qo}, // shape_LSE: [sq, nhq]
      {params.h_qo, _1{}}, // stride_LSE: [nhq, 1]
      // LSE_log2
      static_cast<float*>(params.softmax_lse_log2_ptr),
      {_1{}, _4{}, params.total_q_rounded * 4}, // stride_LSE_log2: [1, 4, sq_rounded*4]
      // sink
      static_cast<float*>(params.sink_ptr),
      {kSinkLayout == flash::SinkLayout::SSH ? params.total_q : 1, params.total_sink, params.h_qo}, // shape_sink: [1, s_sink, nhq] or [sq, s_sink, nhq]
      {params.total_sink * params.h_qo, params.h_qo, _1{}}, // stride_sink: [s_sink*nhq, nhq, 1]
      // dsink
      static_cast<float*>(params.dsink_ptr),
      static_cast<float*>(params.dsink_reduce_buf_ptr),
      static_cast<unsigned int*>(params.dsink_reduce_cnt_ptr), // shape_dsink_reduce_cnt: [nhq,]
      {params.num_m_block, params.total_sink, params.h_qo}, // shape_dsink_reduce_buf: [num_m_block, s_sink, nhq]
      {params.total_sink * params.h_qo, params.h_qo, _1{}}, // stride_dsink_reduce_buf: [nhq, 1]
      // meta
      params.num_m_block,
      params.total_q,
      params.total_sink};

  typename PreprocessKernel::Params preprocess_params = PreprocessKernel::to_underlying_arguments(preprocess_args);
  dim3 grid_m(1, params.num_m_block, params.h_qo);

  cutlass::kernel_launch<PreprocessKernel>(
      grid_m, PreprocessKernel::MaxThreadsPerBlock, PreprocessKernel::SharedStorageSize, stream, preprocess_params, /*launch_with_pdl=*/false);

  CHECK_CUDA_KERNEL_LAUNCH();

  if constexpr (ProfileMode)
    MagiEvents::stop("bwd_preprocess");
}

template <
    int Arch,
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    bool Has_softcap,
    typename Element,
    typename ElementDkv,
    bool Deterministic,
    bool SwapBwdQKLoop,
    int Stages = 2,
    int Stages_dO = 2,
    int Stages_dS = 2,
    bool SdP_swapAB = true,
    bool dKV_swapAB = false,
    bool dQ_swapAB = false,
    int NumMmaWarpGroups = 2,
    int AtomLayoutMSdP = 1,
    int AtomLayoutNdKV = 2,
    int AtomLayoutMdQ = 1,
    bool V_in_regs = false,
    bool RangeMerge = false,
    bool DisableBwdDkvAtomicReduction = false,
    bool ProfileMode = false>
void run_flash_bwd(Flash_bwd_params& params, cudaStream_t stream) {
  using ElementAccum = float;
  using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;
  using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;

  // Launch the pre-processing kernel of the ffa backward pass
  BOOL_SWITCH(params.has_sink(), Has_sink, [&] {
    switch (params.sink_layout) {
      case flash::SinkLayout::SH:
        run_flash_bwd_pre_process<TileShape_MK, Element, ElementAccum, ArchTag, Has_sink, flash::SinkLayout::SH, ProfileMode>(params, stream);
        break;
      case flash::SinkLayout::SSH:
        run_flash_bwd_pre_process<TileShape_MK, Element, ElementAccum, ArchTag, Has_sink, flash::SinkLayout::SSH, ProfileMode>(params, stream);
        break;
      default:
        throw std::runtime_error("Unsupported sink layout");
    }
  });

  // Run the main kernel of the ffa backward pass
  if constexpr (ProfileMode)
    MagiEvents::start("bwd_run");

  using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using ClusterShape = cute::Shape<_1, Int<1>, _1>; // Currently doesn't not support cluster

  // Get Mainloop, TileScheduler, Epilogue and AttnKernel
  using CollectiveMainloop = flash::CollectiveMainloopBwdSm90<
      Stages,
      Stages_dO,
      Stages_dS,
      ClusterShape,
      TileShape_MNK,
      Element,
      ElementAccum,
      cutlass::arch::Sm90,
      Has_softcap,
      Deterministic,
      SwapBwdQKLoop,
      SdP_swapAB,
      dKV_swapAB,
      dQ_swapAB,
      NumMmaWarpGroups,
      AtomLayoutMSdP,
      AtomLayoutNdKV,
      AtomLayoutMdQ,
      V_in_regs>;
  using Scheduler = flash::DynamicPersistentTileScheduler<
      SwapBwdQKLoop ? kBlockM : kBlockN,
      CollectiveMainloop::NumMmaThreads,
      CollectiveMainloop::NumProducerThreads,
      /*WarpSpecialized=*/Arch >= 90,
      Deterministic>;
  using CollectiveEpilogue = flash::CollectiveEpilogueBwd<
      TileShape_MNK,
      ElementDkv,
      ElementAccum,
      ArchTag,
      typename Scheduler::BlockCoordType,
      CollectiveMainloop::NumMmaThreads,
      dKV_swapAB,
      NumMmaWarpGroups*(Arch >= 90 ? 1 : cutlass::NumWarpsPerWarpGroup) / AtomLayoutNdKV,
      DisableBwdDkvAtomicReduction,
      Deterministic,
      SwapBwdQKLoop>;
  using AttnKernel = flash::enable_sm90_or_later<flash::FlashAttnBwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler, RangeMerge>>;

  typename CollectiveMainloop::Arguments mainloop_args{
      static_cast<Element const*>(params.q_ptr),
      {params.total_q, params.d, params.h_qo}, // shape_Q
      {params.q_row_stride, _1{}, params.q_head_stride}, // stride_Q
      static_cast<Element const*>(params.k_ptr),
      {params.total_k, params.d, params.h_kv}, // shape_K
      {params.k_row_stride, _1{}, params.k_head_stride}, // stride_K
      static_cast<Element const*>(params.v_ptr),
      {params.v_row_stride, _1{}, params.v_head_stride}, // stride_V
      static_cast<Element const*>(params.do_ptr),
      {params.do_row_stride, _1{}, params.do_head_stride}, // stride_dO
      // k for outer-loop and q for inner-loop
      static_cast<ElementAccum*>(params.dq_ptr),
      {params.total_q, params.d, params.h_qo}, // shape_dQ
      {params.dq_row_stride, _1{}, params.dq_head_stride}, // stride_dQ
      // q for outer-loop and k for inner-loop
      static_cast<ElementAccum*>(params.dk_ptr),
      {params.total_k, params.d, params.h_kv}, // shape_dK
      {params.dk_row_stride, _1{}, params.dk_head_stride}, // stride_dK
      static_cast<ElementAccum*>(params.dv_ptr),
      {params.total_k, params.d, params.h_kv}, // shape_dV
      {params.dv_row_stride, _1{}, params.dv_head_stride}, // stride_dV
      static_cast<float*>(params.softmax_lse_log2_ptr),
      {_4{}, params.total_q_rounded, params.h_qo}, // shape_LSE
      {_1{}, _4{}, params.total_q_rounded * 4}, // stride_LSE_log2
      static_cast<float*>(params.dsoftmax_sum),
      {_1{}, _4{}, params.total_q_rounded * 4}, // stride_dPsum
      params.scale_softmax,
      params.softcap,
      params.q_ranges,
      params.k_ranges,
      params.dq_determin_conflict_state,
      params.dq_determin_range_locks,
      params.attn_type_map};

  typename CollectiveEpilogue::Arguments epilogue_args{
      // q for outer-loop and k for inner-loop
      static_cast<typename CollectiveEpilogue::Element*>(params.dq_ptr),
      {params.total_q, params.d, params.h_qo}, // shape_dQ
      {params.dq_row_stride, _1{}, params.dq_head_stride}, // stride_dQ
      // k for outer-loop and q for inner-loop
      static_cast<typename CollectiveEpilogue::Element*>(params.dk_ptr),
      {params.total_k, params.d, params.h_kv}, // shape_dK
      {params.dk_row_stride, _1{}, params.dk_head_stride}, // stride_dK
      static_cast<typename CollectiveEpilogue::Element*>(params.dv_ptr),
      {params.total_k, params.d, params.h_kv}, // shape_dV
      {params.dv_row_stride, _1{}, params.dv_head_stride}, // stride_dV
      params.h_qo,
      params.h_kv,
      params.q_ranges,
      params.k_ranges,
      params.determin_range_locks,
  };

  typename flash::TileSchedulerArguments scheduler_args{/*num_heads=*/params.h_qo,
                                                        /*num_batches=*/params.merge_batch_size,
                                                        /*tile_count_semaphore=*/params.tile_count_semaphore,
                                                        /*ranges=*/params.k_ranges,
                                                        /*merge_ranges=*/params.merge_k_ranges,
                                                        /*range_map=*/params.bwd_kq_map,
                                                        /*determin_conflict_state=*/params.determin_conflict_state,
                                                        /*bwd_unique_count=*/params.bwd_unique_count};

  int device;
  cudaGetDevice(&device);
  typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args});

  dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
  dim3 block_dims = AttnKernel::get_block_shape();
  int smem_size = AttnKernel::SharedStorageSize;

  /* DEBUG */
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
  // printf("smem_size = %d, q = %d, k = %d, v = %d, do = %d, ds = %d, dqacc = %d, lse = %d, dpsum = %d\n", smem_size,
  // smem_size_q, smem_size_k, smem_size_v, smem_size_do, smem_size_ds, smem_size_dqacc, smem_size_lse,
  // smem_size_dpsum);

  if constexpr (size(ClusterShape{}) > 1) {
    void const* kernel = (void const*)cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) { // exceed static shared memory size limit (48KB on Hopper)
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLauncher::launch(grid_dims, cluster_dims, block_dims, smem_size, stream, kernel, kernel_params, /*launch_with_pdl=*/false);
  } else {
    if (smem_size >= 48 * 1024) { // exceed static shared memory size limit (48KB on Hopper)
      CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<AttnKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, /*launch_with_pdl=*/false);
  }
  CHECK_CUDA_KERNEL_LAUNCH();

  if constexpr (ProfileMode)
    MagiEvents::stop("bwd_run");
}

template <
    int Arch,
    typename T,
    typename TDkv,
    int kHeadDim,
    bool Has_softcap,
    bool DisableBwdDkvAtomicReduction,
    bool Deterministic,
    bool SwapBwdQKLoop,
    bool ProfileMode>
void run_mha_bwd_(Flash_bwd_params& params, cudaStream_t stream) {
  static_assert(sizeof(T) == 2, "Only 16bit computation are supported");
  static constexpr int kBlockM = std::get<0>(tile_size_bwd_sm90(kHeadDim, /*element_size=*/sizeof(T), Has_softcap));
  static constexpr int kBlockN = std::get<1>(tile_size_bwd_sm90(kHeadDim, /*element_size=*/sizeof(T), Has_softcap));

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

  BOOL_SWITCH(params.merge_k_ranges != nullptr, RangeMerge, [&] {
    run_flash_bwd<
        /*Arch=*/Arch,
        /*kHeadDim=*/kHeadDim,
        /*kBlockM=*/kBlockM,
        /*kBlockN=*/kBlockN,
        /*Has_softcap=*/Has_softcap,
        /*Element=*/T,
        /*ElementDkv=*/TDkv,
        /*Deterministic=*/Deterministic,
        /*SwapBwdQKLoop=*/SwapBwdQKLoop,
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
        /*RangeMerge=*/RangeMerge,
        /*DisableBwdDkvAtomicReduction=*/DisableBwdDkvAtomicReduction,
        /*ProfileMode=*/ProfileMode>(params, stream);
  });
}
