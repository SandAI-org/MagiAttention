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

#include "bwd_tile_scheduler.hpp"
#include "epilogue_bwd.hpp"
#include "extensions/profile_utils.h"
#include "flash.h"
#include "flash_bwd_kernel_sm90.h"
#include "flash_bwd_preprocess_kernel.h"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "static_switch.h"
#include "tile_size.h"

using namespace cute;

template <typename TileShape_MK, typename Element, typename ArchTag, bool Has_sink, flash::SinkLayout kSinkLayout, bool ProfileMode>
void run_flash_bwd_pre_process(Flash_bwd_params& params, cudaStream_t stream) {
  if constexpr (ProfileMode)
    MagiEvents::start("bwd_preprocess");

  using PreprocessKernel = flash::FlashAttnBwdPreprocess<
      /*TileShape_MK_=*/TileShape_MK,
      /*Element=*/Element,
      /*ArchTag_=*/ArchTag,
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
    typename ElementDq,
    typename ElementDkv,
    bool Deterministic,
    bool BwdInnerLoopK,
    bool PackGQA,
    bool CatGQA,
    int PackGQAFactor,
    int Stages,
    int Stages_dO,
    int Stages_dS,
    bool SdP_swapAB,
    bool dKV_swapAB,
    bool dQ_swapAB,
    int NumConsumerWarpGroups,
    int AtomLayoutMSdP,
    int AtomLayoutNdKV,
    int AtomLayoutMdQ,
    bool RangeMerge,
    bool BlockSparse,
    bool IndexSparse,
    bool InnerDirMaxToMin,
    int MaskMode,
    bool InnerStoreInProducer,
    int BwdProducerRegs,
    int BwdConsumerRegs,
    int InnerStoreMode,
    bool OuterStoreNeedReduction,
    int OuterStoreMode,
    int Stages_V,
    int Tma1dSmemRowPad,
    int SparseKBlockSize,
    int InnerLoadMode,
    bool UnionDkvSmem,
    int InnerStoreStages,
    bool ProfileMode,
    bool PerfDebugSkipVLoad_,
    bool PerfDebugSkipDvStore_,
    bool PerfDebugSkipDkStore_,
    bool PerfDebugSkipDvMma_>
void run_flash_bwd(Flash_bwd_params& params, cudaStream_t stream) {
  using ElementAccum = float;
  using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;
  using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;

  // Launch the pre-processing kernel of the ffa backward pass
  BOOL_SWITCH(params.has_sink(), Has_sink, [&] {
    switch (params.sink_layout) {
      case flash::SinkLayout::SH:
        run_flash_bwd_pre_process<TileShape_MK, Element, ArchTag, Has_sink, flash::SinkLayout::SH, ProfileMode>(params, stream);
        break;
      case flash::SinkLayout::SSH:
        run_flash_bwd_pre_process<TileShape_MK, Element, ArchTag, Has_sink, flash::SinkLayout::SSH, ProfileMode>(params, stream);
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
      BwdInnerLoopK,
      SdP_swapAB,
      dKV_swapAB,
      dQ_swapAB,
      PackGQA,
      CatGQA,
      RangeMerge,
      BlockSparse,
      IndexSparse,
      InnerDirMaxToMin,
      MaskMode,
      InnerStoreInProducer,
      InnerStoreMode,
      PackGQAFactor,
      NumConsumerWarpGroups,
      AtomLayoutMSdP,
      AtomLayoutNdKV,
      AtomLayoutMdQ,
      Stages_V,
      Tma1dSmemRowPad,
      SparseKBlockSize,
      InnerLoadMode,
      UnionDkvSmem,
      InnerStoreStages,
      PerfDebugSkipVLoad_,
      PerfDebugSkipDvStore_,
      PerfDebugSkipDkStore_,
      PerfDebugSkipDvMma_>;

  using Scheduler = flash::DynamicPersistentTileSchedulerBwd<
      BwdInnerLoopK ? kBlockM : kBlockN,
      CollectiveMainloop::NumConsumerThreads,
      CollectiveMainloop::NumProducerThreads,
      /*WarpSpecialized=*/Arch >= 90,
      /*PackGQA=*/PackGQA,
      /*CatGQA=*/CatGQA,
      /*BwdInnerLoopK*/ BwdInnerLoopK,
      /*Deterministic=*/Deterministic>;

  using CollectiveEpilogue = flash::CollectiveEpilogueBwd<
      TileShape_MNK,
      ElementDq,
      ElementDkv,
      ElementAccum,
      ArchTag,
      typename Scheduler::BlockCoordType,
      dQ_swapAB,
      dKV_swapAB,
      NumConsumerWarpGroups,
      AtomLayoutMdQ,
      AtomLayoutNdKV,
      OuterStoreNeedReduction,
      OuterStoreMode,
      Deterministic,
      BwdInnerLoopK,
      /*PackGQA=*/PackGQA,
      /*CatGQA=*/CatGQA,
      /*PackGQAFactor=*/PackGQAFactor,
      /*IndexSparse=*/IndexSparse,
      /*SparseKBlockSize=*/SparseKBlockSize>;
  using AttnKernel = flash::enable_sm90_or_later<
      flash::FlashAttnBwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler, RangeMerge, InnerDirMaxToMin, BwdProducerRegs, BwdConsumerRegs>>;

  typename CollectiveMainloop::Arguments mainloop_args{
      static_cast<Element const*>(params.q_ptr),
      static_cast<Element const*>(params.do_ptr),
      static_cast<ElementAccum*>(params.dq_ptr),
      {params.total_q, Int<kHeadDim>{}, params.h_qo}, // shape_QdOdQ
      {params.q_row_stride, _1{}, Int<kHeadDim>{}}, // stride_Q
      {params.do_row_stride, _1{}, Int<kHeadDim>{}}, // stride_dO
      {params.dq_row_stride, _1{}, Int<kHeadDim>{}}, // stride_dQ
      static_cast<Element const*>(params.k_ptr),
      static_cast<Element const*>(params.v_ptr),
      static_cast<ElementAccum*>(params.dk_ptr),
      static_cast<ElementAccum*>(params.dv_ptr),
      {params.total_k, Int<kHeadDim>{}, params.h_kv}, // shape_KVdKdV
      {params.total_k, Int<kHeadDim>{}, params.h_kv}, // shape_dKdV
      {params.k_row_stride, _1{}, Int<kHeadDim>{}}, // stride_K
      {params.v_row_stride, _1{}, Int<kHeadDim>{}}, // stride_V
      {params.dk_row_stride, _1{}, Int<kHeadDim>{}}, // stride_dK
      {params.dv_row_stride, _1{}, Int<kHeadDim>{}}, // stride_dV
      static_cast<float*>(params.softmax_lse_log2_ptr),
      static_cast<float*>(params.dsoftmax_sum),
      {_4{}, params.total_q_rounded, params.h_qo}, // shape_LSEdPsum
      {_1{}, _4{}, params.total_q_rounded * 4}, // stride_LSE
      {_1{}, _4{}, params.total_q_rounded * 4}, // stride_dPsum
      params.scale_softmax,
      params.softcap,
      params.q_ranges,
      params.k_ranges,
      params.attn_type_map,
      params.bwd_kq_map,
      params.dq_determin_conflict_state,
      params.dq_determin_range_locks,
      params.index_sparse_indices,
      params.inner_indices_cnt};

  typename CollectiveEpilogue::Arguments epilogue_args{
      // q for outer-loop and k for inner-loop
      static_cast<typename CollectiveEpilogue::ElementDq*>(params.dq_ptr),
      {params.total_q, Int<kHeadDim>{}, params.h_qo}, // shape_dQ
      {params.dq_row_stride, _1{}, params.dq_head_stride}, // stride_dQ
      // k for outer-loop and q for inner-loop
      static_cast<typename CollectiveEpilogue::ElementDkv*>(params.dk_ptr),
      {params.total_k, Int<kHeadDim>{}, params.h_kv}, // shape_dK
      {params.dk_row_stride, _1{}, params.dk_head_stride}, // stride_dK
      static_cast<typename CollectiveEpilogue::ElementDkv*>(params.dv_ptr),
      {params.total_k, Int<kHeadDim>{}, params.h_kv}, // shape_dV
      {params.dv_row_stride, _1{}, params.dv_head_stride}, // stride_dV
      params.h_qo,
      params.h_kv,
      params.q_ranges,
      params.k_ranges,
      params.determin_range_locks,
  };

  typename flash::TileSchedulerArguments scheduler_args{/*num_heads_q=*/params.h_qo,
                                                        /*num_heads_kv=*/params.h_kv,
                                                        /*num_batches=*/params.merge_batch_size,
                                                        /*tile_count_semaphore=*/params.tile_count_semaphore,
                                                        /*ranges=*/BwdInnerLoopK ? params.q_ranges : params.k_ranges,
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

  if constexpr (!BwdInnerLoopK && !OuterStoreNeedReduction) {
    if constexpr (ProfileMode)
      MagiEvents::start("bwd_postprocess");

    run_flash_bwd_dkv_postprocess_<ElementDkv, kHeadDim>(params, stream);
    CHECK_CUDA_KERNEL_LAUNCH();
    if constexpr (ProfileMode)
      MagiEvents::stop("bwd_postprocess");
  }

  if constexpr (ProfileMode)
    MagiEvents::stop("bwd_run");
}

template <
    int Arch,
    typename T,
    typename TDq,
    typename TDkv,
    int kHeadDim,
    bool Has_softcap,
    bool OuterStoreNeedReduction,
    bool Deterministic,
    bool RangeMerge,
    bool BwdInnerLoopK,
    bool PackGQA,
    bool CatGQA,
    int PackGQAFactor,
    bool BlockSparse,
    bool IndexSparse,
    bool InnerDirMaxToMin,
    int MaskMode,
    bool InnerStoreInProducer,
    int BwdProducerRegs,
    int BwdConsumerRegs,
    int InnerStoreMode,
    int BwdTileM,
    int BwdTileN,
    int BwdStages,
    int BwdStagesDs,
    int BwdStagesV,
    int BwdTma1dSmemRowPad,
    int SparseKBlockSize,
    int InnerLoadMode,
    bool BwdUnionDkvSmem,
    int InnerStoreStages,
    int OuterStoreMode,
    bool ProfileMode,
    bool PerfDebugSkipVLoad,
    bool PerfDebugSkipDvStore,
    bool PerfDebugSkipDkStore,
    bool PerfDebugSkipDvMma>
void run_mha_bwd_(Flash_bwd_params& params, cudaStream_t stream) {
  static_assert(sizeof(T) == 2, "Only 16bit computation are supported");
  // BwdTileM/N, BwdStages/Ds: 0 = use default, >0 = override (env: MAGI_ATTENTION_FFA_BWD_TILE_M/N, MAGI_ATTENTION_FFA_BWD_STAGES/DS).
  static constexpr int kBlockM =
      BwdTileM > 0 ? BwdTileM : std::get<0>(tile_size_bwd_sm90<BwdInnerLoopK, (IndexSparse && !BwdInnerLoopK)>(kHeadDim, /*element_size=*/sizeof(T), Has_softcap));
  static constexpr int kBlockN =
      BwdTileN > 0 ? BwdTileN : std::get<1>(tile_size_bwd_sm90<BwdInnerLoopK, (IndexSparse && !BwdInnerLoopK)>(kHeadDim, /*element_size=*/sizeof(T), Has_softcap));

  static constexpr int Stages = BwdStages > 0 ? BwdStages : 2;
  static constexpr int Stages_dO = Stages >= 2 ? (kHeadDim <= 128 ? 2 : 1) : 1;
  static constexpr int Stages_dS = BwdStagesDs > 0 ? BwdStagesDs : (kHeadDim <= 128 ? (kBlockM <= 64 ? 2 : 1) : 1);
  static constexpr int Stages_V = BwdStagesV > 0 ? BwdStagesV : Stages;
  static constexpr int Tma1dSmemRowPad = BwdTma1dSmemRowPad;
  static constexpr bool UnionDkvSmem = BwdUnionDkvSmem;

  static constexpr bool SdP_swapAB = kHeadDim <= 128 ? true : false;
  static constexpr bool dKV_swapAB = kHeadDim <= 128 ? false : true;
  static constexpr bool dQ_swapAB = kHeadDim <= 64 ? false : true;

  // NOTE: when BwdInnerLoopK is true, we only support 2 NumConsumerWarpGroups,
  // since no more named barriers for more groups
  static constexpr int NumConsumerWarpGroups = BwdInnerLoopK ? 2 : (kHeadDim == 192 ? 3 : 2);

  // NOTE: when BwdInnerLoopK is not supported (i.e. always false),
  // all the atom layouts are set specifically for tile size (128, 128, 64) and (64, 128, 64),
  // however, when BwdInnerLoopK is true, we need to use new tile size due to shared memory limits,
  // including (64, 128, 64) and (64, 64, 128),
  // thus the atom layouts are accordingly adjusted here case-by-case,
  // but we need to find a better way to set these layout parameters.
  static constexpr int AtomLayoutMSdP = kBlockN <= 64 ? 2 : 1;
  static constexpr int AtomLayoutNdKV = kHeadDim <= 128 ? (kBlockN <= 64 ? 1 : 2) : 1;
  static constexpr int AtomLayoutMdQ = kHeadDim <= 64 ? (kBlockM <= 64 ? 1 : 2) : 1;

  if constexpr (RangeMerge) {
    assert(params.merge_k_ranges != nullptr && params.bwd_kq_map != nullptr && params.bwd_unique_count != nullptr);
  }

  run_flash_bwd<
      /*Arch=*/Arch,
      /*kHeadDim=*/kHeadDim,
      /*kBlockM=*/kBlockM,
      /*kBlockN=*/kBlockN,
      /*Has_softcap=*/Has_softcap,
      /*Element=*/T,
      /*ElementDq=*/TDq,
      /*ElementDkv=*/TDkv,
      /*Deterministic=*/Deterministic,
      /*BwdInnerLoopK=*/BwdInnerLoopK,
      /*PackGQA=*/PackGQA,
      /*CatGQA=*/CatGQA,
      /*PackGQAFactor=*/PackGQAFactor,
      /*Stages=*/Stages,
      /*Stages_dO=*/Stages_dO,
      /*Stages_dS=*/Stages_dS,
      /*SdP_swapAB=*/SdP_swapAB,
      /*dKV_swapAB=*/dKV_swapAB,
      /*dQ_swapAB=*/dQ_swapAB,
      /*NumConsumerWarpGroups=*/NumConsumerWarpGroups,
      /*AtomLayoutMSdP=*/AtomLayoutMSdP,
      /*AtomLayoutNdKV=*/AtomLayoutNdKV,
      /*AtomLayoutMdQ=*/AtomLayoutMdQ,
      /*RangeMerge=*/RangeMerge,
      /*BlockSparse=*/BlockSparse,
      /*IndexSparse=*/IndexSparse,
      /*InnerDirMaxToMin=*/InnerDirMaxToMin,
      /*MaskMode=*/MaskMode,
      /*InnerStoreInProducer=*/InnerStoreInProducer,
      /*BwdProducerRegs=*/BwdProducerRegs,
      /*BwdConsumerRegs=*/BwdConsumerRegs,
      /*InnerStoreMode=*/InnerStoreMode,
      /*OuterStoreNeedReduction=*/OuterStoreNeedReduction,
      /*OuterStoreMode=*/OuterStoreMode,
      /*Stages_V=*/Stages_V,
      /*Tma1dSmemRowPad=*/Tma1dSmemRowPad,
      /*SparseKBlockSize=*/SparseKBlockSize,
      /*InnerLoadMode=*/InnerLoadMode,
      /*UnionDkvSmem=*/UnionDkvSmem,
      /*InnerStoreStages=*/InnerStoreStages,
      /*ProfileMode=*/ProfileMode,
      /*PerfDebugSkipVLoad=*/PerfDebugSkipVLoad,
      /*PerfDebugSkipDvStore=*/PerfDebugSkipDvStore,
      /*PerfDebugSkipDkStore=*/PerfDebugSkipDkStore,
      /*PerfDebugSkipDvMma=*/PerfDebugSkipDvMma>(params, stream);
}
