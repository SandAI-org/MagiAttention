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

#include <cute/tensor.hpp>

#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h> // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/kernel_launch.h>

#include "epilogue_fwd.hpp"
#include "flash.h"
#include "flash_fwd_kernel_sm90.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "mainloop_fwd_sparse_sm90_gmma_ws.hpp"
#include "static_switch.h"
#include "tile_scheduler.hpp"
#include "tile_size.h"

using namespace cute;

template <
    int Arch,
    int kBlockM,
    int kBlockN,
    int kHeadDim,
    int ClusterM,
    typename Element,
    typename ElementOut,
    bool Has_softcap,
    bool DisableFwdAtomicReduction,
    bool Deterministic,
    bool MergeRange,
    bool SwapAB,
    bool ProfileMode = false,
    bool SparseLoad = false>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;
  // Get tile size and kernel configuration for SM90
  // if SwapAB, mma V @ P is SS mode
  static constexpr bool MmaPV_is_RS = !SwapAB;
  static constexpr bool IntraWGOverlap = true;

  static constexpr int kStages = 2;

  // get tile shape
  using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;
  // get cluster shape
  using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;

  // Get Mainloop, TileScheduler, Epilogue and AttnKernel
  using CollectiveMainloop = std::conditional_t<
      !SparseLoad,
      flash::CollectiveMainloopFwdSm90<
          kStages,
          ClusterShape,
          TileShape_MNK,
          Element,
          float,
          cutlass::arch::Sm90,
          Has_softcap,
          MmaPV_is_RS,
          IntraWGOverlap,
          MergeRange,
          SwapAB>,
      flash::CollectiveMainloopSparseFwdSm90<
          kStages,
          ClusterShape,
          TileShape_MNK,
          Element,
          float,
          cutlass::arch::Sm90,
          Has_softcap,
          MmaPV_is_RS,
          IntraWGOverlap,
          MergeRange>>;
  using Scheduler = flash::
      DynamicPersistentTileScheduler<kBlockM, CollectiveMainloop::NumMmaThreads, CollectiveMainloop::NumProducerThreads, Arch >= 90 /*WarpSpecialized*/, Deterministic>;
  using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
      TileShape_MNK_PV,
      ClusterShape,
      ElementOut,
      ArchTag,
      typename Scheduler::BlockCoordType,
      CollectiveMainloop::NumMmaThreads,
      DisableFwdAtomicReduction,
      Deterministic,
      SwapAB>;
  using AttnKernel = flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler, MergeRange>>;

  typename CollectiveMainloop::StrideV v_strides = make_stride(params.v_row_stride, _1{}, params.v_head_stride);
  typename CollectiveMainloop::Arguments mainloop_args = [&]() {
    if constexpr (SparseLoad) {
      return typename CollectiveMainloop::Arguments{
          static_cast<Element const*>(params.q_ptr), // Q
          {params.total_q, params.d, params.h_qo}, // shape_Q
          {params.q_row_stride, _1{}, params.q_head_stride}, // stride_Q
          static_cast<Element*>(params.k_ptr), // K
          {params.total_k, params.d, params.h_kv}, // shape_K
          {params.k_row_stride, _1{}, params.k_head_stride}, // stride_K
          static_cast<Element*>(params.v_ptr), // V
          params.d, // headdim_v
          v_strides, // stride_V
          params.scale_softmax,
          params.softcap,
          params.q_ranges,
          params.k_ranges,
          params.attn_type_map,
          params.qk_map,
          params.sparse_load_loop_count, // loop count for each unique Q range when sparse load
          params.sparse_load_invalid_count, // invalid token count for each unique Q range when sparse load
          params.equal_k_range_size // whether all K ranges are of equal size
      };
    } else {
      return typename CollectiveMainloop::Arguments{
          static_cast<Element const*>(params.q_ptr), // Q
          {params.total_q, params.d, params.h_qo}, // shape_Q
          {params.q_row_stride, _1{}, params.q_head_stride}, // stride_Q
          static_cast<Element*>(params.k_ptr), // K
          {params.total_k, params.d, params.h_kv}, // shape_K
          {params.k_row_stride, _1{}, params.k_head_stride}, // stride_K
          static_cast<Element*>(params.v_ptr), // V
          params.d, // headdim_v
          v_strides, // stride_V
          params.scale_softmax,
          params.softcap,
          params.q_ranges,
          params.k_ranges,
          params.attn_type_map,
          params.qk_map};
    }
  }();

  typename CollectiveEpilogue::Arguments epilogue_args{
      static_cast<ElementOut*>(params.o_ptr), // O
      {params.total_q, params.d, params.h_qo}, // shape_O
      {params.o_row_stride, _1{}, params.o_head_stride}, // stride_O
      static_cast<float*>(params.softmax_lse_ptr), // LSE
      {params.h_qo, _1{}}, // stride_LSE
      params.h_qo,
      params.h_kv,
      params.range_locks,
      params.q_ranges,
      params.k_ranges,
      params.determin_range_locks,
  };

  typename flash::TileSchedulerArguments scheduler_args{/*num_heads=*/params.h_qo,
                                                        /*num_batches=*/params.merge_batch_size,
                                                        /*tile_count_semaphore=*/params.tile_count_semaphore,
                                                        /*ranges=*/params.q_ranges,
                                                        /*merge_ranges=*/params.merge_q_ranges,
                                                        /*range_map=*/params.qk_map,
                                                        /*determin_conflict_state=*/params.determin_conflict_state,
                                                        /*unique_count=*/params.unique_count};

  int device;
  CHECK_CUDA(cudaGetDevice(&device));
  typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args});

  dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
  dim3 block_dims = AttnKernel::get_block_shape();
  int smem_size = AttnKernel::SharedStorageSize;
  // int smem_size_q = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_q));
  // int smem_size_k = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k));
  // int smem_size_v = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v));
  // printf("smem_size = %d, q = %d, k = %d, v = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v);
  // Get the ptr to kernel function.
  if constexpr (size(ClusterShape{}) > 1) {
    void const* kernel = (void const*)cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
  } else {
    auto kernel = cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    // kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
    cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/);
  }
  CHECK_CUDA_KERNEL_LAUNCH();
}

template <
    int Arch,
    int kBlockM,
    int kBlockN,
    typename T,
    typename T_out,
    int kHeadDim,
    bool Has_softcap,
    bool DisableFwdAtomicReduction,
    bool Deterministic,
    bool SwapAB,
    bool kProfileMode,
    bool kSparseLoad>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream) {
  static_assert(sizeof(T) == 2, "Only 16bit computation are supported");
  // TODO: support cluster launch
  static constexpr bool Enable_cluster = false;
  CLUSTER_SWITCH(cutlass::ceil_div(params.total_q, kBlockM) % 2 == 0, Use_cluster, [&] {
    static constexpr int ClusterM = Enable_cluster && Use_cluster ? 2 : 1;
    BOOL_SWITCH(params.merge_q_ranges != nullptr, MergeRange, [&] {
      run_flash_fwd<
          /*Arch=*/Arch,
          /*kBlockM=*/kBlockM,
          /*kBlockN=*/kBlockN,
          /*kHeadDim=*/kHeadDim,
          /*ClusterM=*/ClusterM,
          /*Element=*/T,
          /*ElementOut=*/T_out,
          /*Has_softcap=*/Has_softcap,
          /*DisableFwdAtomicReduction=*/DisableFwdAtomicReduction,
          /*Deterministic=*/Deterministic,
          /*MergeRange=*/MergeRange,
          /*SwapAB=*/SwapAB,
          /*ProfileMode=*/kProfileMode,
          /*SparseLoad=*/kSparseLoad>(params, stream);
    });
  });
}
