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

#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

#include "bwd_tile_scheduler.hpp"
#include "mask.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_, bool RangeMerge_, bool InnerDirMaxToMin_, int ProducerRegs_, int ConsumerRegs_>
class FlashAttnBwdSm90 {
 public:
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
  using TiledMmaSdP = typename CollectiveMainloop::TiledMmaSdP;
  using TiledMmadKV = typename CollectiveMainloop::TiledMmadKV;
  using TiledMmadQ = typename CollectiveMainloop::TiledMmadQ;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ClusterShape = typename CollectiveMainloop::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  static constexpr bool dKV_swapAB = CollectiveMainloop::dKV_swapAB;
  static constexpr bool dQ_swapAB = CollectiveMainloop::dQ_swapAB;
  static constexpr bool BwdInnerLoopK = CollectiveMainloop::BwdInnerLoopK;
  static constexpr bool BlockSparse = CollectiveMainloop::BlockSparse;
  static constexpr bool IndexSparse = CollectiveMainloop::IndexSparse;
  static constexpr bool IsSparse = CollectiveMainloop::IsSparse;
  static constexpr bool InnerStoreInProducer = CollectiveMainloop::InnerStoreInProducer;
  static constexpr InnerLoadMode kInnerLoadMode = CollectiveMainloop::kInnerLoadMode;
  static constexpr int NumProducerLoaderThreads = CollectiveMainloop::ProducerConsts::kLoaderThreads;

  template <typename Pipeline, typename Storage, typename PipelineParamsT>
  CUTLASS_DEVICE static Pipeline make_inner_pipeline(Storage& storage, PipelineParamsT const& pipeline_params) {
    if constexpr (kInnerLoadMode == InnerLoadMode::Tma) {
      return Pipeline(storage, pipeline_params, ClusterShape{});
    } else {
      return Pipeline(storage, pipeline_params);
    }
  }

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static constexpr bool Deterministic = CollectiveEpilogue::Deterministic;

  // Sanity check
  static_assert(ArchTag::kMinComputeCapability >= 90);

  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename flash::TileSchedulerArguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using BwdNamedBarriers = std::conditional_t<BwdInnerLoopK, BwdNamedBarriersLoopK, BwdNamedBarriersLoopQ>;

  static constexpr bool RangeMerge = RangeMerge_;
  static constexpr auto kInnerDir = InnerDirMaxToMin_ ? flash::DispatchDirection::MaxToMin : flash::DispatchDirection::MinToMax;
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumConsumerWarpGroups = CUTE_STATIC_V(size(TiledMmaSdP{})) / cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaSdP{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static_assert(NumConsumerWarpGroups == 2 || NumConsumerWarpGroups == 3);
  static_assert(BarrierManager::check<BwdNamedBarriers, NumConsumerWarpGroups>());

  static_assert(ProducerRegs_ % 8 == 0 && ProducerRegs_ >= 24 && ProducerRegs_ <= 256);
  static_assert(ConsumerRegs_ % 8 == 0 && ConsumerRegs_ >= 24 && ConsumerRegs_ <= 256);
  static constexpr uint32_t kTotalWarpGroups = NumLoadWarpGroups + NumConsumerWarpGroups;
  static constexpr uint32_t kAvgRegsPerThread = (65536 / (kTotalWarpGroups * 128) / 8) * 8;
  static_assert(NumLoadWarpGroups * ProducerRegs_ + NumConsumerWarpGroups * ConsumerRegs_ <= kAvgRegsPerThread * kTotalWarpGroups);

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128> {
      union {
        typename CollectiveMainloop::TensorStorage mainloop;
        typename CollectiveEpilogue::TensorStorage epilogue;
      };
    } tensors;

    // k for outer-loop and q for inner-loop
    struct PipelineStorageLoopQ : cute::aligned_struct<16> {
      alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_KV;
      alignas(16) typename CollectiveMainloop::MainloopPipeline::SharedStorage pipeline_q;
      alignas(16) typename CollectiveMainloop::MainloopPipeline_dO::SharedStorage pipeline_do;
      alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
    };

    // q for outer-loop and k for inner-loop
    struct PipelineStorageLoopK : cute::aligned_struct<16> {
      alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_QdO;
      alignas(16) typename CollectiveMainloop::MainloopPipeline::SharedStorage pipeline_k;
      alignas(16) typename CollectiveMainloop::MainloopPipeline_V::SharedStorage pipeline_v;
      alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
    };

    using PipelineStorage = std::conditional_t<BwdInnerLoopK, PipelineStorageLoopK, PipelineStorageLoopQ>;

    PipelineStorage pipelines;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the
  // aliased type.
  static Params to_underlying_arguments(Arguments const& args) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
    return {
        CollectiveMainloop::to_underlying_arguments(args.mainloop),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue),
        hw_info,
        TileScheduler::to_underlying_arguments(args.scheduler)};
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  // Entry point
  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    if constexpr (BwdInnerLoopK) { // q for outer-loop and k for inner-loop
      run_bwd_with_loop_k(params, smem_buf);
    } else { // k for outer-loop and q for inner-loop
      run_bwd_with_loop_q(params, smem_buf);
    }
  }

  // Run the FFA backward pass
  // k for outer-loop and q for inner-loop
  CUTLASS_DEVICE
  void run_bwd_with_loop_q(Params const& params, char* smem_buf) {
    static_assert(!BwdInnerLoopK, "run_bwd_with_loop_q() must be called when BwdInnerLoopK is false");

    static constexpr int NumConsumerThreads = NumConsumerWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumCopyThreads = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipeline_dO = typename CollectiveMainloop::MainloopPipeline_dO;
    using PipelineParams_dO = typename MainloopPipeline_dO::Params;
    using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
    static constexpr bool Q_dO_same_stages = std::is_same_v<MainloopPipeline, MainloopPipeline_dO>;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // Get thread index in warp group
    int const warp_group_thread_idx = canonical_thread_idx_in_warpgroup_nosync();
    // Get warp group index
    int const warp_group_idx = cutlass::canonical_warp_group_idx();

    // Initialize the barriers of K,V
    if (warp_idx == 0 && lane_predicate) {
      shared_storage.pipelines.barrier_KV.init(/*numThreads=*/1);
    }

    PipelineParams pipeline_params_q;
    if constexpr ((kInnerLoadMode == InnerLoadMode::Tma)) {
      pipeline_params_q.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ + CollectiveMainloop::TmaTransactionBytesLSE;
      pipeline_params_q.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
      pipeline_params_q.is_leader = warp_group_thread_idx == 0;
      pipeline_params_q.num_consumers = NumConsumerThreads;
    } else {
      pipeline_params_q.consumer_arv_count = NumConsumerThreads;
      pipeline_params_q.producer_arv_count = NumProducerLoaderThreads;
    }
    MainloopPipeline pipeline_q = make_inner_pipeline<MainloopPipeline>(shared_storage.pipelines.pipeline_q, pipeline_params_q);

    PipelineParams_dO pipeline_params_do;
    if constexpr ((kInnerLoadMode == InnerLoadMode::Tma)) {
      auto role_do = warp_group_idx == 0 ? MainloopPipeline_dO::ThreadCategory::Producer : MainloopPipeline_dO::ThreadCategory::Consumer;
      pipeline_params_do = {pipeline_params_q.transaction_bytes, role_do, pipeline_params_q.is_leader, pipeline_params_q.num_consumers};
    } else {
      pipeline_params_do.consumer_arv_count = NumConsumerThreads;
      pipeline_params_do.producer_arv_count = NumProducerLoaderThreads;
    }
    MainloopPipeline_dO pipeline_do = make_inner_pipeline<MainloopPipeline_dO>(
        shared_storage.pipelines.pipeline_do,
        cute::conditional_return < Q_dO_same_stages && (kInnerLoadMode == InnerLoadMode::Tma) > (pipeline_params_q, pipeline_params_do));

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    // We need this to guarantee that pipeline initialization is visible to
    // all producers and consumer blocks in the cluster
    sync_cga_threads<ClusterShape>();

    TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

    using BlockMetaT = typename CollectiveMainloop::template BlockMeta</*IsProducer=*/true>;
    using BlockMetaConsumerT = typename CollectiveMainloop::template BlockMeta</*IsProducer=*/false>;

    using ProducerBlockMetaT = std::conditional_t<IsSparse, typename CollectiveMainloop::template SparseLoopQBlockMeta<true>, BlockMetaT>;

    using ProducerConsts = typename CollectiveMainloop::ProducerConsts;

    if (warp_group_idx == 0) { // Producer
      cutlass::arch::warpgroup_reg_dealloc<ProducerRegs_>();

      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
      bool const is_loader = ProducerConsts::is_loader(warp_idx_in_warpgroup);
      bool const is_inner_storer = ProducerConsts::is_inner_storer(warp_idx_in_warpgroup);

      if (is_loader) { // Load K,V and pipeline Q,dO
        // Initialize producer write pipeline states of Q,dO
        PipelineState smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState_dO smem_pipe_write_do = cutlass::make_producer_start_state<MainloopPipeline_dO>();

        // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
        BarrierManager::sync<NumConsumerThreads + NumProducerLoaderThreads>(BwdNamedBarriers::KVEmpty);

        bool const is_leader_warp = warp_idx_in_warpgroup == 0;

        // For each work tile job:
        //  1. load this n block of K,V from global memory into shared memory
        //  2. pipeline the loads of Q,dO for each m block from global memory into shared memory
        // Tracks if ANY outer tile was valid across the loop. If none were valid
        // (all work tiles empty in sparse scenarios), load_tail must be skipped
        // to avoid deadlocking on a pipeline that was never produced into.
        bool any_tile_valid = false;
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_block_info = is_leader_warp ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler)
                                                   : scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
             work_block_info.is_valid(params.scheduler);
             work_block_info = is_leader_warp ? scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_block_info)
                                              : scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_block_info)) {
          auto block_coord = work_block_info.get_block_coord();

          auto scheduler_prefetch = [&scheduler, &params, &work_block_info]() { scheduler.prefetch_next_work(params.scheduler, work_block_info); };

          // Run the producer load pipeline
          int thread_idx = threadIdx.x % NumProducerLoaderThreads;
          ProducerBlockMetaT block_meta{params.mainloop, block_coord, shared_storage, thread_idx};
          bool tile_valid = mainloop.template load_with_loop_q<kInnerDir>(
              params.mainloop, pipeline_q, pipeline_do, smem_pipe_write_q, smem_pipe_write_do, shared_storage, block_meta);

          // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
          if (tile_valid) {
            any_tile_valid = true;
            BarrierManager::sync<NumConsumerThreads + NumProducerLoaderThreads>(BwdNamedBarriers::KVEmpty);
          }

          scheduler_prefetch();
        }
        // Skip producer_tail when no valid tiles were processed: the consumer
        // never consumed any pipeline stages, so producer_tail would deadlock
        // waiting on the empty_barrier (same pattern as FWD fix bbd75d69).
        if (any_tile_valid) {
          mainloop.load_tail_with_loop_q(pipeline_q, pipeline_do, smem_pipe_write_q, smem_pipe_write_do);
        }
      } else if (is_inner_storer) { // store partial dQ (TMA or scatter reduce-add)
        // For each work tile job:
        //  1. atomic reduce-add the computed partial dQ from shared memory into global memory
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_block_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_block_info.is_valid(params.scheduler);
             work_block_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_block_info)) {
          auto block_coord = work_block_info.get_block_coord();

          if constexpr (IsSparse) {
            typename CollectiveMainloop::template SparseLoopQBlockMeta<false> block_meta{params.mainloop, block_coord, shared_storage};
            mainloop.template store_dq<kInnerDir>(params.mainloop, shared_storage, block_meta);
          } else {
            BlockMetaT block_meta{params.mainloop, block_coord, shared_storage};
            mainloop.template store_dq<kInnerDir>(params.mainloop, shared_storage, block_meta);
          }
        }
      }
    } else { // Consumer
      // Allocate the registers for the consumer WGs
      cutlass::arch::warpgroup_reg_alloc<ConsumerRegs_>();

      // Initialize tiled mma object for dK=dS^TQ, dV=P^TdO
      TiledMmadKV tiled_mma_dKV;

      // Initialize consumer read pipeline states of Q,dO
      PipelineState smem_pipe_read_q;
      PipelineState_dO smem_pipe_read_do;

      // Initialize mma consumers
      mainloop.mma_init();
      scheduler.init_consumer();

      // For each work tile job:
      //  1. run mma consumer to compute partial dQ,dK,dV as the consumer prologue/mainloop
      //  2. accumulate partial dK,dV into the zero-initialized register fragments
      //  3. atomic reduce-add partial dQ into the global memory
      //  4. store the reduced dK,dV into the global memory as the consumer epilogue
      int work_idx = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (auto work_block_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_block_info.is_valid(params.scheduler);
           work_block_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_block_info)) {
        auto block_coord = work_block_info.get_block_coord();
        auto det_msg = work_block_info.get_det_msg();

        // Init the zero-initialized register SMEM buffer for dK and dV
        Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
        Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
        clear(tdKrdK);
        clear(tdVrdV);

        // Run the mma to compute partial dQ,dK,dV
        using ConsumerBlockMetaT = std::conditional_t<IsSparse, typename CollectiveMainloop::template SparseLoopQBlockMeta<false>, BlockMetaConsumerT>;
        ConsumerBlockMetaT block_meta{params.mainloop, block_coord, shared_storage};

        auto epilogue_block_coord = block_meta.get_epilogue_coord();

        bool tile_valid = mainloop.template mma_with_loop_q<kInnerDir>(
            params.mainloop,
            pipeline_q,
            pipeline_do,
            smem_pipe_read_q,
            smem_pipe_read_do,
            tdKrdK,
            tdVrdV,
            threadIdx.x - NumCopyThreads,
            work_idx,
            block_meta,
            shared_storage);

        // Run the epilogue to store reduced dK (scaled),dV
        if (tile_valid) {
#pragma unroll
          for (int i = 0; i < size(tdKrdK); ++i) {
            tdKrdK(i) *= params.mainloop.softmax_scale;
          }
          ++work_idx;
          epilogue.store_dkv(params.epilogue, tdKrdK, tdVrdV, shared_storage, tiled_mma_dKV, threadIdx.x - NumCopyThreads, epilogue_block_coord, det_msg);
          BarrierManager::arrive<NumConsumerThreads + NumProducerLoaderThreads>(BwdNamedBarriers::KVEmpty);
        } else {
          epilogue.store_zero_dkv(params.epilogue, threadIdx.x - NumCopyThreads, epilogue_block_coord, det_msg);
        }
      }
      epilogue.store_tail();
    }
  }

  // Run the FFA backward pass
  // q for outer-loop and k for inner-loop
  CUTLASS_DEVICE
  void run_bwd_with_loop_k(Params const& params, char* smem_buf) {
    static_assert(BwdInnerLoopK, "run_bwd_with_loop_k() must be called when BwdInnerLoopK is true");

    static constexpr int NumConsumerThreads = NumConsumerWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumCopyThreads = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    using PipelineState = typename CollectiveMainloop::PipelineState;
    using MainloopPipeline_V = typename CollectiveMainloop::MainloopPipeline_V;
    using PipelineState_V = typename CollectiveMainloop::PipelineState_V;
    using PipelineParams = typename MainloopPipeline::Params;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // Get thread index in warp group
    int const warp_group_thread_idx = canonical_thread_idx_in_warpgroup_nosync();
    // Get warp group index
    int const warp_group_idx = cutlass::canonical_warp_group_idx();

    // Initialize the barriers of Q,dO
    if (warp_idx == 0 && lane_predicate) {
      shared_storage.pipelines.barrier_QdO.init(/*numThreads=*/1);
    }

    // Initialize pipelines of K,V
    // NOTE: we're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    PipelineParams pipeline_params_k;
    pipeline_params_k.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
    if constexpr ((kInnerLoadMode == InnerLoadMode::Tma)) {
      pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
      pipeline_params_k.is_leader = warp_group_thread_idx == 0;
      pipeline_params_k.num_consumers = NumConsumerThreads;
    } else {
      pipeline_params_k.consumer_arv_count = NumConsumerThreads;
      pipeline_params_k.producer_arv_count = NumProducerLoaderThreads;
    }
    using PipelineParams_V = typename CollectiveMainloop::MainloopPipeline_V::Params;
    PipelineParams_V pipeline_params_v;
    pipeline_params_v.role = warp_group_idx == 0 ? MainloopPipeline_V::ThreadCategory::Producer : MainloopPipeline_V::ThreadCategory::Consumer;
    if constexpr ((kInnerLoadMode == InnerLoadMode::Tma)) {
      pipeline_params_v.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
      pipeline_params_v.is_leader = warp_group_thread_idx == 0;
      pipeline_params_v.num_consumers = NumConsumerThreads;
    } else {
      pipeline_params_v.consumer_arv_count = NumConsumerThreads;
      pipeline_params_v.producer_arv_count = NumProducerLoaderThreads;
    }

    MainloopPipeline pipeline_k = make_inner_pipeline<MainloopPipeline>(shared_storage.pipelines.pipeline_k, pipeline_params_k);
    MainloopPipeline_V pipeline_v = make_inner_pipeline<MainloopPipeline_V>(shared_storage.pipelines.pipeline_v, pipeline_params_v);

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    // We need this to guarantee that pipeline initialization is visible to
    // all producers and consumer blocks in the cluster
    sync_cga_threads<ClusterShape>();

    TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

    using BlockMetaT = typename CollectiveMainloop::template BlockMeta</*IsProducer=*/true>;
    using BlockMetaConsumerT = std::conditional_t<
        IsSparse,
        typename CollectiveMainloop::template SparseLoopKBlockMeta<false>,
        typename CollectiveMainloop::template BlockMeta</*IsProducer=*/false>>;

    if (warp_group_idx == 0) { // Producer
      // Deallocate the registers for the producer WG,
      // which allows the consumer WGs to have more registers
      cutlass::arch::warpgroup_reg_dealloc<ProducerRegs_>();

      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

      using ProducerBlockMetaT = std::conditional_t<IsSparse, typename CollectiveMainloop::template SparseLoopKBlockMeta<true>, BlockMetaT>;

      using ProducerConsts_LoopK = typename CollectiveMainloop::ProducerConsts;
      bool const is_loader = ProducerConsts_LoopK::is_loader(warp_idx_in_warpgroup);
      bool const is_inner_storer = ProducerConsts_LoopK::is_inner_storer(warp_idx_in_warpgroup);

      if (is_loader) { // Load Q,dO and pipeline K,V
        // Initialize producer write pipeline states of K,V
        PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState_V smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline_V>();

        // Wait for the MMA warpgroups to say that smem_q and smem_do are ready
        BarrierManager::sync<NumConsumerThreads + NumProducerLoaderThreads>(BwdNamedBarriers::QdOEmpty);

        // For each work tile job:
        //  1. load this m block of Q,dO from global memory into shared memory
        //  2. pipeline the loads of K,V for each n block from global memory into shared memory
        bool const is_leader_warp = warp_idx_in_warpgroup == 0;
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_block_info = is_leader_warp ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler)
                                                   : scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
             work_block_info.is_valid(params.scheduler);
             work_block_info = is_leader_warp ? scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_block_info)
                                              : scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_block_info)) {
          auto block_coord = work_block_info.get_block_coord();
          auto scheduler_prefetch = [&scheduler, &params, &work_block_info]() { scheduler.prefetch_next_work(params.scheduler, work_block_info); };

          // Run the producer load pipeline
          int thread_idx = threadIdx.x % NumProducerLoaderThreads;
          ProducerBlockMetaT block_meta{params.mainloop, block_coord, shared_storage, thread_idx};
          bool has_tile_valid =
              mainloop.template load_with_loop_k<kInnerDir>(params.mainloop, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v, shared_storage, block_meta);

          // Wait for the MMA warpgroups to say that smem_q and smem_do are ready
          if (has_tile_valid) {
            BarrierManager::sync<NumConsumerThreads + NumProducerLoaderThreads>(BwdNamedBarriers::QdOEmpty);
          }

          scheduler_prefetch();
        }
        mainloop.load_tail_with_loop_k(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
      } else if (is_inner_storer) { // store partial dKV
        // For each work tile job:
        //  1. atomic reduce-add the computed partial dK,dV from shared memory into global memory
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_block_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_block_info.is_valid(params.scheduler);
             work_block_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_block_info)) {
          auto block_coord = work_block_info.get_block_coord();

          BlockMetaConsumerT block_meta{params.mainloop, block_coord, shared_storage};
          mainloop.template store_dkv<kInnerDir>(params.mainloop, shared_storage, block_meta);
        }
      }
    } else { // Consumer
      // Allocate the registers for the consumer WGs
      cutlass::arch::warpgroup_reg_alloc<ConsumerRegs_>();

      // Initialize tiled mma object for dQ=dSK
      TiledMmadQ tiled_mma_dQ;

      // Initialize consumer read pipeline states of K,V
      PipelineState smem_pipe_read_k;
      PipelineState_V smem_pipe_read_v;

      // Initialize mma consumers
      mainloop.mma_init();
      scheduler.init_consumer();

      // For each work tile job:
      //  1. run mma consumer to compute partial dQ,dK,dV as the consumer prologue/mainloop
      //  2. accumulate partial dQ into the zero-initialized register fragments
      //  3. atomic reduce-add partial dK,dV into the global memory
      //  4. store the reduced dQ into the global memory as the consumer epilogue
      int work_idx = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (auto work_block_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_block_info.is_valid(params.scheduler);
           work_block_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_block_info)) {
        auto block_coord = work_block_info.get_block_coord();

        // Init the zero-initialized register SMEM buffer for dQ
        Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
        clear(tdQrdQ);

        // Run the mma to compute partial dQ,dK,dV
        BlockMetaConsumerT block_meta{params.mainloop, block_coord, shared_storage};
        auto epilogue_block_coord = block_meta.get_epilogue_coord();

        bool tile_valid = mainloop.template mma_with_loop_k<kInnerDir>(
            params.mainloop, pipeline_k, pipeline_v, smem_pipe_read_k, smem_pipe_read_v, tdQrdQ, threadIdx.x - NumCopyThreads, work_idx, block_meta, shared_storage);

        // Run the epilogue to store reduced dQ (scaled)
        if (tile_valid) {
#pragma unroll
          for (int i = 0; i < size(tdQrdQ); ++i) {
            tdQrdQ(i) *= params.mainloop.softmax_scale;
          }
          ++work_idx;
          if constexpr (!Deterministic) {
            epilogue.store_dq(params.epilogue, tdQrdQ, shared_storage, tiled_mma_dQ, threadIdx.x - NumCopyThreads, epilogue_block_coord);
          } else {
            static_assert(!Deterministic, "Deterministic mode is not supported yet when BwdInnerLoopK is true.");
          }
          BarrierManager::arrive<NumConsumerThreads + NumProducerLoaderThreads>(BwdNamedBarriers::QdOEmpty);
        } else {
          if constexpr (!Deterministic) {
            epilogue.store_zero_dq(params.epilogue, threadIdx.x - NumCopyThreads, epilogue_block_coord);
          } else {
            static_assert(!Deterministic, "Deterministic mode is not supported yet when BwdInnerLoopK is true.");
          }
        }
      }
      epilogue.store_tail();
    }
  }
};

} // namespace flash
