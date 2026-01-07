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

#include "tile_scheduler.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_, bool RangeMerge_>
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
  static constexpr bool SwapBwdQKLoop = CollectiveMainloop::SwapBwdQKLoop;

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
  using BwdNamedBarriers = std::conditional_t<SwapBwdQKLoop, BwdNamedBarriersLoopK, BwdNamedBarriersLoopQ>;
  static_assert(
      static_cast<uint32_t>(BwdNamedBarriers::kNumBarriers) <= MaxNumUserNamedBarriers,
      "Exceeding the maximum number of user defined named barriers allowed.");

  static constexpr bool RangeMerge = RangeMerge_;
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMmaSdP{})) / cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaSdP{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

  // Register requirement for Load and Math WGs
  // static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 2 ? 24 : 32;
  // static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 240 : 160;
  // If you want to print from the producer warp, you'd need to increase the
  // number of registers Otherwise you'll get CUDA error.
  // we allocate more registers for producer to avoid register spilling for now.
  static constexpr uint32_t LoadRegisterRequirement = 40;
  static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 232 : 152;

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
      alignas(16) typename CollectiveMainloop::MainloopPipeline::SharedStorage pipeline_v;
      alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
    };

    using PipelineStorage = std::conditional_t<SwapBwdQKLoop, PipelineStorageLoopK, PipelineStorageLoopQ>;

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
    if constexpr (SwapBwdQKLoop) { // q for outer-loop and k for inner-loop
      run_bwd_with_loop_k(params, smem_buf);
    } else { // k for outer-loop and q for inner-loop
      run_bwd_with_loop_q(params, smem_buf);
    }
  }

  // Run the FFA backward pass
  // k for outer-loop and q for inner-loop
  CUTLASS_DEVICE
  void run_bwd_with_loop_q(Params const& params, char* smem_buf) {
    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
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

    // Initialize pipelines of Q,dO
    // NOTE: we're counting on pipeline_q to call cutlass::arch::fence_barrier_init();
    PipelineParams pipeline_params_q;
    pipeline_params_q.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ + CollectiveMainloop::TmaTransactionBytesLSE;
    pipeline_params_q.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_q.is_leader = warp_group_thread_idx == 0;
    pipeline_params_q.num_consumers = NumMmaThreads;
    MainloopPipeline pipeline_q(shared_storage.pipelines.pipeline_q, pipeline_params_q, ClusterShape{});

    auto role_do = warp_group_idx == 0 ? MainloopPipeline_dO::ThreadCategory::Producer : MainloopPipeline_dO::ThreadCategory::Consumer;
    PipelineParams_dO pipeline_params_do{pipeline_params_q.transaction_bytes, role_do, pipeline_params_q.is_leader, pipeline_params_q.num_consumers};
    MainloopPipeline_dO pipeline_do( // Q,LSE share the same pipeline params as dO,dPsum
        shared_storage.pipelines.pipeline_do, cute::conditional_return<Q_dO_same_stages>(pipeline_params_q, pipeline_params_do), ClusterShape{});

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    // We need this to guarantee that pipeline initialization is visible to
    // all producers and consumer blocks in the cluster
    sync_cga_threads<ClusterShape>();

    TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

    if (warp_group_idx == 0) { // Producer
      // Deallocate the registers for the producer WG,
      // which allows the consumer WGs to have more registers
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
      if (warp_idx_in_warpgroup == 0) { // Load K,V and pipeline Q,dO
        // Initialize producer write pipeline states of Q,dO
        PipelineState smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState_dO smem_pipe_write_do = cutlass::make_producer_start_state<MainloopPipeline_dO>();

        // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, /*id=*/static_cast<uint32_t>(BwdNamedBarriers::KVEmpty));

        // For each work tile job:
        //  1. load this n block of K,V from global memory into shared memory
        //  2. pipeline the loads of Q,dO for each m block from global memory into shared memory
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler); work_tile_info.is_valid(params.scheduler);
             work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info)) {
          // get block_coord without deterministic message
          auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
          auto block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), get<2>(block_coord_));
          auto [n_block, bidh, bidb_idx] = block_coord;

          auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() { scheduler.prefetch_next_work(params.scheduler, work_tile_info); };

          // Run the producer load pipeline
          bool tile_valid = false;
          if constexpr (RangeMerge) {
            int loop_count = (bidb_idx < *params.scheduler.unique_count - 1) ? (params.scheduler.range_map[bidb_idx + 1] - params.scheduler.range_map[bidb_idx])
                                                                             : (params.scheduler.num_batches - params.scheduler.range_map[bidb_idx]);
            int bidb_start = params.scheduler.range_map[bidb_idx];

            for (int idx = 0; idx < loop_count; ++idx) {
              int bidb = bidb_start + idx;
              block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), bidb);
              bool tile_valid_tmp =
                  mainloop.load_with_loop_q(params.mainloop, pipeline_q, pipeline_do, smem_pipe_write_q, smem_pipe_write_do, shared_storage, block_coord, tile_valid);

              tile_valid = tile_valid || tile_valid_tmp;
            }
          } else {
            tile_valid =
                mainloop.load_with_loop_q(params.mainloop, pipeline_q, pipeline_do, smem_pipe_write_q, smem_pipe_write_do, shared_storage, block_coord, tile_valid);
          }

          // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
          if (tile_valid) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, /*id=*/static_cast<uint32_t>(BwdNamedBarriers::KVEmpty));
          }

          scheduler_prefetch();
        }
        mainloop.load_tail_with_loop_q(pipeline_q, pipeline_do, smem_pipe_write_q, smem_pipe_write_do);
      } else if (warp_idx_in_warpgroup == 1) { // store partial dQ
        // For each work tile job:
        //  1. atomic reduce-add the computed partial dQ from shared memory into global memory
        int bidb_last = 0;
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_tile_info.is_valid(params.scheduler);
             work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
          // get block_coord without deterministic message
          auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
          auto block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), get<2>(block_coord_));
          auto [n_block, bidh, bidb_idx] = block_coord;

          if constexpr (RangeMerge) {
            int loop_count = (bidb_idx < *params.scheduler.unique_count - 1) ? (params.scheduler.range_map[bidb_idx + 1] - params.scheduler.range_map[bidb_idx])
                                                                             : (params.scheduler.num_batches - params.scheduler.range_map[bidb_idx]);
            int bidb_start = params.scheduler.range_map[bidb_idx];

            for (int idx = 0; idx < loop_count; ++idx) {
              int bidb = bidb_start + idx;
              block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), bidb);
              if constexpr (!Deterministic) {
                mainloop.store_dq(params.mainloop, shared_storage, block_coord);
              } else {
                mainloop.store_dq(params.mainloop, shared_storage, block_coord, bidb_last);
                bidb_last = bidb;
              }
            }
          } else {
            if constexpr (!Deterministic) {
              mainloop.store_dq(params.mainloop, shared_storage, block_coord);
            } else {
              mainloop.store_dq(params.mainloop, shared_storage, block_coord, bidb_last);
              bidb_last = bidb_idx;
            }
          }
        }
      }
    } else { // Consumer
      // Allocate the registers for the consumer WGs
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

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
      for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_tile_info.is_valid(params.scheduler);
           work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
        // Get block_coord without deterministic message
        auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
        auto block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), get<2>(block_coord_));

        // Init the zero-initialized register accumulator for dK and dV
        Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
        Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
        clear(tdKrdK);
        clear(tdVrdV);

        // Run the mma to compute partial dQ,dK,dV
        bool tile_valid = false;
        if constexpr (RangeMerge) {
          int bidb_idx = get<2>(block_coord);
          int loop_count = (bidb_idx < *params.scheduler.unique_count - 1) ? (params.scheduler.range_map[bidb_idx + 1] - params.scheduler.range_map[bidb_idx])
                                                                           : (params.scheduler.num_batches - params.scheduler.range_map[bidb_idx]);
          int bidb_start = params.scheduler.range_map[bidb_idx];

          for (int idx = 0; idx < loop_count; ++idx) {
            int bidb = bidb_start + idx;
            block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), bidb);

            // dK and dV output accumulator.
            bool tile_valid_tmp = mainloop.mma_with_loop_q(
                params.mainloop,
                pipeline_q,
                pipeline_do,
                smem_pipe_read_q,
                smem_pipe_read_do,
                tdKrdK,
                tdVrdV,
                threadIdx.x - NumCopyThreads,
                work_idx,
                block_coord,
                shared_storage,
                tile_valid);

            tile_valid = tile_valid || tile_valid_tmp;
          }
          if constexpr (Deterministic) {
            cute::get<2>(block_coord_) = get<2>(block_coord);
          }
        } else {
          tile_valid = mainloop.mma_with_loop_q(
              params.mainloop,
              pipeline_q,
              pipeline_do,
              smem_pipe_read_q,
              smem_pipe_read_do,
              tdKrdK,
              tdVrdV,
              threadIdx.x - NumCopyThreads,
              work_idx,
              block_coord,
              shared_storage,
              tile_valid);
        }

        // Run the epilogue to store reduced dK (scaled),dV
        // NOTE: dQ is scaled inside mma_with_loop_k before atomic reduce-add
        if (tile_valid) {
#pragma unroll
          for (int i = 0; i < size(tdKrdK); ++i) {
            tdKrdK(i) *= params.mainloop.softmax_scale;
          }
          ++work_idx;
          if constexpr (!Deterministic) {
            epilogue.store_dkv(params.epilogue, tdKrdK, tdVrdV, shared_storage, tiled_mma_dKV, threadIdx.x - NumCopyThreads, block_coord);
          } else {
            epilogue.store_dkv(params.epilogue, tdKrdK, tdVrdV, shared_storage, tiled_mma_dKV, threadIdx.x - NumCopyThreads, block_coord_);
          }
          cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, /*id=*/static_cast<uint32_t>(BwdNamedBarriers::KVEmpty));
        } else {
          if constexpr (!Deterministic) {
            epilogue.store_zero_dkv(params.epilogue, threadIdx.x - NumCopyThreads, block_coord);
          } else {
            epilogue.store_zero_dkv(params.epilogue, threadIdx.x - NumCopyThreads, block_coord_);
          }
        }
      }
      epilogue.store_tail();
    }
  }

  // Run the FFA backward pass
  // q for outer-loop and k for inner-loop
  CUTLASS_DEVICE
  void run_bwd_with_loop_k(Params const& params, char* smem_buf) {
    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumCopyThreads = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    using PipelineState = typename CollectiveMainloop::PipelineState;
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
    pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params_k.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_k.is_leader = warp_group_thread_idx == 0;
    pipeline_params_k.num_consumers = NumMmaThreads;
    PipelineParams pipeline_params_v = pipeline_params_k; // K,V share the same pipeline params

    MainloopPipeline pipeline_k(shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipelines.pipeline_v, pipeline_params_v, ClusterShape{});

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    // We need this to guarantee that pipeline initialization is visible to
    // all producers and consumer blocks in the cluster
    sync_cga_threads<ClusterShape>();

    TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

    if (warp_group_idx == 0) { // Producer
      // Deallocate the registers for the producer WG,
      // which allows the consumer WGs to have more registers
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
      if (warp_idx_in_warpgroup == 0) { // Load Q,dO and pipeline K,V
        // Initialize producer write pipeline states of K,V
        PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

        // Wait for the MMA warpgroups to say that smem_q and smem_do are ready
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, /*id=*/static_cast<uint32_t>(BwdNamedBarriers::QdOEmpty));

        // For each work tile job:
        //  1. load this m block of Q,dO from global memory into shared memory
        //  2. pipeline the loads of K,V for each n block from global memory into shared memory
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler); work_tile_info.is_valid(params.scheduler);
             work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info)) {
          // get block_coord without deterministic message
          auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
          auto block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), get<2>(block_coord_));
          auto [m_block, bidh, bidb_idx] = block_coord;

          auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() { scheduler.prefetch_next_work(params.scheduler, work_tile_info); };

          // Run the producer load pipeline
          bool tile_valid = false;
          if constexpr (RangeMerge) {
            static_assert(!RangeMerge, "RangeMerge mode is not supported yet when SwapBwdQKLoop is true.");
          } else {
            tile_valid =
                mainloop.load_with_loop_k(params.mainloop, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v, shared_storage, block_coord, tile_valid);
          }

          // Wait for the MMA warpgroups to say that smem_q and smem_do are ready
          if (tile_valid) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, /*id=*/static_cast<uint32_t>(BwdNamedBarriers::QdOEmpty));
          }

          scheduler_prefetch();
        }
        mainloop.load_tail_with_loop_k(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
      } else if (warp_idx_in_warpgroup == 1) { // store partial dKV
        // For each work tile job:
        //  1. atomic reduce-add the computed partial dK,dV from shared memory into global memory
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_tile_info.is_valid(params.scheduler);
             work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
          // get block_coord without deterministic message
          auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
          auto block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), get<2>(block_coord_));
          auto [m_block, bidh, bidb_idx] = block_coord;

          if constexpr (RangeMerge) {
            static_assert(!RangeMerge, "RangeMerge mode is not supported yet when SwapBwdQKLoop is true.");
          } else {
            if constexpr (!Deterministic) {
              mainloop.store_dkv(params.mainloop, shared_storage, block_coord);
            } else {
              static_assert(!Deterministic, "Deterministic mode is not supported yet when SwapBwdQKLoop is true.");
            }
          }
        }
      }
    } else { // Consumer
      // Allocate the registers for the consumer WGs
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      // Initialize tiled mma object for dQ=dSK
      TiledMmadQ tiled_mma_dQ;

      // Initialize consumer read pipeline states of K,V
      PipelineState smem_pipe_read_k;
      PipelineState smem_pipe_read_v;

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
      for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_tile_info.is_valid(params.scheduler);
           work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
        // Get block_coord without deterministic message
        auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
        auto block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), get<2>(block_coord_));

        // Init the zero-initialized register accumulator for dQ
        Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
        clear(tdQrdQ);

        // Run the mma to compute partial dQ,dK,dV
        bool tile_valid = false;
        if constexpr (RangeMerge) {
          static_assert(!RangeMerge, "RangeMerge mode is not supported yet when SwapBwdQKLoop is true.");
        } else {
          tile_valid = mainloop.mma_with_loop_k(
              params.mainloop,
              pipeline_k,
              pipeline_v,
              smem_pipe_read_k,
              smem_pipe_read_v,
              tdQrdQ,
              threadIdx.x - NumCopyThreads,
              work_idx,
              block_coord,
              shared_storage,
              tile_valid);
        }

        // Run the epilogue to store reduced dQ (scaled)
        // NOTE: dK is scaled inside mma_with_loop_k before atomic reduce-add
        if (tile_valid) {
#pragma unroll
          for (int i = 0; i < size(tdQrdQ); ++i) {
            tdQrdQ(i) *= params.mainloop.softmax_scale;
          }
          ++work_idx;
          if constexpr (!Deterministic) {
            epilogue.store_dq(params.epilogue, tdQrdQ, shared_storage, tiled_mma_dQ, threadIdx.x - NumCopyThreads, block_coord);
          } else {
            static_assert(!Deterministic, "Deterministic mode is not supported yet when SwapBwdQKLoop is true.");
          }
          cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, /*id=*/static_cast<uint32_t>(BwdNamedBarriers::QdOEmpty));
        } else {
          if constexpr (!Deterministic) {
            epilogue.store_zero_dq(params.epilogue, threadIdx.x - NumCopyThreads, block_coord);
          } else {
            static_assert(!Deterministic, "Deterministic mode is not supported yet when SwapBwdQKLoop is true.");
          }
        }
      }
      epilogue.store_tail();
    }
  }
};

} // namespace flash
