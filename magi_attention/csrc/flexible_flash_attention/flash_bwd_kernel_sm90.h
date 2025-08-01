/**********************************************************************************
 * Copyright (c) 2025 SandAI. All Rights Reserved.
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
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ClusterShape = typename CollectiveMainloop::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static constexpr bool dKV_swapAB = CollectiveMainloop::dKV_swapAB;

  static constexpr bool RangeMerge = RangeMerge_;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static constexpr bool Deterministic = CollectiveEpilogue::Deterministic;

  static_assert(ArchTag::kMinComputeCapability >= 90);

  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename flash::TileSchedulerArguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMmaSdP{})) / cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaSdP{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

  /// Register requirement for Load and Math WGs
  static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 2 ? 24 : 32;
  static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 240 : 160;
  // If you want to print from the producer warp, you'd need to increase the
  // number of registers Otherwise you'll get CUDA error. static constexpr
  // uint32_t LoadRegisterRequirement = 40; static constexpr uint32_t
  // MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 232 : 152;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128> {
      union {
        typename CollectiveMainloop::TensorStorage mainloop;
        typename CollectiveEpilogue::TensorStorage epilogue;
      };
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16> {
      alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_KV;
      alignas(16) typename CollectiveMainloop::MainloopPipeline::SharedStorage pipeline_q;
      alignas(16) typename CollectiveMainloop::MainloopPipeline_dO::SharedStorage pipeline_do;
      alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
    } pipelines;
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

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
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

    // Obtain warp index
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ + CollectiveMainloop::TmaTransactionBytesLSE;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
      shared_storage.pipelines.barrier_KV.init(1 /*numThreads*/);
    }
    // We're counting on pipeline_q to call cutlass::arch::fence_barrier_init();
    MainloopPipeline pipeline_q(shared_storage.pipelines.pipeline_q, pipeline_params, ClusterShape{});
    auto role_dO = warp_group_idx == 0 ? MainloopPipeline_dO::ThreadCategory::Producer : MainloopPipeline_dO::ThreadCategory::Consumer;
    PipelineParams_dO pipeline_params_dO{pipeline_params.transaction_bytes, role_dO, pipeline_params.is_leader, pipeline_params.num_consumers};
    MainloopPipeline_dO pipeline_do(
        shared_storage.pipelines.pipeline_do, cute::conditional_return<Q_dO_same_stages>(pipeline_params, pipeline_params_dO), ClusterShape{});

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    // We need this to guarantee that the Pipeline init is visible to all
    // producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
      cute::cluster_arrive_relaxed();
      cute::cluster_wait();
    } else {
      __syncthreads();
    }

    TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

    if (warp_group_idx == 0) { // Producer
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
      if (warp_idx_in_warpgroup == 0) { // Load K, V, and do TMA on Q and dO
        PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState_dO smem_pipe_write_do = cutlass::make_producer_start_state<MainloopPipeline_dO>();

        // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);

        for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler); work_tile_info.is_valid(params.scheduler);
             work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info)) {
          auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
          // get block_coord without deterministic message
          auto block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), get<2>(block_coord_));
          auto [n_block, bidh, bidb_idx] = block_coord;

          auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() { scheduler.prefetch_next_work(params.scheduler, work_tile_info); };

          bool tile_valid = false;

          if constexpr (RangeMerge) {
            int loop_count = (bidb_idx < *params.scheduler.unique_count - 1) ? (params.scheduler.range_map[bidb_idx + 1] - params.scheduler.range_map[bidb_idx])
                                                                             : (params.scheduler.num_batches - params.scheduler.range_map[bidb_idx]);
            int bidb_start = params.scheduler.range_map[bidb_idx];

            for (int idx = 0; idx < loop_count; ++idx) {
              int bidb = bidb_start + idx;
              block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), bidb);
              bool tile_valid_tmp =
                  mainloop.load(params.mainloop, pipeline_q, pipeline_do, smem_pipe_write, smem_pipe_write_do, shared_storage, block_coord, tile_valid);

              tile_valid = tile_valid || tile_valid_tmp;
            }
          } else {
            tile_valid = mainloop.load(params.mainloop, pipeline_q, pipeline_do, smem_pipe_write, smem_pipe_write_do, shared_storage, block_coord, tile_valid);
          }

          if (tile_valid) {
            // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);
          }

          scheduler_prefetch();
        }
        mainloop.load_tail(pipeline_q, pipeline_do, smem_pipe_write, smem_pipe_write_do);
      } else if (warp_idx_in_warpgroup == 1) {
        int bidb_last = 0;
        for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_tile_info.is_valid(params.scheduler);
             work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
          auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
          // get block_coord without deterministic message
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
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
      // Initialize matmul objects.
      TiledMmadKV tiled_mma_dKV;

      PipelineState smem_pipe_read;
      PipelineState_dO smem_pipe_read_do;

      mainloop.mma_init();
      scheduler.init_consumer();

      int work_idx = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_tile_info.is_valid(params.scheduler);
           work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
        auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
        // get block_coord without deterministic message
        auto block_coord = cute::make_tuple(get<0>(block_coord_), get<1>(block_coord_), get<2>(block_coord_));

        Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, select < !dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1 > (TileShape_MNK{}));
        Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, select < !dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1 > (TileShape_MNK{}));
        clear(tdKrdK);
        clear(tdVrdV);
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
            bool tile_valid_tmp = mainloop.mma(
                params.mainloop,
                pipeline_q,
                pipeline_do,
                smem_pipe_read,
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
          tile_valid = mainloop.mma(
              params.mainloop,
              pipeline_q,
              pipeline_do,
              smem_pipe_read,
              smem_pipe_read_do,
              tdKrdK,
              tdVrdV,
              threadIdx.x - NumCopyThreads,
              work_idx,
              block_coord,
              shared_storage,
              tile_valid);
        }

        if (tile_valid) {
#pragma unroll
          for (int i = 0; i < size(tdKrdK); ++i) {
            tdKrdK(i) *= params.mainloop.softmax_scale;
          }
          ++work_idx;
          if constexpr (!Deterministic) {
            epilogue.store(params.epilogue, tdKrdK, tdVrdV, shared_storage, tiled_mma_dKV, threadIdx.x - NumCopyThreads, block_coord);
          } else {
            epilogue.store(params.epilogue, tdKrdK, tdVrdV, shared_storage, tiled_mma_dKV, threadIdx.x - NumCopyThreads, block_coord_);
          }
          cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);
        } else {
          if constexpr (!Deterministic) {
            epilogue.store_zero(params.epilogue, threadIdx.x - NumCopyThreads, block_coord);
          } else {
            epilogue.store_zero(params.epilogue, threadIdx.x - NumCopyThreads, block_coord_);
          }
        }
      }
      epilogue.store_tail();
    }
  }
};

} // namespace flash
