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

#include <thrust/pair.h>

#include "cute/tensor.hpp"

#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cutlass/arch/grid_dependency_control.h>

#include "fwd_tile_scheduler.hpp"
#include "inner_ldst_mode.hpp"
#include "mask.h"
#include "softmax.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_, bool RangeMerge_, bool InnerDirMaxToMin_, int ProducerRegs_, int ConsumerRegs_>
class FlashAttnFwdSm90 {
 public:
  // Type Aliases
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  static constexpr bool RangeMerge = RangeMerge_;
  static constexpr bool Has_softcap = CollectiveMainloop::Has_softcap;
  static constexpr bool Use_TMA_Q = CollectiveMainloop::Use_TMA_Q;
  static constexpr InnerLoadMode kInnerLoadMode = CollectiveMainloop::kInnerLoadMode;

  // KV pipelines come in two flavors with different constructor signatures: dense TMA
  // (PipelineTmaAsync, takes ClusterShape) and scatter cp.async (PipelineAsync, no
  // cluster argument). This helper hides the difference at every construction site.
  template <typename Pipeline, typename Storage, typename PipelineParamsT>
  CUTLASS_DEVICE static Pipeline make_kv_pipeline(Storage& storage, PipelineParamsT const& pipeline_params) {
    if constexpr (kInnerLoadMode == InnerLoadMode::Tma) {
      return Pipeline(storage, pipeline_params, ClusterShape{});
    } else {
      return Pipeline(storage, pipeline_params);
    }
  }
  static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
  static constexpr int NumConsumerThreads = CollectiveMainloop::NumConsumerThreads;
  static constexpr bool Deterministic = CollectiveEpilogue::Deterministic;
  static constexpr bool PackGQA = CollectiveMainloop::PackGQA;
  static constexpr bool SwapAB = CollectiveMainloop::SwapAB;
  static constexpr bool BlockSparse = CollectiveMainloop::BlockSparse;
  static constexpr bool IndexSparse = CollectiveMainloop::IndexSparse;
  static constexpr bool ReturnMaxLogits = CollectiveEpilogue::ReturnMaxLogits;
  static constexpr int NumMaxLogits = CollectiveEpilogue::NumMaxLogits;

  // Mainloop derived types
  // using BlockMeta = typename CollectiveMainloop::BlockMeta;
  using TileShape_MNK_PV = typename CollectiveMainloop::TileShape_MNK_PV;
  using TileShape_MNK_PV_Active = typename CollectiveMainloop::TileShape_MNK_PV_Active;
  using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV_Active;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ClusterShape = typename CollectiveMainloop::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using BarrierQ = std::conditional_t<Use_TMA_Q, cutlass::arch::ClusterTransactionBarrier, cutlass::arch::ClusterBarrier>;

  // Epilogue derived types
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using BlockCoordType = typename CollectiveEpilogue::BlockCoordType;

  // Sanity check
  static_assert(ArchTag::kMinComputeCapability >= 90);

  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename flash::TileSchedulerArguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr auto kInnerDir = InnerDirMaxToMin_ ? flash::DispatchDirection::MaxToMin : flash::DispatchDirection::MinToMax;
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumConsumerWarpGroups = CUTE_STATIC_V(size(TiledMmaPV{})) / cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaPV{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static_assert(NumConsumerWarpGroups == 1 || NumConsumerWarpGroups == 2 || NumConsumerWarpGroups == 3);

  static_assert(ProducerRegs_ % 8 == 0 && ProducerRegs_ >= 24 && ProducerRegs_ <= 256);
  static_assert(ConsumerRegs_ % 8 == 0 && ConsumerRegs_ >= 24 && ConsumerRegs_ <= 256);

  // Kernel level shared memory storage
  // We overlap the shared memory for the mainloop and epilogue.
  // However, we only want smem_o to overlap with smem_v and nothing else,
  // so we'll pad in case sizeof(smem_o) > sizeof(smem_v).
  static constexpr int mainloop_smem_padding_ =
      int(sizeof(typename CollectiveEpilogue::TensorStorage)) - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v)));
  static constexpr int mainloop_smem_padding = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _1> {
      cute::array_aligned<float, ReturnMaxLogits ? NumMaxLogits : 0> smem_max_logits;
      union {
        struct {
          cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
          typename CollectiveMainloop::TensorStorage mainloop;
        };
        // We want smem_o to line up with the start of smem_v
        typename CollectiveEpilogue::TensorStorage epilogue;
      };
    } tensors;
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      alignas(16) BarrierQ barrier_Q;
      alignas(16) cutlass::arch::ClusterBarrier barrier_O;
      alignas(16) typename CollectiveMainloop::MainloopPipelineK::SharedStorage pipeline_k;
      alignas(16) typename CollectiveMainloop::MainloopPipelineV::SharedStorage pipeline_v;
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
    static constexpr int MmaThreadOffset = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
    static constexpr int kBlockN = get<2>(TileShape_MNK_PV{});

    using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
    using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
    using PipelineState = typename CollectiveMainloop::PipelineState;
    using PipelineParamsK = typename MainloopPipelineK::Params;
    using PipelineParamsV = typename MainloopPipelineV::Params;

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
    int warp_group_idx = cutlass::canonical_warp_group_idx();

    // Initialize the barriers of Q,O
    if (warp_idx == 0 && lane_predicate) {
      shared_storage.pipelines.barrier_Q.init(/*numThreads=*/Use_TMA_Q ? 1 : NumProducerThreads);
      // TODO: Fix if TMA store O is used
      shared_storage.pipelines.barrier_O.init(size(ClusterShape{}) * NumConsumerThreads);
    }

    if constexpr (ReturnMaxLogits) {
      for (int i = threadIdx.x; i < NumMaxLogits; i += MaxThreadsPerBlock) {
        shared_storage.tensors.smem_max_logits[i] = -INFINITY;
      }
    }

    // Initialize pipelines of K,V
    // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    PipelineParamsK pipeline_params_k;
    pipeline_params_k.role = warp_group_idx == 0 ? MainloopPipelineK::ThreadCategory::Producer : MainloopPipelineK::ThreadCategory::Consumer;
    if constexpr (kInnerLoadMode == InnerLoadMode::Tma) {
      pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
      pipeline_params_k.is_leader = warp_group_thread_idx == 0;
      pipeline_params_k.num_consumers = NumConsumerThreads;
    } else {
      pipeline_params_k.consumer_arv_count = NumConsumerThreads;
      pipeline_params_k.producer_arv_count = NumProducerThreads;
    }

    static_assert(is_same_v<PipelineParamsK, PipelineParamsV>);
    PipelineParamsV pipeline_params_v = pipeline_params_k; // K,V share the same pipeline params

    MainloopPipelineK pipeline_k = make_kv_pipeline<MainloopPipelineK>(shared_storage.pipelines.pipeline_k, pipeline_params_k);
    MainloopPipelineV pipeline_v = make_kv_pipeline<MainloopPipelineV>(shared_storage.pipelines.pipeline_v, pipeline_params_v);

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    // We need this to guarantee that pipeline initialization is visible to
    // all producers and consumer blocks in the cluster
    sync_cga_threads<ClusterShape>();

    TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

    if (warp_group_idx == 0) { // Producer
      using BlockMetaT = std::conditional_t<
          BlockSparse,
          typename CollectiveMainloop::BlockSparseProducerBlockMeta,
          std::conditional_t<
              IndexSparse,
              typename CollectiveMainloop::template IndexSparseBlockMeta</*IsProducer=*/true>,
              typename CollectiveMainloop::BlockMeta</*IsProducer=*/true>>>;

      // Deallocate the registers for the producer WG,
      // which allows the consumer WGs to have more registers
      cutlass::arch::warpgroup_reg_dealloc<ProducerRegs_>();

      // Initialize producer write pipeline states of K,V
      PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipelineK>();
      PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipelineV>();

      // Initialize the work index
      int work_idx = 0;

      // Get some block-level information
      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
      int thread_idx = threadIdx.x % NumProducerThreads;

      static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;

      // TMA 2D paths: SingleProducerWarp=true → warps 1-3 exit.
      // Scatter paths (TMA 1D / cp.async): full warp group needed for per-row loads.
      if constexpr (SingleProducerWarp) {
        if (warp_idx_in_warpgroup != 0) {
          return;
        }
      }

      // REVIEW: when should non-first warps be considered as consumers ?
      if (!SingleProducerWarp && warp_idx_in_warpgroup != 0) {
        scheduler.init_consumer();
      }

      // BlockSparse: only warp 0 does scheduling (atomicAdd); Dense/IndexSparse: SingleProducerWarp or warp 0
      auto is_scheduler_warp = [&]() {
        if constexpr (BlockSparse) {
          return warp_idx_in_warpgroup == 0;
        } else {
          return SingleProducerWarp || warp_idx_in_warpgroup == 0;
        }
      };

      // For each work tile job:
      // 1. load this m block of Q from global memory into shared memory
      // 2. pipeline the loads of K,V for each n block from global memory into shared memory
      for (auto work_block_info = is_scheduler_warp() ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler)
                                                      : scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
           work_block_info.is_valid(params.scheduler);
           work_block_info = is_scheduler_warp() ? scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_block_info)
                                                 : scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_block_info)) {
        auto block_coord = work_block_info.get_block_coord();

        BlockMetaT block_meta{params.mainloop, block_coord, shared_storage, thread_idx};

        auto scheduler_prefetch = [&scheduler, &params, &work_block_info]() { scheduler.prefetch_next_work(params.scheduler, work_block_info); };

        bool has_tile_valid = mainloop.template load<kInnerDir>(
            params.mainloop, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v, shared_storage, scheduler_prefetch, block_meta, work_idx, thread_idx);

        scheduler_prefetch();
        if (has_tile_valid) {
          ++work_idx;
        }
      }
      mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v, shared_storage, work_idx);
    } else { // Consumer
      using BlockMetaT = std::conditional_t<
          BlockSparse,
          typename CollectiveMainloop::BlockSparseConsumerBlockMeta,
          std::conditional_t<
              IndexSparse,
              typename CollectiveMainloop::template IndexSparseBlockMeta</*IsProducer=*/false>,
              typename CollectiveMainloop::BlockMeta</*IsProducer=*/false>>>;

      // Allocate the registers for the consumer WGs
      cutlass::arch::warpgroup_reg_alloc<ConsumerRegs_>();

      // Initialize tiled mma object for O=PV
      TiledMmaPV tiled_mma_pv;

      // Initialize consumer read pipeline states of K,V
      // NOTE: we don't need separate variables smem_pipe_release_k and
      // smem_pipe_release_v (like in Cutlass's gemm) because the read and
      // release pipeline states are always the same.
      PipelineState smem_pipe_read_k;
      PipelineState smem_pipe_read_v;

      // Initialize mma consumers
      scheduler.init_consumer();
      mainloop.mma_init();

      // For each work tile job:
      //  1. run mma consumer to compute partial O as the consumer prologue/mainloop
      //  2. accumulate partial O into the zero-initialized register fragments
      //  3. store the reduced O into the global memory as the consumer epilogue
      int work_idx = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (auto work_block_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler); work_block_info.is_valid(params.scheduler);
           // get_next_work will be called before the epilogue
      ) {
        auto block_coord = work_block_info.get_block_coord();
        auto det_msg = work_block_info.get_det_msg();

        BlockMetaT block_meta = BlockMetaT{params.mainloop, block_coord, shared_storage};

        auto epilogue_block_coord = block_meta.get_epilogue_coord();

        // Init softmax object
        float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
        flash::Softmax<
            !SwapAB ? 2 * (2 * kBlockM / NumConsumerThreads) : 32 * kBlockM / NumConsumerThreads,
            /*Max_offset=*/0,
            /*SwapAB=*/SwapAB>
            softmax(softmax_scale_log2);
        typename flash::Softmax<
            !SwapAB ? 2 * (2 * kBlockM / NumConsumerThreads) : 32 * kBlockM / NumConsumerThreads,
            /*Max_offset=*/0,
            /*SwapAB=*/SwapAB>::TensorT scores_scale;

        // Init the zero-initialized register SMEM buffer for O
        Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV_Active{}));
        clear(tOrO);

        // Run the mma to compute partial O
        bool has_tile_valid = mainloop.template mma<kInnerDir>(
            params.mainloop,
            pipeline_k,
            pipeline_v,
            smem_pipe_read_k,
            smem_pipe_read_v,
            tOrO,
            softmax,
            scores_scale,
            threadIdx.x - MmaThreadOffset,
            work_idx,
            block_meta,
            shared_storage);

        // NOTE: get next work before epilogue so that the next tile is ready to go.
        work_block_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_block_info);

        if (has_tile_valid) {
          if constexpr (!ReturnMaxLogits) {
            epilogue.store(
                params.epilogue,
                tOrO,
                softmax.row_sum,
                shared_storage,
                tiled_mma_pv,
                threadIdx.x - MmaThreadOffset,
                epilogue_block_coord,
                block_meta.seqlen_info,
                det_msg);
          } else {
            epilogue.store(
                params.epilogue,
                tOrO,
                softmax.row_sum,
                shared_storage,
                tiled_mma_pv,
                threadIdx.x - MmaThreadOffset,
                epilogue_block_coord,
                block_meta.seqlen_info,
                det_msg,
                softmax.row_max);
          }
        } else {
          epilogue.store_zero(params.epilogue, threadIdx.x - MmaThreadOffset, epilogue_block_coord, block_meta.seqlen_info, det_msg);
        }
      }
      // barrier_O guards smem_o/smem_v union: producer waits before V load,
      // consumer arrives after O store. work_idx counts *valid* tiles (incremented
      // only when epilogue.store() runs, which arrives barrier_O internally).
      // Invalid tiles go through store_zero() which intentionally does NOT arrive
      // barrier_O (the producer never loaded V for invalid tiles either).
      // If the entire loop had zero valid tiles, nobody arrived → producer's
      // load_tail would deadlock in barrier_O.wait(0). Manually arrive here.
      if (work_idx == 0) {
#pragma unroll
        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
          shared_storage.pipelines.barrier_O.arrive(cta_id);
        }
      }
      // epilogue tail only contains ReturnMaxLogits logic so we skip it if not needed
      if constexpr (ReturnMaxLogits) {
        epilogue.store_tail(params.epilogue, shared_storage, threadIdx.x - MmaThreadOffset);
      }
    }
  }
};

} // namespace flash
