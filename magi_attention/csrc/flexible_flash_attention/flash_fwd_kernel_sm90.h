/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cutlass/arch/grid_dependency_control.h"

#include "seqlen.h"
#include "utils.h"
#include "softmax.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_, bool MergeRange_>
class FlashAttnFwdSm90 {

public:

    // Type Aliases
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    static constexpr bool MergeRange = MergeRange_;
    static constexpr bool Has_softcap = CollectiveMainloop::Has_softcap;
    static constexpr bool Use_TMA_Q = CollectiveMainloop::Use_TMA_Q;
    static constexpr bool Use_TMA_KV = CollectiveMainloop::Use_TMA_KV;
    static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
    static constexpr int NumMmaThreadsQK = CollectiveMainloop::NumMmaThreadsQK;
    
    using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;

    // Mainloop derived types
    using TileShape_MNK_PV = typename CollectiveMainloop::TileShape_MNK_PV;
    using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using ClusterShape = typename CollectiveMainloop::ClusterShape;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;
    using BarrierQ = std::conditional_t<Use_TMA_Q, cutlass::arch::ClusterTransactionBarrier, cutlass::arch::ClusterBarrier>;

    // Epilogue derived types
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams = typename TileScheduler::Params;

    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMmaPV{})) / cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaPV{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    /// Register requirement for Load and Math WGs
    // If we use cp.async to load K and V, we need more registers for the producer WG.
    static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 1 ? 56 : (NumMmaWarpGroups == 2 ? (Use_TMA_KV ? 24 : 40) : 32);
    static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 1 ? 256 : (NumMmaWarpGroups == 2 ? (Use_TMA_KV ? 240 : 232) : 160);
    // If you want to print from the producer warp, you'd need to increase the number of registers
    // Otherwise you'll get CUDA error.
    // static constexpr uint32_t LoadRegisterRequirement = 40;
    // static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 232 : 152;

    // Kernel level shared memory storage
    // We overlap the shared memory for the mainloop and epilogue. However, we only want smem_o to overlap with smem_v
    // and nothing else, so we'll pad in case sizeof(smem_o) > sizeof(smem_v).
    static constexpr int mainloop_smem_padding_ = int(sizeof(typename CollectiveEpilogue::TensorStorage)) - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v)));
    static constexpr int mainloop_smem_padding = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128, _1> {
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

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");

        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            hw_info,
            TileScheduler::to_underlying_arguments(args.scheduler)
        };
    }

    // Computes the kernel launch grid shape based on runtime parameters
    static dim3
    get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
    }

    static dim3
    get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {

        static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
        // The offset of the first thread of the first mma warp group
        static constexpr int MmaThreadOffset = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});

        using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
        using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
        using PipelineState = typename CollectiveMainloop::PipelineState;
        using PipelineParamsK = typename MainloopPipelineK::Params;
        using PipelineParamsV = typename MainloopPipelineV::Params;

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int const lane_predicate = cute::elect_one_sync();
        int const warp_idx = cutlass::canonical_warp_idx_sync();

        // Issue Tma Descriptor Prefetch from a single thread (the first thread of the first warp)
        if (warp_idx == 0 && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
            CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
        }

        // Get thread index in warp group
        int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
        // Get warp group index
        int warp_group_idx = cutlass::canonical_warp_group_idx();

        // Initialize the barriers
        if (warp_idx == 0 && lane_predicate) {
            shared_storage.pipelines.barrier_Q.init(Use_TMA_Q ? 1 : NumProducerThreads /*numThreads*/);
            // TODO: Fix if TMA store O is used
            shared_storage.pipelines.barrier_O.init(size(ClusterShape{}) * NumMmaThreads);
        }

        // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
        PipelineParamsK pipeline_params_k;
        pipeline_params_k.role = warp_group_idx == 0
            ? MainloopPipelineK::ThreadCategory::Producer
            : MainloopPipelineK::ThreadCategory::Consumer;
        if constexpr (Use_TMA_KV) {
            pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
            pipeline_params_k.is_leader = warp_group_thread_idx == 0;
            pipeline_params_k.num_consumers = NumMmaThreads;
        } else {
            pipeline_params_k.consumer_arv_count = NumMmaThreads;
            pipeline_params_k.producer_arv_count = NumProducerThreads;
        }

        static_assert(is_same_v<PipelineParamsK, PipelineParamsV>);
        PipelineParamsV pipeline_params_v = pipeline_params_k;

        MainloopPipelineK pipeline_k = [&] {
            if constexpr (Use_TMA_KV) {
                return MainloopPipelineK(shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{});
            } else {
                return MainloopPipelineK(shared_storage.pipelines.pipeline_k, pipeline_params_k);
            }
        }();

        // MainloopPipelineV pipeline_v(shared_storage.pipelines.pipeline_v, pipeline_params_v, ClusterShape{});
        MainloopPipelineV pipeline_v = [&] {
            if constexpr (Use_TMA_KV) {
                return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_v, ClusterShape{});
            } else {
                return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_v);
            }
        }();

        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;

        // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
        if constexpr (size(ClusterShape{}) > 1) {
            cute::cluster_arrive_relaxed();
            cute::cluster_wait();
        } else {
            __syncthreads();
        }

        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

        if (warp_group_idx == 0) {  // Producer
            // Deallocate the registers for the producer WG, this makes the consumer WG have more registers
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

            // Initialize the producer pipeline state
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();

            // Initialize the work index
            int work_idx = 0;

            // Get some block-level information
            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            // Currently, SingleProducerWarp is always true
            static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;


            // Only the first warp in the warp group needs to issue the TMA load instruction
            if constexpr (SingleProducerWarp) {
                if (warp_idx_in_warpgroup != 0) { return; }
            }

            // wtfff?
            if (!SingleProducerWarp && warp_idx_in_warpgroup != 0) { scheduler.init_consumer(); }

            // cutlass::arch::wait_on_dependent_grids();

            // Load Q, K, V
            for (auto work_tile_info = SingleProducerWarp || warp_idx_in_warpgroup == 0 ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler) : scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 work_tile_info = SingleProducerWarp || warp_idx_in_warpgroup == 0 ? scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info) : scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
                

                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
                    scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                };

                bool tile_valid = false;

                // TODO: move it to compile time
                if constexpr (MergeRange) {
                    int bidb_idx = get<2>(block_coord);
                    int loop_count = bidb_idx < *params.scheduler.unique_count - 1 ? params.scheduler.range_map[bidb_idx + 1] - params.scheduler.range_map[bidb_idx] : params.scheduler.num_batch - params.scheduler.range_map[bidb_idx];
                    int bidb_start = params.scheduler.range_map[bidb_idx];

                    for (int idx = 0; idx < loop_count; ++idx) {
                        int bidb = bidb_start + idx;
                        block_coord = cute::make_tuple(get<0>(block_coord), get<1>(block_coord), bidb);
                        SeqlenInfo_t seqlen_info{
                            bidb,
                            params.mainloop.q_ranges, 
                            params.mainloop.k_ranges,
                        };
                        bool current_tile_valid = mainloop.load(params.mainloop, pipeline_k, pipeline_v, smem_pipe_write,
                                    shared_storage, scheduler_prefetch, seqlen_info, block_coord, work_idx, tile_valid);

                        tile_valid = tile_valid || current_tile_valid;
                    }
                }
                else {
                    SeqlenInfo_t seqlen_info{
                        get<2>(block_coord),
                        params.mainloop.q_ranges, 
                        params.mainloop.k_ranges,
                    };
                    tile_valid = mainloop.load(params.mainloop, pipeline_k, pipeline_v, smem_pipe_write,
                                  shared_storage, scheduler_prefetch, seqlen_info, block_coord, work_idx, tile_valid);
                }

                scheduler_prefetch();
                if (tile_valid) {
                    ++work_idx;
                }
            }
            mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write, shared_storage, work_idx);
        } else {  // Consumer
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

            // Initialize matmul objects.
            TiledMmaPV tiled_mma_pv;

            PipelineState smem_pipe_read;
            // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
            // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

            scheduler.init_consumer();
            mainloop.mma_init();

            int work_idx = 0;

            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                work_tile_info.is_valid(params.scheduler);
                // get_next_work will be called before the epilogue
                ) {

                // If there's tanh softcap, the scaling will be done before tanh.
                float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
                flash::Softmax<2 * (2 * kBlockM / NumMmaThreads), /*Max_offset=*/ 0> softmax(softmax_scale_log2);
                typename flash::Softmax<2 * (2 * kBlockM / NumMmaThreads), /*Max_offset=*/ 0>::TensorT scores_scale;
                // Attention output (GEMM-II) accumulator.
                Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));
                clear(tOrO);
                bool tile_valid = false;
                auto block_coord = work_tile_info.get_block_coord(params.scheduler);

                if constexpr (MergeRange) {
                    int bidb_idx = get<2>(block_coord);
                    int loop_count = bidb_idx < *params.scheduler.unique_count - 1 ? params.scheduler.range_map[bidb_idx + 1] - params.scheduler.range_map[bidb_idx] : params.scheduler.num_batch - params.scheduler.range_map[bidb_idx];
                    int bidb_start = params.scheduler.range_map[bidb_idx];

                    for (int idx = 0; idx < loop_count; ++idx) {

                        int bidb = bidb_start + idx;
                        block_coord = cute::make_tuple(get<0>(block_coord), get<1>(block_coord), bidb);
                        SeqlenInfo_t seqlen_info{
                            bidb,
                            params.mainloop.q_ranges, 
                            params.mainloop.k_ranges,
                        };
                        bool current_tile_valid = mainloop.mma(
                            params.mainloop, pipeline_k, pipeline_v, smem_pipe_read,
                            tOrO, softmax, scores_scale, threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage, 
                            tile_valid
                        );

                        tile_valid = tile_valid || current_tile_valid;
                    }
                }
                else {
                    SeqlenInfo_t seqlen_info{
                        get<2>(block_coord),
                        params.mainloop.q_ranges, 
                        params.mainloop.k_ranges,
                    };
                    tile_valid = mainloop.mma(
                        params.mainloop, pipeline_k, pipeline_v, smem_pipe_read,
                        tOrO, softmax, scores_scale, threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage,
                        tile_valid
                    );
                }

                // Do this here before the epilogue so that the next tile is ready to go.
                work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info);
                if (tile_valid) {
                    ++work_idx;
            
                    cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);

                    // Get the final scores_scale
                    cute::copy(softmax.finalize(), scores_scale);
                    // Rescale tOrO
                    softmax.rescale_o(tOrO, scores_scale);

                    // if (threadIdx.x == 128) { printf("Before epilogue, bid.x = %d, bid.y = %d, bid.z = %d, m_block = %d, bidb = %d, split_idx = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, m_block, bidb, split_idx); }
                    epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
                                   threadIdx.x - MmaThreadOffset, block_coord);
                } else {
                    // Write 0 to gO and -inf to gLSE.
                    epilogue.store_zero(params.epilogue, threadIdx.x - MmaThreadOffset, block_coord);
                }
            }
            epilogue.store_tail();
        }
    }
};

} // namespace flash
