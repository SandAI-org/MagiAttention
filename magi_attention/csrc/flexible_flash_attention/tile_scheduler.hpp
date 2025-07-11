/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

// Host side kernel arguments
struct TileSchedulerArguments {
    int const num_heads;
    int const num_batches;
    int* const tile_count_semaphore = nullptr;
    int* const ranges = nullptr;
    int* const merge_ranges = nullptr;
    int* const range_map = nullptr;
};

///////////////////////////////////////////////////////////////////////////////

template<int kBlock, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp, bool WarpSpecialized=true>
class DynamicPersistentTileScheduler { 
    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

public:
    using SharedStorage = int4;

protected:
    SharedStorage* const work_info_smem;

public:

    // Device side kernel params
    struct Params {
        int num_heads;
        int num_batches;
        int* const tile_count_semaphore;
        int* const ranges;
        int* const merge_ranges;
        int* const range_map;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.tile_count_semaphore != nullptr);
        assert(args.num_heads < (1 << 16));
        int * const ranges = args.merge_ranges ? args.merge_ranges : args.ranges;
        return {args.num_heads, args.num_batches,
                args.tile_count_semaphore, ranges,
                args.merge_ranges, args.range_map};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            // if (blockIdx.x >= 0 && (threadIdx.x == 128 || threadIdx.x == 0)) { printf("blockIdx.x = %d, threadIdx.x = %d, checking valid, bidb = %d, params.num_batches = %d\n", blockIdx.x, threadIdx.x, bidb, params.num_batches); }
            return bidb < params.num_batches;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block, bidh, bidb};
        }
    };

    CUTLASS_DEVICE
    DynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
        int lane = threadIdx.x % cutlass::NumThreadsPerWarp;

        // Helper function to calculate how many blocks are needed to compute the current batch
        auto get_num_m_blocks = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            int seqlen = batch_idx < params.num_batches ? params.ranges[2 * batch_idx + 1] - params.ranges[2 * batch_idx] : 0;
            return batch_idx < params.num_batches && lane < cutlass::NumThreadsPerWarp - 1 ? cute::ceil_div(seqlen, kBlock) : 0;
        };

        int num_m_blocks = get_num_m_blocks(current_work.bidb);  // Different for each lane
        // Cumulative number of blocks for the next 31 batches
        int num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks);
        // Total number of blocks for the next 31 batches
        int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
        // Only the lower 16 bits are the actual bidh
        int current_bidh = current_work.bidh;
        int group_end_tile = current_work.tile_idx - current_work.block - current_bidh * __shfl_sync(0xffffffff, num_m_blocks, 0 /*lane*/) + m_blocks_in_group * params.num_heads;  // Same for all lanes
        int bidb = current_work.bidb;
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Before while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, cur tile_idx = %d, cur block = %d, cur bidh = %d, num_m_blocks = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, current_work.bidb, num_m_blocks, next_tile_idx, current_work.tile_idx, current_work.block, current_bidh, num_m_blocks, group_end_tile, m_blocks_in_group);
        // }
        // if (threadIdx.x == 0 && blockIdx.x == 0) { printf("tile_idx = %d, group_end_tile = %d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d\n", current_work.tile_idx, group_end_tile, num_m_blocks_cumulative, m_blocks_in_group); }
        while (group_end_tile <= next_tile_idx) {
            bidb += cutlass::NumThreadsPerWarp - 1;
            if (bidb >= params.num_batches) {
                // if (blockIdx.x <= 9 && threadIdx.x == 0) {
                //     printf("Returning early, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
                // }
                return {next_tile_idx, 0, 0, params.num_batches};
            }
            num_m_blocks = get_num_m_blocks(bidb);
            num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks);
            m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
            group_end_tile += m_blocks_in_group * params.num_heads;
            // if (blockIdx.x <= 9 && threadIdx.x == 0) {
            //     printf("Bottom of while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
            // }
        }
        int group_start_tile = group_end_tile - m_blocks_in_group * params.num_heads;
        // The next problem to process is the first one that does not have ending tile position
        // that is greater than or equal to tile index.
        int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * params.num_heads <= next_tile_idx));
        // if (threadIdx.x == 31 || threadIdx.x == 0) { printf("blockIdx.x = %d, tidx %d, group_start_tile = %d, num_m_blocks_cumulative = %d, num_heads = %d, next_tile_idx = %d, ballot = %x, batch_idx_in_group = %d\n", blockIdx.x, threadIdx.x, group_start_tile, num_m_blocks_cumulative, params.num_heads, next_tile_idx, tmp, batch_idx_in_group); }
        bidb += batch_idx_in_group;
        num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
        int mh_block = next_tile_idx - group_start_tile - (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) * params.num_heads;
        int bidh = mh_block / num_m_blocks;
        int block = mh_block - bidh * num_m_blocks;
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Before returning, blockIdx.x = %d, threadIdx.x = %d, group_start_tile = %d, batch_idx_in_group = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d, mh_block = %d, bidh = %d, block = %d\n", blockIdx.x, threadIdx.x, group_start_tile, batch_idx_in_group, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group, mh_block, bidh, block);
        // }
        return {next_tile_idx, block, bidh, bidb};
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        // Don't arrive at the TileCountSmemEmpty barrier here, because get_initial_work will do that
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 has the next tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb};
            work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int4 work_info = *work_info_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
        }
    }

};

} // flash
