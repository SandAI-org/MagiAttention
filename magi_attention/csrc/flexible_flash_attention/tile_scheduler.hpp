/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <thrust/pair.h>

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

// Host side kernel arguments
struct TileSchedulerArguments {
    // num_head is num_head_q if not PackGQA, else num_head_k
    int const num_blocks, num_head, num_batch, num_splits;
    int const qhead_per_khead;
    int const seqlen;  // Only used if Varlen and cu_seqlens == nullptr and seqused == nullptr
    int const seqlen_k, headdim, headdim_v, element_size;  // Used to calculate L2 swizzling
    int* const tile_count_semaphore = nullptr;
    int* const cu_seqlens = nullptr;
    int* const ranges = nullptr;
    int* const seqused = nullptr;
    // int* const num_m_blocks_ptr = nullptr;
    int* const num_splits_dynamic_ptr = nullptr;
    int* const merge_ranges = nullptr;
    int* const range_map = nullptr;
    int* determin_conflict_state = nullptr;
};

///////////////////////////////////////////////////////////////////////////////

template<bool Varlen=false, bool Split=false, bool PackGQA=false, int kBlock=128>
class SingleTileScheduler {

public:

    using SharedStorage = int;

    // Device side kernel params
    struct Params {
        // num_blocks is num_blocks_m, where m is the number of blocks in the M dimension (blockIdx.x)
        // num_head is num_head_q if not PackGQA, else qhead_per_khead
        // num_batch is num_batch_q, which means ranges.size(0)
        int const num_blocks, num_head, num_batch, num_splits;
        int const qhead_per_khead;
        int const seqlen;
        cutlass::FastDivmod nsplits_divmod;
        int* const cu_seqlens;
        int* const ranges;
        int* const seqused;
        int const* const num_splits_dynamic_ptr = nullptr;
        int* const merge_ranges = nullptr;
        int* const range_map = nullptr;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(!Split || !Varlen || args.num_splits_dynamic_ptr != nullptr);
        assert(!Split || !Varlen || args.num_splits < (1 << 16)); // We use the top 16 bits to store num_splits
        int * const ranges = args.merge_ranges ? args.merge_ranges : args.ranges;
        return {args.num_blocks, args.num_head, args.num_batch, !Split ? 1 : args.num_splits,
                args.qhead_per_khead, args.seqlen,
                cutlass::FastDivmod(!Split ? 1 : args.num_splits),
                !Varlen ? nullptr : args.cu_seqlens,
                !Varlen ? nullptr : ranges,
                !Varlen ? nullptr : args.seqused,
                args.num_splits_dynamic_ptr,
                args.merge_ranges,
                args.range_map};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        /**
         * NOTE: Here, we shift the batch size to the x-dimension because the z-dimension must be less than 65536.
         *  Thus once the batch size is too large (>= 65536), the z-dimension may overflow, causing an implicit kernel-launch error.
         *  What's worse, this error may not be explicitly raised, resulting in the kernel being skipped.
         */
        // return {uint32_t(params.num_blocks), uint32_t((!Split ? 1 : params.num_splits) * params.num_head), uint32_t(params.num_batch)};
        return {uint32_t(params.num_batch), uint32_t(params.num_blocks), uint32_t((!Split ? 1 : params.num_splits) * params.num_head)};
    }

    struct WorkTileInfo {
        int block_idx = 0;
        int bidh = 0;
        int bidb = 0;
        int split_idx = 0;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb >= 0;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block_idx, bidh, bidb, !Split ? 0 : split_idx};
        }

    };

    CUTLASS_DEVICE
    SingleTileScheduler(SharedStorage* const smem_scheduler) { }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        /**
         * NOTE: Here, we shift the batch size to the x-dimension because the z-dimension must be less than 65536.
         *  Thus once the batch size is too large (>= 65536), the z-dimension may overflow, causing an implicit kernel-launch error.
         *  What's worse, this error may not be explicitly raised, resulting in the kernel being skipped.
         */
        // WorkTileInfo work_info {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), 0};
        WorkTileInfo work_info {int(blockIdx.y), int(blockIdx.z), int(blockIdx.x), 0};
        if constexpr (Split) {
            int split_idx;
            work_info.bidh = params.nsplits_divmod.divmod(split_idx, work_info.bidh);
            work_info.split_idx = split_idx;
        }
        bool is_valid_tile = true;
        if constexpr (Varlen) {
            int seqlen = params.seqlen;
            if (params.seqused) {
                seqlen = params.seqused[work_info.bidb];
            } else if (params.cu_seqlens) {
                seqlen = params.cu_seqlens[work_info.bidb + 1] - params.cu_seqlens[work_info.bidb];
            } else if (params.ranges) {
                seqlen = params.ranges[2 * work_info.bidb + 1] - params.ranges[2 * work_info.bidb];
            }
            if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
            is_valid_tile = work_info.block_idx * kBlock < seqlen;
        }
        if constexpr (Varlen && Split) {
            int num_splits_dynamic = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[work_info.bidb] : params.num_splits;
            // Use the top 16 bits to store num_splits
            work_info.split_idx |= (num_splits_dynamic << 16);
            is_valid_tile &= work_info.split_idx < num_splits_dynamic;
        }
        work_info.bidb = is_valid_tile ? work_info.bidb : -1;
        return work_info;
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {0, 0, -1, 0};
    }

};

///////////////////////////////////////////////////////////////////////////////

template<int kBlock, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp, bool Split=false, bool Deterministic=false, bool PackGQA=false, bool WarpSpecialized=true>
class VarlenDynamicPersistentTileScheduler {

    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

public:
    using SharedStorage = std::conditional_t<Deterministic, thrust::pair<int4, int2>, int4>;

protected:
    SharedStorage* const work_info_smem;

public:

    // Device side kernel params
    struct Params {
        int num_head, num_batch;
        int const qhead_per_khead;
        int const seqlen;
        cutlass::FastDivmod head_divmod;
        cutlass::FastDivmod nsplits_divmod;
        int* const tile_count_semaphore;
        int* const cu_seqlens;
        int* const ranges;
        int* const seqused;
        // int* const num_m_blocks_ptr;
        int const* const num_splits_dynamic_ptr;
        int* const merge_ranges;
        int* const range_map;
        int* determin_conflict_state;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        // If Split, for the purpose of scheduling, we pretend that instead there are
        // (args.num_splits * args.num_head) number of heads.
        assert(args.tile_count_semaphore != nullptr);
        assert(args.num_head < (1 << 16));  // We use the top 16 bits to store num_splits & split_idx
        assert(!Split || args.num_splits < (1 << 8)); // We use the top 8 bits to store num_splits
        int * const ranges = args.merge_ranges ? args.merge_ranges : args.ranges;
        return {args.num_head, args.num_batch,
                args.qhead_per_khead, args.seqlen,
                cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(!Split ? 1 : args.num_splits),
                args.tile_count_semaphore, args.cu_seqlens, ranges, args.seqused,
                // args.num_m_blocks_ptr, args.num_splits_dynamic_ptr};
                args.num_splits_dynamic_ptr, args.merge_ranges, args.range_map,
                args.determin_conflict_state};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;

        // give extra memory when deterministic == true
        using extra_vars_type = std::conditional_t<
            Deterministic,
            cute::tuple<int32_t, int32_t>,
            cute::tuple<>  // no memory use
        >;

        using BlockCoordType = std::conditional_t<
            Deterministic,
            cute::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>,
            cute::tuple<int32_t, int32_t, int32_t>
        >;

        extra_vars_type conflict_bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            // if (blockIdx.x >= 0 && (threadIdx.x == 128 || threadIdx.x == 0)) { printf("blockIdx.x = %d, threadIdx.x = %d, checking valid, bidb = %d, params.num_batch = %d\n", blockIdx.x, threadIdx.x, bidb, params.num_batch); }
            return bidb < params.num_batch;
        }

        CUTLASS_DEVICE
        BlockCoordType
        get_block_coord(Params const& params) const {
            if constexpr (!Split) {
                if constexpr (!Deterministic) {
                    return {block, bidh, bidb};
                } else {
                    return {block, bidh, bidb, cute::get<0>(conflict_bidb), cute::get<1>(conflict_bidb)};
                }
            } else {
                // the top 8 bits of bidh store num_splits and the next 8 bits store split_idx
                // reinterpret_cast to uint32_t to make sure we're not doing sign extension when we shift
                uint32_t bidh_packed = reinterpret_cast<uint32_t const&>(bidh);
                uint32_t bidh_actual_u = bidh_packed & 0x0000FFFF;
                int bidh_actual = reinterpret_cast<int&>(bidh_actual_u);
                // Use the top 16 bits of split_idx to store num_splits and the next 16 bits to store split_idx
                // uint32_t split_idx_u = ((bidh_packed & 0x00FF0000) >> 16) + ((bidh_packed & 0xFF000000) >> 8);
                // int split_idx = reinterpret_cast<int&>(split_idx_u);
                // int bidh_actual = params.nsplits_divmod.divmod(split_idx, bidh);
                // if (threadIdx.x == 128) {
                //     printf("blockIdx.x = %d, bidb = %d, bidh = %d, bidh_actual = %d, split_idx = %d\n", blockIdx.x, bidb, bidh, bidh_actual, split_idx);
                // }
                if constexpr (!Deterministic) {
                    return {block, bidh_actual, bidb};
                } else {
                    return {block, bidh_actual, bidb, cute::get<0>(conflict_bidb), cute::get<1>(conflict_bidb)};
                }
            }
        }
    };

    CUTLASS_DEVICE
    VarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
        int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
        auto get_num_m_blocks = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            int seqlen = params.seqlen * (!PackGQA ? 1 : params.qhead_per_khead);
            if (seqlen > kBlock) {
                if (params.seqused) {
                    seqlen = batch_idx < params.num_batch ? params.seqused[batch_idx] : 0;
                } else if (params.cu_seqlens) {
                    int cur_cu_seqlen = batch_idx <= params.num_batch ? params.cu_seqlens[batch_idx] : 0;
                    int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
                    seqlen = next_cu_seqlen - cur_cu_seqlen;
                } else if (params.ranges) {
                    seqlen = batch_idx < params.num_batch ? params.ranges[2 * batch_idx + 1] - params.ranges[2 * batch_idx] : 0;
                } else {
                    seqlen = params.seqlen;
                }
                if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
            }
            return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                ? cute::ceil_div(seqlen, kBlock) : 0;
                // ? params.num_m_blocks_ptr[batch_idx] : 0;
        };

        auto get_num_splits = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                ? (!Split ? 1 : (params.num_splits_dynamic_ptr
                                ? params.num_splits_dynamic_ptr[batch_idx]
                                : params.nsplits_divmod.divisor))
                : 0;
        };

        int num_m_blocks = get_num_m_blocks(current_work.bidb);  // Different for each lane
        int num_splits = get_num_splits(current_work.bidb);
        int num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
        // Cumulative number of blocks for the next 31 batches
        int num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
        // Total number of blocks for the next 31 batches
        int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
        // Only the lower 16 bits are the actual bidh
        int current_bidh = !Split ? current_work.bidh : (current_work.bidh & 0x0000FFFF);
        int group_end_tile = current_work.tile_idx - current_work.block - current_bidh * __shfl_sync(0xffffffff, num_split_m_blocks, 0 /*lane*/) + m_blocks_in_group * params.num_head;  // Same for all lanes
        if constexpr (Split) {
            int current_split_idx = (current_work.bidh & 0x00FF0000) >> 16;
            group_end_tile -= current_split_idx * __shfl_sync(0xffffffff, num_m_blocks, 0 /*lane*/);
        }
        int bidb = current_work.bidb;
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Before while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, cur tile_idx = %d, cur block = %d, cur bidh = %d, num_split_m_blocks = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, current_work.bidb, num_m_blocks, next_tile_idx, current_work.tile_idx, current_work.block, current_bidh, num_split_m_blocks, group_end_tile, m_blocks_in_group);
        // }
        // if (threadIdx.x == 0 && blockIdx.x == 0) { printf("tile_idx = %d, group_end_tile = %d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d\n", current_work.tile_idx, group_end_tile, num_m_blocks_cumulative, m_blocks_in_group); }
        while (group_end_tile <= next_tile_idx) {
            bidb += cutlass::NumThreadsPerWarp - 1;
            if (bidb >= params.num_batch) {
                // if (blockIdx.x <= 9 && threadIdx.x == 0) {
                //     printf("Returning early, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
                // }
                if constexpr (!Deterministic) {
                    return {next_tile_idx, 0, 0, params.num_batch};
                } else {
                    return {next_tile_idx, 0, 0, params.num_batch, cute::make_tuple(0,0)};
                }
            }
            num_m_blocks = get_num_m_blocks(bidb);
            num_splits = get_num_splits(bidb);
            num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
            num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
            m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
            group_end_tile += m_blocks_in_group * params.num_head;
            // if (blockIdx.x <= 9 && threadIdx.x == 0) {
            //     printf("Bottom of while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
            // }
        }
        int group_start_tile = group_end_tile - m_blocks_in_group * params.num_head;
        // The next problem to process is the first one that does not have ending tile position
        // that is greater than or equal to tile index.
        int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx));
        // if (threadIdx.x == 31 || threadIdx.x == 0) { printf("blockIdx.x = %d, tidx %d, group_start_tile = %d, num_m_blocks_cumulative = %d, num_head = %d, next_tile_idx = %d, ballot = %x, batch_idx_in_group = %d\n", blockIdx.x, threadIdx.x, group_start_tile, num_m_blocks_cumulative, params.num_head, next_tile_idx, tmp, batch_idx_in_group); }
        bidb += batch_idx_in_group;
        num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
        if constexpr (Split) { num_splits = __shfl_sync(0xffffffff, num_splits, batch_idx_in_group); }
        int mh_block = next_tile_idx - group_start_tile - (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) * params.num_head;
        int bidh = mh_block / num_m_blocks;
        int block = mh_block - bidh * num_m_blocks;
        if constexpr (Split) {
            int bidh_actual = bidh / num_splits;
            int split_idx = bidh - bidh_actual * num_splits;
            // TODO: idk why this gives wrong answer nondeterministically
            // int bidh_actual, split_idx;
            // split_idx = params.head_divmod.divmod(bidh_actual, bidh);
            // Use the top 8 bits to store num_splits and the next 8 bits to store split_idx
            // reinterpret_cast to uint32_t to make sure we're not doing sign extension when we shift
            uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
            // if (threadIdx.x == 0) {
            //     printf("blockIdx.x = %d, group_start_tiled = %d, bidb = %d, batch_idx_in_group = %d, mh_block = %d, num_m_blocks = %d, bidh = %d, bidh_actual = %d, split_idx = %d, num_splits = %d, bidh_packed = %d\n", blockIdx.x, group_start_tile, bidb, batch_idx_in_group, mh_block, num_m_blocks, bidh, bidh_actual, split_idx, num_splits, bidh_packed);
            // }
            bidh = reinterpret_cast<int&>(bidh_packed);
        }
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Before returning, blockIdx.x = %d, threadIdx.x = %d, group_start_tile = %d, batch_idx_in_group = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d, mh_block = %d, bidh = %d, block = %d\n", blockIdx.x, threadIdx.x, group_start_tile, batch_idx_in_group, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group, mh_block, bidh, block);
        // }
        if constexpr (!Deterministic) {
            return {next_tile_idx, block, bidh, bidb};
        } else {
            auto get_conflict_bidb = [&] (int bidb_last, int bidb_now, int block_now) {
                // l_output_offset = params.ranges[2 * bidb]
                // r_output_offset = params.ranges[2 * bidb + 1]
                // [l, r) is the range of bidb, r is not include
                // block_size = kBlock
                uint32_t smid = blockIdx.x;
                uint32_t sm_stride = gridDim.x;
                int * conflict_state = params.determin_conflict_state;
                // update conflict state
                while (bidb_last < bidb_now) {
                    int q_l_index = params.ranges[2 * bidb_last], q_r_index = params.ranges[2 * bidb_last + 1];
                    int l = q_l_index / kBlock + lane;
                    int block_num = cute::ceil_div(q_r_index - q_l_index, kBlock);
                    int r = (q_l_index + block_num * kBlock - 1) / kBlock;
                    while(l <= r) {
                        conflict_state[l * sm_stride + smid] = bidb_last + 1; // batch id + 1 (make it different to inital value 0)
                        l += cutlass::NumThreadsPerWarp;
                    }
                    bidb_last++;
                }
                int l = params.ranges[2 * bidb_now];
                int r = params.ranges[2 * bidb_now + 1];
                bool l_arrive_twice = (l % kBlock != 0) && (block_now == 0);
                bool r_arrive_twice = (l % kBlock != 0) && (block_now == (r - l + kBlock - 1) / kBlock - 1);
                int left_conflict_index = l / kBlock + block_now;
                int right_conflict_index = (l + kBlock - 1) / kBlock + block_now;
                __syncwarp();
                return cute::make_tuple((conflict_state[left_conflict_index * sm_stride + smid] << 1) | l_arrive_twice,
                                        (conflict_state[right_conflict_index * sm_stride + smid] << 1) | r_arrive_twice);
            };

            auto conflict_bidb = get_conflict_bidb(current_work.bidb, bidb, block);
            return {next_tile_idx, block, bidh, bidb, conflict_bidb};
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                if constexpr (!Deterministic) {
                    *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
                } else {
                    *work_info_smem = thrust::make_pair<int4, int2>(
                        make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb),
                        make_int2(cute::get<0>(work_info.conflict_bidb), cute::get<1>(work_info.conflict_bidb))
                    );
                }
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
            if constexpr (!Deterministic) {
                WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb};
                work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
                flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
                if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                    *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
                }
                flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
                return work_info;
            } else {
                WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb, cute::make_tuple(0,0)};
                work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
                flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
                if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                    *work_info_smem = thrust::make_pair(
                        make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb),
                        make_int2(cute::get<0>(work_info.conflict_bidb), cute::get<1>(work_info.conflict_bidb))
                    );
                }
                flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
                return work_info;
            }
        } else {
            if constexpr (!Deterministic) {
                flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
                int4 work_info = *work_info_smem;
                flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
                return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
            } else {
                flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
                int4 work_info = (*work_info_smem).first;
                int2 conflict_info = (*work_info_smem).second;
                flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
                return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w, cute::make_tuple(conflict_info.x, conflict_info.y)};
            }
        }
    }

};

} // flash
