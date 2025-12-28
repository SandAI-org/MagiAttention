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

#include <thrust/pair.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/fast_math.h>

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

// Host side kernel arguments
struct TileSchedulerArguments {
  int const num_heads_q;
  int const num_heads_kv;
  int const num_batches;
  int const total_q;
  int* const tile_count_semaphore = nullptr;
  int2* const ranges = nullptr;
  int2* const merge_ranges = nullptr;
  int* const range_map = nullptr;
  int* determin_conflict_state = nullptr;
  int* const unique_count = nullptr;
};

///////////////////////////////////////////////////////////////////////////////

template <
    int kBlock,
    int NumMmaThreads = 2 * cutlass::NumThreadsPerWarpGroup,
    int NumProducerThreads = cutlass::NumThreadsPerWarp,
    bool WarpSpecialized = true,
    bool PackGQA = false,
    bool Deterministic = false>
class DynamicPersistentTileSchedulerFwd {
  static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
  static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

 public:
  using SharedStorage = std::conditional_t<Deterministic, thrust::pair<int4, int3>, int4>;
  using BlockCoordType = std::conditional_t<Deterministic, cute::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>, cute::tuple<int32_t, int32_t, int32_t>>;

 protected:
  SharedStorage* const work_info_smem;

  mutable int cached_total_tiles_ = -1;

 public:
  // Device side kernel params
  struct Params {
    int num_heads;
    int num_heads_kv;
    int qhead_per_kvhead;
    int qhead_per_khead_;
    int num_batches;
    int total_q;
    // int total_tiles_per_inter_group;
    int* const tile_count_semaphore;
    int2* const ranges;
    int2* const merge_ranges;
    int* const range_map;
    int* determin_conflict_state;
    int* const unique_count = nullptr;
  };

  static Params to_underlying_arguments(TileSchedulerArguments const& args) {
    int qhead_per_kvhead = !PackGQA ? 1 : (args.num_heads_q / args.num_heads_kv);
    int qhead_per_khead_ = args.num_heads_q / args.num_heads_kv;

    if constexpr (PackGQA) {
      qhead_per_khead_ = 1;
    }

    int num_heads = !PackGQA ? args.num_heads_q : args.num_heads_kv;
    assert(args.tile_count_semaphore != nullptr);
    assert(args.num_heads < (1 << 16));
    int2* const ranges = args.merge_ranges ? args.merge_ranges : args.ranges;

    return {
        num_heads,
        args.num_heads_kv,
        qhead_per_kvhead,
        qhead_per_khead_,
        args.num_batches,
        args.total_q,
        args.tile_count_semaphore,
        ranges,
        args.merge_ranges,
        args.range_map,
        args.determin_conflict_state,
        args.unique_count};
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
    return {uint32_t(num_sm)};
  }

  struct WorkTileInfo {
    int tile_idx, block, bidh, bidb;

    // give extra memory when deterministic == true
    using extra_vars_type = std::conditional_t<
        Deterministic,
        cute::tuple<int32_t, int32_t, int32_t>,
        cute::tuple<> // no memory use
        >;

    extra_vars_type conflict_batch_msg;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      // if (blockIdx.x >= 0 && (threadIdx.x == 128 || threadIdx.x == 0)) { printf("blockIdx.x = %d, threadIdx.x = %d, checking valid, bidb = %d, params.num_batches =
      // %d\n", blockIdx.x, threadIdx.x, bidb, params.num_batches); }
      int actual_num_batches = params.unique_count ? *params.unique_count : params.num_batches;
      return bidb < actual_num_batches;
    }

    CUTLASS_DEVICE
    BlockCoordType get_block_coord(Params const& params) const {
      if constexpr (!Deterministic) {
        return {block, bidh, bidb};
      } else {
        return {block, bidh, bidb, cute::get<0>(conflict_batch_msg), cute::get<1>(conflict_batch_msg), cute::get<2>(conflict_batch_msg)};
      }
    }
  };

  CUTLASS_DEVICE
  DynamicPersistentTileSchedulerFwd(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};

  CUTLASS_DEVICE
  int compute_exact_total_tiles(Params const& params) const {
    int lane = threadIdx.x % 32;
    int actual_num_batches = params.unique_count ? *params.unique_count : params.num_batches;
    int total_m_blocks = 0;
    /*if (lane == 0) {
      printf("compute_exact_total_tiles\n");
    }*/
    for (int bidb_start = 0; bidb_start < actual_num_batches; bidb_start += 32) {
      int batch_idx = bidb_start + lane;
      int m_blocks_this_batch = 0;

      if (batch_idx < actual_num_batches) {
        int2 range = params.ranges[batch_idx];
        int seqlen = range.y - range.x;
        if (seqlen > 0) {
          m_blocks_this_batch = cute::ceil_div(seqlen * params.qhead_per_kvhead, kBlock);
        }
      }

#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        m_blocks_this_batch += __shfl_xor_sync(0xffffffff, m_blocks_this_batch, offset);
      }

      total_m_blocks += m_blocks_this_batch;
    }

    return total_m_blocks * params.qhead_per_khead_;
  }

  CUTLASS_DEVICE
  WorkTileInfo tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
    int lane = threadIdx.x % cutlass::NumThreadsPerWarp;

    if (cached_total_tiles_ == -1)
      cached_total_tiles_ = compute_exact_total_tiles(params);

    // int total_tiles_per_group = params.total_tiles_per_inter_group;
    int total_tiles_per_group = cached_total_tiles_;

    int next_inter_group_idx = next_tile_idx / total_tiles_per_group;
    int next_tile_idx_in_group = next_tile_idx % total_tiles_per_group;

    int current_inter_group_idx = current_work.tile_idx / total_tiles_per_group;
    bool is_same_group = (next_inter_group_idx == current_inter_group_idx);
    int actual_num_batches = params.unique_count ? *params.unique_count : params.num_batches;

    if (total_tiles_per_group <= 0) {
      return {next_tile_idx, 0, 0, actual_num_batches};
    }

    if (next_inter_group_idx >= params.num_heads_kv) {
      return {next_tile_idx, 0, 0, actual_num_batches};
    }

    int bidb = is_same_group ? current_work.bidb : 0;
    int G = params.qhead_per_khead_;
    // Helper function to calculate how many blocks are needed to compute the current batch

    auto get_num_m_blocks = [&](int bidb_start) {
      int batch_idx = lane + bidb_start;
      int2 range = params.ranges[batch_idx];
      int seqlen = batch_idx < actual_num_batches ? range.y - range.x : 0;
      return batch_idx < actual_num_batches && lane < cutlass::NumThreadsPerWarp - 1 ? cute::ceil_div(seqlen * params.qhead_per_kvhead, kBlock) : 0;
    };

    // int num_m_blocks = get_num_m_blocks(current_work.bidb); // Different for each lane
    int num_m_blocks = get_num_m_blocks(bidb);
    // Cumulative number of blocks for the next 31 batches
    int num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks);
    // Total number of blocks for the next 31 batches
    int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);

    int group_end_tile;
    if (is_same_group) {
      int current_tile_in_group = current_work.tile_idx % total_tiles_per_group;

      group_end_tile = current_tile_in_group - current_work.block * G - (current_work.bidh % G) + m_blocks_in_group * G;
    } else {
      group_end_tile = m_blocks_in_group * G;
    }
    // Only the lower 16 bits are the actual bidh
    // int current_bidh = current_work.bidh;
    // int group_end_tile = current_work.tile_idx - current_work.block - current_bidh * __shfl_sync(0xffffffff, num_m_blocks, 0 /*lane*/) +
    //     m_blocks_in_group * params.num_heads; // Same for all lanes
    // int bidb = current_work.bidb;
    // if (blockIdx.x <= 9 && threadIdx.x == 0) {
    //     printf("Before while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, cur tile_idx = %d, cur block = %d, cur bidh = %d,
    //     num_m_blocks = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, current_work.bidb, num_m_blocks, next_tile_idx,
    //     current_work.tile_idx, current_work.block, current_bidh, num_m_blocks, group_end_tile, m_blocks_in_group);
    // }
    // if (threadIdx.x == 0 && blockIdx.x == 0) { printf("tile_idx = %d, group_end_tile = %d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d\n",
    // current_work.tile_idx, group_end_tile, num_m_blocks_cumulative, m_blocks_in_group); }
    while (group_end_tile <= next_tile_idx_in_group) {
      bidb += cutlass::NumThreadsPerWarp - 1;
      if (bidb >= actual_num_batches) {
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Returning early, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group =
        //     %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
        // }
        return {next_tile_idx, 0, 0, actual_num_batches};
      }
      num_m_blocks = get_num_m_blocks(bidb);
      num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks);
      m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
      // group_end_tile += m_blocks_in_group * params.num_heads;
      group_end_tile += m_blocks_in_group * G;
      // if (blockIdx.x <= 9 && threadIdx.x == 0) {
      //     printf("Bottom of while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group =
      //     %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
      // }
    }
    // int group_start_tile = group_end_tile - m_blocks_in_group * params.num_heads;
    int group_start_tile = group_end_tile - m_blocks_in_group * G;
    // The next problem to process is the first one that does not have ending tile position
    // that is greater than or equal to tile index.
    // int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * params.num_heads <= next_tile_idx));
    int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * G <= next_tile_idx_in_group));
    // if (threadIdx.x == 31 || threadIdx.x == 0) { printf("blockIdx.x = %d, tidx %d, group_start_tile = %d, num_m_blocks_cumulative = %d, num_heads = %d, next_tile_idx
    // = %d, ballot = %x, batch_idx_in_group = %d\n", blockIdx.x, threadIdx.x, group_start_tile, num_m_blocks_cumulative, params.num_heads, next_tile_idx, tmp,
    // batch_idx_in_group); }
    bidb += batch_idx_in_group;
    num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
    int prev_cumulative = (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1));

    int mh_block = next_tile_idx_in_group - group_start_tile - prev_cumulative * G;
    // int mh_block =
    //    next_tile_idx - group_start_tile - (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) *
    //    params.num_heads;
    // int bidh = mh_block / num_m_blocks;
    // int block = mh_block - bidh * num_m_blocks;
    int block = mh_block / G;
    int intragroup_idx = mh_block % G;
    int bidh = next_inter_group_idx * G + intragroup_idx;
    // if (blockIdx.x <= 9 && threadIdx.x == 0) {
    //     printf("Before returning, blockIdx.x = %d, threadIdx.x = %d, group_start_tile = %d, batch_idx_in_group = %d, bidb = %d, num_m_blocks = %d, next_tile_idx =
    //     %d, group_end_tile = %d, m_blocks_in_group = %d, mh_block = %d, bidh = %d, block = %d\n", blockIdx.x, threadIdx.x, group_start_tile, batch_idx_in_group,
    //     bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group, mh_block, bidh, block);
    // }
    if (lane == 0) {
      // printf("next_tile_idx: %d  block: %d bidb: %d bidh: %d total_tiles_per_group: %d G: %d intragroup_idx: %d next_inter_group_idx: %d mh_block: %d num_m_blocks:
      // %d num_m_blocks_cumulative: %d m_blocks_in_group: %d \n", next_tile_idx, block, bidb, bidh, total_tiles_per_group, G, intragroup_idx, next_inter_group_idx,
      // mh_block, num_m_blocks, num_m_blocks, num_m_blocks_cumulative, m_blocks_in_group);
    }
    if constexpr (!Deterministic) {
      return {next_tile_idx, block, bidh, bidb};
    } else {
      auto get_conflict_batch_msg = [&](int bidb_last, int bidb_now, int block_now) {
        int const qhead_per_khead = params.qhead_per_kvhead;

        uint32_t smid = blockIdx.x;
        uint32_t sm_stride = gridDim.x;
        uint32_t block_stride = (params.total_q + kBlock - 1) / kBlock + 1;
        uint32_t head_offset = next_inter_group_idx * block_stride;
        int* conflict_state = params.determin_conflict_state;

        // update missed batch's conflict state, loop for bidb_last ~ bidb_now
        while (bidb_last < bidb_now) {
          int2 bidb_last_lr = params.ranges[bidb_last];

          int bidb_last_l_physical = bidb_last_lr.x * qhead_per_khead;
          int bidb_last_r_physical = bidb_last_lr.y * qhead_per_khead;

          int l = bidb_last_l_physical / kBlock + lane;
          int block_num = cute::ceil_div(bidb_last_r_physical - bidb_last_l_physical, kBlock);
          int r = (bidb_last_l_physical + block_num * kBlock - 1) / kBlock;

          while (l <= r) {
            int global_block_idx = head_offset + l;
            conflict_state[global_block_idx * sm_stride + smid] = bidb_last + 1;
            l += cutlass::NumThreadsPerWarp;
          }
          bidb_last++;
        }

        // calc arrive message: l_arrive_twice & r_arrive_twice
        // each range_lock needs to arrive twice to make sure conflict batch has been completed
        // because range_lock block and batch's block may start from a different offset
        // eg. kBlock=10, range_lock is 0~10 10~20 20~30, batch's block is 5~15 15~20
        //     so the 0～10 wait 5~15, 10~20 wait 5~15 and 15~20, 20~30 wait 15~20
        //     there is two kind range_lock A, B
        //     range_lock A (as 10~20) may need to wait two block of same batch arrive
        //     range_lock B (as 0~10, 20~30) only need to wait one batch block arrive
        //     as for range_lock B, the batch block should arrive twice
        //     so that the arrive time can equal range_lock A
        //     batch block 5~15 should arrive left range_lock 0~10 twice, but right range_lock 10~20 once (l_arrive_twice == true)
        //     batch block 15~20 should arrive left range_lock 10~20 once, but right range_lock 20~30 twice (r_arrive_twice == true)
        int2 lr = params.ranges[bidb_now];
        int l_physical = lr.x * qhead_per_khead;
        int r_physical = lr.y * qhead_per_khead;

        bool l_arrive_twice = (l_physical % kBlock != 0) && (block_now == 0);

        int total_blocks_physical = cute::ceil_div(r_physical - l_physical, kBlock);
        bool r_arrive_twice = (l_physical % kBlock != 0) && (block_now == total_blocks_physical - 1);

        // int left_conflict_index = l_physical / kBlock + block_now;
        // int right_conflict_index = (l_physical + kBlock - 1) / kBlock + block_now;

        int left_conflict_index = head_offset + (l_physical / kBlock + block_now);
        int right_conflict_index = head_offset + ((l_physical + kBlock - 1) / kBlock + block_now);

        __syncwarp();
        return cute::make_tuple(
            (conflict_state[left_conflict_index * sm_stride + smid] << 1) | l_arrive_twice,
            (conflict_state[right_conflict_index * sm_stride + smid] << 1) | r_arrive_twice,
            bidb_now);
      };

      // int bidb_last = current_work.bidb * next_inter_group_idx + next_inter_group_idx;
      // int bidb_now = bidb * next_inter_group_idx + next_inter_group_idx;

      // auto conflict_batch_msg = get_conflict_batch_msg(current_work.bidb, bidb, block);
      auto conflict_batch_msg = get_conflict_batch_msg(is_same_group ? current_work.bidb : 0, bidb, block);

      return {next_tile_idx, block, bidh, bidb, conflict_batch_msg};
    }
  }

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
    if constexpr (IsProducerWarp) {
      WorkTileInfo work_info = tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
      if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
        if constexpr (!Deterministic) {
          *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
        } else {
          *work_info_smem = thrust::make_pair<int4, int3>(
              make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb),
              make_int3(cute::get<0>(work_info.conflict_batch_msg), cute::get<1>(work_info.conflict_batch_msg), cute::get<2>(work_info.conflict_batch_msg)));
        }
      }
      flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/); // TileCountSmemFull
      return work_info;
    } else {
      return get_next_work<false>(params, {0, 0, 0, 0});
    }
  }

  CUTLASS_DEVICE
  void init_consumer() const {
    // Don't arrive at the TileCountSmemEmpty barrier here, because get_initial_work will do that
  }

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
    if (threadIdx.x % NumProducerThreads == 0) {
      current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
    }
  }

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    if constexpr (IsProducerWarp) {
      // thread 0 has the next tile_idx, just need to broadcast to the rest of warp 0
      int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
      if constexpr (!Deterministic) {
        WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb};
        work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
        flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/); // TileCountSmemEmpty
        if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
          *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
        }
        flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/); // TileCountSmemFull
        return work_info;
      } else {
        WorkTileInfo work_info = {
            __shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb, cute::make_tuple(0, 0, 0)};
        work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
        flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/); // TileCountSmemEmpty
        if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
          *work_info_smem = thrust::make_pair(
              make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb),
              make_int3(cute::get<0>(work_info.conflict_batch_msg), cute::get<1>(work_info.conflict_batch_msg), cute::get<2>(work_info.conflict_batch_msg)));
        }
        flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/); // TileCountSmemFull
        return work_info;
      }
    } else {
      if constexpr (!Deterministic) {
        flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/); // TileCountSmemFull
        int4 work_info = *work_info_smem;
        flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/); // TileCountSmemEmpty
        return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
      } else {
        flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/); // TileCountSmemFull
        int4 work_info = (*work_info_smem).first;
        int3 conflict_info = (*work_info_smem).second;
        flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/); // TileCountSmemEmpty
        return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w, cute::make_tuple(conflict_info.x, conflict_info.y, conflict_info.z)};
      }
    }
  }
};

} // namespace flash
