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

/**********************************************************************************
 * Copyright (c) 2025 DeepSeek. All Rights Reserved.
 *
 * Licensed under the MIT License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *********************************************************************************/

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"

namespace magi_attn_comm::grpcoll {

namespace layout {

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(
    const int64_t* topk_idx,
    int* num_tokens_per_rank,
    int* num_tokens_per_rdma_rank,
    int* num_tokens_per_expert,
    bool* is_token_in_rank,
    int num_tokens,
    int num_topk,
    int num_ranks,
    int num_experts) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x);

  // Count expert statistics
  // by the first (num_experts // kNumExpertsPerSM) SMs
  __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
  int expert_begin_idx = sm_id * kNumExpertsPerSM, expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
  if (expert_begin_idx < expert_end_idx) {
    /** Per-thread count
     * 1. num_tokens_per_expert_per_thread[tid][local_eid]:
     *    the num of tokens sent to expert local_eid by thread tid in this SM,
     *    covering the token with idxs: [tid + i * kNumThreads], i=0,1,2,...
     */
#pragma unroll
    for (int i = 0; i < kNumExpertsPerSM; ++i)
      num_tokens_per_expert_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx = topk_idx + i * num_topk;
#pragma unroll
      for (int j = 0, expert_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
          ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
      }
    }
    __syncthreads();

    /** Sum up
     * 1. for num_tokens_per_expert:
     *  each thread tid in this SM will sum up the num of tokens sent to eid (expert_begin_idx + tid)
     *  by looping over the partial results in num_tokens_per_expert_per_thread[:][tid]
     */
    EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
    if (expert_begin_idx + thread_id < expert_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_expert_per_thread[i][thread_id];
      num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
    }
    return;
  }

  if (num_tokens_per_rdma_rank != nullptr)
    EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS);

  // Count rank statistics
  // by the last (num_ranks // kNumRanksPerSM) SMs
  constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
  __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
  __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
  auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
  int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM, rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
  int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS, rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;
  if (rank_begin_idx < rank_end_idx) {
    const auto num_expert_per_rank = num_experts / num_ranks;
    auto expert_begin = rank_begin_idx * num_expert_per_rank;
    auto expert_end = rank_end_idx * num_expert_per_rank;

    /** Per-thread count
     * 1. num_tokens_per_rank_per_thread[tid][local_rid]:
     *  the num of tokens sent to rank rid (local_rid + rank_begin_idx)
     *  counted by thread tid in this SM,
     *  covering the token with idxs: [tid + i * kNumThreads], i=0,1,2,...
     *
     * 2. num_tokens_per_rdma_rank_per_thread[tid][local_rid]:
     *  the num of tokens sent to RDMA rank rid (local_rid + rdma_rank_begin_idx)
     *  counted by thread tid in this SM,
     *  covering the token with idxs: [tid + i * kNumThreads], i=0,1,2,...
     *
     * 3. is_in_rank[local_rid]:
     *  whether the token is sent to rank rid (local_rid + rank_begin_idx)
     *  for each token with idx: tid + i * kNumThreads, i=0,1,2,...
     *
     * 4. is_in_rdma_rank[local_rid]:
     *  whether the token is sent to RDMA rank rid (local_rid + rdma_rank_begin_idx)
     *  for each token with idx: tid + i * kNumThreads, i=0,1,2,...
     */
#pragma unroll
    for (int i = 0; i < kNumRanksPerSM; ++i)
      num_tokens_per_rank_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = 0; i < kNumRDMARanksPerSM; ++i)
      num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx = topk_idx + i * num_topk;
      int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
#pragma unroll
      for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin <= expert_idx and expert_idx < expert_end) {
          // Count single rank
          rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
          is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
        }
      }

      auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
#pragma unroll
      for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
        shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
        num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
      }

#pragma unroll
      for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
        num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
    }
    __syncthreads();

    /** Sum up
     * 1. for num_tokens_per_rank:
     *  each thread tid in this SM will sum up the num of tokens sent to rid (rank_begin_idx + tid)
     *  by looping over the partial results in num_tokens_per_rank_per_thread[:][tid]
     *
     * 2. for num_tokens_per_rdma_rank:
     *  each thread tid in this SM will sum up the num of tokens sent to rid (rdma_rank_begin_idx + tid)
     *  by looping over the partial results in num_tokens_per_rdma_rank_per_thread[:][tid]
     *
     * 3. for is_token_in_rank:
     *  it has no need to sum up
     */
    EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
    if (rank_begin_idx + thread_id < rank_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_rank_per_thread[i][thread_id];
      num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
    }

    if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
      num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
    }
  }
}

void get_dispatch_layout(
    const int64_t* topk_idx,
    int* num_tokens_per_rank,
    int* num_tokens_per_rdma_rank,
    int* num_tokens_per_expert,
    bool* is_token_in_rank,
    int num_tokens,
    int num_topk,
    int num_ranks,
    int num_experts,
    cudaStream_t stream) {
  constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
  int num_sms_for_expert_stats = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
  int num_sms_for_rank_stats = (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
  int num_sms = num_sms_for_expert_stats + num_sms_for_rank_stats;
  EP_STATIC_ASSERT(kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  LAUNCH_KERNEL(
      &cfg,
      (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
      topk_idx,
      num_tokens_per_rank,
      num_tokens_per_rdma_rank,
      num_tokens_per_expert,
      is_token_in_rank,
      num_tokens,
      num_topk,
      num_ranks,
      num_experts);
}

template <int kNumThreads, int kMaxNumRanks>
__global__ void get_a2av_perm_idx(
    const int64_t* output_split_sizes,
    const int* src_idx,
    int64_t* unperm_from_a2av_idx,
    int64_t* perm_to_a2av_idx,
    int num_tokens, /* unused */
    int num_ranks,
    int num_splits) {
  auto thread_id = static_cast<int>(threadIdx.x);

  __shared__ int64_t rank_split_sizes[kNumThreads][kMaxNumRanks];
  __shared__ int64_t curr_offset_per_rank[kMaxNumRanks + 1];

// init rank_split_sizes
#pragma unroll
  for (int i = 0; i < num_ranks; ++i)
    rank_split_sizes[thread_id][i] = 0;

  // init curr_offset_per_rank
  if (thread_id < num_ranks + 1)
    curr_offset_per_rank[thread_id] = 0;

  __syncthreads();

// per-thread count partial rank_split_sizes
// rank_split_sizes[tid][rid]: the partial sum of split sizes recved from rank rid
// counted by thread tid
#pragma unroll
  for (int i = thread_id; i < num_splits; i += kNumThreads) {
    auto rank = src_idx[i];
    auto split_size = output_split_sizes[i];
    rank_split_sizes[thread_id][rank] += split_size;
  }
  __syncthreads();

  // sum up rank_split_sizes
  // rank_split_sizes[rid][rid]: the total sum of split sizes recved from rank rid
  if (thread_id < num_ranks) {
    int64_t sum = 0;

// sum up for partial results in each thread
#pragma unroll
    for (int i = 0; i < kNumThreads; ++i)
      sum += rank_split_sizes[i][thread_id];
    rank_split_sizes[thread_id][thread_id] = sum;
  }
  __syncthreads();

  // prefix sum for each rank by thread 0
  // rank_split_sizes[rid][rid]: the start offset of the a2av split buffer recved from rank rid
  // NOTE: since num_ranks are usually small, we don't need to use Blelloch scan algorithm
  if (thread_id == 0) {
    int64_t prefix_sum = 0;
#pragma unroll
    for (int rid = 0; rid < num_ranks; ++rid) {
      auto current = rank_split_sizes[rid][rid];
      rank_split_sizes[rid][rid] = prefix_sum;
      prefix_sum += current;
    }
  }
  __syncthreads();

// TODO: find a better way to parallelize
// especially when all the split sizes are small thus the number of splits is too large
// compute unperm_from_a2av_idx and perm_to_a2av_idx
// a2a_output[unperm_from_a2av_idx] => output
// output[perm_to_a2av_idx] => a2a_output
#pragma unroll
  for (int i = 0; i < num_splits; ++i) {
    auto rank = src_idx[i];
    auto split_size = output_split_sizes[i];
    auto a2av_offset_this_rank = rank_split_sizes[rank][rank];
    auto a2av_offset_this_split = a2av_offset_this_rank + curr_offset_per_rank[rank];
    auto start_token_idx = curr_offset_per_rank[num_ranks];

#pragma unroll
    for (int j = thread_id; j < split_size; j += kNumThreads) {
      auto token_idx = start_token_idx + j;
      auto a2av_token_idx = a2av_offset_this_split + j;

      unperm_from_a2av_idx[token_idx] = a2av_token_idx;
      perm_to_a2av_idx[a2av_token_idx] = token_idx;
    }

    if (thread_id == 0) {
      curr_offset_per_rank[num_ranks] += split_size; // start_token_idx
      curr_offset_per_rank[rank] += split_size;
    }
    __syncthreads();
  }
}

void get_a2av_perm_idx(
    const int64_t* output_split_sizes,
    const int* src_idx,
    int64_t* unperm_from_a2av_idx,
    int64_t* perm_to_a2av_idx,
    int num_tokens, /* unused */
    int num_ranks,
    int num_splits,
    cudaStream_t stream) {
  constexpr int num_sms = 1, kNumThreads = 256, kMaxNumRanks = 16;
  EP_STATIC_ASSERT(kNumThreads >= kMaxNumRanks, "kNumThreads should NOT less than kMaxNumRanks");
  EP_HOST_ASSERT(num_ranks <= kMaxNumRanks);

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  LAUNCH_KERNEL(
      &cfg, (get_a2av_perm_idx<kNumThreads, kMaxNumRanks>), output_split_sizes, src_idx, unperm_from_a2av_idx, perm_to_a2av_idx, num_tokens, num_ranks, num_splits);
}

} // namespace layout

} // namespace magi_attn_comm::grpcoll
