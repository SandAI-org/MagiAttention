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

#include <cstring>
#include <vector>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#ifndef DISABLE_NVSHMEM
#include "ibgda_device.cuh"
#include "nvshmem.h"
#endif

namespace magi_attn_comm::grpcoll {

namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {
  barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream) {
  SETUP_LAUNCH_CONFIG(1, 32, stream);
  RANKS_SWITCH(num_ranks, kNumRanks, [&] {
      LAUNCH_KERNEL(&cfg, barrier<kNumRanks>, barrier_signal_ptrs, rank);
  });
}

} // namespace intranode

namespace internode {

#ifndef DISABLE_NVSHMEM
nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
nvshmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
  nvshmemx_uniqueid_t unique_id;
  nvshmemx_get_uniqueid(&unique_id);
  std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
  std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
  return result;
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
  nvshmemx_uniqueid_t root_unique_id;
  nvshmemx_init_attr_t attr;
  std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
  nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

  // Create sub-RDMA teams
  // NOTE: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
  if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
    GRPCOLL_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
    GRPCOLL_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
    GRPCOLL_HOST_ASSERT(
        nvshmem_team_split_strided(
            NVSHMEM_TEAM_WORLD, rank % NUM_MAX_NVL_PEERS, NUM_MAX_NVL_PEERS, num_ranks / NUM_MAX_NVL_PEERS, &cpu_rdma_team_config, 0, &cpu_rdma_team) == 0);
    GRPCOLL_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
  }

  nvshmem_barrier_all();
  return nvshmem_my_pe();
}

void* alloc(size_t size, size_t alignment) {
  return nvshmem_align(alignment, size);
}

void free(void* ptr) {
  nvshmem_free(ptr);
}

void barrier() {
  nvshmem_barrier_all();
}

void finalize() {
  if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
    nvshmem_team_destroy(cpu_rdma_team);
    cpu_rdma_team = NVSHMEM_TEAM_INVALID;
  }
  nvshmem_finalize();
}
#endif

} // namespace internode

} // namespace magi_attn_comm::grpcoll
