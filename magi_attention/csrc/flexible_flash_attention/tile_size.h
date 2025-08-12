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

#include <tuple>

/**
 * @brief Determine tile size and configuration for forward propagation on SM90 architecture
 *
 * This function determines the optimal tile configuration based on head dimension size,
 * returning a tuple with four elements:
 * - kBlockM: M dimension size of the tile
 * - kBlockN: N dimension size of the tile
 * - MmaPV_is_RS: Whether P in MMA PV is in registers
 * - IntraWGOverlap: Whether to enable intra-workgroup overlapped computation
 *
 * @param headdim Attention head dimension size
 * @param element_size Element size, defaults to 2 bytes (FP16/BF16)
 * @param softcap Whether to enable softcap, defaults to false
 * @return std::tuple<int, int, bool, bool> Returns a tuple of tile configuration, {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}
 */
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(int headdim, int element_size = 2, bool softcap = false) {
  // Currently only support FP16/BF16
  assert(element_size == 2);

  if (headdim <= 64) {
    // return {same_hdim ? 192 : 64, same_hdim ? 128 : 64, same_hdim, same_hdim};
    // With this workaround in Cutlass 3.8, tile size 192 x 128 got slower for non-causal, idk why
    // https://github.com/NVIDIA/cutlass/blob/833f6990e031b48b4cd2fcf55e0849c51ef6bac2/include/cute/container/tuple.hpp#L131
    return {192, 128, true, true};
    // Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
    // return {192, is_causal || is_local ? 192 : 176, true, false};
  } else if (headdim <= 128) {
    return {128, 128, true, true};
    // {128, 192, false, false} and {192, 128, false, true} are quite good too
    // 128 x 192 hits the limit of smem if MmaPV_is_RS, 128 x 144 hits the limit if !MmaPV_is_RS
  } else if (headdim <= 192) {
    return {128, 96, true, true}; // 128 x 112 hits the limit of smem
  } else {
    return {128, 64, true, true};
  }
}

/**
 * @brief Determine tile size for backward propagation on SM90 architecture
 *
 * This function determines the optimal tile configuration based on head dimension size,
 * returning a tuple with two elements:
 * - kBlockM: M dimension size of the tile
 * - kBlockN: N dimension size of the tile
 *
 * @param headdim Attention head dimension size
 * @param element_size Element size, defaults to 2 bytes (FP16/BF16)
 * @return std::tuple<int, int> Returns a tuple of tile configuration, {kBlockM, kBlockN}
 */
constexpr std::tuple<int, int> tile_size_bwd_sm90(int headdim, int element_size = 2, bool softcap = false) {
  // Currently only support FP16/BF16
  assert(element_size == 2);

  if (headdim <= 64) {
    return {128, 128};
  } else if (headdim <= 128) {
    // if (softcap) {
    return {64, 128};
    // }
    // else {
    //     return {80, 128};
    // }
  } else if (headdim <= 192) {
    return {64, 64};
  } else {
    return {64, 64};
  }
}
