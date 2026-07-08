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

#include <tuple>

/**
 * @brief Determine tile size and configuration for forward propagation on SM90 architecture
 *
 * This function determines the optimal tile configuration based on head dimension size,
 * returning a tuple with four elements:
 * - kBlockM: M dimension size of the tile
 * - kBlockN: N dimension size of the tile
 * - MmaPV_is_RS: Whether P in MMA PV is in registers
 *
 * @param headdim Attention head dimension size
 * @param element_size Element size, defaults to 2 bytes (FP16/BF16)
 * @param softcap Whether to enable softcap, defaults to false
 * @return std::tuple<int, int, bool> Returns a tuple of tile configuration, {kBlockM, kBlockN, MmaPV_is_RS}
 */
constexpr std::tuple<int, int, bool> tile_size_fwd_sm90(int headdim, int element_size = 2, bool softcap = false) {
  // Currently only support FP16/BF16
  assert(element_size == 2);

  if (headdim <= 64) {
    // return {same_hdim ? 192 : 64, same_hdim ? 128 : 64, same_hdim, same_hdim};
    // With this workaround in Cutlass 3.8, tile size 192 x 128 got slower for non-causal, idk why
    // https://github.com/NVIDIA/cutlass/blob/833f6990e031b48b4cd2fcf55e0849c51ef6bac2/include/cute/container/tuple.hpp#L131
    return {192, 128, true};
    // Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
    // return {192, is_causal || is_local ? 192 : 176, true};
  } else if (headdim <= 128) {
    // Synced with Python tile_size_fwd_sm90 in _flex_flash_attn_jit.py.
    // NOTE: this C++ fallback is not used in practice — Python always resolves
    // tile sizes before Jinja rendering — but must stay in sync to avoid confusion.
    return {128, 128, true};
    // {128, 192, false} and {192, 128, false} are quite good too
    // 128 x 192 hits the limit of smem if MmaPV_is_RS, 128 x 144 hits the limit if !MmaPV_is_RS
  } else if (headdim <= 192) {
    return {128, 96, true}; // 128 x 112 hits the limit of smem
  } else {
    return {128, 64, true};
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
 *
 * NOTE: when BwdInnerLoopK is true, the shared memory usage pattern changes,
 * and theoretically, the shared memory storage of mainloop will discard `dq_acc` and add `dk_acc`, `dv_acc`,
 * accordingly, the shared memory storage of epilogue will discard `dk` and `dv`, and add `dq`,
 * so the total shared memory usage may increase `(2 * kBlockN - kBlockM) * kHeadDim * ElementSize` bytes,
 * which might be unacceptable for some cases like `kBlockM=64, kBlockN=128, kHeadDim=128, ElementSize=2`.
 */
template <bool BwdInnerLoopK, bool IndexSparseLoopQ = false>
constexpr std::tuple<int, int> tile_size_bwd_sm90(int headdim, int element_size = 2, bool softcap = false) {
  // Currently only support FP16/BF16
  assert(element_size == 2);

  // IndexSparse LoopQ (inv-indices): outer=K token (1 valid row), inner=Q tiles.
  // K can't be packed → minimize kBlockN. Use {64, 64} which satisfies all WGMMA
  // constraints with the default hd=128 swapAB flags (SdP_swapAB=true → M=kBlockN=64≥64).
  if constexpr (IndexSparseLoopQ) {
    static_assert(!BwdInnerLoopK, "IndexSparseLoopQ requires BwdInnerLoopK=false (LoopQ)");
    if (headdim <= 128)
      return {64, 64};
    else
      return {64, 64};
  }

  if (headdim <= 64) {
    if constexpr (BwdInnerLoopK)
      return {64, 128}; // {128, 128, 64} => {64, 128, 64}
    else
      return {128, 128};
  } else if (headdim <= 128) {
    if constexpr (BwdInnerLoopK)
      return {128, 64}; // dK_acc/dV_acc union + kStages_dS=1 → 196 KB fits 228 KB SMEM limit
    else
      return {64, 128};
  } else if (headdim <= 192) {
    return {64, 64};
  } else {
    return {64, 64};
  }
}
