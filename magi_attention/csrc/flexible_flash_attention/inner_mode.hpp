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

#pragma once

namespace flash {

// Inner-loop KV load strategy for sparse attention scatter paths.
// Tma:     2D TMA descriptor — auto-selected when tiles are physically contiguous
// CpAsync: cp.async per-row scatter (8×16B per row for hd=128)
//
// Note: Tma1d (cp.async.bulk 1D) was investigated for inner loads but is NOT viable:
// SM90_BULK_COPY_G2S does a LINEAR memcpy, but WGMMA requires SMEM in atom-tiled
// layout (even INTER/no-swizzle uses 8×8 atoms). tile_to_shape makes rows non-contiguous
// across atoms — row 0 spans offsets 0-7, then jumps to 64-71 (next atom), not 8-15.
// This means cp.async.bulk per-row (256B) would corrupt data. Per-atom copies (16B each)
// have the same instruction count as cp.async, so no benefit.
// Tma1d (=1) for InnerLoadMode is reserved but currently falls through to CpAsync.
enum class InnerLoadMode : int { Tma = 0, Tma1d = 1, CpAsync = 2 };

// 2-way inner-loop scatter store strategy (BWD dX accumulation).
// Tma1d:   cp.reduce.async.bulk per-row (bulk reduce-add from row-contiguous smem)
// CpAsync: scalar atomicAdd fallback
enum class InnerStoreMode : int { Tma1d = 1, CpAsync = 2 };

} // namespace flash
