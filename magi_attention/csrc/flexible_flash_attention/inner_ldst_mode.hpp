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
enum class InnerLoadMode : int { Tma = 0, CpAsync = 2 };

// Inner-loop store strategy (BWD dX accumulation to global memory).
// Tma:         2D TMA reduce-add full-tile from GMMA K-major swizzled SMEM (default for dense paths).
//              TMA descriptor encodes the swizzle, so R2S must use the same layout (no STS-style swap).
// Tma1d:       cp.reduce.async.bulk per-row from linear SMEM (scatter: non-contiguous GMEM destinations)
// AtomicAdd:   scalar atomicAdd from SMEM (scatter or dense fallback)
// BypassSmem:  skip SMEM buffer entirely — consumer does register atomicAdd to gmem
//              (eliminates inner dKV SMEM buffer, barriers, and store warps; works for both dense/scatter)
enum class InnerStoreMode : int { Tma = 0, Tma1d = 1, AtomicAdd = 2, BypassSmem = 3 };

// Outer-loop store strategy (epilogue O/dQ/dKV write to global memory).
// Tma:   full-tile 2D TMA store (SM90_TMA_STORE or SM90_TMA_REDUCE_ADD).
//        SMEM layout must be GMMA K-major swizzled (TMA descriptor encodes the layout).
// Stg:   per-thread R2S to SMEM then STG.128 to GMEM with residual guard (flash::copy).
//        FWD uses bank-conflict-free SmemLayoutOSTS (FA3-style swizzle) when SwapAB=false.
// Tma1d: R2S to linear (unswizzled) SMEM, then per-row cp.async.bulk S2G.
//        Linear layout enables bulk DMA but may have R2S bank conflicts.
//        Experimental: benchmarks needed to compare vs Stg (STS swizzle + STG.128).
enum class OuterStoreMode : int { Tma = 0, Stg = 1, Tma1d = 2 };

} // namespace flash
