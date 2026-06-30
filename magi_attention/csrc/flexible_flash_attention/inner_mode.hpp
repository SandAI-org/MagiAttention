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

// 2-way inner-loop KV load strategy for sparse attention scatter paths.
// Tma:     2D TMA descriptor — auto-selected when tiles are physically contiguous
// CpAsync: cp.async per-row scatter
enum class InnerLoadMode : int { Tma = 0, CpAsync = 2 };

// 2-way inner-loop scatter store strategy (BWD dX accumulation).
// Tma1d:   cp.reduce.async.bulk per-row (bulk reduce-add from row-contiguous smem)
// CpAsync: scalar atomicAdd fallback
enum class InnerStoreMode : int { Tma1d = 1, CpAsync = 2 };

} // namespace flash
