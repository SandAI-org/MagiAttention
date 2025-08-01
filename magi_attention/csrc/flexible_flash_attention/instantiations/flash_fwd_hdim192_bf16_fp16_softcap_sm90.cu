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

#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM192
template void run_mha_fwd_<90, cutlass::bfloat16_t, cutlass::half_t, 192, true, false>(Flash_fwd_params& params, cudaStream_t stream);
#endif
