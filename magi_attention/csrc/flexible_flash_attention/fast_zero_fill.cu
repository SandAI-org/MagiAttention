// Copyright (c) 2025 SandAI. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fast_zero_fill_launch_template.h"

template void run_fast_zero_fill_<float, 64>(Flash_fwd_params& params, cudaStream_t stream);
template void run_fast_zero_fill_<float, 128>(Flash_fwd_params& params, cudaStream_t stream);
template void run_fast_zero_fill_<float, 192>(Flash_fwd_params& params, cudaStream_t stream);

template void run_fast_zero_fill_<cutlass::bfloat16_t, 64>(Flash_fwd_params& params, cudaStream_t stream);
template void run_fast_zero_fill_<cutlass::bfloat16_t, 128>(Flash_fwd_params& params, cudaStream_t stream);
template void run_fast_zero_fill_<cutlass::bfloat16_t, 192>(Flash_fwd_params& params, cudaStream_t stream);

template void run_fast_zero_fill_<cutlass::half_t, 64>(Flash_fwd_params& params, cudaStream_t stream);
template void run_fast_zero_fill_<cutlass::half_t, 128>(Flash_fwd_params& params, cudaStream_t stream);
template void run_fast_zero_fill_<cutlass::half_t, 192>(Flash_fwd_params& params, cudaStream_t stream);
