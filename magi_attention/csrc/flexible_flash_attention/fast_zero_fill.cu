#include "fast_zero_fill_launch_template.h"

template void run_fast_zero_fill_<float, 64>(Flash_fwd_params &params, cudaStream_t stream);
template void run_fast_zero_fill_<float, 128>(Flash_fwd_params &params, cudaStream_t stream);
template void run_fast_zero_fill_<float, 192>(Flash_fwd_params &params, cudaStream_t stream);

template void run_fast_zero_fill_<cutlass::bfloat16_t, 64>(Flash_fwd_params &params, cudaStream_t stream);
template void run_fast_zero_fill_<cutlass::bfloat16_t, 128>(Flash_fwd_params &params, cudaStream_t stream);
template void run_fast_zero_fill_<cutlass::bfloat16_t, 192>(Flash_fwd_params &params, cudaStream_t stream);

template void run_fast_zero_fill_<cutlass::half_t, 64>(Flash_fwd_params &params, cudaStream_t stream);
template void run_fast_zero_fill_<cutlass::half_t, 128>(Flash_fwd_params &params, cudaStream_t stream);
template void run_fast_zero_fill_<cutlass::half_t, 192>(Flash_fwd_params &params, cudaStream_t stream);