#include "repeat_reduce_launch_template.h"

template void repeat_reduce_cute_kernel<float, 64>(torch::cuda::nccl::GroupReducePostProcessArgs& args);
template void repeat_reduce_cute_kernel<float, 128>(torch::cuda::nccl::GroupReducePostProcessArgs& args);
template void repeat_reduce_cute_kernel<float, 192>(torch::cuda::nccl::GroupReducePostProcessArgs& args);

template void repeat_reduce_cute_kernel<cutlass::bfloat16_t, 64>(torch::cuda::nccl::GroupReducePostProcessArgs& args);
template void repeat_reduce_cute_kernel<cutlass::bfloat16_t, 128>(torch::cuda::nccl::GroupReducePostProcessArgs& args);
template void repeat_reduce_cute_kernel<cutlass::bfloat16_t, 192>(torch::cuda::nccl::GroupReducePostProcessArgs& args);

template void repeat_reduce_cute_kernel<cutlass::half_t, 64>(torch::cuda::nccl::GroupReducePostProcessArgs& args);
template void repeat_reduce_cute_kernel<cutlass::half_t, 128>(torch::cuda::nccl::GroupReducePostProcessArgs& args);
template void repeat_reduce_cute_kernel<cutlass::half_t, 192>(torch::cuda::nccl::GroupReducePostProcessArgs& args);