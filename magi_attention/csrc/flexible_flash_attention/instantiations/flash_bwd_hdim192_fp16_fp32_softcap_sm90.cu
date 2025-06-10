// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM192
template void run_mha_bwd_<90, cutlass::half_t, float, 192, true>(Flash_bwd_params &params, cudaStream_t stream);
#endif
