# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from magi_attention.kernel.triton.comm.device_a2av import on_device_a2av_kernel


def on_device_a2av_triton_impl(
    output: torch.Tensor,
    output_splits: torch.Tensor,
    input: torch.Tensor,
    input_splits: torch.Tensor,
    group: dist.ProcessGroup,
    BLOCKS_PER_REMOTE_RANK=8,
    UNROLL_FACTOR: int = 8,
    BLOCK_SIZE: int = 16384,
):
    assert (
        input.is_contiguous()
        and output.is_contiguous()
        and input_splits.is_contiguous()
        and output_splits.is_contiguous()
    )
    assert input.dim() == output.dim() == 2 and output.shape[1] == input.shape[1]
    assert input_splits.shape[0] == output_splits.shape[0] == group.size()
    assert input.dtype == output.dtype == torch.bfloat16
    assert input_splits.dtype == output_splits.dtype == torch.int64

    stride0 = output.shape[1]

    # redendezvous: exchange handle with each other device in the given group
    # and return the `_SymmetricMemory` handle
    # so that they can directly access the remote data in any other device
    # whose data ptrs are all stored in the handle's `buffer_ptrs_dev`, which is a vector of data ptrs
    # where buffer_ptrs_dev[i] is the data ptr of the i-th device
    # and the signal ptrs are stored in the handle's `signal_pad_ptrs_dev`, which is a vector of signal ptrs
    # where signal_pad_ptrs_dev[i] is the signal ptr of the i-th device
    input_hdl = symm_mem.rendezvous(input, group=group)
    input_splits_hdl = symm_mem.rendezvous(input_splits, group=group)

    # determine the number of blocks
    num_blocks = input_hdl.world_size * BLOCKS_PER_REMOTE_RANK

    on_device_a2av_kernel[(num_blocks, 1, 1)](
        output,
        output_splits,
        input_hdl.buffer_ptrs_dev,
        input_splits_hdl.buffer_ptrs_dev,
        input_hdl.signal_pad_ptrs_dev,
        stride0=stride0,
        rank=input_hdl.rank,
        world_size=input_hdl.world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        UNROLL_FACTOR=UNROLL_FACTOR,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return output
