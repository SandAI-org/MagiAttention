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

import triton
import triton.language as tl


@triton.jit
def get_tid():
    """Get the thread_idx_in_block for x,y,z dim respectively,
    simulating (threadIdx.x, threadIdx.y, threadIdx.z)
    """

    return tl.inline_asm_elementwise(
        # %tid.x, %tid.y, %tid.z are the special registers
        # to store threadIdx.x, threadIdx.y, threadIdx.z
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",  # output to three regular 32bit registers
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,  # no side effects
        pack=1,
    )


@triton.jit
def get_ntid():
    """Get the num_threads_in_block for x,y,z dim respectively,
    simulating (blockDim.x, blockDim.y, blockDim.z)
    """

    return tl.inline_asm_elementwise(
        # %ntid.x, %ntid.y, %ntid.z are the special registers
        # to store blockDim.x, blockDim.y, blockDim.z
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",  # output to three regular 32bit registers
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,  # no side effects
        pack=1,
    )


@triton.jit
def get_flat_tid():
    """Get the flatten thread index in a 3d block,
    simulating (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x)
    """

    tid_x, tid_y, tid_z = get_tid()
    ntid_x, ntid_y, _ = get_ntid()

    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def get_bid():
    """Get the block index in a 3d grid,
    simulating (blockIdx.x, blockIdx.y, blockIdx.z)
    """

    return tl.program_id(0), tl.program_id(1), tl.program_id(2)


@triton.jit
def get_nbid():
    """Get the num_blocks_in_grid for x,y,z dim respectively,
    simulating (gridDim.x, gridDim.y, gridDim.z)
    """

    return tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)


@triton.jit
def get_flat_bid():
    """Get the flatten block index in a 3d grid,
    simulating (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x)
    """

    bid_x, bid_y, bid_z = get_bid()
    nbid_x, nbid_y, _ = get_nbid()

    return bid_z * nbid_y * nbid_x + bid_y * nbid_x + bid_x
